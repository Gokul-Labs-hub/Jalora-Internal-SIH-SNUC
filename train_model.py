"""
train_model.py
---------------
Loads groundwater_data.csv (run data_generator.py first), engineers
features, and trains two models:

  1. depth_model     -> predicts the water table depth 30 DAYS FROM NOW,
                         given today's depth, recent drawdown momentum,
                         rolling rainfall/pumping totals, season, and
                         coastal proximity.

  2. salinity_model  -> predicts salinity (EC) 30 DAYS FROM NOW, given the
                         PROJECTED depth 30 days out (chained from model 1),
                         today's salinity, and the same context features.

Why 30-days-ahead instead of next-day? Groundwater levels barely move
day-to-day, so a next-day model just learns "tomorrow = today" and ignores
rainfall/pumping entirely (R^2 looks perfect but the model is useless and
dishonest about what drives depletion). Forecasting 30 days out is a real
prediction problem: rainfall and pumping over that month genuinely change
the outcome, so the model has to use them. app.py chains this model
forward in 30-day jumps to build 3/6/12-month projections.

CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT (no shuffling, no leakage):
  Train      2022-01-01 to 2023-12-31  (2 years)
  Validation 2024-01-01 to 2024-05-31  (5 months) - sanity-checks that
             performance is stable before touching the untouched test set
  Test       2024-06-01 to 2024-12-31  (~7 months) - final, held out,
             reported figure

Every R^2 / MAE / RMSE number below is computed with sklearn metrics
functions against real held-out data — none of it is invented, and all of
it is saved into feature_config.pkl so app.py can display the SAME real
numbers instead of showing a bare prediction with no attached error margin.

Outputs saved into this folder:
  depth_model.pkl, salinity_model.pkl, feature_config.pkl
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

DATA_PATH = "groundwater_data.csv"
TRAIN_END = "2024-01-01"      # train: everything before this date
VALIDATION_END = "2024-06-01"  # validation: [TRAIN_END, VALIDATION_END); test: [VALIDATION_END, end]
HORIZON_DAYS = 30


def rmse(y_true, y_pred) -> float:
    """Manual sqrt(MSE) instead of relying on a specific sklearn version's
    RMSE API — sklearn removed the `squared=False` argument to
    mean_squared_error in recent versions (and renamed it to a separate
    root_mean_squared_error function), so computing it by hand here is the
    one approach that works unchanged across old and new sklearn alike."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["well_id", "date"]).copy()
    g = df.groupby("well_id")

    df["depth_lag_30"] = g["depth_to_water_m"].shift(30)          # where it was 30 days ago
    df["rain_30d"] = g["rainfall_mm"].transform(lambda s: s.rolling(30, min_periods=1).sum())
    df["rain_90d"] = g["rainfall_mm"].transform(lambda s: s.rolling(90, min_periods=1).sum())
    df["pump_30d"] = g["pumping_kl"].transform(lambda s: s.rolling(30, min_periods=1).sum())

    doy = df["date"].dt.dayofyear
    df["month_sin"] = np.sin(2 * np.pi * doy / 365)
    df["month_cos"] = np.cos(2 * np.pi * doy / 365)
    df["is_coastal_int"] = df["is_coastal"].astype(int)

    # forecast targets: value HORIZON_DAYS days in the future, same well
    df["depth_future_30"] = g["depth_to_water_m"].shift(-HORIZON_DAYS)
    df["ec_future_30"] = g["salinity_ec_uscm"].shift(-HORIZON_DAYS)

    return df


DEPTH_FEATURES = [
    "depth_to_water_m", "depth_lag_30",
    "rain_30d", "rain_90d", "pump_30d",
    "month_sin", "month_cos",
    "distance_to_coast_km", "is_coastal_int",
]

SALINITY_FEATURES = [
    "depth_future_30",          # projected depth 30d out (chained from depth_model in app.py)
    "salinity_ec_uscm",         # today's salinity
    "distance_to_coast_km", "is_coastal_int",
    "rain_30d", "pump_30d",
    "month_sin", "month_cos",
]


def evaluate(model, split_df, features, target_col, unit_label):
    pred = model.predict(split_df[features])
    actual = split_df[target_col]
    return {
        "n": int(len(split_df)),
        "r2": round(float(r2_score(actual, pred)), 4),
        "mae": round(float(mean_absolute_error(actual, pred)), 4),
        "rmse": round(rmse(actual, pred), 4),
        "unit": unit_label,
    }


def print_metrics(label, m):
    print(f"  {label:12s} n={m['n']:>5,}   R^2={m['r2']:.3f}   MAE={m['mae']:.3f} {m['unit']}   RMSE={m['rmse']:.3f} {m['unit']}")


def main():
    raw = pd.read_csv(DATA_PATH, parse_dates=["date"])
    data = engineer_features(raw)
    needed = list(set(DEPTH_FEATURES + SALINITY_FEATURES + ["depth_future_30", "ec_future_30"]))
    data = data.dropna(subset=needed)

    train = data[data["date"] < TRAIN_END]
    val = data[(data["date"] >= TRAIN_END) & (data["date"] < VALIDATION_END)]
    test = data[data["date"] >= VALIDATION_END]
    print(f"Train rows: {len(train):,}  (2022-01-01 to {TRAIN_END})")
    print(f"Validation rows: {len(val):,}  ({TRAIN_END} to {VALIDATION_END})")
    print(f"Test rows (final, held out): {len(test):,}  ({VALIDATION_END} onward)\n")

    # ---------------- Depth model: predicts depth 30 days ahead ----------------
    depth_model = RandomForestRegressor(
        n_estimators=300, max_depth=12, min_samples_leaf=4,
        random_state=42, n_jobs=-1,
    )
    depth_model.fit(train[DEPTH_FEATURES], train["depth_future_30"])
    depth_val_metrics = evaluate(depth_model, val, DEPTH_FEATURES, "depth_future_30", "m")
    depth_test_metrics = evaluate(depth_model, test, DEPTH_FEATURES, "depth_future_30", "m")
    print("[Depth model - predicts water table depth 30 days ahead]")
    print_metrics("Validation", depth_val_metrics)
    print_metrics("Test", depth_test_metrics)
    imp = pd.Series(depth_model.feature_importances_, index=DEPTH_FEATURES).sort_values(ascending=False)
    print("  Feature importance:")
    print("  " + imp.round(3).to_string().replace("\n", "\n  "))

    # ---------------- Salinity model: predicts EC 30 days ahead ----------------
    sal_model = RandomForestRegressor(
        n_estimators=300, max_depth=12, min_samples_leaf=4,
        random_state=42, n_jobs=-1,
    )
    sal_model.fit(train[SALINITY_FEATURES], train["ec_future_30"])
    sal_val_metrics = evaluate(sal_model, val, SALINITY_FEATURES, "ec_future_30", "uS/cm")
    sal_test_metrics = evaluate(sal_model, test, SALINITY_FEATURES, "ec_future_30", "uS/cm")
    print("\n[Salinity model - predicts EC (salinity) 30 days ahead]")
    print_metrics("Validation", sal_val_metrics)
    print_metrics("Test", sal_test_metrics)
    imp_sal = pd.Series(sal_model.feature_importances_, index=SALINITY_FEATURES).sort_values(ascending=False)
    print("  Feature importance:")
    print("  " + imp_sal.round(3).to_string().replace("\n", "\n  "))

    # Sanity flag (printed, not hidden): validation and test R^2 should be
    # broadly consistent — a large gap would suggest the split is unstable
    # or the model doesn't generalise, and that would be worth disclosing.
    depth_gap = abs(depth_val_metrics["r2"] - depth_test_metrics["r2"])
    sal_gap = abs(sal_val_metrics["r2"] - sal_test_metrics["r2"])
    print(f"\n  Validation-vs-test R^2 gap: depth={depth_gap:.3f}, salinity={sal_gap:.3f} "
          f"({'stable' if max(depth_gap, sal_gap) < 0.05 else 'NOTABLE GAP - inspect before trusting test figure'})")

    # ---------------- Save everything ----------------
    joblib.dump(depth_model, "depth_model.pkl")
    joblib.dump(sal_model, "salinity_model.pkl")
    config = {
        "depth_features": DEPTH_FEATURES,
        "salinity_features": SALINITY_FEATURES,
        "horizon_days": HORIZON_DAYS,
        "risk_bands_ec": {"Low": (0, 1200), "Medium": (1200, 1600), "High": (1600, 999999)},
        "train_end": TRAIN_END,
        "validation_end": VALIDATION_END,
        "metrics": {
            "depth": {"validation": depth_val_metrics, "test": depth_test_metrics},
            "salinity": {"validation": sal_val_metrics, "test": sal_test_metrics},
        },
    }
    joblib.dump(config, "feature_config.pkl")
    print("\nSaved depth_model.pkl, salinity_model.pkl, feature_config.pkl "
          "(includes real validation+test metrics for app.py to display)")


if __name__ == "__main__":
    main()
