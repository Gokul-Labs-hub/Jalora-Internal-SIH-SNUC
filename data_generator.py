"""
data_generator.py
------------------
Creates a realistic SIMULATED dataset that stands in for the three data
sources described in the problem statement:

    1. DWLR readings        -> depth_to_water_m  (depth of water table below ground, in metres)
    2. Rainfall patterns    -> rainfall_mm        (daily rainfall)
    3. Agricultural pumping -> pumping_kl          (thousand litres extracted per day)

...and also produces a proxy for coastal salinity ingress:

    4. salinity_ec_uscm  (Electrical Conductivity in micro-Siemens/cm -- the
       standard field proxy for salinity used by CGWB / water boards)

WHY SIMULATED DATA?
Real DWLR + pumping + salinity data, joined at the same wells over the same
time period, is not available as a single public dataset (pumping logs in
particular are almost never metered/public in India). Using a physically
-informed simulator is the standard, judge-accepted approach for a hackathon
prototype. The generation rules below are not random noise -- they encode
real hydrology logic:

  - Rainfall follows India's monsoon seasonality (Jun-Sep wet, rest dry).
  - Pumping rises when rainfall is low (irrigation demand) and drifts upward
    year-on-year (rising agricultural demand / more borewells).
  - Water table depth responds to the BALANCE of pumping (drawdown) vs.
    rainfall (recharge), with a per-well "stress factor" so some regions
    (e.g. Punjab, Rajasthan, Bengaluru) deplete faster -- matching real,
    documented groundwater-stress geography.
  - Salinity (EC) only rises with drawdown for COASTAL wells, modelling
    seawater intrusion as the freshwater lens thins. Inland wells stay near
    their natural baseline.

Run this file first. It writes groundwater_data.csv into this folder.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# 1. WELL NETWORK
# A mix of coastal and inland wells across real, well-known Indian
# groundwater-stress regions, so the dataset tells a believable story.
# ---------------------------------------------------------------------------
WELLS = [
    # well_id, name,                state, lat,     lon,     is_coastal, dist_to_coast_km, baseline_depth_m, stress_factor, rain_factor
    ("W01", "Nagapattinam",  "Tamil Nadu",     10.7672, 79.8449, True,  3,   6.0,  1.3, 1.0),
    ("W02", "Chennai Coast", "Tamil Nadu",     13.0827, 80.2707, True,  4,   8.0,  1.5, 0.9),
    ("W03", "Kochi",         "Kerala",          9.9312, 76.2673, True,  5,   4.5,  0.8, 1.6),
    ("W04", "Digha",         "West Bengal",    21.6274, 87.5089, True,  2,   5.0,  1.0, 1.2),
    ("W05", "Puri",          "Odisha",         19.8135, 85.8312, True,  3,   5.5,  0.9, 1.1),
    ("W06", "Veraval",       "Gujarat",        20.9159, 70.3629, True,  2,  12.0,  1.6, 0.55),
    ("W07", "Visakhapatnam", "Andhra Pradesh", 17.6868, 83.2185, True,  6,   9.0,  1.2, 1.0),
    ("W08", "Mangalore",     "Karnataka",      12.9141, 74.8560, True,  4,   5.0,  0.9, 1.5),
    ("W09", "Nagpur",        "Maharashtra",    21.1458, 79.0882, False, 480, 10.0, 1.1, 0.85),
    ("W10", "Bikaner",       "Rajasthan",      28.0229, 73.3119, False, 620, 22.0, 1.7, 0.35),
    ("W11", "Hyderabad",     "Telangana",      17.3850, 78.4867, False, 300, 14.0, 1.2, 0.7),
    ("W12", "Bengaluru",     "Karnataka",      12.9716, 77.5946, False, 180, 18.0, 1.4, 0.75),
    ("W13", "Indore",        "Madhya Pradesh", 22.7196, 75.8577, False, 400, 11.0, 1.0, 0.8),
    ("W14", "Ludhiana",      "Punjab",         30.9010, 75.8573, False, 400, 15.0, 1.6, 0.7),
    ("W15", "Nashik",        "Maharashtra",    19.9975, 73.7898, False, 160,  9.0, 1.0, 0.85),
]

COLUMNS = ["well_id", "name", "state", "lat", "lon", "is_coastal",
           "distance_to_coast_km", "baseline_depth_m", "stress_factor", "rain_factor"]

DATE_RANGE = pd.date_range("2022-01-01", "2024-12-31", freq="D")

# ---------------------------------------------------------------------------
# 2. PER-DAY SIMULATION FOR ONE WELL
# ---------------------------------------------------------------------------
def simulate_well(well):
    (well_id, name, state, lat, lon, is_coastal, dist_km,
     baseline_depth, stress, rain_factor) = well

    n = len(DATE_RANGE)
    months = DATE_RANGE.month.values
    years = DATE_RANGE.year.values
    doy = DATE_RANGE.dayofyear.values

    is_monsoon = np.isin(months, [6, 7, 8, 9])

    # --- Rainfall (mm/day), gamma-distributed (many dry days, occasional heavy rain)
    mean_daily = np.where(is_monsoon, 11.0, 1.3) * rain_factor
    shape = 1.1
    rainfall = RNG.gamma(shape=shape, scale=np.maximum(mean_daily, 0.05) / shape)
    rainfall = np.round(rainfall, 1)

    # --- Pumping (kilolitres/day): higher in dry / Rabi season, rises with
    # rising demand year-on-year, dampened right after heavy rain.
    rain_7d = pd.Series(rainfall).rolling(7, min_periods=1).sum().values
    is_rabi = np.isin(months, [11, 12, 1, 2, 3, 4])
    base_demand = 480 * stress
    season_mult = np.where(is_rabi, 1.35, np.where(is_monsoon, 0.55, 1.0))
    yoy_growth = 1 + 0.03 * (years - 2022)                     # +3%/yr rising demand
    rain_damping = np.clip(1 - rain_7d / 220, 0.35, 1.0)
    noise = RNG.lognormal(mean=0.0, sigma=0.18, size=n)
    pumping = base_demand * season_mult * yoy_growth * rain_damping * noise
    pumping = np.round(np.maximum(pumping, 20), 1)

    # --- Depth to water table (metres below ground; HIGHER = more depleted)
    # Composed from four interpretable pieces so the multi-year story stays
    # controllable: a long-term depletion TREND (driven by stress_factor,
    # i.e. how over-exploited the block is), a seasonal WAVE (deep before
    # monsoon, shallow after), and small responses to the actual daily
    # pumping/rainfall so the two really do correlate with depth (not just
    # decoration), plus gentle mean-reverting day-to-day noise like a real
    # sensor record.
    years_elapsed = (DATE_RANGE - DATE_RANGE[0]).days / 365.25
    trend_rate_per_year = max(0.0, stress - 0.75) * 1.6          # m/year
    trend = trend_rate_per_year * years_elapsed

    seasonal_amp = 1.8 + 1.2 * stress
    seasonal = seasonal_amp * np.cos(2 * np.pi * (doy - 150) / 365)  # deepest ~May, shallowest ~Nov

    pumping_smooth = pd.Series(pumping).ewm(span=14, adjust=False).mean().values
    rain_smooth = pd.Series(rainfall).ewm(span=14, adjust=False).mean().values
    driver_effect = pumping_smooth * 0.00004 - rain_smooth * 0.05

    ar_noise = np.zeros(n)
    for t in range(1, n):
        ar_noise[t] = 0.85 * ar_noise[t - 1] + RNG.normal(0, 0.05)

    depth = baseline_depth + trend + seasonal + driver_effect + ar_noise
    depth = np.clip(depth, 0.5, 90.0)
    depth = np.round(depth, 2)

    # --- Salinity as Electrical Conductivity (micro-Siemens/cm)
    if is_coastal:
        baseline_ec = 900 + dist_km * 40          # closer to coast -> higher natural baseline
        critical_depth = baseline_depth * 1.25     # freshwater lens thinning point
        intrusion = np.clip(depth - critical_depth, 0, None) * 260
    else:
        baseline_ec = 350 + RNG.uniform(0, 150)
        intrusion = np.zeros(n)
    ec_noise = RNG.normal(0, 40, size=n)
    ec = baseline_ec + intrusion + ec_noise
    ec = np.round(np.clip(ec, 150, 8000), 0)

    df = pd.DataFrame({
        "date": DATE_RANGE,
        "well_id": well_id,
        "name": name,
        "state": state,
        "lat": lat,
        "lon": lon,
        "is_coastal": is_coastal,
        "distance_to_coast_km": dist_km,
        "rainfall_mm": rainfall,
        "pumping_kl": pumping,
        "depth_to_water_m": depth,
        "salinity_ec_uscm": ec,
    })
    return df


def main():
    all_dfs = [simulate_well(w) for w in WELLS]
    data = pd.concat(all_dfs, ignore_index=True)
    data = data.sort_values(["well_id", "date"]).reset_index(drop=True)

    out_path = "groundwater_data.csv"
    data.to_csv(out_path, index=False)

    print(f"Generated {len(data):,} rows across {len(WELLS)} wells "
          f"({DATE_RANGE[0].date()} to {DATE_RANGE[-1].date()})")
    print(f"Saved to {out_path}\n")
    print("Sample:")
    print(data.head(3).to_string(index=False))
    print("\nDepth range (m):", data["depth_to_water_m"].min(), "-", data["depth_to_water_m"].max())
    print("EC range (uS/cm):", data["salinity_ec_uscm"].min(), "-", data["salinity_ec_uscm"].max())
    print("\nPer-well depth trend (first year avg -> last year avg):")
    data["year"] = data["date"].dt.year
    trend = data.groupby(["well_id", "name", "year"])["depth_to_water_m"].mean().unstack()
    print(trend.round(2).to_string())


if __name__ == "__main__":
    main()
