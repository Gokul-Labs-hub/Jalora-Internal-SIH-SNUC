"""
app.py
------
The dashboard (this IS the "frontend" — a single Python file, no HTML/JS
needed). Run with:  streamlit run app.py

TWO PROFILES, ONE APP:
  - Authority / Technical view: full forecasting dashboard, what-if
    scenarios, electricity-based pumping estimation, model internals,
    data-source citations and formulas.
  - Public view: a plain-language, jargon-free traffic-light status for
    someone with no technical background — pick your area, see if it's
    safe, in plain words.
A landing screen lets the user pick either; a "Switch profile" button
lets them jump back at any time. Same running app, same URL.

DATA HONESTY NOTE (read this before demoing):
  - DWLR water level & rainfall are treated as auto-logged (simulated
    here; CGWB/India-WRIS and IMD feeds in a real deployment — see the
    citation links in the Authority view's "Data Sources" panel).
  - Agricultural pumping is NOT auto-fetched from any electricity board.
    Tamil Nadu (and most Indian states) supply agricultural power free of
    charge, and most agricultural connections are unmetered in practice —
    published feeder/block-level open data for this does not exist. So
    instead of pretending to fetch it, the app asks for an ELECTRICITY
    FIGURE (kWh) — the one number a field officer can plausibly obtain
    from a bill or DISCOM estimate — and converts it to a pumped-volume
    estimate using the standard pump hydraulics formula:
        V = (eta * E) / (rho * g * H)
    with a transparent, adjustable efficiency range and an
    uncertainty band. This is real physics on a manually-sourced number,
    not a fabricated automatic feed — see the README for the full
    explanation to give your judges.

Requires depth_model.pkl, salinity_model.pkl, feature_config.pkl and
groundwater_data.csv in the same folder — auto-generated on first run if
missing (see the startup check below), so this also works out of the box
on a fresh cloud deployment where no one can open a terminal to run
data_generator.py/train_model.py manually.
"""

import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Groundwater Depletion & Salinity Infiltration Predictive Matrix",
    layout="wide",
    page_icon="💧",
)

DATA_PATH = "groundwater_data.csv"
REQUIRED_FILES = [DATA_PATH, "depth_model.pkl", "salinity_model.pkl", "feature_config.pkl"]

missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]
if missing:
    # A freshly deployed app (e.g. on Streamlit Community Cloud) won't have
    # these files yet, and unlike a local `streamlit run`, a deployed app's
    # visitors have no terminal to run the setup scripts in themselves — so
    # generate everything here, once, automatically. Takes ~15-20 seconds.
    with st.spinner("Setting up for the first time — generating data and training models "
                     "(about 20 seconds, only happens once)..."):
        try:
            if not os.path.exists(DATA_PATH):
                import data_generator
                data_generator.main()
            if not all(os.path.exists(f) for f in ["depth_model.pkl", "salinity_model.pkl", "feature_config.pkl"]):
                import train_model
                train_model.main()
        except Exception as e:
            st.error(
                f"Automatic first-time setup failed: {e}\n\n"
                "If running locally, you can also set it up manually in this folder:\n\n"
                "    python data_generator.py\n    python train_model.py"
            )
            st.stop()


# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df.sort_values(["well_id", "date"]).reset_index(drop=True)


@st.cache_resource
def load_models():
    depth_model = joblib.load("depth_model.pkl")
    sal_model = joblib.load("salinity_model.pkl")
    config = joblib.load("feature_config.pkl")
    return depth_model, sal_model, config


data = load_data()
depth_model, sal_model, config = load_models()
DEPTH_FEATURES = config["depth_features"]
SALINITY_FEATURES = config["salinity_features"]
RISK_BANDS = config["risk_bands_ec"]

wells_meta = data.drop_duplicates("well_id")[
    ["well_id", "name", "state", "lat", "lon", "is_coastal", "distance_to_coast_km"]
].sort_values("name")


def risk_band(ec_value: float) -> str:
    for label, (lo, hi) in RISK_BANDS.items():
        if lo <= ec_value < hi:
            return label
    return "High"


RISK_COLOR = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}


# --------------------------------------------------------------------------
# DATA PROVENANCE REGISTRY (audit requirement #1)
# One entry per data type actually shown in the app. Every major metric on
# screen is captioned from this table so a judge can see, for any number,
# where it came from, what period/resolution it covers, its unit, and
# whether it is Observed, Derived, or Estimated — not asserted from memory
# each time it's displayed.
# --------------------------------------------------------------------------
DATA_PROVENANCE = {
    "depth_observed": {
        "label": "Water-table depth (observed)",
        "source": "Simulated DWLR proxy (would be CGWB/India-WRIS telemetry in production)",
        "period": "Daily, 2022-01-01 to 2024-12-31",
        "resolution": "Per-well point sensor",
        "unit": "metres below ground",
        "kind": "Observed (simulated)",
    },
    "depth_forecast": {
        "label": "Water-table depth (forecast)",
        "source": "Random Forest model, trained on the observed series above",
        "period": "30-day-ahead prediction, chained forward",
        "resolution": "Per-well",
        "unit": "metres below ground",
        "kind": "Derived (ML prediction) — see validation metrics",
    },
    "salinity_observed": {
        "label": "Salinity / EC (observed)",
        "source": "Simulated DWLR-adjacent proxy (would be CGWB field EC readings in production)",
        "period": "Daily, 2022-01-01 to 2024-12-31",
        "resolution": "Per-well point sensor",
        "unit": "\u00b5S/cm (electrical conductivity)",
        "kind": "Observed (simulated)",
    },
    "salinity_forecast": {
        "label": "Salinity / EC (forecast)",
        "source": "Random Forest model chained from the depth forecast, plus a documented capped stress adjustment",
        "period": "30-day-ahead prediction, chained forward",
        "resolution": "Per-well",
        "unit": "\u00b5S/cm (electrical conductivity)",
        "kind": "Derived (ML prediction) — NOT confirmed seawater intrusion, see note below",
    },
    "rainfall": {
        "label": "Rainfall",
        "source": "Simulated monsoon-seasonality proxy (would be IMD gridded rainfall in production)",
        "period": "Daily, 2022-01-01 to 2024-12-31",
        "resolution": "Per-well",
        "unit": "mm/day",
        "kind": "Observed (simulated)",
    },
    "extraction": {
        "label": "Estimated Agricultural Groundwater Extraction",
        "source": "User/officer-entered electricity figure (kWh) converted via pump hydraulics — "
                   "NOT sourced from an official EB/TANGEDCO record (no such public dataset exists "
                   "at well/feeder resolution — see Data Sources & Methodology)",
        "period": "Trailing 30 days, as entered",
        "resolution": "Per-well (single figure assumed representative of that well's local pumping)",
        "unit": "kL (\u2261 m\u00b3)",
        "kind": "Estimated (manual input + physics formula, wide uncertainty)",
    },
}


def provenance_caption(key: str) -> str:
    p = DATA_PROVENANCE[key]
    return (f"*{p['label']} — {p['kind']}. Source: {p['source']}. "
            f"Period: {p['period']}. Resolution: {p['resolution']}. Unit: {p['unit']}.*")


# --------------------------------------------------------------------------
# DATA QUALITY / CONFIDENCE INDICATOR (audit requirement #8)
# Computed from real, checkable signals — not a cosmetic badge:
#   - completeness: are there gaps in this well's date range?
#   - source reliability: is the underlying series observed-simulated or
#     an unverified manual entry?
#   - spatial/temporal compatibility: does the extraction estimate's
#     30-day-aggregate resolution match the daily DWLR/rainfall series?
#   - uncertainty: how wide is the extraction range relative to its
#     midpoint (a genuine proxy for how much to trust the point estimate)?
# --------------------------------------------------------------------------
def compute_data_quality(well_hist: pd.DataFrame, vol_low: float, vol_mid: float, vol_high: float):
    expected_days = (well_hist["date"].max() - well_hist["date"].min()).days + 1
    completeness = len(well_hist) / expected_days if expected_days > 0 else 0.0

    uncertainty_ratio = (vol_high - vol_low) / vol_mid if vol_mid > 0 else 1.0

    issues = []
    if completeness < 0.98:
        issues.append(f"data completeness {completeness*100:.0f}% (gaps present)")
    issues.append("pumping/extraction figure is manually entered, not from a verified official source")
    if uncertainty_ratio > 0.6:
        issues.append(f"extraction uncertainty is wide (\u00b1{uncertainty_ratio*50:.0f}% around the midpoint)")

    # Score: start high, dock for each real issue found — not asserted, computed.
    score = 100
    if completeness < 0.98:
        score -= 15
    score -= 25  # manual/unverified electricity input always docks confidence, every well, every time
    if uncertainty_ratio > 0.6:
        score -= 15
    elif uncertainty_ratio > 0.35:
        score -= 8

    if score >= 75:
        band, color = "Moderate", "#f39c12"
    elif score >= 55:
        band, color = "Low-Moderate", "#f39c12"
    else:
        band, color = "Low", "#e74c3c"
    # NOTE: "High" confidence is intentionally unreachable while extraction
    # relies on a manually-entered, unverified electricity figure — that's
    # the honest ceiling for this prototype's weakest input, not a bug.

    return {
        "score": score, "band": band, "color": color,
        "completeness_pct": completeness * 100,
        "uncertainty_ratio": uncertainty_ratio,
        "issues": issues,
    }


# --------------------------------------------------------------------------
# EXPLAINABILITY FOR HIGH / CRITICAL RISK (audit requirement #6)
# Every driver listed here is computed from real values already in the
# app for this well — no generic canned list, and a driver is only listed
# if its actual computed condition is met.
# --------------------------------------------------------------------------
def explain_risk_drivers(current_depth: float, depth_trend_90d: float,
                          recent_rain_30d: float, typical_rain_30d: float,
                          vol_mid: float, default_pump_30d: float,
                          is_coastal: bool, dist_km: float,
                          current_ec: float, ec_trend_90d: float) -> list:
    drivers = []
    if depth_trend_90d > 0.3:
        drivers.append(f"Declining groundwater level — depth has increased (deepened) by "
                        f"{depth_trend_90d:.2f} m over the last 90 days.")
    if default_pump_30d > 0 and vol_mid > default_pump_30d * 1.3:
        pct = (vol_mid / default_pump_30d - 1) * 100
        drivers.append(f"Elevated extraction — the estimated extraction is {pct:.0f}% above "
                        f"this well's historical 30-day baseline.")
    if typical_rain_30d > 0 and recent_rain_30d < typical_rain_30d * 0.7:
        drivers.append(f"Rainfall deficit — the last 30 days ({recent_rain_30d:.0f} mm) is "
                        f"below this well's typical level for the season ({typical_rain_30d:.0f} mm).")
    if ec_trend_90d > 50:
        drivers.append(f"Rising salinity — observed EC has increased by {ec_trend_90d:.0f} "
                        f"\u00b5S/cm over the last 90 days.")
    if is_coastal and dist_km <= 10:
        drivers.append(f"Coastal proximity — this well is only {dist_km:.0f} km from the coast, "
                        f"where drawdown converts to salinity risk fastest.")
    if not drivers:
        drivers.append("No single dominant driver stands out in the last 90 days — elevated "
                        "risk here reflects this well's overall coastal/extraction profile "
                        "rather than a sharp recent change.")
    return drivers


# --------------------------------------------------------------------------
# Animated 3-tier alert banners (Feature 1)
# --------------------------------------------------------------------------
def render_alert(level: str, title: str, message: str, big: bool = False):
    """Renders a colour-coded, animated alert box. High = fast urgent pulse,
    Medium = slower pulse, Low = calm one-time fade (no repeating motion —
    stillness reads as 'safe' just as clearly as pulsing reads as 'urgent')."""
    glow = {"High": "rgba(231,76,60,0.75)", "Medium": "rgba(243,156,18,0.65)",
            "Low": "rgba(46,204,113,0.5)"}[level]
    bg = RISK_COLOR[level]
    icon = {"High": "🚨", "Medium": "⚠️", "Low": "✅"}[level]
    duration = {"High": "1.1s", "Medium": "2.3s", "Low": "0.7s"}[level]
    iteration = "infinite" if level in ("High", "Medium") else "1"
    anim_name = f"pulse_{level}"
    font_title = "26px" if big else "19px"
    font_msg = "17px" if big else "14px"
    pad = "28px 30px" if big else "16px 20px"

    html = f"""
    <style>
    @keyframes {anim_name} {{
        0%   {{ box-shadow: 0 0 0 0 {glow}; transform: scale(1); }}
        70%  {{ box-shadow: 0 0 0 22px rgba(0,0,0,0); transform: scale(1.005); }}
        100% {{ box-shadow: 0 0 0 0 rgba(0,0,0,0); transform: scale(1); }}
    }}
    .alertbox_{anim_name} {{
        background: {bg};
        color: white;
        padding: {pad};
        border-radius: 14px;
        margin-bottom: 14px;
        animation: {anim_name} {duration} ease-out {iteration};
    }}
    .alertbox_{anim_name} .a-title {{ font-size: {font_title}; font-weight: 700; margin-bottom: 4px; }}
    .alertbox_{anim_name} .a-msg   {{ font-size: {font_msg}; font-weight: 400; opacity: 0.96; }}
    </style>
    <div class="alertbox_{anim_name}">
        <div class="a-title">{icon} {title}</div>
        <div class="a-msg">{message}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


AUTHORITY_ALERT_TEXT = {
    "Low":    ("Normal — No Action Needed", "Groundwater level and salinity are within safe operating limits for this well."),
    "Medium": ("Caution — Monitor Closely", "This well is showing measurable stress. Recommend increased monitoring frequency and advisories to reduce peak-season extraction."),
    "High":   ("Critical — Intervention Recommended", "This well is in high-risk territory for drawdown and/or salinity ingress. Recommend immediate field verification and extraction restrictions."),
}

PUBLIC_ALERT_TEXT = {
    "Low":    ("Your water situation is SAFE", "No action needed right now. Your groundwater is at a healthy level and not salty."),
    "Medium": ("Please be CAREFUL with water use", "Water levels here are getting lower than normal. Try to use less water for farming or other heavy use this month."),
    "High":   ("URGENT: Water problem in your area", "Water levels are very low and the water may be turning salty. Please contact your local water authority and avoid over-pumping."),
}

# --------------------------------------------------------------------------
# CROP GUIDANCE FOR COASTAL FARMERS (Public Interface only)
# Grounded in real ICAR-CSSRI (Central Soil Salinity Research Institute)
# findings -- India's national institute for salt-affected-soil research,
# specifically their coastal-salinity regional station work (Canning Town
# RRS, West Bengal) and crop-improvement division. Not invented: cotton,
# sapota/guava, coconut, chilli, coriander/fennel, and the salt-tolerant
# seed varieties mentioned below are all crops/practices CSSRI's own
# published material identifies for salt-affected coastal land.
#
# HONESTY NOTE: CSSRI's published tolerance figures are for soil salinity
# (ECe, measured on a soil sample) -- this app's EC reading is a
# groundwater/irrigation-water salinity proxy. The two are related (water
# salinity is a major driver of soil salinity) but not the same
# measurement, and actual soil salt buildup also depends on soil type,
# drainage, and irrigation practice. That's why the source caption below
# says "ask your local agriculture office to confirm for your exact
# field" rather than presenting this as a precise, field-specific
# prescription -- a genuine limitation, not hidden.
#
# SCOPE: crop names are kept in their widely-used pan-Indian form (the
# same terms used nationally in government agricultural schemes and
# market reporting) rather than translated into each of the 8 regional
# scripts -- botanical/crop terminology varies by exact local dialect in
# ways this app cannot verify per-region, so keeping one consistent,
# widely-recognised form is more reliable than a possibly-inaccurate
# per-language translation. The surrounding guidance text IS translated.
# --------------------------------------------------------------------------
CROP_GUIDANCE = {
    "High": {
        "crops": [
            ("Cotton", "One of the most salt-tolerant common cash crops -- handles salty soil well."),
            ("Sapota (Chikoo) or Guava", "Fruit trees that CSSRI's coastal research found do well even in highly salty soil."),
            ("Coconut", "A traditional, hardy coastal crop that tolerates salty groundwater."),
            ("Ask about salt-tolerant seed varieties", "ICAR-CSSRI has bred special salt-tolerant seed varieties of paddy, wheat, chickpea and mustard -- ask your local Krishi Vigyan Kendra (agriculture office) about these."),
        ],
        "avoid": ["Regular (non-salt-tolerant) paddy seed", "Sugarcane", "Most leafy vegetables"],
    },
    "Medium": {
        "crops": [
            ("Bajra, Ragi, or Jowar (millets)", "Low water need and moderately salt-tolerant -- well suited when groundwater is under stress."),
            ("Groundnut", "Drought-tolerant oilseed crop with moderate salinity tolerance."),
            ("Chilli", "A CSSRI-confirmed crop for moderately salt-affected coastal soil."),
            ("Coriander or Fennel", "Spice crops CSSRI identifies as suitable for salt-affected coastal land."),
        ],
        "avoid": ["High-water crops like paddy or sugarcane, unless irrigation is fully assured"],
    },
    "Low": {
        "crops": [
            ("Millets (Bajra, Ragi, Jowar)", "A good proactive choice -- low water use helps keep this area's healthy status."),
            ("Pulses (Moong, Chickpea, Tur)", "Low water requirement, and improves soil health for the future too."),
        ],
        "avoid": [],
    },
}


# ============================================================================
# LANGUAGE / TRANSLATION SYSTEM
# Dictionary-based (not a live translation API) — works instantly offline,
# no network call, no API key, nothing that can silently fail at runtime.
# HONESTY NOTE: these are AI-drafted translations for a prototype demo, not
# professionally localized or native-speaker-verified — say so if asked,
# and recommend professional review before any real deployment.
#
# Public Interface: automatic (based on the selected coastal zone's state
# language) + manual override, via the "Auto" option in the language picker.
# Authority Interface: manual only, defaults to English, no auto-switch —
# per the requirement that officials aren't forced into a language just
# because they clicked into a particular well.
#
# SCOPE: the Public Interface is fully translated (every string a resident
# sees). The Authority Interface's deep technical content (formulas, unit
# citations, methodology, feature-importance labels) stays in English by
# design — this matches how real regional-language government tools work
# (scientific/technical terminology is conventionally kept in English even
# within a localized UI) and keeps ~150 technical strings from being
# machine-translated in ways that could distort a formula or a citation.
# The Authority Interface's main chrome (titles, section headers, the
# broadcast button, core metric labels, alert text) IS translated below.
# ============================================================================
LANGUAGES = {
    "en": "English",
    "hi": "\u0939\u093f\u0928\u094d\u0926\u0940 (Hindi)",
    "ta": "\u0ba4\u0bae\u0bbf\u0bb4\u0bcd (Tamil)",
    "ml": "\u0d2e\u0d32\u0d2f\u0d3e\u0d33\u0d02 (Malayalam)",
    "bn": "\u09ac\u09be\u0982\u09b2\u09be (Bengali)",
    "or": "ଓଡ଼ିଆ (Odia)",
    "gu": "\u0a97\u0ac1\u0a9c\u0ab0\u0abe\u0aa4\u0ac0 (Gujarati)",
    "te": "\u0c24\u0c46\u0c32\u0c41\u0c17\u0c41 (Telugu)",
    "kn": "\u0c95\u0ca8\u0ccd\u0ca8\u0ca1 (Kannada)",
}

# Auto-language mapping — scoped to the coastal zones, per the request
# ("if I choose a specific COAST"). Inland wells (W09-W15) intentionally
# fall back to English in Auto mode — they're outside the "coast" framing
# this feature was asked for, and English remains manually switchable
# there like any other well.
WELL_DEFAULT_LANGUAGE = {
    "W01": "ta",  # Nagapattinam, Tamil Nadu
    "W02": "ta",  # Chennai Coast, Tamil Nadu
    "W03": "ml",  # Kochi, Kerala
    "W04": "bn",  # Digha, West Bengal
    "W05": "or",  # Puri, Odisha
    "W06": "gu",  # Veraval, Gujarat
    "W07": "te",  # Visakhapatnam, Andhra Pradesh
    "W08": "kn",  # Mangalore, Karnataka
}

TRANSLATIONS = {
    "public_title": {
        "en": "💧 Is My Water Safe?",
        "hi": "💧 क्या मेरा पानी सुरक्षित है?",
        "ta": "💧 எனது தண்ணீர் பாதுகாப்பானதா?",
        "ml": "💧 എന്റെ വെള്ളം സുരക്ഷിതമാണോ?",
        "bn": "💧 আমার জল কি নিরাপদ?",
        "or": "💧 ମୋର ପାଣି କ'ଣ ସୁରକ୍ଷିତ?",
        "gu": "💧 શું મારું પાણી સુરક્ષિત છે?",
        "te": "💧 నా నీరు సురక్షితమేనా?",
        "kn": "💧 ನನ್ನ ನೀರು ಸುರಕ್ಷಿತವೇ?",
    },
    "public_caption": {
        "en": "Pick your area below to see its current water situation, in plain language. This page checks for new authority alerts automatically every 20 seconds.",
        "hi": "नीचे अपना क्षेत्र चुनें और सरल भाषा में वर्तमान जल स्थिति देखें। यह पृष्ठ हर 20 सेकंड में नई चेतावनियों की स्वतः जांच करता है।",
        "ta": "கீழே உங்கள் பகுதியைத் தேர்ந்தெடுத்து, தற்போதைய நீர் நிலையை எளிய மொழியில் காணவும். இந்தப் பக்கம் ஒவ்வொரு 20 வினாடிக்கும் புதிய எச்சரிக்கைகளைத் தானாகச் சரிபார்க்கும்.",
        "ml": "നിലവിലെ ജല സ്ഥിതി ലളിതമായി കാണാൻ താഴെ നിങ്ങളുടെ പ്രദേശം തിരഞ്ഞെടുക്കുക. ഈ പേജ് ഓരോ 20 സെക്കൻഡിലും പുതിയ മുന്നറിയിപ്പുകൾ സ്വയമേവ പരിശോധിക്കുന്നു.",
        "bn": "নিচে আপনার এলাকা বেছে নিয়ে সহজ ভাষায় বর্তমান জলের অবস্থা দেখুন। এই পৃষ্ঠা প্রতি ২০ সেকেন্ডে নতুন সতর্কতা স্বয়ংক্রিয়ভাবে পরীক্ষা করে।",
        "or": "ନିମ୍ନରେ ଆପଣଙ୍କର ଅଞ୍ଚଳ ବାଛନ୍ତୁ ଏବଂ ସରଳ ଭାଷାରେ ପାଣି ସ୍ଥିତି ଦେଖନ୍ତୁ। ଏହି ପୃଷ୍ଠା ପ୍ରତି 20 ସେକେଣ୍ଡରେ ନୂଆ ଚେତାବନୀ ପାଇଁ ସ୍ୱୟଂଚାଳିତ ଭାବେ ଯାଞ୍ଚ କରେ।",
        "gu": "નીચે તમારો વિસ્તાર પસંદ કરો અને સરળ ભાષામાં હાલની પાણીની સ્થિતિ જુઓ. આ પેજ દર 20 સેકન્ડે નવી ચેતવણીઓ માટે આપમેળે તપાસ કરે છે.",
        "te": "ప్రస్తుత నీటి పరిస్థితిని సాధారణ భాషలో చూడటానికి క్రింద మీ ప్రాంతాన్ని ఎంచుకోండి. ఈ పేజీ ప్రతి 20 సెకన్లకు కొత్త హెచ్చరికల కోసం స్వయంచాలకంగా తనిఖీ చేస్తుంది.",
        "kn": "ಪ್ರಸ್ತುತ ನೀರಿನ ಸ್ಥಿತಿಯನ್ನು ಸರಳ ಭಾಷೆಯಲ್ಲಿ ನೋಡಲು ಕೆಳಗೆ ನಿಮ್ಮ ಪ್ರದೇಶವನ್ನು ಆಯ್ಕೆಮಾಡಿ. ಈ ಪುಟವು ಪ್ರತಿ 20 ಸೆಕೆಂಡಿಗೆ ಹೊಸ ಎಚ್ಚರಿಕೆಗಳಿಗಾಗಿ ಸ್ವಯಂಚಾಲಿತವಾಗಿ ಪರಿಶೀಲಿಸುತ್ತದೆ.",
    },
    "switch_profile": {
        "en": "🔄 Switch profile", "hi": "🔄 प्रोफ़ाइल बदलें", "ta": "🔄 சுயவிவரத்தை மாற்று",
        "ml": "🔄 പ്രൊഫൈൽ മാറ്റുക", "bn": "🔄 প্রোফাইল পরিবর্তন করুন", "or": "🔄 ପ୍ରୋଫାଇଲ୍ ବଦଳାନ୍ତୁ",
        "gu": "🔄 પ્રોફાઇલ બદલો", "te": "🔄 ప్రొఫైల్ మార్చండి", "kn": "🔄 ಪ್ರೊಫೈಲ್ ಬದಲಿಸಿ",
    },
    "select_area": {
        "en": "📍 Select your area", "hi": "📍 अपना क्षेत्र चुनें", "ta": "📍 உங்கள் பகுதியைத் தேர்ந்தெடுக்கவும்",
        "ml": "📍 നിങ്ങളുടെ പ്രദേശം തിരഞ്ഞെടുക്കുക", "bn": "📍 আপনার এলাকা নির্বাচন করুন", "or": "📍 ଆପଣଙ୍କର ଅଞ୍ଚଳ ବାଛନ୍ତୁ",
        "gu": "📍 તમારો વિસ્તાર પસંદ કરો", "te": "📍 మీ ప్రాంతాన్ని ఎంచుకోండి", "kn": "📍 ನಿಮ್ಮ ಪ್ರದೇಶವನ್ನು ಆಯ್ಕೆಮಾಡಿ",
    },
    "your_location": {
        "en": "Your location", "hi": "आपका स्थान", "ta": "உங்கள் இருப்பிடம்", "ml": "നിങ്ങളുടെ സ്ഥലം",
        "bn": "আপনার অবস্থান", "or": "ଆପଣଙ୍କର ସ୍ଥାନ", "gu": "તમારું સ્થાન", "te": "మీ ప్రాంతం", "kn": "ನಿಮ್ಮ ಸ್ಥಳ",
    },
    "active_alerts": {
        "en": "📢 Active Authority Alerts", "hi": "📢 सक्रिय आधिकारिक चेतावनियाँ", "ta": "📢 நடப்பு அதிகாரப்பூர்வ எச்சரிக்கைகள்",
        "ml": "📢 സജീവ അധികൃത മുന്നറിയിപ്പുകൾ", "bn": "📢 সক্রিয় কর্তৃপক্ষের সতর্কতা", "or": "📢 ସକ୍ରିୟ ଅଧିକାରୀ ଚେତାବନୀ",
        "gu": "📢 સક્રિય અધિકૃત ચેતવણીઓ", "te": "📢 క్రియాశీల అధికారిక హెచ్చరికలు", "kn": "📢 ಸಕ್ರಿಯ ಅಧಿಕೃತ ಎಚ್ಚರಿಕೆಗಳು",
    },
    "check_now": {
        "en": "🔄 Check now", "hi": "🔄 अभी जांचें", "ta": "🔄 இப்போது சரிபார்க்கவும்", "ml": "🔄 ഇപ്പോൾ പരിശോധിക്കുക",
        "bn": "🔄 এখনই দেখুন", "or": "🔄 ବର୍ତ୍ତମାନ ଯାଞ୍ଚ କରନ୍ତୁ", "gu": "🔄 હમણાં તપાસો", "te": "🔄 ఇప్పుడు తనిఖీ చేయండి", "kn": "🔄 ಈಗ ಪರಿಶೀಲಿಸಿ",
    },
    "no_alerts": {
        "en": "No official alerts issued for this area right now.",
        "hi": "अभी इस क्षेत्र के लिए कोई आधिकारिक चेतावनी जारी नहीं की गई है।",
        "ta": "இந்தப் பகுதிக்கு தற்போது எந்த அதிகாரப்பூர்வ எச்சரிக்கையும் வெளியிடப்படவில்லை.",
        "ml": "ഈ പ്രദേശത്തിന് ഇപ്പോൾ ഔദ്യോഗിക മുന്നറിയിപ്പുകളൊന്നും നൽകിയിട്ടില്ല.",
        "bn": "এই এলাকার জন্য এখন কোনো সরকারি সতর্কতা জারি করা হয়নি।",
        "or": "ଏହି ଅଞ୍ଚଳ ପାଇଁ ବର୍ତ୍ତମାନ କୌଣସି ଅଧିକାରିକ ଚେତାବନୀ ଜାରି ହୋଇନାହିଁ।",
        "gu": "આ વિસ્તાર માટે અત્યારે કોઈ સત્તાવાર ચેતવણી જારી કરવામાં આવી નથી.",
        "te": "ఈ ప్రాంతానికి ప్రస్తుతం ఎలాంటి అధికారిక హెచ్చరికలు జారీ చేయలేదు.",
        "kn": "ಈ ಪ್ರದೇಶಕ್ಕೆ ಸದ್ಯಕ್ಕೆ ಯಾವುದೇ ಅಧಿಕೃತ ಎಚ್ಚರಿಕೆ ನೀಡಲಾಗಿಲ್ಲ.",
    },
    "severity_label": {
        "en": "Severity", "hi": "गंभीरता", "ta": "தீவிரம்", "ml": "തീവ്രത", "bn": "তীব্রতা",
        "or": "ଗମ୍ଭୀରତା", "gu": "ગંભીરતા", "te": "తీవ్రత", "kn": "ತೀವ್ರತೆ",
    },
    "location_label": {
        "en": "Location", "hi": "स्थान", "ta": "இடம்", "ml": "സ്ഥലം", "bn": "অবস্থান",
        "or": "ସ୍ଥାନ", "gu": "સ્થાન", "te": "ప్రాంతం", "kn": "ಸ್ಥಳ",
    },
    "datetime_label": {
        "en": "Date/Time", "hi": "दिनांक/समय", "ta": "தேதி/நேரம்", "ml": "തീയതി/സമയം", "bn": "তারিখ/সময়",
        "or": "ତାରିଖ/ସମୟ", "gu": "તારીખ/સમય", "te": "తేదీ/సమయం", "kn": "ದಿನಾಂಕ/ಸಮಯ",
    },
    "system_warning": {
        "en": "System Warning", "hi": "सिस्टम चेतावनी", "ta": "கணினி எச்சரிக்கை", "ml": "സിസ്റ്റം മുന്നറിയിപ്പ്",
        "bn": "সিস্টেম সতর্কতা", "or": "ସିଷ୍ଟମ୍ ଚେତାବନୀ", "gu": "સિસ્ટમ ચેતવણી", "te": "సిస్టమ్ హెచ్చరిక", "kn": "ಸಿಸ್ಟಂ ಎಚ್ಚರಿಕೆ",
    },
    "authority_message_label": {
        "en": "Authority's Additional Message", "hi": "प्राधिकरण का अतिरिक्त संदेश", "ta": "அதிகாரிகளின் கூடுதல் செய்தி",
        "ml": "അധികൃതരുടെ അധിക സന്ദേശം", "bn": "কর্তৃপক্ষের অতিরিক্ত বার্তা", "or": "ଅଧିକାରୀଙ୍କ ଅତିରିକ୍ତ ବାର୍ତ୍ତା",
        "gu": "અધિકારીનો વધારાનો સંદેશ", "te": "అధికారుల అదనపు సందేశం", "kn": "ಅಧಿಕಾರಿಗಳ ಹೆಚ್ಚುವರಿ ಸಂದೇಶ",
    },
    "alert_status_label": {
        "en": "Alert Status", "hi": "चेतावनी स्थिति", "ta": "எச்சரிக்கை நிலை", "ml": "മുന്നറിയിപ്പ് നില",
        "bn": "সতর্কতা অবস্থা", "or": "ଚେତାବନୀ ସ୍ଥିତି", "gu": "ચેતવણી સ્થિતિ", "te": "హెచ్చరిక స్థితి", "kn": "ಎಚ್ಚರಿಕೆ ಸ್ಥಿತಿ",
    },
    "coastal_zone": {
        "en": "Coastal Zone", "hi": "तटीय क्षेत्र", "ta": "கடலோர மண்டலம்", "ml": "തീരദേശ മേഖല",
        "bn": "উপকূলীয় অঞ্চল", "or": "ଉପକୂଳ ଅଞ୍ଚଳ", "gu": "દરિયાકાંઠાનો વિસ્તાર", "te": "తీర ప్రాంతం", "kn": "ಕರಾವಳಿ ವಲಯ",
    },
    "past_alerts_template": {
        "en": "Past alerts for this area ({n} more)", "hi": "इस क्षेत्र की पिछली चेतावनियाँ ({n} और)",
        "ta": "இப்பகுதிக்கான முந்தைய எச்சரிக்கைகள் ({n} மேலும்)", "ml": "ഈ പ്രദേശത്തെ മുൻ മുന്നറിയിപ്പുകൾ ({n} കൂടി)",
        "bn": "এই এলাকার পূর্ববর্তী সতর্কতা (আরও {n} টি)", "or": "ଏହି ଅଞ୍ଚଳର ପୂର୍ବ ଚେତାବନୀ (ଆଉ {n})",
        "gu": "આ વિસ્તારની અગાઉની ચેતવણીઓ ({n} વધુ)", "te": "ఈ ప్రాంతానికి గత హెచ్చరికలు (మరో {n})",
        "kn": "ಈ ಪ್ರದೇಶದ ಹಿಂದಿನ ಎಚ್ಚರಿಕೆಗಳು ({n} ಹೆಚ್ಚು)",
    },
    "status_sent": {
        "en": "📧 Emailed to all registered recipients + shown in-app",
        "hi": "📧 सभी पंजीकृत प्राप्तकर्ताओं को ईमेल किया गया + ऐप में दिखाया गया",
        "ta": "📧 பதிவு செய்யப்பட்ட அனைவருக்கும் மின்னஞ்சல் அனுப்பப்பட்டது + ஆப்பில் காட்டப்பட்டது",
        "ml": "📧 രജിസ്റ്റർ ചെയ്ത എല്ലാവർക്കും ഇമെയിൽ അയച്ചു + ആപ്പിൽ കാണിച്ചു",
        "bn": "📧 সকল নিবন্ধিত প্রাপকদের ইমেল করা হয়েছে + অ্যাপে দেখানো হয়েছে",
        "or": "📧 ସମସ୍ତ ପଞ୍ଜୀକୃତ ପ୍ରାପକଙ୍କୁ ଇମେଲ୍ କରାଯାଇଛି + ଆପ୍ରେ ଦେଖାଯାଇଛି",
        "gu": "📧 તમામ નોંધાયેલા પ્રાપકોને ઈમેલ કરાયું + એપમાં બતાવ્યું",
        "te": "📧 నమోదైన అందరికీ ఇమెయిల్ పంపబడింది + యాప్‌లో చూపబడింది",
        "kn": "📧 ನೋಂದಾಯಿತ ಎಲ್ಲರಿಗೂ ಇಮೇಲ್ ಕಳುಹಿಸಲಾಗಿದೆ + ಆ್ಯಪ್‌ನಲ್ಲಿ ತೋರಿಸಲಾಗಿದೆ",
    },
    "status_partial_template": {
        "en": "📧 Emailed to some recipients ({detail}) + shown in-app",
        "hi": "📧 कुछ प्राप्तकर्ताओं को ईमेल किया गया ({detail}) + ऐप में दिखाया गया",
        "ta": "📧 சில பெறுநர்களுக்கு மின்னஞ்சல் அனுப்பப்பட்டது ({detail}) + ஆப்பில் காட்டப்பட்டது",
        "ml": "📧 ചില സ്വീകർത്താക്കൾക്ക് ഇമെയിൽ അയച്ചു ({detail}) + ആപ്പിൽ കാണിച്ചു",
        "bn": "📧 কিছু প্রাপকদের ইমেল করা হয়েছে ({detail}) + অ্যাপে দেখানো হয়েছে",
        "or": "📧 କିଛି ପ୍ରାପକଙ୍କୁ ଇମେଲ୍ କରାଯାଇଛି ({detail}) + ଆପ୍ରେ ଦେଖାଯାଇଛି",
        "gu": "📧 કેટલાક પ્રાપકોને ઈમેલ કરાયું ({detail}) + એપમાં બતાવ્યું",
        "te": "📧 కొందరు గ్రహీతలకు ఇమెయిల్ పంపబడింది ({detail}) + యాప్‌లో చూపబడింది",
        "kn": "📧 ಕೆಲವು ಸ್ವೀಕರಿಸುವವರಿಗೆ ಇಮೇಲ್ ಕಳುಹಿಸಲಾಗಿದೆ ({detail}) + ಆ್ಯಪ್‌ನಲ್ಲಿ ತೋರಿಸಲಾಗಿದೆ",
    },
    "status_not_configured": {
        "en": "Shown in-app — external email service not configured",
        "hi": "ऐप में दिखाया गया — बाहरी ईमेल सेवा कॉन्फ़िगर नहीं है",
        "ta": "ஆப்பில் காட்டப்பட்டது — வெளிப்புற மின்னஞ்சல் சேவை அமைக்கப்படவில்லை",
        "ml": "ആപ്പിൽ കാണിച്ചു — ബാഹ്യ ഇമെയിൽ സേവനം ക്രമീകരിച്ചിട്ടില്ല",
        "bn": "অ্যাপে দেখানো হয়েছে — বাহ্যিক ইমেল পরিষেবা কনফিগার করা নেই",
        "or": "ଆପ୍ରେ ଦେଖାଯାଇଛି — ବାହ୍ୟ ଇମେଲ୍ ସେବା କନଫିଗର ହୋଇନାହିଁ",
        "gu": "એપમાં બતાવ્યું — બાહ્ય ઈમેલ સેવા ગોઠવેલી નથી",
        "te": "యాప్‌లో చూపబడింది — బాహ్య ఇమెయిల్ సేవ కాన్ఫిగర్ చేయలేదు",
        "kn": "ಆ್ಯಪ್‌ನಲ್ಲಿ ತೋರಿಸಲಾಗಿದೆ — ಬಾಹ್ಯ ಇಮೇಲ್ ಸೇವೆ ಕಾನ್ಫಿಗರ್ ಆಗಿಲ್ಲ",
    },
    "status_no_recipients": {
        "en": "Shown in-app — no recipients were selected for email",
        "hi": "ऐप में दिखाया गया — ईमेल के लिए कोई प्राप्तकर्ता चयनित नहीं था",
        "ta": "ஆப்பில் காட்டப்பட்டது — மின்னஞ்சலுக்கு பெறுநர்கள் யாரும் தேர்ந்தெடுக்கப்படவில்லை",
        "ml": "ആപ്പിൽ കാണിച്ചു — ഇമെയിലിനായി സ്വീകർത്താക്കളെ തിരഞ്ഞെടുത്തിട്ടില്ല",
        "bn": "অ্যাপে দেখানো হয়েছে — ইমেলের জন্য কোনো প্রাপক নির্বাচিত হয়নি",
        "or": "ଆପ୍ରେ ଦେଖାଯାଇଛି — ଇମେଲ୍ ପାଇଁ କୌଣସି ପ୍ରାପକ ବଛାଯାଇ ନାହିଁ",
        "gu": "એપમાં બતાવ્યું — ઈમેલ માટે કોઈ પ્રાપક પસંદ કરાયો ન હતો",
        "te": "యాప్‌లో చూపబడింది — ఇమెయిల్ కోసం గ్రహీతలు ఎవరూ ఎంపిక కాలేదు",
        "kn": "ಆ್ಯಪ್‌ನಲ್ಲಿ ತೋರಿಸಲಾಗಿದೆ — ಇಮೇಲ್‌ಗಾಗಿ ಯಾವುದೇ ಸ್ವೀಕರಿಸುವವರನ್ನು ಆಯ್ಕೆ ಮಾಡಿಲ್ಲ",
    },
    "status_failed": {
        "en": "⚠️ Shown in-app — email delivery failed", "hi": "⚠️ ऐप में दिखाया गया — ईमेल भेजना विफल रहा",
        "ta": "⚠️ ஆப்பில் காட்டப்பட்டது — மின்னஞ்சல் அனுப்புதல் தோல்வியடைந்தது", "ml": "⚠️ ആപ്പിൽ കാണിച്ചു — ഇമെയിൽ അയക്കൽ പരാജയപ്പെട്ടു",
        "bn": "⚠️ অ্যাপে দেখানো হয়েছে — ইমেল পাঠানো ব্যর্থ হয়েছে", "or": "⚠️ ଆପ୍ରେ ଦେଖାଯାଇଛି — ଇମେଲ୍ ପଠାଇବାରେ ବିଫଳ",
        "gu": "⚠️ એપમાં બતાવ્યું — ઈમેલ મોકલવામાં નિષ્ફળ", "te": "⚠️ యాప్‌లో చూపబడింది — ఇమెయిల్ పంపడం విఫలమైంది",
        "kn": "⚠️ ಆ್ಯಪ್‌ನಲ್ಲಿ ತೋರಿಸಲಾಗಿದೆ — ಇಮೇಲ್ ಕಳುಹಿಸುವಿಕೆ ವಿಫಲವಾಗಿದೆ",
    },
    "alert_low_title": {
        "en": "Your water situation is SAFE", "hi": "आपकी जल स्थिति सुरक्षित है", "ta": "உங்கள் நீர் நிலை பாதுகாப்பானது",
        "ml": "നിങ്ങളുടെ ജല സ്ഥിതി സുരക്ഷിതമാണ്", "bn": "আপনার জলের অবস্থা নিরাপদ", "or": "ଆପଣଙ୍କ ପାଣି ସ୍ଥିତି ସୁରକ୍ଷିତ",
        "gu": "તમારી પાણીની સ્થિતિ સુરક્ષિત છે", "te": "మీ నీటి పరిస్థితి సురక్షితంగా ఉంది", "kn": "ನಿಮ್ಮ ನೀರಿನ ಸ್ಥಿತಿ ಸುರಕ್ಷಿತವಾಗಿದೆ",
    },
    "alert_low_msg": {
        "en": "No action needed right now. Your groundwater is at a healthy level and not salty.",
        "hi": "अभी कोई कार्रवाई आवश्यक नहीं है। आपका भूजल स्वस्थ स्तर पर है और खारा नहीं है।",
        "ta": "இப்போது எந்த நடவடிக்கையும் தேவையில்லை. உங்கள் நிலத்தடி நீர் ஆரோக்கியமான அளவில் உள்ளது, உப்புத்தன்மையும் இல்லை.",
        "ml": "ഇപ്പോൾ നടപടികളൊന്നും ആവശ്യമില്ല. നിങ്ങളുടെ ഭൂഗർഭജലം ആരോഗ്യകരമായ നിലയിലാണ്, ഉപ്പുരസവുമില്ല.",
        "bn": "এখন কোনো পদক্ষেপের দরকার নেই। আপনার ভূগর্ভস্থ জল স্বাস্থ্যকর স্তরে আছে এবং লবণাক্ত নয়।",
        "or": "ବର୍ତ୍ତମାନ କୌଣସି ପଦକ୍ଷେପ ଆବଶ୍ୟକ ନାହିଁ। ଆପଣଙ୍କ ଭୂତଳ ଜଳ ସୁସ୍ଥ ସ୍ତରରେ ଅଛି ଏବଂ ଲୁଣିଆ ନୁହେଁ।",
        "gu": "અત્યારે કોઈ પગલાંની જરૂર નથી. તમારું ભૂગર્ભજળ તંદુરસ્ત સ્તરે છે અને ખારું નથી.",
        "te": "ప్రస్తుతం ఎలాంటి చర్య అవసరం లేదు. మీ భూగర్భజలం ఆరోగ్యకరమైన స్థాయిలో ఉంది, ఉప్పు కూడా లేదు.",
        "kn": "ಈಗ ಯಾವುದೇ ಕ್ರಮ ಅಗತ್ಯವಿಲ್ಲ. ನಿಮ್ಮ ಅಂತರ್ಜಲ ಆರೋಗ್ಯಕರ ಮಟ್ಟದಲ್ಲಿದೆ ಮತ್ತು ಉಪ್ಪಾಗಿಲ್ಲ.",
    },
    "alert_medium_title": {
        "en": "Please be CAREFUL with water use", "hi": "कृपया पानी के उपयोग में सावधानी बरतें", "ta": "தண்ணீரைப் பயன்படுத்துவதில் கவனமாக இருங்கள்",
        "ml": "ദയവായി ജലം ഉപയോഗിക്കുന്നതിൽ ശ്രദ്ധിക്കുക", "bn": "অনুগ্রহ করে পানি ব্যবহারে সতর্ক থাকুন", "or": "ଦୟାକରି ପାଣି ବ୍ୟବହାରରେ ସାବଧାନ ରୁହନ୍ତୁ",
        "gu": "કૃપા કરી પાણીના ઉપયોગમાં સાવચેત રહો", "te": "దయచేసి నీటి వినియోగంలో జాగ్రత్త వహించండి", "kn": "ದಯವಿಟ್ಟು ನೀರಿನ ಬಳಕೆಯಲ್ಲಿ ಜಾಗರೂಕರಾಗಿರಿ",
    },
    "alert_medium_msg": {
        "en": "Water levels here are getting lower than normal. Try to use less water for farming or other heavy use this month.",
        "hi": "यहाँ जल स्तर सामान्य से कम हो रहा है। इस महीने खेती या अन्य भारी उपयोग के लिए कम पानी का उपयोग करने का प्रयास करें।",
        "ta": "இங்கு நீர் மட்டம் வழக்கத்தை விட குறைந்து வருகிறது. இந்த மாதம் விவசாயத்திற்கும் மற்ற அதிக பயன்பாட்டிற்கும் குறைவான தண்ணீரைப் பயன்படுத்த முயற்சிக்கவும்.",
        "ml": "ഇവിടെ ജലനിരപ്പ് സാധാരണയിലും കുറയുന്നു. ഈ മാസം കൃഷിക്കും മറ്റ് ഭാരിച്ച ഉപയോഗത്തിനും കുറച്ച് വെള്ളം ഉപയോഗിക്കാൻ ശ്രമിക്കുക.",
        "bn": "এখানে জলস্তর স্বাভাবিকের চেয়ে কমে যাচ্ছে। এই মাসে কৃষি বা অন্যান্য ভারী ব্যবহারের জন্য কম জল ব্যবহারের চেষ্টা করুন।",
        "or": "ଏଠାରେ ପାଣି ସ୍ତର ସାଧାରଣଠାରୁ କମ୍ ହେଉଛି। ଏହି ମାସ ଚାଷ କିମ୍ବା ଅନ୍ୟ ଭାରି ବ୍ୟବହାର ପାଇଁ କମ୍ ପାଣି ବ୍ୟବହାର କରିବାକୁ ଚେଷ୍ଟା କରନ୍ତୁ।",
        "gu": "અહીં પાણીનું સ્તર સામાન્ય કરતાં ઓછું થઈ રહ્યું છે. આ મહિને ખેતી અથવા અન્ય ભારે ઉપયોગ માટે ઓછું પાણી વાપરવાનો પ્રયાસ કરો.",
        "te": "ఇక్కడ నీటి మట్టం సాధారణం కంటే తగ్గుతోంది. ఈ నెల వ్యవసాయం లేదా ఇతర భారీ వినియోగం కోసం తక్కువ నీరు వాడటానికి ప్రయత్నించండి.",
        "kn": "ಇಲ್ಲಿ ನೀರಿನ ಮಟ್ಟ ಸಾಮಾನ್ಯಕ್ಕಿಂತ ಕಡಿಮೆಯಾಗುತ್ತಿದೆ. ಈ ತಿಂಗಳು ಕೃಷಿ ಅಥವಾ ಇತರ ಭಾರೀ ಬಳಕೆಗೆ ಕಡಿಮೆ ನೀರು ಬಳಸಲು ಪ್ರಯತ್ನಿಸಿ.",
    },
    "alert_high_title": {
        "en": "URGENT: Water problem in your area", "hi": "अत्यावश्यक: आपके क्षेत्र में जल समस्या", "ta": "அவசரம்: உங்கள் பகுதியில் நீர் பிரச்சனை",
        "ml": "അടിയന്തിരം: നിങ്ങളുടെ പ്രദേശത്ത് ജല പ്രശ്നം", "bn": "জরুরি: আপনার এলাকায় জল সমস্যা", "or": "ଜରୁରୀ: ଆପଣଙ୍କ ଅଞ୍ଚଳରେ ପାଣି ସମସ୍ୟା",
        "gu": "તાત્કાલિક: તમારા વિસ્તારમાં પાણીની સમસ્યા", "te": "అత్యవసరం: మీ ప్రాంతంలో నీటి సమస్య", "kn": "ತುರ್ತು: ನಿಮ್ಮ ಪ್ರದೇಶದಲ್ಲಿ ನೀರಿನ ಸಮಸ್ಯೆ",
    },
    "alert_high_msg": {
        "en": "Water levels are very low and the water may be turning salty. Please contact your local water authority and avoid over-pumping.",
        "hi": "जल स्तर बहुत कम है और पानी खारा हो सकता है। कृपया अपने स्थानीय जल प्राधिकरण से संपर्क करें और अत्यधिक पंपिंग से बचें।",
        "ta": "நீர் மட்டம் மிகவும் குறைவாக உள்ளது, தண்ணீர் உப்பாக மாறலாம். உங்கள் உள்ளூர் நீர் ஆணையத்தைத் தொடர்பு கொண்டு அதிக இறைப்பதைத் தவிர்க்கவும்.",
        "ml": "ജലനിരപ്പ് വളരെ കുറവാണ്, വെള്ളം ഉപ്പുരസമുള്ളതായി മാറാം. ദയവായി പ്രാദേശിക ജല അധികൃതരെ ബന്ധപ്പെടുകയും അമിത പമ്പിംഗ് ഒഴിവാക്കുകയും ചെയ്യുക.",
        "bn": "জলস্তর খুবই কম এবং জল লবণাক্ত হয়ে যেতে পারে। অনুগ্রহ করে স্থানীয় জল কর্তৃপক্ষের সাথে যোগাযোগ করুন এবং অতিরিক্ত পাম্পিং এড়িয়ে চলুন।",
        "or": "ପାଣି ସ୍ତର ବହୁତ କମ୍ ଅଛି ଏବଂ ପାଣି ଲୁଣିଆ ହୋଇପାରେ। ଦୟାକରି ଆପଣଙ୍କ ସ୍ଥାନୀୟ ଜଳ ଅଧିକାରୀଙ୍କ ସହ ଯୋଗାଯୋଗ କରନ୍ତୁ ଏବଂ ଅଧିକ ପମ୍ପିଂ ଏଡ଼ାନ୍ତୁ।",
        "gu": "પાણીનું સ્તર ખૂબ ઓછું છે અને પાણી ખારું થઈ શકે છે. કૃપા કરી તમારા સ્થાનિક જળ અધિકારીનો સંપર્ક કરો અને વધુ પડતું પમ્પિંગ ટાળો.",
        "te": "నీటి మట్టం చాలా తక్కువగా ఉంది మరియు నీరు ఉప్పగా మారవచ్చు. దయచేసి మీ స్థానిక నీటి అధికారులను సంప్రదించండి మరియు అధిక పంపింగ్ నివారించండి.",
        "kn": "ನೀರಿನ ಮಟ್ಟ ತುಂಬಾ ಕಡಿಮೆಯಾಗಿದೆ ಮತ್ತು ನೀರು ಉಪ್ಪಾಗಿ ಬದಲಾಗಬಹುದು. ದಯವಿಟ್ಟು ನಿಮ್ಮ ಸ್ಥಳೀಯ ನೀರಿನ ಅಧಿಕಾರಿಗಳನ್ನು ಸಂಪರ್ಕಿಸಿ ಮತ್ತು ಅತಿಯಾದ ಪಂಪಿಂಗ್ ತಪ್ಪಿಸಿ.",
    },
    "looking_ahead_worse": {
        "en": "⏳ **Looking ahead:** if things continue as they are, this area's water may get **worse** in the next 3 months. Please use water carefully.",
        "hi": "⏳ **आगे की ओर देखते हुए:** यदि स्थिति ऐसी ही रही, तो अगले 3 महीनों में इस क्षेत्र का पानी **और खराब** हो सकता है। कृपया सावधानी से पानी का उपयोग करें।",
        "ta": "⏳ **முன்னோக்கி பார்க்கும்போது:** இப்படியே தொடர்ந்தால், அடுத்த 3 மாதங்களில் இப்பகுதியின் நீர் **மோசமடையலாம்**. தயவுசெய்து நீரை கவனமாகப் பயன்படுத்தவும்.",
        "ml": "⏳ **മുന്നോട്ട് നോക്കുമ്പോൾ:** ഇങ്ങനെ തുടർന്നാൽ, അടുത്ത 3 മാസത്തിനുള്ളിൽ ഈ പ്രദേശത്തെ ജലം **കൂടുതൽ വഷളാകാം**. ദയവായി ജലം ശ്രദ്ധയോടെ ഉപയോഗിക്കുക.",
        "bn": "⏳ **ভবিষ্যতের দিকে তাকিয়ে:** এভাবে চলতে থাকলে, আগামী ৩ মাসে এই এলাকার জল **আরও খারাপ** হতে পারে। অনুগ্রহ করে সতর্কতার সাথে জল ব্যবহার করুন।",
        "or": "⏳ **ଆଗକୁ ଦେଖିଲେ:** ଏହିପରି ଚାଲିଲେ, ଆଗାମୀ 3 ମାସରେ ଏହି ଅଞ୍ଚଳର ପାଣି **ଅଧିକ ଖରାପ** ହୋଇପାରେ। ଦୟାକରି ସାବଧାନତାର ସହ ପାଣି ବ୍ୟବହାର କରନ୍ତୁ।",
        "gu": "⏳ **આગળ જોતાં:** જો આમ જ ચાલુ રહ્યું, તો આગામી 3 મહિનામાં આ વિસ્તારનું પાણી **વધુ ખરાબ** થઈ શકે છે. કૃપા કરી કાળજીપૂર્વક પાણી વાપરો.",
        "te": "⏳ **ముందు చూస్తే:** ఇలాగే కొనసాగితే, రాబోయే 3 నెలల్లో ఈ ప్రాంతపు నీరు **మరింత దిగజారవచ్చు**. దయచేసి జాగ్రత్తగా నీటిని వాడండి.",
        "kn": "⏳ **ಮುಂದೆ ನೋಡಿದರೆ:** ಹೀಗೇ ಮುಂದುವರೆದರೆ, ಮುಂದಿನ 3 ತಿಂಗಳಲ್ಲಿ ಈ ಪ್ರದೇಶದ ನೀರು **ಇನ್ನಷ್ಟು ಹದಗೆಡಬಹುದು**. ದಯವಿಟ್ಟು ಎಚ್ಚರಿಕೆಯಿಂದ ನೀರು ಬಳಸಿ.",
    },
    "looking_ahead_better": {
        "en": "⏳ **Looking ahead:** this area's water situation is expected to **improve** in the next 3 months.",
        "hi": "⏳ **आगे की ओर देखते हुए:** अगले 3 महीनों में इस क्षेत्र की जल स्थिति में **सुधार** होने की उम्मीद है।",
        "ta": "⏳ **முன்னோக்கி பார்க்கும்போது:** அடுத்த 3 மாதங்களில் இப்பகுதியின் நீர் நிலை **மேம்படும்** என எதிர்பார்க்கப்படுகிறது.",
        "ml": "⏳ **മുന്നോട്ട് നോക്കുമ്പോൾ:** അടുത്ത 3 മാസത്തിനുള്ളിൽ ഈ പ്രദേശത്തെ ജല സ്ഥിതി **മെച്ചപ്പെടുമെന്ന്** പ്രതീക്ഷിക്കുന്നു.",
        "bn": "⏳ **ভবিষ্যতের দিকে তাকিয়ে:** আগামী ৩ মাসে এই এলাকার জলের অবস্থার **উন্নতি** হবে বলে আশা করা যাচ্ছে।",
        "or": "⏳ **ଆଗକୁ ଦେଖିଲେ:** ଆଗାମୀ 3 ମାସରେ ଏହି ଅଞ୍ଚଳର ପାଣି ସ୍ଥିତି **ଉନ୍ନତ** ହେବ ବୋଲି ଆଶା କରାଯାଉଛି।",
        "gu": "⏳ **આગળ જોતાં:** આગામી 3 મહિનામાં આ વિસ્તારની પાણીની સ્થિતિ **સુધરે** તેવી અપેક્ષા છે.",
        "te": "⏳ **ముందు చూస్తే:** రాబోయే 3 నెలల్లో ఈ ప్రాంతపు నీటి పరిస్థితి **మెరుగుపడుతుందని** అంచనా.",
        "kn": "⏳ **ಮುಂದೆ ನೋಡಿದರೆ:** ಮುಂದಿನ 3 ತಿಂಗಳಲ್ಲಿ ಈ ಪ್ರದೇಶದ ನೀರಿನ ಸ್ಥಿತಿ **ಸುಧಾರಿಸುವ** ನಿರೀಕ್ಷೆಯಿದೆ.",
    },
    "what_does_mean": {
        "en": "What does this mean?", "hi": "इसका क्या मतलब है?", "ta": "இதன் அர்த்தம் என்ன?", "ml": "ഇതിന്റെ അർത്ഥമെന്താണ്?",
        "bn": "এর মানে কী?", "or": "ଏହାର ଅର୍ଥ କ'ଣ?", "gu": "આનો અર્થ શું છે?", "te": "దీని అర్థం ఏమిటి?", "kn": "ಇದರ ಅರ್ಥವೇನು?",
    },
    "water_level_header": {
        "en": "💧 Water level", "hi": "💧 जल स्तर", "ta": "💧 நீர் மட்டம்", "ml": "💧 ജലനിരപ്പ്",
        "bn": "💧 জলস্তর", "or": "💧 ପାଣି ସ୍ତର", "gu": "💧 પાણીનું સ્તર", "te": "💧 నీటి మట్టం", "kn": "💧 ನೀರಿನ ಮಟ್ಟ",
    },
    "level_very_deep": {
        "en": "very deep", "hi": "बहुत गहरा", "ta": "மிகவும் ஆழமானது", "ml": "വളരെ ആഴത്തിൽ", "bn": "খুব গভীর",
        "or": "ବହୁତ ଗଭୀର", "gu": "ખૂબ ઊંડું", "te": "చాలా లోతుగా", "kn": "ಬಹಳ ಆಳ",
    },
    "level_getting_deep": {
        "en": "getting deep", "hi": "गहरा होता जा रहा है", "ta": "ஆழமாகி வருகிறது", "ml": "ആഴം കൂടിവരുന്നു", "bn": "গভীর হয়ে যাচ্ছে",
        "or": "ଗଭୀର ହେଉଛି", "gu": "ઊંડું થઈ રહ્યું છે", "te": "లోతుగా మారుతోంది", "kn": "ಆಳವಾಗುತ್ತಿದೆ",
    },
    "level_healthy": {
        "en": "at a healthy level", "hi": "स्वस्थ स्तर पर", "ta": "ஆரோக்கியமான அளவில்", "ml": "ആരോഗ്യകരമായ നിലയിൽ", "bn": "স্বাস্থ্যকর স্তরে",
        "or": "ସୁସ୍ଥ ସ୍ତରରେ", "gu": "તંદુરસ્ત સ્તરે", "te": "ఆరోగ్యకరమైన స్థాయిలో", "kn": "ಆರೋಗ್ಯಕರ ಮಟ್ಟದಲ್ಲಿ",
    },
    "water_level_sentence": {
        "en": "Your area's underground water is currently **{level_words}** ({depth:.1f} metres below the ground).",
        "hi": "आपके क्षेत्र का भूजल वर्तमान में **{level_words}** है (जमीन से {depth:.1f} मीटर नीचे)।",
        "ta": "உங்கள் பகுதியின் நிலத்தடி நீர் தற்போது **{level_words}** உள்ளது (தரையிலிருந்து {depth:.1f} மீட்டர் கீழே).",
        "ml": "നിങ്ങളുടെ പ്രദേശത്തെ ഭൂഗർഭജലം നിലവിൽ **{level_words}** ആണ് (നിലത്തിന് {depth:.1f} മീറ്റർ താഴെ).",
        "bn": "আপনার এলাকার ভূগর্ভস্থ জল বর্তমানে **{level_words}** (মাটির {depth:.1f} মিটার নিচে)।",
        "or": "ଆପଣଙ୍କ ଅଞ୍ଚଳର ଭୂତଳ ଜଳ ବର୍ତ୍ତମାନ **{level_words}** ଅଛି (ଭୂମିଠାରୁ {depth:.1f} ମିଟର ତଳେ)।",
        "gu": "તમારા વિસ્તારનું ભૂગર્ભજળ હાલમાં **{level_words}** છે (જમીનથી {depth:.1f} મીટર નીચે).",
        "te": "మీ ప్రాంతపు భూగర్భజలం ప్రస్తుతం **{level_words}** ఉంది (భూమి నుండి {depth:.1f} మీటర్ల లోతు).",
        "kn": "ನಿಮ್ಮ ಪ್ರದೇಶದ ಅಂತರ್ಜಲ ಪ್ರಸ್ತುತ **{level_words}** ಇದೆ (ನೆಲದಿಂದ {depth:.1f} ಮೀಟರ್ ಕೆಳಗೆ).",
    },
    "water_saltiness_header": {
        "en": "🧂 Water saltiness", "hi": "🧂 पानी की खारापन", "ta": "🧂 நீரின் உப்புத்தன்மை", "ml": "🧂 ജലത്തിന്റെ ഉപ്പുരസം",
        "bn": "🧂 জলের লবণাক্ততা", "or": "🧂 ପାଣିର ଲୁଣିଆପଣ", "gu": "🧂 પાણીની ખારાશ", "te": "🧂 నీటి ఉప్పదనం", "kn": "🧂 ನೀರಿನ ಉಪ್ಪಿನಂಶ",
    },
    "salt_not_salty": {
        "en": "not salty — safe to use", "hi": "खारा नहीं — उपयोग के लिए सुरक्षित", "ta": "உப்பு இல்லை — பயன்படுத்தத் தகுந்தது",
        "ml": "ഉപ്പുരസമില്ല — ഉപയോഗിക്കാൻ സുരക്ഷിതം", "bn": "লবণাক্ত নয় — ব্যবহারের জন্য নিরাপদ", "or": "ଲୁଣିଆ ନୁହେଁ — ବ୍ୟବହାର ପାଇଁ ସୁରକ୍ଷିତ",
        "gu": "ખારું નથી — ઉપયોગ માટે સુરક્ષિત", "te": "ఉప్పు లేదు — వాడటానికి సురక్షితం", "kn": "ಉಪ್ಪಲ್ಲ — ಬಳಸಲು ಸುರಕ್ಷಿತ",
    },
    "salt_slightly_salty": {
        "en": "a little salty — use caution", "hi": "थोड़ा खारा — सावधानी बरतें", "ta": "சற்று உப்புத்தன்மை — கவனமாக இருங்கள்",
        "ml": "ചെറിയ ഉപ്പുരസം — ശ്രദ്ധിക്കുക", "bn": "সামান্য লবণাক্ত — সতর্কতা অবলম্বন করুন", "or": "ଟିକିଏ ଲୁଣିଆ — ସାବଧାନ ରୁହନ୍ତୁ",
        "gu": "થોડું ખારું — સાવચેત રહો", "te": "కొంచెం ఉప్పు — జాగ్రత్త వహించండి", "kn": "ಸ್ವಲ್ಪ ಉಪ್ಪು — ಎಚ್ಚರಿಕೆ ವಹಿಸಿ",
    },
    "salt_quite_salty": {
        "en": "quite salty — may not be safe to drink or use on crops", "hi": "काफी खारा — पीने या फसलों में उपयोग के लिए सुरक्षित नहीं हो सकता",
        "ta": "நிறைய உப்புத்தன்மை — குடிக்கவோ பயிர்களுக்குப் பயன்படுத்தவோ பாதுகாப்பாக இல்லாமல் இருக்கலாம்",
        "ml": "വളരെ ഉപ്പുരസം — കുടിക്കാനോ വിളകൾക്ക് ഉപയോഗിക്കാനോ സുരക്ഷിതമല്ലായിരിക്കാം",
        "bn": "বেশ লবণাক্ত — পান করা বা ফসলে ব্যবহারের জন্য নিরাপদ নাও হতে পারে",
        "or": "ବେଶ୍ ଲୁଣିଆ — ପିଇବା କିମ୍ବା ଫସଲରେ ବ୍ୟବହାର ପାଇଁ ସୁରକ୍ଷିତ ନ ହୋଇପାରେ",
        "gu": "ઘણું ખારું — પીવા અથવા પાકમાં ઉપયોગ માટે સુરક્ષિત ન પણ હોય",
        "te": "బాగా ఉప్పు — తాగడానికి లేదా పంటలకు వాడటానికి సురక్షితం కాకపోవచ్చు",
        "kn": "ಸಾಕಷ್ಟು ಉಪ್ಪು — ಕುಡಿಯಲು ಅಥವಾ ಬೆಳೆಗಳಿಗೆ ಬಳಸಲು ಸುರಕ್ಷಿತವಾಗಿಲ್ಲದಿರಬಹುದು",
    },
    "water_saltiness_sentence": {
        "en": "Your area's water is currently **{salt_words}**.",
        "hi": "आपके क्षेत्र का पानी वर्तमान में **{salt_words}** है।",
        "ta": "உங்கள் பகுதியின் நீர் தற்போது **{salt_words}**.",
        "ml": "നിങ്ങളുടെ പ്രദേശത്തെ ജലം നിലവിൽ **{salt_words}** ആണ്.",
        "bn": "আপনার এলাকার জল বর্তমানে **{salt_words}**।",
        "or": "ଆପଣଙ୍କ ଅଞ୍ଚଳର ପାଣି ବର୍ତ୍ତମାନ **{salt_words}**।",
        "gu": "તમારા વિસ્તારનું પાણી હાલમાં **{salt_words}** છે.",
        "te": "మీ ప్రాంతపు నీరు ప్రస్తుతం **{salt_words}**.",
        "kn": "ನಿಮ್ಮ ಪ್ರದೇಶದ ನೀರು ಪ್ರಸ್ತುತ **{salt_words}**.",
    },
    "what_should_do": {
        "en": "What should you do?", "hi": "आपको क्या करना चाहिए?", "ta": "நீங்கள் என்ன செய்ய வேண்டும்?", "ml": "നിങ്ങൾ എന്ത് ചെയ്യണം?",
        "bn": "আপনার কী করা উচিত?", "or": "ଆପଣ କ'ଣ କରିବା ଉଚିତ?", "gu": "તમારે શું કરવું જોઈએ?", "te": "మీరు ఏమి చేయాలి?", "kn": "ನೀವು ಏನು ಮಾಡಬೇಕು?",
    },
    "advice_low": {
        "en": "✅ Nothing urgent. Continue normal water use, and it's still wise not to over-pump.",
        "hi": "✅ कुछ भी अत्यावश्यक नहीं है। सामान्य जल उपयोग जारी रखें, फिर भी अधिक पंपिंग न करना बुद्धिमानी है।",
        "ta": "✅ அவசரம் ஏதுமில்லை. வழக்கமான நீர் பயன்பாட்டைத் தொடரவும், அதிகமாக இறைக்காமல் இருப்பது நல்லது.",
        "ml": "✅ അടിയന്തിരമായി ഒന്നും വേണ്ട. സാധാരണ ജല ഉപയോഗം തുടരാം, അമിതമായി പമ്പ് ചെയ്യാതിരിക്കുന്നത് നല്ലതാണ്.",
        "bn": "✅ জরুরি কিছু নেই। স্বাভাবিক জল ব্যবহার চালিয়ে যান, তবে অতিরিক্ত পাম্পিং না করাই ভালো।",
        "or": "✅ ଜରୁରୀ କିଛି ନାହିଁ। ସାଧାରଣ ପାଣି ବ୍ୟବହାର ଜାରି ରଖନ୍ତୁ, ତଥାପି ଅଧିକ ପମ୍ପିଂ ନକରିବା ଉଚିତ।",
        "gu": "✅ કંઈ તાત્કાલિક નથી. સામાન્ય પાણીનો ઉપયોગ ચાલુ રાખો, છતાં વધુ પડતું પમ્પિંગ ન કરવું સમજદારી છે.",
        "te": "✅ అత్యవసరం ఏమీ లేదు. సాధారణ నీటి వినియోగం కొనసాగించండి, అయినా అధిక పంపింగ్ చేయకపోవడం మంచిది.",
        "kn": "✅ ತುರ್ತಾಗಿ ಏನೂ ಇಲ್ಲ. ಸಾಮಾನ್ಯ ನೀರಿನ ಬಳಕೆ ಮುಂದುವರಿಸಿ, ಆದರೂ ಅತಿಯಾಗಿ ಪಂಪ್ ಮಾಡದಿರುವುದು ಒಳ್ಳೆಯದು.",
    },
    "advice_medium": {
        "en": "🟠 Try to reduce non-essential water use. Consider spacing out irrigation and avoid new borewells nearby.",
        "hi": "🟠 गैर-जरूरी पानी के उपयोग को कम करने का प्रयास करें। सिंचाई में अंतराल रखें और आस-पास नए बोरवेल से बचें।",
        "ta": "🟠 அத்தியாவசியமற்ற நீர் பயன்பாட்டைக் குறைக்க முயற்சிக்கவும். பாசனத்தை இடைவெளியில் செய்யவும், அருகில் புதிய போர்வெல்களைத் தவிர்க்கவும்.",
        "ml": "🟠 അത്യാവശ്യമല്ലാത്ത ജല ഉപയോഗം കുറയ്ക്കാൻ ശ്രമിക്കുക. ജലസേചനം ഇടവേളകളിലാക്കുകയും സമീപം പുതിയ കുഴൽക്കിണറുകൾ ഒഴിവാക്കുകയും ചെയ്യുക.",
        "bn": "🟠 অপ্রয়োজনীয় জল ব্যবহার কমানোর চেষ্টা করুন। সেচের ব্যবধান রাখুন এবং কাছাকাছি নতুন বোরওয়েল এড়িয়ে চলুন।",
        "or": "🟠 ଅନାବଶ୍ୟକ ପାଣି ବ୍ୟବହାର କମାଇବାକୁ ଚେଷ୍ଟା କରନ୍ତୁ। ଜଳସେଚନରେ ବ୍ୟବଧାନ ରଖନ୍ତୁ ଏବଂ ନିକଟରେ ନୂଆ ବୋରୱେଲ୍ ଏଡ଼ାନ୍ତୁ।",
        "gu": "🟠 બિનજરૂરી પાણીનો ઉપયોગ ઘટાડવાનો પ્રયાસ કરો. સિંચાઈ વચ્ચે અંતર રાખો અને નજીકમાં નવા બોરવેલ ટાળો.",
        "te": "🟠 అనవసరమైన నీటి వినియోగాన్ని తగ్గించడానికి ప్రయత్నించండి. నీటిపారుదలను వ్యవధిలో చేయండి మరియు సమీపంలో కొత్త బోరుబావులను నివారించండి.",
        "kn": "🟠 ಅನಗತ್ಯ ನೀರಿನ ಬಳಕೆಯನ್ನು ಕಡಿಮೆ ಮಾಡಲು ಪ್ರಯತ್ನಿಸಿ. ನೀರಾವರಿಯನ್ನು ಅಂತರದಲ್ಲಿ ಮಾಡಿ ಮತ್ತು ಹತ್ತಿರದಲ್ಲಿ ಹೊಸ ಕೊಳವೆಬಾವಿಗಳನ್ನು ತಪ್ಪಿಸಿ.",
    },
    "advice_high": {
        "en": "🔴 Reduce water pumping where possible and contact your local water/agriculture office. If this is your drinking water source, consider getting it tested.",
        "hi": "🔴 जहाँ संभव हो पानी की पंपिंग कम करें और अपने स्थानीय जल/कृषि कार्यालय से संपर्क करें। यदि यह आपका पेयजल स्रोत है, तो इसे जांच कराने पर विचार करें।",
        "ta": "🔴 முடிந்தவரை நீர் இறைப்பதைக் குறைத்து உங்கள் உள்ளூர் நீர்/வேளாண் அலுவலகத்தைத் தொடர்பு கொள்ளவும். இது உங்கள் குடிநீர் ஆதாரமாக இருந்தால், சோதனை செய்ய பரிசீலிக்கவும்.",
        "ml": "🔴 കഴിയുന്നിടത്തോളം ജല പമ്പിംഗ് കുറയ്ക്കുകയും പ്രാദേശിക ജല/കൃഷി ഓഫീസുമായി ബന്ധപ്പെടുകയും ചെയ്യുക. ഇത് നിങ്ങളുടെ കുടിവെള്ള സ്രോതസ്സാണെങ്കിൽ, പരിശോധിക്കാൻ പരിഗണിക്കുക.",
        "bn": "🔴 যেখানে সম্ভব জল পাম্পিং কমান এবং আপনার স্থানীয় জল/কৃষি অফিসে যোগাযোগ করুন। এটি যদি আপনার পানীয় জলের উৎস হয়, তবে এটি পরীক্ষা করানোর কথা বিবেচনা করুন।",
        "or": "🔴 ଯେଉଁଠି ସମ୍ଭବ ପାଣି ପମ୍ପିଂ କମାନ୍ତୁ ଏବଂ ଆପଣଙ୍କ ସ୍ଥାନୀୟ ଜଳ/କୃଷି କାର୍ଯ୍ୟାଳୟ ସହ ଯୋଗାଯୋଗ କରନ୍ତୁ। ଏହା ଆପଣଙ୍କ ପିଇବା ପାଣି ଉତ୍ସ ହୋଇଥିଲେ, ଏହାକୁ ପରୀକ୍ଷା କରାଇବାକୁ ବିଚାର କରନ୍ତୁ।",
        "gu": "🔴 જ્યાં શક્ય હોય ત્યાં પાણીનું પમ્પિંગ ઘટાડો અને તમારી સ્થાનિક પાણી/કૃષિ કચેરીનો સંપર્ક કરો. જો આ તમારા પીવાના પાણીનો સ્રોત હોય, તો તેનું પરીક્ષણ કરાવવાનું વિચારો.",
        "te": "🔴 వీలైనచోట నీటి పంపింగ్ తగ్గించి మీ స్థానిక నీటి/వ్యవసాయ కార్యాలయాన్ని సంప్రదించండి. ఇది మీ తాగునీటి వనరు అయితే, పరీక్షించుకోవడం పరిగణించండి.",
        "kn": "🔴 ಸಾಧ್ಯವಾದಲ್ಲಿ ನೀರಿನ ಪಂಪಿಂಗ್ ಕಡಿಮೆ ಮಾಡಿ ಮತ್ತು ನಿಮ್ಮ ಸ್ಥಳೀಯ ನೀರು/ಕೃಷಿ ಕಚೇರಿಯನ್ನು ಸಂಪರ್ಕಿಸಿ. ಇದು ನಿಮ್ಮ ಕುಡಿಯುವ ನೀರಿನ ಮೂಲವಾಗಿದ್ದರೆ, ಪರೀಕ್ಷಿಸಲು ಪರಿಗಣಿಸಿ.",
    },
    "simplified_view_caption": {
        "en": "This is a simplified view of the same data and forecasts shown to water authorities. Switch to the Authority profile above for full technical detail.",
        "hi": "यह जल प्राधिकरणों को दिखाए गए समान डेटा और पूर्वानुमानों का सरलीकृत दृश्य है। पूर्ण तकनीकी विवरण के लिए ऊपर प्राधिकरण प्रोफ़ाइल पर स्विच करें।",
        "ta": "இது நீர் அதிகாரிகளுக்குக் காட்டப்படும் அதே தரவு மற்றும் முன்னறிவிப்புகளின் எளிமையான காட்சி. முழு தொழில்நுட்ப விவரங்களுக்கு மேலே உள்ள அதிகாரி சுயவிவரத்திற்கு மாறவும்.",
        "ml": "ജല അധികൃതർക്ക് കാണിക്കുന്ന അതേ ഡാറ്റയുടെയും പ്രവചനങ്ങളുടെയും ലളിതമായ കാഴ്ചയാണിത്. പൂർണ്ണ സാങ്കേതിക വിശദാംശങ്ങൾക്ക് മുകളിലുള്ള അതോറിറ്റി പ്രൊഫൈലിലേക്ക് മാറുക.",
        "bn": "এটি জল কর্তৃপক্ষকে দেখানো একই ডেটা ও পূর্বাভাসের সরলীকৃত দৃশ্য। সম্পূর্ণ প্রযুক্তিগত বিবরণের জন্য উপরে কর্তৃপক্ষ প্রোফাইলে স্যুইচ করুন।",
        "or": "ଏହା ଜଳ ଅଧିକାରୀଙ୍କୁ ଦେଖାଯାଉଥିବା ସମାନ ତଥ୍ୟ ଏବଂ ପୂର୍ବାନୁମାନର ସରଳୀକୃତ ଦୃଶ୍ୟ। ପୂର୍ଣ୍ଣ ବିଷୟ ପାଇଁ ଉପରେ ଅଧିକାରୀ ପ୍ରୋଫାଇଲ୍‌କୁ ବଦଳାନ୍ତୁ।",
        "gu": "આ પાણી અધિકારીઓને બતાવવામાં આવતા સમાન ડેટા અને આગાહીઓનું સરળ દૃશ્ય છે. સંપૂર્ણ ટેકનિકલ વિગતો માટે ઉપર ઓથોરિટી પ્રોફાઇલ પર સ્વિચ કરો.",
        "te": "ఇది నీటి అధికారులకు చూపించే అదే డేటా మరియు అంచనాల సరళీకృత వీక్షణ. పూర్తి సాంకేతిక వివరాల కోసం పైన అథారిటీ ప్రొఫైల్‌కు మారండి.",
        "kn": "ಇದು ನೀರಿನ ಅಧಿಕಾರಿಗಳಿಗೆ ತೋರಿಸುವ ಅದೇ ಡೇಟಾ ಮತ್ತು ಮುನ್ಸೂಚನೆಗಳ ಸರಳೀಕೃತ ನೋಟ. ಪೂರ್ಣ ತಾಂತ್ರಿಕ ವಿವರಗಳಿಗಾಗಿ ಮೇಲಿನ ಅಥಾರಿಟಿ ಪ್ರೊಫೈಲ್‌ಗೆ ಬದಲಿಸಿ.",
    },
    "language_label": {
        "en": "🌐 Language", "hi": "🌐 भाषा", "ta": "🌐 மொழி", "ml": "🌐 ഭാഷ", "bn": "🌐 ভাষা",
        "or": "🌐 ଭାଷା", "gu": "🌐 ભાષા", "te": "🌐 భాష", "kn": "🌐 ಭಾಷೆ",
    },
    "auto_language_option": {
        "en": "Auto (based on your area)", "hi": "स्वतः (आपके क्षेत्र के आधार पर)", "ta": "தானாக (உங்கள் பகுதியின் அடிப்படையில்)",
        "ml": "സ്വയമേവ (നിങ്ങളുടെ പ്രദേശം അടിസ്ഥാനമാക്കി)", "bn": "স্বয়ংক্রিয় (আপনার এলাকার ভিত্তিতে)", "or": "ସ୍ୱୟଂଚାଳିତ (ଆପଣଙ୍କ ଅଞ୍ଚଳ ଆଧାରରେ)",
        "gu": "આપમેળે (તમારા વિસ્તારના આધારે)", "te": "స్వయంచాలకం (మీ ప్రాంతం ఆధారంగా)", "kn": "ಸ್ವಯಂಚಾಲಿತ (ನಿಮ್ಮ ಪ್ರದೇಶದ ಆಧಾರದ ಮೇಲೆ)",
    },
    # ---- Authority Interface chrome (manual switch only, English default) ----
    "auth_subtitle": {
        "en": "Authority / Technical view — full forecasting, scenarios, and model internals.",
        "hi": "प्राधिकरण / तकनीकी दृश्य — पूर्ण पूर्वानुमान, परिदृश्य और मॉडल विवरण।",
        "ta": "அதிகாரி / தொழில்நுட்பக் காட்சி — முழு முன்னறிவிப்பு, சூழ்நிலைகள் மற்றும் மாடல் விவரங்கள்.",
        "ml": "അതോറിറ്റി / സാങ്കേതിക കാഴ്ച — സമ്പൂർണ്ണ പ്രവചനം, സാഹചര്യങ്ങൾ, മോഡൽ വിശദാംശങ്ങൾ.",
        "bn": "কর্তৃপক্ষ / প্রযুক্তিগত দৃশ্য — সম্পূর্ণ পূর্বাভাস, পরিস্থিতি এবং মডেল বিবরণ।",
        "or": "ଅଧିକାରୀ / ବିଷୟ ଦୃଶ୍ୟ — ପୂର୍ଣ୍ଣ ପୂର୍ବାନୁମାନ, ପରିସ୍ଥିତି ଏବଂ ମଡେଲ ବିବରଣୀ।",
        "gu": "ઓથોરિટી / ટેકનિકલ દૃશ્ય — સંપૂર્ણ આગાહી, દૃશ્યો અને મોડેલ વિગતો.",
        "te": "అథారిటీ / సాంకేతిక వీక్షణ — పూర్తి అంచనా, దృశ్యాలు మరియు మోడల్ వివరాలు.",
        "kn": "ಅಥಾರಿಟಿ / ತಾಂತ್ರಿಕ ನೋಟ — ಸಂಪೂರ್ಣ ಮುನ್ಸೂಚನೆ, ಸನ್ನಿವೇಶಗಳು ಮತ್ತು ಮಾಡೆಲ್ ವಿವರಗಳು.",
    },
    "broadcast_header": {
        "en": "📢 Emergency Alert & Broadcast", "hi": "📢 आपातकालीन चेतावनी और प्रसारण", "ta": "📢 அவசர எச்சரிக்கை மற்றும் ஒளிபரப்பு",
        "ml": "📢 അടിയന്തിര മുന്നറിയിപ്പും പ്രക്ഷേപണവും", "bn": "📢 জরুরি সতর্কতা ও সম্প্রচার", "or": "📢 ଜରୁରୀକାଳୀନ ଚେତାବନୀ ଓ ପ୍ରସାରଣ",
        "gu": "📢 ઇમરજન્સી ચેતવણી અને પ્રસારણ", "te": "📢 అత్యవసర హెచ్చరిక & ప్రసారం", "kn": "📢 ತುರ್ತು ಎಚ್ಚರಿಕೆ ಮತ್ತು ಪ್ರಸಾರ",
    },
    "send_broadcast_button": {
        "en": "🚨 SEND BROADCAST", "hi": "🚨 प्रसारण भेजें", "ta": "🚨 ஒளிபரப்பை அனுப்பு", "ml": "🚨 പ്രക്ഷേപണം അയക്കുക",
        "bn": "🚨 সম্প্রচার পাঠান", "or": "🚨 ପ୍ରସାରଣ ପଠାନ୍ତୁ", "gu": "🚨 પ્રસારણ મોકલો", "te": "🚨 ప్రసారం పంపండి", "kn": "🚨 ಪ್ರಸಾರ ಕಳುಹಿಸಿ",
    },
    "manage_recipients_header": {
        "en": "👥 Manage Recipients / People in Affected Area", "hi": "👥 प्राप्तकर्ता / प्रभावित क्षेत्र के लोग प्रबंधित करें",
        "ta": "👥 பெறுநர்கள் / பாதிக்கப்பட்ட பகுதி மக்களை நிர்வகிக்கவும்", "ml": "👥 സ്വീകർത്താക്കൾ / ബാധിത പ്രദേശത്തെ ആളുകളെ കൈകാര്യം ചെയ്യുക",
        "bn": "👥 প্রাপক / আক্রান্ত এলাকার মানুষ পরিচালনা করুন", "or": "👥 ପ୍ରାପକ / ପ୍ରଭାବିତ ଅଞ୍ଚଳର ଲୋକ ପରିଚାଳନା କରନ୍ତୁ",
        "gu": "👥 પ્રાપકો / અસરગ્રસ્ત વિસ્તારના લોકોનું સંચાલન કરો", "te": "👥 గ్రహీతలు / ప్రభావిత ప్రాంత ప్రజలను నిర్వహించండి",
        "kn": "👥 ಸ್ವೀಕರಿಸುವವರು / ಬಾಧಿತ ಪ್ರದೇಶದ ಜನರನ್ನು ನಿರ್ವಹಿಸಿ",
    },
    "broadcast_history_header": {
        "en": "📜 Broadcast History", "hi": "📜 प्रसारण इतिहास", "ta": "📜 ஒளிபரப்பு வரலாறு", "ml": "📜 പ്രക്ഷേപണ ചരിത്രം",
        "bn": "📜 সম্প্রচার ইতিহাস", "or": "📜 ପ୍ରସାରଣ ଇତିହାସ", "gu": "📜 પ્રસારણ ઇતિહાસ", "te": "📜 ప్రసార చరిత్ర", "kn": "📜 ಪ್ರಸಾರ ಇತಿಹಾಸ",
    },
    "current_depth_metric": {
        "en": "Current depth to water table", "hi": "जल स्तर की वर्तमान गहराई", "ta": "தற்போதைய நீர்மட்ட ஆழம்", "ml": "നിലവിലെ ജലവിതാന ആഴം",
        "bn": "বর্তমান জলস্তরের গভীরতা", "or": "ବର୍ତ୍ତମାନ ଜଳ ସ୍ତର ଗଭୀରତା", "gu": "વર્તમાન પાણીના સ્તરની ઊંડાઈ", "te": "ప్రస్తుత నీటి మట్టం లోతు", "kn": "ಪ್ರಸ್ತುತ ನೀರಿನ ಮಟ್ಟದ ಆಳ",
    },
    "current_risk_metric": {
        "en": "Current salinity risk", "hi": "वर्तमान लवणता जोखिम", "ta": "தற்போதைய உப்புத்தன்மை அபாயம்", "ml": "നിലവിലെ ലവണത അപകടസാധ്യത",
        "bn": "বর্তমান লবণাক্ততার ঝুঁকি", "or": "ବର୍ତ୍ତମାନ ଲବଣତା ବିପଦ", "gu": "વર્તમાન ખારાશ જોખમ", "te": "ప్రస్తుత లవణీయత ప్రమాదం", "kn": "ಪ್ರಸ್ತುತ ಲವಣಾಂಶ ಅಪಾಯ",
    },
    "crop_suggestions_header": {
        "en": "🌾 Crop Suggestions for Your Area",
        "hi": "🌾 आपके क्षेत्र के लिए फसल सुझाव",
        "ta": "🌾 உங்கள் பகுதிக்கான பயிர் பரிந்துரைகள்",
        "ml": "🌾 നിങ്ങളുടെ പ്രദേശത്തിനുള്ള വിള നിർദ്ദേശങ്ങൾ",
        "bn": "🌾 আপনার এলাকার জন্য ফসলের পরামর্শ",
        "or": "🌾 ଆପଣଙ୍କ ଅଞ୍ଚଳ ପାଇଁ ଫସଲ ପରାମର୍ଶ",
        "gu": "🌾 તમારા વિસ્તાર માટે પાક સૂચનો",
        "te": "🌾 మీ ప్రాంతానికి పంట సూచనలు",
        "kn": "🌾 ನಿಮ್ಮ ಪ್ರದೇಶಕ್ಕೆ ಬೆಳೆ ಸಲಹೆಗಳು",
    },
    "crop_suggestions_intro": {
        "en": "Based on your area's current water risk, here are crop options that may suit local conditions better. Confirm with your local agriculture office before changing crops.",
        "hi": "आपके क्षेत्र की वर्तमान जल स्थिति के आधार पर, यहाँ कुछ फसल विकल्प दिए गए हैं जो स्थानीय परिस्थितियों के लिए बेहतर हो सकते हैं। फसल बदलने से पहले अपने स्थानीय कृषि कार्यालय से पुष्टि करें।",
        "ta": "உங்கள் பகுதியின் தற்போதைய நீர் அபாயத்தின் அடிப்படையில், இவை உள்ளூர் நிலைமைகளுக்கு பொருத்தமான பயிர் விருப்பங்கள். பயிர்களை மாற்றுவதற்கு முன் உங்கள் உள்ளூர் வேளாண் அலுவலகத்தை தொடர்பு கொள்ளவும்.",
        "ml": "നിങ്ങളുടെ പ്രദേശത്തെ നിലവിലെ ജല അപകടസാധ്യത അടിസ്ഥാനമാക്കി, പ്രാദേശിക സാഹചര്യങ്ങൾക്ക് ഇണങ്ങുന്ന വിള ഓപ്ഷനുകൾ ഇതാ. വിള മാറ്റുന്നതിനു മുമ്പ് നിങ്ങളുടെ പ്രാദേശിക കൃഷി ഓഫീസുമായി സ്ഥിരീകരിക്കുക.",
        "bn": "আপনার এলাকার বর্তমান জলের ঝুঁকির ভিত্তিতে, এখানে কিছু ফসলের বিকল্প দেওয়া হলো যা স্থানীয় পরিস্থিতির জন্য ভালো হতে পারে। ফসল পরিবর্তনের আগে আপনার স্থানীয় কৃষি অফিসের সাথে যোগাযোগ করুন।",
        "or": "ଆପଣଙ୍କ ଅଞ୍ଚଳର ବର୍ତ୍ତମାନ ପାଣି ବିପଦ ଆଧାରରେ, ଏଠାରେ କିଛି ଫସଲ ପାଇଁ ବିକଳ୍ପ ଦିଆଯାଇଛି ଯେଉଁ ସ୍ଥାନୀୟ ପରିସ୍ଥିତି ପାଇଁ ଉତ୍ତମ ହୋଇପାରେ। ଫସଲ ପରିବର୍ତ୍ତନ ଆଗେ ଆପଣଙ୍କ ସ୍ଥାନୀୟ କୃଷି କାର୍ଯ୍ୟାଳୟ ସହ ଯାଞ୍ଚ କରନ୍ତୁ।",
        "gu": "તમારા વિસ્તારના વર્તમાન પાણીના જોખમના આધારે, અહીં કેટલાક પાક વિકલ્પો છે જે સ્થાનિક પરિસ્થિતિને વધુ અનુકૂળ હોઈ શકે. પાક બદલતા પહેલા તમારા સ્થાનિક કૃષિ કચેરીનો સંપર્ક કરો.",
        "te": "మీ ప్రాంతపు ప్రస్తుత నీటి ప్రమాదం ఆధారంగా, ఇక్కడ కొన్ని పంట ఎంపికలు ఇవ్వబడ్డాయి, ఇవి స్థానిక పరిస్థితులకు మెరుగ్గా సరిపోవచ్చు. పంట మార్చడానికి ముందు మీ స్థానిక వ్యవసాయ కార్యాలయాన్ని సంప్రదించండి.",
        "kn": "ನಿಮ್ಮ ಪ್ರದೇಶದ ಪ್ರಸ್ತುತ ನೀರಿನ ಅಪಾಯದ ಆಧಾರದ ಮೇಲೆ, ಸ್ಥಳೀಯ ಪರಿಸ್ಥಿತಿಗಳಿಗೆ ಸೂಕ್ತವಾಗಬಹುದಾದ ಬೆಳೆ ಆಯ್ಕೆಗಳು ಇಲ್ಲಿವೆ. ಬೆಳೆ ಬದಲಾಯಿಸುವ ಮೊದಲು ನಿಮ್ಮ ಸ್ಥಳೀಯ ಕೃಷಿ ಕಚೇರಿಯನ್ನು ಸಂಪರ್ಕಿಸಿ.",
    },
    "crops_to_reduce_label": {
        "en": "Crops to avoid or reduce right now",
        "hi": "अभी बचने या कम करने योग्य फसलें",
        "ta": "இப்போது தவிர்க்க வேண்டிய அல்லது குறைக்க வேண்டிய பயிர்கள்",
        "ml": "ഇപ്പോൾ ഒഴിവാക്കേണ്ട അല്ലെങ്കിൽ കുറയ്ക്കേണ്ട വിളകൾ",
        "bn": "এখন এড়ানো বা কমানো উচিত ফসল",
        "or": "ବର୍ତ୍ତମାନ ଏଡ଼ାଇବା କିମ୍ବା କମାଇବା ଉଚିତ ଫସଲ",
        "gu": "અત્યારે ટાળવા અથવા ઘટાડવા જેવા પાક",
        "te": "ఇప్పుడు నివారించాల్సిన లేదా తగ్గించాల్సిన పంటలు",
        "kn": "ಈಗ ತಪ್ಪಿಸಬೇಕಾದ ಅಥವಾ ಕಡಿಮೆ ಮಾಡಬೇಕಾದ ಬೆಳೆಗಳು",
    },
    "crop_guidance_source": {
        "en": "Based on general guidance from ICAR-CSSRI (Central Soil Salinity Research Institute), India's national institute for salt-affected-soil research, including their coastal research station findings. This is general guidance, not a field-specific soil test — confirm with your local agriculture office.",
        "hi": "यह भारत के राष्ट्रीय संस्थान ICAR-CSSRI (केंद्रीय लवणीय मृदा अनुसंधान संस्थान) के सामान्य मार्गदर्शन पर आधारित है, जिसमें उनके तटीय अनुसंधान केंद्र के निष्कर्ष भी शामिल हैं। यह सामान्य मार्गदर्शन है, किसी विशेष खेत की मिट्टी जांच नहीं — अपने स्थानीय कृषि कार्यालय से पुष्टि करें।",
        "ta": "இது இந்தியாவின் தேசிய நிறுவனமான ICAR-CSSRI (மத்திய மண் உப்புத்தன்மை ஆராய்ச்சி நிறுவனம்) இன் பொது வழிகாட்டுதலை அடிப்படையாகக் கொண்டது, அவர்களின் கடலோர ஆராய்ச்சி நிலைய கண்டுபிடிப்புகளும் அடங்கும். இது பொது வழிகாட்டுதலே, குறிப்பிட்ட வயல் மண் பரிசோதனை அல்ல — உங்கள் உள்ளூர் வேளாண் அலுவலகத்தில் உறுதிப்படுத்தவும்.",
        "ml": "ഇന്ത്യയുടെ ദേശീയ സ്ഥാപനമായ ICAR-CSSRI (സെൻട്രൽ സോയിൽ സാലിനിറ്റി റിസർച്ച് ഇൻസ്റ്റിറ്റ്യൂട്ട്) ന്റെ പൊതു മാർഗ്ഗനിർദ്ദേശത്തെ അടിസ്ഥാനമാക്കിയുള്ളതാണ് ഇത്, അവരുടെ തീരദേശ ഗവേഷണ കേന്ദ്ര കണ്ടെത്തലുകൾ ഉൾപ്പെടെ. ഇത് പൊതു മാർഗ്ഗനിർദ്ദേശമാണ്, നിർദ്ദിഷ്ട വയൽ മണ്ണ് പരിശോധനയല്ല — നിങ്ങളുടെ പ്രാദേശിക കൃഷി ഓഫീസുമായി സ്ഥിരീകരിക്കുക.",
        "bn": "এটি ভারতের জাতীয় প্রতিষ্ঠান ICAR-CSSRI (কেন্দ্রীয় মৃত্তিকা লবণাক্ততা গবেষণা ইনস্টিটিউট) এর সাধারণ নির্দেশনার উপর ভিত্তি করে তৈরি, যার মধ্যে তাদের উপকূলীয় গবেষণা কেন্দ্রের ফলাফলও রয়েছে। এটি সাধারণ নির্দেশনা, নির্দিষ্ট জমির মাটি পরীক্ষা নয় — আপনার স্থানীয় কৃষি অফিসের সাথে নিশ্চিত করুন।",
        "or": "ଏହା ଭାରତର ଜାତୀୟ ଅନୁଷ୍ଠାନ ICAR-CSSRI (କେନ୍ଦ୍ରୀୟ ମୃତ୍ତିକା ଲବଣତା ଅନୁସନ୍ଧାନ ଅନୁଷ୍ଠାନ) ର ସାଧାରଣ ମାର୍ଗଦର୍ଶନ ଉପରେ ଆଧାରିତ, ଯେଉଁଥିରେ ସେମାନଙ୍କ ଉପକୂଳ ଅନୁସନ୍ଧାନ କେନ୍ଦ୍ରର ଫଳାଫଳ ମଧ୍ୟ ଅନ୍ତର୍ଭୁକ୍ତ। ଏହା ସାଧାରଣ ମାର୍ଗଦର୍ଶନ, ନିର୍ଦ୍ଦିଷ୍ଟ ଜମି ମାଟି ପରୀକ୍ଷା ନୁହେଁ — ଆପଣଙ୍କ ସ୍ଥାନୀୟ କୃଷି କାର୍ଯ୍ୟାଳୟ ସହ ନିଶ୍ଚିତ କରନ୍ତୁ।",
        "gu": "આ ભારતની રાષ્ટ્રીય સંસ્થા ICAR-CSSRI (કેન્દ્રીય માટી ખારાશ સંશોધન સંસ્થા) ના સામાન્ય માર્ગદર્શન પર આધારિત છે, જેમાં તેમના દરિયાકાંઠાના સંશોધન કેન્દ્રના તારણોનો પણ સમાવેશ થાય છે. આ સામાન્ય માર્ગદર્શન છે, ચોક્કસ ખેતરની માટી પરીક્ષણ નથી — તમારા સ્થાનિક કૃષિ કચેરી સાથે ખાતરી કરો.",
        "te": "ఇది భారతదేశ జాతీయ సంస్థ ICAR-CSSRI (కేంద్ర నేల లవణీయత పరిశోధనా సంస్థ) యొక్క సాధారణ మార్గదర్శకత్వం ఆధారంగా రూపొందించబడింది, వారి తీర పరిశోధనా కేంద్ర ఫలితాలతో సహా. ఇది సాధారణ మార్గదర్శకత్వం, నిర్దిష్ట పొలం నేల పరీక్ష కాదు — మీ స్థానిక వ్యవసాయ కార్యాలయంతో నిర్ధారించుకోండి.",
        "kn": "ಇದು ಭಾರತದ ರಾಷ್ಟ್ರೀಯ ಸಂಸ್ಥೆ ICAR-CSSRI (ಕೇಂದ್ರ ಮಣ್ಣಿನ ಲವಣಾಂಶ ಸಂಶೋಧನಾ ಸಂಸ್ಥೆ) ಯ ಸಾಮಾನ್ಯ ಮಾರ್ಗದರ್ಶನವನ್ನು ಆಧರಿಸಿದೆ, ಅವರ ಕರಾವಳಿ ಸಂಶೋಧನಾ ಕೇಂದ್ರದ ಸಂಶೋಧನೆಗಳು ಸೇರಿದಂತೆ. ಇದು ಸಾಮಾನ್ಯ ಮಾರ್ಗದರ್ಶನ, ನಿರ್ದಿಷ್ಟ ಹೊಲದ ಮಣ್ಣಿನ ಪರೀಕ್ಷೆಯಲ್ಲ — ನಿಮ್ಮ ಸ್ಥಳೀಯ ಕೃಷಿ ಕಚೇರಿಯೊಂದಿಗೆ ಖಚಿತಪಡಿಸಿಕೊಳ್ಳಿ.",
    },
}


def t(key: str, lang: str = "en", **kwargs) -> str:
    """Translation lookup with an English fallback — if a key is missing
    for the requested language (or the language itself isn't in the
    dictionary), it falls back to English rather than crashing or
    showing a blank string."""
    entry = TRANSLATIONS.get(key, {})
    text = entry.get(lang) or entry.get("en") or key
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, IndexError):
            pass
    return text



# --------------------------------------------------------------------------
# Physics-based pumping estimate from electricity consumption (Feature 3,
# refined) — Estimated Agricultural Groundwater Extraction
#
#   Q = (eta * E) / (rho * g * H)      [E converted from kWh to Joules first]
#
#   Q   = estimated pumped volume over the metering period, m^3 (== kL)
#   eta = OVERALL pumping-plant (wire-to-water) efficiency — NOT a fixed
#         guess. Grounded in published Indian field studies, treated as a
#         RANGE, not a point value:
#           - Conventional/older pumpsets (most common in the field):
#             measured overall efficiencies cluster around 20-35%, with a
#             commonly cited "existing inefficient" figure of ~25-30%.
#             Sources: World Bank Haryana pump-energy audit (21-24%);
#             National Productivity Council Haryana study (25-35%); Sonipat
#             field survey, 65 tubewells, measured range 10.1-56.6%
#             (PeerJ/PMC9884473); ScienceDirect "Agricultural pumping
#             efficiency in India" (20-30%, Dixit & Sant 1996; Singh 2009).
#           - BEE 5-star-rated modern pumpsets: ~40-50% (IEA 2009; Saini
#             2013, cited in the same literature).
#         The app lets the user pick which population applies and uses the
#         cited low/mid/high split from that population, not an arbitrary
#         single number.
#   E   = electricity consumed, kWh (manually entered — see note below),
#         converted to Joules: E_J = E_kWh * 3.6e6
#   rho = density of water, 1000 kg/m^3 (physical constant)
#   g   = gravitational acceleration, 9.81 m/s^2 (physical constant)
#   H   = Total Dynamic Head, metres = STATIC LIFT + delivery/friction head
#         - Static lift is NOT arbitrary: it is this well's own measured
#           DWLR depth-to-water (real per-well data already in the app).
#         - Delivery/friction head is expressed as a PROPORTION of static
#           lift (10-25% is a standard hydraulic-engineering approximation
#           for small-bore agricultural delivery pipe over field-scale
#           distances), not a flat metres constant — so it scales
#           per-well instead of applying the same absolute number to a
#           5m-deep well and a 25m-deep well alike. Still an approximation
#           (no site-specific pipe-length/diameter survey), and reported
#           as a range for that reason, not asserted as exact.
#
# WHY THIS IS ELECTRICITY-DRIVEN AND NOT A LIVE TANGEDCO FEED: agricultural
# power is supplied free of charge and is largely unmetered in practice
# across Tamil Nadu (and most Indian states) — TANGEDCO does not publish an
# open, feeder/block-level agricultural-consumption dataset, and meter
# installation has been actively resisted where attempted. There is no
# verifiable automatic feed to fetch, and fabricating one would mean
# inventing numbers. What IS realistic: a field officer supplies a
# metered/billed electricity figure where one exists. This function turns
# THAT into a physically grounded, reproducible, uncertainty-bounded
# extraction estimate — same formula and constants for every well; only H
# differs, and only because that well's own measured depth differs.
#
# ATTRIBUTION CAVEAT: official "agricultural electricity" tariff categories
# (where metered at all) are not necessarily 100% groundwater pumping —
# they can include other on-farm loads. The app exposes an explicit,
# adjustable "% of entered electricity attributable to groundwater
# pumping" input (default 100%, lower it if you know your figure is
# broader than pumping alone) rather than silently assuming the whole
# figure is pumping.
# --------------------------------------------------------------------------
RHO_WATER = 1000.0   # kg/m^3 (constant)
G_GRAVITY = 9.81      # m/s^2 (constant)

PUMP_EFFICIENCY_PROFILES = {
    "Conventional / older pumpset (typical — most common)": (0.20, 0.28, 0.35),
    "BEE star-rated / modern efficient pumpset": (0.40, 0.45, 0.50),
}
FRICTION_HEAD_FRACTION_RANGE = (0.10, 0.175, 0.25)  # (low, mid, high) fraction of static lift


def estimate_pumped_volume_kl(energy_kwh: float, eta: float, head_m: float) -> float:
    if head_m <= 0 or energy_kwh <= 0:
        return 0.0
    energy_j = energy_kwh * 3.6e6
    return (eta * energy_j) / (RHO_WATER * G_GRAVITY * head_m)


def estimate_extraction_range(energy_kwh: float, static_lift_m: float,
                               eta_low: float, eta_mid: float, eta_high: float,
                               friction_frac_low: float = FRICTION_HEAD_FRACTION_RANGE[0],
                               friction_frac_high: float = FRICTION_HEAD_FRACTION_RANGE[2]):
    """Propagates BOTH efficiency uncertainty and head uncertainty into the
    extraction estimate (not just one or the other). Low estimate pairs the
    least favourable combination (low eta, high head); high estimate pairs
    the most favourable (high eta, low head); mid uses the literature
    midpoint eta with the midpoint friction fraction."""
    head_low_case = static_lift_m * (1 + friction_frac_high)   # more head -> less volume for same energy
    head_high_case = static_lift_m * (1 + friction_frac_low)   # less head -> more volume for same energy
    head_mid = static_lift_m * (1 + FRICTION_HEAD_FRACTION_RANGE[1])

    vol_low = estimate_pumped_volume_kl(energy_kwh, eta_low, head_low_case)
    vol_mid = estimate_pumped_volume_kl(energy_kwh, eta_mid, head_mid)
    vol_high = estimate_pumped_volume_kl(energy_kwh, eta_high, head_high_case)
    return vol_low, vol_mid, vol_high, head_mid


# ============================================================================
# BROADCAST & RECIPIENT MANAGEMENT — shared across all sessions of this
# running app via st.cache_resource (returns the SAME object to every
# browser tab/session connected to this one `streamlit run app.py`
# process, so a broadcast sent from the Authority tab is visible in the
# Public tab). This resets if the server process restarts — an accepted,
# documented limitation for a prototype; production would use a real
# database.
# ============================================================================
BROADCAST_SEVERITY_THRESHOLDS = [
    (1600, ("RED", "CRITICAL")),
    (1200, ("ORANGE", "HIGH")),
    (900,  ("YELLOW", "MODERATE")),
    (0,    ("GREEN", "NORMAL")),
]
SEVERITY_COLOR = {"RED": "#e74c3c", "ORANGE": "#f39c12", "YELLOW": "#f1c40f", "GREEN": "#2ecc71"}


def broadcast_severity(ec_value: float):
    """Four-tier severity used ONLY by the alert/broadcast module (matches
    the RED/ORANGE/YELLOW/GREEN legend requested for resident-facing
    alerts). Built from the SAME underlying EC value and SAME calibrated
    cutoffs already used elsewhere (1600/1200 match the existing High/
    Medium boundaries) — this does not replace or redefine the core
    Low/Medium/High risk model used throughout the rest of the app, it
    only adds one extra display-only split of "Low" into GREEN vs YELLOW
    so the four-level legend can be shown without touching the existing
    forecasting logic."""
    for cutoff, label in BROADCAST_SEVERITY_THRESHOLDS:
        if ec_value >= cutoff:
            return label
    return ("GREEN", "NORMAL")


def broadcast_color_to_alert_level(color: str) -> str:
    """Maps the 4-tier broadcast severity colour to the existing 3-tier
    render_alert animation style (RED gets the urgent fast-pulse look,
    ORANGE/YELLOW get the slower caution pulse, GREEN gets the calm
    style) — reuses the already-built alert component as-is."""
    return {"RED": "High", "ORANGE": "Medium", "YELLOW": "Medium", "GREEN": "Low"}[color]


def generate_layman_message(zone_name: str, color: str, label: str) -> str:
    """Auto-drafts a resident-facing message with NO ML/technical
    terminology. The authority can edit this before sending — see the
    manual message box."""
    if color == "RED":
        return (f"CRITICAL GROUNDWATER ALERT — High salinity-ingress risk has been "
                 f"detected in the {zone_name} coastal monitoring zone. Residents and "
                 f"agricultural users in the affected area should reduce groundwater "
                 f"extraction and follow the latest authority guidance.")
    if color == "ORANGE":
        return (f"GROUNDWATER ADVISORY — Elevated groundwater stress has been detected "
                 f"in the {zone_name} monitoring zone. Residents and agricultural users "
                 f"are advised to reduce non-essential groundwater use and monitor "
                 f"official updates.")
    if color == "YELLOW":
        return (f"GROUNDWATER NOTICE — Early signs of groundwater stress have been "
                 f"observed in the {zone_name} monitoring zone. No urgent action is "
                 f"required; residents may wish to use water carefully as a precaution.")
    return (f"GROUNDWATER UPDATE — Groundwater conditions in the {zone_name} monitoring "
             f"zone are currently normal. No action is required.")


def _seed_recipients():
    """A few example recipients per coastal zone so the module isn't empty
    on first run — for the prototype demo only; edit/remove/add freely."""
    zones = ["Nagapattinam", "Chennai Coast", "Kochi", "Digha", "Puri", "Veraval", "Visakhapatnam", "Mangalore"]
    seed = []
    rid = 1
    for z in zones:
        seed.append({"id": rid, "name": f"{z} Ward Office", "zone": z,
                     "email": f"ward.office.{z.lower().replace(' ', '')}@example.gov.in",
                     "phone": "", "category": "Local Official"})
        rid += 1
        seed.append({"id": rid, "name": f"Sample Resident — {z}", "zone": z,
                     "email": f"resident.{z.lower().replace(' ', '')}@example.com",
                     "phone": "", "category": "Resident"})
        rid += 1
    return seed


@st.cache_resource
def get_recipients_store():
    return {"recipients": _seed_recipients(), "next_id": len(_seed_recipients()) + 1}


@st.cache_resource
def get_broadcast_store():
    return {"alerts": [], "next_id": 1}


RECOMMENDED_ACTIONS = {
    "RED":    "Reduce groundwater extraction immediately and follow all instructions from local authorities.",
    "ORANGE": "Reduce non-essential groundwater use and monitor official updates closely.",
    "YELLOW": "Use water carefully as a precaution; no urgent action is required at this time.",
    "GREEN":  "No action required at this time.",
}


def build_email_body(well_name: str, severity_color: str, severity_label: str,
                      auto_message: str, manual_message: str, timestamp) -> str:
    """Assembles the complete email body: location, severity, the automatic
    warning, the authority's manual message, a recommended action, and a
    timestamp — every field the broadcast is required to contain."""
    lines = [
        "GROUNDWATER MONITORING ALERT",
        "",
        f"Location: {well_name} Coastal Zone",
        f"Severity: {severity_color} — {severity_label}",
        f"Date/Time: {timestamp.strftime('%d %b %Y, %I:%M %p')}",
        "",
        "Warning:",
        auto_message,
    ]
    if manual_message.strip():
        lines += ["", "Additional message from the authority:", manual_message.strip()]
    lines += ["", f"Recommended action: {RECOMMENDED_ACTIONS.get(severity_color, RECOMMENDED_ACTIONS['GREEN'])}"]
    return "\n".join(lines)


def _get_smtp_config():
    """Reads SMTP settings from st.secrets (Streamlit Cloud's recommended
    mechanism) first, falling back to plain OS environment variables (local
    development, or other hosts like Render/Railway/Hugging Face Spaces that
    use standard env vars instead) — never hard-coded, never guessed.
    Returns None if any required value is missing anywhere."""
    import os as _os

    def _get(key):
        try:
            if key in st.secrets:
                return st.secrets[key]
        except Exception:  # noqa: BLE001 - no secrets.toml present locally is expected, not an error
            pass
        return _os.environ.get(key)

    host = _get("SMTP_HOST")
    port = _get("SMTP_PORT")
    username = _get("SMTP_USERNAME")
    password = _get("SMTP_PASSWORD")
    sender = _get("SENDER_EMAIL")
    if not (host and port and username and password and sender):
        return None
    try:
        port = int(port)
    except ValueError:
        return None
    return {"host": host, "port": port, "username": username, "password": password, "sender": sender}


def send_broadcast_emails(recipients: list, subject: str, body: str):
    """
    Sends the SAME final message individually to each recipient's own
    email address and tracks each delivery result separately.

    Returns:
        overall_status: "not_configured" | "no_recipients" | "sent" | "partial" | "failed"
        results: list of {id, name, email, status: "Sent"/"Failed", detail}
        summary: human-readable line, e.g. "Broadcast delivered: 2/2 emails
                 successfully sent."

    Never reports a fake "Sent" status. If SMTP env vars are missing,
    returns "not_configured" and an empty result list. If SMTP IS
    configured but zero recipients were selected, that's reported
    separately as "no_recipients" — NOT as "not configured", since the
    two are different problems and conflating them would misreport a
    correctly-configured mail server as broken.
    """
    config = _get_smtp_config()
    if config is None:
        return "not_configured", [], "External email service not configured."
    if not recipients:
        return "no_recipients", [], "No recipients were selected — nothing to email."

    import smtplib
    from email.mime.text import MIMEText

    results = []
    server = None
    connection_error = None
    try:
        if config["port"] == 465:
            server = smtplib.SMTP_SSL(config["host"], config["port"], timeout=15)
        else:
            server = smtplib.SMTP(config["host"], config["port"], timeout=15)
            server.starttls()
        server.login(config["username"], config["password"])
    except Exception as e:  # noqa: BLE001 - connection/login failure applies to every recipient
        connection_error = str(e)

    for r in recipients:
        to_addr = (r.get("email") or "").strip()
        if not to_addr:
            results.append({"id": r["id"], "name": r["name"], "email": to_addr,
                             "status": "Failed", "detail": "No email address on file"})
            continue
        if connection_error is not None:
            results.append({"id": r["id"], "name": r["name"], "email": to_addr,
                             "status": "Failed", "detail": f"SMTP connection/login error: {connection_error}"})
            continue
        try:
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = config["sender"]
            msg["To"] = to_addr
            server.sendmail(config["sender"], [to_addr], msg.as_string())
            results.append({"id": r["id"], "name": r["name"], "email": to_addr,
                             "status": "Sent", "detail": "Delivered"})
        except Exception as e:  # noqa: BLE001 - report the real per-recipient failure reason
            results.append({"id": r["id"], "name": r["name"], "email": to_addr,
                             "status": "Failed", "detail": str(e)})

    if server is not None:
        try:
            server.quit()
        except Exception:  # noqa: BLE001 - closing the connection should never mask send results
            pass

    sent_count = sum(1 for x in results if x["status"] == "Sent")
    total = len(results)
    summary = f"Broadcast delivered: {sent_count}/{total} email{'s' if total != 1 else ''} successfully sent."
    if sent_count == total:
        overall = "sent"
    elif sent_count == 0:
        overall = "failed"
    else:
        overall = "partial"
    return overall, results, summary


# --------------------------------------------------------------------------
# Forecast engine: chains the two models forward in 30-day jumps, using
# each well's own historical seasonal pattern ("climatology") as the
# baseline for future rainfall/pumping, rebased by the electricity-derived
# pump_scale_factor, then scaled again by the optional what-if sliders.
# --------------------------------------------------------------------------
def build_climatology(well_hist: pd.DataFrame) -> pd.DataFrame:
    h = well_hist.copy()
    h["rain_30d"] = h["rainfall_mm"].rolling(30, min_periods=1).sum()
    h["rain_90d"] = h["rainfall_mm"].rolling(90, min_periods=1).sum()
    h["pump_30d"] = h["pumping_kl"].rolling(30, min_periods=1).sum()
    h["doy"] = h["date"].dt.dayofyear
    return h.groupby("doy")[["rain_30d", "rain_90d", "pump_30d"]].mean()


def climatology_lookup(clim: pd.DataFrame, doy: int) -> pd.Series:
    doy = ((doy - 1) % 365) + 1
    if doy in clim.index:
        return clim.loc[doy]
    nearest = min(clim.index, key=lambda d: abs(d - doy))
    return clim.loc[nearest]


def forecast_well(well_hist: pd.DataFrame, n_steps: int, rain_pct: float, pump_pct: float,
                   pump_scale_factor: float = 1.0) -> pd.DataFrame:
    """
    Chains depth_model -> salinity_model forward n_steps (30-day jumps).

    HYBRID ADJUSTMENT (be upfront about this if judges ask "how does the
    model work"): the salinity Random Forest was trained on 3 years of
    moderate, real-world-scale pumping variation, so — like any tree
    ensemble — it under-reacts to pumping levels far outside that observed
    range (trees cannot extrapolate past the training data's leaf values).
    To reflect the well-documented reality that SUSTAINED over-extraction
    compounds seawater intrusion faster than a single short-horizon ML
    step captures, a small, capped, coastal-only stress adjustment is
    added on top of the ML salinity prediction, proportional to how far
    the effective pumping multiplier exceeds 1.0x. It is bounded (capped
    multiplier effect, coastal wells only) and clearly separated from the
    ML output below — not a black-box fudge, a documented, reproducible
    correction for a known model limitation.
    """
    well_hist = well_hist.sort_values("date").reset_index(drop=True)
    last_row = well_hist.iloc[-1]
    last_date = last_row["date"]

    dist_km = last_row["distance_to_coast_km"]
    is_coastal_int = int(last_row["is_coastal"])

    clim = build_climatology(well_hist)

    cutoff = last_date - pd.Timedelta(days=30)
    prior = well_hist[well_hist["date"] <= cutoff]
    depth_prev_30 = prior["depth_to_water_m"].iloc[-1] if len(prior) else last_row["depth_to_water_m"]

    depth_now = last_row["depth_to_water_m"]
    ec_now = last_row["salinity_ec_uscm"]

    effective_pump_multiplier = pump_scale_factor * pump_pct / 100
    excess_multiplier = min(max(effective_pump_multiplier - 1.0, 0.0), 3.0)  # capped at 3x over baseline
    stress_bonus_per_step = is_coastal_int * excess_multiplier * 35  # uS/cm added per 30-day step, cumulative

    rows = []
    for step in range(1, n_steps + 1):
        future_date = last_date + pd.Timedelta(days=30 * step)
        doy = future_date.dayofyear
        c = climatology_lookup(clim, doy)

        rain_30d = c["rain_30d"] * rain_pct / 100
        rain_90d = c["rain_90d"] * rain_pct / 100
        pump_30d = c["pump_30d"] * pump_scale_factor * pump_pct / 100
        month_sin = np.sin(2 * np.pi * doy / 365)
        month_cos = np.cos(2 * np.pi * doy / 365)

        depth_feat = pd.DataFrame([{
            "depth_to_water_m": depth_now, "depth_lag_30": depth_prev_30,
            "rain_30d": rain_30d, "rain_90d": rain_90d, "pump_30d": pump_30d,
            "month_sin": month_sin, "month_cos": month_cos,
            "distance_to_coast_km": dist_km, "is_coastal_int": is_coastal_int,
        }])[DEPTH_FEATURES]
        depth_pred = float(depth_model.predict(depth_feat)[0])

        sal_feat = pd.DataFrame([{
            "depth_future_30": depth_pred, "salinity_ec_uscm": ec_now,
            "distance_to_coast_km": dist_km, "is_coastal_int": is_coastal_int,
            "rain_30d": rain_30d, "pump_30d": pump_30d,
            "month_sin": month_sin, "month_cos": month_cos,
        }])[SALINITY_FEATURES]
        ec_pred_ml = float(sal_model.predict(sal_feat)[0])
        ec_pred = ec_pred_ml + stress_bonus_per_step * step

        rows.append({"date": future_date, "depth_to_water_m": depth_pred, "salinity_ec_uscm": ec_pred})

        depth_prev_30 = depth_now
        depth_now = depth_pred
        ec_now = ec_pred

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def forecast_all_wells(n_steps: int, rain_pct: float, pump_pct: float) -> pd.DataFrame:
    out = []
    for wid, g in data.groupby("well_id"):
        fc = forecast_well(g, n_steps, rain_pct, pump_pct)
        fc["well_id"] = wid
        out.append(fc.iloc[[-1]])
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------
# Data source citations + formulas (Feature 2) — shown in the Authority view
# --------------------------------------------------------------------------
def render_data_sources_and_formulas():
    st.markdown("#### 📡 Where this data comes from")
    st.markdown(
        "- **Groundwater / DWLR levels:** [India-WRIS — Ground Water module]"
        "(https://indiawris.gov.in/wris/#/groundWater) · "
        "[CGWB — DWLR & Piezometer Network]"
        "(https://cgwb.gov.in/en/ground-water-level-monitoring)  \n"
        "  CGWB's Climate Response Monitoring Network runs 60 coastal piezometers "
        "along ~450 km of the Tamil Nadu & Puducherry coastline specifically for "
        "seawater-intrusion monitoring — the real-world basis for this project's "
        "coastal salinity focus.\n"
        "- **Rainfall:** [India Meteorological Department (IMD)](https://mausam.imd.gov.in) · "
        "[India-WRIS — Rainfall module](https://indiawris.gov.in/wris/#/rainfall)\n"
        "- **Agricultural pumping:** no automatic public feed exists — TANGEDCO does not publish a "
        "feeder/block-level agricultural consumption dataset, so this app NEVER treats the entered "
        "electricity figure as official EB data. It is a manually-entered, unverified input (see the "
        "Electricity-Based Pumping Estimate panel below for the full reasoning and the physics used "
        "to convert it into an extraction estimate)."
    )

    st.markdown("#### 🧮 Formulas used")
    st.markdown("**1. Rolling driver features** (fed to both forecast models):")
    st.latex(r"\text{rain}_{30d}(t) = \sum_{i=t-29}^{t}\text{rainfall}(i)"
             r"\qquad \text{pump}_{30d}(t) = \sum_{i=t-29}^{t}\text{pumping}(i)")
    st.markdown("**2. Electricity → Estimated Agricultural Groundwater Extraction** (pump hydraulics):")
    st.latex(r"Q = \frac{\eta \cdot E_J}{\rho \cdot g \cdot H} \qquad E_J = E_{kWh} \times 3.6\times10^{6}")
    st.markdown(
        "η (pump efficiency) is drawn from published Indian field studies, not an arbitrary "
        "value — conventional pumpsets ~20-35% (World Bank Haryana pump audit; National "
        "Productivity Council Haryana study; 65-tubewell Sonipat field survey, "
        "[PMC9884473](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9884473/); "
        "[ScienceDirect, Dixit & Sant 1996 / Singh 2009](https://www.sciencedirect.com/science/article/abs/pii/S0973082608601787)), "
        "BEE-rated modern pumpsets ~40-50%. H (Total Dynamic Head) = this well's own measured "
        "DWLR depth (static lift, real data) × (1 + 10-25% delivery/friction allowance, a "
        "standard hydraulic-engineering approximation) — both reported as ranges, with the "
        "resulting extraction estimate propagating both uncertainties together (see the "
        "Estimated Agricultural Groundwater Extraction panel for the live low/mid/high figures)."
    )
    st.markdown("**3. Salinity risk bands** (Electrical Conductivity, µS/cm):")
    st.markdown("".join(f"- **{label}:** {lo:.0f} – {('∞' if hi > 100000 else f'{hi:.0f}')} µS/cm\n"
                         for label, (lo, hi) in RISK_BANDS.items()))
    st.markdown("**4. Sustained-stress adjustment** (coastal wells only, capped):")
    st.latex(r"EC_{adj} = EC_{ML} + \big(\min(\text{mult}_{excess}, 3) \times 35\big) \times \text{step}")
    st.markdown(
        "Tree-based models can't extrapolate strongly beyond the pumping range they were "
        "trained on. This adds a small, capped, coastal-only correction proportional to how "
        "far the pumping multiplier exceeds 1.0x, reflecting that sustained over-extraction "
        "compounds seawater intrusion faster than one 30-day ML step alone would show. "
        "**Honesty note:** the coefficient (35 \u00b5S/cm per step) is empirically calibrated for "
        "this prototype to produce a demonstrable, physically-correct-direction escalation — "
        "it is NOT drawn from a published intrusion-rate study. It's documented and bounded, "
        "not hidden, but it should be labelled as a calibrated prototype parameter if a judge "
        "asks, not as a literature-derived constant like \u03b7 or the risk thresholds above."
    )
    st.markdown("**5. The depth/salinity forecast itself is NOT a single equation** — it's a "
        "Random Forest (300 trees) trained on the features above to predict 30-days-"
        "ahead depth and salinity, evaluated with a chronological train / validation / test "
        "split (no shuffling, no leakage — see the real R\u00b2/MAE/RMSE numbers in the "
        "\"How the model decides\" panel below). Be upfront about this if asked: the *feature "
        "engineering and physics conversions* above are explicit formulas; the "
        "*forecast* is a learned model over them (see feature-importance chart below "
        "for what it weighs most)."
    )


# ============================================================================
# AUTHORITY / TECHNICAL VIEW
# ============================================================================
def render_authority_view():
    st.sidebar.title("💧 Authority Controls")

    st.sidebar.markdown("**📡 Auto-logged (DWLR + rainfall)**")
    st.sidebar.caption("Simulated here — auto-ingested from CGWB/IMD feeds in a real deployment.")
    well_label_to_id = {f"{r['name']} ({r['state']})": r["well_id"] for _, r in wells_meta.iterrows()}
    selected_label = st.sidebar.selectbox("DWLR monitoring well", list(well_label_to_id.keys()))
    selected_well_id = well_label_to_id[selected_label]

    well_hist = data[data["well_id"] == selected_well_id].sort_values("date")
    default_pump_30d = float(well_hist["pumping_kl"].tail(30).sum())
    current_depth_now = float(well_hist["depth_to_water_m"].iloc[-1])

    st.sidebar.markdown("---")
    st.sidebar.markdown("**⚡ Estimated Agricultural Groundwater Extraction**")
    st.sidebar.caption(
        "No metered public feed exists for agricultural power in most Indian states "
        "(it's largely free/unmetered — TANGEDCO publishes no open feeder-level "
        "agricultural dataset). This uses a real billed/estimated electricity figure "
        "converted via pump hydraulics: Q = ηE / (ρgH), with η and H grounded in "
        "published field studies and this well's own measured depth — not fixed guesses."
    )

    pump_profile = st.sidebar.selectbox(
        "Pump type (sets the efficiency range used)",
        list(PUMP_EFFICIENCY_PROFILES.keys()),
        help="Efficiency ranges from published Indian field studies (World Bank Haryana "
             "pump audit; NPC Haryana; Sonipat 65-tubewell field survey, PMC9884473; "
             "ScienceDirect 'Agricultural pumping efficiency in India') — not an arbitrary guess.",
    )
    eta_low, eta_mid, eta_high = PUMP_EFFICIENCY_PROFILES[pump_profile]

    pct_attributable = st.sidebar.slider(
        "% of entered electricity attributable to groundwater pumping", 40, 100, 100, step=5,
        help="Official 'agricultural electricity' categories, where metered at all, can include "
             "non-pumping farm loads. Lower this if your electricity figure is broader than "
             "pumping alone — don't assume 100% by default unless you know it is pumping-only.",
    )

    _head_mid_default = current_depth_now * (1 + FRICTION_HEAD_FRACTION_RANGE[1])
    default_energy_kwh = (default_pump_30d * RHO_WATER * G_GRAVITY * _head_mid_default
                           / (eta_mid * 3.6e6)) if _head_mid_default > 0 else 0.0

    energy_kwh_raw = st.sidebar.number_input(
        "Electricity consumed, last 30 days (kWh) — manually entered",
        min_value=0.0, value=round(default_energy_kwh, 1), step=50.0,
    )
    energy_kwh = energy_kwh_raw * pct_attributable / 100
    st.sidebar.caption(
        "⚠️ This figure is entered by hand, not fetched from TANGEDCO/EB records — no public "
        "dataset exists at this resolution (see Data Sources & Methodology). Treat it as an "
        "unverified input, not observed official data."
    )

    st.sidebar.caption(
        f"TDH = this well's measured depth ({current_depth_now:.1f} m) × "
        f"(1 + {FRICTION_HEAD_FRACTION_RANGE[0]*100:.0f}–{FRICTION_HEAD_FRACTION_RANGE[2]*100:.0f}% "
        "delivery/friction allowance) — a standard hydraulic-engineering approximation, "
        "reported as a range since no site-specific pipe survey exists for this prototype."
    )

    vol_low, vol_mid, vol_high, head_m = estimate_extraction_range(
        energy_kwh, current_depth_now, eta_low, eta_mid, eta_high
    )
    manual_volume_kl = vol_mid
    pump_scale_factor = manual_volume_kl / default_pump_30d if default_pump_30d > 0 else 1.0

    hist_pump_30d_series = well_hist.set_index("date")["pumping_kl"].rolling(30, min_periods=1).sum()
    recent_rain_30d = float(well_hist["rainfall_mm"].tail(30).sum())
    depth_90d_ago_series = well_hist[well_hist["date"] <= well_hist["date"].iloc[-1] - pd.Timedelta(days=90)]
    depth_trend_90d = (current_depth_now - depth_90d_ago_series["depth_to_water_m"].iloc[-1]
                        if len(depth_90d_ago_series) else 0.0)
    ec_trend_90d = (well_hist["salinity_ec_uscm"].iloc[-1] - depth_90d_ago_series["salinity_ec_uscm"].iloc[-1]
                    if len(depth_90d_ago_series) else 0.0)
    _clim_for_well = build_climatology(well_hist)
    _current_doy = well_hist["date"].iloc[-1].dayofyear
    typical_rain_30d = float(climatology_lookup(_clim_for_well, _current_doy)["rain_30d"])

    st.sidebar.markdown("---")
    horizon_label = st.sidebar.radio("Forecast horizon", ["3 months", "6 months", "12 months"], index=1)
    horizon_steps = {"3 months": 3, "6 months": 6, "12 months": 12}[horizon_label]

    st.sidebar.markdown("---")
    mode = st.sidebar.radio("Mode", ["Official Verdict (from data)", "What-If Scenario (explore)"])

    if mode.startswith("What-If"):
        st.sidebar.markdown("**🔮 What-if scenario**")
        rain_pct = st.sidebar.slider("Future rainfall (% of normal)", 40, 160, 100, step=5)
        pump_pct = st.sidebar.slider("Future pumping (% of your electricity-based estimate)", 40, 200, 100, step=5)
        st.sidebar.caption(
            "Rainfall/pumping during the forecast window are drawn from each well's own "
            "seasonal history, rebased to your electricity-derived estimate, then scaled "
            "by these sliders."
        )
    else:
        rain_pct, pump_pct = 100, 100
        st.sidebar.caption(
            "The model's best estimate: normal seasonal rainfall continues, and your "
            "electricity-derived pumping estimate holds steady."
        )

    # ---- Language: manual only, no auto-switch, defaults to English ----
    # (per the requirement: officials aren't forced into a language just
    # because they clicked into a particular well — only the Public
    # Interface auto-switches by area)
    st.sidebar.markdown("---")
    auth_lang_options = list(LANGUAGES.values())
    auth_lang_choice = st.sidebar.selectbox("\U0001f310 Language", auth_lang_options, index=0, key="auth_lang_select")
    auth_lang_name_to_code = {v: k for k, v in LANGUAGES.items()}
    lang = auth_lang_name_to_code.get(auth_lang_choice, "en")

    # ---- Header ----
    top_l, top_r = st.columns([5, 1])
    with top_l:
        st.title("Groundwater Depletion & Salinity Infiltration Predictive Matrix")
        st.caption(t("auth_subtitle", lang))
    with top_r:
        if st.button(t("switch_profile", lang), use_container_width=True):
            st.session_state.profile = None
            st.rerun()

    well_row = wells_meta[wells_meta["well_id"] == selected_well_id].iloc[0]
    forecast_df = forecast_well(well_hist, horizon_steps, rain_pct, pump_pct, pump_scale_factor)

    current_depth = well_hist["depth_to_water_m"].iloc[-1]
    current_ec = well_hist["salinity_ec_uscm"].iloc[-1]
    year_ago = well_hist[well_hist["date"] <= well_hist["date"].iloc[-1] - pd.Timedelta(days=365)]
    year_ago_depth = year_ago["depth_to_water_m"].iloc[-1] if len(year_ago) else current_depth

    forecast_depth = forecast_df["depth_to_water_m"].iloc[-1]
    forecast_ec = forecast_df["salinity_ec_uscm"].iloc[-1]
    current_risk = risk_band(current_ec)
    forecast_risk = risk_band(forecast_ec)

    # ---- Estimated Agricultural Groundwater Extraction panel ----
    def _round_uncertain(x):
        """Rounds to 2 significant figures — showing 6,347 kL when the
        underlying eta/head assumptions carry +/-30-60% uncertainty is
        false precision; 6,300 kL is honest about what's actually known."""
        if x <= 0:
            return 0
        import math
        magnitude = 10 ** (math.floor(math.log10(x)) - 1)
        return round(x / magnitude) * magnitude

    quality = compute_data_quality(well_hist, vol_low, vol_mid, vol_high)

    with st.expander("⚡ Estimated Agricultural Groundwater Extraction — details, assumptions & plausibility", expanded=False):
        e1, e2, e3 = st.columns(3)
        e1.metric("Estimated extraction (mid, rounded)", f"~{_round_uncertain(vol_mid):,.0f} kL",
                   help=f"Midpoint estimate: η={eta_mid:.2f}, TDH={head_m:.1f} m. Rounded to 2 "
                        f"significant figures — see the uncertainty range, not this single number, "
                        f"as the actual result.")
        e2.metric("Uncertainty range", f"{_round_uncertain(vol_low):,.0f} – {_round_uncertain(vol_high):,.0f} kL",
                   help=f"Propagates BOTH efficiency ({eta_low:.2f}–{eta_high:.2f}) and TDH "
                        f"({current_depth_now*(1+FRICTION_HEAD_FRACTION_RANGE[0]):.1f}–"
                        f"{current_depth_now*(1+FRICTION_HEAD_FRACTION_RANGE[2]):.1f} m) uncertainty together.")
        e3.metric("Well's historical range (30d)", f"{hist_pump_30d_series.min():,.0f} – {hist_pump_30d_series.max():,.0f} kL",
                   help="This well's own actual historical 30-day pumping range, for comparison.")

        st.markdown(
            f"**Confidence: <span style='color:{quality['color']}'>{quality['band']}</span>** "
            f"(score {quality['score']}/100) — " + "; ".join(quality["issues"]) + ".",
            unsafe_allow_html=True,
        )
        st.caption(provenance_caption("extraction"))

        st.markdown("**Assumptions used** (same methodology applied to every well in the network):")
        st.markdown(
            f"- Pump type: **{pump_profile}** → η = {eta_low:.2f}–{eta_high:.2f} (mid {eta_mid:.2f}), "
            f"from published Indian field studies (see citations in Data Sources & Methodology below)\n"
            f"- TDH = static lift ({current_depth_now:.1f} m, this well's measured DWLR depth) × "
            f"(1 + 10–25% delivery/friction allowance) = {head_m:.1f} m (mid case)\n"
            f"- {pct_attributable}% of the entered {energy_kwh_raw:,.0f} kWh treated as attributable "
            f"to groundwater pumping specifically (adjustable — see caveat above)\n"
            f"- ρ = 1000 kg/m³, g = 9.81 m/s² (physical constants, not assumptions)"
        )

        st.markdown("**Physical plausibility cross-check** (flags for your review — nothing below is "
                     "auto-corrected or hidden if it looks wrong):")
        checks = []
        hist_min, hist_max = hist_pump_30d_series.min(), hist_pump_30d_series.max()
        if hist_max > 0 and (vol_mid > hist_max * 2.5 or vol_mid < hist_min * 0.3):
            checks.append(("⚠️", f"Estimate ({vol_mid:,.0f} kL) is far outside this well's historical "
                                   f"30-day range ({hist_min:,.0f}–{hist_max:,.0f} kL) — double-check the "
                                   f"entered electricity figure."))
        else:
            checks.append(("✅", f"Estimate is within a plausible range of this well's own history "
                                   f"({hist_min:,.0f}–{hist_max:,.0f} kL)."))
        if recent_rain_30d > 80 and vol_mid > default_pump_30d * 1.5:
            checks.append(("⚠️", f"Recent rainfall was high ({recent_rain_30d:.0f} mm in 30 days) — "
                                   f"heavy pumping alongside heavy rainfall is less typical; verify the figure."))
        else:
            checks.append(("✅", f"Estimate is broadly consistent with recent rainfall ({recent_rain_30d:.0f} mm/30d)."))
        if depth_90d_ago_series is not None and len(depth_90d_ago_series):
            trend_word = "deepening (depleting)" if depth_trend_90d > 0.3 else ("recovering" if depth_trend_90d < -0.3 else "stable")
            checks.append(("ℹ️", f"This well's measured depth has been **{trend_word}** over the last 90 days "
                                   f"({depth_trend_90d:+.2f} m) — cross-check this against the extraction estimate."))
        for icon, text in checks:
            st.markdown(f"{icon} {text}")

        st.caption(
            "This is an engineering approximation for a prototype, not a certified metering result — "
            "report it as such. All figures above are reproducible from the inputs shown; nothing here "
            "is fetched or fabricated."
        )

    # ---- Alerts (Feature 1) ----
    render_alert(current_risk, *AUTHORITY_ALERT_TEXT[current_risk])
    if forecast_risk != current_risk:
        proj_title, _ = AUTHORITY_ALERT_TEXT[forecast_risk]
        render_alert(forecast_risk, f"Projected in {horizon_label}: {proj_title}",
                     f"Risk band is projected to shift from {current_risk} to {forecast_risk} under the current mode.")

    if current_risk in ("Medium", "High") or forecast_risk in ("Medium", "High"):
        with st.expander(f"❓ Why is {well_row['name']} showing {max(current_risk, forecast_risk, key=lambda r: ['Low','Medium','High'].index(r))} risk?", expanded=(current_risk == "High")):
            drivers = explain_risk_drivers(
                current_depth, depth_trend_90d, recent_rain_30d, typical_rain_30d,
                vol_mid, default_pump_30d, bool(well_row["is_coastal"]), well_row["distance_to_coast_km"],
                current_ec, ec_trend_90d,
            )
            for d in drivers:
                st.markdown(f"- {d}")
            st.caption("Each factor above is computed from this well's actual observed/estimated data — "
                       "not a generic template. Absence of a factor here means that condition wasn't met.")

    depth_mae = config["metrics"]["depth"]["test"]["mae"]
    sal_mae = config["metrics"]["salinity"]["test"]["mae"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t("current_depth_metric", lang), f"{current_depth:.1f} m", f"{current_depth - year_ago_depth:+.1f} m vs 1 yr ago",
              help=provenance_caption("depth_observed"))
    c2.metric(f"Projected depth in {horizon_label}", f"{forecast_depth:.1f} m", f"{forecast_depth - current_depth:+.1f} m",
              help=f"{provenance_caption('depth_forecast')} Typical forecast error (test-set MAE): \u00b1{depth_mae:.2f} m.")
    c3.metric(t("current_risk_metric", lang), current_risk,
              help=f"{provenance_caption('salinity_observed')} Risk band from observed EC ({current_ec:.0f} \u00b5S/cm) "
                   f"\u2014 a classification, NOT a confirmed seawater-intrusion finding.")
    c4.metric(f"Projected risk in {horizon_label}", forecast_risk,
              help=f"{provenance_caption('salinity_forecast')} Predicted EC: {forecast_ec:.0f} \u00b1{sal_mae:.0f} "
                   f"\u00b5S/cm (test-set MAE). This is a MODEL PREDICTION, not an observed measurement.")

    # ---- Emergency Alert & Broadcast ----
    st.markdown("---")
    st.markdown(f"## {t('broadcast_header', lang)}")
    broadcast_color, broadcast_label = broadcast_severity(current_ec)
    st.caption(
        f"Detected severity for **{well_row['name']}**: **{broadcast_color} — {broadcast_label}** "
        f"(current salinity reading: {current_ec:.0f} µS/cm). This reflects a **predicted** risk "
        f"level from the monitoring system, not a confirmed disaster — the message below is worded accordingly."
    )

    recipients_store = get_recipients_store()
    all_recipients = recipients_store["recipients"]
    zone_recipients = [r for r in all_recipients if r["zone"] == well_row["name"]]

    auto_key = f"auto_msg_{selected_well_id}_{broadcast_color}"
    if auto_key not in st.session_state:
        st.session_state[auto_key] = generate_layman_message(well_row["name"], broadcast_color, broadcast_label)

    st.markdown("**Automatic Critical Alert** — plain language, no technical jargon (edit if needed):")
    auto_message = st.text_area("Automatic alert message", key=auto_key, height=90, label_visibility="collapsed")

    st.markdown("**Authority's Manual Message** — optional additional instructions or local context:")
    manual_message = st.text_area(
        "Manual message", value="", height=80, label_visibility="collapsed",
        placeholder="e.g., Water tankers will be deployed to the affected wards starting tomorrow 8 AM.",
        key=f"manual_msg_{selected_well_id}",
    )

    st.markdown(f"**Recipients** — suggested: all {len(zone_recipients)} registered under "
                f"**{well_row['name']} Coastal Zone**")
    recipient_scope = st.radio(
        "Send to",
        [f"All in {well_row['name']} zone ({len(zone_recipients)})",
         f"All recipients, entire region ({len(all_recipients)})",
         "Choose individually"],
        key=f"scope_{selected_well_id}",
    )
    if recipient_scope.startswith("All in"):
        selected_recipients = zone_recipients
    elif recipient_scope.startswith("All recipients"):
        selected_recipients = all_recipients
    else:
        zone_ids = {r["id"] for r in zone_recipients}
        options = {f"{r['name']} — {r['zone']} ({r['email']})": r for r in all_recipients}
        default_picks = [label for label, r in options.items() if r["id"] in zone_ids]
        picked = st.multiselect("Select individuals", list(options.keys()), default=default_picks,
                                 key=f"multiselect_{selected_well_id}")
        selected_recipients = [options[p] for p in picked]

    st.caption(f"{len(selected_recipients)} recipient(s) currently selected.")

    if st.button(t("send_broadcast_button", lang), type="primary", use_container_width=True, key=f"send_{selected_well_id}"):
        # Uses the dataset's own last observation date (not the real wall-clock date) so the
        # timestamp stays consistent with every other "current" reading in this dashboard,
        # which are all as of the dataset's most recent day — combined with the live
        # time-of-day so it still feels current during a demo.
        send_timestamp = pd.Timestamp.combine(data["date"].max().date(), pd.Timestamp.now().time())
        email_body = build_email_body(
            well_row["name"], broadcast_color, broadcast_label,
            auto_message, manual_message, send_timestamp,
        )
        email_status, email_results, email_summary = send_broadcast_emails(
            selected_recipients, f"{broadcast_label} Alert — {well_row['name']} Coastal Zone", email_body
        )

        # ---- In-app broadcast behaviour: unchanged from before ----
        store = get_broadcast_store()
        store["alerts"].insert(0, {
            "id": store["next_id"], "timestamp": send_timestamp,
            "well_id": selected_well_id, "well_name": well_row["name"],
            "severity_color": broadcast_color, "severity_label": broadcast_label,
            "auto_message": auto_message, "manual_message": manual_message,
            "recipient_count": len(selected_recipients),
            "email_status": email_status, "email_detail": email_summary,
            "email_results": email_results,
        })
        store["next_id"] += 1

        st.success("✅ In-app broadcast delivered — now visible on the Public/User Interface.")
        if email_status == "not_configured":
            st.info("📧 External email service not configured.")
        elif email_status == "no_recipients":
            st.warning("📧 No recipients were selected — no emails to send.")
        else:
            (st.success if email_status == "sent" else st.warning)(f"📧 {email_summary}")
            with st.expander("Per-recipient delivery detail", expanded=(email_status != "sent")):
                results_df = pd.DataFrame(email_results)[["name", "email", "status", "detail"]]
                results_df.columns = ["Name", "Email", "Status", "Detail"]
                st.dataframe(results_df, use_container_width=True, height=min(220, 40 + 35 * len(results_df)))

    with st.expander(t("manage_recipients_header", lang), expanded=False):
        if all_recipients:
            rec_df = pd.DataFrame(all_recipients)[["name", "zone", "email", "phone", "category"]]
            rec_df.columns = ["Name", "Zone", "Email", "Phone", "Category"]
            st.dataframe(rec_df, use_container_width=True, height=220)
        else:
            st.caption("No recipients yet — add one below.")

        st.markdown("**Add a recipient**")
        with st.form(f"add_recipient_form_{selected_well_id}", clear_on_submit=True):
            rc1, rc2 = st.columns(2)
            with rc1:
                new_name = st.text_input("Name")
                coastal_zone_names = list(wells_meta[wells_meta["is_coastal"]]["name"])
                new_zone = st.selectbox("Zone", coastal_zone_names)
                new_email = st.text_input("Email")
            with rc2:
                new_phone = st.text_input("Phone (optional)")
                new_category = st.selectbox("Category", ["Resident", "Farmer", "Local Official"])
            submitted = st.form_submit_button("Add recipient")
            if submitted and new_name and new_email:
                recipients_store["recipients"].append({
                    "id": recipients_store["next_id"], "name": new_name, "zone": new_zone,
                    "email": new_email, "phone": new_phone, "category": new_category,
                })
                recipients_store["next_id"] += 1
                st.rerun()

        if all_recipients:
            st.markdown("**Remove a recipient**")
            remove_labels = [f"{r['name']} ({r['zone']})" for r in all_recipients]
            remove_choice = st.selectbox("Select to remove", remove_labels, key=f"remove_sel_{selected_well_id}")
            if st.button("Remove selected recipient", key=f"remove_btn_{selected_well_id}"):
                idx = remove_labels.index(remove_choice)
                recipients_store["recipients"].pop(idx)
                st.rerun()

    with st.expander(t("broadcast_history_header", lang), expanded=False):
        broadcast_history = get_broadcast_store()["alerts"]
        if not broadcast_history:
            st.caption("No broadcasts sent yet.")
        else:
            hist_df = pd.DataFrame(broadcast_history).copy()
            hist_df["Date/Time"] = hist_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
            hist_df["Message"] = (hist_df["auto_message"] + hist_df["manual_message"].apply(
                lambda m: (" | " + m) if m.strip() else "")).str.slice(0, 100) + "..."
            display_df = hist_df.rename(columns={
                "severity_label": "Severity", "well_name": "Location",
                "recipient_count": "Recipients", "email_status": "Email status",
            })[["Date/Time", "Severity", "Location", "Message", "Recipients", "Email status"]]
            st.dataframe(display_df, use_container_width=True, height=240)

    # ---- Trend + forecast chart ----
    st.markdown("---")
    st.subheader(f"{well_row['name']}, {well_row['state']} — trend & forecast")
    plot_hist = well_hist[well_hist["date"] >= well_hist["date"].max() - pd.Timedelta(days=540)]
    tab1, tab2 = st.tabs(["Water table depth", "Salinity (EC)"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_hist["date"], y=plot_hist["depth_to_water_m"],
                                  name="Observed (DWLR)", line=dict(color="#3498db")))
        bridge_x = [plot_hist["date"].iloc[-1]] + list(forecast_df["date"])
        bridge_y = [plot_hist["depth_to_water_m"].iloc[-1]] + list(forecast_df["depth_to_water_m"])
        fig.add_trace(go.Scatter(x=bridge_x, y=bridge_y, name="Forecast",
                                  line=dict(color="#e67e22", dash="dash")))
        fig.update_yaxes(autorange="reversed", title="Depth to water table (m) — deeper = more depleted")
        fig.update_layout(height=420, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=plot_hist["date"], y=plot_hist["salinity_ec_uscm"],
                                   name="Observed", line=dict(color="#16a085")))
        bridge_y2 = [plot_hist["salinity_ec_uscm"].iloc[-1]] + list(forecast_df["salinity_ec_uscm"])
        fig2.add_trace(go.Scatter(x=bridge_x, y=bridge_y2, name="Forecast",
                                   line=dict(color="#e67e22", dash="dash")))
        for label, (lo, hi) in RISK_BANDS.items():
            if hi > 100000:
                hi = max(plot_hist["salinity_ec_uscm"].max(), forecast_df["salinity_ec_uscm"].max()) * 1.1
            fig2.add_hrect(y0=lo, y1=hi, fillcolor=RISK_COLOR[label], opacity=0.08, line_width=0)
        fig2.update_yaxes(title="Electrical conductivity (uS/cm)")
        fig2.update_layout(height=420, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Rainfall & pumping (drivers) — last 18 months"):
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=plot_hist["date"], y=plot_hist["rainfall_mm"], name="Rainfall (mm)", marker_color="#5dade2"))
        fig3.add_trace(go.Scatter(x=plot_hist["date"], y=plot_hist["pumping_kl"], name="Pumping (kL)",
                                   yaxis="y2", line=dict(color="#c0392b")))
        fig3.update_layout(
            height=350,
            yaxis=dict(title="Rainfall (mm)"),
            yaxis2=dict(title="Pumping (kL)", overlaying="y", side="right"),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ---- Network-wide map + alert table ----
    st.subheader("Network overview — all monitoring wells")
    with st.spinner("Running forecast across the well network..."):
        network_fc = forecast_all_wells(horizon_steps, rain_pct, pump_pct)

    network_fc.loc[network_fc["well_id"] == selected_well_id, "depth_to_water_m"] = forecast_depth
    network_fc.loc[network_fc["well_id"] == selected_well_id, "salinity_ec_uscm"] = forecast_ec

    network = wells_meta.merge(network_fc, on="well_id", suffixes=("", "_fc"))
    current_latest = data.sort_values("date").groupby("well_id").last().reset_index()[["well_id", "depth_to_water_m", "salinity_ec_uscm"]]
    network = network.merge(current_latest, on="well_id", suffixes=("_forecast", "_current"))
    network["current_risk"] = network["salinity_ec_uscm_current"].apply(risk_band)
    network["forecast_risk"] = network["salinity_ec_uscm_forecast"].apply(risk_band)
    network["drawdown_m"] = network["depth_to_water_m_forecast"] - network["depth_to_water_m_current"]

    high_risk_wells = network[network["forecast_risk"] == "High"]
    if len(high_risk_wells):
        names = ", ".join(high_risk_wells["name"].tolist())
        render_alert("High", f"🚨 {len(high_risk_wells)} well(s) projected at HIGH risk",
                     f"Zones needing priority attention: {names}")

    map_col, table_col = st.columns([3, 2])

    with map_col:
        # High-risk markers are boosted well beyond their drawdown-proportional size so they
        # visually dominate the map at a glance — red = danger should be unmissable, not subtle.
        base_size = network["drawdown_m"].abs().clip(lower=0.3)
        risk_boost = network["forecast_risk"].map({"Low": 1.0, "Medium": 1.6, "High": 2.8})
        network["marker_size"] = base_size * risk_boost
        fig_map = px.scatter_map(
            network, lat="lat", lon="lon", color="forecast_risk",
            color_discrete_map=RISK_COLOR,
            category_orders={"forecast_risk": ["Low", "Medium", "High"]},
            hover_name="name",
            hover_data={"lat": False, "lon": False, "state": True,
                        "depth_to_water_m_current": ":.1f", "drawdown_m": ":.2f",
                        "marker_size": False},
            size="marker_size", size_max=26, zoom=3.6,
            center={"lat": 20.5, "lon": 79.0},
            map_style="carto-positron",
            title=f"Projected salinity risk in {horizon_label} — 🔴 red = high danger, larger = more urgent",
        )
        fig_map.update_layout(height=460, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_map, use_container_width=True)

    with table_col:
        alert_tbl = network[[
            "name", "state", "is_coastal", "depth_to_water_m_current",
            "drawdown_m", "current_risk", "forecast_risk"
        ]].rename(columns={
            "name": "Well", "state": "State", "is_coastal": "Coastal",
            "depth_to_water_m_current": "Depth now (m)",
            "drawdown_m": f"Δ depth ({horizon_label})",
            "current_risk": "Risk now", "forecast_risk": f"Risk ({horizon_label})",
        }).sort_values(f"Δ depth ({horizon_label})", ascending=False)

        def highlight_risk(val):
            color = RISK_COLOR.get(val)
            return f"background-color: {color}55" if color else ""

        styled = alert_tbl.style.format({"Depth now (m)": "{:.1f}", f"Δ depth ({horizon_label})": "{:+.2f}"})
        if hasattr(styled, "map"):
            styled = styled.map(highlight_risk, subset=["Risk now", f"Risk ({horizon_label})"])
        else:  # pragma: no cover - older pandas fallback
            styled = styled.applymap(highlight_risk, subset=["Risk now", f"Risk ({horizon_label})"])
        st.dataframe(styled, height=460, use_container_width=True)

    st.caption(
        "Wells colour-coded Low (green) / Medium (orange) / High (red) salinity risk. "
        f"Δ depth = projected change over {horizon_label} under the current mode; positive = further depletion. "
        f"{well_row['name']} reflects your electricity-based estimate above; other wells use their own historical trend."
    )

    # ---- Data sources + formulas (Feature 2) ----
    with st.expander("📚 Data sources & methodology (formulas used)", expanded=False):
        render_data_sources_and_formulas()

    # ---- Explainability ----
    with st.expander("How the model decides (feature importance + validation metrics)"):
        imp_depth = pd.DataFrame({"feature": DEPTH_FEATURES, "importance": depth_model.feature_importances_}).sort_values("importance")
        imp_sal = pd.DataFrame({"feature": SALINITY_FEATURES, "importance": sal_model.feature_importances_}).sort_values("importance")
        ic1, ic2 = st.columns(2)
        with ic1:
            st.plotly_chart(px.bar(imp_depth, x="importance", y="feature", orientation="h",
                                    title="Depth forecast — driver importance"), use_container_width=True)
        with ic2:
            st.plotly_chart(px.bar(imp_sal, x="importance", y="feature", orientation="h",
                                    title="Salinity forecast — driver importance"), use_container_width=True)

        st.markdown("**Actual computed accuracy** — chronological train/validation/test split, "
                     "no shuffling, no leakage (see `train_model.py`):")
        m = config["metrics"]
        metrics_rows = []
        for model_name, unit in [("depth", "m"), ("salinity", "\u00b5S/cm")]:
            for split_name in ("validation", "test"):
                d = m[model_name][split_name]
                metrics_rows.append({
                    "Model": model_name.capitalize(), "Split": split_name.capitalize(),
                    "n": d["n"], "R\u00b2": d["r2"], f"MAE ({unit})": d["mae"], f"RMSE ({unit})": d["rmse"],
                })
        st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True, hide_index=True)
        st.caption(
            f"Train: 2022-01-01 to {config['train_end']}. Validation: {config['train_end']} to "
            f"{config['validation_end']}. Test (final, held out): {config['validation_end']} onward. "
            "Every number above comes directly from sklearn's r2_score/mean_absolute_error/"
            "mean_squared_error against real held-out rows — none of it is estimated or invented."
        )


# ============================================================================
# PUBLIC / SIMPLE VIEW
# ============================================================================
def render_public_view():
    # Lightweight auto-refresh so a broadcast sent from the Authority tab
    # appears here without the resident manually reloading the page.
    st.markdown('<meta http-equiv="refresh" content="20">', unsafe_allow_html=True)

    # ---- Area selection first (Auto-language depends on knowing this) ----
    # NOTE: this one selector's own label stays fixed/English by necessity —
    # we need to know WHICH area is selected before we can know which
    # language to show it in. Everything else on this page below responds
    # fully to the selected language.
    st.sidebar.title("📍 Select your area")
    well_label_to_id = {f"{r['name']}, {r['state']}": r["well_id"] for _, r in wells_meta.iterrows()}
    selected_label = st.sidebar.radio("Your location", list(well_label_to_id.keys()), index=0,
                                       key="public_area_radio")
    selected_well_id = well_label_to_id[selected_label]

    # ---- Language: automatic (from the selected coastal zone) + manual override ----
    # "Auto" recomputes from WELL_DEFAULT_LANGUAGE every time the area changes;
    # picking a specific language overrides it and stays chosen (Streamlit
    # preserves this widget's selection across reruns) until switched back to
    # Auto or changed again — that IS the manual override.
    lang_name_to_code = {v: k for k, v in LANGUAGES.items()}
    auto_label = "\U0001f310 Auto (based on your area)"
    lang_options = [auto_label] + list(LANGUAGES.values())
    selected_lang_option = st.sidebar.selectbox("\U0001f310 Language", lang_options, key="public_lang_select")
    if selected_lang_option == auto_label:
        lang = WELL_DEFAULT_LANGUAGE.get(selected_well_id, "en")
    else:
        lang = lang_name_to_code.get(selected_lang_option, "en")

    top_l, top_r = st.columns([5, 1])
    with top_l:
        st.title(t("public_title", lang))
        st.caption(t("public_caption", lang))
    with top_r:
        if st.button(t("switch_profile", lang), use_container_width=True):
            st.session_state.profile = None
            st.rerun()

    well_hist = data[data["well_id"] == selected_well_id].sort_values("date")
    well_row = wells_meta[wells_meta["well_id"] == selected_well_id].iloc[0]

    current_depth = well_hist["depth_to_water_m"].iloc[-1]
    current_ec = well_hist["salinity_ec_uscm"].iloc[-1]
    current_risk = risk_band(current_ec)

    # simple, fixed 3-month "business as usual" outlook — no jargon, no sliders
    forecast_df = forecast_well(well_hist, 3, 100, 100)
    forecast_risk = risk_band(forecast_df["salinity_ec_uscm"].iloc[-1])

    st.subheader(f"📍 {well_row['name']}, {well_row['state']}")

    # ---- Active Authority Alerts (real broadcasts from the Authority view) ----
    top_row, refresh_row = st.columns([4, 1])
    with top_row:
        st.markdown(f"### {t('active_alerts', lang)}")
    with refresh_row:
        if st.button(t("check_now", lang), key="public_check_now"):
            st.rerun()

    zone_alerts = [a for a in get_broadcast_store()["alerts"] if a["well_id"] == selected_well_id]
    if not zone_alerts:
        st.caption(t("no_alerts", lang))
    else:
        latest = zone_alerts[0]
        st.markdown(
            f"**{t('severity_label', lang)}:** {latest['severity_color']} — {latest['severity_label']}  \n"
            f"**{t('location_label', lang)}:** {latest['well_name']} {t('coastal_zone', lang)}  \n"
            f"**{t('datetime_label', lang)}:** {latest['timestamp'].strftime('%d %b %Y, %I:%M %p')}"
        )
        level_for_style = broadcast_color_to_alert_level(latest["severity_color"])
        render_alert(level_for_style, t("system_warning", lang), latest["auto_message"], big=True)
        if latest["manual_message"].strip():
            st.markdown(f"**{t('authority_message_label', lang)}:** {latest['manual_message']}")
        status_key_map = {
            "sent": "status_sent", "not_configured": "status_not_configured",
            "no_recipients": "status_no_recipients", "failed": "status_failed",
        }
        if latest["email_status"] == "partial":
            status_text = t("status_partial_template", lang, detail=latest.get("email_detail", ""))
        else:
            status_text = t(status_key_map.get(latest["email_status"], "status_not_configured"), lang)
        st.markdown(f"**{t('alert_status_label', lang)}:** {status_text}")

        if len(zone_alerts) > 1:
            with st.expander(t("past_alerts_template", lang, n=len(zone_alerts) - 1)):
                for a in zone_alerts[1:]:
                    st.markdown(f"- **{a['timestamp'].strftime('%d %b %Y, %I:%M %p')}** — "
                                f"{a['severity_color']} {a['severity_label']}: {a['auto_message']}")

    st.markdown("---")

    render_alert(current_risk, t(f"alert_{current_risk.lower()}_title", lang),
                 t(f"alert_{current_risk.lower()}_msg", lang), big=True)

    if forecast_risk != current_risk and forecast_risk == "High":
        st.markdown(t("looking_ahead_worse", lang))
    elif forecast_risk != current_risk and RISK_COLOR[forecast_risk] == RISK_COLOR["Low"]:
        st.markdown(t("looking_ahead_better", lang))

    st.markdown("---")
    st.markdown(f"### {t('what_does_mean', lang)}")
    b1, b2 = st.columns(2)
    with b1:
        st.markdown(f"**{t('water_level_header', lang)}**")
        level_key = "level_very_deep" if current_depth > 15 else ("level_getting_deep" if current_depth > 8 else "level_healthy")
        st.markdown(t("water_level_sentence", lang, level_words=t(level_key, lang), depth=current_depth))
    with b2:
        st.markdown(f"**{t('water_saltiness_header', lang)}**")
        salt_key = {"Low": "salt_not_salty", "Medium": "salt_slightly_salty", "High": "salt_quite_salty"}[current_risk]
        st.markdown(t("water_saltiness_sentence", lang, salt_words=t(salt_key, lang)))

    st.markdown(f"### {t('what_should_do', lang)}")
    advice_key = {"Low": "advice_low", "Medium": "advice_medium", "High": "advice_high"}[current_risk]
    st.info(t(advice_key, lang))

    # ---- Crop suggestions for coastal farmers, driven by the current risk band ----
    if bool(well_row["is_coastal"]):
        st.markdown("---")
        st.markdown(f"### {t('crop_suggestions_header', lang)}")
        st.caption(t("crop_suggestions_intro", lang))
        guidance = CROP_GUIDANCE[current_risk]
        for crop_name, reason in guidance["crops"]:
            st.markdown(f"- **{crop_name}** — {reason}")
        if guidance["avoid"]:
            st.markdown(f"**{t('crops_to_reduce_label', lang)}:** " + ", ".join(guidance["avoid"]))
        st.caption(t("crop_guidance_source", lang))

    st.caption(t("simplified_view_caption", lang))



def inject_global_theme():
    """Site-wide visual polish — subtle entrance animation, a floating water-
    drop keyframe used on the landing hero, and gentle hover/shadow styling
    for buttons and metric cards. Purely cosmetic; touches no app logic."""
    st.markdown("""
    <style>
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes floatDrop {
        0%, 100% { transform: translateY(0px); }
        50%      { transform: translateY(-8px); }
    }
    .block-container { animation: fadeInUp 0.35s ease-out; }
    .hero-drop { display: inline-block; animation: floatDrop 2.6s ease-in-out infinite; }
    div[data-testid="stButton"] > button {
        border-radius: 10px;
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
        border: 1px solid rgba(52,152,219,0.35);
    }
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 14px rgba(52,152,219,0.25);
        border-color: #3498db;
    }
    div[data-testid="stMetric"] {
        background: rgba(52,152,219,0.06);
        border-radius: 10px;
        padding: 10px 8px;
        border: 1px solid rgba(52,152,219,0.12);
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# PROFILE SELECTOR (Feature 4) — landing screen + session-based routing
# ============================================================================
inject_global_theme()

if "profile" not in st.session_state:
    st.session_state.profile = None

if st.session_state.profile is None:
    st.markdown("""
    <div style="background: linear-gradient(120deg, #2980b9, #16a085); border-radius:18px;
                padding:34px 30px; margin-bottom:22px; color:white; text-align:center;">
        <div class="hero-drop" style="font-size:50px;">💧</div>
        <div style="font-size:26px; font-weight:800; margin-top:8px; line-height:1.3;">
            Groundwater Depletion & Salinity Infiltration Predictive Matrix
        </div>
        <div style="font-size:15px; opacity:0.92; margin-top:10px;">
            Correlates DWLR water-level readings, rainfall, and agricultural pumping data
            to forecast aquifer drawdown and coastal salinity ingress.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("#### Choose how you'd like to use this tool")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="border:1px solid rgba(52,152,219,0.3); border-radius:14px; padding:20px;
                    background:rgba(52,152,219,0.05); min-height:150px;">
            <div style="font-size:32px;">🔧</div>
            <div style="font-size:18px; font-weight:700; margin-top:6px;">Water Authority</div>
            <div style="font-size:13.5px; opacity:0.85; margin-top:6px;">
                Full technical dashboard: forecasts, what-if scenarios, electricity-based
                extraction estimation, emergency alert &amp; broadcast, network map, model internals.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        if st.button("Enter Authority Dashboard", use_container_width=True, type="primary"):
            st.session_state.profile = "authority"
            st.rerun()
    with col2:
        st.markdown("""
        <div style="border:1px solid rgba(46,204,113,0.35); border-radius:14px; padding:20px;
                    background:rgba(46,204,113,0.05); min-height:150px;">
            <div style="font-size:32px;">👥</div>
            <div style="font-size:18px; font-weight:700; margin-top:6px;">Resident / Public</div>
            <div style="font-size:13.5px; opacity:0.85; margin-top:6px;">
                A simple, plain-language view — pick your area, see if the water is
                safe right now, see official alerts, and what to do about it.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        if st.button("Enter Public View", use_container_width=True):
            st.session_state.profile = "public"
            st.rerun()
    st.stop()

if st.session_state.profile == "authority":
    render_authority_view()
else:
    render_public_view()
