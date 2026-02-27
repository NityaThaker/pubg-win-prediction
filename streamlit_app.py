import os
import pickle
import numpy as np
import streamlit as st
import gdown

# ── 1. Download model files from Google Drive ─────────────────────────────────
MODEL_FILE_ID    = "1rPxQVZ1u1gAEdftkAeAANXoGKFrAMxQM"          # pubg_model_v5.pkl
FEATURES_FILE_ID = "1j01b9zB43HqkvUqhvKlPxCzxHLcaz7Bo"       # day5_feature_cols.pkl

MODEL_PATH    = "pubg_model_v5.pkl"
FEATURES_PATH = "day5_feature_cols.pkl"

@st.cache_resource(show_spinner="Loading model… (first run only)")
def load_model_and_features():
    if not os.path.exists(MODEL_PATH):
        gdown.download(
            f"https://drive.google.com/uc?id={MODEL_FILE_ID}",
            MODEL_PATH, quiet=False
        )
    if not os.path.exists(FEATURES_PATH):
        gdown.download(
            f"https://drive.google.com/uc?id={FEATURES_FILE_ID}",
            FEATURES_PATH, quiet=False
        )
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f:
        feature_cols = pickle.load(f)
    return model, feature_cols

model, feature_cols = load_model_and_features()

# ── 2. Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PUBG Win Predictor",
    page_icon="🎮",
    layout="centered"
)

st.title("🎮 PUBG Win Placement Predictor")
st.markdown(
    "Enter your match stats below. The model predicts your **winPlacePerc** "
    "(0 = last place, 1 = winner)."
)
st.info(
    "⚠️ **Note:** Predictions will be conservative (0.4–0.6) for manual input "
    "because 45 group/match aggregate features can't be computed without full "
    "match data. This is expected — the Player Lookup system (Day 12) will fix this.",
    icon="ℹ️"
)

# ── 3. Input form ──────────────────────────────────────────────────────────────
st.subheader("Your Stats")

col1, col2, col3 = st.columns(3)

with col1:
    kills          = st.number_input("Kills",            0, 50,  2)
    assists        = st.number_input("Assists",          0, 30,  0)
    damage_dealt   = st.number_input("Damage Dealt",     0, 10000, 250)
    headshotKills  = st.number_input("Headshot Kills",   0, 50,  0)
    DBNOs          = st.number_input("DBNOs",            0, 50,  0)

with col2:
    walkDistance   = st.number_input("Walk Distance (m)",  0, 15000, 1500)
    rideDistance   = st.number_input("Ride Distance (m)",  0, 30000, 0)
    swimDistance   = st.number_input("Swim Distance (m)",  0, 5000,  0)
    boosts         = st.number_input("Boosts Used",        0, 30,    2)
    heals          = st.number_input("Heals Used",         0, 30,    1)

with col3:
    weaponsAcquired = st.number_input("Weapons Acquired",  0, 100,  5)
    killPlace       = st.number_input("Kill Place (rank)", 1, 100,  40)
    matchDuration   = st.number_input("Match Duration (s)", 0, 2500, 1800)
    maxPlace        = st.number_input("Max Place",         1, 100,  96)
    numGroups       = st.number_input("Num Groups",        1, 100,  50)

# ── 4. Feature engineering (mirrors Day 5 logic) ──────────────────────────────
def build_features(raw: dict, feature_cols: list) -> np.ndarray:
    r = raw  # shorthand

    # Basic derived
    total_distance   = r["walkDistance"] + r["rideDistance"] + r["swimDistance"]
    kill_death_ratio = r["kills"] / (r["kills"] + 1)          # proxy — no deaths col
    heals_boosts     = r["heals"] + r["boosts"]
    kills_per_dist   = r["kills"] / (total_distance + 1)
    headshot_rate    = r["headshotKills"] / (r["kills"] + 1)
    items_collected  = r["weaponsAcquired"] + r["heals"] + r["boosts"]
    kill_place_norm  = r["killPlace"] / (r["maxPlace"] + 1)
    walkDist_norm    = r["walkDistance"] / (total_distance + 1)
    rideDist_norm    = r["rideDistance"] / (total_distance + 1)
    boosts_per_walk  = r["boosts"] / (r["walkDistance"] + 1)
    dmg_per_kill     = r["damageDealt"] / (r["kills"] + 1)
    survival_score   = total_distance * 0.4 + r["heals"] * 5 + r["boosts"] * 8
    aggression_score = r["kills"] * 10 + r["damageDealt"] * 0.05 + r["DBNOs"] * 3

    # Build a dict of every engineered feature set to 0 by default
    feat = {col: 0.0 for col in feature_cols}

    # Fill in the ones we can compute
    mapping = {
        "kills":            r["kills"],
        "assists":          r["assists"],
        "damageDealt":      r["damageDealt"],
        "headshotKills":    r["headshotKills"],
        "DBNOs":            r["DBNOs"],
        "walkDistance":     r["walkDistance"],
        "rideDistance":     r["rideDistance"],
        "swimDistance":     r["swimDistance"],
        "boosts":           r["boosts"],
        "heals":            r["heals"],
        "weaponsAcquired":  r["weaponsAcquired"],
        "killPlace":        r["killPlace"],
        "matchDuration":    r["matchDuration"],
        "maxPlace":         r["maxPlace"],
        "numGroups":        r["numGroups"],
        "total_distance":   total_distance,
        "kill_death_ratio": kill_death_ratio,
        "heals_boosts":     heals_boosts,
        "kills_per_dist":   kills_per_dist,
        "headshot_rate":    headshot_rate,
        "items_collected":  items_collected,
        "kill_place_norm":  kill_place_norm,
        "walkDist_norm":    walkDist_norm,
        "rideDist_norm":    rideDist_norm,
        "boosts_per_walk":  boosts_per_walk,
        "dmg_per_kill":     dmg_per_kill,
        "survival_score":   survival_score,
        "aggression_score": aggression_score,
    }

    for k, v in mapping.items():
        if k in feat:
            feat[k] = float(v)

    # Return values in exact column order the model expects
    return np.array([feat[col] for col in feature_cols]).reshape(1, -1)

# ── 5. Predict ─────────────────────────────────────────────────────────────────
if st.button("🔮 Predict Placement", use_container_width=True):
    raw = {
        "kills": kills, "assists": assists, "damageDealt": damage_dealt,
        "headshotKills": headshotKills, "DBNOs": DBNOs,
        "walkDistance": walkDistance, "rideDistance": rideDistance,
        "swimDistance": swimDistance, "boosts": boosts, "heals": heals,
        "weaponsAcquired": weaponsAcquired, "killPlace": killPlace,
        "matchDuration": matchDuration, "maxPlace": maxPlace,
        "numGroups": numGroups,
    }

    X = build_features(raw, feature_cols)
    prediction = float(model.predict(X)[0])
    prediction = float(np.clip(prediction, 0.0, 1.0))

    st.divider()
    st.subheader("Prediction Result")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Win Place %", f"{prediction:.1%}")
    with col_b:
        if prediction >= 0.80:
            st.success("🏆 Chicken Dinner territory!")
        elif prediction >= 0.60:
            st.info("💪 Strong performance — top 40%")
        elif prediction >= 0.40:
            st.warning("😐 Average placement")
        else:
            st.error("💀 Rough match — work on survival")

    st.progress(prediction)

    st.caption(
        f"Raw score: {prediction:.4f} | "
        f"Model features used: {len(feature_cols)} | "
        f"Group/match features defaulted to 0 (see note above)"
    )

# ── 6. Footer ──────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built by Nitya Thaker · "
    "[GitHub](https://github.com/NityaThaker/pubg-win-prediction) · "
    "Model: LightGBM · MAE: 0.0513 · Trained on 1.1M PUBG matches"
)