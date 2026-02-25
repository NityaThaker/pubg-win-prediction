import streamlit as st
import requests
import pandas as pd
import os

# ── page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "PUBG Win Predictor",
    page_icon  = "🎯",
    layout     = "wide"
)

# ── session state init ─────────────────────────────────────────────────────
if "match_history" not in st.session_state:
    st.session_state.match_history = []

# ── API URL ────────────────────────────────────────────────────────────────
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")

# ── header ─────────────────────────────────────────────────────────────────
st.title("🎯 PUBG Win Placement Predictor")
st.markdown("Enter your match stats below to predict your finish placement and get a personalised scouting report.")

# ── health check ───────────────────────────────────────────────────────────
try:
    health = requests.get(f"{FASTAPI_URL}/health", timeout=5).json()
    if health.get("model_loaded"):
        st.sidebar.success(f"✅ API Connected — {health['feature_count']} features loaded")
    else:
        st.sidebar.error("⚠️ API connected but model not loaded")
except:
    st.sidebar.error("❌ Cannot reach API — check the URL above")

st.markdown("---")

# ── input form ─────────────────────────────────────────────────────────────
st.subheader("📊 Match Stats Input")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Kill Stats**")
    kills          = st.slider("Kills",           0, 35,   2)
    killPlace      = st.slider("Kill Place",      1, 100,  20)
    killPoints     = st.number_input("Kill Points",     0.0, 2000.0, 1000.0, step=10.0)
    killStreaks     = st.slider("Kill Streaks",    0, 20,   1)
    headshotKills  = st.slider("Headshot Kills",  0, 35,   1)
    longestKill    = st.number_input("Longest Kill (m)", 0.0, 1500.0, 50.0, step=1.0)
    DBNOs          = st.slider("DBNOs",            0, 40,   1)
    assists        = st.slider("Assists",          0, 20,   0)

with col2:
    st.markdown("**Distance & Damage**")
    damageDealt    = st.number_input("Damage Dealt",     0.0, 5000.0, 200.0, step=10.0)
    walkDistance   = st.number_input("Walk Distance (m)",0.0, 25000.0,1500.0, step=50.0)
    rideDistance   = st.number_input("Ride Distance (m)",0.0, 40000.0,0.0,   step=50.0)
    swimDistance   = st.number_input("Swim Distance (m)",0.0, 3000.0, 0.0,   step=10.0)
    roadKills      = st.slider("Road Kills",       0, 20,   0)
    vehicleDestroys= st.slider("Vehicle Destroys", 0, 20,   0)

with col3:
    st.markdown("**Survival & Match Context**")
    heals          = st.slider("Heals",            0, 40,   2)
    boosts         = st.slider("Boosts",           0, 20,   1)
    revives        = st.slider("Revives",          0, 10,   0)
    teamKills      = st.slider("Team Kills",       0, 4,    0)
    weaponsAcquired= st.slider("Weapons Acquired", 0, 30,   4)
    matchDuration  = st.slider("Match Duration (s)",1, 2400, 1800)
    maxPlace       = st.slider("Max Place",        1, 100,  100)
    numGroups      = st.slider("Num Groups",       1, 100,  50)
    rankPoints     = st.number_input("Rank Points",  0.0, 5000.0, 1500.0, step=10.0)
    winPoints      = st.number_input("Win Points",   0.0, 5000.0, 1200.0, step=10.0)

# ── build payload ──────────────────────────────────────────────────────────
payload = {
    "kills": kills, "killPlace": killPlace, "killPoints": killPoints,
    "killStreaks": killStreaks, "headshotKills": headshotKills,
    "longestKill": longestKill, "DBNOs": DBNOs, "assists": assists,
    "damageDealt": damageDealt, "walkDistance": walkDistance,
    "rideDistance": rideDistance, "swimDistance": swimDistance,
    "roadKills": roadKills, "vehicleDestroys": vehicleDestroys,
    "heals": heals, "boosts": boosts, "revives": revives,
    "teamKills": teamKills, "weaponsAcquired": weaponsAcquired,
    "matchDuration": matchDuration, "maxPlace": maxPlace,
    "numGroups": numGroups, "rankPoints": rankPoints, "winPoints": winPoints
}

st.markdown("---")

# ── predict button ─────────────────────────────────────────────────────────
if st.button("🚀 Predict My Placement", use_container_width=True):

    with st.spinner("Running prediction..."):

        try:
            # ── call /predict ──────────────────────────────────────────────
            pred_res = requests.post(f"{FASTAPI_URL}/predict", json=payload, timeout=10).json()

            # ── call /scouting ─────────────────────────────────────────────
            scout_res = requests.post(f"{FASTAPI_URL}/scouting", json=payload, timeout=10).json()

            winPerc = pred_res["winPlacePerc"]
            pct     = pred_res["win_probability_pct"]
            band    = pred_res["confidence_band"]

            # ── prediction result ──────────────────────────────────────────
            st.markdown("## 🏆 Prediction Result")

            m1, m2, m3 = st.columns(3)
            m1.metric("Win Place %",     f"{winPerc:.4f}")
            m2.metric("Top % of Players", f"{100 - pct:.1f}%")
            m3.metric("Confidence Band",  band)

            st.markdown(f"**Placement Score:** {winPerc:.4f} out of 1.0")
            st.progress(float(winPerc))
            st.caption(pred_res["message"])

            st.markdown("---")

            # ── scouting report ────────────────────────────────────────────
            st.markdown("## 🔍 Scouting Report")

            arch  = scout_res["archetype"]
            emoji = scout_res["archetype_emoji"]

            st.markdown(f"### {emoji} {arch}")

            s1, s2 = st.columns(2)

            with s1:
                st.markdown("**✅ Strengths**")
                for s in scout_res["strengths"]:
                    st.markdown(f"- {s}")

            with s2:
                st.markdown("**⚠️ Weaknesses**")
                for w in scout_res["weaknesses"]:
                    st.markdown(f"- {w}")

            st.info(f"💡 **Coach Tip:** {scout_res['tip']}")

            st.markdown("**Stats Summary**")
            summary = scout_res["stats_summary"]
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Kills",         summary["kills"])
            c2.metric("Damage",        summary["damage_dealt"])
            c3.metric("Distance (m)",  summary["total_distance"])
            c4.metric("Heals+Boosts",  summary["heals_boosts"])
            c5.metric("Headshot Rate", summary["headshot_rate"])

            st.markdown("---")

            # ── save to match history ──────────────────────────────────────
            st.session_state.match_history.append({
                "Match":        len(st.session_state.match_history) + 1,
                "winPlacePerc": winPerc,
                "Archetype":    arch,
                "Kills":        kills,
                "Damage":       damageDealt,
            })

            st.success("✅ Match saved to history!")

        except Exception as e:
            st.error(f"❌ Error: {str(e)} — Make sure your FastAPI server is running and the URL is correct.")

st.markdown("---")

# ── match history ──────────────────────────────────────────────────────────
st.markdown("## 📈 Match History & Trend")

if len(st.session_state.match_history) == 0:
    st.info("No matches recorded yet. Submit a prediction above to start tracking your history.")

else:
    history_df = pd.DataFrame(st.session_state.match_history)

    st.line_chart(
        history_df.set_index("Match")["winPlacePerc"],
        use_container_width=True
    )

    st.markdown("**All Recorded Matches**")
    st.dataframe(history_df, use_container_width=True)

    if st.button("🗑️ Clear Match History"):
        st.session_state.match_history = []
        st.rerun()

st.markdown("---")
st.caption("PUBG Win Placement Predictor — Built with LightGBM + FastAPI + Streamlit | MAE: 0.0513")