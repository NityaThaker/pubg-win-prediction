import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import gc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="PUBG Win Predictor", layout="wide")

# ── Google Drive file IDs — FILL THESE IN ───────────────────────────────────
MODEL_FILE_ID   = "1rPxQVZ1u1gAEdftkAeAANXoGKFrAMxQM"
FEATURE_FILE_ID = "1j01b9zB43HqkvUqhvKlPxCzxHLcaz7Bo"
LOOKUP_FILE_ID  = "1btXkjuc3Q_5mqnhYn3_-E_ilSgKa7Abj"   

# ── Download helpers ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_features():
    import gdown
    if not os.path.exists("pubg_model_v5.pkl"):
        gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}",
                       "pubg_model_v5.pkl", quiet=False)
    if not os.path.exists("day5_feature_cols.pkl"):
        gdown.download(f"https://drive.google.com/uc?id={FEATURE_FILE_ID}",
                       "day5_feature_cols.pkl", quiet=False)
    with open("pubg_model_v5.pkl", "rb") as f:
        model = pickle.load(f)
    with open("day5_feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    return model, feature_cols


@st.cache_data(show_spinner="Loading player lookup dataset…")
def load_lookup_data():
    import gdown
    if not os.path.exists("day12_lookup_dataset.parquet"):
        gdown.download(f"https://drive.google.com/uc?id={LOOKUP_FILE_ID}",
                       "day12_lookup_dataset.parquet", quiet=False)
    df = pd.read_parquet("day12_lookup_dataset.parquet")
    return df


# ── Feature engineering (must match training exactly) ───────────────────────
def build_manual_features(user_input: dict, feature_cols: list) -> np.ndarray:
    """
    For manual input — fills aggregate features with 0 (known limitation).
    Unchanged from Day 9/11.
    """
    row = {col: 0.0 for col in feature_cols}
    direct = ['kills', 'damageDealt', 'walkDistance', 'rideDistance',
              'swimDistance', 'boosts', 'heals', 'weaponsAcquired',
              'headshotKills', 'assists', 'DBNOs', 'revives', 'killPlace',
              'killStreaks', 'longestKill', 'teamKills', 'roadKills',
              'vehicleDestroys', 'matchDuration', 'maxPlace', 'numGroups']
    for k in direct:
        if k in row and k in user_input:
            row[k] = float(user_input[k])

    td = row.get('walkDistance', 0) + row.get('rideDistance', 0) + row.get('swimDistance', 0)
    row['totalDistance'] = td
    row['healsAndBoosts'] = row.get('heals', 0) + row.get('boosts', 0)
    row['killsAndAssists'] = row.get('kills', 0) + row.get('assists', 0)
    row['headshotRate'] = row.get('headshotKills', 0) / (row.get('kills', 0) + 1)
    row['killsPerDistance'] = row.get('kills', 0) / (td + 1)
    row['damagePerKill'] = row.get('damageDealt', 0) / (row.get('kills', 0) + 1)
    row['walkRatio'] = row.get('walkDistance', 0) / (td + 1)
    row['rideRatio'] = row.get('rideDistance', 0) / (td + 1)
    row['killsPerMatch'] = row.get('kills', 0) / (row.get('matchDuration', 1800) / 60 + 1)
    row['damagePerMinute'] = row.get('damageDealt', 0) / (row.get('matchDuration', 1800) / 60 + 1)

    return np.array([[row.get(c, 0.0) for c in feature_cols]])


# ── Playstyle helpers ────────────────────────────────────────────────────────
STYLE_COLORS = {
    'Aggressive': '#e74c3c',
    'Passive':    '#3498db',
    'Sniper':     '#9b59b6',
    'Balanced':   '#2ecc71',
}

def classify_playstyle(row):
    kills        = row['kills']
    damage       = row['damageDealt']
    walk         = row['walkDistance']
    headshot_rate = row.get('headshotRate', 0)
    if headshot_rate >= 0.5 and kills >= 2:
        return 'Sniper'
    elif kills >= 3 or damage >= 300:
        return 'Aggressive'
    elif walk >= 2000 and kills <= 1:
        return 'Passive'
    else:
        return 'Balanced'


def get_playstyle_evolution(group_id, df):
    rows = df[df['groupId'] == group_id].copy().reset_index(drop=True)
    if rows.empty:
        return None, None
    rows['playstyle'] = rows.apply(classify_playstyle, axis=1)
    rows['match_number'] = range(1, len(rows) + 1)
    best    = rows.loc[rows['winPlacePerc'].idxmax()]
    worst   = rows.loc[rows['winPlacePerc'].idxmin()]
    summary = {
        'num_matches':       len(rows),
        'avg_placement':     rows['winPlacePerc'].mean(),
        'avg_kills':         rows['kills'].mean(),
        'avg_damage':        rows['damageDealt'].mean(),
        'playstyle_counts':  rows['playstyle'].value_counts().to_dict(),
        'dominant_playstyle': rows['playstyle'].value_counts().index[0],
        'best_match':        best[['matchId','kills','damageDealt','winPlacePerc','playstyle']].to_dict(),
        'worst_match':       worst[['matchId','kills','damageDealt','winPlacePerc','playstyle']].to_dict(),
    }
    return rows, summary


def plot_evolution(group_id, evo_df, evo_summary):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(f'Playstyle Evolution — Group: {group_id}', fontsize=14, fontweight='bold')

    colors = [STYLE_COLORS.get(s, 'gray') for s in evo_df['playstyle']]
    x = evo_df['match_number']

    ax = axes[0, 0]
    ax.scatter(x, evo_df['winPlacePerc'], c=colors, s=80, zorder=5)
    ax.plot(x, evo_df['winPlacePerc'], color='gray', linewidth=1, alpha=0.5)
    ax.set_title('Placement Over Matches')
    ax.set_xlabel('Match #'); ax.set_ylabel('winPlacePerc'); ax.set_ylim(0, 1)

    ax = axes[0, 1]
    ax.bar(x, evo_df['kills'], color=colors)
    ax.set_title('Kills Per Match')
    ax.set_xlabel('Match #'); ax.set_ylabel('Kills')

    ax = axes[1, 0]
    ax.bar(x, evo_df['damageDealt'], color=colors, alpha=0.8)
    ax.set_title('Damage Per Match')
    ax.set_xlabel('Match #'); ax.set_ylabel('Damage')

    ax = axes[1, 1]
    counts = evo_summary['playstyle_counts']
    ax.pie(counts.values(),
           labels=counts.keys(),
           colors=[STYLE_COLORS.get(k, 'gray') for k in counts.keys()],
           autopct='%1.0f%%', startangle=90)
    ax.set_title('Playstyle Distribution')

    legend_elements = [Patch(facecolor=v, label=k) for k, v in STYLE_COLORS.items()]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=9)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    return fig


# ── App ──────────────────────────────────────────────────────────────────────
st.title("🎮 PUBG Win Placement Predictor")

model, feature_cols = load_model_and_features()

tab1, tab2 = st.tabs(["🎯 Manual Input", "🔍 Player Lookup"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Manual Input (unchanged from Day 11)
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Predict from Manual Stats")
    st.info("ℹ️ Manual input uses 0 for aggregate match features, so predictions tend toward 0.4–0.6. "
            "Use the **Player Lookup** tab for fully accurate predictions.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Combat")
        kills         = st.slider("Kills",           0, 30, 2)
        damage        = st.slider("Damage Dealt",    0, 3000, 200)
        assists       = st.slider("Assists",         0, 15, 0)
        headshotKills = st.slider("Headshot Kills",  0, 20, 0)
        DBNOs         = st.slider("DBNOs",           0, 20, 0)
        killStreaks    = st.slider("Kill Streaks",    0, 10, 0)
        longestKill   = st.slider("Longest Kill (m)",0, 1000, 0)
        killPlace     = st.slider("Kill Place",      1, 100, 50)

    with col2:
        st.subheader("Movement")
        walkDistance  = st.slider("Walk Distance (m)",  0, 10000, 1500)
        rideDistance  = st.slider("Ride Distance (m)",  0, 20000, 0)
        swimDistance  = st.slider("Swim Distance (m)",  0, 2000, 0)

        st.subheader("Survival")
        boosts        = st.slider("Boosts",  0, 20, 2)
        heals         = st.slider("Heals",   0, 20, 1)
        revives       = st.slider("Revives", 0, 10, 0)

    with col3:
        st.subheader("Match Info")
        weaponsAcquired = st.slider("Weapons Acquired",  0, 20, 3)
        matchDuration   = st.slider("Match Duration (s)", 600, 2400, 1800)
        maxPlace        = st.slider("Max Place",          1, 100, 90)
        numGroups       = st.slider("Num Groups",         1, 100, 50)
        teamKills       = st.slider("Team Kills",         0, 5, 0)
        roadKills       = st.slider("Road Kills",         0, 5, 0)
        vehicleDestroys = st.slider("Vehicle Destroys",   0, 5, 0)

    if st.button("🎯 Predict", key="manual_predict"):
        user_input = dict(kills=kills, damageDealt=damage, assists=assists,
                          headshotKills=headshotKills, DBNOs=DBNOs,
                          killStreaks=killStreaks, longestKill=longestKill,
                          killPlace=killPlace, walkDistance=walkDistance,
                          rideDistance=rideDistance, swimDistance=swimDistance,
                          boosts=boosts, heals=heals, revives=revives,
                          weaponsAcquired=weaponsAcquired, matchDuration=matchDuration,
                          maxPlace=maxPlace, numGroups=numGroups,
                          teamKills=teamKills, roadKills=roadKills,
                          vehicleDestroys=vehicleDestroys)

        X = build_manual_features(user_input, feature_cols)
        pred = float(np.clip(model.predict(X)[0], 0, 1))

        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Predicted Win Placement", f"{pred:.3f}")
            st.metric("Top %", f"{(1 - pred) * 100:.1f}%")
        with col_b:
            if pred >= 0.8:
                st.success("🏆 Top 20% — Excellent performance!")
            elif pred >= 0.6:
                st.info("👍 Top 40% — Good performance.")
            elif pred >= 0.4:
                st.warning("⚠️ Middle of the pack.")
            else:
                st.error("💀 Bottom 40% — Tough match.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Player Lookup (NEW Day 12)
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🔍 Player Lookup — Real Data from Kaggle Dataset")
    st.markdown(
        "Enter a **groupId** from the PUBG Kaggle dataset to get fully accurate predictions "
        "with complete match context, and see how the player's playstyle evolved across matches."
    )

    # Load lookup data (cached)
    df_lookup = load_lookup_data()

    # Show some sample groupIds to help users try it
    with st.expander("📋 Don't have a groupId? Click here for samples"):
        sample_ids = df_lookup['groupId'].value_counts()
        sample_ids = sample_ids[sample_ids >= 2].index[:20].tolist()
        st.write("Copy one of these groupIds to try:")
        st.dataframe(pd.DataFrame({'groupId': sample_ids, 'matches_in_sample': [
            df_lookup[df_lookup['groupId'] == g].shape[0] for g in sample_ids
        ]}))

    group_id_input = st.text_input("Enter groupId:", placeholder="e.g. 0a7e4a00b8c3d2f1...")

    if st.button("🔍 Look Up Player", key="lookup_btn") and group_id_input.strip():
        gid = group_id_input.strip()
        rows = df_lookup[df_lookup['groupId'] == gid].copy()

        if rows.empty:
            st.error(f"groupId `{gid}` not found in the lookup dataset. "
                     f"Try one of the sample IDs above.")
        else:
            st.success(f"Found {len(rows)} match(es) for this group.")

            # ── Predictions ──────────────────────────────────────────────
            st.subheader("📊 Predictions vs Actual Placement")
            valid_features = [c for c in feature_cols if c in rows.columns]
            X = rows[valid_features].fillna(0).values
            preds = np.clip(model.predict(X), 0, 1)
            rows = rows.reset_index(drop=True)
            rows['predicted_winPlacePerc'] = preds
            rows['error'] = abs(rows['winPlacePerc'] - rows['predicted_winPlacePerc'])

            display_cols = ['matchId', 'kills', 'damageDealt', 'walkDistance',
                            'boosts', 'heals', 'weaponsAcquired',
                            'winPlacePerc', 'predicted_winPlacePerc', 'error']
            display_cols = [c for c in display_cols if c in rows.columns]

            st.dataframe(rows[display_cols].style.format({
                'winPlacePerc': '{:.3f}',
                'predicted_winPlacePerc': '{:.3f}',
                'error': '{:.3f}',
                'damageDealt': '{:.0f}',
                'walkDistance': '{:.0f}',
            }), use_container_width=True)

            avg_error = rows['error'].mean()
            st.metric("Average Prediction Error (MAE)", f"{avg_error:.4f}")

            # ── Playstyle Evolution ───────────────────────────────────────
            st.subheader("🎭 Playstyle Evolution")
            evo_df, evo_summary = get_playstyle_evolution(gid, df_lookup)

            if evo_summary:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Matches Found",      evo_summary['num_matches'])
                m2.metric("Avg Placement",      f"{evo_summary['avg_placement']:.3f}")
                m3.metric("Avg Kills",          f"{evo_summary['avg_kills']:.1f}")
                m4.metric("Dominant Playstyle", evo_summary['dominant_playstyle'])

                st.markdown("**Best Match:**")
                bm = evo_summary['best_match']
                st.write(f"Match `{bm['matchId']}` — "
                         f"Kills: {bm['kills']}, Damage: {bm['damageDealt']:.0f}, "
                         f"Placement: {bm['winPlacePerc']:.3f}, Style: {bm['playstyle']}")

                st.markdown("**Worst Match:**")
                wm = evo_summary['worst_match']
                st.write(f"Match `{wm['matchId']}` — "
                         f"Kills: {wm['kills']}, Damage: {wm['damageDealt']:.0f}, "
                         f"Placement: {wm['winPlacePerc']:.3f}, Style: {wm['playstyle']}")

                if len(evo_df) >= 2:
                    fig = plot_evolution(gid, evo_df, evo_summary)
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.info("Only 1 match found for this group — evolution chart needs 2+ matches.")

    gc.collect()