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

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PUBG Win Predictor",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        border: 1px solid #333;
    }
    .metric-label {
        font-size: 13px;
        color: #aaa;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #fff;
    }
    .playstyle-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 14px;
    }
    .section-divider {
        border-top: 1px solid #333;
        margin: 24px 0;
    }
    .info-box {
        background: #1a2a3a;
        border-left: 4px solid #3498db;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

MODEL_FILE_ID   = "1rPxQVZ1u1gAEdftkAeAANXoGKFrAMxQM"
FEATURE_FILE_ID = "1j01b9zB43HqkvUqhvKlPxCzxHLcaz7Bo"
LOOKUP_FILE_ID  = "1btXkjuc3Q_5mqnhYn3_-E_ilSgKa7Abj"

# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
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


@st.cache_data(show_spinner="Loading player database...")
def load_lookup_data():
    import gdown
    if not os.path.exists("day12_lookup_dataset.parquet"):
        gdown.download(f"https://drive.google.com/uc?id={LOOKUP_FILE_ID}",
                       "day12_lookup_dataset.parquet", quiet=False)
    df = pd.read_parquet("day12_lookup_dataset.parquet")
    return df


# ── Helpers ───────────────────────────────────────────────────────────────────
STYLE_COLORS = {
    'Aggressive': '#e74c3c',
    'Passive':    '#3498db',
    'Sniper':     '#9b59b6',
    'Balanced':   '#2ecc71',
}

STYLE_EMOJI = {
    'Aggressive': '⚔️',
    'Passive':    '🛡️',
    'Sniper':     '🎯',
    'Balanced':   '⚖️',
}

def classify_playstyle(row):
    kills         = row['kills']
    damage        = row['damageDealt']
    walk          = row['walkDistance']
    headshot_rate = row.get('headshotRate', 0)
    if headshot_rate >= 0.5 and kills >= 2:
        return 'Sniper'
    elif kills >= 3 or damage >= 300:
        return 'Aggressive'
    elif walk >= 2000 and kills <= 1:
        return 'Passive'
    else:
        return 'Balanced'


def placement_label(p):
    if p >= 0.8:
        return "Top 20% — Excellent!", "success"
    elif p >= 0.6:
        return "Top 40% — Good performance.", "info"
    elif p >= 0.4:
        return "Middle of the pack.", "warning"
    else:
        return "Bottom 40% — Tough match.", "error"


def build_manual_features(user_input, feature_cols):
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
    row['totalDistance']    = td
    row['healsAndBoosts']   = row.get('heals', 0) + row.get('boosts', 0)
    row['killsAndAssists']  = row.get('kills', 0) + row.get('assists', 0)
    row['headshotRate']     = row.get('headshotKills', 0) / (row.get('kills', 0) + 1)
    row['killsPerDistance'] = row.get('kills', 0) / (td + 1)
    row['damagePerKill']    = row.get('damageDealt', 0) / (row.get('kills', 0) + 1)
    row['walkRatio']        = row.get('walkDistance', 0) / (td + 1)
    row['rideRatio']        = row.get('rideDistance', 0) / (td + 1)
    row['killsPerMatch']    = row.get('kills', 0) / (row.get('matchDuration', 1800) / 60 + 1)
    row['damagePerMinute']  = row.get('damageDealt', 0) / (row.get('matchDuration', 1800) / 60 + 1)

    return np.array([[row.get(c, 0.0) for c in feature_cols]])


def get_playstyle_evolution(group_id, df):
    rows = df[df['groupId'] == group_id].copy().reset_index(drop=True)
    if rows.empty:
        return None, None
    rows['playstyle']    = rows.apply(classify_playstyle, axis=1)
    rows['match_number'] = range(1, len(rows) + 1)

    unique_matches = rows.drop_duplicates(subset='matchId')
    best  = unique_matches.loc[unique_matches['winPlacePerc'].idxmax()]
    worst = unique_matches.loc[unique_matches['winPlacePerc'].idxmin()]

    summary = {
        'num_players':        len(rows),
        'num_matches':        unique_matches.shape[0],
        'avg_placement':      rows['winPlacePerc'].mean(),
        'avg_kills':          rows['kills'].mean(),
        'avg_damage':         rows['damageDealt'].mean(),
        'playstyle_counts':   rows['playstyle'].value_counts().to_dict(),
        'dominant_playstyle': rows['playstyle'].value_counts().index[0],
        'best_match':         best[['matchId','kills','damageDealt','winPlacePerc','playstyle']].to_dict(),
        'worst_match':        worst[['matchId','kills','damageDealt','winPlacePerc','playstyle']].to_dict(),
    }
    return rows, summary


def plot_evolution(group_id, evo_df, evo_summary):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.patch.set_facecolor('#0e1117')
    for ax in axes.flat:
        ax.set_facecolor('#1e1e2e')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333')

    fig.suptitle(f'Group Stats — {group_id[:14]}...', fontsize=13,
                 fontweight='bold', color='white')

    colors = [STYLE_COLORS.get(s, 'gray') for s in evo_df['playstyle']]
    x = evo_df['match_number']

    ax = axes[0, 0]
    ax.scatter(x, evo_df['winPlacePerc'], c=colors, s=100, zorder=5)
    ax.plot(x, evo_df['winPlacePerc'], color='#555', linewidth=1)
    ax.set_title('Placement (higher = better)')
    ax.set_xlabel('Player #'); ax.set_ylabel('winPlacePerc')
    ax.set_ylim(0, 1)

    ax = axes[0, 1]
    ax.bar(x, evo_df['kills'], color=colors)
    ax.set_title('Kills Per Player')
    ax.set_xlabel('Player #'); ax.set_ylabel('Kills')

    ax = axes[1, 0]
    ax.bar(x, evo_df['damageDealt'], color=colors, alpha=0.85)
    ax.set_title('Damage Per Player')
    ax.set_xlabel('Player #'); ax.set_ylabel('Damage')

    ax = axes[1, 1]
    counts = evo_summary['playstyle_counts']
    ax.pie(counts.values(),
           labels=counts.keys(),
           colors=[STYLE_COLORS.get(k, 'gray') for k in counts.keys()],
           autopct='%1.0f%%', startangle=90,
           textprops={'color': 'white'})
    ax.set_title('Playstyle Distribution')

    legend_elements = [Patch(facecolor=v, label=f"{STYLE_EMOJI.get(k,'')} {k}")
                       for k, v in STYLE_COLORS.items()]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=4, fontsize=9, facecolor='#1e1e2e',
               labelcolor='white', edgecolor='#333')

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    return fig


# ── App Header ────────────────────────────────────────────────────────────────
st.markdown("# PUBG Win Placement Predictor")
st.markdown("*Predict match placement using a LightGBM model trained on 1M+ PUBG matches.*")
st.markdown("---")

# ── Load resources ────────────────────────────────────────────────────────────
try:
    model, feature_cols = load_model_and_features()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Manual Input", "Player Lookup"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Manual Input
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Enter Your Match Stats")
    st.markdown(
        '<div class="info-box">Manual input fills aggregate match features with 0, '
        'so predictions lean toward 0.4–0.6. For fully accurate predictions use the '
        '<b>Player Lookup</b> tab.</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Combat")
        kills         = st.slider("Kills",            0, 30, 2)
        damage        = st.slider("Damage Dealt",     0, 3000, 200)
        assists       = st.slider("Assists",          0, 15, 0)
        headshotKills = st.slider("Headshot Kills",   0, 20, 0)
        DBNOs         = st.slider("DBNOs",            0, 20, 0)
        killStreaks    = st.slider("Kill Streaks",     0, 10, 0)
        longestKill   = st.slider("Longest Kill (m)", 0, 1000, 0)
        killPlace     = st.slider("Kill Place",       1, 100, 50)

    with col2:
        st.markdown("#### Movement")
        walkDistance = st.slider("Walk Distance (m)", 0, 10000, 1500)
        rideDistance = st.slider("Ride Distance (m)", 0, 20000, 0)
        swimDistance = st.slider("Swim Distance (m)", 0, 2000, 0)

        st.markdown("#### Survival")
        boosts  = st.slider("Boosts",  0, 20, 2)
        heals   = st.slider("Heals",   0, 20, 1)
        revives = st.slider("Revives", 0, 10, 0)

    with col3:
        st.markdown("#### Match Info")
        weaponsAcquired = st.slider("Weapons Acquired",   0, 20, 3)
        matchDuration   = st.slider("Match Duration (s)", 600, 2400, 1800)
        maxPlace        = st.slider("Max Place",          1, 100, 90)
        numGroups       = st.slider("Num Groups",         1, 100, 50)
        teamKills       = st.slider("Team Kills",         0, 5, 0)
        roadKills       = st.slider("Road Kills",         0, 5, 0)
        vehicleDestroys = st.slider("Vehicle Destroys",   0, 5, 0)

    st.markdown("---")

    if st.button("Predict My Placement", use_container_width=True):
        try:
            user_input = dict(
                kills=kills, damageDealt=damage, assists=assists,
                headshotKills=headshotKills, DBNOs=DBNOs,
                killStreaks=killStreaks, longestKill=longestKill,
                killPlace=killPlace, walkDistance=walkDistance,
                rideDistance=rideDistance, swimDistance=swimDistance,
                boosts=boosts, heals=heals, revives=revives,
                weaponsAcquired=weaponsAcquired, matchDuration=matchDuration,
                maxPlace=maxPlace, numGroups=numGroups,
                teamKills=teamKills, roadKills=roadKills,
                vehicleDestroys=vehicleDestroys
            )
            X    = build_manual_features(user_input, feature_cols)
            pred = float(np.clip(model.predict(X)[0], 0, 1))
            label, level = placement_label(pred)

            fake_row = {
                'kills': kills, 'damageDealt': damage,
                'walkDistance': walkDistance,
                'headshotRate': headshotKills / (kills + 1)
            }
            style = classify_playstyle(fake_row)

            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted Placement", f"{pred:.3f}")
            c2.metric("Top %", f"{(1 - pred) * 100:.1f}%")
            c3.metric("Playstyle", style)

            if level == "success":
                st.success(label)
            elif level == "info":
                st.info(label)
            elif level == "warning":
                st.warning(label)
            else:
                st.error(label)

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Player Lookup
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Look Up a Real Player Group")
    st.markdown(
        '<div class="info-box">Enter a <b>groupId</b> from the PUBG Kaggle dataset. '
        'The model will use full match context for accurate predictions across all 105 features.</div>',
        unsafe_allow_html=True
    )

    try:
        df_lookup = load_lookup_data()
    except Exception as e:
        st.error(f"Failed to load lookup dataset: {e}")
        st.stop()

    with st.expander("Don't have a groupId? Click here for samples"):
        st.markdown("**Guaranteed examples for each playstyle:**")

        style_samples = {}
        for style in ['Aggressive', 'Sniper', 'Passive', 'Balanced']:
            df_lookup['_ps'] = df_lookup.apply(classify_playstyle, axis=1)
            match = df_lookup[df_lookup['_ps'] == style]
            if not match.empty:
                style_samples[style] = match['groupId'].iloc[0]

        for style, gid in style_samples.items():
            st.code(f"{style}: {gid}")

        st.markdown("---")
        st.markdown("**Or pick any groupId from the dataset:**")
        random_ids = df_lookup['groupId'].drop_duplicates().sample(10, random_state=42).tolist()
        st.dataframe(pd.DataFrame({'groupId': random_ids}), use_container_width=True)

    group_id_input = st.text_input(
        "Enter groupId:",
        placeholder="e.g. 6f0931849c42fc...",
        help="Copy a groupId from the samples above or from the Kaggle dataset."
    )

    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        lookup_clicked = st.button("Look Up", use_container_width=True)

    if lookup_clicked:
        if not group_id_input.strip():
            st.warning("Please enter a groupId first.")
        else:
            gid = group_id_input.strip()
            rows = df_lookup[df_lookup['groupId'] == gid].copy()

            if rows.empty:
                st.error(
                    f"groupId `{gid}` not found in the lookup dataset. "
                    "Try one of the sample IDs from the expander above."
                )
            else:
                st.success(f"Found {len(rows)} player(s) for this group.")

                # ── Predictions ───────────────────────────────────────────
                st.markdown("---")
                st.markdown("### Predictions vs Actual Placement")

                try:
                    valid_features = [c for c in feature_cols if c in rows.columns]
                    X     = rows[valid_features].fillna(0).values
                    preds = np.clip(model.predict(X), 0, 1)
                    rows  = rows.reset_index(drop=True)
                    rows['predicted_winPlacePerc'] = preds
                    rows['error']     = abs(rows['winPlacePerc'] - rows['predicted_winPlacePerc'])
                    rows['playstyle'] = rows.apply(classify_playstyle, axis=1)

                    display_cols = ['kills', 'damageDealt', 'walkDistance',
                                    'boosts', 'heals', 'weaponsAcquired',
                                    'playstyle', 'winPlacePerc',
                                    'predicted_winPlacePerc', 'error']
                    display_cols = [c for c in display_cols if c in rows.columns]

                    st.dataframe(
                        rows[display_cols].style.format({
                            'winPlacePerc':           '{:.3f}',
                            'predicted_winPlacePerc': '{:.3f}',
                            'error':                  '{:.3f}',
                            'damageDealt':            '{:.0f}',
                            'walkDistance':           '{:.0f}',
                        }).background_gradient(subset=['winPlacePerc'], cmap='RdYlGn'),
                        use_container_width=True
                    )

                    avg_error = rows['error'].mean()
                    avg_pred  = rows['predicted_winPlacePerc'].mean()
                    label, level = placement_label(avg_pred)

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Avg Predicted Placement", f"{avg_pred:.3f}")
                    m2.metric("Avg Prediction Error",    f"{avg_error:.4f}")
                    m3.metric("Actual Placement",        f"{rows['winPlacePerc'].mean():.3f}")

                    if level == "success":
                        st.success(label)
                    elif level == "info":
                        st.info(label)
                    elif level == "warning":
                        st.warning(label)
                    else:
                        st.error(label)

                except Exception as e:
                    st.error(f"Prediction error: {e}")

                # ── Playstyle Breakdown ───────────────────────────────────
                st.markdown("---")
                st.markdown("### Group Playstyle Breakdown")

                try:
                    evo_df, evo_summary = get_playstyle_evolution(gid, df_lookup)

                    if evo_summary:
                        dominant = evo_summary['dominant_playstyle']
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Players in Group",   evo_summary['num_players'])
                        m2.metric("Avg Placement",      f"{evo_summary['avg_placement']:.3f}")
                        m3.metric("Avg Kills",          f"{evo_summary['avg_kills']:.1f}")
                        m4.metric("Dominant Playstyle", dominant)

                        if evo_summary['num_matches'] > 1:
                            c1, c2 = st.columns(2)
                            with c1:
                                bm = evo_summary['best_match']
                                st.markdown("**Best Match:**")
                                st.write(f"Match `{bm['matchId']}` — "
                                         f"Kills: {bm['kills']}, "
                                         f"Damage: {bm['damageDealt']:.0f}, "
                                         f"Placement: {bm['winPlacePerc']:.3f}, "
                                         f"Style: {bm['playstyle']}")
                            with c2:
                                wm = evo_summary['worst_match']
                                st.markdown("**Worst Match:**")
                                st.write(f"Match `{wm['matchId']}` — "
                                         f"Kills: {wm['kills']}, "
                                         f"Damage: {wm['damageDealt']:.0f}, "
                                         f"Placement: {wm['winPlacePerc']:.3f}, "
                                         f"Style: {wm['playstyle']}")
                        else:
                            st.info("This group appears in 1 match in the dataset sample. "
                                    "Best/Worst match comparison requires 2+ matches.")

                        fig = plot_evolution(gid, evo_df, evo_summary)
                        st.pyplot(fig)
                        plt.close()

                except Exception as e:
                    st.error(f"Playstyle analysis error: {e}")

                gc.collect()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#666; font-size:13px;'>"
    "Built by <b>Nitya Thaker</b> &nbsp;·&nbsp; "
    "LightGBM &nbsp;·&nbsp; Trained on 1M+ PUBG matches &nbsp;·&nbsp; "
    "<a href='https://github.com/NityaThaker/pubg-win-prediction' "
    "style='color:#3498db;'>GitHub</a>"
    "</div>",
    unsafe_allow_html=True
)