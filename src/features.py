"""
features.py
===========
Complete feature-engineering pipeline for the PUBG Win Placement
Prediction project.

Usage
-----
from src.features import build_features, reduce_mem_usage, EXCLUDE_COLS

df = pd.read_csv("train_V2.csv")
df = reduce_mem_usage(df)
df = build_features(df)
"""

from __future__ import annotations

import gc
import warnings
from typing import List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Columns that must never be model features ─────────────────────────────────
EXCLUDE_COLS: List[str] = [
    "Id", "groupId", "matchId", "matchType",
    "winPlacePerc", "is_cheater", "is_cheater_rule", "is_cheater_iso",
]

KEEP_STRING_COLS: List[str] = ["matchId", "groupId"]


# ─────────────────────────────────────────────────────────────────────────────
def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Downcast numeric columns to reduce RAM footprint.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe to compress.
    verbose : bool
        Print memory saving summary.

    Returns
    -------
    pd.DataFrame
        Same dataframe with reduced dtypes.
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type).startswith("int"):
                for dtype in [np.int8, np.int16, np.int32, np.int64]:
                    if c_min > np.iinfo(dtype).min and c_max < np.iinfo(dtype).max:
                        df[col] = df[col].astype(dtype)
                        break
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    if verbose:
        print(f"Memory: {start_mem:.1f} MB → {end_mem:.1f} MB "
              f"({100*(start_mem-end_mem)/start_mem:.1f}% reduction)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
def _safe_div(a: pd.Series, b: pd.Series, fill: float = 0.0) -> pd.Series:
    """Element-wise division with zero-denominator protection."""
    return np.where(b == 0, fill, a / b)


# ─────────────────────────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the complete 105-feature engineering pipeline.

    The function mutates *df* in-place and also returns it for
    convenience in chaining.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame after ``reduce_mem_usage`` has been applied.
        Must contain the 29 original PUBG columns.

    Returns
    -------
    pd.DataFrame
        Same dataframe with ~105 new feature columns appended.
    """

    # ── 1. Cheater flags ─────────────────────────────────────────────────────
    df["is_cheater_rule"] = (
        (df["kills"] > 30) | (df["damageDealt"] > 6000) | (df["headshotKills"] > 20)
    ).astype(np.int8)
    df["is_cheater"] = df["is_cheater_rule"]

    # ── 2. Combat features ───────────────────────────────────────────────────
    df["damagePerKill"]       = _safe_div(df["damageDealt"], df["kills"])
    df["headshotRate"]        = _safe_div(df["headshotKills"], df["kills"])
    df["vehicleKillRate"]     = _safe_div(df["roadKills"], df["kills"])
    df["killToDamageRatio"]   = _safe_div(df["kills"], df["damageDealt"])
    df["combatInvolvement"]   = df["kills"] + df["assists"] + df["DBNOs"]
    df["longRangeKillRate"]   = _safe_div(
        (df["longestKill"] > 100).astype(float), df["kills"]
    )

    # ── 3. Movement features ─────────────────────────────────────────────────
    df["totalDistance"]      = df["walkDistance"] + df["rideDistance"] + df["swimDistance"]
    df["walkRatio"]          = _safe_div(df["walkDistance"], df["totalDistance"])
    df["rideRatio"]          = _safe_div(df["rideDistance"], df["totalDistance"])
    df["killsPerTotalDist"]  = _safe_div(df["kills"], df["totalDistance"] + 1)
    df["killsPerDistance"]   = df["killsPerTotalDist"]
    df["weaponsPerDistance"] = _safe_div(df["weaponsAcquired"], df["totalDistance"] + 1)
    df["damagePerDistance"]  = _safe_div(df["damageDealt"], df["totalDistance"] + 1)

    # ── 4. Survival proxies (no timeSurvived) ────────────────────────────────
    df["killPlaceNorm"]  = _safe_div(df["killPlace"], df["maxPlace"])
    df["survivalRatio"]  = 1 - df["killPlaceNorm"]
    df["survivalInverse"]= _safe_div(1.0, df["killPlace"] + 1)
    df["boostIntensity"] = df["boosts"] * df["heals"]
    df["survivalScore"]  = df["walkDistance"] * 0.5 + df["boosts"] * 10 + df["heals"] * 5

    # ── 5. Resource / looting features ───────────────────────────────────────
    df["totalHealing"]        = df["heals"] + df["boosts"]
    df["healsPerKill"]        = _safe_div(df["heals"], df["kills"] + 1)
    df["boostHealRatio"]      = _safe_div(df["boosts"], df["totalHealing"] + 1)
    df["weaponsAcquiredLog"]  = np.log1p(df["weaponsAcquired"])
    df["healsPerWeapon"]      = _safe_div(df["heals"], df["weaponsAcquired"] + 1)
    df["boostsPerWeapon"]     = _safe_div(df["boosts"], df["weaponsAcquired"] + 1)
    df["crateHunterScore"]    = df["weaponsAcquired"] + df["boosts"] * 2
    df["looting_efficiency"]  = _safe_div(
        df["weaponsAcquired"] + df["heals"], df["totalDistance"] + 1
    )
    df["healerScore"]         = df["heals"] * 2 + df["revives"] * 5

    # ── 6. Day 4 match-type flags ────────────────────────────────────────────
    if "matchType" in df.columns:
        df["is_solo"]  = df["matchType"].str.contains("solo",  case=False, na=False).astype(np.int8)
        df["is_duo"]   = df["matchType"].str.contains("duo",   case=False, na=False).astype(np.int8)
        df["is_squad"] = df["matchType"].str.contains("squad", case=False, na=False).astype(np.int8)
    else:
        df["is_solo"] = df["is_duo"] = df["is_squad"] = 0

    # ── 7. Lobby-size buckets ────────────────────────────────────────────────
    df["lobbySize_small"]  = (df["maxPlace"] < 30).astype(np.int8)
    df["lobbySize_medium"] = ((df["maxPlace"] >= 30) & (df["maxPlace"] < 70)).astype(np.int8)
    df["lobbySize_large"]  = (df["maxPlace"] >= 70).astype(np.int8)

    # ── 8. Composite scores ──────────────────────────────────────────────────
    df["expectedWinRate"]    = _safe_div(1.0, df["numGroups"])
    df["survival_mobility"]  = df["walkDistance"] * df["boosts"]
    df["combat_dominance"]   = df["kills"] * df["damageDealt"]
    df["damagePerKillClean"] = df["damagePerKill"].clip(0, 2000)

    # ── 9. Group (team) aggregations ─────────────────────────────────────────
    for col, aggs in {
        "kills":        ["sum", "mean", "max", "std"],
        "damageDealt":  ["sum", "mean", "max", "std"],
        "walkDistance": ["sum", "mean", "max", "std"],
    }.items():
        grp = df.groupby("groupId")[col]
        for agg in aggs:
            fname = f"grp_{col.replace('Dealt','').replace('Distance','walkDistance')}"
            # keep naming consistent
            short = col if col != "damageDealt" else "damage"
            short = short if short != "walkDistance" else "walkDistance"
            df[f"grp_{short}_{agg}"] = grp.transform(agg)

    df["teamSize"]            = df.groupby("groupId")["Id"].transform("count") if "Id" in df.columns else 1
    df["killShareInTeam"]     = _safe_div(df["kills"],        df["grp_kills_sum"])
    df["damageShareInTeam"]   = _safe_div(df["damageDealt"],  df["grp_damage_sum"])
    df["distShareInTeam"]     = _safe_div(df["walkDistance"], df["grp_walkDistance_sum"])
    df["isBestKillerInTeam"]  = (df["kills"]        == df["grp_kills_max"]).astype(np.int8)
    df["isBestDamagerInTeam"] = (df["damageDealt"]  == df["grp_damage_max"]).astype(np.int8)

    # ── 10. Match normalizations ─────────────────────────────────────────────
    for col, fname in [
        ("kills",       "killsVsMatchAvg"),
        ("damageDealt", "damageVsMatchAvg"),
        ("walkDistance","distanceVsMatchAvg"),
    ]:
        match_mean = df.groupby("matchId")[col].transform("mean")
        df[fname] = _safe_div(df[col], match_mean + 1e-5)

    df["killRankInMatch"]    = df.groupby("matchId")["kills"].rank(ascending=False, method="min")
    df["damageRankInMatch"]  = df.groupby("matchId")["damageDealt"].rank(ascending=False, method="min")
    df["match_playerCount"]  = df.groupby("matchId")["matchId"].transform("count")

    # ── 11. Percentile ranks within match ────────────────────────────────────
    pct_cols = {
        "kills":        "killPercentileInMatch",
        "damageDealt":  "damagePercentileInMatch",
        "walkDistance": "distancePercentileInMatch",
        "totalHealing": "healingPercentileInMatch",
        "boosts":       "boostPercentileInMatch",
        "walkDistance": "walkPercInMatch",
        "killPlace":    "killPlacePercentileInMatch",
    }
    for src, dst in pct_cols.items():
        df[dst] = df.groupby("matchId")[src].rank(pct=True)

    # ── 12. Team survival features ───────────────────────────────────────────
    df["teamTotalHealing"] = df.groupby("groupId")["totalHealing"].transform("sum")
    df["teamAvgDistance"]  = df.groupby("groupId")["totalDistance"].transform("mean")
    df["teamSurvivalScore"]= df.groupby("groupId")["survivalScore"].transform("sum")
    df["teamBoosts"]       = df.groupby("groupId")["boosts"].transform("sum")
    df["teamBestSurvival"] = df.groupby("groupId")["survivalScore"].transform("max")
    df["playerSurvivalRank"] = df.groupby("groupId")["survivalScore"].rank(ascending=False, method="min")

    # ── 13. Log transforms ───────────────────────────────────────────────────
    df["damageDealt_log"]  = np.log1p(df["damageDealt"])
    df["totalDistance_log"]= np.log1p(df["totalDistance"])
    df["kills_log"]        = np.log1p(df["kills"])
    df["walkDistance_log"] = np.log1p(df["walkDistance"])

    # ── 14. Clipped features ─────────────────────────────────────────────────
    df["kills_clipped"]    = df["kills"].clip(0, 20)
    df["damage_clipped"]   = df["damageDealt"].clip(0, 3000)
    df["distance_clipped"] = df["totalDistance"].clip(0, 10000)

    gc.collect()
    return df


# ─────────────────────────────────────────────────────────────────────────────
def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Return the list of model-ready feature column names.

    Call this *after* ``build_features`` has been run.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe that has gone through ``build_features``.

    Returns
    -------
    List[str]
        Columns safe to pass to the LightGBM model.
    """
    return [
        c for c in df.columns
        if c not in EXCLUDE_COLS
        and df[c].dtype != object
    ]
