"""
day6_scouting_report.py
=======================
Reusable PUBG Scouting Report module — built on Day 6.
Used by Day 9 Streamlit frontend and Day 8 FastAPI backend.

Usage:
    from day6_scouting_report import generate_scouting_report, classify_archetype
    report = generate_scouting_report(stats_dict, player_name="YourName")
    print(report["full_text"])
"""

import numpy as np
import pandas as pd
import pickle
import os

# ── Constants ────────────────────────────────────────────────
ARCHETYPE_COLORS = {
    "Aggressive Rusher": "#ff6b35",
    "Survivalist":       "#4fc3f7",
    "Sniper":            "#69f0ae",
    "Support":           "#f5c518",
    "Balanced":          "#9e9e9e",
}

ARCHETYPE_DESCRIPTIONS = {
    "Aggressive Rusher": "High kills + high damage + compact movement",
    "Survivalist":       "High distance + consistent healing + low kills",
    "Sniper":            "Long-range kills + high headshot rate",
    "Support":           "High revives + heals + low kill count",
    "Balanced":          "Versatile — no single extreme stat",
}


def _percentile_from_array(value: float, array: np.ndarray) -> float:
    """Return 0-100 percentile of value within array."""
    return float(np.mean(array <= value) * 100)


def classify_archetype(stats: dict, pct_cache: dict) -> tuple:
    """
    Classify player into one of 5 archetypes using raw stats + percentile cache.
    
    Args:
        stats:     dict of raw PUBG player stats
        pct_cache: dict mapping stat names to numpy arrays for percentile computation
    
    Returns:
        (archetype_name: str, emoji: str)
    """
    eps = 1e-6
    kills   = stats.get("kills", 0)
    damage  = stats.get("damageDealt", 0)
    walk    = stats.get("walkDistance", 0)
    ride    = stats.get("rideDistance", 0)
    swim    = stats.get("swimDistance", 0)
    heals   = stats.get("heals", 0)
    boosts  = stats.get("boosts", 0)
    lk      = stats.get("longestKill", 0)
    revives = stats.get("revives", 0)
    hs_kills= stats.get("headshotKills", 0)
    hs_rate = hs_kills / (kills + eps)
    totalDist = walk + ride + swim

    kill_pct  = _percentile_from_array(kills,   pct_cache.get("kills",   np.array([0])))
    dmg_pct   = _percentile_from_array(damage,  pct_cache.get("damageDealt", np.array([0])))
    dist_pct  = _percentile_from_array(totalDist, pct_cache.get("walkDistance", np.array([0])))
    heal_pct  = _percentile_from_array(heals,   pct_cache.get("heals",   np.array([0])))
    boost_pct = _percentile_from_array(boosts,  pct_cache.get("boosts",  np.array([0])))
    lk_pct    = _percentile_from_array(lk,      pct_cache.get("longestKill", np.array([0])))
    hs_pct    = _percentile_from_array(hs_rate, pct_cache.get("headshotKills", np.array([0])))
    rev_pct   = _percentile_from_array(revives, pct_cache.get("revives", np.array([0])))

    if kill_pct > 70 and dmg_pct > 65 and dist_pct < 55:
        return "Aggressive Rusher", "🔫"
    elif dist_pct > 70 and kill_pct < 45 and (heal_pct + boost_pct) / 2 > 55:
        return "Survivalist", "🏃"
    elif lk_pct > 80 and hs_pct > 65:
        return "Sniper", "🎯"
    elif rev_pct > 75 and (heal_pct + boost_pct) / 2 > 65 and kill_pct < 40:
        return "Support", "🛡️"
    else:
        return "Balanced", "⚖️"


def generate_scouting_report(
    stats_dict: dict,
    model,
    feature_cols: list,
    pct_cache: dict,
    player_name: str = "Player",
) -> dict:
    """
    Generate a complete PUBG scouting report for one player.

    Args:
        stats_dict:   dict of raw PUBG stats (kills, damageDealt, etc.)
        model:        loaded LightGBM model (pubg_model_v5.pkl)
        feature_cols: list of feature names (day5_feature_cols.pkl)
        pct_cache:    dict mapping stat names → np.array for percentile scoring
        player_name:  display name string

    Returns:
        dict with keys:
            player_name, archetype, emoji, summary,
            predicted_winPlacePerc, percentiles, tips, full_text
    """
    eps = 1e-6
    s = {k: float(v) for k, v in stats_dict.items()}

    defaults = dict(kills=0, assists=0, damageDealt=0, headshotKills=0,
                    heals=0, boosts=0, walkDistance=500, rideDistance=0,
                    swimDistance=0, weaponsAcquired=3, longestKill=0,
                    killPlace=50, maxPlace=100, roadKills=0, revives=0,
                    matchDuration=1800, teamSize=1, match_playerCount=100)
    for k, v in defaults.items():
        s.setdefault(k, v)

    totalDist = s["walkDistance"] + s["rideDistance"] + s["swimDistance"]
    totalHeal = s["heals"] + s["boosts"]

    # Build feature vector
    row = _build_feature_row_static(s, totalDist, totalHeal, pct_cache)
    feat_df = pd.DataFrame([row])
    for col in feature_cols:
        if col not in feat_df.columns:
            feat_df[col] = 0.0
    feat_df = feat_df[feature_cols]
    feat_df.replace([float("inf"), float("-inf")], float("nan"), inplace=True)
    feat_df.fillna(0, inplace=True)
    feat_df = feat_df.astype(np.float32)

    pred = float(np.clip(model.predict(feat_df)[0], 0, 1))

    # Archetype
    archetype, emoji = classify_archetype(s, pct_cache)

    # Percentiles
    def _pct(val, key):
        return round(_percentile_from_array(val, pct_cache.get(key, np.array([0]))))

    kills   = s["kills"]
    damage  = s["damageDealt"]
    heals   = s["heals"]
    boosts  = s["boosts"]
    lk      = s["longestKill"]
    revives = s["revives"]

    percentiles = {
        "kills":        _pct(kills,       "kills"),
        "damage":       _pct(damage,      "damageDealt"),
        "distance":     _pct(totalDist,   "walkDistance"),
        "healing":      _pct(heals,       "heals"),
        "boosts":       _pct(boosts,      "boosts"),
        "longest_kill": _pct(lk,          "longestKill"),
        "revives":      _pct(revives,     "revives"),
    }

    def _word(p):
        if p >= 90: return "elite"
        if p >= 75: return "top-tier"
        if p >= 55: return "above average"
        if p >= 40: return "average"
        if p >= 20: return "below average"
        return "low"

    # Summary
    summaries = {
        "Aggressive Rusher": (
            f"{player_name} is a high-octane fighter who charges into combat early and often. "
            f"With {_word(percentiles['kills'])} kill output and {_word(percentiles['damage'])} "
            f"damage dealt, this player is a constant threat in gunfights. "
            f"Winning comes through eliminating opponents, not avoiding them."
        ),
        "Survivalist": (
            f"{player_name} is a patient, zone-aware player who prioritizes staying alive over fighting. "
            f"With {_word(percentiles['distance'])} movement distance and consistent healing, "
            f"this player excels at outlasting opponents. Deep placement is the specialty."
        ),
        "Sniper": (
            f"{player_name} is a precision marksman winning engagements from range. "
            f"The {_word(percentiles['longest_kill'])} longest kill and high headshot accuracy "
            f"signal excellent positioning and aim. High ground and zone edges are home territory."
        ),
        "Support": (
            f"{player_name} is the backbone of their squad. "
            f"With {_word(percentiles['revives'])} revive numbers and strong healing output, "
            f"this player keeps teammates alive long past when others would fall. "
            f"Best in squad formats where team survival extends to late circles."
        ),
        "Balanced": (
            f"{player_name} is a versatile all-rounder with no glaring weaknesses. "
            f"Predicted placement of {pred:.1%} reflects solid fundamentals across "
            f"combat, movement, and survival. Sharpening one specialty could break into the top tier."
        ),
    }

    summary = summaries.get(archetype, summaries["Balanced"])

    # Tips
    tips = []
    if percentiles["kills"] < 40:
        tips.append("💡 Practice aim — kills are below average. Hot-drop more for gunfight reps.")
    if percentiles["distance"] < 30:
        tips.append("💡 Move more — low distance means bad rotations. Walk the blue zone edge.")
    if percentiles["healing"] < 35:
        tips.append("💡 Pick up more heals — you are under-healing, costing you late-game fights.")
    if percentiles["boosts"] < 30:
        tips.append("💡 Use energy drinks and painkillers — passive heal and speed in final circles.")
    if percentiles["longest_kill"] < 25:
        tips.append("💡 Engage from farther — long-range kills improve positioning habits.")
    if percentiles["damage"] < 35:
        tips.append("💡 Deal more chip damage — assists help your team even without finishing kills.")
    if not tips:
        tips.append("🏆 Well-rounded stats! Focus on consistency and closing out top-5 situations.")

    win_label = f"Predicted finish: top {100 - round(pred * 100)}% (winPlacePerc = {pred:.3f})"

    full_text = f"""
╔══════════════════════════════════════════════════════════╗
  🎮  PUBG SCOUTING REPORT  —  {player_name}
╚══════════════════════════════════════════════════════════╝

  ARCHETYPE:   {emoji} {archetype}
  PREDICTION:  {win_label}

──────────────────────────────────────────────────────────
  PLAYER PROFILE
──────────────────────────────────────────────────────────
  {summary}

──────────────────────────────────────────────────────────
  PERCENTILE BREAKDOWN  (vs dataset)
──────────────────────────────────────────────────────────
  Kills         : top {100-percentiles["kills"]:>3}%  ({_word(percentiles["kills"])})
  Damage        : top {100-percentiles["damage"]:>3}%  ({_word(percentiles["damage"])})
  Distance      : top {100-percentiles["distance"]:>3}%  ({_word(percentiles["distance"])})
  Healing       : top {100-percentiles["healing"]:>3}%  ({_word(percentiles["healing"])})
  Boosts        : top {100-percentiles["boosts"]:>3}%  ({_word(percentiles["boosts"])})
  Longest Kill  : top {100-percentiles["longest_kill"]:>3}%  ({_word(percentiles["longest_kill"])})
  Revives       : top {100-percentiles["revives"]:>3}%  ({_word(percentiles["revives"])})

──────────────────────────────────────────────────────────
  IMPROVEMENT TIPS
──────────────────────────────────────────────────────────
{"  " + chr(10) + "  ".join(tips)}

══════════════════════════════════════════════════════════
"""
    return {
        "player_name":            player_name,
        "archetype":              archetype,
        "emoji":                  emoji,
        "archetype_description":  ARCHETYPE_DESCRIPTIONS.get(archetype, ""),
        "archetype_color":        ARCHETYPE_COLORS.get(archetype, "#888888"),
        "summary":                summary,
        "predicted_winPlacePerc": round(pred, 4),
        "percentiles":            percentiles,
        "tips":                   tips,
        "full_text":              full_text,
    }


def _build_feature_row_static(s, totalDist, totalHeal, pct_cache):
    """Build engineered feature dict from raw stats. Static version for module use."""
    eps = 1e-6

    def _pct_val(v, k):
        return _percentile_from_array(v, pct_cache.get(k, np.array([0]))) / 100

    return {
        "kills": s["kills"], "assists": s["assists"],
        "damageDealt": s["damageDealt"], "headshotKills": s["headshotKills"],
        "heals": s["heals"], "boosts": s["boosts"],
        "walkDistance": s["walkDistance"], "rideDistance": s["rideDistance"],
        "swimDistance": s["swimDistance"], "weaponsAcquired": s["weaponsAcquired"],
        "longestKill": s["longestKill"], "killPlace": s["killPlace"],
        "maxPlace": s["maxPlace"], "roadKills": s["roadKills"],
        "revives": s["revives"], "matchDuration": s["matchDuration"],
        "DBNOs": s.get("DBNOs", 0), "killPoints": s.get("killPoints", 0),
        "killStreaks": s.get("killStreaks", 0), "rankPoints": s.get("rankPoints", 0),
        "teamKills": s.get("teamKills", 0), "vehicleDestroys": s.get("vehicleDestroys", 0),
        "winPoints": s.get("winPoints", 0),
        "damagePerKill":      s["damageDealt"] / (s["kills"] + eps),
        "headshotRate":       s["headshotKills"] / (s["kills"] + eps),
        "killsPerDistance":   s["kills"] / (s["walkDistance"] + eps),
        "vehicleKillRate":    s["roadKills"] / (s["kills"] + eps),
        "damagePerDistance":  s["damageDealt"] / (s["walkDistance"] + eps),
        "combatInvolvement":  (s["kills"] + s["assists"]) / (s["maxPlace"] + eps),
        "killToDamageRatio":  s["kills"] / (s["damageDealt"] + eps),
        "longRangeKillRate":  s["longestKill"] / (s["kills"] + eps),
        "totalDistance":      totalDist,
        "walkRatio":          s["walkDistance"] / (totalDist + eps),
        "rideRatio":          s["rideDistance"] / (totalDist + eps),
        "weaponsPerDistance": s["weaponsAcquired"] / (totalDist + eps),
        "survivalScore":      s["boosts"] + s["heals"] + s["walkDistance"] / 1000,
        "killsPerTotalDist":  s["kills"] / (totalDist + eps),
        "survivalRatio":      1 - s["killPlace"] / (s["maxPlace"] + eps),
        "survivalInverse":    s["maxPlace"] - s["killPlace"],
        "killPlaceNorm":      s["killPlace"] / (s["maxPlace"] + eps),
        "boostIntensity":     s["boosts"] / (s["matchDuration"] / 60 + eps),
        "totalHealing":       totalHeal,
        "healsPerKill":       s["heals"] / (s["kills"] + eps),
        "boostHealRatio":     s["boosts"] / (totalHeal + eps),
        "weaponsAcquiredLog": np.log1p(s["weaponsAcquired"]),
        "healsPerWeapon":     s["heals"] / (s["weaponsAcquired"] + eps),
        "boostsPerWeapon":    s["boosts"] / (s["weaponsAcquired"] + eps),
        "crateHunterScore":   s["weaponsAcquired"] * totalDist / 1000,
        "grp_kills_sum": s["kills"], "grp_kills_mean": s["kills"],
        "grp_kills_max": s["kills"], "grp_kills_std": 0,
        "grp_damageDealt_sum": s["damageDealt"], "grp_damageDealt_mean": s["damageDealt"],
        "grp_damageDealt_max": s["damageDealt"], "grp_damageDealt_std": 0,
        "grp_walkDistance_sum": s["walkDistance"], "grp_walkDistance_mean": s["walkDistance"],
        "grp_walkDistance_max": s["walkDistance"], "grp_walkDistance_std": 0,
        "killShareInTeam": 1.0, "damageShareInTeam": 1.0, "distShareInTeam": 1.0,
        "isBestKillerInTeam": 1, "isBestDamagerInTeam": 1,
        "teamSize": s["teamSize"],
        "killsVsMatchAvg":    s["kills"] / 2,
        "damageVsMatchAvg":   s["damageDealt"] / 150,
        "distanceVsMatchAvg": totalDist / 1000,
        "killRankInMatch":    1 - s["killPlace"] / (s["maxPlace"] + eps),
        "damageRankInMatch":  0.5,
        "match_playerCount":  s["match_playerCount"],
        "is_solo":  int(s["teamSize"] == 1),
        "is_duo":   int(s["teamSize"] == 2),
        "is_squad": int(s["teamSize"] >= 3),
        "lobbySize_small":   int(s["match_playerCount"] < 50),
        "lobbySize_medium":  int(50 <= s["match_playerCount"] < 90),
        "lobbySize_large":   int(s["match_playerCount"] >= 90),
        "expectedWinRate":   1 / (s["match_playerCount"] + eps),
        "survival_mobility": totalDist * s["boosts"],
        "combat_dominance":  s["kills"] * s["damageDealt"],
        "looting_efficiency": s["weaponsAcquired"] / (totalDist + eps),
        "damagePerKillClean": s["damageDealt"] / s["kills"] if s["kills"] > 0 else 0,
        "healerScore":        s["heals"] * 2 + s["revives"] * 3,
        "killPercentileInMatch":      _pct_val(s["kills"],      "kills"),
        "damagePercentileInMatch":    _pct_val(s["damageDealt"],"damageDealt"),
        "distancePercentileInMatch":  _pct_val(s["walkDistance"],"walkDistance"),
        "healingPercentileInMatch":   _pct_val(totalHeal,       "heals"),
        "boostPercentileInMatch":     _pct_val(s["boosts"],     "boosts"),
        "walkPercInMatch":            _pct_val(s["walkDistance"],"walkDistance"),
        "killPlacePercentileInMatch": 1 - s["killPlace"] / (s["maxPlace"] + eps),
        "teamTotalHealing":   totalHeal,
        "teamAvgDistance":    totalDist,
        "teamSurvivalScore":  s["boosts"] + s["heals"] + s["walkDistance"] / 1000,
        "teamBoosts":         s["boosts"],
        "teamBestSurvival":   s["boosts"] + s["heals"] + s["walkDistance"] / 1000,
        "playerSurvivalRank": 1.0,
        "damageDealt_log":    np.log1p(s["damageDealt"]),
        "totalDistance_log":  np.log1p(totalDist),
        "kills_log":          np.log1p(s["kills"]),
        "walkDistance_log":   np.log1p(s["walkDistance"]),
        "kills_clipped":      min(s["kills"], 20),
        "damage_clipped":     min(s["damageDealt"], 1500),
        "distance_clipped":   min(totalDist, 10000),
    }
