"""
scouting_report.py
==================
Player archetype classification and scouting-report generator for the
PUBG Win Placement Prediction project.

Archetypes
----------
🔫 Aggressive Rusher  — High kills, low survival
🏃 Survivalist        — High distance, low kills
🎯 Sniper             — High longest-kill, high headshot rate
🛡️  Support            — High revives/heals, low kills
⚖️  Balanced           — Everything else

Usage
-----
from src.scouting_report import classify_archetype, generate_scouting_report

archetype = classify_archetype(stats_dict)
report    = generate_scouting_report(stats_dict, model, feature_cols)
print(report["summary"])
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
ARCHETYPES: Dict[str, Dict] = {
    "Aggressive Rusher": {
        "emoji": "🔫",
        "description": "Fights everything that moves. Lives fast, dies young.",
        "color": "#e74c3c",
    },
    "Survivalist": {
        "emoji": "🏃",
        "description": "Rotates constantly. Wins by outlasting, not outgunning.",
        "color": "#2ecc71",
    },
    "Sniper": {
        "emoji": "🎯",
        "description": "Long-range specialist. Picks targets from afar.",
        "color": "#3498db",
    },
    "Support": {
        "emoji": "🛡️",
        "description": "Team-first player. Revives and heals teammates.",
        "color": "#9b59b6",
    },
    "Balanced": {
        "emoji": "⚖️",
        "description": "Adaptable all-rounder. Solid across every category.",
        "color": "#f39c12",
    },
}

IMPROVEMENT_TIPS: Dict[str, List[str]] = {
    "Aggressive Rusher": [
        "Use boosts proactively — they add speed and buy time.",
        "Pick fights only when you have cover advantage.",
        "Loot faster; dying with no heals is avoidable.",
    ],
    "Survivalist": [
        "You rotate well — commit to more engagements in the final ring.",
        "Practice spray control at mid-range to convert more kills.",
        "Carry at least one grenade for zone-push situations.",
    ],
    "Sniper": [
        "Pair long-range play with a reliable close-range backup.",
        "Use suppressors to avoid giving away your position.",
        "Reposition after every shot in late game.",
    ],
    "Support": [
        "Balance heal resources — save some boosts for yourself.",
        "Deal damage even as a support; assists win chicken dinners too.",
        "Use vehicles for fast revive runs across open ground.",
    ],
    "Balanced": [
        "Identify your strongest stat and double down on it.",
        "Consistent play beats gambling — keep your current approach.",
        "Review your endgame rotations to squeeze out more top-10 finishes.",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
def classify_archetype(stats: Dict[str, float | int]) -> str:
    """Classify a player into one of the five PUBG archetypes.

    Parameters
    ----------
    stats : dict
        Raw PUBG stats dictionary with at least:
        kills, damageDealt, longestKill, headshotKills,
        walkDistance, rideDistance, swimDistance,
        revives, heals, boosts.

    Returns
    -------
    str
        One of: "Aggressive Rusher", "Survivalist", "Sniper",
        "Support", "Balanced".
    """
    kills       = float(stats.get("kills", 0))
    damage      = float(stats.get("damageDealt", 0))
    longest     = float(stats.get("longestKill", 0))
    hs_kills    = float(stats.get("headshotKills", 0))
    walk        = float(stats.get("walkDistance", 0))
    ride        = float(stats.get("rideDistance", 0))
    swim        = float(stats.get("swimDistance", 0))
    revives     = float(stats.get("revives", 0))
    heals       = float(stats.get("heals", 0))
    boosts      = float(stats.get("boosts", 0))

    total_dist  = walk + ride + swim
    hs_rate     = hs_kills / kills if kills > 0 else 0.0

    scores: Dict[str, float] = {
        "Aggressive Rusher": kills * 3 + damage / 100,
        "Survivalist":       total_dist / 200 + boosts,
        "Sniper":            longest / 50 + hs_rate * 10,
        "Support":           revives * 5 + heals * 2,
        "Balanced":          2.5,   # baseline — beaten only by a clear category
    }

    return max(scores, key=lambda k: scores[k])


# ─────────────────────────────────────────────────────────────────────────────
def generate_scouting_report(
    stats: Dict[str, float | int | str],
    model: Any,
    feature_cols: List[str],
) -> Dict[str, Any]:
    """Generate a full scouting report for one player.

    Parameters
    ----------
    stats : dict
        Raw PUBG stats for the player.
    model : LightGBM Booster
        Loaded prediction model.
    feature_cols : list of str
        Ordered feature column names.

    Returns
    -------
    dict with keys:
        archetype, emoji, description, color,
        winPlacePerc, percentile, rank,
        improvement_tips, summary, raw_stats
    """
    # Lazy import to avoid circular dependency at module load
    from src.predict import predict_single

    pred_result  = predict_single(stats, model, feature_cols)
    archetype    = classify_archetype(stats)
    arch_info    = ARCHETYPES[archetype]
    tips         = IMPROVEMENT_TIPS[archetype]

    win_pct      = pred_result["winPlacePerc"]
    percentile   = pred_result["percentile"]
    rank_label   = pred_result["rank"]

    summary = (
        f"{arch_info['emoji']} {archetype} | "
        f"Predicted finish: top {100 - percentile}% | "
        f"{rank_label}"
    )

    return {
        "archetype":       archetype,
        "emoji":           arch_info["emoji"],
        "description":     arch_info["description"],
        "color":           arch_info["color"],
        "winPlacePerc":    win_pct,
        "percentile":      percentile,
        "rank":            rank_label,
        "improvement_tips": tips,
        "summary":         summary,
        "raw_stats":       stats,
    }
