import os
import pickle
import numpy as np
import pandas as pd

MODEL_PATH    = os.getenv("MODEL_PATH",    "/content/drive/MyDrive/PUBG_Project/pubg_model_v5.pkl")
FEATURES_PATH = os.getenv("FEATURES_PATH", "/content/drive/MyDrive/PUBG_Project/day5_feature_cols.pkl")

model        = None
feature_cols = None

def load_model():
    global model, feature_cols
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f:
        feature_cols = pickle.load(f)
    print(f"✅ Model loaded from {MODEL_PATH}")
    print(f"✅ Features loaded: {len(feature_cols)} columns")

def get_model():
    return model

def get_feature_cols():
    return feature_cols

def predict_winplace(raw: dict) -> float:
    """
    Takes raw player stats dict from the API request,
    builds a feature row matching the 105 training features,
    and returns a winPlacePerc prediction between 0 and 1.
    """
    # ── derived features (same logic as Day 2 feature engineering) ──
    kills          = raw.get("kills", 0)
    damageDealt    = raw.get("damageDealt", 0)
    walkDistance   = raw.get("walkDistance", 0)
    rideDistance   = raw.get("rideDistance", 0)
    swimDistance   = raw.get("swimDistance", 0)
    heals          = raw.get("heals", 0)
    boosts         = raw.get("boosts", 0)
    headshotKills  = raw.get("headshotKills", 0)
    DBNOs          = raw.get("DBNOs", 0)
    assists        = raw.get("assists", 0)
    killPlace      = raw.get("killPlace", 50)
    maxPlace       = raw.get("maxPlace", 100)
    numGroups      = raw.get("numGroups", 50)
    matchDuration  = raw.get("matchDuration", 1800)
    weaponsAcquired= raw.get("weaponsAcquired", 0)
    killStreaks     = raw.get("killStreaks", 0)
    longestKill    = raw.get("longestKill", 0)
    roadKills      = raw.get("roadKills", 0)
    vehicleDestroys= raw.get("vehicleDestroys", 0)
    revives        = raw.get("revives", 0)
    teamKills      = raw.get("teamKills", 0)
    rankPoints     = raw.get("rankPoints", 0)
    winPoints      = raw.get("winPoints", 0)
    killPoints     = raw.get("killPoints", 0)

    totalDistance      = walkDistance + rideDistance + swimDistance
    healsBoosts        = heals + boosts
    killsAssists       = kills + assists
    headshotRate       = headshotKills / max(kills, 1)
    damagePerKill      = damageDealt / max(kills, 1)
    killPlaceRatio     = killPlace / max(maxPlace, 1)
    numGroupsRatio     = numGroups / max(maxPlace, 1)
    walkDistancePerSec = walkDistance / max(matchDuration, 1)
    rideDistancePerSec = rideDistance / max(matchDuration, 1)
    damagePerSec       = damageDealt / max(matchDuration, 1)
    killsPerSec        = kills / max(matchDuration, 1)
    boostPerWalk       = boosts / max(walkDistance / 1000, 0.001)

    # ── build feature row with all 105 features ──────────────────────
    # Group/match aggregate features are 0 (single player input)
    feature_row = {col: 0.0 for col in feature_cols}

    # Fill in what we know from raw input
    mapping = {
        "kills":               kills,
        "damageDealt":         damageDealt,
        "walkDistance":        walkDistance,
        "rideDistance":        rideDistance,
        "swimDistance":        swimDistance,
        "heals":               heals,
        "boosts":              boosts,
        "headshotKills":       headshotKills,
        "DBNOs":               DBNOs,
        "assists":             assists,
        "killPlace":           killPlace,
        "maxPlace":            maxPlace,
        "numGroups":           numGroups,
        "matchDuration":       matchDuration,
        "weaponsAcquired":     weaponsAcquired,
        "killStreaks":         killStreaks,
        "longestKill":         longestKill,
        "roadKills":           roadKills,
        "vehicleDestroys":     vehicleDestroys,
        "revives":             revives,
        "teamKills":           teamKills,
        "rankPoints":          rankPoints,
        "winPoints":           winPoints,
        "killPoints":          killPoints,
        "totalDistance":       totalDistance,
        "healsBoosts":         healsBoosts,
        "killsAssists":        killsAssists,
        "headshotRate":        headshotRate,
        "damagePerKill":       damagePerKill,
        "killPlaceRatio":      killPlaceRatio,
        "numGroupsRatio":      numGroupsRatio,
        "walkDistancePerSec":  walkDistancePerSec,
        "rideDistancePerSec":  rideDistancePerSec,
        "damagePerSec":        damagePerSec,
        "killsPerSec":         killsPerSec,
        "boostPerWalk":        boostPerWalk,
    }

    for col, val in mapping.items():
        if col in feature_row:
            feature_row[col] = val

    # ── predict ───────────────────────────────────────────────────────
    df   = pd.DataFrame([feature_row])[feature_cols]
    pred = float(model.predict(df)[0])

    # Clamp to valid range
    pred = max(0.0, min(1.0, pred))
    return round(pred, 4)
