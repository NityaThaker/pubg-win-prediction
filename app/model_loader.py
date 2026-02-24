import pickle
import threading
import numpy as np
import pandas as pd
from pathlib import Path

DRIVE_DIR    = Path("/content/drive/MyDrive/PUBG_Project")
MODEL_PATH   = DRIVE_DIR / "pubg_model_v5.pkl"
FEATS_PATH   = DRIVE_DIR / "day5_feature_cols.pkl"

model        = None
feature_cols = None
_model_lock  = threading.Lock()

def load_model():
    global model, feature_cols
    with _model_lock:
        if model is not None:
            return
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        if not FEATS_PATH.exists():
            raise FileNotFoundError(f"Feature list not found at {FEATS_PATH}")
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(FEATS_PATH, "rb") as f:
            feature_cols = pickle.load(f)
        print(f"Model loaded — {len(feature_cols)} features expected")

def get_model():        return model
def get_feature_cols(): return feature_cols

def build_features(raw):
    d = dict(raw)

    # ── kill features ──────────────────────────────────────────────────────
    d["killsNorm"]              = d["kills"] * (100 / max(d["maxPlace"], 1))
    d["killsPerWalkDist"]       = d["kills"] / max(d["walkDistance"], 1)
    d["killsPerRideDist"]       = d["kills"] / max(d["rideDistance"] + 1, 1)
    d["headshotRate"]           = d["headshotKills"] / max(d["kills"], 1)
    d["killStreakRate"]          = d["killStreaks"] / max(d["kills"], 1)
    d["killPlaceNorm"]          = d["killPlace"] / max(d["maxPlace"], 1)
    d["killPlaceOverMaxPlace"]  = d["killPlace"] / max(d["maxPlace"], 1)

    # ── distance features ──────────────────────────────────────────────────
    d["totalDistance"]          = d["walkDistance"] + d["rideDistance"] + d["swimDistance"]
    d["walkDistanceNorm"]       = d["walkDistance"] / max(d["totalDistance"], 1)
    d["rideDistanceNorm"]       = d["rideDistance"] / max(d["totalDistance"], 1)
    d["walkDistancePerSec"]     = d["walkDistance"] / max(d["matchDuration"], 1)
    d["rideDistancePerSec"]     = d["rideDistance"] / max(d["matchDuration"], 1)
    d["distancePerKill"]        = d["totalDistance"] / max(d["kills"], 1)

    # ── healing / survival ─────────────────────────────────────────────────
    d["healsAndBoosts"]         = d["heals"] + d["boosts"]
    d["healsPerWalkDist"]       = d["heals"] / max(d["walkDistance"], 1)
    d["boostsPerWalkDist"]      = d["boosts"] / max(d["walkDistance"], 1)
    d["healsBoostsPerDist"]     = d["healsAndBoosts"] / max(d["totalDistance"], 1)

    # ── damage features ────────────────────────────────────────────────────
    d["damagePerKill"]          = d["damageDealt"] / max(d["kills"], 1)
    d["damagePerWalkDist"]      = d["damageDealt"] / max(d["walkDistance"], 1)
    d["damageNorm"]             = d["damageDealt"] * (100 / max(d["maxPlace"], 1))

    # ── team features ──────────────────────────────────────────────────────
    d["teamKillRate"]           = d["teamKills"] / max(d["kills"] + 1, 1)
    d["reviveRate"]             = d["revives"] / max(d["numGroups"], 1)

    # ── weapon / loot ──────────────────────────────────────────────────────
    d["weaponsPerDist"]         = d["weaponsAcquired"] / max(d["totalDistance"], 1)

    # ── rank / points ──────────────────────────────────────────────────────
    d["rankPointsNorm"]         = d["rankPoints"] / max(d["rankPoints"] + 1, 1)
    d["winPointsNorm"]          = d["winPoints"] / max(d["winPoints"] + 1, 1)
    d["killPointsNorm"]         = d["killPoints"] / max(d["killPoints"] + 1, 1)
    d["totalPoints"]            = d["rankPoints"] + d["winPoints"] + d["killPoints"]

    # ── match-level context ────────────────────────────────────────────────
    d["playersPerGroup"]        = d["maxPlace"] / max(d["numGroups"], 1)
    d["groupsPerMatch"]         = d["numGroups"] / max(d["maxPlace"], 1)

    # ── additional high-importance ratios (improvement 5) ──────────────────
    d["killDistanceOverWalk"]   = d["longestKill"] / max(d["walkDistance"], 1)
    d["dbnoPerKill"]            = d["DBNOs"] / max(d["kills"], 1)
    d["assistPerKill"]          = d["assists"] / max(d["kills"], 1)
    d["roadKillPct"]            = d["roadKills"] / max(d["kills"], 1)

    # ── cheater flags (default 0 for API users) ────────────────────────────
    d["is_cheater"]             = 0
    d["is_cheater_rule"]        = 0
    d["is_cheater_iso"]         = 0

    # ── build dataframe ────────────────────────────────────────────────────
    df = pd.DataFrame([d])
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_cols]
    df.replace([float("inf"), float("-inf")], float("nan"), inplace=True)
    df.fillna(0, inplace=True)
    return df.astype("float32")

def predict_winplace(raw):
    df   = build_features(raw)
    pred = model.predict(df)[0]
    return round(float(max(0.0, min(1.0, pred))), 4)