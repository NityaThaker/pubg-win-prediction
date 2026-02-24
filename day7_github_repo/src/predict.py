"""
predict.py
==========
Prediction pipeline for the PUBG Win Placement Prediction model.

Usage
-----
from src.predict import load_model, predict_single, predict_batch

model, feature_cols = load_model("models/pubg_model_v5.pkl",
                                  "models/day5_feature_cols.pkl")

# Single player dict
result = predict_single(stats_dict, model, feature_cols)
print(result)  # {"winPlacePerc": 0.72, "percentile": 72}

# Batch DataFrame
predictions = predict_batch(df, model, feature_cols)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.features import build_features, reduce_mem_usage, EXCLUDE_COLS


# ─────────────────────────────────────────────────────────────────────────────
def load_model(
    model_path: str | Path,
    feature_cols_path: str | Path | None = None,
) -> Tuple[Any, List[str] | None]:
    """Load a pickled LightGBM model and optional feature-column list.

    Parameters
    ----------
    model_path : str or Path
        Path to the ``.pkl`` model file.
    feature_cols_path : str or Path or None
        Path to the pickled feature-column list produced on Day 5.
        If *None*, ``feature_cols`` will be ``None`` — you must
        supply the column list yourself when calling predict functions.

    Returns
    -------
    model : LightGBM Booster
        Loaded model object.
    feature_cols : list of str or None
        Feature column names, or None if no path was supplied.
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    feature_cols: List[str] | None = None
    if feature_cols_path is not None:
        with open(feature_cols_path, "rb") as f:
            feature_cols = pickle.load(f)

    print(f"✅ Model loaded from {model_path}")
    if feature_cols is not None:
        print(f"   Feature columns: {len(feature_cols)}")
    return model, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
def predict_single(
    stats_dict: Dict[str, float | int | str],
    model: Any,
    feature_cols: List[str],
) -> Dict[str, float]:
    """Predict win-placement percentile for a single player.

    Parameters
    ----------
    stats_dict : dict
        Raw PUBG stats for one player row.  Must include at minimum
        the 29 original dataset columns (missing values default to 0).
    model : LightGBM Booster
        Loaded model from ``load_model``.
    feature_cols : list of str
        Ordered feature column list from ``load_model`` or
        ``get_feature_cols``.

    Returns
    -------
    dict with keys:
        - ``winPlacePerc`` (float 0–1) — predicted placement percentile
        - ``percentile``   (int 0–100) — same value as a percentage
        - ``rank``         (str)        — human-readable rank label
    """
    row = pd.DataFrame([stats_dict])

    # Fill missing columns with 0
    for col in feature_cols:
        if col not in row.columns:
            row[col] = 0.0

    row = reduce_mem_usage(row, verbose=False)
    row = build_features(row)

    # Align columns
    X = row.reindex(columns=feature_cols, fill_value=0).astype(np.float32)
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

    pred = float(model.predict(X)[0])
    pred = float(np.clip(pred, 0.0, 1.0))

    return {
        "winPlacePerc": round(pred, 4),
        "percentile":   int(pred * 100),
        "rank":         _rank_label(pred),
    }


# ─────────────────────────────────────────────────────────────────────────────
def predict_batch(
    df: pd.DataFrame,
    model: Any,
    feature_cols: List[str],
    already_engineered: bool = False,
) -> np.ndarray:
    """Predict win-placement percentile for a batch DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with raw PUBG stats (or already-engineered features
        if ``already_engineered=True``).
    model : LightGBM Booster
        Loaded model.
    feature_cols : list of str
        Feature column list.
    already_engineered : bool
        If True, skip ``build_features`` (e.g. during training).

    Returns
    -------
    np.ndarray of shape (n,)
        Predicted winPlacePerc values clipped to [0, 1].
    """
    if not already_engineered:
        df = reduce_mem_usage(df.copy(), verbose=False)
        df = build_features(df)

    X = df.reindex(columns=feature_cols, fill_value=0).astype(np.float32)
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

    preds = model.predict(X)
    return np.clip(preds, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
def _rank_label(pred: float) -> str:
    """Convert a winPlacePerc prediction to a human-readable rank label."""
    if pred >= 0.90:
        return "🏆 Top 10% — Chicken Dinner Contender"
    elif pred >= 0.75:
        return "🥇 Top 25% — Strong Finisher"
    elif pred >= 0.50:
        return "🥈 Top 50% — Above Average"
    elif pred >= 0.25:
        return "🥉 Bottom 50% — Needs Improvement"
    else:
        return "💀 Bottom 25% — Early Elimination Risk"
