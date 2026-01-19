import numpy as np
import pandas as pd
from typing import List, Optional

def average_layer25_predictions(df: pd.DataFrame, prefix: str = "meta_chaser_") -> pd.DataFrame:
    """
    Extracts and averages Layer 2.5 Chaser predictions from the input DataFrame.

    Layer 2.5 Integration produces features like:
    - meta_chaser_prediction (Ensemble)
    - meta_chaser_ind_xgb (Individual XGB)
    - meta_chaser_ind_cat (Individual CatBoost)
    - meta_chaser_confidence

    This function aggregates them into a robust signal for Layer 4.

    Args:
        df: Input DataFrame containing Layer 2.5 outputs.
        prefix: Prefix used for Layer 2.5 features.

    Returns:
        DataFrame with averaged chaser features.
    """
    out = pd.DataFrame(index=df.index)

    # 1. Identify Prediction Columns
    # Look for ensemble prediction and individual model predictions
    # Standard names from Layer25Integration.get_meta_learner_features

    # Check for direct columns first (backward compatibility if prefix differs)
    candidates = [
        "meta_chaser_prediction",
        "meta_chaser_ind_xgb",
        "meta_chaser_ind_cat",
        "meta_chaser_ind_lgbm"
    ]

    pred_cols = [c for c in candidates if c in df.columns]

    # Also search by prefix if generic
    if not pred_cols:
        pred_cols = [c for c in df.columns if prefix in c and "prediction" in c and "confidence" not in c]

    if not pred_cols:
        # Fallback: Return zeros if no Chaser features found
        out["chaser_prob_avg"] = 0.0
        out["chaser_prob_std"] = 0.0
        out["chaser_high_conf_prob"] = 0.0
        return out

    # 2. Compute Aggregates
    preds = df[pred_cols].values

    # Mean Prediction
    out["chaser_prob_avg"] = np.mean(preds, axis=1)

    # Prediction Stability/Disagreement (Std)
    if len(pred_cols) > 1:
        out["chaser_prob_std"] = np.std(preds, axis=1)
    else:
        out["chaser_prob_std"] = 0.0

    # 3. Confidence-Weighted Prediction (if confidence available)
    conf_col = "meta_chaser_confidence"
    if conf_col in df.columns:
        conf = df[conf_col].values
        # High confidence gate: only pass prediction if conf > 0.5 (example)
        # Or simple product
        out["chaser_high_conf_prob"] = out["chaser_prob_avg"] * conf
    else:
        out["chaser_high_conf_prob"] = out["chaser_prob_avg"] # Fallback

    return out
