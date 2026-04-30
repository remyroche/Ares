import logging
from typing import Callable, List, Literal, Optional

import numpy as np
import pandas as pd

from extreme_price_movements.feature_select.scoring import UtilityConfig

logger = logging.getLogger(__name__)
_SHAP_FALLBACK_LOGGED = False


def _fallback_model_importance(model: object, feature_names: List[str]) -> pd.DataFrame:
    """Return tree-model importances when SHAP is unavailable."""
    raw = getattr(model, "feature_importances_", None)
    if raw is None and hasattr(model, "booster_"):
        try:
            raw = model.booster_.feature_importance(importance_type="gain")
        except Exception:
            raw = None
    values = np.asarray(raw if raw is not None else [], dtype=np.float32)
    if values.shape[0] != len(feature_names):
        values = np.zeros(len(feature_names), dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.DataFrame({"feature": feature_names, "shap_mean_abs": values})


def block_permutation_importance(
    model,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    base_pred: np.ndarray,
    base_metric: float,
    base_utility: float,
    feature_names: List[str],
    metric_fn: Callable,
    utility_fn: Callable,
    block_ids: Optional[np.ndarray],
    n_repeats: int,
    seed: int,
    max_samples: int = 5000,
) -> pd.DataFrame:
    """Computes permutation importance per feature within optional blocks."""
    rng = np.random.RandomState(seed)

    n_samples = len(X_val)
    if n_samples > max_samples:
        indices = np.linspace(0, n_samples - 1, max_samples, dtype=np.int32)
        X_val = X_val.iloc[indices].copy()
        if hasattr(y_val, "iloc"):
            y_val = y_val.iloc[indices].values
        else:
            y_val = np.asarray(y_val)[indices]
        if hasattr(base_pred, "iloc"):
            base_pred = base_pred.iloc[indices].values
        else:
            base_pred = np.asarray(base_pred)[indices]
        if block_ids is not None:
            block_ids = np.asarray(block_ids)[indices]
        n_samples = len(X_val)
    else:
        y_val = np.asarray(y_val)
        base_pred = np.asarray(base_pred)
        if block_ids is not None:
             block_ids = np.asarray(block_ids)

    n_features = len(feature_names)
    delta_metrics = np.zeros((n_features, n_repeats))
    delta_utilities = np.zeros((n_features, n_repeats))

    blocks = []
    if block_ids is not None:
        unique_blocks = np.unique(block_ids)
        for b in unique_blocks:
            blocks.append(np.where(block_ids == b)[0])
    else:
        blocks.append(np.arange(n_samples))

    # Downcast arrays internally for block permutation
    for i, feature in enumerate(feature_names):
        for r in range(n_repeats):
            X_perm = X_val.copy()
            col_vals = X_perm[feature].values.copy()

            for b_idx in blocks:
                b_col = col_vals[b_idx]
                rng.shuffle(b_col)
                col_vals[b_idx] = b_col

            X_perm[feature] = col_vals

            pred_perm = model.predict(X_perm[feature_names])

            metric_perm = metric_fn(y_val, pred_perm, X_perm)
            utility_perm = utility_fn(y_val, pred_perm, X_perm)

            delta_metrics[i, r] = metric_perm - base_metric
            delta_utilities[i, r] = base_utility - utility_perm

    return pd.DataFrame({
        "feature": feature_names,
        "perm_metric_mean": np.mean(delta_metrics, axis=1),
        "perm_metric_std": np.std(delta_metrics, axis=1),
        "perm_importance_mean": np.mean(delta_utilities, axis=1),
        "perm_importance_std": np.std(delta_utilities, axis=1),
    })

def compute_shap_importance(
    model,
    X_val: pd.DataFrame,
    feature_names: List[str],
    model_kind: Literal["binary", "regression", "quantile"],
    sample_size: int = 5000,
    seed: int = 42
) -> pd.DataFrame:
    """Computes SHAP feature importances."""
    if len(X_val) > sample_size:
        X_sample = X_val.sample(n=sample_size, random_state=seed)
    else:
        X_sample = X_val.copy()

    X_sample = X_sample[feature_names]

    global _SHAP_FALLBACK_LOGGED
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    except Exception as exc:
        if not _SHAP_FALLBACK_LOGGED:
            logger.warning(
                "SHAP importance unavailable; using model feature importances "
                "as fallback: %s",
                exc,
            )
            _SHAP_FALLBACK_LOGGED = True
        return _fallback_model_importance(model, feature_names)

    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]

    shap_arr = np.asarray(shap_values)
    if shap_arr.ndim == 3 and shap_arr.shape[-1] == 2:
        shap_arr = shap_arr[:, :, 1]
    if shap_arr.ndim != 2 or shap_arr.shape[1] != len(feature_names):
        return _fallback_model_importance(model, feature_names)

    shap_mean_abs = np.mean(np.abs(shap_arr), axis=0)

    return pd.DataFrame({
        "feature": feature_names,
        "shap_mean_abs": shap_mean_abs
    })
