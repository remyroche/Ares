import pandas as pd
import numpy as np
import shap
from typing import Callable, List, Optional, Tuple, Literal
from extreme_price_movements.feature_select.scoring import UtilityConfig

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
) -> pd.DataFrame:
    """Computes permutation importance per feature within optional blocks."""
    rng = np.random.RandomState(seed)

    n_samples = len(X_val)
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

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]

    shap_mean_abs = np.mean(np.abs(shap_values), axis=0)

    return pd.DataFrame({
        "feature": feature_names,
        "shap_mean_abs": shap_mean_abs
    })
