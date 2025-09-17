"""
Feature selection utilities: mutual information estimation and mRMR screening.

This module provides:
  - Robust MI estimation for continuous features/targets using sklearn's
    mutual_info_classif/regression (with preprocessing) as a practical default.
  - A simple, scalable mRMR (Max-Relevance Min-Redundancy) selector that
    selects features maximizing MI with target while penalizing redundancy
    via pairwise MI between already selected features.

Notes:
  - All computations assume features are lagged appropriately upstream.
  - For continuous targets, regression MI is used; for discrete targets,
    classification MI is used. Targets with few unique values are treated as
    classification to avoid estimator bias.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import system_logger

# Try to reuse the richer selection framework if available
try:
    from src.training.utils.feature_selection.selection_methods import MRMRSelector  # type: ignore
    _HAS_TRAINING_MRMR = True
except Exception:
    MRMRSelector = None  # type: ignore
    _HAS_TRAINING_MRMR = False


@dataclass
class MRMRConfig:
    """Configuration for mRMR selection."""
    max_features: int = 50
    redundancy_penalty: float = 0.5  # lambda in mRMR score: score = relevance - lambda * redundancy
    normalize_scores: bool = True
    random_state: int = 42


def _is_classification_target(y: np.ndarray) -> bool:
    unique = np.unique(y[~np.isnan(y)])
    return len(unique) < min(len(y) * 0.1, 50)


def estimate_mutual_information(X: pd.DataFrame, y: np.ndarray, random_state: int = 42) -> np.ndarray:
    """Estimate MI(feature_i; y) for each feature_i in X.

    Uses sklearn MI estimators with simple handling of classification vs regression.
    """
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    X_mat = X.fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_mat)

    if _is_classification_target(y):
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        mi = mutual_info_classif(X_scaled, y_enc, random_state=random_state)
    else:
        mi = mutual_info_regression(X_scaled, y, random_state=random_state)

    # Normalize to [0,1] to make mRMR lambda interpretable
    if np.max(mi) > 0:
        mi = mi / np.max(mi)
    return mi


def estimate_pairwise_mi(X: pd.DataFrame, random_state: int = 42) -> np.ndarray:
    """Estimate symmetric MI matrix between features (approximate, scalable).

    Implementation detail: For speed, we use mutual_info_regression on each pair
    by treating one feature as target. This is symmetric only approximately; we
    average MI(i->j) and MI(j->i) to symmetrize.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import mutual_info_regression

    features = X.columns.tolist()
    Xv = X.fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(Xv)

    n = X_scaled.shape[1]
    mi_mat = np.zeros((n, n))
    rng = np.random.RandomState(random_state)

    # Compute MI row-wise; for n large this is O(n^2), consider subsampling in production
    for i in range(n):
        xi = X_scaled[:, [i]]
        for j in range(i + 1, n):
            yj = X_scaled[:, j]
            mi_ij = mutual_info_regression(xi, yj, random_state=random_state)[0]
            mi_ji = mutual_info_regression(X_scaled[:, [j]], X_scaled[:, i], random_state=random_state)[0]
            mij = (mi_ij + mi_ji) / 2.0
            mi_mat[i, j] = mi_mat[j, i] = mij

    if np.max(mi_mat) > 0:
        mi_mat = mi_mat / np.max(mi_mat)
    return mi_mat


def mrmr_select(X: pd.DataFrame, y: np.ndarray, config: Optional[MRMRConfig] = None) -> List[str]:
    """Select features using mRMR criterion.

    If the training feature_selection framework is available, delegate to it;
    otherwise use the local implementation.
    """
    cfg = config or MRMRConfig()
    logger = system_logger.getChild('mRMR')

    feature_names = list(X.columns)
    if len(feature_names) == 0:
        return []

    # Prefer robust selector from training utils if present
    if _HAS_TRAINING_MRMR and MRMRSelector is not None:
        try:
            selector = MRMRSelector(config={
                'relevance_method': 'mutual_info',
                'redundancy_method': 'mutual_info',
                'n_neighbors': 3,
            })
            res = selector.select_features(X.values, y, feature_names, n_features=cfg.max_features)
            sel = res.get('selected_features', [])
            if sel:
                return sel
        except Exception as e:
            logger.warning(f"Training MRMRSelector failed ({e}); falling back to local mRMR")

    # Local fallback: MI-based mRMR
    mi_y = estimate_mutual_information(X, y, random_state=cfg.random_state)

    if len(feature_names) <= cfg.max_features:
        logger.info(f"mRMR pass-through: {len(feature_names)} <= max_features")
        return feature_names

    mi_mat = estimate_pairwise_mi(X, random_state=cfg.random_state)

    selected: List[int] = []
    remaining: List[int] = list(range(len(feature_names)))

    first_idx = int(np.argmax(mi_y))
    selected.append(first_idx)
    remaining.remove(first_idx)

    while len(selected) < min(cfg.max_features, len(feature_names)) and remaining:
        best_idx = None
        best_score = -1e9
        for j in remaining:
            relevance = mi_y[j]
            redundancy = np.mean([mi_mat[j, s] for s in selected]) if selected else 0.0
            score = relevance - cfg.redundancy_penalty * redundancy
            if score > best_score:
                best_score = score
                best_idx = j
        selected.append(best_idx)  # type: ignore[arg-type]
        remaining.remove(best_idx)  # type: ignore[arg-type]

    return [feature_names[i] for i in selected]

