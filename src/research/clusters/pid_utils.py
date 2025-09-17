"""
Partial Information Decomposition (PID) utilities (simplified).

Goal: provide pragmatic estimates of unique, redundant, and synergistic
information of small feature groups (pairs/triads) relative to a target.

This is a simplified approach intended for research discovery:
  - Redundancy ~ min(MI(x_i; y)) across features in set
  - Unique(x_i) ~ MI(x_i; y) - Redundancy (clipped at 0)
  - Synergy ~ MI([x1,x2]; y) - max(MI(x1; y), MI(x2; y))

For triads, we approximate synergy by comparing joint MI to best pair MI.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import system_logger

# Try to reuse enhanced PID from training utils if present
try:
    from src.training.utils.feature_selection.enhanced_partial_information_decomposition import (
        PIDCalculator, PIDConfig, PIDMeasure  # type: ignore
    )
    _HAS_ENHANCED_PID = True
except Exception:
    PIDCalculator = None  # type: ignore
    PIDConfig = None  # type: ignore
    PIDMeasure = None  # type: ignore
    _HAS_ENHANCED_PID = False


def _is_classification_target(y: np.ndarray) -> bool:
    unique = np.unique(y[~np.isnan(y)])
    return len(unique) < min(len(y) * 0.1, 50)


def _estimate_mi(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> float:
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    if _is_classification_target(y):
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        mi = mutual_info_classif(Xs, y_enc, random_state=random_state)
        return float(np.mean(mi))
    else:
        mi = mutual_info_regression(Xs, y, random_state=random_state)
        return float(np.mean(mi))


def pid_pair(X: pd.DataFrame, y: np.ndarray, f1: str, f2: str, random_state: int = 42) -> Dict[str, float]:
    """PID for a pair {f1, f2} relative to y (simplified).

    Returns dict with keys: 'mi_f1', 'mi_f2', 'mi_joint', 'redundancy', 'unique_f1',
    'unique_f2', 'synergy'.
    """
    logger = system_logger.getChild('PID')

    # Prefer enhanced PID if available
    if _HAS_ENHANCED_PID and PIDCalculator is not None and PIDConfig is not None and PIDMeasure is not None:
        try:
            calc = PIDCalculator(PIDConfig())
            x1 = X[[f1]].fillna(0).values.squeeze()
            x2 = X[[f2]].fillna(0).values.squeeze()
            yy = pd.Series(y).fillna(0).values.squeeze()
            res = calc.compute_pid(x1, x2, yy)
            # Take I_MIN result as baseline
            pid_res = res.get(PIDMeasure.I_MIN)
            if pid_res is not None:
                return {
                    'mi_f1': np.nan,  # not directly returned; keep simplified fields absent
                    'mi_f2': np.nan,
                    'mi_joint': pid_res.total_mi,
                    'redundancy': pid_res.redundant,
                    'unique_f1': pid_res.unique_x1,
                    'unique_f2': pid_res.unique_x2,
                    'synergy': pid_res.synergistic,
                }
        except Exception as e:
            logger.warning(f"Enhanced PID failed ({e}); falling back to simplified PID")
    x1 = X[[f1]].fillna(0).values
    x2 = X[[f2]].fillna(0).values
    mi_f1 = _estimate_mi(x1, y, random_state)
    mi_f2 = _estimate_mi(x2, y, random_state)
    mi_joint = _estimate_mi(np.concatenate([x1, x2], axis=1), y, random_state)

    redundancy = min(mi_f1, mi_f2)
    unique_f1 = max(0.0, mi_f1 - redundancy)
    unique_f2 = max(0.0, mi_f2 - redundancy)
    synergy = max(0.0, mi_joint - max(mi_f1, mi_f2))

    return {
        'mi_f1': mi_f1,
        'mi_f2': mi_f2,
        'mi_joint': mi_joint,
        'redundancy': redundancy,
        'unique_f1': unique_f1,
        'unique_f2': unique_f2,
        'synergy': synergy,
    }


def pid_triad(X: pd.DataFrame, y: np.ndarray, features: List[str], random_state: int = 42) -> Dict[str, float]:
    """Approximate PID for triad of features relative to y.

    Compare joint MI vs best pair MI to approximate higher-order synergy.
    """
    # Use enhanced PID in pairwise passes if available; triad support could be added similarly
    assert len(features) == 3, "pid_triad expects exactly 3 features"
    f1, f2, f3 = features
    x1 = X[[f1]].fillna(0).values
    x2 = X[[f2]].fillna(0).values
    x3 = X[[f3]].fillna(0).values

    mi_f = [_estimate_mi(x1, y, random_state), _estimate_mi(x2, y, random_state), _estimate_mi(x3, y, random_state)]
    mi_pairs = [
        _estimate_mi(np.concatenate([x1, x2], axis=1), y, random_state),
        _estimate_mi(np.concatenate([x1, x3], axis=1), y, random_state),
        _estimate_mi(np.concatenate([x2, x3], axis=1), y, random_state),
    ]
    mi_joint = _estimate_mi(np.concatenate([x1, x2, x3], axis=1), y, random_state)

    redundancy = min(mi_f)
    unique_sum = sum(max(0.0, m - redundancy) for m in mi_f)
    pair_best = max(mi_pairs)
    synergy = max(0.0, mi_joint - pair_best)

    return {
        'mi_f1': mi_f[0],
        'mi_f2': mi_f[1],
        'mi_f3': mi_f[2],
        'mi_best_pair': pair_best,
        'mi_joint': mi_joint,
        'redundancy': redundancy,
        'unique_sum': unique_sum,
        'synergy': synergy,
    }

