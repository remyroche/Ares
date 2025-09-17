"""
Partial Information Decomposition (PID) utilities.

This module delegates PID to src/training/utils/feature_selection/enhanced_partial_information_decomposition
for mathematically grounded decomposition. If unavailable, it fails fast.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import system_logger

# Use enhanced PID; do not provide simplified fallbacks
from src.training.utils.feature_selection.enhanced_partial_information_decomposition import (  # type: ignore
    PIDCalculator, PIDConfig, PIDMeasure
)


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
    """PID for a pair {f1, f2} relative to y using enhanced PID.

    Returns dict with keys: 'mi_joint', 'redundancy', 'unique_f1', 'unique_f2', 'synergy'.
    """
    calc = PIDCalculator(PIDConfig())
    x1 = X[[f1]].fillna(0).values.squeeze()
    x2 = X[[f2]].fillna(0).values.squeeze()
    yy = pd.Series(y).fillna(0).values.squeeze()
    res = calc.compute_pid(x1, x2, yy)
    pid_res = res.get(PIDMeasure.I_MIN)
    if pid_res is None:
        raise RuntimeError("Enhanced PID did not return I_MIN result")
    return {
        'mi_joint': pid_res.total_mi,
        'redundancy': pid_res.redundant,
        'unique_f1': pid_res.unique_x1,
        'unique_f2': pid_res.unique_x2,
        'synergy': pid_res.synergistic,
    }
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
    """Triad PID placeholder; raise for now to avoid silent approximations."""
    raise NotImplementedError("Triad PID should use enhanced multivariate PID; not implemented here")

