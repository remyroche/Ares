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
    """Triad PID using enhanced multivariate PID when available, with fallback approximation."""
    if len(features) != 3:
        raise ValueError("Triad PID requires exactly 3 features")
    
    try:
        # Try to use enhanced multivariate PID
        calc = PIDCalculator(PIDConfig())
        x1 = X[[features[0]]].fillna(0).values.squeeze()
        x2 = X[[features[1]]].fillna(0).values.squeeze()
        x3 = X[[features[2]]].fillna(0).values.squeeze()
        yy = pd.Series(y).fillna(0).values.squeeze()
        
        # For triad PID, we need to compute multiple pairwise PIDs and combine them
        # This is a simplified approach - in practice, you'd want true multivariate PID
        
        # Compute pairwise PIDs
        pid_12 = calc.compute_pid(x1, x2, yy)
        pid_13 = calc.compute_pid(x1, x3, yy)
        pid_23 = calc.compute_pid(x2, x3, yy)
        
        # Extract results
        pid_12_res = pid_12.get(PIDMeasure.I_MIN)
        pid_13_res = pid_13.get(PIDMeasure.I_MIN)
        pid_23_res = pid_23.get(PIDMeasure.I_MIN)
        
        if pid_12_res is None or pid_13_res is None or pid_23_res is None:
            raise RuntimeError("Enhanced PID did not return I_MIN results")
        
        # Compute joint mutual information
        mi_joint = _estimate_mi(np.column_stack([x1, x2, x3]), y, random_state)
        
        # Approximate triad components
        # This is a simplified approximation - true multivariate PID is more complex
        redundancy = min(pid_12_res.redundant, pid_13_res.redundant, pid_23_res.redundant)
        unique_1 = max(0.0, pid_12_res.unique_x1 + pid_13_res.unique_x1 - redundancy)
        unique_2 = max(0.0, pid_12_res.unique_x2 + pid_23_res.unique_x1 - redundancy)
        unique_3 = max(0.0, pid_13_res.unique_x2 + pid_23_res.unique_x2 - redundancy)
        
        # Synergy is what's left after accounting for individual and pairwise contributions
        individual_contributions = unique_1 + unique_2 + unique_3 + redundancy
        synergy = max(0.0, mi_joint - individual_contributions)
        
        return {
            'mi_joint': mi_joint,
            'redundancy': redundancy,
            'unique_f1': unique_1,
            'unique_f2': unique_2,
            'unique_f3': unique_3,
            'synergy': synergy,
            'pairwise_12': pid_12_res.total_mi,
            'pairwise_13': pid_13_res.total_mi,
            'pairwise_23': pid_23_res.total_mi
        }
        
    except Exception as e:
        # Fallback to simplified approximation
        print(f"Enhanced triad PID failed: {e}, using simplified approximation")
        
        # Compute individual mutual informations
        mi_1 = _estimate_mi(X[[features[0]]].fillna(0).values, y, random_state)
        mi_2 = _estimate_mi(X[[features[1]]].fillna(0).values, y, random_state)
        mi_3 = _estimate_mi(X[[features[2]]].fillna(0).values, y, random_state)
        
        # Compute pairwise mutual informations
        mi_12 = _estimate_mi(X[[features[0], features[1]]].fillna(0).values, y, random_state)
        mi_13 = _estimate_mi(X[[features[0], features[2]]].fillna(0).values, y, random_state)
        mi_23 = _estimate_mi(X[[features[1], features[2]]].fillna(0).values, y, random_state)
        
        # Compute joint mutual information
        mi_joint = _estimate_mi(X[features].fillna(0).values, y, random_state)
        
        # Simplified triad approximation
        redundancy = min(mi_1, mi_2, mi_3)
        unique_1 = max(0.0, mi_1 - redundancy)
        unique_2 = max(0.0, mi_2 - redundancy)
        unique_3 = max(0.0, mi_3 - redundancy)
        
        # Synergy approximation
        synergy = max(0.0, mi_joint - (mi_1 + mi_2 + mi_3 - redundancy))
        
        return {
            'mi_joint': mi_joint,
            'redundancy': redundancy,
            'unique_f1': unique_1,
            'unique_f2': unique_2,
            'unique_f3': unique_3,
            'synergy': synergy,
            'pairwise_12': mi_12,
            'pairwise_13': mi_13,
            'pairwise_23': mi_23,
            'approximation': True
        }

