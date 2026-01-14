"""
Fast Information Theory Functions - Numba Optimized
===================================================

This module provides high-performance, JIT-compiled implementations of
Information Theory metrics (Entropy, Mutual Information) and utility functions
(Discretization, Histogram) to replace slower Scikit-Learn/Scipy versions.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
import warnings

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Dummy decorators
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def prange(x):
        return range(x)

@njit(parallel=True)
def discretize_features_numba(X: np.ndarray, bins: int = 10) -> np.ndarray:
    """
    Fast discretization of features into equal-width bins.

    Args:
        X: Feature matrix (n_samples, n_features)
        bins: Number of bins

    Returns:
        Discretized matrix (n_samples, n_features)
    """
    n_samples, n_features = X.shape
    X_disc = np.zeros((n_samples, n_features), dtype=np.int32)

    for j in prange(n_features):
        col = X[:, j]
        # Handle NaNs and Infs
        valid_mask = np.isfinite(col)
        if not np.any(valid_mask):
            continue

        col_valid = col[valid_mask]
        min_val = np.min(col_valid)
        max_val = np.max(col_valid)

        if max_val == min_val:
            continue

        # Binning
        # bin = floor((val - min) / (max - min) * bins)
        # clip to [0, bins-1]
        scale = bins / (max_val - min_val + 1e-9)

        for i in range(n_samples):
            val = col[i]
            if np.isfinite(val):
                bin_idx = int((val - min_val) * scale)
                if bin_idx >= bins:
                    bin_idx = bins - 1
                elif bin_idx < 0:
                    bin_idx = 0
                X_disc[i, j] = bin_idx
            else:
                X_disc[i, j] = -1 # Sentinel for missing

    return X_disc

@njit
def numba_histogram_2d(x: np.ndarray, y: np.ndarray, bins: int) -> np.ndarray:
    """
    Compute 2D histogram (contingency table) for two integer arrays.
    Assumes arrays are already discretized to [0, bins-1].

    Args:
        x: Discretized array 1
        y: Discretized array 2
        bins: Number of bins

    Returns:
        2D histogram matrix (bins, bins)
    """
    hist = np.zeros((bins, bins), dtype=np.float64)
    n = len(x)

    for i in range(n):
        idx_x = x[i]
        idx_y = y[i]

        if idx_x >= 0 and idx_y >= 0:
            hist[idx_x, idx_y] += 1.0

    return hist

@njit
def numba_mutual_info(hist_2d: np.ndarray) -> float:
    """
    Calculate Mutual Information from a 2D histogram (contingency table).
    MI(X, Y) = sum(p_xy * log(p_xy / (p_x * p_y)))

    Args:
        hist_2d: Contingency table

    Returns:
        Mutual Information (nats)
    """
    n_samples = np.sum(hist_2d)
    if n_samples == 0:
        return 0.0

    p_xy = hist_2d / n_samples
    p_x = np.sum(p_xy, axis=1)
    p_y = np.sum(p_xy, axis=0)

    mi = 0.0
    bins = hist_2d.shape[0]

    for i in range(bins):
        for j in range(bins):
            if p_xy[i, j] > 0:
                mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j] + 1e-12))

    return max(0.0, mi)

@njit(parallel=True)
def vectorized_pairwise_mi(X_disc: np.ndarray, bins: int) -> np.ndarray:
    """
    Compute pairwise Mutual Information matrix for discretized features.

    Args:
        X_disc: Discretized feature matrix (n_samples, n_features)
        bins: Number of bins used

    Returns:
        Symmetric MI matrix (n_features, n_features)
    """
    n_features = X_disc.shape[1]
    mi_matrix = np.zeros((n_features, n_features), dtype=np.float64)

    # Iterate over unique pairs
    for i in prange(n_features):
        for j in range(i, n_features):
            if i == j:
                # MI(X, X) is Entropy H(X)
                # We can compute it via same histogram logic
                hist = numba_histogram_2d(X_disc[:, i], X_disc[:, i], bins)
                score = numba_mutual_info(hist)
            else:
                hist = numba_histogram_2d(X_disc[:, i], X_disc[:, j], bins)
                score = numba_mutual_info(hist)

            mi_matrix[i, j] = score
            mi_matrix[j, i] = score

    return mi_matrix
