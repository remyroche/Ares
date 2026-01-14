import numpy as np
from numba import njit, prange

@njit(cache=True)
def compute_financial_weights_numba(abs_returns):
    """
    Compute financial weights using Numba.

    Args:
        abs_returns: 1D numpy array of absolute returns

    Returns:
        1D numpy array of weights
    """
    n = len(abs_returns)
    total_abs_ret = np.sum(abs_returns) + 1e-9
    weights = abs_returns / total_abs_ret * n

    # Simple quantile clip
    q01 = np.percentile(weights, 1.0)
    q99 = np.percentile(weights, 99.0)

    # Clip in place
    for i in range(n):
        if weights[i] < q01:
            weights[i] = q01
        elif weights[i] > q99:
            weights[i] = q99

    # Re-normalize
    total_weight = np.sum(weights) + 1e-9
    weights = weights / total_weight * n

    return weights

@njit(parallel=True, cache=True)
def extract_prob_features_numba(probs):
    """
    Extract probability features: logit, confidence.
    probs: 2D array (n_samples, n_prob_cols)
    Returns:
        logits: (n_samples, n_prob_cols)
        confidences: (n_samples, n_prob_cols)
    """
    n_samples, n_cols = probs.shape
    logits = np.empty((n_samples, n_cols), dtype=np.float32)
    confidences = np.empty((n_samples, n_cols), dtype=np.float32)

    epsilon = 1e-6

    for i in prange(n_samples):
        for j in range(n_cols):
            p = probs[i, j]
            # Clip
            if p < epsilon:
                p = epsilon
            elif p > 1.0 - epsilon:
                p = 1.0 - epsilon

            logits[i, j] = np.log(p / (1.0 - p))
            confidences[i, j] = np.abs(p - 0.5) * 2.0

    return logits, confidences

@njit(parallel=True, cache=True)
def compute_prob_stats_numba(probs):
    """
    Compute mean, std, min, max, range for probabilities.
    probs: 2D array (n_samples, n_prob_cols)
    Returns:
        means, stds, mins, maxs, ranges (all 1D arrays)
    """
    n_samples, n_cols = probs.shape
    means = np.empty(n_samples, dtype=np.float32)
    stds = np.empty(n_samples, dtype=np.float32)
    mins = np.empty(n_samples, dtype=np.float32)
    maxs = np.empty(n_samples, dtype=np.float32)
    ranges = np.empty(n_samples, dtype=np.float32)

    for i in prange(n_samples):
        sum_val = 0.0
        min_val = 100.0 # Probs are <= 1
        max_val = -1.0

        # Pass 1: Sum, min, max
        for j in range(n_cols):
            val = probs[i, j]
            sum_val += val
            if val < min_val:
                min_val = val
            if val > max_val:
                max_val = val

        mean = sum_val / n_cols
        means[i] = mean
        mins[i] = min_val
        maxs[i] = max_val
        ranges[i] = max_val - min_val

        # Pass 2: Std
        sum_sq_diff = 0.0
        for j in range(n_cols):
            diff = probs[i, j] - mean
            sum_sq_diff += diff * diff

        stds[i] = np.sqrt(sum_sq_diff / (n_cols - 1)) if n_cols > 1 else 0.0

    return means, stds, mins, maxs, ranges
