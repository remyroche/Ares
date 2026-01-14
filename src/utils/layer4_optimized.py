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

@njit(cache=True)
def rolling_sadf_score_numba(returns, window):
    """
    Calculate rolling SADF-like score (t-stat of AR(1) coefficient).
    O(N * window) optimized with Numba.
    """
    n = len(returns)
    scores = np.zeros(n, dtype=np.float64)

    # Needs at least window + 2 points
    for i in range(window, n):
        # Extract window
        # We need returns[i-window : i]
        # x = returns[t-1], y = returns[t]
        # slice: chunk = returns[i-window : i]
        # x = chunk[:-1], y = chunk[1:]

        start_idx = i - window
        end_idx = i

        # Manually compute covariance and variance for regression
        # y = alpha + beta * x

        n_points = window - 1
        if n_points < 2:
            continue

        mean_x = 0.0
        mean_y = 0.0

        # First pass: means
        for j in range(n_points):
            val_x = returns[start_idx + j]
            val_y = returns[start_idx + j + 1]
            mean_x += val_x
            mean_y += val_y

        mean_x /= n_points
        mean_y /= n_points

        # Second pass: cov and var
        cov_xy = 0.0
        var_x = 0.0

        for j in range(n_points):
            val_x = returns[start_idx + j]
            val_y = returns[start_idx + j + 1]

            diff_x = val_x - mean_x
            diff_y = val_y - mean_y

            cov_xy += diff_x * diff_y
            var_x += diff_x * diff_x

        var_x /= n_points # Population var or sample? numpy var is usually population by default?
                          # Code used np.var(x), which is population var (div n).
        cov_xy /= n_points # np.cov is sample (div n-1).

        # Wait, the original code used:
        # beta = np.cov(x, y)[0, 1] / np.var(x)
        # np.cov divides by N-1. np.var divides by N.
        # This is mixed, but let's stick to standard OLS slope formula:
        # beta = sum((x-mx)(y-my)) / sum((x-mx)^2)
        # This cancels out the N or N-1 factor.

        if var_x < 1e-12:
            scores[i] = 0.0
            continue

        beta = cov_xy / var_x # wait, if I use the sums directly: sum_xy_diff / sum_xx_diff
        # But cov_xy above is sum / N. var_x is sum / N. So ratio is correct.
        # Wait, if np.cov uses N-1 and np.var uses N, then factor is (N)/(N-1).
        # Let's use sums to be precise.

        sum_sq_diff_x = var_x * n_points
        sum_prod_diff = cov_xy * n_points

        beta = sum_prod_diff / sum_sq_diff_x
        alpha = mean_y - beta * mean_x

        # Calculate residuals std
        sum_sq_resid = 0.0
        for j in range(n_points):
            val_x = returns[start_idx + j]
            val_y = returns[start_idx + j + 1]
            y_pred = alpha + beta * val_x
            resid = val_y - y_pred
            sum_sq_resid += resid * resid

        std_resid = np.sqrt(sum_sq_resid / n_points)

        if std_resid < 1e-12:
            scores[i] = 0.0
        else:
            # t_stat = beta / (std_resid / sqrt(n * var_x))
            # The original code: np.std(residuals) / np.sqrt(len(x) * np.var(x))
            # This is standard error of beta approximation.
            denom = std_resid / np.sqrt(n_points * var_x)
            if denom < 1e-12:
                 scores[i] = 0.0
            else:
                 scores[i] = np.abs(beta / denom)

    return scores

@njit(cache=True)
def rolling_cusum_scores_numba(returns, mean_ret):
    """
    Calculate CUSUM scores.
    """
    n = len(returns)
    scores = np.zeros(n, dtype=np.float64)

    cusum_pos = 0.0
    cusum_neg = 0.0

    for i in range(n):
        diff = returns[i] - mean_ret

        cusum_pos = cusum_pos + diff
        if cusum_pos < 0:
            cusum_pos = 0

        cusum_neg = cusum_neg + diff
        if cusum_neg > 0:
            cusum_neg = 0

        scores[i] = np.abs(cusum_pos) + np.abs(cusum_neg)

    return scores

@njit(cache=True)
def compute_proxy_entropy_numba(returns, window=100, n_bins=10):
    """
    Compute proxy entropy using a rolling histogram of returns.
    This serves as a fast approximation when granular data is unavailable.

    Args:
        returns: 1D numpy array of returns
        window: Rolling window size
        n_bins: Number of histogram bins

    Returns:
        1D numpy array of entropy scores
    """
    n = len(returns)
    entropy = np.zeros(n, dtype=np.float32)

    # Pre-allocate histogram array
    hist = np.zeros(n_bins, dtype=np.int32)

    # Need at least window points
    for i in range(window, n):
        # Extract window
        start_idx = i - window
        end_idx = i

        # Get min/max for binning
        min_val = 1000.0
        max_val = -1000.0

        # Single pass to find range
        for j in range(start_idx, end_idx):
            val = returns[j]
            if val < min_val:
                min_val = val
            if val > max_val:
                max_val = val

        # Handle zero variance
        if max_val <= min_val:
            entropy[i] = 0.0
            continue

        # Reset histogram
        for b in range(n_bins):
            hist[b] = 0

        bin_width = (max_val - min_val) / n_bins

        # Fill histogram
        for j in range(start_idx, end_idx):
            val = returns[j]
            bin_idx = int((val - min_val) / bin_width)
            if bin_idx >= n_bins:
                bin_idx = n_bins - 1
            hist[bin_idx] += 1

        # Calculate entropy
        # H = -sum(p * log(p))
        ent = 0.0
        for b in range(n_bins):
            count = hist[b]
            if count > 0:
                p = count / window
                ent -= p * np.log(p)

        # Normalize by log(n_bins) to get 0-1 range?
        # Or return raw Shannon entropy (nats).
        # Let's return raw nats as it's standard.
        entropy[i] = ent

    return entropy
