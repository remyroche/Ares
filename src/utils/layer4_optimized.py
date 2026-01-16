import numpy as np
from numba import njit, prange

@njit(cache=True)
def compute_financial_weights_numba(abs_returns, volatility):
    """
    Compute financial weights using Numba.
    Now incorporates volatility normalization to prevent procyclicality.

    Args:
        abs_returns: 1D numpy array of absolute returns
        volatility: 1D numpy array of volatility (must be positive)

    Returns:
        1D numpy array of weights
    """
    n = len(abs_returns)

    # volatility regime normalization: |r| / sigma
    # Avoid division by zero with small epsilon
    raw_weights = abs_returns / (volatility + 1e-6)

    # Initial Normalization to mean=1
    total_raw = np.sum(raw_weights) + 1e-9
    weights = raw_weights / total_raw * n

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
    O(N) optimized with Numba using incremental Welford/Sums.
    """
    n = len(returns)
    scores = np.zeros(n, dtype=np.float64)

    # We need at least window points to start producing scores
    # The regression uses pairs (x, y) = (returns[t-1], returns[t])
    # The window size 'window' implies 'window' returns, so 'window-1' pairs.
    n_points = window - 1
    if n_points < 2:
        return scores

    # Accumulators for sums
    sum_x = 0.0
    sum_y = 0.0
    sum_xx = 0.0
    sum_yy = 0.0
    sum_xy = 0.0

    # Initialize sums for the first window (indices 0 to window-1)
    # returns[0...window-1]
    # pairs: (r[0], r[1]), (r[1], r[2]), ..., (r[window-2], r[window-1])

    # Actually, the original code takes returns[i-window : i]
    # So for i=window, it takes returns[0:window].
    # x = returns[0:window-1], y = returns[1:window]

    # Pre-fill for the first window
    for j in range(n_points):
        val_x = returns[j]
        val_y = returns[j+1]

        sum_x += val_x
        sum_y += val_y
        sum_xx += val_x * val_x
        sum_yy += val_y * val_y
        sum_xy += val_x * val_y

    # Now iterate
    # The first score corresponds to i=window.
    # The loop in slow version: for i in range(window, n)

    for i in range(window, n):
        # 1. Compute stats for current window

        # Avoid division by zero
        # Variance of X: E[X^2] - (E[X])^2
        # We need sum_sq_diff_x = sum_xx - (sum_x^2)/N

        term_x = (sum_x * sum_x) / n_points
        sum_sq_diff_x = sum_xx - term_x

        # Covariance: sum_xy - (sum_x * sum_y)/N
        term_xy = (sum_x * sum_y) / n_points
        sum_prod_diff = sum_xy - term_xy

        # Check variance
        # Using a small epsilon for float stability
        if sum_sq_diff_x < 1e-12:
            scores[i] = 0.0
        else:
            beta = sum_prod_diff / sum_sq_diff_x

            # Alpha = mean_y - beta * mean_x
            mean_x = sum_x / n_points
            mean_y = sum_y / n_points
            alpha = mean_y - beta * mean_x

            # RSS = sum((y - alpha - beta*x)^2)
            # Expanded: sum(y^2) - 2*alpha*sum(y) - 2*beta*sum(xy) + N*alpha^2 + 2*alpha*beta*sum(x) + beta^2*sum(x^2)

            rss = sum_yy - 2*alpha*sum_y - 2*beta*sum_xy + \
                  n_points*alpha*alpha + 2*alpha*beta*sum_x + beta*beta*sum_xx

            # Precision issues can make rss slightly negative close to 0
            if rss < 0:
                rss = 0.0

            std_resid = np.sqrt(rss / n_points)

            # Variance of x (population) for the denominator formula used in original
            # var_x = sum_sq_diff_x / n_points

            if std_resid < 1e-12:
                scores[i] = 0.0
            else:
                # denom = std_resid / sqrt(n_points * var_x)
                # var_x = sum_sq_diff_x / n_points
                # sqrt(n_points * sum_sq_diff_x / n_points) = sqrt(sum_sq_diff_x)
                denom = std_resid / np.sqrt(sum_sq_diff_x)

                if denom < 1e-12:
                    scores[i] = 0.0
                else:
                    scores[i] = np.abs(beta / denom)

        # 2. Update sums for next step
        # We are moving from i to i+1.
        # Current window ends at i (exclusive), so returns[i-window : i]
        # Next window ends at i+1 (exclusive), so returns[i-window+1 : i+1]

        # Remove old pair: (returns[i-window], returns[i-window+1])
        # Add new pair: (returns[i-1], returns[i])

        # Be careful with indices.
        # At step i (current loop), we used pairs up to (returns[i-2], returns[i-1]).
        # Wait, let's re-verify the indices.
        # Loop i: range(window, n)
        # slow version: start_idx = i - window, end_idx = i
        # chunk = returns[start_idx : end_idx] (length window)
        # pairs j=0..n_points-1:
        # x = returns[start_idx + j]
        # y = returns[start_idx + j + 1]
        # Last pair j=n_points-1: x = returns[start_idx + window - 2], y = returns[start_idx + window - 1]
        # i.e., x = returns[i-2], y = returns[i-1].

        # So for loop i, the pairs are from index (i-window) to (i-1).

        # Before moving to i+1:
        # We need to remove the pair at start_idx:
        # x_old = returns[i-window], y_old = returns[i-window+1]

        # And add the pair at the end:
        # x_new = returns[i-1], y_new = returns[i]

        # But wait, the loop continues to n-1.
        if i < n - 1:
            # Prepare for next iteration (i+1)
            idx_old = i - window
            x_old = returns[idx_old]
            y_old = returns[idx_old + 1]

            x_new = returns[i-1] # wait, indices?
            # if i=window, next i=window+1.
            # pairs should end at returns[window].
            # so x=returns[window-1], y=returns[window].
            # i-1 = window-1. Correct.
            y_new = returns[i]

            sum_x = sum_x - x_old + x_new
            sum_y = sum_y - y_old + y_new
            sum_xx = sum_xx - x_old*x_old + x_new*x_new
            sum_yy = sum_yy - y_old*y_old + y_new*y_new
            sum_xy = sum_xy - x_old*y_old + x_new*y_new

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
