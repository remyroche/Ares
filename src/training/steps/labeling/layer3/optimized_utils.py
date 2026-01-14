"""
Optimized Utils for Layer 3 - Numba & Vectorized functions

This module provides Numba-optimized functions for Layer 3 operations,
specifically targeting heavy computations like Rolling OLS (HAR),
Conditional Mutual Information (CMI), and other statistical features.
"""

import numpy as np
from numba import njit, prange

@njit(fastmath=True)
def numba_rolling_ols_3factor(y: np.ndarray, x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, window: int) -> np.ndarray:
    """
    Perform a rolling OLS regression with 3 independent variables (and constant).
    Y = b0 + b1*x1 + b2*x2 + b3*x3 + e

    This is an O(N) implementation using incremental updates for X'X and X'y matrices if possible,
    but for stability and simplicity with Numba, we use a sliding window approach with direct solver.
    Since K=4 (small), direct inversion of (4x4) matrix per step is very fast.

    Args:
        y: Target array (N,)
        x1: Predictor 1 (N,)
        x2: Predictor 2 (N,)
        x3: Predictor 3 (N,)
        window: Rolling window size

    Returns:
        y_hat: Predicted values (N,)
    """
    n = len(y)
    y_hat = np.zeros(n, dtype=np.float64)

    # We need at least 'window' samples
    if n < window:
        return y_hat

    # Pre-allocate X matrix for the window (window x 4)
    # [1, x1, x2, x3]
    X_window = np.ones((window, 4), dtype=np.float64)

    for i in range(window, n):
        # Fill window data
        start_idx = i - window
        end_idx = i

        # We want to predict y[i] based on coefficients from [i-window : i]
        # BUT standard rolling regression predicts y[t] using model fit on [t-window+1 : t+1] (inclusive)
        # OR it predicts y[t+1] using model fit up to t.
        # HAR logic usually: Regress R_{t+1} on R_d, R_w, R_m (known at t).
        # So we fit on past pairs (X_{t-k}, Y_{t-k}) and apply to X_t.
        # But here inputs are already shifted?
        # In calculate_studentized_har_target:
        # X = [var_d.shift(1), var_w.shift(1), var_m.shift(1)]
        # This means X[i] contains variances from yesterday, predicting Returns[i].
        # So at time i, we can use data up to i-1 to fit the model, and then apply to X[i].
        # Or usually rolling OLS includes current observation in the fit?
        # Standard: RollingOLS at index i uses data from i-window+1 to i.
        # We will do that.

        # Extract Y and X for the window
        Y_sub = y[start_idx:end_idx]

        X_window[:, 1] = x1[start_idx:end_idx]
        X_window[:, 2] = x2[start_idx:end_idx]
        X_window[:, 3] = x3[start_idx:end_idx]
        # X_window[:, 0] is already 1s

        # Solve (X'X)b = X'Y
        # XT_X = X_window.T @ X_window
        # XT_Y = X_window.T @ Y_sub

        # Manual matrix multiplication for small fixed size is often faster or just use np.dot
        XT = X_window.T
        XT_X = np.dot(XT, X_window)
        XT_Y = np.dot(XT, Y_sub)

        # Add small ridge to diagonal for stability
        for k in range(4):
            XT_X[k, k] += 1e-6

        # Solve for betas
        try:
            betas = np.linalg.solve(XT_X, XT_Y)

            # Predict for current step i
            # The features for step i are x1[i], x2[i], x3[i]
            # y_hat[i] = b0 + b1*x1[i] + b2*x2[i] + b3*x3[i]
            y_hat[i] = betas[0] + betas[1]*x1[i] + betas[2]*x2[i] + betas[3]*x3[i]

        except:
            # Singular matrix or other error
            y_hat[i] = 0.0

    return y_hat

@njit(parallel=True, fastmath=True)
def numba_conditional_correlation(
    X_val: np.ndarray,
    y_val: np.ndarray,
    base_preds: np.ndarray,
    top_indices: np.ndarray
) -> np.ndarray:
    """
    Calculate conditional correlation for selected features in parallel.
    Corr(Feature - BasePred, Target - BasePred)

    Args:
        X_val: Feature matrix (N, F)
        y_val: Target vector (N,)
        base_preds: Base predictions vector (N,)
        top_indices: Indices of features to check

    Returns:
        Array of conditional correlation scores
    """
    n_selected = len(top_indices)
    scores = np.zeros(n_selected, dtype=np.float64)

    # Calculate target residual once
    res_y = y_val - base_preds
    std_y = np.std(res_y)
    if std_y < 1e-9:
        return scores # All zeros

    res_y_norm = (res_y - np.mean(res_y)) / std_y

    # Iterate over selected features in parallel
    for i in prange(n_selected):
        feat_idx = top_indices[i]
        feat_col = X_val[:, feat_idx]

        res_feat = feat_col - base_preds
        std_feat = np.std(res_feat)

        if std_feat > 1e-9:
            res_feat_norm = (res_feat - np.mean(res_feat)) / std_feat
            # Correlation = E[XY] for standardized variables
            corr = np.mean(res_feat_norm * res_y_norm)
            scores[i] = np.abs(corr)
        else:
            scores[i] = 0.0

    return scores

@njit(fastmath=True)
def numba_calculate_har_features(volatility: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """
    Calculate HAR variance features (d, w, m) efficiently.

    Args:
        volatility: Volatility array
        periods: Array of 3 integers [d, w, m] e.g. [96, 480, 1920]

    Returns:
        Matrix of shape (N, 3) with lagged variance features
    """
    n = len(volatility)
    variance = volatility ** 2
    output = np.zeros((n, 3), dtype=np.float64)

    for j in range(3):
        p = periods[j]
        # Compute rolling mean
        # Using simple cumulative sum trick for O(N) rolling mean
        # cumsum[i] - cumsum[i-p]

        current_sum = 0.0
        # Initialize first window
        for i in range(n):
            current_sum += variance[i]
            if i >= p:
                current_sum -= variance[i-p]
                output[i, j] = current_sum / p
            elif i >= 0:
                output[i, j] = current_sum / (i + 1)

    # Shift by 1 as required (lagged features)
    # output[i] should contain mean variance up to i-1
    # We can do this by shifting the array down
    res = np.zeros_like(output)
    res[1:, :] = output[:-1, :]

    return res
