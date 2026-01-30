"""
Numba-optimized Augmented Dickey-Fuller (ADF) test.

This module provides a high-performance implementation of the ADF test
using Numba JIT compilation, specifically designed for adaptive fractional
differencing applications where speed is critical.
"""

import numpy as np
import math

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

@njit(fastmath=True)
def _numba_adf_aic(x, maxlag):
    """
    Calculate ADF t-statistic using AIC to select optimal lag.

    Optimized implementation:
    1. Pre-computes the full design matrix X for the largest lag.
    2. Computes the full XtX and Xty matrices once (O(N*K^2)).
    3. Iterates through lags by slicing the pre-computed matrices (O(K^4)).

    This avoids re-allocating and re-filling the design matrix inside the loop,
    reducing complexity from O(N*K^3) to O(N*K^2 + K^4). Since N (window size)
    is typically much larger than K (maxlag), this yields significant speedups.

    Args:
        x: Input time series (array)
        maxlag: Maximum lag to check

    Returns:
        Best t-statistic (float)
    """
    n = len(x)

    # Calculate differences and previous values
    dx = np.diff(x)
    x_prev = x[:-1]

    # The target vector y must be fixed for all lag comparisons (AIC requirement).
    # We start from index `maxlag` in the differenced series.
    start_idx = maxlag
    y_target = dx[start_idx:]
    effective_n = len(y_target)

    if effective_n <= 5:
        return 0.0

    # Pre-allocate full design matrix X
    # Columns: [const, level, lag_1, ..., lag_max]
    n_cols_total = maxlag + 2
    X_full = np.zeros((effective_n, n_cols_total), dtype=np.float64)

    # Fill constant (col 0) and level (col 1)
    X_full[:, 0] = 1.0
    X_full[:, 1] = x_prev[start_idx:]

    # Fill lag columns (col 2 to maxlag+1)
    # lag k corresponds to dx[start_idx-k : start_idx-k+effective_n]
    for k in range(1, maxlag + 1):
        s_start = start_idx - k
        s_end = s_start + effective_n
        X_full[:, 1 + k] = dx[s_start:s_end]

    # Compute full moment matrices once
    # This is the heavy O(N*K^2) operation
    XtX_full = X_full.T @ X_full
    Xty_full = X_full.T @ y_target
    yy = np.dot(y_target, y_target)

    best_aic = 1e15
    best_tstat = 0.0

    # Iterate over lags p from 0 to maxlag
    # For a given p, we use the first p+2 columns of X
    for p in range(maxlag + 1):
        k = p + 2

        # Check degrees of freedom
        if effective_n <= k:
            continue

        # Slice the pre-computed matrices
        # In Numba/NumPy, this creates a view (very cheap)
        XtX = XtX_full[:k, :k]
        Xty = Xty_full[:k]

        # Solve OLS: XtX * beta = Xty
        try:
            beta = np.linalg.solve(XtX, Xty)
        except:
            continue

        # Calculate RSS
        # RSS = (y - Xb)'(y - Xb) = y'y - 2b'X'y + b'X'Xb
        # Since X'Xb = X'y (normal equations), RSS = y'y - b'X'y
        rss = yy - np.dot(beta, Xty)

        if rss <= 0:
            continue

        # Calculate AIC
        # AIC = n * log(RSS/n) + 2k
        aic = effective_n * math.log(rss / effective_n) + 2 * k

        if aic < best_aic:
            best_aic = aic

            # Calculate t-statistic for the level coefficient (gamma, index 1)
            # t = beta[1] / se_gamma
            # se_gamma = sqrt(sigma^2 * (XtX)^-1[1,1])
            sigma2 = rss / (effective_n - k)

            # We need the (1,1) element of the inverse of XtX.
            # We can solve XtX * z = e1, where e1 = [0, 1, 0, ...], then z[1] is inv(XtX)[1,1].
            # This is O(K^3) but K is small.

            # Construct e1
            e1 = np.zeros(k)
            e1[1] = 1.0

            try:
                z = np.linalg.solve(XtX, e1)
                inv_XtX_11 = z[1]

                if sigma2 * inv_XtX_11 > 0:
                    se_gamma = math.sqrt(sigma2 * inv_XtX_11)
                    best_tstat = beta[1] / se_gamma
                else:
                    best_tstat = 0.0
            except:
                best_tstat = 0.0

    return best_tstat

__all__ = ['_numba_adf_aic']
