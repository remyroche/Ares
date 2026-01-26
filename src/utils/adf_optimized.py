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
def _numba_ols_aic(y, X, n_samples):
    """
    Perform OLS and return AIC and t-statistic for the second coefficient (gamma).
    beta = [const, gamma, delta_1, ..., delta_p]
    """
    n_params = X.shape[1]

    # beta = (X.T X)^-1 X.T y
    Xt = X.T
    XtX = Xt @ X
    Xty = Xt @ y

    # Use lstsq for stability, or solve for speed
    # We use solve on normal equations for max speed, assuming full rank
    try:
        beta = np.linalg.solve(XtX, Xty)
    except:
        return 1e9, 0.0 # Error case

    # Calculate RSS
    # rss = (y - X beta).T (y - X beta)
    #     = y.T y - 2 beta.T X.T y + beta.T X.T X beta
    #     = y.T y - beta.T (2 Xty - XtX beta)
    # Or just y - pred

    y_pred = X @ beta
    resid = y - y_pred
    rss = np.sum(resid * resid)

    if rss <= 0:
        return 1e9, 0.0

    # AIC = N * log(RSS/N) + 2k
    # Note: statsmodels uses the provided n_samples (input length) for scaling AIC?
    # Statsmodels: "nobs = len(y)".
    # AIC = nobs * log(rss/nobs) + 2*k
    aic = n_samples * math.log(rss / n_samples) + 2 * n_params

    # Calculate t-stat for gamma (index 1)
    # Var(beta) = sigma^2 * (X.T X)^-1
    sigma2 = rss / (n_samples - n_params)

    # We need the (1,1) element of inverse(XtX)
    # We can compute full inverse
    try:
        inv_XtX = np.linalg.inv(XtX)
        var_gamma = sigma2 * inv_XtX[1, 1]
        if var_gamma > 0:
            se_gamma = math.sqrt(var_gamma)
            t_stat = beta[1] / se_gamma
        else:
            t_stat = 0.0
    except:
        t_stat = 0.0

    return aic, t_stat

@njit(fastmath=True)
def _numba_adf_aic(x, maxlag):
    """
    Calculate ADF t-statistic using AIC to select optimal lag.

    Args:
        x: Input time series (array)
        maxlag: Maximum lag to check

    Returns:
        Best t-statistic (float)
    """
    n = len(x)

    # Prepare common data
    # diff_x = delta y_t
    dx = np.diff(x)

    # x_prev = y_{t-1}
    x_prev = x[:-1]

    # Effective sample size for all regressions must be the same for AIC comparison
    # We must start from index `maxlag`
    # dx array has length n-1. Indices 0..n-2
    # We use dx[maxlag:] as target y

    start_idx = maxlag

    # Slicing from dx
    # dx has length N_diff = n - 1
    # We want dx[t] for t >= maxlag (where t is index in dx)
    # Target vector y
    y_target = dx[start_idx:]
    effective_n = len(y_target)

    if effective_n <= 5:
        return 0.0

    # Pre-allocate design matrix columns
    # We need to construct X for each lag p in 0..maxlag
    # X columns: [const, x_prev, dx_lag1, ..., dx_lagp]
    # All must be aligned to the same y_target

    # Constant column
    col_const = np.ones(effective_n, dtype=np.float64)

    # Level column: x_prev aligned with y_target
    # y_target corresponds to dx[i] for i in start_idx..end
    # dx[i] = x[i+1] - x[i]. It predicts x[i+1] (or rather diff).
    # The regressor is x[i] (which is x_prev[i])
    col_level = x_prev[start_idx:]

    best_aic = 1e15
    best_tstat = 0.0

    # Loop over lags
    for p in range(maxlag + 1):
        # Construct X
        n_cols = 2 + p
        X = np.zeros((effective_n, n_cols), dtype=np.float64)

        X[:, 0] = col_const
        X[:, 1] = col_level

        # Add lag columns
        # Lag 1: dx[t-1]. If target is dx[t], we need dx[t-1].
        # For target index i (in y_target), corresponding global index in dx is `start_idx + i`
        # We need dx[`start_idx + i - 1`]
        # ...
        # Lag k: dx[`start_idx + i - k`]

        # Vectorized fill
        for k in range(1, p + 1):
            # source slice from dx
            # start: start_idx - k
            # end: start_idx - k + effective_n
            s_start = start_idx - k
            s_end = s_start + effective_n
            X[:, 1 + k] = dx[s_start:s_end]

        aic, t_stat = _numba_ols_aic(y_target, X, effective_n)

        if aic < best_aic:
            best_aic = aic
            best_tstat = t_stat

    return best_tstat

__all__ = ['_numba_adf_aic']
