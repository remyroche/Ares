import numpy as np
from numba import njit, prange

@njit(parallel=False)
def rolling_chow_test_numba(
    errors: np.ndarray,
    window: int,
    threshold_mean: float = 2.0,
    threshold_var: float = 3.0
) -> np.ndarray:
    """
    Perform rolling Chow test for structural breaks using Numba.

    Args:
        errors: Array of prediction errors
        window: Window size for break detection
        threshold_mean: Threshold for mean difference (in standard deviations)
        threshold_var: Threshold for variance ratio

    Returns:
        Boolean array indicating structural breaks
    """
    n = len(errors)
    breaks = np.zeros(n, dtype=np.int8)

    if n < 2 * window:
        return breaks

    # Pre-calculate rolling statistics could be faster, but let's do single pass
    # for memory efficiency and cache locality.

    for i in prange(window, n - window):
        # Slice windows
        # Note: In Numba, slicing creates views or copies depending on context.
        # Direct indexing is often safer for performance.

        # Calculate stats for 'before' window [i-window : i]
        mean_before = 0.0
        var_before = 0.0
        for j in range(i - window, i):
            mean_before += errors[j]
        mean_before /= window

        for j in range(i - window, i):
            diff = errors[j] - mean_before
            var_before += diff * diff
        var_before /= window

        # Calculate stats for 'after' window [i : i+window]
        mean_after = 0.0
        var_after = 0.0
        for j in range(i, i + window):
            mean_after += errors[j]
        mean_after /= window

        for j in range(i, i + window):
            diff = errors[j] - mean_after
            var_after += diff * diff
        var_after /= window

        # Test statistics
        mean_diff = abs(mean_before - mean_after)

        # Avoid division by zero
        if var_before < 1e-9: var_before = 1e-9
        if var_after < 1e-9: var_after = 1e-9

        var_ratio = max(var_before, var_after) / min(var_before, var_after)

        # Check conditions
        std_before = np.sqrt(var_before)
        if mean_diff > threshold_mean * std_before or var_ratio > threshold_var:
            breaks[i] = 1

    return breaks
