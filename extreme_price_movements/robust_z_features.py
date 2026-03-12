import numpy as np
import pandas as pd
from numba import njit, prange
import numba

@njit(parallel=True)
def rolling_robust_z_numba(x: np.ndarray, window: int) -> np.ndarray:
    n, m = x.shape
    out = np.full_like(x, np.nan)
    for j in prange(m):
        for i in range(window - 1, n):
            w = x[i - window + 1: i + 1, j]
            valid = w[np.isfinite(w)]
            if len(valid) > 0:
                med = np.median(valid)
                mad = np.median(np.abs(valid - med))
                eps = 1e-6
                out[i, j] = (x[i, j] - med) / (1.4826 * mad + eps)
    return out

def compute_all_robust_z_features(data: pd.DataFrame, feature_dict: dict, bph: float) -> dict:
    """
    Computes all base features and their robust-z for mask optimization.
    Expects data panel format or similar long format stacked data.
    Wait, data here is already flattened stacked data.
    """
    # But robust z requires temporal sequences.
    # Data is stacked as [t1_s1, t1_s2, ..., t2_s1, t2_s2, ...]?
    # Let's check how _apply_regime_search_slice_plan / data_stacked is formed.
    pass
