import numpy as np
import pandas as pd
import extreme_price_movements.fast_funcs as ff
from .utils import tprint, check_inf_nan

class CausalFeatureTransformer:
    def __init__(self, winsor_qt=0.02, roll_window=24*30, debug=False):
        tprint(f"Entering function: __init__ in feature_transforms.py")
        self.winsor_qt = winsor_qt
        self.roll_window = roll_window
        self.debug = debug

    def transform(self, df: pd.DataFrame, name: str = "unknown") -> pd.DataFrame:
        """
        Applies Log + Causal Winsorization + Causal Z-Score.
        Optimized to use contiguous float32 numpy arrays and fused kernels to minimize memory traffic.
        """
        # tprint(f"Entering function: transform in feature_transforms.py")

        # 1. Convert to contiguous float32 matrix
        # Avoid pandas overhead for intermediate steps
        mat = np.ascontiguousarray(df.to_numpy(dtype=np.float32, copy=False))

        # 2. Log transform
        # Using arcsinh is safer for 0 and negative values than log.
        mat = np.arcsinh(mat)

        # 3. Rolling Quantiles
        # Call underlying parallel kernel directly to stay in numpy
        # Returns two matrices (float32)
        # Note: we use (1.0 - winsor_qt) to ensure float type for numba if qt is float
        lo_mat, hi_mat = ff._numba_rolling_quantile_dual_parallel(
            mat, self.roll_window, self.winsor_qt, 1.0 - self.winsor_qt
        )

        # 4. Causal Clip (with fused ffill logic inside)
        mat = ff._numba_causal_clip_parallel(mat, lo_mat, hi_mat)

        # 5. Causal Z-Score (Fused Mean/Std/Z)
        mat = ff._numba_rolling_zscore_parallel(mat, self.roll_window)

        # Wrap back to DataFrame
        z = pd.DataFrame(mat, index=df.index, columns=df.columns)

        # Check inf/nan
        if self.debug:
            check_inf_nan(z, name)

        return z

def log_winsor_zscore_rolling(series: pd.Series, window: int = 720, qt: float = 0.02) -> pd.Series:
    """Helper for single series causal transform"""
    tprint(f"Entering function: log_winsor_zscore_rolling in feature_transforms.py")

    # 1. To numpy (avoiding DataFrame roundtrips)
    arr = series.to_numpy(dtype=np.float32, copy=False)

    # 2. Arcsinh
    arr = np.arcsinh(arr)

    # 3. Quantiles (use 1D helper directly)
    n = len(arr)
    lo = np.empty(n, dtype=np.float32)
    hi = np.empty(n, dtype=np.float32)
    ff._numba_rolling_quantile_dual_1d(arr, window, qt, 1.0-qt, lo, hi)

    # 4. Clip (use 1D helper)
    arr = ff._numba_causal_clip_with_ffill_1d(arr, lo, hi)

    # 5. Z-Score (use 1D helper)
    arr = ff._numba_rolling_zscore_nan_safe_1d(arr, window)

    return pd.Series(arr, index=series.index, name=series.name)
