import numpy as np
import pandas as pd
import extreme_price_movements.fast_funcs as ff
from extreme_price_movements.fast_funcs import _numba_rolling_quantile_dual_parallel, _numba_rolling_zscore_parallel
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
        Optimized to minimize pandas object churn.
        """
        # tprint(f"Entering function: transform in feature_transforms.py")

        # 1. Log transform (signed log for negative values handling)
        # Convert to contiguous float32 array once to stay in numpy land
        mat = np.ascontiguousarray(df.to_numpy(dtype=np.float32, copy=False))

        # Arcsinh transform
        mat = np.arcsinh(mat, dtype=np.float32)

        # 2. Causal Winsorization (Rolling Quantile)
        # Use parallel kernel directly
        # Returns lower, upper bounds as 2D arrays (same shape as mat)
        lower, upper = _numba_rolling_quantile_dual_parallel(mat, self.roll_window, self.winsor_qt, 1.0 - self.winsor_qt)

        # Clip using mask to avoid ffill and full array operations
        # "Do causal clip only when bounds are defined; otherwise leave value as-is."
        mask = ~np.isnan(lower) & ~np.isnan(upper)

        # Only clip where bounds are valid
        if np.any(mask):
            # Advanced indexing assignment works efficiently
            mat[mask] = np.clip(mat[mask], lower[mask], upper[mask])

        # 3. Causal Z-Score (Rolling Mean/Std fused)
        # Use parallel z-score kernel which computes z-score in one pass
        z = _numba_rolling_zscore_parallel(mat, self.roll_window)

        # 4. Wrap result back to DataFrame
        out_df = pd.DataFrame(z, index=df.index, columns=df.columns)

        # Optional check
        if self.debug:
            check_inf_nan(out_df, name)

        return out_df

def log_winsor_zscore_rolling(series: pd.Series, window: int = 720, qt: float = 0.02) -> pd.Series:
    """Helper for single series causal transform"""
    tprint(f"Entering function: log_winsor_zscore_rolling in feature_transforms.py")

    # Treat as 1-column matrix
    vals = series.to_numpy(dtype=np.float32)
    mat = vals.reshape(-1, 1)

    # Log
    mat = np.arcsinh(mat, dtype=np.float32)

    # Quantiles
    lower, upper = _numba_rolling_quantile_dual_parallel(mat, window, qt, 1.0 - qt)

    # Clip
    mask = ~np.isnan(lower) & ~np.isnan(upper)
    if np.any(mask):
        mat[mask] = np.clip(mat[mask], lower[mask], upper[mask])

    # Z-Score
    z = _numba_rolling_zscore_parallel(mat, window)

    # Return Series
    return pd.Series(z.ravel(), index=series.index, name=series.name)
