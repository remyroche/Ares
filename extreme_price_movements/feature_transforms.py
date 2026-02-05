import numpy as np
import pandas as pd
import extreme_price_movements.fast_funcs as ff
from .utils import tprint, check_inf_nan

class CausalFeatureTransformer:
    def __init__(self, winsor_qt=0.02, roll_window=24*30):
        # tprint(f"Entering function: __init__ in feature_transforms.py")
        self.winsor_qt = winsor_qt
        self.roll_window = roll_window

    def transform_numpy(self, mat: np.ndarray, name: str = "unknown") -> np.ndarray:
        """
        Numpy/Numba optimized transform.
        """
        # 1. Log transform
        # Using arcsinh is safer for 0 and negative values than log.
        out = np.arcsinh(mat)

        # 2. Causal Winsorization (Rolling Quantile)
        lower, upper = ff._numba_rolling_quantile_dual_parallel(out, self.roll_window, self.winsor_qt, 1 - self.winsor_qt)

        # Note: We skip ffill() here as it is expensive and numba quantiles are usually dense after warm-up.
        # NaNs in lower/upper (at start) will result in NaNs in clipped output, which is correct/expected behavior.

        # Clip
        # np.clip handles element-wise min/max arrays
        out = np.clip(out, lower, upper)

        # 3. Causal Z-Score (Rolling Mean/Std)
        mu = ff._numba_rolling_mean_mat(out, self.roll_window)
        sigma = ff._numba_rolling_std_mat(out, self.roll_window)

        z = (out - mu) / (sigma + 1e-12)

        return z.astype(np.float32)

    def transform(self, df: pd.DataFrame, name: str = "unknown") -> pd.DataFrame:
        """
        Applies Log + Causal Winsorization + Causal Z-Score.
        """
        # tprint(f"Entering function: transform in feature_transforms.py")
        mat = df.to_numpy(dtype=np.float32)
        res = self.transform_numpy(mat, name)

        df_res = pd.DataFrame(res, index=df.index, columns=df.columns)
        check_inf_nan(df_res, name)
        return df_res

def log_winsor_zscore_rolling(series: pd.Series, window: int = 720, qt: float = 0.02) -> pd.Series:
    """Helper for single series causal transform"""
    tprint(f"Entering function: log_winsor_zscore_rolling in feature_transforms.py")
    x = np.arcsinh(series)
    x_df = x.to_frame()

    lo_df, hi_df = ff.numba_rolling_quantile_dual(x_df, window, qt, 1-qt)

    lo = lo_df[lo_df.columns[0]]
    hi = hi_df[hi_df.columns[0]]

    x = x.clip(lower=lo, upper=hi)

    x_df = x.to_frame()
    mu_df = ff.numba_rolling_mean(x_df, window)
    sd_df = ff.numba_rolling_std(x_df, window)

    mu = mu_df[mu_df.columns[0]]
    sd = sd_df[sd_df.columns[0]]

    return (x - mu) / (sd + 1e-12)
