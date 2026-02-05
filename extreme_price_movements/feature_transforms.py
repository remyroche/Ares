import numpy as np
import pandas as pd
import extreme_price_movements.fast_funcs as ff
from .utils import tprint, check_inf_nan

class CausalFeatureTransformer:
    def __init__(self, winsor_qt=0.02, roll_window=24*30):
        tprint(f"Entering function: __init__ in feature_transforms.py")
        self.winsor_qt = winsor_qt
        self.roll_window = roll_window

    def transform(self, df: pd.DataFrame, name: str = "unknown") -> pd.DataFrame:
        """
        Applies Log + Causal Winsorization + Causal Z-Score.
        df: DataFrame (time x features) or (time x symbols) for a single feature?
        Usually features are wide panels (time x symbols) per feature key.
        """
        # tprint(f"Entering function: transform in feature_transforms.py")
        # 1. Log transform (signed log for negative values handling)
        # Using arcsinh is safer for 0 and negative values than log.
        # But if user insists on log, maybe log1p? Arcsinh is standard for this.
        # "scaled/normalized with log"

        # We need to handle features that are already ratios (ret, etc).
        # Assuming arcsinh is a good default.
        out = np.arcsinh(df)

        # 2. Causal Winsorization (Rolling Quantile)
        # This is expensive (O(N*W)).
        # Approximated by expanding window or fixed window?
        # User said "winsorisation (top 2%)".
        # Let's use rolling quantile.

        lower = ff.numba_rolling_quantile(out, self.roll_window, self.winsor_qt)
        upper = ff.numba_rolling_quantile(out, self.roll_window, 1 - self.winsor_qt)

        # Forward fill limits to handle gaps
        lower = lower.ffill()
        upper = upper.ffill()

        # Clip
        # Ideally we clip row by row, but vectorized clip with other dfs:
        # out.clip(lower=lower, upper=upper) works in pandas.
        out = out.clip(lower=lower, upper=upper, axis=0)

        # 3. Causal Z-Score (Rolling Mean/Std)
        mu = ff.numba_rolling_mean(out, self.roll_window)
        sigma = ff.numba_rolling_std(out, self.roll_window)

        z = (out - mu) / (sigma + 1e-12)

        # Fill initial NaNs with 0 or drop?
        # We keep them as NaNs, let downstream handle.
        z = z.astype(np.float32)
        check_inf_nan(z, name)
        return z

def log_winsor_zscore_rolling(series: pd.Series, window: int = 720, qt: float = 0.02) -> pd.Series:
    """Helper for single series causal transform"""
    tprint(f"Entering function: log_winsor_zscore_rolling in feature_transforms.py")
    x = np.arcsinh(series)
    x_df = x.to_frame()

    lo_df = ff.numba_rolling_quantile(x_df, window, qt)
    hi_df = ff.numba_rolling_quantile(x_df, window, 1-qt)

    lo = lo_df[lo_df.columns[0]]
    hi = hi_df[hi_df.columns[0]]

    x = x.clip(lower=lo, upper=hi)

    x_df = x.to_frame()
    mu_df = ff.numba_rolling_mean(x_df, window)
    sd_df = ff.numba_rolling_std(x_df, window)

    mu = mu_df[mu_df.columns[0]]
    sd = sd_df[sd_df.columns[0]]

    return (x - mu) / (sd + 1e-12)
