import numpy as np
import pandas as pd

class CausalFeatureTransformer:
    def __init__(self, winsor_qt=0.02, roll_window=24*30):
        self.winsor_qt = winsor_qt
        self.roll_window = roll_window

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies Log + Causal Winsorization + Causal Z-Score.
        df: DataFrame (time x features) or (time x symbols) for a single feature?
        Usually features are wide panels (time x symbols) per feature key.
        """
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

        lower = out.rolling(self.roll_window, min_periods=24).quantile(self.winsor_qt)
        upper = out.rolling(self.roll_window, min_periods=24).quantile(1 - self.winsor_qt)

        # Forward fill limits to handle gaps
        lower = lower.ffill()
        upper = upper.ffill()

        # Clip
        # Ideally we clip row by row, but vectorized clip with other dfs:
        # out.clip(lower=lower, upper=upper) works in pandas.
        out = out.clip(lower=lower, upper=upper, axis=0)

        # 3. Causal Z-Score (Rolling Mean/Std)
        mu = out.rolling(self.roll_window, min_periods=24).mean()
        sigma = out.rolling(self.roll_window, min_periods=24).std(ddof=0)

        z = (out - mu) / (sigma + 1e-12)

        # Fill initial NaNs with 0 or drop?
        # We keep them as NaNs, let downstream handle.
        return z.astype(np.float32)

def log_winsor_zscore_rolling(series: pd.Series, window: int = 720, qt: float = 0.02) -> pd.Series:
    """Helper for single series causal transform"""
    x = np.arcsinh(series)
    lo = x.rolling(window, min_periods=24).quantile(qt)
    hi = x.rolling(window, min_periods=24).quantile(1-qt)
    x = x.clip(lower=lo, upper=hi)
    mu = x.rolling(window, min_periods=24).mean()
    sd = x.rolling(window, min_periods=24).std()
    return (x - mu) / (sd + 1e-12)
