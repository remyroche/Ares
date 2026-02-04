# feature_transforms.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd

import extreme_price_movements.fast_funcs as ff
from .utils import tprint


ArrayLike = Union[pd.Series, pd.DataFrame]


def _ensure_numeric_frame(x: ArrayLike) -> pd.DataFrame:
    """
    Convert Series/DataFrame to a numeric DataFrame; coerce non-numeric to NaN.
    Replace inf with NaN. Keep index/cols stable.
    """
    tprint(f"Entering function: _ensure_numeric_frame with input type: {type(x)}")
    if isinstance(x, pd.Series):
        df = x.to_frame()
    elif isinstance(x, pd.DataFrame):
        df = x
    else:
        raise TypeError(f"Expected Series or DataFrame, got {type(x)}")

    # Coerce to numeric (safe for mixed dtypes) and sanitize infinities.
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def _arcsinh(df: pd.DataFrame) -> pd.DataFrame:
    # Pandas preserves index/cols via ufunc dispatch.
    return np.arcsinh(df)


def _clip_with_bounds(x: pd.DataFrame, lo: pd.DataFrame, hi: pd.DataFrame) -> pd.DataFrame:
    # Align + clip; no axis arg (avoid surprises).
    # Assumes lo/hi share x's index/cols; enforce for safety.
    tprint("Entering function: _clip_with_bounds")
    if not (lo.index.equals(x.index) and lo.columns.equals(x.columns)):
        lo = lo.reindex_like(x)
    if not (hi.index.equals(x.index) and hi.columns.equals(x.columns)):
        hi = hi.reindex_like(x)
    return x.clip(lower=lo, upper=hi)


@dataclass
class CausalFeatureTransformer:
    """
    Monotone transform (arcsinh) + causal winsorization + causal z-score.

    Causality convention:
      - If include_current_in_stats=True: bounds/stats at time t are computed using data <= t.
      - If include_current_in_stats=False: bounds/stats at time t are computed using data <= t-1
        (strictly prior), via a shift(1).

    For most "predict next bar using features at end of bar t", include_current_in_stats=True is fine.
    For "act on same close" or stricter pipelines, set include_current_in_stats=False.
    """

    winsor_qt: float = 0.02
    roll_window: int = 24 * 30
    include_current_in_stats: bool = True

    # Stability controls
    eps: float = 1e-12
    sigma_floor: Optional[float] = None  # e.g. 1e-3
    z_clip: Optional[float] = None       # e.g. 10.0

    # Output dtype
    out_dtype: np.dtype = np.float32

    def transform(self, x: ArrayLike) -> pd.DataFrame:
        tprint("Entering function: transform in feature_transforms.py")

        tprint("Converting to numeric frame...")
        df = _ensure_numeric_frame(x)
        tprint(f"Numeric frame shape: {df.shape}")

        # 1) Monotone transform (robust to 0/negatives)
        x0 = _arcsinh(df)

        # Decide what history is allowed to inform bounds/stats at time t
        hist = x0 if self.include_current_in_stats else x0.shift(1)

        # 2) Causal winsorization using rolling quantiles on allowed history
        tprint("Calculating causal winsorization bounds (rolling quantile)...")
        lo = ff.numba_rolling_quantile(hist, self.roll_window, self.winsor_qt)
        hi = ff.numba_rolling_quantile(hist, self.roll_window, 1.0 - self.winsor_qt)

        # Forward-fill bounds across gaps (still causal)
        lo = lo.ffill()
        hi = hi.ffill()

        tprint("Clipping with bounds...")
        x1 = _clip_with_bounds(x0, lo, hi)

        # 3) Causal z-score: rolling mean/std on allowed history of clipped values
        hist2 = x1 if self.include_current_in_stats else x1.shift(1)

        tprint("Calculating causal z-score stats (mean/std)...")
        mu = ff.numba_rolling_mean(hist2, self.roll_window)
        sigma = ff.numba_rolling_std(hist2, self.roll_window)

        if self.sigma_floor is not None:
            # Floor in-place via vectorized clip
            sigma = sigma.clip(lower=self.sigma_floor)

        tprint("Finalizing z-score computation...")
        z = (x1 - mu) / (sigma + self.eps)

        if self.z_clip is not None:
            z = z.clip(lower=-float(self.z_clip), upper=float(self.z_clip))

        # Preserve original shape; cast once at the end
        return z.astype(self.out_dtype, copy=False)


def log_winsor_zscore_rolling(
    series: pd.Series,
    window: int = 720,
    qt: float = 0.02,
    include_current_in_stats: bool = True,
    eps: float = 1e-12,
    sigma_floor: Optional[float] = None,
    z_clip: Optional[float] = None,
    out_dtype: np.dtype = np.float32,
) -> pd.Series:
    """
    Single-series helper with the same semantics as CausalFeatureTransformer.

    Vectorization notes:
      - Avoids repeated to_frame conversions.
      - Uses DataFrame-based ff.* once, then returns Series.
    """
    tprint("Entering function: log_winsor_zscore_rolling in feature_transforms.py")

    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x0 = np.arcsinh(s)

    hist = x0 if include_current_in_stats else x0.shift(1)

    # Compute bounds via DataFrame interface expected by ff.*
    tprint("Calculating winsorization bounds...")
    hist_df = hist.to_frame(name=series.name if series.name is not None else "x")

    lo_df = ff.numba_rolling_quantile(hist_df, window, qt).ffill()
    hi_df = ff.numba_rolling_quantile(hist_df, window, 1.0 - qt).ffill()

    # Extract bounds as Series (aligned to index)
    lo = lo_df.iloc[:, 0]
    hi = hi_df.iloc[:, 0]

    tprint("Clipping series...")
    x1 = x0.clip(lower=lo, upper=hi)

    hist2 = x1 if include_current_in_stats else x1.shift(1)
    hist2_df = hist2.to_frame(name=hist_df.columns[0])

    tprint("Calculating z-score stats...")
    mu = ff.numba_rolling_mean(hist2_df, window).iloc[:, 0]
    sigma = ff.numba_rolling_std(hist2_df, window).iloc[:, 0]

    if sigma_floor is not None:
        sigma = sigma.clip(lower=sigma_floor)

    tprint("Finalizing z-score...")
    z = (x1 - mu) / (sigma + eps)

    if z_clip is not None:
        z = z.clip(lower=-float(z_clip), upper=float(z_clip))

    return z.astype(out_dtype, copy=False)
