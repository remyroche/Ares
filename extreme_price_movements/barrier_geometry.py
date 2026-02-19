from __future__ import annotations

import math
from typing import Union

import numpy as np
import pandas as pd

EPS = 1e-12

ArrayLike = Union[pd.DataFrame, np.ndarray, float]


def apply_horizon_scaling(
    values: ArrayLike,
    *,
    horizon: int,
    scaling: str = "none",
    alpha: float = 0.5,
    base: float = 4.0,
) -> ArrayLike:
    """Scale barrier geometry by horizon using a shared canonical rule."""
    if scaling == "none":
        scale = 1.0
    elif scaling == "sqrt":
        scale = math.sqrt(max(float(horizon), EPS) / max(float(base), EPS))
    elif scaling == "power":
        scale = (max(float(horizon), EPS) / max(float(base), EPS)) ** float(alpha)
    else:
        raise ValueError(f"Unknown horizon scaling: {scaling}")
    return values * scale


def make_effective_tp(
    tp_raw: ArrayLike,
    *,
    horizon: int,
    horizon_scaling: str,
    lo: float,
    hi: float,
    horizon_alpha: float = 0.5,
    horizon_base: float = 4.0,
) -> ArrayLike:
    """Canonical TP post-processing: horizon-scale then clamp to absolute bounds."""
    tp_scaled = apply_horizon_scaling(
        tp_raw,
        horizon=horizon,
        scaling=horizon_scaling,
        alpha=horizon_alpha,
        base=horizon_base,
    )
    if isinstance(tp_scaled, pd.DataFrame):
        return tp_scaled.clip(lower=float(lo), upper=float(hi))
    return np.clip(tp_scaled, float(lo), float(hi))


def effective_tp_floor(*, tp_abs_lo_pct: float, tp_min_abs_pct: float, tp_min_bps: float) -> float:
    """Canonical effective TP floor in return units."""
    return max(float(tp_abs_lo_pct), float(tp_min_abs_pct), float(tp_min_bps) / 10000.0)


def effective_sl_floor(*, sl_abs_lo_pct: float, sl_min_abs_pct: float, sl_min_bps: float) -> float:
    """Canonical effective SL floor in return units."""
    return max(float(sl_abs_lo_pct), float(sl_min_abs_pct), float(sl_min_bps) / 10000.0)
