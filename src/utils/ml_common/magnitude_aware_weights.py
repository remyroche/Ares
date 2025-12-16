import numpy as np
import pandas as pd
from typing import Optional, Union


def compute_magnitude_aware_classification_weights(
    *,
    realized_returns: Union[pd.Series, np.ndarray],
    risk_unit: Optional[Union[pd.Series, np.ndarray]] = None,
    base_weights: Optional[Union[pd.Series, np.ndarray]] = None,
    mag_cap: float = 3.0,
    alpha: float = 1.0,
    w_min: float = 0.05,
    w_max: float = 20.0,
    normalize_mean: bool = True,
    eps: float = 1e-8,
) -> np.ndarray:
    r = np.asarray(realized_returns, dtype=float).reshape(-1)

    if risk_unit is None:
        ru = np.full_like(r, np.nan, dtype=float)
    else:
        ru = np.asarray(risk_unit, dtype=float).reshape(-1)

    if ru.shape[0] != r.shape[0]:
        ru = np.full_like(r, np.nan, dtype=float)

    ru = np.abs(ru)
    ru = np.where(np.isfinite(ru) & (ru > eps), ru, np.nan)

    fallback_ru = np.nanmedian(np.abs(r[np.isfinite(r)]))
    if (not np.isfinite(fallback_ru)) or fallback_ru <= eps:
        fallback_ru = 0.01

    ru = np.where(np.isfinite(ru), ru, fallback_ru)

    x = np.abs(r) / (ru + eps)
    x = np.where(np.isfinite(x), x, 0.0)

    mag_cap = float(mag_cap)
    if (not np.isfinite(mag_cap)) or mag_cap <= 0.0:
        mag_cap = 3.0

    alpha = float(alpha)
    if not np.isfinite(alpha):
        alpha = 1.0

    x_clip = np.clip(x, 0.0, mag_cap)
    mag = np.log1p(x_clip)
    w_mag = 1.0 + alpha * mag

    w = w_mag

    if base_weights is not None:
        bw = np.asarray(base_weights, dtype=float).reshape(-1)
        if bw.shape[0] == r.shape[0]:
            bw = np.where(np.isfinite(bw), bw, 1.0)
            w = w * bw

    w = np.where(np.isfinite(w), w, 1.0)

    w_min = float(w_min)
    w_max = float(w_max)
    if (not np.isfinite(w_min)) or w_min <= 0.0:
        w_min = 0.05
    if (not np.isfinite(w_max)) or w_max <= w_min:
        w_max = max(20.0, w_min * 2.0)

    w = np.clip(w, w_min, w_max)

    if normalize_mean:
        mean_w = float(np.mean(w)) if w.size else 1.0
        if np.isfinite(mean_w) and mean_w > eps:
            w = w / mean_w

    return w
