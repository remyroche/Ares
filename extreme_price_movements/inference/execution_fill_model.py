"""Shared execution fill models for policy optimisation and live replay.

The functions in this module are intentionally small and deterministic. They
exist so optimiser/backtest code and live replay code price market/stop exits
with the same adverse gap assumptions.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def stop_exit_fill_price(
    *,
    side: str,
    stop_px: float,
    candle_high: float,
    candle_low: float,
    base_gap_bps: float,
    alpha_through: float,
    max_gap_bps: float,
    quote_half_spread_bps: float = 0.0,
) -> Tuple[bool, float]:
    """Return whether a stop crossed and the modeled market-stop fill price."""
    stop = float(stop_px)
    high = float(candle_high)
    low = float(candle_low)
    if not np.isfinite(stop) or stop <= 0.0 or not np.isfinite(high) or not np.isfinite(low):
        return False, float("nan")

    base_gap = stop * float(base_gap_bps) / 10000.0
    max_gap = stop * float(max_gap_bps) / 10000.0
    quote_half_spread = max(0.0, float(quote_half_spread_bps)) / 10000.0
    side_l = str(side).lower()
    if side_l == "long":
        low = low - stop * quote_half_spread
        if low > stop:
            return False, float("nan")
        through = max(stop - low, 0.0)
        gap = min(base_gap + float(alpha_through) * through, max_gap)
        return True, float(stop - gap)
    if side_l == "short":
        high = high + stop * quote_half_spread
        if high < stop:
            return False, float("nan")
        through = max(high - stop, 0.0)
        gap = min(base_gap + float(alpha_through) * through, max_gap)
        return True, float(stop + gap)
    raise ValueError(side)


def stop_exit_fill_price_array(
    *,
    side: np.ndarray,
    stop_px: np.ndarray,
    candle_high: np.ndarray,
    candle_low: np.ndarray,
    base_gap_bps: float,
    alpha_through: float,
    max_gap_bps: float,
    quote_half_spread_bps: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized variant of :func:`stop_exit_fill_price`.

    `side` follows the existing optimiser convention: non-negative means long,
    negative means short.
    """
    side_arr = np.asarray(side, dtype=np.float32)
    stop = np.asarray(stop_px, dtype=np.float64)
    high = np.asarray(candle_high, dtype=np.float64)
    low = np.asarray(candle_low, dtype=np.float64)
    is_long = side_arr >= 0.0
    is_short = ~is_long
    quote_half_spread = max(0.0, float(quote_half_spread_bps)) / 10000.0
    high_trigger = np.where(is_long, high - stop * quote_half_spread, high + stop * quote_half_spread)
    low_trigger = np.where(is_long, low - stop * quote_half_spread, low + stop * quote_half_spread)
    finite = np.isfinite(stop) & (stop > 0.0) & np.isfinite(high) & np.isfinite(low)
    hit = finite & ((is_long & (low_trigger <= stop)) | (is_short & (high_trigger >= stop)))
    through = np.where(
        is_long,
        np.maximum(stop - low_trigger, 0.0),
        np.maximum(high_trigger - stop, 0.0),
    )
    gap = np.minimum(
        stop * float(base_gap_bps) / 10000.0 + float(alpha_through) * through,
        stop * float(max_gap_bps) / 10000.0,
    )
    exit_px = np.where(is_long, stop - gap, stop + gap)
    exit_px = np.where(hit & np.isfinite(exit_px) & (exit_px > 0.0), exit_px, np.nan)
    return hit, exit_px.astype(np.float32, copy=False)
