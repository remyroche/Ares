"""Exact hourly ATR-normalized meaningful-MFE label-grid primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


SCHEMA = "meaningful_mfe_label_grid_v1"


@dataclass(frozen=True)
class MeaningfulMFEGridSpec:
    horizon_hours: int
    upper_atr: float
    upper_return_floor: float = 0.015
    lower_atr: float = 1.0
    temperature: float = 0.35
    round_trip_cost: float = 0.01

    @property
    def name(self) -> str:
        upper = str(self.upper_atr).replace(".", "p")
        return f"h{self.horizon_hours}_u{upper}atr"


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def build_meaningful_mfe_grid_labels(
    *,
    entry_price: np.ndarray,
    future_high: np.ndarray,
    future_low: np.ndarray,
    future_close: np.ndarray,
    atr_fraction: np.ndarray,
    side_sign: np.ndarray,
    spec: MeaningfulMFEGridSpec,
) -> pd.DataFrame:
    """Build first-touch, soft, and supporting labels for one grid cell.

    Paths start at the executable decision bar.  Same-hour favorable/adverse
    conflicts are conservatively adverse because hourly OHLC does not reveal
    intrabar ordering.
    """

    if (
        spec.horizon_hours <= 0
        or spec.upper_atr <= 0.0
        or spec.lower_atr <= 0.0
        or spec.temperature <= 0.0
        or spec.round_trip_cost < 0.0
    ):
        raise ValueError("meaningful-MFE grid parameters are invalid")
    entry = np.asarray(entry_price, dtype=np.float64).reshape(-1)
    high = np.asarray(future_high, dtype=np.float64)
    low = np.asarray(future_low, dtype=np.float64)
    close = np.asarray(future_close, dtype=np.float64)
    atr = np.asarray(atr_fraction, dtype=np.float64).reshape(-1)
    side = np.asarray(side_sign, dtype=np.float64).reshape(-1)
    if high.ndim != 2 or high.shape != low.shape or high.shape != close.shape:
        raise ValueError("high, low, and close paths must have equal 2D shapes")
    rows = len(entry)
    if high.shape[0] != rows or len(atr) != rows or len(side) != rows:
        raise ValueError("meaningful-MFE grid inputs must align")
    horizon = int(spec.horizon_hours)
    if high.shape[1] < horizon:
        raise ValueError("path does not cover the requested hourly horizon")
    high = high[:, :horizon]
    low = low[:, :horizon]
    close = close[:, :horizon]

    valid = (
        np.isfinite(entry)
        & (entry > 0.0)
        & np.isfinite(atr)
        & (atr > 0.0)
        & np.isfinite(side)
        & (side != 0.0)
        & np.isfinite(high).all(axis=1)
        & np.isfinite(low).all(axis=1)
        & np.isfinite(close).all(axis=1)
    )
    favorable = np.where(
        side[:, None] > 0.0,
        high / entry[:, None] - 1.0,
        1.0 - low / entry[:, None],
    )
    adverse = np.where(
        side[:, None] > 0.0,
        1.0 - low / entry[:, None],
        high / entry[:, None] - 1.0,
    )
    directional_close = side[:, None] * (close / entry[:, None] - 1.0)
    upper_return = np.maximum(
        float(spec.upper_atr) * atr,
        float(spec.upper_return_floor),
    )
    lower_return = float(spec.lower_atr) * atr
    favorable_touch = favorable >= upper_return[:, None]
    adverse_touch = adverse >= lower_return[:, None]
    has_favorable = favorable_touch.any(axis=1)
    has_adverse = adverse_touch.any(axis=1)
    first_favorable = np.where(has_favorable, favorable_touch.argmax(axis=1), horizon)
    first_adverse = np.where(has_adverse, adverse_touch.argmax(axis=1), horizon)
    favorable_first = valid & has_favorable & (first_favorable < first_adverse)
    # Equality is a same-hour conflict and therefore adverse.
    adverse_first = valid & has_adverse & (first_adverse <= first_favorable)
    timeout = valid & ~favorable_first & ~adverse_first

    favorable_progress = np.clip(
        np.maximum(favorable, 0.0).max(axis=1) / upper_return,
        0.0,
        1.0,
    )
    adverse_progress = np.clip(
        np.maximum(adverse, 0.0).max(axis=1) / lower_return,
        0.0,
        1.0,
    )
    soft = _sigmoid(
        (favorable_progress - adverse_progress) / float(spec.temperature)
    )
    early_bonus = np.clip(1.0 - (first_favorable + 1.0) / horizon, 0.0, 1.0)
    soft = np.where(
        favorable_first,
        np.maximum(soft, 0.75 + 0.25 * early_bonus),
        soft,
    )
    soft = np.where(adverse_first, np.minimum(soft, 0.25 * favorable_progress), soft)
    soft = np.where(valid, np.clip(soft, 0.0, 1.0), np.nan)

    peak = np.maximum(favorable, 0.0).max(axis=1)
    peak_index = np.argmax(favorable, axis=1)
    eighty_level = 0.8 * peak
    reaches_eighty = favorable >= eighty_level[:, None]
    time_to_eighty = reaches_eighty.argmax(axis=1).astype(float) + 1.0
    time_to_eighty = np.where(valid & (peak > 0.0), time_to_eighty, np.nan)
    economic_eighty_touch = favorable >= (0.8 * upper_return)[:, None]
    reaches_economic_eighty = valid & economic_eighty_touch.any(axis=1)
    first_economic_eighty = economic_eighty_touch.argmax(axis=1).astype(float) + 1.0
    time_to_economic_eighty = np.where(
        valid,
        np.where(reaches_economic_eighty, first_economic_eighty, float(horizon)),
        np.nan,
    )
    economic_eighty_time_quality = np.where(
        reaches_economic_eighty,
        np.clip(1.0 - time_to_economic_eighty / float(horizon), 0.0, 1.0),
        0.0,
    )

    early_bars = min(3, horizon)
    early_favorable_atr = (
        np.maximum(directional_close[:, :early_bars], 0.0).max(axis=1) / atr
    )
    early_adverse_atr = (
        np.maximum(adverse[:, :early_bars], 0.0).max(axis=1) / atr
    )
    # Continuous supervision for "not adverse and not flat": 0.10 ATR of
    # directional close progress versus 0.25 ATR of adverse excursion.
    early_path_quality = _sigmoid(
        (early_favorable_atr - 0.10 - early_adverse_atr) / 0.20
    )
    early_clean_nonflat = (
        valid & (early_favorable_atr >= 0.10) & (early_adverse_atr < 0.25)
    )

    time = np.arange(1.0, horizon + 1.0)
    centered_time = time - time.mean()
    denominator = float(np.square(centered_time).sum())
    slope_return_per_hour = (
        (directional_close * centered_time[None, :]).sum(axis=1) / denominator
        if denominator > 0.0
        else np.zeros(rows)
    )
    slope_atr_per_hour = slope_return_per_hour / atr
    slope_atr_per_hour_clip_10 = np.clip(slope_atr_per_hour, -10.0, 10.0)
    economic_margin = upper_return - float(spec.round_trip_cost)

    outcome = np.full(rows, "invalid", dtype=object)
    outcome[timeout] = "timeout"
    outcome[adverse_first] = "adverse_first_or_conflict"
    outcome[favorable_first] = "favorable_first"
    return pd.DataFrame(
        {
            "grid_name": spec.name,
            "horizon_hours": horizon,
            "upper_atr": float(spec.upper_atr),
            "upper_return_floor": float(spec.upper_return_floor),
            "lower_atr": float(spec.lower_atr),
            "round_trip_cost": float(spec.round_trip_cost),
            "label_valid": valid,
            "soft_label": soft.astype(np.float32),
            "favorable_first": np.where(valid, favorable_first.astype(float), np.nan).astype(np.float32),
            "adverse_first": np.where(valid, adverse_first.astype(float), np.nan).astype(np.float32),
            "timeout": np.where(valid, timeout.astype(float), np.nan).astype(np.float32),
            "outcome": outcome,
            "upper_return": np.where(valid, upper_return, np.nan).astype(np.float32),
            "favorable_barrier_net_of_cost": np.where(valid, economic_margin, np.nan).astype(np.float32),
            "peak_mfe_atr": np.where(valid, peak / atr, np.nan).astype(np.float32),
            "time_to_80pct_mfe_hours": time_to_eighty.astype(np.float32),
            "reaches_80pct_economic_barrier": np.where(
                valid, reaches_economic_eighty.astype(float), np.nan
            ).astype(np.float32),
            "time_to_80pct_economic_barrier_hours": time_to_economic_eighty.astype(np.float32),
            "economic_barrier_time_quality": np.where(
                valid, economic_eighty_time_quality, np.nan
            ).astype(np.float32),
            "early_3bar_favorable_atr": np.where(valid, early_favorable_atr, np.nan).astype(np.float32),
            "early_3bar_adverse_atr": np.where(valid, early_adverse_atr, np.nan).astype(np.float32),
            "early_3bar_path_quality": np.where(valid, early_path_quality, np.nan).astype(np.float32),
            "early_3bar_clean_nonflat": np.where(valid, early_clean_nonflat.astype(float), np.nan).astype(np.float32),
            "future_close_slope_atr_per_hour": np.where(valid, slope_atr_per_hour, np.nan).astype(np.float32),
            "future_close_slope_atr_per_hour_clip_10": np.where(
                valid, slope_atr_per_hour_clip_10, np.nan
            ).astype(np.float32),
            "first_favorable_hour": np.where(favorable_first, first_favorable + 1.0, np.nan).astype(np.float32),
            "first_adverse_hour": np.where(adverse_first, first_adverse + 1.0, np.nan).astype(np.float32),
            "peak_hour": np.where(valid, peak_index + 1.0, np.nan).astype(np.float32),
        }
    )
