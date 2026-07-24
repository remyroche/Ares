"""Causal future-path targets and diagnostics for auxiliary timing/MFE models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

TARGET_SCHEMA = "path_auxiliary_targets_v6_supportive_future_paths_12h"
USEFUL_HORIZONS_HOURS: tuple[float, ...] = (2.0, 4.0, 8.0, 12.0)
SUPPORTIVE_HORIZONS_HOURS: tuple[float, ...] = (2.0, 4.0, 8.0, 12.0)
PEAK_MFE_ATR_CLIP = 10.0
MAE_BEFORE_MEANINGFUL_MFE_ATR_CLIP = PEAK_MFE_ATR_CLIP
# A fixed 10 ATR/hour cap prevents very small bar sizes from producing an
# unbounded rate target while retaining the established path-MFE tail scale.
FUTURE_SLOPE_ATR_PER_HOUR_CLIP = PEAK_MFE_ATR_CLIP
MIN_USABLE_MFE_ATR = 1.5
MIN_USABLE_MFE_RETURN = 0.015

# Keep the five head names and materialized targets in one import-safe location.
# ``path_auxiliary_lgbm`` owns model fitting; this module owns label semantics.
TARGET_COLUMNS: dict[str, str] = {
    "time_to_first_meaningful_mfe": "__log1p_time_to_first_meaningful_mfe_hours_12h__",
    "peak_mfe_12h_atr": "__log1p_peak_mfe_atr_12h__",
    "mae_before_meaningful_mfe_atr": "__log1p_mae_before_meaningful_mfe_atr_12h__",
    "bars_before_price_stops_decreasing": "__log1p_bars_before_price_stops_decreasing_12h__",
    "future_slope_atr_per_hour": "__log1p_future_slope_atr_per_hour_12h__",
}

# These columns are labels/support diagnostics, never inference features. Keep
# tuple order stable: runners use it for manifests, read-only loading, and
# deterministic sample-weight/diagnostic selection.
_LEGACY_SUPPORTIVE_LABEL_COLUMNS: dict[str, tuple[str, ...]] = {
    "peak_mfe_12h_atr": (
        "__peak_mfe_ge_0_5atr_12h__",
        "__peak_mfe_ge_1atr_12h__",
        "__peak_mfe_ge_1_5atr_12h__",
        "__peak_mfe_ge_2atr_12h__",
        "__peak_mfe_ge_3atr_12h__",
        "__peak_mfe_ge_4atr_12h__",
        "__peak_mfe_bars_above_50pct_12h__",
        "__peak_mfe_bars_above_80pct_12h__",
        "__peak_mfe_fraction_above_50pct_12h__",
        "__peak_mfe_fraction_above_80pct_12h__",
        "__mfe_integral_atr_hours_12h__",
    ),
    "time_to_first_meaningful_mfe": (
        "__mfe_ratio_to_peak_at_2h_12h__",
        "__mfe_ratio_to_peak_at_4h_12h__",
        "__mfe_ratio_to_peak_at_8h_12h__",
        "__peak_mfe_within_1h_12h__",
        "__peak_mfe_within_2h_12h__",
        "__peak_mfe_within_4h_12h__",
        "__peak_mfe_within_8h_12h__",
    ),
    "mae_before_meaningful_mfe_atr": (
        "__pre_mfe_mae_event_12h__",
        "__pre_mfe_mae_ge_0_25atr_12h__",
        "__pre_mfe_mae_ge_0_5atr_12h__",
        "__pre_mfe_mae_ge_0_75atr_12h__",
        "__pre_mfe_mae_ge_1atr_12h__",
        "__pre_mfe_mae_ge_1_5atr_12h__",
        "__pre_mfe_mae_0_25atr_before_meaningful_mfe_12h__",
        "__pre_mfe_mae_0_5atr_before_meaningful_mfe_12h__",
        "__pre_mfe_mae_0_75atr_before_meaningful_mfe_12h__",
        "__pre_mfe_mae_1atr_before_meaningful_mfe_12h__",
        "__pre_mfe_mae_1_5atr_before_meaningful_mfe_12h__",
        "__meaningful_mfe_before_mae_0_25atr_12h__",
        "__meaningful_mfe_before_mae_0_5atr_12h__",
        "__meaningful_mfe_before_mae_0_75atr_12h__",
        "__meaningful_mfe_before_mae_1atr_12h__",
        "__meaningful_mfe_before_mae_1_5atr_12h__",
        "__pre_mfe_underwater_bars_12h__",
        "__pre_mfe_underwater_fraction_12h__",
    ),
    "bars_before_price_stops_decreasing": (
        "__adverse_trough_atr_12h__",
        "__adverse_trough_bar_12h__",
        "__adverse_trough_recovery_fraction_12h__",
        "__adverse_trough_recovered_50pct_12h__",
        "__adverse_trough_recovered_80pct_12h__",
        "__adverse_trough_recovered_100pct_12h__",
        "__adverse_trough_recovery_50pct_confirmed_2bars_12h__",
        "__adverse_trough_recovery_100pct_confirmed_2bars_12h__",
        "__bars_from_adverse_trough_to_full_recovery_12h__",
        "__time_from_adverse_trough_to_full_recovery_hours_12h__",
    ),
    "future_slope_atr_per_hour": (
        "__future_slope_atr_per_hour_2h__",
        "__future_slope_atr_per_hour_4h__",
        "__future_slope_atr_per_hour_8h__",
        "__future_slope_atr_per_hour_12h__",
        "__time_to_peak_mfe_hours_12h__",
        "__time_to_50pct_peak_mfe_hours_12h__",
        "__time_to_80pct_peak_mfe_hours_12h__",
        "__bars_to_peak_mfe_12h__",
        "__bars_to_50pct_peak_mfe_12h__",
        "__bars_to_80pct_peak_mfe_12h__",
        "__mfe_mae_path_efficiency_12h__",
        "__mfe_integral_path_efficiency_12h__",
        "__mfe_timing_path_efficiency_12h__",
        "__mfe_persistence_path_efficiency_12h__",
    ),
}

# Stable runner-facing aliases. The old horizon-suffixed names are still
# materialized below, but consumers that load support targets should use these
# exact names so sample weights, manifests, and diagnostics share one contract.
SUPPORTIVE_LABEL_COLUMNS: dict[str, tuple[str, ...]] = {
    "peak_mfe_12h_atr": (
        "__mfe_ge_0_5atr__",
        "__mfe_ge_1_0atr__",
        "__mfe_ge_1_5atr__",
        "__mfe_ge_2_0atr__",
        "__mfe_ge_3_0atr__",
        "__mfe_ge_4_0atr__",
        "__peak_mfe_atr_clip_6__",
        "__peak_mfe_atr_clip_8__",
        "__bars_above_50pct_peak__",
        "__bars_above_80pct_peak__",
        "__fraction_bars_above_50pct_peak__",
        "__fraction_bars_above_80pct_peak__",
        "__log1p_peak_mfe_12h_atr__",
        "__favorable_path_integral_atr__",
    ),
    "time_to_first_meaningful_mfe": (
        "__mfe_2h_over_mfe_12h__",
        "__mfe_4h_over_mfe_12h__",
        "__mfe_8h_over_mfe_12h__",
        "__peak_within_1h__",
        "__peak_within_2h__",
        "__peak_within_4h__",
        "__peak_within_8h__",
    ),
    "mae_before_meaningful_mfe_atr": (
        "__reaches_1_5atr_within_12h__",
        "__mae_before_1_5atr_mfe__",
        "__mae_until_horizon_if_no_1_5atr__",
        "__pre_1_5_mfe_mae_ge_0_25atr__",
        "__pre_1_5_mfe_mae_ge_0_50atr__",
        "__pre_1_5_mfe_mae_ge_0_75atr__",
        "__pre_1_5_mfe_mae_ge_1_00atr__",
        "__pre_1_5_mfe_mae_ge_1_50atr__",
        "__hits_minus_1_0atr_before_plus_1_5atr__",
        "__hits_minus_0_5atr_before_plus_1_5atr__",
        "__bars_below_entry_before_1_5atr__",
        "__fraction_bars_below_entry_before_1_5atr__",
    ),
    "bars_before_price_stops_decreasing": (
        "__bars_to_25pct_adverse_recovery__",
        "__bars_to_50pct_adverse_recovery__",
        "__adverse_trough_within_60m__",
        "__adverse_trough_within_120m__",
        "__bars_to_confirmed_adverse_trough__",
        "__mfe_before_60m_atr__",
        "__reaches_1_5atr_before_trough_confirmation__",
        "__trough_before_1_5atr_mfe__",
    ),
    "future_slope_atr_per_hour": (
        "__future_slope_atr_per_hour_2h__",
        "__future_slope_atr_per_hour_4h__",
        "__future_slope_atr_per_hour_8h__",
        "__future_slope_atr_per_hour_12h__",
        "__bars_to_1atr__",
        "__bars_to_1_5atr__",
        "__bars_to_2atr__",
        "__bars_to_80pct_peak__",
        "__mfe_2h_over_mfe_12h__",
        "__mfe_4h_over_mfe_12h__",
        "__mfe_8h_over_mfe_12h__",
        "__path_efficiency_12h__",
        "__path_efficiency_to_1_5atr__",
        "__path_efficiency_to_2atr__",
        "__path_efficiency_to_80pct_peak__",
        "__path_efficiency_to_90pct_peak__",
        "__path_efficiency_to_first_meaningful_mfe__",
    ),
}
ALL_SUPPORTIVE_LABEL_COLUMNS: tuple[str, ...] = tuple(
    column
    for target_name in TARGET_COLUMNS
    for column in SUPPORTIVE_LABEL_COLUMNS[target_name]
)
MODEL_FAMILY_LABEL_COLUMNS: dict[str, tuple[str, ...]] = {
    "peak_mfe_12h_atr": ("__meaningful_mfe_reached_12h__",),
    "time_to_first_meaningful_mfe": ("__meaningful_mfe_reached_12h__",),
    "mae_before_meaningful_mfe_atr": ("__meaningful_mfe_reached_12h__",),
    "bars_before_price_stops_decreasing": ("__bars_to_confirmed_adverse_trough__",),
    "future_slope_atr_per_hour": (),
}
ALL_MODEL_FAMILY_LABEL_COLUMNS: tuple[str, ...] = tuple(
    dict.fromkeys(
        column
        for target_name in TARGET_COLUMNS
        for column in MODEL_FAMILY_LABEL_COLUMNS[target_name]
    )
)


@dataclass(frozen=True)
class PathAuxiliaryTargets:
    peak_mfe_return_12h: np.ndarray
    peak_mfe_atr_12h: np.ndarray
    time_to_first_meaningful_mfe_hours_12h: np.ndarray
    mae_before_meaningful_mfe_atr_12h: np.ndarray
    bars_before_price_stops_decreasing_12h: np.ndarray
    future_slope_atr_per_hour_12h: np.ndarray
    log1p_peak_mfe_atr_12h: np.ndarray
    log1p_time_to_first_meaningful_mfe_hours_12h: np.ndarray
    log1p_mae_before_meaningful_mfe_atr_12h: np.ndarray
    log1p_bars_before_price_stops_decreasing_12h: np.ndarray
    log1p_future_slope_atr_per_hour_12h: np.ndarray
    supportive_columns: dict[str, np.ndarray]
    valid: np.ndarray
    timing_valid: np.ndarray
    meaningful_mfe_reached: np.ndarray

    def as_columns(self) -> dict[str, np.ndarray]:
        return {
            "__peak_mfe_return_12h__": self.peak_mfe_return_12h,
            "__peak_mfe_atr_12h__": self.peak_mfe_atr_12h,
            "__time_to_first_meaningful_mfe_hours_12h__": self.time_to_first_meaningful_mfe_hours_12h,
            "__mae_before_meaningful_mfe_atr_12h__": self.mae_before_meaningful_mfe_atr_12h,
            "__bars_before_price_stops_decreasing_12h__": self.bars_before_price_stops_decreasing_12h,
            "__future_slope_atr_per_hour_12h__": self.future_slope_atr_per_hour_12h,
            "__log1p_peak_mfe_atr_12h__": self.log1p_peak_mfe_atr_12h,
            "__log1p_time_to_first_meaningful_mfe_hours_12h__": self.log1p_time_to_first_meaningful_mfe_hours_12h,
            "__log1p_mae_before_meaningful_mfe_atr_12h__": self.log1p_mae_before_meaningful_mfe_atr_12h,
            "__log1p_bars_before_price_stops_decreasing_12h__": self.log1p_bars_before_price_stops_decreasing_12h,
            "__log1p_future_slope_atr_per_hour_12h__": self.log1p_future_slope_atr_per_hour_12h,
            **self.supportive_columns,
            "__path_auxiliary_target_valid__": self.valid.astype(np.int8),
            "__time_to_first_meaningful_mfe_target_valid__": self.timing_valid.astype(
                np.int8
            ),
            "__meaningful_mfe_reached_12h__": self.meaningful_mfe_reached.astype(
                np.int8
            ),
        }


def build_path_auxiliary_targets(
    *,
    entry_price: np.ndarray,
    future_high: np.ndarray,
    future_low: np.ndarray,
    atr_fraction: np.ndarray,
    side_sign: np.ndarray,
    bar_minutes: int = 60,
    horizon_hours: int = 12,
) -> PathAuxiliaryTargets:
    """Build raw 12-hour targets from the executable decision-time bar onward.

    ``future_high`` and ``future_low`` must start with the bar whose open is the
    executable entry at ``decision_ts = signal_ts + timeframe``. No exit policy
    is applied: peak MFE is measured over the complete requested horizon so it
    remains a stable auxiliary target. Favorable excursions below
    ``max(1.5 * ATR, 1.5% of entry)`` are treated as zero usable MFE. The timing
    target is first passage of that floor, with unreached valid rows censored at
    12 hours.

    The three path-shape targets do not include costs. ``mae_before...`` uses
    adverse high/low excursion from entry through the meaningful-hit bar (or
    the full path when unreached). ``bars_before_price_stops_decreasing`` is
    deliberately side-normalized: it is the one-based path-bar number of the
    long minimum low / short maximum high strictly before that hit, with entry
    represented as bar zero. ``future_slope...`` is 80% of capped eventual MFE
    divided by the time to its first 80%-of-peak attainment.

    Historical MFE and timing columns retain their ceiling-to-bar behavior for
    compatibility. The new shape targets use only complete bars whose end lies
    inside the declared horizon, so they never inspect a partial bar beyond it.
    """

    entry = np.asarray(entry_price, dtype=np.float64).reshape(-1)
    high = np.asarray(future_high, dtype=np.float64)
    low = np.asarray(future_low, dtype=np.float64)
    atr = np.asarray(atr_fraction, dtype=np.float64).reshape(-1)
    side = np.asarray(side_sign, dtype=np.float64).reshape(-1)
    if high.ndim != 2 or low.ndim != 2 or high.shape != low.shape:
        raise ValueError("future_high/future_low must be equal two-dimensional arrays")
    n = len(entry)
    if high.shape[0] != n or len(atr) != n or len(side) != n:
        raise ValueError("path target inputs must have equal row counts")
    if int(bar_minutes) <= 0 or int(horizon_hours) <= 0:
        raise ValueError("bar_minutes and horizon_hours must be positive")
    horizon_bars = int(np.ceil(60.0 * float(horizon_hours) / float(bar_minutes)))
    if high.shape[1] < horizon_bars:
        raise ValueError("future path does not contain the complete requested horizon")
    high = high[:, :horizon_bars]
    low = low[:, :horizon_bars]
    shape_horizon_bars = int(np.floor(60.0 * float(horizon_hours) / float(bar_minutes)))
    if shape_horizon_bars <= 0:
        raise ValueError("bar_minutes exceeds the requested path-shape horizon")

    valid_row = (
        np.isfinite(entry)
        & (entry > 0.0)
        & np.isfinite(atr)
        & (atr > 0.0)
        & np.isfinite(side)
        & (side != 0.0)
    )
    long_fav = high / entry[:, None] - 1.0
    short_fav = 1.0 - low / entry[:, None]
    favorable = np.where(side[:, None] > 0.0, long_fav, short_fav)
    long_adverse = 1.0 - low / entry[:, None]
    short_adverse = high / entry[:, None] - 1.0
    adverse = np.where(side[:, None] > 0.0, long_adverse, short_adverse)
    favorable = np.where(np.isfinite(favorable), favorable, -np.inf)
    has_path = np.all(np.isfinite(high) & np.isfinite(low), axis=1)
    valid = valid_row & has_path

    peak = np.max(favorable, axis=1)
    peak = np.where(valid, np.maximum(peak, 0.0), np.nan)
    usable_floor = np.maximum(
        MIN_USABLE_MFE_ATR * atr,
        MIN_USABLE_MFE_RETURN,
    )
    usable_peak = valid & (peak >= usable_floor)
    # Economically insignificant favorable noise is a zero-magnitude target,
    # not a peak whose timing the model should learn.
    peak = np.where(valid, np.where(usable_peak, peak, 0.0), np.nan)
    meaningful_hits = favorable >= usable_floor[:, None]
    meaningful_reached = valid & np.any(meaningful_hits, axis=1)
    first_meaningful_index = np.argmax(meaningful_hits, axis=1)
    reached_hours = (
        (first_meaningful_index.astype(np.float64) + 1.0) * float(bar_minutes) / 60.0
    )
    # A bar size that does not divide the horizon can make the final partial
    # bar end slightly beyond it. The timing target remains right-censored at
    # the declared horizon in every supported bar layout.
    reached_hours = np.minimum(reached_hours, float(horizon_hours))
    # Unreached, otherwise valid paths are censored at the declared horizon.
    # They remain timing-model examples rather than disappearing from training.
    time_hours = np.where(
        valid,
        np.where(meaningful_reached, reached_hours, float(horizon_hours)),
        np.nan,
    )
    timing_valid = valid
    peak_atr = np.divide(
        peak,
        atr,
        out=np.full(n, np.nan, dtype=np.float64),
        where=valid & (atr > 0.0),
    )
    # Near-zero stale ATR denominators can otherwise create values above 1e20
    # and dominate selection and regression loss. Ten ATR already represents
    # an extreme path and keeps the target sample-independent.
    peak_atr = np.clip(peak_atr, 0.0, PEAK_MFE_ATR_CLIP)

    shape_favorable = favorable[:, :shape_horizon_bars]
    shape_adverse = adverse[:, :shape_horizon_bars]
    shape_meaningful_hits = meaningful_hits[:, :shape_horizon_bars]
    shape_meaningful_reached = valid & np.any(shape_meaningful_hits, axis=1)
    shape_first_meaningful_index = np.argmax(shape_meaningful_hits, axis=1)
    path_indices = np.arange(shape_horizon_bars, dtype=np.int64)[None, :]
    # Include the hit bar for adverse depth, because the path can dip below
    # entry before reaching the threshold within that OHLC bar.
    mae_stop = np.where(
        shape_meaningful_reached,
        shape_first_meaningful_index + 1,
        shape_horizon_bars,
    )
    mae_before_hit_return = np.maximum(
        np.max(
            np.where(path_indices < mae_stop[:, None], shape_adverse, -np.inf),
            axis=1,
        ),
        0.0,
    )
    mae_before_hit_atr = np.divide(
        mae_before_hit_return,
        atr,
        out=np.full(n, np.nan, dtype=np.float64),
        where=valid,
    )
    mae_before_hit_atr = np.where(valid, mae_before_hit_atr, np.nan)
    mae_before_hit_atr = np.clip(
        mae_before_hit_atr,
        0.0,
        MAE_BEFORE_MEANINGFUL_MFE_ATR_CLIP,
    )

    # The semantic name is side-normalized: a long path stops decreasing at
    # its lowest low, while a short path stops decreasing at its highest high.
    # Entry is an explicit zero-bar candidate, and the meaningful-hit bar is
    # excluded so this is the adverse turning point *preceding* that hit.
    turning_stop = np.where(
        shape_meaningful_reached,
        shape_first_meaningful_index,
        shape_horizon_bars,
    )
    turning_adverse = np.where(
        path_indices < turning_stop[:, None], shape_adverse, -np.inf
    )
    turning_with_entry = np.concatenate(
        [np.zeros((n, 1), dtype=np.float64), turning_adverse], axis=1
    )
    bars_before_turning = np.argmax(turning_with_entry, axis=1).astype(np.float64)
    bars_before_turning = np.where(valid, bars_before_turning, np.nan)
    bars_before_turning = np.clip(
        bars_before_turning,
        0.0,
        float(shape_horizon_bars),
    )

    # Use the same 10-ATR peak cap as the established MFE target before
    # locating the 80% threshold, so a single extreme wick cannot dominate
    # either target's scale.
    raw_peak_return = np.where(
        valid,
        np.maximum(np.max(shape_favorable, axis=1), 0.0),
        np.nan,
    )
    raw_peak_atr = np.divide(
        raw_peak_return,
        atr,
        out=np.full(n, np.nan, dtype=np.float64),
        where=valid,
    )
    slope_peak_atr = np.clip(raw_peak_atr, 0.0, PEAK_MFE_ATR_CLIP)
    slope_peak_return = slope_peak_atr * atr
    first_80pct_index = np.argmax(
        shape_favorable >= (0.8 * slope_peak_return)[:, None], axis=1
    )
    bar_hours = float(bar_minutes) / 60.0
    hours_to_first_80pct = np.minimum(
        (first_80pct_index.astype(np.float64) + 1.0) * bar_hours,
        float(horizon_hours),
    )
    future_slope = np.divide(
        0.8 * slope_peak_atr,
        np.maximum(hours_to_first_80pct, bar_hours),
        out=np.zeros(n, dtype=np.float64),
        where=valid & (slope_peak_atr > 0.0),
    )
    future_slope = np.where(valid, future_slope, np.nan)
    future_slope = np.clip(future_slope, 0.0, FUTURE_SLOPE_ATR_PER_HOUR_CLIP)

    # Support labels use only complete post-decision bars. Invalid paths are
    # NaN; the primary timing target and trough-recovery time are right-censored.
    shape_fav_positive = np.maximum(shape_favorable, 0.0)
    shape_adv_positive = np.maximum(shape_adverse, 0.0)
    shape_peak_return = np.where(valid, np.max(shape_fav_positive, axis=1), np.nan)
    shape_peak_atr = np.divide(
        shape_peak_return, atr, out=np.full(n, np.nan), where=valid
    )
    bar_hours = float(bar_minutes) / 60.0
    support: dict[str, np.ndarray] = {}

    for threshold, label in (
        (0.5, "0_5"),
        (1.0, "1"),
        (1.5, "1_5"),
        (2.0, "2"),
        (3.0, "3"),
        (4.0, "4"),
    ):
        support[f"__peak_mfe_ge_{label}atr_12h__"] = np.where(
            valid, (shape_peak_atr >= threshold).astype(float), np.nan
        ).astype(np.float32)

    positive_shape_peak = shape_peak_return > 0.0
    for threshold, label in ((0.5, "50"), (0.8, "80")):
        above_peak = (
            shape_fav_positive >= threshold * shape_peak_return[:, None]
        ) & positive_shape_peak[:, None]
        bars = np.sum(above_peak, axis=1)
        support[f"__peak_mfe_bars_above_{label}pct_12h__"] = np.where(
            valid, bars, np.nan
        ).astype(np.float32)
        support[f"__peak_mfe_fraction_above_{label}pct_12h__"] = np.where(
            valid, bars / float(shape_horizon_bars), np.nan
        ).astype(np.float32)
    # Cap each bar before integration so stale, tiny ATR values remain finite.
    instantaneous_mfe_atr = np.divide(
        shape_fav_positive,
        atr[:, None],
        out=np.zeros_like(shape_fav_positive),
        where=valid[:, None] & (atr[:, None] > 0.0),
    )
    support["__mfe_integral_atr_hours_12h__"] = np.where(
        valid,
        np.sum(np.minimum(instantaneous_mfe_atr, PEAK_MFE_ATR_CLIP), axis=1)
        * bar_hours,
        np.nan,
    ).astype(np.float32)

    peak_index = np.argmax(shape_fav_positive, axis=1)
    for hours in (2.0, 4.0, 8.0):
        prefix_bars = int(np.floor(hours / bar_hours))
        if hours > float(horizon_hours) or prefix_bars <= 0:
            support[f"__mfe_ratio_to_peak_at_{int(hours)}h_12h__"] = np.full(
                n, np.nan, dtype=np.float32
            )
            continue
        prefix_peak = np.max(shape_fav_positive[:, :prefix_bars], axis=1)
        ratio = np.divide(
            prefix_peak, shape_peak_return, out=np.zeros(n), where=positive_shape_peak
        )
        support[f"__mfe_ratio_to_peak_at_{int(hours)}h_12h__"] = np.where(
            valid, np.clip(ratio, 0.0, 1.0), np.nan
        ).astype(np.float32)
    for hours in (1.0, 2.0, 4.0, 8.0):
        prefix_bars = int(np.floor(hours / bar_hours))
        available = hours <= float(horizon_hours) and prefix_bars > 0
        support[f"__peak_mfe_within_{int(hours)}h_12h__"] = (
            np.where(
                valid,
                (positive_shape_peak & (peak_index < prefix_bars)).astype(float),
                np.nan,
            ).astype(np.float32)
            if available
            else np.full(n, np.nan, dtype=np.float32)
        )

    pre_mask = path_indices < mae_stop[:, None]
    pre_adverse = np.where(pre_mask, shape_adv_positive, -np.inf)
    pre_adverse_peak = np.maximum(np.max(pre_adverse, axis=1), 0.0)
    support["__pre_mfe_mae_event_12h__"] = np.where(
        valid, (pre_adverse_peak > 0.0).astype(float), np.nan
    ).astype(np.float32)
    for threshold, label in (
        (0.25, "0_25"),
        (0.5, "0_5"),
        (0.75, "0_75"),
        (1.0, "1"),
        (1.5, "1_5"),
    ):
        hit = pre_mask & (shape_adv_positive >= threshold * atr[:, None])
        reached = np.any(hit, axis=1)
        first_index = np.argmax(hit, axis=1)
        support[f"__pre_mfe_mae_ge_{label}atr_12h__"] = np.where(
            valid, reached.astype(float), np.nan
        ).astype(np.float32)
        support[f"__pre_mfe_mae_{label}atr_before_meaningful_mfe_12h__"] = np.where(
            valid, reached.astype(float), np.nan
        ).astype(np.float32)
        mfe_first = (
            shape_meaningful_reached
            & reached
            & (shape_first_meaningful_index < first_index)
        )
        support[f"__meaningful_mfe_before_mae_{label}atr_12h__"] = np.where(
            valid, mfe_first.astype(float), np.nan
        ).astype(np.float32)
    underwater_bars = np.sum(pre_mask & (shape_adv_positive > 0.0), axis=1)
    support["__pre_mfe_underwater_bars_12h__"] = np.where(
        valid, underwater_bars, np.nan
    ).astype(np.float32)
    support["__pre_mfe_underwater_fraction_12h__"] = np.where(
        valid, underwater_bars / np.maximum(mae_stop, 1), np.nan
    ).astype(np.float32)

    trough_return = np.where(valid, np.max(shape_adv_positive, axis=1), np.nan)
    trough_index = np.argmax(shape_adv_positive, axis=1)
    trough_valid = valid & (trough_return > 0.0)
    trough_atr = np.divide(trough_return, atr, out=np.full(n, np.nan), where=valid)
    support["__adverse_trough_atr_12h__"] = np.where(
        valid, np.clip(trough_atr, 0.0, PEAK_MFE_ATR_CLIP), np.nan
    ).astype(np.float32)
    support["__adverse_trough_bar_12h__"] = np.where(
        valid, trough_index + 1.0, np.nan
    ).astype(np.float32)
    post_trough = path_indices > trough_index[:, None]
    post_favorable = np.max(np.where(post_trough, shape_favorable, -np.inf), axis=1)
    post_favorable = np.where(
        trough_index < shape_horizon_bars - 1, post_favorable, -trough_return
    )
    recovery = np.divide(
        post_favorable + trough_return,
        trough_return,
        out=np.zeros(n),
        where=trough_return > 0.0,
    )
    recovery = np.clip(recovery, 0.0, 1.0)
    support["__adverse_trough_recovery_fraction_12h__"] = np.where(
        trough_valid, recovery, np.nan
    ).astype(np.float32)
    per_bar_recovery = np.divide(
        shape_favorable + trough_return[:, None],
        trough_return[:, None],
        out=np.zeros_like(shape_favorable),
        where=trough_return[:, None] > 0.0,
    )
    per_bar_recovery = np.where(post_trough, per_bar_recovery, -np.inf)
    for threshold, label in ((0.5, "50"), (0.8, "80"), (1.0, "100")):
        recovered = per_bar_recovery >= threshold
        support[f"__adverse_trough_recovered_{label}pct_12h__"] = np.where(
            trough_valid, np.any(recovered, axis=1).astype(float), np.nan
        ).astype(np.float32)
        if label in {"50", "100"}:
            support[f"__adverse_trough_recovery_{label}pct_confirmed_2bars_12h__"] = (
                np.where(
                    trough_valid, (np.sum(recovered, axis=1) >= 2).astype(float), np.nan
                ).astype(np.float32)
            )
    full_recovery = per_bar_recovery >= 1.0
    first_recovery_index = np.argmax(full_recovery, axis=1)
    bars_to_recovery = np.where(
        np.any(full_recovery, axis=1),
        first_recovery_index - trough_index,
        shape_horizon_bars,
    )
    support["__bars_from_adverse_trough_to_full_recovery_12h__"] = np.where(
        trough_valid, bars_to_recovery, np.nan
    ).astype(np.float32)
    support["__time_from_adverse_trough_to_full_recovery_hours_12h__"] = np.where(
        trough_valid,
        np.minimum(bars_to_recovery * bar_hours, float(horizon_hours)),
        np.nan,
    ).astype(np.float32)

    def _slope_for_horizon(hours: float) -> np.ndarray:
        prefix_bars = int(np.floor(hours / bar_hours))
        if hours > float(horizon_hours) or prefix_bars <= 0:
            return np.full(n, np.nan)
        prefix = shape_fav_positive[:, :prefix_bars]
        prefix_peak = np.max(prefix, axis=1)
        prefix_atr = np.divide(prefix_peak, atr, out=np.zeros(n), where=valid)
        first_80 = np.argmax(prefix >= (0.8 * prefix_peak)[:, None], axis=1)
        slope = np.divide(
            0.8 * np.clip(prefix_atr, 0.0, PEAK_MFE_ATR_CLIP),
            np.maximum((first_80 + 1.0) * bar_hours, bar_hours),
            out=np.zeros(n),
            where=prefix_peak > 0.0,
        )
        return np.where(
            valid, np.clip(slope, 0.0, FUTURE_SLOPE_ATR_PER_HOUR_CLIP), np.nan
        )

    for hours in (2.0, 4.0, 8.0):
        support[f"__future_slope_atr_per_hour_{int(hours)}h__"] = _slope_for_horizon(
            hours
        ).astype(np.float32)
    for fraction, label in ((1.0, "peak"), (0.5, "50pct_peak"), (0.8, "80pct_peak")):
        first_index = np.argmax(
            shape_fav_positive >= (fraction * shape_peak_return)[:, None], axis=1
        )
        support[f"__time_to_{label}_mfe_hours_12h__"] = np.where(
            valid & positive_shape_peak, (first_index + 1.0) * bar_hours, np.nan
        ).astype(np.float32)
        support[f"__bars_to_{label}_mfe_12h__"] = np.where(
            valid & positive_shape_peak, first_index + 1.0, np.nan
        ).astype(np.float32)
    total_excursion = shape_peak_return + trough_return
    support["__mfe_mae_path_efficiency_12h__"] = np.where(
        valid,
        np.divide(
            shape_peak_return,
            total_excursion,
            out=np.zeros(n),
            where=total_excursion > 0.0,
        ),
        np.nan,
    ).astype(np.float32)
    integral_efficiency = np.divide(
        np.sum(shape_fav_positive, axis=1),
        shape_peak_return * shape_horizon_bars,
        out=np.zeros(n),
        where=positive_shape_peak,
    )
    support["__mfe_integral_path_efficiency_12h__"] = np.where(
        valid & positive_shape_peak, integral_efficiency, np.nan
    ).astype(np.float32)
    support["__mfe_timing_path_efficiency_12h__"] = np.where(
        valid & positive_shape_peak,
        1.0 - (peak_index + 1.0) / shape_horizon_bars,
        np.nan,
    ).astype(np.float32)
    support["__mfe_persistence_path_efficiency_12h__"] = support[
        "__peak_mfe_fraction_above_80pct_12h__"
    ].copy()

    aliases = {
        "__mfe_ge_0_5atr__": "__peak_mfe_ge_0_5atr_12h__",
        "__mfe_ge_1atr__": "__peak_mfe_ge_1atr_12h__",
        "__mfe_ge_1_5atr__": "__peak_mfe_ge_1_5atr_12h__",
        "__mfe_ge_2atr__": "__peak_mfe_ge_2atr_12h__",
        "__mfe_ge_3atr__": "__peak_mfe_ge_3atr_12h__",
        "__mfe_ge_4atr__": "__peak_mfe_ge_4atr_12h__",
        "__bars_above_50pct_peak__": "__peak_mfe_bars_above_50pct_12h__",
        "__bars_above_80pct_peak__": "__peak_mfe_bars_above_80pct_12h__",
        "__fraction_bars_above_50pct_peak__": "__peak_mfe_fraction_above_50pct_12h__",
        "__fraction_bars_above_80pct_peak__": "__peak_mfe_fraction_above_80pct_12h__",
        "__mfe_integral_atr_hours__": "__mfe_integral_atr_hours_12h__",
        "__mfe_ratio_at_2h_to_peak__": "__mfe_ratio_to_peak_at_2h_12h__",
        "__mfe_ratio_at_4h_to_peak__": "__mfe_ratio_to_peak_at_4h_12h__",
        "__mfe_ratio_at_8h_to_peak__": "__mfe_ratio_to_peak_at_8h_12h__",
        "__peak_within_1h__": "__peak_mfe_within_1h_12h__",
        "__peak_within_2h__": "__peak_mfe_within_2h_12h__",
        "__peak_within_4h__": "__peak_mfe_within_4h_12h__",
        "__peak_within_8h__": "__peak_mfe_within_8h_12h__",
        "__pre_mfe_mae_event__": "__pre_mfe_mae_event_12h__",
        "__pre_mfe_underwater_bars__": "__pre_mfe_underwater_bars_12h__",
        "__pre_mfe_underwater_fraction__": "__pre_mfe_underwater_fraction_12h__",
        "__adverse_trough_atr__": "__adverse_trough_atr_12h__",
        "__adverse_trough_bar__": "__adverse_trough_bar_12h__",
        "__adverse_trough_recovery_fraction__": "__adverse_trough_recovery_fraction_12h__",
        "__adverse_trough_recovered_50pct__": "__adverse_trough_recovered_50pct_12h__",
        "__adverse_trough_recovered_80pct__": "__adverse_trough_recovered_80pct_12h__",
        "__adverse_trough_recovered_100pct__": "__adverse_trough_recovered_100pct_12h__",
        "__adverse_trough_recovery_50pct_confirmed_2bars__": "__adverse_trough_recovery_50pct_confirmed_2bars_12h__",
        "__adverse_trough_recovery_100pct_confirmed_2bars__": "__adverse_trough_recovery_100pct_confirmed_2bars_12h__",
        "__bars_from_adverse_trough_to_full_recovery__": "__bars_from_adverse_trough_to_full_recovery_12h__",
        "__time_from_adverse_trough_to_full_recovery_hours__": "__time_from_adverse_trough_to_full_recovery_hours_12h__",
        "__future_slope_2h_atr_per_hour__": "__future_slope_atr_per_hour_2h__",
        "__future_slope_4h_atr_per_hour__": "__future_slope_atr_per_hour_4h__",
        "__future_slope_8h_atr_per_hour__": "__future_slope_atr_per_hour_8h__",
        "__time_to_peak_mfe_hours__": "__time_to_peak_mfe_hours_12h__",
        "__time_to_50pct_peak_mfe_hours__": "__time_to_50pct_peak_mfe_hours_12h__",
        "__time_to_80pct_peak_mfe_hours__": "__time_to_80pct_peak_mfe_hours_12h__",
        "__bars_to_peak_mfe__": "__bars_to_peak_mfe_12h__",
        "__bars_to_50pct_peak_mfe__": "__bars_to_50pct_peak_mfe_12h__",
        "__bars_to_80pct_peak_mfe__": "__bars_to_80pct_peak_mfe_12h__",
        "__mfe_mae_path_efficiency__": "__mfe_mae_path_efficiency_12h__",
        "__mfe_integral_path_efficiency__": "__mfe_integral_path_efficiency_12h__",
        "__mfe_timing_path_efficiency__": "__mfe_timing_path_efficiency_12h__",
        "__mfe_persistence_path_efficiency__": "__mfe_persistence_path_efficiency_12h__",
    }
    for threshold in ("0_25", "0_5", "0_75", "1", "1_5"):
        aliases[f"__pre_mfe_mae_ge_{threshold}atr__"] = (
            f"__pre_mfe_mae_ge_{threshold}atr_12h__"
        )
        aliases[f"__pre_mfe_mae_{threshold}atr_before_meaningful_mfe__"] = (
            f"__pre_mfe_mae_{threshold}atr_before_meaningful_mfe_12h__"
        )
        aliases[f"__meaningful_mfe_before_mae_{threshold}atr__"] = (
            f"__meaningful_mfe_before_mae_{threshold}atr_12h__"
        )
    support.update({alias: support[source].copy() for alias, source in aliases.items()})
    support["__future_slope_12h_atr_per_hour__"] = future_slope.astype(np.float32)

    # Exact supportive-label contract used by the auxiliary runner. Keep these
    # aliases explicit so report names remain independent of compatibility
    # columns retained above.
    support["__mfe_ge_1_0atr__"] = support["__peak_mfe_ge_1atr_12h__"].copy()
    support["__mfe_ge_2_0atr__"] = support["__peak_mfe_ge_2atr_12h__"].copy()
    support["__mfe_ge_3_0atr__"] = support["__peak_mfe_ge_3atr_12h__"].copy()
    support["__mfe_ge_4_0atr__"] = support["__peak_mfe_ge_4atr_12h__"].copy()
    support["__peak_mfe_atr_clip_6__"] = np.where(
        valid, np.clip(shape_peak_atr, 0.0, 6.0), np.nan
    ).astype(np.float32)
    support["__peak_mfe_atr_clip_8__"] = np.where(
        valid, np.clip(shape_peak_atr, 0.0, 8.0), np.nan
    ).astype(np.float32)
    support["__log1p_peak_mfe_12h_atr__"] = np.where(
        valid, np.log1p(np.clip(shape_peak_atr, 0.0, PEAK_MFE_ATR_CLIP)), np.nan
    ).astype(np.float32)
    support["__favorable_path_integral_atr__"] = support[
        "__mfe_integral_atr_hours_12h__"
    ].copy()
    for hours in (2, 4, 8):
        support[f"__mfe_{hours}h_over_mfe_12h__"] = support[
            f"__mfe_ratio_to_peak_at_{hours}h_12h__"
        ].copy()

    hit_15 = shape_fav_positive >= 1.5 * atr[:, None]
    reached_15 = valid & np.any(hit_15, axis=1)
    first_15 = np.argmax(hit_15, axis=1)
    stop_15 = np.where(reached_15, first_15 + 1, shape_horizon_bars)
    pre_15_mask = path_indices < stop_15[:, None]
    mae_pre_15 = np.maximum(
        np.max(np.where(pre_15_mask, shape_adv_positive, -np.inf), axis=1), 0.0
    )
    mae_pre_15_atr = np.divide(mae_pre_15, atr, out=np.full(n, np.nan), where=valid)
    support["__reaches_1_5atr_within_12h__"] = np.where(
        valid, reached_15.astype(float), np.nan
    ).astype(np.float32)
    support["__mae_before_1_5atr_mfe__"] = np.where(
        valid, np.clip(mae_pre_15_atr, 0.0, PEAK_MFE_ATR_CLIP), np.nan
    ).astype(np.float32)
    support["__mae_until_horizon_if_no_1_5atr__"] = np.where(
        valid & ~reached_15,
        np.clip(mae_pre_15_atr, 0.0, PEAK_MFE_ATR_CLIP),
        np.nan,
    ).astype(np.float32)
    for threshold, label in (
        (0.25, "0_25"),
        (0.50, "0_50"),
        (0.75, "0_75"),
        (1.00, "1_00"),
        (1.50, "1_50"),
    ):
        adverse_hit = pre_15_mask & (shape_adv_positive >= threshold * atr[:, None])
        support[f"__pre_1_5_mfe_mae_ge_{label}atr__"] = np.where(
            valid, np.any(adverse_hit, axis=1).astype(float), np.nan
        ).astype(np.float32)
    support["__hits_minus_1_0atr_before_plus_1_5atr__"] = support[
        "__pre_1_5_mfe_mae_ge_1_00atr__"
    ].copy()
    support["__hits_minus_0_5atr_before_plus_1_5atr__"] = support[
        "__pre_1_5_mfe_mae_ge_0_50atr__"
    ].copy()
    below_entry = pre_15_mask & (shape_adv_positive > 0.0)
    below_bars = np.sum(below_entry, axis=1)
    support["__bars_below_entry_before_1_5atr__"] = np.where(
        valid, below_bars, np.nan
    ).astype(np.float32)
    support["__fraction_bars_below_entry_before_1_5atr__"] = np.where(
        valid, below_bars / np.maximum(stop_15, 1), np.nan
    ).astype(np.float32)

    def _bars_to_recovery_fraction(fraction: float) -> np.ndarray:
        recovered = per_bar_recovery >= fraction
        any_recovered = trough_valid & np.any(recovered, axis=1)
        first = np.argmax(recovered, axis=1)
        return np.where(any_recovered, first - trough_index, np.nan).astype(np.float32)

    support["__bars_to_25pct_adverse_recovery__"] = _bars_to_recovery_fraction(0.25)
    support["__bars_to_50pct_adverse_recovery__"] = _bars_to_recovery_fraction(0.50)
    trough_minutes = (trough_index + 1.0) * float(bar_minutes)
    support["__adverse_trough_within_60m__"] = np.where(
        trough_valid, (trough_minutes <= 60.0).astype(float), np.nan
    ).astype(np.float32)
    support["__adverse_trough_within_120m__"] = np.where(
        trough_valid, (trough_minutes <= 120.0).astype(float), np.nan
    ).astype(np.float32)
    confirmation_index = trough_index + 2
    confirmation_available = trough_valid & (confirmation_index < shape_horizon_bars)
    support["__bars_to_confirmed_adverse_trough__"] = np.where(
        confirmation_available, confirmation_index + 1.0, np.nan
    ).astype(np.float32)
    first_hour_bars = max(1, min(shape_horizon_bars, int(np.floor(1.0 / bar_hours))))
    mfe_60_atr = np.divide(
        np.max(shape_fav_positive[:, :first_hour_bars], axis=1),
        atr,
        out=np.full(n, np.nan),
        where=valid,
    )
    support["__mfe_before_60m_atr__"] = np.where(
        valid, np.clip(mfe_60_atr, 0.0, PEAK_MFE_ATR_CLIP), np.nan
    ).astype(np.float32)
    support["__reaches_1_5atr_before_trough_confirmation__"] = np.where(
        valid,
        (reached_15 & confirmation_available & (first_15 <= confirmation_index)).astype(
            float
        ),
        np.nan,
    ).astype(np.float32)
    support["__trough_before_1_5atr_mfe__"] = np.where(
        valid,
        (trough_valid & reached_15 & (trough_index < first_15)).astype(float),
        np.nan,
    ).astype(np.float32)

    support["__future_slope_atr_per_hour_12h__"] = future_slope.astype(np.float32)
    support["__bars_to_80pct_peak__"] = support["__bars_to_80pct_peak_mfe_12h__"].copy()

    path_with_entry = np.concatenate(
        [np.zeros((n, 1), dtype=np.float64), shape_fav_positive], axis=1
    )

    def _first_hit_and_efficiency(
        level_return: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        hits = shape_fav_positive >= level_return[:, None]
        reached = valid & (level_return > 0.0) & np.any(hits, axis=1)
        first = np.argmax(hits, axis=1)
        increments = np.abs(np.diff(path_with_entry, axis=1))
        active = path_indices <= first[:, None]
        variation = np.sum(np.where(active, increments, 0.0), axis=1)
        efficiency = np.divide(
            level_return,
            variation,
            out=np.zeros(n, dtype=np.float64),
            where=reached & (variation > 0.0),
        )
        bars = np.where(reached, first + 1.0, np.nan)
        return bars.astype(np.float32), np.where(
            reached, np.clip(efficiency, 0.0, 1.0), np.nan
        ).astype(np.float32)

    bars_1, _ = _first_hit_and_efficiency(1.0 * atr)
    bars_15, efficiency_15 = _first_hit_and_efficiency(1.5 * atr)
    bars_2, efficiency_2 = _first_hit_and_efficiency(2.0 * atr)
    _, efficiency_80 = _first_hit_and_efficiency(0.8 * shape_peak_return)
    _, efficiency_90 = _first_hit_and_efficiency(0.9 * shape_peak_return)
    _, efficiency_meaningful = _first_hit_and_efficiency(usable_floor)
    support["__bars_to_1atr__"] = bars_1
    support["__bars_to_1_5atr__"] = bars_15
    support["__bars_to_2atr__"] = bars_2
    total_variation = np.sum(np.abs(np.diff(path_with_entry, axis=1)), axis=1)
    full_efficiency = np.divide(
        shape_peak_return,
        total_variation,
        out=np.zeros(n, dtype=np.float64),
        where=valid & (total_variation > 0.0),
    )
    support["__path_efficiency_12h__"] = np.where(
        valid & (total_variation > 0.0),
        np.clip(full_efficiency, 0.0, 1.0),
        np.nan,
    ).astype(np.float32)
    support["__path_efficiency_to_1_5atr__"] = efficiency_15
    support["__path_efficiency_to_2atr__"] = efficiency_2
    support["__path_efficiency_to_80pct_peak__"] = efficiency_80
    support["__path_efficiency_to_90pct_peak__"] = efficiency_90
    support["__path_efficiency_to_first_meaningful_mfe__"] = efficiency_meaningful
    return PathAuxiliaryTargets(
        peak_mfe_return_12h=peak.astype(np.float32),
        peak_mfe_atr_12h=peak_atr.astype(np.float32),
        time_to_first_meaningful_mfe_hours_12h=time_hours.astype(np.float32),
        mae_before_meaningful_mfe_atr_12h=mae_before_hit_atr.astype(np.float32),
        bars_before_price_stops_decreasing_12h=bars_before_turning.astype(np.float32),
        future_slope_atr_per_hour_12h=future_slope.astype(np.float32),
        log1p_peak_mfe_atr_12h=np.log1p(peak_atr).astype(np.float32),
        log1p_time_to_first_meaningful_mfe_hours_12h=np.log1p(time_hours).astype(
            np.float32
        ),
        log1p_mae_before_meaningful_mfe_atr_12h=np.log1p(mae_before_hit_atr).astype(
            np.float32
        ),
        log1p_bars_before_price_stops_decreasing_12h=np.log1p(
            bars_before_turning
        ).astype(np.float32),
        log1p_future_slope_atr_per_hour_12h=np.log1p(future_slope).astype(np.float32),
        supportive_columns=support,
        valid=valid,
        timing_valid=timing_valid,
        meaningful_mfe_reached=meaningful_reached,
    )


def _finite_pair(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(p)
    return y[mask], p[mask]


def _spearman_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) <= 1 or np.ptp(y_true) == 0.0 or np.ptp(y_pred) == 0.0:
        return 0.0
    rho = spearmanr(y_true, y_pred, nan_policy="omit").statistic
    return float(rho) if np.isfinite(rho) else 0.0


def _positive_target_regression_metrics(
    true_log_target: np.ndarray,
    pred_log_target: np.ndarray,
    *,
    target_name: str,
    top_fractions: Sequence[float] = (0.10, 0.05, 0.01),
    huber_delta: float = 1.0,
) -> dict[str, float]:
    """Return target-namespaced diagnostics for a non-negative log1p target."""

    y, p = _finite_pair(true_log_target, pred_log_target)
    if len(y) == 0:
        return {"rows": 0.0}
    true_natural = np.expm1(y).clip(min=0.0)
    pred_natural = np.expm1(p).clip(min=0.0)
    natural_error = pred_natural - true_natural
    abs_natural_error = np.abs(natural_error)
    delta = max(float(huber_delta), 1e-9)
    natural_huber = np.where(
        abs_natural_error <= delta,
        0.5 * natural_error**2,
        delta * (abs_natural_error - 0.5 * delta),
    )
    order = np.argsort(-p, kind="stable")
    out = {
        "rows": float(len(y)),
        f"{target_name}_log_mae": float(mean_absolute_error(y, p)),
        f"{target_name}_natural_mae": float(
            mean_absolute_error(true_natural, pred_natural)
        ),
        f"{target_name}_natural_rmse": float(
            np.sqrt(mean_squared_error(true_natural, pred_natural))
        ),
        f"{target_name}_natural_huber": float(np.mean(natural_huber)),
        f"{target_name}_spearman_ic": _spearman_ic(y, p),
    }
    for frac in top_fractions:
        k = max(1, int(np.ceil(float(frac) * len(order))))
        out[f"{target_name}_top_{int(round(100 * frac)):02d}pct_realized"] = float(
            np.mean(true_natural[order[:k]])
        )
    return out


def mae_before_meaningful_mfe_regression_metrics(
    true_log_mae_before_meaningful_mfe_atr: np.ndarray,
    pred_log_mae_before_meaningful_mfe_atr: np.ndarray,
    *,
    top_fractions: Sequence[float] = (0.10, 0.05, 0.01),
    huber_delta: float = 1.0,
) -> dict[str, float]:
    """Diagnostics for the ATR-normalized adverse-depth path target."""

    return _positive_target_regression_metrics(
        true_log_mae_before_meaningful_mfe_atr,
        pred_log_mae_before_meaningful_mfe_atr,
        target_name="mae_before_meaningful_mfe_atr",
        top_fractions=top_fractions,
        huber_delta=huber_delta,
    )


def future_slope_atr_per_hour_regression_metrics(
    true_log_future_slope_atr_per_hour: np.ndarray,
    pred_log_future_slope_atr_per_hour: np.ndarray,
    *,
    top_fractions: Sequence[float] = (0.10, 0.05, 0.01),
    huber_delta: float = 1.0,
) -> dict[str, float]:
    """Diagnostics for the 80%-of-peak favorable-accumulation-rate target."""

    return _positive_target_regression_metrics(
        true_log_future_slope_atr_per_hour,
        pred_log_future_slope_atr_per_hour,
        target_name="future_slope_atr_per_hour",
        top_fractions=top_fractions,
        huber_delta=huber_delta,
    )


def bars_before_price_stops_decreasing_regression_metrics(
    true_log_bars_before_price_stops_decreasing: np.ndarray,
    pred_log_bars_before_price_stops_decreasing: np.ndarray,
    *,
    decision_bars: Sequence[int] = (1, 2, 4, 8, 12),
) -> dict[str, float]:
    """Diagnostics for the side-normalized adverse turning-point bar target."""

    y, p = _finite_pair(
        true_log_bars_before_price_stops_decreasing,
        pred_log_bars_before_price_stops_decreasing,
    )
    if len(y) == 0:
        return {"rows": 0.0}
    true_bars = np.expm1(y).clip(min=0.0)
    pred_bars = np.expm1(p).clip(min=0.0)
    out = {
        "rows": float(len(y)),
        "bars_before_price_stops_decreasing_log_mae": float(mean_absolute_error(y, p)),
        "bars_before_price_stops_decreasing_mae_bars": float(
            mean_absolute_error(true_bars, pred_bars)
        ),
        "bars_before_price_stops_decreasing_spearman_ic": _spearman_ic(y, p),
    }
    for bars in decision_bars:
        threshold = max(int(bars), 0)
        out[f"bars_before_price_stops_decreasing_accuracy_by_{threshold}_bars"] = float(
            np.mean((true_bars <= threshold) == (pred_bars <= threshold))
        )
    return out


def timing_regression_metrics(
    true_log_time: np.ndarray,
    pred_log_time: np.ndarray,
    *,
    horizons_hours: Sequence[float] = USEFUL_HORIZONS_HOURS,
) -> dict[str, float]:
    y, p = _finite_pair(true_log_time, pred_log_time)
    if len(y) == 0:
        return {"rows": 0.0}
    true_hours = np.expm1(y).clip(min=0.0)
    pred_hours = np.expm1(p).clip(min=0.0)
    rho = (
        spearmanr(y, p, nan_policy="omit").statistic
        if len(y) > 1 and np.ptp(y) > 0.0 and np.ptp(p) > 0.0
        else np.nan
    )
    out = {
        "rows": float(len(y)),
        "mae_log_time": float(mean_absolute_error(y, p)),
        "mae_hours": float(mean_absolute_error(true_hours, pred_hours)),
        # A constant prediction has no ranking information. Treat its
        # undefined Spearman correlation as zero rather than invalidating an
        # otherwise finite regression trial.
        "spearman_ic": float(rho) if np.isfinite(rho) else 0.0,
    }
    for horizon in horizons_hours:
        out[f"accuracy_meaningful_mfe_by_{int(horizon)}h"] = float(
            np.mean((true_hours <= float(horizon)) == (pred_hours <= float(horizon)))
        )
    return out


def peak_mfe_regression_metrics(
    true_log_peak_mfe_atr: np.ndarray,
    pred_log_peak_mfe_atr: np.ndarray,
    *,
    top_fractions: Sequence[float] = (0.10, 0.05, 0.01),
    huber_delta: float = 1.0,
) -> dict[str, float]:
    y, p = _finite_pair(true_log_peak_mfe_atr, pred_log_peak_mfe_atr)
    if len(y) == 0:
        return {"rows": 0.0}
    error = p - y
    abs_error = np.abs(error)
    delta = max(float(huber_delta), 1e-9)
    huber = np.where(
        abs_error <= delta,
        0.5 * error**2,
        delta * (abs_error - 0.5 * delta),
    )
    rho = (
        spearmanr(y, p, nan_policy="omit").statistic
        if len(y) > 1 and np.ptp(y) > 0.0 and np.ptp(p) > 0.0
        else np.nan
    )
    realized = np.expm1(y).clip(min=0.0)
    order = np.argsort(-p, kind="stable")
    out = {
        "rows": float(len(y)),
        "mae_log_peak_mfe_atr": float(mean_absolute_error(y, p)),
        "huber_loss": float(np.mean(huber)),
        "rmse_log_peak_mfe_atr": float(np.sqrt(mean_squared_error(y, p))),
        "spearman_ic": float(rho) if np.isfinite(rho) else 0.0,
    }
    for frac in top_fractions:
        k = max(1, int(np.ceil(float(frac) * len(order))))
        out[f"top_{int(round(100 * frac)):02d}pct_realized_peak_mfe_atr"] = float(
            np.mean(realized[order[:k]])
        )
    return out


def required_target_columns() -> tuple[str, ...]:
    return tuple(
        PathAuxiliaryTargets(
            *(np.empty(0, dtype=np.float32) for _ in range(11)),
            supportive_columns={
                column: np.empty(0, dtype=np.float32)
                for column in dict.fromkeys(
                    column
                    for groups in (
                        _LEGACY_SUPPORTIVE_LABEL_COLUMNS,
                        SUPPORTIVE_LABEL_COLUMNS,
                    )
                    for columns in groups.values()
                    for column in columns
                    if column != "__future_slope_atr_per_hour_12h__"
                )
            },
            valid=np.empty(0, dtype=bool),
            timing_valid=np.empty(0, dtype=bool),
            meaningful_mfe_reached=np.empty(0, dtype=bool),
        ).as_columns()
    )
