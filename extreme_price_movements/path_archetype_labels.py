"""Causal future-path summaries and frozen economic path-archetype labels.

The values emitted here are realised targets.  They must never be supplied as
live features.  In particular, ``path_archetype`` is a deterministic training
target; an inference-time classifier must predict it from validated pre-entry
features rather than reading it from a candidate row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .path_auxiliary_targets import MIN_USABLE_MFE_ATR, MIN_USABLE_MFE_RETURN

PATH_HORIZONS_HOURS: tuple[int, ...] = (1, 2, 4, 8, 12, 24)
PATH_SUMMARY_PREFIX = "path_arch_"
PATH_ARCHETYPE_RULE_VERSION = "economic_path_v6_shape_strength_15atr_stop1r"
CATBOOST_ARCHETYPE_COST_RETURN = 0.01
CATBOOST_ARCHETYPE_ATR_FLOOR = 1.50
CATBOOST_ARCHETYPE_NET_MARGIN_ATR = 0.50
PATH_SHAPE_TYPES: tuple[str, ...] = (
    "fast_clean_winner",
    "fast_winner_early_drawdown",
    "slow_grinder",
    "late_breakout",
    "early_mfe_full_reversal",
    "immediate_adverse_path",
    "noisy_timeout_usable_mfe",
    "dead_timeout",
)
ATR_REALIZATION_THRESHOLDS: tuple[float, ...] = (1.50, 2.00, 3.00, 5.00)
PATH_REALIZATION_STRENGTH_TYPES: tuple[str, ...] = (
    "below_150atr",
    "atr150_200",
    "atr200_300",
    "atr300_500",
    "atr500_plus",
)
PATH_ARCHETYPE_TYPES: tuple[str, ...] = tuple(
    f"{shape}__{strength}"
    for shape in PATH_SHAPE_TYPES
    for strength in PATH_REALIZATION_STRENGTH_TYPES
)


@dataclass(frozen=True)
class PathArchetypeLabelConfig:
    """Column and execution-time contract for materialising path targets."""

    timestamp_col: str = "__ts__"
    symbol_col: str = "__symbol__"
    side_col: str = "side"
    bar_timestamp_col: str = "timestamp"
    bar_symbol_col: str = "symbol"
    decision_delay_hours: int = 1
    bar_hours: float = 1.0
    horizons_hours: tuple[int, ...] = PATH_HORIZONS_HOURS
    prefix: str = PATH_SUMMARY_PREFIX
    rule_version: str = PATH_ARCHETYPE_RULE_VERSION
    default_cost_return: float = CATBOOST_ARCHETYPE_COST_RETURN
    default_activation_r: float = 1.0


def _utc(values: Iterable[object]) -> pd.Series:
    return pd.to_datetime(pd.Series(values), errors="coerce", utc=True)


def _first_existing(columns: Iterable[str], candidates: Sequence[str]) -> str | None:
    available = set(map(str, columns))
    return next((column for column in candidates if column in available), None)


def _side_sign(values: pd.Series) -> np.ndarray:
    raw = values.astype(str).str.strip().str.lower()
    sign = np.where(raw.isin(("short", "-1", "sell", "s")).to_numpy(), -1.0, 1.0)
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    return np.where(np.isfinite(numeric), np.where(numeric < 0.0, -1.0, 1.0), sign)


def _empty_summary(prefix: str, horizons: Sequence[int]) -> dict[str, float]:
    result = {f"{prefix}mfe_{h}h_r": np.nan for h in horizons}
    result.update({f"{prefix}mae_{h}h_r": np.nan for h in horizons})
    result.update(
        {
            f"{prefix}time_to_025r_h": np.nan,
            f"{prefix}time_to_05r_h": np.nan,
            f"{prefix}time_to_1r_h": np.nan,
            f"{prefix}time_to_tp_h": np.nan,
            f"{prefix}time_to_trailing_h": np.nan,
            f"{prefix}time_to_stop_h": np.nan,
            f"{prefix}mfe_before_mae": np.nan,
            f"{prefix}mae_before_mfe": np.nan,
            f"{prefix}time_to_first_meaningful_mfe_h": np.nan,
            f"{prefix}time_to_90pct_peak_mfe_h": np.nan,
            f"{prefix}usable_mfe_floor_return": np.nan,
            f"{prefix}usable_mfe_threshold_r": np.nan,
            f"{prefix}raw_peak_mfe_r": np.nan,
            f"{prefix}peak_mfe_r": np.nan,
            f"{prefix}peak_mfe_atr": np.nan,
            f"{prefix}mfe_to_cost": np.nan,
            f"{prefix}mfe_to_activation_distance": np.nan,
            f"{prefix}early_late_ratio": np.nan,
            f"{prefix}efficiency": np.nan,
            f"{prefix}reversal_count": np.nan,
            f"{prefix}final_return_r": np.nan,
            f"{prefix}final_to_peak": np.nan,
            f"{prefix}cost_atr": np.nan,
            f"{prefix}meaningful_mfe_threshold_atr": np.nan,
            f"{prefix}peak_mfe_minus_cost_atr": np.nan,
            f"{prefix}peak_mfe_div_cost": np.nan,
            f"{prefix}reaches_meaningful_mfe": np.nan,
            f"{prefix}bars_to_meaningful_mfe": np.nan,
            f"{prefix}bars_to_80pct_peak": np.nan,
            f"{prefix}bars_to_90pct_peak": np.nan,
            f"{prefix}mfe_2h_over_mfe_12h": np.nan,
            f"{prefix}mfe_4h_over_mfe_12h": np.nan,
            f"{prefix}mfe_8h_over_mfe_12h": np.nan,
            f"{prefix}bars_to_stop": np.nan,
            f"{prefix}stop_before_meaningful_mfe": np.nan,
            f"{prefix}mfe_before_stop_r": np.nan,
            f"{prefix}mae_2h_r": np.nan,
            f"{prefix}mae_4h_r": np.nan,
            f"{prefix}mae_before_meaningful_mfe_r": np.nan,
            f"{prefix}bars_below_entry_before_meaningful_mfe": np.nan,
            f"{prefix}adverse_area_before_meaningful_mfe_r": np.nan,
            f"{prefix}path_efficiency_to_meaningful_mfe": np.nan,
            f"{prefix}path_efficiency_to_90pct_peak": np.nan,
            f"{prefix}future_slope_atr_per_hour_4h": np.nan,
            f"{prefix}future_slope_atr_per_hour_12h": np.nan,
            f"{prefix}late_minus_early_slope": np.nan,
            f"{prefix}final_return_net_1pct": np.nan,
            f"{prefix}peak_retention_ratio": np.nan,
            f"{prefix}fraction_bars_above_50pct_peak": np.nan,
            f"{prefix}risk_fraction": np.nan,
            f"{prefix}atr_fraction": np.nan,
        }
    )
    for hour in range(1, 13):
        result[f"{prefix}raw_mfe_r_{hour}h"] = np.nan
        result[f"{prefix}raw_mfe_atr_{hour}h"] = np.nan
        result[f"{prefix}raw_mae_r_{hour}h"] = np.nan
        result[f"{prefix}close_return_r_{hour}h"] = np.nan
        result[f"{prefix}cumulative_variation_r_{hour}h"] = np.nan
    for threshold in ATR_REALIZATION_THRESHOLDS:
        token = f"{int(round(threshold * 100)):03d}atr"
        result[f"{prefix}reached_{token}"] = np.nan
        result[f"{prefix}time_to_{token}_h"] = np.nan
    return result


def path_summary_columns(prefix: str = PATH_SUMMARY_PREFIX) -> tuple[str, ...]:
    return tuple(_empty_summary(prefix, PATH_HORIZONS_HOURS))


def summarize_side_relative_path(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    *,
    entry_price: float,
    risk_distance: float,
    atr_fraction: float,
    side_sign: float,
    bar_hours: float = 1.0,
    horizons_hours: Sequence[int] = PATH_HORIZONS_HOURS,
    take_profit_r: float | None = None,
    trailing_trigger_r: float | None = None,
    stop_r: float | None = None,
    cost_return: float = CATBOOST_ARCHETYPE_COST_RETURN,
    activation_distance_return: float = np.nan,
    prefix: str = PATH_SUMMARY_PREFIX,
) -> dict[str, float]:
    """Summarise an already-causal future OHLC path in side-relative R units.

    ``highs`` and ``lows`` determine MFE/MAE and barrier timing while closes
    determine directional efficiency and reversals.  The zero bar is part of
    the future path and is required to occur at or after ``decision_ts`` by the
    materializer.
    """
    out = _empty_summary(prefix, horizons_hours)
    high = np.asarray(highs, dtype=np.float64)
    low = np.asarray(lows, dtype=np.float64)
    close = np.asarray(closes, dtype=np.float64)
    if (
        high.size == 0
        or high.shape != low.shape
        or high.shape != close.shape
        or not np.isfinite(
            [entry_price, risk_distance, atr_fraction, side_sign, bar_hours]
        ).all()
        or entry_price <= 0.0
        or risk_distance <= 0.0
        or atr_fraction <= 0.0
        or bar_hours <= 0.0
        or not (
            np.isfinite(high).all()
            and np.isfinite(low).all()
            and np.isfinite(close).all()
        )
    ):
        return out

    if side_sign >= 0.0:
        favorable = (high - entry_price) / risk_distance
        adverse = (low - entry_price) / risk_distance
        close_r = (close - entry_price) / risk_distance
    else:
        favorable = (entry_price - low) / risk_distance
        adverse = (entry_price - high) / risk_distance
        close_r = (entry_price - close) / risk_distance

    risk_fraction = risk_distance / entry_price
    if not np.isfinite(activation_distance_return) or activation_distance_return <= 0.0:
        activation_r = (
            trailing_trigger_r
            if trailing_trigger_r is not None
            and np.isfinite(trailing_trigger_r)
            and trailing_trigger_r > 0.0
            else take_profit_r
            if take_profit_r is not None
            and np.isfinite(take_profit_r)
            and take_profit_r > 0.0
            else 1.0
        )
        activation_distance_return = float(activation_r) * risk_fraction
    usable_floor_return = max(
        MIN_USABLE_MFE_ATR * atr_fraction,
        MIN_USABLE_MFE_RETURN,
    )
    usable_threshold_r = usable_floor_return / risk_fraction

    for horizon in horizons_hours:
        bars = int(round(float(horizon) / bar_hours))
        if bars <= 0 or high.size < bars:
            continue
        raw_horizon_peak = float(np.max(favorable[:bars]))
        out[f"{prefix}mfe_{horizon}h_r"] = (
            raw_horizon_peak if raw_horizon_peak >= usable_threshold_r else 0.0
        )
        out[f"{prefix}mae_{horizon}h_r"] = float(np.min(adverse[:bars]))

    def first_at_or_above(threshold: float | None) -> float:
        if threshold is None or not np.isfinite(threshold):
            return np.nan
        found = np.flatnonzero(
            favorable >= max(float(threshold), usable_threshold_r)
        )
        return float((found[0] + 1) * bar_hours) if found.size else np.nan

    def first_at_or_below(threshold: float | None) -> float:
        if threshold is None or not np.isfinite(threshold):
            return np.nan
        found = np.flatnonzero(adverse <= -abs(float(threshold)))
        return float((found[0] + 1) * bar_hours) if found.size else np.nan

    raw_peak_i, trough_i = int(np.argmax(favorable)), int(np.argmin(adverse))
    raw_peak = float(favorable[raw_peak_i])
    peak_is_usable = raw_peak >= usable_threshold_r
    peak_i = raw_peak_i if peak_is_usable else -1
    half = max(1, favorable.size // 2)
    early_peak_raw = float(np.max(favorable[:half]))
    late_peak_raw = (
        float(np.max(favorable[half:])) if half < favorable.size else early_peak_raw
    )
    early_peak = early_peak_raw if early_peak_raw >= usable_threshold_r else 0.0
    late_peak = late_peak_raw if late_peak_raw >= usable_threshold_r else 0.0
    variation = float(np.abs(np.diff(np.r_[0.0, close_r])).sum())
    moves = np.sign(np.diff(close_r))
    moves = moves[moves != 0.0]
    peak = raw_peak if peak_is_usable else 0.0
    raw_peak_return = max(raw_peak * risk_fraction, 0.0)
    archetype_cost_return = CATBOOST_ARCHETYPE_COST_RETURN
    cost_atr = archetype_cost_return / atr_fraction
    meaningful_mfe_threshold_atr = max(
        CATBOOST_ARCHETYPE_ATR_FLOOR,
        cost_atr + CATBOOST_ARCHETYPE_NET_MARGIN_ATR,
    )
    meaningful_mfe_threshold_r = (
        meaningful_mfe_threshold_atr * atr_fraction / risk_fraction
    )
    cost_aware_hits = np.flatnonzero(favorable >= meaningful_mfe_threshold_r)
    cost_aware_reached = bool(cost_aware_hits.size)
    cost_aware_i = int(cost_aware_hits[0]) if cost_aware_reached else favorable.size - 1
    support_peak = max(raw_peak, 0.0)
    peak_80_hits = (
        np.flatnonzero(favorable >= 0.80 * support_peak)
        if support_peak > 0.0
        else np.array([], dtype=np.int64)
    )
    peak_90_hits_cost = (
        np.flatnonzero(favorable >= 0.90 * support_peak)
        if support_peak > 0.0
        else np.array([], dtype=np.int64)
    )
    stop_threshold = (
        abs(float(stop_r)) if stop_r is not None and np.isfinite(stop_r) else np.nan
    )
    stop_hits = (
        np.flatnonzero(adverse <= -stop_threshold)
        if np.isfinite(stop_threshold)
        else np.array([], dtype=np.int64)
    )
    stop_i = int(stop_hits[0]) if stop_hits.size else favorable.size - 1
    pre_meaningful_slice = slice(0, cost_aware_i + 1)
    pre_meaningful_close = close_r[pre_meaningful_slice]
    pre_meaningful_adverse = adverse[pre_meaningful_slice]

    def raw_horizon_peak(hours: float) -> float:
        count = min(favorable.size, max(1, int(round(hours / bar_hours))))
        return max(float(np.max(favorable[:count])), 0.0)

    def atr_slope(hours: float) -> float:
        count = min(close_r.size, max(2, int(round(hours / bar_hours))))
        y = close_r[:count] * risk_fraction / atr_fraction
        x = (np.arange(count, dtype=np.float64) + 1.0) * bar_hours
        x_centered = x - x.mean()
        denominator = float(np.dot(x_centered, x_centered))
        return (
            float(np.dot(x_centered, y - y.mean()) / denominator)
            if denominator > 0.0
            else 0.0
        )

    mfe_2h_raw = raw_horizon_peak(2.0)
    mfe_4h_raw = raw_horizon_peak(4.0)
    mfe_8h_raw = raw_horizon_peak(8.0)
    mfe_12h_raw = raw_horizon_peak(12.0)
    slope_4h = atr_slope(4.0)
    slope_12h = atr_slope(12.0)
    meaningful_variation = float(
        np.abs(np.diff(np.r_[0.0, pre_meaningful_close])).sum()
    )
    peak_90_i = int(peak_90_hits_cost[0]) if peak_90_hits_cost.size else raw_peak_i
    peak_90_close = close_r[: peak_90_i + 1]
    peak_90_variation = float(np.abs(np.diff(np.r_[0.0, peak_90_close])).sum())
    final_return_net = float(close_r[-1] * risk_fraction - archetype_cost_return)
    peak_net_return = float(raw_peak_return - archetype_cost_return)
    first_meaningful = np.flatnonzero(favorable >= usable_threshold_r)
    time_first_meaningful = (
        float((first_meaningful[0] + 1) * bar_hours)
        if first_meaningful.size
        else float(favorable.size * bar_hours)
    )
    peak_90_hits = np.flatnonzero(favorable >= 0.90 * raw_peak)
    time_peak_90 = (
        float((peak_90_hits[0] + 1) * bar_hours)
        if peak_is_usable and peak_90_hits.size
        else float(favorable.size * bar_hours)
    )
    out.update(
        {
            f"{prefix}time_to_025r_h": first_at_or_above(0.25),
            f"{prefix}time_to_05r_h": first_at_or_above(0.50),
            f"{prefix}time_to_1r_h": first_at_or_above(1.0),
            f"{prefix}time_to_tp_h": first_at_or_above(take_profit_r),
            f"{prefix}time_to_trailing_h": first_at_or_above(trailing_trigger_r),
            f"{prefix}time_to_stop_h": first_at_or_below(stop_r),
            f"{prefix}mfe_before_mae": (
                float(peak_i < trough_i) if peak_is_usable else np.nan
            ),
            f"{prefix}mae_before_mfe": (
                float(trough_i < peak_i) if peak_is_usable else np.nan
            ),
            f"{prefix}time_to_first_meaningful_mfe_h": time_first_meaningful,
            f"{prefix}time_to_90pct_peak_mfe_h": time_peak_90,
            f"{prefix}usable_mfe_floor_return": usable_floor_return,
            f"{prefix}usable_mfe_threshold_r": usable_threshold_r,
            f"{prefix}raw_peak_mfe_r": raw_peak,
            f"{prefix}peak_mfe_r": peak,
            f"{prefix}peak_mfe_atr": float(
                np.clip(raw_peak_return / atr_fraction, 0.0, 10.0)
            ),
            f"{prefix}mfe_to_cost": (
                raw_peak_return / cost_return
                if np.isfinite(cost_return) and cost_return > 0.0
                else np.nan
            ),
            f"{prefix}mfe_to_activation_distance": (
                raw_peak_return / activation_distance_return
                if np.isfinite(activation_distance_return)
                and activation_distance_return > 0.0
                else np.nan
            ),
            f"{prefix}early_late_ratio": early_peak / max(abs(late_peak), 1e-6),
            f"{prefix}efficiency": float(close_r[-1] / variation) if variation else 0.0,
            f"{prefix}reversal_count": float(np.sum(moves[1:] != moves[:-1])),
            f"{prefix}final_return_r": float(close_r[-1]),
            f"{prefix}final_to_peak": (
                float(close_r[-1] / max(abs(peak), 1e-6))
                if peak_is_usable
                else np.nan
            ),
            f"{prefix}cost_atr": cost_atr,
            f"{prefix}meaningful_mfe_threshold_atr": meaningful_mfe_threshold_atr,
            f"{prefix}peak_mfe_minus_cost_atr": float(
                raw_peak_return / atr_fraction - cost_atr
            ),
            f"{prefix}peak_mfe_div_cost": (
                raw_peak_return / archetype_cost_return
            ),
            f"{prefix}reaches_meaningful_mfe": float(cost_aware_reached),
            f"{prefix}bars_to_meaningful_mfe": float(cost_aware_i + 1),
            f"{prefix}bars_to_80pct_peak": float(
                (peak_80_hits[0] + 1) if peak_80_hits.size else favorable.size
            ),
            f"{prefix}bars_to_90pct_peak": float(
                (peak_90_hits_cost[0] + 1)
                if peak_90_hits_cost.size
                else favorable.size
            ),
            f"{prefix}mfe_2h_over_mfe_12h": mfe_2h_raw / max(mfe_12h_raw, 1e-6),
            f"{prefix}mfe_4h_over_mfe_12h": mfe_4h_raw / max(mfe_12h_raw, 1e-6),
            f"{prefix}mfe_8h_over_mfe_12h": mfe_8h_raw / max(mfe_12h_raw, 1e-6),
            f"{prefix}bars_to_stop": float(
                (stop_hits[0] + 1) if stop_hits.size else favorable.size
            ),
            f"{prefix}stop_before_meaningful_mfe": float(
                bool(stop_hits.size)
                and (not cost_aware_reached or int(stop_hits[0]) < cost_aware_i)
            ),
            f"{prefix}mfe_before_stop_r": max(
                float(np.max(favorable[: stop_i + 1])), 0.0
            ),
            f"{prefix}mae_before_meaningful_mfe_r": float(
                abs(min(float(np.min(pre_meaningful_adverse)), 0.0))
            ),
            f"{prefix}bars_below_entry_before_meaningful_mfe": float(
                np.sum(pre_meaningful_close < 0.0)
            ),
            f"{prefix}adverse_area_before_meaningful_mfe_r": float(
                np.maximum(-pre_meaningful_close, 0.0).sum() * bar_hours
            ),
            f"{prefix}path_efficiency_to_meaningful_mfe": (
                float(
                    np.clip(
                        meaningful_mfe_threshold_r
                        / max(meaningful_variation, 1e-6),
                        0.0,
                        1.0,
                    )
                )
                if cost_aware_reached
                else 0.0
            ),
            f"{prefix}path_efficiency_to_90pct_peak": (
                float(
                    np.clip(
                        0.90 * support_peak / max(peak_90_variation, 1e-6),
                        0.0,
                        1.0,
                    )
                )
                if support_peak > 0.0
                else 0.0
            ),
            f"{prefix}future_slope_atr_per_hour_4h": slope_4h,
            f"{prefix}future_slope_atr_per_hour_12h": slope_12h,
            f"{prefix}late_minus_early_slope": slope_12h - slope_4h,
            f"{prefix}final_return_net_1pct": final_return_net,
            f"{prefix}peak_retention_ratio": (
                float(np.clip(final_return_net / peak_net_return, -5.0, 2.0))
                if peak_net_return > 1e-8
                else np.nan
            ),
            f"{prefix}fraction_bars_above_50pct_peak": float(
                np.mean(close_r >= 0.50 * support_peak)
                if support_peak > 0.0
                else 0.0
            ),
            f"{prefix}risk_fraction": risk_fraction,
            f"{prefix}atr_fraction": atr_fraction,
        }
    )
    for hour in range(1, 13):
        count = min(favorable.size, max(1, int(round(hour / bar_hours))))
        raw_mfe_r = max(float(np.max(favorable[:count])), 0.0)
        out[f"{prefix}raw_mfe_r_{hour}h"] = raw_mfe_r
        out[f"{prefix}raw_mfe_atr_{hour}h"] = (
            raw_mfe_r * risk_fraction / atr_fraction
        )
        out[f"{prefix}raw_mae_r_{hour}h"] = float(
            np.min(adverse[:count])
        )
        out[f"{prefix}close_return_r_{hour}h"] = float(close_r[count - 1])
        out[f"{prefix}cumulative_variation_r_{hour}h"] = float(
            np.abs(np.diff(np.r_[0.0, close_r[:count]])).sum()
        )
    for threshold in ATR_REALIZATION_THRESHOLDS:
        token = f"{int(round(threshold * 100)):03d}atr"
        threshold_r = threshold * atr_fraction / risk_fraction
        hits = np.flatnonzero(favorable >= threshold_r)
        out[f"{prefix}reached_{token}"] = float(bool(hits.size))
        out[f"{prefix}time_to_{token}_h"] = (
            float((hits[0] + 1) * bar_hours) if hits.size else np.nan
        )
    return out


def deterministic_path_archetype(
    summary: Mapping[str, object], *, prefix: str = PATH_SUMMARY_PREFIX
) -> str | None:
    """Apply the frozen, interpretable economic type rules.

    Rule precedence is part of ``economic_path_v6_shape_strength_15atr_stop1r`` and respects
    event order.
    A later stop cannot relabel a path that reached a useful favorable excursion
    first as an immediate failure. No fitted quantity or cluster identity enters.
    """

    def value(name: str) -> float:
        try:
            return float(summary[f"{prefix}{name}"])
        except (KeyError, TypeError, ValueError):
            return np.nan

    peak = value("peak_mfe_r")
    final = value("final_return_r")
    mfe_4h, mfe_12h = value("mfe_4h_r"), value("mfe_12h_r")
    mae_4h = value("mae_4h_r")
    time_1r, time_stop = value("time_to_1r_h"), value("time_to_stop_h")
    time_first = value("time_to_first_meaningful_mfe_h")
    time_peak_90 = value("time_to_90pct_peak_mfe_h")
    if not np.isfinite([peak, final, mfe_4h, mfe_12h, mae_4h]).all():
        return None
    favorable_before_stop = np.isfinite(time_1r) and (
        not np.isfinite(time_stop) or time_1r <= time_stop
    )
    if np.isfinite(time_stop) and time_stop <= 2.0 and not favorable_before_stop:
        return "immediate_adverse_path"
    if peak >= 0.75 and time_peak_90 <= 8.0 and final <= 0.0:
        return "early_mfe_full_reversal"
    if time_first <= 2.0 and time_peak_90 <= 4.0 and final > 0.0:
        return (
            "fast_clean_winner"
            if mae_4h > -0.50
            else "fast_winner_early_drawdown"
        )
    if mfe_4h < 0.50 and mfe_12h >= 1.0:
        return "late_breakout"
    if peak >= 0.50 and final > 0.0:
        return "slow_grinder"
    if peak >= 0.50:
        return "noisy_timeout_usable_mfe"
    return "dead_timeout"


def deterministic_path_realization_strength(
    summary: Mapping[str, object], *, prefix: str = PATH_SUMMARY_PREFIX
) -> str | None:
    """Return the frozen ATR-relative peak-MFE strength band."""

    try:
        peak_atr = float(summary[f"{prefix}peak_mfe_atr"])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(peak_atr):
        return None
    if peak_atr < 1.50:
        return "below_150atr"
    if peak_atr < 2.00:
        return "atr150_200"
    if peak_atr < 3.00:
        return "atr200_300"
    if peak_atr < 5.00:
        return "atr300_500"
    return "atr500_plus"


def deterministic_combined_path_archetype(
    summary: Mapping[str, object], *, prefix: str = PATH_SUMMARY_PREFIX
) -> str | None:
    """Combine path shape and ATR-relative realization strength."""

    shape = deterministic_path_archetype(summary, prefix=prefix)
    strength = deterministic_path_realization_strength(summary, prefix=prefix)
    if shape is None or strength is None:
        return None
    return f"{shape}__{strength}"


def _geometry_arrays(
    frame: pd.DataFrame,
    *,
    default_cost_return: float,
    default_activation_r: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    entry_col = _first_existing(
        frame.columns, ("entry_price", "__entry_price__", "entry", "__entry__")
    )
    stop_col = _first_existing(
        frame.columns, ("stop_price", "__stop_price__", "stop", "__sl_price__")
    )
    risk_col = _first_existing(
        frame.columns,
        ("risk_distance", "risk_abs", "stop_distance", "__risk_distance__"),
    )
    barrier_col = _first_existing(
        frame.columns, ("__barrier_pct__", "barrier_pct", "risk_pct")
    )
    atr_col = _first_existing(
        frame.columns,
        (
            "atr_fraction",
            "__atr_fraction__",
            "__path_auxiliary_atr_fraction__",
            "atr_pct",
            "atr_pct_base",
        ),
    )
    tp_col = _first_existing(
        frame.columns, ("take_profit_price", "tp_price", "__tp_price__")
    )
    trail_col = _first_existing(
        frame.columns,
        (
            "trailing_activation_price",
            "trail_activation_price",
            "__trailing_activation_price__",
        ),
    )
    tp_r_col = _first_existing(frame.columns, ("take_profit_r", "tp_r", "__tp_r__"))
    trail_r_col = _first_existing(
        frame.columns,
        ("trailing_trigger_r", "trailing_activation_r", "trail_r"),
    )
    stop_r_col = _first_existing(frame.columns, ("stop_r", "sl_r", "__sl_r__"))
    cost_col = _first_existing(
        frame.columns,
        (
            "path_cost_return",
            "round_trip_cost_return",
            "execution_cost_return",
            "cost_return",
        ),
    )
    activation_return_col = _first_existing(
        frame.columns,
        ("activation_distance_return", "trailing_activation_distance_return"),
    )
    n = len(frame)
    entry = (
        pd.to_numeric(frame[entry_col], errors="coerce").to_numpy(np.float64)
        if entry_col
        else np.full(n, np.nan)
    )
    risk = (
        pd.to_numeric(frame[risk_col], errors="coerce").to_numpy(np.float64)
        if risk_col
        else np.full(n, np.nan)
    )
    if stop_col:
        stop = pd.to_numeric(frame[stop_col], errors="coerce").to_numpy(np.float64)
        risk = np.where(np.isfinite(risk) & (risk > 0.0), risk, np.abs(entry - stop))
    barrier = (
        pd.to_numeric(frame[barrier_col], errors="coerce").to_numpy(np.float64)
        if barrier_col
        else np.full(n, np.nan)
    )
    atr_fraction = (
        pd.to_numeric(frame[atr_col], errors="coerce").to_numpy(np.float64)
        if atr_col
        else np.full(n, np.nan)
    )
    if barrier_col:
        risk = np.where(np.isfinite(risk) & (risk > 0.0), risk, np.abs(entry * barrier))
    tp = (
        pd.to_numeric(frame[tp_col], errors="coerce").to_numpy(np.float64)
        if tp_col
        else np.full(n, np.nan)
    )
    trail = (
        pd.to_numeric(frame[trail_col], errors="coerce").to_numpy(np.float64)
        if trail_col
        else np.full(n, np.nan)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        tp_r = np.abs(tp - entry) / risk
        trail_r = np.abs(trail - entry) / risk
    if tp_r_col:
        direct_tp_r = pd.to_numeric(frame[tp_r_col], errors="coerce").to_numpy(
            np.float64
        )
        tp_r = np.where(np.isfinite(direct_tp_r), direct_tp_r, tp_r)
    if trail_r_col:
        direct_trail_r = pd.to_numeric(frame[trail_r_col], errors="coerce").to_numpy(
            np.float64
        )
        trail_r = np.where(np.isfinite(direct_trail_r), direct_trail_r, trail_r)
    has_risk_geometry = (np.isfinite(risk) & (risk > 0.0)) | (
        np.isfinite(barrier) & (barrier > 0.0)
    )
    stop_r = np.where(has_risk_geometry, 1.0, np.nan)
    if stop_r_col:
        direct_stop_r = pd.to_numeric(frame[stop_r_col], errors="coerce").to_numpy(
            np.float64
        )
        stop_r = np.where(np.isfinite(direct_stop_r), direct_stop_r, stop_r)
    cost_return = (
        pd.to_numeric(frame[cost_col], errors="coerce").to_numpy(np.float64)
        if cost_col
        else np.full(n, float(default_cost_return), dtype=np.float64)
    )
    cost_return = np.where(
        np.isfinite(cost_return) & (cost_return > 0.0),
        cost_return,
        float(default_cost_return),
    )
    if activation_return_col:
        activation_return = pd.to_numeric(
            frame[activation_return_col], errors="coerce"
        ).to_numpy(np.float64)
    else:
        activation_return = np.full(n, np.nan, dtype=np.float64)
    risk_fraction = np.divide(
        risk,
        entry,
        out=np.full(n, np.nan, dtype=np.float64),
        where=np.isfinite(entry) & (entry > 0.0),
    )
    activation_r = np.where(
        np.isfinite(trail_r) & (trail_r > 0.0),
        trail_r,
        np.where(
            np.isfinite(tp_r) & (tp_r > 0.0), tp_r, float(default_activation_r)
        ),
    )
    inferred_activation_return = activation_r * risk_fraction
    activation_return = np.where(
        np.isfinite(activation_return) & (activation_return > 0.0),
        activation_return,
        inferred_activation_return,
    )
    return (
        entry,
        risk,
        tp_r,
        trail_r,
        stop_r,
        barrier,
        atr_fraction,
        cost_return,
        activation_return,
    )


def _summarize_side_relative_path_batch(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    *,
    entry_price: np.ndarray,
    risk_distance: np.ndarray,
    atr_fraction: np.ndarray,
    side_sign: np.ndarray,
    bar_hours: float,
    horizons_hours: Sequence[int],
    take_profit_r: np.ndarray,
    trailing_trigger_r: np.ndarray,
    stop_r: np.ndarray,
    cost_return: np.ndarray,
    archetype_cost_return: np.ndarray | None = None,
    activation_distance_return: np.ndarray,
    prefix: str,
) -> dict[str, np.ndarray]:
    """Vectorized equivalent of ``summarize_side_relative_path`` for equal paths.

    Callers keep batches bounded so the three ``rows x horizon`` path matrices do
    not grow with the complete candidate universe.
    """
    rows, bars = highs.shape
    summary = {
        column: np.full(rows, np.nan, dtype=np.float64)
        for column in _empty_summary(prefix, horizons_hours)
    }
    valid = (
        (lows.shape == highs.shape)
        & (closes.shape == highs.shape)
        & np.isfinite(highs).all(axis=1)
        & np.isfinite(lows).all(axis=1)
        & np.isfinite(closes).all(axis=1)
        & np.isfinite(entry_price)
        & np.isfinite(risk_distance)
        & np.isfinite(atr_fraction)
        & np.isfinite(side_sign)
        & (entry_price > 0.0)
        & (risk_distance > 0.0)
        & (atr_fraction > 0.0)
        & np.isfinite(bar_hours)
        & (bar_hours > 0.0)
    )
    if not valid.any():
        return summary

    with np.errstate(divide="ignore", invalid="ignore"):
        long_favorable = (highs - entry_price[:, None]) / risk_distance[:, None]
        long_adverse = (lows - entry_price[:, None]) / risk_distance[:, None]
        long_close = (closes - entry_price[:, None]) / risk_distance[:, None]
        short_favorable = (entry_price[:, None] - lows) / risk_distance[:, None]
        short_adverse = (entry_price[:, None] - highs) / risk_distance[:, None]
        short_close = (entry_price[:, None] - closes) / risk_distance[:, None]
    is_long = side_sign[:, None] >= 0.0
    favorable = np.where(is_long, long_favorable, short_favorable)
    adverse = np.where(is_long, long_adverse, short_adverse)
    close_r = np.where(is_long, long_close, short_close)
    with np.errstate(divide="ignore", invalid="ignore"):
        usable_floor_return = np.maximum(
            MIN_USABLE_MFE_ATR * atr_fraction,
            MIN_USABLE_MFE_RETURN,
        )
        usable_threshold_r = usable_floor_return / (risk_distance / entry_price)

    for horizon in horizons_hours:
        horizon_bars = int(round(float(horizon) / bar_hours))
        if horizon_bars <= 0 or bars < horizon_bars:
            continue
        raw_horizon_peak = np.max(favorable[:, :horizon_bars], axis=1)
        summary[f"{prefix}mfe_{horizon}h_r"][valid] = np.where(
            raw_horizon_peak >= usable_threshold_r, raw_horizon_peak, 0.0
        )[valid]
        summary[f"{prefix}mae_{horizon}h_r"][valid] = np.min(
            adverse[:, :horizon_bars], axis=1
        )[valid]

    def first_at_or_above(values: np.ndarray, threshold: np.ndarray) -> np.ndarray:
        result = np.full(rows, np.nan, dtype=np.float64)
        eligible = valid & np.isfinite(threshold)
        required = np.maximum(threshold, usable_threshold_r)
        hits = values >= required[:, None]
        found = eligible & hits.any(axis=1)
        result[found] = (np.argmax(hits[found], axis=1) + 1) * bar_hours
        return result

    def first_at_or_below(values: np.ndarray, threshold: np.ndarray) -> np.ndarray:
        result = np.full(rows, np.nan, dtype=np.float64)
        eligible = valid & np.isfinite(threshold)
        hits = values <= -np.abs(threshold[:, None])
        found = eligible & hits.any(axis=1)
        result[found] = (np.argmax(hits[found], axis=1) + 1) * bar_hours
        return result

    raw_peak_i = np.argmax(favorable, axis=1)
    trough_i = np.argmin(adverse, axis=1)
    row_i = np.arange(rows)
    half = max(1, bars // 2)
    early_peak_raw = np.max(favorable[:, :half], axis=1)
    late_peak_raw = (
        np.max(favorable[:, half:], axis=1) if half < bars else early_peak_raw
    )
    variation = np.abs(close_r[:, 0]) + np.abs(np.diff(close_r, axis=1)).sum(axis=1)
    moves = np.sign(np.diff(close_r, axis=1))
    reversals = np.zeros(rows, dtype=np.float64)
    previous_move = np.zeros(rows, dtype=np.float64)
    for column in range(moves.shape[1]):
        current_move = moves[:, column]
        nonzero = current_move != 0.0
        reversals += (
            nonzero & (previous_move != 0.0) & (current_move != previous_move)
        )
        previous_move = np.where(nonzero, current_move, previous_move)
    raw_peak = favorable[row_i, raw_peak_i]
    peak_is_usable = raw_peak >= usable_threshold_r
    peak = np.where(peak_is_usable, raw_peak, 0.0)
    risk_fraction = risk_distance / entry_price
    raw_peak_return = np.maximum(raw_peak * risk_fraction, 0.0)
    # The established v6 corpus used a fixed 1% total cost.  Exact-policy
    # historical replays may supply their realised per-row fee return instead;
    # keeping this opt-in preserves the frozen v6 behaviour for every existing
    # caller while allowing an explicitly signed execution target to retain its
    # own cost accounting.
    if archetype_cost_return is None:
        effective_archetype_cost = np.full(
            rows, CATBOOST_ARCHETYPE_COST_RETURN, dtype=np.float64
        )
    else:
        effective_archetype_cost = np.asarray(
            archetype_cost_return, dtype=np.float64
        ).reshape(-1)
        if len(effective_archetype_cost) != rows:
            raise ValueError("archetype_cost_return must align one-for-one with paths")
        effective_archetype_cost = np.where(
            np.isfinite(effective_archetype_cost) & (effective_archetype_cost > 0.0),
            effective_archetype_cost,
            CATBOOST_ARCHETYPE_COST_RETURN,
        )
    cost_atr = effective_archetype_cost / atr_fraction
    meaningful_mfe_threshold_atr = np.maximum(
        CATBOOST_ARCHETYPE_ATR_FLOOR,
        cost_atr + CATBOOST_ARCHETYPE_NET_MARGIN_ATR,
    )
    meaningful_mfe_threshold_r = (
        meaningful_mfe_threshold_atr * atr_fraction / risk_fraction
    )
    cost_aware_hits = favorable >= meaningful_mfe_threshold_r[:, None]
    cost_aware_reached = cost_aware_hits.any(axis=1)
    cost_aware_i = np.where(
        cost_aware_reached, np.argmax(cost_aware_hits, axis=1), bars - 1
    ).astype(np.int64)
    support_peak = np.maximum(raw_peak, 0.0)
    peak_80_hits = (support_peak > 0.0)[:, None] & (
        favorable >= (0.80 * support_peak)[:, None]
    )
    peak_90_hits_cost = (support_peak > 0.0)[:, None] & (
        favorable >= (0.90 * support_peak)[:, None]
    )
    peak_80_i = np.where(
        peak_80_hits.any(axis=1), np.argmax(peak_80_hits, axis=1), bars - 1
    ).astype(np.int64)
    peak_90_i_cost = np.where(
        peak_90_hits_cost.any(axis=1),
        np.argmax(peak_90_hits_cost, axis=1),
        bars - 1,
    ).astype(np.int64)
    stop_hits_cost = np.isfinite(stop_r)[:, None] & (
        adverse <= -np.abs(stop_r)[:, None]
    )
    stop_reached_cost = stop_hits_cost.any(axis=1)
    stop_i_cost = np.where(
        stop_reached_cost, np.argmax(stop_hits_cost, axis=1), bars - 1
    ).astype(np.int64)
    cumulative_adverse = np.minimum.accumulate(adverse, axis=1)
    cumulative_mfe = np.maximum.accumulate(favorable, axis=1)
    cumulative_below = np.cumsum(close_r < 0.0, axis=1)
    cumulative_adverse_area = np.cumsum(np.maximum(-close_r, 0.0), axis=1) * bar_hours
    cumulative_variation = np.cumsum(
        np.abs(np.diff(np.column_stack((np.zeros(rows), close_r)), axis=1)),
        axis=1,
    )

    def raw_horizon_peak(hours: float) -> np.ndarray:
        count = min(bars, max(1, int(round(hours / bar_hours))))
        return np.maximum(np.max(favorable[:, :count], axis=1), 0.0)

    def atr_slope(hours: float) -> np.ndarray:
        count = min(bars, max(2, int(round(hours / bar_hours))))
        x = (np.arange(count, dtype=np.float64) + 1.0) * bar_hours
        x -= x.mean()
        denominator = float(np.dot(x, x))
        if denominator <= 0.0:
            return np.zeros(rows, dtype=np.float64)
        y = close_r[:, :count] * risk_fraction[:, None] / atr_fraction[:, None]
        return np.dot(y, x) / denominator

    mfe_2h_raw = raw_horizon_peak(2.0)
    mfe_4h_raw = raw_horizon_peak(4.0)
    mfe_8h_raw = raw_horizon_peak(8.0)
    mfe_12h_raw = raw_horizon_peak(12.0)
    slope_4h = atr_slope(4.0)
    slope_12h = atr_slope(12.0)
    final_return_net = close_r[:, -1] * risk_fraction - effective_archetype_cost
    peak_net_return = raw_peak_return - effective_archetype_cost
    peak_mfe_atr = np.clip(raw_peak_return / atr_fraction, 0.0, 10.0)
    activation_r = np.where(
        np.isfinite(trailing_trigger_r) & (trailing_trigger_r > 0.0),
        trailing_trigger_r,
        np.where(
            np.isfinite(take_profit_r) & (take_profit_r > 0.0),
            take_profit_r,
            1.0,
        ),
    )
    effective_activation_return = np.where(
        np.isfinite(activation_distance_return)
        & (activation_distance_return > 0.0),
        activation_distance_return,
        activation_r * risk_fraction,
    )
    mfe_to_cost = np.divide(
        raw_peak_return,
        cost_return,
        out=np.full(rows, np.nan, dtype=np.float64),
        where=np.isfinite(cost_return) & (cost_return > 0.0),
    )
    mfe_to_activation = np.divide(
        raw_peak_return,
        effective_activation_return,
        out=np.full(rows, np.nan, dtype=np.float64),
        where=np.isfinite(effective_activation_return)
        & (effective_activation_return > 0.0),
    )
    meaningful_hits = favorable >= usable_threshold_r[:, None]
    meaningful_reached = meaningful_hits.any(axis=1)
    time_first_meaningful = np.where(
        meaningful_reached,
        (np.argmax(meaningful_hits, axis=1).astype(np.float64) + 1.0) * bar_hours,
        float(bars) * bar_hours,
    )
    peak_90_hits = favorable >= (0.90 * raw_peak)[:, None]
    time_peak_90 = np.where(
        peak_is_usable & peak_90_hits.any(axis=1),
        (np.argmax(peak_90_hits, axis=1).astype(np.float64) + 1.0) * bar_hours,
        float(bars) * bar_hours,
    )
    early_peak = np.where(
        early_peak_raw >= usable_threshold_r, early_peak_raw, 0.0
    )
    late_peak = np.where(
        late_peak_raw >= usable_threshold_r, late_peak_raw, 0.0
    )
    final = close_r[:, -1]
    efficiency = np.zeros(rows, dtype=np.float64)
    np.divide(final, variation, out=efficiency, where=variation != 0.0)

    summary.update(
        {
            f"{prefix}time_to_025r_h": first_at_or_above(
                favorable, np.full(rows, 0.25, dtype=np.float64)
            ),
            f"{prefix}time_to_05r_h": first_at_or_above(
                favorable, np.full(rows, 0.50, dtype=np.float64)
            ),
            f"{prefix}time_to_1r_h": first_at_or_above(
                favorable, np.full(rows, 1.0, dtype=np.float64)
            ),
            f"{prefix}time_to_tp_h": first_at_or_above(favorable, take_profit_r),
            f"{prefix}time_to_trailing_h": first_at_or_above(
                favorable, trailing_trigger_r
            ),
            f"{prefix}time_to_stop_h": first_at_or_below(adverse, stop_r),
            f"{prefix}mfe_before_mae": np.where(
                peak_is_usable, raw_peak_i < trough_i, np.nan
            ),
            f"{prefix}mae_before_mfe": np.where(
                peak_is_usable, trough_i < raw_peak_i, np.nan
            ),
            f"{prefix}time_to_first_meaningful_mfe_h": time_first_meaningful,
            f"{prefix}time_to_90pct_peak_mfe_h": time_peak_90,
            f"{prefix}usable_mfe_floor_return": usable_floor_return,
            f"{prefix}usable_mfe_threshold_r": usable_threshold_r,
            f"{prefix}raw_peak_mfe_r": raw_peak,
            f"{prefix}peak_mfe_r": peak,
            f"{prefix}peak_mfe_atr": peak_mfe_atr,
            f"{prefix}mfe_to_cost": mfe_to_cost,
            f"{prefix}mfe_to_activation_distance": mfe_to_activation,
            f"{prefix}early_late_ratio": early_peak
            / np.maximum(np.abs(late_peak), 1e-6),
            f"{prefix}efficiency": efficiency,
            f"{prefix}reversal_count": reversals,
            f"{prefix}final_return_r": final,
            f"{prefix}final_to_peak": np.where(
                peak_is_usable, final / np.maximum(np.abs(peak), 1e-6), np.nan
            ),
            f"{prefix}cost_atr": cost_atr,
            f"{prefix}meaningful_mfe_threshold_atr": meaningful_mfe_threshold_atr,
            f"{prefix}peak_mfe_minus_cost_atr": raw_peak_return / atr_fraction
            - cost_atr,
            f"{prefix}peak_mfe_div_cost": raw_peak_return
            / effective_archetype_cost,
            f"{prefix}reaches_meaningful_mfe": cost_aware_reached.astype(np.float64),
            f"{prefix}bars_to_meaningful_mfe": (cost_aware_i + 1).astype(np.float64),
            f"{prefix}bars_to_80pct_peak": (peak_80_i + 1).astype(np.float64),
            f"{prefix}bars_to_90pct_peak": (peak_90_i_cost + 1).astype(np.float64),
            f"{prefix}mfe_2h_over_mfe_12h": mfe_2h_raw
            / np.maximum(mfe_12h_raw, 1e-6),
            f"{prefix}mfe_4h_over_mfe_12h": mfe_4h_raw
            / np.maximum(mfe_12h_raw, 1e-6),
            f"{prefix}mfe_8h_over_mfe_12h": mfe_8h_raw
            / np.maximum(mfe_12h_raw, 1e-6),
            f"{prefix}bars_to_stop": (stop_i_cost + 1).astype(np.float64),
            f"{prefix}stop_before_meaningful_mfe": (
                stop_reached_cost
                & (~cost_aware_reached | (stop_i_cost < cost_aware_i))
            ).astype(np.float64),
            f"{prefix}mfe_before_stop_r": np.maximum(
                cumulative_mfe[row_i, stop_i_cost], 0.0
            ),
            f"{prefix}mae_before_meaningful_mfe_r": np.abs(
                np.minimum(cumulative_adverse[row_i, cost_aware_i], 0.0)
            ),
            f"{prefix}bars_below_entry_before_meaningful_mfe": cumulative_below[
                row_i, cost_aware_i
            ].astype(np.float64),
            f"{prefix}adverse_area_before_meaningful_mfe_r": cumulative_adverse_area[
                row_i, cost_aware_i
            ],
            f"{prefix}path_efficiency_to_meaningful_mfe": np.where(
                cost_aware_reached,
                np.clip(
                    meaningful_mfe_threshold_r
                    / np.maximum(
                        cumulative_variation[row_i, cost_aware_i], 1e-6
                    ),
                    0.0,
                    1.0,
                ),
                0.0,
            ),
            f"{prefix}path_efficiency_to_90pct_peak": np.where(
                support_peak > 0.0,
                np.clip(
                    0.90
                    * support_peak
                    / np.maximum(
                        cumulative_variation[row_i, peak_90_i_cost], 1e-6
                    ),
                    0.0,
                    1.0,
                ),
                0.0,
            ),
            f"{prefix}future_slope_atr_per_hour_4h": slope_4h,
            f"{prefix}future_slope_atr_per_hour_12h": slope_12h,
            f"{prefix}late_minus_early_slope": slope_12h - slope_4h,
            f"{prefix}final_return_net_1pct": final_return_net,
            f"{prefix}peak_retention_ratio": np.divide(
                final_return_net,
                peak_net_return,
                out=np.full(rows, np.nan, dtype=np.float64),
                where=peak_net_return > 1e-8,
            ).clip(-5.0, 2.0),
            f"{prefix}fraction_bars_above_50pct_peak": np.mean(
                close_r >= (0.50 * support_peak)[:, None], axis=1
            )
            * (support_peak > 0.0),
            f"{prefix}risk_fraction": risk_fraction,
            f"{prefix}atr_fraction": atr_fraction,
        }
    )
    for hour in range(1, 13):
        count = min(bars, max(1, int(round(hour / bar_hours))))
        raw_mfe_r = np.maximum(np.max(favorable[:, :count], axis=1), 0.0)
        summary[f"{prefix}raw_mfe_r_{hour}h"] = raw_mfe_r
        summary[f"{prefix}raw_mfe_atr_{hour}h"] = (
            raw_mfe_r * risk_fraction / atr_fraction
        )
        summary[f"{prefix}raw_mae_r_{hour}h"] = np.min(
            adverse[:, :count], axis=1
        )
        summary[f"{prefix}close_return_r_{hour}h"] = close_r[:, count - 1]
        summary[f"{prefix}cumulative_variation_r_{hour}h"] = cumulative_variation[
            :, count - 1
        ]
    for threshold in ATR_REALIZATION_THRESHOLDS:
        token = f"{int(round(threshold * 100)):03d}atr"
        threshold_return = threshold * atr_fraction
        threshold_r = threshold_return / risk_fraction
        hits = favorable >= threshold_r[:, None]
        found = valid & hits.any(axis=1)
        summary[f"{prefix}reached_{token}"] = found.astype(np.float64)
        time_to_threshold = np.full(rows, np.nan, dtype=np.float64)
        time_to_threshold[found] = (
            np.argmax(hits[found], axis=1) + 1
        ) * bar_hours
        summary[f"{prefix}time_to_{token}_h"] = time_to_threshold
    for values in summary.values():
        values[~valid] = np.nan
    return summary


def _deterministic_path_archetypes_batch(
    summary: Mapping[str, np.ndarray], *, prefix: str
) -> np.ndarray:
    """Apply the frozen path-shape rules in precedence order."""
    rows = len(next(iter(summary.values())))

    def values(name: str) -> np.ndarray:
        return summary.get(name, np.full(rows, np.nan, dtype=np.float64))

    peak = values(f"{prefix}peak_mfe_r")
    final = values(f"{prefix}final_return_r")
    mfe_4h = values(f"{prefix}mfe_4h_r")
    mfe_12h = values(f"{prefix}mfe_12h_r")
    mae_4h = values(f"{prefix}mae_4h_r")
    time_1r = values(f"{prefix}time_to_1r_h")
    time_stop = values(f"{prefix}time_to_stop_h")
    time_first = values(f"{prefix}time_to_first_meaningful_mfe_h")
    time_peak_90 = values(f"{prefix}time_to_90pct_peak_mfe_h")
    valid = np.isfinite(np.column_stack((peak, final, mfe_4h, mfe_12h, mae_4h))).all(
        axis=1
    )
    labels = np.full(rows, None, dtype=object)
    remaining = valid.copy()
    favorable_before_stop = np.isfinite(time_1r) & (
        ~np.isfinite(time_stop) | (time_1r <= time_stop)
    )

    immediate = remaining & np.isfinite(time_stop) & (time_stop <= 2.0) & (
        ~favorable_before_stop
    )
    labels[immediate] = "immediate_adverse_path"
    remaining &= ~immediate

    early_reversal = (
        remaining
        & (peak >= 0.75)
        & (time_peak_90 <= 8.0)
        & (final <= 0.0)
    )
    labels[early_reversal] = "early_mfe_full_reversal"
    remaining &= ~early_reversal

    fast_winner = remaining & (time_first <= 2.0) & (time_peak_90 <= 4.0) & (
        final > 0.0
    )
    labels[fast_winner & (mae_4h > -0.50)] = "fast_clean_winner"
    labels[fast_winner & ~(mae_4h > -0.50)] = "fast_winner_early_drawdown"
    remaining &= ~fast_winner

    late_breakout = remaining & (mfe_4h < 0.50) & (mfe_12h >= 1.0)
    labels[late_breakout] = "late_breakout"
    remaining &= ~late_breakout

    slow_grinder = remaining & (peak >= 0.50) & (final > 0.0)
    labels[slow_grinder] = "slow_grinder"
    remaining &= ~slow_grinder

    noisy_timeout = remaining & (peak >= 0.50)
    labels[noisy_timeout] = "noisy_timeout_usable_mfe"
    labels[remaining & ~noisy_timeout] = "dead_timeout"
    return labels


def _deterministic_realization_strength_batch(
    summary: Mapping[str, np.ndarray], *, prefix: str
) -> np.ndarray:
    peak_atr = np.asarray(summary[f"{prefix}peak_mfe_atr"], dtype=np.float64)
    labels = np.full(len(peak_atr), None, dtype=object)
    valid = np.isfinite(peak_atr)
    labels[valid & (peak_atr < 1.50)] = "below_150atr"
    labels[valid & (peak_atr >= 1.50) & (peak_atr < 2.00)] = "atr150_200"
    labels[valid & (peak_atr >= 2.00) & (peak_atr < 3.00)] = "atr200_300"
    labels[valid & (peak_atr >= 3.00) & (peak_atr < 5.00)] = "atr300_500"
    labels[valid & (peak_atr >= 5.00)] = "atr500_plus"
    return labels


def materialize_path_archetypes(
    candidates: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    config: PathArchetypeLabelConfig = PathArchetypeLabelConfig(),
) -> pd.DataFrame:
    """Attach causal 24-hour summaries and frozen target labels to candidates.

    Input timestamps are normalised as UTC.  A row is valid only when every
    expected canonical bar from ``signal + decision_delay`` through the maximum
    requested horizon is present, finite, and no earlier than the decision.
    """
    required_candidates = {config.timestamp_col, config.symbol_col}
    missing = sorted(required_candidates.difference(candidates.columns))
    if missing:
        raise ValueError(f"candidates missing required UTC key columns: {missing}")
    if config.side_col not in candidates:
        side_alt = _first_existing(candidates.columns, ("side_name", "__side__"))
        if side_alt is None:
            raise ValueError(f"candidates missing side column {config.side_col!r}")
        candidates = candidates.copy()
        candidates[config.side_col] = candidates[side_alt]
    required_bars = {
        config.bar_timestamp_col,
        config.bar_symbol_col,
        "high",
        "low",
        "close",
    }
    missing_bars = sorted(required_bars.difference(bars.columns))
    if missing_bars:
        raise ValueError(f"canonical bars missing required columns: {missing_bars}")

    out = candidates.copy()
    out[config.timestamp_col] = _utc(out[config.timestamp_col]).to_numpy()
    if out[config.timestamp_col].isna().any():
        raise ValueError("candidates contain invalid timestamps")
    out["__decision_ts__"] = out[config.timestamp_col] + pd.Timedelta(
        hours=config.decision_delay_hours
    )
    horizon_bars = int(round(max(config.horizons_hours) / config.bar_hours))
    out["__label_end_ts__"] = out["__decision_ts__"] + pd.Timedelta(
        hours=(horizon_bars - 1) * config.bar_hours
    )
    out["path_archetype_rule_version"] = config.rule_version
    if "discovery_cluster_id" in out:
        out["discovery_cluster_id"] = pd.to_numeric(
            out["discovery_cluster_id"], errors="coerce"
        ).astype("Int16")
    else:
        out["discovery_cluster_id"] = pd.Series(pd.NA, index=out.index, dtype="Int16")
    out["path_arch_complete_24h"] = np.zeros(len(out), dtype=np.int8)
    out["path_shape_archetype"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out["path_realization_strength"] = pd.Series(
        pd.NA, index=out.index, dtype="string"
    )
    out["path_archetype"] = pd.Series(pd.NA, index=out.index, dtype="string")

    bar_columns = [
        config.bar_timestamp_col,
        config.bar_symbol_col,
        "high",
        "low",
        "close",
    ]
    if "open" in bars:
        bar_columns.append("open")
    normal_bars = bars.loc[:, bar_columns].copy()
    normal_bars[config.bar_timestamp_col] = _utc(
        normal_bars[config.bar_timestamp_col]
    ).to_numpy()
    normal_bars = normal_bars.dropna(
        subset=[config.bar_timestamp_col, config.bar_symbol_col]
    )
    for column in ("high", "low", "close", "open"):
        if column not in normal_bars:
            continue
        normal_bars[column] = pd.to_numeric(
            normal_bars[column], errors="coerce"
        ).astype(np.float32)
    normal_bars = normal_bars.sort_values(
        [config.bar_symbol_col, config.bar_timestamp_col], kind="mergesort"
    )
    if normal_bars.duplicated([config.bar_symbol_col, config.bar_timestamp_col]).any():
        raise ValueError(
            "canonical bars contain duplicate exact UTC symbol/timestamp keys"
        )
    bar_groups: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]] = {}
    for symbol, group in normal_bars.groupby(config.bar_symbol_col, sort=False):
        bar_groups[str(symbol)] = (
            group[config.bar_timestamp_col].astype("int64").to_numpy(dtype=np.int64),
            group["high"].to_numpy(dtype=np.float32),
            group["low"].to_numpy(dtype=np.float32),
            group["close"].to_numpy(dtype=np.float32),
            group["open"].to_numpy(dtype=np.float32) if "open" in group else None,
        )
    (
        entry,
        risk,
        tp_r,
        trail_r,
        stop_r,
        barrier,
        atr_fraction,
        cost_return,
        activation_return,
    ) = _geometry_arrays(
        out,
        default_cost_return=config.default_cost_return,
        default_activation_r=config.default_activation_r,
    )
    signs = _side_sign(out[config.side_col])
    summary_values = {
        column: np.full(len(out), np.nan, dtype=np.float64)
        for column in _empty_summary(config.prefix, config.horizons_hours)
    }
    step_ns = int(pd.Timedelta(hours=config.bar_hours).value)
    offsets = np.arange(horizon_bars, dtype=np.int64) * step_ns
    path_offsets = np.arange(horizon_bars, dtype=np.int64)
    decision_ns = out["__decision_ts__"].astype("int64").to_numpy(dtype=np.int64)
    candidate_symbols = out[config.symbol_col].astype(str).to_numpy()
    # Keep the extracted 24-bar matrices bounded even for million-row inputs.
    batch_rows = 32_768
    for symbol in pd.unique(candidate_symbols):
        group = bar_groups.get(symbol)
        if group is None:
            continue
        times, high, low, close, open_ = group
        positions = np.flatnonzero(candidate_symbols == symbol)
        for batch_start in range(0, len(positions), batch_rows):
            batch_positions = positions[batch_start : batch_start + batch_rows]
            batch_decisions = decision_ns[batch_positions]
            starts = np.searchsorted(times, batch_decisions, side="left")
            in_bounds = starts + horizon_bars <= len(times)
            if not in_bounds.any():
                continue
            bounded_positions = batch_positions[in_bounds]
            bounded_decisions = batch_decisions[in_bounds]
            bounded_starts = starts[in_bounds]
            path_indices = bounded_starts[:, None] + path_offsets
            exact_path = np.all(
                times[path_indices] == bounded_decisions[:, None] + offsets, axis=1
            )
            if not exact_path.any():
                continue
            path_positions = bounded_positions[exact_path]
            path_indices = path_indices[exact_path]
            path_entry = entry[path_positions].copy()
            if open_ is not None:
                path_entry = np.where(
                    np.isfinite(path_entry), path_entry, open_[path_indices[:, 0]]
                )
            path_risk = risk[path_positions].copy()
            missing_risk = ~np.isfinite(path_risk) & np.isfinite(barrier[path_positions])
            path_risk[missing_risk] = np.abs(
                path_entry[missing_risk] * barrier[path_positions][missing_risk]
            )
            batch_summary = _summarize_side_relative_path_batch(
                high[path_indices],
                low[path_indices],
                close[path_indices],
                entry_price=path_entry,
                risk_distance=path_risk,
                atr_fraction=atr_fraction[path_positions],
                side_sign=signs[path_positions],
                bar_hours=config.bar_hours,
                horizons_hours=config.horizons_hours,
                take_profit_r=tp_r[path_positions],
                trailing_trigger_r=trail_r[path_positions],
                stop_r=stop_r[path_positions],
                cost_return=cost_return[path_positions],
                activation_distance_return=activation_return[path_positions],
                prefix=config.prefix,
            )
            for column, values in batch_summary.items():
                summary_values[column][path_positions] = values

    complete = np.isfinite(
        summary_values[f"{config.prefix}mfe_{max(config.horizons_hours)}h_r"]
    )
    out["path_arch_complete_24h"] = complete.astype(np.int8)
    shape = _deterministic_path_archetypes_batch(
        summary_values, prefix=config.prefix
    )
    strength = _deterministic_realization_strength_batch(
        summary_values, prefix=config.prefix
    )
    combined = np.full(len(out), None, dtype=object)
    combined_valid = pd.notna(shape) & pd.notna(strength)
    combined[combined_valid] = np.char.add(
        np.char.add(shape[combined_valid].astype(str), "__"),
        strength[combined_valid].astype(str),
    )
    out["path_shape_archetype"] = pd.array(shape, dtype="string")
    out["path_realization_strength"] = pd.array(strength, dtype="string")
    out["path_archetype"] = pd.array(combined, dtype="string")
    summary_frame = pd.DataFrame(summary_values, index=out.index).astype(np.float32)
    existing_summary = out.columns.intersection(summary_frame.columns)
    if len(existing_summary):
        out = out.drop(columns=list(existing_summary))
    return pd.concat([out, summary_frame], axis=1, copy=False)


def path_archetype_support_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Return deterministic type support plus economically useful 24-hour means."""
    required = {"path_archetype", "path_arch_complete_24h"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"materialized frame missing support fields: {missing}")
    valid = frame.loc[
        frame["path_arch_complete_24h"].astype(bool) & frame["path_archetype"].notna()
    ].copy()
    metrics = [
        "path_arch_peak_mfe_r",
        "path_arch_peak_mfe_atr",
        "path_arch_mfe_to_cost",
        "path_arch_mfe_to_activation_distance",
        "path_arch_time_to_first_meaningful_mfe_h",
        "path_arch_time_to_90pct_peak_mfe_h",
        "path_arch_mae_24h_r",
        "path_arch_final_return_r",
        "path_arch_efficiency",
        "path_arch_cost_atr",
        "path_arch_meaningful_mfe_threshold_atr",
        "path_arch_peak_mfe_minus_cost_atr",
        "path_arch_peak_mfe_div_cost",
        "path_arch_reaches_meaningful_mfe",
        "path_arch_bars_to_meaningful_mfe",
        "path_arch_bars_to_stop",
        "path_arch_stop_before_meaningful_mfe",
        "path_arch_mae_before_meaningful_mfe_r",
        "path_arch_adverse_area_before_meaningful_mfe_r",
        "path_arch_path_efficiency_to_meaningful_mfe",
        "path_arch_path_efficiency_to_90pct_peak",
        "path_arch_future_slope_atr_per_hour_4h",
        "path_arch_future_slope_atr_per_hour_12h",
        "path_arch_final_return_net_1pct",
        "path_arch_peak_retention_ratio",
        "path_arch_fraction_bars_above_50pct_peak",
        *[
            f"path_arch_reached_{int(round(threshold * 100)):03d}atr"
            for threshold in ATR_REALIZATION_THRESHOLDS
        ],
        *[
            f"path_arch_time_to_{int(round(threshold * 100)):03d}atr_h"
            for threshold in ATR_REALIZATION_THRESHOLDS
        ],
    ]
    metrics = [column for column in metrics if column in valid]
    grouped = valid.groupby("path_archetype", sort=True, observed=True)
    result = grouped.size().rename("rows").to_frame()
    result["support_fraction"] = result["rows"] / max(len(valid), 1)
    for metric in metrics:
        result[f"mean_{metric}"] = grouped[metric].mean()
    return result.reset_index()
