"""Causal sizing normalizers for exact one-minute portfolio replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from extreme_price_movements.simple_policy_1m_ablation import capacity_select


@dataclass(frozen=True)
class SelectedTrades:
    local_index: np.ndarray
    timestamp_ns: np.ndarray
    exit_ns: np.ndarray
    base_size: np.ndarray
    archetype: np.ndarray


def selected_trades(rows: pd.DataFrame, outputs: Mapping[str, np.ndarray], bar_minutes: int = 1) -> SelectedTrades:
    timestamps = pd.to_datetime(rows["timestamp"], utc=True).astype("int64").to_numpy(dtype=np.int64)
    symbols = pd.Categorical(rows["symbol"].astype(str)).codes.astype(np.int32)
    exits = np.asarray(outputs["exit_bars"], dtype=np.int32)
    finite = np.isfinite(np.asarray(outputs["net_return"], dtype=np.float64)) & (exits >= 0)
    chosen = np.flatnonzero(finite & capacity_select(timestamps, symbols, exits, int(bar_minutes)))
    rank = pd.to_numeric(rows.iloc[chosen]["rank_pct"], errors="coerce").fillna(0.9).to_numpy(dtype=np.float64)
    base = 0.075 + 0.075 * np.power(np.clip(rank, 0.0, 1.0), 1.1)
    minute_ns = int(bar_minutes) * 60 * 1_000_000_000
    archetype = rows.iloc[chosen]["policy_archetype"].astype(str).to_numpy()
    return SelectedTrades(
        local_index=chosen,
        timestamp_ns=timestamps[chosen],
        exit_ns=timestamps[chosen] + (exits[chosen].astype(np.int64) + 1) * minute_ns,
        base_size=base,
        archetype=archetype,
    )


def _weighted_mean(values: np.ndarray, weights: np.ndarray, fallback: float = 1.0) -> float:
    if not len(values) or not np.isfinite(values).any() or float(np.sum(weights)) <= 0.0:
        return float(fallback)
    return float(np.average(values, weights=weights))


def rolling_window_normalize(
    fit_rows: pd.DataFrame,
    fit_outputs: Mapping[str, np.ndarray],
    fit_size: np.ndarray,
    apply_rows: pd.DataFrame,
    apply_outputs: Mapping[str, np.ndarray],
    apply_size: np.ndarray,
    *,
    window_hours: float,
    lower: float = 0.50,
    upper: float = 1.50,
) -> np.ndarray:
    """Normalize by the base-size-weighted mean of strictly prior admitted entries."""
    fit = selected_trades(fit_rows, fit_outputs)
    apply = selected_trades(apply_rows, apply_outputs)
    result = np.asarray(apply_size, dtype=np.float64).copy()
    history_ts = list(fit.timestamp_ns.astype(np.int64))
    history_value = list(np.asarray(fit_size, dtype=np.float64)[fit.local_index])
    history_weight = list(fit.base_size)
    window_ns = int(float(window_hours) * 3_600 * 1_000_000_000)
    for timestamp in np.unique(apply.timestamp_ns):
        keep = [i for i, ts in enumerate(history_ts) if ts < timestamp and ts >= timestamp - window_ns]
        baseline = _weighted_mean(
            np.asarray([history_value[i] for i in keep]),
            np.asarray([history_weight[i] for i in keep]),
        )
        local = np.flatnonzero(apply.timestamp_ns == timestamp)
        target = apply.local_index[local]
        raw = np.asarray(apply_size, dtype=np.float64)[target]
        result[target] = np.clip(raw / max(baseline, 1e-9), lower, upper)
        for j in local:
            history_ts.append(int(timestamp))
            history_value.append(float(np.asarray(apply_size)[apply.local_index[j]]))
            history_weight.append(float(apply.base_size[j]))
    return result


def archetype_ewma_normalize(
    fit_rows: pd.DataFrame,
    fit_outputs: Mapping[str, np.ndarray],
    fit_size: np.ndarray,
    apply_rows: pd.DataFrame,
    apply_outputs: Mapping[str, np.ndarray],
    apply_size: np.ndarray,
    *,
    half_life_hours: float = 24.0,
    lower: float = 0.50,
    upper: float = 1.50,
) -> np.ndarray:
    """Normalize each archetype against its strictly prior causal EWMA opportunity level."""
    fit = selected_trades(fit_rows, fit_outputs)
    apply = selected_trades(apply_rows, apply_outputs)
    result = np.asarray(apply_size, dtype=np.float64).copy()
    numerator: dict[str, float] = {}
    denominator: dict[str, float] = {}
    last_ts: dict[str, int] = {}
    half_ns = max(float(half_life_hours) * 3_600 * 1_000_000_000, 1.0)

    def update(arch: str, timestamp: int, value: float, weight: float) -> None:
        previous = last_ts.get(arch, timestamp)
        decay = 0.5 ** (max(timestamp - previous, 0) / half_ns)
        numerator[arch] = numerator.get(arch, 0.0) * decay + value * weight
        denominator[arch] = denominator.get(arch, 0.0) * decay + weight
        last_ts[arch] = timestamp

    order = np.argsort(fit.timestamp_ns, kind="mergesort")
    for j in order:
        update(str(fit.archetype[j]), int(fit.timestamp_ns[j]), float(np.asarray(fit_size)[fit.local_index[j]]), float(fit.base_size[j]))
    global_baseline = _weighted_mean(np.asarray(fit_size)[fit.local_index], fit.base_size)
    for timestamp in np.unique(apply.timestamp_ns):
        local = np.flatnonzero(apply.timestamp_ns == timestamp)
        for j in local:
            arch = str(apply.archetype[j])
            baseline = numerator.get(arch, 0.0) / max(denominator.get(arch, 0.0), 1e-12) if arch in numerator else global_baseline
            target = apply.local_index[j]
            result[target] = np.clip(float(np.asarray(apply_size)[target]) / max(baseline, 1e-9), lower, upper)
        for j in local:
            target = apply.local_index[j]
            update(str(apply.archetype[j]), int(timestamp), float(np.asarray(apply_size)[target]), float(apply.base_size[j]))
    return result


def open_portfolio_budget_normalize(
    apply_rows: pd.DataFrame,
    apply_outputs: Mapping[str, np.ndarray],
    apply_size: np.ndarray,
    *,
    lower: float = 0.50,
    upper: float = 1.50,
) -> np.ndarray:
    """Scale new entries to the baseline open-notional budget across entry vintages."""
    selected = selected_trades(apply_rows, apply_outputs)
    result = np.asarray(apply_size, dtype=np.float64).copy()
    open_exit: list[int] = []
    open_actual: list[float] = []
    open_baseline: list[float] = []
    for timestamp in np.unique(selected.timestamp_ns):
        keep = [i for i, exit_ts in enumerate(open_exit) if exit_ts > timestamp]
        open_exit = [open_exit[i] for i in keep]
        open_actual = [open_actual[i] for i in keep]
        open_baseline = [open_baseline[i] for i in keep]
        local = np.flatnonzero(selected.timestamp_ns == timestamp)
        target = selected.local_index[local]
        base = selected.base_size[local]
        raw_notional = base * np.asarray(apply_size, dtype=np.float64)[target]
        target_after = float(np.sum(open_baseline) + np.sum(base))
        available = max(target_after - float(np.sum(open_actual)), 0.0)
        scale = available / max(float(np.sum(raw_notional)), 1e-12)
        adjusted = np.clip(np.asarray(apply_size, dtype=np.float64)[target] * scale, lower, upper)
        result[target] = adjusted
        for j, row_index in enumerate(target):
            open_exit.append(int(selected.exit_ns[local[j]]))
            open_actual.append(float(base[j] * adjusted[j]))
            open_baseline.append(float(base[j]))
    return result


def bounded_dynamic_exposure_normalize(
    apply_rows: pd.DataFrame,
    apply_outputs: Mapping[str, np.ndarray],
    apply_size: np.ndarray,
    *,
    exposure_band: float,
    lower: float = 0.50,
    upper: float = 1.50,
) -> np.ndarray:
    """Keep current cross-sectional allocation while bounding each entry cohort's total exposure."""
    selected = selected_trades(apply_rows, apply_outputs)
    result = np.asarray(apply_size, dtype=np.float64).copy()
    band = abs(float(exposure_band))
    for timestamp in np.unique(selected.timestamp_ns):
        local = np.flatnonzero(selected.timestamp_ns == timestamp)
        target = selected.local_index[local]
        raw = np.asarray(apply_size, dtype=np.float64)[target]
        raw_mean = _weighted_mean(raw, selected.base_size[local])
        cohort_target = float(np.clip(raw_mean, 1.0 - band, 1.0 + band))
        adjusted = raw * cohort_target / max(raw_mean, 1e-9)
        result[target] = np.clip(adjusted, lower, upper)
    return result
