"""Fold-local per-strategy performance labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyPerformanceLabels:
    strategy: str
    timestamps: pd.Index
    strategy_performance: pd.Series
    ewma_performance: pd.Series
    bad_label: pd.Series
    good_label: pd.Series
    bad_sample_weight: pd.Series
    good_sample_weight: pd.Series
    loss_streak_hours: pd.Series
    loss_streak_bad_pressure: pd.Series
    loss_density_bad_pressure: pd.Series
    drawdown_bad_pressure: pd.Series
    utility_bad_pressure: pd.Series
    forward_bad_pressure: pd.Series
    cooldown_bad_pressure: pd.Series
    composite_bad_pressure: pd.Series
    anchors: dict[str, float]


@dataclass(frozen=True)
class StrategyPerformanceLabelBundle:
    by_strategy: dict[str, StrategyPerformanceLabels]
    diagnostics: pd.DataFrame


def bad_label_from_perf(
    x: float,
    worst: float,
    median: float,
    best: float,
    eps: float = 1e-12,
) -> float:
    """Map fold-local EWMA performance to a bad-performance soft label."""

    value = float(x)
    if not np.isfinite(value):
        value = float(median) if np.isfinite(float(median)) else 0.0
    if value <= median:
        y = 0.5 + 0.5 * (median - value) / max(median - worst, eps)
    else:
        y = 0.5 - 0.5 * (value - median) / max(best - median, eps)
    return float(np.clip(y, 0.0, 1.0))


def _sample_weight(label: pd.Series) -> pd.Series:
    distance = (pd.to_numeric(label, errors="coerce").fillna(0.5) - 0.5).abs()
    return ((1.0 + 2.0 * distance) ** 2).astype(float)


def _median_step_hours(index: pd.Index) -> float:
    if isinstance(index, pd.DatetimeIndex) and len(index) > 1:
        diffs = index.sort_values().to_series().diff().dropna().dt.total_seconds().to_numpy(dtype=float)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        if len(diffs):
            return max(float(np.nanmedian(diffs) / 3600.0), 1e-9)
    return 1.0


def _loss_streak_hours(performance: pd.Series) -> pd.Series:
    values = pd.to_numeric(performance, errors="coerce")
    step_hours = _median_step_hours(performance.index)
    out = np.zeros(len(values), dtype=np.float32)
    current_count = 0
    current_start = None
    datetime_index = isinstance(performance.index, pd.DatetimeIndex)
    for pos, (timestamp, value) in enumerate(values.items()):
        if pd.isna(value):
            if current_start is not None:
                if datetime_index:
                    out[pos] = max(
                        float((timestamp - current_start).total_seconds() / 3600.0 + step_hours),
                        step_hours,
                    )
                else:
                    out[pos] = float(current_count * step_hours)
            continue
        if float(value) < 0.0:
            if current_start is None:
                current_start = timestamp
                current_count = 0
            current_count += 1
            if datetime_index:
                out[pos] = max(
                    float((timestamp - current_start).total_seconds() / 3600.0 + step_hours),
                    step_hours,
                )
            else:
                out[pos] = float(current_count * step_hours)
        else:
            current_start = None
            current_count = 0
            out[pos] = 0.0
    return pd.Series(out, index=performance.index, dtype=float)


def _loss_streak_bad_pressure(
    streak_hours: pd.Series,
    *,
    min_hours: float | None,
    full_hours: float | None,
) -> pd.Series:
    if min_hours is None or float(min_hours) <= 0.0:
        return pd.Series(0.0, index=streak_hours.index, dtype=float)
    start = float(min_hours)
    end = float(full_hours) if full_hours is not None else start * 2.0
    end = max(end, start + 1e-9)
    pressure = ((pd.to_numeric(streak_hours, errors="coerce").fillna(0.0) - start) / (end - start)).clip(0.0, 1.0)
    return pressure.astype(float)


def _window_bars(index: pd.Index, window_hours: float) -> int:
    return max(1, int(np.ceil(float(window_hours) / _median_step_hours(index))))


def _max_pressure(series_list: list[pd.Series], index: pd.Index) -> pd.Series:
    if not series_list:
        return pd.Series(0.0, index=index, dtype=float)
    frame = pd.concat([s.reindex(index).fillna(0.0) for s in series_list], axis=1)
    return frame.max(axis=1).clip(0.0, 1.0).astype(float)


def _rolling_loss_density_pressure(
    performance: pd.Series,
    *,
    windows_hours: Sequence[float],
    min_negative_share: float,
    full_negative_share: float,
) -> pd.Series:
    values = pd.to_numeric(performance, errors="coerce")
    observed = values.notna().astype(float)
    negative = values.lt(0.0).fillna(False).astype(float)
    pressures: list[pd.Series] = []
    start = float(np.clip(min_negative_share, 0.0, 1.0))
    end = float(np.clip(full_negative_share, start + 1e-9, 1.0))
    for window in windows_hours:
        bars = _window_bars(values.index, float(window))
        observed_count = observed.rolling(bars, min_periods=1).sum()
        negative_count = negative.rolling(bars, min_periods=1).sum()
        share = (negative_count / observed_count.replace(0.0, np.nan)).fillna(0.0)
        pressures.append(((share - start) / max(end - start, 1e-9)).clip(0.0, 1.0))
    return _max_pressure(pressures, values.index)


def _rolling_drawdown_pressure(
    performance: pd.Series,
    *,
    windows_hours: Sequence[float],
    drawdown_anchor_quantile: float,
) -> pd.Series:
    values = pd.to_numeric(performance, errors="coerce").fillna(0.0).astype(float)
    cumulative = values.cumsum()
    drawdown = (cumulative.cummax() - cumulative).clip(lower=0.0)
    positive_drawdown = drawdown.loc[drawdown.gt(0.0)]
    if positive_drawdown.empty:
        drawdown_pressure = pd.Series(0.0, index=values.index, dtype=float)
    else:
        anchor = float(positive_drawdown.quantile(float(np.clip(drawdown_anchor_quantile, 0.01, 1.0))))
        drawdown_pressure = (drawdown / max(anchor, 1e-12)).clip(0.0, 1.0)
    pressures = [drawdown_pressure]
    for window in windows_hours:
        bars = _window_bars(values.index, float(window))
        rolling_return = values.rolling(bars, min_periods=1).sum()
        negative_returns = rolling_return.loc[rolling_return.lt(0.0)]
        if negative_returns.empty:
            pressures.append(pd.Series(0.0, index=values.index, dtype=float))
            continue
        anchor = abs(float(negative_returns.quantile(0.10)))
        pressures.append((-rolling_return / max(anchor, 1e-12)).clip(0.0, 1.0))
    return _max_pressure(pressures, values.index)


def _rolling_utility_bad_pressure(
    performance: pd.Series,
    *,
    windows_hours: Sequence[float],
    z_score: float,
) -> pd.Series:
    values = pd.to_numeric(performance, errors="coerce")
    observed = values.notna().astype(float)
    filled = values.fillna(0.0).astype(float)
    pressures: list[pd.Series] = []
    z = max(float(z_score), 0.0)
    for window in windows_hours:
        bars = _window_bars(values.index, float(window))
        count = observed.rolling(bars, min_periods=1).sum()
        mean = filled.rolling(bars, min_periods=1).sum() / count.replace(0.0, np.nan)
        std = values.rolling(bars, min_periods=min(2, bars)).std(ddof=0).fillna(0.0)
        lcb = (mean - z * std / np.sqrt(count.replace(0.0, np.nan))).fillna(0.0)
        negative_lcb = lcb.loc[lcb.lt(0.0)]
        if negative_lcb.empty:
            pressures.append(pd.Series(0.0, index=values.index, dtype=float))
            continue
        anchor = abs(float(negative_lcb.quantile(0.10)))
        pressures.append((-lcb / max(anchor, 1e-12)).clip(0.0, 1.0))
    return _max_pressure(pressures, values.index)


def _forward_bad_pressure(
    performance: pd.Series,
    *,
    window_hours: float,
    min_negative_share: float,
    full_negative_share: float,
) -> pd.Series:
    values = pd.to_numeric(performance, errors="coerce")
    observed = values.notna().astype(float)
    filled = values.fillna(0.0).astype(float)
    negative = values.lt(0.0).fillna(False).astype(float)
    bars = _window_bars(values.index, float(window_hours))
    future_observed = observed.shift(-1).iloc[::-1].rolling(bars, min_periods=1).sum().iloc[::-1]
    future_negative = negative.shift(-1).iloc[::-1].rolling(bars, min_periods=1).sum().iloc[::-1]
    future_return = filled.shift(-1).iloc[::-1].rolling(bars, min_periods=1).sum().iloc[::-1]
    start = float(np.clip(min_negative_share, 0.0, 1.0))
    end = float(np.clip(full_negative_share, start + 1e-9, 1.0))
    negative_share = (future_negative / future_observed.replace(0.0, np.nan)).fillna(0.0)
    density_pressure = ((negative_share - start) / max(end - start, 1e-9)).clip(0.0, 1.0)
    negative_future = future_return.loc[future_return.lt(0.0)]
    if negative_future.empty:
        return density_pressure.astype(float)
    anchor = abs(float(negative_future.quantile(0.10)))
    return _max_pressure(
        [density_pressure, (-future_return / max(anchor, 1e-12)).clip(0.0, 1.0)],
        values.index,
    )


def _cooldown_bad_pressure(
    risk_pressure: pd.Series,
    *,
    cooldown_hours: float,
    trigger: float,
) -> pd.Series:
    if float(cooldown_hours) <= 0.0:
        return pd.Series(0.0, index=risk_pressure.index, dtype=float)
    bars = _window_bars(risk_pressure.index, float(cooldown_hours))
    trigger_value = float(np.clip(trigger, 0.0, 1.0))
    values = pd.to_numeric(risk_pressure, errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
    out = np.zeros(len(values), dtype=np.float32)
    remaining = 0
    for i, value in enumerate(values):
        if value >= trigger_value:
            remaining = bars
        if remaining > 0:
            out[i] = max(out[i], remaining / float(bars))
            remaining -= 1
    return pd.Series(out, index=risk_pressure.index, dtype=float)


def _normalise_modes(values: Sequence[str] | None) -> set[str]:
    modes: set[str] = set()
    for value in values or ():
        for part in str(value).split(","):
            text = part.strip().lower()
            if text:
                modes.add(text)
    return modes


def _strategy_modes(
    strategy: str,
    *,
    global_modes: Sequence[str] | None,
    strategy_mode_map: Mapping[str, Sequence[str]] | None,
) -> tuple[set[str], bool]:
    if strategy_mode_map and str(strategy) in strategy_mode_map:
        return _normalise_modes(strategy_mode_map[str(strategy)]), True
    modes = _normalise_modes(global_modes)
    return modes, bool(modes)


def _mode_enabled(mode: str, *, modes: set[str], explicit_profile: bool, weight: float) -> bool:
    if float(weight) <= 0.0:
        return False
    if explicit_profile:
        return mode in modes
    return True if not modes else mode in modes


def _blend_bad_label_with_pressure(
    bad: pd.Series,
    pressure: pd.Series,
    *,
    weight: float,
) -> pd.Series:
    clipped_weight = float(np.clip(weight, 0.0, 1.0))
    if clipped_weight <= 0.0 or not bool(pressure.gt(0.0).any()):
        return bad.astype(float)
    pressure_bad = (0.5 + 0.5 * pressure.reindex(bad.index).fillna(0.0)).clip(0.5, 1.0)
    blended = (1.0 - clipped_weight) * bad + clipped_weight * np.maximum(bad, pressure_bad)
    return bad.where(pressure <= 0.0, blended).clip(0.0, 1.0).astype(float)


def _normalise_timestamps(values: pd.Series | pd.Index) -> pd.DatetimeIndex | pd.Index:
    ts = pd.to_datetime(values, utc=True, errors="coerce")
    if pd.notna(ts).all():
        return pd.DatetimeIndex(ts)
    return pd.Index(values)


def causal_ewma(
    series: pd.Series,
    *,
    halflife: str | int = "3D",
) -> pd.Series:
    """Causal EWMA over a timestamp-sorted series.

    Pandas' time-aware EWMA is causal but uses ``adjust=True``.  That is fine
    here because no future samples enter the mean.
    """

    ordered = pd.to_numeric(series, errors="coerce").astype(float).sort_index()
    ordered = ordered.fillna(0.0)
    if isinstance(halflife, str):
        try:
            index = pd.DatetimeIndex(pd.to_datetime(ordered.index, utc=True))
            return ordered.ewm(halflife=pd.Timedelta(halflife), times=index).mean()
        except Exception:
            pass
    return ordered.ewm(halflife=max(float(halflife), 1e-9), adjust=True).mean()


def _aggregate_strategy_performance(
    frame: pd.DataFrame,
    *,
    strategy: str,
    strategy_col: str,
    timestamp_col: str,
    performance_col: str,
    timestamps: pd.Index,
    fill_missing: bool = True,
) -> pd.Series:
    rows = frame.loc[frame[strategy_col].astype(str) == str(strategy)]
    if rows.empty:
        fill_value = 0.0 if fill_missing else np.nan
        return pd.Series(fill_value, index=timestamps, dtype=float)
    grouped = (
        rows.assign(
            __timestamp__=_normalise_timestamps(rows[timestamp_col]),
            __performance__=pd.to_numeric(rows[performance_col], errors="coerce"),
        )
        .groupby("__timestamp__", sort=True)["__performance__"]
        .mean()
    )
    out = grouped.reindex(timestamps)
    if fill_missing:
        out = out.fillna(0.0)
    return out.astype(float)


def _anchors(
    ewma: pd.Series,
    *,
    mode: Literal["winsorized", "minmax"],
    lower_q: float,
    upper_q: float,
) -> dict[str, float]:
    values = pd.to_numeric(ewma, errors="coerce").replace([np.inf, -np.inf], np.nan)
    values = values.dropna()
    if values.empty:
        return {"worst": 0.0, "median": 0.0, "best": 0.0}
    if mode == "winsorized":
        worst = float(values.quantile(float(lower_q)))
        median = float(values.quantile(0.50))
        best = float(values.quantile(float(upper_q)))
    else:
        worst = float(values.min())
        median = float(values.median())
        best = float(values.max())
    return {"worst": worst, "median": median, "best": best}


def build_strategy_performance_labels(
    trades_or_signals: pd.DataFrame,
    *,
    strategy_col: str,
    timestamp_col: str,
    performance_col: str,
    strategies: Sequence[str],
    ewma_halflife: str | int = "3D",
    anchor_mode: Literal["winsorized", "minmax"] = "winsorized",
    lower_anchor_quantile: float = 0.01,
    upper_anchor_quantile: float = 0.99,
    loss_streak_target_min_hours: float | None = None,
    loss_streak_target_full_hours: float | None = None,
    loss_streak_label_weight: float = 1.0,
    loss_streak_sample_weight_multiplier: float = 2.0,
    risk_label_modes: Sequence[str] | None = None,
    strategy_risk_label_modes: Mapping[str, Sequence[str]] | None = None,
    rolling_bad_regime_windows_hours: Sequence[float] = (24.0, 72.0, 168.0),
    loss_density_label_weight: float = 0.0,
    loss_density_min_negative_share: float = 0.55,
    loss_density_full_negative_share: float = 0.80,
    drawdown_label_weight: float = 0.0,
    drawdown_anchor_quantile: float = 0.90,
    utility_label_weight: float = 0.0,
    utility_lcb_z_score: float = 1.0,
    forward_bad_label_weight: float = 0.0,
    forward_bad_window_hours: float = 72.0,
    cooldown_label_weight: float = 0.0,
    cooldown_hours: float = 24.0,
    cooldown_trigger: float = 0.75,
) -> StrategyPerformanceLabelBundle:
    """Build timestamp-level soft bad/good labels inside one training fold."""

    required = {strategy_col, timestamp_col, performance_col}
    missing = sorted(required.difference(trades_or_signals.columns))
    if missing:
        raise KeyError(f"Missing required label columns: {missing}")
    frame = trades_or_signals.copy()
    timestamps = pd.Index(_normalise_timestamps(frame[timestamp_col])).dropna()
    timestamps = pd.Index(sorted(pd.unique(timestamps)))
    by_strategy: dict[str, StrategyPerformanceLabels] = {}
    diag_rows: list[dict[str, object]] = []
    for strategy in [str(s) for s in strategies]:
        perf = _aggregate_strategy_performance(
            frame,
            strategy=strategy,
            strategy_col=strategy_col,
            timestamp_col=timestamp_col,
            performance_col=performance_col,
            timestamps=timestamps,
        )
        observed_perf = _aggregate_strategy_performance(
            frame,
            strategy=strategy,
            strategy_col=strategy_col,
            timestamp_col=timestamp_col,
            performance_col=performance_col,
            timestamps=timestamps,
            fill_missing=False,
        )
        ewma = causal_ewma(perf, halflife=ewma_halflife).reindex(timestamps)
        anchors = _anchors(
            ewma,
            mode=anchor_mode,
            lower_q=lower_anchor_quantile,
            upper_q=upper_anchor_quantile,
        )
        bad = ewma.map(
            lambda value: bad_label_from_perf(
                value,
                anchors["worst"],
                anchors["median"],
                anchors["best"],
            )
        ).astype(float)
        modes, explicit_profile = _strategy_modes(
            strategy,
            global_modes=risk_label_modes,
            strategy_mode_map=strategy_risk_label_modes,
        )
        streak_hours = _loss_streak_hours(observed_perf)
        streak_pressure = _loss_streak_bad_pressure(
            streak_hours,
            min_hours=loss_streak_target_min_hours,
            full_hours=loss_streak_target_full_hours,
        )
        density_pressure = _rolling_loss_density_pressure(
            observed_perf,
            windows_hours=rolling_bad_regime_windows_hours,
            min_negative_share=loss_density_min_negative_share,
            full_negative_share=loss_density_full_negative_share,
        )
        drawdown_pressure = _rolling_drawdown_pressure(
            perf,
            windows_hours=rolling_bad_regime_windows_hours,
            drawdown_anchor_quantile=drawdown_anchor_quantile,
        )
        utility_pressure = _rolling_utility_bad_pressure(
            observed_perf,
            windows_hours=rolling_bad_regime_windows_hours,
            z_score=utility_lcb_z_score,
        )
        forward_pressure = _forward_bad_pressure(
            observed_perf,
            window_hours=forward_bad_window_hours,
            min_negative_share=loss_density_min_negative_share,
            full_negative_share=loss_density_full_negative_share,
        )
        enabled_pressures: list[pd.Series] = []
        pressure_weights: list[tuple[pd.Series, float, str]] = [
            (streak_pressure, loss_streak_label_weight, "streak"),
            (density_pressure, loss_density_label_weight, "density"),
            (drawdown_pressure, drawdown_label_weight, "drawdown"),
            (utility_pressure, utility_label_weight, "utility"),
            (forward_pressure, forward_bad_label_weight, "forward"),
        ]
        for pressure, weight, mode in pressure_weights:
            if _mode_enabled(mode, modes=modes, explicit_profile=explicit_profile, weight=float(weight)):
                bad = _blend_bad_label_with_pressure(bad, pressure, weight=float(weight))
                enabled_pressures.append(pressure)
        pre_cooldown_composite = _max_pressure(enabled_pressures, timestamps)
        cooldown_pressure = _cooldown_bad_pressure(
            pre_cooldown_composite,
            cooldown_hours=cooldown_hours,
            trigger=cooldown_trigger,
        )
        if _mode_enabled("cooldown", modes=modes, explicit_profile=explicit_profile, weight=float(cooldown_label_weight)):
            bad = _blend_bad_label_with_pressure(bad, cooldown_pressure, weight=float(cooldown_label_weight))
            enabled_pressures.append(cooldown_pressure)
        composite_pressure = _max_pressure(enabled_pressures, timestamps)
        good = (1.0 - bad).astype(float)
        bad_weight = _sample_weight(bad)
        good_weight = _sample_weight(good)
        if float(loss_streak_sample_weight_multiplier) > 0.0 and bool(composite_pressure.gt(0.0).any()):
            risk_multiplier = (1.0 + float(loss_streak_sample_weight_multiplier) * composite_pressure).clip(lower=1.0)
            bad_weight = (bad_weight * risk_multiplier).astype(float)
            good_weight = (good_weight * risk_multiplier).astype(float)
        by_strategy[strategy] = StrategyPerformanceLabels(
            strategy=strategy,
            timestamps=timestamps,
            strategy_performance=perf,
            ewma_performance=ewma,
            bad_label=bad,
            good_label=good,
            bad_sample_weight=bad_weight,
            good_sample_weight=good_weight,
            loss_streak_hours=streak_hours,
            loss_streak_bad_pressure=streak_pressure,
            loss_density_bad_pressure=density_pressure,
            drawdown_bad_pressure=drawdown_pressure,
            utility_bad_pressure=utility_pressure,
            forward_bad_pressure=forward_pressure,
            cooldown_bad_pressure=cooldown_pressure,
            composite_bad_pressure=composite_pressure,
            anchors=dict(anchors),
        )
        diag_rows.append(
            {
                "strategy": strategy,
                "timestamp_count": int(len(timestamps)),
                "observed_row_count": int(
                    (frame[strategy_col].astype(str) == strategy).sum()
                ),
                "anchor_mode": str(anchor_mode),
                "worst_anchor": float(anchors["worst"]),
                "median_anchor": float(anchors["median"]),
                "best_anchor": float(anchors["best"]),
                "ewma_halflife": str(ewma_halflife),
                "loss_streak_target_min_hours": float(loss_streak_target_min_hours or 0.0),
                "loss_streak_target_full_hours": float(loss_streak_target_full_hours or 0.0),
                "risk_label_modes": ",".join(sorted(modes)),
                "max_loss_streak_hours": float(streak_hours.max()) if len(streak_hours) else 0.0,
                "loss_streak_pressure_share": float(streak_pressure.gt(0.0).mean())
                if len(streak_pressure)
                else 0.0,
                "max_loss_streak_bad_pressure": float(streak_pressure.max()) if len(streak_pressure) else 0.0,
                "loss_density_pressure_share": float(density_pressure.gt(0.0).mean())
                if len(density_pressure)
                else 0.0,
                "drawdown_pressure_share": float(drawdown_pressure.gt(0.0).mean())
                if len(drawdown_pressure)
                else 0.0,
                "utility_pressure_share": float(utility_pressure.gt(0.0).mean())
                if len(utility_pressure)
                else 0.0,
                "forward_bad_pressure_share": float(forward_pressure.gt(0.0).mean())
                if len(forward_pressure)
                else 0.0,
                "cooldown_pressure_share": float(cooldown_pressure.gt(0.0).mean())
                if len(cooldown_pressure)
                else 0.0,
                "composite_bad_pressure_share": float(composite_pressure.gt(0.0).mean())
                if len(composite_pressure)
                else 0.0,
                "max_composite_bad_pressure": float(composite_pressure.max())
                if len(composite_pressure)
                else 0.0,
            }
        )
    return StrategyPerformanceLabelBundle(
        by_strategy=by_strategy,
        diagnostics=pd.DataFrame(diag_rows),
    )
