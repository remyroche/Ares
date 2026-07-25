"""Soft-binary auxiliary and execution-EV ablation primitives.

The functions in this module are deliberately small and deterministic.  They
turn continuous path outcomes into policy-anchored soft labels and define the
strict expanding June/July evaluation calendar used by the exact-policy
ablation runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

HEADS: tuple[str, ...] = (
    "peak_mfe_12h_atr",
    "time_to_first_meaningful_mfe",
    "mae_before_meaningful_mfe_atr",
    "bars_before_price_stops_decreasing",
    "future_slope_atr_per_hour",
)


@dataclass(frozen=True)
class SoftLabelParameters:
    peak_temperature_atr: float = 0.25
    timing_midpoint_hours: float = 4.0
    timing_temperature_hours: float = 1.5
    mae_midpoint_atr: float = 0.50
    mae_temperature_atr: float = 0.15
    turn_midpoint_bars: float = 4.0
    turn_temperature_bars: float = 1.5
    slope_temperature_atr_per_hour: float = 0.15


def sigmoid(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Numerically stable logistic transform."""

    array = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(array, -40.0, 40.0)))


def auxiliary_soft_targets(
    frame: pd.DataFrame,
    params: SoftLabelParameters = SoftLabelParameters(),
) -> pd.DataFrame:
    """Build one economically oriented soft-binary target per auxiliary head.

    Peak uses the larger of 1.5 ATR and the 1% cost hurdle.  Timing rewards a
    meaningful MFE before four hours.  MAE rewards a clean pre-MFE path.
    Adverse-turn timing rewards an early confirmed trough, but discounts paths
    that suffer a large pre-MFE MAE.  Slope rewards positive favorable drift.
    """

    required = {
        "__path_auxiliary_atr_fraction__",
        "__peak_mfe_atr_12h__",
        "__time_to_first_meaningful_mfe_hours_12h__",
        "__mae_before_meaningful_mfe_atr_12h__",
        "__bars_to_confirmed_adverse_trough__",
        "__future_slope_atr_per_hour_12h__",
        "__meaningful_mfe_reached_12h__",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            "soft auxiliary targets missing columns: " + ", ".join(missing)
        )

    def numeric(column: str, *, fill: float | None = None) -> np.ndarray:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
        if fill is not None:
            values = np.nan_to_num(values, nan=float(fill))
        if not np.isfinite(values).all():
            raise ValueError(f"soft auxiliary target input {column!r} must be finite")
        return values

    atr_fraction = np.maximum(numeric("__path_auxiliary_atr_fraction__"), 1e-6)
    peak = np.maximum(numeric("__peak_mfe_atr_12h__"), 0.0)
    time = np.clip(numeric("__time_to_first_meaningful_mfe_hours_12h__"), 0.0, 12.0)
    mae = np.maximum(numeric("__mae_before_meaningful_mfe_atr_12h__"), 0.0)
    turn = np.clip(
        numeric("__bars_to_confirmed_adverse_trough__", fill=12.0), 0.0, 12.0
    )
    slope = numeric("__future_slope_atr_per_hour_12h__")
    reached = np.clip(numeric("__meaningful_mfe_reached_12h__"), 0.0, 1.0)

    hurdle = np.maximum(1.5, 0.01 / atr_fraction)
    peak_soft = sigmoid((peak - hurdle) / params.peak_temperature_atr)
    timing_soft = reached * sigmoid(
        (params.timing_midpoint_hours - time) / params.timing_temperature_hours
    )
    mae_soft = sigmoid((params.mae_midpoint_atr - mae) / params.mae_temperature_atr)
    turn_early = sigmoid(
        (params.turn_midpoint_bars - turn) / params.turn_temperature_bars
    )
    # A quick trough after a very deep adverse excursion is not economically
    # clean.  Multiplying by the MAE label prevents that failure mode.
    turn_soft = turn_early * mae_soft
    slope_soft = sigmoid(slope / params.slope_temperature_atr_per_hour)
    return pd.DataFrame(
        {
            "peak_mfe_12h_atr": peak_soft,
            "time_to_first_meaningful_mfe": timing_soft,
            "mae_before_meaningful_mfe_atr": mae_soft,
            "bars_before_price_stops_decreasing": turn_soft,
            "future_slope_atr_per_hour": slope_soft,
        },
        index=frame.index,
        dtype=np.float32,
    )


def execution_ev_soft_target(
    net_return: Sequence[float] | np.ndarray,
    *,
    threshold: float,
    temperature: float,
) -> np.ndarray:
    """Convert exact-policy net return to a soft probability-like utility."""

    if temperature <= 0.0:
        raise ValueError("execution-EV soft-label temperature must be positive")
    values = np.asarray(net_return, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("execution-EV net returns must be finite")
    return sigmoid((values - float(threshold)) / float(temperature)).astype(np.float32)


def expanding_month_folds(
    timestamps: Sequence[object] | pd.Series,
    *,
    validation_months: Sequence[str] = ("2026-06", "2026-07"),
    purge_hours: float = 25.0,
) -> list[dict[str, object]]:
    """Return strict train-before-validation folds with a fixed purge."""

    ts = pd.to_datetime(pd.Series(timestamps), utc=True, errors="coerce")
    if ts.isna().any():
        raise ValueError("fold timestamps must be valid UTC")
    folds: list[dict[str, object]] = []
    for fold, month in enumerate(validation_months):
        period = pd.Period(month, freq="M")
        start = pd.Timestamp(period.start_time, tz="UTC")
        end = pd.Timestamp((period + 1).start_time, tz="UTC")
        cutoff = start - pd.Timedelta(hours=float(purge_hours))
        train = np.flatnonzero(ts.lt(cutoff).to_numpy())
        valid = np.flatnonzero(ts.ge(start).to_numpy() & ts.lt(end).to_numpy())
        if not len(train) or not len(valid):
            raise ValueError(f"fold {month} has empty train or validation rows")
        folds.append(
            {
                "fold": fold,
                "month": month,
                "train_indices": train,
                "validation_indices": valid,
                "train_cutoff": cutoff,
                "validation_start": start,
                "validation_end": end,
            }
        )
    return folds


def top_fraction_mask(
    frame: pd.DataFrame,
    score: Sequence[float] | np.ndarray,
    *,
    fraction: float = 0.10,
    group_columns: Sequence[str] = ("__ts__", "side_name"),
) -> np.ndarray:
    """Select high scores within exact timestamp/side opportunity groups."""

    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    values = np.asarray(score, dtype=np.float64)
    if len(values) != len(frame):
        raise ValueError("score must align to frame")
    work = frame.loc[:, list(group_columns)].copy()
    work["_position"] = np.arange(len(frame))
    work["_score"] = values
    work = work.loc[np.isfinite(work["_score"])].sort_values(
        [*group_columns, "_score", "_position"],
        ascending=[*[True] * len(group_columns), False, True],
        kind="stable",
    )
    grouped = work.groupby(list(group_columns), sort=False, dropna=False)
    work["_rank"] = grouped.cumcount() + 1
    work["_rows"] = grouped["_position"].transform("size")
    selected = work["_rank"] <= np.maximum(
        1, np.ceil(work["_rows"] * float(fraction)).astype(int)
    )
    mask = np.zeros(len(frame), dtype=bool)
    mask[work.loc[selected, "_position"].to_numpy(np.int64)] = True
    return mask


def economic_metrics(
    frame: pd.DataFrame,
    score: Sequence[float] | np.ndarray,
    *,
    return_column: str = "execution_net_ev_12h",
    admitted: Sequence[bool] | np.ndarray | None = None,
) -> Mapping[str, float | int]:
    """Exact-policy economics under global and timestamp-side top deciles."""

    values = np.asarray(score, dtype=np.float64)
    returns = pd.to_numeric(frame[return_column], errors="coerce").to_numpy(np.float64)
    valid = np.isfinite(values) & np.isfinite(returns)
    if admitted is not None:
        valid &= np.asarray(admitted, dtype=bool)
    local = frame.loc[valid].reset_index(drop=True)
    local_score = values[valid]
    local_return = returns[valid]
    if not len(local):
        return {
            "rows": 0,
            "mean_net_return": np.nan,
            "global_top10_rows": 0,
            "global_top10_mean_net_return": np.nan,
            "timestamp_side_top10_rows": 0,
            "timestamp_side_top10_mean_net_return": np.nan,
        }
    global_order = np.argsort(-local_score, kind="stable")
    global_n = max(1, int(np.ceil(len(local) * 0.10)))
    global_selected = global_order[:global_n]
    timestamp_selected = top_fraction_mask(local, local_score)
    return {
        "rows": int(len(local)),
        "mean_net_return": float(np.mean(local_return)),
        "global_top10_rows": int(global_n),
        "global_top10_mean_net_return": float(np.mean(local_return[global_selected])),
        "timestamp_side_top10_rows": int(timestamp_selected.sum()),
        "timestamp_side_top10_mean_net_return": float(
            np.mean(local_return[timestamp_selected])
        ),
    }
