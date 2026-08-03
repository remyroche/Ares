"""Primitives for the meaningful-MFE event-classifier ablation.

The baseline target is a side-normalized, ATR-normalized soft triple barrier:

* upper barrier: max(1.5 ATR, the canonical 1.5% meaningful-return floor);
* lower barrier: 1.0 ATR;
* timeout: 12 hours;
* same-hour upper/lower conflicts: adverse barrier wins conservatively.

The target remains continuous so near misses and fast clean hits retain more
information than a hard first-touch label.  These helpers contain no estimator
code and are deterministic enough to unit test independently of the runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


@dataclass(frozen=True)
class TripleBarrierSoftLabel:
    upper_atr: float = 1.5
    upper_return_floor: float = 0.015
    lower_atr: float = 1.0
    horizon_hours: float = 12.0
    temperature: float = 0.35
    use_time_bonus: bool = True


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def atr_soft_triple_barrier_labels(
    frame: pd.DataFrame,
    contract: TripleBarrierSoftLabel = TripleBarrierSoftLabel(),
) -> pd.DataFrame:
    """Return soft and hard outcomes for one ATR-normalized triple barrier.

    The canonical path table stores adverse excursion through the meaningful
    hit bar.  Therefore an hourly bar which touches both barriers is assigned
    to the adverse outcome; this is deliberately conservative and avoids
    inventing intrabar ordering from hourly OHLC.
    """

    if contract.upper_atr <= 0.0 or contract.lower_atr <= 0.0:
        raise ValueError("triple-barrier ATR levels must be positive")
    if contract.horizon_hours <= 0.0 or contract.temperature <= 0.0:
        raise ValueError("horizon and temperature must be positive")
    required = {
        "__path_auxiliary_atr_fraction__",
        "__peak_mfe_atr_clip_8__",
        "__mae_before_meaningful_mfe_atr_12h__",
        "__time_to_first_meaningful_mfe_hours_12h__",
        "__meaningful_mfe_reached_12h__",
        "__path_auxiliary_target_valid__",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            "meaningful-MFE triple-barrier target missing columns: "
            + ", ".join(missing)
        )

    def values(column: str) -> np.ndarray:
        return pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)

    valid = values("__path_auxiliary_target_valid__") > 0.5
    atr_fraction = values("__path_auxiliary_atr_fraction__")
    peak_atr = values("__peak_mfe_atr_clip_8__")
    mae_atr = values("__mae_before_meaningful_mfe_atr_12h__")
    time_hours = values("__time_to_first_meaningful_mfe_hours_12h__")
    reached = values("__meaningful_mfe_reached_12h__") > 0.5
    finite = (
        np.isfinite(atr_fraction)
        & (atr_fraction > 0.0)
        & np.isfinite(peak_atr)
        & np.isfinite(mae_atr)
        & np.isfinite(time_hours)
    )
    valid &= finite

    upper_atr = np.maximum(
        float(contract.upper_atr),
        float(contract.upper_return_floor) / np.maximum(atr_fraction, 1e-8),
    )
    favorable_progress = np.clip(peak_atr / upper_atr, 0.0, 1.0)
    adverse_progress = np.clip(mae_atr / float(contract.lower_atr), 0.0, 1.0)

    # MAE includes the meaningful-hit bar. Treat a simultaneous hourly touch
    # as adverse because the within-bar order is not observable.
    adverse_first = valid & (mae_atr >= float(contract.lower_atr))
    favorable_first = valid & reached & ~adverse_first
    timeout = valid & ~favorable_first & ~adverse_first
    hard = favorable_first.astype(np.float32)

    relative_progress = (favorable_progress - adverse_progress) / float(
        contract.temperature
    )
    soft = _sigmoid(relative_progress)
    if contract.use_time_bonus:
        early = np.clip(1.0 - time_hours / float(contract.horizon_hours), 0.0, 1.0)
        favorable_floor = 0.75 + 0.25 * early
    else:
        favorable_floor = np.full(len(frame), 0.75, dtype=np.float64)
    soft = np.where(favorable_first, np.maximum(soft, favorable_floor), soft)
    soft = np.where(
        adverse_first,
        np.minimum(soft, 0.25 * favorable_progress),
        soft,
    )
    soft = np.where(valid, np.clip(soft, 0.0, 1.0), np.nan)

    outcome = np.full(len(frame), "invalid", dtype=object)
    outcome[timeout] = "timeout"
    outcome[adverse_first] = "adverse_first_or_conflict"
    outcome[favorable_first] = "favorable_first"
    return pd.DataFrame(
        {
            "tb_soft_label": soft.astype(np.float32),
            "tb_hard_label": np.where(valid, hard, np.nan).astype(np.float32),
            "tb_outcome": outcome,
            "tb_upper_atr": np.where(valid, upper_atr, np.nan).astype(np.float32),
            "tb_favorable_progress": np.where(valid, favorable_progress, np.nan).astype(
                np.float32
            ),
            "tb_adverse_progress": np.where(valid, adverse_progress, np.nan).astype(
                np.float32
            ),
            "tb_valid": valid,
        },
        index=frame.index,
    )


def competing_risk_targets(labels: pd.DataFrame) -> pd.DataFrame:
    """Encode timeout, adverse-first, and favorable-first outcomes.

    Rows marked ``adverse_first_or_conflict`` which also reached meaningful
    MFE are order-ambiguous at the resolution of the stored path target.  The
    ambiguity flag permits an ablation which reduces their training weight
    without silently relabeling them as favorable.
    """

    required = {"tb_outcome", "tb_soft_label"}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError("competing-risk labels missing columns: " + ", ".join(missing))
    outcome = labels["tb_outcome"].astype(str)
    valid = outcome.isin(
        ["timeout", "adverse_first_or_conflict", "favorable_first"]
    )
    class_id = np.full(len(labels), -1, dtype=np.int8)
    class_id[outcome.eq("timeout").to_numpy()] = 0
    class_id[outcome.eq("adverse_first_or_conflict").to_numpy()] = 1
    class_id[outcome.eq("favorable_first").to_numpy()] = 2
    reached_source = next(
        (
            labels[column]
            for column in (
                "meaningful_mfe_reached",
                "__meaningful_mfe_reached_12h__",
            )
            if column in labels
        ),
        pd.Series(0.0, index=labels.index),
    )
    reached = (
        pd.to_numeric(reached_source, errors="coerce")
        .fillna(0.0)
        .gt(0.5)
        .to_numpy()
    )
    ambiguous = (
        outcome.eq("adverse_first_or_conflict").to_numpy() & reached & valid.to_numpy()
    )
    soft = pd.to_numeric(labels["tb_soft_label"], errors="coerce").to_numpy(np.float64)
    # Conditional quality is only a supervised target for favorable-first
    # rows. Rescale the favorable floor [0.75, 1] to [0, 1].
    quality = np.where(
        class_id == 2,
        np.clip((soft - 0.75) / 0.25, 0.0, 1.0),
        np.nan,
    )
    return pd.DataFrame(
        {
            "risk_class": class_id,
            "favorable_first": (class_id == 2).astype(np.float32),
            "adverse_first": (class_id == 1).astype(np.float32),
            "timeout": (class_id == 0).astype(np.float32),
            "order_ambiguous": ambiguous,
            "conditional_quality": quality.astype(np.float32),
            "risk_valid": valid.to_numpy(),
        },
        index=labels.index,
    )


def event_quality_scores(
    favorable_probability: Sequence[float],
    conditional_quality: Sequence[float],
    *,
    quality_floor: float = 0.25,
) -> dict[str, np.ndarray]:
    """Compose event probability and conditional quality without conflation."""

    if not 0.0 <= quality_floor <= 1.0:
        raise ValueError("quality_floor must be between zero and one")
    probability = np.clip(
        np.asarray(favorable_probability, dtype=np.float64), 0.0, 1.0
    )
    quality = np.clip(np.asarray(conditional_quality, dtype=np.float64), 0.0, 1.0)
    if probability.shape != quality.shape:
        raise ValueError("probability and conditional quality must align")
    adjusted_quality = quality_floor + (1.0 - quality_floor) * quality
    return {
        "probability_x_quality": probability * adjusted_quality,
        "probability_gated_quality": np.where(
            probability >= 0.5, adjusted_quality, probability * adjusted_quality
        ),
    }


def first_21d_admission(
    timestamps: Sequence[object],
    score: Sequence[float],
    realized_net_return: Sequence[float],
    *,
    fit_days: int = 21,
) -> dict[str, object]:
    """Fit a causal isotonic admission rule on the first observed days."""

    ts = pd.to_datetime(pd.Series(timestamps), utc=True, errors="coerce")
    values = np.asarray(score, dtype=np.float64)
    returns = np.asarray(realized_net_return, dtype=np.float64)
    if len(ts) != len(values) or values.shape != returns.shape:
        raise ValueError("admission timestamps, scores, and returns must align")
    finite = ts.notna().to_numpy() & np.isfinite(values) & np.isfinite(returns)
    ordered_days = ts.loc[finite].dt.floor("D").drop_duplicates().sort_values()
    if len(ordered_days) < int(fit_days) + 1:
        raise ValueError("insufficient days for causal admission calibration")
    cutoff = pd.Timestamp(ordered_days.iloc[int(fit_days) - 1]) + pd.Timedelta(days=1)
    fit = finite & ts.lt(cutoff).to_numpy()
    evaluate = finite & ts.ge(cutoff).to_numpy()
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(values[fit], returns[fit])
    calibrated = np.full(len(values), np.nan, dtype=np.float64)
    calibrated[evaluate] = calibrator.predict(values[evaluate])
    admitted = evaluate & (calibrated > 0.0)
    return {
        "fit_days": int(fit_days),
        "fit_rows": int(fit.sum()),
        "fit_end_exclusive": cutoff,
        "evaluation_rows": int(evaluate.sum()),
        "admitted_rows": int(admitted.sum()),
        "evaluation_mask": evaluate,
        "admitted_mask": admitted,
        "calibrated_expected_net_return": calibrated,
        "admission_rule": "isotonic_expected_net_return_gt_0",
    }


def expanding_resolved_month_folds(
    timestamps: Sequence[object] | pd.Series,
    label_resolved_at: Sequence[object] | pd.Series,
    *,
    validation_months: Sequence[str] = ("2026-05", "2026-06", "2026-07"),
) -> list[dict[str, object]]:
    """Build expanding folds whose training labels resolve before validation."""

    ts = pd.to_datetime(pd.Series(timestamps), utc=True, errors="coerce")
    resolved = pd.to_datetime(pd.Series(label_resolved_at), utc=True, errors="coerce")
    if ts.isna().any() or resolved.isna().any():
        raise ValueError("timestamps and label resolution must be valid UTC")
    if len(ts) != len(resolved):
        raise ValueError("timestamps and label resolution must align")
    folds: list[dict[str, object]] = []
    for fold, month in enumerate(validation_months):
        period = pd.Period(month, freq="M")
        start = pd.Timestamp(period.start_time, tz="UTC")
        end = pd.Timestamp((period + 1).start_time, tz="UTC")
        train = np.flatnonzero((ts < start).to_numpy() & (resolved < start).to_numpy())
        valid = np.flatnonzero((ts >= start).to_numpy() & (ts < end).to_numpy())
        if not len(train) or not len(valid):
            raise ValueError(f"empty train or validation rows for {month}")
        folds.append(
            {
                "fold": fold,
                "month": month,
                "train_indices": train,
                "validation_indices": valid,
                "validation_start": start,
                "validation_end": end,
                "training_label_resolved_max": resolved.iloc[train].max(),
            }
        )
    return folds
