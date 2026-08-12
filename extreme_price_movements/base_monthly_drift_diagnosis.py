"""Pure diagnostics for monthly base-model drift attribution.

This module deliberately has no model fitting, artifact I/O, or target
construction.  A runner must first produce frozen- and refit-model scores on
the *same candidate population*, then may use these functions to measure
prediction stability and label-free attribution conditions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


class BaseMonthlyDriftDiagnosticError(ValueError):
    """Raised when a paired monthly score surface is not comparable."""


DEFAULT_TOP_FRACTIONS: tuple[float, ...] = (0.05, 0.30, 0.40)


def _validate_top_fractions(top_fractions: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in top_fractions)
    if not values or any(not 0.0 < value <= 1.0 for value in values):
        raise BaseMonthlyDriftDiagnosticError("top_fractions must contain values in (0, 1]")
    if len(set(values)) != len(values):
        raise BaseMonthlyDriftDiagnosticError("top_fractions must be unique")
    return values


def _fraction_label(value: float) -> str:
    return f"top_{int(round(value * 100)):02d}"


def _top_ids(frame: pd.DataFrame, *, score: str, identity: Sequence[str], fraction: float) -> set[tuple[object, ...]]:
    """Deterministically select a pooled top fraction, resolving ties by identity."""
    ordered = frame.sort_values([score, *identity], ascending=[False, *([True] * len(identity))], kind="stable")
    count = max(1, int(np.ceil(fraction * len(ordered))))
    return set(map(tuple, ordered.loc[:, list(identity)].iloc[:count].itertuples(index=False, name=None)))


def paired_score_stability(
    paired: pd.DataFrame,
    *,
    frozen_score: str = "frozen_score",
    refit_score: str = "refit_score",
    identity: Sequence[str] = ("candidate_id",),
    top_fractions: Sequence[float] = DEFAULT_TOP_FRACTIONS,
) -> dict[str, float | int]:
    """Measure frozen/refit score stability on one exact candidate surface.

    ``paired`` must already be a one-row-per-candidate aligned table.  This is
    intentionally stricter than accepting two frames: accidental comparison
    of different populations would otherwise look like model drift.
    """
    keys = tuple(identity)
    required = [*keys, frozen_score, refit_score]
    missing = [name for name in required if name not in paired]
    if missing:
        raise BaseMonthlyDriftDiagnosticError(f"paired scores lack required columns: {missing}")
    if not len(paired):
        raise BaseMonthlyDriftDiagnosticError("paired scores are empty")
    if paired.loc[:, list(keys)].isna().any().any() or paired.duplicated(list(keys)).any():
        raise BaseMonthlyDriftDiagnosticError("paired scores require non-null unique candidate identity")

    frozen = pd.to_numeric(paired[frozen_score], errors="coerce").to_numpy(float)
    refit = pd.to_numeric(paired[refit_score], errors="coerce").to_numpy(float)
    if not np.isfinite(frozen).all() or not np.isfinite(refit).all():
        raise BaseMonthlyDriftDiagnosticError("paired scores require finite frozen and refit values")

    diff = refit - frozen
    result: dict[str, float | int] = {
        "rows": int(len(paired)),
        "score_spearman": float(pd.Series(frozen).corr(pd.Series(refit), method="spearman")),
        "score_mae": float(np.mean(np.abs(diff))),
        "score_rmse": float(np.sqrt(np.mean(np.square(diff)))),
        "frozen_mean": float(np.mean(frozen)),
        "refit_mean": float(np.mean(refit)),
        "mean_shift": float(np.mean(diff)),
    }
    for fraction in _validate_top_fractions(top_fractions):
        left = _top_ids(paired, score=frozen_score, identity=keys, fraction=fraction)
        right = _top_ids(paired, score=refit_score, identity=keys, fraction=fraction)
        intersection = len(left.intersection(right))
        union = len(left.union(right))
        label = _fraction_label(fraction)
        result[f"{label}_rows"] = int(len(left))
        result[f"{label}_overlap_rows"] = int(intersection)
        result[f"{label}_overlap_fraction"] = float(intersection / len(left))
        result[f"{label}_jaccard"] = float(intersection / union)
    return result


@dataclass(frozen=True)
class DriftAttributionThresholds:
    """Predeclared thresholds for a descriptive, not causal, attribution."""

    refit_ic_gain: float = 0.01
    score_spearman_max_for_model_drift: float = 0.95
    top05_overlap_max_for_model_drift: float = 0.80
    input_psi: float = 0.20
    input_extrapolation_rate: float = 0.05
    relationship_ic_drop: float = 0.02
    calibration_slope_shift: float = 0.25


def classify_drift_attribution(
    metrics: Mapping[str, float | int],
    *,
    thresholds: DriftAttributionThresholds = DriftAttributionThresholds(),
) -> str:
    """Classify evidence using only caller-supplied diagnostic summaries.

    Required inputs are ``frozen_rank_ic``, ``refit_rank_ic``,
    ``score_spearman``, ``top_05_overlap_fraction``, ``max_feature_psi``,
    ``max_feature_extrapolation_rate``, ``train_rank_ic``, and
    ``calibration_slope_shift``.  The output intentionally permits mixed
    causes; it must not be read as a causal proof.
    """
    required = (
        "frozen_rank_ic", "refit_rank_ic", "score_spearman", "top_05_overlap_fraction",
        "max_feature_psi", "max_feature_extrapolation_rate", "train_rank_ic", "calibration_slope_shift",
    )
    missing = [name for name in required if name not in metrics]
    if missing:
        raise BaseMonthlyDriftDiagnosticError(f"attribution metrics lack required fields: {missing}")
    values = {name: float(metrics[name]) for name in required}
    if not np.isfinite(list(values.values())).all():
        raise BaseMonthlyDriftDiagnosticError("attribution metrics must be finite")

    model = (
        values["refit_rank_ic"] - values["frozen_rank_ic"] >= thresholds.refit_ic_gain
        and (
            values["score_spearman"] <= thresholds.score_spearman_max_for_model_drift
            or values["top_05_overlap_fraction"] <= thresholds.top05_overlap_max_for_model_drift
        )
    )
    population = (
        values["max_feature_psi"] >= thresholds.input_psi
        or values["max_feature_extrapolation_rate"] >= thresholds.input_extrapolation_rate
    )
    relationship = (
        values["train_rank_ic"] - values["frozen_rank_ic"] >= thresholds.relationship_ic_drop
        or abs(values["calibration_slope_shift"]) >= thresholds.calibration_slope_shift
    )
    labels = [
        label for label, active in (
            ("MODEL_DRIFT", model),
            ("INPUT_POPULATION_DRIFT", population),
            ("ECONOMIC_RELATIONSHIP_DRIFT", relationship),
        ) if active
    ]
    return "NO_DOMINANT_DRIFT" if not labels else "+".join(labels) if len(labels) > 1 else labels[0]

