"""Causal teacher-rank sample weighting for the strict-R3 stack.

The utilities in this module never construct a teacher score.  Callers must
provide a previously materialised OOF/prequential teacher rank on ``[0, 1]``.
Missing teacher ranks retain their existing row weight and never receive a tail
boost.  This makes teacher coverage explicit instead of silently dropping older
training rows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd


Layer = Literal["base", "residual"]


@dataclass(frozen=True)
class DistillationWeightSpec:
    name: str
    use_score_weight: bool = False
    score_power: float = 1.5
    score_floor: float = 0.25
    positive_top_fraction: float | None = None
    positive_boost: float = 1.0
    negative_top_fraction: float | None = None
    negative_boost: float = 1.0
    minimum_weight: float = 0.25
    maximum_weight: float = 4.0

    def validate(self) -> None:
        if self.score_power <= 0.0:
            raise ValueError("score_power must be positive")
        if not 0.0 <= self.score_floor <= 1.0:
            raise ValueError("score_floor must be in [0, 1]")
        for value, label in (
            (self.positive_top_fraction, "positive_top_fraction"),
            (self.negative_top_fraction, "negative_top_fraction"),
        ):
            if value is not None and not 0.0 < value <= 1.0:
                raise ValueError(f"{label} must be in (0, 1]")
        if self.positive_boost < 1.0 or self.negative_boost < 1.0:
            raise ValueError("tail boosts cannot downweight their declared classes")
        if not 0.0 < self.minimum_weight <= 1.0 <= self.maximum_weight:
            raise ValueError("weight cap must straddle one")


def initial_screen_specs(*, power: float = 1.5) -> tuple[DistillationWeightSpec, ...]:
    """Return the predeclared D0--D4 first-round funnel."""

    return (
        DistillationWeightSpec("D0"),
        DistillationWeightSpec("D1", use_score_weight=True, score_power=power),
        DistillationWeightSpec(
            "D2", positive_top_fraction=0.60, positive_boost=1.5,
        ),
        DistillationWeightSpec(
            "D3", negative_top_fraction=0.20, negative_boost=1.5,
        ),
        DistillationWeightSpec(
            "D4", use_score_weight=True, score_power=power,
            positive_top_fraction=0.60, positive_boost=1.5,
            negative_top_fraction=0.20, negative_boost=1.5,
        ),
    )


def _mean_one_bounded(raw: np.ndarray, lower: float, upper: float) -> np.ndarray:
    values = np.asarray(raw, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError("weights must be a non-empty vector")
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("weights must be finite and strictly positive")
    # Solve mean(clip(scale * raw, lower, upper)) == 1.  The monotonic
    # projection preserves ordering while satisfying both declared caps.
    left, right = 0.0, max(1.0 / max(values.min(), 1e-12), 1.0)
    while np.clip(right * values, lower, upper).mean() < 1.0:
        right *= 2.0
    for _ in range(80):
        middle = 0.5 * (left + right)
        if np.clip(middle * values, lower, upper).mean() < 1.0:
            left = middle
        else:
            right = middle
    return np.clip(right * values, lower, upper).astype(np.float32)


def _class_masks(frame: pd.DataFrame, layer: Layer) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if layer == "base":
        if "r3_class" not in frame:
            raise KeyError("base distillation requires r3_class")
        target = pd.to_numeric(frame["r3_class"], errors="coerce").to_numpy(float)
        positive = target == 2.0
        negative = target == 0.0
        weak = target == 1.0
    elif layer == "residual":
        if "policy_residual_bps" not in frame:
            raise KeyError("residual distillation requires policy_residual_bps")
        target = pd.to_numeric(frame["policy_residual_bps"], errors="coerce").to_numpy(float)
        positive = target > 100.0
        negative = target <= -150.0
        weak = ~(positive | negative)
    else:
        raise ValueError(f"unsupported distillation layer: {layer}")
    return positive, negative, weak


def build_distillation_weights(
    frame: pd.DataFrame,
    *,
    teacher_rank_column: str,
    layer: Layer,
    spec: DistillationWeightSpec,
    existing_weight: Sequence[float] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Combine existing weights with one causal teacher-rank curriculum."""

    spec.validate()
    if teacher_rank_column not in frame:
        raise KeyError(f"missing causal teacher rank: {teacher_rank_column}")
    teacher = pd.to_numeric(frame[teacher_rank_column], errors="coerce").to_numpy(float)
    covered = np.isfinite(teacher)
    q = np.clip(np.where(covered, teacher, 0.0), 0.0, 1.0)
    existing = (
        np.ones(len(frame), dtype=float)
        if existing_weight is None else np.asarray(existing_weight, dtype=float)
    )
    if existing.shape != (len(frame),):
        raise ValueError("existing weights are not row-aligned")
    if not np.isfinite(existing).all() or (existing <= 0.0).any():
        raise ValueError("existing weights must be finite and positive")
    positive, negative, weak = _class_masks(frame, layer)
    score_weight = np.ones(len(frame), dtype=float)
    if spec.use_score_weight:
        score_weight[covered] = spec.score_floor + (1.0 - spec.score_floor) * q[covered] ** spec.score_power
    positive_boost = np.ones(len(frame), dtype=float)
    if spec.positive_top_fraction is not None:
        positive_boost[
            covered & positive & (q >= 1.0 - spec.positive_top_fraction)
        ] = spec.positive_boost
    negative_boost = np.ones(len(frame), dtype=float)
    if spec.negative_top_fraction is not None:
        negative_boost[
            covered & negative & (q >= 1.0 - spec.negative_top_fraction)
        ] = spec.negative_boost
    raw = existing * score_weight * positive_boost * negative_boost
    final = _mean_one_bounded(raw, spec.minimum_weight, spec.maximum_weight)
    total = max(float(final.sum()), 1e-12)
    teacher_decile = np.minimum((q * 10.0).astype(int), 9)
    decile_mass = {
        str(decile): float(final[covered & (teacher_decile == decile)].sum() / total)
        for decile in range(10)
    }
    class_names = {"adverse": negative, "weak": weak, "clear": positive}
    class_rows = {name: int(mask.sum()) for name, mask in class_names.items()}
    class_weight_mass = {
        name: float(final[mask].sum() / total) for name, mask in class_names.items()
    }
    order = np.argsort(q, kind="stable")
    top5 = order[max(0, len(order) - int(np.ceil(0.05 * len(order)))):]
    audit: dict[str, object] = {
        "layer": layer,
        "spec": asdict(spec),
        "rows": int(len(frame)),
        "teacher_covered_rows": int(covered.sum()),
        "teacher_coverage": float(covered.mean()),
        "weight_min": float(final.min()),
        "weight_p05": float(np.quantile(final, 0.05)),
        "weight_p50": float(np.quantile(final, 0.50)),
        "weight_p95": float(np.quantile(final, 0.95)),
        "weight_max": float(final.max()),
        "weight_mean": float(final.mean()),
        "effective_sample_size": float(total * total / np.square(final).sum()),
        "effective_sample_ratio": float(total * total / np.square(final).sum() / len(final)),
        "teacher_top5_weight_share": float(final[top5].sum() / total),
        "teacher_decile_weight_mass": decile_mass,
        "class_rows": class_rows,
        "class_weight_mass": class_weight_mass,
    }
    return final, audit

