"""Side-specific, causal ordinal residual-meta primitives.

The module deliberately models *base error*, not absolute trade success.  It
contains only deterministic target, weighting, mapping and diagnostic logic;
the chronological runner owns model fitting and feature selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


EPS = 1e-12
TOP_FRACTIONS: tuple[float, ...] = (.01, .05, .10)


class OrdinalResidualMetaError(ValueError):
    """Raised when a residual-meta contract would be ill-defined."""


@dataclass(frozen=True)
class ResidualClassMap:
    """Prior-only class economics, shrunk side×class -> class global mean."""

    threshold_bps: float
    side_class_mean_bps: Mapping[str, tuple[float, float, float]]
    side_class_support: Mapping[str, tuple[int, int, int]]
    global_class_mean_bps: tuple[float, float, float]
    global_class_support: tuple[int, int, int]
    shrinkage_support: float


def require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [name for name in columns if name not in frame]
    if missing:
        raise OrdinalResidualMetaError(f"frame lacks required fields: {missing}")


def residual_bps(
    frame: pd.DataFrame,
    *,
    net_column: str = "net_bps",
    base_column: str = "prequential_base_expected_net_bps",
) -> np.ndarray:
    require_columns(frame, (net_column, base_column))
    net = pd.to_numeric(frame[net_column], errors="coerce").to_numpy(float)
    base = pd.to_numeric(frame[base_column], errors="coerce").to_numpy(float)
    if not np.isfinite(net).all() or not np.isfinite(base).all():
        raise OrdinalResidualMetaError("net/base values must be finite")
    return net - base


def ordinal_labels(values: Sequence[float], threshold_bps: float) -> np.ndarray:
    """C0 down, C1 neutral, C2 up; boundaries share the declared cost scale."""
    if threshold_bps <= 0:
        raise OrdinalResidualMetaError("materiality threshold must be positive")
    value = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(value).all():
        raise OrdinalResidualMetaError("residual labels must be finite")
    return np.where(value <= -threshold_bps, 0, np.where(value >= threshold_bps, 2, 1)).astype(np.int8)


def fit_soft_binary_residual_scale(
    values: Sequence[float],
    *,
    lower_percentile: float | None = None,
    upper_percentile: float | None = None,
    extrema_bps: float | None = None,
) -> tuple[float, float]:
    """Fit a training-only, zero-centred residual soft-label scale.

    A negative residual means the base overestimated realised net value; a
    positive residual means it underestimated it.  Quantile clips deliberately
    use independent lower and upper scales, so extreme errors cannot dominate
    either tail.  Fixed extrema are symmetric economic alternatives.
    """
    value = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(value).all() or not len(value):
        raise OrdinalResidualMetaError("soft residual scale requires finite non-empty values")
    using_quantiles = lower_percentile is not None or upper_percentile is not None
    if using_quantiles == (extrema_bps is not None):
        raise OrdinalResidualMetaError("declare either percentile clips or symmetric extrema")
    if using_quantiles:
        if lower_percentile is None or upper_percentile is None or not (0.0 < lower_percentile < upper_percentile < 100.0):
            raise OrdinalResidualMetaError("soft residual percentiles must satisfy 0 < lower < upper < 100")
        lower, upper = np.percentile(value, [lower_percentile, upper_percentile])
    else:
        if extrema_bps is None or extrema_bps <= 0:
            raise OrdinalResidualMetaError("soft residual extrema must be positive")
        lower, upper = -float(extrema_bps), float(extrema_bps)
    # Do not let an asymmetric empirical distribution move the economic zero.
    # A side which has no support either side of zero cannot define this target.
    if lower >= -1e-8 or upper <= 1e-8:
        raise OrdinalResidualMetaError("soft residual scale lacks both over- and under-confidence support")
    return float(lower), float(upper)


def soft_binary_residual_labels(
    values: Sequence[float],
    *,
    lower_bps: float,
    upper_bps: float,
) -> np.ndarray:
    """Map residual bps to [0, 1], with exact calibration at 0.5.

    ``0`` denotes clipped base overconfidence and ``1`` clipped base
    underconfidence.  The mapping is piecewise linear around zero, retaining a
    meaningful 0.5 for economically accurate base estimates even when the
    tail scales are asymmetric.
    """
    value = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(value).all() or not lower_bps < 0.0 < upper_bps:
        raise OrdinalResidualMetaError("soft residual labels require finite values and lower < 0 < upper")
    below = 0.5 + 0.5 * np.clip(value / abs(float(lower_bps)), -1.0, 0.0)
    above = 0.5 + 0.5 * np.clip(value / float(upper_bps), 0.0, 1.0)
    return np.where(value <= 0.0, below, above).astype(np.float32)


def cumulative_to_simplex(p_gt_negative: Sequence[float], p_gt_positive: Sequence[float]) -> np.ndarray:
    """Monotonically repair cumulative probabilities and return C0/C1/C2."""
    low = np.clip(np.asarray(p_gt_negative, dtype=float).reshape(-1), 0.0, 1.0)
    high = np.clip(np.asarray(p_gt_positive, dtype=float).reshape(-1), 0.0, 1.0)
    if len(low) != len(high) or not np.isfinite(low).all() or not np.isfinite(high).all():
        raise OrdinalResidualMetaError("cumulative probabilities must be finite and aligned")
    # P(R > +T) cannot exceed P(R > -T).  Averaging would alter both heads;
    # clipping the upper event is the conservative correction.
    high = np.minimum(high, low)
    result = np.column_stack((1.0 - low, low - high, high))
    result = np.clip(result, 0.0, 1.0)
    result /= np.maximum(result.sum(axis=1, keepdims=True), EPS)
    return result.astype(np.float32)


def fit_residual_class_map(
    frame: pd.DataFrame,
    *,
    threshold_bps: float,
    side_column: str = "side_name",
    net_column: str = "net_bps",
    base_column: str = "prequential_base_expected_net_bps",
    shrinkage_support: float = 500.0,
) -> ResidualClassMap:
    """Fit only on resolved training rows supplied by the chronological caller."""
    if shrinkage_support <= 0:
        raise OrdinalResidualMetaError("shrinkage support must be positive")
    require_columns(frame, (side_column,))
    side = frame[side_column].astype(str).str.lower()
    if not set(side.unique()).issubset({"long", "short"}):
        raise OrdinalResidualMetaError("residual map requires explicit long/short sides")
    value = residual_bps(frame, net_column=net_column, base_column=base_column)
    label = ordinal_labels(value, threshold_bps)
    global_mean, global_support = [], []
    for klass in range(3):
        mask = label == klass
        global_support.append(int(mask.sum()))
        global_mean.append(float(value[mask].mean()) if mask.any() else float(value.mean()))
    side_mean: dict[str, tuple[float, float, float]] = {}
    side_support: dict[str, tuple[int, int, int]] = {}
    for name in ("long", "short"):
        local = side.to_numpy() == name
        means, supports = [], []
        for klass in range(3):
            mask = local & (label == klass)
            n = int(mask.sum())
            supports.append(n)
            raw = float(value[mask].mean()) if n else global_mean[klass]
            shrink = n / (n + float(shrinkage_support))
            means.append(shrink * raw + (1.0 - shrink) * global_mean[klass])
        side_mean[name], side_support[name] = tuple(means), tuple(supports)
    return ResidualClassMap(
        threshold_bps=float(threshold_bps), side_class_mean_bps=side_mean,
        side_class_support=side_support, global_class_mean_bps=tuple(global_mean),
        global_class_support=tuple(global_support), shrinkage_support=float(shrinkage_support),
    )


def reconstruct_expected_residual(probability: Sequence[Sequence[float]], sides: Sequence[object], mapping: ResidualClassMap) -> np.ndarray:
    probability = np.asarray(probability, dtype=float)
    if probability.ndim != 2 or probability.shape[1] != 3 or not np.isfinite(probability).all():
        raise OrdinalResidualMetaError("class probabilities must be finite Nx3")
    if (probability < -1e-7).any() or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-5):
        raise OrdinalResidualMetaError("class probabilities must be simplexes")
    side = pd.Series(sides, dtype="string").str.lower().to_numpy()
    if len(side) != len(probability) or not set(side).issubset({"long", "short"}):
        raise OrdinalResidualMetaError("sides must be aligned long/short values")
    means = np.vstack([mapping.side_class_mean_bps[str(name)] for name in side])
    return (probability * means).sum(axis=1).astype(np.float32)


def policy_training_mask(
    frame: pd.DataFrame,
    *,
    rank_column: str = "base_side_rank",
    candidate_column: str = "candidate_id",
    top_fraction: float = .30,
    lower_fraction: float = .10,
) -> np.ndarray:
    """Top side-local policy population plus deterministic lower control rows."""
    if not 0 < top_fraction <= 1 or not 0 <= lower_fraction <= 1:
        raise OrdinalResidualMetaError("invalid policy-population fractions")
    require_columns(frame, (rank_column, candidate_column))
    rank = pd.to_numeric(frame[rank_column], errors="coerce").to_numpy(float)
    if not np.isfinite(rank).all() or (rank < 0).any() or (rank > 1).any():
        raise OrdinalResidualMetaError("base rank must be finite in [0, 1]")
    primary = rank >= 1.0 - top_fraction
    # Stable ID hashing makes lower-control sampling independent of outcome and
    # dataframe ordering.  The probability is exactly declared, not tuned.
    hashed = pd.util.hash_pandas_object(frame[candidate_column].astype(str), index=False).to_numpy(np.uint64)
    unit = (hashed % np.uint64(10_000_000)).astype(float) / 10_000_000.0
    return primary | ((~primary) & (unit < lower_fraction))


def sample_weights(
    frame: pd.DataFrame,
    labels: Sequence[int],
    *,
    rank_column: str = "base_side_rank",
    residual: Sequence[float],
    certainty: Sequence[float] | None = None,
    materiality_floor: float = 50.0,
    materiality_cap: float = 300.0,
    minimum: float = .25,
    maximum: float = 4.0,
) -> np.ndarray:
    """Policy relevance × exact-label certainty × clipped materiality × balance."""
    require_columns(frame, (rank_column,))
    target = np.asarray(labels, dtype=int).reshape(-1)
    value = np.asarray(residual, dtype=float).reshape(-1)
    rank = pd.to_numeric(frame[rank_column], errors="coerce").to_numpy(float)
    if len(target) != len(frame) or len(value) != len(frame) or not np.isfinite(value).all():
        raise OrdinalResidualMetaError("weight inputs must align and be finite")
    relevance = np.where(rank >= .90, 1.0, np.where(rank >= .70, .75, .35))
    if certainty is None:
        confidence = np.ones(len(frame), dtype=float)
    else:
        confidence = np.asarray(certainty, dtype=float).reshape(-1)
        if len(confidence) != len(frame) or not np.isfinite(confidence).all():
            raise OrdinalResidualMetaError("certainty must be finite and aligned")
        confidence = np.clip(confidence, .25, 1.0)
    materiality = np.clip(np.abs(value), materiality_floor, materiality_cap) / 100.0
    counts = np.bincount(target, minlength=max(int(target.max()) + 1, 3)).astype(float)
    class_weight = np.sqrt(len(target) / np.maximum(len(counts) * counts, 1.0))
    weight = relevance * confidence * materiality * class_weight[target]
    weight = np.clip(weight / max(float(weight.mean()), EPS), minimum, maximum)
    return weight.astype(np.float32)


def classifier_diagnostics(labels: Sequence[int], probability: Sequence[Sequence[float]]) -> dict[str, float]:
    """Class metrics are diagnostics; promotion uses the economic evaluator."""
    y = np.asarray(labels, dtype=int).reshape(-1)
    p = np.asarray(probability, dtype=float)
    if p.ndim != 2 or len(p) != len(y) or p.shape[1] < 2:
        raise OrdinalResidualMetaError("diagnostic probability shape is invalid")
    p = np.clip(p, EPS, 1.0); p /= p.sum(axis=1, keepdims=True)
    onehot = np.eye(p.shape[1])[y]
    rps = float(np.mean(np.sum((np.cumsum(p, axis=1) - np.cumsum(onehot, axis=1)) ** 2, axis=1) / (p.shape[1] - 1)))
    result = {"rps": rps, "log_loss": float(log_loss(y, p, labels=list(range(p.shape[1])))), "brier_multiclass": float(np.mean(np.sum((p - onehot) ** 2, axis=1)))}
    predicted = p.argmax(axis=1)
    for klass in range(p.shape[1]):
        mask = y == klass
        result[f"recall_c{klass}"] = float((predicted[mask] == klass).mean()) if mask.any() else np.nan
        result[f"brier_c{klass}"] = float(brier_score_loss(mask.astype(int), p[:, klass]))
    return result


__all__ = [
    "TOP_FRACTIONS", "OrdinalResidualMetaError", "ResidualClassMap", "classifier_diagnostics",
    "cumulative_to_simplex", "fit_residual_class_map", "fit_soft_binary_residual_scale", "ordinal_labels", "policy_training_mask",
    "reconstruct_expected_residual", "residual_bps", "sample_weights",
    "soft_binary_residual_labels",
]
