"""Train-only sample weights for hierarchical side-label optimization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd


TARGET_EXPONENT_GRID = (1.0, 1.25, 1.5, 1.75, 2.0)
WEIGHT_RANGE_RATIO_MIN = 3.0
WEIGHT_RANGE_RATIO_MAX = 12.0


@dataclass(frozen=True)
class TargetStrengthWeightSpec:
    exponent: float = 1.0
    archetype_balance_gamma: float = 0.25
    archetype_factor_min: float = 0.80
    archetype_factor_max: float = 1.25
    timestamp_factor_min: float = 0.50
    timestamp_factor_max: float = 2.00
    raw_clip_quantile: float = 0.99
    weight_range_ratio: float = 4.00


def _bounded_mean_one_scale(
    raw: np.ndarray,
    *,
    lower: float,
    upper: float,
) -> np.ndarray:
    """Scale positive raw weights to bounded mean one using monotone bisection."""

    if raw.size == 0:
        return raw.astype(np.float32)
    values = np.nan_to_num(raw, nan=1.0, posinf=upper, neginf=lower)
    values = np.maximum(values, np.finfo(np.float64).tiny)
    lo, hi = 0.0, 1.0
    while float(np.mean(np.clip(hi * values, lower, upper))) < 1.0:
        hi *= 2.0
        if not np.isfinite(hi):
            raise FloatingPointError("could not bracket bounded mean-one scale")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        mean = float(np.mean(np.clip(mid * values, lower, upper)))
        if mean < 1.0:
            lo = mid
        else:
            hi = mid
    result = np.clip(0.5 * (lo + hi) * values, lower, upper)
    return result.astype(np.float32, copy=False)


def build_target_strength_weights(
    target_soft: Sequence[float] | pd.Series | np.ndarray,
    *,
    timestamps: Sequence[Any] | pd.Series | np.ndarray,
    archetypes: Sequence[Any] | pd.Series | np.ndarray,
    spec: TargetStrengthWeightSpec,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build bounded opportunity weights from permitted training rows only.

    Timestamp balancing prevents dense cross-sections from dominating. The
    tempered archetype correction protects smaller groups without equalizing
    away the economic target-strength ordering inside each archetype.
    """

    target = np.asarray(target_soft, dtype=np.float64)
    ts = pd.Series(timestamps, copy=False)
    archetype = pd.Series(archetypes, copy=False).fillna("__missing__").astype(str)
    if not (len(target) == len(ts) == len(archetype)):
        raise ValueError("target, timestamps, and archetypes must have equal length")
    if float(spec.exponent) not in TARGET_EXPONENT_GRID:
        raise ValueError(f"exponent must be one of {TARGET_EXPONENT_GRID}")
    if not 0.0 < float(spec.raw_clip_quantile) <= 1.0:
        raise ValueError("raw_clip_quantile must be in (0, 1]")
    if not WEIGHT_RANGE_RATIO_MIN <= float(spec.weight_range_ratio) <= WEIGHT_RANGE_RATIO_MAX:
        raise ValueError(
            "weight_range_ratio must be within "
            f"[{WEIGHT_RANGE_RATIO_MIN}, {WEIGHT_RANGE_RATIO_MAX}]"
        )
    if target.size == 0:
        return np.empty(0, dtype=np.float32), {
            "schema": "target_strength_weight_v1",
            "spec": asdict(spec),
            "rows": 0,
        }

    finite_target = np.clip(
        np.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0
    )
    strength = np.power(finite_target, float(spec.exponent))
    raw = strength.copy()

    timestamp_counts = ts.groupby(ts, dropna=False).transform("size").to_numpy(
        dtype=np.float64
    )
    timestamp_factor = 1.0 / np.maximum(timestamp_counts, 1.0)
    timestamp_factor /= max(float(np.mean(timestamp_factor)), 1e-12)
    timestamp_factor = np.clip(
        timestamp_factor,
        float(spec.timestamp_factor_min),
        float(spec.timestamp_factor_max),
    )

    archetype_counts = archetype.groupby(archetype, dropna=False).transform(
        "size"
    ).to_numpy(dtype=np.float64)
    unique_counts = archetype.value_counts(dropna=False).to_numpy(dtype=np.float64)
    median_support = float(np.median(unique_counts)) if unique_counts.size else 1.0
    archetype_factor = np.power(
        median_support / np.maximum(archetype_counts, 1.0),
        float(spec.archetype_balance_gamma),
    )
    archetype_factor = np.clip(
        archetype_factor,
        float(spec.archetype_factor_min),
        float(spec.archetype_factor_max),
    )

    raw *= timestamp_factor * archetype_factor
    finite_raw = raw[np.isfinite(raw)]
    raw_p99 = (
        float(np.quantile(finite_raw, float(spec.raw_clip_quantile)))
        if finite_raw.size
        else 1.0
    )
    raw = np.minimum(np.nan_to_num(raw, nan=1.0), max(raw_p99, 1e-12))
    weight_max = float(np.sqrt(float(spec.weight_range_ratio)))
    weight_min = 1.0 / weight_max
    weights = _bounded_mean_one_scale(
        raw,
        lower=weight_min,
        upper=weight_max,
    )
    total = max(float(np.sum(weights, dtype=np.float64)), 1e-12)
    ess = total * total / max(
        float(np.sum(np.square(weights, dtype=np.float64))), 1e-12
    )
    top_decile = finite_target >= float(np.quantile(finite_target, 0.90))
    diagnostics = {
        "schema": "target_strength_weight_v1",
        "spec": asdict(spec),
        "derived_weight_min": weight_min,
        "derived_weight_max": weight_max,
        "rows": int(len(weights)),
        "raw_weight_p99": raw_p99,
        "weight_mean": float(np.mean(weights)) if len(weights) else float("nan"),
        "weight_min": float(np.min(weights)) if len(weights) else float("nan"),
        "weight_max": float(np.max(weights)) if len(weights) else float("nan"),
        "effective_sample_size": float(ess),
        "effective_sample_fraction": float(ess / len(weights)) if len(weights) else 0.0,
        "top_target_decile_weight_share": (
            float(np.sum(weights[top_decile], dtype=np.float64) / total)
            if len(weights)
            else 0.0
        ),
        "archetype_count": int(archetype.nunique(dropna=False)),
        "timestamp_count": int(ts.nunique(dropna=False)),
    }
    return weights, diagnostics
