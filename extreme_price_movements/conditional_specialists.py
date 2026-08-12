"""Utilities and contracts for portable pair-condition specialists.

The module is deliberately model-agnostic.  It owns only the causal condition
spine contract, soft memberships, support/portability statistics, and the
deterministic complementarity calculations used by the runner.  Training and
OOF materialisation remain in the existing LambdaRank pipeline.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ConditionalSpecialistConfig:
    """Default scale for the portable pair-condition funnel."""

    global_seed: int = 20260806
    max_raw_spine_features: int = 60
    min_raw_spine_features: int = 40
    hard_alias_spearman: float = 0.995
    local_redundancy_spearman: float = 0.98
    soft_transition_width_quantile: float = 0.05
    minimum_single_activation_share: float = 0.05
    maximum_single_activation_share: float = 0.40
    maximum_pairs_before_screen: int = 5000
    maximum_states_per_pair: int = 4
    minimum_effective_rows: float = 1000.0
    minimum_effective_queries: int = 250
    minimum_supported_months: int = 3
    minimum_nonadjacent_months: int = 3
    minimum_month_effective_queries: int = 50
    top_candidates_per_side: int = 200
    top_conditions_for_full_feature_scan: int = 80
    specialist_min_features: int = 30
    # The linked orthogonal-specialist brief predeclares this cap funnel.  A
    # later one-SE/portability gate may select a smaller prefix, but the
    # discovery artifact must retain evidence for every cap that is available
    # in the causal predictive pool.
    specialist_feature_caps: tuple[int, ...] = (40, 60, 80, 100, 120)
    specialist_max_features: int = 120
    group_mda_repeats: int = 3
    equal_condition_month_weighting: bool = True
    condition_weight_exponent: float = 1.5
    membership_exponents: tuple[float, ...] = (1.0, 1.5, 2.0)
    residual_grade_edges: tuple[float, ...] = (-150.0, -50.0, 50.0, 150.0)

    def to_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["specialist_feature_caps"] = list(self.specialist_feature_caps)
        out["membership_exponents"] = list(self.membership_exponents)
        out["residual_grade_edges"] = list(self.residual_grade_edges)
        return out


def effective_rows(weights: np.ndarray) -> float:
    """Kish effective sample size for soft memberships."""

    w = np.asarray(weights, dtype=np.float64)
    finite = np.isfinite(w) & (w > 0.0)
    if not finite.any():
        return 0.0
    x = w[finite]
    den = float(np.square(x).sum())
    return float(np.square(x.sum()) / den) if den > 0.0 else 0.0


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0.0)
    if not ok.any():
        return float("nan")
    return float(np.average(x[ok], weights=w[ok]))


def weighted_corr(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    """Weighted Pearson correlation; callers pass rank-transformed inputs."""

    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(w) & (w > 0.0)
    if ok.sum() < 3:
        return float("nan")
    a, b, w = a[ok], b[ok], w[ok]
    sw = float(w.sum())
    ma = float(np.dot(w, a) / sw)
    mb = float(np.dot(w, b) / sw)
    va = float(np.dot(w, (a - ma) ** 2) / sw)
    vb = float(np.dot(w, (b - mb) ** 2) / sw)
    if va <= 1e-15 or vb <= 1e-15:
        return float("nan")
    return float(np.dot(w, (a - ma) * (b - mb)) / sw / np.sqrt(va * vb))


def portability_score(values: Iterable[float], *, mad_penalty: float = 0.5, worst_penalty: float = 1.0) -> float:
    """Portable score used by both cheap and full response screens."""

    x = np.asarray([float(v) for v in values if np.isfinite(v)], dtype=np.float64)
    if x.size == 0:
        return float("nan")
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    worst = float(np.min(x))
    return median - mad_penalty * mad - worst_penalty * max(0.0, -worst)


def soft_regions(values: np.ndarray, low_q: float = 0.25, high_q: float = 0.75, width_quantile: float = 0.05) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Return smooth low/high memberships fitted from the supplied training rows.

    Quantiles and transition scales must be computed on outer-train only.  The
    logistic tails keep memberships usable on future rows without introducing
    a hard discontinuity at q25/q75.
    """

    x = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(x)
    if finite.sum() < 8:
        raise ValueError("soft region fit requires at least eight finite values")
    q05, q25, q75, q95 = np.nanquantile(x, [0.05, low_q, high_q, 0.95])
    scale = max(float(np.nanquantile(x[finite], 0.5 + width_quantile / 2.0) - np.nanquantile(x[finite], 0.5 - width_quantile / 2.0)), 1e-8)
    scale = max(scale, abs(float(q75 - q25)) * width_quantile, 1e-8)
    fill = float(np.nanmedian(x[finite]))
    z = np.where(finite, x, fill)
    low = 1.0 / (1.0 + np.exp(np.clip((z - float(q25)) / scale, -40.0, 40.0)) )
    high = 1.0 / (1.0 + np.exp(np.clip((float(q75) - z) / scale, -40.0, 40.0)) )
    low[~finite] = 0.0
    high[~finite] = 0.0
    return low.astype(np.float32), high.astype(np.float32), {
        "q05": float(q05), "q25": float(q25), "q75": float(q75), "q95": float(q95),
        "scale": float(scale), "fill_median": fill,
        "low_share": float(np.mean(low >= 0.5)), "high_share": float(np.mean(high >= 0.5)),
    }


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() == 0:
        return float("nan")
    x, y = x[ok], y[ok]
    nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
    if nx <= 1e-12 or ny <= 1e-12:
        return 1.0
    return float(1.0 - np.dot(x, y) / (nx * ny))


def weighted_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    x = np.clip(np.asarray(a, dtype=np.float64), 0.0, None)
    y = np.clip(np.asarray(b, dtype=np.float64), 0.0, None)
    den = float(np.maximum(x, y).sum())
    return float(np.minimum(x, y).sum() / den) if den > 0.0 else 0.0


def ordinal_residual_grade(residual: np.ndarray, edges: tuple[float, ...] = (-150.0, -50.0, 50.0, 150.0)) -> np.ndarray:
    """Canonical five-grade residual label used by the existing meta layer."""

    e0, e1, e2, e3 = map(float, edges)
    r = np.asarray(residual, dtype=np.float64)
    return np.select((r <= e0, r <= e1, r <= e2, r <= e3), (0, 1, 2, 3), default=4).astype(np.int32)
