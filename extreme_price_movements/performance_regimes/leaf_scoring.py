"""Leaf edge, OOF contribution, stability, and pruning rules."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from extreme_price_movements.performance_regimes.membership import (
    active_block_count,
    get_active_positions,
)


def weighted_brier(y_true: pd.Series, pred: pd.Series, sample_weight: pd.Series) -> float:
    y = pd.to_numeric(y_true, errors="coerce").to_numpy(dtype=float)
    p = pd.to_numeric(pred, errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(sample_weight, errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(y) & np.isfinite(p) & np.isfinite(w) & (w >= 0.0)
    if not ok.any():
        return np.nan
    return float(np.average((y[ok] - np.clip(p[ok], 0.0, 1.0)) ** 2, weights=np.maximum(w[ok], 1e-12)))


def weighted_logloss(y_true: pd.Series, pred: pd.Series, sample_weight: pd.Series) -> float:
    y = pd.to_numeric(y_true, errors="coerce").to_numpy(dtype=float)
    p = pd.to_numeric(pred, errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(sample_weight, errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(y) & np.isfinite(p) & np.isfinite(w) & (w >= 0.0)
    if not ok.any():
        return np.nan
    p = np.clip(p[ok], 1e-6, 1.0 - 1e-6)
    y = np.clip(y[ok], 0.0, 1.0)
    return float(np.average(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)), weights=np.maximum(w[ok], 1e-12)))


def score_directional_edges(
    *,
    direction: Literal["bad", "good"],
    leaf_label_mean: float,
    global_label_mean: float,
    leaf_strategy_perf_mean: float,
    global_strategy_perf_mean: float,
    weighted_coverage: float,
    alpha: float = 1.0,
) -> dict[str, float]:
    if direction == "bad":
        directional_label_edge = float(leaf_label_mean - global_label_mean)
        directional_perf_edge = float(global_strategy_perf_mean - leaf_strategy_perf_mean)
    else:
        directional_label_edge = float(leaf_label_mean - global_label_mean)
        directional_perf_edge = float(leaf_strategy_perf_mean - global_strategy_perf_mean)
    positive_label_edge = max(0.0, directional_label_edge)
    positive_perf_edge = max(0.0, directional_perf_edge)
    exponent = float(alpha)
    return {
        "directional_label_edge": directional_label_edge,
        "positive_label_edge": positive_label_edge,
        "label_edge_mass": float(weighted_coverage * positive_label_edge**exponent),
        "directional_perf_edge": directional_perf_edge,
        "positive_perf_edge": positive_perf_edge,
        "perf_edge_mass": float(weighted_coverage * positive_perf_edge**exponent),
    }


def score_leaf_oof_contribution(
    leaf,
    model_oof_predictions: pd.Series,
    baseline_oof_predictions: pd.Series,
    y_true: pd.Series,
    sample_weight: pd.Series,
    *,
    metric: Literal["brier", "logloss", "weighted_mse"] = "brier",
) -> float:
    """Estimate positive OOF loss degradation attributable to active membership."""

    active_positions = get_active_positions(leaf)
    active = np.zeros(len(model_oof_predictions), dtype=bool)
    valid_positions = active_positions[(active_positions >= 0) & (active_positions < len(active))]
    active[valid_positions] = True
    if active.size != len(model_oof_predictions):
        active = np.resize(active, len(model_oof_predictions)).astype(bool)
    model_pred = pd.to_numeric(model_oof_predictions, errors="coerce").copy()
    ablated = model_pred.copy()
    base_pred = pd.to_numeric(baseline_oof_predictions, errors="coerce")
    ablated.iloc[active] = base_pred.iloc[active]
    if metric == "logloss":
        loss_model = weighted_logloss(y_true, model_pred, sample_weight)
        loss_ablated = weighted_logloss(y_true, ablated, sample_weight)
    else:
        loss_model = weighted_brier(y_true, model_pred, sample_weight)
        loss_ablated = weighted_brier(y_true, ablated, sample_weight)
    if not np.isfinite(loss_model) or not np.isfinite(loss_ablated):
        return 0.0
    return float(loss_ablated - loss_model)


def estimate_leaf_stability(
    leaf,
    *,
    time_blocks: pd.Series,
    bootstrap_count: int = 32,
    min_block_count: int = 4,
) -> float:
    """Score activation and edge sign stability across time blocks."""

    blocks = pd.Series(time_blocks).reset_index(drop=True)
    active_positions = get_active_positions(leaf)
    membership_length = len(getattr(leaf, "timestamp_membership", []))
    if len(blocks) != membership_length:
        membership_length = max(membership_length, int(active_positions.max()) + 1 if active_positions.size else 0)
        blocks = pd.Series(np.arange(membership_length) // max(membership_length // max(min_block_count, 1), 1))
    unique_blocks = blocks.dropna().unique().tolist()
    if not unique_blocks:
        return 0.0
    block_codes = pd.Categorical(blocks).codes.astype(np.int64)
    active_blocks = active_block_count(active_positions, block_codes)
    positive_edge_blocks = (
        active_blocks
        if float(getattr(leaf, "directional_label_edge", 0.0)) > 0.0
        or float(getattr(leaf, "directional_perf_edge", 0.0)) > 0.0
        else 0
    )
    active_frequency = active_blocks / max(len(unique_blocks), 1)
    edge_frequency = positive_edge_blocks / max(active_blocks, 1)
    contribution_positive = 1.0 if float(getattr(leaf, "oof_contribution", 0.0)) > 0.0 else 0.0
    if int(bootstrap_count) <= 0:
        bootstrap_frequency = contribution_positive
    else:
        bootstrap_frequency = contribution_positive
    coverage_term = min(1.0, active_blocks / max(int(min_block_count), 1))
    return float(np.clip(active_frequency * edge_frequency * bootstrap_frequency * coverage_term, 0.0, 1.0))


def _positive_quantile(values: Sequence[float], q: float) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v) and v > 0.0], dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.quantile(arr, float(q)))


def prune_leaves(
    leaves: Sequence,
    *,
    min_stability: float = 0.50,
    absolute_min_coverage: float = 0.0025,
    label_edge_mass_quantile: float = 0.50,
    perf_edge_mass_quantile: float = 0.50,
    exceptional_contribution_share: float = 0.05,
    exceptional_edge_quantile: float = 0.90,
) -> list:
    from extreme_price_movements.performance_regimes.leaf_deduplication import (
        deduplicate_leaves_by_jaccard,
    )

    positive = [leaf for leaf in leaves if float(getattr(leaf, "oof_contribution", 0.0)) > 0.0]
    min_label_edge_mass = _positive_quantile(
        [float(getattr(leaf, "label_edge_mass", 0.0)) for leaf in positive],
        label_edge_mass_quantile,
    )
    min_perf_edge_mass = _positive_quantile(
        [float(getattr(leaf, "perf_edge_mass", 0.0)) for leaf in positive],
        perf_edge_mass_quantile,
    )
    exceptional_edge = _positive_quantile(
        [float(getattr(leaf, "positive_label_edge", 0.0)) for leaf in positive],
        exceptional_edge_quantile,
    )
    kept = []
    for leaf in leaves:
        contribution = float(getattr(leaf, "oof_contribution", 0.0))
        weighted_coverage = float(getattr(leaf, "weighted_coverage", 0.0))
        stability = float(getattr(leaf, "stability", 0.0))
        label_edge_mass = float(getattr(leaf, "label_edge_mass", 0.0))
        perf_edge_mass = float(getattr(leaf, "perf_edge_mass", 0.0))
        exceptional_keep = (
            float(getattr(leaf, "contribution_share", 0.0)) >= exceptional_contribution_share
            and float(getattr(leaf, "positive_label_edge", 0.0)) >= exceptional_edge
            and weighted_coverage >= absolute_min_coverage
        )
        keep = (
            contribution > 0.0
            and weighted_coverage >= absolute_min_coverage
            and stability >= min_stability
            and (
                label_edge_mass >= min_label_edge_mass
                or perf_edge_mass >= min_perf_edge_mass
                or exceptional_keep
            )
        )
        if keep:
            kept.append(leaf)
    return deduplicate_leaves_by_jaccard(kept)


def with_updated_leaf_scores(leaf, **updates):
    """Return a frozen dataclass leaf with selected fields updated."""

    try:
        return replace(leaf, **updates)
    except TypeError:
        for key, value in updates.items():
            setattr(leaf, key, value)
        return leaf
