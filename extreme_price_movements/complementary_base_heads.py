"""Shared causal handoff utilities for complementary base-head experiments.

The module deliberately contains no label construction or fold policy.  It
keeps two otherwise easy-to-miss contracts in one tested place:

* a held row's percentile is measured against the *training* score CDF, not
  against its own month; and
* committee agreement is a same-row transform of those causal percentiles.

The resulting fields are safe inputs to a residual learner once its base-head
scores are themselves OOF/frozen.  They are not outcome-derived features.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


AGREEMENT_FEATURES: tuple[str, ...] = (
    "base_heads_frac_rank_ge_p99",
    "base_heads_frac_rank_ge_p95",
    "base_heads_frac_rank_ge_p90",
    "base_heads_weighted_mean_conviction",
    "base_heads_median_conviction",
    "base_heads_prediction_dispersion",
    "base_heads_prediction_std",
    "base_heads_prediction_iqr",
    "base_heads_agreement_entropy",
)


def causal_rank_norm(train_scores: Sequence[float], held_scores: Sequence[float]) -> np.ndarray:
    """Map held scores to their empirical CDF under prior training scores."""
    reference = np.sort(np.asarray(train_scores, dtype=float)[np.isfinite(train_scores)])
    held = np.asarray(held_scores, dtype=float)
    if reference.size == 0:
        return np.full(held.shape, 0.5, dtype=np.float32)
    result = np.searchsorted(reference, held, side="right") / float(reference.size)
    result[~np.isfinite(held)] = 0.5
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def _normalised_weights(columns: Sequence[str], weights: Mapping[str, float] | None) -> np.ndarray:
    raw = np.asarray([1.0 if weights is None else float(weights.get(column, 1.0)) for column in columns], dtype=float)
    raw[~np.isfinite(raw) | (raw < 0.0)] = 0.0
    if raw.sum() <= 0.0:
        raw[:] = 1.0
    return raw / raw.sum()


def agreement_features(
    frame: pd.DataFrame,
    rank_columns: Sequence[str],
    *,
    weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Create deterministic committee agreement/disagreement fields.

    ``rank_columns`` must contain rank-normalised base scores in ``[0, 1]``.
    Entropy is computed over each head's low/middle/high conviction bucket and
    normalised by ``log(3)``; zero means unanimous and one means maximum
    disagreement across those three bins.
    """
    if not rank_columns:
        raise ValueError("at least one rank-normalised base score is required")
    missing = [column for column in rank_columns if column not in frame]
    if missing:
        raise KeyError(f"missing base rank columns: {missing}")
    values = frame.loc[:, list(rank_columns)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    values = np.where(np.isfinite(values), np.clip(values, 0.0, 1.0), 0.5)
    conviction = 2.0 * values - 1.0
    w = _normalised_weights(rank_columns, weights)
    ordered = np.sort(values, axis=1)
    count = values.shape[1]
    low = (values < 0.10).sum(axis=1).astype(float)
    high = (values >= 0.90).sum(axis=1).astype(float)
    middle = count - low - high
    memberships = np.column_stack((low, middle, high)) / float(count)
    entropy = -(memberships * np.log(np.maximum(memberships, 1e-12))).sum(axis=1) / np.log(3.0)
    q75 = np.quantile(values, 0.75, axis=1)
    q25 = np.quantile(values, 0.25, axis=1)
    result = pd.DataFrame(index=frame.index)
    result["base_heads_frac_rank_ge_p99"] = (values >= 0.99).mean(axis=1).astype(np.float32)
    result["base_heads_frac_rank_ge_p95"] = (values >= 0.95).mean(axis=1).astype(np.float32)
    result["base_heads_frac_rank_ge_p90"] = (values >= 0.90).mean(axis=1).astype(np.float32)
    result["base_heads_weighted_mean_conviction"] = (conviction @ w).astype(np.float32)
    result["base_heads_median_conviction"] = np.median(conviction, axis=1).astype(np.float32)
    result["base_heads_prediction_dispersion"] = (ordered[:, -1] - ordered[:, 0]).astype(np.float32)
    result["base_heads_prediction_std"] = values.std(axis=1, ddof=0).astype(np.float32)
    result["base_heads_prediction_iqr"] = (q75 - q25).astype(np.float32)
    result["base_heads_agreement_entropy"] = entropy.astype(np.float32)
    return result


def global_tail_metrics(
    frame: pd.DataFrame,
    *,
    score_column: str,
    net_column: str = "exact_net_bps",
    gross_column: str = "exact_gross_bps",
    tails: Sequence[float] = (0.01, 0.02, 0.05),
) -> dict[str, float]:
    """Return global (never per-timestamp) economic tail metrics."""
    required = {"candidate_id", score_column, net_column, gross_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"tail metric input missing {missing}")
    ranked = frame.sort_values([score_column, "candidate_id"], ascending=[False, True], kind="stable")
    output: dict[str, float] = {}
    for tail in tails:
        n = max(1, int(np.ceil(len(ranked) * float(tail))))
        chosen = ranked.head(n)
        key = int(round(tail * 100))
        output[f"top{key}_rows"] = float(n)
        output[f"top{key}_net_bps"] = float(pd.to_numeric(chosen[net_column], errors="coerce").mean())
        output[f"top{key}_gross_bps"] = float(pd.to_numeric(chosen[gross_column], errors="coerce").mean())
    return output


__all__ = [
    "AGREEMENT_FEATURES",
    "agreement_features",
    "causal_rank_norm",
    "global_tail_metrics",
]
