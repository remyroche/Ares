"""Causal cluster and residual-committee context for N5 challengers.

The builders operate on already-prequential predictions.  Outcome-bearing
state is indexed by its label-availability timestamp and joined strictly
before each decision timestamp.  Raw K9 slots are never pooled across bundle
identities: cluster history is reset for every exact Geometry/K9 bundle and
collapsed with the current row's memberships before leaving this module.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


K9_CLUSTERS = 9
CLUSTER_HORIZONS = (3, 7, 14)
# Fixed score-CDF strata make historical cluster residuals relevant to the
# current candidate's confidence regime.  They are deliberately predeclared,
# not fitted on held outcomes or cross-sectional held ranks.
SCORE_CONDITIONED_EDGES = (0.70, 0.85, 0.95)
HEAD_SURPRISE_HORIZONS = (3, 7)
HEAD_WEIGHT_HORIZON_DAYS = 28


def k9_membership_columns() -> tuple[str, ...]:
    return tuple(f"k09__cluster_{cluster:02d}__membership" for cluster in range(K9_CLUSTERS))


def _strict_window_sums(
    decision_ns: np.ndarray,
    available_ns: np.ndarray,
    values: np.ndarray,
    days: int,
) -> np.ndarray:
    """Return sums over ``[decision-days, decision)`` for every decision."""

    order = np.argsort(available_ns, kind="stable")
    available = available_ns[order]
    matrix = np.asarray(values, dtype=np.float64)[order]
    cumulative = np.vstack(
        [np.zeros((1, matrix.shape[1]), dtype=np.float64), np.cumsum(matrix, axis=0)]
    )
    right = np.searchsorted(available, decision_ns, side="left")
    left = np.searchsorted(
        available,
        decision_ns - int(pd.Timedelta(days=days).value),
        side="left",
    )
    return cumulative[right] - cumulative[left]


def cluster_recent_correctness_fields() -> tuple[str, ...]:
    suffixes = (
        "support",
        "mean_residual_bps",
        "directional_rate",
        "positive_rate",
        "positive100_rate",
        "approx_rate",
        "adverse100_rate",
        "adverse200_rate",
    )
    return tuple(
        f"cluster_recent_{days}d_{suffix}"
        for days in CLUSTER_HORIZONS
        for suffix in suffixes
    )


def cluster_score_conditioned_correctness_fields() -> tuple[str, ...]:
    """Causal K9 history fields conditional on a fixed score-CDF stratum."""

    suffixes = (
        "support",
        "mean_residual_bps",
        "positive_rate",
        "adverse100_rate",
        "adverse200_rate",
    )
    return tuple(
        f"cluster_scorecond_{days}d_{suffix}"
        for days in CLUSTER_HORIZONS
        for suffix in suffixes
    )


def build_cluster_recent_correctness(
    frame: pd.DataFrame,
    *,
    bundle_column: str = "geometry_bundle_sha256",
    decision_column: str = "__decision_ts__",
    availability_column: str = "policy_label_available_ts",
    outcome_column: str = "policy_net_bps",
    anchor_column: str = "base_anchor_bps",
    valid_column: str = "policy_path_valid",
    score_column: str = "final_score",
    shrinkage_support: float = 20.0,
    membership_power: float = 1.0,
) -> pd.DataFrame:
    """Membership-weight prior-resolved residual health inside each K9 bundle.

    ``membership_power`` is a bounded activation-sharpening transform applied
    only to the already frozen K9 posterior before soft aggregation.  It does
    not refit K9, change cluster IDs, or expose a raw cluster slot.  One is
    the unmodified posterior; values above one are explicitly versioned
    ablations for a nearly uniform posterior.
    """

    if not np.isfinite(membership_power) or membership_power < 1.0:
        raise ValueError("membership_power must be finite and at least one")

    memberships = k9_membership_columns()
    required = {
        bundle_column,
        decision_column,
        availability_column,
        outcome_column,
        anchor_column,
        valid_column,
        score_column,
        *memberships,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"cluster correctness lacks {missing}")
    decision = pd.to_datetime(frame[decision_column], utc=True, errors="raise")
    available = pd.to_datetime(frame[availability_column], utc=True, errors="coerce")
    membership = (
        frame.loc[:, list(memberships)]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(np.float64)
    )
    membership /= np.maximum(membership.sum(axis=1, keepdims=True), 1e-12)
    if membership_power != 1.0:
        membership = np.power(membership, float(membership_power))
        membership /= np.maximum(membership.sum(axis=1, keepdims=True), 1e-12)
    net = pd.to_numeric(frame[outcome_column], errors="coerce").to_numpy(float)
    anchor = pd.to_numeric(frame[anchor_column], errors="coerce").to_numpy(float)
    residual = np.clip(net - anchor, -1_000.0, 1_000.0)
    score = pd.to_numeric(frame[score_column], errors="coerce").to_numpy(float)
    valid = (
        frame[valid_column].fillna(False).astype(bool).to_numpy()
        & np.isfinite(residual)
        & available.notna().to_numpy()
    )
    output = pd.DataFrame(0.0, index=frame.index, columns=cluster_recent_correctness_fields())
    decision_ns_all = decision.astype("int64").to_numpy()
    available_ns_all = available.astype("int64").to_numpy()
    bundles = frame[bundle_column].astype("string")
    for bundle_id, index_values in bundles.groupby(bundles, sort=False).groups.items():
        if pd.isna(bundle_id):
            continue
        row = frame.index.get_indexer(index_values)
        event = row[valid[row]]
        if not len(event):
            continue
        event_membership = membership[event]
        event_residual = residual[event]
        directional_hit = (score[event] >= 0.5) == (event_residual > 0.0)
        components = np.stack(
            [
                event_membership,
                event_membership * event_residual[:, None],
                event_membership * directional_hit[:, None],
                event_membership * (event_residual > 0.0)[:, None],
                event_membership * (event_residual > 100.0)[:, None],
                event_membership * (np.abs(event_residual) <= 50.0)[:, None],
                event_membership * (event_residual <= -100.0)[:, None],
                event_membership * (event_residual <= -200.0)[:, None],
            ],
            axis=1,
        ).reshape(len(event), -1)
        global_components = np.column_stack(
            [
                np.ones(len(event)), event_residual, directional_hit.astype(float),
                event_residual > 0.0, event_residual > 100.0,
                np.abs(event_residual) <= 50.0,
                event_residual <= -100.0, event_residual <= -200.0,
            ]
        ).astype(float)
        for days in CLUSTER_HORIZONS:
            sums = _strict_window_sums(
                decision_ns_all[row], available_ns_all[event], components, days,
            ).reshape(len(row), 8, K9_CLUSTERS)
            global_sums = _strict_window_sums(
                decision_ns_all[row], available_ns_all[event], global_components, days,
            )
            current_membership = membership[row]
            support_by_cluster = sums[:, 0, :]
            effective_support = np.sum(current_membership * support_by_cluster, axis=1)
            prefix = f"cluster_recent_{days}d_"
            output.iloc[row, output.columns.get_loc(prefix + "support")] = effective_support
            global_support = global_sums[:, 0]
            for position, suffix in enumerate(
                (
                    "mean_residual_bps", "directional_rate", "positive_rate",
                    "positive100_rate", "approx_rate", "adverse100_rate",
                    "adverse200_rate",
                ),
                start=1,
            ):
                numerator = np.sum(current_membership * sums[:, position, :], axis=1)
                prior = np.divide(
                    global_sums[:, position], global_support,
                    out=np.zeros(len(row), dtype=float), where=global_support > 0.0,
                )
                value = np.divide(
                    numerator + float(shrinkage_support) * prior,
                    effective_support + float(shrinkage_support),
                    out=np.zeros_like(numerator),
                    where=global_support > 0.0,
                )
                output.iloc[row, output.columns.get_loc(prefix + suffix)] = value
    return output.astype(np.float32)


def build_cluster_score_conditioned_correctness(
    frame: pd.DataFrame,
    *,
    bundle_column: str = "geometry_bundle_sha256",
    decision_column: str = "__decision_ts__",
    availability_column: str = "policy_label_available_ts",
    outcome_column: str = "policy_net_bps",
    anchor_column: str = "base_anchor_bps",
    valid_column: str = "policy_path_valid",
    score_column: str = "final_score",
    shrinkage_support: float = 20.0,
    membership_power: float = 1.0,
) -> pd.DataFrame:
    """Soft K9 health using only prior outcomes in the candidate's score band.

    A high-conviction candidate should not inherit a cluster's residual error
    chiefly from weak-score candidates.  Score bands are fixed CDF intervals,
    while every outcome remains strictly label-available before the current
    decision.  The aggregate remains candidate-specific:
    ``sum_k membership(i,k) * state(score_band(i), k, t-)``.
    """

    if not np.isfinite(membership_power) or membership_power < 1.0:
        raise ValueError("membership_power must be finite and at least one")
    memberships = k9_membership_columns()
    required = {
        bundle_column, decision_column, availability_column, outcome_column,
        anchor_column, valid_column, score_column, *memberships,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"score-conditioned cluster correctness lacks {missing}")
    decision = pd.to_datetime(frame[decision_column], utc=True, errors="raise")
    available = pd.to_datetime(frame[availability_column], utc=True, errors="coerce")
    membership = (
        frame.loc[:, list(memberships)].apply(pd.to_numeric, errors="coerce")
        .fillna(0.0).clip(lower=0.0).to_numpy(np.float64)
    )
    membership /= np.maximum(membership.sum(axis=1, keepdims=True), 1e-12)
    if membership_power != 1.0:
        membership = np.power(membership, float(membership_power))
        membership /= np.maximum(membership.sum(axis=1, keepdims=True), 1e-12)
    net = pd.to_numeric(frame[outcome_column], errors="coerce").to_numpy(float)
    anchor = pd.to_numeric(frame[anchor_column], errors="coerce").to_numpy(float)
    residual = np.clip(net - anchor, -1_000.0, 1_000.0)
    score = pd.to_numeric(frame[score_column], errors="coerce").to_numpy(float)
    # Missing score maps to the broadest band, retaining causal coverage but
    # never claiming high-confidence history for an unknown score.
    score_band = np.digitize(np.nan_to_num(score, nan=-np.inf), SCORE_CONDITIONED_EDGES)
    valid = (
        frame[valid_column].fillna(False).astype(bool).to_numpy()
        & np.isfinite(residual)
        & available.notna().to_numpy()
    )
    output = pd.DataFrame(
        0.0, index=frame.index, columns=cluster_score_conditioned_correctness_fields(),
    )
    decision_ns = decision.astype("int64").to_numpy()
    available_ns = available.astype("int64").to_numpy()
    bundles = frame[bundle_column].astype("string")
    suffixes = (
        "mean_residual_bps", "positive_rate", "adverse100_rate", "adverse200_rate",
    )
    for bundle_id, index_values in bundles.groupby(bundles, sort=False).groups.items():
        if pd.isna(bundle_id):
            continue
        bundle_rows = frame.index.get_indexer(index_values)
        for band in range(len(SCORE_CONDITIONED_EDGES) + 1):
            rows = bundle_rows[score_band[bundle_rows] == band]
            events = rows[valid[rows]]
            if not len(rows) or not len(events):
                continue
            event_membership = membership[events]
            event_residual = residual[events]
            components = np.stack(
                [
                    event_membership,
                    event_membership * event_residual[:, None],
                    event_membership * (event_residual > 0.0)[:, None],
                    event_membership * (event_residual <= -100.0)[:, None],
                    event_membership * (event_residual <= -200.0)[:, None],
                ], axis=1,
            ).reshape(len(events), -1)
            global_components = np.column_stack([
                np.ones(len(events)), event_residual, event_residual > 0.0,
                event_residual <= -100.0, event_residual <= -200.0,
            ]).astype(float)
            for days in CLUSTER_HORIZONS:
                sums = _strict_window_sums(
                    decision_ns[rows], available_ns[events], components, days,
                ).reshape(len(rows), 5, K9_CLUSTERS)
                global_sums = _strict_window_sums(
                    decision_ns[rows], available_ns[events], global_components, days,
                )
                current_membership = membership[rows]
                support_by_cluster = sums[:, 0, :]
                effective_support = np.sum(current_membership * support_by_cluster, axis=1)
                prefix = f"cluster_scorecond_{days}d_"
                output.iloc[rows, output.columns.get_loc(prefix + "support")] = effective_support
                global_support = global_sums[:, 0]
                for position, suffix in enumerate(suffixes, start=1):
                    numerator = np.sum(current_membership * sums[:, position, :], axis=1)
                    prior = np.divide(
                        global_sums[:, position], global_support,
                        out=np.zeros(len(rows), dtype=float), where=global_support > 0.0,
                    )
                    value = np.divide(
                        numerator + float(shrinkage_support) * prior,
                        effective_support + float(shrinkage_support),
                        out=np.zeros_like(numerator), where=global_support > 0.0,
                    )
                    output.iloc[rows, output.columns.get_loc(prefix + suffix)] = value
    return output.astype(np.float32)


def residual_head_state_fields() -> tuple[str, ...]:
    base = (
        "residual_heads_frac_rank_ge_p99",
        "residual_heads_frac_rank_ge_p95",
        "residual_heads_frac_rank_ge_p90",
        "residual_heads_weighted_mean_conviction",
        "residual_heads_median_conviction",
        "residual_heads_prediction_dispersion",
        "residual_heads_prediction_std",
        "residual_heads_prediction_iqr",
        "residual_heads_prediction_mad",
        "residual_heads_agreement_entropy",
    )
    adjusted = tuple(
        field
        for days in HEAD_SURPRISE_HORIZONS
        for field in (
            f"residual_heads_mean_rank_minus_hit_surprise_{days}d",
            f"residual_heads_frac_adjusted_ge_p99_{days}d",
            f"residual_heads_frac_adjusted_ge_p95_{days}d",
            f"residual_heads_frac_adjusted_ge_p90_{days}d",
            f"residual_heads_hit_surprise_support_{days}d",
        )
    )
    return base + adjusted


def build_residual_head_state(
    frame: pd.DataFrame,
    head_rank_columns: Sequence[str],
    *,
    decision_column: str = "__decision_ts__",
    availability_column: str = "policy_label_available_ts",
    outcome_column: str = "policy_net_bps",
    anchor_column: str = "base_anchor_bps",
    valid_column: str = "policy_path_valid",
) -> pd.DataFrame:
    """Causal committee agreement and rank-minus-hit-surprise summaries."""

    heads = tuple(dict.fromkeys(map(str, head_rank_columns)))
    if len(heads) < 2:
        raise ValueError("residual committee state needs at least two head ranks")
    required = {
        decision_column,
        availability_column,
        outcome_column,
        anchor_column,
        valid_column,
        *heads,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"residual committee state lacks {missing}")
    raw_rank_frame = frame.loc[:, list(heads)].apply(pd.to_numeric, errors="coerce")
    head_available = raw_rank_frame.notna().all(axis=1).to_numpy()
    rank = (
        raw_rank_frame
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.5)
        .clip(0.0, 1.0)
        .to_numpy(np.float64)
    )
    decision = pd.to_datetime(frame[decision_column], utc=True, errors="raise")
    available = pd.to_datetime(frame[availability_column], utc=True, errors="coerce")
    residual = (
        pd.to_numeric(frame[outcome_column], errors="coerce")
        - pd.to_numeric(frame[anchor_column], errors="coerce")
    ).to_numpy(float)
    valid = (
        frame[valid_column].fillna(False).astype(bool).to_numpy()
        & np.isfinite(residual)
        & available.notna().to_numpy()
        & head_available
    )
    event_rank = rank[valid]
    actual_positive = residual[valid] > 0.0
    predicted_positive = event_rank >= 0.5
    hit = predicted_positive == actual_positive[:, None]
    expected_hit = np.maximum(event_rank, 1.0 - event_rank)
    surprise_event = hit.astype(float) - expected_hit
    support_event = np.ones_like(event_rank, dtype=float)
    decision_ns = decision.astype("int64").to_numpy()
    available_ns = available.astype("int64").to_numpy()[valid]
    surprise_by_horizon: dict[int, np.ndarray] = {}
    support_by_horizon: dict[int, np.ndarray] = {}
    for days in (*HEAD_SURPRISE_HORIZONS, HEAD_WEIGHT_HORIZON_DAYS):
        surprise_sum = _strict_window_sums(
            decision_ns, available_ns, surprise_event, days,
        )
        support = _strict_window_sums(
            decision_ns, available_ns, support_event, days,
        )
        surprise_by_horizon[days] = np.divide(
            surprise_sum,
            support,
            out=np.zeros_like(surprise_sum),
            where=support > 0.0,
        )
        support_by_horizon[days] = support

    conviction = 2.0 * rank - 1.0
    reliability = np.clip(0.5 + surprise_by_horizon[HEAD_WEIGHT_HORIZON_DAYS], 0.05, 1.0)
    reliability /= np.maximum(reliability.sum(axis=1, keepdims=True), 1e-12)
    ordered = np.sort(rank, axis=1)
    q25 = np.quantile(rank, 0.25, axis=1)
    q75 = np.quantile(rank, 0.75, axis=1)
    low = (rank < 0.10).mean(axis=1)
    high = (rank >= 0.90).mean(axis=1)
    middle = 1.0 - low - high
    agreement = np.column_stack([low, middle, high])
    entropy = -np.sum(agreement * np.log(np.maximum(agreement, 1e-12)), axis=1) / np.log(3.0)
    output = pd.DataFrame(index=frame.index)
    output["residual_heads_frac_rank_ge_p99"] = (rank >= 0.99).mean(axis=1)
    output["residual_heads_frac_rank_ge_p95"] = (rank >= 0.95).mean(axis=1)
    output["residual_heads_frac_rank_ge_p90"] = (rank >= 0.90).mean(axis=1)
    output["residual_heads_weighted_mean_conviction"] = np.sum(reliability * conviction, axis=1)
    output["residual_heads_median_conviction"] = np.median(conviction, axis=1)
    output["residual_heads_prediction_dispersion"] = ordered[:, -1] - ordered[:, 0]
    output["residual_heads_prediction_std"] = rank.std(axis=1)
    output["residual_heads_prediction_iqr"] = q75 - q25
    output["residual_heads_prediction_mad"] = np.median(
        np.abs(rank - np.median(rank, axis=1, keepdims=True)), axis=1,
    )
    output["residual_heads_agreement_entropy"] = entropy
    for days in HEAD_SURPRISE_HORIZONS:
        adjusted = np.clip(rank - surprise_by_horizon[days], 0.0, 1.0)
        output[f"residual_heads_mean_rank_minus_hit_surprise_{days}d"] = adjusted.mean(axis=1)
        for percentile in (99, 95, 90):
            output[f"residual_heads_frac_adjusted_ge_p{percentile}_{days}d"] = (
                adjusted >= percentile / 100.0
            ).mean(axis=1)
        output[f"residual_heads_hit_surprise_support_{days}d"] = support_by_horizon[days].mean(axis=1)
    return output.loc[:, list(residual_head_state_fields())].astype(np.float32)


__all__ = [
    "build_cluster_recent_correctness",
    "build_residual_head_state",
    "cluster_recent_correctness_fields",
    "k9_membership_columns",
    "residual_head_state_fields",
]
