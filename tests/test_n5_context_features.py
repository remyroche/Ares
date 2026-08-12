from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.n5_context_features import (
    build_cluster_recent_correctness,
    build_cluster_score_conditioned_correctness,
    build_residual_head_state,
    cluster_recent_correctness_fields,
    cluster_score_conditioned_correctness_fields,
    k9_membership_columns,
    residual_head_state_fields,
)


def _frame() -> pd.DataFrame:
    timestamp = pd.to_datetime(
        [
            "2026-01-01 00:00Z", "2026-01-01 00:00Z",
            "2026-01-03 00:00Z", "2026-01-03 00:00Z",
            "2026-01-06 00:00Z", "2026-01-06 00:00Z",
            "2026-01-10 00:00Z", "2026-01-10 00:00Z",
        ],
        utc=True,
    )
    frame = pd.DataFrame(
        {
            "candidate_id": [f"c{index}" for index in range(len(timestamp))],
            "__decision_ts__": timestamp,
            "policy_label_available_ts": timestamp + pd.Timedelta(hours=12),
            "policy_path_valid": True,
            "policy_net_bps": [200.0, -100.0, 300.0, -200.0, 150.0, 50.0, -50.0, 250.0],
            "base_anchor_bps": 50.0,
            "final_score": [0.9, 0.8, 0.9, 0.2, 0.8, 0.4, 0.9, 0.3],
            "geometry_bundle_sha256": ["a"] * 6 + ["b"] * 2,
            "conditional_head__h0__rank": [0.99, 0.90, 0.96, 0.20, 0.93, 0.40, 0.99, 0.80],
            "conditional_head__h1__rank": [0.98, 0.10, 0.91, 0.30, 0.85, 0.70, 0.95, 0.20],
        }
    )
    for cluster, column in enumerate(k9_membership_columns()):
        frame[column] = 1.0 if cluster == 0 else 0.0
    return frame


def test_residual_head_state_has_requested_raw_and_surprise_adjusted_fields() -> None:
    frame = _frame()
    fields = ["conditional_head__h0__rank", "conditional_head__h1__rank"]
    state = build_residual_head_state(frame, fields)
    assert tuple(state.columns) == residual_head_state_fields()
    assert state.loc[0, "residual_heads_frac_rank_ge_p95"] == 1.0
    assert state.loc[0, "residual_heads_mean_rank_minus_hit_surprise_3d"] == np.float32(0.985)
    assert state.loc[4, "residual_heads_hit_surprise_support_3d"] > 0.0


def test_head_surprise_is_strictly_prior_resolved() -> None:
    original = _frame()
    changed = original.copy()
    changed.loc[changed["__decision_ts__"].ge("2026-01-06"), "policy_net_bps"] = -999.0
    fields = ["conditional_head__h0__rank", "conditional_head__h1__rank"]
    first = build_residual_head_state(original, fields)
    second = build_residual_head_state(changed, fields)
    early = original["__decision_ts__"].le("2026-01-06")
    np.testing.assert_allclose(first.loc[early], second.loc[early])


def test_cluster_recent_correctness_does_not_cross_geometry_bundles() -> None:
    frame = _frame()
    state = build_cluster_recent_correctness(frame, shrinkage_support=1.0)
    assert tuple(state.columns) == cluster_recent_correctness_fields()
    # Bundle B begins on 10 January and must not inherit bundle A outcomes.
    assert state.loc[6, "cluster_recent_14d_support"] == 0.0
    assert state.loc[7, "cluster_recent_14d_support"] == 0.0


def test_cluster_recent_correctness_is_future_mutation_invariant() -> None:
    original = _frame()
    changed = original.copy()
    changed.loc[changed["__decision_ts__"].ge("2026-01-06"), "policy_net_bps"] = 999.0
    first = build_cluster_recent_correctness(original, shrinkage_support=1.0)
    second = build_cluster_recent_correctness(changed, shrinkage_support=1.0)
    early = original["__decision_ts__"].le("2026-01-06")
    np.testing.assert_allclose(first.loc[early], second.loc[early])


def test_cluster_correctness_shrinks_toward_causal_global_prior() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["positive", "negative", "current"],
            "__decision_ts__": pd.to_datetime(
                [
                    "2026-01-01 00:00Z", "2026-01-01 00:00Z",
                    "2026-01-03 00:00Z",
                ], utc=True,
            ),
            "policy_label_available_ts": pd.to_datetime(
                ["2026-01-01 12:00Z", "2026-01-01 12:00Z", "2026-01-03 12:00Z"], utc=True,
            ),
            "policy_path_valid": True,
            "policy_net_bps": [200.0, -100.0, 0.0],
            "base_anchor_bps": 50.0,
            "final_score": [0.9, 0.1, 0.9],
            "geometry_bundle_sha256": "same",
        }
    )
    for cluster, column in enumerate(k9_membership_columns()):
        frame[column] = 0.0
    frame.loc[[0, 2], k9_membership_columns()[0]] = 1.0
    frame.loc[1, k9_membership_columns()[1]] = 1.0
    state = build_cluster_recent_correctness(frame, shrinkage_support=1.0)
    assert state.loc[2, "cluster_recent_3d_mean_residual_bps"] == 75.0


def test_cluster_correctness_is_soft_k9_membership_weighted() -> None:
    frame = _frame().iloc[:3].copy()
    memberships = k9_membership_columns()
    frame.loc[:, list(memberships)] = 0.0
    # The two resolved rows belong to opposite frozen K9 clusters.  The held
    # candidate receives their state by soft membership, never a hard label.
    frame.loc[0, memberships[0]] = 1.0
    frame.loc[1, memberships[1]] = 1.0
    frame.loc[2, memberships[0]] = 0.25
    frame.loc[2, memberships[1]] = 0.75
    frame.loc[2, "__decision_ts__"] = pd.Timestamp("2026-01-03T00:00:00Z")
    state = build_cluster_recent_correctness(frame, shrinkage_support=0.0)
    # Prior residuals are +150 and -150 bps.  A 25/75 membership blend is -75.
    assert state.loc[2, "cluster_recent_3d_mean_residual_bps"] == np.float32(-75.0)
    assert state.loc[2, "cluster_recent_3d_support"] == np.float32(1.0)


def test_score_conditioned_cluster_history_uses_candidate_band_and_membership() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["high", "low", "held"],
            "__decision_ts__": pd.to_datetime(
                ["2026-01-01 00:00Z", "2026-01-01 00:00Z", "2026-01-03 00:00Z"], utc=True,
            ),
            "policy_label_available_ts": pd.to_datetime(
                ["2026-01-01 12:00Z", "2026-01-01 12:00Z", "2026-01-03 12:00Z"], utc=True,
            ),
            "policy_path_valid": True,
            "policy_net_bps": [150.0, -50.0, 0.0],
            "base_anchor_bps": 50.0,
            "final_score": [0.98, 0.50, 0.98],
            "geometry_bundle_sha256": "same",
        }
    )
    for cluster, column in enumerate(k9_membership_columns()):
        frame[column] = 0.0
    frame.loc[0, k9_membership_columns()[0]] = 1.0
    frame.loc[1, k9_membership_columns()[1]] = 1.0
    frame.loc[2, k9_membership_columns()[0]] = 0.25
    frame.loc[2, k9_membership_columns()[1]] = 0.75
    state = build_cluster_score_conditioned_correctness(frame, shrinkage_support=0.0)
    assert tuple(state.columns) == cluster_score_conditioned_correctness_fields()
    # The held high-score row receives only the high-score cluster-0 residual
    # (+100 bps), even though it is also 75% associated with cluster 1.
    assert state.loc[2, "cluster_scorecond_3d_mean_residual_bps"] == np.float32(100.0)
    assert state.loc[2, "cluster_scorecond_3d_support"] == np.float32(0.25)
