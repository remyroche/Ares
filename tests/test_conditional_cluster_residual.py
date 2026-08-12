from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.conditional_cluster_residual import (
    ClusterContract,
    cluster_condition_economics,
    materialize_cluster_features,
    select_oof_path_rows,
    soft_cluster_residual_target,
)


def test_select_oof_path_rows_rejects_meta_train_by_default() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["a", "a", "b"],
            "__ts__": pd.to_datetime(["2024-01-01T00:00Z"] * 3),
            "fold": ["f1", "f2", "f1"],
            "meta_partition": ["meta_train", "test", "meta_train"],
        }
    )
    selected, audit = select_oof_path_rows(frame)
    assert selected.candidate_id.tolist() == ["a"]
    assert selected.meta_partition.tolist() == ["test"]
    assert int(audit.loc[audit.source_partition.eq("__summary__"), "selected_rows"].iloc[0]) == 1


def test_soft_cluster_memberships_are_normalized_over_represented_mass() -> None:
    fields = ["family_a", "family_b"]
    frame = pd.DataFrame(
        {
            "family_a": [0.4, -0.2],
            "family_b": [0.1, 0.3],
            "family_abs_share__family_a": [0.8, 0.0],
            "family_abs_share__family_b": [0.2, 1.0],
            "family_confidence_share__family_a": [0.6, 0.0],
            "family_confidence_share__family_b": [0.1, 0.9],
        }
    )
    contracts = [
        ClusterContract("c0", ("family_a",), (0,), 0.0),
        ClusterContract("c1", ("family_b",), (1,), 0.0),
    ]
    out = materialize_cluster_features(frame, contracts, family_fields=fields)
    memberships = out[["cluster__c0__membership", "cluster__c1__membership"]].to_numpy()
    np.testing.assert_allclose(memberships.sum(axis=1), 1.0)
    np.testing.assert_allclose(soft_cluster_residual_target(np.array([10.0, -20.0]), memberships[:, 0]), [8.0, 0.0])


def test_condition_economics_uses_training_edges_and_reports_net_delta() -> None:
    train = pd.DataFrame({"context": np.arange(100, dtype=float), "residual_bps": np.linspace(-20, 20, 100), "net_bps": np.linspace(80, 120, 100)})
    test = pd.DataFrame({"context": np.arange(100, dtype=float), "residual_bps": np.linspace(20, 60, 100), "net_bps": np.linspace(120, 160, 100)})
    cluster = pd.DataFrame({"cluster__c0__membership": np.ones(100, dtype=float)})
    out = cluster_condition_economics(
        test,
        cluster,
        cluster_ids=["c0"],
        context_fields=["context"],
        train_frame=train,
        bins=5,
    )
    assert len(out) == 5
    assert out["weighted_net_bps"].notna().all()
    assert (out["weighted_net_bps"] > 100).all()
