from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.trust_sizing_ablation import ParentExpectation
from scripts.run_strict_r3_n5_bundle_local_recurrence import (
    ROLE_OUTPUTS,
    _activation_aggregates,
    _cluster_statistics,
    _materialize_roles,
    _role_map,
)


def _cluster_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "final_score": [0.1, 0.2, 0.3, 0.4],
            "policy_net_bps": [100.0, -100.0, 50.0, -50.0],
        }
    )
    for cluster in range(9):
        frame[f"k09__cluster_{cluster:02d}__membership"] = 0.0
        frame[f"k09__cluster_{cluster:02d}__negative_distance"] = 0.0
        frame[f"k09__cluster_{cluster:02d}__confidence"] = 0.0
    frame.loc[[0, 2], "k09__cluster_00__membership"] = 1.0
    frame.loc[[0, 2], "k09__cluster_00__confidence"] = 0.8
    frame.loc[[1, 3], "k09__cluster_01__membership"] = 1.0
    frame.loc[[1, 3], "k09__cluster_01__confidence"] = 0.4
    return frame


def test_roles_are_semantic_and_activation_aggregate_sums_role_contributions() -> None:
    train = _cluster_frame()
    parent = ParentExpectation(
        edges=np.asarray([], dtype=float),
        means=np.asarray([0.0], dtype=float),
        global_mean=0.0,
    )
    statistics = _cluster_statistics(train, parent)
    ordering = _role_map(statistics)
    assert ordering[0] == 0
    assert ordering[-1] == 1

    held = train.iloc[[0]].copy()
    held.loc[:, "k09__cluster_00__membership"] = 0.75
    held.loc[:, "k09__cluster_01__membership"] = 0.25
    held.loc[:, "k09__cluster_00__confidence"] = 0.8
    held.loc[:, "k09__cluster_01__confidence"] = 0.4
    roles, role_fields = _materialize_roles(held, ordering, statistics)
    aggregate, aggregate_fields, _ = _activation_aggregates(
        train,
        held,
        parent,
        cluster_statistics=statistics,
    )

    assert len(role_fields) == 9 * len(ROLE_OUTPUTS)
    assert not any("cluster_" in field for field in role_fields)
    for category, aggregate_field in zip(ROLE_OUTPUTS, aggregate_fields):
        role_sum = roles[
            [field for field in role_fields if field.endswith(category)]
        ].sum(axis=1)
        np.testing.assert_allclose(role_sum, aggregate[aggregate_field])

    np.testing.assert_allclose(
        aggregate["bundle_activation_expected_residual_bps"],
        [37.5],
    )
    np.testing.assert_allclose(
        aggregate["bundle_activation_effective_support"],
        [0.5],
    )
    np.testing.assert_allclose(
        aggregate["bundle_activation_confidence"],
        [0.7],
    )
