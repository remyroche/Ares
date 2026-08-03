from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_meaningful_mfe_exact_grid_base_residual_sensitivity import (
    ARMS,
    RESIDUAL_SHRINKAGE,
    TASKS,
    TRANSFER_NAMES,
    _crossfit_plan,
    _economic_rows,
    config_routed_feature_pools,
)
from scripts.run_meaningful_mfe_exact_grid_reset import TRANSFER_SPECS


def test_config_routing_preserves_base_and_explicit_disjoint_meta_sensitivity() -> None:
    base_features = [f"base_{index}" for index in range(8)]
    meta_features = [f"meta_{index}" for index in range(8)]
    base, meta, report = config_routed_feature_pools(
        [*base_features, "overlap", *meta_features, "unused"],
        configured_base_by_side={
            "long": [
                *[f"capture_candidate__{value}" for value in base_features],
                "overlap",
            ],
            "short": [*base_features, "overlap"],
        },
        configured_meta=["overlap", *[f"capture_candidate__{value}" for value in meta_features]],
    )

    assert base == {
        "long": [*base_features, "overlap"],
        "short": [*base_features, "overlap"],
    }
    assert meta == ["overlap", *meta_features]
    assert report["base_meta_overlap_by_side"] == {
        "long": ["overlap"],
        "short": ["overlap"],
    }
    assert report["disjoint_meta_by_side"] == {
        "long": meta_features,
        "short": meta_features,
    }


def test_config_routing_rejects_insufficient_disjoint_meta_pool() -> None:
    try:
        config_routed_feature_pools(
            [f"feature_{index}" for index in range(12)],
            configured_base_by_side={
                "long": [f"feature_{index}" for index in range(8)],
                "short": [f"feature_{index}" for index in range(8)],
            },
            configured_meta=[f"feature_{index}" for index in range(8)],
        )
    except ValueError as error:
        assert "disjoint-meta" in str(error)
    else:
        raise AssertionError("empty disjoint-meta sensitivity was accepted")


def test_crossfit_plan_is_temporal_purged_and_has_no_identity_overlap() -> None:
    timestamps = pd.date_range("2026-06-01", periods=80, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "__ts__": timestamps,
            # These timestamps deliberately make the explicit decision purge,
            # rather than merely resolution order, observable in the test.
            "label_resolution_utc": timestamps - pd.Timedelta(minutes=1),
            "execution_decision_utc": timestamps - pd.Timedelta(hours=13),
        }
    )

    plan = _crossfit_plan(
        frame, folds=4, min_train_rows=6, min_validation_rows=4
    )

    assert len(plan) == 3
    for item in plan:
        train = np.asarray(item["train"], dtype=int)
        validation = np.asarray(item["validation"], dtype=int)
        start = item["validation_start"]
        assert not set(train).intersection(validation)
        assert (frame.iloc[train]["label_resolution_utc"] < start).all()
        assert (
            frame.iloc[train]["execution_decision_utc"]
            < start - pd.Timedelta(hours=12)
        ).all()
        assert (frame.iloc[validation]["__ts__"] >= start).all()


def test_required_arms_targets_and_exact_reset_transfer_contract_are_frozen() -> None:
    assert ARMS == (
        "base_only",
        "monolithic_union",
        "configured_base_residual",
        "disjoint_meta_sensitivity",
    )
    assert TASKS == ("any_touch", "clean_first")
    available = tuple(spec.name for spec in TRANSFER_SPECS)
    assert all(name in available for name in TRANSFER_NAMES)
    assert TRANSFER_NAMES == (
        "may_to_june",
        "june_to_july",
        "july_to_june_matched",
    )
    assert RESIDUAL_SHRINKAGE == {"long": 0.5, "short": 0.25}


def test_shared_economics_contract_accepts_base_residual_scored_rows() -> None:
    rows = 20
    frame = pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-07-01", periods=rows, freq="h", tz="UTC"),
            "__symbol__": ["A"] * rows,
            "side_name": ["long"] * 10 + ["short"] * 10,
            "candidate_id": [f"id-{index:02d}" for index in range(rows)],
            "execution_net_ev_12h": np.linspace(-0.02, 0.02, rows),
            "execution_gross_ev_12h": np.linspace(-0.01, 0.03, rows),
            "execution_cost_return": np.full(rows, 0.01),
            "any_touch": np.tile([0, 1], 10),
            "clean_first": np.tile([0, 1, 0, 0], 5),
            "timeout": np.tile([0, 0, 1, 0], 5),
        }
    )
    for task in TASKS:
        for arm in ARMS:
            frame[f"score_{task}_{arm}"] = np.linspace(0.0, 1.0, rows)

    result = _economic_rows(frame, "toy")

    assert len(result) == len(TASKS) * len(ARMS) * 3
    assert all(row["evaluation"] == "toy" for row in result)
