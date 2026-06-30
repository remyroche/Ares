from __future__ import annotations

import pandas as pd
import pytest

from scripts.build_market_state_rank_reference_router import (
    build_rank_reference_router_schedule,
)
from scripts.run_market_state_short_boll_rank_scope_switch import (
    load_rank_reference_router_schedule,
)


def _priority_schedule() -> pd.DataFrame:
    ts0 = pd.Timestamp("2026-06-20T00:00:00Z")
    ts1 = pd.Timestamp("2026-06-20T01:00:00Z")
    return pd.DataFrame(
        {
            "timestamp": [ts0, ts0, ts1, ts1],
            "head": ["short_asset", "short_boll", "short_asset", "short_boll"],
            "portfolio_priority_adjustment": [0.10, -0.10, -0.20, 0.20],
        }
    )


def test_router_schedule_hard_routes_from_priority_diff() -> None:
    out = build_rank_reference_router_schedule(
        _priority_schedule(),
        blend_scale=0.0,
        margin=0.0,
    )

    assert out["short_boll_rank_scope"].tolist() == ["global_rank", "timestamp_rank"]
    assert out["short_boll_timestamp_weight"].tolist() == [0.0, 1.0]
    assert out["router_layer"].unique().tolist() == ["rank_reference_before_threshold"]
    assert out["promotion_status"].unique().tolist() == ["shadow_only"]
    assert out["changes_thresholds"].eq(False).all()
    assert out["changes_scores"].eq(False).all()


def test_router_schedule_sigmoid_weight_is_bounded() -> None:
    out = build_rank_reference_router_schedule(
        _priority_schedule(),
        blend_scale=0.10,
        min_timestamp_weight=0.20,
        max_timestamp_weight=0.80,
    )

    assert out["short_boll_timestamp_weight"].between(0.20, 0.80).all()
    assert out["short_boll_timestamp_weight"].iloc[0] < 0.50
    assert out["short_boll_timestamp_weight"].iloc[1] > 0.50


def test_router_schedule_rejects_duplicate_head_timestamp_rows() -> None:
    duped = pd.concat([_priority_schedule(), _priority_schedule().iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate timestamp/head"):
        build_rank_reference_router_schedule(duped)


def test_rank_scope_replay_loader_accepts_formal_router_schedule(tmp_path) -> None:
    schedule = build_rank_reference_router_schedule(_priority_schedule())
    path = tmp_path / "router.parquet"
    schedule.to_parquet(path, index=False)

    loaded = load_rank_reference_router_schedule(path)

    assert loaded["short_boll_timestamp_weight"].between(0.0, 1.0).all()
    assert set(loaded["short_boll_rank_scope"]) == {"global_rank", "timestamp_rank"}


def test_rank_scope_replay_loader_rejects_out_of_range_weights(tmp_path) -> None:
    schedule = build_rank_reference_router_schedule(_priority_schedule())
    schedule.loc[0, "short_boll_timestamp_weight"] = 1.25
    path = tmp_path / "router.parquet"
    schedule.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="weights must be in"):
        load_rank_reference_router_schedule(path)
