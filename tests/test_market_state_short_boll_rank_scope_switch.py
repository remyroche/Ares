from __future__ import annotations

import pandas as pd

from scripts.run_market_state_short_boll_rank_scope_switch import (
    _candidate_file,
    assert_rank_contract_candidate_parity,
    build_rank_scope_priority_candidates,
    build_rank_scope_candidates,
    load_priority_action_schedule,
    load_priority_switch_schedule,
)


def _candidates(scope: str, short_boll_rank: float) -> pd.DataFrame:
    ts = pd.Timestamp("2026-06-20T00:00:00Z")
    return pd.DataFrame(
        {
            "timestamp": [ts, ts],
            "symbol": ["A/USD:USD", "B/USD:USD"],
            "side": ["short", "short"],
            "strategy_id": ["short_asset_s1", "short_boll_s1"],
            "head": ["short_asset", "short_boll"],
            "normalized_rank_score": [0.90, short_boll_rank],
            "base_strategy_threshold": [0.70, 0.70],
            "calibrated_score": [0.8, 0.8],
            "entry_price": [100.0, 100.0],
            "exit_timestamp": [ts + pd.Timedelta(hours=1), ts + pd.Timedelta(hours=1)],
            "exit_price": [99.0, 99.0],
            "net_return": [0.01, 0.02],
            "gross_return": [0.012, 0.022],
            "holding_bars": [4, 4],
            "simple_policy_exit_reason": ["tp", "tp"],
            "rank_scope": [scope, scope],
        }
    )


def test_state_switch_uses_timestamp_short_boll_when_priority_lifts_it(tmp_path) -> None:
    schedule = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-06-20T00:00:00Z")] * 2,
            "head": ["short_asset", "short_boll"],
            "portfolio_priority_adjustment": [-0.1, 0.1],
        }
    )
    schedule_path = tmp_path / "schedule.parquet"
    schedule.to_parquet(schedule_path, index=False)
    switch = load_priority_switch_schedule(schedule_path, margin=0.0)

    out, arm_schedule = build_rank_scope_candidates(
        _candidates("timestamp", 0.95),
        _candidates("global", 0.72),
        switch,
        arm="R2_state_switch_short_boll",
    )

    short_boll = out.loc[out["head"].eq("short_boll")].iloc[0]
    assert short_boll["normalized_rank_score"] == 0.95
    assert short_boll["short_boll_rank_scope"] == "timestamp_rank"
    assert arm_schedule["short_boll_rank_scope"].iloc[0] == "timestamp_rank"


def test_state_switch_uses_global_short_boll_when_priority_does_not_lift_it(tmp_path) -> None:
    schedule = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-06-20T00:00:00Z")] * 2,
            "head": ["short_asset", "short_boll"],
            "portfolio_priority_adjustment": [0.1, -0.1],
        }
    )
    schedule_path = tmp_path / "schedule.parquet"
    schedule.to_parquet(schedule_path, index=False)
    switch = load_priority_switch_schedule(schedule_path, margin=0.0)

    out, arm_schedule = build_rank_scope_candidates(
        _candidates("timestamp", 0.95),
        _candidates("global", 0.72),
        switch,
        arm="R2_state_switch_short_boll",
    )

    short_boll = out.loc[out["head"].eq("short_boll")].iloc[0]
    assert short_boll["normalized_rank_score"] == 0.72
    assert short_boll["short_boll_rank_scope"] == "global_rank"
    assert arm_schedule["short_boll_rank_scope"].iloc[0] == "global_rank"


def test_state_blend_interpolates_rank_columns_only(tmp_path) -> None:
    schedule = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-06-20T00:00:00Z")] * 2,
            "head": ["short_asset", "short_boll"],
            "portfolio_priority_adjustment": [0.0, 0.0],
        }
    )
    schedule_path = tmp_path / "schedule.parquet"
    schedule.to_parquet(schedule_path, index=False)
    switch = load_priority_switch_schedule(schedule_path, margin=0.0, blend_scale=0.0)

    out, arm_schedule = build_rank_scope_candidates(
        _candidates("timestamp", 0.95),
        _candidates("global", 0.75),
        switch.assign(short_boll_timestamp_weight=0.25),
        arm="R3_state_blended_short_boll",
    )

    short_boll = out.loc[out["head"].eq("short_boll")].iloc[0]
    assert short_boll["short_boll_rank_scope"] == "state_blend"
    assert short_boll["normalized_rank_score"] == 0.80
    assert short_boll["net_return"] == 0.02
    assert short_boll["simple_policy_exit_reason"] == "tp"
    assert arm_schedule["short_boll_timestamp_weight"].iloc[0] == 0.25


def test_composed_rank_scope_priority_candidates_apply_priority_after_rank_blend(tmp_path) -> None:
    switch = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-06-20T00:00:00Z")],
            "short_boll_rank_scope": ["state_blend"],
            "short_boll_timestamp_weight": [0.25],
        }
    )
    priority = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-06-20T00:00:00Z")] * 2,
            "head": ["short_asset", "short_boll"],
            "portfolio_priority_adjustment": [-0.1, 0.2],
            "portfolio_priority_multiplier": [0.9, 1.2],
            "priority_arm": ["test", "test"],
        }
    )

    out, schedule, coverage = build_rank_scope_priority_candidates(
        _candidates("timestamp", 0.95),
        _candidates("global", 0.75),
        switch,
        priority,
        rank_arm="R3_state_blended_short_boll",
        combo_arm="R5_state_blended_plus_priority",
    )

    short_boll = out.loc[out["head"].eq("short_boll")].iloc[0]
    assert short_boll["normalized_rank_score"] == 0.80
    assert short_boll["portfolio_priority_adjustment"] == 0.2
    assert short_boll["portfolio_priority_multiplier"] == 1.2
    assert short_boll["rank_scope_arm"] == "R5_state_blended_plus_priority"
    assert coverage["coverage"] == 1.0
    assert schedule["priority_actions_applied"].eq(True).all()


def test_load_priority_action_schedule_rejects_duplicates(tmp_path) -> None:
    schedule = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-06-20T00:00:00Z")] * 2,
            "head": ["short_boll", "short_boll"],
            "portfolio_priority_adjustment": [0.1, 0.2],
        }
    )
    path = tmp_path / "priority.parquet"
    schedule.to_parquet(path, index=False)

    try:
        load_priority_action_schedule(path)
    except ValueError as exc:
        assert "duplicate timestamp/head" in str(exc)
    else:
        raise AssertionError("expected duplicate schedule failure")


def test_candidate_file_prefers_broad_ledger(tmp_path) -> None:
    simple = tmp_path / "simple_policy_optimiser"
    simple.mkdir()
    broad = simple / "simple_policy_candidates_broad.parquet"
    deployable = simple / "simple_policy_candidates.parquet"
    broad.write_bytes(b"broad")
    deployable.write_bytes(b"deployable")

    assert _candidate_file(simple) == broad


def test_rank_contract_candidate_parity_rejects_missing_short_boll_keys() -> None:
    t1 = _candidates("timestamp", 0.95)
    global_candidates = _candidates("global", 0.72).iloc[:1].copy()

    try:
        assert_rank_contract_candidate_parity(t1, global_candidates)
    except ValueError as exc:
        assert "missing from global ledger" in str(exc)
    else:
        raise AssertionError("expected candidate parity failure")


def test_rank_contract_candidate_parity_rejects_non_rank_outcome_mismatch() -> None:
    t1 = _candidates("timestamp", 0.95)
    global_candidates = _candidates("global", 0.72)
    mask = global_candidates["head"].eq("short_boll")
    global_candidates.loc[mask, "net_return"] = -0.05

    try:
        assert_rank_contract_candidate_parity(t1, global_candidates)
    except ValueError as exc:
        assert "non-rank column mismatch: net_return" in str(exc)
    else:
        raise AssertionError("expected candidate parity failure")
