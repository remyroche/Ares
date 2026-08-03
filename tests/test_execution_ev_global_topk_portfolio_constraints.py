from __future__ import annotations

from dataclasses import replace

import pandas as pd

from extreme_price_movements.portfolio_policy_replay import PortfolioPolicyParams
from scripts.ablate_execution_ev_global_topk_portfolio_constraints import (
    attach_oof_fold,
    build_constraint_arms,
    evaluate_arm,
)


def _candidates() -> pd.DataFrame:
    timestamp = pd.Timestamp("2026-06-01 00:00", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": [timestamp, timestamp, timestamp],
            "symbol": ["A", "B", "C"],
            "side": ["long", "long", "short"],
            "strategy_id": ["execution_ev_residual"] * 3,
            "base_strategy_threshold": [0.0] * 3,
            "calibrated_score": [0.9, 0.8, 0.7],
            "normalized_rank_score": [0.99, 0.98, 0.97],
            "entry_price": [1.0] * 3,
            "exit_timestamp": [timestamp + pd.Timedelta(hours=2)] * 3,
            "exit_price": [1.01] * 3,
            "net_return": [0.01, 0.01, 0.01],
            "gross_return": [0.02, 0.02, 0.02],
            "holding_bars": [2.0] * 3,
            "simple_policy_exit_reason": ["timeout"] * 3,
            "candidate_id": ["c0", "c1", "c2"],
            "oof_fold": [0, 1, 2],
            "portfolio_fixed_position_size": [10.0] * 3,
            "price_gap_bps": [0.0] * 3,
            "expected_friction_bps": [0.0] * 3,
        }
    )


def test_constraint_arms_are_one_factor_and_activate_count_constraints() -> None:
    baseline = PortfolioPolicyParams(
        enforce_position_count_cap=False,
        max_concurrent_positions=64,
        max_total_wallet_allocation_pct=0.70,
        max_concurrent_per_symbol=1,
        max_new_entries_per_bar=2,
    )
    arms = build_constraint_arms(
        baseline,
        concurrency_caps=(8,),
        wallet_caps=(0.4,),
        per_symbol_caps=(2,),
        per_side_caps=(3,),
        new_entry_caps=(1,),
    )
    by_name = {name: params for name, _, _, params in arms}
    assert by_name["concurrency_total_8"].enforce_position_count_cap
    assert by_name["concurrency_total_8"].max_concurrent_positions == 8
    assert by_name["wallet_allocation_0p4"].max_total_wallet_allocation_pct == 0.4
    assert by_name["per_symbol_2"].max_concurrent_per_symbol == 2
    assert by_name["per_side_3"].enforce_position_count_cap
    assert by_name["per_side_3"].max_concurrent_per_side == 3
    assert by_name["new_entries_per_bar_1"].max_new_entries_per_bar == 1
    assert by_name["wallet_allocation_0p4"].max_new_entries_per_bar == baseline.max_new_entries_per_bar


def test_attach_oof_fold_never_changes_the_supplied_global_book() -> None:
    candidates = _candidates().drop(columns="oof_fold")
    oof = pd.DataFrame(
        {
            "__ts__": candidates["timestamp"],
            "__symbol__": candidates["symbol"],
            "side_name": candidates["side"],
            "candidate_id": candidates["candidate_id"],
            "mapped__is_oof": [True] * len(candidates),
            "outer_fold": [0, 1, 2],
        }
    )
    attached = attach_oof_fold(
        candidates, oof, score_col="mapped", fold_col="outer_fold"
    )
    assert attached["candidate_id"].tolist() == candidates["candidate_id"].tolist()
    assert attached["oof_fold"].tolist() == [0, 1, 2]


def test_evaluate_arm_reports_month_and_latest_fold_from_fixed_candidates() -> None:
    candidates = _candidates()
    params = PortfolioPolicyParams(
        global_threshold_floor=0.0,
        max_new_entries_per_bar=3,
        max_total_wallet_allocation_pct=0.7,
        min_position_size=1.0,
        cooldown_hours_after_loss=0.0,
    )
    decisions, _, metrics, monthly, folds = evaluate_arm(
        candidates, params, initial_wallet=1_000.0, latest_fold=2
    )
    assert len(decisions) == len(candidates)
    assert metrics["trade_count"] > 0
    assert monthly["accepted_trades"].sum() == metrics["trade_count"]
    assert folds["accepted_trades"].sum() == metrics["trade_count"]
    assert folds.loc[folds["oof_fold"].eq(2), "is_latest_fold"].all()
