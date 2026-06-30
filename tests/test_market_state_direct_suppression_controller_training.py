from __future__ import annotations

import pandas as pd

from scripts.train_market_state_direct_suppression_controller import (
    chronological_fold_plan,
    evaluate_policy_grid,
    prepare_training_frame,
)


def _ledger_row(fold: int, timestamp: str, key: str, utility: float) -> dict:
    return {
        "fold": fold,
        "timestamp": pd.Timestamp(timestamp, tz="UTC"),
        "controller_arm": "S1_observed_axes_shared_response",
        "decision_key": key,
        "head": "short_boll" if utility > 0 else "short_asset",
        "side": "short",
        "strategy_id": f"strategy_{key}",
        "rank_score": 0.72,
        "base_threshold": 0.70,
        "rank_minus_base_threshold": 0.02,
        "frontier_distance": 0.02,
        "required_threshold_raise_to_suppress": 0.02,
        "risk_severity": 0.5,
        "prediction_coverage": 1.0,
        "state_ood_score_mean": 0.0,
        "state_ood_score_max": 0.0,
        "state_ood_share": 0.0,
        "state_low_input_coverage_share": 0.0,
        "mean_pred_utility": utility,
        "mean_pred_lcb": utility - 0.01,
        "mean_pred_full_sl": 0.5 if utility > 0 else 0.1,
        "mean_pred_timeout": 0.1,
        "frontier_candidate_count": 1,
        "accepted_frontier_candidate_count": 1,
        "frontier_sample_weight": 1.0,
        "loss_avoided_if_suppressed": max(utility, 0.0),
        "winner_pnl_sacrificed_if_suppressed": max(-utility, 0.0),
        "direct_defensive_utility": utility,
        "direct_suppression_profitable": utility > 0,
        "direct_suppression_full_sl": utility > 0,
        "direct_suppression_timeout": False,
    }


def test_chronological_fold_plan_uses_only_prior_folds() -> None:
    raw = pd.DataFrame(
        [
            _ledger_row(1, "2026-05-01T00:00:00Z", "a", 0.02),
            _ledger_row(1, "2026-05-01T01:00:00Z", "b", -0.01),
            _ledger_row(2, "2026-05-02T00:00:00Z", "c", 0.03),
            _ledger_row(2, "2026-05-02T01:00:00Z", "d", -0.02),
            _ledger_row(3, "2026-05-03T00:00:00Z", "e", 0.04),
        ]
    )
    frame = prepare_training_frame(raw)

    plan = chronological_fold_plan(frame)

    assert [item["valid_fold"] for item in plan] == [2, 3]
    assert plan[0]["train_folds"] == [1]
    assert plan[1]["train_folds"] == [1, 2]
    for item in plan:
        train_timestamps = set(frame.loc[item["train_index"], "timestamp"])
        valid_timestamps = set(frame.loc[item["valid_index"], "timestamp"])
        assert train_timestamps.isdisjoint(valid_timestamps)
        assert max(train_timestamps) < min(valid_timestamps)


def test_policy_grid_respects_max_delta_and_direct_utility_gate() -> None:
    pred = pd.DataFrame(
        [
            {
                "prediction_available": True,
                "controller_arm": "S1",
                "valid_fold": 2,
                "decision_key": "win1",
                "head": "short_boll",
                "required_threshold_raise_to_suppress": 0.02,
                "pred_suppression_profit_prob": 0.90,
                "pred_direct_utility": 0.02,
                "loss_avoided_if_suppressed": 0.03,
                "winner_pnl_sacrificed_if_suppressed": 0.0,
                "direct_defensive_utility": 0.03,
            },
            {
                "prediction_available": True,
                "controller_arm": "S1",
                "valid_fold": 3,
                "decision_key": "win2",
                "head": "short_boll",
                "required_threshold_raise_to_suppress": 0.04,
                "pred_suppression_profit_prob": 0.80,
                "pred_direct_utility": 0.015,
                "loss_avoided_if_suppressed": 0.02,
                "winner_pnl_sacrificed_if_suppressed": 0.0,
                "direct_defensive_utility": 0.02,
            },
            {
                "prediction_available": True,
                "controller_arm": "S1",
                "valid_fold": 3,
                "decision_key": "too_far",
                "head": "short_boll",
                "required_threshold_raise_to_suppress": 0.12,
                "pred_suppression_profit_prob": 0.99,
                "pred_direct_utility": 0.10,
                "loss_avoided_if_suppressed": 1.0,
                "winner_pnl_sacrificed_if_suppressed": 0.0,
                "direct_defensive_utility": 1.0,
            },
            {
                "prediction_available": True,
                "controller_arm": "S2",
                "valid_fold": 2,
                "decision_key": "loser",
                "head": "short_asset",
                "required_threshold_raise_to_suppress": 0.02,
                "pred_suppression_profit_prob": 0.90,
                "pred_direct_utility": 0.02,
                "loss_avoided_if_suppressed": 0.0,
                "winner_pnl_sacrificed_if_suppressed": 0.04,
                "direct_defensive_utility": -0.04,
            },
        ]
    )

    grid, selection = evaluate_policy_grid(pred, max_delta=0.08, min_suppressed_rows=2)

    assert selection["selected_arm"] == "S1"
    selected = selection["selected_policy"]
    assert selected["suppressed_rows"] == 2
    assert selected["suppressed_unique_decision_keys"] == 2
    assert selected["defensive_success"] == 0.05
    assert selected["loss_avoided"] == 0.05
    assert selected["winner_pnl_sacrificed"] == 0.0
    assert selected["positive_fold_share"] == 1.0
    assert not grid.loc[grid["controller_arm"].eq("S2"), "passes_diagnostic_gate"].any()


def test_policy_grid_requires_recurrent_suppression_across_folds() -> None:
    pred = pd.DataFrame(
        [
            {
                "prediction_available": True,
                "controller_arm": "S1",
                "valid_fold": 2,
                "decision_key": "win1",
                "head": "short_boll",
                "required_threshold_raise_to_suppress": 0.02,
                "pred_suppression_profit_prob": 0.90,
                "pred_direct_utility": 0.02,
                "loss_avoided_if_suppressed": 0.03,
                "winner_pnl_sacrificed_if_suppressed": 0.0,
                "direct_defensive_utility": 0.03,
            },
            {
                "prediction_available": True,
                "controller_arm": "S1",
                "valid_fold": 2,
                "decision_key": "win2",
                "head": "short_boll",
                "required_threshold_raise_to_suppress": 0.03,
                "pred_suppression_profit_prob": 0.85,
                "pred_direct_utility": 0.02,
                "loss_avoided_if_suppressed": 0.02,
                "winner_pnl_sacrificed_if_suppressed": 0.0,
                "direct_defensive_utility": 0.02,
            },
            {
                "prediction_available": True,
                "controller_arm": "S1",
                "valid_fold": 3,
                "decision_key": "not_selected",
                "head": "short_boll",
                "required_threshold_raise_to_suppress": 0.02,
                "pred_suppression_profit_prob": 0.10,
                "pred_direct_utility": -0.01,
                "loss_avoided_if_suppressed": 0.05,
                "winner_pnl_sacrificed_if_suppressed": 0.0,
                "direct_defensive_utility": 0.05,
            },
        ]
    )

    grid, selection = evaluate_policy_grid(
        pred,
        max_delta=0.08,
        min_suppressed_rows=2,
        min_suppressed_folds=2,
    )

    best = grid.loc[
        grid["controller_arm"].eq("S1")
        & grid["probability_cutoff"].eq(0.70)
        & grid["utility_cutoff"].eq(0.010)
    ].iloc[0]
    assert best["suppressed_rows"] == 2
    assert best["suppressed_folds"] == 1
    assert best["suppression_fold_share"] == 0.5
    assert best["positive_fold_share"] == 0.5
    assert not bool(best["passes_diagnostic_gate"])
    assert selection["selected_arm"] is None
    assert selection["reason"] == "no_policy_grid_row_passed_diagnostic_gate"


def test_policy_grid_can_select_head_scoped_threshold_policy() -> None:
    pred = pd.DataFrame(
        [
            {
                "prediction_available": True,
                "controller_arm": "S1",
                "valid_fold": 2,
                "decision_key": "boll_win1",
                "head": "short_boll",
                "required_threshold_raise_to_suppress": 0.02,
                "pred_suppression_profit_prob": 0.90,
                "pred_direct_utility": 0.02,
                "loss_avoided_if_suppressed": 0.03,
                "winner_pnl_sacrificed_if_suppressed": 0.0,
                "direct_defensive_utility": 0.03,
            },
            {
                "prediction_available": True,
                "controller_arm": "S1",
                "valid_fold": 3,
                "decision_key": "boll_win2",
                "head": "short_boll",
                "required_threshold_raise_to_suppress": 0.03,
                "pred_suppression_profit_prob": 0.85,
                "pred_direct_utility": 0.02,
                "loss_avoided_if_suppressed": 0.02,
                "winner_pnl_sacrificed_if_suppressed": 0.0,
                "direct_defensive_utility": 0.02,
            },
            {
                "prediction_available": True,
                "controller_arm": "S1",
                "valid_fold": 2,
                "decision_key": "asset_winner1",
                "head": "short_asset",
                "required_threshold_raise_to_suppress": 0.02,
                "pred_suppression_profit_prob": 0.95,
                "pred_direct_utility": 0.02,
                "loss_avoided_if_suppressed": 0.0,
                "winner_pnl_sacrificed_if_suppressed": 0.10,
                "direct_defensive_utility": -0.10,
            },
            {
                "prediction_available": True,
                "controller_arm": "S1",
                "valid_fold": 3,
                "decision_key": "asset_winner2",
                "head": "short_asset",
                "required_threshold_raise_to_suppress": 0.03,
                "pred_suppression_profit_prob": 0.95,
                "pred_direct_utility": 0.02,
                "loss_avoided_if_suppressed": 0.0,
                "winner_pnl_sacrificed_if_suppressed": 0.10,
                "direct_defensive_utility": -0.10,
            },
        ]
    )

    grid, selection = evaluate_policy_grid(
        pred,
        max_delta=0.08,
        min_suppressed_rows=2,
        min_suppressed_folds=2,
        policy_scopes=("controller_arm", "controller_arm_head"),
    )

    pooled = grid.loc[
        grid["policy_scope"].eq("controller_arm")
        & grid["controller_arm"].eq("S1")
        & grid["probability_cutoff"].eq(0.70)
        & grid["utility_cutoff"].eq(0.010)
    ].iloc[0]
    assert pooled["suppressed_rows"] == 4
    assert pooled["defensive_success"] < 0
    assert not bool(pooled["passes_diagnostic_gate"])
    assert selection["selected_arm"] == "S1"
    assert selection["selected_policy_scope"] == "controller_arm_head"
    assert selection["selected_target_head"] == "short_boll"
    selected = selection["selected_policy"]
    assert selected["suppressed_rows"] == 2
    assert selected["suppressed_folds"] == 2
    assert selected["defensive_success"] == 0.05


def test_policy_grid_can_select_strategy_scoped_threshold_policy() -> None:
    pred = pd.DataFrame(
        [
            {
                "prediction_available": True,
                "controller_arm": "S1",
                "valid_fold": 2,
                "decision_key": "good_strategy_win1",
                "head": "short_boll",
                "strategy_id": "short_boll_good",
                "required_threshold_raise_to_suppress": 0.02,
                "pred_suppression_profit_prob": 0.90,
                "pred_direct_utility": 0.02,
                "loss_avoided_if_suppressed": 0.03,
                "winner_pnl_sacrificed_if_suppressed": 0.0,
                "direct_defensive_utility": 0.03,
            },
            {
                "prediction_available": True,
                "controller_arm": "S1",
                "valid_fold": 3,
                "decision_key": "good_strategy_win2",
                "head": "short_boll",
                "strategy_id": "short_boll_good",
                "required_threshold_raise_to_suppress": 0.03,
                "pred_suppression_profit_prob": 0.85,
                "pred_direct_utility": 0.02,
                "loss_avoided_if_suppressed": 0.02,
                "winner_pnl_sacrificed_if_suppressed": 0.0,
                "direct_defensive_utility": 0.02,
            },
            {
                "prediction_available": True,
                "controller_arm": "S1",
                "valid_fold": 2,
                "decision_key": "bad_strategy_winner1",
                "head": "short_boll",
                "strategy_id": "short_boll_bad",
                "required_threshold_raise_to_suppress": 0.02,
                "pred_suppression_profit_prob": 0.95,
                "pred_direct_utility": 0.02,
                "loss_avoided_if_suppressed": 0.0,
                "winner_pnl_sacrificed_if_suppressed": 0.10,
                "direct_defensive_utility": -0.10,
            },
            {
                "prediction_available": True,
                "controller_arm": "S1",
                "valid_fold": 3,
                "decision_key": "bad_strategy_winner2",
                "head": "short_boll",
                "strategy_id": "short_boll_bad",
                "required_threshold_raise_to_suppress": 0.03,
                "pred_suppression_profit_prob": 0.95,
                "pred_direct_utility": 0.02,
                "loss_avoided_if_suppressed": 0.0,
                "winner_pnl_sacrificed_if_suppressed": 0.10,
                "direct_defensive_utility": -0.10,
            },
        ]
    )

    grid, selection = evaluate_policy_grid(
        pred,
        max_delta=0.08,
        min_suppressed_rows=2,
        min_suppressed_folds=2,
        policy_scopes=("controller_arm", "controller_arm_head", "controller_arm_strategy"),
    )

    pooled = grid.loc[
        grid["policy_scope"].eq("controller_arm")
        & grid["controller_arm"].eq("S1")
        & grid["probability_cutoff"].eq(0.70)
        & grid["utility_cutoff"].eq(0.010)
    ].iloc[0]
    head_scoped = grid.loc[
        grid["policy_scope"].eq("controller_arm_head")
        & grid["target_head"].eq("short_boll")
        & grid["probability_cutoff"].eq(0.70)
        & grid["utility_cutoff"].eq(0.010)
    ].iloc[0]

    assert pooled["suppressed_rows"] == 4
    assert head_scoped["suppressed_rows"] == 4
    assert pooled["defensive_success"] < 0
    assert head_scoped["defensive_success"] < 0
    assert not bool(pooled["passes_diagnostic_gate"])
    assert not bool(head_scoped["passes_diagnostic_gate"])
    assert selection["selected_arm"] == "S1"
    assert selection["selected_policy_scope"] == "controller_arm_strategy"
    assert selection["selected_target_strategy_id"] == "short_boll_good"
    selected = selection["selected_policy"]
    assert selected["target_strategy_id"] == "short_boll_good"
    assert selected["suppressed_rows"] == 2
    assert selected["suppressed_folds"] == 2
    assert selected["defensive_success"] == 0.05
