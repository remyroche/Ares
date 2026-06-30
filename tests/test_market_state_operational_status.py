from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd

from scripts.report_market_state_operational_status import (
    build_operational_status,
    write_operational_status,
)


def _stack_config() -> dict:
    return {
        "active_stack": {
            "name": "T1_repaired_static_baseline",
            "active_score_column": "calibrated_score",
            "policy_variant": "refit_bar4_strategy_bar2",
            "rank_contract": "anchor_global_policy_rank_reference",
            "rank_scope": "global_over_time",
            "enabled_heads": ["short_asset", "short_boll"],
            "disabled_heads": ["long_bars", "long_dist"],
            "qfail_active": False,
            "native_reliability_blend_active": False,
            "market_state_shadow_logging_only": True,
            "market_state_threshold_controller_active": False,
            "market_state_priority_modulation_active": False,
        },
        "market_state_experiment_contract": {
            "name": "T1_repaired_static_baseline_experiment_contract",
            "rank_contract": "anchor_global_policy_rank_reference",
            "rank_scope": "global_over_time",
            "enabled_heads": ["short_asset", "short_boll"],
            "disabled_heads": ["long_bars", "long_dist"],
            "policy_variant": "refit_bar4_strategy_bar2",
            "qfail_active": False,
            "controller_action_scope": "penalty_only_per_strategy_threshold_raises",
            "controller_action_constraints": [
                "raise_thresholds_only",
                "no_score_changes",
                "no_rank_changes",
                "no_auction_reordering",
            ],
            "attribution_note": "fixed T1 global-rank attribution baseline",
        },
        "market_state_controller_validation": {
            "status": "not_promoted_shadow_only",
            "plan_completion_audit": {
                "passed_structural_audit": True,
                "hard_failure_count": 0,
                "status_counts": {"complete": 35, "gate_blocked": 3, "shadow_only": 1},
            },
            "strategy_response_quality_gate": {
                "quality_gate_passed": True,
                "quality_passing_heads": ["short_asset", "short_boll"],
            },
            "t1_timestamp_strategy_response_quality_gate": {
                "quality_gate_passed": False,
                "quality_passing_heads": ["short_boll"],
            },
            "shadow_controller_monitor": {
                "bundle_count": 2,
                "shadow_promotion_gate_passed": False,
                "shadow_promotion_failures": ["defensive_success_not_positive"],
                "append_window_discovery": {
                    "appendable_candidate_count": 0,
                    "already_monitored_count": 2,
                    "excluded_candidate_count": 1,
                },
            },
            "global_rank_threshold_controller_no_backfill_shadow_monitor": {
                "status": "not_promoted_negative_later_windows",
                "window_count": 2,
                "min_later_window_count": 3,
                "windows": [
                    {
                        "period_start": "2026-06-23T09:00:00+00:00",
                        "period_end": "2026-06-24T08:00:00+00:00",
                        "total_net_pnl_delta": -5.0,
                        "baseline_trade_count": 49,
                        "shadow_trade_count": 46,
                    },
                    {
                        "period_start": "2026-06-24T09:00:00+00:00",
                        "period_end": "2026-06-25T08:00:00+00:00",
                        "total_net_pnl_delta": -6.0,
                        "baseline_trade_count": 66,
                        "shadow_trade_count": 56,
                    },
                ],
                "positive_delta_window_share": 0.0,
                "action_only_positive_window_share": 1.0,
                "sum_total_net_pnl_delta": -11.0,
                "sum_full_path_replay_net_pnl_delta": -11.0,
                "sum_action_only_fixed_common_size_net_pnl_delta": 15.0,
                "sum_path_dependent_common_trade_net_pnl_delta": -26.0,
                "locked_accepted_overlay_available_window_count": 2,
                "locked_accepted_overlay_positive_window_share": 1.0,
                "locked_accepted_overlay_suppression_window_share": 1.0,
                "sum_locked_accepted_overlay_net_pnl_delta": 15.0,
                "sum_locked_accepted_overlay_removed_trade_count": 13,
                "sum_locked_accepted_overlay_defensive_success": 15.0,
                "locked_accepted_overlay_promotion_gate_passed": False,
                "locked_accepted_overlay_promotion_gate_failures": [
                    "locked_accepted_overlay_insufficient_later_window_count"
                ],
                "sum_indirect_path_or_capacity_net_pnl_delta": -26.0,
                "sum_indirect_path_or_capacity_removed_trade_count": 4,
                "sum_indirect_path_or_capacity_defensive_success": -20.0,
                "sum_indirect_path_or_capacity_winner_pnl_sacrificed": 24.0,
                "all_score_manifest_artifact_hashes_complete": False,
                "score_manifest_contract_versions": [],
                "windows_missing_score_input_hash_fields": 2,
                "windows_missing_required_output_hashes": 0,
                "promotion_gate_passed": False,
                "promotion_gate_failures": [
                    "full_path_replay_negative_despite_positive_action_only_counterfactual",
                    "score_manifest_artifact_hashes_missing",
                ],
            },
            "global_rank_threshold_controller_no_backfill_next_window_readiness": {
                "readiness_dir": "readiness_dir",
                "summary_json": "readiness_dir/next_no_backfill_shadow_window_readiness.json",
                "report_md": "readiness_dir/next_no_backfill_shadow_window_readiness_report.md",
                "readiness_csv": "readiness_dir/next_no_backfill_shadow_window_readiness.csv",
                "status": "not_scoreable_yet",
                "feature_store_dir": "data_perp/features/20260627_010000",
                "feature_timestamp_max": "2026-06-27T00:00:00+00:00",
                "maturity_buffer_hours": 16,
                "maturity_cutoff": "2026-06-26T08:00:00+00:00",
                "next_window_start": "2026-06-26T08:00:00+00:00",
                "target_window_end": "2026-06-27T07:00:00+00:00",
                "proposed_scoreable_window_end": "2026-06-26T08:00:00+00:00",
                "mature_timestamp_count_available": 1,
                "min_timestamp_count": 3,
                "target_window_hours": 24,
                "scoreable_min_window_now": False,
                "scoreable_full_window_now": False,
                "missing_feature_hours_for_min_window": 2,
                "missing_feature_hours_for_full_window": 23,
                "latest_anchor_candidate_timestamp_max": "2026-06-26T07:00:00+00:00",
                "failures": ["insufficient_matured_timestamps_for_minimum_shadow_window"],
                "next_action": "wait_for_or_generate_more_feature_history_before_next_shadow_score",
            },
            "global_rank_threshold_controller_no_backfill_walkforward": {
                "status": "shadow_candidate_rejected_by_direct_accepted_frontier_gate",
                "artifact_dir": "no_backfill_walkforward_dir",
                "promotion_gate_failures": [
                    "accepted_frontier_reselection_no_arm_passed",
                    "direct_accepted_threshold_suppression_not_recurrent",
                ],
                "promotion_gate_passed": False,
                "direct_suppression_training_ledger": {
                    "artifact_contract": "direct_accepted_frontier_training_ledger_v1",
                    "ledger_dir": "direct_ledger_dir",
                    "ledger_parquet": (
                        "direct_ledger_dir/direct_accepted_frontier_training_ledger.parquet"
                    ),
                    "report_md": "direct_ledger_dir/direct_accepted_frontier_training_report.md",
                    "row_count": 92,
                    "unique_decision_key_count": 23,
                    "direct_profitable_rate": 0.5652173913043478,
                    "mean_direct_defensive_utility": -0.0003434591972950935,
                    "total_direct_defensive_utility": -0.031598246151148604,
                    "current_schedule_suppressed_rows": 4,
                    "current_schedule_defensive_utility": 0.02457198033670408,
                    "interpretation": "This is a training ledger, not a promotion artifact.",
                    "by_head": {
                        "short_asset": {
                            "mean_direct_defensive_utility": -0.013762181432476575,
                        },
                        "short_boll": {
                            "mean_direct_defensive_utility": 0.017100879708440837,
                        },
                    },
                },
                "direct_suppression_shadow_training": {
                    "artifact_contract": (
                        "direct_accepted_frontier_suppression_controller_training_v1"
                    ),
                    "training_dir": "direct_training_dir",
                    "summary_json": "direct_training_dir/direct_suppression_training_summary.json",
                    "report_md": "direct_training_dir/direct_suppression_training_report.md",
                    "model_path": "direct_training_dir/direct_suppression_shadow_models.joblib",
                    "model_mode": "lgbm",
                    "oof_rows": 48,
                    "oof_unique_decision_keys": 12,
                    "oof_profit_auc": 0.9982142857142857,
                    "oof_average_precision": 0.9975369458128078,
                    "oof_utility_spearman": 0.8284237196171829,
                    "selected_arm": None,
                    "selection_reason": "no_policy_grid_row_passed_diagnostic_gate",
                    "best_attempt_controller_arm": "S1_observed_axes_shared_response",
                    "best_attempt_suppressed_rows": 1,
                    "best_attempt_defensive_success": 0.058977099207927465,
                    "promotion_allowed": False,
                    "interpretation": (
                        "The OOF model separates profitable direct suppressions, but no "
                        "policy-grid row passes the diagnostic action gate."
                    ),
                },
            },
        },
        "market_state_priority_modulation_validation": {
            "status": "not_promoted_shadow_only",
            "promotion_gate_failures": ["accepted_jaccard_below_required_95pct"],
            "global_rank_opportunity_headmix_validation": {
                "opportunity_routing_gate_passed": False,
                "promotion_gate_failures": ["fewer_than_2_action_windows"],
                "window_count": 3,
                "median_delta_net_pnl": 0.0,
                "q25_delta_net_pnl": 0.0,
                "positive_delta_window_share": 0.3333333333333333,
                "min_shadow_active_head_count": 2,
                "max_shadow_dominant_head_share": 0.94,
                "median_head_trade_share_l1_delta": 0.0,
                "max_head_trade_share_l1_delta": 0.01,
                "audit_dir": "priority_headmix_dir",
                "recurrent_challenger_selected": False,
                "recurrent_selection_reason": "no_recurrent_gate_passing_arm",
                "recurrent_best_candidate": "cap_0p6_zge_0p5",
                "recurrent_best_candidate_failures": [
                    "fewer_than_required_action_windows",
                    "timeout_worsened_in_a_window",
                ],
                "recurrent_challenger_json": "priority_headmix_dir/recurrent_shadow_challenger.json",
            },
            "short_boll_state_rank_scope_switch_shadow": {
                "aggregate_validation": {
                    "shadow_promotion_gate_passed": False,
                    "shadow_promotion_failures": ["later_median_delta_not_positive"],
                    "later_blend_median_delta_net_pnl": -9.0,
                }
            },
        },
    }


def test_operational_status_separates_structural_completion_from_promotion() -> None:
    status = build_operational_status(_stack_config())

    assert status["structural_complete"] is True
    assert status["production_ready"] is False
    assert status["active_score_column"] == "calibrated_score"
    assert status["active_policy_variant"] == "refit_bar4_strategy_bar2"
    assert status["active_rank_contract"] == "anchor_global_policy_rank_reference"
    assert status["experiment_rank_contract"] == "anchor_global_policy_rank_reference"
    assert status["experiment_rank_scope"] == "global_over_time"
    assert status["experiment_contract_matches_active_stack"] is True
    assert status["native_reliability_blend_active"] is False
    assert status["market_state_shadow_logging_only"] is True
    assert status["threshold_controller_active"] is False
    assert status["response_quality_gate_passed"] is True
    assert status["response_quality_passing_heads"] == ["short_asset", "short_boll"]
    assert status["shadow_monitor_gate_passed"] is False
    threshold_blocker = next(
        row for row in status["blockers"] if row["component"] == "threshold_controller"
    )
    assert "defensive_success_not_positive" in threshold_blocker["failures"]
    assert (
        "full_path_replay_negative_despite_positive_action_only_counterfactual"
        in threshold_blocker["failures"]
    )
    assert "score_manifest_artifact_hashes_missing" in threshold_blocker["failures"]
    assert status["threshold_controller_no_backfill_monitor_indirect_path_delta"] == -26.0
    assert status["threshold_controller_no_backfill_monitor_locked_overlay_available_windows"] == 2
    assert status["threshold_controller_no_backfill_monitor_locked_overlay_positive_share"] == 1.0
    assert status["threshold_controller_no_backfill_monitor_locked_overlay_suppression_share"] == 1.0
    assert status["threshold_controller_no_backfill_monitor_locked_overlay_delta"] == 15.0
    assert status["threshold_controller_no_backfill_monitor_locked_overlay_removed_trades"] == 13
    assert status["threshold_controller_no_backfill_monitor_locked_overlay_defensive_success"] == 15.0
    assert (
        status["threshold_controller_no_backfill_monitor_locked_overlay_promotion_gate_passed"]
        is False
    )
    assert status["threshold_controller_no_backfill_monitor_locked_overlay_failures"] == [
        "locked_accepted_overlay_insufficient_later_window_count"
    ]
    assert status["threshold_controller_no_backfill_monitor_indirect_removed_trades"] == 4
    assert status["threshold_controller_no_backfill_monitor_indirect_defensive_success"] == -20.0
    assert (
        status["threshold_controller_no_backfill_monitor_indirect_winner_pnl_sacrificed"]
        == 24.0
    )
    assert status["threshold_controller_no_backfill_monitor_artifact_hashes_complete"] is False
    assert status["threshold_controller_no_backfill_monitor_windows_missing_score_input_hash_fields"] == 2
    assert status["threshold_controller_no_backfill_monitor_min_later_window_count"] == 3
    assert status["threshold_controller_no_backfill_monitor_additional_windows_needed"] == 1
    assert status["threshold_controller_no_backfill_next_readiness_status"] == "not_scoreable_yet"
    assert (
        status["threshold_controller_no_backfill_next_feature_timestamp_max"]
        == "2026-06-27T00:00:00+00:00"
    )
    assert status["threshold_controller_no_backfill_next_mature_timestamp_count"] == 1
    assert status["threshold_controller_no_backfill_next_scoreable_min_window_now"] is False
    assert status["threshold_controller_no_backfill_next_missing_feature_hours_min"] == 2
    assert (
        status["threshold_controller_no_backfill_monitor_latest_window_end"]
        == "2026-06-25T08:00:00+00:00"
    )
    assert len(status["threshold_controller_no_backfill_monitor_window_periods"]) == 2
    assert (
        status["threshold_controller_direct_suppression_ledger_contract"]
        == "direct_accepted_frontier_training_ledger_v1"
    )
    assert status["threshold_controller_direct_suppression_ledger_rows"] == 92
    assert status["threshold_controller_direct_suppression_ledger_unique_keys"] == 23
    assert (
        status["threshold_controller_direct_suppression_ledger_profitable_rate"]
        == 0.5652173913043478
    )
    assert (
        status["threshold_controller_direct_suppression_ledger_mean_utility"]
        == -0.0003434591972950935
    )
    assert (
        status["threshold_controller_direct_suppression_ledger_current_schedule_utility"]
        == 0.02457198033670408
    )
    assert status["threshold_controller_direct_suppression_ledger_short_asset_mean_utility"] < 0.0
    assert status["threshold_controller_direct_suppression_ledger_short_boll_mean_utility"] > 0.0
    assert (
        status["threshold_controller_direct_suppression_training_contract"]
        == "direct_accepted_frontier_suppression_controller_training_v1"
    )
    assert status["threshold_controller_direct_suppression_training_model_mode"] == "lgbm"
    assert status["threshold_controller_direct_suppression_training_oof_rows"] == 48
    assert status["threshold_controller_direct_suppression_training_oof_unique_keys"] == 12
    assert status["threshold_controller_direct_suppression_training_oof_auc"] == 0.9982142857142857
    assert (
        status["threshold_controller_direct_suppression_training_selected_arm"]
        is None
    )
    assert (
        status["threshold_controller_direct_suppression_training_selection_reason"]
        == "no_policy_grid_row_passed_diagnostic_gate"
    )
    assert status["threshold_controller_direct_suppression_training_best_attempt_rows"] == 1
    assert status["threshold_controller_direct_suppression_training_promotion_allowed"] is False
    assert status["rank_scope_router_gate_passed"] is False
    assert status["head_priority_headmix_gate_passed"] is False
    assert status["head_priority_headmix_window_count"] == 3
    assert status["head_priority_headmix_failures"] == ["fewer_than_2_action_windows"]
    assert status["head_priority_headmix_min_shadow_active_head_count"] == 2
    assert status["head_priority_recurrent_challenger_selected"] is False
    assert status["head_priority_recurrent_selection_reason"] == "no_recurrent_gate_passing_arm"
    assert status["head_priority_recurrent_best_candidate"] == "cap_0p6_zge_0p5"
    assert status["head_priority_recurrent_best_candidate_failures"] == [
        "fewer_than_required_action_windows",
        "timeout_worsened_in_a_window",
    ]
    assert {row["component"] for row in status["blockers"]} == {
        "threshold_controller",
        "rank_scope_router",
        "head_priority_modulation",
    }
    assert status["next_actions"][0] == "keep_scoring_new_later_windows_until_appendable_shadow_bundles_exist"
    assert (
        "keep_locked_accepted_overlay_shadow_only_until_suppression_is_recurrent"
        in status["next_actions"]
    )
    assert "wait_for_or_generate_more_feature_history_before_next_shadow_score" in status["next_actions"]
    assert "score_1_additional_no_backfill_shadow_window" in status["next_actions"]
    assert (
        "train_threshold_controller_on_direct_suppression_training_ledger"
        not in status["next_actions"]
    )
    assert (
        "accumulate_more_direct_suppression_support_before_shadow_policy_activation"
        in status["next_actions"]
    )


def test_operational_status_prefers_combined_direct_suppression_artifacts() -> None:
    config = _stack_config()
    no_backfill = config["market_state_controller_validation"][
        "global_rank_threshold_controller_no_backfill_walkforward"
    ]
    no_backfill["direct_suppression_combined_training_ledger"] = {
        "artifact_contract": "direct_accepted_frontier_training_ledger_v1",
        "aggregation_contract": "combined_direct_accepted_frontier_training_ledger_v1",
        "ledger_dir": "combined_ledger_dir",
        "ledger_parquet": "combined_ledger_dir/direct_accepted_frontier_training_ledger.parquet",
        "report_md": "combined_ledger_dir/direct_accepted_frontier_training_report.md",
        "row_count": 112,
        "unique_decision_key_count": 43,
        "direct_profitable_rate": 0.5535714285714286,
        "mean_direct_defensive_utility": -0.0018869281357901297,
        "total_direct_defensive_utility": -0.21133595120849452,
        "current_schedule_suppressed_rows": 4,
        "current_schedule_defensive_utility": 0.02457198033670408,
        "by_head": {
            "short_asset": {"mean_direct_defensive_utility": -0.0129},
            "short_boll": {"mean_direct_defensive_utility": 0.0138},
        },
    }
    no_backfill["direct_suppression_combined_shadow_training"] = {
        "artifact_contract": "direct_accepted_frontier_suppression_controller_training_v1",
        "training_dir": "combined_training_dir",
        "summary_json": "combined_training_dir/direct_suppression_training_summary.json",
        "report_md": "combined_training_dir/direct_suppression_training_report.md",
        "model_path": "combined_training_dir/direct_suppression_shadow_models.joblib",
        "model_mode": "lgbm",
        "oof_rows": 68,
        "oof_unique_decision_keys": 32,
        "oof_profit_auc": 0.7995614035087719,
        "oof_average_precision": 0.8918860849980369,
        "oof_utility_spearman": 0.7873866857883117,
        "selected_arm": None,
        "selection_reason": "no_policy_grid_row_passed_diagnostic_gate",
        "best_attempt_controller_arm": "S1_observed_axes_shared_response",
        "best_attempt_suppressed_rows": 1,
        "best_attempt_defensive_success": 0.058977099207927465,
        "promotion_allowed": False,
    }

    status = build_operational_status(config)

    assert status["threshold_controller_direct_suppression_ledger_variant"] == "combined"
    assert (
        status["threshold_controller_direct_suppression_ledger_aggregation_contract"]
        == "combined_direct_accepted_frontier_training_ledger_v1"
    )
    assert status["threshold_controller_direct_suppression_ledger_rows"] == 112
    assert status["threshold_controller_direct_suppression_training_variant"] == "combined"
    assert status["threshold_controller_direct_suppression_training_oof_rows"] == 68
    assert status["threshold_controller_direct_suppression_training_oof_auc"] == 0.7995614035087719


def test_write_operational_status_outputs_json_csv_and_markdown(tmp_path: Path) -> None:
    status = build_operational_status(_stack_config())

    write_operational_status(status, tmp_path)

    payload = json.loads((tmp_path / "market_state_operational_status.json").read_text())
    assert payload["production_ready"] is False
    assert (
        "full_path_replay_negative_despite_positive_action_only_counterfactual"
        in payload["threshold_controller_no_backfill_monitor_failures"]
    )
    assert (
        "score_manifest_artifact_hashes_missing"
        in payload["threshold_controller_no_backfill_monitor_failures"]
    )
    blockers = pd.read_csv(tmp_path / "market_state_operational_status_blockers.csv")
    assert set(blockers["component"]) == {
        "threshold_controller",
        "rank_scope_router",
        "head_priority_modulation",
    }
    report = (tmp_path / "market_state_operational_status_report.md").read_text()
    assert "implementation completeness" in report
    assert "Market-State Experiment Contract" in report
    assert "anchor_global_policy_rank_reference" in report
    assert "Matches active stack contract: `True`" in report
    assert "Production ready: `False`" in report
    assert "Artifact hashes complete: `False`" in report
    assert "Additional windows needed: `1`" in report
    assert "Locked overlay delta: `15.0`" in report
    assert "Locked overlay removed trades: `13`" in report
    assert "Locked overlay promotion gate passed: `False`" in report
    assert (
        "locked_accepted_overlay_insufficient_later_window_count"
        in report
    )
    assert "Sum indirect path/capacity delta: `-26.0`" in report
    assert "Indirect path/capacity removed trades: `4`" in report
    assert "Direct Accepted-Frontier Training Ledger" in report
    assert "Rows: `92`" in report
    assert "short_asset mean direct utility: `-0.013762181432476575`" in report
    assert "short_boll mean direct utility: `0.017100879708440837`" in report
    assert "Direct Suppression Shadow Training" in report
    assert "OOF AUC: `0.9982142857142857`" in report
    assert "Selected shadow arm: `None`" in report
    assert "Best attempt suppressed rows: `1`" in report
    assert "Latest scored window end: `2026-06-25T08:00:00+00:00`" in report
    assert "Next No-Backfill Shadow Window Readiness" in report
    assert "Status: `not_scoreable_yet`" in report
    assert "Mature timestamps available: `1`" in report
    assert "Missing feature hours for minimum window: `2`" in report
    assert "Head-mix opportunity priority gate passed: `False`" in report
    assert "Head-mix max dominant-head share: `0.94`" in report
    assert "Recurrent challenger selected: `False`" in report
    assert "Recurrent best candidate: `cap_0p6_zge_0p5`" in report


def test_operational_status_flags_stale_experiment_contract() -> None:
    config = deepcopy(_stack_config())
    config["market_state_experiment_contract"]["rank_contract"] = "short_boll_timestamp_rank"
    config["market_state_experiment_contract"]["rank_scope"] = "within_timestamp"

    status = build_operational_status(config)

    assert status["experiment_contract_matches_active_stack"] is False
    blocker = next(row for row in status["blockers"] if row["component"] == "experiment_contract")
    assert blocker["severity"] == "hard"
    assert blocker["reason"] == "market_state_experiment_contract_mismatch_active_stack"
    assert blocker["failures"] == ["rank_contract_mismatch", "rank_scope_mismatch"]
