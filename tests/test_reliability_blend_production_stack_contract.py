import json
from pathlib import Path

from scripts import materialize_t1_repaired_static_baseline as t1_materializer


def _load_stack() -> dict:
    return json.loads(Path("config/reliability_blend_production_stack.json").read_text())


def test_production_stack_tracks_active_global_rank_and_shadow_controller() -> None:
    payload = _load_stack()

    active = payload["active_stack"]
    assert active["rank_contract"] == "anchor_global_policy_rank_reference"
    assert active["rank_scope"] == "global_over_time"
    assert active["rank_reference_run_id"] == "reliability_blend_anchor_rank_reference_20260625_prejune"
    assert active["promotion_status"] == "active_candidate_pending_production_governance"
    assert active["qfail_active"] is False
    assert active["native_reliability_blend_active"] is False
    assert active["market_state_threshold_controller_active"] is False
    assert active["market_state_priority_modulation_active"] is False
    assert active["market_state_priority_modulation_mode"] == "shadow_only"
    assert active["enabled_heads"] == ["short_asset", "short_boll"]
    assert active["disabled_heads"] == ["long_bars", "long_dist"]

    artifact = payload["production_artifact"]
    artifact_manifest = json.loads(Path(artifact["manifest"]).read_text())
    artifact_stack = artifact_manifest["active_stack"]
    assert artifact_stack["rank_contract"] == active["rank_contract"]
    assert artifact_stack["rank_scope"] == active["rank_scope"]
    assert artifact_stack["rank_reference_run_id"] == active["rank_reference_run_id"]
    assert artifact_manifest["validation"]["rank_reference_contract"]["passed"] is True
    assert artifact_manifest["validation"]["rank_reference_contract"]["failures"] == []
    assert artifact_manifest["validation"]["disabled_heads_absent"] is True
    assert artifact_manifest["validation"]["accepted_decision_keys_unique"] is True
    assert artifact_manifest["validation"]["active_score_alias_reliability_equals_calibrated"] is True

    metrics = payload["june_15_22_metrics"]
    summary = artifact_manifest["summary"]
    assert metrics["trade_count"] == int(summary["trade_count"])
    assert abs(float(metrics["net_pnl"]) - float(summary["net_pnl"])) <= 1e-9
    assert abs(float(metrics["gross_pnl"]) - float(summary["gross_pnl"])) <= 1e-9
    assert abs(float(metrics["cost_pnl"]) - float(summary["cost_pnl"])) <= 1e-9
    assert abs(float(metrics["full_sl_rate"]) - float(summary["full_sl_rate"])) <= 1e-12
    assert abs(float(metrics["timeout_rate"]) - float(summary["timeout_rate"])) <= 1e-12

    by_head = {row["head"]: row for row in artifact_manifest["by_head"]}
    assert set(by_head) == {"short_asset", "short_boll"}
    assert metrics["short_asset_trade_count"] == int(by_head["short_asset"]["trade_count"])
    assert metrics["short_boll_trade_count"] == int(by_head["short_boll"]["trade_count"])
    assert abs(float(metrics["short_asset_net_pnl"]) - float(by_head["short_asset"]["net_pnl"])) <= 1e-9
    assert abs(float(metrics["short_boll_net_pnl"]) - float(by_head["short_boll"]["net_pnl"])) <= 1e-9
    assert Path(artifact["report"]).exists()
    assert Path(artifact["candidates_broad"]).exists()
    assert Path(artifact["accepted_trades"]).exists()

    evidence = payload["parallel_validation_artifacts"]["rank_contract_evidence_audit"]
    evidence_payload = json.loads(Path(evidence["audit_json"]).read_text())
    assert evidence["promotion_gate_passed"] is True
    assert evidence["global_rank_promoted"] is True
    assert evidence["timestamp_rank_remains_provisional"] is False
    assert evidence["active_contract_recommendation"] == active["rank_contract"]
    assert evidence_payload["active_contract_recommendation"] == active["rank_contract"]
    assert evidence_payload["promotion_gate_passed"] is True
    assert evidence_payload["failures"] == []

    controller = payload["market_state_controller_validation"]
    assert controller["qfail_active"] is False
    assert controller["rank_contract"] == active["rank_contract"]
    assert controller["controller_action_scope"] == "threshold_raises_only"
    assert controller["changes_scores_or_ranks"] is False
    assert controller["changes_auction_ordering"] is False

    no_backfill = controller["global_rank_threshold_controller_no_backfill_walkforward"]
    assert no_backfill["status"] == "shadow_candidate_rejected_by_direct_accepted_frontier_gate"
    assert no_backfill["selected_arm"] is None
    assert no_backfill["controller_no_backfill_overlay"] is True
    assert no_backfill["selected_action_entrants"] == 0
    assert no_backfill["selected_removed_trades"] == 0
    assert float(no_backfill["selected_defensive_success"]) == 0.0
    assert float(no_backfill["selected_winner_pnl_sacrificed"]) == 0.0
    assert no_backfill["walkforward_promotion_gate_audit_passed"] is False
    assert no_backfill["walkforward_overlay_selection_gate_passed"] is False
    assert no_backfill["walkforward_expected_selected_arm"] is None
    assert no_backfill["promotion_gate_passed"] is False
    assert "accepted_frontier_reselection_no_arm_passed" in no_backfill["promotion_gate_failures"]
    assert "direct_accepted_threshold_suppression_not_recurrent" in no_backfill["promotion_gate_failures"]
    assert no_backfill["accepted_frontier_selected_arm"] is None
    assert no_backfill["accepted_frontier_selection_reason"] == "no_arm_passed_selection_gates"
    assert no_backfill["accepted_frontier_gate_source"] == "baseline_accepted_suppression"
    assert no_backfill["accepted_frontier_overlay_gate_uses_action_metrics"] is False
    assert Path(no_backfill["accepted_frontier_reselection_payload"]).exists()
    assert Path(no_backfill["accepted_frontier_reselection_summary_csv"]).exists()
    assert Path(no_backfill["accepted_frontier_reselection_report"]).exists()
    direct_ledger = no_backfill["direct_suppression_training_ledger"]
    direct_ledger_summary = json.loads(Path(direct_ledger["summary_json"]).read_text())
    assert direct_ledger["artifact_contract"] == "direct_accepted_frontier_training_ledger_v1"
    assert direct_ledger["artifact_contract"] == direct_ledger_summary["artifact_contract"]
    assert direct_ledger["source_dir"] == no_backfill["artifact_dir"]
    assert direct_ledger["row_count"] == direct_ledger_summary["row_count"]
    assert direct_ledger["baseline_accepted_rows"] == direct_ledger_summary["baseline_accepted_rows"]
    assert direct_ledger["unique_decision_key_count"] == direct_ledger_summary["unique_decision_key_count"]
    assert direct_ledger["duplicate_controller_decision_key_rows"] == 0
    assert direct_ledger["direct_profitable_rate"] == direct_ledger_summary["direct_profitable_rate"]
    assert direct_ledger["current_schedule_suppressed_rows"] == direct_ledger_summary[
        "current_schedule_suppressed_rows"
    ]
    assert direct_ledger["current_schedule_defensive_utility"] == direct_ledger_summary[
        "current_schedule_defensive_utility"
    ]
    assert direct_ledger["rank_score_sources"] == ["effective_rank_score"]
    assert direct_ledger["by_head"]["short_asset"]["mean_direct_defensive_utility"] < 0.0
    assert direct_ledger["by_head"]["short_boll"]["mean_direct_defensive_utility"] > 0.0
    assert "training ledger, not a promotion artifact" in direct_ledger["interpretation"]
    assert Path(direct_ledger["ledger_dir"]).exists()
    assert Path(direct_ledger["ledger_parquet"]).exists()
    assert Path(direct_ledger["ledger_csv"]).exists()
    assert Path(direct_ledger["by_group_csv"]).exists()
    assert Path(direct_ledger["report_md"]).exists()
    direct_training = no_backfill["direct_suppression_shadow_training"]
    direct_training_summary = json.loads(Path(direct_training["summary_json"]).read_text())
    assert (
        direct_training["artifact_contract"]
        == "direct_accepted_frontier_suppression_controller_training_v1"
    )
    assert direct_training["artifact_contract"] == direct_training_summary["artifact_contract"]
    assert direct_training["ledger_rows"] == direct_training_summary["ledger_rows"]
    assert direct_training["unique_decision_keys"] == direct_training_summary["unique_decision_keys"]
    assert direct_training["oof_rows"] == direct_training_summary["oof"]["oof_rows"]
    assert direct_training["oof_unique_decision_keys"] == direct_training_summary["oof"][
        "oof_unique_decision_keys"
    ]
    assert direct_training["oof_profit_auc"] == direct_training_summary["oof"]["prob_auc"]
    assert direct_training["oof_average_precision"] == direct_training_summary["oof"][
        "prob_average_precision"
    ]
    assert direct_training["oof_utility_spearman"] == direct_training_summary["oof"][
        "utility_spearman"
    ]
    assert direct_training["selected_arm"] is None
    assert direct_training_summary["selection"]["selected_arm"] is None
    assert direct_training["selection_reason"] == direct_training_summary["selection"]["reason"]
    assert direct_training["best_attempt_suppressed_rows"] == direct_training_summary["selection"][
        "best_attempt"
    ]["suppressed_rows"]
    assert direct_training["promotion_allowed"] is False
    assert direct_training_summary["promotion_allowed"] is False
    assert Path(direct_training["training_dir"]).exists()
    assert Path(direct_training["report_md"]).exists()
    assert Path(direct_training["oof_predictions"]).exists()
    assert Path(direct_training["policy_grid"]).exists()
    assert Path(direct_training["feature_importance"]).exists()
    assert Path(direct_training["feature_spec"]).exists()
    assert Path(direct_training["model_path"]).exists()
    assert any(
        row["arm"] == "S1_observed_axes_shared_response__post_selection_overlay"
        and row["passed_selection_gates"] is False
        and "defensive_success_not_positive" in row["selection_fail_reasons"]
        for row in no_backfill["accepted_frontier_reselection_arm_summary"]
    )

    selected = json.loads(
        (Path(no_backfill["artifact_dir"]) / "walkforward_selected_controller_candidate.json").read_text()
    )
    walkforward_audit = json.loads(Path(no_backfill["walkforward_promotion_gate_audit_json"]).read_text())
    bundle_manifest = json.loads((Path(no_backfill["bundle_dir"]) / "manifest.json").read_text())
    controller_config = json.loads(
        (Path(no_backfill["bundle_dir"]) / "strategy_threshold_controller_config.json").read_text()
    )
    accepted_reselection = json.loads(Path(no_backfill["accepted_frontier_reselection_payload"]).read_text())
    assert selected["selected_arm"] is None
    assert selected["reason"] == "no_arm_passed_selection_gates"
    assert selected["selection_policy"]["select_no_backfill_overlay_only"] is True
    assert accepted_reselection["selected_arm"] is None
    assert accepted_reselection["reason"] == "no_arm_passed_selection_gates"
    assert accepted_reselection["selection_policy"]["suppression_gate_source"] == "baseline_accepted_suppression"
    assert accepted_reselection["selection_policy"]["overlay_gate_uses_action_metrics"] is False
    assert walkforward_audit["passed"] is True
    assert walkforward_audit["expected_selected_arm"] == selected["selected_arm"]
    assert Path(no_backfill["walkforward_promotion_gate_audit_report"]).exists()
    assert Path(no_backfill["walkforward_promotion_gate_audit_selection_csv"]).exists()
    assert bundle_manifest["selected_arm"] == "S1_observed_axes_shared_response__post_selection_overlay"
    assert bundle_manifest["rank_contract"] == active["rank_contract"]
    assert bundle_manifest["controller_enabled_scope"] == "disabled_by_activation_registry"
    assert no_backfill["selected_arm"] is None
    assert no_backfill["bundle_controller_execution_enabled"] is False
    assert controller_config["controller_execution_enabled"] is False
    assert controller_config["state_spec"]["controller_no_backfill_overlay"] is True

    latest_shadow = controller["global_rank_threshold_controller_no_backfill_shadow_score_latest"]
    assert latest_shadow["status"] in {
        "shadow_score_negative_not_promoted",
        "shadow_score_positive_not_promoted",
    }
    assert latest_shadow["monitor_status"] == "not_promoted_negative_later_windows"
    assert latest_shadow["monitor_promotion_gate_passed"] is False
    assert latest_shadow["monitor_controller_should_remain_disabled"] is True
    assert latest_shadow["controller_no_backfill_overlay"] is True
    assert latest_shadow["controller_execution_enabled"] is False
    assert latest_shadow["shadow_controller_only"] is True
    assert latest_shadow["rank_contract"] == "anchor_global_policy_rank_reference"
    assert latest_shadow["shadow_subset_of_baseline"] is True
    assert latest_shadow["added_trade_count"] == 0
    assert latest_shadow["removed_trade_count"] > 0
    assert float(latest_shadow["eval_feature_store_timestamp_coverage"]) == 1.0
    assert int(latest_shadow["eval_source_feature_count"]) > 0
    assert float(latest_shadow["monitor_median_total_net_pnl_delta"]) < 0.0
    assert float(latest_shadow["monitor_q25_total_net_pnl_delta"]) < 0.0
    assert latest_shadow["promotion_gate_passed"] is False
    assert "negative_median_later_window_total_delta_net_pnl" in latest_shadow["promotion_gate_failures"]
    assert "defensive_success_not_positive" in latest_shadow["promotion_gate_failures"]
    assert Path(latest_shadow["score_dir"]).exists()
    assert Path(latest_shadow["bundle_dir"]).exists()
    assert Path(latest_shadow["bundle_contract_audit_json"]).exists()

    shadow_monitor = controller["global_rank_threshold_controller_no_backfill_shadow_monitor"]
    assert shadow_monitor["status"] == "not_promoted_negative_later_windows"
    assert shadow_monitor["window_count"] >= 2
    assert 0.0 < float(shadow_monitor["positive_delta_window_share"]) < 0.5
    assert float(shadow_monitor["median_total_net_pnl_delta"]) < 0.0
    assert float(shadow_monitor["q25_total_net_pnl_delta"]) < 0.0
    assert float(shadow_monitor["sum_total_net_pnl_delta"]) < 0.0
    assert float(shadow_monitor["sum_full_path_replay_net_pnl_delta"]) < 0.0
    assert float(shadow_monitor["sum_action_only_fixed_common_size_net_pnl_delta"]) < 0.0
    assert float(shadow_monitor["sum_path_dependent_common_trade_net_pnl_delta"]) < 0.0
    assert int(shadow_monitor["sum_added_trade_count"]) == 0
    assert int(shadow_monitor["sum_removed_trade_count"]) > 0
    assert float(shadow_monitor["sum_removed_loss_avoided"]) > 0.0
    assert float(shadow_monitor["sum_removed_winner_pnl_sacrificed"]) > float(
        shadow_monitor["sum_removed_loss_avoided"]
    )
    assert float(shadow_monitor["sum_common_net_pnl_delta"]) < 0.0
    assert shadow_monitor["all_shadow_subset_of_baseline"] is True
    assert shadow_monitor["all_source_contracts_passed"] is True
    assert shadow_monitor["promotion_gate_passed"] is False
    assert "positive_later_window_share_not_above_chance" in shadow_monitor["promotion_gate_failures"]
    assert "defensive_success_not_positive" in shadow_monitor["promotion_gate_failures"]
    assert (
        "suppressed_loss_avoided_not_greater_than_winner_pnl_sacrificed"
        in shadow_monitor["promotion_gate_failures"]
    )
    assert Path(shadow_monitor["monitor_dir"]).exists()
    assert Path(shadow_monitor["window_metrics_csv"]).exists()
    assert Path(shadow_monitor["by_head_csv"]).exists()


def test_t1_materializer_default_materializes_global_rank_baseline() -> None:
    assert "T1_global_rank_static_baseline_active" in str(t1_materializer.DEFAULT_OUTPUT_DIR)
    assert t1_materializer._rank_contract_scope("anchor_global_policy_rank_reference") == "global_over_time"
    assert (
        t1_materializer._rank_contract_promotion_status("anchor_global_policy_rank_reference")
        == "active_candidate_pending_production_governance"
    )
    assert "pre-June walk-forward" in t1_materializer._rank_contract_promotion_basis(
        "anchor_global_policy_rank_reference"
    )


def test_timestamp_rank_contract_remains_explicit_legacy_comparison() -> None:
    contract = "short_boll_timestamp_rank"

    assert t1_materializer._rank_contract_scope(contract) == "within_timestamp"
    assert t1_materializer._rank_contract_promotion_status(contract) == "provisional"
    assert t1_materializer._rank_contract_promotion_basis(contract) == "June attribution replay"
