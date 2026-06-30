import json
from pathlib import Path

import pandas as pd

from scripts.audit_market_state_plan_completion import audit_market_state_plan_completion


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_minimal_bundle(root: Path) -> tuple[Path, Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    _write_json(
        root / "manifest.json",
        {
            "rank_contract": "anchor_global_policy_rank_reference",
            "policy_variant": "refit_bar4_strategy_bar2",
            "active_heads": ["short_asset", "short_boll"],
            "disabled_heads": ["long_bars", "long_dist"],
        },
    )
    invariants = {
        "controller_changes_scores_or_ranks": False,
        "controller_changes_auction_ordering": False,
        "controller_can_lower_thresholds": False,
        "latent_gmm_active_controller_input": False,
    }
    source_audit = {
        "overall_passed": True,
        "required_source": "feature_store_market_aggregates",
        "actual_order_book_features_allowed": False,
        "candidate_population_fallback_allowed_for_production": False,
    }
    _write_json(
        root / "market_state_feature_contract.json",
        {
            "rank_contract": "anchor_global_policy_rank_reference",
            "active_heads": ["short_asset", "short_boll"],
            "disabled_heads": ["long_bars", "long_dist"],
            "invariants": invariants,
            "source_contract_audit": source_audit,
        },
    )
    _write_json(
        root / "market_state_universe_contract.json",
        {
            "strategy_independent": True,
            "candidate_independent": True,
            "eligible_symbols": ["BTC_USD:USD"],
            "eligible_symbol_count": 1,
            "minimum_history": ["ok"],
            "minimum_volume": ["ok"],
            "oi_coverage_requirements": ["optional"],
            "funding_coverage_requirements": ["optional"],
        },
    )
    _write_json(
        root / "strategy_threshold_controller_config.json",
        {
            "baseline_contract": {
                "q_fail_enabled": False,
                "active_heads": ["short_asset", "short_boll"],
                "disabled_heads": ["long_bars", "long_dist"],
                "rank_contract": "anchor_global_policy_rank_reference",
            },
            "controller": {"penalty_only": True},
            "validation": {
                "chronological_complete_timestamp_folds": True,
                "embargo_hours": 96,
            },
        },
    )
    _write_json(root / "walkforward_selected_controller_candidate.json", {"selected_arm": None})
    _write_json(
        root / "market_state_controller_contract_audit.json",
        {
            "passed": True,
            "completion_grade_passed": True,
            "failures": [],
            "artifact_audit_checks": [
                "market_state_one_row_per_fold_split_arm_timestamp",
                "no_forbidden_market_state_columns",
                "oof_state_values_match_timestamp_panel",
                "response_oof_uses_oof_state_scores",
                "state_threshold_never_below_base_threshold",
                "missing_or_ood_state_falls_back_to_base_threshold",
                "static_baseline_replay_parity",
                "accepted_decision_keys_unique_when_available",
            ],
        },
    )
    _write_json(
        root / "promotion_gate_audit" / "market_state_controller_promotion_gate_audit.json",
        {"promotion_gate_passed": False, "controller_should_remain_disabled": True},
    )
    _write_json(root / "market_state_target_definitions.json", {})
    _write_json(root / "artifact_hashes.json", {})
    for name in [
        "market_state_training_reference.joblib",
        "market_state_target_cdfs.joblib",
        "market_state_lgbm_models.joblib",
        "strategy_rank_outcome_curves.joblib",
        "strategy_response_models.joblib",
        "strategy_response_ebm_models.joblib",
    ]:
        (root / name).write_bytes(b"placeholder")

    pd.DataFrame(
        {
            "fold": [1],
            "split": ["valid"],
            "state_arm": ["S1_observed_axes_shared_response"],
            "timestamp": [pd.Timestamp("2026-05-01", tz="UTC")],
            "state_ood_score": [0.1],
            "state_drift_score": [0.2],
            "state_uncertainty": [0.3],
            "state_input_coverage": [1.0],
            "state_liquidity_stress_proxy": [0.4],
            "state_shock": [0.5],
            "state_compression": [0.2],
            "state_trend": [0.1],
            "state_deleveraging": [0.0],
            "state_novelty": [0.1],
            "state_transition": [0.2],
            "forecast_h6_shock_up": [0.6],
            "forecast_h6_shock_down": [0.1],
            "forecast_h6_rv_ratio": [0.2],
        }
    ).to_parquet(root / "market_state_timestamp_panel.parquet", index=False)
    pd.DataFrame(
        {
            "fold": [1],
            "split": ["valid"],
            "state_arm": ["S1_observed_axes_shared_response"],
            "timestamp": [pd.Timestamp("2026-05-01", tz="UTC")],
            "state_ood_score": [0.1],
            "state_drift_score": [0.2],
            "state_uncertainty": [0.3],
            "state_input_coverage": [1.0],
            "state_liquidity_stress_proxy": [0.4],
            "state_shock": [0.5],
            "state_compression": [0.2],
            "state_trend": [0.1],
            "state_deleveraging": [0.0],
            "state_novelty": [0.1],
            "state_transition": [0.2],
            "forecast_h6_shock_up": [0.6],
            "forecast_h6_shock_down": [0.1],
            "forecast_h6_rv_ratio": [0.2],
        }
    ).to_parquet(root / "market_state_oof_predictions.parquet", index=False)
    pd.DataFrame({"feature": ["x"], "coverage": [1.0]}).to_csv(root / "market_state_feature_coverage.csv", index=False)
    pd.DataFrame(
        {
            "state_level": ["forecast"] * 5,
            "state_head": [
                "forecast_h6_shock_up",
                "forecast_h6_shock_down",
                "forecast_h6_rv_ratio",
                "forecast_h6_trend_efficiency",
                "forecast_h6_liquidity_stress_proxy",
            ],
            "component_group": ["return_shock", "return_shock", "volatility_tail", "trend", "liquidity_proxy"],
            "mean_tail_average_precision": [0.2] * 5,
            "mean_tail_brier_p90": [0.2] * 5,
            "mean_tail_ece_5bin": [0.1] * 5,
            "mean_tail_recall_p90": [0.1] * 5,
            "mean_tail_false_alarm_rate_p90": [0.1] * 5,
            "collapsed_folds": [0] * 5,
        }
    ).to_csv(root / "market_state_head_diagnostics.csv", index=False)
    pd.DataFrame(
        {
            "state_head": ["forecast_h6_shock_up"],
            "recommended_status": ["active_candidate"],
            "activation_registry_version": ["market_state_activation_registry_v1"],
        }
    ).to_csv(root / "market_state_activation_registry.csv", index=False)
    pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-05-01", tz="UTC")] * 3,
            "strategy_id": ["s1"] * 3,
            "head": ["short_asset"] * 3,
            "_rank": [0.6, 0.7, 0.8],
            "resid_utility": [0.0, 0.1, -0.1],
            "resid_full_sl": [0.0, 0.1, -0.1],
            "resid_timeout": [0.0, 0.1, -0.1],
        }
    ).to_parquet(root / "strategy_residual_target_ledger.parquet", index=False)
    pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-05-01", tz="UTC")],
            "strategy_id": ["s1"],
            "symbol": ["BTC"],
            "side": ["short"],
            "head": ["short_asset"],
            "fold": [1],
            "arm": ["S0_baseline_static_thresholds"],
        }
    ).to_parquet(root / "accepted_trades.parquet", index=False)
    pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-05-01", tz="UTC")],
            "strategy_id": ["s1"],
            "head": ["short_asset"],
            "fold": [1],
            "arm": ["S1_observed_axes_shared_response"],
            "state_prediction_contract": ["outer_fold_validation_state_scores"],
            "pred_resid_utility": [0.1],
            "pred_resid_utility_lcb": [0.0],
            "pred_resid_full_sl": [0.0],
            "pred_resid_timeout": [0.0],
        }
    ).to_parquet(root / "strategy_response_oof_predictions.parquet", index=False)
    pd.DataFrame({"target": ["pred_resid_utility"]}).to_csv(root / "strategy_state_effect_matrix.csv", index=False)
    pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-05-01", tz="UTC")],
            "strategy_id": ["s1"],
            "head": ["short_asset"],
            "base_threshold": [0.7],
            "state_threshold": [0.72],
        }
    ).to_parquet(root / "strategy_threshold_schedule.parquet", index=False)
    for name in [
        "strategy_threshold_action_audit.csv",
        "walkforward_threshold_action_utility.csv",
        "walkforward_threshold_action_edge_validation.csv",
        "walkforward_threshold_candidate_suppression_aggregate.csv",
        "walkforward_threshold_baseline_accepted_suppression_aggregate.csv",
        "market_state_leave_one_head_out_aggregate.csv",
    ]:
        pd.DataFrame({"x": [1]}).to_csv(root / name, index=False)
    pd.DataFrame(
        {
            "arm": [
                "S0_baseline_static_thresholds",
                "S1_observed_axes_shared_response",
                "S2_observed_forecast_shared_response",
                "S7_pruned_state_pack",
            ]
        }
    ).to_csv(root / "portfolio_replay_summary.csv", index=False)
    pd.DataFrame({"arm": ["S0_baseline_static_thresholds"], "head": ["short_asset"]}).to_csv(
        root / "portfolio_replay_by_head.csv", index=False
    )
    pd.DataFrame({"arm": ["S0_baseline_static_thresholds"], "jaccard_vs_baseline": [1.0]}).to_csv(
        root / "walkforward_overlap.csv", index=False
    )

    state_quality_dir = root / "state_quality"
    _write_json(
        state_quality_dir / "market_state_head_quality_gate.json",
        {
            "passed": True,
            "state_heads": 1,
            "active_candidates": ["forecast_h6_shock_up"],
            "grade_counts": {"watch_active_candidate": 1},
            "forecast_quality_failure_heads": [],
        },
    )
    response_quality_dir = root / "response_quality"
    _write_json(
        response_quality_dir / "market_state_strategy_response_quality_gate.json",
        {"passed": True, "quality_passing_arm_count": 0},
    )
    priority_dir = root / "priority"
    _write_json(priority_dir / "market_state_priority_shadow_promotion_gate.json", {"promotion_gate_passed": False})
    backend_dir = root / "backend"
    backend_dir.mkdir()
    pd.DataFrame({"backend": ["lgbm", "xgb"]}).to_csv(backend_dir / "backend_metric_comparison.csv", index=False)
    return state_quality_dir, response_quality_dir, priority_dir, backend_dir


def test_market_state_plan_completion_audit_maps_requirements(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    state_quality, response_quality, priority, backend = _write_minimal_bundle(artifact)
    output = tmp_path / "out"

    payload = audit_market_state_plan_completion(
        artifact,
        output,
        state_head_quality_dir=state_quality,
        strategy_response_quality_dir=response_quality,
        shadow_priority_audit_dir=priority,
        backend_comparison_dir=backend,
    )

    checklist = pd.read_csv(output / "market_state_plan_completion_checklist.csv")

    assert payload["hard_failure_count"] == 0, payload["hard_failures"]
    assert "gate_blocked" in set(checklist["status"])
    assert "shadow_only" in set(checklist["status"])
    response_gate = checklist.loc[checklist["requirement_id"].astype(str).eq("14.3")]
    assert not response_gate.empty
    assert response_gate["status"].iloc[0] == "gate_blocked"
    assert "quality_passing_arm_count=0" in response_gate["notes"].iloc[0]
    assert "quality_gate_passed=False" in response_gate["notes"].iloc[0]
    assert "support_blocked_heads" in response_gate["notes"].iloc[0]
    assert (output / "market_state_plan_completion_audit.md").exists()


def test_market_state_plan_completion_audit_reports_strategy_scoped_direct_suppression_block(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    state_quality, response_quality, priority, backend = _write_minimal_bundle(artifact)
    direct = tmp_path / "direct_suppression"
    _write_json(
        direct / "direct_suppression_training_summary.json",
        {
            "policy_grid": {
                "policy_scopes": [
                    "controller_arm",
                    "controller_arm_head",
                    "controller_arm_strategy",
                    "controller_arm_head_strategy",
                ],
                "min_suppressed_folds": 2,
            },
            "selection": {
                "selected_arm": None,
                "reason": "no_policy_grid_row_passed_diagnostic_gate",
                "best_attempt": {
                    "policy_scope": "controller_arm_head_strategy",
                    "target_head": "short_boll",
                    "target_strategy_id": "short_boll_strategy",
                    "suppressed_rows": 1,
                    "suppressed_folds": 1,
                },
            },
            "oof": {
                "prob_auc": 0.88,
                "prob_average_precision": 0.92,
                "utility_spearman": 0.81,
            },
            "promotion_allowed": False,
        },
    )

    payload = audit_market_state_plan_completion(
        artifact,
        tmp_path / "out",
        state_head_quality_dir=state_quality,
        strategy_response_quality_dir=response_quality,
        shadow_priority_audit_dir=priority,
        backend_comparison_dir=backend,
        direct_suppression_training_dir=direct,
    )

    checklist = pd.read_csv(tmp_path / "out" / "market_state_plan_completion_checklist.csv")
    row = checklist.loc[checklist["requirement_id"].astype(str).eq("12.6")]
    assert not row.empty
    assert row["status"].iloc[0] == "gate_blocked"
    assert "controller_arm_head_strategy" in row["notes"].iloc[0]
    assert "best_strategy=short_boll_strategy" in row["notes"].iloc[0]
    assert payload["direct_suppression_selected_arm"] is None
    assert payload["direct_suppression_best_attempt_target_strategy_id"] == "short_boll_strategy"


def test_market_state_plan_completion_audit_reports_direct_suppression_actionability_blocker(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    state_quality, response_quality, priority, backend = _write_minimal_bundle(artifact)
    direct = tmp_path / "direct_suppression"
    _write_json(
        direct / "direct_suppression_training_summary.json",
        {
            "policy_grid": {
                "policy_scopes": [
                    "controller_arm",
                    "controller_arm_head",
                    "controller_arm_strategy",
                    "controller_arm_head_strategy",
                ],
                "min_suppressed_folds": 2,
            },
            "selection": {
                "selected_arm": None,
                "reason": "no_policy_grid_row_passed_diagnostic_gate",
                "best_attempt": {
                    "policy_scope": "controller_arm",
                    "target_head": None,
                    "suppressed_rows": 2,
                    "suppressed_folds": 2,
                },
            },
            "oof": {
                "prob_auc": 0.8716,
                "prob_average_precision": 0.9172,
                "utility_spearman": 0.8198,
            },
        },
    )
    actionability = tmp_path / "actionability"
    _write_json(
        actionability / "direct_suppression_actionability_audit.json",
        {
            "selected_arm": None,
            "selection_reason": "no_policy_grid_row_passed_diagnostic_gate",
            "dominant_blocker": "nonrecurrent_positive_action_folds",
            "passing_policy_rows": 0,
            "recurrent_support_policy_rows": 18,
            "max_recurrent_defensive_success": 0.053985,
            "max_recurrent_positive_fold_share": 1 / 6,
        },
    )

    payload = audit_market_state_plan_completion(
        artifact,
        tmp_path / "out",
        state_head_quality_dir=state_quality,
        strategy_response_quality_dir=response_quality,
        shadow_priority_audit_dir=priority,
        backend_comparison_dir=backend,
        direct_suppression_training_dir=direct,
        direct_suppression_actionability_audit_dir=actionability,
    )

    checklist = pd.read_csv(tmp_path / "out" / "market_state_plan_completion_checklist.csv")
    row = checklist.loc[checklist["requirement_id"].astype(str).eq("12.6")]
    assert not row.empty
    assert row["status"].iloc[0] == "gate_blocked"
    assert "actionability_dominant_blocker=nonrecurrent_positive_action_folds" in row["notes"].iloc[0]
    assert "actionability_recurrent_support_rows=18" in row["notes"].iloc[0]
    assert "actionability_max_recurrent_positive_fold_share=0.16666666666666666" in row["notes"].iloc[0]
    assert payload["direct_suppression_actionability_dominant_blocker"] == "nonrecurrent_positive_action_folds"
    assert payload["direct_suppression_actionability_recurrent_support_rows"] == 18


def test_market_state_plan_completion_audit_accepts_root_level_promotion_audit(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    state_quality, response_quality, priority, backend = _write_minimal_bundle(artifact)
    nested = artifact / "promotion_gate_audit" / "market_state_controller_promotion_gate_audit.json"
    root_level = artifact / "market_state_controller_promotion_gate_audit.json"
    payload_text = nested.read_text(encoding="utf-8")
    nested.unlink()
    root_level.write_text(payload_text, encoding="utf-8")

    payload = audit_market_state_plan_completion(
        artifact,
        tmp_path / "out",
        state_head_quality_dir=state_quality,
        strategy_response_quality_dir=response_quality,
        shadow_priority_audit_dir=priority,
        backend_comparison_dir=backend,
    )

    assert payload["hard_failure_count"] == 0, payload["hard_failures"]


def test_market_state_plan_completion_audit_accepts_external_controller_promotion_audit(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    state_quality, response_quality, priority, backend = _write_minimal_bundle(artifact)
    nested = artifact / "promotion_gate_audit" / "market_state_controller_promotion_gate_audit.json"
    nested.unlink()
    external = tmp_path / "controller_promotion"
    _write_json(
        external / "market_state_controller_promotion_gate_audit.json",
        {"promotion_gate_passed": False, "controller_should_remain_disabled": True},
    )

    payload = audit_market_state_plan_completion(
        artifact,
        tmp_path / "out",
        controller_promotion_audit_dir=external,
        state_head_quality_dir=state_quality,
        strategy_response_quality_dir=response_quality,
        shadow_priority_audit_dir=priority,
        backend_comparison_dir=backend,
    )

    assert payload["hard_failure_count"] == 0, payload["hard_failures"]
    assert payload["controller_promotion_audit_dir"] == str(external)


def test_market_state_plan_completion_audit_accepts_selected_shadow_arm_when_execution_disabled(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    state_quality, response_quality, priority, backend = _write_minimal_bundle(artifact)
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    manifest.update(
        {
            "selected_arm": "S1_observed_axes_shared_response__post_selection_overlay",
            "controller_execution_enabled": False,
            "shadow_controller_only": True,
        }
    )
    _write_json(artifact / "manifest.json", manifest)
    _write_json(
        artifact / "walkforward_selected_controller_candidate.json",
        {"selected_arm": "S1_observed_axes_shared_response__post_selection_overlay"},
    )

    payload = audit_market_state_plan_completion(
        artifact,
        tmp_path / "out",
        state_head_quality_dir=state_quality,
        strategy_response_quality_dir=response_quality,
        shadow_priority_audit_dir=priority,
        backend_comparison_dir=backend,
    )
    checklist = pd.read_csv(tmp_path / "out" / "market_state_plan_completion_checklist.csv")
    inactive = checklist.loc[checklist["requirement_id"].astype(str).eq("0.2")]

    assert payload["hard_failure_count"] == 0, payload["hard_failures"]
    assert inactive["status"].iloc[0] == "gate_blocked"
    assert "manifest_selected_arm=S1_observed_axes_shared_response__post_selection_overlay" in inactive["notes"].iloc[0]


def test_market_state_plan_completion_audit_accepts_no_backfill_selected_overlay(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    state_quality, response_quality, priority, backend = _write_minimal_bundle(artifact)
    _write_json(
        artifact / "walkforward_selected_controller_candidate.json",
        {
            "selected_arm": "S1_observed_axes_shared_response__post_selection_overlay",
            "selection_policy": {"select_no_backfill_overlay_only": True},
            "selected_metrics": {"action_entrants": 0.0},
        },
    )
    _write_json(
        artifact / "promotion_gate_audit" / "market_state_controller_promotion_gate_audit.json",
        {"promotion_gate_passed": True, "controller_should_remain_disabled": False},
    )

    payload = audit_market_state_plan_completion(
        artifact,
        tmp_path / "out",
        state_head_quality_dir=state_quality,
        strategy_response_quality_dir=response_quality,
        shadow_priority_audit_dir=priority,
        backend_comparison_dir=backend,
    )
    checklist = pd.read_csv(tmp_path / "out" / "market_state_plan_completion_checklist.csv")
    inactive = checklist.loc[checklist["requirement_id"].astype(str).eq("0.2")]
    promotion = checklist.loc[checklist["requirement_id"].astype(str).eq("20.1")]

    assert payload["hard_failure_count"] == 0, payload["hard_failures"]
    assert inactive["status"].iloc[0] == "complete"
    assert promotion["status"].iloc[0] == "complete"
    assert "selected_no_backfill_overlay=True" in inactive["notes"].iloc[0]


def test_market_state_plan_completion_audit_blocks_selected_overlay_when_not_promotion_ready(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    state_quality, response_quality, priority, backend = _write_minimal_bundle(artifact)
    _write_json(
        artifact / "walkforward_selected_controller_candidate.json",
        {
            "selected_arm": "S1_observed_axes_shared_response__post_selection_overlay",
            "selection_policy": {"select_no_backfill_overlay_only": True},
            "selected_metrics": {"action_entrants": 0.0},
        },
    )
    _write_json(
        artifact / "promotion_gate_audit" / "market_state_controller_promotion_gate_audit.json",
        {
            "promotion_gate_passed": True,
            "controller_promotion_ready": False,
            "controller_should_remain_disabled": True,
            "action_attribution_gate": {
                "passed": False,
                "failures": ["direct_suppression_defensive_success_not_positive"],
            },
        },
    )

    payload = audit_market_state_plan_completion(
        artifact,
        tmp_path / "out",
        state_head_quality_dir=state_quality,
        strategy_response_quality_dir=response_quality,
        shadow_priority_audit_dir=priority,
        backend_comparison_dir=backend,
    )
    checklist = pd.read_csv(tmp_path / "out" / "market_state_plan_completion_checklist.csv")
    inactive = checklist.loc[checklist["requirement_id"].astype(str).eq("0.2")]
    promotion = checklist.loc[checklist["requirement_id"].astype(str).eq("20.1")]

    assert payload["hard_failure_count"] == 0, payload["hard_failures"]
    assert payload["controller_promotion_ready"] is False
    assert payload["controller_action_attribution_gate_passed"] is False
    assert inactive["status"].iloc[0] == "gate_blocked"
    assert promotion["status"].iloc[0] == "gate_blocked"
    assert "controller_promotion_ready=False" in inactive["notes"].iloc[0]
    assert "action_attribution_gate_passed=False" in promotion["notes"].iloc[0]


def test_market_state_plan_completion_audit_shadow_monitor_blocks_selected_overlay(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    state_quality, response_quality, priority, backend = _write_minimal_bundle(artifact)
    _write_json(
        artifact / "walkforward_selected_controller_candidate.json",
        {
            "selected_arm": "S1_observed_axes_shared_response__post_selection_overlay",
            "selection_policy": {"select_no_backfill_overlay_only": True},
            "selected_metrics": {"action_entrants": 0.0},
        },
    )
    _write_json(
        artifact / "promotion_gate_audit" / "market_state_controller_promotion_gate_audit.json",
        {"promotion_gate_passed": True, "controller_should_remain_disabled": False},
    )
    monitor = tmp_path / "shadow_monitor"
    _write_json(
        monitor / "shadow_controller_monitor_summary.json",
        {
            "shadow_promotion_gate_passed": False,
            "controller_should_remain_disabled": True,
            "total_shadow_realized_defensive_success": -0.1,
        },
    )

    payload = audit_market_state_plan_completion(
        artifact,
        tmp_path / "out",
        state_head_quality_dir=state_quality,
        strategy_response_quality_dir=response_quality,
        shadow_priority_audit_dir=priority,
        shadow_controller_monitor_dir=monitor,
        backend_comparison_dir=backend,
    )
    checklist = pd.read_csv(tmp_path / "out" / "market_state_plan_completion_checklist.csv")
    inactive = checklist.loc[checklist["requirement_id"].astype(str).eq("0.2")]
    promotion = checklist.loc[checklist["requirement_id"].astype(str).eq("20.1")]

    assert payload["hard_failure_count"] == 0, payload["hard_failures"]
    assert payload["shadow_controller_monitor_should_disable"] is True
    assert inactive["status"].iloc[0] == "gate_blocked"
    assert promotion["status"].iloc[0] == "gate_blocked"
    assert "shadow_monitor_should_disable=True" in inactive["notes"].iloc[0]
    assert "shadow_monitor_defensive_success=-0.1" in promotion["notes"].iloc[0]


def test_market_state_plan_completion_audit_no_backfill_monitor_blocks_selected_overlay(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    state_quality, response_quality, priority, backend = _write_minimal_bundle(artifact)
    _write_json(
        artifact / "walkforward_selected_controller_candidate.json",
        {
            "selected_arm": "S1_observed_axes_shared_response__post_selection_overlay",
            "selection_policy": {"select_no_backfill_overlay_only": True},
            "selected_metrics": {"action_entrants": 0.0},
        },
    )
    _write_json(
        artifact / "promotion_gate_audit" / "market_state_controller_promotion_gate_audit.json",
        {"promotion_gate_passed": True, "controller_should_remain_disabled": False},
    )
    monitor = tmp_path / "no_backfill_monitor"
    _write_json(
        monitor / "no_backfill_shadow_monitor_summary.json",
        {
            "promotion_gate_passed": False,
            "promotion_gate_failures": ["indirect_path_or_capacity_delta_negative"],
            "direct_threshold_only_promotion_gate_passed": True,
            "locked_accepted_overlay_promotion_gate_passed": False,
            "sum_locked_accepted_overlay_defensive_success": -0.2,
        },
    )

    payload = audit_market_state_plan_completion(
        artifact,
        tmp_path / "out",
        state_head_quality_dir=state_quality,
        strategy_response_quality_dir=response_quality,
        shadow_priority_audit_dir=priority,
        shadow_controller_monitor_dir=monitor,
        backend_comparison_dir=backend,
    )
    checklist = pd.read_csv(tmp_path / "out" / "market_state_plan_completion_checklist.csv")
    inactive = checklist.loc[checklist["requirement_id"].astype(str).eq("0.2")]
    promotion = checklist.loc[checklist["requirement_id"].astype(str).eq("20.1")]

    assert payload["hard_failure_count"] == 0, payload["hard_failures"]
    assert payload["shadow_controller_monitor_contract"] == "no_backfill_shadow_monitor_summary"
    assert payload["shadow_controller_monitor_gate_passed"] is False
    assert payload["shadow_controller_monitor_should_disable"] is True
    assert payload["shadow_controller_direct_threshold_gate_passed"] is True
    assert payload["shadow_controller_locked_overlay_gate_passed"] is False
    assert inactive["status"].iloc[0] == "gate_blocked"
    assert promotion["status"].iloc[0] == "gate_blocked"
    assert "shadow_monitor_contract=no_backfill_shadow_monitor_summary" in inactive["notes"].iloc[0]
    assert "direct_threshold_gate_passed=True" in inactive["notes"].iloc[0]
    assert "locked_overlay_gate_passed=False" in promotion["notes"].iloc[0]
    assert "shadow_monitor_defensive_success=-0.2" in promotion["notes"].iloc[0]


def test_market_state_plan_completion_audit_accepts_external_pruned_ablation_evidence(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    state_quality, response_quality, priority, backend = _write_minimal_bundle(artifact)
    replay = pd.read_csv(artifact / "portfolio_replay_summary.csv")
    replay = replay.loc[~replay["arm"].astype(str).eq("S7_pruned_state_pack")]
    replay.to_csv(artifact / "portfolio_replay_summary.csv", index=False)
    external = tmp_path / "external_ablation"
    external.mkdir()
    pd.DataFrame({"arm": ["S7_pruned_state_pack"]}).to_csv(
        external / "portfolio_replay_summary.csv", index=False
    )

    payload = audit_market_state_plan_completion(
        artifact,
        tmp_path / "out",
        state_head_quality_dir=state_quality,
        strategy_response_quality_dir=response_quality,
        shadow_priority_audit_dir=priority,
        backend_comparison_dir=backend,
        ablation_matrix_evidence_dir=external,
    )
    checklist = pd.read_csv(tmp_path / "out" / "market_state_plan_completion_checklist.csv")
    ablation = checklist.loc[checklist["requirement_id"].astype(str).eq("17.1")]

    assert payload["hard_failure_count"] == 0, payload["hard_failures"]
    assert ablation["status"].iloc[0] == "complete"
    assert f"external_dir={external}" in ablation["notes"].iloc[0]


def test_market_state_plan_completion_audit_accepts_head_priority_promotion_audit(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    state_quality, response_quality, priority, backend = _write_minimal_bundle(artifact)
    (priority / "market_state_priority_shadow_promotion_gate.json").unlink()
    _write_json(
        priority / "market_state_head_priority_promotion_gate_audit.json",
        {
            "single_window_replay_gate_passed": True,
            "passing_candidate_count": 1,
            "production_passing_candidate_count": 0,
            "priority_should_remain_shadow": True,
            "production_blockers": [
                "changes_scores_or_ranks_rank_prior_shadow_only",
                "fewer_than_3_replay_windows",
            ],
        },
    )

    payload = audit_market_state_plan_completion(
        artifact,
        tmp_path / "out",
        state_head_quality_dir=state_quality,
        strategy_response_quality_dir=response_quality,
        shadow_priority_audit_dir=priority,
        backend_comparison_dir=backend,
    )
    checklist = pd.read_csv(tmp_path / "out" / "market_state_plan_completion_checklist.csv")
    priority_row = checklist.loc[checklist["requirement_id"].astype(str).eq("21.2")]

    assert payload["hard_failure_count"] == 0, payload["hard_failures"]
    assert priority_row["status"].iloc[0] == "shadow_only"
    assert "production_passing_candidate_count=0" in priority_row["notes"].iloc[0]
    assert "changes_scores_or_ranks_rank_prior_shadow_only" in priority_row["notes"].iloc[0]


def test_market_state_plan_completion_audit_reports_recurrent_priority_rejection(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    state_quality, response_quality, priority, backend = _write_minimal_bundle(artifact)
    _write_json(
        priority / "recurrent_shadow_challenger.json",
        {
            "selected": False,
            "reason": "no_recurrent_gate_passing_arm",
            "best_candidate": {
                "arm_selector": "cap_0p6_zge_0p5",
                "fail_reasons": "fewer_than_required_action_windows;timeout_worsened_in_a_window",
                "action_window_count": 1,
                "positive_action_window_count": 1,
            },
        },
    )

    payload = audit_market_state_plan_completion(
        artifact,
        tmp_path / "out",
        state_head_quality_dir=state_quality,
        strategy_response_quality_dir=response_quality,
        shadow_priority_audit_dir=priority,
        backend_comparison_dir=backend,
    )
    checklist = pd.read_csv(tmp_path / "out" / "market_state_plan_completion_checklist.csv")
    priority_row = checklist.loc[checklist["requirement_id"].astype(str).eq("21.2")]

    assert payload["hard_failure_count"] == 0, payload["hard_failures"]
    assert payload["shadow_priority_recurrent_challenger_selected"] is False
    assert payload["shadow_priority_recurrent_selection_reason"] == "no_recurrent_gate_passing_arm"
    assert payload["shadow_priority_recurrent_best_candidate"] == "cap_0p6_zge_0p5"
    assert priority_row["status"].iloc[0] == "shadow_only"
    assert "recurrent_selected=False" in priority_row["notes"].iloc[0]
    assert "recurrent_best_candidate=cap_0p6_zge_0p5" in priority_row["notes"].iloc[0]
    assert "timeout_worsened_in_a_window" in priority_row["notes"].iloc[0]


def test_market_state_plan_completion_audit_reports_missing_bundle(tmp_path: Path) -> None:
    payload = audit_market_state_plan_completion(tmp_path / "missing", tmp_path / "out")

    assert payload["hard_failure_count"] > 0
    assert payload["passed_structural_audit"] is False
