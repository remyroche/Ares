from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_market_state_head_quality import audit_market_state_head_quality


def _diagnostics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "state_level": "forecast",
                "state_head": "forecast_h6_shock_up",
                "component_group": "return_shock",
                "aggregate_status": "active",
                "folds_seen": 3,
                "trained_folds": 3,
                "fallback_folds": 0,
                "shadow_disabled_folds": 0,
                "active_fold_share": 1.0,
                "fallback_fold_share": 0.0,
                "mean_source_count": 20,
                "mean_validation_rows": 120,
                "mean_validation_top_decile_lift": 0.20,
                "mean_tail_average_precision": 0.30,
                "mean_tail_ap_lift_p90": 0.12,
                "mean_tail_brier_p90": 0.20,
                "mean_tail_ece_5bin": 0.15,
                "mean_tail_false_alarm_rate_p90": 0.10,
                "mean_tail_recall_p90": 0.20,
                "collapsed_folds": 0,
                "positive_validation_lift_share": 1.0,
                "mean_oof_coverage": 0.95,
                "min_oof_coverage": 0.92,
                "mean_target_rows": 600,
                "mean_target_std": 0.50,
                "status_counts": '{"active": 3}',
                "disable_reasons": np.nan,
            },
            {
                "state_level": "forecast",
                "state_head": "forecast_h6_bad",
                "component_group": "bad_group",
                "aggregate_status": "active",
                "folds_seen": 3,
                "trained_folds": 1,
                "fallback_folds": 2,
                "shadow_disabled_folds": 0,
                "active_fold_share": 0.33,
                "fallback_fold_share": 0.67,
                "mean_source_count": 20,
                "mean_validation_rows": 10,
                "mean_validation_top_decile_lift": -0.05,
                "mean_tail_average_precision": 0.05,
                "mean_tail_ap_lift_p90": -0.01,
                "mean_tail_brier_p90": 0.60,
                "mean_tail_ece_5bin": 0.70,
                "mean_tail_false_alarm_rate_p90": 0.20,
                "mean_tail_recall_p90": 0.0,
                "collapsed_folds": 1,
                "positive_validation_lift_share": 0.0,
                "mean_oof_coverage": 0.40,
                "min_oof_coverage": 0.20,
                "mean_target_rows": 20,
                "mean_target_std": 0.0,
                "status_counts": '{"fallback": 2, "active": 1}',
                "disable_reasons": "weak_or_unstable_forecast_skill",
            },
            {
                "state_level": "observed_axis",
                "state_head": "state_shock",
                "component_group": "return_shock",
                "aggregate_status": "active",
                "folds_seen": 3,
                "trained_folds": 0,
                "fallback_folds": 0,
                "shadow_disabled_folds": 0,
                "active_fold_share": 1.0,
                "fallback_fold_share": 0.0,
                "mean_source_count": 5,
                "mean_validation_rows": 120,
                "collapsed_folds": 0,
                "mean_oof_coverage": 1.0,
                "min_oof_coverage": 1.0,
                "status_counts": '{"active": 3}',
                "disable_reasons": np.nan,
            },
        ]
    )


def _activation() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "state_head": "forecast_h6_shock_up",
                "recommended_status": "active_candidate",
                "activation_disable_reason": np.nan,
                "forecast_skill_gate_pass": True,
                "response_gate_pass": True,
                "action_gate_pass": True,
                "leave_one_out_gate_pass": True,
                "defensive_action_gate_pass": True,
                "max_abs_spearman_corr": 0.40,
                "redundant_with": np.nan,
                "redundancy_group": "return_shock",
                "redundancy_flag": False,
                "response_mean_abs_spearman": 0.20,
                "response_sign_stability": 0.80,
                "threshold_raise_share": 0.20,
                "suppressed_candidate_count": 10,
                "mean_state_ood_share": 0.01,
                "loo_median_increment_net_pnl": 10.0,
                "loo_q25_increment_net_pnl": 2.0,
                "loo_positive_increment_share": 1.0,
                "loo_state_head_defensive_success": 5.0,
                "loo_state_head_loss_avoided": 5.0,
                "loo_state_head_winner_pnl_sacrificed": 0.0,
                "loo_state_head_net_action_pnl_delta": 5.0,
            },
            {
                "state_head": "forecast_h6_bad",
                "recommended_status": "disabled_candidate",
                "activation_disable_reason": "weak_or_unstable_forecast_skill;no_positive_leave_one_out_increment",
                "forecast_skill_gate_pass": False,
                "response_gate_pass": False,
                "action_gate_pass": False,
                "leave_one_out_gate_pass": False,
                "defensive_action_gate_pass": False,
                "max_abs_spearman_corr": 0.99,
                "redundant_with": "forecast_h6_shock_up",
                "redundancy_group": "return_shock",
                "redundancy_flag": True,
                "response_mean_abs_spearman": 0.0,
                "response_sign_stability": 0.0,
                "threshold_raise_share": 0.0,
                "suppressed_candidate_count": 0,
                "mean_state_ood_share": 0.20,
                "loo_median_increment_net_pnl": -5.0,
                "loo_q25_increment_net_pnl": -10.0,
                "loo_positive_increment_share": 0.0,
                "loo_state_head_defensive_success": -2.0,
                "loo_state_head_loss_avoided": 0.0,
                "loo_state_head_winner_pnl_sacrificed": 2.0,
                "loo_state_head_net_action_pnl_delta": -2.0,
            },
            {
                "state_head": "state_shock",
                "recommended_status": "shadow",
                "activation_disable_reason": "no_positive_leave_one_out_increment",
                "forecast_skill_gate_pass": True,
                "response_gate_pass": True,
                "action_gate_pass": True,
                "leave_one_out_gate_pass": False,
                "defensive_action_gate_pass": False,
                "max_abs_spearman_corr": 0.40,
                "redundant_with": np.nan,
                "redundancy_group": "return_shock",
                "redundancy_flag": False,
                "response_mean_abs_spearman": 0.10,
                "response_sign_stability": 0.60,
                "threshold_raise_share": 0.10,
                "suppressed_candidate_count": 10,
                "mean_state_ood_share": 0.02,
                "loo_median_increment_net_pnl": 0.0,
                "loo_q25_increment_net_pnl": -1.0,
                "loo_positive_increment_share": 0.0,
                "loo_state_head_defensive_success": 0.0,
                "loo_state_head_loss_avoided": 0.0,
                "loo_state_head_winner_pnl_sacrificed": 0.0,
                "loo_state_head_net_action_pnl_delta": 0.0,
            },
        ]
    )


def test_market_state_head_quality_audit_writes_quality_outputs(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    output_dir = tmp_path / "out"
    artifact_dir.mkdir()
    _diagnostics().to_csv(artifact_dir / "market_state_head_diagnostics.csv", index=False)
    _activation().to_csv(artifact_dir / "market_state_activation_registry.csv", index=False)

    payload = audit_market_state_head_quality(artifact_dir, output_dir)

    by_head = pd.read_csv(output_dir / "market_state_head_quality_by_head.csv")
    good = by_head.loc[by_head["state_head"].eq("forecast_h6_shock_up")].iloc[0]
    bad = by_head.loc[by_head["state_head"].eq("forecast_h6_bad")].iloc[0]

    assert payload["passed"] is True
    assert payload["state_heads"] == 3
    assert payload["active_candidates"] == ["forecast_h6_shock_up"]
    assert bool(good["state_head_quality_passed"])
    assert good["state_head_quality_grade"] == "execution_candidate"
    assert not bool(bad["state_head_quality_passed"])
    assert "low_active_fold_share" in bad["state_head_quality_fail_reasons"]
    assert "collapsed_output" in bad["state_head_quality_fail_reasons"]
    assert (output_dir / "market_state_head_quality_by_group.csv").exists()
    assert (output_dir / "market_state_head_quality_report.md").exists()


def test_market_state_head_quality_audit_reports_missing_diagnostics(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"

    payload = audit_market_state_head_quality(tmp_path / "missing", output_dir)

    assert payload["passed"] is False
    assert "missing or empty" in payload["failures"][0]
