import json

import pandas as pd

from scripts.report_c3el_next_ablation_plan import build_plan


def test_build_plan_prioritizes_short_asset_guarded_labels_and_short_boll_threshold_trials(tmp_path):
    labels = tmp_path / "labels.csv"
    thresholds = tmp_path / "thresholds.csv"
    score_gates = tmp_path / "score_gates.csv"
    conditioning = tmp_path / "conditioning.csv"
    conditioning_folds = tmp_path / "conditioning_folds.csv"
    weekly_conditions = tmp_path / "weekly_conditions.csv"
    rules = tmp_path / "rules.csv"
    readiness = tmp_path / "readiness.json"
    fallback_dir = tmp_path / "fallback"
    fallback_dir.mkdir()

    pd.DataFrame(
        [
            {
                "head": "short_asset",
                "diagnosis": "sparse_low_precision_headroom",
                "current_positive_rate": 0.036,
                "relaxed_full_positive_rate": 0.085,
                "full_gain_to_worst_abs_ratio": 0.194,
            },
            {
                "head": "short_boll",
                "diagnosis": "usable_label_support",
                "current_positive_rate": 0.054,
                "relaxed_full_positive_rate": 0.096,
                "full_gain_to_worst_abs_ratio": 0.571,
            },
            {
                "head": "long_dist",
                "diagnosis": "negative_oracle_headroom",
                "current_positive_rate": 0.035,
                "relaxed_full_positive_rate": 0.128,
                "full_gain_to_worst_abs_ratio": -0.075,
            },
        ]
    ).to_csv(labels, index=False)
    pd.DataFrame(
        [
            {
                "candidate": "short_asset_default",
                "head": "short_asset",
                "diagnosis": "holdout_selection_negative",
            },
            {
                "candidate": "short_boll_combo",
                "head": "short_boll",
                "diagnosis": "missing_threshold_trial_artifact",
                "threshold_trial_eligible_count": 0,
            },
            {
                "candidate": "short_boll_guard_grid",
                "head": "short_boll",
                "diagnosis": "holdout_selection_negative",
                "threshold_trial_eligible_count": 70,
                "threshold_trial_positive_count": 0,
                "threshold_trial_best_value": -176.48,
            },
        ]
    ).to_csv(thresholds, index=False)
    pd.DataFrame(
        [
            {
                "head": "short_asset",
                "week_start": "ALL",
                "diagnosis": "gate_passes_some_groups",
                "rows": 565,
                "score_eligible_groups": 28,
                "guard_action_feature_min_groups": 0,
                "gate_kept_groups": 25,
            },
            {
                "head": "short_boll",
                "week_start": "ALL",
                "diagnosis": "gate_passes_some_groups",
                "rows": 632,
                "score_eligible_groups": 14,
                "guard_action_feature_min_groups": 11,
                "gate_kept_groups": 1,
            },
        ]
    ).to_csv(score_gates, index=False)
    pd.DataFrame(
        [
            {
                "head": "short_boll",
                "feature": "notional_exiting_4h",
                "direction": "low",
                "quantile": 0.5,
                "selected_sum_delta_full_J": 12353.29,
                "selected_positive_week_share": 0.667,
                "selected_mean_delta_full_J": 14.97,
            }
        ]
    ).to_csv(conditioning, index=False)
    pd.DataFrame(
        [
            {
                "head": "short_boll",
                "week_start": "2026-06-01T00:00:00+00:00",
                "kept_eval_groups": 13,
                "action_feature_max_guarded_eval_groups": 0,
                "action_feature_min_guarded_eval_groups": 0,
            },
            {
                "head": "short_boll",
                "week_start": "2026-06-08T00:00:00+00:00",
                "kept_eval_groups": 15,
                "action_feature_max_guarded_eval_groups": 0,
                "action_feature_min_guarded_eval_groups": 0,
            },
        ]
    ).to_csv(conditioning_folds, index=False)
    pd.DataFrame(
        [
            {
                "head": "short_boll",
                "feature": "cooldown_hours_max__mean",
                "direction": "high",
                "threshold": 21.8583,
                "selected_delta_net_pnl_sum": 11919.64,
                "selected_positive_week_share": 1.0,
                "selected_worst_delta_net_pnl": 952.61,
            }
        ]
    ).to_csv(weekly_conditions, index=False)
    pd.DataFrame(
        [
            {
                "rule": "strict__cooldown_count_lte_38_5",
                "passes_min_rows": True,
                "rows": 21,
                "positive_share": 0.8095,
                "positive_day_share": 0.7778,
                "sum_delta_full_J": 9335.26,
                "worst_delta_full_J": -687.61,
                "coverage_of_strict": 0.75,
                "score": 100.0,
            },
        ]
    ).to_csv(rules, index=False)
    pd.DataFrame(
        [
            {
                "rule_name": "rule_1",
                "rule_family": "single",
                "rule": "{}",
                "keep_count": 13,
                "positive_e50_rate": 1.0,
                "delta_full_J_sum": 8673.0,
                "delta_full_J_worst": 130.0,
                "objective": 100.0,
            }
        ]
    ).to_csv(fallback_dir / "all_filter_trials.csv", index=False)
    pd.DataFrame(
        [
            {"heldout_day": "2026-06-11", "delta_full_J_sum": 100.0},
            {"heldout_day": "2026-06-12", "delta_full_J_sum": -25.0},
        ]
    ).to_csv(fallback_dir / "leave_one_day_filter_validation.csv", index=False)
    readiness.write_text(json.dumps({"unlabeled_target_rows": 0}))

    plan = build_plan(
        label_objectives=labels,
        threshold_diagnostics=thresholds,
        score_gate_diagnostics=score_gates,
        conditioning_slices=conditioning,
        conditioning_ablation_folds=conditioning_folds,
        weekly_conditions=weekly_conditions,
        rule_candidates=rules,
        fallback_filter_dir=fallback_dir,
        readiness_manifest=readiness,
    )
    by_head = plan.set_index("head")

    assert by_head.loc["short_asset", "priority"] == "P0"
    assert by_head.loc["short_asset", "recommended_action"] == "collect_forward_labels_for_guarded_strict_rule"
    assert by_head.loc["short_asset", "best_guard_rule"] == "strict__cooldown_count_lte_38_5"
    assert by_head.loc["short_asset", "best_filter_loo_delta_full_J"] == 75.0
    assert "do not broadly relax labels" in by_head.loc["short_asset", "rationale"]

    assert by_head.loc["short_boll", "priority"] == "P1"
    assert by_head.loc["short_boll", "recommended_action"] == "redesign_short_boll_action_label_or_regime_conditioning"
    assert by_head.loc["short_boll", "threshold_diagnosis"] == "holdout_selection_negative"
    assert "zero positive holdout value" in by_head.loc["short_boll", "rationale"]
    assert "notional_exiting_4h low q0.50" in by_head.loc["short_boll", "rationale"]
    assert "non-binding" in by_head.loc["short_boll", "rationale"]
    assert "cooldown_hours_max__mean high 21.86" in by_head.loc["short_boll", "rationale"]
    assert by_head.loc["short_boll", "best_conditioning_feature"] == "notional_exiting_4h"
    assert by_head.loc["short_boll", "best_conditioning_sum_delta_full_J"] == 12353.29
    assert by_head.loc["short_boll", "conditioning_ablation_status"] == "nonbinding_feature_guard"
    assert by_head.loc["short_boll", "conditioning_ablation_kept_groups"] == 28
    assert by_head.loc["short_boll", "conditioning_ablation_guarded_groups"] == 0
    assert by_head.loc["short_boll", "best_weekly_condition_feature"] == "cooldown_hours_max__mean"
    assert by_head.loc["short_boll", "best_weekly_condition_delta_net_pnl"] == 11919.64
    assert by_head.loc["short_boll", "score_eligible_groups"] == 14
    assert by_head.loc["short_boll", "feature_guard_blocked_groups"] == 11

    assert by_head.loc["long_dist", "priority"] == "P9"
    assert by_head.loc["long_dist", "recommended_action"] == "disable_size_action_learning_keep_diagnostic"
