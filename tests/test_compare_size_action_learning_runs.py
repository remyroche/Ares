from pathlib import Path

import pandas as pd

from scripts.compare_size_action_learning_runs import (
    DEFAULT_ARMS,
    _summarize_raw_candidate_recall,
    _summarize_bottlenecks,
    _summarize_promotion_gates,
    _summarize_replay,
    _summarize_stage1_recall_tradeoff,
)


def test_summarize_replay_reports_fold_aligned_delta_vs_c0(tmp_path: Path) -> None:
    run_dir = tmp_path / "exact_state_size_action_learning_unit"
    run_dir.mkdir()
    pd.DataFrame(
        [
            {"fold_id": 0, "arm": "C0_baseline", "net_pnl": 100.0, "trade_count": 10, "full_sl_rate": 0.20, "timeout_rate": 0.10},
            {"fold_id": 1, "arm": "C0_baseline", "net_pnl": 50.0, "trade_count": 8, "full_sl_rate": 0.10, "timeout_rate": 0.00},
            {"fold_id": 0, "arm": "C3da_test", "net_pnl": 125.0, "trade_count": 9, "full_sl_rate": 0.10, "timeout_rate": 0.10},
            {"fold_id": 1, "arm": "C3da_test", "net_pnl": 45.0, "trade_count": 8, "full_sl_rate": 0.10, "timeout_rate": 0.25},
        ]
    ).to_csv(run_dir / "size_action_fold_summary.csv", index=False)

    summary = _summarize_replay(run_dir, arms={"C0_baseline", "C3da_test"})
    c3da = summary.loc[summary["arm"].eq("C3da_test")].iloc[0]

    assert c3da["delta_net_pnl_sum"] == 20.0
    assert c3da["median_delta_net_pnl"] == 10.0
    assert c3da["positive_delta_folds"] == 1
    assert c3da["delta_trade_count_sum"] == -1
    assert c3da["mean_delta_full_sl_rate"] == -0.05


def test_default_arms_include_c3fl_consensus_arm() -> None:
    assert "C3fl_bagged_safety_c3ed_or_high_rank_calibrated_group_hurdle_consensus_union_gate" in DEFAULT_ARMS


def test_default_arms_include_c3fm_no_action_tail_arm() -> None:
    assert "C3fm_bagged_safety_c3ed_or_calibrated_group_hurdle_no_action_tail_union_gate" in DEFAULT_ARMS


def test_default_arms_include_c3fn_foldfit_action_acceptance_arm() -> None:
    assert "C3fn_bagged_safety_c3ed_or_calibrated_group_hurdle_foldfit_action_acceptance_union_gate" in DEFAULT_ARMS


def test_default_arms_include_c3fo_positive_value_acceptance_arm() -> None:
    assert "C3fo_bagged_safety_c3ed_or_calibrated_group_hurdle_positive_value_acceptance_union_gate" in DEFAULT_ARMS


def test_default_arms_include_c3fp_rank_calibrated_value_arm() -> None:
    assert "C3fp_bagged_safety_c3ed_or_rank_calibrated_value_secondary_union_gate" in DEFAULT_ARMS


def test_default_arms_include_c3fq_stage1_group_action_selector_arm() -> None:
    assert "C3fq_calibrated_stage1_group_action_selector" in DEFAULT_ARMS


def test_summarize_bottlenecks_reports_missed_oracle_stage1_block(tmp_path: Path) -> None:
    run_dir = tmp_path / "exact_state_size_action_learning_unit"
    run_dir.mkdir()
    key_a = {"fold_id": 0, "timestamp": "2026-05-01 00:00:00+00:00", "strategy_id": "short_asset"}
    key_b = {"fold_id": 0, "timestamp": "2026-05-01 01:00:00+00:00", "strategy_id": "short_asset"}
    pd.DataFrame(
        [
            {"arm": "C1_exact_state_oracle_full", "selected": True, "selected_delta_full_J": 10.0, **key_a},
            {"arm": "C1_exact_state_oracle_full", "selected": True, "selected_delta_full_J": 20.0, **key_b},
            {"arm": "C3db_test", "selected": True, "selected_delta_full_J": 10.0, **key_a},
            {"arm": "C3db_test", "selected": False, "selected_delta_full_J": 0.0, **key_b},
        ]
    ).to_csv(run_dir / "size_action_selector_transfer_diagnostics.csv", index=False)
    pd.DataFrame(
        [
            {
                "arm": "C3db_test",
                "p_intervene": 0.50,
                "pred_multiplier": 0.0,
                "pred_multiplier_confidence": 0.95,
                "pred_delta_J": -1.0,
                "nonoverlap_suppressed": False,
                **key_a,
            },
            {
                "arm": "C3db_test",
                "p_intervene": 0.02,
                "pred_multiplier": 0.0,
                "pred_multiplier_confidence": 0.80,
                "pred_delta_J": -1.0,
                "nonoverlap_suppressed": False,
                **key_b,
            },
        ]
    ).to_csv(run_dir / "size_action_schedules.csv", index=False)

    summary = _summarize_bottlenecks(run_dir, arms={"C3db_test"})
    row = summary.loc[summary["arm"].eq("C3db_test")].iloc[0]

    assert row["oracle_positive_groups"] == 2
    assert row["selected_positive_groups"] == 1
    assert row["missed_oracle_groups"] == 1
    assert row["missed_oracle_gain_sum"] == 20.0
    assert row["missed_stage1_low_share_p_lt_0_40"] == 1.0
    assert row["missed_multiplier_conf_low_share_lt_0_90"] == 1.0


def test_summarize_promotion_gates_reports_strict_readiness(tmp_path: Path) -> None:
    run_dir = tmp_path / "exact_state_size_action_learning_unit"
    run_dir.mkdir()
    pd.DataFrame(
        [
            {"fold_id": 0, "arm": "C0_baseline", "net_pnl": 100.0, "trade_count": 10, "full_sl_rate": 0.20, "timeout_rate": 0.10},
            {"fold_id": 1, "arm": "C0_baseline", "net_pnl": 100.0, "trade_count": 10, "full_sl_rate": 0.20, "timeout_rate": 0.10},
            {"fold_id": 0, "arm": "C3dq_test", "net_pnl": 110.0, "trade_count": 10, "full_sl_rate": 0.18, "timeout_rate": 0.10, "mean_multiplier": 0.99},
            {"fold_id": 1, "arm": "C3dq_test", "net_pnl": 105.0, "trade_count": 10, "full_sl_rate": 0.19, "timeout_rate": 0.10, "mean_multiplier": 0.99},
        ]
    ).to_csv(run_dir / "size_action_fold_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "fold_id": 0,
                "arm": "C3dq_test",
                "scheduled_groups": 100,
                "intervention_count": 5,
                "intervention_rate": 0.05,
                "positive_action_count": 5,
                "positive_action_rate": 1.0,
                "realized_delta_full_J_sum": 10.0,
                "realized_delta_immediate_J_sum": 1.0,
                "oracle_positive_group_capture_rate": 0.1,
                "oracle_gain_capture_ratio": 0.1,
            },
            {
                "fold_id": 1,
                "arm": "C3dq_test",
                "scheduled_groups": 100,
                "intervention_count": 5,
                "intervention_rate": 0.05,
                "positive_action_count": 5,
                "positive_action_rate": 1.0,
                "realized_delta_full_J_sum": 5.0,
                "realized_delta_immediate_J_sum": 1.0,
                "oracle_positive_group_capture_rate": 0.1,
                "oracle_gain_capture_ratio": 0.1,
            },
        ]
    ).to_csv(run_dir / "size_action_action_quality.csv", index=False)
    pd.DataFrame(
        [
            {"arm": "C3dq_test", "fold_id": 0, "score_bucket": 9, "rows": 10, "mean_delta_full_J": 5.0, "positive_delta_share": 1.0},
            {"arm": "C3dq_test", "fold_id": 1, "score_bucket": 9, "rows": 10, "mean_delta_full_J": 4.0, "positive_delta_share": 1.0},
        ]
    ).to_csv(run_dir / "size_action_predicted_benefit_deciles.csv", index=False)
    pd.DataFrame(
        [
            {"arm": "C3dq_test", "folds": 2, "median_delta_net_pnl": 7.5, "q25_delta_net_pnl": 6.25, "median_exposure_ratio": 0.99},
        ]
    ).to_csv(run_dir / "size_action_promotion_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "fold_id": fold_id,
                "split": "eval",
                "timestamp": f"2026-05-0{fold_id + 1} 00:{idx:02d}:00+00:00",
                "strategy_id": "a",
                "group_can_bind": 1.0,
            }
            for fold_id in (0, 1)
            for idx in range(50)
        ]
        + [
            {"fold_id": 0, "split": "train", "timestamp": "2026-04-30 00:00:00+00:00", "strategy_id": "a", "group_can_bind": 1.0},
        ]
    ).to_csv(run_dir / "size_action_exact_panel.csv", index=False)

    summary = _summarize_promotion_gates(run_dir, arms={"C3dq_test"})
    row = summary.loc[summary["arm"].eq("C3dq_test")].iloc[0]

    assert row["intervention_rate_total"] == 0.05
    assert row["binding_opportunity_groups"] == 100
    assert row["binding_intervention_rate_total"] == 0.1
    assert row["precision_total"] == 1.0
    assert bool(row["promotion_ready"])
    assert row["failed_gates"] == ""


def test_summarize_stage1_recall_tradeoff_reports_score_ceiling(tmp_path: Path) -> None:
    run_dir = tmp_path / "exact_state_size_action_learning_unit"
    run_dir.mkdir()
    pd.DataFrame(
        [
            {"fold_id": 0, "p_intervene": 0.9, "stage1_cal_mean_gain": 5.0, "y_intervene": 1, "best_gain": 10.0, "group_can_bind": 1},
            {"fold_id": 0, "p_intervene": 0.8, "stage1_cal_mean_gain": 4.0, "y_intervene": 0, "best_gain": 0.0, "group_can_bind": 1},
            {"fold_id": 0, "p_intervene": 0.1, "stage1_cal_mean_gain": 3.0, "y_intervene": 1, "best_gain": 20.0, "group_can_bind": 1},
            {"fold_id": 0, "p_intervene": 0.0, "stage1_cal_mean_gain": 2.0, "y_intervene": 0, "best_gain": 0.0, "group_can_bind": 0},
        ]
    ).to_csv(run_dir / "size_action_stage1_group_scores.csv", index=False)

    summary = _summarize_stage1_recall_tradeoff(run_dir)
    row = summary.loc[(summary["score_column"].eq("p_intervene")) & (summary["top_fraction"].eq(0.20))].iloc[0]

    assert row["eligible_groups"] == 3
    assert row["selected_groups"] == 1
    assert row["selected_positive_groups"] == 1
    assert row["precision"] == 1.0
    assert row["recall"] == 0.5


def test_summarize_raw_candidate_recall_reports_oracle_score_ceiling(tmp_path: Path) -> None:
    run_dir = tmp_path / "exact_state_size_action_learning_unit"
    run_dir.mkdir()
    rows = []
    for idx in range(20):
        is_oracle = idx in {0, 7}
        rows.append(
            {
                "fold_id": 0,
                "timestamp": f"2026-05-01 {idx:02d}:00:00+00:00",
                "strategy_id": "short_asset",
                "multiplier": 0.5,
                "action_binds": 1.0,
                "group_can_bind": 1.0,
                "y_intervene": float(is_oracle),
                "best_gain": 100.0 if is_oracle else 0.0,
                "p_intervene": 0.95 if idx == 0 else (0.20 if is_oracle else 0.10),
                "stage1_cal_positive_rate": 0.8 if idx == 0 else 0.2,
                "p_action_value_positive": 0.8 if idx == 0 else 0.1,
                "p_action_economic_positive": 0.8 if idx == 0 else 0.1,
                "cal_positive_rate": 0.8 if idx == 0 else 0.1,
                "ranker_score": 1.0 if idx == 0 else 0.0,
                "pred_delta_J": 25.0 if idx == 0 else -5.0,
                "cal_mean_delta_J": 10.0 if idx == 0 else -2.0,
                "cal_lcb_mean_delta_J": 1.0 if idx == 0 else -4.0,
                "cal_q10_delta_J": -1.0,
                "cal_q25_delta_J": 0.0 if idx == 0 else -10.0,
                "strategy_rank_q90": 0.8,
                "strategy_open_count": 0.0,
            }
        )
    pd.DataFrame(rows).to_csv(run_dir / "size_action_eval_action_scores.csv", index=False)

    summary = _summarize_raw_candidate_recall(run_dir)
    strict = summary.loc[summary["group"].eq("strict_oracle")].iloc[0]
    top = summary.loc[summary["group"].eq("top_0.05_by_calibrated_group_hurdle_score")].iloc[0]

    assert strict["groups"] == 2
    assert strict["positive_value_floor_pass_share"] == 0.5
    assert top["selected_oracle_groups"] == 1
    assert top["oracle_recall"] == 0.5
    assert top["oracle_gain_capture"] == 0.5
