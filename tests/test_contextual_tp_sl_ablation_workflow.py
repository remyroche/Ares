from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pandas as pd

from scripts.build_contextual_tp_sl_ablation_dashboard import _candidate_dashboard
from scripts.build_cumulative_flat_frozen_gate_ledger import _dedupe, _post_cutoff_summary
from scripts.ablate_contextual_tp_sl_conditional_head_filters import (
    _add_condition_flags,
    _condition_mask,
)
from scripts.run_contextual_tp_sl_ablation_workflow import (
    _diagnostic_family_coverage_frame,
    _eligible_head_gate_summary_frame,
    _head_eligibility_frame,
    _live_outcome_summary_frame,
    _live_runtime_health_frame,
    _policy_evidence_by_head_frame,
    _policy_outcome_deficit_frame,
    _write_head_subset_ledger,
)
from scripts.run_latest_frozen_dual_scoring_gate_if_ready import _scan_one
from scripts.sweep_contextual_tp_sl_arm_combinations import _load_requested_combo_ids
from scripts.build_contextual_tp_sl_evidence_matrix import (
    _best_dev_rows,
    _canonical_head_from_series,
    _consistency_breakdowns,
    _deployment_verdict,
    _evidence_gap_frames,
    _family_attribution_from_scorecards,
    _guardrail_matrix,
    _head_action_opportunity_frame,
    _long_period_adequacy_matrix,
    _marginal_family_ablation_from_scorecards,
    _readiness_summary,
    _requested_reliability_family_verdict,
    _risk_profile_candidates,
    _tail_repair_frontier_from_existing_grids,
    _tail_frontier_rerun_shortlist,
    _weekly_tail_tradeoff,
)
from scripts.replay_contextual_tp_sl_oracle_label_selector import _select_rule
from scripts.replay_contextual_tp_sl_state_aware_oracle_selector import _complete_baseline_decisions


def test_contextual_filter_conditions_cover_requested_reliability_families() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-05-01", periods=8, freq="h", tz="UTC"),
            "strategy_id": ["short_asset_alpha"] * 8,
            "generated_weighted_hr_surprise_24": [0.2, 0.3, 0.4, 0.5, -1.0, -0.9, -0.8, -0.7],
            "generated_strategy_score_ood_abs_z": [0.1, 0.2, 0.3, 0.4, 2.5, 0.2, 2.8, 0.3],
            "generated_score_abs_diff_24": [0.1, 0.2, 0.3, 0.4, 0.5, 2.6, 2.7, 0.2],
            "generated_score_uncertainty_p1mp": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 2.9, 3.0],
        }
    )
    flagged = _add_condition_flags(frame, threshold_mode="full_sample", min_history=1)

    recent_plus = _condition_mask(
        flagged,
        "recent_hr_bad_and_any_ood_drift_uncertainty",
        threshold_mode="full_sample",
        min_history=1,
    )
    two_of_four = _condition_mask(
        flagged,
        "two_of_four_bad_reliability",
        threshold_mode="full_sample",
        min_history=1,
    )
    ood_uncertainty = _condition_mask(
        flagged,
        "ood_high_and_uncertainty_high",
        threshold_mode="full_sample",
        min_history=1,
    )

    assert bool(recent_plus.iloc[4]) is True
    assert bool(recent_plus.iloc[5]) is True
    assert bool(recent_plus.iloc[6]) is False
    assert bool(two_of_four.iloc[6]) is True
    assert bool(ood_uncertainty.iloc[6]) is True
    assert bool(recent_plus.iloc[7]) is False
    assert bool(ood_uncertainty.iloc[7]) is False


def test_oracle_label_selector_defaults_to_baseline_until_prior_success() -> None:
    oracle = pd.DataFrame(
        [
            {
                "candidate_rule": "weak_two_of_four_rank_m002",
                "week_start": pd.Timestamp("2026-05-04", tz="UTC"),
                "pass_full_pnl_tail_gate": True,
                "full_net_tail_positive": True,
                "delta_full_objective": 10.0,
                "delta_full_net_pnl": 20.0,
                "delta_full_weekly_q20_pnl": 2.0,
                "delta_intervention_week_net_pnl": 5.0,
            }
        ]
    )

    early, _, _ = _select_rule(
        oracle,
        rules=["weak_two_of_four_rank_m002"],
        week_start=pd.Timestamp("2026-05-04", tz="UTC"),
        lookback_weeks=4,
        mode="prior_full_gate",
        min_history=1,
        min_successes=1,
        min_median_net=0.0,
    )
    later, _, stats = _select_rule(
        oracle,
        rules=["weak_two_of_four_rank_m002"],
        week_start=pd.Timestamp("2026-05-11", tz="UTC"),
        lookback_weeks=4,
        mode="prior_full_gate",
        min_history=1,
        min_successes=1,
        min_median_net=0.0,
    )

    assert early == "baseline"
    assert later == "weak_two_of_four_rank_m002"
    assert int(stats["history_full_gate_count"]) == 1


def test_state_aware_selector_fills_missing_weeks_with_baseline() -> None:
    cache = {
        "baseline": pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2026-05-04T00:00:00Z", "2026-05-11T00:00:00Z", "2026-05-18T00:00:00Z"],
                    utc=True,
                ),
                "strategy_id": ["short_asset_alpha"] * 3,
                "symbol": ["BTC", "ETH", "SOL"],
            }
        )
    }
    decisions = pd.DataFrame(
        [
            {
                "week_start": pd.Timestamp("2026-05-11", tz="UTC").isoformat(),
                "selected_rule": "weak_recent_hr_drift_rank_m002",
            }
        ]
    )

    completed = _complete_baseline_decisions(cache, decisions)

    assert len(completed) == 3
    assert completed["selected_rule"].tolist() == [
        "baseline",
        "weak_recent_hr_drift_rank_m002",
        "baseline",
    ]
    assert completed["selection_reason"].iloc[0] == "fallback_missing_state_features"


def _diagnostic_cols(value: float | None = 0.1) -> dict[str, float | None]:
    return {
        "generated_score_uncertainty_p1mp": value,
        "generated_score_entropy": value,
        "generated_score_abs_distance_from_half": value,
        "generated_score_abs_diff_1": value,
        "generated_score_abs_diff_4": value,
        "generated_score_abs_diff_24": value,
        "generated_score_abs_minus_prev24_mean": value,
        "generated_score_prev24_std": value,
        "generated_strategy_score_shift_abs_z": value,
        "generated_strategy_score_ood_abs_z": value,
        "generated_strategy_barrier_ood_abs_z": value,
        "generated_strategy_friction_ood_abs_z": value,
        "generated_hr_surprise_24": value,
        "generated_hr_surprise_96": value,
        "generated_weighted_hr_surprise_24": value,
        "generated_weighted_hr_surprise_96": value,
        "generated_loss_rate_24": value,
        "generated_loss_rate_96": value,
    }


def test_cumulative_flat_ledger_dedupe_prefers_diagnostic_complete_duplicate() -> None:
    ts = pd.Timestamp("2026-06-27T00:00:00Z")
    rows = pd.DataFrame(
        [
            {
                "timestamp": ts,
                "strategy_id": "short_asset_alpha",
                "symbol": "BTC",
                "side": "short",
                "rank_pct": 0.91,
                "_diagnostic_finite_count": 1,
                "_source_order": 0,
            },
            {
                "timestamp": ts,
                "strategy_id": "short_asset_alpha",
                "symbol": "BTC",
                "side": "short",
                "rank_pct": 0.92,
                "_diagnostic_finite_count": 9,
                "_source_order": 1,
            },
            {
                "timestamp": ts + pd.Timedelta(hours=1),
                "strategy_id": "long_bars_alpha",
                "symbol": "ETH",
                "side": "long",
                "rank_pct": 0.81,
                "_diagnostic_finite_count": 3,
                "_source_order": 0,
            },
        ]
    )

    deduped, meta = _dedupe(rows)

    assert meta["duplicate_rows"] == 1
    assert meta["rows_after_dedupe"] == 2
    kept = deduped.loc[deduped["strategy_id"].eq("short_asset_alpha")].iloc[0]
    assert kept["rank_pct"] == 0.92
    assert kept["_diagnostic_finite_count"] == 9


def test_live_outcome_summary_frame_keeps_action_and_maturity_counts() -> None:
    summary = {
        "prediction_rows": 665,
        "trade_log_rows": 333,
        "traded_rows": 17,
        "realized_traded_rows": 3,
        "unresolved_traded_rows": 14,
        "realized_timestamps": 3,
        "realized_active_heads": 1,
        "timestamp_min": "2026-06-30T07:27:33+00:00",
        "timestamp_max": "2026-07-01T12:05:15+00:00",
        "trade_log_timestamp_max": "2026-07-01T14:06:39+00:00",
        "prediction_to_trade_log_lag_minutes": 121.4,
        "prediction_ledger_stale_vs_trade_log": True,
        "realized_timestamp_min": "2026-06-30T11:05:04+00:00",
        "realized_timestamp_max": "2026-07-01T09:06:01+00:00",
    }

    frame = _live_outcome_summary_frame(summary)

    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["prediction_rows"] == 665
    assert row["traded_rows"] == 17
    assert row["trade_log_rows"] == 333
    assert row["realized_traded_rows"] == 3
    assert row["unresolved_traded_rows"] == 14
    assert row["realized_active_heads"] == 1
    assert bool(row["prediction_ledger_stale_vs_trade_log"]) is True
    assert row["prediction_to_trade_log_lag_minutes"] == 121.4


def test_diagnostic_family_coverage_frame_reports_reliability_families() -> None:
    source = {
        "uncertainty_columns_present": 3,
        "uncertainty_columns_required": 3,
        "uncertainty_finite_rows": 100,
        "uncertainty_finite_row_rate": 1.0,
        "uncertainty_finite_cells": 300,
        "uncertainty_finite_cell_rate": 1.0,
        "drift_columns_present": 6,
        "drift_columns_required": 6,
        "drift_finite_rows": 80,
        "drift_finite_row_rate": 0.8,
        "drift_finite_cells": 450,
        "drift_finite_cell_rate": 0.75,
        "ood_columns_present": 3,
        "ood_columns_required": 3,
        "ood_finite_rows": 70,
        "ood_finite_row_rate": 0.7,
        "ood_finite_cells": 180,
        "ood_finite_cell_rate": 0.6,
        "recent_hit_rate_surprise_columns_present": 6,
        "recent_hit_rate_surprise_columns_required": 6,
        "recent_hit_rate_surprise_finite_rows": 55,
        "recent_hit_rate_surprise_finite_row_rate": 0.55,
        "recent_hit_rate_surprise_finite_cells": 240,
        "recent_hit_rate_surprise_finite_cell_rate": 0.4,
    }

    frame = _diagnostic_family_coverage_frame(source).set_index("family")

    assert set(frame.index) == {"uncertainty", "drift", "ood", "recent_hit_rate_surprise"}
    assert int(frame.loc["drift", "columns_present"]) == 6
    assert float(frame.loc["recent_hit_rate_surprise", "finite_row_rate"]) == 0.55
    assert int(frame.loc["ood", "finite_cells"]) == 180


def test_live_runtime_health_frame_reports_append_and_monitor_lag(tmp_path) -> None:
    log_path = tmp_path / "live.log"
    log_path.write_text(
        "\n".join(
            [
                "[2026-07-01 12:00:00 UTC] Prediction ledger appended: rows=10 path=data/prediction_ledger.parquet",
                "[2026-07-01 12:05:00 UTC] Monitoring 6 active positions for price action...",
                "[2026-07-01 12:05:01 UTC] INFERENCE_MONITOR_HEARTBEAT {\"active_positions\": 6}",
                "[2026-07-01 12:15:00 UTC] Prediction ledger appended: rows=4 path=data/prediction_ledger.parquet",
                "[2026-07-01 12:45:00 UTC] Monitoring 7 active positions for price action...",
                "[2026-07-01 12:45:01 UTC] INFERENCE_MONITOR_HEARTBEAT {\"active_positions\": 7}",
            ]
        )
        + "\n"
    )
    summary = {
        "timestamp_max": "2026-07-01T12:14:15+00:00",
        "trade_log_timestamp_max": "2026-07-01T13:00:00+00:00",
    }

    frame = _live_runtime_health_frame(log_path, summary)
    row = frame.iloc[0]

    assert int(row["ledger_append_events"]) == 2
    assert int(row["ledger_append_rows_total"]) == 14
    assert int(row["last_ledger_append_rows"]) == 4
    assert int(row["last_monitor_active_positions"]) == 7
    assert float(row["minutes_heartbeat_after_last_append"]) == 30.016666666666666
    assert float(row["minutes_trade_log_after_last_append"]) == 45.0


def test_eligible_head_gate_summary_frame_keeps_decision_columns(tmp_path) -> None:
    gate_dir = tmp_path / "gate"
    gate_dir.mkdir()
    pd.DataFrame(
        [
            {
                "bundle": "reliability_families",
                "tested_feature_families": "drift,recent_hit_rate_surprise",
                "baseline_trade_count": 8,
                "best_delta_pnl_variant": "q85_only",
                "best_delta_net_pnl": 0.0,
                "best_delta_full_sl_rate": 0.0,
                "max_adjusted_rows": 7,
                "max_adjusted_share": 0.008,
                "min_accepted_jaccard": 0.875,
                "total_entrants": 0,
                "total_removed": 2,
                "max_adjusted_acceptance_changed": 1,
                "promotion_ready": False,
                "failed_checks": "enough_eval_rows,enough_accepted_trades",
                "irrelevant_large_column": "drop",
            }
        ]
    ).to_csv(gate_dir / "frozen_dual_scoring_gate_summary.csv", index=False)

    frame = _eligible_head_gate_summary_frame(gate_dir)

    assert list(frame["bundle"]) == ["reliability_families"]
    assert "irrelevant_large_column" not in frame.columns
    assert int(frame.loc[0, "baseline_trade_count"]) == 8
    assert bool(frame.loc[0, "promotion_ready"]) is False


def test_evidence_matrix_keeps_best_development_rows_and_verdict(tmp_path) -> None:
    dev = pd.DataFrame(
        [
            {
                "variant": "baseline",
                "delta_vs_baseline_net_pnl": 0.0,
                "delta_vs_baseline_objective_avgweek_0p7dayq35_0p3dayq20": 0.0,
                "development_candidate_pass": False,
            },
            {
                "variant": "strong",
                "role": "high_return_gate_challenger",
                "delta_vs_baseline_net_pnl": 10.0,
                "delta_vs_baseline_full_sl_rate": -0.01,
                "delta_vs_baseline_objective_avgweek_0p7dayq35_0p3dayq20": 3.0,
                "delta_vs_baseline_weekly_q05_pnl": -100.0,
                "positive_net_month_share": 1.0,
                "positive_net_head_share": 1.0,
                "development_candidate_pass": True,
                "deployment_candidate_pass": False,
                "candidate_status": "research_candidate_waiting_frozen_evidence",
            },
            {
                "variant": "weak",
                "role": "diagnostic",
                "delta_vs_baseline_net_pnl": 20.0,
                "delta_vs_baseline_full_sl_rate": 0.01,
                "delta_vs_baseline_objective_avgweek_0p7dayq35_0p3dayq20": 1.0,
                "delta_vs_baseline_weekly_q05_pnl": 50.0,
                "positive_net_month_share": 0.2,
                "positive_net_head_share": 0.5,
                "development_candidate_pass": False,
                "deployment_candidate_pass": False,
                "candidate_status": "rejected_or_diagnostic",
            },
        ]
    )

    best = _best_dev_rows(dev, top_n=2)

    assert list(best["variant"]) == ["strong", "weak"]
    tradeoff = _weekly_tail_tradeoff(best).set_index("variant")
    assert tradeoff.loc["strong", "tail_tradeoff_note"] == "higher_pnl_but_weaker_weekly_q05"
    assert tradeoff.loc["weak", "tail_tradeoff_note"] == "mixed_tail_tradeoff"
    status, reason = _deployment_verdict(
        {"ready_sources": 0},
        pd.DataFrame([{"promotion_ready": False, "baseline_trade_count": 8}]),
    )
    assert status == "research_only_waiting_evidence"
    assert "baseline accepted=8" in reason


def test_evidence_matrix_readiness_summary_extracts_gate_counts(tmp_path) -> None:
    readiness_dir = tmp_path / "workflow" / "readiness"
    readiness_dir.mkdir(parents=True)
    payload = {
        "ready_sources": 0,
        "ran_gate": False,
        "requirements": {
            "min_post_cutoff_rows": 2000,
            "min_post_cutoff_timestamps": 40,
            "min_policy_action_rows": 50,
            "min_policy_outcome_rows": 50,
        },
        "nearest_source": {
            "rejection_reasons": "policy_action_rows_lt_50",
            "post_cutoff_rows": 1153,
            "post_cutoff_timestamps": 59,
            "policy_action_rows_estimate": 32,
            "policy_outcome_rows_estimate": 18,
            "policy_action_head_counts": '{"long_dist": 14}',
            "policy_outcome_head_counts": '{"long_dist": 12}',
            "policy_outcome_low_required_head_counts": '{"short_bollinger": 0}',
            "drift_finite_row_rate": 0.98,
            "recent_hit_rate_surprise_finite_row_rate": 1.0,
        },
    }
    (readiness_dir / "latest_flat_frozen_gate_readiness.json").write_text(
        __import__("json").dumps(payload)
    )

    summary = _readiness_summary(tmp_path / "workflow")

    assert summary["post_cutoff_rows"] == 1153
    assert summary["post_cutoff_rows_required"] == 2000
    assert summary["policy_action_rows"] == 32
    assert summary["low_required_head_counts"] == '{"short_bollinger": 0}'
    assert summary["drift_finite_row_rate"] == 0.98


def test_evidence_matrix_family_attribution_consolidates_scorecards(tmp_path) -> None:
    scorecard = tmp_path / "scorecard"
    effects = tmp_path / "effects"
    scorecard.mkdir()
    effects.mkdir()
    pd.DataFrame(
        [
            {
                "variant": "baseline",
                "family": "baseline_or_other",
                "delta_vs_baseline_net_pnl": 0.0,
                "scorecard_score": 0.0,
            },
            {
                "variant": "recent_drift_half",
                "family": "recent_hr+drift",
                "delta_vs_baseline_net_pnl": 100.0,
                "delta_vs_baseline_full_sl_rate": -0.01,
                "delta_vs_baseline_objective_avgweek_0p7dayq35_0p3dayq20": 12.0,
                "weekly_q20_pnl": 7.0,
                "scorecard_score": 150.0,
            },
        ]
    ).to_csv(scorecard / "promotion_scorecard.csv", index=False)
    pd.DataFrame(
        [
            {
                "evidence_family": "expanding",
                "variant": "recent_drift_ood",
                "family": "recent_hr+drift+OOD",
                "delta_net_pnl": 80.0,
                "delta_objective_week": 9.0,
                "delta_full_sl_rate": -0.02,
                "delta_q20_week_net_pnl": 3.0,
                "delta_q35_week_net_pnl": 4.0,
                "scorecard_score": 99.0,
            }
        ]
    ).to_csv(scorecard / "expanding_family_scorecard.csv", index=False)
    pd.DataFrame(
        [
            {
                "head": "long_bars",
                "diagnostic_family": "recent_hr_surprise",
                "label": "recent_hr_surprise_gte850_sizex250",
                "delta_tail_objective": 10.0,
                "delta_q20_day_net_pnl": 1.0,
                "delta_q35_day_net_pnl": 2.0,
            },
            {
                "head": "short_asset",
                "diagnostic_family": "recent_hr_surprise",
                "label": "recent_hr_surprise_gte850_sizex500",
                "delta_tail_objective": -5.0,
                "delta_q20_day_net_pnl": -1.0,
                "delta_q35_day_net_pnl": -2.0,
            },
        ]
    ).to_csv(effects / "best_by_head_family.csv", index=False)

    frame = _family_attribution_from_scorecards(scorecard, effects)

    assert set(frame["source"]) == {
        "promotion_scorecard",
        "expanding_family_scorecard",
        "head_family_tail_effect",
    }
    promotion = frame.loc[frame["source"].eq("promotion_scorecard")].iloc[0]
    assert promotion["family"] == "recent_hr+drift"
    assert float(promotion["delta_net_pnl"]) == 100.0
    per_head = frame.loc[frame["source"].eq("head_family_tail_effect")].iloc[0]
    assert per_head["best_head"] == "long_bars"
    assert int(per_head["positive_head_count"]) == 1
    assert int(per_head["tested_head_count"]) == 2


def test_requested_reliability_family_verdict_tracks_requested_inputs() -> None:
    attribution = pd.DataFrame(
        [
            {
                "source": "promotion_scorecard",
                "family": "recent_hr+drift",
                "best_variant": "recent_drift_half",
                "delta_net_pnl": 100.0,
                "delta_tail_objective": None,
                "delta_q20_pnl": 20.0,
                "positive_head_count": None,
                "tested_head_count": None,
            },
            {
                "source": "expanding_family_scorecard",
                "family": "recent_hr+drift+OOD",
                "best_variant": "recent_drift_ood",
                "delta_net_pnl": 50.0,
                "delta_tail_objective": None,
                "delta_q20_pnl": 3.0,
                "positive_head_count": None,
                "tested_head_count": None,
            },
            {
                "source": "head_family_tail_effect",
                "family": "uncertainty",
                "best_variant": "uncertainty_gte700",
                "delta_net_pnl": None,
                "delta_tail_objective": 8.0,
                "delta_q20_pnl": 1.0,
                "positive_head_count": 1,
                "tested_head_count": 4,
            },
        ]
    )
    readiness = {
        "drift_columns_present": 6,
        "drift_columns_required": 6,
        "drift_finite_row_rate": 0.98,
        "recent_hit_rate_surprise_columns_present": 6,
        "recent_hit_rate_surprise_columns_required": 6,
        "recent_hit_rate_surprise_finite_row_rate": 1.0,
        "ood_columns_present": 3,
        "ood_columns_required": 3,
        "ood_finite_row_rate": 0.42,
        "uncertainty_columns_present": 3,
        "uncertainty_columns_required": 3,
        "uncertainty_finite_row_rate": 0.41,
    }

    verdict = _requested_reliability_family_verdict(attribution, readiness).set_index("family")

    assert set(verdict.index) == {"drift", "recent_hit_rate_surprise", "ood", "uncertainty"}
    assert verdict.loc["drift", "verdict"] == "helpful_in_tests"
    assert verdict.loc["recent_hit_rate_surprise", "verdict"] == "helpful_in_tests"
    assert verdict.loc["ood", "verdict"] == "helpful_in_tests"
    assert verdict.loc["uncertainty", "verdict"] == "helpful_in_tests"
    assert float(verdict.loc["ood", "finite_row_rate"]) == 0.42
    assert int(verdict.loc["uncertainty", "positive_head_count"]) == 1


def test_requested_reliability_family_verdict_treats_missing_count_metadata_as_unknown() -> None:
    attribution = pd.DataFrame(
        [
            {
                "source": "promotion_scorecard",
                "family": "recent_hr+drift",
                "best_variant": "recent_drift_half",
                "delta_net_pnl": 100.0,
                "delta_tail_objective": None,
                "delta_q20_pnl": 10.0,
            }
        ]
    )
    readiness = {
        "drift_finite_row_rate": 0.98,
        "recent_hit_rate_surprise_finite_row_rate": 1.0,
        "ood_finite_row_rate": 0.42,
        "uncertainty_finite_row_rate": 0.41,
    }

    verdict = _requested_reliability_family_verdict(attribution, readiness).set_index("family")

    assert verdict.loc["drift", "verdict"] == "helpful_in_tests"
    assert verdict.loc["recent_hit_rate_surprise", "verdict"] == "helpful_in_tests"
    assert verdict.loc["ood", "verdict"] == "present_not_yet_tested"
    assert verdict.loc["uncertainty", "verdict"] == "present_not_yet_tested"


def test_marginal_family_ablation_compares_addon_families_to_simpler_variant(tmp_path) -> None:
    scorecard = tmp_path / "scorecard"
    scorecard.mkdir()
    pd.DataFrame(
        [
            {
                "variant": "baseline",
                "family": "baseline_or_other",
                "delta_net_pnl": None,
                "tail_objective_delta": None,
                "delta_full_sl_rate": None,
                "delta_q20": None,
                "delta_q35": None,
                "scorecard_score": None,
            },
            {
                "variant": "recent_hr_only",
                "family": "recent_hr",
                "delta_net_pnl": 12.0,
                "tail_objective_delta": 3.0,
                "delta_full_sl_rate": -0.01,
                "delta_q20": 2.0,
                "delta_q35": 1.0,
                "scorecard_score": 14.0,
            },
        ]
    ).assign(evidence_family="tailgrid").to_csv(scorecard / "tailgrid_recent_hr_scorecard.csv", index=False)
    pd.DataFrame(
        [
            {
                "variant": "recent_drift_half",
                "family": "recent_hr+drift",
                "delta_net_pnl": 100.0,
                "delta_objective_week": 10.0,
                "delta_full_sl_rate": -0.01,
                "delta_q20_week_net_pnl": 5.0,
                "delta_q35_week_net_pnl": 7.0,
                "scorecard_score": 120.0,
            },
            {
                "variant": "recent_drift_ood_half",
                "family": "recent_hr+drift+OOD",
                "delta_net_pnl": 130.0,
                "delta_objective_week": 11.0,
                "delta_full_sl_rate": -0.02,
                "delta_q20_week_net_pnl": 6.0,
                "delta_q35_week_net_pnl": 9.0,
                "scorecard_score": 150.0,
            },
            {
                "variant": "uncertainty_recent_drift_half",
                "family": "recent_hr+drift+uncertainty",
                "delta_net_pnl": 95.0,
                "delta_objective_week": 8.0,
                "delta_full_sl_rate": -0.01,
                "delta_q20_week_net_pnl": 4.0,
                "delta_q35_week_net_pnl": 6.0,
                "scorecard_score": 100.0,
            },
        ]
    ).assign(evidence_family="expanding").to_csv(scorecard / "expanding_family_scorecard.csv", index=False)

    frame = _marginal_family_ablation_from_scorecards(scorecard)

    ood = frame.loc[frame["family"].eq("ood")].iloc[0]
    uncertainty = frame.loc[frame["family"].eq("uncertainty")].iloc[0]
    recent = frame.loc[frame["family"].eq("recent_hit_rate_surprise")].iloc[0]
    assert recent["comparison_type"] == "marginal_vs_without_family"
    assert recent["baseline_variant"] == "baseline"
    assert float(recent["marginal_delta_net_pnl"]) == 12.0
    assert ood["comparison_type"] == "marginal_vs_without_family"
    assert ood["baseline_variant"] == "recent_drift_half"
    assert float(ood["marginal_delta_net_pnl"]) == 30.0
    assert float(ood["marginal_scorecard_score"]) == 30.0
    assert uncertainty["baseline_variant"] == "recent_drift_half"
    assert float(uncertainty["marginal_delta_net_pnl"]) == -5.0


def test_evidence_matrix_risk_profiles_rank_pnl_tail_and_churn() -> None:
    dashboard = pd.DataFrame(
        [
            {
                "variant": "baseline",
                "delta_vs_baseline_net_pnl": 0.0,
                "delta_vs_baseline_full_sl_rate": 0.0,
                "jaccard_vs_baseline": 1.0,
            },
            {
                "variant": "high_pnl",
                "role": "high_return",
                "delta_vs_baseline_net_pnl": 100.0,
                "delta_vs_baseline_full_sl_rate": -0.01,
                "delta_vs_baseline_weekly_q05_pnl": 10.0,
                "delta_vs_baseline_weekly_q10_pnl": 5.0,
                "delta_vs_baseline_weekly_q20_pnl": 20.0,
                "delta_vs_baseline_weekly_q35_pnl": 15.0,
                "positive_net_month_share": 1.0,
                "positive_net_head_share": 1.0,
                "jaccard_vs_baseline": 0.5,
            },
            {
                "variant": "tail",
                "role": "tail",
                "delta_vs_baseline_net_pnl": 40.0,
                "delta_vs_baseline_full_sl_rate": -0.03,
                "delta_vs_baseline_weekly_q05_pnl": 30.0,
                "delta_vs_baseline_weekly_q10_pnl": 20.0,
                "delta_vs_baseline_weekly_q20_pnl": 10.0,
                "delta_vs_baseline_weekly_q35_pnl": 8.0,
                "positive_net_month_share": 0.8,
                "positive_net_head_share": 0.8,
                "jaccard_vs_baseline": 0.1,
            },
            {
                "variant": "low_churn",
                "role": "low_churn",
                "delta_vs_baseline_net_pnl": 30.0,
                "delta_vs_baseline_full_sl_rate": -0.02,
                "delta_vs_baseline_weekly_q05_pnl": 12.0,
                "delta_vs_baseline_weekly_q10_pnl": 8.0,
                "delta_vs_baseline_weekly_q20_pnl": 12.0,
                "delta_vs_baseline_weekly_q35_pnl": 9.0,
                "positive_net_month_share": 0.9,
                "positive_net_head_share": 0.9,
                "jaccard_vs_baseline": 0.98,
            },
        ]
    )

    profiles = _risk_profile_candidates(dashboard)
    top = profiles.loc[profiles["risk_profile_rank"].eq(1)].set_index("risk_profile")

    assert top.loc["high_return", "variant"] == "high_pnl"
    assert top.loc["conservative_tail", "variant"] == "tail"
    assert top.loc["low_churn_tail", "variant"] == "low_churn"
    assert "baseline" not in set(profiles["variant"])


def test_long_period_adequacy_promotes_balanced_broad_candidate() -> None:
    dashboard = pd.DataFrame(
        [
            {
                "variant": "baseline",
                "role": "baseline",
                "trade_count": 6000,
            },
            {
                "variant": "broad",
                "role": "candidate",
                "trade_count": 6200,
                "months": 5,
                "heads": 4,
                "positive_net_month_share": 1.0,
                "positive_net_head_share": 1.0,
                "min_month_delta_net_pnl": 10.0,
                "min_head_delta_net_pnl": 5.0,
                "delta_vs_baseline_net_pnl": 1000.0,
                "delta_vs_baseline_hit_rate": 0.01,
                "delta_vs_baseline_full_sl_rate": -0.01,
                "delta_vs_baseline_timeout_rate": 0.0,
                "delta_vs_baseline_objective_avgweek_0p7dayq35_0p3dayq20": 20.0,
                "delta_vs_baseline_weekly_q05_pnl": 1.0,
                "delta_vs_baseline_weekly_q10_pnl": -2.0,
                "delta_vs_baseline_weekly_q20_pnl": 3.0,
                "delta_vs_baseline_weekly_q35_pnl": 4.0,
                "jaccard_vs_baseline": 0.9,
                "candidate_status": "research",
            },
            {
                "variant": "thin",
                "role": "candidate",
                "trade_count": 100,
                "months": 1,
                "heads": 1,
                "positive_net_month_share": 1.0,
                "positive_net_head_share": 1.0,
                "min_month_delta_net_pnl": 10.0,
                "min_head_delta_net_pnl": 5.0,
                "delta_vs_baseline_net_pnl": 2000.0,
                "delta_vs_baseline_full_sl_rate": -0.02,
                "delta_vs_baseline_objective_avgweek_0p7dayq35_0p3dayq20": 30.0,
                "delta_vs_baseline_weekly_q05_pnl": 10.0,
                "delta_vs_baseline_weekly_q20_pnl": 10.0,
            },
        ]
    )

    adequacy = _long_period_adequacy_matrix(dashboard).set_index("variant")

    assert bool(adequacy.loc["broad", "long_period_core_pass"]) is True
    assert bool(adequacy.loc["broad", "long_period_strict_tail_pass"]) is False
    assert bool(adequacy.loc["broad", "long_period_pragmatic_tail_pass"]) is True
    assert adequacy.loc["broad", "long_period_decision"] == (
        "pragmatic_long_period_research_candidate_q10_warning"
    )
    assert bool(adequacy.loc["thin", "long_period_core_pass"]) is False


def test_tail_repair_frontier_mines_delta_and_absolute_grids(tmp_path) -> None:
    promotion_dir = tmp_path / "contextual_tp_sl_current_candidate_promotion_table_v1_20260701"
    promotion_dir.mkdir()
    pd.DataFrame(
        [
            {
                "variant": "baseline",
                "net_pnl": 100.0,
                "weekly_q10_pnl": -10.0,
            },
            {
                "variant": "delta_candidate",
                "net_pnl": 120.0,
                "delta_vs_baseline_net_pnl": 20.0,
                "full_sl_rate": 0.2,
                "delta_vs_baseline_full_sl_rate": -0.01,
                "delta_vs_baseline_weekly_q05_pnl": 1.0,
                "delta_vs_baseline_weekly_q10_pnl": 2.0,
                "delta_vs_baseline_weekly_q20_pnl": 3.0,
                "delta_vs_baseline_weekly_q35_pnl": 4.0,
            },
        ]
    ).to_csv(promotion_dir / "candidate_promotion_summary.csv", index=False)
    combo_dir = tmp_path / "contextual_tp_sl_combo_sweep_demo"
    combo_dir.mkdir()
    pd.DataFrame(
        [
            {
                "combo_id": "combo_tail",
                "candidate_start": "2026-01-01",
                "candidate_end": "2026-06-30",
                "net_pnl": 200.0,
                "trade_count": 500,
                "full_sl_rate": 0.1,
                "weekly_q05_pnl": 1.0,
                "weekly_q10_pnl": 2.0,
                "weekly_q20_pnl": 3.0,
                "weekly_q35_pnl": 4.0,
            }
        ]
    ).to_csv(combo_dir / "head_arm_combination_summary.csv", index=False)

    frontier = _tail_repair_frontier_from_existing_grids(tmp_path)

    assert {"delta_candidate", "combo_tail"}.issubset(set(frontier["candidate_id"]))
    delta = frontier.loc[frontier["candidate_id"].eq("delta_candidate")].iloc[0]
    combo = frontier.loc[frontier["candidate_id"].eq("combo_tail")].iloc[0]
    assert delta["evidence_basis"] == "delta_vs_baseline"
    assert bool(delta["strict_weekly_tail_positive"]) is True
    assert combo["evidence_basis"] == "absolute_not_baseline_aligned"
    assert bool(combo["q10_positive"]) is True


def test_tail_frontier_rerun_shortlist_writes_combo_file_and_command(tmp_path) -> None:
    source = tmp_path / "reports/contextual_tp_sl_combo_sweep_demo"
    source.mkdir(parents=True)
    (source / "head_arm_combination_summary.json").write_text(
        __import__("json").dumps({"source_dir": "data_perp/reports/source_candidates"})
    )
    frontier = pd.DataFrame(
        [
            {
                "source_name": "contextual_tp_sl_combo_sweep_demo",
                "evidence_basis": "absolute_not_baseline_aligned",
                "candidate_id": "long_bars:S_long_dist:R_short_asset:R_short_bollinger:R",
                "candidate_start": "2026-01-01",
                "candidate_end": "2026-06-30",
                "trade_count": 100,
                "net_pnl": 1000.0,
                "weekly_q05_pnl": 1.0,
                "weekly_q10_pnl": 2.0,
                "weekly_q20_pnl": 3.0,
                "weekly_q35_pnl": 4.0,
                "full_sl_rate": 0.2,
                "frontier_score": 10.0,
                "strict_weekly_tail_positive": True,
            }
        ]
    )
    out = tmp_path / "reports/out"

    shortlist = _tail_frontier_rerun_shortlist(frontier, tmp_path / "reports", out)

    row = shortlist.iloc[0]
    assert row["source_dir"] == "data_perp/reports/source_candidates"
    assert "--combo-file" in row["rerun_command"]
    combo_file = tmp_path / "reports" / row["combo_file"].split("data_perp/reports/", 1)[-1]
    if not combo_file.exists():
        combo_file = out / "tail_frontier_combo_files/contextual_tp_sl_combo_sweep_demo_combo_ids.csv"
    assert combo_file.exists()
    assert "long_bars:S_long_dist:R_short_asset:R_short_bollinger:R" in combo_file.read_text()


def test_sweep_combo_subset_loader_accepts_cli_and_csv(tmp_path) -> None:
    combo_file = tmp_path / "combos.csv"
    pd.DataFrame({"combo_id": ["long_bars:S_long_dist:R_short_asset:R_short_bollinger:R", ""]}).to_csv(
        combo_file,
        index=False,
    )

    combos = _load_requested_combo_ids(["long_bars:J_long_dist:R_short_asset:R_short_bollinger:R"], combo_file)

    assert combos == {
        "long_bars:S_long_dist:R_short_asset:R_short_bollinger:R",
        "long_bars:J_long_dist:R_short_asset:R_short_bollinger:R",
    }


def test_evidence_matrix_guardrails_expose_tail_shortfalls() -> None:
    dashboard = pd.DataFrame(
        [
            {
                "variant": "baseline",
                "delta_vs_baseline_net_pnl": 0.0,
            },
            {
                "variant": "strict",
                "delta_vs_baseline_net_pnl": 100.0,
                "delta_vs_baseline_full_sl_rate": -0.01,
                "delta_vs_baseline_objective_avgweek_0p7dayq35_0p3dayq20": 10.0,
                "delta_vs_baseline_weekly_q05_pnl": 1.0,
                "delta_vs_baseline_weekly_q10_pnl": 2.0,
                "delta_vs_baseline_weekly_q20_pnl": 3.0,
                "delta_vs_baseline_weekly_q35_pnl": 4.0,
            },
            {
                "variant": "warning",
                "delta_vs_baseline_net_pnl": 200.0,
                "delta_vs_baseline_full_sl_rate": -0.02,
                "delta_vs_baseline_objective_avgweek_0p7dayq35_0p3dayq20": 20.0,
                "delta_vs_baseline_weekly_q05_pnl": 5.0,
                "delta_vs_baseline_weekly_q10_pnl": -1.0,
                "delta_vs_baseline_weekly_q20_pnl": 6.0,
                "delta_vs_baseline_weekly_q35_pnl": 7.0,
            },
            {
                "variant": "bad_tail",
                "delta_vs_baseline_net_pnl": 300.0,
                "delta_vs_baseline_full_sl_rate": -0.03,
                "delta_vs_baseline_objective_avgweek_0p7dayq35_0p3dayq20": 30.0,
                "delta_vs_baseline_weekly_q05_pnl": -5.0,
                "delta_vs_baseline_weekly_q10_pnl": -1.0,
                "delta_vs_baseline_weekly_q20_pnl": 6.0,
                "delta_vs_baseline_weekly_q35_pnl": 7.0,
            },
        ]
    )

    matrix = _guardrail_matrix(dashboard).set_index("variant")

    assert bool(matrix.loc["strict", "strict_tail_pass"]) is True
    assert matrix.loc["strict", "guardrail_label"] == "strict_pnl_tail_pass"
    assert bool(matrix.loc["warning", "strict_tail_pass"]) is False
    assert bool(matrix.loc["warning", "pragmatic_tail_pass"]) is True
    assert matrix.loc["warning", "guardrail_label"] == "pnl_tail_candidate_with_tail_warning"
    assert matrix.loc["warning", "tail_shortfall_fields"] == "weekly_q10_pnl"
    assert matrix.loc["bad_tail", "guardrail_label"] == "pnl_candidate_tail_incomplete"


def test_evidence_matrix_consistency_breakdowns_filter_to_top_candidates(tmp_path) -> None:
    pd.DataFrame(
        [
            {
                "variant": "keep",
                "months": 5,
                "positive_net_months": 5,
                "positive_net_month_share": 1.0,
                "mean_month_delta_net_pnl": 10.0,
                "min_month_delta_net_pnl": 2.0,
            },
            {
                "variant": "drop",
                "months": 5,
                "positive_net_months": 2,
                "positive_net_month_share": 0.4,
                "mean_month_delta_net_pnl": -1.0,
                "min_month_delta_net_pnl": -5.0,
            },
        ]
    ).to_csv(tmp_path / "candidate_monthly_consistency.csv", index=False)
    pd.DataFrame(
        [
            {
                "variant": "keep",
                "heads": 4,
                "positive_net_heads": 4,
                "positive_net_head_share": 1.0,
                "mean_head_delta_net_pnl": 12.0,
                "min_head_delta_net_pnl": 3.0,
            },
            {
                "variant": "drop",
                "heads": 4,
                "positive_net_heads": 1,
                "positive_net_head_share": 0.25,
                "mean_head_delta_net_pnl": -2.0,
                "min_head_delta_net_pnl": -6.0,
            },
        ]
    ).to_csv(tmp_path / "candidate_head_consistency.csv", index=False)
    best = pd.DataFrame([{"variant": "keep"}])

    monthly, head = _consistency_breakdowns(tmp_path, best)

    assert list(monthly["variant"]) == ["keep"]
    assert int(monthly.loc[0, "months"]) == 5
    assert list(head["variant"]) == ["keep"]
    assert int(head.loc[0, "heads"]) == 4


def test_evidence_matrix_evidence_gap_frames_report_global_and_head_deficits(tmp_path) -> None:
    readiness_dir = tmp_path / "workflow" / "readiness"
    readiness_dir.mkdir(parents=True)
    payload = {
        "requirements": {
            "min_post_cutoff_rows": 2000,
            "min_post_cutoff_timestamps": 40,
            "min_post_cutoff_active_heads": 3,
            "min_policy_action_rows": 50,
            "min_policy_action_timestamps": 10,
            "min_policy_outcome_rows": 50,
            "min_policy_outcome_timestamps": 10,
            "min_policy_outcome_rows_per_action_head": 3,
            "min_policy_outcome_rows_per_required_head": 3,
            "required_policy_outcome_head": ["long_bars", "short_bollinger"],
        },
        "nearest_source": {
            "post_cutoff_rows": 1153,
            "post_cutoff_timestamps": 59,
            "post_cutoff_active_heads": 4,
            "policy_action_rows_estimate": 32,
            "policy_action_timestamps_estimate": 20,
            "policy_outcome_rows_estimate": 18,
            "policy_outcome_timestamps_estimate": 12,
            "policy_action_head_counts": '{"long_bars": 3, "short_asset": 15}',
            "policy_outcome_head_counts": '{"long_bars": 3}',
        },
    }
    (readiness_dir / "latest_flat_frozen_gate_readiness.json").write_text(
        __import__("json").dumps(payload)
    )

    gap, head_gap = _evidence_gap_frames(tmp_path / "workflow")
    gate_gap = gap.set_index("gate")
    heads = head_gap.set_index("head")

    assert int(gate_gap.loc["post_cutoff_rows", "deficit"]) == 847
    assert bool(gate_gap.loc["post_cutoff_timestamps", "pass"]) is True
    assert int(gate_gap.loc["policy_action_rows", "deficit"]) == 18
    assert int(gate_gap.loc["policy_outcome_rows", "deficit"]) == 32
    assert int(heads.loc["short_bollinger", "matured_outcome_deficit"]) == 3
    assert heads.loc["short_bollinger", "status"] == "needs_policy_action_and_matured_outcomes"
    assert int(heads.loc["short_asset", "matured_outcome_deficit"]) == 3
    assert heads.loc["short_asset", "status"] == "needs_matured_outcomes"
    assert heads.loc["long_bars", "status"] == "ready"


def test_evidence_matrix_canonical_head_maps_short_boll_alias() -> None:
    heads = pd.Series(["short_boll", "", "long_dist", None])
    strategies = pd.Series(["", "short_bollinger_alpha", "", "short_asset_alpha"])

    mapped = _canonical_head_from_series(heads, strategies)

    assert list(mapped) == ["short_bollinger", "short_bollinger", "long_dist", "short_asset"]


def test_evidence_matrix_head_action_opportunity_reports_rank_blockers(tmp_path) -> None:
    workflow = tmp_path / "workflow"
    readiness_dir = workflow / "readiness"
    ledger_dir = workflow / "ledger"
    readiness_dir.mkdir(parents=True)
    ledger_dir.mkdir()
    ledger = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-06-27T00:00:00Z"),
                "head": "short_boll",
                "strategy_id": "short_bollinger_alpha",
                "rank_pct": 0.99,
                "portfolio_decision": None,
                "net_return": 0.01,
            },
            {
                "timestamp": pd.Timestamp("2026-06-27T01:00:00Z"),
                "head": "short_bollinger",
                "strategy_id": "short_bollinger_alpha",
                "adjusted_rank_score": 0.41,
                "portfolio_decision": "rank_rejected",
            },
            {
                "timestamp": pd.Timestamp("2026-06-27T00:00:00Z"),
                "head": "",
                "strategy_id": "short_asset_alpha",
                "adjusted_rank_score": 0.92,
                "portfolio_decision": "traded",
                "was_traded": True,
                "live_replay_net_return": 0.02,
            },
        ]
    )
    ledger_path = ledger_dir / "candidates.parquet"
    ledger.to_parquet(ledger_path, index=False)
    payload = {
        "cutoff": "2026-06-27T00:00:00+00:00",
        "nearest_source": {"path": str(ledger_path)},
        "requirements": {},
    }
    (readiness_dir / "latest_flat_frozen_gate_readiness.json").write_text(
        __import__("json").dumps(payload)
    )

    frame = _head_action_opportunity_frame(workflow).set_index("head")

    assert int(frame.loc["short_bollinger", "candidate_rows"]) == 2
    assert int(frame.loc["short_bollinger", "policy_action_rows"]) == 0
    assert int(frame.loc["short_bollinger", "rank_rejected_rows"]) == 1
    assert frame.loc["short_bollinger", "action_blocker_hint"] == "rank_rejected_below_policy_threshold"
    assert int(frame.loc["short_asset", "policy_action_rows"]) == 1
    assert frame.loc["short_asset", "action_blocker_hint"] == "has_policy_actions"


def test_policy_evidence_by_head_frame_reports_action_and_outcome_balance() -> None:
    source = {
        "policy_action_head_counts": '{"long_bars": 3, "short_asset": 10}',
        "policy_outcome_head_counts": '{"short_asset": 4}',
    }

    frame = _policy_evidence_by_head_frame(source).set_index("head")

    assert int(frame.loc["long_bars", "policy_action_rows"]) == 3
    assert int(frame.loc["long_bars", "matured_outcome_rows"]) == 0
    assert float(frame.loc["long_bars", "matured_per_action_rate"]) == 0.0
    assert int(frame.loc["short_asset", "policy_action_rows"]) == 10
    assert int(frame.loc["short_asset", "matured_outcome_rows"]) == 4
    assert float(frame.loc["short_asset", "matured_per_action_rate"]) == 0.4


def test_policy_outcome_deficit_frame_reports_blocking_heads() -> None:
    source = {
        "policy_outcome_low_action_head_counts": '{"short_asset": 3}',
        "policy_outcome_low_required_head_counts": '{"short_bollinger": 0}',
    }
    requirements = {
        "min_policy_outcome_rows_per_action_head": 5,
        "min_policy_outcome_rows_per_required_head": 2,
    }

    frame = _policy_outcome_deficit_frame(source, requirements).set_index(["gate", "head"])

    assert int(frame.loc[("action_head_minimum", "short_asset"), "observed_matured_outcomes"]) == 3
    assert int(frame.loc[("action_head_minimum", "short_asset"), "required_matured_outcomes"]) == 5
    assert int(frame.loc[("required_head_minimum", "short_bollinger"), "observed_matured_outcomes"]) == 0
    assert int(frame.loc[("required_head_minimum", "short_bollinger"), "required_matured_outcomes"]) == 2


def test_head_eligibility_frame_labels_required_and_eligible_heads() -> None:
    source = {
        "policy_action_head_counts": '{"long_dist": 14, "short_asset": 15}',
        "policy_outcome_head_counts": '{"long_dist": 12, "short_asset": 3}',
    }
    requirements = {
        "required_policy_outcome_head": ["long_dist", "short_asset", "short_bollinger"],
        "min_policy_outcome_rows_per_required_head": 3,
        "min_policy_outcome_rows_per_action_head": 3,
    }

    frame = _head_eligibility_frame(source, requirements).set_index("head")

    assert frame.loc["long_dist", "head_evidence_status"] == "eligible"
    assert frame.loc["long_dist", "matured_outcomes_needed"] == 0
    assert frame.loc["short_asset", "head_evidence_status"] == "eligible"
    assert frame.loc["short_bollinger", "head_evidence_status"] == "needs_action_evidence"
    assert frame.loc["short_bollinger", "matured_outcomes_needed"] == 3


def test_write_head_subset_ledger_filters_to_eligible_heads(tmp_path) -> None:
    ledger = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-06-27T00:00:00Z"),
                "strategy_id": "long_dist_alpha",
                "symbol": "BTC",
            },
            {
                "timestamp": pd.Timestamp("2026-06-27T01:00:00Z"),
                "strategy_id": "short_asset_alpha",
                "symbol": "ETH",
            },
            {
                "timestamp": pd.Timestamp("2026-06-27T02:00:00Z"),
                "strategy_id": "short_bollinger_alpha",
                "symbol": "SOL",
            },
            {
                "timestamp": pd.Timestamp("2026-06-27T03:00:00Z"),
                "strategy_id": "short_boll_alpha",
                "symbol": "XRP",
            },
        ]
    )
    source = tmp_path / "ledger.parquet"
    output = tmp_path / "eligible.parquet"
    ledger.to_parquet(source, index=False)

    meta = _write_head_subset_ledger(source, output, ["long_dist", "short_asset", "short_bollinger"])
    subset = pd.read_parquet(output)

    assert meta["enabled"] is True
    assert meta["input_rows"] == 4
    assert meta["output_rows"] == 4
    assert meta["dropped_rows"] == 0
    assert set(subset["strategy_id"]) == {
        "long_dist_alpha",
        "short_asset_alpha",
        "short_bollinger_alpha",
        "short_boll_alpha",
    }


def test_cumulative_ledger_post_cutoff_summary_canonicalizes_short_boll_alias() -> None:
    frame = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-06-27T00:00:00Z"),
                "strategy_id": "short_boll_alpha",
            },
            {
                "timestamp": pd.Timestamp("2026-06-27T01:00:00Z"),
                "strategy_id": "short_bollinger_alpha",
            },
            {
                "timestamp": pd.Timestamp("2026-06-27T02:00:00Z"),
                "strategy_id": "long_dist_alpha",
            },
        ]
    )

    summary = _post_cutoff_summary(frame, "2026-06-27T00:00:00+00:00")

    assert summary["post_cutoff_rows"] == 3
    assert summary["post_cutoff_active_heads"] == 2


def test_flat_frozen_gate_readiness_requires_enough_rows_timestamps_and_heads(tmp_path) -> None:
    diagnostic_cols = _diagnostic_cols()
    rows = []
    for idx, ts in enumerate(pd.date_range("2026-06-27", periods=4, freq="h", tz="UTC")):
        for head in ("short_asset", "long_bars"):
            rows.append(
                {
                    "timestamp": ts,
                    "strategy_id": f"{head}_alpha",
                    "symbol": f"SYM{idx}",
                    "auction_rank_score": 0.8 if idx < 2 and head == "short_asset" else None,
                    **diagnostic_cols,
                }
            )
    path = tmp_path / "candidates.parquet"
    pd.DataFrame(rows).to_parquet(path)
    args = Namespace(
        min_diagnostic_group_features=1,
        min_post_cutoff_rows=20,
        min_post_cutoff_timestamps=6,
        min_post_cutoff_active_heads=3,
        min_policy_action_rows=3,
        min_policy_action_timestamps=3,
        min_policy_outcome_rows=3,
        min_policy_outcome_timestamps=3,
        min_policy_outcome_rows_per_action_head=0,
        required_policy_outcome_head=None,
        min_policy_outcome_rows_per_required_head=1,
        min_diagnostic_group_finite_rate=0.25,
    )

    record = _scan_one(path, pd.Timestamp("2026-06-27T00:00:00Z"), args)

    assert record["ready"] is False
    assert "post_cutoff_rows_lt_20" in record["rejection_reasons"]
    assert "post_cutoff_timestamps_lt_6" in record["rejection_reasons"]
    assert "post_cutoff_active_heads_lt_3" in record["rejection_reasons"]
    assert "policy_action_rows_lt_3" in record["rejection_reasons"]
    assert "policy_action_timestamps_lt_3" in record["rejection_reasons"]
    assert "policy_outcome_rows_lt_3" in record["rejection_reasons"]
    assert "policy_outcome_timestamps_lt_3" in record["rejection_reasons"]
    assert record["policy_action_rows_estimate"] == 2
    assert record["policy_action_timestamps_estimate"] == 2
    assert record["policy_action_head_counts"] == '{"short_asset": 2}'
    assert record["policy_action_estimate_source"] == "finite_auction_rank_score"
    assert record["policy_outcome_rows_estimate"] == 0
    assert record["policy_outcome_timestamps_estimate"] == 0
    assert record["policy_outcome_head_counts"] == "{}"
    assert record["uncertainty_columns_present"] == 3
    assert record["drift_columns_present"] == 6
    assert record["ood_columns_present"] == 3
    assert record["recent_hit_rate_surprise_columns_present"] == 6
    assert record["uncertainty_finite_row_rate"] == 1.0
    assert record["drift_finite_row_rate"] == 1.0
    assert record["ood_finite_row_rate"] == 1.0
    assert record["recent_hit_rate_surprise_finite_row_rate"] == 1.0


def test_flat_frozen_gate_policy_action_estimate_combines_replay_and_live_rows(tmp_path) -> None:
    diagnostic_cols = _diagnostic_cols()
    rows = [
        {
            "timestamp": pd.Timestamp("2026-06-27T00:00:00Z"),
            "strategy_id": "short_asset_alpha",
            "symbol": "BTC",
            "auction_rank_score": 0.82,
            "net_return": 0.01,
            **diagnostic_cols,
        },
        {
            "timestamp": pd.Timestamp("2026-06-27T01:00:00Z"),
            "strategy_id": "long_bars_alpha",
            "symbol": "ETH",
            "auction_rank_score": 0.81,
            "net_return": -0.02,
            **diagnostic_cols,
        },
        {
            "timestamp": pd.Timestamp("2026-06-30T08:00:00Z"),
            "strategy_id": "short_asset_alpha",
            "symbol": "SOL",
            "adjusted_rank_score": 0.93,
            "was_traded": True,
            "portfolio_decision": "traded",
            **diagnostic_cols,
        },
        {
            "timestamp": pd.Timestamp("2026-06-30T09:00:00Z"),
            "strategy_id": "short_asset_alpha",
            "symbol": "XRP",
            "adjusted_rank_score": 0.91,
            "was_traded": False,
            "portfolio_decision": "portfolio_rejected",
            **diagnostic_cols,
        },
    ]
    path = tmp_path / "mixed_candidates.parquet"
    pd.DataFrame(rows).to_parquet(path)
    args = Namespace(
        min_diagnostic_group_features=1,
        min_post_cutoff_rows=4,
        min_post_cutoff_timestamps=4,
        min_post_cutoff_active_heads=2,
        min_policy_action_rows=3,
        min_policy_action_timestamps=3,
        min_policy_outcome_rows=2,
        min_policy_outcome_timestamps=2,
        min_policy_outcome_rows_per_action_head=0,
        required_policy_outcome_head=None,
        min_policy_outcome_rows_per_required_head=1,
        min_diagnostic_group_finite_rate=0.25,
    )

    record = _scan_one(path, pd.Timestamp("2026-06-27T00:00:00Z"), args)

    assert record["ready"] is True
    assert record["policy_action_rows_estimate"] == 3
    assert record["policy_action_timestamps_estimate"] == 3
    assert record["policy_outcome_rows_estimate"] == 2
    assert record["policy_outcome_timestamps_estimate"] == 2
    assert record["policy_action_head_counts"] == '{"long_bars": 1, "short_asset": 2}'
    assert record["policy_outcome_head_counts"] == '{"long_bars": 1, "short_asset": 1}'
    assert "finite_auction_rank_score" in record["policy_action_estimate_source"]
    assert "was_traded_true" in record["policy_action_estimate_source"]
    assert record["policy_outcome_estimate_source"] == "finite_net_return"


def test_flat_frozen_gate_canonicalizes_short_boll_alias_for_required_head(tmp_path) -> None:
    diagnostic_cols = _diagnostic_cols()
    rows = [
        {
            "timestamp": pd.Timestamp("2026-06-27T00:00:00Z"),
            "strategy_id": "short_boll_alpha",
            "symbol": "BTC",
            "auction_rank_score": 0.82,
            "net_return": 0.01,
            **diagnostic_cols,
        },
        {
            "timestamp": pd.Timestamp("2026-06-27T01:00:00Z"),
            "strategy_id": "short_bollinger_alpha",
            "symbol": "ETH",
            "auction_rank_score": 0.83,
            "net_return": 0.02,
            **diagnostic_cols,
        },
        {
            "timestamp": pd.Timestamp("2026-06-27T02:00:00Z"),
            "strategy_id": "short_boll_alpha",
            "symbol": "SOL",
            "auction_rank_score": 0.84,
            "net_return": 0.03,
            **diagnostic_cols,
        },
    ]
    path = tmp_path / "short_boll_alias.parquet"
    pd.DataFrame(rows).to_parquet(path)
    args = Namespace(
        min_diagnostic_group_features=1,
        min_post_cutoff_rows=3,
        min_post_cutoff_timestamps=3,
        min_post_cutoff_active_heads=1,
        min_policy_action_rows=3,
        min_policy_action_timestamps=3,
        min_policy_outcome_rows=3,
        min_policy_outcome_timestamps=3,
        min_policy_outcome_rows_per_action_head=0,
        required_policy_outcome_head=["short_bollinger"],
        min_policy_outcome_rows_per_required_head=3,
        min_diagnostic_group_finite_rate=0.25,
    )

    record = _scan_one(path, pd.Timestamp("2026-06-27T00:00:00Z"), args)

    assert record["ready"] is True
    assert record["post_cutoff_active_heads"] == 1
    assert record["policy_action_head_counts"] == '{"short_bollinger": 3}'
    assert record["policy_outcome_head_counts"] == '{"short_bollinger": 3}'
    assert record["policy_outcome_low_required_head_counts"] == "{}"


def test_flat_frozen_gate_can_require_per_head_matured_outcomes(tmp_path) -> None:
    diagnostic_cols = _diagnostic_cols()
    rows = [
        {
            "timestamp": pd.Timestamp("2026-06-27T00:00:00Z"),
            "strategy_id": "short_asset_alpha",
            "symbol": "BTC",
            "auction_rank_score": 0.82,
            "net_return": 0.01,
            **diagnostic_cols,
        },
        {
            "timestamp": pd.Timestamp("2026-06-27T01:00:00Z"),
            "strategy_id": "short_asset_alpha",
            "symbol": "ETH",
            "auction_rank_score": 0.81,
            **diagnostic_cols,
        },
        {
            "timestamp": pd.Timestamp("2026-06-27T02:00:00Z"),
            "strategy_id": "long_dist_alpha",
            "symbol": "SOL",
            "auction_rank_score": 0.83,
            "net_return": 0.02,
            **diagnostic_cols,
        },
    ]
    path = tmp_path / "head_gate_candidates.parquet"
    pd.DataFrame(rows).to_parquet(path)
    args = Namespace(
        min_diagnostic_group_features=1,
        min_post_cutoff_rows=3,
        min_post_cutoff_timestamps=3,
        min_post_cutoff_active_heads=2,
        min_policy_action_rows=3,
        min_policy_action_timestamps=3,
        min_policy_outcome_rows=2,
        min_policy_outcome_timestamps=2,
        min_policy_outcome_rows_per_action_head=2,
        required_policy_outcome_head=["short_bollinger"],
        min_policy_outcome_rows_per_required_head=1,
        min_diagnostic_group_finite_rate=0.25,
    )

    record = _scan_one(path, pd.Timestamp("2026-06-27T00:00:00Z"), args)

    assert record["ready"] is False
    assert "policy_outcome_rows_per_action_head_lt_2:long_dist,short_asset" in record["rejection_reasons"]
    assert "policy_outcome_rows_per_required_head_lt_1:short_bollinger" in record["rejection_reasons"]
    assert record["policy_outcome_low_action_head_counts"] == '{"long_dist": 1, "short_asset": 1}'
    assert record["policy_outcome_required_heads"] == '["short_bollinger"]'
    assert record["policy_outcome_low_required_head_counts"] == '{"short_bollinger": 0}'


def test_flat_frozen_gate_rejects_schema_only_diagnostic_families(tmp_path) -> None:
    rows = []
    for idx, ts in enumerate(pd.date_range("2026-06-27", periods=4, freq="h", tz="UTC")):
        diagnostics = _diagnostic_cols()
        for col in (
            "generated_score_uncertainty_p1mp",
            "generated_score_entropy",
            "generated_score_abs_distance_from_half",
            "generated_strategy_score_ood_abs_z",
            "generated_strategy_barrier_ood_abs_z",
            "generated_strategy_friction_ood_abs_z",
        ):
            diagnostics[col] = None
        rows.append(
            {
                "timestamp": ts,
                "strategy_id": "short_asset_alpha",
                "symbol": f"SYM{idx}",
                "auction_rank_score": 0.8,
                "net_return": 0.01,
                **diagnostics,
            }
        )
    path = tmp_path / "schema_only_diagnostics.parquet"
    pd.DataFrame(rows).to_parquet(path)
    args = Namespace(
        min_diagnostic_group_features=1,
        min_post_cutoff_rows=4,
        min_post_cutoff_timestamps=4,
        min_post_cutoff_active_heads=1,
        min_policy_action_rows=4,
        min_policy_action_timestamps=4,
        min_policy_outcome_rows=4,
        min_policy_outcome_timestamps=4,
        min_policy_outcome_rows_per_action_head=0,
        required_policy_outcome_head=None,
        min_policy_outcome_rows_per_required_head=1,
        min_diagnostic_group_finite_rate=0.25,
    )

    record = _scan_one(path, pd.Timestamp("2026-06-27T00:00:00Z"), args)

    assert record["ready"] is False
    assert "low_diagnostic_group_finite_rate:uncertainty,ood" in record["rejection_reasons"]
    assert record["uncertainty_columns_present"] == 3
    assert record["ood_columns_present"] == 3
    assert record["uncertainty_finite_row_rate"] == 0.0
    assert record["ood_finite_row_rate"] == 0.0
    assert record["drift_finite_row_rate"] == 1.0
    assert record["recent_hit_rate_surprise_finite_row_rate"] == 1.0


def test_dashboard_separates_research_candidate_from_deployment_candidate() -> None:
    promotion = pd.DataFrame(
        [
            {
                "variant": "candidate_a",
                "delta_vs_baseline_net_pnl": 100.0,
                "delta_vs_baseline_full_sl_rate": -0.01,
                "delta_vs_baseline_objective_avgweek_0p7dayq35_0p3dayq20": 5.0,
            },
            {
                "variant": "candidate_b",
                "delta_vs_baseline_net_pnl": 100.0,
                "delta_vs_baseline_full_sl_rate": 0.02,
                "delta_vs_baseline_objective_avgweek_0p7dayq35_0p3dayq20": 5.0,
            },
        ]
    )
    monthly = pd.DataFrame(
        [
            {"variant": "candidate_a", "positive_net_month_share": 1.0},
            {"variant": "candidate_b", "positive_net_month_share": 1.0},
        ]
    )
    heads = pd.DataFrame(
        [
            {"variant": "candidate_a", "positive_net_head_share": 1.0},
            {"variant": "candidate_b", "positive_net_head_share": 1.0},
        ]
    )
    readiness = {
        "ready_sources": 0,
        "requirements": {
            "min_post_cutoff_rows": 2000,
            "min_post_cutoff_timestamps": 40,
            "min_post_cutoff_active_heads": 3,
            "min_policy_action_rows": 50,
            "min_policy_action_timestamps": 10,
            "min_policy_outcome_rows": 50,
            "min_policy_outcome_timestamps": 10,
        },
        "selected_source": None,
        "nearest_source": {
            "policy_action_rows_estimate": 15,
            "policy_action_timestamps_estimate": 9,
            "policy_action_estimate_source": "finite_auction_rank_score",
            "policy_outcome_rows_estimate": 15,
            "policy_outcome_timestamps_estimate": 9,
            "policy_outcome_estimate_source": "finite_net_return",
        },
    }
    ledger_manifest = {
        "post_cutoff": {
            "post_cutoff_rows": 488,
            "post_cutoff_timestamps": 22,
            "post_cutoff_active_heads": 4,
        }
    }

    dashboard = _candidate_dashboard(promotion, monthly, heads, readiness, ledger_manifest)
    statuses = dict(zip(dashboard["variant"], dashboard["candidate_status"], strict=True))

    assert statuses["candidate_a"] == "research_candidate_waiting_frozen_evidence"
    assert statuses["candidate_b"] == "rejected_or_diagnostic"
    row = dashboard.loc[dashboard["variant"].eq("candidate_a")].iloc[0]
    assert bool(row["development_candidate_pass"]) is True
    assert bool(row["deployment_candidate_pass"]) is False
    assert row["post_cutoff_rows_needed"] == 1512
    assert row["post_cutoff_timestamps_needed"] == 18
    assert row["policy_action_rows_needed"] == 35
    assert row["policy_action_timestamps_needed"] == 1
    assert row["policy_action_estimate_source"] == "finite_auction_rank_score"
    assert row["policy_outcome_rows_needed"] == 35
    assert row["policy_outcome_timestamps_needed"] == 1
    assert row["policy_outcome_estimate_source"] == "finite_net_return"


def test_head_portfolio_overlay_applies_only_target_head() -> None:
    from scripts.ablate_contextual_tp_sl_head_portfolio_overlays import (
        _apply_overlay,
        _load_overlay_specs,
        _parse_overlay_spec,
    )

    specs = _load_overlay_specs(
        ["soft_short=short_asset:size=0.5,priority=0.25,rank=-0.02,strategy_cap=1"],
        None,
    )
    overlay = _parse_overlay_spec(specs["soft_short"])
    frame = pd.DataFrame(
        {
            "strategy_id": [
                "short_asset_alpha",
                "long_bars_alpha",
            ],
            "portfolio_size_multiplier": [1.0, 1.0],
            "portfolio_priority_multiplier": [1.0, 1.0],
            "portfolio_rank_adjustment": [0.0, 0.0],
        }
    )

    adjusted = _apply_overlay(frame, overlay)

    short = adjusted.loc[adjusted["strategy_id"].eq("short_asset_alpha")].iloc[0]
    long = adjusted.loc[adjusted["strategy_id"].eq("long_bars_alpha")].iloc[0]
    assert short["portfolio_size_multiplier"] == 0.5
    assert short["portfolio_priority_multiplier"] == 0.25
    assert short["portfolio_rank_adjustment"] == -0.02
    assert short["portfolio_max_concurrent_per_strategy"] == 1.0
    assert long["portfolio_size_multiplier"] == 1.0
    assert long["portfolio_priority_multiplier"] == 1.0
    assert long["portfolio_rank_adjustment"] == 0.0
    assert pd.isna(long["portfolio_max_concurrent_per_strategy"])


def test_conditional_head_filter_binds_bad_reliability_rows_only() -> None:
    from scripts.ablate_contextual_tp_sl_conditional_head_filters import _apply_rule

    frame = pd.DataFrame(
        {
            "strategy_id": ["short_asset_alpha"] * 5 + ["long_dist_alpha"] * 5,
            "generated_weighted_hr_surprise_24": [-0.9, -0.7, -0.1, 0.0, 0.1] * 2,
            "generated_strategy_score_ood_abs_z": [0.1, 0.2, 0.3, 0.4, 2.0] * 2,
            "generated_score_abs_diff_24": [0.01, 0.02, 0.03, 0.04, 0.05] * 2,
            "generated_score_uncertainty_p1mp": [0.1, 0.2, 0.3, 0.4, 0.5] * 2,
            "portfolio_rank_adjustment": 0.0,
        }
    )
    rule = {
        "heads": ["short_asset"],
        "condition": "recent_hr_bad",
        "action": "rank",
        "value": -0.02,
    }

    adjusted = _apply_rule(frame, rule)

    short_adjusted = adjusted.loc[adjusted["strategy_id"].eq("short_asset_alpha")]
    long_adjusted = adjusted.loc[adjusted["strategy_id"].eq("long_dist_alpha")]
    assert int(short_adjusted["conditional_filter_bound"].sum()) == 2
    assert int(long_adjusted["conditional_filter_bound"].sum()) == 0
    assert (short_adjusted.loc[short_adjusted["conditional_filter_bound"].eq(1), "portfolio_rank_adjustment"] == -0.02).all()
    assert (long_adjusted["portfolio_rank_adjustment"] == 0.0).all()


def test_conditional_head_filter_expanding_threshold_uses_prior_timestamps_only() -> None:
    from scripts.ablate_contextual_tp_sl_conditional_head_filters import _apply_rule

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 00:00",
                    "2026-01-01 00:00",
                    "2026-01-01 01:00",
                    "2026-01-01 01:00",
                    "2026-01-01 02:00",
                    "2026-01-01 02:00",
                ],
                utc=True,
            ),
            "strategy_id": ["short_asset_alpha"] * 6,
            "generated_weighted_hr_surprise_24": [0.4, 0.2, 0.3, 0.1, -0.8, 0.5],
            "portfolio_rank_adjustment": 0.0,
        }
    )
    rule = {
        "heads": ["short_asset"],
        "condition": "recent_hr_bad",
        "action": "rank",
        "value": -0.02,
    }

    adjusted = _apply_rule(frame, rule, threshold_mode="expanding", min_history=4)

    assert adjusted.loc[:3, "conditional_filter_bound"].sum() == 0
    assert adjusted.loc[4, "conditional_filter_bound"] == 1
    assert adjusted.loc[5, "conditional_filter_bound"] == 0
    assert adjusted.loc[4, "portfolio_rank_adjustment"] == -0.02


def test_conditional_head_filter_flat_loader_repairs_missing_head(tmp_path: Path) -> None:
    from scripts.ablate_contextual_tp_sl_conditional_head_filters import _load_flat_candidate_table

    path = tmp_path / "flat_candidates.parquet"
    pd.DataFrame(
        {
            "strategy_id": [
                "long_dist_ema20_atr_demo",
                "short_bollinger_demo",
            ],
            "head": [None, ""],
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-01"], utc=True),
            "symbol": ["A/USD:USD", "B/USD:USD"],
        }
    ).to_parquet(path, index=False)

    loaded = _load_flat_candidate_table(path)

    assert loaded["head"].tolist() == ["long_dist", "short_bollinger"]


def test_frozen_reliability_gate_uses_bundle_thresholds_and_required_heads() -> None:
    from scripts.run_frozen_reliability_challenger_gate_if_ready import _scan_args

    bundle = {
        "forward_validation_requirements": {
            "minimum_post_cutoff_rows": 123,
            "minimum_post_cutoff_timestamps": 17,
            "minimum_policy_action_rows": 19,
            "minimum_policy_outcome_rows": 23,
            "minimum_policy_outcome_rows_per_required_head": 5,
            "required_matured_outcome_heads": ["long_bars", "short_bollinger"],
        }
    }
    args = Namespace(
        min_post_cutoff_rows=0,
        min_post_cutoff_timestamps=0,
        min_post_cutoff_active_heads=0,
        min_policy_action_rows=0,
        min_policy_action_timestamps=0,
        min_policy_outcome_rows=0,
        min_policy_outcome_timestamps=0,
        min_policy_outcome_rows_per_action_head=0,
        min_policy_outcome_rows_per_required_head=0,
        min_diagnostic_group_features=1,
        min_diagnostic_group_finite_rate=0.25,
    )

    scan_args = _scan_args(bundle, args)

    assert scan_args.min_post_cutoff_rows == 123
    assert scan_args.min_post_cutoff_timestamps == 17
    assert scan_args.min_policy_action_rows == 19
    assert scan_args.min_policy_outcome_rows == 23
    assert scan_args.required_policy_outcome_head == ["long_bars", "short_bollinger"]
    assert scan_args.min_policy_outcome_rows_per_required_head == 5


def test_frozen_reliability_gate_readiness_deficits_are_explicit() -> None:
    from scripts.run_frozen_reliability_challenger_gate_if_ready import _readiness_deficit_rows

    bundle = {
        "forward_validation_requirements": {
            "minimum_post_cutoff_rows": 2000,
            "minimum_post_cutoff_timestamps": 40,
            "minimum_active_heads": 3,
            "minimum_policy_action_rows": 50,
            "minimum_policy_action_timestamps": 10,
            "minimum_policy_outcome_rows": 50,
            "minimum_policy_outcome_timestamps": 10,
            "minimum_policy_outcome_rows_per_required_head": 3,
            "required_matured_outcome_heads": ["long_bars", "short_bollinger"],
        }
    }
    args = Namespace(
        min_post_cutoff_rows=0,
        min_post_cutoff_timestamps=0,
        min_post_cutoff_active_heads=0,
        min_policy_action_rows=0,
        min_policy_action_timestamps=0,
        min_policy_outcome_rows=0,
        min_policy_outcome_timestamps=0,
        min_policy_outcome_rows_per_action_head=0,
        min_policy_outcome_rows_per_required_head=0,
        min_diagnostic_group_features=1,
        min_diagnostic_group_finite_rate=0.25,
    )
    scan = pd.DataFrame(
        {
            "path": ["candidate.parquet"],
            "exists": [True],
            "ready": [False],
            "timestamp_max": ["2026-07-01T12:00:00+00:00"],
            "post_cutoff_rows": [1153],
            "post_cutoff_timestamps": [59],
            "post_cutoff_active_heads": [4],
            "policy_action_rows_estimate": [32],
            "policy_action_timestamps_estimate": [20],
            "policy_outcome_rows_estimate": [18],
            "policy_outcome_timestamps_estimate": [12],
            "policy_outcome_head_counts": ['{"long_bars": 3}'],
        }
    )

    deficits = _readiness_deficit_rows(bundle, args, scan).set_index(["gate", "head"])

    assert int(deficits.loc[("post_cutoff_rows", ""), "deficit"]) == 847
    assert int(deficits.loc[("policy_action_rows", ""), "deficit"]) == 18
    assert int(deficits.loc[("policy_outcome_rows", ""), "deficit"]) == 32
    assert bool(deficits.loc[("policy_outcome_rows_per_required_head", "long_bars"), "pass"]) is True
    assert int(deficits.loc[("policy_outcome_rows_per_required_head", "short_bollinger"), "deficit"]) == 3


def test_frozen_reliability_gate_writes_only_frozen_rule_specs(tmp_path: Path) -> None:
    from scripts.run_frozen_reliability_challenger_gate_if_ready import _rule_file
    import json

    bundle = {
        "rules": {
            "lb_drift_rank_m0025": {
                "role": "conservative",
                "heads": ["long_bars"],
                "condition": "drift_high",
                "action": "rank",
                "value": -0.0025,
                "metrics": {"delta_net_pnl": 1.0},
            }
        }
    }

    path = _rule_file(bundle, tmp_path)
    rules = json.loads(path.read_text())

    assert set(rules) == {"none", "lb_drift_rank_m0025"}
    assert rules["none"] == {}
    assert rules["lb_drift_rank_m0025"] == {
        "heads": ["long_bars"],
        "condition": "drift_high",
        "action": "rank",
        "value": -0.0025,
    }


def test_frozen_reliability_gate_candidate_paths_default_root_and_dedupe(tmp_path: Path) -> None:
    from scripts.run_frozen_reliability_challenger_gate_if_ready import DEFAULT_CANDIDATES, _candidate_paths

    root = tmp_path / "reports"
    discovered = (
        root
        / "contextual_tp_sl_ablation_workflow_v99_demo"
        / "cumulative_ledger"
        / "cumulative_flat_candidates.parquet"
    )
    discovered.parent.mkdir(parents=True)
    discovered.write_text("placeholder")

    explicit = tmp_path / "explicit.parquet"
    explicit.write_text("placeholder")
    args = Namespace(candidate=[str(explicit), str(explicit)], root=[str(root)])

    paths = _candidate_paths(args)

    assert paths == [explicit, discovered]

    default_paths = _candidate_paths(Namespace(candidate=None, root=None))

    assert default_paths == list(DEFAULT_CANDIDATES)


def test_conditional_filter_bootstrap_preserves_week_blocks() -> None:
    from scripts.report_conditional_filter_bootstrap_confidence import _bootstrap_rule

    daily = pd.DataFrame(
        {
            "day": ["2026-01-05", "2026-01-06", "2026-01-12", "2026-01-13"] * 2,
            "rule_id": ["none"] * 4 + ["candidate"] * 4,
            "net_pnl": [10.0, 10.0, 20.0, 20.0, 11.0, 12.0, 19.0, 22.0],
        }
    )
    weekly = pd.DataFrame(
        {
            "week": ["2026-01-05/2026-01-11", "2026-01-12/2026-01-18"] * 2,
            "rule_id": ["none", "none", "candidate", "candidate"],
            "net_pnl": [20.0, 40.0, 23.0, 41.0],
        }
    )

    observed, samples = _bootstrap_rule(
        daily,
        weekly,
        "candidate",
        "none",
        n_bootstrap=20,
        seed=7,
    )

    assert observed["delta_net_pnl"] == 4.0
    assert observed["delta_avg_week_pnl"] == 2.0
    assert len(samples) == 20
    assert set(samples["delta_net_pnl"]).issubset({2.0, 4.0, 6.0})


def test_frozen_reliability_status_requires_fresh_gate_for_production(tmp_path: Path) -> None:
    from scripts.audit_frozen_reliability_challenger_status import (
        _best_scorecard_rows,
        _candidate_rows,
        _candidate_tradeoff_rows,
        _feature_family_readout,
        _freeze_decision_matrix,
        _fresh_evidence_gap_rows,
        _gate_summary_rows,
        _head_family_scope_recommendation_rows,
        _long_period_family_robustness_rows,
        _long_period_robustness_rows,
        _monthly_delta_rows,
        _nearest_source_family_coverage_rows,
        _promotion_blocker_rows,
        _reliability_ab_selection_frontier,
        _reliability_ab_scorecard_rows,
        _requested_family_verdict,
        _requested_family_decision_rows,
        _selection_policy_rows,
        _scorecard_marginal_family_ablation,
        _status_from_gate,
        _tail_aversion_sensitivity,
        _tail_aversion_switch_points,
        _temporal_stability_rows,
        _worst_week_rows,
    )

    decision = pd.DataFrame(
        {
            "rule_id": ["candidate"],
            "delta_net_pnl": [100.0],
            "delta_objective": [10.0],
            "active_weeks": [3],
            "active_positive_week_share": [1.0],
            "worst_week_delta": [0.0],
        }
    )
    bootstrap = pd.DataFrame(
        {
            "rule_id": ["candidate"],
            "prob_delta_net_pnl_positive": [0.99],
            "prob_delta_objective_positive": [0.99],
            "delta_net_pnl_p05": [5.0],
            "delta_objective_p05": [1.0],
        }
    )
    bundle = {
        "rules": {"candidate": {"role": "test"}},
        "forward_validation_requirements": {
            "minimum_post_cutoff_rows": 2000,
            "minimum_policy_action_rows": 50,
            "minimum_policy_outcome_rows": 50,
            "minimum_policy_outcome_rows_per_required_head": 3,
            "required_matured_outcome_heads": ["long_bars", "short_bollinger"],
        },
    }

    candidates = _candidate_rows(decision, bootstrap, bundle, min_bootstrap_prob=0.95)
    fresh_ready, blockers = _status_from_gate(
        {
            "ran_gate": False,
            "ready_sources": 0,
            "nearest_source": {
                "rejection_reasons": "policy_outcome_rows_lt_50",
                "post_cutoff_rows": 1153,
                "policy_action_rows_estimate": 32,
                "policy_outcome_rows_estimate": 18,
                "policy_outcome_low_required_head_counts": '{"short_bollinger": 0}',
                "drift_finite_row_rate": 0.98,
                "recent_hit_rate_surprise_finite_row_rate": 1.0,
                "ood_finite_row_rate": 0.42,
                "uncertainty_finite_row_rate": 0.41,
            },
        }
    )
    gate_summary = _gate_summary_rows(
        {
            "ran_gate": False,
            "ready_sources": 0,
            "nearest_source": {
                "post_cutoff_rows": 1153,
                "policy_action_rows_estimate": 32,
                "policy_outcome_rows_estimate": 18,
                "policy_outcome_low_required_head_counts": '{"short_bollinger": 0}',
                "drift_finite_row_rate": 0.98,
                "recent_hit_rate_surprise_finite_row_rate": 1.0,
                "ood_finite_row_rate": 0.42,
                "uncertainty_finite_row_rate": 0.41,
            },
        }
    ).iloc[0]
    gap_frame = _fresh_evidence_gap_rows(
        bundle,
        {
            "ran_gate": False,
            "nearest_source": {
                "post_cutoff_rows": 1153,
                "post_cutoff_timestamps": 59,
                "post_cutoff_active_heads": 4,
                "policy_action_rows_estimate": 32,
                "policy_action_timestamps_estimate": 20,
                "policy_outcome_rows_estimate": 18,
                "policy_outcome_timestamps_estimate": 12,
                "policy_outcome_head_counts": '{"long_bars": 3, "short_bollinger": 0}',
            },
        },
    ).set_index(["gate", "head"])
    blocker_frame = _promotion_blocker_rows(gap_frame.reset_index()).set_index(["blocker", "head"])

    assert bool(candidates.loc[0, "research_pass"]) is True
    assert bool(candidates.loc[0, "tail_clean"]) is True
    assert fresh_ready is False
    assert blockers == ["policy_outcome_rows_lt_50"]
    assert int(gate_summary["post_cutoff_rows"]) == 1153
    assert int(gate_summary["policy_action_rows"]) == 32
    assert int(gate_summary["policy_outcome_rows"]) == 18
    assert gate_summary["required_head_outcome_gaps"] == '{"short_bollinger": 0}'
    assert float(gate_summary["drift_finite_row_rate"]) == 0.98
    assert float(gate_summary["recent_hr_finite_row_rate"]) == 1.0

    coverage_ledger = tmp_path / "coverage_ledger.parquet"
    pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-06-26 00:00:00+00:00",
                    "2026-06-26 01:00:00+00:00",
                    "2026-06-26 02:00:00+00:00",
                    "2026-06-25 23:00:00+00:00",
                ]
            ),
            "strategy_id": [
                "long_bars_alpha",
                "long_bars_beta",
                "short_bollinger_gamma",
                "long_bars_old",
            ],
            "generated_score_uncertainty_p1mp": [0.1, None, None, 0.9],
            "generated_score_entropy": [None, None, None, 0.8],
            "generated_score_abs_distance_from_half": [None, None, None, 0.7],
            "generated_score_abs_diff_1": [0.2, 0.3, None, 0.6],
            "generated_strategy_score_ood_abs_z": [None, None, 2.0, 0.1],
            "generated_hr_surprise_24": [-0.2, -0.3, -0.4, 0.0],
        }
    ).to_parquet(coverage_ledger, index=False)
    coverage = _nearest_source_family_coverage_rows(
        {
            "cutoff": "2026-06-26T00:00:00Z",
            "nearest_source": {"path": str(coverage_ledger)},
        }
    ).set_index(["family", "head"])
    assert int(coverage.loc[("drift", "long_bars"), "post_cutoff_rows"]) == 2
    assert float(coverage.loc[("drift", "long_bars"), "finite_row_rate"]) == 1.0
    assert float(coverage.loc[("drift", "short_bollinger"), "finite_row_rate"]) == 0.0
    assert float(coverage.loc[("ood", "short_bollinger"), "finite_row_rate"]) == 1.0
    assert float(coverage.loc[("uncertainty", "long_bars"), "finite_row_rate"]) == 0.5

    assert int(gap_frame.loc[("post_cutoff_rows", ""), "deficit"]) == 847
    assert int(gap_frame.loc[("policy_action_rows", ""), "deficit"]) == 18
    assert int(gap_frame.loc[("policy_outcome_rows", ""), "deficit"]) == 32
    assert bool(gap_frame.loc[("policy_outcome_rows_per_required_head", "long_bars"), "pass"]) is True
    assert int(gap_frame.loc[("policy_outcome_rows_per_required_head", "short_bollinger"), "deficit"]) == 3
    assert int(blocker_frame.loc[("post_cutoff_rows", ""), "deficit"]) == 847
    assert int(blocker_frame.loc[("policy_outcome_rows_per_required_head", "short_bollinger"), "deficit"]) == 3
    assert "matured outcomes" in blocker_frame.loc[
        ("policy_outcome_rows_per_required_head", "short_bollinger"), "next_action"
    ]

    scorecard_dir = tmp_path / "scorecards"
    scorecard_dir.mkdir()
    pd.DataFrame(
        {
            "variant": ["baseline", "recent_drift", "recent_drift_ood", "uncertainty_recent_drift_half"],
            "family": [
                "baseline_or_other",
                "recent_hr+drift",
                "recent_hr+drift+OOD",
                "recent_hr+drift+uncertainty",
            ],
            "delta_vs_baseline_net_pnl": [0.0, 10.0, 8.0, 6.0],
            "delta_vs_baseline_objective_avgweek_0p7dayq35_0p3dayq20": [0.0, 1.0, 0.8, 0.4],
            "hit_rate": [0.4, 0.42, 0.41, 0.405],
            "full_sl_rate": [0.3, 0.29, 0.28, 0.295],
            "weekly_q05_pnl": [-5.0, -1.0, 0.0, -2.0],
            "scorecard_score": [0.0, 10.0, 8.0, 6.0],
        }
    ).to_csv(scorecard_dir / "promotion_scorecard.csv", index=False)
    pd.DataFrame(
        {
            "strategy_id": ["s1", "s2"],
            "arm": ["joint_all", "performance_only"],
            "is_best_validation_arm": [True, False],
            "delta_objective_vs_static": [0.2, 0.1],
            "delta_net_pnl_vs_static": [2.0, 1.0],
            "uncertainty_feature_count": [3, 0],
            "drift_feature_count": [6, 0],
            "ood_feature_count": [2, 0],
            "recent_perf_feature_count": [8, 8],
        }
    ).to_csv(scorecard_dir / "feature_family_readout.csv", index=False)

    best_scorecard = _best_scorecard_rows(scorecard_dir)
    ab_scorecard = _reliability_ab_scorecard_rows(scorecard_dir).set_index("variant")
    ab_frontier = _reliability_ab_selection_frontier(ab_scorecard.reset_index()).set_index(
        ["source", "evidence_family", "policy_id"]
    )
    feature_readout = _feature_family_readout(scorecard_dir)
    marginal_ablation = _scorecard_marginal_family_ablation(scorecard_dir)
    requested_verdict = _requested_family_verdict(marginal_ablation, gate_summary.to_frame().T).set_index("family")
    requested_decisions = _requested_family_decision_rows(
        requested_verdict.reset_index(),
        marginal_ablation,
    ).set_index("family")
    head_recommendations = _head_family_scope_recommendation_rows(
        pd.DataFrame(
            [
                {
                    "family": "drift",
                    "head": "long_bars",
                    "post_cutoff_rows": 20,
                    "finite_row_rate": 0.9,
                    "finite_cell_rate": 0.5,
                    "columns_present": 6,
                    "columns_required": 6,
                },
                {
                    "family": "ood",
                    "head": "long_dist",
                    "post_cutoff_rows": 20,
                    "finite_row_rate": 0.1,
                    "finite_cell_rate": 0.05,
                    "columns_present": 3,
                    "columns_required": 3,
                },
                {
                    "family": "ood",
                    "head": "short_bollinger",
                    "post_cutoff_rows": 20,
                    "finite_row_rate": 0.9,
                    "finite_cell_rate": 0.8,
                    "columns_present": 3,
                    "columns_required": 3,
                },
                {
                    "family": "uncertainty",
                    "head": "short_bollinger",
                    "post_cutoff_rows": 20,
                    "finite_row_rate": 0.9,
                    "finite_cell_rate": 0.8,
                    "columns_present": 3,
                    "columns_required": 3,
                },
            ]
        ),
        requested_decisions.reset_index(),
    ).set_index(["family", "head"])

    assert best_scorecard.loc[0, "source"] == "promotion"
    assert best_scorecard.loc[0, "variant"] == "recent_drift"
    assert best_scorecard.loc[0, "family"] == "recent_hr+drift"
    assert float(best_scorecard.loc[0, "delta_net_pnl"]) == 10.0
    assert float(ab_scorecard.loc["recent_drift", "delta_net_pnl"]) == 10.0
    assert bool(ab_scorecard.loc["recent_drift", "contains_drift"]) is True
    assert bool(ab_scorecard.loc["recent_drift", "contains_recent_hit_rate_surprise"]) is True
    assert bool(ab_scorecard.loc["recent_drift_ood", "contains_ood"]) is True
    assert bool(ab_scorecard.loc["uncertainty_recent_drift_half", "contains_uncertainty"]) is True
    assert bool(ab_scorecard.loc["uncertainty_recent_drift_half", "contains_drift"]) is True
    assert (
        bool(ab_scorecard.loc["uncertainty_recent_drift_half", "contains_recent_hit_rate_surprise"])
        is True
    )
    assert ab_scorecard.loc["recent_drift", "ab_verdict"] == "pnl_positive_tail_mixed"
    assert (
        ab_frontier.loc[("promotion", "promotion", "max_pnl"), "selected_variant"]
        == "recent_drift"
    )
    assert (
        ab_frontier.loc[("promotion", "promotion", "balanced_pnl_tail"), "selected_variant"]
        == "recent_drift"
    )
    assert bool(ab_frontier.loc[("promotion", "promotion", "max_pnl"), "contains_drift"]) is True
    assert bool(ab_frontier.loc[("promotion", "promotion", "max_pnl"), "contains_recent_hit_rate_surprise"]) is True
    assert feature_readout.iloc[0]["arm"] == "joint_all"
    assert int(feature_readout.iloc[0]["best_validation_strategies"]) == 1
    assert requested_verdict.loc["drift", "verdict"] == "helpful_in_tests"
    assert requested_verdict.loc["recent_hit_rate_surprise", "verdict"] == "helpful_in_tests"
    assert requested_verdict.loc["ood", "verdict"] == "tested_no_clear_lift"
    assert requested_verdict.loc["uncertainty", "verdict"] == "tested_no_clear_lift"
    assert requested_decisions.loc["drift", "decision"] == "default_keep_candidate"
    assert requested_decisions.loc["recent_hit_rate_surprise", "decision"] == "default_keep_candidate"
    assert requested_decisions.loc["ood", "decision"] == "diagnostic_only_do_not_default"
    assert requested_decisions.loc["uncertainty", "decision"] == "diagnostic_only_do_not_default"
    assert head_recommendations.loc[("drift", "long_bars"), "recommendation"] == "default_keep_for_head"
    assert head_recommendations.loc[("ood", "long_dist"), "recommendation"] == "coverage_gap_do_not_use"
    assert (
        head_recommendations.loc[("uncertainty", "short_bollinger"), "recommendation"]
        == "diagnostic_only_do_not_default"
    )

    tradeoff = _candidate_tradeoff_rows(
        pd.DataFrame(
            {
                "rule_id": ["pnl", "tail", "dominated"],
                "role": ["pnl", "tail", "dominated"],
                "delta_net_pnl": [200.0, 150.0, 100.0],
                "delta_objective": [20.0, 15.0, 10.0],
                "active_positive_week_share": [0.8, 1.0, 0.7],
                "worst_week_delta": [-20.0, 0.0, -30.0],
                "prob_delta_net_pnl_positive": [0.99, 0.99, 0.95],
                "prob_delta_objective_positive": [0.99, 0.99, 0.95],
            }
        )
    ).set_index("rule_id")

    assert bool(tradeoff.loc["pnl", "pareto_efficient"]) is True
    assert bool(tradeoff.loc["tail", "pareto_efficient"]) is True
    assert bool(tradeoff.loc["dominated", "pareto_efficient"]) is False
    assert float(tradeoff.loc["pnl", "net_pnl_share_of_best"]) == 1.0
    assert float(tradeoff.loc["tail", "worst_week_safety"]) == 1.0

    sensitivity = _tail_aversion_sensitivity(tradeoff.reset_index())
    switch_points = _tail_aversion_switch_points(tradeoff.reset_index())
    policies = _selection_policy_rows(tradeoff.reset_index()).set_index("policy_id")
    selected_by_weight = sensitivity.set_index("tail_weight")["selected_rule_id"].to_dict()

    assert selected_by_weight[0.0] == "pnl"
    assert selected_by_weight[1.0] == "tail"
    assert set(sensitivity["selected_rule_id"]).issubset({"pnl", "tail"})
    assert len(switch_points) == 1
    assert switch_points.iloc[0]["rule_a"] in {"pnl", "tail"}
    assert 0.0 < float(switch_points.iloc[0]["tail_weight_switch"]) < 1.0
    assert policies.loc["pnl_dominant", "selected_rule_id"] == "pnl"
    assert policies.loc["balanced_default", "selected_rule_id"] == "pnl"
    assert policies.loc["tail_aware", "selected_rule_id"] == "tail"
    assert policies.loc["hard_tail_clean", "selected_rule_id"] == "tail"
    freeze_matrix = _freeze_decision_matrix(
        policies.reset_index(),
        tradeoff.reset_index().assign(research_pass=True, tail_clean=lambda x: x["rule_id"].eq("tail")),
        pd.DataFrame(
            {
                "rule_id": ["pnl", "tail"],
                "fresh_status": ["fresh_negative_or_tail_warning", "no_fresh_binding"],
                "delta_net_pnl": [-5.0, 0.0],
                "delta_objective": [-1.0, 0.0],
                "delta_hit_rate": [-0.1, 0.0],
                "delta_full_sl_rate": [0.1, 0.0],
            }
        ),
        False,
        ["post_cutoff_rows_lt_2000"],
    ).set_index("policy_id")
    assert freeze_matrix.loc["pnl_dominant", "recommendation"] == "do_not_promote_wait_for_clean_fresh"
    assert freeze_matrix.loc["balanced_default", "recommendation"] == "do_not_promote_wait_for_clean_fresh"
    assert freeze_matrix.loc["tail_aware", "recommendation"] == "keep_frozen_wait_for_binding"
    assert freeze_matrix.loc["hard_tail_clean", "recommendation"] == "keep_frozen_wait_for_binding"

    decision_pack_dir = tmp_path / "decision_pack"
    decision_pack_dir.mkdir()
    pd.DataFrame(
        {
            "rule_id": ["pnl", "pnl", "pnl", "tail", "tail", "tail"],
            "week": ["w1", "w2", "w3", "w1", "w2", "w3"],
            "baseline_trades": [10, 10, 10, 10, 10, 10],
            "trades": [11, 11, 10, 11, 10, 10],
            "delta_net_pnl": [10.0, -2.0, 0.0, 5.0, 0.0, -1.0],
            "delta_trades": [1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        }
    ).to_csv(decision_pack_dir / "decision_pack_week_deltas.csv", index=False)
    pd.DataFrame(
        {
            "rule_id": ["pnl", "pnl", "tail", "tail"],
            "month": ["m1", "m2", "m1", "m2"],
            "delta_net_pnl": [8.0, -1.0, 4.0, 0.0],
            "delta_trades": [1.0, 1.0, 1.0, 0.0],
        }
    ).to_csv(decision_pack_dir / "decision_pack_month_deltas.csv", index=False)
    pd.DataFrame(
        {
            "rule_id": ["pnl", "pnl", "tail", "tail"],
            "head": ["h1", "h2", "h1", "h2"],
            "delta_net_pnl": [7.0, -1.0, 3.0, 2.0],
        }
    ).to_csv(decision_pack_dir / "decision_pack_head_deltas.csv", index=False)

    temporal = _temporal_stability_rows(decision_pack_dir).set_index("rule_id")
    monthly = _monthly_delta_rows(decision_pack_dir)
    worst_weeks = _worst_week_rows(decision_pack_dir, n=2)
    long_period = _long_period_robustness_rows(
        tradeoff.reset_index(),
        temporal.reset_index(),
        {
            "rules": {
                "pnl": {"condition": "drift_high", "role": "pnl"},
                "tail": {"condition": "recent_hr_bad", "role": "tail"},
            }
        },
    ).set_index("rule_id")
    family_period = _long_period_family_robustness_rows(long_period.reset_index()).set_index("family")

    assert int(temporal.loc["pnl", "active_weeks"]) == 2
    assert float(temporal.loc["pnl", "positive_active_week_share"]) == 0.5
    assert int(temporal.loc["tail", "active_weeks"]) == 2
    assert float(temporal.loc["tail", "positive_active_month_share"]) == 1.0
    assert int(temporal.loc["pnl", "negative_heads"]) == 1
    assert int(temporal.loc["tail", "positive_heads"]) == 2
    assert monthly["month"].tolist() == ["m1", "m2", "m1", "m2"]
    assert worst_weeks.groupby("rule_id").size().to_dict() == {"pnl": 2, "tail": 2}
    assert worst_weeks.loc[worst_weeks["rule_id"].eq("pnl"), "week"].tolist()[0] == "w2"
    assert worst_weeks.loc[worst_weeks["rule_id"].eq("tail"), "week"].tolist()[0] == "w3"
    assert long_period.loc["pnl", "families"] == "drift"
    assert long_period.loc["pnl", "long_period_verdict"] == "pnl_strong_tail_mixed"
    assert long_period.loc["tail", "families"] == "recent_hit_rate_surprise"
    assert long_period.loc["tail", "long_period_verdict"] == "tail_clean_broadly_consistent"
    assert family_period.loc["recent_hit_rate_surprise", "best_rule_id"] == "tail"
    assert family_period.loc["drift", "best_rule_id"] == "pnl"


def test_frozen_reliability_status_classifies_postcutoff_preview_rows(tmp_path: Path) -> None:
    from scripts.audit_frozen_reliability_challenger_status import (
        _family_action_impact_rows,
        _preview_decision_pack_rows,
        _postcutoff_preview_rows,
    )

    preview_dir = tmp_path / "fresh_preview"
    preview_out = preview_dir / "postcutoff_preview"
    preview_out.mkdir(parents=True)
    (preview_dir / "fresh_binding_rules.json").write_text(
        """
{
  "lb_drift_rank_m0025": {"condition": "drift_high", "action": "rank", "value": -0.0025, "heads": ["long_bars"]},
  "fresh_positive": {"condition": "recent_hr_bad", "action": "rank", "value": -0.02, "heads": ["long_bars"]},
  "fresh_any_bad_all_rank_m02": {"condition": "any_bad_reliability", "action": "rank", "value": -0.02, "heads": ["long_bars", "short_bollinger"]}
}
""".strip()
        + "\n"
    )
    (preview_out / "postcutoff_preview_manifest.json").write_text('{"cutoff": "2026-06-26T00:00:00Z"}\n')
    decision_pack = preview_dir / "decision_pack"
    decision_pack.mkdir()
    pd.DataFrame(
        {
            "rule_id": ["fresh_positive", "fresh_any_bad_all_rank_m02"],
            "days": [10, 10],
            "weeks": [2, 2],
            "active_days": [3, 4],
            "active_weeks": [2, 2],
            "delta_net_pnl": [100.0, 80.0],
            "delta_objective": [10.0, -5.0],
            "delta_weighted_daily_tail": [0.0, -20.0],
            "active_positive_week_share": [1.0, 0.5],
            "worst_day_delta": [0.0, -40.0],
            "worst_week_delta": [0.0, -50.0],
            "entrant_trades": [4, 5],
            "removed_trades": [3, 6],
            "entrant_minus_removed_net_pnl": [100.0, 80.0],
            "entrant_minus_removed_hit_rate": [0.3, -0.1],
            "entrant_minus_removed_full_sl_rate": [-0.2, 0.1],
        }
    ).to_csv(decision_pack / "decision_pack_summary.csv", index=False)
    pd.DataFrame(
        {
            "rule_id": ["none", "lb_drift_rank_m0025", "fresh_positive", "fresh_any_bad_all_rank_m02"],
            "trades": [27, 27, 29, 26],
            "delta_trades": [0, 0, 2, -1],
            "net_pnl": [404.7, 404.7, 455.0, 270.9],
            "delta_net_pnl": [0.0, 0.0, 50.3, -133.8],
            "delta_objective": [0.0, 0.0, 4.2, -7.5],
            "hit_rate": [0.4444, 0.4444, 0.48, 0.423],
            "delta_hit_rate": [0.0, 0.0, 0.0356, -0.0214],
            "full_sl_rate": [0.3704, 0.3704, 0.35, 0.3846],
            "delta_full_sl_rate": [0.0, 0.0, -0.0204, 0.0142],
            "entrant_trades": [0, 0, 2, 0],
            "removed_trades": [0, 0, 0, 1],
            "entrant_minus_removed_net_pnl": [0.0, 0.0, 50.3, -133.8],
            "entrant_minus_removed_hit_rate": [0.0, 0.0, 1.0, -1.0],
            "worst_day_delta": [0.0, 0.0, 0.0, -133.8],
        }
    ).to_csv(preview_out / "postcutoff_preview_summary.csv", index=False)

    rows = _postcutoff_preview_rows([preview_dir]).set_index("rule_id")
    preview_decisions = _preview_decision_pack_rows([preview_dir]).set_index("rule_id")

    assert rows.loc["none", "fresh_status"] == "baseline"
    assert rows.loc["lb_drift_rank_m0025", "fresh_status"] == "no_fresh_binding"
    assert rows.loc["lb_drift_rank_m0025", "families"] == "drift"
    assert rows.loc["fresh_positive", "families"] == "recent_hit_rate_surprise"
    assert rows.loc["fresh_positive", "fresh_status"] == "fresh_positive"
    assert rows.loc["fresh_any_bad_all_rank_m02", "fresh_status"] == "fresh_negative_or_tail_warning"
    assert rows.loc["fresh_any_bad_all_rank_m02", "families"] == "drift,recent_hit_rate_surprise,ood,uncertainty"
    assert rows.loc["lb_drift_rank_m0025", "cutoff"] == "2026-06-26T00:00:00Z"
    assert preview_decisions.loc["fresh_positive", "long_window_status"] == "long_window_positive_tail_clean"
    assert (
        preview_decisions.loc["fresh_any_bad_all_rank_m02", "long_window_status"]
        == "long_window_positive_tail_mixed"
    )

    action = _family_action_impact_rows(
        pd.DataFrame(
            {
                "rule_id": ["lb_drift_rank_m0025"],
                "delta_net_pnl": [100.0],
                "delta_objective": [10.0],
                "active_weeks": [2],
                "active_positive_week_share": [1.0],
                "worst_week_delta": [0.0],
                "entrant_trades": [3],
                "removed_trades": [3],
                "entrant_minus_removed_net_pnl": [100.0],
                "entrant_minus_removed_hit_rate": [0.25],
            }
        ),
        rows.reset_index(),
        preview_decisions.reset_index(),
        {"rules": {"lb_drift_rank_m0025": {"condition": "drift_high", "heads": ["long_bars"], "action": "rank"}}},
    )

    drift_rows = action.loc[action["family"].eq("drift")].set_index(["evidence_scope", "rule_id"])
    assert bool(drift_rows.loc[("long_window", "lb_drift_rank_m0025"), "action_binding"]) is True
    assert drift_rows.loc[("long_window", "lb_drift_rank_m0025"), "status"] == "long_window_positive_tail_clean"
    any_bad_families = set(
        action.loc[
            action["rule_id"].eq("fresh_any_bad_all_rank_m02")
            & action["evidence_scope"].eq("post_cutoff_preview"),
            "family",
        ].astype(str)
    )
    assert any_bad_families == {"drift", "recent_hit_rate_surprise", "ood", "uncertainty"}
    long_preview = action.loc[
        action["rule_id"].eq("fresh_positive")
        & action["evidence_scope"].eq("long_window_preview_replay")
    ]
    assert set(long_preview["family"].astype(str)) == {"recent_hit_rate_surprise"}
    assert long_preview.iloc[0]["status"] == "long_window_positive_tail_clean"


def test_head_family_recommendation_requires_variant_scope_compatibility() -> None:
    from scripts.audit_frozen_reliability_challenger_status import _head_family_scope_recommendation_rows

    coverage = pd.DataFrame(
        [
            {
                "family": "ood",
                "head": "short_bollinger",
                "post_cutoff_rows": 100,
                "finite_row_rate": 0.95,
                "finite_cell_rate": 0.9,
                "columns_present": 3,
                "columns_required": 3,
            },
            {
                "family": "ood",
                "head": "long_bars",
                "post_cutoff_rows": 100,
                "finite_row_rate": 0.95,
                "finite_cell_rate": 0.9,
                "columns_present": 3,
                "columns_required": 3,
            },
        ]
    )
    family_decisions = pd.DataFrame(
        [
            {
                "family": "ood",
                "decision": "conditional_head_scoped_use",
                "verdict": "helpful_in_tests",
                "best_variant": "recent_drift_ood_long_bars_half",
                "best_marginal_delta_net_pnl": 100.0,
            }
        ]
    )

    rows = _head_family_scope_recommendation_rows(coverage, family_decisions).set_index(["family", "head"])

    assert rows.loc[("ood", "long_bars"), "recommendation"] == "head_scoped_candidate"
    assert bool(rows.loc[("ood", "long_bars"), "best_variant_scope_compatible"]) is True
    assert (
        rows.loc[("ood", "short_bollinger"), "recommendation"]
        == "head_scoped_candidate_needs_matched_variant"
    )
    assert bool(rows.loc[("ood", "short_bollinger"), "best_variant_scope_compatible"]) is False


def test_multiwindow_selector_has_tolerant_balanced_profile() -> None:
    from scripts.report_conditional_filter_multiwindow_selection import _profile_recommendations

    summary = pd.DataFrame(
        [
            {
                "rule_id": "mild_recent_hr_drift",
                "core_pnl_tail_gate_count": 2,
                "core_strict_tail_gate_count": 2,
                "core_min_delta_objective": -10.0,
                "core_min_delta_net_pnl": 0.0,
                "full_delta_objective": 55.0,
                "full_delta_net_pnl": 1500.0,
                "full_delta_weekly_q20": 0.0,
                "full_delta_weighted_daily_tail": 0.0,
            },
            {
                "rule_id": "strictly_bad",
                "core_pnl_tail_gate_count": 1,
                "core_strict_tail_gate_count": 1,
                "core_min_delta_objective": -50.0,
                "core_min_delta_net_pnl": -5.0,
                "full_delta_objective": 100.0,
                "full_delta_net_pnl": 2000.0,
                "full_delta_weekly_q20": 0.0,
                "full_delta_weighted_daily_tail": 0.0,
            },
        ]
    )

    profiles = _profile_recommendations(
        summary,
        tolerant_min_core_gates=2,
        tolerant_min_core_net_pnl=0.0,
        tolerant_min_core_objective=-25.0,
    )

    assert profiles["balanced_pnl_tail"] is None
    assert profiles["strict_tail"] is None
    assert profiles["balanced_tolerant"]["rule_id"] == "mild_recent_hr_drift"


def test_frozen_reliability_audit_reads_multiwindow_selection(tmp_path: Path) -> None:
    from scripts.audit_frozen_reliability_challenger_status import _multiwindow_selection_rows

    selection_dir = tmp_path / "multiwindow"
    selection_dir.mkdir()
    (selection_dir / "multiwindow_selection.json").write_text(
        """
        {
          "recommended": {
            "rule_id": "lb_recent_hr_drift_rank_m005",
            "full_delta_net_pnl": 1471.0
          },
          "best_by_sort_order": {
            "rule_id": "lb_recent_hr_drift_rank_m005",
            "full_delta_net_pnl": 1471.0
          },
          "profile_recommendations": {
            "balanced_tolerant": {
              "rule_id": "lb_recent_hr_drift_rank_m005",
              "full_delta_net_pnl": 1471.0
            },
            "strict_tail": null
          }
        }
        """
    )
    pd.DataFrame(
        [
            {
                "rule_id": "lb_recent_hr_drift_rank_m005",
                "core_pnl_tail_gate_count": 2,
                "core_strict_tail_gate_count": 2,
                "core_min_delta_objective": -17.5,
                "core_min_delta_net_pnl": 0.0,
                "full_delta_objective": 54.5,
                "full_delta_net_pnl": 1471.0,
                "full_delta_weekly_q20": 0.0,
                "full_delta_weighted_daily_tail": 0.0,
                "entrant_minus_removed_hit_rate": 0.29,
            }
        ]
    ).to_csv(selection_dir / "multiwindow_candidate_summary.csv", index=False)

    rows = _multiwindow_selection_rows([selection_dir]).set_index("selection_kind")

    assert rows.loc["recommended", "rule_id"] == "lb_recent_hr_drift_rank_m005"
    assert bool(rows.loc["recommended", "profile_pass"]) is True
    assert bool(rows.loc["best_by_sort_order", "profile_pass"]) is False
    assert float(rows.loc["recommended", "core_min_delta_objective"]) == -17.5


def test_frozen_reliability_candidate_registry_separates_stages(tmp_path: Path) -> None:
    from scripts.audit_frozen_reliability_challenger_status import _candidate_registry_rows

    selection_dir = tmp_path / "multiwindow"
    selection_dir.mkdir()
    attribution_dir = tmp_path / "attribution"
    attribution_dir.mkdir()
    (selection_dir / "multiwindow_selection.json").write_text(
        f"""
        {{
          "attribution_dir": "{attribution_dir}",
          "recommended": {{"rule_id": "lb_recent_hr_drift_rank_m005"}},
          "best_by_sort_order": {{"rule_id": "recent_hr_drift_rank_m005"}},
          "profile_recommendations": {{"balanced_tolerant": {{"rule_id": "lb_recent_hr_drift_rank_m005"}}}}
        }}
        """
    )
    pd.DataFrame(
        [
            {
                "rule_id": "lb_recent_hr_drift_rank_m005",
                "rule_spec": '{"action":"rank","condition":"recent_hr_bad_or_drift_high","heads":["long_bars"],"value":-0.005}',
            },
            {
                "rule_id": "recent_hr_drift_rank_m005",
                "rule_spec": '{"action":"rank","condition":"recent_hr_bad_or_drift_high","heads":["long_bars","short_asset"],"value":-0.005}',
            },
            {
                "rule_id": "ood_uncertainty_rank_m005",
                "rule_spec": '{"action":"rank","condition":"ood_high_and_uncertainty_high","heads":["long_bars","short_asset"],"value":-0.005}',
            },
        ]
    ).to_csv(attribution_dir / "conditional_filter_summary.csv", index=False)

    candidates = pd.DataFrame(
        [
            {
                "rule_id": "lb_drift_rank_m0025",
                "delta_net_pnl": 10.0,
                "delta_objective": 2.0,
                "active_positive_week_share": 1.0,
                "worst_week_delta": 0.0,
                "research_pass": True,
                "tail_clean": True,
            },
            {
                "rule_id": "lb_drift_rank_m005",
                "delta_net_pnl": 20.0,
                "delta_objective": 3.0,
                "active_positive_week_share": 0.9,
                "worst_week_delta": -1.0,
                "research_pass": True,
                "tail_clean": False,
            },
        ]
    )
    multiwindow = pd.DataFrame(
        [
            {
                "selection_dir": str(selection_dir),
                "selection_kind": "recommended",
                "rule_id": "lb_recent_hr_drift_rank_m005",
                "profile_pass": True,
                "full_delta_net_pnl": 5.0,
            },
            {
                "selection_dir": str(selection_dir),
                "selection_kind": "best_by_sort_order",
                "rule_id": "recent_hr_drift_rank_m005",
                "profile_pass": False,
                "full_delta_net_pnl": 6.0,
            },
            {
                "selection_dir": str(selection_dir),
                "selection_kind": "profile:diagnostic_ood_uncertainty",
                "rule_id": "ood_uncertainty_rank_m005",
                "profile_pass": False,
                "full_delta_net_pnl": -2.0,
            },
        ]
    )
    bundle = {
        "rules": {
            "lb_drift_rank_m0025": {
                "role": "tail",
                "heads": ["long_bars"],
                "condition": "drift_high",
                "action": "rank",
                "value": -0.0025,
            },
            "lb_drift_rank_m005": {
                "role": "pnl",
                "heads": ["long_bars"],
                "condition": "drift_high",
                "action": "rank",
                "value": -0.005,
            },
        }
    }

    rows = _candidate_registry_rows(
        candidates,
        multiwindow,
        bundle,
        [selection_dir],
        fresh_ready=False,
        fresh_blockers=["post_cutoff_rows_lt_2000"],
    ).set_index("rule_id")

    assert rows.loc["lb_drift_rank_m0025", "candidate_state"] == "frozen_tail_candidate_wait_fresh"
    assert rows.loc["lb_drift_rank_m005", "candidate_state"] == "frozen_pnl_candidate_wait_fresh"
    assert (
        rows.loc["lb_recent_hr_drift_rank_m005", "candidate_state"]
        == "multiwindow_research_candidate_needs_freeze_pack"
    )
    assert rows.loc["lb_recent_hr_drift_rank_m005", "families"] == "drift,recent_hit_rate_surprise"
    assert rows.loc["recent_hr_drift_rank_m005", "candidate_state"] == "diagnostic_only_profile_failed"
    assert rows.loc["ood_uncertainty_rank_m005", "families"] == "ood,uncertainty"
    assert rows.loc["ood_uncertainty_rank_m005", "candidate_state"] == "diagnostic_only_profile_failed"


def test_materialize_frozen_reliability_research_bundle_preserves_rule_and_metrics(tmp_path: Path) -> None:
    from scripts.materialize_frozen_reliability_research_bundle import _materialize_bundle

    attribution_dir = tmp_path / "attribution"
    attribution_dir.mkdir()
    selection_dir = tmp_path / "selection"
    selection_dir.mkdir()
    diagnostic_dir = tmp_path / "diagnostic"
    diagnostic_dir.mkdir()
    out_dir = tmp_path / "bundle"

    pd.DataFrame(
        [
            {
                "combo_id": "flat_candidate_set",
                "rule_id": "none",
                "rule_spec": "{}",
                "threshold_mode": "expanding",
                "min_threshold_history": 500,
                "candidate_rows": 1000,
                "candidate_start": "2026-01-01 00:00:00+00:00",
                "candidate_end": "2026-06-01 00:00:00+00:00",
            },
            {
                "combo_id": "flat_candidate_set",
                "rule_id": "lb_recent_hr_drift_rank_m005",
                "rule_spec": '{"action":"rank","condition":"recent_hr_bad_or_drift_high","heads":["long_bars"],"value":-0.005}',
                "threshold_mode": "expanding",
                "min_threshold_history": 500,
                "candidate_rows": 1000,
                "candidate_start": "2026-01-01 00:00:00+00:00",
                "candidate_end": "2026-06-01 00:00:00+00:00",
            },
        ]
    ).to_csv(attribution_dir / "conditional_filter_summary.csv", index=False)
    (attribution_dir / "conditional_filter_summary.json").write_text(
        '{"source":"flat.parquet","source_mode":"flat_candidate_table"}'
    )
    pd.DataFrame(
        {
            "day": ["2026-01-01", "2026-06-01"],
            "rule_id": ["none", "lb_recent_hr_drift_rank_m005"],
            "net_pnl": [1.0, 2.0],
        }
    ).to_csv(attribution_dir / "conditional_filter_daily.csv", index=False)

    (selection_dir / "multiwindow_selection.json").write_text(
        f"""
        {{
          "attribution_dir": "{attribution_dir}",
          "baseline_rule": "none",
          "recommended": {{"rule_id": "lb_recent_hr_drift_rank_m005"}},
          "profile_recommendations": {{"balanced_tolerant": {{"rule_id": "lb_recent_hr_drift_rank_m005"}}}}
        }}
        """
    )
    pd.DataFrame(
        [
            {
                "rule_id": "lb_recent_hr_drift_rank_m005",
                "core_pnl_tail_gate_count": 2,
                "core_strict_tail_gate_count": 2,
                "core_min_delta_objective": -17.5,
                "full_delta_net_pnl": 1471.0,
                "full_delta_objective": 54.5,
                "full_delta_weekly_q20": 0.0,
                "full_delta_weighted_daily_tail": 0.0,
                "entrant_minus_removed_hit_rate": 0.29,
            }
        ]
    ).to_csv(selection_dir / "multiwindow_candidate_summary.csv", index=False)

    (diagnostic_dir / "multiwindow_selection.json").write_text(
        f'{{"attribution_dir":"{attribution_dir}","baseline_rule":"none"}}'
    )
    pd.DataFrame(
        [
            {
                "rule_id": "ood_uncertainty_rank_m005",
                "rule_spec": '{"action":"rank","condition":"ood_high_and_uncertainty_high","heads":["long_bars"],"value":-0.005}',
            }
        ]
    ).to_csv(attribution_dir / "conditional_filter_summary.csv", mode="a", index=False, header=False)
    pd.DataFrame(
        [
            {
                "rule_id": "ood_uncertainty_rank_m005",
                "core_pnl_tail_gate_count": 0,
                "core_min_delta_objective": -100.0,
                "full_delta_net_pnl": -10.0,
                "full_delta_objective": -2.0,
                "full_delta_weekly_q20": -1.0,
                "full_delta_weighted_daily_tail": 0.0,
            }
        ]
    ).to_csv(diagnostic_dir / "multiwindow_candidate_summary.csv", index=False)

    bundle = _materialize_bundle(
        bundle_id="test_bundle",
        attribution_dir=attribution_dir,
        selection_dir=selection_dir,
        rule_id="lb_recent_hr_drift_rank_m005",
        out_dir=out_dir,
        role="research",
        promotion_note="needs fresh gate",
        baseline_rule="none",
        diagnostic_selection_dirs=[diagnostic_dir],
    )

    rule = bundle["rules"]["lb_recent_hr_drift_rank_m005"]
    assert rule["heads"] == ["long_bars"]
    assert rule["families"] == ["drift", "recent_hit_rate_surprise"]
    assert float(rule["metrics"]["full_delta_net_pnl"]) == 1471.0
    assert bundle["candidate_universe"]["source"] == "flat.parquet"
    assert bundle["forward_validation_requirements"]["minimum_post_cutoff_rows"] == 2000
    assert bundle["diagnostic_family_evidence"][0]["families"] == "ood,uncertainty"
    assert (out_dir / "test_bundle.json").exists()
    assert (out_dir / "frozen_reliability_rules.json").exists()


def test_frozen_reliability_profile_selector_preserves_pnl_tail_tradeoff(tmp_path: Path) -> None:
    from scripts.select_frozen_reliability_candidate_profile import run

    status = tmp_path / "candidate_status.csv"
    pd.DataFrame(
        [
            {
                "rule_id": "tail",
                "role": "tail",
                "delta_net_pnl": 100.0,
                "delta_objective": 10.0,
                "active_positive_week_share": 1.0,
                "worst_week_delta": 0.0,
                "entrant_minus_removed_hit_rate": 0.2,
                "delta_net_pnl_p05": 5.0,
                "delta_objective_p05": 1.0,
                "research_pass": True,
                "tail_clean": True,
            },
            {
                "rule_id": "pnl",
                "role": "pnl",
                "delta_net_pnl": 300.0,
                "delta_objective": 20.0,
                "active_positive_week_share": 0.8,
                "worst_week_delta": -50.0,
                "entrant_minus_removed_hit_rate": 0.1,
                "delta_net_pnl_p05": 10.0,
                "delta_objective_p05": 2.0,
                "research_pass": True,
                "tail_clean": False,
            },
            {
                "rule_id": "replacement",
                "role": "replacement",
                "delta_net_pnl": 80.0,
                "delta_objective": 8.0,
                "active_positive_week_share": 0.7,
                "worst_week_delta": -30.0,
                "entrant_minus_removed_hit_rate": 0.5,
                "delta_net_pnl_p05": 4.0,
                "delta_objective_p05": 1.0,
                "research_pass": True,
                "tail_clean": False,
            },
        ]
    ).to_csv(status, index=False)
    (tmp_path / "frozen_reliability_status.json").write_text(
        json.dumps(
            {
                "research_ready": True,
                "fresh_ready": False,
                "production_ready": False,
                "fresh_blockers": ["post_cutoff_rows_lt_2000"],
                "gate_summary": {
                    "post_cutoff_rows": 1833,
                    "policy_action_rows": 130,
                    "policy_outcome_rows": 116,
                },
            }
        )
    )

    out_dir = tmp_path / "out"
    run([status], out_dir)
    selected = pd.read_csv(out_dir / "frozen_reliability_profile_selection.csv").set_index("profile")
    scored = pd.read_csv(out_dir / "frozen_reliability_profile_scored_candidates.csv").set_index("rule_id")

    assert selected.loc["tail_first", "selected_rule_id"] == "tail"
    assert selected.loc["pnl_first", "selected_rule_id"] == "pnl"
    assert selected.loc["replacement_quality", "selected_rule_id"] == "replacement"
    assert bool(scored.loc["tail", "pareto_frontier"]) is True
    assert bool(scored.loc["pnl", "pareto_frontier"]) is True
    assert bool(scored.loc["tail", "source_fresh_ready"]) is False
    assert scored.loc["tail", "source_fresh_blockers"] == "post_cutoff_rows_lt_2000"
    assert selected.loc["tail_first", "fresh_blockers"] == "post_cutoff_rows_lt_2000"


def test_frozen_reliability_long_period_comparison_merges_status_artifacts(tmp_path: Path) -> None:
    from scripts.report_frozen_reliability_long_period_comparison import run

    status_dir = tmp_path / "frozen_reliability_test_status_20260701"
    status_dir.mkdir()
    (status_dir / "frozen_reliability_status.json").write_text(
        json.dumps(
            {
                "research_ready": True,
                "fresh_ready": False,
                "production_ready": False,
                "fresh_blockers": ["post_cutoff_rows_lt_2000"],
                "gate_summary": {
                    "post_cutoff_rows": 10,
                    "policy_action_rows": 4,
                    "policy_outcome_rows": 3,
                },
            }
        )
    )
    pd.DataFrame(
        [
            {
                "rule_id": "rule_a",
                "role": "candidate",
                "delta_net_pnl": 12.0,
                "delta_objective": 3.0,
                "active_positive_week_share": 1.0,
                "worst_week_delta": 0.0,
                "entrant_minus_removed_hit_rate": 0.1,
                "tail_clean": True,
            }
        ]
    ).to_csv(status_dir / "frozen_reliability_candidate_status.csv", index=False)
    pd.DataFrame(
        [
            {
                "rule_id": "rule_a",
                "weeks": 2,
                "active_weeks": 1,
                "q25_week_delta": 0.0,
                "median_week_delta": 2.0,
            }
        ]
    ).to_csv(status_dir / "frozen_reliability_temporal_stability.csv", index=False)
    pd.DataFrame(
        [
            {
                "rule_id": "rule_a",
                "month": "2026-06",
                "delta_net_pnl": 12.0,
                "delta_trades": -1,
                "delta_hit_rate": 0.02,
                "delta_full_sl_rate": -0.01,
            },
            {
                "rule_id": "rule_a",
                "month": "2026-05",
                "delta_net_pnl": -1.0,
                "delta_trades": 0,
                "delta_hit_rate": 0.0,
                "delta_full_sl_rate": 0.0,
            }
        ]
    ).to_csv(status_dir / "frozen_reliability_monthly_deltas.csv", index=False)
    pd.DataFrame(
        [
            {
                "rule_id": "rule_a",
                "week": "2026-06-01/2026-06-07",
                "delta_net_pnl": -2.0,
                "delta_trades": 0,
                "delta_hit_rate": 0.0,
                "delta_full_sl_rate": 0.0,
            }
        ]
    ).to_csv(status_dir / "frozen_reliability_worst_weeks.csv", index=False)

    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    pd.DataFrame(
        [
            {
                "profile": "balanced",
                "selected_rule_id": "rule_a",
                "delta_net_pnl": 12.0,
                "delta_objective": 3.0,
                "tail_clean": True,
                "fresh_ready": False,
                "production_ready": False,
                "fresh_blockers": "post_cutoff_rows_lt_2000",
            }
        ]
    ).to_csv(profile_dir / "frozen_reliability_profile_selection.csv", index=False)
    family_dir = tmp_path / "family_evidence"
    family_dir.mkdir()
    pd.DataFrame(
        [
            {
                "label": "longbars_uncertainty_only",
                "diagnostic_family": "uncertainty",
                "daily_weekly_objective": 4.0,
                "sum_delta_net_pnl": 11.0,
                "positive_week_count": 3,
                "positive_week_share": 0.75,
                "q35_day_delta_net_pnl": 1.0,
                "q20_day_delta_net_pnl": 0.5,
                "mean_day_full_sl_delta": -0.01,
                "june_net_delta": 2.0,
                "positive_month_count": 2,
            },
            {
                "label": "longbars_better_uncertainty",
                "diagnostic_family": "uncertainty",
                "daily_weekly_objective": 5.0,
                "sum_delta_net_pnl": 12.0,
                "positive_week_count": 4,
                "positive_week_share": 0.8,
                "q35_day_delta_net_pnl": 2.0,
                "q20_day_delta_net_pnl": 1.0,
                "mean_day_full_sl_delta": -0.02,
                "june_net_delta": 4.0,
                "positive_month_count": 2,
            },
            {
                "label": "longbars_tail_uncertainty",
                "diagnostic_family": "uncertainty",
                "daily_weekly_objective": 2.0,
                "sum_delta_net_pnl": 5.0,
                "positive_week_count": 3,
                "positive_week_share": 0.7,
                "q35_day_delta_net_pnl": 3.0,
                "q20_day_delta_net_pnl": 3.0,
                "mean_day_full_sl_delta": -0.03,
                "june_net_delta": 2.0,
                "positive_month_count": 2,
            },
            {
                "label": "shortasset_uncertainty_only",
                "diagnostic_family": "uncertainty",
                "daily_weekly_objective": 1.0,
                "sum_delta_net_pnl": 2.0,
                "positive_week_count": 2,
                "positive_week_share": 0.6,
                "q35_day_delta_net_pnl": 0.2,
                "q20_day_delta_net_pnl": 0.1,
                "mean_day_full_sl_delta": -0.005,
                "june_net_delta": 1.0,
                "positive_month_count": 1,
            }
        ]
    ).to_csv(family_dir / "diagnostic_family_long_window_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "label": "longbars_uncertainty_only",
                "diagnostic_family": "uncertainty",
                "months": 2,
                "positive_month_count": 2,
                "min_month_delta_net_pnl": 2.0,
                "apr_jun_delta_net_pnl": 9.0,
                "june_delta_net_pnl": 3.0,
            },
            {
                "label": "longbars_better_uncertainty",
                "diagnostic_family": "uncertainty",
                "months": 2,
                "positive_month_count": 2,
                "min_month_delta_net_pnl": 3.0,
                "apr_jun_delta_net_pnl": 10.0,
                "june_delta_net_pnl": 4.0,
            },
            {
                "label": "longbars_tail_uncertainty",
                "diagnostic_family": "uncertainty",
                "months": 2,
                "positive_month_count": 2,
                "min_month_delta_net_pnl": 1.0,
                "apr_jun_delta_net_pnl": 5.0,
                "june_delta_net_pnl": 2.0,
            },
            {
                "label": "shortasset_uncertainty_only",
                "diagnostic_family": "uncertainty",
                "months": 2,
                "positive_month_count": 1,
                "min_month_delta_net_pnl": 0.0,
                "apr_jun_delta_net_pnl": 2.0,
                "june_delta_net_pnl": 1.0,
            }
        ]
    ).to_csv(family_dir / "diagnostic_family_monthly_summary.csv", index=False)
    decision_pack_dir = tmp_path / "decision_pack"
    decision_pack_dir.mkdir()
    pd.DataFrame(
        [
            {
                "rule_id": "rule_a",
                "delta_avg_week_pnl": 3.0,
                "delta_weighted_daily_tail": 1.5,
                "delta_daily_q20": 1.0,
                "delta_daily_q35": 1.714285714,
                "delta_weekly_q20": 0.5,
                "delta_weekly_q35": 0.75,
            }
        ]
    ).to_csv(decision_pack_dir / "decision_pack_summary.csv", index=False)
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({"rules": {"rule_a": {"role": "candidate"}}}))
    gate_dir = tmp_path / "gate"
    gate_dir.mkdir()
    (gate_dir / "frozen_reliability_gate_manifest.json").write_text(
        json.dumps(
            {
                "bundle": str(bundle),
                "cutoff": "2026-06-27T00:00:00+00:00",
                "ran_gate": False,
                "ready_sources": 0,
                "nearest_source": {
                    "post_cutoff_rows": 8,
                    "policy_action_rows_estimate": 2,
                    "policy_outcome_rows_estimate": 1,
                    "rejection_reasons": "policy_outcome_rows_lt_50",
                },
                "readiness_deficits": [
                    {
                        "gate": "policy_outcome_rows",
                        "observed": 1,
                        "required": 50,
                        "deficit": 49,
                        "pass": False,
                    }
                ],
            }
        )
    )

    out_dir = tmp_path / "out"
    payload = run([status_dir], out_dir, profile_dir, [decision_pack_dir], [gate_dir], [family_dir])

    summary = pd.read_csv(out_dir / "frozen_reliability_long_period_candidate_summary.csv")
    decision = pd.read_csv(out_dir / "frozen_reliability_long_period_decision_matrix.csv")
    gate_snapshots = pd.read_csv(out_dir / "frozen_reliability_long_period_gate_snapshots.csv")
    gate_deficits = pd.read_csv(out_dir / "frozen_reliability_long_period_gate_deficits.csv")
    family_evidence = pd.read_csv(out_dir / "frozen_reliability_long_period_family_evidence.csv")
    frontier = pd.read_csv(out_dir / "frozen_reliability_long_period_promotion_frontier.csv")
    champions = pd.read_csv(out_dir / "frozen_reliability_long_period_scope_champions.csv")
    monthly = pd.read_csv(out_dir / "frozen_reliability_long_period_monthly_deltas.csv")
    profile = pd.read_csv(out_dir / "frozen_reliability_long_period_profile_winners.csv")

    assert payload["candidate_count"] == 1
    assert payload["decision_matrix_rows"] == 1
    assert payload["decision_pack_rows"] == 1
    assert payload["gate_snapshot_rows"] == 1
    assert payload["gate_deficit_rows"] == 1
    assert payload["gate_override_rows"] == 1
    assert payload["family_evidence_rows"] == 4
    assert payload["promotion_frontier_rows"] == 5
    assert payload["scope_champion_rows"] == 9
    assert summary.loc[0, "rule_id"] == "rule_a"
    assert bool(summary.loc[0, "fresh_ready"]) is False
    assert summary.loc[0, "fresh_blockers"] == "post_cutoff_rows_lt_2000"
    assert summary.loc[0, "post_cutoff_rows"] == 10
    assert summary.loc[0, "delta_weighted_daily_tail"] == 1.5
    assert summary.loc[0, "delta_daily_q20"] == 1.0
    assert decision.loc[0, "selected_profiles"] == "balanced"
    assert decision.loc[0, "decision_state"] == "research_tail_robust_wait_fresh"
    assert bool(decision.loc[0, "tail_robust_pass"]) is True
    assert bool(decision.loc[0, "fresh_ready_pass"]) is False
    assert decision.loc[0, "current_fresh_blockers"] == "policy_outcome_rows_lt_50"
    assert gate_snapshots.loc[0, "current_post_cutoff_rows"] == 8
    assert gate_deficits.loc[0, "rule_id"] == "rule_a"
    assert gate_deficits.loc[0, "gate"] == "policy_outcome_rows"
    assert gate_deficits.loc[0, "deficit"] == 49
    assert bool(gate_deficits.loc[0, "pass"]) is False
    family_by_label = family_evidence.set_index("label")
    assert family_by_label.loc["longbars_uncertainty_only", "diagnostic_family"] == "uncertainty"
    assert family_by_label.loc["longbars_uncertainty_only", "daily_weekly_objective"] == 4.0
    frontier_by_item = frontier.set_index("item_id")
    assert frontier_by_item.loc["rule_a", "verdict"] == "frozen_tail_robust_wait_fresh"
    assert frontier_by_item.loc["longbars_uncertainty_only", "verdict"] == "family_pnl_tail_clean_research"
    assert frontier_by_item.loc["longbars_uncertainty_only", "comparison_scope"] == "long_bars"
    assert frontier_by_item.loc["shortasset_uncertainty_only", "comparison_scope"] == "short_asset"
    assert bool(frontier_by_item.loc["longbars_uncertainty_only", "pareto_dominated"]) is True
    assert frontier_by_item.loc["longbars_uncertainty_only", "dominated_by"] == "longbars_better_uncertainty"
    assert bool(frontier_by_item.loc["longbars_better_uncertainty", "pareto_dominated"]) is False
    assert bool(frontier_by_item.loc["shortasset_uncertainty_only", "pareto_dominated"]) is False
    assert frontier_by_item.loc["rule_a", "positive_month_count"] == 1
    assert frontier_by_item.loc["rule_a", "min_month_delta_net_pnl"] == -1.0
    assert frontier_by_item.loc["longbars_uncertainty_only", "months"] == 2
    assert frontier_by_item.loc["longbars_uncertainty_only", "apr_jun_delta_net_pnl"] == 9.0
    champion_by_scope_role = champions.set_index(["comparison_scope", "champion_role"])
    assert champion_by_scope_role.loc[("frozen_candidate", "best_pnl"), "champion_item_id"] == "rule_a"
    assert champion_by_scope_role.loc[("long_bars", "best_pnl"), "champion_item_id"] == "longbars_better_uncertainty"
    assert champion_by_scope_role.loc[("long_bars", "best_tail"), "champion_item_id"] == "longbars_tail_uncertainty"
    assert champion_by_scope_role.loc[("short_asset", "balanced"), "champion_item_id"] == "shortasset_uncertainty_only"
    assert summary.loc[0, "current_post_cutoff_rows"] == 8
    assert set(monthly["month"]) == {"2026-05", "2026-06"}
    assert profile.loc[0, "selected_rule_id"] == "rule_a"
    assert (out_dir / "frozen_reliability_long_period_comparison.md").exists()


def test_weekly_combo_switch_parser_handles_underscored_head_names() -> None:
    from scripts.replay_contextual_tp_sl_weekly_combo_switching import _combo_id, _parse_combo_id

    combo = "long_bars:S_long_dist:R_short_asset:J_short_bollinger:I"

    parsed = _parse_combo_id(combo)

    assert parsed == {
        "long_bars": "static",
        "long_dist": "rank_only",
        "short_asset": "joint_all",
        "short_bollinger": "independent_all",
    }
    assert _combo_id(parsed) == combo


def test_weekly_rule_selector_tail_guard_blocks_negative_tail_candidate() -> None:
    from scripts.replay_contextual_tp_sl_weekly_rule_selector import BASELINE_ID, _select_for_week

    week_start = pd.Timestamp("2026-02-01", tz="UTC")
    daily = pd.DataFrame(
        {
            "candidate_id": [BASELINE_ID, BASELINE_ID, "candidate", "candidate"],
            "day_start": [
                week_start - pd.Timedelta(days=6),
                week_start - pd.Timedelta(days=5),
                week_start - pd.Timedelta(days=6),
                week_start - pd.Timedelta(days=5),
            ],
            "net_pnl": [10.0, 10.0, 100.0, -1000.0],
        }
    )
    weekly = pd.DataFrame(
        {
            "candidate_id": [BASELINE_ID, "candidate"],
            "week_start": [week_start - pd.Timedelta(days=7), week_start - pd.Timedelta(days=7)],
            "net_pnl": [20.0, 200.0],
        }
    )

    selected, reason, stats = _select_for_week(
        daily,
        weekly,
        candidate_ids=[BASELINE_ID, "candidate"],
        week_start=week_start,
        lookback_weeks=2,
        min_history_weeks=1,
        mode="objective_tail_guard",
        min_objective_delta=0.0,
        min_net_delta=0.0,
    )

    assert selected == BASELINE_ID
    assert reason == "fallback_objective_tail_guard"
    assert stats == {}


def test_weekly_intervention_oracle_replaces_only_selected_week() -> None:
    from scripts.replay_contextual_tp_sl_weekly_intervention_oracle import _stream_for_one_week

    weeks = pd.to_datetime(["2026-05-04", "2026-05-11"], utc=True)
    baseline = pd.DataFrame(
        {
            "timestamp": [weeks[0], weeks[1]],
            "strategy_id": ["short_asset_alpha", "short_asset_alpha"],
            "symbol": ["BTC", "ETH"],
            "source": ["baseline_w1", "baseline_w2"],
        }
    )
    candidate = pd.DataFrame(
        {
            "timestamp": [weeks[0], weeks[1]],
            "strategy_id": ["short_asset_alpha", "short_asset_alpha"],
            "symbol": ["BTC", "ETH"],
            "source": ["candidate_w1", "candidate_w2"],
        }
    )

    stream = _stream_for_one_week(baseline, candidate, weeks[0])

    assert set(stream["source"]) == {"candidate_w1", "baseline_w2"}
    assert "baseline_w1" not in set(stream["source"])
    assert "candidate_w2" not in set(stream["source"])


def test_frozen_reliability_acceptance_matrix_tracks_requested_families(tmp_path: Path) -> None:
    from scripts.report_frozen_reliability_acceptance_matrix import run

    status_path = tmp_path / "candidate_status.csv"
    pd.DataFrame(
        {
            "rule_id": [
                "lb_drift_rank_m0025",
                "lb_recent_hr_drift_rank_m005",
                "all_ood_uncertainty_rank_m005",
            ],
            "role": ["tail", "replacement", "diagnostic"],
            "delta_net_pnl": [100.0, 80.0, 20.0],
            "delta_objective": [10.0, 8.0, 2.0],
            "active_positive_week_share": [1.0, 0.8, 0.6],
            "worst_week_delta": [0.0, -100.0, -250.0],
            "prob_delta_net_pnl_positive": [0.99, 0.99, 0.99],
            "prob_delta_objective_positive": [0.99, 0.99, 0.99],
            "delta_net_pnl_p05": [1.0, 1.0, 1.0],
            "delta_objective_p05": [1.0, 1.0, 1.0],
            "entrant_minus_removed_hit_rate": [0.2, 0.3, 0.1],
            "entrant_minus_removed_full_sl_rate": [-0.2, -0.1, 0.1],
            "research_pass": [True, True, True],
            "tail_clean": [True, False, False],
        }
    ).to_csv(status_path, index=False)
    gaps_path = tmp_path / "fresh_gaps.csv"
    pd.DataFrame(
        {
            "gate": ["post_cutoff_rows", "policy_action_rows"],
            "head": ["", ""],
            "observed": [1833, 130],
            "required": [2000, 50],
            "deficit": [167, 0],
            "pass": [False, True],
        }
    ).to_csv(gaps_path, index=False)
    family_verdict_path = tmp_path / "family_verdict.csv"
    pd.DataFrame(
        {
            "family": ["drift", "recent_hit_rate_surprise", "ood", "uncertainty"],
            "finite_row_rate": [0.99, 1.0, 0.64, 0.63],
            "tested_in_scorecards": [True, True, True, True],
            "best_long_window_delta_net_pnl": [100.0, 80.0, 10.0, 0.0],
            "best_q20_delta_pnl": [5.0, 4.0, -1.0, 0.0],
            "verdict": [
                "helpful_in_tests",
                "helpful_in_tests",
                "helpful_in_tests",
                "tested_no_clear_lift",
            ],
        }
    ).to_csv(family_verdict_path, index=False)
    ab_scorecard_path = tmp_path / "ab_scorecard.csv"
    pd.DataFrame(
        {
            "source": ["expanding_family", "expanding_family", "promotion"],
            "evidence_family": ["ood_test", "uncertainty_test", "promotion"],
            "variant": ["recent_drift_ood", "uncertainty_recent_drift", "recent_drift"],
            "family": ["recent_hr+drift+OOD", "recent_hr+drift+uncertainty", "recent_hr+drift"],
            "delta_net_pnl": [30.0, 20.0, 100.0],
            "delta_objective": [3.0, 2.0, 10.0],
            "delta_full_sl_rate": [-0.02, 0.01, -0.01],
            "tail_metric": [5.0, -5.0, 0.0],
            "delta_q20_pnl": [1.0, -1.0, 2.0],
            "delta_q35_pnl": [1.0, 0.0, 2.0],
            "scorecard_score": [30.0, 20.0, 100.0],
            "contains_drift": [True, True, True],
            "contains_recent_hit_rate_surprise": [True, True, True],
            "contains_ood": [True, False, False],
            "contains_uncertainty": [False, True, False],
            "ab_verdict": [
                "pnl_and_tail_supportive",
                "pnl_positive_tail_weak",
                "pnl_and_tail_supportive",
            ],
        }
    ).to_csv(ab_scorecard_path, index=False)
    marginal_path = tmp_path / "marginal_family_ablation.csv"
    pd.DataFrame(
        {
            "family": ["ood", "uncertainty", "drift"],
            "variant": ["recent_drift_ood", "uncertainty_recent_drift", "recent_drift"],
            "baseline_variant": ["recent_drift", "recent_drift", "baseline"],
            "marginal_delta_net_pnl": [7.0, -3.0, 100.0],
            "marginal_delta_objective": [1.0, -1.0, 10.0],
            "marginal_delta_q20_pnl": [-2.0, -1.0, 5.0],
            "marginal_delta_q35_pnl": [2.0, 0.0, 5.0],
            "marginal_scorecard_score": [8.0, -4.0, 110.0],
        }
    ).to_csv(marginal_path, index=False)

    out_dir = tmp_path / "acceptance"
    run(
        candidate_status=[status_path],
        fresh_evidence_gap=[gaps_path],
        family_verdict=[family_verdict_path],
        ab_scorecard=[ab_scorecard_path],
        marginal_family_ablation=[marginal_path],
        out_dir=out_dir,
    )

    matrix = pd.read_csv(out_dir / "frozen_reliability_acceptance_matrix.csv").set_index("rule_id")
    ab_matrix = pd.read_csv(out_dir / "frozen_reliability_ab_acceptance_matrix.csv").set_index("variant")
    summary = pd.read_csv(out_dir / "frozen_reliability_acceptance_family_summary.csv").set_index("family")

    assert matrix.loc["lb_drift_rank_m0025", "acceptance_state"] == "tail_research_ready_wait_fresh"
    assert (
        matrix.loc["lb_recent_hr_drift_rank_m005", "acceptance_state"]
        == "replacement_research_ready_wait_fresh"
    )
    assert matrix.loc["all_ood_uncertainty_rank_m005", "acceptance_state"] == "pnl_research_ready_wait_fresh"
    assert bool(matrix.loc["lb_recent_hr_drift_rank_m005", "uses_recent_hit_rate_surprise"]) is True
    assert bool(matrix.loc["lb_recent_hr_drift_rank_m005", "uses_drift"]) is True
    assert bool(matrix.loc["all_ood_uncertainty_rank_m005", "uses_ood"]) is True
    assert bool(matrix.loc["all_ood_uncertainty_rank_m005", "uses_uncertainty"]) is True
    assert int(summary.loc["drift", "candidate_count"]) == 2
    assert int(summary.loc["recent_hit_rate_surprise", "replacement_quality_count"]) == 1
    assert int(summary.loc["ood", "replacement_quality_count"]) == 0
    assert summary.loc["ood", "scorecard_verdict"] == "helpful_in_tests"
    assert summary.loc["uncertainty", "scorecard_verdict"] == "tested_no_clear_lift"
    assert float(summary.loc["ood", "finite_row_rate"]) == 0.64
    assert ab_matrix.loc["recent_drift_ood", "ab_acceptance_state"] == "ab_pnl_tail_supportive"
    assert ab_matrix.loc["uncertainty_recent_drift", "ab_acceptance_state"] == "ab_pnl_tail_weak"
    assert int(summary.loc["ood", "ab_tail_supportive_count"]) == 1
    assert int(summary.loc["uncertainty", "ab_tail_supportive_count"]) == 0
    assert float(summary.loc["ood", "best_marginal_delta_net_pnl"]) == 7.0
    assert float(summary.loc["uncertainty", "best_marginal_delta_net_pnl"]) == -3.0
    assert summary.loc["ood", "best_marginal_baseline_variant"] == "recent_drift"
    assert "post_cutoff_rows_deficit_167" in matrix.loc["lb_drift_rank_m0025", "fresh_blockers"]


def test_reliability_family_freezeability_separates_frozen_explicit_and_diagnostic(
    tmp_path: Path,
) -> None:
    from scripts.report_reliability_family_freezeability import run

    status_path = tmp_path / "candidate_status.csv"
    pd.DataFrame(
        {
            "rule_id": ["lb_drift_rank_m005"],
            "role": ["pnl"],
            "promotion_note": ["drift challenger"],
            "delta_net_pnl": [100.0],
            "delta_objective": [10.0],
        }
    ).to_csv(status_path, index=False)
    conditional_path = tmp_path / "conditional_filter_summary.csv"
    pd.DataFrame(
        {
            "rule_id": ["lb_recent_hr_ood_rank_m005", "lb_uncertainty_rank_m005"],
            "rule_spec": [
                '{"action": "rank", "condition": "recent_hr_bad_or_ood_high", "heads": ["long_bars"], "value": -0.005}',
                '{"action": "rank", "condition": "uncertainty_high", "heads": ["long_bars"], "value": -0.005}',
            ],
            "objective_avgweek_0p7dayq35_0p3dayq20": [50.0, 40.0],
        }
    ).to_csv(conditional_path, index=False)
    verdict_path = tmp_path / "verdict.csv"
    pd.DataFrame(
        {
            "family": ["drift", "recent_hit_rate_surprise", "ood", "uncertainty"],
            "verdict": [
                "helpful_in_tests",
                "helpful_in_tests",
                "helpful_in_tests",
                "tested_no_clear_lift",
            ],
            "finite_row_rate": [1.0, 1.0, 0.7, 0.7],
        }
    ).to_csv(verdict_path, index=False)
    marginal_path = tmp_path / "marginal.csv"
    pd.DataFrame(
        {
            "family": ["ood", "uncertainty"],
            "variant": ["recent_drift_ood", "uncertainty_recent_drift"],
            "baseline_variant": ["recent_drift", "recent_drift"],
            "marginal_delta_net_pnl": [20.0, 0.0],
            "marginal_delta_objective": [5.0, 0.0],
            "marginal_delta_q20_pnl": [-1.0, 0.0],
            "marginal_scorecard_score": [30.0, 0.0],
        }
    ).to_csv(marginal_path, index=False)

    out_dir = tmp_path / "freezeability"
    run(
        candidate_status=[status_path],
        conditional_summary=[conditional_path],
        family_verdict=[verdict_path],
        marginal_family_ablation=[marginal_path],
        out_dir=out_dir,
    )

    rows = pd.read_csv(out_dir / "reliability_family_freezeability.csv").set_index("family")

    assert rows.loc["drift", "freezeability_decision"] == "already_frozen_candidate_wait_fresh"
    assert (
        rows.loc["ood", "freezeability_decision"]
        == "explicit_rule_available_tail_mixed_needs_multiwindow"
    )
    assert rows.loc["uncertainty", "freezeability_decision"] == "diagnostic_only_no_positive_marginal_lift"
    assert rows.loc["ood", "explicit_rule_id"] == "lb_recent_hr_ood_rank_m005"
    assert float(rows.loc["ood", "marginal_delta_net_pnl"]) == 20.0


def test_reliability_family_multiwindow_gate_distinguishes_tail_pass_and_reject(
    tmp_path: Path,
) -> None:
    from scripts.report_reliability_family_multiwindow_gate import run

    summary_path = tmp_path / "multiwindow_candidate_summary.csv"
    pd.DataFrame(
        {
            "rule_id": [
                "sa_ood_rank_m002",
                "recent_hr_ood_rank_m005",
                "lb_uncertainty_rank_m002",
                "lb_two_signal_rank_m005",
            ],
            "core_pnl_tail_gate_count": [3, 2, 0, 3],
            "core_strict_tail_gate_count": [3, 1, 0, 3],
            "core_min_delta_objective": [1.0, -2.0, -10.0, 1.0],
            "core_min_delta_net_pnl": [10.0, 5.0, -5.0, 10.0],
            "core_min_delta_weekly_q20": [0.0, -1.0, -5.0, 0.0],
            "core_min_delta_weighted_daily_tail": [0.0, 0.0, -1.0, 0.0],
            "full_delta_objective": [10.0, 5.0, -1.0, 7.0],
            "full_delta_net_pnl": [100.0, 50.0, -10.0, 70.0],
            "june_delta_objective": [5.0, 2.0, -2.0, 3.0],
            "june_delta_net_pnl": [30.0, 20.0, -5.0, 30.0],
            "entrant_minus_removed_hit_rate": [0.2, 0.1, -0.1, 0.1],
        }
    ).to_csv(summary_path, index=False)

    out_dir = tmp_path / "multiwindow"
    run(multiwindow_summary=[summary_path], out_dir=out_dir)

    candidates = pd.read_csv(out_dir / "reliability_family_multiwindow_candidates.csv").set_index("rule_id")
    family = pd.read_csv(out_dir / "reliability_family_multiwindow_summary.csv").set_index("family")

    assert candidates.loc["sa_ood_rank_m002", "multiwindow_gate_state"] == "multiwindow_strict_tail_pass"
    assert candidates.loc["recent_hr_ood_rank_m005", "multiwindow_gate_state"] == "multiwindow_mixed_positive"
    assert candidates.loc["lb_uncertainty_rank_m002", "multiwindow_gate_state"] == "reject_nonpositive_full"
    assert family.loc["ood", "recommendation"] == "freeze_candidate_available"
    assert family.loc["uncertainty", "recommendation"] == "composite_only_needs_family_isolation"
    assert int(family.loc["uncertainty", "focused_tail_pass_count"]) == 0
    assert int(family.loc["uncertainty", "composite_tail_pass_count"]) == 1
