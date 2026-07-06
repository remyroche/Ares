from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_gmm_cluster_policy_smoke import (
    CALIBRATION_SELECTOR_BY_POLICY,
    FINAL_STAGE_BY_POLICY,
    _apply_cluster_side_policy_score,
    _assign_cluster_side_policy,
    _build_label_viability_matrix,
    _build_train_meta_readiness_matrix,
    _calibrated_soft_risk_score,
    _calibration_viability_rows,
    _fit_side_calibrated_score,
    _fit_side_calibrated_risk_predictions,
    _oracle_recall_summary,
    _per_timestamp_top_mask,
    _ranker_relevance,
    _recall_preserving_calibrated_risk_score,
    _risk_constrained_backfill_top_score,
    _risk_capped_score,
    _risk_trimmed_top_score,
    _score_quantile_side_rows,
    _selection_metrics,
    _side_capped_top_score,
    _side_exposure_capped_score,
    _stage_gate_diagnostic_rows,
)


def test_cluster_side_policy_splits_cancelling_cluster_sides() -> None:
    train_side_stats = pd.DataFrame(
        [
            {
                "cluster": 0,
                "side_name": "long",
                "side": 1,
                "train_side_rows": 100,
                "train_side_mean_u": -0.0010,
                "train_side_hit_u": 0.35,
                "train_side_q10_u": -0.02,
                "train_side_bad_mae_1r_rate": 0.60,
            },
            {
                "cluster": 0,
                "side_name": "short",
                "side": -1,
                "train_side_rows": 100,
                "train_side_mean_u": 0.0015,
                "train_side_hit_u": 0.45,
                "train_side_q10_u": -0.01,
                "train_side_bad_mae_1r_rate": 0.60,
            },
        ]
    )
    cluster_policy = pd.DataFrame(
        [{"cluster": 0, "cluster_policy_action": "throttle", "cluster_policy_reason": "long_short_cancellation"}]
    )

    policy = _assign_cluster_side_policy(
        train_side_stats,
        cluster_policy,
        min_allow_mean_u=0.00025,
        min_allow_hit_u=0.40,
        block_mean_u=-0.00025,
        min_side_rows=10,
    )

    by_side = policy.set_index("side_name")
    assert by_side.loc["long", "cluster_side_policy_action"] == "block"
    assert by_side.loc["short", "cluster_side_policy_action"] == "allow_normal"


def test_cluster_side_policy_score_applies_actions_per_side() -> None:
    side_policy = pd.DataFrame(
        [
            {"cluster": 0, "side": 1, "side_name": "long", "cluster_side_policy_action": "block"},
            {
                "cluster": 0,
                "side": -1,
                "side_name": "short",
                "cluster_side_policy_action": "allow_normal",
                "cluster_side_policy_adjustment": 0.2,
            },
            {
                "cluster": 1,
                "side": 1,
                "side_name": "long",
                "cluster_side_policy_action": "allow_high_threshold",
                "cluster_side_policy_adjustment": 0.1,
            },
            {"cluster": 1, "side": -1, "side_name": "short", "cluster_side_policy_action": "block"},
        ]
    )

    adjusted, eligible = _apply_cluster_side_policy_score(
        pd.Series([0.99, 0.80, 0.95, 0.70], dtype=np.float32),
        np.asarray([0, 0, 1, 1], dtype=np.int32),
        pd.Series([1, -1, 1, -1], dtype=np.int8),
        side_policy,
        top_frac=1.00,
    )

    assert not bool(eligible.iloc[0])
    assert bool(eligible.iloc[1])
    assert bool(eligible.iloc[2])
    assert not bool(eligible.iloc[3])
    assert np.isnan(float(adjusted.iloc[0]))
    assert float(adjusted.iloc[1]) > 0.80
    assert float(adjusted.iloc[2]) > 0.95
    assert np.isnan(float(adjusted.iloc[3]))


def test_side_calibrated_score_uses_side_specific_utility_mapping() -> None:
    train_score = np.asarray(([0.1, 0.2, 0.9, 1.0] * 10), dtype=np.float32)
    train_metrics = pd.DataFrame(
        {
            "side": ([1, 1, 1, 1] * 5) + ([-1, -1, -1, -1] * 5),
            "u_policy_net": (
                [-0.002, -0.001, 0.003, 0.004] * 5
                + [0.004, 0.003, -0.002, -0.001] * 5
            ),
        }
    )
    valid_score = pd.Series([0.95, 0.95], dtype=np.float32)
    valid_side = pd.Series([1, -1], dtype=np.int8)

    calibrated = _fit_side_calibrated_score(
        train_score=train_score,
        train_metrics=train_metrics,
        valid_score=valid_score,
        valid_side=valid_side,
        n_bins=2,
        min_bin_rows=1,
    )

    assert float(calibrated.iloc[0]) > 0.0
    assert float(calibrated.iloc[1]) < 0.0


def test_calibrated_soft_risk_score_penalizes_predicted_risks() -> None:
    score = _calibrated_soft_risk_score(
        pd.Series([0.004, 0.004], dtype=np.float32),
        pd.DataFrame(
            {
                "pred_bad_mae": [0.0, 1.0],
                "pred_timeout": [0.0, 1.0],
                "pred_lower_tail": [0.0, 1.0],
            }
        ),
        np.asarray([0.0, 0.001], dtype=np.float32),
        bad_mae_lambda=0.001,
        timeout_lambda=0.001,
        lower_tail_lambda=0.001,
    )

    assert float(score.iloc[0]) == np.float32(0.004)
    assert float(score.iloc[1]) < float(score.iloc[0])


def test_side_calibrated_risk_predictions_map_raw_scores_to_realized_rates() -> None:
    train_predictions = pd.DataFrame(
        {
            "pred_bad_mae": ([0.1, 0.2, 0.8, 0.9] * 10),
            "pred_timeout": ([0.1, 0.2, 0.8, 0.9] * 10),
            "pred_lower_tail": ([0.1, 0.2, 0.8, 0.9] * 10),
        }
    )
    train_metrics = pd.DataFrame(
        {
            "side": ([1, 1, 1, 1] * 5) + ([-1, -1, -1, -1] * 5),
            "mae_norm": ([0.3, 0.4, 1.2, 1.4] * 5) + ([1.2, 1.4, 0.3, 0.4] * 5),
            "is_timeout": ([0, 0, 1, 1] * 5) + ([1, 1, 0, 0] * 5),
            "u_policy_net": ([0.01, 0.02, -0.03, -0.04] * 5)
            + ([-0.03, -0.04, 0.01, 0.02] * 5),
        }
    )
    valid_predictions = pd.DataFrame(
        {
            "pred_bad_mae": [0.85, 0.85],
            "pred_timeout": [0.85, 0.85],
            "pred_lower_tail": [0.85, 0.85],
        }
    )

    calibrated = _fit_side_calibrated_risk_predictions(
        train_predictions=train_predictions,
        train_metrics=train_metrics,
        valid_predictions=valid_predictions,
        valid_side=pd.Series([1, -1], dtype=np.int8),
        n_bins=2,
        min_bin_rows=1,
    )

    assert float(calibrated.iloc[0]["pred_bad_mae"]) > 0.9
    assert float(calibrated.iloc[1]["pred_bad_mae"]) < 0.1


def test_recall_preserving_calibrated_risk_score_keeps_candidates_and_penalizes_risk() -> None:
    score = _recall_preserving_calibrated_risk_score(
        base_score=pd.Series([0.010, 0.010, 0.020], dtype=np.float32),
        discovery_score=pd.Series([1.0, 1.0, 1.0], dtype=np.float32),
        calibrated_risk_predictions=pd.DataFrame(
            {
                "pred_bad_mae": [0.10, 0.90, 0.10],
                "pred_timeout": [0.00, 0.00, 0.00],
                "pred_lower_tail": [0.00, 0.00, 0.00],
            }
        ),
        candidate_mask=np.asarray([True, True, False]),
        bad_mae_lambda=0.010,
        timeout_lambda=0.0,
        lower_tail_lambda=0.0,
        discovery_lambda=0.0,
    )

    assert np.isfinite(float(score.iloc[0]))
    assert np.isfinite(float(score.iloc[1]))
    assert np.isnan(float(score.iloc[2]))
    assert float(score.iloc[0]) > float(score.iloc[1])


def test_selection_metrics_reports_weekly_lower_tail() -> None:
    frame = pd.DataFrame(
        {
            "__symbol__": ["A", "B", "A", "B"],
            "__ts__": pd.to_datetime(["2026-06-01", "2026-06-02", "2026-06-09", "2026-06-10"]),
        }
    )
    metrics = pd.DataFrame(
        {
            "u_policy_net": [0.01, 0.03, -0.02, -0.04],
            "ret_net": [0.01, 0.03, -0.02, -0.04],
            "barrier": [0.01, 0.01, 0.02, 0.02],
            "mae_norm": [0.5, 0.5, 1.2, 1.2],
            "mfe_norm": [1.0, 1.0, 0.4, 0.4],
            "is_timeout": [0.0, 0.0, 1.0, 1.0],
            "bars_to_mfe": [2.0, 3.0, 4.0, 5.0],
            "side": [1, -1, 1, -1],
        }
    )
    target = pd.DataFrame({"target_soft": [1.0, 1.0, 0.0, 0.0], "target_hard": [1, 1, 0, 0]})

    row = _selection_metrics(
        frame=frame,
        metrics=metrics,
        target=target,
        score=pd.Series([4.0, 3.0, 2.0, 1.0], dtype=np.float32),
        arm="test",
        selector="test",
        period="2026-06",
        top_frac=1.0,
    )

    assert row["selected_week_count"] == 2
    assert np.isclose(row["worst_weekly_mean_u"], -0.03)
    assert np.isclose(row["weekly_mean_u_q10"], -0.025)
    assert row["positive_week_rate"] == 0.5


def test_oracle_recall_summary_detects_missed_top_rows() -> None:
    oracle = pd.Series([10.0, 9.0, 1.0, 0.0], dtype=np.float32)
    model = pd.Series([0.0, 0.1, 0.9, 1.0], dtype=np.float32)

    recall = _oracle_recall_summary(
        score=model,
        oracle_score=oracle,
        top_frac=0.50,
    )

    assert recall["oracle_top_rows"] == 2
    assert recall["model_selected_rows_for_recall"] == 2
    assert recall["oracle_overlap_rows"] == 0
    assert recall["oracle_recall_at_model_top_k"] == 0.0
    assert recall["oracle_top_score_percentile_mean"] < 0.50


def test_risk_capped_score_filters_high_predicted_bad_mae() -> None:
    score = _risk_capped_score(
        pd.Series([0.4, 0.3, 0.2], dtype=np.float32),
        pd.DataFrame(
            {
                "pred_bad_mae": [0.20, 0.80, 0.40],
                "pred_timeout": [0.01, 0.01, 0.20],
                "pred_lower_tail": [0.05, 0.05, 0.05],
            }
        ),
        max_pred_bad_mae=0.65,
        max_pred_timeout=0.10,
        max_pred_lower_tail=0.20,
    )

    assert np.isfinite(float(score.iloc[0]))
    assert np.isnan(float(score.iloc[1]))
    assert np.isnan(float(score.iloc[2]))


def test_side_capped_top_score_limits_dominant_side_without_forcing_quota() -> None:
    score = pd.Series([0.90, 0.80, 0.70, 0.60, 0.10], dtype=np.float32)
    side = pd.Series([-1, -1, -1, 1, 1], dtype=np.int8)

    capped = _side_capped_top_score(score, side, 0.80, max_side_share=0.75)
    selected = capped.notna()

    assert int(selected.sum()) == 4
    assert int(((side < 0) & selected).sum()) == 3
    assert int(((side > 0) & selected).sum()) == 1


def test_risk_constrained_backfill_top_score_prioritizes_low_risk_with_limited_backfill() -> None:
    score = pd.Series([0.90, 0.80, 0.70, 0.60, 0.50], dtype=np.float32)
    side = pd.Series([-1, -1, 1, 1, -1], dtype=np.int8)
    risk = pd.DataFrame(
        {
            "pred_bad_mae": [0.90, 0.20, 0.30, 0.58, 0.70],
            "pred_timeout": [0.05, 0.05, 0.05, 0.15, 0.05],
            "pred_lower_tail": [0.10, 0.10, 0.10, 0.25, 0.10],
        }
    )

    constrained = _risk_constrained_backfill_top_score(
        score,
        side,
        risk,
        0.80,
        max_side_share=0.75,
        primary_max_pred_bad_mae=0.50,
        primary_max_pred_timeout=0.12,
        primary_max_pred_lower_tail=0.20,
        backfill_max_pred_bad_mae=0.62,
        backfill_max_pred_timeout=0.18,
        backfill_max_pred_lower_tail=0.35,
        max_backfill_share=0.25,
    )

    selected = constrained.notna()
    assert not bool(selected.iloc[0])
    assert bool(selected.iloc[1])
    assert bool(selected.iloc[2])
    assert bool(selected.iloc[3])
    assert not bool(selected.iloc[4])
    assert int(selected.sum()) == 3


def test_risk_constrained_backfill_top_score_enforces_actual_side_share_after_sparse_risk_filter() -> None:
    score = pd.Series([0.90, 0.80, 0.70, 0.60, 0.50], dtype=np.float32)
    side = pd.Series([-1, -1, -1, 1, 1], dtype=np.int8)
    risk = pd.DataFrame(
        {
            "pred_bad_mae": [0.20, 0.30, 0.40, 0.20, 0.90],
            "pred_timeout": [0.05, 0.05, 0.05, 0.05, 0.05],
            "pred_lower_tail": [0.10, 0.10, 0.10, 0.10, 0.10],
        }
    )

    constrained = _risk_constrained_backfill_top_score(
        score,
        side,
        risk,
        1.00,
        max_side_share=0.60,
        primary_max_pred_bad_mae=0.50,
        primary_max_pred_timeout=0.12,
        primary_max_pred_lower_tail=0.20,
        backfill_max_pred_bad_mae=0.62,
        backfill_max_pred_timeout=0.18,
        backfill_max_pred_lower_tail=0.35,
        max_backfill_share=0.00,
    )

    selected = constrained.notna()
    assert int(selected.sum()) == 2
    assert int(((side < 0) & selected).sum()) == 1
    assert int(((side > 0) & selected).sum()) == 1
    assert np.isnan(float(constrained.iloc[1]))
    assert np.isnan(float(constrained.iloc[2]))


def test_risk_trimmed_top_score_protects_top_scores_and_trims_lower_ranked_path_risk() -> None:
    score = pd.Series([0.90, 0.80, 0.70, 0.60, 0.50], dtype=np.float32)
    side = pd.Series([-1, 1, -1, 1, -1], dtype=np.int8)
    risk = pd.DataFrame(
        {
            "pred_bad_mae": [0.95, 0.10, 0.90, 0.80, 0.10],
            "pred_timeout": [0.05, 0.05, 0.05, 0.05, 0.05],
            "pred_lower_tail": [0.10, 0.10, 0.10, 0.10, 0.10],
        }
    )

    trimmed = _risk_trimmed_top_score(
        score,
        side,
        risk,
        1.00,
        max_side_share=0.80,
        trim_share=0.40,
        protect_top_score_share=0.40,
        bad_mae_weight=1.0,
        timeout_weight=0.0,
        lower_tail_weight=0.0,
    )

    selected = trimmed.notna()
    assert bool(selected.iloc[0])
    assert bool(selected.iloc[1])
    assert not bool(selected.iloc[2])
    assert not bool(selected.iloc[3])
    assert bool(selected.iloc[4])


def test_side_exposure_capped_score_drops_weakest_overexposed_side() -> None:
    score = pd.Series([0.90, 0.80, 0.70, 0.60, 0.50], dtype=np.float32)
    side = pd.Series([-1, -1, -1, -1, 1], dtype=np.int8)

    capped = _side_exposure_capped_score(score, side, max_side_share=0.75)
    selected = capped.notna()

    assert int(selected.sum()) == 4
    assert int(((side < 0) & selected).sum()) == 3
    assert int(((side > 0) & selected).sum()) == 1
    assert np.isnan(float(capped.iloc[3]))


def test_score_quantile_side_rows_splits_by_side_and_score_bin() -> None:
    score = pd.Series([0.1, 0.2, 0.8, 0.9, 0.1, 0.2, 0.8, 0.9], dtype=np.float32)
    metrics = pd.DataFrame(
        {
            "side": [1, 1, 1, 1, -1, -1, -1, -1],
            "u_policy_net": [-0.02, -0.01, 0.01, 0.02, 0.02, 0.01, -0.01, -0.02],
            "mae_norm": [1.2, 1.1, 0.5, 0.4, 0.4, 0.5, 1.1, 1.2],
            "is_timeout": [0, 0, 0, 0, 0, 0, 1, 1],
        }
    )

    rows = pd.DataFrame(
        _score_quantile_side_rows(
            month="2026-06",
            label_arm="label",
            weight_arm="weight",
            selector="score",
            score=score,
            metrics=metrics,
            n_bins=2,
            min_bin_rows=1,
        )
    )

    long_rows = rows[rows["side_name"].eq("long")].sort_values("score_quantile")
    short_rows = rows[rows["side_name"].eq("short")].sort_values("score_quantile")
    assert float(long_rows.iloc[-1]["mean_u"]) > float(long_rows.iloc[0]["mean_u"])
    assert float(short_rows.iloc[-1]["mean_u"]) < float(short_rows.iloc[0]["mean_u"])
    assert float(long_rows.iloc[-1]["bad_mae_1r_rate"]) < float(long_rows.iloc[0]["bad_mae_1r_rate"])


def test_per_timestamp_top_mask_selects_top_n_per_timestamp() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                ["2026-06-01", "2026-06-01", "2026-06-01", "2026-06-02", "2026-06-02"]
            )
        }
    )
    score = pd.Series([0.1, 0.9, 0.8, 0.2, 0.7], dtype=np.float32)

    mask = _per_timestamp_top_mask(frame, score, top_n=2)

    assert mask.tolist() == [False, True, True, True, True]


def test_stage_gate_diagnostic_rows_reports_oracle_recall_by_stage() -> None:
    frame = pd.DataFrame({"__ts__": pd.to_datetime(["2026-06-01"] * 4)})
    metrics = pd.DataFrame(
        {
            "side": [1, -1, 1, -1],
            "u_policy_net": [0.4, 0.3, -0.1, -0.2],
            "mae_norm": [0.5, 1.2, 0.4, 1.4],
            "is_timeout": [0, 1, 0, 0],
        }
    )

    rows = pd.DataFrame(
        _stage_gate_diagnostic_rows(
            month="2026-06",
            label_arm="label",
            weight_arm="weight",
            top_frac=0.50,
            frame=frame,
            metrics=metrics,
            stage_masks={
                "all": np.asarray([True, True, True, True]),
                "miss_one_oracle": np.asarray([True, False, True, True]),
            },
        )
    )

    all_row = rows[(rows["stage"].eq("all")) & (rows["side_name"].eq("all"))].iloc[0]
    miss_row = rows[(rows["stage"].eq("miss_one_oracle")) & (rows["side_name"].eq("all"))].iloc[0]
    assert all_row["oracle_recall"] == 1.0
    assert miss_row["oracle_recall"] == 0.5


def test_stage_gate_diagnostics_separate_pre_risk_discovery_from_admission() -> None:
    frame = pd.DataFrame({"__ts__": pd.to_datetime(["2026-06-01"] * 5)})
    metrics = pd.DataFrame(
        {
            "side": [1, -1, 1, -1, 1],
            "u_policy_net": [0.5, 0.4, 0.1, -0.1, -0.2],
            "mae_norm": [0.4, 1.3, 0.5, 1.1, 0.6],
            "is_timeout": [0, 1, 0, 0, 0],
        }
    )

    rows = pd.DataFrame(
        _stage_gate_diagnostic_rows(
            month="2026-06",
            label_arm="label",
            weight_arm="weight",
            top_frac=0.40,
            frame=frame,
            metrics=metrics,
            stage_masks={
                "stageA_candidate_union_pre_risk": np.asarray([True, True, True, False, False]),
                "stageA_candidate_union": np.asarray([True, False, True, False, False]),
            },
        )
    )

    pre_risk = rows[
        rows["stage"].eq("stageA_candidate_union_pre_risk") & rows["side_name"].eq("all")
    ].iloc[0]
    admitted = rows[rows["stage"].eq("stageA_candidate_union") & rows["side_name"].eq("all")].iloc[0]
    assert pre_risk["oracle_recall"] == 1.0
    assert admitted["oracle_recall"] == 0.5


def test_oracle_enriched_ranker_relevance_penalizes_bad_path_top_utility() -> None:
    frame = pd.DataFrame({"__ts__": pd.to_datetime(["2026-06-01"] * 5)})
    metrics = pd.DataFrame(
        {
            "u_policy_net": [0.10, 0.09, 0.02, 0.01, -0.01],
            "mae_norm": [2.0, 0.4, 0.4, 0.4, 0.4],
            "is_timeout": [1, 0, 0, 0, 0],
        }
    )

    plain = _ranker_relevance(frame, metrics)
    enriched = _ranker_relevance(frame, metrics, mode="oracle_enriched")

    assert plain[0] >= plain[1]
    assert enriched[0] <= enriched[1]


def test_path_quality_ranker_relevance_strongly_prefers_clean_positive_path() -> None:
    frame = pd.DataFrame({"__ts__": pd.to_datetime(["2026-06-01"] * 6)})
    metrics = pd.DataFrame(
        {
            "u_policy_net": [0.12, 0.08, 0.04, 0.02, -0.01, -0.04],
            "mae_norm": [2.0, 0.3, 0.4, 0.5, 0.4, 0.3],
            "is_timeout": [1, 0, 0, 0, 0, 0],
        }
    )

    plain = _ranker_relevance(frame, metrics)
    path_quality = _ranker_relevance(frame, metrics, mode="path_quality")

    assert plain[0] >= plain[1]
    assert path_quality[1] > path_quality[0]


def test_calibration_viability_requires_utility_up_and_bad_mae_down() -> None:
    rows = []
    for period in ("2026-05", "2026-06"):
        for side_name in ("long", "short"):
            rows.extend(
                [
                    {
                        "period": period,
                        "label_arm": "label",
                        "weight_arm": "weight",
                        "selector": "selector",
                        "side_name": side_name,
                        "score_quantile": 1,
                        "score_quantiles": 2,
                        "rows": 20,
                        "mean_score": 0.1,
                        "mean_u": -0.001,
                        "hit_u": 0.4,
                        "q10_u": -0.01,
                        "bad_mae_1r_rate": 0.70,
                        "timeout_rate": 0.05,
                    },
                    {
                        "period": period,
                        "label_arm": "label",
                        "weight_arm": "weight",
                        "selector": "selector",
                        "side_name": side_name,
                        "score_quantile": 2,
                        "score_quantiles": 2,
                        "rows": 20,
                        "mean_score": 0.9,
                        "mean_u": 0.003,
                        "hit_u": 0.6,
                        "q10_u": -0.002,
                        "bad_mae_1r_rate": 0.30,
                        "timeout_rate": 0.04,
                    },
                ]
            )

    viability = _calibration_viability_rows(pd.DataFrame(rows))

    row = viability.iloc[0]
    assert row["calibration_group_count"] == 4
    assert bool(row["learnability_pass"])
    assert bool(row["bad_mae_calibration_pass"])
    assert float(row["utility_monotonic_rate"]) == 1.0
    assert float(row["bad_mae_improves_rate"]) == 1.0


def test_label_viability_matrix_identifies_final_recall_as_failed_gate_after_passing_risk() -> None:
    aggregate = pd.DataFrame(
        [
            {
                "arm": "label::weight::selector",
                "label_arm": "label",
                "weight_arm": "weight",
                "cluster_policy": "s7d_oracle_enriched_ranker_risk_cap_score",
                "top_frac": 0.03,
                "months": 3,
                "positive_months": 3,
                "mean_u": 0.004,
                "worst_month_mean_u": 0.001,
                "hit_u": 0.55,
                "q10_u": -0.001,
                "weekly_mean_u_q10": -0.001,
                "worst_weekly_mean_u": -0.002,
                "selected_week_count": 4,
                "positive_week_rate": 0.75,
                "selected_rows": 75,
                "selected_long_share": 0.45,
                "selected_short_share": 0.55,
                "no_trade_rate": 0.99,
                "score_finite_frac": 0.01,
                "score_ic_u": 0.1,
                "oracle_recall_at_model_top_k": 0.003,
                "oracle_precision_at_model_top_k": 0.01,
                "oracle_top_score_percentile_mean": 0.8,
                "oracle_top_score_percentile_q10": 0.6,
                "bad_mae_1r_rate": 0.40,
                "timeout_rate": 0.05,
            }
        ]
    )
    calibration = pd.DataFrame(
        [
            {
                "period": period,
                "label_arm": "label",
                "weight_arm": "weight",
                "selector": "S7d_oracle_enriched_ranker_risk_cap_score",
                "side_name": side_name,
                "score_quantile": quantile,
                "score_quantiles": 2,
                "rows": 50,
                "mean_score": float(quantile),
                "mean_u": -0.001 if quantile == 1 else 0.003,
                "hit_u": 0.45 if quantile == 1 else 0.60,
                "q10_u": -0.01 if quantile == 1 else -0.002,
                "bad_mae_1r_rate": 0.70 if quantile == 1 else 0.30,
                "timeout_rate": 0.05,
            }
            for period in ("2026-05", "2026-06")
            for side_name in ("long", "short")
            for quantile in (1, 2)
        ]
    )
    stage = pd.DataFrame(
        [
            {
                "period": period,
                "label_arm": "label",
                "weight_arm": "weight",
                "top_frac": 0.03,
                "stage": stage_name,
                "side_name": "all",
                "rows": 100,
                "row_share": 0.1,
                "oracle_rows": 50,
                "oracle_hit_rows": 30 if stage_name == "stageA_candidate_union" else 0,
                "oracle_recall": 0.60 if stage_name == "stageA_candidate_union" else 0.003,
                "mean_u": 0.002,
                "q10_u": -0.002,
                "bad_mae_1r_rate": 0.40,
                "timeout_rate": 0.05,
                "lower_tail_rate": 0.10,
            }
            for period in ("2026-05", "2026-06")
            for stage_name in ("stageA_candidate_union", "final_S7d")
        ]
    )

    matrix = _build_label_viability_matrix(
        aggregate=aggregate,
        calibration_diagnostics=calibration,
        stage_gate_diagnostics=stage,
        evaluation_utility_source="__u_econ_net__",
    )

    row = matrix.iloc[0]
    assert bool(row["candidate_discovery_pass"])
    assert bool(row["tail_risk_pass"])
    assert not bool(row["final_oracle_recall_pass"])
    assert row["first_failed_gate"] == "final_oracle_recall"
    assert not bool(row["active_label"])


def test_label_viability_matrix_requires_monthly_tail_risk_stability() -> None:
    aggregate = pd.DataFrame(
        [
            {
                "arm": "label::weight::selector",
                "label_arm": "label",
                "weight_arm": "weight",
                "cluster_policy": "s14_path_quality_risk_trim_score",
                "top_frac": 0.03,
                "months": 3,
                "positive_months": 3,
                "mean_u": 0.004,
                "worst_month_mean_u": 0.001,
                "hit_u": 0.55,
                "q10_u": -0.001,
                "weekly_mean_u_q10": -0.001,
                "worst_month_weekly_mean_u_q10": -0.002,
                "worst_weekly_mean_u": -0.002,
                "selected_week_count": 4,
                "positive_week_rate": 0.75,
                "selected_rows": 75,
                "selected_long_share": 0.45,
                "selected_short_share": 0.55,
                "no_trade_rate": 0.99,
                "score_finite_frac": 0.01,
                "score_ic_u": 0.1,
                "oracle_recall_at_model_top_k": 0.03,
                "oracle_precision_at_model_top_k": 0.05,
                "oracle_top_score_percentile_mean": 0.8,
                "oracle_top_score_percentile_q10": 0.6,
                "bad_mae_1r_rate": 0.49,
                "max_month_bad_mae_1r_rate": 0.65,
                "timeout_rate": 0.10,
                "max_month_timeout_rate": 0.12,
            }
        ]
    )
    calibration = pd.DataFrame(
        [
            {
                "period": period,
                "label_arm": "label",
                "weight_arm": "weight",
                "selector": "S12_path_quality_ranker_score",
                "side_name": side_name,
                "score_quantile": quantile,
                "score_quantiles": 2,
                "rows": 50,
                "mean_score": float(quantile),
                "mean_u": -0.001 if quantile == 1 else 0.003,
                "hit_u": 0.45 if quantile == 1 else 0.60,
                "q10_u": -0.01 if quantile == 1 else -0.002,
                "bad_mae_1r_rate": 0.70 if quantile == 1 else 0.30,
                "timeout_rate": 0.05,
            }
            for period in ("2026-05", "2026-06")
            for side_name in ("long", "short")
            for quantile in (1, 2)
        ]
    )
    stage = pd.DataFrame(
        [
            {
                "period": period,
                "label_arm": "label",
                "weight_arm": "weight",
                "top_frac": 0.03,
                "stage": stage_name,
                "side_name": "all",
                "rows": 100,
                "row_share": 0.1,
                "oracle_rows": 50,
                "oracle_hit_rows": 30,
                "oracle_recall": 0.60 if stage_name == "stageA_candidate_union" else 0.03,
                "mean_u": 0.002,
                "q10_u": -0.002,
                "bad_mae_1r_rate": 0.40,
                "timeout_rate": 0.05,
                "lower_tail_rate": 0.10,
            }
            for period in ("2026-05", "2026-06")
            for stage_name in ("stageA_candidate_union", "final_S14_path_quality_risk_trim")
        ]
    )

    matrix = _build_label_viability_matrix(
        aggregate=aggregate,
        calibration_diagnostics=calibration,
        stage_gate_diagnostics=stage,
        evaluation_utility_source="__u_econ_net__",
    )

    row = matrix.iloc[0]
    assert bool(row["tail_risk_pass"])
    assert not bool(row["monthly_tail_risk_pass"])
    assert row["first_failed_gate"] == "monthly_tail_risk"
    assert not bool(row["active_label"])


def test_train_meta_readiness_matrix_exports_only_active_candidates() -> None:
    viability = pd.DataFrame(
        [
            {
                "active_label": True,
                "label_arm": "label",
                "weight_arm": "weight",
                "cluster_policy": "s14_path_quality_risk_trim_score",
                "top_frac": 0.03,
                "calibration_selector": "S12_path_quality_ranker_score",
                "mean_u": 0.004,
                "worst_month_mean_u": 0.001,
                "bad_mae_1r_rate": 0.49,
                "max_month_bad_mae_1r_rate": 0.58,
                "timeout_rate": 0.10,
                "max_month_timeout_rate": 0.16,
                "final_stage_oracle_recall_mean": 0.023,
                "final_stage_oracle_recall_min": 0.006,
                "selected_rows": 545,
                "max_selected_side_share": 0.70,
                "label_viability_score": 100.0,
            },
            {
                "active_label": False,
                "label_arm": "label",
                "weight_arm": "weight",
                "cluster_policy": "bad_selector",
                "top_frac": 0.03,
            },
        ]
    )

    readiness = _build_train_meta_readiness_matrix(
        viability,
        labels_path=Path("labels"),
        feature_dir=Path("features"),
        feature_list_csv=Path("features.csv"),
        evaluation_utility_source="__u_econ_net__",
    )

    assert len(readiness) == 1
    row = readiness.iloc[0]
    assert row["readiness_status"] == "candidate_for_train_base_meta_smoke"
    assert row["cluster_policy"] == "s14_path_quality_risk_trim_score"
    assert not bool(row["is_final_promotion_ready"])
    assert "train_meta_oos_profitability" in row["required_next_checks"]


def test_s15_side_path_quality_policy_is_viability_mapped() -> None:
    assert (
        CALIBRATION_SELECTOR_BY_POLICY["s15_side_path_quality_risk_trim_score"]
        == "S15_side_path_quality_ranker_score"
    )
    assert FINAL_STAGE_BY_POLICY["s15_side_path_quality_risk_trim_score"] == (
        "final_S15_side_path_quality_risk_trim"
    )
    assert (
        CALIBRATION_SELECTOR_BY_POLICY["s16_discovery_path_quality_risk_trim_score"]
        == "S16_discovery_path_quality_blend_score"
    )
    assert FINAL_STAGE_BY_POLICY["s16_discovery_path_quality_risk_trim_score"] == (
        "final_S16_discovery_path_quality_risk_trim"
    )
