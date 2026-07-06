from pathlib import Path

import pandas as pd

from scripts.run_label_feature_store_model_smoke import (
    _clean_dirty_selected_diagnostics,
    _constrained_top_indices,
    _fit_clean_dirty_positive_risk_score,
    _fit_feature_gap_risk_score,
    _fit_lgbm_conditional_binary_prediction,
    _fit_lgbm_binary_risk_prediction,
    _fixed_artifact_targets,
    _oracle_recall_stats,
    _per_timestamp_top_mask,
    _ranker_relevance,
    _spearman,
    _timestamp_rank_percentile,
)
from scripts.run_gmm_train_base_learnability_smoke import (
    DEFAULT_THRESHOLDS,
    _expanded_target_candidate_rows,
    build_learnability_check,
)


def _candidate() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "label_arm": "OPTIMIZED_ECONOMIC_TARGET",
                "weight_arm": "W0_base",
                "cluster_policy": "s14_path_quality_risk_trim_score",
                "top_frac": 0.03,
                "evaluation_utility_source": "__u_econ_net__",
            }
        ]
    )


def _manifest() -> dict:
    features = [f"feature_{idx}" for idx in range(30)] + [
        "binned_return_entropy_24",
        "state_spectral_top3_reconstruction_error",
    ]
    return {
        "features": features,
        "feature_count": len(features),
        "feature_store": {
            "requested_features": 40,
            "retained_features": 32,
            "mean_feature_finite_frac": 0.98,
            "min_feature_finite_frac": 0.72,
        },
        "outputs": {"aggregate": "aggregate.csv"},
    }


def test_fast_spearman_matches_pandas_average_rank_with_ties_and_nans() -> None:
    x = pd.Series([1.0, 2.0, 2.0, None, 5.0, 5.0, 8.0, -1.0])
    y = pd.Series([3.0, 1.0, 1.0, 7.0, None, 4.0, 6.0, 2.0])
    mask = x.notna() & y.notna()
    expected = x[mask].rank(method="average").corr(y[mask].rank(method="average"))

    assert abs(_spearman(x, y) - float(expected)) < 1e-12


def _aggregate(**overrides: float) -> pd.DataFrame:
    row = {
        "label_arm": "OPTIMIZED_ECONOMIC_TARGET",
        "weight_arm": "W0_base",
        "top_frac": 0.03,
        "months": 3,
        "positive_months": 3,
        "mean_u": 0.0015,
        "worst_month_mean_u": 0.0002,
        "hit_u": 0.55,
        "q10_u": -0.004,
        "delta_mean_u_vs_period": 0.001,
        "score_ic_u": 0.02,
        "score_ic_label": 0.12,
        "decile_spearman_u": 0.45,
        "top_bottom_decile_spread_u": 0.003,
        "bad_mae_1r_rate": 0.42,
        "wide_barrier_25bps_rate": 0.02,
        "timeout_rate": 0.08,
        "selected_long_share": 0.48,
        "selected_short_share": 0.52,
        "max_selected_side_share": 0.52,
        "mean_selected_rows": 80,
        "min_selected_rows": 25,
        "selector_variant": "raw_utility",
        "arm": "OPTIMIZED_ECONOMIC_TARGET::W0_base",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_build_learnability_check_passes_candidate_with_model_and_risk_signal(tmp_path: Path) -> None:
    report = build_learnability_check(
        report_dir=tmp_path,
        smoke_manifest=_manifest(),
        aggregate=_aggregate(),
        candidates=_candidate(),
        thresholds=dict(DEFAULT_THRESHOLDS),
    )

    assert report["status"] == "pass"
    assert report["passed_next_check"] == "train_base_final_policy_readiness"
    assert report["best_passing_candidate"]["selector_variant"] == "raw_utility"
    candidate = report["candidate_checks"][0]
    assert candidate["status"] == "pass"
    assert candidate["feature_contract"]["gmm_context_feature_count"] == 2


def test_build_learnability_check_fails_bad_mae_and_nonmonotone_score(tmp_path: Path) -> None:
    report = build_learnability_check(
        report_dir=tmp_path,
        smoke_manifest=_manifest(),
        aggregate=_aggregate(
            score_ic_u=-0.01,
            top_bottom_decile_spread_u=-0.002,
            bad_mae_1r_rate=0.66,
            max_selected_side_share=1.0,
        ),
        candidates=_candidate(),
        thresholds=dict(DEFAULT_THRESHOLDS),
    )

    assert report["status"] == "fail"
    failed_metrics = {
        check["metric"] for check in report["candidate_checks"][0]["failed_checks"]
    }
    assert {
        "score_ic_u",
        "top_bottom_decile_spread_u",
        "bad_mae_1r_rate",
        "max_selected_side_share",
    } <= failed_metrics


def test_build_learnability_check_can_pass_on_repaired_variant(tmp_path: Path) -> None:
    aggregate = pd.concat(
        [
            _aggregate(
                selector_variant="raw_utility",
                arm="OPTIMIZED_ECONOMIC_TARGET::W0_base",
                mean_u=0.003,
                bad_mae_1r_rate=0.72,
                max_selected_side_share=1.0,
            ),
            _aggregate(
                selector_variant="bad_mae_timeout_penalty_side_cap_70",
                arm="OPTIMIZED_ECONOMIC_TARGET::W0_base::bad_mae_timeout_penalty_side_cap_70",
                mean_u=0.001,
                worst_month_mean_u=0.0001,
                bad_mae_1r_rate=0.44,
                max_selected_side_share=0.70,
            ),
        ],
        ignore_index=True,
    )

    report = build_learnability_check(
        report_dir=tmp_path,
        smoke_manifest=_manifest(),
        aggregate=aggregate,
        candidates=_candidate(),
        thresholds=dict(DEFAULT_THRESHOLDS),
    )

    assert report["status"] == "pass"
    assert report["passed_variant_count"] == 1
    assert (
        report["best_passing_candidate"]["selector_variant"]
        == "bad_mae_timeout_penalty_side_cap_70"
    )


def test_build_learnability_check_splits_candidate_readiness_from_final_policy(
    tmp_path: Path,
) -> None:
    report = build_learnability_check(
        report_dir=tmp_path,
        smoke_manifest=_manifest(),
        aggregate=_aggregate(
            selector_variant="s8_lgbm_utility_ranker_stageA_rerank_side_cap_70",
            arm=(
                "OPTIMIZED_ECONOMIC_TARGET::W0_base::"
                "s8_lgbm_utility_ranker_stageA_rerank_side_cap_70"
            ),
            mean_u=0.0030,
            worst_month_mean_u=0.0020,
            positive_months=3,
            bad_mae_1r_rate=0.60,
            timeout_rate=0.08,
            stageA_candidate_oracle_recall=0.65,
            final_oracle_recall=0.045,
            hard_risk_cap_no_trade_rate=0.20,
            clean_positive_rate=0.20,
            dirty_positive_rate=0.30,
        ),
        candidates=_candidate(),
        thresholds=dict(DEFAULT_THRESHOLDS),
    )

    assert report["status"] == "candidate_for_train_meta_path_filter_smoke"
    assert report["gate_1a_train_base_candidate_readiness"]["status"] == "pass"
    assert report["gate_1b_train_base_final_policy_readiness"]["status"] == "fail"
    assert report["best_candidate_readiness"]["candidate_readiness_status"] == "pass"
    assert report["meta_filter_handoff"]["status"] == "ready"


def test_constrained_top_indices_enforces_side_cap_after_no_trade_trim() -> None:
    score = pd.Series([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    side = pd.Series([1, 1, 1, 1, -1, -1])
    eligible = pd.Series([True, True, True, True, True, False])

    selected, diag = _constrained_top_indices(
        score=score,
        side=side,
        eligible=eligible,
        top_frac=1.0,
        max_side_share=0.70,
    )

    selected_side = side.iloc[selected]
    assert len(selected) < 5
    assert selected_side.value_counts(normalize=True).max() <= 0.70
    assert diag["hard_risk_cap_no_trade_rate"] > 0.0


def test_constrained_top_indices_all_one_side_keeps_available_rows() -> None:
    score = pd.Series([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    side = pd.Series([1, 1, 1, 1, 1, 1])
    eligible = pd.Series([True, True, True, True, True, True])

    selected, diag = _constrained_top_indices(
        score=score,
        side=side,
        eligible=eligible,
        top_frac=0.50,
        max_side_share=0.70,
    )

    assert len(selected) > 0
    assert diag["hard_risk_cap_selected_rows"] == len(selected)
    assert diag["hard_risk_cap_no_trade_rate"] < 1.0


def test_fixed_artifact_targets_include_path_safe_economic_variants() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                [
                    "2026-04-01",
                    "2026-04-01",
                    "2026-04-02",
                    "2026-04-02",
                    "2026-04-02",
                ]
            ),
            "__y_econ_soft__": [0.9, 0.8, 0.2, 0.7, 0.75],
            "__y_econ_bin__": [1.0, 1.0, 0.0, 1.0, 1.0],
            "__u_econ_net__": [0.01, 0.008, -0.005, 0.006, 0.007],
        }
    )
    metrics = pd.DataFrame(
        {
            "u_policy_net": [0.01, 0.008, -0.005, 0.006, 0.007],
            "mae_norm": [0.30, 1.40, 0.20, 0.40, 0.85],
            "mfe_norm": [2.0, 2.0, 0.1, 1.2, 0.9],
            "is_timeout": [False, False, True, True, False],
        }
    )

    targets = _fixed_artifact_targets(frame, metrics)

    assert "OPTIMIZED_ECONOMIC_PATH_SAFE_TARGET" in targets
    assert "OPTIMIZED_ECONOMIC_BAD_MAE_CONTRAST_TARGET" in targets
    assert "OPTIMIZED_ECONOMIC_CLEAN_RANK_TARGET" in targets
    assert "OPTIMIZED_ECONOMIC_TIMEOUT_SAFE_TARGET" in targets
    assert "OPTIMIZED_ECONOMIC_STRICT_PATH_FIRST_TARGET" in targets
    assert "OPTIMIZED_ECONOMIC_CLEAN_UTILITY_RANK_TARGET" in targets
    assert "OPTIMIZED_ECONOMIC_PATH_FIRST_CLEAN_RELEVANCE_TARGET" in targets
    assert "OPTIMIZED_ECONOMIC_S24_BROAD_PATH_FIRST_SOURCE_TARGET" in targets
    assert "OPTIMIZED_ECONOMIC_TIMEOUT_AWARE_CLEAN_SOURCE_TARGET" in targets
    assert "OPTIMIZED_ECONOMIC_EXEC_MARGIN_STABLE_TARGET" in targets
    assert (
        targets["OPTIMIZED_ECONOMIC_PATH_SAFE_TARGET"]["target_soft"].iloc[0]
        > targets["OPTIMIZED_ECONOMIC_PATH_SAFE_TARGET"]["target_soft"].iloc[1]
    )
    assert (
        targets["OPTIMIZED_ECONOMIC_TIMEOUT_SAFE_TARGET"]["target_soft"].iloc[0]
        > targets["OPTIMIZED_ECONOMIC_TIMEOUT_SAFE_TARGET"]["target_soft"].iloc[2]
    )
    assert (
        targets["OPTIMIZED_ECONOMIC_STRICT_PATH_FIRST_TARGET"]["target_soft"].iloc[0]
        > targets["OPTIMIZED_ECONOMIC_STRICT_PATH_FIRST_TARGET"]["target_soft"].iloc[1]
    )
    assert (
        targets["OPTIMIZED_ECONOMIC_CLEAN_UTILITY_RANK_TARGET"]["target_soft"].iloc[0]
        > targets["OPTIMIZED_ECONOMIC_CLEAN_UTILITY_RANK_TARGET"]["target_soft"].iloc[1]
    )
    assert (
        targets["OPTIMIZED_ECONOMIC_CLEAN_UTILITY_RANK_TARGET"]["target_soft"].iloc[1]
        < 0.10
    )
    assert (
        targets["OPTIMIZED_ECONOMIC_PATH_FIRST_CLEAN_RELEVANCE_TARGET"]["target_soft"].iloc[0]
        > targets["OPTIMIZED_ECONOMIC_PATH_FIRST_CLEAN_RELEVANCE_TARGET"]["target_soft"].iloc[1]
    )
    assert (
        targets["OPTIMIZED_ECONOMIC_PATH_FIRST_CLEAN_RELEVANCE_TARGET"]["target_soft"].iloc[1]
        > targets["OPTIMIZED_ECONOMIC_PATH_FIRST_CLEAN_RELEVANCE_TARGET"]["target_soft"].iloc[2]
    )
    assert (
        targets["OPTIMIZED_ECONOMIC_S24_BROAD_PATH_FIRST_SOURCE_TARGET"]["target_soft"].iloc[0]
        > targets["OPTIMIZED_ECONOMIC_S24_BROAD_PATH_FIRST_SOURCE_TARGET"]["target_soft"].iloc[1]
    )
    assert (
        targets["OPTIMIZED_ECONOMIC_S24_BROAD_PATH_FIRST_SOURCE_TARGET"]["target_soft"].iloc[1]
        > targets["OPTIMIZED_ECONOMIC_S24_BROAD_PATH_FIRST_SOURCE_TARGET"]["target_soft"].iloc[2]
    )
    timeout_aware = targets["OPTIMIZED_ECONOMIC_TIMEOUT_AWARE_CLEAN_SOURCE_TARGET"]
    assert timeout_aware["target_soft"].iloc[0] > timeout_aware["target_soft"].iloc[4]
    assert timeout_aware["target_soft"].iloc[4] > timeout_aware["target_soft"].iloc[3]
    assert timeout_aware["target_soft"].iloc[3] > timeout_aware["target_soft"].iloc[1]
    assert timeout_aware["target_hard"].iloc[0] == 1.0
    assert timeout_aware["target_hard"].iloc[3] == 0.0
    exec_margin_stable = targets["OPTIMIZED_ECONOMIC_EXEC_MARGIN_STABLE_TARGET"]
    assert exec_margin_stable["target_soft"].iloc[0] > exec_margin_stable["target_soft"].iloc[1]
    assert exec_margin_stable["target_soft"].iloc[0] > exec_margin_stable["target_soft"].iloc[3]
    assert exec_margin_stable["target_soft"].iloc[1] <= 0.03
    assert exec_margin_stable["target_hard"].iloc[0] == 1.0
    assert exec_margin_stable["target_hard"].iloc[1] == 0.0


def test_target_candidate_expansion_preserves_source_label() -> None:
    candidates = _candidate()

    expanded = _expanded_target_candidate_rows(
        candidates,
        target_label_arms=["OPTIMIZED_ECONOMIC_PATH_SAFE_TARGET"],
    )

    assert expanded["label_arm"].tolist() == [
        "OPTIMIZED_ECONOMIC_TARGET",
        "OPTIMIZED_ECONOMIC_PATH_SAFE_TARGET",
    ]
    assert expanded["source_label_arm"].nunique() == 1
    assert expanded["is_target_variant"].tolist() == [False, True]


def test_feature_gap_risk_score_uses_risky_train_direction() -> None:
    x_train = pd.DataFrame(
        {
            "risky_high": [0.1] * 160 + [1.0] * 160,
            "noise": ([0.1, 0.2, 0.3, 0.4] * 80),
        }
    )
    train_metrics = pd.DataFrame(
        {
            "u_policy_net": [0.01] * 160 + [-0.01] * 160,
            "mae_norm": [0.2] * 160 + [1.4] * 160,
            "is_timeout": [False] * 320,
        }
    )
    x_valid = pd.DataFrame(
        {
            "risky_high": [0.0, 0.5, 1.0],
            "noise": [0.3, 0.3, 0.3],
        }
    )

    risk, diag = _fit_feature_gap_risk_score(
        x_train=x_train,
        x_valid=x_valid,
        train_metrics=train_metrics,
        top_k=1,
    )

    assert diag["feature_gap_risk_features"] == "risky_high"
    assert risk.iloc[2] > risk.iloc[1] > risk.iloc[0]


def test_clean_dirty_positive_risk_score_uses_positive_path_contrast() -> None:
    x_train = pd.DataFrame(
        {
            "dirty_positive_high": [0.1] * 120 + [1.0] * 120 + [0.2] * 120,
            "generic_loser_high": [0.2] * 120 + [0.2] * 120 + [1.0] * 120,
        }
    )
    train_metrics = pd.DataFrame(
        {
            "u_policy_net": [0.01] * 120 + [0.02] * 120 + [-0.01] * 120,
            "mae_norm": [0.2] * 120 + [1.4] * 120 + [1.8] * 120,
            "is_timeout": [False] * 360,
        }
    )
    x_valid = pd.DataFrame(
        {
            "dirty_positive_high": [0.0, 0.5, 1.0],
            "generic_loser_high": [1.0, 0.5, 0.0],
        }
    )

    risk, diag = _fit_clean_dirty_positive_risk_score(
        x_train=x_train,
        x_valid=x_valid,
        train_metrics=train_metrics,
        top_k=1,
    )

    assert diag["clean_dirty_positive_risk_features"] == "dirty_positive_high"
    assert diag["clean_dirty_positive_train_clean_rows"] == 120
    assert diag["clean_dirty_positive_train_dirty_rows"] == 120
    assert risk.iloc[2] > risk.iloc[1] > risk.iloc[0]


def test_lgbm_binary_risk_prediction_single_class_fallback() -> None:
    x_train = pd.DataFrame({"feature": [0.1, 0.2, 0.3, 0.4]})
    x_valid = pd.DataFrame({"feature": [0.5, 0.6]})
    pred, status = _fit_lgbm_binary_risk_prediction(
        x_train=x_train,
        y_train=pd.Series([1.0, 1.0, 1.0, 1.0]),
        x_valid=x_valid,
        seeds=[17],
    )

    assert status == "single_class"
    assert pred.tolist() == [1.0, 1.0]


def test_lgbm_conditional_binary_prediction_insufficient_rows_fallback() -> None:
    x_train = pd.DataFrame({"feature": [0.1, 0.2, 0.3, 0.4]})
    x_valid = pd.DataFrame({"feature": [0.5, 0.6]})
    pred, status = _fit_lgbm_conditional_binary_prediction(
        x_train=x_train,
        y_train=pd.Series([0.0, 1.0, 0.0, 1.0]),
        train_mask=pd.Series([True, False, True, False]),
        x_valid=x_valid,
        seeds=[17],
        min_train_rows=3,
    )

    assert status == "insufficient_conditional_rows"
    assert pd.isna(pred).all()


def test_lgbm_conditional_binary_prediction_uses_positional_mask_with_sparse_index() -> None:
    x_train = pd.DataFrame(
        {"feature": [0.1, 0.2, 0.3, 0.4, 1.1, 1.2]},
        index=[10, 11, 12, 13, 14, 15],
    )
    x_valid = pd.DataFrame({"feature": [0.15, 1.15]})
    pred, status = _fit_lgbm_conditional_binary_prediction(
        x_train=x_train,
        y_train=pd.Series([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], index=x_train.index),
        train_mask=pd.Series([True, True, True, True, False, False], index=x_train.index),
        x_valid=x_valid,
        seeds=[17],
        sample_weight=pd.Series([1.0] * 6, index=x_train.index),
        min_train_rows=4,
    )

    assert status in {"conditional_ok", "conditional_lightgbm_unavailable"}
    assert len(pred) == len(x_valid)


def test_per_timestamp_top_mask_and_oracle_recall_stats_capture_stage_a_discovery() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                ["2026-06-01", "2026-06-01", "2026-06-01", "2026-06-02", "2026-06-02"]
            )
        }
    )
    score = pd.Series([0.1, 0.9, 0.8, 0.7, 0.2], dtype="float32")
    metrics = pd.DataFrame(
        {
            "side": [1, -1, 1, -1, 1],
            "u_policy_net": [0.01, 0.20, -0.01, 0.30, -0.02],
            "mae_norm": [0.4, 0.6, 1.2, 0.7, 1.5],
            "is_timeout": [0, 0, 1, 0, 1],
        }
    )

    mask = _per_timestamp_top_mask(frame, score, top_n=1)
    stats = _oracle_recall_stats(
        metrics=metrics,
        mask=mask,
        top_frac=0.40,
        prefix="stageA_candidate",
    )

    assert mask.tolist() == [False, True, False, True, False]
    assert stats["stageA_candidate_oracle_recall"] == 1.0
    assert stats["stageA_candidate_short_oracle_recall"] == 1.0


def test_timestamp_rank_percentile_is_group_local_and_directional() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                ["2026-06-01", "2026-06-01", "2026-06-01", "2026-06-02", "2026-06-02"]
            )
        }
    )
    values = pd.Series([0.4, 0.1, 0.9, 0.2, 0.8], dtype="float32")

    high_good = _timestamp_rank_percentile(frame, values, ascending=True)
    low_good = _timestamp_rank_percentile(frame, values, ascending=False)

    assert high_good.iloc[2] == 1.0
    assert high_good.iloc[1] < high_good.iloc[0] < high_good.iloc[2]
    assert low_good.iloc[1] == 1.0
    assert low_good.iloc[3] > low_good.iloc[4]


def test_learnability_check_blocks_s7_low_final_oracle_recall(tmp_path: Path) -> None:
    report = build_learnability_check(
        report_dir=tmp_path,
        smoke_manifest=_manifest(),
        aggregate=_aggregate(
            selector_variant="s7_two_stage_candidate_rerank_side_cap_70",
            arm="OPTIMIZED_ECONOMIC_TARGET::W0_base::s7_two_stage_candidate_rerank_side_cap_70",
            stageA_candidate_oracle_recall=0.80,
            final_oracle_recall=0.005,
            hard_risk_cap_no_trade_rate=0.10,
        ),
        candidates=_candidate(),
        thresholds=dict(DEFAULT_THRESHOLDS),
    )

    assert report["status"] == "fail"
    failed_metrics = {
        check["metric"] for check in report["candidate_checks"][0]["failed_checks"]
    }
    assert "final_oracle_recall" in failed_metrics


def test_ranker_relevance_path_quality_penalizes_bad_mae_top_utility() -> None:
    frame = pd.DataFrame({"__ts__": pd.to_datetime(["2026-06-01"] * 5)})
    metrics = pd.DataFrame(
        {
            "u_policy_net": [0.10, 0.08, 0.02, -0.01, -0.02],
            "mae_norm": [1.8, 0.3, 0.4, 0.3, 0.2],
            "is_timeout": [0, 0, 0, 0, 0],
        }
    )
    target = pd.DataFrame(
        {
            "target_soft": [0.9, 0.8, 0.4, 0.1, 0.0],
            "target_hard": [1, 1, 0, 0, 0],
        }
    )

    utility = _ranker_relevance(frame, metrics, target, mode="utility_quintile")
    path_quality = _ranker_relevance(frame, metrics, target, mode="path_quality")

    assert utility[0] >= utility[1]
    assert path_quality[1] > path_quality[0]


def test_ranker_relevance_clean_oracle_prefers_clean_profitable_path() -> None:
    frame = pd.DataFrame({"__ts__": pd.to_datetime(["2026-06-01"] * 6)})
    metrics = pd.DataFrame(
        {
            "u_policy_net": [0.12, 0.09, 0.07, 0.02, -0.01, -0.02],
            "mae_norm": [1.7, 0.3, 0.4, 0.5, 0.2, 0.3],
            "is_timeout": [0, 0, 1, 0, 0, 0],
        }
    )
    target = pd.DataFrame(
        {
            "target_soft": [0.9, 0.8, 0.7, 0.2, 0.1, 0.0],
            "target_hard": [1, 1, 1, 0, 0, 0],
        }
    )

    utility = _ranker_relevance(frame, metrics, target, mode="utility_quintile")
    clean_oracle = _ranker_relevance(frame, metrics, target, mode="clean_oracle")

    assert utility[0] >= utility[1]
    assert clean_oracle[1] > clean_oracle[0]
    assert clean_oracle[1] > clean_oracle[2]


def test_ranker_relevance_path_first_clean_demotes_dirty_positive() -> None:
    frame = pd.DataFrame({"__ts__": pd.to_datetime(["2026-06-01"] * 6)})
    metrics = pd.DataFrame(
        {
            "side": [1, 1, 1, 1, -1, -1],
            "u_policy_net": [0.12, 0.08, 0.03, 0.01, 0.07, -0.01],
            "mae_norm": [1.7, 0.3, 0.4, 0.5, 0.3, 0.2],
            "is_timeout": [0, 0, 1, 0, 0, 0],
        }
    )
    target = pd.DataFrame(
        {
            "target_soft": [0.9, 0.8, 0.7, 0.3, 0.6, 0.0],
            "target_hard": [1, 1, 1, 0, 1, 0],
        }
    )

    path_first = _ranker_relevance(frame, metrics, target, mode="path_first_clean")
    strict = _ranker_relevance(frame, metrics, target, mode="path_first_clean_dirty_zero")

    assert path_first[1] > path_first[0]
    assert path_first[1] > path_first[2]
    assert path_first[0] == 1
    assert strict[0] == 0
    assert strict[1] > strict[0]


def test_ranker_relevance_s24_broad_path_first_preserves_dirty_weak_positive() -> None:
    frame = pd.DataFrame({"__ts__": pd.to_datetime(["2026-06-01"] * 6)})
    metrics = pd.DataFrame(
        {
            "side": [1, 1, 1, 1, -1, -1],
            "u_policy_net": [0.12, 0.08, 0.03, 0.01, 0.07, -0.01],
            "mae_norm": [1.7, 0.3, 0.4, 0.5, 0.3, 0.2],
            "is_timeout": [0, 0, 1, 0, 0, 0],
        }
    )
    target = pd.DataFrame(
        {
            "target_soft": [0.2, 0.9, 0.1, 0.5, 0.8, 0.0],
            "target_hard": [0, 1, 0, 1, 1, 0],
        }
    )

    broad = _ranker_relevance(
        frame,
        metrics,
        target,
        mode="s24_broad_path_first_source",
    )
    strict = _ranker_relevance(
        frame,
        metrics,
        target,
        mode="s24_broad_path_first_dirty_zero",
    )

    assert broad[0] == 1
    assert broad[1] > broad[0]
    assert broad[3] >= 2
    assert strict[0] == 0
    assert strict[1] > strict[0]


def test_clean_dirty_selected_diagnostics_reports_score_gap_by_side() -> None:
    metrics = pd.DataFrame(
        {
            "side": [1, 1, -1, -1],
            "u_policy_net": [0.04, 0.03, 0.05, -0.01],
            "mae_norm": [0.4, 1.3, 0.5, 0.2],
            "is_timeout": [0, 0, 0, 0],
        }
    )
    score = pd.Series([0.9, 0.4, 0.8, 0.1], dtype="float32")

    rows = _clean_dirty_selected_diagnostics(
        metrics=metrics,
        score=score,
        selector="s19_path_first_ranker_stageA_rerank_side_cap_70",
        month="2026-06",
        top_frac=0.75,
        selected_idx=pd.Series([0, 1, 2]).to_numpy(),
        base_fields={"label_arm": "label", "weight_arm": "weight"},
    )

    all_row = next(row for row in rows if row["side"] == "all")
    assert all_row["clean_positive_rate"] > all_row["dirty_positive_rate"]
    assert all_row["score_gap_clean_minus_dirty"] > 0.0
    assert all_row["dirty_positive_share_of_positive_u"] > 0.0
