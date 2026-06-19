from pathlib import Path

import numpy as np
import pandas as pd

from extreme_price_movements.config import CFG
from extreme_price_movements.unsupervised_regime_learning.diagnostics import (
    compute_quality_report,
    select_primitive_features,
    select_representatives_by_spearman,
    stratified_period_sample_positions,
)
from extreme_price_movements.unsupervised_regime_learning.feature_registry import (
    BINARY_PRIMITIVE_FEATURES,
    UNSUPERVISED_REGIME_PRIMITIVE_FEATURES,
)
from extreme_price_movements.unsupervised_regime_learning.operators import (
    fit_transform_svd_knn_features,
    generate_autocorr_operator_features,
    generate_eigenvalue_summary_features,
    generate_pair_operator_features,
    generate_quantile_operator_features,
    score_pair_candidates,
)
from extreme_price_movements.unsupervised_regime_learning.pipeline import (
    build_operator_feature_frame,
    fit_unsupervised_regime_learning_features,
)
from extreme_price_movements.unsupervised_regime_learning.regime_hpo import (
    DEFAULT_REGIME_HPO_SEARCH_SPACE,
    RegimeHPOConfig,
    _feature_conditional_learnability,
    run_advanced_regime_learning_hpo,
)
from extreme_price_movements.unsupervised_regime_learning.regime_models import (
    ADVANCED_REGIME_LEARNING_SCHEMA_VERSION,
    AdvancedRegimeLearningArtifact,
    AdvancedRegimeLearningConfig,
    _cv_auc_trend_vol,
    _transition_duration_arrays,
    _trend_vol_matrix,
    augment_frame_with_regime_artifact,
    fit_advanced_regime_learning,
    load_advanced_regime_learning_artifact,
    minimum_duration_smooth_by_frame,
    save_advanced_regime_learning_artifact,
)


def _sample_frame(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    ts = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    rows = []
    for symbol_i, symbol in enumerate(["BTC", "ETH"]):
        t = np.arange(n, dtype=np.float64)
        rows.append(
            pd.DataFrame(
                {
                    "timestamp": ts,
                    "symbol": symbol,
                    "trend": np.sin(t / 8.0) + symbol_i * 0.05,
                    "vol": np.cos(t / 9.0) + 0.02 * t + symbol_i * 0.03,
                    "flow": rng.normal(0.0, 0.5, n).cumsum(),
                    "bad": 0.0,
                }
            )
        )
    frame = pd.concat(rows, ignore_index=True)
    frame.loc[frame["symbol"].eq("BTC").head(4).index, "trend"] = 0.0
    return frame


def test_unsupervised_regime_learning_config_wires_primitives():
    cfg = CFG["UNSUPERVISED_REGIME_LEARNING"]

    assert "primitive_feature_keys" in cfg
    assert cfg["primitive_feature_keys"] == UNSUPERVISED_REGIME_PRIMITIVE_FEATURES
    assert not set(BINARY_PRIMITIVE_FEATURES).intersection(cfg["primitive_feature_keys"])
    assert cfg["excluded_primitive_feature_keys"] == BINARY_PRIMITIVE_FEATURES
    assert "funding_mom_w" in cfg["primitive_feature_keys"]
    assert "asset_minus_mkt_oi_1d_cp_z_8_32_96" in cfg["primitive_feature_keys"]
    assert len(cfg["primitive_feature_keys"]) == len(set(cfg["primitive_feature_keys"]))
    assert cfg["regime_models"]["stability_bootstraps"] >= 1
    assert cfg["regime_models"]["selector_backend"] in {"lgbm", "random_forest"}
    assert cfg["regime_models"]["bayesian_gmm_covariance_type"] == "diag"
    assert "hdbscan_min_cluster_size_fraction" in cfg["regime_models"]
    assert "hmm_transmat_self_bias" in cfg["regime_models"]
    assert "spectral_n_neighbors" in cfg["regime_models"]
    assert "kmeans_n_init" in cfg["regime_models"]
    assert "lgbm_feature_fraction" in cfg["regime_models"]
    assert DEFAULT_REGIME_HPO_SEARCH_SPACE["bayesian_gmm_covariance_type"] == ["diag"]
    for key in [
        "bayesian_gmm_reg_covar",
        "hdbscan_min_samples",
        "hmm_covariance_type",
        "spectral_affinity",
        "kmeans_algorithm",
        "lgbm_lambda_l2",
    ]:
        assert key in DEFAULT_REGIME_HPO_SEARCH_SPACE


def test_quality_filter_uses_per_symbol_warmup_and_zero_as_bad():
    frame = _sample_frame()
    report = compute_quality_report(
        frame,
        ["trend", "vol", "bad"],
        warmup_rows=4,
        min_good_row_fraction=0.90,
    )

    assert bool(report.loc["trend", "keep"])
    assert bool(report.loc["vol", "keep"])
    assert not bool(report.loc["bad", "keep"])
    assert report.loc["bad", "low_quality_fraction"] == 1.0


def test_primitive_selection_scores_and_prunes_features():
    frame = _sample_frame()
    result = select_primitive_features(
        frame,
        ["trend", "vol", "flow", "bad"],
        target_features=2,
        warmup_rows=4,
        min_good_row_fraction=0.90,
        block_hours=24,
        min_block_rows=8,
    )

    assert len(result.selected_features) == 2
    assert "bad" not in result.selected_features
    assert set(result.selected_features).issubset({"trend", "vol", "flow"})
    assert "dynamics_score" in result.diagnostics.columns


def test_stratified_period_sample_positions_spans_time_and_symbols():
    frame = _sample_frame(n=30)
    positions = np.arange(len(frame), dtype=int)

    sampled = stratified_period_sample_positions(
        frame,
        positions,
        max_rows=12,
        n_periods=3,
    )

    assert len(sampled) <= 12
    assert frame.iloc[sampled]["timestamp"].min() == frame["timestamp"].min()
    assert frame.iloc[sampled]["timestamp"].max() == frame["timestamp"].max()
    assert set(frame.iloc[sampled]["symbol"]) == {"BTC", "ETH"}


def test_regime_minimum_duration_smoothing_respects_symbol_boundaries():
    frame = pd.DataFrame(
        {
            "timestamp": list(pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")) * 2,
            "symbol": ["BTC"] * 3 + ["ETH"] * 3,
        }
    )
    labels = np.asarray([1, 1, 0, 0, 0, 0], dtype=np.int64)

    smoothed = minimum_duration_smooth_by_frame(
        labels,
        frame,
        min_duration=2,
        timestamp_col="timestamp",
        symbol_col="symbol",
    )

    assert smoothed.tolist() == [1, 1, 1, 0, 0, 0]


def test_regime_transition_duration_keeps_noise_separate_from_regime_zero():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC"),
            "symbol": ["BTC"] * 3,
        }
    )
    labels = np.asarray([-1, 0, 0], dtype=np.int64)
    prob_max = np.ones(3, dtype=np.float32)

    _hazard, time_since, expected_duration = _transition_duration_arrays(
        frame,
        labels,
        prob_max,
        timestamp_col="timestamp",
        symbol_col="symbol",
    )

    assert time_since.tolist() == [0.0, 0.0, 1.0]
    assert expected_duration.tolist() == [1.0, 1.0, 1.0]


def test_nontriviality_trend_vol_proxy_does_not_fallback_to_arbitrary_features():
    matrix = np.arange(12, dtype=np.float32).reshape(6, 2)
    trend_vol = _trend_vol_matrix(matrix, ["flow", "liquidity_ratio"])
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    frame = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=6, freq="h", tz="UTC")})

    assert trend_vol.shape == (6, 0)
    assert _cv_auc_trend_vol(
        trend_vol,
        labels,
        blocks=[np.arange(0, 3, dtype=np.int64), np.arange(3, 6, dtype=np.int64)],
        random_state=1,
        max_features=96,
        max_rows=100,
    ) == 0.5


def test_spearman_pruning_uses_sampled_candidate_cap():
    frame = _sample_frame(n=40)
    scores = {"trend": 3.0, "vol": 2.0, "flow": 1.0}

    selected, threshold = select_representatives_by_spearman(
        frame,
        ["trend", "vol", "flow"],
        scores,
        target_features=2,
        max_candidates=2,
        max_corr_rows=12,
        corr_time_bins=3,
    )

    assert selected == ["trend", "vol"]
    assert threshold >= 0.96


def test_operator_generation_smoke():
    frame = _sample_frame()
    features = ["trend", "vol", "flow"]

    quantile = generate_quantile_operator_features(
        frame, features, window=12, min_periods=4
    )
    autocorr = generate_autocorr_operator_features(
        frame, features, window=12, min_periods=5
    )
    pair_scores = score_pair_candidates(
        frame,
        features,
        mechanisms={"trend": "trend", "vol": "volatility", "flow": "liquidity"},
        rolling_window=12,
        min_periods=5,
    )
    svd_knn, state = fit_transform_svd_knn_features(
        frame,
        features,
        svd_components=[2, 16],
        knn_svd_components=16,
        knn_neighbors=3,
    )

    assert "q_iqr__trend" in quantile.columns
    assert "autocorr_lag1_w12__trend" in autocorr.columns
    assert not pair_scores.empty
    assert {"svd2_00", "svd2_01", "svd16_knn_density"}.issubset(svd_knn.columns)
    assert state["enabled"] is True


def test_svd_knn_reference_caps_use_stratified_sampling():
    frame = _sample_frame(n=40)

    _features, state = fit_transform_svd_knn_features(
        frame,
        ["trend", "vol", "flow"],
        svd_components=[2, 16],
        knn_svd_components=16,
        knn_neighbors=3,
        max_reference_rows=20,
        knn_max_reference_rows=8,
        sample_time_bins=4,
    )

    assert state["original_reference_rows"] == len(frame)
    assert state["sampled_reference_rows"] <= 20
    assert state["knn"]["reference_rows"] <= 8


def test_advanced_regime_learning_builds_artifacts_and_adapter_features():
    frame = _sample_frame(n=36)
    config = AdvancedRegimeLearningConfig(
        selector_backend="random_forest",
        stability_bootstraps=2,
        stability_top_m=3,
        n_estimators=8,
        max_classifier_rows=200,
        n_regimes=3,
        mfa_regimes=3,
        mfa_factors=2,
        mfa_max_iter=3,
        ae_epochs=1,
        ae_latent_dim=3,
        ae_hidden_dim=8,
        ae_batch_size=64,
        min_regime_duration=2,
        max_rows=200,
    )

    artifact = fit_advanced_regime_learning(
        frame,
        ["trend", "vol", "flow"],
        config=config,
    )

    assert artifact.schema_version == "unsupervised_regime_learning_v2"
    assert not artifact.stability_frequencies.empty
    assert not artifact.real_vs_null_importances.empty
    assert len(artifact.mfa_responsibilities) == len(frame)
    assert np.allclose(
        artifact.mfa_responsibilities.sum(axis=1).to_numpy(dtype=float),
        1.0,
        atol=1e-4,
    )
    assert not artifact.regime_labels.empty
    assert not artifact.regime_transition_features.empty
    assert any(col.endswith("_regime_prob_entropy") for col in artifact.regime_transition_features.columns)
    assert any(col.endswith("_regime_prob_max") for col in artifact.regime_transition_features.columns)
    assert any(col.endswith("_regime_prob_change_1h") for col in artifact.regime_transition_features.columns)
    assert any(col.endswith("_regime_prob_change_4h") for col in artifact.regime_transition_features.columns)
    assert any(col.endswith("_regime_prob_change_24h") for col in artifact.regime_transition_features.columns)
    assert any(col.endswith("_regime_transition_hazard") for col in artifact.regime_transition_features.columns)
    assert any(col.endswith("_time_since_regime_change") for col in artifact.regime_transition_features.columns)
    assert any(col.endswith("_expected_regime_duration") for col in artifact.regime_transition_features.columns)
    assert not artifact.regime_feature_importance.empty
    assert {"method", "regime", "feature", "importance", "signed_shift"}.issubset(
        artifact.regime_feature_importance.columns
    )
    assert not artifact.regime_tradability_diagnostics.empty
    assert {
        "method",
        "regime",
        "support_fraction",
        "symbol_count",
        "volatility_stress_score",
        "tradability_score",
    }.issubset(artifact.regime_tradability_diagnostics.columns)
    assert not artifact.ae_feature_gates.empty
    assert artifact.method_embeddings
    assert "raw_selected_kmeans" in set(artifact.regime_diagnostics["method"])
    assert "raw_spectral_spectral" in set(artifact.regime_diagnostics["method"])
    spectral_diag = artifact.regime_diagnostics.set_index("method").loc["raw_spectral_spectral"]
    assert spectral_diag["assessment_cluster_method"] == "spectral"
    assessment_cols = {
        "AUC_tv",
        "AUC_non_trend_vol",
        "AUC_all_structure",
        "Incremental_AUC_over_trend_vol",
        "IncrementalNonTriviality",
        "NonTrendVolSignal",
        "TrendVolReplicaPenalty",
        "NonTriviality",
        "OOS_Stability",
        "Dwell_Quality",
        "Transition_Stability",
        "Feature_Stability",
        "Null_Robustness",
        "Window_Robustness",
        "Geometry_Separation",
        "TotalScore",
    }
    assert assessment_cols.issubset(artifact.regime_diagnostics.columns)
    assert not artifact.method_keep_decisions.loc[
        artifact.method_keep_decisions["is_baseline"],
        "keep",
    ].any()
    for row in artifact.method_keep_decisions.itertuples(index=False):
        expected_keep = (not bool(row.is_baseline)) and (
            float(row.TotalScore) > float(row.baseline_score) + float(config.keep_candidate_margin)
            and float(row.stability) > float(row.baseline_stability) + float(config.keep_candidate_margin)
        )
        assert bool(row.keep) is expected_keep
    assert artifact.specialist_candidate_features == []
    assert artifact.diagnostics["specialist_integration"] == "disabled_assessment_only"
    assert not artifact.pipeline_steps.empty
    assert {
        "01_matrix_scaling",
        "02_real_vs_null_stability_selection",
        "06_regime_discovery_assessment",
        "08_model_regime_feature_package",
    }.issubset(set(artifact.pipeline_steps["step"]))
    assert not artifact.model_regime_features.empty
    assert artifact.materialized_features.equals(artifact.model_regime_features)
    assert not artifact.model_regime_feature_metrics.empty
    assert {
        "feature",
        "source_group",
        "candidate_tier",
        "method_total_score",
        "finite_fraction",
    }.issubset(artifact.model_regime_feature_metrics.columns)
    assert artifact.diagnostics["model_regime_feature_count"] == artifact.model_regime_features.shape[1]


def test_advanced_regime_learning_artifact_round_trips(tmp_path):
    frame = _sample_frame(n=20)
    artifact = fit_advanced_regime_learning(
        frame,
        ["trend", "vol", "flow"],
        config=AdvancedRegimeLearningConfig(
            selector_backend="random_forest",
            stability_bootstraps=1,
            stability_top_m=2,
            n_estimators=4,
            n_regimes=2,
            mfa_regimes=2,
            mfa_factors=1,
            mfa_max_iter=2,
            ae_epochs=1,
            ae_latent_dim=2,
            ae_hidden_dim=6,
            ae_batch_size=32,
            max_rows=100,
        ),
    )

    paths = save_advanced_regime_learning_artifact(artifact, tmp_path)
    loaded = load_advanced_regime_learning_artifact(tmp_path)

    assert {"artifact", "manifest"}.issubset(paths)
    assert "dataframes" in paths
    assert loaded.schema_version == artifact.schema_version
    assert loaded.selected_features == artifact.selected_features
    assert loaded.regime_labels.shape == artifact.regime_labels.shape
    assert loaded.regime_tradability_diagnostics.shape == artifact.regime_tradability_diagnostics.shape
    assert loaded.pipeline_steps.shape == artifact.pipeline_steps.shape
    assert loaded.model_regime_features.shape == artifact.model_regime_features.shape
    assert loaded.model_regime_feature_metrics.shape == artifact.model_regime_feature_metrics.shape


def test_regime_hpo_runs_trials_and_persists_metrics(tmp_path):
    frame = _sample_frame(n=16)
    base_config = AdvancedRegimeLearningConfig(
        selector_backend="random_forest",
        stability_bootstraps=1,
        stability_top_m=2,
        n_estimators=3,
        max_classifier_rows=120,
        n_regimes=2,
        mfa_regimes=2,
        mfa_factors=1,
        mfa_max_iter=1,
        ae_epochs=1,
        ae_latent_dim=2,
        ae_hidden_dim=6,
        ae_batch_size=32,
        min_regime_duration=1,
        max_rows=100,
        regime_assessment_bootstraps=1,
        regime_assessment_windows=2,
        regime_assessment_null_repeats=1,
    )
    hpo_config = RegimeHPOConfig(
        max_trials=2,
        max_hpo_rows=20,
        hpo_sample_time_bins=2,
        max_trial_rows=20,
        max_trial_classifier_rows=40,
        max_trial_ae_train_rows=16,
        max_trial_assessment_rows=20,
        max_trial_leaf_trees=8,
        median_pruner_warmup_trials=1,
        median_pruner_stop_after_pruned_streak=99,
        early_stopping_patience=99,
        objective_mode="learnability",
        search_space={
            "n_regimes": [2, 3],
            "min_regime_duration": [1],
            "leaf_embedding_dim": [2],
            "bayesian_gmm_covariance_type": ["diag"],
            "bayesian_gmm_reg_covar": [1e-5],
            "hdbscan_min_samples": [3],
            "hmm_covariance_type": ["diag"],
            "hmm_transmat_self_bias": [2.0],
            "spectral_n_neighbors": [4],
            "spectral_affinity": ["nearest_neighbors"],
            "kmeans_n_init": [5],
            "kmeans_algorithm": ["lloyd"],
            "lgbm_feature_fraction": [0.8],
            "lgbm_lambda_l2": [0.1],
        },
        artifact_output_dir=tmp_path,
        store_trial_artifacts=False,
    )

    result = run_advanced_regime_learning_hpo(
        frame,
        ["trend", "vol", "flow"],
        base_config=base_config,
        hpo_config=hpo_config,
    )

    assert len(result.trials) == 2
    assert result.best_artifact is not None
    assert result.best_config is not None
    assert result.best_config.max_rows <= 20
    assert result.best_config.max_classifier_rows <= 40
    assert result.best_config.ae_max_train_rows <= 16
    assert result.best_config.regime_assessment_max_auc_rows <= 20
    assert "n_regimes" in result.best_trial_params
    assert result.trials["hpo_sampled_rows"].max() <= 20
    assert result.trials["hpo_sampling"].eq("stratified_period_symbol").all()
    assert result.trials["hpo_stop_reason"].eq("completed_max_trials").all()
    assert "median_pruner_reference" in result.trials.columns
    assert "median_pruned" in result.trials.columns
    assert result.trials.loc[result.trials["trial_id"].eq(1), "median_pruner_reference"].notna().all()
    objective_cols = {
        "hpo_objective_mode",
        "structure_score",
        "learnability_score",
        "learnability_hpo_score",
        "self_predictability",
        "soft_state_quality",
        "compression_quality",
        "support_quality",
        "specialist_feature_helpfulness",
        "feature_conditional_learnability",
        "feature_conditional_accuracy_all",
        "feature_conditional_accuracy_trend_vol",
        "feature_conditional_incremental_accuracy",
        "feature_conditional_trend_vol_replica_penalty",
        "score_dispersion_penalty",
    }
    assert objective_cols.issubset(result.trials.columns)
    assert result.trials["hpo_objective_mode"].eq("learnability").all()
    assert result.trials["learnability_score"].notna().all()
    assert result.trials["feature_conditional_learnability"].notna().all()
    bounded_objective_cols = [
        "learnability_score",
        "self_predictability",
        "soft_state_quality",
        "compression_quality",
        "support_quality",
        "specialist_feature_helpfulness",
        "feature_conditional_learnability",
        "feature_conditional_trend_vol_replica_penalty",
        "score_dispersion_penalty",
        "support_shortfall",
        "turnover_excess",
        "compute_penalty",
        "top_total_score",
        "top_nontriviality",
        "top_oos_stability",
        "top_dwell_quality",
        "top_transition_stability",
        "top_feature_stability",
        "top_null_robustness",
        "top_window_robustness",
        "top_geometry_separation",
    ]
    for col in bounded_objective_cols:
        values = pd.to_numeric(result.trials[col], errors="coerce").dropna()
        assert values.between(0.0, 1.0).all(), col
    assert result.trials["param_bayesian_gmm_covariance_type"].eq("diag").all()
    assert result.trials["param_kmeans_n_init"].eq(5).all()
    assert result.trials["param_lgbm_feature_fraction"].eq(0.8).all()
    assert result.trials["hpo_score"].notna().any()
    assert not result.trial_steps.empty
    assert not result.trial_model_feature_metrics.empty
    assert {"trials", "trial_steps", "manifest", "best_artifact", "best_manifest"}.issubset(
        result.output_paths
    )
    for key in ["trials", "trial_steps", "manifest", "best_artifact", "best_manifest"]:
        assert pd.notna(result.output_paths[key])
        assert Path(result.output_paths[key]).exists()


def test_feature_conditional_learnability_rewards_incremental_structure():
    rng = np.random.default_rng(13)
    n = 120
    labels = ((np.arange(n) % 4) >= 2).astype(np.int64)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC"),
            "symbol": "BTC",
            "trend_strength_percentile": rng.normal(size=n),
            "realized_volatility_24h": rng.normal(size=n),
            "flow_structure": labels.astype(float) + rng.normal(scale=0.03, size=n),
        }
    )
    artifact = AdvancedRegimeLearningArtifact(
        schema_version=ADVANCED_REGIME_LEARNING_SCHEMA_VERSION,
        selected_features=[],
        conservative_features=[],
        strong_features=[],
        exploratory_features=[],
        stability_frequencies=pd.DataFrame(),
        real_vs_null_importances=pd.DataFrame(),
        leaf_embeddings=pd.DataFrame(),
        raw_baseline_embeddings=pd.DataFrame(),
        ae_latents=pd.DataFrame(),
        contrastive_ae_latents=pd.DataFrame(),
        contrastive_leaf_latents=pd.DataFrame(),
        mfa_responsibilities=pd.DataFrame(),
        mfa_feature_relevance=pd.DataFrame(),
        ae_feature_gates=pd.DataFrame(),
        regime_labels=pd.DataFrame({"candidate_smoothed_regime": labels}),
        regime_probabilities=pd.DataFrame(),
        regime_transition_features=pd.DataFrame(),
        regime_feature_importance=pd.DataFrame(),
        regime_tradability_diagnostics=pd.DataFrame(),
        regime_diagnostics=pd.DataFrame(),
        pipeline_steps=pd.DataFrame(),
        model_regime_features=pd.DataFrame(),
        model_regime_feature_metrics=pd.DataFrame(),
        materialized_features=pd.DataFrame(),
        materialized_feature_groups={},
        specialist_candidate_features=[],
        method_keep_decisions=pd.DataFrame(),
        row_keys=frame[["timestamp", "symbol"]],
    )

    metrics = _feature_conditional_learnability(
        artifact,
        "candidate",
        frame,
        ["trend_strength_percentile", "realized_volatility_24h", "flow_structure"],
        hpo_config=RegimeHPOConfig(max_trial_assessment_rows=1000),
        random_state=19,
    )

    trend_vol_baseline = max(0.5, metrics["feature_conditional_accuracy_trend_vol"])
    assert metrics["feature_conditional_accuracy_all"] > trend_vol_baseline + 0.20
    assert metrics["feature_conditional_incremental_accuracy"] > 0.20
    assert metrics["feature_conditional_learnability"] > 0.50


def test_advanced_regime_learning_handles_single_row():
    frame = _sample_frame(n=1).head(1)
    artifact = fit_advanced_regime_learning(
        frame,
        ["trend", "vol", "flow"],
        config=AdvancedRegimeLearningConfig(
            selector_backend="random_forest",
            stability_bootstraps=1,
            stability_top_m=2,
            n_estimators=2,
            n_regimes=3,
            mfa_regimes=2,
            mfa_factors=1,
            mfa_max_iter=1,
            ae_epochs=1,
            ae_latent_dim=2,
            ae_hidden_dim=4,
            ae_batch_size=8,
        ),
    )

    assert len(artifact.regime_labels) == 1
    assert not artifact.regime_probabilities.empty


def test_quantile_operators_match_linear_window_quantiles():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC"),
            "symbol": "BTC",
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    out = generate_quantile_operator_features(
        frame,
        ["x"],
        window=4,
        min_periods=4,
    )

    assert np.isclose(out.loc[3, "q_iqr__x"], 1.5)
    assert np.isclose(out.loc[3, "q_tail_width__x"], 2.7)
    assert np.isclose(out.loc[3, "q_upper_tail__x"], 1.35)
    assert np.isclose(out.loc[3, "q_lower_tail__x"], 1.35)
    assert np.isclose(out.loc[3, "q_tail_asym__x"], 0.0, atol=1e-6)
    assert out.loc[3, "q_percentile_rank__x"] == 1.0


def test_pair_operators_handle_duplicate_index_labels():
    frame = _sample_frame(n=16)
    frame.index = np.repeat(np.arange(len(frame) // 2), 2)
    features = ["trend", "vol", "flow"]
    pair_scores = score_pair_candidates(
        frame,
        features,
        mechanisms={"trend": "trend", "vol": "volatility", "flow": "liquidity"},
        rolling_window=6,
        min_periods=4,
    )
    pair_features = generate_pair_operator_features(
        frame,
        pair_scores,
        window=6,
        min_periods=4,
    )

    assert len(pair_features) == len(frame)
    assert any(col.startswith("corr_w6__") for col in pair_features.columns)


def test_pair_operators_match_sample_covariance_and_correlation():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"),
            "symbol": "BTC",
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [2.0, 4.0, 6.0, 8.0],
        }
    )
    pair_scores = pd.DataFrame([{"feature_i": "x", "feature_j": "y"}])

    out = generate_pair_operator_features(
        frame,
        pair_scores,
        window=3,
        min_periods=3,
    )

    assert np.isclose(out.loc[2, "cov_w3__x__y"], 2.0)
    assert np.isclose(out.loc[2, "corr_w3__x__y"], 1.0)


def test_eigenvalue_summaries_handle_duplicate_index_labels():
    frame = _sample_frame(n=16)
    frame.index = np.repeat(np.arange(len(frame) // 2), 2)

    out = generate_eigenvalue_summary_features(
        frame,
        {"structure": ["trend", "vol", "flow"]},
        window=6,
        min_periods=4,
    )

    assert len(out) == len(frame)
    assert "eig_largest_share__structure" in out.columns


def test_build_operator_feature_frame_caps_pair_materialization():
    frame = _sample_frame(n=24)
    cfg = {
        "quality": {"warmup_rows": 2, "min_good_row_fraction": 0.80},
        "operators": {
            "quantile_window": 6,
            "autocorr_window": 6,
            "pair_window": 6,
            "eigen_window": 6,
            "min_periods": 3,
            "max_pair_candidates_for_generation": 1,
            "svd_components": [2],
            "knn_svd_components": 2,
            "knn_neighbors": 2,
            "svd_mode": "full_reference",
        },
        "operator_selection": {"target_features": 4},
    }

    derived, pair_scores, _state = build_operator_feature_frame(
        frame,
        ["trend", "vol", "flow"],
        cfg=cfg,
    )

    pair_cols = [
        col
        for col in derived.columns
        if str(col).startswith("cov_w") or str(col).startswith("corr_w")
    ]
    assert len(pair_scores) >= 1
    assert len(pair_cols) == 2


def test_pair_scoring_includes_sparse_dependency_graph_edges():
    rng = np.random.default_rng(7)
    n = 96
    z = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC"),
            "symbol": "BTC",
            "edge_a": z + rng.normal(scale=0.05, size=n) + 2.0,
            "edge_b": z + rng.normal(scale=0.05, size=n) + 2.0,
            "noise": rng.normal(size=n) + 2.0,
        }
    )

    scores = score_pair_candidates(
        frame,
        ["edge_a", "edge_b", "noise"],
        mechanisms={
            "edge_a": "trend",
            "edge_b": "liquidity",
            "noise": "volatility",
        },
        rolling_window=12,
        min_periods=8,
        sparse_graph_enabled=True,
        sparse_graph_block_hours=24,
        sparse_graph_min_block_rows=16,
        sparse_graph_alpha=0.01,
        sparse_graph_partial_corr_threshold=1e-4,
        sparse_graph_weight=1.0,
    )

    top = scores.iloc[0]
    assert {top.feature_i, top.feature_j} == {"edge_a", "edge_b"}
    assert top.graph_edge_stability > 0.0
    assert top.graph_edge_strength > 0.0
    assert top.sparse_graph_score > 0.0


def test_rolling_operators_sort_by_timestamp_before_rolling():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-02", "2026-01-01", "2026-01-03"], utc=True
            ),
            "symbol": ["BTC", "BTC", "BTC"],
            "x": [2.0, 1.0, 3.0],
        }
    )
    out = generate_quantile_operator_features(
        frame,
        ["x"],
        window=2,
        min_periods=1,
    )

    jan1 = frame["timestamp"].eq(pd.Timestamp("2026-01-01", tz="UTC"))
    jan2 = frame["timestamp"].eq(pd.Timestamp("2026-01-02", tz="UTC"))
    jan3 = frame["timestamp"].eq(pd.Timestamp("2026-01-03", tz="UTC"))
    assert np.isnan(out.loc[jan1, "q_percentile_rank__x"].iloc[0])
    assert out.loc[jan2, "q_percentile_rank__x"].iloc[0] == 1.0
    assert out.loc[jan3, "q_percentile_rank__x"].iloc[0] == 1.0


def test_pipeline_uses_prior_only_svd_knn_blocks():
    frame = _sample_frame(n=48)
    cfg = {
        "primitive_feature_keys": ["trend", "vol", "flow"],
        "quality": {"warmup_rows": 2, "min_good_row_fraction": 0.80},
        "primitive_selection": {
            "target_features": 3,
            "block_hours": 12,
            "min_block_rows": 4,
        },
        "operators": {
            "quantile_window": 6,
            "autocorr_window": 6,
            "pair_window": 6,
            "eigen_window": 6,
            "min_periods": 3,
            "svd_components": [2, 16],
            "knn_svd_components": 16,
            "knn_neighbors": 3,
            "svd_mode": "walk_forward_prior_only",
            "svd_walk_forward_block_hours": 12,
            "svd_min_prior_rows": 8,
        },
        "operator_selection": {"target_features": 20},
    }

    result = fit_unsupervised_regime_learning_features(frame, cfg=cfg)

    assert result.operators.svd_state["mode"] == "walk_forward_prior_only"
    assert result.operators.svd_state["enabled_blocks"] > 0
    assert "svd16_knn_density" in result.operators.svd_knn_features


def test_pipeline_can_enable_advanced_regime_models():
    frame = _sample_frame(n=24)
    cfg = {
        "primitive_feature_keys": ["trend", "vol", "flow"],
        "quality": {"warmup_rows": 2, "min_good_row_fraction": 0.80},
        "primitive_selection": {
            "target_features": 3,
            "block_hours": 12,
            "min_block_rows": 4,
        },
        "operators": {
            "quantile_window": 6,
            "autocorr_window": 6,
            "pair_window": 6,
            "eigen_window": 6,
            "min_periods": 3,
            "svd_components": [2],
            "knn_svd_components": 2,
            "knn_neighbors": 2,
            "svd_mode": "full_reference",
            "max_pair_candidates_for_generation": 1,
        },
        "operator_selection": {"target_features": 8},
        "regime_models": {
            "enabled": True,
            "selector_backend": "random_forest",
            "stability_bootstraps": 2,
            "stability_top_m": 4,
            "n_estimators": 6,
            "max_classifier_rows": 200,
            "n_regimes": 3,
            "mfa_regimes": 3,
            "mfa_factors": 2,
            "mfa_max_iter": 2,
            "ae_epochs": 1,
            "ae_latent_dim": 2,
            "ae_hidden_dim": 8,
            "ae_batch_size": 64,
            "min_regime_duration": 2,
            "max_rows": 200,
        },
    }

    result = fit_unsupervised_regime_learning_features(frame, cfg=cfg)

    assert result.regime_models is not None
    assert not result.regime_models.regime_diagnostics.empty
    assert "TotalScore" in result.regime_models.regime_diagnostics.columns
    assert result.regime_models.specialist_candidate_features == []
    assert not result.pipeline_steps.empty
    assert "06_fit_advanced_regime_models" in set(result.pipeline_steps["step"])
    assert not result.regime_models.pipeline_steps.empty
    assert not result.regime_models.model_regime_features.empty
