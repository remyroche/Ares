from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import extreme_price_movements.unsupervised_regime_learning.regime_models as regime_models_module
from extreme_price_movements.config import CFG
from extreme_price_movements.features import _add_regime_panel_composite_features
from extreme_price_movements.unsupervised_regime_learning.context_features import (
    RegimeContextFeatureConfig,
    build_regime_context_feature_frame,
    generate_signal_regime_interaction_features,
)
from extreme_price_movements.unsupervised_regime_learning.diagnostics import (
    compute_quality_report,
    select_primitive_features,
    select_representatives_by_spearman,
    stratified_period_sample_positions,
)
from extreme_price_movements.unsupervised_regime_learning.feature_registry import (
    BINARY_PRIMITIVE_FEATURES,
    CROSS_ASSET_DECOUPLING_FEATURES,
    FAMILY_REGIME_SPECS,
    LEVERAGE_UNWIND_FEATURES,
    LOW_PARTICIPATION_REBOUND_FEATURES,
    MARKET_LIQUIDATION_COMPOSITE_FEATURES,
    MARKET_LIQUIDATION_FUNDING_FEATURES,
    MARKET_LIQUIDATION_OHLCV_FEATURES,
    MARKET_LIQUIDATION_OI_FEATURES,
    SESSION_MICROSTRUCTURE_FEATURES,
    UNSUPERVISED_REGIME_PRIMITIVE_FEATURES,
)
from extreme_price_movements.market_regime_change_contract import (
    MARKET_REGIME_CHANGE_FEATURE_KEYS,
)
from extreme_price_movements.unsupervised_regime_learning.lgbm_feature_filter import (
    _LIGHTGBM_AVAILABLE,
    RegimeFeatureLGBMFilterConfig,
    _period_folds,
    _time_order_and_period_codes,
    extract_lgbm_reuse_contract,
    select_regime_lgbm_addon_features,
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
    _cluster_embedding,
    _cv_auc_trend_vol,
    _feature_family,
    _oof_failure_mode_helpfulness_score,
    _residualize_against_controls,
    _transition_duration_arrays,
    _trend_vol_matrix,
    fit_advanced_regime_learning,
    load_advanced_regime_learning_artifact,
    minimum_duration_smooth_by_frame,
    save_advanced_regime_learning_artifact,
)
from extreme_price_movements.unsupervised_regime_learning.validation import (
    regime_pipeline_validation_summary,
    validate_regime_learning_artifact,
)
from scripts.run_unsupervised_regime_learning_poc import (
    _load_aligned_base_oof_predictions,
    _select_base_run_by_oof_overlap,
    _select_oof_prediction_columns,
    _split_lgbm_feature_buckets,
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
    assert not set(BINARY_PRIMITIVE_FEATURES).intersection(
        cfg["primitive_feature_keys"]
    )
    assert cfg["excluded_primitive_feature_keys"] == BINARY_PRIMITIVE_FEATURES
    assert "funding_mom_w" in cfg["primitive_feature_keys"]
    assert "asset_minus_mkt_oi_1d_cp_z_8_32_96" in cfg["primitive_feature_keys"]
    assert set(LEVERAGE_UNWIND_FEATURES).issubset(set(cfg["primitive_feature_keys"]))
    assert set(LOW_PARTICIPATION_REBOUND_FEATURES).issubset(
        set(cfg["primitive_feature_keys"])
    )
    assert set(CROSS_ASSET_DECOUPLING_FEATURES).issubset(
        set(cfg["primitive_feature_keys"])
    )
    assert set(SESSION_MICROSTRUCTURE_FEATURES).issubset(
        set(cfg["primitive_feature_keys"])
    )
    assert set(MARKET_LIQUIDATION_OI_FEATURES).issubset(
        set(cfg["primitive_feature_keys"])
    )
    assert set(MARKET_LIQUIDATION_FUNDING_FEATURES).issubset(
        set(cfg["primitive_feature_keys"])
    )
    assert set(MARKET_LIQUIDATION_OHLCV_FEATURES).issubset(
        set(cfg["primitive_feature_keys"])
    )
    assert set(MARKET_LIQUIDATION_COMPOSITE_FEATURES).issubset(
        set(cfg["primitive_feature_keys"])
    )
    assert set(MARKET_REGIME_CHANGE_FEATURE_KEYS).issubset(
        set(cfg["primitive_feature_keys"])
    )
    assert "mkt_oi_flush_breadth_recovery_4h" in cfg["primitive_feature_keys"]
    assert "mkt_pct_price_up_oi_down_1h" in cfg["primitive_feature_keys"]
    assert "market_pc1_variance_share_24h" in cfg["primitive_feature_keys"]
    assert "mkt_flush_exhaustion_score" in cfg["primitive_feature_keys"]
    assert "price_up_oi_down_1h_rz" in CFG["MODEL_DIRECT_BASE_FEATURE_KEYS"]
    assert "mkt_oi_flush_breadth_recovery_4h" in CFG["REGIME_ADAPTOR_FEATURE_ORDER"]
    assert len(cfg["primitive_feature_keys"]) == len(set(cfg["primitive_feature_keys"]))
    assert cfg["regime_models"]["family_regime_specs"] == FAMILY_REGIME_SPECS
    configured_family_names = {
        str(spec["base_name"]) for spec in cfg["regime_models"]["family_regime_specs"]
    }
    assert {
        "family_leverage_unwind",
        "family_low_participation_rebound",
        "family_cross_asset_decoupling",
        "family_session_microstructure",
        "family_liquidation_lifecycle",
    }.issubset(configured_family_names)
    assert cfg["regime_models"]["stability_bootstraps"] >= 1
    assert cfg["regime_models"]["selector_backend"] in {"lgbm", "random_forest"}
    assert cfg["regime_models"]["bayesian_gmm_covariance_type"] == "diag"
    assert "hdbscan_min_cluster_size_fraction" in cfg["regime_models"]
    assert "hmm_transmat_self_bias" in cfg["regime_models"]
    assert "spectral_n_neighbors" in cfg["regime_models"]
    assert "kmeans_n_init" in cfg["regime_models"]
    assert "lgbm_feature_fraction" in cfg["regime_models"]
    assert cfg["regime_models"]["primary_trading_horizon_hours"] == 6
    assert 6 in cfg["regime_models"]["transition_change_horizons"]
    assert cfg["regime_models"]["enable_residual_structure_regimes"] is True
    assert cfg["regime_models"]["enable_family_regime_specs"] is True
    assert cfg["regime_models"]["model_helpfulness_incremental_auc_target"] > 0
    assert cfg["regime_models"]["model_helpfulness_min_stability_to_keep"] > 0
    assert cfg["regime_models"]["conditional_signal_learnability_weight"] >= 0.5
    assert (
        cfg["regime_models"]["conditional_signal_learnability_interaction_features"] > 0
    )
    assert cfg["regime_models"]["conditional_signal_learnability_min_score_to_keep"] > 0
    assert cfg["regime_models"]["future_structure_family_min_score_to_keep"] > 0
    assert cfg["regime_models"]["oof_failure_helpfulness_weight"] > 0
    assert cfg["regime_models"]["family_regime_market_summary_enabled"] is True
    assert cfg["regime_models"]["family_regime_market_summary_max_features"] > 0
    assert cfg["regime_models"]["useful_regime_model_helpfulness_weight"] >= 0.5
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


def test_regime_composite_fill_preserves_existing_liquidation_features():
    idx = pd.date_range("2026-01-01", periods=8, freq="h", tz="UTC")
    cols = pd.Index(["BTC/USD:USD", "ETH/USD:USD"])
    original = pd.DataFrame(
        np.linspace(0.0, 1.0, len(idx) * len(cols), dtype=np.float32).reshape(
            len(idx), len(cols)
        ),
        index=idx,
        columns=cols,
    )
    feats = {"liquidation_onset_score": original.copy()}

    generated = _add_regime_panel_composite_features(
        feats,
        {"liquidation_onset_score"},
        CFG,
        idx,
        cols,
    )

    assert "liquidation_onset_score" in generated
    pd.testing.assert_frame_equal(feats["liquidation_onset_score"], original)


def test_regime_family_resolver_supports_market_context_families():
    cases = {
        "unwind_score": "crowding",
        "oi_value_7d_log_chg_z_180d": "open_interest",
        "asset_minus_mkt_oi_1d_ts_resid": "open_interest",
        "funding_1d_chg_peer_resid": "funding",
        "basis_fund_div_mkt_resid": "funding",
        "basis_per_atr": "basis",
        "premium_proxy_mom_8h": "mark_index",
        "spot_perp_volume_ratio_24h": "liquidity",
        "volume_price_corr_ts_resid": "liquidity",
        "compression_score": "compression",
        "symbol_minus_mkt_ret_24h": "asset_relative_return",
        "xasset_asset_minus_mkt_funding": "asset_relative_funding",
        "xasset_asset_minus_mkt_ob_pressure_z_24h": "asset_relative_orderbook",
        "market_dispersion_24h": "cross_sectional",
        "loc_session_pos_24": "session_microstructure",
        "funding_phase_sin": "session_microstructure",
        "ob_spread_bps_z_24h": "orderbook",
        "xasset_mkt_ob_stress_z_24h": "cross_asset_orderbook",
        "eig_top3_share__session_microstructure": "session_microstructure",
        "eig_turnover__cross_asset_orderbook": "cross_asset_orderbook",
        "q_tail_width__mark_index": "mark_index",
    }
    for feature, expected_family in cases.items():
        assert _feature_family(feature) == expected_family


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
            "timestamp": list(
                pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
            )
            * 2,
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
    frame = pd.DataFrame(
        {"timestamp": pd.date_range("2026-01-01", periods=6, freq="h", tz="UTC")}
    )

    assert trend_vol.shape == (6, 0)
    assert (
        _cv_auc_trend_vol(
            trend_vol,
            labels,
            blocks=[np.arange(0, 3, dtype=np.int64), np.arange(3, 6, dtype=np.int64)],
            random_state=1,
            max_features=96,
            max_rows=100,
        )
        == 0.5
    )


def test_oof_failure_mode_helpfulness_scores_aligned_failures():
    n = 80
    labels = np.asarray(([0, 1] * (n // 2)), dtype=np.int64)
    probs = np.column_stack([labels == 0, labels == 1]).astype(np.float32)
    target = np.where(labels == 1, 0.0, 1.0).astype(np.float32)
    oof_pred = np.where(labels == 1, 0.90, 0.75).astype(np.float32)
    blocks = [np.arange(i, n, 4, dtype=np.int64) for i in range(4)]

    metrics = _oof_failure_mode_helpfulness_score(
        labels,
        probs,
        downstream_target=target,
        base_oof_pred=oof_pred,
        oos_blocks=blocks,
        config=AdvancedRegimeLearningConfig(
            oof_failure_min_coverage=0.5,
            oof_failure_helpfulness_target=0.05,
            regime_assessment_max_auc_rows=200,
        ),
    )

    assert metrics["OOFCoverage"] == 1.0
    assert metrics["OOFFailureModeHelpfulness"] > 0.0
    assert metrics["OOFResidualStateSeparation"] > 0.0


def test_residual_structure_removes_trend_vol_projection():
    trend = np.linspace(-2.0, 2.0, 40, dtype=np.float32)
    vol = np.cos(np.linspace(0.0, 4.0, 40, dtype=np.float32))
    residual_signal = np.sin(np.linspace(0.0, 8.0, 40, dtype=np.float32))
    flow = 2.0 * trend - 0.5 * vol + residual_signal
    matrix = np.column_stack([trend, vol, flow]).astype(np.float32)

    residual, residual_features, diag = _residualize_against_controls(
        matrix,
        ["trend_strength_percentile", "realized_volatility_24h", "flow_structure"],
        control_families=("trend", "volatility"),
    )

    assert diag["status"] == "completed"
    assert residual_features == ["flow_structure"]
    assert residual.shape == (40, 1)
    assert abs(float(np.corrcoef(residual[:, 0], trend)[0, 1])) < 0.15
    assert abs(float(np.corrcoef(residual[:, 0], vol)[0, 1])) < 0.15


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
    steps_by_name = artifact.pipeline_steps.set_index("step")
    assert (
        steps_by_name.loc[
            "04_autoencoder_latents",
            "sparse_input_source",
        ]
        == "05_final_regime_learning_feature_set"
    )
    assert (
        steps_by_name.loc[
            "04_autoencoder_latents",
            "contrastive_input_source",
        ]
        == "05_final_regime_learning_feature_set"
    )
    assert (
        steps_by_name.loc[
            "04_autoencoder_latents",
            "contrastive_leaf_input_source",
        ]
        == "03_leaf_and_raw_embeddings"
    )
    assert (
        steps_by_name.loc[
            "05_mixture_factor_analyzers",
            "input_source",
        ]
        == "05_final_regime_learning_feature_set"
    )
    assert any(
        col.endswith("_regime_prob_entropy")
        for col in artifact.regime_transition_features.columns
    )
    assert any(
        col.endswith("_regime_prob_max")
        for col in artifact.regime_transition_features.columns
    )
    assert any(
        col.endswith("_regime_prob_change_1h")
        for col in artifact.regime_transition_features.columns
    )
    assert any(
        col.endswith("_regime_prob_change_4h")
        for col in artifact.regime_transition_features.columns
    )
    assert any(
        col.endswith("_regime_prob_change_6h")
        for col in artifact.regime_transition_features.columns
    )
    assert any(
        col.endswith("_regime_prob_change_12h")
        for col in artifact.regime_transition_features.columns
    )
    assert any(
        col.endswith("_regime_prob_change_24h")
        for col in artifact.regime_transition_features.columns
    )
    assert any(
        col.endswith("_regime_transition_hazard")
        for col in artifact.regime_transition_features.columns
    )
    assert any(
        col.endswith("_time_since_regime_change")
        for col in artifact.regime_transition_features.columns
    )
    assert any(
        col.endswith("_expected_regime_duration")
        for col in artifact.regime_transition_features.columns
    )
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
    assert "residual_structure_kmeans" in set(artifact.regime_diagnostics["method"])
    assert "regime_family" in artifact.regime_diagnostics.columns
    assert "residual_structure" in set(
        artifact.regime_diagnostics["regime_family"].astype(str)
    )
    discovery_step = artifact.pipeline_steps.set_index("step").loc[
        "06_regime_discovery_assessment"
    ]
    assert int(discovery_step["family_candidate_method_count"]) >= 2
    assert discovery_step["future_structure_target_status"] in {
        "completed",
        "too_few_valid_rows",
        "degenerate_target",
        "no_future_rows",
    }
    assert (
        artifact.diagnostics["training_inference_contract"][
            "primary_trading_horizon_hours"
        ]
        == 6
    )
    assert artifact.diagnostics["assessment"]["score"] == "UsefulRegimeScore"
    spectral_diag = artifact.regime_diagnostics.set_index("method").loc[
        "raw_spectral_spectral"
    ]
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
        "UnsupervisedQualityScore",
        "UsefulRegimeScore",
        "FutureStructureAUC",
        "TrendVolFutureStructureAUC",
        "IncrementalFutureStructureAUC",
        "FutureStructureHelpfulness",
        "SignalOnlyFutureStructureAUC",
        "RegimeConditionedSignalAUC",
        "IncrementalConditionalSignalAUC",
        "ConditionalSignalLearnability",
        "ModelHelpfulness",
        "StructuralModelHelpfulness",
        "ConditionalSignalLearnabilityProxy",
        "OOFCoverage",
        "OOFFailureAUC",
        "OOFFailureAUCIncrement",
        "OOFResidualStateSeparation",
        "OOFFalsePositiveStateSeparation",
        "OOFPrecisionStateSeparation",
        "OOFFailureModeHelpfulness",
    }
    assert assessment_cols.issubset(artifact.regime_diagnostics.columns)
    assert not artifact.method_keep_decisions.loc[
        artifact.method_keep_decisions["is_baseline"],
        "keep",
    ].any()
    for row in artifact.method_keep_decisions.itertuples(index=False):
        conditional_enough = float(row.ConditionalSignalLearnability) >= float(
            config.conditional_signal_learnability_min_score_to_keep
        )
        strong_family_structure = bool(row.strong_family_structure)
        expected_keep = (not bool(row.is_baseline)) and (
            float(row.UsefulRegimeScore)
            > float(row.baseline_score) + float(config.keep_candidate_margin)
            and float(row.ModelHelpfulness)
            >= float(config.model_helpfulness_min_score_to_keep)
            and (conditional_enough or strong_family_structure)
            and float(row.stability)
            >= float(config.model_helpfulness_min_stability_to_keep)
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
        "method_useful_regime_score",
        "method_model_helpfulness",
        "method_total_score",
        "finite_fraction",
    }.issubset(artifact.model_regime_feature_metrics.columns)
    assert (
        artifact.diagnostics["model_regime_feature_count"]
        == artifact.model_regime_features.shape[1]
    )
    contract = artifact.diagnostics["training_inference_contract"]
    assert contract["regime_model_scope"] == "pooled_across_assets"
    assert contract["row_application_scope"] == "per_asset_independent"
    assert contract["context_feature_builder"] == "build_regime_context_feature_frame"
    validation_report = validate_regime_learning_artifact(artifact)
    assert not validation_report.empty
    assert validation_report.loc[
        validation_report["check"].eq("uses_final_feature_set"),
        "passed",
    ].all()
    summary = regime_pipeline_validation_summary(validation_report)
    assert summary["check_count"] >= 8


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
    assert (
        loaded.regime_tradability_diagnostics.shape
        == artifact.regime_tradability_diagnostics.shape
    )
    assert loaded.pipeline_steps.shape == artifact.pipeline_steps.shape
    assert loaded.model_regime_features.shape == artifact.model_regime_features.shape
    assert (
        loaded.model_regime_feature_metrics.shape
        == artifact.model_regime_feature_metrics.shape
    )


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
    assert (
        result.trials.loc[result.trials["trial_id"].eq(1), "median_pruner_reference"]
        .notna()
        .all()
    )
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
        "top_useful_regime_score",
        "top_model_helpfulness",
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
    assert {
        "trials",
        "trial_steps",
        "manifest",
        "best_artifact",
        "best_manifest",
    }.issubset(result.output_paths)
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


def test_regime_context_features_are_train_inference_deterministic():
    frame = (
        pd.DataFrame(
            {
                "timestamp": list(
                    pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
                )
                * 3,
                "symbol": ["BTC"] * 4 + ["ETH"] * 4 + ["SOL"] * 4,
            }
        )
        .sort_values(["timestamp", "symbol"], kind="mergesort")
        .reset_index(drop=True)
    )
    n = len(frame)
    regime_outputs = pd.DataFrame(
        {
            "candidate_regime_prob_00": np.linspace(0.1, 0.9, n),
            "candidate_regime_prob_01": np.linspace(0.9, 0.1, n),
            "candidate_regime_prob_entropy": np.linspace(0.2, 0.8, n),
            "candidate_regime_prob_max": np.linspace(0.55, 0.95, n),
            "candidate_regime_prob_change_1h": np.linspace(0.0, 0.5, n),
            "candidate_smoothed_regime": (np.arange(n) % 2).astype(float),
            "candidate_regime_transition_hazard": np.linspace(0.0, 1.0, n),
            "candidate_time_since_regime_change": np.linspace(1.0, 12.0, n),
            "candidate_expected_regime_duration": np.linspace(2.0, 10.0, n),
        }
    )

    cfg = RegimeContextFeatureConfig(max_residual_features=8)
    train_features, train_diag = build_regime_context_feature_frame(
        frame,
        regime_outputs,
        config=cfg,
    )
    inference_features, inference_diag = build_regime_context_feature_frame(
        frame.copy(),
        regime_outputs.copy(),
        config=cfg,
    )

    pd.testing.assert_frame_equal(train_features, inference_features)
    assert (
        train_diag["train_inference_parity_surface"]
        == "deterministic_row_level_regime_output_transform"
    )
    assert inference_diag["output_feature_count"] == train_features.shape[1]
    assert any(col.startswith("url_asset__") for col in train_features.columns)
    assert any(col.startswith("url_xs_z__") for col in train_features.columns)
    assert any(col.startswith("url_market__") for col in train_features.columns)
    assert "url_market__mean__candidate_regime_prob_00" in train_features.columns
    residual_cols = [
        col for col in train_features.columns if col.startswith("url_xs_z__")
    ]
    grouped = (
        train_features[residual_cols].groupby(frame["timestamp"], sort=False).mean()
    )
    assert np.allclose(grouped.to_numpy(dtype=float), 0.0, atol=1e-5)
    assert "url_market__mean__candidate_regime_prob_entropy" in train_features.columns
    assert "url_market__mean__candidate_regime_prob_change_1h" in train_features.columns
    assert (
        "url_market__mean__candidate_regime_transition_hazard" in train_features.columns
    )
    assert train_diag["groups"].get("latent_asset_context", 0) > 0
    assert train_diag["groups"].get("latent_market_context", 0) > 0
    assert train_diag["groups"].get("latent_cross_sectional_context", 0) > 0
    assert train_diag["groups"].get("context_portfolio_asset", 0) > 0
    assert train_diag["groups"].get("context_portfolio_market", 0) > 0
    assert train_diag["groups"].get("context_portfolio_cross_sectional", 0) > 0
    for col in [
        "url_asset__latent_candidate_uncertainty",
        "url_asset__latent_candidate_transition_pressure",
        "url_asset__latent_candidate_maturity",
        "url_asset__latent_candidate_conditional_confidence",
        "url_asset__ctx_portfolio_candidate_risk_budget",
        "url_asset__ctx_portfolio_candidate_risk_cut",
        "url_asset__ctx_portfolio_risk_budget_mean",
        "url_market__mean__ctx_portfolio_risk_budget_mean",
        "url_xs_z__ctx_portfolio_risk_budget_mean",
        "url_asset__latent_defensive_no_trade_score",
        "url_asset__latent_conditional_confidence_score",
        "url_market__latent_defensive_no_trade_score",
        "url_market__latent_conditional_confidence_score",
        "url_xs_z__latent_defensive_no_trade_score",
    ]:
        assert col in train_features.columns
    bounded_latent_cols = [
        col
        for col in train_features.columns
        if "latent_" in col and not col.startswith("url_xs_z__")
    ]
    bounded_portfolio_cols = [
        col
        for col in train_features.columns
        if "ctx_portfolio_" in col and not col.startswith("url_xs_z__")
    ]
    bounded = train_features[bounded_latent_cols + bounded_portfolio_cols].to_numpy(
        dtype=float
    )
    assert np.isfinite(bounded).all()
    assert np.nanmin(bounded) >= -1e-6
    assert np.nanmax(bounded) <= 1.0 + 1e-6
    assert (
        train_features["url_asset__latent_defensive_no_trade_score"].max()
        > train_features["url_asset__latent_defensive_no_trade_score"].min()
    )
    assert (
        train_features["url_asset__latent_conditional_confidence_score"].max()
        > train_features["url_asset__latent_conditional_confidence_score"].min()
    )


def test_regime_context_features_align_by_position_and_reject_wrong_lengths():
    frame = pd.DataFrame(
        {
            "timestamp": list(
                pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
            )
            * 2,
            "symbol": ["BTC"] * 3 + ["ETH"] * 3,
        },
        index=[10, 11, 12, 20, 21, 22],
    ).sort_values(["timestamp", "symbol"], kind="mergesort")
    n = len(frame)
    regime_outputs = pd.DataFrame(
        {
            "candidate_regime_prob_00": np.linspace(0.2, 0.8, n),
            "candidate_smoothed_regime": (np.arange(n) % 2).astype(float),
        }
    )

    features, diag = build_regime_context_feature_frame(frame, regime_outputs)

    assert features.index.equals(frame.index)
    assert diag["row_alignment"] == "positional"
    assert not features.empty
    with pytest.raises(ValueError, match="same row count"):
        build_regime_context_feature_frame(frame, regime_outputs.iloc[:-1])
    with pytest.raises(ValueError, match="index does not match"):
        build_regime_context_feature_frame(
            frame,
            regime_outputs,
            config=RegimeContextFeatureConfig(allow_positional_row_alignment=False),
        )


def test_regime_lgbm_period_folds_use_stratified_period_sampling():
    n = 240
    ts = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    y = ((np.arange(n) // 12) % 2).astype(np.int8)
    _order, period_codes = _time_order_and_period_codes(n, ts, n_periods=8)

    folds = _period_folds(
        n,
        y,
        ts,
        n_folds=4,
        sample_fraction=0.5,
        max_rows=80,
        random_state=23,
        stratified_period_bins=8,
    )

    assert folds
    val_union = np.concatenate([val for _train, val in folds])
    assert len(val_union) <= 80
    assert np.unique(period_codes[val_union]).size >= 6
    for train, val in folds:
        assert len(train) <= int(np.ceil(0.5 * (80 - len(val)))) + 2
        assert np.unique(y[train]).size == 2
        assert np.unique(period_codes[train]).size >= 4


def test_regime_lgbm_addon_filter_reuses_contract_and_ranks_features():
    if not _LIGHTGBM_AVAILABLE:
        pytest.skip("lightgbm unavailable")
    rng = np.random.default_rng(17)
    n = 180
    ts = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    signal = rng.normal(size=n)
    regime_good = rng.normal(size=n)
    portfolio_budget = np.clip(
        0.50 + 0.35 * np.tanh(regime_good) + 0.15 * rng.normal(size=n), 0.0, 1.0
    )
    base_oof_pred = 1.0 / (1.0 + np.exp(-signal))
    y = ((signal > 0.0) & (portfolio_budget > np.nanmedian(portfolio_budget))).astype(
        int
    )
    frame = pd.DataFrame(
        {
            "signal_a": signal,
            "signal_b": rng.normal(size=n),
            "url_asset__regime_good": regime_good,
            "url_asset__ctx_portfolio_test_risk_budget": portfolio_budget,
            "url_market__regime_noise": rng.normal(size=n),
        }
    )
    contract = extract_lgbm_reuse_contract(
        {
            "train_base": {
                "selected_features": ["signal_a", "signal_b"],
                "best_params": {
                    "n_estimators": 40,
                    "learning_rate": 0.05,
                    "max_depth": 3,
                },
            }
        },
        stage="train_base",
    )

    result = select_regime_lgbm_addon_features(
        frame,
        y,
        base_features=contract["selected_features"],
        regime_features=[
            "url_asset__regime_good",
            "url_asset__ctx_portfolio_test_risk_budget",
            "url_market__regime_noise",
        ],
        timestamps=ts,
        base_oof_pred=base_oof_pred,
        reused_model_params=contract["params"],
        config=RegimeFeatureLGBMFilterConfig(
            n_folds=3,
            max_trees=40,
            min_child_samples=5,
            structural_min_trees_using=1,
            structural_min_tree_fraction=0.0,
            structural_min_leaf_path_share=0.0,
            structural_min_sample_path_exposure=0.0,
            structural_min_gain_ratio=0.0,
            use_shadow_gain=False,
            base_context_filter_enabled=False,
            risk_budget_scaler_min_fold_fraction=0.34,
            risk_budget_scaler_min_scaled_hr_lift=-1e-9,
            risk_budget_scaler_min_high_low_hr_lift=-1e-9,
            risk_budget_scaler_min_failure_avoidance=-1.0,
            lift_positive_fold_fraction=0.34,
            stability_positive_fold_fraction=0.34,
            score_positive_fold_fraction=0.34,
            min_group_rows=5,
            top_n=1,
        ),
    )

    assert result.diagnostics["status"] == "completed"
    assert not result.fold_metrics.empty
    assert not result.feature_metrics.empty
    assert result.selected_features
    assert result.selected_features[0] in set(result.feature_metrics["feature"])
    for col in [
        "signal_uplift_mean_abs",
        "signal_uplift_context_pass",
        "oof_available_rate",
        "median_oof_failure_lift",
        "risk_gate_acceptance_score",
        "risk_gate_acceptance_pass",
        "risk_budget_scaler_score",
        "risk_budget_scaler_pass",
    ]:
        assert col in result.feature_metrics.columns
    for col in [
        "signal_uplift_pair_count",
        "oof_failure_lift",
        "risk_budget_scaled_hr_lift",
    ]:
        assert col in result.fold_metrics.columns
    budget_metrics = result.feature_metrics.set_index("feature").loc[
        "url_asset__ctx_portfolio_test_risk_budget"
    ]
    assert budget_metrics["source"] == "context_portfolio"
    assert float(budget_metrics["risk_budget_scaler_available_rate"]) > 0.0
    assert bool(budget_metrics["risk_budget_scaler_pass"])
    assert result.diagnostics["risk_budget_scaler_pass_count"] >= 1


def test_signal_regime_interaction_features_are_bounded_and_deterministic():
    frame = pd.DataFrame(
        {
            "signal_a": [1.0, 2.0, 3.0, 4.0],
            "signal_b": [0.5, 0.25, 0.0, -0.25],
            "url_asset__regime_prob_00": [0.2, 0.4, 0.6, 0.8],
            "url_market__regime_prob_entropy": [0.8, 0.6, 0.4, 0.2],
        }
    )

    out, diag = generate_signal_regime_interaction_features(
        frame,
        ["signal_a", "signal_b"],
        ["url_asset__regime_prob_00", "url_market__regime_prob_entropy"],
        config=RegimeContextFeatureConfig(
            max_signal_interaction_signal_features=1,
            max_signal_interaction_regime_features=2,
            max_signal_regime_interaction_features=2,
        ),
    )

    assert diag["status"] == "completed"
    assert out.shape[1] <= 2
    assert all(col.startswith("url_sigreg__") for col in out.columns)
    out2, diag2 = generate_signal_regime_interaction_features(
        frame,
        ["signal_a", "signal_b"],
        ["url_asset__regime_prob_00", "url_market__regime_prob_entropy"],
        config=RegimeContextFeatureConfig(
            max_signal_interaction_signal_features=1,
            max_signal_interaction_regime_features=2,
            max_signal_regime_interaction_features=2,
        ),
    )
    pd.testing.assert_frame_equal(out, out2)
    assert diag2["output_feature_count"] == diag["output_feature_count"]


def test_poc_base_oof_selection_prefers_overlap_and_label_head(tmp_path):
    ts = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    sample = (
        pd.DataFrame(
            {
                "timestamp": list(ts) * 2,
                "symbol": ["BTC"] * 4 + ["ETH"] * 4,
            }
        )
        .sort_values(["timestamp", "symbol"], kind="mergesort")
        .reset_index(drop=True)
    )
    label_path = tmp_path / "train_alpha_beta_5.parquet"
    data_root = tmp_path / "data"

    def write_oof(run_id: str, frame: pd.DataFrame) -> None:
        run_dir = data_root / "artifacts" / run_id
        (run_dir / "oof").mkdir(parents=True)
        (run_dir / "base_models_intermediate.pkl").write_bytes(b"dummy")
        frame.to_parquet(run_dir / "oof" / "base_oof_all.parquet", index=False)

    low = sample.head(2).copy()
    low["oof_alpha_beta_H5"] = [0.1, 0.2]
    low["oof_unrelated_H5"] = [0.9, 0.8]
    write_oof("20260619_low_overlap", low)

    high = sample.copy()
    high["oof_alpha_beta_H5"] = np.linspace(0.2, 0.8, len(high))
    high["oof_unrelated_H5"] = np.linspace(0.8, 0.2, len(high))
    high["oof_alpha_beta_H5_sigma"] = 0.0
    write_oof("20260619_high_overlap", high)

    cols, match = _select_oof_prediction_columns(
        ["timestamp", "symbol", "oof_alpha_beta_H5", "oof_unrelated_H5"],
        label_path,
    )
    assert cols == ["oof_alpha_beta_H5"]
    assert match["match_type"] == "label_head_and_horizon"

    run_id, diag = _select_base_run_by_oof_overlap(
        data_root,
        sample,
        label_path=label_path,
    )

    assert run_id == "20260619_high_overlap"
    assert diag["selected"]["finite_coverage"] == 1.0
    assert (
        diag["selected"]["prediction_column_match"]["match_type"]
        == "label_head_and_horizon"
    )

    pred, pred_diag = _load_aligned_base_oof_predictions(
        data_root / "artifacts" / run_id,
        sample,
        label_path=label_path,
    )
    assert pred_diag["coverage"] == 1.0
    assert pred_diag["selected_prediction_columns"] == ["oof_alpha_beta_H5"]
    assert np.allclose(
        pred.to_numpy(dtype=float), high["oof_alpha_beta_H5"].to_numpy(dtype=float)
    )


def test_poc_lgbm_feature_buckets_separate_risk_and_exploratory():
    metrics = pd.DataFrame(
        {
            "feature": ["additive", "risk", "interaction", "rejected_risk"],
            "source": [
                "probability",
                "probability",
                "signal_regime_interaction",
                "leaf",
            ],
            "context_role": [
                "mixed_or_weak_context",
                "risk_gate",
                "risk_gate",
                "risk_gate",
            ],
            "rank_score": [0.3, 0.2, 0.4, 0.9],
            "risk_gate_acceptance_score": [0.0, 0.5, 0.4, 0.1],
            "risk_gate_acceptance_pass": [False, True, True, False],
            "oof_failure_alignment_pass": [False, True, False, False],
            "signal_uplift_context_pass": [False, True, True, False],
            "opportunity_context_pass": [False, False, False, False],
            "context_helper_candidate_pass": [False, False, True, False],
            "signal_uplift_mean_abs": [0.01, 0.2, 0.3, 0.01],
            "pre_redundancy_keep": [True, True, True, True],
            "redundancy_keep": [True, True, True, True],
            "source_keep": [True, True, True, True],
        }
    )

    summary, frames = _split_lgbm_feature_buckets(
        metrics,
        ["additive", "risk", "interaction", "rejected_risk"],
    )

    assert summary["selected_additive_features"] == ["additive"]
    assert summary["accepted_oof_aligned_risk_gates"] == ["risk"]
    assert summary["production_risk_gates"] == ["risk"]
    assert summary["candidate_context_helpers"] == ["interaction"]
    assert summary["exploratory_context_interactions"] == ["interaction"]
    assert "rejected_risk" in summary["diagnostic_only_regime_features"]
    assert "rejected_risk" not in summary["selected_additive_features"]
    assert frames["accepted_oof_aligned_risk_gates"]["feature"].tolist() == ["risk"]
    assert frames["production_risk_gates"]["feature"].tolist() == ["risk"]
    assert frames["candidate_context_helpers"]["feature"].tolist() == ["interaction"]


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
    assert "UsefulRegimeScore" in result.regime_models.regime_diagnostics.columns
    assert result.regime_models.specialist_candidate_features == []
    assert not result.pipeline_steps.empty
    assert "06_fit_advanced_regime_models" in set(result.pipeline_steps["step"])
    assert not result.regime_models.pipeline_steps.empty
    assert not result.regime_models.model_regime_features.empty


def test_bayesian_gmm_cluster_falls_back_on_numerical_failure(monkeypatch):
    class FailingBGM:
        def fit_predict(self, x):
            raise ValueError("ill-defined empirical covariance")

    monkeypatch.setattr(
        regime_models_module,
        "_bayesian_gmm_model",
        lambda *args, **kwargs: FailingBGM(),
    )
    rng = np.random.default_rng(123)
    x = rng.normal(size=(40, 4)).astype(np.float32)

    labels, probs, used = _cluster_embedding(
        x,
        method="bayesian_gmm",
        n_regimes=3,
        random_state=42,
        config=AdvancedRegimeLearningConfig(kmeans_n_init=2),
    )

    assert labels.shape == (40,)
    assert probs is None
    assert used == "kmeans"
