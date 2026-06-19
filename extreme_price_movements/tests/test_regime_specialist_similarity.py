import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.regime_specialist_similarity import (
    RegimeSimilarityConfig,
    SpecialistWeightConfig,
    build_regime_specialist_training_frame,
    compute_regime_similarity_to_current,
    compute_specialist_sample_weights,
    current_regime_recency_weights,
    shrink_self_distillation_towards_one,
    weighted_drift_baseline,
)
from extreme_price_movements.regime_specialist_feature_engineering import (
    RegimeFeatureEngineeringConfig,
    _auc_lift,
    _candidate_pool,
    _cv_rank_and_scores,
    _mean_nearest_distance,
    build_regime_specialist_feature_engineering_artifact,
)


def _synthetic_regime_frame() -> pd.DataFrame:
    n_days = 70
    rows = []
    for day in range(n_days):
        for sym_i, symbol in enumerate(("BTC/USD:USD", "ETH/USD:USD")):
            t = pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=day)
            phase = day / 8.0 + sym_i * 0.3
            current_bump = 1.0 if day >= n_days - 21 else 0.0
            rows.append(
                {
                    "timestamp": t,
                    "symbol": symbol,
                    "volatility_percentile": 0.4 + 0.2 * np.sin(phase) + 0.1 * current_bump,
                    "volume_percentile": 0.5 + 0.2 * np.cos(phase),
                    "volume_regime_marker": current_bump + 0.05 * np.sin(phase),
                    "liquidity_rank": 0.4 + 0.1 * sym_i + 0.2 * current_bump + 0.03 * np.sin(day / 2.0),
                    "correlation_percentile": 0.3 + 0.15 * np.sin(day / 13.0),
                    "cross_sectional_dispersion": 0.2 + 0.1 * np.cos(day / 10.0),
                    "funding_average": 0.01 * np.sin(day / 5.0),
                    "funding_dispersion": 0.02 + 0.01 * np.cos(day / 4.0),
                    "aggregate_oi_growth": 0.03 * np.sin(day / 6.0),
                    "oi_over_volume": 0.5 + 0.1 * np.cos(day / 9.0),
                    "breadth": 0.45 + 0.25 * np.sin(day / 11.0),
                    "trend_strength": 0.3 + 0.3 * np.cos(day / 7.0),
                    "price_entropy": 0.6 + 0.1 * np.sin(day / 12.0),
                    "feature_drift_psi_core": 0.1 + 0.05 * current_bump + 0.02 * np.sin(phase),
                    "feature_drift_ks_core": 0.08 + 0.03 * np.cos(phase),
                    "feature_drift_wasserstein_core": 0.05 + 0.02 * np.sin(day / 4.0),
                    "feature_drift_mahalanobis_core": 1.5 + 0.4 * current_bump + 0.1 * sym_i,
                    "feature_covariance_drift": 0.05 + 0.04 * current_bump,
                    "base_model_feature_drift": 0.07 + 0.01 * sym_i,
                    "meta_model_feature_drift": 0.06 + 0.02 * np.sin(day / 3.0),
                    "prediction_distribution_drift": 0.04 + 0.02 * current_bump,
                    "feature_0": np.sin(phase),
                    "feature_1": np.cos(phase),
                    "feature_2": np.sin(day / 3.0) + sym_i * 0.1,
                    "feature_3": np.cos(day / 5.0),
                    "return_1h": 0.01 * np.sin(day / 6.0 + sym_i),
                }
            )
    return pd.DataFrame(rows)


def test_current_regime_recency_weights_sum_to_one_and_favor_recent_rows():
    ts = pd.date_range("2026-01-01", periods=4, freq="7D", tz="UTC")
    weights = current_regime_recency_weights(ts, current_end=ts[-1])
    assert weights.sum() == pytest.approx(1.0)
    assert weights.iloc[-1] > weights.iloc[0]


def test_regime_specialist_training_frame_outputs_similarity_and_weights():
    frame = _synthetic_regime_frame()
    cfg = RegimeSimilarityConfig(
        ae_enabled=False,
        min_candidate_rows=8,
        min_current_rows=8,
        knn_k=3,
        max_knn_current_rows=40,
        max_knn_candidate_rows=40,
    )
    out, diag = build_regime_specialist_training_frame(
        frame,
        selected_feature_columns=["feature_0", "feature_1", "feature_2", "feature_3"],
        similarity_config=cfg,
        weight_config=SpecialistWeightConfig(
            tau_current=10.0,
            tau_analogue=10.0,
            tau_normal=10.0,
            tau_irrelevant=10.0,
        ),
    )

    assert diag["similarity"]["enabled"] is True
    assert out["similarity_to_current"].between(0.0, 1.0).all()
    assert out["window_similarity"].between(0.0, 1.0).all()
    assert out["day_similarity"].between(0.0, 1.0).all()
    assert out["day_similarity_available"].dtype == bool
    assert out["regime_specialist_sample_weight"].mean() == pytest.approx(1.0)
    assert np.isfinite(out["regime_specialist_sample_weight"].to_numpy()).all()
    assert set(out["regime_specialist_bucket"]).issubset(
        {"current", "analogue", "normal", "irrelevant"}
    )
    current = out["regime_specialist_bucket"] == "current"
    assert current.any()
    assert out.loc[current, "current_regime_recency_weight"].sum() == pytest.approx(1.0)
    assert out.loc[~current, "current_regime_recency_weight"].sum() == pytest.approx(0.0)
    assert diag["similarity"]["block_scaling"]["combined_from_normalized_distances"] is True
    assert (
        diag["similarity"]["block_scaling"]["internal_distance_scaling"]
        == "component_robust_scale_within_weighted_blocks"
    )
    assert diag["similarity"]["block_scaling"]["tau"] > 0.0
    assert "feature_cov_eig" in diag["similarity"]["block_scaling"]["covariance_subblocks"]["weights"]
    assert "asset_corr_eig" in diag["similarity"]["block_scaling"]["covariance_subblocks"]["weights"]
    assert "psi" in diag["similarity"]["block_scaling"]["drift_subblocks"]["weights"]
    assert "wasserstein" in diag["similarity"]["block_scaling"]["drift_subblocks"]["weights"]
    assert diag["similarity"]["drift_normalization"]["enabled"] is True
    assert diag["similarity"]["drift_normalization"]["counts"]["psi"] > 0
    assert diag["similarity"]["drift_normalization"]["counts"]["ks"] > 0
    assert diag["similarity"]["drift_normalization"]["counts"]["wasserstein"] > 0
    assert diag["similarity"]["drift_normalization"]["counts"]["mahalanobis"] > 0
    assert diag["similarity"]["drift_normalization"]["baseline_covariance_norm"] > 0.0
    assert diag["similarity"]["weights"]["feature_drift_distance"] == pytest.approx(0.40)
    assert diag["similarity"]["weights"]["covariance_distance"] == pytest.approx(0.35)
    assert diag["similarity"]["weights"]["domain_classifier_distance"] == pytest.approx(0.0)
    assert diag["similarity"]["scaling"]["source"] == "pre_current_history"
    assert diag["similarity"]["autoencoder"]["used"] is False
    assert diag["similarity"]["autoencoder"]["reason"] == "disabled"
    assert diag["similarity"]["knn"]["mode"] == "global_knn"
    assert diag["similarity"]["knn"]["max_candidate_rows_per_window"] == 40
    assert diag["weighted_drift_baseline"]["enabled"] is True
    assert diag["weighted_drift_baseline"]["feature_count"] > 0
    assert "feature_drift_psi_core" in diag["weighted_drift_baseline"]["stats"]


def test_regime_specialist_training_frame_reports_unsupervised_regime_artifact_without_injection():
    frame = _synthetic_regime_frame()
    materialized = pd.DataFrame(
        {
            "url_leaf_00": np.linspace(0.0, 1.0, len(frame), dtype=np.float32),
            "url_mfa_gamma_00": np.where(
                np.arange(len(frame)) % 2 == 0,
                0.8,
                0.2,
            ).astype(np.float32),
        },
        index=frame.index,
    )

    class _Artifact:
        schema_version = "unsupervised_regime_learning_v2"
        materialized_features = materialized
        specialist_candidate_features = ["url_leaf_00", "url_mfa_gamma_00"]

    out, diag = build_regime_specialist_training_frame(
        frame,
        selected_feature_columns=["feature_0", "feature_1"],
        similarity_config=RegimeSimilarityConfig(
            ae_enabled=False,
            min_candidate_rows=8,
            min_current_rows=8,
            knn_k=3,
            max_knn_current_rows=40,
            max_knn_candidate_rows=40,
        ),
        unsupervised_regime_artifact=_Artifact(),
    )

    unsup = diag["similarity"]["unsupervised_regime_learning"]
    assert unsup["used"] is False
    assert unsup["reason"] == "assessment_only_not_injected"
    assert "url_leaf_00" not in out.columns


def test_feature_engineering_artifact_selects_safe_regime_features():
    frame = _synthetic_regime_frame()
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    current_start = ts.max() - pd.Timedelta(days=21)
    current_mask = (ts >= current_start).to_numpy(dtype=bool)
    historical_mask = (ts < current_start).to_numpy(dtype=bool)

    artifact = build_regime_specialist_feature_engineering_artifact(
        frame,
        candidate_features=[
            "volume_regime_marker",
            "liquidity_rank",
            "volume_percentile",
            "feature_drift_psi_core",
            "prediction_distribution_drift",
            "feature_0",
        ],
        current_mask=current_mask,
        historical_mask=historical_mask,
        config=RegimeFeatureEngineeringConfig(
            univariate_folds=3,
            univariate_subsample_per_class=200,
            corr_subsample_per_class=200,
            relief_subsample_per_class=200,
            lgbm_enabled=False,
            elasticnet_enabled=False,
            max_pair_candidates=50,
        ),
    )

    assert artifact.selected_features
    assert "volume_regime_marker" in artifact.selected_features
    assert "liquidity_rank" in artifact.feature_report["feature"].astype(str).to_list()
    assert "feature_drift_psi_core" not in artifact.selected_features
    assert "prediction_distribution_drift" not in artifact.selected_features
    assert artifact.row_scores["regime_domain_current_likeness"].between(0.0, 1.0).all()
    assert not artifact.materialized_features.empty
    assert "generated_drift" in artifact.materialized_feature_groups
    assert "score" in artifact.materialized_feature_groups
    assert all(col.startswith("fe_score__") for col in artifact.materialized_feature_groups["score"])
    assert artifact.diagnostics["selected_feature_count"] <= 40
    assert artifact.diagnostics["drift"]["available"] is True
    assert artifact.diagnostics["drift"]["feature_count"] > 0
    assert artifact.diagnostics["drift"]["compute_limits"]["drift_knn_max_rows"] > 0
    assert artifact.diagnostics["domain_score_smoothing"]["enabled"] is True
    assert artifact.diagnostics["selected_drift_feature_count"] == 0
    assert artifact.diagnostics["current_relative_drift_discriminator_policy"][
        "included_in_discriminator"
    ] is False
    assert artifact.diagnostics["validation"]["drift"]["available"] is False
    assert artifact.diagnostics["validation"]["raw_plus_drift"]["available"] is False
    assert artifact.materialized_feature_groups["generated_drift"]


def test_feature_engineering_knn_distance_is_chunked_and_finite():
    rng = np.random.default_rng(7)
    a = rng.normal(size=(60, 6)).astype(np.float32)
    b = rng.normal(size=(70, 6)).astype(np.float32)
    distance = _mean_nearest_distance(
        a,
        b,
        1e-12,
        max_rows=25,
        chunk_pairs=200,
    )

    assert np.isfinite(distance)
    assert distance >= 0.0


def test_feature_engineering_auc_lift_handles_ties_and_safe_ks_names():
    assert _auc_lift(np.ones(4), np.asarray([0, 0, 1, 1])) == pytest.approx(0.0)

    pool = _candidate_pool(
        pd.DataFrame(
            {
                "spread_ticks": [1.0, 2.0],
                "feature_ks_core": [0.1, 0.2],
                "raw_state_knn_distance": [0.3, 0.4],
                "state_reconstruction_error": [0.5, 0.6],
            }
        ),
        ["spread_ticks", "feature_ks_core", "raw_state_knn_distance", "state_reconstruction_error"],
    )
    assert "spread_ticks" in pool
    assert "feature_ks_core" not in pool
    assert "raw_state_knn_distance" not in pool
    assert "state_reconstruction_error" not in pool


def test_feature_engineering_cv_disables_scores_when_grouped_folds_are_invalid():
    rng = np.random.default_rng(123)
    y = np.asarray([0] * 10 + [1] * 10, dtype=bool)
    matrix = rng.normal(size=(20, 3)).astype(np.float32)
    groups = np.asarray([0] * 10 + [1] * 10)
    scores, rank, diag = _cv_rank_and_scores(
        matrix,
        ["a", "b", "c"],
        y,
        np.ones(len(y), dtype=bool),
        groups,
        kind="elasticnet",
        max_per_class=20,
        config=RegimeFeatureEngineeringConfig(grouped_cv_folds=2, grouped_cv_repeats=1),
    )

    assert diag["enabled"] is False
    assert diag["reason"] == "no_valid_grouped_cv_folds"
    assert rank == {}
    assert np.allclose(scores, 0.5)


def test_feature_engineering_single_active_model_is_not_shrunk_to_neutral():
    frame = _synthetic_regime_frame()
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    current_start = ts.max() - pd.Timedelta(days=21)
    current_mask = (ts >= current_start).to_numpy(dtype=bool)
    historical_mask = (ts < current_start).to_numpy(dtype=bool)

    artifact = build_regime_specialist_feature_engineering_artifact(
        frame,
        candidate_features=[
            "volume_regime_marker",
            "liquidity_rank",
            "volume_percentile",
            "feature_0",
            "feature_1",
        ],
        current_mask=current_mask,
        historical_mask=historical_mask,
        config=RegimeFeatureEngineeringConfig(
            univariate_folds=3,
            univariate_subsample_per_class=200,
            corr_subsample_per_class=200,
            relief_subsample_per_class=200,
            lgbm_enabled=False,
            elasticnet_enabled=True,
            grouped_cv_folds=3,
            grouped_cv_repeats=1,
            elasticnet_max_samples_per_class=300,
            max_pair_candidates=20,
        ),
    )

    assert artifact.diagnostics["elasticnet"]["enabled"] is True
    assert np.allclose(
        artifact.row_scores["regime_domain_current_likeness_raw"].to_numpy(),
        artifact.row_scores["regime_elasticnet_current_likeness"].to_numpy(),
    )


def test_feature_engineering_domain_scores_are_timestamp_aggregated_and_smoothed():
    frame = _synthetic_regime_frame()
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    current_start = ts.max() - pd.Timedelta(days=21)
    current_mask = (ts >= current_start).to_numpy(dtype=bool)
    historical_mask = (ts < current_start).to_numpy(dtype=bool)

    artifact = build_regime_specialist_feature_engineering_artifact(
        frame,
        candidate_features=[
            "volume_regime_marker",
            "liquidity_rank",
            "volume_percentile",
            "feature_0",
            "feature_1",
        ],
        current_mask=current_mask,
        historical_mask=historical_mask,
        config=RegimeFeatureEngineeringConfig(
            univariate_folds=3,
            univariate_subsample_per_class=200,
            corr_subsample_per_class=200,
            relief_subsample_per_class=200,
            lgbm_enabled=False,
            elasticnet_enabled=True,
            grouped_cv_folds=3,
            grouped_cv_repeats=1,
            elasticnet_max_samples_per_class=300,
            max_pair_candidates=20,
            domain_score_ewma_half_life_days=1.0,
            domain_score_ewma_max_days=4.0,
        ),
    )
    scores = frame[["timestamp"]].join(artifact.row_scores)

    assert artifact.diagnostics["domain_score_smoothing"]["enabled"] is True
    assert artifact.diagnostics["domain_score_smoothing"]["aggregation"] == "timestamp_mean"
    assert artifact.diagnostics["domain_score_smoothing"]["max_days"] == pytest.approx(4.0)
    assert "regime_domain_current_likeness_raw" in artifact.row_scores
    assert "regime_domain_current_likeness_timestamp_mean" in artifact.row_scores
    assert "regime_domain_current_likeness_ewma" in artifact.row_scores
    assert scores.groupby("timestamp")["regime_domain_current_likeness"].nunique().max() == 1


def test_similarity_can_use_feature_engineering_domain_block():
    frame = _synthetic_regime_frame()
    out, diag = build_regime_specialist_training_frame(
        frame,
        selected_feature_columns=["feature_0", "feature_1", "feature_2", "feature_3"],
        similarity_config=RegimeSimilarityConfig(
            ae_enabled=False,
            feature_engineering_enabled=True,
            feature_engineering_lgbm_enabled=False,
            feature_engineering_elasticnet_enabled=True,
            feature_engineering_univariate_subsample_per_class=200,
            feature_engineering_max_pair_candidates=50,
            min_candidate_rows=8,
            min_current_rows=8,
            knn_k=3,
            max_knn_current_rows=40,
            max_knn_candidate_rows=40,
        ),
        weight_config=SpecialistWeightConfig(
            tau_current=10.0,
            tau_analogue=10.0,
            tau_normal=10.0,
            tau_irrelevant=10.0,
        ),
    )

    fe_diag = diag["similarity"]["feature_engineering"]
    assert fe_diag["enabled"] is True
    assert fe_diag["used"] is True
    assert fe_diag["selected_feature_count"] > 0
    assert fe_diag["classifier_score_used"] is True
    assert fe_diag["diagnostics"]["elasticnet"]["oof_rows"] > 0
    assert fe_diag["diagnostics"]["elasticnet"]["one_se"]["enabled"] is True
    assert fe_diag["diagnostics"]["elasticnet"]["compute_limits"]["max_permutation_rows"] > 0
    assert "fold_presence" in fe_diag["diagnostics"]["elasticnet"]["initial_rank"]
    assert "permutation" in fe_diag["diagnostics"]["elasticnet"]["initial_rank"]
    assert fe_diag["diagnostics"]["drift"]["available"] is True
    assert fe_diag["diagnostics"]["selected_drift_feature_count"] == 0
    assert fe_diag["diagnostics"]["current_relative_drift_discriminator_policy"][
        "included_in_discriminator"
    ] is False
    usage = fe_diag["materialized_feature_usage"]
    assert usage["generated_drift_to_drift_only"]
    assert set(usage["generated_drift_to_drift_only"]).issubset(set(usage["excluded_from_knn"]))
    assert set(usage["score_to_domain_classifier_only"]).issubset(set(usage["excluded_from_knn"]))
    knn_columns = set(diag["similarity"]["columns"]["knn"])
    assert not (knn_columns & set(usage["generated_drift_to_drift_only"]))
    assert not (knn_columns & set(usage["score_to_domain_classifier_only"]))
    assert diag["similarity"]["weights"]["domain_classifier_distance"] == pytest.approx(0.10)
    assert "domain_classifier_distance_median" in diag["similarity"]["block_scaling"]
    assert out["similarity_to_current"].between(0.0, 1.0).all()


def test_similarity_uses_domain_classifier_when_no_raw_feature_selected(monkeypatch):
    import extreme_price_movements.regime_specialist_feature_engineering as fe_module

    frame = _synthetic_regime_frame()

    def fake_build(work, **kwargs):
        n = len(work)
        scores = pd.DataFrame(
            {
                "regime_domain_current_likeness": np.linspace(0.2, 0.8, n),
            },
            index=work.index,
        )
        materialized = pd.DataFrame(
            {
                "fe_raw__raw_state_knn_distance": np.linspace(0.1, 0.9, n),
                "fe_pair__duo_feature_0_product_feature_1": np.linspace(-1.0, 1.0, n),
                "fe_pair__duo_feature_0_distance_feature_1": np.linspace(1.0, 2.0, n),
                "fe_drift__drift_ks_mean": np.linspace(0.0, 1.0, n),
                "fe_score__regime_domain_current_likeness": scores["regime_domain_current_likeness"].to_numpy(),
            },
            index=work.index,
        )
        return type(
            "FakeArtifact",
            (),
            {
                "selected_features": ["duo__feature_0__product__feature_1", "drift_ks_mean"],
                "selected_raw_features": [],
                "selected_pair_features": ["duo__feature_0__product__feature_1"],
                "selected_drift_features": ["drift_ks_mean"],
                "row_scores": scores,
                "materialized_features": materialized,
                "materialized_feature_groups": {
                    "raw_state": ["fe_raw__raw_state_knn_distance"],
                    "pair_geometry": [
                        "fe_pair__duo_feature_0_product_feature_1",
                        "fe_pair__duo_feature_0_distance_feature_1",
                    ],
                    "generated_drift": ["fe_drift__drift_ks_mean"],
                    "score": ["fe_score__regime_domain_current_likeness"],
                },
                "lgbm_features": ["drift_ks_mean"],
                "elasticnet_features": [],
                "diagnostics": {
                    "lgbm": {"enabled": True},
                    "elasticnet": {"enabled": False},
                    "selected_drift_feature_count": 1,
                },
            },
        )()

    monkeypatch.setattr(fe_module, "build_regime_specialist_feature_engineering_artifact", fake_build)
    out, diag = build_regime_specialist_training_frame(
        frame,
        selected_feature_columns=["feature_0", "feature_1", "feature_2", "feature_3"],
        similarity_config=RegimeSimilarityConfig(
            ae_enabled=False,
            feature_engineering_enabled=True,
            min_candidate_rows=8,
            min_current_rows=8,
            knn_k=3,
            max_knn_current_rows=40,
            max_knn_candidate_rows=40,
        ),
        weight_config=SpecialistWeightConfig(
            tau_current=10.0,
            tau_analogue=10.0,
            tau_normal=10.0,
            tau_irrelevant=10.0,
        ),
    )

    fe_diag = diag["similarity"]["feature_engineering"]
    assert fe_diag["used"] is True
    assert fe_diag["classifier_score_used"] is True
    assert fe_diag["selected_raw_feature_count"] == 0
    usage = fe_diag["materialized_feature_usage"]
    assert "fe_pair__duo_feature_0_product_feature_1" in diag["similarity"]["columns"]["knn"]
    assert "fe_pair__duo_feature_0_product_feature_1" in diag["similarity"]["columns"]["covariance"]
    assert "fe_raw__raw_state_knn_distance" not in diag["similarity"]["columns"]["covariance"]
    assert "fe_pair__duo_feature_0_distance_feature_1" not in diag["similarity"]["columns"]["covariance"]
    assert "fe_raw__raw_state_knn_distance" not in diag["similarity"]["columns"]["knn"]
    assert "fe_pair__duo_feature_0_distance_feature_1" not in diag["similarity"]["columns"]["knn"]
    assert "fe_drift__drift_ks_mean" in diag["similarity"]["columns"]["drift"]
    assert "fe_drift__drift_ks_mean" not in diag["similarity"]["columns"]["knn"]
    assert "fe_score__regime_domain_current_likeness" not in diag["similarity"]["columns"]["knn"]
    assert "fe_raw__raw_state_knn_distance" in usage["raw_state_excluded_from_knn"]
    assert "fe_pair__duo_feature_0_distance_feature_1" in usage["pair_geometry_excluded_from_knn"]
    assert "fe_raw__raw_state_knn_distance" in usage["raw_state_excluded_from_state_blocks"]
    assert "fe_pair__duo_feature_0_distance_feature_1" in usage["pair_geometry_excluded_from_state_blocks"]
    assert set(usage["raw_state_excluded_from_knn"]).issubset(set(usage["excluded_from_knn"]))
    assert set(usage["pair_geometry_excluded_from_knn"]).issubset(set(usage["excluded_from_knn"]))
    assert set(usage["generated_drift_to_drift_only"]).issubset(set(usage["excluded_from_knn"]))
    assert set(usage["score_to_domain_classifier_only"]).issubset(set(usage["excluded_from_knn"]))
    assert diag["similarity"]["weights"]["domain_classifier_distance"] == pytest.approx(0.10)
    assert out["similarity_to_current"].between(0.0, 1.0).all()


def test_feature_engineering_uses_global_assessment_but_returns_local_training_rows(monkeypatch):
    import extreme_price_movements.regime_specialist_feature_engineering as fe_module

    global_frame = _synthetic_regime_frame().copy()
    global_frame["strategy_id"] = np.where(np.arange(len(global_frame)) % 2 == 0, "strategy_a", "strategy_b")
    local_frame = global_frame.loc[global_frame["strategy_id"] == "strategy_a"].copy()
    captured: dict[str, object] = {}

    def fake_build(work, **kwargs):
        captured["rows"] = len(work)
        captured["strategy_ids"] = sorted(set(work.get("strategy_id", pd.Series([], dtype=str)).astype(str)))
        n = len(work)
        scores = pd.DataFrame(
            {"regime_domain_current_likeness": np.linspace(0.2, 0.8, n, dtype=np.float32)},
            index=work.index,
        )
        return type(
            "FakeArtifact",
            (),
            {
                "selected_features": ["feature_0"],
                "selected_raw_features": ["feature_0"],
                "selected_pair_features": [],
                "selected_drift_features": [],
                "row_scores": scores,
                "materialized_features": pd.DataFrame(index=work.index),
                "materialized_feature_groups": {
                    "raw_state": [],
                    "pair_geometry": [],
                    "generated_drift": [],
                    "score": [],
                },
                "lgbm_features": ["feature_0"],
                "elasticnet_features": [],
                "diagnostics": {
                    "lgbm": {"enabled": True},
                    "elasticnet": {"enabled": False},
                    "selected_drift_feature_count": 0,
                },
            },
        )()

    monkeypatch.setattr(fe_module, "build_regime_specialist_feature_engineering_artifact", fake_build)
    out, diag = build_regime_specialist_training_frame(
        local_frame,
        selected_feature_columns=["feature_0", "feature_1", "feature_2", "feature_3"],
        similarity_config=RegimeSimilarityConfig(
            ae_enabled=False,
            feature_engineering_enabled=True,
            min_candidate_rows=8,
            min_current_rows=8,
            knn_k=3,
            max_knn_current_rows=40,
            max_knn_candidate_rows=40,
        ),
        weight_config=SpecialistWeightConfig(
            tau_current=10.0,
            tau_analogue=10.0,
            tau_normal=10.0,
            tau_irrelevant=10.0,
        ),
        assessment_frame=global_frame,
    )

    sim_diag = diag["similarity"]
    assert captured["rows"] == len(global_frame)
    assert captured["strategy_ids"] == ["strategy_a", "strategy_b"]
    assert sim_diag["assessment_scope"]["mode"] == "global_assessment_local_training"
    assert sim_diag["assessment_scope"]["assessment_rows"] == len(global_frame)
    assert sim_diag["assessment_scope"]["local_training_rows"] == len(local_frame)
    assert out.index.equals(local_frame.index)
    assert len(out) == len(local_frame)
    assert len(out["regime_specialist_sample_weight"]) == len(local_frame)


def test_global_assessment_alignment_prefers_timestamp_symbol_after_local_reset_index():
    global_frame = _synthetic_regime_frame()
    local_frame = global_frame.loc[global_frame["symbol"] == "BTC/USD:USD"].reset_index(drop=True)
    out, diag = build_regime_specialist_training_frame(
        local_frame,
        selected_feature_columns=["feature_0", "feature_1", "feature_2", "feature_3"],
        similarity_config=RegimeSimilarityConfig(
            ae_enabled=False,
            min_candidate_rows=8,
            min_current_rows=8,
            knn_k=3,
            max_knn_current_rows=40,
            max_knn_candidate_rows=40,
        ),
        weight_config=SpecialistWeightConfig(
            tau_current=10.0,
            tau_analogue=10.0,
            tau_normal=10.0,
            tau_irrelevant=10.0,
        ),
        assessment_frame=global_frame,
    )

    scope = diag["similarity"]["assessment_scope"]
    assert scope["alignment"] == "timestamp_symbol"
    assert scope["alignment_ok"] is True
    assert scope["aligned_rows"] == len(local_frame)
    assert scope["aligned_fraction"] == pytest.approx(1.0)
    assert out.index.equals(local_frame.index)


def test_global_assessment_alignment_failure_disables_specialist_weights():
    global_frame = _synthetic_regime_frame()
    local_frame = global_frame.loc[global_frame["symbol"] == "BTC/USD:USD"].reset_index(drop=True)
    local_frame["symbol"] = "NO_MATCH"
    out, diag = build_regime_specialist_training_frame(
        local_frame,
        selected_feature_columns=["feature_0", "feature_1", "feature_2", "feature_3"],
        similarity_config=RegimeSimilarityConfig(
            ae_enabled=False,
            min_candidate_rows=8,
            min_current_rows=8,
            knn_k=3,
            max_knn_current_rows=40,
            max_knn_candidate_rows=40,
        ),
        weight_config=SpecialistWeightConfig(
            tau_current=10.0,
            tau_analogue=10.0,
            tau_normal=10.0,
            tau_irrelevant=10.0,
        ),
        assessment_frame=global_frame,
    )

    scope = diag["similarity"]["assessment_scope"]
    assert diag["similarity"]["enabled"] is False
    assert scope["alignment_ok"] is False
    assert scope["aligned_rows"] == 0
    assert diag["sample_weight"]["should_train_specialist"] is False
    assert np.allclose(out["regime_specialist_sample_weight"].to_numpy(dtype=np.float32), 1.0)


def test_current_end_excludes_future_rows_from_similarity_and_weights():
    frame = _synthetic_regime_frame()
    current_end = frame["timestamp"].max() - pd.Timedelta(days=10)
    out, diag = build_regime_specialist_training_frame(
        frame,
        current_end=current_end,
        selected_feature_columns=["feature_0", "feature_1", "feature_2", "feature_3"],
        similarity_config=RegimeSimilarityConfig(
            ae_enabled=False,
            min_candidate_rows=8,
            min_current_rows=8,
            knn_k=3,
            max_knn_current_rows=40,
            max_knn_candidate_rows=40,
        ),
        weight_config=SpecialistWeightConfig(
            tau_current=10.0,
            tau_analogue=10.0,
            tau_normal=10.0,
            tau_irrelevant=10.0,
        ),
    )
    future = frame["timestamp"] > current_end
    assert diag["similarity"]["future_excluded_rows"] == int(future.sum())
    assert (out.loc[future, "regime_specialist_bucket"] == "future_excluded").all()
    assert out.loc[future, "similarity_to_current"].eq(0.0).all()
    assert out.loc[future, "regime_specialist_sample_weight"].eq(0.0).all()
    active = ~future
    assert out.loc[active, "regime_specialist_sample_weight"].mean() == pytest.approx(1.0)


def test_disabled_similarity_returns_non_trainable_neutral_weights():
    frame = _synthetic_regime_frame().tail(6).copy()
    out, diag = build_regime_specialist_training_frame(
        frame,
        selected_feature_columns=["feature_0", "feature_1"],
        similarity_config=RegimeSimilarityConfig(
            min_current_rows=100,
            min_candidate_rows=100,
            ae_enabled=False,
        ),
    )
    assert diag["similarity"]["enabled"] is False
    assert diag["sample_weight"]["should_train_specialist"] is False
    assert out["regime_specialist_sample_weight"].mean() == pytest.approx(1.0)


def test_invalid_timestamp_rows_do_not_create_nan_buckets():
    frame = _synthetic_regime_frame()
    invalid = frame.iloc[[0]].copy()
    invalid.index = pd.Index([999_999])
    invalid["timestamp"] = "not-a-timestamp"
    frame = pd.concat([frame, invalid], axis=0)

    out, diag = build_regime_specialist_training_frame(
        frame,
        selected_feature_columns=["feature_0", "feature_1", "feature_2", "feature_3"],
        similarity_config=RegimeSimilarityConfig(
            ae_enabled=False,
            min_candidate_rows=8,
            min_current_rows=8,
            knn_k=3,
            max_knn_current_rows=40,
            max_knn_candidate_rows=40,
        ),
        weight_config=SpecialistWeightConfig(
            tau_current=10.0,
            tau_analogue=10.0,
            tau_normal=10.0,
            tau_irrelevant=10.0,
        ),
    )

    assert diag["similarity"]["enabled"] is True
    assert out.loc[999_999, "regime_specialist_bucket"] == "irrelevant"
    assert out.loc[999_999, "similarity_to_current"] == pytest.approx(0.0)
    assert out["regime_specialist_bucket"].notna().all()
    assert np.isfinite(out["regime_specialist_sample_weight"].to_numpy()).all()


def test_insufficient_current_rows_do_not_mark_history_as_perfect_similarity():
    frame = _synthetic_regime_frame()
    out, diag = compute_regime_similarity_to_current(
        frame,
        selected_feature_columns=["feature_0", "feature_1", "feature_2", "feature_3"],
        config=RegimeSimilarityConfig(
            min_current_rows=10_000,
            min_candidate_rows=8,
            ae_enabled=False,
        ),
    )
    current = out["regime_specialist_bucket"] == "current"
    normal = out["regime_specialist_bucket"] == "normal"

    assert diag["similarity_unavailable"] is True
    assert current.any()
    assert normal.any()
    assert out.loc[current, "similarity_to_current"].eq(1.0).all()
    assert out.loc[normal, "similarity_to_current"].eq(0.0).all()
    assert out.loc[normal, "day_similarity_available"].eq(False).all()


def test_day_similarity_unavailable_is_explicit_and_neutral_for_multiplier():
    frame = _synthetic_regime_frame()
    out, diag = build_regime_specialist_training_frame(
        frame,
        selected_feature_columns=["feature_0", "feature_1", "feature_2", "feature_3"],
        similarity_config=RegimeSimilarityConfig(
            ae_enabled=False,
            day_similarity_strength=0.75,
            day_similarity_min_rows=10_000,
            min_candidate_rows=8,
            min_current_rows=8,
            knn_k=3,
            max_knn_current_rows=40,
            max_knn_candidate_rows=40,
        ),
        weight_config=SpecialistWeightConfig(
            tau_current=10.0,
            tau_analogue=10.0,
            tau_normal=10.0,
            tau_irrelevant=10.0,
        ),
    )
    historical = out["regime_specialist_bucket"] != "current"
    active_historical = historical & (out["regime_specialist_bucket"] != "future_excluded")

    assert diag["similarity"]["day_similarity"]["available_rows"] == int((~historical).sum())
    assert out.loc[active_historical, "day_similarity_available"].eq(False).all()
    assert out.loc[active_historical, "day_similarity"].eq(0.0).all()
    assert np.allclose(
        out.loc[active_historical, "similarity_to_current"].to_numpy(),
        out.loc[active_historical, "window_similarity"].to_numpy(),
    )


def test_label_horizon_excludes_rows_that_overlap_current_boundary():
    frame = _synthetic_regime_frame()
    current_end = frame["timestamp"].max() - pd.Timedelta(days=10)
    out, diag = compute_regime_similarity_to_current(
        frame,
        current_end=current_end,
        selected_feature_columns=["feature_0", "feature_1", "feature_2", "feature_3"],
        config=RegimeSimilarityConfig(
            label_horizon_hours=48.0,
            ae_enabled=False,
            min_candidate_rows=8,
            min_current_rows=8,
            knn_k=3,
            max_knn_current_rows=40,
            max_knn_candidate_rows=40,
        ),
    )
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    overlap = (ts <= current_end) & (ts + pd.Timedelta(hours=48) > current_end)

    assert diag["label_end_source"] == "label_horizon_hours"
    assert diag["label_overlap_excluded_rows"] == int(overlap.sum())
    assert out.loc[overlap, "regime_specialist_bucket"].eq("irrelevant").all()
    assert out.loc[overlap, "similarity_to_current"].eq(0.0).all()


def test_similarity_output_preserves_original_index():
    frame = _synthetic_regime_frame().iloc[:80].copy()
    frame.index = pd.Index(np.arange(1000, 1000 + len(frame)), name="row_id")
    sim, _diag = compute_regime_similarity_to_current(
        frame,
        selected_feature_columns=["feature_0", "feature_1", "feature_2", "feature_3"],
        config=RegimeSimilarityConfig(
            ae_enabled=False,
            min_candidate_rows=8,
            min_current_rows=8,
        ),
    )
    assert sim.index.equals(frame.index)


def test_explicit_column_diagnostics_and_asset_covariance():
    frame = _synthetic_regime_frame()
    out, diag = build_regime_specialist_training_frame(
        frame,
        selected_feature_columns=["feature_0", "feature_1", "feature_2", "feature_3"],
        market_columns=["volatility_percentile", "missing_market_feature"],
        asset_return_col="return_1h",
        similarity_config=RegimeSimilarityConfig(
            ae_enabled=True,
            ae_min_windows=99,
            min_candidate_rows=8,
            min_current_rows=8,
            knn_k=3,
            max_knn_current_rows=40,
            max_knn_candidate_rows=40,
        ),
        weight_config=SpecialistWeightConfig(
            tau_current=10.0,
            tau_analogue=10.0,
            tau_normal=10.0,
            tau_irrelevant=10.0,
        ),
    )

    sim_diag = diag["similarity"]
    assert out["similarity_to_current"].between(0.0, 1.0).all()
    assert sim_diag["column_selection"]["market"]["source"] == "explicit"
    assert sim_diag["column_selection"]["market"]["missing_requested"] == [
        "missing_market_feature"
    ]
    assert sim_diag["asset_covariance"]["enabled"] is True
    assert sim_diag["asset_covariance"]["return_col"] == "return_1h"
    assert sim_diag["asset_covariance"]["max_assets"] == 100
    assert sim_diag["asset_covariance"]["shrinkage"] == pytest.approx(0.10)
    assert sim_diag["autoencoder"]["used"] is False
    assert sim_diag["autoencoder"]["reason"] == "insufficient_candidate_windows"


def test_day_similarity_is_blended_not_a_hard_multiplier():
    frame = _synthetic_regime_frame()
    strength = 0.35
    out, _diag = build_regime_specialist_training_frame(
        frame,
        selected_feature_columns=["feature_0", "feature_1", "feature_2", "feature_3"],
        similarity_config=RegimeSimilarityConfig(
            ae_enabled=False,
            day_similarity_strength=strength,
            day_similarity_min_rows=2,
            min_candidate_rows=8,
            min_current_rows=8,
            knn_k=3,
            max_knn_current_rows=40,
            max_knn_candidate_rows=40,
        ),
        weight_config=SpecialistWeightConfig(
            tau_current=10.0,
            tau_analogue=10.0,
            tau_normal=10.0,
            tau_irrelevant=10.0,
        ),
    )

    historical = out["regime_specialist_bucket"] != "current"
    floor = out.loc[historical, "window_similarity"] * (1.0 - strength)
    assert (out.loc[historical, "similarity_to_current"] + 1e-6 >= floor).all()


def test_low_memory_output_and_capped_window_diagnostics():
    frame = _synthetic_regime_frame()
    out, diag = build_regime_specialist_training_frame(
        frame,
        selected_feature_columns=["feature_0", "feature_1", "feature_2", "feature_3"],
        include_input_columns=False,
        similarity_config=RegimeSimilarityConfig(
            ae_enabled=False,
            candidate_window_days=5,
            max_window_diagnostics=1,
            min_candidate_rows=4,
            min_current_rows=8,
            knn_k=3,
            max_knn_current_rows=40,
            max_knn_historical_rows=80,
        ),
        weight_config=SpecialistWeightConfig(
            tau_current=10.0,
            tau_analogue=10.0,
            tau_normal=10.0,
            tau_irrelevant=10.0,
        ),
    )

    assert "feature_0" not in out.columns
    assert "regime_specialist_sample_weight" in out.columns
    assert diag["weighted_drift_baseline"]["enabled"] is True
    assert diag["similarity"]["window_diagnostics_count"] >= len(
        diag["similarity"]["window_diagnostics"]
    )
    assert len(diag["similarity"]["window_diagnostics"]) <= 1


def test_lgbm_regime_specialist_shadow_and_active_weight_hooks():
    from extreme_price_movements import lgbm_pipeline as lp

    frame = _synthetic_regime_frame()
    selected = ["feature_0", "feature_1", "feature_2", "feature_3", "return_1h"]
    base_cfg = {
        "lgbm_regime_specialist_enabled": True,
        "lgbm_regime_specialist_objectives": ["train_base", "train_meta"],
        "lgbm_regime_specialist_ae_enabled": False,
        "lgbm_regime_specialist_min_candidate_rows": 8,
        "lgbm_regime_specialist_min_current_rows": 8,
        "lgbm_regime_specialist_knn_k": 3,
        "lgbm_regime_specialist_max_knn_current_rows": 40,
        "lgbm_regime_specialist_max_knn_historical_rows": 80,
        "lgbm_regime_specialist_tau_current": 1.0,
        "lgbm_regime_specialist_tau_analogue": 1.0,
        "lgbm_regime_specialist_tau_normal": 1.0,
        "lgbm_regime_specialist_tau_irrelevant": 1.0,
    }
    shadow_bundle = lp._build_lgbm_regime_specialist_bundle(
        frame,
        selected,
        timestamps=frame["timestamp"].to_numpy(),
        assets=frame["symbol"].to_numpy(),
        objective_mode="train_base",
        cfg={**base_cfg, "lgbm_regime_specialist_shadow_only": True},
        random_state=7,
        label="unit",
    )
    assert shadow_bundle["metrics"]["regime_specialist_enabled"] is True
    assert shadow_bundle["metrics"]["regime_specialist_sample_weight_applied"] is False
    unchanged, shadow_diag = lp._apply_lgbm_regime_specialist_weights(
        np.ones(len(frame), dtype=np.float32),
        shadow_bundle,
    )
    assert shadow_diag["applied"] is False
    assert unchanged.mean() == pytest.approx(1.0)

    active_bundle = lp._build_lgbm_regime_specialist_bundle(
        frame,
        selected,
        timestamps=frame["timestamp"].to_numpy(),
        assets=frame["symbol"].to_numpy(),
        objective_mode="train_meta",
        cfg={
            **base_cfg,
            "lgbm_regime_specialist_shadow_only": False,
            "lgbm_regime_specialist_apply_sample_weight": True,
            "lgbm_regime_specialist_apply_distillation_shrink": True,
        },
        random_state=11,
        label="unit",
    )
    adjusted, active_diag = lp._apply_lgbm_regime_specialist_weights(
        np.ones(len(frame), dtype=np.float32),
        active_bundle,
    )
    assert active_bundle["apply_sample_weight"] is True
    assert active_bundle["apply_distillation_shrink"] is True
    assert active_bundle["metrics"]["regime_specialist_weighted_drift_baseline_enabled"] is True
    assert active_bundle["metrics"]["regime_specialist_weighted_drift_baseline_feature_count"] > 0
    assert active_diag["applied"] is True
    assert adjusted.mean() == pytest.approx(1.0)
    wide_base_weight = np.geomspace(0.05, 20.0, len(frame)).astype(np.float32)
    adjusted_wide, wide_diag = lp._apply_lgbm_regime_specialist_weights(
        wide_base_weight,
        active_bundle,
    )
    assert wide_diag["applied"] is True
    assert wide_diag["base_weight_preconditioned_policy"] == "unit_mean_compress_0.7_1.3"
    assert wide_diag["base_weight_preconditioned_min"] >= 0.7 - 1e-6
    assert wide_diag["base_weight_preconditioned_max"] <= 1.3 + 1e-6
    assert wide_diag["base_weight_preconditioned_mean"] == pytest.approx(1.0, abs=1e-6)
    assert adjusted_wide.mean() == pytest.approx(1.0)
    sim = lp._lgbm_regime_specialist_similarity_for_idx(active_bundle)
    assert sim is not None
    assert len(sim) == len(frame)
    current_metrics = lp._lgbm_regime_specialist_current_metrics(
        np.asarray([0, 1] * (len(frame) // 2), dtype=np.float32),
        np.linspace(0.0, 1.0, len(frame), dtype=np.float32),
        active_bundle,
        classifier=True,
    )
    assert current_metrics["current_regime_metrics_available"] is True
    assert current_metrics["current_regime_metric_rows"] > 0

    global_frame = frame.copy()
    global_frame["strategy_id"] = np.where(np.arange(len(global_frame)) % 2 == 0, "strategy_a", "strategy_b")
    local_frame = global_frame.loc[global_frame["strategy_id"] == "strategy_a"].copy()
    global_assessment_bundle = lp._build_lgbm_regime_specialist_bundle(
        local_frame,
        selected,
        timestamps=local_frame["timestamp"].to_numpy(),
        assets=local_frame["symbol"].to_numpy(),
        assessment_X_df=global_frame,
        assessment_timestamps=global_frame["timestamp"].to_numpy(),
        assessment_assets=global_frame["symbol"].to_numpy(),
        objective_mode="train_base",
        cfg={**base_cfg, "lgbm_regime_specialist_shadow_only": True},
        random_state=13,
        label="unit_global_assessment",
    )
    assert len(global_assessment_bundle["weights"]) == len(local_frame)
    assert global_assessment_bundle["metrics"]["regime_specialist_assessment_mode"] == "global_assessment_local_training"
    assert global_assessment_bundle["metrics"]["regime_specialist_assessment_rows"] == len(global_frame)
    assert global_assessment_bundle["metrics"]["regime_specialist_local_training_rows"] == len(local_frame)
    assert global_assessment_bundle["metrics"]["regime_specialist_assessment_alignment"] == "timestamp_symbol"
    assert global_assessment_bundle["metrics"]["regime_specialist_assessment_alignment_ok"] is True
    assert "current_regime_precision10" in current_metrics


def test_lgbm_regime_specialist_disabled_still_builds_fe_diagnostics(monkeypatch):
    from extreme_price_movements import lgbm_pipeline as lp
    import extreme_price_movements.regime_specialist_feature_engineering as fe_module

    frame = _synthetic_regime_frame()
    selected = ["feature_0", "feature_1", "volume_regime_marker"]
    captured: dict[str, int] = {"calls": 0, "rows": 0}

    def fake_build(work, **kwargs):
        captured["calls"] += 1
        captured["rows"] = len(work)
        scores = pd.DataFrame(
            {
                "regime_domain_current_likeness": np.linspace(
                    0.2,
                    0.8,
                    len(work),
                    dtype=np.float32,
                )
            },
            index=work.index,
        )
        return type(
            "FakeFeatureEngineeringArtifact",
            (),
            {
                "schema_version": "test_fe_schema",
                "selected_features": ["volume_regime_marker"],
                "selected_raw_features": ["volume_regime_marker"],
                "selected_pair_features": [],
                "selected_drift_features": [],
                "lgbm_features": ["volume_regime_marker"],
                "elasticnet_features": ["feature_0"],
                "lgbm_feature_scores": {"volume_regime_marker": 0.7},
                "elasticnet_feature_scores": {"feature_0": 0.4},
                "final_feature_scores": {"volume_regime_marker": 0.6},
                "row_scores": scores,
                "materialized_features": pd.DataFrame(index=work.index),
                "materialized_feature_groups": {
                    "raw_state": [],
                    "pair_geometry": [],
                    "generated_drift": [],
                    "score": [],
                },
                "feature_report": pd.DataFrame(
                    {
                        "feature": ["volume_regime_marker"],
                        "univariate_score": [0.9],
                        "auc_lift_mean": [0.2],
                        "auc_lift_std": [0.01],
                        "ks_mean": [0.3],
                        "ks_std": [0.02],
                        "median_shift_mean": [0.5],
                        "median_shift_std": [0.03],
                        "sign_consistency": [1.0],
                        "fold_pass_rate": [1.0],
                        "selected_univariate": [True],
                    }
                ),
                "diagnostics": {
                    "selected_feature_count": 1,
                    "selected_raw_feature_count": 1,
                    "selected_pair_feature_count": 0,
                    "selected_drift_feature_count": 0,
                    "lgbm": {
                        "enabled": True,
                        "oof_rows": 123,
                        "fold_auc_lift_mean": 0.123,
                        "fold_auc_lift_std": 0.012,
                    },
                    "elasticnet": {
                        "enabled": True,
                        "oof_rows": 121,
                        "fold_auc_lift_mean": 0.111,
                        "fold_auc_lift_std": 0.011,
                    },
                    "validation": {
                        "enabled": True,
                        "raw": {"mean": 0.1, "std": 0.01, "folds": 3},
                        "drift": {"mean": 0.2, "std": 0.02, "folds": 3},
                        "raw_plus_drift": {"mean": 0.3, "std": 0.03, "folds": 3},
                    },
                },
            },
        )()

    monkeypatch.setattr(fe_module, "build_regime_specialist_feature_engineering_artifact", fake_build)

    cfg = {
        "lgbm_regime_specialist_enabled": False,
        "lgbm_regime_specialist_feature_engineering_diagnostics_enabled": True,
        "lgbm_regime_specialist_feature_engineering_diagnostics_final_only": True,
        "lgbm_regime_specialist_objectives": ["train_base", "train_meta"],
    }
    candidate_bundle = lp._build_lgbm_regime_specialist_bundle(
        frame,
        selected,
        timestamps=frame["timestamp"].to_numpy(),
        assets=frame["symbol"].to_numpy(),
        objective_mode="train_base",
        cfg=cfg,
        random_state=23,
        label="candidate",
    )
    assert captured["calls"] == 0
    assert candidate_bundle["metrics"]["regime_specialist_feature_engineering_reason"] == "deferred_to_final"

    final_bundle = lp._build_lgbm_regime_specialist_bundle(
        frame,
        selected,
        timestamps=frame["timestamp"].to_numpy(),
        assets=frame["symbol"].to_numpy(),
        objective_mode="train_base",
        cfg=cfg,
        random_state=23,
        label="final",
    )
    assert captured["calls"] == 1
    assert captured["rows"] == len(frame)
    assert final_bundle["enabled"] is False
    assert np.allclose(final_bundle["weights"], 1.0)
    assert final_bundle["metrics"]["regime_specialist_sample_weight_applied"] is False
    assert final_bundle["metrics"]["regime_specialist_feature_engineering_diagnostics_enabled"] is True
    assert final_bundle["metrics"]["regime_specialist_feature_engineering_diagnostic_only"] is True
    assert final_bundle["metrics"]["regime_specialist_feature_engineering_selected_feature_count"] == 1
    assert final_bundle["metrics"]["regime_specialist_feature_engineering_lgbm_fold_auc_lift_mean"] == pytest.approx(0.123)
    assert final_bundle["metrics"]["regime_specialist_feature_engineering_elasticnet_fold_auc_lift_mean"] == pytest.approx(0.111)
    assert final_bundle["metrics"]["regime_specialist_feature_engineering_validation_raw_plus_drift_auc_lift_mean"] == pytest.approx(0.3)
    fe_diag = final_bundle["diagnostics"]["feature_engineering"]
    assert fe_diag["diagnostic_only"] is True
    assert fe_diag["selected_features"] == ["volume_regime_marker"]
    assert fe_diag["top_final_features"][0]["feature"] == "volume_regime_marker"


def test_lgbm_regime_specialist_builds_unsupervised_artifact_when_enabled(monkeypatch):
    from extreme_price_movements import lgbm_pipeline as lp
    import extreme_price_movements.unsupervised_regime_learning.regime_models as regime_models

    frame = _synthetic_regime_frame()
    selected = ["feature_0", "feature_1", "volume_regime_marker"]
    lp._REGIME_SPECIALIST_UNSUPERVISED_CACHE.clear()
    captured: dict[str, object] = {"calls": 0, "features": []}

    def fake_fit(work, features, config):
        captured["calls"] = int(captured["calls"]) + 1
        captured["features"] = list(features)
        materialized = pd.DataFrame(
            {
                "url_leaf_00": np.linspace(0.0, 1.0, len(work), dtype=np.float32),
                "url_mfa_gamma_00": np.linspace(1.0, 0.0, len(work), dtype=np.float32),
            },
            index=work.index,
        )
        return type(
            "FakeAdvancedRegimeArtifact",
            (),
            {
                "schema_version": "unsupervised_regime_learning_v2",
                "selected_features": list(features),
                "specialist_candidate_features": ["url_leaf_00", "url_mfa_gamma_00"],
                "materialized_features": materialized,
                "diagnostics": {
                    "kept_methods": ["leaf_pca_bayesian_gmm", "mfa"],
                    "baseline_score": 0.25,
                },
            },
        )()

    monkeypatch.setattr(regime_models, "fit_advanced_regime_learning", fake_fit)

    bundle = lp._build_lgbm_regime_specialist_bundle(
        frame,
        selected,
        timestamps=frame["timestamp"].to_numpy(),
        assets=frame["symbol"].to_numpy(),
        objective_mode="train_base",
        cfg={
            "lgbm_regime_specialist_enabled": True,
            "lgbm_regime_specialist_shadow_only": True,
            "lgbm_regime_specialist_feature_engineering_enabled": False,
            "lgbm_regime_specialist_min_current_rows": 8,
            "lgbm_regime_specialist_min_candidate_rows": 8,
            "lgbm_regime_specialist_current_window_days": 21.0,
            "lgbm_regime_specialist_candidate_window_days": 14.0,
            "UNSUPERVISED_REGIME_LEARNING": {
                "regime_models": {
                    "enabled": True,
                    "selector_backend": "random_forest",
                }
            },
        },
        random_state=29,
        label="final",
    )

    assert captured["calls"] == 1
    assert "volume_regime_marker" in captured["features"]
    artifact_diag = bundle["diagnostics"]["unsupervised_regime_learning_artifact"]
    assert artifact_diag["enabled"] is True
    assert artifact_diag["used"] is False
    assert artifact_diag["reason"] == "assessment_only_not_injected"
    assert bundle["metrics"]["regime_specialist_unsupervised_regime_learning_enabled"] is True
    assert bundle["metrics"]["regime_specialist_unsupervised_regime_learning_used"] is False
    assert bundle["diagnostics"]["similarity"]["unsupervised_regime_learning"]["used"] is False


def test_unsupervised_regime_artifact_aligns_by_timestamp_symbol():
    from extreme_price_movements.unsupervised_regime_learning.regime_models import (
        augment_frame_with_regime_artifact,
    )

    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC"),
            "symbol": ["BTC", "ETH", "BTC"],
        },
        index=[100, 101, 102],
    )
    row_keys = frame[["timestamp", "symbol"]].copy()
    materialized = pd.DataFrame(
        {"url_leaf_00": [0.1, 0.2, 0.3]},
        index=[10, 11, 12],
    )
    artifact = type(
        "FakeAdvancedRegimeArtifact",
        (),
        {
            "schema_version": "unsupervised_regime_learning_v2",
            "row_keys": row_keys.set_index(materialized.index),
            "materialized_features": materialized,
            "specialist_candidate_features": ["url_leaf_00"],
        },
    )()

    augmented, candidates, diag = augment_frame_with_regime_artifact(frame, artifact)

    assert diag["used"] is True
    assert candidates == ["url_leaf_00"]
    assert augmented["url_leaf_00"].tolist() == pytest.approx([0.1, 0.2, 0.3])


def test_lgbm_unsupervised_cache_key_depends_on_feature_values(monkeypatch):
    from extreme_price_movements import lgbm_pipeline as lp
    import extreme_price_movements.unsupervised_regime_learning.regime_models as regime_models

    base = _synthetic_regime_frame().head(30).copy()
    changed = base.copy()
    changed["feature_0"] = changed["feature_0"] + 10.0
    selected = ["feature_0", "feature_1"]
    lp._REGIME_SPECIALIST_UNSUPERVISED_CACHE.clear()
    captured = {"calls": 0}

    def fake_fit(work, features, config):
        captured["calls"] += 1
        materialized = pd.DataFrame(
            {"url_leaf_00": np.full(len(work), float(captured["calls"]), dtype=np.float32)},
            index=work.index,
        )
        return type(
            "FakeAdvancedRegimeArtifact",
            (),
            {
                "schema_version": "unsupervised_regime_learning_v2",
                "selected_features": list(features),
                "specialist_candidate_features": ["url_leaf_00"],
                "materialized_features": materialized,
                "diagnostics": {"kept_methods": ["leaf_pca_bayesian_gmm"]},
            },
        )()

    monkeypatch.setattr(regime_models, "fit_advanced_regime_learning", fake_fit)
    cfg = {"UNSUPERVISED_REGIME_LEARNING": {"regime_models": {"enabled": "True"}}}

    _artifact1, diag1 = lp._lgbm_unsupervised_regime_artifact(
        base,
        selected,
        cfg=cfg,
        random_state=7,
    )
    _artifact2, diag2 = lp._lgbm_unsupervised_regime_artifact(
        changed,
        selected,
        cfg=cfg,
        random_state=7,
    )

    assert captured["calls"] == 2
    assert diag1["cache_hit"] is False
    assert diag2["cache_hit"] is False


def test_lgbm_unsupervised_string_false_disables_stage():
    from extreme_price_movements import lgbm_pipeline as lp

    active = lp._unsupervised_regime_learning_cfg(
        {"UNSUPERVISED_REGIME_LEARNING": {"regime_models": {"enabled": "False"}}}
    )

    assert active["regime_models"]["enabled"] is False


def test_lgbm_regime_specialist_distillation_shrink_hook():
    from extreme_price_movements import lgbm_pipeline as lp

    adjusted = lp._regime_specialist_shrink_weight_towards_one(
        np.asarray([2.0, 0.5, 3.0], dtype=np.float32),
        np.asarray([1.0, 0.0, 0.5], dtype=np.float32),
        cfg={"lgbm_regime_specialist_distillation_power": 1.0},
    )
    assert adjusted[0] == pytest.approx(2.0)
    assert adjusted[1] == pytest.approx(1.0)
    assert adjusted[2] == pytest.approx(2.0)


def test_specialist_sample_weights_cap_bucket_masses_and_normalize():
    df = pd.DataFrame(
        {
            "regime_specialist_bucket": (
                ["current"] * 20
                + ["analogue"] * 20
                + ["normal"] * 80
                + ["irrelevant"] * 80
            ),
            "similarity_to_current": (
                [1.0] * 20
                + [0.8] * 20
                + [0.3] * 80
                + [0.05] * 80
            ),
        }
    )
    weights, diag = compute_specialist_sample_weights(
        df,
        config=SpecialistWeightConfig(
            tau_current=1.0,
            tau_analogue=1.0,
            tau_normal=1.0,
            tau_irrelevant=1.0,
            min_weight=0.10,
            max_weight=2.00,
        ),
    )
    assert weights.mean() == pytest.approx(1.0)
    assert weights.min() >= 0.10 - 1e-6
    assert weights.max() <= 2.00 + 1e-6
    assert diag["less_interesting_mass"] <= diag["less_interesting_mass_cap"] + 1e-9
    assert 0.10 <= diag["less_interesting_mass_cap"] <= 0.50
    assert diag["current_mass"] + diag["analogue_mass"] >= 0.50 - 1e-9
    assert diag["adaptive_n_eff"] == pytest.approx(
        diag["effective_current"] + diag["effective_analogue"],
    )
    assert diag["replay_n_eff"] == pytest.approx(
        diag["effective_normal"] + diag["effective_irrelevant"],
    )
    assert 0.0 <= diag["adaptive_n_eff_reliability"] <= 1.0
    assert 0.0 <= diag["replay_n_eff_reliability"] <= 1.0
    assert diag["bucket_mass_basis"] == "n_eff_reliability"
    assert diag["should_train_specialist"] is True
    assert diag["bucket_mass_caps_enforced"] is True


def test_specialist_sample_weights_report_when_bucket_caps_cannot_be_enforced():
    df = pd.DataFrame(
        {
            "regime_specialist_bucket": ["normal"] * 20 + ["irrelevant"] * 20,
            "similarity_to_current": [0.3] * 20 + [0.0] * 20,
        }
    )
    weights, diag = compute_specialist_sample_weights(
        df,
        config=SpecialistWeightConfig(
            tau_current=1.0,
            tau_analogue=1.0,
            tau_normal=1.0,
            tau_irrelevant=1.0,
        ),
    )

    assert weights.mean() == pytest.approx(1.0)
    assert diag["should_train_specialist"] is False
    assert diag["bucket_mass_caps_enforced"] is False
    assert diag["bucket_mass_cap_reason"] == "no_current_or_analogue_rows"


def test_specialist_sample_weights_cap_less_interesting_mass_as_combined_bucket():
    df = pd.DataFrame(
        {
            "regime_specialist_bucket": (
                ["current"] * 100
                + ["analogue"] * 100
                + ["normal"] * 300
                + ["irrelevant"] * 300
            ),
            "similarity_to_current": (
                [1.0] * 100
                + [0.9] * 100
                + [0.5] * 300
                + [0.2] * 300
            ),
        }
    )
    weights, diag = compute_specialist_sample_weights(
        df,
        config=SpecialistWeightConfig(
            tau_current=1.0,
            tau_analogue=1.0,
            tau_normal=1.0,
            tau_irrelevant=1.0,
            less_interesting_min_mass=0.10,
            less_interesting_max_mass=0.10,
            min_weight=0.01,
            max_weight=50.0,
        ),
    )

    assert weights.mean() == pytest.approx(1.0)
    assert diag["less_interesting_mass_cap"] == pytest.approx(0.10)
    assert diag["less_interesting_mass"] <= 0.10 + 1e-9
    assert diag["actual_less_interesting_weight_mass"] <= 0.10 + 1e-6
    assert diag["normal_mass"] + diag["irrelevant_mass"] == pytest.approx(
        diag["less_interesting_mass"]
    )


def test_specialist_sample_weights_replay_uses_similarity_continuum():
    df = pd.DataFrame(
        {
            "regime_specialist_bucket": (
                ["current"] * 100
                + ["analogue"] * 100
                + ["normal"] * 100
                + ["irrelevant"] * 100
            ),
            "similarity_to_current": (
                [1.0] * 100
                + [0.8] * 100
                + [0.4] * 100
                + [0.1] * 100
            ),
        }
    )
    weights, diag = compute_specialist_sample_weights(
        df,
        config=SpecialistWeightConfig(
            tau_current=1.0,
            tau_analogue=1.0,
            tau_normal=1.0,
            tau_irrelevant=1.0,
            less_interesting_min_mass=0.10,
            less_interesting_max_mass=0.50,
            min_current_plus_analogue_mass=0.50,
            min_weight=0.001,
            max_weight=50.0,
        ),
    )

    replay = df["regime_specialist_bucket"].isin(["normal", "irrelevant"])
    normal = df["regime_specialist_bucket"].eq("normal")
    irrelevant = df["regime_specialist_bucket"].eq("irrelevant")

    assert weights.mean() == pytest.approx(1.0)
    assert diag["less_interesting_mass"] <= 0.50 + 1e-9
    assert diag["current_mass"] + diag["analogue_mass"] >= 0.50 - 1e-9
    assert weights.loc[normal].mean() > weights.loc[irrelevant].mean()
    assert weights.loc[replay].sum() / weights.sum() == pytest.approx(
        diag["actual_less_interesting_weight_mass"],
    )


def test_specialist_sample_weights_recency_is_secondary_to_similarity():
    df = pd.DataFrame(
        {
            "regime_specialist_bucket": ["current", "current"],
            "similarity_to_current": [1.0, 1.0],
            "recency": [1.0, 0.25],
        },
    )
    sqrt_weights, sqrt_diag = compute_specialist_sample_weights(
        df,
        recency_col="recency",
        config=SpecialistWeightConfig(
            recency_power=0.5,
            tau_current=1.0,
            min_weight=0.001,
            max_weight=100.0,
        ),
    )
    linear_weights, _linear_diag = compute_specialist_sample_weights(
        df,
        recency_col="recency",
        config=SpecialistWeightConfig(
            recency_power=1.0,
            tau_current=1.0,
            min_weight=0.001,
            max_weight=100.0,
        ),
    )

    assert sqrt_diag["recency_power"] == pytest.approx(0.5)
    assert sqrt_weights.iloc[0] / sqrt_weights.iloc[1] == pytest.approx(2.0)
    assert linear_weights.iloc[0] / linear_weights.iloc[1] == pytest.approx(4.0)


def test_self_distillation_shrinks_towards_one_for_low_similarity():
    adjusted = shrink_self_distillation_towards_one(
        [2.0, 0.5, 3.0],
        [1.0, 0.0, 0.5],
        power=1.0,
    )
    assert adjusted[0] == pytest.approx(2.0)
    assert adjusted[1] == pytest.approx(1.0)
    assert adjusted[2] == pytest.approx(2.0)


def test_weighted_drift_baseline_uses_current_regime_weights():
    frame = pd.DataFrame(
        {
            "feature_drift_psi_core": [1.0, 2.0, 10.0],
            "feature_drift_ks_core": [0.5, 1.5, 5.0],
            "current_regime_recency_weight": [0.25, 0.75, 0.0],
        }
    )
    baseline = weighted_drift_baseline(
        frame,
        drift_columns=["feature_drift_psi_core", "feature_drift_ks_core"],
    )
    assert baseline["enabled"] is True
    assert baseline["stats"]["feature_drift_psi_core"]["weighted_mean"] == pytest.approx(1.75)
    assert baseline["stats"]["feature_drift_ks_core"]["weighted_mean"] == pytest.approx(1.25)
