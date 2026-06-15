import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.model_drift_features import (
    MODEL_DRIFT_FEATURE_KEYS,
    ROW_LOCAL_DRIFT_FEATURE_KEYS,
    fit_model_drift_state,
    transform_model_drift_features,
)


def test_model_drift_features_are_artifact_backed_and_stable():
    rng = np.random.default_rng(42)
    x_train = pd.DataFrame(
        rng.normal(size=(120, 6)).astype(np.float32),
        columns=[f"f{i}" for i in range(6)],
    )
    state = fit_model_drift_state(x_train, feature_columns=list(x_train.columns), window=20)
    assert state["enabled"] is True

    drift = transform_model_drift_features(x_train.iloc[:10], state)

    assert list(drift.columns) == list(MODEL_DRIFT_FEATURE_KEYS)
    assert np.isfinite(drift.to_numpy(dtype=np.float32)).all()
    assert drift["regime_centroid_similarity_train"].between(0.0, 1.0).all()
    assert drift["feature_drift_psi_core_50"].between(0.0, 1.0).all()
    assert drift["feature_drift_psi_core_80"].between(0.0, 1.0).all()
    assert drift["feature_drift_ks_bin_mean"].between(0.0, 1.0).all()
    assert drift["feature_drift_ks_bin_max"].between(0.0, 1.0).all()
    assert np.allclose(drift["feature_drift_psi_core_80"], 0.0)
    assert np.allclose(drift["feature_drift_ks_bin_mean"], 0.0)
    assert np.allclose(drift["feature_drift_cov_shift"], 0.0)
    for key in ROW_LOCAL_DRIFT_FEATURE_KEYS:
        assert key in drift.columns
        assert np.isfinite(drift[key].to_numpy(dtype=np.float32)).all()
    assert drift["inference_drift_score"].between(0.0, 1.0).all()


def test_model_drift_score_increases_for_shifted_rows():
    rng = np.random.default_rng(7)
    x_train = pd.DataFrame(
        rng.normal(size=(200, 5)).astype(np.float32),
        columns=[f"f{i}" for i in range(5)],
    )
    state = fit_model_drift_state(x_train, feature_columns=list(x_train.columns))
    in_domain = transform_model_drift_features(x_train.iloc[:20], state)
    shifted = transform_model_drift_features(x_train.iloc[:20] + 4.0, state)

    assert float(shifted["inference_drift_score"].mean()) > float(
        in_domain["inference_drift_score"].mean()
    )
    assert float(shifted["regime_centroid_similarity_train"].mean()) < float(
        in_domain["regime_centroid_similarity_train"].mean()
    )
    assert float(shifted["row_drift_v1_psi_bin_mean"].mean()) > float(
        in_domain["row_drift_v1_psi_bin_mean"].mean()
    )
    assert float(shifted["row_drift_v1_ks_bin_mean"].mean()) >= float(
        in_domain["row_drift_v1_ks_bin_mean"].mean()
    )
    assert np.allclose(shifted["feature_drift_cov_shift"], 0.0)
    assert np.allclose(shifted["feature_drift_psi_bin_mean"], 0.0)


def test_model_drift_features_are_row_stable_across_batch_shapes():
    rng = np.random.default_rng(11)
    x_train = pd.DataFrame(
        rng.normal(size=(180, 5)).astype(np.float32),
        columns=[f"f{i}" for i in range(5)],
    )
    state = fit_model_drift_state(
        x_train,
        feature_columns=list(x_train.columns),
        window=20,
    )
    batch = x_train.iloc[10:18].copy()
    batch_drift = transform_model_drift_features(batch, state, index=batch.index)

    for idx in batch.index:
        single = transform_model_drift_features(
            batch.loc[[idx]],
            state,
            index=pd.Index([idx]),
        )
        assert np.allclose(
            batch_drift.loc[idx].to_numpy(dtype=np.float32),
            single.iloc[0].to_numpy(dtype=np.float32),
            rtol=0.0,
            atol=1e-7,
        )


def test_meta_model_drift_materializer_exports_prefixed_reporting_features():
    from extreme_price_movements.inference.model_orchestrator import ModelOrchestrator

    rng = np.random.default_rng(123)
    x_train = pd.DataFrame(
        rng.normal(size=(160, 4)).astype(np.float32),
        columns=[f"f{i}" for i in range(4)],
    )
    state = fit_model_drift_state(
        x_train,
        feature_columns=list(x_train.columns),
        window=20,
    )

    class MetaModel:
        model_drift_state_ = state
        feature_columns = list(x_train.columns)

    orchestrator = ModelOrchestrator({}, {})
    out = orchestrator._materialize_meta_model_drift_features(
        x_train.iloc[:8].copy(),
        MetaModel(),
        include_all=True,
        prefix="meta_lgbm",
    )

    for key in MODEL_DRIFT_FEATURE_KEYS:
        col = f"meta_lgbm_{key}"
        assert col in out.columns
        assert np.isfinite(out[col].to_numpy(dtype=np.float32)).all()
    assert "meta_lgbm_feature_drift_psi_core" in out.columns
    assert "meta_lgbm_feature_drift_ks_core" in out.columns
    assert "feature_drift_psi_core_80" not in out.columns


def test_alpha_model_context_materializes_base_lgbm_aliases_for_meta_contract():
    from extreme_price_movements.inference.model_orchestrator import ModelOrchestrator

    class AlphaModel:
        def transform_meta_features(self, frame):
            return pd.DataFrame(
                {
                    "uncertainty_score": [0.2, 0.3],
                    "inference_drift_score": [0.4, 0.5],
                    "feature_drift_psi_core_80": [0.12, 0.22],
                    "feature_drift_ks_bin_mean": [0.13, 0.23],
                },
                index=frame.index,
            )

    class MetaModel:
        feature_columns = [
            "base_lgbm_uncertainty_score",
            "base_lgbm_inference_drift_score",
            "base_lgbm_feature_drift_psi_core",
            "base_lgbm_feature_drift_ks_core",
        ]

    orchestrator = ModelOrchestrator(
        {
            "bundle": {
                "alpha_models": {
                    "long_test": {
                        "model": AlphaModel(),
                        "feat_cols": ["feature_a"],
                    }
                }
            }
        },
        {},
    )
    features = pd.DataFrame(
        {"feature_a": [1.0, 2.0]},
        index=pd.Index(["BTC/USD:USD", "ETH/USD:USD"]),
    )
    out = orchestrator._materialize_alpha_model_meta_features(
        features,
        MetaModel(),
        side="long",
        kind="long_test",
    )

    assert np.allclose(out["base_lgbm_uncertainty_score"], [0.2, 0.3])
    assert np.allclose(out["base_lgbm_inference_drift_score"], [0.4, 0.5])
    assert np.allclose(out["base_lgbm_feature_drift_psi_core"], [0.12, 0.22])
    assert np.allclose(out["base_lgbm_feature_drift_ks_core"], [0.13, 0.23])


def test_base_predictive_atlas_features_capture_hit_rate_surprise():
    from extreme_price_movements.training import _append_base_predictive_atlas_features

    n = 80
    ts = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    regime = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    score = np.r_[np.full(n // 2, 0.62), np.full(n // 2, 0.62)]
    y = np.r_[
        np.tile([1.0, 0.0, 0.0, 0.0], n // 8),
        np.tile([1.0, 1.0, 1.0, 0.0], n // 8),
    ]
    frame = pd.DataFrame(
        {
            "base_lgbm_uncertainty_score": regime,
            "base_lgbm_inference_drift_score": regime * 0.5,
            "base_H10_pred_std": regime * 0.1,
        }
    )
    df = pd.DataFrame({"__ts__": ts, "__symbol__": ["BTC"] * n})

    out, cols = _append_base_predictive_atlas_features(
        frame,
        df=df,
        strategy_id="long_test",
        primary_horizon=10,
        base_score=score,
        y_binary=y,
        cfg={
            "base_predictive_atlas_min_rows": 20,
            "base_predictive_atlas_min_support": 5,
            "base_predictive_atlas_min_cluster_rows": 5,
            "base_predictive_atlas_max_clusters": 2,
        },
    )

    assert "base_lgbm_predictive_atlas_hit_rate_surprise" in cols
    late_bad = out.iloc[30:40]["base_lgbm_predictive_atlas_hit_rate_surprise"].mean()
    late_good = out.iloc[70:80]["base_lgbm_predictive_atlas_hit_rate_surprise"].mean()
    assert late_good > late_bad
    assert out["base_lgbm_predictive_atlas_hit_rate_surprise_z"].abs().max() <= 8.0


def test_base_predictive_atlas_waits_for_label_maturity():
    from extreme_price_movements.training import _append_base_predictive_atlas_features

    n = 24
    horizon = 5
    ts = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "base_lgbm_uncertainty_score": np.linspace(0.0, 1.0, n),
            "base_H5_pred_std": np.linspace(1.0, 0.0, n),
        }
    )
    df = pd.DataFrame({"__ts__": ts, "__symbol__": ["BTC"] * n})
    score = np.full(n, 0.8, dtype=np.float32)
    y = np.ones(n, dtype=np.float32)

    out, _ = _append_base_predictive_atlas_features(
        frame,
        df=df,
        strategy_id="long_test",
        primary_horizon=horizon,
        base_score=score,
        y_binary=y,
        cfg={
            "base_predictive_atlas_min_rows": 10,
            "base_predictive_atlas_min_support": 1,
            "base_predictive_atlas_min_cluster_rows": 100,
        },
    )

    early = out.iloc[: horizon + 1]["base_lgbm_predictive_atlas_support_n"]
    assert np.allclose(early.fillna(0.0).to_numpy(), 0.0)
    assert out.iloc[horizon + 2]["base_lgbm_predictive_atlas_support_n"] >= 1.0


def test_causal_base_performance_features_wait_for_label_maturity():
    from extreme_price_movements.training import _causal_base_performance_feature_arrays

    n = 12
    horizon = 3
    ts = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    frame = pd.DataFrame({"__ts__": ts, "__symbol__": ["BTC"] * n})
    score = np.asarray([0.9, 0.8, 0.7, 0.6, 0.55, 0.52, 0.51, 0.5, 0.49, 0.48, 0.47, 0.46])
    y = np.asarray([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    rank = np.linspace(1.0, 0.8, n)

    feats = _causal_base_performance_feature_arrays(
        frame=frame,
        base_prob=score,
        outcome=y,
        base_rank_pct=rank,
        asset_vals=np.asarray(["BTC"] * n),
        horizon_hours=horizon,
        rank_top_frac=0.5,
        window=20,
        min_periods=2,
    )

    assert np.isnan(feats["prob_error"][horizon])
    assert feats["prob_error"][horizon + 1] == pytest.approx(abs(score[0] - y[0]))
    assert np.isnan(feats["recent_hit_rate_20"][horizon + 1])
    assert np.isfinite(feats["recent_hit_rate_20"][horizon + 2])
