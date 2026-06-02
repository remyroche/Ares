import numpy as np
import pandas as pd

from extreme_price_movements.config import CFG
from extreme_price_movements.inference.model_orchestrator import ModelOrchestrator
from extreme_price_movements.training_utils import get_meta_feature_keys
import extreme_price_movements.training as tr


def test_meta_policy_slice_availability_exempts_model_derived_lgbm_features(monkeypatch):
    df = pd.DataFrame(
        {
            "raw_good": [1.0, 2.0, 3.0, 4.0],
            "base_prob_x_vol_regime": [np.nan, np.nan, np.nan, np.nan],
            "pred_H10": [np.nan, np.nan, np.nan, np.nan],
            "feature_drift_psi_core": [np.nan, np.nan, np.nan, np.nan],
        }
    )

    def fake_feature_store_availability_matrix(feature_cols, *, cfg):
        assert list(feature_cols) == ["raw_good", "base_prob_x_vol_regime"]
        finite = np.asarray(
            [
                [True, False],
                [True, False],
                [True, False],
                [True, False],
            ],
            dtype=bool,
        )
        return finite, int(finite.shape[0]), "fake policy slice"

    monkeypatch.setattr(
        tr,
        "_feature_store_availability_matrix",
        fake_feature_store_availability_matrix,
    )

    kept = tr._recent_feature_availability_filter(
        df,
        list(df.columns),
        cfg={"lgbm_feature_recent_min_coverage": 0.85, "lgbm_feature_recent_min_rows": 1},
        context="LGBM model race meta_demo_clf",
        exempt_features={
            c for c in df.columns if tr._is_lgbm_model_derived_meta_feature(c)
        },
    )

    assert kept == ["raw_good", "pred_H10", "feature_drift_psi_core"]
    assert "base_prob_x_vol_regime" not in kept


def test_meta_performance_feature_groups_survive_portable_config():
    clf_keys = set(get_meta_feature_keys("clf", CFG))

    assert "recent_global_rolling_ic_5d" in clf_keys
    assert "recent_global_confidence_surprise_5d" in clf_keys
    assert "recent_global_top15_hit_rate_5d" in clf_keys
    assert "base_model_score" in clf_keys
    assert "prob_error" in clf_keys
    assert "recent_hit_rate_20" in clf_keys


def test_inference_materializes_base_performance_meta_features():
    model = type(
        "MetaModelStub",
        (),
        {
            "feature_columns": [
                "pred_H10",
                "base_model_score",
                "base_model_score_pct",
                "base_model_margin",
                "prob_error",
                "recent_prob_error_20",
                "recent_hit_rate_20",
                "base_model_abs_error_roll20",
                "recent_global_rolling_ic_5d",
            ]
        },
    )()
    orch = ModelOrchestrator(
        {}, runtime_cfg={"meta_trade_rank_window": 20, "strict_feature_parity": False}
    )
    features = pd.DataFrame({"pred_H10": [0.2, 0.8]})

    out = orch._materialize_meta_model_derived_features(
        features,
        model,
        side="long",
        kind="demo",
    )

    assert out["base_model_score"].tolist() == [0.2, 0.8]
    assert out["base_model_score_pct"].tolist() == [0.5, 0.5]
    assert np.allclose(out["base_model_margin"].to_numpy(), [0.3, 0.3])
    assert out["prob_error"].tolist() == [0.5, 0.5]
    assert out["recent_prob_error_20"].tolist() == [0.5, 0.5]
    assert out["recent_hit_rate_20"].tolist() == [0.5, 0.5]
    assert out["base_model_abs_error_roll20"].tolist() == [0.5, 0.5]
    assert out["recent_global_rolling_ic_5d"].tolist() == [0.0, 0.0]


def test_strict_inference_does_not_neutralize_non_causal_meta_features():
    model = type(
        "MetaModelStub",
        (),
        {
            "feature_columns": [
                "pred_H10",
                "prob_error",
                "recent_hit_rate_20",
                "recent_global_rolling_ic_5d",
            ]
        },
    )()
    orch = ModelOrchestrator(
        {}, runtime_cfg={"meta_trade_rank_window": 20, "strict_feature_parity": True}
    )
    features = pd.DataFrame({"pred_H10": [0.2, 0.8]})

    out = orch._materialize_meta_model_derived_features(
        features,
        model,
        side="long",
        kind="demo",
    )

    assert "prob_error" not in out.columns
    assert "recent_hit_rate_20" not in out.columns
    assert "recent_global_rolling_ic_5d" not in out.columns


def test_inference_materializes_drift_aliases_from_artifact_state(monkeypatch):
    model = type(
        "MetaModelStub",
        (),
        {
            "feature_columns": [
                "pred_H5_regime_centroid_similarity_train",
                "base_H5_feature_drift_psi_core_80",
                "pred_demo_H5_mahalanobis_mean_shift",
                "pred_demo_H5_reg_rare_leaf_low_support_score",
                "feature_drift_psi_core",
            ],
            "model_drift_state_": {"enabled": True},
        },
    )()
    orch = ModelOrchestrator({}, runtime_cfg={"strict_feature_parity": True})
    features = pd.DataFrame({"pred_H5": [0.2, 0.8]})

    def fake_transform_model_drift_features(*args, **kwargs):
        return pd.DataFrame(
            {
                "regime_centroid_similarity_train": [0.91, 0.82],
                "feature_drift_psi_core_80": [0.11, 0.22],
                "mahalanobis_mean_shift": [1.5, 2.5],
                "rare_leaf_low_support_score": [0.03, 0.04],
            },
            index=features.index,
        )

    monkeypatch.setattr(
        "extreme_price_movements.inference.model_orchestrator.transform_model_drift_features",
        fake_transform_model_drift_features,
    )

    out = orch._materialize_meta_model_drift_features(features, model)

    assert np.allclose(out["pred_H5_regime_centroid_similarity_train"], [0.91, 0.82])
    assert np.allclose(out["base_H5_feature_drift_psi_core_80"], [0.11, 0.22])
    assert np.allclose(out["pred_demo_H5_mahalanobis_mean_shift"], [1.5, 2.5])
    assert np.allclose(out["pred_demo_H5_reg_rare_leaf_low_support_score"], [0.03, 0.04])
    assert np.allclose(out["feature_drift_psi_core"], [0.11, 0.22])
