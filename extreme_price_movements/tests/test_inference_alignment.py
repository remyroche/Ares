import numpy as np
import pandas as pd

from extreme_price_movements.ebm_on_lgbm import EBMOnLGBMModel
from extreme_price_movements.inference.feature_generator import (
    get_features_for_candidates,
)
from extreme_price_movements.inference.model_orchestrator import ModelOrchestrator
from extreme_price_movements.inference.trade_executor import TradeExecutor
from extreme_price_movements.model_drift_features import (
    fit_model_drift_state,
    transform_model_drift_features,
)


def test_model_orchestrator_uses_flattened_ridge_weights():
    bundle = {
        "alpha_models": {},
        "meta_models": {},
        "spike_models": {},
        "ridge_weights": {
            "weights": {
                "long_mr_feat_a": 0.5,
                "long_mr_feat_b": -0.25,
            },
            "params_per_bucket": {
                "long_mr": {"cooldown_hours": 2},
            },
        },
        "bucket_params": {},
    }
    full_state = {
        "bundle": bundle,
        "bucket_params": {},
        "ridge_sizer": None,
    }
    orchestrator = ModelOrchestrator(full_state, {})
    features = pd.DataFrame(
        {
            "feat_a": [2.0],
            "feat_b": [4.0],
        },
        index=["BTC/USDT"],
    )
    position, confidence = orchestrator.compute_ridge_position_size(
        features, "long", "mr"
    )
    assert position.index.tolist() == ["BTC/USDT"]
    assert abs(float(position.iloc[0]) - 0.0) < 1e-12
    assert 0.0 <= float(confidence["confidence"]) <= 1.0


def test_trade_executor_bucket_lookup_normalizes_case():
    executor = TradeExecutor(
        mode="shadow",
        bucket_params={
            "long_mr": {"cooldown_hours": 3.0, "sl_mult": 1.7},
        },
        config={},
    )
    lower = executor.get_bucket_params("long_mr")
    upper = executor.get_bucket_params("LONG_MR")
    assert lower["cooldown_hours"] == 3.0
    assert upper["cooldown_hours"] == 3.0
    assert "sl_mult" not in lower
    assert "sl_mult" not in upper


def test_predict_alpha_renames_synthetic_ebm_contract_from_feat_cols_order():
    class CapturingAlphaModel:
        def __init__(self):
            self.best_model = EBMOnLGBMModel(mode="classifier")
            self.best_model.raw_selected_features = ["f0", "f1"]
            self.best_model.selected_features = ["f0", "f1"]
            self.seen_columns = None

        def predict(self, X):
            self.seen_columns = list(X.columns)
            return [0.7] * len(X)

    model = CapturingAlphaModel()
    orchestrator = ModelOrchestrator(
        {
            "alpha_models": {
                "long_demo": {
                    "model": model,
                    "feat_cols": ["real_a", "real_b"],
                }
            }
        }
    )
    features = pd.DataFrame(
        {"unrelated_first": [999.0], "real_b": [2.0], "real_a": [1.0]},
        index=["AAA/USDC"],
    )

    preds = orchestrator.predict_alpha(features, "long", "long_demo")

    assert float(preds.iloc[0]) == 0.7
    assert model.seen_columns == ["f0", "f1"]


def test_predict_alpha_maps_model_race_lgbm_fn_contract_from_feat_cols_positions():
    class _InnerLgbm:
        selected_features = ["f1", "f3"]
        input_feature_names = []

    class CapturingRaceModel:
        def __init__(self):
            self.best_model = _InnerLgbm()
            self.seen_columns = None
            self.seen_values = None

        def predict(self, X):
            self.seen_columns = list(X.columns)
            self.seen_values = X.iloc[0].to_dict()
            return [0.8] * len(X)

    model = CapturingRaceModel()
    orchestrator = ModelOrchestrator(
        {
            "alpha_models": {
                "long_demo": {
                    "model": model,
                    "feat_cols": ["real_0", "real_1", "real_2", "real_3"],
                }
            }
        }
    )
    features = pd.DataFrame(
        {"real_3": [3.0], "real_1": [1.0], "real_0": [0.0], "real_2": [2.0]},
        index=["AAA/USDC"],
    )

    preds = orchestrator.predict_alpha(features, "long", "long_demo")

    assert float(preds.iloc[0]) == 0.8
    assert model.seen_columns == ["f1", "f3"]
    assert model.seen_values == {"f1": 1.0, "f3": 3.0}


def test_predict_alpha_strict_mode_refuses_nonfinite_lgbm_training_frame():
    class CapturingAlphaModel:
        def __init__(self):
            self.seen_values = None

        def predict(self, X):
            self.seen_values = X.iloc[0].to_dict()
            return [0.6] * len(X)

    model = CapturingAlphaModel()
    orchestrator = ModelOrchestrator(
        {
            "alpha_models": {
                "long_demo": {
                    "model": model,
                    "feat_cols": ["finite", "missing", "infinite"],
                }
            }
        },
        {"strict_feature_parity": True},
    )
    features = pd.DataFrame(
        {"finite": [1.5], "missing": [np.nan], "infinite": [np.inf]},
        index=["AAA/USDC"],
    )

    preds = orchestrator.predict_alpha(features, "long", "long_demo")

    assert preds.empty
    assert model.seen_values is None


def test_predict_alpha_strict_mode_drops_only_nonfinite_rows():
    class CapturingAlphaModel:
        def __init__(self):
            self.seen_index = None
            self.seen_values = None

        def predict(self, X):
            self.seen_index = list(X.index)
            self.seen_values = X.to_dict("index")
            return [0.6] * len(X)

    model = CapturingAlphaModel()
    orchestrator = ModelOrchestrator(
        {
            "alpha_models": {
                "long_demo": {
                    "model": model,
                    "feat_cols": ["finite", "maybe_missing"],
                }
            }
        },
        {"strict_feature_parity": True},
    )
    features = pd.DataFrame(
        {"finite": [1.5, 2.5], "maybe_missing": [np.nan, 3.5]},
        index=["BAD/USDC", "GOOD/USDC"],
    )

    preds = orchestrator.predict_alpha(features, "long", "long_demo")

    assert list(preds.index) == ["GOOD/USDC"]
    assert float(preds.iloc[0]) == 0.6
    assert model.seen_index == ["GOOD/USDC"]
    assert model.seen_values == {"GOOD/USDC": {"finite": 2.5, "maybe_missing": 3.5}}


def test_predict_meta_materializes_rsi_regime_interaction():
    class CapturingMetaModel:
        feature_columns = [
            "pred_H10",
            "rsi_z",
            "regime_vol_score",
            "rsi_z_x_regime_vol",
        ]

        def __init__(self):
            self.seen_values = None

        def predict(self, X):
            self.seen_values = X.iloc[0].to_dict()
            return [0.55] * len(X)

    model = CapturingMetaModel()
    orchestrator = ModelOrchestrator(
        {"meta_models": {"long_demo_clf": model}},
        {"strict_feature_parity": True},
    )
    features = pd.DataFrame(
        {
            "pred_H10": [0.6],
            "rsi_z": [2.0],
            "regime_vol_score": [3.0],
        },
        index=["AAA/USDC"],
    )

    preds = orchestrator.predict_meta(features, "long", "long_demo")

    assert float(preds.iloc[0]) == 0.55
    assert np.isclose(model.seen_values["pred_H10"], 0.6)
    assert model.seen_values["rsi_z"] == 2.0
    assert model.seen_values["regime_vol_score"] == 3.0
    assert model.seen_values["rsi_z_x_regime_vol"] == 6.0


def test_predict_meta_materializes_training_meta_interactions():
    class CapturingMetaModel:
        feature_columns = [
            "pred_H10",
            "pred_demo_H10_vote_entropy",
            "base_med_x_side_aligned_trend",
            "base_med_x_efficiency_ratio",
            "base_med_x_vol_z_24h_minus_96h",
            "base_prob_x_vol_regime",
            "base_prob_x_entropy",
        ]

        def __init__(self):
            self.seen_values = None

        def predict(self, X):
            self.seen_values = X.iloc[0].to_dict()
            return [0.61] * len(X)

    model = CapturingMetaModel()
    orchestrator = ModelOrchestrator(
        {"meta_models": {"long_demo_clf": model}},
        {"strict_feature_parity": True},
    )
    features = pd.DataFrame(
        {
            "pred_H10": [0.8],
            "pred_H10_vote_entropy": [0.15],
            "trend_slope_72h": [2.0],
            "efficiency_ratio_20": [0.25],
            "vol_z24": [1.5],
            "volatility_zscore": [0.5],
            "regime_vol_score": [3.0],
            "regime_transition_entropy_12h": [0.4],
        },
        index=["AAA/USDC"],
    )

    preds = orchestrator.predict_meta(features, "long", "long_demo")

    assert float(preds.iloc[0]) == 0.61
    assert np.isclose(model.seen_values["pred_demo_H10_vote_entropy"], 0.15)
    assert np.isclose(model.seen_values["base_med_x_side_aligned_trend"], 1.6)
    assert np.isclose(model.seen_values["base_med_x_efficiency_ratio"], 0.2)
    assert np.isclose(model.seen_values["base_med_x_vol_z_24h_minus_96h"], 0.8)
    assert np.isclose(model.seen_values["base_prob_x_vol_regime"], 2.4)
    assert np.isclose(model.seen_values["base_prob_x_entropy"], 0.32)


def test_predict_meta_materializes_alpha_drift_features_for_parity():
    class AlphaModel:
        def __init__(self):
            self.seen_index = None

        def transform_meta_features(self, X):
            self.seen_index = list(X.index)
            return pd.DataFrame(
                {
                    "regime_centroid_similarity_train": [0.91, 0.87],
                    "feature_drift_psi_core": [0.12, 0.18],
                    "feature_drift_cov_shift": [0.03, 0.04],
                },
                index=X.index,
            )

    class CapturingMetaModel:
        feature_columns = [
            "long_demo",
            "regime_centroid_similarity_train",
            "feature_drift_psi_core",
            "feature_drift_cov_shift",
        ]

        def __init__(self):
            self.seen = None

        def predict(self, X):
            self.seen = X.copy()
            return [0.5] * len(X)

    alpha = AlphaModel()
    meta = CapturingMetaModel()
    orchestrator = ModelOrchestrator(
        {
            "alpha_models": {
                "long_demo": {
                    "model": alpha,
                    "feat_cols": ["ret24h"],
                }
            },
            "meta_models": {"long_demo_clf": meta},
        },
        {"strict_feature_parity": True},
    )
    features = pd.DataFrame(
        {
            "long_demo": [0.6, 0.7],
            "ret24h": [0.01, -0.02],
        },
        index=["AAA/USDC", "BBB/USDC"],
    )

    preds = orchestrator.predict_meta(features, "long", "long_demo")

    assert list(preds.index) == ["AAA/USDC", "BBB/USDC"]
    assert alpha.seen_index == ["AAA/USDC", "BBB/USDC"]
    assert np.allclose(
        meta.seen["regime_centroid_similarity_train"].to_numpy(),
        [0.91, 0.87],
    )
    assert np.allclose(meta.seen["feature_drift_psi_core"].to_numpy(), [0.12, 0.18])
    assert np.allclose(meta.seen["feature_drift_cov_shift"].to_numpy(), [0.03, 0.04])


def test_predict_meta_overwrites_stale_alpha_drift_aliases():
    class AlphaModel:
        def transform_meta_features(self, X):
            return pd.DataFrame(
                {
                    "rare_leaf_low_support_score": [0.25],
                    "regime_centroid_similarity_train_pc0": [0.75],
                    "regime_centroid_similarity_train_window_p10": [0.66],
                },
                index=X.index,
            )

    class CapturingMetaModel:
        feature_columns = [
            "long_demo",
            "pred_H10_rare_leaf_low_support_score",
            "pred_H10_regime_centroid_similarity_train_pc0",
            "pred_H10_regime_centroid_similarity_train_window_p10",
        ]

        def __init__(self):
            self.seen = None

        def predict(self, X):
            self.seen = X.copy()
            return [0.5] * len(X)

    meta = CapturingMetaModel()
    orchestrator = ModelOrchestrator(
        {
            "alpha_models": {
                "long_demo": {"model": AlphaModel(), "feat_cols": ["ret24h"]}
            },
            "meta_models": {"long_demo_clf": meta},
        },
        {"strict_feature_parity": True},
    )
    features = pd.DataFrame(
        {
            "long_demo": [0.6],
            "ret24h": [0.01],
            "pred_H10_rare_leaf_low_support_score": [9.0],
            "pred_H10_regime_centroid_similarity_train_pc0": [9.0],
            "pred_H10_regime_centroid_similarity_train_window_p10": [9.0],
        },
        index=["AAA/USDC"],
    )

    preds = orchestrator.predict_meta(features, "long", "long_demo")

    assert float(preds.iloc[0]) == 0.5
    assert np.isclose(meta.seen["pred_H10_rare_leaf_low_support_score"].iloc[0], 0.25)
    assert np.isclose(
        meta.seen["pred_H10_regime_centroid_similarity_train_pc0"].iloc[0],
        0.75,
    )
    assert np.isclose(
        meta.seen["pred_H10_regime_centroid_similarity_train_window_p10"].iloc[0],
        0.66,
    )


def test_predict_meta_overwrites_stale_meta_drift_features():
    class CapturingMetaModel:
        feature_columns = [
            "long_demo",
            "ret24h",
            "regime_centroid_similarity_train",
            "feature_drift_cov_shift",
            "pred_H5_feature_drift_cov_shift",
        ]

        def __init__(self):
            self.seen = None
            train = pd.DataFrame(
                {"ret24h": np.linspace(-1.0, 1.0, 20, dtype=np.float32)}
            )
            self.model_drift_state_ = fit_model_drift_state(
                train,
                feature_columns=["ret24h"],
                window=5,
            )

        def predict(self, X):
            self.seen = X.copy()
            return [0.5] * len(X)

    meta = CapturingMetaModel()
    orchestrator = ModelOrchestrator(
        {
            "alpha_models": {},
            "meta_models": {"long_demo_clf": meta},
        },
        {"strict_feature_parity": True},
    )
    features = pd.DataFrame(
        {
            "long_demo": [0.6],
            "ret24h": [0.25],
            "regime_centroid_similarity_train": [9.0],
            "feature_drift_cov_shift": [9.0],
        },
        index=["AAA/USDC"],
    )
    expected = transform_model_drift_features(
        features[["ret24h"]],
        meta.model_drift_state_,
        index=features.index,
    )

    preds = orchestrator.predict_meta(features, "long", "long_demo")

    assert float(preds.iloc[0]) == 0.5
    assert np.isclose(
        meta.seen["regime_centroid_similarity_train"].iloc[0],
        expected["regime_centroid_similarity_train"].iloc[0],
    )
    assert np.isclose(
        meta.seen["feature_drift_cov_shift"].iloc[0],
        expected["feature_drift_cov_shift"].iloc[0],
    )
    assert np.isclose(
        meta.seen["pred_H5_feature_drift_cov_shift"].iloc[0],
        expected["feature_drift_cov_shift"].iloc[0],
    )
    assert not np.isclose(meta.seen["feature_drift_cov_shift"].iloc[0], 9.0)


def test_predict_meta_refuses_missing_rank_percentile_context():
    class CapturingMetaModel:
        feature_columns = ["pred_H10", "base_model_score_pct"]

        def __init__(self):
            self.called = False

        def predict(self, X):
            self.called = True
            return [0.5] * len(X)

    model = CapturingMetaModel()
    orchestrator = ModelOrchestrator(
        {"meta_models": {"long_demo_clf": model}},
        {"strict_feature_parity": True},
    )
    features = pd.DataFrame({"pred_H10": [0.8]}, index=["AAA/USDC"])

    preds = orchestrator.predict_meta(features, "long", "long_demo")

    assert preds.empty
    assert model.called is False


def test_run_full_chain_fails_closed_when_position_sizer_rejects(monkeypatch):
    orchestrator = ModelOrchestrator({}, {"strict_feature_parity": True})

    monkeypatch.setattr(
        orchestrator,
        "predict_alpha",
        lambda features, side, kind: pd.Series([0.8], index=features.index),
    )
    monkeypatch.setattr(
        orchestrator,
        "predict_meta",
        lambda features, side, kind: pd.Series([0.7], index=features.index),
    )
    monkeypatch.setattr(
        orchestrator,
        "compute_ridge_position_size",
        lambda features, side, kind: (
            pd.Series([0.0], index=features.index),
            {"confidence": 0.0},
        ),
    )

    called_entry_policy = {"value": False}

    def _entry_policy(*args, **kwargs):
        called_entry_policy["value"] = True
        return {}

    monkeypatch.setattr(orchestrator, "compute_entry_policy", _entry_policy)

    features = pd.DataFrame({"ret1h": [0.01]}, index=["AAA/USDC"])
    result = orchestrator.run_full_chain("AAA/USDC", "long", features, kind="demo")

    assert result["action"] == "no_entry"
    assert result["reason"] == "position_sizer_rejected"
    assert result["sizing_source"] == "position_sizer_rejected"
    assert result["position_size"] == 0.0
    assert called_entry_policy["value"] is False


def test_get_features_for_candidates_uses_asof_not_future_latest():
    idx = pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T02:00:00Z"], utc=True)
    feats = {
        "ret1h": pd.DataFrame({"AAA/USDC": [1.0, 99.0]}, index=idx),
    }

    row = get_features_for_candidates(
        feats,
        ["AAA/USDC"],
        ts=pd.Timestamp("2026-01-01T01:00:00Z"),
    )

    assert row.loc["AAA/USDC", "ret1h"] == 1.0


def test_get_features_for_candidates_preserves_timestamped_nan():
    idx = pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"], utc=True)
    feats = {
        "basis": pd.DataFrame({"AAA/USDC": [1.0, np.nan]}, index=idx),
    }

    row = get_features_for_candidates(
        feats,
        ["AAA/USDC"],
        ts=pd.Timestamp("2026-01-01T01:00:00Z"),
    )

    assert "basis" in row.columns
    assert pd.isna(row.loc["AAA/USDC", "basis"])
