import pandas as pd

from extreme_price_movements.ebm_on_lgbm import EBMOnLGBMModel
from extreme_price_movements.inference.feature_generator import get_features_for_candidates
from extreme_price_movements.inference.model_orchestrator import ModelOrchestrator
from extreme_price_movements.inference.trade_executor import TradeExecutor


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
    position, confidence = orchestrator.compute_ridge_position_size(features, "long", "mr")
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


def test_get_features_for_candidates_uses_asof_not_future_latest():
    idx = pd.to_datetime(
        ["2026-01-01T00:00:00Z", "2026-01-01T02:00:00Z"], utc=True
    )
    feats = {
        "ret1h": pd.DataFrame({"AAA/USDC": [1.0, 99.0]}, index=idx),
    }

    row = get_features_for_candidates(
        feats,
        ["AAA/USDC"],
        ts=pd.Timestamp("2026-01-01T01:00:00Z"),
    )

    assert row.loc["AAA/USDC", "ret1h"] == 1.0
