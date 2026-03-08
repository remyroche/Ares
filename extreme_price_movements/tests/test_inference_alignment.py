import pandas as pd

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
    assert upper["sl_mult"] == 1.7
