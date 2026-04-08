import json
from importlib import import_module

import pandas as pd

from extreme_price_movements.model_loader import load_bucket_params

load_trades_for_bucket = import_module(
    "extreme_price_movements.tpsl_optimiser.00_load_trades"
).load_trades_for_bucket


def test_load_bucket_params_prefers_strategy_params_json(tmp_path):
    run_id = "20260408_120000"
    ridge_dir = tmp_path / "artifacts" / run_id / "ridge_sizer"
    models_dir = tmp_path / "artifacts" / run_id / "models"
    ridge_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    legacy_payload = {
        "buckets": {
            "LONG_MR": {
                "tp_sl": {"tp_mult": 1.0, "sl_mult": 0.5},
                "profit_exit": {"act_n": 0.1, "be_act_n": 0.2},
            }
        }
    }
    strategy_payload = {
        "buckets": {
            "my_strategy_id": {
                "tp_sl": {"tp_mult": 2.5, "sl_mult": 1.5},
                "profit_exit": {"act_n": 0.5, "be_act_n": 0.4},
            }
        }
    }
    (models_dir / "bucket_params.json").write_text(json.dumps(legacy_payload))
    (ridge_dir / "strategy_params.json").write_text(json.dumps(strategy_payload))

    params = load_bucket_params(run_id, str(tmp_path))
    assert "my_strategy_id" in params
    assert params["my_strategy_id"]["tp_mult"] == 2.5
    assert params["my_strategy_id"]["sl_mult"] == 1.5
    assert "LONG_MR" not in params


def test_load_trades_for_bucket_prefers_strategy_id_column():
    trades = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC"),
            "bucket": ["LONG_MR", "LONG_MR", "SHORT_TF", "SHORT_TF"],
            "strategy_id": ["my_strategy_id", "my_strategy_id", "other_id", "other_id"],
            "confidence": [0.9, 0.8, 0.2, 0.1],
            "entry_price": [100.0, 101.0, 102.0, 103.0],
            "exit_price": [101.0, 102.0, 101.0, 100.0],
            "is_long": [1, 1, 0, 0],
        }
    )

    filtered = load_trades_for_bucket(trades, "my_strategy_id")
    assert len(filtered) == 1
    assert filtered["strategy_id"].tolist() == ["my_strategy_id"]
