import importlib.util
from pathlib import Path

import pandas as pd


def _load_generator_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "generate_policy_oos_predictions.py"
    spec = importlib.util.spec_from_file_location("generate_policy_oos_predictions", script)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_policy_oos_generator_filters_to_trained_universe():
    module = _load_generator_module()
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01T00:00:00Z"] * 3),
            "symbol": ["AAA/USD:USD", "ZZZ/USD:USD", "AAA/USD:USD"],
            "clf": [0.9, 0.8, 0.7],
        }
    )

    filtered, report = module._filter_policy_oos_to_trained_universe(
        df,
        trained_universe={"AAA/USD:USD"},
    )

    assert filtered["symbol"].tolist() == ["AAA/USD:USD", "AAA/USD:USD"]
    assert report["input_rows"] == 3
    assert report["kept_rows"] == 2
    assert report["dropped_rows"] == 1
    assert report["dropped_symbol_sample"] == ["ZZZ/USD:USD"]


def test_policy_oos_generator_writes_row_level_contract_columns():
    module = _load_generator_module()
    rows = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-22T10:00:00Z"]),
            "symbol": ["AAA/USD:USD"],
            "clf": [0.9],
        }
    )

    out = module._attach_policy_oos_contract_columns(
        rows,
        market_mode="perps",
        source_model_fit_end="2024-09-08T04:00:00+00:00",
        generation_source="generated_from_train_meta_state:labels",
    )

    assert out.loc[0, "market_mode"] == "perps"
    assert out.loc[0, "policy_oos_source_model_fit_end"] == (
        "2024-09-08T04:00:00+00:00"
    )
    assert out.loc[0, "policy_oos_generation_source"] == (
        "generated_from_train_meta_state:labels"
    )
