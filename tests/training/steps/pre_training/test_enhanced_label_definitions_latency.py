import importlib.util
from pathlib import Path

import pandas as pd
import pytest

MODULE_PATH = Path(__file__).resolve().parents[4] / "src" / "training" / "steps" / "pre_training" / "profit_labeling" / "enhanced_label_definitions.py"
spec = importlib.util.spec_from_file_location("enhanced_label_definitions_test", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

EnhancedLabelDefinitions = module.EnhancedLabelDefinitions
TradingCosts = module.TradingCosts


def _make_market_data():
    index = pd.date_range("2024-01-01", periods=6, freq="15min")
    return pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104, 105],
            "high": [101, 102, 103, 104, 105, 106],
            "low": [99, 100, 101, 102, 103, 104],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5],
            "volume": [5000, 5000, 5000, 5000, 5000, 5000],
        },
        index=index,
    )


def test_expected_returns_use_next_bar_open():
    labeler = EnhancedLabelDefinitions()
    market_data = _make_market_data()
    context = labeler._build_execution_context(market_data, labeler.analyst_config.horizon_minutes)

    expected_returns = labeler._calculate_expected_returns(
        market_data,
        labeler.analyst_config.horizon_minutes,
        entry_prices=context["entry_prices"],
        exit_prices=context["exit_prices"],
    )

    first_entry = context["entry_prices"].iloc[0]
    first_exit = context["exit_prices"].iloc[0]
    manual_return = (first_exit - first_entry) / first_entry

    assert expected_returns.iloc[0] == pytest.approx(manual_return)
    assert expected_returns.iloc[-1] == 0.0


def test_trading_costs_apply_entry_and_exit_slippage():
    labeler = EnhancedLabelDefinitions()
    market_data = _make_market_data()
    context = labeler._build_execution_context(market_data, labeler.analyst_config.horizon_minutes)

    costs = TradingCosts(
        maker_fee=0.001,
        taker_fee=0.002,
        slippage_pct=0.001,
        min_trade_size=0.0,
    )

    trading_costs = labeler._calculate_trading_costs(
        market_data,
        costs,
        entry_prices=context["entry_prices"],
        exit_prices=context["exit_prices"],
    )

    first_entry = context["entry_prices"].iloc[0]
    trade_notional = market_data["volume"].iloc[0] * first_entry * 0.01
    expected_cost = trade_notional * (2 * costs.taker_fee + 2 * costs.slippage_pct)

    assert trading_costs.iloc[0] == pytest.approx(expected_cost)
    assert trading_costs.iloc[-1] == 0.0


def test_generate_analyst_labels_records_execution_metadata():
    labeler = EnhancedLabelDefinitions()
    market_data = _make_market_data()
    volatility = pd.Series(0.02, index=market_data.index)

    labels, confidence = labeler.generate_analyst_labels(market_data, volatility)

    metadata = labeler.get_execution_latency_metadata()
    assert metadata["signal_to_execution_delay_bars"] == 1
    assert metadata["entry_price_source"].startswith("next")
    # Trades without future data should be disabled
    assert labels.iloc[-1] == 0
    assert pd.isna(confidence.iloc[-1]) or confidence.iloc[-1] <= 1.0
