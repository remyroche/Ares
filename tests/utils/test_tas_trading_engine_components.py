import importlib.util
import numpy as np
import pandas as pd
import pytest

if importlib.util.find_spec("torch") is None:
    pytest.skip("torch not available", allow_module_level=True)

from src.utils.ml_common.optimization.tas.trading.trading_engine import (
    TradingConfig,
    TradingEngine,
)


class _AcceptAllRisk:
    def check_trade_risk(self, *args, **kwargs):
        return True

    def check_signal_risk(self, *args, **kwargs):
        return True


def _build_market_data() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=30, freq="D")
    closes = np.linspace(100, 110, len(index))
    highs = closes + 1
    lows = closes - 1
    volume = np.linspace(1_000, 1_500, len(index))
    return pd.DataFrame(
        {"close": closes, "high": highs, "low": lows, "volume": volume},
        index=index,
    )


def test_utils_trading_engine_generates_confident_signals():
    engine = TradingEngine(TradingConfig(), risk_manager=_AcceptAllRisk())

    assert engine.signal_generator is not None
    assert engine.risk_manager is not None

    signals = engine.generate_trading_signals(
        market_data=_build_market_data(),
        regime_info={"confidence": 0.9, "symbol": "UTIL"},
    )

    assert isinstance(signals, list)
    assert signals, "Expected confident signals"
    for signal in signals:
        assert "confidence" in signal
        assert signal["confidence"] == pytest.approx(0.9)
