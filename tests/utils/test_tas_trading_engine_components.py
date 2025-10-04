import importlib.util
import pytest

if importlib.util.find_spec("torch") is None:
    pytest.skip("torch not available", allow_module_level=True)

from src.utils.ml_common.optimization.tas.trading.trading_engine import (
    TradingConfig,
    TradingEngine,
)


def test_utils_trading_engine_imports_components():
    engine = TradingEngine(TradingConfig())
    assert engine.signal_generator is not None
    assert engine.position_manager is not None
    assert engine.risk_manager is not None
    assert engine.performance_monitor is not None
