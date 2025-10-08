import importlib.util
import logging
import sys
import types
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

from src.utils.common_ml.backtesting.backtesting_engine import BacktestingConfig, WalkForwardValidator


def _stub_module(name: str, attrs: Dict[str, Any] | None = None, is_package: bool = False) -> types.ModuleType:
    module = types.ModuleType(name)
    if is_package:
        module.__path__ = []  # type: ignore[attr-defined]
    if attrs:
        for key, value in attrs.items():
            setattr(module, key, value)
    sys.modules[name] = module
    return module


src_pkg = _stub_module("src.training", is_package=True)
steps_pkg = _stub_module("src.training.steps", is_package=True)
backtesting_pkg = _stub_module("src.training.steps.backtesting", is_package=True)

_stub_module("src.training.steps.backtesting.consolidated_backtesting_step")
_stub_module(
    "src.training.steps.backtesting.unified_config",
    {
        "UnifiedBacktestingConfig": type("UnifiedBacktestingConfig", (), {}),
        "ExecutionMode": type("ExecutionMode", (), {}),
    },
)

_stub_module(
    "src.utils.ml_common.vectorized_backtesting",
    {
        "VectorizedBacktestEngine": type("VectorizedBacktestEngine", (), {}),
        "VectorizedBacktestConfig": type("VectorizedBacktestConfig", (), {}),
    },
)
_stub_module(
    "src.utils.ml_common.cvlsa",
    {"CVLSAValidator": type("CVLSAValidator", (), {})},
)
_stub_module(
    "src.utils.ml_common.optimization",
    {"HyperparameterOptimizer": type("HyperparameterOptimizer", (), {})},
)
_stub_module(
    "src.utils.common_ml.backtesting.monte_carlo_engine",
    {
        "MonteCarloEngine": type("MonteCarloEngine", (), {}),
        "MonteCarloConfig": type("MonteCarloConfig", (), {}),
    },
)
_stub_module(
    "src.utils.common_ml.backtesting.ab_testing_engine",
    {
        "ABTestingEngine": type("ABTestingEngine", (), {}),
        "ABTestConfig": type("ABTestConfig", (), {}),
    },
)

profit_labeling_pkg = _stub_module("src.training.steps.pre_training.profit_labeling", is_package=True)
vol_labeler_stub = _stub_module("src.training.steps.pre_training.profit_labeling.volatility_aware_labeler")
setattr(profit_labeling_pkg, "volatility_aware_labeler", vol_labeler_stub)


_REAL_ENGINE_SPEC = importlib.util.spec_from_file_location(
    "src.training.steps.backtesting.real_backtesting_engine",
    Path(__file__).resolve().parents[4] / "src/training/steps/backtesting/real_backtesting_engine.py",
)
real_backtesting_module = importlib.util.module_from_spec(_REAL_ENGINE_SPEC)
assert _REAL_ENGINE_SPEC is not None and _REAL_ENGINE_SPEC.loader is not None
_REAL_ENGINE_SPEC.loader.exec_module(real_backtesting_module)
RealBacktestingEngine = real_backtesting_module.RealBacktestingEngine


class DummyBacktestingConfig:
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.capacity_limit = 0.2
        self.market_impact_coefficient = 0.1
        self.turnover_warning_threshold = 0.5


class DummyConfig:
    def __init__(self):
        self.backtesting = DummyBacktestingConfig()


@pytest.fixture
def sample_trade_portfolio() -> Dict[str, Any]:
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    portfolio = {
        "cash": 100500.0,
        "position": 0.0,
        "equity": 100500.0,
        "trades": [
            {"timestamp": dates[0], "action": "buy", "price": 10.0, "shares": 500},
            {"timestamp": dates[2], "action": "sell", "price": 11.0, "shares": 500},
        ],
    }
    data = pd.DataFrame(
        {
            "open": [10.0, 10.5, 11.0],
            "high": [10.5, 11.0, 11.5],
            "low": [9.5, 10.0, 10.5],
            "close": [10.2, 10.8, 11.2],
            "volume": [1000, 1100, 1200],
        },
        index=dates,
    )
    return {"portfolio": portfolio, "data": data}


def test_walk_forward_metrics_include_turnover(sample_trade_portfolio):
    config = BacktestingConfig(
        symbol="TEST",
        exchange="SIM",
        timeframe="1d",
        data_dir="/tmp",
        enable_gpu_acceleration=False,
        enable_memory_optimization=False,
        enable_parallel_processing=False,
        capacity_limit=0.5,
        market_impact_coefficient=0.1,
    )

    validator = WalkForwardValidator(config)
    metrics = validator._calculate_metrics(
        sample_trade_portfolio["portfolio"],
        sample_trade_portfolio["data"],
    )

    expected_turnover = (500 * 10.0 + 500 * 11.0) / ((100000.0 + 100500.0) / 2)
    expected_market_impact = expected_turnover * 0.1
    expected_total_return = ((100500.0 - 100000.0) / 100000.0) - expected_market_impact

    assert metrics["turnover"] == pytest.approx(expected_turnover, rel=1e-6)
    assert metrics["average_holding_period_days"] == pytest.approx(2.0, rel=1e-6)
    assert metrics["capacity_utilization"] == pytest.approx(expected_turnover / 0.5, rel=1e-6)
    assert metrics["market_impact_cost"] == pytest.approx(expected_market_impact, rel=1e-6)
    assert metrics["total_return"] == pytest.approx(expected_total_return, rel=1e-6)


def test_real_engine_turnover_metrics():
    config = DummyConfig()
    engine = RealBacktestingEngine.__new__(RealBacktestingEngine)
    engine.config = config
    engine.logger = logging.getLogger("RealBacktestingEngineTest")

    equity_curve = [100000.0, 101000.0, 102000.0]
    trade_log = [
        {"timestamp": pd.Timestamp("2020-01-01"), "action": "BUY", "price": 100.0, "shares": 100},
        {
            "timestamp": pd.Timestamp("2020-01-03"),
            "action": "SELL",
            "price": 102.0,
            "shares": 100,
            "profit": 200.0,
        },
    ]

    metrics = engine._calculate_performance_metrics(
        equity_curve,
        trade_log,
        trade_profits=[200.0],
    )

    expected_turnover = (100 * 100.0 + 100 * 102.0) / ((100000.0 + 102000.0) / 2)
    expected_market_impact = expected_turnover * 0.1

    assert metrics["turnover"] == pytest.approx(expected_turnover, rel=1e-6)
    assert metrics["average_holding_period_days"] == pytest.approx(2.0, rel=1e-6)
    assert metrics["capacity_utilization"] == pytest.approx(expected_turnover / 0.2, rel=1e-6)
    assert metrics["market_impact_cost"] == pytest.approx(expected_market_impact, rel=1e-6)
    assert metrics["total_return"] == pytest.approx(((102000.0 - 100000.0) / 100000.0) - expected_market_impact, rel=1e-6)
    assert metrics["raw_total_return"] == pytest.approx((102000.0 - 100000.0) / 100000.0, rel=1e-6)
