
import sys
import types
from contextlib import contextmanager

if 'torch' not in sys.modules:
    torch_stub = types.ModuleType('torch')

    class _Tensor:  # minimal placeholder
        pass

    torch_stub.Tensor = _Tensor
    torch_stub.float32 = 'float32'
    torch_stub.float64 = 'float64'
    torch_stub.device = lambda *args, **kwargs: None

    @contextmanager
    def _no_grad():
        yield

    torch_stub.no_grad = _no_grad
    torch_stub.tanh = lambda x: x
    torch_stub.sigmoid = lambda x: x

    cuda_ns = types.SimpleNamespace(
        is_available=lambda: False,
        empty_cache=lambda: None,
        synchronize=lambda: None,
    )
    torch_stub.cuda = cuda_ns

    mps_ns = types.SimpleNamespace(
        is_available=lambda: False,
        empty_cache=lambda: None,
        synchronize=lambda: None,
    )
    torch_stub.backends = types.SimpleNamespace(mps=mps_ns)
    torch_stub.mps = mps_ns
    torch_stub._C = types.SimpleNamespace(_cuda_emptyCache=lambda: None)

    nn_module = types.ModuleType('torch.nn')

    class _Module:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return args[0] if args else None

    class _ModuleList(list):
        def append(self, module):
            super().append(module)

    class _LSTM(_Module):
        def __call__(self, x, *args, **kwargs):
            return x, None

    class _BatchNorm1d(_Module):
        pass

    class _Linear(_Module):
        pass

    class _Dropout(_Module):
        pass

    nn_module.Module = _Module
    nn_module.ModuleList = _ModuleList
    nn_module.LSTM = _LSTM
    nn_module.BatchNorm1d = _BatchNorm1d
    nn_module.Linear = _Linear
    nn_module.Dropout = _Dropout

    F_module = types.ModuleType('torch.nn.functional')
    F_module.relu = lambda x: x
    F_module.leaky_relu = lambda x, negative_slope=0.01: x
    F_module.elu = lambda x, alpha=1.0: x
    F_module.gelu = lambda x: x

    torch_stub.nn = nn_module
    torch_stub.nn.functional = F_module

    optim_module = types.ModuleType('torch.optim')

    class _Optimizer:
        def __init__(self, *args, **kwargs):
            self.params = args
            self.kwargs = kwargs

        def step(self):
            return None

        def zero_grad(self):
            return None

    def _optimizer_factory(*args, **kwargs):
        return _Optimizer(*args, **kwargs)

    optim_module.Optimizer = _Optimizer
    optim_module.Adam = _optimizer_factory
    optim_module.SGD = _optimizer_factory
    optim_module.lr_scheduler = types.SimpleNamespace(StepLR=lambda *a, **k: None)

    torch_stub.optim = optim_module
    sys.modules['torch'] = torch_stub
    sys.modules['torch.nn'] = nn_module
    sys.modules['torch.nn.functional'] = F_module
    sys.modules['torch.optim'] = optim_module
    sys.modules['torch.optim.lr_scheduler'] = optim_module.lr_scheduler
    sys.modules['torch.backends'] = torch_stub.backends
    sys.modules['torch.backends.mps'] = mps_ns

import types
from pathlib import Path

tas_regime_path = Path(__file__).resolve().parents[2] / 'src/training/steps/market_analysis/tas_regime'
tas_regime_module = types.ModuleType('src.training.steps.market_analysis.tas_regime')
tas_regime_module.__path__ = [str(tas_regime_path)]
sys.modules['src.training.steps.market_analysis.tas_regime'] = tas_regime_module

import numpy as np
import pandas as pd

from src.training.steps.market_analysis.tas_regime.trading.trading_engine import (
    TradingConfig,
    TradingEngine,
)
from src.training.steps.market_analysis.tas_regime.trading.signal_generator import (
    TradingSignalGenerator,
)
from src.training.steps.market_analysis.tas_regime.trading.position_manager import (
    PositionManager,
)
from src.training.steps.market_analysis.tas_regime.trading.risk_manager import (
    RiskManager,
)
from src.training.steps.market_analysis.tas_regime.trading.performance_monitor import (
    TradingPerformanceMonitor,
)


def _build_market_data() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=60, freq="D")
    closes = np.linspace(100, 120, len(index))
    highs = closes + 1
    lows = closes - 1
    volume = np.linspace(1_000, 2_000, len(index))
    return pd.DataFrame(
        {
            "close": closes,
            "high": highs,
            "low": lows,
            "volume": volume,
        },
        index=index,
    )


def test_trading_engine_components_initialise():
    engine = TradingEngine(TradingConfig())

    assert isinstance(engine.signal_generator, TradingSignalGenerator)
    assert isinstance(engine.position_manager, PositionManager)
    assert isinstance(engine.risk_manager, RiskManager)
    assert isinstance(engine.performance_monitor, TradingPerformanceMonitor)


def test_trading_engine_generates_and_executes_signals(monkeypatch):
    config = TradingConfig()
    engine = TradingEngine(config)

    market_data = _build_market_data()
    regime_info = {
        "regime_id": "bullish_regime",
        "confidence": 0.9,
        "symbol": "TEST",
        "architecture": "tree_v1",
    }

    signals = engine.generate_trading_signals(market_data, regime_info)
    assert signals, "Expected at least one signal"

    final_price = float(market_data["close"].iloc[-1])
    monkeypatch.setattr(engine, "_get_current_price", lambda symbol: final_price)

    results = engine.execute_signals(signals)
    assert results, "Expected at least one executed trade"

    for trade in results:
        assert trade.symbol == "TEST"
        assert trade.quantity > 0

    metrics = engine.performance_monitor.get_metrics()
    assert metrics["total_trades"] == len(results)
    assert metrics["equity_curve_length"] >= len(results)
