import math
import sys
import types
from pathlib import Path
from types import MethodType

import numpy as np
import pandas as pd

torch_stub = types.ModuleType("torch")
torch_stub.__version__ = "0.0"
torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
torch_stub.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
torch_stub.device = lambda *args, **kwargs: None
torch_stub.tensor = lambda *args, **kwargs: None
torch_stub.float32 = float
torch_stub.float64 = float
sys.modules.setdefault("torch", torch_stub)
sys.modules.setdefault("torch.nn", types.ModuleType("torch.nn"))

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.nas_tas.core.nas_engine import NASEngine
from src.utils.nas_tas.core.tas_engine import TASEngine
from src.utils.ml_common.optimization.grid_utils import GridSearchOptimizer


def _dummy_market_data() -> pd.DataFrame:
    values = np.linspace(1.0, 5.0, num=10)
    return pd.DataFrame({
        'open': values,
        'high': values + 0.5,
        'low': values - 0.5,
        'close': values + 0.25,
        'volume': np.linspace(100, 200, num=10)
    })


def _simple_search_space() -> dict:
    return {
        'layers': {'type': 'int', 'low': 1, 'high': 2},
        'dropout': {'type': 'categorical', 'choices': [0.0, 0.25]}
    }


def _make_engine_without_init(cls):
    actual_cls = getattr(cls, "__wrapped__", cls)
    engine = actual_cls.__new__(actual_cls)
    engine.grid_optimizer = GridSearchOptimizer()
    return engine


def test_nas_grid_search_generates_expected_trials():
    engine = _make_engine_without_init(NASEngine)

    def _mock_evaluate(self, data, params):
        return params['layers'] + params['dropout']

    engine._evaluate_architecture = MethodType(_mock_evaluate, engine)

    data = _dummy_market_data()
    search_space = _simple_search_space()

    grid_params = engine.grid_optimizer.generate_grid(search_space, max_trials=3)
    assert len(grid_params) == 3

    best_params, best_score, trials = engine._grid_search(data, search_space, n_trials=3)

    assert len(trials) == len(grid_params)
    trial_params = [trial['params'] for trial in trials]
    for params in trial_params:
        assert params in grid_params

    expected_best = max(grid_params, key=lambda params: params['layers'] + params['dropout'])
    assert best_params == expected_best
    assert math.isclose(best_score, expected_best['layers'] + expected_best['dropout'])


def test_tas_grid_strategy_search_generates_expected_trials():
    engine = _make_engine_without_init(TASEngine)

    def _mock_strategy(self, data, params, regime_analysis=None):
        return params['layers'] - params['dropout']

    engine._evaluate_strategy = MethodType(_mock_strategy, engine)

    data = _dummy_market_data()
    search_space = _simple_search_space()

    grid_params = engine.grid_optimizer.generate_grid(search_space, max_trials=4)
    assert len(grid_params) == 4

    best_params, best_score, trials = engine._grid_strategy_search(
        data,
        search_space,
        n_trials=4,
        regime_analysis=None
    )

    assert len(trials) == len(grid_params)
    trial_params = [trial['params'] for trial in trials]
    for params in trial_params:
        assert params in grid_params

    expected_best = max(grid_params, key=lambda params: params['layers'] - params['dropout'])
    assert best_params == expected_best
    assert math.isclose(best_score, expected_best['layers'] - expected_best['dropout'])
