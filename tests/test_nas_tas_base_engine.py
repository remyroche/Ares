"""Unit tests ensuring NAS and TAS engines reuse the shared base implementation."""

from __future__ import annotations

from typing import Any, Dict
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

def _ensure_namespace_module(name: str, path: str) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [path]
    module.__package__ = name
    try:
        import importlib.machinery

        module.__spec__ = importlib.machinery.ModuleSpec(
            name=name, loader=None, is_package=True
        )
    except Exception:  # pragma: no cover - defensive guard for older interpreters
        module.__spec__ = None
    sys.modules[name] = module


_ensure_namespace_module("src", os.path.join(PROJECT_ROOT, "src"))
_ensure_namespace_module("src.utils", os.path.join(PROJECT_ROOT, "src", "utils"))
_ensure_namespace_module(
    "src.utils.nas_tas", os.path.join(PROJECT_ROOT, "src", "utils", "nas_tas")
)

def _register_module(name: str, attrs: Dict[str, Any]) -> None:
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module


class _DummyContextManager:
    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    def __enter__(self) -> "_DummyContextManager":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # type: ignore[override]
        return False


def _noop(*_: Any, **__: Any) -> None:
    return None


def _identity(value):
    return value


def _ensure_test_stubs() -> None:
    common_ops_attrs = {
        "cleanup_m1_optimizers": _noop,
        "get_m1_cpu_optimizer": lambda: None,
        "get_m1_gpu_manager": lambda: None,
        "get_m1_memory_optimizer": lambda: None,
        "get_memory_usage": lambda: 0,
        "gpu_context": lambda name="": _DummyContextManager(),
        "integrate_with_m1_optimizers": lambda: {"success": False},
        "memory_checkpoint": lambda name="": _DummyContextManager(),
        "optimize_memory": _noop,
        "guard_dataframe_nulls": lambda df, threshold=0.0: df,
        "optimize_dataframe_dtypes": lambda df: df,
        "validate_dataframe_columns": lambda df, required: all(
            col in df.columns for col in required
        ),
        "calculate_data_quality_metrics": lambda df: {"rows": len(df)},
        "create_data_quality_report": lambda df: {"columns": list(df.columns)},
        "safe_copy": lambda df: df.copy(),
    }
    _register_module("src.utils.common_operations", common_ops_attrs)

    class _CommonUtilities:
        pass

    _register_module(
        "src.utils.common_utilities", {"CommonUtilities": _CommonUtilities}
    )

    class _DummyKlinesManager:
        def read_data(self, **_: Any):
            raise RuntimeError("klines access not available in test stub")

    _register_module(
        "src.utils.data.klines_parquet",
        {
            "get_klines_manager": lambda: _DummyKlinesManager(),
            "validate_klines_data": lambda df: {"valid": True, "errors": []},
        },
    )

    def _safe_array(value: Any) -> np.ndarray:
        return np.asarray(value)

    math_attrs = {
        "MathValidation": type("MathValidation", (), {}),
        "safe_mean": lambda arr: float(np.mean(_safe_array(arr))) if len(arr) else 0.0,
        "safe_std": lambda arr: float(np.std(_safe_array(arr))) if len(arr) else 0.0,
        "safe_percentile": lambda arr, q: float(np.percentile(_safe_array(arr), q))
        if len(arr)
        else 0.0,
        "validate_correlation_matrix": lambda matrix: matrix.size > 0,
        "validate_finite": lambda value, *_: float(value),
        "safe_divide": lambda a, b: float(a) / float(b) if b else 0.0,
        "safe_log": lambda x: float(np.log(x)),
        "safe_power": lambda x, p: float(np.power(x, p)),
        "safe_sqrt": lambda x: float(np.sqrt(x)),
        "safe_weighted_average": lambda values, weights: float(
            np.average(np.asarray(values), weights=np.asarray(weights))
        ),
        "safe_kelly_calculation": lambda win_rate, avg_win, avg_loss: 0.0,
        "validate_numeric_array": lambda arr, *_: _safe_array(arr),
    }
    _register_module("src.utils.math_validation", math_attrs)

    class _MatrixOperations:
        def normalize_matrix(self, matrix):
            return matrix

        def calculate_rolling_returns(self, data):
            return np.diff(data[:, 3], prepend=data[0, 3])

        def calculate_rolling_volatility(self, returns):
            return np.abs(returns)

        def calculate_trend_strength(self, data):
            return np.gradient(data[:, 3])

    class _EnhancedMatrixOperations:
        def add_polynomial_features(self, data, degree=2):
            return np.column_stack([data, data ** degree])

        def add_technical_features(self, data):
            return data

    class _BatchMatrixOperations:
        pass

    class _VectorizedCore:
        def compute_performance_metric(self, features, complexity, depth, width):
            return float(np.mean(features) * complexity * (depth + width))

        def compute_strategy_performance(self, features, entry, exit):
            return float(np.mean(features) * (entry + exit))

        def classify_regimes(self, features):
            return (features[:, 0] > np.median(features[:, 0])).astype(int)

    _register_module(
        "src.utils.matrix_operations.unified_operations",
        {"MatrixOperations": _MatrixOperations},
    )
    _register_module(
        "src.utils.matrix_operations.enhanced_operations",
        {"EnhancedMatrixOperations": _EnhancedMatrixOperations},
    )
    _register_module(
        "src.utils.matrix_operations.batch_operations",
        {"BatchMatrixOperations": _BatchMatrixOperations},
    )
    _register_module(
        "src.utils.matrix_operations.vectorized_core",
        {"VectorizedCore": _VectorizedCore},
    )
    _register_module(
        "src.utils.matrix_operations.convenience",
        {"MatrixConvenience": type("MatrixConvenience", (), {"add_trading_features": _identity})},
    )

    class _Serializer:
        def save(self, data, path):
            return True

        def load(self, path):
            return {}

    _register_module(
        "src.utils.serialization_utils", {"UniversalSerializer": _Serializer}
    )

    tprint_attrs = {
        "LogLevel": type("LogLevel", (), {"INFO": 1}),
        "tprint_debug": _noop,
        "tprint_error": _noop,
        "tprint_info": _noop,
        "tprint_logged": lambda *a, **k: (lambda cls: cls),
        "tprint_progress": _noop,
        "tprint_success": _noop,
        "tprint_timer": lambda name: (lambda fn: fn),
        "tprint_warning": _noop,
        "tprint_structured": _noop,
    }
    _register_module("src.utils.tprint", tprint_attrs)

    class _BayesianOptimizer:
        def configure(self, **_: Any) -> None:
            self._trial = 0

        def suggest(self) -> Dict[str, float]:
            self._trial += 1
            return {"score": float(self._trial)}

        def update(self, params: Dict[str, float], score: float) -> None:
            pass

    class _GridOptimizer:
        def generate_grid(self, search_space: Dict[str, Any], max_trials: int):
            return [{"score": float(i + 1)} for i in range(max_trials)]

    class _HierarchicalHPO(_BayesianOptimizer):
        pass

    _register_module(
        "src.utils.ml_common.optimization.bayesian_entry_timing_optimizer",
        {"BayesianEntryTimingOptimizer": _BayesianOptimizer},
    )
    _register_module(
        "src.utils.ml_common.optimization.grid_utils",
        {"GridSearchOptimizer": _GridOptimizer},
    )
    _register_module(
        "src.utils.ml_common.optimization.hierarchical_hpo",
        {"HierarchicalHPO": _HierarchicalHPO},
    )
    _register_module(
        "src.utils.ml_common.optimization.regime_specific_tpsl_optimizer",
        {"RegimeSpecificTPSLOptimizer": type("RegimeSpecificTPSLOptimizer", (), {})},
    )
    _register_module(
        "src.utils.nas_tas.ml_common.optimization.bayesian_entry_timing_optimizer",
        {"BayesianEntryTimingOptimizer": _BayesianOptimizer},
    )
    _register_module(
        "src.utils.nas_tas.ml_common.optimization.grid_utils",
        {"GridSearchOptimizer": _GridOptimizer},
    )
    _register_module(
        "src.utils.nas_tas.ml_common.optimization.hierarchical_hpo",
        {"HierarchicalHPO": _HierarchicalHPO},
    )
    _register_module(
        "src.utils.nas_tas.ml_common.optimization.regime_specific_tpsl_optimizer",
        {"RegimeSpecificTPSLOptimizer": type("RegimeSpecificTPSLOptimizer", (), {})},
    )

    class _FinancialMetrics:
        def __init__(self, sharpe_ratio: float = 0.5) -> None:
            self.sharpe_ratio = sharpe_ratio

    _register_module(
        "src.utils.ml_common.optimization.shared_utils.evaluation_metrics",
        {"FinancialMetricCalculator": lambda: type("Calculator", (), {"calculate": lambda self, **_: _FinancialMetrics()})()},
    )
    _register_module(
        "src.utils.nas_tas.ml_common.optimization.shared_utils.evaluation_metrics",
        {"FinancialMetricCalculator": lambda: type("Calculator", (), {"calculate": lambda self, **_: _FinancialMetrics()})()},
    )

    data_attrs = {
        "BasicReturnsEngineer": type(
            "BasicReturnsEngineer", (), {"add_basic_returns": lambda self, df: df}
        ),
        "FeatureEngineer": type(
            "FeatureEngineer",
            (),
            {
                "add_technical_indicators": lambda self, df: df,
                "add_price_features": lambda self, df: df,
                "add_volume_features": lambda self, df: df,
                "add_time_features": lambda self, df: df,
            },
        ),
        "GapDetector": type("GapDetector", (), {"detect_gaps": lambda self, df: []}),
        "UnifiedDataUtils": type(
            "UnifiedDataUtils",
            (),
            {
                "standardize_data": lambda self, df: df,
                "add_derived_features": lambda self, df: df,
            },
        ),
        "DataProcessor": type("DataProcessor", (), {}),
    }
    _register_module("src.utils.data.basic_returns_engineer", {"BasicReturnsEngineer": data_attrs["BasicReturnsEngineer"]})
    _register_module("src.utils.data.feature_engineer", {"FeatureEngineer": data_attrs["FeatureEngineer"]})
    _register_module("src.utils.data.gap_detector", {"GapDetector": data_attrs["GapDetector"]})
    _register_module(
        "src.utils.data.unified_data_utils", {"UnifiedDataUtils": data_attrs["UnifiedDataUtils"]}
    )
    _register_module(
        "src.utils.data.processing.data_processing", {"DataProcessor": data_attrs["DataProcessor"]}
    )


_ensure_test_stubs()

from src.utils.nas_tas.core import BaseSearchEngine, NASEngine, TASEngine


def _make_test_market_data(rows: int = 12) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="1min")
    return pd.DataFrame(
        {
            "open": np.linspace(1.0, 2.0, rows),
            "high": np.linspace(1.1, 2.1, rows),
            "low": np.linspace(0.9, 1.9, rows),
            "close": np.linspace(1.05, 2.05, rows),
            "volume": np.linspace(100.0, 200.0, rows),
        },
        index=index,
    )


class _DummyFinancialMetrics:
    def __init__(self, sharpe_ratio: float = 0.5) -> None:
        self.sharpe_ratio = sharpe_ratio


def test_nas_engine_uses_base_search(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = NASEngine(config={})
    assert isinstance(engine, BaseSearchEngine)

    market_data = _make_test_market_data()

    monkeypatch.setattr(engine, "_create_feature_matrix", lambda data, **_: np.ones((len(data), 2)))
    monkeypatch.setattr(engine, "_validate_feature_matrix", lambda matrix: True)
    monkeypatch.setattr(
        engine,
        "_compute_score",
        lambda features, params, **_: float(params.get("score", 0.0)),
    )

    class _DummyBayesian:
        def __init__(self) -> None:
            self._trial = 0

        def configure(self, **_: object) -> None:
            self._trial = 0

        def suggest(self) -> Dict[str, float]:
            suggestion = {"score": float(self._trial + 1)}
            self._trial += 1
            return suggestion

        def update(self, params: Dict[str, float], score: float) -> None:
            assert "score" in params
            assert score == params["score"]

    engine.bayesian_optimizer = _DummyBayesian()

    results = engine.search_architectures(
        market_data,
        search_space={},
        optimization_method="bayesian_tpe",
        n_trials=3,
    )

    assert results["best_architecture"]["score"] == pytest.approx(3.0)
    assert results["performance_metrics"]["improvement_rate"] == pytest.approx(1.0)
    engine.cleanup()


def test_tas_engine_uses_base_search(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = TASEngine(config={})
    assert isinstance(engine, BaseSearchEngine)

    market_data = _make_test_market_data()

    monkeypatch.setattr(engine, "_create_feature_matrix", lambda data, **_: np.ones((len(data), 2)))
    monkeypatch.setattr(engine, "_validate_feature_matrix", lambda matrix: True)
    monkeypatch.setattr(
        engine,
        "_compute_score",
        lambda features, params, **_: float(params.get("score", 0.0)),
    )

    class _DummyGrid:
        def generate_grid(self, search_space, max_trials):
            return [
                {"score": 0.1},
                {"score": 0.4},
                {"score": -0.2},
            ]

    engine.grid_optimizer = _DummyGrid()
    monkeypatch.setattr(
        engine.financial_metric_calculator,
        "calculate",
        lambda **_: _DummyFinancialMetrics(sharpe_ratio=0.3),
    )

    results = engine.search_strategies(
        market_data,
        search_space={},
        optimization_method="grid",
        n_trials=3,
        include_regime_specific=False,
    )

    assert results["best_strategy"]["score"] == pytest.approx(0.4)
    assert len(engine.strategy_history) == 1
    engine.cleanup()
