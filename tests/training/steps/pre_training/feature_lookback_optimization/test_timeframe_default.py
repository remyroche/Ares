import asyncio
import sys
import types
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd
import pytest

# The production module uses a legacy relative import path that expects
# ``src.training.market_analysis`` to exist. In tests we provide lightweight
# module shims that point to the real implementation under
# ``src.training.steps.market_analysis`` so the import succeeds without
# modifying production code.
market_analysis_shim = sys.modules.setdefault(
    "src.training.market_analysis", types.ModuleType("src.training.market_analysis")
)
market_analysis_shim.__path__ = []

components_shim = sys.modules.setdefault(
    "src.training.market_analysis.components",
    types.ModuleType("src.training.market_analysis.components"),
)
components_shim.__path__ = []
market_analysis_shim.components = components_shim

base_component_module = types.ModuleType(
    "src.training.market_analysis.components.base_component"
)


class _BaseMarketAnalysisComponent:
    def __init__(self, config=None):
        self.config = config

    def validate_artifacts(self, artifacts):
        return True


class _ComponentConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _ComponentResult:
    def __init__(
        self,
        success: bool,
        artifacts=None,
        execution_time: float | None = None,
        error: Exception | None = None,
        metrics: dict | None = None,
        warnings: list | None = None,
        metadata=None,
    ):
        self.success = success
        self.artifacts = artifacts or {}
        self.execution_time = execution_time
        self.error = error
        self.metrics = metrics or {}
        self.warnings = warnings or []
        self.metadata = metadata or {}

        if (self.success and self.error is not None) or (not self.success and self.error is None):
            raise ValueError("Invalid ComponentResult state for stub")


base_component_module.BaseMarketAnalysisComponent = _BaseMarketAnalysisComponent
base_component_module.ComponentConfig = _ComponentConfig
base_component_module.ComponentResult = _ComponentResult

sys.modules[
    "src.training.market_analysis.components.base_component"
] = base_component_module
components_shim.base_component = base_component_module

package_name = "src.training.steps.pre_training.feature_lookback_optimization"
package_path = Path(__file__).resolve().parents[5] / "src" / "training" / "steps" / "pre_training" / "feature_lookback_optimization"

feature_package = sys.modules.setdefault(
    package_name,
    types.ModuleType(package_name),
)
feature_package.__path__ = [str(package_path)]

module_name = f"{package_name}.feature_lookback_optimization"
module_spec = spec_from_file_location(module_name, package_path / "feature_lookback_optimization.py")
feature_module = module_from_spec(module_spec)
feature_module.__package__ = package_name
sys.modules[module_name] = feature_module
module_spec.loader.exec_module(feature_module)

FeatureLookbackOptimizationComponent = feature_module.FeatureLookbackOptimizationComponent
OptimizationStatus = feature_module.OptimizationStatus


class _NoOpMonitoring:
    """Monitoring stub that safely ignores all metric calls."""

    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return None

        return _noop


class _FailingValidationFramework:
    """Validation stub that raises to short-circuit execution after setup."""

    def validate_data(self, *_args, **_kwargs):
        raise RuntimeError("validation not available")


def test_execute_defaults_to_fifteen_minute_timeframe():
    component = FeatureLookbackOptimizationComponent.__new__(FeatureLookbackOptimizationComponent)

    component.config = types.SimpleNamespace(symbol="BTCUSDT", exchange="binance", timeframe="5m")
    component.monitoring = _NoOpMonitoring()
    component._monitor_performance = lambda *args, **kwargs: None
    component.performance_monitor = {"error_counts": 0}
    component.optimization_status = OptimizationStatus.PENDING
    component.validation_framework = _FailingValidationFramework()

    async def _fake_enhanced_data_handling(_data, _pipeline_state):
        return pd.DataFrame(
            {
                "open": [1, 2],
                "high": [1, 2],
                "low": [1, 2],
                "close": [1, 2],
                "volume": [1, 2],
            }
        )

    component._enhanced_data_handling = _fake_enhanced_data_handling

    recorded_timeframes = {}

    def _fake_load_recent_labeling_results(symbol, exchange, timeframe):
        recorded_timeframes["labeling"] = timeframe
        return {}

    def _fake_load_recent_regime_results(symbol, exchange, timeframe):
        recorded_timeframes["regime"] = timeframe
        return {}

    component._load_recent_labeling_results = _fake_load_recent_labeling_results
    component._load_recent_regime_splitting_results = _fake_load_recent_regime_results

    pipeline_state = {"symbol": "BTCUSDT", "exchange": "binance"}

    result = asyncio.run(component.execute(data=None, pipeline_state=pipeline_state))

    assert recorded_timeframes["labeling"] == "15m"
    assert recorded_timeframes["regime"] == "15m"
    assert result.success is False
