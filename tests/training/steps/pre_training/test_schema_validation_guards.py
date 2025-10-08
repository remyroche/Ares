import asyncio
from types import SimpleNamespace
from typing import Optional

import pandas as pd
import pytest

from dataclasses import dataclass
from enum import Enum

from src.training.steps.pre_training.validation.schemas import validate_labeled_dataset


class _ValidationStatus(Enum):
    PASSED = "passed"


@dataclass
class _ValidationSummary:
    total_rules: int
    passed: int
    failed: int
    warnings: int
    skipped: int
    critical_failures: int
    overall_status: _ValidationStatus
    quality_score: float
    recommendations: list


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_multi_horizon_component_schema_failure(monkeypatch):
    from src.training.steps.pre_training.multi_horizon_profit_labeler import MultiHorizonProfitLabelerComponent

    component = MultiHorizonProfitLabelerComponent()

    async def _failing_labeling(*args, **kwargs):
        df = pd.DataFrame({"immediate_opportunity": [1]}, index=[0])
        validate_labeled_dataset(df, context="tests.invalid")

    monkeypatch.setattr(component.labeler, "execute_labeling", _failing_labeling)

    result = await component.execute(
        None,
        {"symbol": "ETHUSDT", "exchange": "binance", "timeframe": "1h"},
    )

    assert result.success is False
    assert result.metadata.get("schema_error", {}).get("schema_key") == "labeled_dataset"


@pytest.mark.anyio
async def test_feature_lookback_component_schema_failure(monkeypatch):
    import sys
    import types

    logging_stub = types.ModuleType("logging_standards_stub")
    logging_stub.get_logger = lambda name: SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )
    logging_stub.log_info = lambda *args, **kwargs: None
    logging_stub.log_warning = lambda *args, **kwargs: None
    logging_stub.log_error = lambda *args, **kwargs: None
    logging_stub.log_success = lambda *args, **kwargs: None
    logging_stub.log_debug = lambda *args, **kwargs: None
    logging_stub.LoggingContext = object
    logging_stub.log_step_progress = lambda *args, **kwargs: None
    logging_stub.log_data_info = lambda *args, **kwargs: None
    logging_stub.log_validation_result = lambda *args, **kwargs: None

    optimized_stub = types.ModuleType("optimized_process_engines_stub")
    class _FakeLookbackEngine:
        def __init__(self, *args, **kwargs):
            self.initialized = True

    optimized_stub.OptimizedFeatureLookbackEngine = _FakeLookbackEngine
    optimized_stub.ProcessType = SimpleNamespace(FEATURE_LOOKBACK="feature_lookback")

    market_stub = types.ModuleType("market_analysis_stub")
    market_stub.__path__ = []  # Mark as package

    modular_stub = types.ModuleType("feature_lookback_optimization_modular")
    modular_stub.FeatureLookbackOptimizationComponent = object

    sys.modules["src.training.steps.market_analysis"] = market_stub
    sys.modules["src.training.steps.market_analysis.logging_standards"] = logging_stub
    sys.modules["src.training.steps.market_analysis.optimized_process_engines"] = optimized_stub
    sys.modules[
        "src.training.steps.pre_training.feature_lookback_optimization.feature_lookback_optimization_modular"
    ] = modular_stub

    from src.training.steps.pre_training.feature_lookback_optimization.feature_lookback_optimization import (
        FeatureLookbackOptimizationComponent,
    )

    component = FeatureLookbackOptimizationComponent()

    summary = _ValidationSummary(
        total_rules=0,
        passed=0,
        failed=0,
        warnings=0,
        skipped=0,
        critical_failures=0,
        overall_status=_ValidationStatus.PASSED,
        quality_score=1.0,
        recommendations=[],
    )

    monkeypatch.setattr(
        component.validator,
        "validate_data",
        lambda data, required_columns=None: (True, summary, data),
    )

    bad_market = pd.DataFrame(
        {
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "volume": [10.0],
        }
    )

    result = await component.execute(bad_market, {})

    assert result.success is False
    assert result.metadata.get("schema_error", {}).get("schema_key") == "raw_ohlcv"


@pytest.mark.anyio
async def test_interactive_component_schema_failure(monkeypatch):
    import sys
    import types

    pymc_stub = types.ModuleType("pymc")
    pymc_stub.Model = type("DummyModel", (), {})
    pymc_stub.sample = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "pymc", pymc_stub)

    aesara_stub = types.ModuleType("aesara")
    tensor_stub = types.ModuleType("aesara.tensor")
    aesara_stub.tensor = tensor_stub
    monkeypatch.setitem(sys.modules, "aesara", aesara_stub)
    monkeypatch.setitem(sys.modules, "aesara.tensor", tensor_stub)

    base_component_stub = types.ModuleType("base_component_stub")

    @dataclass
    class _StubComponentResult:
        success: bool
        artifacts: Optional[dict] = None
        metadata: Optional[dict] = None
        error_message: Optional[str] = None
        execution_time: float = 0.0

    @dataclass
    class _StubComponentConfig:
        symbol: str = "ETHUSDT"
        exchange: str = "binance"
        timeframe: str = "1h"

    class _StubBaseComponent:
        def __init__(self, *args, **kwargs):
            pass

    base_component_stub.ComponentResult = _StubComponentResult
    base_component_stub.BaseComponent = _StubBaseComponent
    base_component_stub.ComponentConfig = _StubComponentConfig
    monkeypatch.setitem(
        sys.modules,
        "src.training.steps.pre_training.components.base_component",
        base_component_stub,
    )

    component_factory_stub = types.ModuleType("component_factory_stub")
    component_factory_stub.ComponentFactory = type(
        "ComponentFactory",
        (),
        {"create": staticmethod(lambda *args, **kwargs: None)},
    )
    monkeypatch.setitem(
        sys.modules,
        "src.training.steps.pre_training.components.component_factory",
        component_factory_stub,
    )

    from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.interactive_feature_generation_component import (
        InteractiveFeatureGenerationComponent,
    )

    module = sys.modules[
        "src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.interactive_feature_generation_component"
    ]
    module.ComponentResult = _StubComponentResult
    module.BaseComponent = _StubBaseComponent

    component = InteractiveFeatureGenerationComponent.__new__(InteractiveFeatureGenerationComponent)
    component.config = SimpleNamespace(symbol="ETHUSDT", exchange="binance", timeframe="1h")
    component.logger = SimpleNamespace(error=lambda *args, **kwargs: None)

    async def _noop_generate(*_args, **_kwargs):
        return SimpleNamespace(
            success=True,
            feature_names=[],
            selected_features=[],
            interaction_features=pd.DataFrame(),
            cross_timeframe_features=pd.DataFrame(),
            execution_time=0.0,
            memory_usage_mb=0.0,
            error_message=None,
            stage_results={},
            performance_metrics={},
            artifacts={},
            features=pd.DataFrame(),
        )

    component.orchestrator = SimpleNamespace(generate_features=_noop_generate)
    component._validate_inputs = lambda *args, **kwargs: None
    component._update_orchestrator_config = lambda *args, **kwargs: None

    data = pd.DataFrame({"feature_a": ["bad"]})

    result = await component.execute({"data": data}, {})

    assert result.success is False
    assert result.metadata.get("schema_error", {}).get("schema_key") == "engineered_features"


@pytest.mark.anyio
async def test_final_feature_selection_schema_failure():
    import sys
    import types
    from src.training.steps.pre_training.components.base_component import ComponentConfig

    logging_stub = types.ModuleType("logging_standards_stub")
    logging_stub.get_logger = lambda name: SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )
    logging_stub.log_info = lambda *args, **kwargs: None
    logging_stub.log_warning = lambda *args, **kwargs: None
    logging_stub.log_error = lambda *args, **kwargs: None
    logging_stub.log_success = lambda *args, **kwargs: None
    logging_stub.log_debug = lambda *args, **kwargs: None
    logging_stub.LoggingContext = object
    logging_stub.log_step_progress = lambda *args, **kwargs: None
    logging_stub.log_data_info = lambda *args, **kwargs: None
    logging_stub.log_validation_result = lambda *args, **kwargs: None

    optimized_stub = types.ModuleType("optimized_process_engines_stub")
    class _FakeSelectionEngine:
        def __init__(self, *args, **kwargs):
            self.initialized = True

    optimized_stub.OptimizedFeatureSelectionEngine = _FakeSelectionEngine
    optimized_stub.ProcessType = SimpleNamespace(FINAL_FEATURE_SELECTION="final_feature_selection")

    sys.modules["src.training.steps.market_analysis.logging_standards"] = logging_stub
    sys.modules["src.training.steps.market_analysis.optimized_process_engines"] = optimized_stub

    from src.training.steps.pre_training.components.final_feature_selection import FinalFeatureSelectionComponent

    component = FinalFeatureSelectionComponent.__new__(FinalFeatureSelectionComponent)
    component.config = ComponentConfig()
    component.memory_optimizer = SimpleNamespace(
        memory_pressure=0.0,
        _apply_memory_optimizations=lambda: None,
        _light_memory_cleanup=lambda: None,
    )
    component.adaptive_engine = SimpleNamespace(get_optimal_strategy=lambda *args, **kwargs: {})
    component.hardware_manager = SimpleNamespace(get_optimal_config=lambda *args, **kwargs: {})

    result = await component.execute(pd.DataFrame({"feature": ["bad"]}), {})

    assert result.success is False
    assert result.metadata.get("schema_error", {}).get("schema_key") == "engineered_features"
