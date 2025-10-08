import logging
import sys
import types

import pytest


pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


class _DummyMemoryOptimizer:
    def __init__(self) -> None:
        self.memory_pressure = 0.1

    def _apply_memory_optimizations(self) -> None:  # pragma: no cover - behaviour mocked
        self.memory_pressure = 0.05

    def _light_memory_cleanup(self) -> None:  # pragma: no cover - behaviour mocked
        self.memory_pressure = 0.0


class _DummyAdaptiveEngine:
    def get_optimal_strategy(self, *_, **__):  # pragma: no cover - behaviour mocked
        return {}


class _DummyHardwareManager:
    def get_optimal_config(self, *_args, **_kwargs):  # pragma: no cover - behaviour mocked
        return {}


class _DummyOptimizedEngine:  # pragma: no cover - behaviour mocked
    def __init__(self, *_, **__):
        self.initialized = True


@pytest.fixture
def component(monkeypatch):
    logging_stub = types.ModuleType("logging_standards_stub")
    base_logger = logging.getLogger("FinalFeatureSelectionComponentTest")

    logging_stub.get_logger = lambda name: logging.getLogger(name)
    logging_stub.log_info = lambda message: base_logger.info(message)
    logging_stub.log_warning = lambda message: base_logger.warning(message)
    logging_stub.log_error = lambda message: base_logger.error(message)
    logging_stub.log_success = lambda message: base_logger.info(message)
    logging_stub.log_debug = lambda message: base_logger.debug(message)
    logging_stub.LoggingContext = object
    logging_stub.log_step_progress = lambda *args, **kwargs: None
    logging_stub.log_data_info = lambda *args, **kwargs: None
    logging_stub.log_validation_result = lambda *args, **kwargs: None

    optimized_stub = types.ModuleType("optimized_process_engines_stub")

    class _StubProcessType:
        FINAL_FEATURE_SELECTION = "final_feature_selection"

    class _StubOptimizedEngine:
        def __init__(self, *_, **__):
            self.initialized = True

    optimized_stub.OptimizedFeatureSelectionEngine = _StubOptimizedEngine
    optimized_stub.ProcessType = _StubProcessType

    sys.modules[
        "src.training.steps.market_analysis.logging_standards"
    ] = logging_stub
    sys.modules[
        "src.training.steps.market_analysis.optimized_process_engines"
    ] = optimized_stub

    final_step_stub = types.ModuleType("final_feature_selection_step_stub")

    async def _stub_run_final_feature_selection_step(*_, **__):
        return True

    final_step_stub.run_final_feature_selection_step = _stub_run_final_feature_selection_step

    sys.modules[
        "src.training.steps.pre_training.final_feature_selection_step"
    ] = final_step_stub

    import src.training.steps.pre_training.multi_horizon_profit_labeler as mh_module

    if not hasattr(mh_module, "create_multi_horizon_labeler"):
        monkeypatch.setattr(
            mh_module,
            "create_multi_horizon_labeler",
            lambda *args, **kwargs: None,
            raising=False,
        )
    if not hasattr(mh_module, "apply_multi_horizon_labeling"):
        monkeypatch.setattr(
            mh_module,
            "apply_multi_horizon_labeling",
            lambda *args, **kwargs: None,
            raising=False,
        )

    from src.training.steps.pre_training.components import final_feature_selection as component_module

    monkeypatch.setattr(
        component_module,
        "get_m1_memory_optimizer",
        lambda memory_limit_gb=8.0: _DummyMemoryOptimizer(),
    )
    monkeypatch.setattr(
        component_module,
        "AdaptiveOptimizationEngine",
        lambda: _DummyAdaptiveEngine(),
    )
    monkeypatch.setattr(
        component_module,
        "UnifiedHardwareManager",
        lambda: _DummyHardwareManager(),
    )
    monkeypatch.setattr(
        component_module,
        "OptimizedFeatureSelectionEngine",
        lambda *args, **kwargs: _DummyOptimizedEngine(),
    )

    from src.training.steps.pre_training.components.final_feature_selection import (
        FinalFeatureSelectionComponent,
    )

    return FinalFeatureSelectionComponent()


def _patch_async_method(monkeypatch, instance, method_name, async_fn):
    monkeypatch.setattr(instance, method_name, types.MethodType(async_fn, instance))


async def test_execute_successful_persistence(monkeypatch, component):
    async def _save_artifacts(self, artifacts, metadata):
        return {"final_feature_selection_result": "path/to/result.json"}

    _patch_async_method(monkeypatch, component, "save_artifacts", _save_artifacts)

    result = await component.execute({}, {})

    assert result.success is True
    assert result.metadata["artifacts_saved_persistently"] is True
    assert result.metadata["artifact_persistence_paths"] == {
        "final_feature_selection_result": "path/to/result.json"
    }


async def test_execute_failure_when_persistence_missing(monkeypatch, component, caplog):
    async def _save_artifacts(self, artifacts, metadata):
        return {}

    _patch_async_method(monkeypatch, component, "save_artifacts", _save_artifacts)

    caplog.set_level("ERROR")

    result = await component.execute({}, {})

    assert result.success is False
    assert result.metadata["artifacts_saved_persistently"] is False
    assert "Artifacts were not saved persistently" in caplog.text
