import asyncio
import importlib
import sys
import types

import pytest

from src.training.steps.pre_training.components import base_component as base_component_module
from src.training.steps.pre_training.components.base_component import (
    BasePreTrainingComponent,
    ComponentResult,
)


class _StubComponent(BasePreTrainingComponent):
    """Minimal component for testing base functionality."""

    def get_required_artifacts(self):
        return ["test_artifact"]

    async def execute(self, data, pipeline_state):
        return ComponentResult(success=True)


class _StubMemoryOptimizer:
    memory_pressure = 0.0

    def _apply_memory_optimizations(self):
        pass

    def _light_memory_cleanup(self):
        pass


class _StubAdaptiveEngine:
    def get_optimal_strategy(self, *args, **kwargs):
        return {
            "hardware_accelerated": True,
            "memory_efficient": True,
            "parallel_processing": False,
        }


class _StubHardwareManager:
    def get_optimal_config(self, *args, **kwargs):
        return {"device": "stub"}


class _StubOptimizedEngine:
    pass


def test_save_artifacts_propagates_errors(monkeypatch):
    component = _StubComponent()

    def _fail(*args, **kwargs):
        raise RuntimeError("storage backend failure")

    monkeypatch.setattr(base_component_module, "persist_artifacts", _fail)

    with pytest.raises(RuntimeError, match="storage backend failure"):
        asyncio.run(component.save_artifacts({"test_artifact": {"value": 1}}, {}))


def test_final_feature_selection_fails_when_artifact_save_fails(monkeypatch):
    # Provide lightweight stubs for the market_analysis package before importing the component
    logging_module = types.ModuleType("src.training.steps.market_analysis.logging_standards")

    def _noop(*args, **kwargs):
        return None

    class _LoggingContext:
        pass

    logging_module.get_logger = lambda name: types.SimpleNamespace(info=_noop, debug=_noop)
    logging_module.log_info = _noop
    logging_module.log_warning = _noop
    logging_module.log_error = _noop
    logging_module.log_success = _noop
    logging_module.log_debug = _noop
    logging_module.LoggingContext = _LoggingContext
    logging_module.log_step_progress = _noop
    logging_module.log_data_info = _noop
    logging_module.log_validation_result = _noop

    optimized_module = types.ModuleType("src.training.steps.market_analysis.optimized_process_engines")
    optimized_module.OptimizedFeatureSelectionEngine = _StubOptimizedEngine
    optimized_module.ProcessType = types.SimpleNamespace(FINAL_FEATURE_SELECTION="final_feature_selection")

    market_analysis_pkg = types.ModuleType("src.training.steps.market_analysis")
    market_analysis_pkg.__path__ = []
    market_analysis_pkg.logging_standards = logging_module
    market_analysis_pkg.optimized_process_engines = optimized_module

    monkeypatch.setitem(sys.modules, "src.training.steps.market_analysis", market_analysis_pkg)
    monkeypatch.setitem(sys.modules, "src.training.steps.market_analysis.logging_standards", logging_module)
    monkeypatch.setitem(sys.modules, "src.training.steps.market_analysis.optimized_process_engines", optimized_module)

    # Ensure a clean import of the component module
    sys.modules.pop("src.training.steps.pre_training.components.final_feature_selection", None)
    ffs_module = importlib.import_module("src.training.steps.pre_training.components.final_feature_selection")

    # Patch heavy dependencies with lightweight stubs
    monkeypatch.setattr(ffs_module, "get_m1_memory_optimizer", lambda memory_limit_gb=8.0: _StubMemoryOptimizer())
    monkeypatch.setattr(ffs_module, "AdaptiveOptimizationEngine", lambda: _StubAdaptiveEngine())
    monkeypatch.setattr(ffs_module, "UnifiedHardwareManager", lambda: _StubHardwareManager())
    monkeypatch.setattr(ffs_module, "OptimizedFeatureSelectionEngine", lambda **kwargs: _StubOptimizedEngine())

    # Capture error logging to ensure component records the failure
    error_logs = []
    monkeypatch.setattr(ffs_module, "log_info", lambda *args, **kwargs: None)
    monkeypatch.setattr(ffs_module, "log_warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(ffs_module, "log_debug", lambda *args, **kwargs: None)
    monkeypatch.setattr(ffs_module, "log_success", lambda *args, **kwargs: None)
    monkeypatch.setattr(ffs_module, "log_error", lambda message: error_logs.append(message))

    # Provide a lightweight implementation of the async selection step
    fake_module = types.ModuleType("src.training.steps.pre_training.final_feature_selection_step")

    async def _fake_run_final_feature_selection_step(**kwargs):
        return True

    fake_module.run_final_feature_selection_step = _fake_run_final_feature_selection_step
    monkeypatch.setitem(sys.modules, "src.training.steps.pre_training.final_feature_selection_step", fake_module)

    component = ffs_module.FinalFeatureSelectionComponent()

    def _fail(*args, **kwargs):
        raise RuntimeError("artifact persistence failure")

    monkeypatch.setattr(base_component_module, "persist_artifacts", _fail)

    result = asyncio.run(component.execute(data={}, pipeline_state={}))

    assert result.success is False
    assert isinstance(result.error, Exception)
    assert "artifact persistence failure" in str(result.error)
    assert result.metrics == {}
    assert any("artifact persistence failure" in warning for warning in result.warnings)
    assert error_logs and "artifact persistence failure" in error_logs[-1]
    assert result.metadata.get("artifacts_saved_persistently") is False
    assert result.metadata.get("artifact_persistence_report") == {}
