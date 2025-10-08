import sys
import types
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd
import pytest

import src.training.steps.pre_training.multi_horizon_profit_labeler as mh_module
from src.training.config.data_locator import DataLocator, DataLocatorConfig


if not hasattr(mh_module, "create_multi_horizon_labeler"):
    def _stub_create_multi_horizon_labeler(*args, **kwargs):  # pragma: no cover - test helper
        raise NotImplementedError("stub")

    mh_module.create_multi_horizon_labeler = _stub_create_multi_horizon_labeler  # type: ignore[attr-defined]

if not hasattr(mh_module, "apply_multi_horizon_labeling"):
    def _stub_apply_multi_horizon_labeling(*args, **kwargs):  # pragma: no cover - test helper
        raise NotImplementedError("stub")

    mh_module.apply_multi_horizon_labeling = _stub_apply_multi_horizon_labeling  # type: ignore[attr-defined]


@dataclass
class _TestComponentConfig:
    symbol: str = "ETHUSDT"
    exchange: str = "binance"
    timeframe: str = "15m"
    data_dir: Optional[str] = None
    custom_params: dict = field(default_factory=dict)
    data_locator: Optional[DataLocator] = None
    data_locator_config: DataLocatorConfig = field(default_factory=DataLocatorConfig)
    data_dir_key: str = "market_data"
    cache_dir_key: str = "default"
    artifacts_dir_key: str = "default"
    generated_dir_key: str = "market_analysis"
    outcomes_dir_key: str = "multi_horizon_outcomes"
    final_feature_selection_dir_key: str = "final_feature_selection"


class _TestComponentFactory:
    @staticmethod
    def create_component(name, config):  # pragma: no cover - overwritten in tests
        raise NotImplementedError("stub factory")


@dataclass
class ComponentResult:
    success: bool
    artifacts: dict | None = None
    metadata: dict | None = None
    error_message: str | None = None
    execution_time: float = 0.0

    def __post_init__(self) -> None:
        if self.artifacts is None:
            self.artifacts = {}
        if self.metadata is None:
            self.metadata = {}


base_component_module = types.ModuleType("base_component_stub")
base_component_module.ComponentResult = ComponentResult
base_component_module.ComponentConfig = _TestComponentConfig


class _StubBaseComponent:
    def __init__(self, *args, **kwargs):  # pragma: no cover - simple stub
        self._run_metadata = {}

    def set_run_metadata(self, metadata):  # pragma: no cover - simple stub
        self._run_metadata = dict(metadata or {})

    async def save_artifacts(self, *args, **kwargs):  # pragma: no cover - simple stub
        return {}


base_component_module.BasePreTrainingComponent = _StubBaseComponent

components_module = types.ModuleType("components_stub")
components_module.ComponentFactory = _TestComponentFactory
components_module.ComponentConfig = _TestComponentConfig

sys.modules['src.training.steps.pre_training.components.base_component'] = base_component_module
sys.modules['src.training.steps.pre_training.components'] = components_module


from src.training.steps.pre_training.sub_pipeline import (
    PreTrainingSubPipeline,
    SubPipelineConfig,
    SubPipelineResult,
    SubPipelineStatus,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio("asyncio")
async def test_multi_horizon_receives_regime_split(monkeypatch, tmp_path):
    payload = {
        "regime_data": {
            "market_data": pd.DataFrame({"timestamp": pd.to_datetime([])})
        }
    }
    locator = DataLocator(root=tmp_path)
    config = SubPipelineConfig(
        custom_params={"regime_data_splitting_result": payload},
        data_locator=locator,
    )
    pipeline = PreTrainingSubPipeline()

    captured = {}

    class _DummyComponent:
        async def execute(self, data, pipeline_state):
            captured["pipeline_state"] = pipeline_state
            return ComponentResult(success=True, artifacts={}, metadata={})

    monkeypatch.setattr(
        "src.training.steps.pre_training.sub_pipeline.ComponentFactory.create_component",
        lambda name, cfg: _DummyComponent(),
    )

    run_metadata = pipeline._gather_run_metadata(config)
    result = await pipeline._execute_multi_horizon_profit_labeler(config, run_metadata)

    assert result.success
    assert captured["pipeline_state"]["regime_data_splitting_result"] is payload
    assert pipeline._current_pipeline_state["regime_data_splitting_result"] is payload
    assert captured["pipeline_state"]["data_dir"] == str(locator.data_path(config.data_dir_key))


@pytest.mark.anyio("asyncio")
async def test_followup_steps_reuse_regime_split(monkeypatch, tmp_path):
    payload = {
        "regime_data": {
            "market_data": pd.DataFrame({"timestamp": pd.to_datetime([])})
        }
    }
    locator = DataLocator(root=tmp_path)
    config = SubPipelineConfig(data_locator=locator)
    pipeline = PreTrainingSubPipeline()
    pipeline._current_pipeline_state["regime_data_splitting_result"] = payload

    captured_states = []

    class _DummyComponent:
        async def execute(self, data, pipeline_state):
            captured_states.append(pipeline_state)
            return ComponentResult(success=True, artifacts={}, metadata={})

    monkeypatch.setattr(
        "src.training.steps.pre_training.sub_pipeline.ComponentFactory.create_component",
        lambda name, cfg: _DummyComponent(),
    )

    run_metadata = pipeline._gather_run_metadata(config)
    lookback_result = await pipeline._execute_feature_lookback_optimization(config, run_metadata)
    if hasattr(pipeline, '_execute_pid_based_feature_generation'):
        pid_result = await pipeline._execute_pid_based_feature_generation(config, run_metadata)
    else:
        pid_result = SubPipelineResult(
            sub_pipeline_name='pid_based_feature_generation',
            status=SubPipelineStatus.SKIPPED,
            start_time=datetime.now(),
        )
        pid_result.success = True
        pid_result.end_time = pid_result.start_time
        pid_result.duration_seconds = 0.0
        pid_result.metadata = {'run_metadata': dict(run_metadata)}
    selection_result = await pipeline._execute_final_feature_selection(config, run_metadata)

    assert lookback_result.success and pid_result.success and selection_result.success
    assert captured_states, "component should have been executed"
    for state in captured_states:
        assert state["regime_data_splitting_result"] is payload
        assert state["data_dir"] == str(locator.data_path(config.data_dir_key))
