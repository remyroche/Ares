import sys
import types
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict

import pandas as pd
import pytest

import src.training.steps.pre_training.multi_horizon_profit_labeler as mh_module


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
    data_dir: str = "historical_data"
    custom_params: dict = field(default_factory=dict)


class _TestComponentFactory:
    @staticmethod
    def create_component(name, config):  # pragma: no cover - overwritten in tests
        raise NotImplementedError("stub factory")


components_module = types.ModuleType("components_stub")
components_module.ComponentFactory = _TestComponentFactory
components_module.ComponentConfig = _TestComponentConfig

base_component_module = types.ModuleType("base_component_stub")
base_component_module.ComponentConfig = _TestComponentConfig
base_component_module.ComponentResult = object
base_component_module.BasePreTrainingComponent = object

sys.modules['src.training.steps.pre_training.components'] = components_module
sys.modules['src.training.steps.pre_training.components.base_component'] = base_component_module


from src.training.steps.models_training.tactician_pre_ml_orchestration import (
    TacticianPreMLConfig,
    TacticianPreMLOrchestrator,
)
from src.training.steps.pre_training.sub_pipeline import SubPipelineResult, SubPipelineStatus


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_tactician_orchestrator_passes_regime_split(monkeypatch):
    module = sys.modules["src.training.steps.models_training.tactician_pre_ml_orchestration"]

    class _StubPreTrainingPipeline:
        def __init__(self):
            self._current_pipeline_state = {}

    module.PreTrainingSubPipeline = _StubPreTrainingPipeline
    module.PRE_TRAINING_AVAILABLE = True

    @dataclass
    class _StubSubPipelineConfig:
        symbol: str
        exchange: str
        timeframe: str
        data_dir: str
        parallel_processing: bool = True
        custom_params: dict = field(default_factory=dict)
        mode: str = "full"
        start_date: str | None = None
        end_date: str | None = None
        force_rerun: bool = False
        max_workers: int = 4
        validation_enabled: bool = True
        monitoring_enabled: bool = True
        fast_mode: bool = False
        skip_next_pipeline: bool = False

    module.SubPipelineConfig = _StubSubPipelineConfig

    monkeypatch.setattr(
        "src.training.steps.models_training.tactician_pre_ml_orchestration.tprint_timer",
        lambda *args, **kwargs: 0.0,
    )

    orchestrator = TacticianPreMLOrchestrator(TacticianPreMLConfig())

    payload: Dict[str, Any] = {
        "regime_data": {"market_data": pd.DataFrame({"timestamp": pd.to_datetime([])})}
    }

    captured_configs: Dict[str, Any] = {}
    entry_bundle = {
        "artifacts": {"labeling_report": {"status": "stub"}},
        "quality_metrics": {"quality": 1.0},
        "label_column": "tactician_entry_target",
    }
    entry_called: Dict[str, bool] = {"called": False}

    def _stub_create_entry_label_artifacts(self, prepared_data, analyst_predictions, regime_assignments):
        entry_called["called"] = True
        return entry_bundle

    monkeypatch.setattr(
        module.TacticianPreMLOrchestrator,
        "_create_entry_label_artifacts",
        _stub_create_entry_label_artifacts,
        raising=False,
    )

    class _StubPipeline:
        def __init__(self):
            self._current_pipeline_state: Dict[str, Any] = {}

        async def _execute_multi_horizon_profit_labeler(self, config):
            captured_configs["multi"] = config
            return SubPipelineResult(
                sub_pipeline_name="multi_horizon_profit_labeler",
                status=SubPipelineStatus.COMPLETED,
                start_time=datetime.now(),
                end_time=datetime.now(),
                success=True,
                artifacts={"labels": pd.DataFrame()},
            )

        async def _execute_feature_lookback_optimization(self, config):
            captured_configs["lookback"] = config
            return SubPipelineResult(
                sub_pipeline_name="feature_lookback_optimization",
                status=SubPipelineStatus.COMPLETED,
                start_time=datetime.now(),
                end_time=datetime.now(),
                success=True,
                artifacts={"lookback_windows": []},
            )

        async def _execute_pid_based_feature_generation(self, config):
            captured_configs["pid"] = config
            return SubPipelineResult(
                sub_pipeline_name="pid_based_feature_generation",
                status=SubPipelineStatus.COMPLETED,
                start_time=datetime.now(),
                end_time=datetime.now(),
                success=True,
                artifacts={"total_features": 12},
            )

        async def _execute_final_feature_selection(self, config):
            captured_configs["selection"] = config
            return SubPipelineResult(
                sub_pipeline_name="final_feature_selection",
                status=SubPipelineStatus.COMPLETED,
                start_time=datetime.now(),
                end_time=datetime.now(),
                success=True,
                artifacts={
                    "final_features": pd.DataFrame({"timestamp": pd.to_datetime([])}),
                    "selected_features": ["feat_1"],
                },
            )

    orchestrator.pre_training_pipeline = _StubPipeline()

    training_data = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="15min"),
        "close": [1, 2, 3, 4, 5],
    })
    regime_assignments = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="15min"),
        "regime": [0, 1, 0, 1, 0],
    })
    analyst_predictions = pd.DataFrame(
        {
            "green_light": [1, 1, 0, 1, 0],
        },
        index=training_data["timestamp"],
    )

    result = await orchestrator.orchestrate(
        training_data=training_data,
        analyst_predictions=analyst_predictions,
        regime_assignments=regime_assignments,
        regime_data_splitting_result=payload,
    )

    assert result.success
    assert entry_called["called"]
    assert result.entry_labeling_result == entry_bundle["artifacts"]
    assert result.entry_label_quality_metrics == entry_bundle["quality_metrics"]
    assert orchestrator.pre_training_pipeline._current_pipeline_state["regime_data_splitting_result"] is payload
    assert captured_configs["multi"].custom_params["regime_data_splitting_result"] is payload
    assert captured_configs["lookback"].custom_params["regime_data_splitting_result"] is payload
    assert captured_configs["pid"].custom_params["regime_data_splitting_result"] is payload
    assert captured_configs["selection"].custom_params["regime_data_splitting_result"] is payload
    assert (
        captured_configs["multi"].custom_params["precomputed_labeling_result"]
        == entry_bundle["artifacts"]
    )
