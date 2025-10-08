import sys
import types

import numpy as np
import pandas as pd
import pytest

components_stub = types.ModuleType("src.training.steps.pre_training.components")


class _StubFactory:
    @staticmethod
    def create_component(*args, **kwargs):  # pragma: no cover - replaced in tests
        raise NotImplementedError


class _StubComponentConfig:  # pragma: no cover - placeholder for type compatibility
    def __init__(self, *args, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


components_stub.ComponentFactory = _StubFactory
components_stub.ComponentConfig = _StubComponentConfig
sys.modules.setdefault("src.training.steps.pre_training.components", components_stub)

from src.training.steps.pre_training.sub_pipeline import (
    PreTrainingSubPipeline,
    SubPipelineConfig,
    SubPipelineStatus,
)
import src.training.steps.pre_training.sub_pipeline as sub_pipeline_module
from src.training.steps.pre_training.validation.data_contracts import (
    DataContractValidationError,
    validate_feature_artifact,
    validate_multi_horizon_labeling_result,
    validate_selection_artifact,
)


class _StubComponentResult:
    def __init__(self, artifacts: dict[str, object]):
        self.success = True
        self.artifacts = artifacts
        self.metadata: dict[str, object] = {}
        self.metrics: dict[str, float] = {}
        self.warnings: list[str] = []
        self.error = None
        self.execution_time = 0.0
        self.output_files: list[str] = []
        self.error_message = None


def _initialize_pipeline_state(pipeline: PreTrainingSubPipeline) -> None:
    pipeline._current_pipeline_state = {}
    pipeline._metrics_sink = None
    pipeline._run_metadata = {}
    pipeline._data_locator = None
    pipeline._seeded_rngs = None
    pipeline._active_seed = None
    sub_pipeline_module.tprint_error = lambda *args, **kwargs: None


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_pre_training_pipeline_rejects_invalid_multi_horizon_artifact(monkeypatch):
    pipeline = PreTrainingSubPipeline()
    _initialize_pipeline_state(pipeline)
    config = SubPipelineConfig()

    index = pd.date_range("2024-01-01", periods=4, freq="h")
    labeled = pd.DataFrame(
        {
            "immediate_opportunity": [1, 0, 1, 0],
            "short_term_opportunity": [0, 1, 0, 1],
            "leverage_adjusted_score": [0.1, -0.2, 0.3, -0.4],
        },
        index=index,
    )
    market = pd.DataFrame(
        {
            "open": np.linspace(100.0, 101.5, len(index)),
            "high": np.linspace(101.0, 102.5, len(index)),
            "low": np.linspace(99.5, 100.5, len(index)),
            "close": np.linspace(100.5, 101.0, len(index)),
            "volume": np.linspace(1_000.0, 1_300.0, len(index)),
        },
        index=index,
    )

    invalid_payload = {
        "labeled_data": labeled,
        "labels": labeled.copy(),
        "horizon_weights": {"t1": 1.0},
        "target_columns": ["immediate_opportunity"],
        "metadata": {"source": "test"},
        "smoothing_settings": {},
        "market_data": market,
    }

    with pytest.raises(DataContractValidationError) as excinfo:
        validate_multi_horizon_labeling_result(
            invalid_payload,
            context="sub_pipeline.multi_horizon_profit_labeler",
        )
    expected_message = str(excinfo.value)

    class _StubMultiHorizonComponent:
        def set_run_metadata(self, metadata):
            return None

        async def execute(self, training_input, pipeline_state):
            return _StubComponentResult(
                {"multi_horizon_labeling_result": invalid_payload}
            )

    monkeypatch.setattr(
        "src.training.steps.pre_training.sub_pipeline.ComponentFactory.create_component",
        lambda step, cfg: _StubMultiHorizonComponent(),
    )

    training_data = pd.DataFrame(
        {
            "open": np.linspace(100.0, 102.0, 120),
            "high": np.linspace(101.0, 103.0, 120),
            "low": np.linspace(99.0, 101.0, 120),
            "close": np.linspace(100.5, 102.5, 120),
            "volume": np.linspace(5_000.0, 6_000.0, 120),
        },
        index=pd.date_range("2024-01-01", periods=120, freq="min"),
    )

    monkeypatch.setattr(
        pipeline,
        "_prepare_component_pipeline_state",
        lambda cfg: {},
    )
    monkeypatch.setattr(
        pipeline,
        "_prepare_interactive_training_input",
        lambda state: {"data": training_data, "targets": {}},
    )

    result = await pipeline._execute_multi_horizon_profit_labeler(config, {})

    assert result.status is SubPipelineStatus.FAILED
    assert result.error_message == expected_message
    assert result.error_code == "PRETRAIN_MH_LABEL_FAILURE_CONTRACT"
    assert result.failure is not None
    assert result.failure.context["contract_context"] == "sub_pipeline.multi_horizon_profit_labeler"
    assert result.failure.context["contract_issues"] == excinfo.value.errors


@pytest.mark.anyio
async def test_pre_training_pipeline_rejects_invalid_interactive_artifact(monkeypatch):
    pipeline = PreTrainingSubPipeline()
    _initialize_pipeline_state(pipeline)
    config = SubPipelineConfig()

    invalid_payload = {
        "features": "not-a-dataframe",
        "feature_names": ["f1"],
        "selected_features": ["f1"],
        "interaction_features": None,
        "cross_timeframe_features": None,
        "execution_time": 1.23,
        "memory_usage_mb": 12.5,
    }

    with pytest.raises(DataContractValidationError) as excinfo:
        validate_feature_artifact(
            invalid_payload,
            context="sub_pipeline.interactive_feature_generation",
        )
    expected_message = str(excinfo.value)

    class _StubInteractiveComponent:
        def set_run_metadata(self, metadata):
            return None

        async def execute(self, _data, _state):
            return _StubComponentResult(
                {"interactive_feature_generation_result": invalid_payload}
            )

    module_path = (
        "src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation."
        "interactive_feature_generation_component"
    )
    stub_module = types.ModuleType("interactive_feature_generation_component")

    class _StubConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    stub_module.InteractiveFeatureGenerationConfig = _StubConfig
    stub_module.create_interactive_feature_generation_component = lambda cfg: _StubInteractiveComponent()
    monkeypatch.setitem(sys.modules, module_path, stub_module)

    monkeypatch.setattr(
        pipeline,
        "_prepare_component_pipeline_state",
        lambda cfg: {},
    )

    result = await pipeline._execute_interactive_feature_generation(config, {})

    assert result.status is SubPipelineStatus.FAILED
    assert result.error_message == expected_message
    assert result.error_code == "PRETRAIN_INTERACTIVE_GEN_FAILURE_CONTRACT"
    assert result.failure is not None
    assert result.failure.context["contract_context"] == "sub_pipeline.interactive_feature_generation"
    assert result.failure.context["contract_issues"] == excinfo.value.errors


@pytest.mark.anyio
async def test_pre_training_pipeline_rejects_invalid_final_selection_artifact(monkeypatch):
    pipeline = PreTrainingSubPipeline()
    _initialize_pipeline_state(pipeline)
    config = SubPipelineConfig()

    invalid_payload = {
        "final_features": ["f1", 2],
        "stage_1_features": ["f1"],
        "stage_2_features": ["f2"],
        "stage_3_features": [],
        "feature_counts": {"initial": 2, "final": 1},
        "stage_scores": {"final": {"score": 0.8}},
    }

    with pytest.raises(DataContractValidationError) as excinfo:
        validate_selection_artifact(
            invalid_payload,
            context="sub_pipeline.final_feature_selection",
        )
    expected_message = str(excinfo.value)

    class _StubFinalSelectionComponent:
        def set_run_metadata(self, metadata):
            return None

        async def execute(self, _data, _state):
            return _StubComponentResult(
                {"final_feature_selection_result": invalid_payload}
            )

    monkeypatch.setattr(
        "src.training.steps.pre_training.sub_pipeline.ComponentFactory.create_component",
        lambda step, cfg: _StubFinalSelectionComponent(),
    )
    monkeypatch.setattr(
        pipeline,
        "_prepare_component_pipeline_state",
        lambda cfg: {},
    )

    result = await pipeline._execute_final_feature_selection(config, {})

    assert result.status is SubPipelineStatus.FAILED
    assert result.error_message == expected_message
    assert result.error_code == "PRETRAIN_FINAL_SELECTION_FAILURE_CONTRACT"
    assert result.failure is not None
    assert result.failure.context["contract_context"] == "sub_pipeline.final_feature_selection"
    assert result.failure.context["contract_issues"] == excinfo.value.errors


def test_prepare_interactive_training_input_requires_multi_horizon_result():
    pipeline = PreTrainingSubPipeline()
    _initialize_pipeline_state(pipeline)

    with pytest.raises(ValueError, match="Multi-horizon labeling result is required"):
        pipeline._prepare_interactive_training_input({"unexpected_key": "value"})
