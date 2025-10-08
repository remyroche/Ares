import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest


class _StubComponentFactory:
    availability: Dict[str, bool] = {}
    created: List[Tuple[str, Any]] = []

    @classmethod
    def reset(cls) -> None:
        cls.availability = {}
        cls.created = []

    @classmethod
    def is_component_available(cls, component_name: str) -> bool:
        return cls.availability.get(component_name, False)

    @classmethod
    def create_component(cls, component_name: str, config: Any = None) -> Any:
        cls.created.append((component_name, config))
        if not cls.is_component_available(component_name):
            raise ValueError(f"{component_name} unavailable")
        return f"{component_name}_component"


@pytest.fixture
def orchestrator_module(monkeypatch):
    original_labeler = sys.modules.get(
        "src.training.steps.pre_training.multi_horizon_profit_labeler"
    )
    original_feature_opt = sys.modules.get(
        "src.training.steps.pre_training.feature_lookback_optimization"
    )
    original_components = sys.modules.get(
        "src.training.steps.pre_training.components"
    )

    stub_labeler_module = types.ModuleType(
        "src.training.steps.pre_training.multi_horizon_profit_labeler"
    )

    class _StubLabeler:
        def __init__(self, *args, **kwargs):
            pass

    class _StubConfig:
        def __init__(self, *args, **kwargs):
            pass

    stub_labeler_module.MultiHorizonProfitLabeler = _StubLabeler
    stub_labeler_module.MultiHorizonConfig = _StubConfig
    sys.modules[
        "src.training.steps.pre_training.multi_horizon_profit_labeler"
    ] = stub_labeler_module

    stub_feature_module = types.ModuleType(
        "src.training.steps.pre_training.feature_lookback_optimization"
    )

    class _StubFeatureComponent:
        def __init__(self, *args, **kwargs):
            pass

    stub_feature_module.FeatureLookbackOptimizationComponent = _StubFeatureComponent
    sys.modules[
        "src.training.steps.pre_training.feature_lookback_optimization"
    ] = stub_feature_module

    stub_components_module = types.ModuleType(
        "src.training.steps.pre_training.components"
    )

    class _StubComponentConfig:
        def __init__(self, custom_params: Dict[str, Any] | None = None):
            self.custom_params = custom_params or {}

    stub_components_module.ComponentFactory = _StubComponentFactory
    stub_components_module.ComponentConfig = _StubComponentConfig
    sys.modules[
        "src.training.steps.pre_training.components"
    ] = stub_components_module

    module_name = "tests.tactician_pre_ml_orchestrator_under_test"
    source_path = Path(__file__).resolve().parents[4] / "src" / "training" / "steps" / "model_training" / "tactician_pre_ml_orchestrator.py"

    spec = importlib.util.spec_from_file_location(module_name, source_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)

    monkeypatch.setattr(module, "ComponentFactory", _StubComponentFactory)
    monkeypatch.setattr(module, "COMMON_OPS_AVAILABLE", False)
    monkeypatch.setattr(module, "PID_GENERATION_AVAILABLE", False)
    monkeypatch.setattr(module, "HORIZON_LABELING_AVAILABLE", False)
    monkeypatch.setattr(module, "FEATURE_SELECTION_AVAILABLE", False)
    monkeypatch.setattr(module, "FEATURE_OPTIMIZATION_AVAILABLE", True)

    yield module

    sys.modules.pop(module_name, None)
    if original_labeler is not None:
        sys.modules[
            "src.training.steps.pre_training.multi_horizon_profit_labeler"
        ] = original_labeler
    else:
        sys.modules.pop(
            "src.training.steps.pre_training.multi_horizon_profit_labeler", None
        )
    if original_feature_opt is not None:
        sys.modules[
            "src.training.steps.pre_training.feature_lookback_optimization"
        ] = original_feature_opt
    else:
        sys.modules.pop(
            "src.training.steps.pre_training.feature_lookback_optimization", None
        )
    if original_components is not None:
        sys.modules[
            "src.training.steps.pre_training.components"
        ] = original_components
    else:
        sys.modules.pop(
            "src.training.steps.pre_training.components", None
        )


def test_feature_optimizer_initialized_via_component_factory(orchestrator_module, monkeypatch):
    _StubComponentFactory.reset()
    feature_key = orchestrator_module.TacticianPreMLOrchestrator.COMPONENT_FACTORY_KEYS[
        'feature_optimization'
    ]
    _StubComponentFactory.availability = {feature_key: True}

    config = orchestrator_module.OrchestratorConfig(
        enable_feature_optimization=True,
        enable_pid_generation=False,
        enable_horizon_labeling=False,
        enable_feature_selection=False,
    )

    orchestrator = orchestrator_module.TacticianPreMLOrchestrator(config)

    assert orchestrator.feature_optimizer == f"{feature_key}_component"
    assert orchestrator.factory_component_status['feature_optimization'] is True

    assert _StubComponentFactory.created, "Factory should have been invoked"
    created_name, created_config = _StubComponentFactory.created[0]
    assert created_name == feature_key
    assert created_config.custom_params['component_alias'] == 'feature_optimization'
    assert (
        created_config.custom_params['factory_component_key']
        == feature_key
    )
    assert (
        created_config.custom_params['max_lookback_periods']
        == config.max_lookback_periods
    )


def test_factory_unavailable_logs_hint(orchestrator_module, monkeypatch):
    _StubComponentFactory.reset()
    warnings: List[str] = []
    monkeypatch.setattr(
        orchestrator_module,
        "tprint_warning",
        lambda message, *args, **kwargs: warnings.append(message),
    )

    config = orchestrator_module.OrchestratorConfig(
        enable_feature_optimization=True,
        enable_pid_generation=False,
        enable_horizon_labeling=False,
        enable_feature_selection=False,
    )

    orchestrator = orchestrator_module.TacticianPreMLOrchestrator(config)

    assert orchestrator.feature_optimizer is None
    assert orchestrator.factory_component_status['feature_optimization'] is False
    feature_key = (
        orchestrator_module.TacticianPreMLOrchestrator.COMPONENT_FACTORY_KEYS[
            'feature_optimization'
        ]
    )
    assert any(feature_key in message for message in warnings), (
        "Expected factory availability warning to reference the missing component"
    )
    assert any("Hint:" in message for message in warnings), (
        "Expected factory availability warning to provide remediation hint"
    )
