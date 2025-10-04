import sys
import types

import numpy as np
import pandas as pd
import pytest

# Provide minimal torch stub to satisfy optional imports triggered by component loading
torch_stub = types.ModuleType("torch")
torch_nn_stub = types.ModuleType("torch.nn")
torch_optim_stub = types.ModuleType("torch.optim")
torch_nn_utils_stub = types.ModuleType("torch.nn.utils")
torch_nn_functional_stub = types.ModuleType("torch.nn.functional")
setattr(torch_stub, "nn", torch_nn_stub)
setattr(torch_stub, "optim", torch_optim_stub)
setattr(torch_nn_stub, "utils", torch_nn_utils_stub)
setattr(torch_nn_stub, "functional", torch_nn_functional_stub)
sys.modules.setdefault("torch", torch_stub)
sys.modules.setdefault("torch.nn", torch_nn_stub)
sys.modules.setdefault("torch.optim", torch_optim_stub)
sys.modules.setdefault("torch.nn.utils", torch_nn_utils_stub)
sys.modules.setdefault("torch.nn.functional", torch_nn_functional_stub)

# Provide minimal classes used by downstream imports
setattr(torch_nn_stub, "Module", type("Module", (), {}))
setattr(torch_stub, "Tensor", type("Tensor", (), {}))
setattr(torch_stub, "device", type("device", (), {}))
torch_nn_utils_prune_stub = types.ModuleType("torch.nn.utils.prune")
sys.modules.setdefault("torch.nn.utils.prune", torch_nn_utils_prune_stub)
setattr(torch_nn_utils_stub, "prune", torch_nn_utils_prune_stub)

# Install lightweight stubs for heavy component dependencies to avoid cascading imports
def _create_placeholder(name: str):
    return type(name, (), {})


def _install_component_stub(module_name: str, **attrs):
    module = types.ModuleType(module_name)
    for attr_name, attr_value in attrs.items():
        setattr(module, attr_name, attr_value)
    sys.modules.setdefault(module_name, module)
    return module


class _ArtifactManagerStub:
    def __init__(self, *args, **kwargs):
        pass


_install_component_stub(
    "src.training.steps.market_analysis.components.component_factory",
    ComponentFactory=_create_placeholder("ComponentFactory"),
)
_install_component_stub(
    "src.training.steps.market_analysis.components.sr_parameter_optimization",
    SRParameterOptimizationComponent=_create_placeholder("SRParameterOptimizationComponent"),
)
_install_component_stub(
    "src.training.steps.market_analysis.components.sr_detection",
    SRDetectionComponent=_create_placeholder("SRDetectionComponent"),
)
_install_component_stub(
    "src.training.steps.market_analysis.components.sr_clustering",
    SRClusteringComponent=_create_placeholder("SRClusteringComponent"),
)
_install_component_stub(
    "src.training.steps.market_analysis.components.nas_regime_discovery",
    NASRegimeDiscoveryComponent=_create_placeholder("NASRegimeDiscoveryComponent"),
)
_install_component_stub(
    "src.training.steps.market_analysis.components.tas_regime_discovery",
    TASRegimeDiscoveryComponent=_create_placeholder("TASRegimeDiscoveryComponent"),
)
_install_component_stub(
    "src.training.steps.market_analysis.components.nas_tas_regime_discovery",
    NASTASRegimeDiscoveryComponent=_create_placeholder("NASTASRegimeDiscoveryComponent"),
)
_install_component_stub(
    "src.training.steps.market_analysis.components.feature_lookback_optimization",
    FeatureLookbackOptimizationComponent=_create_placeholder("FeatureLookbackOptimizationComponent"),
)
_install_component_stub(
    "src.training.steps.market_analysis.components.cross_timeframe_analysis",
    CrossTimeframeAnalysisComponent=_create_placeholder("CrossTimeframeAnalysisComponent"),
)
_install_component_stub(
    "src.training.steps.market_analysis.components.regime_models_training",
    RegimeModelsTrainingComponent=_create_placeholder("RegimeModelsTrainingComponent"),
)
_install_component_stub(
    "src.training.steps.market_analysis.components.regime_ensemble_training",
    RegimeEnsembleTrainingComponent=_create_placeholder("RegimeEnsembleTrainingComponent"),
)
_install_component_stub(
    "src.training.steps.market_analysis.components.artifact_manager",
    ArtifactManager=_ArtifactManagerStub,
)

from src.training.steps.market_analysis.pid_based_feature_generation.pid_based_feature_generation_component import (
    PIDBasedFeatureGenerationComponent,
)
from src.training.steps.market_analysis.pid_based_feature_generation.optimized_lookback_integration import (
    IntegrationStatus,
    LookbackIntegrationResult,
)
from src.training.steps.market_analysis.pid_based_feature_generation.pid_based_feature_orchestrator import (
    GenerationStatus,
    OrchestratorResult,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_pid_orchestration_handles_nan_targets(monkeypatch):
    component = PIDBasedFeatureGenerationComponent()

    market_data = pd.DataFrame(
        {
            "feature_a": np.arange(5, dtype=float),
            "feature_b": np.linspace(10, 14, num=5),
        }
    )

    pipeline_state = {
        "multi_horizon_labeling_result": {
            "labeled_data": pd.DataFrame(
                {
                    "long_overall_opportunity": [0.1, np.nan, 0.2, np.nan, -0.1],
                    "short_overall_opportunity": [np.nan, -0.2, np.nan, 0.3, 0.0],
                }
            )
        },
        "feature_lookback_optimization_result": {},
    }

    async def fake_load_and_validate(_data):
        return market_data

    monkeypatch.setattr(component, "_load_and_validate_market_data", fake_load_and_validate)

    lookback_result = LookbackIntegrationResult(
        optimized_lookback_periods={name: 1 for name in market_data.columns},
        integration_status=IntegrationStatus.SUCCESS,
        features_optimized=len(market_data.columns),
        optimization_quality_score=1.0,
    )

    def fake_integrate(_opt_results, _columns):
        return lookback_result

    monkeypatch.setattr(
        component.lookback_integration,
        "integrate_optimized_lookback_periods",
        fake_integrate,
    )

    orchestrator_call_args = {}

    async def fake_orchestrate(data, feature_names, optimized_lookback_periods, target):
        orchestrator_call_args["data_length"] = len(data)
        orchestrator_call_args["feature_names"] = feature_names
        orchestrator_call_args["target"] = target

        result = OrchestratorResult()
        result.generation_status = GenerationStatus.COMPLETED
        result.total_features_generated = 0
        result.combined_feature_names = []
        result.overall_quality_score = 1.0
        result.interaction_result = None
        result.cross_timeframe_result = None
        result.polynomial_result = None  # type: ignore[attr-defined]
        return result

    monkeypatch.setattr(
        component.orchestrator,
        "orchestrate_feature_generation",
        fake_orchestrate,
    )

    async def fake_validate(_result):
        return {"is_valid": True, "quality_score": 1.0, "issues": []}

    monkeypatch.setattr(component, "_validate_generation_results", fake_validate)

    async def fake_create_artifacts(_orchestrator_result, _lookback_result, _validation_result, _market_data):
        return {"pid_based_feature_generation_result": {}}

    monkeypatch.setattr(component, "_create_comprehensive_artifacts", fake_create_artifacts)

    def fake_final_report(_artifacts, _validation_result, _orchestrator_result):
        return {}

    monkeypatch.setattr(component, "_generate_final_report", fake_final_report)

    result = await component.execute(market_data, pipeline_state)

    assert result.success is True
    assert orchestrator_call_args["data_length"] == len(market_data)
    assert orchestrator_call_args["feature_names"] == list(market_data.columns)

    target = orchestrator_call_args["target"]
    assert set(target.keys()) == {"long", "short"}
    assert len(target["long"]) == len(market_data)
    assert len(target["short"]) == len(market_data)

    # Original NaN values should be replaced to keep array length stable
    np.testing.assert_allclose(target["long"], [0.1, 0.0, 0.2, 0.0, -0.1])
    np.testing.assert_allclose(target["short"], [0.0, -0.2, 0.0, 0.3, 0.0])
