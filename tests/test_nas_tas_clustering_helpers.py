import importlib
import importlib.util
import sys
import types
from pathlib import Path

import asyncio
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src  # noqa: F401  # Ensure base package is registered
import src.training  # noqa: F401


def _ensure_stub_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = module


_ensure_stub_package("src.training.steps", ROOT / "src" / "training" / "steps")
_ensure_stub_package(
    "src.training.steps.market_analysis",
    ROOT / "src" / "training" / "steps" / "market_analysis",
)
_ensure_stub_package(
    "src.training.steps.market_analysis.components",
    ROOT / "src" / "training" / "steps" / "market_analysis" / "components",
)
_ensure_stub_package(
    "src.training.steps.market_analysis.regime_analysis",
    ROOT / "src" / "training" / "steps" / "market_analysis" / "regime_analysis",
)

sys.modules["src.training"].steps = sys.modules["src.training.steps"]  # type: ignore[attr-defined]
sys.modules["src.training.steps"].market_analysis = sys.modules[
    "src.training.steps.market_analysis"
]  # type: ignore[attr-defined]
sys.modules["src.training.steps.market_analysis"].components = sys.modules[
    "src.training.steps.market_analysis.components"
]  # type: ignore[attr-defined]
sys.modules["src.training.steps.market_analysis"].regime_analysis = sys.modules[
    "src.training.steps.market_analysis.regime_analysis"
]  # type: ignore[attr-defined]

PIPELINE_MODULE_PATH = (
    ROOT
    / "src"
    / "training"
    / "steps"
    / "market_analysis"
    / "regime_analysis"
    / "nas_tas_clustering.py"
)
MODULE_PATH = ROOT / "src" / "training" / "steps" / "market_analysis" / "components" / "nas_tas_clustering.py"

pipeline_module_name = (
    "src.training.steps.market_analysis.regime_analysis.nas_tas_clustering"
)
pipeline_spec = importlib.util.spec_from_file_location(
    pipeline_module_name, PIPELINE_MODULE_PATH
)
pipeline_module = importlib.util.module_from_spec(pipeline_spec)
assert pipeline_spec is not None and pipeline_spec.loader is not None
sys.modules[pipeline_module_name] = pipeline_module
pipeline_spec.loader.exec_module(pipeline_module)  # type: ignore[union-attr]
regime_package = sys.modules["src.training.steps.market_analysis.regime_analysis"]
regime_package.nas_tas_clustering = pipeline_module  # type: ignore[attr-defined]
regime_package.NASTASClusteringPipeline = pipeline_module.NASTASClusteringPipeline  # type: ignore[attr-defined]
module_name = "src.training.steps.market_analysis.components.nas_tas_clustering"
spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
nas_module = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
sys.modules[module_name] = nas_module
spec.loader.exec_module(nas_module)  # type: ignore[union-attr]
sys.modules["src.training.steps.market_analysis.components"].nas_tas_clustering = nas_module  # type: ignore[attr-defined]

if not hasattr(nas_module, "get_logger"):
    class _DummyLogger:
        def __getattr__(self, _: str):
            return lambda *args, **kwargs: None

    nas_module.get_logger = lambda name: _DummyLogger()  # type: ignore[attr-defined]

NASTASClusteringComponent = nas_module.NASTASClusteringComponent
from src.training.steps.market_analysis.regime_analysis.nas_tas_clustering import (
    NASTASClusteringPipeline,
)


def _build_component():
    return NASTASClusteringComponent()


def _build_pipeline():
    component = _build_component()
    return component, NASTASClusteringPipeline(component)


def test_extract_regime_counts_prefers_max_and_bounds():
    component, pipeline = _build_pipeline()
    pipeline_state = {
        "nas_tas_regime_discovery_result": {
            "tas_regime_count": 6,
            "nas_regime_count": 18,
        }
    }

    n_regimes = pipeline.extract_regime_counts(pipeline_state)

    assert n_regimes == 15  # bounded to maximum allowed value


def test_extract_regime_counts_defaults_when_missing():
    _, pipeline = _build_pipeline()

    n_regimes = pipeline.extract_regime_counts({})

    assert n_regimes == 8


def test_prepare_features_uses_shared_utility(monkeypatch):
    component, pipeline = _build_pipeline()
    market_data = pd.DataFrame({"close": [1, 2, 3]})

    prepared = pd.DataFrame({"feature": [0.1, 0.2, 0.3]})

    def fake_prepare_market_features(data, feature_config, verbose=True):
        assert data.equals(market_data)
        assert feature_config == component.feature_config
        assert verbose is True
        return prepared

    monkeypatch.setattr(
        "src.training.steps.market_analysis.regime_analysis.nas_tas_clustering.prepare_market_features",
        fake_prepare_market_features,
    )

    features = pipeline.prepare_features(market_data)

    pd.testing.assert_frame_equal(features, prepared)
    pd.testing.assert_frame_equal(component.features, prepared)


def test_prepare_features_raises_when_none(monkeypatch):
    _, pipeline = _build_pipeline()
    market_data = pd.DataFrame({"close": [1, 2, 3]})

    def fake_prepare_market_features(data, feature_config, verbose=True):
        return None

    monkeypatch.setattr(
        "src.training.steps.market_analysis.regime_analysis.nas_tas_clustering.prepare_market_features",
        fake_prepare_market_features,
    )

    with pytest.raises(ValueError):
        pipeline.prepare_features(market_data)


def test_build_artifacts_delegates_to_consolidated(monkeypatch):
    component, pipeline = _build_pipeline()
    clustering_result = {"cluster_assignments": [0, 1], "n_clusters": 2}
    cluster_characteristics = {"0": {}}
    clustering_metrics = {"stability": 0.9}
    market_data = pd.DataFrame({"close": [1, 2]})

    expected_artifacts = {"nas_tas_clustering_result": {}}

    def fake_create(self, result, characteristics, metrics, data):
        assert result is clustering_result
        assert characteristics is cluster_characteristics
        assert metrics is clustering_metrics
        assert data.equals(market_data)
        return expected_artifacts

    monkeypatch.setattr(
        NASTASClusteringComponent,
        "_create_consolidated_artifacts",
        fake_create,
    )

    artifacts = pipeline.build_artifacts(
        clustering_result,
        cluster_characteristics,
        clustering_metrics,
        market_data,
    )

    assert artifacts is expected_artifacts


def test_execute_runs_pipeline_integration(monkeypatch):
    component = _build_component()

    class FakePipeline:
        def __init__(self):
            self.calls = []
            self.clustering_result = {
                "n_clusters": 3,
                "cluster_assignments": [0, 1, 1],
            }

        def extract_regime_counts(self, pipeline_state):
            self.calls.append("extract_regime_counts")
            assert pipeline_state == {"state": True}
            return 5

        def validate_configuration(self):
            self.calls.append("validate_configuration")

        def prepare_features(self, market_data):
            self.calls.append("prepare_features")
            assert isinstance(market_data, pd.DataFrame)
            component.features = "features"
            return "features"

        async def perform_clustering(self, features, market_data):
            self.calls.append("perform_clustering")
            assert features == "features"
            assert isinstance(market_data, pd.DataFrame)
            return self.clustering_result

        def generate_cluster_characteristics(self, market_data, clustering_result):
            self.calls.append("generate_cluster_characteristics")
            assert clustering_result is self.clustering_result
            return {"characteristics": True}

        def calculate_clustering_metrics(self, clustering_result, cluster_characteristics):
            self.calls.append("calculate_clustering_metrics")
            assert cluster_characteristics == {"characteristics": True}
            return {"metrics": True}

        def build_artifacts(
            self,
            clustering_result,
            cluster_characteristics,
            clustering_metrics,
            market_data,
        ):
            self.calls.append("build_artifacts")
            assert clustering_metrics == {"metrics": True}
            return {"nas_tas_clustering_result": {"n_clusters": clustering_result["n_clusters"]}}

    fake_pipeline = FakePipeline()

    monkeypatch.setattr(component, "_create_pipeline", lambda: fake_pipeline)
    monkeypatch.setattr(
        component,
        "_create_clustering_config_using_shared_utils",
        lambda: {"config": True},
    )

    market_data = pd.DataFrame({"close": [1, 2, 3]})

    result = asyncio.run(component.execute(market_data, {"state": True}))

    assert result.success is True
    assert result.artifacts == {"nas_tas_clustering_result": {"n_clusters": 3}}
    assert component.config.n_regimes == 5
    assert fake_pipeline.calls == [
        "extract_regime_counts",
        "validate_configuration",
        "prepare_features",
        "perform_clustering",
        "generate_cluster_characteristics",
        "calculate_clustering_metrics",
        "build_artifacts",
    ]
