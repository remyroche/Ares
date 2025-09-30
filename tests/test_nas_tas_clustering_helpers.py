import importlib
import importlib.util
import sys
import types
from pathlib import Path

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

sys.modules["src.training"].steps = sys.modules["src.training.steps"]  # type: ignore[attr-defined]
sys.modules["src.training.steps"].market_analysis = sys.modules[
    "src.training.steps.market_analysis"
]  # type: ignore[attr-defined]
sys.modules["src.training.steps.market_analysis"].components = sys.modules[
    "src.training.steps.market_analysis.components"
]  # type: ignore[attr-defined]

MODULE_PATH = ROOT / "src" / "training" / "steps" / "market_analysis" / "components" / "nas_tas_clustering.py"

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


def _build_component():
    if not hasattr(nas_module, "get_m1_gpu_manager"):
        nas_module.get_m1_gpu_manager = lambda: None  # type: ignore[attr-defined]
    if not hasattr(nas_module, "get_m1_memory_optimizer"):
        nas_module.get_m1_memory_optimizer = lambda: None  # type: ignore[attr-defined]
    if not hasattr(nas_module, "get_m1_cpu_optimizer"):
        nas_module.get_m1_cpu_optimizer = lambda: None  # type: ignore[attr-defined]
    if not hasattr(nas_module, "tprint_structured"):
        nas_module.tprint_structured = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    return NASTASClusteringComponent()


def test_extract_regime_counts_prefers_max_and_bounds():
    component = _build_component()
    pipeline_state = {
        "nas_tas_regime_discovery_result": {
            "tas_regime_count": 6,
            "nas_regime_count": 18,
        }
    }

    n_regimes = component._extract_regime_counts(pipeline_state)

    assert n_regimes == 15  # bounded to maximum allowed value
    assert component.config.regime_search_min == 5
    assert component.config.regime_search_max == 15
    assert component.config.n_regimes == 15


def test_extract_regime_counts_defaults_when_missing():
    component = _build_component()

    n_regimes = component._extract_regime_counts({})

    assert n_regimes == 8
    assert component.config.regime_search_min == 5
    assert component.config.regime_search_max == 15
    assert component.config.n_regimes == 8


def test_estimate_regime_range_uses_candidate_metrics():
    component = _build_component()
    pipeline_state = {
        "nas_tas_regime_discovery_result": {
            "regime_candidates": [
                {"n_regimes": 6, "metrics": {"silhouette": 0.25, "bic": 1200, "aic": 1300}},
                {"n_regimes": 10, "metrics": {"silhouette": 0.6, "bic": 900, "aic": 950}},
                {"n_regimes": 14, "metrics": {"silhouette": 0.58, "bic": 880, "aic": 920}},
            ]
        }
    }

    min_bound, max_bound, suggested = component._estimate_regime_range(pipeline_state)

    assert (min_bound, max_bound, suggested) == (10, 14, 14)


def test_extract_regime_counts_uses_dynamic_bounds():
    component = _build_component()
    pipeline_state = {
        "nas_tas_regime_discovery_result": {
            "regime_candidates": [
                {"n_regimes": 6, "metrics": {"silhouette": 0.25, "bic": 1200, "aic": 1300}},
                {"n_regimes": 10, "metrics": {"silhouette": 0.6, "bic": 900, "aic": 950}},
                {"n_regimes": 14, "metrics": {"silhouette": 0.58, "bic": 880, "aic": 920}},
            ],
            "tas_regime_count": 9,
            "nas_regime_count": 18,
        }
    }

    n_regimes = component._extract_regime_counts(pipeline_state)

    assert n_regimes == 14
    assert component.config.regime_search_min == 10
    assert component.config.regime_search_max == 14
    assert component.config.n_regimes == 14


def test_prepare_features_uses_shared_utility(monkeypatch):
    component = _build_component()
    market_data = pd.DataFrame({"close": [1, 2, 3]})

    prepared = pd.DataFrame({"feature": [0.1, 0.2, 0.3]})

    def fake_prepare_market_features(data, feature_config, verbose=True):
        assert data.equals(market_data)
        assert feature_config == component.feature_config
        assert verbose is True
        return prepared

    monkeypatch.setattr(
        "src.training.steps.market_analysis.components.nas_tas_clustering.prepare_market_features",
        fake_prepare_market_features,
    )

    features = component._prepare_features(market_data)

    pd.testing.assert_frame_equal(features, prepared)
    pd.testing.assert_frame_equal(component.features, prepared)


def test_prepare_features_raises_when_none(monkeypatch):
    component = _build_component()
    market_data = pd.DataFrame({"close": [1, 2, 3]})

    def fake_prepare_market_features(data, feature_config, verbose=True):
        return None

    monkeypatch.setattr(
        "src.training.steps.market_analysis.components.nas_tas_clustering.prepare_market_features",
        fake_prepare_market_features,
    )

    with pytest.raises(ValueError):
        component._prepare_features(market_data)


def test_build_artifacts_delegates_to_consolidated(monkeypatch):
    component = _build_component()
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

    artifacts = component._build_artifacts(
        clustering_result,
        cluster_characteristics,
        clustering_metrics,
        market_data,
    )

    assert artifacts is expected_artifacts


def test_fit_metric_weights_uses_regression_when_history_available():
    component = _build_component()

    component.metric_weight_history = [
        {
            "metrics": {"composite": {"silhouette": 1.0, "davies_bouldin": 0.0}},
            "validation_target": 1.0,
        },
        {
            "metrics": {"composite": {"silhouette": 0.0, "davies_bouldin": 1.0}},
            "validation_target": 2.0,
        },
    ]

    metric_outputs = {"composite": {"silhouette": 0.6, "davies_bouldin": 0.4}}
    learned = component._fit_metric_weights(metric_outputs, validation_metric=1.4)

    assert "composite" in learned
    assert learned["composite"] == pytest.approx(
        {"silhouette": 1 / 3, "davies_bouldin": 2 / 3}, rel=1e-6
    )
    assert component.learned_weights["composite"] == pytest.approx(
        {"silhouette": 1 / 3, "davies_bouldin": 2 / 3}, rel=1e-6
    )
    assert component.metric_weight_history[-1]["fitted_weights"]["composite"] == pytest.approx(
        {"silhouette": 1 / 3, "davies_bouldin": 2 / 3}, rel=1e-6
    )


def test_fit_metric_weights_falls_back_to_median_when_insufficient_history(monkeypatch):
    component = _build_component()
    component.metric_weight_history = [
        {
            "metrics": {"composite": {"silhouette": 0.5, "davies_bouldin": 0.5}},
            "validation_target": 1.0,
            "fitted_weights": {
                "composite": {"silhouette": 0.7, "davies_bouldin": 0.3}
            },
        }
    ]

    monkeypatch.setattr(component, "_estimate_validation_metric", lambda *_: None)

    learned = component._fit_metric_weights(
        {"composite": {"silhouette": 0.4, "davies_bouldin": 0.6}},
        validation_metric=None,
    )

    assert learned["composite"] == pytest.approx(
        {"silhouette": 0.7, "davies_bouldin": 0.3}, rel=1e-6
    )


def test_create_consolidated_artifacts_persists_learned_weights():
    component = _build_component()
    component.learned_weights = {
        "regime": {
            "economic": 0.2,
            "volatility": 0.3,
            "volume": 0.25,
            "structural_trend": 0.25,
        }
    }
    component.metric_weight_history = [
        {
            "timestamp": "2024-01-01T00:00:00",
            "metrics": {"regime": {"economic": 0.2}},
            "validation_target": 1.0,
            "fitted_weights": {
                "regime": {
                    "economic": 0.2,
                    "volatility": 0.3,
                    "volume": 0.25,
                    "structural_trend": 0.25,
                }
            },
        }
    ]

    clustering_result = {
        "cluster_assignments": [0, 1],
        "cluster_centers": [[0.0], [1.0]],
        "n_clusters": 2,
        "algorithm_used": "test",
        "success": True,
        "execution_time": 1.0,
    }
    cluster_characteristics = {}
    clustering_metrics = {}
    market_data = pd.DataFrame({"close": [1.0, 2.0]})

    artifacts = component._create_consolidated_artifacts(
        clustering_result, cluster_characteristics, clustering_metrics, market_data
    )

    metadata = artifacts["execution_metadata"]
    assert metadata["learned_metric_weights"]["regime"] == pytest.approx(
        {
            "economic": 0.2,
            "volatility": 0.3,
            "volume": 0.25,
            "structural_trend": 0.25,
        },
        rel=1e-6,
    )
    assert metadata["metric_weight_history"]


def test_clustering_config_uses_learned_regime_weights():
    component = _build_component()
    component.learned_weights["regime"] = {
        "economic": 0.1,
        "volatility": 0.2,
        "volume": 0.3,
        "structural_trend": 0.4,
    }

    config = component._create_clustering_config_using_shared_utils()

    assert config["economic_weight"] == pytest.approx(0.1)
    assert config["volatility_regime_weight"] == pytest.approx(0.2)
    assert config["volume_regime_weight"] == pytest.approx(0.3)
    assert config["structural_trend_weight"] == pytest.approx(0.4)
