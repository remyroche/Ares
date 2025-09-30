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


def test_extract_regime_counts_defaults_when_missing():
    component = _build_component()

    n_regimes = component._extract_regime_counts({})

    assert n_regimes == 8


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


def test_metric_weight_learning_updates_state():
    component = _build_component()

    metric_outputs = {
        'composite': {
            'silhouette': 0.8,
            'davies_bouldin': 0.7,
            'calinski_harabasz': 0.6,
            'stability': 0.5,
            'consensus': 0.9,
        },
        'regime': {
            'economic_weight': 0.7,
            'volatility_regime_weight': 0.6,
            'volume_regime_weight': 0.5,
            'structural_trend_weight': 0.4,
        },
        'temporal': {
            'autocorrelation': 0.6,
            'inverse_variance': 0.7,
            'trend_consistency': 0.5,
            'regime_persistence': 0.4,
        },
    }

    learned = component._fit_metric_weights(metric_outputs, validation_metric=1.2)

    assert set(learned.keys()) == {'composite', 'regime', 'temporal'}
    assert sum(learned['composite'].values()) == pytest.approx(1.0)
    assert sum(learned['regime'].values()) == pytest.approx(1.0)
    assert sum(learned['temporal'].values()) == pytest.approx(1.0)
    assert component.learned_weights['regime'] == learned['regime']


def test_metric_weight_fallback_uses_historical_median():
    component = _build_component()

    component.metric_weight_history = [
        {
            'metrics': {},
            'validation_target': 1.0,
            'fitted_weights': {
                'regime': {
                    'economic_weight': 0.5,
                    'volatility_regime_weight': 0.2,
                    'volume_regime_weight': 0.2,
                    'structural_trend_weight': 0.1,
                }
            },
        },
        {
            'metrics': {},
            'validation_target': 0.8,
            'fitted_weights': {
                'regime': {
                    'economic_weight': 0.4,
                    'volatility_regime_weight': 0.3,
                    'volume_regime_weight': 0.2,
                    'structural_trend_weight': 0.1,
                }
            },
        },
    ]

    weights = component._get_weights(
        'regime',
        {
            'economic_weight': 0.0,
            'volatility_regime_weight': 0.0,
            'volume_regime_weight': 0.0,
            'structural_trend_weight': 0.0,
        },
    )

    assert sum(weights.values()) == pytest.approx(1.0)
    # Median between [0.5,0.4] etc. equals 0.45 for economic weight
    assert weights['economic_weight'] == pytest.approx(0.45, rel=1e-6)
