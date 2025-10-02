import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import silhouette_score

ROOT_DIR = Path(__file__).resolve().parents[5]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def load_clustering_module():
    package_specs = [
        ("src", ROOT_DIR / "src"),
        ("src.training", ROOT_DIR / "src/training"),
        ("src.training.steps", ROOT_DIR / "src/training/steps"),
        ("src.training.steps.market_analysis", ROOT_DIR / "src/training/steps/market_analysis"),
        (
            "src.training.steps.market_analysis.components",
            ROOT_DIR / "src/training/steps/market_analysis/components",
        ),
    ]

    for name, path in package_specs:
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module

    module_name = "src.training.steps.market_analysis.components.nas_tas_clustering"
    module_path = ROOT_DIR / "src/training/steps/market_analysis/components/nas_tas_clustering.py"

    if module_name in sys.modules:
        del sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def clustering_component(monkeypatch):
    module = load_clustering_module()

    resources = types.SimpleNamespace(
        matrix_ops=None,
        vectorized_core=None,
        batch_processor=None,
        hardware_manager=None,
        m1_gpu_optimizer=None,
        m1_memory_optimizer=None,
        m1_cpu_optimizer=None,
    )

    class _DummySetup:
        def initialize(self):
            return resources

    monkeypatch.setattr(module, 'HardwareSetup', lambda: _DummySetup(), raising=False)
    component = module.NASTASClusteringComponent()
    return component, module


def _generate_synthetic_features(assignments: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(1234)
    signal_1 = assignments.astype(float) + rng.normal(scale=0.1, size=assignments.shape[0])
    signal_2 = np.sin(assignments / assignments.max()) + rng.normal(scale=0.05, size=assignments.shape[0])
    noise = rng.normal(scale=1.5, size=assignments.shape[0])
    return np.column_stack([signal_1, signal_2, noise])


def test_feature_pruning_and_pca_loadings(clustering_component):
    component, module = clustering_component

    assignments = np.repeat(np.arange(3), 40)
    features = _generate_synthetic_features(assignments)
    market_data = pd.DataFrame({'close': features[:, 0]})

    component.pipeline_state = {
        'nas_tas_regime_discovery_result': {
            'tas_assignments': assignments.tolist(),
            'nas_assignments': assignments.tolist(),
        }
    }
    component.features = features

    selected_features, feature_names, metadata = component._select_regime_features(
        features=features,
        market_data=market_data,
        target_n_features=2,
    )

    assert metadata['selection_performed'] is True
    assert metadata['method'].startswith('regime_scoring')
    assert 'feature_scores' in metadata and metadata['feature_scores']
    assert metadata['selected_n_features'] <= 2
    ranked_feature_names = [
        name for name, _ in sorted(metadata['feature_scores'].items(), key=lambda item: item[1], reverse=True)
    ]
    assert set(feature_names) == set(ranked_feature_names[: len(feature_names)])

    context = module.ClusteringContext(
        original_features=selected_features,
        market_data=market_data,
        original_feature_names=feature_names,
        feature_scores=component.feature_scores,
    )

    component._optimize_features(context)

    assert context.optimized_features is not None
    assert context.optimized_feature_names
    assert set(context.optimized_feature_names).issubset(set(feature_names))
    assert context.pca_loading_scores
    assert component.feature_names == context.optimized_feature_names
    assert component.feature_scores == context.feature_scores

    # PCA should retain the highest-loading signals
    loading_values = list(context.pca_loading_scores.values())
    assert all(value >= 0.0 for value in loading_values)

    raw_silhouette = silhouette_score(features, assignments)
    selected_silhouette = silhouette_score(selected_features, assignments)
    optimized_silhouette = silhouette_score(context.optimized_features, assignments)

    assert selected_silhouette >= raw_silhouette - 0.03
    assert optimized_silhouette >= selected_silhouette - 0.1
