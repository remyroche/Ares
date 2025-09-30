import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[5]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.training.steps.market_analysis.shared_utils.calibration_registry import (
    get_quality_thresholds,
    reset_quality_calibration,
    update_quality_calibration,
)
from src.training.steps.market_analysis.regime_analysis import metrics as metrics_module


def load_clustering_module():
    package_specs = [
        ("src", ROOT_DIR / "src"),
        ("src.training", ROOT_DIR / "src/training"),
        ("src.training.steps", ROOT_DIR / "src/training/steps"),
        ("src.training.steps.market_analysis", ROOT_DIR / "src/training/steps/market_analysis"),
        ("src.training.steps.market_analysis.components", ROOT_DIR / "src/training/steps/market_analysis/components"),
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


@pytest.fixture(autouse=True)
def _reset_calibration_state():
    reset_quality_calibration()
    yield
    reset_quality_calibration()


@pytest.fixture
def fresh_component_module():
    return load_clustering_module()


def test_calibration_updates_thresholds(monkeypatch, fresh_component_module):
    module = fresh_component_module
    dummy_resources = types.SimpleNamespace(
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
            return dummy_resources

    monkeypatch.setattr(module, 'HardwareSetup', lambda: _DummySetup(), raising=False)
    monkeypatch.setattr(module, 'get_m1_gpu_manager', lambda: None, raising=False)
    monkeypatch.setattr(module, 'get_m1_memory_optimizer', lambda: None, raising=False)
    monkeypatch.setattr(module, 'get_m1_cpu_optimizer', lambda: None, raising=False)
    monkeypatch.setattr(module, 'tprint_structured', lambda *_, **__: None, raising=False)

    component = module.NASTASClusteringComponent()
    component._initialize_execution_metadata()

    baseline_thresholds = component._get_calibrated_quality_thresholds()
    assert baseline_thresholds['min_regime_persistence'] == pytest.approx(0.7)
    assert baseline_thresholds['max_feature_noise_ratio'] == pytest.approx(0.3)
    assert baseline_thresholds['min_temporal_stability'] == pytest.approx(0.6)

    features = np.array(
        [
            [1.0, 0.9],
            [1.1, 1.0],
            [0.95, 1.05],
            [1.05, 0.95],
        ]
    )
    assignments = np.array([0, 0, 1, 1])

    context = module.ClusteringContext(
        original_features=features,
        market_data=pd.DataFrame({'close': [1.0, 1.1, 1.2, 1.3]}),
        optimized_features=features,
        raw_assignments=assignments,
        smoothed_assignments=assignments,
        fusion_metadata={},
        optimization_metrics={'final_score': 0.6},
    )

    final_quality = {
        'silhouette_score': 0.65,
        'davies_bouldin_score': 0.8,
        'calinski_harabasz_score': 120.0,
        'intra_cluster_dispersion': 0.5,
        'inter_cluster_dispersion': 1.1,
        'cluster_compactness': 0.55,
    }

    monkeypatch.setattr(component, '_calculate_feature_regime_persistence', lambda *_: 0.85)
    monkeypatch.setattr(component, '_calculate_feature_noise_ratio', lambda *_: 0.2)
    monkeypatch.setattr(component, '_calculate_feature_temporal_stability', lambda *_: 0.75)
    monkeypatch.setattr(component, '_calculate_stability_score', lambda *_: 0.65)

    component._calibrate_quality_thresholds(context, final_quality)

    first_thresholds = component._get_calibrated_quality_thresholds()
    assert first_thresholds['min_regime_persistence'] == pytest.approx(0.85)
    assert first_thresholds['max_feature_noise_ratio'] == pytest.approx(0.2)
    assert first_thresholds['min_temporal_stability'] == pytest.approx(0.75)

    calibration_meta = component.execution_metadata['quality_calibration']
    assert calibration_meta['metric_thresholds']['silhouette']['excellent'] == pytest.approx(0.65)

    monkeypatch.setattr(component, '_calculate_feature_regime_persistence', lambda *_: 0.6)
    monkeypatch.setattr(component, '_calculate_feature_noise_ratio', lambda *_: 0.4)
    monkeypatch.setattr(component, '_calculate_feature_temporal_stability', lambda *_: 0.55)
    monkeypatch.setattr(component, '_calculate_stability_score', lambda *_: 0.45)

    context.optimization_metrics['final_score'] = 0.4
    second_quality = final_quality | {'silhouette_score': 0.45, 'davies_bouldin_score': 1.2}

    component._calibrate_quality_thresholds(context, second_quality)

    second_thresholds = component._get_calibrated_quality_thresholds()
    assert second_thresholds['min_regime_persistence'] == pytest.approx(0.725, rel=1e-3)
    assert second_thresholds['max_feature_noise_ratio'] == pytest.approx(0.4)
    assert second_thresholds['min_temporal_stability'] == pytest.approx(0.65)

    updated_meta = component.execution_metadata['quality_calibration']
    assert updated_meta['metric_thresholds']['silhouette']['excellent'] == pytest.approx(0.63, rel=1e-2)


def test_metric_interpreters_respect_calibration():
    # Defaults apply when history is empty
    reset_quality_calibration()
    assert metrics_module.interpret_silhouette(0.72) == "Excellent clustering"

    # Update calibration to enforce stricter thresholds
    update_quality_calibration(
        {
            'metric_thresholds': {
                'silhouette': {'excellent': 0.9, 'good': 0.7, 'fair': 0.5},
            }
        }
    )

    assert metrics_module.interpret_silhouette(0.91) == "Excellent clustering"
    assert metrics_module.interpret_silhouette(0.72) == "Good clustering"
    assert metrics_module.interpret_silhouette(0.52) == "Fair clustering"
    assert metrics_module.interpret_silhouette(0.3) == "Poor clustering"

    # Ensure other metrics still fall back correctly when calibration omitted
    reset_quality_calibration()
    thresholds = get_quality_thresholds()
    assert thresholds['min_regime_persistence'] == pytest.approx(0.7)
