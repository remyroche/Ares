import sys
import types

import numpy as np
import pandas as pd


def _ensure_pydantic_stubs():
    if 'pydantic' not in sys.modules:
        pydantic_stub = types.ModuleType('pydantic')

        def field_stub(*args, **kwargs):  # type: ignore[override]
            return kwargs.get('default')

        pydantic_stub.Field = field_stub  # type: ignore[attr-defined]
        sys.modules['pydantic'] = pydantic_stub

    if 'pydantic_settings' not in sys.modules:
        settings_stub = types.ModuleType('pydantic_settings')

        class BaseSettings:  # pylint: disable=too-few-public-methods
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        settings_stub.BaseSettings = BaseSettings  # type: ignore[attr-defined]
        sys.modules['pydantic_settings'] = settings_stub


_ensure_pydantic_stubs()

from src.training.steps.market_analysis.components.regime_models_training import (
    RegimeModelsTrainingComponent,
)


def _build_pipeline_state(features: np.ndarray, feature_names):
    return {
        'artifacts': {
            'nas_tas_clustering_result': {
                'original_features': features,
                'feature_names': feature_names,
            }
        }
    }


def test_prepare_training_data_applies_feature_selector():
    component = RegimeModelsTrainingComponent()
    rng = np.random.default_rng(42)
    samples, feature_count = 60, 12
    features = rng.normal(size=(samples, feature_count))
    feature_names = [f"feature_{idx}" for idx in range(feature_count)]
    data = pd.DataFrame(features, columns=feature_names)
    labels = rng.integers(0, 3, size=samples)

    X_selected, y, selection_info = component._prepare_training_data(
        data,
        labels,
        _build_pipeline_state(features.copy(), feature_names),
    )

    assert X_selected.shape[0] == samples
    assert y.shape[0] == samples
    assert selection_info['retained_feature_count'] == X_selected.shape[1]
    assert 0 < selection_info['retained_feature_count'] <= feature_count
    assert set(selection_info['selected_indices']).issubset(set(range(feature_count)))
    assert selection_info['selected_feature_names'] == [
        feature_names[idx] for idx in selection_info['selected_indices']
    ]
    assert selection_info['feature_importances']


def test_feature_mask_reused_for_prediction_path():
    component = RegimeModelsTrainingComponent()
    rng = np.random.default_rng(7)
    samples, feature_count = 40, 10
    features = rng.normal(size=(samples, feature_count))
    feature_names = [f"feature_{idx}" for idx in range(feature_count)]
    data = pd.DataFrame(features, columns=feature_names)
    labels = rng.integers(0, 2, size=samples)

    X_selected, _, selection_info = component._prepare_training_data(
        data,
        labels,
        _build_pipeline_state(features.copy(), feature_names),
    )

    reapplied = component._apply_feature_selection(features, selection_info)
    np.testing.assert_allclose(X_selected, reapplied)
    assert selection_info['selected_feature_names'] == [
        feature_names[idx] for idx in selection_info['selected_indices']
    ]
