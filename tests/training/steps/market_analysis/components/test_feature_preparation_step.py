import asyncio
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
import types
from types import SimpleNamespace

if 'torch' not in sys.modules:  # pragma: no cover - dependency stub for tests
    torch_stub = types.ModuleType('torch')
    torch_nn_stub = types.ModuleType('torch.nn')
    torch_nn_stub.Module = object
    torch_stub.nn = torch_nn_stub
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_optim_stub = types.ModuleType('torch.optim')
    torch_stub.optim = torch_optim_stub
    sys.modules['torch'] = torch_stub
    sys.modules['torch.nn'] = torch_nn_stub
    sys.modules['torch.optim'] = torch_optim_stub


def _load_feature_preparation_module():
    """Load the feature preparation step without importing heavy dependencies."""

    root_dir = Path(__file__).resolve().parents[5]
    package_specs = [
        ("src", root_dir / "src"),
        ("src.training", root_dir / "src/training"),
        ("src.training.steps", root_dir / "src/training/steps"),
        ("src.training.steps.market_analysis", root_dir / "src/training/steps/market_analysis"),
        ("src.training.steps.market_analysis.clusters", root_dir / "src/training/steps/market_analysis/clusters"),
    ]

    for name, path in package_specs:
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module

    module_name = "src.training.steps.market_analysis.clusters.step1_feature_preparation"
    module_path = root_dir / "src/training/steps/market_analysis/clusters/step1_feature_preparation.py"

    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_feature_module = _load_feature_preparation_module()
FeaturePreparationStep = _feature_module.FeaturePreparationStep
ClusteringContext = _feature_module.ClusteringContext


def test_feature_preparation_injects_shared_results(monkeypatch):
    """Ensure shared utility outputs populate the clustering context."""

    step = FeaturePreparationStep(verbose=False)

    feature_array = np.array([[0.1, 1.2], [0.3, 0.8], [0.5, 0.6]], dtype=float)
    feature_names = ['alpha', 'beta']
    feature_df = pd.DataFrame(feature_array, columns=feature_names)
    stage_metadata = {
        'operations': [
            {'type': 'correlation_filter', 'removed_columns': ['legacy_feature']}
        ]
    }
    metadata = {
        'stage_metadata': stage_metadata,
        'feature_columns': feature_names,
        'summary': {'total_features': len(feature_names)},
    }

    shared_result = SimpleNamespace(
        features_array=feature_array,
        features_df=feature_df,
        summary={'total_features': len(feature_names)},
        metadata=metadata,
    )

    async def fake_prepare(self, market_data, config):  # pragma: no cover - monkeypatched
        return shared_result

    optimize_called = {'value': False}

    async def fake_optimize(self, ctx, config):  # pragma: no cover - monkeypatched
        optimize_called['value'] = True
        # Ensure the shared result has already populated the context
        assert np.array_equal(ctx.original_features, feature_array)
        assert ctx.original_feature_names == feature_names
        assert ctx.pre_pca_feature_names == feature_names
        return ctx

    monkeypatch.setattr(
        FeaturePreparationStep,
        '_prepare_features_using_shared_utils',
        fake_prepare,
    )
    monkeypatch.setattr(
        FeaturePreparationStep,
        '_optimize_features',
        fake_optimize,
    )

    market_data = pd.DataFrame(
        {
            'open': [1, 2, 3],
            'high': [2, 3, 4],
            'low': [0.5, 1.5, 2.5],
            'close': [1.5, 2.5, 3.5],
            'volume': [100, 110, 120],
        }
    )

    context = ClusteringContext(
        original_features=np.zeros_like(feature_array),
        market_data=market_data,
        original_feature_names=None,
        feature_scores=None,
    )

    config = SimpleNamespace(use_cv_enhancement=False)

    updated_context = asyncio.run(step.execute(context, config))

    assert optimize_called['value'] is True
    assert np.array_equal(updated_context.original_features, feature_array)
    assert updated_context.original_feature_names == feature_names
    assert updated_context.pre_pca_feature_names == feature_names
    assert updated_context.pre_pca_feature_count == len(feature_names)
    assert updated_context.feature_scores == {}
    assert updated_context.dropped_feature_names == ['legacy_feature']
    assert updated_context.summary['feature_preparation']['metadata'] == metadata
