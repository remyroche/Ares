import sys
import types
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB


def _ensure_package_hierarchy(package_path: str) -> types.ModuleType:
    parts = package_path.split('.') if package_path else []
    parent = None

    for idx in range(1, len(parts) + 1):
        name = '.'.join(parts[:idx])
        module = sys.modules.get(name)
        if module is None:
            module = types.ModuleType(name)
            module.__path__ = []  # Mark as namespace package
            sys.modules[name] = module
        if parent is not None:
            setattr(parent, parts[idx - 1], module)
        parent = module

    if parent is None:
        parent = types.ModuleType('')

    return parent


def _register_stub_module(full_name: str, attributes: dict) -> None:
    package_name, _, module_basename = full_name.rpartition('.')
    package = _ensure_package_hierarchy(package_name)
    module = types.ModuleType(full_name)
    for attr_name, attr_value in attributes.items():
        setattr(module, attr_name, attr_value)
    sys.modules[full_name] = module
    if package_name:
        setattr(package, module_basename, module)


class _DummyDetector:
    def __init__(self, *args, **kwargs):
        pass

    def detect_regimes(self, market_data):
        length = len(market_data) if market_data is not None else 1
        probabilities = np.zeros((length, 1))
        probabilities[:, 0] = 1.0
        return types.SimpleNamespace(
            regime_predictions=np.zeros(length, dtype=int),
            regime_probabilities=probabilities,
        )


class _DummyConfig:
    def __init__(self, *args, **kwargs):
        pass


class _DummyNASConfig(_DummyConfig):
    @staticmethod
    def create_short_term_trading_config():
        return _DummyNASConfig()


def _ensure_heavy_dependency_stubs():
    # Regime detector stubs
    _register_stub_module(
        'src.training.steps.market_analysis.tas_regime.core.tas_regime_detector',
        {
            'TASRegimeDetector': _DummyDetector,
            'TASRegimeConfig': _DummyConfig,
        },
    )
    _register_stub_module(
        'src.training.steps.market_analysis.nas_regime.core.perfect_nas_regime_detector',
        {
            'PerfectNASRegimeDetector': _DummyDetector,
            'PerfectNASConfig': _DummyNASConfig,
        },
    )
    _register_stub_module(
        'src.training.steps.market_analysis.hybrid_nas_tas_regime.core.hybrid_regime_detector',
        {
            'HybridNASTASRegimeDetector': _DummyDetector,
            'HybridRegimeConfig': _DummyConfig,
        },
    )

    # ML common utilities stubs
    _register_stub_module(
        'src.utils.ml_common.common_operations',
        {'get_ml_common_operations': lambda: object()},
    )
    _register_stub_module(
        'src.utils.ml_common.validation',
        {'get_validation_framework': lambda: object()},
    )


_ensure_heavy_dependency_stubs()

ROOT = Path(__file__).resolve().parents[4]
MODEL_SELECTOR_PATH = ROOT / 'src' / 'training' / 'steps' / 'models_training' / 'nas_tas' / 'model_selector.py'

spec = importlib.util.spec_from_file_location('test_model_selector_module', MODEL_SELECTOR_PATH)
assert spec and spec.loader
model_selector_module = importlib.util.module_from_spec(spec)
sys.modules['test_model_selector_module'] = model_selector_module
spec.loader.exec_module(model_selector_module)

ModelSelector = model_selector_module.ModelSelector
ModelSelectionConfig = model_selector_module.ModelSelectionConfig
SelectionStrategy = model_selector_module.SelectionStrategy


def _build_market_data(rng: np.random.Generator, samples: int = 60) -> pd.DataFrame:
    features = rng.normal(size=(samples, 3))
    columns = ['f0', 'f1', 'f2']
    return pd.DataFrame(features, columns=columns)


def _train_base_models(X: pd.DataFrame, y: np.ndarray):
    rf = RandomForestClassifier(n_estimators=8, random_state=0)
    rf.fit(X, y)

    lr = LogisticRegression(max_iter=200, solver='lbfgs', random_state=0)
    lr.fit(X, y)

    gb = GradientBoostingClassifier(random_state=0)
    gb.fit(X, y)

    nb = GaussianNB()
    nb.fit(X, y)

    return rf, lr, gb, nb


def _register_selector_models(selector: ModelSelector, X: pd.DataFrame, y: np.ndarray):
    rf, lr, gb, nb = _train_base_models(X, y)

    regime_models = {
        0: {
            'random_forest': {
                'model': rf,
                'val_metrics': {'f1_score': 0.82, 'accuracy': 0.81, 'precision': 0.83, 'recall': 0.80},
            },
            'logistic_regression': {
                'model': lr,
                'val_metrics': {'f1_score': 0.74, 'accuracy': 0.73, 'precision': 0.75, 'recall': 0.72},
            },
        },
        1: {
            'gradient_boosting': {
                'model': gb,
                'val_metrics': {'f1_score': 0.88, 'accuracy': 0.87, 'precision': 0.89, 'recall': 0.86},
            },
            'naive_bayes': {
                'model': nb,
                'val_metrics': {'f1_score': 0.71, 'accuracy': 0.70, 'precision': 0.72, 'recall': 0.69},
            },
        },
    }

    selector.register_models(regime_models)

    return regime_models


def test_adaptive_selection_uses_registered_model_ids_for_history():
    rng = np.random.default_rng(42)
    market_data = _build_market_data(rng)
    labels = rng.integers(0, 2, size=len(market_data))

    config = ModelSelectionConfig(
        selection_strategy=SelectionStrategy.ADAPTIVE,
        enable_ensemble=False,
        enable_adaptive_selection=True,
        save_selection_history=False,
    )
    selector = ModelSelector(config)

    regime_models = _register_selector_models(selector, market_data, labels)

    # Update history for the models expected to be selected in each regime
    selector.update_model_performance('regime_0_random_forest', {
        'f1_score': 0.83,
        'accuracy': 0.82,
        'precision': 0.84,
        'recall': 0.81,
    })
    selector.update_model_performance('regime_1_gradient_boosting', {
        'f1_score': 0.89,
        'accuracy': 0.88,
        'precision': 0.90,
        'recall': 0.87,
    })

    result_regime_0 = selector.select_model(market_data, current_regime=0)
    assert result_regime_0.selected_model_type == 'random_forest'
    assert result_regime_0.historical_performance['f1_score'] == pytest.approx(0.83)
    assert result_regime_0.expected_performance['accuracy'] == pytest.approx(0.82)

    result_regime_1 = selector.select_model(market_data, current_regime=1)
    assert result_regime_1.selected_model_type == 'gradient_boosting'
    assert result_regime_1.historical_performance['precision'] == pytest.approx(0.90)

    # Adaptive weights should include both regimes' winning model types and stay normalized
    assert {'random_forest', 'gradient_boosting'}.issubset(selector.adaptation_weights.keys())
    assert sum(selector.adaptation_weights.values()) == pytest.approx(1.0)


def test_ensemble_selection_reuses_pretrained_identifier_and_history():
    rng = np.random.default_rng(7)
    market_data = _build_market_data(rng)
    labels = rng.integers(0, 2, size=len(market_data))

    config = ModelSelectionConfig(
        selection_strategy=SelectionStrategy.ENSEMBLE,
        enable_ensemble=True,
        enable_adaptive_selection=False,
        save_selection_history=False,
    )
    selector = ModelSelector(config)

    regime_models = _register_selector_models(selector, market_data, labels)

    base_model_ids = [info['model_id'] for info in selector.available_models[0].values()]
    rf = regime_models[0]['random_forest']['model']
    lr = regime_models[0]['logistic_regression']['model']

    pretrained_ensemble = VotingClassifier(
        estimators=[('random_forest', rf), ('logistic_regression', lr)],
        voting='soft',
        weights=[0.6, 0.4],
    )
    pretrained_ensemble.fit(market_data.values, labels)

    ensemble_name = 'regime0_soft'
    ensemble_id = f'ensemble_{ensemble_name}'

    selector.register_models(
        {},
        ensemble_models={
            ensemble_name: {
                'model': pretrained_ensemble,
                'val_metrics': {'f1_score': 0.86, 'accuracy': 0.85, 'precision': 0.87, 'recall': 0.84},
                'base_models': base_model_ids,
                'weights': {'random_forest': 0.6, 'logistic_regression': 0.4},
            }
        },
    )

    selector.update_model_performance(ensemble_id, {
        'f1_score': 0.87,
        'accuracy': 0.86,
        'precision': 0.88,
        'recall': 0.85,
    })

    result = selector.select_model(market_data, current_regime=0)

    assert result.selected_model_type == 'ensemble'
    assert result.selected_model is pretrained_ensemble
    assert result.historical_performance['f1_score'] == pytest.approx(0.87)
    assert result.expected_performance['recall'] == pytest.approx(0.85)
    assert result.ensemble_weights == {'random_forest': 0.6, 'logistic_regression': 0.4}

