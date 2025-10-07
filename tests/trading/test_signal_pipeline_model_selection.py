import asyncio
from datetime import datetime
from pathlib import Path
import importlib.util
import sys
import types

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import tests.trading.test_model_selector_service as selector_tests

config_pkg = sys.modules.setdefault('src.trading.config', types.ModuleType('src.trading.config'))
config_pkg.__path__ = [str(REPO_ROOT / 'src' / 'trading' / 'config')]

signal_pkg = sys.modules.setdefault('src.trading.signal_generation', types.ModuleType('src.trading.signal_generation'))
signal_pkg.__path__ = [str(REPO_ROOT / 'src' / 'trading' / 'signal_generation')]

model_selection_pkg = sys.modules.setdefault('src.trading.model_selection', types.ModuleType('src.trading.model_selection'))
model_selection_pkg.__path__ = [str(REPO_ROOT / 'src' / 'trading' / 'model_selection')]

config_spec = importlib.util.spec_from_file_location(
    'src.trading.config.trading_config',
    REPO_ROOT / 'src' / 'trading' / 'config' / 'trading_config.py'
)
config_module = importlib.util.module_from_spec(config_spec)
sys.modules[config_spec.name] = config_module
config_spec.loader.exec_module(config_module)

model_selection_spec = importlib.util.spec_from_file_location(
    'src.trading.model_selection',
    REPO_ROOT / 'src' / 'trading' / 'model_selection' / '__init__.py'
)
model_selection_module = importlib.util.module_from_spec(model_selection_spec)
sys.modules[model_selection_spec.name] = model_selection_module
model_selection_spec.loader.exec_module(model_selection_module)

signal_spec = importlib.util.spec_from_file_location(
    'src.trading.signal_generation.signal_pipeline',
    REPO_ROOT / 'src' / 'trading' / 'signal_generation' / 'signal_pipeline.py'
)
signal_module = importlib.util.module_from_spec(signal_spec)
sys.modules[signal_spec.name] = signal_module
signal_spec.loader.exec_module(signal_module)

TradingConfig = config_module.TradingConfig
SignalGenerationPipeline = signal_module.SignalGenerationPipeline
ModelSelectionResult = model_selection_module.ModelSelectionResult


class StubModelSelectorService:
    def __init__(self, results_by_timeframe):
        self.results_by_timeframe = results_by_timeframe
        self.calls = []

    def select_models_for_trading(self, *, timeframe, **kwargs):
        self.calls.append(timeframe)
        return self.results_by_timeframe[timeframe]


def _market_data():
    return pd.DataFrame(
        {
            'open': [1.0, 1.1],
            'high': [1.05, 1.15],
            'low': [0.95, 1.05],
            'close': [1.02, 1.12],
            'volume': [1000, 1100],
        }
    )


def _selection_result(model_suffix: str, regime_id: int, confidence: float) -> ModelSelectionResult:
    selection_metadata = {'timeframe': model_suffix}
    return ModelSelectionResult(
        selected_models={'random_forest': f'rf_{model_suffix}'},
        ensemble_weights={'random_forest': {f'rf_{model_suffix}': 1.0}},
        regime_id=regime_id,
        confidence_score=confidence,
        selection_metadata=selection_metadata,
        confirmation_status='single_timeframe'
    )


def test_cross_timeframe_confirmation_disabled():
    config = TradingConfig()
    config.cross_timeframe_confirmation['enabled'] = False

    pipeline = SignalGenerationPipeline(config)
    pipeline.model_selector_service = StubModelSelectorService(
        {
            '15m': _selection_result('15m', regime_id=1, confidence=0.8),
            '5m': _selection_result('5m', regime_id=2, confidence=0.6),
        }
    )

    result = asyncio.run(
        pipeline._select_models_for_trading(
            market_data=_market_data(),
            symbol='ETHUSDT',
            timestamp=datetime.utcnow(),
        )
    )

    assert result.confirmation_status == 'disabled'
    assert result.confidence_score == pytest.approx((0.8 + 0.6) / 2)
    assert result.confirmation_details['action'] == 'disabled'
    assert result.selection_metadata['cross_timeframe_confirmation']['enabled'] is False


def test_cross_timeframe_confirmation_downgrades_confidence():
    config = TradingConfig()
    config.cross_timeframe_confirmation.update(
        {
            'enabled': True,
            'max_regime_difference': 0,
            'max_confidence_delta': 0.05,
            'downgrade_confidence_factor': 0.4,
            'reject_on_disagreement': False,
        }
    )

    pipeline = SignalGenerationPipeline(config)
    pipeline.model_selector_service = StubModelSelectorService(
        {
            '15m': _selection_result('15m', regime_id=1, confidence=0.9),
            '5m': _selection_result('5m', regime_id=3, confidence=0.6),
        }
    )

    result = asyncio.run(
        pipeline._select_models_for_trading(
            market_data=_market_data(),
            symbol='ETHUSDT',
            timestamp=datetime.utcnow(),
        )
    )

    expected_original = (0.9 + 0.6) / 2
    assert result.confirmation_status == 'downgraded'
    assert result.selection_metadata['original_confidence'] == pytest.approx(expected_original)
    assert result.confidence_score == pytest.approx(expected_original * 0.4)
    assert 'regime_mismatch' in result.confirmation_details['disagreement_reasons']
    assert result.selection_metadata['cross_timeframe_confirmation']['action'] == 'downgrade'


def test_cross_timeframe_confirmation_rejects_on_disagreement():
    config = TradingConfig()
    config.cross_timeframe_confirmation.update(
        {
            'enabled': True,
            'max_regime_difference': 0,
            'max_confidence_delta': 0.05,
            'reject_on_disagreement': True,
            'rejection_confidence': 0.1,
        }
    )

    pipeline = SignalGenerationPipeline(config)
    pipeline.model_selector_service = StubModelSelectorService(
        {
            '15m': _selection_result('15m', regime_id=2, confidence=0.85),
            '5m': _selection_result('5m', regime_id=3, confidence=0.55),
        }
    )

    result = asyncio.run(
        pipeline._select_models_for_trading(
            market_data=_market_data(),
            symbol='ETHUSDT',
            timestamp=datetime.utcnow(),
        )
    )

    assert result.confirmation_status == 'rejected'
    assert result.selected_models == {'analyst': 'default', 'tactician': 'default'}
    assert result.confidence_score == pytest.approx(0.1)
    confirmation = result.confirmation_details
    assert confirmation['confirmation_passed'] is False
    assert confirmation['action'] == 'reject'
    assert 'confidence_delta_exceeded' in confirmation['disagreement_reasons'] or 'regime_mismatch' in confirmation['disagreement_reasons']
    assert result.selection_metadata['fallback'] is True
