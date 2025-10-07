import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import log_loss

from src.models.stacker_lgbm_gate import (
    StackerLGBMGateConfig,
    create_stacker_lgbm_gate,
)


def _build_synthetic_training_data(seed: int = 42):
    rng = np.random.default_rng(seed)
    n_samples = 400
    volatility = np.concatenate([
        np.full(n_samples // 2, 0.2),
        np.full(n_samples // 2, 1.5),
    ])
    trend = volatility + rng.normal(scale=0.1, size=n_samples)
    liquidity = rng.normal(size=n_samples)
    analyst_prob = np.clip(0.55 + 0.3 * (volatility > 1.0), 0.05, 0.95)
    tactician_prob = np.clip(0.65 - 0.25 * (volatility > 1.0), 0.05, 0.95)

    prob_switch = np.where(volatility > 1.0, analyst_prob, tactician_prob)
    y = (rng.random(n_samples) < prob_switch).astype(int)

    base_predictions = {
        'analyst': {'probability': analyst_prob},
        'tactician': {'probability': tactician_prob},
    }
    regime_features = {
        'volatility_level': volatility,
        'trend_score': trend,
        'liquidity_z': liquidity,
    }
    return base_predictions, regime_features, y


def _fit_gated_stacker():
    base_predictions, regime_features, y = _build_synthetic_training_data()
    config = StackerLGBMGateConfig()
    config.gating.epochs = 250
    model = create_stacker_lgbm_gate(config)
    model.fit(base_predictions, y, regime_features)
    return model, base_predictions, regime_features, y


def _load_signal_combiner_with_stubs():
    module_name = 'src.trading.signal_generation.signal_combiner'
    if module_name in sys.modules:
        module = sys.modules[module_name]
        analyst_mod = sys.modules['src.trading.signal_generation.analyst_signals']
        tactician_mod = sys.modules['src.trading.signal_generation.tactician_signals']
        return (
            module.SignalCombiner,
            analyst_mod.AnalystSignal,
            analyst_mod.SignalType,
            analyst_mod.SignalStrength,
            tactician_mod.TacticianSignal,
            tactician_mod.TimingSignal,
            tactician_mod.TimingConfidence,
            tactician_mod.PositionSizing,
        )

    saved_modules = {key: sys.modules.get(key) for key in [
        'src.trading',
        'src.trading.signal_generation',
        'src.trading.signal_generation.analyst_signals',
        'src.trading.signal_generation.tactician_signals',
    ]}

    trading_pkg = types.ModuleType('src.trading')
    trading_pkg.__path__ = []
    sys.modules['src.trading'] = trading_pkg

    signal_pkg = types.ModuleType('src.trading.signal_generation')
    signal_pkg.__path__ = []
    sys.modules['src.trading.signal_generation'] = signal_pkg

    analyst_module = types.ModuleType('src.trading.signal_generation.analyst_signals')

    class SignalType(Enum):
        BUY = 'buy'
        SELL = 'sell'
        HOLD = 'hold'
        CLOSE = 'close'

    class SignalStrength(Enum):
        WEAK = 'weak'
        MODERATE = 'moderate'
        STRONG = 'strong'
        VERY_STRONG = 'very_strong'

    @dataclass
    class AnalystSignal:
        timestamp: datetime
        symbol: str
        signal_type: SignalType
        signal_strength: SignalStrength
        confidence_score: float
        price_target: float | None = None
        stop_loss: float | None = None
        market_health_score: float = 0.0
        volatility_score: float = 0.0
        liquidation_risk_score: float = 0.0
        feature_importance: dict = field(default_factory=dict)
        ml_predictions: dict = field(default_factory=dict)
        nas_prediction: dict | None = None
        nas_confidence: float = 0.0
        nas_architecture_type: str | None = None
        regime_id: int | None = None
        metadata: dict = field(default_factory=dict)

    analyst_module.SignalType = SignalType
    analyst_module.SignalStrength = SignalStrength
    analyst_module.AnalystSignal = AnalystSignal
    sys.modules['src.trading.signal_generation.analyst_signals'] = analyst_module

    tactician_module = types.ModuleType('src.trading.signal_generation.tactician_signals')

    class TimingSignal(Enum):
        ENTER_LONG = 'enter_long'
        ENTER_SHORT = 'enter_short'
        EXIT_LONG = 'exit_long'
        EXIT_SHORT = 'exit_short'
        HOLD = 'hold'
        CLOSE_ALL = 'close_all'

    class TimingConfidence(Enum):
        LOW = 'low'
        MEDIUM = 'medium'
        HIGH = 'high'
        VERY_HIGH = 'very_high'

    @dataclass
    class PositionSizing:
        recommended_size: float
        max_size: float
        leverage: float
        risk_per_trade: float
        kelly_fraction: float
        confidence_multiplier: float

    @dataclass
    class TacticianSignal:
        timestamp: datetime
        symbol: str
        timing_signal: TimingSignal
        confidence: TimingConfidence
        confidence_score: float
        position_sizing: PositionSizing
        scenario_predictions: dict = field(default_factory=dict)
        risk_metrics: dict = field(default_factory=dict)
        timing_indicators: dict = field(default_factory=dict)
        tas_prediction: dict | None = None
        tas_confidence: float = 0.0
        tas_architecture_type: str | None = None
        signal_type: int | None = None
        metadata: dict = field(default_factory=dict)

    tactician_module.TimingSignal = TimingSignal
    tactician_module.TimingConfidence = TimingConfidence
    tactician_module.PositionSizing = PositionSizing
    tactician_module.TacticianSignal = TacticianSignal
    sys.modules['src.trading.signal_generation.tactician_signals'] = tactician_module

    spec = importlib.util.spec_from_file_location(
        module_name,
        Path(__file__).resolve().parents[1] / 'src' / 'trading' / 'signal_generation' / 'signal_combiner.py',
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)

    for key, value in saved_modules.items():
        if value is not None:
            sys.modules[key] = value

    return (
        module.SignalCombiner,
        analyst_module.AnalystSignal,
        analyst_module.SignalType,
        analyst_module.SignalStrength,
        tactician_module.TacticianSignal,
        tactician_module.TimingSignal,
        tactician_module.TimingConfidence,
        tactician_module.PositionSizing,
    )


def test_gated_stacker_increases_analyst_weight_in_high_volatility():
    model, base_predictions, regime_features, y = _fit_gated_stacker()
    outputs = model.combine_outputs(base_predictions, regime_features)
    weights = outputs['weights']
    volatility = regime_features['volatility_level']

    high_mask = volatility > 1.0
    low_mask = ~high_mask

    high_vol_weight = float(np.mean(weights['analyst'][high_mask]))
    low_vol_weight = float(np.mean(weights['analyst'][low_mask]))

    assert high_vol_weight > low_vol_weight

    combined_prob = outputs['probability']
    equal_weight_prob = 0.5 * base_predictions['analyst']['probability'] + 0.5 * base_predictions['tactician']['probability']
    gated_loss = log_loss(y, combined_prob)
    baseline_loss = log_loss(y, equal_weight_prob)
    assert gated_loss < baseline_loss


def test_signal_combiner_uses_gated_weights():
    model, base_predictions, regime_features, _ = _fit_gated_stacker()
    (
        SignalCombiner,
        AnalystSignal,
        SignalType,
        SignalStrength,
        TacticianSignal,
        TimingSignal,
        TimingConfidence,
        PositionSizing,
    ) = _load_signal_combiner_with_stubs()

    combiner = SignalCombiner({'stacker_model': model})

    timestamp = datetime.utcnow()
    analyst_signal = AnalystSignal(
        timestamp=timestamp,
        symbol='BTCUSDT',
        signal_type=SignalType.BUY,
        signal_strength=SignalStrength.STRONG,
        confidence_score=0.78,
        price_target=None,
        stop_loss=None,
        market_health_score=1.1,
        volatility_score=1.4,
        liquidation_risk_score=0.05,
        feature_importance={},
        ml_predictions={'probability': 0.8},
        metadata={
            'probability': 0.8,
            'utility': 0.06,
            'volatility_level': 1.4,
            'trend_score': 1.1,
        },
    )

    position = PositionSizing(
        recommended_size=0.5,
        max_size=1.0,
        leverage=2.0,
        risk_per_trade=0.02,
        kelly_fraction=0.3,
        confidence_multiplier=1.0,
    )
    tactician_signal = TacticianSignal(
        timestamp=timestamp,
        symbol='BTCUSDT',
        timing_signal=TimingSignal.ENTER_LONG,
        confidence=TimingConfidence.HIGH,
        confidence_score=0.62,
        position_sizing=position,
        scenario_predictions={},
        risk_metrics={'liquidity_z': -0.15, 'expected_utility': 0.03},
        metadata={'probability': 0.55, 'utility': 0.03},
    )

    additional_context = {
        'volatility_level': 1.4,
        'trend_score': 1.05,
        'liquidity_z': -0.1,
    }

    combined_signal = asyncio.run(
        combiner._weighted_average_combination(
            analyst_signal,
            tactician_signal,
            additional_context,
        )
    )

    assert combined_signal is not None
    assert 'gated' in combined_signal.metadata
    gated_meta = combined_signal.metadata['gated']
    assert gated_meta['weights']['analyst'] > gated_meta['weights']['tactician']
    assert combined_signal.confidence == pytest.approx(gated_meta['probability'])
    assert combined_signal.metadata['combined_confidence'] == pytest.approx(combined_signal.confidence)

