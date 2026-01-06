"""
Category-Specific Feature Generators

This module provides feature generators organized by category, including:
- Returns features (price returns, log returns, etc.)
- Momentum features (RSI, MACD, etc.)
- Volume features (volume ratios, OBV, etc.)
- Volatility features (Bollinger Bands, ATR, etc.)
- Trend features (moving averages, trend indicators, etc.)
- Oscillator features (stochastic, Williams %R, etc.)
- Support/Resistance features (pivot points, levels, etc.)
- Candlestick pattern features (doji, hammer, etc.)
- HMM regime features (regime detection, regime-specific features, etc.)
- Acceleration features (momentum, acceleration, jerk, trend strength, etc.)
- Interaction features (momentum divergence, volatility-volume, etc.)
- Cross-timeframe features (multi-timeframe momentum, volatility, etc.)
- Entropy features (15 comprehensive entropy indicators)
- Regime transition features (transition probabilities, switch patterns, dynamics)
- Regime persistence features (duration, survival probability, exhaustion indicators)
- Market structure features (support/resistance within regimes, swing structure, fractals)
- Regime probability features (HMM probabilities, confidence, dynamics, patterns)
- Regime uncertainty features (classification entropy, confusion scores, ambiguity)

LAZY LOADING: Categories are loaded only when accessed to improve import performance.
"""

import sys
from typing import Any, Dict, Optional, Callable
import warnings

# Lazy loading cache
_category_modules: Dict[str, Any] = {}
_category_imports: Dict[str, Dict[str, str]] = {
    'returns': {
        'module': '.returns',
        'classes': [
            'ReturnsFeatureGenerator', 'LogReturnsGenerator', 'SimpleReturnsGenerator',
            'CumulativeReturnsGenerator', 'RollingReturnsGenerator', 'ReturnsVolatilityGenerator',
            'ReturnsSkewnessGenerator', 'ReturnsKurtosisGenerator', 'ReturnGenerator',
            'SharpeRatioGenerator', 'AmihudIlliquidityRatioGenerator',
            'AdvancedCumulativeReturnsGenerator', 'RollingZScoreReturnsGenerator',
            'ARCoefficientsGenerator', 'LjungBoxTestGenerator',
            'create_returns_generators', 'create_default_returns_generators'
        ]
    },
    'momentum': {
        'module': '.momentum',
        'classes': [
            'MomentumFeatureGenerator', 'RSIGenerator', 'MACDGenerator', 'StochasticGenerator',
            'WilliamsRGenerator', 'MomentumOscillatorGenerator', 'RateOfChangeGenerator',
            'AdvancedMomentumGenerator', 'PriceAccelerationGenerator', 'VolumeMomentumGenerator',
            'MomentumEndpointsGenerator', 'MACDDeltaGenerator', 'RSIZScoreGenerator',
            'StochasticKDGenerator', 'DonchianChannelGenerator',
            'create_momentum_generators', 'create_default_momentum_generators'
        ]
    },
    'volume': {
        'module': '.volume',
        'classes': [
            'VolumeFeatureGenerator', 'VolumeZScoreGenerator', 'VolumeMARatiosGenerator',
            'CMFGenerator', 'VWAPDeviationsGenerator', 'OrderFlowImbalanceGenerator',
            'VolumeVolatilityElasticityGenerator', 'create_default_volume_generators'
        ]
    },
    'volatility': {
        'module': '.volatility',
        'classes': [
            'VolatilityFeatureGenerator', 'create_default_volatility_generators'
        ]
    },
    'trend': {
        'module': '.trend',
        'classes': ['TrendFeatureGenerator']
    },
    'oscillator': {
        'module': '.oscillator',
        'classes': ['OscillatorFeatureGenerator']
    },
    'support_resistance': {
        'module': '.support_resistance',
        'classes': ['SupportResistanceFeatureGenerator']
    },
    'custom_support_resistance': {
        'module': '.custom_support_resistance',
        'classes': [
            'SRStrengthGenerator', 'SRDistanceGenerator', 'SRTouchCountGenerator',
            'SRQualityGenerator', 'VolumeWeightedSRGenerator', 'DynamicSRGenerator',
            'create_default_custom_sr_generators'
        ]
    },
    'candlestick_pattern': {
        'module': '.candlestick_pattern',
        'classes': ['CandlestickPatternFeatureGenerator']
    },
    'interaction': {
        'module': '.interaction',
        'classes': [
            'InteractionFeatureGenerator', 'MomentumDivergenceGenerator', 'MomentumVolumeGenerator',
            'MomentumVolatilityGenerator', 'MomentumTrendGenerator', 'VolatilityVolumeGenerator',
            'VolatilityPriceGenerator', 'VolatilityHighLowGenerator', 'VolatilityMomentumGenerator',
            'VolatilityTrendGenerator', 'create_interaction_generators',
            'CrossTimeframeInteractionGenerator', 'FeatureRatioGenerator',
            'PolynomialFeatureGenerator', 'CorrelationInteractionGenerator',
            'create_default_interaction_generators'
        ]
    },
    'cross_timeframe': {
        'module': '.cross_timeframe',
        'classes': [
            'CrossTimeframeFeatureGenerator', 'create_default_cross_timeframe_generators'
        ]
    },
    'entropy': {
        'module': '.entropy',
        'classes': ['create_default_entropy_generators']
    },
    'autoencoder': {
        'module': '.autoencoder',
        'classes': [
            'create_default_autoencoder_generators', 'PatchTSTRepresentationGenerator',
            'TFTEncoderRepresentationGenerator', 'AutoencoderRepresentationGenerator',
            'ContrastiveLearningGenerator'
        ]
    },
    'microstructure': {
        'module': '.microstructure_features',
        'classes': ['create_default_microstructure_generators']
    },
    'time': {
        'module': '.time',
        'classes': ['create_default_time_generators']
    },
    'advanced_statistical': {
        'module': '.advanced_statistical',
        'classes': [
            'HurstExponentGenerator', 'JumpIndicatorsGenerator', 'CVaRGenerator',
            'MaxDrawdownGenerator', 'RollingSkewnessKurtosisGenerator',
            'TrendPersistenceGenerator', 'create_advanced_statistical_generators'
        ]
    },
    'regime_feature_integration': {
        'module': '.regime_feature_integration',
        'classes': [
            'RegimeFeatureIntegration', 'RegimeFeatureConfig', 'generate_regime_features'
        ]
    },
    'regime_transitions': {
        'module': '.regime_transitions',
        'classes': [
            'RegimeTransitionGenerator', 'create_regime_transition_generators'
        ]
    },
    'regime_persistence': {
        'module': '.regime_persistence',
        'classes': [
            'RegimePersistenceGenerator', 'create_regime_persistence_generators'
        ]
    },
    'market_structure': {
        'module': '.market_structure',
        'classes': [
            'MarketStructureGenerator', 'create_market_structure_generators'
        ]
    },
    'regime_probability': {
        'module': '.regime_probability',
        'classes': [
            'RegimeProbabilityGenerator', 'create_regime_probability_generators'
        ]
    },
    'regime_uncertainty': {
        'module': '.regime_uncertainty',
        'classes': [
            'RegimeUncertaintyGenerator', 'create_regime_uncertainty_generators'
        ]
    },
    'multi_timeframe_ewma': {
        'module': '.multi_timeframe_ewma',
        'classes': [
            'MultiTimeframeEWMAReturnsGenerator', 'MultiTimeframeEWMAVolatilityGenerator',
            'MultiTimeframeEWMATrendGenerator', 'MultiTimeframeEWMAVolumeGenerator'
        ]
    },
    'labeling': {
        'module': '.labeling',
        'classes': [
            'LabelingFeatureGenerator', 'MarketFeaturesGenerator',
            'SignalFeaturesGenerator', 'LabelingFeatureConfig'
        ]
    }
}

def _load_category_lazy(category_name: str) -> Any:
    """Load a category module lazily."""
    if category_name in _category_modules:
        return _category_modules[category_name]
    
    if category_name not in _category_imports:
        raise AttributeError(f"Category '{category_name}' not found in feature generation categories")
    
    try:
        module_path = _category_imports[category_name]['module']
        module = __import__(module_path, package=__name__, fromlist=[''])
        _category_modules[category_name] = module
        return module
    except ImportError as e:
        warnings.warn(f"Failed to load category '{category_name}': {e}")
        _category_modules[category_name] = None
        return None

def __getattr__(name: str) -> Any:
    """Lazy loading implementation for category attributes."""
    # Check if this is a category class/function we need to load
    for category_name, category_info in _category_imports.items():
        if name in category_info['classes']:
            module = _load_category_lazy(category_name)
            if module is not None:
                return getattr(module, name, None)
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Minimal immediate exports - everything else is lazy loaded
__all__ = [
    # Core category generators (lazy loaded)
    'ReturnsFeatureGenerator',
    'LogReturnsGenerator', 
    'SimpleReturnsGenerator',
    'CumulativeReturnsGenerator',
    'RollingReturnsGenerator',
    'ReturnsVolatilityGenerator',
    'ReturnsSkewnessGenerator',
    'ReturnsKurtosisGenerator',
    'ReturnGenerator',
    'SharpeRatioGenerator',
    'AmihudIlliquidityRatioGenerator',
    'AdvancedCumulativeReturnsGenerator',
    'RollingZScoreReturnsGenerator',
    'ARCoefficientsGenerator',
    'LjungBoxTestGenerator',
    'create_returns_generators',
    'create_default_returns_generators',
    
    'MomentumFeatureGenerator',
    'RSIGenerator',
    'MACDGenerator',
    'StochasticGenerator',
    'WilliamsRGenerator',
    'MomentumOscillatorGenerator',
    'RateOfChangeGenerator',
    'AdvancedMomentumGenerator',
    'PriceAccelerationGenerator',
    'VolumeMomentumGenerator',
    'MomentumEndpointsGenerator',
    'MACDDeltaGenerator',
    'RSIZScoreGenerator',
    'StochasticKDGenerator',
    'DonchianChannelGenerator',
    'create_momentum_generators',
    'create_default_momentum_generators',
    
    'VolumeFeatureGenerator',
    'VolumeZScoreGenerator',
    'VolumeMARatiosGenerator',
    'CMFGenerator',
    'VWAPDeviationsGenerator',
    'OrderFlowImbalanceGenerator',
    'VolumeVolatilityElasticityGenerator',
    'create_default_volume_generators',
    
    'VolatilityFeatureGenerator',
    'create_default_volatility_generators',
    
    'TrendFeatureGenerator',
    'OscillatorFeatureGenerator',
    'SupportResistanceFeatureGenerator',
    
    'SRStrengthGenerator',
    'SRDistanceGenerator',
    'SRTouchCountGenerator',
    'SRQualityGenerator',
    'VolumeWeightedSRGenerator',
    'DynamicSRGenerator',
    'create_default_custom_sr_generators',
    
    'CandlestickPatternFeatureGenerator',
    
    # Interaction features
    'InteractionFeatureGenerator',
    'MomentumDivergenceGenerator',
    'MomentumVolumeGenerator',
    'MomentumVolatilityGenerator',
    'MomentumTrendGenerator',
    'VolatilityVolumeGenerator',
    'VolatilityPriceGenerator',
    'VolatilityHighLowGenerator',
    'VolatilityMomentumGenerator',
    'VolatilityTrendGenerator',
    'create_interaction_generators',
    'FeatureRatioGenerator',
    'PolynomialFeatureGenerator',
    'CorrelationInteractionGenerator',
    'create_default_interaction_generators',

    # Other categories
    'create_returns_generators',
    'create_default_returns_generators',
    'create_momentum_generators',
    'create_default_momentum_generators',
    'create_default_volume_generators',
    'create_default_microstructure_generators',
    'create_default_entropy_generators',
    'create_default_autoencoder_generators',
    'PatchTSTRepresentationGenerator',
    'TFTEncoderRepresentationGenerator',
    'AutoencoderRepresentationGenerator',
    'ContrastiveLearningGenerator',
    'create_default_cross_timeframe_generators',
    'create_default_time_generators',
    # NEW CATEGORIES - Advanced Statistical and Spectral/Wavelet
    'HurstExponentGenerator',
    'JumpIndicatorsGenerator',
    'CVaRGenerator',
    'MaxDrawdownGenerator',
    'RollingSkewnessKurtosisGenerator',
    'TrendPersistenceGenerator',
    'create_advanced_statistical_generators',
    'RegimeFeatureIntegration',
    'RegimeFeatureConfig',
    'generate_regime_features',
    # NEW REGIME FEATURE CATEGORIES - Enhanced Regime Analysis
    'RegimeTransitionGenerator',
    'create_regime_transition_generators',
    'RegimePersistenceGenerator',
    'create_regime_persistence_generators',
    'MarketStructureGenerator',
    'create_market_structure_generators',
    'RegimeProbabilityGenerator',
    'create_regime_probability_generators',
    'RegimeUncertaintyGenerator',
    'create_regime_uncertainty_generators',
    # Multi-timeframe EWMA features
    'MultiTimeframeEWMAReturnsGenerator',
    'MultiTimeframeEWMAVolatilityGenerator',
    'MultiTimeframeEWMATrendGenerator',
    'MultiTimeframeEWMAVolumeGenerator',
    # Labeling features for meta-labeling
    'LabelingFeatureGenerator',
    'MarketFeaturesGenerator',
    'SignalFeaturesGenerator',
    'LabelingFeatureConfig',
    # Meta features
]
