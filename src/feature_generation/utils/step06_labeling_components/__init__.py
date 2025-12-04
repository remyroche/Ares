"""
Step06 Labeling Components

This package contains labeling components for step06 including:
- Optimized triple barrier labeling
- Fractional triple barrier labeling
- Regime-aware triple barrier labeling
- Profit-based feature engineering
- Trend-aware meta-labeling with ZigZag and confluence signals
"""

from .profit_based_feature_engineering import ProfitBasedFeatureEngineering
from .regime_aware_triple_barrier_labeling import RegimeAwareTripleBarrierLabeling
from .fractional_triple_barrier_labeling import FractionalTripleBarrierLabeling
from .optimized_triple_barrier_labeling import OptimizedTripleBarrierLabeling
from .trend_aware_meta_labeling import (
    TrendAwareMetaLabeler,
    TrendAwareTripleBarrierConfig,
    TrendDirection,
    ZigZagSwing,
    BollingerBandsSignal,
    OBVDivergence,
    ZigZagResult,
    create_trend_aware_meta_labeler,
    apply_trend_aware_meta_labeling,
)

try:
    OPTIMIZED_LABELING_AVAILABLE = True
except ImportError:
    OPTIMIZED_LABELING_AVAILABLE = False

try:
    FRACTIONAL_LABELING_AVAILABLE = True
except ImportError:
    FRACTIONAL_LABELING_AVAILABLE = False

try:
    REGIME_AWARE_LABELING_AVAILABLE = True
except ImportError:
    REGIME_AWARE_LABELING_AVAILABLE = False

try:
    PROFIT_BASED_FEATURES_AVAILABLE = True
except ImportError:
    PROFIT_BASED_FEATURES_AVAILABLE = False

try:
    TREND_AWARE_LABELING_AVAILABLE = True
except ImportError:
    TREND_AWARE_LABELING_AVAILABLE = False

__all__ = [
    'OptimizedTripleBarrierLabeling',
    'FractionalTripleBarrierLabeling',
    'RegimeAwareTripleBarrierLabeling',
    'ProfitBasedFeatureEngineering',
    'TrendAwareMetaLabeler',
    'TrendAwareTripleBarrierConfig',
    'TrendDirection',
    'ZigZagSwing',
    'BollingerBandsSignal',
    'OBVDivergence',
    'ZigZagResult',
    'create_trend_aware_meta_labeler',
    'apply_trend_aware_meta_labeling',
    'OPTIMIZED_LABELING_AVAILABLE',
    'FRACTIONAL_LABELING_AVAILABLE',
    'REGIME_AWARE_LABELING_AVAILABLE',
    'PROFIT_BASED_FEATURES_AVAILABLE',
    'TREND_AWARE_LABELING_AVAILABLE',
]
