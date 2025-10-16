"""
from .profit_based_feature_engineering import ProfitBasedFeatureEngineering
from .regime_aware_triple_barrier_labeling import RegimeAwareTripleBarrierLabeling
from .fractional_triple_barrier_labeling import FractionalTripleBarrierLabeling
from .optimized_triple_barrier_labeling import OptimizedTripleBarrierLabeling
Step06 Labeling Components

This package contains labeling components for step06 including:
- Optimized triple barrier labeling
- Fractional triple barrier labeling
- Regime-aware triple barrier labeling
- Profit-based feature engineering
"""

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

__all__ = [
    'OptimizedTripleBarrierLabeling',
    'FractionalTripleBarrierLabeling',
    'RegimeAwareTripleBarrierLabeling',
    'ProfitBasedFeatureEngineering',
    'OPTIMIZED_LABELING_AVAILABLE',
    'FRACTIONAL_LABELING_AVAILABLE',
    'REGIME_AWARE_LABELING_AVAILABLE',
    'PROFIT_BASED_FEATURES_AVAILABLE'
]
