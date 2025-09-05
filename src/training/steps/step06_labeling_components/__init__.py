"""
Step06 Labeling Components

This package contains labeling components for step06 including:
- Optimized triple barrier labeling
- Fractional triple barrier labeling
- Regime-aware triple barrier labeling
- Profit-based feature engineering
"""

try:
    from .optimized_triple_barrier_labeling import OptimizedTripleBarrierLabeling
    OPTIMIZED_LABELING_AVAILABLE = True
except ImportError:
    OPTIMIZED_LABELING_AVAILABLE = False

try:
    from .fractional_triple_barrier_labeling import FractionalTripleBarrierLabeling
    FRACTIONAL_LABELING_AVAILABLE = True
except ImportError:
    FRACTIONAL_LABELING_AVAILABLE = False

try:
    from .regime_aware_triple_barrier_labeling import RegimeAwareTripleBarrierLabeling
    REGIME_AWARE_LABELING_AVAILABLE = True
except ImportError:
    REGIME_AWARE_LABELING_AVAILABLE = False

try:
    from .profit_based_feature_engineering import ProfitBasedFeatureEngineering
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