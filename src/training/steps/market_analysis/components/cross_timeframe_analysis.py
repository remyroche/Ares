"""
Cross Timeframe Analysis Component - Updated to use Interactive Feature Generation

This component has been updated to use the new interactive feature generation system
that replaces the old PID-based feature generation with comprehensive
feature generation including interaction, lookback optimization, and cross-timeframe features.

Key Features:
- Uses optimized lookback periods from feature_lookback_optimization
- Leverages matrix_operations/ for all calculations
- Data-driven Bayesian lookback optimization
- Comprehensive validation and error handling
- Hardware-optimized computations

This is a compatibility wrapper that maintains the same interface while using
the new interactive feature generation system.
"""

# Import the battle-tested interactive feature generation component
from src.feature_generation.utils.enhanced_data_driven_interaction_generator import (
    EnhancedDataDrivenInteractionGenerator as _InteractiveFeatureGenerationComponent
)

# Import base component for compatibility
from .base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult

# Re-export the component with the original name for backward compatibility
CrossTimeframeAnalysisComponent = _InteractiveFeatureGenerationComponent

# Maintain backward compatibility by providing the same interface
__all__ = ['CrossTimeframeAnalysisComponent', 'BaseMarketAnalysisComponent', 'ComponentConfig', 'ComponentResult']
