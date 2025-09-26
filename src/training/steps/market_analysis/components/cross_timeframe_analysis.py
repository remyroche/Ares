"""
Cross Timeframe Analysis Component - Updated to use PID-Based Feature Generation

This component has been updated to use the new PID-based feature generation system
that replaces the old cross_timeframe_analysis functionality with comprehensive
feature generation including interaction, polynomial, and cross-timeframe features.

Key Features:
- Uses optimized lookback periods from feature_lookback_optimization
- Leverages matrix_operations/ for all calculations
- Generates up to 200 total features (100 interaction + 50 polynomial + 50 cross-timeframe)
- Comprehensive validation and error handling
- Hardware-optimized computations

This is a compatibility wrapper that maintains the same interface while using
the new PID-based feature generation system.
"""

# Import the new PID-based feature generation component
from ..pid_based_feature_generation.pid_based_feature_generation_component import (
    PIDBasedFeatureGenerationComponent as _PIDBasedFeatureGenerationComponent
)

# Import base component for compatibility
from .base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult

# Re-export the component with the original name for backward compatibility
CrossTimeframeAnalysisComponent = _PIDBasedFeatureGenerationComponent

# Maintain backward compatibility by providing the same interface
__all__ = [
    "CrossTimeframeAnalysisComponent",
    "BaseMarketAnalysisComponent",
    "ComponentConfig",
    "ComponentResult",
]
