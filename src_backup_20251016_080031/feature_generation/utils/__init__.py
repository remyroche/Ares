"""
Feature Generation Utils Package

This package contains advanced feature engineering utilities, optimization systems,
and analysis tools. Previously located in src/feature_engineering/.

Main Components:
- Optimization system (unified_optimizer.py, optimization/)
- Advanced feature engineering (step06_* files)
- Cross-timeframe analysis (cross_timeframe_*)
- Matrix operations and GPU acceleration
- Triple barrier labeling and regime analysis
- Utility containers and dependency injection
"""

# Import main utility classes for easy access
try:
    from .step06_utility_container import (
        Step06UtilityContainer,
        UtilityConfig,
        get_utility_container,
        utility_container_context,
        inject_utilities
    )
except ImportError:
    pass

try:
    from .step06_enhanced_feature_engineering import (
        EnhancedFeatureEngineering
    )
except ImportError:
    pass

try:
    from .optimization import (
        FeatureGenerationOptimizer,
        FeatureOptimizationConfig,
        FeatureOptimizationResult,
        OptimizationMethod,
        get_feature_optimizer,
        optimize_feature_lookback,
        get_optimization_config,
        LookbackOptimizer  # Backward compatibility
    )
    OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    OPTIMIZATION_AVAILABLE = False
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Optimization system not available: {e}")

# Advanced utilities
try:
    from .cross_timeframe_analysis_pipeline import CrossTimeframeAnalysisPipeline
    from .fractional_differentiation_pipeline import FractionalDifferentiationPipeline
    from .enhanced_matrix_operations import EnhancedMatrixOperations
except ImportError:
    pass

# Feature validation
try:
    from .math_validation import (
        validate_feature_quality,
        validate_features_dataframe,
        feature_validation_decorator
    )
except ImportError:
    pass

__version__ = "2.0.0"
__description__ = "Feature Generation Utils - Advanced feature engineering and optimization utilities"
