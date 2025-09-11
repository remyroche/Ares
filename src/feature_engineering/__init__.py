"""
Feature Engineering Package

This package provides comprehensive feature engineering utilities including:
- Enhanced feature engineering with 200+ features
- Matrix operations with GPU acceleration
- Feature generation optimization
- Advanced feature interactions and transformations
- Cross-timeframe analysis and fractional differentiation
- Triple barrier labeling and regime-aware processing
- Comprehensive utility container with dependency injection

Available utilities:
- step06_utility_container: Dependency injection container for utility services
- step06_enhanced_feature_engineering: Advanced feature engineering utilities
- step06_comprehensive_implementation: Comprehensive implementation utilities
- step06_enhanced_feature_engineering_step: Feature engineering step utilities
- step06_labeling_components: Triple barrier labeling utilities
- cross_timeframe_interaction_features: Cross timeframe feature generation utilities
- cross_timeframe_analysis_pipeline: Comprehensive cross timeframe analysis
- fractional_differentiation_pipeline: Fractional differentiation for stationarity
- enhanced_matrix_operations: GPU-accelerated matrix operations

Usage:
    from src.feature_engineering import (
        Step06UtilityContainer,
        EnhancedFeatureEngineering,
        OptimizedTripleBarrierLabeling,
        CrossTimeframeFeatureGenerator,
        FractionalDifferentiationPipeline,
        EnhancedMatrixOperations
    )
"""

# Import main utility classes for easy access
from .step06_utility_container import (
    Step06UtilityContainer,
    UtilityConfig,
    get_utility_container,
    utility_container_context,
    inject_utilities
)

from .step06_enhanced_feature_engineering import (
    EnhancedFeatureEngineering
)

from .enhanced_matrix_operations import (
    EnhancedMatrixOperations,
    GPUError,
    MemoryError,
    OptimizationError
)

from .step06_comprehensive_implementation import (
    Step06ComprehensiveImplementation
)

from .step06_enhanced_feature_engineering_step import (
    EnhancedFeatureEngineeringStep
)

from .step06_labeling_components.optimized_triple_barrier_labeling import (
    OptimizedTripleBarrierLabeling
)

from .step06_labeling_components.fractional_triple_barrier_labeling import (
    FractionalTripleBarrierLabeling
)

from .step06_labeling_components.regime_specific_triple_barrier_optimizer import (
    RegimeSpecificTripleBarrierOptimizer
)

from .step06_labeling_components.profit_based_feature_engineering import (
    ProfitBasedFeatureEngineering
)

from .step06_labeling_components.regime_aware_triple_barrier_labeling import (
    RegimeAwareTripleBarrierLabeling
)

from .cross_timeframe_interaction_features import (
    CrossTimeframeFeatureGenerator,
    CrossTimeframeConfig,
    InteractionConfig
)

from .cross_timeframe_analysis_pipeline import (
    CrossTimeframeAnalysisPipeline,
    CrossTimeframeConfig as AnalysisConfig
)

from .fractional_differentiation_pipeline import (
    FractionalDifferentiationPipeline,
    FractionalDiffConfig
)

from .feature_generation_optimization import (
    FeatureGenerationOptimizer,
    FeatureOptimizationConfig,
    FeatureOptimizationResult,
    OptimizationMethod,
    get_feature_optimizer,
    optimize_feature_lookback
)

__all__ = [
    # Utility container
    'Step06UtilityContainer',
    'UtilityConfig',
    'get_utility_container',
    'utility_container_context',
    'inject_utilities',
    
    # Feature engineering
    'EnhancedFeatureEngineering',
    'Step06ComprehensiveImplementation',
    'EnhancedFeatureEngineeringStep',
    
    # Matrix operations
    'EnhancedMatrixOperations',
    'GPUError',
    'MemoryError',
    'OptimizationError',
    
    # Labeling components
    'OptimizedTripleBarrierLabeling',
    'FractionalTripleBarrierLabeling',
    'RegimeSpecificTripleBarrierOptimizer',
    'ProfitBasedFeatureEngineering',
    'RegimeAwareTripleBarrierLabeling',
    
    # Cross-timeframe analysis
    'CrossTimeframeFeatureGenerator',
    'CrossTimeframeConfig',
    'InteractionConfig',
    'CrossTimeframeAnalysisPipeline',
    'AnalysisConfig',
    
    # Fractional differentiation
    'FractionalDifferentiationPipeline',
    'FractionalDiffConfig',
    
    # Feature generation optimization
    'FeatureGenerationOptimizer',
    'FeatureOptimizationConfig',
    'FeatureOptimizationResult',
    'OptimizationMethod',
    'get_feature_optimizer',
    'optimize_feature_lookback'
]

__version__ = "1.0.0"
__author__ = "Ares Trading Bot Team"
__description__ = "Feature Engineering Package - Comprehensive feature engineering, matrix operations, and analysis utilities"