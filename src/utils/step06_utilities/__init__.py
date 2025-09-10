"""
Step06 Utilities Bank

This package contains all the utilities that were previously part of step06.
These utilities are now available as a bank of reusable components that can be
imported and used by other parts of the system without being part of the main
pipeline.

Available utilities:
- step06_utility_container: Dependency injection container for utility services
- step06_enhanced_feature_engineering: Advanced feature engineering utilities
- step06_comprehensive_implementation: Comprehensive implementation utilities
- step06_enhanced_feature_engineering_step: Feature engineering step utilities
- step06_labeling_components: Triple barrier labeling utilities
- cross_timeframe_interaction_features: Cross timeframe feature generation utilities
- cross_timeframe_analysis_pipeline: Comprehensive cross timeframe analysis
- fractional_differentiation_pipeline: Fractional differentiation for stationarity

Usage:
    from src.utils.step06_utilities import (
        Step06UtilityContainer,
        EnhancedFeatureEngineering,
        OptimizedTripleBarrierLabeling,
        CrossTimeframeFeatureGenerator,
        FractionalDifferentiationPipeline
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

__all__ = [
    'Step06UtilityContainer',
    'UtilityConfig',
    'get_utility_container',
    'utility_container_context',
    'inject_utilities',
    'EnhancedFeatureEngineering',
    'Step06ComprehensiveImplementation',
    'EnhancedFeatureEngineeringStep',
    'OptimizedTripleBarrierLabeling',
    'FractionalTripleBarrierLabeling',
    'RegimeSpecificTripleBarrierOptimizer',
    'CrossTimeframeFeatureGenerator',
    'CrossTimeframeConfig',
    'InteractionConfig',
    'CrossTimeframeAnalysisPipeline',
    'AnalysisConfig',
    'FractionalDifferentiationPipeline',
    'FractionalDiffConfig'
]

__version__ = "1.0.0"
__author__ = "Ares Trading Bot Team"
__description__ = "Step06 Utilities Bank - Reusable components for feature engineering and labeling"