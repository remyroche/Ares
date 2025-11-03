"""
Shared utilities for optimization module.

This submodule contains utilities shared across different optimization methods:
- Evolutionary search algorithms
- Feature engineering utilities
- Advanced metrics and evaluation
- Integration verification
"""

# Make imports optional to avoid circular dependencies
try:
    from .evolutionary_search import (
        EvolutionaryAlgorithmManager,
        EvolutionaryConfig,
        EvolutionaryResult,
        create_evolutionary_algorithm_manager,
        Individual
    )
    EVOLUTIONARY_AVAILABLE = True
except ImportError:
    EVOLUTIONARY_AVAILABLE = False
    EvolutionaryAlgorithmManager = None
    EvolutionaryConfig = None
    EvolutionaryResult = None
    create_evolutionary_algorithm_manager = None
    Individual = None

try:
    from .feature_engineering import (
        UnifiedFeatureEngineer,
        FeatureConfig,
        FeatureEngineeringResult,
        create_unified_feature_engineer
    )
    FEATURE_ENGINEERING_AVAILABLE = True
except ImportError:
    FEATURE_ENGINEERING_AVAILABLE = False
    UnifiedFeatureEngineer = None
    FeatureConfig = None
    FeatureEngineeringResult = None
    create_unified_feature_engineer = None

try:
    from .advanced_metrics import (
        AdvancedEvaluator,
        AdvancedEvaluationResult,
        create_advanced_evaluator
    )
    ADVANCED_METRICS_AVAILABLE = True
except ImportError:
    ADVANCED_METRICS_AVAILABLE = False
    AdvancedEvaluator = None
    AdvancedEvaluationResult = None
    create_advanced_evaluator = None

try:
    from .evaluation_metrics import (
        UnifiedEvaluator,
        create_unified_evaluator
    )
    EVALUATION_METRICS_AVAILABLE = True
except ImportError:
    EVALUATION_METRICS_AVAILABLE = False
    UnifiedEvaluator = None
    create_unified_evaluator = None

try:
    from .integration_verification import SharedUtilsIntegrationVerifier
    INTEGRATION_VERIFICATION_AVAILABLE = True
except ImportError:
    INTEGRATION_VERIFICATION_AVAILABLE = False
    SharedUtilsIntegrationVerifier = None

__all__ = [
    # Evolutionary search
    'EvolutionaryAlgorithmManager',
    'EvolutionaryConfig',
    'EvolutionaryResult',
    'create_evolutionary_algorithm_manager',
    'Individual',
    'EVOLUTIONARY_AVAILABLE',
    
    # Feature engineering
    'UnifiedFeatureEngineer',
    'FeatureConfig',
    'FeatureEngineeringResult',
    'create_unified_feature_engineer',
    'FEATURE_ENGINEERING_AVAILABLE',
    
    # Advanced metrics
    'AdvancedEvaluator',
    'AdvancedEvaluationResult',
    'create_advanced_evaluator',
    'ADVANCED_METRICS_AVAILABLE',
    
    # Evaluation metrics
    'UnifiedEvaluator',
    'create_unified_evaluator',
    'EVALUATION_METRICS_AVAILABLE',
    
    # Integration verification
    'SharedUtilsIntegrationVerifier',
    'INTEGRATION_VERIFICATION_AVAILABLE'
]

