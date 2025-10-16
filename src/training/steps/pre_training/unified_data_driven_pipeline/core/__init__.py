"""
Core components of the Unified Data-Driven Feature Pipeline

Note: The main pipeline classes have been moved to the consolidated implementation.
This module now only exports the core configuration and component classes.
"""

from .config import (
    UnifiedPipelineConfig,
    create_default_config,
    create_high_performance_config,
    create_memory_efficient_config,
    create_fast_config
)

from .economic_evaluator import (
    EconomicPeriodEvaluator,
    EconomicEvaluationConfig,
    EconomicPeriodEvaluationResult,
    create_economic_evaluator
)

from .intelligent_feature_selector import (
    IntelligentFeatureSelector,
    FeatureSelectionConfig,
    FeatureSelectionResult,
    create_intelligent_feature_selector
)

from .vectorbt_optimizer import (
    VectorBTOptimizer,
    VectorBTConfig,
    VectorBTOptimizationResult,
    create_vectorbt_optimizer
)

from .template_interaction_generator import (
    TemplateInteractionGenerator,
    TemplateConfig,
    InteractionTemplate,
    create_template_interaction_generator
)

from .modular_architecture import (
    ModularArchitecture,
    ValidationLevel,
    ErrorSeverity,
    ErrorCategory,
    create_modular_architecture
)

__all__ = [
    # Configuration
    'UnifiedPipelineConfig',
    'create_default_config',
    'create_high_performance_config',
    'create_memory_efficient_config',
    'create_fast_config',

    # Core components
    'EconomicPeriodEvaluator',
    'EconomicEvaluationConfig',
    'EconomicPeriodEvaluationResult',
    'create_economic_evaluator',

    'IntelligentFeatureSelector',
    'FeatureSelectionConfig',
    'FeatureSelectionResult',
    'create_intelligent_feature_selector',

    'VectorBTOptimizer',
    'VectorBTConfig',
    'VectorBTOptimizationResult',
    'create_vectorbt_optimizer',

    'TemplateInteractionGenerator',
    'TemplateConfig',
    'InteractionTemplate',
    'create_template_interaction_generator',

    'ModularArchitecture',
    'ValidationLevel',
    'ErrorSeverity',
    'ErrorCategory',
    'create_modular_architecture'
]
