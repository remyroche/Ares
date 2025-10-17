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
    ModularComponent,
    ExampleModularComponent,
    ValidationLevel,
    ValidationResult,
    ErrorInfo,
    PerformanceMetric,
    MetricType,
    MetricLevel,
    ErrorSeverity,
    ErrorCategory,
    create_modular_architecture,
    create_modular_component
)

from .migration_utils import (
    ComponentMigrationAnalyzer,
    ComponentMigrationWrapper,
    MigrationReport,
    migrate_base_component_to_modular,
    create_component_wrapper,
    validate_migration_compatibility,
    generate_migration_report,
    create_simple_migration,
    create_advanced_migration
)

from .modular_pipeline_integration import (
    ModularPipelineOrchestrator,
    create_modular_pipeline_orchestrator,
    integrate_with_consolidated_pipeline
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
    'ModularComponent',
    'ExampleModularComponent',
    'ValidationLevel',
    'ValidationResult',
    'ErrorInfo',
    'PerformanceMetric',
    'MetricType',
    'MetricLevel',
    'ErrorSeverity',
    'ErrorCategory',
    'create_modular_architecture',
    'create_modular_component',
    
    # Migration utilities
    'ComponentMigrationAnalyzer',
    'ComponentMigrationWrapper',
    'MigrationReport',
    'migrate_base_component_to_modular',
    'create_component_wrapper',
    'validate_migration_compatibility',
    'generate_migration_report',
    'create_simple_migration',
    'create_advanced_migration',
    
    # Pipeline integration
    'ModularPipelineOrchestrator',
    'create_modular_pipeline_orchestrator',
    'integrate_with_consolidated_pipeline'
]
