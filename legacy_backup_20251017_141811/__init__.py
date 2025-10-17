# src/training/steps/models_training/__init__.py

"""Models Training Package.

This package contains all models training components including:
- Tactician pre-ML orchestration
- Model training pipelines
- Training step orchestration
- Pre-ML feature engineering
- ModularComponent architecture for ML workflows

NEW FEATURES:
- ModularComponent base class for all training components
- ML-specific state management and performance monitoring
- Comprehensive error handling and logging
- Configuration management and validation
- Model checkpointing and serialization
- Migration utilities for existing components
"""

# Import ModularComponent architecture
from .unified_data_driven_pipeline.core import (
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
    create_modular_component,
    ModelsTrainingMigrationUtils,
    ComponentAnalysis,
    MigrationResult,
    analyze_component,
    validate_migration_compatibility,
    create_component_wrapper,
    migrate_component,
    generate_migration_report
)

# Import base components
from .components.base_component import BaseModelsTrainingComponent

# Import migrated components
from .components.analyst_models_training_modular import (
    AnalystModelsTrainingModular,
    AnalystModelType as AnalystModelsModelType,
    AnalystModelsTrainingConfig,
    AnalystModelsTrainingResult,
    create_analyst_models_training
)

from .components.analyst_ensemble_training_modular import (
    AnalystEnsembleTrainingModular,
    EnsembleMethod,
    AnalystEnsembleTrainingConfig,
    AnalystEnsembleTrainingResult,
    create_analyst_ensemble_training
)

from .components.ml_entry_timing_labeler_modular import (
    MLEntryTimingLabelerModular,
    LabelingMethod,
    MLModelType,
    MLEntryTimingConfig,
    MLEntryTimingResult,
    create_ml_entry_timing_labeler
)

from .unified_training_pipeline_modular import (
    UnifiedTrainingPipelineModular,
    TrainingPhase,
    UnifiedTrainingConfig,
    UnifiedTrainingResult,
    create_unified_training_pipeline
)

# Import migration utilities
from .migrate_components import ModelsTrainingMigrationManager
from .validate_migrations import MigrationValidator

__version__ = "2.0.0"

__all__ = [
    # Core ModularComponent architecture
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
    'create_modular_component',
    
    # Migration utilities
    'ModelsTrainingMigrationUtils',
    'ComponentAnalysis',
    'MigrationResult',
    'analyze_component',
    'validate_migration_compatibility',
    'create_component_wrapper',
    'migrate_component',
    'generate_migration_report',
    
    # Base components
    'BaseModelsTrainingComponent',
    
    # Migrated components
    'AnalystModelsTrainingModular',
    'AnalystModelsModelType',
    'AnalystModelsTrainingConfig',
    'AnalystModelsTrainingResult',
    'create_analyst_models_training',
    
    'AnalystEnsembleTrainingModular',
    'EnsembleMethod',
    'AnalystEnsembleTrainingConfig',
    'AnalystEnsembleTrainingResult',
    'create_analyst_ensemble_training',
    
    'MLEntryTimingLabelerModular',
    'LabelingMethod',
    'MLModelType',
    'MLEntryTimingConfig',
    'MLEntryTimingResult',
    'create_ml_entry_timing_labeler',
    
    'UnifiedTrainingPipelineModular',
    'TrainingPhase',
    'UnifiedTrainingConfig',
    'UnifiedTrainingResult',
    'create_unified_training_pipeline',
    
    # Migration utilities
    'ModelsTrainingMigrationManager',
    'MigrationValidator'
]
