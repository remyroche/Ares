# src/training/steps/models_training/__init__.py

"""Models Training Package - Unified Training Architecture

This package contains all models training components using the unified BaseTrainer architecture:

CORE COMPONENTS:
- BaseTrainer: Abstract base class for all trainers
- AnalystBaseTrainer: Base trainer for Analyst models
- TacticianBaseTrainer: Base trainer for Tactician models
- AnalystEnsembleTrainer: Ensemble trainer for Analyst models
- TacticianEnsembleTrainer: Ensemble trainer for Tactician models

TRAINING COMPONENTS:
- AnalystBaseTraining: Individual Analyst model training
- AnalystEnsembleTraining: Analyst ensemble model training
- TacticianBaseTraining: Individual Tactician model training
- TacticianEnsembleTraining: Tactician ensemble model training

KEY FEATURES:
- Unified training interface across all model types
- Common training patterns and lifecycle management
- Standardized configuration and validation
- Performance monitoring and checkpointing
- Error handling and recovery mechanisms
- Role-specific feature engineering
- Ensemble training capabilities
"""

# Import core training architecture
from .core import (
    # Base classes
    BaseTrainer,
    TrainingConfig,
    TrainingResult,
    ValidationResult,
    PredictionResult,
    TrainingRole,
    ModelType,
    
    # Role-specific base trainers
    AnalystBaseTrainer,
    AnalystTrainingConfig,
    AnalystModelType,
    TacticianBaseTrainer,
    TacticianTrainingConfig,
    TacticianModelType,
    
    # Role-specific ensemble trainers
    AnalystEnsembleTrainer,
    AnalystEnsembleTrainingConfig,
    EnsembleMethod,
    TacticianEnsembleTrainer,
    TacticianEnsembleTrainingConfig,
    TacticianEnsembleMethod,
)

# Import training components
from .components.analyst_base_training import (
    AnalystBaseTraining,
    AnalystBaseTrainingConfig,
    AnalystBaseTrainingResult,
    create_analyst_base_training,
    execute_analyst_base_training
)

from .components.analyst_ensemble_training import (
    AnalystEnsembleTraining,
    AnalystEnsembleTrainingConfig,
    AnalystEnsembleTrainingResult,
    create_analyst_ensemble_training,
    execute_analyst_ensemble_training
)

from .components.tactician_base_training import (
    TacticianBaseTraining,
    TacticianBaseTrainingConfig,
    TacticianBaseTrainingResult,
    create_tactician_base_training,
    execute_tactician_base_training
)

from .components.tactician_ensemble_training import (
    TacticianEnsembleTraining,
    TacticianEnsembleTrainingConfig,
    TacticianEnsembleTrainingResult,
    create_tactician_ensemble_training,
    execute_tactician_ensemble_training
)

# Import legacy components for backward compatibility
from .components.base_component import (
    ModularComponent,
    ErrorInfo,
    ErrorSeverity,
    ErrorCategory,
    BaseModelsTrainingComponent
)

from .components.ml_entry_timing_labeler_modular import (
    MLEntryTimingLabelerModular,
    LabelingMethod,
    MLModelType,
    MLEntryTimingConfig,
    MLEntryTimingResult,
    create_ml_entry_timing_labeler
)

__version__ = "3.0.0"

__all__ = [
    # Core training architecture
    'BaseTrainer',
    'TrainingConfig',
    'TrainingResult',
    'ValidationResult',
    'PredictionResult',
    'TrainingRole',
    'ModelType',
    
    # Role-specific base trainers
    'AnalystBaseTrainer',
    'AnalystTrainingConfig',
    'AnalystModelType',
    'TacticianBaseTrainer',
    'TacticianTrainingConfig',
    'TacticianModelType',
    
    # Role-specific ensemble trainers
    'AnalystEnsembleTrainer',
    'AnalystEnsembleTrainingConfig',
    'EnsembleMethod',
    'TacticianEnsembleTrainer',
    'TacticianEnsembleTrainingConfig',
    'TacticianEnsembleMethod',
    
    # Training components
    'AnalystBaseTraining',
    'AnalystBaseTrainingConfig',
    'AnalystBaseTrainingResult',
    'create_analyst_base_training',
    'execute_analyst_base_training',
    
    'AnalystEnsembleTraining',
    'AnalystEnsembleTrainingConfig',
    'AnalystEnsembleTrainingResult',
    'create_analyst_ensemble_training',
    'execute_analyst_ensemble_training',
    
    'TacticianBaseTraining',
    'TacticianBaseTrainingConfig',
    'TacticianBaseTrainingResult',
    'create_tactician_base_training',
    'execute_tactician_base_training',
    
    'TacticianEnsembleTraining',
    'TacticianEnsembleTrainingConfig',
    'TacticianEnsembleTrainingResult',
    'create_tactician_ensemble_training',
    'execute_tactician_ensemble_training',
    
    # Legacy components (for backward compatibility)
    'ModularComponent',
    'ErrorInfo',
    'ErrorSeverity',
    'ErrorCategory',
    'BaseModelsTrainingComponent',
    
    'MLEntryTimingLabelerModular',
    'LabelingMethod',
    'MLModelType',
    'MLEntryTimingConfig',
    'MLEntryTimingResult',
    'create_ml_entry_timing_labeler'
]