# src/training/steps/models_training/__init__.py

"""Models Training Package.

This package contains all models training components including:
- Tactician pre-ML orchestration
- Model training pipelines (standalone and unified)
- Training step orchestration
- Pre-ML feature engineering
- Short/Long timeframe differentiation support
"""

# Core pipeline exports
from .tactician_models_training_pipeline import (
    TacticianModelsTrainingPipeline,
    TacticianModelsTrainingPipelineConfig,
    TacticianModelsTrainingPipelineResult,
    TimeFrame as ModelsTimeFrame,
    execute_tactician_models_training_pipeline
)

from .tactician_ensemble_training_pipeline import (
    TacticianEnsembleTrainingPipeline,
    TacticianEnsembleTrainingPipelineConfig,
    TacticianEnsembleTrainingPipelineResult,
    TimeFrame as EnsembleTimeFrame,
    execute_tactician_ensemble_training_pipeline
)

# Unified pipeline (legacy support)
from .tactician_training_pipeline import (
    TacticianTrainingPipeline,
    TacticianTrainingPipelineConfig,
    TacticianTrainingPipelineResult,
    TimeFrame,
    execute_tactician_training_pipeline
)

# Individual training components
from .tactician_models_training import (
    TacticianModelsTrainingStep,
    TacticianModelsTrainingConfig,
    TacticianModelsTrainingResult,
    TacticianModelType,
    execute_tactician_models_training
)

from .tactician_ensemble_training import (
    TacticianEnsembleTrainingStep,
    TacticianEnsembleTrainingConfig,
    TacticianEnsembleTrainingResult,
    execute_tactician_ensemble_training
)

__version__ = "1.0.0"