"""
Core Training Components - Unified Training Architecture

This package provides the core training components for the unified training
architecture, including base trainers, model trainers, ensemble trainers,
and the pipeline orchestrator.

Key Components:
- BaseTrainer: Abstract base class for all trainers
- ModelTrainer: Individual model training implementation
- EnsembleTrainer: Ensemble model training implementation
- TrainingPipelineOrchestrator: Unified pipeline orchestration
"""

from .base_trainer import (
    BaseTrainer, TrainingConfig, TrainingResult, ValidationResult, 
    PredictionResult, TrainingRole, ModelType
)

from .model_trainer import ModelTrainer

from .ensemble_trainer import EnsembleTrainer, EnsembleStrategy

from .pipeline_orchestrator import (
    TrainingPipelineOrchestrator, PipelineConfig, PipelineResult,
    PipelinePhase, PipelineStatus
)

# New role-specific trainers
from .analyst_base_trainer import (
    AnalystBaseTrainer, AnalystTrainingConfig, AnalystModelType
)

from .tactician_base_trainer import (
    TacticianBaseTrainer, TacticianTrainingConfig, TacticianModelType
)

from .analyst_ensemble_trainer import (
    AnalystEnsembleTrainer, AnalystEnsembleTrainingConfig, EnsembleMethod
)

from .tactician_ensemble_trainer import (
    TacticianEnsembleTrainer, TacticianEnsembleTrainingConfig, TacticianEnsembleMethod
)

__all__ = [
    # Base classes
    'BaseTrainer',
    'TrainingConfig',
    'TrainingResult',
    'ValidationResult',
    'PredictionResult',
    'TrainingRole',
    'ModelType',
    
    # Trainers
    'ModelTrainer',
    'EnsembleTrainer',
    'EnsembleStrategy',
    
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
    
    # Orchestrator
    'TrainingPipelineOrchestrator',
    'PipelineConfig',
    'PipelineResult',
    'PipelinePhase',
    'PipelineStatus'
]
