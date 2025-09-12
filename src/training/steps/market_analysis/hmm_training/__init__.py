"""
HMM Training Module

This module provides HMM-based model training that occurs after HMM clustering
in the MARKET_ANALYSIS stage. It includes base model training and ensemble training.

Components:
- hmm_models_training: Base models training, HPO, saving, metrics
- hmm_ensemble_training: Meta-model, HPO, saving, metrics
"""

from .hmm_models_training_refactored import (
    HMMModelsTrainingRefactored as HMMModelsTraining,
    create_hmm_models_training_refactored as create_hmm_models_training,
    execute_hmm_models_training_refactored as execute_hmm_models_training
)
from .hmm_ensemble_training import (
    HMMEnsembleTrainingRefactored as HMMEnsembleTraining,
    create_hmm_ensemble_training_refactored as create_hmm_ensemble_training,
    execute_hmm_ensemble_training_refactored as execute_hmm_ensemble_training
)

__all__ = [
    'HMMModelsTraining', 'HMMEnsembleTraining',
    'create_hmm_models_training', 'create_hmm_ensemble_training',
    'execute_hmm_models_training', 'execute_hmm_ensemble_training'
]