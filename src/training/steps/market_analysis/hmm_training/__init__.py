"""
HMM Training Module

This module provides HMM-based model training that occurs after HMM clustering
in the MARKET_ANALYSIS stage. It includes base model training and ensemble training.

Components:
- hmm_models_training: Base models training, HPO, saving, metrics
- hmm_ensemble_training: Meta-model, HPO, saving, metrics
"""

from .hmm_models_training import HMMModelsTraining
from .hmm_ensemble_training import HMMEnsembleTraining

__all__ = ['HMMModelsTraining', 'HMMEnsembleTraining']