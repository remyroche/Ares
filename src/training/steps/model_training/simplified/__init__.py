"""
Simplified Model Training Steps

This package provides simplified, consolidated model training steps that replace
the complex steps 9-17 with three focused modules:

1. General Model Training + HPO - For general ML models with hyperparameter optimization
2. Analyst Model Training + Ensemble Management - For Analyst system with ensemble capabilities  
3. Tactician Model Training + HPO - For Tactician system with advanced tactical features

All modules utilize M1 optimizations and modern ML practices.
"""

# General model training has been removed from the pipeline
# from .general_model_training import (
#     GeneralModelTrainer,
#     ModelTrainingConfig,
#     ModelTrainingResults,
#     ModelType,
#     TaskType,
#     ModelFactory
# )

# Note: analyst_model_training.py and tactician_model_training.py have been removed
# Use the new comprehensive training steps instead:
# - analyst_models_training.py
# - analyst_ensemble_training.py  
# - tactician_models_training.py
# - tactician_ensemble_training.py

__all__ = [
    # General model training has been removed from the pipeline
    # 'GeneralModelTrainer',
    # 'ModelTrainingConfig',
    # 'ModelTrainingResults',
    # 'ModelType',
    # 'TaskType',
    # 'ModelFactory'
]

__version__ = "1.0.0"
__author__ = "Ares Trading Bot Team"
__description__ = "Simplified Model Training Steps - Consolidated training for General, Analyst, and Tactician models"