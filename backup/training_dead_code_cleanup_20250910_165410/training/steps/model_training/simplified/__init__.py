"""
Simplified Model Training Steps

This package provides simplified, consolidated model training steps that replace
the complex steps 9-17 with three focused modules:

1. General Model Training + HPO - For general ML models with hyperparameter optimization
2. Analyst Model Training + Ensemble Management - For Analyst system with ensemble capabilities  
3. Tactician Model Training + HPO - For Tactician system with advanced tactical features

All modules utilize M1 optimizations and modern ML practices.
"""

from .general_model_training import (
    GeneralModelTrainer,
    ModelTrainingConfig,
    ModelTrainingResults,
    ModelType,
    TaskType,
    ModelFactory
)

from .analyst_model_training import (
    AnalystModelTrainer,
    AnalystTrainingConfig,
    AnalystTrainingResults,
    AnalystModelType
)

from .tactician_model_training import (
    TacticianModelTrainer,
    TacticianTrainingConfig,
    TacticianTrainingResults,
    TacticianModelType,
    TacticalStrategy
)

__all__ = [
    # General Model Training
    'GeneralModelTrainer',
    'ModelTrainingConfig',
    'ModelTrainingResults',
    'ModelType',
    'TaskType',
    'ModelFactory',
    
    # Analyst Model Training
    'AnalystModelTrainer',
    'AnalystTrainingConfig',
    'AnalystTrainingResults',
    'AnalystModelType',
    
    # Tactician Model Training
    'TacticianModelTrainer',
    'TacticianTrainingConfig',
    'TacticianTrainingResults',
    'TacticianModelType',
    'TacticalStrategy'
]

__version__ = "1.0.0"
__author__ = "Ares Trading Bot Team"
__description__ = "Simplified Model Training Steps - Consolidated training for General, Analyst, and Tactician models"