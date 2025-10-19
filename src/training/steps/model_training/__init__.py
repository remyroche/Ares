"""
Model Training Steps Module.

This module registers all model training steps for autonomous execution.
"""

from src.training.steps.base_step import step_registry
from .analyst_models_training_refactored import AnalystModelsTrainingStepRefactored

# Register model training steps
step_registry.register("analyst_models_training", AnalystModelsTrainingStepRefactored)