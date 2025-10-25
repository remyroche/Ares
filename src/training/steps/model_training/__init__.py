"""
Model Training Steps Module.

This module registers all model training steps for autonomous execution.
"""

from src.training.steps.base_step import step_registry
from .analyst_models_training_refactored import AnalystModelsTrainingStepRefactored

# Import training steps
from .analyst_base_training_step import AnalystBaseTrainingStep
from .analyst_ensemble_training_step import AnalystEnsembleTrainingStep
from .tactician_base_training_step import TacticianBaseTrainingStep
from .tactician_ensemble_training_step import TacticianEnsembleTrainingStep

# Register model training steps
step_registry.register("analyst_base_training", AnalystBaseTrainingStep)
step_registry.register("analyst_ensemble_training", AnalystEnsembleTrainingStep)
step_registry.register("tactician_base_training", TacticianBaseTrainingStep)
step_registry.register("tactician_ensemble_training", TacticianEnsembleTrainingStep)
step_registry.register("analyst_models_training", AnalystModelsTrainingStepRefactored)