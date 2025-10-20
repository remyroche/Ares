"""
Model Training Steps Module.

This module registers all model training steps for autonomous execution.
"""

from src.training.steps.base_step import step_registry
from .analyst_models_training_refactored import AnalystModelsTrainingStepRefactored

# Import new BaseClass step wrappers
from ..models_training.analyst_base_training_step import AnalystBaseTrainingStep
from ..models_training.analyst_ensemble_training_step import AnalystEnsembleTrainingStep
from ..models_training.tactician_base_training_step import TacticianBaseTrainingStep
from ..models_training.tactician_ensemble_training_step import TacticianEnsembleTrainingStep

# Register legacy model training steps
step_registry.register("analyst_models_training", AnalystModelsTrainingStepRefactored)

# Register new BaseClass model training steps
step_registry.register("analyst_base_training", AnalystBaseTrainingStep)
step_registry.register("analyst_ensemble_training", AnalystEnsembleTrainingStep)
step_registry.register("tactician_base_training", TacticianBaseTrainingStep)
step_registry.register("tactician_ensemble_training", TacticianEnsembleTrainingStep)