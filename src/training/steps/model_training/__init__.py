"""
Model Training Steps Module.

This module registers all model training steps for autonomous execution.
"""

from src.training.steps.base_step import step_registry
from .analyst_models_training_refactored import AnalystModelsTrainingStepRefactored

# Import new BaseClass components
from ..models_training.components.analyst_base_training import AnalystBaseTraining
from ..models_training.components.analyst_ensemble_training_simple import SimpleAnalystEnsembleTraining
from ..models_training.components.tactician_base_training import TacticianBaseTraining
from ..models_training.components.tactician_ensemble_training import TacticianEnsembleTraining

# Register legacy model training steps
step_registry.register("analyst_models_training", AnalystModelsTrainingStepRefactored)

# Register new BaseClass model training steps
step_registry.register("analyst_base_training", AnalystBaseTraining)
step_registry.register("analyst_ensemble_training", SimpleAnalystEnsembleTraining)
step_registry.register("tactician_base_training", TacticianBaseTraining)
step_registry.register("tactician_ensemble_training", TacticianEnsembleTraining)