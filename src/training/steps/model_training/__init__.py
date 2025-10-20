"""
Model Training Steps Module.

This module registers all model training steps for autonomous execution.
"""

from src.training.steps.base_step import step_registry
from .analyst_models_training_refactored import AnalystModelsTrainingStepRefactored

# Import new BaseClass components
from ..models_training.components.analyst_models_training_modular import AnalystModelsTrainingModular
from ..models_training.components.analyst_ensemble_training_modular import AnalystEnsembleTrainingModular
from .tactician_models_training_refactored import TacticianModelsTrainingStepRefactored
from .tactician_ensemble_training import TacticianEnsembleTrainingStep

# Register legacy model training steps
step_registry.register("analyst_models_training", AnalystModelsTrainingStepRefactored)

# Register new BaseClass model training steps
step_registry.register("analyst_base_training", AnalystModelsTrainingModular)
step_registry.register("analyst_ensemble_training", AnalystEnsembleTrainingModular)
step_registry.register("tactician_base_training", TacticianModelsTrainingStepRefactored)
step_registry.register("tactician_ensemble_training", TacticianEnsembleTrainingStep)