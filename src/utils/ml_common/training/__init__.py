"""
Training module for ML common utilities with universal validation integration.
"""

from typing import TYPE_CHECKING
from src.utils.lazy_module_loader import make_lazy_getattr, make_lazy_dir

# Define lazy loading map
_EXPORT_MAP = {
    'BaseTrainingStep': '.base_training_step',
    'PerRegimeTrainingStep': '.per_regime_training_step',
    'EnsembleTrainingStep': '.ensemble_training_step',
    'TrainingUtils': '.training_utils',
    
    # Universal validation integration
    'UniversalValidationIntegrator': '.universal_validation_integration',
    'ValidationIntegrationConfig': '.universal_validation_integration',
    'get_validation_integrator': '.universal_validation_integration',
    'validate_training_data': '.universal_validation_integration',
    'validate_trained_model': '.universal_validation_integration',
    'validate_hpo_trial': '.universal_validation_integration'
}

__all__ = list(_EXPORT_MAP.keys())

# Static typing support
if TYPE_CHECKING:
    from .base_training_step import BaseTrainingStep
    from .per_regime_training_step import PerRegimeTrainingStep
    from .ensemble_training_step import EnsembleTrainingStep
    from .training_utils import TrainingUtils
    from .universal_validation_integration import (
        UniversalValidationIntegrator,
        ValidationIntegrationConfig,
        get_validation_integrator,
        validate_training_data,
        validate_trained_model,
        validate_hpo_trial
    )

# Use generalized lazy loading helpers
__getattr__ = make_lazy_getattr(_EXPORT_MAP, __package__)
__dir__ = make_lazy_dir(_EXPORT_MAP, globals())
