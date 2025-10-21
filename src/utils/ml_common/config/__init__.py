"""
Configuration module for ML common utilities.
"""

from .base_training_config import (
    BaseTrainingConfig,
    EnsembleTrainingConfig,
    TacticianTrainingConfig,
    HMMTrainingConfig
)

# Create PerRegimeTrainingConfig as an alias to BaseTrainingConfig
# since we no longer train per-regime
PerRegimeTrainingConfig = BaseTrainingConfig

# Re-export unified loader helpers for convenience
try:
    from src.common.config import (
        save_to_file as save_config_to_file,
        load_from_file as load_config_from_file,
    )
except Exception:
    # In contexts where src.common is not available during partial installs/tests
    save_config_to_file = None
    load_config_from_file = None

__all__ = [
    'BaseTrainingConfig',
    'PerRegimeTrainingConfig',  # Alias for backward compatibility
    'EnsembleTrainingConfig',
    'TacticianTrainingConfig',
    'HMMTrainingConfig',
    'save_config_to_file',
    'load_config_from_file'
]
