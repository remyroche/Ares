"""
Unified configuration package

Provides shared loader utilities and re-exports domain-specific config classes
for ML, validation, and code quality to reduce duplication and standardize I/O.
"""

from .loader import (
    save_to_file,
    load_from_file,
    to_serializable_dict,
    instantiate_from_dict,
    merge_dicts,
)

# Domain re-exports (kept light to avoid heavy import trees)
from .ml import (
    BaseTrainingConfig,
    PerRegimeTrainingConfig,
    EnsembleTrainingConfig,
    TacticianTrainingConfig,
    HMMTrainingConfig,
    UniversalTimeframeConfig,
    UniversalTimeframeManager,
    save_training_config,
    load_training_config,
)

from .code_quality import (
    CodeQualityConfig,
    AnalysisConfig,
    ReportingConfig,
    save_code_quality_config,
    load_code_quality_config,
)

from .validation import (
    EnhancedValidationConfig,
    UniversalMLValidationConfig,
    save_validation_config,
    load_validation_config,
)

__all__ = [
    # Loader utilities
    "save_to_file",
    "load_from_file",
    "to_serializable_dict",
    "instantiate_from_dict",
    "merge_dicts",
    # ML
    "BaseTrainingConfig",
    "PerRegimeTrainingConfig",
    "EnsembleTrainingConfig",
    "TacticianTrainingConfig",
    "HMMTrainingConfig",
    "UniversalTimeframeConfig",
    "UniversalTimeframeManager",
    "save_training_config",
    "load_training_config",
    # Code Quality
    "CodeQualityConfig",
    "AnalysisConfig",
    "ReportingConfig",
    "save_code_quality_config",
    "load_code_quality_config",
    # Validation
    "EnhancedValidationConfig",
    "UniversalMLValidationConfig",
    "save_validation_config",
    "load_validation_config",
]

