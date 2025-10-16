"""
Validation configuration re-exports and helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Type, Union

from .loader import save_to_file as _save_to_file, load_from_file as _load_from_file

from src.utils.ml_common.validation.enhanced_validation import (
    EnhancedValidationConfig,
)
from src.utils.ml_common.validation.universal_ml_validation import (
    UniversalMLValidationConfig,
)

def save_validation_config(config: Union[EnhancedValidationConfig, UniversalMLValidationConfig], filepath: Union[str, Path]) -> None:
    _save_to_file(config, filepath)

def load_validation_config(
    filepath: Union[str, Path], target_cls: Type[Union[EnhancedValidationConfig, UniversalMLValidationConfig]] = EnhancedValidationConfig
) -> Union[EnhancedValidationConfig, UniversalMLValidationConfig]:
    return _load_from_file(filepath, target_cls)  # type: ignore[return-value]

__all__ = [
    "EnhancedValidationConfig",
    "UniversalMLValidationConfig",
    "save_validation_config",
    "load_validation_config",
]
