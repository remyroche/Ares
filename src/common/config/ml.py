"""
ML configuration re-exports and helpers.

This module exposes existing ML config dataclasses via a single import path and
provides save/load helpers backed by the shared loader utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Type, Union

from .loader import save_to_file as _save_to_file, load_from_file as _load_from_file

# Re-export existing classes to avoid breaking imports
from src.utils.ml_common.config.base_training_config import (
    BaseTrainingConfig,
    PerRegimeTrainingConfig,
    EnsembleTrainingConfig,
    TacticianTrainingConfig,
    HMMTrainingConfig,
)
from src.utils.ml_common.config.universal_timeframe_config import (
    UniversalTimeframeConfig,
    UniversalTimeframeManager,
)

def save_training_config(config: BaseTrainingConfig, filepath: Union[str, Path]) -> None:
    _save_to_file(config, filepath)

def load_training_config(
    filepath: Union[str, Path], target_cls: Type[BaseTrainingConfig] = BaseTrainingConfig
) -> BaseTrainingConfig:
    return _load_from_file(filepath, target_cls)  # type: ignore[return-value]

__all__ = [
    "BaseTrainingConfig",
    "PerRegimeTrainingConfig",
    "EnsembleTrainingConfig",
    "TacticianTrainingConfig",
    "HMMTrainingConfig",
    "UniversalTimeframeConfig",
    "UniversalTimeframeManager",
    "save_training_config",
    "load_training_config",
]
