"""
Code quality configuration re-exports and helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Type, Union

from .loader import save_to_file as _save_to_file, load_from_file as _load_from_file

from code_quality.core.config import (
    CodeQualityConfig,
    AnalysisConfig,
    ReportingConfig,
)

def save_code_quality_config(config: CodeQualityConfig, filepath: Union[str, Path]) -> None:
    _save_to_file(config, filepath)

def load_code_quality_config(
    filepath: Union[str, Path], target_cls: Type[CodeQualityConfig] = CodeQualityConfig
) -> CodeQualityConfig:
    return _load_from_file(filepath, target_cls)  # type: ignore[return-value]

__all__ = [
    "CodeQualityConfig",
    "AnalysisConfig",
    "ReportingConfig",
    "save_code_quality_config",
    "load_code_quality_config",
]
