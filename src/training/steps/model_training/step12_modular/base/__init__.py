"""
Step 12 Modular: Base Module

This module contains shared utilities and base classes for Step 12.
"""

from .imports import (
    validate_step12_imports,
    safe_import_manager,
    BLANK_TRAINING_LOOKBACK_DAYS,
    CONFIG,
    REQUIRED_MODULES,
    TORCH_AVAILABLE,
    pipeline_standards,
    dependency_status
)
from .utils import (
    ensure_directory,
    safe_json_dump,
    error,
    failed,
    timeout,
    warning,
    get_unified_data_loader
)
from .logger import setup_step12_logger

__all__ = [
    # Imports
    'validate_step12_imports',
    'safe_import_manager',
    'BLANK_TRAINING_LOOKBACK_DAYS',
    'CONFIG',
    'REQUIRED_MODULES',
    'TORCH_AVAILABLE',
    'pipeline_standards',
    'dependency_status',

    # Utils
    'ensure_directory',
    'safe_json_dump',
    'error',
    'failed',
    'timeout',
    'warning',
    'get_unified_data_loader',

    # Logger
    'setup_step12_logger'
]
