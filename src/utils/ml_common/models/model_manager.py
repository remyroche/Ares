"""
Model Management Utilities

Common model management patterns shared across all training modules.
Now implemented as a thin wrapper over the unified src.utils.model_manager.ModelManager.
"""

from typing import Any
from src.utils.model_manager import ModelManager as _UnifiedModelManager


class ModelManager(_UnifiedModelManager):
    """Thin wrapper to maintain backward-compatible constructor signature."""

    def __init__(self, save_path: str, save_format: str = "joblib") -> None:
        # Delegate to unified manager with training-style parameters
        super().__init__(config=None, save_path=save_path, save_format=save_format)