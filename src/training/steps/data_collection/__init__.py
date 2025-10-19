"""
Data Collection Steps Package

This package contains autonomous data collection steps that have been refactored
from the original sub-pipeline architecture to use the new BaseStep pattern.
"""

# Import the existing step (but we'll handle import errors gracefully)
try:
    from .klines_downloading_processing import KlinesDataProcessingPipeline
    KLINES_STEP_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import KlinesDataProcessingPipeline: {e}")
    KLINES_STEP_AVAILABLE = False

# Register steps with the global registry
from src.training.steps.base_step import step_registry

# Register DATA_COLLECTION steps if available
if KLINES_STEP_AVAILABLE:
    step_registry.register('data_download', KlinesDataProcessingPipeline)

__all__ = [
    'KlinesDataProcessingPipeline' if KLINES_STEP_AVAILABLE else None
]
