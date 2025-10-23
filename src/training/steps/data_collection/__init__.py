"""
Data Collection Steps Package

This package contains autonomous data collection steps that have been refactored
from the original sub-pipeline architecture to use the new BaseStep pattern.
"""

# Import the existing steps (but we'll handle import errors gracefully)
try:
    from .klines_downloading_processing import KlinesDataProcessingPipeline
    KLINES_STEP_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import KlinesDataProcessingPipeline: {e}")
    KLINES_STEP_AVAILABLE = False


# Register steps with the global registry (lazy import to avoid circular imports)
def register_data_collection_steps():
    """Register data collection steps with the global registry."""
    try:
        from src.training.steps.base_step import step_registry
        
        # Register DATA_COLLECTION steps if available
        if KLINES_STEP_AVAILABLE:
            step_registry.register('data_download', KlinesDataProcessingPipeline)

    except ImportError as e:
        print(f"Warning: Could not register data collection steps: {e}")

__all__ = [
    'KlinesDataProcessingPipeline' if KLINES_STEP_AVAILABLE else None
]
