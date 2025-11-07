"""
Data Collection Steps Package

This package contains autonomous data collection steps that have been refactored
from the original sub-pipeline architecture to use the new BaseStep pattern.
"""

# Defer import to avoid circular dependency with BaseStep
# Import happens when this module is accessed, not when it's first loaded
KlinesDataProcessingPipeline = None
KLINES_STEP_AVAILABLE = False

def _lazy_import_klines():
    """Lazy import of KlinesDataProcessingPipeline to avoid circular dependency."""
    global KlinesDataProcessingPipeline, KLINES_STEP_AVAILABLE

    if KlinesDataProcessingPipeline is not None:
        return  # Already imported

    try:
        from .klines_downloading_processing import KlinesDataProcessingPipeline as _KlinesClass
        KlinesDataProcessingPipeline = _KlinesClass
        KLINES_STEP_AVAILABLE = True
    except ImportError as e:
        # Silently fail - this is expected if dependencies are missing
        KLINES_STEP_AVAILABLE = False

# Register steps with the global registry (lazy import to avoid circular dependency)
def _register_steps():
    """Register steps with the global registry (called after all modules are loaded)."""
    # Lazy import to ensure BaseStep is fully loaded
    _lazy_import_klines()

    if not KLINES_STEP_AVAILABLE:
        return

    try:
        # Use lazy import to avoid circular dependency
        from src.training.steps.base_step import step_registry
        step_registry.register('data_download', KlinesDataProcessingPipeline)
    except (ImportError, AttributeError) as e:
        # Silently fail if registration is not possible
        pass

# Build __all__ list - will be populated after lazy import
def __getattr__(name):
    """Support lazy loading of KlinesDataProcessingPipeline."""
    if name == 'KlinesDataProcessingPipeline':
        _lazy_import_klines()
        return KlinesDataProcessingPipeline
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ['KlinesDataProcessingPipeline']

# Register steps after module initialization completes
import atexit
atexit.register(_register_steps)
