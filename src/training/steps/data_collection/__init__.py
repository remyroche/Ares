"""
Data Collection Steps Package

This package contains autonomous data collection steps that have been refactored
from the original sub-pipeline architecture to use the new BaseStep pattern.
"""

# Import the existing step (but we'll handle import errors gracefully)
print("🔍 DEBUG: Starting import of KlinesDataProcessingPipeline...")
try:
    from .klines_downloading_processing import KlinesDataProcessingPipeline
    print("✅ DEBUG: Successfully imported KlinesDataProcessingPipeline")
    KLINES_STEP_AVAILABLE = True
except ImportError as e:
    print(f"❌ DEBUG: Could not import KlinesDataProcessingPipeline: {e}")
    KLINES_STEP_AVAILABLE = False

# Register steps with the global registry (lazy import to avoid circular dependency)
# Move registration to a separate function to avoid circular import during module loading
def _register_steps():
    """Register steps with the global registry (called after all modules are loaded)."""
    if not KLINES_STEP_AVAILABLE:
        return

    print("🔍 DEBUG: Attempting to register with step registry...")
    try:
        # Use lazy import to avoid circular dependency
        import sys
        import importlib
        base_step_module = importlib.import_module('src.training.steps.base_step')
        step_registry = getattr(base_step_module, 'step_registry', None)
        if step_registry:
            step_registry.register('data_download', KlinesDataProcessingPipeline)
            print("✅ DEBUG: Successfully registered KlinesDataProcessingPipeline")
        else:
            print("❌ DEBUG: step_registry not found in base_step module")
    except ImportError as e:
        print(f"❌ DEBUG: Could not register KlinesDataProcessingPipeline with step registry: {e}")

# Build __all__ list dynamically
__all__ = []
if KLINES_STEP_AVAILABLE:
    __all__.append('KlinesDataProcessingPipeline')

# Register steps after all modules are loaded to avoid circular import
import atexit
atexit.register(_register_steps)
