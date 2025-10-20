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

try:
    from .data_collection_orchestrator import DataCollectionOrchestrator
    ORCHESTRATOR_STEP_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import DataCollectionOrchestrator: {e}")
    ORCHESTRATOR_STEP_AVAILABLE = False

try:
    from .data_preparation.step01_data_collection import DataCollectionStep
    STEP01_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import DataCollectionStep: {e}")
    STEP01_AVAILABLE = False

try:
    from .data_preparation.step02_data_reading import DataReadingStep
    STEP02_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import DataReadingStep: {e}")
    STEP02_AVAILABLE = False

# Register steps with the global registry
from src.training.steps.base_step import step_registry

# Register DATA_COLLECTION steps if available
if KLINES_STEP_AVAILABLE:
    step_registry.register('data_download', KlinesDataProcessingPipeline)

if ORCHESTRATOR_STEP_AVAILABLE:
    step_registry.register('data_collection_orchestrator', DataCollectionOrchestrator)

if STEP01_AVAILABLE:
    step_registry.register('data_collection_step01', DataCollectionStep)

if STEP02_AVAILABLE:
    step_registry.register('data_reading_step02', DataReadingStep)

__all__ = [
    'KlinesDataProcessingPipeline' if KLINES_STEP_AVAILABLE else None
]
