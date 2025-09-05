"""Step 5: Labeling with Simplified Architecture.

This module provides a simplified, well-structured labeling step that maintains
all functionality while dramatically reducing complexity through modular design.

Key Simplifications:
- Extracted monitoring systems into separate modules
- Extracted decorator system with fallback mechanisms  
- Extracted labeling components into focused classes
- Centralized dependency management
- Simplified main class focused on core functionality

The original complex implementation has been refactored into:
- monitoring/ - Function call monitoring, error handling, performance tracking, validation
- decorators.py - Centralized decorator system with fallbacks
- labeling_components.py - Core labeling logic components
- dependencies.py - Dependency management and validation
- step05_labeling_simplified.py - Simplified main implementation

This file now serves as a compatibility layer that imports from the simplified implementation.
"""

# Import the simplified implementation
from .step05_labeling_simplified import (
    LabelingStep,
    run_step,
)

# Re-export for compatibility
__all__ = [
    "LabelingStep",
    "run_step",
]