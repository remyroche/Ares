"""
Data Preprocessing for TAS

This module provides a wrapper around the unified data preprocessing system
for tree architecture search, maintaining backward compatibility.
"""

# Import the unified data preprocessing system
from src.utils.nas_tas.data_preprocessing import (
    UnifiedDataPreprocessor,
    PreprocessingConfig,
    PreprocessingResult,
    PreprocessingStep
)

# Backward compatibility aliases
DataPreprocessor = UnifiedDataPreprocessor