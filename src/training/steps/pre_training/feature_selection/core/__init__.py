"""
Core feature selection modules.

This package contains the core functionality for feature selection including
pipeline logic, selection algorithms, and optimization strategies.
"""

from .config import (
    BaseFeatureSelectionConfig,
    ModelSpecificConfig,
    QualityThresholdsConfig,
    ValidationConfig,
    AdvancedSelectionConfig,
    FeatureSelectionConfig,
    FeatureSelectionResult
)

from .selector import FeatureSelector
from .optimizer import FeatureSelectionOptimizer

__all__ = [
    # Configuration classes
    'BaseFeatureSelectionConfig',
    'ModelSpecificConfig', 
    'QualityThresholdsConfig',
    'ValidationConfig',
    'AdvancedSelectionConfig',
    'FeatureSelectionConfig',
    'FeatureSelectionResult',
    
    # Core classes
    'FeatureSelector',
    'FeatureSelectionOptimizer'
]