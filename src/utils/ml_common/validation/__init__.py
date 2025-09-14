"""
ML Common - Validation Module

This module contains all validation functionality including:
- Cross-validation utilities
- Model stability assessment
- Threshold optimization
- Validation metrics
"""

from .validation_utils import ValidationFramework, ConfigurationValidator
from .cv_utils import TemporalCrossValidator, PurgedKFold, CrossValidationUtilities
from .cv import PurgedSplitConfig
from .stability import feature_selection_stability, aggregate_time_blocks, StabilityAnalyzer
from .thresholding import optimize_threshold, calibrate_probabilities

__all__ = [
    # Validation Utils
    'ValidationFramework', 'ConfigurationValidator',

    # Cross-validation
    'TemporalCrossValidator', 'PurgedKFold', 'CrossValidationUtilities', 'PurgedSplitConfig',

    # Stability Analysis
    'feature_selection_stability', 'aggregate_time_blocks', 'StabilityAnalyzer',

    # Threshold Optimization
    'optimize_threshold', 'calibrate_probabilities'
]