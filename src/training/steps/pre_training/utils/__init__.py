"""
Pre-training utilities package.

This package provides common utilities for all pre-training steps,
including enhanced validation, logging, and data processing utilities.
"""

from .validation_utils import (
    PreTrainingValidator, ValidationConfig, ValidationContext,
    validate_feature_generation_inputs, validate_feature_selection_inputs,
    validate_cross_validation_inputs, validate_label_generation_inputs,
    validate_inputs
)

# Re-export validation utilities from the enhanced fast_failing_validation
from ..feature_lookback_optimization.utils.fast_failing_validation import (
    FastFailingValidator, ValidationResult, ValidationSeverity,
    validate_dataframe_basic, validate_feature_data, validate_target_data,
    validate_preprocessing_inputs, validate_model_inputs,
    validate_optimization_inputs_fast_fail, validate_feature_calculation_inputs
)

__all__ = [
    # Main validator classes
    'PreTrainingValidator', 'ValidationConfig', 'ValidationContext',
    
    # Convenience validation functions
    'validate_feature_generation_inputs', 'validate_feature_selection_inputs',
    'validate_cross_validation_inputs', 'validate_label_generation_inputs',
    'validate_inputs',
    
    # Core validation utilities
    'FastFailingValidator', 'ValidationResult', 'ValidationSeverity',
    'validate_dataframe_basic', 'validate_feature_data', 'validate_target_data',
    'validate_preprocessing_inputs', 'validate_model_inputs',
    'validate_optimization_inputs_fast_fail', 'validate_feature_calculation_inputs'
]