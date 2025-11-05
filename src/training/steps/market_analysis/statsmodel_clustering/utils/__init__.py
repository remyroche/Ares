"""
Utils for Statsmodels Clustering

This module provides utility functions for statsmodels regime switching models,
including result conversion, validation, and diagnostics.

Key Components:
- ResultConverter: Convert between different result formats
- ModelValidator: Validate models and data
- ModelDiagnostics: Comprehensive model analysis
"""

from .result_converter import (
    ResultConverter,
    ConversionConfig,
    ConversionResult,
    convert_statsmodels_to_pyro,
    convert_pyro_to_statsmodels,
    create_unified_result,
    save_result_to_file
)

from .validation import (
    ModelValidator,
    ValidationConfig,
    ValidationResult,
    validate_input_data,
    validate_model_fit,
    cross_validate_regime_model
)

from .diagnostics import (
    ModelDiagnostics,
    DiagnosticsConfig,
    DiagnosticsResult,
    analyze_model_fit,
    analyze_regime_stability,
    create_diagnostics_report
)

__all__ = [
    # Result conversion
    'ResultConverter',
    'ConversionConfig',
    'ConversionResult',
    'convert_statsmodels_to_pyro',
    'convert_pyro_to_statsmodels',
    'create_unified_result',
    'save_result_to_file',
    
    # Validation
    'ModelValidator',
    'ValidationConfig',
    'ValidationResult',
    'validate_input_data',
    'validate_model_fit',
    'cross_validate_regime_model',
    
    # Diagnostics
    'ModelDiagnostics',
    'DiagnosticsConfig',
    'DiagnosticsResult',
    'analyze_model_fit',
    'analyze_regime_stability',
    'create_diagnostics_report'
]