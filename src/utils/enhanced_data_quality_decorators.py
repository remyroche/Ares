"""
Enhanced Data Quality Decorators
Provides comprehensive data quality validation decorators for the training pipeline.
"""

import functools
from typing import Any, Optional

# Handle optional dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

from src.utils.logger import system_logger


class EnhancedDataQualityDecorators:
    """Enhanced Data Quality Decorators for Comprehensive Validation"""

    def __init__(self):
        self.logger = system_logger.getChild("EnhancedDataQualityDecorators")

    @staticmethod
    def extract_data_from_args(args: tuple, kwargs: dict) -> Optional[Any]:
        """Extract DataFrame from function arguments."""
        # Look for DataFrame in positional arguments
        for arg in args:
            if hasattr(arg, 'shape'):  # Check if it's DataFrame-like
                return arg

        # Look for DataFrame in keyword arguments
        for key, value in kwargs.items():
            if hasattr(value, 'shape'):  # Check if it's DataFrame-like
                return value

        return None

    @staticmethod
    def update_data_in_args_kwargs(modified_data: Any, args: tuple, kwargs: dict) -> tuple:
        """Update DataFrame in original args/kwargs with modified data."""
        # Update DataFrame in positional arguments
        new_args = list(args)
        found = False
        for i, arg in enumerate(new_args):
            if hasattr(arg, 'shape') and arg is not modified_data:
                # Check if this is the original DataFrame (by comparing shape and columns)
                if hasattr(modified_data, 'shape') and hasattr(arg, 'shape'):
                    if arg.shape[0] == modified_data.shape[0]:  # Same number of rows
                        new_args[i] = modified_data
                        found = True
                        break

        # Update DataFrame in keyword arguments if not found in args
        if not found:
            for key, value in kwargs.items():
                if hasattr(value, 'shape') and value is not modified_data:
                    # Check if this is the original DataFrame
                    if hasattr(modified_data, 'shape') and hasattr(value, 'shape'):
                        if value.shape[0] == modified_data.shape[0]:  # Same number of rows
                            kwargs[key] = modified_data
                            found = True
                            break

        return tuple(new_args), kwargs

    @staticmethod
    def validate_constant_features(func):
        """Decorator to detect and remove constant features."""
        @functools.wraps(func)

    @staticmethod
    def validate_low_variance_features(func):
        """Decorator to detect and remove low variance features."""
        @functools.wraps(func)

    @staticmethod
    def validate_data_completeness(func):
        """Decorator to validate data completeness and handle missing data."""
        @functools.wraps(func)

    @staticmethod
    def validate_datetime_index(func):
        """Decorator to validate and fix datetime index."""
        @functools.wraps(func)

    @staticmethod
    def validate_multi_timeframe_alignment(func):
        """Decorator to validate multi-timeframe data alignment."""
        @functools.wraps(func)

    @staticmethod
    def validate_hmm_data_requirements(func):
        """Decorator to validate HMM data requirements."""
        @functools.wraps(func)

    @staticmethod
    def validate_data_structure(func):
        """Decorator to validate data structure and completeness."""
        @functools.wraps(func)

    @staticmethod
    def optimize_memory_usage(func):
        """Decorator to optimize memory usage of DataFrames."""
        @functools.wraps(func)

    @staticmethod
    def comprehensive_data_validation(func):
        """Comprehensive data validation decorator combining multiple checks."""
        @functools.wraps(func)

    @staticmethod
    def validate_memory_optimized_data_quality(func):
        """Memory-optimized validation decorator."""
        @functools.wraps(func)

    @staticmethod
    def validate_feature_engineering_pipeline(func):
        """Specialized decorator for feature engineering pipeline validation."""
        @functools.wraps(func)

    @staticmethod
    def validate_hmm_regime_discovery(func):
        """Specialized decorator for HMM regime discovery validation."""
        @functools.wraps(func)

    @staticmethod
    def validate_multi_timeframe_processing(func):
        """Specialized decorator for multi-timeframe processing validation."""
        @functools.wraps(func)


# Create standalone decorator functions for easier import
validate_constant_features = EnhancedDataQualityDecorators.validate_constant_features
validate_low_variance_features = EnhancedDataQualityDecorators.validate_low_variance_features
validate_data_completeness = EnhancedDataQualityDecorators.validate_data_completeness
validate_datetime_index = EnhancedDataQualityDecorators.validate_datetime_index
validate_multi_timeframe_alignment = EnhancedDataQualityDecorators.validate_multi_timeframe_alignment
validate_hmm_data_requirements = EnhancedDataQualityDecorators.validate_hmm_data_requirements
validate_data_structure = EnhancedDataQualityDecorators.validate_data_structure
optimize_memory_usage = EnhancedDataQualityDecorators.optimize_memory_usage
comprehensive_data_validation = EnhancedDataQualityDecorators.comprehensive_data_validation
validate_memory_optimized_data_quality = EnhancedDataQualityDecorators.validate_memory_optimized_data_quality
validate_feature_engineering_pipeline = EnhancedDataQualityDecorators.validate_feature_engineering_pipeline
validate_hmm_regime_discovery = EnhancedDataQualityDecorators.validate_hmm_regime_discovery
validate_multi_timeframe_processing = EnhancedDataQualityDecorators.validate_multi_timeframe_processing

__all__ = [
    "EnhancedDataQualityDecorators",
    "validate_constant_features",
    "validate_low_variance_features",
    "validate_data_completeness",
    "validate_datetime_index",
    "validate_multi_timeframe_alignment",
    "validate_hmm_data_requirements",
    "validate_data_structure",
    "optimize_memory_usage",
    "comprehensive_data_validation",
    "validate_memory_optimized_data_quality",
    "validate_feature_engineering_pipeline",
    "validate_hmm_regime_discovery",
    "validate_multi_timeframe_processing",
]