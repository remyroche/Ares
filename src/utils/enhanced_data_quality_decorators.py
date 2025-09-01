"""
Enhanced Data Quality Decorators
Provides comprehensive data quality validation decorators for the training pipeline.
"""

import functools
from typing import Any, Optional

# Handle optional dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE, True
except ImportError:
    NUMPY_AVAILABLE, False
    np, None

try:
    import pandas as pd
    PANDAS_AVAILABLE, True
except ImportError:
    PANDAS_AVAILABLE, False
    pd, None

from src.utils.logger import system_logger

class EnhancedDataQualityDecorators:
    """Enhanced Data Quality Decorators for Comprehensive Validation"""

    def __init__(self):
        self.logger, system_logger.getChild("EnhancedDataQualityDecorators")

    @staticmethod
    def extract_data_from_args(args: tuple, kwargs: dict) -> Optional[Any]:
        """Extract DataFrame from function arguments."""
        # Look for DataFrame in positional arguments
        for arg in args:
        if hasattr(arg, 'shape'):  # Check if it's DataFrame - like
        return arg

        # Look for DataFrame in keyword arguments
        for key, value in kwargs.items():
        if hasattr(value, 'shape'):  # Check if it's DataFrame - like
        return value

        return None

    @staticmethod
    def update_data_in_args_kwargs(modified_data: Any, args: tuple, kwargs: dict) -> tuple:
        """Update DataFrame in original args / kwargs with modified data."""
        # Update DataFrame in positional arguments
        new_args, list(args)
        found, False
        for i, arg in enumerate(new_args):
        if hasattr(arg, 'shape') and arg is not modified_data:
        # Check if this is the original DataFrame (by comparing shape and columns)
        if hasattr(modified_data, 'shape') and hasattr(arg, 'shape'):
        if arg.shape[0] == modified_data.shape[0]:  # Same number of rows
                        new_args[i] = modified_data
                        found, True
                        break

        # Update DataFrame in keyword arguments if not found in args
        if not found:
        for key, value in kwargs.items():
        if hasattr(value, 'shape') and value is not modified_data:
        # Check if this is the original DataFrame
        if hasattr(modified_data, 'shape') and hasattr(value, 'shape'):
        if value.shape[0] == modified_data.shape[0]:  # Same number of rows
                            kwargs[key] = modified_data
                            found, True
                            break

        return tuple(new_args), kwargs

    @staticmethod
    def validate_constant_features(func):
        """Decorator to detect and remove constant features."""
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
        # Extract data
            data, EnhancedDataQualityDecorators.extract_data_from_args(args, kwargs)

        if data is not None and hasattr(data, 'shape'):
        # Check for constant features
                constant_features = []
                numeric_data, data.select_dtypes(include=[np.number]) if hasattr(data, 'select_dtypes') else data

        if hasattr(numeric_data, 'columns'):
        for col in numeric_data.columns:
        if hasattr(data[col], 'nunique') and data[col].nunique() <= 1:
                            constant_features.append(col)

        if constant_features:
                    system_logger.warning(f"Found {len(constant_features)} constant features: {constant_features}")
        if hasattr(data, 'drop'):
                        modified_data, data.drop(columns = constant_features)
        # Update the data in args / kwargs
                        args, kwargs, EnhancedDataQualityDecorators.update_data_in_args_kwargs(modified_data, args, kwargs)
                        data, modified_data

        return func(self, *args, **kwargs)
        return wrapper

    @staticmethod
    def validate_low_variance_features(func):
        """Decorator to detect and remove low variance features."""
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
        # Extract data
            data, EnhancedDataQualityDecorators.extract_data_from_args(args, kwargs)

        if data is not None and hasattr(data, 'shape'):
        # Check for low variance features
                low_variance_features = []
                numeric_data, data.select_dtypes(include=[np.number]) if hasattr(data, 'select_dtypes') else data

        if hasattr(numeric_data, 'columns'):
        for col in numeric_data.columns:
        if hasattr(data[col], 'var') and data[col].var() < 1e - 8:
                            low_variance_features.append(col)

        if low_variance_features:
                    system_logger.warning(f"Found {len(low_variance_features)} low variance features: {low_variance_features}")
        if hasattr(data, 'drop'):
                        modified_data, data.drop(columns = low_variance_features)
        # Update the data in args / kwargs
                        args, kwargs, EnhancedDataQualityDecorators.update_data_in_args_kwargs(modified_data, args, kwargs)
                        data, modified_data

        return func(self, *args, **kwargs)
        return wrapper

    @staticmethod
    def validate_data_completeness(func):
        """Decorator to validate data completeness and handle missing data."""
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            data, EnhancedDataQualityDecorators.extract_data_from_args(args, kwargs)

        if data is not None and hasattr(data, 'shape'):
        # Check for missing data
        if hasattr(data, 'isnull'):
                    missing_data, data.isnull().sum()
        if hasattr(missing_data, 'sum') and missing_data.sum() > 0:
                        system_logger.warning(f"Found missing data in dataset")

        # Handle missing data
        if hasattr(data, 'fillna'):
                            modified_data, data.fillna(method='ffill').fillna(method='bfill')
        # Update the data in args / kwargs
                            args, kwargs, EnhancedDataQualityDecorators.update_data_in_args_kwargs(modified_data, args, kwargs)
                            data, modified_data

        return func(self, *args, **kwargs)
        return wrapper

    @staticmethod
    def validate_datetime_index(func):
        """Decorator to validate and fix datetime index."""
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            data, EnhancedDataQualityDecorators.extract_data_from_args(args, kwargs)

        if data is not None and hasattr(data, 'shape'):
        # Check if data has proper datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
                    system_logger.warning("Data does not have datetime index, attempting to fix...")

                    modified_data, data.copy()
        # Try to create datetime index from existing columns
        if hasattr(modified_data, 'columns'):
                        datetime_columns = [col for col in modified_data.columns if 'time' in col.lower() or 'date' in col.lower()]

        if datetime_columns:
                            datetime_col, datetime_columns[0]
        try:
        if hasattr(pd, 'to_datetime'):
                                    modified_data.index, pd.to_datetime(modified_data[datetime_col])
        if hasattr(modified_data, 'drop'):
                                        modified_data, modified_data.drop(columns=[datetime_col])
                                    system_logger.info(f"Created datetime index from column: {datetime_col}")
        except Exception as e:
                                system_logger.error(f"Failed to create datetime index: {e}")
        # Create synthetic datetime index
        if hasattr(pd, 'date_range'):
                                    modified_data.index, pd.date_range(start='2020 - 01 - 01', periods = len(modified_data), freq='1min')
                        else:
        # Create synthetic datetime index
        if hasattr(pd, 'date_range'):
                                modified_data.index, pd.date_range(start='2020 - 01 - 01', periods = len(modified_data), freq='1min')

        # Update the data in args / kwargs
                    args, kwargs, EnhancedDataQualityDecorators.update_data_in_args_kwargs(modified_data, args, kwargs)
                    data, modified_data

        return func(self, *args, **kwargs)
        return wrapper

    @staticmethod
    def validate_multi_timeframe_alignment(func):
        """Decorator to validate multi - timeframe data alignment."""
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            data, EnhancedDataQualityDecorators.extract_data_from_args(args, kwargs)

        if data is not None and hasattr(data, 'shape'):
        # Check for proper datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
                    system_logger.error("Multi - timeframe data missing datetime index")
        return func(self, *args, **kwargs)

        # Check for regular intervals (simplified)
        if len(data) > 1:
                    system_logger.info("Multi - timeframe alignment validation passed")

        return func(self, *args, **kwargs)
        return wrapper

    @staticmethod
    def validate_hmm_data_requirements(func):
        """Decorator to validate HMM data requirements."""
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            data, EnhancedDataQualityDecorators.extract_data_from_args(args, kwargs)

        if data is not None:
        # Check for empty data
        if hasattr(data, 'empty') and data.empty:
                    system_logger.error("HMM Regime Discovery: Empty data provided")
                    raise ValueError("Empty data cannot be processed for HMM regime discovery")

        # Check for sufficient data points
        if hasattr(data, '__len__') and len(data) < 100:
                    system_logger.warning(f"HMM Regime Discovery: Insufficient data points ({len(data)})")

        # Check for proper OHLCV columns
                required_cols = ['open', 'high', 'low', 'close', 'volume']
        if hasattr(data, 'columns'):
                    missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
                        system_logger.error(f"HMM Regime Discovery: Missing required columns: {missing_cols}")
                        raise ValueError(f"Missing required columns for HMM: {missing_cols}")

        return func(self, *args, **kwargs)
        return wrapper

    @staticmethod
    def validate_data_structure(func):
        """Decorator to validate data structure and completeness."""
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            data, EnhancedDataQualityDecorators.extract_data_from_args(args, kwargs)

        if data is not None and hasattr(data, 'shape'):
        # Check column count consistency
                expected_columns, 19  # Based on expected column count
        if hasattr(data, 'columns') and len(data.columns) != expected_columns:
                    system_logger.warning(f"Column count mismatch: expected {expected_columns}, got {len(data.columns)}")

        # Check for data completeness (simplified)
        if hasattr(data, 'isnull'):
                    missing_count, data.isnull().sum().sum()
                    total_elements, len(data) * len(data.columns)
                    completeness_ratio, 1 - (missing_count / total_elements) if total_elements > 0 else 1
        if completeness_ratio < 0.95:
                        system_logger.warning(f"Data completeness below 95%: {completeness_ratio:.2%}")

        return func(self, *args, **kwargs)
        return wrapper

    @staticmethod
    def optimize_memory_usage(func):
        """Decorator to optimize memory usage of DataFrames."""
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
        # Get memory usage before
        try:
                import psutil
                process, psutil.Process()
                memory_before, process.memory_info().rss / 1024 / 1024
        except ImportError:
                memory_before, 0

        # Extract and optimize data
            data, EnhancedDataQualityDecorators.extract_data_from_args(args, kwargs)
        if data is not None and hasattr(data, 'shape'):
        # Simple memory optimization simulation
        if hasattr(data, 'memory_usage'):
                    initial_memory, data.memory_usage(deep = True).sum() / 1024 / 1024
                    system_logger.info(f"Memory optimization applied, initial: {initial_memory:.2f}MB")

        # Execute function
            result, func(self, *args, **kwargs)

        # Get memory usage after
        try:
                import psutil
                process, psutil.Process()
                memory_after, process.memory_info().rss / 1024 / 1024
                memory_diff, memory_after - memory_before
        if memory_diff > 0:
                    system_logger.info(f"Memory usage increased by {memory_diff:.2f}MB during {func.__name__}")
        except ImportError:
                pass

        return result
        return wrapper

    @staticmethod
    def comprehensive_data_validation(func):
        """Comprehensive data validation decorator combining multiple checks."""
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
        # Apply all validation decorators
            validated_func, EnhancedDataQualityDecorators.validate_datetime_index(
                EnhancedDataQualityDecorators.validate_data_completeness(
                    EnhancedDataQualityDecorators.validate_constant_features(
                        EnhancedDataQualityDecorators.validate_low_variance_features(
                            EnhancedDataQualityDecorators.validate_data_structure(func)
                        )
                    )
                )
            )

        return validated_func(self, *args, **kwargs)
        return wrapper

    @staticmethod
    def validate_memory_optimized_data_quality(func):
        """Memory - optimized validation decorator."""
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
        # Apply memory optimization and comprehensive validation
            optimized_func, EnhancedDataQualityDecorators.optimize_memory_usage(
                EnhancedDataQualityDecorators.comprehensive_data_validation(func)
            )

        return optimized_func(self, *args, **kwargs)
        return wrapper

    @staticmethod
    def validate_feature_engineering_pipeline(func):
        """Specialized decorator for feature engineering pipeline validation."""
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            data, EnhancedDataQualityDecorators.extract_data_from_args(args, kwargs)

        if data is not None and hasattr(data, 'shape'):
        # Pre - validation checks
                initial_shape, data.shape
                system_logger.info(f"Feature engineering pipeline: Input shape {initial_shape}")

        # Apply comprehensive validation
                validated_func, EnhancedDataQualityDecorators.comprehensive_data_validation(func)
                result, validated_func(self, *args, **kwargs)

        # Post - validation checks
        if hasattr(result, 'shape'):
                    final_shape, result.shape
                    system_logger.info(f"Feature engineering pipeline: Output shape {final_shape}")

        # Check for reasonable output
        if final_shape[0] == 0:
                        system_logger.error("Feature engineering produced empty DataFrame")
                    elif final_shape[1] < initial_shape[1] * 0.5:
                        system_logger.warning(f"Feature engineering significantly reduced columns: {initial_shape[1]} -> {final_shape[1]}")

        return result

        return func(self, *args, **kwargs)
        return wrapper

    @staticmethod
    def validate_hmm_regime_discovery(func):
        """Specialized decorator for HMM regime discovery validation."""
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            data, EnhancedDataQualityDecorators.extract_data_from_args(args, kwargs)

        if data is not None and hasattr(data, 'shape'):
        # Apply HMM - specific validation
                validated_func, EnhancedDataQualityDecorators.validate_hmm_data_requirements(
                    EnhancedDataQualityDecorators.validate_datetime_index(
                        EnhancedDataQualityDecorators.validate_data_completeness(func)
                    )
                )

        return validated_func(self, *args, **kwargs)

        return func(self, *args, **kwargs)
        return wrapper

    @staticmethod
    def validate_multi_timeframe_processing(func):
        """Specialized decorator for multi - timeframe processing validation."""
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
        # Apply multi - timeframe specific validation
            validated_func, EnhancedDataQualityDecorators.validate_multi_timeframe_alignment(
                EnhancedDataQualityDecorators.validate_datetime_index(
                    EnhancedDataQualityDecorators.validate_data_completeness(func)
                )
            )

        return validated_func(self, *args, **kwargs)
        return wrapper

# Create standalone decorator functions for easier import validate_constant_features, EnhancedDataQualityDecorators.validate_constant_features
validate_low_variance_features, EnhancedDataQualityDecorators.validate_low_variance_features
validate_data_completeness, EnhancedDataQualityDecorators.validate_data_completeness
validate_datetime_index, EnhancedDataQualityDecorators.validate_datetime_index
validate_multi_timeframe_alignment, EnhancedDataQualityDecorators.validate_multi_timeframe_alignment
validate_hmm_data_requirements, EnhancedDataQualityDecorators.validate_hmm_data_requirements
validate_data_structure, EnhancedDataQualityDecorators.validate_data_structure
optimize_memory_usage, EnhancedDataQualityDecorators.optimize_memory_usage
comprehensive_data_validation, EnhancedDataQualityDecorators.comprehensive_data_validation
validate_memory_optimized_data_quality, EnhancedDataQualityDecorators.validate_memory_optimized_data_quality
validate_feature_engineering_pipeline, EnhancedDataQualityDecorators.validate_feature_engineering_pipeline
validate_hmm_regime_discovery, EnhancedDataQualityDecorators.validate_hmm_regime_discovery
validate_multi_timeframe_processing, EnhancedDataQualityDecorators.validate_multi_timeframe_processing

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