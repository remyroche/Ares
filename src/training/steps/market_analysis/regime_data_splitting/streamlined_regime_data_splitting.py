"""
import warnings
Streamlined Regime Data Splitting Component

This module provides a consolidated, high-performance implementation of regime data splitting
that combines the best features from the previous implementations while using modern utility
modules for optimal performance and maintainability.

Key improvements:
- Single unified implementation
- Streaming data processing
- Memory-efficient operations
- Comprehensive data quality validation
- Hardware optimization integration
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import warnings

import pandas as pd
import numpy as np

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Import our standardized utilities
from src.utils.common_operations import (
    safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
    calculate_data_quality_metrics, safe_merge_dataframes, safe_groupby_operation,
    safe_apply_function, create_summary_statistics, safe_drop_columns,
    safe_rename_columns, validate_timestamp_column, safe_timestamp_conversion,
    get_dataframe_info, safe_filter_dataframe, create_data_quality_report,
    safe_to_parquet, safe_read_parquet, validate_dataframe_schema,
    optimize_dataframe_dtypes, safe_fillna, safe_float, safe_int,
    validate_finite, validate_positive, validate_range, safe_divide,
    safe_log, safe_sqrt, safe_power, safe_mean, safe_std, safe_percentage_change,
    safe_kelly_calculation, safe_weighted_average, validate_correlation_matrix,
    safe_matrix_inverse, math_safe, timed_operation, format_bytes,
    chunked_iterable, parallel_map, get_m1_gpu_manager, get_m1_memory_optimizer,
    get_m1_cpu_optimizer, cleanup_m1_optimizers, integrate_with_m1_optimizers,
    memory_checkpoint, gpu_context, optimize_memory, get_memory_usage
)

from src.utils.math_validation import (
    safe_divide as math_safe_divide, safe_log as math_safe_log,
    safe_sqrt as math_safe_sqrt, safe_power as math_safe_power,
    validate_finite as math_validate_finite, validate_positive as math_validate_positive,
    validate_range as math_validate_range, safe_kelly_calculation as math_safe_kelly,
    safe_weighted_average as math_safe_weighted_avg, safe_percentage_change as math_safe_pct_change,
    safe_correlation as math_safe_corr, safe_covariance as math_safe_cov,
    safe_mean as math_safe_mean, safe_std as math_safe_std,
    safe_percentile as math_safe_percentile, validate_correlation_matrix as math_validate_corr_matrix,
    safe_matrix_inverse as math_safe_matrix_inv, math_safe as math_safe_func,
    MathValidation, MathValidationError
)

from src.utils.data.quality.data_quality import DataQualityFramework
from src.utils.data.validation.validators import CrossStepValidator
from src.utils.hardware import (
    get_integrated_hardware_manager, 
    get_comprehensive_optimizer,
    memory_optimized, 
    comprehensive_memory_optimization,
    optimize_dataframe, 
    optimize_array,
    m1_optimized,
    WorkloadCategory,
    MemoryOptimizationLevel
)
()

            # Garbage collection
            import gc
            gc.collect()

            self.logger.debug("🧹 Periodic cleanup performed")

        except Exception as e:
            self.logger.exception(f"Periodic cleanup failed: {e}")

    async def _parallel_merge_chunks(self, chunks: List[pd.DataFrame]) -> pd.DataFrame:
        """Merge chunks using parallel processing."""
        try:
            self.logger.info(f"🔄 Parallel merging {len(chunks)} chunks")

            if self.cpu_optimizer:
                # Use parallel processing for merge
                merge_func = self.cpu_optimizer.parallel_map_m1(
                    lambda x: x,
                    chunks
                )

                # Merge results
                result_chunks = list(merge_func)
                result = safe_merge_dataframes(result_chunks)
            else:
                # Fallback to sequential merge
                result = safe_merge_dataframes(chunks)

            self.logger.info("✅ Parallel merge completed")
            return result

        except Exception as e:
            self.logger.exception(f"Parallel merge failed, falling back to sequential: {e}")
            return safe_merge_dataframes(chunks)

    def _validate_temporal_continuity(self, data: pd.DataFrame) -> List[str]:
        """Validate temporal continuity of the data."""
        issues = []

        try:
            timestamps = data['timestamp']
            time_diffs = timestamps.diff().dt.total_seconds()

            # Check for duplicate timestamps
            duplicate_timestamps = timestamps.duplicated().sum()
            if duplicate_timestamps > 0:
                issues.append(f"Found {duplicate_timestamps} duplicate timestamps")

            # Check for gaps larger than expected
            max_gap = time_diffs.max()
            if max_gap > 3600:  # More than 1 hour gap
                issues.append(f"Large time gap detected: {max_gap:.0f} seconds")

            # Check for backwards timestamps
            backwards_count = (time_diffs < 0).sum()
            if backwards_count > 0:
                issues.append(f"Found {backwards_count} backwards timestamps")

            # Check for irregular intervals
            if len(time_diffs) > 1:
                expected_interval = time_diffs.median()
                irregular_intervals = (time_diffs - expected_interval).abs() > (expected_interval * 0.5)
                irregular_count = irregular_intervals.sum()

                if irregular_count > len(data) * 0.1:  # More than 10% irregular
                    issues.append(f"Found {irregular_count} irregular time intervals")

        except Exception as e:
            issues.append(f"Temporal validation failed: {e}")

        return issues

    def _validate_data_completeness(self, data: pd.DataFrame) -> List[str]:
        """Validate data completeness."""
        issues = []

        try:
            total_rows = len(data)

            # Check for missing values in critical columns
            critical_columns = ['timestamp', 'composite_cluster_id']
            for col in critical_columns:
                if col in data.columns:
                    missing_count = data[col].isna().sum()
                    if missing_count > 0:
                        missing_pct = (missing_count / total_rows) * 100
                        issues.append(f"Missing {missing_count} values ({missing_pct:.1f}%) in {col}")

            # Check overall data completeness
            total_missing = data.isna().sum().sum()
            if total_missing > 0:
                missing_pct = (total_missing / (total_rows * len(data.columns))) * 100
                if missing_pct > 5:  # More than 5% missing
                    issues.append(f"High missing data rate: {missing_pct:.1f}%")

            # Check for empty data
            if total_rows == 0:
                issues.append("Dataset is empty")

            # Check for minimum required data points
            if total_rows < 100:
                issues.append(f"Insufficient data points: {total_rows} (minimum 100 required)")

        except Exception as e:
            issues.append(f"Completeness validation failed: {e}")

        return issues

    def _validate_data_consistency(self, data: pd.DataFrame) -> List[str]:
        """Validate data consistency."""
        issues = []

        try:
            # Check data type consistency
            for col in data.columns:
                dtype = data[col].dtype
                if dtype == 'object':
                    # Check if object column should be numeric
                    try:
                        pd.to_numeric(data[col], errors='coerce')
                        issues.append(f"Object column '{col}' may contain numeric data")
                    except:
                        pass

            # Check for mixed data types in numeric columns
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if data[col].dtype != 'int64' and data[col].dtype != 'float64':
                    issues.append(f"Unexpected numeric dtype in {col}: {data[col].dtype}")

            # Check for data range consistency
            if 'price_close' in data.columns:
                price_data = data['price_close']
                if price_data.min() <= 0:
                    issues.append("Negative or zero prices detected")

            if 'volume' in data.columns:
                volume_data = data['volume']
                if volume_data.min() < 0:
                    issues.append("Negative volume values detected")

        except Exception as e:
            issues.append(f"Consistency validation failed: {e}")

        return issues

    def _validate_regime_transitions(self, data: pd.DataFrame) -> List[str]:
        """Validate regime transitions."""
        issues = []

        try:
            if 'composite_cluster_id' not in data.columns:
                return issues

            regime_ids = data['composite_cluster_id']

            # Check for rapid regime changes
            regime_changes = regime_ids.diff().ne(0)
            change_count = regime_changes.sum()

            if change_count > 0:
                change_rate = change_count / len(data)

                # More than 10% regime changes
                if change_rate > 0.1:
                    issues.append(f"High regime change rate: {change_rate:.1%}")

                # Check for very frequent changes (more than once per hour)
                if 'timestamp' in data.columns:
                    time_span = (data['timestamp'].max() - data['timestamp'].min()).total_seconds() / 3600
                    if time_span > 0:
                        changes_per_hour = change_count / time_span
                        if changes_per_hour > 10:
                            issues.append(f"Excessive regime changes: {changes_per_hour:.1f} per hour")

            # Check for regime stability
            regime_sizes = regime_ids.value_counts()
            if len(regime_sizes) > 0:
                min_regime_size = regime_sizes.min()
                if min_regime_size < 10:
                    issues.append(f"Very small regimes detected: minimum size {min_regime_size}")

        except Exception as e:
            issues.append(f"Regime transition validation failed: {e}")

        return issues

    def cleanup_resources(self):
        """Clean up resources and optimizers."""
        try:
            if self.memory_optimizer:
                cleanup_m1_optimizers()

            # Clean up hardware managers
            if hasattr(self, 'hardware_manager'):
                self.hardware_manager.cleanup()

            self.logger.info("🧹 Resources cleaned up successfully")
        except Exception as e:
            self.logger.exception(f"Error during resource cleanup: {e}")

# Factory function for easy instantiation
def create_streamlined_regime_splitting(config: Optional[Dict[str, Any]] = None) -> StreamlinedRegimeDataSplitting:
    """Create a streamlined regime data splitting instance."""
    return StreamlinedRegimeDataSplitting(config)

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and
                VECTORBT_AVAILABLE)

    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str,
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

    def _pandas_rolling_operation(self, data: pd.Series, operation: str,
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    def _vectorbt_apply_operation(self, data: pd.Series, func,
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)

        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
