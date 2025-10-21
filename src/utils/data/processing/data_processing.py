"""
Unified Data Processing Utilities

This module consolidates all data processing, cleaning, and optimization functionality
from multiple previous modules into a single, comprehensive framework.

Consolidated from:
- cleaners.py
- optimizers.py
- data_quality_fixer.py
"""

import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

from src.utils.logger import system_logger
from src.utils.tprint import tprint_data_preview, tprint_data_format

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

class DataProcessor:
    """Unified data processing utilities for cleaning, optimization, and transformation."""

    def __init__(self) -> None:
        """Initialize data processor."""
        self.logger = system_logger.getChild('DataProcessor')

    def regularize_timestamps(
        self,
        data: pd.DataFrame,
        expected_interval: Optional[timedelta] = None,
        tolerance_seconds: int = 30,
        method: str = "forward_fill",
    ) -> pd.DataFrame:
        """
        Regularize timestamps in a DataFrame to ensure consistent intervals.

        Args:
            data: DataFrame with timestamp index or timestamp column
            expected_interval: Expected time interval (if None, will be inferred)
            tolerance_seconds: Tolerance for irregular intervals in seconds
            method: Method for handling missing values ('forward_fill', 'interpolate', 'drop')

        Returns:
            DataFrame with regularized timestamps
        """
        try:
            if data is None or len(data) == 0:
                return data

            # Make a copy to avoid modifying original data
            processed_data = data.copy()

            # Ensure timestamp is the index
            if "timestamp" in processed_data.columns:
                processed_data = processed_data.set_index("timestamp")
            elif not isinstance(processed_data.index, pd.DatetimeIndex):
                self.logger.warning("⚠️ No timestamp column found, cannot regularize intervals")
                return data

            # Sort by timestamp
            processed_data = processed_data.sort_index()

            # Check for irregular intervals
            time_diffs = processed_data.index.to_series().diff().dropna()
            if len(time_diffs) == 0:
                return data

            # Calculate expected interval if not provided
            if expected_interval is None:
                expected_interval = (
                    time_diffs.mode().iloc[0]
                    if len(time_diffs.mode()) > 0
                    else time_diffs.median()
                )

            # Identify irregular intervals
            irregular_mask = abs(time_diffs - expected_interval) > timedelta(seconds=tolerance_seconds)
            irregular_ratio = irregular_mask.sum() / len(time_diffs)

            if irregular_ratio > 0.0001:  # If more than 0.01% irregular intervals
                self.logger.info(f"🔄 Regularizing timestamps (irregular ratio: {irregular_ratio:.3f})")
                tprint_data_preview(processed_data, "before_timestamp_regularization")
                tprint_data_format(processed_data, "before_timestamp_regularization_format", level="DEBUG")

                # Create a regular timestamp index
                start_time = processed_data.index.min()
                end_time = processed_data.index.max()

                # Determine the frequency string based on expected interval
                freq = self._get_frequency_string(expected_interval)

                # Create regular timestamp index
                regular_index = pd.date_range(start=start_time, end=end_time, freq=freq)

                # Reindex data to regular intervals
                if method == "forward_fill":
                    processed_data = processed_data.reindex(regular_index, method="ffill")
                elif method == "interpolate":
                    # CRITICAL GAPS: Never interpolate data that contains critical gaps
                    # Check for large time gaps that indicate critical data issues
                    max_gap_seconds = time_diffs.max().total_seconds() if len(time_diffs) > 0 else 0

                    # If there are gaps > 30 minutes (1800 seconds), this is likely critical data
                    if max_gap_seconds > 1800:
                        self.logger.error('🚨 CRITICAL DATA GAPS DETECTED - REFUSING TO INTERPOLATE')
                        self.logger.error(f'🚨 Maximum gap: {max_gap_seconds} seconds - CRITICAL GAPS MUST NOT BE INTERPOLATED')
                        self.logger.warning('⚠️ Switching to forward_fill to preserve data integrity')
                        processed_data = processed_data.reindex(regular_index, method="ffill")
                    else:
                        processed_data = processed_data.reindex(regular_index).interpolate(method="time")
                elif method == "drop":
                    processed_data = processed_data.reindex(regular_index)
                else:
                    processed_data = processed_data.reindex(regular_index, method="ffill")

                # Drop rows that are completely NaN (before the first valid data point)
                processed_data = processed_data.dropna(how="all")

                self.logger.info(f"✅ Regularized timestamps: {len(processed_data)} rows with {freq} intervals")
                tprint_data_preview(processed_data, "after_timestamp_regularization")
                tprint_data_format(processed_data, "after_timestamp_regularization_format", level="DEBUG")

            return processed_data

        except Exception as e:
            self.logger.exception(f"🚨 Error regularizing timestamps: {e}")
            return data

    def _get_frequency_string(self, interval: timedelta) -> str:
        """Convert timedelta to pandas frequency string."""
        total_seconds = interval.total_seconds()

        if total_seconds <= 60:
            return "1T"  # 1 minute
        if total_seconds <= 300:
            return "5T"  # 5 minutes
        if total_seconds <= 900:
            return "15T"  # 15 minutes
        if total_seconds <= 3600:
            return "1H"  # 1 hour
        if total_seconds <= 14400:
            return "4H"  # 4 hours
        return "1D"  # 1 day

    def preprocess_data_for_multi_timeframe(
        self,
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame] = None,
        order_flow_data: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Preprocess data for multi-timeframe feature engineering.

        Args:
            price_data: Price data DataFrame
            volume_data: Volume data DataFrame (optional)
            order_flow_data: Order flow data DataFrame (optional)

        Returns:
            Tuple of preprocessed DataFrames
        """
        try:
            # Regularize timestamps for all data
            processed_price = self.regularize_timestamps(price_data)
            processed_volume = (
                self.regularize_timestamps(volume_data) if volume_data is not None else None
            )
            processed_order_flow = (
                self.regularize_timestamps(order_flow_data)
                if order_flow_data is not None
                else None
            )

            self.logger.info("✅ Data preprocessed for multi-timeframe feature engineering")
            return processed_price, processed_volume, processed_order_flow

        except Exception as e:
            self.logger.exception(f"🚨 Error preprocessing data for multi-timeframe: {e}")
            return price_data, volume_data, order_flow_data

    def validate_and_fix_data_quality(
        self,
        data: pd.DataFrame,
        data_type: str = "klines_ohlcv",
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validate and fix common data quality issues.

        Args:
            data: DataFrame to validate and fix
            data_type: Type of data ('klines_ohlcv', 'aggtrades', etc.)

        Returns:
            Tuple of (fixed_data, validation_results)
        """
        validation_results = {
            "original_shape": data.shape,
            "issues_fixed": [],
            "warnings": [],
            "errors": [],
        }

        try:
            fixed_data = data.copy()
            tprint_data_preview(fixed_data, f"before_quality_validation_{data_type}")
            tprint_data_format(fixed_data, f"before_quality_validation_format_{data_type}", level="DEBUG")

            # Fix common issues based on data type
            if data_type == "klines_ohlcv":
                fixed_data, issues = self._fix_ohlcv_issues(fixed_data)
                validation_results["issues_fixed"].extend(issues)

            # Regularize timestamps
            fixed_data = self.regularize_timestamps(fixed_data)

            validation_results["final_shape"] = fixed_data.shape
            self.logger.info(
                f"✅ Data quality validation completed: {len(validation_results['issues_fixed'])} issues fixed"
            )
            tprint_data_preview(fixed_data, f"after_quality_validation_{data_type}")
            tprint_data_format(fixed_data, f"after_quality_validation_format_{data_type}", level="DEBUG")

            return fixed_data, validation_results

        except Exception as e:
            self.logger.exception(f"🚨 Error in data quality validation: {e}")
            validation_results["errors"].append(str(e))
            return data, validation_results

    def _fix_ohlcv_issues(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Fix common OHLCV data issues."""
        issues = []

        # Fix negative prices
        for col in ["open", "high", "low", "close"]:
            if col in data.columns:
                negative_mask = data[col] < 0
                if negative_mask.any():
                    data.loc[negative_mask, col] = data.loc[negative_mask, col].abs()
                    issues.append(f"Fixed {negative_mask.sum()} negative {col} values")

        # Fix OHLC consistency
        if all(col in data.columns for col in ["open", "high", "low", "close"]):
            # High should be >= max of open, close
            high_violations = data["high"] < data[["open", "close"]].max(axis=1)
            if high_violations.any():
                data.loc[high_violations, "high"] = data.loc[
                    high_violations, ["open", "close"]
                ].max(axis=1)
                issues.append(f"Fixed {high_violations.sum()} high price violations")

            # Low should be <= min of open, close
            low_violations = data["low"] > data[["open", "close"]].min(axis=1)
            if low_violations.any():
                data.loc[low_violations, "low"] = data.loc[
                    low_violations, ["open", "close"]
                ].min(axis=1)
                issues.append(f"Fixed {low_violations.sum()} low price violations")

        # Fix zero volume
        if "volume" in data.columns:
            zero_volume = data["volume"] == 0
            if zero_volume.any():
                # Replace zero volume with small positive value
                data.loc[zero_volume, "volume"] = 0.001
                issues.append(f"Fixed {zero_volume.sum()} zero volume values")

        return data, issues

    def optimize_dataframe_dtypes(
        self,
        df: pd.DataFrame,
        preserve_categorical: bool = True,
    ) -> pd.DataFrame:
        """
        Optimize DataFrame data types to reduce memory usage while preserving functionality.

        Args:
            df: Input DataFrame
            preserve_categorical: Whether to preserve categorical columns

        Returns:
            DataFrame with optimized data types
        """
        initial_memory = df.memory_usage(deep=True).sum()
        self.logger.info(
            f"🔧 Optimizing data types - Initial memory: {initial_memory / 1024**2:.2f} MB"
        )

        optimized_df = df.copy()

        # Optimize numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            col_type = df[col].dtype

            # Skip if already optimized
            if col_type in ["int8", "int16", "int32", "float16", "float32"]:
                continue

            # Optimize integers
            if col_type in ["int64"]:
                c_min = df[col].min()
                c_max = df[col].max()

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    optimized_df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    optimized_df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    optimized_df[col] = df[col].astype(np.int32)

            # Optimize floats
            elif col_type in ["float64"]:
                # Check if we can use float32 (lose some precision but save memory)
                if df[col].isnull().sum() == 0:  # No NaN values
                    try:
                        # Test if conversion preserves values within tolerance
                        float32_vals = df[col].astype(np.float32)
                        if np.allclose(df[col], float32_vals, rtol=1e-5):
                            optimized_df[col] = float32_vals
                    except Exception:
                        pass

        # Optimize categorical columns
        if preserve_categorical:
            for col in df.select_dtypes(include=["object"]).columns:
                if (
                    len(df) > 0 and df[col].nunique() / len(df) < 0.5
                ):  # Less than 50% unique values
                    optimized_df[col] = df[col].astype("category")

        # Optimize boolean columns
        for col in df.columns:
            if df[col].dtype == "object":
                if df[col].isin([True, False, 1, 0, "True", "False", "1", "0"]).all():
                    optimized_df[col] = (
                        df[col]
                        .map(
                            {
                                "True": True,
                                "False": False,
                                "1": True,
                                "0": False,
                                1: True,
                                0: False,
                            }
                        )
                        .astype("bool")
                    )

        final_memory = optimized_df.memory_usage(deep=True).sum()
        memory_reduction = (
            (initial_memory - final_memory) / initial_memory if initial_memory else 0.0
        )

        self.logger.info("🔧 Data type optimization complete:")
        self.logger.info(f"   Initial memory: {initial_memory / 1024**2:.2f} MB")
        self.logger.info(f"   Final memory: {final_memory / 1024**2:.2f} MB")
        self.logger.info(f"   Memory reduction: {memory_reduction:.1%}")
        tprint_data_preview(optimized_df, "after_dtype_optimization")

        return optimized_df

    def get_optimal_dtypes_for_features(self) -> Dict[str, str]:
        """
        Get optimal data types for common feature engineering outputs.

        Returns:
            Dictionary mapping feature patterns to optimal data types
        """
        return {
            # Price-based features (typically float32 is sufficient)
            "price_": "float32",
            "close_": "float32",
            "high_": "float32",
            "low_": "float32",
            "open_": "float32",
            # Volume features (can often use int32)
            "volume_": "int32",
            "vol_": "int32",
            # Technical indicators (float32 is sufficient)
            "rsi_": "float32",
            "sma_": "float32",
            "ema_": "float32",
            "bb_": "float32",
            "macd_": "float32",
            "stoch_": "float32",
            # Cluster features (categorical or int8)
            "cluster_": "int8",
            "intensity_cluster_": "float32",
            # Correlation features (float32)
            "correlation_": "float32",
            "corr_": "float32",
            # Volatility features (float32)
            "volatility_": "float32",
            "vol_": "float32",
            # Momentum features (float32)
            "momentum_": "float32",
            "mom_": "float32",
            # Spread features (float32)
            "spread_": "float32",
            "bid_ask_": "float32",
            # Impact features (float32)
            "impact_": "float32",
            "price_impact": "float32",
            "volume_impact": "float32",
        }

    def apply_feature_specific_optimization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature-specific data type optimizations based on feature names.

        Args:
            df: Input DataFrame with features

        Returns:
            DataFrame with optimized data types
        """
        optimized_df = df.copy()
        optimal_dtypes = self.get_optimal_dtypes_for_features()

        for col in df.columns:
            col_lower = col.lower()

            # Find matching pattern
            for pattern, dtype in optimal_dtypes.items():
                if pattern in col_lower:
                    try:
                        if dtype == "int8":
                            # For cluster IDs, ensure they're small integers
                            if col_lower.startswith("cluster_") or "cluster" in col_lower:
                                optimized_df[col] = df[col].astype("int8")
                        elif dtype == "float32":
                            # For float features, use float32 if no precision loss
                            if df[col].dtype == "float64":
                                try:
                                    float32_vals = df[col].astype("float32")
                                    if np.allclose(df[col], float32_vals, rtol=1e-5):
                                        optimized_df[col] = float32_vals
                                except Exception:
                                    pass
                        elif dtype == "int32":
                            # For volume features, use int32 if possible
                            if df[col].dtype == "int64":
                                c_min = df[col].min()
                                c_max = df[col].max()
                                if (
                                    c_min > np.iinfo(np.int32).min
                                    and c_max < np.iinfo(np.int32).max
                                ):
                                    optimized_df[col] = df[col].astype("int32")
                    except Exception as e:
                        self.logger.debug(f"Could not optimize {col} to {dtype}: {e}")
                    break

        return optimized_df

    def optimize_feature_engineering_pipeline(
        self,
        df: pd.DataFrame,
        stage: str = "input",
    ) -> pd.DataFrame:
        """
        Optimize DataFrame for feature engineering pipeline stages.

        Args:
            df: Input DataFrame
            stage: Pipeline stage ("input", "intermediate", "output")

        Returns:
            Optimized DataFrame
        """
        if stage == "input":
            # For input data, be conservative with optimizations
            return self.optimize_dataframe_dtypes(df, preserve_categorical=True)

        if stage == "intermediate":
            # For intermediate calculations, be more aggressive
            return self.optimize_dataframe_dtypes(df, preserve_categorical=False)

        if stage == "output":
            # For final output, apply feature-specific optimizations
            return self.apply_feature_specific_optimization(df)

        return df

    def fix_data_quality_issues(
        self, data: pd.DataFrame, timestamp_column: str = "timestamp"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Fix common data quality issues in the DataFrame.

        Args:
            data: DataFrame to fix
            timestamp_column: Name of the timestamp column

        Returns:
            Tuple of (fixed_data, fix_report)
        """
        fix_report = {
            "original_rows": len(data),
            "duplicates_removed": 0,
            "index_fixed": False,
            "timestamp_converted": False,
            "final_rows": 0,
            "issues_fixed": [],
        }
        self.logger.info(f"🔧 Starting data quality fixes for {len(data)} rows")
        fixed_data = data.copy()

        if timestamp_column in fixed_data.columns:
            fixed_data, timestamp_fixed = self._fix_timestamp_column(fixed_data, timestamp_column)
            if timestamp_fixed:
                fix_report["timestamp_converted"] = True
                fix_report["issues_fixed"].append("timestamp_converted")

        if timestamp_column in fixed_data.columns:
            original_count = len(fixed_data)
            fixed_data = self._remove_duplicate_timestamps(fixed_data, timestamp_column)
            duplicates_removed = original_count - len(fixed_data)
            fix_report["duplicates_removed"] = duplicates_removed
            if duplicates_removed > 0:
                fix_report["issues_fixed"].append(f"removed_{duplicates_removed}_duplicates")
                self.logger.info(f"🗑️ Removed {duplicates_removed} duplicate timestamps")

        if timestamp_column in fixed_data.columns:
            fixed_data, index_fixed = self._fix_non_monotonic_index(fixed_data, timestamp_column)
            if index_fixed:
                fix_report["index_fixed"] = True
                fix_report["issues_fixed"].append("index_sorted")
                self.logger.info("📈 Fixed non-monotonic timestamp index")

        if timestamp_column in fixed_data.columns:
            fixed_data = self._set_datetime_index(fixed_data, timestamp_column)

        fix_report["final_rows"] = len(fixed_data)
        self.logger.info(f"✅ Data quality fixes completed: {fix_report['original_rows']} → {fix_report['final_rows']} rows")
        self.logger.info(f"🔧 Issues fixed: {', '.join(fix_report['issues_fixed'])}")
        return (fixed_data, fix_report)

    def _fix_timestamp_column(self, data: pd.DataFrame, timestamp_column: str) -> Tuple[pd.DataFrame, bool]:
        """Fix timestamp column format."""
        if not pd.api.types.is_datetime64_any_dtype(data[timestamp_column]):
            try:
                data[timestamp_column] = pd.to_datetime(data[timestamp_column])
                self.logger.info("🕒 Converted timestamp column to datetime")
                return (data, True)
            except Exception as e:
                self.logger.warning(f"⚠️ Could not convert timestamp column: {e}")
                return (data, False)
        return (data, False)

    def _remove_duplicate_timestamps(self, data: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
        """Remove duplicate timestamps, keeping the last occurrence."""
        original_count = len(data)
        data = data.drop_duplicates(subset=[timestamp_column], keep="last")
        removed_count = original_count - len(data)
        if removed_count > 0:
            self.logger.info(f"🗑️ Removed {removed_count} duplicate timestamps")
        return data

    def _fix_non_monotonic_index(self, data: pd.DataFrame, timestamp_column: str) -> Tuple[pd.DataFrame, bool]:
        """Fix non-monotonic timestamp index by sorting."""
        if not data[timestamp_column].is_monotonic_increasing:
            data = data.sort_values(timestamp_column).reset_index(drop=True)
            self.logger.info("📈 Sorted data by timestamp to fix non-monotonic index")
            return (data, True)
        return (data, False)

    def _set_datetime_index(self, data: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
        """Set datetime index and remove the timestamp column."""
        if timestamp_column in data.columns:
            data = data.set_index(timestamp_column)
            self.logger.info("📅 Set datetime index")
        return data

    def validate_fixed_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate the fixed data quality."""
        validation_results = {
            "total_rows": len(data),
            "duplicate_timestamps": 0,
            "monotonic_index": True,
            "datetime_index": isinstance(data.index, pd.DatetimeIndex),
            "quality_score": 100.0,
        }
        if isinstance(data.index, pd.DatetimeIndex):
            validation_results["duplicate_timestamps"] = data.index.duplicated().sum()
            validation_results["monotonic_index"] = data.index.is_monotonic_increasing

        quality_score = 100.0
        if validation_results["duplicate_timestamps"] > 0:
            quality_score -= min(20, validation_results["duplicate_timestamps"] / len(data) * 100)
        if not validation_results["monotonic_index"]:
            quality_score -= 10
        if not validation_results["datetime_index"]:
            quality_score -= 5

        validation_results["quality_score"] = max(0, quality_score)
        return validation_results

    def calculate_enhanced_quality_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate enhanced quality metrics for a DataFrame with comprehensive analysis.

        Args:
            data: DataFrame to analyze

        Returns:
            Dictionary with enhanced quality metrics
        """
        if data is None or len(data) == 0:
            return {}

        metrics = {}

        try:
            # Basic metrics
            total_rows = len(data)
            total_cols = len(data.columns)
            missing_values = int(data.isnull().sum().sum())

            # Data type distribution
            numeric_cols = len(data.select_dtypes(include=[np.number]).columns)
            categorical_cols = len(data.select_dtypes(include=['object', 'category']).columns)
            datetime_cols = len(data.select_dtypes(include=['datetime']).columns)

            # Statistical metrics for numeric columns
            numeric_data = data.select_dtypes(include=[np.number])
            if not numeric_len(data) == 0:
                # Volatility metrics
                volatility = numeric_data.std() / numeric_data.mean().replace(0, 1e-8)
                volatility = volatility.replace([np.inf, -np.inf], 0).fillna(0)

                # Skewness and kurtosis (robust calculation)
                skewness = numeric_data.skew().fillna(0)
                kurtosis = numeric_data.kurtosis().fillna(0)

                # Range and outliers
                q75 = numeric_data.quantile(0.75)
                q25 = numeric_data.quantile(0.25)
                iqr = q75 - q25
                iqr = iqr.replace(0, 1e-8)  # Avoid division by zero

                outlier_lower = q25 - 1.5 * iqr
                outlier_upper = q75 + 1.5 * iqr

                # Count outliers
                outliers = ((numeric_data < outlier_lower) | (numeric_data > outlier_upper)).sum()

                metrics.update({
                    'volatility_mean': float(volatility.mean()),
                    'volatility_std': float(volatility.std()),
                    'skewness_mean': float(skewness.mean()),
                    'kurtosis_mean': float(kurtosis.mean()),
                    'outlier_count': int(outliers.sum()),
                    'outlier_ratio': float(outliers.sum() / (total_rows * numeric_cols)) if total_cols > 0 else 0.0,
                })

            # Data consistency metrics
            if isinstance(data.index, pd.DatetimeIndex):
                # Time series specific metrics
                time_gaps = data.index.to_series().diff().dropna()
                if not len(time_gaps) == 0:
                    expected_freq = time_gaps.mode().iloc[0] if len(time_gaps.mode()) > 0 else time_gaps.median()
                    irregular_gaps = abs(time_gaps - expected_freq) > pd.Timedelta(seconds=30)
                    gap_ratio = irregular_gaps.sum() / len(time_gaps)

                    metrics.update({
                        'time_consistency_ratio': float(1.0 - gap_ratio),
                        'avg_time_gap_seconds': float(time_gaps.mean().total_seconds()),
                        'max_time_gap_seconds': float(time_gaps.max().total_seconds()),
                    })

            # Column completeness
            col_completeness = 1.0 - (data.isnull().sum() / total_rows)

            # Data diversity (unique ratio for categorical-like columns)
            diversity_scores = []
            for col in data.columns:
                if data[col].dtype in ['object', 'category'] or data[col].nunique() < 50:
                    unique_ratio = data[col].nunique() / total_rows
                    diversity_scores.append(min(1.0, unique_ratio * 10))  # Scale up for diversity

            metrics.update({
                'total_rows': total_rows,
                'total_columns': total_cols,
                'missing_values': missing_values,
                'missing_ratio': float(missing_values / (total_rows * total_cols)) if total_cols > 0 else 0.0,
                'numeric_columns': numeric_cols,
                'categorical_columns': categorical_cols,
                'datetime_columns': datetime_cols,
                'column_completeness_mean': float(col_completeness.mean()),
                'column_completeness_std': float(col_completeness.std()),
                'data_diversity_score': float(np.mean(diversity_scores)) if diversity_scores else 0.0,
            })

            # Overall quality score (0-100)
            quality_score = 100.0

            # Penalize missing data
            missing_penalty = metrics.get('missing_ratio', 0) * 50
            quality_score -= missing_penalty

            # Penalize outliers
            outlier_penalty = min(20, metrics.get('outlier_ratio', 0) * 200)
            quality_score -= outlier_penalty

            # Penalize time inconsistencies
            time_penalty = (1.0 - metrics.get('time_consistency_ratio', 1.0)) * 10
            quality_score -= time_penalty

            # Bonus for data diversity
            diversity_bonus = metrics.get('data_diversity_score', 0) * 5
            quality_score += diversity_bonus

            metrics['overall_quality_score'] = max(0.0, min(100.0, quality_score))

        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating enhanced quality metrics: {e}")
            metrics = {'error': str(e), 'overall_quality_score': 50.0}  # Default neutral score

        return metrics

# Convenience functions for backwards compatibility
def regularize_timestamps(
    data: pd.DataFrame,
    expected_interval: Optional[timedelta] = None,
    tolerance_seconds: int = 30,
    method: str = "forward_fill",
) -> pd.DataFrame:
    """Regularize timestamps in a DataFrame to ensure consistent intervals."""
    processor = DataProcessor()
    return processor.regularize_timestamps(data, expected_interval, tolerance_seconds, method)

def preprocess_data_for_multi_timeframe(
    price_data: pd.DataFrame,
    volume_data: Optional[pd.DataFrame] = None,
    order_flow_data: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Preprocess data for multi-timeframe feature engineering."""
    processor = DataProcessor()
    return processor.preprocess_data_for_multi_timeframe(price_data, volume_data, order_flow_data)

def validate_and_fix_data_quality(
    data: pd.DataFrame,
    data_type: str = "klines_ohlcv",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Validate and fix common data quality issues."""
    processor = DataProcessor()
    return processor.validate_and_fix_data_quality(data, data_type)

def optimize_dataframe_dtypes(
    df: pd.DataFrame,
    preserve_categorical: bool = True,
) -> pd.DataFrame:
    """Optimize DataFrame data types to reduce memory usage."""
    processor = DataProcessor()
    return processor.optimize_dataframe_dtypes(df, preserve_categorical)

def get_optimal_dtypes_for_features() -> Dict[str, str]:
    """Get optimal data types for common feature engineering outputs."""
    processor = DataProcessor()
    return processor.get_optimal_dtypes_for_features()

def apply_feature_specific_optimization(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature-specific data type optimizations based on feature names."""
    processor = DataProcessor()
    return processor.apply_feature_specific_optimization(df)

def optimize_feature_engineering_pipeline(
    df: pd.DataFrame,
    stage: str = "input",
) -> pd.DataFrame:
    """Optimize DataFrame for feature engineering pipeline stages."""
    processor = DataProcessor()
    return processor.optimize_feature_engineering_pipeline(df, stage)

# Create global instance for backwards compatibility
data_processor = DataProcessor()
