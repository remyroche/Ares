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
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager

from src.utils.logger import system_logger
from src.utils.tprint import tprint

class RegimeSplittingStatus(Enum):
    """Status enumeration for regime splitting operations."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VALIDATION_FAILED = "validation_failed"

@dataclass
class RegimeSplittingMetrics:
    """Comprehensive metrics for regime splitting operations."""
    total_data_points: int = 0
    regime_count: int = 0
    regime_distribution: Dict[int, int] = field(default_factory=dict)
    processing_time_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    validation_checks_passed: int = 0
    validation_checks_failed: int = 0
    warnings_count: int = 0
    errors_count: int = 0
    data_quality_score: float = 0.0
    regime_continuity_score: float = 0.0

@dataclass
class RegimeSplittingResult:
    """Result container for regime splitting operations."""
    success: bool
    data: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: RegimeSplittingMetrics = field(default_factory=RegimeSplittingMetrics)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @classmethod
    def success_result(cls, data: pd.DataFrame, metadata: Dict[str, Any] = None) -> 'RegimeSplittingResult':
        """Create a successful result."""
        return cls(
            success=True,
            data=data,
            metadata=metadata or {},
            metrics=RegimeSplittingMetrics()
        )

    @classmethod
    def failure_result(cls, error: str, metadata: Dict[str, Any] = None) -> 'RegimeSplittingResult':
        """Create a failure result."""
        return cls(
            success=False,
            errors=[error],
            metadata=metadata or {}
        )

class StreamlinedRegimeDataSplitting:
    """
    Streamlined regime data splitting with optimized performance and memory usage.

    This implementation consolidates the functionality from multiple previous files
    into a single, well-structured component that uses modern utility modules.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the streamlined regime data splitting component."""
        tprint('🔧 Initializing StreamlinedRegimeDataSplitting')
        self.config = config or {}
        self.logger = system_logger.getChild('StreamlinedRegimeSplitting')
        tprint('✅ Logger initialized')

        # Initialize hardware optimizations
        self.hardware_manager = UnifiedHardwareManager()
        self.memory_optimizer = get_m1_memory_optimizer()
        self.cpu_optimizer = get_m1_cpu_optimizer()
        self.gpu_manager = get_m1_gpu_manager()
        tprint('✅ Hardware optimizations initialized')

        # Initialize data validation
        self.data_quality_framework = DataQualityFramework()
        self.cross_step_validator = CrossStepValidator()
        tprint('✅ Data validation utilities initialized')

        # Initialize metrics tracking
        self.metrics = RegimeSplittingMetrics()
        self.start_time: Optional[float] = None
        tprint('✅ Metrics tracking initialized')

        # Initialize streaming configuration
        self.chunk_size = self.config.get('chunk_size', 10000)
        self.max_memory_gb = self.config.get('max_memory_gb', 8.0)
        tprint(f'📊 Streaming config: chunk_size={self.chunk_size}, max_memory_gb={self.max_memory_gb}')

        self.logger.info("✅ Streamlined regime data splitting initialized")
        tprint("✅ Streamlined regime data splitting initialized")

    async def execute_regime_splitting(
        self,
        training_input: Dict[str, Any],
        pipeline_state: Dict[str, Any]
    ) -> RegimeSplittingResult:
        """
        Execute streamlined regime data splitting with comprehensive validation and optimization.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            RegimeSplittingResult with tagged data and metadata
        """
        self.start_time = time.time()
        self.logger.info("🚀 Starting streamlined regime data splitting")
        tprint("🚀 Starting streamlined regime data splitting")

        try:
            # Validate inputs
            tprint("🔍 Validating inputs")
            validation_result = self._validate_inputs(training_input)
            if not validation_result:
                tprint("❌ Input validation failed")
                return RegimeSplittingResult.failure_result("Input validation failed")
            tprint("✅ Input validation passed")

            # Load and validate regime data
            tprint("📊 Loading regime data")
            regime_data = await self._load_regime_data_optimized(training_input)
            if regime_data is None:
                tprint("❌ Failed to load regime data")
                return RegimeSplittingResult.failure_result("Failed to load regime data")
            tprint(f"✅ Regime data loaded: {len(regime_data)} rows")

            # Perform data quality assessment
            tprint("📈 Assessing data quality")
            quality_score = self._assess_data_quality(regime_data)
            if quality_score < 0.5:
                tprint(f"❌ Data quality too low: {quality_score}")
                return RegimeSplittingResult.failure_result("Data quality too low for processing")
            tprint(f"✅ Data quality score: {quality_score:.2f}")

            # Apply regime tagging using streaming approach
            tprint("🏷️ Applying regime tagging with streaming approach")
            tagged_data = await self._apply_regime_tagging_streaming(regime_data)
            tprint("✅ Regime tagging completed")

            # Validate tagged data
            tprint("🔍 Validating tagged data")
            validation_passed = self._validate_tagged_data(tagged_data)
            if not validation_passed:
                tprint("❌ Tagged data validation failed")
                return RegimeSplittingResult.failure_result("Tagged data validation failed")
            tprint("✅ Tagged data validation passed")

            # Calculate final metrics
            self._calculate_final_metrics(tagged_data)

            # Create result
            result = RegimeSplittingResult.success_result(
                data=tagged_data,
                metadata={
                    'symbol': training_input.get('symbol'),
                    'exchange': training_input.get('exchange'),
                    'timeframe': training_input.get('timeframe'),
                    'regime_count': self.metrics.regime_count,
                    'processing_time': time.time() - self.start_time,
                    'data_quality_score': self.metrics.data_quality_score
                }
            )
            result.metrics = self.metrics

            self.logger.info(f"✅ Regime data splitting completed in {time.time() - self.start_time:.2f}s")
            return result

        except Exception as e:
            error_msg = f"Regime data splitting failed: {str(e)}"
            self.logger.exception(error_msg)
            return RegimeSplittingResult.failure_result(error_msg)

    def _validate_inputs(self, training_input: Dict[str, Any]) -> bool:
        """Validate input parameters."""
        required_keys = ['symbol', 'exchange', 'data_dir']

        for key in required_keys:
            if key not in training_input or not training_input[key]:
                self.logger.error(f"Missing required parameter: {key}")
                return False

        # Validate symbol and exchange formats
        symbol = training_input['symbol']
        exchange = training_input['exchange']

        if not isinstance(symbol, str) or len(symbol) < 3:
            self.logger.error(f"Invalid symbol format: {symbol}")
            return False

        if not isinstance(exchange, str) or len(exchange) < 2:
            self.logger.error(f"Invalid exchange format: {exchange}")
            return False

        return True

    async def _load_regime_data_optimized(self, training_input: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Load regime data with memory optimization."""
        try:
            symbol = training_input['symbol']
            exchange = training_input['exchange']
            data_dir = training_input['data_dir']

            # Construct path using standardized utilities
            data_path = Path(data_dir) / exchange.lower() / symbol.lower() / 'processed_data.parquet'

            if not data_path.exists():
                self.logger.error(f"Processed data file not found: {data_path}")
                return None

            # Load data with memory optimization
            self.logger.info(f"📊 Loading regime data from {data_path}")

            # Use memory checkpoint for large data loading
            with memory_checkpoint("regime_data_loading"):
                data = safe_read_parquet(data_path)

            if data is None or data.empty:
                self.logger.error("No data loaded or data is empty")
                return None

            # Optimize data types for memory efficiency
            data = optimize_dataframe_dtypes(data)

            self.logger.info(f"✅ Loaded {len(data)} rows of regime data")
            return data

        except Exception as e:
            self.logger.exception(f"Failed to load regime data: {e}")
            return None

    def _assess_data_quality(self, data: pd.DataFrame) -> float:
        """Assess data quality using comprehensive framework."""
        try:
            # Use data quality framework for comprehensive assessment
            quality_result = self.data_quality_framework.validate_dataframe_quality(
                data,
                context="regime_data_splitting"
            )

            # Store quality metrics
            self.metrics.data_quality_score = getattr(quality_result, 'overall_score', 0.0)
            self.metrics.validation_checks_passed = len(getattr(quality_result, 'passed_checks', []))
            self.metrics.validation_checks_failed = len(getattr(quality_result, 'failed_checks', []))

            if hasattr(quality_result, 'warnings') and quality_result.warnings:
                self.metrics.warnings_count = len(quality_result.warnings)
                for warning in quality_result.warnings:
                    self.logger.warning(f"⚠️ Data quality warning: {warning}")

            if hasattr(quality_result, 'issues') and quality_result.issues:
                for issue in quality_result.issues:
                    self.logger.error(f"❌ Data quality issue: {issue}")

            self.logger.info(f"📈 Data quality score: {quality_result.overall_score:.2f}")
            return quality_result.overall_score

        except Exception as e:
            self.logger.exception(f"Data quality assessment failed: {e}")
            return 0.0

    async def _apply_regime_tagging_streaming(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply regime tagging using streaming approach with memory optimization."""
        try:
            self.logger.info("🏷️ Applying regime tagging using streaming approach")

            # Initialize hardware optimizations for streaming
            await self._initialize_streaming_optimizations()

            # Process data in chunks to manage memory usage
            chunks = self._create_data_chunks(data)

            tagged_chunks = []
            processed_count = 0

            # Process chunks with async optimization
            for i, chunk in enumerate(chunks):
                self.logger.info(f"📦 Processing chunk {i+1}/{len(chunks)} ({len(chunk)} rows)")

                # Apply regime tagging to chunk with memory optimization
                tagged_chunk = await self._apply_regime_tags_to_chunk_optimized(chunk)

                if tagged_chunk is not None:
                    tagged_chunks.append(tagged_chunk)
                    processed_count += len(tagged_chunk)

                # Memory checkpoint after each chunk
                if self.memory_optimizer:
                    self.memory_optimizer.memory_checkpoint(f"chunk_{i}_processed")

                # Periodic cleanup and memory optimization
                if i % 5 == 0 and i > 0:
                    await self._perform_periodic_cleanup()

            # Merge all tagged chunks with memory optimization
            if not tagged_chunks:
                raise ValueError("No chunks were successfully processed")

            # Use parallel merge if available
            if len(tagged_chunks) > 1 and self.cpu_optimizer:
                result = await self._parallel_merge_chunks(tagged_chunks)
            else:
                result = safe_merge_dataframes(tagged_chunks)

            self.logger.info(f"✅ Applied regime tagging to {len(result)} rows ({processed_count} processed)")
            return result

        except Exception as e:
            self.logger.exception(f"Regime tagging failed: {e}")
            raise

    def _create_data_chunks(self, data: pd.DataFrame) -> List[pd.DataFrame]:
        """Create data chunks for streaming processing."""
        try:
            total_rows = len(data)
            chunk_size = min(self.chunk_size, total_rows)

            if total_rows <= chunk_size:
                return [data]

            chunks = []
            for i in range(0, total_rows, chunk_size):
                chunk = data.iloc[i:i + chunk_size].copy()
                chunks.append(chunk)

            self.logger.info(f"📦 Created {len(chunks)} chunks of size {chunk_size}")
            return chunks

        except Exception as e:
            self.logger.exception(f"Failed to create data chunks: {e}")
            return [data]

    def _apply_regime_tags_to_chunk(self, chunk: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Apply regime tags to a data chunk."""
        try:
            # Validate chunk data
            if chunk.empty:
                self.logger.warning("⚠️ Empty chunk encountered")
                return None

            # Apply regime tagging logic here
            # For now, this is a placeholder - would need to load regime models
            # and apply them to the data chunk

            # Add placeholder regime column
            chunk = chunk.copy()
            chunk['composite_cluster_id'] = np.random.randint(0, 5, size=len(chunk))

            # Validate temporal continuity
            if 'timestamp' in chunk.columns:
                chunk = chunk.sort_values('timestamp').reset_index(drop=True)

            return chunk

        except Exception as e:
            self.logger.exception(f"Failed to apply regime tags to chunk: {e}")
            return None

    def _validate_tagged_data(self, data: pd.DataFrame) -> bool:
        """Validate tagged data integrity with comprehensive checks."""
        try:
            # Check required columns
            required_columns = ['timestamp', 'composite_cluster_id']
            for col in required_columns:
                if col not in data.columns:
                    self.logger.error(f"Missing required column: {col}")
                    return False

            # Validate data types
            if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
                self.logger.error("Timestamp column must be datetime type")
                return False

            # Validate regime IDs
            if 'composite_cluster_id' in data.columns:
                regime_ids = data['composite_cluster_id']

                # Check for NaN values
                nan_count = regime_ids.isna().sum()
                if nan_count > 0:
                    self.logger.warning(f"⚠️ Found {nan_count} NaN values in regime IDs")
                    self.metrics.warnings_count += 1

                # Check for infinite values
                if pd.api.types.is_numeric_dtype(regime_ids):
                    inf_count = np.isinf(regime_ids).sum()
                    if inf_count > 0:
                        self.logger.error(f"❌ Found {inf_count} infinite values in regime IDs")
                        return False

                # Check for reasonable regime ID range
                unique_regimes = regime_ids.nunique()
                self.metrics.regime_count = unique_regimes

                if unique_regimes == 0:
                    self.logger.error("❌ No regimes found in tagged data")
                    return False

                if unique_regimes > 20:
                    self.logger.warning(f"⚠️ High number of regimes detected: {unique_regimes}")
                    self.metrics.warnings_count += 1

                # Validate regime ID data type
                if not pd.api.types.is_integer_dtype(regime_ids) and not pd.api.types.is_object_dtype(regime_ids):
                    self.logger.warning("⚠️ Regime IDs should be integers")
                    self.metrics.warnings_count += 1

            # Validate temporal continuity with comprehensive checks
            if 'timestamp' in data.columns:
                temporal_issues = self._validate_temporal_continuity(data)
                if temporal_issues:
                    for issue in temporal_issues:
                        self.logger.warning(f"⚠️ Temporal issue: {issue}")
                        self.metrics.warnings_count += 1

            # Validate data completeness
            completeness_issues = self._validate_data_completeness(data)
            if completeness_issues:
                for issue in completeness_issues:
                    self.logger.error(f"❌ Completeness issue: {issue}")
                    self.metrics.errors_count += 1
                return False

            # Validate data consistency
            consistency_issues = self._validate_data_consistency(data)
            if consistency_issues:
                for issue in consistency_issues:
                    self.logger.warning(f"⚠️ Consistency issue: {issue}")
                    self.metrics.warnings_count += 1

            # Validate regime transitions
            transition_issues = self._validate_regime_transitions(data)
            if transition_issues:
                for issue in transition_issues:
                    self.logger.warning(f"⚠️ Regime transition issue: {issue}")
                    self.metrics.warnings_count += 1

            self.logger.info("✅ Tagged data validation passed")
            return True

        except Exception as e:
            self.logger.exception(f"Tagged data validation failed: {e}")
            return False

    def _calculate_final_metrics(self, data: pd.DataFrame):
        """Calculate final processing metrics."""
        try:
            self.metrics.total_data_points = len(data)
            self.metrics.processing_time_seconds = time.time() - self.start_time

            # Calculate memory usage
            if self.memory_optimizer:
                self.metrics.memory_usage_mb = self.memory_optimizer.get_current_memory_usage()

            # Calculate regime distribution
            if 'composite_cluster_id' in data.columns:
                regime_counts = data['composite_cluster_id'].value_counts()
                self.metrics.regime_distribution = {int(k): int(v) for k, v in regime_counts.to_dict().items()}

            # Calculate regime continuity score
            if 'composite_cluster_id' in data.columns:
                regime_changes = data['composite_cluster_id'].diff().ne(0).sum()
                self.metrics.regime_continuity_score = 1.0 - (regime_changes / len(data))

            self.logger.info(f"📊 Final metrics: {self.metrics.total_data_points} points, {self.metrics.regime_count} regimes")

        except Exception as e:
            self.logger.exception(f"Failed to calculate final metrics: {e}")

    async def _initialize_streaming_optimizations(self):
        """Initialize hardware optimizations for streaming processing."""
        try:
            # Initialize memory monitoring
            if self.memory_optimizer:
                self.memory_optimizer.start_m1_memory_monitoring()

            # Initialize CPU optimization
            if self.cpu_optimizer:
                self.cpu_optimizer.optimize_function_for_m1(self._apply_regime_tags_to_chunk_optimized)

            # Initialize GPU optimization if available
            if self.gpu_manager:
                self.gpu_manager.optimize_dataframe_for_m1(pd.DataFrame())

            self.logger.info("✅ Streaming optimizations initialized")

        except Exception as e:
            self.logger.exception(f"Failed to initialize streaming optimizations: {e}")

    async def _apply_regime_tags_to_chunk_optimized(self, chunk: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Apply regime tags to a data chunk with memory optimization."""
        try:
            # Validate chunk data
            if chunk.empty:
                self.logger.warning("⚠️ Empty chunk encountered")
                return None

            price_column = self._get_price_close_column(chunk)
            if price_column is None:
                self.logger.warning("⚠️ Missing price column for regime tagging")
                chunk = chunk.copy()
                chunk['composite_cluster_id'] = -1
                return chunk

            # Use GPU optimization if available for large chunks while keeping deterministic logic
            if len(chunk) > 1000 and self.gpu_manager:
                with gpu_context():
                    chunk = self._apply_regime_tags_to_chunk_cpu(chunk)
            else:
                if self.cpu_optimizer:
                    optimized_func = self.cpu_optimizer.optimize_function_for_m1(
                        self._apply_regime_tags_to_chunk_cpu
                    )
                    chunk = optimized_func(chunk)
                else:
                    chunk = self._apply_regime_tags_to_chunk_cpu(chunk)

            # Validate temporal continuity
            if 'timestamp' in chunk.columns:
                chunk = chunk.sort_values('timestamp').reset_index(drop=True)

            return chunk

        except Exception as e:
            self.logger.exception(f"Failed to apply regime tags to chunk: {e}")
            return None

    def _get_price_close_column(self, chunk: pd.DataFrame) -> Optional[str]:
        """Select the appropriate price column for regime tagging."""
        if 'price_close' in chunk.columns:
            return 'price_close'
        if 'close' in chunk.columns:
            return 'close'
        return None

    def _apply_regime_tags_to_chunk_cpu(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """CPU-optimized regime tagging implementation."""
        chunk = chunk.copy()

        price_column = self._get_price_close_column(chunk)
        if price_column is None:
            self.logger.warning("⚠️ Missing price column for CPU regime tagging")
            chunk['composite_cluster_id'] = -1
            return chunk

        price_series = chunk[price_column]
        price_changes = price_series.pct_change().fillna(0)
        volatility = price_changes.rolling(20, min_periods=1).std().fillna(0)

        regime_bins = [-float('inf'), 0.01, 0.05, 0.1, float('inf')]
        regime_labels = [0, 1, 2, 3]
        regime_categories = pd.cut(volatility, bins=regime_bins, labels=regime_labels)
        regime_codes = regime_categories.cat.codes.replace(-1, regime_labels[0])

        chunk['composite_cluster_id'] = regime_codes.astype(int)

        return chunk

    async def _perform_periodic_cleanup(self):
        """Perform periodic cleanup during streaming."""
        try:
            # Clean up memory
            if self.memory_optimizer:
                self.memory_optimizer.optimize_memory()

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
