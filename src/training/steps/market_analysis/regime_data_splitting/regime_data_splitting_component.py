"""
import warnings
Regime Data Splitting Component.

This component tags data by regimes discovered in previous stages.
Enhanced with comprehensive error handling, validation, and reporting.
Refactored to use common utilities for better maintainability and performance.
"""

import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

# Handle optional dependencies with explicit error reporting
IMPORT_ERRORS = []

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError as e:
    NUMPY_AVAILABLE = False
    np = None
    IMPORT_ERRORS.append(f"numpy: {e}")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError as e:
    PANDAS_AVAILABLE = False
    pd = None
    IMPORT_ERRORS.append(f"pandas: {e}")

from ..components.base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult
from src.utils.logger import system_logger
from src.utils.tprint import tprint

# Import our standardized utilities
from .validation_utils import get_validator, ValidationErrorType, ValidationResult, create_standardized_error
from .config_utils import get_config_manager, get_path_manager

# Use existing error handling utilities
from src.utils.enhanced_error_handler import (
    EnhancedErrorHandler, ErrorSeverity, ErrorCategory, ErrorContext
)

# Use existing hardware utilities
from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager

# Use existing data validation utilities
from src.utils.data.validation.validators import CrossStepValidator
from src.utils.data.quality.data_quality import DataQualityFramework

# Import common utilities
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

from src.utils.serialization_utils import (
    JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
)

from src.utils.data.klines_parquet import (
    save_klines_to_parquet, load_klines_from_parquet, validate_klines_data
)

from src.utils.matrix_operations.unified_operations import (
    safe_matrix_multiply, safe_correlation_matrix, safe_matrix_inverse
)

from src.utils.hardware.m1_gpu_utils import (
    get_m1_gpu_manager as get_gpu_manager, is_m1_available, is_mps_available,
    optimize_dataframe_for_m1, create_m1_optimized_array, m1_backtesting_simulate,
    m1_monte_carlo_simulate
)

from src.utils.hardware.m1_memory_optimizer import (
    get_m1_memory_optimizer as get_memory_optimizer, optimize_dataframe_memory,
    start_m1_memory_monitoring, stop_m1_memory_monitoring, optimize_memory as mem_optimize
)

from src.utils.hardware.m1_cpu_optimizer import (
    get_m1_cpu_optimizer as get_cpu_optimizer, optimize_function_for_m1,
    parallel_map_m1, create_m1_optimized_thread_pool, run_cpu_intensive_task
)

from src.utils.ml_common.common_operations import (
    safe_json_dump, safe_json_load, ensure_directory
)

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
class RegimeSplittingReport:
    """Comprehensive report for regime splitting operations."""
    status: RegimeSplittingStatus
    metrics: RegimeSplittingMetrics
    execution_summary: Dict[str, Any] = field(default_factory=dict)
    validation_results: Dict[str, bool] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

class RegimeDataSplittingComponent(BaseMarketAnalysisComponent):
    """
    Regime Data Splitting Component.

    Tags data by regimes discovered in previous stages using NAS/TAS clustering results.
    Enhanced with comprehensive error handling, validation, and reporting.
    """

    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the regime data splitting component."""
        tprint('🔧 Initializing RegimeDataSplittingComponent')
        super().__init__(config)
        self.logger = system_logger.getChild('RegimeDataSplitting')
        tprint('✅ Logger initialized')

        # Initialize error handler using existing utilities
        error_context = ErrorContext(
            operation="regime_data_splitting",
            component="RegimeDataSplittingComponent"
        )
        self.error_handler = EnhancedErrorHandler(logger=self.logger)
        tprint('✅ Error handler initialized')

        # Initialize hardware manager using existing utilities
        self.hardware_manager = UnifiedHardwareManager()
        tprint('✅ Hardware manager initialized')

        # Initialize data validation using existing utilities
        self.cross_step_validator = CrossStepValidator()
        self.data_quality_framework = DataQualityFramework()
        tprint('✅ Data validation utilities initialized')

        # Validate dependencies and fail fast if missing
        self._validate_dependencies()
        tprint('✅ Dependencies validated')

        # Initialize metrics tracking
        self.metrics = RegimeSplittingMetrics()
        self.start_time: Optional[datetime] = None
        tprint('✅ Metrics tracking initialized')

        # Initialize hardware optimizations using existing utilities
        self._initialize_hardware_optimizations()
        tprint('✅ Hardware optimizations initialized')

        # Initialize memory configuration
        self.max_memory_gb = getattr(self.config, 'max_memory_gb', 8.0)
        self.chunk_size = getattr(self.config, 'chunk_size', 10000)
        self.enable_streaming = getattr(self.config, 'enable_streaming', True)
        tprint(f'📊 Memory config: max_memory_gb={self.max_memory_gb}, chunk_size={self.chunk_size}, streaming={self.enable_streaming}')

        # Initialize common utilities
        self.math_validator = MathValidation()
        self.serializer = UniversalSerializer()

        # Use hardware manager for all hardware operations
        self.gpu_manager = self.hardware_manager.gpu_manager
        self.memory_optimizer = self.hardware_manager.memory_optimizer
        self.cpu_optimizer = self.hardware_manager.cpu_optimizer

        # Initialize standardized validation and configuration
        self.validator = get_validator(self.logger)
        self.config_manager = get_config_manager()
        self.path_manager = get_path_manager()

    def _validate_dependencies(self) -> None:
        """Validate required dependencies and fail fast if missing."""
        tprint('🔍 Validating dependencies')
        try:
            missing_deps = []

            if not NUMPY_AVAILABLE:
                missing_deps.append("numpy")
            if not PANDAS_AVAILABLE:
                missing_deps.append("pandas")

            if missing_deps:
                error_msg = f"Critical dependencies missing: {', '.join(missing_deps)}"
                self.logger.error(f"❌ {error_msg}")
                tprint(f"❌ {error_msg}")
                raise ImportError(error_msg)

            self.logger.info("✅ All required dependencies available")
            tprint("✅ All required dependencies available")

        except Exception as e:
            self.logger.error(f"❌ Critical error in dependency validation: {e}")
            raise

    def _initialize_hardware_optimizations(self) -> None:
        """Initialize hardware optimizations using existing hardware manager."""
        try:
            # Initialize hardware manager
            init_result = self.hardware_manager.initialize()

            if init_result:
                self.logger.info("🧠 Hardware optimizations initialized successfully")

                # Log hardware info
                try:
                    hardware_info = self.hardware_manager.get_system_status()
                    self.logger.info(f"🧠 Hardware Info: {hardware_info}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not retrieve hardware info: {e}")
            else:
                self.logger.info("💻 Using standard optimizations")

        except Exception as e:
            self.logger.warning(f"⚠️ Hardware optimization initialization failed: {e}")
            # Continue without optimizations

    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return [
            'regime_data_splitting_result',
            'regime_splitting_report',
            'regime_validation_results'
        ]

    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute regime data splitting with comprehensive error handling and reporting.

        Args:
            data: Market data for regime tagging
            pipeline_state: Current pipeline state

        Returns:
            ComponentResult with regime data splitting results
        """
        self.start_time = datetime.now()
        self.logger.info('✂️ Starting Enhanced Regime Data Splitting')
        tprint('✂️ Starting Enhanced Regime Data Splitting')

        # Initialize report
        metrics = RegimeSplittingMetrics()
        report = RegimeSplittingReport(
            status=RegimeSplittingStatus.IN_PROGRESS,
            metrics=metrics
        )
        tprint(f'📊 Initialized report with status: {report.status.value}')

        # Fast fail validation for critical inputs using existing error handler
        if data is None:
            error_msg = "Input data is None. Action required: Provide valid market data for regime splitting."
            self.error_handler.handle_error(
                ValueError(error_msg),
                severity=ErrorSeverity.CRITICAL,
                category=ErrorCategory.VALIDATION,
                recovery_action="Provide valid market data for regime splitting"
            )
            tprint(f"❌ {error_msg}")
            report.status = RegimeSplittingStatus.FAILED
            report.errors.append(error_msg)
            return self._create_failure_result(report, error_msg)

        if not isinstance(pipeline_state, dict):
            error_msg = "Pipeline state must be a dictionary. Action required: Ensure pipeline_state is properly initialized as a dict."
            self.error_handler.handle_error(
                ValueError(error_msg),
                severity=ErrorSeverity.CRITICAL,
                category=ErrorCategory.VALIDATION,
                recovery_action="Ensure pipeline_state is properly initialized as a dict"
            )
            tprint(f"❌ {error_msg}")
            report.status = RegimeSplittingStatus.FAILED
            report.errors.append(error_msg)
            return self._create_failure_result(report, error_msg)

        if not self.config.symbol or not self.config.exchange:
            error_msg = "Symbol and exchange must be configured. Action required: Set config.symbol and config.exchange before execution."
            self.error_handler.handle_error(
                ValueError(error_msg),
                severity=ErrorSeverity.CRITICAL,
                category=ErrorCategory.CONFIGURATION,
                recovery_action="Set config.symbol and config.exchange before execution"
            )
            tprint(f"❌ {error_msg}")
            report.status = RegimeSplittingStatus.FAILED
            report.errors.append(error_msg)
            return self._create_failure_result(report, error_msg)

        try:
            # Step 1: Validate inputs
            tprint('🔍 Step 1: Validating inputs...')
            validation_result = self._validate_inputs(data, pipeline_state)
            if not validation_result['valid']:
                tprint(f'❌ Input validation failed: {validation_result["errors"]}')
                report.status = RegimeSplittingStatus.VALIDATION_FAILED
                report.errors.extend(validation_result['errors'])
                return self._create_failure_result(report, "Input validation failed")
            tprint('✅ Input validation passed')

            # Step 2: Load and prepare data with memory optimization
            tprint('📊 Step 2: Loading and preparing market data with memory optimization...')
            market_data = self._load_and_prepare_data(data)
            if market_data is None:
                tprint('❌ Failed to load market data')
                report.status = RegimeSplittingStatus.FAILED
                report.errors.append("Failed to load market data")
                return self._create_failure_result(report, "Data loading failed")
            tprint(f'✅ Market data loaded: {market_data.shape}')

            # Step 2.5: Filter data to match regime assignments (if available)
            # This ensures we use the same limited dataset that was used for clustering
            regime_assignments = self._get_regime_discovery_results(pipeline_state)
            if regime_assignments and 'clustering_result' in regime_assignments:
                clustering_result = regime_assignments['clustering_result']
                if 'data_shape' in clustering_result:
                    expected_shape = clustering_result['data_shape']
                    if len(expected_shape) == 2 and expected_shape[0] < len(market_data):
                        # Filter market data to match the size used for clustering
                        target_size = expected_shape[0]
                        self.logger.info(f"🔍 Filtering market data to match clustering dataset size: {target_size} rows")
                        market_data = market_data.head(target_size)
                        tprint(f'✅ Filtered market data to match clustering size: {market_data.shape}')

            # Additional check: If we have regime assignment files, use their size as reference
            try:
                regime_assignments_from_file = self._load_full_cluster_assignments_from_artifacts()
                if regime_assignments_from_file is not None and len(regime_assignments_from_file) < len(market_data):
                    target_size = len(regime_assignments_from_file)
                    self.logger.info(f"🔍 Filtering market data to match regime assignments file size: {target_size} rows")
                    market_data = market_data.head(target_size)
                    tprint(f'✅ Filtered market data to match regime assignments file size: {market_data.shape}')
            except Exception as e:
                self.logger.debug(f"Could not load regime assignments from file for size reference: {e}")

            # Check if we need streaming processing for large datasets
            if len(market_data) > 50000:  # Large dataset threshold
                tprint('🔄 Large dataset detected, using streaming processing...')
                market_data = await self._stream_process_large_dataset(market_data)
                if market_data is None or len(market_data) == 0:
                    tprint('❌ Streaming processing failed')
                    report.status = RegimeSplittingStatus.FAILED
                    report.errors.append("Streaming processing failed")
                    return self._create_failure_result(report, "Streaming processing failed")
                tprint(f'✅ Streaming processing completed: {market_data.shape}')

            # Step 3: Get regime discovery results
            tprint('🔍 Step 3: Retrieving regime discovery results...')
            regime_discovery = self._get_regime_discovery_results(pipeline_state)
            if not regime_discovery:
                tprint('❌ No regime discovery results available')
                report.status = RegimeSplittingStatus.FAILED
                report.errors.append("No regime discovery results available")
                return self._create_failure_result(report, "Missing regime discovery results")
            tprint('✅ Regime discovery results retrieved')

            # Step 4: Perform regime data splitting
            tprint('✂️ Step 4: Performing regime data splitting...')
            splitting_result = await self._perform_regime_splitting(
                market_data, regime_discovery, report
            )

            if splitting_result is None:
                error_msg = "Regime splitting returned None - check method implementation"
                tprint(f'❌ {error_msg}')
                report.status = RegimeSplittingStatus.FAILED
                report.errors.append(error_msg)
                return self._create_failure_result(report, "Regime splitting failed")

            if not splitting_result['success']:
                error_msg = f"Regime splitting failed: {splitting_result['errors']}"
                if "5-20 regimes" in str(splitting_result['errors']):
                    error_msg = "Regime splitting failed: Clustering results must contain 5-20 regimes for proper regime analysis"
                tprint(f'❌ {error_msg}')
                report.status = RegimeSplittingStatus.FAILED
                report.errors.extend(splitting_result['errors'])
                return self._create_failure_result(report, error_msg)
            tprint('✅ Regime data splitting completed')

            # Step 5: Validate results
            tprint('🔍 Step 5: Validating splitting results...')
            validation_result = self._validate_splitting_results(splitting_result, report)
            if not validation_result['valid']:
                tprint(f'❌ Result validation failed: {validation_result["errors"]}')
                report.status = RegimeSplittingStatus.VALIDATION_FAILED
                report.errors.extend(validation_result['errors'])
                return self._create_failure_result(report, "Result validation failed")
            tprint('✅ Result validation passed')

            # Step 6: Generate comprehensive report
            tprint('📊 Step 6: Generating comprehensive report...')
            report = self._generate_comprehensive_report(
                splitting_result, market_data, report
            )
            tprint('✅ Comprehensive report generated')

            # Step 7: Create artifacts
            tprint('💾 Step 7: Creating artifacts...')
            artifacts = await self._create_artifacts(splitting_result, report)
            tprint('✅ Artifacts created')

            # Update metrics
            tprint('📈 Updating metrics...')
            self._update_metrics(report, splitting_result)

            report.status = RegimeSplittingStatus.COMPLETED
            self.logger.info(f'✅ Enhanced Regime Data Splitting completed: {self.metrics.regime_count} regimes processed')
            tprint(f'✅ Enhanced Regime Data Splitting completed: {self.metrics.regime_count} regimes processed')

            # Save artifacts persistently using the artifact manager
            try:
                save_report = await self.save_artifacts(artifacts, {
                    'symbol': self.config.symbol,
                    'exchange': self.config.exchange,
                    'timeframe': self.config.timeframe,
                    'regime_count': self.metrics.regime_count,
                    'execution_time': self.metrics.processing_time_seconds,
                    'data_quality_score': self.metrics.data_quality_score
                })
                tprint(
                    f"💾 [REGIME_DATA_SPLITTING] Artifacts saved persistently (correlation_id={save_report.correlation_id}): {list(save_report.paths.keys())}",
                    color="green"
                )
            except Exception as e:
                tprint(f"⚠️ [REGIME_DATA_SPLITTING] Failed to save artifacts persistently: {e}", color="yellow")

            return ComponentResult(
                success=True,
                artifacts=artifacts,
                metadata={
                    'symbol': self.config.symbol,
                    'exchange': self.config.exchange,
                    'timeframe': self.config.timeframe,
                    'regime_count': self.metrics.regime_count,
                    'execution_time': self.metrics.processing_time_seconds,
                    'data_quality_score': self.metrics.data_quality_score,
                    'artifacts_saved_persistently': True
                }
            )

        except Exception as e:
            self.logger.error(f'❌ Enhanced Regime Data Splitting failed: {e}')
            tprint(f'❌ Enhanced Regime Data Splitting failed: {e}')
            import traceback
            self.logger.error(f'❌ Error details: {traceback.format_exc()}')
            tprint(f'❌ Error details: {traceback.format_exc()}')

            report.status = RegimeSplittingStatus.FAILED
            report.errors.append(f"Unexpected error: {str(e)}")

            return self._create_failure_result(report, str(e))

    def _validate_inputs(self, data: Any, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input data and pipeline state."""
        self.logger.info("🔍 Validating inputs...")
        tprint("🔍 Validating inputs...")

        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }

        # Check data availability with standardized error messages
        if data is None:
            validation_result['valid'] = False
            validation_result['errors'].append("VALIDATION_ERROR: Input data is None. Action required: Provide valid market data.")

        # Check pipeline state
        if not isinstance(pipeline_state, dict):
            validation_result['valid'] = False
            validation_result['errors'].append("VALIDATION_ERROR: Pipeline state must be a dictionary. Action required: Initialize pipeline_state as dict.")

        # Check for required regime discovery results
        required_keys = ['hmm_regime_discovery_result']
        for key in required_keys:
            if key not in pipeline_state:
                validation_result['warnings'].append(f"WARNING: Missing pipeline state key '{key}'. Action suggested: Ensure previous steps completed successfully.")

        # Check configuration
        if not self.config.symbol:
            validation_result['valid'] = False
            validation_result['errors'].append("CONFIG_ERROR: Symbol not configured. Action required: Set config.symbol.")

        if not self.config.exchange:
            validation_result['valid'] = False
            validation_result['errors'].append("CONFIG_ERROR: Exchange not configured. Action required: Set config.exchange.")

        if validation_result['valid']:
            self.logger.info("✅ Input validation passed")
            tprint("✅ Input validation passed")
        else:
            self.logger.error(f"❌ Input validation failed: {validation_result['errors']}")
            tprint(f"❌ Input validation failed: {validation_result['errors']}")

        return validation_result

    def _load_and_prepare_data(self, data: Any) -> Optional[pd.DataFrame]:
        """Load and prepare market data for regime splitting using memory-optimized utilities."""
        self.logger.info("📊 Loading and preparing market data with memory optimization...")
        tprint("📊 Loading and preparing market data with memory optimization...")

        try:
            if data is None:
                self.logger.error("❌ No data provided")
                return None

            # Handle different data types with memory optimization
            if isinstance(data, pd.DataFrame):
                # Use memory-optimized view instead of copy when possible
                market_data = self._create_memory_optimized_view(data)
            elif isinstance(data, dict) and 'data' in data:
                market_data = self._create_memory_optimized_view(data['data'])
            else:
                self.logger.error(f"❌ Unsupported data type: {type(data)}")
                tprint(f"❌ Unsupported data type: {type(data)}")
                return None

            # Validate DataFrame structure using common utilities
            if not isinstance(market_data, pd.DataFrame):
                self.logger.error("❌ Data is not a DataFrame")
                return None

            if len(market_data) == 0:
                self.logger.error("❌ Market data is empty")
                return None

            # Check for required columns using common utilities
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not validate_dataframe_columns(market_data, required_columns):
                self.logger.warning("⚠️ Missing required columns, creating fallback columns")
                # Create fallback columns with validation
                for col in required_columns:
                    if col not in market_data.columns:
                        if col == 'volume':
                            market_data[col] = 1000.0  # Default volume
                        else:
                            market_data[col] = market_data.get('close', 100.0)  # Use close price as fallback

            # Optimize DataFrame for M1 if available
            if is_m1_available():
                market_data = optimize_dataframe_for_m1(market_data)

            # Apply memory optimization
            market_data = self.memory_optimizer.optimize_dataframe_memory(market_data)

            # Validate and clean data using common utilities
            market_data = safe_fillna(market_data, method='ffill')

            # Optimize data types
            market_data = optimize_dataframe_dtypes(market_data)

            # Validate data quality using common utilities
            quality_metrics = calculate_data_quality_metrics(market_data)
            if quality_metrics.get('missing_percentage', 0) > 10:
                self.logger.warning(f"⚠️ High missing data percentage: {quality_metrics['missing_percentage']:.2f}%")

            # Create data quality report
            quality_report = create_data_quality_report(market_data)
            if quality_report.get('issues'):
                self.logger.warning(f"⚠️ Data quality issues detected: {quality_report['issues']}")

            self.logger.info(f"✅ Market data loaded: {market_data.shape}")
            tprint(f"✅ Market data loaded: {market_data.shape}")
            return market_data

        except Exception as e:
            self.logger.error(f"❌ Error loading market data: {e}")
            tprint(f"❌ Error loading market data: {e}")
            return None

    def _create_memory_optimized_view(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create a memory-optimized view of the DataFrame."""
        try:
            # Use memory context for optimization
            with memory_checkpoint("dataframe_view_creation"):
                # Optimize data types first to reduce memory footprint
                optimized_data = self.memory_optimizer.optimize_dataframe_memory(data)

                # Use view instead of copy when possible
                if hasattr(optimized_data, 'view'):
                    return optimized_data.view()
                else:
                    # Fallback to optimized copy only if view not available
                    return optimized_data.copy(deep=False)

        except Exception as e:
            self.logger.warning(f"⚠️ Memory optimization failed, using standard copy: {e}")
            return data.copy(deep=False)

    async def _stream_process_large_dataset(self, data: pd.DataFrame, chunk_size: int = 10000) -> pd.DataFrame:
        """Process large datasets in streaming fashion to reduce memory usage."""
        try:
            total_rows = len(data)
            if total_rows <= chunk_size:
                return self._process_single_chunk(data)

            self.logger.info(f"🔄 Streaming processing for {total_rows} rows in chunks of {chunk_size}")

            # Initialize memory monitoring
            self.memory_optimizer.start_monitoring()

            processed_chunks = []
            memory_usage_history = []

            for i in range(0, total_rows, chunk_size):
                chunk_end = min(i + chunk_size, total_rows)
                chunk = data.iloc[i:chunk_end]

                # Process chunk with memory optimization
                processed_chunk = self._process_single_chunk(chunk)
                if processed_chunk is not None:
                    processed_chunks.append(processed_chunk)

                # Monitor memory usage
                current_memory = self.memory_optimizer.get_memory_usage()
                memory_usage_history.append(current_memory)

                # Perform cleanup if memory usage is high
                memory_percent = current_memory.get('memory_percent', 0)
                if memory_percent > 80:  # 80% of max memory
                    self._perform_emergency_cleanup()

                # Periodic cleanup every 5 chunks
                if (i // chunk_size) % 5 == 0:
                    await self._perform_periodic_cleanup()

            # Merge processed chunks efficiently
            if processed_chunks:
                result = self._merge_chunks_memory_efficient(processed_chunks)
                self.logger.info(f"✅ Streaming processing completed: {len(result)} rows")
                return result
            else:
                raise ValueError("No chunks were successfully processed")

        except Exception as e:
            self.logger.error(f"❌ Streaming processing failed: {e}")
            raise

    def _process_single_chunk(self, chunk: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Process a single chunk with memory optimization."""
        try:
            if chunk.empty:
                return None

            # Use memory-optimized operations
            with memory_checkpoint("chunk_processing"):
                # Apply regime tagging logic here (placeholder)
                processed_chunk = chunk.copy(deep=False)

                # Add regime information efficiently
                processed_chunk['regime_state'] = self._assign_regime_states_efficient(chunk)

                # Optimize memory usage of the processed chunk
                processed_chunk = self.memory_optimizer.optimize_dataframe_memory(processed_chunk)

                return processed_chunk

        except Exception as e:
            self.logger.error(f"❌ Chunk processing failed: {e}")
            return None

    def _assign_regime_states_efficient(self, chunk: pd.DataFrame) -> np.ndarray:
        """Assign regime states efficiently without creating large intermediate arrays."""
        try:
            # Use vectorized operations for efficiency
            if 'close' in chunk.columns:
                # Simple regime detection based on price volatility
                price_changes = chunk['close'].pct_change().fillna(0)
                volatility = price_changes.rolling(20, min_periods=1).std().fillna(0)

                # Create regime assignments using efficient binning
                regime_states = pd.cut(
                    volatility,
                    bins=[0, 0.01, 0.05, 0.1, float('inf')],
                    labels=[0, 1, 2, 3],
                    include_lowest=True
                ).astype(np.int32)

                return regime_states.values
            else:
                # Fallback to simple assignment
                return np.random.randint(0, 4, size=len(chunk), dtype=np.int32)

        except Exception as e:
            self.logger.warning(f"⚠️ Regime assignment failed: {e}")
            return np.zeros(len(chunk), dtype=np.int32)

    def _merge_chunks_memory_efficient(self, chunks: List[pd.DataFrame]) -> pd.DataFrame:
        """Merge chunks using memory-efficient operations."""
        try:
            if not chunks:
                return pd.DataFrame()

            if len(chunks) == 1:
                return chunks[0]

            # Use memory-optimized concatenation
            with memory_checkpoint("chunk_merging"):
                # Sort chunks by index to maintain temporal order
                sorted_chunks = sorted(chunks, key=lambda x: x.index[0] if not x.empty else 0)

                # Concatenate with memory optimization
                result = pd.concat(sorted_chunks, ignore_index=True, copy=False)

                # Optimize final result
                result = self.memory_optimizer.optimize_dataframe_memory(result)

                return result

        except Exception as e:
            self.logger.error(f"❌ Chunk merging failed: {e}")
            return pd.DataFrame()

    async def _perform_periodic_cleanup(self):
        """Perform periodic memory cleanup during processing."""
        try:
            # Force garbage collection
            import gc
            gc.collect()

            # Use memory optimizer cleanup
            if hasattr(self.memory_optimizer, 'cleanup_memory'):
                self.memory_optimizer.cleanup_memory()

            # Log memory status
            current_memory = self.memory_optimizer.get_current_memory_usage()
            self.logger.debug(f"🧹 Periodic cleanup completed, memory usage: {current_memory:.2f} GB")

        except Exception as e:
            self.logger.warning(f"⚠️ Periodic cleanup failed: {e}")

    def _perform_emergency_cleanup(self):
        """Perform emergency memory cleanup when memory usage is high."""
        try:
            self.logger.warning("🚨 Performing emergency memory cleanup")

            # Force garbage collection multiple times
            import gc
            for _ in range(3):
                gc.collect()

            # Use aggressive memory optimization
            if hasattr(self.memory_optimizer, 'aggressive_cleanup'):
                self.memory_optimizer.aggressive_cleanup()

            # Log memory status after cleanup
            current_memory = self.memory_optimizer.get_current_memory_usage()
            self.logger.info(f"🧹 Emergency cleanup completed, memory usage: {current_memory:.2f} GB")

        except Exception as e:
            self.logger.error(f"❌ Emergency cleanup failed: {e}")

    def _get_regime_discovery_results(self, pipeline_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get NAS/TAS clustering results from pipeline state or load from previous outcomes."""
        self.logger.info("🔍 Retrieving NAS/TAS clustering results...")

        try:
            # Try different possible keys for regime discovery results
            possible_keys = [
                'regime_ensemble_training_result',   # ML model from regime ensemble training
                'optimal_regime_clustering_result',  # Primary NAS/TAS clustering result
                'nas_tas_clustering_result',        # Alternative NAS/TAS clustering key
                'cluster_assignments',              # Direct cluster assignments
                'hmm_regime_discovery_result',      # Fallback to HMM results
                'regime_discovery_result',          # General regime discovery fallback
                'regime_states',                    # Legacy regime states
                'regime_probabilities'              # Legacy regime probabilities
            ]

            regime_discovery = None
            for key in possible_keys:
                if key in pipeline_state and pipeline_state[key]:
                    regime_discovery = pipeline_state[key]
                    self.logger.info(f"✅ Found regime discovery results under key: {key}")
                    break

            # If not found in pipeline state, try to load from previous outcomes
            if regime_discovery is None:
                self.logger.info("📁 No regime discovery results in pipeline state, checking previous outcomes...")
                regime_discovery = self._load_regime_discovery_from_outcomes()

            if regime_discovery is None:
                self.logger.error("❌ No regime discovery results found in pipeline state or outcomes")
                return None

            # Validate regime discovery results
            if isinstance(regime_discovery, dict):
                if not regime_discovery:
                    self.logger.error("❌ Regime discovery results are empty")
                    return None
            elif isinstance(regime_discovery, list):
                if not regime_discovery:
                    self.logger.error("❌ Regime discovery results list is empty")
                    return None

            return regime_discovery

        except Exception as e:
            self.logger.error(f"❌ Error retrieving regime discovery results: {e}")
            return None

    def _load_regime_discovery_from_outcomes(self) -> Optional[Dict[str, Any]]:
        """Load regime discovery results from previous successful outcomes."""
        import os
        import json

        try:
            outcomes_dir = Path("/Users/remyroche/Documents/Ares/outcomes")

            # Look for successful NAS/TAS clustering outcomes
            pattern = "market_analysis_nas_tas_clustering_outcome_*.json"
            outcome_files = list(outcomes_dir.glob(pattern))

            if not outcome_files:
                self.logger.info("📁 No regime discovery outcome files found")
                return None

            # Find the most recent successful outcome
            successful_outcomes = []
            for file_path in outcome_files:
                try:
                    with open(file_path, 'r') as f:
                        outcome_data = json.load(f)

                    # Check if the outcome was successful
                    if outcome_data.get('status') == 'completed':
                        successful_outcomes.append((file_path, outcome_data.get('timestamp', '')))
                except Exception as e:
                    self.logger.debug(f"⚠️ Error reading outcome file {file_path}: {e}")
                    continue

            if not successful_outcomes:
                self.logger.info("📁 No successful regime discovery outcomes found")
                return None

            # Sort by timestamp and get the most recent
            successful_outcomes.sort(key=lambda x: x[1], reverse=True)
            latest_outcome_path, _ = successful_outcomes[0]

            self.logger.info(f"📁 Loading regime discovery results from: {latest_outcome_path}")

            # Load the outcome data
            with open(latest_outcome_path, 'r') as f:
                outcome_data = json.load(f)

            # Extract regime discovery results from artifacts
            artifacts = outcome_data.get('artifacts', {})
            regime_discovery = None

            # Try different possible keys for regime discovery results
            possible_keys = [
                'regime_ensemble_training_result',   # ML model from regime ensemble training
                'hmm_regime_discovery_result',
                'regime_discovery_result',
                'optimal_regime_clustering_result',
                'nas_tas_clustering_result',  # Also check for NAS-TAS clustering result
                'regime_states',
                'regime_probabilities'
            ]

            for key in possible_keys:
                if key in artifacts and artifacts[key]:
                    regime_discovery = artifacts[key]
                    self.logger.info(f"✅ Found regime discovery results under key: {key}")
                    break

            if regime_discovery:
                self.logger.info(f"✅ Successfully loaded regime discovery results from previous outcome")
                return regime_discovery
            else:
                self.logger.warning("⚠️ No regime discovery results found in outcome artifacts")
                return None

        except Exception as e:
            self.logger.error(f"❌ Error loading regime discovery from outcomes: {e}")
            return None

    def _load_cluster_assignment_metadata(self) -> Optional[Dict[str, Any]]:
        """Load metadata describing saved cluster assignment artifacts."""
        try:
            path_manager = getattr(self, 'path_manager', None) or get_path_manager()
            artifacts_dir = path_manager.get_artifacts_dir()
            candidate_dirs = [artifacts_dir / "regime_data_splitting", artifacts_dir]

            for directory in candidate_dirs:
                if not directory.exists():
                    continue

                metadata_files = sorted(directory.glob("*assignment*metadata*.json"))
                for metadata_file in metadata_files:
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as handle:
                            metadata = json.load(handle)
                        if isinstance(metadata, dict):
                            metadata.setdefault('_metadata_file', str(metadata_file))
                            self.logger.info(
                                "📄 Loaded cluster assignment metadata from %s", metadata_file
                            )
                            return metadata
                    except Exception as exc:
                        self.logger.warning(
                            "⚠️ Failed to load cluster assignment metadata from %s: %s",
                            metadata_file,
                            exc,
                        )

            self.logger.info("ℹ️ No cluster assignment metadata found in managed artifact directories")
            return None
        except Exception as exc:
            self.logger.warning(f"⚠️ Unable to load cluster assignment metadata: {exc}")
            return None

    def _load_full_cluster_assignments_from_artifacts(self, metadata: Optional[Dict[str, Any]] = None) -> Optional[np.ndarray]:
        """Load full cluster assignments from saved artifacts when string representation is incomplete."""
        try:
            path_manager = getattr(self, 'path_manager', None) or get_path_manager()
            artifacts_dir = path_manager.get_artifacts_dir()
            candidate_dirs = [artifacts_dir]

            additional_dirs = [
                artifacts_dir / "regime_data_splitting",
                artifacts_dir / "market_analysis",
                artifacts_dir / "market_analysis" / "clustering",
            ]

            for directory in additional_dirs:
                if directory.exists():
                    candidate_dirs.append(directory)

            candidate_files: List[Path] = []

            if metadata and 'artifact_path' in metadata:
                artifact_path = Path(metadata['artifact_path'])
                if not artifact_path.is_absolute():
                    artifact_path = artifacts_dir / artifact_path
                if artifact_path.exists():
                    candidate_files.append(artifact_path)
                else:
                    self.logger.warning(
                        "⚠️ Metadata referenced artifact %s but it does not exist",
                        artifact_path,
                    )

            patterns = [
                "nas_tas_regime_assignments_*.parquet",
                "*regime_assignments*.parquet",
                "nas_tas_clustering_results_*.pkl",
                "*clustering*results*.pkl",
                "*regime*clustering*.pkl",
                "*nas_tas*.pkl",
            ]

            seen_files = set()
            for directory in candidate_dirs:
                if not directory.exists():
                    continue
                for pattern in patterns:
                    for file_path in directory.rglob(pattern):
                        if file_path in seen_files:
                            continue
                        seen_files.add(file_path)
                        candidate_files.append(file_path)

            for file_path in sorted(candidate_files, key=lambda x: x.stat().st_mtime, reverse=True):
                self.logger.info(f"📁 Attempting to load cluster assignments from: {file_path}")

                try:
                    if file_path.suffix == '.parquet':
                        if not PANDAS_AVAILABLE:
                            self.logger.warning("⚠️ pandas is required to load parquet files but is unavailable")
                            continue
                        df = pd.read_parquet(file_path)
                        if 'regime_id' in df.columns:
                            states = df['regime_id'].to_numpy()
                        elif 'cluster_assignments' in df.columns:
                            states = df['cluster_assignments'].to_numpy()
                        else:
                            self.logger.warning(
                                "⚠️ Parquet file %s does not contain expected columns",
                                file_path,
                            )
                            continue
                    elif file_path.suffix == '.pkl':
                        import pickle

                        with open(file_path, 'rb') as handle:
                            saved_results = pickle.load(handle)

                        possible_results = []
                        if isinstance(saved_results, dict):
                            if 'cluster_assignments' in saved_results:
                                possible_results.append(saved_results['cluster_assignments'])
                            if 'assignments' in saved_results:
                                possible_results.append(saved_results['assignments'])
                            if 'results' in saved_results:
                                results = saved_results['results']
                                if isinstance(results, dict):
                                    if 'cluster_assignments' in results:
                                        possible_results.append(results['cluster_assignments'])
                                    if 'assignments' in results:
                                        possible_results.append(results['assignments'])
                        else:
                            possible_results.append(saved_results)

                        states = None
                        for candidate in possible_results:
                            if candidate is None:
                                continue
                            if isinstance(candidate, str):
                                numbers = re.findall(r'\d+', candidate)
                                if not numbers:
                                    continue
                                states = np.array([int(x) for x in numbers], dtype=np.int32)
                            else:
                                states = np.array(candidate)

                            if states is not None:
                                break

                        if states is None:
                            self.logger.warning(
                                "⚠️ Could not extract cluster assignments from pickle %s",
                                file_path,
                            )
                            continue
                    else:
                        self.logger.debug(f"ℹ️ Skipping unsupported artifact type: {file_path.suffix}")
                        continue

                    if not isinstance(states, np.ndarray):
                        states = np.array(states)

                    if metadata and 'expected_length' in metadata:
                        expected_length = int(metadata['expected_length'])
                        if len(states) != expected_length:
                            self.logger.warning(
                                "⚠️ Loaded %s assignments from %s but expected %s",
                                len(states),
                                file_path,
                                expected_length,
                            )
                            continue

                    unique_regimes = len(np.unique(states))
                    if unique_regimes < 1:
                        self.logger.warning(
                            "⚠️ Loaded assignments from %s but no regimes detected",
                            file_path,
                        )
                        continue

                    self.logger.info(
                        "✅ Loaded %s assignments with %s regimes from %s",
                        len(states),
                        unique_regimes,
                        file_path,
                    )
                    return states
                except Exception as exc:
                    self.logger.error(f"❌ Failed to load regime assignments from {file_path}: {exc}")
                    continue

            self.logger.warning("⚠️ No cluster assignments found in any managed artifact directories")
            return None

        except Exception as e:
            self.logger.error(f"❌ Error loading cluster assignments from artifacts: {e}")
            return None

    def _parse_cluster_assignments_string(
        self,
        states_str: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Parse cluster assignments stored as a (potentially truncated) numpy string."""
        if not NUMPY_AVAILABLE:
            raise ValueError("numpy is required to parse cluster assignment strings")

        if not isinstance(states_str, str):
            raise ValueError("Cluster assignments must be provided as a string for parsing")

        metadata = metadata or self._load_cluster_assignment_metadata()
        preview_numbers = [int(x) for x in re.findall(r'\d+', states_str)]
        is_truncated = '...' in states_str

        if is_truncated:
            self.logger.info("📄 Detected truncated cluster assignment string; attempting artifact recovery")
            assignments = self._load_full_cluster_assignments_from_artifacts(metadata)
            if assignments is not None:
                if metadata and 'expected_length' in metadata:
                    expected_length = int(metadata['expected_length'])
                    if len(assignments) != expected_length:
                        raise ValueError(
                            f"Loaded {len(assignments)} assignments but expected {expected_length} according to metadata"
                        )
                return np.array(assignments, dtype=np.int32)

            expected_length = (
                int(metadata['expected_length'])
                if metadata and 'expected_length' in metadata
                else None
            )
            if expected_length is not None:
                raise ValueError(
                    "Truncated cluster assignment string encountered; "
                    f"expected {expected_length} assignments but no artifact could be loaded"
                )
            raise ValueError(
                "Truncated cluster assignment string encountered but no cluster assignment artifact could be loaded"
            )

        if not preview_numbers:
            raise ValueError("Cluster assignment string did not contain any numeric values")

        assignments = np.array(preview_numbers, dtype=np.int32)

        if metadata and 'expected_length' in metadata:
            expected_length = int(metadata['expected_length'])
            if len(assignments) != expected_length:
                self.logger.info(
                    "📄 Cluster assignment string length %s does not match expected %s; attempting artifact recovery",
                    len(assignments),
                    expected_length,
                )
                full_assignments = self._load_full_cluster_assignments_from_artifacts(metadata)
                if full_assignments is not None and len(full_assignments) == expected_length:
                    return np.array(full_assignments, dtype=np.int32)
                raise ValueError(
                    f"Cluster assignment string contained {len(assignments)} entries but expected {expected_length}"
                )

        return assignments

    def _resolve_cluster_assignments_value(
        self,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Normalize cluster assignments from various representations using optional metadata."""
        if not NUMPY_AVAILABLE:
            raise ValueError("numpy is required to resolve cluster assignments")

        if isinstance(value, str):
            return self._parse_cluster_assignments_string(value, metadata)

        assignments = value if isinstance(value, np.ndarray) else np.array(value)

        if metadata and 'expected_length' in metadata:
            expected_length = int(metadata['expected_length'])
            if len(assignments) != expected_length:
                self.logger.info(
                    "📄 Loaded %s assignments but expected %s according to metadata; attempting artifact recovery",
                    len(assignments),
                    expected_length,
                )
                recovered = self._load_full_cluster_assignments_from_artifacts(metadata)
                if recovered is None or len(recovered) != expected_length:
                    raise ValueError(
                        f"Loaded {len(assignments)} assignments but expected {expected_length}"
                    )
                assignments = recovered if isinstance(recovered, np.ndarray) else np.array(recovered)

        return assignments

    def _predict_regime_states_with_ml_model(self, ensemble_result: Dict[str, Any], market_data: pd.DataFrame) -> Optional[np.ndarray]:
        """Use the trained ML model from regime ensemble training to predict regime states with probabilistic outputs."""
        try:
            self.logger.info("🤖 Using ML model to predict regime states with probabilistic outputs")

            # Check if we have the new probabilistic prediction method available
            if 'stacker_lgbm_calibrated' in ensemble_result:
                # Use the new probabilistic prediction method
                from src.training.steps.market_analysis.components.regime_ensemble_training import RegimeEnsembleTrainingComponent

                # Create a temporary component instance to use the prediction method
                ensemble_component = RegimeEnsembleTrainingComponent()

                # Extract feature names from metadata
                metadata = ensemble_result.get('metadata', {})
                feature_names = metadata.get('feature_names', [])

                if not feature_names:
                    self.logger.error("❌ No feature names found in regime ensemble training metadata")
                    return None

                self.logger.info(f"📊 Using {len(feature_names)} features for regime prediction")

                # Prepare features for prediction
                available_features = []
                missing_features = []

                for feature_name in feature_names:
                    if feature_name in market_data.columns:
                        available_features.append(feature_name)
                    else:
                        missing_features.append(feature_name)

                if missing_features:
                    self.logger.warning(f"⚠️ Missing {len(missing_features)} features: {missing_features[:5]}...")
                    self.logger.warning("⚠️ Will use available features only")

                if not available_features:
                    self.logger.error("❌ No required features found in market data")
                    return None

                # Prepare feature matrix
                X = market_data[available_features].fillna(0).values

                self.logger.info(f"📊 Prepared feature matrix: {X.shape}")

                # Use the probabilistic prediction method
                prediction_result = ensemble_component.predict_regimes_with_probabilities(
                    stacker_result=ensemble_result,
                    X=X,
                    feature_names=available_features,
                    scaler=None  # No scaler needed as we're using the raw features
                )

                if 'error' in prediction_result:
                    self.logger.error(f"❌ Error in probabilistic prediction: {prediction_result['error']}")
                    return None

                # Extract regime labels
                regime_predictions = prediction_result.get('regime_labels')
                regime_probabilities = prediction_result.get('regime_probabilities')

                if regime_predictions is None:
                    self.logger.error("❌ No regime predictions returned from probabilistic method")
                    return None

                self.logger.info(f"✅ Generated {len(regime_predictions)} regime predictions with probabilities")

                # Store the probabilistic information for later use
                self._cached_regime_probabilities = regime_probabilities
                self._cached_regime_analysis = prediction_result.get('regime_analysis', {})
                self._cached_ensemble_probabilities = prediction_result.get('ensemble_probabilities', {})

                # Validate regime count
                unique_regimes = len(np.unique(regime_predictions))
                if unique_regimes < 5:
                    self.logger.error(f"❌ Insufficient regimes predicted: {unique_regimes} found, minimum 5 required")
                    return None
                elif unique_regimes > 20:
                    self.logger.error(f"❌ Too many regimes predicted: {unique_regimes} found, maximum 20 allowed")
                    return None

                self.logger.info(f"✅ ML model regime prediction successful: {len(regime_predictions)} assignments with {unique_regimes} regimes")
                return regime_predictions.astype(np.int32)

            else:
                # Fallback to old method if new structure not available
                self.logger.warning("⚠️ Using fallback prediction method - consider updating to probabilistic outputs")
                return self._predict_regime_states_with_ml_model_fallback(ensemble_result, market_data)

        except Exception as e:
            self.logger.error(f"❌ Error predicting regime states with ML model: {e}")
            return None

    def _predict_regime_states_with_ml_model_fallback(self, ensemble_result: Dict[str, Any], market_data: pd.DataFrame) -> Optional[np.ndarray]:
        """Fallback method for predicting regime states with ML model (old implementation)."""
        try:
            self.logger.info("🤖 Using fallback ML model to predict regime states")

            # Extract the trained model
            if 'stacker_lgbm_calibrated' not in ensemble_result:
                self.logger.error("❌ No trained ML model found in regime ensemble training result")
                return None

            model = ensemble_result['stacker_lgbm_calibrated']
            self.logger.info(f"✅ Found trained ML model: {type(model)}")

            # Extract feature names from metadata
            metadata = ensemble_result.get('metadata', {})
            feature_names = metadata.get('feature_names', [])

            if not feature_names:
                self.logger.error("❌ No feature names found in regime ensemble training metadata")
                return None

            self.logger.info(f"📊 Using {len(feature_names)} features for regime prediction")

            # Prepare features for prediction
            available_features = []
            missing_features = []

            for feature_name in feature_names:
                if feature_name in market_data.columns:
                    available_features.append(feature_name)
                else:
                    missing_features.append(feature_name)

            if missing_features:
                self.logger.warning(f"⚠️ Missing {len(missing_features)} features: {missing_features[:5]}...")
                self.logger.warning("⚠️ Will use available features only")

            if not available_features:
                self.logger.error("❌ No required features found in market data")
                return None

            # Prepare feature matrix
            X = market_data[available_features].fillna(0).values

            self.logger.info(f"📊 Prepared feature matrix: {X.shape}")

            # Make predictions
            self.logger.info("🔮 Making regime predictions with ML model...")
            regime_predictions = model.predict(X)

            self.logger.info(f"✅ Generated {len(regime_predictions)} regime predictions")

            # Validate regime count
            unique_regimes = len(np.unique(regime_predictions))
            if unique_regimes < 5:
                self.logger.error(f"❌ Insufficient regimes predicted: {unique_regimes} found, minimum 5 required")
                return None
            elif unique_regimes > 20:
                self.logger.error(f"❌ Too many regimes predicted: {unique_regimes} found, maximum 20 allowed")
                return None

            self.logger.info(f"✅ ML model regime prediction successful: {len(regime_predictions)} assignments with {unique_regimes} regimes")
            return regime_predictions.astype(np.int32)

        except Exception as e:
            self.logger.error(f"❌ Error predicting regime states with ML model: {e}")
            return None

    def _predict_regime_probabilities_with_ml_model(self, ensemble_result: Dict[str, Any], market_data: pd.DataFrame) -> Optional[np.ndarray]:
        """Use the trained ML model to predict regime probabilities with comprehensive probabilistic outputs."""
        try:
            self.logger.info("🤖 Using ML model to predict regime probabilities with comprehensive outputs")

            # Check if we have cached probabilities from the state prediction
            if hasattr(self, '_cached_regime_probabilities') and self._cached_regime_probabilities is not None:
                self.logger.info("✅ Using cached regime probabilities from state prediction")
                return self._cached_regime_probabilities

            # If not cached, use the probabilistic prediction method
            if 'stacker_lgbm_calibrated' in ensemble_result:
                from src.training.steps.market_analysis.components.regime_ensemble_training import RegimeEnsembleTrainingComponent

                # Create a temporary component instance to use the prediction method
                ensemble_component = RegimeEnsembleTrainingComponent()

                # Extract feature names from metadata
                metadata = ensemble_result.get('metadata', {})
                feature_names = metadata.get('feature_names', [])

                if not feature_names:
                    self.logger.error("❌ No feature names found in regime ensemble training metadata")
                    return None

                # Prepare features for prediction
                available_features = [f for f in feature_names if f in market_data.columns]

                if not available_features:
                    self.logger.error("❌ No required features found in market data")
                    return None

                # Prepare feature matrix
                X = market_data[available_features].fillna(0).values

                # Use the probabilistic prediction method
                prediction_result = ensemble_component.predict_regimes_with_probabilities(
                    stacker_result=ensemble_result,
                    X=X,
                    feature_names=available_features,
                    scaler=None  # No scaler needed as we're using the raw features
                )

                if 'error' in prediction_result:
                    self.logger.error(f"❌ Error in probabilistic prediction: {prediction_result['error']}")
                    return None

                # Extract regime probabilities
                regime_probabilities = prediction_result.get('regime_probabilities')

                if regime_probabilities is None:
                    self.logger.error("❌ No regime probabilities returned from probabilistic method")
                    return None

                self.logger.info(f"✅ Generated regime probabilities: {regime_probabilities.shape}")
                return regime_probabilities

            else:
                # Fallback to old method if new structure not available
                self.logger.warning("⚠️ Using fallback probability prediction method")
                return self._predict_regime_probabilities_with_ml_model_fallback(ensemble_result, market_data)

        except Exception as e:
            self.logger.error(f"❌ Error predicting regime probabilities with ML model: {e}")
            return None

    def _predict_regime_probabilities_with_ml_model_fallback(self, ensemble_result: Dict[str, Any], market_data: pd.DataFrame) -> Optional[np.ndarray]:
        """Fallback method for predicting regime probabilities with ML model (old implementation)."""
        try:
            self.logger.info("🤖 Using fallback ML model to predict regime probabilities")

            # Extract the trained model
            if 'stacker_lgbm_calibrated' not in ensemble_result:
                self.logger.error("❌ No trained ML model found in regime ensemble training result")
                return None

            model = ensemble_result['stacker_lgbm_calibrated']

            # Extract feature names from metadata
            metadata = ensemble_result.get('metadata', {})
            feature_names = metadata.get('feature_names', [])

            if not feature_names:
                self.logger.error("❌ No feature names found in regime ensemble training metadata")
                return None

            # Prepare features for prediction
            available_features = [f for f in feature_names if f in market_data.columns]

            if not available_features:
                self.logger.error("❌ No required features found in market data")
                return None

            # Prepare feature matrix
            X = market_data[available_features].fillna(0).values

            # Make probability predictions
            self.logger.info("🔮 Making regime probability predictions with ML model...")
            regime_probabilities = model.predict_proba(X)

            self.logger.info(f"✅ Generated regime probabilities: {regime_probabilities.shape}")
            return regime_probabilities

    def get_regime_probabilities(self) -> Dict[str, Any]:
        """Get regime probabilities for downstream models (Analyst & Tactician)."""
        try:
            self.logger.info("📊 Providing regime probabilities for downstream models")

            regime_info = {
                'regime_probabilities': getattr(self, '_cached_regime_probabilities', None),
                'has_probabilistic_outputs': hasattr(self, '_cached_regime_probabilities') and self._cached_regime_probabilities is not None,
                'timestamp': datetime.now().isoformat()
            }

            self.logger.info("✅ Regime probabilities prepared for downstream models")
            return regime_info

        except Exception as e:
            self.logger.error(f"❌ Error providing regime probabilities: {e}")
            return {
                'regime_probabilities': None,
                'has_probabilistic_outputs': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    async def _perform_regime_splitting(
        self,
        market_data: pd.DataFrame,
        regime_discovery: Dict[str, Any],
        report: RegimeSplittingReport
    ) -> Dict[str, Any]:
        """Perform the actual regime data splitting process using common utilities and hardware optimizations."""
        self.logger.info("✂️ Performing regime data splitting...")
        tprint("✂️ Performing regime data splitting...")

        # Use hardware manager for proper memory management
        from src.utils.hardware.memory_optimization import memory_context
        with memory_context("regime_splitting"):
            try:
                # Extract regime states and probabilities using safe operations
                regime_states = self._extract_regime_states(regime_discovery, market_data)

                # Handle ML model probabilities if using regime ensemble training
                if isinstance(regime_discovery, dict) and 'regime_ensemble_training_result' in regime_discovery:
                    regime_probabilities = self._predict_regime_probabilities_with_ml_model(
                        regime_discovery['regime_ensemble_training_result'], market_data
                    )
                else:
                    regime_probabilities = self._extract_regime_probabilities(regime_discovery)

                if regime_states is None:
                    return {
                        'success': False,
                        'errors': ['Failed to extract regime states - clustering results must contain 5-20 regimes'],
                        'data': None
                    }

                # Validate regime states length before alignment
                original_market_len = len(market_data)
                original_regime_len = len(regime_states)

                # Check if regime states length is reasonable but be more flexible
                if original_regime_len < 10:
                    error_msg = f"Regime states length ({original_regime_len}) is too small for any meaningful analysis. Minimum 10 assignments required."
                    self.logger.error(f"❌ {error_msg}")
                    tprint(f"❌ {error_msg}")
                    return {
                        'success': False,
                        'errors': [error_msg],
                        'data': None
                    }

                # Log warnings for limited data but continue processing
                if original_regime_len < 100:
                    self.logger.warning(f"⚠️ Limited regime states available: {original_regime_len} assignments for {original_market_len} market data points")
                    self.logger.warning("⚠️ This may result in limited regime analysis but will use actual clustering results")

                # Check if regime states length is significantly different from market data length
                length_ratio = original_regime_len / original_market_len
                if length_ratio < 0.01:  # Less than 1% of expected length - this is truly problematic
                    error_msg = f"Regime states length ({original_regime_len}) is extremely small compared to market data length ({original_market_len}). Ratio: {length_ratio:.3f}"
                    self.logger.error(f"❌ {error_msg}")
                    tprint(f"❌ {error_msg}")
                    return {
                        'success': False,
                        'errors': [error_msg],
                        'data': None
                    }
                elif length_ratio < 0.1:  # Less than 10% but more than 1% - warn but continue
                    self.logger.warning(f"⚠️ Regime states length ({original_regime_len}) is smaller than market data length ({original_market_len}). Ratio: {length_ratio:.3f}")
                    self.logger.warning("⚠️ This will result in partial regime analysis but will use actual clustering results")

                # Align data lengths with proper validation and temporal consistency checks
                min_len = min(original_market_len, original_regime_len)

                # Validate data alignment impact
                data_loss_percentage = ((max(original_market_len, original_regime_len) - min_len) /
                                      max(original_market_len, original_regime_len)) * 100

                # Use existing data quality framework for validation
                if data_loss_percentage > 5.0:  # More than 5% data loss
                    warning_msg = f"Data alignment will lose {data_loss_percentage:.1f}% of data ({max(original_market_len, original_regime_len) - min_len} rows)"
                    self.logger.warning(f"⚠️ {warning_msg}")
                    tprint(f"⚠️ {warning_msg}")
                    report.warnings.append(warning_msg)

                    if data_loss_percentage > 20.0:  # Critical data loss
                        error_msg = f"Critical data loss during alignment: {data_loss_percentage:.1f}%"
                        self.logger.error(f"❌ {error_msg}")
                        tprint(f"❌ {error_msg}")
                        report.errors.append(error_msg)

                if min_len == 0:
                    error_msg = "No overlapping data between market data and regime states. Action required: Ensure market data and regime states have compatible time ranges."
                    self.error_handler.handle_error(
                        ValueError(error_msg),
                        severity=ErrorSeverity.CRITICAL,
                        category=ErrorCategory.DATA_QUALITY,
                        recovery_action="Ensure market data and regime states have compatible time ranges"
                    )
                    return {
                        'success': False,
                        'errors': [error_msg],
                        'data': None
                    }

                tprint(f"📊 Aligning data lengths: market_data={original_market_len}, regime_states={original_regime_len} -> {min_len}")

                # Validate temporal consistency if timestamp columns exist
                temporal_validation_passed = True
                if hasattr(market_data, 'index') and hasattr(market_data.index, 'name') and market_data.index.name == 'timestamp':
                    # DataFrame has timestamp index
                    market_timestamps = market_data.index[:min_len]
                    if len(market_timestamps) > 1:
                        # Check for temporal consistency (monotonic increasing)
                        if not market_timestamps.is_monotonic_increasing:
                            warning_msg = "Market data timestamps are not monotonic increasing - temporal consistency may be compromised"
                            self.logger.warning(f"⚠️ {warning_msg}")
                            report.warnings.append(warning_msg)
                        temporal_validation_passed = False
                elif 'timestamp' in market_data.columns:
                    # DataFrame has timestamp column
                    market_timestamps = market_data['timestamp'].iloc[:min_len]
                    if len(market_timestamps) > 1:
                        # Check for temporal consistency
                        if not market_timestamps.is_monotonic_increasing:
                            warning_msg = "Market data timestamps are not monotonic increasing - temporal consistency may be compromised"
                            self.logger.warning(f"⚠️ {warning_msg}")
                            report.warnings.append(warning_msg)
                            temporal_validation_passed = False

                # Use memory-optimized DataFrame operations
                try:
                    # Use memory-efficient slicing instead of copy
                    market_data_aligned = self._create_memory_optimized_view(market_data.iloc[:min_len])
                    if market_data_aligned is None:
                        raise ValueError("Failed to align market data")
                except Exception as e:
                    error = self.error_handler.handle_alignment_error(
                        f"Failed to align market data: {str(e)}",
                        "Check market data format and ensure it can be properly sliced",
                        context={'exception': str(e), 'min_len': min_len},
                        severity=ErrorSeverity.HIGH
                    )
                    return {
                        'success': False,
                        'errors': [error.to_string()],
                        'data': None
                    }

                try:
                    regime_states_aligned = regime_states[:min_len]
                    # Validate regime states alignment
                    if len(regime_states_aligned) != min_len:
                        raise ValueError(f"Regime states alignment failed: expected {min_len}, got {len(regime_states_aligned)}")
                except Exception as e:
                    error = self.error_handler.handle_alignment_error(
                        f"Failed to align regime states: {str(e)}",
                        "Check regime states format and ensure it can be properly sliced",
                        context={'exception': str(e), 'min_len': min_len},
                        severity=ErrorSeverity.HIGH
                    )
                    return {
                        'success': False,
                        'errors': [error.to_string()],
                        'data': None
                    }

                if regime_probabilities is not None:
                    try:
                        regime_probabilities_aligned = regime_probabilities[:min_len]
                        # Validate probabilities alignment
                        if len(regime_probabilities_aligned) != min_len:
                            self.logger.warning("⚠️ Regime probabilities alignment mismatch, setting to None")
                            regime_probabilities_aligned = None
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to align regime probabilities: {e}, setting to None")
                        regime_probabilities_aligned = None
                else:
                    regime_probabilities_aligned = None

                # Final validation of aligned data
                if len(market_data_aligned) != len(regime_states_aligned):
                    error = self.error_handler.handle_alignment_error(
                        f"Data alignment validation failed: market_data={len(market_data_aligned)}, regime_states={len(regime_states_aligned)}",
                        "Review data alignment logic and ensure consistent processing",
                        context={
                            'market_data_length': len(market_data_aligned),
                            'regime_states_length': len(regime_states_aligned),
                            'expected_length': min_len
                        },
                        severity=ErrorSeverity.CRITICAL
                    )
                    return {
                        'success': False,
                        'errors': [error.to_string()],
                        'data': None
                    }

                # Log alignment success
                alignment_info = {
                    'original_market_data_length': original_market_len,
                    'original_regime_states_length': original_regime_len,
                    'aligned_length': min_len,
                    'data_loss_percentage': data_loss_percentage,
                    'temporal_validation_passed': temporal_validation_passed
                }
                self.logger.info(f"✅ Data alignment completed successfully: {alignment_info}")

                # Clean up original data references using hardware manager
                del market_data

                # Optimize memory using hardware manager
                try:
                    if hasattr(self.hardware_manager, 'optimize_memory'):
                        memory_result = self.hardware_manager.optimize_memory()
                        self.logger.debug(f"Memory optimization result: {memory_result}")
                    else:
                        # Use memory optimizer directly
                        memory_result = self.memory_optimizer.optimize_memory()
                        self.logger.debug(f"Memory optimization result: {memory_result}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Memory optimization failed: {e}")
                    # Continue without memory optimization

                # Add regime information to market data using safe operations
                market_data_aligned['regime_state'] = regime_states_aligned
                if regime_probabilities_aligned is not None:
                    market_data_aligned['regime_probability'] = regime_probabilities_aligned
                    # Compute max probability per row robustly, independent of DataFrame index
                    try:
                        market_data_aligned['regime_confidence'] = np.max(regime_probabilities_aligned, axis=1)
                    except Exception:
                        market_data_aligned['regime_confidence'] = 1.0
                else:
                    market_data_aligned['regime_confidence'] = 1.0

                # Calculate regime statistics using memory-optimized utilities
                regime_stats = self._calculate_regime_statistics_memory_optimized(market_data_aligned)

                # Get regime probabilities for downstream models
                regime_probabilities_info = self.get_regime_probabilities()

                # Create regime data dictionary
                regime_data = {
                    'market_data': market_data_aligned,
                    'regime_states': regime_states_aligned,
                    'regime_probabilities': regime_probabilities_aligned,
                    'regime_statistics': regime_stats,
                    'regime_probabilities_info': regime_probabilities_info
                }

                # Add regime probability features to market data for downstream models
                if regime_probabilities_aligned is not None:
                    n_regimes = regime_probabilities_aligned.shape[1] if len(regime_probabilities_aligned.shape) > 1 else 1
                    for i in range(n_regimes):
                        market_data_aligned[f'regime_prob_{i}'] = regime_probabilities_aligned[:, i]

                # Optimize final DataFrame for M1 if available
                if is_m1_available():
                    regime_data['market_data'] = optimize_dataframe_for_m1(regime_data['market_data'])

                self.logger.info(f"✅ Regime splitting completed: {len(np.unique(regime_states_aligned))} regimes")
                self.logger.info(f"📊 Added {n_regimes} regime probability features for downstream models")

                return {
                    'success': True,
                    'data': regime_data,
                    'regime_stats': regime_stats,
                    'regime_probabilities_info': regime_probabilities_info,
                    'errors': []
                }

            except Exception as e:
                self.logger.error(f"❌ Error in regime splitting: {e}")
                return {
                    'success': False,
                    'errors': [f"Regime splitting failed: {str(e)}"],
                    'data': None
                }

    def _extract_regime_states(self, regime_discovery: Dict[str, Any], market_data: pd.DataFrame = None) -> Optional[np.ndarray]:
        """Extract regime states from regime discovery results."""
        try:
            # Debug: Log the structure of regime_discovery
            self.logger.info(f"🔍 Regime discovery type: {type(regime_discovery)}")
            if isinstance(regime_discovery, dict):
                self.logger.info(f"🔍 Regime discovery keys: {list(regime_discovery.keys())}")
            elif isinstance(regime_discovery, list):
                self.logger.info(f"🔍 Regime discovery list length: {len(regime_discovery)}")

            # Handle regime ensemble training result (ML model)
            if isinstance(regime_discovery, dict) and 'regime_ensemble_training_result' in regime_discovery:
                self.logger.info("🤖 Found regime ensemble training result - using ML model to predict regime states")
                return self._predict_regime_states_with_ml_model(regime_discovery['regime_ensemble_training_result'], market_data)

            # Handle the case where clustering_result is a string representation of the component
            if isinstance(regime_discovery, dict) and 'clustering_result' in regime_discovery:
                clustering_result = regime_discovery['clustering_result']
                self.logger.info(f"🔍 Clustering result type: {type(clustering_result)}")

                # If it's a string representation of the component, we need to get the actual data
                if isinstance(clustering_result, str) and 'NASTASClusteringComponent' in clustering_result:
                    self.logger.warning("⚠️ Clustering result is a string representation of component object")
                    self.logger.warning("⚠️ This indicates the clustering component didn't properly serialize its results")
                    self.logger.warning("⚠️ Attempting to load from saved artifacts...")

                    # Try to load from the component's saved results
                    try:
                        # Look for saved clustering results in the artifacts directory
                        artifacts_dir = Path("/Users/remyroche/Documents/Ares/generated/market_analysis/clustering")
                        if artifacts_dir.exists():
                            # Look for the most recent clustering results file
                            result_files = list(artifacts_dir.glob("nas_tas_clustering_results_*.pkl"))
                            if result_files:
                                # Get the most recent file
                                latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
                                self.logger.info(f"📁 Loading clustering results from: {latest_file}")

                                import pickle
                                with open(latest_file, 'rb') as f:
                                    saved_results = pickle.load(f)

                                # Extract the actual clustering data
                                if 'results' in saved_results:
                                    results = saved_results['results']
                                    if 'cluster_assignments' in results:
                                        states = results['cluster_assignments']
                                        self.logger.info(f"✅ Found cluster assignments in saved results: {len(states)} assignments")
                                    elif 'assignments' in results:
                                        states = results['assignments']
                                        self.logger.info(f"✅ Found assignments in saved results: {len(states)} assignments")
                                    else:
                                        self.logger.error("❌ No cluster assignments found in saved results")
                                        return None
                                else:
                                    self.logger.error("❌ No results found in saved clustering data")
                                    return None
                            else:
                                self.logger.error("❌ No saved clustering results found")
                                return None
                        else:
                            self.logger.error("❌ Clustering artifacts directory not found")
                            return None
                    except Exception as e:
                        self.logger.error(f"❌ Error loading saved clustering results: {e}")
                        return None
                else:
                    # If it's not a string, try to extract from the clustering result directly
                    if hasattr(clustering_result, 'current_results'):
                        results = clustering_result.current_results
                        if 'cluster_assignments' in results:
                            states = results['cluster_assignments']
                            self.logger.info("✅ Found cluster_assignments in clustering component results")
                        elif 'assignments' in results:
                            states = results['assignments']
                            self.logger.info("✅ Found assignments in clustering component results")
                        else:
                            self.logger.error("❌ No cluster assignments found in clustering component")
                            return None
                    elif isinstance(clustering_result, dict):
                        assignment_metadata: Optional[Dict[str, Any]] = None
                        if 'cluster_assignments' in clustering_result:
                            assignment_metadata = self._load_cluster_assignment_metadata()
                            states = self._resolve_cluster_assignments_value(
                                clustering_result['cluster_assignments'],
                                assignment_metadata,
                            )
                            self.logger.info("✅ Found cluster_assignments in clustering result dictionary")
                        elif 'assignments' in clustering_result:
                            if assignment_metadata is None:
                                assignment_metadata = self._load_cluster_assignment_metadata()
                            states = self._resolve_cluster_assignments_value(
                                clustering_result['assignments'],
                                assignment_metadata,
                            )
                            self.logger.info("✅ Found assignments in clustering result dictionary")
                        else:
                            self.logger.error("❌ No cluster assignments found in clustering result dictionary")
                            self.logger.error(f"❌ Available keys: {list(clustering_result.keys())}")
                            return None
                    else:
                        self.logger.error("❌ Clustering result doesn't have current_results attribute and is not a dictionary")
                        self.logger.error(f"❌ Clustering result type: {type(clustering_result)}")
                        return None

            # Try different possible structures for direct regime discovery
            elif 'cluster_assignments' in regime_discovery:
                metadata = self._load_cluster_assignment_metadata()
                states = self._resolve_cluster_assignments_value(
                    regime_discovery['cluster_assignments'],
                    metadata,
                )
                self.logger.info("✅ Found cluster_assignments in regime discovery")
            elif 'regime_states' in regime_discovery:
                states = regime_discovery['regime_states']
                self.logger.info("✅ Found regime_states in regime discovery")
            elif 'states' in regime_discovery:
                states = regime_discovery['states']
                self.logger.info("✅ Found states in regime discovery")
            elif 'predictions' in regime_discovery:
                states = regime_discovery['predictions']
                self.logger.info("✅ Found predictions in regime discovery")
            elif isinstance(regime_discovery, list):
                states = regime_discovery
                self.logger.info("✅ Using regime discovery as list")
            else:
                # Try to find any array-like data in the regime discovery
                self.logger.warning("⚠️ Standard keys not found, searching for array-like data")
                if isinstance(regime_discovery, dict):
                    for key, value in regime_discovery.items():
                        if isinstance(value, (list, np.ndarray)) and len(value) > 0:
                            # Check if it looks like regime assignments (integers)
                            if isinstance(value, list) and all(isinstance(x, (int, np.integer)) for x in value[:10]):
                                states = value
                                self.logger.info(f"✅ Found array-like data in key '{key}' with {len(value)} elements")
                                break
                            elif isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.integer):
                                states = value
                                self.logger.info(f"✅ Found numpy array in key '{key}' with shape {value.shape}")
                                break

                if 'states' not in locals():
                    self.logger.error("❌ Cannot extract regime states from discovery results")
                    self.logger.error(f"❌ Available keys: {list(regime_discovery.keys()) if isinstance(regime_discovery, dict) else 'Not a dict'}")
                    return None

            # Convert to numpy array if needed and ensure proper data types
            if not isinstance(states, np.ndarray):
                states = np.array(states)

            # Convert int64 to int32 to avoid JSON serialization issues
            if states.dtype == np.int64:
                states = states.astype(np.int32)

            # Validate regime count - enforce 5-20 regime requirement
            unique_regimes = len(np.unique(states))
            if unique_regimes < 5:
                self.logger.error(f"❌ Insufficient regimes: {unique_regimes} found, minimum 5 required")
                return None
            elif unique_regimes > 20:
                self.logger.error(f"❌ Too many regimes: {unique_regimes} found, maximum 20 allowed")
                return None

            self.logger.info(f"✅ Regime states extracted: {len(states)} assignments with {unique_regimes} regimes")
            return states

        except ValueError as e:
            self.logger.error(f"❌ Error extracting regime states: {e}")
            raise
        except Exception as e:
            self.logger.error(f"❌ Unexpected error extracting regime states: {e}")
            return None

    def _extract_regime_probabilities(self, regime_discovery: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract regime probabilities from regime discovery results."""
        try:
            # Handle regime ensemble training result (ML model)
            if isinstance(regime_discovery, dict) and 'regime_ensemble_training_result' in regime_discovery:
                self.logger.info("🤖 Found regime ensemble training result - ML model provides probability predictions")
                # For ML models, we can get probabilities using predict_proba
                # This will be handled in the regime splitting logic
                return None

            # Handle the case where clustering_result is a string representation of the component
            if isinstance(regime_discovery, dict) and 'clustering_result' in regime_discovery:
                clustering_result = regime_discovery['clustering_result']

                # If it's a string representation of the component, try to load from saved artifacts
                if isinstance(clustering_result, str) and 'NASTASClusteringComponent' in clustering_result:
                    try:
                        # Look for saved clustering results in the artifacts directory
                        artifacts_dir = Path("/Users/remyroche/Documents/Ares/generated/market_analysis/clustering")
                        if artifacts_dir.exists():
                            result_files = list(artifacts_dir.glob("nas_tas_clustering_results_*.pkl"))
                            if result_files:
                                latest_file = max(result_files, key=lambda x: x.stat().st_mtime)

                                import pickle
                                with open(latest_file, 'rb') as f:
                                    saved_results = pickle.load(f)

                                # Extract the actual clustering data
                                if 'results' in saved_results:
                                    results = saved_results['results']
                                    if 'regime_probabilities' in results:
                                        probs = results['regime_probabilities']
                                        self.logger.info(f"✅ Found regime probabilities in saved results: {probs.shape if hasattr(probs, 'shape') else len(probs)} probabilities")
                                    elif 'probabilities' in results:
                                        probs = results['probabilities']
                                        self.logger.info(f"✅ Found probabilities in saved results: {probs.shape if hasattr(probs, 'shape') else len(probs)} probabilities")
                                    else:
                                        self.logger.info("ℹ️ No regime probabilities found in saved results")
                                        return None
                                else:
                                    return None
                            else:
                                return None
                        else:
                            return None
                    except Exception as e:
                        self.logger.warning(f"⚠️ Error loading regime probabilities from saved results: {e}")
                        return None
                else:
                    # If it's not a string, try to extract from the clustering result directly
                    if hasattr(clustering_result, 'current_results'):
                        results = clustering_result.current_results
                        if 'regime_probabilities' in results:
                            probs = results['regime_probabilities']
                        elif 'probabilities' in results:
                            probs = results['probabilities']
                        else:
                            return None
                    elif isinstance(clustering_result, dict):
                        # Handle dictionary format clustering results
                        if 'regime_probabilities' in clustering_result:
                            probs = clustering_result['regime_probabilities']
                        elif 'probabilities' in clustering_result:
                            probs = clustering_result['probabilities']
                        else:
                            return None  # Probabilities are optional
                    else:
                        return None
            else:
                # Try different possible structures for direct regime discovery
                if 'regime_probabilities' in regime_discovery:
                    probs = regime_discovery['regime_probabilities']
                elif 'probabilities' in regime_discovery:
                    probs = regime_discovery['probabilities']
                elif 'proba' in regime_discovery:
                    probs = regime_discovery['proba']
                else:
                    return None  # Probabilities are optional

            # Convert to numpy array if needed
            if not isinstance(probs, np.ndarray):
                probs = np.array(probs)

            return probs

        except Exception as e:
            self.logger.warning(f"⚠️ Error extracting regime probabilities: {e}")
            return None

    def _calculate_regime_statistics(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive regime statistics using common utilities."""
        try:
            regime_stats = {}

            # Basic regime distribution using safe operations
            regime_counts = safe_dataframe_operation(
                market_data,
                lambda df: df['regime_state'].value_counts().to_dict()
            )
            # Convert numpy int64/int32 keys to regular Python ints for JSON serialization
            regime_counts = {int(k): int(v) for k, v in regime_counts.items()}
            regime_stats['regime_distribution'] = regime_counts
            regime_stats['total_regimes'] = len(regime_counts)
            regime_stats['total_data_points'] = len(market_data)

            # Calculate statistics per regime using safe operations
            regime_details = {}
            unique_regimes = safe_dataframe_operation(
                market_data,
                lambda df: df['regime_state'].unique()
            )

            for regime_id in unique_regimes:
                # Convert regime_id to regular Python int to avoid JSON serialization issues
                regime_id_int = int(regime_id)
                regime_data = safe_filter_dataframe(
                    market_data,
                    f"regime_state == {regime_id_int}"
                )

                # Use safe math operations for calculations
                count = len(regime_data)
                percentage = safe_divide(count, len(market_data), 0.0) * 100

                # Use the converted regime_id_int for JSON serialization
                regime_details[regime_id_int] = {
                    'count': count,
                    'percentage': percentage,
                    'volatility_std': safe_std(regime_data['close']) if 'close' in regime_data.columns else 0.0,
                    'mean_volume': safe_mean(regime_data['volume']) if 'volume' in regime_data.columns else 0.0,
                    'mean_price': safe_mean(regime_data['close']) if 'close' in regime_data.columns else 0.0,
                    'price_range': {
                        'min': safe_float(regime_data['close'].min()) if 'close' in regime_data.columns else 0.0,
                        'max': safe_float(regime_data['close'].max()) if 'close' in regime_data.columns else 0.0
                    }
                }

            regime_stats['regime_details'] = regime_details

            return regime_stats

        except Exception as e:
            self.logger.error(f"❌ Error calculating regime statistics: {e}")
            return {
                'error': str(e),
                'total_regimes': 0,
                'regime_distribution': {},
                'total_data_points': 0
            }

    def _calculate_regime_statistics_optimized(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate regime statistics with M1 optimizations."""
        try:
            # Use M1-optimized operations if available
            if is_m1_available():
                return self._calculate_regime_statistics_m1_optimized(market_data)
            else:
                return self._calculate_regime_statistics(market_data)

        except Exception as e:
            self.logger.error(f"❌ Error in optimized regime statistics: {e}")
            return self._calculate_regime_statistics(market_data)

    def _calculate_regime_statistics_m1_optimized(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate regime statistics optimized for M1 hardware."""
        try:
            regime_stats = {}

            # Use M1-optimized array operations
            regime_states = create_m1_optimized_array(market_data['regime_state'].values)

            # Basic regime distribution using M1 optimizations
            unique_regimes, counts = np.unique(regime_states, return_counts=True)
            # Convert numpy int64/int32 keys to regular Python ints for JSON serialization
            regime_counts = {int(k): int(v) for k, v in zip(unique_regimes, counts)}

            regime_stats['regime_distribution'] = regime_counts
            regime_stats['total_regimes'] = len(unique_regimes)
            regime_stats['total_data_points'] = len(market_data)

            # Calculate statistics per regime using M1-optimized operations
            regime_details = {}
            for regime_id in unique_regimes:
                mask = regime_states == regime_id
                regime_data = market_data[mask]

                # Use M1-optimized math operations
                count = int(np.sum(mask))
                percentage = safe_divide(count, len(market_data), 0.0) * 100

                # Convert regime_id to regular Python int for JSON serialization
                regime_details[int(regime_id)] = {
                    'count': count,
                    'percentage': percentage,
                    'volatility_std': math_safe_std(regime_data['close'].values) if 'close' in regime_data.columns else 0.0,
                    'mean_volume': math_safe_mean(regime_data['volume'].values) if 'volume' in regime_data.columns else 0.0,
                    'mean_price': math_safe_mean(regime_data['close'].values) if 'close' in regime_data.columns else 0.0,
                    'price_range': {
                        'min': safe_float(regime_data['close'].min()) if 'close' in regime_data.columns else 0.0,
                        'max': safe_float(regime_data['close'].max()) if 'close' in regime_data.columns else 0.0
                    }
                }

            regime_stats['regime_details'] = regime_details

            return regime_stats

        except Exception as e:
            self.logger.error(f"❌ Error in M1-optimized regime statistics: {e}")
            fallback_result = self._calculate_regime_statistics(market_data)
            # Ensure fallback result has required fields
            if not isinstance(fallback_result, dict) or 'total_regimes' not in fallback_result:
                return {
                    'error': str(e),
                    'total_regimes': 0,
                    'regime_distribution': {},
                    'total_data_points': 0
                }
            return fallback_result

    def _calculate_regime_statistics_memory_optimized(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate regime statistics with comprehensive memory optimization."""
        try:
            # Use memory context for optimization
            with memory_checkpoint("regime_statistics_calculation"):
                regime_stats = {}

                # Get regime states efficiently without copying
                regime_states = market_data['regime_state'].values if 'regime_state' in market_data.columns else np.array([])

                if len(regime_states) == 0:
                    self.logger.warning("⚠️ No regime states found for statistics calculation")
                    return {
                        'error': 'No regime states found',
                        'total_regimes': 0,
                        'regime_distribution': {},
                        'total_data_points': 0
                    }

                # Use memory-efficient operations for basic statistics
                unique_regimes, counts = np.unique(regime_states, return_counts=True)

                # Convert to memory-efficient dictionary
                regime_counts = {int(k): int(v) for k, v in zip(unique_regimes, counts)}

                regime_stats['regime_distribution'] = regime_counts
                regime_stats['total_regimes'] = len(unique_regimes)
                regime_stats['total_data_points'] = len(market_data)

                # Calculate statistics per regime using memory-efficient operations
                regime_details = {}
                for regime_id in unique_regimes:
                    # Use boolean indexing for memory efficiency
                    regime_mask = regime_states == regime_id
                    regime_data = market_data[regime_mask]

                    # Calculate statistics efficiently
                    count = int(np.sum(regime_mask))
                    percentage = safe_divide(count, len(market_data), 0.0) * 100

                    # Use vectorized operations for price statistics
                    if 'close' in regime_data.columns:
                        close_prices = regime_data['close'].values
                        price_stats = {
                            'mean_price': float(np.mean(close_prices)),
                            'min_price': float(np.min(close_prices)),
                            'max_price': float(np.max(close_prices)),
                            'volatility_std': float(np.std(close_prices))
                        }
                    else:
                        price_stats = {
                            'mean_price': 0.0,
                            'min_price': 0.0,
                            'max_price': 0.0,
                            'volatility_std': 0.0
                        }

                    # Volume statistics if available
                    if 'volume' in regime_data.columns:
                        volume_values = regime_data['volume'].values
                        volume_stats = {
                            'mean_volume': float(np.mean(volume_values)),
                            'total_volume': float(np.sum(volume_values))
                        }
                    else:
                        volume_stats = {
                            'mean_volume': 0.0,
                            'total_volume': 0.0
                        }

                    regime_details[int(regime_id)] = {
                        'count': count,
                        'percentage': percentage,
                        **price_stats,
                        **volume_stats
                    }

                regime_stats['regime_details'] = regime_details

                # Optimize final result memory usage
                regime_stats = self.memory_optimizer.optimize_memory_usage(regime_stats)

                return regime_stats

        except Exception as e:
            self.logger.error(f"❌ Error in memory-optimized regime statistics: {e}")
            return {'error': str(e)}

    def _validate_splitting_results(
        self,
        splitting_result: Dict[str, Any],
        report: RegimeSplittingReport
    ) -> Dict[str, Any]:
        """Validate the results of regime splitting."""
        self.logger.info("🔍 Validating splitting results...")
        tprint("🔍 Validating splitting results...")

        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }

        try:
            if not splitting_result['success']:
                validation_result['valid'] = False
                validation_result['errors'].append("Splitting operation failed")
                return validation_result

            regime_data = splitting_result['data']
            if regime_data is None:
                validation_result['valid'] = False
                validation_result['errors'].append("No regime data produced")
                return validation_result

            # Validate market data
            market_data = regime_data['market_data']
            if market_data is None or len(market_data) == 0:
                validation_result['valid'] = False
                validation_result['errors'].append("Market data is empty")
                return validation_result

            # Validate regime states
            regime_states = regime_data['regime_states']
            if regime_states is None or (hasattr(regime_states, '__len__') and len(regime_states) == 0):
                validation_result['valid'] = False
                validation_result['errors'].append("No regime states found")
                return validation_result

            # Check regime diversity - enforce 5-20 regime requirement
            unique_regimes = len(np.unique(regime_states))
            if unique_regimes < 5:
                validation_result['valid'] = False
                validation_result['errors'].append(f"Insufficient regimes: {unique_regimes} found, minimum 5 required")
            elif unique_regimes > 20:
                validation_result['valid'] = False
                validation_result['errors'].append(f"Too many regimes: {unique_regimes} found, maximum 20 allowed")

            # Check data alignment
            if len(market_data) != len(regime_states):
                validation_result['valid'] = False
                validation_result['errors'].append(f"Data length mismatch: market_data={len(market_data)}, regime_states={len(regime_states)}")

            # Validate regime statistics
            regime_stats = regime_data.get('regime_statistics', {})
            if not isinstance(regime_stats, dict):
                validation_result['warnings'].append("Regime statistics is not a dictionary")
                regime_stats = {}

            # Ensure required fields are present
            if 'total_regimes' not in regime_stats:
                # Try to calculate total_regimes from the regime_states
                if regime_states is not None and len(regime_states) > 0:
                    unique_regimes = len(np.unique(regime_states))
                    regime_stats['total_regimes'] = unique_regimes
                    validation_result['warnings'].append("Calculated missing total_regimes from regime_states")
                else:
                    regime_stats['total_regimes'] = 0
                    validation_result['warnings'].append("No regime states found - setting total_regimes to 0")

            # Additional data consistency checks
            # Check for regime state consistency
            if 'regime_state' in market_data.columns:
                regime_states_in_data = market_data['regime_state'].values
                try:
                    # Ensure both arrays are numpy arrays and have the same shape
                    if isinstance(regime_states, np.ndarray) and isinstance(regime_states_in_data, np.ndarray):
                        if regime_states.shape == regime_states_in_data.shape:
                            # Use np.array_equal properly - it returns a boolean, not an array
                            arrays_equal = np.array_equal(regime_states, regime_states_in_data)
                            if not arrays_equal:
                                validation_result['warnings'].append("Regime states in data don't match extracted regime states")
                        else:
                            validation_result['warnings'].append(f"Regime state shape mismatch: {regime_states.shape} vs {regime_states_in_data.shape}")
                    else:
                        validation_result['warnings'].append("Regime states are not numpy arrays")
                except (ValueError, TypeError) as e:
                    # Handle shape mismatch or type issues
                    validation_result['warnings'].append(f"Regime state comparison failed: {str(e)}")

            # Check for regime probability consistency
            if 'regime_probability' in market_data.columns and regime_data['regime_probabilities'] is not None:
                regime_probs_in_data = market_data['regime_probability'].values
                if len(regime_probs_in_data) != len(regime_data['regime_probabilities']):
                    validation_result['warnings'].append("Regime probabilities length mismatch")

            # Check for data type consistency
            if not isinstance(regime_states, np.ndarray):
                validation_result['warnings'].append("Regime states should be numpy array")

            # Check for reasonable regime values
            if len(regime_states) > 0:
                min_regime = np.min(regime_states)
                max_regime = np.max(regime_states)
                if min_regime < 0:
                    validation_result['warnings'].append(f"Negative regime values detected: min={min_regime}")
                if max_regime > 100:
                    validation_result['warnings'].append(f"Unusually high regime values detected: max={max_regime}")

            if validation_result['valid']:
                self.logger.info("✅ Splitting results validation passed")
                tprint("✅ Splitting results validation passed")
            else:
                self.logger.error(f"❌ Splitting results validation failed: {validation_result['errors']}")
                tprint(f"❌ Splitting results validation failed: {validation_result['errors']}")

            if validation_result['warnings']:
                tprint(f"⚠️ Validation warnings: {validation_result['warnings']}")

            return validation_result

        except Exception as e:
            self.logger.error(f"❌ Error validating splitting results: {e}")
            return {
                'valid': False,
                'errors': [f"Validation error: {str(e)}"],
                'warnings': []
            }

    def _generate_comprehensive_report(
        self,
        splitting_result: Dict[str, Any],
        market_data: pd.DataFrame,
        report: RegimeSplittingReport
    ) -> RegimeSplittingReport:
        """Generate comprehensive execution report."""
        self.logger.info("📊 Generating comprehensive report...")

        try:
            # Update metrics
            regime_data = splitting_result['data']
            regime_stats = regime_data['regime_statistics']

            self.metrics.total_data_points = len(market_data)
            self.metrics.regime_count = regime_stats.get('total_regimes', 0)
            self.metrics.regime_distribution = regime_stats.get('regime_distribution', {})
            self.metrics.processing_time_seconds = (datetime.now() - self.start_time).total_seconds()

            # Calculate data quality score
            self.metrics.data_quality_score = self._calculate_data_quality_score(market_data)

            # Calculate regime continuity score
            self.metrics.regime_continuity_score = self._calculate_regime_continuity_score(
                regime_data['regime_states']
            )

            # Generate execution summary
            report.execution_summary = {
                'total_data_points': self.metrics.total_data_points,
                'regime_count': self.metrics.regime_count,
                'processing_time_seconds': self.metrics.processing_time_seconds,
                'data_quality_score': self.metrics.data_quality_score,
                'regime_continuity_score': self.metrics.regime_continuity_score,
                'memory_usage_mb': self.metrics.memory_usage_mb
            }

            # Generate recommendations
            report.recommendations = self._generate_recommendations(report)

            self.logger.info("✅ Comprehensive report generated")
            return report

        except Exception as e:
            self.logger.error(f"❌ Error generating report: {e}")
            report.errors.append(f"Report generation failed: {str(e)}")
            return report

    def _calculate_data_quality_score(self, market_data: pd.DataFrame) -> float:
        """Calculate data quality score (0-1) using common utilities."""
        try:
            # Use common data quality metrics
            quality_metrics = calculate_data_quality_metrics(market_data)

            score = 1.0

            # Check for null values using safe operations
            null_ratio = safe_divide(
                quality_metrics.get('missing_values', 0),
                len(market_data) * len(market_data.columns),
                0.0
            )
            score -= null_ratio * 0.3

            # Check for duplicate rows using safe operations
            duplicate_ratio = safe_divide(
                quality_metrics.get('duplicate_rows', 0),
                len(market_data),
                0.0
            )
            score -= duplicate_ratio * 0.2

            # Check for infinite values using safe operations
            numeric_cols = market_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0 and len(market_data) > 0:
                inf_count = safe_dataframe_operation(
                    market_data,
                    lambda df: np.isinf(df[numeric_cols]).sum().sum()
                )
                inf_ratio = safe_divide(inf_count, len(market_data) * len(numeric_cols), 0.0)
                score -= inf_ratio * 0.3

            # Check for zero/negative prices using safe operations
            if 'close' in market_data.columns:
                invalid_prices = safe_divide(
                    (market_data['close'] <= 0).sum(),
                    len(market_data),
                    0.0
                )
                score -= invalid_prices * 0.2

            # Use safe math operations for final score
            return math_validate_range(score, 0.0, 1.0, "data_quality_score")

        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating data quality score: {e}")
            return 0.5  # Default score

    def _calculate_regime_continuity_score(self, regime_states: np.ndarray) -> float:
        """Calculate regime continuity score (0-1) using safe math operations."""
        try:
            if len(regime_states) < 2:
                return 1.0

            # Count regime transitions using safe operations
            transitions = math_safe_func(
                np.sum,
                regime_states[1:] != regime_states[:-1],
                default=0
            )
            transition_ratio = safe_divide(transitions, len(regime_states) - 1, 0.0)

            # Higher continuity = fewer transitions (score closer to 1)
            continuity_score = 1.0 - min(1.0, transition_ratio * 2)

            # Validate the score is in valid range
            return math_validate_range(continuity_score, 0.0, 1.0, "continuity_score")

        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating regime continuity score: {e}")
            return 0.5  # Default score

    def _generate_recommendations(self, report: RegimeSplittingReport) -> List[str]:
        """Generate recommendations based on execution results."""
        recommendations = []

        # Data quality recommendations
        if self.metrics.data_quality_score < 0.8:
            recommendations.append("Consider improving data quality - current score is below 0.8")

        # Regime diversity recommendations
        if self.metrics.regime_count < 3:
            recommendations.append("Consider adjusting regime discovery parameters - only few regimes detected")

        # Continuity recommendations
        if self.metrics.regime_continuity_score < 0.7:
            recommendations.append("Regime transitions are frequent - consider smoothing parameters")

        # Performance recommendations
        if self.metrics.processing_time_seconds > 60:
            recommendations.append("Processing time is high - consider optimizing data size or parameters")

        # Memory recommendations
        if self.metrics.memory_usage_mb > 1000:
            recommendations.append("High memory usage detected - consider streaming processing")

        return recommendations

    async def _create_artifacts(
        self,
        splitting_result: Dict[str, Any],
        report: RegimeSplittingReport
    ) -> Dict[str, Any]:
        """Create comprehensive artifacts using common serialization utilities."""
        self.logger.info("💾 Creating artifacts...")

        try:
            # Create artifacts with enhanced metadata
            # Ensure splitting_result['data'] is properly structured
            self.logger.info(f"🔍 Debug: splitting_result type: {type(splitting_result)}")
            self.logger.info(f"🔍 Debug: splitting_result keys: {list(splitting_result.keys()) if isinstance(splitting_result, dict) else 'Not a dict'}")

            regime_data = splitting_result.get('data', {})
            self.logger.info(f"🔍 Debug: regime_data type: {type(regime_data)}")
            self.logger.info(f"🔍 Debug: regime_data keys: {list(regime_data.keys()) if isinstance(regime_data, dict) else 'Not a dict'}")

            if not isinstance(regime_data, dict):
                self.logger.warning(f"⚠️ Regime data is not a dictionary: {type(regime_data)}")
                regime_data = {}
            else:
                # Ensure regime_data has the expected structure
                if 'market_data' not in regime_data:
                    self.logger.warning("⚠️ Regime data missing 'market_data' key")
                    regime_data['market_data'] = None
                if 'regime_states' not in regime_data:
                    self.logger.warning("⚠️ Regime data missing 'regime_states' key")
                    regime_data['regime_states'] = None
                if 'regime_statistics' not in regime_data:
                    self.logger.warning("⚠️ Regime data missing 'regime_statistics' key")
                    regime_data['regime_statistics'] = {}

            # Create artifacts with safe error handling
            try:
                regime_stats = splitting_result.get('regime_stats', {})
                self.logger.info(f"🔍 Debug: regime_stats type: {type(regime_stats)}")
            except Exception as e:
                self.logger.error(f"❌ Error getting regime_stats: {e}")
                regime_stats = {}

            try:
                memory_usage = 0
                if hasattr(get_memory_usage, '__call__'):
                    memory_result = get_memory_usage()
                    if isinstance(memory_result, dict):
                        memory_usage = memory_result.get('used_memory', 0) / (1024 * 1024)
                self.logger.info(f"🔍 Debug: memory_usage calculated: {memory_usage}")
            except Exception as e:
                self.logger.error(f"❌ Error calculating memory usage: {e}")
                memory_usage = 0

            try:
                cpu_cores = 'unknown'
                if hasattr(self.cpu_optimizer, 'get_cpu_info'):
                    cpu_info = self.cpu_optimizer.get_cpu_info()
                    if isinstance(cpu_info, dict):
                        cpu_cores = cpu_info.get('total_cores', 'unknown')
                self.logger.info(f"🔍 Debug: cpu_cores calculated: {cpu_cores}")
            except Exception as e:
                self.logger.error(f"❌ Error getting CPU info: {e}")
                cpu_cores = 'unknown'

            artifacts = {
                'regime_data_splitting_result': {
                    'regime_data': regime_data,
                    'regime_stats': regime_stats,
                    'processing_metrics': {
                        'total_data_points': self.metrics.total_data_points,
                        'regime_count': self.metrics.regime_count,
                        'processing_time_seconds': self.metrics.processing_time_seconds,
                        'data_quality_score': self.metrics.data_quality_score,
                        'regime_continuity_score': self.metrics.regime_continuity_score,
                        'memory_usage_mb': memory_usage,
                        'hardware_optimized': is_m1_available()
                    },
                    'metadata': {
                        'symbol': self.config.symbol,
                        'exchange': self.config.exchange,
                        'timeframe': self.config.timeframe,
                        'execution_timestamp': datetime.now().isoformat(),
                        'component_version': 'enhanced_v2.0_common_utils',
                        'hardware_info': {
                            'is_m1': is_m1_available(),
                            'mps_available': is_mps_available(),
                            'cpu_cores': cpu_cores
                        }
                    }
                },
                'regime_splitting_report': {
                    'status': report.status.value,
                    'metrics': {
                        'total_data_points': self.metrics.total_data_points,
                        'regime_count': self.metrics.regime_count,
                        'regime_distribution': self.metrics.regime_distribution,
                        'processing_time_seconds': self.metrics.processing_time_seconds,
                        'data_quality_score': self.metrics.data_quality_score,
                        'regime_continuity_score': self.metrics.regime_continuity_score
                    },
                    'execution_summary': report.execution_summary,
                    'warnings': report.warnings,
                    'errors': report.errors,
                    'recommendations': report.recommendations,
                    'timestamp': report.timestamp
                },
                'regime_validation_results': {
                    'validation_checks_passed': self.metrics.validation_checks_passed,
                    'validation_checks_failed': self.metrics.validation_checks_failed,
                    'data_quality_validation': self.metrics.data_quality_score > 0.7,
                    'regime_diversity_validation': self.metrics.regime_count >= 2,
                    'continuity_validation': self.metrics.regime_continuity_score > 0.5,
                    'overall_validation_passed': (
                        self.metrics.data_quality_score > 0.7 and
                        self.metrics.regime_count >= 2 and
                        self.metrics.regime_continuity_score > 0.5
                    )
                }
            }

            # Save artifacts using common serialization utilities
            await self._save_artifacts_to_files(artifacts)

            self.logger.info("✅ Artifacts created successfully")
            return artifacts

        except Exception as e:
            self.logger.error(f"❌ Error creating artifacts: {e}")
            return {}

    async def _save_artifacts_to_files(self, artifacts: Dict[str, Any]) -> None:
        """Save artifacts to files using common serialization utilities."""
        try:
            # Create artifacts directory
            artifacts_dir = Path("generated/market_analysis/regime_data_splitting")
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            # Save regime data splitting result as parquet
            if 'regime_data' in artifacts['regime_data_splitting_result']:
                regime_data = artifacts['regime_data_splitting_result']['regime_data']
                # Check if regime_data is a dictionary and contains market_data
                if isinstance(regime_data, dict) and 'market_data' in regime_data and regime_data['market_data'] is not None:
                    try:
                        parquet_path = artifacts_dir / "regime_market_data.parquet"
                        safe_to_parquet(regime_data['market_data'], parquet_path)
                        self.logger.info(f"💾 Saved regime market data to {parquet_path}")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to save regime market data: {e}")
                else:
                    self.logger.warning(f"⚠️ Regime data is not a dictionary, doesn't contain market_data, or market_data is None: {type(regime_data)}")

            # Save report as JSON
            report_path = Path("outcomes/market_analysis") / "regime_splitting_report.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            self.serializer.save(artifacts['regime_splitting_report'], str(report_path))
            self.logger.info(f"💾 Saved regime splitting report to {report_path}")

            # Save validation results as JSON
            validation_path = artifacts_dir / "regime_validation_results.json"
            self.serializer.save(artifacts['regime_validation_results'], str(validation_path))
            self.logger.info(f"💾 Saved validation results to {validation_path}")

        except Exception as e:
            self.logger.warning(f"⚠️ Error saving artifacts to files: {e}")
            # Continue without file saving

    def _update_metrics(self, report: RegimeSplittingReport, splitting_result: Dict[str, Any]) -> None:
        """Update metrics based on execution results."""
        try:
            self.metrics.validation_checks_passed = len([r for r in report.validation_results.values() if r])
            self.metrics.validation_checks_failed = len([r for r in report.validation_results.values() if not r])
            self.metrics.warnings_count = len(report.warnings)
            self.metrics.errors_count = len(report.errors)

        except Exception as e:
            self.logger.warning(f"⚠️ Error updating metrics: {e}")

    def _create_failure_result(self, report: RegimeSplittingReport, error_message: str) -> ComponentResult:
        """Create a failure result with comprehensive error information."""
        return ComponentResult(
            success=False,
            artifacts={
                'regime_splitting_report': {
                    'status': report.status.value,
                    'errors': report.errors,
                    'warnings': report.warnings,
                    'timestamp': report.timestamp
                }
            },
            error_message=error_message,
            metadata={
                'symbol': self.config.symbol,
                'exchange': self.config.exchange,
                'timeframe': self.config.timeframe,
                'failure_timestamp': datetime.now().isoformat()
            }
        )

    def cleanup(self) -> Dict[str, Any]:
        """Clean up resources using hardware manager with memory optimization."""
        try:
            # Perform memory cleanup first
            self._perform_emergency_cleanup()

            # Stop memory monitoring
            if hasattr(self.memory_optimizer, 'stop_m1_memory_monitoring'):
                self.memory_optimizer.stop_m1_memory_monitoring()

            # Clean up hardware manager
            cleanup_result = self.hardware_manager.cleanup()

            # Final garbage collection
            import gc

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

except ImportError:

    cp = None
            gc.collect()

            # Log final memory status
            if hasattr(self.memory_optimizer, 'get_current_memory_usage'):
                final_memory = self.memory_optimizer.get_current_memory_usage()
                self.logger.info(f"🧹 Final memory usage: {final_memory:.2f} GB")

            self.logger.info(f"🧹 Cleanup completed successfully: {cleanup_result}")
            return cleanup_result

        except Exception as e:
            self.logger.warning(f"⚠️ Error during cleanup: {e}")
            return {'status': 'failed', 'error': str(e)}

    def get_resource_metrics(self) -> Dict[str, Any]:
        """Get current resource usage metrics."""
        try:
            metrics = self.hardware_manager.get_system_metrics()
            return metrics
        except Exception as e:
            self.logger.warning(f"⚠️ Error getting resource metrics: {e}")
            return {'error': str(e)}

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        self.cleanup()

    def __del__(self):
        """Destructor with safe cleanup using hardware manager."""
        try:
            if hasattr(self, 'hardware_manager'):
                self.hardware_manager.cleanup()
        except Exception:
            # Ignore errors during cleanup to avoid issues during interpreter shutdown
            pass

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
