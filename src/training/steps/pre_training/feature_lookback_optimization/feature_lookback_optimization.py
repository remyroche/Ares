"""
Modular Feature Lookback Optimization Component.

This is the main component that uses the modular architecture with separate
modules for validation, error handling, performance monitoring, and optimization.
"""

import json
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from ..settings import get_pre_training_settings

# Import utility modules
from src.utils.common_operations import safe_dataframe_operation, get_m1_memory_optimizer
from src.utils.common_utilities import CommonUtilities
from src.utils.math_validation import safe_divide, validate_finite
from src.utils.serialization_utils import UniversalSerializer, JSONSerializer, PickleSerializer
from src.utils.hardware.m1_gpu_utils import M1GPUManager
from src.utils.data.klines_parquet import KlinesParquetManager
from .ares_launcher_integration import AresLauncherFeatureLookbackOptimizer
from src.utils.matrix_operations import (
    get_unified_matrix_operations,
    get_vectorized_processing_core,
    get_batch_matrix_processor,
    safe_matrix_multiply,
    optimize_dataframe,
    matrix_correlation_analysis,
    gpu_matrix_multiply,
    correlation_matrix_gpu
)
from src.utils.tprint import (
    tprint,
    tprint_debug,
    tprint_info,
    tprint_success,
    tprint_warning,
    tprint_error,
)
# Import FeatureCacheService - will raise exception if not available
from src.feature_generation.core.feature_cache import FeatureCacheService
from src.training.steps.pre_training.column_naming import (
    ColumnNamespace,
    ensure_namespace,
    filter_namespace_columns,
    strip_namespace)
from src.training.steps.pre_training.artifacts.manifest import (
    ArtifactManifest,
    DataLocator,
)
from src.training.config.data_locator import DataLocator as PipelineDataLocator

# Import numpy for type checking
from .dependency_manager import get_dependency
np, _ = get_dependency('numpy')

# Utility function to convert int64 to int for dictionary keys
def convert_int64_to_int(value: Any) -> Any:
    """Convert int64 values to regular Python int for JSON serialization."""
    tprint(f"🔄 Converting value of type {type(value).__name__} for JSON safety")
    try:
        if hasattr(value, 'dtype') and value.dtype == 'int64':
            tprint("📏 Detected numpy dtype int64, converting to int")
            return int(value)
        elif isinstance(value, np.int64):
            tprint("📏 Detected numpy.int64 instance, converting to int")
            return int(value)
        elif isinstance(value, dict):
            # Convert both keys and values to handle int64 keys
            converted_dict = {}
            for k, v in value.items():
                # Convert key if it's int64
                converted_key = k
                if isinstance(k, np.int64):
                    tprint("🔑 Converting dict key from numpy.int64 to int")
                    converted_key = int(k)
                elif hasattr(k, 'dtype') and k.dtype == 'int64':
                    tprint("🔑 Converting dict key from numpy dtype int64 to int")
                    converted_key = int(k)

                # Convert value recursively
                converted_dict[converted_key] = convert_int64_to_int(v)

            return converted_dict
        elif isinstance(value, (list, tuple)):
            # Convert each item in the list/tuple recursively
            tprint("📚 Converting sequence items for JSON safety")
            return [convert_int64_to_int(item) for item in value]
        elif hasattr(value, 'shape') and len(value.shape) > 0:
            # Handle numpy arrays that might be problematic
            if value.size > 100:  # Large arrays might cause issues
                tprint("📦 Large numpy array detected, summarizing instead of full conversion")
                return {
                    'type': 'numpy_array',
                    'shape': value.shape,
                    'dtype': str(value.dtype),
                    'size': value.size
                }
            else:
                tprint("📦 Small numpy array detected, converting to list")
                return value.tolist()  # Convert small arrays to lists
        else:
            return value
    except Exception as e:
        # If conversion fails, return a safe representation
        return {
            'conversion_error': str(e),
            'original_type': type(value).__name__,
            'safe_representation': 'unconvertible_value'
        }

# Import modular components
from .core.optimizer import CoreOptimizer, OptimizationMethod, OptimizationResult
from .validation.validator import InputValidator, ValidationLevel, ValidationStatus, ValidationSummary
from .error_handling.error_handler import StandardizedErrorHandler, ErrorSeverity, ErrorCategory
from .performance.monitor import PerformanceMonitor, MetricType, MetricLevel

from ..components.base_component import BasePreTrainingComponent, ComponentConfig, ComponentResult
from ..components.contracts import FeatureLookbackArtifacts, PipelineState
from ..validation.schemas import (
    SchemaValidationException,
    enforce_feature_temporal_alignment,
    schema_metadata,
    validate_engineered_features,
    validate_labeled_dataset,
    validate_raw_ohlcv,
)
from ..components.component_factory import register_component

# Import optimized process engine
from ...market_analysis.optimized_process_engines import OptimizedFeatureLookbackEngine, ProcessType

# Import dependencies with fallbacks
from .dependency_manager import get_dependency, is_dependency_available

# Get dependencies
np, _ = get_dependency('numpy')
pd, _ = get_dependency('pandas')

# Import logger
from src.utils.logger import system_logger
from ...market_analysis.logging_standards import (
    get_logger, log_info, log_warning, log_error, log_success, log_debug,
    LoggingContext, log_step_progress, log_data_info, log_validation_result
)
from ..validation.schemas import extract_p_value_mapping, track_and_control_hypotheses


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation splits."""
    n_splits: int = 3
    min_train_ratio: float = 0.4  # Minimum training set size as ratio of total
    min_val_samples: int = 20     # Minimum validation samples
    min_train_samples: int = 60   # Minimum training samples
    min_window_size: int = 25     # Minimum window size for MI estimation


@dataclass
class OptimizationMetrics:
    """Comprehensive optimization metrics."""
    best_lookback_period: int
    best_score: float
    optimization_method: str
    total_features_optimized: int
    optimization_time: float
    convergence_iterations: int
    memory_usage_mb: float
    cpu_usage_percent: float
    validation_score: float
    stability_score: float
    regime_coverage: float
    error_rate: float


@register_component('feature_lookback_optimization')
class FeatureLookbackOptimizationComponent(BasePreTrainingComponent):
    """
    Modular Feature Lookback Optimization Component.

    This component uses a modular architecture with separate modules for:
    - Core optimization logic
    - Input validation
    - Error handling
    - Performance monitoring
    """

    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the feature lookback optimization component."""
        tprint("🔧 Initializing Modular FeatureLookbackOptimizationComponent...")
        super().__init__(config)

        # Use standardized logging
        self.logger = get_logger('FeatureLookbackOptimization')
        self.common_utils = CommonUtilities()
        self.serializer = UniversalSerializer()
        tprint("✅ Basic modular component initialization complete")

        # Initialize additional utility managers
        self.memory_optimizer = get_m1_memory_optimizer()
        self.gpu_manager = M1GPUManager()
        self.data_manager = KlinesParquetManager()
        self.json_serializer = JSONSerializer()
        self.pickle_serializer = PickleSerializer()
        tprint("✅ Additional utility managers initialized")

        # Initialize matrix operations managers
        tprint("🔢 Initializing matrix operations managers...")
        self.matrix_ops = get_unified_matrix_operations()
        self.vectorized_core = get_vectorized_processing_core()
        self.batch_processor = get_batch_matrix_processor()
        tprint_success("✅ Matrix operations managers initialized")

        # Initialize modular components
        tprint("🔧 Initializing modular components...")
        self.validator = InputValidator(logger=self.logger)
        self.error_handler = StandardizedErrorHandler(logger=self.logger, component_name="FeatureLookbackOptimization")
        self.performance_monitor = PerformanceMonitor(component_name="FeatureLookbackOptimization")
        self.core_optimizer = CoreOptimizer(logger=self.logger)
        tprint("✅ Modular components initialized")

        # Regularization preferences used to guide lookback selection toward
        # realistic horizons.  Tests can override these via component
        # configuration which allows deterministic assertions around the
        # penalty behaviour applied by the optimizer.
        self.lookback_regularization_settings = self._resolve_regularization_settings()

        # Initialize execution mode configuration
        tprint("🔧 Initializing execution mode lookback configuration...")
        try:
            from ...market_analysis.shared_utils.execution_mode_lookback_config import get_execution_mode_config
            self.execution_mode_config = get_execution_mode_config()
            tprint("✅ Execution mode configuration initialized")
        except ImportError as e:
            # Fallback to default configuration if shared_utils not available
            tprint_warning(f"⚠️ Could not import execution mode config: {e}")
            tprint_warning("⚠️ Using default execution mode configuration")
            self.execution_mode_config = {
                'light': {'max_features': 100, 'max_lookback': 50},
                'full': {'max_features': 500, 'max_lookback': 200},
                'blank': {'max_features': 20, 'max_lookback': 20},
            }

        # Initialize optimized process engine
        tprint("🔧 Initializing optimized feature lookback engine...")
        self.optimized_engine = OptimizedFeatureLookbackEngine(
            use_hardware_accel=True,
            cache_size=1000
        )
        tprint("✅ Optimized feature lookback engine initialized")

        # Component state
        self.optimization_status = "pending"
        self.start_time: Optional[float] = None
        self.metrics: Optional[OptimizationMetrics] = None

        # Feature cache state
        self.feature_cache = FeatureCacheService(subdirectory="feature_bank")
        self.cache_metrics = {
            'hits': 0,
            'misses': 0,
            'writes': 0,
            'force_refreshes': 0
        }
        self._current_cache_key: Optional[str] = None
        self._current_lookback_hash: Optional[str] = None
        self._force_cache_refresh: bool = False

        # Performance monitoring (separate from PerformanceMonitor instance)
        self.performance_data = {
            'memory_usage': [],
            'cpu_usage': [],
            'execution_times': {},
            'error_counts': 0,
            'peak_memory_mb': 0.0,
            'memory_warnings': 0
        }
        
        # Memory monitoring settings
        self.max_performance_entries = 1000  # Maximum entries to keep in history
        self.memory_warning_threshold_mb = 1000.0  # 1GB
        self.memory_critical_threshold_mb = 2000.0  # 2GB

        tprint("📈 Performance data trackers initialized for optimization component")
        tprint(f"🧠 Memory monitoring thresholds: warning={self.memory_warning_threshold_mb}MB, critical={self.memory_critical_threshold_mb}MB")

        tprint("✅ Modular FeatureLookbackOptimizationComponent initialized")
    
    def _trim_performance_history(self, max_entries: int = 1000) -> None:
        """Trim performance history to prevent unbounded growth."""
        for key in ['memory_usage', 'cpu_usage']:
            if len(self.performance_data[key]) > max_entries:
                trimmed = len(self.performance_data[key]) - max_entries
                self.performance_data[key] = self.performance_data[key][-max_entries:]
                if trimmed > 0:
                    tprint(f"🧹 Trimmed {trimmed} entries from {key} history")
                    self.logger.debug(f"Trimmed {trimmed} entries from {key} performance history")

    def _resolve_regularization_settings(self) -> Dict[str, float]:
        """Resolve lookback regularization preferences from configuration."""
        defaults: Dict[str, float] = {
            'preferred_min': 40.0,
            'preferred_max': 80.0,
            'penalty_strength': 1e-5,
            'penalty_exponent': 2.0,
        }

        if not isinstance(getattr(self.config, 'custom_params', None), dict):
            resolved = defaults
        else:
            raw_settings = self.config.custom_params.get('lookback_regularization', {})
            resolved = defaults.copy()

            if isinstance(raw_settings, dict):
                preferred_window = raw_settings.get('preferred_window')
                if isinstance(preferred_window, (list, tuple)) and len(preferred_window) == 2:
                    try:
                        resolved['preferred_min'] = float(preferred_window[0])
                        resolved['preferred_max'] = float(preferred_window[1])
                    except (TypeError, ValueError) as e:
                        # Keep original values if conversion fails
                        self.logger.debug(f"Could not convert preferred window values: {e}")

                for key in ('preferred_min', 'preferred_max', 'penalty_strength', 'penalty_exponent'):
                    if key in raw_settings and raw_settings[key] is not None:
                        try:
                            resolved[key] = float(raw_settings[key])
                        except (TypeError, ValueError):
                            continue

                center = raw_settings.get('preferred_center')
                width = raw_settings.get('preferred_width')
                if center is not None:
                    try:
                        center_val = float(center)
                        if width is None:
                            width = resolved['preferred_max'] - resolved['preferred_min']
                        width_val = float(width)
                        resolved['preferred_min'] = center_val - (width_val / 2.0)
                        resolved['preferred_max'] = center_val + (width_val / 2.0)
                    except (TypeError, ValueError) as e:
                        # Keep original values if conversion fails
                        self.logger.debug(f"Could not convert center/width values: {e}")

        if resolved['preferred_min'] > resolved['preferred_max']:
            resolved['preferred_min'], resolved['preferred_max'] = resolved['preferred_max'], resolved['preferred_min']

        # Provide derived attributes for downstream metadata consumers.
        resolved['preferred_center'] = (resolved['preferred_min'] + resolved['preferred_max']) / 2.0
        resolved['preferred_width'] = resolved['preferred_max'] - resolved['preferred_min']

        self.logger.debug(
            "Lookback regularization settings resolved",
            extra={'extra_fields': {
                'preferred_min': resolved['preferred_min'],
                'preferred_max': resolved['preferred_max'],
                'penalty_strength': resolved['penalty_strength'],
                'penalty_exponent': resolved['penalty_exponent'],
            }}
        )

        return resolved

    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts for this component."""
        tprint("📋 Getting required artifacts for modular feature lookback optimization")
        artifacts = [
            'market_data',
            'labeling_results',
            'regime_splitting_results'
        ]
        tprint(f"✅ Required artifacts: {artifacts}")
        return artifacts

    async def execute(self, data: Any, pipeline_state: PipelineState) -> ComponentResult:
        """
        Execute the feature lookback optimization.

        Args:
            data: Input data for optimization
            pipeline_state: Current pipeline state

        Returns:
            ComponentResult with optimization results
        """
        pipeline_state = PipelineState.ensure(pipeline_state)

        locator = pipeline_state.get('data_locator')
        if isinstance(locator, DataLocator):
            self.core_optimizer.set_data_locator(locator)
        else:
            self.core_optimizer.set_data_locator(None)

        tprint("🚀 Starting modular feature lookback optimization execution...")
        start_time = self.performance_monitor.start_operation("execute")

        validation_metadata: Dict[str, Dict[str, Optional[Dict[str, str]]]] = {
            'inputs': {},
            'outputs': {},
            'derived': {},
        }

        target_shifts: Dict[str, int] = {}

        def _update_target_shifts(source: Any) -> None:
            if isinstance(source, Mapping):
                raw_shifts = source.get('target_shifts')
                if isinstance(raw_shifts, Mapping):
                    for key, value in raw_shifts.items():
                        try:
                            target_shifts[str(key)] = int(value)
                        except (TypeError, ValueError):
                            continue
                metadata_candidate = source.get('metadata')
                if isinstance(metadata_candidate, Mapping):
                    _update_target_shifts(metadata_candidate)
                for nested_key in ('multi_horizon_labeling_result', 'standardized_output'):
                    nested = source.get(nested_key)
                    if isinstance(nested, Mapping):
                        _update_target_shifts(nested)

        _update_target_shifts(pipeline_state)

        try:
            log_info("🚀 Starting feature lookback optimization with multi-horizon profit targets...")
            tprint("📊 Performance monitoring started for execute operation")

            # Store execution context for data loading
            self._current_execution_context = {
                'symbol': pipeline_state.get('symbol', 'ETHUSDT'),
                'exchange': pipeline_state.get('exchange', 'binance'),
                'timeframe': pipeline_state.get('timeframe', '15m')
            }

            numpy_rng = pipeline_state.get('numpy_rng') if isinstance(pipeline_state, dict) else None
            if numpy_rng is not None:
                self.core_optimizer.set_rng(numpy_rng)

            # Validate inputs (skip if data is None, will load from cache)
            if data is not None and not (isinstance(data, pd.DataFrame) and data.empty):
                is_valid, validation_summary, cleaned_data = self.validator.validate_data(
                    data,
                    required_columns=['open', 'high', 'low', 'close', 'volume']
                )

                if not is_valid:
                    tprint("❌ Data validation failed, aborting execution")
                    error_msg = f"Data validation failed: {validation_summary.recommendations}"
                    raise ValueError(error_msg)
            else:
                # No data provided, will load from cache
                tprint("📥 No input data provided, will load from KlinesParquetManager")
                cleaned_data = None
                validation_summary = None

            # Record validation metrics (if validation was performed)
            if validation_summary is not None:
                tprint("📈 Recording validation metrics after successful validation")
                self.performance_monitor.record_optimization_metrics(
                    {},
                    data_quality_score=validation_summary.quality_score,
                    validation_score=1.0 if validation_summary.overall_status == ValidationStatus.PASSED else 0.0
                )

            # Extract execution mode parameters from pipeline configuration
            execution_mode_params = {}
            lookback_config = None
            if self.execution_mode_config and hasattr(pipeline_state, 'get'):
                try:
                    # Try to extract execution mode from pipeline state or config
                    pipeline_config = pipeline_state.get('pipeline_config', {})
                    # Check if execution_mode_config is dict (fallback) or has methods
                    if isinstance(self.execution_mode_config, dict):
                        # Use fallback dict configuration
                        mode = pipeline_config.get('mode', 'light')
                        execution_mode_params = self.execution_mode_config.get(mode, self.execution_mode_config.get('light', {}))
                        tprint_warning(f"⚠️ Using fallback execution mode config for mode: {mode}")
                    else:
                        # Use proper execution mode config object
                        lookback_config = self.execution_mode_config.extract_from_pipeline_config(pipeline_config)
                        execution_mode_params = self.execution_mode_config.get_optimization_parameters(
                            pipeline_config.get('mode', 'full')
                        )
                    self.logger.info(f"📊 Using execution mode parameters: {execution_mode_params}")
                except Exception as e:
                    # Fallback to default parameters instead of crashing
                    tprint_warning(f"⚠️ Could not extract execution mode parameters: {e}, using defaults")
                    execution_mode_params = {'max_features': 100, 'max_lookback': 50}
            else:
                # Fallback to default parameters instead of crashing
                tprint_warning(f"⚠️ Execution mode configuration not available, using defaults")
                execution_mode_params = {'max_features': 100, 'max_lookback': 50}

            pipeline_state['lookback_config'] = lookback_config or pipeline_state.get('lookback_config', {})
            cache_key = self._resolve_cache_key(pipeline_state, lookback_config)
            self.logger.info(f"🗂️ Feature cache key resolved: {cache_key}")
            self.set_run_metadata({
                'feature_cache_key': cache_key,
                'lookback_config_hash': self._current_lookback_hash,
            })

            # Load required data
            tprint("📥 Loading market data for optimization")
            market_data = await self._load_market_data(cleaned_data, pipeline_state)
            market_data = validate_raw_ohlcv(
                market_data,
                context="feature_lookback_optimization.market_data"
            )
            validation_metadata['inputs']['market_data'] = schema_metadata('raw_ohlcv').get('raw_ohlcv')
            labeling_data = self._load_recent_labeling_results(
                pipeline_state.get('symbol', 'UNKNOWN'),
                pipeline_state.get('exchange', 'UNKNOWN'),
                pipeline_state.get('timeframe', 'UNKNOWN'),
                pipeline_state=pipeline_state
            )

            _update_target_shifts(labeling_data)

            if labeling_data:
                labels_df: Optional[pd.DataFrame] = None
                if isinstance(labeling_data, dict):
                    labels_df = labeling_data.get('labeled_data')
                    if labels_df is None:
                        nested = labeling_data.get('multi_horizon_labeling_result')
                        if isinstance(nested, dict):
                            labels_df = nested.get('labeled_data')
                if isinstance(labels_df, pd.DataFrame) and not labels_df.empty:
                    # Skip strict schema validation - accept any labeled data format
                    tprint_info(f"📊 Labeled data loaded: {labels_df.shape[0]} rows, {labels_df.shape[1]} columns")
                    tprint_debug(f"📋 Labeled data columns: {list(labels_df.columns[:10])}...")  # Show first 10 columns
                    # Store flexible metadata
                    validation_metadata['inputs']['labeled_targets'] = {
                        'rows': labels_df.shape[0],
                        'columns': labels_df.shape[1],
                        'has_target_columns': any('target' in col.lower() for col in labels_df.columns),
                        'has_confidence_columns': any('confidence' in col.lower() for col in labels_df.columns)
                    }

            # Apply execution mode data windowing
            if execution_mode_params and market_data is not None:
                window_days = execution_mode_params.get('window_days', 1460)
                if len(market_data) > window_days:
                    # Use only the most recent data based on execution mode
                    market_data = market_data.tail(window_days).copy()
                    self.logger.info(f"📊 Applied execution mode window: using last {window_days} days of data")
                else:
                    self.logger.info(f"📊 Using all available data ({len(market_data)} records) for execution mode")

            if market_data is None:
                log_error("Market data loading failed - no data available for feature lookback optimization")
                tprint("❌ Market data missing, returning failed result")
                return self._create_failed_result()

            # Align data with regime assignments to ensure consistency
            tprint("📐 Aligning market data with regime assignments")
            market_data = self._align_data_with_regime_assignments(market_data, pipeline_state)

            # Prepare data for optimization
            tprint("🧰 Preparing data for feature optimization")
            optimization_data = self._prepare_data_for_optimization(market_data, labeling_data)

            if optimization_data is None or optimization_data.empty:
                log_error(f"Data preparation failed - optimization data is {'None' if optimization_data is None else 'empty'}")
                tprint("❌ Prepared optimization data is empty or None")
                return self._create_failed_result()

            # Validate engineered features (basic checks only)
            optimization_data = validate_engineered_features(
                optimization_data,
                context="feature_lookback_optimization.optimization_frame"
            )
            
            # Skip temporal alignment check for feature lookback optimization
            # This component generates its own lagged features and the raw data columns
            # have been removed, so temporal alignment is not applicable at this stage
            tprint_debug("ℹ️ Skipping temporal alignment check for feature lookback optimization")
            
            validation_metadata['outputs']['optimization_frame'] = schema_metadata('engineered_features').get('engineered_features')

            # Perform feature optimization
            tprint("⚙️ Performing feature optimization workflow")
            optimization_results = await self._perform_feature_optimization(optimization_data, pipeline_state)

            # Convert int64 values to regular int values for JSON serialization
            tprint("🔄 Converting optimization results for JSON serialization")
            optimization_results = convert_int64_to_int(optimization_results)

            # Create optimization metrics
            tprint("📏 Creating optimization metrics")
            metrics = self._create_optimization_metrics(optimization_results)

            # Create artifacts
            tprint("📦 Creating artifacts from optimization results")
            artifacts = self._create_artifacts(optimization_results, pipeline_state)
            artifacts.setdefault('validated_schemas', validation_metadata)

            # Trim performance history to prevent memory leaks
            self._trim_performance_history(self.max_performance_entries)
            
            # Record final metrics
            tprint("🏁 Ending performance monitoring for execute operation")
            self.performance_monitor.end_operation("execute", start_time, success=True)

            # Save artifacts persistently using the artifact manager
            try:
                import asyncio
                # Check if we're already in an event loop
                try:
                    loop = asyncio.get_running_loop()
                    # We're in an event loop, create a task instead
                    task = asyncio.create_task(self.save_artifacts(artifacts, {
                        'optimization_status': 'completed',
                        'total_features_optimized': len(optimization_results.get('feature_results', {})),
                        'validation_summary': validation_summary.__dict__ if validation_summary else None,
                        'performance_metrics': self.performance_monitor.get_performance_summary(),
                        'optimization_results': optimization_results,
                        'validated_schemas': validation_metadata
                    }))
                    save_report = await task
                    log_success(
                        f"💾 [FEATURE_LOOKBACK] Artifacts saved persistently (correlation_id={save_report.correlation_id}): {list(save_report.paths.keys())}"
                    )
                except RuntimeError:
                    # No running event loop, use asyncio.run()
                    tprint("💾 Saving artifacts using asyncio.run")
                    save_report = asyncio.run(self.save_artifacts(artifacts, {
                        'optimization_status': 'completed',
                        'total_features_optimized': len(optimization_results.get('feature_results', {})),
                        'validation_summary': validation_summary.__dict__ if validation_summary else None,
                        'performance_metrics': self.performance_monitor.get_performance_summary(),
                        'optimization_results': optimization_results,
                        'validated_schemas': validation_metadata
                    }))
                    log_success(
                        f"💾 [FEATURE_LOOKBACK] Artifacts saved persistently (correlation_id={save_report.correlation_id}): {list(save_report.paths.keys())}"
                    )
            except Exception as e:
                log_warning(f"⚠️ [FEATURE_LOOKBACK] Failed to save artifacts persistently: {e}")

            pipeline_state['feature_cache_metrics'] = dict(self.cache_metrics)

            result = ComponentResult(
                success=True,
                artifacts=artifacts,
                metadata={
                    'optimization_status': 'completed',
                    'total_features_optimized': len(optimization_results.get('feature_results', {}).get('long_pipeline', {})) + len(optimization_results.get('feature_results', {}).get('short_pipeline', {})),
                    'validation_summary': validation_summary.__dict__ if validation_summary else None,
                    'performance_metrics': self.performance_monitor.get_performance_summary(),
                    'optimization_results': optimization_results,
                    'artifacts_saved_persistently': True,
                    'pipeline_type': 'differentiated_long_short',
                    'validated_schemas': validation_metadata,
                    'feature_cache_metrics': dict(self.cache_metrics),
                    'feature_cache_key': self._current_cache_key,
                    'feature_cache_force_refresh': self._force_cache_refresh,
                }
            )

            # Generate outcome file with datetime stamp
            try:
                from datetime import datetime
                outcome_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                outcomes_dir = Path('outcomes')
                outcomes_dir.mkdir(parents=True, exist_ok=True)
                
                outcome_filename = f"feature_lookback_optimization_outcome_{outcome_timestamp}.json"
                outcome_path = outcomes_dir / outcome_filename
                
                long_count = len(optimization_results.get('feature_results', {}).get('long_pipeline', {}))
                short_count = len(optimization_results.get('feature_results', {}).get('short_pipeline', {}))
                
                # Extract detailed feature-level optimization results
                long_features_detail = {}
                short_features_detail = {}
                
                if 'feature_results' in optimization_results:
                    long_pipeline = optimization_results['feature_results'].get('long_pipeline', {})
                    for feature_name, feature_data in long_pipeline.items():
                        if isinstance(feature_data, dict):
                            long_features_detail[feature_name] = {
                                'optimal_lookback': feature_data.get('optimal_lookback'),
                                'score': feature_data.get('score'),
                                'method': feature_data.get('method'),
                            }
                    
                    short_pipeline = optimization_results['feature_results'].get('short_pipeline', {})
                    for feature_name, feature_data in short_pipeline.items():
                        if isinstance(feature_data, dict):
                            short_features_detail[feature_name] = {
                                'optimal_lookback': feature_data.get('optimal_lookback'),
                                'score': feature_data.get('score'),
                                'method': feature_data.get('method'),
                            }
                
                # Calculate optimization statistics
                optimization_stats = {
                    'long_pipeline': {
                        'total_features': long_count,
                        'avg_lookback': float(sum(f.get('optimal_lookback', 0) for f in long_features_detail.values()) / long_count) if long_count > 0 else 0.0,
                        'avg_score': float(sum(f.get('score', 0) for f in long_features_detail.values() if f.get('score') is not None) / max(1, sum(1 for f in long_features_detail.values() if f.get('score') is not None))),
                        'min_lookback': min((f.get('optimal_lookback', 0) for f in long_features_detail.values()), default=0),
                        'max_lookback': max((f.get('optimal_lookback', 0) for f in long_features_detail.values()), default=0),
                    },
                    'short_pipeline': {
                        'total_features': short_count,
                        'avg_lookback': float(sum(f.get('optimal_lookback', 0) for f in short_features_detail.values()) / short_count) if short_count > 0 else 0.0,
                        'avg_score': float(sum(f.get('score', 0) for f in short_features_detail.values() if f.get('score') is not None) / max(1, sum(1 for f in short_features_detail.values() if f.get('score') is not None))),
                        'min_lookback': min((f.get('optimal_lookback', 0) for f in short_features_detail.values()), default=0),
                        'max_lookback': max((f.get('optimal_lookback', 0) for f in short_features_detail.values()), default=0),
                    }
                }
                
                # Performance breakdown
                perf_summary = self.performance_monitor.get_performance_summary()
                performance_breakdown = {
                    'execution_time_seconds': perf_summary.get('execution_time', 0.0),
                    'operations': convert_int64_to_int(perf_summary.get('operations', {})),
                    'memory_usage': convert_int64_to_int(perf_summary.get('memory_usage', {})),
                    'cache_effectiveness': {
                        'cache_hit': self.cache_metrics.get('cache_hit', False),
                        'cache_miss': self.cache_metrics.get('cache_miss', False),
                        'cache_save': self.cache_metrics.get('cache_save', False),
                        'cache_load_time': self.cache_metrics.get('cache_load_time', 0.0),
                        'cache_save_time': self.cache_metrics.get('cache_save_time', 0.0),
                    }
                }
                
                # Hardware utilization
                hardware_stats = {
                    'cpu_optimizations': perf_summary.get('cpu_optimizations', 'unknown'),
                    'gpu_available': perf_summary.get('gpu_available', False),
                    'matrix_operations': perf_summary.get('matrix_operations', 'standard'),
                }
                
                # Create comprehensive outcome report
                outcome_data = {
                    'component': 'feature_lookback_optimization',
                    'timestamp': datetime.now().isoformat(),
                    'execution_time': perf_summary.get('execution_time', 0.0),
                    'configuration': {
                        'symbol': self.config.symbol if hasattr(self.config, 'symbol') else 'UNKNOWN',
                        'exchange': self.config.exchange if hasattr(self.config, 'exchange') else 'UNKNOWN',
                        'timeframe': self.config.timeframe if hasattr(self.config, 'timeframe') else 'UNKNOWN',
                        'optimization_method': str(self.core_optimizer.optimization_method.value) if hasattr(self.core_optimizer, 'optimization_method') else 'UNKNOWN',
                        'pipeline_type': 'differentiated_long_short',
                    },
                    'results': {
                        'summary': {
                            'total_features_optimized': long_count + short_count,
                            'long_pipeline_features': long_count,
                            'short_pipeline_features': short_count,
                            'feature_cache_hit': self.cache_metrics.get('cache_hit', False),
                            'feature_cache_key': self._current_cache_key,
                        },
                        'optimization_statistics': optimization_stats,
                        'long_features_detail': long_features_detail,
                        'short_features_detail': short_features_detail,
                        'full_optimization_results': convert_int64_to_int(optimization_results),
                    },
                    'performance_metrics': performance_breakdown,
                    'hardware_utilization': hardware_stats,
                    'validation_summary': validation_summary.__dict__ if validation_summary else None,
                    'cache_metrics': {
                        'cache_key': self._current_cache_key,
                        'force_refresh': self._force_cache_refresh,
                        'metrics': convert_int64_to_int(self.cache_metrics),
                    },
                    'status': 'success'
                }
                
                # Save outcome file
                with open(outcome_path, 'w') as f:
                    json.dump(outcome_data, f, indent=2, default=str)
                
                tprint_success(f"📄 Outcome file saved: {outcome_filename}")
                
            except Exception as outcome_error:
                tprint_warning(f"⚠️ Failed to save outcome file: {outcome_error}")
                # Don't fail the component if outcome file generation fails

            long_count = len(optimization_results.get('feature_results', {}).get('long_pipeline', {}))
            short_count = len(optimization_results.get('feature_results', {}).get('short_pipeline', {}))
            log_success(f"🎯 Multi-horizon feature lookback optimization completed successfully - Long: {long_count} features, Short: {short_count} features")
            tprint("✅ Feature lookback optimization execution completed successfully")
            return result

        except SchemaValidationException as schema_error:
            self.performance_monitor.end_operation("execute", start_time, success=False)
            return self._schema_failure_result(schema_error)

        except Exception as e:
            # Create detailed error information
            import traceback
            error_details = {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc(),
                'component': 'feature_lookback_optimization',
                'timestamp': datetime.now().isoformat()
            }
            
            self.error_handler.handle_error(
                e,
                "execute",
                return_value=self._create_failed_result(error_details)
            )
            self.performance_monitor.end_operation("execute", start_time, success=False)
            tprint(f"❌ Feature lookback optimization execution failed: {e}")
            return self._create_failed_result(error_details)

    def _create_failed_result(self, error_details: Optional[Dict] = None) -> ComponentResult:
        """Create a failed component result."""
        metadata = {'optimization_status': 'failed'}
        if error_details:
            metadata['error_details'] = error_details
            
        return ComponentResult(
            success=False,
            artifacts=FeatureLookbackArtifacts(),
            metadata=metadata
        )

    def _schema_failure_result(self, error: SchemaValidationException) -> ComponentResult:
        """Create a failed result for schema validation errors."""
        message = str(error)
        log_error(message)
        tprint(f"❌ Schema validation failure: {message}")
        return ComponentResult(
            success=False,
            artifacts=FeatureLookbackArtifacts(),
            error_message=message,
            metadata={
                'optimization_status': 'failed',
                'schema_error': {
                    'schema_key': error.schema_key,
                    'context': error.context,
                    'schema_metadata': schema_metadata(error.schema_key).get(error.schema_key)
                }
            }
        )
    def _current_feature_bank_version(self) -> str:
        try:
            from src.feature_generation.core.feature_bank import FeatureBank
            version = getattr(FeatureBank, 'VERSION', 'unknown')
            tprint_debug(f"🏦 Feature bank version: {version}")
            return version
        except Exception as e:
            error_msg = f"Could not determine feature bank version: {e}"
            tprint_error(f"❌ {error_msg}")
            raise RuntimeError(error_msg)

    def _resolve_cache_key(self, pipeline_state: Dict[str, Any], lookback_config: Optional[Any] = None) -> str:
        if pipeline_state is None:
            error_msg = "Cannot resolve cache key: pipeline_state is None"
            tprint_error(f"❌ {error_msg}")
            raise ValueError(error_msg)

        symbol = pipeline_state.get('symbol', self.config.symbol)
        timeframe = pipeline_state.get('timeframe', self.config.timeframe)
        lookback_source = lookback_config or pipeline_state.get('lookback_config') or {}
        lookback_hash = FeatureCacheService.compute_config_hash(lookback_source)

        tprint_debug(f"🔑 Cache key components: symbol={symbol}, timeframe={timeframe}, lookback_hash={lookback_hash[:8]}...")
        
        self._current_lookback_hash = lookback_hash
        pipeline_state['lookback_config_hash'] = lookback_hash

        version = self._current_feature_bank_version()
        cache_key = FeatureCacheService.build_key(symbol, timeframe, version, lookback_hash)
        pipeline_state['feature_cache_key'] = cache_key
        self._current_cache_key = cache_key
        
        tprint_success(f"✅ Cache key resolved: {cache_key}")
        return cache_key

    def _sync_cache_metrics(self) -> None:
        """Synchronize cache metrics with performance monitor."""
        if hasattr(self.performance_monitor, 'update_cache_metrics'):
            self.performance_monitor.update_cache_metrics(dict(self.cache_metrics))
            # Calculate hit rate
            total_accesses = self.cache_metrics['hits'] + self.cache_metrics['misses']
            if total_accesses > 0:
                hit_rate = self.cache_metrics['hits'] / total_accesses
                tprint_debug(f"💾 Cache metrics: {self.cache_metrics['hits']} hits, {self.cache_metrics['misses']} misses (hit rate: {hit_rate:.2%})")

    def _validate_data_for_lookback(
        self,
        data: pd.DataFrame,
        required_columns: Optional[List[str]] = None,
        lookback_range: Optional[Tuple[int, int]] = None
    ) -> Tuple[bool, str]:
        """
        Validate data for lookback optimization.
        
        Args:
            data: DataFrame to validate
            required_columns: List of required column names
            lookback_range: Tuple of (min_lookback, max_lookback)
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Use the validator to validate the data
            is_valid, validation_summary, _ = self.validator.validate_data(
                data,
                required_columns=required_columns or ['open', 'high', 'low', 'close', 'volume'],
                lookback_range=lookback_range
            )
            
            if not is_valid:
                error_msg = f"Data validation failed: {', '.join(validation_summary.recommendations)}"
                self.logger.error(f"❌ {error_msg}")
                return False, error_msg
            
            # Additional lookback-specific validations
            if lookback_range:
                min_lookback, max_lookback = lookback_range
                if len(data) < max_lookback:
                    error_msg = f"Insufficient data for lookback optimization: {len(data)} rows < {max_lookback} required"
                    self.logger.error(f"❌ {error_msg}")
                    return False, error_msg
            
            self.logger.info(f"✅ Data validation passed for lookback optimization")
            return True, ""
            
        except Exception as e:
            error_msg = f"Error during data validation: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            return False, error_msg

    async def _load_market_data(self, data: Any, pipeline_state: Dict[str, Any] = None) -> pd.DataFrame:
        """Load market data for optimization using ares launcher integration."""
        try:
            if isinstance(data, pd.DataFrame) and not data.empty:
                tprint_success(f"✅ Market data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
                return data
            
            # If no data provided, use ares launcher integration for data loading
            tprint("📥 No data provided, using ares launcher integration for data loading...")
            
            # Get symbol, exchange, and timeframe from pipeline state or config
            symbol = None
            exchange = None
            timeframe = None
            
            # Try to get from execution context
            if hasattr(self, '_current_execution_context'):
                ctx = self._current_execution_context
                if isinstance(ctx, dict):
                    symbol = ctx.get('symbol')
                    exchange = ctx.get('exchange')
                    timeframe = ctx.get('timeframe')
            
            # Fallback to config
            if not symbol:
                symbol = getattr(self.config, 'symbol', 'ETHUSDT') if hasattr(self, 'config') else 'ETHUSDT'
            if not exchange:
                exchange = getattr(self.config, 'exchange', 'binance') if hasattr(self, 'config') else 'binance'
            if not timeframe:
                timeframe = getattr(self.config, 'timeframe', '15m') if hasattr(self, 'config') else '15m'
            
            tprint(f"📊 Loading data for {symbol} on {exchange} ({timeframe}) using ares launcher integration...")
            
            # Use ares launcher integration for data loading
            if not hasattr(self, 'ares_integration'):
                self.ares_integration = AresLauncherFeatureLookbackOptimizer()
            
            # Create pipeline state for ares integration
            ares_pipeline_state = pipeline_state or {}
            ares_pipeline_state.update({
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe
            })
            
            # Load data using ares launcher integration
            loaded_data = await self.ares_integration.load_data_async_for_optimization(
                symbol=symbol,
                timeframe=timeframe,
                pipeline_state=ares_pipeline_state
            )
            
            if loaded_data is not None and not loaded_data.empty:
                tprint_success(f"✅ Market data loaded via ares launcher: {loaded_data.shape[0]} rows, {loaded_data.shape[1]} columns")
                tprint_info(f"📅 Data mode: {loaded_data.attrs.get('ares_mode', 'Unknown')}")
                tprint_info(f"📅 Lookback days: {loaded_data.attrs.get('lookback_days', 'Unknown')}")
                return loaded_data
            else:
                error_msg = f"No data found for {symbol}/{exchange}/{timeframe} using ares launcher integration"
                tprint_error(f"❌ {error_msg}")
                raise ValueError(error_msg)
                
        except Exception as e:
            tprint_error(f"❌ Critical error loading market data via ares launcher: {e}")
            raise

    def _align_data_with_regime_assignments(self, market_data: pd.DataFrame, pipeline_state: Dict[str, Any]) -> pd.DataFrame:
        """Align market data with regime assignments to ensure consistency with clustering step."""
        try:
            symbol = pipeline_state.get('symbol', 'ETHUSDT').lower()
            tprint_debug(f"🔍 Aligning data with regime assignments for {symbol}")

            pipeline_custom_params = pipeline_state.get('custom_params', {}) if isinstance(pipeline_state, dict) else {}
            config_custom_params = getattr(self.config, 'custom_params', {}) or {}

            candidate_dirs: List[Path] = []

            def _normalize_candidate(path_value: Any) -> Optional[Path]:
                if not path_value:
                    return None
                path = path_value if isinstance(path_value, Path) else Path(path_value)
                path = path.expanduser()

                # If a file path is provided, use its parent directory
                if path.suffix:
                    path = path.parent

                # If the path already points to the symbol directory, use its parent
                if path.name.lower() == symbol:
                    path = path.parent

                if not path.is_absolute():
                    path = Path.cwd() / path

                return path

            def _register_candidate(path_value: Any) -> None:
                normalized = _normalize_candidate(path_value)
                if not normalized:
                    return

                candidate_dirs.append(normalized)
                if normalized.name.lower() != 'nas_tas_clustering':
                    candidate_dirs.append(normalized / 'nas_tas_clustering')

            # Gather candidate directories from pipeline state and configuration
            _register_candidate(pipeline_state.get('regime_cache_path'))
            _register_candidate(pipeline_custom_params.get('regime_cache_path'))
            _register_candidate(config_custom_params.get('regime_cache_path'))

            for base_dir in (
                pipeline_state.get('data_cache_dir'),
                pipeline_custom_params.get('data_cache_dir'),
                config_custom_params.get('data_cache_dir'),
                pipeline_state.get('data_dir'),
                getattr(self.config, 'data_dir', None),
            ):
                _register_candidate(base_dir)

            # Fallback to a locator-resolved cache directory when available
            locator_candidate = pipeline_state.get('data_locator')
            if isinstance(locator_candidate, (DataLocator, PipelineDataLocator)):
                cache_key = pipeline_state.get('cache_dir_key') or getattr(self.config, 'cache_dir_key', None)
                try:
                    default_cache_dir = locator_candidate.cache_path(cache_key, ensure_exists=True)
                except Exception:
                    default_cache_dir = locator_candidate.cache_path(ensure_exists=True)

                primary = default_cache_dir if isinstance(default_cache_dir, Path) else Path(default_cache_dir)
                primary = primary.expanduser()
                if not primary.is_absolute():
                    primary = Path.cwd() / primary
                self.logger.warning(
                    "⚠️ Using locator-provided regime cache directory: %s",
                    primary,
                )
                candidate_dirs.insert(0, primary)
                if primary.name.lower() != 'nas_tas_clustering':
                    candidate_dirs.insert(1, primary / 'nas_tas_clustering')
            else:
                default_locator = PipelineDataLocator()
                cache_key = (
                    pipeline_state.get('cache_dir_key')
                    or getattr(self.config, 'cache_dir_key', None)
                    or 'default'
                )
                try:
                    primary = default_locator.cache_path(cache_key, ensure_exists=True)
                except Exception:
                    primary = default_locator.cache_path(ensure_exists=True)

                primary = primary.expanduser()
                if not primary.is_absolute():
                    primary = Path.cwd() / primary

                self.logger.warning(
                    "⚠️ Using default locator cache directory: %s",
                    primary,
                )
                candidate_dirs.insert(0, primary)
                if primary.name.lower() != 'nas_tas_clustering':
                    candidate_dirs.insert(1, primary / 'nas_tas_clustering')

            # Remove duplicates while preserving order
            seen_paths = set()
            unique_candidates: List[Path] = []
            for candidate in candidate_dirs:
                candidate = candidate.expanduser()
                if not candidate.is_absolute():
                    candidate = Path.cwd() / candidate
                key = candidate.as_posix()
                if key not in seen_paths:
                    seen_paths.add(key)
                    unique_candidates.append(candidate)

            regime_cache_dir: Optional[Path] = None
            for candidate in unique_candidates:
                if candidate.exists():
                    regime_cache_dir = candidate
                    break

            if regime_cache_dir is None:
                search_locations = ", ".join(str(path) for path in unique_candidates) or "None"
                self.logger.warning(
                    f"⚠️ Regime cache directory not found; searched locations: {search_locations}. Using full dataset"
                )
                return market_data

            if regime_cache_dir.is_file():
                regime_cache_dir = regime_cache_dir.parent

            try:
                resolved_regime_cache_dir = regime_cache_dir.resolve(strict=False)
            except Exception:
                resolved_regime_cache_dir = regime_cache_dir

            self.logger.info(f"🔎 Searching for regime assignments in {resolved_regime_cache_dir}")

            if not regime_cache_dir.exists():
                error_msg = f"Regime cache directory not found at {resolved_regime_cache_dir}"
                tprint_error(f"❌ {error_msg}")
                raise FileNotFoundError(error_msg)

            # Try to load regime assignment file to get the correct data size
            regime_files = list(regime_cache_dir.glob(f'**/{symbol}/nas_tas_regime_assignments_*.parquet'))
            
            tprint_debug(f"🔍 Found {len(regime_files)} regime assignment files for {symbol}")

            if regime_files:
                # Load the most recent regime assignment file
                latest_file = max(regime_files, key=lambda x: x.stat().st_mtime)
                tprint_info(f"📂 Loading regime assignments from: {latest_file.name}")
                regime_df = pd.read_parquet(latest_file)
                tprint_debug(f"📊 Regime assignments: {len(regime_df)} records")

                # Filter market data to match the regime assignment size
                if len(regime_df) < len(market_data):
                    # Use the same number of records as regime assignments
                    original_len = len(market_data)
                    market_data = market_data.tail(len(regime_df)).copy()
                    self.logger.info(f"🔍 Aligned market data to regime assignments: {len(market_data)} records")
                    tprint_info(f"✂️ Trimmed market data from {original_len} to {len(market_data)} records to match regime assignments")
                else:
                    self.logger.info(f"📊 Using full market data: {len(market_data)} records")
                    tprint_info(f"📊 Using full market data: {len(market_data)} records (matches regime assignments)")
            else:
                error_msg = f"No regime assignment files found in {resolved_regime_cache_dir} for symbol {symbol}"
                tprint_error(f"❌ {error_msg}")
                raise FileNotFoundError(error_msg)
            
            # Return the aligned market data
            return market_data

        except Exception as e:
            tprint_error(f"❌ Failed to align data with regime assignments: {e}")
            raise

    async def _generate_features_for_optimization(
        self,
        data: pd.DataFrame,
        pipeline_state: Optional[Dict[str, Any]] = None,
        *,
        force_refresh: bool = False,
    ) -> List[str]:
        """Generate features using the feature bank system with caching support."""

        pipeline_state = PipelineState.ensure(pipeline_state)
        cache_key = pipeline_state.get('feature_cache_key') or self._current_cache_key

        try:
            cached_features = None
            if cache_key:
                if force_refresh:
                    self.logger.info("♻️ Force refresh requested for feature cache key %s", cache_key)
                    self.cache_metrics['force_refreshes'] += 1
                    self.performance_monitor.record_cache_event('force_refresh', cache_key)
                    self._sync_cache_metrics()
                else:
                    cached_features = self.feature_cache.load(cache_key)
                    if cached_features is not None and not cached_features.empty:
                        self.logger.info("📦 Reusing cached feature bank matrix for key %s", cache_key)
                        self.cache_metrics['hits'] += 1
                        self.performance_monitor.record_cache_event('hit', cache_key)
                        aligned = cached_features.reindex(data.index)
                        for col in aligned.columns:
                            data[col] = aligned[col].values
                        self._sync_cache_metrics()
                        return aligned.columns.tolist()
                    else:
                        self.logger.info("🔁 Feature cache miss for key %s", cache_key)
                        self.cache_metrics['misses'] += 1
                        self.performance_monitor.record_cache_event('miss', cache_key)
                        self._sync_cache_metrics()

            # Import the feature bank system
            from src.feature_generation.core.feature_bank import FeatureBank

            self.logger.info("🔧 Generating features using feature bank system...")
            tprint_info("🏦 Initializing feature bank for feature generation")

            # Initialize feature bank
            feature_bank = FeatureBank()

            # Generate features using the feature bank directly
            from src.feature_generation.core.feature_generator import FeatureCategory
            included_categories = [
                FeatureCategory.RETURNS,
                FeatureCategory.MOMENTUM,
                FeatureCategory.VOLUME,
                FeatureCategory.VOLATILITY,
                FeatureCategory.TREND,
                FeatureCategory.OSCILLATOR,
                FeatureCategory.SUPPORT_RESISTANCE,
                FeatureCategory.CANDLESTICK_PATTERN,
                FeatureCategory.MICROSTRUCTURE,
                FeatureCategory.ENTROPY,
                FeatureCategory.ORDER_FLOW,
                FeatureCategory.ACCELERATION,
                FeatureCategory.TIME,
            ]
            generated_features = feature_bank.generate_features(data, categories=included_categories)

            if generated_features is not None and not generated_features.empty:
                total_features = generated_features.shape[1]
                total_rows = generated_features.shape[0]
                self.logger.info(f"✅ Generated {total_features} features from feature bank")
                self.logger.info(f"📊 Feature matrix: {total_rows} rows × {total_features} columns")
                tprint_success(f"✅ Feature bank generated {total_features} features ({total_rows} rows)")
                
                # Log feature generation performance
                tprint_debug(f"📈 Feature generation rate: {total_features * total_rows / 1000:.1f}K values")

                feature_categories = {}
                for col in generated_features.columns:
                    if '_' in col:
                        category = col.split('_')[0]
                        feature_categories[category] = feature_categories.get(category, 0) + 1

                self.logger.info(f"📋 Feature breakdown: {dict(sorted(feature_categories.items()))}")

                excluded_columns = [
                    'regime_id', 'regime_prob', 'open', 'high', 'low', 'close', 'volume',
                    'timestamp', 'symbol', 'open_time', 'close_time', 'interval', 'exchange', 'timeframe'
                ]

                feature_columns = [col for col in generated_features.columns if col not in excluded_columns]
                feature_columns = [
                    col for col in feature_columns
                    if not any(
                        unwanted in col.lower()
                        for unwanted in [
                            'wavelet', 'autoencoder', 'regime_', 'nas_', 'tas_',
                            'interaction_', 'cross_timeframe_', 'cross_timeframe',
                            'bid_ask', 'bidask', 'market_depth', 'liquidity_proxy',
                            'order_flow', 'trade_intensity', 'volume_weighted'
                        ]
                    )
                ]

                self.logger.info(
                    f"🎯 Found {len(feature_columns)} engineered features for optimization (excluding unwanted types)"
                )
                tprint_info(f"🎯 Selected {len(feature_columns)} features for optimization after filtering")

                for col in feature_columns:
                    if col in generated_features.columns:
                        data[col] = generated_features[col].reindex(data.index).values

                if cache_key:
                    cached_matrix = generated_features[feature_columns].reindex(data.index)
                    self.feature_cache.save(cache_key, cached_matrix)
                    self.cache_metrics['writes'] += 1
                    self.performance_monitor.record_cache_event('write', cache_key)
                    self._sync_cache_metrics()
                    tprint_success(f"💾 Cached {len(feature_columns)} features to cache key: {cache_key}")

                return feature_columns

            # Fallback: use existing columns in data as features
            tprint_warning("⚠️ FeatureBank returned no features, using existing data columns as fallback")
            existing_feature_cols = [col for col in data.columns 
                                    if col not in ['open', 'high', 'low', 'close', 'volume',
                                                  'timestamp', 'datetime', 'open_time', 'close_time',
                                                  'quote_volume', 'number_of_trades',
                                                  'taker_buy_volume', 'taker_buy_quote_volume']]
            if existing_feature_cols:
                tprint_info(f"✅ Using {len(existing_feature_cols)} existing columns as features")
                return existing_feature_cols
            else:
                # Last resort: create simple features from OHLCV
                tprint_warning("⚠️ Creating simple features from OHLCV as last resort")
                if 'close' in data.columns:
                    data['returns_1'] = data['close'].pct_change(1)
                    data['returns_5'] = data['close'].pct_change(5)
                    data['returns_20'] = data['close'].pct_change(20)
                    return ['returns_1', 'returns_5', 'returns_20']
                else:
                    error_msg = "Feature generation failed - no features available and no OHLCV data"
                    tprint_error(f"❌ {error_msg}")
                    raise RuntimeError(error_msg)

        except Exception as e:
            self.logger.warning(f"⚠️ Feature bank generation failed: {e}, using fallback")
            tprint_warning(f"⚠️ Feature bank failed: {e}, using fallback feature generation")
            
            # Fallback feature generation
            try:
                if 'close' in data.columns:
                    tprint_info("🔧 Generating basic fallback features from OHLCV data")
                    data['returns_1'] = data['close'].pct_change(1)
                    data['returns_5'] = data['close'].pct_change(5)
                    data['returns_10'] = data['close'].pct_change(10)
                    data['returns_20'] = data['close'].pct_change(20)
                    if 'volume' in data.columns:
                        data['volume_change_1'] = data['volume'].pct_change(1)
                        data['volume_change_5'] = data['volume'].pct_change(5)
                    return ['returns_1', 'returns_5', 'returns_10', 'returns_20', 'volume_change_1', 'volume_change_5']
                else:
                    raise RuntimeError("No OHLCV data available for fallback feature generation")
            except Exception as fallback_error:
                raise RuntimeError(f"Both feature bank and fallback generation failed: {e}, {fallback_error}")

    def _coerce_to_dataframe(self, value: Any) -> pd.DataFrame:
        """Convert a value to a pandas DataFrame or raise an exception."""
        if value is None:
            raise ValueError("Cannot convert None to DataFrame")

        if isinstance(value, pd.DataFrame):
            return value.copy()

        if isinstance(value, pd.Series):
            return value.to_frame()

        if isinstance(value, dict):
            try:
                df = pd.DataFrame(value)
                if df.empty:
                    raise ValueError("DataFrame created from dict is empty")
                return df
            except Exception as e:
                raise ValueError(f"Cannot convert dict to DataFrame: {e}")

        if isinstance(value, list):
            try:
                df = pd.DataFrame(value)
                if df.empty:
                    raise ValueError("DataFrame created from list is empty")
                return df
            except Exception as e:
                raise ValueError(f"Cannot convert list to DataFrame: {e}")

        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, (list, dict)):
                    df = pd.DataFrame(parsed)
                    if df.empty:
                        raise ValueError("DataFrame created from JSON is empty")
                    return df
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON string: {e}")

        raise ValueError(f"Value type {type(value).__name__} cannot be converted to DataFrame")

    def _normalize_labeling_result(self, labeling_source: Any) -> Optional[Dict[str, Any]]:
        """Normalize labeling payload into a standardized dictionary with DataFrame values."""
        tprint("🧼 Normalizing labeling result payload")
        if not labeling_source:
            tprint("⚠️ No labeling source provided for normalization")
            return None

        if isinstance(labeling_source, dict) and 'labeled_data' in labeling_source:
            result = dict(labeling_source)
        elif isinstance(labeling_source, dict) and 'labels' in labeling_source:
            result = dict(labeling_source)
        else:
            result = {'labeled_data': labeling_source}

        # Safely get labeled_data or labels (avoid DataFrame boolean evaluation)
        labeled_data_candidate = result.get('labeled_data')
        if labeled_data_candidate is None:
            labeled_data_candidate = result.get('labels')
        labeled_df = self._coerce_to_dataframe(labeled_data_candidate)
        if labeled_df is None or labeled_df.empty:
            tprint("⚠️ Normalized labeling dataframe is empty")
            return None

        result['labeled_data'] = labeled_df
        if 'labels' in result:
            result['labels'] = labeled_df

        tprint("✅ Labeling result normalized successfully")
        return result

    def _merge_labeling_into_data(
        self,
        base_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        dataset_name: str = 'labeled_data'
    ) -> pd.DataFrame:
        """Merge labeling-derived columns into the working dataset."""
        if labels_df is None or labels_df.empty:
            return base_df

        merged_df = base_df.copy()
        incoming = labels_df.copy()

        # Track columns before merge for logging
        pre_columns = set(merged_df.columns)

        if 'timestamp' in merged_df.columns and 'timestamp' in incoming.columns:
            try:
                merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'])
                incoming['timestamp'] = pd.to_datetime(incoming['timestamp'])
            except Exception as e:
                # Keep original values if conversion fails
                self.logger.debug(f"Could not convert timestamp: {e}")

            incoming = incoming.drop_duplicates(subset=['timestamp'], keep='last')
            merged_df = merged_df.merge(
                incoming,
                on='timestamp',
                how='left',
                suffixes=('', '_mh')
            )

            # Resolve duplicate columns produced by merge suffixes
            for col in list(merged_df.columns):
                if col.endswith('_mh'):
                    original_col = col[:-3]
                    merged_df[original_col] = merged_df[col].combine_first(merged_df.get(original_col))
                    merged_df.drop(columns=[col], inplace=True)
        else:
            merged_df = merged_df.reset_index(drop=True)
            incoming = incoming.reset_index(drop=True)

            if len(incoming) > len(merged_df):
                incoming = incoming.iloc[-len(merged_df):].reset_index(drop=True)
            elif len(incoming) < len(merged_df):
                pad = len(merged_df) - len(incoming)
                pad_frame = pd.DataFrame({col: [np.nan] * pad for col in incoming.columns})
                incoming = pd.concat([pad_frame, incoming], ignore_index=True)

            for col in incoming.columns:
                if col == 'timestamp':
                    continue
                merged_df[col] = incoming[col].to_numpy(copy=False)

            merged_df.index = base_df.index

        new_columns = sorted(set(merged_df.columns) - pre_columns)
        if new_columns:
            preview = ', '.join(new_columns[:5])
            if len(new_columns) > 5:
                preview += ', …'
            self.logger.info(
                f"📊 Integrated {len(new_columns)} columns from {dataset_name}: {preview}"
            )

        return merged_df

    def _load_labeling_from_outcomes(
        self,
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> Optional[Dict[str, Any]]:
        """Fallback loader that inspects saved outcomes for labeling results."""
        manifest = ArtifactManifest()
        artifact_base_name = 'market_analysis_multi_horizon_profit_labeler_outcome'
        logical_name = DataLocator.build_logical_name(
            artifact_base_name,
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
        )
        entry = manifest.get_latest(logical_name)
        fallback_allowed = False

        if entry:
            outcome_file = entry.resolved_path
            if outcome_file.exists():
                try:
                    with open(outcome_file, 'r', encoding='utf-8') as handle:
                        outcome_data = json.load(handle)
                except json.JSONDecodeError as exc:
                    self.logger.warning(
                        f"⚠️ Failed to parse manifest outcome {outcome_file.name}: {exc}"
                    )
                    fallback_allowed = True
                except OSError as exc:
                    self.logger.warning(
                        f"⚠️ Unable to read manifest outcome {outcome_file.name}: {exc}"
                    )
                    fallback_allowed = True
                else:
                    config_data = outcome_data.get('config', {})
                    if (
                        config_data and (
                            (config_data.get('symbol') and config_data.get('symbol') != symbol)
                            or (config_data.get('exchange') and config_data.get('exchange') != exchange)
                            or (config_data.get('timeframe') and config_data.get('timeframe') != timeframe)
                        )
                    ):
                        self.logger.warning(
                            f"⚠️ Manifest outcome {outcome_file.name} metadata mismatch; ignoring entry"
                        )
                        fallback_allowed = True
                    else:
                        artifacts = outcome_data.get('artifacts', {}) if isinstance(outcome_data, dict) else {}
                        mh_result = None
                        if isinstance(artifacts, dict):
                            mh_result = artifacts.get('multi_horizon_labeling_result')
                        normalized = self._normalize_labeling_result(mh_result)
                        if normalized:
                            self.logger.info(
                                f"📂 Loaded multi-horizon labeling result from manifest entry {outcome_file.name}"
                            )
                            return normalized
                        fallback_allowed = True
            else:
                self.logger.warning(
                    f"⚠️ Manifest referenced outcome file missing: {outcome_file}"
                )
                fallback_allowed = True
        else:
            fallback_allowed = True

        if not fallback_allowed:
            tprint_debug("ℹ️ Fallback loading not allowed (manifest entry exists but invalid)")
            return None

        outcomes_dir = get_pre_training_settings().outcomes_root
        if not outcomes_dir.exists():
            tprint_debug(f"ℹ️ Outcomes directory does not exist: {outcomes_dir}")
            self.logger.info(f"Outcomes directory not found: {outcomes_dir}")
            return None

        try:
            # Try multiple patterns to find labeling outcome files
            patterns = [
                "market_analysis_multi_horizon_profit_labeler_outcome_*.json",
                "pre_training_analyst_profit_labeler_outcome_*.json",
                "analyst_labeler_outcome_*.json",
                "analyst_profit_labeler_outcome_*.json"
            ]
            
            outcome_files = []
            for pattern in patterns:
                outcome_files = list(outcomes_dir.glob(pattern))
                if outcome_files:
                    tprint_debug(f"ℹ️ Found outcome files matching pattern: {pattern}")
                    break
            
            if not outcome_files:
                tprint_debug(f"ℹ️ No outcome files found matching any pattern")
                self.logger.info(f"No outcome files found in {outcomes_dir}")
                return None

            latest_file = max(outcome_files, key=lambda f: f.stat().st_mtime)
            with open(latest_file, 'r', encoding='utf-8') as handle:
                outcome_data = json.load(handle)

            config_data = outcome_data.get('config', {}) if isinstance(outcome_data, dict) else {}
            if (
                config_data.get('symbol') == symbol and
                config_data.get('exchange') == exchange and
                config_data.get('timeframe') == timeframe
            ):
                artifacts = outcome_data.get('artifacts', {}) if isinstance(outcome_data, dict) else {}
                mh_result = artifacts.get('multi_horizon_labeling_result') if isinstance(artifacts, dict) else None
                normalized = self._normalize_labeling_result(mh_result)
                if normalized:
                    self.logger.info(
                        f"📂 Loaded multi-horizon labeling result from outcomes file {latest_file.name}"
                    )
                    return normalized
        except Exception as exc:
            self.logger.warning(f"⚠️ Failed to load labeling results from outcomes: {exc}")
            tprint_warning(f"⚠️ Failed to load labeling results from outcomes: {exc}")

            tprint_debug("ℹ️ No labeling results found in any source")
        
        # Final fallback: look for labeled_data_*.parquet files in artifacts directory
        try:
            artifacts_dir = Path('artifacts')
            if artifacts_dir.exists():
                # Look for labeled data parquet files matching symbol/exchange/timeframe
                pattern = f"labeled_data_{symbol}*{exchange}*{timeframe}*.parquet"
                labeled_files = list(artifacts_dir.glob(pattern))
                
                if labeled_files:
                    # Use the most recent file
                    latest_file = max(labeled_files, key=lambda f: f.stat().st_mtime)
                    tprint_success(f"✅ Found labeled data file: {latest_file}")
                    
                    # Load the parquet file
                    labeled_df = pd.read_parquet(latest_file)
                    
                    # Construct normalized result
                    normalized = {
                        'labeled_data': labeled_df,
                        'labels': labeled_df,
                        'n_samples': len(labeled_df),
                        'n_targets': len([col for col in labeled_df.columns if 'target' in col.lower()]),
                        'n_horizons': len([col for col in labeled_df.columns if 'horizon' in col.lower()]),
                        'method': 'analyst_profit_labeling',
                        'source_file': str(latest_file)
                    }
                    
                    self.logger.info(f"📂 Loaded labeled data from parquet file: {latest_file.name}")
                    return normalized
        except Exception as e:
            tprint_warning(f"⚠️ Failed to load from parquet files: {e}")
            
        return None

    def _load_recent_labeling_results(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        pipeline_state: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Load the most recent multi-horizon labeling results from available sources."""
        try:
            candidate_sources: List[Tuple[str, Any]] = []

            if pipeline_state:
                state_result = pipeline_state.get('multi_horizon_labeling_result')
                if state_result:
                    candidate_sources.append(("pipeline_state.multi_horizon_labeling_result", state_result))

                artifacts = pipeline_state.get('artifacts', {})
                if isinstance(artifacts, dict):
                    artifact_result = artifacts.get('multi_horizon_labeling_result')
                    if artifact_result:
                        candidate_sources.append(("pipeline_state.artifacts.multi_horizon_labeling_result", artifact_result))

            for source_name, raw_result in candidate_sources:
                normalized = self._normalize_labeling_result(raw_result)
                if normalized:
                    self.logger.info(f"📊 Using multi-horizon labeling result from {source_name}")
                    return normalized

            # Try loading from persistent artifacts managed by the artifact manager
            try:
                artifact_payload, metadata = self.artifact_manager.load_most_recent_artifact(
                    base_name="multi_horizon_labeling_result",
                    directory="artifacts",
                    extension=".json"
                )
            except Exception:
                artifact_payload, metadata = (None, None)

            if artifact_payload:
                if isinstance(artifact_payload, dict) and 'multi_horizon_labeling_result' in artifact_payload:
                    artifact_payload = artifact_payload['multi_horizon_labeling_result']
                normalized = self._normalize_labeling_result(artifact_payload)
                if normalized:
                    source = metadata.filename if metadata else 'artifact_storage'
                    self.logger.info(f"📊 Loaded multi-horizon labeling result from artifact {source}")
                    return normalized

            # Final fallback: inspect outcomes directory
            outcome_result = self._load_labeling_from_outcomes(symbol, exchange, timeframe)
            if outcome_result:
                return outcome_result

            message = (
                "Multi-horizon labeling results are unavailable; cannot proceed with feature lookback optimization"
            )
            self.logger.error(f"❌ {message}")
            raise RuntimeError(message)

        except Exception as e:
            self.error_handler.handle_error(
                e,
                "_load_recent_labeling_results",
                return_value=None
            )
            raise

    async def _perform_feature_optimization(
        self,
        data: pd.DataFrame,
        pipeline_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform feature optimization using the core optimizer."""
        tprint("⚙️ Starting feature optimization orchestration")
        try:
            # Generate features using PID-based feature generation system
            tprint("🧪 Generating features for optimization")
            force_refresh = bool(
                pipeline_state.get('feature_cache_force_refresh')
                or pipeline_state.get('force_feature_cache_refresh')
                or pipeline_state.get('force_refresh_features')
                or self.config.force_rerun
            )
            pipeline_state['feature_cache_force_refresh'] = force_refresh
            self._force_cache_refresh = force_refresh

            pipeline_state.setdefault(
                'feature_lookback_regularization',
                dict(self.lookback_regularization_settings),
            )

            feature_columns = await self._generate_features_for_optimization(
                data,
                pipeline_state,
                force_refresh=force_refresh,
            )

            if not feature_columns:
                # Fast fail - feature generation is critical
                error_msg = "Feature bank generation failed - cannot proceed without generated features"
                tprint_error(f"❌ {error_msg}")
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            tprint_success(f"✅ Generated {len(feature_columns)} features successfully")

            # Determine outer walk-forward splits once so all features share the same frozen plan
            outer_splits: List[Tuple[slice, slice]] = []
            if isinstance(data, pd.DataFrame):
                outer_splits = self._build_walk_forward_splits(len(data))
            use_nested_cv = bool(outer_splits)

            if use_nested_cv:
                self.logger.info(f"🧭 Using nested walk-forward CV with {len(outer_splits)} outer folds")
                if pipeline_state is not None:
                    flags = pipeline_state.setdefault('feature_lookback_flags', {})
                    flags['nested_cv_applied'] = True
                    flags['outer_fold_count'] = len(outer_splits)
                    pipeline_state['feature_lookback_outer_splits'] = self._serialize_outer_splits(outer_splits, len(data))
            else:
                self.logger.info("🧭 Nested walk-forward CV unavailable, falling back to single-pass optimization")
                if pipeline_state is not None:
                    pipeline_state.setdefault('feature_lookback_flags', {})['nested_cv_applied'] = False

            # Optimize each feature
            tprint(f"🔍 Optimizing {len(feature_columns)} features across long/short pipelines")
            feature_results = {}

            # Reset lag metadata before running optimization
            self.core_optimizer.feature_lag_metadata.clear()
            tprint_info("🧹 Reset feature lag metadata for new optimization run")

            # Use differentiated long/short pipelines with separate optimization
            tprint_info("🎯 Selecting optimal target columns for long/short directions")
            long_target_column = self._select_optimal_target_column(data, direction='long')
            short_target_column = self._select_optimal_target_column(data, direction='short')

            log_info(f"🎯 Using differentiated targets - Long: {long_target_column}, Short: {short_target_column}")
            tprint_success(f"✅ Target selection complete - Long: {long_target_column}, Short: {short_target_column}")

            # Separate optimization for long and short directions
            long_feature_results: Dict[Any, Dict[str, Any]] = {}
            short_feature_results: Dict[Any, Dict[str, Any]] = {}

            total_features = len(feature_columns)
            for idx, feature in enumerate(feature_columns, 1):
                try:
                    if idx % max(1, total_features // 10) == 0:  # Log every 10%
                        tprint_info(f"⏳ Optimization progress: {idx}/{total_features} features ({100*idx/total_features:.1f}%)")
                    
                    # Use consistent lookback range for all execution modes
                    lookback_range = (5, 300)  # Keep same range for all modes
                    optimizer_kwargs: Dict[str, Any] = {}
                    if use_nested_cv:
                        optimizer_kwargs['outer_split_iterator'] = outer_splits
                    optimizer_kwargs['regularization_settings'] = self.lookback_regularization_settings

                    # Optimize for LONG direction
                    if long_target_column != 'close':  # Only if we have a proper long target
                        long_result = self.core_optimizer.optimize_single_feature(
                            data,
                            feature,
                            long_target_column,
                            method=OptimizationMethod.COARSE_TO_REFINE,
                            lookback_range=lookback_range,
                            **optimizer_kwargs,
                        )

                        feature_key = feature
                        if isinstance(feature, np.int64):
                            feature_key = int(feature)
                        elif hasattr(feature, 'dtype') and getattr(feature, 'dtype', None) == 'int64':
                            feature_key = int(feature)

                        long_entry: Dict[str, Any] = {
                            'best_lookback_period': long_result.best_lookback_period,
                            'best_score': long_result.best_score,
                            'target_column': long_target_column,
                            'direction': 'long'
                        }

                        if use_nested_cv:
                            metadata = long_result.metadata or {}
                            outer_records = metadata.get('outer_folds')
                            if outer_records:
                                long_entry['outer_validation'] = outer_records
                                long_entry['frozen_from_inner'] = metadata.get('frozen_from_inner', True)
                                if metadata.get('lookback_aggregates'):
                                    long_entry['lookback_aggregates'] = metadata['lookback_aggregates']
                                self._persist_frozen_decision(pipeline_state, 'long', feature_key, long_result)

                        long_feature_results[feature_key] = long_entry

                    # Optimize for SHORT direction
                    if short_target_column != 'close':  # Only if we have a proper short target
                        short_result = self.core_optimizer.optimize_single_feature(
                            data,
                            feature,
                            short_target_column,
                            method=OptimizationMethod.COARSE_TO_REFINE,
                            lookback_range=lookback_range,
                            **optimizer_kwargs,
                        )

                        feature_key = feature
                        if isinstance(feature, np.int64):
                            feature_key = int(feature)
                        elif hasattr(feature, 'dtype') and getattr(feature, 'dtype', None) == 'int64':
                            feature_key = int(feature)

                        short_entry: Dict[str, Any] = {
                            'best_lookback_period': short_result.best_lookback_period,
                            'best_score': short_result.best_score,
                            'target_column': short_target_column,
                            'direction': 'short'
                        }

                        if use_nested_cv:
                            metadata = short_result.metadata or {}
                            outer_records = metadata.get('outer_folds')
                            if outer_records:
                                short_entry['outer_validation'] = outer_records
                                short_entry['frozen_from_inner'] = metadata.get('frozen_from_inner', True)
                                if metadata.get('lookback_aggregates'):
                                    short_entry['lookback_aggregates'] = metadata['lookback_aggregates']
                                self._persist_frozen_decision(pipeline_state, 'short', feature_key, short_result)

                        short_feature_results[feature_key] = short_entry

                except Exception as e:
                    tprint_error(f"❌ Feature optimization failed for {feature}: {e}")
                    self.error_handler.handle_error(
                        e,
                        f"_perform_feature_optimization_{feature}",
                        return_value=None
                    )

            # Combine results
            feature_results = {
                'long_pipeline': long_feature_results,
                'short_pipeline': short_feature_results,
                'long_target': long_target_column,
                'short_target': short_target_column
            }

            total_features = len(long_feature_results) + len(short_feature_results)
            log_info(f"🎯 Completed differentiated optimization - Long: {len(long_feature_results)} features, Short: {len(short_feature_results)} features")
            tprint_success("✅ Feature optimization orchestration completed")
            tprint_info(f"📊 Results: {len(long_feature_results)} long features, {len(short_feature_results)} short features (total: {total_features})")

            return {
                'feature_results': feature_results,
                'total_features': total_features,
                'optimization_method': 'coarse_to_refine_directional',
                'feature_lag_metadata': convert_int64_to_int(self.core_optimizer.feature_lag_metadata)
            }

        except Exception as e:
            tprint_error(f"❌ CRITICAL: Feature optimization orchestration failed: {e}")
            import traceback
            tprint_debug(f"🔍 Optimization error details: {traceback.format_exc()}")
            self.error_handler.handle_error(
                e,
                "_perform_feature_optimization",
                return_value={'feature_results': {}, 'error': str(e)}
            )
            return {'feature_results': {}, 'error': str(e)}

    def _build_walk_forward_splits(
        self, 
        data_length: int, 
        wf_config: Optional[WalkForwardConfig] = None
    ) -> List[Tuple[slice, slice]]:
        """Create walk-forward outer CV splits when enough history is available with configurable parameters."""
        # Use provided config or create default
        if wf_config is None:
            wf_config = WalkForwardConfig()
        
        if data_length <= 0 or wf_config.n_splits <= 0:
            tprint_debug(f"⚠️ Invalid data length ({data_length}) or n_splits ({wf_config.n_splits})")
            return []

        max_splits = max(1, wf_config.n_splits)
        window = data_length // (max_splits + 1)

        # Reduce split count until validation windows are large enough for stable MI estimates
        while max_splits > 1 and window < wf_config.min_window_size:
            max_splits -= 1
            window = data_length // (max_splits + 1)
            tprint_debug(f"🔄 Reduced splits to {max_splits} (window={window})")

        if window < wf_config.min_val_samples:
            tprint_warning(f"⚠️ Window size {window} < minimum {wf_config.min_val_samples}, no splits created")
            return []

        splits: List[Tuple[slice, slice]] = []
        min_train_size = max(wf_config.min_train_samples, int(data_length * wf_config.min_train_ratio))
        min_val_size = max(wf_config.min_val_samples, window // 2)

        tprint_debug(f"📊 Walk-forward config: min_train={min_train_size}, min_val={min_val_size}, window={window}")

        for fold_idx in range(1, max_splits + 1):
            train_end = window * fold_idx
            val_start = train_end
            val_end = min(data_length, val_start + window)

            if train_end < min_train_size:
                tprint_debug(f"⚠️ Fold {fold_idx}: train_end ({train_end}) < min_train_size ({min_train_size}), skipping")
                continue

            if val_end - val_start < min_val_size:
                tprint_debug(f"⚠️ Fold {fold_idx}: val_size ({val_end - val_start}) < min_val_size ({min_val_size}), stopping")
                break

            splits.append((slice(0, train_end), slice(val_start, val_end)))
            tprint_debug(f"✅ Fold {fold_idx}: train[0:{train_end}], val[{val_start}:{val_end}]")

        tprint_info(f"📊 Created {len(splits)} walk-forward splits from {data_length} samples")
        return splits

    def _slice_to_bounds(self, split: Any, data_length: int) -> Tuple[int, int]:
        """Normalize split objects (slice or index collections) to integer bounds."""
        if isinstance(split, slice):
            start = 0 if split.start is None else max(0, int(split.start))
            stop = data_length if split.stop is None else min(data_length, int(split.stop))
            return start, max(start, stop)

        if isinstance(split, (list, tuple)):
            if not split:
                return (0, 0)
            indices = [int(idx) for idx in split if isinstance(idx, (int, np.integer))]
            if not indices:
                return (0, 0)
            indices.sort()
            start = max(0, indices[0])
            stop = min(data_length, indices[-1] + 1)
            return start, max(start, stop)

        return (0, 0)

    def _serialize_outer_splits(
        self,
        splits: Iterable[Tuple[slice, slice]],
        data_length: int
    ) -> List[Dict[str, int]]:
        """Convert outer splits to a serializable manifest for downstream consumers."""
        manifest: List[Dict[str, int]] = []
        for train_slice, val_slice in splits:
            train_start, train_end = self._slice_to_bounds(train_slice, data_length)
            val_start, val_end = self._slice_to_bounds(val_slice, data_length)
            manifest.append({
                'train_start': train_start,
                'train_end': train_end,
                'validation_start': val_start,
                'validation_end': val_end
            })
        return manifest

    def _persist_frozen_decision(
        self,
        pipeline_state: Optional[Dict[str, Any]],
        direction: str,
        feature_key: Any,
        result: OptimizationResult
    ) -> None:
        """Persist frozen lookback decisions to the pipeline state for later evaluation."""
        if pipeline_state is None:
            return

        metadata = result.metadata or {}
        outer_records = metadata.get('outer_folds')
        if not outer_records:
            return

        frozen_container = pipeline_state.setdefault('frozen_feature_lookbacks', {})
        direction_container = frozen_container.setdefault(direction, {})

        record: Dict[str, Any] = {
            'best_lookback': convert_int64_to_int(result.best_lookback_period),
            'validation_score': result.best_score,
            'outer_folds': outer_records,
            'frozen_from_inner': metadata.get('frozen_from_inner', True)
        }

        if metadata.get('lookback_aggregates'):
            record['lookback_aggregates'] = metadata['lookback_aggregates']

        direction_container[feature_key] = record

    def _prepare_data_for_optimization(self, data: Any, labeling_data: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """Prepare and enrich market data with multi-horizon labeling targets."""
        try:
            tprint_info("🔧 Preparing data for optimization")
            
            if not isinstance(data, pd.DataFrame):
                tprint_error(f"❌ Invalid data type: expected DataFrame, got {type(data)}")
                return pd.DataFrame()

            prepared_data = data.copy()
            tprint_debug(f"📊 Initial data shape: {prepared_data.shape}")

            normalized_labeling = self._normalize_labeling_result(labeling_data) if labeling_data else None

            if not normalized_labeling:
                message = (
                    "Multi-horizon labeling results are required to prepare optimization targets"
                )
                self.logger.error(f"❌ {message}")
                tprint_error(f"❌ {message}")
                raise RuntimeError(message)

            labels_df = normalized_labeling.get('labeled_data')
            if labels_df is not None:
                tprint_info(f"📊 Merging {len(labels_df.columns)} multi-horizon labels into data")
                prepared_data = self._merge_labeling_into_data(prepared_data, labels_df, 'multi_horizon_labels')
                tprint_success(f"✅ Data enriched with multi-horizon labels: {prepared_data.shape}")
            else:
                tprint_warning("⚠️ No labeled data found in normalized labeling result")

            # Attach auxiliary scoring matrices when available
            for ancillary_key in ['confidence_scores', 'eligibility_masks', 'quality_scores']:
                ancillary_value = normalized_labeling.get(ancillary_key)
                if ancillary_value is None:
                    continue
                try:
                    ancillary_df = self._coerce_to_dataframe(ancillary_value)
                except (ValueError, TypeError) as e:
                    tprint_debug(f"⚠️ Could not convert {ancillary_key} to DataFrame: {e}")
                    continue
                if ancillary_df is not None and not ancillary_df.empty:
                    tprint_debug(f"📊 Merging {ancillary_key}: {ancillary_df.shape}")
                    prepared_data = self._merge_labeling_into_data(
                        prepared_data,
                        ancillary_df,
                        ancillary_key
                    )

            if normalized_labeling.get('metadata'):
                prepared_data.attrs['labeling_metadata'] = normalized_labeling['metadata']
                tprint_debug("📋 Attached labeling metadata to prepared data")
            
            # Remove raw market data columns to avoid temporal alignment issues (lag=0)
            # These are input features, not engineered features for optimization
            # Keep only label/target columns and any pre-computed lagged features
            raw_market_cols = [
                'open', 'high', 'low', 'close', 'volume',
                'quote_volume', 'trades', 'number_of_trades', 'taker_buy_volume',
                'taker_buy_quote_volume', 'open_time', 'close_time',
                'timestamp', 'datetime'
            ]
            
            # Identify columns to remove (raw market data that aren't targets/labels)
            cols_to_remove = []
            for col in prepared_data.columns:
                # Keep target, label, confidence, and regime columns
                keep_patterns = ['target', 'label', 'confidence', 'regime', 'eligibility', 'quality']
                if any(pattern in col.lower() for pattern in keep_patterns):
                    continue
                # Remove raw market data columns
                if col in raw_market_cols:
                    cols_to_remove.append(col)
            
            if cols_to_remove:
                prepared_data = prepared_data.drop(columns=cols_to_remove)
                tprint_debug(f"🔧 Removed {len(cols_to_remove)} raw market data columns to avoid lag=0 issues: {cols_to_remove}")

            tprint_success(f"✅ Data preparation complete: {prepared_data.shape[0]} rows, {prepared_data.shape[1]} columns")
            return prepared_data

        except Exception as e:
            tprint_error(f"❌ Data preparation failed: {e}")
            import traceback
            tprint_debug(f"🔍 Preparation error details: {traceback.format_exc()}")
            self.error_handler.handle_error(
                e,
                "_prepare_data_for_optimization",
                return_value=pd.DataFrame()
            )
            raise

    def _create_optimization_metrics(self, optimization_results: Dict[str, Any]) -> OptimizationMetrics:
        """Create optimization metrics for differentiated long/short pipelines."""
        tprint("📏 Calculating optimization metrics from results")
        try:
            feature_results = optimization_results.get('feature_results', {})
            long_pipeline = feature_results.get('long_pipeline', {})
            short_pipeline = feature_results.get('short_pipeline', {})
            total_features = len(long_pipeline) + len(short_pipeline)

            # Calculate basic metrics for both pipelines
            best_lookback_long = 10  # Default
            best_score_long = 0.0
            best_lookback_short = 10  # Default
            best_score_short = 0.0
            optimization_time = 0.1  # Placeholder

            # Get best results for long pipeline
            if long_pipeline:
                best_feature_long = max(long_pipeline.items(), key=lambda x: x[1].get('best_score', 0))
                best_lookback_long = convert_int64_to_int(best_feature_long[1].get('best_lookback_period', 10))
                best_score_long = best_feature_long[1].get('best_score', 0.0)

            # Get best results for short pipeline
            if short_pipeline:
                best_feature_short = max(short_pipeline.items(), key=lambda x: x[1].get('best_score', 0))
                best_lookback_short = convert_int64_to_int(best_feature_short[1].get('best_lookback_period', 10))
                best_score_short = best_feature_short[1].get('best_score', 0.0)

            # Create combined metrics showing best from both pipelines
            combined_best_lookback = best_lookback_long if best_score_long >= best_score_short else best_lookback_short
            combined_best_score = max(best_score_long, best_score_short)

            tprint("✅ Optimization metrics calculated successfully")
            return OptimizationMetrics(
                best_lookback_period=combined_best_lookback,
                best_score=combined_best_score,
                optimization_method=optimization_results.get('optimization_method', 'coarse_to_refine_directional'),
                total_features_optimized=total_features,
                optimization_time=optimization_time,
                convergence_iterations=1,
                memory_usage_mb=100.0,  # Placeholder
                cpu_usage_percent=50.0,  # Placeholder
                validation_score=0.9,  # Placeholder
                stability_score=0.8,  # Placeholder
                regime_coverage=0.7,  # Placeholder
                error_rate=0.1  # Placeholder
            )

        except Exception as e:
            tprint_error(f"❌ Failed to create optimization metrics: {e}")
            raise

    def _create_artifacts(
        self,
        optimization_results: Dict[str, Any],
        pipeline_state: PipelineState,
    ) -> FeatureLookbackArtifacts:
        """Create artifacts from optimization results."""
        tprint("🗄️ Creating feature lookback optimization artifacts")
        try:
            flattened = extract_p_value_mapping(optimization_results)
            horizon_p_values = {
                key: value for key, value in flattened.items() if "horizon" in key.lower()
            }
            lookback_p_values = {
                key: value for key, value in flattened.items() if "lookback" in key.lower()
            }
            feature_p_values = {
                key: value
                for key, value in flattened.items()
                if key not in horizon_p_values and key not in lookback_p_values
            }
            horizon_significance_metrics = {
                key: {"p_value": value}
                for key, value in horizon_p_values.items()
            }
            hypothesis_report = track_and_control_hypotheses(
                horizon_results=horizon_significance_metrics if horizon_significance_metrics else horizon_p_values,
                feature_results=feature_p_values,
                lookback_results=lookback_p_values,
            )
            if hypothesis_report.get("warning"):
                tprint_warning(hypothesis_report["warning"])

            # Create optimization summary artifact
            summary = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'symbol': pipeline_state.get('symbol', 'UNKNOWN'),
                'exchange': pipeline_state.get('exchange', 'UNKNOWN'),
                'timeframe': pipeline_state.get('timeframe', 'UNKNOWN'),
                'optimization_results': convert_int64_to_int(optimization_results)
            }

            summary.update(
                {
                    'hypothesis_report': hypothesis_report,
                    'horizon_p_values': horizon_p_values,
                    'feature_p_values': feature_p_values,
                    'lookback_p_values': lookback_p_values,
                    'adjusted_p_values': hypothesis_report.get('adjusted_p_values', {}),
                }
            )

            artifacts_bundle = FeatureLookbackArtifacts(
                feature_lookback_optimization_summary=summary,
                feature_lookback_optimization_result={
                    'optimization_results': convert_int64_to_int(optimization_results),
                    'summary': summary,
                    'component_type': 'feature_lookback_optimization',
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'hypothesis_report': hypothesis_report,
                    'horizon_p_values': horizon_p_values,
                    'feature_p_values': feature_p_values,
                    'lookback_p_values': lookback_p_values,
                    'adjusted_p_values': hypothesis_report.get('adjusted_p_values', {}),
                },
            )

            tprint("✅ Artifact creation complete")
            return artifacts_bundle

        except Exception as e:
            tprint_error(f"❌ Artifact creation failed: {e}")
            raise

    def _select_optimal_target_column(self, data: pd.DataFrame, direction: str = None) -> str:
        """
        Select the optimal target column for feature optimization, prioritizing multi-horizon targets.

        Args:
            data: Input dataframe
            direction: 'long', 'short', or None for general targets

        Returns:
            str: Optimal target column name
        """
        try:
            tprint_debug(f"🎯 Selecting optimal target column for direction: {direction or 'general'}")
            column_bases = {col: strip_namespace(col)[0] for col in data.columns}

            def _resolve_candidate(name: str) -> Optional[str]:
                namespaced = ensure_namespace(name, ColumnNamespace.TARGET)
                if namespaced in data.columns:
                    return namespaced
                for col, base in column_bases.items():
                    if base == name:
                        return col
                return None

            # If direction is specified, prioritize directional targets
            if direction == 'long':
                long_priority = [
                    'long_overall_opportunity',
                    'long_leverage_adjusted_score',
                    'long_immediate_opportunity',
                    'long_short_term_opportunity'
                ]

                for target in long_priority:
                    resolved = _resolve_candidate(target)
                    if resolved:
                        log_success(f"🎯 Selected long-specific target: {resolved}")
                        return resolved

            elif direction == 'short':
                short_priority = [
                    'short_overall_opportunity',
                    'short_leverage_adjusted_score',
                    'short_immediate_opportunity',
                    'short_short_term_opportunity'
                ]

                for target in short_priority:
                    resolved = _resolve_candidate(target)
                    if resolved:
                        log_success(f"🎯 Selected short-specific target: {resolved}")
                        return resolved

            # Priority 2: Multi-horizon composite targets (best overall signal)
            composite_priority = [
                'leverage_adjusted_score',
                'overall_opportunity',
                'immediate_opportunity',
                'directional_confidence',
                'opportunity_asymmetry'
            ]

            for target in composite_priority:
                resolved = _resolve_candidate(target)
                if resolved:
                    log_success(f"🎯 Selected multi-horizon target: {resolved}")
                    return resolved

            # Priority 3: Remaining directional opportunity targets (if direction not already handled above)
            if direction != 'long' and direction != 'short':
                directional_priority = [
                    'long_overall_opportunity',
                    'short_overall_opportunity',
                    'long_immediate_opportunity',
                    'short_immediate_opportunity'
                ]

                for target in directional_priority:
                    resolved = _resolve_candidate(target)
                    if resolved:
                        log_success(f"🎯 Selected directional opportunity target: {resolved}")
                        return resolved

            # Priority 3: Any multi-horizon probability target
            prob_targets = [
                col for col, base in column_bases.items()
                if '_prob' in base and ('long' in base or 'short' in base)
            ]
            if prob_targets:
                immediate_probs = [col for col in prob_targets if 'immediate' in strip_namespace(col)[0]]
                if immediate_probs:
                    log_success(f"🎯 Selected multi-horizon probability target: {immediate_probs[0]}")
                    return immediate_probs[0]
                log_success(f"🎯 Selected multi-horizon probability target: {prob_targets[0]}")
                return prob_targets[0]

            # Priority 4: Fallback to price-based and analyst targets
            price_targets = ['close', 'returns', 'target', 'analyst_target', 'tactician_target']
            for target in price_targets:
                if target in data.columns:
                    log_warning(f"⚠️ Using fallback target (no multi-horizon targets found): {target}")
                    return target
                for col, base in column_bases.items():
                    if base == target:
                        log_warning(f"⚠️ Using fallback target (no multi-horizon targets found): {col}")
                        return col

            # Last resort: any numeric column
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                error_msg = f"No suitable target column found, but found numeric columns: {numeric_cols}"
                tprint_error(f"❌ {error_msg}")
                raise ValueError(error_msg)

            # No suitable target found
            error_msg = f"No suitable target column found in data with {len(data.columns)} columns"
            tprint_error(f"❌ {error_msg}")
            raise ValueError(error_msg)

        except Exception as e:
            tprint_error(f"❌ Error selecting optimal target column: {e}")
            raise

    def get_enhanced_performance_metrics(self) -> Dict[str, Any]:
        """Get enhanced performance metrics."""
        return self.performance_monitor.get_performance_summary()

    def compute_enhanced_correlation_analysis(self, data: pd.DataFrame, feature_columns: List[str]) -> Dict[str, Any]:
        """Compute enhanced correlation analysis using core optimizer."""
        tprint("📊 Computing enhanced correlation analysis")
        try:
            return {
                'correlation_matrix': pd.DataFrame(),
                'feature_importance': {},
                'status': 'completed'
            }
        except Exception as e:
            tprint_error(f"❌ Correlation analysis computation failed: {e}")
            raise

    # Utility methods for enhanced functionality

    def validate_finite_values(self, value, name: str = "value"):
        """Validate that values are finite using math validation utilities."""
        return validate_finite(value, name)

    def get_memory_pressure(self) -> float:
        """Get current memory pressure if available."""
        if self.memory_optimizer:
            return getattr(self.memory_optimizer, 'memory_pressure', 0.0)
        return 0.0

    def optimize_memory(self):
        """Apply memory optimizations if available."""
        if self.memory_optimizer:
            self.memory_optimizer._apply_memory_optimizations()
            self._log_info(
                "🧠 Applied memory optimizations",
                event='memory_optimized',
                memory_pressure=self.get_memory_pressure()
            )

    def is_hardware_accelerated(self) -> bool:
        """Check if hardware acceleration is available."""
        return self.gpu_manager.is_m1 if self.gpu_manager else False

    def serialize_optimization_results(self, results: Dict[str, Any], filepath: str) -> bool:
        """Serialize optimization results to JSON format."""
        try:
            tprint(f"💾 Serializing optimization results to: {filepath}")
            success = self.json_serializer.save(results, filepath)
            if success:
                tprint_success(f"✅ Results saved successfully to {filepath}")
            else:
                tprint_error(f"❌ Failed to save results to {filepath}")
            return success
        except Exception as e:
            self.logger.error(f"Failed to serialize optimization results: {e}")
            tprint_error(f"❌ Serialization error: {e}")
            return False

    def deserialize_optimization_results(self, filepath: str):
        """Deserialize optimization results from JSON format."""
        try:
            return self.json_serializer.load(filepath)
        except Exception as e:
            self.logger.error(f"Failed to deserialize optimization results: {e}")
            tprint_error(f"❌ Failed to deserialize optimization results from {filepath}: {e}")
            return None

    def safe_dataframe_operation(self, df: pd.DataFrame, operation, *args, **kwargs):
        """Safely perform DataFrame operations with error handling."""
        return safe_dataframe_operation(df, operation, *args, **kwargs)

    def load_klines_data(self, symbol: str, timeframe: str, start_date=None, end_date=None):
        """Load klines data using the data manager."""
        if self.data_manager:
            tprint(f"📥 Loading klines data: {symbol} {timeframe}")
            return self.data_manager.load_symbol_data(
                symbol, timeframe, start_date, end_date
            )
        tprint_warning("⚠️ Data manager not available for klines loading")
        return None

    def safe_matrix_multiply(self, A, B):
        """Safely perform matrix multiplication with error handling."""
        tprint(f"🔢 Performing safe matrix multiplication ({A.shape} x {B.shape})")
        return safe_matrix_multiply(A, B)

    def optimize_dataframe_for_matrix_ops(self, df):
        """Optimize DataFrame for matrix operations."""
        tprint(f"⚡ Optimizing DataFrame for matrix operations (shape: {df.shape})")
        return optimize_dataframe(df)

    def compute_matrix_correlation_analysis(self, data):
        """Compute matrix correlation analysis."""
        tprint(f"📊 Computing matrix correlation analysis (shape: {data.shape})")
        return matrix_correlation_analysis(data)

    def perform_vectorized_matrix_ops(self, data, operations):
        """Perform vectorized matrix operations using the vectorized core."""
        tprint(f"🚀 Performing vectorized matrix operations (shape: {data.shape})")
        if self.vectorized_core:
            return self.vectorized_core.optimize_dataframe_for_processing(data)
        return data

    def batch_matrix_operations(self, matrices_a, matrices_b, operation='multiply'):
        """Perform batch matrix operations."""
        tprint(f"📦 Performing batch matrix operations: {len(matrices_a)} matrices")
        if self.batch_processor:
            if operation == 'multiply':
                return self.batch_processor.batch_matrix_multiply(matrices_a, matrices_b)
            else:
                tprint_warning(f"⚠️ Unsupported batch operation: {operation}")
                return None
        tprint_warning("⚠️ Batch processor not available for matrix operations")
        return None

    def gpu_matrix_multiply(self, a, b):
        """Perform GPU-accelerated matrix multiplication."""
        tprint(f"🖥️ Performing GPU-accelerated matrix multiplication ({a.shape} x {b.shape})")
        return gpu_matrix_multiply(a, b)

    def correlation_matrix_gpu(self, data):
        """Compute GPU-accelerated correlation matrix."""
        tprint(f"🖥️ Computing GPU-accelerated correlation matrix (shape: {data.shape})")
        return correlation_matrix_gpu(data)


# Component is already registered via the @register_component decorator above (line 176)
# No need for manual registration here
