"""
Pre-Training Sub-Pipeline - Feature Engineering Steps

This module provides the pre-training sub-pipeline with the 4 feature engineering steps
that were moved from market_analysis:

1. analyst_profit_labeler OR tactician_entry_labeler - Role-specific profit labeling (automatic selection based on timeframe)
2. feature_lookback_optimization - Optimize feature lookback periods
3. interactive_feature_generation - End-to-end interactive feature generation with comprehensive approach
4. final_feature_selection - Final multi-stage feature selection (120→100→80→60)

TIMEFRAME RESOLUTION:
====================

Each step can receive a timeframe parameter. The resolution follows a strict priority order:

1. Explicit parameter (passed directly to the step)
2. Custom parameters 'timeframe' key
3. Pipeline overrides 'timeframe' key
4. Global primary timeframe configuration (from universal_timeframe_config)
5. Default fallback: '15m'

Analyst orchestrations automatically elevate to '60m' to align with their higher
granularity requirements for strategic decision-making.

The resolution logic is centralized in SubPipelineConfig.resolve_timeframe() method
to ensure consistency across all steps and prevent scattered timeframe logic.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Set, Tuple, TypedDict
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
import numpy as np
import json
import socket
import subprocess
import hashlib
import os
import traceback
import uuid
import time

# Import core utilities for enhanced functionality
from src.utils.common_operations import (
    safe_json_load, safe_json_dump, ensure_directory, safe_file_exists,
    validate_dataframe, validate_dataframe_columns, safe_dataframe_operation,
    safe_merge_dataframes, safe_drop_columns, safe_rename_columns,
    calculate_data_quality_metrics, get_dataframe_info, optimize_dataframe_dtypes,
    safe_rolling, safe_groupby_operation, safe_apply_function, safe_filter_dataframe,
    safe_correlation, safe_float, safe_int, validate_finite, validate_positive,
    safe_divide, safe_mean, safe_std, format_bytes, timed_operation,
    get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
    memory_checkpoint, gpu_context, optimize_memory, get_memory_usage
)
from src.utils.tprint import tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, tprint_success

# Import additional hardware optimization tools for caching and memory monitoring
from src.utils.hardware.unified_hardware_manager import (
    get_unified_hardware_manager,
    HardwareConfig,
    WorkloadType,
    OptimizationLevel
)
try:
    from src.utils.math_validation import safe_math_operation
except ImportError:
    # Fallback if math_validation doesn't have the expected functions
    def safe_math_operation(func, *args, **kwargs):
        """Fallback safe math operation."""
        try:
            return func(*args, **kwargs)
        except Exception:
            return None

DEFAULT_STEP_TIME_BUDGETS: Dict[str, float] = {
    'multi_horizon_profit_labeler': 600.0,
    'feature_lookback_optimization': 900.0,
    'interactive_feature_generation': 1200.0,
    'final_feature_selection': 600.0,
}

def _default_data_locator_config() -> DataLocatorConfig:
    """Build a :class:`DataLocatorConfig` from the global settings."""

    return get_pre_training_settings().to_data_locator_config()

@dataclass(frozen=True)
class StepSpec:
    """Specification describing an executable pre-training step."""

    name: str
    component_key: str
    executor_method: str
    display_name: str
    description: str
    order: int
    enabled: bool = True
    include_in_default_sequence: bool = True

STEP_REGISTRY: Dict[str, StepSpec] = {
    'analyst-labeler': StepSpec(
        name='analyst-labeler',
        component_key='analyst_profit_labeler',
        executor_method='_execute_analyst_profit_labeler',
        display_name='1a. Analyst Labeler',
        description='Apply triple barrier method-inspired, per-regime, volatility and noise-aware labeling (15m timeframe).',
        order=1,
    ),
    'tactician-labeler': StepSpec(
        name='tactician-labeler',
        component_key='tactician_entry_labeler',
        executor_method='_execute_tactician_entry_labeler',
        display_name='1b. Tactician Labeler',
        description='Apply triple barrier method-inspired, per-regime, volatility and noise-aware labeling (15m timeframe).',
        order=2,
    ),
    'feature_generation_data_validation_step': StepSpec(
        name='feature_generation_data_validation_step',
        component_key='feature_generation_data_validation_step',
        executor_method='_execute_feature_generation_data_validation_step',
        display_name='2. Data Validation Step',
        description='Data validation and quality assessment for feature generation pipeline.',
        order=3,
    ),
    'feature_generation_labeling_integration_step': StepSpec(
        name='feature_generation_labeling_integration_step',
        component_key='feature_generation_labeling_integration_step',
        executor_method='_execute_feature_generation_labeling_integration_step',
        display_name='3. Labeling Integration Step',
        description='Analyst/Tactician labeling integration for unified pipeline.',
        order=4,
    ),
    'feature_generation_feature_generation_step': StepSpec(
        name='feature_generation_feature_generation_step',
        component_key='feature_generation_feature_generation_step',
        executor_method='_execute_feature_generation_feature_generation_step',
        display_name='4. Feature Generation Step',
        description='Multi-method feature generation for unified pipeline.',
        order=5,
    ),
    'feature_generation_feature_selection_step': StepSpec(
        name='feature_generation_feature_selection_step',
        component_key='feature_generation_feature_selection_step',
        executor_method='_execute_feature_generation_feature_selection_step',
        display_name='5. Feature Selection Step',
        description='Intelligent feature selection for unified pipeline.',
        order=6,
    ),
    'feature_generation_period_lookback_optimization_step': StepSpec(
        name='feature_generation_period_lookback_optimization_step',
        component_key='feature_generation_period_lookback_optimization',
        executor_method='_execute_feature_generation_period_lookback_optimization',
        display_name='6. Period + Lookback Optimization Step',
        description='Combined period and lookback optimization for features in unified pipeline.',
        order=7,
    ),
    'feature_generation_interaction_generation_step': StepSpec(
        name='feature_generation_interaction_generation_step',
        component_key='feature_generation_interaction_generation_step',
        executor_method='_execute_feature_generation_interaction_generation_step',
        display_name='7. Interaction Generation Step',
        description='Feature interaction generation for unified pipeline.',
        order=8,
    ),
    'feature_generation_vectorization_step': StepSpec(
        name='feature_generation_vectorization_step',
        component_key='feature_generation_vectorization_step',
        executor_method='_execute_feature_generation_vectorization_step',
        display_name='8. Vectorization Step',
        description='Feature vectorization optimization for unified pipeline.',
        order=9,
    ),
    'feature_generation_final_validation_step': StepSpec(
        name='feature_generation_final_validation_step',
        component_key='feature_generation_final_validation_step',
        executor_method='_execute_feature_generation_final_validation_step',
        display_name='9. Final Validation Step',
        description='Final validation and quality check for unified pipeline.',
        order=10,
    ),
}

STEP_PROGRESS_ICONS: Dict[str, str] = {
    'analyst-labeler': '📈',
    'tactician-labeler': '🎲',
    'feature_generation_data_validation_step': '🔍',
    'feature_generation_labeling_integration_step': '🔗',
    'feature_generation_feature_generation_step': '⚙️',
    'feature_generation_feature_selection_step': '🎯',
    'feature_generation_period_lookback_optimization_step': '📊',
    'feature_generation_interaction_generation_step': '🔧',
    'feature_generation_vectorization_step': '🚀',
    'feature_generation_final_validation_step': '✅',
}

try:  # pragma: no cover - platform specific import
    import resource
except ImportError:  # pragma: no cover
    resource = None

from src.utils.logger import system_logger
from src.utils.enhanced_artifact_manager import get_artifact_manager
from src.utils.version_manager import get_version_manager
from src.utils.random_seeding import SeededRNGs, seed_rngs, set_global_seed
from src.utils.tprint import tprint, tprint_error, tprint_warning
from .logging_utils import (
    PreTrainingEventLogger,
    StepLogContext,
    configure_pre_training_logging,
)

# Import component system
from .components import ComponentFactory
from .components.component_factory import ComponentConfig
from .metrics_sink import MetricsSink, MetricsSinkConfig
from src.training.config.data_locator import DataLocator, DataLocatorConfig, LocatorPaths
from src.training.steps.pre_training.validation.data_contracts import (
    DataContractValidationError,
    validate_feature_artifact,
    validate_multi_horizon_labeling_result,
    validate_selection_artifact,
)
from src.training.steps.pre_training.validation.schemas import (
    validate_raw_ohlcv,
    validate_engineered_features,
    SchemaValidationException,
)
from .settings import get_pre_training_settings
from src.training.common.component_result import ComponentError
from src.utils.ml_common.config.universal_timeframe_config import get_primary_timeframe

logger = system_logger.getChild('PreTrainingSubPipeline')

class ValidationCache:
    """Cache system for expensive validation operations."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}

    def _make_key(self, operation: str, df_hash: str, context: str = "") -> str:
        """Generate a unique cache key for the validation operation."""
        key_components = [operation, df_hash]
        if context:
            key_components.append(context)
        return hashlib.md5("|".join(key_components).encode()).hexdigest()

    def _get_dataframe_hash(self, df: pd.DataFrame) -> str:
        """Generate a hash of the DataFrame for caching purposes."""
        try:
#             # Use shape, columns, and sample of data for hashing
            sample = df.head(1000) if len(df) > 1000 else df
            hash_input = f"{df.shape}|{list(df.columns)}|{sample.to_string()}"
            return hashlib.md5(hash_input.encode()).hexdigest()
        except Exception:
#             # Fallback to basic shape-based hash
            return hashlib.md5(f"{df.shape}|{list(df.columns)}".encode()).hexdigest()

    def get(self, operation: str, df: pd.DataFrame, context: str = "") -> Optional[Any]:
        """Retrieve cached validation result."""
        df_hash = self._get_dataframe_hash(df)
        cache_key = self._make_key(operation, df_hash, context)

        # Check if key exists and hasn't expired
        if cache_key in self.cache:
            cached_item = self.cache[cache_key]
            if time.time() - cached_item['timestamp'] < self.ttl_seconds:
                self.access_times[cache_key] = time.time()
                logger.debug(f"🗂️ Cache hit for {operation} validation")
                return cached_item['result']

        return None

    def put(self, operation: str, df: pd.DataFrame, result: Any, context: str = "") -> None:
        """Store validation result in cache."""
        df_hash = self._get_dataframe_hash(df)
        cache_key = self._make_key(operation, df_hash, context)

        # Implement LRU eviction if cache is full
        if len(self.cache) >= self.max_size and cache_key not in self.cache:
#             # Remove oldest accessed item
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]

        # Store the result
        self.cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }
        self.access_times[cache_key] = time.time()

        logger.debug(f"🗂️ Cached {operation} validation result")

    def clear(self) -> None:
        """Clear all cached validation results."""
        self.cache.clear()
        self.access_times.clear()
        logger.info("🗂️ Validation cache cleared")

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_ratio': self._calculate_hit_ratio()
        }

    def _calculate_hit_ratio(self) -> float:
        """Calculate cache hit ratio (simplified)."""
        # This would need more sophisticated tracking in a real implementation
        return 0.0

class MemoryAwareValidationManager:
    """Manager for memory-aware validation operations with hardware optimization."""

    def __init__(self):
        self.memory_optimizer = get_m1_memory_optimizer()
        self.hardware_manager = get_unified_hardware_manager()
        self.validation_cache = ValidationCache()
        self.memory_checkpoints: List[str] = []

    def start_validation_session(self, component_name: str) -> str:
        """Start a validation session with memory monitoring."""
        session_id = f"{component_name}_{uuid.uuid4().hex[:8]}"

        # Configure hardware for intensive validation workload
        hardware_config = HardwareConfig(
            memory_optimization_level=OptimizationLevel.AGGRESSIVE,
            enable_memory_pooling=True,
            enable_predictive_allocation=True,
            intensive_thresholds={
                'memory_usage': 95.0,  # Allow high memory usage for validation
                'cpu_usage': 90.0
            }
        )

        # Initialize hardware manager for validation workload
        self.hardware_manager.config = hardware_config
        self.hardware_manager.initialize()

        # Start memory monitoring
        self.memory_optimizer.start_monitoring()

        # Create memory checkpoint
        checkpoint_id = f"validation_{session_id}"
        self.memory_checkpoints.append(checkpoint_id)

        logger.info(f"🧠 Started memory-aware validation session: {session_id}")
        return session_id

    def end_validation_session(self, session_id: str) -> Dict[str, Any]:
        """End validation session and return memory statistics."""
        # Stop memory monitoring
        self.memory_optimizer.stop_monitoring()

        # Get memory statistics
        memory_stats = self.memory_optimizer.get_memory_stats()
        cache_stats = self.validation_cache.stats()

        # Clear hardware optimizations
        self.hardware_manager.shutdown()

        logger.info(f"🧠 Ended validation session {session_id}")

        return {
            'session_id': session_id,
            'memory_stats': memory_stats,
            'cache_stats': cache_stats,
            'memory_checkpoints_used': len(self.memory_checkpoints)
        }

    def validate_with_memory_monitoring(
        self,
        operation: str,
        df: pd.DataFrame,
        validation_func: Callable,
        context: str = ""
    ) -> Any:
        """Perform validation with memory monitoring and caching."""

        # Check cache first
        cached_result = self.validation_cache.get(operation, df, context)
        if cached_result is not None:
            return cached_result

        # Create memory checkpoint for this validation
        with memory_checkpoint(f"validation_{operation}"):
#             # Get initial memory state
            initial_memory = get_memory_usage()

            try:
#                 # Perform validation
                result = validation_func(df)

#                 # Store in cache
                self.validation_cache.put(operation, df, result, context)

#                 # Log memory usage (simplified for now)
                final_memory = get_memory_usage()
                memory_used = final_memory - initial_memory

                if memory_used > 100 * 1024 * 1024:  # Log if > 100MB used
                    logger.info(f"🧠 Validation {operation} used {memory_used / 1024 / 1024:.1f} MB")

                return result

            except Exception as e:
                logger.error(f"❌ Validation {operation} failed: {e}")
                raise

    def optimize_dataframe_for_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for validation operations."""
        with memory_checkpoint("dataframe_optimization"):
            optimized_df = self.memory_optimizer.optimize_dataframe_memory(df)

#             # Apply additional optimizations for validation
            if hasattr(optimized_df, 'memory_usage'):
                initial_memory = optimized_df.memory_usage(deep=True).sum()

#                 # Convert to more memory-efficient dtypes where safe
                for col in optimized_df.select_dtypes(include=['float64']):
                    if optimized_df[col].isna().sum() == 0:  # No missing values
                        try:
                            optimized_df[col] = optimized_df[col].astype('float32')
                        except (ValueError, TypeError):
                            pass  # Keep original dtype if conversion fails

                final_memory = optimized_df.memory_usage(deep=True).sum()
                saved_memory = initial_memory - final_memory

                if saved_memory > 0:
                    logger.debug(f"🧠 DataFrame optimized for validation: {saved_memory / 1024 / 1024:.1f} MB saved")

            return optimized_df

class UnexpectedArtifactKeyError(RuntimeError):
    """Raised when a component emits artifacts outside the documented schema."""

    def __init__(self, step_name: str, unexpected_keys: Iterable[str]):
        keys = sorted(str(key) for key in unexpected_keys)
        message = (
            f"Step '{step_name}' produced unexpected artifact keys: {', '.join(keys)}"
        )
        super().__init__(message)
        self.step_name = step_name
        self.keys: Tuple[str, ...] = tuple(keys)

class PipelineState(dict):
    """Mutable mapping describing the canonical pre-training pipeline state.

    The state exposes a dictionary-like interface for backwards compatibility
    while constraining which artifact keys each component may contribute. The
    keys are grouped by component:

    * ``analyst-labeler``
        - ``multi_horizon_labeling_result``: Validated labeling payload.
        - ``labeling_report``: Structured diagnostic report.
        - ``standardized_output``: Normalised label view for downstream steps.
        - ``validated_schemas``: Schema metadata applied during validation.
    * ``tactician-labeler``
        - ``multi_horizon_labeling_result``: Validated labeling payload.
        - ``labeling_report``: Structured diagnostic report.
        - ``standardized_output``: Normalised label view for downstream steps.
        - ``validated_schemas``: Schema metadata applied during validation.
    * ``feature_generation_data_validation_step``
        - ``data_validation_result``: Data validation results.
        - ``validation_report``: Validation diagnostic report.
        - ``validated_schemas``: Schema metadata for validation outputs.
    * ``feature_generation_labeling_integration_step``
        - ``labeling_integration_result``: Integrated labeling results.
        - ``integrated_labels``: Combined analyst/tactician labels.
        - ``validated_schemas``: Schema metadata for integrated outputs.
    * ``feature_generation_feature_generation_step``
        - ``feature_generation_result``: Generated feature results.
        - ``generated_features``: Feature dataset.
        - ``validated_schemas``: Schema metadata for generated features.
    * ``feature_generation_feature_selection_step``
        - ``feature_selection_result``: Feature selection results.
        - ``selected_features``: Selected feature dataset.
        - ``validated_schemas``: Schema metadata for selection outputs.
    * ``feature_generation_period_lookback_optimization_step``
        - ``period_optimization_result``: Period optimization results.
        - ``lookback_optimization_result``: Lookback optimization results.
        - ``combined_optimization_result``: Combined optimization results.
        - ``validated_schemas``: Schema metadata for optimization outputs.
    * ``feature_generation_interaction_generation_step``
        - ``interaction_generation_result``: Interaction generation results.
        - ``interaction_features``: Generated interaction features.
        - ``validated_schemas``: Schema metadata for interaction outputs.
    * ``feature_generation_vectorization_step``
        - ``vectorization_result``: Vectorization results.
        - ``vectorized_features``: Vectorized feature dataset.
        - ``validated_schemas``: Schema metadata for vectorization outputs.
    * ``feature_generation_final_validation_step``
        - ``final_validation_result``: Final validation results.
        - ``validation_report``: Final validation report.
        - ``validated_schemas``: Schema metadata for final validation outputs.

    Additional non-artifact keys (e.g. ``random_seed`` or
    ``regime_data_splitting_result``) are written directly by the pipeline and
    remain unconstrained. Any unexpected artifact keys cause an
    :class:`UnexpectedArtifactKeyError` to be raised so upstream bugs are
    surfaced early.
    """

    #: Allowed artifact keys per pipeline step.
    _STEP_ARTIFACT_KEYS: Dict[str, frozenset[str]] = {
        'analyst-labeler': frozenset({
            'multi_horizon_labeling_result',
            'labeling_report',
            'standardized_output',
            'validated_schemas',
        }),
        'tactician-labeler': frozenset({
            'multi_horizon_labeling_result',
            'labeling_report',
            'standardized_output',
            'validated_schemas',
        }),
        'feature_generation_data_validation_step': frozenset({
            'data_validation_result',
            'validation_report',
            'validated_schemas',
        }),
        'feature_generation_labeling_integration_step': frozenset({
            'labeling_integration_result',
            'integrated_labels',
            'validated_schemas',
        }),
        'feature_generation_feature_generation_step': frozenset({
            'feature_generation_result',
            'generated_features',
            'validated_schemas',
        }),
        'feature_generation_feature_selection_step': frozenset({
            'feature_selection_result',
            'selected_features',
            'validated_schemas',
        }),
        'feature_generation_period_lookback_optimization_step': frozenset({
            'period_optimization_result',
            'lookback_optimization_result',
            'combined_optimization_result',
            'validated_schemas',
        }),
        'feature_generation_interaction_generation_step': frozenset({
            'interaction_generation_result',
            'interaction_features',
            'validated_schemas',
        }),
        'feature_generation_vectorization_step': frozenset({
            'vectorization_result',
            'vectorized_features',
            'validated_schemas',
        }),
        'feature_generation_final_validation_step': frozenset({
            'final_validation_result',
            'validation_report',
            'validated_schemas',
        }),
    }

    def merge_step_artifacts(
        self,
        step_name: str,
        artifacts: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Merge validated step artifacts into the pipeline state.

        Args:
            step_name: Registry name of the step emitting ``artifacts``.
            artifacts: Mapping of artifact keys to payloads.

        Returns:
            Subset of ``artifacts`` containing only schema-approved keys.

        Raises:
            UnexpectedArtifactKeyError: If ``artifacts`` contains unexpected
                keys for ``step_name``.
        """

        if not artifacts:
            return {}

        allowed_keys = self._STEP_ARTIFACT_KEYS.get(step_name)
        if allowed_keys is None:
            raise UnexpectedArtifactKeyError(step_name, artifacts.keys())

        unexpected = set(artifacts) - allowed_keys
        if unexpected:
            raise UnexpectedArtifactKeyError(step_name, unexpected)

        merged: Dict[str, Any] = {
            key: artifacts[key]
            for key in allowed_keys
            if key in artifacts
        }
        super().update(merged)
        return merged

class ExecutionMode(Enum):
    """Execution modes for sub-pipelines."""
    FULL = "full"          # Complete execution with all features
    LIGHT = "light"        # Lightweight execution with essential features only
    BLANK = "blank"        # Minimal execution for testing/validation

class SubPipelineStatus(Enum):
    """Status of sub-pipeline execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class LoggingConfig:
    """Logging configuration for the sub-pipeline."""
    level: str = "INFO"
    enable_console: bool = True
    enable_file: bool = False
    log_file: Optional[str] = None

@dataclass
class SubPipelineConfig:
    """Configuration for sub-pipeline execution.

    The timeframe is resolved in the following priority order:

    1. Explicit ``timeframe`` argument supplied to the config
    2. Overrides supplied via ``custom_params`` or ``pipeline`` metadata
    3. The globally configured primary timeframe
    4. A final fallback to ``'15m'``

    Analyst-oriented runs (identified by their role metadata) are always promoted to
    ``'60m'`` regardless of the earlier sources to preserve expected aggregation.
    """

    mode: ExecutionMode = ExecutionMode.FULL
    symbol: str = "ETHUSDT"
    exchange: str = "binance"
    timeframe: Optional[str] = None  # Resolved during initialization
    data_dir: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    # Direction control for trading (used by labeling components)
    enable_long_positions: bool = True
    enable_short_positions: bool = True
    force_rerun: bool = False
    parallel_processing: bool = True
    max_workers: int = 4
    validation_enabled: bool = True
    monitoring_enabled: bool = True
    fast_mode: bool = False
    skip_next_pipeline: bool = False
    custom_params: Dict[str, Any] = field(default_factory=dict)
    random_seed: Optional[int] = None
    pipeline: Dict[str, Any] = field(default_factory=dict)

    # Direction control for trading (inherited from main pipeline config)
    enable_long_positions: bool = True
    enable_short_positions: bool = True

    label_imbalance_warning_threshold: float = 0.75
    nan_rate_warning_threshold: float = 0.05
    duplicate_index_warning_threshold: float = 0.02
    metrics_output_path: Optional[str] = None
    metrics_output_format: str = "csv"
    metrics_prometheus_enabled: bool = False
    step_time_budgets: Dict[str, float] = field(default_factory=lambda: DEFAULT_STEP_TIME_BUDGETS.copy())
    market_data_batch_size: Optional[int] = None
    market_data_window_days: Optional[int] = None
    data_locator_config: DataLocatorConfig = field(default_factory=_default_data_locator_config)
    data_locator: Optional[DataLocator] = None
    data_dir_key: str = "market_data"
    cache_dir_key: str = "default"
    artifacts_dir_key: str = "default"
    generated_dir_key: str = "market_analysis"
    outcomes_dir_key: str = "multi_horizon_outcomes"
    use_existing_data: bool = False
    final_feature_selection_dir_key: str = "final_feature_selection"
    _path_view: Optional[LocatorPaths] = field(default=None, init=False, repr=False)
    """
    Metrics capture configuration.

    Defaults:
        metrics_output_path: ``artifacts/pre_training_metrics.<format>``
        metrics_output_format: ``csv``
        metrics_prometheus_enabled: ``False``
    """

    def __post_init__(self) -> None:
        custom_params_map = self._normalise_mapping(self.custom_params)
        pipeline_map = self._normalise_mapping(self.pipeline)

        resolved_timeframe = self.resolve_timeframe(
            explicit=self.timeframe,
            custom_params=custom_params_map,
            pipeline_overrides=pipeline_map,
        )

        self.custom_params = custom_params_map
        self.pipeline = pipeline_map
        self.timeframe = resolved_timeframe
        self.custom_params.setdefault('timeframe', resolved_timeframe)
        self.pipeline['timeframe'] = resolved_timeframe

    @staticmethod
    def _normalise_mapping(source: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        if isinstance(source, dict):
            return source
        if isinstance(source, Mapping):
            return dict(source)
        return {}

    @classmethod
    def resolve_timeframe(
        cls,
        *,
        explicit: Optional[str] = None,
        custom_params: Optional[Mapping[str, Any]] = None,
        pipeline_overrides: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """
        Resolve timeframe using priority-based fallback system.

        Priority order (highest to lowest):
        1. Explicit parameter (passed directly to step)
        2. Custom parameters 'timeframe' key
        3. Pipeline overrides 'timeframe' key
        4. Global primary timeframe configuration
        5. Default fallback: '15m'

        Special case: Analyst runs are automatically elevated to '60m'
        for strategic decision-making compatibility.

        Args:
            explicit: Timeframe explicitly passed to the step
            custom_params: Custom parameters dictionary (may contain 'timeframe')
            pipeline_overrides: Pipeline-level overrides (may contain 'timeframe')

        Returns:
            Resolved timeframe string (e.g., '15m', '60m', '1h')
        """
        custom_map = dict(custom_params) if isinstance(custom_params, Mapping) else {}
        pipeline_map = dict(pipeline_overrides) if isinstance(pipeline_overrides, Mapping) else {}

        # Priority-based candidate selection
        candidates = (
            explicit,
            custom_map.get('timeframe'),
            pipeline_map.get('timeframe'),
            get_primary_timeframe(),
            '15m',  # Final fallback
        )

        timeframe = next((str(candidate) for candidate in candidates if candidate), '15m')

        # Analyst elevation: Strategic runs need higher granularity
        if cls._is_analyst_run(custom_map, pipeline_map):
            timeframe = '60m'

        return timeframe

    @staticmethod
    def _is_truthy_flag(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            normalized = value.strip().lower()
            return normalized in {'1', 'true', 'yes', 'y', 'on', 'enabled'}
        return False

    @classmethod
    def _is_analyst_run(
        cls,
        *sources: Mapping[str, Any],
    ) -> bool:
        role_keys = (
            'role',
            'pipeline_role',
            'execution_role',
            'run_role',
        )
        analyst_flags = (
            'analyst_mode',
            'is_analyst_run',
        )

        for source in sources:
            if not isinstance(source, Mapping):
                continue
            for key in role_keys:
                value = source.get(key)
                if isinstance(value, str) and value.strip().lower() == 'analyst':
                    return True
            for key in analyst_flags:
                value = source.get(key)
                if cls._is_truthy_flag(value):
                    return True
        return False

    def attach_locator(self, locator: DataLocator) -> None:
        """Attach a :class:`DataLocator` instance to the configuration."""

        self.data_locator = locator
        self._path_view = LocatorPaths(locator)

    def _ensure_paths(self) -> LocatorPaths:
        if self.data_locator is None:
            self.attach_locator(DataLocator(self.data_locator_config))
        elif self._path_view is None or self._path_view.locator is not self.data_locator:
            self._path_view = LocatorPaths(self.data_locator)
        return self._path_view

    @property
    def paths(self) -> LocatorPaths:
        return self._ensure_paths()

    @property
    def data(self) -> Any:
        return self.paths.data

    @property
    def cache(self) -> Any:
        return self.paths.cache

    @property
    def artifacts(self) -> Any:
        return self.paths.artifacts

    @property
    def generated(self) -> Any:
        return self.paths.generated

    @property
    def config_paths(self) -> Any:
        return self.paths.config

    @property
    def config_files(self) -> Any:
        """Alias for backwards compatibility with callers expecting ``config``."""

        return self.paths.config

    @property
    def config_root(self) -> Path:
        return self.paths.config.root

    @property
    def config(self) -> Any:
        """Expose configuration files via ``config`` attribute for convenience."""

        return self.paths.config

    @property
    def enabled_steps(self) -> List[str]:
        """Return list of enabled step names for pipeline execution."""
        # Return the default sequence of steps from the registry
        return [
            spec.name for spec in STEP_REGISTRY.values()
            if spec.enabled and spec.include_in_default_sequence
        ]

@dataclass
class SubPipelineFailure:
    """Structured failure details for sub-pipeline execution."""

    error_code: str
    message: str
    step: str
    exception: Optional[str] = None
    traceback: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    raw_exception: Optional[BaseException] = field(default=None, repr=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable representation of the failure."""
        payload = {
            'error_code': self.error_code,
            'message': self.message,
            'step': self.step,
            'context': self.context,
        }
        if self.exception:
            payload['exception'] = self.exception
        if self.traceback:
            payload['traceback'] = self.traceback
        return payload

@dataclass
class SubPipelineResult:
    """Result of sub-pipeline execution."""
    sub_pipeline_name: str
    status: SubPipelineStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    success: bool = False
    output_files: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    failure: Optional[SubPipelineFailure] = None

class PipelineResultDict(TypedDict, total=False):
    """Type definition for pipeline execution results."""

    success: bool
    execution_time: float
    total_steps: int
    completed_steps: int
    results: Dict[str, Any]
    warnings: List[str]
    errors: List[str]
    error_message: Optional[str]
    error_code: Optional[str]
    failure: Optional[SubPipelineFailure]
    error_summary: Optional[str]

class PreTrainingSubPipeline:
    """
    Pre-Training Sub-Pipeline for Feature Engineering Steps.

    Executes the 4 feature engineering steps in sequence:
    1. multi_horizon_profit_labeler
    2. feature_lookback_optimization
    3. interactive_feature_generation
    4. final_feature_selection
    """

    STEP_ERROR_CODES: Dict[str, str] = {
        'multi_horizon_profit_labeler': 'PRETRAIN_MH_LABEL_FAILURE',
        'feature_lookback_optimization': 'PRETRAIN_LOOKBACK_OPT_FAILURE',
        'interactive_feature_generation': 'PRETRAIN_INTERACTIVE_GEN_FAILURE',
        'optimized_lookback_generation': 'PRETRAIN_OPT_LOOKBACK_FAILURE',
        'final_feature_selection': 'PRETRAIN_FINAL_SELECTION_FAILURE',
        'pipeline': 'PRETRAIN_PIPELINE_FAILURE',
    }

    def __init__(self):
        """Initialize the pre-training sub-pipeline."""
        self.logger = logger.getChild('PreTrainingSubPipeline')
        tprint("🚀 Initialized PreTrainingSubPipeline")
        self.event_logger = PreTrainingEventLogger(configure_pre_training_logging())
        self.results: List[SubPipelineResult] = []
        self._current_pipeline_state: Dict[str, Any] = {}
        self._metrics_sink: Optional[MetricsSink] = None
        self._run_metadata: Dict[str, Any] = {}
        self._data_locator: Optional[DataLocator] = None
        self._seeded_rngs: Optional[SeededRNGs] = None
        # Artifact chain: stores artifacts from each step for the next step
        self._artifact_chain: Dict[str, Any] = {}
        self._active_seed: Optional[int] = None
        self._settings = get_pre_training_settings()
        self._missing_components: Set[str] = set()

        # Initialize memory-aware validation manager
        self.validation_manager = MemoryAwareValidationManager()

        # Register all available components
        from .components import ComponentRegistry
        ComponentRegistry.register_all_components()

        self._refresh_component_registry()

    def _store_artifacts_in_chain(self, step_name: str, artifacts: Any) -> None:
        """Store artifacts from a completed step for use by subsequent steps."""
        if artifacts is not None:
            self._artifact_chain[step_name] = artifacts
            self.logger.info(f"📦 Stored artifacts from {step_name} in chain")
            tprint(f"📦 Artifact chain updated: {step_name}")

    def _get_artifacts_from_chain(self, step_name: str) -> Any:
        """Retrieve artifacts from a previous step."""
        artifacts = self._artifact_chain.get(step_name)
        if artifacts:
            self.logger.info(f"📥 Retrieved artifacts from {step_name}")
        return artifacts

    def _get_all_previous_artifacts(self) -> Dict[str, Any]:
        """Get all artifacts from previous steps."""
        return dict(self._artifact_chain)

    @staticmethod
    def _get_step_spec(step_name: str) -> Optional[StepSpec]:
        """Return the registry specification for a step."""
        return STEP_REGISTRY.get(step_name)

    def _refresh_component_registry(self) -> None:
        """Synchronize component availability with the registered step list."""

        available_components = set(ComponentFactory.get_available_components())
        step_components = {spec.component_key for spec in STEP_REGISTRY.values()}

        missing_components = step_components - available_components
        extra_components = available_components - step_components

        if missing_components and missing_components != self._missing_components:
            message = (
                "Some pre-training steps are unavailable because their components "
                f"are not registered: {sorted(missing_components)}"
            )
            self.logger.warning(message)
            self.event_logger.warning(
                message,
                context={
                    'step': 'component_registry',
                    'missing_components': sorted(missing_components),
                    'available_components': sorted(available_components),
                },
            )

        if extra_components:
            self.logger.debug(
                "📋 Component factory exposes additional components not in the step registry: %s",
                sorted(extra_components),
            )

        self._missing_components = missing_components

    def _get_ordered_step_specs(
        self,
        *,
        include_disabled: bool = False,
        sequence_only: bool = False,
    ) -> List[StepSpec]:
        """Return registry specs ordered by execution priority."""

        self._refresh_component_registry()

        specs = [
            spec
            for spec in STEP_REGISTRY.values()
            if include_disabled or spec.enabled
        ]

        if sequence_only:
            specs = [spec for spec in specs if spec.include_in_default_sequence]

        if not include_disabled:
            specs = [
                spec
                for spec in specs
                if spec.component_key not in self._missing_components
            ]

        return sorted(specs, key=lambda spec: (spec.order, spec.name))

    # ------------------------------------------------------------------
    # Run metadata helpers
    # ------------------------------------------------------------------
    def _default_step_error_code(cls, step_name: str) -> str:
        base_code = cls.STEP_ERROR_CODES.get(step_name)
        if base_code:
            return base_code
        normalized = step_name.upper().replace(' ', '_')
        return f'PRETRAIN_{normalized}_FAILURE'

    @staticmethod
    def _extract_component_error_code(component_result: Any, default_code: str) -> str:
        component_errors = getattr(component_result, 'errors', None)
        if component_errors:
            for err in component_errors:
                if isinstance(err, ComponentError) and err.code:
                    return err.code
        for attr in ('error_code', 'error_code_slug'):
            value = getattr(component_result, attr, None)
            if value:
                return str(value)
        metadata = getattr(component_result, 'metadata', None)
        if isinstance(metadata, dict):
            for key in ('error_code', 'failure_code', 'error_code_slug'):
                value = metadata.get(key)
                if value:
                    return str(value)
        return default_code

    @staticmethod
    def _extend_messages(target: List[str], messages: Iterable[Any]) -> None:
        seen = set(target)
        for message in messages:
            if message is None:
                continue
            text = str(message).strip()
            if not text or text in seen:
                continue
            target.append(text)
            seen.add(text)

    def _collect_component_warnings(self, component_result: Any) -> List[str]:
        warnings: List[str] = []
        self._extend_messages(warnings, getattr(component_result, 'warnings', []) or [])
        return warnings

    def _collect_component_errors(self, component_result: Any) -> List[str]:
        errors: List[str] = []
        component_errors = getattr(component_result, 'errors', []) or []
        formatted_errors: List[str] = []
        for item in component_errors:
            if isinstance(item, ComponentError):
                text = f"[{item.code}] {item.message}" if item.code else item.message
                if item.details:
                    details_preview = ', '.join(f"{k}={v}" for k, v in list(item.details.items())[:3])
                    if details_preview:
                        text = f"{text} (details: {details_preview})"
                formatted_errors.append(text)
            else:
                formatted_errors.append(str(item))
        self._extend_messages(errors, formatted_errors)
        error_message = getattr(component_result, 'error_message', None)
        if error_message:
            self._extend_messages(errors, [error_message])
        return errors

    def _extend_pipeline_collections(self, pipeline_results: Dict[str, Any], result: SubPipelineResult) -> None:
        self._extend_messages(pipeline_results.setdefault('warnings', []), result.warnings)
        self._extend_messages(pipeline_results.setdefault('errors', []), result.errors)

    @staticmethod
    def _should_continue_on_error(config: SubPipelineConfig) -> bool:
        pipeline_cfg = getattr(config, 'pipeline', {}) or {}
        if isinstance(pipeline_cfg, dict) and pipeline_cfg.get('continue_on_error') is not None:
            return bool(pipeline_cfg.get('continue_on_error'))
        custom_params = getattr(config, 'custom_params', {}) or {}
        if isinstance(custom_params, dict):
            pipeline_params = custom_params.get('pipeline')
            if isinstance(pipeline_params, dict) and pipeline_params.get('continue_on_error') is not None:
                return bool(pipeline_params.get('continue_on_error'))
        return False

    async def _run_pipeline_step(
        self,
        *,
        spec: StepSpec,
        config: SubPipelineConfig,
        run_id: str,
        results: Dict[str, Any],
        metrics_sink: Optional[MetricsSink],
        step_metric_records: List[Dict[str, Any]],
        continue_on_error: bool,
        step_failures: List[Tuple[str, SubPipelineFailure, SubPipelineResult]],
        start_time: datetime,
        step_index: int,
        total_steps: int,
    ) -> Optional[PipelineResultDict]:
        icon = STEP_PROGRESS_ICONS.get(spec.name, '🚀')
        self.logger.info(
            f"{icon} Step {step_index}/{total_steps}: {spec.display_name}"
        )

        context = StepLogContext(
            run_id=run_id,
            step=spec.name,
            symbol=config.symbol,
            timeframe=config.timeframe,
        )
        self.event_logger.step_begin(context)

        executor = getattr(self, spec.executor_method, None)
        if executor is None:
            message = (
                f"Executor '{spec.executor_method}' missing for step "
                f"'{spec.name}'. Implement the executor or disable the step."
            )
            failure = self._create_failure(
                'pipeline',
                f"{self._default_step_error_code('pipeline')}_MISSING_EXECUTOR",
                message,
            )
            self.logger.error(f"❌ {message}")
            self.event_logger.step_end(
                context,
                duration_ms=0.0,
                success=False,
                error=message,
                extra={'error_code': failure.error_code},
            )
            return self._apply_failure_to_results(
                results,
                failure,
                start_time,
                metrics_sink,
                step_metric_records,
                config,
            )

        result = await executor(config, self._run_metadata)
        self._capture_step_timing_metrics(spec.name, result, config, results)
        rows_in, rows_out = self._resolve_row_counts(result)
        context.rows_in = rows_in
        context.rows_out = rows_out

        if result.success:
            results['completed_steps'] += 1

        self._record_step_metrics(spec.name, result, results, metrics_sink, step_metric_records)
        step_duration_ms = self._result_duration_ms(result)
        self._extend_pipeline_collections(results, result)

        if not result.success:
            failure = self._resolve_failure_from_result(
                spec.name,
                result,
                f"{spec.display_name} failed",
            )
            code_text = f"[{failure.error_code}] " if failure.error_code else ''
            self.logger.error(
                f"❌ {spec.display_name} failed: {code_text}{failure.message}"
            )
            self.event_logger.step_end(
                context,
                duration_ms=step_duration_ms,
                success=False,
                error=failure.message,
                extra={'error_code': failure.error_code},
            )
            results['results'][spec.name] = result.artifacts
            results['error_message'] = failure.message
            results['error_code'] = failure.error_code
            step_failures.append((spec.name, failure, result))
            if not continue_on_error:
                return self._apply_failure_to_results(
                    results,
                    failure,
                    start_time,
                    metrics_sink,
                    step_metric_records,
                    config,
                )

            warning_message = (
                f"⚠️ Continue-on-error enabled; proceeding after {spec.name} failure"
            )
            tprint_warning(warning_message)
            self.logger.warning(warning_message)
            return None

        artifacts = result.artifacts or {}
        if artifacts:
            try:
                merged_artifacts = self._current_pipeline_state.merge_step_artifacts(
                    spec.name,
                    artifacts,
                )
            except UnexpectedArtifactKeyError as merge_error:
                failure = self._create_failure(
                    spec.name,
                    f"{self._default_step_error_code(spec.name)}_SCHEMA",
                    str(merge_error),
                    context={'unexpected_keys': merge_error.keys},
                )
                self.logger.error(f"❌ {merge_error}")
                self.event_logger.step_end(
                    context,
                    duration_ms=step_duration_ms,
                    success=False,
                    error=str(merge_error),
                    extra={'error_code': failure.error_code},
                )
                return self._apply_failure_to_results(
                    results,
                    failure,
                    start_time,
                    metrics_sink,
                    step_metric_records,
                    config,
                )

            results['results'][spec.name] = merged_artifacts
        else:
            message = (
                f"{spec.display_name} completed without emitting artifacts; "
                f"this violates the documented pipeline contract."
            )
            failure = self._create_failure(
                spec.name,
                f"{self._default_step_error_code(spec.name)}_MISSING_ARTIFACTS",
                message,
            )
            self.logger.error(f"❌ {message}")
            self.event_logger.step_end(
                context,
                duration_ms=step_duration_ms,
                success=False,
                error=message,
                extra={'error_code': failure.error_code},
            )
            return self._apply_failure_to_results(
                results,
                failure,
                start_time,
                metrics_sink,
                step_metric_records,
                config,
            )

        self.event_logger.step_end(
            context,
            duration_ms=step_duration_ms,
            success=True,
            extra={'artifact_keys': sorted(result.artifacts.keys())},
        )

        return None
    def _build_event_context(
        self,
        step: str,
        *,
        config: Optional[SubPipelineConfig] = None,
        rows_in: Optional[int] = None,
        rows_out: Optional[int] = None,
        duration_ms: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Construct structured logging context with standard fields."""

        context: Dict[str, Any] = {
            'run_id': self._run_metadata.get('run_id', 'unknown'),
            'step': step,
            'symbol': (config.symbol if config else self._run_metadata.get('symbol')) or 'unknown',
            'timeframe': (config.timeframe if config else self._run_metadata.get('timeframe')) or 'unknown',
            'rows_in': rows_in,
            'rows_out': rows_out,
            'duration_ms': duration_ms,
        }
        if extra:
            context.update(extra)
        return context

    def _create_failure(
        self,
        step_name: str,
        error_code: str,
        message: str,
        exception: Optional[BaseException] = None,
        traceback_str: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> SubPipelineFailure:
        if exception is not None and traceback_str is None:
            traceback_str = ''.join(
                traceback.format_exception(type(exception), exception, exception.__traceback__)
            )
        failure = SubPipelineFailure(
            error_code=error_code,
            message=message,
            step=step_name,
            exception=str(exception) if exception else None,
            traceback=traceback_str,
            context=context or {},
            raw_exception=exception,
        )
        return failure

    def _compose_error_summary(
        self,
        failure: SubPipelineFailure,
        errors: Iterable[str],
    ) -> str:
        summary_parts: List[str] = []
        step_label = self._get_step_display_name(failure.step)
        summary_parts.append(f"{step_label}: {failure.message}")

        unique_errors: List[str] = []
        self._extend_messages(unique_errors, errors)
        # Avoid repeating the primary failure message if it's already captured.
        unique_errors = [msg for msg in unique_errors if msg != failure.message]

        if unique_errors:
            preview = '; '.join(unique_errors[:3])
            if len(unique_errors) > 3:
                preview += f" (+{len(unique_errors) - 3} more)"
            summary_parts.append(f"Details: {preview}")

        return ' | '.join(summary_parts)

    def _handle_contract_error(
        self,
        result: SubPipelineResult,
        step_name: str,
        error: DataContractValidationError,
    ) -> SubPipelineResult:
        """Convert a data contract validation error into a structured failure."""

        result.status = SubPipelineStatus.FAILED
        result.success = False
        result.error_message = str(error)
        result.error_code = f"{self._default_step_error_code(step_name)}_CONTRACT"
        result.failure = self._create_failure(
            step_name,
            result.error_code,
            result.error_message,
            context={
                'contract_context': error.context,
                'contract_issues': error.errors,
            },
        )
        self._extend_messages(result.errors, [str(error)])
        contract_issues = getattr(error, 'errors', None)
        if contract_issues:
            issue_messages = [str(issue) for issue in contract_issues if issue]
            self._extend_messages(result.errors, issue_messages)
        return result

    def _resolve_failure_from_result(
        self,
        step_name: str,
        step_result: SubPipelineResult,
        default_message: str,
    ) -> SubPipelineFailure:
        error_code = step_result.error_code or self._default_step_error_code(step_name)
        message = step_result.error_message or default_message
        context = {
            'status': step_result.status.value,
            'metadata': step_result.metadata,
            'artifacts_keys': sorted((step_result.artifacts or {}).keys()),
        }
        if step_result.failure:
            merged_context = dict(step_result.failure.context)
            merged_context.update({k: v for k, v in context.items() if v is not None})
            return self._create_failure(
                step_name,
                step_result.failure.error_code or error_code,
                step_result.failure.message or message,
                exception=step_result.failure.raw_exception,
                traceback_str=step_result.failure.traceback,
                context=merged_context,
            )

        return self._create_failure(
            step_name,
            error_code,
            message,
            context=context,
        )

    def _apply_failure_to_results(
        self,
        pipeline_results: Dict[str, Any],
        failure: SubPipelineFailure,
        start_time: datetime,
        metrics_sink: Optional[MetricsSink],
        step_metric_records: List[Dict[str, Any]],
        config: Optional[SubPipelineConfig] = None,
    ) -> PipelineResultDict:
        failure_time = datetime.now()
        pipeline_results['success'] = False
        pipeline_results['failure'] = failure
        pipeline_results['error_code'] = failure.error_code
        pipeline_results['error_message'] = failure.message
        errors = pipeline_results.setdefault('errors', [])
        summary = self._compose_error_summary(failure, errors)
        pipeline_results['error_summary'] = summary
        rows_in: Optional[int] = None
        rows_out: Optional[int] = None
        duration_ms: Optional[float] = None
        if failure.context:
            rows_in = self._search_numeric_fields(
                failure.context,
                ('rows_in', 'input_rows', 'rows_before', 'n_rows_in', 'samples_in'),
                depth=1,
            )
            rows_out = self._search_numeric_fields(
                failure.context,
                ('rows_out', 'output_rows', 'rows_after', 'n_rows_out', 'samples_out'),
                depth=1,
            )
            if isinstance(failure.context.get('duration_ms'), (int, float)):
                duration_ms = float(failure.context['duration_ms'])
            else:
                duration_seconds = self._search_numeric_fields(
                    failure.context,
                    ('duration_seconds',),
                    depth=1,
                )
                if duration_seconds is not None:
                    duration_ms = float(duration_seconds) * 1000.0

        message = summary or failure.message
        event_extra: Dict[str, Any] = {
            'error_code': failure.error_code,
            'failure_context': failure.context,
        }
        if summary:
            self.logger.error(summary)
            event_extra['error_summary'] = summary
        else:
            self.logger.error(failure.message)
        if failure.traceback:
            event_extra['traceback'] = failure.traceback

        self.event_logger.error(
            message,
            context=self._build_event_context(
                failure.step,
                config=config,
                rows_in=rows_in,
                rows_out=rows_out,
                duration_ms=duration_ms,
                extra=event_extra,
            ),
        )

        self._log_step_timing_summary(pipeline_results)

        failure_metadata = dict(self._run_metadata)
        failure_metadata['end_time_utc'] = datetime.utcnow().isoformat() + 'Z'
        failure_metadata['duration_seconds'] = (failure_time - start_time).total_seconds()
        self._run_metadata = failure_metadata

        self.event_logger.pipeline_end(
            run_id=failure_metadata.get('run_id', 'unknown'),
            symbol=config.symbol if config else pipeline_results.get('symbol', 'unknown'),
            timeframe=config.timeframe if config else pipeline_results.get('timeframe', 'unknown'),
            mode=config.mode.value if config else pipeline_results.get('mode', 'unknown'),
            success=False,
            duration_ms=failure_metadata.get('duration_seconds', 0.0) * 1000.0,
            completed_steps=pipeline_results.get('completed_steps', 0),
            total_steps=pipeline_results.get('total_steps', 0),
            metadata=dict(self._run_metadata),
            error=summary or failure.message,
        )

        finalized = self._finalize_results(
            pipeline_results,
            start_time,
            metrics_sink,
            step_metric_records,
            failure_time,
        )

        if os.getenv('ARES_STRICT') == '1':
            if failure.raw_exception is not None:
                raise failure.raw_exception
            raise RuntimeError(failure.message)

        return finalized

    def _gather_run_metadata(self, config: SubPipelineConfig, seed: Optional[int] = None) -> Dict[str, Any]:
        """Collect reproducibility metadata for the current run."""

        def _safe_git_sha() -> str:
            try:
                return subprocess.check_output(
                    ['git', 'rev-parse', 'HEAD'],
                    stderr=subprocess.DEVNULL,
                ).decode().strip()
            except Exception:
                return 'unknown'

        def _config_hash() -> str:
            config_dict = {
                key: getattr(config, key)
                for key in config.__dataclass_fields__.keys()
            }

            def _serialize(value: Any) -> Any:
                if isinstance(value, Enum):
                    return value.value
                if isinstance(value, dict):
                    return {str(k): _serialize(v) for k, v in sorted(value.items())}
                if isinstance(value, list):
                    return [_serialize(v) for v in value]
                return value

            serialized = json.dumps({k: _serialize(v) for k, v in sorted(config_dict.items())}, sort_keys=True, default=str)
            return hashlib.sha256(serialized.encode('utf-8')).hexdigest()

        def _data_snapshot_id() -> str:
            custom_params = config.custom_params or {}
            for key in ('data_snapshot_id', 'snapshot_id', 'data_snapshot'):
                if key in custom_params and custom_params[key]:
                    return str(custom_params[key])
            return 'unknown'

        def _rng_seed() -> Any:
            return seed

        start_timestamp = datetime.utcnow().isoformat() + 'Z'

        return {
            'run_id': uuid.uuid4().hex,
            'git_sha': _safe_git_sha(),
            'config_hash': _config_hash(),
            'data_snapshot_id': _data_snapshot_id(),
            'rng_seed': _rng_seed(),
            'symbol': config.symbol,
            'timeframe': config.timeframe,
            'mode': config.mode.value,
            'host_name': socket.gethostname(),
            'start_time_utc': start_timestamp,
            'end_time_utc': None,
            'duration_seconds': None,
        }

    def _merge_run_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Attach run metadata to metadata dictionaries."""
        merged = dict(metadata or {})
        merged['run_metadata'] = dict(self._run_metadata)
        return merged

    def _emit_effective_configuration(self, config: SubPipelineConfig) -> None:
        """Log the resolved filesystem configuration for operator visibility."""

        locator = self._data_locator or self._resolve_data_locator(config)
        config.attach_locator(locator)
        summary = config.paths.summary()
        summary_json = json.dumps(summary, indent=2, sort_keys=True)

        settings_summary = self._settings.summary()
        settings_json = json.dumps(settings_summary, indent=2, sort_keys=True)

        self.logger.info('📁 Effective filesystem configuration:\n%s', summary_json)
        self.logger.info('⚙️ Effective pre-training settings:\n%s', settings_json)
        self.event_logger.info(
            "Effective filesystem configuration resolved",
            context={
                'run_id': self._run_metadata.get('run_id'),
                'step': 'pipeline.configuration',
                'symbol': config.symbol,
                'timeframe': config.timeframe,
                'configuration': summary,
            },
        )
        self.event_logger.info(
            "Pre-training settings resolved",
            context={
                'run_id': self._run_metadata.get('run_id'),
                'step': 'pipeline.settings',
                'symbol': config.symbol,
                'timeframe': config.timeframe,
                'configuration': settings_summary,
            },
        )

    def _resolve_data_locator(self, config: SubPipelineConfig) -> DataLocator:
        """Return a data locator instance for the current run."""

        if isinstance(config.data_locator, DataLocator):
            config.attach_locator(config.data_locator)
            return config.data_locator

        locator = DataLocator(config.data_locator_config)
        config.attach_locator(locator)
        return locator

    async def execute_pipeline(self, config: SubPipelineConfig) -> PipelineResultDict:
        """
        Execute the complete pre-training pipeline with comprehensive logging.

        Args:
            config: Configuration for pipeline execution

        Returns:
            PipelineResultDict containing execution results with typed fields
        """
        tprint("🚀 Starting pre-training pipeline execution")
        tprint_info(f"📋 Pipeline configuration: {len(config.enabled_steps)} enabled steps, symbol: {config.symbol}")

        seed = self._resolve_random_seed(config)
        config.random_seed = seed
        if config.custom_params is None:
            config.custom_params = {}
        config.custom_params.setdefault('random_seed', seed)
        self._seeded_rngs = set_global_seed(seed)
        self._active_seed = seed

        tprint_info(f"🎲 Random seed resolved: {seed}")

        run_metadata = self._gather_run_metadata(config, seed)
        self._run_metadata = dict(run_metadata)
        self._current_pipeline_state['symbol'] = config.symbol
        self._current_pipeline_state['exchange'] = config.exchange
        self._current_pipeline_state['timeframe'] = config.timeframe
        self._current_pipeline_state['random_seed'] = seed
        if self._seeded_rngs is not None:
            self._current_pipeline_state['seeded_rngs'] = self._seeded_rngs
            self._current_pipeline_state['numpy_rng'] = self._seeded_rngs.numpy
            self._current_pipeline_state['python_rng'] = self._seeded_rngs.python

        # Log pipeline execution start
        tprint(f"🚀 Starting pre-training pipeline execution for {config.symbol}_{config.exchange}_{config.timeframe}")
        tprint_debug(f"📋 Pipeline metadata: {metadata_block}")

        metadata_block = json.dumps(self._run_metadata, indent=2, sort_keys=True)

        self.logger.info('🚀 Starting Pre-Training Sub-Pipeline execution')
        self.logger.info(f'📊 Symbol: {config.symbol}, Exchange: {config.exchange}')
        self.logger.info(f'⏰ Timeframe: {config.timeframe}, Mode: {config.mode.value}')
        self.logger.info(f'🎲 Random seed: {seed}')
        self.logger.info(f'🧾 Run metadata:\n{metadata_block}')

        run_id = self._run_metadata.get('run_id', 'unknown')
        self.event_logger.pipeline_begin(
            run_id=run_id,
            symbol=config.symbol,
            timeframe=config.timeframe,
            mode=config.mode.value,
            metadata=dict(self._run_metadata),
        )

        start_time = datetime.now()

        self._data_locator = self._resolve_data_locator(config)
        self._emit_effective_configuration(config)

        metrics_sink = self._create_metrics_sink(config)
        self._metrics_sink = metrics_sink
        step_metric_records: List[Dict[str, Any]] = []

        self._data_locator = self._resolve_data_locator(config)

        sequence_specs = self._get_ordered_step_specs(sequence_only=True)
        sequence_step_count = len(sequence_specs)
        continue_on_error = self._should_continue_on_error(config)
        step_failures: List[Tuple[str, SubPipelineFailure, SubPipelineResult]] = []

        # Check if we're running a specific component independently
        # If so, only run the steps that are needed for that component
        independent_component = self._get_independent_component_name(config)
        if independent_component:
            self.logger.info(f"🔄 Running independent component: {independent_component}")
#             # Filter sequence specs to only include the independent component
            sequence_specs = [spec for spec in sequence_specs if spec.name == independent_component]
            sequence_step_count = len(sequence_specs)
            results['total_steps'] = sequence_step_count
#             # Set flag to indicate we're running independently to prevent dependent steps
            config.custom_params['independent_component_execution'] = True
        else:
#             # When running the full pipeline, also check for use_existing_data flag
#             # If using existing data, skip steps that require downloading new data
            use_existing_data = config.custom_params.get('use_existing_data', False)
            if use_existing_data:
                self.logger.info("📁 Using existing data mode - skipping download-dependent steps")
#                 # Skip multi_horizon_profit_labeler since it may require downloads
#                 # But keep other steps that can work with existing data
#                 # This is intentional - no action needed

        results = {
            'success': False,
            'execution_time': 0.0,
            'total_steps': sequence_step_count,
            'completed_steps': 0,
            'results': {},
            'warnings': [],
            'errors': [],
            'error_message': None,
            'error_code': None,
            'failure': None,
            'error_summary': None,
            'metrics': {
                'steps': {},
            },
        }
        results['metrics']['random_seed'] = seed

        try:
#             # Step 1: Multi-Horizon Profit Labeler
            mh_context = StepLogContext(
                run_id=run_id,
                step='multi_horizon_profit_labeler',
                symbol=config.symbol,
                timeframe=config.timeframe,
            )
            self.event_logger.step_begin(mh_context)
            self.logger.info('🎯 Step 1: Multi-Horizon Profit Labeler')
# LEGACY METHOD CALL REMOVED
#             self._capture_step_timing_metrics('multi_horizon_profit_labeler', mh_result, config, results)
            mh_result = None  # Initialize to None since legacy method is removed
            rows_in, rows_out = self._resolve_row_counts(mh_result) if mh_result else (None, None)
            mh_context.rows_in = rows_in
            mh_context.rows_out = rows_out
#             if mh_result.success:
#                 results['completed_steps'] += 1
#             self._record_step_metrics('multi_horizon_profit_labeler', mh_result, results, metrics_sink, step_metric_records)
            mh_duration_ms = self._result_duration_ms(mh_result) if mh_result else None
#             self._extend_pipeline_collections(results, mh_result)
#             if not mh_result.success:
#                 failure = self._resolve_failure_from_result(
#                     'multi_horizon_profit_labeler',
#                     mh_result,
#                     'Multi-horizon profit labeling failed',
#                 )
#                 code_text = f"[{failure.error_code}] " if failure.error_code else ''
#                 self.logger.error(
#                     f"❌ Multi-horizon profit labeling failed: {code_text}{failure.message}"
#                 )
#                 self.event_logger.step_end(
#                     mh_context,
#                     duration_ms=mh_duration_ms,
#                     success=False,
#                     error=failure.message,
#                     extra={'error_code': failure.error_code},
#                 )
#                 results['results']['multi_horizon_profit_labeler'] = mh_result.artifacts
#                 results['error_message'] = failure.message
#                 results['error_code'] = failure.error_code
#                 step_failures.append(('multi_horizon_profit_labeler', failure, mh_result))
#                 if not continue_on_error:
#                     return self._apply_failure_to_results(
#                         results,
#                         failure,
#                         start_time,
#                         metrics_sink,
#                         step_metric_records,
#                         config,
#                     )
#                 else:
#                     # Only log continue-on-error warning if actually continuing
#                     self.event_logger.warning(
#                         "Continue-on-error enabled; proceeding after multi_horizon_profit_labeler failure",
#                         context=self._build_event_context(
#                             'multi_horizon_profit_labeler',
#                             config=config,
#                             rows_in=mh_context.rows_in,
#                             rows_out=mh_context.rows_out,
#                             duration_ms=mh_duration_ms,
#                             extra={
#                                 'error_code': failure.error_code,
#                                 'continue_on_error': True,
#                             },
#                         ),
#                     )
#                     self.logger.warning(
#                         "Continue-on-error enabled; proceeding after multi_horizon_profit_labeler failure",
#                     )
#             if mh_result.success:
#                 # Validate artifacts before updating state
#                 if 'multi_horizon_labeling_result' in mh_result.artifacts:
#                     labeled_data = mh_result.artifacts.get('multi_horizon_labeling_result', {}).get('labeled_data', pd.DataFrame())
#                     if isinstance(labeled_data, pd.DataFrame) and not labeled_data.empty:
#                         self.logger.info(f"✅ Multi-horizon profit labeling completed for {config.symbol}")
#                         self.logger.info(f"   → Labels generated: {len(labeled_data.columns)} columns")
#                         results['results']['multi_horizon_profit_labeler'] = mh_result.artifacts
#                         self._current_pipeline_state.update(mh_result.artifacts)
#                     else:
#                         message = "Multi-horizon labeling artifact validation failed"
#                         failure = self._create_failure(
#                             'multi_horizon_profit_labeler',
#                             mh_result.error_code or self._default_step_error_code('multi_horizon_profit_labeler'),
#                             message,
#                             context={'reason': 'empty_or_invalid_labeled_data'},
#                         )
#                         self.logger.error(f"❌ {message}")

#             # Validate artifacts before updating state
            artifacts = mh_result.artifacts or {} if mh_result else {}

            if 'multi_horizon_labeling_result' in artifacts:
#                 labeled_data = artifacts.get('multi_horizon_labeling_result', {}).get('labeled_data', pd.DataFrame())
#                 if isinstance(labeled_data, pd.DataFrame) and not labeled_data.empty:
#                     self.logger.info(f"✅ Multi-horizon profit labeling completed for {config.symbol}")
#                     self.logger.info(f"   → Labels generated: {len(labeled_data.columns)} columns")
#                     try:
#                         merged_artifacts = self._current_pipeline_state.merge_step_artifacts(
#                             'multi_horizon_profit_labeler',
#                             artifacts,
#                         )
#                     except UnexpectedArtifactKeyError as merge_error:
#                         failure = self._create_failure(
#                             'multi_horizon_profit_labeler',
#                             f"{self._default_step_error_code('multi_horizon_profit_labeler')}_SCHEMA",
#                             str(merge_error),
#                             context={'unexpected_keys': merge_error.keys},
#                         )
#                         self.logger.error(f"❌ {merge_error}")
                        self.event_logger.step_end(
                            mh_context,
                            duration_ms=mh_duration_ms,
                            success=False,
                            error=str(merge_error),
                            extra={'error_code': failure.error_code},
                        )
                        return self._apply_failure_to_results(
                            results,
                            failure,
                            start_time,
                            metrics_sink,
                            step_metric_records,
                            config,
                        )

#                     results['results']['multi_horizon_profit_labeler'] = merged_artifacts
#                 else:
#                     message = "Missing multi_horizon_labeling_result artifact"
#                     failure = self._create_failure(
#                         'multi_horizon_profit_labeler',
#                         mh_result.error_code or self._default_step_error_code('multi_horizon_profit_labeler'),
#                         message,
#                         context={'reason': 'missing_artifact'},
#                     )
#                     self.logger.error(f"❌ {message}")
#                     self.event_logger.step_end(
#                         mh_context,
#                         duration_ms=mh_duration_ms,
#                         success=False,
#                         error=message,
#                         extra={'error_code': failure.error_code},
#                     )
#                     return self._apply_failure_to_results(
#                         results,
#                         failure,
#                         start_time,
#                         metrics_sink,
#                         step_metric_records,
#                         config,
#                     )
        except ImportError as e:
            message = f"Missing dependencies: {str(e)}"
            failure = self._create_failure(
#                 'pipeline',
#                 f"{self._default_step_error_code('pipeline')}_IMPORT",
#                 message,
#                 exception=e,
            )
            self.logger.error(f"❌ Pre-Training Sub-Pipeline failed: [{failure.error_code}] {message}")
            self.event_logger.error(message, context={'error_code': failure.error_code, 'run_id': run_id})
            return self._apply_failure_to_results(results, failure, start_time, metrics_sink, step_metric_records, config)
        except FileNotFoundError as e:
            message = f"Missing files: {str(e)}"
            failure = self._create_failure(
                'pipeline',
                f"{self._default_step_error_code('pipeline')}_MISSING_FILE",
                message,
                exception=e,
            )
            self.logger.error(f"❌ Pre-Training Sub-Pipeline failed: [{failure.error_code}] {message}")
            self.event_logger.error(message, context={'error_code': failure.error_code, 'run_id': run_id})
            return self._apply_failure_to_results(results, failure, start_time, metrics_sink, step_metric_records, config)
        except MemoryError as e:
            message = f"Memory error: {str(e)}"
            failure = self._create_failure(
                'pipeline',
                f"{self._default_step_error_code('pipeline')}_MEMORY",
                message,
                exception=e,
            )
            self.logger.error(f"❌ Pre-Training Sub-Pipeline failed: [{failure.error_code}] {message}")
            self.event_logger.error(message, context={'error_code': failure.error_code, 'run_id': run_id})
            return self._apply_failure_to_results(results, failure, start_time, metrics_sink, step_metric_records, config)
        except Exception as e:
            message = f"Unexpected error: {str(e)}"
            failure = self._create_failure(
                'pipeline',
                f"{self._default_step_error_code('pipeline')}_ERROR",
                message,
                exception=e,
            )
            self.logger.error(f"❌ Pre-Training Sub-Pipeline failed: [{failure.error_code}] {message}")
            self.event_logger.error(message, context={'error_code': failure.error_code, 'run_id': run_id})
            return self._apply_failure_to_results(results, failure, start_time, metrics_sink, step_metric_records, config)

        try:
            if not sequence_specs:
                message = (
                    'No pre-training steps are enabled in STEP_REGISTRY. '
                    'Enable at least one step or adjust the configuration.'
                )
#             # Step 2: Feature Lookback Optimization
            flo_context = StepLogContext(
                run_id=run_id,
                step='feature_lookback_optimization',
                symbol=config.symbol,
                timeframe=config.timeframe,
            )
            self.event_logger.step_begin(flo_context)
            self.logger.info('⚙️ Step 2: Feature Lookback Optimization')
# LEGACY METHOD CALL REMOVED
#             self._capture_step_timing_metrics('feature_lookback_optimization', flo_result, config, results)
            flo_result = None  # Initialize to None since legacy method is removed
            rows_in, rows_out = self._resolve_row_counts(flo_result) if flo_result else (None, None)
            flo_context.rows_in = rows_in
            flo_context.rows_out = rows_out
#             if flo_result.success:
#                 results['completed_steps'] += 1
#             self._record_step_metrics('feature_lookback_optimization', flo_result, results, metrics_sink, step_metric_records)
            flo_duration_ms = self._result_duration_ms(flo_result) if flo_result else None
#             self._extend_pipeline_collections(results, flo_result)
#             if not flo_result.success:
#                 failure = self._resolve_failure_from_result(
#                     'feature_lookback_optimization',
#                     flo_result,
#                     'Feature lookback optimization failed',
#                 )
#                 code_text = f"[{failure.error_code}] " if failure.error_code else ''
#                 self.logger.error(
#                     f"❌ Feature lookback optimization failed: {code_text}{failure.message}"
#                 )
#                 self.event_logger.step_end(
#                     flo_context,
#                     duration_ms=flo_duration_ms,
#                     success=False,
#                     error=failure.message,
#                     extra={'error_code': failure.error_code},
#                 )
#                 results['results']['feature_lookback_optimization'] = flo_result.artifacts
#                 results['error_message'] = failure.message
#                 results['error_code'] = failure.error_code
#                 step_failures.append(('feature_lookback_optimization', failure, flo_result))
#                 if not continue_on_error:
#                     return self._apply_failure_to_results(
#                         results,
#                         failure,
#                         start_time,
#                         metrics_sink,
#                         step_metric_records,
#                         config,
#                     )

            flo_artifacts = flo_result.artifacts or {} if flo_result else {}

#             # Validate artifacts before updating state
            if 'feature_lookback_optimization_result' in flo_artifacts:
                optimized_features = flo_artifacts.get('feature_lookback_optimization_result', {}).get('optimized_features', {})
#                 self.logger.info(f"✅ Feature lookback optimization completed for {config.symbol}")
#                 self.logger.info(f"   → Features optimized: {len(optimized_features)}")
#                 try:
#                     merged_flo_artifacts = self._current_pipeline_state.merge_step_artifacts(
#                         'feature_lookback_optimization',
#                         flo_artifacts,
#                     )
#                 except UnexpectedArtifactKeyError as merge_error:
#                     failure = self._create_failure(
#                         'feature_lookback_optimization',
#                         f"{self._default_step_error_code('feature_lookback_optimization')}_SCHEMA",
#                         str(merge_error),
#                         context={'unexpected_keys': merge_error.keys},
#                     )
#                     self.logger.error(f"❌ {merge_error}")
#                     self.event_logger.step_end(
#                         flo_context,
#                         duration_ms=flo_duration_ms,
#                         success=False,
#                         error=str(merge_error),
#                         extra={'error_code': failure.error_code},
#                     )
#                     return self._apply_failure_to_results(
#                         results,
#                         failure,
#                         start_time,
#                         metrics_sink,
#                         step_metric_records,
#                         config,
#                     )
#                 else:
#                     # Only log continue-on-error warning if actually continuing
#                     self.event_logger.warning(
#                         "Continue-on-error enabled; proceeding after feature_lookback_optimization failure",
#                         context=self._build_event_context(
#                             'feature_lookback_optimization',
#                             config=config,
#                             rows_in=flo_context.rows_in,
#                             rows_out=flo_context.rows_out,
#                             duration_ms=flo_duration_ms,
#                             extra={
#                                 'error_code': failure.error_code,
#                                 'continue_on_error': True,
#                             },
#                         ),
#                     )
#                     self.logger.warning(
#                         "Continue-on-error enabled; proceeding after feature_lookback_optimization failure",
#                     )
#             if flo_result.success:
#                 # Validate artifacts before updating state
#                 if 'feature_lookback_optimization_result' in flo_result.artifacts:
#                     optimized_features = flo_result.artifacts.get('feature_lookback_optimization_result', {}).get('optimized_features', {})
                self.logger.info(f"✅ Feature lookback optimization completed for {config.symbol}")
                self.logger.info(f"   → Features optimized: {len(optimized_features)}")
#                     results['results']['feature_lookback_optimization'] = flo_result.artifacts
#                     self._current_pipeline_state.update(flo_result.artifacts)
#                 else:
#                     self.logger.warning("⚠️ Feature lookback optimization completed but artifact structure unexpected")
#                     # results['results']['feature_lookback_optimization'] = flo_result.artifacts
#                     # self._current_pipeline_state.update(flo_result.artifacts)

#                 results['results']['feature_lookback_optimization'] = merged_flo_artifacts
#             elif flo_artifacts:
#                 # self.logger.warning("⚠️ Feature lookback optimization completed but artifact structure unexpected")
#                 try:
#                     merged_flo_artifacts = self._current_pipeline_state.merge_step_artifacts(
#                         'feature_lookback_optimization',
#                         flo_artifacts,
#                     )
#                 except UnexpectedArtifactKeyError as merge_error:
#                     failure = self._create_failure(
#                         'feature_lookback_optimization',
#                         f"{self._default_step_error_code('feature_lookback_optimization')}_SCHEMA",
#                         str(merge_error),
#                         context={'unexpected_keys': merge_error.keys},
#                     )
#                     self.logger.error(f"❌ {merge_error}")
#                     self.event_logger.step_end(
#                         flo_context,
#                         duration_ms=flo_duration_ms,
#                         success=False,
#                         error=str(merge_error),
#                         extra={'error_code': failure.error_code},
#                     )
#                     return self._apply_failure_to_results(
#                         results,
#                         failure,
#                         start_time,
#                         metrics_sink,
#                         step_metric_records,
#                         config,
#                     )

#                 results['results']['feature_lookback_optimization'] = merged_flo_artifacts

#                 self.event_logger.step_end(
#                     flo_context,
#                     duration_ms=flo_duration_ms,
#                     success=True,
#                     extra={'artifact_keys': sorted(flo_result.artifacts.keys())},
#                 )

#             # Step 3: Interactive Feature Generation
#             # Skip interactive feature generation when running components independently
            running_independently = config.custom_params.get('independent_component_execution', False)
            if running_independently:
                self.logger.info('🔄 Skipping interactive feature generation when running independently')
#                 # Create a mock successful result to maintain pipeline flow
#                 interactive_result = SubPipelineResult(
#                     sub_pipeline_name='interactive_feature_generation',
#                     status=SubPipelineStatus.COMPLETED,
#                     start_time=datetime.now(),
#                     end_time=datetime.now(),
#                     duration_seconds=0.0,
#                     success=True,
#                     artifacts={'interactive_feature_generation_result': {}},
#                     output_files=[],
#                     error_message=None,
#                     error_code=None,
#                     metadata={'skipped': 'independent_execution'}
#                 )
                # Create context for logging
                interactive_context = StepLogContext(
                    run_id=run_id,
                    step='interactive_feature_generation',
                    symbol=config.symbol,
                    timeframe=config.timeframe,
                )
#                 rows_in, rows_out = 0, 0
#                 interactive_duration_ms = 0.0
#             else:
#                 # interactive_context = StepLogContext(
#                     run_id=run_id,
#                     step='interactive_feature_generation',
#                     symbol=config.symbol,
#                     timeframe=config.timeframe,
#                 )
#                 self.event_logger.step_begin(interactive_context)
#                 self.logger.info('🔧 Step 3: Interactive Feature Generation')
# LEGACY METHOD CALL REMOVED
                interactive_result = None  # Initialize to None since legacy method is removed
                rows_in, rows_out = self._resolve_row_counts(interactive_result) if interactive_result else (None, None)
                interactive_duration_ms = self._result_duration_ms(interactive_result) if interactive_result else None

#             self._capture_step_timing_metrics('interactive_feature_generation', interactive_result, config, results)
            interactive_context.rows_in = rows_in
            interactive_context.rows_out = rows_out
#             if interactive_result.success:
#                 results['completed_steps'] += 1
#             self._record_step_metrics('interactive_feature_generation', interactive_result, results, metrics_sink, step_metric_records)
#             self._extend_pipeline_collections(results, interactive_result)
#             if not interactive_result.success and not running_independently:
#                 failure = self._resolve_failure_from_result(
#                     'interactive_feature_generation',
#                     interactive_result,
#                     'Interactive feature generation failed',
#                 )
#                 failure = self._create_failure(
#                     'pipeline',
#                     f"{self._default_step_error_code('pipeline')}_NO_STEPS",
#                     message,
#                 )
#                 self.logger.error(f'❌ {message}')
#                 self.event_logger.error(
#                     message,
#                     context={'error_code': failure.error_code, 'run_id': run_id},
#                 )
#                 return self._apply_failure_to_results(
#                     results,
#                     failure,
#                     start_time,
#                     metrics_sink,
#                     step_metric_records,
#                     config,
#                 )

            for index, spec in enumerate(sequence_specs, start=1):
                step_result = await self._run_pipeline_step(
                    spec=spec,
                    config=config,
                    run_id=run_id,
                    results=results,
                    metrics_sink=metrics_sink,
                    step_metric_records=step_metric_records,
                    continue_on_error=continue_on_error,
                    step_failures=step_failures,
                    start_time=start_time,
                    step_index=index,
                    total_steps=sequence_step_count,
                )
            interactive_artifacts = interactive_result.artifacts or {} if interactive_result else {}

#             # Validate artifacts before updating state
            if 'interactive_feature_generation_result' in interactive_artifacts:
                features = interactive_artifacts.get('interactive_feature_generation_result', {}).get('features', {})
                self.logger.info(f"✅ Interactive feature generation completed for {config.symbol}")
                self.logger.info(f"   → Features generated: {len(features)}")
                try:
                    merged_interactive_artifacts = self._current_pipeline_state.merge_step_artifacts(
                        'interactive_feature_generation',
                        interactive_artifacts,
                    )
                except UnexpectedArtifactKeyError as merge_error:
                    failure = self._create_failure(
                        'interactive_feature_generation',
                        f"{self._default_step_error_code('interactive_feature_generation')}_SCHEMA",
                        str(merge_error),
                        context={'unexpected_keys': merge_error.keys},
                    )
                    self.logger.error(f"❌ {merge_error}")
                    self.event_logger.step_end(
                        interactive_context,
                        duration_ms=interactive_duration_ms,
                        success=False,
                        error=str(merge_error),
                        extra={'error_code': failure.error_code},
                    )
                    return self._apply_failure_to_results(
                        results,
                        failure,
                        start_time,
                        metrics_sink,
                        step_metric_records,
                        config,
                    )
                self.event_logger.warning(
                    "Continue-on-error enabled; proceeding after interactive_feature_generation failure",
                    context=self._build_event_context(
                        'interactive_feature_generation',
                        config=config,
                        rows_in=interactive_context.rows_in,
                        rows_out=interactive_context.rows_out,
                        duration_ms=interactive_duration_ms,
                        extra={
                            'error_code': failure.error_code,
                            'continue_on_error': True,
                        },
                    ),
                )
                self.logger.warning(
                    "Continue-on-error enabled; proceeding after interactive_feature_generation failure",
                )
#             if interactive_result.success:
#                 # Validate artifacts before updating state
#                 if 'interactive_feature_generation_result' in interactive_result.artifacts:
#                     features = interactive_result.artifacts.get('interactive_feature_generation_result', {}).get('features', {})
#                     self.logger.info(f"✅ Interactive feature generation completed for {config.symbol}")
#                     self.logger.info(f"   → Features generated: {len(features)}")
#                     results['results']['interactive_feature_generation'] = interactive_result.artifacts
#                     self._current_pipeline_state.update(interactive_result.artifacts)
#                 else:
#                     self.logger.warning("⚠️ Interactive feature generation completed but artifact structure unexpected")
#                     results['results']['interactive_feature_generation'] = interactive_result.artifacts
#                     self._current_pipeline_state.update(interactive_result.artifacts)

#                 results['results']['interactive_feature_generation'] = merged_interactive_artifacts
            elif interactive_artifacts:
                pass  # Legacy method removed - no action needed
#                 self.logger.warning("⚠️ Interactive feature generation completed but artifact structure unexpected")
#                 try:
#                     merged_interactive_artifacts = self._current_pipeline_state.merge_step_artifacts(
#                         'interactive_feature_generation',
#                         interactive_artifacts,
#                     )
#                 except UnexpectedArtifactKeyError as merge_error:
#                     failure = self._create_failure(
#                         'interactive_feature_generation',
#                         f"{self._default_step_error_code('interactive_feature_generation')}_SCHEMA",
#                         str(merge_error),
#                         context={'unexpected_keys': merge_error.keys},
#                     )
#                     self.logger.error(f"❌ {merge_error}")
#                     self.event_logger.step_end(
#                         interactive_context,
#                         duration_ms=interactive_duration_ms,
#                         success=False,
#                         error=str(merge_error),
#                         extra={'error_code': failure.error_code},
#                     )
#                     return self._apply_failure_to_results(
#                         results,
#                         failure,
#                         start_time,
#                         metrics_sink,
#                         step_metric_records,
#                         config,
#                     )

#                 results['results']['interactive_feature_generation'] = merged_interactive_artifacts

#                 self.event_logger.step_end(
#                     interactive_context,
#                     duration_ms=interactive_duration_ms,
#                     success=True,
#                     extra={'artifact_keys': sorted(interactive_result.artifacts.keys())},
#                 )

#             # Step 4: Final Feature Selection
#             ffs_context = StepLogContext(
#                 run_id=run_id,
#                 step='final_feature_selection',
#                 symbol=config.symbol,
#                 timeframe=config.timeframe,
#             )
#             self.event_logger.step_begin(ffs_context)
#             self.logger.info('🎯 Step 4: Final Feature Selection')
# LEGACY METHOD CALL REMOVED
            ffs_duration_ms = None  # Initialize to None since legacy method is removed
#             ffs_result = None  # Initialize to None since legacy method is removed
#             self._capture_step_timing_metrics('final_feature_selection', ffs_result, config, results)
#             rows_in, rows_out = self._resolve_row_counts(ffs_result)
#             ffs_context.rows_in = rows_in
#             ffs_context.rows_out = rows_out
#             if ffs_result.success:
#                 results['completed_steps'] += 1
#             self._record_step_metrics('final_feature_selection', ffs_result, results, metrics_sink, step_metric_records)
#             ffs_duration_ms = None  # Initialize to None since legacy method is removed
#             self._extend_pipeline_collections(results, ffs_result)
#             if not ffs_result.success:
#                 failure = self._resolve_failure_from_result(
#                     'final_feature_selection',
#                     ffs_result,
#                     'Final feature selection failed',
#                 )
#                 code_text = f"[{failure.error_code}] " if failure.error_code else ''
#                 self.logger.error(
#                     f"❌ Final feature selection failed: {code_text}{failure.message}"
#                 )
#                 self.event_logger.step_end(
#                     ffs_context,
#                     duration_ms=ffs_duration_ms,
#                     success=False,
#                     error=failure.message,
#                     extra={'error_code': failure.error_code},
#                 )
#                 results['results']['final_feature_selection'] = ffs_result.artifacts
#                 results['error_message'] = failure.message
#                 results['error_code'] = failure.error_code
#                 step_failures.append(('final_feature_selection', failure, ffs_result))
#                 if not continue_on_error:
#                     return self._apply_failure_to_results(
#                         results,
#                         failure,
#                         start_time,
#                         metrics_sink,
#                         step_metric_records,
#                         config,
#                     )
#                 else:
#                     # Only log continue-on-error warning if actually continuing
#                     self.event_logger.warning(
#                         "Continue-on-error enabled; proceeding after final_feature_selection failure",
#                         context=self._build_event_context(
#                             'final_feature_selection',
#                             config=config,
#                             rows_in=ffs_context.rows_in,
#                             rows_out=ffs_context.rows_out,
#                             duration_ms=ffs_duration_ms,
#                             extra={
#                                 'error_code': failure.error_code,
#                                 'continue_on_error': True,
#                             },
#                         ),
#                     )
#                     self.logger.warning(
#                         "Continue-on-error enabled; proceeding after final_feature_selection failure",
#                     )
#             if ffs_result.success:
#                 # Validate artifacts before updating state
#                 if 'final_feature_selection_result' in ffs_result.artifacts:
#                     selected_features = ffs_result.artifacts.get('final_feature_selection_result', {}).get('selected_features', [])
#                     self.logger.info(f"✅ Final feature selection completed for {config.symbol}")
#                     self.logger.info(f"   → Final features: {len(selected_features)}")
#                     results['results']['final_feature_selection'] = ffs_result.artifacts
#                     self._current_pipeline_state.update(ffs_result.artifacts)
#                 else:
#                     self.logger.warning("⚠️ Final feature selection completed but artifact structure unexpected")
#                     results['results']['final_feature_selection'] = ffs_result.artifacts
#                     self._current_pipeline_state.update(ffs_result.artifacts)

#                 self.event_logger.step_end(
#                     ffs_context,
#                     duration_ms=ffs_duration_ms,
#                     success=True,
#                     extra={'artifact_keys': sorted(ffs_result.artifacts.keys())},
#                 )
#                 if step_result is not None:
#                     return step_result

#             if step_failures:
#                 # primary_failure = step_failures[0][1]
#                 return self._apply_failure_to_results(
#                     results,
#                     primary_failure,
#                     start_time,
#                     metrics_sink,
#                     step_metric_records,
#                     config,
#                 )

            end_time = datetime.now()
#             results['success'] = True
#             results['execution_time'] = (end_time - start_time).total_seconds()
#             results['completed_steps'] = sequence_step_count

#             # End memory-aware validation session and collect statistics
#             memory_stats = self.validation_manager.end_validation_session(validation_session_id)
#             results['memory_stats'] = memory_stats
#             results['metrics']['memory_stats'] = memory_stats

#             end_metadata = dict(self._run_metadata)
#             end_metadata['end_time_utc'] = datetime.utcnow().isoformat() + 'Z'
#             end_metadata['duration_seconds'] = results['execution_time']
#             self._run_metadata = end_metadata

            completion_block = json.dumps(self._run_metadata, indent=2, sort_keys=True)
#             pipeline_duration_ms = results['execution_time'] * 1000.0
#             # Enhanced success logging with tprint
#             tprint_success(f"🎉 Pre-Training Sub-Pipeline execution completed successfully for {config.symbol}")
#             tprint(f"⏱️ Total execution time: {results['execution_time']:.2f} seconds")
#             tprint(f"📊 Steps completed: {pipeline_results.get('completed_steps', 0)}/{pipeline_results.get('total_steps', 0)}")
#             tprint(f"💾 Artifacts generated: {len(results.get('results', {}))}")

#             self.logger.info(
#                 f'🎉 Pre-Training Sub-Pipeline execution completed successfully for {config.symbol}'
#             )
#             # self.logger.info(
#                 # f"⏱️ Total execution time: {results['execution_time']:.2f} seconds"
#             )
            self.logger.info(f"🧾 Run metadata:\n{completion_block}")
#             self.event_logger.pipeline_end(
#                 run_id=run_id,
#                 symbol=config.symbol,
#                 timeframe=config.timeframe,
#                 mode=config.mode.value,
#                 success=True,
#                 duration_ms=pipeline_duration_ms,
#                 completed_steps=results['completed_steps'],
#                 total_steps=results['total_steps'],
#                 metadata=dict(self._run_metadata),
#             )
        except ImportError as e:
            message = f"Missing dependencies: {str(e)}"
            failure = self._create_failure(
                'pipeline',
                f"{self._default_step_error_code('pipeline')}_IMPORT",
                message,
                exception=e,
            )
            self.logger.error(f"❌ Pre-Training Sub-Pipeline failed: [{failure.error_code}] {message}")
            self.event_logger.error(message, context={'error_code': failure.error_code, 'run_id': run_id})
            return self._apply_failure_to_results(results, failure, start_time, metrics_sink, step_metric_records, config)
        except FileNotFoundError as e:
            message = f"Missing files: {str(e)}"
            failure = self._create_failure(
                'pipeline',
                f"{self._default_step_error_code('pipeline')}_MISSING_FILE",
                message,
                exception=e,
            )
            self.logger.error(f"❌ Pre-Training Sub-Pipeline failed: [{failure.error_code}] {message}")
            self.event_logger.error(message, context={'error_code': failure.error_code, 'run_id': run_id})
            return self._apply_failure_to_results(results, failure, start_time, metrics_sink, step_metric_records, config)
        except MemoryError as e:
            message = f"Memory error: {str(e)}"
            failure = self._create_failure(
                'pipeline',
                f"{self._default_step_error_code('pipeline')}_MEMORY",
                message,
                exception=e,
            )
            self.logger.error(f"❌ Pre-Training Sub-Pipeline failed: [{failure.error_code}] {message}")
            self.event_logger.error(message, context={'error_code': failure.error_code, 'run_id': run_id})
            return self._apply_failure_to_results(results, failure, start_time, metrics_sink, step_metric_records, config)
        except Exception as e:
            message = f"Unexpected error: {str(e)}"
            failure = self._create_failure(
                'pipeline',
                f"{self._default_step_error_code('pipeline')}_UNEXPECTED",
                message,
                exception=e,
            )
            self.logger.error(f"❌ Pre-Training Sub-Pipeline failed: [{failure.error_code}] {message}")
            self.event_logger.error(message, context={'error_code': failure.error_code, 'run_id': run_id, 'traceback': failure.traceback})
            return self._apply_failure_to_results(results, failure, start_time, metrics_sink, step_metric_records, config)

        return self._finalize_results(results, start_time, metrics_sink, step_metric_records, end_time if results.get('success') else None)

    # ------------------------------------------------------------------
    # Metrics helpers
    # ------------------------------------------------------------------
    def _create_metrics_sink(self, config: SubPipelineConfig) -> Optional[MetricsSink]:
        output_path: Optional[Path] = None
        if config.metrics_output_path:
            output_path = Path(config.metrics_output_path)
        else:
            extension = 'jsonl' if config.metrics_output_format.lower() == 'jsonl' else 'csv'
            metrics_dir_setting = self._settings.metrics.output_dir
            if metrics_dir_setting is not None:
                base_dir = metrics_dir_setting.resolved
#                 # Use common utility for directory creation with error handling
                if not ensure_directory(base_dir):
                    tprint_error(f"Failed to create base directory: {base_dir}")
                    raise ValueError(f"Cannot create base directory: {base_dir}")
                filename = self._settings.metrics.filename
                if Path(filename).suffix:
                    metrics_filename = filename
                else:
                    metrics_filename = f'{filename}.{extension}'
                output_path = base_dir / metrics_filename
            else:
                locator = self._data_locator or self._resolve_data_locator(config)
                base_dir = locator.artifacts_path(
                    config.artifacts_dir_key,
                    ensure_exists=True,
                )
                output_path = base_dir / f'pre_training_metrics.{extension}'

        if output_path is None and not config.metrics_prometheus_enabled:
            self.logger.warning("No metrics output path configured and Prometheus metrics disabled. Metrics will not be collected.")
            return None

        if output_path is None:
            locator = self._data_locator or self._resolve_data_locator(config)
            base_dir = locator.artifacts_path(
                config.artifacts_dir_key,
                ensure_exists=True,
            )
            output_path = base_dir / f'pre_training_metrics.{config.metrics_output_format.lower()}'

        sink_config = MetricsSinkConfig(
            output_path=output_path,
            output_format=config.metrics_output_format,
            enable_prometheus=config.metrics_prometheus_enabled,
        )
        return MetricsSink(sink_config)

    def _record_step_metrics(
        self,
        step_name: str,
        result: SubPipelineResult,
        pipeline_results: Dict[str, Any],
        metrics_sink: Optional[MetricsSink],
        step_metric_records: List[Dict[str, Any]],
    ) -> None:
        if metrics_sink is None:
            return

        record = self._base_metrics_record()
        row_counts = self._extract_row_counts(result.artifacts)
        row_count_total = sum(row_counts.values()) if row_counts else 0
        label_skew = self._compute_label_distribution_skew(result.metadata)
        memory_peak_mb = self._get_memory_usage_mb()

        record.update({
            'record_type': 'step',
            'step_name': step_name,
            'status': result.status.value,
            'success': result.success,
            'duration_seconds': result.duration_seconds,
            'row_count_total': row_count_total,
            'row_count_details': json.dumps(row_counts, sort_keys=True),
            'memory_peak_mb': memory_peak_mb,
            'label_distribution_skew': label_skew,
            'timestamp': datetime.utcnow().isoformat(),
            'artifact_count': len(result.artifacts),
            'metadata_keys': ','.join(sorted(result.metadata.keys())) if result.metadata else '',
            'total_steps': pipeline_results.get('total_steps'),
            'completed_steps': pipeline_results.get('completed_steps'),
            'total_row_count': row_count_total,
            'max_memory_peak_mb': memory_peak_mb,
            'average_label_distribution_skew': label_skew,
            'error_message': result.error_message or '',
        })

        step_metric_records.append(record)
        metrics_sink.write(record)

    def _capture_step_timing_metrics(
        self,
        step_name: str,
        result: SubPipelineResult,
        config: SubPipelineConfig,
        pipeline_results: Dict[str, Any],
    ) -> None:
        duration = result.duration_seconds or 0.0
        budget = self._get_step_budget(config, step_name)
        over_budget_seconds = 0.0
        over_budget = False

        if budget is not None and duration > budget:
            over_budget = True
            over_budget_seconds = duration - budget
            warning_message = (
                f"⚠️ Step '{step_name}' duration {duration:.2f}s exceeded budget {budget:.2f}s by {over_budget_seconds:.2f}s"
            )
            self.logger.warning(warning_message)

        result.metadata.setdefault('timing', {})
        result.metadata['timing'].update(
            {
                'duration_seconds': duration,
                'budget_seconds': budget,
                'over_budget_seconds': over_budget_seconds,
                'over_budget': over_budget,
            }
        )
        if over_budget:
            result.metadata.setdefault('timing_alerts', []).append(
                {
                    'message': warning_message,
                    'over_budget_seconds': over_budget_seconds,
                }
            )

        pipeline_results.setdefault('metrics', {}).setdefault('steps', {})[step_name] = {
            'duration_seconds': duration,
            'budget_seconds': budget,
            'over_budget_seconds': over_budget_seconds,
            'over_budget': over_budget,
        }

    def _result_duration_ms(self, result: SubPipelineResult) -> Optional[float]:
        """Return step duration in milliseconds if available."""
        if result.duration_seconds is not None:
            return result.duration_seconds * 1000.0
        if result.start_time and result.end_time:
            return (result.end_time - result.start_time).total_seconds() * 1000.0
        return None

    def _resolve_row_counts(self, result: SubPipelineResult) -> Tuple[Optional[int], Optional[int]]:
        """Infer input/output row counts from result metadata and artifacts."""

        rows_in = self._search_numeric_fields(
            result.metadata,
            ('rows_in', 'input_rows', 'rows_before', 'n_rows_in', 'samples_in'),
        )
        rows_out = self._search_numeric_fields(
            result.metadata,
            ('rows_out', 'output_rows', 'rows_after', 'n_rows_out', 'samples_out'),
        )

        if rows_out is None and result.artifacts:
            row_counts = self._extract_row_counts(result.artifacts)
            if row_counts:
                rows_out = max(row_counts.values())
                if rows_in is None:
                    rows_in = rows_out

        return rows_in, rows_out

    def _search_numeric_fields(self, data: Any, keys: Tuple[str, ...], depth: int = 3) -> Optional[int]:
        if depth < 0 or data is None:
            return None
        if isinstance(data, dict):
            for key in keys:
                value = data.get(key)
                if isinstance(value, (int, float)):
                    return int(value)
            for value in data.values():
                resolved = self._search_numeric_fields(value, keys, depth - 1)
                if resolved is not None:
                    return resolved
        elif isinstance(data, (list, tuple)):
            for item in data:
                resolved = self._search_numeric_fields(item, keys, depth - 1)
                if resolved is not None:
                    return resolved
        return None

    @staticmethod
    def _get_step_budget(config: SubPipelineConfig, step_name: str) -> Optional[float]:
        budgets = config.step_time_budgets or {}
        if step_name in budgets:
            return budgets[step_name]
        return DEFAULT_STEP_TIME_BUDGETS.get(step_name)

    def _log_step_timing_summary(self, pipeline_results: Dict[str, Any]) -> None:
        step_metrics = pipeline_results.get('metrics', {}).get('steps', {})
        if not step_metrics:
            return

        for step_name in (
            'multi_horizon_profit_labeler',
            'feature_lookback_optimization',
            'interactive_feature_generation',
            'final_feature_selection',
        ):
#             # This is intentional - no action needed for these steps
            continue

        self.logger.info("📈 Step duration summary:")
        self.event_logger.info(
            "Step duration summary emitted",
            context=self._build_event_context(
                'pipeline.summary',
                extra={'steps': sorted(step_metrics.keys())},
            ),
        )
        for spec in self._get_ordered_step_specs(sequence_only=True):
            step_name = spec.name
            metrics = step_metrics.get(step_name)
            if not metrics:
                continue
            label = self._get_step_display_name(step_name)
            duration = metrics.get('duration_seconds') or 0.0
            budget = metrics.get('budget_seconds')
            over_budget = metrics.get('over_budget')
            over_budget_seconds = metrics.get('over_budget_seconds') or 0.0
            status_icon = '⚠️' if over_budget else '✅'
            budget_text = ''
            if budget is not None:
                budget_text = f" (budget {budget:.2f}s"
                if over_budget:
                    budget_text += f", over by {over_budget_seconds:.2f}s"
                budget_text += ')'
            message = f"   {status_icon} {label}: {duration:.2f}s{budget_text}"
            self.logger.info(message)
            self.event_logger.info(
                "Step duration summary",
                context=self._build_event_context(
                    f'pipeline.summary.{step_name}',
                    duration_ms=duration * 1000.0,
                    extra={
                        'duration_seconds': duration,
                        'budget_seconds': budget,
                        'over_budget': over_budget,
                        'over_budget_seconds': over_budget_seconds,
                    },
                ),
            )

    @staticmethod
    def _get_step_display_name(step_name: str) -> str:
        spec = STEP_REGISTRY.get(step_name)
        if spec is not None:
            return spec.display_name
        return step_name

    def _emit_pipeline_metrics(
        self,
        metrics_sink: MetricsSink,
        step_metric_records: List[Dict[str, Any]],
        results: Dict[str, Any],
    ) -> None:
        total_row_count = sum(record.get('row_count_total') or 0 for record in step_metric_records)
        max_memory_peak = max(
            (record.get('memory_peak_mb') for record in step_metric_records if record.get('memory_peak_mb') is not None),
            default=None,
        )
        label_skew_values = [
            record.get('label_distribution_skew')
            for record in step_metric_records
            if record.get('label_distribution_skew') is not None
        ]
        average_label_skew = (sum(label_skew_values) / len(label_skew_values)) if label_skew_values else None
        artifact_count = sum(record.get('artifact_count') or 0 for record in step_metric_records)
        row_detail_map = {
            record['step_name']: record.get('row_count_total', 0)
            for record in step_metric_records
            if record.get('step_name')
        }
        metadata_keys = sorted({
            key
            for record in step_metric_records
            for key in (record.get('metadata_keys', '') or '').split(',')
            if key
        })

        pipeline_record = self._base_metrics_record()
        pipeline_record.update({
            'record_type': 'pipeline',
            'step_name': 'pipeline_total',
            'status': 'completed' if results.get('success') else 'failed',
            'success': results.get('success', False),
            'duration_seconds': results.get('execution_time'),
            'row_count_total': total_row_count,
            'row_count_details': json.dumps(row_detail_map, sort_keys=True),
            'memory_peak_mb': max_memory_peak,
            'label_distribution_skew': average_label_skew,
            'timestamp': datetime.utcnow().isoformat(),
            'artifact_count': artifact_count,
            'metadata_keys': ','.join(metadata_keys),
            'total_steps': results.get('total_steps'),
            'completed_steps': results.get('completed_steps'),
            'total_row_count': total_row_count,
            'max_memory_peak_mb': max_memory_peak,
            'average_label_distribution_skew': average_label_skew,
            'error_message': results.get('error_message') or '',
        })

        metrics_sink.write(pipeline_record)

    def _finalize_results(
        self,
        results: Dict[str, Any],
        start_time: datetime,
        metrics_sink: Optional[MetricsSink],
        step_metric_records: List[Dict[str, Any]],
        end_time: Optional[datetime] = None,
    ) -> PipelineResultDict:
        end_time = end_time or datetime.now()
        results['execution_time'] = (end_time - start_time).total_seconds()
        results.setdefault('metrics', {})['total_execution_time'] = results['execution_time']
        if metrics_sink is not None:
            self._emit_pipeline_metrics(metrics_sink, step_metric_records, results)
        return results

    @staticmethod
    def _base_metrics_record() -> Dict[str, Any]:
        fields = [
            'record_type',
            'step_name',
            'status',
            'success',
            'duration_seconds',
            'row_count_total',
            'row_count_details',
            'memory_peak_mb',
            'label_distribution_skew',
            'timestamp',
            'artifact_count',
            'metadata_keys',
            'total_steps',
            'completed_steps',
            'total_row_count',
            'max_memory_peak_mb',
            'average_label_distribution_skew',
            'error_message',
        ]
        return {field: None for field in fields}

    @staticmethod
    def _extract_row_counts(artifacts: Dict[str, Any]) -> Dict[str, int]:
        row_counts: Dict[str, int] = {}

        def _walk(prefix: str, value: Any) -> None:
            key_name = prefix or 'root'
            if isinstance(value, pd.DataFrame):
                row_counts[key_name] = int(value.shape[0])
            elif isinstance(value, pd.Series):
                row_counts[key_name] = int(value.shape[0])
            elif isinstance(value, np.ndarray):
                row_counts[key_name] = int(value.shape[0])
            elif isinstance(value, dict):
                for key, nested_value in value.items():
                    next_prefix = f"{key_name}.{key}" if prefix else str(key)
                    _walk(next_prefix, nested_value)
            elif isinstance(value, (list, tuple)):
                for index, nested_value in enumerate(value):
                    next_prefix = f"{key_name}[{index}]"
                    _walk(next_prefix, nested_value)

        for key, value in artifacts.items():
            _walk(key, value)

        return row_counts

    @staticmethod
    def _compute_label_distribution_skew(metadata: Dict[str, Any]) -> Optional[float]:
        if not metadata:
            return None

        label_distribution = metadata.get('label_distribution')
        if not isinstance(label_distribution, dict):
            return None

        values: List[float] = []

        def _collect_values(data: Any) -> None:
            if isinstance(data, dict):
                for nested in data.values():
                    _collect_values(nested)
            elif isinstance(data, (int, float)):
                values.append(float(data))

        _collect_values(label_distribution)

        if not values:
            return None

        total = sum(values)
        if total > 0:
            normalized = [value / total for value in values]
        else:
            normalized = values

        return max(normalized) - min(normalized) if normalized else None

    @staticmethod
    def _get_memory_usage_mb() -> Optional[float]:
        if resource is None:
            return None
        usage = resource.getrusage(resource.RUSAGE_SELF)
        max_rss = getattr(usage, 'ru_maxrss', None)
        if max_rss is None:
            return None
        # On Linux ru_maxrss is reported in kilobytes.
        return max_rss / 1024.0

    async def execute(
        self,
        training_input: Dict[str, Any],
        pipeline_state: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute the pre-training pipeline with backward compatible interface and enhanced logging.

        Args:
            training_input: Input data for the pipeline
            pipeline_state: Current pipeline state

        Returns:
            Dictionary containing execution results
        """
        tprint("🔄 Executing pre-training pipeline with backward compatible interface")
        tprint_info(f"📥 Pipeline input keys: {list(training_input.keys())}")

        # Extract configuration from pipeline state
        locator = pipeline_state.get('data_locator')
        data_dir_key = pipeline_state.get('data_dir_key', 'market_data')
        data_dir = pipeline_state.get('data_dir')
        if data_dir is None and isinstance(locator, DataLocator):
            data_dir = str(locator.data_path(data_dir_key))
            tprint_info(f"📁 Resolved data directory: {data_dir}")

        custom_params_source = pipeline_state.get('custom_params')
        custom_params = dict(custom_params_source) if isinstance(custom_params_source, Mapping) else {}
        random_seed_value = pipeline_state.get('random_seed')
        resolved_seed: Optional[int] = None
        if random_seed_value is not None:
            try:
                resolved_seed = int(random_seed_value)
                custom_params.setdefault('random_seed', resolved_seed)
            except (TypeError, ValueError):
                resolved_seed = None

        pipeline_overrides: Dict[str, Any] = {}
        if isinstance(pipeline_state, Mapping):
            for key in (
                'timeframe',
                'role',
                'pipeline_role',
                'execution_role',
                'run_role',
                'analyst_mode',
                'is_analyst_run',
            ):
                if key in pipeline_state:
                    pipeline_overrides[key] = pipeline_state[key]

        resolved_timeframe = SubPipelineConfig.resolve_timeframe(
            explicit=None,
            custom_params=custom_params,
            pipeline_overrides=pipeline_overrides,
        )
        pipeline_overrides['timeframe'] = resolved_timeframe

        custom_params.setdefault('timeframe', resolved_timeframe)

        config = SubPipelineConfig(
            symbol=pipeline_state.get('symbol', 'ETHUSDT'),
            exchange=pipeline_state.get('exchange', 'binance'),
            timeframe=resolved_timeframe,
            data_dir=data_dir,
            mode=ExecutionMode.FULL,  # Default to full mode
            custom_params=custom_params,
            random_seed=resolved_seed,
            data_locator=locator if isinstance(locator, DataLocator) else None,
            data_dir_key=data_dir_key,
            pipeline=pipeline_overrides,
        )

        # Execute the pipeline
        return await self.execute_pipeline(config)

    def _prepare_component_pipeline_state(self, config: SubPipelineConfig) -> PipelineState:
        """Construct the pipeline state passed to individual components."""
        locator = self._data_locator or self._resolve_data_locator(config)

        if config.data_dir:
            data_dir_path = Path(config.data_dir).expanduser()
            if not data_dir_path.is_absolute():
                data_dir_path = locator.data_path(config.data_dir_key).joinpath(data_dir_path).resolve()
        else:
            data_dir_path = locator.data_path(config.data_dir_key)

        cache_dir_path = locator.cache_path(config.cache_dir_key)
        artifacts_dir_path = locator.artifacts_path(config.artifacts_dir_key)
        generated_dir_path = locator.generated_path(config.generated_dir_key)
        outcomes_dir_path = locator.artifacts_path(
            config.outcomes_dir_key,
            ensure_exists=True,
        )
        final_feature_selection_dir = locator.generated_path(
            config.final_feature_selection_dir_key,
            ensure_exists=True,
        )

        pipeline_state = PipelineState({
            'symbol': config.symbol,
            'exchange': config.exchange,
            'timeframe': config.timeframe,
            'data_dir': str(data_dir_path),
            'data_cache_dir': str(cache_dir_path),
            'artifacts_dir': str(artifacts_dir_path),
            'generated_dir': str(generated_dir_path),
            'outcomes_dir': str(outcomes_dir_path),
            'final_feature_selection_dir': str(final_feature_selection_dir),
            'data_dir_key': config.data_dir_key,
            'cache_dir_key': config.cache_dir_key,
            'artifacts_dir_key': config.artifacts_dir_key,
            'generated_dir_key': config.generated_dir_key,
            'outcomes_dir_key': config.outcomes_dir_key,
            'final_feature_selection_dir_key': config.final_feature_selection_dir_key,
            'data_locator': locator,
            'custom_params': self._build_component_custom_params(config),
            'quality_thresholds': self._get_quality_thresholds(config),
            'market_data_batch_size': config.market_data_batch_size,
            'market_data_window_days': config.market_data_window_days,
            'independent_component_execution': True,  # Flag to indicate we're running a component independently
        })

        pipeline_state.update({k: v for k, v in self._current_pipeline_state.items() if k not in pipeline_state})
        if self._seeded_rngs is not None:
            pipeline_state['random_seed'] = self._seeded_rngs.seed
            pipeline_state['python_rng'] = self._seeded_rngs.python
            pipeline_state['numpy_rng'] = self._seeded_rngs.numpy
            pipeline_state['seeded_rngs'] = self._seeded_rngs

        # Add artifacts from previous steps for artifact chaining
        if self._artifact_chain:
            pipeline_state['previous_artifacts'] = self._get_all_previous_artifacts()
            self.logger.info(f"📦 Added {len(self._artifact_chain)} previous artifacts to pipeline state")

        regime_cache_path = config.custom_params.get('regime_cache_path') if config.custom_params else None
        if not regime_cache_path:
            data_cache_dir = config.custom_params.get('data_cache_dir') if config.custom_params else None
            if data_cache_dir:
                regime_cache_path = str((Path(data_cache_dir).expanduser() / 'nas_tas_clustering').resolve(strict=False))
        # Note: regime is a string, not an object with cache_dir attribute
        # if not regime_cache_path and self._settings.regime.cache_dir is not None:
        #     regime_cache_path = str(self._settings.regime.cache_dir.resolved)

        if regime_cache_path:
            pipeline_state['regime_cache_path'] = regime_cache_path

        regime_split = config.custom_params.get('regime_data_splitting_result')
        if regime_split is None:
            regime_split = self._current_pipeline_state.get('regime_data_splitting_result')

        if regime_split is not None:
            pipeline_state['regime_data_splitting_result'] = regime_split
            self._current_pipeline_state['regime_data_splitting_result'] = regime_split

        return pipeline_state

    def _get_quality_thresholds(self, config: SubPipelineConfig) -> Dict[str, float]:
        """Return the data quality thresholds configured for the pipeline."""
        return {
            'label_imbalance': float(config.label_imbalance_warning_threshold),
            'nan_rate': float(config.nan_rate_warning_threshold),
            'duplicate_index': float(config.duplicate_index_warning_threshold),
        }

    def _build_component_custom_params(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """Augment component custom parameters with quality thresholds."""
        params = dict(config.custom_params or {})
        seed = config.random_seed if config.random_seed is not None else self._active_seed
        if seed is not None:
            params.setdefault('random_seed', seed)
        params.setdefault('quality_thresholds', self._get_quality_thresholds(config))
        if config.market_data_batch_size is not None:
            params.setdefault('market_data_batch_size', config.market_data_batch_size)
        if config.market_data_window_days is not None:
            params.setdefault('market_data_window_days', config.market_data_window_days)

        # Add direction settings from main pipeline config
        params.setdefault('enable_long_positions', config.enable_long_positions)
        params.setdefault('enable_short_positions', config.enable_short_positions)

        return params

    def _prepare_interactive_training_input(
        self,
        pipeline_state: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """Prepare the training input dictionary for interaction feature generation."""

        mh_result = pipeline_state.get('multi_horizon_labeling_result')
        if mh_result is None:
            mh_result = self._current_pipeline_state.get('multi_horizon_labeling_result', {})

        # Check if we're running a component that doesn't require multi-horizon labeling results
        # This allows components like analyst_profit_labeler to run independently
        running_independent_component = pipeline_state.get('independent_component_execution', False)

        if not mh_result and not running_independent_component:
            raise ValueError("Multi-horizon labeling result is required for interactive feature generation")

        market_data_batches = mh_result.get('market_data_batches') if mh_result else None
        market_data = mh_result.get('market_data') if mh_result else None

        if market_data is None and market_data_batches:
            market_data = pd.concat(market_data_batches, axis=0).sort_index()

        # If running independently and market_data is still None, try to load it from pipeline state
        if market_data is None and running_independent_component:
            market_data = pipeline_state.get('market_data')
            if market_data is None:
#                 # Try to load from data locator if available
                data_locator = pipeline_state.get('data_locator')
                if data_locator:
                    symbol = pipeline_state.get('symbol')
                    exchange = pipeline_state.get('exchange')
                    timeframe = pipeline_state.get('timeframe')
                    if symbol and exchange and timeframe:
                        try:
                            market_data = data_locator.load_klines(
                                symbol=symbol,
                                exchange=exchange,
                                timeframe=timeframe
                            )
                        except Exception as e:
                            self.logger.warning(f"⚠️ Could not load market data from data locator: {e}")

        if market_data is None and not running_independent_component:
            raise ValueError("Market data is missing from multi-horizon labeling result")

        labels_df = mh_result.get('labeled_data') if mh_result else None
        if labels_df is None or (isinstance(labels_df, pd.DataFrame) and len(labels_df) == 0):
            labels_df = mh_result.get('labels') if mh_result else None
        targets: Dict[str, pd.Series] = {}
        if isinstance(labels_df, pd.DataFrame):
            targets = {column: labels_df[column] for column in labels_df.columns}

        training_input: Dict[str, Any] = {
            'data': market_data,
            'targets': targets,
        }

        if market_data_batches:
            training_input['data_batches'] = list(market_data_batches)

        return training_input
    def _resolve_random_seed(self, config: SubPipelineConfig) -> int:
        """Resolve the seed for deterministic execution."""
        env_seed = os.environ.get('ARES_RANDOM_SEED')
        if env_seed is not None:
            try:
                return int(env_seed)
            except (TypeError, ValueError) as e:
                self.logger.warning(f"Invalid ARES_RANDOM_SEED value '{env_seed}': {e}. Using default seed.")
        custom_params = config.custom_params or {}
        for key in ('rng_seed', 'seed', 'random_seed'):
            if key in custom_params and custom_params[key] is not None:
                try:
                    return int(custom_params[key])
                except (TypeError, ValueError):
                    continue
        return 42

    def _extend_with_quality_metadata(
        self,
        metadata: Optional[Dict[str, Any]],
        metrics: Dict[str, Any],
        alerts: List[str],
        config: SubPipelineConfig,
    ) -> Dict[str, Any]:
        """Merge computed quality metrics and alerts into metadata."""
        merged_metadata: Dict[str, Any] = dict(metadata or {})
        if metrics:
            merged_metadata['quality_metrics'] = metrics
        if alerts:
            merged_metadata['quality_alerts'] = alerts
        merged_metadata.setdefault('quality_thresholds', self._get_quality_thresholds(config))
        return merged_metadata

    def _analyze_component_quality(
        self,
        component_name: str,
        artifacts: Dict[str, Any],
        config: SubPipelineConfig,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Inspect component artifacts and compute quality metrics and alerts."""
        thresholds = self._get_quality_thresholds(config)
        metrics: Dict[str, Any] = {}
        alerts: List[str] = []
        visited_frames: Dict[int, Dict[str, Any]] = {}

        def log_warning(message: str) -> None:
            alerts.append(message)
            self.logger.warning(message)

        def handle_dataframe(dataset_name: str, df: pd.DataFrame) -> None:
            if df is None or len(df) == 0:
                return
            df_id = id(df)
            if df_id in visited_frames:
                metrics[dataset_name] = visited_frames[df_id]
                return

#             # Validate DataFrame schema before computing metrics with memory monitoring
            try:
#                 # Optimize DataFrame for validation
                optimized_df = self.validation_manager.optimize_dataframe_for_validation(df)

                if dataset_name.lower().endswith(('ohlcv', 'price', 'market_data')):
#                     # Validate as OHLCV data with caching and memory monitoring
                    def validate_ohlcv_func(dataframe):
                        return validate_raw_ohlcv(dataframe, context=f'quality_metrics.{dataset_name}')

                    validated_df = self.validation_manager.validate_with_memory_monitoring(
                        f"ohlcv_validation_{dataset_name}",
                        optimized_df,
                        validate_ohlcv_func,
                        context=f'quality_metrics.{dataset_name}'
                    )
                else:
#                     # Validate as engineered features with caching and memory monitoring
                    def validate_features_func(dataframe):
                        return validate_engineered_features(dataframe, context=f'quality_metrics.{dataset_name}')

                    validated_df = self.validation_manager.validate_with_memory_monitoring(
                        f"features_validation_{dataset_name}",
                        optimized_df,
                        validate_features_func,
                        context=f'quality_metrics.{dataset_name}'
                    )

#                 # Use validated dataframe for metrics computation
                dataset_metrics, dataset_alerts = self._compute_dataframe_quality_metrics(
                    component_name,
                    dataset_name,
                    validated_df,
                    thresholds,
                )
            except SchemaValidationException as e:
#                 # Log validation error but continue with original dataframe
                log_warning(f"⚠️ Schema validation failed for {dataset_name}: {e}")
                dataset_metrics, dataset_alerts = self._compute_dataframe_quality_metrics(
                    component_name,
                    dataset_name,
                    df,
                    thresholds,
                )
            except Exception as e:
#                 # Handle unexpected validation errors
                log_warning(f"⚠️ Unexpected error during validation for {dataset_name}: {e}")
                dataset_metrics, dataset_alerts = self._compute_dataframe_quality_metrics(
                    component_name,
                    dataset_name,
                    df,
                    thresholds,
                )

            visited_frames[df_id] = dataset_metrics
            metrics[dataset_name] = dataset_metrics
            for alert in dataset_alerts:
                log_warning(alert)

        def traverse(prefix: str, value: Any, visited: Optional[Set[int]] = None) -> None:
            if visited is None:
                visited = set()

#             # Cycle detection: prevent infinite recursion on circular references
            value_id = id(value)
            if value_id in visited:
                log_warning(f"⚠️ Circular reference detected in artifacts at {prefix}, skipping")
                return

            if isinstance(value, pd.DataFrame):
                handle_dataframe(prefix, value)
            elif isinstance(value, dict):
                visited.add(value_id)
                try:
                    for key, nested_value in value.items():
                        nested_prefix = f"{prefix}.{key}" if prefix else key
                        traverse(nested_prefix, nested_value, visited)
                finally:
                    visited.discard(value_id)  # Clean up to avoid memory leaks

        traverse('', artifacts)
        return metrics, alerts

    def _compute_dataframe_quality_metrics(
        self,
        component_name: str,
        dataset_name: str,
        df: pd.DataFrame,
        thresholds: Dict[str, float],
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Compute quality statistics for a DataFrame and build alert messages."""
        dataset_metrics: Dict[str, Any] = {}
        alerts: List[str] = []

        total_cells = int(df.size)
        nan_rate = float(df.isna().sum().sum() / total_cells) if total_cells else 0.0
        dataset_metrics['nan_rate'] = nan_rate
        if nan_rate >= thresholds['nan_rate'] > 0:
            alerts.append(
                f"⚠️ [{component_name}] {dataset_name} NaN rate {nan_rate:.2%} exceeds threshold {thresholds['nan_rate']:.2%}"
            )

        duplicate_share = 0.0
        if len(df.index) > 0:
            duplicate_mask = df.index.duplicated()
            duplicate_share = float(duplicate_mask.mean()) if duplicate_mask.any() else 0.0
        dataset_metrics['duplicate_index_share'] = duplicate_share
        if duplicate_share > thresholds['duplicate_index'] > 0:
            alerts.append(
                f"⚠️ [{component_name}] {dataset_name} duplicate index share {duplicate_share:.2%} exceeds threshold {thresholds['duplicate_index']:.2%}."
                f" Remediation: Remove or deduplicate data points. Check for data merging issues, "
                f"time synchronization problems, or incorrect data aggregation logic."
            )

        column_metrics: Dict[str, Any] = {}
        max_dominant_share = 0.0
        max_dominant_column: Optional[str] = None
        for column in df.columns:
            series = df[column].dropna()
            unique_count = series.nunique(dropna=True)
            if unique_count == 0 or unique_count > 20:
                continue
            counts = series.value_counts(dropna=True, normalize=True)
            if counts.empty:
                continue
            dominant_value = counts.index[0]
            dominant_share = float(counts.iloc[0])
            column_metrics[str(column)] = {
                'dominant_value': str(dominant_value),
                'dominant_share': dominant_share,
                'distribution': {str(k): float(v) for k, v in counts.items()},
            }
            if dominant_share > max_dominant_share:
                max_dominant_share = dominant_share
                max_dominant_column = str(column)
            if dominant_share >= thresholds['label_imbalance'] > 0:
                alerts.append(
                    f"⚠️ [{component_name}] {dataset_name}.{column} dominant label share {dominant_share:.2%} exceeds threshold {thresholds['label_imbalance']:.2%}."
                    f" Remediation: Consider data augmentation, class balancing techniques (SMOTE, undersampling), "
                    f"or adjusting class weights in model training. This may indicate sampling bias in data collection."
                )

        if column_metrics:
            dataset_metrics['label_balance'] = {
                'columns': column_metrics,
                'max_dominant_share': max_dominant_share,
                'max_dominant_column': max_dominant_column,
            }

        return dataset_metrics, alerts

    async def _execute_analyst_profit_labeler(
        self,
        config: SubPipelineConfig,
        run_metadata: Dict[str, Any],
    ) -> SubPipelineResult:
        """Execute analyst profit labeler (15m timeframe, strategic labeling)."""
        result = SubPipelineResult(
            sub_pipeline_name='analyst_profit_labeler',
            status=SubPipelineStatus.RUNNING,
            start_time=datetime.now()
        )
        result.error_code = self._default_step_error_code('analyst_profit_labeler')

        try:
            # Analyst runs use 15m timeframe for tactical analysis
            # Normalize timeframe format (15m -> 15m for data loading)
            analyst_timeframe_config = '15m'  # For component config
            analyst_timeframe_data = '15m'  # For data loading

            # Convert config to component config with analyst-specific timeframe
            component_config = ComponentConfig(
                name='analyst_profit_labeler',
                enabled=True,
                parameters={
                    'symbol': config.symbol,
                    'exchange': config.exchange,
                    'timeframe': analyst_timeframe_config,
                    'data_dir': config.data_dir,
                    **self._build_component_custom_params(config)
                }
            )

            # Create component using factory
            self.logger.info(f"🔍 Available components: {ComponentFactory.get_available_components()}")
            component = ComponentFactory.create_component('analyst_profit_labeler', component_config)
            if component is None:
                raise ValueError(f"Component 'analyst_profit_labeler' not found. Available components: {ComponentFactory.get_available_components()}")
            if hasattr(component, 'set_run_metadata'):
                component.set_run_metadata(run_metadata)

            # Execute component
            pipeline_state = self._prepare_component_pipeline_state(config)

            # Load market data directly for labeling component
            # analyst_profit_labeler creates labels, it doesn't consume them
            from src.utils.data.klines_parquet import KlinesParquetManager

            try:
                klines_manager = KlinesParquetManager()
                # Load data respecting the configured date range (light mode = 20 days)
                market_data = klines_manager.read_data(
                    symbol=config.symbol,
                    interval=analyst_timeframe_data,
                    start_date=config.start_date,
                    end_date=config.end_date
                )
                
                if market_data is None or len(market_data) == 0:
                    raise ValueError(f"No market data found for {config.symbol} {analyst_timeframe_data}")

                # Add market data to pipeline state
                pipeline_state['market_data'] = market_data
                
                # Execute the component
                component_result = await component.execute(market_data, pipeline_state)
                
                if component_result.success:
                    result.status = SubPipelineStatus.COMPLETED
                    result.success = True
                    result.metadata = component_result.metadata
                    result.artifacts = component_result.artifacts
                else:
                    result.status = SubPipelineStatus.FAILED
                    result.error_message = component_result.error_message
                    result.error_code = self._extract_component_error_code(
                        component_result,
                        self._default_step_error_code('analyst_profit_labeler'),
                    )

                    if component_result.success:
                        result.metadata = self._merge_run_metadata(result.metadata)
                        result.artifacts = component_result.artifacts
                    else:
                        failure_context = {
                            'component': 'analyst_profit_labeler',
                            'symbol': config.symbol,
                            'timeframe': analyst_timeframe_config,
                            'error': component_result.error_message
                        }
                        result.failure = self._create_failure(
                            'analyst_profit_labeler',
                            result.error_code or self._default_step_error_code('analyst_profit_labeler'),
                            result.error_message or 'Analyst profit labeler failed',
                            context=failure_context,
                        )

            except Exception as e:
                result.status = SubPipelineStatus.FAILED
                result.end_time = datetime.now()
                result.duration_seconds = (result.end_time - result.start_time).total_seconds()
                self.logger.error(f"❌ Analyst profit labeler failed - missing dependencies: {e}")
                result.metadata = self._merge_run_metadata(result.metadata)
                result.error_code = f"{self._default_step_error_code('analyst_profit_labeler')}_IMPORT"
                result.failure = self._create_failure(
                    'analyst_profit_labeler',
                    result.error_code,
                    f"Missing dependencies: {e}",
                    context={'component': 'analyst_profit_labeler', 'error': str(e)}
                )
            except FileNotFoundError as e:
                result.status = SubPipelineStatus.FAILED
                result.end_time = datetime.now()
                result.duration_seconds = (result.end_time - result.start_time).total_seconds()
                self.logger.error(f"❌ Analyst profit labeler failed - missing files: {e}")
                result.metadata = self._merge_run_metadata(result.metadata)
                result.error_code = f"{self._default_step_error_code('analyst_profit_labeler')}_MISSING_FILE"
                result.failure = self._create_failure(
                    'analyst_profit_labeler',
                    result.error_code,
                    f"Missing files: {e}",
                    context={'component': 'analyst_profit_labeler', 'error': str(e)}
                )
            except Exception as e:
                result.status = SubPipelineStatus.FAILED
                result.end_time = datetime.now()
                result.duration_seconds = (result.end_time - result.start_time).total_seconds()
                self.logger.error(f"❌ Analyst profit labeler failed - unexpected error: {e}")
                result.metadata = self._merge_run_metadata(result.metadata)
                result.error_code = f"{self._default_step_error_code('analyst_profit_labeler')}_UNEXPECTED"
                result.failure = self._create_failure(
                    'analyst_profit_labeler',
                    result.error_code,
                    f"Unexpected error: {e}",
                    context={'component': 'analyst_profit_labeler', 'error': str(e)}
                )

        except Exception as e:
            result.status = SubPipelineStatus.FAILED
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            self.logger.error(f"❌ Analyst profit labeler failed: {e}")
            result.metadata = self._merge_run_metadata(result.metadata)
            result.error_code = f"{self._default_step_error_code('analyst_profit_labeler')}_EXECUTION"
            result.failure = self._create_failure(
                'analyst_profit_labeler',
                result.error_code,
                f"Execution failed: {e}",
                context={'component': 'analyst_profit_labeler', 'error': str(e)}
            )

        result.end_time = datetime.now()
        result.duration_seconds = (result.end_time - result.start_time).total_seconds()
        return result

    async def _execute_tactician_entry_labeler(
        self,
        config: SubPipelineConfig,
        run_metadata: Dict[str, Any],
    ) -> SubPipelineResult:
        """Execute tactician entry labeler (15m timeframe, tactical labeling)."""
        result = SubPipelineResult(
            sub_pipeline_name='tactician_entry_labeler',
            status=SubPipelineStatus.RUNNING,
            start_time=datetime.now()
        )
        result.error_code = self._default_step_error_code('tactician_entry_labeler')

        try:
            # Tactician runs use 15m timeframe for tactical entry timing
            tactician_timeframe = '15m'

            # Convert config to component config with tactician-specific timeframe
            component_config = ComponentConfig(
                name='tactician_entry_labeler',
                enabled=True,
                parameters=self._build_component_custom_params(config)
            )

            # Create component using factory
            component = ComponentFactory.create_component('tactician_entry_labeler', component_config)
            if hasattr(component, 'set_run_metadata'):
                component.set_run_metadata(run_metadata)

            # Execute component
            pipeline_state = self._prepare_component_pipeline_state(config)

            # Load market data directly for labeling component
            # tactician_entry_labeler creates labels, it doesn't consume them
            from src.utils.data.klines_parquet import KlinesParquetManager

            try:
                klines_manager = KlinesParquetManager()
                market_data = klines_manager.read_data(
                    symbol=config.symbol,
                    interval=tactician_timeframe,
                    start_date=config.start_date,
                    end_date=config.end_date
                )
                self.logger.info(f"✅ Loaded {len(market_data)} rows of market data for tactician labeling")
            except Exception as e:
                self.logger.error(f"❌ Failed to load market data: {e}")
                raise ValueError(f"Could not load market data for tactician_entry_labeler: {e}") from e

            if market_data is None or len(market_data) == 0:
                raise ValueError("Market data is required for tactician_entry_labeler but none was loaded")

            component_result = await component.execute(market_data, pipeline_state)
            component_result.metadata = self._merge_run_metadata(component_result.metadata)
            result.warnings = self._collect_component_warnings(component_result)
            result.errors = self._collect_component_errors(component_result)

            result.status = SubPipelineStatus.COMPLETED if component_result.success else SubPipelineStatus.FAILED
            result.success = component_result.success
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            result.artifacts = component_result.artifacts
            result.error_message = component_result.error_message
            result.error_code = self._extract_component_error_code(
                component_result,
                self._default_step_error_code('tactician_entry_labeler'),
            )

            if component_result.success:
                try:
                    artifacts = component_result.artifacts or {}
                    if 'multi_horizon_labeling_result' in artifacts:
                        validated_contract = validate_multi_horizon_labeling_result(
                            artifacts['multi_horizon_labeling_result'],
                            context='sub_pipeline.tactician_entry_labeler',
                        )
                        artifacts['multi_horizon_labeling_result'] = validated_contract
                        result.artifacts = artifacts
                except DataContractValidationError as contract_error:
                    self.event_logger.error(
                        "Contract validation error",
                        context={
                            'run_id': self._run_metadata.get('run_id'),
                            'step': 'tactician_entry_labeler.validation',
                            'symbol': self._run_metadata.get('symbol'),
                            'timeframe': self._run_metadata.get('timeframe'),
                            'error': str(contract_error),
                        },
                    )
                    return self._handle_contract_error(result, 'tactician_entry_labeler', contract_error)

                quality_metrics, quality_alerts = self._analyze_component_quality(
                    'tactician_entry_labeler',
                    result.artifacts,
                    config,
                )
                result.metadata = self._extend_with_quality_metadata(
                    component_result.metadata,
                    quality_metrics,
                    quality_alerts,
                    config,
                )
                if result.warnings:
                    warnings_meta = result.metadata.setdefault('warnings', [])
                    self._extend_messages(warnings_meta, result.warnings)
            else:
                result.metadata = self._merge_run_metadata(component_result.metadata or {
                    'component_type': 'tactician_entry_labeler'
                })
                failure_context = {
                    'component_metadata': component_result.metadata,
                    'artifacts_keys': sorted((component_result.artifacts or {}).keys()),
                }
                result.failure = self._create_failure(
                    'tactician_entry_labeler',
                    result.error_code or self._default_step_error_code('tactician_entry_labeler'),
                    result.error_message or 'Tactician entry labeler failed',
                    context=failure_context,
                )
                if result.error_message:
                    self._extend_messages(result.errors, [result.error_message])

        except ImportError as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = f"Missing dependencies: {str(e)}"
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            self.logger.error(f"❌ Tactician entry labeler failed - missing dependencies: {e}")
            result.metadata = self._merge_run_metadata(result.metadata)
            result.error_code = f"{self._default_step_error_code('tactician_entry_labeler')}_IMPORT"
            result.failure = self._create_failure(
                'tactician_entry_labeler',
                result.error_code,
                result.error_message,
                exception=e,
            )
            self._extend_messages(result.errors, [result.error_message])
        except FileNotFoundError as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = f"Missing files: {str(e)}"
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            self.logger.error(f"❌ Tactician entry labeler failed - missing files: {e}")
            result.metadata = self._merge_run_metadata(result.metadata)
            result.error_code = f"{self._default_step_error_code('tactician_entry_labeler')}_MISSING_FILE"
            result.failure = self._create_failure(
                'tactician_entry_labeler',
                result.error_code,
                result.error_message,
                exception=e,
            )
            self._extend_messages(result.errors, [result.error_message])
        except Exception as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = f"Unexpected error: {str(e)}"
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            self.logger.error(f"❌ Tactician entry labeler failed - unexpected error: {e}")
            result.metadata = self._merge_run_metadata(result.metadata)
            result.error_code = f"{self._default_step_error_code('tactician_entry_labeler')}_UNEXPECTED"
            result.failure = self._create_failure(
                'tactician_entry_labeler',
                result.error_code,
                result.error_message,
                exception=e,
            )
            self._extend_messages(result.errors, [result.error_message or str(e)])

        return result

    async def _execute_unified_data_driven_pipeline(
        self,
        config: SubPipelineConfig,
        run_metadata: Dict[str, Any],
    ) -> SubPipelineResult:
        """Execute unified data-driven pipeline with tactician/analyst labeling integration."""
        result = SubPipelineResult(
            sub_pipeline_name='unified_data_driven_pipeline',
            status=SubPipelineStatus.RUNNING,
            start_time=datetime.now()
        )
        result.error_code = self._default_step_error_code('unified_data_driven_pipeline')

        try:
#             # Import the unified data driven pipeline
            from src.training.steps.pre_training.unified_data_driven_pipeline.consolidated_pipeline import (
                UnifiedDataDrivenPipeline,
                create_default_config
            )

#             # Determine labeling type based on custom parameters or run metadata
            labeling_type = config.custom_params.get('labeling_type', 'analyst')
            if 'tactician' in run_metadata.get('run_type', '').lower():
                labeling_type = 'tactician'
            elif 'analyst' in run_metadata.get('run_type', '').lower():
                labeling_type = 'analyst'

#             # Both analyst and tactician use 15m timeframe
            if config.timeframe != '15m':
                tprint_info(f"🔄 Adjusting timeframe from {config.timeframe} to 15m for unified pipeline")
                config.timeframe = '15m'

#             # Create pipeline configuration
            pipeline_config = create_default_config()
            pipeline_config.labeling_type = labeling_type
            pipeline_config.enable_labeling_optimization = True
            pipeline_config.labeling_quality_threshold = 0.7

#             # Create pipeline instance
            pipeline = UnifiedDataDrivenPipeline(pipeline_config)

#             # Prepare pipeline state
            pipeline_state = self._prepare_component_pipeline_state(config)

#             # Load market data
            from src.utils.data.klines_parquet import KlinesParquetManager

            try:
                klines_manager = KlinesParquetManager()
                market_data = klines_manager.read_data(
                    symbol=config.symbol,
                    timeframe=config.timeframe,
                    start_date=config.start_date,
                    end_date=config.end_date
                )
                self.logger.info(f"✅ Loaded {len(market_data)} rows of market data for unified pipeline")
            except Exception as e:
                self.logger.error(f"❌ Failed to load market data: {e}")
                raise ValueError(f"Could not load market data for unified_data_driven_pipeline: {e}") from e

            if market_data is None or len(market_data) == 0:
                raise ValueError("Market data is required for unified_data_driven_pipeline but none was loaded")

#             # Get labels from previous pipeline steps (analyst_profit_labeler or tactician_entry_labeler)
            labels = None

#             # Look for labels in various possible artifact structures
            if 'multi_horizon_labeling_result' in pipeline_state:
                labeling_result = pipeline_state['multi_horizon_labeling_result']
                if 'labeled_data' in labeling_result and 'target' in labeling_result['labeled_data'].columns:
                    labels = labeling_result['labeled_data']['target']
                    tprint_info(f"✅ Using {len(labels)} labels from multi_horizon_labeling_result")
                elif 'labels' in labeling_result and isinstance(labeling_result['labels'], pd.DataFrame):
#                     # Extract target column from labels DataFrame
                    if 'target' in labeling_result['labels'].columns:
                        labels = labeling_result['labels']['target']
                        tprint_info(f"✅ Using {len(labels)} labels from multi_horizon_labeling_result.labels")
                    else:
#                         # Use the first column as target if no 'target' column
                        labels = labeling_result['labels'].iloc[:, 0]
                        tprint_info(f"✅ Using {len(labels)} labels from multi_horizon_labeling_result.labels (first column)")

            if labels is None:
                raise ValueError("No labels found from previous pipeline steps. Please ensure analyst_profit_labeler or tactician_entry_labeler runs before unified_data_driven_pipeline.")

#             # Execute pipeline with labels
            pipeline_result = await pipeline.process(market_data, targets=labels, pipeline_state=pipeline_state)

#             # Process results
            if pipeline_result.success:
                tprint_success(f"✅ Unified data-driven pipeline completed successfully")

#                 # Store results in artifacts
                result.artifacts = {
                    'unified_pipeline_result': pipeline_result,
                    'labeling_type': labeling_type,
                    'selected_features': pipeline_result.selected_features,
                    'interaction_features': pipeline_result.interaction_features,
                    'feature_quality_score': pipeline_result.feature_quality_score,
                    'performance_metrics': pipeline_result.performance_metrics
                }

                result.metadata = {
                    'labeling_type': labeling_type,
                    'features_selected': len(pipeline_result.selected_features),
                    'interactions_generated': len(pipeline_result.interaction_features),
                    'quality_score': pipeline_result.feature_quality_score,
                    'execution_time': pipeline_result.execution_time_seconds
                }

                result.status = SubPipelineStatus.COMPLETED
            else:
                tprint_error(f"❌ Unified data-driven pipeline failed: {pipeline_result.error_message}")
                result.status = SubPipelineStatus.FAILED
                result.error_message = pipeline_result.error_message
                result.failure = self._create_failure(
                    'unified_data_driven_pipeline',
                    result.error_code,
                    result.error_message or 'Unified data-driven pipeline failed'
                )

        except ImportError as e:
            result.error_message = f"Missing dependencies: {str(e)}"
            result.status = SubPipelineStatus.FAILED
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            self.logger.error(f"❌ Unified data-driven pipeline failed - missing dependencies: {e}")
            result.metadata = self._merge_run_metadata(result.metadata)
            result.error_code = f"{self._default_step_error_code('unified_data_driven_pipeline')}_IMPORT"
            result.failure = self._create_failure(
                'unified_data_driven_pipeline',
                result.error_code,
                result.error_message,
                exception=e,
            )
            self._extend_messages(result.errors, [result.error_message])

        except Exception as e:
            result.error_message = f"Unexpected error: {str(e)}"
            result.status = SubPipelineStatus.FAILED
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            self.logger.error(f"❌ Unified data-driven pipeline failed - unexpected error: {e}")
            result.metadata = self._merge_run_metadata(result.metadata)
            result.error_code = f"{self._default_step_error_code('unified_data_driven_pipeline')}_UNEXPECTED"
            result.failure = self._create_failure(
                'unified_data_driven_pipeline',
                result.error_code,
                result.error_message,
                exception=e,
            )
            self._extend_messages(result.errors, [result.error_message or str(e)])

        # Store artifacts in chain for next steps (success or failure)
        if result.artifacts:
            self._store_artifacts_in_chain('unified_data_driven_pipeline', result.artifacts)

        return result

    # Feature Generation Steps Executors
    async def _execute_feature_generation_data_validation_step(
        self,
        config: SubPipelineConfig,
        run_metadata: Dict[str, Any],
    ) -> SubPipelineResult:
        """Execute feature generation data validation step."""
        result = SubPipelineResult(
            sub_pipeline_name='feature_generation_data_validation_step',
            status=SubPipelineStatus.RUNNING,
            start_time=datetime.now()
        )
        result.error_code = self._default_step_error_code('feature_generation_data_validation_step')

        try:
            from src.training.steps.pre_training.unified_data_driven_pipeline.steps.feature_generation_data_validation_step import (
                handle_feature_generation_data_validation_step
            )

#             # Execute step
            step_result = await handle_feature_generation_data_validation_step(
                symbol=config.symbol,
                timeframe=config.timeframe,
                direction=config.direction,
                intensity=config.execution_mode.value if hasattr(config, 'execution_mode') else 'blank',
                lookback_days=getattr(config, 'lookback_days', None),
                start_date=config.start_date,
                end_date=config.end_date,
                exchange=getattr(config, 'exchange', 'binance'),
                custom_overrides=config.custom_params
            )

            if step_result.success:
                result.status = SubPipelineStatus.COMPLETED
                result.artifacts = step_result.artifacts or {}
                result.metadata = {
                    'data_quality_score': step_result.data_quality_score,
                    'validation_metadata': step_result.validation_metadata
                }
            else:
                result.status = SubPipelineStatus.FAILED
                result.error_message = step_result.error_message

        except Exception as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = str(e)
            result.error_code = f"{self._default_step_error_code('feature_generation_data_validation_step')}_UNEXPECTED"

        result.end_time = datetime.now()
        result.duration_seconds = (result.end_time - result.start_time).total_seconds()
        return result

    async def _execute_feature_generation_feature_generation_step(
        self,
        config: SubPipelineConfig,
        run_metadata: Dict[str, Any],
    ) -> SubPipelineResult:
        """Execute feature generation feature generation step."""
        result = SubPipelineResult(
            sub_pipeline_name='feature_generation_feature_generation_step',
            status=SubPipelineStatus.RUNNING,
            start_time=datetime.now()
        )
        result.error_code = self._default_step_error_code('feature_generation_feature_generation_step')

        try:
            from src.training.steps.pre_training.unified_data_driven_pipeline.steps.feature_generation_feature_generation_step import (
                handle_feature_generation_feature_generation_step
            )

#             # Execute step
            step_result = await handle_feature_generation_feature_generation_step(
                symbol=config.symbol,
                timeframe=config.timeframe,
                direction=config.direction,
                intensity=config.execution_mode.value if hasattr(config, 'execution_mode') else 'blank',
                lookback_days=getattr(config, 'lookback_days', None),
                start_date=config.start_date,
                end_date=config.end_date,
                exchange=getattr(config, 'exchange', 'binance'),
                custom_overrides=config.custom_params
            )

            if step_result.success:
                result.status = SubPipelineStatus.COMPLETED
                result.artifacts = step_result.artifacts or {}
                result.metadata = {
                    'feature_count': len(step_result.features.columns),
                    'generation_metrics': step_result.generation_metrics
                }
            else:
                result.status = SubPipelineStatus.FAILED
                result.error_message = step_result.error_message

        except Exception as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = str(e)
            result.error_code = f"{self._default_step_error_code('feature_generation_feature_generation_step')}_UNEXPECTED"

        result.end_time = datetime.now()
        result.duration_seconds = (result.end_time - result.start_time).total_seconds()
        return result

    async def _execute_feature_generation_feature_selection_step(
        self,
        config: SubPipelineConfig,
        run_metadata: Dict[str, Any],
    ) -> SubPipelineResult:
        """Execute feature generation feature selection step."""
        result = SubPipelineResult(
            sub_pipeline_name='feature_generation_feature_selection_step',
            status=SubPipelineStatus.RUNNING,
            start_time=datetime.now()
        )
        result.error_code = self._default_step_error_code('feature_generation_feature_selection_step')

        try:
            from src.training.steps.pre_training.unified_data_driven_pipeline.steps.feature_generation_feature_selection_step import (
                handle_feature_generation_feature_selection_step
            )

#             # Execute step
            step_result = await handle_feature_generation_feature_selection_step(
                symbol=config.symbol,
                timeframe=config.timeframe,
                direction=config.direction,
                intensity=config.execution_mode.value if hasattr(config, 'execution_mode') else 'blank',
                lookback_days=getattr(config, 'lookback_days', None),
                start_date=config.start_date,
                end_date=config.end_date,
                exchange=getattr(config, 'exchange', 'binance'),
                custom_overrides=config.custom_params
            )

            if step_result.success:
                result.status = SubPipelineStatus.COMPLETED
                result.artifacts = step_result.artifacts or {}
                result.metadata = {
                    'selected_feature_count': len(step_result.selected_features.columns),
                    'selection_metrics': step_result.selection_metrics
                }
            else:
                result.status = SubPipelineStatus.FAILED
                result.error_message = step_result.error_message

        except Exception as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = str(e)
            result.error_code = f"{self._default_step_error_code('feature_generation_feature_selection_step')}_UNEXPECTED"

        result.end_time = datetime.now()
        result.duration_seconds = (result.end_time - result.start_time).total_seconds()
        return result

    async def _execute_feature_generation_period_optimization_step(
        self,
        config: SubPipelineConfig,
        run_metadata: Dict[str, Any],
    ) -> SubPipelineResult:
        """Execute feature generation period optimization step."""
        result = SubPipelineResult(
            sub_pipeline_name='feature_generation_period_optimization_step',
            status=SubPipelineStatus.RUNNING,
            start_time=datetime.now()
        )
        result.error_code = self._default_step_error_code('feature_generation_period_optimization_step')

        try:
            from src.training.steps.pre_training.unified_data_driven_pipeline.steps.feature_generation_period_optimization_step import (
                handle_feature_generation_period_optimization_step
            )

#             # Execute step
            step_result = await handle_feature_generation_period_optimization_step(
                symbol=config.symbol,
                timeframe=config.timeframe,
                direction=config.direction,
                intensity=config.execution_mode.value if hasattr(config, 'execution_mode') else 'blank',
                lookback_days=getattr(config, 'lookback_days', None),
                start_date=config.start_date,
                end_date=config.end_date,
                exchange=getattr(config, 'exchange', 'binance'),
                custom_overrides=config.custom_params
            )

            if step_result.success:
                result.status = SubPipelineStatus.COMPLETED
                result.artifacts = step_result.artifacts or {}
                result.metadata = {
                    'optimized_periods': len(step_result.optimal_periods),
                    'optimization_metrics': step_result.optimization_metrics
                }
            else:
                result.status = SubPipelineStatus.FAILED
                result.error_message = step_result.error_message

        except Exception as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = str(e)
            result.error_code = f"{self._default_step_error_code('feature_generation_period_optimization_step')}_UNEXPECTED"

        result.end_time = datetime.now()
        result.duration_seconds = (result.end_time - result.start_time).total_seconds()
        return result

    async def _execute_feature_generation_period_lookback_optimization(
        self,
        config: SubPipelineConfig,
        run_metadata: Dict[str, Any],
    ) -> SubPipelineResult:
        """Execute combined period and lookback optimization step."""
        result = SubPipelineResult(
            sub_pipeline_name='feature_generation_period_lookback_optimization_step',
            status=SubPipelineStatus.RUNNING,
            start_time=datetime.now()
        )
        result.error_code = self._default_step_error_code('feature_generation_period_lookback_optimization_step')

        combined_artifacts = {}
        combined_metadata = {}
        total_optimized_periods = 0
        total_optimized_lookbacks = 0

        try:
            # Execute period optimization step
            tprint("🔄 Executing period optimization...")
            period_result = await self._execute_feature_generation_period_optimization_step(config, run_metadata)

            if period_result.success:
                combined_artifacts.update(period_result.artifacts or {})
                combined_metadata.update(period_result.metadata or {})
                total_optimized_periods = period_result.metadata.get('optimized_periods', 0)
                tprint_success("✅ Period optimization completed")
            else:
                tprint_error(f"❌ Period optimization failed: {period_result.error_message}")
                result.status = SubPipelineStatus.FAILED
                result.error_message = f"Period optimization failed: {period_result.error_message}"
                result.end_time = datetime.now()
                result.duration_seconds = (result.end_time - result.start_time).total_seconds()
                return result

            # Execute lookback optimization step
            tprint("🔄 Executing lookback optimization...")
            lookback_result = await self._execute_feature_generation_lookback_optimization_step(config, run_metadata)

            if lookback_result.success:
                combined_artifacts.update(lookback_result.artifacts or {})
                combined_metadata.update(lookback_result.metadata or {})
                total_optimized_lookbacks = lookback_result.metadata.get('optimized_lookbacks', 0)
                tprint_success("✅ Lookback optimization completed")
            else:
                tprint_error(f"❌ Lookback optimization failed: {lookback_result.error_message}")
                result.status = SubPipelineStatus.FAILED
                result.error_message = f"Lookback optimization failed: {lookback_result.error_message}"
                result.end_time = datetime.now()
                result.duration_seconds = (result.end_time - result.start_time).total_seconds()
                return result

            # Success - combine results
            result.status = SubPipelineStatus.COMPLETED
            result.artifacts = combined_artifacts
            result.metadata = {
                'optimized_periods': total_optimized_periods,
                'optimized_lookbacks': total_optimized_lookbacks,
                'period_optimization_metrics': period_result.metadata.get('optimization_metrics', {}),
                'lookback_optimization_metrics': lookback_result.metadata.get('optimization_metrics', {}),
                'combined_optimization_metrics': {
                    'total_optimized_parameters': total_optimized_periods + total_optimized_lookbacks,
                    'period_optimization_success': period_result.success,
                    'lookback_optimization_success': lookback_result.success,
                }
            }
            tprint_success(f"✅ Combined period + lookback optimization completed (periods: {total_optimized_periods}, lookbacks: {total_optimized_lookbacks})")

            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            return result

        except Exception as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = str(e)
            result.error_code = f"{self._default_step_error_code('feature_generation_period_lookback_optimization_step')}_UNEXPECTED"
            tprint_error(f"❌ Combined period + lookback optimization failed: {e}")

        result.end_time = datetime.now()
        result.duration_seconds = (result.end_time - result.start_time).total_seconds()
        return result

    async def _execute_feature_generation_lookback_optimization_step(
        self,
        config: SubPipelineConfig,
        run_metadata: Dict[str, Any],
    ) -> SubPipelineResult:
        """Execute feature generation lookback optimization step."""
        result = SubPipelineResult(
            sub_pipeline_name='feature_generation_lookback_optimization_step',
            status=SubPipelineStatus.RUNNING,
            start_time=datetime.now()
        )
        result.error_code = self._default_step_error_code('feature_generation_lookback_optimization_step')

        try:
            from src.training.steps.pre_training.unified_data_driven_pipeline.steps.feature_generation_lookback_optimization_step import (
                handle_feature_generation_lookback_optimization_step
            )

#             # Execute step
            step_result = await handle_feature_generation_lookback_optimization_step(
                symbol=config.symbol,
                timeframe=config.timeframe,
                direction=config.direction,
                intensity=config.execution_mode.value if hasattr(config, 'execution_mode') else 'blank',
                lookback_days=getattr(config, 'lookback_days', None),
                start_date=config.start_date,
                end_date=config.end_date,
                exchange=getattr(config, 'exchange', 'binance'),
                custom_overrides=config.custom_params
            )

            if step_result.success:
                result.status = SubPipelineStatus.COMPLETED
                result.artifacts = step_result.artifacts or {}
                result.metadata = {
                    'optimized_lookbacks': len(step_result.optimal_lookbacks),
                    'optimization_metrics': step_result.optimization_metrics
                }
            else:
                result.status = SubPipelineStatus.FAILED
                result.error_message = step_result.error_message

        except Exception as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = str(e)
            result.error_code = f"{self._default_step_error_code('feature_generation_lookback_optimization_step')}_UNEXPECTED"

        result.end_time = datetime.now()
        result.duration_seconds = (result.end_time - result.start_time).total_seconds()
        return result

    async def _execute_feature_generation_interaction_generation_step(
        self,
        config: SubPipelineConfig,
        run_metadata: Dict[str, Any],
    ) -> SubPipelineResult:
        """Execute feature generation interaction generation step."""
        result = SubPipelineResult(
            sub_pipeline_name='feature_generation_interaction_generation_step',
            status=SubPipelineStatus.RUNNING,
            start_time=datetime.now()
        )
        result.error_code = self._default_step_error_code('feature_generation_interaction_generation_step')

        try:
            from src.training.steps.pre_training.unified_data_driven_pipeline.steps.feature_generation_interaction_generation_step import (
                handle_feature_generation_interaction_generation_step
            )

#             # Execute step
            step_result = await handle_feature_generation_interaction_generation_step(
                symbol=config.symbol,
                timeframe=config.timeframe,
                direction=config.direction,
                intensity=config.execution_mode.value if hasattr(config, 'execution_mode') else 'blank',
                lookback_days=getattr(config, 'lookback_days', None),
                start_date=config.start_date,
                end_date=config.end_date,
                exchange=getattr(config, 'exchange', 'binance'),
                custom_overrides=config.custom_params
            )

            if step_result.success:
                result.status = SubPipelineStatus.COMPLETED
                result.artifacts = step_result.artifacts or {}
                result.metadata = {
                    'interaction_feature_count': len(step_result.interaction_features.columns),
                    'generation_metrics': step_result.generation_metrics
                }
            else:
                result.status = SubPipelineStatus.FAILED
                result.error_message = step_result.error_message

        except Exception as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = str(e)
            result.error_code = f"{self._default_step_error_code('feature_generation_interaction_generation_step')}_UNEXPECTED"

        result.end_time = datetime.now()
        result.duration_seconds = (result.end_time - result.start_time).total_seconds()
        return result

    async def _execute_feature_generation_vectorization_step(
        self,
        config: SubPipelineConfig,
        run_metadata: Dict[str, Any],
    ) -> SubPipelineResult:
        """Execute feature generation vectorization step."""
        result = SubPipelineResult(
            sub_pipeline_name='feature_generation_vectorization_step',
            status=SubPipelineStatus.RUNNING,
            start_time=datetime.now()
        )
        result.error_code = self._default_step_error_code('feature_generation_vectorization_step')

        try:
            from src.training.steps.pre_training.unified_data_driven_pipeline.steps.feature_generation_vectorization_step import (
                handle_feature_generation_vectorization_step
            )

#             # Execute step
            step_result = await handle_feature_generation_vectorization_step(
                symbol=config.symbol,
                timeframe=config.timeframe,
                direction=config.direction,
                intensity=config.execution_mode.value if hasattr(config, 'execution_mode') else 'blank',
                lookback_days=getattr(config, 'lookback_days', None),
                start_date=config.start_date,
                end_date=config.end_date,
                exchange=getattr(config, 'exchange', 'binance'),
                custom_overrides=config.custom_params
            )

            if step_result.success:
                result.status = SubPipelineStatus.COMPLETED
                result.artifacts = step_result.artifacts or {}
                result.metadata = {
                    'vectorized_feature_count': len(step_result.vectorized_features.columns),
                    'performance_metrics': step_result.performance_metrics
                }
            else:
                result.status = SubPipelineStatus.FAILED
                result.error_message = step_result.error_message

        except Exception as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = str(e)
            result.error_code = f"{self._default_step_error_code('feature_generation_vectorization_step')}_UNEXPECTED"

        result.end_time = datetime.now()
        result.duration_seconds = (result.end_time - result.start_time).total_seconds()
        return result

    async def _execute_feature_generation_labeling_integration_step(
        self,
        config: SubPipelineConfig,
        run_metadata: Dict[str, Any],
    ) -> SubPipelineResult:
        """Execute feature generation labeling integration step."""
        result = SubPipelineResult(
            sub_pipeline_name='feature_generation_labeling_integration_step',
            status=SubPipelineStatus.RUNNING,
            start_time=datetime.now()
        )
        result.error_code = self._default_step_error_code('feature_generation_labeling_integration_step')

        try:
            from src.training.steps.pre_training.unified_data_driven_pipeline.steps.feature_generation_labeling_integration_step import (
                handle_feature_generation_labeling_integration_step
            )

#             # Execute step
            step_result = await handle_feature_generation_labeling_integration_step(
                symbol=config.symbol,
                timeframe=config.timeframe,
                direction=config.direction,
                intensity=config.execution_mode.value if hasattr(config, 'execution_mode') else 'blank',
                lookback_days=getattr(config, 'lookback_days', None),
                start_date=config.start_date,
                end_date=config.end_date,
                exchange=getattr(config, 'exchange', 'binance'),
                custom_overrides=config.custom_params
            )

            if step_result.success:
                result.status = SubPipelineStatus.COMPLETED
                result.artifacts = step_result.artifacts or {}
                result.metadata = {
                    'labeled_data_shape': step_result.labeled_data.shape,
                    'quality_metrics': step_result.quality_metrics
                }
            else:
                result.status = SubPipelineStatus.FAILED
                result.error_message = step_result.error_message

        except Exception as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = str(e)
            result.error_code = f"{self._default_step_error_code('feature_generation_labeling_integration_step')}_UNEXPECTED"

        result.end_time = datetime.now()
        result.duration_seconds = (result.end_time - result.start_time).total_seconds()
        return result

    async def _execute_feature_generation_final_validation_step(
        self,
        config: SubPipelineConfig,
        run_metadata: Dict[str, Any],
    ) -> SubPipelineResult:
        """Execute feature generation final validation step."""
        result = SubPipelineResult(
            sub_pipeline_name='feature_generation_final_validation_step',
            status=SubPipelineStatus.RUNNING,
            start_time=datetime.now()
        )
        result.error_code = self._default_step_error_code('feature_generation_final_validation_step')

        try:
            from src.training.steps.pre_training.unified_data_driven_pipeline.steps.feature_generation_final_validation_step import (
                handle_feature_generation_final_validation_step
            )

#             # Execute step
            step_result = await handle_feature_generation_final_validation_step(
                symbol=config.symbol,
                timeframe=config.timeframe,
                direction=config.direction,
                intensity=config.execution_mode.value if hasattr(config, 'execution_mode') else 'blank',
                lookback_days=getattr(config, 'lookback_days', None),
                start_date=config.start_date,
                end_date=config.end_date,
                exchange=getattr(config, 'exchange', 'binance'),
                custom_overrides=config.custom_params
            )

            if step_result.success:
                result.status = SubPipelineStatus.COMPLETED
                result.artifacts = step_result.artifacts or {}
                result.metadata = {
                    'final_dataset_shape': step_result.final_dataset.shape,
                    'quality_metrics': step_result.quality_metrics,
                    'pipeline_summary': step_result.pipeline_summary
                }
            else:
                result.status = SubPipelineStatus.FAILED
                result.error_message = step_result.error_message

        except Exception as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = str(e)
            result.error_code = f"{self._default_step_error_code('feature_generation_final_validation_step')}_UNEXPECTED"

        result.end_time = datetime.now()
        result.duration_seconds = (result.end_time - result.start_time).total_seconds()
        return result

    async def _execute_feature_lookback_optimization(
        self,
        config: SubPipelineConfig,
        run_metadata: Dict[str, Any],
    ) -> SubPipelineResult:
        """Execute feature lookback optimization with timeframe support."""
        result = SubPipelineResult(
            sub_pipeline_name='feature_lookback_optimization',
            status=SubPipelineStatus.RUNNING,
            start_time=datetime.now()
        )
        result.error_code = self._default_step_error_code('feature_lookback_optimization')

        try:
#             # Convert config to component config
            component_config = ComponentConfig(
                symbol=config.symbol,
                exchange=config.exchange,
                timeframe=config.timeframe,
                data_dir=config.data_dir,
                custom_params=self._build_component_custom_params(config)
            )

#             # Create component using factory
            component = ComponentFactory.create_component('feature_lookback_optimization', component_config)
            if hasattr(component, 'set_run_metadata'):
                component.set_run_metadata(run_metadata)

#             # Load market data for feature lookback optimization using ares launcher integration
#             from src.training.steps.pre_training.feature_lookback_optimization.ares_launcher_integration import AresLauncherFeatureLookbackOptimizer
            import pandas as pd

#             # Normalize timeframe for data loading (60m -> 1h)
            timeframe_map = {'60m': '1h', '1h': '1h', '4h': '4h', '1d': '1d', '15m': '15m', '5m': '5m', '1m': '1m'}
            normalized_timeframe = timeframe_map.get(config.timeframe, config.timeframe)

            try:
#                 # Use ares launcher integration for data loading
#                 tprint("🔧 [SUB_PIPELINE] Initializing ares launcher integration for feature lookback optimization...")
#                 ares_optimizer = AresLauncherFeatureLookbackOptimizer()
#                 tprint_success("✅ [SUB_PIPELINE] Ares launcher integration initialized")
#
#                 # Create pipeline state for ares integration
#                 ares_pipeline_state = {
#                     'symbol': config.symbol,
#                     'exchange': config.exchange,
#                     'timeframe': normalized_timeframe,
#                     'execution_mode': getattr(config, 'execution_mode', 'light'),  # Default to light mode
#                     'lookback_days': getattr(config, 'lookback_days', None),
#                     'intensity_percentage': getattr(config, 'intensity_percentage', None)
#                 }
#
#                 tprint("📋 [SUB_PIPELINE] Pipeline state for ares integration:")
#                 tprint_info(f"   → Symbol: {config.symbol}")
#                 tprint_info(f"   → Exchange: {config.exchange}")
#                 tprint_info(f"   → Timeframe: {normalized_timeframe}")
#                 tprint_info(f"   → Execution mode: {ares_pipeline_state['execution_mode']}")
#                 tprint_debug(f"   → Lookback days: {ares_pipeline_state['lookback_days']}")
#                 tprint_debug(f"   → Intensity percentage: {ares_pipeline_state['intensity_percentage']}")
#                 tprint_debug(f"   → Full pipeline state: {ares_pipeline_state}")
#
#                 # Load data using ares launcher integration
#                 tprint("📥 [SUB_PIPELINE] Loading data using ares launcher integration...")
#                 market_data = ares_optimizer.load_data_for_optimization(
#                     symbol=config.symbol,
#                     timeframe=normalized_timeframe,
#                     pipeline_state=ares_pipeline_state
#                 )
#
#                 if market_data is not None and not len(market_data) == 0:
#                     tprint_success(f"✅ [SUB_PIPELINE] Loaded {len(market_data)} rows of market data via ares launcher integration")
#                     tprint_info(f"📊 [SUB_PIPELINE] Data summary:")
#                     tprint_info(f"   → Shape: {market_data.shape}")
#                     tprint_info(f"   → Date range: {market_data.index.min().date()} to {market_data.index.max().date()}")
#                     tprint_info(f"   → Data mode: {market_data.attrs.get('ares_mode', 'Unknown')}")
#                     tprint_info(f"   → Lookback days: {market_data.attrs.get('lookback_days', 'Unknown')}")
#                     tprint_debug(f"   → Data columns: {list(market_data.columns)}")
#                     tprint_debug(f"   → Memory usage: {market_data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
#
#                     # Basic data transformation
#                     tprint("🔄 [SUB_PIPELINE] Applying basic data transformations...")
#                     if 'open_time' in market_data.columns and not isinstance(market_data.index, pd.DatetimeIndex):
#                         tprint_debug("   → Setting 'open_time' as index")
#                         market_data = market_data.set_index('open_time')
#                     elif not isinstance(market_data.index, pd.DatetimeIndex):
#                         try:
#                             tprint_debug("   → Converting index to datetime")
#                             market_data.index = pd.to_datetime(market_data.index)
#                         except Exception as e:
#                             # Keep original dtype if conversion fails
#                             tprint_debug(f"   → Could not convert index to datetime: {e}")
#                     else:
#                         tprint_debug("   → Index is already datetime, no conversion needed")
#                 else:
                    tprint_warning("⚠️ [SUB_PIPELINE] No market data loaded via ares launcher integration, will pass None to component")
                    tprint_debug("   → This could be due to:")
                    tprint_debug("     - No data available for the specified parameters")
                    tprint_debug("     - Data loading error in ares launcher integration")
                    tprint_debug("     - Invalid symbol/timeframe combination")
                    market_data = None
            except Exception as e:
                tprint_warning(f"⚠️ [SUB_PIPELINE] Could not load market data via ares launcher integration: {e}")
                tprint_debug(f"   → Exception type: {type(e).__name__}")
                tprint_debug(f"   → Exception details: {str(e)}")
                tprint_debug("   → Will pass None to component and let it handle data loading")
                market_data = None

#             # Execute component
            pipeline_state = self._prepare_component_pipeline_state(config)
            component_result = await component.execute(market_data, pipeline_state)
            component_result.metadata = self._merge_run_metadata(component_result.metadata)
            result.warnings = self._collect_component_warnings(component_result)
            result.errors = self._collect_component_errors(component_result)
            result.warnings = self._collect_component_warnings(component_result)
            result.errors = self._collect_component_errors(component_result)
            result.warnings = self._collect_component_warnings(component_result)
            result.errors = self._collect_component_errors(component_result)
            result.warnings = self._collect_component_warnings(component_result)
            result.errors = self._collect_component_errors(component_result)

            result.status = SubPipelineStatus.COMPLETED if component_result.success else SubPipelineStatus.FAILED
            result.success = component_result.success
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            result.artifacts = component_result.artifacts
            result.error_message = component_result.error_message
            result.error_code = self._extract_component_error_code(
                component_result,
                self._default_step_error_code('feature_lookback_optimization'),
            )
            if component_result.success:
                quality_metrics, quality_alerts = self._analyze_component_quality(
                    'feature_lookback_optimization',
                    result.artifacts,
                    config,
                )
                result.metadata = self._extend_with_quality_metadata(
                    component_result.metadata,
                    quality_metrics,
                    quality_alerts,
                    config,
                )
                if result.warnings:
                    warnings_meta = result.metadata.setdefault('warnings', [])
                    self._extend_messages(warnings_meta, result.warnings)
            else:
                result.metadata = self._merge_run_metadata(component_result.metadata or {})
                failure_context = {
                    'component_metadata': component_result.metadata,
                    'artifacts_keys': sorted((component_result.artifacts or {}).keys()),
                }
                result.failure = self._create_failure(
                    'feature_lookback_optimization',
                    result.error_code or self._default_step_error_code('feature_lookback_optimization'),
                    result.error_message or 'Feature lookback optimization failed',
                    context=failure_context,
                )
                if result.error_message:
                    self._extend_messages(result.errors, [result.error_message])

        except ImportError as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = f"Missing dependencies: {str(e)}"
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            self.logger.error(f"❌ Feature lookback optimization failed - missing dependencies: {e}")
            result.metadata = self._merge_run_metadata(result.metadata)
            result.error_code = f"{self._default_step_error_code('feature_lookback_optimization')}_IMPORT"
            result.failure = self._create_failure(
                'feature_lookback_optimization',
                result.error_code,
                result.error_message,
                exception=e,
            )
            self._extend_messages(result.errors, [result.error_message])
        except FileNotFoundError as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = f"Missing files: {str(e)}"
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            self.logger.error(f"❌ Feature lookback optimization failed - missing files: {e}")
            result.metadata = self._merge_run_metadata(result.metadata)
            result.error_code = f"{self._default_step_error_code('feature_lookback_optimization')}_MISSING_FILE"
            result.failure = self._create_failure(
                'feature_lookback_optimization',
                result.error_code,
                result.error_message,
                exception=e,
            )
            self._extend_messages(result.errors, [result.error_message])
        except Exception as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = str(e)
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            self.logger.error(f"❌ Feature lookback optimization failed with unexpected error: {e}")
            trace = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
            self.logger.error(f"🔍 Error details: {trace}")
            result.metadata = self._merge_run_metadata(result.metadata)
            result.error_code = f"{self._default_step_error_code('feature_lookback_optimization')}_UNEXPECTED"
            result.failure = self._create_failure(
                'feature_lookback_optimization',
                result.error_code,
                result.error_message or 'Feature lookback optimization failed',
                exception=e,
                traceback_str=trace,
            )
            self._extend_messages(result.errors, [result.error_message or str(e)])

        # Store artifacts in chain for next steps (success or failure)
        if result.artifacts:
            self._store_artifacts_in_chain('feature_lookback_optimization', result.artifacts)

        return result

    async def _execute_interactive_feature_generation(
        self,
        config: SubPipelineConfig,
        run_metadata: Dict[str, Any],
    ) -> SubPipelineResult:
        """Execute interactive feature generation with timeframe support."""
        result = SubPipelineResult(
            sub_pipeline_name='interactive_feature_generation',
            status=SubPipelineStatus.RUNNING,
            start_time=datetime.now()
        )

        # TODO: Implement interactive feature generation logic
        # This method appears to be incomplete - add the actual implementation here

        result.status = SubPipelineStatus.COMPLETED
        result.end_time = datetime.now()
        result.duration_seconds = (result.end_time - result.start_time).total_seconds()

        return result

    async def _execute_final_feature_selection(
        self,
        config: SubPipelineConfig,
        run_metadata: Dict[str, Any],
    ) -> SubPipelineResult:
        """Execute final feature selection with timeframe support."""
        tprint("🎯 Starting final feature selection execution")
        tprint_info(f"📊 Symbol: {config.symbol}, Exchange: {config.exchange}, Timeframe: {config.timeframe}")

        result = SubPipelineResult(
            sub_pipeline_name='final_feature_selection',
            status=SubPipelineStatus.RUNNING,
            start_time=datetime.now()
        )
        result.error_code = self._default_step_error_code('final_feature_selection')

        try:
#             # Convert config to component config
            component_config = ComponentConfig(
                symbol=config.symbol,
                exchange=config.exchange,
                timeframe=config.timeframe,
                data_dir=config.data_dir,
                custom_params=self._build_component_custom_params(config)
            )

#             # Create component using factory
            component = ComponentFactory.create_component('final_feature_selection', component_config)
            if hasattr(component, 'set_run_metadata'):
                component.set_run_metadata(run_metadata)

#             # Execute component
            pipeline_state = self._prepare_component_pipeline_state(config)
            component_result = await component.execute(None, pipeline_state)
            component_result.metadata = self._merge_run_metadata(component_result.metadata)

            result.status = SubPipelineStatus.COMPLETED if component_result.success else SubPipelineStatus.FAILED
            result.success = component_result.success
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            result.artifacts = component_result.artifacts
            result.error_message = component_result.error_message
            result.error_code = self._extract_component_error_code(
                component_result,
                self._default_step_error_code('final_feature_selection'),
            )
            if component_result.success:
                try:
                    artifacts = component_result.artifacts or {}
                    selection_payload = artifacts.get('final_feature_selection_result')
                    if selection_payload and 'final_features' in selection_payload:
                        validated_selection = validate_selection_artifact(
                            selection_payload,
                            context='sub_pipeline.final_feature_selection',
                        )
                        artifacts['final_feature_selection_result'] = validated_selection
                        result.artifacts = artifacts
                except DataContractValidationError as contract_error:
                    self.event_logger.error(
                        "Contract validation error",
                        context={
                            'run_id': self._run_metadata.get('run_id'),
                            'step': 'final_feature_selection.validation',
                            'symbol': self._run_metadata.get('symbol'),
                            'timeframe': self._run_metadata.get('timeframe'),
                            'error': str(contract_error),
                        },
                    )
                    return self._handle_contract_error(result, 'final_feature_selection', contract_error)

                quality_metrics, quality_alerts = self._analyze_component_quality(
                    'final_feature_selection',
                    result.artifacts,
                    config,
                )
                result.metadata = self._extend_with_quality_metadata(
                    component_result.metadata,
                    quality_metrics,
                    quality_alerts,
                    config,
                )
                if result.warnings:
                    warnings_meta = result.metadata.setdefault('warnings', [])
                    self._extend_messages(warnings_meta, result.warnings)
            else:
                result.metadata = self._merge_run_metadata(component_result.metadata)
                failure_context = {
                    'component_metadata': component_result.metadata,
                    'artifacts_keys': sorted((component_result.artifacts or {}).keys()),
                }
                result.failure = self._create_failure(
                    'final_feature_selection',
                    result.error_code or self._default_step_error_code('final_feature_selection'),
                    result.error_message or 'Final feature selection failed',
                    context=failure_context,
                )
                if result.error_message:
                    self._extend_messages(result.errors, [result.error_message])

        except ImportError as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = f"Missing dependencies: {str(e)}"
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            self.logger.error(f"❌ Final feature selection failed - missing dependencies: {e}")
            result.metadata = self._merge_run_metadata(result.metadata)
            result.error_code = f"{self._default_step_error_code('final_feature_selection')}_IMPORT"
            result.failure = self._create_failure(
                'final_feature_selection',
                result.error_code,
                result.error_message,
                exception=e,
            )
            self._extend_messages(result.errors, [result.error_message])
        except FileNotFoundError as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = f"Missing files: {str(e)}"
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            self.logger.error(f"❌ Final feature selection failed - missing files: {e}")
            result.metadata = self._merge_run_metadata(result.metadata)
            result.error_code = f"{self._default_step_error_code('final_feature_selection')}_MISSING_FILE"
            result.failure = self._create_failure(
                'final_feature_selection',
                result.error_code,
                result.error_message,
                exception=e,
            )
            self._extend_messages(result.errors, [result.error_message])
        except Exception as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = str(e)
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            self.logger.error(f"❌ Final feature selection failed with unexpected error: {e}")
            trace = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
            self.logger.error(f"🔍 Error details: {trace}")
# LEGACY METHOD REMOVED - _execute_interactive_feature_generation (lines 4903-5070)
        component_flags = [
            'analyst_profit_labeler', 'tactician_entry_labeler', 'multi_horizon_profit_labeler'
        ]

        for flag in component_flags:
            if config.custom_params.get(flag, False):
                return flag

        return None

    async def execute_sub_pipeline(self, sub_pipeline_name: str, config: SubPipelineConfig) -> SubPipelineResult:
        """Execute a specific sub-pipeline."""
        # Get the step specification
        spec = self._get_step_spec(sub_pipeline_name)
        if spec is None:
            raise ValueError(f"Unknown sub-pipeline: {sub_pipeline_name}")

        # Execute the step
        self.logger.info(f"🚀 Executing pre-training step: {sub_pipeline_name}")
        
        # Get the executor method
        executor_method = getattr(self, spec.executor_method, None)
        if executor_method is None:
            raise ValueError(f"Executor method '{spec.executor_method}' not found for sub-pipeline '{sub_pipeline_name}'")

        # Create run metadata
        run_metadata = {
            'sub_pipeline_name': sub_pipeline_name,
            'timestamp': datetime.now().isoformat(),
            'config': config.to_dict() if hasattr(config, 'to_dict') else {}
        }

        # Execute the method
        result = await executor_method(config, run_metadata)
        
        # Store the result
        self.results.append(result)
        
        return result

    async def execute_sub_pipeline_with_next(self, sub_pipeline_name: str, config: SubPipelineConfig) -> SubPipelineResult:
        """Execute a specific sub-pipeline and automatically trigger subsequent sub-pipelines."""
        # For pre-training, execute the default enabled steps in sequence
        ordered_specs = self._get_ordered_step_specs(sequence_only=True)
        ordered_names = [spec.name for spec in ordered_specs]

        try:
            start_index = ordered_names.index(sub_pipeline_name)
            steps_to_run = ordered_specs[start_index:]
        except ValueError:
#             # Step not part of the default sequence; execute it directly
            direct_spec = self._get_step_spec(sub_pipeline_name)
            if direct_spec is None:
                raise ValueError(f"Unknown sub-pipeline: {sub_pipeline_name}")
            steps_to_run = [direct_spec]

        # Execute all steps starting from the specified one
        for spec in steps_to_run:
            step_name = spec.name
            self.logger.info(f"🚀 Executing pre-training step: {step_name}")

            result = await self.execute_sub_pipeline(step_name, config)
            self.results.append(result)

#             # If this step failed, stop the sequence
            if not result.success:
                self.logger.error(f"❌ Step {step_name} failed, stopping execution sequence")
                break

        # Return the first result (the one that was requested)
        return self.results[0] if self.results else None

    def get_available_sub_pipelines(self) -> List[str]:
        """Get list of available sub-pipelines."""
        return [
            'analyst-labeler',
            'tactician-labeler',
            'feature_generation_data_validation_step',
            'feature_generation_labeling_integration_step',
            'feature_generation_feature_generation_step',
            'feature_generation_feature_selection_step',
            'feature_generation_period_lookback_optimization_step',
            'feature_generation_interaction_generation_step',
            'feature_generation_vectorization_step',
            'feature_generation_final_validation_step'
        ]

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary with all results."""
        return {
            'total_sub_pipelines': len(self.results),
            'successful_sub_pipelines': len([r for r in self.results if r.success]),
            'failed_sub_pipelines': len([r for r in self.results if not r.success]),
            'total_execution_time': sum(r.duration_seconds for r in self.results),
            'sub_pipeline_results': [
                {
                    'name': r.sub_pipeline_name,
                    'status': r.status.value,
                    'success': r.success,
                    'execution_time': r.duration_seconds,
                    'error_message': r.error_message
                }
                for r in self.results
            ]
        }

# Convenience function for direct execution
async def execute_pre_training_pipeline(config: SubPipelineConfig) -> Dict[str, Any]:
    """
    Execute the pre-training pipeline with the given configuration.

    Args:
        config: Configuration for pipeline execution

    Returns:
        Dictionary containing execution results
    """
    pipeline = PreTrainingSubPipeline()
    return await pipeline.execute_pipeline(config)
# LEGACY METHOD REMOVED - _execute_optimized_lookback_generation (lines 5071-5204)
# LEGACY METHOD REMOVED - _execute_final_feature_selection (lines 5205-end)
