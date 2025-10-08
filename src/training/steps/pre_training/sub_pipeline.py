"""
Pre-Training Sub-Pipeline - Feature Engineering Steps

This module provides the pre-training sub-pipeline with the 4 feature engineering steps
that were moved from market_analysis:

1. multi_horizon_profit_labeler - Apply multi-horizon profit labeling
2. feature_lookback_optimization - Optimize feature lookback periods
3. interactive_feature_generation - End-to-end interactive feature generation with comprehensive approach
4. final_feature_selection - Final multi-stage feature selection (120→100→80→60)

Each step can receive a timeframe parameter, with default 15m.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple, TypedDict
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

DEFAULT_STEP_TIME_BUDGETS: Dict[str, float] = {
    'multi_horizon_profit_labeler': 600.0,
    'feature_lookback_optimization': 900.0,
    'interactive_feature_generation': 1200.0,
    'final_feature_selection': 600.0,
}


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
    'multi_horizon_profit_labeler': StepSpec(
        name='multi_horizon_profit_labeler',
        component_key='multi_horizon_profit_labeler',
        executor_method='_execute_multi_horizon_profit_labeler',
        display_name='Multi-horizon labeling',
        description='Apply multi-horizon profit labeling to market data.',
        order=10,
    ),
    'feature_lookback_optimization': StepSpec(
        name='feature_lookback_optimization',
        component_key='feature_lookback_optimization',
        executor_method='_execute_feature_lookback_optimization',
        display_name='Feature optimization',
        description='Optimize feature lookback periods using modular optimization.',
        order=20,
    ),
    'optimized_lookback_generation': StepSpec(
        name='optimized_lookback_generation',
        component_key='optimized_lookback_generation',
        executor_method='_execute_optimized_lookback_generation',
        display_name='Optimized lookback generation',
        description='Generate optimized lookback matrices with hardware acceleration.',
        order=30,
        include_in_default_sequence=False,
    ),
    'interactive_feature_generation': StepSpec(
        name='interactive_feature_generation',
        component_key='interactive_feature_generation',
        executor_method='_execute_interactive_feature_generation',
        display_name='Interactive feature generation',
        description='Produce interactive roadmap features with analyst oversight.',
        order=40,
    ),
    'final_feature_selection': StepSpec(
        name='final_feature_selection',
        component_key='final_feature_selection',
        executor_method='_execute_final_feature_selection',
        display_name='Final feature selection',
        description='Perform the staged final feature selection.',
        order=50,
    ),
}

try:  # pragma: no cover - platform specific import
    import resource
except ImportError:  # pragma: no cover
    resource = None


from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_error, tprint_warning
from src.utils.enhanced_artifact_manager import get_artifact_manager
from src.utils.version_manager import get_version_manager
from src.utils.random_seeding import SeededRNGs, seed_rngs
from src.utils.tprint import tprint, tprint_error, tprint_warning
from .logging_utils import (
    PreTrainingEventLogger,
    StepLogContext,
    configure_pre_training_logging,
)

# Import component system
from .components import ComponentFactory, ComponentConfig
from .metrics_sink import MetricsSink, MetricsSinkConfig
from src.training.config.data_locator import DataLocator, DataLocatorConfig, LocatorPaths
from src.training.steps.pre_training.validation.data_contracts import (
    DataContractValidationError,
    validate_feature_artifact,
    validate_multi_horizon_labeling_result,
    validate_selection_artifact,
)

logger = system_logger.getChild('PreTrainingSubPipeline')


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

    * ``multi_horizon_profit_labeler``
        - ``multi_horizon_labeling_result``: Validated labeling payload.
        - ``labeling_report``: Structured diagnostic report.
        - ``standardized_output``: Normalised label view for downstream steps.
        - ``validated_schemas``: Schema metadata applied during validation.
    * ``feature_lookback_optimization``
        - ``feature_lookback_optimization_result``: Optimisation payload.
        - ``feature_lookback_optimization_summary``: Human readable summary.
        - ``validated_schemas``: Schema metadata for optimisation outputs.
    * ``interactive_feature_generation``
        - ``interactive_feature_generation_result``: Feature catalogue.
        - ``stage_results``: Intermediate stage diagnostics.
        - ``performance_metrics``: Recorded performance metrics.
        - ``artifacts``: Auxiliary artifact bundle emitted by the orchestrator.
        - ``validated_schemas``: Schema metadata for generated features.
    * ``optimized_lookback_generation``
        - ``optimized_lookback_results``: Lookback optimisation payload.
        - ``feature_interaction_matrix``: Numpy matrix of generated features.
        - ``feature_names``: Names for ``feature_interaction_matrix`` columns.
        - ``optimization_metrics``: Performance metrics for the optimisation.
        - ``hardware_utilization_report``: Hardware utilisation diagnostics.
        - ``feature_generation_metadata``: Summary metadata for generation.
    * ``final_feature_selection``
        - ``final_feature_selection_result``: Final selection manifest.
        - ``validated_schemas``: Schema metadata for selection outputs.

    Additional non-artifact keys (e.g. ``random_seed`` or
    ``regime_data_splitting_result``) are written directly by the pipeline and
    remain unconstrained. Any unexpected artifact keys cause an
    :class:`UnexpectedArtifactKeyError` to be raised so upstream bugs are
    surfaced early.
    """

    #: Allowed artifact keys per pipeline step.
    _STEP_ARTIFACT_KEYS: Dict[str, frozenset[str]] = {
        'multi_horizon_profit_labeler': frozenset({
            'multi_horizon_labeling_result',
            'labeling_report',
            'standardized_output',
            'validated_schemas',
        }),
        'feature_lookback_optimization': frozenset({
            'feature_lookback_optimization_result',
            'feature_lookback_optimization_summary',
            'validated_schemas',
        }),
        'interactive_feature_generation': frozenset({
            'interactive_feature_generation_result',
            'stage_results',
            'performance_metrics',
            'artifacts',
            'validated_schemas',
        }),
        'optimized_lookback_generation': frozenset({
            'optimized_lookback_results',
            'feature_interaction_matrix',
            'feature_names',
            'optimization_metrics',
            'hardware_utilization_report',
            'feature_generation_metadata',
            'validated_schemas',
        }),
        'final_feature_selection': frozenset({
            'final_feature_selection_result',
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
    """Configuration for sub-pipeline execution."""

    mode: ExecutionMode = ExecutionMode.FULL
    symbol: str = "ETHUSDT"
    exchange: str = "binance"
    timeframe: str = "1h"  # Default timeframe for pre-training steps (analyst)
    data_dir: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    force_rerun: bool = False
    parallel_processing: bool = True
    max_workers: int = 4
    validation_enabled: bool = True
    monitoring_enabled: bool = True
    fast_mode: bool = False
    skip_next_pipeline: bool = False
    custom_params: Dict[str, Any] = field(default_factory=dict)
    pipeline: Dict[str, Any] = field(default_factory=dict)
    label_imbalance_warning_threshold: float = 0.75
    nan_rate_warning_threshold: float = 0.05
    duplicate_index_warning_threshold: float = 0.02
    metrics_output_path: Optional[str] = None
    metrics_output_format: str = "csv"
    metrics_prometheus_enabled: bool = False
    step_time_budgets: Dict[str, float] = field(default_factory=lambda: DEFAULT_STEP_TIME_BUDGETS.copy())
    market_data_batch_size: Optional[int] = None
    market_data_window_days: Optional[int] = None
    data_locator_config: DataLocatorConfig = field(default_factory=DataLocatorConfig)
    data_locator: Optional[DataLocator] = None
    data_dir_key: str = "market_data"
    cache_dir_key: str = "default"
    artifacts_dir_key: str = "default"
    generated_dir_key: str = "market_analysis"
    outcomes_dir_key: str = "multi_horizon_outcomes"
    final_feature_selection_dir_key: str = "final_feature_selection"
    _path_view: Optional[LocatorPaths] = field(default=None, init=False, repr=False)
    """
    Metrics capture configuration.

    Defaults:
        metrics_output_path: ``artifacts/pre_training_metrics.<format>``
        metrics_output_format: ``csv``
        metrics_prometheus_enabled: ``False``
    """

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
        self.event_logger = PreTrainingEventLogger(configure_pre_training_logging())
        self.results: List[SubPipelineResult] = []
        self._current_pipeline_state: Dict[str, Any] = {}
        self._metrics_sink: Optional[MetricsSink] = None
        self._run_metadata: Dict[str, Any] = {}
        self._data_locator: Optional[DataLocator] = None
        self._seeded_rngs: Optional[SeededRNGs] = None
        self._active_seed: Optional[int] = None
        self._missing_components: Set[str] = set()

        self._refresh_component_registry()

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
        self._extend_messages(errors, component_errors)
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
        if summary:
            tprint_error(f"❌ {summary}")
            self.logger.error(summary)
        else:
            tprint_error(f"❌ {failure.message}")
            self.logger.error(failure.message)

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

        self.logger.info('📁 Effective filesystem configuration:\n%s', summary_json)
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
        Execute the complete pre-training pipeline.

        Args:
            config: Configuration for pipeline execution

        Returns:
            PipelineResultDict containing execution results with typed fields
        """
        seed = self._resolve_random_seed(config)
        self._seeded_rngs = seed_rngs(seed)
        self._active_seed = seed

        run_metadata = self._gather_run_metadata(config, seed)
        self._run_metadata = dict(run_metadata)
        self._current_pipeline_state['random_seed'] = seed
        if self._seeded_rngs is not None:
            self._current_pipeline_state['seeded_rngs'] = self._seeded_rngs
            self._current_pipeline_state['numpy_rng'] = self._seeded_rngs.numpy
            self._current_pipeline_state['python_rng'] = self._seeded_rngs.python

        metadata_block = json.dumps(self._run_metadata, indent=2, sort_keys=True)

        self.logger.info('🚀 Starting Pre-Training Sub-Pipeline execution')
        self.logger.info(f'📊 Symbol: {config.symbol}, Exchange: {config.exchange}')
        self.logger.info(f'⏰ Timeframe: {config.timeframe}, Mode: {config.mode.value}')
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
            # Step 1: Multi-Horizon Profit Labeler
            mh_context = StepLogContext(
                run_id=run_id,
                step='multi_horizon_profit_labeler',
                symbol=config.symbol,
                timeframe=config.timeframe,
            )
            self.event_logger.step_begin(mh_context)
            self.logger.info('🎯 Step 1: Multi-Horizon Profit Labeler')
            mh_result = await self._execute_multi_horizon_profit_labeler(config, self._run_metadata)
            self._capture_step_timing_metrics('multi_horizon_profit_labeler', mh_result, config, results)
            rows_in, rows_out = self._resolve_row_counts(mh_result)
            mh_context.rows_in = rows_in
            mh_context.rows_out = rows_out
            if mh_result.success:
                results['completed_steps'] += 1
            self._record_step_metrics('multi_horizon_profit_labeler', mh_result, results, metrics_sink, step_metric_records)
            mh_duration_ms = self._result_duration_ms(mh_result)
            self._extend_pipeline_collections(results, mh_result)
            if not mh_result.success:
                failure = self._resolve_failure_from_result(
                    'multi_horizon_profit_labeler',
                    mh_result,
                    'Multi-horizon profit labeling failed',
                )
                code_text = f"[{failure.error_code}] " if failure.error_code else ''
                self.logger.error(
                    f"❌ Multi-horizon profit labeling failed: {code_text}{failure.message}"
                )
                self.event_logger.step_end(
                    mh_context,
                    duration_ms=mh_duration_ms,
                    success=False,
                    error=failure.message,
                    extra={'error_code': failure.error_code},
                )
                results['results']['multi_horizon_profit_labeler'] = mh_result.artifacts
                results['error_message'] = failure.message
                results['error_code'] = failure.error_code
                step_failures.append(('multi_horizon_profit_labeler', failure, mh_result))
                if not continue_on_error:
                    return self._apply_failure_to_results(
                        results,
                        failure,
                        start_time,
                        metrics_sink,
                        step_metric_records,
                        config,
                    )
                tprint_warning(
                    "⚠️ Continue-on-error enabled; proceeding after multi_horizon_profit_labeler failure"
                )
                self.logger.warning(
                    "Continue-on-error enabled; proceeding after multi_horizon_profit_labeler failure",
                )
            if mh_result.success:
                # Validate artifacts before updating state
                if 'multi_horizon_labeling_result' in mh_result.artifacts:
                    labeled_data = mh_result.artifacts.get('multi_horizon_labeling_result', {}).get('labeled_data', pd.DataFrame())
                    if isinstance(labeled_data, pd.DataFrame) and not labeled_data.empty:
                        self.logger.info(f"✅ Multi-horizon profit labeling completed for {config.symbol}")
                        self.logger.info(f"   → Labels generated: {len(labeled_data.columns)} columns")
                        results['results']['multi_horizon_profit_labeler'] = mh_result.artifacts
                        self._current_pipeline_state.update(mh_result.artifacts)
                    else:
                        message = "Multi-horizon labeling artifact validation failed"
                        failure = self._create_failure(
                            'multi_horizon_profit_labeler',
                            mh_result.error_code or self._default_step_error_code('multi_horizon_profit_labeler'),
                            message,
                            context={'reason': 'empty_or_invalid_labeled_data'},
                        )
                        self.logger.error(f"❌ {message}")

            # Validate artifacts before updating state
            artifacts = mh_result.artifacts or {}

            if 'multi_horizon_labeling_result' in artifacts:
                labeled_data = artifacts.get('multi_horizon_labeling_result', {}).get('labeled_data', pd.DataFrame())
                if isinstance(labeled_data, pd.DataFrame) and not labeled_data.empty:
                    self.logger.info(f"✅ Multi-horizon profit labeling completed for {config.symbol}")
                    self.logger.info(f"   → Labels generated: {len(labeled_data.columns)} columns")
                    try:
                        merged_artifacts = self._current_pipeline_state.merge_step_artifacts(
                            'multi_horizon_profit_labeler',
                            artifacts,
                        )
                    except UnexpectedArtifactKeyError as merge_error:
                        failure = self._create_failure(
                            'multi_horizon_profit_labeler',
                            f"{self._default_step_error_code('multi_horizon_profit_labeler')}_SCHEMA",
                            str(merge_error),
                            context={'unexpected_keys': merge_error.keys},
                        )
                        self.logger.error(f"❌ {merge_error}")
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

                    results['results']['multi_horizon_profit_labeler'] = merged_artifacts
                else:
                    message = "Missing multi_horizon_labeling_result artifact"
                    failure = self._create_failure(
                        'multi_horizon_profit_labeler',
                        mh_result.error_code or self._default_step_error_code('multi_horizon_profit_labeler'),
                        message,
                        context={'reason': 'missing_artifact'},
                    )
                    self.logger.error(f"❌ {message}")
                    self.event_logger.step_end(
                        mh_context,
                        duration_ms=mh_duration_ms,
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

            if mh_result.success:
                self.event_logger.step_end(
                    mh_context,
                    duration_ms=mh_duration_ms,
                    success=True,
                    extra={'artifact_keys': sorted(mh_result.artifacts.keys())},
                )

            # Step 2: Feature Lookback Optimization
            flo_context = StepLogContext(
                run_id=run_id,
                step='feature_lookback_optimization',
                symbol=config.symbol,
                timeframe=config.timeframe,
            )
            self.event_logger.step_begin(flo_context)
            self.logger.info('⚙️ Step 2: Feature Lookback Optimization')
            flo_result = await self._execute_feature_lookback_optimization(config, self._run_metadata)
            self._capture_step_timing_metrics('feature_lookback_optimization', flo_result, config, results)
            rows_in, rows_out = self._resolve_row_counts(flo_result)
            flo_context.rows_in = rows_in
            flo_context.rows_out = rows_out
            if flo_result.success:
                results['completed_steps'] += 1
            self._record_step_metrics('feature_lookback_optimization', flo_result, results, metrics_sink, step_metric_records)
            flo_duration_ms = self._result_duration_ms(flo_result)
            self._extend_pipeline_collections(results, flo_result)
            if not flo_result.success:
                failure = self._resolve_failure_from_result(
                    'feature_lookback_optimization',
                    flo_result,
                    'Feature lookback optimization failed',
                )
                code_text = f"[{failure.error_code}] " if failure.error_code else ''
                self.logger.error(
                    f"❌ Feature lookback optimization failed: {code_text}{failure.message}"
                )
                self.event_logger.step_end(
                    flo_context,
                    duration_ms=flo_duration_ms,
                    success=False,
                    error=failure.message,
                    extra={'error_code': failure.error_code},
                )
                results['results']['feature_lookback_optimization'] = flo_result.artifacts
                results['error_message'] = failure.message
                results['error_code'] = failure.error_code
                step_failures.append(('feature_lookback_optimization', failure, flo_result))
                if not continue_on_error:
                return self._apply_failure_to_results(
                    results,
                    failure,
                    start_time,
                    metrics_sink,
                    step_metric_records,
                    config,
                )

            flo_artifacts = flo_result.artifacts or {}

            # Validate artifacts before updating state
            if 'feature_lookback_optimization_result' in flo_artifacts:
                optimized_features = flo_artifacts.get('feature_lookback_optimization_result', {}).get('optimized_features', {})
                self.logger.info(f"✅ Feature lookback optimization completed for {config.symbol}")
                self.logger.info(f"   → Features optimized: {len(optimized_features)}")
                try:
                    merged_flo_artifacts = self._current_pipeline_state.merge_step_artifacts(
                        'feature_lookback_optimization',
                        flo_artifacts,
                    )
                except UnexpectedArtifactKeyError as merge_error:
                    failure = self._create_failure(
                        'feature_lookback_optimization',
                        f"{self._default_step_error_code('feature_lookback_optimization')}_SCHEMA",
                        str(merge_error),
                        context={'unexpected_keys': merge_error.keys},
                    )
                    self.logger.error(f"❌ {merge_error}")
                    self.event_logger.step_end(
                        flo_context,
                        duration_ms=flo_duration_ms,
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
                tprint_warning(
                    "⚠️ Continue-on-error enabled; proceeding after feature_lookback_optimization failure"
                )
                self.logger.warning(
                    "Continue-on-error enabled; proceeding after feature_lookback_optimization failure",
                )
            if flo_result.success:
                # Validate artifacts before updating state
                if 'feature_lookback_optimization_result' in flo_result.artifacts:
                    optimized_features = flo_result.artifacts.get('feature_lookback_optimization_result', {}).get('optimized_features', {})
                    self.logger.info(f"✅ Feature lookback optimization completed for {config.symbol}")
                    self.logger.info(f"   → Features optimized: {len(optimized_features)}")
                    results['results']['feature_lookback_optimization'] = flo_result.artifacts
                    self._current_pipeline_state.update(flo_result.artifacts)
                else:
                    self.logger.warning("⚠️ Feature lookback optimization completed but artifact structure unexpected")
                    results['results']['feature_lookback_optimization'] = flo_result.artifacts
                    self._current_pipeline_state.update(flo_result.artifacts)

                results['results']['feature_lookback_optimization'] = merged_flo_artifacts
            elif flo_artifacts:
                self.logger.warning("⚠️ Feature lookback optimization completed but artifact structure unexpected")
                try:
                    merged_flo_artifacts = self._current_pipeline_state.merge_step_artifacts(
                        'feature_lookback_optimization',
                        flo_artifacts,
                    )
                except UnexpectedArtifactKeyError as merge_error:
                    failure = self._create_failure(
                        'feature_lookback_optimization',
                        f"{self._default_step_error_code('feature_lookback_optimization')}_SCHEMA",
                        str(merge_error),
                        context={'unexpected_keys': merge_error.keys},
                    )
                    self.logger.error(f"❌ {merge_error}")
                    self.event_logger.step_end(
                        flo_context,
                        duration_ms=flo_duration_ms,
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

                results['results']['feature_lookback_optimization'] = merged_flo_artifacts

                self.event_logger.step_end(
                    flo_context,
                    duration_ms=flo_duration_ms,
                    success=True,
                    extra={'artifact_keys': sorted(flo_result.artifacts.keys())},
                )

            # Step 3: Interactive Feature Generation
            interactive_context = StepLogContext(
                run_id=run_id,
                step='interactive_feature_generation',
                symbol=config.symbol,
                timeframe=config.timeframe,
            )
            self.event_logger.step_begin(interactive_context)
            self.logger.info('🔧 Step 3: Interactive Feature Generation')
            interactive_result = await self._execute_interactive_feature_generation(config, self._run_metadata)
            self._capture_step_timing_metrics('interactive_feature_generation', interactive_result, config, results)
            rows_in, rows_out = self._resolve_row_counts(interactive_result)
            interactive_context.rows_in = rows_in
            interactive_context.rows_out = rows_out
            if interactive_result.success:
                results['completed_steps'] += 1
            self._record_step_metrics('interactive_feature_generation', interactive_result, results, metrics_sink, step_metric_records)
            interactive_duration_ms = self._result_duration_ms(interactive_result)
            self._extend_pipeline_collections(results, interactive_result)
            if not interactive_result.success:
                failure = self._resolve_failure_from_result(
                    'interactive_feature_generation',
                    interactive_result,
                    'Interactive feature generation failed',
                )
                code_text = f"[{failure.error_code}] " if failure.error_code else ''
                self.logger.error(
                    f"❌ Interactive feature generation failed: {code_text}{failure.message}"
                )
                self.event_logger.step_end(
                    interactive_context,
                    duration_ms=interactive_duration_ms,
                    success=False,
                    error=failure.message,
                    extra={'error_code': failure.error_code},
                )
                results['results']['interactive_feature_generation'] = interactive_result.artifacts
                results['error_message'] = failure.message
                results['error_code'] = failure.error_code
                step_failures.append(('interactive_feature_generation', failure, interactive_result))
                if not continue_on_error:
                return self._apply_failure_to_results(
                    results,
                    failure,
                    start_time,
                    metrics_sink,
                    step_metric_records,
                    config,
                )

            interactive_artifacts = interactive_result.artifacts or {}

            # Validate artifacts before updating state
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
                tprint_warning(
                    "⚠️ Continue-on-error enabled; proceeding after interactive_feature_generation failure"
                )
                self.logger.warning(
                    "Continue-on-error enabled; proceeding after interactive_feature_generation failure",
                )
            if interactive_result.success:
                # Validate artifacts before updating state
                if 'interactive_feature_generation_result' in interactive_result.artifacts:
                    features = interactive_result.artifacts.get('interactive_feature_generation_result', {}).get('features', {})
                    self.logger.info(f"✅ Interactive feature generation completed for {config.symbol}")
                    self.logger.info(f"   → Features generated: {len(features)}")
                    results['results']['interactive_feature_generation'] = interactive_result.artifacts
                    self._current_pipeline_state.update(interactive_result.artifacts)
                else:
                    self.logger.warning("⚠️ Interactive feature generation completed but artifact structure unexpected")
                    results['results']['interactive_feature_generation'] = interactive_result.artifacts
                    self._current_pipeline_state.update(interactive_result.artifacts)

                results['results']['interactive_feature_generation'] = merged_interactive_artifacts
            elif interactive_artifacts:
                self.logger.warning("⚠️ Interactive feature generation completed but artifact structure unexpected")
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

                results['results']['interactive_feature_generation'] = merged_interactive_artifacts

                self.event_logger.step_end(
                    interactive_context,
                    duration_ms=interactive_duration_ms,
                    success=True,
                    extra={'artifact_keys': sorted(interactive_result.artifacts.keys())},
                )

            # Step 4: Final Feature Selection
            ffs_context = StepLogContext(
                run_id=run_id,
                step='final_feature_selection',
                symbol=config.symbol,
                timeframe=config.timeframe,
            )
            self.event_logger.step_begin(ffs_context)
            self.logger.info('🎯 Step 4: Final Feature Selection')
            ffs_result = await self._execute_final_feature_selection(config, self._run_metadata)
            self._capture_step_timing_metrics('final_feature_selection', ffs_result, config, results)
            rows_in, rows_out = self._resolve_row_counts(ffs_result)
            ffs_context.rows_in = rows_in
            ffs_context.rows_out = rows_out
            if ffs_result.success:
                results['completed_steps'] += 1
            self._record_step_metrics('final_feature_selection', ffs_result, results, metrics_sink, step_metric_records)
            ffs_duration_ms = self._result_duration_ms(ffs_result)
            self._extend_pipeline_collections(results, ffs_result)
            if not ffs_result.success:
                failure = self._resolve_failure_from_result(
                    'final_feature_selection',
                    ffs_result,
                    'Final feature selection failed',
                )
                code_text = f"[{failure.error_code}] " if failure.error_code else ''
                self.logger.error(
                    f"❌ Final feature selection failed: {code_text}{failure.message}"
                )
                self.event_logger.step_end(
                    ffs_context,
                    duration_ms=ffs_duration_ms,
                    success=False,
                    error=failure.message,
                    extra={'error_code': failure.error_code},
                )
                results['results']['final_feature_selection'] = ffs_result.artifacts
                results['error_message'] = failure.message
                results['error_code'] = failure.error_code
                step_failures.append(('final_feature_selection', failure, ffs_result))
                if not continue_on_error:
                    return self._apply_failure_to_results(
                        results,
                        failure,
                        start_time,
                        metrics_sink,
                        step_metric_records,
                        config,
                    )
                tprint_warning(
                    "⚠️ Continue-on-error enabled; proceeding after final_feature_selection failure"
                )
                self.logger.warning(
                    "Continue-on-error enabled; proceeding after final_feature_selection failure",
                )
            if ffs_result.success:
                # Validate artifacts before updating state
                if 'final_feature_selection_result' in ffs_result.artifacts:
                    selected_features = ffs_result.artifacts.get('final_feature_selection_result', {}).get('selected_features', [])
                    self.logger.info(f"✅ Final feature selection completed for {config.symbol}")
                    self.logger.info(f"   → Final features: {len(selected_features)}")
                    results['results']['final_feature_selection'] = ffs_result.artifacts
                    self._current_pipeline_state.update(ffs_result.artifacts)
                else:
                    self.logger.warning("⚠️ Final feature selection completed but artifact structure unexpected")
                    results['results']['final_feature_selection'] = ffs_result.artifacts
                    self._current_pipeline_state.update(ffs_result.artifacts)

                self.event_logger.step_end(
                    ffs_context,
                    duration_ms=ffs_duration_ms,
                    success=True,
                    extra={'artifact_keys': sorted(ffs_result.artifacts.keys())},
                )

            if step_failures:
                primary_failure = step_failures[0][1]
                return self._apply_failure_to_results(
                    results,
                    primary_failure,
                    start_time,
                    metrics_sink,
                    step_metric_records,
                    config,
                )

            ffs_artifacts = ffs_result.artifacts or {}

            # Validate artifacts before updating state
            if 'final_feature_selection_result' in ffs_artifacts:
                selected_features = ffs_artifacts.get('final_feature_selection_result', {}).get('selected_features', [])
                self.logger.info(f"✅ Final feature selection completed for {config.symbol}")
                self.logger.info(f"   → Final features: {len(selected_features)}")
                try:
                    merged_ffs_artifacts = self._current_pipeline_state.merge_step_artifacts(
                        'final_feature_selection',
                        ffs_artifacts,
                    )
                except UnexpectedArtifactKeyError as merge_error:
                    failure = self._create_failure(
                        'final_feature_selection',
                        f"{self._default_step_error_code('final_feature_selection')}_SCHEMA",
                        str(merge_error),
                        context={'unexpected_keys': merge_error.keys},
                    )
                    self.logger.error(f"❌ {merge_error}")
                    self.event_logger.step_end(
                        ffs_context,
                        duration_ms=ffs_duration_ms,
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

                results['results']['final_feature_selection'] = merged_ffs_artifacts
            elif ffs_artifacts:
                self.logger.warning("⚠️ Final feature selection completed but artifact structure unexpected")
                try:
                    merged_ffs_artifacts = self._current_pipeline_state.merge_step_artifacts(
                        'final_feature_selection',
                        ffs_artifacts,
                    )
                except UnexpectedArtifactKeyError as merge_error:
                    failure = self._create_failure(
                        'final_feature_selection',
                        f"{self._default_step_error_code('final_feature_selection')}_SCHEMA",
                        str(merge_error),
                        context={'unexpected_keys': merge_error.keys},
                    )
                    self.logger.error(f"❌ {merge_error}")
                    self.event_logger.step_end(
                        ffs_context,
                        duration_ms=ffs_duration_ms,
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

                results['results']['final_feature_selection'] = merged_ffs_artifacts

            self.event_logger.step_end(
                ffs_context,
                duration_ms=ffs_duration_ms,
                success=True,
                extra={'artifact_keys': sorted(ffs_result.artifacts.keys())},
            )

            # Success
            end_time = datetime.now()
            results['success'] = True
            results['execution_time'] = (end_time - start_time).total_seconds()
            results['completed_steps'] = sequence_step_count

            end_metadata = dict(self._run_metadata)
            end_metadata['end_time_utc'] = datetime.utcnow().isoformat() + 'Z'
            end_metadata['duration_seconds'] = results['execution_time']
            self._run_metadata = end_metadata
            completion_block = json.dumps(self._run_metadata, indent=2, sort_keys=True)

            pipeline_duration_ms = results['execution_time'] * 1000.0
            self.logger.info(f"🎉 Pre-Training Sub-Pipeline execution completed successfully for {config.symbol}")
            self.logger.info(f"⏱️ Total execution time: {results['execution_time']:.2f} seconds")
            self.logger.info(f"🧾 Run metadata summary:\n{completion_block}")
            self.event_logger.pipeline_end(
                run_id=run_id,
                symbol=config.symbol,
                timeframe=config.timeframe,
                mode=config.mode.value,
                success=True,
                duration_ms=pipeline_duration_ms,
                completed_steps=results['completed_steps'],
                total_steps=results['total_steps'],
                metadata=dict(self._run_metadata),
            )


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
            self.logger.error(
                f"❌ Pre-Training Sub-Pipeline failed: [{failure.error_code}] {message}"
            )
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
        elif config.metrics_output_path is None:
            extension = 'jsonl' if config.metrics_output_format.lower() == 'jsonl' else 'csv'
            locator = self._data_locator or self._resolve_data_locator(config)
            base_dir = locator.artifacts_path(
                config.artifacts_dir_key,
                ensure_exists=True,
            )
            output_path = base_dir / f'pre_training_metrics.{extension}'

        if output_path is None and not config.metrics_prometheus_enabled:
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
            pass

        tprint("📈 Step duration summary:")
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
                context={
                    'run_id': self._run_metadata.get('run_id'),
                    'step': f'pipeline.summary.{step_name}',
                    'symbol': self._run_metadata.get('symbol'),
                    'timeframe': self._run_metadata.get('timeframe'),
                    'duration_seconds': duration,
                    'budget_seconds': budget,
                    'over_budget': over_budget,
                    'over_budget_seconds': over_budget_seconds,
                },
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
        Execute the pre-training pipeline with backward compatible interface.

        Args:
            training_input: Input data for the pipeline
            pipeline_state: Current pipeline state

        Returns:
            Dictionary containing execution results
        """
        # Extract configuration from pipeline state
        locator = pipeline_state.get('data_locator')
        data_dir_key = pipeline_state.get('data_dir_key', 'market_data')
        data_dir = pipeline_state.get('data_dir')
        if data_dir is None and isinstance(locator, DataLocator):
            data_dir = str(locator.data_path(data_dir_key))

        config = SubPipelineConfig(
            symbol=pipeline_state.get('symbol', 'ETHUSDT'),
            exchange=pipeline_state.get('exchange', 'binance'),
            timeframe=pipeline_state.get('timeframe', '1h'),  # Default 1h for pre-training (analyst)
            data_dir=data_dir,
            mode=ExecutionMode.FULL,  # Default to full mode
            custom_params=pipeline_state.get('custom_params', {}),
            data_locator=locator if isinstance(locator, DataLocator) else None,
            data_dir_key=data_dir_key,
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
        })

        pipeline_state.update({k: v for k, v in self._current_pipeline_state.items() if k not in pipeline_state})
        if self._seeded_rngs is not None:
            pipeline_state['random_seed'] = self._seeded_rngs.seed
            pipeline_state['python_rng'] = self._seeded_rngs.python
            pipeline_state['numpy_rng'] = self._seeded_rngs.numpy
            pipeline_state['seeded_rngs'] = self._seeded_rngs

        regime_cache_path = config.custom_params.get('regime_cache_path') if config.custom_params else None
        if not regime_cache_path:
            data_cache_dir = config.custom_params.get('data_cache_dir') if config.custom_params else None
            if data_cache_dir:
                regime_cache_path = str((Path(data_cache_dir).expanduser() / 'nas_tas_clustering').resolve(strict=False))

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
        if self._active_seed is not None:
            params.setdefault('random_seed', self._active_seed)
        params.setdefault('quality_thresholds', self._get_quality_thresholds(config))
        if config.market_data_batch_size is not None:
            params.setdefault('market_data_batch_size', config.market_data_batch_size)
        if config.market_data_window_days is not None:
            params.setdefault('market_data_window_days', config.market_data_window_days)
        return params

    def _prepare_interactive_training_input(
        self,
        pipeline_state: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """Prepare the training input dictionary for interaction feature generation."""

        mh_result = pipeline_state.get('multi_horizon_labeling_result')
        if mh_result is None:
            mh_result = self._current_pipeline_state.get('multi_horizon_labeling_result', {})

        if not mh_result:
            raise ValueError("Multi-horizon labeling result is required for interactive feature generation")

        market_data_batches = mh_result.get('market_data_batches')
        market_data = mh_result.get('market_data')

        if market_data is None and market_data_batches:
            market_data = pd.concat(market_data_batches, axis=0).sort_index()

        if market_data is None:
            raise ValueError("Market data is missing from multi-horizon labeling result")

        labels_df = mh_result.get('labeled_data')
        if labels_df is None or (isinstance(labels_df, pd.DataFrame) and labels_df.empty):
            labels_df = mh_result.get('labels')
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
            except (TypeError, ValueError):
                pass
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
            if df is None or df.empty:
                return
            df_id = id(df)
            if df_id in visited_frames:
                metrics[dataset_name] = visited_frames[df_id]
                return

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

        def traverse(prefix: str, value: Any) -> None:
            if isinstance(value, pd.DataFrame):
                handle_dataframe(prefix, value)
            elif isinstance(value, dict):
                for key, nested_value in value.items():
                    nested_prefix = f"{prefix}.{key}" if prefix else key
                    traverse(nested_prefix, nested_value)

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
                f"⚠️ [{component_name}] {dataset_name} duplicate index share {duplicate_share:.2%} exceeds threshold {thresholds['duplicate_index']:.2%}"
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
                    f"⚠️ [{component_name}] {dataset_name}.{column} dominant label share {dominant_share:.2%} exceeds threshold {thresholds['label_imbalance']:.2%}"
                )

        if column_metrics:
            dataset_metrics['label_balance'] = {
                'columns': column_metrics,
                'max_dominant_share': max_dominant_share,
                'max_dominant_column': max_dominant_column,
            }

        return dataset_metrics, alerts

    async def _execute_multi_horizon_profit_labeler(
        self,
        config: SubPipelineConfig,
        run_metadata: Dict[str, Any],
    ) -> SubPipelineResult:
        """Execute multi-horizon profit labeler with timeframe support."""
        result = SubPipelineResult(
            sub_pipeline_name='multi_horizon_profit_labeler',
            status=SubPipelineStatus.RUNNING,
            start_time=datetime.now()
        )
        result.error_code = self._default_step_error_code('multi_horizon_profit_labeler')

        try:
            custom_params = config.custom_params or {}
            precomputed_result = custom_params.get('precomputed_labeling_result')

            if precomputed_result:
                self.logger.info('📥 Using precomputed entry labeling result for tactician pipeline')
                try:
                    merged_artifacts = self._current_pipeline_state.merge_step_artifacts(
                        'multi_horizon_profit_labeler',
                        precomputed_result,
                    )
                except UnexpectedArtifactKeyError as merge_error:
                    result.status = SubPipelineStatus.FAILED
                    result.success = False
                    result.end_time = datetime.now()
                    result.duration_seconds = (result.end_time - result.start_time).total_seconds()
                    result.error_message = str(merge_error)
                    result.error_code = (
                        f"{self._default_step_error_code('multi_horizon_profit_labeler')}_SCHEMA"
                    )
                    result.failure = self._create_failure(
                        'multi_horizon_profit_labeler',
                        result.error_code,
                        result.error_message,
                        context={'unexpected_keys': merge_error.keys},
                    )
                    return result

                result.status = SubPipelineStatus.COMPLETED
                result.success = True
                result.end_time = datetime.now()
                result.duration_seconds = (result.end_time - result.start_time).total_seconds()
                result.artifacts = merged_artifacts
                base_metadata = self._merge_run_metadata({
                    'component_type': 'multi_horizon_profit_labeler',
                    'source': 'precomputed',
                    'labeling_method': precomputed_result.get('multi_horizon_labeling_result', {}).get('method', 'tactician_entry_labeling')
                })
                quality_metrics, quality_alerts = self._analyze_component_quality(
                    'multi_horizon_profit_labeler',
                    precomputed_result,
                    config,
                )
                result.metadata = self._extend_with_quality_metadata(
                    base_metadata,
                    quality_metrics,
                    quality_alerts,
                    config,
                )
                return result

            # Convert config to component config
            component_config = ComponentConfig(
                symbol=config.symbol,
                exchange=config.exchange,
                timeframe=config.timeframe,
                data_dir=config.data_dir,
                custom_params=self._build_component_custom_params(config)
            )

            # Create component using factory
            component = ComponentFactory.create_component('multi_horizon_profit_labeler', component_config)
            if hasattr(component, 'set_run_metadata'):
                component.set_run_metadata(run_metadata)

            # Execute component
            pipeline_state = self._prepare_component_pipeline_state(config)
            training_input = self._prepare_interactive_training_input(pipeline_state)
            component_result = await component.execute(training_input, pipeline_state)
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
                self._default_step_error_code('multi_horizon_profit_labeler'),
            )
            if component_result.success:
                try:
                    artifacts = component_result.artifacts or {}
                    if 'multi_horizon_labeling_result' in artifacts:
                        validated_contract = validate_multi_horizon_labeling_result(
                            artifacts['multi_horizon_labeling_result'],
                            context='sub_pipeline.multi_horizon_profit_labeler',
                        )
                        artifacts['multi_horizon_labeling_result'] = validated_contract
                        result.artifacts = artifacts
                except DataContractValidationError as contract_error:
                    self.event_logger.error(
                        "Contract validation error",
                        context={
                            'run_id': self._run_metadata.get('run_id'),
                            'step': 'multi_horizon_profit_labeler.validation',
                            'symbol': self._run_metadata.get('symbol'),
                            'timeframe': self._run_metadata.get('timeframe'),
                            'error': str(contract_error),
                        },
                    )
                    return self._handle_contract_error(result, 'multi_horizon_profit_labeler', contract_error)

                quality_metrics, quality_alerts = self._analyze_component_quality(
                    'multi_horizon_profit_labeler',
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
                    'component_type': 'multi_horizon_profit_labeler'
                })
                failure_context = {
                    'component_metadata': component_result.metadata,
                    'artifacts_keys': sorted((component_result.artifacts or {}).keys()),
                }
                result.failure = self._create_failure(
                    'multi_horizon_profit_labeler',
                    result.error_code or self._default_step_error_code('multi_horizon_profit_labeler'),
                    result.error_message or 'Multi-horizon profit labeler failed',
                    context=failure_context,
                )
                if result.error_message:
                    self._extend_messages(result.errors, [result.error_message])

        except ImportError as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = f"Missing dependencies: {str(e)}"
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            self.logger.error(f"❌ Multi-horizon profit labeler failed - missing dependencies: {e}")
            result.metadata = self._merge_run_metadata(result.metadata)
            result.error_code = f"{self._default_step_error_code('multi_horizon_profit_labeler')}_IMPORT"
            result.failure = self._create_failure(
                'multi_horizon_profit_labeler',
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
            self.logger.error(f"❌ Multi-horizon profit labeler failed - missing files: {e}")
            result.metadata = self._merge_run_metadata(result.metadata)
            result.error_code = f"{self._default_step_error_code('multi_horizon_profit_labeler')}_MISSING_FILE"
            result.failure = self._create_failure(
                'multi_horizon_profit_labeler',
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
            self.logger.error(f"❌ Multi-horizon profit labeler failed with unexpected error: {e}")
            trace = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
            self.logger.error(f"🔍 Error details: {trace}")
            result.metadata = self._merge_run_metadata(result.metadata)
            result.error_code = f"{self._default_step_error_code('multi_horizon_profit_labeler')}_UNEXPECTED"
            result.failure = self._create_failure(
                'multi_horizon_profit_labeler',
                result.error_code,
                result.error_message or 'Multi-horizon profit labeler failed',
                exception=e,
                traceback_str=trace,
            )
            self._extend_messages(result.errors, [result.error_message or str(e)])

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
            # Convert config to component config
            component_config = ComponentConfig(
                symbol=config.symbol,
                exchange=config.exchange,
                timeframe=config.timeframe,
                data_dir=config.data_dir,
                custom_params=self._build_component_custom_params(config)
            )

            # Create component using factory
            component = ComponentFactory.create_component('feature_lookback_optimization', component_config)
            if hasattr(component, 'set_run_metadata'):
                component.set_run_metadata(run_metadata)

            # Execute component
            pipeline_state = self._prepare_component_pipeline_state(config)
            component_result = await component.execute(None, pipeline_state)
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
        result.error_code = self._default_step_error_code('interactive_feature_generation')

        try:
            component_config = ComponentConfig(
                symbol=config.symbol,
                exchange=config.exchange,
                timeframe=config.timeframe,
                data_dir=config.data_dir,
                custom_params=self._build_component_custom_params(config),
            )

            component = ComponentFactory.create_component(
                'interactive_feature_generation',
                component_config,
            )
            if hasattr(component, 'set_run_metadata'):
                component.set_run_metadata(run_metadata)

            # Execute component
            pipeline_state = self._prepare_component_pipeline_state(config)
            component_result = await component.execute(None, pipeline_state)
            component_result.metadata = self._merge_run_metadata(component_result.metadata)

            result.status = SubPipelineStatus.COMPLETED if component_result.success else SubPipelineStatus.FAILED
            result.success = component_result.success
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            result.artifacts = component_result.artifacts
            result.output_files = component_result.output_files
            result.error_message = component_result.error_message
            result.error_code = self._extract_component_error_code(
                component_result,
                self._default_step_error_code('interactive_feature_generation'),
            )
            if component_result.success:
                try:
                    artifacts = component_result.artifacts or {}
                    feature_payload = artifacts.get('interactive_feature_generation_result')
                    if feature_payload:
                        validated_features = validate_feature_artifact(
                            feature_payload,
                            context='sub_pipeline.interactive_feature_generation',
                        )
                        artifacts['interactive_feature_generation_result'] = validated_features
                        result.artifacts = artifacts
                except DataContractValidationError as contract_error:
                    self.event_logger.error(
                        "Contract validation error",
                        context={
                            'run_id': self._run_metadata.get('run_id'),
                            'step': 'interactive_feature_generation.validation',
                            'symbol': self._run_metadata.get('symbol'),
                            'timeframe': self._run_metadata.get('timeframe'),
                            'error': str(contract_error),
                        },
                    )
                    return self._handle_contract_error(result, 'interactive_feature_generation', contract_error)

                quality_metrics, quality_alerts = self._analyze_component_quality(
                    'interactive_feature_generation',
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
                    'interactive_feature_generation',
                    result.error_code or self._default_step_error_code('interactive_feature_generation'),
                    result.error_message or 'Interactive feature generation failed',
                    context=failure_context,
                )
                if result.error_message:
                    self._extend_messages(result.errors, [result.error_message])

        except ValueError as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = str(e)
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            self.logger.error(f"❌ Interactive feature generation unavailable: {e}")
            result.metadata = self._merge_run_metadata(result.metadata)
            result.error_code = f"{self._default_step_error_code('interactive_feature_generation')}_UNAVAILABLE"
            result.failure = self._create_failure(
                'interactive_feature_generation',
                result.error_code,
                result.error_message,
                exception=e,
            )
            self._extend_messages(result.errors, [result.error_message])
        except ImportError as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = f"Missing dependencies: {str(e)}"
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            self.logger.error(f"❌ Interactive feature generation failed - missing dependencies: {e}")
            result.metadata = self._merge_run_metadata(result.metadata)
            result.error_code = f"{self._default_step_error_code('interactive_feature_generation')}_IMPORT"
            result.failure = self._create_failure(
                'interactive_feature_generation',
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
            self.logger.error(f"❌ Interactive feature generation failed - missing files: {e}")
            result.metadata = self._merge_run_metadata(result.metadata)
            result.error_code = f"{self._default_step_error_code('interactive_feature_generation')}_MISSING_FILE"
            result.failure = self._create_failure(
                'interactive_feature_generation',
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
            self.logger.error(f"❌ Interactive feature generation failed with unexpected error: {e}")
            trace = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
            self.logger.error(f"🔍 Error details: {trace}")
            result.metadata = self._merge_run_metadata(result.metadata)
            result.error_code = f"{self._default_step_error_code('interactive_feature_generation')}_UNEXPECTED"
            result.failure = self._create_failure(
                'interactive_feature_generation',
                result.error_code,
                result.error_message or 'Interactive feature generation failed',
                exception=e,
                traceback_str=trace,
            )
            self._extend_messages(result.errors, [result.error_message or str(e)])

        return result

    async def _execute_optimized_lookback_generation(
        self,
        config: SubPipelineConfig,
        run_metadata: Dict[str, Any],
    ) -> SubPipelineResult:
        """Execute optimized lookback generation with matrix operations and hardware acceleration."""
        result = SubPipelineResult(
            sub_pipeline_name='optimized_lookback_generation',
            status=SubPipelineStatus.RUNNING,
            start_time=datetime.now()
        )
        result.error_code = self._default_step_error_code('optimized_lookback_generation')

        try:
            # Convert config to component config
            component_config = ComponentConfig(
                symbol=config.symbol,
                exchange=config.exchange,
                timeframe=config.timeframe,
                data_dir=config.data_dir,
                custom_params=self._build_component_custom_params(config)
            )

            # Create component using factory
            component = ComponentFactory.create_component('optimized_lookback_generation', component_config)
            if hasattr(component, 'set_run_metadata'):
                component.set_run_metadata(run_metadata)

            # Execute component
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
                self._default_step_error_code('optimized_lookback_generation'),
            )
            if component_result.success:
                quality_metrics, quality_alerts = self._analyze_component_quality(
                    'optimized_lookback_generation',
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
                    'optimized_lookback_generation',
                    result.error_code or self._default_step_error_code('optimized_lookback_generation'),
                    result.error_message or 'Optimized lookback generation failed',
                    context=failure_context,
                )
                if result.error_message:
                    self._extend_messages(result.errors, [result.error_message])

        except ImportError as e:
            result.status = SubPipelineStatus.FAILED
            result.error_message = f"Missing dependencies: {str(e)}"
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            self.logger.error(f"❌ Optimized lookback generation failed - missing dependencies: {e}")
            result.metadata = self._merge_run_metadata(result.metadata)
            result.error_code = f"{self._default_step_error_code('optimized_lookback_generation')}_IMPORT"
            result.failure = self._create_failure(
                'optimized_lookback_generation',
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
            self.logger.error(f"❌ Optimized lookback generation failed - missing files: {e}")
            result.metadata = self._merge_run_metadata(result.metadata)
            result.error_code = f"{self._default_step_error_code('optimized_lookback_generation')}_MISSING_FILE"
            result.failure = self._create_failure(
                'optimized_lookback_generation',
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
            self.logger.error(f"❌ Optimized lookback generation failed with unexpected error: {e}")
            trace = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
            self.logger.error(f"🔍 Error details: {trace}")
            result.metadata = self._merge_run_metadata(result.metadata)
            result.error_code = f"{self._default_step_error_code('optimized_lookback_generation')}_UNEXPECTED"
            result.failure = self._create_failure(
                'optimized_lookback_generation',
                result.error_code,
                result.error_message or 'Optimized lookback generation failed',
                exception=e,
                traceback_str=trace,
            )
            self._extend_messages(result.errors, [result.error_message or str(e)])

        return result

    async def _execute_final_feature_selection(
        self,
        config: SubPipelineConfig,
        run_metadata: Dict[str, Any],
    ) -> SubPipelineResult:
        """Execute final feature selection with timeframe support."""
        result = SubPipelineResult(
            sub_pipeline_name='final_feature_selection',
            status=SubPipelineStatus.RUNNING,
            start_time=datetime.now()
        )
        result.error_code = self._default_step_error_code('final_feature_selection')

        try:
            # Convert config to component config
            component_config = ComponentConfig(
                symbol=config.symbol,
                exchange=config.exchange,
                timeframe=config.timeframe,
                data_dir=config.data_dir,
                custom_params=self._build_component_custom_params(config)
            )

            # Create component using factory
            component = ComponentFactory.create_component('final_feature_selection', component_config)
            if hasattr(component, 'set_run_metadata'):
                component.set_run_metadata(run_metadata)

            # Execute component
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
            result.metadata = self._merge_run_metadata(result.metadata)
            result.error_code = f"{self._default_step_error_code('final_feature_selection')}_UNEXPECTED"
            result.failure = self._create_failure(
                'final_feature_selection',
                result.error_code,
                result.error_message or 'Final feature selection failed',
                exception=e,
                traceback_str=trace,
            )
            self._extend_messages(result.errors, [result.error_message or str(e)])

        return result

    def get_available_sub_pipelines(self) -> List[str]:
        """Get list of available sub-pipelines for pre-training stage."""
        return [spec.name for spec in self._get_ordered_step_specs()]

    async def execute_sub_pipeline(self, sub_pipeline_name: str, config: SubPipelineConfig) -> SubPipelineResult:
        """Execute a specific sub-pipeline."""
        if not self._run_metadata:
            self._run_metadata = self._gather_run_metadata(config)

        self._refresh_component_registry()

        spec = STEP_REGISTRY.get(sub_pipeline_name)
        if spec and spec.component_key in self._missing_components:
            available_components = ComponentFactory.get_available_components()
            message = (
                f"Component '{spec.component_key}' required for '{sub_pipeline_name}' is not registered."
            )
            error_code = f"{self._default_step_error_code(sub_pipeline_name)}_UNAVAILABLE"
            now = datetime.now()
            result = SubPipelineResult(
                sub_pipeline_name=sub_pipeline_name,
                status=SubPipelineStatus.FAILED,
                start_time=now,
                end_time=now,
                duration_seconds=0.0,
                success=False,
                error_message=message,
                error_code=error_code,
            )
            result.metadata = self._merge_run_metadata({'available_components': available_components})
            result.failure = self._create_failure(
                sub_pipeline_name,
                error_code,
                message,
                context={
                    'requested_component': spec.component_key,
                    'available_components': available_components,
                },
            )
            self.event_logger.error(
                message,
                context={
                    'run_id': self._run_metadata.get('run_id'),
                    'step': sub_pipeline_name,
                    'symbol': config.symbol,
                    'timeframe': config.timeframe,
                    'requested_component': spec.component_key,
                    'available_components': available_components,
                },
            )
            return result

        if sub_pipeline_name == 'multi_horizon_profit_labeler':
            return await self._execute_multi_horizon_profit_labeler(config, self._run_metadata)
        elif sub_pipeline_name == 'feature_lookback_optimization':
            return await self._execute_feature_lookback_optimization(config, self._run_metadata)
        elif sub_pipeline_name == 'optimized_lookback_generation':
            return await self._execute_optimized_lookback_generation(config, self._run_metadata)
        elif sub_pipeline_name == 'interactive_feature_generation':
            return await self._execute_interactive_feature_generation(config, self._run_metadata)
        elif sub_pipeline_name == 'final_feature_selection':
            return await self._execute_final_feature_selection(config, self._run_metadata)

        spec = self._get_step_spec(sub_pipeline_name)
        if spec is None:
            available = self.get_available_sub_pipelines()
            self.logger.error(f"❌ Unknown sub-pipeline requested: {sub_pipeline_name}")
            self.logger.info(f"📋 Available sub-pipelines: {available}")
            self.event_logger.error(
                "Unknown sub-pipeline requested",
                context={
                    'run_id': self._run_metadata.get('run_id'),
                    'step': 'pipeline.dispatch',
                    'symbol': config.symbol,
                    'timeframe': config.timeframe,
                    'requested_sub_pipeline': sub_pipeline_name,
                    'available_sub_pipelines': available,
                },
            )

        if not spec.enabled:
            message = (
                f"Sub-pipeline '{sub_pipeline_name}' is currently disabled. "
                f"Reason: {getattr(spec, 'description', 'temporarily unavailable')}"
            )
            self.logger.warning(message)
            self.event_logger.warning(
                message,
                context={
                    'run_id': self._run_metadata.get('run_id'),
                    'step': 'pipeline.dispatch',
                    'symbol': config.symbol,
                    'timeframe': config.timeframe,
                    'requested_sub_pipeline': sub_pipeline_name,
                    'status': 'disabled',
                },
            )
            raise ValueError(message)

        executor = getattr(self, spec.executor_method, None)
        if executor is None:
            message = (
                f"Sub-pipeline '{sub_pipeline_name}' is registered but missing executor "
                f"'{spec.executor_method}'. Implement the executor or disable the step in "
                "STEP_REGISTRY."
            )
            self.logger.error(message)
            self.event_logger.error(
                message,
                context={
                    'run_id': self._run_metadata.get('run_id'),
                    'step': 'pipeline.dispatch',
                    'symbol': config.symbol,
                    'timeframe': config.timeframe,
                    'requested_sub_pipeline': sub_pipeline_name,
                    'missing_executor': spec.executor_method,
                },
            )
            raise RuntimeError(message)

        return await executor(config, self._run_metadata)

    async def execute_sub_pipeline_with_next(self, sub_pipeline_name: str, config: SubPipelineConfig) -> SubPipelineResult:
        """Execute a specific sub-pipeline and automatically trigger subsequent sub-pipelines."""
        # For pre-training, execute the default enabled steps in sequence
        ordered_specs = self._get_ordered_step_specs(sequence_only=True)
        ordered_names = [spec.name for spec in ordered_specs]

        try:
            start_index = ordered_names.index(sub_pipeline_name)
            steps_to_run = ordered_specs[start_index:]
        except ValueError:
            # Step not part of the default sequence; execute it directly
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

            # If this step failed, stop the sequence
            if not result.success:
                self.logger.error(f"❌ Step {step_name} failed, stopping execution sequence")
                break
        
        # Return the first result (the one that was requested)
        return self.results[0] if self.results else None

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
