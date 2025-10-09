"""
Base Component for Pre-Training Pipeline Components.

This module provides the base classes for all pre-training pipeline components.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path

from src.training.common.artifact_persistence import SaveReport, persist_artifacts
from src.training.common.component_result import ComponentError, ComponentResult as _BaseComponentResult
from src.utils.enhanced_artifact_manager import get_artifact_manager
from src.utils.logger import system_logger
from src.utils.version_manager import get_version_manager
from src.utils.tprint import tprint_success
from src.utils.common_operations import safe_dataframe_operation, get_m1_memory_optimizer
from src.utils.common_utilities import CommonUtilities
from src.utils.math_validation import validate_finite
from src.utils.serialization_utils import JSONSerializer, PickleSerializer
from src.utils.data.klines_parquet import KlinesParquetManager
from src.utils.hardware.m1_gpu_utils import M1GPUManager
from src.utils.matrix_operations import (
    get_unified_matrix_operations,
    get_vectorized_processing_core,
    get_batch_matrix_processor,
    safe_matrix_multiply,
    optimize_dataframe,
    matrix_correlation_analysis
)
from ..logging_utils import PreTrainingEventLogger, configure_pre_training_logging
from ..settings import get_pre_training_settings
from .contracts import (
    ArtifactBundle,
    GenericArtifacts,
    PipelineState,
    validate_artifact_bundle,
)

logger = system_logger.getChild('PreTrainingComponent')
component_event_logger = PreTrainingEventLogger(configure_pre_training_logging())

def _default_data_directory() -> str:
    return str(get_pre_training_settings().data_root)


@dataclass
class ComponentConfig:
    """Configuration for pre-training components."""
    symbol: str = "ETHUSDT"
    exchange: str = "binance"
    timeframe: str = "15m"  # Default timeframe for pre-training
    data_dir: str = field(default_factory=_default_data_directory)
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    force_rerun: bool = False
    validation_enabled: bool = True
    monitoring_enabled: bool = True
    fast_mode: bool = False
    custom_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.custom_params is None:
            self.custom_params = {}
        details = {
            'symbol': self.symbol,
            'exchange': self.exchange,
            'timeframe': self.timeframe,
            'force_rerun': self.force_rerun,
            'validation_enabled': self.validation_enabled,
            'monitoring_enabled': self.monitoring_enabled,
            'fast_mode': self.fast_mode,
        }
        logger.debug(
            "🛠️ ComponentConfig initialized",
            extra={'extra_fields': {'event': 'component_config_init', **details}},
        )
        component_event_logger.info(
            "ComponentConfig initialized",
            context={'component': 'ComponentConfig', **details},
        )

class ComponentResult(_BaseComponentResult):
    """Pre-training specific ComponentResult with structured logging."""

    def __init__(
        self,
        success: bool,
        artifacts: Optional[ArtifactBundle] = None,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        execution_time: float = 0.0,
        metrics: Optional[Dict[str, float]] = None,
        warnings: Optional[List[str]] = None,
        errors: Optional[List[ComponentError]] = None,
        error: Optional[Union[Exception, ComponentError]] = None,
        error_message: Optional[str] = None,
    ) -> None:
        bundle = artifacts or GenericArtifacts()
        validated_bundle = validate_artifact_bundle(bundle)
        super().__init__(
            success,
            validated_bundle.as_payload(),
            metadata=metadata,
            execution_time=execution_time,
            metrics=metrics,
            warnings=warnings,
            errors=errors,
            error=error,
            error_message=error_message,
        )
        self.artifact_bundle = validated_bundle
        details = {
            'success': self.success,
            'artifacts_keys': list(self.artifacts.keys()),
            'metadata_keys': list(self.metadata.keys()),
            'metrics_keys': list(self.metrics.keys()),
            'warnings': list(self.warnings),
            'errors': [err.to_dict() for err in self.errors],
            'error': self.error_message,
            'execution_time': self.execution_time,
        }
        logger.debug(
            "📦 ComponentResult initialized",
            extra={'extra_fields': {'event': 'component_result_init', **details}},
        )
        component_event_logger.info(
            "ComponentResult initialized",
            context={'component': 'ComponentResult', **details},
        )

class BasePreTrainingComponent(ABC):
    """
    Base class for all pre-training pipeline components.
    
    Provides common functionality and interface for all components.
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the component."""
        self.config = config or ComponentConfig()
        self.logger = logger.getChild(self.__class__.__name__)
        self.event_logger = PreTrainingEventLogger(configure_pre_training_logging())
        self.artifact_manager = get_artifact_manager()
        self.version_manager = get_version_manager()
        self._run_metadata: Dict[str, Any] = {}

        # Initialize core utility managers
        self.common_utils = CommonUtilities()
        self.memory_optimizer = get_m1_memory_optimizer()
        self.gpu_manager = M1GPUManager()
        self.data_manager = KlinesParquetManager()
        self.json_serializer = JSONSerializer()
        self.pickle_serializer = PickleSerializer()

        # Initialize matrix operations managers
        self.matrix_ops = get_unified_matrix_operations()
        self.vectorized_core = get_vectorized_processing_core()
        self.batch_processor = get_batch_matrix_processor()

        self._log_info(
            f"🚀 Initialized {self.__class__.__name__} with utility and matrix operation managers",
            event='component_initialized',
            utility_managers=['common_utils', 'memory_optimizer', 'gpu_manager', 'data_manager'],
            matrix_managers=['matrix_ops', 'vectorized_core', 'batch_processor']
        )

    def set_run_metadata(self, metadata: Optional[Dict[str, Any]]) -> None:
        """Store run metadata for later use."""
        self._run_metadata = dict(metadata or {})

    def _component_context(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        base = {
            'run_id': self._run_metadata.get('run_id'),
            'step': f'component.{self.__class__.__name__}',
            'component': self.__class__.__name__,
            'symbol': self.config.symbol,
            'exchange': self.config.exchange,
            'timeframe': self.config.timeframe,
        }
        if extra:
            base.update(extra)
        return base

    def _log_info(self, message: str, **context: Any) -> None:
        payload = self._component_context(context)
        self.logger.info(message)
        self.event_logger.info(message, context=payload)

    def _log_warning(self, message: str, **context: Any) -> None:
        payload = self._component_context(context)
        self.logger.warning(message)
        self.event_logger.warning(message, context=payload)

    def _log_error(self, message: str, **context: Any) -> None:
        payload = self._component_context(context)
        self.logger.error(message)
        self.event_logger.error(message, context=payload)

    def _log_debug(self, message: str, **context: Any) -> None:
        payload = self._component_context(context)
        self.logger.debug(message)
        self.event_logger.info(message, context=payload)

    def _log_success(self, message: str, **context: Any) -> None:
        self._log_info(message, **context)

    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        self._log_debug(
            f"🧩 get_required_artifacts called on base class {self.__class__.__name__}",
            event='component_get_required_artifacts',
        )
        # Default implementation - subclasses should override this
        return []

    async def execute(self, data: Any, pipeline_state: PipelineState) -> ComponentResult:
        """Execute the component."""
        self._log_debug(
            f"⚙️ execute called on base class {self.__class__.__name__}",
            data_type=type(data).__name__,
            event='component_execute_called',
        )
        # Default implementation - subclasses should override this
        return ComponentResult(
            success=False,
            error_message=f"Component '{self.__class__.__name__}' does not implement the execute() method. "
                         f"Please implement this method in your component class.",
            execution_time=0.0
        )

    async def save_artifacts(
        self,
        artifacts: Union[ArtifactBundle, Mapping[str, Any]],
        metadata: Mapping[str, Any],
    ) -> SaveReport:
        """Save artifacts persistently."""

        artifact_payload = (
            artifacts.as_payload() if isinstance(artifacts, ArtifactBundle) else dict(artifacts)
        )
        metadata = dict(metadata or {})
        metadata['run_metadata'] = dict(self._run_metadata)

        self._log_info(
            f"💾 Saving {len(artifact_payload)} artifacts for {self.__class__.__name__}",
            event='component_save_artifacts',
            metadata_keys=list(metadata.keys()),
        )
        component_name = self.__class__.__name__
        base_artifact_dir = Path(self.artifact_manager.base_paths["artifacts"]) / component_name

        report = persist_artifacts(
            component_name=component_name,
            artifacts=artifact_payload,
            metadata=metadata,
            base_dir=base_artifact_dir,
            logger=self.logger,
        )

        self._log_success(
            f"✅ Artifacts persisted for {component_name}",
            event='component_artifacts_persisted',
            correlation_id=report.correlation_id,
            artifact_count=len(artifact_payload),
            duration=report.duration,
        )

        return report

    def validate_config(self) -> bool:
        """Validate the component configuration."""
        if not self.config.symbol:
            raise ValueError("Symbol is required")
        if not self.config.exchange:
            raise ValueError("Exchange is required")
        if not self.config.timeframe:
            raise ValueError("Timeframe is required")
        self._log_success(
            f"✅ Configuration validated for {self.__class__.__name__}",
            event='component_config_validated',
        )
        return True

    def safe_dataframe_operation(self, df: pd.DataFrame, operation, *args, **kwargs):
        """Safely perform DataFrame operations with error handling."""
        return safe_dataframe_operation(df, operation, *args, **kwargs)

    def validate_finite_values(self, value, name: str = "value"):
        """Validate that values are finite using math validation utilities."""
        return validate_finite(value, name)

    def serialize_data_json(self, data, filepath: str) -> bool:
        """Serialize data to JSON format."""
        return self.json_serializer.save(data, filepath)

    def deserialize_data_json(self, filepath: str):
        """Deserialize data from JSON format."""
        return self.json_serializer.load(filepath)

    def serialize_data_pickle(self, data, filepath: str) -> bool:
        """Serialize data to pickle format."""
        return self.pickle_serializer.save(data, filepath)

    def deserialize_data_pickle(self, filepath: str):
        """Deserialize data from pickle format."""
        return self.pickle_serializer.load(filepath)

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
                event='component_memory_optimized',
                memory_pressure=self.get_memory_pressure()
            )

    def is_hardware_accelerated(self) -> bool:
        """Check if hardware acceleration is available."""
        return self.gpu_manager.is_m1 if self.gpu_manager else False

    def load_klines_data(self, symbol: str, timeframe: str, start_date=None, end_date=None):
        """Load klines data using the data manager."""
        if self.data_manager:
            return self.data_manager.load_symbol_data(
                symbol, timeframe, start_date, end_date
            )
        return None

    def safe_matrix_multiply(self, A, B):
        """Safely perform matrix multiplication with error handling."""
        return safe_matrix_multiply(A, B)

    def optimize_dataframe_for_matrix_ops(self, df):
        """Optimize DataFrame for matrix operations."""
        return optimize_dataframe(df)

    def compute_matrix_correlation_analysis(self, data):
        """Compute matrix correlation analysis."""
        return matrix_correlation_analysis(data)

    def perform_vectorized_matrix_ops(self, data, operations):
        """Perform vectorized matrix operations using the vectorized core."""
        if self.vectorized_core:
            return self.vectorized_core.optimize_dataframe_for_processing(data)
        return data

    def batch_matrix_operations(self, matrices_a, matrices_b, operation='multiply'):
        """Perform batch matrix operations."""
        if self.batch_processor:
            if operation == 'multiply':
                return self.batch_processor.batch_matrix_multiply(matrices_a, matrices_b)
        return None

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the component."""
        status = {
            'component_name': self.__class__.__name__,
            'config': self.config,
            'required_artifacts': self.get_required_artifacts()
        }
        self._log_info(
            f"📊 Status requested for {self.__class__.__name__}",
            event='component_status_requested',
            required_artifacts=status['required_artifacts'],
        )
        return status
