"""
Base Component for Pre-Training Pipeline Components.

This module provides the base classes for all pre-training pipeline components.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
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
from ..logging_utils import PreTrainingEventLogger, configure_pre_training_logging
from .contracts import (
    ArtifactBundle,
    GenericArtifacts,
    PipelineState,
    validate_artifact_bundle,
)

logger = system_logger.getChild('PreTrainingComponent')
component_event_logger = PreTrainingEventLogger(configure_pre_training_logging())

@dataclass
class ComponentConfig:
    """Configuration for pre-training components."""
    symbol: str = "ETHUSDT"
    exchange: str = "binance"
    timeframe: str = "15m"  # Default timeframe for pre-training
    data_dir: str = "historical_data"
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
        self.artifacts = validated_bundle
        details = {
            'success': self.success,
            'artifacts_keys': list(self.artifacts.keys()),
            'metadata_keys': list(self.metadata.keys()),
            'metrics_keys': list(self.metrics.keys()),
            'warnings': list(self.warnings),
            'errors': [str(err) for err in self.errors],
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
        self._log_info(
            f"🚀 Initialized {self.__class__.__name__}",
            event='component_initialized',
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

    @abstractmethod
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        self._log_debug(
            f"🧩 get_required_artifacts called on base class {self.__class__.__name__}",
            event='component_get_required_artifacts',
        )
        raise NotImplementedError("Subclasses must implement get_required_artifacts")

    @abstractmethod
    async def execute(self, data: Any, pipeline_state: PipelineState) -> ComponentResult:
        """Execute the component."""
        self._log_debug(
            f"⚙️ execute called on base class {self.__class__.__name__}",
            data_type=type(data).__name__,
            event='component_execute_called',
        )
        raise NotImplementedError("Subclasses must implement execute")

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
