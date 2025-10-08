"""
Base Component for Pre-Training Pipeline Components.

This module provides the base classes for all pre-training pipeline components.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.utils.enhanced_artifact_manager import get_artifact_manager
from src.utils.version_manager import get_version_manager
from src.utils.tprint import (
    tprint,
    tprint_error,
    tprint_warning,
    tprint_success,
    tprint_debug,
    tprint_info,
)

logger = system_logger.getChild('PreTrainingComponent')

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
        tprint_debug(
            "🛠️ ComponentConfig initialized",
            {
                'symbol': self.symbol,
                'exchange': self.exchange,
                'timeframe': self.timeframe,
                'force_rerun': self.force_rerun,
                'validation_enabled': self.validation_enabled,
                'monitoring_enabled': self.monitoring_enabled,
                'fast_mode': self.fast_mode
            }
        )

@dataclass
class ComponentResult:
    """Result of component execution."""
    success: bool
    artifacts: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    error_message: Optional[str] = None
    execution_time: float = 0.0

    def __post_init__(self):
        if self.artifacts is None:
            self.artifacts = {}
        if self.metadata is None:
            self.metadata = {}
        tprint_debug(
            "📦 ComponentResult initialized",
            {
                'success': self.success,
                'artifacts_keys': list(self.artifacts.keys()),
                'metadata_keys': list(self.metadata.keys()),
                'error_message': self.error_message,
                'execution_time': self.execution_time
            }
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
        self.artifact_manager = get_artifact_manager()
        self.version_manager = get_version_manager()
        self._current_run_metadata: Dict[str, Any] = {}
        tprint_success(
            f"🚀 Initialized {self.__class__.__name__}",
            {
                'symbol': self.config.symbol,
                'exchange': self.config.exchange,
                'timeframe': self.config.timeframe
            }
        )

    def set_run_metadata(self, metadata: Optional[Dict[str, Any]]) -> None:
        """Attach run-level reproducibility metadata to the component."""
        self._current_run_metadata = metadata or {}

    def get_run_metadata(self) -> Dict[str, Any]:
        """Return a copy of the currently attached run metadata."""
        return dict(self._current_run_metadata)

    @abstractmethod
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        tprint_debug(
            f"🧩 get_required_artifacts called on base class {self.__class__.__name__}"
        )
        raise NotImplementedError("Subclasses must implement get_required_artifacts")

    @abstractmethod
    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """Execute the component."""
        tprint_debug(
            f"⚙️ execute called on base class {self.__class__.__name__}",
            {'data_type': type(data).__name__}
        )
        raise NotImplementedError("Subclasses must implement execute")

    async def save_artifacts(self, artifacts: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, str]:
        """
        Save artifacts persistently.
        
        Args:
            artifacts: Artifacts to save
            metadata: Metadata for the artifacts
            
        Returns:
            Dictionary mapping artifact names to file paths
        """
        run_metadata = dict(self._current_run_metadata)
        metadata_with_run = dict(metadata)
        existing_run_metadata = metadata_with_run.get('run_metadata') if isinstance(metadata_with_run.get('run_metadata'), dict) else {}
        metadata_with_run['run_metadata'] = {**run_metadata, **existing_run_metadata}

        tprint_info(
            f"💾 Saving {len(artifacts)} artifacts for {self.__class__.__name__}",
            {
                'metadata_keys': list(metadata_with_run.keys()),
                'run_metadata': metadata_with_run['run_metadata']
            }
        )
        saved_files = {}

        for artifact_name, artifact_data in artifacts.items():
            # Create artifact metadata
            artifact_metadata = {
                'component': self.__class__.__name__,
                'timestamp': datetime.now().isoformat(),
                'symbol': self.config.symbol,
                'exchange': self.config.exchange,
                'timeframe': self.config.timeframe,
                **metadata_with_run
            }

            artifact_payload = self._attach_run_metadata_to_artifact(
                artifact_data,
                metadata_with_run['run_metadata']
            )

            # Save artifact
            file_path = self.artifact_manager.save_artifact(
                data=artifact_payload,
                base_name=artifact_name,
                extension=".json"
            )

            saved_files[artifact_name] = file_path
            self.logger.info(f"💾 Saved artifact {artifact_name} to {file_path}")
            tprint_success(
                f"✅ Artifact saved: {artifact_name}",
                {'path': file_path}
            )

        return saved_files

    def _attach_run_metadata_to_artifact(self, artifact_data: Any, run_metadata: Dict[str, Any]) -> Any:
        """Embed run metadata within the artifact payload for persistence."""
        if isinstance(artifact_data, dict):
            artifact_copy = dict(artifact_data)
            existing_run_metadata = artifact_copy.get('run_metadata') if isinstance(artifact_copy.get('run_metadata'), dict) else {}
            artifact_copy['run_metadata'] = {**run_metadata, **existing_run_metadata}
            return artifact_copy

        return {
            'data': artifact_data,
            'run_metadata': dict(run_metadata)
        }

    def validate_config(self) -> bool:
        """Validate the component configuration."""
        if not self.config.symbol:
            raise ValueError("Symbol is required")
        if not self.config.exchange:
            raise ValueError("Exchange is required")
        if not self.config.timeframe:
            raise ValueError("Timeframe is required")
        tprint_success(
            f"✅ Configuration validated for {self.__class__.__name__}",
            {
                'symbol': self.config.symbol,
                'exchange': self.config.exchange,
                'timeframe': self.config.timeframe
            }
        )
        return True

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the component."""
        status = {
            'component_name': self.__class__.__name__,
            'config': self.config,
            'required_artifacts': self.get_required_artifacts()
        }
        tprint_info(
            f"📊 Status requested for {self.__class__.__name__}",
            {
                'required_artifacts': status['required_artifacts'],
                'symbol': self.config.symbol,
                'exchange': self.config.exchange
            }
        )
        return status
