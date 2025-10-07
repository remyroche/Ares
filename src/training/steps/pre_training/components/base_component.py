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
from src.utils.tprint import tprint, tprint_error, tprint_warning, tprint_success, tprint_debug

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
    
    @abstractmethod
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        pass
    
    @abstractmethod
    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """Execute the component."""
        pass
    
    async def save_artifacts(self, artifacts: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, str]:
        """
        Save artifacts persistently.
        
        Args:
            artifacts: Artifacts to save
            metadata: Metadata for the artifacts
            
        Returns:
            Dictionary mapping artifact names to file paths
        """
        try:
            saved_files = {}
            
            for artifact_name, artifact_data in artifacts.items():
                # Create artifact metadata
                artifact_metadata = {
                    'component': self.__class__.__name__,
                    'timestamp': datetime.now().isoformat(),
                    'symbol': self.config.symbol,
                    'exchange': self.config.exchange,
                    'timeframe': self.config.timeframe,
                    **metadata
                }
                
                # Save artifact
                file_path = self.artifact_manager.save_artifact(
                    data=artifact_data,
                    base_name=artifact_name,
                    extension=".json"
                )
                
                saved_files[artifact_name] = file_path
                self.logger.info(f"💾 Saved artifact {artifact_name} to {file_path}")
            
            return saved_files
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save artifacts: {e}")
            return {}
    
    def validate_config(self) -> bool:
        """Validate the component configuration."""
        if not self.config.symbol:
            raise ValueError("Symbol is required")
        if not self.config.exchange:
            raise ValueError("Exchange is required")
        if not self.config.timeframe:
            raise ValueError("Timeframe is required")
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the component."""
        return {
            'component_name': self.__class__.__name__,
            'config': self.config,
            'required_artifacts': self.get_required_artifacts()
        }