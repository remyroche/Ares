"""
Enhanced BaseStep with Unified Artifact Management

This module provides an enhanced BaseStep class that integrates seamlessly
with the unified artifact management system, providing step-based workflows
with comprehensive artifact handling.
"""

from __future__ import annotations

import asyncio
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
from pathlib import Path
import logging

from .base_step import BaseStep
from src.utils.unified_artifact_system import (
    UnifiedArtifactSystem, UnifiedConfig, UnifiedMetadata, EnhancedBaseStep
)
from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_success, tprint_info, tprint_warning, tprint_error


class StepArtifactManager:
    """
    Step-specific artifact manager that provides a clean interface
    for step-based artifact operations.
    """
    
    def __init__(self, step_name: str, artifact_system: UnifiedArtifactSystem, 
                 symbol: Optional[str] = None, exchange: Optional[str] = None,
                 interval: Optional[str] = None, direction: str = "long", 
                 model: str = "Analyst"):
        """Initialize the step artifact manager."""
        self.step_name = step_name
        self.artifact_system = artifact_system
        self.symbol = symbol
        self.exchange = exchange
        self.interval = interval
        self.direction = direction
        self.model = model
        
        # Set context
        self.artifact_system.set_context(
            step_name=step_name, symbol=symbol, exchange=exchange,
            interval=interval, direction=direction, model=model
        )
        
        self.logger = system_logger.getChild(f"StepArtifactManager.{step_name}")
    
    def store_input(self, data: Any, name: str, data_type: str = "input") -> str:
        """Store input data for this step."""
        artifact_name = f"{self.step_name}_input_{name}"
        return self.artifact_system.store_unified(
            data=data, artifact_name=artifact_name, artifact_type=data_type,
            symbol=self.symbol, exchange=self.exchange, interval=self.interval
        )
    
    def store_output(self, data: Any, name: str, data_type: str = "output") -> str:
        """Store output data for this step."""
        artifact_name = f"{self.step_name}_output_{name}"
        return self.artifact_system.store_unified(
            data=data, artifact_name=artifact_name, artifact_type=data_type,
            symbol=self.symbol, exchange=self.exchange, interval=self.interval
        )
    
    def store_intermediate(self, data: Any, name: str, data_type: str = "intermediate") -> str:
        """Store intermediate data for this step."""
        artifact_name = f"{self.step_name}_intermediate_{name}"
        return self.artifact_system.store_unified(
            data=data, artifact_name=artifact_name, artifact_type=data_type,
            symbol=self.symbol, exchange=self.exchange, interval=self.interval
        )
    
    def load_input(self, name: str, data_type: str = "input") -> Any:
        """Load input data for this step."""
        artifact_name = f"{self.step_name}_input_{name}"
        return self.artifact_system.load_unified(
            artifact_name=artifact_name, artifact_type=data_type,
            symbol=self.symbol, exchange=self.exchange, interval=self.interval
        )
    
    def load_output(self, name: str, data_type: str = "output") -> Any:
        """Load output data for this step."""
        artifact_name = f"{self.step_name}_output_{name}"
        return self.artifact_system.load_unified(
            artifact_name=artifact_name, artifact_type=data_type,
            symbol=self.symbol, exchange=self.exchange, interval=self.interval
        )
    
    def load_intermediate(self, name: str, data_type: str = "intermediate") -> Any:
        """Load intermediate data for this step."""
        artifact_name = f"{self.step_name}_intermediate_{name}"
        return self.artifact_system.load_unified(
            artifact_name=artifact_name, artifact_type=data_type,
            symbol=self.symbol, exchange=self.exchange, interval=self.interval
        )
    
    def store_klines(self, df: pd.DataFrame, batch_id: Optional[str] = None) -> str:
        """Store klines data using the specialized klines manager."""
        if not all(col in df.columns for col in ['timestamp', 'open', 'high', 'low', 'close', 'volume']):
            raise ValueError("DataFrame must contain OHLCV columns for klines storage")
        
        return self.artifact_system.store_klines(
            df=df, symbol=self.symbol, exchange=self.exchange, interval=self.interval,
            batch_id=batch_id
        )
    
    def load_klines(self, start_time: Optional[datetime] = None, 
                   end_time: Optional[datetime] = None, batch_id: Optional[str] = None) -> pd.DataFrame:
        """Load klines data using the specialized klines manager."""
        return self.artifact_system.load_klines(
            symbol=self.symbol, exchange=self.exchange, interval=self.interval,
            start_time=start_time, end_time=end_time, batch_id=batch_id
        )
    
    def list_artifacts(self, pattern: str = "*", artifact_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List artifacts for this step."""
        return self.artifact_system.list_artifacts(pattern, artifact_type)
    
    def get_metadata(self, name: str) -> Optional[UnifiedMetadata]:
        """Get metadata for an artifact."""
        return self.artifact_system.get_metadata(name)
    
    def cleanup_step_artifacts(self) -> None:
        """Clean up artifacts specific to this step."""
        artifacts = self.list_artifacts()
        step_artifacts = [a for a in artifacts if a['name'].startswith(f"{self.step_name}_")]
        
        for artifact in step_artifacts:
            self.artifact_system.delete_artifact(artifact['name'], artifact['type'])
        
        tprint_info(f"🧹 Cleaned up {len(step_artifacts)} artifacts for step {self.step_name}")


class EnhancedBaseStep(BaseStep):
    """
    Enhanced BaseStep with unified artifact management.
    
    This class provides:
    - Seamless artifact management for step-based workflows
    - Automatic context setting and management
    - Specialized methods for different data types
    - Performance tracking and metrics
    - Error handling and recovery
    """
    
    def __init__(self, config: Dict[str, Any], artifact_system: Optional[UnifiedArtifactSystem] = None):
        """Initialize the enhanced base step."""
        super().__init__(config)
        
        # Initialize artifact system
        self.artifact_system = artifact_system or UnifiedArtifactSystem()
        
        # Extract context from config
        self.step_name = config.get('step_name', self.__class__.__name__)
        self.symbol = config.get('symbol')
        self.exchange = config.get('exchange')
        self.interval = config.get('interval')
        self.direction = config.get('direction', 'long')
        self.model = config.get('model', 'Analyst')
        
        # Initialize step artifact manager
        self.artifacts = StepArtifactManager(
            step_name=self.step_name,
            artifact_system=self.artifact_system,
            symbol=self.symbol,
            exchange=self.exchange,
            interval=self.interval,
            direction=self.direction,
            model=self.model
        )
        
        # Step execution tracking
        self._execution_start_time: Optional[datetime] = None
        self._execution_end_time: Optional[datetime] = None
        self._execution_success: bool = False
        self._execution_error: Optional[str] = None
        
        # Step metrics
        self._step_metrics = {
            'executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'total_execution_time': 0.0,
            'average_execution_time': 0.0,
            'artifacts_created': 0,
            'artifacts_loaded': 0
        }
        
        self.logger = system_logger.getChild(f"EnhancedBaseStep.{self.step_name}")
        tprint_info(f"🚀 INITIALIZED ENHANCED STEP: {self.step_name}")
    
    async def execute(self, data: Any) -> Any:
        """
        Execute the step with enhanced artifact management.
        
        This method wraps the actual step execution with:
        - Execution timing
        - Error handling
        - Artifact management
        - Metrics collection
        """
        self._execution_start_time = datetime.utcnow()
        self._step_metrics['executions'] += 1
        
        try:
            tprint_info(f"🔄 EXECUTING STEP: {self.step_name}")
            
            # Execute the actual step logic
            result = await self._execute_step(data)
            
            # Mark execution as successful
            self._execution_success = True
            self._step_metrics['successful_executions'] += 1
            
            # Calculate execution time
            self._execution_end_time = datetime.utcnow()
            execution_time = (self._execution_end_time - self._execution_start_time).total_seconds()
            self._step_metrics['total_execution_time'] += execution_time
            self._step_metrics['average_execution_time'] = (
                self._step_metrics['total_execution_time'] / self._step_metrics['executions']
            )
            
            tprint_success(f"✅ STEP COMPLETED: {self.step_name} ({execution_time:.2f}s)")
            return result
            
        except Exception as e:
            # Mark execution as failed
            self._execution_success = False
            self._execution_error = str(e)
            self._step_metrics['failed_executions'] += 1
            
            self._execution_end_time = datetime.utcnow()
            execution_time = (self._execution_end_time - self._execution_start_time).total_seconds()
            
            tprint_error(f"❌ STEP FAILED: {self.step_name} - {str(e)} ({execution_time:.2f}s)")
            raise
    
    @abstractmethod
    async def _execute_step(self, data: Any) -> Any:
        """
        Execute the actual step logic.
        
        This method should be implemented by subclasses to define
        the specific logic for the step.
        """
        pass
    
    def validate_config(self) -> None:
        """Validate the step configuration."""
        required_fields = ['step_name']
        missing_fields = [field for field in required_fields if not self.config.get(field)]
        
        if missing_fields:
            raise ValueError(f"Missing required configuration fields: {missing_fields}")
        
        # Validate artifact system
        if not self.artifact_system:
            raise ValueError("Artifact system is required")
        
        tprint_success(f"✅ CONFIG VALIDATION PASSED: {self.step_name}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive step status and metrics."""
        return {
            'step_name': self.step_name,
            'symbol': self.symbol,
            'exchange': self.exchange,
            'interval': self.interval,
            'direction': self.direction,
            'model': self.model,
            'execution_status': {
                'last_execution_success': self._execution_success,
                'last_execution_error': self._execution_error,
                'last_execution_start': self._execution_start_time.isoformat() if self._execution_start_time else None,
                'last_execution_end': self._execution_end_time.isoformat() if self._execution_end_time else None,
                'last_execution_duration': (
                    (self._execution_end_time - self._execution_start_time).total_seconds()
                    if self._execution_start_time and self._execution_end_time else None
                )
            },
            'metrics': self._step_metrics.copy(),
            'artifact_count': len(self.artifacts.list_artifacts()),
            'artifact_system_metrics': self.artifact_system.get_performance_metrics()
        }
    
    def get_step_artifacts(self) -> List[Dict[str, Any]]:
        """Get all artifacts created by this step."""
        return self.artifacts.list_artifacts()
    
    def cleanup_step(self) -> None:
        """Clean up all artifacts created by this step."""
        self.artifacts.cleanup_step_artifacts()
        tprint_info(f"🧹 STEP CLEANUP COMPLETED: {self.step_name}")
    
    def reset_step(self) -> None:
        """Reset step state and metrics."""
        self._execution_start_time = None
        self._execution_end_time = None
        self._execution_success = False
        self._execution_error = None
        
        self._step_metrics = {
            'executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'total_execution_time': 0.0,
            'average_execution_time': 0.0,
            'artifacts_created': 0,
            'artifacts_loaded': 0
        }
        
        tprint_info(f"🔄 STEP RESET: {self.step_name}")
    
    # Convenience methods for common operations
    def store_dataframe(self, df: pd.DataFrame, name: str, data_type: str = "data") -> str:
        """Store a DataFrame with automatic type detection."""
        if self._is_klines_dataframe(df):
            return self.artifacts.store_klines(df)
        else:
            return self.artifacts.store_output(df, name, data_type)
    
    def load_dataframe(self, name: str, data_type: str = "data") -> pd.DataFrame:
        """Load a DataFrame."""
        return self.artifacts.load_output(name, data_type)
    
    def store_model(self, model: Any, name: str) -> str:
        """Store a machine learning model."""
        return self.artifacts.store_output(model, name, "model")
    
    def load_model(self, name: str) -> Any:
        """Load a machine learning model."""
        return self.artifacts.load_output(name, "model")
    
    def store_metadata(self, metadata: Dict[str, Any], name: str) -> str:
        """Store metadata."""
        return self.artifacts.store_output(metadata, name, "metadata")
    
    def load_metadata(self, name: str) -> Dict[str, Any]:
        """Load metadata."""
        return self.artifacts.load_output(name, "metadata")
    
    def _is_klines_dataframe(self, df: pd.DataFrame) -> bool:
        """Check if a DataFrame contains klines data."""
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        return all(col in df.columns for col in required_columns)
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of step execution."""
        return {
            'step_name': self.step_name,
            'total_executions': self._step_metrics['executions'],
            'success_rate': (
                self._step_metrics['successful_executions'] / self._step_metrics['executions']
                if self._step_metrics['executions'] > 0 else 0
            ),
            'average_execution_time': self._step_metrics['average_execution_time'],
            'last_execution_success': self._execution_success,
            'artifacts_created': self._step_metrics['artifacts_created'],
            'artifacts_loaded': self._step_metrics['artifacts_loaded']
        }


# Factory functions
def create_enhanced_step(step_class: type, config: Dict[str, Any], 
                        artifact_system: Optional[UnifiedArtifactSystem] = None) -> EnhancedBaseStep:
    """Create an enhanced step instance."""
    return step_class(config, artifact_system)


def create_step_with_artifacts(step_name: str, config: Dict[str, Any],
                              artifact_system: Optional[UnifiedArtifactSystem] = None) -> EnhancedBaseStep:
    """Create a step with artifact management capabilities."""
    
    class DynamicStep(EnhancedBaseStep):
        def __init__(self, config):
            super().__init__(config, artifact_system)
        
        async def _execute_step(self, data: Any) -> Any:
            # Default implementation - can be overridden
            return data
    
    return DynamicStep(config)