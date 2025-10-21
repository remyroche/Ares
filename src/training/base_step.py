"""
Base Step for Training Pipeline with Comprehensive Type Safety

This module defines a simple BaseStep class that can be inherited by other training steps.
It provides a basic structure for steps in the training pipeline with enhanced type safety
and comprehensive tpritn logging.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, List, TypeVar, Generic, Protocol, runtime_checkable, Literal, Final, ClassVar, cast, overload, Callable, Type, Tuple, Set, FrozenSet, Mapping, MutableMapping, Sequence, MutableSequence, Iterable, Iterator, Generator, Awaitable, Coroutine, AnyStr, Text, BinaryIO, IO
from datetime import datetime
import logging

# Import tpritn for comprehensive logging
from src.utils.tprint import (
    tprint, tprint_success, tprint_info, tprint_warning, tprint_error,
    tprint_debug, tprint_performance, tprint_progress, tprint_structured,
    tprint_exception, tprint_with_level, LogLevel, TPrintConfig
)

# Type definitions for better type safety
T = TypeVar('T')
ConfigType = TypeVar('ConfigType', bound=Dict[str, Any])
DataType = TypeVar('DataType', bound=Any)

# Protocol definitions for better type checking
@runtime_checkable
class Validatable(Protocol):
    """Protocol for validatable objects."""
    def validate(self) -> bool: ...

@runtime_checkable
class Executable(Protocol):
    """Protocol for executable objects."""
    async def execute(self, data: Any) -> Any: ...

class BaseStep(ABC):
    """
    Abstract Base Class for a training step with comprehensive type safety.
    
    This class provides a foundation for all training steps with:
    - Comprehensive type hints for better IDE support and type checking
    - Extensive tpritn logging for better debugging and monitoring
    - Runtime validation of configuration and data
    - Protocol-based interfaces for better extensibility
    """
    
    # Class variables for type hints
    config: Dict[str, Any]
    logger: logging.Logger
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the base step with comprehensive validation.
        
        Args:
            config: Configuration dictionary for the step
            
        Raises:
            ValueError: If config is empty or invalid
            TypeError: If config is not a dictionary
        """
        # Validate input parameters
        if not isinstance(config, dict):
            raise TypeError(f"config must be a dictionary, got: {type(config).__name__}")
        if not config:
            raise ValueError("config cannot be empty")
        
        tprint_info(f"🚀 Initializing BaseStep with config keys: {list(config.keys())}")
        
        self.config = config
        self.logger = logging.getLogger(f"ares.training.step.{self.__class__.__name__}")
        
        tprint_success(f"✅ BaseStep initialized: {self.__class__.__name__}")

    @abstractmethod
    async def execute(self, data: Any) -> Any:
        """
        Execute the logic of the training step with comprehensive validation.
        
        Args:
            data: Input data for the step
            
        Returns:
            Processed data or result from the step
            
        Raises:
            NotImplementedError: If not implemented by subclass
            ValueError: If data is invalid
        """
        # Validate input parameters
        if data is None:
            raise ValueError("data cannot be None")
        
        tprint_info(f"🚀 Executing training step: {self.__class__.__name__}")
        tprint_debug(f"📊 Input data type: {type(data).__name__}")
        
        # This method must be implemented by subclasses
        raise NotImplementedError("Subclasses must implement the execute method")

    @abstractmethod
    def validate_config(self) -> None:
        """
        Validate the configuration for the step with comprehensive checks.
        
        Raises:
            NotImplementedError: If not implemented by subclass
            ValueError: If configuration is invalid
            TypeError: If configuration types are incorrect
        """
        tprint_info(f"🔍 Validating configuration for step: {self.__class__.__name__}")
        
        # This method must be implemented by subclasses
        raise NotImplementedError("Subclasses must implement the validate_config method")

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status and metrics of the step with comprehensive information.
        
        Returns:
            Dictionary containing step status and metrics
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        tprint_info(f"📊 Getting status for step: {self.__class__.__name__}")
        
        # This method must be implemented by subclasses
        raise NotImplementedError("Subclasses must implement the get_status method")
    
    def _validate_config_common(self) -> None:
        """
        Common configuration validation that can be used by subclasses.
        
        Raises:
            ValueError: If common configuration requirements are not met
        """
        tprint_debug("🔍 Performing common configuration validation")
        
        if not self.config:
            raise ValueError("Configuration cannot be empty")
        
        # Check for required common fields
        required_fields = ['step_name', 'execution_mode']
        missing_fields = [field for field in required_fields if field not in self.config]
        
        if missing_fields:
            raise ValueError(f"Missing required configuration fields: {missing_fields}")
        
        # Validate field types
        if not isinstance(self.config.get('step_name'), str):
            raise TypeError("step_name must be a string")
        
        if not isinstance(self.config.get('execution_mode'), str):
            raise TypeError("execution_mode must be a string")
        
        tprint_success("✅ Common configuration validation passed")
    
    def _log_step_start(self, step_name: str) -> None:
        """
        Log the start of a step execution.
        
        Args:
            step_name: Name of the step being executed
        """
        tprint_info(f"🚀 Starting step execution: {step_name}")
        
    def _log_step_end(self, step_name: str, success: bool, execution_time: float) -> None:
        """
        Log the end of a step execution.
        
        Args:
            step_name: Name of the step that was executed
            success: Whether the step completed successfully
            execution_time: Time taken to execute the step in seconds
        """
        if success:
            tprint_success(f"✅ Step completed successfully: {step_name} in {execution_time:.2f}s")
        else:
            tprint_error(f"❌ Step failed: {step_name} after {execution_time:.2f}s")
    
    def _log_data_info(self, data: Any, operation: str) -> None:
        """
        Log information about data being processed.
        
        Args:
            data: Data being processed
            operation: Operation being performed on the data
        """
        data_type = type(data).__name__
        data_size = len(data) if hasattr(data, '__len__') else 'unknown'
        tprint_debug(f"📊 {operation} data: type={data_type}, size={data_size}")
    
    def _log_config_info(self) -> None:
        """
        Log configuration information for debugging.
        """
        tprint_debug(f"⚙️ Configuration: {self.config}")
    
    def _validate_data_type(self, data: Any, expected_type: Type, operation: str) -> None:
        """
        Validate that data is of the expected type.
        
        Args:
            data: Data to validate
            expected_type: Expected type of the data
            operation: Operation being performed (for error messages)
            
        Raises:
            TypeError: If data is not of the expected type
        """
        if not isinstance(data, expected_type):
            raise TypeError(f"{operation} expected {expected_type.__name__}, got {type(data).__name__}")
        
        tprint_debug(f"✅ Data type validation passed for {operation}: {type(data).__name__}")
    
    def _get_config_value(self, key: str, default: Any = None, expected_type: Type = None) -> Any:
        """
        Get a configuration value with type validation.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            expected_type: Expected type of the value
            
        Returns:
            Configuration value or default
            
        Raises:
            TypeError: If value is not of expected type
        """
        value = self.config.get(key, default)
        
        if expected_type is not None and value is not None:
            if not isinstance(value, expected_type):
                raise TypeError(f"Config value '{key}' must be {expected_type.__name__}, got {type(value).__name__}")
        
        tprint_debug(f"🔧 Retrieved config value: {key} = {value}")
        return value
    
    def _log_performance_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Log performance metrics in a structured way.
        
        Args:
            metrics: Dictionary containing performance metrics
        """
        tprint_structured(metrics, LogLevel.INFO)
        
    def _log_error_with_context(self, error: Exception, context: str) -> None:
        """
        Log an error with additional context information.
        
        Args:
            error: Exception that occurred
            context: Additional context about where the error occurred
        """
        tprint_error(f"❌ Error in {context}: {str(error)}")
        tprint_exception(error, f"Context: {context}")
    
    def _log_success_with_metrics(self, operation: str, metrics: Dict[str, Any]) -> None:
        """
        Log a successful operation with associated metrics.
        
        Args:
            operation: Name of the operation that succeeded
            metrics: Metrics associated with the operation
        """
        tprint_success(f"✅ {operation} completed successfully")
        tprint_structured(metrics, LogLevel.INFO)
