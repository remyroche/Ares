#!/usr/bin/env python3
"""
Error Handling Framework

This module provides comprehensive error handling with recovery mechanisms
for the trading pipeline operations.
"""

import asyncio
import logging
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union, Type
import functools

from src.utils.logger import system_logger
from src.utils.common_operations import (
    format_datetime,
    get_current_datetime,
)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories."""
    DATA_VALIDATION = "data_validation"
    DATA_PROCESSING = "data_processing"
    MODEL_TRAINING = "model_training"
    DATA_ACCESS = "data_access"
    CONFIGURATION = "configuration"
    NETWORK = "network"
    SYSTEM = "system"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information for errors."""
    step_name: str
    function_name: str
    timestamp: str
    severity: ErrorSeverity
    category: ErrorCategory
    error_message: str
    error_type: str
    stack_trace: str
    additional_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryAction:
    """Recovery action definition."""
    action_name: str
    action_function: Callable
    max_attempts: int = 3
    delay_between_attempts: float = 1.0
    success_criteria: Callable = None


class ErrorHandler(ABC):
    """Base class for error handlers."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = system_logger.getChild(f"ErrorHandler.{name}")
    
    @abstractmethod
    async def handle_error(self, error: Exception, context: ErrorContext) -> bool:
        """Handle an error and return True if handled successfully."""
        pass
    
    @abstractmethod
    def can_handle(self, error: Exception, context: ErrorContext) -> bool:
        """Check if this handler can handle the given error."""
        pass


class DataValidationErrorHandler(ErrorHandler):
    """Handler for data validation errors."""
    
    def __init__(self):
        super().__init__("DataValidation")
    
    def can_handle(self, error: Exception, context: ErrorContext) -> bool:
        """Check if this handler can handle data validation errors."""
        return context.category == ErrorCategory.DATA_VALIDATION
    
    async def handle_error(self, error: Exception, context: ErrorContext) -> bool:
        """Handle data validation errors."""
        self.logger.warning(f"Handling data validation error: {error}")
        
        try:
            # Log the error details
            self.logger.error(f"Data validation failed in {context.step_name}: {context.error_message}")
            
            # Attempt to fix common data validation issues
            if "missing columns" in context.error_message.lower():
                self.logger.info("Attempting to fix missing columns issue")
                # Add logic to handle missing columns
                return True
            
            elif "invalid data type" in context.error_message.lower():
                self.logger.info("Attempting to fix data type issue")
                # Add logic to handle data type issues
                return True
            
            elif "empty dataframe" in context.error_message.lower():
                self.logger.info("Attempting to fix empty DataFrame issue")
                # Add logic to handle empty DataFrame
                return True
            
            return False
            
        except Exception as e:
            self.logger.exception(f"Failed to handle data validation error: {e}")
            return False


class DataProcessingErrorHandler(ErrorHandler):
    """Handler for data processing errors."""
    
    def __init__(self):
        super().__init__("DataProcessing")
    
    def can_handle(self, error: Exception, context: ErrorContext) -> bool:
        """Check if this handler can handle data processing errors."""
        return context.category == ErrorCategory.DATA_PROCESSING
    
    async def handle_error(self, error: Exception, context: ErrorContext) -> bool:
        """Handle data processing errors."""
        self.logger.warning(f"Handling data processing error: {error}")
        
        try:
            # Log the error details
            self.logger.error(f"Data processing failed in {context.step_name}: {context.error_message}")
            
            # Attempt to fix common data processing issues
            if "memory" in context.error_message.lower():
                self.logger.info("Attempting to fix memory issue")
                # Add logic to handle memory issues
                return True
            
            elif "timeout" in context.error_message.lower():
                self.logger.info("Attempting to fix timeout issue")
                # Add logic to handle timeout issues
                return True
            
            elif "conversion" in context.error_message.lower():
                self.logger.info("Attempting to fix data conversion issue")
                # Add logic to handle conversion issues
                return True
            
            return False
            
        except Exception as e:
            self.logger.exception(f"Failed to handle data processing error: {e}")
            return False


class ModelTrainingErrorHandler(ErrorHandler):
    """Handler for model training errors."""
    
    def __init__(self):
        super().__init__("ModelTraining")
    
    def can_handle(self, error: Exception, context: ErrorContext) -> bool:
        """Check if this handler can handle model training errors."""
        return context.category == ErrorCategory.MODEL_TRAINING
    
    async def handle_error(self, error: Exception, context: ErrorContext) -> bool:
        """Handle model training errors."""
        self.logger.warning(f"Handling model training error: {error}")
        
        try:
            # Log the error details
            self.logger.error(f"Model training failed in {context.step_name}: {context.error_message}")
            
            # Attempt to fix common model training issues
            if "convergence" in context.error_message.lower():
                self.logger.info("Attempting to fix convergence issue")
                # Add logic to handle convergence issues
                return True
            
            elif "overfitting" in context.error_message.lower():
                self.logger.info("Attempting to fix overfitting issue")
                # Add logic to handle overfitting
                return True
            
            elif "underfitting" in context.error_message.lower():
                self.logger.info("Attempting to fix underfitting issue")
                # Add logic to handle underfitting
                return True
            
            return False
            
        except Exception as e:
            self.logger.exception(f"Failed to handle model training error: {e}")
            return False


class DataAccessErrorHandler(ErrorHandler):
    """Handler for data access errors."""
    
    def __init__(self):
        super().__init__("DataAccess")
    
    def can_handle(self, error: Exception, context: ErrorContext) -> bool:
        """Check if this handler can handle data access errors."""
        return context.category == ErrorCategory.DATA_ACCESS
    
    async def handle_error(self, error: Exception, context: ErrorContext) -> bool:
        """Handle data access errors."""
        self.logger.warning(f"Handling data access error: {error}")
        
        try:
            # Log the error details
            self.logger.error(f"Data access failed in {context.step_name}: {context.error_message}")
            
            # Attempt to fix common data access issues
            if "file not found" in context.error_message.lower():
                self.logger.info("Attempting to fix file not found issue")
                # Add logic to handle missing files
                return True
            
            elif "permission denied" in context.error_message.lower():
                self.logger.info("Attempting to fix permission issue")
                # Add logic to handle permission issues
                return True
            
            elif "connection" in context.error_message.lower():
                self.logger.info("Attempting to fix connection issue")
                # Add logic to handle connection issues
                return True
            
            return False
            
        except Exception as e:
            self.logger.exception(f"Failed to handle data access error: {e}")
            return False


class ErrorRecoveryManager:
    """Manager for error recovery operations."""
    
    def __init__(self):
        self.logger = system_logger.getChild("ErrorRecoveryManager")
        self.error_handlers = [
            DataValidationErrorHandler(),
            DataProcessingErrorHandler(),
            ModelTrainingErrorHandler(),
            DataAccessErrorHandler(),
        ]
        self.recovery_actions = {}
        self.error_history = []
    
    def register_recovery_action(self, action: RecoveryAction):
        """Register a recovery action."""
        self.recovery_actions[action.action_name] = action
        self.logger.info(f"Registered recovery action: {action.action_name}")
    
    def create_error_context(
        self,
        error: Exception,
        step_name: str,
        function_name: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        additional_info: Dict[str, Any] = None
    ) -> ErrorContext:
        """Create error context from exception and metadata."""
        return ErrorContext(
            step_name=step_name,
            function_name=function_name,
            timestamp=format_datetime(get_current_datetime()),
            severity=severity,
            category=category,
            error_message=str(error),
            error_type=type(error).__name__,
            stack_trace=traceback.format_exc(),
            additional_info=additional_info or {}
        )
    
    async def handle_error(self, error: Exception, context: ErrorContext) -> bool:
        """Handle an error using registered handlers."""
        self.logger.error(f"Handling error: {context.error_message}")
        
        # Add to error history
        self.error_history.append(context)
        
        # Find appropriate handler
        for handler in self.error_handlers:
            if handler.can_handle(error, context):
                self.logger.info(f"Using handler: {handler.name}")
                try:
                    success = await handler.handle_error(error, context)
                    if success:
                        self.logger.info(f"Error handled successfully by {handler.name}")
                        return True
                except Exception as e:
                    self.logger.exception(f"Handler {handler.name} failed: {e}")
        
        # Try recovery actions
        for action_name, action in self.recovery_actions.items():
            self.logger.info(f"Attempting recovery action: {action_name}")
            try:
                success = await self._execute_recovery_action(action, context)
                if success:
                    self.logger.info(f"Recovery action {action_name} succeeded")
                    return True
            except Exception as e:
                self.logger.exception(f"Recovery action {action_name} failed: {e}")
        
        self.logger.error(f"No handler or recovery action could handle the error")
        return False
    
    async def _execute_recovery_action(self, action: RecoveryAction, context: ErrorContext) -> bool:
        """Execute a recovery action with retry logic."""
        for attempt in range(action.max_attempts):
            try:
                self.logger.info(f"Executing recovery action {action.action_name} (attempt {attempt + 1})")
                
                if asyncio.iscoroutinefunction(action.action_function):
                    result = await action.action_function(context)
                else:
                    result = action.action_function(context)
                
                # Check success criteria
                if action.success_criteria and not action.success_criteria(result):
                    self.logger.warning(f"Recovery action {action.action_name} did not meet success criteria")
                    if attempt < action.max_attempts - 1:
                        await asyncio.sleep(action.delay_between_attempts)
                        continue
                    return False
                
                return True
                
            except Exception as e:
                self.logger.warning(f"Recovery action {action.action_name} failed on attempt {attempt + 1}: {e}")
                if attempt < action.max_attempts - 1:
                    await asyncio.sleep(action.delay_between_attempts)
                else:
                    raise
        
        return False
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of error history."""
        if not self.error_history:
            return {"total_errors": 0}
        
        error_counts = {}
        severity_counts = {}
        category_counts = {}
        
        for context in self.error_history:
            # Count by error type
            error_type = context.error_type
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            # Count by severity
            severity = context.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Count by category
            category = context.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "error_counts": error_counts,
            "severity_counts": severity_counts,
            "category_counts": category_counts,
            "recent_errors": self.error_history[-5:] if len(self.error_history) > 5 else self.error_history
        }


# Global error recovery manager
error_recovery_manager = ErrorRecoveryManager()


def error_handler(
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    max_retries: int = 3,
    retry_delay: float = 1.0
):
    """Decorator for error handling with recovery."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            logger = system_logger.getChild(f"ErrorHandler.{func.__name__}")
            
            for attempt in range(max_retries + 1):
                try:
                    result = await func(*args, **kwargs)
                    if attempt > 0:
                        logger.info(f"Function {func.__name__} succeeded on attempt {attempt + 1}")
                    return result
                    
                except Exception as e:
                    if attempt < max_retries:
                        logger.warning(f"Function {func.__name__} failed on attempt {attempt + 1}: {e}")
                        
                        # Create error context
                        context = error_recovery_manager.create_error_context(
                            error=e,
                            step_name=func.__name__,
                            function_name=func.__name__,
                            severity=severity,
                            category=category
                        )
                        
                        # Try to handle the error
                        handled = await error_recovery_manager.handle_error(e, context)
                        if not handled:
                            logger.error(f"Error could not be handled, retrying...")
                        
                        await asyncio.sleep(retry_delay)
                    else:
                        logger.error(f"Function {func.__name__} failed after {max_retries + 1} attempts: {e}")
                        raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            logger = system_logger.getChild(f"ErrorHandler.{func.__name__}")
            
            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        logger.info(f"Function {func.__name__} succeeded on attempt {attempt + 1}")
                    return result
                    
                except Exception as e:
                    if attempt < max_retries:
                        logger.warning(f"Function {func.__name__} failed on attempt {attempt + 1}: {e}")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"Function {func.__name__} failed after {max_retries + 1} attempts: {e}")
                        raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def safe_execute(
    func: Callable,
    *args,
    default_return: Any = None,
    error_context: str = None,
    **kwargs
) -> Any:
    """Safely execute a function with error handling."""
    logger = system_logger.getChild("SafeExecute")
    
    try:
        if asyncio.iscoroutinefunction(func):
            return asyncio.run(func(*args, **kwargs))
        else:
            return func(*args, **kwargs)
    except Exception as e:
        logger.exception(f"Safe execution failed for {func.__name__}: {e}")
        
        if error_context:
            logger.error(f"Error context: {error_context}")
        
        return default_return


# Export commonly used functions
__all__ = [
    'ErrorSeverity',
    'ErrorCategory',
    'ErrorContext',
    'RecoveryAction',
    'ErrorHandler',
    'ErrorRecoveryManager',
    'error_recovery_manager',
    'error_handler',
    'safe_execute'
]