"""
Standardized Error Handler Module

This module provides standardized error handling utilities for the trading system.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import traceback
from enum import Enum


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ErrorCategory(str, Enum):
    """Error categories."""
    DATA_QUALITY = "data_quality"
    MODEL_TRAINING = "model_training"
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"
    RESOURCE = "resource"
    NETWORK = "network"
    VALIDATION = "validation"
    UNKNOWN = "unknown"


class ErrorContext:
    """Error context information."""
    
    def __init__(self, step_name: str, operation: str, **kwargs):
        self.step_name = step_name
        self.operation = operation
        self.timestamp = datetime.now().isoformat()
        self.data_context = kwargs.get('data_context', {})
        self.config_context = kwargs.get('config_context', {})
        self.user_context = kwargs.get('user_context', {})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            'step_name': self.step_name,
            'operation': self.operation,
            'timestamp': self.timestamp,
            'data_context': self.data_context,
            'config_context': self.config_context,
            'user_context': self.user_context
        }


class ErrorRecord:
    """Error record with full context."""
    
    def __init__(self, error: Exception, context: ErrorContext, severity: ErrorSeverity = ErrorSeverity.ERROR):
        self.error = error
        self.context = context
        self.severity = severity
        self.category = self._categorize_error(error)
        self.traceback = traceback.format_exc()
        self.recovery_strategy = self._get_recovery_strategy()
        self.timestamp = datetime.now().isoformat()
    
    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize the error based on its type and message."""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        if any(word in error_message for word in ['data', 'schema', 'validation']):
            return ErrorCategory.DATA_QUALITY
        elif any(word in error_message for word in ['model', 'training', 'prediction']):
            return ErrorCategory.MODEL_TRAINING
        elif any(word in error_message for word in ['config', 'setting', 'parameter']):
            return ErrorCategory.CONFIGURATION
        elif any(word in error_message for word in ['import', 'module', 'dependency']):
            return ErrorCategory.DEPENDENCY
        elif any(word in error_message for word in ['memory', 'disk', 'cpu']):
            return ErrorCategory.RESOURCE
        elif any(word in error_message for word in ['network', 'connection', 'timeout']):
            return ErrorCategory.NETWORK
        else:
            return ErrorCategory.UNKNOWN
    
    def _get_recovery_strategy(self) -> str:
        """Get recovery strategy based on error category."""
        strategies = {
            ErrorCategory.DATA_QUALITY: "Validate and clean input data",
            ErrorCategory.MODEL_TRAINING: "Check model parameters and retry",
            ErrorCategory.CONFIGURATION: "Verify configuration parameters",
            ErrorCategory.DEPENDENCY: "Install missing dependencies",
            ErrorCategory.RESOURCE: "Check system resources and retry",
            ErrorCategory.NETWORK: "Check network connection and retry",
            ErrorCategory.VALIDATION: "Review validation rules and data",
            ErrorCategory.UNKNOWN: "Review error details and implement specific handling"
        }
        return strategies.get(self.category, "Unknown recovery strategy")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error record to dictionary."""
        return {
            'error_type': type(self.error).__name__,
            'error_message': str(self.error),
            'severity': self.severity.value,
            'category': self.category.value,
            'context': self.context.to_dict(),
            'traceback': self.traceback,
            'recovery_strategy': self.recovery_strategy,
            'timestamp': self.timestamp
        }


class StandardizedErrorHandler:
    """Standardized error handler for the trading system."""
    
    def __init__(self):
        self.error_history: List[ErrorRecord] = []
        self.max_history_size = 1000
    
    def handle_error(self, error: Exception, context: ErrorContext, severity: ErrorSeverity = ErrorSeverity.ERROR) -> ErrorRecord:
        """Handle an error and create an error record."""
        error_record = ErrorRecord(error, context, severity)
        self.error_history.append(error_record)
        
        # Keep history manageable
        if len(self.error_history) > self.max_history_size:
            self.error_history = self.error_history[-self.max_history_size//2:]
        
        return error_record
    
    def get_errors_by_category(self, category: ErrorCategory) -> List[ErrorRecord]:
        """Get all errors of a specific category."""
        return [record for record in self.error_history if record.category == category]
    
    def get_errors_by_severity(self, severity: ErrorSeverity) -> List[ErrorRecord]:
        """Get all errors of a specific severity."""
        return [record for record in self.error_history if record.severity == severity]
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get a summary of all errors."""
        if not self.error_history:
            return {"total_errors": 0, "by_category": {}, "by_severity": {}}
        
        by_category = {}
        by_severity = {}
        
        for record in self.error_history:
            # Count by category
            cat = record.category.value
            by_category[cat] = by_category.get(cat, 0) + 1
            
            # Count by severity
            sev = record.severity.value
            by_severity[sev] = by_severity.get(sev, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "by_category": by_category,
            "by_severity": by_severity,
            "recent_errors": [record.to_dict() for record in self.error_history[-10:]]
        }


# Global instance
standardized_error_handler = StandardizedErrorHandler()


def handle_errors(exceptions=Exception, default_return=None, context="unknown"):
    """Decorator for standardized error handling."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                # Create error context
                error_context = ErrorContext(
                    step_name=context,
                    operation=func.__name__
                )
                
                # Handle the error
                error_record = standardized_error_handler.handle_error(e, error_context)
                
                # Return default value if specified
                if default_return is not None:
                    return default_return
                
                # Re-raise the exception
                raise
        
        return wrapper
    return decorator