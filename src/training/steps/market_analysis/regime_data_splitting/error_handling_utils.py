"""
Standardized error handling utilities for regime data splitting module.

This module provides consistent error handling patterns, standardized error messages,
and comprehensive error context management for the regime data splitting components.
"""

import logging
import traceback
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime


class ErrorSeverity(Enum):
    """Error severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    WARNING = "warning"


class ErrorCategory(Enum):
    """Error categories for classification."""
    VALIDATION_ERROR = "validation_error"
    DATA_ERROR = "data_error"
    CONFIG_ERROR = "config_error"
    PROCESSING_ERROR = "processing_error"
    RESOURCE_ERROR = "resource_error"
    ALIGNMENT_ERROR = "alignment_error"
    MODEL_ERROR = "model_error"
    IO_ERROR = "io_error"


@dataclass
class StandardizedError:
    """Standardized error structure."""
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    action_required: str
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    component: Optional[str] = None
    traceback_info: Optional[str] = None
    
    def to_string(self) -> str:
        """Convert error to standardized string format."""
        return f"{self.category.value.upper()}: {self.message}. Action required: {self.action_required}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary."""
        return {
            'category': self.category.value,
            'severity': self.severity.value,
            'message': self.message,
            'action_required': self.action_required,
            'context': self.context,
            'timestamp': self.timestamp,
            'component': self.component,
            'traceback_info': self.traceback_info
        }


class StandardizedErrorHandler:
    """Standardized error handler for consistent error management."""
    
    def __init__(self, component_name: str, logger: Optional[logging.Logger] = None):
        self.component_name = component_name
        self.logger = logger or logging.getLogger(__name__)
        self.error_history: List[StandardizedError] = []
    
    def create_error(
        self,
        category: ErrorCategory,
        severity: ErrorSeverity,
        message: str,
        action_required: str,
        context: Optional[Dict[str, Any]] = None,
        include_traceback: bool = False
    ) -> StandardizedError:
        """Create a standardized error."""
        error = StandardizedError(
            category=category,
            severity=severity,
            message=message,
            action_required=action_required,
            context=context or {},
            component=self.component_name
        )
        
        if include_traceback:
            error.traceback_info = traceback.format_exc()
        
        # Store in history
        self.error_history.append(error)
        
        return error
    
    def handle_validation_error(
        self,
        message: str,
        action_required: str,
        context: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.HIGH
    ) -> StandardizedError:
        """Handle validation errors with consistent pattern."""
        error = self.create_error(
            category=ErrorCategory.VALIDATION_ERROR,
            severity=severity,
            message=message,
            action_required=action_required,
            context=context
        )
        
        # Log based on severity
        if severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
            self.logger.error(f"❌ {error.to_string()}")
        else:
            self.logger.warning(f"⚠️ {error.to_string()}")
        
        return error
    
    def handle_data_error(
        self,
        message: str,
        action_required: str,
        context: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.HIGH
    ) -> StandardizedError:
        """Handle data-related errors."""
        error = self.create_error(
            category=ErrorCategory.DATA_ERROR,
            severity=severity,
            message=message,
            action_required=action_required,
            context=context
        )
        
        if severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
            self.logger.error(f"❌ {error.to_string()}")
        else:
            self.logger.warning(f"⚠️ {error.to_string()}")
        
        return error
    
    def handle_config_error(
        self,
        message: str,
        action_required: str,
        context: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.HIGH
    ) -> StandardizedError:
        """Handle configuration errors."""
        error = self.create_error(
            category=ErrorCategory.CONFIG_ERROR,
            severity=severity,
            message=message,
            action_required=action_required,
            context=context
        )
        
        self.logger.error(f"❌ {error.to_string()}")
        return error
    
    def handle_processing_error(
        self,
        message: str,
        action_required: str,
        context: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        include_traceback: bool = True
    ) -> StandardizedError:
        """Handle processing errors."""
        error = self.create_error(
            category=ErrorCategory.PROCESSING_ERROR,
            severity=severity,
            message=message,
            action_required=action_required,
            context=context,
            include_traceback=include_traceback
        )
        
        if severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
            self.logger.error(f"❌ {error.to_string()}")
        else:
            self.logger.warning(f"⚠️ {error.to_string()}")
        
        return error
    
    def handle_resource_error(
        self,
        message: str,
        action_required: str,
        context: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM
    ) -> StandardizedError:
        """Handle resource-related errors."""
        error = self.create_error(
            category=ErrorCategory.RESOURCE_ERROR,
            severity=severity,
            message=message,
            action_required=action_required,
            context=context
        )
        
        if severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
            self.logger.error(f"❌ {error.to_string()}")
        else:
            self.logger.warning(f"⚠️ {error.to_string()}")
        
        return error
    
    def handle_alignment_error(
        self,
        message: str,
        action_required: str,
        context: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.HIGH
    ) -> StandardizedError:
        """Handle data alignment errors."""
        error = self.create_error(
            category=ErrorCategory.ALIGNMENT_ERROR,
            severity=severity,
            message=message,
            action_required=action_required,
            context=context
        )
        
        if severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
            self.logger.error(f"❌ {error.to_string()}")
        else:
            self.logger.warning(f"⚠️ {error.to_string()}")
        
        return error
    
    def handle_model_error(
        self,
        message: str,
        action_required: str,
        context: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM
    ) -> StandardizedError:
        """Handle model-related errors."""
        error = self.create_error(
            category=ErrorCategory.MODEL_ERROR,
            severity=severity,
            message=message,
            action_required=action_required,
            context=context
        )
        
        if severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
            self.logger.error(f"❌ {error.to_string()}")
        else:
            self.logger.warning(f"⚠️ {error.to_string()}")
        
        return error
    
    def handle_io_error(
        self,
        message: str,
        action_required: str,
        context: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.HIGH
    ) -> StandardizedError:
        """Handle I/O related errors."""
        error = self.create_error(
            category=ErrorCategory.IO_ERROR,
            severity=severity,
            message=message,
            action_required=action_required,
            context=context
        )
        
        if severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
            self.logger.error(f"❌ {error.to_string()}")
        else:
            self.logger.warning(f"⚠️ {error.to_string()}")
        
        return error
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors encountered."""
        if not self.error_history:
            return {'total_errors': 0, 'by_category': {}, 'by_severity': {}}
        
        by_category = {}
        by_severity = {}
        
        for error in self.error_history:
            # Count by category
            category = error.category.value
            by_category[category] = by_category.get(category, 0) + 1
            
            # Count by severity
            severity = error.severity.value
            by_severity[severity] = by_severity.get(severity, 0) + 1
        
        return {
            'total_errors': len(self.error_history),
            'by_category': by_category,
            'by_severity': by_severity,
            'latest_errors': [error.to_dict() for error in self.error_history[-5:]]  # Last 5 errors
        }
    
    def clear_error_history(self) -> None:
        """Clear the error history."""
        self.error_history.clear()


# Convenience functions for common error patterns
def create_validation_error(message: str, action_required: str, component: str = "regime_data_splitting") -> str:
    """Create a standardized validation error message."""
    return f"VALIDATION_ERROR: {message}. Action required: {action_required}"

def create_data_error(message: str, action_required: str, component: str = "regime_data_splitting") -> str:
    """Create a standardized data error message."""
    return f"DATA_ERROR: {message}. Action required: {action_required}"

def create_config_error(message: str, action_required: str, component: str = "regime_data_splitting") -> str:
    """Create a standardized configuration error message."""
    return f"CONFIG_ERROR: {message}. Action required: {action_required}"

def create_processing_error(message: str, action_required: str, component: str = "regime_data_splitting") -> str:
    """Create a standardized processing error message."""
    return f"PROCESSING_ERROR: {message}. Action required: {action_required}"

def create_alignment_error(message: str, action_required: str, component: str = "regime_data_splitting") -> str:
    """Create a standardized alignment error message."""
    return f"ALIGNMENT_ERROR: {message}. Action required: {action_required}"


# Global error handler instance
_global_error_handler = None

def get_error_handler(component_name: str, logger: Optional[logging.Logger] = None) -> StandardizedErrorHandler:
    """Get or create a standardized error handler."""
    global _global_error_handler
    
    if _global_error_handler is None:
        _global_error_handler = StandardizedErrorHandler(component_name, logger)
    
    return _global_error_handler

def reset_error_handler():
    """Reset the global error handler (useful for testing)."""
    global _global_error_handler
    _global_error_handler = None