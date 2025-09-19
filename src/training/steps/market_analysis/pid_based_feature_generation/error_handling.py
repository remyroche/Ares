"""
Standardized Error Handling for PID-Based Feature Generation

This module provides consistent error handling patterns across all feature generators
to ensure predictable behavior and better debugging capabilities.
"""

import logging
import traceback
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum
from dataclasses import dataclass, field
from contextlib import contextmanager

# Import tprint for consistent logging
try:
    from src.utils.tprint import tprint_error, tprint_warning, tprint_info, tprint_debug
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)


class ErrorSeverity(Enum):
    """Error severity levels."""
    CRITICAL = "critical"  # System cannot continue
    HIGH = "high"         # Feature generation fails
    MEDIUM = "medium"     # Partial failure, fallback available
    LOW = "low"          # Warning, operation continues


class ErrorCategory(Enum):
    """Error categories for better classification."""
    DATA_VALIDATION = "data_validation"
    TYPE_CONVERSION = "type_conversion"
    COMPUTATION = "computation"
    MEMORY = "memory"
    IMPORT_DEPENDENCY = "import_dependency"
    CONFIGURATION = "configuration"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass
class StandardError:
    """Standardized error information."""
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    original_exception: Optional[Exception] = None
    context: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    recoverable: bool = True


class PIDFeatureGenerationError(Exception):
    """Base exception for PID feature generation errors."""
    
    def __init__(self, error: StandardError):
        self.error = error
        super().__init__(error.message)


class ErrorHandler:
    """Centralized error handler for consistent error management."""
    
    def __init__(self, logger_name: str = "PIDFeatureGeneration"):
        self.logger = logging.getLogger(logger_name)
        self.error_history: List[StandardError] = []
    
    def handle_error(
        self, 
        exception: Exception, 
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None,
        raise_on_critical: bool = True
    ) -> StandardError:
        """
        Handle an error with standardized processing.
        
        Args:
            exception: The original exception
            category: Error category
            severity: Error severity level
            context: Additional context information
            suggestions: Suggested remediation steps
            raise_on_critical: Whether to raise exception on critical errors
            
        Returns:
            StandardError object with processed error information
        """
        # Create standardized error
        error = StandardError(
            category=category,
            severity=severity,
            message=str(exception),
            original_exception=exception,
            context=context or {},
            suggestions=suggestions or [],
            recoverable=severity != ErrorSeverity.CRITICAL
        )
        
        # Add to error history
        self.error_history.append(error)
        
        # Log based on severity
        if severity == ErrorSeverity.CRITICAL:
            tprint_error(f"CRITICAL {category.value}: {error.message}")
            if error.context:
                tprint_error(f"Context: {error.context}")
            if raise_on_critical:
                raise PIDFeatureGenerationError(error)
        elif severity == ErrorSeverity.HIGH:
            tprint_error(f"HIGH {category.value}: {error.message}")
            if error.suggestions:
                tprint_error(f"Suggestions: {error.suggestions}")
        elif severity == ErrorSeverity.MEDIUM:
            tprint_warning(f"MEDIUM {category.value}: {error.message}")
            if error.suggestions:
                tprint_warning(f"Suggestions: {error.suggestions}")
        else:  # LOW
            tprint_info(f"LOW {category.value}: {error.message}")
        
        return error
    
    def handle_data_validation_error(
        self, 
        exception: Exception, 
        data_shape: Optional[tuple] = None,
        data_type: Optional[str] = None,
        feature_count: Optional[int] = None
    ) -> StandardError:
        """Handle data validation errors with specific context."""
        context = {}
        if data_shape:
            context['data_shape'] = data_shape
        if data_type:
            context['data_type'] = data_type
        if feature_count:
            context['feature_count'] = feature_count
        
        suggestions = [
            "Check input data format and dimensions",
            "Ensure data contains numeric values",
            "Verify feature names match data columns"
        ]
        
        return self.handle_error(
            exception,
            ErrorCategory.DATA_VALIDATION,
            ErrorSeverity.HIGH,
            context,
            suggestions
        )
    
    def handle_computation_error(
        self, 
        exception: Exception, 
        operation: str,
        input_shapes: Optional[Dict[str, tuple]] = None
    ) -> StandardError:
        """Handle computation errors with operation context."""
        context = {'operation': operation}
        if input_shapes:
            context['input_shapes'] = input_shapes
        
        suggestions = [
            f"Check inputs for {operation} operation",
            "Verify data contains no NaN or infinite values",
            "Consider reducing data size if memory issues"
        ]
        
        return self.handle_error(
            exception,
            ErrorCategory.COMPUTATION,
            ErrorSeverity.MEDIUM,
            context,
            suggestions
        )
    
    def handle_memory_error(
        self, 
        exception: Exception, 
        operation: str,
        data_size: Optional[int] = None
    ) -> StandardError:
        """Handle memory errors with size context."""
        context = {'operation': operation}
        if data_size:
            context['data_size'] = data_size
        
        suggestions = [
            "Reduce batch size or data dimensions",
            "Enable memory optimization settings",
            "Consider using chunked processing"
        ]
        
        return self.handle_error(
            exception,
            ErrorCategory.MEMORY,
            ErrorSeverity.HIGH,
            context,
            suggestions
        )
    
    def handle_import_error(
        self, 
        exception: Exception, 
        module_name: str,
        fallback_available: bool = False
    ) -> StandardError:
        """Handle import dependency errors."""
        context = {
            'module_name': module_name,
            'fallback_available': fallback_available
        }
        
        suggestions = [
            f"Install missing dependency: {module_name}",
            "Check Python environment and package versions"
        ]
        
        if fallback_available:
            suggestions.append("Fallback mechanism will be used")
        
        severity = ErrorSeverity.MEDIUM if fallback_available else ErrorSeverity.HIGH
        
        return self.handle_error(
            exception,
            ErrorCategory.IMPORT_DEPENDENCY,
            severity,
            context,
            suggestions,
            raise_on_critical=False
        )
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors encountered."""
        if not self.error_history:
            return {'total_errors': 0}
        
        summary = {
            'total_errors': len(self.error_history),
            'by_category': {},
            'by_severity': {},
            'recoverable_count': sum(1 for e in self.error_history if e.recoverable),
            'critical_count': sum(1 for e in self.error_history if e.severity == ErrorSeverity.CRITICAL)
        }
        
        # Count by category
        for error in self.error_history:
            category = error.category.value
            severity = error.severity.value
            
            summary['by_category'][category] = summary['by_category'].get(category, 0) + 1
            summary['by_severity'][severity] = summary['by_severity'].get(severity, 0) + 1
        
        return summary


@contextmanager
def safe_operation(
    error_handler: ErrorHandler,
    operation_name: str,
    category: ErrorCategory = ErrorCategory.COMPUTATION,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    default_return: Any = None,
    context: Optional[Dict[str, Any]] = None
):
    """
    Context manager for safe operations with standardized error handling.
    
    Usage:
        with safe_operation(error_handler, "matrix_multiplication", default_return=np.zeros((10, 10))) as op:
            result = np.dot(matrix_a, matrix_b)
            op.set_result(result)
    """
    class OperationResult:
        def __init__(self):
            self.result = default_return
            self.success = False
        
        def set_result(self, result):
            self.result = result
            self.success = True
    
    op_result = OperationResult()
    
    try:
        tprint_debug(f"Starting operation: {operation_name}")
        yield op_result
        if op_result.success:
            tprint_debug(f"Operation completed successfully: {operation_name}")
    except Exception as e:
        error_handler.handle_error(
            e, 
            category=category, 
            severity=severity,
            context={**(context or {}), 'operation': operation_name}
        )
        tprint_warning(f"Operation failed, using default return: {operation_name}")
    
    return op_result.result


def create_fallback_result(result_class, **kwargs):
    """Create a fallback result object for failed operations."""
    try:
        # Try to create the result with default values
        return result_class(**kwargs)
    except Exception:
        # If that fails, create a minimal dict-based result
        return {
            'features': {},
            'feature_names': [],
            'scores': {},
            'total_features_generated': 0,
            'execution_time': 0.0,
            'success': False,
            'error': 'Fallback result created due to operation failure'
        }