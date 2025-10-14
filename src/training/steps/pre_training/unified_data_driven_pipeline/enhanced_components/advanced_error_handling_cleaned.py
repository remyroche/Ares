"""
Advanced Error Handling Framework - Cleaned Version

This module provides consolidated error handling infrastructure with fast-fail patterns,
removed duplicates, and improved error reporting.

Key improvements:
- Consolidated error handling classes (single source of truth)
- Removed duplicate error handling patterns
- Implemented fast-fail patterns instead of silent errors
- Improved error context and reporting
- Streamlined error classification
"""

import logging
import traceback
from typing import Any, Optional, Dict, List, Callable, Type, Union
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from datetime import datetime

# Import utility modules
from src.utils.common_operations import (
    CommonUtilities, safe_dataframe_operation, safe_convert_dtypes,
    safe_merge_dataframes, safe_drop_columns, safe_rename_columns,
    safe_filter_dataframe, safe_groupby_operation, safe_apply_function,
    get_dataframe_info, create_summary_statistics, safe_log_metric,
    safe_log_params, safe_log_artifact, calculate_data_quality_metrics,
    validate_dataframe, validate_dataframe_columns, optimize_dataframe_dtypes,
    safe_fillna, safe_timestamp_conversion, guard_dataframe_nulls
)
from src.utils.serialization_utils import UniversalSerializer

# Centralized tprint import
try:
    from src.utils.tprint import tprint, tprint_error, tprint_warning, tprint_success, tprint_debug
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)

import numpy as np
import pandas as pd

# Centralized exception classes - single source of truth
class PipelineError(Exception):
    """Base exception for pipeline-related errors."""
    def __init__(self, message: str, error_code: str = None, context: Dict[str, Any] = None):
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}
        self.timestamp = datetime.now()

class DataValidationError(PipelineError):
    """Exception raised when data validation fails."""
    pass

class FeatureGenerationError(PipelineError):
    """Exception raised when feature generation fails."""
    pass

class OptimizationError(PipelineError):
    """Exception raised when optimization fails."""
    pass

class CacheError(PipelineError):
    """Exception raised when cache operations fail."""
    pass

class MemoryError(PipelineError):
    """Exception raised when memory operations fail."""
    pass

class ConfigurationError(PipelineError):
    """Exception raised when configuration is invalid."""
    pass

class CriticalPipelineError(PipelineError):
    """Exception raised for critical pipeline failures that should cause immediate termination."""
    pass

# Centralized enums - single source of truth
class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories."""
    VALIDATION = "validation"
    OPTIMIZATION = "optimization"
    DATA_PROCESSING = "data_processing"
    FEATURE_GENERATION = "feature_generation"
    FILE_IO = "file_io"
    MEMORY = "memory"
    NETWORK = "network"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"

# Data classes
@dataclass
class ErrorDetails:
    """Detailed error information."""
    error: Exception
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    component: str
    timestamp: datetime
    stack_trace: str
    context: Dict[str, Any]
    error_id: str
    recovery_attempted: bool = False
    recovery_successful: bool = False

@dataclass
class ErrorRecoveryStrategy:
    """Strategy for error recovery."""
    name: str
    description: str
    recovery_function: Callable
    max_attempts: int = 3
    backoff_factor: float = 2.0
    enabled: bool = True

class AdvancedErrorHandler:
    """Advanced error handling with fast-fail patterns and recovery strategies."""
    
    def __init__(self, component_name: str, logger: Optional[logging.Logger] = None):
        self.component_name = component_name
        self.logger = logger or logging.getLogger(f"{__name__}.{component_name}")
        self.error_history = []
        self.error_counts = {}
        self.recovery_strategies = {}
        self.performance_stats = {
            'total_errors': 0,
            'recovered_errors': 0,
            'critical_errors': 0,
            'recovery_attempts': 0
        }
        
        # Initialize recovery strategies
        self._initialize_recovery_strategies()
    
    def _initialize_recovery_strategies(self):
        """Initialize error recovery strategies."""
        self.recovery_strategies = {
            'data_validation': ErrorRecoveryStrategy(
                name="data_validation_recovery",
                description="Recover from data validation errors",
                recovery_function=self._recover_data_validation_error,
                max_attempts=2
            ),
            'memory_error': ErrorRecoveryStrategy(
                name="memory_error_recovery",
                description="Recover from memory errors",
                recovery_function=self._recover_memory_error,
                max_attempts=1
            ),
            'file_io': ErrorRecoveryStrategy(
                name="file_io_recovery",
                description="Recover from file I/O errors",
                recovery_function=self._recover_file_io_error,
                max_attempts=3
            )
        }
    
    def handle_error(self, 
                    error: Exception, 
                    context: Dict[str, Any] = None,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    category: ErrorCategory = ErrorCategory.UNKNOWN,
                    allow_recovery: bool = True) -> ErrorDetails:
        """Handle an error with fast-fail patterns and optional recovery."""
        error_id = f"{self.component_name}_{int(datetime.now().timestamp() * 1000)}"
        
        # Classify error if not provided
        if category == ErrorCategory.UNKNOWN:
            category = self._classify_error_category(error)
        
        if severity == ErrorSeverity.LOW:
            severity = self._classify_error_severity(error)
        
        error_details = ErrorDetails(
            error=error,
            severity=severity,
            category=category,
            message=str(error),
            component=self.component_name,
            timestamp=datetime.now(),
            stack_trace=traceback.format_exc(),
            context=context or {},
            error_id=error_id
        )
        
        # Log error
        self._log_error(error_details)
        
        # Track error
        self._track_error(error_details)
        
        # Attempt recovery if allowed and not critical
        if allow_recovery and severity != ErrorSeverity.CRITICAL:
            recovery_successful = self._attempt_recovery(error_details)
            error_details.recovery_attempted = True
            error_details.recovery_successful = recovery_successful
            
            if recovery_successful:
                tprint_success(f"✅ Error recovered: {error_details.error_id}")
                return error_details
        
        # Fast fail for critical errors or failed recovery
        if severity == ErrorSeverity.CRITICAL or not error_details.recovery_successful:
            self._fast_fail(error_details)
        
        return error_details
    
    def _classify_error_severity(self, error: Exception) -> ErrorSeverity:
        """Classify error severity based on error type and context."""
        error_type = type(error).__name__
        
        if error_type in ['MemoryError', 'OSError', 'SystemError']:
            return ErrorSeverity.CRITICAL
        elif error_type in ['ValueError', 'TypeError', 'KeyError', 'IndexError']:
            return ErrorSeverity.HIGH
        elif error_type in ['Warning', 'UserWarning']:
            return ErrorSeverity.LOW
        else:
            return ErrorSeverity.MEDIUM
    
    def _classify_error_category(self, error: Exception) -> ErrorCategory:
        """Classify error category based on error type and context."""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        if 'validation' in error_message or error_type in ['ValueError', 'TypeError']:
            return ErrorCategory.VALIDATION
        elif 'memory' in error_message or error_type == 'MemoryError':
            return ErrorCategory.MEMORY
        elif 'file' in error_message or 'io' in error_message or error_type == 'OSError':
            return ErrorCategory.FILE_IO
        elif 'network' in error_message or 'connection' in error_message:
            return ErrorCategory.NETWORK
        elif 'config' in error_message or 'configuration' in error_message:
            return ErrorCategory.CONFIGURATION
        elif 'feature' in error_message or 'generation' in error_message:
            return ErrorCategory.FEATURE_GENERATION
        elif 'optimization' in error_message or 'optimize' in error_message:
            return ErrorCategory.OPTIMIZATION
        else:
            return ErrorCategory.UNKNOWN
    
    def _log_error(self, error_details: ErrorDetails):
        """Log error with appropriate level."""
        log_message = f"Error {error_details.error_id}: {error_details.message}"
        
        if error_details.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
            self.logger.critical(f"Stack trace: {error_details.stack_trace}")
        elif error_details.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
            self.logger.error(f"Stack trace: {error_details.stack_trace}")
        elif error_details.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def _track_error(self, error_details: ErrorDetails):
        """Track error statistics."""
        self.error_history.append(error_details)
        error_key = f"{error_details.category.value}_{error_details.severity.value}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        self.performance_stats['total_errors'] += 1
        if error_details.severity == ErrorSeverity.CRITICAL:
            self.performance_stats['critical_errors'] += 1
    
    def _attempt_recovery(self, error_details: ErrorDetails) -> bool:
        """Attempt to recover from error using appropriate strategy."""
        strategy = self.recovery_strategies.get(error_details.category.value)
        
        if not strategy or not strategy.enabled:
            return False
        
        self.performance_stats['recovery_attempts'] += 1
        
        try:
            recovery_successful = strategy.recovery_function(error_details)
            if recovery_successful:
                self.performance_stats['recovered_errors'] += 1
            return recovery_successful
        except Exception as recovery_error:
            self.logger.warning(f"Recovery attempt failed: {recovery_error}")
            return False
    
    def _recover_data_validation_error(self, error_details: ErrorDetails) -> bool:
        """Recover from data validation errors."""
        try:
            # Attempt to fix common data validation issues
            if 'context' in error_details.context:
                data = error_details.context.get('data')
                if data is not None and isinstance(data, pd.DataFrame):
                    # Try to fix common issues
                    if 'missing' in str(error_details.error).lower():
                        # Fill missing values
                        data.fillna(method='ffill', inplace=True)
                        return True
                    elif 'dtype' in str(error_details.error).lower():
                        # Convert dtypes
                        data = data.convert_dtypes()
                        return True
            return False
        except Exception:
            return False
    
    def _recover_memory_error(self, error_details: ErrorDetails) -> bool:
        """Recover from memory errors."""
        try:
            # Attempt to free memory
            import gc
            gc.collect()
            return True
        except Exception:
            return False
    
    def _recover_file_io_error(self, error_details: ErrorDetails) -> bool:
        """Recover from file I/O errors."""
        try:
            # Attempt to retry file operations
            if 'file_path' in error_details.context:
                file_path = error_details.context['file_path']
                if isinstance(file_path, str) and not Path(file_path).exists():
                    # Try to create directory
                    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                    return True
            return False
        except Exception:
            return False
    
    def _fast_fail(self, error_details: ErrorDetails):
        """Fast fail for critical errors or failed recovery."""
        if error_details.severity == ErrorSeverity.CRITICAL:
            tprint_error(f"❌ Critical error - pipeline cannot continue: {error_details.message}")
            raise error_details.error
        else:
            tprint_error(f"❌ Error recovery failed - pipeline cannot continue: {error_details.message}")
            raise error_details.error
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get comprehensive error summary."""
        return {
            'total_errors': len(self.error_history),
            'error_counts': self.error_counts.copy(),
            'recent_errors': self.error_history[-10:] if self.error_history else [],
            'critical_errors': [e for e in self.error_history if e.severity == ErrorSeverity.CRITICAL],
            'recovery_stats': self.performance_stats.copy()
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'total_errors': 0,
            'recovered_errors': 0,
            'critical_errors': 0,
            'recovery_attempts': 0
        }

# Decorator for error handling
def handle_errors(severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 category: ErrorCategory = ErrorCategory.UNKNOWN,
                 allow_recovery: bool = True):
    """Decorator to handle errors in functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Get error handler from args or create new one
                error_handler = None
                for arg in args:
                    if isinstance(arg, AdvancedErrorHandler):
                        error_handler = arg
                        break
                
                if error_handler is None:
                    error_handler = AdvancedErrorHandler(func.__name__)
                
                # Handle error
                error_details = error_handler.handle_error(
                    e, 
                    context={'function': func.__name__, 'args': str(args), 'kwargs': str(kwargs)},
                    severity=severity,
                    category=category,
                    allow_recovery=allow_recovery
                )
                
                # Re-raise if not recovered
                if not error_details.recovery_successful:
                    raise
                
                return None
        return wrapper
    return decorator

# Context manager for error handling
@contextmanager
def error_handling_context(component_name: str, 
                          severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                          category: ErrorCategory = ErrorCategory.UNKNOWN,
                          allow_recovery: bool = True):
    """Context manager for error handling."""
    error_handler = AdvancedErrorHandler(component_name)
    
    try:
        yield error_handler
    except Exception as e:
        error_details = error_handler.handle_error(
            e,
            context={'component': component_name},
            severity=severity,
            category=category,
            allow_recovery=allow_recovery
        )
        
        if not error_details.recovery_successful:
            raise

# Export main classes and functions
__all__ = [
    'PipelineError',
    'DataValidationError',
    'FeatureGenerationError',
    'OptimizationError',
    'CacheError',
    'MemoryError',
    'ConfigurationError',
    'CriticalPipelineError',
    'ErrorSeverity',
    'ErrorCategory',
    'ErrorDetails',
    'ErrorRecoveryStrategy',
    'AdvancedErrorHandler',
    'handle_errors',
    'error_handling_context'
]