"""
Step06 Enhanced Validation Framework

This module provides comprehensive validation framework for step06 operations
including function validation, tracking, and context management.
"""

import logging
import time
import warnings
from typing import Any, Dict, List, Optional, Callable, Union
from functools import wraps
from enum import Enum
from dataclasses import dataclass, field

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

except ImportError:
    
    cp = None

# Setup logging
logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Validation levels for step06 operations."""
    BASIC = "basic"
    STANDARD = "standard"
    ENHANCED = "enhanced"
    STRICT = "strict"
    COMPREHENSIVE = "comprehensive"
    DETAILED = "detailed"

class FunctionStatus(Enum):
    """Function execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class ValidationContext:
    """Context for validation operations."""
    function_name: str
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    status: FunctionStatus = FunctionStatus.PENDING
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class Step06FunctionTracker:
    """Tracks function execution for step06 operations."""
    
    def __init__(self):
        self.functions: Dict[str, ValidationContext] = {}
        self.execution_history: List[ValidationContext] = []
    
    def start_function(self, function_name: str, validation_level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationContext:
        """Start tracking a function."""
        context = ValidationContext(
            function_name=function_name,
            validation_level=validation_level,
            start_time=time.time(),
            status=FunctionStatus.RUNNING
        )
        self.functions[function_name] = context
        return context
    
    def complete_function(self, function_name: str, errors: List[str] = None, warnings: List[str] = None, metadata: Dict[str, Any] = None):
        """Complete function tracking."""
        if function_name in self.functions:
            context = self.functions[function_name]
            context.end_time = time.time()
            context.status = FunctionStatus.COMPLETED
            if errors:
                context.errors.extend(errors)
            if warnings:
                context.warnings.extend(warnings)
            if metadata:
                context.metadata.update(metadata)
            
            self.execution_history.append(context)
    
    def fail_function(self, function_name: str, error: str, metadata: Dict[str, Any] = None):
        """Mark function as failed."""
        if function_name in self.functions:
            context = self.functions[function_name]
            context.end_time = time.time()
            context.status = FunctionStatus.FAILED
            context.errors.append(error)
            if metadata:
                context.metadata.update(metadata)
            
            self.execution_history.append(context)
    
    def get_function_status(self, function_name: str) -> Optional[FunctionStatus]:
        """Get function status."""
        if function_name in self.functions:
            return self.functions[function_name].status
        return None
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary."""
        total_functions = len(self.execution_history)
        completed = len([f for f in self.execution_history if f.status == FunctionStatus.COMPLETED])
        failed = len([f for f in self.execution_history if f.status == FunctionStatus.FAILED])
        
        return {
            'total_functions': total_functions,
            'completed': completed,
            'failed': failed,
            'success_rate': completed / total_functions if total_functions > 0 else 0,
            'execution_time': sum(f.end_time - f.start_time for f in self.execution_history if f.end_time and f.start_time)
        }

# Global tracker instance
step06_function_tracker = Step06FunctionTracker()

def step06_function_validator(validation_level: ValidationLevel = ValidationLevel.STANDARD):
    """Decorator for step06 function validation."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            function_name = func.__name__
            context = step06_function_tracker.start_function(function_name, validation_level)
            
            try:
                logger.info(f"Starting {function_name} with validation level {validation_level.value}")
                result = func(*args, **kwargs)
                
                step06_function_tracker.complete_function(function_name)
                logger.info(f"Completed {function_name} successfully")
                return result
                
            except Exception as e:
                error_msg = f"Error in {function_name}: {str(e)}"
                step06_function_tracker.fail_function(function_name, error_msg)
                logger.error(error_msg)
                raise
        
        return wrapper
    return decorator

def step06_validation_context(function_name: str, validation_level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationContext:
    """Create validation context for step06 operations."""
    return step06_function_tracker.start_function(function_name, validation_level)

def get_step06_validation_summary() -> Dict[str, Any]:
    """Get step06 validation summary."""
    return step06_function_tracker.get_execution_summary()

# Additional validation utilities
def validate_step06_inputs(data: Any, required_type: type = None, min_size: int = None) -> bool:
    """Validate step06 inputs."""
    try:
        if required_type and not isinstance(data, required_type):
            raise ValueError(f"Expected {required_type}, got {type(data)}")
        
        if min_size is not None:
            if hasattr(data, '__len__') and len(data) < min_size:
                raise ValueError(f"Expected minimum size {min_size}, got {len(data)}")
        
        return True
    except Exception as e:
        logger.error(f"Step06 input validation failed: {e}")
        return False

def validate_step06_outputs(data: Any, expected_type: type = None, min_size: int = None) -> bool:
    """Validate step06 outputs."""
    try:
        if expected_type and not isinstance(data, expected_type):
            raise ValueError(f"Expected {expected_type}, got {type(data)}")
        
        if min_size is not None:
            if hasattr(data, '__len__') and len(data) < min_size:
                raise ValueError(f"Expected minimum size {min_size}, got {len(data)}")
        
        return True
    except Exception as e:
        logger.error(f"Step06 output validation failed: {e}")
        return False

def step06_performance_monitor(func: Callable) -> Callable:
    """Monitor step06 function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        logger.info(f"Step06 function {func.__name__} executed in {execution_time:.2f} seconds")
        
        return result
    return wrapper

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
