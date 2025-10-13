"""
Modular Architecture Component

This module provides a modular architecture system inspired by FeatureLookbackOptimizationComponent,
with separate modules for core optimization, validation, error handling, and performance monitoring.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
import logging
import time
from abc import ABC, abstractmethod
from enum import Enum
import traceback
from contextlib import contextmanager

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
        tprint_debug, tprint_performance
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)
    def tprint_performance(*args, **kwargs): print("PERFORMANCE:", *args, **kwargs)

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation levels for input validation."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    EXHAUSTIVE = "exhaustive"


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories."""
    VALIDATION = "validation"
    PROCESSING = "processing"
    PERFORMANCE = "performance"
    MEMORY = "memory"
    CONFIGURATION = "configuration"
    EXTERNAL = "external"


class MetricType(Enum):
    """Types of performance metrics."""
    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    CACHE_HIT_RATE = "cache_hit_rate"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"


class MetricLevel(Enum):
    """Levels of metric aggregation."""
    OPERATION = "operation"
    COMPONENT = "component"
    SYSTEM = "system"


@dataclass
class ValidationResult:
    """Result from input validation."""
    is_valid: bool
    validation_level: ValidationLevel
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]


@dataclass
class ErrorInfo:
    """Information about an error."""
    error_id: str
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    component: str
    timestamp: float
    stack_trace: str
    context: Dict[str, Any]


@dataclass
class PerformanceMetric:
    """Performance metric data."""
    metric_type: MetricType
    metric_level: MetricLevel
    value: float
    unit: str
    timestamp: float
    component: str
    metadata: Dict[str, Any]


class BaseModule(ABC):
    """Base class for all modular components."""
    
    def __init__(self, name: str, logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or logging.getLogger(f"{__name__}.{name}")
        self.performance_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'total_time': 0.0
        }
    
    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """Process method to be implemented by subclasses."""
        pass
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'total_time': 0.0
        }


class InputValidator(BaseModule):
    """Modular input validation component."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__("InputValidator", logger)
        self.validation_rules = self._initialize_validation_rules()
    
    def _initialize_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize validation rules for different data types."""
        return {
            'dataframe': {
                'required_columns': ['close'],
                'min_rows': 10,
                'max_nan_ratio': 0.5,
                'numeric_columns_only': True
            },
            'series': {
                'min_length': 10,
                'max_nan_ratio': 0.5,
                'numeric_only': True
            },
            'periods': {
                'min_value': 1,
                'max_value': 1000,
                'integer_only': True
            }
        }
    
    def validate_dataframe(self, 
                          data: pd.DataFrame, 
                          level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationResult:
        """Validate a DataFrame."""
        errors = []
        warnings = []
        
        try:
            # Basic validation
            if not isinstance(data, pd.DataFrame):
                errors.append("Data must be a pandas DataFrame")
                return ValidationResult(False, level, errors, warnings, {})
            
            if data.empty:
                errors.append("DataFrame cannot be empty")
                return ValidationResult(False, level, errors, warnings, {})
            
            # Check required columns
            required_columns = self.validation_rules['dataframe']['required_columns']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                errors.append(f"Missing required columns: {missing_columns}")
            
            # Check minimum rows
            min_rows = self.validation_rules['dataframe']['min_rows']
            if len(data) < min_rows:
                errors.append(f"DataFrame must have at least {min_rows} rows")
            
            # Check for excessive NaN values
            max_nan_ratio = self.validation_rules['dataframe']['max_nan_ratio']
            for col in data.columns:
                nan_ratio = data[col].isna().sum() / len(data)
                if nan_ratio > max_nan_ratio:
                    warnings.append(f"Column {col} has {nan_ratio:.2%} NaN values")
            
            # Check numeric columns
            if self.validation_rules['dataframe']['numeric_columns_only']:
                non_numeric_cols = [col for col in data.columns 
                                  if not pd.api.types.is_numeric_dtype(data[col])]
                if non_numeric_cols:
                    warnings.append(f"Non-numeric columns found: {non_numeric_cols}")
            
            # Additional validation based on level
            if level in [ValidationLevel.STRICT, ValidationLevel.EXHAUSTIVE]:
                self._validate_dataframe_strict(data, errors, warnings)
            
            is_valid = len(errors) == 0
            
            self.performance_stats['total_operations'] += 1
            if is_valid:
                self.performance_stats['successful_operations'] += 1
            else:
                self.performance_stats['failed_operations'] += 1
            
            return ValidationResult(
                is_valid=is_valid,
                validation_level=level,
                errors=errors,
                warnings=warnings,
                metadata={
                    'shape': data.shape,
                    'columns': list(data.columns),
                    'dtypes': data.dtypes.to_dict()
                }
            )
            
        except Exception as e:
            self.performance_stats['failed_operations'] += 1
            return ValidationResult(
                is_valid=False,
                validation_level=level,
                errors=[f"Validation error: {str(e)}"],
                warnings=warnings,
                metadata={}
            )
    
    def _validate_dataframe_strict(self, data: pd.DataFrame, errors: List[str], warnings: List[str]):
        """Additional strict validation for DataFrame."""
        # Check for constant columns
        constant_cols = [col for col in data.columns if data[col].nunique() <= 1]
        if constant_cols:
            warnings.append(f"Constant columns found: {constant_cols}")
        
        # Check for infinite values
        inf_cols = []
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                if np.isinf(data[col]).any():
                    inf_cols.append(col)
        if inf_cols:
            warnings.append(f"Columns with infinite values: {inf_cols}")
    
    def validate_series(self, 
                       series: pd.Series, 
                       level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationResult:
        """Validate a pandas Series."""
        errors = []
        warnings = []
        
        try:
            if not isinstance(series, pd.Series):
                errors.append("Data must be a pandas Series")
                return ValidationResult(False, level, errors, warnings, {})
            
            if len(series) < self.validation_rules['series']['min_length']:
                errors.append(f"Series must have at least {self.validation_rules['series']['min_length']} values")
            
            nan_ratio = series.isna().sum() / len(series)
            if nan_ratio > self.validation_rules['series']['max_nan_ratio']:
                warnings.append(f"Series has {nan_ratio:.2%} NaN values")
            
            if self.validation_rules['series']['numeric_only'] and not pd.api.types.is_numeric_dtype(series):
                errors.append("Series must be numeric")
            
            is_valid = len(errors) == 0
            
            self.performance_stats['total_operations'] += 1
            if is_valid:
                self.performance_stats['successful_operations'] += 1
            else:
                self.performance_stats['failed_operations'] += 1
            
            return ValidationResult(
                is_valid=is_valid,
                validation_level=level,
                errors=errors,
                warnings=warnings,
                metadata={
                    'length': len(series),
                    'dtype': str(series.dtype),
                    'nunique': series.nunique()
                }
            )
            
        except Exception as e:
            self.performance_stats['failed_operations'] += 1
            return ValidationResult(
                is_valid=False,
                validation_level=level,
                errors=[f"Series validation error: {str(e)}"],
                warnings=warnings,
                metadata={}
            )
    
    def process(self, data: Union[pd.DataFrame, pd.Series], 
                level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationResult:
        """Process validation for data."""
        if isinstance(data, pd.DataFrame):
            return self.validate_dataframe(data, level)
        elif isinstance(data, pd.Series):
            return self.validate_series(data, level)
        else:
            return ValidationResult(
                is_valid=False,
                validation_level=level,
                errors=["Unsupported data type for validation"],
                warnings=[],
                metadata={}
            )


class ErrorHandler(BaseModule):
    """Modular error handling component."""
    
    def __init__(self, component_name: str, logger: Optional[logging.Logger] = None):
        super().__init__("ErrorHandler", logger)
        self.component_name = component_name
        self.error_history = []
        self.error_counts = {}
    
    def handle_error(self, 
                    error: Exception, 
                    context: Dict[str, Any] = None,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    category: ErrorCategory = ErrorCategory.PROCESSING) -> ErrorInfo:
        """Handle an error and return error information."""
        error_id = f"{self.component_name}_{int(time.time() * 1000)}"
        
        error_info = ErrorInfo(
            error_id=error_id,
            severity=severity,
            category=category,
            message=str(error),
            component=self.component_name,
            timestamp=time.time(),
            stack_trace=traceback.format_exc(),
            context=context or {}
        )
        
        # Log error
        self.logger.error(f"Error {error_id}: {error_info.message}")
        if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self.logger.error(f"Stack trace: {error_info.stack_trace}")
        
        # Track error
        self.error_history.append(error_info)
        error_key = f"{category.value}_{severity.value}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        self.performance_stats['total_operations'] += 1
        self.performance_stats['failed_operations'] += 1
        
        return error_info
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors."""
        return {
            'total_errors': len(self.error_history),
            'error_counts': self.error_counts.copy(),
            'recent_errors': self.error_history[-10:] if self.error_history else [],
            'critical_errors': [e for e in self.error_history if e.severity == ErrorSeverity.CRITICAL]
        }
    
    def process(self, error: Exception, **kwargs) -> ErrorInfo:
        """Process an error."""
        return self.handle_error(error, **kwargs)


class PerformanceMonitor(BaseModule):
    """Modular performance monitoring component."""
    
    def __init__(self, component_name: str, logger: Optional[logging.Logger] = None):
        super().__init__("PerformanceMonitor", logger)
        self.component_name = component_name
        self.metrics = []
        self.start_times = {}
    
    @contextmanager
    def monitor_operation(self, operation_name: str, metric_type: MetricType = MetricType.EXECUTION_TIME):
        """Context manager for monitoring operations."""
        start_time = time.time()
        self.start_times[operation_name] = start_time
        
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            metric = PerformanceMetric(
                metric_type=metric_type,
                metric_level=MetricLevel.OPERATION,
                value=duration,
                unit="seconds",
                timestamp=end_time,
                component=self.component_name,
                metadata={'operation': operation_name}
            )
            
            self.metrics.append(metric)
            self.start_times.pop(operation_name, None)
    
    def record_metric(self, 
                     metric_type: MetricType, 
                     value: float, 
                     unit: str = "",
                     metadata: Dict[str, Any] = None):
        """Record a performance metric."""
        metric = PerformanceMetric(
            metric_type=metric_type,
            metric_level=MetricLevel.OPERATION,
            value=value,
            unit=unit,
            timestamp=time.time(),
            component=self.component_name,
            metadata=metadata or {}
        )
        
        self.metrics.append(metric)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.metrics:
            return {}
        
        # Group metrics by type
        metrics_by_type = {}
        for metric in self.metrics:
            metric_type = metric.metric_type.value
            if metric_type not in metrics_by_type:
                metrics_by_type[metric_type] = []
            metrics_by_type[metric_type].append(metric.value)
        
        # Calculate statistics
        summary = {}
        for metric_type, values in metrics_by_type.items():
            summary[metric_type] = {
                'count': len(values),
                'mean': np.mean(values),
                'min': np.min(values),
                'max': np.max(values),
                'std': np.std(values)
            }
        
        return summary
    
    def process(self, operation: Callable, *args, **kwargs) -> Any:
        """Process an operation with monitoring."""
        with self.monitor_operation(operation.__name__):
            return operation(*args, **kwargs)


class CoreOptimizer(BaseModule):
    """Modular core optimization component."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__("CoreOptimizer", logger)
        self.optimization_history = []
    
    def optimize_parameters(self, 
                           objective_function: Callable,
                           parameter_space: Dict[str, Tuple[float, float]],
                           max_iterations: int = 100) -> Dict[str, Any]:
        """Optimize parameters using a simple grid search."""
        start_time = time.time()
        
        try:
            best_params = None
            best_score = float('-inf')
            
            # Simple grid search (in practice, you'd use more sophisticated methods)
            param_names = list(parameter_space.keys())
            param_ranges = list(parameter_space.values())
            
            # Generate parameter combinations
            param_combinations = self._generate_param_combinations(param_ranges, max_iterations)
            
            for params in param_combinations:
                try:
                    param_dict = dict(zip(param_names, params))
                    score = objective_function(**param_dict)
                    
                    if score > best_score:
                        best_score = score
                        best_params = param_dict
                        
                except Exception as e:
                    self.logger.warning(f"Parameter evaluation failed: {e}")
                    continue
            
            optimization_result = {
                'best_params': best_params,
                'best_score': best_score,
                'total_evaluations': len(param_combinations),
                'optimization_time': time.time() - start_time
            }
            
            self.optimization_history.append(optimization_result)
            
            self.performance_stats['total_operations'] += 1
            self.performance_stats['successful_operations'] += 1
            self.performance_stats['total_time'] += time.time() - start_time
            
            return optimization_result
            
        except Exception as e:
            self.performance_stats['failed_operations'] += 1
            self.logger.error(f"Optimization failed: {e}")
            return {
                'best_params': None,
                'best_score': float('-inf'),
                'total_evaluations': 0,
                'optimization_time': time.time() - start_time,
                'error': str(e)
            }
    
    def _generate_param_combinations(self, 
                                   param_ranges: List[Tuple[float, float]], 
                                   max_combinations: int) -> List[List[float]]:
        """Generate parameter combinations for optimization."""
        # Simple grid search implementation
        combinations = []
        
        # Calculate grid size
        n_params = len(param_ranges)
        grid_size = int(max_combinations ** (1/n_params))
        
        for i in range(min(grid_size ** n_params, max_combinations)):
            combination = []
            temp = i
            for min_val, max_val in param_ranges:
                grid_index = temp % grid_size
                param_value = min_val + (max_val - min_val) * grid_index / (grid_size - 1)
                combination.append(param_value)
                temp //= grid_size
            combinations.append(combination)
        
        return combinations
    
    def process(self, objective_function: Callable, **kwargs) -> Dict[str, Any]:
        """Process optimization."""
        return self.optimize_parameters(objective_function, **kwargs)


class ModularArchitecture:
    """Main modular architecture coordinator."""
    
    def __init__(self, component_name: str, logger: Optional[logging.Logger] = None):
        self.component_name = component_name
        self.logger = logger or logging.getLogger(f"{__name__}.{component_name}")
        
        # Initialize modular components
        self.validator = InputValidator(self.logger)
        self.error_handler = ErrorHandler(component_name, self.logger)
        self.performance_monitor = PerformanceMonitor(component_name, self.logger)
        self.core_optimizer = CoreOptimizer(self.logger)
        
        tprint_info(f"🏗️ Modular architecture initialized for {component_name}")
    
    def validate_inputs(self, data: Union[pd.DataFrame, pd.Series], 
                       level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationResult:
        """Validate inputs using the modular validator."""
        return self.validator.process(data, level)
    
    def handle_error(self, error: Exception, **kwargs) -> ErrorInfo:
        """Handle errors using the modular error handler."""
        return self.error_handler.handle_error(error, **kwargs)
    
    def monitor_operation(self, operation: Callable, *args, **kwargs) -> Any:
        """Monitor operations using the modular performance monitor."""
        return self.performance_monitor.process(operation, *args, **kwargs)
    
    def optimize_parameters(self, objective_function: Callable, **kwargs) -> Dict[str, Any]:
        """Optimize parameters using the modular optimizer."""
        return self.core_optimizer.process(objective_function, **kwargs)
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get comprehensive system summary."""
        return {
            'component_name': self.component_name,
            'validator_stats': self.validator.get_performance_stats(),
            'error_handler_stats': self.error_handler.get_error_summary(),
            'performance_monitor_stats': self.performance_monitor.get_performance_summary(),
            'core_optimizer_stats': self.core_optimizer.get_performance_stats()
        }


# Convenience functions
def create_modular_architecture(component_name: str, 
                               logger: Optional[logging.Logger] = None) -> ModularArchitecture:
    """Create a modular architecture instance."""
    return ModularArchitecture(component_name, logger)


# Export main classes and functions
__all__ = [
    'ModularArchitecture',
    'InputValidator',
    'ErrorHandler',
    'PerformanceMonitor',
    'CoreOptimizer',
    'ValidationLevel',
    'ErrorSeverity',
    'ErrorCategory',
    'MetricType',
    'MetricLevel',
    'ValidationResult',
    'ErrorInfo',
    'PerformanceMetric',
    'create_modular_architecture'
]