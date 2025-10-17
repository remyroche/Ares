"""
Modular Architecture Component

This module provides a modular architecture system inspired by FeatureLookbackOptimizationComponent,
with separate modules for core optimization, validation, error handling, and performance monitoring.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable, Type
from dataclasses import dataclass
import logging
import time
from abc import ABC, abstractmethod
from enum import Enum
import traceback
from contextlib import contextmanager

# Import utility modules
from src.utils.common_utilities import (
    safe_dataframe_operation, validate_dataframe_columns,
    analyze_nan_values_detailed, calculate_data_quality_metrics,
    create_data_quality_report, get_dataframe_info, create_summary_statistics,
    safe_convert_dtypes, safe_merge_dataframes, safe_drop_columns,
    safe_rename_columns, safe_filter_dataframe, safe_groupby_operation,
    safe_apply_function
)

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

class ModularComponent(ABC):
    """
    Abstract base class for modular components in the unified data-driven pipeline.
    
    This class provides a standardized interface for creating modular, reusable components
    that can be composed together in the data processing pipeline. Each component follows
    a consistent lifecycle and provides comprehensive functionality for:
    
    - Initialization and cleanup
    - Input validation
    - Data processing
    - Configuration management
    - State management
    - Performance monitoring
    - Serialization and persistence
    
    Subclasses must implement the abstract methods to define their specific behavior.
    
    Example:
        class MyComponent(ModularComponent):
            def initialize(self) -> bool:
                # Initialize component resources
                return True
            
            def process(self, data: Any, **kwargs) -> Any:
                # Process the input data
                return processed_data
            
            # ... implement other abstract methods
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        """Initialize the modular component."""
        self.name = name
        self.config = config or {}
        self.logger = logger or logging.getLogger(f"{__name__}.{name}")
        self.performance_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'total_time': 0.0
        }
        self._initialized = False
        self._state = {}

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the component. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def process(self, data: Any, **kwargs) -> Any:
        """Process the input data. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def validate_input(self, data: Any) -> ValidationResult:
        """Validate input data. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def get_component_info(self) -> Dict[str, Any]:
        """Get component information. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def get_dependencies(self) -> List[str]:
        """Get list of component dependencies. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def get_output_schema(self) -> Dict[str, Any]:
        """Get expected output schema. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def get_required_config(self) -> List[str]:
        """Get list of required configuration parameters. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def can_process(self, data: Any) -> bool:
        """Check if component can process the given data. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def get_processing_capabilities(self) -> Dict[str, Any]:
        """Get component processing capabilities. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def estimate_processing_time(self, data: Any) -> float:
        """Estimate processing time for given data. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def get_memory_requirements(self, data: Any) -> Dict[str, Any]:
        """Get memory requirements for processing data. Must be implemented by subclasses."""
        pass

    def is_initialized(self) -> bool:
        """Check if component is initialized."""
        return self._initialized

    def set_state(self, key: str, value: Any) -> None:
        """Set component state."""
        self._state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get component state."""
        return self._state.get(key, default)

    def get_all_state(self) -> Dict[str, Any]:
        """Get all component state."""
        return self._state.copy()

    def clear_state(self) -> None:
        """Clear component state."""
        self._state.clear()

    def update_config(self, config: Dict[str, Any]) -> None:
        """Update component configuration."""
        self.config.update(config)

    def get_config(self, key: str = None, default: Any = None) -> Any:
        """Get configuration value."""
        if key is None:
            return self.config.copy()
        return self.config.get(key, default)

    def validate_config(self) -> bool:
        """Validate component configuration."""
        required_config = self.get_required_config()
        missing_config = [key for key in required_config if key not in self.config]
        if missing_config:
            self.logger.error(f"Missing required configuration: {missing_config}")
            return False
        return True

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()

    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self.performance_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'total_time': 0.0
        }

    def get_status(self) -> Dict[str, Any]:
        """Get component status."""
        return {
            'name': self.name,
            'initialized': self._initialized,
            'config': self.config,
            'performance_stats': self.performance_stats,
            'state': self._state
        }

    def serialize(self) -> Dict[str, Any]:
        """Serialize component for persistence."""
        return {
            'name': self.name,
            'config': self.config,
            'state': self._state,
            'performance_stats': self.performance_stats
        }

    def deserialize(self, data: Dict[str, Any]) -> None:
        """Deserialize component from persisted data."""
        self.config = data.get('config', {})
        self._state = data.get('state', {})
        self.performance_stats = data.get('performance_stats', {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'total_time': 0.0
        })

    def _update_performance_stats(self, success: bool, processing_time: float) -> None:
        """Update performance statistics."""
        self.performance_stats['total_operations'] += 1
        if success:
            self.performance_stats['successful_operations'] += 1
        else:
            self.performance_stats['failed_operations'] += 1
        self.performance_stats['total_time'] += processing_time

    def _log_operation(self, operation: str, success: bool, processing_time: float) -> None:
        """Log operation details."""
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(f"Operation '{operation}' {status} in {processing_time:.4f}s")

    def _validate_dependencies(self, dependencies: List[str]) -> bool:
        """Validate that all dependencies are available."""
        # This is a placeholder - in a real implementation, you'd check
        # if the dependencies are actually available in the system
        return True

    def _check_memory_usage(self, data: Any) -> bool:
        """Check if there's enough memory to process the data."""
        # This is a placeholder - in a real implementation, you'd check
        # actual memory usage and available memory
        return True

    def _safe_process(self, data: Any, **kwargs) -> Any:
        """Safely process data with error handling and performance tracking."""
        start_time = time.time()
        success = False
        result = None
        
        try:
            # Validate input
            validation_result = self.validate_input(data)
            if not validation_result.is_valid:
                raise ValueError(f"Input validation failed: {validation_result.errors}")
            
            # Check if component can process the data
            if not self.can_process(data):
                raise ValueError(f"Component {self.name} cannot process the given data")
            
            # Check memory requirements
            if not self._check_memory_usage(data):
                raise MemoryError(f"Insufficient memory to process data in component {self.name}")
            
            # Process the data
            result = self.process(data, **kwargs)
            success = True
            
        except Exception as e:
            self.logger.error(f"Error in component {self.name}: {str(e)}")
            raise
        finally:
            processing_time = time.time() - start_time
            self._update_performance_stats(success, processing_time)
            self._log_operation("process", success, processing_time)
        
        return result

class ExampleModularComponent(ModularComponent):
    """Example implementation of ModularComponent for demonstration purposes."""

    def __init__(self, name: str = "example_component", config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(name, config, logger)
        self.processing_window = self.get_config('processing_window', 20)
        self.threshold = self.get_config('threshold', 0.5)

    def initialize(self) -> bool:
        """Initialize the example component."""
        try:
            if not self.validate_config():
                return False
            
            self._initialized = True
            self.logger.info(f"Component {self.name} initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize component {self.name}: {e}")
            return False

    def process(self, data: Any, **kwargs) -> Any:
        """Process the input data."""
        if isinstance(data, pd.DataFrame):
            # Example: Calculate rolling mean
            if 'close' in data.columns:
                result = data['close'].rolling(window=self.processing_window).mean()
                return result
            else:
                raise ValueError("DataFrame must contain 'close' column")
        else:
            raise ValueError("Data must be a pandas DataFrame")

    def validate_input(self, data: Any) -> ValidationResult:
        """Validate input data."""
        errors = []
        warnings = []
        
        if not isinstance(data, pd.DataFrame):
            errors.append("Data must be a pandas DataFrame")
            return ValidationResult(False, ValidationLevel.STANDARD, errors, warnings, {})
        
        if len(data) < self.processing_window:
            errors.append(f"Data must have at least {self.processing_window} rows")
        
        if 'close' not in data.columns:
            errors.append("DataFrame must contain 'close' column")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            validation_level=ValidationLevel.STANDARD,
            errors=errors,
            warnings=warnings,
            metadata={'shape': data.shape, 'columns': list(data.columns)}
        )

    def cleanup(self) -> None:
        """Cleanup resources."""
        self._initialized = False
        self.clear_state()
        self.logger.info(f"Component {self.name} cleaned up")

    def get_component_info(self) -> Dict[str, Any]:
        """Get component information."""
        return {
            'name': self.name,
            'type': 'example_component',
            'version': '1.0.0',
            'description': 'Example modular component for demonstration',
            'processing_window': self.processing_window,
            'threshold': self.threshold
        }

    def get_dependencies(self) -> List[str]:
        """Get list of component dependencies."""
        return ['pandas', 'numpy']

    def get_output_schema(self) -> Dict[str, Any]:
        """Get expected output schema."""
        return {
            'type': 'pandas.Series',
            'index_type': 'DatetimeIndex',
            'dtype': 'float64',
            'description': 'Rolling mean of close prices'
        }

    def get_required_config(self) -> List[str]:
        """Get list of required configuration parameters."""
        return ['processing_window', 'threshold']

    def can_process(self, data: Any) -> bool:
        """Check if component can process the given data."""
        return isinstance(data, pd.DataFrame) and 'close' in data.columns

    def get_processing_capabilities(self) -> Dict[str, Any]:
        """Get component processing capabilities."""
        return {
            'input_types': ['pandas.DataFrame'],
            'required_columns': ['close'],
            'output_type': 'pandas.Series',
            'supports_parallel': True,
            'memory_efficient': True
        }

    def estimate_processing_time(self, data: Any) -> float:
        """Estimate processing time for given data."""
        if isinstance(data, pd.DataFrame):
            # Simple estimation based on data size
            return len(data) * 0.0001  # 0.1ms per row
        return 0.0

    def get_memory_requirements(self, data: Any) -> Dict[str, Any]:
        """Get memory requirements for processing data."""
        if isinstance(data, pd.DataFrame):
            estimated_memory = len(data) * 8 * 1.5  # Rough estimate
            return {
                'estimated_memory_mb': estimated_memory / (1024 * 1024),
                'peak_memory_mb': estimated_memory * 2 / (1024 * 1024),
                'memory_efficient': True
            }
        return {'estimated_memory_mb': 0, 'peak_memory_mb': 0, 'memory_efficient': True}

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

    def process(self, *args, **kwargs) -> Any:
        """Process method to be implemented by subclasses."""
        raise NotImplementedError(f"Subclasses must implement the process method. Class: {self.__class__.__name__}")

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
        """Validate a DataFrame using enhanced utilities."""
        errors = []
        warnings = []

        try:
            # Basic validation
            if not isinstance(data, pd.DataFrame):
                errors.append("Data must be a pandas DataFrame")
                return ValidationResult(False, level, errors, warnings, {})

            if len(data) == 0:
                errors.append("DataFrame cannot be empty")
                return ValidationResult(False, level, errors, warnings, {})

            # Use utility function for column validation
            required_columns = self.validation_rules['dataframe']['required_columns']
            if not validate_dataframe_columns(data, required_columns):
                missing_columns = [col for col in required_columns if col not in data.columns]
                errors.append(f"Missing required columns: {missing_columns}")

            # Check minimum rows
            min_rows = self.validation_rules['dataframe']['min_rows']
            if len(data) < min_rows:
                errors.append(f"DataFrame must have at least {min_rows} rows")

            # Enhanced NaN analysis using utilities
            nan_analysis = analyze_nan_values_detailed(data)
            quality_metrics = calculate_data_quality_metrics(data)

            # Check for excessive NaN values using utility analysis
            max_nan_ratio = self.validation_rules['dataframe']['max_nan_ratio']
            for col, nan_pct in nan_analysis['feature_nan_percentages'].items():
                if nan_pct > max_nan_ratio * 100:
                    warnings.append(f"Column {col} has {nan_pct:.2f}% NaN values")

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
                    'dtypes': data.dtypes.to_dict(),
                    'nan_analysis': nan_analysis,
                    'quality_metrics': quality_metrics
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

    def safe_dataframe_operation(self, data: pd.DataFrame, operation: Callable, *args, **kwargs) -> pd.DataFrame:
        """Safely perform DataFrame operation using utilities."""
        return safe_dataframe_operation(data, operation, *args, **kwargs)

    def validate_dataframe_columns(self, data: pd.DataFrame, required_columns: List[str]) -> bool:
        """Validate DataFrame columns using utilities."""
        return validate_dataframe_columns(data, required_columns)

    def analyze_data_quality(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """Analyze data quality using utilities."""
        if isinstance(data, pd.DataFrame):
            nan_analysis = analyze_nan_values_detailed(data)
            quality_metrics = calculate_data_quality_metrics(data)
            quality_report = create_data_quality_report(data)
            dataframe_info = get_dataframe_info(data)
            summary_stats = create_summary_statistics(data)

            return {
                'nan_analysis': nan_analysis,
                'quality_metrics': quality_metrics,
                'quality_report': quality_report,
                'dataframe_info': dataframe_info,
                'summary_statistics': summary_stats
            }
        else:
            # Convert numpy array to DataFrame for analysis
            if data.ndim == 2:
                df = pd.DataFrame(data)
                return self.analyze_data_quality(df)
            else:
                return {'error': 'Unsupported data type for quality analysis'}

    def safe_convert_dtypes(self, data: pd.DataFrame, dtype_mapping: Dict[str, str]) -> pd.DataFrame:
        """Safely convert DataFrame dtypes using utilities."""
        return safe_convert_dtypes(data, dtype_mapping)

    def safe_merge_dataframes(self, df1: pd.DataFrame, df2: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Safely merge DataFrames using utilities."""
        return safe_merge_dataframes(df1, df2, **kwargs)

    def safe_drop_columns(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Safely drop columns using utilities."""
        return safe_drop_columns(data, columns)

    def safe_rename_columns(self, data: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
        """Safely rename columns using utilities."""
        return safe_rename_columns(data, column_mapping)

    def safe_filter_dataframe(self, data: pd.DataFrame, condition: str) -> pd.DataFrame:
        """Safely filter DataFrame using utilities."""
        return safe_filter_dataframe(data, condition)

    def safe_groupby_operation(self, data: pd.DataFrame, group_cols: List[str], agg_dict: Dict[str, str]) -> pd.DataFrame:
        """Safely perform groupby operation using utilities."""
        return safe_groupby_operation(data, group_cols, agg_dict)

    def safe_apply_function(self, data: pd.DataFrame, func: Callable, axis: int = 0) -> pd.DataFrame:
        """Safely apply function to DataFrame using utilities."""
        return safe_apply_function(data, func, axis)

# Convenience functions
def create_modular_architecture(component_name: str,
                               logger: Optional[logging.Logger] = None) -> ModularArchitecture:
    """Create a modular architecture instance."""
    return ModularArchitecture(component_name, logger)

def create_modular_component(component_class: Type[ModularComponent],
                           name: str,
                           config: Optional[Dict[str, Any]] = None,
                           logger: Optional[logging.Logger] = None) -> ModularComponent:
    """Create a modular component instance."""
    return component_class(name, config, logger)

# Export main classes and functions
__all__ = [
    'ModularComponent',
    'ExampleModularComponent',
    'ModularArchitecture',
    'BaseModule',
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
    'create_modular_architecture',
    'create_modular_component'
]
