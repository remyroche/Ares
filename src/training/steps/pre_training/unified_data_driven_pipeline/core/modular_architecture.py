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

    def initialize(self) -> bool:
        """
        Initialize the component and its resources.
        
        This method should:
        1. Validate configuration
        2. Initialize any required resources
        3. Set up internal state
        4. Perform any necessary setup operations
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Validate configuration
            if not self.validate_config():
                self.logger.error(f"Configuration validation failed for component {self.name}")
                return False
            
            # Initialize component-specific resources
            init_success = self._initialize_resources()
            if not init_success:
                self.logger.error(f"Resource initialization failed for component {self.name}")
                return False
            
            # Set initialization flag
            self._initialized = True
            self.logger.info(f"Component {self.name} initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize component {self.name}: {str(e)}")
            return False

    def process(self, data: Any, **kwargs) -> Any:
        """
        Process the input data.
        
        This method should:
        1. Validate input data
        2. Perform the main processing logic
        3. Return processed results
        4. Handle errors gracefully
        
        Args:
            data: Input data to process
            **kwargs: Additional processing parameters
            
        Returns:
            Any: Processed data
            
        Raises:
            ValueError: If input data is invalid
            RuntimeError: If processing fails
        """
        if not self._initialized:
            raise RuntimeError(f"Component {self.name} is not initialized")
        
        # Validate input
        validation_result = self.validate_input(data)
        if not validation_result.is_valid:
            raise ValueError(f"Input validation failed: {validation_result.errors}")
        
        # Check if component can process the data
        if not self.can_process(data):
            raise ValueError(f"Component {self.name} cannot process the given data")
        
        # Perform processing with error handling
        try:
            result = self._process_data(data, **kwargs)
            self.logger.debug(f"Component {self.name} processed data successfully")
            return result
        except Exception as e:
            self.logger.error(f"Processing failed in component {self.name}: {str(e)}")
            raise RuntimeError(f"Processing failed: {str(e)}")

    def validate_input(self, data: Any) -> ValidationResult:
        """
        Validate input data.
        
        This method should:
        1. Check data type and structure
        2. Validate required fields/columns
        3. Check data quality
        4. Return comprehensive validation results
        
        Args:
            data: Data to validate
            
        Returns:
            ValidationResult: Validation results with errors, warnings, and metadata
        """
        errors = []
        warnings = []
        metadata = {}
        
        try:
            # Basic type validation
            if data is None:
                errors.append("Input data cannot be None")
                return ValidationResult(False, ValidationLevel.STANDARD, errors, warnings, metadata)
            
            # Get validation rules
            validation_rules = self._get_validation_rules()
            
            # Type-specific validation
            if isinstance(data, pd.DataFrame):
                validation_result = self._validate_dataframe(data, validation_rules)
            elif isinstance(data, pd.Series):
                validation_result = self._validate_series(data, validation_rules)
            elif isinstance(data, np.ndarray):
                validation_result = self._validate_array(data, validation_rules)
            elif isinstance(data, (list, tuple)):
                validation_result = self._validate_sequence(data, validation_rules)
            else:
                # Generic validation
                validation_result = self._validate_generic(data, validation_rules)
            
            errors.extend(validation_result.errors)
            warnings.extend(validation_result.warnings)
            metadata.update(validation_result.metadata)
            
            # Additional component-specific validation
            component_validation = self._validate_component_specific(data)
            errors.extend(component_validation.get('errors', []))
            warnings.extend(component_validation.get('warnings', []))
            metadata.update(component_validation.get('metadata', {}))
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                validation_level=ValidationLevel.STANDARD,
                errors=errors,
                warnings=warnings,
                metadata=metadata
            )
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            return ValidationResult(False, ValidationLevel.STANDARD, errors, warnings, metadata)

    def cleanup(self) -> None:
        """
        Cleanup resources and reset state.
        
        This method should:
        1. Release any allocated resources
        2. Clear internal state
        3. Reset performance statistics
        4. Prepare component for reinitialization
        """
        try:
            # Cleanup component-specific resources
            self._cleanup_resources()
            
            # Reset state
            self.clear_state()
            
            # Reset performance statistics
            self.reset_stats()
            
            # Reset initialization flag
            self._initialized = False
            
            self.logger.info(f"Component {self.name} cleaned up successfully")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup of component {self.name}: {str(e)}")

    def get_component_info(self) -> Dict[str, Any]:
        """
        Get component information.
        
        Returns:
            Dict[str, Any]: Component metadata including name, type, version, etc.
        """
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'version': getattr(self, 'version', '1.0.0'),
            'description': getattr(self, 'description', f'Modular component: {self.name}'),
            'initialized': self._initialized,
            'config': self.config.copy(),
            'dependencies': self.get_dependencies(),
            'capabilities': self.get_processing_capabilities()
        }

    def get_dependencies(self) -> List[str]:
        """
        Get list of component dependencies.
        
        Returns:
            List[str]: List of required dependencies (packages, modules, etc.)
        """
        # Default dependencies - can be overridden by subclasses
        return ['pandas', 'numpy']

    def get_output_schema(self) -> Dict[str, Any]:
        """
        Get expected output schema.
        
        Returns:
            Dict[str, Any]: Schema describing the expected output format
        """
        return {
            'type': 'Any',
            'description': 'Processed data output',
            'metadata': {
                'component': self.name,
                'timestamp': time.time()
            }
        }

    def get_required_config(self) -> List[str]:
        """
        Get list of required configuration parameters.
        
        Returns:
            List[str]: List of required configuration keys
        """
        # Default required config - can be overridden by subclasses
        return []

    def can_process(self, data: Any) -> bool:
        """
        Check if component can process the given data.
        
        Args:
            data: Data to check
            
        Returns:
            bool: True if component can process the data, False otherwise
        """
        try:
            # Basic checks
            if data is None:
                return False
            
            if not self._initialized:
                return False
            
            # Check data type compatibility
            supported_types = self.get_processing_capabilities().get('input_types', [])
            if supported_types and not any(isinstance(data, eval(t)) for t in supported_types):
                return False
            
            # Check memory requirements
            memory_req = self.get_memory_requirements(data)
            if memory_req.get('estimated_memory_mb', 0) > self.get_config('memory_limit_mb', 1024):
                return False
            
            return True
            
        except Exception:
            return False

    def get_processing_capabilities(self) -> Dict[str, Any]:
        """
        Get component processing capabilities.
        
        Returns:
            Dict[str, Any]: Capabilities including supported input types, features, etc.
        """
        return {
            'input_types': ['pandas.DataFrame', 'pandas.Series', 'numpy.ndarray', 'list', 'tuple'],
            'output_types': ['pandas.DataFrame', 'pandas.Series', 'numpy.ndarray', 'dict', 'list'],
            'supports_parallel': True,
            'memory_efficient': True,
            'supports_streaming': False,
            'max_input_size': self.get_config('max_input_size', 1000000),
            'features': ['validation', 'error_handling', 'performance_monitoring']
        }

    def estimate_processing_time(self, data: Any) -> float:
        """
        Estimate processing time for given data.
        
        Args:
            data: Data to estimate processing time for
            
        Returns:
            float: Estimated processing time in seconds
        """
        try:
            # Get base processing time from config
            base_time = self.get_config('base_processing_time', 0.001)
            
            # Estimate based on data size
            if hasattr(data, '__len__'):
                size_factor = len(data) * 0.000001  # 1 microsecond per item
            else:
                size_factor = 0.001
            
            # Add complexity factor
            complexity_factor = self.get_config('complexity_factor', 1.0)
            
            # Calculate estimated time
            estimated_time = base_time + (size_factor * complexity_factor)
            
            # Apply performance multiplier from config
            performance_multiplier = self.get_config('performance_multiplier', 1.0)
            
            return estimated_time * performance_multiplier
            
        except Exception:
            return 0.1  # Default fallback

    def get_memory_requirements(self, data: Any) -> Dict[str, Any]:
        """
        Get memory requirements for processing data.
        
        Args:
            data: Data to analyze
            
        Returns:
            Dict[str, Any]: Memory requirements including estimated and peak memory usage
        """
        try:
            # Calculate base memory usage
            base_memory = 0
            
            if hasattr(data, 'memory_usage'):
                # For pandas objects
                base_memory = data.memory_usage(deep=True).sum()
            elif hasattr(data, 'nbytes'):
                # For numpy arrays
                base_memory = data.nbytes
            elif hasattr(data, '__len__'):
                # Rough estimate for other objects
                base_memory = len(data) * 8  # Assume 8 bytes per item
            
            # Convert to MB
            estimated_memory_mb = base_memory / (1024 * 1024)
            
            # Add overhead factor
            overhead_factor = self.get_config('memory_overhead_factor', 1.5)
            peak_memory_mb = estimated_memory_mb * overhead_factor
            
            return {
                'estimated_memory_mb': estimated_memory_mb,
                'peak_memory_mb': peak_memory_mb,
                'memory_efficient': estimated_memory_mb < 100,  # Less than 100MB
                'overhead_factor': overhead_factor
            }
            
        except Exception:
            return {
                'estimated_memory_mb': 10.0,
                'peak_memory_mb': 20.0,
                'memory_efficient': True,
                'overhead_factor': 1.5
            }

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

    def _initialize_resources(self) -> bool:
        """Initialize component-specific resources. Override in subclasses."""
        return True

    def _cleanup_resources(self) -> None:
        """Cleanup component-specific resources. Override in subclasses."""
        pass

    def _process_data(self, data: Any, **kwargs) -> Any:
        """Process data with component-specific logic. Override in subclasses."""
        # Default implementation - just return the data
        return data

    def _get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules for this component. Override in subclasses."""
        return {
            'min_size': 1,
            'max_size': 1000000,
            'required_attributes': [],
            'data_types': ['pandas.DataFrame', 'pandas.Series', 'numpy.ndarray', 'list', 'tuple']
        }

    def _validate_dataframe(self, data: pd.DataFrame, rules: Dict[str, Any]) -> ValidationResult:
        """Validate pandas DataFrame."""
        errors = []
        warnings = []
        metadata = {'type': 'DataFrame', 'shape': data.shape, 'columns': list(data.columns)}
        
        # Check size constraints
        if len(data) < rules.get('min_size', 1):
            errors.append(f"DataFrame too small: {len(data)} < {rules.get('min_size', 1)}")
        if len(data) > rules.get('max_size', 1000000):
            warnings.append(f"DataFrame large: {len(data)} > {rules.get('max_size', 1000000)}")
        
        # Check required columns
        required_attrs = rules.get('required_attributes', [])
        missing_cols = [col for col in required_attrs if col not in data.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Check for NaN values
        nan_count = data.isnull().sum().sum()
        if nan_count > 0:
            warnings.append(f"DataFrame contains {nan_count} NaN values")
        
        return ValidationResult(len(errors) == 0, ValidationLevel.STANDARD, errors, warnings, metadata)

    def _validate_series(self, data: pd.Series, rules: Dict[str, Any]) -> ValidationResult:
        """Validate pandas Series."""
        errors = []
        warnings = []
        metadata = {'type': 'Series', 'length': len(data), 'dtype': str(data.dtype)}
        
        # Check size constraints
        if len(data) < rules.get('min_size', 1):
            errors.append(f"Series too small: {len(data)} < {rules.get('min_size', 1)}")
        if len(data) > rules.get('max_size', 1000000):
            warnings.append(f"Series large: {len(data)} > {rules.get('max_size', 1000000)}")
        
        # Check for NaN values
        nan_count = data.isnull().sum()
        if nan_count > 0:
            warnings.append(f"Series contains {nan_count} NaN values")
        
        return ValidationResult(len(errors) == 0, ValidationLevel.STANDARD, errors, warnings, metadata)

    def _validate_array(self, data: np.ndarray, rules: Dict[str, Any]) -> ValidationResult:
        """Validate numpy array."""
        errors = []
        warnings = []
        metadata = {'type': 'ndarray', 'shape': data.shape, 'dtype': str(data.dtype)}
        
        # Check size constraints
        total_size = data.size
        if total_size < rules.get('min_size', 1):
            errors.append(f"Array too small: {total_size} < {rules.get('min_size', 1)}")
        if total_size > rules.get('max_size', 1000000):
            warnings.append(f"Array large: {total_size} > {rules.get('max_size', 1000000)}")
        
        # Check for NaN values
        if np.isnan(data).any():
            warnings.append("Array contains NaN values")
        
        return ValidationResult(len(errors) == 0, ValidationLevel.STANDARD, errors, warnings, metadata)

    def _validate_sequence(self, data: Union[list, tuple], rules: Dict[str, Any]) -> ValidationResult:
        """Validate list or tuple."""
        errors = []
        warnings = []
        metadata = {'type': type(data).__name__, 'length': len(data)}
        
        # Check size constraints
        if len(data) < rules.get('min_size', 1):
            errors.append(f"Sequence too small: {len(data)} < {rules.get('min_size', 1)}")
        if len(data) > rules.get('max_size', 1000000):
            warnings.append(f"Sequence large: {len(data)} > {rules.get('max_size', 1000000)}")
        
        return ValidationResult(len(errors) == 0, ValidationLevel.STANDARD, errors, warnings, metadata)

    def _validate_generic(self, data: Any, rules: Dict[str, Any]) -> ValidationResult:
        """Validate generic data type."""
        errors = []
        warnings = []
        metadata = {'type': type(data).__name__}
        
        # Basic validation
        if data is None:
            errors.append("Data cannot be None")
        
        return ValidationResult(len(errors) == 0, ValidationLevel.STANDARD, errors, warnings, metadata)

    def _validate_component_specific(self, data: Any) -> Dict[str, Any]:
        """Validate data with component-specific rules. Override in subclasses."""
        return {'errors': [], 'warnings': [], 'metadata': {}}

class ExampleModularComponent(ModularComponent):
    """Example implementation of ModularComponent for demonstration purposes."""

    def __init__(self, name: str = "example_component", config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(name, config, logger)
        self.processing_window = self.get_config('processing_window', 20)
        self.threshold = self.get_config('threshold', 0.5)
        self.version = "1.0.0"
        self.description = "Example modular component for demonstration"

    def _initialize_resources(self) -> bool:
        """Initialize example component resources."""
        try:
            # Set up processing parameters
            self.set_state('processing_window', self.processing_window)
            self.set_state('threshold', self.threshold)
            self.set_state('initialization_time', time.time())
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize resources: {e}")
            return False

    def _cleanup_resources(self) -> None:
        """Cleanup example component resources."""
        # Clear any cached data
        self.set_state('last_processed_data', None)
        self.set_state('processing_cache', {})

    def _process_data(self, data: Any, **kwargs) -> Any:
        """Process data with example-specific logic."""
        if isinstance(data, pd.DataFrame):
            # Example: Calculate rolling mean
            if 'close' in data.columns:
                result = data['close'].rolling(window=self.processing_window).mean()
                # Store processing metadata
                self.set_state('last_processed_data', {
                    'shape': data.shape,
                    'processing_window': self.processing_window,
                    'result_length': len(result)
                })
                return result
            else:
                raise ValueError("DataFrame must contain 'close' column")
        else:
            raise ValueError("Data must be a pandas DataFrame")

    def _get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules for example component."""
        return {
            'min_size': self.processing_window,
            'max_size': 1000000,
            'required_attributes': ['close'],
            'data_types': ['pandas.DataFrame']
        }

    def _validate_component_specific(self, data: Any) -> Dict[str, Any]:
        """Validate data with example-specific rules."""
        errors = []
        warnings = []
        metadata = {}
        
        if isinstance(data, pd.DataFrame):
            # Check if we have enough data for the processing window
            if len(data) < self.processing_window:
                errors.append(f"Data must have at least {self.processing_window} rows for processing")
            
            # Check if close column exists and is numeric
            if 'close' in data.columns:
                if not pd.api.types.is_numeric_dtype(data['close']):
                    errors.append("'close' column must be numeric")
                else:
                    # Check for extreme values
                    close_values = data['close'].dropna()
                    if len(close_values) > 0:
                        q99 = close_values.quantile(0.99)
                        q01 = close_values.quantile(0.01)
                        if (close_values > q99 * 10).any():
                            warnings.append("Extreme values detected in 'close' column")
                        metadata['close_stats'] = {
                            'mean': close_values.mean(),
                            'std': close_values.std(),
                            'min': close_values.min(),
                            'max': close_values.max()
                        }
            else:
                errors.append("DataFrame must contain 'close' column")
        
        return {'errors': errors, 'warnings': warnings, 'metadata': metadata}

    def get_component_info(self) -> Dict[str, Any]:
        """Get component information."""
        base_info = super().get_component_info()
        base_info.update({
            'type': 'example_component',
            'version': self.version,
            'description': self.description,
            'processing_window': self.processing_window,
            'threshold': self.threshold
        })
        return base_info

    def get_dependencies(self) -> List[str]:
        """Get list of component dependencies."""
        return ['pandas', 'numpy']

    def get_output_schema(self) -> Dict[str, Any]:
        """Get expected output schema."""
        return {
            'type': 'pandas.Series',
            'index_type': 'DatetimeIndex',
            'dtype': 'float64',
            'description': 'Rolling mean of close prices',
            'length': 'variable (input_length - processing_window + 1)',
            'index_alignment': 'matches input DataFrame index'
        }

    def get_required_config(self) -> List[str]:
        """Get list of required configuration parameters."""
        return ['processing_window', 'threshold']

    def get_processing_capabilities(self) -> Dict[str, Any]:
        """Get component processing capabilities."""
        base_capabilities = super().get_processing_capabilities()
        base_capabilities.update({
            'input_types': ['pandas.DataFrame'],
            'required_columns': ['close'],
            'output_type': 'pandas.Series',
            'supports_parallel': True,
            'memory_efficient': True,
            'supports_streaming': False,
            'features': ['rolling_calculations', 'statistical_analysis', 'data_validation']
        })
        return base_capabilities

    def estimate_processing_time(self, data: Any) -> float:
        """Estimate processing time for given data."""
        base_time = super().estimate_processing_time(data)
        
        if isinstance(data, pd.DataFrame):
            # More accurate estimation for rolling calculations
            rows = len(data)
            window = self.processing_window
            # Rolling operations are O(n) but with window overhead
            complexity_factor = 1 + (window / 100)  # Window size affects performance
            estimated_time = (rows * 0.00005) * complexity_factor  # 0.05ms per row base
            return max(base_time, estimated_time)
        
        return base_time

    def get_memory_requirements(self, data: Any) -> Dict[str, Any]:
        """Get memory requirements for processing data."""
        base_requirements = super().get_memory_requirements(data)
        
        if isinstance(data, pd.DataFrame):
            # Rolling calculations need additional memory for intermediate results
            input_memory = base_requirements['estimated_memory_mb']
            # Rolling operations typically need 2-3x input memory
            rolling_overhead = 2.5
            estimated_memory = input_memory * rolling_overhead
            peak_memory = estimated_memory * 1.5  # Peak during processing
            
            base_requirements.update({
                'estimated_memory_mb': estimated_memory,
                'peak_memory_mb': peak_memory,
                'memory_efficient': estimated_memory < 500,  # Less than 500MB
                'rolling_overhead': rolling_overhead
            })
        
        return base_requirements

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
