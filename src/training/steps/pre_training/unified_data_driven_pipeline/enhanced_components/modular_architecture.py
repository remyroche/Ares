"""
Enhanced Modular Architecture for UnifiedDataDrivenPipeline

This module provides an enhanced modular architecture system with:
- Input validation with multiple validation levels
- Standardized error handling with error categorization
- Performance monitoring with detailed metrics
- Memory management and optimization
- Hardware acceleration detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import time
import logging
import psutil
import gc
from pathlib import Path
import json
import pickle

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
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

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Validation levels for input data."""
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
    metric_type: str
    value: float
    unit: str
    timestamp: float
    component: str
    metadata: Dict[str, Any]

class InputValidator:
    """Enhanced input validation component."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
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
        """Validate a DataFrame with enhanced checks."""
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
                nan_pct = data[col].isna().sum() / len(data) * 100
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

class ErrorHandler:
    """Enhanced error handling component."""

    def __init__(self, component_name: str, logger: Optional[logging.Logger] = None):
        self.component_name = component_name
        self.logger = logger or logging.getLogger(__name__)
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
            stack_trace=str(error),
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

        return error_info

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors."""
        return {
            'total_errors': len(self.error_history),
            'error_counts': self.error_counts.copy(),
            'recent_errors': self.error_history[-10:] if self.error_history else [],
            'critical_errors': [e for e in self.error_history if e.severity == ErrorSeverity.CRITICAL]
        }

class PerformanceMonitor:
    """Enhanced performance monitoring component."""

    def __init__(self, component_name: str, logger: Optional[logging.Logger] = None):
        self.component_name = component_name
        self.logger = logger or logging.getLogger(__name__)
        self.metrics = []
        self.start_times = {}

    def start_operation(self, operation_name: str) -> float:
        """Start monitoring an operation."""
        start_time = time.time()
        self.start_times[operation_name] = start_time
        return start_time

    def end_operation(self, operation_name: str) -> float:
        """End monitoring an operation and record the metric."""
        if operation_name not in self.start_times:
            return 0.0

        end_time = time.time()
        duration = end_time - self.start_times[operation_name]

        metric = PerformanceMetric(
            metric_type="execution_time",
            value=duration,
            unit="seconds",
            timestamp=end_time,
            component=self.component_name,
            metadata={'operation': operation_name}
        )

        self.metrics.append(metric)
        del self.start_times[operation_name]

        return duration

    def record_metric(self,
                     metric_type: str,
                     value: float,
                     unit: str = "",
                     metadata: Dict[str, Any] = None):
        """Record a performance metric."""
        metric = PerformanceMetric(
            metric_type=metric_type,
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
            metric_type = metric.metric_type
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

class MemoryManager:
    """Enhanced memory management component."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.memory_history = []

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()

            return {
                'rss': memory_info.rss / (1024 * 1024),  # MB
                'vms': memory_info.vms / (1024 * 1024),  # MB
                'percent': process.memory_percent()
            }
        except Exception as e:
            tprint_debug(f"Memory monitoring failed: {e}")
            return {'rss': 0.0, 'vms': 0.0, 'percent': 0.0}

    def optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory usage."""
        try:
            # Force garbage collection
            gc.collect()

            # Get memory before and after
            before = self.get_memory_usage()

            # Additional optimization could be added here

            after = self.get_memory_usage()

            return {
                'before': before,
                'after': after,
                'freed': before['rss'] - after['rss']
            }
        except Exception as e:
            tprint_debug(f"Memory optimization failed: {e}")
            return {'error': str(e)}

class HardwareAccelerator:
    """Hardware acceleration detection and management."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.hardware_info = self._detect_hardware()

    def _detect_hardware(self) -> Dict[str, Any]:
        """Detect available hardware acceleration."""
        hardware = {
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total / (1024**3),  # GB
            'gpu_available': False,
            'cuda_available': False
        }

        # GPU support removed
        hardware['cuda_available'] = False
        hardware['gpu_available'] = False

        # Check for other GPU libraries
        try:
            import torch
            if torch.cuda.is_available():
                hardware['gpu_available'] = True
                tprint_debug("✅ PyTorch with CUDA detected - GPU support available")
            else:
                tprint_debug("ℹ️ PyTorch available but CUDA not available")
        except ImportError as e:
            tprint_debug(f"ℹ️ PyTorch not available: {str(e)}")
            hardware['gpu_available'] = False

        return hardware

    def get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information."""
        return self.hardware_info.copy()

class ModularArchitecture:
    """Enhanced modular architecture coordinator."""

    def __init__(self, component_name: str, logger: Optional[logging.Logger] = None):
        self.component_name = component_name
        self.logger = logger or logging.getLogger(f"{__name__}.{component_name}")

        # Initialize modular components
        self.validator = InputValidator(self.logger)
        self.error_handler = ErrorHandler(component_name, self.logger)
        self.performance_monitor = PerformanceMonitor(component_name, self.logger)
        self.memory_manager = MemoryManager(self.logger)
        self.hardware_accelerator = HardwareAccelerator(self.logger)

        tprint_info(f"🏗️ Enhanced modular architecture initialized for {component_name}")

    def validate_inputs(self, data: Union[pd.DataFrame, pd.Series],
                       level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationResult:
        """Validate inputs using the modular validator."""
        return self.validator.validate_dataframe(data, level)

    def handle_error(self, error: Exception, **kwargs) -> ErrorInfo:
        """Handle errors using the modular error handler."""
        return self.error_handler.handle_error(error, **kwargs)

    def start_monitoring(self, operation_name: str) -> float:
        """Start monitoring an operation."""
        return self.performance_monitor.start_operation(operation_name)

    def stop_monitoring(self, operation_name: str) -> float:
        """Stop monitoring an operation."""
        return self.performance_monitor.end_operation(operation_name)

    def get_system_summary(self) -> Dict[str, Any]:
        """Get comprehensive system summary."""
        return {
            'component_name': self.component_name,
            'error_handler_stats': self.error_handler.get_error_summary(),
            'performance_monitor_stats': self.performance_monitor.get_performance_summary(),
            'memory_usage': self.memory_manager.get_memory_usage(),
            'hardware_info': self.hardware_accelerator.get_hardware_info()
        }

# Convenience functions
def create_modular_architecture(component_name: str,
                               logger: Optional[logging.Logger] = None) -> ModularArchitecture:
    """Create an enhanced modular architecture instance."""
    return ModularArchitecture(component_name, logger)

# Export main classes and functions
__all__ = [
    'ModularArchitecture',
    'InputValidator',
    'ErrorHandler',
    'PerformanceMonitor',
    'MemoryManager',
    'HardwareAccelerator',
    'ValidationLevel',
    'ErrorSeverity',
    'ErrorCategory',
    'ValidationResult',
    'ErrorInfo',
    'PerformanceMetric',
    'create_modular_architecture'
]
