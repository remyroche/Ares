"""
Modular Architecture for UnifiedDataDrivenPipeline

This module provides a modular architecture system inspired by FeatureLookbackOptimizationComponent,
with separate modules for:
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
    COMPUTATION = "computation"
    MEMORY = "memory"
    IO = "io"
    NETWORK = "network"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class MetricType(Enum):
    """Metric types for performance monitoring."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class MetricLevel(Enum):
    """Metric levels for performance monitoring."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ValidationSummary:
    """Summary of validation results."""
    is_valid: bool
    validation_level: ValidationLevel
    errors: List[str]
    warnings: List[str]
    recommendations: List[str]
    validation_time: float
    n_checks_performed: int


@dataclass
class ErrorInfo:
    """Information about an error."""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    component: str
    timestamp: float
    context: Dict[str, Any]
    stack_trace: Optional[str] = None


@dataclass
class PerformanceMetric:
    """Performance metric data."""
    name: str
    value: float
    metric_type: MetricType
    level: MetricLevel
    timestamp: float
    component: str
    metadata: Dict[str, Any]


class InputValidator:
    """Advanced input validator with multiple validation levels."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the input validator."""
        self.logger = logger or logging.getLogger(__name__)
        self.validation_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'validation_time': 0.0
        }
    
    def validate_data(
        self,
        data: Any,
        required_columns: List[str],
        validation_level: ValidationLevel = ValidationLevel.STANDARD
    ) -> Tuple[bool, ValidationSummary, Any]:
        """
        Validate input data with specified validation level.
        
        Args:
            data: Input data to validate
            required_columns: List of required column names
            validation_level: Level of validation to perform
            
        Returns:
            Tuple of (is_valid, validation_summary, cleaned_data)
        """
        start_time = time.time()
        
        try:
            self.validation_stats['total_validations'] += 1
            
            # Basic validation
            if not self._basic_validation(data, required_columns):
                return False, self._create_validation_summary(
                    False, validation_level, ["Basic validation failed"], [], [], 
                    time.time() - start_time, 1
                ), None
            
            # Level-specific validation
            if validation_level == ValidationLevel.BASIC:
                return True, self._create_validation_summary(
                    True, validation_level, [], [], [], 
                    time.time() - start_time, 1
                ), data
            
            # Standard validation
            errors, warnings, recommendations = self._standard_validation(data, required_columns)
            if validation_level == ValidationLevel.STANDARD:
                is_valid = len(errors) == 0
                return is_valid, self._create_validation_summary(
                    is_valid, validation_level, errors, warnings, recommendations,
                    time.time() - start_time, 2
                ), data
            
            # Strict validation
            errors, warnings, recommendations = self._strict_validation(data, required_columns)
            if validation_level == ValidationLevel.STRICT:
                is_valid = len(errors) == 0
                return is_valid, self._create_validation_summary(
                    is_valid, validation_level, errors, warnings, recommendations,
                    time.time() - start_time, 3
                ), data
            
            # Exhaustive validation
            errors, warnings, recommendations = self._exhaustive_validation(data, required_columns)
            is_valid = len(errors) == 0
            
            validation_time = time.time() - start_time
            self.validation_stats['validation_time'] += validation_time
            
            if is_valid:
                self.validation_stats['successful_validations'] += 1
            else:
                self.validation_stats['failed_validations'] += 1
            
            return is_valid, self._create_validation_summary(
                is_valid, validation_level, errors, warnings, recommendations,
                validation_time, 4
            ), data
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return False, self._create_validation_summary(
                False, validation_level, [f"Validation error: {e}"], [], [],
                time.time() - start_time, 0
            ), None
    
    def _basic_validation(self, data: Any, required_columns: List[str]) -> bool:
        """Perform basic validation."""
        try:
            if data is None:
                return False
            
            if not hasattr(data, 'columns'):
                return False
            
            if not hasattr(data, 'shape'):
                return False
            
            if data.empty:
                return False
            
            return True
        except:
            return False
    
    def _standard_validation(
        self, 
        data: pd.DataFrame, 
        required_columns: List[str]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Perform standard validation."""
        errors = []
        warnings = []
        recommendations = []
        
        try:
            # Check required columns
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                errors.append(f"Missing required columns: {missing_columns}")
            
            # Check data types
            for col in required_columns:
                if col in data.columns:
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        warnings.append(f"Column {col} is not numeric")
                        recommendations.append(f"Consider converting {col} to numeric")
            
            # Check for excessive missing values
            missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
            if missing_ratio > 0.5:
                warnings.append(f"High missing value ratio: {missing_ratio:.2%}")
                recommendations.append("Consider data imputation or removal of rows/columns")
            
            # Check data size
            if data.shape[0] < 10:
                warnings.append("Very small dataset")
                recommendations.append("Consider using more data for reliable results")
            
        except Exception as e:
            errors.append(f"Standard validation error: {e}")
        
        return errors, warnings, recommendations
    
    def _strict_validation(
        self, 
        data: pd.DataFrame, 
        required_columns: List[str]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Perform strict validation."""
        errors, warnings, recommendations = self._standard_validation(data, required_columns)
        
        try:
            # Check for infinite values
            inf_count = np.isinf(data.select_dtypes(include=[np.number])).sum().sum()
            if inf_count > 0:
                errors.append(f"Found {inf_count} infinite values")
                recommendations.append("Remove or replace infinite values")
            
            # Check for extreme outliers
            for col in data.select_dtypes(include=[np.number]).columns:
                if col in required_columns:
                    q99 = data[col].quantile(0.99)
                    q01 = data[col].quantile(0.01)
                    iqr = q99 - q01
                    outliers = ((data[col] > q99 + 3 * iqr) | (data[col] < q01 - 3 * iqr)).sum()
                    if outliers > len(data) * 0.05:  # More than 5% outliers
                        warnings.append(f"Column {col} has {outliers} extreme outliers")
                        recommendations.append(f"Consider outlier treatment for {col}")
            
            # Check for constant columns
            constant_columns = data.columns[data.nunique() <= 1]
            if len(constant_columns) > 0:
                warnings.append(f"Constant columns found: {constant_columns.tolist()}")
                recommendations.append("Remove constant columns")
            
        except Exception as e:
            errors.append(f"Strict validation error: {e}")
        
        return errors, warnings, recommendations
    
    def _exhaustive_validation(
        self, 
        data: pd.DataFrame, 
        required_columns: List[str]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Perform exhaustive validation."""
        errors, warnings, recommendations = self._strict_validation(data, required_columns)
        
        try:
            # Check for duplicate rows
            duplicate_rows = data.duplicated().sum()
            if duplicate_rows > 0:
                warnings.append(f"Found {duplicate_rows} duplicate rows")
                recommendations.append("Consider removing duplicate rows")
            
            # Check for perfect correlations
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = data[numeric_cols].corr()
                perfect_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) > 0.99:
                            perfect_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
                
                if perfect_corr_pairs:
                    warnings.append(f"Perfect correlations found: {perfect_corr_pairs}")
                    recommendations.append("Consider removing highly correlated features")
            
            # Check memory usage
            memory_usage = data.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            if memory_usage > 1000:  # More than 1GB
                warnings.append(f"High memory usage: {memory_usage:.1f} MB")
                recommendations.append("Consider data optimization or chunking")
            
        except Exception as e:
            errors.append(f"Exhaustive validation error: {e}")
        
        return errors, warnings, recommendations
    
    def _create_validation_summary(
        self,
        is_valid: bool,
        validation_level: ValidationLevel,
        errors: List[str],
        warnings: List[str],
        recommendations: List[str],
        validation_time: float,
        n_checks: int
    ) -> ValidationSummary:
        """Create validation summary."""
        return ValidationSummary(
            is_valid=is_valid,
            validation_level=validation_level,
            errors=errors,
            warnings=warnings,
            recommendations=recommendations,
            validation_time=validation_time,
            n_checks_performed=n_checks
        )
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return self.validation_stats.copy()


class StandardizedErrorHandler:
    """Standardized error handler with error categorization and severity levels."""
    
    def __init__(self, logger: Optional[logging.Logger] = None, component_name: str = "Unknown"):
        """Initialize the error handler."""
        self.logger = logger or logging.getLogger(__name__)
        self.component_name = component_name
        self.error_stats = {
            'total_errors': 0,
            'errors_by_category': {category.value: 0 for category in ErrorCategory},
            'errors_by_severity': {severity.value: 0 for severity in ErrorSeverity},
            'recent_errors': []
        }
    
    def handle_error(
        self,
        error: Exception,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None
    ) -> ErrorInfo:
        """
        Handle an error with standardized categorization.
        
        Args:
            error: The exception that occurred
            category: Error category
            severity: Error severity level
            context: Additional context information
            
        Returns:
            ErrorInfo object with error details
        """
        try:
            error_id = f"{self.component_name}_{int(time.time() * 1000)}"
            timestamp = time.time()
            
            # Create error info
            error_info = ErrorInfo(
                error_id=error_id,
                category=category,
                severity=severity,
                message=str(error),
                component=self.component_name,
                timestamp=timestamp,
                context=context or {},
                stack_trace=self._get_stack_trace(error)
            )
            
            # Update statistics
            self.error_stats['total_errors'] += 1
            self.error_stats['errors_by_category'][category.value] += 1
            self.error_stats['errors_by_severity'][severity.value] += 1
            
            # Add to recent errors (keep last 100)
            self.error_stats['recent_errors'].append(error_info)
            if len(self.error_stats['recent_errors']) > 100:
                self.error_stats['recent_errors'] = self.error_stats['recent_errors'][-100:]
            
            # Log based on severity
            if severity == ErrorSeverity.CRITICAL:
                self.logger.critical(f"CRITICAL ERROR [{error_id}]: {error}")
            elif severity == ErrorSeverity.HIGH:
                self.logger.error(f"HIGH ERROR [{error_id}]: {error}")
            elif severity == ErrorSeverity.MEDIUM:
                self.logger.warning(f"MEDIUM ERROR [{error_id}]: {error}")
            else:
                self.logger.info(f"LOW ERROR [{error_id}]: {error}")
            
            return error_info
            
        except Exception as e:
            self.logger.error(f"Error handling failed: {e}")
            return ErrorInfo(
                error_id="error_handler_failed",
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.CRITICAL,
                message=f"Error handler failed: {e}",
                component=self.component_name,
                timestamp=time.time(),
                context={}
            )
    
    def _get_stack_trace(self, error: Exception) -> str:
        """Get stack trace for an error."""
        try:
            import traceback
            return traceback.format_exc()
        except:
            return str(error)
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        return self.error_stats.copy()
    
    def get_recent_errors(self, limit: int = 10) -> List[ErrorInfo]:
        """Get recent errors."""
        return self.error_stats['recent_errors'][-limit:]


class PerformanceMonitor:
    """Performance monitor with detailed metrics and event tracking."""
    
    def __init__(self, component_name: str = "Unknown"):
        """Initialize the performance monitor."""
        self.component_name = component_name
        self.metrics = []
        self.operation_times = {}
        self.performance_stats = {
            'total_operations': 0,
            'total_execution_time': 0.0,
            'memory_usage_mb': 0.0,
            'peak_memory_usage_mb': 0.0,
            'cpu_usage_percent': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def start_operation(self, operation_name: str) -> float:
        """Start timing an operation."""
        start_time = time.time()
        self.operation_times[operation_name] = start_time
        return start_time
    
    def end_operation(self, operation_name: str) -> float:
        """End timing an operation and record the metric."""
        if operation_name not in self.operation_times:
            return 0.0
        
        end_time = time.time()
        duration = end_time - self.operation_times[operation_name]
        
        # Record metric
        self.record_metric(
            name=f"{operation_name}_duration",
            value=duration,
            metric_type=MetricType.TIMER,
            level=MetricLevel.INFO
        )
        
        # Update stats
        self.performance_stats['total_operations'] += 1
        self.performance_stats['total_execution_time'] += duration
        
        del self.operation_times[operation_name]
        return duration
    
    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        level: MetricLevel = MetricLevel.INFO,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a performance metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            metric_type=metric_type,
            level=level,
            timestamp=time.time(),
            component=self.component_name,
            metadata=metadata or {}
        )
        
        self.metrics.append(metric)
        
        # Keep only last 1000 metrics
        if len(self.metrics) > 1000:
            self.metrics = self.metrics[-1000:]
    
    def record_memory_usage(self):
        """Record current memory usage."""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            self.record_metric(
                name="memory_usage_mb",
                value=memory_mb,
                metric_type=MetricType.GAUGE,
                level=MetricLevel.INFO
            )
            
            self.performance_stats['memory_usage_mb'] = memory_mb
            self.performance_stats['peak_memory_usage_mb'] = max(
                self.performance_stats['peak_memory_usage_mb'], memory_mb
            )
            
        except Exception as e:
            tprint_debug(f"Memory monitoring failed: {e}")
    
    def record_cpu_usage(self):
        """Record current CPU usage."""
        try:
            cpu_percent = psutil.cpu_percent()
            
            self.record_metric(
                name="cpu_usage_percent",
                value=cpu_percent,
                metric_type=MetricType.GAUGE,
                level=MetricLevel.INFO
            )
            
            self.performance_stats['cpu_usage_percent'] = cpu_percent
            
        except Exception as e:
            tprint_debug(f"CPU monitoring failed: {e}")
    
    def record_cache_event(self, event_type: str, cache_key: str):
        """Record cache event."""
        if event_type == "hit":
            self.performance_stats['cache_hits'] += 1
        elif event_type == "miss":
            self.performance_stats['cache_misses'] += 1
        
        self.record_metric(
            name=f"cache_{event_type}",
            value=1.0,
            metric_type=MetricType.COUNTER,
            level=MetricLevel.DEBUG,
            metadata={"cache_key": cache_key}
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()
        
        # Add calculated metrics
        if self.performance_stats['total_operations'] > 0:
            stats['average_operation_time'] = (
                self.performance_stats['total_execution_time'] / 
                self.performance_stats['total_operations']
            )
        
        if self.performance_stats['cache_hits'] + self.performance_stats['cache_misses'] > 0:
            stats['cache_hit_rate'] = (
                self.performance_stats['cache_hits'] / 
                (self.performance_stats['cache_hits'] + self.performance_stats['cache_misses'])
            )
        
        return stats
    
    def get_metrics_by_type(self, metric_type: MetricType) -> List[PerformanceMetric]:
        """Get metrics by type."""
        return [m for m in self.metrics if m.metric_type == metric_type]
    
    def get_recent_metrics(self, limit: int = 100) -> List[PerformanceMetric]:
        """Get recent metrics."""
        return self.metrics[-limit:]
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'total_operations': 0,
            'total_execution_time': 0.0,
            'memory_usage_mb': 0.0,
            'peak_memory_usage_mb': 0.0,
            'cpu_usage_percent': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        self.metrics = []
        self.operation_times = {}


class MemoryManager:
    """Memory management and optimization utilities."""
    
    def __init__(self):
        """Initialize the memory manager."""
        self.memory_stats = {
            'total_allocations': 0,
            'total_deallocations': 0,
            'peak_memory_usage': 0.0,
            'current_memory_usage': 0.0
        }
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.memory_stats['current_memory_usage'] = memory_mb
            self.memory_stats['peak_memory_usage'] = max(
                self.memory_stats['peak_memory_usage'], memory_mb
            )
            return memory_mb
        except:
            return 0.0
    
    def optimize_memory(self):
        """Optimize memory usage."""
        try:
            # Force garbage collection
            collected = gc.collect()
            self.memory_stats['total_deallocations'] += collected
            
            # Get memory usage after optimization
            current_usage = self.get_memory_usage()
            
            tprint_debug(f"🧠 Memory optimization: collected {collected} objects, current usage: {current_usage:.1f} MB")
            
        except Exception as e:
            tprint_debug(f"Memory optimization failed: {e}")
    
    def get_memory_pressure(self) -> str:
        """Get memory pressure level."""
        try:
            memory_usage = self.get_memory_usage()
            
            if memory_usage < 100:
                return "low"
            elif memory_usage < 500:
                return "medium"
            elif memory_usage < 1000:
                return "high"
            else:
                return "critical"
        except:
            return "unknown"
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return self.memory_stats.copy()


class HardwareAccelerator:
    """Hardware acceleration detection and management."""
    
    def __init__(self):
        """Initialize the hardware accelerator."""
        self.acceleration_info = self._detect_acceleration()
    
    def _detect_acceleration(self) -> Dict[str, Any]:
        """Detect available hardware acceleration."""
        info = {
            'gpu_available': False,
            'm1_available': False,
            'vectorization_available': False,
            'parallel_processing_available': False
        }
        
        try:
            # Check for GPU
            try:
                import torch
                info['gpu_available'] = torch.cuda.is_available()
            except ImportError:
                pass
            
            # Check for M1 acceleration
            try:
                import platform
                if platform.processor() == 'arm':
                    info['m1_available'] = True
            except:
                pass
            
            # Check for vectorization
            try:
                import numpy as np
                info['vectorization_available'] = hasattr(np, 'vectorize')
            except:
                pass
            
            # Check for parallel processing
            try:
                import multiprocessing
                info['parallel_processing_available'] = multiprocessing.cpu_count() > 1
            except:
                pass
            
        except Exception as e:
            tprint_debug(f"Hardware detection failed: {e}")
        
        return info
    
    def is_accelerated(self) -> bool:
        """Check if any acceleration is available."""
        return any(self.acceleration_info.values())
    
    def get_acceleration_info(self) -> Dict[str, Any]:
        """Get acceleration information."""
        return self.acceleration_info.copy()


def create_modular_architecture(component_name: str = "UnifiedPipeline") -> Tuple[InputValidator, StandardizedErrorHandler, PerformanceMonitor, MemoryManager, HardwareAccelerator]:
    """Create modular architecture components."""
    logger = logging.getLogger(component_name)
    
    validator = InputValidator(logger)
    error_handler = StandardizedErrorHandler(logger, component_name)
    performance_monitor = PerformanceMonitor(component_name)
    memory_manager = MemoryManager()
    hardware_accelerator = HardwareAccelerator()
    
    return validator, error_handler, performance_monitor, memory_manager, hardware_accelerator