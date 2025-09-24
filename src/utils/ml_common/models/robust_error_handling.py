"""
Robust Error Handling & Validation for CVLSA

This module implements comprehensive error handling and validation with:
1. Comprehensive input validation for all components
2. Robust error recovery mechanisms
3. Detailed error logging and reporting
4. Graceful degradation strategies
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Type
from dataclasses import dataclass, field
import logging
import time
import traceback
import warnings
from contextlib import contextmanager
from functools import wraps
import inspect
from enum import Enum

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ValidationError(Exception):
    """Custom validation error."""
    def __init__(self, message: str, field: str = None, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
        self.message = message
        self.field = field
        self.severity = severity
        super().__init__(message)

@dataclass
class ErrorReport:
    """Comprehensive error report."""
    timestamp: float
    error_type: str
    error_message: str
    severity: ErrorSeverity
    component: str
    function_name: str
    input_validation: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: bool = False
    fallback_used: bool = False
    stack_trace: str = ""
    performance_impact: float = 0.0

@dataclass
class ValidationConfig:
    """Configuration for validation and error handling."""
    # Input validation
    validate_inputs: bool = True
    strict_validation: bool = False
    allow_nan_values: bool = False
    allow_infinite_values: bool = False
    
    # Error handling
    enable_error_recovery: bool = True
    max_recovery_attempts: int = 3
    recovery_timeout: float = 30.0
    
    # Logging and reporting
    log_all_errors: bool = True
    log_performance_impact: bool = True
    error_reporting_threshold: ErrorSeverity = ErrorSeverity.MEDIUM
    
    # Graceful degradation
    enable_fallback: bool = True
    fallback_strategies: List[str] = field(default_factory=lambda: ['simplified', 'cached', 'default'])
    
    # Performance monitoring
    track_error_performance: bool = True
    performance_threshold: float = 1.0  # Seconds

class InputValidator:
    """Comprehensive input validation system."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.validation_history: List[Dict[str, Any]] = []
        
        logger.info("🔍 Input validator initialized")
    
    def validate_dataframe(self, data: Any, name: str = "dataframe", 
                          required_columns: Optional[List[str]] = None,
                          min_rows: int = 1) -> pd.DataFrame:
        """Validate DataFrame input."""
        validation_info = {
            'timestamp': time.time(),
            'component': 'dataframe_validation',
            'input_name': name,
            'validation_passed': False
        }
        
        try:
            # Check if it's a DataFrame
            if not isinstance(data, pd.DataFrame):
                if isinstance(data, np.ndarray):
                    data = pd.DataFrame(data)
                elif isinstance(data, (list, tuple)):
                    data = pd.DataFrame(data)
                else:
                    raise ValidationError(f"Input '{name}' must be a DataFrame, got {type(data)}")
            
            # Check for empty DataFrame
            if data.empty:
                if min_rows > 0:
                    raise ValidationError(f"Input '{name}' is empty, minimum {min_rows} rows required")
                else:
                    logger.warning(f"Input '{name}' is empty")
            
            # Check minimum rows
            if len(data) < min_rows:
                raise ValidationError(f"Input '{name}' has {len(data)} rows, minimum {min_rows} required")
            
            # Check required columns
            if required_columns:
                missing_columns = set(required_columns) - set(data.columns)
                if missing_columns:
                    raise ValidationError(f"Input '{name}' missing required columns: {missing_columns}")
            
            # Check for NaN values
            if not self.config.allow_nan_values:
                nan_columns = data.columns[data.isnull().any()].tolist()
                if nan_columns:
                    if self.config.strict_validation:
                        raise ValidationError(f"Input '{name}' contains NaN values in columns: {nan_columns}")
                    else:
                        logger.warning(f"Input '{name}' contains NaN values in columns: {nan_columns}")
            
            # Check for infinite values
            if not self.config.allow_infinite_values:
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                inf_columns = []
                for col in numeric_cols:
                    if np.isinf(data[col]).any():
                        inf_columns.append(col)
                
                if inf_columns:
                    if self.config.strict_validation:
                        raise ValidationError(f"Input '{name}' contains infinite values in columns: {inf_columns}")
                    else:
                        logger.warning(f"Input '{name}' contains infinite values in columns: {inf_columns}")
            
            validation_info['validation_passed'] = True
            validation_info['rows'] = len(data)
            validation_info['columns'] = len(data.columns)
            
            logger.debug(f"✅ DataFrame validation passed: {name} ({len(data)} rows, {len(data.columns)} columns)")
            
        except ValidationError as e:
            validation_info['error'] = str(e)
            validation_info['severity'] = e.severity.value
            logger.error(f"❌ DataFrame validation failed: {name} - {e}")
            raise
        except Exception as e:
            validation_info['error'] = str(e)
            validation_info['severity'] = 'critical'
            logger.error(f"❌ DataFrame validation error: {name} - {e}")
            raise ValidationError(f"DataFrame validation error: {e}", severity=ErrorSeverity.CRITICAL)
        finally:
            self.validation_history.append(validation_info)
        
        return data
    
    def validate_array(self, data: Any, name: str = "array", 
                      shape: Optional[Tuple[int, ...]] = None,
                      dtype: Optional[Type] = None,
                      min_elements: int = 1) -> np.ndarray:
        """Validate array input."""
        validation_info = {
            'timestamp': time.time(),
            'component': 'array_validation',
            'input_name': name,
            'validation_passed': False
        }
        
        try:
            # Convert to numpy array if needed
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            
            # Check shape
            if shape is not None:
                if data.shape != shape:
                    raise ValidationError(f"Input '{name}' shape {data.shape} doesn't match expected {shape}")
            
            # Check minimum elements
            if data.size < min_elements:
                raise ValidationError(f"Input '{name}' has {data.size} elements, minimum {min_elements} required")
            
            # Check dtype
            if dtype is not None:
                if not np.issubdtype(data.dtype, dtype):
                    logger.warning(f"Input '{name}' dtype {data.dtype} doesn't match expected {dtype}")
            
            # Check for NaN values
            if not self.config.allow_nan_values:
                if np.isnan(data).any():
                    if self.config.strict_validation:
                        raise ValidationError(f"Input '{name}' contains NaN values")
                    else:
                        logger.warning(f"Input '{name}' contains NaN values")
            
            # Check for infinite values
            if not self.config.allow_infinite_values:
                if np.isinf(data).any():
                    if self.config.strict_validation:
                        raise ValidationError(f"Input '{name}' contains infinite values")
                    else:
                        logger.warning(f"Input '{name}' contains infinite values")
            
            validation_info['validation_passed'] = True
            validation_info['shape'] = data.shape
            validation_info['dtype'] = str(data.dtype)
            
            logger.debug(f"✅ Array validation passed: {name} {data.shape}")
            
        except ValidationError as e:
            validation_info['error'] = str(e)
            validation_info['severity'] = e.severity.value
            logger.error(f"❌ Array validation failed: {name} - {e}")
            raise
        except Exception as e:
            validation_info['error'] = str(e)
            validation_info['severity'] = 'critical'
            logger.error(f"❌ Array validation error: {name} - {e}")
            raise ValidationError(f"Array validation error: {e}", severity=ErrorSeverity.CRITICAL)
        finally:
            self.validation_history.append(validation_info)
        
        return data
    
    def validate_model_config(self, config: Any, name: str = "config") -> Dict[str, Any]:
        """Validate model configuration."""
        validation_info = {
            'timestamp': time.time(),
            'component': 'config_validation',
            'input_name': name,
            'validation_passed': False
        }
        
        try:
            # Convert to dictionary if needed
            if hasattr(config, '__dict__'):
                config_dict = config.__dict__
            elif isinstance(config, dict):
                config_dict = config
            else:
                raise ValidationError(f"Input '{name}' must be a configuration object or dictionary")
            
            # Validate required parameters
            required_params = ['input_dim', 'output_dim']
            missing_params = [param for param in required_params if param not in config_dict]
            if missing_params:
                raise ValidationError(f"Configuration '{name}' missing required parameters: {missing_params}")
            
            # Validate parameter types and ranges
            if 'input_dim' in config_dict:
                if not isinstance(config_dict['input_dim'], int) or config_dict['input_dim'] <= 0:
                    raise ValidationError(f"Configuration '{name}' input_dim must be a positive integer")
            
            if 'output_dim' in config_dict:
                if not isinstance(config_dict['output_dim'], int) or config_dict['output_dim'] <= 0:
                    raise ValidationError(f"Configuration '{name}' output_dim must be a positive integer")
            
            if 'seq_length' in config_dict:
                if not isinstance(config_dict['seq_length'], int) or config_dict['seq_length'] <= 0:
                    raise ValidationError(f"Configuration '{name}' seq_length must be a positive integer")
            
            validation_info['validation_passed'] = True
            validation_info['parameters'] = list(config_dict.keys())
            
            logger.debug(f"✅ Config validation passed: {name}")
            
        except ValidationError as e:
            validation_info['error'] = str(e)
            validation_info['severity'] = e.severity.value
            logger.error(f"❌ Config validation failed: {name} - {e}")
            raise
        except Exception as e:
            validation_info['error'] = str(e)
            validation_info['severity'] = 'critical'
            logger.error(f"❌ Config validation error: {name} - {e}")
            raise ValidationError(f"Config validation error: {e}", severity=ErrorSeverity.CRITICAL)
        finally:
            self.validation_history.append(validation_info)
        
        return config_dict

class ErrorRecovery:
    """Advanced error recovery system."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.recovery_history: List[Dict[str, Any]] = []
        self.fallback_strategies: Dict[str, Callable] = {}
        
        # Initialize fallback strategies
        self._init_fallback_strategies()
        
        logger.info("🔄 Error recovery system initialized")
    
    def _init_fallback_strategies(self):
        """Initialize fallback strategies."""
        self.fallback_strategies = {
            'simplified': self._simplified_fallback,
            'cached': self._cached_fallback,
            'default': self._default_fallback
        }
    
    def attempt_recovery(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """Attempt error recovery with multiple strategies."""
        recovery_info = {
            'timestamp': time.time(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'recovery_attempted': False,
            'recovery_successful': False,
            'strategy_used': None
        }
        
        try:
            recovery_info['recovery_attempted'] = True
            
            # Try different recovery strategies
            for strategy_name in self.config.fallback_strategies:
                if strategy_name in self.fallback_strategies:
                    try:
                        logger.info(f"🔄 Attempting recovery with strategy: {strategy_name}")
                        result = self.fallback_strategies[strategy_name](error, context)
                        
                        if result is not None:
                            recovery_info['recovery_successful'] = True
                            recovery_info['strategy_used'] = strategy_name
                            logger.info(f"✅ Recovery successful with strategy: {strategy_name}")
                            break
                    
                    except Exception as recovery_error:
                        logger.warning(f"Recovery strategy {strategy_name} failed: {recovery_error}")
                        continue
            
            if not recovery_info['recovery_successful']:
                logger.error("❌ All recovery strategies failed")
            
        except Exception as e:
            logger.error(f"Recovery system error: {e}")
        finally:
            self.recovery_history.append(recovery_info)
        
        return recovery_info['recovery_successful'], recovery_info
    
    def _simplified_fallback(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Simplified fallback strategy."""
        logger.info("🔄 Using simplified fallback strategy")
        
        # Return simplified version of the operation
        if 'data' in context:
            data = context['data']
            if isinstance(data, pd.DataFrame):
                # Return first few rows
                return data.head(100)
            elif isinstance(data, np.ndarray):
                # Return first few samples
                return data[:100]
        
        return None
    
    def _cached_fallback(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Cached fallback strategy."""
        logger.info("🔄 Using cached fallback strategy")
        
        # Try to use cached results
        if 'cache_key' in context:
            # Implementation would depend on cache system
            logger.info(f"Attempting to retrieve cached result for key: {context['cache_key']}")
        
        return None
    
    def _default_fallback(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Default fallback strategy."""
        logger.info("🔄 Using default fallback strategy")
        
        # Return default values based on context
        if 'expected_shape' in context:
            shape = context['expected_shape']
            return np.zeros(shape)
        
        return None

class RobustErrorHandler:
    """Main robust error handling system."""
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        
        # Initialize components
        self.validator = InputValidator(self.config)
        self.recovery = ErrorRecovery(self.config)
        
        # Error tracking
        self.error_reports: List[ErrorReport] = []
        self.performance_impact: float = 0.0
        
        logger.info("🛡️ Robust error handler initialized")
    
    def validate_inputs(self, **kwargs) -> Dict[str, Any]:
        """Validate all input parameters."""
        validation_results = {}
        
        for param_name, param_value in kwargs.items():
            try:
                if isinstance(param_value, pd.DataFrame):
                    validation_results[param_name] = self.validator.validate_dataframe(
                        param_value, param_name
                    )
                elif isinstance(param_value, np.ndarray):
                    validation_results[param_name] = self.validator.validate_array(
                        param_value, param_name
                    )
                elif hasattr(param_value, '__dict__'):
                    validation_results[param_name] = self.validator.validate_model_config(
                        param_value, param_name
                    )
                else:
                    validation_results[param_name] = param_value
                    
            except ValidationError as e:
                if self.config.strict_validation:
                    raise
                else:
                    logger.warning(f"Input validation warning for {param_name}: {e}")
                    validation_results[param_name] = param_value
        
        return validation_results
    
    def handle_operation(self, operation_func: Callable, *args, **kwargs) -> Tuple[bool, Any, ErrorReport]:
        """Handle an operation with comprehensive error handling."""
        start_time = time.time()
        error_report = None
        
        try:
            # Validate inputs if enabled
            if self.config.validate_inputs:
                validated_kwargs = self.validate_inputs(**kwargs)
                kwargs.update(validated_kwargs)
            
            # Execute operation
            result = operation_func(*args, **kwargs)
            
            # Calculate performance impact
            execution_time = time.time() - start_time
            if execution_time > self.config.performance_threshold:
                self.performance_impact += execution_time
                logger.warning(f"⚠️ Operation took {execution_time:.2f}s (threshold: {self.config.performance_threshold}s)")
            
            return True, result, None
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Create error report
            error_report = ErrorReport(
                timestamp=time.time(),
                error_type=type(e).__name__,
                error_message=str(e),
                severity=self._determine_error_severity(e),
                component=operation_func.__name__,
                function_name=operation_func.__name__,
                input_validation=self.validator.validation_history[-1] if self.validator.validation_history else {},
                stack_trace=traceback.format_exc(),
                performance_impact=execution_time
            )
            
            # Attempt recovery if enabled
            if self.config.enable_error_recovery:
                context = {
                    'args': args,
                    'kwargs': kwargs,
                    'function_name': operation_func.__name__
                }
                
                recovery_successful, recovery_info = self.recovery.attempt_recovery(e, context)
                error_report.recovery_attempted = True
                error_report.recovery_successful = recovery_successful
                
                if recovery_successful:
                    logger.info("✅ Error recovery successful")
                    return True, recovery_info, error_report
                else:
                    error_report.fallback_used = True
                    logger.warning("⚠️ Using fallback after recovery failure")
            
            # Log error if severity meets threshold
            if self._should_log_error(error_report):
                logger.error(f"❌ Operation failed: {error_report.error_message}")
                if error_report.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                    logger.error(f"Stack trace: {error_report.stack_trace}")
            
            # Store error report
            self.error_reports.append(error_report)
            
            return False, None, error_report
    
    def _determine_error_severity(self, error: Exception) -> ErrorSeverity:
        """Determine error severity based on error type and message."""
        if isinstance(error, ValidationError):
            return error.severity
        elif isinstance(error, (MemoryError, OSError)):
            return ErrorSeverity.CRITICAL
        elif isinstance(error, (ValueError, TypeError)):
            return ErrorSeverity.HIGH
        elif isinstance(error, (Warning, UserWarning)):
            return ErrorSeverity.LOW
        else:
            return ErrorSeverity.MEDIUM
    
    def _should_log_error(self, error_report: ErrorReport) -> bool:
        """Determine if error should be logged based on severity."""
        severity_levels = {
            ErrorSeverity.LOW: 1,
            ErrorSeverity.MEDIUM: 2,
            ErrorSeverity.HIGH: 3,
            ErrorSeverity.CRITICAL: 4
        }
        
        threshold_level = severity_levels.get(self.config.error_reporting_threshold, 2)
        error_level = severity_levels.get(error_report.severity, 2)
        
        return error_level >= threshold_level
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get comprehensive error summary."""
        if not self.error_reports:
            return {'total_errors': 0}
        
        severity_counts = {}
        for report in self.error_reports:
            severity = report.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'total_errors': len(self.error_reports),
            'severity_breakdown': severity_counts,
            'recovery_attempts': sum(1 for r in self.error_reports if r.recovery_attempted),
            'recovery_successes': sum(1 for r in self.error_reports if r.recovery_successful),
            'fallback_usage': sum(1 for r in self.error_reports if r.fallback_used),
            'total_performance_impact': sum(r.performance_impact for r in self.error_reports),
            'validation_history': self.validator.validation_history
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status."""
        error_summary = self.get_error_summary()
        
        # Calculate health score
        total_errors = error_summary.get('total_errors', 0)
        critical_errors = error_summary.get('severity_breakdown', {}).get('critical', 0)
        high_errors = error_summary.get('severity_breakdown', {}).get('high', 0)
        
        health_score = 100
        if critical_errors > 0:
            health_score -= critical_errors * 20
        if high_errors > 0:
            health_score -= high_errors * 10
        if total_errors > 10:
            health_score -= (total_errors - 10) * 2
        
        health_score = max(0, health_score)
        
        return {
            'health_score': health_score,
            'status': 'healthy' if health_score > 80 else 'degraded' if health_score > 50 else 'critical',
            'error_summary': error_summary,
            'performance_impact': self.performance_impact
        }

def robust_operation(operation_func: Callable) -> Callable:
    """Decorator for robust operation handling."""
    @wraps(operation_func)
    def wrapper(*args, **kwargs):
        # This would be used with a global error handler instance
        # For now, just execute the function with basic error handling
        try:
            return operation_func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Operation {operation_func.__name__} failed: {e}")
            raise
    
    return wrapper

# Factory functions
def create_robust_error_handler(config: Optional[ValidationConfig] = None) -> RobustErrorHandler:
    """Create robust error handler."""
    return RobustErrorHandler(config)