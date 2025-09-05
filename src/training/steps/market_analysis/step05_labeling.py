"""Step 5: Labeling with Simplified Architecture.

This module provides a simplified, well-structured labeling step that maintains
all functionality while dramatically reducing complexity through modular design.

Key Simplifications:
- Extracted monitoring systems into separate modules
- Extracted decorator system with fallback mechanisms  
- Extracted labeling components into focused classes
- Centralized dependency management
- Simplified main class focused on core functionality
"""
import asyncio
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import time
from datetime import datetime
import json
import hashlib
import numpy as np
import pandas as pd
import traceback
import inspect
from functools import wraps
from dataclasses import dataclass, field
from enum import Enum
import re
import os
import gc
from collections import defaultdict, Counter
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from src.utils.common_operations import ensure_directory, safe_json_dump
from src.utils.pipeline_standards import PipelineStandards, pipeline_standards
REQUIRED_MODULES = [
    'pandas', 'numpy', 'psutil', 
    'src.utils.centralized_decorators', 
    'src.utils.logger', 
    'src.utils.enhanced_mlflow_integration', 
    'src.analyst.meta_labeling_system',
    'threading', 'multiprocessing', 'concurrent.futures',
    'collections', 'gc', 'warnings', 're', 'os'
]
dependency_status = PipelineStandards.validate_environment_dependencies(REQUIRED_MODULES)
centralized_decorators = PipelineStandards.safe_import('src.utils.centralized_decorators', None)
from src.utils.logger import system_logger
enhanced_mlflow = PipelineStandards.safe_import('src.utils.enhanced_mlflow_integration', None)
meta_labeling_system = PipelineStandards.safe_import('src.analyst.meta_labeling_system', None)
psutil = PipelineStandards.safe_import('psutil', None)
numpy = PipelineStandards.safe_import('numpy', None)
pandas = PipelineStandards.safe_import('pandas', None)

# Additional imports for comprehensive monitoring
threading_module = PipelineStandards.safe_import('threading', None)
multiprocessing_module = PipelineStandards.safe_import('multiprocessing', None)
concurrent_futures = PipelineStandards.safe_import('concurrent.futures', None)
collections_module = PipelineStandards.safe_import('collections', None)
gc_module = PipelineStandards.safe_import('gc', None)
warnings_module = PipelineStandards.safe_import('warnings', None)
re_module = PipelineStandards.safe_import('re', None)
os_module = PipelineStandards.safe_import('os', None)

# =============================================================================
# SIMPLIFIED MONITORING SYSTEM
# =============================================================================

@dataclass
class SimpleCallRecord:
    """Simple record of a function call."""
    function_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    execution_time: float = 0.0
    success: bool = True
    error: Optional[str] = None

class SimpleMonitor:
    """Simplified function call monitoring system."""
    
    def __init__(self, logger: Any = None):
        self.logger = logger or create_fallback_logger()
        self.call_history: List[SimpleCallRecord] = []
        
    def start_call(self, function_name: str) -> SimpleCallRecord:
        """Start monitoring a function call."""
        record = SimpleCallRecord(function_name=function_name, start_time=datetime.now())
        self.logger.info(f"🚀 Starting: {function_name}")
        return record
    
    def end_call(self, record: SimpleCallRecord, success: bool = True, error: str = None) -> None:
        """End monitoring a function call."""
        record.end_time = datetime.now()
        record.execution_time = (record.end_time - record.start_time).total_seconds()
        record.success = success
        record.error = error
        
        if success:
            self.logger.info(f"✅ Completed: {record.function_name} in {record.execution_time:.3f}s")
        else:
            self.logger.error(f"❌ Failed: {record.function_name} - {error}")
        
        self.call_history.append(record)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get simple summary of function calls."""
        if not self.call_history:
            return {'total_calls': 0, 'success_rate': 0, 'total_time': 0}
        
        total_calls = len(self.call_history)
        successful_calls = len([c for c in self.call_history if c.success])
        total_time = sum(c.execution_time for c in self.call_history)
        
        return {
            'total_calls': total_calls,
            'successful_calls': successful_calls,
            'failed_calls': total_calls - successful_calls,
            'success_rate': successful_calls / total_calls * 100,
            'total_time': total_time,
            'average_time': total_time / total_calls
        }

def simple_monitor(monitor: SimpleMonitor):
    """Simple decorator for function call monitoring."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            record = monitor.start_call(func.__name__)
            try:
                result = await func(*args, **kwargs)
                monitor.end_call(record, success=True)
                return result
            except Exception as e:
                monitor.end_call(record, success=False, error=str(e))
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            record = monitor.start_call(func.__name__)
            try:
                result = func(*args, **kwargs)
                monitor.end_call(record, success=True)
                return result
            except Exception as e:
                monitor.end_call(record, success=False, error=str(e))
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

class SimpleErrorHandler:
    """Simplified error handling system."""
    
    def __init__(self, logger: Any = None):
        self.logger = logger or create_fallback_logger()
        self.error_count = 0
        self.error_types: Dict[str, int] = {}
        
    def handle_error(self, function_name: str, error: Exception) -> None:
        """Handle and log function errors."""
        self.error_count += 1
        error_type = type(error).__name__
        self.error_types[error_type] = self.error_types.get(error_type, 0) + 1
        
        self.logger.error(f"❌ Error in {function_name}: {error_type} - {str(error)}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get simple error summary."""
        return {
            'total_errors': self.error_count,
            'error_types': self.error_types.copy()
        }

def simple_error_handler(handler: SimpleErrorHandler):
    """Simple decorator for error handling."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                handler.handle_error(func.__name__, e)
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handler.handle_error(func.__name__, e)
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

class SimplePerformanceMonitor:
    """Simplified performance monitoring system."""
    
    def __init__(self, logger: Any = None):
        self.logger = logger or create_fallback_logger()
        self.execution_times: Dict[str, List[float]] = {}
        
    def record_execution_time(self, function_name: str, execution_time: float) -> None:
        """Record execution time for a function."""
        if function_name not in self.execution_times:
            self.execution_times[function_name] = []
        self.execution_times[function_name].append(execution_time)
        
        if execution_time > 60:
            self.logger.warning(f"⚠️ Slow execution: {function_name} took {execution_time:.3f}s")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get simple performance summary."""
        summary = {}
        for function_name, times in self.execution_times.items():
            summary[function_name] = {
                'total_calls': len(times),
                'average_time': sum(times) / len(times),
                'max_time': max(times),
                'min_time': min(times)
            }
        return summary

def simple_performance_monitor(monitor: SimplePerformanceMonitor):
    """Simple decorator for performance monitoring."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                monitor.record_execution_time(func.__name__, execution_time)
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                monitor.record_execution_time(func.__name__, execution_time)
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                monitor.record_execution_time(func.__name__, execution_time)
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                monitor.record_execution_time(func.__name__, execution_time)
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

class SimpleValidator:
    """Simplified validation system."""
    
    def __init__(self, logger: Any = None):
        self.logger = logger or create_fallback_logger()
        self.validation_count = 0
        self.validation_errors = 0
    
    def validate_dataframe(self, data: Any, required_columns: List[str] = None) -> bool:
        """Simple DataFrame validation."""
        self.validation_count += 1
        
        if not isinstance(data, pd.DataFrame):
            self.logger.error(f"❌ Expected DataFrame, got {type(data).__name__}")
            self.validation_errors += 1
            return False
        
        if data.empty:
            self.logger.warning("⚠️ DataFrame is empty")
            return False
        
        if required_columns:
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                self.logger.error(f"❌ Missing required columns: {missing_columns}")
                self.validation_errors += 1
                return False
        
        return True
    
    def validate_labels(self, data: pd.DataFrame, label_column: str = 'label') -> bool:
        """Simple label validation."""
        if label_column not in data.columns:
            return True  # No labels to validate
        
        labels = data[label_column].dropna()
        if len(labels) == 0:
            self.logger.warning("⚠️ No labels found")
            return False
        
        # Check for reasonable label distribution
        label_counts = labels.value_counts()
        if len(label_counts) == 1:
            self.logger.warning("⚠️ Only one label class found")
        
        return True
    
    def get_summary(self) -> Dict[str, Any]:
        """Get simple validation summary."""
        return {
            'total_validations': self.validation_count,
            'validation_errors': self.validation_errors,
            'success_rate': (self.validation_count - self.validation_errors) / self.validation_count * 100 if self.validation_count > 0 else 0
        }
    
def simple_validation(validator: SimpleValidator, required_columns: List[str] = None):
    """Simple decorator for validation."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Validate first argument if it's a DataFrame
            if args and isinstance(args[0], pd.DataFrame):
                if not validator.validate_dataframe(args[0], required_columns):
                    raise ValueError(f"Input validation failed for {func.__name__}")
            
            result = await func(*args, **kwargs)
            
            # Validate output if it's a DataFrame
            if isinstance(result, pd.DataFrame):
                validator.validate_labels(result)
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Validate first argument if it's a DataFrame
            if args and isinstance(args[0], pd.DataFrame):
                if not validator.validate_dataframe(args[0], required_columns):
                    raise ValueError(f"Input validation failed for {func.__name__}")
            
            result = func(*args, **kwargs)
            
            # Validate output if it's a DataFrame
            if isinstance(result, pd.DataFrame):
                validator.validate_labels(result)
            
            return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

def create_fallback_logger() -> Any:
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

def create_fallback_decorator() -> Any:
    def decorator(func: Callable) -> None:
        return func
    return decorator
# Initialize system logger and decorators
if system_logger is None:
    system_logger = create_fallback_logger()

# Initialize centralized decorators
if centralized_decorators is None:
    comprehensive_data_validation = create_fallback_decorator()
    handle_errors = create_fallback_decorator()
    memory_efficient = create_fallback_decorator()
    resource_monitor = create_fallback_decorator()
    secure_data_processing = create_fallback_decorator()
    validate_data_structure = create_fallback_decorator()
    with_tracing_span = create_fallback_decorator()
    quality_gate = create_fallback_decorator()
    monitor_feature_engineering = create_fallback_decorator()
else:
    comprehensive_data_validation = centralized_decorators.comprehensive_data_validation
    handle_errors = centralized_decorators.handle_errors
    memory_efficient = centralized_decorators.memory_efficient
    resource_monitor = centralized_decorators.resource_monitor
    secure_data_processing = centralized_decorators.secure_data_processing
    validate_data_structure = centralized_decorators.validate_data_structure
    with_tracing_span = centralized_decorators.with_tracing_span
    quality_gate = centralized_decorators.quality_gate
    monitor_feature_engineering = centralized_decorators.monitor_feature_engineering

# Initialize MLflow decorators
if enhanced_mlflow is None:
    with_enhanced_mlflow_logging = create_fallback_decorator()
    log_step_report = lambda *args, **kwargs: 'fallback_report'
    create_detailed_step_report = lambda *args, **kwargs: {}
    log_step_metrics = lambda *args, **kwargs: None
    log_step_dataframe_with_standardized_name = lambda *args, **kwargs: 'fallback_dataframe'
    log_step_artifact_with_standardized_name = lambda *args, **kwargs: 'fallback_artifact'
else:
    with_enhanced_mlflow_logging = enhanced_mlflow.with_enhanced_mlflow_logging
    log_step_report = enhanced_mlflow.log_step_report
    create_detailed_step_report = enhanced_mlflow.create_detailed_step_report
    log_step_metrics = enhanced_mlflow.log_step_metrics
    log_step_dataframe_with_standardized_name = enhanced_mlflow.log_step_dataframe_with_standardized_name
    log_step_artifact_with_standardized_name = enhanced_mlflow.log_step_artifact_with_standardized_name

logger = system_logger.getChild('Step5Labeling')

class LabelingStep:
    """Step 5: Labeling with simplified monitoring and validation."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild('LabelingStep')
        self.standards = pipeline_standards
        self.start_time = None
        self.step_timings = {}
        
        # Initialize monitoring systems (using existing decorators)
        self.function_monitor = SimpleMonitor(self.logger)
        self.error_handler = SimpleErrorHandler(self.logger)
        self.performance_monitor = SimplePerformanceMonitor(self.logger)
        self.validator = SimpleValidator(self.logger)
        
        self._validate_environment()
        self._initialize_components()

    def _validate_environment(self) -> None:
        """Validate environment dependencies."""
        self.logger.info('🔍 Validating environment dependencies...')
        missing_modules = [module for module, available in dependency_status.items() if not available]
        if missing_modules:
            self.logger.warning(f'⚠️ Missing optional modules: {missing_modules}')
            self.logger.info('📝 Pipeline will continue with fallback implementations')
        else:
            self.logger.info('✅ All required dependencies available')

    def _initialize_components(self) -> None:
        """Initialize labeling components."""
        self.logger.info('🔧 Initializing labeling components...')
        labeling_cfg = self.config.get('vectorized_labelling_orchestrator', {})
        self.auto_recalculate_hmm_barriers = bool(labeling_cfg.get('auto_recalculate_hmm_barriers', True))
        self.regime_col = str(labeling_cfg.get('hmm_barrier_regime_column', 'hmm_regime'))
        self.time_barrier_minutes = int(labeling_cfg.get('time_barrier_minutes', 30))
        self.max_lookahead = int(labeling_cfg.get('max_lookahead', 100))
        self.logger.info(f'📋 Labeling configuration loaded')

def validate_string_field(data: str, max_length: int, context: dict) -> dict:
    """Validate string field data."""
    result = {'valid': True, 'errors': []}
    
    if len(data) > max_length:
                result['valid'] = False
                result['errors'].append(f"String too long (max: {max_length})")
            
            # Check pattern constraints
            pattern = context.get('pattern')
            if pattern and not re.match(pattern, data):
                result['valid'] = False
                result['errors'].append(f"String doesn't match required pattern: {pattern}")
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result
    
    def _validate_numeric_input(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate numeric input."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            if not isinstance(data, (int, float, np.number)):
                result['valid'] = False
                result['errors'].append(f"Expected numeric, got {type(data).__name__}")
                return result
            
            # Check range constraints
            min_value = context.get('min_value', float('-inf'))
            max_value = context.get('max_value', float('inf'))
            
            if data < min_value:
                result['valid'] = False
                result['errors'].append(f"Value too small (min: {min_value})")
            
            if data > max_value:
                result['valid'] = False
                result['errors'].append(f"Value too large (max: {max_value})")
            
            # Check for NaN or infinite values
            if np.isnan(data) or np.isinf(data):
                result['valid'] = False
                result['errors'].append("Value is NaN or infinite")
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result
    
    def _validate_path_input(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate path input."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            path = Path(data) if not isinstance(data, Path) else data
            
            # Check if path exists
            must_exist = context.get('must_exist', True)
            if must_exist and not path.exists():
                result['valid'] = False
                result['errors'].append(f"Path does not exist: {path}")
                return result
            
            # Check if it's a file or directory
            expected_type = context.get('expected_type', 'file')  # 'file' or 'directory'
            if path.exists():
                if expected_type == 'file' and not path.is_file():
                    result['valid'] = False
                    result['errors'].append(f"Expected file, got directory: {path}")
                elif expected_type == 'directory' and not path.is_dir():
                    result['valid'] = False
                    result['errors'].append(f"Expected directory, got file: {path}")
            
            # Check file extension
            expected_extensions = context.get('expected_extensions', [])
            if expected_extensions and path.suffix.lower() not in expected_extensions:
                result['valid'] = False
                result['errors'].append(f"Invalid file extension. Expected: {expected_extensions}")
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result
    
    def _validate_dataframe_output(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate DataFrame output."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            if data is None:
                result['valid'] = False
                result['errors'].append("Output is None")
                return result
            
            if not isinstance(data, pd.DataFrame):
                result['valid'] = False
                result['errors'].append(f"Expected DataFrame output, got {type(data).__name__}")
                return result
            
            if data.empty:
                result['warnings'].append("Output DataFrame is empty")
            
            # Check for required output columns
            required_columns = context.get('required_columns', [])
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                result['valid'] = False
                result['errors'].append(f"Missing required output columns: {missing_columns}")
            
            # Check data quality
            if 'label' in data.columns:
                label_counts = data['label'].value_counts()
                if len(label_counts) == 0:
                    result['warnings'].append("No labels generated")
                elif len(label_counts) == 1:
                    result['warnings'].append("Only one label class generated")
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result
    
    def _validate_boolean_output(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate boolean output."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            if not isinstance(data, bool):
                result['valid'] = False
                result['errors'].append(f"Expected boolean output, got {type(data).__name__}")
                return result
            
            # Check expected value
            expected_value = context.get('expected_value')
            if expected_value is not None and data != expected_value:
                result['warnings'].append(f"Expected {expected_value}, got {data}")
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result
    
    def _validate_numeric_output(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate numeric output."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            if not isinstance(data, (int, float, np.number)):
                result['valid'] = False
                result['errors'].append(f"Expected numeric output, got {type(data).__name__}")
                return result
            
            # Check for NaN or infinite values
            if np.isnan(data) or np.isinf(data):
                result['valid'] = False
                result['errors'].append("Output is NaN or infinite")
            
            # Check range constraints
            min_value = context.get('min_value', float('-inf'))
            max_value = context.get('max_value', float('inf'))
            
            if data < min_value or data > max_value:
                result['warnings'].append(f"Output value {data} outside expected range [{min_value}, {max_value}]")
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result
    
    def _validate_series_output(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate Series output."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            if data is None:
                result['valid'] = False
                result['errors'].append("Output is None")
                return result
            
            if not isinstance(data, pd.Series):
                result['valid'] = False
                result['errors'].append(f"Expected Series output, got {type(data).__name__}")
                return result
            
            if data.empty:
                result['warnings'].append("Output Series is empty")
            
            # Check for NaN values
            if data.isna().any():
                nan_count = data.isna().sum()
                result['warnings'].append(f"Output Series contains {nan_count} NaN values")
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result
    
    def _validate_data_completeness(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate data completeness."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            if isinstance(data, pd.DataFrame):
                total_cells = data.size
                missing_cells = data.isna().sum().sum()
                completeness_ratio = (total_cells - missing_cells) / total_cells if total_cells > 0 else 0
                
                min_completeness = context.get('min_completeness', 0.95)
                if completeness_ratio < min_completeness:
                    result['warnings'].append(f"Data completeness {completeness_ratio:.2%} below threshold {min_completeness:.2%}")
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result
    
    def _validate_data_types(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate data types."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            if isinstance(data, pd.DataFrame):
                expected_types = context.get('expected_types', {})
                for col, expected_type in expected_types.items():
                    if col in data.columns:
                        actual_type = data[col].dtype
                        if not pd.api.types.is_dtype_equal(actual_type, expected_type):
                            result['warnings'].append(f"Column '{col}' type mismatch: {actual_type} vs {expected_type}")
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result
    
    def _validate_data_ranges(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate data ranges."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            if isinstance(data, pd.DataFrame):
                column_ranges = context.get('column_ranges', {})
                for col, (min_val, max_val) in column_ranges.items():
                    if col in data.columns:
                        col_data = data[col].dropna()
                        if len(col_data) > 0:
                            if col_data.min() < min_val or col_data.max() > max_val:
                                result['warnings'].append(f"Column '{col}' values outside range [{min_val}, {max_val}]")
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result
    
    def _validate_data_consistency(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate data consistency."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            if isinstance(data, pd.DataFrame):
                # Check for duplicate rows
                if data.duplicated().any():
                    duplicate_count = data.duplicated().sum()
                    result['warnings'].append(f"Found {duplicate_count} duplicate rows")
                
                # Check for inconsistent data patterns
                if 'close' in data.columns and 'high' in data.columns and 'low' in data.columns:
                    invalid_ohlc = (data['close'] > data['high']) | (data['close'] < data['low'])
                    if invalid_ohlc.any():
                        invalid_count = invalid_ohlc.sum()
                        result['warnings'].append(f"Found {invalid_count} rows with invalid OHLC relationships")
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result
    
    def _validate_execution_time(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate execution time."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            execution_time = context.get('execution_time', 0)
            max_time = context.get('max_execution_time', 300)  # 5 minutes default
            
            if execution_time > max_time:
                result['warnings'].append(f"Execution time {execution_time:.2f}s exceeds threshold {max_time}s")
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result
    
    def _validate_memory_usage(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate memory usage."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            memory_usage = context.get('memory_usage_mb', 0)
            max_memory = context.get('max_memory_mb', 1000)  # 1GB default
            
            if memory_usage > max_memory:
                result['warnings'].append(f"Memory usage {memory_usage:.1f}MB exceeds threshold {max_memory}MB")
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result
    
    def _validate_cpu_usage(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate CPU usage."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            cpu_usage = context.get('cpu_usage_percent', 0)
            max_cpu = context.get('max_cpu_percent', 80)  # 80% default
            
            if cpu_usage > max_cpu:
                result['warnings'].append(f"CPU usage {cpu_usage:.1f}% exceeds threshold {max_cpu}%")
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result
    
    def _validate_labeling_logic(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate labeling logic."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            if isinstance(data, pd.DataFrame) and 'label' in data.columns:
                labels = data['label'].dropna()
                
                # Check label distribution
                if len(labels) > 0:
                    label_counts = labels.value_counts()
                    total_labels = len(labels)
                    
                    # Check for extreme class imbalance
                    if len(label_counts) > 1:
                        max_count = label_counts.max()
                        min_count = label_counts.min()
                        imbalance_ratio = max_count / min_count
                        
                        if imbalance_ratio > 10:
                            result['warnings'].append(f"Severe class imbalance detected (ratio: {imbalance_ratio:.1f})")
                    
                    # Check for reasonable label distribution
                    for label, count in label_counts.items():
                        percentage = count / total_labels * 100
                        if percentage < 1:
                            result['warnings'].append(f"Very few samples for label {label} ({percentage:.1f}%)")
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result
    
    def _validate_regime_logic(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate regime logic."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            if isinstance(data, pd.DataFrame):
                regime_columns = [col for col in data.columns if 'regime' in col.lower()]
                
                for regime_col in regime_columns:
                    regimes = data[regime_col].dropna()
                    if len(regimes) > 0:
                        regime_counts = regimes.value_counts()
                        
                        # Check for reasonable number of regimes
                        if len(regime_counts) < 2:
                            result['warnings'].append(f"Only {len(regime_counts)} regime(s) detected in {regime_col}")
                        elif len(regime_counts) > 10:
                            result['warnings'].append(f"Too many regimes ({len(regime_counts)}) in {regime_col}")
                        
                        # Check for regime balance
                        if len(regime_counts) > 1:
                            max_count = regime_counts.max()
                            min_count = regime_counts.min()
                            balance_ratio = max_count / min_count
                            
                            if balance_ratio > 5:
                                result['warnings'].append(f"Unbalanced regimes in {regime_col} (ratio: {balance_ratio:.1f})")
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result
    
    def _validate_triple_barrier_logic(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate triple barrier logic."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            if isinstance(data, pd.DataFrame):
                tb_columns = [col for col in data.columns if 'triple_barrier' in col.lower()]
                
                for tb_col in tb_columns:
                    tb_labels = data[tb_col].dropna()
                    if len(tb_labels) > 0:
                        # Check for valid triple barrier labels (-1, 0, 1)
                        valid_labels = tb_labels.isin([-1, 0, 1])
                        if not valid_labels.all():
                            invalid_labels = tb_labels[~valid_labels].unique()
                            result['warnings'].append(f"Invalid triple barrier labels in {tb_col}: {invalid_labels}")
                        
                        # Check label distribution
                        label_counts = tb_labels.value_counts()
                        total_labels = len(tb_labels)
                        
                        # Check for too many neutral labels
                        neutral_count = label_counts.get(0, 0)
                        neutral_ratio = neutral_count / total_labels
                        
                        if neutral_ratio > 0.8:
                            result['warnings'].append(f"Too many neutral labels in {tb_col} ({neutral_ratio:.1%})")
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result
    
    def validate_function_input(self, function_name: str, input_data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate function input using all applicable rules."""
        try:
            validation_result = {
                'function_name': function_name,
                'validation_type': 'input',
                'timestamp': datetime.now().isoformat(),
                'overall_valid': True,
                'rule_results': {},
                'errors': [],
                'warnings': []
            }
            
            context = context or {}
            
            # Run input validation rules
            for rule_name, rules in self.validation_rules.items():
                if rule_name in ['input_validation', 'data_quality']:
                    for rule in rules:
                        try:
                            rule_result = rule(input_data, context)
                            validation_result['rule_results'][rule_name] = rule_result
                            
                            if not rule_result['valid']:
                                validation_result['overall_valid'] = False
                                validation_result['errors'].extend(rule_result['errors'])
                            
                            validation_result['warnings'].extend(rule_result['warnings'])
                            
                        except Exception as e:
                            validation_result['errors'].append(f"Rule {rule_name} failed: {str(e)}")
                            validation_result['overall_valid'] = False
            
            # Store validation result
            self.validation_history.append(validation_result)
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to validate function input: {e}")
            return {'overall_valid': False, 'errors': [str(e)], 'warnings': []}
    
    def validate_function_output(self, function_name: str, output_data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate function output using all applicable rules."""
        try:
            validation_result = {
                'function_name': function_name,
                'validation_type': 'output',
                'timestamp': datetime.now().isoformat(),
                'overall_valid': True,
                'rule_results': {},
                'errors': [],
                'warnings': []
            }
            
            context = context or {}
            
            # Run output validation rules
            for rule_name, rules in self.validation_rules.items():
                if rule_name in ['output_validation', 'data_quality', 'business_logic']:
                    for rule in rules:
                        try:
                            rule_result = rule(output_data, context)
                            validation_result['rule_results'][rule_name] = rule_result
                            
                            if not rule_result['valid']:
                                validation_result['overall_valid'] = False
                                validation_result['errors'].extend(rule_result['errors'])
                            
                            validation_result['warnings'].extend(rule_result['warnings'])
                            
                        except Exception as e:
                            validation_result['errors'].append(f"Rule {rule_name} failed: {str(e)}")
                            validation_result['overall_valid'] = False
            
            # Store validation result
            self.validation_history.append(validation_result)
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to validate function output: {e}")
            return {'overall_valid': False, 'errors': [str(e)], 'warnings': []}
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        try:
            if not self.validation_history:
                return {'total_validations': 0, 'message': 'No validation data recorded'}
            
            # Analyze validation results
            total_validations = len(self.validation_history)
            successful_validations = len([v for v in self.validation_history if v['overall_valid']])
            failed_validations = total_validations - successful_validations
            
            # Group by function
            function_validations = {}
            for validation in self.validation_history:
                func_name = validation['function_name']
                if func_name not in function_validations:
                    function_validations[func_name] = {'input': [], 'output': []}
                function_validations[func_name][validation['validation_type']].append(validation)
            
            # Analyze error patterns
            error_patterns = {}
            warning_patterns = {}
            
            for validation in self.validation_history:
                for error in validation['errors']:
                    error_patterns[error] = error_patterns.get(error, 0) + 1
                
                for warning in validation['warnings']:
                    warning_patterns[warning] = warning_patterns.get(warning, 0) + 1
            
            return {
                'total_validations': total_validations,
                'successful_validations': successful_validations,
                'failed_validations': failed_validations,
                'success_rate': (successful_validations / total_validations * 100) if total_validations > 0 else 0,
                'function_validations': function_validations,
                'error_patterns': error_patterns,
                'warning_patterns': warning_patterns,
                'most_common_errors': sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:5],
                'most_common_warnings': sorted(warning_patterns.items(), key=lambda x: x[1], reverse=True)[:5]
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate validation report: {e}")
            return {}
    
    def log_validation_report(self, report: Dict[str, Any]) -> None:
        """Log comprehensive validation report."""
        try:
            if report.get('total_validations', 0) == 0:
                self.logger.info("📋 No validation data recorded")
                return
            
            self.logger.info("📋 COMPREHENSIVE VALIDATION REPORT")
            self.logger.info("=" * 50)
            self.logger.info(f"Total Validations: {report['total_validations']}")
            self.logger.info(f"Successful Validations: {report['successful_validations']}")
            self.logger.info(f"Failed Validations: {report['failed_validations']}")
            self.logger.info(f"Success Rate: {report['success_rate']:.1f}%")
            
            # Function-specific validation results
            function_validations = report.get('function_validations', {})
            if function_validations:
                self.logger.info(f"\n🔍 FUNCTION VALIDATION RESULTS:")
                for func_name, validations in function_validations.items():
                    input_validations = validations.get('input', [])
                    output_validations = validations.get('output', [])
                    
                    input_success = len([v for v in input_validations if v['overall_valid']])
                    output_success = len([v for v in output_validations if v['overall_valid']])
                    
                    self.logger.info(f"   {func_name}:")
                    self.logger.info(f"     Input Validations: {input_success}/{len(input_validations)} successful")
                    self.logger.info(f"     Output Validations: {output_success}/{len(output_validations)} successful")
            
            # Most common errors
            most_common_errors = report.get('most_common_errors', [])
            if most_common_errors:
                self.logger.info(f"\n❌ MOST COMMON ERRORS:")
                for error, count in most_common_errors:
                    self.logger.info(f"   - {error}: {count} occurrences")
            
            # Most common warnings
            most_common_warnings = report.get('most_common_warnings', [])
            if most_common_warnings:
                self.logger.info(f"\n⚠️ MOST COMMON WARNINGS:")
                for warning, count in most_common_warnings:
                    self.logger.info(f"   - {warning}: {count} occurrences")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log validation report: {e}")

def comprehensive_validation(validator: ComprehensiveValidationFramework):
    """Decorator for comprehensive validation."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Validate inputs
            input_context = {
                'function_name': func.__name__,
                'args_count': len(args),
                'kwargs_keys': list(kwargs.keys())
            }
            
            # Validate first argument if it's a DataFrame
            if args and isinstance(args[0], pd.DataFrame):
                input_validation = validator.validate_function_input(func.__name__, args[0], input_context)
                if not input_validation['overall_valid']:
                    validator.logger.warning(f"⚠️ Input validation failed for {func.__name__}: {input_validation['errors']}")
            
            try:
                result = await func(*args, **kwargs)
                
                # Validate output
                output_context = {
                    'function_name': func.__name__,
                    'execution_time': getattr(func, '_execution_time', 0)
                }
                
                output_validation = validator.validate_function_output(func.__name__, result, output_context)
                if not output_validation['overall_valid']:
                    validator.logger.warning(f"⚠️ Output validation failed for {func.__name__}: {output_validation['errors']}")
                
                return result
                
            except Exception as e:
                # Log validation failure
                validator.logger.error(f"❌ Function {func.__name__} failed with error: {e}")
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Validate inputs
            input_context = {
                'function_name': func.__name__,
                'args_count': len(args),
                'kwargs_keys': list(kwargs.keys())
            }
            
            # Validate first argument if it's a DataFrame
            if args and isinstance(args[0], pd.DataFrame):
                input_validation = validator.validate_function_input(func.__name__, args[0], input_context)
                if not input_validation['overall_valid']:
                    validator.logger.warning(f"⚠️ Input validation failed for {func.__name__}: {input_validation['errors']}")
            
            try:
                result = func(*args, **kwargs)
                
                # Validate output
                output_context = {
                    'function_name': func.__name__,
                    'execution_time': getattr(func, '_execution_time', 0)
                }
                
                output_validation = validator.validate_function_output(func.__name__, result, output_context)
                if not output_validation['overall_valid']:
                    validator.logger.warning(f"⚠️ Output validation failed for {func.__name__}: {output_validation['errors']}")
                
                return result
                
            except Exception as e:
                # Log validation failure
                validator.logger.error(f"❌ Function {func.__name__} failed with error: {e}")
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


import numpy as np
import pandas as pd

logger = system_logger.getChild('Step5Labeling')

class LabelingStep:
    """Simplified Step 5: Labeling with modular architecture.

    This class focuses on core labeling functionality while delegating
    monitoring, validation, and complex logic to specialized modules.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild('LabelingStep')
        self.start_time: Optional[float] = None
        self.step_timings: Dict[str, float] = {}
        
        # Initialize core components
        self.comprehensive_labeling = ComprehensiveLabeling(config, self.logger)
        
        # Initialize monitoring systems (optional)
        self._initialize_monitoring_systems()
        
        # Validate environment
        self._validate_environment()
        
        self.logger.info('✅ LabelingStep initialized with simplified architecture')

    def _initialize_monitoring_systems(self) -> None:
        """Initialize monitoring systems if available."""
        try:
            # Initialize monitoring systems
            self.function_monitor = FunctionCallMonitor(self.logger)
            self.error_handler = EnhancedErrorHandler(self.logger)
            self.performance_monitor = PerformanceMonitor(self.logger)
            self.validation_framework = ComprehensiveValidationFramework(self.logger)
            
            # Setup function monitoring
            self._setup_function_monitoring()
            
            self.logger.info('✅ Monitoring systems initialized')
        except Exception as e:
            self.logger.warning(f'⚠️ Monitoring systems not available: {e}')
            self.function_monitor = None
            self.error_handler = None
            self.performance_monitor = None
            self.validation_framework = None

    def _setup_function_monitoring(self) -> None:
        """Setup function monitoring with validation rules and performance thresholds."""
        if self.function_monitor is None:
            return
        
        # Set performance thresholds for key functions
        self.function_monitor.performance_thresholds = {
            'execute_labeling': 300.0,  # 5 minutes
            'generate_comprehensive_labels': 180.0,  # 3 minutes
        }
        
        # Set custom validation rules
        self.function_monitor.validation_rules = {
            'execute_labeling': self._validate_execute_labeling_result,
            'generate_comprehensive_labels': self._validate_labeling_result,
        }

    def _validate_execute_labeling_result(self, call_record) -> bool:
        """Validate execute_labeling function result."""
        if call_record.return_value is None:
            return False
        return isinstance(call_record.return_value, bool)

    def _validate_labeling_result(self, call_record) -> bool:
        """Validate labeling function result."""
        if call_record.return_value is None:
            return False
        
        # Check if return value is a DataFrame
        if not isinstance(call_record.return_value, pd.DataFrame):
            return False
        
        # Check if DataFrame has required columns
        required_columns = ['label']
        if not all(col in call_record.return_value.columns for col in required_columns):
            return False
        
        # Check if DataFrame is not empty
        if len(call_record.return_value) == 0:
            return False
        
        return True

    def _validate_regime_labels_result(self, call_record: FunctionCallRecord) -> bool:
        """Validate regime labels function result."""
        if call_record.return_value is None:
            return False
        
        # Check if return value is a Series
        if not isinstance(call_record.return_value, pd.Series):
            return False
        
        # Check if Series is not empty
        if len(call_record.return_value) == 0:
            return False
        
        return True

    def _validate_triple_barrier_result(self, call_record: FunctionCallRecord) -> bool:
        """Validate triple barrier function result."""
        if call_record.return_value is None:
            return False
        
        # Check if return value is a DataFrame
        if not isinstance(call_record.return_value, pd.DataFrame):
            return False
        
        # Check if DataFrame has required columns
        required_columns = ['triple_barrier_label']
        if not all(col in call_record.return_value.columns for col in required_columns):
            return False
        
        return True

    def _validate_meta_labels_result(self, call_record: FunctionCallRecord) -> bool:
        """Validate meta labels function result."""
        if call_record.return_value is None:
            return False
        
        # Check if return value is a DataFrame
        if not isinstance(call_record.return_value, pd.DataFrame):
            return False
        
        return True

    @with_tracing_span(span_name='compute_labeling_fingerprint')
    def _compute_labeling_fingerprint(self, triple_barrier_path: Path) -> Dict[str, Any]:
        """Compute a stable fingerprint of source labeling inputs to ensure idempotence.

        Uses source file size and mtime plus relevant config toggles.
        """
        try:
            stat = triple_barrier_path.stat()
            relevant_cfg = {'vectorized_labelling_orchestrator': self.config.get('vectorized_labelling_orchestrator', {}), 'labeling': self.config.get('labeling', {}), 'time_barrier_minutes': getattr(self, 'time_barrier_minutes', None), 'max_lookahead': getattr(self, 'max_lookahead', None), 'regime_col': getattr(self, 'regime_col', None), 'auto_recalculate_hmm_barriers': getattr(self, 'auto_recalculate_hmm_barriers', None)}
            relevant_cfg_json = json.dumps(relevant_cfg, sort_keys=True, default=str)
            cfg_hash = hashlib.sha256(relevant_cfg_json.encode('utf-8')).hexdigest()
            return {'source_path': str(triple_barrier_path), 'source_size': stat.st_size, 'source_mtime': int(stat.st_mtime), 'config_hash': cfg_hash}
        except Exception as e:
            self.logger.warning(f'⚠️ Failed to compute labeling fingerprint: {e}')
            return {}

    def _validate_environment(self) -> None:
        """Validate environment dependencies."""
        self.logger.info('🔍 Validating environment dependencies...')
        missing_modules = [k for k, ok in dependency_manager.get_dependency_status().items() if not ok]
        if missing_modules:
            self.logger.warning(f'⚠️ Missing optional modules: {missing_modules}')
            self.logger.info('📝 Pipeline will continue with fallback implementations')
        else:
            self.logger.info('✅ All required dependencies available')

    async def initialize(self) -> None:
        """Initialize the labeling step."""
        self.start_time = time.time()
        self.logger.info('🚀 Initializing Labeling Step...')
        self.logger.info('📋 Step 5 Configuration:')
        self.logger.info(f"   - Symbol: {self.config.get('SYMBOL', 'N/A')}")
        self.logger.info(f"   - Exchange: {self.config.get('EXCHANGE', 'N/A')}")
        self.logger.info(f"   - Timeframe: {self.config.get('TIMEFRAME', 'N/A')}")
        self.logger.info(f"   - Data Directory: {self.config.get('DATA_DIR', 'N/A')}")
        self.logger.info('✅ Labeling Step initialized successfully')

    def _log_step_timing(self, step_name: str, start_time: float) -> None:
        """Log timing information for a step."""
        elapsed = time.time() - start_time
        self.step_timings[step_name] = elapsed
        self.logger.info(f'⏱️ {step_name} completed in {elapsed:.2f} seconds')

    @with_tracing_span(span_name='execute_labeling')
    @comprehensive_data_validation()
    @handle_errors()
    @resource_monitor()
    @quality_gate()
    @log_execution_time()
    @comprehensive_data_validation
    @memory_efficient
    @resource_monitor
    @secure_data_processing
    @validate_data_structure
    async def execute_labeling(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str = 'data_cache',
        force_rerun: bool = False,
    ) -> bool:
        """Execute the labeling step with comprehensive monitoring."""
        step_start = time.time()
        self.logger.info(f'🚀 Executing Labeling for {symbol} on {exchange}')
        
        try:
            # Setup paths
            triple_barrier_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_triple_barrier_labels.parquet'
            if not triple_barrier_path.exists():
                self.logger.error(f'❌ Triple barrier labels not found at {triple_barrier_path}')
                return False
            
            self.logger.info(f'📁 Loading triple barrier labels from {triple_barrier_path}')
            labeled_dir = ensure_directory(Path(data_dir) / 'training' / 'labeled_data')
            output_path = labeled_dir / f'{exchange}_{symbol}_{timeframe}_labeled_data.parquet'
            metadata_path = labeled_dir / f'{exchange}_{symbol}_{timeframe}_labeling_metadata.json'
            
            # Check for idempotence
            current_fp = self._compute_labeling_fingerprint(triple_barrier_path)
            if not force_rerun and output_path.exists() and metadata_path.exists():
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        existing_meta = json.load(f)
                    existing_fp = existing_meta.get('source_fingerprint', {})
                    if existing_fp == current_fp and existing_meta.get('total_samples', 0) > 0:
                        self.logger.info('🟢 Labeling is idempotent: existing outputs match current inputs. Skipping recomputation.')
                        self._log_step_timing('Labeling (skipped)', step_start)
                        return True
                except Exception as e:
                    self.logger.warning(f'⚠️ Failed to read existing labeling metadata: {e}')
            
            # Load data
            data = pd.read_parquet(triple_barrier_path)
            self.logger.info(f'✅ Loaded data with shape: {data.shape}')
            
            # Ensure regime labels are present/consistent
            try:
                from .utils.regime_data_access import ensure_regime_labels, get_regime_column
                data = ensure_regime_labels(
                    data,
                    exchange=exchange,
                    symbol=symbol,
                    timeframe=timeframe,
                    data_dir=data_dir,
                )
                detected_col = get_regime_column(data)
                if detected_col and detected_col != self.comprehensive_labeling.regime_aware_labeling.regime_col:
                    self.logger.info(f"🔁 Using detected regime column '{detected_col}' instead of '{self.comprehensive_labeling.regime_aware_labeling.regime_col}'")
                    self.comprehensive_labeling.regime_aware_labeling.regime_col = detected_col
            except Exception:
                pass
            
            # Generate comprehensive labels
            data = await self.comprehensive_labeling.generate_comprehensive_labels(data, symbol, exchange, timeframe)
            if data is None:
                self.logger.error('❌ Comprehensive labeling failed')
                return False
            
            # Save labeled data
            data.to_parquet(output_path)
            self.logger.info(f'✅ Labeled data saved to {output_path}')
            
            # Generate metadata
            label_distribution = {}
            if 'label' in data.columns:
                label_distribution = data['label'].value_counts().to_dict()
            
            metadata = {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'total_samples': int(len(data)),
                'label_distribution': label_distribution,
                'created_at': pd.Timestamp.now().isoformat(),
                'labeling_config': self.config.get('labeling', {}),
                'source_fingerprint': current_fp
            }
            safe_json_dump(metadata, metadata_path, indent=2, default=str)
            
            self._log_step_timing('execute_labeling', step_start)
            
            # Log artifacts and reports
            await self._log_step5_artifacts_and_report(symbol, exchange, timeframe, data_dir, data, output_path, metadata_path)
            
            # Generate monitoring reports if available
            if self.function_monitor:
                await self._generate_and_log_monitoring_reports()
            
            return True
            
        except Exception as e:
            self.logger.exception(f'❌ Error in labeling: {e}')
            
            # Generate monitoring reports even on failure
            if self.function_monitor:
                await self._generate_and_log_monitoring_reports()
            
            return False

    @with_tracing_span(span_name='generate_function_call_report')
    async def _generate_and_log_function_call_report(self) -> None:
        """Generate and log comprehensive function call report with detailed analysis."""
        try:
            if not self.function_monitor:
                return
            
            self.logger.info('📊 Generating comprehensive monitoring reports...')
            
            # Generate function call report
            report = self.function_monitor.generate_comprehensive_report()
            self.function_monitor.log_detailed_report(report)
            
            # Save report to file
            await self._save_function_call_report(report)
            
            # Log function-to-function call relationships
            await self._log_function_call_relationships()
            
            # Analyze and log detailed completion outcomes
            outcome_analysis = await self._analyze_function_completion_outcomes()
            await self._log_detailed_completion_report(outcome_analysis)
            
            # Generate and log error summary report
            error_summary = self.error_handler.generate_error_summary_report()
            self.error_handler.log_error_summary_report(error_summary)
            
            # Generate and log performance report
            performance_report = self.performance_monitor.generate_performance_report()
            self.performance_monitor.log_performance_report(performance_report)
            
            # Generate and log validation report
            validation_report = self.validation_framework.generate_validation_report()
            self.validation_framework.log_validation_report(validation_report)
            
            self.logger.info('✅ Comprehensive function call report generated and logged successfully')
            
        except Exception as e:
            self.logger.error(f'❌ Failed to generate function call report: {e}')

    @with_tracing_span(span_name='save_function_call_report')
    async def _save_function_call_report(self, report: FunctionCallReport) -> None:
        """Save function call report to file."""
        try:
            report_dir = Path(self.config.get('DATA_DIR', 'data_cache')) / 'reports' / 'step05_function_calls'
            report_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = report_dir / f'function_call_report_{timestamp}.json'
            
            # Convert report to serializable format
            report_data = {
                'timestamp': timestamp,
                'total_calls': report.total_calls,
                'successful_calls': report.successful_calls,
                'failed_calls': report.failed_calls,
                'total_execution_time': report.total_execution_time,
                'average_execution_time': report.average_execution_time,
                'performance_summary': report.performance_summary,
                'error_summary': report.error_summary,
                'validation_summary': report.validation_summary,
                'function_call_details': [
                    {
                        'function_name': call.function_name,
                        'call_id': call.call_id,
                        'start_time': call.start_time.isoformat(),
                        'end_time': call.end_time.isoformat() if call.end_time else None,
                        'status': call.status.value,
                        'execution_time': call.execution_time,
                        'called_functions': call.called_functions,
                        'validation_results': call.validation_results,
                        'error_details': call.error_details,
                        'has_exception': call.exception is not None
                    }
                    for call in report.function_call_details
                ]
            }
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            self.logger.info(f'💾 Function call report saved to {report_file}')
            
        except Exception as e:
            self.logger.error(f'❌ Failed to save function call report: {e}')

    @with_tracing_span(span_name='log_function_call_relationships')
    async def _log_function_call_relationships(self) -> None:
        """Log detailed function-to-function call relationships."""
        try:
            self.logger.info('🔗 FUNCTION-TO-FUNCTION CALL RELATIONSHIPS')
            self.logger.info('=' * 50)
            
            # Generate validation report
            if self.validation_framework:
                validation_report = self.validation_framework.generate_validation_report()
                self.validation_framework.log_validation_report(validation_report)
            
            self.logger.info('✅ Monitoring reports generated and logged successfully')
            
        except Exception as e:
            self.logger.error(f'❌ Failed to generate monitoring reports: {e}')

    @with_tracing_span(span_name='analyze_function_completion_outcomes')
    async def _analyze_function_completion_outcomes(self) -> Dict[str, Any]:
        """Analyze detailed function completion outcomes with comprehensive metrics."""
        try:
            execution_metadata = {
                'start_time': datetime.now().isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': 0.0,
                'memory_usage_mb': 0.0,
                'cpu_usage_percent': 0.0,
                'data_quality_score': 1.0,
                'processing_efficiency': 1.0
            }
            
            artifacts_generated = [
                str(output_path),
                str(metadata_path),
                f'{exchange}_{symbol}_{timeframe}_labeling_metrics.json'
            ]
            
            metrics_calculated = {
                'labeling_success': 1.0,
                'total_samples': len(labeled_data) if labeled_data is not None else 0,
                'labeled_samples': len(labeled_data[labeled_data['label'].notna()]) if labeled_data is not None else 0,
                'label_distribution': labeled_data['label'].value_counts().to_dict() if labeled_data is not None and 'label' in labeled_data.columns else {},
                'triple_barrier_distribution': labeled_data['triple_barrier_label'].value_counts().to_dict() if labeled_data is not None and 'triple_barrier_label' in labeled_data.columns else {}
            }
            
            training_input = {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'data_dir': data_dir
            }
            
            step_data = {
                'output_path': str(output_path),
                'metadata_path': str(metadata_path),
                'data_shape': list(labeled_data.shape) if labeled_data is not None else [],
                'label_columns': list(labeled_data.columns) if labeled_data is not None else []
            }
            
            report_data = {
                'step_name': 'step05_labeling',
                'step_data': step_data,
                'training_input': training_input,
                'execution_metadata': execution_metadata,
                'artifacts_generated': artifacts_generated,
                'metrics_calculated': metrics_calculated,
                'errors_encountered': []
            }
            
            report_name = log_step_report(
                config=self.config,
                step_name='step05_labeling',
                report_data=report_data,
                report_type='labeling_report',
                additional_metadata={
                    'labeling_success': True,
                    'timeframe': timeframe,
                    'asset': symbol,
                    'lookback_period': self.config.get('lookback_days', 1095),
                    'project_version': self.config.get('project_version', '1.0.0')
                }
            )
            self.logger.info(f'✅ Logged labeling report: {report_name}')
            
            if labeled_data is not None:
                artifact_name = log_step_dataframe_with_standardized_name(
                    config=self.config,
                    step_name='step05_labeling',
                    df=labeled_data,
                    artifact_type='labeled_data',
                    additional_metadata={
                        'artifact_type': 'labeled_data',
                        'dataframe_shape': list(labeled_data.shape),
                        'label_distribution': labeled_data['label'].value_counts().to_dict() if 'label' in labeled_data.columns else {},
                        'asset': symbol,
                        'lookback_period': self.config.get('lookback_days', 1095),
                        'project_version': self.config.get('project_version', '1.0.0'),
                        'timeframe': timeframe
                    }
                )
                self.logger.info(f'✅ Logged labeled data: {artifact_name}')
            
            if metadata_path.exists():
                metadata_artifact_name = log_step_artifact_with_standardized_name(
                    config=self.config,
                    step_name='step05_labeling',
                    artifact_path=str(metadata_path),
                    artifact_type='labeling_metadata',
                    additional_metadata={
                        'metadata_type': 'labeling_metadata',
                        'timeframe': timeframe,
                        'asset': symbol,
                        'lookback_period': self.config.get('lookback_days', 1095),
                        'project_version': self.config.get('project_version', '1.0.0')
                    }
                )
                self.logger.info(f'✅ Logged labeling metadata: {metadata_artifact_name}')
            
            log_step_metrics(
                config=self.config,
                step_name='step05_labeling',
                metrics=metrics_calculated,
                additional_metadata={
                    'metrics_type': 'labeling_performance',
                    'timeframe': timeframe,
                    'asset': symbol,
                    'lookback_period': self.config.get('lookback_days', 1095),
                    'project_version': self.config.get('project_version', '1.0.0')
                }
            )
            self.logger.info('✅ Step 5 artifacts and reports logged successfully')
            
        except Exception as e:
            self.logger.error(f'❌ Failed to log step 5 artifacts and reports: {e}')

    @with_tracing_span(span_name='log_detailed_completion_report')
    async def _log_detailed_completion_report(self, outcome_analysis: Dict[str, Any]) -> None:
        """Log detailed function completion report with comprehensive analysis."""
        try:
            self.logger.info("🏷️ Starting labeling step with validation...")
            
            # Validate input data if available
            data = pipeline_state.get('dataframe') or pipeline_state.get('validated_data')
            if data is not None and isinstance(data, pd.DataFrame):
                data = self._validate_and_fix_input_data(data)
                pipeline_state['dataframe'] = data
            
            # Execute labeling
            symbol = training_input.get('symbol', 'ETHUSDT')
            exchange = training_input.get('exchange', 'BINANCE')
            timeframe = training_input.get('timeframe', '1m')
            data_dir = training_input.get('data_dir', 'data')
            
            success = await self.execute_labeling(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir
            )
            
            return {
                'success': success,
                'step_name': 'step05_labeling',
                'message': 'Labeling completed successfully' if success else 'Labeling failed'
            }
            
        except Exception as e:
            self.logger.error(f'❌ Failed to log detailed completion report: {e}')

    @with_tracing_span(span_name='log_step5_artifacts_and_report')
    async def _log_step5_artifacts_and_report(self, symbol: str, exchange: str, timeframe: str, data_dir: str, labeled_data: pd.DataFrame, output_path: Path, metadata_path: Path) -> None:
        """Log step 5 artifacts and create detailed report."""
        try:
            execution_metadata = {'start_time': datetime.now().isoformat(), 'end_time': datetime.now().isoformat(), 'duration_seconds': 0.0, 'memory_usage_mb': 0.0, 'cpu_usage_percent': 0.0, 'data_quality_score': 1.0, 'processing_efficiency': 1.0}
            artifacts_generated = [str(output_path), str(metadata_path), f'{exchange}_{symbol}_{timeframe}_labeling_metrics.json']
            metrics_calculated = {'labeling_success': 1.0, 'total_samples': len(labeled_data) if labeled_data is not None else 0, 'labeled_samples': len(labeled_data[labeled_data['label'].notna()]) if labeled_data is not None else 0, 'label_distribution': labeled_data['label'].value_counts().to_dict() if labeled_data is not None and 'label' in labeled_data.columns else {}, 'triple_barrier_distribution': labeled_data['triple_barrier_label'].value_counts().to_dict() if labeled_data is not None and 'triple_barrier_label' in labeled_data.columns else {}}
            training_input = {'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe, 'data_dir': data_dir}
            step_data = {'output_path': str(output_path), 'metadata_path': str(metadata_path), 'data_shape': list(labeled_data.shape) if labeled_data is not None else [], 'label_columns': list(labeled_data.columns) if labeled_data is not None else []}
            report_data = create_detailed_step_report(step_name='step05_labeling', step_data=step_data, training_input=training_input, execution_metadata=execution_metadata, artifacts_generated=artifacts_generated, metrics_calculated=metrics_calculated, errors_encountered=[])
            report_name = log_step_report(config=self.config, step_name='step05_labeling', report_data=report_data, report_type='labeling_report', additional_metadata={'labeling_success': True, 'timeframe': timeframe, 'asset': symbol, 'lookback_period': self.config.get('lookback_days', 1095), 'project_version': self.config.get('project_version', '1.0.0')})
            self.logger.info(f'✅ Logged labeling report: {report_name}')
            if labeled_data is not None:
                artifact_name = log_step_dataframe_with_standardized_name(config=self.config, step_name='step05_labeling', df=labeled_data, artifact_type='labeled_data', additional_metadata={'artifact_type': 'labeled_data', 'dataframe_shape': list(labeled_data.shape), 'label_distribution': labeled_data['label'].value_counts().to_dict() if 'label' in labeled_data.columns else {}, 'asset': symbol, 'lookback_period': self.config.get('lookback_days', 1095), 'project_version': self.config.get('project_version', '1.0.0'), 'timeframe': timeframe})
                self.logger.info(f'✅ Logged labeled data: {artifact_name}')
            if metadata_path.exists():
                metadata_artifact_name = log_step_artifact_with_standardized_name(config=self.config, step_name='step05_labeling', artifact_path=str(metadata_path), artifact_type='labeling_metadata', additional_metadata={'metadata_type': 'labeling_metadata', 'timeframe': timeframe, 'asset': symbol, 'lookback_period': self.config.get('lookback_days', 1095), 'project_version': self.config.get('project_version', '1.0.0')})
                self.logger.info(f'✅ Logged labeling metadata: {metadata_artifact_name}')
            log_step_metrics(config=self.config, step_name='step05_labeling', metrics=metrics_calculated, additional_metadata={'metrics_type': 'labeling_performance', 'timeframe': timeframe, 'asset': symbol, 'lookback_period': self.config.get('lookback_days', 1095), 'project_version': self.config.get('project_version', '1.0.0')})
            self.logger.info('✅ Step 5 artifacts and reports logged successfully')
        except Exception as e:
            self.logger.error(f'❌ Failed to log step 5 artifacts and reports: {e}')

    @resource_monitor()
    @handle_errors()
    @with_tracing_span(span_name='generate_comprehensive_labels')
    async def _generate_comprehensive_labels(self, data: pd.DataFrame, symbol: str, exchange: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Generate comprehensive labels combining multiple labeling strategies with regime-aware triple barrier method."
        
        self.logger.info("🔍 Validating input data for labeling...")
        
        # Validate data quality using pipeline standards
        validation_result = pipeline_standards.validate_data_quality(data, 'unified')
        
        if not validation_result.passed:
            self.logger.warning(f"⚠️ Data quality issues detected: {validation_result.quality_score:.2f}")
            for issue in validation_result.issues:
                self.logger.warning(f"   - {issue.message}")
      
        return True

    def _create_regime_labeler(self):
        """Create and configure the regime labeler."""
        try:
            from .training.steps.step06_labeling_components.regime_aware_triple_barrier_labeling import RegimeAwareTripleBarrierLabeling
            return RegimeAwareTripleBarrierLabeling(
                default_profit_take_multiplier=0.002,
                default_stop_loss_multiplier=0.001,
                default_time_barrier_minutes=self.time_barrier_minutes,
                default_max_lookahead=self.max_lookahead
            )
        except ImportError as e:
            self.logger.error(f'❌ Failed to import RegimeAwareTripleBarrierLabeling: {e}')
            return None

    @with_tracing_span(span_name='generate_labels_with_regime_labeler')
    def _generate_labels_with_regime_labeler(self, regime_labeler, data: pd.DataFrame) -> Optional[pd.Series]:
        """Generate labels using the regime labeler."""
        try:
            fixed_data = pipeline_standards.enforce_schema(fixed_data, 'unified')
            self.logger.info("✅ Applied schema enforcement")
        except Exception as e:
            self.logger.warning(f"⚠️ Schema enforcement failed: {e}")
        
        # Set datetime index if timestamp column exists
        if 'timestamp' in fixed_data.columns and not isinstance(fixed_data.index, pd.DatetimeIndex):
            try:
                fixed_data['timestamp'] = pd.to_datetime(fixed_data['timestamp'])
                fixed_data = fixed_data.set_index('timestamp')
                self.logger.info("📅 Set datetime index")
            except Exception as e:
                self.logger.warning(f"⚠️ Could not set datetime index: {e}")
        
        # Final validation
        final_validation = pipeline_standards.validate_data_quality(fixed_data, 'unified')
        self.logger.info(f"✅ Final data quality score: {final_validation.quality_score:.2f}")
        
        return fixed_data

    @handle_errors()
    @with_tracing_span(span_name='generate_regime_aware_labels')
    async def _generate_regime_aware_labels(self, data: pd.DataFrame, symbol: str, exchange: str, timeframe: str) -> Optional[pd.Series]:
        """Generate regime-aware triple barrier labels using RegimeSpecificTripleBarrierOptimizer."""
        try:
            self.logger.info('🔧 Generating regime-aware triple barrier labels...')
            
            # Validate inputs
            if not self._validate_regime_aware_inputs(data):
                return None
            
            # Create regime labeler
            regime_labeler = self._create_regime_labeler()
            if regime_labeler is None:
                return None
            
            # Generate labels with function-to-function call tracking
            # Get current call ID for tracking
            current_call_id = None
            for call_id, call_record in self.function_monitor.active_calls.items():
                if call_record.function_name == '_generate_regime_aware_labels':
                    current_call_id = call_id
                    break
            
            # Track the function-to-function call
            if current_call_id:
                self.function_monitor.record_function_to_function_call(current_call_id, '_generate_labels_with_regime_labeler')
            
            return self._generate_labels_with_regime_labeler(regime_labeler, data)
            
        except Exception as e:
            self.logger.exception(f'❌ Error in regime-aware labeling: {e}')
            return None

async def run_step(
    symbol: str,
    exchange: str,
    timeframe: str,
    data_dir: str = None,
    force_rerun: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> bool:
    """Run the labeling step with simplified architecture.

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe for data
        data_dir: Data directory (will use standardized path if None)
        force_rerun: Force rerun the step
        config: Configuration dictionary

    Returns:
        True if successful, False otherwise
    """
    if config is None:
        config = {}
    if data_dir is None:
        if pipeline_standards:
            data_dir = pipeline_standards.build_path('processed_data', exchange, symbol)
        else:
            data_dir = 'data_cache'
    
    step_config = {
        'SYMBOL': symbol,
        'EXCHANGE': exchange,
        'TIMEFRAME': timeframe,
        'DATA_DIR': data_dir,
        'labeling': {
            'enable_meta_labeling': True,
            'enable_trend_labels': True,
            'enable_volatility_labels': True,
            'composite_label_strategy': 'weighted_combination'
        },
        'vectorized_labelling_orchestrator': {
            'auto_recalculate_hmm_barriers': True,
            'hmm_barrier_regime_column': 'hmm_regime',
            'time_barrier_minutes': 30,
            'max_lookahead': 100,
            'profit_take_multiplier': 0.002,
            'stop_loss_multiplier': 0.001
        },
        **config
    }
    
    step = LabelingStep(step_config)
    await step.initialize()
    return await step.execute_labeling(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir=data_dir,
        force_rerun=force_rerun
    )


if __name__ == '__main__':
    async def test() -> None:
        success = await run_step(symbol='ETHUSDT', exchange='BINANCE', timeframe='1m', data_dir='data_cache')
        print(f'Step 5 result: {success}')
    
    asyncio.run(test())