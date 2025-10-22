
# src/utils/__init__.py
# This file makes the 'utils' directory a Python package.

"""
Utils Package - Lazy Loading Implementation

This module provides lazy loading to avoid circular dependencies and improve
import performance. All major utilities are loaded on-demand.
"""

import logging
from typing import Any, Dict, Optional, Callable
from functools import wraps

# Setup basic logging
logger = logging.getLogger(__name__)
# Import core utilities
try:
    from .core import (
        CommonOperations,
        get_common_operations,
        safe_json_load,
        safe_json_dump,
        safe_read_parquet,
        ensure_directory,
        safe_dataframe_operation,
        safe_get,
        safe_set,
        safe_list_get,
        safe_list_append,
        merge_dicts,
        flatten_list,
        validate_type,
        safe_convert,
        create_fallback_logger
    )
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False

# Import data utilities
try:
    from .data import (
        DataProcessor,
        DataValidator,
        DataTransformer,
        DataFrameValidator,
        DataFrameCleaner,
        DataFrameTransformer,
        validate_dataframe,
        clean_dataframe,
        transform_dataframe,
        get_dataframe_info
    )
    DATA_AVAILABLE = True
except Exception:  # pragma: no cover - fallback when optional deps fail
    DATA_AVAILABLE = False

# Import config utilities
try:
    from .config import (
        EnvironmentConfig,
        FileConfig,
        ConfigManager,
        get_env_var,
        get_env_bool,
        get_env_int,
        get_env_float,
        get_env_list,
        load_config_file,
        global_config
    )
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

# Import hardware utilities
try:
    from .hardware import (
        HardwareManager,
        optimize_memory,
        get_system_info,
        memory_optimized,
        m1_optimized,
        memory_efficient_function,
        gc_optimized_function,
        force_cleanup,
        get_memory_stats
    )
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False

# Lazy loading registry
_LAZY_IMPORTS: Dict[str, Callable] = {}
_IMPORT_CACHE: Dict[str, Any] = {}

def _register_lazy_import(name: str, import_func: Callable) -> None:
    """Register a lazy import function."""
    _LAZY_IMPORTS[name] = import_func

def _lazy_import(name: str) -> Any:
    """Lazy import with caching."""
    if name in _IMPORT_CACHE:
        return _IMPORT_CACHE[name]
    
    if name in _LAZY_IMPORTS:
        try:
            result = _LAZY_IMPORTS[name]()
            _IMPORT_CACHE[name] = result
            return result
        except ImportError as e:
            logger.warning(f"Failed to lazy import {name}: {e}")
            return None
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Register lazy imports
def _get_core_utilities():
    """Lazy loader for core utilities."""
    try:
        from .core import *
        return locals()
    except ImportError as e:
        logger.warning(f"Core utilities not available: {e}")
        return {}

def _get_data_utilities():
    """Lazy loader for data utilities."""
    try:
        from .data import *
        return locals()
    except ImportError as e:
        logger.warning(f"Data utilities not available: {e}")
        return {}

def _get_config_utilities():
    """Lazy loader for config utilities."""
    try:
        from .config import *
        return locals()
    except ImportError as e:
        logger.warning(f"Config utilities not available: {e}")
        return {}

def _get_hardware_utilities():
    """Lazy loader for hardware utilities."""
    try:
        from .hardware import *
        return locals()
    except ImportError as e:
        logger.warning(f"Hardware utilities not available: {e}")
        return {}

def _get_function_call_monitor():
    """Lazy loader for function call monitor."""
    try:
        from .function_call_monitor import (
            FunctionCallMonitor,
            FunctionCallStatus,
            ValidationLevel,
            monitor_function_calls,
            monitor_basic,
            monitor_standard,
            monitor_comprehensive,
            get_function_call_monitor,
            log_function_call_summary
        )
        return {
            'FunctionCallMonitor': FunctionCallMonitor,
            'FunctionCallStatus': FunctionCallStatus,
            'ValidationLevel': ValidationLevel,
            'monitor_function_calls': monitor_function_calls,
            'monitor_basic': monitor_basic,
            'monitor_standard': monitor_standard,
            'monitor_comprehensive': monitor_comprehensive,
            'get_function_call_monitor': get_function_call_monitor,
            'log_function_call_summary': log_function_call_summary
        }
    except ImportError as e:
        logger.warning(f"Function call monitor not available: {e}")
        return {}

def _get_error_handler():
    """Lazy loader for error handler."""
    try:
        from .enhanced_error_handler import (
            EnhancedErrorHandler,
            ErrorSeverity,
            ErrorCategory,
            ErrorContext,
            ErrorRecord,
            handle_errors_with_tracking,
            handle_errors_basic,
            handle_errors_strict,
            get_error_handler,
            log_error_summary
        )
        return {
            'EnhancedErrorHandler': EnhancedErrorHandler,
            'ErrorSeverity': ErrorSeverity,
            'ErrorCategory': ErrorCategory,
            'ErrorContext': ErrorContext,
            'ErrorRecord': ErrorRecord,
            'handle_errors_with_tracking': handle_errors_with_tracking,
            'handle_errors_basic': handle_errors_basic,
            'handle_errors_strict': handle_errors_strict,
            'get_error_handler': get_error_handler,
            'log_error_summary': log_error_summary
        }
    except ImportError as e:
        logger.warning(f"Error handler not available: {e}")
        return {}

def _get_performance_utils():
    """Lazy loader for performance utilities."""
    try:
        from .performance_utils import (
            PerformanceMonitor,
            MemoryProfiler,
            SystemMonitor,
            global_monitor,
            timer,
            profile_function,
            time_function,
            benchmark_function,
            get_system_info
        )
        return {
            'PerformanceMonitor': PerformanceMonitor,
            'MemoryProfiler': MemoryProfiler,
            'SystemMonitor': SystemMonitor,
            'global_monitor': global_monitor,
            'timer': timer,
            'profile_function': profile_function,
            'time_function': time_function,
            'benchmark_function': benchmark_function,
            'get_system_info': get_system_info
        }
    except ImportError as e:
        logger.warning(f"Performance utilities not available: {e}")
        return {}

def _get_data_processing_utils():
    """Lazy loader for data processing utilities."""
    try:
        from .data_processing_utils import (
            DataFrameValidator,
            DataFrameCleaner,
            DataFrameTransformer,
            validate_dataframe,
            clean_dataframe,
            transform_dataframe,
            get_dataframe_info
        )
        return {
            'DataFrameValidator': DataFrameValidator,
            'DataFrameCleaner': DataFrameCleaner,
            'DataFrameTransformer': DataFrameTransformer,
            'validate_dataframe': validate_dataframe,
            'clean_dataframe': clean_dataframe,
            'transform_dataframe': transform_dataframe,
            'get_dataframe_info': get_dataframe_info
        }
    except ImportError as e:
        logger.warning(f"Data processing utilities not available: {e}")
        return {}

# Register all lazy imports
_register_lazy_import('core', _get_core_utilities)
_register_lazy_import('data', _get_data_utilities)
_register_lazy_import('config', _get_config_utilities)
_register_lazy_import('hardware', _get_hardware_utilities)
_register_lazy_import('function_call_monitor', _get_function_call_monitor)
_register_lazy_import('error_handler', _get_error_handler)
_register_lazy_import('performance_utils', _get_performance_utils)
_register_lazy_import('data_processing_utils', _get_data_processing_utils)

# Module-level lazy loading
def __getattr__(name: str) -> Any:
    """Lazy loading for module attributes."""
    return _lazy_import(name)

# Build __all__ list dynamically based on available modules
__all__ = []

# Add core utilities if available
if CORE_AVAILABLE:
    __all__.extend([
        'CommonOperations', 'get_common_operations',
        'safe_json_load', 'safe_json_dump', 'safe_read_parquet', 'ensure_directory',
        'safe_dataframe_operation', 'safe_get', 'safe_set', 'safe_list_get', 'safe_list_append',
        'merge_dicts', 'flatten_list', 'validate_type', 'safe_convert', 'create_fallback_logger'
    ])

# Note: Data, config, and hardware utilities need to be imported explicitly
# from their specific modules due to file consolidation

# Add function call monitoring if available
if FUNCTION_CALL_MONITOR_AVAILABLE:
    __all__.extend([
        'FunctionCallMonitor',
        'FunctionCallStatus',
        'ValidationLevel',
        'monitor_function_calls',
        'monitor_basic',
        'monitor_standard',
        'monitor_comprehensive',
        'get_function_call_monitor',
        'log_function_call_summary'
    ])

# Add function validation if available
if FUNCTION_VALIDATION_AVAILABLE:
    __all__.extend([
        'FunctionValidator',
        'ValidationSeverity',
        'ValidationCategory',
        'ValidationIssue',
        'ValidationResult',
        'validate_function_entry',
        'validate_function_output',
        'get_function_validator'
    ])

# Add error handler if available
if ERROR_HANDLER_AVAILABLE:
    __all__.extend([
        'EnhancedErrorHandler',
        'ErrorSeverity',
        'ErrorCategory',
        'ErrorContext',
        'ErrorRecord',
        'handle_errors_with_tracking',
        'handle_errors_basic',
        'handle_errors_strict',
        'get_error_handler',
        'log_error_summary'
    ])

# Add serialization if available
if SERIALIZATION_AVAILABLE:
    __all__.extend([
        'JSONSerializer',
        'PickleSerializer',
        'ParquetSerializer',
        'UniversalSerializer',
        'save_json',
        'load_json',
        'save_pickle',
        'load_pickle',
        'save_parquet',
        'load_parquet',
        'save_data',
        'load_data'
    ])

# Add config if available
if CONFIG_AVAILABLE:
    __all__.extend([
        'EnvironmentConfig',
        'FileConfig',
        'ConfigManager',
        'get_env_var',
        'get_env_bool',
        'get_env_int',
        'get_env_float',
        'get_env_list',
        'load_config_file',
        'global_config'
    ])

# Add performance if available
if PERFORMANCE_AVAILABLE:
    __all__.extend([
        'PerformanceMonitor',
        'MemoryProfiler',
        'SystemMonitor',
        'global_monitor',
        'timer',
        'profile_function',
        'time_function',
        'benchmark_function',
        'get_system_info'
    ])

# Add data processing if available
if DATA_PROCESSING_AVAILABLE:
    __all__.extend([
        'DataFrameValidator',
        'DataFrameCleaner',
        'DataFrameTransformer',
        'validate_dataframe',
        'clean_dataframe',
        'transform_dataframe',
        'get_dataframe_info'
    ])

# Add monitoring if available
if MONITORING_AVAILABLE:
    __all__.extend([
        'UnifiedPerformanceMonitor',
        'FunctionTracker',
        'LoggingPatterns',
        'monitoring_global_monitor',
        'global_tracker',
        'track_function',
        'monitor_function_calls',
        'function_tracker',
        'logging_patterns'
    ])

# Add parameter loader if available
if PARAMETER_LOADER_AVAILABLE:
    __all__.extend([
        'ParameterLoader',
        'SRParameterLoader',
        'initialize_sr_parameters',
        'load_sr_parameters',
        'load_parameters',
        'global_parameter_loader'
    ])
