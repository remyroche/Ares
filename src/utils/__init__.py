
# src/utils/__init__.py
# This file makes the 'utils' directory a Python package.

# Import monitoring systems
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
    FUNCTION_CALL_MONITOR_AVAILABLE = True
except ImportError:
    FUNCTION_CALL_MONITOR_AVAILABLE = False

# Make all imports optional to handle missing dependencies gracefully
try:
    from .function_validation_framework import (
        FunctionValidator,
        ValidationSeverity,
        ValidationCategory,
        ValidationIssue,
        ValidationResult,
        validate_function_entry,
        validate_function_output,
        get_function_validator
    )
    FUNCTION_VALIDATION_AVAILABLE = True
except ImportError:
    FUNCTION_VALIDATION_AVAILABLE = False

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
    ERROR_HANDLER_AVAILABLE = True
except ImportError:
    ERROR_HANDLER_AVAILABLE = False

# Import new utility modules
try:
    from .serialization_utils import (
        JSONSerializer,
        PickleSerializer,
        ParquetSerializer,
        UniversalSerializer,
        save_json,
        load_json,
        save_pickle,
        load_pickle,
        save_parquet,
        load_parquet,
        save_data,
        load_data
    )
    SERIALIZATION_AVAILABLE = True
except ImportError:
    SERIALIZATION_AVAILABLE = False

try:
    from .config_utils import (
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
    PERFORMANCE_AVAILABLE = True
except ImportError:
    PERFORMANCE_AVAILABLE = False

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
    DATA_PROCESSING_AVAILABLE = True
except ImportError:
    DATA_PROCESSING_AVAILABLE = False

try:
    from .monitoring_utils import (
        UnifiedPerformanceMonitor,
        FunctionTracker,
        LoggingPatterns,
        global_monitor as monitoring_global_monitor,
        global_tracker,
        track_function,
        monitor_function_calls,
        function_tracker,
        logging_patterns
    )
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

try:
    from .parameter_loader import (
        ParameterLoader,
        SRParameterLoader,
        initialize_sr_parameters,
        load_sr_parameters,
        load_parameters,
        global_parameter_loader
    )
    PARAMETER_LOADER_AVAILABLE = True
except ImportError:
    PARAMETER_LOADER_AVAILABLE = False

# Build __all__ list dynamically based on available modules
__all__ = []

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
