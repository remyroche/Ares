"""
Step04 Dependency Injection Container

This module provides a comprehensive dependency injection container for Step04 utilities,
ensuring extensive use of all utility modules with proper dependency management.
"""

import logging
from typing import Any, Dict, Optional, Type, TypeVar, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

T = TypeVar('T')

@dataclass
class UtilityConfig:
    """Configuration for utility initialization."""
    # Common Operations
    enable_common_operations: bool = True
    common_operations_config: Dict[str, Any] = field(default_factory=dict)
    
    # Common Utilities
    enable_common_utilities: bool = True
    common_utilities_config: Dict[str, Any] = field(default_factory=dict)
    
    # Math Validation
    enable_math_validation: bool = True
    math_validation_config: Dict[str, Any] = field(default_factory=dict)
    
    # Parquet Utils
    enable_parquet_utils: bool = True
    parquet_utils_config: Dict[str, Any] = field(default_factory=dict)
    
    # Serialization Utils
    enable_serialization_utils: bool = True
    serialization_utils_config: Dict[str, Any] = field(default_factory=dict)
    
    # Data Processing Utils
    enable_data_processing_utils: bool = True
    data_processing_utils_config: Dict[str, Any] = field(default_factory=dict)
    
    # M1 GPU Utils
    enable_m1_gpu_utils: bool = True
    m1_gpu_utils_config: Dict[str, Any] = field(default_factory=dict)
    
    # M1 Memory Optimizer
    enable_m1_memory_optimizer: bool = True
    m1_memory_optimizer_config: Dict[str, Any] = field(default_factory=dict)
    
    # M1 CPU Optimizer
    enable_m1_cpu_optimizer: bool = True
    m1_cpu_optimizer_config: Dict[str, Any] = field(default_factory=dict)

class UtilityProvider(ABC):
    """Abstract base class for utility providers."""
    
    @abstractmethod
    def get_utility(self, utility_type: str) -> Any:
        """Get a utility instance."""
        pass
    
    @abstractmethod
    def is_available(self, utility_type: str) -> bool:
        """Check if utility is available."""
        pass

class Step04UtilityProvider(UtilityProvider):
    """Step04-specific utility provider with extensive utility integration."""
    
    def __init__(self, config: UtilityConfig):
        self.config = config
        self.logger = logger.getChild('Step04UtilityProvider')
        self._utilities: Dict[str, Any] = {}
        self._initialized = False
        
    def initialize(self) -> None:
        """Initialize all utilities based on configuration."""
        if self._initialized:
            return
            
        self.logger.info('🔧 Initializing Step04 utility providers...')
        
        # Initialize Common Operations
        if self.config.enable_common_operations:
            self._init_common_operations()
            
        # Initialize Common Utilities
        if self.config.enable_common_utilities:
            self._init_common_utilities()
            
        # Initialize Math Validation
        if self.config.enable_math_validation:
            self._init_math_validation()
            
        # Initialize Parquet Utils
        if self.config.enable_parquet_utils:
            self._init_parquet_utils()
            
        # Initialize Serialization Utils
        if self.config.enable_serialization_utils:
            self._init_serialization_utils()
            
        # Initialize Data Processing Utils
        if self.config.enable_data_processing_utils:
            self._init_data_processing_utils()
            
        # Initialize M1 GPU Utils
        if self.config.enable_m1_gpu_utils:
            self._init_m1_gpu_utils()
            
        # Initialize M1 Memory Optimizer
        if self.config.enable_m1_memory_optimizer:
            self._init_m1_memory_optimizer()
            
        # Initialize M1 CPU Optimizer
        if self.config.enable_m1_cpu_optimizer:
            self._init_m1_cpu_optimizer()
            
        self._initialized = True
        self.logger.info(f'✅ Step04 utility providers initialized: {len(self._utilities)} utilities available')
    
    def _init_common_operations(self) -> None:
        """Initialize common operations utilities."""
        try:
            from src.utils.common_operations import (
                get_current_datetime, get_today, format_datetime, parse_datetime,
                create_empty_dataframe, safe_fillna, safe_rolling, safe_mean, safe_std,
                ensure_directory, safe_file_exists, safe_json_dump, safe_json_load,
                safe_sleep, safe_gather, create_async_task, safe_append, safe_extend,
                safe_dict_get, safe_dict_items, safe_lower, safe_upper, safe_join,
                get_logger, setup_basic_logging, create_argument_parser, add_common_arguments,
                safe_exception_handler, safe_float, safe_int, suggest_float_uniform,
                suggest_int_uniform, validate_dataframe, validate_numeric_range,
                optimize_dataframe_dtypes, timed_operation, format_bytes, chunked_iterable,
                parallel_map, safe_log_metric, safe_log_params, safe_log_artifact,
                standardize_price_action_probabilities, safe_copy, safe_deepcopy,
                safe_resample, align_dataframes, safe_defaultdict, safe_counter,
                safe_deque, safe_glob, list_files, get_latest_file, safe_read_parquet,
                safe_to_parquet, list_parquet_files, generate_hash, generate_cache_key,
                validate_dataframe_schema, validate_data_quality
            )
            
            self._utilities['common_operations'] = {
                'datetime_ops': {
                    'get_current_datetime': get_current_datetime,
                    'get_today': get_today,
                    'format_datetime': format_datetime,
                    'parse_datetime': parse_datetime
                },
                'dataframe_ops': {
                    'create_empty_dataframe': create_empty_dataframe,
                    'safe_fillna': safe_fillna,
                    'safe_rolling': safe_rolling,
                    'safe_mean': safe_mean,
                    'safe_std': safe_std,
                    'optimize_dataframe_dtypes': optimize_dataframe_dtypes,
                    'validate_dataframe_schema': validate_dataframe_schema,
                    'validate_data_quality': validate_data_quality
                },
                'file_ops': {
                    'ensure_directory': ensure_directory,
                    'safe_file_exists': safe_file_exists,
                    'safe_json_dump': safe_json_dump,
                    'safe_json_load': safe_json_load,
                    'safe_read_parquet': safe_read_parquet,
                    'safe_to_parquet': safe_to_parquet,
                    'list_parquet_files': list_parquet_files,
                    'safe_glob': safe_glob,
                    'list_files': list_files,
                    'get_latest_file': get_latest_file
                },
                'async_ops': {
                    'safe_sleep': safe_sleep,
                    'safe_gather': safe_gather,
                    'create_async_task': create_async_task
                },
                'list_ops': {
                    'safe_append': safe_append,
                    'safe_extend': safe_extend,
                    'safe_dict_get': safe_dict_get,
                    'safe_dict_items': safe_dict_items,
                    'safe_lower': safe_lower,
                    'safe_upper': safe_upper,
                    'safe_join': safe_join
                },
                'logging_ops': {
                    'get_logger': get_logger,
                    'setup_basic_logging': setup_basic_logging,
                    'safe_log_metric': safe_log_metric,
                    'safe_log_params': safe_log_params,
                    'safe_log_artifact': safe_log_artifact
                },
                'utility_ops': {
                    'create_argument_parser': create_argument_parser,
                    'add_common_arguments': add_common_arguments,
                    'safe_exception_handler': safe_exception_handler,
                    'safe_float': safe_float,
                    'safe_int': safe_int,
                    'suggest_float_uniform': suggest_float_uniform,
                    'suggest_int_uniform': suggest_int_uniform,
                    'validate_dataframe': validate_dataframe,
                    'validate_numeric_range': validate_numeric_range,
                    'timed_operation': timed_operation,
                    'format_bytes': format_bytes,
                    'chunked_iterable': chunked_iterable,
                    'parallel_map': parallel_map,
                    'standardize_price_action_probabilities': standardize_price_action_probabilities,
                    'safe_copy': safe_copy,
                    'safe_deepcopy': safe_deepcopy,
                    'safe_resample': safe_resample,
                    'align_dataframes': align_dataframes,
                    'safe_defaultdict': safe_defaultdict,
                    'safe_counter': safe_counter,
                    'safe_deque': safe_deque,
                    'generate_hash': generate_hash,
                    'generate_cache_key': generate_cache_key
                }
            }
            self.logger.info('✅ Common operations utilities initialized')
        except ImportError as e:
            self.logger.warning(f'⚠️ Common operations utilities not available: {e}')
    
    def _init_common_utilities(self) -> None:
        """Initialize common utilities."""
        try:
            from src.utils.common_utilities import (
                safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
                calculate_data_quality_metrics, safe_merge_dataframes, safe_groupby_operation,
                safe_apply_function, create_summary_statistics, safe_drop_columns,
                safe_rename_columns, validate_timestamp_column, safe_timestamp_conversion,
                get_dataframe_info, safe_filter_dataframe, create_data_quality_report
            )
            
            self._utilities['common_utilities'] = {
                'dataframe_operations': {
                    'safe_dataframe_operation': safe_dataframe_operation,
                    'validate_dataframe_columns': validate_dataframe_columns,
                    'safe_convert_dtypes': safe_convert_dtypes,
                    'safe_merge_dataframes': safe_merge_dataframes,
                    'safe_groupby_operation': safe_groupby_operation,
                    'safe_apply_function': safe_apply_function,
                    'safe_drop_columns': safe_drop_columns,
                    'safe_rename_columns': safe_rename_columns,
                    'safe_filter_dataframe': safe_filter_dataframe
                },
                'data_quality': {
                    'calculate_data_quality_metrics': calculate_data_quality_metrics,
                    'create_data_quality_report': create_data_quality_report,
                    'validate_timestamp_column': validate_timestamp_column,
                    'safe_timestamp_conversion': safe_timestamp_conversion
                },
                'data_analysis': {
                    'create_summary_statistics': create_summary_statistics,
                    'get_dataframe_info': get_dataframe_info
                }
            }
            self.logger.info('✅ Common utilities initialized')
        except ImportError as e:
            self.logger.warning(f'⚠️ Common utilities not available: {e}')
    
    def _init_math_validation(self) -> None:
        """Initialize math validation utilities."""
        try:
            from src.utils.math_validation import (
                safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
                validate_positive, validate_range, safe_kelly_calculation,
                safe_weighted_average, safe_percentage_change, validate_correlation_matrix,
                safe_matrix_inverse, math_safe, MathValidationError
            )
            
            self._utilities['math_validation'] = {
                'basic_math': {
                    'safe_divide': safe_divide,
                    'safe_log': safe_log,
                    'safe_sqrt': safe_sqrt,
                    'safe_power': safe_power
                },
                'validation': {
                    'validate_finite': validate_finite,
                    'validate_positive': validate_positive,
                    'validate_range': validate_range,
                    'validate_correlation_matrix': validate_correlation_matrix
                },
                'financial_math': {
                    'safe_kelly_calculation': safe_kelly_calculation,
                    'safe_weighted_average': safe_weighted_average,
                    'safe_percentage_change': safe_percentage_change
                },
                'matrix_ops': {
                    'safe_matrix_inverse': safe_matrix_inverse
                },
                'decorators': {
                    'math_safe': math_safe
                },
                'exceptions': {
                    'MathValidationError': MathValidationError
                }
            }
            self.logger.info('✅ Math validation utilities initialized')
        except ImportError as e:
            self.logger.warning(f'⚠️ Math validation utilities not available: {e}')
    
    def _init_parquet_utils(self) -> None:
        """Initialize parquet utilities."""
        try:
            from src.utils.parquet_utils import get_parquet_utils, ParquetUtils
            
            parquet_utils = get_parquet_utils()
            self._utilities['parquet_utils'] = {
                'parquet_utils': parquet_utils,
                'validate_parquet_file': parquet_utils.validate_parquet_file,
                'safe_read_parquet': parquet_utils.safe_read_parquet,
                'repair_parquet_file': parquet_utils.repair_parquet_file
            }
            self.logger.info('✅ Parquet utilities initialized')
        except ImportError as e:
            self.logger.warning(f'⚠️ Parquet utilities not available: {e}')
    
    def _init_serialization_utils(self) -> None:
        """Initialize serialization utilities."""
        try:
            from src.utils.serialization_utils import (
                JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer,
                save_json, load_json, save_pickle, load_pickle, save_parquet, load_parquet,
                save_data, load_data, SerializationError
            )
            
            self._utilities['serialization_utils'] = {
                'serializers': {
                    'JSONSerializer': JSONSerializer,
                    'PickleSerializer': PickleSerializer,
                    'ParquetSerializer': ParquetSerializer,
                    'UniversalSerializer': UniversalSerializer
                },
                'convenience_functions': {
                    'save_json': save_json,
                    'load_json': load_json,
                    'save_pickle': save_pickle,
                    'load_pickle': load_pickle,
                    'save_parquet': save_parquet,
                    'load_parquet': load_parquet,
                    'save_data': save_data,
                    'load_data': load_data
                },
                'exceptions': {
                    'SerializationError': SerializationError
                }
            }
            self.logger.info('✅ Serialization utilities initialized')
        except ImportError as e:
            self.logger.warning(f'⚠️ Serialization utilities not available: {e}')
    
    def _init_data_processing_utils(self) -> None:
        """Initialize data processing utilities."""
        try:
            from src.utils.data_processing_utils import (
                DataQualityLevel, DataQualityIssue, DataQualityReport,
                DataFrameValidator, DataFrameCleaner, DataFrameTransformer,
                validate_dataframe, clean_dataframe, transform_dataframe, get_dataframe_info
            )
            
            self._utilities['data_processing_utils'] = {
                'data_structures': {
                    'DataQualityLevel': DataQualityLevel,
                    'DataQualityIssue': DataQualityIssue,
                    'DataQualityReport': DataQualityReport
                },
                'validators': {
                    'DataFrameValidator': DataFrameValidator,
                    'validate_dataframe': validate_dataframe
                },
                'processors': {
                    'DataFrameCleaner': DataFrameCleaner,
                    'DataFrameTransformer': DataFrameTransformer,
                    'clean_dataframe': clean_dataframe,
                    'transform_dataframe': transform_dataframe
                },
                'utilities': {
                    'get_dataframe_info': get_dataframe_info
                }
            }
            self.logger.info('✅ Data processing utilities initialized')
        except ImportError as e:
            self.logger.warning(f'⚠️ Data processing utilities not available: {e}')
    
    def _init_m1_gpu_utils(self) -> None:
        """Initialize M1 GPU utilities."""
        try:
            from src.utils.m1_gpu_utils import (
                get_m1_gpu_manager, M1GPUManager, M1PerformanceOptimizer,
                create_m1_optimized_config, initialize_m1_gpu, m1_tensor_multiply,
                m1_batch_process, m1_monte_carlo_simulate
            )
            
            gpu_manager = get_m1_gpu_manager()
            self._utilities['m1_gpu_utils'] = {
                'gpu_manager': gpu_manager,
                'M1GPUManager': M1GPUManager,
                'M1PerformanceOptimizer': M1PerformanceOptimizer,
                'create_m1_optimized_config': create_m1_optimized_config,
                'initialize_m1_gpu': initialize_m1_gpu,
                'm1_tensor_multiply': m1_tensor_multiply,
                'm1_batch_process': m1_batch_process,
                'm1_monte_carlo_simulate': m1_monte_carlo_simulate
            }
            self.logger.info('✅ M1 GPU utilities initialized')
        except ImportError as e:
            self.logger.warning(f'⚠️ M1 GPU utilities not available: {e}')
    
    def _init_m1_memory_optimizer(self) -> None:
        """Initialize M1 memory optimizer."""
        try:
            from src.utils.m1_memory_optimizer import (
                get_m1_memory_optimizer, M1MemoryOptimizer, M1DataManager,
                create_memory_efficient_dataframe, memory_efficient_groupby
            )
            
            memory_optimizer = get_m1_memory_optimizer()
            self._utilities['m1_memory_optimizer'] = {
                'memory_optimizer': memory_optimizer,
                'M1MemoryOptimizer': M1MemoryOptimizer,
                'M1DataManager': M1DataManager,
                'create_memory_efficient_dataframe': create_memory_efficient_dataframe,
                'memory_efficient_groupby': memory_efficient_groupby
            }
            self.logger.info('✅ M1 memory optimizer initialized')
        except ImportError as e:
            self.logger.warning(f'⚠️ M1 memory optimizer not available: {e}')
    
    def _init_m1_cpu_optimizer(self) -> None:
        """Initialize M1 CPU optimizer."""
        try:
            from src.utils.m1_cpu_optimizer import (
                get_m1_cpu_optimizer, M1CPUOptimizer, M1BatchProcessor,
                initialize_m1_cpu_optimizer, parallel_map, parallel_dataframe_operation,
                parallel_monte_carlo_simulation, optimized_monte_carlo_worker
            )
            
            cpu_optimizer = get_m1_cpu_optimizer()
            self._utilities['m1_cpu_optimizer'] = {
                'cpu_optimizer': cpu_optimizer,
                'M1CPUOptimizer': M1CPUOptimizer,
                'M1BatchProcessor': M1BatchProcessor,
                'initialize_m1_cpu_optimizer': initialize_m1_cpu_optimizer,
                'parallel_map': parallel_map,
                'parallel_dataframe_operation': parallel_dataframe_operation,
                'parallel_monte_carlo_simulation': parallel_monte_carlo_simulation,
                'optimized_monte_carlo_worker': optimized_monte_carlo_worker
            }
            self.logger.info('✅ M1 CPU optimizer initialized')
        except ImportError as e:
            self.logger.warning(f'⚠️ M1 CPU optimizer not available: {e}')
    
    def get_utility(self, utility_type: str) -> Any:
        """Get a utility instance."""
        if not self._initialized:
            self.initialize()
        return self._utilities.get(utility_type)
    
    def is_available(self, utility_type: str) -> bool:
        """Check if utility is available."""
        return utility_type in self._utilities
    
    def get_utility_function(self, utility_type: str, function_name: str) -> Optional[Callable]:
        """Get a specific utility function."""
        utility = self.get_utility(utility_type)
        if utility and isinstance(utility, dict):
            # First try direct lookup
            if function_name in utility:
                return utility[function_name]

            # If not found directly, search through nested dictionaries
            for category, functions in utility.items():
                if isinstance(functions, dict) and function_name in functions:
                    return functions[function_name]

        return None
    
    def get_all_utilities(self) -> Dict[str, Any]:
        """Get all available utilities."""
        if not self._initialized:
            self.initialize()
        return self._utilities.copy()
    
    def get_utility_summary(self) -> Dict[str, Any]:
        """Get a summary of all available utilities."""
        if not self._initialized:
            self.initialize()
        
        summary = {}
        for utility_type, utility_data in self._utilities.items():
            if isinstance(utility_data, dict):
                summary[utility_type] = {
                    'available': True,
                    'functions': list(utility_data.keys()),
                    'function_count': len(utility_data)
                }
            else:
                summary[utility_type] = {
                    'available': True,
                    'type': type(utility_data).__name__
                }
        
        return summary

class Step04DependencyContainer:
    """Main dependency injection container for Step04."""
    
    def __init__(self, config: Optional[UtilityConfig] = None):
        self.config = config or UtilityConfig()
        self.logger = logger.getChild('Step04DependencyContainer')
        self._provider: Optional[Step04UtilityProvider] = None
        
    def get_provider(self) -> Step04UtilityProvider:
        """Get the utility provider."""
        if self._provider is None:
            self._provider = Step04UtilityProvider(self.config)
            self._provider.initialize()
        return self._provider
    
    def get_utility(self, utility_type: str) -> Any:
        """Get a utility instance."""
        return self.get_provider().get_utility(utility_type)
    
    def get_utility_function(self, utility_type: str, function_name: str) -> Optional[Callable]:
        """Get a specific utility function."""
        return self.get_provider().get_utility_function(utility_type, function_name)
    
    def is_utility_available(self, utility_type: str) -> bool:
        """Check if utility is available."""
        return self.get_provider().is_available(utility_type)
    
    def get_all_utilities(self) -> Dict[str, Any]:
        """Get all available utilities."""
        return self.get_provider().get_all_utilities()
    
    def get_utility_summary(self) -> Dict[str, Any]:
        """Get a summary of all available utilities."""
        return self.get_provider().get_utility_summary()
    
    def create_utility_context(self) -> 'Step04UtilityContext':
        """Create a utility context for easy access."""
        return Step04UtilityContext(self)

class Step04UtilityContext:
    """Context manager for easy utility access in Step04."""
    
    def __init__(self, container: Step04DependencyContainer):
        self.container = container
        self.logger = logger.getChild('Step04UtilityContext')
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    # Common Operations
    @property
    def common_ops(self) -> Dict[str, Any]:
        """Get common operations utilities."""
        return self.container.get_utility('common_operations') or {}
    
    # Common Utilities
    @property
    def common_utils(self) -> Dict[str, Any]:
        """Get common utilities."""
        return self.container.get_utility('common_utilities') or {}
    
    # Math Validation
    @property
    def math_validation(self) -> Dict[str, Any]:
        """Get math validation utilities."""
        return self.container.get_utility('math_validation') or {}
    
    # Parquet Utils
    @property
    def parquet_utils(self) -> Dict[str, Any]:
        """Get parquet utilities."""
        return self.container.get_utility('parquet_utils') or {}
    
    # Serialization Utils
    @property
    def serialization_utils(self) -> Dict[str, Any]:
        """Get serialization utilities."""
        return self.container.get_utility('serialization_utils') or {}
    
    # Data Processing Utils
    @property
    def data_processing_utils(self) -> Dict[str, Any]:
        """Get data processing utilities."""
        return self.container.get_utility('data_processing_utils') or {}
    
    # M1 GPU Utils
    @property
    def m1_gpu_utils(self) -> Dict[str, Any]:
        """Get M1 GPU utilities."""
        return self.container.get_utility('m1_gpu_utils') or {}
    
    # M1 Memory Optimizer
    @property
    def m1_memory_optimizer(self) -> Dict[str, Any]:
        """Get M1 memory optimizer."""
        return self.container.get_utility('m1_memory_optimizer') or {}
    
    # M1 CPU Optimizer
    @property
    def m1_cpu_optimizer(self) -> Dict[str, Any]:
        """Get M1 CPU optimizer."""
        return self.container.get_utility('m1_cpu_optimizer') or {}
    
    def get_function(self, utility_type: str, function_name: str) -> Optional[Callable]:
        """Get a specific utility function."""
        return self.container.get_utility_function(utility_type, function_name)
    
    def log_utility_usage(self, utility_type: str, function_name: str, success: bool = True):
        """Log utility usage for monitoring."""
        status = "✅" if success else "❌"
        self.logger.debug(f'{status} Used {utility_type}.{function_name}')

# Global container instance
_global_container: Optional[Step04DependencyContainer] = None

def get_step04_container(config: Optional[UtilityConfig] = None) -> Step04DependencyContainer:
    """Get the global Step04 dependency container."""
    global _global_container
    if _global_container is None:
        _global_container = Step04DependencyContainer(config)
    return _global_container

def get_step04_utilities() -> Step04UtilityContext:
    """Get Step04 utilities context for easy access."""
    container = get_step04_container()
    return container.create_utility_context()

def create_step04_config(**kwargs) -> UtilityConfig:
    """Create a Step04 utility configuration."""
    return UtilityConfig(**kwargs)

# Convenience functions for direct utility access
def get_common_ops():
    """Get common operations utilities."""
    return get_step04_utilities().common_ops

def get_common_utils():
    """Get common utilities."""
    return get_step04_utilities().common_utils

def get_math_validation():
    """Get math validation utilities."""
    return get_step04_utilities().math_validation

def get_parquet_utils():
    """Get parquet utilities."""
    return get_step04_utilities().parquet_utils

def get_serialization_utils():
    """Get serialization utilities."""
    return get_step04_utilities().serialization_utils

def get_data_processing_utils():
    """Get data processing utilities."""
    return get_step04_utilities().data_processing_utils

def get_m1_gpu_utils():
    """Get M1 GPU utilities."""
    return get_step04_utilities().m1_gpu_utils

def get_m1_memory_optimizer():
    """Get M1 memory optimizer."""
    return get_step04_utilities().m1_memory_optimizer

def get_m1_cpu_optimizer():
    """Get M1 CPU optimizer."""
    return get_step04_utilities().m1_cpu_optimizer

if __name__ == '__main__':
    # Test the dependency injection container
    config = create_step04_config()
    container = get_step04_container(config)
    
    with get_step04_utilities() as utils:
        summary = container.get_utility_summary()
        print("Step04 Utility Summary:")
        for utility_type, info in summary.items():
            print(f"  {utility_type}: {info}")