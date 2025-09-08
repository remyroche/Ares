"""
Step03 Dependency Injection Container

This module provides a comprehensive dependency injection container for Step03 utilities,
ensuring all specified utilities are extensively used throughout the HMM clustering pipeline.
"""

import logging
from typing import Any, Dict, Optional, Type, TypeVar
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Import all utility modules
from src.utils.common_operations import (
    get_current_datetime, get_today, format_datetime, parse_datetime,
    create_empty_dataframe, safe_fillna, safe_rolling, safe_mean, safe_std,
    ensure_directory, safe_file_exists, safe_json_dump, safe_json_load,
    safe_sleep, safe_gather, create_async_task, safe_append, safe_extend,
    safe_dict_get, safe_dict_items, safe_lower, safe_upper, safe_join,
    get_logger, setup_basic_logging, safe_exception_handler, safe_float, safe_int,
    suggest_float_uniform, suggest_int_uniform, validate_dataframe, validate_numeric_range,
    optimize_dataframe_dtypes, timed_operation, format_bytes, chunked_iterable,
    parallel_map, safe_log_metric, safe_log_params, safe_log_artifact,
    safe_read_parquet, safe_to_parquet, list_parquet_files, generate_hash,
    generate_cache_key, safe_copy, safe_deepcopy, safe_glob, list_files,
    get_latest_file, validate_dataframe_schema, validate_data_quality,
    safe_resample, align_dataframes, safe_defaultdict, safe_counter, safe_deque
)

from src.utils.common_utilities import (
    safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
    calculate_data_quality_metrics, safe_merge_dataframes, safe_groupby_operation,
    safe_apply_function, create_summary_statistics, safe_drop_columns,
    safe_rename_columns, validate_timestamp_column, safe_timestamp_conversion,
    get_dataframe_info, safe_filter_dataframe, create_data_quality_report
)

from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite, validate_positive,
    validate_range, safe_kelly_calculation, safe_weighted_average, safe_percentage_change,
    validate_correlation_matrix, safe_matrix_inverse, math_safe, MathValidationError
)

from src.utils.parquet_utils import ParquetUtils, get_parquet_utils

from src.utils.serialization_utils import (
    JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer,
    save_json, load_json, save_pickle, load_pickle, save_parquet, load_parquet,
    save_data, load_data, SerializationError
)

from src.utils.data_processing_utils import (
    DataFrameValidator, DataFrameCleaner, DataFrameTransformer,
    validate_dataframe as validate_df_processing, clean_dataframe, transform_dataframe,
    get_dataframe_info as get_df_info, DataQualityLevel, DataQualityIssue, DataQualityReport
)

from src.utils.m1_gpu_utils import (
    M1GPUManager, M1PerformanceOptimizer, initialize_m1_gpu, get_m1_gpu_manager,
    m1_tensor_multiply, m1_batch_process, m1_monte_carlo_simulate, create_m1_optimized_config
)

from src.utils.m1_memory_optimizer import (
    M1MemoryOptimizer, M1DataManager, get_m1_memory_optimizer,
    create_memory_efficient_dataframe, memory_efficient_groupby
)

from src.utils.m1_cpu_optimizer import (
    M1CPUOptimizer, M1BatchProcessor, get_m1_cpu_optimizer, initialize_m1_cpu_optimizer,
    parallel_map as cpu_parallel_map, parallel_dataframe_operation, parallel_monte_carlo_simulation,
    optimized_monte_carlo_worker
)

logger = logging.getLogger(__name__)

T = TypeVar('T')

@dataclass
class Step03Config:
    """Configuration for Step03 dependency injection."""
    enable_gpu_optimization: bool = True
    enable_memory_optimization: bool = True
    enable_cpu_optimization: bool = True
    enable_math_validation: bool = True
    enable_data_validation: bool = True
    enable_serialization: bool = True
    enable_parquet_operations: bool = True
    max_memory_usage_gb: float = 8.0
    max_workers: int = 4
    cache_ttl_seconds: int = 3600
    enable_extensive_logging: bool = True

class ServiceProvider(ABC):
    """Abstract base class for service providers."""
    
    @abstractmethod
    def get_service(self, service_type: Type[T]) -> T:
        """Get a service instance."""
        pass

class Step03ServiceProvider(ServiceProvider):
    """Service provider for Step03 utilities with dependency injection."""
    
    def __init__(self, config: Optional[Step03Config] = None):
        self.config = config or Step03Config()
        self.logger = get_logger(__name__)
        self._services: Dict[Type, Any] = {}
        self._initialized = False
        
        # Initialize all services
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize all utility services."""
        try:
            self.logger.info("🔧 Initializing Step03 utility services...")
            
            # Initialize M1 optimizers
            if self.config.enable_gpu_optimization:
                self._services[M1GPUManager] = get_m1_gpu_manager()
                self._services[M1PerformanceOptimizer] = M1PerformanceOptimizer(
                    self._services[M1GPUManager]
                )
            
            if self.config.enable_memory_optimization:
                self._services[M1MemoryOptimizer] = get_m1_memory_optimizer()
                self._services[M1DataManager] = M1DataManager(
                    self._services[M1MemoryOptimizer]
                )
            
            if self.config.enable_cpu_optimization:
                self._services[M1CPUOptimizer] = get_m1_cpu_optimizer()
                self._services[M1BatchProcessor] = M1BatchProcessor(
                    self._services[M1CPUOptimizer]
                )
            
            # Initialize data processing utilities
            if self.config.enable_data_validation:
                self._services[DataFrameValidator] = DataFrameValidator()
                self._services[DataFrameCleaner] = DataFrameCleaner()
                self._services[DataFrameTransformer] = DataFrameTransformer()
            
            # Initialize file operations
            if self.config.enable_parquet_operations:
                self._services[ParquetUtils] = get_parquet_utils()
            
            # Initialize serialization utilities
            if self.config.enable_serialization:
                self._services[JSONSerializer] = JSONSerializer()
                self._services[PickleSerializer] = PickleSerializer()
                self._services[ParquetSerializer] = ParquetSerializer()
                self._services[UniversalSerializer] = UniversalSerializer()
            
            self._initialized = True
            self.logger.info("✅ Step03 utility services initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Step03 services: {e}")
            raise
    
    def get_service(self, service_type: Type[T]) -> T:
        """Get a service instance with dependency injection."""
        if not self._initialized:
            self._initialize_services()
        
        if service_type in self._services:
            return self._services[service_type]
        
        # Create service on demand for common operations
        if service_type == M1GPUManager:
            return get_m1_gpu_manager()
        elif service_type == M1MemoryOptimizer:
            return get_m1_memory_optimizer()
        elif service_type == M1CPUOptimizer:
            return get_m1_cpu_optimizer()
        elif service_type == ParquetUtils:
            return get_parquet_utils()
        elif service_type == DataFrameValidator:
            return DataFrameValidator()
        elif service_type == DataFrameCleaner:
            return DataFrameCleaner()
        elif service_type == DataFrameTransformer:
            return DataFrameTransformer()
        
        raise ValueError(f"Service type {service_type} not available")
    
    def get_common_operations(self) -> Dict[str, Any]:
        """Get all common operations utilities."""
        return {
            'datetime': {
                'get_current_datetime': get_current_datetime,
                'get_today': get_today,
                'format_datetime': format_datetime,
                'parse_datetime': parse_datetime
            },
            'dataframe': {
                'create_empty_dataframe': create_empty_dataframe,
                'safe_fillna': safe_fillna,
                'safe_rolling': safe_rolling,
                'safe_mean': safe_mean,
                'safe_std': safe_std,
                'validate_dataframe': validate_dataframe,
                'validate_dataframe_schema': validate_dataframe_schema,
                'validate_data_quality': validate_data_quality,
                'optimize_dataframe_dtypes': optimize_dataframe_dtypes,
                'safe_resample': safe_resample,
                'align_dataframes': align_dataframes
            },
            'file_operations': {
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
            'async_operations': {
                'safe_sleep': safe_sleep,
                'safe_gather': safe_gather,
                'create_async_task': create_async_task
            },
            'list_operations': {
                'safe_append': safe_append,
                'safe_extend': safe_extend,
                'safe_dict_get': safe_dict_get,
                'safe_dict_items': safe_dict_items,
                'safe_defaultdict': safe_defaultdict,
                'safe_counter': safe_counter,
                'safe_deque': safe_deque
            },
            'string_operations': {
                'safe_lower': safe_lower,
                'safe_upper': safe_upper,
                'safe_join': safe_join
            },
            'logging': {
                'get_logger': get_logger,
                'setup_basic_logging': setup_basic_logging,
                'safe_log_metric': safe_log_metric,
                'safe_log_params': safe_log_params,
                'safe_log_artifact': safe_log_artifact
            },
            'validation': {
                'safe_exception_handler': safe_exception_handler,
                'safe_float': safe_float,
                'safe_int': safe_int,
                'validate_numeric_range': validate_numeric_range
            },
            'optimization': {
                'suggest_float_uniform': suggest_float_uniform,
                'suggest_int_uniform': suggest_int_uniform,
                'timed_operation': timed_operation,
                'format_bytes': format_bytes,
                'chunked_iterable': chunked_iterable,
                'parallel_map': parallel_map
            },
            'utility': {
                'generate_hash': generate_hash,
                'generate_cache_key': generate_cache_key,
                'safe_copy': safe_copy,
                'safe_deepcopy': safe_deepcopy
            }
        }
    
    def get_common_utilities(self) -> Dict[str, Any]:
        """Get all common utilities for data processing."""
        return {
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
                'create_summary_statistics': create_summary_statistics,
                'create_data_quality_report': create_data_quality_report,
                'get_dataframe_info': get_dataframe_info
            },
            'timestamp_operations': {
                'validate_timestamp_column': validate_timestamp_column,
                'safe_timestamp_conversion': safe_timestamp_conversion
            }
        }
    
    def get_math_validation(self) -> Dict[str, Any]:
        """Get all mathematical validation utilities."""
        return {
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
            'matrix_operations': {
                'safe_matrix_inverse': safe_matrix_inverse
            },
            'decorators': {
                'math_safe': math_safe
            },
            'exceptions': {
                'MathValidationError': MathValidationError
            }
        }
    
    def get_serialization_utils(self) -> Dict[str, Any]:
        """Get all serialization utilities."""
        return {
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
    
    def get_m1_optimizers(self) -> Dict[str, Any]:
        """Get all M1 optimization utilities."""
        return {
            'gpu': {
                'M1GPUManager': self.get_service(M1GPUManager),
                'M1PerformanceOptimizer': self.get_service(M1PerformanceOptimizer),
                'm1_tensor_multiply': m1_tensor_multiply,
                'm1_batch_process': m1_batch_process,
                'm1_monte_carlo_simulate': m1_monte_carlo_simulate,
                'create_m1_optimized_config': create_m1_optimized_config
            },
            'memory': {
                'M1MemoryOptimizer': self.get_service(M1MemoryOptimizer),
                'M1DataManager': self.get_service(M1DataManager),
                'create_memory_efficient_dataframe': create_memory_efficient_dataframe,
                'memory_efficient_groupby': memory_efficient_groupby
            },
            'cpu': {
                'M1CPUOptimizer': self.get_service(M1CPUOptimizer),
                'M1BatchProcessor': self.get_service(M1BatchProcessor),
                'parallel_map': cpu_parallel_map,
                'parallel_dataframe_operation': parallel_dataframe_operation,
                'parallel_monte_carlo_simulation': parallel_monte_carlo_simulation,
                'optimized_monte_carlo_worker': optimized_monte_carlo_worker
            }
        }
    
    def get_data_processing_utils(self) -> Dict[str, Any]:
        """Get all data processing utilities."""
        return {
            'validators': {
                'DataFrameValidator': self.get_service(DataFrameValidator),
                'validate_dataframe': validate_df_processing
            },
            'cleaners': {
                'DataFrameCleaner': self.get_service(DataFrameCleaner),
                'clean_dataframe': clean_dataframe
            },
            'transformers': {
                'DataFrameTransformer': self.get_service(DataFrameTransformer),
                'transform_dataframe': transform_dataframe
            },
            'utilities': {
                'get_dataframe_info': get_df_info
            },
            'data_structures': {
                'DataQualityLevel': DataQualityLevel,
                'DataQualityIssue': DataQualityIssue,
                'DataQualityReport': DataQualityReport
            }
        }
    
    def get_parquet_utils(self) -> Dict[str, Any]:
        """Get parquet utilities."""
        return {
            'ParquetUtils': self.get_service(ParquetUtils),
            'get_parquet_utils': get_parquet_utils
        }
    
    def get_all_utilities(self) -> Dict[str, Any]:
        """Get all utilities organized by category."""
        return {
            'common_operations': self.get_common_operations(),
            'common_utilities': self.get_common_utilities(),
            'math_validation': self.get_math_validation(),
            'serialization': self.get_serialization_utils(),
            'm1_optimizers': self.get_m1_optimizers(),
            'data_processing': self.get_data_processing_utils(),
            'parquet': self.get_parquet_utils()
        }

# Global service provider instance
_step03_service_provider: Optional[Step03ServiceProvider] = None

def get_step03_service_provider(config: Optional[Step03Config] = None) -> Step03ServiceProvider:
    """Get the global Step03 service provider instance."""
    global _step03_service_provider
    if _step03_service_provider is None:
        _step03_service_provider = Step03ServiceProvider(config)
    return _step03_service_provider

def inject_step03_utilities(func):
    """Decorator to inject Step03 utilities into function parameters."""
    def wrapper(*args, **kwargs):
        # Get service provider
        service_provider = get_step03_service_provider()
        
        # Add utilities to kwargs
        kwargs['utils'] = service_provider.get_all_utilities()
        kwargs['services'] = service_provider
        
        return func(*args, **kwargs)
    return wrapper

class Step03UtilityMixin:
    """Mixin class to provide Step03 utilities to any class."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._service_provider = get_step03_service_provider()
        self._utils = self._service_provider.get_all_utilities()
    
    @property
    def utils(self) -> Dict[str, Any]:
        """Get all utilities."""
        return self._utils
    
    @property
    def services(self) -> Step03ServiceProvider:
        """Get service provider."""
        return self._service_provider
    
    def get_common_ops(self):
        """Get common operations utilities."""
        return self._utils['common_operations']
    
    def get_common_utils(self):
        """Get common utilities."""
        return self._utils['common_utilities']
    
    def get_math_validation(self):
        """Get math validation utilities."""
        return self._utils['math_validation']
    
    def get_serialization(self):
        """Get serialization utilities."""
        return self._utils['serialization']
    
    def get_m1_optimizers(self):
        """Get M1 optimization utilities."""
        return self._utils['m1_optimizers']
    
    def get_data_processing(self):
        """Get data processing utilities."""
        return self._utils['data_processing']
    
    def get_parquet_utils(self):
        """Get parquet utilities."""
        return self._utils['parquet']

# Convenience functions for easy access
def get_step03_utils() -> Dict[str, Any]:
    """Get all Step03 utilities."""
    return get_step03_service_provider().get_all_utilities()

def get_step03_services() -> Step03ServiceProvider:
    """Get Step03 service provider."""
    return get_step03_service_provider()

def initialize_step03_utilities(config: Optional[Step03Config] = None) -> Step03ServiceProvider:
    """Initialize Step03 utilities with configuration."""
    return get_step03_service_provider(config)

# Export main classes and functions
__all__ = [
    'Step03Config',
    'Step03ServiceProvider',
    'Step03UtilityMixin',
    'get_step03_service_provider',
    'get_step03_utils',
    'get_step03_services',
    'initialize_step03_utilities',
    'inject_step03_utilities'
]