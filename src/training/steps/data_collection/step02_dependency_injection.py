"""
Dependency Injection Container for Step 2: Data Reading

This module provides a comprehensive dependency injection container that manages
all utility dependencies for step02, ensuring proper initialization and lifecycle
management of all utility modules.
"""

import logging
from typing import Any, Dict, Optional, Type, TypeVar, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import threading
import time

T = TypeVar('T')

@dataclass
class ServiceDefinition:
    """Definition of a service in the DI container."""
    service_type: Type
    implementation: Type
    singleton: bool = True
    dependencies: list[str] = field(default_factory=list)
    factory: Optional[Callable] = None
    instance: Optional[Any] = None
    initialized: bool = False

class DependencyInjectionContainer:
    """Comprehensive dependency injection container for step02 utilities."""
    
    def __init__(self):
        self._services: Dict[str, ServiceDefinition] = {}
        self._instances: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self.logger = logging.getLogger(f"{__name__}.DependencyInjectionContainer")
        
        # Register all utility services
        self._register_utility_services()
    
    def _register_utility_services(self):
        """Register all utility services with the container."""
        
        # Common Operations Services
        self.register_singleton(
            'common_operations',
            object,  # Will be replaced with actual module
            factory=lambda: self._import_common_operations()
        )
        
        # Common Utilities Services
        self.register_singleton(
            'common_utilities',
            object,
            factory=lambda: self._import_common_utilities()
        )
        
        # Math Validation Services
        self.register_singleton(
            'math_validation',
            object,
            factory=lambda: self._import_math_validation()
        )
        
        # Parquet Utils Services
        self.register_singleton(
            'parquet_utils',
            object,
            factory=lambda: self._import_parquet_utils()
        )
        
        # Serialization Utils Services
        self.register_singleton(
            'serialization_utils',
            object,
            factory=lambda: self._import_serialization_utils()
        )
        
        # Data Processing Utils Services
        self.register_singleton(
            'data_processing_utils',
            object,
            factory=lambda: self._import_data_processing_utils()
        )
        
        # M1 GPU Utils Services
        self.register_singleton(
            'm1_gpu_utils',
            object,
            factory=lambda: self._import_m1_gpu_utils()
        )
        
        # M1 Memory Optimizer Services
        self.register_singleton(
            'm1_memory_optimizer',
            object,
            factory=lambda: self._import_m1_memory_optimizer()
        )
        
        # M1 CPU Optimizer Services
        self.register_singleton(
            'm1_cpu_optimizer',
            object,
            factory=lambda: self._import_m1_cpu_optimizer()
        )
        
        # Utility Manager Service
        self.register_singleton(
            'utility_manager',
            'UtilityManager',
            factory=lambda: self._create_utility_manager()
        )
    
    def _import_common_operations(self):
        """Import and return common operations utilities."""
        try:
            from src.utils.common_operations import (
                safe_read_parquet, safe_to_parquet, ensure_directory, safe_json_dump, safe_json_load,
                safe_mean, safe_std, safe_fillna, safe_rolling, create_empty_dataframe,
                get_current_datetime, format_datetime, parse_datetime, safe_file_exists,
                safe_append, safe_extend, safe_dict_get, safe_dict_items, safe_lower, safe_upper,
                safe_join, get_logger, setup_basic_logging, safe_float, safe_int,
                validate_dataframe, validate_numeric_range, optimize_dataframe_dtypes,
                safe_resample, align_dataframes, safe_copy, safe_deepcopy,
                generate_hash, generate_cache_key, list_parquet_files,
                safe_sleep, safe_gather, create_async_task, timed_operation,
                format_bytes, chunked_iterable, parallel_map, safe_log_metric,
                safe_log_params, safe_log_artifact, standardize_price_action_probabilities
            )
            
            return {
                'safe_read_parquet': safe_read_parquet,
                'safe_to_parquet': safe_to_parquet,
                'ensure_directory': ensure_directory,
                'safe_json_dump': safe_json_dump,
                'safe_json_load': safe_json_load,
                'safe_mean': safe_mean,
                'safe_std': safe_std,
                'safe_fillna': safe_fillna,
                'safe_rolling': safe_rolling,
                'create_empty_dataframe': create_empty_dataframe,
                'get_current_datetime': get_current_datetime,
                'format_datetime': format_datetime,
                'parse_datetime': parse_datetime,
                'safe_file_exists': safe_file_exists,
                'safe_append': safe_append,
                'safe_extend': safe_extend,
                'safe_dict_get': safe_dict_get,
                'safe_dict_items': safe_dict_items,
                'safe_lower': safe_lower,
                'safe_upper': safe_upper,
                'safe_join': safe_join,
                'get_logger': get_logger,
                'setup_basic_logging': setup_basic_logging,
                'safe_float': safe_float,
                'safe_int': safe_int,
                'validate_dataframe': validate_dataframe,
                'validate_numeric_range': validate_numeric_range,
                'optimize_dataframe_dtypes': optimize_dataframe_dtypes,
                'safe_resample': safe_resample,
                'align_dataframes': align_dataframes,
                'safe_copy': safe_copy,
                'safe_deepcopy': safe_deepcopy,
                'generate_hash': generate_hash,
                'generate_cache_key': generate_cache_key,
                'list_parquet_files': list_parquet_files,
                'safe_sleep': safe_sleep,
                'safe_gather': safe_gather,
                'create_async_task': create_async_task,
                'timed_operation': timed_operation,
                'format_bytes': format_bytes,
                'chunked_iterable': chunked_iterable,
                'parallel_map': parallel_map,
                'safe_log_metric': safe_log_metric,
                'safe_log_params': safe_log_params,
                'safe_log_artifact': safe_log_artifact,
                'standardize_price_action_probabilities': standardize_price_action_probabilities
            }
        except ImportError as e:
            self.logger.error(f"Failed to import common_operations: {e}")
            return {}
    
    def _import_common_utilities(self):
        """Import and return common utilities."""
        try:
            from src.utils.common_utilities import (
                safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
                calculate_data_quality_metrics, safe_merge_dataframes, safe_groupby_operation,
                safe_apply_function, create_summary_statistics, safe_drop_columns,
                safe_rename_columns, validate_timestamp_column, safe_timestamp_conversion,
                get_dataframe_info, safe_filter_dataframe, create_data_quality_report
            )
            
            return {
                'safe_dataframe_operation': safe_dataframe_operation,
                'validate_dataframe_columns': validate_dataframe_columns,
                'safe_convert_dtypes': safe_convert_dtypes,
                'calculate_data_quality_metrics': calculate_data_quality_metrics,
                'safe_merge_dataframes': safe_merge_dataframes,
                'safe_groupby_operation': safe_groupby_operation,
                'safe_apply_function': safe_apply_function,
                'create_summary_statistics': create_summary_statistics,
                'safe_drop_columns': safe_drop_columns,
                'safe_rename_columns': safe_rename_columns,
                'validate_timestamp_column': validate_timestamp_column,
                'safe_timestamp_conversion': safe_timestamp_conversion,
                'get_dataframe_info': get_dataframe_info,
                'safe_filter_dataframe': safe_filter_dataframe,
                'create_data_quality_report': create_data_quality_report
            }
        except ImportError as e:
            self.logger.error(f"Failed to import common_utilities: {e}")
            return {}
    
    def _import_math_validation(self):
        """Import and return math validation utilities."""
        try:
            from src.utils.math_validation import (
                safe_divide, safe_log, safe_sqrt, safe_power, validate_finite, validate_positive,
                validate_range, safe_kelly_calculation, safe_weighted_average, safe_percentage_change,
                validate_correlation_matrix, safe_matrix_inverse, math_safe
            )
            
            return {
                'safe_divide': safe_divide,
                'safe_log': safe_log,
                'safe_sqrt': safe_sqrt,
                'safe_power': safe_power,
                'validate_finite': validate_finite,
                'validate_positive': validate_positive,
                'validate_range': validate_range,
                'safe_kelly_calculation': safe_kelly_calculation,
                'safe_weighted_average': safe_weighted_average,
                'safe_percentage_change': safe_percentage_change,
                'validate_correlation_matrix': validate_correlation_matrix,
                'safe_matrix_inverse': safe_matrix_inverse,
                'math_safe': math_safe
            }
        except ImportError as e:
            self.logger.error(f"Failed to import math_validation: {e}")
            return {}
    
    def _import_parquet_utils(self):
        """Import and return parquet utilities."""
        try:
            from src.utils.parquet_utils import ParquetUtils, get_parquet_utils
            
            return {
                'ParquetUtils': ParquetUtils,
                'get_parquet_utils': get_parquet_utils
            }
        except ImportError as e:
            self.logger.error(f"Failed to import parquet_utils: {e}")
            return {}
    
    def _import_serialization_utils(self):
        """Import and return serialization utilities."""
        try:
            from src.utils.serialization_utils import (
                JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer,
                save_json, load_json, save_pickle, load_pickle, save_parquet, load_parquet,
                save_data, load_data
            )
            
            return {
                'JSONSerializer': JSONSerializer,
                'PickleSerializer': PickleSerializer,
                'ParquetSerializer': ParquetSerializer,
                'UniversalSerializer': UniversalSerializer,
                'save_json': save_json,
                'load_json': load_json,
                'save_pickle': save_pickle,
                'load_pickle': load_pickle,
                'save_parquet': save_parquet,
                'load_parquet': load_parquet,
                'save_data': save_data,
                'load_data': load_data
            }
        except ImportError as e:
            self.logger.error(f"Failed to import serialization_utils: {e}")
            return {}
    
    def _import_data_processing_utils(self):
        """Import and return data processing utilities."""
        try:
            from src.utils.data_processing_utils import (
                DataFrameValidator, DataFrameCleaner, DataFrameTransformer,
                validate_dataframe, clean_dataframe, transform_dataframe, get_dataframe_info
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
            self.logger.error(f"Failed to import data_processing_utils: {e}")
            return {}
    
    def _import_m1_gpu_utils(self):
        """Import and return M1 GPU utilities."""
        try:
            from src.utils.m1_gpu_utils import (
                M1GPUManager, M1PerformanceOptimizer, initialize_m1_gpu, get_m1_gpu_manager,
                m1_tensor_multiply, m1_batch_process, m1_monte_carlo_simulate
            )
            
            return {
                'M1GPUManager': M1GPUManager,
                'M1PerformanceOptimizer': M1PerformanceOptimizer,
                'initialize_m1_gpu': initialize_m1_gpu,
                'get_m1_gpu_manager': get_m1_gpu_manager,
                'm1_tensor_multiply': m1_tensor_multiply,
                'm1_batch_process': m1_batch_process,
                'm1_monte_carlo_simulate': m1_monte_carlo_simulate
            }
        except ImportError as e:
            self.logger.error(f"Failed to import m1_gpu_utils: {e}")
            return {}
    
    def _import_m1_memory_optimizer(self):
        """Import and return M1 memory optimizer."""
        try:
            from src.utils.m1_memory_optimizer import (
                M1MemoryOptimizer, M1DataManager, get_m1_memory_optimizer,
                create_memory_efficient_dataframe, memory_efficient_groupby
            )
            
            return {
                'M1MemoryOptimizer': M1MemoryOptimizer,
                'M1DataManager': M1DataManager,
                'get_m1_memory_optimizer': get_m1_memory_optimizer,
                'create_memory_efficient_dataframe': create_memory_efficient_dataframe,
                'memory_efficient_groupby': memory_efficient_groupby
            }
        except ImportError as e:
            self.logger.error(f"Failed to import m1_memory_optimizer: {e}")
            return {}
    
    def _import_m1_cpu_optimizer(self):
        """Import and return M1 CPU optimizer."""
        try:
            from src.utils.m1_cpu_optimizer import (
                M1CPUOptimizer, M1BatchProcessor, get_m1_cpu_optimizer, initialize_m1_cpu_optimizer,
                parallel_map, parallel_dataframe_operation, parallel_monte_carlo_simulation,
                optimized_monte_carlo_worker
            )
            
            return {
                'M1CPUOptimizer': M1CPUOptimizer,
                'M1BatchProcessor': M1BatchProcessor,
                'get_m1_cpu_optimizer': get_m1_cpu_optimizer,
                'initialize_m1_cpu_optimizer': initialize_m1_cpu_optimizer,
                'parallel_map': parallel_map,
                'parallel_dataframe_operation': parallel_dataframe_operation,
                'parallel_monte_carlo_simulation': parallel_monte_carlo_simulation,
                'optimized_monte_carlo_worker': optimized_monte_carlo_worker
            }
        except ImportError as e:
            self.logger.error(f"Failed to import m1_cpu_optimizer: {e}")
            return {}
    
    def _create_utility_manager(self):
        """Create and return the utility manager."""
        return UtilityManager(self)
    
    def register_singleton(self, name: str, service_type: Type, factory: Optional[Callable] = None, dependencies: list = None):
        """Register a singleton service."""
        with self._lock:
            self._services[name] = ServiceDefinition(
                service_type=service_type,
                implementation=service_type,
                singleton=True,
                dependencies=dependencies or [],
                factory=factory
            )
    
    def register_transient(self, name: str, service_type: Type, factory: Optional[Callable] = None, dependencies: list = None):
        """Register a transient service."""
        with self._lock:
            self._services[name] = ServiceDefinition(
                service_type=service_type,
                implementation=service_type,
                singleton=False,
                dependencies=dependencies or [],
                factory=factory
            )
    
    def get(self, name: str) -> Any:
        """Get a service instance."""
        with self._lock:
            if name not in self._services:
                raise ValueError(f"Service '{name}' not registered")
            
            service_def = self._services[name]
            
            if service_def.singleton and service_def.instance is not None:
                return service_def.instance
            
            # Create instance
            if service_def.factory:
                instance = service_def.factory()
            else:
                # Resolve dependencies
                dependencies = {}
                for dep_name in service_def.dependencies:
                    dependencies[dep_name] = self.get(dep_name)
                
                instance = service_def.implementation(**dependencies)
            
            if service_def.singleton:
                service_def.instance = instance
                service_def.initialized = True
            
            return instance
    
    def get_all_utilities(self) -> Dict[str, Any]:
        """Get all utility modules."""
        return {
            'common_operations': self.get('common_operations'),
            'common_utilities': self.get('common_utilities'),
            'math_validation': self.get('math_validation'),
            'parquet_utils': self.get('parquet_utils'),
            'serialization_utils': self.get('serialization_utils'),
            'data_processing_utils': self.get('data_processing_utils'),
            'm1_gpu_utils': self.get('m1_gpu_utils'),
            'm1_memory_optimizer': self.get('m1_memory_optimizer'),
            'm1_cpu_optimizer': self.get('m1_cpu_optimizer')
        }

class UtilityManager:
    """Manager class that provides easy access to all utilities."""
    
    def __init__(self, container: DependencyInjectionContainer):
        self.container = container
        self.logger = logging.getLogger(f"{__name__}.UtilityManager")
        self._utilities = None
        self._initialized = False
    
    def initialize(self):
        """Initialize all utilities."""
        if self._initialized:
            return
        
        self.logger.info("🔧 Initializing utility manager...")
        
        try:
            self._utilities = self.container.get_all_utilities()
            self._initialized = True
            self.logger.info("✅ Utility manager initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize utility manager: {e}")
            raise
    
    @property
    def common_ops(self):
        """Get common operations utilities."""
        if not self._initialized:
            self.initialize()
        return self._utilities['common_operations']
    
    @property
    def common_utils(self):
        """Get common utilities."""
        if not self._initialized:
            self.initialize()
        return self._utilities['common_utilities']
    
    @property
    def math_validation(self):
        """Get math validation utilities."""
        if not self._initialized:
            self.initialize()
        return self._utilities['math_validation']
    
    @property
    def parquet_utils(self):
        """Get parquet utilities."""
        if not self._initialized:
            self.initialize()
        return self._utilities['parquet_utils']
    
    @property
    def serialization_utils(self):
        """Get serialization utilities."""
        if not self._initialized:
            self.initialize()
        return self._utilities['serialization_utils']
    
    @property
    def data_processing_utils(self):
        """Get data processing utilities."""
        if not self._initialized:
            self.initialize()
        return self._utilities['data_processing_utils']
    
    @property
    def m1_gpu_utils(self):
        """Get M1 GPU utilities."""
        if not self._initialized:
            self.initialize()
        return self._utilities['m1_gpu_utils']
    
    @property
    def m1_memory_optimizer(self):
        """Get M1 memory optimizer."""
        if not self._initialized:
            self.initialize()
        return self._utilities['m1_memory_optimizer']
    
    @property
    def m1_cpu_optimizer(self):
        """Get M1 CPU optimizer."""
        if not self._initialized:
            self.initialize()
        return self._utilities['m1_cpu_optimizer']
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all utilities."""
        if not self._initialized:
            return {'status': 'not_initialized'}
        
        health_status = {
            'status': 'initialized',
            'utilities': {},
            'timestamp': time.time()
        }
        
        for name, utils in self._utilities.items():
            try:
                if name == 'common_operations' and 'get_common_operations_health_status' in utils:
                    health_status['utilities'][name] = utils['get_common_operations_health_status']()
                else:
                    health_status['utilities'][name] = {'status': 'available', 'functions': len(utils)}
            except Exception as e:
                health_status['utilities'][name] = {'status': 'error', 'error': str(e)}
        
        return health_status

# Global container instance
_container = None

def get_container() -> DependencyInjectionContainer:
    """Get the global dependency injection container."""
    global _container
    if _container is None:
        _container = DependencyInjectionContainer()
    return _container

def get_utility_manager() -> UtilityManager:
    """Get the global utility manager."""
    container = get_container()
    return container.get('utility_manager')