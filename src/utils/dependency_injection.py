"""
Dependency Injection Container for Step07 Utilities

This module provides a comprehensive dependency injection system for managing
and injecting utility dependencies throughout the step07 enhanced matrix operations.
"""

import logging
from typing import Any, Dict, List, Optional, Type, TypeVar, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import inspect
import functools

logger = logging.getLogger(__name__)

T = TypeVar('T')

@dataclass
class ServiceDefinition:
    """Definition of a service in the DI container."""
    service_type: Type
    implementation: Any
    singleton: bool = True
    dependencies: List[str] = field(default_factory=list)
    factory: Optional[Callable] = None
    instance: Optional[Any] = field(default=None, init=False)

class DependencyInjectionContainer:
    """Comprehensive dependency injection container for step07 utilities."""

    def __init__(self):
        self._services: Dict[str, ServiceDefinition] = {}
        self._instances: Dict[str, Any] = {}
        self.logger = logger.getChild('DIContainer')

    def register_singleton(self, name: str, service_type: Type, implementation: Any = None,
                          dependencies: List[str] = None) -> None:
        """Register a singleton service."""
        self._register_service(name, service_type, implementation, True, dependencies or [])

    def register_transient(self, name: str, service_type: Type, implementation: Any = None,
                          dependencies: List[str] = None) -> None:
        """Register a transient service."""
        self._register_service(name, service_type, implementation, False, dependencies or [])

    def register_factory(self, name: str, service_type: Type, factory: Callable,
                        dependencies: List[str] = None) -> None:
        """Register a service with a factory function."""
        self._services[name] = ServiceDefinition(
            service_type=service_type,
            implementation=None,
            singleton=True,
            dependencies=dependencies or [],
            factory=factory
        )
        self.logger.debug(f"Registered factory service: {name}")

    def _register_service(self, name: str, service_type: Type, implementation: Any,
                         singleton: bool, dependencies: List[str]) -> None:
        """Internal method to register a service."""
        if implementation is None:
            implementation = service_type

        self._services[name] = ServiceDefinition(
            service_type=service_type,
            implementation=implementation,
            singleton=singleton,
            dependencies=dependencies
        )
        self.logger.debug(f"Registered {'singleton' if singleton else 'transient'} service: {name}")

    def get(self, name: str) -> Any:
        """Get a service instance by name."""
        if name not in self._services:
            raise ValueError(f"Service '{name}' not registered")

        service_def = self._services[name]

        # Return existing singleton instance
        if service_def.singleton and name in self._instances:
            return self._instances[name]

        # Create new instance
        instance = self._create_instance(service_def)

        # Store singleton instance
        if service_def.singleton:
            self._instances[name] = instance

        return instance

    def get_by_type(self, service_type: Type[T]) -> T:
        """Get a service instance by type."""
        for name, service_def in self._services.items():
            if service_def.service_type == service_type:
                return self.get(name)
        raise ValueError(f"Service of type '{service_type}' not registered")

    def _create_instance(self, service_def: ServiceDefinition) -> Any:
        """Create an instance of a service."""
        try:
            # Use factory if available
            if service_def.factory:
                return self._create_from_factory(service_def)

            # Use implementation
            implementation = service_def.implementation

            # Check if it's a class that needs instantiation
            if inspect.isclass(implementation):
                return self._create_from_class(implementation, service_def.dependencies)
            else:
                # Already an instance
                return implementation

        except Exception as e:
            self.logger.error(f"Failed to create instance for {service_def.service_type}: {e}")
            raise

    def _create_from_factory(self, service_def: ServiceDefinition) -> Any:
        """Create instance from factory function."""
        # Resolve dependencies
        resolved_deps = [self.get(dep) for dep in service_def.dependencies]
        return service_def.factory(*resolved_deps)

    def _create_from_class(self, cls: Type, dependencies: List[str]) -> Any:
        """Create instance from class with dependency injection."""
        # Get constructor signature
        sig = inspect.signature(cls.__init__)
        params = {}

        # Resolve dependencies by name
        for dep_name in dependencies:
            if dep_name in self._services:
                params[dep_name] = self.get(dep_name)

        # Try to instantiate with resolved dependencies
        try:
            return cls(**params)
        except TypeError:
            # Fallback to no-args constructor
            return cls()

    def is_registered(self, name: str) -> bool:
        """Check if a service is registered."""
        return name in self._services

    def get_all_services(self) -> Dict[str, ServiceDefinition]:
        """Get all registered services."""
        return self._services.copy()

    def clear(self) -> None:
        """Clear all services and instances."""
        self._services.clear()
        self._instances.clear()
        self.logger.debug("Cleared all services and instances")

def inject_dependencies(func: Callable) -> Callable:
    """Decorator to inject dependencies into function parameters."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get the global container (you might want to pass this differently)
        container = get_global_container()

        # Get function signature
        sig = inspect.signature(func)

        # Inject dependencies
        for param_name, param in sig.parameters.items():
            if param_name not in kwargs and param_name in container._services:
                kwargs[param_name] = container.get(param_name)

        return func(*args, **kwargs)
    return wrapper

# Global container instance
_global_container: Optional[DependencyInjectionContainer] = None

def get_global_container() -> DependencyInjectionContainer:
    """Get the global dependency injection container."""
    global _global_container
    if _global_container is None:
        _global_container = DependencyInjectionContainer()
    return _global_container

def setup_step07_dependencies() -> DependencyInjectionContainer:
    """Setup all step07 utility dependencies."""
    container = get_global_container()

    # Clear existing services
    container.clear()

    # Register utility services
    _register_common_operations(container)
    _register_common_utilities(container)
    _register_math_validation(container)
    _register_parquet_utils(container)
    _register_serialization_utils(container)
    _register_data_processing_utils(container)
    _register_m1_optimizers(container)

    logger.info("✅ Step07 dependency injection container setup complete")
    return container

def _register_common_operations(container: DependencyInjectionContainer) -> None:
    """Register common operations utilities."""
    try:
        from .common_operations import (
            get_current_datetime, get_today, format_datetime, parse_datetime,
            create_empty_dataframe, safe_fillna, safe_rolling, safe_mean, safe_std,
            ensure_directory, safe_file_exists, safe_json_dump, safe_json_load,
            safe_sleep, safe_gather, create_async_task, safe_append, safe_extend,
            safe_dict_get, safe_dict_items, safe_lower, safe_upper, safe_join,
            get_logger, setup_basic_logging, create_argument_parser, add_common_arguments,
            safe_exception_handler, safe_float, safe_int, suggest_float_uniform,
            suggest_int_uniform, validate_dataframe, validate_numeric_range,
            validate_dataframe_schema, validate_data_quality, optimize_dataframe_dtypes,
            timed_operation, format_bytes, chunked_iterable, parallel_map,
            standardize_price_action_probabilities
        )

        # Register as singleton services
        container.register_singleton('datetime_utils', type(get_current_datetime), get_current_datetime)
        container.register_singleton('dataframe_utils', type(create_empty_dataframe), create_empty_dataframe)
        container.register_singleton('file_utils', type(ensure_directory), ensure_directory)
        container.register_singleton('json_utils', type(safe_json_dump), safe_json_dump)
        container.register_singleton('async_utils', type(safe_sleep), safe_sleep)
        container.register_singleton('list_utils', type(safe_append), safe_append)
        container.register_singleton('dict_utils', type(safe_dict_get), safe_dict_get)
        container.register_singleton('string_utils', type(safe_lower), safe_lower)
        container.register_singleton('logging_utils', type(get_logger), get_logger)
        container.register_singleton('validation_utils', type(validate_dataframe), validate_dataframe)
        container.register_singleton('math_utils', type(safe_mean), safe_mean)
        container.register_singleton('optimization_utils', type(optimize_dataframe_dtypes), optimize_dataframe_dtypes)
        container.register_singleton('performance_utils', type(timed_operation), timed_operation)
        container.register_singleton('parallel_utils', type(parallel_map), parallel_map)
        container.register_singleton('probability_utils', type(standardize_price_action_probabilities), standardize_price_action_probabilities)

        logger.debug("✅ Registered common_operations utilities")

    except ImportError as e:
        logger.warning(f"Failed to register common_operations utilities: {e}")

def _register_common_utilities(container: DependencyInjectionContainer) -> None:
    """Register common utilities."""
    try:
        from .common_utilities import (
            safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
            calculate_data_quality_metrics, safe_merge_dataframes, safe_groupby_operation,
            safe_apply_function, create_summary_statistics, safe_drop_columns,
            safe_rename_columns, validate_timestamp_column, safe_timestamp_conversion,
            get_dataframe_info, safe_filter_dataframe, create_data_quality_report
        )

        # Register as singleton services
        container.register_singleton('dataframe_operations', type(safe_dataframe_operation), safe_dataframe_operation)
        container.register_singleton('dataframe_validation', type(validate_dataframe_columns), validate_dataframe_columns)
        container.register_singleton('dataframe_conversion', type(safe_convert_dtypes), safe_convert_dtypes)
        container.register_singleton('data_quality_metrics', type(calculate_data_quality_metrics), calculate_data_quality_metrics)
        container.register_singleton('dataframe_merge', type(safe_merge_dataframes), safe_merge_dataframes)
        container.register_singleton('dataframe_groupby', type(safe_groupby_operation), safe_groupby_operation)
        container.register_singleton('dataframe_apply', type(safe_apply_function), safe_apply_function)
        container.register_singleton('summary_statistics', type(create_summary_statistics), create_summary_statistics)
        container.register_singleton('dataframe_drop', type(safe_drop_columns), safe_drop_columns)
        container.register_singleton('dataframe_rename', type(safe_rename_columns), safe_rename_columns)
        container.register_singleton('timestamp_validation', type(validate_timestamp_column), validate_timestamp_column)
        container.register_singleton('timestamp_conversion', type(safe_timestamp_conversion), safe_timestamp_conversion)
        container.register_singleton('dataframe_info', type(get_dataframe_info), get_dataframe_info)
        container.register_singleton('dataframe_filter', type(safe_filter_dataframe), safe_filter_dataframe)
        container.register_singleton('data_quality_report', type(create_data_quality_report), create_data_quality_report)

        logger.debug("✅ Registered common_utilities")

    except ImportError as e:
        logger.warning(f"Failed to register common_utilities: {e}")

def _register_math_validation(container: DependencyInjectionContainer) -> None:
    """Register math validation utilities."""
    try:
        from ..math_validation import (
            safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
            validate_positive, validate_range, safe_kelly_calculation,
            safe_weighted_average, safe_percentage_change, validate_correlation_matrix,
            safe_matrix_inverse, math_safe
        )

        # Register as singleton services
        container.register_singleton('safe_divide', type(safe_divide), safe_divide)
        container.register_singleton('safe_log', type(safe_log), safe_log)
        container.register_singleton('safe_sqrt', type(safe_sqrt), safe_sqrt)
        container.register_singleton('safe_power', type(safe_power), safe_power)
        container.register_singleton('validate_finite', type(validate_finite), validate_finite)
        container.register_singleton('validate_positive', type(validate_positive), validate_positive)
        container.register_singleton('validate_range', type(validate_range), validate_range)
        container.register_singleton('kelly_calculation', type(safe_kelly_calculation), safe_kelly_calculation)
        container.register_singleton('weighted_average', type(safe_weighted_average), safe_weighted_average)
        container.register_singleton('percentage_change', type(safe_percentage_change), safe_percentage_change)
        container.register_singleton('correlation_validation', type(validate_correlation_matrix), validate_correlation_matrix)
        container.register_singleton('matrix_inverse', type(safe_matrix_inverse), safe_matrix_inverse)
        container.register_singleton('math_safe_decorator', type(math_safe), math_safe)

        logger.debug("✅ Registered math_validation utilities")

    except ImportError as e:
        logger.warning(f"Failed to register math_validation utilities: {e}")

def _register_parquet_utils(container: DependencyInjectionContainer) -> None:
    """Register parquet utilities."""
    try:
        from src.utils.parquet_utils import ParquetUtils, get_parquet_utils

        # Register as singleton services
        container.register_singleton('parquet_utils', ParquetUtils, get_parquet_utils)

        logger.debug("✅ Registered parquet_utils")

    except ImportError as e:
        logger.warning(f"Failed to register parquet_utils: {e}")

def _register_serialization_utils(container: DependencyInjectionContainer) -> None:
    """Register serialization utilities."""
    try:
        from .serialization_utils import (
            JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer,
            save_json, load_json, save_pickle, load_pickle, save_parquet, load_parquet,
            save_data, load_data
        )

        # Register as singleton services
        container.register_singleton('json_serializer', JSONSerializer, JSONSerializer)
        container.register_singleton('pickle_serializer', PickleSerializer, PickleSerializer)
        container.register_singleton('parquet_serializer', ParquetSerializer, ParquetSerializer)
        container.register_singleton('universal_serializer', UniversalSerializer, UniversalSerializer)
        container.register_singleton('save_json', type(save_json), save_json)
        container.register_singleton('load_json', type(load_json), load_json)
        container.register_singleton('save_pickle', type(save_pickle), save_pickle)
        container.register_singleton('load_pickle', type(load_pickle), load_pickle)
        container.register_singleton('save_parquet', type(save_parquet), save_parquet)
        container.register_singleton('load_parquet', type(load_parquet), load_parquet)
        container.register_singleton('save_data', type(save_data), save_data)
        container.register_singleton('load_data', type(load_data), load_data)

        logger.debug("✅ Registered serialization_utils")

    except ImportError as e:
        logger.warning(f"Failed to register serialization_utils: {e}")

def _register_data_processing_utils(container: DependencyInjectionContainer) -> None:
    """Register data processing utilities."""
    try:
        from .data_processing_utils import (
            DataFrameValidator, DataFrameCleaner, DataFrameTransformer,
            validate_dataframe, clean_dataframe, transform_dataframe, get_dataframe_info
        )

        # Register as singleton services
        container.register_singleton('dataframe_validator', DataFrameValidator, DataFrameValidator)
        container.register_singleton('dataframe_cleaner', DataFrameCleaner, DataFrameCleaner)
        container.register_singleton('dataframe_transformer', DataFrameTransformer, DataFrameTransformer)
        container.register_singleton('validate_dataframe_util', type(validate_dataframe), validate_dataframe)
        container.register_singleton('clean_dataframe_util', type(clean_dataframe), clean_dataframe)
        container.register_singleton('transform_dataframe_util', type(transform_dataframe), transform_dataframe)
        container.register_singleton('get_dataframe_info_util', type(get_dataframe_info), get_dataframe_info)

        logger.debug("✅ Registered data_processing_utils")

    except ImportError as e:
        logger.warning(f"Failed to register data_processing_utils: {e}")

def _register_m1_optimizers(container: DependencyInjectionContainer) -> None:
    """Register M1 optimization utilities."""
    try:
        from .hardware.m1_gpu_utils import M1GPUManager, get_m1_gpu_manager, initialize_m1_gpu
        from .hardware.m1_memory_optimizer import M1MemoryOptimizer, get_m1_memory_optimizer
        from .hardware.m1_cpu_optimizer import M1CPUOptimizer, get_m1_cpu_optimizer, initialize_m1_cpu_optimizer

        # Register as singleton services
        container.register_singleton('m1_gpu_manager', M1GPUManager, get_m1_gpu_manager)
        container.register_singleton('m1_memory_optimizer', M1MemoryOptimizer, get_m1_memory_optimizer)
        container.register_singleton('m1_cpu_optimizer', M1CPUOptimizer, get_m1_cpu_optimizer)

        logger.debug("✅ Registered M1 optimization utilities")

    except ImportError as e:
        logger.warning(f"Failed to register M1 optimization utilities: {e}")

# Convenience functions for step07
def get_step07_utility(utility_name: str) -> Any:
    """Get a step07 utility by name."""
    container = get_global_container()
    return container.get(utility_name)

def inject_step07_utilities(func: Callable) -> Callable:
    """Decorator to inject step07 utilities into function parameters."""
    return inject_dependencies(func)

__all__ = [
    'DependencyInjectionContainer',
    'ServiceDefinition',
    'inject_dependencies',
    'get_global_container',
    'setup_step07_dependencies',
    'get_step07_utility',
    'inject_step07_utilities'
]
