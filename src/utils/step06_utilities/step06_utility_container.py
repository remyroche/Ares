"""
Step06 Utility Container with Dependency Injection

This module provides a comprehensive dependency injection container for all step06 utilities,
ensuring proper initialization, configuration, and lifecycle management of utility services.
"""

import logging
from typing import Any, Dict, List, Optional, Type, TypeVar, Callable, Union
from functools import wraps
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import asyncio
from contextlib import asynccontextmanager
import threading
import time

# Import all utility modules
from src.utils.common_operations import (
    get_current_datetime, get_today, format_datetime, parse_datetime,
    create_empty_dataframe, safe_fillna, safe_rolling, safe_mean, safe_std,
    ensure_directory, safe_file_exists, safe_json_dump, safe_json_load,
    safe_sleep, safe_gather, create_async_task, safe_append, safe_extend,
    safe_dict_get, safe_dict_items, safe_lower, safe_upper, safe_join,
    get_logger, setup_basic_logging, safe_float, safe_int,
    validate_dataframe, validate_numeric_range, optimize_dataframe_dtypes,
    timed_operation, format_bytes, chunked_iterable, parallel_map,
    safe_log_metric, safe_log_params, safe_log_artifact
)

from src.utils.common_utilities import (
    safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
    calculate_data_quality_metrics, safe_merge_dataframes, safe_groupby_operation,
    safe_apply_function, create_summary_statistics, safe_drop_columns,
    safe_rename_columns, validate_timestamp_column, safe_timestamp_conversion,
    get_dataframe_info, safe_filter_dataframe, create_data_quality_report
)

from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    validate_positive, validate_range, safe_kelly_calculation,
    safe_weighted_average, safe_percentage_change, validate_correlation_matrix,
    safe_matrix_inverse, math_safe, MathValidationError
)

from src.utils.parquet_utils import ParquetUtils, get_parquet_utils

from src.utils.serialization_utils import (
    JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer,
    save_json, load_json, save_pickle, load_pickle, save_parquet, load_parquet,
    save_data, load_data, SerializationError
)

from src.utils.data_processing_utils import (
    DataFrameValidator, DataFrameCleaner, DataFrameTransformer,
    validate_dataframe as validate_df_advanced, clean_dataframe, transform_dataframe,
    get_dataframe_info as get_df_info_advanced, DataQualityLevel, DataQualityIssue,
    DataQualityReport
)

from src.utils.m1_gpu_utils import (
    M1GPUManager, M1PerformanceOptimizer, initialize_m1_gpu, get_m1_gpu_manager,
    m1_tensor_multiply, m1_batch_process, m1_monte_carlo_simulate
)

from src.utils.m1_memory_optimizer import (
    M1MemoryOptimizer, M1DataManager, get_m1_memory_optimizer,
    create_memory_efficient_dataframe, memory_efficient_groupby
)

from src.utils.m1_cpu_optimizer import (
    M1CPUOptimizer, M1BatchProcessor, get_m1_cpu_optimizer, initialize_m1_cpu_optimizer,
    parallel_map as cpu_parallel_map, parallel_dataframe_operation, parallel_monte_carlo_simulation
)

T = TypeVar('T')

@dataclass
class UtilityConfig:
    """Configuration for utility services."""
    # Common operations config
    enable_common_operations: bool = True
    common_operations_log_level: str = "INFO"
    
    # Data processing config
    enable_data_processing: bool = True
    data_processing_chunk_size: int = 10000
    data_processing_memory_limit_mb: int = 1000
    
    # Math validation config
    enable_math_validation: bool = True
    math_validation_epsilon: float = 1e-10
    
    # Parquet utils config
    enable_parquet_utils: bool = True
    parquet_compression: str = "snappy"
    
    # Serialization config
    enable_serialization: bool = True
    serialization_compression: bool = False
    
    # M1 optimization config
    enable_m1_gpu: bool = True
    enable_m1_memory: bool = True
    enable_m1_cpu: bool = True
    m1_memory_limit_gb: float = 8.0
    m1_max_workers: int = 8
    
    # Performance config
    enable_performance_tracking: bool = True
    performance_log_interval: int = 100

class ServiceLifecycle(ABC):
    """Abstract base class for service lifecycle management."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the service."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup the service."""
        pass
    
    @abstractmethod
    def is_healthy(self) -> bool:
        """Check if the service is healthy."""
        pass

class UtilityService(ServiceLifecycle):
    """Base class for utility services."""
    
    def __init__(self, name: str, config: UtilityConfig):
        self.name = name
        self.config = config
        self.logger = get_logger(f"Step06Utility.{name}")
        self._initialized = False
        self._healthy = False
    
    async def initialize(self) -> None:
        """Initialize the service."""
        if self._initialized:
            return
        
        self.logger.info(f"🔧 Initializing {self.name} service")
        await self._do_initialize()
        self._initialized = True
        self._healthy = True
        self.logger.info(f"✅ {self.name} service initialized")
    
    async def cleanup(self) -> None:
        """Cleanup the service."""
        if not self._initialized:
            return
        
        self.logger.info(f"🧹 Cleaning up {self.name} service")
        await self._do_cleanup()
        self._initialized = False
        self._healthy = False
        self.logger.info(f"✅ {self.name} service cleaned up")
    
    def is_healthy(self) -> bool:
        """Check if the service is healthy."""
        return self._healthy
    
    @abstractmethod
    async def _do_initialize(self) -> None:
        """Service-specific initialization."""
        pass
    
    @abstractmethod
    async def _do_cleanup(self) -> None:
        """Service-specific cleanup."""
        pass

class CommonOperationsService(UtilityService):
    """Service for common operations utilities."""
    
    def __init__(self, config: UtilityConfig):
        super().__init__("CommonOperations", config)
        self.operations = {
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
                'safe_std': safe_std
            },
            'file': {
                'ensure_directory': ensure_directory,
                'safe_file_exists': safe_file_exists,
                'safe_json_dump': safe_json_dump,
                'safe_json_load': safe_json_load
            },
            'async': {
                'safe_sleep': safe_sleep,
                'safe_gather': safe_gather,
                'create_async_task': create_async_task
            },
            'list': {
                'safe_append': safe_append,
                'safe_extend': safe_extend,
                'safe_dict_get': safe_dict_get,
                'safe_dict_items': safe_dict_items
            },
            'string': {
                'safe_lower': safe_lower,
                'safe_upper': safe_upper,
                'safe_join': safe_join
            },
            'validation': {
                'validate_dataframe': validate_dataframe,
                'validate_numeric_range': validate_numeric_range,
                'safe_float': safe_float,
                'safe_int': safe_int
            },
            'performance': {
                'timed_operation': timed_operation,
                'format_bytes': format_bytes,
                'chunked_iterable': chunked_iterable,
                'parallel_map': parallel_map
            }
        }
    
    async def _do_initialize(self) -> None:
        """Initialize common operations service."""
        if self.config.enable_common_operations:
            setup_basic_logging(getattr(logging, self.config.common_operations_log_level))
            self.logger.info("✅ Common operations utilities loaded")
    
    async def _do_cleanup(self) -> None:
        """Cleanup common operations service."""
        pass
    
    def get_operation(self, category: str, operation: str) -> Callable:
        """Get a specific operation."""
        if category not in self.operations:
            raise ValueError(f"Unknown operation category: {category}")
        if operation not in self.operations[category]:
            raise ValueError(f"Unknown operation: {operation} in category: {category}")
        return self.operations[category][operation]

class DataProcessingService(UtilityService):
    """Service for data processing utilities."""
    
    def __init__(self, config: UtilityConfig):
        super().__init__("DataProcessing", config)
        self.validator = None
        self.cleaner = None
        self.transformer = None
    
    async def _do_initialize(self) -> None:
        """Initialize data processing service."""
        if self.config.enable_data_processing:
            self.validator = DataFrameValidator()
            self.cleaner = DataFrameCleaner({
                'chunk_size': self.config.data_processing_chunk_size,
                'memory_limit_mb': self.config.data_processing_memory_limit_mb
            })
            self.transformer = DataFrameTransformer()
            self.logger.info("✅ Data processing utilities initialized")
    
    async def _do_cleanup(self) -> None:
        """Cleanup data processing service."""
        self.validator = None
        self.cleaner = None
        self.transformer = None

class MathValidationService(UtilityService):
    """Service for mathematical validation utilities."""
    
    def __init__(self, config: UtilityConfig):
        super().__init__("MathValidation", config)
        self.epsilon = config.math_validation_epsilon
    
    async def _do_initialize(self) -> None:
        """Initialize math validation service."""
        if self.config.enable_math_validation:
            self.logger.info(f"✅ Math validation utilities initialized (epsilon: {self.epsilon})")
    
    async def _do_cleanup(self) -> None:
        """Cleanup math validation service."""
        pass

class ParquetService(UtilityService):
    """Service for parquet utilities."""
    
    def __init__(self, config: UtilityConfig):
        super().__init__("Parquet", config)
        self.parquet_utils = None
    
    async def _do_initialize(self) -> None:
        """Initialize parquet service."""
        if self.config.enable_parquet_utils:
            self.parquet_utils = get_parquet_utils()
            self.logger.info(f"✅ Parquet utilities initialized (compression: {self.config.parquet_compression})")
    
    async def _do_cleanup(self) -> None:
        """Cleanup parquet service."""
        self.parquet_utils = None

class SerializationService(UtilityService):
    """Service for serialization utilities."""
    
    def __init__(self, config: UtilityConfig):
        super().__init__("Serialization", config)
        self.serializers = {}
    
    async def _do_initialize(self) -> None:
        """Initialize serialization service."""
        if self.config.enable_serialization:
            self.serializers = {
                'json': JSONSerializer(),
                'pickle': PickleSerializer(),
                'parquet': ParquetSerializer(),
                'universal': UniversalSerializer()
            }
            self.logger.info(f"✅ Serialization utilities initialized (compression: {self.config.serialization_compression})")
    
    async def _do_cleanup(self) -> None:
        """Cleanup serialization service."""
        self.serializers = {}

class M1GPUService(UtilityService):
    """Service for M1 GPU utilities."""
    
    def __init__(self, config: UtilityConfig):
        super().__init__("M1GPU", config)
        self.gpu_manager = None
        self.performance_optimizer = None
    
    async def _do_initialize(self) -> None:
        """Initialize M1 GPU service."""
        if self.config.enable_m1_gpu:
            self.gpu_manager = initialize_m1_gpu()
            self.performance_optimizer = M1PerformanceOptimizer(self.gpu_manager)
            self.logger.info("✅ M1 GPU utilities initialized")
    
    async def _do_cleanup(self) -> None:
        """Cleanup M1 GPU service."""
        if self.gpu_manager:
            self.gpu_manager.optimize_memory()
        self.gpu_manager = None
        self.performance_optimizer = None

class M1MemoryService(UtilityService):
    """Service for M1 memory utilities."""
    
    def __init__(self, config: UtilityConfig):
        super().__init__("M1Memory", config)
        self.memory_optimizer = None
        self.data_manager = None
    
    async def _do_initialize(self) -> None:
        """Initialize M1 memory service."""
        if self.config.enable_m1_memory:
            self.memory_optimizer = M1MemoryOptimizer(
                memory_limit_gb=self.config.m1_memory_limit_gb
            )
            self.data_manager = M1DataManager(self.memory_optimizer)
            self.logger.info(f"✅ M1 memory utilities initialized (limit: {self.config.m1_memory_limit_gb}GB)")
    
    async def _do_cleanup(self) -> None:
        """Cleanup M1 memory service."""
        if self.memory_optimizer:
            self.memory_optimizer.optimize_memory()
        self.memory_optimizer = None
        self.data_manager = None

class M1CPUService(UtilityService):
    """Service for M1 CPU utilities."""
    
    def __init__(self, config: UtilityConfig):
        super().__init__("M1CPU", config)
        self.cpu_optimizer = None
        self.batch_processor = None
    
    async def _do_initialize(self) -> None:
        """Initialize M1 CPU service."""
        if self.config.enable_m1_cpu:
            self.cpu_optimizer = initialize_m1_cpu_optimizer()
            self.batch_processor = M1BatchProcessor(
                self.cpu_optimizer, 
                batch_size=self.config.data_processing_chunk_size
            )
            self.logger.info(f"✅ M1 CPU utilities initialized (max workers: {self.config.m1_max_workers})")
    
    async def _do_cleanup(self) -> None:
        """Cleanup M1 CPU service."""
        self.cpu_optimizer = None
        self.batch_processor = None

class Step06UtilityContainer:
    """Dependency injection container for step06 utilities."""
    
    def __init__(self, config: Optional[UtilityConfig] = None):
        self.config = config or UtilityConfig()
        self.logger = get_logger("Step06UtilityContainer")
        self._services: Dict[str, UtilityService] = {}
        self._initialized = False
        self._lock = threading.Lock()
    
    async def initialize(self) -> None:
        """Initialize all utility services."""
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
            
            self.logger.info("🚀 Initializing Step06 Utility Container")
            
            # Initialize all services
            services = [
                CommonOperationsService(self.config),
                DataProcessingService(self.config),
                MathValidationService(self.config),
                ParquetService(self.config),
                SerializationService(self.config),
                M1GPUService(self.config),
                M1MemoryService(self.config),
                M1CPUService(self.config)
            ]
            
            for service in services:
                try:
                    await service.initialize()
                    self._services[service.name] = service
                except Exception as e:
                    self.logger.error(f"❌ Failed to initialize {service.name}: {e}")
                    raise
            
            self._initialized = True
            self.logger.info("✅ Step06 Utility Container initialized")
    
    async def cleanup(self) -> None:
        """Cleanup all utility services."""
        if not self._initialized:
            return
        
        with self._lock:
            if not self._initialized:
                return
            
            self.logger.info("🧹 Cleaning up Step06 Utility Container")
            
            for service in reversed(list(self._services.values())):
                try:
                    await service.cleanup()
                except Exception as e:
                    self.logger.error(f"❌ Failed to cleanup {service.name}: {e}")
            
            self._services.clear()
            self._initialized = False
            self.logger.info("✅ Step06 Utility Container cleaned up")
    
    def get_service(self, service_name: str) -> UtilityService:
        """Get a utility service by name."""
        if not self._initialized:
            raise RuntimeError("Container not initialized")
        
        if service_name not in self._services:
            raise ValueError(f"Service not found: {service_name}")
        
        return self._services[service_name]
    
    def get_common_operations(self) -> CommonOperationsService:
        """Get common operations service."""
        return self.get_service("CommonOperations")
    
    def get_data_processing(self) -> DataProcessingService:
        """Get data processing service."""
        return self.get_service("DataProcessing")
    
    def get_math_validation(self) -> MathValidationService:
        """Get math validation service."""
        return self.get_service("MathValidation")
    
    def get_parquet(self) -> ParquetService:
        """Get parquet service."""
        return self.get_service("Parquet")
    
    def get_serialization(self) -> SerializationService:
        """Get serialization service."""
        return self.get_service("Serialization")
    
    def get_m1_gpu(self) -> M1GPUService:
        """Get M1 GPU service."""
        return self.get_service("M1GPU")
    
    def get_m1_memory(self) -> M1MemoryService:
        """Get M1 memory service."""
        return self.get_service("M1Memory")
    
    def get_m1_cpu(self) -> M1CPUService:
        """Get M1 CPU service."""
        return self.get_service("M1CPU")
    
    def is_healthy(self) -> bool:
        """Check if all services are healthy."""
        if not self._initialized:
            return False
        
        return all(service.is_healthy() for service in self._services.values())
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get health report for all services."""
        if not self._initialized:
            return {"status": "not_initialized", "services": {}}
        
        services_health = {}
        for name, service in self._services.items():
            services_health[name] = {
                "healthy": service.is_healthy(),
                "initialized": service._initialized
            }
        
        return {
            "status": "healthy" if all(s["healthy"] for s in services_health.values()) else "unhealthy",
            "services": services_health,
            "total_services": len(self._services),
            "healthy_services": sum(1 for s in services_health.values() if s["healthy"])
        }

# Global container instance
_container: Optional[Step06UtilityContainer] = None
_container_lock = threading.Lock()

async def get_utility_container(config: Optional[UtilityConfig] = None) -> Step06UtilityContainer:
    """Get or create the global utility container."""
    global _container
    
    with _container_lock:
        if _container is None:
            _container = Step06UtilityContainer(config)
            await _container.initialize()
        return _container

async def cleanup_utility_container() -> None:
    """Cleanup the global utility container."""
    global _container
    
    with _container_lock:
        if _container is not None:
            await _container.cleanup()
            _container = None

@asynccontextmanager
async def utility_container_context(config: Optional[UtilityConfig] = None):
    """Context manager for utility container lifecycle."""
    container = await get_utility_container(config)
    try:
        yield container
    finally:
        await cleanup_utility_container()

# Convenience functions for quick access
async def get_common_ops() -> CommonOperationsService:
    """Get common operations service."""
    container = await get_utility_container()
    return container.get_common_operations()

async def get_data_proc() -> DataProcessingService:
    """Get data processing service."""
    container = await get_utility_container()
    return container.get_data_processing()

async def get_math_val() -> MathValidationService:
    """Get math validation service."""
    container = await get_utility_container()
    return container.get_math_validation()

async def get_parquet_svc() -> ParquetService:
    """Get parquet service."""
    container = await get_utility_container()
    return container.get_parquet()

async def get_serialization_svc() -> SerializationService:
    """Get serialization service."""
    container = await get_utility_container()
    return container.get_serialization()

async def get_m1_gpu_svc() -> M1GPUService:
    """Get M1 GPU service."""
    container = await get_utility_container()
    return container.get_m1_gpu()

async def get_m1_memory_svc() -> M1MemoryService:
    """Get M1 memory service."""
    container = await get_utility_container()
    return container.get_m1_memory()

async def get_m1_cpu_svc() -> M1CPUService:
    """Get M1 CPU service."""
    container = await get_utility_container()
    return container.get_m1_cpu()

# Decorator for automatic utility injection
def inject_utilities(*service_names: str):
    """Decorator to inject utility services into functions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            container = await get_utility_container()
            injected_services = {}
            
            for service_name in service_names:
                if service_name == "common_ops":
                    injected_services[service_name] = container.get_common_operations()
                elif service_name == "data_proc":
                    injected_services[service_name] = container.get_data_processing()
                elif service_name == "math_val":
                    injected_services[service_name] = container.get_math_validation()
                elif service_name == "parquet":
                    injected_services[service_name] = container.get_parquet()
                elif service_name == "serialization":
                    injected_services[service_name] = container.get_serialization()
                elif service_name == "m1_gpu":
                    injected_services[service_name] = container.get_m1_gpu()
                elif service_name == "m1_memory":
                    injected_services[service_name] = container.get_m1_memory()
                elif service_name == "m1_cpu":
                    injected_services[service_name] = container.get_m1_cpu()
                else:
                    raise ValueError(f"Unknown service: {service_name}")
            
            return await func(*args, **kwargs, **injected_services)
        return wrapper
    return decorator

# Example usage and testing
async def test_utility_container():
    """Test the utility container functionality."""
    config = UtilityConfig(
        enable_common_operations=True,
        enable_data_processing=True,
        enable_math_validation=True,
        enable_parquet_utils=True,
        enable_serialization=True,
        enable_m1_gpu=True,
        enable_m1_memory=True,
        enable_m1_cpu=True
    )
    
    async with utility_container_context(config) as container:
        # Test service access
        common_ops = container.get_common_operations()
        data_proc = container.get_data_processing()
        math_val = container.get_math_validation()
        
        # Test health
        health_report = container.get_health_report()
        print(f"Health Report: {health_report}")
        
        # Test operations
        current_time = common_ops.get_operation('datetime', 'get_current_datetime')()
        print(f"Current time: {current_time}")
        
        return health_report

if __name__ == "__main__":
    asyncio.run(test_utility_container())