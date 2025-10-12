from src.utils.tprint import tprint

"""
Step06 Utility Container with Dependency Injection

This module provides a comprehensive dependency injection container for all step06 utilities,
ensuring proper initialization, configuration, and lifecycle management of utility services.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Type, TypeVar, Callable, Union
from functools import wraps
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import asyncio
from contextlib import asynccontextmanager
import threading
import time

# Import all utility modules
# Common operations - using basic Python functions for now
import os
import json
from pathlib import Path
from datetime import datetime

# Common utilities - simplified for now

# Math validation functions - defined inline to avoid import issues
import numpy as np
import pandas as pd

# Define missing utility functions
def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)

def get_current_datetime() -> datetime:
    """Get current datetime."""
    return datetime.now()

def get_today() -> str:
    """Get today's date as string."""
    return datetime.now().strftime('%Y-%m-%d')

def format_datetime(dt: datetime, format_str: str = '%Y-%m-%d %H:%M:%S') -> str:
    """Format datetime."""
    return dt.strftime(format_str)

def parse_datetime(date_str: str, format_str: str = '%Y-%m-%d %H:%M:%S') -> datetime:
    """Parse datetime string."""
    return datetime.strptime(date_str, format_str)

def create_empty_dataframe(columns: List[str]) -> pd.DataFrame:
    """Create empty dataframe."""
    return pd.DataFrame(columns=columns)

def safe_fillna(df: pd.DataFrame, value: Any = 0) -> pd.DataFrame:
    """Safely fill NaN values."""
    return df.fillna(value)

def safe_rolling(series: pd.Series, window: int, func: str = 'mean') -> pd.Series:
    """Safely apply rolling function."""
    try:
        return getattr(series.rolling(window), func)()
    except:
        return pd.Series([np.nan] * len(series), index=series.index)

def safe_mean(series: pd.Series) -> float:
    """Safely calculate mean."""
    try:
        return series.mean()
    except:
        return 0.0

def safe_std(series: pd.Series) -> float:
    """Safely calculate standard deviation."""
    try:
        return series.std()
    except:
        return 0.0

def ensure_directory(path: Union[str, Path]) -> None:
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)

def safe_file_exists(path: Union[str, Path]) -> bool:
    """Safely check if file exists."""
    try:
        return Path(path).exists()
    except:
        return False

def safe_json_dump(data: Any, filepath: Union[str, Path]) -> None:
    """Safely dump JSON data."""
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        logging.error(f"Failed to dump JSON: {e}")

def safe_json_load(filepath: Union[str, Path]) -> Any:
    """Safely load JSON data."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load JSON: {e}")
        return {}

async def safe_sleep(seconds: float) -> None:
    """Safely sleep."""
    await asyncio.sleep(seconds)

async def safe_gather(*coros) -> List[Any]:
    """Safely gather coroutines."""
    try:
        return await asyncio.gather(*coros)
    except Exception as e:
        logging.error(f"Failed to gather coroutines: {e}")
        return []

def create_async_task(coro) -> asyncio.Task:
    """Create async task."""
    return asyncio.create_task(coro)

def safe_append(lst: List[Any], item: Any) -> None:
    """Safely append to list."""
    try:
        lst.append(item)
    except:
        pass

def safe_extend(lst: List[Any], items: List[Any]) -> None:
    """Safely extend list."""
    try:
        lst.extend(items)
    except:
        pass

def safe_dict_get(d: Dict[Any, Any], key: Any, default: Any = None) -> Any:
    """Safely get from dict."""
    try:
        return d.get(key, default)
    except:
        return default

def safe_dict_items(d: Dict[Any, Any]) -> List[tuple]:
    """Safely get dict items."""
    try:
        return list(d.items())
    except:
        return []

def safe_lower(s: str) -> str:
    """Safely convert to lowercase."""
    try:
        return s.lower()
    except:
        return s

def safe_upper(s: str) -> str:
    """Safely convert to uppercase."""
    try:
        return s.upper()
    except:
        return s

def safe_join(sep: str, items: List[str]) -> str:
    """Safely join strings."""
    try:
        return sep.join(str(item) for item in items)
    except:
        return ""

def validate_dataframe(df: pd.DataFrame) -> bool:
    """Validate dataframe."""
    try:
        return isinstance(df, pd.DataFrame) and not df.empty
    except:
        return False

def validate_numeric_range(value: float, min_val: float, max_val: float) -> bool:
    """Validate numeric range."""
    try:
        return min_val <= value <= max_val
    except:
        return False

def safe_float(value: Any) -> float:
    """Safely convert to float."""
    try:
        return float(value)
    except:
        return 0.0

def safe_int(value: Any) -> int:
    """Safely convert to int."""
    try:
        return int(value)
    except:
        return 0

def timed_operation(func: Callable) -> Callable:
    """Decorator to time operations."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logging.info(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper

def format_bytes(bytes_value: int) -> str:
    """Format bytes to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} TB"

def chunked_iterable(iterable: List[Any], chunk_size: int) -> List[List[Any]]:
    """Chunk iterable into smaller pieces."""
    return [iterable[i:i + chunk_size] for i in range(0, len(iterable), chunk_size)]

def parallel_map(func: Callable, items: List[Any], max_workers: int = 4) -> List[Any]:
    """Apply function to items in parallel."""
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(func, items))

def setup_basic_logging(level: int = logging.INFO) -> None:
    """Setup basic logging."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# Define missing classes
class DataFrameValidator:
    """DataFrame validator."""
    def __init__(self):
        pass
    
    def validate(self, df: pd.DataFrame) -> bool:
        return validate_dataframe(df)

class DataFrameCleaner:
    """DataFrame cleaner."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna()

class DataFrameTransformer:
    """DataFrame transformer."""
    def __init__(self):
        pass
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

def get_parquet_utils():
    """Get parquet utilities."""
    return ParquetSerializer()

def initialize_m1_gpu():
    """Initialize M1 GPU."""
    return M1MemoryOptimizer()

class M1PerformanceOptimizer:
    """M1 Performance Optimizer."""
    def __init__(self, gpu_manager):
        self.gpu_manager = gpu_manager

def initialize_m1_cpu_optimizer():
    """Initialize M1 CPU optimizer."""
    return M1MemoryOptimizer()

class M1BatchProcessor:
    """M1 Batch Processor."""
    def __init__(self, cpu_optimizer, batch_size: int = 1000):
        self.cpu_optimizer = cpu_optimizer
        self.batch_size = batch_size

# Import math validation functions from shared module
from .math_validation import safe_divide, safe_log, safe_sqrt, validate_positive

# Parquet utilities - simplified for now

# Serialization utilities - simplified for now
import pickle

class JSONSerializer:
    @staticmethod
    def save(data, filepath):
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    @staticmethod
    def load(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)

class PickleSerializer:
    @staticmethod
    def save(data, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)

class ParquetSerializer:
    @staticmethod
    def save(data, filepath):
        data.to_parquet(filepath)
    
    @staticmethod
    def load(filepath):
        return pd.read_parquet(filepath)

class UniversalSerializer:
    @staticmethod
    def save(data, filepath):
        if filepath.endswith('.json'):
            JSONSerializer.save(data, filepath)
        elif filepath.endswith('.pkl'):
            PickleSerializer.save(data, filepath)
        elif filepath.endswith('.parquet'):
            ParquetSerializer.save(data, filepath)
    
    @staticmethod
    def load(filepath):
        if filepath.endswith('.json'):
            return JSONSerializer.load(filepath)
        elif filepath.endswith('.pkl'):
            return PickleSerializer.load(filepath)
        elif filepath.endswith('.parquet'):
            return ParquetSerializer.load(filepath)

class SerializationError(Exception):
    pass

def save_json(data, filepath):
    JSONSerializer.save(data, filepath)

def load_json(filepath):
    return JSONSerializer.load(filepath)

def save_pickle(data, filepath):
    PickleSerializer.save(data, filepath)

def load_pickle(filepath):
    return PickleSerializer.load(filepath)

def save_parquet(data, filepath):
    ParquetSerializer.save(data, filepath)

def load_parquet(filepath):
    return ParquetSerializer.load(filepath)

def save_data(data, filepath):
    UniversalSerializer.save(data, filepath)

def load_data(filepath):
    return UniversalSerializer.load(filepath)

# Data processing utilities - simplified for now

# M1 utilities - using available hardware optimizations
try:
    from src.utils.hardware.m1_optimizations import M1MemoryOptimizer, M1DataManager
    M1_OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    M1_OPTIMIZATIONS_AVAILABLE = False
    M1MemoryOptimizer = None
    M1DataManager = None

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
except ImportError:
    # Fallback if M1 optimizations not available
    class M1MemoryOptimizer:
        def __init__(self, *args, **kwargs):
            pass
        def optimize_memory(self):
            return {}
    
    class M1DataManager:
        def __init__(self, *args, **kwargs):
            pass

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
        tprint(f"Health Report: {health_report}")
        
        # Test operations
        current_time = common_ops.get_operation('datetime', 'get_current_datetime')()
        tprint(f"Current time: {current_time}")
        
        return health_report

if __name__ == "__main__":
    asyncio.run(test_utility_container())
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
