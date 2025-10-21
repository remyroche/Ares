"""Enhanced unified artifact and path management for reads/writes.

Provides a single place to resolve data, reports, cache, optimization, and tmp
paths based on configuration. Ensures directories exist before use.

This is a simplified wrapper around the refactored artifact manager components.
"""

from __future__ import annotations

import sys
import threading
import asyncio
import json
import pickle
import gzip
import io
import time
import uuid
import shutil
import gc
from pathlib import Path
from typing import Optional, Any, Dict, List, Tuple, Union, Callable, TypeVar, Generic, Protocol, runtime_checkable, Literal, Final, ClassVar, cast, overload
from dataclasses import dataclass, field
from contextlib import nullcontext, contextmanager, asynccontextmanager
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod

from .artifact_storage import ArtifactStorage
from .compression_manager import CompressionManager, CompressionConfig
from .cache_manager import CacheManager, CacheConfig
from .memory_manager import MemoryManager, MemoryConfig
from .path_manager import PathManager
from .logger import system_logger
from .tprint import (
    tprint, tprint_success, tprint_info, tprint_warning, tprint_error,
    tprint_debug, tprint_performance, tprint_progress, tprint_structured,
    tprint_exception, tprint_with_level, tprint_data_preview, LogLevel, TPrintConfig
)
from .common_operations import ensure_directory
from .version_manager import get_version_manager
from ..config.pipeline_modes import get_mode_config, get_mode_lookback_days
from ..data.ares_launcher_data_loader import AresLauncherDataLoader

# Type definitions for better type safety
T = TypeVar('T')
DataFrameType = TypeVar('DataFrameType', bound=Any)
ModelType = TypeVar('ModelType', bound=Any)
MetadataType = TypeVar('MetadataType', bound=Dict[str, Any])

# Protocol definitions for better type checking
@runtime_checkable
class DataProcessor(Protocol):
    """Protocol for data processing objects."""
    def process(self, data: Any) -> Any: ...
    def validate(self, data: Any) -> bool: ...

@runtime_checkable
class Cacheable(Protocol):
    """Protocol for cacheable objects."""
    def get_cache_key(self) -> str: ...
    def get_size_bytes(self) -> int: ...

# Import hardware optimization tools
try:
    from .hardware import (
        get_integrated_hardware_manager, memory_optimized, 
        performance_tracked, force_cleanup, get_memory_stats,
        optimize_dataframe, optimize_array, cache_result,
        MemoryOptimizationLevel
    )
    HARDWARE_OPTIMIZATION_AVAILABLE: Final[bool] = True
except ImportError:
    HARDWARE_OPTIMIZATION_AVAILABLE: Final[bool] = False
    # Create dummy functions and classes for compatibility
    class MemoryOptimizationLevel:
        AGGRESSIVE: Final[str] = "AGGRESSIVE"
        BALANCED: Final[str] = "BALANCED"
        CONSERVATIVE: Final[str] = "CONSERVATIVE"
    
    def get_integrated_hardware_manager() -> Optional[Any]: return None
    def memory_optimized(*args: Any, **kwargs: Any) -> Callable[[T], T]: return lambda f: f
    def performance_tracked(*args: Any, **kwargs: Any) -> Callable[[T], T]: return lambda f: f
    def force_cleanup() -> None:
        """Force garbage collection and memory cleanup."""
        import gc
        gc.collect()
        try:
            # Try to import and use hardware-specific cleanup if available
            from src.utils.hardware import force_cleanup as hw_force_cleanup
            hw_force_cleanup()
        except ImportError:
            pass
    def get_memory_stats() -> Dict[str, Any]: return {}
    def optimize_dataframe(df: Any) -> Any: return df
    def optimize_array(arr: Any) -> Any: return arr
    def cache_result(*args: Any, **kwargs: Any) -> Callable[[T], T]: return lambda f: f

# Import optional dependencies
try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE: Final[bool] = True
    NUMPY_AVAILABLE: Final[bool] = True
except ImportError:
    PANDAS_AVAILABLE: Final[bool] = False
    NUMPY_AVAILABLE: Final[bool] = False

try:
    import lz4.frame
    LZ4_AVAILABLE: Final[bool] = True
except ImportError:
    LZ4_AVAILABLE: Final[bool] = False

try:
    import psutil
    PSUTIL_AVAILABLE: Final[bool] = True
except ImportError:
    PSUTIL_AVAILABLE: Final[bool] = False



class CompressionType(Enum):
    """Supported compression algorithms."""
    NONE: Final[str] = "none"
    GZIP: Final[str] = "gzip"
    LZ4: Final[str] = "lz4"
    AUTO: Final[str] = "auto"  # Automatically choose best compression


class OperationType(Enum):
    """Types of artifact operations."""
    SAVE: Final[str] = "save"
    LOAD: Final[str] = "load"
    DELETE: Final[str] = "delete"
    LIST: Final[str] = "list"


class RetryStrategy(Enum):
    """Retry strategies for failed operations."""
    EXPONENTIAL_BACKOFF: Final[str] = "exponential_backoff"
    LINEAR_BACKOFF: Final[str] = "linear_backoff"
    FIXED_DELAY: Final[str] = "fixed_delay"


@dataclass
class RetryConfig:
    """Configuration for retry mechanisms."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    retryable_exceptions: Tuple[type, ...] = (OSError, IOError, ConnectionError)
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.base_delay <= 0:
            raise ValueError("base_delay must be positive")
        if self.max_delay <= 0:
            raise ValueError("max_delay must be positive")
        if self.base_delay > self.max_delay:
            raise ValueError("base_delay must be less than or equal to max_delay")




@dataclass
class MemoryConfig:
    """Configuration for memory management."""
    max_memory_mb: float = 2000.0
    cache_memory_mb: float = 500.0
    spill_threshold_mb: float = 150.0
    cleanup_interval_seconds: float = 300.0
    enable_gc_collection: bool = True
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be positive")
        if self.cache_memory_mb <= 0:
            raise ValueError("cache_memory_mb must be positive")
        if self.spill_threshold_mb <= 0:
            raise ValueError("spill_threshold_mb must be positive")
        if self.cleanup_interval_seconds <= 0:
            raise ValueError("cleanup_interval_seconds must be positive")


@dataclass
class ArtifactMetadata:
    """Enhanced metadata for artifacts."""
    artifact_key: str
    step_name: str
    artifact_type: str
    size_bytes: int
    compressed_size_bytes: Optional[int] = None
    checksum: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    modified_at: datetime = field(default_factory=datetime.utcnow)
    compression_used: CompressionType = CompressionType.NONE
    storage_location: str = "memory"
    parent_artifacts: List[str] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    description: str = ""
    version: str = "1.0"
    
    def __post_init__(self) -> None:
        """Validate metadata after initialization."""
        if not self.artifact_key:
            raise ValueError("artifact_key cannot be empty")
        if not self.step_name:
            raise ValueError("step_name cannot be empty")
        if not self.artifact_type:
            raise ValueError("artifact_type cannot be empty")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")
        if self.compressed_size_bytes is not None and self.compressed_size_bytes < 0:
            raise ValueError("compressed_size_bytes must be non-negative")
    
    def get_compression_ratio(self) -> float:
        """Get compression ratio if compressed size is available."""
        if self.compressed_size_bytes is None or self.size_bytes == 0:
            return 1.0
        return self.compressed_size_bytes / self.size_bytes
    
    def is_compressed(self) -> bool:
        """Check if artifact is compressed."""
        return self.compression_used != CompressionType.NONE
    
    def get_storage_efficiency(self) -> float:
        """Get storage efficiency (1.0 = no compression, <1.0 = compressed)."""
        return self.get_compression_ratio()


@dataclass
class OperationMetrics:
    """Metrics for artifact operations."""
    operation_type: OperationType
    artifact_key: str
    step_name: str
    duration_seconds: float
    success: bool
    error_message: Optional[str] = None
    retry_count: int = 0
    bytes_processed: int = 0
    compression_ratio: float = 1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self) -> None:
        """Validate metrics after initialization."""
        if not self.artifact_key:
            raise ValueError("artifact_key cannot be empty")
        if not self.step_name:
            raise ValueError("step_name cannot be empty")
        if self.duration_seconds < 0:
            raise ValueError("duration_seconds must be non-negative")
        if self.retry_count < 0:
            raise ValueError("retry_count must be non-negative")
        if self.bytes_processed < 0:
            raise ValueError("bytes_processed must be non-negative")
        if self.compression_ratio <= 0:
            raise ValueError("compression_ratio must be positive")
    
    def get_throughput_mbps(self) -> float:
        """Get throughput in MB/s."""
        if self.duration_seconds == 0:
            return 0.0
        return (self.bytes_processed / (1024 * 1024)) / self.duration_seconds
    
    def get_efficiency_score(self) -> float:
        """Get operation efficiency score (0.0 to 1.0)."""
        if not self.success:
            return 0.0
        
        # Base score from success
        score = 1.0
        
        # Penalize for retries
        if self.retry_count > 0:
            score *= (1.0 - (self.retry_count * 0.1))
        
        # Reward for compression
        if self.compression_ratio < 1.0:
            score *= (1.0 + (1.0 - self.compression_ratio))
        
        return max(0.0, min(1.0, score))


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    artifact_key: str
    data: Any
    metadata: ArtifactMetadata
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    memory_size_mb: float = 0.0
    
    def __post_init__(self) -> None:
        """Validate cache entry after initialization."""
        if not self.artifact_key:
            raise ValueError("artifact_key cannot be empty")
        if self.access_count < 0:
            raise ValueError("access_count must be non-negative")
        if self.memory_size_mb < 0:
            raise ValueError("memory_size_mb must be non-negative")
    
    def update_access(self) -> None:
        """Update access statistics."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1
    
    def get_access_frequency(self) -> float:
        """Get access frequency (accesses per hour)."""
        if self.access_count == 0:
            return 0.0
        
        time_diff = datetime.utcnow() - self.last_accessed
        hours_elapsed = time_diff.total_seconds() / 3600
        
        if hours_elapsed == 0:
            return float('inf')
        
        return self.access_count / hours_elapsed
    
    def is_stale(self, max_age_hours: float = 24.0) -> bool:
        """Check if cache entry is stale."""
        time_diff = datetime.utcnow() - self.last_accessed
        hours_elapsed = time_diff.total_seconds() / 3600
        return hours_elapsed > max_age_hours


# Step category mapping for organized artifact storage
STEP_CATEGORIES: Final[Dict[str, List[str]]] = {
    'data_collection': ['step01', 'data_downloader', 'klines_downloading_processing'],
    'market_analysis': ['step02', 'market_analysis', 'sr_detection', 'regime_discovery'],
    'pre_training': ['step02_5', 'feature_generation', 'pre_training'],
    'models_training': ['step03', 'model_training', 'analyst_models', 'tactician_models'],
    'backtesting': ['step04', 'backtesting', 'real_parameters_optimization']
}


def get_step_category(step_name: str) -> str:
    """Determine the category for a step based on its name."""
    tprint_debug(f"🔍 Determining step category for: {step_name}")
    
    if not step_name or not isinstance(step_name, str):
        tprint_warning(f"⚠️ Invalid step name provided: {step_name}")
        return 'pre_training'  # Default fallback
    
    step_name_lower = step_name.lower()
    for category, patterns in STEP_CATEGORIES.items():
        if any(pattern.lower() in step_name_lower for pattern in patterns):
            tprint_info(f"✅ Step '{step_name}' categorized as '{category}'")
            return category
    
    tprint_warning(f"⚠️ No category found for step '{step_name}', using default: pre_training")
    return 'pre_training'  # Default fallback


def _format_data_preview(data: Any, artifact_name: str) -> str:
    """Format a data preview for tprint output with enhanced type safety."""
    tprint_debug(f"📊 Formatting data preview for artifact: {artifact_name}")
    
    try:
        # Try to import pandas for DataFrame handling
        import pandas as pd
        import numpy as np
        
        if isinstance(data, pd.DataFrame):
            rows, cols = data.shape
            file_size_mb = data.memory_usage(deep=True).sum() / (1024 * 1024)
            
            # Get first 10 columns and 5 rows
            preview_cols = data.columns[:10].tolist()
            preview_data = data.iloc[:5, :10]
            
            preview_str = f"DataFrame: {rows:,} rows × {cols:,} cols | {file_size_mb:.2f}MB\n"
            preview_str += f"Columns: {', '.join(preview_cols[:5])}{'...' if len(preview_cols) > 5 else ''}\n"
            preview_str += f"Preview (5×10):\n{preview_data.to_string(max_cols=10, max_rows=5)}"
            
            tprint_debug(f"✅ DataFrame preview generated: {rows} rows, {cols} cols, {file_size_mb:.2f}MB")
            return preview_str
            
        elif isinstance(data, np.ndarray):
            shape = data.shape
            file_size_mb = data.nbytes / (1024 * 1024)
            
            preview_str = f"NumPy Array: {shape} | {file_size_mb:.2f}MB\n"
            if len(shape) == 2:
                preview_str += f"Preview (5×10):\n{data[:5, :10]}"
            else:
                preview_str += f"Preview: {data.flat[:10]}..."
            
            tprint_debug(f"✅ NumPy array preview generated: {shape}, {file_size_mb:.2f}MB")
            return preview_str
            
        elif isinstance(data, (list, tuple)):
            length = len(data)
            file_size_mb = sum(sys.getsizeof(item) for item in data[:100]) / (1024 * 1024)  # Estimate
            
            preview_str = f"List/Tuple: {length:,} items | ~{file_size_mb:.2f}MB\n"
            preview_str += f"Preview: {data[:5]}{'...' if length > 5 else ''}"
            
            tprint_debug(f"✅ List/Tuple preview generated: {length} items, ~{file_size_mb:.2f}MB")
            return preview_str
            
        elif isinstance(data, dict):
            length = len(data)
            file_size_mb = sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in list(data.items())[:50]) / (1024 * 1024)  # Estimate
            
            preview_str = f"Dict: {length:,} keys | ~{file_size_mb:.2f}MB\n"
            preview_str += f"Keys: {list(data.keys())[:5]}{'...' if length > 5 else ''}"
            
            tprint_debug(f"✅ Dict preview generated: {length} keys, ~{file_size_mb:.2f}MB")
            return preview_str
            
        else:
            file_size_mb = sys.getsizeof(data) / (1024 * 1024)
            data_type = type(data).__name__
            tprint_debug(f"✅ Generic object preview generated: {data_type}, {file_size_mb:.2f}MB")
            return f"{data_type}: {file_size_mb:.2f}MB"
            
    except Exception as e:
        error_msg = f"Preview unavailable: {str(e)[:50]}..."
        tprint_error(f"❌ Failed to format data preview for {artifact_name}: {e}")
        return error_msg


class ArtifactManager:
    """Simplified artifact manager that uses refactored components with comprehensive type safety."""
    
    # Class variables for type hints
    _storage: ArtifactStorage
    _path_manager: PathManager
    _compression: Optional[CompressionManager]
    _cache: Optional[CacheManager]
    _memory: Optional[MemoryManager]
    _lock: Optional[threading.RLock]
    _async_lock: Optional[asyncio.Lock]
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the artifact manager with comprehensive type safety and logging.
        
        Args:
            config: Configuration dictionary with type-safe validation
        
        Raises:
            ValueError: If configuration is invalid
            TypeError: If configuration types are incorrect
        """
        tprint_info("🚀 Initializing ArtifactManager with enhanced type safety")
        
        # Validate config type
        if not isinstance(config, dict):
            raise TypeError(f"config must be a dict, got {type(config).__name__}")
        
        self.logger = system_logger.getChild("ArtifactManager")
        
        # Initialize base directory with validation
        data_dir = config.get("paths", {}).get("data_dir", "data")
        if not isinstance(data_dir, str):
            raise TypeError(f"data_dir must be a string, got {type(data_dir).__name__}")
        
        self.base_dir = Path(data_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        tprint_success(f"✅ Base directory initialized: {self.base_dir}")
        
        # Initialize components with error handling
        try:
            self._storage = ArtifactStorage(self.base_dir)
            self._path_manager = PathManager(self.base_dir)
            tprint_success("✅ Core storage components initialized")
        except Exception as e:
            tprint_error(f"❌ Failed to initialize core components: {e}")
            raise
        
        # Initialize optional components with validation
        if config.get("enable_compression", True):
            try:
                compression_config = CompressionConfig()
                self._compression = CompressionManager(compression_config)
                tprint_success("✅ Compression manager initialized")
            except Exception as e:
                tprint_warning(f"⚠️ Compression manager initialization failed: {e}")
                self._compression = None
        else:
            self._compression = None
            tprint_info("ℹ️ Compression disabled")
        
        if config.get("enable_caching", True):
            try:
                cache_config = CacheConfig(
                    max_size_mb=config.get("max_cache_size_mb", 512.0),
                    enable_thread_safety=config.get("enable_thread_safety", True)
                )
                self._cache = CacheManager(cache_config)
                tprint_success("✅ Cache manager initialized")
            except Exception as e:
                tprint_warning(f"⚠️ Cache manager initialization failed: {e}")
                self._cache = None
        else:
            self._cache = None
            tprint_info("ℹ️ Caching disabled")
        
        if config.get("enable_memory_optimization", True):
            try:
                memory_config = MemoryConfig(
                    max_memory_mb=config.get("max_memory_mb", 2000.0),
                    spill_threshold_mb=config.get("spill_threshold_mb", 150.0)
                )
                spill_dir = self.base_dir / "spilled"
                self._memory = MemoryManager(memory_config, spill_dir)
                tprint_success("✅ Memory manager initialized")
            except Exception as e:
                tprint_warning(f"⚠️ Memory manager initialization failed: {e}")
                self._memory = None
        else:
            self._memory = None
            tprint_info("ℹ️ Memory optimization disabled")
        
        # Thread safety with validation
        if config.get("enable_thread_safety", True):
            try:
                import threading
                import asyncio
                self._lock = threading.RLock()
                self._async_lock = asyncio.Lock()
                tprint_success("✅ Thread safety enabled")
            except Exception as e:
                tprint_warning(f"⚠️ Thread safety initialization failed: {e}")
                self._lock = None
                self._async_lock = None
        else:
            self._lock = None
            self._async_lock = None
            tprint_info("ℹ️ Thread safety disabled")
        
        # Store original config for compatibility
        self.config: Dict[str, Any] = config
        
        # Add context attributes for BaseStep compatibility
        self._current_model: str = ""
        self._current_direction: str = ""
        self._current_symbol: Optional[str] = None
        self._current_exchange: Optional[str] = None
        self._current_information: Optional[str] = None
        self._current_execution_mode: str = "light"  # Default to light mode
        
        self._artifacts_dir = self.base_dir / "artifacts"
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)
        tprint_success(f"✅ Artifacts directory initialized: {self._artifacts_dir}")
        
        # Initialize data loader for mode-aware data fetching
        self._data_loader: Optional[AresLauncherDataLoader] = None
        
        # Initialize performance metrics with type safety
        self._performance_metrics: Dict[str, Union[int, float]] = {
            'cache_hits': 0,
            'cache_misses': 0,
            'compression_savings_mb': 0.0,
            'optimization_savings_mb': 0.0,
            'spill_operations': 0,
            'lazy_loads': 0
        }
        
        # Initialize memory profiles for enhanced storage
        self._memory_profiles: Dict[str, Dict[str, Any]] = {}
        self._total_memory_mb: float = 0.0
        
        tprint_success("🎉 ArtifactManager initialization completed successfully")
    
    def _lock_context(self) -> Union[threading.RLock, nullcontext]:
        """Get lock context manager with type safety."""
        if self._lock is not None:
            return self._lock
        return nullcontext()
    
    async def _async_lock_context(self) -> Union[asyncio.Lock, nullcontext]:
        """Get async lock context manager with type safety."""
        if self._async_lock is not None:
            return self._async_lock
        return nullcontext()
    
    def set_context(self, step_name: str, symbol: Optional[str] = None, 
                   exchange: Optional[str] = None, datetime_param: Optional[Any] = None, 
                   information: Optional[str] = None, direction: str = "long", 
                   model: str = "Analyst", execution_mode: str = "light") -> None:
        """Set the current context for path generation with comprehensive validation.
        
        Args:
            step_name: Name of the current step
            symbol: Trading symbol (optional)
            exchange: Exchange name (optional)
            datetime_param: Datetime parameter (optional)
            information: Information type (optional)
            direction: Trading direction (default: "long")
            model: Model type (default: "Analyst")
            execution_mode: Execution mode for data fetching (default: "light")
            
        Raises:
            ValueError: If required parameters are invalid
            TypeError: If parameter types are incorrect
        """
        # Validate input parameters
        if not isinstance(step_name, str) or not step_name.strip():
            raise ValueError(f"step_name must be a non-empty string, got: {step_name}")
        if symbol is not None and not isinstance(symbol, str):
            raise TypeError(f"symbol must be a string or None, got: {type(symbol).__name__}")
        if exchange is not None and not isinstance(exchange, str):
            raise TypeError(f"exchange must be a string or None, got: {type(exchange).__name__}")
        if not isinstance(direction, str):
            raise TypeError(f"direction must be a string, got: {type(direction).__name__}")
        if not isinstance(model, str):
            raise TypeError(f"model must be a string, got: {type(model).__name__}")
        if not isinstance(execution_mode, str):
            raise TypeError(f"execution_mode must be a string, got: {type(execution_mode).__name__}")
        
        tprint_info(f"📁 SETTING CONTEXT: {step_name} | {symbol} | {exchange} | {direction} | {model} | {execution_mode}")
        
        with self._lock_context():
            try:
                # Store context attributes for BaseStep compatibility
                self._current_symbol = symbol
                self._current_exchange = exchange
                self._current_direction = direction
                self._current_model = model
                self._current_information = information
                self._current_execution_mode = execution_mode
                
                self._path_manager.set_context(
                    step_name=step_name,
                    symbol=symbol,
                    exchange=exchange,
                    datetime_param=datetime_param,
                    information=information,
                    direction=direction,
                    model=model
                )
                
                tprint_success(f"✅ Context set successfully for step: {step_name}")
                
            except Exception as e:
                tprint_error(f"❌ Failed to set context: {e}")
                raise
    
    def save(self, data: Any, artifact_name: str, 
             artifact_type: str = "data", 
             compression: str = "auto",
             metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save an artifact with comprehensive validation and error handling.
        
        Args:
            data: Data to save (any type)
            artifact_name: Name for the artifact (must be non-empty string)
            artifact_type: Type of artifact (default: "data")
            compression: Compression method (default: "auto")
            metadata: Optional metadata dictionary
            
        Returns:
            Path where artifact was saved
            
        Raises:
            ValueError: If artifact_name is empty or invalid
            TypeError: If parameter types are incorrect
            Exception: If save operation fails
        """
        # Validate input parameters
        if not isinstance(artifact_name, str) or not artifact_name.strip():
            raise ValueError(f"artifact_name must be a non-empty string, got: {artifact_name}")
        if not isinstance(artifact_type, str):
            raise TypeError(f"artifact_type must be a string, got: {type(artifact_type).__name__}")
        if not isinstance(compression, str):
            raise TypeError(f"compression must be a string, got: {type(compression).__name__}")
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError(f"metadata must be a dict or None, got: {type(metadata).__name__}")
        
        with self._lock_context():
            try:
                # Print data preview before saving
                preview = _format_data_preview(data, artifact_name)
                tprint_info(f"💾 SAVING ARTIFACT: {artifact_name}")
                tprint_info(f"📊 Data Preview:\n{preview}")
                
                # Add enhanced data preview
                tprint_data_preview(data, f"saving_artifact_{artifact_name}", level=LogLevel.INFO)
                
                # Get current step name from path manager
                step_name = self._path_manager._current_step_name or "unknown"
                
                # Generate path
                file_path = self._path_manager.get_artifact_path(
                    step_name=step_name,
                    key=artifact_name,
                    file_extension="parquet"
                )
                
                # Optimize data if memory manager is available
                if self._memory and hasattr(data, 'memory_usage'):  # DataFrame
                    tprint_debug(f"🔧 Optimizing DataFrame for artifact: {artifact_name}")
                    data = self._memory.optimize_dataframe(data)
                
                # Save artifact
                success = self._storage.save_artifact(
                    data=data,
                    file_path=file_path,
                    artifact_type=artifact_type,
                    metadata=metadata
                )
                
                if not success:
                    raise Exception(f"Failed to save artifact {artifact_name}")
                
                # Cache if enabled
                if self._cache:
                    tprint_debug(f"💾 Caching artifact: {artifact_name}")
                    self._cache.put(artifact_name, data)
                
                # Profile memory usage if memory manager is available
                if self._memory:
                    tprint_debug(f"📊 Profiling memory usage for artifact: {artifact_name}")
                    self._memory.profile_memory_usage(artifact_name, data)
                
                # Print success message
                tprint_success(f"✅ ARTIFACT SAVED: {artifact_name} → {file_path}")
                
                return str(file_path)
                
            except Exception as e:
                tprint_error(f"❌ FAILED TO SAVE ARTIFACT: {artifact_name} - {str(e)}")
                raise
    
    def get_artifact(self, artifact_name: str, 
                    artifact_type: str = "data") -> Optional[Any]:
        """Retrieve an artifact with comprehensive validation and error handling.
        
        Args:
            artifact_name: Name of the artifact to retrieve
            artifact_type: Type of artifact to retrieve (default: "data")
            
        Returns:
            Retrieved data or None if not found
            
        Raises:
            ValueError: If artifact_name is empty or invalid
            TypeError: If parameter types are incorrect
        """
        # Validate input parameters
        if not isinstance(artifact_name, str) or not artifact_name.strip():
            raise ValueError(f"artifact_name must be a non-empty string, got: {artifact_name}")
        if not isinstance(artifact_type, str):
            raise TypeError(f"artifact_type must be a string, got: {type(artifact_type).__name__}")
        
        with self._lock_context():
            try:
                tprint_info(f"🔍 LOADING ARTIFACT: {artifact_name}")
                
                # Check cache first
                if self._cache:
                    cached_data = self._cache.get(artifact_name)
                    if cached_data is not None:
                        tprint_success(f"✅ ARTIFACT LOADED FROM CACHE: {artifact_name}")
                        preview = _format_data_preview(cached_data, artifact_name)
                        tprint_info(f"📊 Data Preview:\n{preview}")
                        return cached_data
                
                # Get current step name from path manager
                step_name = self._path_manager._current_step_name or "unknown"
                
                # Find artifact file
                file_path = self._path_manager.find_artifact(
                    step_name=step_name,
                    key=artifact_name,
                    artifact_type=artifact_type
                )
                
                if file_path is None:
                    tprint_warning(f"⚠️  ARTIFACT NOT FOUND: {artifact_name}")
                    return None
                
                # Load artifact
                data = self._storage.load_artifact(file_path)
                
                # Add enhanced data preview
                if data is not None:
                    tprint_data_preview(data, f"loaded_artifact_{artifact_name}", level=LogLevel.INFO)
                
                if data is not None:
                    # Cache if enabled
                    if self._cache:
                        tprint_debug(f"💾 Caching loaded artifact: {artifact_name}")
                        self._cache.put(artifact_name, data)
                    
                    # Profile memory usage if memory manager is available
                    if self._memory:
                        tprint_debug(f"📊 Profiling memory usage for loaded artifact: {artifact_name}")
                        self._memory.profile_memory_usage(artifact_name, data)
                    
                    # Print data preview after loading
                    preview = _format_data_preview(data, artifact_name)
                    tprint_success(f"✅ ARTIFACT LOADED: {artifact_name}")
                    tprint_info(f"📊 Data Preview:\n{preview}")
                else:
                    tprint_warning(f"⚠️  FAILED TO LOAD ARTIFACT DATA: {artifact_name}")
                
                return data
                
            except Exception as e:
                tprint_error(f"❌ FAILED TO LOAD ARTIFACT: {artifact_name} - {str(e)}")
                return None
    
    def delete_artifact(self, artifact_name: str, artifact_type: str = "data") -> bool:
        """Delete an artifact with comprehensive validation and error handling.
        
        Args:
            artifact_name: Name of the artifact to delete
            artifact_type: Type of artifact to delete (default: "data")
            
        Returns:
            True if deletion was successful, False otherwise
            
        Raises:
            ValueError: If artifact_name is empty or invalid
            TypeError: If parameter types are incorrect
        """
        # Validate input parameters
        if not isinstance(artifact_name, str) or not artifact_name.strip():
            raise ValueError(f"artifact_name must be a non-empty string, got: {artifact_name}")
        if not isinstance(artifact_type, str):
            raise TypeError(f"artifact_type must be a string, got: {type(artifact_type).__name__}")
        
        with self._lock_context():
            try:
                tprint_info(f"🗑️  DELETING ARTIFACT: {artifact_name}")
                
                # Get current step name from path manager
                step_name = self._path_manager._current_step_name or "unknown"
                
                # Find artifact file
                file_path = self._path_manager.find_artifact(
                    step_name=step_name,
                    key=artifact_name,
                    artifact_type=artifact_type
                )
                
                if file_path is None:
                    tprint_warning(f"⚠️  ARTIFACT NOT FOUND FOR DELETION: {artifact_name}")
                    return False
                
                # Delete from storage
                success = self._storage.delete_artifact(file_path)
                
                # Remove from cache if enabled
                if self._cache:
                    tprint_debug(f"🗑️ Removing from cache: {artifact_name}")
                    self._cache.remove(artifact_name)
                
                # Remove from memory profiles if memory manager is available
                if self._memory and artifact_name in self._memory._memory_profiles:
                    tprint_debug(f"📊 Removing from memory profiles: {artifact_name}")
                    profile = self._memory._memory_profiles.pop(artifact_name)
                    self._memory._total_memory_mb -= profile.memory_usage_mb
                
                if success:
                    tprint_success(f"✅ ARTIFACT DELETED: {artifact_name}")
                else:
                    tprint_warning(f"⚠️  FAILED TO DELETE ARTIFACT: {artifact_name}")
                
                return success
                
            except Exception as e:
                tprint_error(f"❌ FAILED TO DELETE ARTIFACT: {artifact_name} - {str(e)}")
                return False
    
    def list_artifacts(self, pattern: str = "*") -> List[Path]:
        """List artifacts matching a pattern with validation.
        
        Args:
            pattern: Glob pattern to match artifacts (default: "*")
            
        Returns:
            List of Path objects matching the pattern
            
        Raises:
            TypeError: If pattern is not a string
        """
        if not isinstance(pattern, str):
            raise TypeError(f"pattern must be a string, got: {type(pattern).__name__}")
        
        tprint_debug(f"🔍 Listing artifacts with pattern: {pattern}")
        
        try:
            artifacts = self._storage.list_artifacts(pattern)
            tprint_info(f"📋 Found {len(artifacts)} artifacts matching pattern: {pattern}")
            return artifacts
        except Exception as e:
            tprint_error(f"❌ Failed to list artifacts with pattern '{pattern}': {e}")
            return []
    
    def clear_cache(self) -> None:
        """Clear the cache with comprehensive logging."""
        tprint_info("🧹 CLEARING CACHE")
        
        try:
            if self._cache:
                cache_stats_before = self._cache.get_stats() if hasattr(self._cache, 'get_stats') else {}
                self._cache.clear()
                tprint_success(f"✅ CACHE CLEARED (stats before: {cache_stats_before})")
            else:
                tprint_info("ℹ️ No cache to clear")
        except Exception as e:
            tprint_error(f"❌ Failed to clear cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics with enhanced type safety."""
        tprint_debug("📊 Collecting comprehensive statistics")
        
        try:
            stats: Dict[str, Any] = {
                "config": {
                    "base_dir": str(self.base_dir),
                    "enable_compression": self._compression is not None,
                    "enable_caching": self._cache is not None,
                    "enable_memory_optimization": self._memory is not None,
                    "enable_thread_safety": self._lock is not None
                },
                "performance_metrics": self._performance_metrics.copy(),
                "memory_profiles": {
                    "total_artifacts": len(self._memory_profiles),
                    "total_memory_mb": self._total_memory_mb
                }
            }
            
            # Add cache stats
            if self._cache:
                try:
                    cache_stats = self._cache.get_stats()
                    stats["cache"] = cache_stats
                    tprint_debug(f"📊 Cache stats: {cache_stats}")
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to get cache stats: {e}")
                    stats["cache"] = {"error": str(e)}
            
            # Add memory stats
            if self._memory:
                try:
                    memory_stats = self._memory.get_memory_stats()
                    stats["memory"] = memory_stats
                    tprint_debug(f"📊 Memory stats: {memory_stats}")
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to get memory stats: {e}")
                    stats["memory"] = {"error": str(e)}
            
            # Add compression stats
            if self._compression:
                try:
                    compression_stats = self._compression.get_compression_stats()
                    stats["compression"] = compression_stats
                    tprint_debug(f"📊 Compression stats: {compression_stats}")
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to get compression stats: {e}")
                    stats["compression"] = {"error": str(e)}
            
            tprint_success("✅ Statistics collected successfully")
            return stats
            
        except Exception as e:
            tprint_error(f"❌ Failed to collect statistics: {e}")
            return {"error": str(e)}
    
    def cleanup(self) -> None:
        """Perform cleanup operations with comprehensive error handling."""
        tprint_info("🧹 PERFORMING CLEANUP")
        
        try:
            # Cleanup cache
            if self._cache:
                tprint_debug("🧹 Cleaning up cache")
                self._cache.periodic_cleanup()
            
            # Cleanup memory
            if self._memory:
                tprint_debug("🧹 Cleaning up memory")
                self._memory.periodic_cleanup()
            
            # Force garbage collection
            tprint_debug("🧹 Forcing garbage collection")
            force_cleanup()
            
            tprint_success("✅ CLEANUP COMPLETED")
            
        except Exception as e:
            tprint_error(f"❌ Cleanup failed: {e}")
    
    async def run_context(self, run_id: str) -> Any:
        """Async context manager for automatic cleanup with validation.
        
        Args:
            run_id: Unique identifier for the run
            
        Yields:
            Path to the run directory
            
        Raises:
            ValueError: If run_id is empty or invalid
            TypeError: If run_id is not a string
        """
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError(f"run_id must be a non-empty string, got: {run_id}")
        
        tprint_info(f"🚀 Starting run context: {run_id}")
        
        async with await self._async_lock_context():
            run_dir = self.base_dir / f"run_{run_id}"
            run_dir.mkdir(parents=True, exist_ok=True)
            tprint_success(f"✅ Run directory created: {run_dir}")
            
            try:
                yield run_dir
            finally:
                # Auto-cleanup run directory
                try:
                    import shutil
                    shutil.rmtree(run_dir, ignore_errors=True)
                    tprint_info(f"🧹 Cleaned up run directory: {run_dir}")
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to cleanup run directory {run_dir}: {e}")
    
    # Compatibility methods for existing code with enhanced type safety
    def get_data_dir(self, *subdirs: str) -> Path:
        """Get data directory path with validation.
        
        Args:
            *subdirs: Subdirectory names to append
            
        Returns:
            Path to the data directory
        """
        tprint_debug(f"📁 Getting data directory with subdirs: {subdirs}")
        return self.base_dir / "data" / Path(*subdirs)
    
    def get_reports_dir(self, *subdirs: str) -> Path:
        """Get reports directory path with validation.
        
        Args:
            *subdirs: Subdirectory names to append
            
        Returns:
            Path to the reports directory
        """
        tprint_debug(f"📁 Getting reports directory with subdirs: {subdirs}")
        return self.base_dir / "reports" / Path(*subdirs)
    
    def get_cache_dir(self, *subdirs: str) -> Path:
        """Get cache directory path with validation.
        
        Args:
            *subdirs: Subdirectory names to append
            
        Returns:
            Path to the cache directory
        """
        tprint_debug(f"📁 Getting cache directory with subdirs: {subdirs}")
        return self.base_dir / "cache" / Path(*subdirs)
    
    def get_optimization_dir(self, *subdirs: str) -> Path:
        """Get optimization directory path with validation.
        
        Args:
            *subdirs: Subdirectory names to append
            
        Returns:
            Path to the optimization directory
        """
        tprint_debug(f"📁 Getting optimization directory with subdirs: {subdirs}")
        return self.base_dir / "optimization" / Path(*subdirs)
    
    def get_tmp_dir(self, *subdirs: str) -> Path:
        """Get temporary directory path with validation.
        
        Args:
            *subdirs: Subdirectory names to append
            
        Returns:
            Path to the temporary directory
        """
        tprint_debug(f"📁 Getting tmp directory with subdirs: {subdirs}")
        return self.base_dir / "tmp" / Path(*subdirs)
    
    def get_tmp_path(self, filename: str) -> Path:
        """Get temporary file path with validation.
        
        Args:
            filename: Name of the temporary file
            
        Returns:
            Path to the temporary file
            
        Raises:
            ValueError: If filename is empty or invalid
            TypeError: If filename is not a string
        """
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError(f"filename must be a non-empty string, got: {filename}")
        
        tprint_debug(f"📁 Getting tmp path for filename: {filename}")
        return self.get_tmp_dir() / filename
    
    def reset_run(self) -> None:
        """Reset run state (compatibility method) with logging."""
        tprint_info("🔄 Resetting run state")
        # The refactored manager handles this automatically
        tprint_success("✅ Run state reset completed")
    
    def get_run_id(self) -> Optional[str]:
        """Get current run ID (compatibility method)."""
        tprint_debug("🔍 Getting run ID (compatibility method)")
        return None
    
    def get_run_dir(self) -> Optional[Path]:
        """Get current run directory (compatibility method)."""
        tprint_debug("🔍 Getting run directory (compatibility method)")
        return self.base_dir
    
    def get_step_category(self, step_name: str) -> str:
        """Get step category for a given step name with validation.
        
        Args:
            step_name: Name of the step
            
        Returns:
            Category name for the step
            
        Raises:
            ValueError: If step_name is empty or invalid
            TypeError: If step_name is not a string
        """
        if not isinstance(step_name, str) or not step_name.strip():
            raise ValueError(f"step_name must be a non-empty string, got: {step_name}")
        
        tprint_debug(f"🔍 Getting step category for: {step_name}")
        return get_step_category(step_name)
    
    def ensure_step_category_directories(self) -> None:
        """Ensure all step category directories exist with logging."""
        tprint_info("📁 Ensuring step category directories exist")
        
        try:
            for category in STEP_CATEGORIES.keys():
                category_dir = self._artifacts_dir / category
                category_dir.mkdir(parents=True, exist_ok=True)
                tprint_debug(f"✅ Ensured directory exists: {category_dir}")
            
            tprint_success("✅ All step category directories ensured")
        except Exception as e:
            tprint_error(f"❌ Failed to ensure step category directories: {e}")
    
    def _get_enhanced_path(self, step_name: str, artifact_name: str, file_extension: str) -> Path:
        """Get enhanced path for artifact with step category organization and validation.
        
        Args:
            step_name: Name of the step
            artifact_name: Name of the artifact
            file_extension: File extension (without dot)
            
        Returns:
            Path to the enhanced artifact file
            
        Raises:
            ValueError: If any parameter is empty or invalid
            TypeError: If parameter types are incorrect
        """
        # Validate input parameters
        if not isinstance(step_name, str) or not step_name.strip():
            raise ValueError(f"step_name must be a non-empty string, got: {step_name}")
        if not isinstance(artifact_name, str) or not artifact_name.strip():
            raise ValueError(f"artifact_name must be a non-empty string, got: {artifact_name}")
        if not isinstance(file_extension, str) or not file_extension.strip():
            raise ValueError(f"file_extension must be a non-empty string, got: {file_extension}")
        
        tprint_debug(f"🔧 Getting enhanced path for: {step_name}/{artifact_name}.{file_extension}")
        
        step_category = get_step_category(step_name)
        category_dir = self._artifacts_dir / step_category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate enhanced filename with context
        context_parts = []
        if hasattr(self, '_current_symbol') and self._current_symbol:
            context_parts.append(self._current_symbol)
        if hasattr(self, '_current_exchange') and self._current_exchange:
            context_parts.append(self._current_exchange)
        if hasattr(self, '_current_direction') and self._current_direction:
            context_parts.append(self._current_direction)
        if hasattr(self, '_current_model') and self._current_model:
            context_parts.append(self._current_model)
        
        if context_parts:
            context_str = "_".join(context_parts)
            filename = f"{artifact_name}_{context_str}.{file_extension}"
        else:
            filename = f"{artifact_name}.{file_extension}"
        
        enhanced_path = category_dir / filename
        tprint_debug(f"✅ Enhanced path generated: {enhanced_path}")
        return enhanced_path
    
    def _load_artifact_from_path(self, path: Path) -> Any:
        """Load artifact from file path with comprehensive error handling.
        
        Args:
            path: Path to the artifact file
            
        Returns:
            Loaded data or None if loading failed
            
        Raises:
            TypeError: If path is not a Path object
        """
        if not isinstance(path, Path):
            raise TypeError(f"path must be a Path object, got: {type(path).__name__}")
        
        tprint_debug(f"📂 Loading artifact from path: {path}")
        
        try:
            if path.suffix == '.parquet':
                if PANDAS_AVAILABLE:
                    tprint_debug(f"📊 Loading parquet file: {path}")
                    data = pd.read_parquet(path)
                    tprint_data_preview(data, f"parquet_{path.stem}", level=LogLevel.DEBUG)
                    return data
                else:
                    tprint_warning(f"⚠️ Pandas not available, cannot load parquet: {path}")
                    return None
            elif path.suffix == '.csv':
                if PANDAS_AVAILABLE:
                    tprint_debug(f"📊 Loading CSV file: {path}")
                    data = pd.read_csv(path, index_col=0)
                    tprint_data_preview(data, f"csv_{path.stem}", level=LogLevel.DEBUG)
                    return data
                else:
                    tprint_warning(f"⚠️ Pandas not available, cannot load CSV: {path}")
                    return None
            elif path.suffix == '.pkl':
                tprint_debug(f"📦 Loading pickle file: {path}")
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                    tprint_data_preview(data, f"pickle_{path.stem}", level=LogLevel.DEBUG)
                    return data
            elif path.suffix == '.json':
                tprint_debug(f"📄 Loading JSON file: {path}")
                with open(path, 'r') as f:
                    return json.load(f)
            else:
                tprint_warning(f"⚠️ Unknown file extension: {path.suffix}")
                return None
        except Exception as e:
            tprint_error(f"❌ Failed to load artifact from {path}: {e}")
            return None
    
    def _find_artifact_fuzzy(self, artifact_name: str, artifact_type: str) -> Optional[Path]:
        """Find artifact using fuzzy matching across all directories with validation.
        
        Args:
            artifact_name: Name of the artifact to find
            artifact_type: Type of artifact to find
            
        Returns:
            Path to the found artifact or None if not found
            
        Raises:
            ValueError: If artifact_name is empty or invalid
            TypeError: If parameter types are incorrect
        """
        if not isinstance(artifact_name, str) or not artifact_name.strip():
            raise ValueError(f"artifact_name must be a non-empty string, got: {artifact_name}")
        if not isinstance(artifact_type, str):
            raise TypeError(f"artifact_type must be a string, got: {type(artifact_type).__name__}")
        
        tprint_debug(f"🔍 Fuzzy searching for artifact: {artifact_name} (type: {artifact_type})")
        
        try:
            if not self._artifacts_dir.exists():
                tprint_warning(f"⚠️ Artifacts directory does not exist: {self._artifacts_dir}")
                return None
            
            # Search in all subdirectories
            for file_path in self._artifacts_dir.rglob("*"):
                if file_path.is_file():
                    # Check if the filename is similar to the artifact name
                    if self._is_similar_name(artifact_name, file_path.stem):
                        # Additional check: ensure it's the right type of file
                        if self._is_correct_file_type(file_path, artifact_type):
                            tprint_success(f"✅ Found artifact with fuzzy matching: {file_path}")
                            return file_path
            
            tprint_warning(f"⚠️ No artifact found with fuzzy matching: {artifact_name}")
            return None
        except Exception as e:
            tprint_error(f"❌ Failed to search with fuzzy matching: {e}")
            return None
    
    def _is_similar_name(self, name1: str, name2: str) -> bool:
        """Check if two names are similar (for fuzzy matching) with validation.
        
        Args:
            name1: First name to compare
            name2: Second name to compare
            
        Returns:
            True if names are similar, False otherwise
        """
        if not isinstance(name1, str) or not isinstance(name2, str):
            return False
        
        try:
            # Simple similarity check
            name1_clean = name1.lower().replace('_', '').replace('-', '')
            name2_clean = name2.lower().replace('_', '').replace('-', '')
            
            # Check if one is contained in the other
            if name1_clean in name2_clean or name2_clean in name1_clean:
                tprint_debug(f"✅ Names similar (containment): '{name1}' <-> '{name2}'")
                return True
            
            # Check for common patterns
            common_patterns = ['data', 'model', 'result', 'output', 'input']
            for pattern in common_patterns:
                if pattern in name1_clean and pattern in name2_clean:
                    tprint_debug(f"✅ Names similar (pattern '{pattern}'): '{name1}' <-> '{name2}'")
                    return True
            
            return False
        except Exception as e:
            tprint_debug(f"⚠️ Error in similarity check: {e}")
            return False
    
    def _is_correct_file_type(self, file_path: Path, artifact_type: str) -> bool:
        """Check if the file type matches the expected artifact type with validation.
        
        Args:
            file_path: Path to the file to check
            artifact_type: Expected artifact type
            
        Returns:
            True if file type matches expected type, False otherwise
        """
        if not isinstance(file_path, Path) or not isinstance(artifact_type, str):
            return False
        
        try:
            file_extension = file_path.suffix.lower()
            
            # Map artifact types to expected file extensions
            type_mappings: Dict[str, List[str]] = {
                "data": [".parquet", ".csv", ".json"],
                "model": [".pkl", ".joblib", ".h5", ".onnx"],
                "metadata": [".json", ".yaml", ".yml"],
                "image": [".png", ".jpg", ".jpeg", ".svg"],
                "text": [".txt", ".md", ".log"]
            }
            
            expected_extensions = type_mappings.get(artifact_type, [".parquet", ".csv", ".json", ".pkl"])
            is_correct = file_extension in expected_extensions
            
            if is_correct:
                tprint_debug(f"✅ File type matches: {file_extension} for type '{artifact_type}'")
            else:
                tprint_debug(f"⚠️ File type mismatch: {file_extension} not in {expected_extensions} for type '{artifact_type}'")
            
            return is_correct
        except Exception as e:
            tprint_debug(f"⚠️ Error checking file type: {e}")
            return True  # Default to True if we can't determine
    
    def store_enhanced(self, key: str, data: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store artifact with enhanced features including memory profiling and spilling.
        
        Args:
            key: Key for the artifact
            data: Data to store
            metadata: Optional metadata
            
        Returns:
            True if storage was successful, False otherwise
            
        Raises:
            ValueError: If key is empty or invalid
            TypeError: If parameter types are incorrect
        """
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"key must be a non-empty string, got: {key}")
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError(f"metadata must be a dict or None, got: {type(metadata).__name__}")
        
        tprint_info(f"🚀 Storing enhanced artifact: {key}")
        
        try:
            # Profile memory usage
            memory_usage_mb = self._profile_memory_usage(key, data)
            tprint_debug(f"📊 Memory usage profiled: {memory_usage_mb:.2f}MB")
            
            # Store using regular save method
            artifact_path = self.save(data, key, "data", "auto", metadata)
            
            if artifact_path:
                # Update performance metrics
                self._performance_metrics['cache_hits'] += 1
                tprint_success(f"✅ Enhanced artifact stored successfully: {key}")
                return True
            else:
                self._performance_metrics['cache_misses'] += 1
                tprint_warning(f"⚠️ Enhanced artifact storage failed: {key}")
                return False
                
        except Exception as e:
            tprint_error(f"❌ Failed to store enhanced artifact {key}: {e}")
            return False
    
    def retrieve_enhanced(self, key: str) -> Optional[Any]:
        """Retrieve artifact with enhanced features including lazy loading.
        
        Args:
            key: Key of the artifact to retrieve
            
        Returns:
            Retrieved data or None if not found
            
        Raises:
            ValueError: If key is empty or invalid
            TypeError: If key is not a string
        """
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"key must be a non-empty string, got: {key}")
        
        tprint_info(f"🔍 Retrieving enhanced artifact: {key}")
        
        try:
            # Try regular retrieval first
            data = self.get_artifact(key, "data")
            if data is not None:
                self._performance_metrics['cache_hits'] += 1
                tprint_success(f"✅ Enhanced artifact retrieved successfully: {key}")
                return data
            else:
                self._performance_metrics['cache_misses'] += 1
                tprint_warning(f"⚠️ Enhanced artifact not found: {key}")
                return None
                
        except Exception as e:
            tprint_error(f"❌ Failed to retrieve enhanced artifact {key}: {e}")
            return None
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics with enhanced calculations.
        
        Returns:
            Dictionary containing performance metrics
        """
        tprint_debug("📊 Collecting performance metrics")
        
        try:
            total_requests = self._performance_metrics['cache_hits'] + self._performance_metrics['cache_misses']
            cache_hit_ratio = (
                self._performance_metrics['cache_hits'] / total_requests
                if total_requests > 0 else 0
            )
            
            metrics = {
                'cache_hits': self._performance_metrics['cache_hits'],
                'cache_misses': self._performance_metrics['cache_misses'],
                'cache_hit_ratio': cache_hit_ratio,
                'compression_savings_mb': self._performance_metrics['compression_savings_mb'],
                'optimization_savings_mb': self._performance_metrics['optimization_savings_mb'],
                'spill_operations': self._performance_metrics['spill_operations'],
                'lazy_loads': self._performance_metrics['lazy_loads'],
                'total_requests': total_requests
            }
            
            tprint_success(f"✅ Performance metrics collected: {cache_hit_ratio:.2%} cache hit ratio")
            return metrics
            
        except Exception as e:
            tprint_error(f"❌ Failed to collect performance metrics: {e}")
            return {"error": str(e)}
    
    def get_memory_analytics(self) -> Dict[str, Any]:
        """Get memory analytics with enhanced calculations.
        
        Returns:
            Dictionary containing memory analytics
        """
        tprint_debug("📊 Collecting memory analytics")
        
        try:
            total_memory_mb = sum(profile.get('memory_usage_mb', 0) for profile in self._memory_profiles.values())
            spilled_count = sum(1 for profile in self._memory_profiles.values() if profile.get('spilled', False))
            
            analytics = {
                'total_memory_mb': total_memory_mb,
                'spilled_count': spilled_count,
                'in_memory_artifacts': len(self._memory_profiles) - spilled_count,
                'total_artifacts': len(self._memory_profiles),
                'average_memory_per_artifact': total_memory_mb / len(self._memory_profiles) if self._memory_profiles else 0,
                'spill_ratio': spilled_count / len(self._memory_profiles) if self._memory_profiles else 0
            }
            
            tprint_success(f"✅ Memory analytics collected: {total_memory_mb:.2f}MB total, {spilled_count} spilled")
            return analytics
            
        except Exception as e:
            tprint_error(f"❌ Failed to collect memory analytics: {e}")
            return {"error": str(e)}
    
    def _profile_memory_usage(self, artifact_id: str, data: Any) -> float:
        """Profile memory usage of an artifact with enhanced error handling.
        
        Args:
            artifact_id: Unique identifier for the artifact
            data: Data to profile
            
        Returns:
            Memory usage in MB
        """
        if not isinstance(artifact_id, str) or not artifact_id.strip():
            tprint_warning(f"⚠️ Invalid artifact_id for memory profiling: {artifact_id}")
            return 0.0
        
        tprint_debug(f"📊 Profiling memory usage for artifact: {artifact_id}")
        
        try:
            memory_usage_mb = 0.0
            
            if PANDAS_AVAILABLE and hasattr(data, 'memory_usage'):
                memory_usage_mb = data.memory_usage(deep=True).sum() / (1024 * 1024)
                tprint_debug(f"📊 DataFrame memory usage: {memory_usage_mb:.2f}MB")
            elif NUMPY_AVAILABLE and hasattr(data, 'nbytes'):
                memory_usage_mb = data.nbytes / (1024 * 1024)
                tprint_debug(f"📊 NumPy array memory usage: {memory_usage_mb:.2f}MB")
            else:
                # Estimate for other types
                try:
                    memory_usage_mb = sys.getsizeof(data) / (1024 * 1024)
                    tprint_debug(f"📊 Generic object memory usage: {memory_usage_mb:.2f}MB")
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to estimate memory usage: {e}")
                    memory_usage_mb = 0.0
            
            # Store profile
            self._memory_profiles[artifact_id] = {
                'memory_usage_mb': memory_usage_mb,
                'spilled': False,
                'last_accessed': datetime.now()
            }
            
            self._total_memory_mb += memory_usage_mb
            tprint_success(f"✅ Memory profiled: {artifact_id} = {memory_usage_mb:.2f}MB")
            return memory_usage_mb
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to profile memory usage for {artifact_id}: {e}")
            return 0.0

    # ============================================================================
    # MODE-AWARE DATA FETCHING METHODS
    # ============================================================================
    
    def _get_data_loader(self) -> AresLauncherDataLoader:
        """Get or create the data loader for mode-aware data fetching."""
        if self._data_loader is None:
            self._data_loader = AresLauncherDataLoader(str(self.base_dir / "data"))
        return self._data_loader
    
    def get_mode_lookback_days(self, mode: Optional[str] = None) -> int:
        """Get lookback days for the specified mode or current context mode.
        
        Args:
            mode: Execution mode ("full", "blank", "light"). If None, uses current context mode.
            
        Returns:
            Number of lookback days for the mode
        """
        if mode is None:
            mode = self._current_execution_mode
        
        try:
            lookback_days = get_mode_lookback_days(mode)
            tprint_debug(f"📅 Mode '{mode}' lookback days: {lookback_days}")
            return lookback_days
        except Exception as e:
            tprint_warning(f"⚠️ Failed to get lookback days for mode '{mode}': {e}")
            return 20  # Default to light mode
    
    def load_data_with_mode(
        self, 
        symbol: str, 
        interval: str, 
        mode: Optional[str] = None,
        data_type: str = "raw",
        columns: Optional[List[str]] = None
    ) -> Optional[Any]:
        """Load data using mode-aware data fetching.
        
        Args:
            symbol: Trading symbol
            interval: Data interval (e.g., "15m", "1h")
            mode: Execution mode ("full", "blank", "light"). If None, uses current context mode.
            data_type: Data type ("raw" or "processed")
            columns: List of columns to load
            
        Returns:
            Loaded DataFrame or None
        """
        if mode is None:
            mode = self._current_execution_mode
        
        tprint_info(f"📊 Loading data with mode-aware fetching: {symbol} ({interval}) in {mode.upper()} mode")
        
        try:
            data_loader = self._get_data_loader()
            data = data_loader.load_data_with_mode(
                symbol=symbol,
                interval=interval,
                mode=mode,
                data_type=data_type,
                columns=columns
            )
            
            if data is not None:
                tprint_success(f"✅ Mode-aware data loaded: {len(data)} records")
                # Cache the data if caching is enabled
                if self._cache:
                    cache_key = f"mode_data_{symbol}_{interval}_{mode}"
                    self._cache.put(cache_key, data)
                    tprint_debug(f"💾 Cached mode-aware data: {cache_key}")
            
            return data
            
        except Exception as e:
            tprint_error(f"❌ Failed to load mode-aware data: {e}")
            return None
    
    def load_klines_with_mode(
        self, 
        symbol: Optional[str] = None, 
        interval: str = "15m", 
        mode: Optional[str] = None,
        data_type: str = "raw"
    ) -> Optional[Any]:
        """Load klines data using mode-aware data fetching with context.
        
        Args:
            symbol: Trading symbol. If None, uses current context symbol.
            interval: Data interval (e.g., "15m", "1h")
            mode: Execution mode ("full", "blank", "light"). If None, uses current context mode.
            data_type: Data type ("raw" or "processed")
            
        Returns:
            Loaded DataFrame or None
        """
        if symbol is None:
            symbol = self._current_symbol
        
        if symbol is None:
            tprint_error("❌ No symbol provided and no current context symbol available")
            return None
        
        if mode is None:
            mode = self._current_execution_mode
        
        tprint_info(f"📊 Loading klines with mode-aware fetching: {symbol} ({interval}) in {mode.upper()} mode")
        
        return self.load_data_with_mode(
            symbol=symbol,
            interval=interval,
            mode=mode,
            data_type=data_type
        )
    
    def get_mode_config(self, mode: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration for the specified mode or current context mode.
        
        Args:
            mode: Execution mode ("full", "blank", "light"). If None, uses current context mode.
            
        Returns:
            Mode configuration dictionary
        """
        if mode is None:
            mode = self._current_execution_mode
        
        try:
            config = get_mode_config(mode)
            tprint_debug(f"📊 Mode '{mode}' configuration retrieved")
            return {
                'name': config.name,
                'description': config.description,
                'lookback_days': config.lookback_days,
                'lookback_years': config.lookback_years,
                'intensity_percentage': config.intensity_percentage,
                'computational_intensity': config.computational_intensity,
                'estimated_duration_minutes': config.estimated_duration_minutes,
                'max_trials': config.max_trials,
                'n_trials': config.n_trials,
                'monte_carlo_samples': config.monte_carlo_samples,
                'ab_test_rounds': config.ab_test_rounds,
                'optuna_trials': config.optuna_trials,
                'optuna_timeout': config.optuna_timeout,
                'batch_size': config.batch_size,
                'epochs': config.epochs,
                'early_stopping_patience': config.early_stopping_patience,
                'cross_validation_folds': config.cross_validation_folds,
                'enable_parallelization': config.enable_parallelization,
                'enable_caching': config.enable_caching,
                'enable_advanced_features': config.enable_advanced_features,
                'enable_ensemble_training': config.enable_ensemble_training,
                'enable_multi_timeframe_training': config.enable_multi_timeframe_training,
                'enable_adaptive_training': config.enable_adaptive_training
            }
        except Exception as e:
            tprint_error(f"❌ Failed to get mode configuration for '{mode}': {e}")
            return {}
    
    def set_execution_mode(self, mode: str) -> None:
        """Set the current execution mode for data fetching.
        
        Args:
            mode: Execution mode ("full", "blank", "light")
            
        Raises:
            ValueError: If mode is invalid
            TypeError: If mode is not a string
        """
        if not isinstance(mode, str):
            raise TypeError(f"mode must be a string, got: {type(mode).__name__}")
        
        valid_modes = ["full", "blank", "light"]
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got: {mode}")
        
        self._current_execution_mode = mode
        tprint_info(f"📊 Execution mode set to: {mode.upper()}")
    
    def get_current_mode(self) -> str:
        """Get the current execution mode.
        
        Returns:
            Current execution mode
        """
        return self._current_execution_mode


def get_analyst_context(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get analyst context from configuration with validation.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing analyst context
        
    Raises:
        TypeError: If config is not a dictionary
    """
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dictionary, got: {type(config).__name__}")
    
    tprint_debug("🔍 Extracting analyst context from configuration")
    
    context = {
        'symbol': config.get('symbol', 'UNKNOWN'),
        'timeframe': config.get('timeframe', '15m'),
        'exchange': config.get('exchange', 'binance'),
        'execution_mode': config.get('execution_mode', 'light')
    }
    
    tprint_success(f"✅ Analyst context extracted: {context}")
    return context

def setup_enhanced_artifact_manager(config: Dict[str, Any]) -> ArtifactManager:
    """Setup enhanced artifact manager with configuration and validation.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured ArtifactManager instance
        
    Raises:
        TypeError: If config is not a dictionary
        ValueError: If configuration is invalid
    """
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dictionary, got: {type(config).__name__}")
    
    tprint_info("🚀 Setting up enhanced artifact manager")
    
    try:
        manager = ArtifactManager(config)
        tprint_success("✅ Enhanced artifact manager setup completed")
        return manager
    except Exception as e:
        tprint_error(f"❌ Failed to setup enhanced artifact manager: {e}")
        raise

def get_pretraining_artifact_manager(config: Dict[str, Any]) -> ArtifactManager:
    """Get pre-training artifact manager with configuration and validation.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured ArtifactManager instance for pre-training
        
    Raises:
        TypeError: If config is not a dictionary
        ValueError: If configuration is invalid
    """
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dictionary, got: {type(config).__name__}")
    
    tprint_info("🚀 Setting up pre-training artifact manager")
    
    try:
        # Add pre-training specific configuration
        pretraining_config = config.copy()
        pretraining_config.update({
            'enable_compression': True,
            'enable_caching': True,
            'enable_memory_optimization': True,
            'max_cache_size_mb': 1024.0,  # Larger cache for pre-training
            'max_memory_mb': 4000.0  # More memory for pre-training
        })
        
        manager = ArtifactManager(pretraining_config)
        tprint_success("✅ Pre-training artifact manager setup completed")
        return manager
    except Exception as e:
        tprint_error(f"❌ Failed to setup pre-training artifact manager: {e}")
        raise

# Training-specific utility functions
def get_step_context_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get step context from configuration with comprehensive validation.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing step context
        
    Raises:
        TypeError: If config is not a dictionary
        ValueError: If config is empty
    """
    # Validate input parameters
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dictionary, got: {type(config).__name__}")
    if not config:
        raise ValueError("config cannot be empty")
    
    tprint_info("🔍 Extracting step context from configuration")
    
    context = {
        'symbol': config.get('symbol', 'UNKNOWN'),
        'timeframe': config.get('timeframe', '15m'),
        'exchange': config.get('exchange', 'binance'),
        'execution_mode': config.get('execution_mode', 'light'),
        'step_name': config.get('step_name', 'unknown')
    }
    
    tprint_success(f"✅ Step context extracted: {context}")
    return context

def create_training_artifact_manager(config: Dict[str, Any]) -> ArtifactManager:
    """Create a training-specific artifact manager with enhanced configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured ArtifactManager instance for training
        
    Raises:
        TypeError: If config is not a dictionary
        ValueError: If configuration is invalid
    """
    # Validate input parameters
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dictionary, got: {type(config).__name__}")
    
    tprint_info("🚀 Creating training-specific artifact manager")
    
    # Add training-specific configuration
    training_config = config.copy()
    training_config.update({
        'enable_compression': True,
        'enable_caching': True,
        'enable_memory_optimization': True,
        'max_cache_size_mb': 2048.0,  # Larger cache for training
        'max_memory_mb': 8000.0,  # More memory for training
        'enable_thread_safety': True,
        'compression': 'auto'
    })
    
    try:
        manager = ArtifactManager(config=training_config)
        tprint_success("✅ Training artifact manager created successfully")
        return manager
    except Exception as e:
        tprint_error(f"❌ Failed to create training artifact manager: {e}")
        raise

def validate_training_config(config: Dict[str, Any]) -> None:
    """Validate training configuration with comprehensive checks.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        TypeError: If config is not a dictionary
        ValueError: If configuration is invalid
    """
    # Validate input parameters
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dictionary, got: {type(config).__name__}")
    if not config:
        raise ValueError("config cannot be empty")
    
    tprint_info("🔍 Validating training configuration")
    
    # Required fields for training
    required_fields = ['step_name', 'execution_mode', 'symbol', 'exchange']
    missing_fields = [field for field in required_fields if field not in config]
    
    if missing_fields:
        raise ValueError(f"Missing required training configuration fields: {missing_fields}")
    
    # Validate field types
    type_validations = {
        'step_name': str,
        'execution_mode': str,
        'symbol': str,
        'exchange': str,
        'timeframe': str,
        'model': str
    }
    
    for field, expected_type in type_validations.items():
        if field in config and not isinstance(config[field], expected_type):
            raise TypeError(f"Config field '{field}' must be {expected_type.__name__}, got {type(config[field]).__name__}")
    
    tprint_success("✅ Training configuration validation passed")

def get_training_metrics(artifact_manager: ArtifactManager) -> Dict[str, Any]:
    """Get comprehensive training metrics from artifact manager.
    
    Args:
        artifact_manager: ArtifactManager instance
        
    Returns:
        Dictionary containing training metrics
        
    Raises:
        TypeError: If artifact_manager is not an ArtifactManager instance
    """
    if not isinstance(artifact_manager, ArtifactManager):
        raise TypeError(f"artifact_manager must be an ArtifactManager instance, got: {type(artifact_manager).__name__}")
    
    tprint_info("📊 Collecting training metrics")
    
    try:
        # Get comprehensive stats
        stats = artifact_manager.get_stats()
        performance_metrics = artifact_manager.get_performance_metrics()
        memory_analytics = artifact_manager.get_memory_analytics()
        
        training_metrics = {
            'stats': stats,
            'performance': performance_metrics,
            'memory': memory_analytics,
            'timestamp': datetime.now().isoformat()
        }
        
        tprint_success(f"✅ Training metrics collected: {len(training_metrics)} metric categories")
        return training_metrics
        
    except Exception as e:
        tprint_error(f"❌ Failed to collect training metrics: {e}")
        return {'error': str(e)}

def log_training_progress(step_name: str, progress: float, message: str = "") -> None:
    """Log training progress with structured information.
    
    Args:
        step_name: Name of the training step
        progress: Progress percentage (0.0 to 1.0)
        message: Optional progress message
        
    Raises:
        ValueError: If step_name is empty or progress is invalid
        TypeError: If parameter types are incorrect
    """
    # Validate input parameters
    if not isinstance(step_name, str) or not step_name.strip():
        raise ValueError(f"step_name must be a non-empty string, got: {step_name}")
    if not isinstance(progress, (int, float)):
        raise TypeError(f"progress must be a number, got: {type(progress).__name__}")
    if not 0.0 <= progress <= 1.0:
        raise ValueError(f"progress must be between 0.0 and 1.0, got: {progress}")
    if not isinstance(message, str):
        raise TypeError(f"message must be a string, got: {type(message).__name__}")
    
    progress_percentage = progress * 100
    
    if message:
        tprint_progress(int(progress_percentage), 100, f"{step_name}: {message}")
    else:
        tprint_progress(int(progress_percentage), 100, step_name)

def log_training_error(step_name: str, error: Exception, context: str = "") -> None:
    """Log training error with comprehensive context.
    
    Args:
        step_name: Name of the training step
        error: Exception that occurred
        context: Optional context information
        
    Raises:
        ValueError: If step_name is empty
        TypeError: If parameter types are incorrect
    """
    # Validate input parameters
    if not isinstance(step_name, str) or not step_name.strip():
        raise ValueError(f"step_name must be a non-empty string, got: {step_name}")
    if not isinstance(error, Exception):
        raise TypeError(f"error must be an Exception, got: {type(error).__name__}")
    if not isinstance(context, str):
        raise TypeError(f"context must be a string, got: {type(context).__name__}")
    
    error_message = f"Training step '{step_name}' failed"
    if context:
        error_message += f" in {context}"
    
    tprint_error(error_message)
    tprint_exception(error, context)

__all__ = [
    'ArtifactManager',
    'get_analyst_context', 
    'setup_enhanced_artifact_manager',
    'get_pretraining_artifact_manager',
    'get_step_category',
    'ensure_step_category_directories',
    'STEP_CATEGORIES',
    # Training-specific functions
    'get_step_context_from_config',
    'create_training_artifact_manager',
    'validate_training_config',
    'get_training_metrics',
    'log_training_progress',
    'log_training_error',
    # Re-export types for better IDE support
    'ArtifactMetadata',
    'OperationMetrics',
    'CacheEntry',
    'CompressionType',
    'OperationType',
    'RetryStrategy',
    'RetryConfig',
    'MemoryConfig',
    'LogLevel',
    'TPrintConfig'
]

