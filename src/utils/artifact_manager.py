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
from typing import Optional, Any, Dict, List, Tuple, Union
from dataclasses import dataclass, field
from contextlib import nullcontext, contextmanager, asynccontextmanager
from datetime import datetime, timedelta
from enum import Enum

from .artifact_storage import ArtifactStorage
from .compression_manager import CompressionManager, CompressionConfig
from .cache_manager import CacheManager, CacheConfig
from .memory_manager import MemoryManager, MemoryConfig
from .path_manager import PathManager
from .logger import system_logger
from .tprint import tprint, tprint_success, tprint_info, tprint_warning, tprint_error
from .common_operations import ensure_directory
from .version_manager import get_version_manager

# Import hardware optimization tools
try:
    from .hardware import (
        get_integrated_hardware_manager, memory_optimized, 
        performance_tracked, force_cleanup, get_memory_stats,
        optimize_dataframe, optimize_array, cache_result,
        MemoryOptimizationLevel
    )
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    HARDWARE_OPTIMIZATION_AVAILABLE = False
    # Create dummy functions and classes for compatibility
    class MemoryOptimizationLevel:
        AGGRESSIVE = "AGGRESSIVE"
        BALANCED = "BALANCED"
        CONSERVATIVE = "CONSERVATIVE"
    
    def get_integrated_hardware_manager(): return None
    def memory_optimized(*args, **kwargs): return lambda f: f
    def performance_tracked(*args, **kwargs): return lambda f: f
    def force_cleanup(): pass
    def get_memory_stats(): return {}
    def optimize_dataframe(df): return df
    def optimize_array(arr): return arr
    def cache_result(*args, **kwargs): return lambda f: f

# Import optional dependencies
try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
    NUMPY_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    NUMPY_AVAILABLE = False

try:
    import lz4.frame
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class CompressionType(Enum):
    """Supported compression algorithms."""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    AUTO = "auto"  # Automatically choose best compression


class OperationType(Enum):
    """Types of artifact operations."""
    SAVE = "save"
    LOAD = "load"
    DELETE = "delete"
    LIST = "list"


class RetryStrategy(Enum):
    """Retry strategies for failed operations."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"


@dataclass
class RetryConfig:
    """Configuration for retry mechanisms."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    retryable_exceptions: Tuple[type, ...] = (OSError, IOError, ConnectionError)




@dataclass
class MemoryConfig:
    """Configuration for memory management."""
    max_memory_mb: float = 2000.0
    cache_memory_mb: float = 500.0
    spill_threshold_mb: float = 150.0
    cleanup_interval_seconds: float = 300.0
    enable_gc_collection: bool = True


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


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    artifact_key: str
    data: Any
    metadata: ArtifactMetadata
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    memory_size_mb: float = 0.0


# Step category mapping for organized artifact storage
STEP_CATEGORIES = {
    'data_collection': ['step01', 'data_downloader', 'klines_downloading_processing'],
    'market_analysis': ['step02', 'market_analysis', 'sr_detection', 'regime_discovery'],
    'pre_training': ['step02_5', 'feature_generation', 'pre_training'],
    'models_training': ['step03', 'model_training', 'analyst_models', 'tactician_models'],
    'backtesting': ['step04', 'backtesting', 'real_parameters_optimization']
}


def get_step_category(step_name: str) -> str:
    """Determine the category for a step based on its name."""
    step_name_lower = step_name.lower()
    for category, patterns in STEP_CATEGORIES.items():
        if any(pattern.lower() in step_name_lower for pattern in patterns):
            return category
    return 'pre_training'  # Default fallback


def _format_data_preview(data: Any, artifact_name: str) -> str:
    """Format a data preview for tprint output."""
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
            
            return preview_str
            
        elif isinstance(data, np.ndarray):
            shape = data.shape
            file_size_mb = data.nbytes / (1024 * 1024)
            
            preview_str = f"NumPy Array: {shape} | {file_size_mb:.2f}MB\n"
            if len(shape) == 2:
                preview_str += f"Preview (5×10):\n{data[:5, :10]}"
            else:
                preview_str += f"Preview: {data.flat[:10]}..."
            
            return preview_str
            
        elif isinstance(data, (list, tuple)):
            length = len(data)
            file_size_mb = sum(sys.getsizeof(item) for item in data[:100]) / (1024 * 1024)  # Estimate
            
            preview_str = f"List/Tuple: {length:,} items | ~{file_size_mb:.2f}MB\n"
            preview_str += f"Preview: {data[:5]}{'...' if length > 5 else ''}"
            
            return preview_str
            
        elif isinstance(data, dict):
            length = len(data)
            file_size_mb = sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in list(data.items())[:50]) / (1024 * 1024)  # Estimate
            
            preview_str = f"Dict: {length:,} keys | ~{file_size_mb:.2f}MB\n"
            preview_str += f"Keys: {list(data.keys())[:5]}{'...' if length > 5 else ''}"
            
            return preview_str
            
        else:
            file_size_mb = sys.getsizeof(data) / (1024 * 1024)
            return f"{type(data).__name__}: {file_size_mb:.2f}MB"
            
    except Exception as e:
        return f"Preview unavailable: {str(e)[:50]}..."


class ArtifactManager:
    """Simplified artifact manager that uses refactored components."""
    
    def __init__(self, config: dict):
        """Initialize the artifact manager.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = system_logger.getChild("ArtifactManager")
        
        # Initialize base directory
        self.base_dir = Path(config.get("paths", {}).get("data_dir", "data"))
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._storage = ArtifactStorage(self.base_dir)
        self._path_manager = PathManager(self.base_dir)
        
        # Initialize optional components
        if config.get("enable_compression", True):
            compression_config = CompressionConfig()
            self._compression = CompressionManager(compression_config)
        else:
            self._compression = None
        
        if config.get("enable_caching", True):
            cache_config = CacheConfig(
                max_size_mb=config.get("max_cache_size_mb", 512.0),
                enable_thread_safety=config.get("enable_thread_safety", True)
            )
            self._cache = CacheManager(cache_config)
        else:
            self._cache = None
        
        if config.get("enable_memory_optimization", True):
            memory_config = MemoryConfig(
                max_memory_mb=config.get("max_memory_mb", 2000.0),
                spill_threshold_mb=config.get("spill_threshold_mb", 150.0)
            )
            spill_dir = self.base_dir / "spilled"
            self._memory = MemoryManager(memory_config, spill_dir)
        else:
            self._memory = None
        
        # Thread safety
        if config.get("enable_thread_safety", True):
            import threading
            import asyncio
            self._lock = threading.RLock()
            self._async_lock = asyncio.Lock()
        else:
            self._lock = None
            self._async_lock = None
        
        # Store original config for compatibility
        self.config = config
    
    def _lock_context(self):
        """Get lock context manager."""
        if self._lock is not None:
            return self._lock
        return nullcontext()
    
    async def _async_lock_context(self):
        """Get async lock context manager."""
        if self._async_lock is not None:
            return self._async_lock
        return nullcontext()
    
    def set_context(self, step_name: str, symbol: Optional[str] = None, 
                   exchange: Optional[str] = None, datetime: Optional[Any] = None, 
                   information: Optional[str] = None, direction: str = "long", 
                   model: str = "Analyst") -> None:
        """Set the current context for path generation."""
        tprint_info(f"📁 SETTING CONTEXT: {step_name} | {symbol} | {exchange} | {direction} | {model}")
        with self._lock_context():
            self._path_manager.set_context(
                step_name=step_name,
                symbol=symbol,
                exchange=exchange,
                datetime=datetime,
                information=information,
                direction=direction,
                model=model
            )
    
    def save(self, data: Any, artifact_name: str, 
             artifact_type: str = "data", 
             compression: str = "auto",
             metadata: Optional[Dict] = None) -> str:
        """Save an artifact."""
        with self._lock_context():
            try:
                # Print data preview before saving
                preview = _format_data_preview(data, artifact_name)
                tprint_info(f"💾 SAVING ARTIFACT: {artifact_name}")
                tprint_info(f"📊 Data Preview:\n{preview}")
                
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
                    self._cache.put(artifact_name, data)
                
                # Profile memory usage if memory manager is available
                if self._memory:
                    self._memory.profile_memory_usage(artifact_name, data)
                
                # Print success message
                tprint_success(f"✅ ARTIFACT SAVED: {artifact_name} → {file_path}")
                
                return str(file_path)
                
            except Exception as e:
                tprint_error(f"❌ FAILED TO SAVE ARTIFACT: {artifact_name} - {str(e)}")
                raise
    
    def get_artifact(self, artifact_name: str, 
                    artifact_type: str = "data") -> Optional[Any]:
        """Retrieve an artifact."""
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
                
                if data is not None:
                    # Cache if enabled
                    if self._cache:
                        self._cache.put(artifact_name, data)
                    
                    # Profile memory usage if memory manager is available
                    if self._memory:
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
        """Delete an artifact."""
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
                    self._cache.remove(artifact_name)
                
                # Remove from memory profiles if memory manager is available
                if self._memory and artifact_name in self._memory._memory_profiles:
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
    
    def list_artifacts(self, pattern: str = "*") -> list[Path]:
        """List artifacts matching a pattern."""
        return self._storage.list_artifacts(pattern)
    
    def clear_cache(self) -> None:
        """Clear the cache."""
        tprint_info("🧹 CLEARING CACHE")
        if self._cache:
            self._cache.clear()
        tprint_success("✅ CACHE CLEARED")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        stats = {
            "config": {
                "base_dir": str(self.base_dir),
                "enable_compression": self._compression is not None,
                "enable_caching": self._cache is not None,
                "enable_memory_optimization": self._memory is not None,
                "enable_thread_safety": self._lock is not None
            }
        }
        
        # Add cache stats
        if self._cache:
            stats["cache"] = self._cache.get_stats()
        
        # Add memory stats
        if self._memory:
            stats["memory"] = self._memory.get_memory_stats()
        
        # Add compression stats
        if self._compression:
            stats["compression"] = self._compression.get_compression_stats()
        
        return stats
    
    def cleanup(self) -> None:
        """Perform cleanup operations."""
        tprint_info("🧹 PERFORMING CLEANUP")
        
        # Cleanup cache
        if self._cache:
            self._cache.periodic_cleanup()
        
        # Cleanup memory
        if self._memory:
            self._memory.periodic_cleanup()
        
        tprint_success("✅ CLEANUP COMPLETED")
    
    async def run_context(self, run_id: str):
        """Async context manager for automatic cleanup."""
        async with await self._async_lock_context():
            run_dir = self.base_dir / f"run_{run_id}"
            run_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                yield run_dir
            finally:
                # Auto-cleanup run directory
                try:
                    import shutil
                    shutil.rmtree(run_dir, ignore_errors=True)
                    tprint_info(f"Cleaned up run directory: {run_dir}")
                except Exception as e:
                    tprint_warning(f"Failed to cleanup run directory {run_dir}: {e}")
    
    # Compatibility methods for existing code
    def get_data_dir(self, *subdirs: str) -> Path:
        """Get data directory path."""
        return self.base_dir / "data" / Path(*subdirs)
    
    def get_reports_dir(self, *subdirs: str) -> Path:
        """Get reports directory path."""
        return self.base_dir / "reports" / Path(*subdirs)
    
    def get_cache_dir(self, *subdirs: str) -> Path:
        """Get cache directory path."""
        return self.base_dir / "cache" / Path(*subdirs)
    
    def get_optimization_dir(self, *subdirs: str) -> Path:
        """Get optimization directory path."""
        return self.base_dir / "optimization" / Path(*subdirs)
    
    def get_tmp_dir(self, *subdirs: str) -> Path:
        """Get temporary directory path."""
        return self.base_dir / "tmp" / Path(*subdirs)
    
    def get_tmp_path(self, filename: str) -> Path:
        """Get temporary file path."""
        return self.get_tmp_dir() / filename
    
    def reset_run(self) -> None:
        """Reset run state (compatibility method)."""
        # The refactored manager handles this automatically
        pass
    
    def get_run_id(self) -> Optional[str]:
        """Get current run ID (compatibility method)."""
        return None
    
    def get_run_dir(self) -> Optional[Path]:
        """Get current run directory (compatibility method)."""
        return self.base_dir

