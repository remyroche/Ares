"""
Progressive Loading for Large Datasets

This module provides progressive loading capabilities for large datasets,
integrating with M1 hardware optimizations, VectorBT, and async patterns.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Iterator, AsyncIterator
import weakref
from pathlib import Path
import json
import pickle

# Optional dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    h5py = None

try:
    import parquet
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    parquet = None

# Import M1 optimizations
try:
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
    from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
    M1_OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    M1_OPTIMIZATIONS_AVAILABLE = False

# Import VectorBT optimizations
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
    VECTORBT_OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    VECTORBT_OPTIMIZATIONS_AVAILABLE = False

# Import async patterns
try:
    from .async_patterns import get_async_operation_manager, AsyncOperationType
    ASYNC_PATTERNS_AVAILABLE = True
except ImportError:
    ASYNC_PATTERNS_AVAILABLE = False

# Import caching and memory profiling
try:
    from .caching_strategies import get_ml_common_cache, CacheConfig, CacheStrategy
    from .memory_profiler import get_memory_profiler, MemoryOptimizationConfig
    PERFORMANCE_MODULES_AVAILABLE = True
except ImportError:
    PERFORMANCE_MODULES_AVAILABLE = False

logger = logging.getLogger(__name__)

class LoadingStrategy(Enum):
    """Progressive loading strategies."""
    CHUNKED = "chunked"
    STREAMING = "streaming"
    LAZY = "lazy"
    MEMORY_MAPPED = "memory_mapped"
    CACHED = "cached"
    ADAPTIVE = "adaptive"

class DataFormat(Enum):
    """Supported data formats."""
    CSV = "csv"
    PARQUET = "parquet"
    HDF5 = "hdf5"
    JSON = "json"
    PICKLE = "pickle"
    NUMPY = "numpy"
    PANDAS = "pandas"

@dataclass
class ProgressiveLoadingConfig:
    """Configuration for progressive loading."""
    
    # Basic settings
    chunk_size: int = 10000
    max_memory_mb: float = 500.0
    enable_compression: bool = True
    enable_caching: bool = True
    
    # Loading strategy
    loading_strategy: LoadingStrategy = LoadingStrategy.ADAPTIVE
    prefer_memory_mapped: bool = True
    enable_streaming: bool = True
    
    # M1 optimizations
    enable_m1_optimizations: bool = True
    use_m1_memory_optimizer: bool = True
    use_m1_cpu_optimizer: bool = True
    use_m1_gpu_optimizer: bool = True
    
    # VectorBT optimizations
    enable_vectorbt_optimizations: bool = True
    use_vectorbt_rolling: bool = True
    
    # Async settings
    enable_async_loading: bool = True
    max_concurrent_chunks: int = 4
    prefetch_chunks: int = 2
    
    # Memory management
    enable_memory_profiling: bool = True
    memory_threshold_mb: float = 1000.0
    gc_threshold: int = 10  # Trigger GC every N chunks
    
    # Caching settings
    cache_chunks: bool = True
    cache_ttl_seconds: int = 3600
    max_cached_chunks: int = 100

@dataclass
class DataChunk:
    """Represents a data chunk."""
    
    chunk_id: str
    data: Any
    start_index: int
    end_index: int
    size_bytes: int
    created_at: float
    accessed_at: float
    access_count: int = 0
    
    def update_access(self):
        """Update access statistics."""
        self.accessed_at = time.time()
        self.access_count += 1

class ProgressiveLoader(ABC):
    """Base class for progressive loaders."""
    
    def __init__(self, config: ProgressiveLoadingConfig):
        self.config = config
        self.logger = logger.getChild(self.__class__.__name__)
        self._m1_memory_optimizer = None
        self._m1_cpu_optimizer = None
        self._m1_gpu_manager = None
        self._vectorbt_optimizer = None
        self._async_manager = None
        self._cache = None
        self._memory_profiler = None
        
        # Initialize optimizations
        if M1_OPTIMIZATIONS_AVAILABLE and config.enable_m1_optimizations:
            if config.use_m1_memory_optimizer:
                self._m1_memory_optimizer = get_m1_memory_optimizer()
            if config.use_m1_cpu_optimizer:
                self._m1_cpu_optimizer = get_m1_cpu_optimizer()
            if config.use_m1_gpu_optimizer:
                self._m1_gpu_manager = get_m1_gpu_manager()
        
        if VECTORBT_OPTIMIZATIONS_AVAILABLE and config.enable_vectorbt_optimizations:
            self._vectorbt_optimizer = get_vectorbt_rolling_optimizer()
        
        if ASYNC_PATTERNS_AVAILABLE and config.enable_async_loading:
            self._async_manager = get_async_operation_manager()
        
        if PERFORMANCE_MODULES_AVAILABLE:
            if config.enable_caching:
                cache_config = CacheConfig(
                    strategy=CacheStrategy.MEMORY,
                    enable_m1_optimizations=config.enable_m1_optimizations,
                    enable_vectorbt_optimizations=config.enable_vectorbt_optimizations
                )
                self._cache = get_ml_common_cache(cache_config)
            
            if config.enable_memory_profiling:
                memory_config = MemoryOptimizationConfig(
                    enable_m1_optimizations=config.enable_m1_optimizations,
                    enable_vectorbt_optimizations=config.enable_vectorbt_optimizations
                )
                self._memory_profiler = get_memory_profiler(memory_config)
    
    @abstractmethod
    async def load_chunk(self, start_index: int, end_index: int) -> DataChunk:
        """Load a specific chunk of data."""
        pass
    
    @abstractmethod
    async def get_total_size(self) -> int:
        """Get total size of the dataset."""
        pass
    
    @abstractmethod
    async def get_chunk_count(self) -> int:
        """Get total number of chunks."""
        pass
    
    async def load_all_chunks(self) -> List[DataChunk]:
        """Load all chunks progressively."""
        total_size = await self.get_total_size()
        chunk_count = await self.get_chunk_count()
        
        chunks = []
        for i in range(chunk_count):
            start_index = i * self.config.chunk_size
            end_index = min((i + 1) * self.config.chunk_size, total_size)
            
            chunk = await self.load_chunk(start_index, end_index)
            chunks.append(chunk)
            
            # Memory management
            if self._memory_profiler and i % self.config.gc_threshold == 0:
                self._memory_profiler._conservative_memory_cleanup()
        
        return chunks
    
    async def load_chunks_async(self, chunk_indices: List[int]) -> List[DataChunk]:
        """Load multiple chunks asynchronously."""
        if not self._async_manager:
            # Fallback to sequential loading
            chunks = []
            for chunk_index in chunk_indices:
                start_index = chunk_index * self.config.chunk_size
                end_index = min((chunk_index + 1) * self.config.chunk_size, await self.get_total_size())
                chunk = await self.load_chunk(start_index, end_index)
                chunks.append(chunk)
            return chunks
        
        # Load chunks asynchronously
        operations = []
        for chunk_index in chunk_indices:
            start_index = chunk_index * self.config.chunk_size
            end_index = min((chunk_index + 1) * self.config.chunk_size, await self.get_total_size())
            
            operation = (
                self.load_chunk,
                (start_index, end_index),
                {},
                AsyncOperationType.MEMORY_INTENSIVE
            )
            operations.append(operation)
        
        results = await self._async_manager.execute_batch_async(operations)
        return [result for result in results if isinstance(result, DataChunk)]

class CSVProgressiveLoader(ProgressiveLoader):
    """Progressive loader for CSV files."""
    
    def __init__(self, filepath: str, config: ProgressiveLoadingConfig):
        super().__init__(config)
        self.filepath = filepath
        self._total_rows = None
        self._cached_chunks: Dict[str, DataChunk] = {}
    
    async def load_chunk(self, start_index: int, end_index: int) -> DataChunk:
        """Load a chunk from CSV file."""
        chunk_id = f"csv_{start_index}_{end_index}"
        
        # Check cache first
        if self.config.enable_caching and chunk_id in self._cached_chunks:
            chunk = self._cached_chunks[chunk_id]
            chunk.update_access()
            return chunk
        
        # Load chunk
        if self._async_manager:
            # Use async loading
            def _load_csv_chunk():
                return pd.read_csv(
                    self.filepath,
                    skiprows=start_index,
                    nrows=end_index - start_index,
                    chunksize=self.config.chunk_size
                )
            
            data = await self._async_manager.execute_async(
                _load_csv_chunk,
                operation_type=AsyncOperationType.FILE_IO
            )
        else:
            # Synchronous loading
            data = pd.read_csv(
                self.filepath,
                skiprows=start_index,
                nrows=end_index - start_index,
                chunksize=self.config.chunk_size
            )
        
        # Apply optimizations
        if self._m1_memory_optimizer:
            data = self._m1_memory_optimizer.optimize_dataframe_memory(data)
        
        if self._vectorbt_optimizer:
            data = self._vectorbt_optimizer.optimize_dataframe(data)
        
        # Create chunk
        chunk = DataChunk(
            chunk_id=chunk_id,
            data=data,
            start_index=start_index,
            end_index=end_index,
            size_bytes=data.memory_usage(deep=True).sum() if hasattr(data, 'memory_usage') else 0,
            created_at=time.time(),
            accessed_at=time.time()
        )
        
        # Cache chunk
        if self.config.enable_caching:
            self._cached_chunks[chunk_id] = chunk
        
        return chunk
    
    async def get_total_size(self) -> int:
        """Get total number of rows in CSV file."""
        if self._total_rows is None:
            if self._async_manager:
                def _count_rows():
                    return sum(1 for _ in open(self.filepath, 'r'))
                
                self._total_rows = await self._async_manager.execute_async(
                    _count_rows,
                    operation_type=AsyncOperationType.FILE_IO
                )
            else:
                self._total_rows = sum(1 for _ in open(self.filepath, 'r'))
        
        return self._total_rows
    
    async def get_chunk_count(self) -> int:
        """Get total number of chunks."""
        total_size = await self.get_total_size()
        return (total_size + self.config.chunk_size - 1) // self.config.chunk_size

class ParquetProgressiveLoader(ProgressiveLoader):
    """Progressive loader for Parquet files."""
    
    def __init__(self, filepath: str, config: ProgressiveLoadingConfig):
        super().__init__(config)
        self.filepath = filepath
        self._total_rows = None
        self._cached_chunks: Dict[str, DataChunk] = {}
    
    async def load_chunk(self, start_index: int, end_index: int) -> DataChunk:
        """Load a chunk from Parquet file."""
        chunk_id = f"parquet_{start_index}_{end_index}"
        
        # Check cache first
        if self.config.enable_caching and chunk_id in self._cached_chunks:
            chunk = self._cached_chunks[chunk_id]
            chunk.update_access()
            return chunk
        
        # Load chunk
        if self._async_manager:
            def _load_parquet_chunk():
                return pd.read_parquet(
                    self.filepath,
                    engine='pyarrow',
                    columns=None,  # Load all columns
                    use_pandas_metadata=True
                ).iloc[start_index:end_index]
            
            data = await self._async_manager.execute_async(
                _load_parquet_chunk,
                operation_type=AsyncOperationType.FILE_IO
            )
        else:
            data = pd.read_parquet(
                self.filepath,
                engine='pyarrow',
                columns=None,
                use_pandas_metadata=True
            ).iloc[start_index:end_index]
        
        # Apply optimizations
        if self._m1_memory_optimizer:
            data = self._m1_memory_optimizer.optimize_dataframe_memory(data)
        
        if self._vectorbt_optimizer:
            data = self._vectorbt_optimizer.optimize_dataframe(data)
        
        # Create chunk
        chunk = DataChunk(
            chunk_id=chunk_id,
            data=data,
            start_index=start_index,
            end_index=end_index,
            size_bytes=data.memory_usage(deep=True).sum() if hasattr(data, 'memory_usage') else 0,
            created_at=time.time(),
            accessed_at=time.time()
        )
        
        # Cache chunk
        if self.config.enable_caching:
            self._cached_chunks[chunk_id] = chunk
        
        return chunk
    
    async def get_total_size(self) -> int:
        """Get total number of rows in Parquet file."""
        if self._total_rows is None:
            if self._async_manager:
                def _count_parquet_rows():
                    return len(pd.read_parquet(self.filepath, engine='pyarrow'))
                
                self._total_rows = await self._async_manager.execute_async(
                    _count_parquet_rows,
                    operation_type=AsyncOperationType.FILE_IO
                )
            else:
                self._total_rows = len(pd.read_parquet(self.filepath, engine='pyarrow'))
        
        return self._total_rows
    
    async def get_chunk_count(self) -> int:
        """Get total number of chunks."""
        total_size = await self.get_total_size()
        return (total_size + self.config.chunk_size - 1) // self.config.chunk_size

class HDF5ProgressiveLoader(ProgressiveLoader):
    """Progressive loader for HDF5 files."""
    
    def __init__(self, filepath: str, dataset_path: str, config: ProgressiveLoadingConfig):
        super().__init__(config)
        self.filepath = filepath
        self.dataset_path = dataset_path
        self._total_rows = None
        self._cached_chunks: Dict[str, DataChunk] = {}
    
    async def load_chunk(self, start_index: int, end_index: int) -> DataChunk:
        """Load a chunk from HDF5 file."""
        chunk_id = f"hdf5_{start_index}_{end_index}"
        
        # Check cache first
        if self.config.enable_caching and chunk_id in self._cached_chunks:
            chunk = self._cached_chunks[chunk_id]
            chunk.update_access()
            return chunk
        
        # Load chunk
        if self._async_manager:
            def _load_hdf5_chunk():
                with h5py.File(self.filepath, 'r') as f:
                    dataset = f[self.dataset_path]
                    return dataset[start_index:end_index]
            
            data = await self._async_manager.execute_async(
                _load_hdf5_chunk,
                operation_type=AsyncOperationType.FILE_IO
            )
        else:
            with h5py.File(self.filepath, 'r') as f:
                dataset = f[self.dataset_path]
                data = dataset[start_index:end_index]
        
        # Convert to DataFrame if needed
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        
        # Apply optimizations
        if self._m1_memory_optimizer:
            data = self._m1_memory_optimizer.optimize_dataframe_memory(data)
        
        if self._vectorbt_optimizer:
            data = self._vectorbt_optimizer.optimize_dataframe(data)
        
        # Create chunk
        chunk = DataChunk(
            chunk_id=chunk_id,
            data=data,
            start_index=start_index,
            end_index=end_index,
            size_bytes=data.memory_usage(deep=True).sum() if hasattr(data, 'memory_usage') else 0,
            created_at=time.time(),
            accessed_at=time.time()
        )
        
        # Cache chunk
        if self.config.enable_caching:
            self._cached_chunks[chunk_id] = chunk
        
        return chunk
    
    async def get_total_size(self) -> int:
        """Get total number of rows in HDF5 dataset."""
        if self._total_rows is None:
            if self._async_manager:
                def _count_hdf5_rows():
                    with h5py.File(self.filepath, 'r') as f:
                        dataset = f[self.dataset_path]
                        return dataset.shape[0]
                
                self._total_rows = await self._async_manager.execute_async(
                    _count_hdf5_rows,
                    operation_type=AsyncOperationType.FILE_IO
                )
            else:
                with h5py.File(self.filepath, 'r') as f:
                    dataset = f[self.dataset_path]
                    self._total_rows = dataset.shape[0]
        
        return self._total_rows
    
    async def get_chunk_count(self) -> int:
        """Get total number of chunks."""
        total_size = await self.get_total_size()
        return (total_size + self.config.chunk_size - 1) // self.config.chunk_size

class AdaptiveProgressiveLoader(ProgressiveLoader):
    """Adaptive progressive loader that chooses the best strategy."""
    
    def __init__(self, filepath: str, config: ProgressiveLoadingConfig):
        super().__init__(config)
        self.filepath = filepath
        self._loader: Optional[ProgressiveLoader] = None
        self._file_format = None
    
    async def _detect_file_format(self) -> DataFormat:
        """Detect file format."""
        filepath = Path(self.filepath)
        suffix = filepath.suffix.lower()
        
        if suffix == '.csv':
            return DataFormat.CSV
        elif suffix == '.parquet':
            return DataFormat.PARQUET
        elif suffix in ['.h5', '.hdf5']:
            return DataFormat.HDF5
        elif suffix == '.json':
            return DataFormat.JSON
        elif suffix == '.pkl':
            return DataFormat.PICKLE
        else:
            # Try to detect by content
            try:
                with open(self.filepath, 'rb') as f:
                    header = f.read(8)
                    if header.startswith(b'PAR1'):
                        return DataFormat.PARQUET
                    elif header.startswith(b'\x89HDF'):
                        return DataFormat.HDF5
            except Exception:
                pass
            
            return DataFormat.CSV  # Default fallback
    
    async def _initialize_loader(self):
        """Initialize the appropriate loader."""
        if self._loader is not None:
            return
        
        self._file_format = await self._detect_file_format()
        
        if self._file_format == DataFormat.CSV:
            self._loader = CSVProgressiveLoader(self.filepath, self.config)
        elif self._file_format == DataFormat.PARQUET:
            self._loader = ParquetProgressiveLoader(self.filepath, self.config)
        elif self._file_format == DataFormat.HDF5:
            # For HDF5, we need to detect the dataset path
            # This is a simplified approach - in practice, you'd want more sophisticated detection
            self._loader = HDF5ProgressiveLoader(self.filepath, '/data', self.config)
        else:
            # Fallback to CSV loader
            self._loader = CSVProgressiveLoader(self.filepath, self.config)
    
    async def load_chunk(self, start_index: int, end_index: int) -> DataChunk:
        """Load a chunk using the appropriate loader."""
        await self._initialize_loader()
        return await self._loader.load_chunk(start_index, end_index)
    
    async def get_total_size(self) -> int:
        """Get total size using the appropriate loader."""
        await self._initialize_loader()
        return await self._loader.get_total_size()
    
    async def get_chunk_count(self) -> int:
        """Get chunk count using the appropriate loader."""
        await self._initialize_loader()
        return await self._loader.get_chunk_count()

class ProgressiveDataProcessor:
    """Processor for progressively loaded data."""
    
    def __init__(self, config: Optional[ProgressiveLoadingConfig] = None):
        self.config = config or ProgressiveLoadingConfig()
        self.logger = logger.getChild('ProgressiveDataProcessor')
        self._loaders: Dict[str, ProgressiveLoader] = {}
        self._memory_profiler = None
        
        if PERFORMANCE_MODULES_AVAILABLE and self.config.enable_memory_profiling:
            memory_config = MemoryOptimizationConfig(
                enable_m1_optimizations=self.config.enable_m1_optimizations,
                enable_vectorbt_optimizations=self.config.enable_vectorbt_optimizations
            )
            self._memory_profiler = get_memory_profiler(memory_config)
    
    async def create_loader(self, filepath: str, loader_type: Optional[LoadingStrategy] = None) -> ProgressiveLoader:
        """Create a progressive loader for a file."""
        if loader_type is None:
            loader_type = self.config.loading_strategy
        
        if loader_type == LoadingStrategy.ADAPTIVE:
            loader = AdaptiveProgressiveLoader(filepath, self.config)
        elif loader_type == LoadingStrategy.CHUNKED:
            # Choose based on file format
            file_format = await self._detect_file_format(filepath)
            if file_format == DataFormat.CSV:
                loader = CSVProgressiveLoader(filepath, self.config)
            elif file_format == DataFormat.PARQUET:
                loader = ParquetProgressiveLoader(filepath, self.config)
            elif file_format == DataFormat.HDF5:
                loader = HDF5ProgressiveLoader(filepath, '/data', self.config)
            else:
                loader = CSVProgressiveLoader(filepath, self.config)
        else:
            # Default to adaptive
            loader = AdaptiveProgressiveLoader(filepath, self.config)
        
        self._loaders[filepath] = loader
        return loader
    
    async def _detect_file_format(self, filepath: str) -> DataFormat:
        """Detect file format."""
        filepath_obj = Path(filepath)
        suffix = filepath_obj.suffix.lower()
        
        if suffix == '.csv':
            return DataFormat.CSV
        elif suffix == '.parquet':
            return DataFormat.PARQUET
        elif suffix in ['.h5', '.hdf5']:
            return DataFormat.HDF5
        elif suffix == '.json':
            return DataFormat.JSON
        elif suffix == '.pkl':
            return DataFormat.PICKLE
        else:
            return DataFormat.CSV  # Default fallback
    
    async def process_data_progressively(
        self,
        filepath: str,
        processor_func: Callable,
        chunk_processor: Optional[Callable] = None
    ) -> List[Any]:
        """Process data progressively chunk by chunk."""
        loader = await self.create_loader(filepath)
        total_chunks = await loader.get_chunk_count()
        
        results = []
        
        for i in range(total_chunks):
            start_index = i * self.config.chunk_size
            end_index = min((i + 1) * self.config.chunk_size, await loader.get_total_size())
            
            # Load chunk
            chunk = await loader.load_chunk(start_index, end_index)
            
            # Process chunk
            if chunk_processor:
                chunk_result = await chunk_processor(chunk.data)
            else:
                chunk_result = await processor_func(chunk.data)
            
            results.append(chunk_result)
            
            # Memory management
            if self._memory_profiler and i % self.config.gc_threshold == 0:
                self._memory_profiler._conservative_memory_cleanup()
        
        return results
    
    async def process_data_streaming(
        self,
        filepath: str,
        processor_func: Callable,
        max_chunks_in_memory: int = 5
    ) -> AsyncIterator[Any]:
        """Process data in streaming fashion."""
        loader = await self.create_loader(filepath)
        total_chunks = await loader.get_chunk_count()
        
        for i in range(total_chunks):
            start_index = i * self.config.chunk_size
            end_index = min((i + 1) * self.config.chunk_size, await loader.get_total_size())
            
            # Load chunk
            chunk = await loader.load_chunk(start_index, end_index)
            
            # Process chunk
            result = await processor_func(chunk.data)
            yield result
            
            # Memory management
            if self._memory_profiler and i % self.config.gc_threshold == 0:
                self._memory_profiler._conservative_memory_cleanup()

# Global progressive data processor
_global_processor: Optional[ProgressiveDataProcessor] = None

def get_progressive_data_processor(config: Optional[ProgressiveLoadingConfig] = None) -> ProgressiveDataProcessor:
    """Get the global progressive data processor."""
    global _global_processor
    
    if _global_processor is None:
        _global_processor = ProgressiveDataProcessor(config)
    
    return _global_processor

def progressive_load(
    filepath: str,
    chunk_size: int = 10000,
    loading_strategy: LoadingStrategy = LoadingStrategy.ADAPTIVE
):
    """Decorator for progressive loading."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            config = ProgressiveLoadingConfig(
                chunk_size=chunk_size,
                loading_strategy=loading_strategy
            )
            processor = ProgressiveDataProcessor(config)
            
            # Process data progressively
            results = await processor.process_data_progressively(
                filepath,
                func,
                **kwargs
            )
            
            return results
        
        return async_wrapper
    return decorator

# Convenience functions
async def load_data_progressively(
    filepath: str,
    processor_func: Callable,
    config: Optional[ProgressiveLoadingConfig] = None
) -> List[Any]:
    """Load and process data progressively."""
    processor = get_progressive_data_processor(config)
    return await processor.process_data_progressively(filepath, processor_func)

async def stream_data_progressively(
    filepath: str,
    processor_func: Callable,
    config: Optional[ProgressiveLoadingConfig] = None
) -> AsyncIterator[Any]:
    """Stream data progressively."""
    processor = get_progressive_data_processor(config)
    async for result in processor.process_data_streaming(filepath, processor_func):
        yield result