from ..standardized_parquet_handler import standardized_parquet_handler
#!/usr/bin/env python3
"""Step03 Memory Management Utility.

Chunked processing and memory optimization for large feature matrices
and datasets to prevent memory issues during regime discovery.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Iterator, Tuple, Optional, Union, List, Callable
import gc
import psutil
import logging
from pathlib import Path

import pickle
from contextlib import contextmanager
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MemoryManager:
    """Memory management utility for chunked processing."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize memory manager with configuration."""
        self.config = config or {}
        self.chunk_size = self.config.get('chunk_size', 10000)
        self.max_memory_usage_gb = self.config.get('max_memory_usage_gb', 8.0)
        self.enable_garbage_collection = self.config.get('enable_garbage_collection', True)
        self.gc_frequency = self.config.get('gc_frequency', 1000)
        self.use_memory_mapping = self.config.get('use_memory_mapping', False)
        self.temp_dir = Path(self.config.get('temp_dir', 'temp'))
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        self._operation_count = 0
        self._temp_files = []
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_gb': memory_info.rss / (1024**3),  # Resident Set Size
            'vms_gb': memory_info.vms / (1024**3),  # Virtual Memory Size
            'percent': process.memory_percent(),
            'available_gb': psutil.virtual_memory().available / (1024**3)
        }
    
    def check_memory_limit(self) -> bool:
        """Check if memory usage is within limits."""
        memory_usage = self.get_memory_usage()
        return memory_usage['rss_gb'] < self.max_memory_usage_gb
    
    def force_garbage_collection(self):
        """Force garbage collection if enabled."""
        if self.enable_garbage_collection:
            gc.collect()
            self._operation_count = 0
    
    def maybe_garbage_collect(self):
        """Run garbage collection if frequency threshold reached."""
        self._operation_count += 1
        if self._operation_count >= self.gc_frequency:
            self.force_garbage_collection()
    
    def chunk_dataframe(self, df: pd.DataFrame, chunk_size: Optional[int] = None) -> Iterator[pd.DataFrame]:
        """Chunk a DataFrame into smaller pieces."""
        if chunk_size is None:
            chunk_size = self.chunk_size
        
        total_rows = len(df)
        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            yield df.iloc[start_idx:end_idx].copy()
    
    def chunk_array(self, array: np.ndarray, chunk_size: Optional[int] = None) -> Iterator[np.ndarray]:
        """Chunk a numpy array into smaller pieces."""
        if chunk_size is None:
            chunk_size = self.chunk_size
        
        total_rows = len(array)
        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            yield array[start_idx:end_idx].copy()
    
    def process_in_chunks(self, 
                         data: Union[pd.DataFrame, np.ndarray], 
                         process_func: Callable,
                         chunk_size: Optional[int] = None,
                         **kwargs) -> List[Any]:
        """Process data in chunks using a processing function."""
        if chunk_size is None:
            chunk_size = self.chunk_size
        
        results = []
        chunk_iter = self.chunk_dataframe(data) if isinstance(data, pd.DataFrame) else self.chunk_array(data, chunk_size)
        
        for i, chunk in enumerate(chunk_iter):
            logger.debug(f"Processing chunk {i+1}, size: {len(chunk)}")
            
            # Check memory before processing
            if not self.check_memory_limit():
                logger.warning("Memory limit exceeded, forcing garbage collection")
                self.force_garbage_collection()
            
            # Process chunk
            try:
                result = process_func(chunk, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {e}")
                raise
            
            # Cleanup chunk
            del chunk
            self.maybe_garbage_collect()
        
        return results
    
    def reduce_chunk_results(self, 
                           results: List[Any], 
                           reduce_func: Callable,
                           **kwargs) -> Any:
        """Reduce chunk results using a reduction function."""
        if not results:
            return None
        
        if len(results) == 1:
            return results[0]
        
        # Process results in pairs to avoid memory issues
        current_results = results.copy()
        
        while len(current_results) > 1:
            new_results = []
            for i in range(0, len(current_results), 2):
                if i + 1 < len(current_results):
                    # Combine two results
                    combined = reduce_func(current_results[i], current_results[i + 1], **kwargs)
                    new_results.append(combined)
                else:
                    # Single result left
                    new_results.append(current_results[i])
            
            current_results = new_results
            self.maybe_garbage_collect()
        
        return current_results[0]
    
    def create_temp_file(self, suffix: str = '.pkl') -> Path:
        """Create a temporary file for data storage."""
        temp_file = self.temp_dir / f"temp_{len(self._temp_files)}{suffix}"
        self._temp_files.append(temp_file)
        return temp_file
    
    def save_to_temp(self, data: Any, suffix: str = '.pkl') -> Path:
        """Save data to a temporary file."""
        temp_file = self.create_temp_file(suffix)
        
        if suffix == '.pkl':
            with open(temp_file, 'wb') as f:
                pickle.dump(data, f)
        elif suffix == '.parquet' and isinstance(data, pd.DataFrame):
            standardized_parquet_handler.write_parquet_standardized(data, temp_file)
        elif suffix == '.npy' and isinstance(data, np.ndarray):
            np.save(temp_file, data)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
        
        return temp_file
    
    def load_from_temp(self, filepath: Path) -> Any:
        """Load data from a temporary file."""
        if filepath.suffix == '.pkl':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        elif filepath.suffix == '.parquet':
            return standardized_parquet_handler.read_parquet_standardized(filepath)
        elif filepath.suffix == '.npy':
            return np.load(filepath)
        else:
            raise ValueError(f"Unsupported file type: {filepath.suffix}")
    
    def cleanup_temp_files(self):
        """Clean up all temporary files."""
        for temp_file in self._temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception as e:
                logger.warning(f"Could not delete temp file {temp_file}: {e}")
        
        self._temp_files.clear()
    
    @contextmanager
    def memory_context(self, operation_name: str = "operation"):
        """Context manager for memory-aware operations."""
        initial_memory = self.get_memory_usage()
        logger.debug(f"Starting {operation_name}, initial memory: {initial_memory['rss_gb']:.2f} GB")
        
        try:
            yield self
        finally:
            final_memory = self.get_memory_usage()
            memory_delta = final_memory['rss_gb'] - initial_memory['rss_gb']
            logger.debug(f"Completed {operation_name}, memory delta: {memory_delta:+.2f} GB")
            
            if memory_delta > 1.0:  # More than 1GB increase
                logger.warning(f"Large memory increase detected: {memory_delta:.2f} GB")
                self.force_garbage_collection()
    
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage."""
        original_memory = df.memory_usage(deep=True).sum() / (1024**2)  # MB
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if df[col].dtype == 'int64':
                if col_min >= 0:
                    if col_max < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif col_max < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif col_max < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                else:
                    if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
            
            elif df[col].dtype == 'float64':
                df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Optimize categorical columns
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')
        
        optimized_memory = df.memory_usage(deep=True).sum() / (1024**2)  # MB
        reduction = (original_memory - optimized_memory) / original_memory * 100
        
        logger.debug(f"DataFrame memory optimized: {original_memory:.1f} MB -> {optimized_memory:.1f} MB ({reduction:.1f}% reduction)")
        
        return df
    
    def process_large_features(self, 
                             data: pd.DataFrame, 
                             feature_func: Callable,
                             chunk_size: Optional[int] = None,
                             **kwargs) -> pd.DataFrame:
        """Process large feature matrices in chunks."""
        if chunk_size is None:
            chunk_size = self.chunk_size
        
        logger.info(f"Processing large feature matrix in chunks of {chunk_size}")
        
        # Process in chunks
        chunk_results = []
        chunk_iter = self.chunk_dataframe(data, chunk_size)
        
        for i, chunk in enumerate(chunk_iter):
            logger.debug(f"Processing feature chunk {i+1}")
            
            # Process chunk
            chunk_features = feature_func(chunk, **kwargs)
            chunk_results.append(chunk_features)
            
            # Cleanup
            del chunk
            self.maybe_garbage_collect()
        
        # Combine results
        logger.info("Combining chunk results")
        combined_features = pd.concat(chunk_results, ignore_index=True)
        
        # Cleanup chunk results
        del chunk_results
        self.force_garbage_collection()
        
        return combined_features
    
    def __del__(self):
        """Cleanup on destruction."""
        self.cleanup_temp_files()

# Global memory manager instance
_global_memory_manager = MemoryManager()

def get_memory_manager(config: Optional[Dict] = None) -> MemoryManager:
    """Get global memory manager instance."""
    if config:
        _global_memory_manager.config.update(config)
    return _global_memory_manager

@contextmanager
def memory_aware_processing(operation_name: str = "operation", config: Optional[Dict] = None):
    """Context manager for memory-aware processing."""
    manager = get_memory_manager(config)
    with manager.memory_context(operation_name):
        yield manager

def chunked_process(data: Union[pd.DataFrame, np.ndarray], 
                   process_func: Callable,
                   chunk_size: Optional[int] = None,
                   config: Optional[Dict] = None,
                   **kwargs) -> List[Any]:
    """Process data in chunks using global memory manager."""
    manager = get_memory_manager(config)
    return manager.process_in_chunks(data, process_func, chunk_size, **kwargs)

def optimize_dataframe_memory(df: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
    """Optimize DataFrame memory usage using global memory manager."""
    manager = get_memory_manager(config)
    return manager.optimize_dataframe(df)