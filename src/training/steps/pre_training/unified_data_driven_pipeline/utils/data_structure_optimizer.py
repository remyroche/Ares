"""
Data Structure Optimizer with int32 Downcasting

Implements optimized data structures, chunked operations, and comprehensive
int32/float32 downcasting throughout the feature interaction generation process.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Iterator
from dataclasses import dataclass
import logging
import gc
import warnings
from contextlib import contextmanager
import psutil

from src.utils.tprint import tprint

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """Configuration for data structure optimization."""
    
    # Data type optimization
    enable_int32_downcasting: bool = True
    enable_float32_downcasting: bool = True
    enable_categorical_optimization: bool = True
    enable_sparse_optimization: bool = True
    
    # Chunked operations
    chunk_size: int = 10000
    enable_chunked_processing: bool = True
    memory_threshold_gb: float = 2.0
    
    # Memory optimization
    enable_memory_mapping: bool = True
    enable_compression: bool = True
    compression_level: int = 6
    
    # Performance
    enable_parallel_processing: bool = True
    max_workers: int = 4

class DataTypeOptimizer:
    """Optimizes data types for memory efficiency."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logger.getChild('DataTypeOptimizer')
        
        # Type mapping for optimization
        self.int32_limits = {
            'min': np.iinfo(np.int32).min,
            'max': np.iinfo(np.int32).max
        }
        
        self.float32_limits = {
            'min': np.finfo(np.float32).min,
            'max': np.finfo(np.float32).max
        }
    
    def downcast_to_int32(self, data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
        """Downcast integer data to int32 where possible."""
        if not self.config.enable_int32_downcasting:
            return data
        
        tprint("🔄 [OPTIMIZE] Downcasting to int32")
        
        if isinstance(data, pd.DataFrame):
            return self._downcast_dataframe_to_int32(data)
        elif isinstance(data, pd.Series):
            return self._downcast_series_to_int32(data)
        elif isinstance(data, np.ndarray):
            return self._downcast_array_to_int32(data)
        
        return data
    
    def _downcast_dataframe_to_int32(self, df: pd.DataFrame) -> pd.DataFrame:
        """Downcast DataFrame columns to int32."""
        optimized_df = df.copy()
        downcast_count = 0
        
        for col in optimized_df.columns:
            if optimized_df[col].dtype == 'int64':
                if self._can_downcast_to_int32(optimized_df[col]):
                    optimized_df[col] = optimized_df[col].astype(np.int32)
                    downcast_count += 1
                    tprint(f"🔧 [OPTIMIZE] Downcasted {col} from int64 to int32")
        
        tprint(f"✅ [OPTIMIZE] Downcasted {downcast_count} columns to int32")
        return optimized_df
    
    def _downcast_series_to_int32(self, series: pd.Series) -> pd.Series:
        """Downcast Series to int32."""
        if series.dtype == 'int64' and self._can_downcast_to_int32(series):
            return series.astype(np.int32)
        return series
    
    def _downcast_array_to_int32(self, array: np.ndarray) -> np.ndarray:
        """Downcast numpy array to int32."""
        if array.dtype == 'int64' and self._can_downcast_to_int32(array):
            return array.astype(np.int32)
        return array
    
    def _can_downcast_to_int32(self, data: Union[pd.Series, np.ndarray]) -> bool:
        """Check if data can be safely downcast to int32."""
        try:
            if isinstance(data, pd.Series):
                data_values = data.dropna()
            else:
                data_values = data
            
            if len(data_values) == 0:
                return True
            
            return (data_values.min() >= self.int32_limits['min'] and 
                   data_values.max() <= self.int32_limits['max'])
        
        except Exception as e:
            tprint(f"⚠️ [OPTIMIZE] Error checking int32 downcast: {e}")
            return False
    
    def downcast_to_float32(self, data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
        """Downcast float data to float32 where possible."""
        if not self.config.enable_float32_downcasting:
            return data
        
        tprint("🔄 [OPTIMIZE] Downcasting to float32")
        
        if isinstance(data, pd.DataFrame):
            return self._downcast_dataframe_to_float32(data)
        elif isinstance(data, pd.Series):
            return self._downcast_series_to_float32(data)
        elif isinstance(data, np.ndarray):
            return self._downcast_array_to_float32(data)
        
        return data
    
    def _downcast_dataframe_to_float32(self, df: pd.DataFrame) -> pd.DataFrame:
        """Downcast DataFrame columns to float32."""
        optimized_df = df.copy()
        downcast_count = 0
        
        for col in optimized_df.columns:
            if optimized_df[col].dtype == 'float64':
                if self._can_downcast_to_float32(optimized_df[col]):
                    optimized_df[col] = optimized_df[col].astype(np.float32)
                    downcast_count += 1
                    tprint(f"🔧 [OPTIMIZE] Downcasted {col} from float64 to float32")
        
        tprint(f"✅ [OPTIMIZE] Downcasted {downcast_count} columns to float32")
        return optimized_df
    
    def _downcast_series_to_float32(self, series: pd.Series) -> pd.Series:
        """Downcast Series to float32."""
        if series.dtype == 'float64' and self._can_downcast_to_float32(series):
            return series.astype(np.float32)
        return series
    
    def _downcast_array_to_float32(self, array: np.ndarray) -> np.ndarray:
        """Downcast numpy array to float32."""
        if array.dtype == 'float64' and self._can_downcast_to_float32(array):
            return array.astype(np.float32)
        return array
    
    def _can_downcast_to_float32(self, data: Union[pd.Series, np.ndarray]) -> bool:
        """Check if data can be safely downcast to float32."""
        try:
            if isinstance(data, pd.Series):
                data_values = data.dropna()
            else:
                data_values = data
            
            if len(data_values) == 0:
                return True
            
            # Check for NaN and inf values
            if np.any(np.isnan(data_values)) or np.any(np.isinf(data_values)):
                return False
            
            return (data_values.min() >= self.float32_limits['min'] and 
                   data_values.max() <= self.float32_limits['max'])
        
        except Exception as e:
            tprint(f"⚠️ [OPTIMIZE] Error checking float32 downcast: {e}")
            return False
    
    def optimize_categorical_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize categorical columns for memory efficiency."""
        if not self.config.enable_categorical_optimization:
            return data
        
        tprint("🔄 [OPTIMIZE] Optimizing categorical columns")
        
        optimized_df = data.copy()
        optimized_count = 0
        
        for col in optimized_df.columns:
            if optimized_df[col].dtype == 'object':
                unique_ratio = optimized_df[col].nunique() / len(optimized_df)
                
                # Convert to categorical if low cardinality
                if unique_ratio < 0.5:  # Less than 50% unique values
                    optimized_df[col] = optimized_df[col].astype('category')
                    optimized_count += 1
                    tprint(f"🔧 [OPTIMIZE] Converted {col} to categorical (cardinality: {optimized_df[col].nunique()})")
        
        tprint(f"✅ [OPTIMIZE] Optimized {optimized_count} categorical columns")
        return optimized_df
    
    def optimize_sparse_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize sparse data using sparse data types."""
        if not self.config.enable_sparse_optimization:
            return data
        
        tprint("🔄 [OPTIMIZE] Optimizing sparse data")
        
        optimized_df = data.copy()
        optimized_count = 0
        
        for col in optimized_df.columns:
            if optimized_df[col].dtype in ['float64', 'int64']:
                # Check if column is sparse (>50% zeros)
                zero_ratio = (optimized_df[col] == 0).sum() / len(optimized_df)
                
                if zero_ratio > 0.5:  # More than 50% zeros
                    if optimized_df[col].dtype == 'float64':
                        optimized_df[col] = optimized_df[col].astype(pd.SparseDtype("float32", 0))
                    else:
                        optimized_df[col] = optimized_df[col].astype(pd.SparseDtype("int32", 0))
                    
                    optimized_count += 1
                    tprint(f"🔧 [OPTIMIZE] Converted {col} to sparse format (zero ratio: {zero_ratio:.2f})")
        
        tprint(f"✅ [OPTIMIZE] Optimized {optimized_count} sparse columns")
        return optimized_df

class ChunkedProcessor:
    """Handles chunked data processing for memory efficiency."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logger.getChild('ChunkedProcessor')
    
    def should_use_chunked_processing(self, data: pd.DataFrame) -> bool:
        """Determine if data should use chunked processing."""
        memory_usage_gb = data.memory_usage(deep=True).sum() / (1024**3)
        should_use = memory_usage_gb > self.config.memory_threshold_gb
        
        tprint(f"🔍 [CHUNK] Memory usage: {memory_usage_gb:.2f} GB, Chunked processing: {'Yes' if should_use else 'No'}")
        return should_use
    
    def create_chunk_iterator(self, data: pd.DataFrame, chunk_size: Optional[int] = None) -> Iterator[pd.DataFrame]:
        """Create iterator for chunked data processing."""
        if chunk_size is None:
            chunk_size = self.config.chunk_size
        
        tprint(f"📊 [CHUNK] Creating chunk iterator with size: {chunk_size:,}")
        
        for i in range(0, len(data), chunk_size):
            yield data.iloc[i:i+chunk_size].copy()
    
    def process_chunks(self, data: pd.DataFrame, processor_func: callable, 
                      chunk_size: Optional[int] = None, **kwargs) -> pd.DataFrame:
        """Process data in chunks."""
        if chunk_size is None:
            chunk_size = self.config.chunk_size
        
        tprint(f"🔄 [CHUNK] Processing data in chunks (chunk_size: {chunk_size:,})")
        
        if len(data) <= chunk_size:
            # Process all at once
            return processor_func(data, **kwargs)
        
        results = []
        num_chunks = (len(data) + chunk_size - 1) // chunk_size
        
        for i, chunk in enumerate(self.create_chunk_iterator(data, chunk_size)):
            chunk_num = i + 1
            tprint(f"📊 [CHUNK] Processing chunk {chunk_num}/{num_chunks}")
            
            chunk_result = processor_func(chunk, **kwargs)
            results.append(chunk_result)
            
            # Memory cleanup after each chunk
            del chunk
            gc.collect()
        
        tprint(f"✅ [CHUNK] Chunked processing completed: {num_chunks} chunks")
        
        # Combine results
        combined_result = pd.concat(results, ignore_index=True)
        del results
        gc.collect()
        
        return combined_result
    
    def parallel_chunk_processing(self, data: pd.DataFrame, processor_func: callable,
                                chunk_size: Optional[int] = None, max_workers: Optional[int] = None,
                                **kwargs) -> pd.DataFrame:
        """Process chunks in parallel."""
        if not self.config.enable_parallel_processing:
            return self.process_chunks(data, processor_func, chunk_size, **kwargs)
        
        if chunk_size is None:
            chunk_size = self.config.chunk_size
        
        if max_workers is None:
            max_workers = self.config.max_workers
        
        tprint(f"🔄 [CHUNK] Parallel chunk processing (workers: {max_workers})")
        
        # Create chunks
        chunks = list(self.create_chunk_iterator(data, chunk_size))
        
        # Process chunks in parallel
        from concurrent.futures import ThreadPoolExecutor
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(processor_func, chunk, **kwargs) for chunk in chunks]
            results = [future.result() for future in futures]
        
        # Combine results
        combined_result = pd.concat(results, ignore_index=True)
        
        tprint(f"✅ [CHUNK] Parallel chunk processing completed")
        
        return combined_result

class DataStructureOptimizer:
    """Main data structure optimizer."""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.logger = logger.getChild('DataStructureOptimizer')
        
        # Initialize components
        self.type_optimizer = DataTypeOptimizer(self.config)
        self.chunked_processor = ChunkedProcessor(self.config)
        
        # Performance tracking
        self.optimization_stats = {
            'total_optimizations': 0,
            'int32_downcasts': 0,
            'float32_downcasts': 0,
            'categorical_optimizations': 0,
            'sparse_optimizations': 0,
            'chunked_operations': 0
        }
        
        tprint("🚀 [OPTIMIZE] Data Structure Optimizer initialized")
    
    def optimize_dataframe(self, data: pd.DataFrame, 
                          apply_chunking: bool = True) -> pd.DataFrame:
        """Comprehensive DataFrame optimization."""
        tprint("🔄 [OPTIMIZE] Starting comprehensive DataFrame optimization")
        
        self.optimization_stats['total_optimizations'] += 1
        
        # Step 1: Downcast to int32
        optimized_data = self.type_optimizer.downcast_to_int32(data)
        self.optimization_stats['int32_downcasts'] += 1
        
        # Step 2: Downcast to float32
        optimized_data = self.type_optimizer.downcast_to_float32(optimized_data)
        self.optimization_stats['float32_downcasts'] += 1
        
        # Step 3: Optimize categorical columns
        optimized_data = self.type_optimizer.optimize_categorical_columns(optimized_data)
        self.optimization_stats['categorical_optimizations'] += 1
        
        # Step 4: Optimize sparse data
        optimized_data = self.type_optimizer.optimize_sparse_data(optimized_data)
        self.optimization_stats['sparse_optimizations'] += 1
        
        # Calculate memory savings
        original_memory = data.memory_usage(deep=True).sum()
        optimized_memory = optimized_data.memory_usage(deep=True).sum()
        memory_savings = (original_memory - optimized_memory) / original_memory * 100
        
        tprint(f"✅ [OPTIMIZE] DataFrame optimization completed")
        tprint(f"📊 [OPTIMIZE] Memory savings: {memory_savings:.1f}% ({original_memory/1024**2:.1f} MB -> {optimized_memory/1024**2:.1f} MB)")
        
        return optimized_data
    
    def optimize_array(self, array: np.ndarray) -> np.ndarray:
        """Optimize numpy array data types."""
        tprint("🔄 [OPTIMIZE] Optimizing numpy array")
        
        # Downcast to int32
        optimized_array = self.type_optimizer.downcast_to_int32(array)
        
        # Downcast to float32
        optimized_array = self.type_optimizer.downcast_to_float32(optimized_array)
        
        # Calculate memory savings
        original_memory = array.nbytes
        optimized_memory = optimized_array.nbytes
        memory_savings = (original_memory - optimized_memory) / original_memory * 100
        
        tprint(f"✅ [OPTIMIZE] Array optimization completed")
        tprint(f"📊 [OPTIMIZE] Memory savings: {memory_savings:.1f}% ({original_memory/1024**2:.1f} MB -> {optimized_memory/1024**2:.1f} MB)")
        
        return optimized_array
    
    def process_with_optimization(self, data: pd.DataFrame, processor_func: callable,
                                **kwargs) -> pd.DataFrame:
        """Process data with automatic optimization and chunking."""
        tprint("🔄 [OPTIMIZE] Processing with automatic optimization")
        
        # Optimize data first
        optimized_data = self.optimize_dataframe(data)
        
        # Check if chunked processing is needed
        if self.chunked_processor.should_use_chunked_processing(optimized_data):
            self.optimization_stats['chunked_operations'] += 1
            result = self.chunked_processor.process_chunks(optimized_data, processor_func, **kwargs)
        else:
            result = processor_func(optimized_data, **kwargs)
        
        tprint("✅ [OPTIMIZE] Processing with optimization completed")
        
        return result
    
    def parallel_process_with_optimization(self, data: pd.DataFrame, processor_func: callable,
                                         **kwargs) -> pd.DataFrame:
        """Process data with optimization and parallel chunking."""
        tprint("🔄 [OPTIMIZE] Parallel processing with optimization")
        
        # Optimize data first
        optimized_data = self.optimize_dataframe(data)
        
        # Use parallel chunked processing
        if self.chunked_processor.should_use_chunked_processing(optimized_data):
            self.optimization_stats['chunked_operations'] += 1
            result = self.chunked_processor.parallel_chunk_processing(optimized_data, processor_func, **kwargs)
        else:
            result = processor_func(optimized_data, **kwargs)
        
        tprint("✅ [OPTIMIZE] Parallel processing with optimization completed")
        
        return result
    
    def get_memory_usage(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, float]:
        """Get memory usage statistics."""
        if isinstance(data, pd.DataFrame):
            total_memory = data.memory_usage(deep=True).sum()
            memory_by_dtype = data.memory_usage(deep=True, by_dtype=True)
        else:
            total_memory = data.nbytes
            memory_by_dtype = {str(data.dtype): total_memory}
        
        return {
            'total_mb': total_memory / (1024**2),
            'total_gb': total_memory / (1024**3),
            'by_dtype': {str(dtype): memory / (1024**2) for dtype, memory in memory_by_dtype.items()}
        }
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        stats = self.optimization_stats.copy()
        
        return stats
    
    @contextmanager
    def optimization_context(self, operation_name: str):
        """Context manager for optimization operations."""
        initial_memory = psutil.virtual_memory().used / (1024**3)
        
        try:
            tprint(f"🔄 [OPTIMIZE] Starting optimization context: {operation_name}")
            yield self
            
        finally:
            final_memory = psutil.virtual_memory().used / (1024**3)
            memory_delta = final_memory - initial_memory
            
            tprint(f"✅ [OPTIMIZE] Completed optimization context: {operation_name} (delta: {memory_delta:+.2f} GB)")
            
            # Cleanup if memory usage increased significantly
            if memory_delta > 1.0:
                gc.collect()
    
    def cleanup(self):
        """Clean up resources."""
        tprint("🧹 [OPTIMIZE] Cleaning up data structure optimizer")
        
        # Clear optimization statistics
        self.optimization_stats.clear()
        
        # Final garbage collection
        gc.collect()
        
        tprint("✅ [OPTIMIZE] Data structure optimizer cleanup completed")
