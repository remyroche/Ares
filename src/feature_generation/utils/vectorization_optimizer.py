"""
Vectorization Optimizer for Feature Engineering

This module provides advanced vectorization optimizations and hardware utilization
enhancements for the feature engineering pipeline.
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass
from contextlib import contextmanager
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio

# Import optimization components
from ...utils.matrix_operations import get_unified_matrix_operations
from ...utils.hardware.unified_hardware_manager import get_unified_hardware_manager, WorkloadType
from ...utils.matrix_operations.vectorized_core import get_vectorized_processing_core

logger = logging.getLogger(__name__)

@dataclass
class VectorizationConfig:
    """Configuration for vectorization optimization."""
    # Chunking Configuration
    chunk_size: int = 10000
    max_chunk_size: int = 50000
    min_chunk_size: int = 1000
    adaptive_chunking: bool = True
    
    # Memory Management
    memory_limit_gb: float = 8.0
    memory_efficiency_threshold: float = 0.8
    enable_memory_pooling: bool = True
    
    # Parallel Processing
    max_workers: int = None  # Auto-detect
    enable_threading: bool = True
    enable_multiprocessing: bool = True
    thread_pool_size: int = 4
    process_pool_size: int = 2
    
    # Hardware Optimization
    enable_gpu_acceleration: bool = True
    enable_simd_optimization: bool = True
    enable_batch_processing: bool = True
    batch_size: int = 1000
    
    # Vectorization Strategy
    vectorization_strategy: str = "adaptive"  # "aggressive", "conservative", "adaptive"
    enable_auto_vectorization: bool = True
    vectorization_threshold: int = 1000  # Minimum rows for vectorization
    
    # Performance Monitoring
    enable_profiling: bool = False
    profile_memory_usage: bool = True
    profile_execution_time: bool = True

class VectorizationOptimizer:
    """
    Advanced vectorization optimizer for feature engineering operations.
    """
    
    def __init__(self, config: Optional[VectorizationConfig] = None):
        """Initialize the vectorization optimizer."""
        self.config = config or VectorizationConfig()
        self.logger = logger.getChild('VectorizationOptimizer')
        
        # Initialize components
        self.matrix_ops = None
        self.hardware_manager = None
        self.vectorized_core = None
        
        # Performance tracking
        self.performance_stats = {
            'total_operations': 0,
            'vectorized_operations': 0,
            'chunked_operations': 0,
            'parallel_operations': 0,
            'gpu_operations': 0,
            'total_execution_time': 0.0,
            'memory_savings': 0.0,
            'speedup_factor': 1.0
        }
        
        # Initialize components
        self._initialize_components()
        
        self.logger.info("✅ Vectorization Optimizer initialized")
    
    def _initialize_components(self):
        """Initialize optimization components."""
        try:
            # Initialize matrix operations
            if self.config.enable_gpu_acceleration:
                self.matrix_ops = get_unified_matrix_operations()
                self.logger.info("✅ Matrix operations initialized")
            
            # Initialize hardware manager
            self.hardware_manager = get_unified_hardware_manager()
            self.hardware_manager.optimize_for_workload(WorkloadType.FEATURE_ENGINEERING)
            self.logger.info("✅ Hardware manager initialized")
            
            # Initialize vectorized core
            self.vectorized_core = get_vectorized_processing_core()
            self.logger.info("✅ Vectorized core initialized")
            
            # Set optimal worker counts
            if self.config.max_workers is None:
                self.config.max_workers = min(mp.cpu_count(), 8)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Some components not available: {e}")
    
    def optimize_dataframe_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        try:
            if self.vectorized_core:
                return self.vectorized_core.optimize_dataframe_for_processing(df)
            else:
                return self._basic_dataframe_optimization(df)
        except Exception as e:
            self.logger.warning(f"DataFrame optimization failed: {e}")
            return df
    
    def _basic_dataframe_optimization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic DataFrame optimization without vectorized core."""
        optimized_df = df.copy()
        
        # Optimize numeric columns
        for col in optimized_df.select_dtypes(include=[np.number]).columns:
            if optimized_df[col].dtype == np.float64:
                # Check if float32 is sufficient
                if (optimized_df[col].max() < np.finfo(np.float32).max and
                    optimized_df[col].min() > np.finfo(np.float32).min):
                    optimized_df[col] = optimized_df[col].astype(np.float32)
            elif optimized_df[col].dtype == np.int64:
                # Check if int32 is sufficient
                if (optimized_df[col].max() < np.iinfo(np.int32).max and
                    optimized_df[col].min() > np.iinfo(np.int32).min):
                    optimized_df[col] = optimized_df[col].astype(np.int32)
        
        return optimized_df
    
    def vectorized_rolling_operations(self, 
                                    data: pd.DataFrame,
                                    operations: List[str],
                                    windows: List[int],
                                    columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Perform vectorized rolling operations with hardware optimization.
        
        Args:
            data: Input DataFrame
            operations: List of operations ('mean', 'std', 'var', 'min', 'max', 'sum')
            windows: List of window sizes
            columns: Columns to process (None = all numeric columns)
            
        Returns:
            DataFrame with rolling features
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        start_time = time.time()
        
        try:
            # Use vectorized core if available
            if self.vectorized_core:
                result = self.vectorized_core.vectorized_rolling_features(
                    data, windows, columns
                )
                self.performance_stats['vectorized_operations'] += 1
            else:
                result = self._fallback_rolling_operations(data, operations, windows, columns)
            
            execution_time = time.time() - start_time
            self.performance_stats['total_execution_time'] += execution_time
            self.performance_stats['total_operations'] += 1
            
            self.logger.debug(f"Rolling operations completed in {execution_time:.3f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Vectorized rolling operations failed: {e}")
            return data
    
    def _fallback_rolling_operations(self, 
                                   data: pd.DataFrame,
                                   operations: List[str],
                                   windows: List[int],
                                   columns: List[str]) -> pd.DataFrame:
        """Fallback rolling operations without vectorized core."""
        result = data.copy()
        
        for window in windows:
            for col in columns:
                if col in data.columns:
                    series = data[col]
                    
                    for operation in operations:
                        if operation == 'mean':
                            result[f'{col}_rolling_mean_{window}'] = self._vectorbt_rolling_operation(series, "mean", window)
                        elif operation == 'std':
                            result[f'{col}_rolling_std_{window}'] = self._vectorbt_rolling_operation(series, "std", window)
                        elif operation == 'var':
                            result[f'{col}_rolling_var_{window}'] = self._vectorbt_rolling_operation(series, "var", window)
                        elif operation == 'min':
                            result[f'{col}_rolling_min_{window}'] = self._vectorbt_rolling_operation(series, "min", window)
                        elif operation == 'max':
                            result[f'{col}_rolling_max_{window}'] = self._vectorbt_rolling_operation(series, "max", window)
                        elif operation == 'sum':
                            result[f'{col}_rolling_sum_{window}'] = self._vectorbt_rolling_operation(series, "sum", window)
        
        return result
    
    def vectorized_correlation_analysis(self, 
                                      data: pd.DataFrame,
                                      method: str = 'pearson') -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Perform vectorized correlation analysis with hardware optimization.
        
        Args:
            data: Input DataFrame
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Tuple of (correlation_matrix, feature_importance)
        """
        start_time = time.time()
        
        try:
            if self.vectorized_core:
                corr_matrix, feature_importance = self.vectorized_core.matrix_correlation_analysis(
                    data, method
                )
                self.performance_stats['vectorized_operations'] += 1
            else:
                corr_matrix, feature_importance = self._fallback_correlation_analysis(data, method)
            
            execution_time = time.time() - start_time
            self.performance_stats['total_execution_time'] += execution_time
            self.performance_stats['total_operations'] += 1
            
            self.logger.debug(f"Correlation analysis completed in {execution_time:.3f}s")
            return corr_matrix, feature_importance
            
        except Exception as e:
            self.logger.error(f"Vectorized correlation analysis failed: {e}")
            return np.array([[1.0]]), pd.DataFrame()
    
    def _fallback_correlation_analysis(self, 
                                     data: pd.DataFrame,
                                     method: str) -> Tuple[np.ndarray, pd.DataFrame]:
        """Fallback correlation analysis without vectorized core."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.shape[1] < 2:
            return np.array([[1.0]]), pd.DataFrame()
        
        # Compute correlation matrix
        if method == 'pearson':
            corr_matrix = numeric_data.corr().values
        elif method == 'spearman':
            corr_matrix = numeric_data.corr(method='spearman').values
        else:  # kendall
            corr_matrix = numeric_data.corr(method='kendall').values
        
        # Compute feature importance
        feature_importance = pd.DataFrame({
            'feature': numeric_data.columns,
            'mean_abs_corr': np.abs(corr_matrix).mean(axis=1),
            'max_corr': np.abs(corr_matrix).max(axis=1),
            'corr_std': np.abs(corr_matrix).std(axis=1)
        })
        
        return corr_matrix, feature_importance
    
    def parallel_feature_generation(self, 
                                  data: pd.DataFrame,
                                  feature_functions: List[Callable],
                                  chunk_size: Optional[int] = None) -> pd.DataFrame:
        """
        Generate features in parallel with optimal chunking.
        
        Args:
            data: Input DataFrame
            feature_functions: List of feature generation functions
            chunk_size: Optional chunk size (auto-detect if None)
            
        Returns:
            DataFrame with generated features
        """
        if chunk_size is None:
            chunk_size = self._calculate_optimal_chunk_size(data)
        
        start_time = time.time()
        
        try:
            # Split data into chunks
            chunks = self._split_dataframe(data, chunk_size)
            
            # Process chunks in parallel
            if self.config.enable_multiprocessing and len(chunks) > 1:
                results = self._process_chunks_multiprocessing(chunks, feature_functions)
            elif self.config.enable_threading and len(chunks) > 1:
                results = self._process_chunks_threading(chunks, feature_functions)
            else:
                results = self._process_chunks_sequential(chunks, feature_functions)
            
            # Combine results
            combined_result = self._combine_chunk_results(results, data.index)
            
            execution_time = time.time() - start_time
            self.performance_stats['total_execution_time'] += execution_time
            self.performance_stats['parallel_operations'] += 1
            self.performance_stats['chunked_operations'] += len(chunks)
            
            self.logger.debug(f"Parallel feature generation completed in {execution_time:.3f}s")
            return combined_result
            
        except Exception as e:
            self.logger.error(f"Parallel feature generation failed: {e}")
            return data
    
    def _calculate_optimal_chunk_size(self, data: pd.DataFrame) -> int:
        """Calculate optimal chunk size based on data and system resources."""
        if not self.config.adaptive_chunking:
            return self.config.chunk_size
        
        # Base chunk size on data size and available memory
        data_size = data.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        available_memory = self._get_available_memory()  # MB
        
        # Calculate optimal chunk size
        if available_memory > 0:
            optimal_size = min(
                max(int(data_size * 0.1), self.config.min_chunk_size),
                min(int(available_memory * 0.1), self.config.max_chunk_size)
            )
        else:
            optimal_size = self.config.chunk_size
        
        return optimal_size
    
    def _get_available_memory(self) -> float:
        """Get available memory in MB."""
        try:
            import psutil
            return psutil.virtual_memory().available / (1024 * 1024)
        except ImportError:
            return 1024.0  # Default fallback

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
