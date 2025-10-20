"""
Enhanced Memory & Data Processing Optimization

This module provides comprehensive memory optimization and data processing
enhancements for the HDBSCAN clustering pipeline.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, ContextManager
from dataclasses import dataclass
import time
import gc
import psutil
from contextlib import contextmanager

# Import utility functions
from src.utils.common_operations import (
    optimize_dataframe_memory,
    safe_divide,
    safe_log,
    safe_sqrt,
    safe_mean,
    safe_std,
    get_memory_usage,
    safe_merge,
    safe_concat
)
from src.utils.math_validation import validate_finite
from src.utils.common_utilities import safe_dataframe_operation
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance, tprint_progress, tprint_timer,
    tprint_logged, LogLevel
)

logger = logging.getLogger(__name__)

@dataclass
class MemoryOptimizationConfig:
    """Configuration for memory optimization."""
    # Memory management
    max_memory_gb: float = 8.0
    memory_cleanup_threshold: float = 0.8
    chunk_size: int = 1000
    
    # Data processing
    enable_memory_optimization: bool = True
    enable_data_validation: bool = True
    enable_safe_operations: bool = True
    
    # Performance monitoring
    enable_memory_monitoring: bool = True
    memory_check_interval: float = 1.0  # seconds

class EnhancedMemoryOptimizer:
    """
    Enhanced memory optimizer with comprehensive data processing optimizations.
    """
    
    @tprint_logged(LogLevel.INFO, include_args=True)
    def __init__(self, config: Optional[MemoryOptimizationConfig] = None):
        """Initialize the enhanced memory optimizer."""
        start_time = time.perf_counter()
        initial_memory = get_memory_usage()
        
        self.config = config or MemoryOptimizationConfig()
        
        # Memory tracking
        self.memory_history = []
        self.peak_memory_usage = initial_memory
        self.optimization_stats = {
            'memory_optimizations': 0,
            'data_validations': 0,
            'safe_operations': 0,
            'memory_savings_mb': 0.0,
            'processing_time': 0.0,
            'initialization_time': 0.0,
            'initial_memory_mb': initial_memory
        }
        
        # Track initialization performance
        init_time = time.perf_counter() - start_time
        self.optimization_stats['initialization_time'] = init_time
        
        tprint_success("✅ EnhancedMemoryOptimizer initialized")
        tprint_debug(f"Initial memory: {initial_memory:.2f}MB, Init time: {init_time:.3f}s")
        tprint_debug(f"Config: max_memory_gb={self.config.max_memory_gb}, chunk_size={self.config.chunk_size}")
        
        logger.info("✅ EnhancedMemoryOptimizer initialized")
    
    @contextmanager
    def memory_monitor(self, operation_name: str = "operation"):
        """Context manager for memory monitoring."""
        if not self.config.enable_memory_monitoring:
            yield
            return
        
        start_memory = self._get_current_memory_usage()
        start_time = time.time()
        
        try:
            yield
        finally:
            end_memory = self._get_current_memory_usage()
            end_time = time.time()
            
            memory_delta = end_memory - start_memory
            processing_time = end_time - start_time
            
            logger.info(f"🧠 Memory monitor - {operation_name}: "
                       f"Memory delta: {memory_delta:.2f}MB, "
                       f"Time: {processing_time:.3f}s")
            
            # Update peak memory usage
            self.peak_memory_usage = max(self.peak_memory_usage, end_memory)
            
            # Store memory history
            self.memory_history.append({
                'operation': operation_name,
                'start_memory': start_memory,
                'end_memory': end_memory,
                'memory_delta': memory_delta,
                'processing_time': processing_time,
                'timestamp': time.time()
            })
    
    def optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage with enhanced features."""
        with self.memory_monitor("dataframe_memory_optimization"):
            original_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
            
            # Use the utility function
            optimized_df = optimize_dataframe_memory(df)
            
            # Additional optimizations
            optimized_df = self._apply_advanced_memory_optimizations(optimized_df)
            
            # Calculate memory savings
            optimized_memory = optimized_df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
            memory_savings = original_memory - optimized_memory
            
            self.optimization_stats['memory_optimizations'] += 1
            self.optimization_stats['memory_savings_mb'] += memory_savings
            
            logger.info(f"💾 Memory optimization: {memory_savings:.2f}MB saved "
                       f"({memory_savings/original_memory*100:.1f}% reduction)")
            
            return optimized_df
    
    def _apply_advanced_memory_optimizations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply advanced memory optimizations."""
        optimized_df = df.copy()
        
        # Convert datetime columns to appropriate types
        for col in optimized_df.select_dtypes(include=['datetime64']).columns:
            if optimized_df[col].dt.tz is None:
                optimized_df[col] = pd.to_datetime(optimized_df[col], utc=True)
        
        # Optimize categorical columns
        for col in optimized_df.select_dtypes(include=['object']).columns:
            if optimized_df[col].nunique() / len(optimized_df) < 0.5:
                optimized_df[col] = optimized_df[col].astype('category')
        
        # Remove unnecessary columns with all NaN values
        nan_columns = optimized_df.columns[optimized_df.isnull().all()].tolist()
        if nan_columns:
            optimized_df = optimized_df.drop(columns=nan_columns)
            logger.info(f"🗑️ Removed {len(nan_columns)} columns with all NaN values")
        
        return optimized_df
    
    def safe_divide(self, a: Union[pd.Series, np.ndarray, float], 
                   b: Union[pd.Series, np.ndarray, float], 
                   fill_value: float = 0.0) -> Union[pd.Series, np.ndarray]:
        """Safely divide with validation."""
        if self.config.enable_safe_operations:
            self.optimization_stats['safe_operations'] += 1
        
        result = safe_divide(a, b, fill_value)
        
        if self.config.enable_data_validation:
            result = self._validate_result(result, "safe_divide")
        
        return result
    
    def safe_log(self, x: Union[pd.Series, np.ndarray], 
                 base: float = np.e, 
                 fill_value: float = 0.0) -> Union[pd.Series, np.ndarray]:
        """Safely compute logarithm with validation."""
        if self.config.enable_safe_operations:
            self.optimization_stats['safe_operations'] += 1
        
        result = safe_log(x, base, fill_value)
        
        if self.config.enable_data_validation:
            result = self._validate_result(result, "safe_log")
        
        return result
    
    def safe_sqrt(self, x: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """Safely compute square root with validation."""
        if self.config.enable_safe_operations:
            self.optimization_stats['safe_operations'] += 1
        
        result = safe_sqrt(x)
        
        if self.config.enable_data_validation:
            result = self._validate_result(result, "safe_sqrt")
        
        return result
    
    def safe_merge(self, left: pd.DataFrame, right: pd.DataFrame, 
                   **kwargs) -> pd.DataFrame:
        """Safely merge DataFrames with memory optimization."""
        with self.memory_monitor("safe_merge"):
            result = safe_merge(left, right, **kwargs)
            
            if self.config.enable_memory_optimization:
                result = self.optimize_dataframe_memory(result)
            
            return result
    
    def safe_concat(self, dataframes: List[pd.DataFrame], 
                   **kwargs) -> pd.DataFrame:
        """Safely concatenate DataFrames with memory optimization."""
        with self.memory_monitor("safe_concat"):
            result = safe_concat(dataframes, **kwargs)
            
            if self.config.enable_memory_optimization:
                result = self.optimize_dataframe_memory(result)
            
            return result
    
    def validate_finite(self, data: Union[pd.Series, np.ndarray, pd.DataFrame], 
                       name: str = "data") -> Union[pd.Series, np.ndarray, pd.DataFrame]:
        """Validate data for finite values with enhanced checking."""
        if self.config.enable_data_validation:
            self.optimization_stats['data_validations'] += 1
            
            if isinstance(data, pd.DataFrame):
                # Validate each column
                for col in data.columns:
                    validate_finite(data[col], f"{name}.{col}")
            else:
                validate_finite(data, name)
        
        return data
    
    def _validate_result(self, result: Union[pd.Series, np.ndarray], 
                        operation: str) -> Union[pd.Series, np.ndarray]:
        """Validate operation result."""
        try:
            if isinstance(result, pd.Series):
                # Check for infinite values
                if np.isinf(result).any():
                    logger.warning(f"⚠️ {operation} produced infinite values")
                    result = result.replace([np.inf, -np.inf], np.nan)
                
                # Check for NaN values
                if result.isnull().all():
                    logger.warning(f"⚠️ {operation} produced all NaN values")
            
            elif isinstance(result, np.ndarray):
                # Check for infinite values
                if np.isinf(result).any():
                    logger.warning(f"⚠️ {operation} produced infinite values")
                    result = np.where(np.isinf(result), np.nan, result)
                
                # Check for NaN values
                if np.isnan(result).all():
                    logger.warning(f"⚠️ {operation} produced all NaN values")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Validation failed for {operation}: {e}")
            return result
    
    def _get_current_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            memory_info = get_memory_usage()
            return memory_info['rss']
        except Exception:
            return 0.0
    
    def force_memory_cleanup(self):
        """Force memory cleanup and garbage collection."""
        logger.info("🧹 Forcing memory cleanup")
        
        # Force garbage collection
        collected = gc.collect()
        logger.info(f"✅ Garbage collection: {collected} objects collected")
        
        # Update memory stats
        current_memory = self._get_current_memory_usage()
        self.peak_memory_usage = max(self.peak_memory_usage, current_memory)
        
        logger.info(f"📊 Current memory usage: {current_memory:.2f}MB")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        current_memory = self._get_current_memory_usage()
        
        return {
            'current_memory_mb': current_memory,
            'peak_memory_mb': self.peak_memory_usage,
            'memory_optimizations': self.optimization_stats['memory_optimizations'],
            'data_validations': self.optimization_stats['data_validations'],
            'safe_operations': self.optimization_stats['safe_operations'],
            'memory_savings_mb': self.optimization_stats['memory_savings_mb'],
            'processing_time': self.optimization_stats['processing_time'],
            'memory_history_count': len(self.memory_history)
        }
    
    def reset_stats(self):
        """Reset optimization statistics."""
        self.memory_history = []
        self.peak_memory_usage = 0.0
        self.optimization_stats = {
            'memory_optimizations': 0,
            'data_validations': 0,
            'safe_operations': 0,
            'memory_savings_mb': 0.0,
            'processing_time': 0.0
        }

# Convenience function
def create_enhanced_memory_optimizer(
    max_memory_gb: float = 8.0,
    enable_memory_optimization: bool = True,
    enable_data_validation: bool = True,
    enable_safe_operations: bool = True,
    enable_memory_monitoring: bool = True
) -> EnhancedMemoryOptimizer:
    """
    Create an enhanced memory optimizer with specified configuration.
    
    Args:
        max_memory_gb: Maximum memory usage in GB
        enable_memory_optimization: Enable memory optimization
        enable_data_validation: Enable data validation
        enable_safe_operations: Enable safe operations
        enable_memory_monitoring: Enable memory monitoring
        
    Returns:
        EnhancedMemoryOptimizer instance
    """
    config = MemoryOptimizationConfig(
        max_memory_gb=max_memory_gb,
        enable_memory_optimization=enable_memory_optimization,
        enable_data_validation=enable_data_validation,
        enable_safe_operations=enable_safe_operations,
        enable_memory_monitoring=enable_memory_monitoring
    )
    
    return EnhancedMemoryOptimizer(config)
