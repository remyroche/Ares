"""
Additional optimization strategies for feature generation.
"""

from abc import ABC, abstractmethod
from src.utils.tprint import tprint
from typing import Dict, Any, Optional
import logging
import numpy as np
import pandas as pd
import time

logger = logging.getLogger(__name__)

class AdaptiveOptimizationStrategy:
    """Adaptive optimization that adjusts based on data characteristics."""

    def __init__(self, config):
        self.config = config
        self.logger = logger.getChild(self.__class__.__name__)
        self.stats = {
            'optimizations_applied': 0,
            'total_time': 0.0,
            'memory_saved_mb': 0.0,
            'strategy_name': self.__class__.__name__
        }

    def optimize_data(self, data: pd.DataFrame, generator) -> pd.DataFrame:
        """Apply adaptive optimization based on data characteristics."""
        try:
            start_time = time.time()
            optimized_data = data
            
            # Analyze data characteristics
            data_size = len(data)
            memory_usage = data.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            
            # Choose optimization level based on data characteristics
            if data_size < 1000:
                optimization_level = "minimal"
            elif data_size < 10000:
                optimization_level = "balanced"
            else:
                optimization_level = "aggressive"
            
            self.logger.debug(f"Adaptive optimization: {optimization_level} for {data_size} rows, {memory_usage:.2f}MB")
            
            # Apply optimizations based on level
            if optimization_level in ["balanced", "aggressive"]:
                # Memory optimization
                if (self.config.enable_memory_optimization and
                    hasattr(generator, 'optimize_dataframe_processing')):
                    try:
                        original_memory = memory_usage
                        optimized_data = generator.optimize_dataframe_processing(data)
                        optimized_data = self._clean_non_finite_values(optimized_data)
                        
                        optimized_memory = optimized_data.memory_usage(deep=True).sum() / 1024 / 1024
                        memory_saved = max(0, original_memory - optimized_memory)
                        
                        self.stats['optimizations_applied'] += 1
                        self.stats['memory_saved_mb'] += memory_saved
                        
                        if self.config.enable_optimization_logging:
                            self.logger.debug(f"Memory optimization: {original_memory:.2f}MB -> {optimized_memory:.2f}MB")
                    except Exception as e:
                        self.logger.warning(f"Memory optimization failed: {e}")
            
            # Final data cleaning
            optimized_data = self._clean_non_finite_values(optimized_data)
            
            self.stats['total_time'] += time.time() - start_time
            return optimized_data
            
        except Exception as e:
            self.logger.error(f"Error in adaptive optimization: {e}")
            return data
    
    def _clean_non_finite_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean non-finite values from the DataFrame."""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in data.columns:
                non_finite_mask = ~np.isfinite(data[col])
                non_finite_count = non_finite_mask.sum()
                
                if non_finite_count > 0:
                    if hasattr(self, 'logger'):
                        self.logger.warning(f"Found {non_finite_count} non-finite values in column '{col}' after optimization")
                    
                    data[col] = data[col].replace([np.inf, -np.inf], np.nan)
                    data[col] = data[col].fillna(method='ffill')
                    data[col] = data[col].fillna(0)
        
        return data

    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return self.stats.copy()

    def reset_stats(self):
        """Reset optimization statistics."""
        self.stats = {
            'optimizations_applied': 0,
            'total_time': 0.0,
            'memory_saved_mb': 0.0,
            'strategy_name': self.__class__.__name__
        }

class MemoryOptimizedStrategy:
    """Memory-focused optimization strategy."""

    def __init__(self, config):
        self.config = config
        self.logger = logger.getChild(self.__class__.__name__)
        self.stats = {
            'optimizations_applied': 0,
            'total_time': 0.0,
            'memory_saved_mb': 0.0,
            'strategy_name': self.__class__.__name__
        }

    def optimize_data(self, data: pd.DataFrame, generator) -> pd.DataFrame:
        """Apply memory-focused optimization."""
        try:
            start_time = time.time()
            optimized_data = data
            
            # Aggressive memory optimization
            if hasattr(generator, 'optimize_dataframe_processing'):
                try:
                    original_memory = data.memory_usage(deep=True).sum() / 1024 / 1024
                    optimized_data = generator.optimize_dataframe_processing(data)
                    optimized_data = self._clean_non_finite_values(optimized_data)
                    
                    optimized_memory = optimized_data.memory_usage(deep=True).sum() / 1024 / 1024
                    memory_saved = max(0, original_memory - optimized_memory)
                    
                    self.stats['optimizations_applied'] += 1
                    self.stats['memory_saved_mb'] += memory_saved
                    
                    if self.config.enable_optimization_logging:
                        self.logger.debug(f"Memory optimization: {original_memory:.2f}MB -> {optimized_memory:.2f}MB (saved: {memory_saved:.2f}MB)")
                except Exception as e:
                    self.logger.warning(f"Memory optimization failed: {e}")
            
            # Additional memory optimizations
            optimized_data = self._optimize_dtypes(optimized_data)
            optimized_data = self._remove_unused_columns(optimized_data)
            
            self.stats['total_time'] += time.time() - start_time
            return optimized_data
            
        except Exception as e:
            self.logger.error(f"Error in memory optimization: {e}")
            return data
    
    def _optimize_dtypes(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types to reduce memory usage."""
        for col in data.columns:
            if data[col].dtype == 'int64':
                if data[col].min() >= 0 and data[col].max() <= 255:
                    data[col] = data[col].astype('uint8')
                elif data[col].min() >= -128 and data[col].max() <= 127:
                    data[col] = data[col].astype('int8')
                elif data[col].min() >= 0 and data[col].max() <= 65535:
                    data[col] = data[col].astype('uint16')
                elif data[col].min() >= -32768 and data[col].max() <= 32767:
                    data[col] = data[col].astype('int16')
                elif data[col].min() >= 0 and data[col].max() <= 4294967295:
                    data[col] = data[col].astype('uint32')
                elif data[col].min() >= -2147483648 and data[col].max() <= 2147483647:
                    data[col] = data[col].astype('int32')
            elif data[col].dtype == 'float64':
                if data[col].min() >= np.finfo(np.float32).min and data[col].max() <= np.finfo(np.float32).max:
                    data[col] = data[col].astype('float32')
        
        return data
    
    def _remove_unused_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove columns that are all NaN or constant."""
        # Remove columns that are all NaN
        data = data.dropna(axis=1, how='all')
        
        # Remove columns that are constant
        for col in data.columns:
            if data[col].nunique() <= 1:
                data = data.drop(columns=[col])
                if self.config.enable_optimization_logging:
                    self.logger.debug(f"Removed constant column: {col}")
        
        return data
    
    def _clean_non_finite_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean non-finite values from the DataFrame."""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in data.columns:
                non_finite_mask = ~np.isfinite(data[col])
                non_finite_count = non_finite_mask.sum()
                
                if non_finite_count > 0:
                    if hasattr(self, 'logger'):
                        self.logger.warning(f"Found {non_finite_count} non-finite values in column '{col}' after optimization")
                    
                    data[col] = data[col].replace([np.inf, -np.inf], np.nan)
                    data[col] = data[col].fillna(method='ffill')
                    data[col] = data[col].fillna(0)
        
        return data

    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return self.stats.copy()

    def reset_stats(self):
        """Reset optimization statistics."""
        self.stats = {
            'optimizations_applied': 0,
            'total_time': 0.0,
            'memory_saved_mb': 0.0,
            'strategy_name': self.__class__.__name__
        }