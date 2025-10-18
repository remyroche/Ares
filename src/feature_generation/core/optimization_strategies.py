from abc import ABC, abstractmethod
from src.utils.tprint import tprint
from typing import Dict, Any, Optional
import logging
import numpy as np
import pandas as pd
import time

logger = logging.getLogger(__name__)

class OptimizationStrategy(ABC):
    """Base class for optimization strategies."""

    def __init__(self, config: 'AutoOptimizationConfig'):
        try:
            # Reduce verbosity - only log on first initialization
            if not hasattr(self.__class__, '_initialization_logged'):
                pass  # tprint statement removed
                self.__class__._initialization_logged = True
            
            self.config = config
            self.logger = logger.getChild(self.__class__.__name__)
            self.stats = {
                'optimizations_applied': 0,
                'total_time': 0.0,
                'memory_saved_mb': 0.0,
                'strategy_name': self.__class__.__name__
            }
            
            # Only log success on first initialization
            if not hasattr(self.__class__, '_success_logged'):
                pass  # tprint statement removed
                self.__class__._success_logged = True

        except Exception as e:
            raise

    @abstractmethod
    def optimize_data(self, data: pd.DataFrame, generator) -> pd.DataFrame:
        """Optimize data using this strategy."""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        try:
            stats = self.stats.copy()
            # Only log if there are actual optimizations applied
            if stats.get('optimizations_applied', 0) > 0:
                pass  # tprint statement removed
            return stats

        except Exception as e:
            return {}

    def reset_stats(self):
        """Reset optimization statistics."""
        try:
            self.stats = {
                'optimizations_applied': 0,
                'total_time': 0.0,
                'memory_saved_mb': 0.0,
                'strategy_name': self.__class__.__name__
            }

        except Exception as e:
            raise

class ConservativeOptimizationStrategy(OptimizationStrategy):
    """Conservative optimization - minimal changes, maximum compatibility."""

    def optimize_data(self, data: pd.DataFrame, generator) -> pd.DataFrame:
        """Apply conservative optimization."""
        try:
            start_time = time.time()
            optimized_data = data

            # Only basic memory optimization
            if (self.config.enable_memory_optimization and
                hasattr(generator, 'optimize_dataframe_processing')):
                try:
                    original_memory = data.memory_usage(deep=True).sum() / 1024 / 1024  # MB

                    optimized_data = generator.optimize_dataframe_processing(data)
                    
                    # Clean any non-finite values introduced by optimization
                    optimized_data = self._clean_non_finite_values(optimized_data)
                    
                    optimized_memory = optimized_data.memory_usage(deep=True).sum() / 1024 / 1024  # MB

                    self.stats['optimizations_applied'] += 1
                    memory_saved = max(0, original_memory - optimized_memory)
                    self.stats['memory_saved_mb'] += memory_saved

                    if self.config.enable_optimization_logging:
                        self.logger.debug(f"Memory optimization applied: {original_memory:.2f}MB -> {optimized_memory:.2f}MB")

                except Exception as e:
                    self.logger.warning(f"Memory optimization failed: {e}")
            else:
                pass  # tprint statement removed

            # Final data cleaning to ensure no non-finite values
            optimized_data = self._clean_non_finite_values(optimized_data)
            
            self.stats['total_time'] += time.time() - start_time
            return optimized_data

        except Exception as e:
            self.logger.error(f"Error in conservative optimization: {e}")
            # Return original data on error
            return data
    
    def _clean_non_finite_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean non-finite values from the DataFrame."""
        import numpy as np
        
        # Check for non-finite values in numeric columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in data.columns:
                # Count non-finite values
                non_finite_mask = ~np.isfinite(data[col])
                non_finite_count = non_finite_mask.sum()
                
                if non_finite_count > 0:
                    # Log the issue
                    if hasattr(self, 'logger'):
                        self.logger.warning(f"Found {non_finite_count} non-finite values in column '{col}' after optimization")
                    
                    # Replace non-finite values with the last valid value (forward fill)
                    data[col] = data[col].replace([np.inf, -np.inf], np.nan)
                    data[col] = data[col].fillna(method='ffill')
                    
                    # If there are still NaN values at the beginning, fill with 0
                    data[col] = data[col].fillna(0)
        
        return data

class BalancedOptimizationStrategy(OptimizationStrategy):
    """Balanced optimization - good performance/quality tradeoff."""

    def optimize_data(self, data: pd.DataFrame, generator) -> pd.DataFrame:
        """Apply balanced optimization."""
        try:
            # Only log once per optimization session to reduce verbosity
            if not hasattr(self, '_optimization_logged'):
                pass  # tprint statement removed
                self._optimization_logged = True
            start_time = time.time()
            optimized_data = data

            # Memory optimization
            if (self.config.enable_memory_optimization and
                hasattr(generator, 'optimize_dataframe_processing')):
                try:
                    # Reduce verbosity - only log significant memory savings
                    original_memory = data.memory_usage(deep=True).sum() / 1024 / 1024  # MB
                    optimized_data = generator.optimize_dataframe_processing(data)
                    
                    # Clean any non-finite values introduced by optimization
                    optimized_data = self._clean_non_finite_values(optimized_data)
                    
                    optimized_memory = optimized_data.memory_usage(deep=True).sum() / 1024 / 1024  # MB

                    self.stats['optimizations_applied'] += 1
                    memory_saved = max(0, original_memory - optimized_memory)
                    self.stats['memory_saved_mb'] += memory_saved

                    # Only log if significant memory was saved
                    if memory_saved > 10:  # Only log if more than 10MB saved
                        pass  # tprint statement removed

                    if self.config.enable_optimization_logging:
                        self.logger.debug(f"Memory optimization applied: {original_memory:.2f}MB -> {optimized_memory:.2f}MB")

                except Exception as e:
                    self.logger.warning(f"Memory optimization failed: {e}")
            else:
                pass  # tprint statement removed

            # VectorBT optimization for large datasets
            if (self.config.enable_vectorbt_optimization and
                len(optimized_data) > self.config.vectorbt_threshold and
                hasattr(generator, '_should_use_vectorbt')):
                try:
                    # Reduce verbosity - only log when VectorBT is actually applied
                    if generator._should_use_vectorbt(optimized_data):
                        optimized_data = self._apply_vectorbt_optimizations(optimized_data, generator)
                        
                        # Clean any non-finite values introduced by VectorBT optimization
                        optimized_data = self._clean_non_finite_values(optimized_data)
                        
                        self.stats['optimizations_applied'] += 1
                        # Only log if significant dataset size
                        if len(optimized_data) > 10000:
                            pass  # tprint statement removed

                        if self.config.enable_optimization_logging:
                            self.logger.debug(f"VectorBT optimization applied for {len(optimized_data)} rows")

                except Exception as e:
                    self.logger.warning(f"VectorBT optimization failed: {e}")
            else:
                pass  # tprint statement removed

            # Rolling operations optimization
            if (self.config.enable_rolling_optimization and
                hasattr(generator, 'enable_rolling_cache')):
                try:
                    if hasattr(generator, 'enable_rolling_cache'):
                        generator.enable_rolling_cache = self.config.enable_rolling_cache
                        generator.rolling_cache_size = self.config.rolling_cache_size
                        self.stats['optimizations_applied'] += 1

                        if self.config.enable_optimization_logging:
                            self.logger.debug("Rolling operations optimization enabled")
                    else:
                        pass  # tprint statement removed

                except Exception as e:
                    self.logger.warning(f"Rolling operations optimization failed: {e}")
            else:
                pass  # tprint statement removed

            # Final data cleaning to ensure no non-finite values
            optimized_data = self._clean_non_finite_values(optimized_data)
            
            self.stats['total_time'] += time.time() - start_time
            return optimized_data

        except Exception as e:
            self.logger.error(f"Error in balanced optimization: {e}")
            # Return original data on error
            return data

    def _apply_vectorbt_optimizations(self, data: pd.DataFrame, generator) -> pd.DataFrame:
        """Apply VectorBT-specific optimizations."""
        # This would include VectorBT-specific data preparation
        # For now, just return the data as-is
        return data
    
    def _clean_non_finite_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean non-finite values from the DataFrame."""
        import numpy as np
        
        # Check for non-finite values in numeric columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in data.columns:
                # Count non-finite values
                non_finite_mask = ~np.isfinite(data[col])
                non_finite_count = non_finite_mask.sum()
                
                if non_finite_count > 0:
                    # Log the issue
                    if hasattr(self, 'logger'):
                        self.logger.warning(f"Found {non_finite_count} non-finite values in column '{col}' after optimization")
                    
                    # Replace non-finite values with the last valid value (forward fill)
                    data[col] = data[col].replace([np.inf, -np.inf], np.nan)
                    data[col] = data[col].fillna(method='ffill')
                    
                    # If there are still NaN values at the beginning, fill with 0
                    data[col] = data[col].fillna(0)
        
        return data

class AggressiveOptimizationStrategy(OptimizationStrategy):
    """Aggressive optimization - maximum performance."""

    def optimize_data(self, data: pd.DataFrame, generator) -> pd.DataFrame:
        """Apply aggressive optimization."""
        start_time = time.time()
        optimized_data = data

        # All available optimizations
        if (self.config.enable_memory_optimization and
            hasattr(generator, 'optimize_dataframe_processing')):
            try:
                original_memory = data.memory_usage(deep=True).sum() / 1024 / 1024  # MB
                optimized_data = generator.optimize_dataframe_processing(data)
                
                # Clean any non-finite values introduced by optimization
                optimized_data = self._clean_non_finite_values(optimized_data)
                
                optimized_memory = optimized_data.memory_usage(deep=True).sum() / 1024 / 1024  # MB

                self.stats['optimizations_applied'] += 1
                self.stats['memory_saved_mb'] += max(0, original_memory - optimized_memory)

                if self.config.enable_optimization_logging:
                    self.logger.debug(f"Memory optimization applied: {original_memory:.2f}MB -> {optimized_memory:.2f}MB")

            except Exception as e:
                self.logger.warning(f"Memory optimization failed: {e}")

        # Chunked processing for very large datasets
        if (self.config.enable_chunked_processing and
            len(optimized_data) > 10000 and
            hasattr(generator, 'chunked_processing')):
            try:
                optimized_data = generator.chunked_processing(
                    optimized_data,
                    lambda x: x,
                    chunk_size=self.config.chunk_size
                )
                self.stats['optimizations_applied'] += 1

                if self.config.enable_optimization_logging:
                    self.logger.debug(f"Chunked processing applied with chunk size {self.config.chunk_size}")

            except Exception as e:
                self.logger.warning(f"Chunked processing failed: {e}")

        # VectorBT optimization
        if (self.config.enable_vectorbt_optimization and
            hasattr(generator, '_should_use_vectorbt')):
            try:
                if generator._should_use_vectorbt(optimized_data):
                    optimized_data = self._apply_vectorbt_optimizations(optimized_data, generator)
                    
                    # Clean any non-finite values introduced by VectorBT optimization
                    optimized_data = self._clean_non_finite_values(optimized_data)
                    
                    self.stats['optimizations_applied'] += 1

                    if self.config.enable_optimization_logging:
                        self.logger.debug(f"VectorBT optimization applied for {len(optimized_data)} rows")

            except Exception as e:
                self.logger.warning(f"VectorBT optimization failed: {e}")

        # Rolling operations optimization
        if (self.config.enable_rolling_optimization and
            hasattr(generator, 'enable_rolling_cache')):
            try:
                if hasattr(generator, 'enable_rolling_cache'):
                    generator.enable_rolling_cache = self.config.enable_rolling_cache
                    generator.rolling_cache_size = self.config.rolling_cache_size
                    self.stats['optimizations_applied'] += 1

                    if self.config.enable_optimization_logging:
                        self.logger.debug("Rolling operations optimization enabled")

            except Exception as e:
                self.logger.warning(f"Rolling operations optimization failed: {e}")

        # Final data cleaning to ensure no non-finite values
        optimized_data = self._clean_non_finite_values(optimized_data)
        
        self.stats['total_time'] += time.time() - start_time
        return optimized_data

    def _apply_vectorbt_optimizations(self, data: pd.DataFrame, generator) -> pd.DataFrame:
        """Apply aggressive VectorBT optimizations."""
        # This would include aggressive VectorBT optimizations
        # For now, just return the data as-is
        return data
    
    def _clean_non_finite_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean non-finite values from the DataFrame."""
        import numpy as np
        
        # Check for non-finite values in numeric columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in data.columns:
                # Count non-finite values
                non_finite_mask = ~np.isfinite(data[col])
                non_finite_count = non_finite_mask.sum()
                
                if non_finite_count > 0:
                    # Log the issue
                    if hasattr(self, 'logger'):
                        self.logger.warning(f"Found {non_finite_count} non-finite values in column '{col}' after optimization")
                    
                    # Replace non-finite values with the last valid value (forward fill)
                    data[col] = data[col].replace([np.inf, -np.inf], np.nan)
                    data[col] = data[col].fillna(method='ffill')
                    
                    # If there are still NaN values at the beginning, fill with 0
                    data[col] = data[col].fillna(0)
        
        return data
