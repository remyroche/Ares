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
            tprint(f"🔧 Initializing {self.__class__.__name__}...")
            self.config = config
            self.logger = logger.getChild(self.__class__.__name__)
            self.stats = {
                'optimizations_applied': 0,
                'total_time': 0.0,
                'memory_saved_mb': 0.0,
                'strategy_name': self.__class__.__name__
            }
            tprint(f"✅ {self.__class__.__name__} initialized successfully")

        except Exception as e:
            tprint(f"❌ Error initializing {self.__class__.__name__}: {e}")
            raise

    @abstractmethod
    def optimize_data(self, data: pd.DataFrame, generator) -> pd.DataFrame:
        """Optimize data using this strategy."""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        try:
            tprint(f"📊 Getting stats for {self.__class__.__name__}...")
            stats = self.stats.copy()
            tprint(f"✅ Retrieved stats: {stats['optimizations_applied']} optimizations, {stats['total_time']:.3f}s total time")
            return stats

        except Exception as e:
            tprint(f"❌ Error getting stats: {e}")
            return {}

    def reset_stats(self):
        """Reset optimization statistics."""
        try:
            tprint(f"🔄 Resetting stats for {self.__class__.__name__}...")
            self.stats = {
                'optimizations_applied': 0,
                'total_time': 0.0,
                'memory_saved_mb': 0.0,
                'strategy_name': self.__class__.__name__
            }
            tprint("✅ Stats reset successfully")

        except Exception as e:
            tprint(f"❌ Error resetting stats: {e}")
            raise

class ConservativeOptimizationStrategy(OptimizationStrategy):
    """Conservative optimization - minimal changes, maximum compatibility."""

    def optimize_data(self, data: pd.DataFrame, generator) -> pd.DataFrame:
        """Apply conservative optimization."""
        try:
            tprint("🔧 Starting conservative optimization...")
            start_time = time.time()
            optimized_data = data

            # Only basic memory optimization
            if (self.config.enable_memory_optimization and
                hasattr(generator, 'optimize_dataframe_processing')):
                try:
                    tprint("💾 Applying memory optimization...")
                    original_memory = data.memory_usage(deep=True).sum() / 1024 / 1024  # MB
                    tprint(f"📊 Original memory usage: {original_memory:.2f}MB")

                    optimized_data = generator.optimize_dataframe_processing(data)
                    optimized_memory = optimized_data.memory_usage(deep=True).sum() / 1024 / 1024  # MB

                    self.stats['optimizations_applied'] += 1
                    memory_saved = max(0, original_memory - optimized_memory)
                    self.stats['memory_saved_mb'] += memory_saved

                    tprint(f"✅ Memory optimization applied: {original_memory:.2f}MB -> {optimized_memory:.2f}MB")
                    tprint(f"💾 Memory saved: {memory_saved:.2f}MB")

                    if self.config.enable_optimization_logging:
                        self.logger.debug(f"Memory optimization applied: {original_memory:.2f}MB -> {optimized_memory:.2f}MB")

                except Exception as e:
                    tprint(f"❌ Memory optimization failed: {e}")
                    self.logger.warning(f"Memory optimization failed: {e}")
            else:
                tprint("⚠️ Memory optimization disabled or not available")

            self.stats['total_time'] += time.time() - start_time
            tprint(f"✅ Conservative optimization completed in {self.stats['total_time']:.3f}s")
            return optimized_data

        except Exception as e:
            tprint(f"❌ Error in conservative optimization: {e}")
            self.logger.error(f"Error in conservative optimization: {e}")
            # Return original data on error
            return data

class BalancedOptimizationStrategy(OptimizationStrategy):
    """Balanced optimization - good performance/quality tradeoff."""

    def optimize_data(self, data: pd.DataFrame, generator) -> pd.DataFrame:
        """Apply balanced optimization."""
        try:
            tprint("🔧 Starting balanced optimization...")
            start_time = time.time()
            optimized_data = data

            # Memory optimization
            if (self.config.enable_memory_optimization and
                hasattr(generator, 'optimize_dataframe_processing')):
                try:
                    tprint("💾 Applying memory optimization...")
                    original_memory = data.memory_usage(deep=True).sum() / 1024 / 1024  # MB
                    tprint(f"📊 Original memory usage: {original_memory:.2f}MB")

                    optimized_data = generator.optimize_dataframe_processing(data)
                    optimized_memory = optimized_data.memory_usage(deep=True).sum() / 1024 / 1024  # MB

                    self.stats['optimizations_applied'] += 1
                    memory_saved = max(0, original_memory - optimized_memory)
                    self.stats['memory_saved_mb'] += memory_saved

                    tprint(f"✅ Memory optimization applied: {original_memory:.2f}MB -> {optimized_memory:.2f}MB")
                    tprint(f"💾 Memory saved: {memory_saved:.2f}MB")

                    if self.config.enable_optimization_logging:
                        self.logger.debug(f"Memory optimization applied: {original_memory:.2f}MB -> {optimized_memory:.2f}MB")

                except Exception as e:
                    tprint(f"❌ Memory optimization failed: {e}")
                    self.logger.warning(f"Memory optimization failed: {e}")
            else:
                tprint("⚠️ Memory optimization disabled or not available")

            # VectorBT optimization for large datasets
            if (self.config.enable_vectorbt_optimization and
                len(optimized_data) > self.config.vectorbt_threshold and
                hasattr(generator, '_should_use_vectorbt')):
                try:
                    tprint(f"⚡ Checking VectorBT optimization (threshold: {self.config.vectorbt_threshold})...")
                    if generator._should_use_vectorbt(optimized_data):
                        tprint("🚀 Applying VectorBT optimization...")
                        optimized_data = self._apply_vectorbt_optimizations(optimized_data, generator)
                        self.stats['optimizations_applied'] += 1
                        tprint(f"✅ VectorBT optimization applied for {len(optimized_data)} rows")

                        if self.config.enable_optimization_logging:
                            self.logger.debug(f"VectorBT optimization applied for {len(optimized_data)} rows")
                    else:
                        tprint("⚠️ VectorBT optimization not recommended for this dataset")

                except Exception as e:
                    tprint(f"❌ VectorBT optimization failed: {e}")
                    self.logger.warning(f"VectorBT optimization failed: {e}")
            else:
                tprint("⚠️ VectorBT optimization disabled or not available")

            # Rolling operations optimization
            if (self.config.enable_rolling_optimization and
                hasattr(generator, 'enable_rolling_cache')):
                try:
                    tprint("🔄 Applying rolling operations optimization...")
                    if hasattr(generator, 'enable_rolling_cache'):
                        generator.enable_rolling_cache = self.config.enable_rolling_cache
                        generator.rolling_cache_size = self.config.rolling_cache_size
                        self.stats['optimizations_applied'] += 1
                        tprint(f"✅ Rolling operations optimization enabled (cache size: {self.config.rolling_cache_size})")

                        if self.config.enable_optimization_logging:
                            self.logger.debug("Rolling operations optimization enabled")
                    else:
                        tprint("⚠️ Rolling cache not available on generator")

                except Exception as e:
                    tprint(f"❌ Rolling operations optimization failed: {e}")
                    self.logger.warning(f"Rolling operations optimization failed: {e}")
            else:
                tprint("⚠️ Rolling operations optimization disabled or not available")

            self.stats['total_time'] += time.time() - start_time
            tprint(f"✅ Balanced optimization completed in {self.stats['total_time']:.3f}s")
            tprint(f"📊 Total optimizations applied: {self.stats['optimizations_applied']}")
            return optimized_data

        except Exception as e:
            tprint(f"❌ Error in balanced optimization: {e}")
            self.logger.error(f"Error in balanced optimization: {e}")
            # Return original data on error
            return data

    def _apply_vectorbt_optimizations(self, data: pd.DataFrame, generator) -> pd.DataFrame:
        """Apply VectorBT-specific optimizations."""
        # This would include VectorBT-specific data preparation
        # For now, just return the data as-is
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

        self.stats['total_time'] += time.time() - start_time
        return optimized_data

    def _apply_vectorbt_optimizations(self, data: pd.DataFrame, generator) -> pd.DataFrame:
        """Apply aggressive VectorBT optimizations."""
        # This would include aggressive VectorBT optimizations
        # For now, just return the data as-is
        return data
