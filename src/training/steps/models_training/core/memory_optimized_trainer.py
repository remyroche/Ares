"""
Memory-Optimized Trainer with Hardware Integration

This module provides memory-optimized training capabilities using hardware utilities
for efficient memory management and performance optimization.
"""

import logging
import gc
import psutil
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

# Import tprint utilities
from src.utils.tprint import (
    tprint, tprint_info, tprint_warning, tprint_error, tprint_success, 
    tprint_debug, tprint_data_format, tprint_data_preview, LogLevel
)

# Hardware optimization imports
from src.utils.hardware.memory_optimized_decorators import (
    memory_optimized, gc_optimized, chunked_processing_auto,
    comprehensive_memory_optimization, MemoryOptimizationLevel,
    optimize_large_dataframes, optimize_large_arrays, optimize_memory_intensive
)
from src.utils.hardware.advanced_memory_manager import (
    get_advanced_memory_manager, MemoryConfig as AdvancedMemoryConfig
)
from src.utils.hardware.dynamic_memory_allocator import (
    get_dynamic_allocator, get_optimal_memory_allocation, WorkloadType,
    update_memory_usage, get_system_recommendations
)
from src.utils.hardware.optimization_decorators import (
    smart_cache, auto_optimize, performance_tracked,
    OptimizationConfig, OptimizationLevel
)
from src.utils.hardware.integrated_hardware_manager import (
    IntegratedHardwareManager, IntegratedHardwareConfig
)

from .error_handling import (
    handle_errors, ErrorContext,
    MLModelTrainerError, ResourceError, ModelTrainingError
)

logger = logging.getLogger(__name__)

class MemoryStrategy(Enum):
    """Memory optimization strategies."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    ADAPTIVE = "adaptive"

@dataclass
class MemoryOptimizedConfig:
    """Configuration for memory-optimized training."""
    strategy: MemoryStrategy = MemoryStrategy.ADAPTIVE
    memory_limit_gb: float = 8.0
    gpu_memory_limit_gb: float = 4.0
    chunk_size: int = 1000
    enable_compression: bool = True
    enable_caching: bool = True
    enable_gc_optimization: bool = True
    enable_data_type_optimization: bool = True
    enable_chunked_processing: bool = True
    memory_monitoring_interval: float = 5.0
    cleanup_threshold: float = 0.8

class MemoryOptimizedTrainer:
    """Memory-optimized trainer with hardware integration."""
    
    def __init__(self, config: MemoryOptimizedConfig = None):
        self.config = config or MemoryOptimizedConfig()
        
        # Initialize hardware manager
        hardware_config = IntegratedHardwareConfig(
            enable_caching=self.config.enable_caching,
            enable_memory_optimization=True,
            memory_limit_gb=self.config.memory_limit_gb,
            gpu_memory_limit_gb=self.config.gpu_memory_limit_gb
        )
        self.hardware_manager = IntegratedHardwareManager(hardware_config)
        
        # Initialize advanced memory manager
        memory_config = AdvancedMemoryConfig(
            max_memory_gb=self.config.memory_limit_gb,
            enable_compression=self.config.enable_compression,
            enable_caching=self.config.enable_caching
        )
        self.memory_manager = get_advanced_memory_manager(memory_config)
        
        # Initialize dynamic allocator
        self.memory_allocator = get_dynamic_allocator()
        
        # Memory monitoring
        self.memory_usage_history = []
        self.peak_memory_usage = 0
        
    @comprehensive_memory_optimization(level=MemoryOptimizationLevel.AGGRESSIVE)
    @performance_tracked(level=OptimizationLevel.HIGH)
    async def train_with_memory_optimization(
        self,
        model_config: Dict[str, Any],
        data: Dict[str, Any],
        training_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Train a model with comprehensive memory optimization.
        
        Args:
            model_config: Model configuration
            data: Training data
            training_config: Training configuration
            
        Returns:
            Training results
        """
        with ErrorContext("memory_optimized_training", self._cleanup_memory):
            tprint_info("Starting memory-optimized training")
            logger.info("Starting memory-optimized training")
            
            # Pre-training memory optimization
            tprint_debug("Applying pre-training memory optimizations")
            await self._pre_training_optimization(data)
            
            # Optimize data types
            tprint_debug("Optimizing data types")
            optimized_data = await self._optimize_data_types(data)
            tprint_data_format(optimized_data, "Optimized data", LogLevel.DEBUG)
            
            # Chunked processing if needed
            if self._should_use_chunked_processing(optimized_data):
                tprint_info("Using chunked processing for large dataset")
                results = await self._train_with_chunked_processing(
                    model_config, optimized_data, training_config
                )
            else:
                tprint_info("Using full data processing")
                results = await self._train_with_full_data(
                    model_config, optimized_data, training_config
                )
            
            # Post-training cleanup
            tprint_debug("Performing post-training cleanup")
            await self._post_training_cleanup()
            
            tprint_success("Memory-optimized training completed")
            return results
    
    async def _pre_training_optimization(self, data: Dict[str, Any]):
        """Apply pre-training memory optimizations."""
        tprint_info("Applying pre-training memory optimizations")
        logger.info("Applying pre-training memory optimizations")
        
        # Get system recommendations
        recommendations = get_system_recommendations(
            workload_type=WorkloadType.MACHINE_LEARNING_TRAINING,
            data_size_gb=self._estimate_data_size(data)
        )
        
        # Update memory allocation based on recommendations
        if recommendations.get('memory_allocation'):
            self.config.memory_limit_gb = recommendations['memory_allocation']['memory_limit_gb']
            self.config.chunk_size = recommendations['memory_allocation'].get('chunk_size', self.config.chunk_size)
        
        # Force garbage collection
        if self.config.enable_gc_optimization:
            gc.collect()
        
        # Clear caches if memory usage is high
        current_usage = self._get_memory_usage()
        if current_usage > self.config.cleanup_threshold:
            self.hardware_manager.clear_caches()
            logger.info("Cleared caches due to high memory usage")
    
    @optimize_large_dataframes
    @optimize_large_arrays
    async def _optimize_data_types(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize data types for memory efficiency."""
        logger.info("Optimizing data types for memory efficiency")
        
        optimized_data = {}
        
        for key, value in data.items():
            if isinstance(value, pd.DataFrame):
                # Optimize DataFrame
                optimized_data[key] = self._optimize_dataframe(value)
            elif isinstance(value, np.ndarray):
                # Optimize NumPy array
                optimized_data[key] = self._optimize_array(value)
            elif isinstance(value, dict):
                # Recursively optimize nested dictionaries
                optimized_data[key] = await self._optimize_data_types(value)
            else:
                optimized_data[key] = value
        
        return optimized_data
    
    def _optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for memory usage."""
        original_memory = df.memory_usage(deep=True).sum()
        
        # Optimize data types
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to category
                if df[col].nunique() / len(df) < 0.5:
                    df[col] = df[col].astype('category')
            elif df[col].dtype == 'int64':
                # Downcast integers
                if df[col].min() >= 0:
                    if df[col].max() < 255:
                        df[col] = df[col].astype('uint8')
                    elif df[col].max() < 65535:
                        df[col] = df[col].astype('uint16')
                    elif df[col].max() < 4294967295:
                        df[col] = df[col].astype('uint32')
                else:
                    if df[col].min() > -128 and df[col].max() < 127:
                        df[col] = df[col].astype('int8')
                    elif df[col].min() > -32768 and df[col].max() < 32767:
                        df[col] = df[col].astype('int16')
                    elif df[col].min() > -2147483648 and df[col].max() < 2147483647:
                        df[col] = df[col].astype('int32')
            elif df[col].dtype == 'float64':
                # Downcast floats
                df[col] = pd.to_numeric(df[col], downcast='float')
        
        optimized_memory = df.memory_usage(deep=True).sum()
        reduction = (original_memory - optimized_memory) / original_memory * 100
        
        if reduction > 0:
            logger.info(f"DataFrame optimization: {reduction:.1f}% memory reduction")
        
        return df
    
    def _optimize_array(self, arr: np.ndarray) -> np.ndarray:
        """Optimize NumPy array for memory usage."""
        original_memory = arr.nbytes
        
        # Optimize data type
        if arr.dtype == np.float64:
            # Check if we can use float32
            if np.all(np.isfinite(arr)) and np.all(np.abs(arr) < 3.4e38):
                arr = arr.astype(np.float32)
        elif arr.dtype == np.int64:
            # Check if we can use smaller integer types
            if arr.min() >= 0:
                if arr.max() < 255:
                    arr = arr.astype(np.uint8)
                elif arr.max() < 65535:
                    arr = arr.astype(np.uint16)
                elif arr.max() < 4294967295:
                    arr = arr.astype(np.uint32)
            else:
                if arr.min() > -128 and arr.max() < 127:
                    arr = arr.astype(np.int8)
                elif arr.min() > -32768 and arr.max() < 32767:
                    arr = arr.astype(np.int16)
                elif arr.min() > -2147483648 and arr.max() < 2147483647:
                    arr = arr.astype(np.int32)
        
        optimized_memory = arr.nbytes
        reduction = (original_memory - optimized_memory) / original_memory * 100
        
        if reduction > 0:
            logger.info(f"Array optimization: {reduction:.1f}% memory reduction")
        
        return arr
    
    def _should_use_chunked_processing(self, data: Dict[str, Any]) -> bool:
        """Determine if chunked processing should be used."""
        data_size_gb = self._estimate_data_size(data)
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # Use chunked processing if data size > 50% of available memory
        return data_size_gb > available_memory_gb * 0.5
    
    @chunked_processing_auto(chunk_size=1000)
    async def _train_with_chunked_processing(
        self,
        model_config: Dict[str, Any],
        data: Dict[str, Any],
        training_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train model with chunked processing."""
        logger.info("Training with chunked processing")
        
        # This would implement actual chunked training logic
        # For now, return a placeholder
        return {
            'model_type': model_config.get('type', 'unknown'),
            'status': 'trained_chunked',
            'memory_optimized': True,
            'timestamp': time.time()
        }
    
    async def _train_with_full_data(
        self,
        model_config: Dict[str, Any],
        data: Dict[str, Any],
        training_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train model with full data in memory."""
        logger.info("Training with full data")
        
        # Monitor memory usage during training
        self._start_memory_monitoring()
        
        try:
            # This would implement actual training logic
            # For now, return a placeholder
            result = {
                'model_type': model_config.get('type', 'unknown'),
                'status': 'trained_full',
                'memory_optimized': True,
                'timestamp': time.time()
            }
            
            return result
            
        finally:
            self._stop_memory_monitoring()
    
    def _start_memory_monitoring(self):
        """Start monitoring memory usage."""
        self.memory_usage_history = []
        self.peak_memory_usage = 0
        
        # This would start a background monitoring task
        # For now, just log the start
        logger.info("Started memory monitoring")
    
    def _stop_memory_monitoring(self):
        """Stop monitoring memory usage."""
        if self.memory_usage_history:
            avg_usage = sum(self.memory_usage_history) / len(self.memory_usage_history)
            logger.info(f"Memory monitoring complete - Peak: {self.peak_memory_usage:.1f}%, Avg: {avg_usage:.1f}%")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage as percentage."""
        return psutil.virtual_memory().percent / 100
    
    def _estimate_data_size(self, data: Dict[str, Any]) -> float:
        """Estimate data size in GB."""
        total_size = 0
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                total_size += value.nbytes
            elif isinstance(value, pd.DataFrame):
                total_size += value.memory_usage(deep=True).sum()
            elif isinstance(value, dict):
                total_size += self._estimate_data_size(value)
        return total_size / (1024**3)
    
    async def _post_training_cleanup(self):
        """Apply post-training memory cleanup."""
        logger.info("Applying post-training cleanup")
        
        # Force garbage collection
        if self.config.enable_gc_optimization:
            gc.collect()
        
        # Clear caches
        self.hardware_manager.clear_caches()
        
        # Update memory usage tracking
        update_memory_usage(
            workload_type=WorkloadType.MACHINE_LEARNING_TRAINING,
            memory_used_gb=self._estimate_data_size({}),
            success=True
        )
    
    async def _cleanup_memory(self):
        """Emergency memory cleanup."""
        logger.warning("Performing emergency memory cleanup")
        
        # Force garbage collection
        gc.collect()
        
        # Clear all caches
        self.hardware_manager.clear_caches()
        
        # Clear memory usage history
        self.memory_usage_history.clear()
        
        logger.info("Emergency memory cleanup completed")

# Factory function
def create_memory_optimized_trainer(config: MemoryOptimizedConfig = None) -> MemoryOptimizedTrainer:
    """Create a memory-optimized trainer with configuration."""
    return MemoryOptimizedTrainer(config)

# Import time for timestamps
import time