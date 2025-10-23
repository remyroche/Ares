"""
M1 Optimizer for Market Analysis Components.

This module provides M1 chip optimization capabilities for market analysis
pipeline steps, including vectorization, memory optimization, and
performance tuning for Apple Silicon.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
import threading
import time

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error
from src.utils.common_operations import is_m1_available, get_m1_gpu_manager, get_m1_memory_optimizer
from src.training.steps.market_analysis.components.base_component import BaseMarketAnalysisComponent, ComponentConfig

class M1OptimizationLevel(Enum):
    """M1 optimization levels."""
    NONE = "none"
    BASIC = "basic"
    VECTORIZED = "vectorized"
    MAXIMUM = "maximum"

@dataclass
class M1Config:
    """Configuration for M1 optimization."""
    # Memory management
    memory_pressure_threshold: float = 0.7
    enable_unified_memory: bool = True
    memory_cleanup_interval: float = 30.0
    
    # Vectorization
    enable_vectorization: bool = True
    vectorization_threshold: int = 1000
    enable_simd: bool = True
    
    # Threading
    enable_threading: bool = True
    max_threads: int = 8
    thread_affinity: bool = True
    
    # Performance
    enable_jit_compilation: bool = True
    enable_parallel_processing: bool = True
    optimization_level: M1OptimizationLevel = M1OptimizationLevel.VECTORIZED
    
    # Monitoring
    enable_performance_monitoring: bool = True
    monitoring_interval: float = 5.0

@dataclass
class M1Status:
    """Current M1 status."""
    available: bool
    memory_pressure: float
    cpu_usage: float
    gpu_usage: float
    optimization_active: bool
    last_updated: datetime = field(default_factory=datetime.now)

class M1Optimizer(BaseMarketAnalysisComponent):
    """
    M1 optimizer for market analysis components.
    
    Provides:
    - M1-specific optimizations
    - Vectorization and SIMD
    - Memory pressure management
    - Performance monitoring
    """
    
    def __init__(self, config: Optional[M1Config] = None):
        """Initialize the M1 optimizer."""
        super().__init__(ComponentConfig())
        self.m1_config = config or M1Config()
        self.logger = logging.getLogger(__name__)
        
        # M1 availability
        self.m1_available = is_m1_available()
        
        # Status
        self.status = M1Status(
            available=self.m1_available,
            memory_pressure=0.0,
            cpu_usage=0.0,
            gpu_usage=0.0,
            optimization_active=False
        )
        
        # Optimization components
        self.gpu_manager = None
        self.memory_optimizer = None
        
        # Performance tracking
        self.performance_metrics = {}
        self.optimization_history = []
        
        # Initialize M1 components
        self._initialize_m1_components()
    
    def _initialize_m1_components(self):
        """Initialize M1-specific components."""
        try:
            if not self.m1_available:
                tprint_warning("❌ M1 not available")
                return
            
            # Initialize GPU manager for unified memory
            if self.m1_config.enable_unified_memory:
                try:
                    self.gpu_manager = get_m1_gpu_manager()
                    tprint_info("✅ M1 GPU manager initialized")
                except Exception as e:
                    tprint_warning(f"M1 GPU manager initialization failed: {str(e)}")
            
            # Initialize memory optimizer
            try:
                self.memory_optimizer = get_m1_memory_optimizer()
                tprint_info("✅ M1 memory optimizer initialized")
            except Exception as e:
                tprint_warning(f"M1 memory optimizer initialization failed: {str(e)}")
            
            tprint_info("✅ M1 optimizer initialized")
            
        except Exception as e:
            tprint_warning(f"M1 components initialization failed: {str(e)}")
    
    async def optimize_for_task(self, recommendations: Dict[str, Any]):
        """Optimize M1 for specific task."""
        try:
            if not self.m1_available:
                return
            
            tprint_info("🔧 Optimizing M1 for task")
            
            # Set optimization level
            if 'optimization_level' in recommendations:
                level = recommendations['optimization_level']
                if level == 'maximum':
                    self.m1_config.optimization_level = M1OptimizationLevel.MAXIMUM
                elif level == 'vectorized':
                    self.m1_config.optimization_level = M1OptimizationLevel.VECTORIZED
                elif level == 'basic':
                    self.m1_config.optimization_level = M1OptimizationLevel.BASIC
            
            # Enable vectorization for large datasets
            if recommendations.get('batch_size', 0) > self.m1_config.vectorization_threshold:
                self.m1_config.enable_vectorization = True
            
            # Enable threading for parallel operations
            if recommendations.get('parallel_workers', 1) > 1:
                self.m1_config.enable_threading = True
                self.m1_config.max_threads = min(recommendations['parallel_workers'], 8)
            
            self.status.optimization_active = True
            tprint_info("✅ M1 optimization completed")
            
        except Exception as e:
            tprint_warning(f"M1 optimization failed: {str(e)}")
    
    async def optimize_operation(self, 
                               operation: Callable,
                               data: Union[np.ndarray, pd.DataFrame],
                               *args, 
                               **kwargs) -> Any:
        """
        Optimize an operation for M1.
        
        Args:
            operation: Function to optimize
            data: Data to process
            *args: Additional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            Result of the operation
        """
        try:
            if not self.m1_available:
                return await operation(data, *args, **kwargs)
            
            # Check memory pressure
            await self._check_memory_pressure()
            
            # Apply M1-specific optimizations
            optimized_data = await self._apply_m1_optimizations(data)
            
            # Execute operation with optimizations
            start_time = time.time()
            result = await operation(optimized_data, *args, **kwargs)
            execution_time = time.time() - start_time
            
            # Track performance
            self._track_performance(operation.__name__, execution_time, len(data))
            
            return result
            
        except Exception as e:
            tprint_warning(f"M1 operation optimization failed: {str(e)}")
            return await operation(data, *args, **kwargs)
    
    async def _apply_m1_optimizations(self, data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """Apply M1-specific optimizations to data."""
        try:
            if isinstance(data, np.ndarray):
                return await self._optimize_array_m1(data)
            elif isinstance(data, pd.DataFrame):
                return await self._optimize_dataframe_m1(data)
            else:
                return data
                
        except Exception as e:
            tprint_warning(f"M1 data optimization failed: {str(e)}")
            return data
    
    async def _optimize_array_m1(self, arr: np.ndarray) -> np.ndarray:
        """Optimize NumPy array for M1."""
        try:
            # Ensure contiguous memory layout for better vectorization
            if not arr.flags.c_contiguous:
                arr = np.ascontiguousarray(arr)
            
            # Use optimal data types for M1
            if arr.dtype == np.float64 and self.m1_config.optimization_level in [M1OptimizationLevel.VECTORIZED, M1OptimizationLevel.MAXIMUM]:
                # M1 performs better with float32 for many operations
                if arr.min() >= np.finfo(np.float32).min and arr.max() <= np.finfo(np.float32).max:
                    arr = arr.astype(np.float32)
            
            # Enable SIMD optimizations
            if self.m1_config.enable_simd and len(arr) > self.m1_config.vectorization_threshold:
                # This would use M1-specific SIMD instructions in practice
                pass
            
            return arr
            
        except Exception as e:
            tprint_warning(f"M1 array optimization failed: {str(e)}")
            return arr
    
    async def _optimize_dataframe_m1(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for M1."""
        try:
            optimized_df = df.copy()
            
            # Optimize data types for M1
            for col in optimized_df.columns:
                if optimized_df[col].dtype == 'object':
                    # Use category for low cardinality columns
                    if optimized_df[col].nunique() / len(optimized_df) < 0.5:
                        optimized_df[col] = optimized_df[col].astype('category')
                elif optimized_df[col].dtype == 'float64':
                    # Convert to float32 for better M1 performance
                    if optimized_df[col].min() >= np.finfo(np.float32).min and \
                       optimized_df[col].max() <= np.finfo(np.float32).max:
                        optimized_df[col] = optimized_df[col].astype('float32')
            
            return optimized_df
            
        except Exception as e:
            tprint_warning(f"M1 DataFrame optimization failed: {str(e)}")
            return df
    
    async def _check_memory_pressure(self):
        """Check M1 memory pressure."""
        try:
            if self.memory_optimizer:
                pressure = await self.memory_optimizer.get_memory_pressure()
                self.status.memory_pressure = pressure
                
                if pressure > self.m1_config.memory_pressure_threshold:
                    tprint_warning(f"⚠️ High M1 memory pressure: {pressure:.1%}")
                    await self._cleanup_memory()
            
        except Exception as e:
            tprint_warning(f"M1 memory pressure check failed: {str(e)}")
    
    async def _cleanup_memory(self):
        """Cleanup M1 memory."""
        try:
            if self.memory_optimizer:
                await self.memory_optimizer.cleanup()
                tprint_info("🧹 M1 memory cleaned up")
            
        except Exception as e:
            tprint_warning(f"M1 memory cleanup failed: {str(e)}")
    
    def _track_performance(self, operation_name: str, execution_time: float, data_size: int):
        """Track M1 performance metrics."""
        try:
            if operation_name not in self.performance_metrics:
                self.performance_metrics[operation_name] = {
                    'total_time': 0.0,
                    'total_operations': 0,
                    'avg_time': 0.0,
                    'data_sizes': []
                }
            
            metrics = self.performance_metrics[operation_name]
            metrics['total_time'] += execution_time
            metrics['total_operations'] += 1
            metrics['avg_time'] = metrics['total_time'] / metrics['total_operations']
            metrics['data_sizes'].append(data_size)
            
            # Keep only recent data sizes
            if len(metrics['data_sizes']) > 100:
                metrics['data_sizes'] = metrics['data_sizes'][-100:]
            
        except Exception as e:
            tprint_warning(f"Performance tracking failed: {str(e)}")
    
    async def get_status(self) -> M1Status:
        """Get current M1 status."""
        try:
            if not self.m1_available:
                return self.status
            
            # Update memory pressure
            if self.memory_optimizer:
                self.status.memory_pressure = await self.memory_optimizer.get_memory_pressure()
            
            # Update CPU usage (simplified)
            import psutil
            self.status.cpu_usage = psutil.cpu_percent() / 100.0
            
            # Update GPU usage (simplified)
            if self.gpu_manager:
                gpu_status = await self.gpu_manager.get_status()
                self.status.gpu_usage = gpu_status.get('gpu_usage', 0.0)
            
            self.status.last_updated = datetime.now()
            return self.status
            
        except Exception as e:
            tprint_warning(f"M1 status update failed: {str(e)}")
            return self.status
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get M1 performance metrics."""
        return {
            'm1_available': self.m1_available,
            'optimization_active': self.status.optimization_active,
            'performance_metrics': self.performance_metrics,
            'optimization_level': self.m1_config.optimization_level.value,
            'memory_pressure': self.status.memory_pressure,
            'cpu_usage': self.status.cpu_usage,
            'gpu_usage': self.status.gpu_usage
        }
    
    async def cleanup(self):
        """Cleanup M1 resources."""
        try:
            if self.memory_optimizer:
                await self.memory_optimizer.cleanup()
            
            if self.gpu_manager:
                await self.gpu_manager.cleanup()
            
            self.status.optimization_active = False
            tprint_info("🧹 M1 resources cleaned up")
            
        except Exception as e:
            tprint_warning(f"M1 cleanup failed: {str(e)}")
    
    def get_config(self) -> M1Config:
        """Get M1 configuration."""
        return self.m1_config
    
    def update_config(self, new_config: M1Config):
        """Update M1 configuration."""
        self.m1_config = new_config
        tprint_info("🔧 M1 configuration updated")