"""
Memory Optimizer for Market Analysis Components.

This module provides memory optimization capabilities for market analysis
pipeline steps, including memory monitoring, cleanup, and optimization
strategies.
"""

import gc
import psutil
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
import threading
import time

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error
from src.training.steps.market_analysis.components.base_component import BaseMarketAnalysisComponent, ComponentConfig

class MemoryLevel(Enum):
    """Memory usage levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class MemoryConfig:
    """Configuration for memory optimization."""
    # Memory thresholds
    max_usage: float = 0.8  # 80% of available memory
    cleanup_threshold: float = 0.9  # 90% triggers cleanup
    warning_threshold: float = 0.7  # 70% triggers warning
    
    # Cleanup settings
    enable_auto_cleanup: bool = True
    cleanup_interval: float = 30.0  # seconds
    aggressive_cleanup_threshold: float = 0.95
    
    # Optimization settings
    enable_memory_mapping: bool = True
    enable_compression: bool = False
    chunk_size_mb: int = 100
    
    # Monitoring
    enable_monitoring: bool = True
    monitoring_interval: float = 5.0  # seconds

@dataclass
class MemoryStatus:
    """Current memory status."""
    total_gb: float
    used_gb: float
    available_gb: float
    usage_percent: float
    level: MemoryLevel
    last_updated: datetime = field(default_factory=datetime.now)

class MemoryOptimizer(BaseMarketAnalysisComponent):
    """
    Memory optimizer for market analysis components.
    
    Provides:
    - Memory monitoring and tracking
    - Automatic cleanup strategies
    - Memory optimization techniques
    - Resource management
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """Initialize the memory optimizer."""
        super().__init__(ComponentConfig())
        self.memory_config = config or MemoryConfig()
        self.logger = logging.getLogger(__name__)
        
        # Memory status
        self.status = MemoryStatus(
            total_gb=0.0,
            used_gb=0.0,
            available_gb=0.0,
            usage_percent=0.0,
            level=MemoryLevel.LOW
        )
        
        # Monitoring
        self.monitoring_thread = None
        self.monitoring_active = False
        
        # Memory tracking
        self.memory_objects = {}
        self.cleanup_history = []
        
        # Initialize memory info
        self._update_memory_status()
    
    def _update_memory_status(self):
        """Update current memory status."""
        try:
            memory_info = psutil.virtual_memory()
            self.status.total_gb = memory_info.total / (1024**3)
            self.status.used_gb = memory_info.used / (1024**3)
            self.status.available_gb = memory_info.available / (1024**3)
            self.status.usage_percent = memory_info.percent / 100.0
            self.status.last_updated = datetime.now()
            
            # Determine memory level
            if self.status.usage_percent < 0.5:
                self.status.level = MemoryLevel.LOW
            elif self.status.usage_percent < 0.7:
                self.status.level = MemoryLevel.MEDIUM
            elif self.status.usage_percent < 0.9:
                self.status.level = MemoryLevel.HIGH
            else:
                self.status.level = MemoryLevel.CRITICAL
                
        except Exception as e:
            tprint_warning(f"Memory status update failed: {str(e)}")
    
    async def optimize_memory_usage(self, 
                                  data: Union[np.ndarray, pd.DataFrame],
                                  operation_type: str = "general") -> Union[np.ndarray, pd.DataFrame]:
        """
        Optimize memory usage for data processing.
        
        Args:
            data: Data to optimize
            operation_type: Type of operation (clustering, training, etc.)
            
        Returns:
            Optimized data
        """
        try:
            tprint_info(f"🔧 Optimizing memory for {operation_type} operation")
            
            # Check current memory usage
            self._update_memory_status()
            
            if self.status.level == MemoryLevel.CRITICAL:
                await self.cleanup_memory(aggressive=True)
            
            # Apply memory optimizations based on data type
            if isinstance(data, pd.DataFrame):
                optimized_data = await self._optimize_dataframe(data, operation_type)
            elif isinstance(data, np.ndarray):
                optimized_data = await self._optimize_array(data, operation_type)
            else:
                optimized_data = data
            
            # Track memory usage
            self._track_memory_usage(optimized_data, operation_type)
            
            tprint_info(f"✅ Memory optimization completed")
            return optimized_data
            
        except Exception as e:
            tprint_error(f"❌ Memory optimization failed: {str(e)}")
            return data
    
    async def _optimize_dataframe(self, df: pd.DataFrame, operation_type: str) -> pd.DataFrame:
        """Optimize DataFrame memory usage."""
        try:
            optimized_df = df.copy()
            
            # Optimize data types
            for col in optimized_df.columns:
                if optimized_df[col].dtype == 'object':
                    # Try to convert to category if low cardinality
                    if optimized_df[col].nunique() / len(optimized_df) < 0.5:
                        optimized_df[col] = optimized_df[col].astype('category')
                elif optimized_df[col].dtype == 'float64':
                    # Try to convert to float32 if precision allows
                    if optimized_df[col].min() >= np.finfo(np.float32).min and \
                       optimized_df[col].max() <= np.finfo(np.float32).max:
                        optimized_df[col] = optimized_df[col].astype('float32')
                elif optimized_df[col].dtype == 'int64':
                    # Try to convert to smaller int types
                    if optimized_df[col].min() >= np.iinfo(np.int32).min and \
                       optimized_df[col].max() <= np.iinfo(np.int32).max:
                        optimized_df[col] = optimized_df[col].astype('int32')
                    elif optimized_df[col].min() >= np.iinfo(np.int16).min and \
                         optimized_df[col].max() <= np.iinfo(np.int16).max:
                        optimized_df[col] = optimized_df[col].astype('int16')
            
            # Remove unnecessary columns if memory is tight
            if self.status.level in [MemoryLevel.HIGH, MemoryLevel.CRITICAL]:
                # Keep only essential columns for the operation
                if operation_type == "clustering":
                    essential_cols = [col for col in optimized_df.columns 
                                    if col in ['timestamp', 'close', 'volume'] or 
                                    col.startswith('feature_')]
                    if essential_cols:
                        optimized_df = optimized_df[essential_cols]
            
            return optimized_df
            
        except Exception as e:
            tprint_warning(f"DataFrame optimization failed: {str(e)}")
            return df
    
    async def _optimize_array(self, arr: np.ndarray, operation_type: str) -> np.ndarray:
        """Optimize NumPy array memory usage."""
        try:
            # Convert to more memory-efficient dtype if possible
            if arr.dtype == np.float64:
                if arr.min() >= np.finfo(np.float32).min and \
                   arr.max() <= np.finfo(np.float32).max:
                    arr = arr.astype(np.float32)
            elif arr.dtype == np.int64:
                if arr.min() >= np.iinfo(np.int32).min and \
                   arr.max() <= np.iinfo(np.int32).max:
                    arr = arr.astype(np.int32)
                elif arr.min() >= np.iinfo(np.int16).min and \
                     arr.max() <= np.iinfo(np.int16).max:
                    arr = arr.astype(np.int16)
            
            # Use memory mapping for large arrays
            if self.memory_config.enable_memory_mapping and arr.nbytes > 100 * 1024 * 1024:  # 100MB
                # This would be implemented with np.memmap in practice
                pass
            
            return arr
            
        except Exception as e:
            tprint_warning(f"Array optimization failed: {str(e)}")
            return arr
    
    def _track_memory_usage(self, data: Any, operation_type: str):
        """Track memory usage of data objects."""
        try:
            if isinstance(data, pd.DataFrame):
                memory_usage = data.memory_usage(deep=True).sum() / (1024**2)  # MB
            elif isinstance(data, np.ndarray):
                memory_usage = data.nbytes / (1024**2)  # MB
            else:
                memory_usage = 0
            
            self.memory_objects[operation_type] = {
                'memory_mb': memory_usage,
                'timestamp': datetime.now(),
                'type': type(data).__name__
            }
            
        except Exception as e:
            tprint_warning(f"Memory tracking failed: {str(e)}")
    
    async def cleanup_memory(self, aggressive: bool = False):
        """Cleanup memory resources."""
        try:
            tprint_info("🧹 Starting memory cleanup")
            
            # Force garbage collection
            collected = gc.collect()
            
            # Clear memory objects if aggressive
            if aggressive:
                self.memory_objects.clear()
            
            # Update memory status
            self._update_memory_status()
            
            # Record cleanup
            self.cleanup_history.append({
                'timestamp': datetime.now(),
                'aggressive': aggressive,
                'collected_objects': collected,
                'memory_after_gb': self.status.used_gb
            })
            
            tprint_info(f"✅ Memory cleanup completed: {collected} objects collected")
            
        except Exception as e:
            tprint_warning(f"Memory cleanup failed: {str(e)}")
    
    async def start_monitoring(self):
        """Start memory monitoring."""
        if self.memory_config.enable_monitoring and not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            tprint_info("🔍 Memory monitoring started")
    
    async def stop_monitoring(self):
        """Stop memory monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
        tprint_info("⏹️ Memory monitoring stopped")
    
    def _monitoring_loop(self):
        """Memory monitoring loop."""
        while self.monitoring_active:
            try:
                self._update_memory_status()
                
                # Check if cleanup is needed
                if self.status.usage_percent > self.memory_config.cleanup_threshold:
                    if self.status.usage_percent > self.memory_config.aggressive_cleanup_threshold:
                        asyncio.run(self.cleanup_memory(aggressive=True))
                    else:
                        asyncio.run(self.cleanup_memory(aggressive=False))
                
                # Warning if memory usage is high
                if self.status.usage_percent > self.memory_config.warning_threshold:
                    tprint_warning(f"⚠️ High memory usage: {self.status.usage_percent:.1%}")
                
                time.sleep(self.memory_config.monitoring_interval)
                
            except Exception as e:
                tprint_warning(f"Memory monitoring error: {str(e)}")
                time.sleep(5.0)
    
    def get_memory_status(self) -> MemoryStatus:
        """Get current memory status."""
        self._update_memory_status()
        return self.status
    
    def get_memory_usage_by_operation(self) -> Dict[str, Any]:
        """Get memory usage breakdown by operation."""
        return {
            'operations': self.memory_objects,
            'cleanup_history': self.cleanup_history[-10:],  # Last 10 cleanups
            'current_status': {
                'total_gb': self.status.total_gb,
                'used_gb': self.status.used_gb,
                'available_gb': self.status.available_gb,
                'usage_percent': self.status.usage_percent,
                'level': self.status.level.value
            }
        }
    
    def estimate_memory_requirements(self, 
                                   data_size: int, 
                                   operation_type: str) -> Dict[str, float]:
        """Estimate memory requirements for an operation."""
        try:
            # Base memory per data point (rough estimates)
            base_memory_per_point = {
                'clustering': 0.001,  # 1KB per point
                'training': 0.01,     # 10KB per point
                'inference': 0.005,   # 5KB per point
                'general': 0.002      # 2KB per point
            }
            
            memory_per_point = base_memory_per_point.get(operation_type, 0.002)
            estimated_memory_gb = (data_size * memory_per_point) / 1024  # Convert to GB
            
            # Add overhead
            overhead_multiplier = 1.5
            total_estimated_gb = estimated_memory_gb * overhead_multiplier
            
            return {
                'estimated_gb': total_estimated_gb,
                'base_memory_gb': estimated_memory_gb,
                'overhead_gb': estimated_memory_gb * 0.5,
                'fits_in_memory': total_estimated_gb < self.status.available_gb,
                'recommended_batch_size': min(data_size, int(self.status.available_gb * 0.5 / memory_per_point * 1024))
            }
            
        except Exception as e:
            tprint_warning(f"Memory estimation failed: {str(e)}")
            return {
                'estimated_gb': 0.0,
                'base_memory_gb': 0.0,
                'overhead_gb': 0.0,
                'fits_in_memory': False,
                'recommended_batch_size': 1000
            }
    
    def get_config(self) -> MemoryConfig:
        """Get memory configuration."""
        return self.memory_config
    
    def update_config(self, new_config: MemoryConfig):
        """Update memory configuration."""
        self.memory_config = new_config
        tprint_info("🔧 Memory configuration updated")