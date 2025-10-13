"""
Memory management utilities for feature selection.

This module provides memory optimization and management capabilities
specifically designed for large-scale feature selection operations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import gc
import psutil
import time
from dataclasses import dataclass

from src.utils.tprint import tprint_debug, tprint_info, tprint_warning, tprint_success

# Import hardware optimization tools
try:
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
    from src.utils.hardware.adaptive_optimization_engine import AdaptiveOptimizationEngine
    from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager
    from src.utils.hardware.advanced_memory_optimizer import AdvancedM1MemoryOptimizer, MemoryStrategy
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    HARDWARE_OPTIMIZATION_AVAILABLE = False


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_memory_gb: float
    available_memory_gb: float
    used_memory_gb: float
    memory_percentage: float
    memory_pressure: float
    cleanup_recommended: bool


class MemoryManager:
    """Memory management for feature selection operations."""
    
    def __init__(self, memory_limit_gb: float = 8.0, strategy: str = 'aggressive'):
        self.memory_limit_gb = memory_limit_gb
        self.strategy = strategy
        self.logger = get_logger("MemoryManager")
        
        # Initialize hardware optimization tools
        self._initialize_hardware_tools()
        
        # Memory tracking
        self.initial_memory = self._get_memory_usage()
        self.peak_memory = self.initial_memory
        self.cleanup_count = 0
    
    def _initialize_hardware_tools(self):
        """Initialize hardware optimization tools."""
        if HARDWARE_OPTIMIZATION_AVAILABLE:
            try:
                self.memory_optimizer = get_m1_memory_optimizer(memory_limit_gb=self.memory_limit_gb)
                self.adaptive_engine = AdaptiveOptimizationEngine()
                self.hardware_manager = UnifiedHardwareManager()
                
                # Initialize advanced memory optimizer
                memory_strategy = MemoryStrategy.AGGRESSIVE if self.strategy == 'aggressive' else MemoryStrategy.CONSERVATIVE
                self.advanced_memory_optimizer = AdvancedM1MemoryOptimizer(
                    memory_limit_gb=self.memory_limit_gb,
                    strategy=memory_strategy
                )
                
                tprint_success("✅ Hardware memory optimization tools initialized")
            except Exception as e:
                tprint_warning(f"⚠️ Hardware memory optimization tools not available: {e}")
                self.memory_optimizer = None
                self.adaptive_engine = None
                self.hardware_manager = None
                self.advanced_memory_optimizer = None
        else:
            self.memory_optimizer = None
            self.adaptive_engine = None
            self.hardware_manager = None
            self.advanced_memory_optimizer = None
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / (1024 ** 3)  # Convert to GB
        except Exception:
            return 0.0
    
    def get_memory_stats(self) -> MemoryStats:
        """Get comprehensive memory statistics."""
        tprint_debug("📊 Getting memory statistics")
        
        try:
            # Get system memory info
            memory = psutil.virtual_memory()
            total_memory_gb = memory.total / (1024 ** 3)
            available_memory_gb = memory.available / (1024 ** 3)
            used_memory_gb = memory.used / (1024 ** 3)
            memory_percentage = memory.percent / 100.0
            
            # Calculate memory pressure
            memory_pressure = used_memory_gb / total_memory_gb
            
            # Determine if cleanup is recommended
            cleanup_recommended = (
                memory_pressure > 0.8 or  # 80% memory usage
                used_memory_gb > self.memory_limit_gb or  # Exceeded limit
                available_memory_gb < 1.0  # Less than 1GB available
            )
            
            stats = MemoryStats(
                total_memory_gb=total_memory_gb,
                available_memory_gb=available_memory_gb,
                used_memory_gb=used_memory_gb,
                memory_percentage=memory_percentage,
                memory_pressure=memory_pressure,
                cleanup_recommended=cleanup_recommended
            )
            
            # Update peak memory tracking
            current_memory = self._get_memory_usage()
            if current_memory > self.peak_memory:
                self.peak_memory = current_memory
            
            tprint_debug(f"   📊 Memory: {used_memory_gb:.2f}GB/{total_memory_gb:.2f}GB ({memory_percentage:.1%})")
            tprint_debug(f"   📊 Pressure: {memory_pressure:.2f}, Cleanup recommended: {cleanup_recommended}")
            
            return stats
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to get memory stats: {e}")
            return MemoryStats(
                total_memory_gb=0.0,
                available_memory_gb=0.0,
                used_memory_gb=0.0,
                memory_percentage=0.0,
                memory_pressure=0.0,
                cleanup_recommended=False
            )
    
    def monitor_memory_pressure(self) -> Dict[str, Any]:
        """Monitor memory pressure and return recommendations."""
        tprint_debug("🧠 Monitoring memory pressure")
        
        try:
            stats = self.get_memory_stats()
            
            recommendations = []
            if stats.cleanup_recommended:
                recommendations.extend([
                    'Reduce batch size',
                    'Enable memory optimization',
                    'Perform aggressive cleanup',
                    'Consider chunked processing'
                ])
            
            if stats.memory_pressure > 0.9:
                recommendations.append('Critical: Immediate cleanup required')
            
            if stats.available_memory_gb < 0.5:
                recommendations.append('Very low available memory')
            
            result = {
                'pressure': stats.memory_pressure,
                'cleanup_triggered': stats.cleanup_recommended,
                'recommendations': recommendations,
                'stats': stats
            }
            
            tprint_debug(f"   📊 Memory pressure: {stats.memory_pressure:.2f}")
            tprint_debug(f"   📊 Recommendations: {len(recommendations)}")
            
            return result
            
        except Exception as e:
            tprint_warning(f"⚠️ Memory pressure monitoring failed: {e}")
            return {
                'pressure': 0.0,
                'cleanup_triggered': False,
                'recommendations': [],
                'stats': None
            }
    
    def perform_cleanup(self, force: bool = False) -> Dict[str, Any]:
        """Perform memory cleanup."""
        tprint_info("🧹 Performing memory cleanup")
        
        try:
            # Get memory stats before cleanup
            stats_before = self.get_memory_stats()
            
            # Perform garbage collection
            collected = gc.collect()
            
            # Use advanced memory optimizer if available
            memory_freed_mb = 0.0
            if self.advanced_memory_optimizer:
                try:
                    cleanup_result = self.advanced_memory_optimizer.cleanup(force=force)
                    memory_freed_mb = cleanup_result.get('memory_freed_mb', 0.0)
                except Exception as e:
                    tprint_warning(f"⚠️ Advanced memory cleanup failed: {e}")
            
            # Get memory stats after cleanup
            stats_after = self.get_memory_stats()
            
            # Calculate memory freed
            memory_freed_gb = stats_before.used_memory_gb - stats_after.used_memory_gb
            memory_freed_mb = memory_freed_gb * 1024
            
            # Update cleanup count
            self.cleanup_count += 1
            
            result = {
                'memory_freed_mb': memory_freed_mb,
                'memory_freed_gb': memory_freed_gb,
                'garbage_collected': collected,
                'cleanup_count': self.cleanup_count,
                'success': True,
                'stats_before': stats_before,
                'stats_after': stats_after
            }
            
            tprint_success(f"   ✅ Memory cleanup completed: {memory_freed_mb:.1f}MB freed")
            tprint_debug(f"   📊 Garbage collected: {collected} objects")
            
            return result
            
        except Exception as e:
            tprint_warning(f"⚠️ Memory cleanup failed: {e}")
            return {
                'memory_freed_mb': 0.0,
                'memory_freed_gb': 0.0,
                'garbage_collected': 0,
                'cleanup_count': self.cleanup_count,
                'success': False,
                'error': str(e)
            }
    
    def optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage."""
        tprint_debug(f"🔧 Optimizing DataFrame memory: {df.shape}")
        
        try:
            # Get initial memory usage
            initial_memory = df.memory_usage(deep=True).sum() / (1024 ** 2)  # MB
            
            # Optimize numeric columns
            for col in df.select_dtypes(include=[np.number]).columns:
                col_type = df[col].dtype
                
                if col_type != np.object_:
                    c_min = df[col].min()
                    c_max = df[col].max()
                    
                    if str(col_type)[:3] == 'int':
                        if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                            df[col] = df[col].astype(np.int8)
                        elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                            df[col] = df[col].astype(np.int16)
                        elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                            df[col] = df[col].astype(np.int32)
                        elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                            df[col] = df[col].astype(np.int64)
                    else:
                        if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                            df[col] = df[col].astype(np.float16)
                        elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                            df[col] = df[col].astype(np.float32)
                        else:
                            df[col] = df[col].astype(np.float64)
            
            # Optimize object columns
            for col in df.select_dtypes(include=['object']).columns:
                num_unique_values = len(df[col].unique())
                num_total_values = len(df[col])
                
                if num_unique_values / num_total_values < 0.5:
                    df[col] = df[col].astype('category')
            
            # Get final memory usage
            final_memory = df.memory_usage(deep=True).sum() / (1024 ** 2)  # MB
            memory_reduction = initial_memory - final_memory
            reduction_percentage = (memory_reduction / initial_memory) * 100
            
            tprint_debug(f"   ✅ Memory optimization: {initial_memory:.1f}MB -> {final_memory:.1f}MB")
            tprint_debug(f"   📊 Reduction: {memory_reduction:.1f}MB ({reduction_percentage:.1f}%)")
            
            return df
            
        except Exception as e:
            tprint_warning(f"⚠️ DataFrame memory optimization failed: {e}")
            return df
    
    def chunked_processing(self, data: pd.DataFrame, chunk_size: int = 1000) -> List[pd.DataFrame]:
        """Split data into memory-efficient chunks."""
        tprint_debug(f"📦 Creating chunks of size {chunk_size}")
        
        try:
            chunks = []
            for i in range(0, len(data), chunk_size):
                chunk = data.iloc[i:i + chunk_size].copy()
                # Optimize chunk memory
                chunk = self.optimize_dataframe_memory(chunk)
                chunks.append(chunk)
            
            tprint_debug(f"   ✅ Created {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            tprint_warning(f"⚠️ Chunked processing failed: {e}")
            return [data]
    
    def get_memory_recommendations(self, data_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Get memory recommendations based on data shape."""
        tprint_debug(f"💡 Getting memory recommendations for shape {data_shape}")
        
        try:
            n_samples, n_features = data_shape
            estimated_memory_gb = (n_samples * n_features * 8) / (1024 ** 3)  # Assuming float64
            
            recommendations = {
                'estimated_memory_gb': estimated_memory_gb,
                'chunk_size': min(1000, max(100, n_samples // 10)),
                'use_chunked_processing': estimated_memory_gb > self.memory_limit_gb * 0.5,
                'enable_memory_optimization': estimated_memory_gb > 1.0,
                'aggressive_cleanup': estimated_memory_gb > self.memory_limit_gb
            }
            
            if estimated_memory_gb > self.memory_limit_gb:
                recommendations['warnings'] = [
                    f'Data size ({estimated_memory_gb:.2f}GB) exceeds memory limit ({self.memory_limit_gb}GB)',
                    'Consider using chunked processing',
                    'Enable aggressive memory cleanup'
                ]
            
            tprint_debug(f"   💡 Estimated memory: {estimated_memory_gb:.2f}GB")
            tprint_debug(f"   💡 Chunked processing: {recommendations['use_chunked_processing']}")
            
            return recommendations
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to get memory recommendations: {e}")
            return {
                'estimated_memory_gb': 0.0,
                'chunk_size': 1000,
                'use_chunked_processing': True,
                'enable_memory_optimization': True,
                'aggressive_cleanup': False
            }
    
    def cleanup_resources(self):
        """Cleanup all resources and reset state."""
        tprint_info("🧹 Cleaning up memory manager resources")
        
        try:
            # Perform final cleanup
            self.perform_cleanup(force=True)
            
            # Reset tracking
            self.peak_memory = 0.0
            self.cleanup_count = 0
            
            tprint_success("   ✅ Memory manager cleanup completed")
            
        except Exception as e:
            tprint_warning(f"   ⚠️ Memory manager cleanup failed: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of memory management."""
        current_stats = self.get_memory_stats()
        
        return {
            'initial_memory_gb': self.initial_memory,
            'peak_memory_gb': self.peak_memory,
            'current_memory_gb': self._get_memory_usage(),
            'cleanup_count': self.cleanup_count,
            'current_stats': current_stats,
            'memory_efficiency': 1.0 - (current_stats.memory_pressure),
            'optimization_active': HARDWARE_OPTIMIZATION_AVAILABLE
        }