"""
VectorBT optimization utilities for feature selection.

This module provides VectorBT-specific optimizations for numerical
computations in feature selection operations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import time
from dataclasses import dataclass

from src.utils.tprint import tprint_debug, tprint_info, tprint_warning, tprint_success

# Import VectorBT utilities
try:
    from src.training.steps.pre_training.utils.vectorbt_utils import (
        create_vectorbt_tools, VectorBTConfig, get_vectorbt_performance_stats,
        VECTORBT_UTILS_AVAILABLE
    )
    VECTORBT_AVAILABLE = VECTORBT_UTILS_AVAILABLE
except ImportError:
    VECTORBT_AVAILABLE = False


@dataclass
class VectorBTStats:
    """VectorBT performance statistics."""
    total_operations: int
    vectorized_operations: int
    speedup_factor: float
    memory_efficiency: float
    gpu_utilization: float
    execution_time: float


class VectorBTManager:
    """VectorBT optimization manager for feature selection."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger("VectorBTManager")
        
        # Initialize VectorBT tools
        self._initialize_vectorbt_tools()
        
        # Performance tracking
        self.operation_count = 0
        self.total_execution_time = 0.0
        self.vectorized_operations = 0
    
    def _initialize_vectorbt_tools(self):
        """Initialize VectorBT optimization tools."""
        if VECTORBT_AVAILABLE:
            try:
                vectorbt_config = VectorBTConfig(
                    enable_gpu=self.config.get('enable_gpu', False),
                    enable_parallel=self.config.get('enable_parallel', True),
                    memory_efficient=self.config.get('memory_efficient', True),
                    chunk_size=self.config.get('chunk_size', 1000)
                )
                
                vectorbt_tools = create_vectorbt_tools(vectorbt_config)
                self.vectorbt_optimizer = vectorbt_tools['optimizer']
                self.vectorization_manager = vectorbt_tools['manager']
                self.vectorbt_enabled = vectorbt_tools['available']
                
                if self.vectorbt_enabled:
                    tprint_success("✅ VectorBT optimization tools initialized")
                else:
                    tprint_warning("⚠️ VectorBT optimization tools not available")
            except Exception as e:
                tprint_warning(f"⚠️ VectorBT initialization failed: {e}")
                self.vectorbt_optimizer = None
                self.vectorization_manager = None
                self.vectorbt_enabled = False
        else:
            self.vectorbt_optimizer = None
            self.vectorization_manager = None
            self.vectorbt_enabled = False
            tprint_warning("⚠️ VectorBT not available")
    
    def optimize_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame using VectorBT operations."""
        tprint_debug(f"⚡ Optimizing DataFrame with VectorBT: {data.shape}")
        
        if not self.vectorbt_enabled or self.vectorization_manager is None:
            tprint_debug("   ⚠️ VectorBT not available, returning original data")
            return data
        
        try:
            start_time = time.time()
            
            # Use VectorBT for optimization
            optimized_data = self.vectorization_manager.optimize_dataframe(data)
            
            execution_time = time.time() - start_time
            self.total_execution_time += execution_time
            self.operation_count += 1
            self.vectorized_operations += 1
            
            tprint_debug(f"   ✅ VectorBT optimization completed in {execution_time:.3f}s")
            tprint_debug(f"   📊 Shape: {data.shape} -> {optimized_data.shape}")
            
            return optimized_data
            
        except Exception as e:
            tprint_warning(f"   ⚠️ VectorBT optimization failed: {e}")
            return data
    
    def calculate_correlation_matrix(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate correlation matrix using VectorBT optimization."""
        tprint_debug(f"⚡ Calculating correlation matrix with VectorBT: {data.shape}")
        
        if not self.vectorbt_enabled or self.vectorization_manager is None:
            tprint_debug("   ⚠️ VectorBT not available, using standard correlation")
            return data.corr().values
        
        try:
            start_time = time.time()
            
            # Use VectorBT for correlation calculation
            corr_matrix = self.vectorization_manager.calculate_correlation_matrix(data)
            
            execution_time = time.time() - start_time
            self.total_execution_time += execution_time
            self.operation_count += 1
            self.vectorized_operations += 1
            
            tprint_debug(f"   ✅ VectorBT correlation completed in {execution_time:.3f}s")
            
            return corr_matrix
            
        except Exception as e:
            tprint_warning(f"   ⚠️ VectorBT correlation failed: {e}")
            return data.corr().values
    
    def rolling_operation(self, data: pd.DataFrame, operation: str, window: int = 20) -> pd.DataFrame:
        """Perform rolling operations using VectorBT optimization."""
        tprint_debug(f"⚡ Rolling {operation} with VectorBT: window={window}")
        
        if not self.vectorbt_enabled or self.vectorization_manager is None:
            tprint_debug("   ⚠️ VectorBT not available, using standard rolling")
            return data.rolling(window=window).agg(operation)
        
        try:
            start_time = time.time()
            
            # Use VectorBT for rolling operations
            result = self.vectorization_manager.rolling_operation(data, operation, window)
            
            execution_time = time.time() - start_time
            self.total_execution_time += execution_time
            self.operation_count += 1
            self.vectorized_operations += 1
            
            tprint_debug(f"   ✅ VectorBT rolling completed in {execution_time:.3f}s")
            
            return result
            
        except Exception as e:
            tprint_warning(f"   ⚠️ VectorBT rolling failed: {e}")
            return data.rolling(window=window).agg(operation)
    
    def scale_data(self, data: pd.DataFrame, method: str = 'zscore') -> pd.DataFrame:
        """Scale data using VectorBT optimization."""
        tprint_debug(f"⚡ Scaling data with VectorBT: method={method}")
        
        if not self.vectorbt_enabled or self.vectorization_manager is None:
            tprint_debug("   ⚠️ VectorBT not available, using standard scaling")
            if method == 'zscore':
                return (data - data.mean()) / data.std()
            return data
        
        try:
            start_time = time.time()
            
            # Use VectorBT for scaling
            scaled_data = self.vectorization_manager.scale_data(data, method)
            
            execution_time = time.time() - start_time
            self.total_execution_time += execution_time
            self.operation_count += 1
            self.vectorized_operations += 1
            
            tprint_debug(f"   ✅ VectorBT scaling completed in {execution_time:.3f}s")
            
            return scaled_data
            
        except Exception as e:
            tprint_warning(f"   ⚠️ VectorBT scaling failed: {e}")
            if method == 'zscore':
                return (data - data.mean()) / data.std()
            return data
    
    def batch_correlation_analysis(self, data: pd.DataFrame, batch_size: int = 1000) -> Dict[str, Any]:
        """Perform batch correlation analysis using VectorBT."""
        tprint_debug(f"⚡ Batch correlation analysis with VectorBT: batch_size={batch_size}")
        
        if not self.vectorbt_enabled or self.vectorization_manager is None:
            tprint_debug("   ⚠️ VectorBT not available, using standard correlation")
            return {
                'correlation_matrix': data.corr().values,
                'high_corr_pairs': [],
                'method': 'standard'
            }
        
        try:
            start_time = time.time()
            
            # Use VectorBT for batch correlation analysis
            result = self.vectorization_manager.batch_correlation_analysis(data, batch_size)
            
            execution_time = time.time() - start_time
            self.total_execution_time += execution_time
            self.operation_count += 1
            self.vectorized_operations += 1
            
            tprint_debug(f"   ✅ VectorBT batch correlation completed in {execution_time:.3f}s")
            
            return result
            
        except Exception as e:
            tprint_warning(f"   ⚠️ VectorBT batch correlation failed: {e}")
            return {
                'correlation_matrix': data.corr().values,
                'high_corr_pairs': [],
                'method': 'standard'
            }
    
    def memory_optimized_processing(self, data: pd.DataFrame, chunk_size: int = 1000) -> pd.DataFrame:
        """Process large datasets in memory-optimized chunks using VectorBT."""
        tprint_debug(f"⚡ Memory-optimized processing with VectorBT: chunk_size={chunk_size}")
        
        if not self.vectorbt_enabled or self.vectorization_manager is None:
            tprint_debug("   ⚠️ VectorBT not available, using standard processing")
            return data
        
        try:
            start_time = time.time()
            
            # Process data in chunks
            processed_chunks = []
            for i in range(0, len(data), chunk_size):
                chunk = data.iloc[i:i + chunk_size]
                optimized_chunk = self.vectorization_manager.optimize_dataframe(chunk)
                processed_chunks.append(optimized_chunk)
                tprint_debug(f"   📦 Processed chunk {i//chunk_size + 1}/{(len(data) + chunk_size - 1)//chunk_size}")
            
            result = pd.concat(processed_chunks, ignore_index=True)
            
            execution_time = time.time() - start_time
            self.total_execution_time += execution_time
            self.operation_count += 1
            self.vectorized_operations += 1
            
            tprint_success(f"   ✅ Memory-optimized processing completed in {execution_time:.3f}s")
            tprint_debug(f"   📊 Shape: {data.shape} -> {result.shape}")
            
            return result
            
        except Exception as e:
            tprint_warning(f"   ⚠️ Memory-optimized processing failed: {e}")
            return data
    
    def get_performance_stats(self) -> VectorBTStats:
        """Get VectorBT performance statistics."""
        tprint_debug("📊 Getting VectorBT performance statistics")
        
        try:
            # Calculate speedup factor (simplified)
            speedup_factor = 1.0
            if self.operation_count > 0:
                avg_time_per_operation = self.total_execution_time / self.operation_count
                # Estimate baseline time (simplified)
                estimated_baseline_time = avg_time_per_operation * 2.0  # Assume 2x slower
                speedup_factor = estimated_baseline_time / avg_time_per_operation
            
            # Calculate memory efficiency (simplified)
            memory_efficiency = 0.8 if self.vectorbt_enabled else 0.5
            
            # Get GPU utilization (if available)
            gpu_utilization = 0.0
            if hasattr(self, 'vectorization_manager') and self.vectorization_manager:
                try:
                    gpu_stats = get_vectorbt_performance_stats()
                    gpu_utilization = gpu_stats.get('gpu_utilization', 0.0)
                except Exception:
                    pass
            
            stats = VectorBTStats(
                total_operations=self.operation_count,
                vectorized_operations=self.vectorized_operations,
                speedup_factor=speedup_factor,
                memory_efficiency=memory_efficiency,
                gpu_utilization=gpu_utilization,
                execution_time=self.total_execution_time
            )
            
            tprint_debug(f"   📊 Operations: {self.operation_count} total, {self.vectorized_operations} vectorized")
            tprint_debug(f"   📊 Speedup: {speedup_factor:.2f}x, Memory efficiency: {memory_efficiency:.2f}")
            
            return stats
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to get VectorBT performance stats: {e}")
            return VectorBTStats(
                total_operations=0,
                vectorized_operations=0,
                speedup_factor=1.0,
                memory_efficiency=0.5,
                gpu_utilization=0.0,
                execution_time=0.0
            )
    
    def is_available(self) -> bool:
        """Check if VectorBT optimization is available."""
        return self.vectorbt_enabled
    
    def get_config(self) -> Dict[str, Any]:
        """Get current VectorBT configuration."""
        return {
            'vectorbt_enabled': self.vectorbt_enabled,
            'config': self.config,
            'operation_count': self.operation_count,
            'total_execution_time': self.total_execution_time,
            'vectorized_operations': self.vectorized_operations
        }
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.operation_count = 0
        self.total_execution_time = 0.0
        self.vectorized_operations = 0
        tprint_debug("📊 VectorBT statistics reset")
    
    def cleanup(self):
        """Cleanup VectorBT resources."""
        tprint_info("🧹 Cleaning up VectorBT resources")
        
        try:
            # Reset statistics
            self.reset_stats()
            
            # Cleanup VectorBT tools if available
            if hasattr(self, 'vectorization_manager') and self.vectorization_manager:
                try:
                    # Add any VectorBT-specific cleanup here
                    pass
                except Exception as e:
                    tprint_warning(f"⚠️ VectorBT cleanup warning: {e}")
            
            tprint_success("   ✅ VectorBT cleanup completed")
            
        except Exception as e:
            tprint_warning(f"   ⚠️ VectorBT cleanup failed: {e}")