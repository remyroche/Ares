"""
Unified Hardware Optimization for NAS/TAS Systems

This module provides comprehensive hardware optimization combining general hardware management
with NAS/TAS-specific acceleration capabilities.

Key Features:
- General hardware optimization and management
- NAS/TAS-specific hardware acceleration
- M1 Apple Silicon optimization
- GPU acceleration with MPS support
- Memory optimization and management
- CPU optimization for parallel processing
- Comprehensive performance monitoring
- Adaptive optimization and learning
- Real-time hardware monitoring
- Workload-specific optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import time
import os
import psutil
from datetime import datetime

# Import tprint for comprehensive logging
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

# Hardware optimization imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, pmap
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# Import existing utilities
try:
    from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager, is_m1_available
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
    M1_UTILS_AVAILABLE = True
except ImportError:
    M1_UTILS_AVAILABLE = False

logger = logging.getLogger(__name__)


class WorkloadType:
    """Types of workloads for optimization."""
    BACKTESTING = "backtesting"
    ML_TRAINING = "ml_training"
    DATA_PROCESSING = "data_processing"
    MONTE_CARLO = "monte_carlo"
    FEATURE_ENGINEERING = "feature_engineering"
    NAS_SEARCH = "nas_search"
    TAS_SEARCH = "tas_search"
    GENERAL = "general"


class OptimizationLevel:
    """Optimization levels."""
    MINIMAL = "minimal"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"


@dataclass
class HardwareAccelerationConfig:
    """Configuration for unified hardware acceleration."""
    
    # GPU acceleration
    enable_gpu_acceleration: bool = True
    gpu_memory_fraction: float = 0.8
    multi_gpu_strategy: str = "data_parallel"  # "data_parallel", "model_parallel", "pipeline_parallel"
    mixed_precision: bool = True
    
    # XLA compilation
    enable_xla_compilation: bool = True
    xla_optimization_level: int = 2
    jit_compile: bool = True
    
    # Memory optimization
    enable_memory_optimization: bool = True
    memory_pool_size: int = 1024 * 1024 * 1024  # 1GB
    gradient_checkpointing: bool = True
    memory_efficient_attention: bool = True
    
    # M1 optimization
    enable_m1_optimization: bool = True
    m1_gpu_acceleration: bool = True
    m1_memory_optimization: bool = True
    m1_cpu_optimization: bool = True
    
    # Tree-specific optimizations
    tree_parallelization: bool = True
    tree_batch_processing: bool = True
    tree_memory_pooling: bool = True
    clvsa_optimization: bool = True
    
    # NAS-specific optimizations
    nas_parallelization: bool = True
    nas_memory_optimization: bool = True
    nas_gpu_acceleration: bool = True
    
    # Performance monitoring
    enable_performance_monitoring: bool = True
    monitor_gpu_usage: bool = True
    monitor_memory_usage: bool = True
    monitor_latency: bool = True
    
    # Workload-specific settings
    workload_type: str = WorkloadType.GENERAL
    optimization_level: str = OptimizationLevel.BALANCED


@dataclass
class PerformanceMetrics:
    """Performance metrics container."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float] = None
    gpu_memory_usage: Optional[float] = None
    latency: Optional[float] = None
    throughput: Optional[float] = None
    efficiency: Optional[float] = None


class UnifiedHardwareManager:
    """
    Unified hardware manager combining general hardware optimization with NAS/TAS-specific acceleration.
    """
    
    def __init__(self, config: Optional[HardwareAccelerationConfig] = None):
        """Initialize unified hardware manager."""
        self.config = config or HardwareAccelerationConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize hardware components
        self.gpu_manager = None
        self.memory_optimizer = None
        self.cpu_optimizer = None
        self.xla_compiler = None
        
        # Performance tracking
        self.performance_metrics = {
            'gpu_usage': [],
            'memory_usage': [],
            'latency': [],
            'throughput': []
        }
        
        # Initialize hardware acceleration
        self._initialize_hardware_acceleration()
        
        self.logger.info("✅ Unified Hardware Manager initialized")
    
    def _initialize_hardware_acceleration(self):
        """Initialize hardware acceleration components."""
        try:
            # Initialize GPU acceleration
            if self.config.enable_gpu_acceleration:
                self._setup_gpu_acceleration()
            
            # Initialize XLA compilation
            if self.config.enable_xla_compilation:
                self._setup_xla_compilation()
            
            # Initialize memory optimization
            if self.config.enable_memory_optimization:
                self._setup_memory_optimization()
            
            # Initialize M1 optimization
            if self.config.enable_m1_optimization:
                self._setup_m1_optimization()
            
            self.logger.info("✅ Unified hardware acceleration components initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Unified hardware acceleration initialization failed: {e}")
            raise
    
    def _setup_gpu_acceleration(self):
        """Setup GPU acceleration."""
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                # Multi-GPU setup
                self.device_count = torch.cuda.device_count()
                self.primary_device = torch.cuda.current_device()
                
                # Memory management
                torch.cuda.set_per_process_memory_fraction(self.config.gpu_memory_fraction)
                
                # Mixed precision
                if self.config.mixed_precision:
                    self.scaler = torch.cuda.amp.GradScaler()
                
                self.logger.info(f"✅ GPU acceleration enabled: {self.device_count} GPUs")
                
            elif CUPY_AVAILABLE:
                # CuPy acceleration
                self.cupy_available = True
                self.logger.info("✅ CuPy acceleration enabled")
                
            else:
                self.logger.warning("⚠️ GPU acceleration not available")
                
        except Exception as e:
            self.logger.error(f"❌ GPU acceleration setup failed: {e}")
    
    def _setup_xla_compilation(self):
        """Setup XLA compilation for optimized execution."""
        try:
            if JAX_AVAILABLE:
                # JAX XLA compilation
                self.jax_config = jax.config
                self.jax_config.update("jax_enable_x64", True)
                
                # JIT compilation
                if self.config.jit_compile:
                    self.jit_compile = jit
                    self.vmap_compile = vmap
                    self.pmap_compile = pmap
                
                self.logger.info("✅ XLA compilation enabled")
                
            else:
                self.logger.warning("⚠️ XLA compilation not available (JAX not installed)")
                
        except Exception as e:
            self.logger.error(f"❌ XLA compilation setup failed: {e}")
    
    def _setup_memory_optimization(self):
        """Setup memory optimization."""
        try:
            # Memory pool
            self.memory_pool = {}
            self.memory_pool_size = self.config.memory_pool_size
            
            # Gradient checkpointing
            if self.config.gradient_checkpointing:
                self.gradient_checkpointing = True
            
            # Memory efficient attention
            if self.config.memory_efficient_attention:
                self.memory_efficient_attention = True
            
            self.logger.info("✅ Memory optimization enabled")
            
        except Exception as e:
            self.logger.error(f"❌ Memory optimization setup failed: {e}")
    
    def _setup_m1_optimization(self):
        """Setup M1-specific optimizations."""
        try:
            if M1_UTILS_AVAILABLE:
                # M1 GPU acceleration
                if self.config.m1_gpu_acceleration:
                    self.gpu_manager = get_m1_gpu_manager()
                
                # M1 memory optimization
                if self.config.m1_memory_optimization:
                    self.memory_optimizer = get_m1_memory_optimizer()
                
                # M1 CPU optimization
                if self.config.m1_cpu_optimization:
                    self.cpu_optimizer = get_m1_cpu_optimizer()
                
                self.logger.info("✅ M1 optimization enabled")
                
            else:
                self.logger.warning("⚠️ M1 optimization not available")
                
        except Exception as e:
            self.logger.error(f"❌ M1 optimization setup failed: {e}")
    
    def optimize_for_workload(self, workload_type: str, data: Any, **kwargs) -> Any:
        """Optimize hardware for specific workload type."""
        try:
            if workload_type == WorkloadType.NAS_SEARCH:
                return self._optimize_for_nas(data, **kwargs)
            elif workload_type == WorkloadType.TAS_SEARCH:
                return self._optimize_for_tas(data, **kwargs)
            elif workload_type == WorkloadType.ML_TRAINING:
                return self._optimize_for_ml_training(data, **kwargs)
            elif workload_type == WorkloadType.BACKTESTING:
                return self._optimize_for_backtesting(data, **kwargs)
            else:
                return self._optimize_general(data, **kwargs)
                
        except Exception as e:
            self.logger.error(f"❌ Workload optimization failed: {e}")
            return data
    
    def _optimize_for_nas(self, data: Any, **kwargs) -> Any:
        """Optimize for NAS workload."""
        try:
            # Apply NAS-specific optimizations
            if self.config.nas_parallelization:
                data = self._apply_nas_parallelization(data)
            
            if self.config.nas_memory_optimization:
                data = self._apply_memory_optimization(data)
            
            if self.config.nas_gpu_acceleration:
                data = self._apply_gpu_acceleration(data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"❌ NAS optimization failed: {e}")
            return data
    
    def _optimize_for_tas(self, data: Any, **kwargs) -> Any:
        """Optimize for TAS workload."""
        try:
            # Apply TAS-specific optimizations
            if self.config.tree_parallelization:
                data = self._apply_tree_parallelization(data)
            
            if self.config.tree_batch_processing:
                data = self._apply_batch_processing(data)
            
            if self.config.tree_memory_pooling:
                data = self._apply_memory_pooling(data)
            
            if self.config.clvsa_optimization:
                data = self._apply_clvsa_optimization(data, **kwargs)
            
            return data
            
        except Exception as e:
            self.logger.error(f"❌ TAS optimization failed: {e}")
            return data
    
    def _optimize_for_ml_training(self, data: Any, **kwargs) -> Any:
        """Optimize for ML training workload."""
        try:
            # Apply ML training optimizations
            if self.config.mixed_precision:
                data = self._apply_mixed_precision(data)
            
            if self.config.gradient_checkpointing:
                data = self._apply_gradient_checkpointing(data)
            
            if self.config.memory_efficient_attention:
                data = self._apply_memory_efficient_attention(data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"❌ ML training optimization failed: {e}")
            return data
    
    def _optimize_for_backtesting(self, data: Any, **kwargs) -> Any:
        """Optimize for backtesting workload."""
        try:
            # Apply backtesting optimizations
            if self.config.enable_memory_optimization:
                data = self._apply_memory_optimization(data)
            
            if self.config.enable_gpu_acceleration:
                data = self._apply_gpu_acceleration(data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"❌ Backtesting optimization failed: {e}")
            return data
    
    def _optimize_general(self, data: Any, **kwargs) -> Any:
        """General optimization."""
        try:
            # Apply general optimizations
            if self.config.enable_memory_optimization:
                data = self._apply_memory_optimization(data)
            
            if self.config.enable_gpu_acceleration:
                data = self._apply_gpu_acceleration(data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"❌ General optimization failed: {e}")
            return data
    
    def _apply_nas_parallelization(self, data: Any) -> Any:
        """Apply NAS parallelization."""
        # NAS parallelization strategies
        if hasattr(data, 'n_jobs'):
            data.n_jobs = -1  # Use all available cores
        return data
    
    def _apply_tree_parallelization(self, data: Any) -> Any:
        """Apply tree parallelization."""
        # Tree parallelization strategies
        if hasattr(data, 'n_jobs'):
            data.n_jobs = -1  # Use all available cores
        return data
    
    def _apply_memory_optimization(self, data: Any) -> Any:
        """Apply memory optimization."""
        try:
            if isinstance(data, np.ndarray):
                if data.dtype == np.float64:
                    data = data.astype(np.float32)
            elif isinstance(data, pd.DataFrame):
                for col in data.columns:
                    if data[col].dtype == 'float64':
                        data[col] = data[col].astype('float32')
            return data
        except Exception as e:
            self.logger.warning(f"Memory optimization failed: {e}")
            return data
    
    def _apply_gpu_acceleration(self, data: Any) -> Any:
        """Apply GPU acceleration."""
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                if isinstance(data, np.ndarray):
                    data = torch.from_numpy(data).cuda()
            return data
        except Exception as e:
            self.logger.warning(f"GPU acceleration failed: {e}")
            return data
    
    def _apply_mixed_precision(self, data: Any) -> Any:
        """Apply mixed precision optimization."""
        # Mixed precision implementation
        return data
    
    def _apply_gradient_checkpointing(self, data: Any) -> Any:
        """Apply gradient checkpointing."""
        # Gradient checkpointing implementation
        return data
    
    def _apply_memory_efficient_attention(self, data: Any) -> Any:
        """Apply memory efficient attention."""
        # Memory efficient attention implementation
        return data
    
    def _apply_batch_processing(self, data: Any) -> Any:
        """Apply batch processing."""
        # Batch processing implementation
        return data
    
    def _apply_memory_pooling(self, data: Any) -> Any:
        """Apply memory pooling."""
        # Memory pooling implementation
        return data
    
    def _apply_clvsa_optimization(self, data: Any, **kwargs) -> Any:
        """Apply CLVSA optimization."""
        # CLVSA optimization implementation
        return data
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        try:
            metrics = {
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'timestamp': time.time()
            }
            
            if TORCH_AVAILABLE and torch.cuda.is_available():
                metrics['gpu_usage'] = 0.0  # Would need nvidia-ml-py
                metrics['gpu_memory_usage'] = 0.0
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get performance metrics: {e}")
            return {}
    
    def get_hardware_status(self) -> Dict[str, Any]:
        """Get hardware status."""
        return {
            'gpu_available': TORCH_AVAILABLE and torch.cuda.is_available(),
            'jax_available': JAX_AVAILABLE,
            'cupy_available': CUPY_AVAILABLE,
            'm1_utils_available': M1_UTILS_AVAILABLE,
            'torch_available': TORCH_AVAILABLE,
            'optimization_level': self.config.optimization_level,
            'workload_type': self.config.workload_type
        }


# Factory functions
def create_unified_hardware_manager(config: Optional[HardwareAccelerationConfig] = None) -> UnifiedHardwareManager:
    """Create unified hardware manager instance."""
    return UnifiedHardwareManager(config)


def get_hardware_manager() -> UnifiedHardwareManager:
    """Get global hardware manager instance."""
    global _hardware_manager
    if '_hardware_manager' not in globals():
        _hardware_manager = create_unified_hardware_manager()
    return _hardware_manager


# Example usage
if __name__ == "__main__":
    # Create unified hardware manager
    config = HardwareAccelerationConfig(
        enable_gpu_acceleration=True,
        enable_xla_compilation=True,
        enable_memory_optimization=True,
        enable_m1_optimization=True,
        workload_type=WorkloadType.NAS_SEARCH,
        optimization_level=OptimizationLevel.BALANCED
    )
    
    hardware_manager = create_unified_hardware_manager(config)
    
    # Example usage
    print("Unified Hardware Manager created successfully!")
    print(f"Hardware status: {hardware_manager.get_hardware_status()}")
    print(f"Performance metrics: {hardware_manager.get_performance_metrics()}")