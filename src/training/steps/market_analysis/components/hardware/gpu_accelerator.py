"""
GPU Accelerator for Market Analysis Components.

This module provides GPU acceleration capabilities for market analysis
pipeline steps, including CUDA operations, memory management, and
performance optimization.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
import asyncio

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error
from src.training.steps.market_analysis.components.base_component import BaseMarketAnalysisComponent, ComponentConfig

class GPUStatus(Enum):
    """GPU status levels."""
    AVAILABLE = "available"
    BUSY = "busy"
    ERROR = "error"
    UNAVAILABLE = "unavailable"

@dataclass
class GPUConfig:
    """Configuration for GPU acceleration."""
    # Memory management
    memory_fraction: float = 0.8
    allow_growth: bool = True
    enable_mixed_precision: bool = True
    
    # Performance settings
    enable_cudnn: bool = True
    enable_tensor_cores: bool = True
    optimization_level: str = "O1"  # O0, O1, O2, O3
    
    # Error handling
    max_retries: int = 3
    fallback_to_cpu: bool = True
    
    # Monitoring
    enable_memory_monitoring: bool = True
    memory_cleanup_threshold: float = 0.9

@dataclass
class GPUStatus:
    """Current GPU status."""
    available: bool
    memory_total_gb: float
    memory_used_gb: float
    memory_free_gb: float
    utilization_percent: float
    temperature_c: Optional[float] = None
    power_usage_w: Optional[float] = None

class GPUAccelerator(BaseMarketAnalysisComponent):
    """
    GPU accelerator for market analysis components.
    
    Provides:
    - GPU memory management
    - CUDA operation acceleration
    - Performance monitoring
    - Automatic fallback to CPU
    """
    
    def __init__(self, config: Optional[GPUConfig] = None):
        """Initialize the GPU accelerator."""
        super().__init__(ComponentConfig())
        self.gpu_config = config or GPUConfig()
        self.logger = logging.getLogger(__name__)
        
        # GPU availability
        self.gpu_available = False
        self.device = None
        self.gpu_status = GPUStatus(
            available=False,
            memory_total_gb=0.0,
            memory_used_gb=0.0,
            memory_free_gb=0.0,
            utilization_percent=0.0
        )
        
        # Initialize GPU
        self._initialize_gpu()
    
    def _initialize_gpu(self):
        """Initialize GPU resources."""
        try:
            import torch
            
            if torch.cuda.is_available():
                self.gpu_available = True
                self.device = torch.device('cuda')
                
                # Get GPU information
                gpu_props = torch.cuda.get_device_properties(0)
                self.gpu_status.memory_total_gb = gpu_props.total_memory / (1024**3)
                
                # Set memory fraction
                torch.cuda.set_per_process_memory_fraction(self.gpu_config.memory_fraction)
                
                # Enable optimizations
                if self.gpu_config.enable_cudnn:
                    torch.backends.cudnn.enabled = True
                    torch.backends.cudnn.benchmark = True
                
                tprint_info(f"✅ GPU initialized: {gpu_props.name} ({self.gpu_status.memory_total_gb:.1f}GB)")
            else:
                tprint_warning("❌ CUDA not available")
                
        except ImportError:
            tprint_warning("❌ PyTorch not available for GPU acceleration")
        except Exception as e:
            tprint_warning(f"❌ GPU initialization failed: {str(e)}")
    
    async def optimize_for_task(self, recommendations: Dict[str, Any]):
        """Optimize GPU for specific task."""
        try:
            if not self.gpu_available:
                return
            
            # Set batch size
            if 'batch_size' in recommendations:
                # This would be used by the calling code
                pass
            
            # Enable mixed precision if recommended
            if recommendations.get('enable_mixed_precision', False):
                self.gpu_config.enable_mixed_precision = True
            
            # Set optimization level
            if 'optimization_level' in recommendations:
                self.gpu_config.optimization_level = recommendations['optimization_level']
            
            tprint_info("🔧 GPU optimized for task")
            
        except Exception as e:
            tprint_warning(f"GPU optimization failed: {str(e)}")
    
    async def accelerate_operation(self, 
                                 operation: Callable,
                                 *args, 
                                 **kwargs) -> Any:
        """
        Accelerate an operation using GPU.
        
        Args:
            operation: Function to accelerate
            *args: Arguments for the operation
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Result of the operation
        """
        if not self.gpu_available:
            if self.gpu_config.fallback_to_cpu:
                tprint_warning("GPU not available, falling back to CPU")
                return await operation(*args, **kwargs)
            else:
                raise RuntimeError("GPU not available and fallback disabled")
        
        try:
            # Move data to GPU if it's a tensor
            gpu_args = []
            for arg in args:
                if isinstance(arg, np.ndarray):
                    import torch
                    gpu_arg = torch.from_numpy(arg).to(self.device)
                    gpu_args.append(gpu_arg)
                else:
                    gpu_args.append(arg)
            
            # Move keyword arguments to GPU
            gpu_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, np.ndarray):
                    import torch
                    gpu_value = torch.from_numpy(value).to(self.device)
                    gpu_kwargs[key] = gpu_value
                else:
                    gpu_kwargs[key] = value
            
            # Execute operation
            result = await operation(*gpu_args, **gpu_kwargs)
            
            # Move result back to CPU if it's a tensor
            if hasattr(result, 'cpu'):
                result = result.cpu().numpy()
            elif isinstance(result, (list, tuple)):
                result = type(result)([
                    item.cpu().numpy() if hasattr(item, 'cpu') else item
                    for item in result
                ])
            
            return result
            
        except Exception as e:
            tprint_warning(f"GPU operation failed: {str(e)}")
            if self.gpu_config.fallback_to_cpu:
                tprint_info("Falling back to CPU operation")
                return await operation(*args, **kwargs)
            else:
                raise
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current GPU status."""
        try:
            if not self.gpu_available:
                return {
                    'available': False,
                    'error': 'GPU not available'
                }
            
            import torch
            
            # Get memory info
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)
            memory_reserved = torch.cuda.memory_reserved() / (1024**3)
            memory_free = self.gpu_status.memory_total_gb - memory_reserved
            
            # Get utilization (simplified)
            utilization = (memory_allocated / self.gpu_status.memory_total_gb) * 100
            
            self.gpu_status.memory_used_gb = memory_allocated
            self.gpu_status.memory_free_gb = memory_free
            self.gpu_status.utilization_percent = utilization
            
            return {
                'available': True,
                'memory_total_gb': self.gpu_status.memory_total_gb,
                'memory_used_gb': memory_allocated,
                'memory_free_gb': memory_free,
                'memory_reserved_gb': memory_reserved,
                'utilization_percent': utilization,
                'gpu_usage': utilization / 100.0,
                'gpu_memory_usage': memory_allocated / self.gpu_status.memory_total_gb
            }
            
        except Exception as e:
            return {
                'available': False,
                'error': str(e)
            }
    
    async def cleanup(self):
        """Cleanup GPU resources."""
        try:
            if self.gpu_available:
                import torch
                torch.cuda.empty_cache()
                tprint_info("🧹 GPU memory cleaned up")
        except Exception as e:
            tprint_warning(f"GPU cleanup failed: {str(e)}")
    
    async def memory_cleanup(self):
        """Force GPU memory cleanup."""
        try:
            if self.gpu_available:
                import torch
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                tprint_info("🧹 GPU memory force cleaned")
        except Exception as e:
            tprint_warning(f"GPU memory cleanup failed: {str(e)}")
    
    def is_available(self) -> bool:
        """Check if GPU is available."""
        return self.gpu_available
    
    def get_device(self):
        """Get GPU device."""
        return self.device if self.gpu_available else None
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get GPU memory information."""
        if not self.gpu_available:
            return {'total': 0.0, 'used': 0.0, 'free': 0.0}
        
        return {
            'total': self.gpu_status.memory_total_gb,
            'used': self.gpu_status.memory_used_gb,
            'free': self.gpu_status.memory_free_gb
        }