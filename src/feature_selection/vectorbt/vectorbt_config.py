"""
VectorBT Feature Selection Configuration

This module provides configuration classes for VectorBT-optimized feature selection.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
import numpy as np


@dataclass
class VectorBTFeatureSelectionConfig:
    """Configuration for VectorBT feature selection operations."""
    
    # VectorBT settings
    enable_vectorbt: bool = True
    use_gpu: bool = False
    chunk_size: int = 10000
    memory_limit_mb: int = 2048
    
    # Enhanced VectorBT settings
    vectorbt_theme: str = "dark"
    vectorbt_freq_precision: int = 0
    vectorbt_array_wrapper: Dict[str, Any] = field(default_factory=lambda: {
        'freq_precision': 0,
        'freq_rep': 'auto',
        'chunk_size': 10000,
        'enable_parallel': True,
        'max_workers': None
    })
    
    # VectorBT-specific optimizations
    enable_vectorbt_rolling: bool = True
    enable_vectorbt_chunked: bool = True
    enable_vectorbt_parallel: bool = True
    vectorbt_rolling_window: int = 1000
    vectorbt_chunk_overlap: int = 100
    
    # Financial data specific optimizations
    enable_vectorbt_financial: bool = True
    vectorbt_freq_inference: bool = True
    vectorbt_resample_freq: str = '1D'
    
    # GPU Acceleration
    enable_gpu: bool = False
    gpu_memory_fraction: float = 0.8
    gpu_device: str = "cuda:0"
    gpu_chunk_size: int = 50000  # Larger chunks for GPU
    enable_cuda_optimizations: bool = True
    cuda_streams: int = 4
    cuda_memory_pool: bool = True
    
    # Intelligent Caching
    enable_caching: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600  # 1 hour
    cache_backend: str = "memory"  # memory, redis, disk
    enable_memoization: bool = True
    memoization_depth: int = 10
    
    # Advanced Parallel Processing
    enable_dask: bool = False
    dask_cluster_type: str = "local"  # local, distributed, kubernetes
    dask_workers: int = 4
    dask_memory_limit: str = "2GB"
    enable_ray: bool = False
    ray_cluster_address: Optional[str] = None
    ray_num_cpus: int = 4
    ray_num_gpus: int = 0
    
    # Advanced Memory Optimization
    enable_memory_mapping: bool = True
    memory_mapping_threshold: int = 100 * 1024 * 1024  # 100MB
    enable_lazy_evaluation: bool = True
    lazy_chunk_size: int = 1000
    enable_memory_pooling: bool = True
    memory_pool_size: int = 10
    
    # Performance settings
    enable_parallel: bool = True
    max_workers: Optional[int] = None
    enable_timing: bool = True
    log_performance: bool = True
    
    # Feature selection parameters
    correlation_threshold: float = 0.95
    variance_threshold: float = 0.01
    mutual_info_k: int = 5
    stability_threshold: float = 0.6
    n_bootstrap: int = 75
    
    # mRMR settings
    mrmr_alpha: float = 0.5
    mrmr_beta: float = 0.5
    mrmr_max_features: int = 50
    
    # Regularization settings
    l1_ratio_range: Tuple[float, float] = (0.1, 1.0)
    alpha_range: Tuple[float, float] = (0.001, 1.0)
    cv_folds: int = 5
    
    # RFE settings
    rfe_step: float = 0.1
    rfe_min_features: int = 1
    
    # Memory optimization
    enable_memory_optimization: bool = True
    enable_chunked_processing: bool = True
    lazy_evaluation: bool = True
    
    # Financial data optimization
    enable_financial_optimization: bool = True
    price_column: Optional[str] = None
    volume_column: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'enable_vectorbt': self.enable_vectorbt,
            'use_gpu': self.use_gpu,
            'chunk_size': self.chunk_size,
            'memory_limit_mb': self.memory_limit_mb,
            'enable_parallel': self.enable_parallel,
            'max_workers': self.max_workers,
            'enable_timing': self.enable_timing,
            'log_performance': self.log_performance,
            'correlation_threshold': self.correlation_threshold,
            'variance_threshold': self.variance_threshold,
            'mutual_info_k': self.mutual_info_k,
            'stability_threshold': self.stability_threshold,
            'n_bootstrap': self.n_bootstrap,
            'mrmr_alpha': self.mrmr_alpha,
            'mrmr_beta': self.mrmr_beta,
            'mrmr_max_features': self.mrmr_max_features,
            'l1_ratio_range': self.l1_ratio_range,
            'alpha_range': self.alpha_range,
            'cv_folds': self.cv_folds,
            'rfe_step': self.rfe_step,
            'rfe_min_features': self.rfe_min_features,
            'enable_memory_optimization': self.enable_memory_optimization,
            'enable_chunked_processing': self.enable_chunked_processing,
            'lazy_evaluation': self.lazy_evaluation,
            'enable_financial_optimization': self.enable_financial_optimization,
            'price_column': self.price_column,
            'volume_column': self.volume_column
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'VectorBTFeatureSelectionConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def update(self, **kwargs) -> 'VectorBTFeatureSelectionConfig':
        """Update config with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
    
    def setup_vectorbt_optimizations(self):
        """Setup VectorBT with optimal settings."""
        try:
            import vectorbt as vbt
            import logging
            
            logger = logging.getLogger(__name__)
            
            # Configure VectorBT theme
            vbt.settings.set_theme(self.vectorbt_theme)
            
            # Configure array wrapper settings
            for key, value in self.vectorbt_array_wrapper.items():
                vbt.settings['array_wrapper'][key] = value
            
            # Enable VectorBT optimizations
            if self.enable_vectorbt_rolling:
                vbt.settings['array_wrapper']['enable_rolling'] = True
            
            if self.enable_vectorbt_chunked:
                vbt.settings['array_wrapper']['enable_chunked'] = True
            
            if self.enable_vectorbt_parallel:
                vbt.settings['array_wrapper']['enable_parallel'] = True
                
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"VectorBT optimization setup failed: {e}")
    
    def setup_gpu_acceleration(self):
        """Setup GPU acceleration for VectorBT operations."""
        try:
            if self.enable_gpu:
                import torch
                import cupy as cp
                
                # Check CUDA availability
                if torch.cuda.is_available():
                    # Configure CUDA device
                    torch.cuda.set_device(self.gpu_device)
                    
                    # Configure CuPy memory pool
                    if self.cuda_memory_pool:
                        cp.cuda.MemoryPool().set_limit(fraction=self.gpu_memory_fraction)
                    
                    # Enable VectorBT GPU operations
                    import vectorbt as vbt
                    vbt.settings['array_wrapper']['enable_gpu'] = True
                    vbt.settings['array_wrapper']['gpu_chunk_size'] = self.gpu_chunk_size
                    
                    return True
                else:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning("CUDA not available, GPU acceleration disabled")
                    return False
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"GPU setup failed: {e}")
            return False
    
    def setup_advanced_parallel_processing(self):
        """Setup advanced parallel processing with Dask/Ray."""
        try:
            parallel_clients = {}
            
            if self.enable_dask:
                import dask
                from dask.distributed import Client
                
                if self.dask_cluster_type == "local":
                    dask_client = Client(
                        n_workers=self.dask_workers,
                        memory_limit=self.dask_memory_limit
                    )
                else:
                    dask_client = Client(self.dask_cluster_type)
                
                parallel_clients['dask'] = dask_client
            
            if self.enable_ray:
                import ray
                
                if not ray.is_initialized():
                    ray.init(
                        address=self.ray_cluster_address,
                        num_cpus=self.ray_num_cpus,
                        num_gpus=self.ray_num_gpus
                    )
                
                parallel_clients['ray'] = ray
            
            return parallel_clients
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Advanced parallel processing setup failed: {e}")
            return {}