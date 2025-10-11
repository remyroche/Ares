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