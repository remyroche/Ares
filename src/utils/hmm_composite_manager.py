#!/usr/bin/env python3
"""
HMM Composite Manager - Legacy Compatibility Layer

This file now serves as a compatibility layer for the modular HMM utilities.
The original monolithic implementation (7,051 lines) has been split into focused modules:

- src.utils.hmm.core_manager: Basic HMM operations and file management
- src.utils.hmm.optimization: Bayesian optimization and parameter tuning  
- src.utils.hmm.hardware_integration: GPU acceleration and hardware optimization

This file maintains backward compatibility while using the new modular structure.
For new code, please use the modular components directly from src.utils.hmm.

DEPRECATED: This monolithic approach is deprecated. Use src.utils.hmm modules instead.
"""

import warnings
warnings.warn(
    "hmm_composite_manager.py is deprecated. Use src.utils.hmm modules instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import the new modular components
try:
    from .hmm import (
        EnhancedHMMCompositeManager as _EnhancedHMMCompositeManager,
        HMMCoreManager,
        HMMBayesianOptimizer,
        HMMHardwareManager,
        get_hmm_composite_manager
    )
    MODULAR_HMM_AVAILABLE = True
except ImportError:
    MODULAR_HMM_AVAILABLE = False

# Backward compatibility imports
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

from .logger import system_logger


# Legacy configuration classes for backward compatibility
@dataclass
class HMMRegimeConfig:
    """Legacy HMM regime configuration - use src.utils.hmm_config.UnifiedHMMConfig instead."""
    n_components: int = 3
    covariance_type: str = "full"
    n_iter: int = 100
    random_state: int = 42


@dataclass
class BayesianOptimizationConfig:
    """Legacy Bayesian optimization config - use src.utils.hmm.optimization instead."""
    n_trials: int = 50
    timeout_minutes: int = 15


@dataclass  
class FeatureEngineeringConfig:
    """Legacy feature engineering config - use src.utils.hmm_config instead."""
    lookback_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50])


@dataclass
class ValidationConfig:
    """Legacy validation config - use src.utils.hmm_config instead."""
    min_regime_duration: int = 10


# Main compatibility class
class EnhancedHMMCompositeManager:
    """
    Legacy HMM Composite Manager - Compatibility Layer
    
    This class provides backward compatibility with the original monolithic
    hmm_composite_manager.py while delegating to the new modular components.
    
    DEPRECATED: Use src.utils.hmm modules directly instead.
    """
    
    def __init__(self):
        """Initialize the legacy composite manager."""
        self.logger = system_logger.getChild('EnhancedHMMCompositeManager')
        
        if MODULAR_HMM_AVAILABLE:
            self._manager = _EnhancedHMMCompositeManager()
            self.logger.info("Using new modular HMM components")
        else:
            self.logger.error("Modular HMM components not available")
            self._manager = None
    
    # File management methods
    def get_composite_cluster_file_path(self, symbol: str, exchange: str, timeframe: str, 
                                       data_dir: str, file_type: str = 'parquet') -> str:
        """Get path for composite cluster files."""
        if self._manager:
            return self._manager.get_composite_cluster_file_path(
                symbol, exchange, timeframe, data_dir, file_type
            )
        # Fallback implementation
        filename = f"{exchange}_{symbol}_{timeframe}_composite_clusters.{file_type}"
        return f"{data_dir}/{filename}"
    
    def file_exists(self, filepath: str) -> bool:
        """Check if file exists."""
        if self._manager:
            return self._manager.file_exists(filepath)
        import os
        return os.path.exists(filepath) and os.path.getsize(filepath) > 0
    
    def load_composite_clusters(self, symbol: str, exchange: str, timeframe: str, 
                               data_dir: str) -> Optional[pd.DataFrame]:
        """Load composite cluster data."""
        if self._manager:
            return self._manager.load_composite_clusters(symbol, exchange, timeframe, data_dir)
        return None
    
    def save_composite_clusters(self, data: pd.DataFrame, symbol: str, exchange: str, 
                               timeframe: str, data_dir: str, 
                               metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Save composite cluster data."""
        if self._manager:
            return self._manager.save_composite_clusters(
                data, symbol, exchange, timeframe, data_dir, metadata
            )
        return False
    
    # Optimization methods
    def optimize_hmm_parameters(self, data: pd.DataFrame, objective_function, 
                               fixed_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Optimize HMM parameters."""
        if self._manager:
            return self._manager.optimize_hmm_parameters(data, objective_function, fixed_params)
        return {
            'n_components': 3,
            'covariance_type': 'full', 
            'n_iter': 100,
            'random_state': 42
        }
    
    # Hardware optimization methods
    def gpu_accelerated_hmm_training(self, data: np.ndarray, n_components: int, 
                                    **kwargs) -> Dict[str, Any]:
        """GPU-accelerated HMM training."""
        if self._manager:
            return self._manager.gpu_accelerated_hmm_training(data, n_components, **kwargs)
        return {}
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        if self._manager:
            return self._manager.get_memory_usage()
        return {}
    
    def cleanup_gpu_memory(self) -> None:
        """Clean up GPU memory."""
        if self._manager:
            self._manager.cleanup_gpu_memory()
    
    # Legacy methods that were in the original file
    def vectorized_hmm_training_batch(self, features_list: List[np.ndarray], 
                                     n_components_list: List[int], 
                                     **kwargs) -> List[Dict[str, Any]]:
        """Legacy vectorized training method."""
        self.logger.warning("vectorized_hmm_training_batch is deprecated")
        results = []
        for features, n_comp in zip(features_list, n_components_list):
            result = self.gpu_accelerated_hmm_training(features, n_comp, **kwargs)
            results.append(result)
        return results
    
    def train_hmm_parallel(self, data_list: List[pd.DataFrame], 
                          config_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Legacy parallel training method."""
        self.logger.warning("train_hmm_parallel is deprecated")
        results = []
        for data, config in zip(data_list, config_list):
            # Convert DataFrame to numpy array
            if isinstance(data, pd.DataFrame):
                data_array = data.select_dtypes(include=[np.number]).values
            else:
                data_array = data
            
            result = self.gpu_accelerated_hmm_training(
                data_array, 
                config.get('n_components', 3),
                **config
            )
            results.append(result)
        return results
    
    def perform_hmm_clustering(self, data: pd.DataFrame, 
                              config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Legacy HMM clustering method."""
        self.logger.warning("perform_hmm_clustering is deprecated")
        config = config or {}
        
        if isinstance(data, pd.DataFrame):
            data_array = data.select_dtypes(include=[np.number]).values
        else:
            data_array = data
        
        return self.gpu_accelerated_hmm_training(
            data_array,
            config.get('n_components', 3),
            **config
        )


# Factory function for backward compatibility
def get_hmm_composite_manager() -> EnhancedHMMCompositeManager:
    """
    Get an instance of the HMM composite manager.
    
    DEPRECATED: Use src.utils.hmm.get_hmm_composite_manager() instead.
    """
    warnings.warn(
        "get_hmm_composite_manager from hmm_composite_manager is deprecated. "
        "Use src.utils.hmm.get_hmm_composite_manager() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    if MODULAR_HMM_AVAILABLE:
        return get_hmm_composite_manager()
    else:
        return EnhancedHMMCompositeManager()


# Export the main class for backward compatibility
__all__ = ['EnhancedHMMCompositeManager', 'get_hmm_composite_manager']