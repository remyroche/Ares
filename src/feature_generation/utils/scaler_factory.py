"""
Scaler Factory for Centralized Scaling Operations

This module provides a factory for creating and managing scalers from
features_common, eliminating code duplication in feature generators.
"""

import logging
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
from enum import Enum

from ..features_common.transforms.base_scaler import BaseScaler
from ..features_common.transforms.vectorbt_scaler import VectorBTScaler, VectorBTBatchScaler

logger = logging.getLogger(__name__)

class ScalerType(Enum):
    """Enum for supported scaler types."""
    ZSCORE = "zscore"
    MINMAX = "minmax"
    ROBUST = "robust"
    QUANTILE = "quantile"
    WINSORIZE = "winsorize"
    RANK = "rank"
    CLIP = "clip"
    ROBUST_ZSCORE = "robust_zscore"
    ADAPTIVE = "adaptive"
    QUANTILE_ROBUST = "quantile_robust"
    WINSORIZE_ADAPTIVE = "winsorize_adaptive"

@dataclass
class ScalerConfig:
    """Configuration for scaler creation."""
    scaler_type: ScalerType
    method: str = 'single'
    window: Optional[int] = None
    quantile_range: tuple = (0.25, 0.75)
    winsorize_limits: tuple = (0.05, 0.05)
    clip_limits: tuple = (-3, 3)
    robust_quantile_range: tuple = (0.25, 0.75)
    adaptive_window: int = 20
    gpu_acceleration: bool = False
    memory_efficient: bool = True
    batch_processing: bool = False
    custom_params: Optional[Dict[str, Any]] = None

class ScalerFactory:
    """
    Factory for creating and managing scalers from features_common.
    
    This class provides a centralized way to create scalers, eliminating
    code duplication across feature generators.
    """
    
    def __init__(self, default_config: Optional[ScalerConfig] = None):
        """
        Initialize the scaler factory.
        
        Args:
            default_config: Default configuration for scalers
        """
        self.default_config = default_config or ScalerConfig(
            scaler_type=ScalerType.ZSCORE,
            method='single',
            gpu_acceleration=False,
            memory_efficient=True
        )
        
        # Scaler registry
        self._scaler_registry = {}
        self._performance_stats = {
            'scalers_created': 0,
            'vectorbt_scalers': 0,
            'pandas_fallbacks': 0,
            'batch_operations': 0
        }
    
    def create_scaler(self, scaler_type: Union[str, ScalerType], 
                     config: Optional[ScalerConfig] = None) -> BaseScaler:
        """
        Create a scaler instance.
        
        Args:
            scaler_type: Type of scaler to create
            config: Optional configuration override
            
        Returns:
            Configured scaler instance
        """
        if isinstance(scaler_type, str):
            scaler_type = ScalerType(scaler_type.lower())
        
        if config is None:
            config = self.default_config
        else:
            # Merge with default config
            config_dict = self.default_config.__dict__.copy()
            config_dict.update(config.__dict__)
            config = ScalerConfig(**config_dict)
        
        # Create cache key
        cache_key = self._get_cache_key(scaler_type, config)
        
        # Return cached scaler if available
        if cache_key in self._scaler_registry:
            return self._scaler_registry[cache_key]
        
        # Create new scaler
        scaler = self._create_scaler_instance(scaler_type, config)
        
        # Cache the scaler
        self._scaler_registry[cache_key] = scaler
        
        # Update performance stats
        self._performance_stats['scalers_created'] += 1
        if isinstance(scaler, VectorBTScaler):
            self._performance_stats['vectorbt_scalers'] += 1
        
        return scaler
    
    def create_batch_scaler(self, scaler_type: Union[str, ScalerType],
                           config: Optional[ScalerConfig] = None) -> VectorBTBatchScaler:
        """
        Create a batch scaler for processing multiple features.
        
        Args:
            scaler_type: Type of scaler to create
            config: Optional configuration override
            
        Returns:
            Configured batch scaler instance
        """
        if isinstance(scaler_type, str):
            scaler_type = ScalerType(scaler_type.lower())
        
        if config is None:
            config = self.default_config
        else:
            # Merge with default config
            config_dict = self.default_config.__dict__.copy()
            config_dict.update(config.__dict__)
            config = ScalerConfig(**config_dict)
        
        # Force batch processing
        config.batch_processing = True
        
        # Create batch scaler
        batch_scaler = VectorBTBatchScaler(
            method=config.method,
            window=config.window,
            quantile_range=config.quantile_range,
            winsorize_limits=config.winsorize_limits,
            clip_limits=config.clip_limits,
            robust_quantile_range=config.robust_quantile_range,
            adaptive_window=config.adaptive_window,
            gpu_acceleration=config.gpu_acceleration,
            memory_efficient=config.memory_efficient
        )
        
        self._performance_stats['batch_operations'] += 1
        return batch_scaler
    
    def get_scaler_for_feature_type(self, feature_type: str, 
                                   data_shape: tuple = None) -> BaseScaler:
        """
        Get an appropriate scaler based on feature type and data characteristics.
        
        Args:
            feature_type: Type of feature (e.g., 'price', 'volume', 'returns')
            data_shape: Shape of the data (rows, columns)
            
        Returns:
            Appropriate scaler instance
        """
        # Feature type to scaler mapping
        feature_scaler_mapping = {
            'price': ScalerType.MINMAX,
            'volume': ScalerType.ROBUST,
            'returns': ScalerType.ZSCORE,
            'volatility': ScalerType.ROBUST_ZSCORE,
            'momentum': ScalerType.ZSCORE,
            'trend': ScalerType.MINMAX,
            'oscillator': ScalerType.MINMAX,
            'volume_ratio': ScalerType.ROBUST,
            'price_change': ScalerType.ZSCORE,
            'technical_indicator': ScalerType.ZSCORE
        }
        
        scaler_type = feature_scaler_mapping.get(feature_type.lower(), ScalerType.ZSCORE)
        
        # Adjust configuration based on data shape
        config = self.default_config
        if data_shape and data_shape[1] > 10:  # Multiple features
            config.batch_processing = True
            return self.create_batch_scaler(scaler_type, config)
        else:
            return self.create_scaler(scaler_type, config)
    
    def _create_scaler_instance(self, scaler_type: ScalerType, 
                               config: ScalerConfig) -> BaseScaler:
        """Create a scaler instance based on type and configuration."""
        try:
            if config.batch_processing:
                return self._create_batch_scaler(scaler_type, config)
            else:
                return self._create_single_scaler(scaler_type, config)
        except Exception as e:
            logger.warning(f"Failed to create {scaler_type.value} scaler: {e}, using fallback")
            self._performance_stats['pandas_fallbacks'] += 1
            return self._create_fallback_scaler(scaler_type, config)
    
    def _create_single_scaler(self, scaler_type: ScalerType, 
                             config: ScalerConfig) -> BaseScaler:
        """Create a single-feature scaler."""
        if scaler_type == ScalerType.ZSCORE:
            return VectorBTScaler(
                method='zscore',
                window=config.window,
                gpu_acceleration=config.gpu_acceleration,
                memory_efficient=config.memory_efficient
            )
        elif scaler_type == ScalerType.MINMAX:
            return VectorBTScaler(
                method='minmax',
                window=config.window,
                gpu_acceleration=config.gpu_acceleration,
                memory_efficient=config.memory_efficient
            )
        elif scaler_type == ScalerType.ROBUST:
            return VectorBTScaler(
                method='robust',
                window=config.window,
                quantile_range=config.quantile_range,
                gpu_acceleration=config.gpu_acceleration,
                memory_efficient=config.memory_efficient
            )
        elif scaler_type == ScalerType.QUANTILE:
            return VectorBTScaler(
                method='quantile',
                window=config.window,
                quantile_range=config.quantile_range,
                gpu_acceleration=config.gpu_acceleration,
                memory_efficient=config.memory_efficient
            )
        elif scaler_type == ScalerType.WINSORIZE:
            return VectorBTScaler(
                method='winsorize',
                window=config.window,
                winsorize_limits=config.winsorize_limits,
                gpu_acceleration=config.gpu_acceleration,
                memory_efficient=config.memory_efficient
            )
        elif scaler_type == ScalerType.RANK:
            return VectorBTScaler(
                method='rank',
                window=config.window,
                gpu_acceleration=config.gpu_acceleration,
                memory_efficient=config.memory_efficient
            )
        elif scaler_type == ScalerType.CLIP:
            return VectorBTScaler(
                method='clip',
                window=config.window,
                clip_limits=config.clip_limits,
                gpu_acceleration=config.gpu_acceleration,
                memory_efficient=config.memory_efficient
            )
        elif scaler_type == ScalerType.ROBUST_ZSCORE:
            return VectorBTScaler(
                method='robust_zscore',
                window=config.window,
                robust_quantile_range=config.robust_quantile_range,
                gpu_acceleration=config.gpu_acceleration,
                memory_efficient=config.memory_efficient
            )
        elif scaler_type == ScalerType.ADAPTIVE:
            return VectorBTScaler(
                method='adaptive',
                window=config.window,
                adaptive_window=config.adaptive_window,
                gpu_acceleration=config.gpu_acceleration,
                memory_efficient=config.memory_efficient
            )
        elif scaler_type == ScalerType.QUANTILE_ROBUST:
            return VectorBTScaler(
                method='quantile_robust',
                window=config.window,
                quantile_range=config.quantile_range,
                robust_quantile_range=config.robust_quantile_range,
                gpu_acceleration=config.gpu_acceleration,
                memory_efficient=config.memory_efficient
            )
        elif scaler_type == ScalerType.WINSORIZE_ADAPTIVE:
            return VectorBTScaler(
                method='winsorize_adaptive',
                window=config.window,
                winsorize_limits=config.winsorize_limits,
                adaptive_window=config.adaptive_window,
                gpu_acceleration=config.gpu_acceleration,
                memory_efficient=config.memory_efficient
            )
        else:
            raise ValueError(f"Unsupported scaler type: {scaler_type}")
    
    def _create_batch_scaler(self, scaler_type: ScalerType, 
                            config: ScalerConfig) -> VectorBTBatchScaler:
        """Create a batch scaler for multiple features."""
        return VectorBTBatchScaler(
            method=scaler_type.value,
            window=config.window,
            quantile_range=config.quantile_range,
            winsorize_limits=config.winsorize_limits,
            clip_limits=config.clip_limits,
            robust_quantile_range=config.robust_quantile_range,
            adaptive_window=config.adaptive_window,
            gpu_acceleration=config.gpu_acceleration,
            memory_efficient=config.memory_efficient
        )
    
    def _create_fallback_scaler(self, scaler_type: ScalerType, 
                               config: ScalerConfig) -> BaseScaler:
        """Create a fallback scaler using basic methods."""
        from ..features_common.transforms.base_scaler import SimpleScaler
        
        return SimpleScaler(
            method=scaler_type.value,
            window=config.window
        )
    
    def _get_cache_key(self, scaler_type: ScalerType, config: ScalerConfig) -> str:
        """Generate cache key for scaler configuration."""
        key_parts = [
            scaler_type.value,
            config.method,
            str(config.window),
            str(config.quantile_range),
            str(config.winsorize_limits),
            str(config.clip_limits),
            str(config.robust_quantile_range),
            str(config.adaptive_window),
            str(config.gpu_acceleration),
            str(config.memory_efficient),
            str(config.batch_processing)
        ]
        return "_".join(key_parts)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self._performance_stats.copy()
    
    def reset_performance_stats(self):
        """Reset performance statistics."""
        self._performance_stats = {
            'scalers_created': 0,
            'vectorbt_scalers': 0,
            'pandas_fallbacks': 0,
            'batch_operations': 0
        }
    
    def clear_cache(self):
        """Clear the scaler cache."""
        self._scaler_registry.clear()

# Global instance
_scaler_factory = None

def get_scaler_factory() -> ScalerFactory:
    """Get the global scaler factory instance."""
    global _scaler_factory
    if _scaler_factory is None:
        _scaler_factory = ScalerFactory()
    return _scaler_factory