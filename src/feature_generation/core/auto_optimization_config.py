from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum

class OptimizationLevel(Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"

@dataclass
class AutoOptimizationConfig:
    """Configuration for automatic optimization."""
    # Core settings
    enable_auto_optimization: bool = True
    optimization_level: OptimizationLevel = OptimizationLevel.BALANCED
    
    # Memory optimization
    enable_memory_optimization: bool = True
    memory_threshold_mb: float = 100.0
    enable_data_compression: bool = True
    enable_chunked_processing: bool = True
    chunk_size: int = 10000
    
    # VectorBT optimization
    enable_vectorbt_optimization: bool = True
    vectorbt_threshold: int = 1000
    enable_gpu_acceleration: bool = False
    
    # Rolling operations optimization
    enable_rolling_optimization: bool = True
    enable_rolling_cache: bool = True
    rolling_cache_size: int = 100
    
    # Performance monitoring
    enable_performance_monitoring: bool = True
    enable_optimization_logging: bool = False
    
    # Strategy-specific settings
    conservative_settings: Dict[str, Any] = None
    balanced_settings: Dict[str, Any] = None
    aggressive_settings: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.conservative_settings is None:
            self.conservative_settings = {
                'enable_memory_optimization': True,
                'enable_vectorbt_optimization': False,
                'enable_rolling_optimization': False,
                'memory_threshold_mb': 500.0
            }
        
        if self.balanced_settings is None:
            self.balanced_settings = {
                'enable_memory_optimization': True,
                'enable_vectorbt_optimization': True,
                'enable_rolling_optimization': True,
                'memory_threshold_mb': 100.0,
                'vectorbt_threshold': 1000
            }
        
        if self.aggressive_settings is None:
            self.aggressive_settings = {
                'enable_memory_optimization': True,
                'enable_vectorbt_optimization': True,
                'enable_rolling_optimization': True,
                'enable_data_compression': True,
                'enable_chunked_processing': True,
                'memory_threshold_mb': 50.0,
                'vectorbt_threshold': 500
            }
    
    def get_settings_for_level(self) -> Dict[str, Any]:
        """Get settings for current optimization level."""
        if self.optimization_level == OptimizationLevel.CONSERVATIVE:
            return self.conservative_settings
        elif self.optimization_level == OptimizationLevel.BALANCED:
            return self.balanced_settings
        elif self.optimization_level == OptimizationLevel.AGGRESSIVE:
            return self.aggressive_settings
        else:
            return self.balanced_settings
    
    def apply_level_settings(self):
        """Apply settings based on current optimization level."""
        level_settings = self.get_settings_for_level()
        
        for key, value in level_settings.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'enable_auto_optimization': self.enable_auto_optimization,
            'optimization_level': self.optimization_level.value,
            'enable_memory_optimization': self.enable_memory_optimization,
            'memory_threshold_mb': self.memory_threshold_mb,
            'enable_data_compression': self.enable_data_compression,
            'enable_chunked_processing': self.enable_chunked_processing,
            'chunk_size': self.chunk_size,
            'enable_vectorbt_optimization': self.enable_vectorbt_optimization,
            'vectorbt_threshold': self.vectorbt_threshold,
            'enable_gpu_acceleration': self.enable_gpu_acceleration,
            'enable_rolling_optimization': self.enable_rolling_optimization,
            'enable_rolling_cache': self.enable_rolling_cache,
            'rolling_cache_size': self.rolling_cache_size,
            'enable_performance_monitoring': self.enable_performance_monitoring,
            'enable_optimization_logging': self.enable_optimization_logging
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AutoOptimizationConfig':
        """Create configuration from dictionary."""
        # Extract optimization level
        if 'optimization_level' in config_dict:
            if isinstance(config_dict['optimization_level'], str):
                config_dict['optimization_level'] = OptimizationLevel(config_dict['optimization_level'])
        
        return cls(**config_dict)