"""
Centralized configuration system for features_common.

This module provides unified configuration management for all optimization
settings, VectorBT configurations, and performance parameters.
"""

from .optimization_config import OptimizationConfig, get_optimization_config
from .vectorbt_config import VectorBTConfig, get_vectorbt_config
from .unified_config import UnifiedConfig, get_unified_config

__all__ = [
    'OptimizationConfig',
    'get_optimization_config',
    'VectorBTConfig', 
    'get_vectorbt_config',
    'UnifiedConfig',
    'get_unified_config'
]

# Initialize default configurations
_default_optimization_config = None
_default_vectorbt_config = None
_default_unified_config = None

def get_default_optimization_config() -> OptimizationConfig:
    """Get the default optimization configuration."""
    global _default_optimization_config
    if _default_optimization_config is None:
        _default_optimization_config = OptimizationConfig()
    return _default_optimization_config

def get_default_vectorbt_config() -> VectorBTConfig:
    """Get the default VectorBT configuration."""
    global _default_vectorbt_config
    if _default_vectorbt_config is None:
        _default_vectorbt_config = VectorBTConfig()
    return _default_vectorbt_config

def get_default_unified_config() -> UnifiedConfig:
    """Get the default unified configuration."""
    global _default_unified_config
    if _default_unified_config is None:
        _default_unified_config = UnifiedConfig()
    return _default_unified_config