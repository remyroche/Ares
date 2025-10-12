"""
VectorBT Optimization Configuration for Final Feature Selection

This module provides comprehensive configuration for VectorBT optimizations
in the final feature selection pipeline.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from enum import Enum


class VectorBTOptimizationLevel(Enum):
    """VectorBT optimization levels."""
    DISABLED = "disabled"
    BASIC = "basic"
    ENHANCED = "enhanced"
    AGGRESSIVE = "aggressive"


class VectorBTMemoryStrategy(Enum):
    """VectorBT memory optimization strategies."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


@dataclass
class VectorBTRollingConfig:
    """Configuration for VectorBT rolling operations."""
    enable_gpu: bool = False
    enable_parallel: bool = True
    memory_efficient: bool = True
    chunk_size: int = 1000
    default_window: int = 20
    correlation_window: int = 50
    stability_window: int = 100
    outlier_window: int = 50
    normalization_window: int = 100


@dataclass
class VectorBTMemoryConfig:
    """Configuration for VectorBT memory optimization."""
    strategy: VectorBTMemoryStrategy = VectorBTMemoryStrategy.BALANCED
    chunk_size: int = 5000
    memory_limit_gb: float = 8.0
    enable_chunked_processing: bool = True
    enable_data_type_optimization: bool = True
    enable_rolling_optimization: bool = True


@dataclass
class VectorBTMatrixConfig:
    """Configuration for VectorBT matrix operations."""
    enable_vectorbt_matrix_ops: bool = True
    enable_gpu_matrix_ops: bool = False
    enable_correlation_optimization: bool = True
    enable_importance_calculation: bool = True
    enable_stability_analysis: bool = True


@dataclass
class VectorBTPerformanceConfig:
    """Configuration for VectorBT performance monitoring."""
    enable_monitoring: bool = True
    enable_detailed_stats: bool = True
    enable_strategy_tracking: bool = True
    enable_speedup_calculation: bool = True
    log_performance_metrics: bool = True


@dataclass
class VectorBTOptimizationConfig:
    """Comprehensive VectorBT optimization configuration."""
    
    # Core settings
    optimization_level: VectorBTOptimizationLevel = VectorBTOptimizationLevel.ENHANCED
    enable_vectorbt: bool = True
    
    # Component configurations
    rolling_config: VectorBTRollingConfig = VectorBTRollingConfig()
    memory_config: VectorBTMemoryConfig = VectorBTMemoryConfig()
    matrix_config: VectorBTMatrixConfig = VectorBTMatrixConfig()
    performance_config: VectorBTPerformanceConfig = VectorBTPerformanceConfig()
    
    # Feature selection specific settings
    enable_feature_importance: bool = True
    enable_stability_analysis: bool = True
    enable_correlation_analysis: bool = True
    enable_outlier_handling: bool = True
    enable_data_normalization: bool = True
    
    # Performance thresholds
    large_dataset_threshold: int = 5000
    memory_pressure_threshold: float = 0.8
    gpu_threshold: int = 10000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'optimization_level': self.optimization_level.value,
            'enable_vectorbt': self.enable_vectorbt,
            'rolling_config': {
                'enable_gpu': self.rolling_config.enable_gpu,
                'enable_parallel': self.rolling_config.enable_parallel,
                'memory_efficient': self.rolling_config.memory_efficient,
                'chunk_size': self.rolling_config.chunk_size,
                'default_window': self.rolling_config.default_window,
                'correlation_window': self.rolling_config.correlation_window,
                'stability_window': self.rolling_config.stability_window,
                'outlier_window': self.rolling_config.outlier_window,
                'normalization_window': self.rolling_config.normalization_window,
            },
            'memory_config': {
                'strategy': self.memory_config.strategy.value,
                'chunk_size': self.memory_config.chunk_size,
                'memory_limit_gb': self.memory_config.memory_limit_gb,
                'enable_chunked_processing': self.memory_config.enable_chunked_processing,
                'enable_data_type_optimization': self.memory_config.enable_data_type_optimization,
                'enable_rolling_optimization': self.memory_config.enable_rolling_optimization,
            },
            'matrix_config': {
                'enable_vectorbt_matrix_ops': self.matrix_config.enable_vectorbt_matrix_ops,
                'enable_gpu_matrix_ops': self.matrix_config.enable_gpu_matrix_ops,
                'enable_correlation_optimization': self.matrix_config.enable_correlation_optimization,
                'enable_importance_calculation': self.matrix_config.enable_importance_calculation,
                'enable_stability_analysis': self.matrix_config.enable_stability_analysis,
            },
            'performance_config': {
                'enable_monitoring': self.performance_config.enable_monitoring,
                'enable_detailed_stats': self.performance_config.enable_detailed_stats,
                'enable_strategy_tracking': self.performance_config.enable_strategy_tracking,
                'enable_speedup_calculation': self.performance_config.enable_speedup_calculation,
                'log_performance_metrics': self.performance_config.log_performance_metrics,
            },
            'feature_selection_settings': {
                'enable_feature_importance': self.enable_feature_importance,
                'enable_stability_analysis': self.enable_stability_analysis,
                'enable_correlation_analysis': self.enable_correlation_analysis,
                'enable_outlier_handling': self.enable_outlier_handling,
                'enable_data_normalization': self.enable_data_normalization,
            },
            'performance_thresholds': {
                'large_dataset_threshold': self.large_dataset_threshold,
                'memory_pressure_threshold': self.memory_pressure_threshold,
                'gpu_threshold': self.gpu_threshold,
            }
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'VectorBTOptimizationConfig':
        """Create configuration from dictionary."""
        config = cls()
        
        # Update optimization level
        if 'optimization_level' in config_dict:
            config.optimization_level = VectorBTOptimizationLevel(config_dict['optimization_level'])
        
        # Update enable_vectorbt
        if 'enable_vectorbt' in config_dict:
            config.enable_vectorbt = config_dict['enable_vectorbt']
        
        # Update rolling config
        if 'rolling_config' in config_dict:
            rolling_dict = config_dict['rolling_config']
            config.rolling_config = VectorBTRollingConfig(**rolling_dict)
        
        # Update memory config
        if 'memory_config' in config_dict:
            memory_dict = config_dict['memory_config']
            if 'strategy' in memory_dict:
                memory_dict['strategy'] = VectorBTMemoryStrategy(memory_dict['strategy'])
            config.memory_config = VectorBTMemoryConfig(**memory_dict)
        
        # Update matrix config
        if 'matrix_config' in config_dict:
            matrix_dict = config_dict['matrix_config']
            config.matrix_config = VectorBTMatrixConfig(**matrix_dict)
        
        # Update performance config
        if 'performance_config' in config_dict:
            perf_dict = config_dict['performance_config']
            config.performance_config = VectorBTPerformanceConfig(**perf_dict)
        
        # Update feature selection settings
        if 'feature_selection_settings' in config_dict:
            fs_dict = config_dict['feature_selection_settings']
            for key, value in fs_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        # Update performance thresholds
        if 'performance_thresholds' in config_dict:
            thresh_dict = config_dict['performance_thresholds']
            for key, value in thresh_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        return config


# Predefined configurations for different use cases
def get_conservative_config() -> VectorBTOptimizationConfig:
    """Get conservative VectorBT configuration."""
    config = VectorBTOptimizationConfig()
    config.optimization_level = VectorBTOptimizationLevel.BASIC
    config.rolling_config.enable_gpu = False
    config.rolling_config.chunk_size = 2000
    config.memory_config.strategy = VectorBTMemoryStrategy.CONSERVATIVE
    config.memory_config.chunk_size = 10000
    return config


def get_enhanced_config() -> VectorBTOptimizationConfig:
    """Get enhanced VectorBT configuration (default)."""
    return VectorBTOptimizationConfig()


def get_aggressive_config() -> VectorBTOptimizationConfig:
    """Get aggressive VectorBT configuration."""
    config = VectorBTOptimizationConfig()
    config.optimization_level = VectorBTOptimizationLevel.AGGRESSIVE
    config.rolling_config.enable_gpu = True
    config.rolling_config.chunk_size = 500
    config.memory_config.strategy = VectorBTMemoryStrategy.AGGRESSIVE
    config.memory_config.chunk_size = 2000
    config.matrix_config.enable_gpu_matrix_ops = True
    config.large_dataset_threshold = 2000
    config.gpu_threshold = 5000
    return config


def get_memory_optimized_config() -> VectorBTOptimizationConfig:
    """Get memory-optimized VectorBT configuration."""
    config = VectorBTOptimizationConfig()
    config.memory_config.strategy = VectorBTMemoryStrategy.AGGRESSIVE
    config.memory_config.chunk_size = 2000
    config.memory_config.enable_chunked_processing = True
    config.memory_config.enable_data_type_optimization = True
    config.large_dataset_threshold = 2000
    return config


def get_gpu_optimized_config() -> VectorBTOptimizationConfig:
    """Get GPU-optimized VectorBT configuration."""
    config = VectorBTOptimizationConfig()
    config.rolling_config.enable_gpu = True
    config.matrix_config.enable_gpu_matrix_ops = True
    config.gpu_threshold = 1000
    config.large_dataset_threshold = 1000
    return config


# Configuration factory
def create_vectorbt_config(
    optimization_level: str = "enhanced",
    memory_strategy: str = "balanced",
    enable_gpu: bool = False,
    custom_settings: Optional[Dict[str, Any]] = None
) -> VectorBTOptimizationConfig:
    """
    Create VectorBT configuration with specified parameters.
    
    Args:
        optimization_level: "conservative", "enhanced", or "aggressive"
        memory_strategy: "conservative", "balanced", or "aggressive"
        enable_gpu: Whether to enable GPU acceleration
        custom_settings: Custom settings to override defaults
    
    Returns:
        VectorBTOptimizationConfig instance
    """
    # Select base configuration
    if optimization_level == "conservative":
        config = get_conservative_config()
    elif optimization_level == "aggressive":
        config = get_aggressive_config()
    else:  # enhanced
        config = get_enhanced_config()
    
    # Apply memory strategy
    if memory_strategy == "conservative":
        config.memory_config.strategy = VectorBTMemoryStrategy.CONSERVATIVE
    elif memory_strategy == "aggressive":
        config.memory_config.strategy = VectorBTMemoryStrategy.AGGRESSIVE
    else:  # balanced
        config.memory_config.strategy = VectorBTMemoryStrategy.BALANCED
    
    # Apply GPU settings
    config.rolling_config.enable_gpu = enable_gpu
    config.matrix_config.enable_gpu_matrix_ops = enable_gpu
    
    # Apply custom settings
    if custom_settings:
        for key, value in custom_settings.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return config