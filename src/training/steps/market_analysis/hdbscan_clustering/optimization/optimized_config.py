"""
Optimized HDBSCAN Configuration

This module provides default configurations for the optimized HDBSCAN regime discovery
system, ensuring optimal performance out of the box.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class OptimizedHDBSCANDefaultConfig:
    """Default configuration for optimized HDBSCAN regime discovery."""
    
    # HDBSCAN parameters
    min_cluster_size: int = 10
    min_samples: int = 5
    cluster_selection_epsilon: float = 0.0
    cluster_selection_method: str = 'eom'
    metric: str = 'euclidean'
    alpha: float = 1.0
    
    # Optimization settings
    enable_hyperparameter_optimization: bool = True
    enable_memory_optimization: bool = True
    enable_vectorized_processing: bool = True
    enable_features_common: bool = True
    
    # Feature generation
    enable_entropy_features: bool = True
    enable_spectral_features: bool = True
    enable_regime_features: bool = True
    enable_normalization_features: bool = True
    
    # Feature selection
    enable_feature_selection: bool = True
    feature_selection_method: str = 'mrmr'  # 'mrmr', 'lasso', 'mutual_info'
    max_features: int = 50
    feature_selection_threshold: float = 0.01
    
    # Performance settings
    enable_parallel_processing: bool = True
    max_memory_gb: float = 8.0
    chunk_size: int = 1000
    n_jobs: int = -1
    
    # Evaluation metrics
    primary_metric: str = 'silhouette'
    enable_cross_validation: bool = True
    cv_folds: int = 5
    
    # Advanced settings
    optimization_level: str = 'high'  # 'high', 'medium', 'low'
    auto_tuning: bool = True
    adaptive_parameters: bool = True

def get_optimized_hdbscan_config(
    symbol: str = "ETHUSDT",
    exchange: str = "binance", 
    timeframe: str = "15m",
    execution_mode: str = "light",
    **kwargs
) -> Dict[str, Any]:
    """
    Get optimized HDBSCAN configuration for a specific trading pair.
    
    Args:
        symbol: Trading symbol (e.g., 'ETHUSDT')
        exchange: Exchange name (e.g., 'binance')
        timeframe: Timeframe (e.g., '15m')
        execution_mode: Execution mode ('full', 'light', 'blank')
        **kwargs: Additional configuration parameters
        
    Returns:
        Dictionary with optimized configuration
    """
    # Base configuration
    config = {
        'symbol': symbol,
        'exchange': exchange,
        'timeframe': timeframe,
        'execution_mode': execution_mode,
        'live_mode': False,
        
        # HDBSCAN parameters
        'min_cluster_size': kwargs.get('min_cluster_size', 10),
        'min_samples': kwargs.get('min_samples', 5),
        'cluster_selection_epsilon': kwargs.get('cluster_selection_epsilon', 0.0),
        'cluster_selection_method': kwargs.get('cluster_selection_method', 'eom'),
        'metric': kwargs.get('metric', 'euclidean'),
        'alpha': kwargs.get('alpha', 1.0),
        
        # Optimization settings
        'enable_hyperparameter_optimization': kwargs.get('enable_hyperparameter_optimization', True),
        'enable_memory_optimization': kwargs.get('enable_memory_optimization', True),
        'enable_vectorized_processing': kwargs.get('enable_vectorized_processing', True),
        'enable_features_common': kwargs.get('enable_features_common', True),
        
        # Feature generation
        'enable_entropy_features': kwargs.get('enable_entropy_features', True),
        'enable_spectral_features': kwargs.get('enable_spectral_features', True),
        'enable_regime_features': kwargs.get('enable_regime_features', True),
        'enable_normalization_features': kwargs.get('enable_normalization_features', True),
        
        # Feature selection
        'enable_feature_selection': kwargs.get('enable_feature_selection', True),
        'feature_selection_method': kwargs.get('feature_selection_method', 'mrmr'),
        'max_features': kwargs.get('max_features', 50),
        'feature_selection_threshold': kwargs.get('feature_selection_threshold', 0.01),
        
        # Performance settings
        'enable_parallel_processing': kwargs.get('enable_parallel_processing', True),
        'max_memory_gb': kwargs.get('max_memory_gb', 8.0),
        'chunk_size': kwargs.get('chunk_size', 1000),
        'n_jobs': kwargs.get('n_jobs', -1),
        
        # Evaluation metrics
        'primary_metric': kwargs.get('primary_metric', 'silhouette'),
        'enable_cross_validation': kwargs.get('enable_cross_validation', True),
        'cv_folds': kwargs.get('cv_folds', 5),
        
        # Advanced settings
        'optimization_level': kwargs.get('optimization_level', 'high'),
        'auto_tuning': kwargs.get('auto_tuning', True),
        'adaptive_parameters': kwargs.get('adaptive_parameters', True),
        
        # Legacy compatibility
        'disable_optimized_version': kwargs.get('disable_optimized_version', False)
    }
    
    return config

def get_high_performance_config(symbol: str = "ETHUSDT") -> Dict[str, Any]:
    """Get high-performance configuration for maximum optimization."""
    return get_optimized_hdbscan_config(
        symbol=symbol,
        min_cluster_size=15,
        min_samples=7,
        cluster_selection_epsilon=0.1,
        enable_hyperparameter_optimization=True,
        enable_memory_optimization=True,
        enable_vectorized_processing=True,
        enable_features_common=True,
        enable_feature_selection=True,
        max_features=30,
        max_memory_gb=16.0,
        chunk_size=2000,
        optimization_level='high'
    )

def get_balanced_config(symbol: str = "ETHUSDT") -> Dict[str, Any]:
    """Get balanced configuration for good performance and reasonable resource usage."""
    return get_optimized_hdbscan_config(
        symbol=symbol,
        min_cluster_size=10,
        min_samples=5,
        cluster_selection_epsilon=0.0,
        enable_hyperparameter_optimization=True,
        enable_memory_optimization=True,
        enable_vectorized_processing=True,
        enable_features_common=True,
        enable_feature_selection=True,
        max_features=50,
        max_memory_gb=8.0,
        chunk_size=1000,
        optimization_level='medium'
    )

def get_fast_config(symbol: str = "ETHUSDT") -> Dict[str, Any]:
    """Get fast configuration for quick processing with minimal optimization."""
    return get_optimized_hdbscan_config(
        symbol=symbol,
        min_cluster_size=8,
        min_samples=3,
        cluster_selection_epsilon=0.0,
        enable_hyperparameter_optimization=False,
        enable_memory_optimization=True,
        enable_vectorized_processing=True,
        enable_features_common=False,
        enable_feature_selection=False,
        max_features=20,
        max_memory_gb=4.0,
        chunk_size=500,
        optimization_level='low'
    )

def get_legacy_config(symbol: str = "ETHUSDT") -> Dict[str, Any]:
    """Get legacy configuration that disables optimized version."""
    return get_optimized_hdbscan_config(
        symbol=symbol,
        disable_optimized_version=True
    )

# Example usage configurations
EXAMPLE_CONFIGS = {
    'high_performance': get_high_performance_config,
    'balanced': get_balanced_config,
    'fast': get_fast_config,
    'legacy': get_legacy_config
}

def get_config_by_name(config_name: str, symbol: str = "ETHUSDT") -> Dict[str, Any]:
    """
    Get configuration by name.
    
    Args:
        config_name: Configuration name ('high_performance', 'balanced', 'fast', 'legacy')
        symbol: Trading symbol
        
    Returns:
        Dictionary with configuration
    """
    if config_name not in EXAMPLE_CONFIGS:
        raise ValueError(f"Unknown configuration: {config_name}. Available: {list(EXAMPLE_CONFIGS.keys())}")
    
    return EXAMPLE_CONFIGS[config_name](symbol)

# Example usage
if __name__ == "__main__":
    print("=== Optimized HDBSCAN Configuration Examples ===")
    
    # High performance configuration
    high_perf_config = get_high_performance_config("ETHUSDT")
    print(f"\n🚀 High Performance Config:")
    print(f"  Min cluster size: {high_perf_config['min_cluster_size']}")
    print(f"  Max features: {high_perf_config['max_features']}")
    print(f"  Memory limit: {high_perf_config['max_memory_gb']}GB")
    print(f"  Optimization level: {high_perf_config['optimization_level']}")
    
    # Balanced configuration
    balanced_config = get_balanced_config("ETHUSDT")
    print(f"\n⚖️ Balanced Config:")
    print(f"  Min cluster size: {balanced_config['min_cluster_size']}")
    print(f"  Max features: {balanced_config['max_features']}")
    print(f"  Memory limit: {balanced_config['max_memory_gb']}GB")
    print(f"  Optimization level: {balanced_config['optimization_level']}")
    
    # Fast configuration
    fast_config = get_fast_config("ETHUSDT")
    print(f"\n⚡ Fast Config:")
    print(f"  Min cluster size: {fast_config['min_cluster_size']}")
    print(f"  Max features: {fast_config['max_features']}")
    print(f"  Memory limit: {fast_config['max_memory_gb']}GB")
    print(f"  Optimization level: {fast_config['optimization_level']}")
    
    # Legacy configuration
    legacy_config = get_legacy_config("ETHUSDT")
    print(f"\n🔄 Legacy Config:")
    print(f"  Optimized version disabled: {legacy_config['disable_optimized_version']}")
