#!/usr/bin/env python3
"""
Balanced HMM Configuration Utilities

This module provides convenient configuration presets and utilities
for creating balanced HMM clustering configurations that ensure
no single cluster exceeds the specified size limits.
"""

from dataclasses import dataclass
from typing import List, Optional
from .enhanced_hmm_clustering import HMMClusteringConfig

def create_balanced_config(
    max_cluster_size_pct: float = 15.0,
    min_cluster_size_pct: float = 5.0,
    n_components: int = 4,
    balancing_method: str = "hybrid"
) -> HMMClusteringConfig:
    """
    Create a balanced HMM clustering configuration.
    
    Args:
        max_cluster_size_pct: Maximum allowed cluster size as percentage
        min_cluster_size_pct: Minimum allowed cluster size as percentage  
        n_components: Number of HMM components/clusters
        balancing_method: Balancing method ("hybrid", "adaptive_splitting", "post_processing")
        
    Returns:
        HMMClusteringConfig with balanced settings
    """
    return HMMClusteringConfig(
        # HMM Parameters
        n_components=n_components,
        covariance_type="full",
        n_iter=150,
        random_state=42,
        
        # Feature Engineering
        lookback_windows=[5, 10, 20, 50],
        technical_indicators=["rsi", "macd", "bollinger_bands", "atr"],
        
        # Optimization
        use_gpu=True,
        use_memory_optimization=True,
        use_cpu_optimization=True,
        
        # Feature Selection
        feature_selection_method="mrmr",
        max_features=30,
        
        # Cluster Balancing (KEY SETTINGS)
        enable_cluster_balancing=True,
        max_cluster_size_pct=max_cluster_size_pct,
        min_cluster_size_pct=min_cluster_size_pct,
        cluster_balancing_method=balancing_method
    )

def create_conservative_balanced_config() -> HMMClusteringConfig:
    """
    Create a conservative balanced configuration with stricter size limits.
    Ensures maximum cluster size of 12% for very balanced distributions.
    """
    return create_balanced_config(
        max_cluster_size_pct=12.0,
        min_cluster_size_pct=8.0,
        n_components=5,  # More clusters for better balance
        balancing_method="hybrid"
    )

def create_aggressive_balanced_config() -> HMMClusteringConfig:
    """
    Create an aggressive balanced configuration that allows slightly larger clusters
    but focuses on splitting dominant clusters quickly.
    """
    return create_balanced_config(
        max_cluster_size_pct=18.0,
        min_cluster_size_pct=3.0,
        n_components=4,
        balancing_method="adaptive_splitting"
    )

def create_forex_balanced_config() -> HMMClusteringConfig:
    """
    Create a balanced configuration optimized for forex markets.
    """
    config = create_balanced_config(
        max_cluster_size_pct=15.0,
        min_cluster_size_pct=5.0,
        n_components=4
    )
    
    # Forex-specific adjustments
    config.lookback_windows = [5, 15, 30, 60]  # Forex-friendly windows
    config.technical_indicators = ["rsi", "macd", "atr", "stochastic"]
    config.n_iter = 200  # More iterations for forex complexity
    
    return config

def create_crypto_balanced_config() -> HMMClusteringConfig:
    """
    Create a balanced configuration optimized for cryptocurrency markets.
    """
    config = create_balanced_config(
        max_cluster_size_pct=15.0,
        min_cluster_size_pct=4.0,  # Allow smaller clusters due to crypto volatility
        n_components=5  # More regimes for crypto complexity
    )
    
    # Crypto-specific adjustments
    config.lookback_windows = [3, 7, 14, 28]  # Shorter windows for crypto speed
    config.technical_indicators = ["rsi", "macd", "bollinger_bands", "atr"]
    config.n_iter = 250  # More iterations for crypto volatility
    
    return config

def create_stock_balanced_config() -> HMMClusteringConfig:
    """
    Create a balanced configuration optimized for stock markets.
    """
    config = create_balanced_config(
        max_cluster_size_pct=15.0,
        min_cluster_size_pct=6.0,
        n_components=4
    )
    
    # Stock-specific adjustments
    config.lookback_windows = [5, 10, 20, 50, 100]  # Include longer windows
    config.technical_indicators = ["rsi", "macd", "bollinger_bands", "atr", "stochastic"]
    config.n_iter = 150
    
    return config

# Preset configurations for quick access
BALANCED_PRESETS = {
    "default": create_balanced_config,
    "conservative": create_conservative_balanced_config,
    "aggressive": create_aggressive_balanced_config,
    "forex": create_forex_balanced_config,
    "crypto": create_crypto_balanced_config,
    "stock": create_stock_balanced_config
}

def get_balanced_preset(preset_name: str) -> HMMClusteringConfig:
    """
    Get a balanced configuration preset by name.
    
    Args:
        preset_name: Name of the preset ("default", "conservative", "aggressive", 
                    "forex", "crypto", "stock")
                    
    Returns:
        HMMClusteringConfig with balanced settings
        
    Raises:
        ValueError: If preset_name is not recognized
    """
    if preset_name not in BALANCED_PRESETS:
        available = list(BALANCED_PRESETS.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
    
    return BALANCED_PRESETS[preset_name]()

def validate_balance_config(config: HMMClusteringConfig) -> List[str]:
    """
    Validate a configuration for proper balance settings.
    
    Args:
        config: HMMClusteringConfig to validate
        
    Returns:
        List of validation warnings/errors
    """
    warnings = []
    
    if not config.enable_cluster_balancing:
        warnings.append("Cluster balancing is disabled - may result in imbalanced clusters")
    
    if config.max_cluster_size_pct > 25.0:
        warnings.append(f"Max cluster size {config.max_cluster_size_pct}% is quite high - consider lowering")
    
    if config.min_cluster_size_pct < 2.0:
        warnings.append(f"Min cluster size {config.min_cluster_size_pct}% is very low - may create tiny clusters")
    
    if config.max_cluster_size_pct <= config.min_cluster_size_pct:
        warnings.append("Max cluster size must be greater than min cluster size")
    
    if config.n_components < 3:
        warnings.append("Very few components may limit balancing effectiveness")
    
    if config.n_components > 8:
        warnings.append("Many components may create overly fragmented clusters")
    
    # Check if balancing method is valid
    valid_methods = ["hybrid", "adaptive_splitting", "cluster_merging", "post_processing"]
    if config.cluster_balancing_method not in valid_methods:
        warnings.append(f"Unknown balancing method '{config.cluster_balancing_method}'. Valid: {valid_methods}")
    
    return warnings

def print_balance_config_summary(config: HMMClusteringConfig):
    """Print a summary of the balance configuration."""
    print("Balanced HMM Configuration Summary")
    print("=" * 40)
    print(f"Cluster Balancing: {'✅ Enabled' if config.enable_cluster_balancing else '❌ Disabled'}")
    
    if config.enable_cluster_balancing:
        print(f"Max Cluster Size: {config.max_cluster_size_pct}%")
        print(f"Min Cluster Size: {config.min_cluster_size_pct}%")
        print(f"Balancing Method: {config.cluster_balancing_method}")
    
    print(f"Number of Components: {config.n_components}")
    print(f"Technical Indicators: {', '.join(config.technical_indicators)}")
    print(f"Lookback Windows: {config.lookback_windows}")
    
    # Validate and show warnings
    warnings = validate_balance_config(config)
    if warnings:
        print("\nValidation Warnings:")
        for warning in warnings:
            print(f"  ⚠️  {warning}")
    else:
        print("\n✅ Configuration looks good!")