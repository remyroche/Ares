#!/usr/bin/env python3
"""
Configuration module for Enhanced HMM Clustering

This module provides various pre-configured settings for different
market analysis scenarios and use cases.

DEPRECATED: This module is deprecated. Use the unified configuration
from src.training.steps.market_analysis.hmm_clustering_config instead.
"""

import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

# Import unified configuration
try:
    from src.training.steps.market_analysis.hmm_clustering_config import (
        UnifiedHMMClusteringConfig,
        MarketType,
        TimeframeType,
        get_config_by_name,
        create_custom_config,
        validate_config
    )
    
    # Provide backward compatibility aliases
    HMMClusteringConfig = UnifiedHMMClusteringConfig
    HMMClusteringConfigFactory = UnifiedHMMClusteringConfig
    ConfigValidator = type('ConfigValidator', (), {'validate_config': staticmethod(validate_config)})
    ConfigPresets = type('ConfigPresets', (), {
        'CRYPTO_BTC_1H': get_config_by_name('crypto_btc_1h'),
        'CRYPTO_DAILY': get_config_by_name('crypto_daily'),
        'FOREX_MAJOR_1H': get_config_by_name('forex_major_1h'),
        'HIGH_FREQUENCY': get_config_by_name('high_frequency'),
        'RESEARCH': get_config_by_name('research')
    })
    
    warnings.warn(
        "This configuration module is deprecated. Use "
        "src.training.steps.market_analysis.hmm_clustering_config instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    UNIFIED_CONFIG_AVAILABLE = True
    
except ImportError:
    UNIFIED_CONFIG_AVAILABLE = False
    warnings.warn(
        "Unified configuration not available, using legacy configuration",
        UserWarning,
        stacklevel=2
    )

class MarketType(Enum):
    """Enumeration of market types for configuration."""
    CRYPTO = "crypto"
    FOREX = "forex"
    STOCKS = "stocks"
    COMMODITIES = "commodities"

class TimeframeType(Enum):
    """Enumeration of timeframe types."""
    INTRADAY = "intraday"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"

@dataclass
class HMMClusteringConfig:
    """Base configuration for HMM clustering."""
    # HMM Parameters
    n_components: int = 3
    covariance_type: str = "full"
    n_iter: int = 100
    random_state: int = 42
    
    # Feature Engineering
    lookback_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    technical_indicators: List[str] = field(default_factory=lambda: [
        "rsi", "macd", "bollinger_bands", "atr", "stochastic"
    ])
    
    # Optimization
    use_gpu: bool = True
    use_memory_optimization: bool = True
    use_cpu_optimization: bool = True
    
    # Cross-validation
    cv_folds: int = 5
    test_size: float = 0.2
    purged_cv: bool = True
    
    # Feature Selection
    feature_selection_method: str = "mrmr"
    max_features: int = 50
    
    # Data Processing
    min_data_points: int = 1000
    max_missing_ratio: float = 0.1
    
    # Regime Analysis
    min_regime_duration: int = 10
    regime_stability_threshold: float = 0.7

class HMMClusteringConfigFactory:
    """Factory for creating HMM clustering configurations."""
    
    @staticmethod
    def create_crypto_config(
        timeframe: TimeframeType = TimeframeType.INTRADAY,
        market_volatility: str = "high"
    ) -> HMMClusteringConfig:
        """Create configuration optimized for cryptocurrency markets."""
        
        if timeframe == TimeframeType.INTRADAY:
            lookback_windows = [5, 10, 20, 50, 100]
            n_components = 5
            min_data_points = 2000
        elif timeframe == TimeframeType.DAILY:
            lookback_windows = [10, 20, 50, 100, 200]
            n_components = 4
            min_data_points = 1000
        else:
            lookback_windows = [5, 10, 20, 50]
            n_components = 3
            min_data_points = 500
        
        # Adjust for market volatility
        if market_volatility == "high":
            technical_indicators = [
                "rsi", "macd", "bollinger_bands", "atr", "stochastic", "williams_r"
            ]
            regime_stability_threshold = 0.6
        else:
            technical_indicators = [
                "rsi", "macd", "bollinger_bands", "atr"
            ]
            regime_stability_threshold = 0.8
        
        return HMMClusteringConfig(
            n_components=n_components,
            lookback_windows=lookback_windows,
            technical_indicators=technical_indicators,
            min_data_points=min_data_points,
            regime_stability_threshold=regime_stability_threshold,
            use_gpu=True,
            use_memory_optimization=True,
            max_features=40
        )
    
    @staticmethod
    def create_forex_config(
        timeframe: TimeframeType = TimeframeType.INTRADAY,
        currency_pair_type: str = "major"
    ) -> HMMClusteringConfig:
        """Create configuration optimized for forex markets."""
        
        if timeframe == TimeframeType.INTRADAY:
            lookback_windows = [5, 10, 20, 50, 100]
            n_components = 4
        else:
            lookback_windows = [10, 20, 50, 100]
            n_components = 3
        
        # Adjust for currency pair type
        if currency_pair_type == "major":
            technical_indicators = [
                "rsi", "macd", "bollinger_bands", "atr", "stochastic"
            ]
            max_features = 30
        else:
            technical_indicators = [
                "rsi", "macd", "bollinger_bands", "atr"
            ]
            max_features = 25
        
        return HMMClusteringConfig(
            n_components=n_components,
            lookback_windows=lookback_windows,
            technical_indicators=technical_indicators,
            max_features=max_features,
            use_gpu=True,
            use_memory_optimization=True,
            regime_stability_threshold=0.75
        )
    
    @staticmethod
    def create_stocks_config(
        timeframe: TimeframeType = TimeframeType.DAILY,
        market_cap: str = "large"
    ) -> HMMClusteringConfig:
        """Create configuration optimized for stock markets."""
        
        if timeframe == TimeframeType.INTRADAY:
            lookback_windows = [5, 10, 20, 50]
            n_components = 4
            min_data_points = 1500
        else:
            lookback_windows = [10, 20, 50, 100]
            n_components = 3
            min_data_points = 1000
        
        # Adjust for market cap
        if market_cap == "large":
            technical_indicators = [
                "rsi", "macd", "bollinger_bands", "atr", "stochastic", "williams_r"
            ]
            regime_stability_threshold = 0.8
        else:
            technical_indicators = [
                "rsi", "macd", "bollinger_bands", "atr"
            ]
            regime_stability_threshold = 0.7
        
        return HMMClusteringConfig(
            n_components=n_components,
            lookback_windows=lookback_windows,
            technical_indicators=technical_indicators,
            min_data_points=min_data_points,
            regime_stability_threshold=regime_stability_threshold,
            use_gpu=True,
            use_memory_optimization=True,
            max_features=35
        )
    
    @staticmethod
    def create_high_frequency_config() -> HMMClusteringConfig:
        """Create configuration for high-frequency trading analysis."""
        return HMMClusteringConfig(
            n_components=6,
            lookback_windows=[3, 5, 10, 20, 50],
            technical_indicators=[
                "rsi", "macd", "bollinger_bands", "atr", "stochastic", "williams_r"
            ],
            min_data_points=5000,
            max_missing_ratio=0.05,
            regime_stability_threshold=0.5,
            min_regime_duration=5,
            use_gpu=True,
            use_memory_optimization=True,
            use_cpu_optimization=True,
            max_features=50,
            n_iter=150
        )
    
    @staticmethod
    def create_low_latency_config() -> HMMClusteringConfig:
        """Create configuration optimized for low latency processing."""
        return HMMClusteringConfig(
            n_components=3,
            lookback_windows=[5, 10, 20],
            technical_indicators=["rsi", "macd", "bollinger_bands"],
            min_data_points=500,
            max_missing_ratio=0.2,
            regime_stability_threshold=0.6,
            min_regime_duration=3,
            use_gpu=True,
            use_memory_optimization=True,
            use_cpu_optimization=True,
            max_features=20,
            n_iter=50,
            cv_folds=3
        )
    
    @staticmethod
    def create_research_config() -> HMMClusteringConfig:
        """Create configuration for research and experimentation."""
        return HMMClusteringConfig(
            n_components=5,
            lookback_windows=[5, 10, 20, 50, 100, 200],
            technical_indicators=[
                "rsi", "macd", "bollinger_bands", "atr", "stochastic", 
                "williams_r", "cci", "roc"
            ],
            min_data_points=3000,
            max_missing_ratio=0.05,
            regime_stability_threshold=0.7,
            min_regime_duration=15,
            use_gpu=True,
            use_memory_optimization=True,
            use_cpu_optimization=True,
            max_features=60,
            n_iter=200,
            cv_folds=7,
            feature_selection_method="mrmr"
        )

class ConfigValidator:
    """Validator for HMM clustering configurations."""
    
    @staticmethod
    def validate_config(config: HMMClusteringConfig) -> List[str]:
        """Validate configuration and return list of warnings/errors."""
        warnings = []
        
        # Validate n_components
        if config.n_components < 2:
            warnings.append("n_components should be at least 2")
        elif config.n_components > 10:
            warnings.append("n_components > 10 may lead to overfitting")
        
        # Validate lookback_windows
        if not config.lookback_windows:
            warnings.append("lookback_windows cannot be empty")
        elif min(config.lookback_windows) < 1:
            warnings.append("All lookback_windows must be >= 1")
        
        # Validate technical_indicators
        valid_indicators = [
            "rsi", "macd", "bollinger_bands", "atr", "stochastic", 
            "williams_r", "cci", "roc", "adx", "mfi"
        ]
        invalid_indicators = set(config.technical_indicators) - set(valid_indicators)
        if invalid_indicators:
            warnings.append(f"Invalid technical indicators: {invalid_indicators}")
        
        # Validate max_features
        if config.max_features < 5:
            warnings.append("max_features should be at least 5")
        elif config.max_features > 100:
            warnings.append("max_features > 100 may lead to overfitting")
        
        # Validate min_data_points
        if config.min_data_points < 100:
            warnings.append("min_data_points should be at least 100")
        
        # Validate max_missing_ratio
        if config.max_missing_ratio < 0 or config.max_missing_ratio > 1:
            warnings.append("max_missing_ratio must be between 0 and 1")
        
        # Validate regime_stability_threshold
        if config.regime_stability_threshold < 0 or config.regime_stability_threshold > 1:
            warnings.append("regime_stability_threshold must be between 0 and 1")
        
        # Validate n_iter
        if config.n_iter < 10:
            warnings.append("n_iter should be at least 10")
        elif config.n_iter > 1000:
            warnings.append("n_iter > 1000 may be computationally expensive")
        
        return warnings

class ConfigPresets:
    """Pre-defined configuration presets for common use cases."""
    
    # Crypto presets
    CRYPTO_BTC_1H = HMMClusteringConfigFactory.create_crypto_config(
        timeframe=TimeframeType.INTRADAY, market_volatility="high"
    )
    
    CRYPTO_ETH_4H = HMMClusteringConfigFactory.create_crypto_config(
        timeframe=TimeframeType.INTRADAY, market_volatility="medium"
    )
    
    CRYPTO_DAILY = HMMClusteringConfigFactory.create_crypto_config(
        timeframe=TimeframeType.DAILY, market_volatility="medium"
    )
    
    # Forex presets
    FOREX_MAJOR_1H = HMMClusteringConfigFactory.create_forex_config(
        timeframe=TimeframeType.INTRADAY, currency_pair_type="major"
    )
    
    FOREX_MINOR_4H = HMMClusteringConfigFactory.create_forex_config(
        timeframe=TimeframeType.INTRADAY, currency_pair_type="minor"
    )
    
    # Stocks presets
    STOCKS_LARGE_DAILY = HMMClusteringConfigFactory.create_stocks_config(
        timeframe=TimeframeType.DAILY, market_cap="large"
    )
    
    STOCKS_SMALL_1H = HMMClusteringConfigFactory.create_stocks_config(
        timeframe=TimeframeType.INTRADAY, market_cap="small"
    )
    
    # Specialized presets
    HIGH_FREQUENCY = HMMClusteringConfigFactory.create_high_frequency_config()
    LOW_LATENCY = HMMClusteringConfigFactory.create_low_latency_config()
    RESEARCH = HMMClusteringConfigFactory.create_research_config()

def get_config_by_name(name: str) -> Optional[HMMClusteringConfig]:
    """Get configuration by name from presets."""
    presets = {
        "crypto_btc_1h": ConfigPresets.CRYPTO_BTC_1H,
        "crypto_eth_4h": ConfigPresets.CRYPTO_ETH_4H,
        "crypto_daily": ConfigPresets.CRYPTO_DAILY,
        "forex_major_1h": ConfigPresets.FOREX_MAJOR_1H,
        "forex_minor_4h": ConfigPresets.FOREX_MINOR_4H,
        "stocks_large_daily": ConfigPresets.STOCKS_LARGE_DAILY,
        "stocks_small_1h": ConfigPresets.STOCKS_SMALL_1H,
        "high_frequency": ConfigPresets.HIGH_FREQUENCY,
        "low_latency": ConfigPresets.LOW_LATENCY,
        "research": ConfigPresets.RESEARCH
    }
    
    return presets.get(name.lower())

def create_custom_config(**kwargs) -> HMMClusteringConfig:
    """Create a custom configuration with specified parameters."""
    base_config = HMMClusteringConfig()
    
    for key, value in kwargs.items():
        if hasattr(base_config, key):
            setattr(base_config, key, value)
        else:
            raise ValueError(f"Invalid configuration parameter: {key}")
    
    return base_config

if __name__ == "__main__":
    # Example usage of configuration system
    print("HMM Clustering Configuration System")
    print("=" * 40)
    
    # Create different configurations
    crypto_config = HMMClusteringConfigFactory.create_crypto_config()
    forex_config = HMMClusteringConfigFactory.create_forex_config()
    research_config = HMMClusteringConfigFactory.create_research_config()
    
    # Validate configurations
    validator = ConfigValidator()
    
    print("\nCrypto Configuration:")
    crypto_warnings = validator.validate_config(crypto_config)
    print(f"Warnings: {crypto_warnings}")
    print(f"Components: {crypto_config.n_components}")
    print(f"Indicators: {crypto_config.technical_indicators}")
    
    print("\nForex Configuration:")
    forex_warnings = validator.validate_config(forex_config)
    print(f"Warnings: {forex_warnings}")
    print(f"Components: {forex_config.n_components}")
    print(f"Indicators: {forex_config.technical_indicators}")
    
    print("\nResearch Configuration:")
    research_warnings = validator.validate_config(research_config)
    print(f"Warnings: {research_warnings}")
    print(f"Components: {research_config.n_components}")
    print(f"Indicators: {research_config.technical_indicators}")
    
    # Test preset retrieval
    preset_config = get_config_by_name("crypto_btc_1h")
    if preset_config:
        print(f"\nPreset Configuration (crypto_btc_1h):")
        print(f"Components: {preset_config.n_components}")
        print(f"Lookback Windows: {preset_config.lookback_windows}")