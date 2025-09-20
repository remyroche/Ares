#!/usr/bin/env python3
"""
Unified HMM Clustering Configuration

This module provides a unified configuration system for HMM clustering,
consolidating all configuration classes and providing backward compatibility.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)


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
class UnifiedHMMClusteringConfig:
    """
    Unified configuration for HMM clustering that consolidates all previous
    configuration classes and provides backward compatibility.
    """
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
    
    # Optimization Settings
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
    
    # Enhanced Configuration (from step03)
    n_trials: int = 50
    timeout_minutes: int = 15
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        'hmm': 0.4, 'kmeans': 0.3, 'dbscan': 0.3
    })
    initial_features: int = 20
    feature_increment: int = 10
    min_improvement: float = 0.001
    patience: int = 3
    
    # Enhanced clustering configuration for 20-ish clusters covering 90% of ETH market states
    target_clusters: int = 20
    cluster_range: List[int] = field(default_factory=lambda: [18, 20, 22, 24, 26])  # Focused around 20
    coverage_target: float = 0.90
    max_clusters: int = 25  # Hard limit to prevent over-clustering
    min_clusters: int = 15  # Minimum for meaningful market state representation
    cv_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'volume': 1.0, 'volatility': 1.0, 'momentum': 1.0, 'trend': 1.0
    })
    min_cluster_size_pct: float = 0.005
    separation_min_std: float = 0.5
    separation_target_share: float = 0.8
    silhouette_min: float = 0.15
    max_silhouette_sample: int = 50000
    dimension_feature_map: Dict[str, List[str]] = field(default_factory=lambda: {
        'volume': ['volume_ratio_48h'],  # Current volume / average 48h volume (48h for 15m)
        'volatility': ['volatility_20', 'volatility_12'],  # 20 and 12 period volatility (5*4 and 3*4 for 15m)
        'momentum': ['momentum_20', 'momentum_12'],  # 20 and 12 period momentum (5*4 and 3*4 for 15m)
        'trend': ['trend_score']  # Directional Signal normalized × ADX
    })
    
    # Note: For consistency, use StandardizedFeatureCalculator.get_primary_features() 
    # instead of dimension_feature_map when available
    
    # Market-specific settings
    market_type: Optional[MarketType] = None
    timeframe_type: Optional[TimeframeType] = None
    
    def __post_init__(self):
        """Post-initialization validation and adjustments."""
        # Validate n_components - Remove artificial limits
        if self.n_components < 2:
            logger.warning("n_components < 2, setting to 2")
            self.n_components = 2
        # Remove artificial upper limit - let optimization determine optimal count
        
        # Validate lookback_windows
        if not self.lookback_windows:
            logger.warning("Empty lookback_windows, using default")
            self.lookback_windows = [5, 10, 20, 50]
        
        # Validate max_features
        if self.max_features < 5:
            logger.warning("max_features < 5, setting to 5")
            self.max_features = 5
        
        # Validate ensemble weights
        if self.ensemble_weights:
            total_weight = sum(self.ensemble_weights.values())
            if abs(total_weight - 1.0) > 0.01:
                logger.warning(f"Ensemble weights sum to {total_weight}, normalizing")
                for key in self.ensemble_weights:
                    self.ensemble_weights[key] /= total_weight
    
    @classmethod
    def create_crypto_config(
        cls, 
        timeframe: TimeframeType = TimeframeType.INTRADAY,
        market_volatility: str = "high"
    ) -> 'UnifiedHMMClusteringConfig':
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
                "rsi", "macd", "bollinger_bands", "atr", "stochastic"
            ]
            regime_stability_threshold = 0.6
        else:
            technical_indicators = [
                "rsi", "macd", "bollinger_bands", "atr"
            ]
            regime_stability_threshold = 0.8
        
        return cls(
            n_components=n_components,
            lookback_windows=lookback_windows,
            technical_indicators=technical_indicators,
            min_data_points=min_data_points,
            regime_stability_threshold=regime_stability_threshold,
            market_type=MarketType.CRYPTO,
            timeframe_type=timeframe,
            max_features=40
        )
    
    @classmethod
    def create_forex_config(
        cls,
        timeframe: TimeframeType = TimeframeType.INTRADAY,
        currency_pair_type: str = "major"
    ) -> 'UnifiedHMMClusteringConfig':
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
        
        return cls(
            n_components=n_components,
            lookback_windows=lookback_windows,
            technical_indicators=technical_indicators,
            max_features=max_features,
            market_type=MarketType.FOREX,
            timeframe_type=timeframe,
            regime_stability_threshold=0.75
        )
    
    @classmethod
    def create_high_frequency_config(cls) -> 'UnifiedHMMClusteringConfig':
        """Create configuration for high-frequency trading analysis."""
        return cls(
            n_components=6,
            lookback_windows=[3, 5, 10, 20, 50],
            technical_indicators=[
                "rsi", "macd", "bollinger_bands", "atr", "stochastic"
            ],
            min_data_points=5000,
            max_missing_ratio=0.05,
            regime_stability_threshold=0.5,
            min_regime_duration=5,
            max_features=50,
            n_iter=150,
            n_trials=100,
            timeout_minutes=30
        )
    
    @classmethod
    def create_research_config(cls) -> 'UnifiedHMMClusteringConfig':
        """Create configuration for research and experimentation."""
        return cls(
            n_components=5,
            lookback_windows=[5, 10, 20, 50, 100, 200],
            technical_indicators=[
                "rsi", "macd", "bollinger_bands", "atr", "stochastic"
            ],
            min_data_points=3000,
            max_missing_ratio=0.05,
            regime_stability_threshold=0.7,
            min_regime_duration=15,
            max_features=60,
            n_iter=200,
            cv_folds=7,
            n_trials=200,
            timeout_minutes=60
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                config_dict[key] = value.value
            else:
                config_dict[key] = value
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'UnifiedHMMClusteringConfig':
        """Create configuration from dictionary."""
        # Handle enum conversion
        if 'market_type' in config_dict and isinstance(config_dict['market_type'], str):
            config_dict['market_type'] = MarketType(config_dict['market_type'])
        if 'timeframe_type' in config_dict and isinstance(config_dict['timeframe_type'], str):
            config_dict['timeframe_type'] = TimeframeType(config_dict['timeframe_type'])
        
        return cls(**config_dict)
    
    def merge_with(self, other_config: Union['UnifiedHMMClusteringConfig', Dict[str, Any]]) -> 'UnifiedHMMClusteringConfig':
        """Merge this configuration with another configuration."""
        if isinstance(other_config, dict):
            other_dict = other_config
        else:
            other_dict = other_config.to_dict()
        
        current_dict = self.to_dict()
        current_dict.update(other_dict)
        
        return self.from_dict(current_dict)


# Backward compatibility aliases
HMMClusteringConfig = UnifiedHMMClusteringConfig


def get_config_by_name(name: str) -> Optional[UnifiedHMMClusteringConfig]:
    """Get configuration by name from presets."""
    presets = {
        "crypto_btc_1h": UnifiedHMMClusteringConfig.create_crypto_config(
            timeframe=TimeframeType.INTRADAY, market_volatility="high"
        ),
        "crypto_daily": UnifiedHMMClusteringConfig.create_crypto_config(
            timeframe=TimeframeType.DAILY, market_volatility="medium"
        ),
        "forex_major_1h": UnifiedHMMClusteringConfig.create_forex_config(
            timeframe=TimeframeType.INTRADAY, currency_pair_type="major"
        ),
        "high_frequency": UnifiedHMMClusteringConfig.create_high_frequency_config(),
        "research": UnifiedHMMClusteringConfig.create_research_config()
    }
    
    return presets.get(name.lower())


def create_custom_config(**kwargs) -> UnifiedHMMClusteringConfig:
    """Create a custom configuration with specified parameters."""
    return UnifiedHMMClusteringConfig(**kwargs)


# Configuration validation
def validate_config(config: UnifiedHMMClusteringConfig) -> List[str]:
    """Validate configuration and return list of warnings/errors."""
    warnings = []
    
    # Validate n_components - Remove artificial limits
    if config.n_components < 2:
        warnings.append("n_components should be at least 2")
    # Remove artificial upper limit - let optimization determine optimal count
    
    # Validate lookback_windows
    if not config.lookback_windows:
        warnings.append("lookback_windows cannot be empty")
    elif min(config.lookback_windows) < 1:
        warnings.append("All lookback_windows must be >= 1")
    
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
    
    return warnings