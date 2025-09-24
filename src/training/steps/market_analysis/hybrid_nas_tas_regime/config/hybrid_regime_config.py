"""
Hybrid NAS-TAS Regime Configuration

Configuration system for the hybrid regime detection that combines:
- Neural Architecture Search (NAS) from nas_regime/
- Tree Architecture Search (TAS) from ml_common TAS system
- Economic and financial relevance evaluation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from enum import Enum


class RegimeCombinationStrategy(Enum):
    """Strategy for combining TAS and NAS inputs."""
    WEIGHTED_AVERAGE = "weighted_average"
    ENSEMBLE_VOTING = "ensemble_voting"
    ECONOMIC_PRIORITY = "economic_priority"
    ADAPTIVE_FUSION = "adaptive_fusion"
    MULTI_OBJECTIVE = "multi_objective"


class EconomicSignificanceType(Enum):
    """Types of economic significance to evaluate."""
    VOLATILITY_REGIME = "volatility_regime"
    TREND_STRENGTH = "trend_strength"
    VOLUME_PROFILE = "volume_profile"
    CORRELATION_STRUCTURE = "correlation_structure"
    MARKET_EFFICIENCY = "market_efficiency"
    LIQUIDITY_REGIME = "liquidity_regime"


@dataclass
class HybridRegimeConfig:
    """
    Configuration for Hybrid NAS-TAS Regime Detection System.

    This replaces the HMM clustering configuration with a hybrid approach
    that combines neural and tree-based architectures.
    """

    # Regime combination strategy
    combination_strategy: RegimeCombinationStrategy = RegimeCombinationStrategy.ADAPTIVE_FUSION

    # Number of regimes to detect
    n_regimes: int = 8

    # TAS (Tree Architecture Search) integration
    tas_config: Dict[str, Any] = field(default_factory=lambda: {
        "clustering_strategy": "auto",
        "tree_models": ["random_forest", "xgboost", "lightgbm", "extra_trees"],
        "max_features_per_model": 50,
        "min_feature_importance": 0.01,
        "weight": 0.4  # Weight in hybrid combination
    })

    # NAS (Neural Architecture Search) integration
    nas_config: Dict[str, Any] = field(default_factory=lambda: {
        "primary_architecture": "hybrid",
        "enable_neural_odes": True,
        "enable_vision_transformers": True,
        "enable_meta_learning": True,
        "search_strategy": "evolutionary",
        "weight": 0.6  # Weight in hybrid combination
    })

    # Economic significance evaluation
    economic_evaluation: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "significance_types": [
            EconomicSignificanceType.VOLATILITY_REGIME.value,
            EconomicSignificanceType.TREND_STRENGTH.value,
            EconomicSignificanceType.VOLUME_PROFILE.value,
            EconomicSignificanceType.CORRELATION_STRUCTURE.value,
            EconomicSignificanceType.MARKET_EFFICIENCY.value,
            EconomicSignificanceType.LIQUIDITY_REGIME.value
        ],
        "min_significance_score": 0.7,
        "volatility_threshold": 0.3,
        "trend_threshold": 0.5,
        "efficiency_threshold": 0.6
    })

    # Financial relevance parameters
    financial_relevance: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "sharpe_ratio_threshold": 0.5,
        "max_drawdown_threshold": 0.15,
        "win_rate_threshold": 0.55,
        "profit_factor_threshold": 1.2,
        "regime_stability_period": 50,  # Minimum periods for stable regime
        "transition_smoothness": 0.8   # Smoothness of regime transitions
    })

    # Clustering parameters
    clustering_config: Dict[str, Any] = field(default_factory=lambda: {
        "algorithm": "adaptive",  # adaptive, kmeans, dbscan, agglomerative
        "distance_metric": "euclidean",
        "min_cluster_size": 20,
        "max_cluster_size": None,
        "cluster_validation": True,
        "optimize_clusters": True
    })

    # Feature engineering
    feature_config: Dict[str, Any] = field(default_factory=lambda: {
        "technical_indicators": True,
        "price_features": True,
        "volume_features": True,
        "volatility_features": True,
        "momentum_features": True,
        "statistical_features": True,
        "correlation_features": True,
        "lookback_periods": [5, 10, 20, 50, 100],
        "normalization": "standard"  # standard, robust, minmax
    })

    # Model validation and testing
    validation_config: Dict[str, Any] = field(default_factory=lambda: {
        "cross_validation_folds": 5,
        "train_test_split": 0.8,
        "validation_metrics": [
            "accuracy", "precision", "recall", "f1_score",
            "silhouette_score", "calinski_harabasz_score",
            "economic_significance", "financial_relevance"
        ],
        "backtesting_enabled": True,
        "backtesting_periods": 100
    })

    # Performance and optimization
    performance_config: Dict[str, Any] = field(default_factory=lambda: {
        "parallel_processing": True,
        "max_workers": None,  # Auto-detect
        "memory_optimization": True,
        "gpu_acceleration": True,
        "cache_results": True,
        "cache_directory": "cache/hybrid_regime",
        "execution_timeout": 300  # seconds
    })

    # Output and reporting
    output_config: Dict[str, Any] = field(default_factory=lambda: {
        "save_regime_data": True,
        "save_economic_analysis": True,
        "save_financial_analysis": True,
        "save_performance_metrics": True,
        "generate_plots": True,
        "output_directory": "generated/market_analysis/hybrid_regime",
        "report_format": ["json", "csv", "html"]
    })

    # Regime tagging configuration
    tagging_config: Dict[str, Any] = field(default_factory=lambda: {
        "tag_existing_data": True,
        "tag_columns": ["regime_id", "regime_confidence", "economic_significance", "financial_relevance"],
        "update_frequency": "daily",
        "preserve_original_data": True,
        "tag_historical_data": True
    })


def create_default_hybrid_config() -> HybridRegimeConfig:
    """Create a default hybrid regime configuration."""
    return HybridRegimeConfig()


def create_economic_focused_config() -> HybridRegimeConfig:
    """Create configuration focused on economic significance."""
    config = HybridRegimeConfig()
    config.combination_strategy = RegimeCombinationStrategy.ECONOMIC_PRIORITY
    config.economic_evaluation["enabled"] = True
    config.economic_evaluation["min_significance_score"] = 0.8
    config.nas_config["weight"] = 0.7
    config.tas_config["weight"] = 0.3
    return config


def create_trading_focused_config() -> HybridRegimeConfig:
    """Create configuration focused on trading viability."""
    config = HybridRegimeConfig()
    config.combination_strategy = RegimeCombinationStrategy.MULTI_OBJECTIVE
    config.financial_relevance["enabled"] = True
    config.validation_config["backtesting_enabled"] = True
    config.output_config["save_financial_analysis"] = True
    return config


def create_adaptive_config() -> HybridRegimeConfig:
    """Create configuration with adaptive fusion strategy."""
    config = HybridRegimeConfig()
    config.combination_strategy = RegimeCombinationStrategy.ADAPTIVE_FUSION
    config.performance_config["parallel_processing"] = True
    config.clustering_config["algorithm"] = "adaptive"
    return config