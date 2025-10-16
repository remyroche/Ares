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
    HIERARCHICAL = "hierarchical"
    PERFORMANCE_ADAPTIVE = "performance_adaptive"
    DYNAMIC_WEIGHTING = "dynamic_weighting"

class EconomicSignificanceType(Enum):
    """Types of economic significance to evaluate."""
    VOLATILITY_REGIME = "volatility_regime"
    TREND_STRENGTH = "trend_strength"
    VOLUME_PROFILE = "volume_profile"
    CORRELATION_STRUCTURE = "correlation_structure"
    MARKET_EFFICIENCY = "market_efficiency"
    LIQUIDITY_REGIME = "liquidity_regime"
    MICRO_REGIME = "micro_regime"
    REGIME_STABILITY = "regime_stability"
    TRANSITION_PROBABILITY = "transition_probability"
    MOMENTUM_REGIME = "momentum_regime"
    VOLUME_MOMENTUM = "volume_momentum"
    PRICE_ACTION = "price_action"
    MARKET_MICROSTRUCTURE = "market_microstructure"
    SECTOR_ROTATION = "sector_rotation"
    SHORT_TERM_MOMENTUM = "short_term_momentum"
    INTRA_BAR_PATTERNS = "intra_bar_patterns"
    MICROSTRUCTURE_PATTERNS = "microstructure_patterns"

class ClusteringAlgorithm(Enum):
    """Available clustering algorithms."""
    KMEANS = "kmeans"
    HIERARCHICAL = "hierarchical"
    DBSCAN = "dbscan"
    GMM = "gmm"
    HDBSCAN = "hdbscan"
    SPECTRAL = "spectral"
    OPTICS = "optics"
    BIRCH = "birch"
    AGGLOMERATIVE = "agglomerative"
    MEANSHIFT = "meanshift"
    ADAPTIVE = "adaptive"
    ENSEMBLE = "ensemble"
    ECONOMIC_KMEANS = "economic_kmeans"
    ECONOMIC_HIERARCHICAL = "economic_hierarchical"
    ECONOMIC_GMM = "economic_gmm"
    ECONOMIC_ADAPTIVE = "economic_adaptive"

@dataclass
class HybridRegimeConfig:
    """
    Configuration for Hybrid NAS-TAS Regime Detection System.

    This replaces the HMM clustering configuration with a hybrid approach
    that combines neural and tree-based architectures.
    """

    # Regime combination strategy
    combination_strategy: RegimeCombinationStrategy = RegimeCombinationStrategy.ADAPTIVE_FUSION

    # Core feature toggles
    enable_multi_timeframe: bool = True
    use_unified_search: bool = True
    use_signal_generation: bool = True

    # Number of regimes to detect
    n_regimes: int = 8

    # TAS (Tree Architecture Search) integration with adaptive weighting for short-term trading
    tas_config: Dict[str, Any] = field(default_factory=lambda: {
        "clustering_strategy": "auto",
        "tree_models": ["random_forest", "xgboost", "lightgbm", "extra_trees"],
        "max_features_per_model": 50,
        "min_feature_importance": 0.01,
        "base_weight": 0.4,  # Base weight in hybrid combination
        "performance_weight": 0.3,  # Weight based on performance
        "adaptive_weighting": True,
        "performance_metrics": ["accuracy", "stability", "economic_significance", "short_term_performance"],
        "weight_update_frequency": 20,  # Update weights every N samples (shorter for 15m trading)
        "min_weight": 0.1,  # Minimum weight allowed
        "max_weight": 0.9,   # Maximum weight allowed
        "short_term_focus": True  # Optimize for short-term patterns
    })

    # NAS (Neural Architecture Search) integration with adaptive weighting for short-term trading
    nas_config: Dict[str, Any] = field(default_factory=lambda: {
        "primary_architecture": "hybrid",
        "enable_neural_odes": True,
        "enable_vision_transformers": True,
        "enable_meta_learning": True,
        "search_strategy": "evolutionary",
        "base_weight": 0.6,  # Base weight in hybrid combination
        "performance_weight": 0.3,  # Weight based on performance
        "adaptive_weighting": True,
        "performance_metrics": ["accuracy", "stability", "economic_significance", "short_term_performance"],
        "weight_update_frequency": 20,  # Update weights every N samples (shorter for 15m trading)
        "min_weight": 0.1,  # Minimum weight allowed
        "max_weight": 0.9,   # Maximum weight allowed
        "short_term_focus": True,  # Optimize for short-term patterns
        "intra_bar_analysis": True  # Enable intra-bar pattern detection
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
            EconomicSignificanceType.LIQUIDITY_REGIME.value,
            EconomicSignificanceType.MICRO_REGIME.value,
            EconomicSignificanceType.REGIME_STABILITY.value,
            EconomicSignificanceType.TRANSITION_PROBABILITY.value,
            EconomicSignificanceType.MOMENTUM_REGIME.value,
            EconomicSignificanceType.VOLUME_MOMENTUM.value,
            EconomicSignificanceType.PRICE_ACTION.value,
            EconomicSignificanceType.MARKET_MICROSTRUCTURE.value,
            EconomicSignificanceType.SECTOR_ROTATION.value,
            EconomicSignificanceType.SHORT_TERM_MOMENTUM.value,
            EconomicSignificanceType.INTRA_BAR_PATTERNS.value,
            EconomicSignificanceType.MICROSTRUCTURE_PATTERNS.value
        ],
        "min_significance_score": 0.7,
        "volatility_threshold": 0.3,
        "trend_threshold": 0.5,
        "efficiency_threshold": 0.6,
        "momentum_threshold": 0.7,
        "volume_threshold": 0.6,
        "momentum_periods": [1, 2, 5, 10],  # For 15m timeframe: 15m, 30m, 1.25h, 2.5h
        "volume_analysis_window": 10,  # Shorter for 15m trading
        "price_action_sensitivity": 0.8,
        "market_microstructure_enabled": True,
        "sector_rotation_detection": True,
        "intra_bar_patterns_enabled": True,
        "microstructure_patterns_enabled": True,
        "short_term_trading": True
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

    # Advanced clustering parameters with economic integration
    clustering_config: Dict[str, Any] = field(default_factory=lambda: {
        "primary_algorithm": ClusteringAlgorithm.ECONOMIC_ADAPTIVE,
        "ensemble_algorithms": [
            ClusteringAlgorithm.ECONOMIC_KMEANS,
            ClusteringAlgorithm.ECONOMIC_HIERARCHICAL,
            ClusteringAlgorithm.KMEANS,
            ClusteringAlgorithm.GMM
        ],
        "distance_metric": "euclidean",
        "min_cluster_size": 20,
        "max_cluster_size": None,
        "cluster_validation": True,
        "optimize_clusters": True,
        "hybrid_clustering": True,
        "regime_specific_clustering": True,
        "economic_weights": True,
        "financial_weights": True,
        "ensemble_method": "voting",  # voting, stacking, bagging
        "frontier_analysis": True,
        "regime_transfer_optimization": True,
        "matrix_optimization": True,
        "hardware_acceleration": True,
        "economic_clustering": True,
        "economic_features": True,
        "momentum_integration": True,
        "volume_integration": True,
        "economic_distance_metric": "economic_euclidean",
        "momentum_threshold": 0.7,
        "volume_threshold": 0.6,
        "economic_significance_weight": 0.3,
        "momentum_weight": 0.25,
        "volume_weight": 0.25
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

    # Performance and optimization with adaptive weighting
    performance_config: Dict[str, Any] = field(default_factory=lambda: {
        "parallel_processing": True,
        "max_workers": None,  # Auto-detect
        "memory_optimization": True,
        "gpu_acceleration": True,
        "cache_results": True,
        "cache_directory": "cache/hybrid_regime",
        "execution_timeout": 300,  # seconds
        "adaptive_weighting": True,
        "performance_tracking": True,
        "weight_optimization": True,
        "matrix_optimization": True,
        "hardware_acceleration": True
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

    # Enhanced regime tagging configuration
    tagging_config: Dict[str, Any] = field(default_factory=lambda: {
        "tag_existing_data": True,
        "tag_columns": [
            "regime_id", "regime_confidence", "economic_significance",
            "financial_relevance", "regime_stability", "micro_regime_id",
            "transition_probability", "regime_duration", "tag_validation_score"
        ],
        "update_frequency": "daily",
        "preserve_original_data": True,
        "tag_historical_data": True,
        "confidence_threshold": 0.7,
        "validation_enabled": True,
        "consistency_checking": True,
        "tag_persistence": True,
        "history_management": True,
        "batch_size": 1000,
        "max_history_days": 365
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
    config.clustering_config["primary_algorithm"] = ClusteringAlgorithm.ADAPTIVE
    config.tas_config["adaptive_weighting"] = True
    config.nas_config["adaptive_weighting"] = True
    return config

def create_hierarchical_config() -> HybridRegimeConfig:
    """Create configuration with hierarchical integration strategy."""
    config = HybridRegimeConfig()
    config.combination_strategy = RegimeCombinationStrategy.HIERARCHICAL
    config.clustering_config["primary_algorithm"] = ClusteringAlgorithm.HIERARCHICAL
    config.clustering_config["ensemble_algorithms"] = [
        ClusteringAlgorithm.HIERARCHICAL,
        ClusteringAlgorithm.AGGLOMERATIVE,
        ClusteringAlgorithm.KMEANS
    ]
    return config

def create_ensemble_config() -> HybridRegimeConfig:
    """Create configuration with ensemble integration strategy."""
    config = HybridRegimeConfig()
    config.combination_strategy = RegimeCombinationStrategy.ENSEMBLE_VOTING
    config.clustering_config["ensemble_method"] = "voting"
    config.clustering_config["ensemble_algorithms"] = [
        ClusteringAlgorithm.KMEANS,
        ClusteringAlgorithm.GMM,
        ClusteringAlgorithm.HIERARCHICAL,
        ClusteringAlgorithm.DBSCAN
    ]
    return config

def create_performance_adaptive_config() -> HybridRegimeConfig:
    """Create configuration with performance-adaptive weighting."""
    config = HybridRegimeConfig()
    config.combination_strategy = RegimeCombinationStrategy.PERFORMANCE_ADAPTIVE
    config.tas_config["adaptive_weighting"] = True
    config.nas_config["adaptive_weighting"] = True
    config.performance_config["weight_optimization"] = True
    config.performance_config["performance_tracking"] = True
    return config
