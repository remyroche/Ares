from src.utils.tprint import tprint

from dataclasses import dataclass, field
from typing import Any

from pydantic import Field

# src/config/enhanced_feature_selection_config.py

@dataclass
class EnhancedFeatureSelectionConfig:
    """
    Enhanced Dynamic Feature Selection Configuration

    Addresses three key requirements:
    1. Dynamic selection process without fixed arbitrary thresholds
    2. Ensures selected features aren't too correlated
    3. Adds interaction features between top features
    """

    # Core feature selection parameters
    target_features: int = Field(
        default = 100, description="Target number of features to select"
    )
    min_features_per_category: int = Field(
        default = 3, description="Minimum features to select from each category"
    )
    max_features_per_category: int = Field(
        default = 20, description="Maximum features to select from each category"
    )

    # Dynamic threshold configuration
    enable_adaptive_thresholds: bool = Field(
        default = True, description="Enable adaptive threshold computation"
    )
    variance_percentile: float = Field(
        default = 25.0, description="Percentile for adaptive variance threshold"
    )
    correlation_adaptive_ranges: dict[str, float] = field(
        default_factory = lambda: {
            "high_feature_count": 0.98,  # >1000 features
            "medium_feature_count": 0.95,  # 500-1000 features
            "low_feature_count": 0.90,  # 200-500 features
            "very_low_feature_count": 0.85,  # <200 features
        }
    )
    mi_percentile: float = Field(
        default = 25.0, description="Percentile for adaptive mutual information threshold"
    )

    # Correlation management
    enable_hierarchical_clustering: bool = Field(
        default = True,
        description="Use hierarchical clustering for correlation filtering",
    )
    max_clusters: int = Field(
        default = 50, description="Maximum number of clusters for correlation filtering"
    )
    clustering_method: str = Field(
        default="ward", description="Hierarchical clustering method"
    )

    # Feature importance methods
    importance_methods: list[str] = field(
        default_factory = lambda: [
            "mutual_info",
            "random_forest",
            "f_statistic",
            "lightgbm",
        ]
    )
    importance_weights: dict[str, float] = field(
        default_factory = lambda: {
            "mutual_info": 0.3,
            "random_forest": 0.3,
            "f_statistic": 0.2,
            "lightgbm": 0.2,
        }
    )

    # Category-aware selection
    enable_category_aware_selection: bool = Field(
        default = True, description="Enable category-aware feature selection"
    )
    category_weights: dict[str, float] = field(
        default_factory = lambda: {
            "momentum": 1.0,
            "volatility": 1.0,
            "liquidity": 1.0,
            "microstructure": 1.0,
            "wavelet": 1.0,
            "sr_distance": 1.0,
            "statistical": 1.0,
            "candlestick": 1.0,
            "interaction": 1.0,
            "transform": 1.0,
            "other": 0.8,  # Slightly lower weight for uncategorized features
        }
    )

    # Interaction features configuration
    enable_interaction_features: bool = Field(
        default = True, description="Enable interaction feature generation"
    )
    max_interaction_features: int = Field(
        default = 50, description="Maximum number of interaction features to generate"
    )
    interaction_methods: list[str] = field(
        default_factory = lambda: [
            "multiplication",
            "ratio",
            "difference",
        ]
    )
    interaction_feature_selection: str = Field(
        default="top_20_plus_category_top3",
        description="Strategy for selecting features for interactions",
    )

    # Advanced correlation filtering
    enable_advanced_correlation_filtering: bool = Field(
        default = True, description="Enable advanced correlation filtering"
    )
    correlation_filtering_method: str = Field(
        default="hierarchical_clustering",
        description="Method for correlation filtering: hierarchical_clustering, recursive_elimination, or threshold_based",
    )
    correlation_threshold_fallback: float = Field(
        default = 0.95,
        description="Fallback correlation threshold if adaptive method fails",
    )

    # Final optimization
    enable_final_optimization: bool = Field(
        default = True, description="Enable final feature optimization"
    )
    final_optimization_method: str = Field(
        default="rfe_lightgbm",
        description="Final optimization method: rfe_lightgbm, recursive_elimination, or importance_based",
    )

    # Performance and monitoring
    enable_performance_monitoring: bool = Field(
        default = True, description="Enable performance monitoring during selection"
    )
    save_selection_metadata: bool = Field(
        default = True, description="Save detailed selection metadata"
    )
    enable_correlation_analysis: bool = Field(
        default = True, description="Enable correlation analysis of selected features"
    )

    # Data quality thresholds
    max_nan_ratio: float = Field(
        default = 0.2, description="Maximum allowed NaN ratio for features"
    )
    constant_variance_threshold: float = Field(
        default = 1e-10, description="Threshold for identifying constant features"
    )

    # Validation and testing
    enable_cross_validation: bool = Field(
        default = True, description="Enable cross-validation during feature selection"
    )
    cv_folds: int = Field(default = 5, description="Number of cross-validation folds")
    enable_stability_analysis: bool = Field(
        default = True, description="Enable feature stability analysis"
    )

    # Logging and reporting
    log_level: str = Field(
        default="INFO", description="Logging level for feature selection process"
    )
    enable_progress_tracking: bool = Field(
        default = True, description="Enable progress tracking during selection"
    )
    save_intermediate_results: bool = Field(
        default = True, description="Save intermediate results for debugging"
    )

def get_default_enhanced_feature_selection_config() -> dict[str, Any]:
    """Get default configuration for enhanced feature selection."""
    config = EnhancedFeatureSelectionConfig()

    return {
        "feature_reduction": {
            "target_features": config.target_features,
            "min_features_per_category": config.min_features_per_category,
            "max_features_per_category": config.max_features_per_category,
            # Dynamic thresholds
            "enable_adaptive_thresholds": config.enable_adaptive_thresholds,
            "variance_percentile": config.variance_percentile,
            "correlation_adaptive_ranges": config.correlation_adaptive_ranges,
            "mi_percentile": config.mi_percentile,
            # Correlation management
            "enable_hierarchical_clustering": config.enable_hierarchical_clustering,
            "max_clusters": config.max_clusters,
            "clustering_method": config.clustering_method,
            # Feature importance
            "importance_methods": config.importance_methods,
            "importance_weights": config.importance_weights,
            # Category-aware selection
            "enable_category_aware_selection": config.enable_category_aware_selection,
            "category_weights": config.category_weights,
            # Interaction features
            "enable_interaction_features": config.enable_interaction_features,
            "max_interaction_features": config.max_interaction_features,
            "interaction_methods": config.interaction_methods,
            "interaction_feature_selection": config.interaction_feature_selection,
            # Advanced correlation filtering
            "enable_advanced_correlation_filtering": config.enable_advanced_correlation_filtering,
            "correlation_filtering_method": config.correlation_filtering_method,
            "correlation_threshold_fallback": config.correlation_threshold_fallback,
            # Final optimization
            "enable_final_optimization": config.enable_final_optimization,
            "final_optimization_method": config.final_optimization_method,
            # Performance and monitoring
            "enable_performance_monitoring": config.enable_performance_monitoring,
            "save_selection_metadata": config.save_selection_metadata,
            "enable_correlation_analysis": config.enable_correlation_analysis,
            # Data quality
            "max_nan_ratio": config.max_nan_ratio,
            "constant_variance_threshold": config.constant_variance_threshold,
            # Validation
            "enable_cross_validation": config.enable_cross_validation,
            "cv_folds": config.cv_folds,
            "enable_stability_analysis": config.enable_stability_analysis,
            # Logging
            "log_level": config.log_level,
            "enable_progress_tracking": config.enable_progress_tracking,
            "save_intermediate_results": config.save_intermediate_results,
        },
    }

def get_optimized_feature_selection_config() -> dict[str, Any]:
    """Get optimized configuration for high-performance feature selection."""
    base_config = get_default_enhanced_feature_selection_config()

    # Optimize for speed and efficiency
    base_config["feature_reduction"].update(
        {
            "target_features": 80,  # Slightly fewer features for efficiency
            "max_interaction_features": 30,  # Fewer interaction features
            "cv_folds": 3,  # Fewer CV folds for speed
            "enable_stability_analysis": False,  # Disable for speed
            "save_intermediate_results": False,  # Disable for memory efficiency
        }
    )

    return base_config

def get_comprehensive_feature_selection_config() -> dict[str, Any]:
    """Get comprehensive configuration for thorough feature selection."""
    base_config = get_default_enhanced_feature_selection_config()

    # Optimize for thoroughness and quality
    base_config["feature_reduction"].update(
        {
            "target_features": 120,  # More features for thoroughness
            "max_interaction_features": 80,  # More interaction features
            "cv_folds": 10,  # More CV folds for robustness
            "enable_stability_analysis": True,  # Enable for quality
            "save_intermediate_results": True,  # Enable for analysis
            "correlation_threshold_fallback": 0.90,  # Stricter correlation filtering
        }
    )

    return base_config

def get_regime_specific_feature_selection_config(regime_type: str) -> dict[str, Any]:
    """Get regime-specific feature selection configuration."""
    base_config = get_default_enhanced_feature_selection_config()

    if regime_type == "trending":
        # Trending regimes benefit from momentum and trend features
        base_config["feature_reduction"].update(
            {
                "category_weights": {
                    "momentum": 1.2,
                    "volatility": 1.1,
                    "liquidity": 1.0,
                    "microstructure": 0.9,
                    "wavelet": 1.0,
                    "sr_distance": 1.1,
                    "statistical": 1.0,
                    "candlestick": 1.0,
                    "interaction": 1.1,
                    "transform": 1.0,
                    "other": 0.8,
                },
            }
        )

    elif regime_type == "mean_reverting":
        # Mean-reverting regimes benefit from statistical and range features
        base_config["feature_reduction"].update(
            {
                "category_weights": {
                    "momentum": 0.9,
                    "volatility": 1.2,
                    "liquidity": 1.0,
                    "microstructure": 1.1,
                    "wavelet": 1.0,
                    "sr_distance": 1.2,
                    "statistical": 1.3,
                    "candlestick": 1.1,
                    "interaction": 1.0,
                    "transform": 1.0,
                    "other": 0.8,
                },
            }
        )

    elif regime_type == "volatile":
        # Volatile regimes benefit from volatility and microstructure features
        base_config["feature_reduction"].update(
            {
                "category_weights": {
                    "momentum": 0.8,
                    "volatility": 1.4,
                    "liquidity": 1.1,
                    "microstructure": 1.3,
                    "wavelet": 1.2,
                    "sr_distance": 1.0,
                    "statistical": 1.1,
                    "candlestick": 1.0,
                    "interaction": 1.0,
                    "transform": 1.0,
                    "other": 0.7,
                },
            }
        )

    return base_config

def get_model_specific_feature_selection_config(model_type: str) -> dict[str, Any]:
    """Get model-specific feature selection configuration."""
    base_config = get_default_enhanced_feature_selection_config()

    if model_type == "AdvancedMambaHybrid":
        # AdvancedMambaHybrid needs multi-timeframe and interaction features
        base_config["feature_reduction"].update(
            {
                "target_features": 100,
                "min_features_per_category": 2,
                "max_features_per_category": 25,
                "category_weights": {
                    "momentum": 1.0,
                    "volatility": 1.0,
                    "liquidity": 0.9,
                    "microstructure": 1.1,
                    "wavelet": 1.0,
                    "sr_distance": 0.8,
                    "statistical": 0.9,
                    "candlestick": 0.8,
                    "interaction": 1.2,  # Higher weight for interaction features
                    "transform": 1.0,
                    "temporal": 1.1,     # Higher weight for temporal features
                    "other": 0.7,
                },
                "enable_interaction_features": True,
                "max_interaction_features": 50,
                "interaction_feature_selection": "top_25_plus_category_top3",
                "correlation_filtering_method": "hierarchical_clustering",
                "correlation_threshold_fallback": 0.88,
            }
        )

    elif model_type == "FinancialResNet":
        # FinancialResNet needs regime and temporal features for classification
        base_config["feature_reduction"].update(
            {
                "target_features": 120,
                "min_features_per_category": 3,
                "max_features_per_category": 30,
                "category_weights": {
                    "momentum": 0.9,
                    "volatility": 1.1,
                    "liquidity": 1.0,
                    "microstructure": 1.2,  # Higher weight for microstructure
                    "wavelet": 1.0,
                    "sr_distance": 1.0,
                    "statistical": 1.0,
                    "candlestick": 0.9,
                    "interaction": 0.8,
                    "transform": 0.9,
                    "temporal": 1.3,     # Higher weight for temporal features
                    "regime": 1.4,       # Highest weight for regime features
                    "other": 0.6,
                },
                "enable_regime_features": True,
                "correlation_filtering_method": "recursive_elimination",
                "correlation_threshold_fallback": 0.95,
                "enable_advanced_correlation_filtering": True,
            }
        )

    elif model_type == "DeepScaler":
        # DeepScaler needs clean, high-quality features for precision
        base_config["feature_reduction"].update(
            {
                "target_features": 80,
                "min_features_per_category": 2,
                "max_features_per_category": 15,
                "category_weights": {
                    "momentum": 1.0,
                    "volatility": 1.1,
                    "liquidity": 0.8,
                    "microstructure": 0.9,
                    "wavelet": 0.9,
                    "sr_distance": 1.0,
                    "statistical": 1.2,  # Higher weight for statistical features
                    "candlestick": 0.8,
                    "interaction": 0.7,
                    "transform": 0.8,
                    "temporal": 0.9,
                    "regime": 0.8,
                    "other": 0.6,
                },
                "enable_precision_filtering": True,
                "correlation_filtering_method": "threshold_based",
                "correlation_threshold_fallback": 0.85,
                "enable_advanced_correlation_filtering": True,
                "max_nan_ratio": 0.1,  # Stricter NaN ratio
                "constant_variance_threshold": 1e-8,  # Tighter constant threshold
            }
        )

    elif model_type == "NBEATS":
        # N-BEATS needs temporal and trend features for time series forecasting
        base_config["feature_reduction"].update(
            {
                "target_features": 70,
                "min_features_per_category": 2,
                "max_features_per_category": 12,
                "category_weights": {
                    "momentum": 0.8,
                    "volatility": 1.0,
                    "liquidity": 0.7,
                    "microstructure": 0.8,
                    "wavelet": 1.1,
                    "sr_distance": 0.9,
                    "statistical": 0.9,
                    "candlestick": 0.7,
                    "interaction": 0.6,
                    "transform": 0.7,
                    "temporal": 1.4,     # Highest weight for temporal features
                    "trend": 1.3,        # High weight for trend features
                    "seasonality": 1.3,  # High weight for seasonality features
                    "regime": 0.8,
                    "other": 0.5,
                },
                "enable_temporal_filtering": True,
                "correlation_filtering_method": "hierarchical_clustering",
                "correlation_threshold_fallback": 0.90,
                "max_clusters": 30,  # Fewer clusters for temporal grouping
            }
        )

    return base_config

# Example usage and validation
if __name__ == "__main__":
    # Test default configuration
    default_config = get_default_enhanced_feature_selection_config()
    tprint("Default Configuration:")
    tprint(f"Target features: {default_config['feature_reduction']['target_features']}")
    tprint(
        f"Enable interaction features: {default_config['feature_reduction']['enable_interaction_features']}"
    )
    tprint(
        f"Correlation filtering method: {default_config['feature_reduction']['correlation_filtering_method']}"
    )

    # Test regime-specific configuration
    trending_config = get_regime_specific_feature_selection_config("trending")
    tprint(
        f"\nTrending Regime - Momentum weight: {trending_config['feature_reduction']['category_weights']['momentum']}"
    )

    # Test optimized configuration
    optimized_config = get_optimized_feature_selection_config()
    tprint(
        f"\nOptimized - Target features: {optimized_config['feature_reduction']['target_features']}"
    )
    tprint(f"Optimized - CV folds: {optimized_config['feature_reduction']['cv_folds']}")
