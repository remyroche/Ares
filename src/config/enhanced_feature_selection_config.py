# src/config/enhanced_feature_selection_config.py

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pydantic import Field


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
    target_features: int = Field(default=100, description="Target number of features to select")
    min_features_per_category: int = Field(default=3, description="Minimum features to select from each category")
    max_features_per_category: int = Field(default=20, description="Maximum features to select from each category")

    # Dynamic threshold configuration
    enable_adaptive_thresholds: bool = Field(default=True, description="Enable adaptive threshold computation")
    variance_percentile: float = Field(default=25.0, description="Percentile for adaptive variance threshold")
    correlation_adaptive_ranges: Dict[str, float] = field(default_factory=lambda: {
        "high_feature_count": 0.98,      # >1000 features
        "medium_feature_count": 0.95,    # 500-1000 features
        "low_feature_count": 0.90,       # 200-500 features
        "very_low_feature_count": 0.85   # <200 features
    })
    mi_percentile: float = Field(default=25.0, description="Percentile for adaptive mutual information threshold")

    # Correlation management
    enable_hierarchical_clustering: bool = Field(default=True, description="Use hierarchical clustering for correlation filtering")
    max_clusters: int = Field(default=50, description="Maximum number of clusters for correlation filtering")
    clustering_method: str = Field(default="ward", description="Hierarchical clustering method")

    # Feature importance methods
    importance_methods: List[str] = field(default_factory=lambda: [
        "mutual_info", "random_forest", "f_statistic", "lightgbm"
    ])
    importance_weights: Dict[str, float] = field(default_factory=lambda: {
        "mutual_info": 0.3,
        "random_forest": 0.3,
        "f_statistic": 0.2,
        "lightgbm": 0.2
    })

    # Category-aware selection
    enable_category_aware_selection: bool = Field(default=True, description="Enable category-aware feature selection")
    category_weights: Dict[str, float] = field(default_factory=lambda: {
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
        "other": 0.8  # Slightly lower weight for uncategorized features
    })

    # Interaction features configuration
    enable_interaction_features: bool = Field(default=True, description="Enable interaction feature generation")
    max_interaction_features: int = Field(default=50, description="Maximum number of interaction features to generate")
    interaction_methods: List[str] = field(default_factory=lambda: [
        "multiplication", "ratio", "difference"
    ])
    interaction_feature_selection: str = Field(
        default="top_20_plus_category_top3",
        description="Strategy for selecting features for interactions"
    )

    # Advanced correlation filtering
    enable_advanced_correlation_filtering: bool = Field(default=True, description="Enable advanced correlation filtering")
    correlation_filtering_method: str = Field(
        default="hierarchical_clustering",
        description="Method for correlation filtering: hierarchical_clustering, recursive_elimination, or threshold_based"
    )
    correlation_threshold_fallback: float = Field(default=0.95, description="Fallback correlation threshold if adaptive method fails")

    # Final optimization
    enable_final_optimization: bool = Field(default=True, description="Enable final feature optimization")
    final_optimization_method: str = Field(
        default="rfe_lightgbm",
        description="Final optimization method: rfe_lightgbm, recursive_elimination, or importance_based"
    )

    # Performance and monitoring
    enable_performance_monitoring: bool = Field(default=True, description="Enable performance monitoring during selection")
    save_selection_metadata: bool = Field(default=True, description="Save detailed selection metadata")
    enable_correlation_analysis: bool = Field(default=True, description="Enable correlation analysis of selected features")

    # Data quality thresholds
    max_nan_ratio: float = Field(default=0.2, description="Maximum allowed NaN ratio for features")
    constant_variance_threshold: float = Field(default=1e-10, description="Threshold for identifying constant features")

    # Validation and testing
    enable_cross_validation: bool = Field(default=True, description="Enable cross-validation during feature selection")
    cv_folds: int = Field(default=5, description="Number of cross-validation folds")
    enable_stability_analysis: bool = Field(default=True, description="Enable feature stability analysis")

    # Logging and reporting
    log_level: str = Field(default="INFO", description="Logging level for feature selection process")
    enable_progress_tracking: bool = Field(default=True, description="Enable progress tracking during selection")
    save_intermediate_results: bool = Field(default=True, description="Save intermediate results for debugging")






# Example usage and validation
if __name__ == "__main__":
    # Test default configuration
    default_config = get_default_enhanced_feature_selection_config()
    print("Default Configuration:")
    print(f"Target features: {default_config['feature_reduction']['target_features']}")
    print(f"Enable interaction features: {default_config['feature_reduction']['enable_interaction_features']}")
    print(f"Correlation filtering method: {default_config['feature_reduction']['correlation_filtering_method']}")

    # Test regime-specific configuration
    trending_config = get_regime_specific_feature_selection_config("trending")
    print(f"\nTrending Regime - Momentum weight: {trending_config['feature_reduction']['category_weights']['momentum']}")

    # Test optimized configuration
    optimized_config = get_optimized_feature_selection_config()
    print(f"\nOptimized - Target features: {optimized_config['feature_reduction']['target_features']}")
    print(f"Optimized - CV folds: {optimized_config['feature_reduction']['cv_folds']}")