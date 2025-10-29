"""
Regime Feature Categorization System

This module provides a comprehensive categorization system for regime features
based on their intended use case. It clearly defines which features should be
used for different purposes to avoid data leakage and ensure optimal performance.

Use Cases:
1. HDBSCAN Clustering - Features optimized for density-based clustering
2. Regime Clustering - Features for general regime identification
3. Regime Models Training - Features for training regime detection models
4. Regime Ensemble Training - Features for meta-learner training

Key Principles:
- Clustering features should NEVER be used during live trading
- Regime detection features should be robust and stable
- Training features should avoid lookahead bias
- Ensemble features should complement base model features
"""

from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import warnings

# Import feature generators
from .regime_features import (
    RegimeStatisticalFeatureGenerator,
    RegimeStructuralTrendFeatureGenerator,
    RegimeVolatilityFeatureGenerator,
    RegimeVolumeFeatureGenerator,
    RegimeEntropyGenerator,
    RegimeComplexityGenerator,
    RegimeFractalDimensionGenerator,
    RegimeHurstExponentGenerator,
    RegimeMemoryStrengthGenerator,
    RegimeCrossAssetGenerator,
    RegimeTransitionProbabilityGenerator,
    RegimeFeatureIntegration
)

from .clustering_features import (
    ClusteringDistanceGenerator,
    ClusteringSeparationGenerator,
    ClusteringStabilityGenerator,
    ClusteringIntegration
)

# Import other feature categories
from .momentum import *
from .volatility import (
    VolatilityFeatureGenerator, VectorBTVolatilityFeatureGenerator,
    VectorBTBollingerBandsGenerator, VectorBTAverageTrueRangeGenerator,
    VectorBTGarmanKlassVolatilityGenerator, VectorBTParkinsonVolatilityGenerator,
    VectorBTRogersSatchellVolatilityGenerator, VectorBTYangZhangVolatilityGenerator,
    create_default_volatility_generators
)
from .trend import *
from .volume import *


class FeatureUseCase(Enum):
    """Enumeration of feature use cases."""
    HDBSCAN_CLUSTERING = "hdbscan_clustering"
    REGIME_CLUSTERING = "regime_clustering"
    REGIME_MODELS_TRAINING = "regime_models_training"
    REGIME_ENSEMBLE_TRAINING = "regime_ensemble_training"
    LIVE_TRADING = "live_trading"  # Features that can be used during live trading


@dataclass
class FeatureCategory:
    """Feature category definition."""
    name: str
    description: str
    use_cases: Set[FeatureUseCase]
    generators: List[Any]
    feature_names: List[str]
    priority: int  # Higher priority = more important for the use case
    stability_required: bool  # Whether feature should be stable over time
    lookahead_safe: bool  # Whether feature avoids lookahead bias


class RegimeFeatureCategorizer:
    """
    Regime Feature Categorizer.
    
    Provides comprehensive categorization of regime features based on their
    intended use case and characteristics.
    """
    
    def __init__(self):
        self.categories = self._initialize_categories()
        self.feature_mappings = self._create_feature_mappings()
    
    def _initialize_categories(self) -> Dict[str, FeatureCategory]:
        """Initialize feature categories."""
        categories = {}
        
        # Core Regime Features (for regime detection and clustering)
        categories["core_regime"] = FeatureCategory(
            name="Core Regime Features",
            description="Essential features for regime identification and clustering. Includes lagged, derived, and temporal features.",
            use_cases={
                FeatureUseCase.HDBSCAN_CLUSTERING,
                FeatureUseCase.REGIME_CLUSTERING,
                FeatureUseCase.REGIME_MODELS_TRAINING,
                FeatureUseCase.REGIME_ENSEMBLE_TRAINING
            },
            generators=[
                RegimeStatisticalFeatureGenerator(),
                RegimeVolatilityFeatureGenerator(),
                RegimeVolumeFeatureGenerator()
            ],
            feature_names=[
                "regime_persistence", "vol_regime_strength", "vol_clustering",
                "vol_regime_change", "volume_regime_strength", "volume_clustering",
                "statistical_persistence", "distribution_stability",
                # Lagged features (windowed features for past 3-5 bars)
                "lagged_ma_3", "lagged_ma_4", "lagged_ma_5",
                "lagged_volatility_3", "lagged_volatility_4", "lagged_volatility_5",
                # Derived features (ratios, normalized indicators, trend strength)
                "price_zscore", "price_to_ma_ratio_5", "price_to_ma_ratio_10", "price_to_ma_ratio_20",
                "returns_cv", "trend_strength_ratio", "volatility_ratio",
                # Temporal awareness features (differences, momentum indicators)
                "temporal_price_diff", "temporal_price_acceleration",
                "momentum_3", "momentum_5", "momentum_10", "momentum_acceleration",
                "roc_3", "roc_5", "roc_10",
                # Comprehensive technical indicators (only features NOT in other categories)
                "ma_slope_3", "ma_slope_4", "ma_slope_5",
                "ema_slope_3", "ema_slope_4", "ema_slope_5",
                "rolling_std_5", "rolling_std_10", "rolling_std_15", "rolling_std_20",
                "price_momentum_1", "price_momentum_2", "price_momentum_3", "price_momentum_4", "price_momentum_5",
                "returns_zscore_5", "returns_zscore_10",
                "ewma_volatility_10", "ewma_volatility_20"
            ],
            priority=10,
            stability_required=True,
            lookahead_safe=True
        )
        
        # Advanced Regime Features (for sophisticated regime analysis)
        categories["advanced_regime"] = FeatureCategory(
            name="Advanced Regime Features",
            description="Advanced features for complex regime analysis",
            use_cases={
                FeatureUseCase.HDBSCAN_CLUSTERING,
                FeatureUseCase.REGIME_CLUSTERING,
                FeatureUseCase.REGIME_MODELS_TRAINING
            },
            generators=[
                RegimeEntropyGenerator(),
                RegimeComplexityGenerator(),
                RegimeFractalDimensionGenerator(),
                RegimeHurstExponentGenerator(),
                RegimeMemoryStrengthGenerator()
            ],
            feature_names=[
                "regime_entropy", "regime_complexity", "regime_fractal_dimension",
                "regime_hurst_exponent", "regime_memory_strength"
            ],
            priority=8,
            stability_required=True,
            lookahead_safe=True
        )
        
        
        # Cross-Asset Features (for multi-asset regime analysis)
        categories["cross_asset"] = FeatureCategory(
            name="Cross-Asset Features",
            description="Cross-asset correlation and regime features",
            use_cases={
                FeatureUseCase.HDBSCAN_CLUSTERING,
                FeatureUseCase.REGIME_CLUSTERING,
                FeatureUseCase.REGIME_MODELS_TRAINING
            },
            generators=[
                RegimeCrossAssetGenerator()
            ],
            feature_names=[
                "cross_timeframe_corr", "regime_persistence_score",
                "price_volume_sync", "regime_sync_strength"
            ],
            priority=6,
            stability_required=True,
            lookahead_safe=True
        )
        
        # Transition Features (for regime change detection)
        categories["transition"] = FeatureCategory(
            name="Transition Features",
            description="Features for regime transition detection",
            use_cases={
                FeatureUseCase.REGIME_MODELS_TRAINING,
                FeatureUseCase.REGIME_ENSEMBLE_TRAINING
            },
            generators=[
                RegimeTransitionProbabilityGenerator()
            ],
            feature_names=[
                "cusum_change_point", "change_point_prob", "regime_change_intensity",
                "transition_prob", "regime_persistence_prob"
            ],
            priority=8,
            stability_required=False,  # These are inherently about change
            lookahead_safe=True
        )
        
        # Clustering-Specific Features (NEVER for live trading)
        categories["clustering_only"] = FeatureCategory(
            name="Clustering-Only Features",
            description="Features designed specifically for clustering algorithms",
            use_cases={
                FeatureUseCase.HDBSCAN_CLUSTERING,
                FeatureUseCase.REGIME_CLUSTERING
            },
            generators=[
                ClusteringDistanceGenerator(),
                ClusteringSeparationGenerator(),
                ClusteringStabilityGenerator()
            ],
            feature_names=[
                "price_distance", "volume_distance", "cluster_compactness",
                "separation_strength", "cluster_consistency", "temporal_stability"
            ],
            priority=9,
            stability_required=True,
            lookahead_safe=True
        )
        
        # Live Trading Features (safe for real-time use)
        categories["live_trading"] = FeatureCategory(
            name="Live Trading Features",
            description="Features safe for live trading and real-time regime detection",
            use_cases={
                FeatureUseCase.LIVE_TRADING,
                FeatureUseCase.REGIME_MODELS_TRAINING,
                FeatureUseCase.REGIME_ENSEMBLE_TRAINING
            },
            generators=[
                # Basic technical indicators that are live-trading safe
            ],
            feature_names=[
                "rsi", "macd", "bollinger_bands", "atr", "sma", "ema",
                "volume_sma", "price_momentum", "volatility_rolling",
                # Lagged features (windowed features for past 3-5 bars)
                "lagged_ma_3", "lagged_ma_4", "lagged_ma_5",
                "lagged_volatility_3", "lagged_volatility_4", "lagged_volatility_5",
                "lagged_range_3", "lagged_range_4", "lagged_range_5",
                "lagged_volume_ma_3", "lagged_volume_ma_4", "lagged_volume_ma_5",
                "lagged_max_3", "lagged_max_4", "lagged_max_5",
                "lagged_min_3", "lagged_min_4", "lagged_min_5",
                # Derived features (ratios, normalized indicators, trend strength)
                "price_zscore", "price_to_ma_ratio_5", "price_to_ma_ratio_10", "price_to_ma_ratio_20",
                "returns_cv", "trend_strength_ratio", "volatility_ratio",
                "range_ratio", "volume_ratio", "mean_reversion_strength_normalized",
                # Temporal awareness features (differences, momentum indicators)
                "temporal_price_diff", "temporal_price_acceleration",
                "momentum_3", "momentum_5", "momentum_10", "momentum_acceleration",
                "roc_3", "roc_5", "roc_10",
                "volatility_trend", "volume_momentum", "volume_roc", "price_position_trend",
                # Comprehensive technical indicators (only features NOT in other categories)
                "ma_slope_3", "ma_slope_4", "ma_slope_5",
                "ema_slope_3", "ema_slope_4", "ema_slope_5",
                "rolling_std_5", "rolling_std_10", "rolling_std_15", "rolling_std_20",
                "price_momentum_1", "price_momentum_2", "price_momentum_3", "price_momentum_4", "price_momentum_5",
                "returns_zscore_5", "returns_zscore_10",
                "ewma_volatility_10", "ewma_volatility_20"
            ],
            priority=5,
            stability_required=True,
            lookahead_safe=True
        )
        
        # Structural Trend Features (for trend regime analysis)
        categories["structural_trend"] = FeatureCategory(
            name="Structural Trend Features",
            description="Features for structural trend regime analysis",
            use_cases={
                FeatureUseCase.HDBSCAN_CLUSTERING,
                FeatureUseCase.REGIME_CLUSTERING,
                FeatureUseCase.REGIME_MODELS_TRAINING,
                FeatureUseCase.REGIME_ENSEMBLE_TRAINING
            },
            generators=[
                RegimeStructuralTrendFeatureGenerator()
            ],
            feature_names=[
                "structural_persistence", "trend_regime_persistence",
                "market_structure_strength", "trend_transition_prob"
            ],
            priority=8,
            stability_required=True,
            lookahead_safe=True
        )
        
        return categories
    
    def _create_feature_mappings(self) -> Dict[FeatureUseCase, List[str]]:
        """Create feature mappings for each use case."""
        mappings = {}
        
        for use_case in FeatureUseCase:
            mappings[use_case] = []
            
            for category in self.categories.values():
                if use_case in category.use_cases:
                    mappings[use_case].extend(category.feature_names)
        
        return mappings
    
    def get_features_for_use_case(self, use_case: FeatureUseCase) -> List[str]:
        """Get all features suitable for a specific use case."""
        return self.feature_mappings.get(use_case, [])
    
    def get_categories_for_use_case(self, use_case: FeatureUseCase) -> List[FeatureCategory]:
        """Get all categories suitable for a specific use case."""
        return [
            category for category in self.categories.values()
            if use_case in category.use_cases
        ]
    
    def get_generators_for_use_case(self, use_case: FeatureUseCase) -> List[Any]:
        """Get all generators suitable for a specific use case."""
        generators = []
        for category in self.get_categories_for_use_case(use_case):
            generators.extend(category.generators)
        return generators
    
    def get_priority_features(self, use_case: FeatureUseCase, max_features: int = 50) -> List[str]:
        """Get priority features for a use case, sorted by importance."""
        categories = self.get_categories_for_use_case(use_case)
        
        # Sort categories by priority
        sorted_categories = sorted(categories, key=lambda x: x.priority, reverse=True)
        
        priority_features = []
        for category in sorted_categories:
            if len(priority_features) >= max_features:
                break
            priority_features.extend(category.feature_names)
        
        return priority_features[:max_features]
    
    def validate_feature_usage(self, features: List[str], use_case: FeatureUseCase) -> Tuple[List[str], List[str]]:
        """Validate feature usage for a specific use case."""
        valid_features = []
        invalid_features = []
        
        allowed_features = self.get_features_for_use_case(use_case)
        
        for feature in features:
            if feature in allowed_features:
                valid_features.append(feature)
            else:
                invalid_features.append(feature)
        
        return valid_features, invalid_features
    
    def get_feature_requirements(self, use_case: FeatureUseCase) -> Dict[str, Any]:
        """Get feature requirements for a specific use case."""
        categories = self.get_categories_for_use_case(use_case)
        
        requirements = {
            "total_categories": len(categories),
            "total_features": len(self.get_features_for_use_case(use_case)),
            "stability_required": any(cat.stability_required for cat in categories),
            "lookahead_safe": all(cat.lookahead_safe for cat in categories),
            "priority_features": self.get_priority_features(use_case, 20),
            "categories": [cat.name for cat in categories]
        }
        
        return requirements


# Convenience functions
def get_hdbscan_features() -> List[str]:
    """Get features optimized for HDBSCAN clustering."""
    categorizer = RegimeFeatureCategorizer()
    return categorizer.get_priority_features(FeatureUseCase.HDBSCAN_CLUSTERING, 100)


def get_regime_clustering_features() -> List[str]:
    """Get features for general regime clustering."""
    categorizer = RegimeFeatureCategorizer()
    return categorizer.get_priority_features(FeatureUseCase.REGIME_CLUSTERING, 80)


def get_regime_models_training_features() -> List[str]:
    """Get features for regime models training."""
    categorizer = RegimeFeatureCategorizer()
    return categorizer.get_priority_features(FeatureUseCase.REGIME_MODELS_TRAINING, 60)


def get_regime_ensemble_training_features() -> List[str]:
    """Get features for regime ensemble training."""
    categorizer = RegimeFeatureCategorizer()
    return categorizer.get_priority_features(FeatureUseCase.REGIME_ENSEMBLE_TRAINING, 40)


def get_live_trading_features() -> List[str]:
    """Get features safe for live trading."""
    categorizer = RegimeFeatureCategorizer()
    return categorizer.get_priority_features(FeatureUseCase.LIVE_TRADING, 30)


def validate_feature_set(features: List[str], use_case: FeatureUseCase) -> Dict[str, Any]:
    """Validate a feature set for a specific use case."""
    categorizer = RegimeFeatureCategorizer()
    valid_features, invalid_features = categorizer.validate_feature_usage(features, use_case)
    
    return {
        "valid_features": valid_features,
        "invalid_features": invalid_features,
        "valid_count": len(valid_features),
        "invalid_count": len(invalid_features),
        "validation_passed": len(invalid_features) == 0,
        "recommendations": categorizer.get_priority_features(use_case, 20)
    }


# Feature usage guidelines
FEATURE_USAGE_GUIDELINES = {
    FeatureUseCase.HDBSCAN_CLUSTERING: {
        "description": "Features optimized for density-based clustering",
        "key_characteristics": [
            "Distance-based features for cluster separation",
            "Density features for cluster identification",
            "Stability features for robust clustering",
            "Cross-asset features for multi-dimensional clustering"
        ],
        "avoid": [
            "Features with lookahead bias",
            "Features that change frequently",
            "Features that are too noisy"
        ],
        "recommended_count": "50-100 features",
        "priority_categories": ["clustering_only", "core_regime", "advanced_regime"]
    },
    
    FeatureUseCase.REGIME_CLUSTERING: {
        "description": "Features for general regime identification and clustering",
        "key_characteristics": [
            "Regime persistence features",
            "Volatility regime features",
            "Volume regime features",
            "Structural trend features"
        ],
        "avoid": [
            "Clustering-specific features (use HDBSCAN for that)",
            "Features with high correlation",
            "Features that are not regime-relevant"
        ],
        "recommended_count": "40-80 features",
        "priority_categories": ["core_regime", "structural_trend", "cross_asset"]
    },
    
    FeatureUseCase.REGIME_MODELS_TRAINING: {
        "description": "Features for training regime detection models",
        "key_characteristics": [
            "Stable features over time",
            "Features that avoid lookahead bias",
            "Features that capture regime transitions",
            "Features that are economically meaningful"
        ],
        "avoid": [
            "Clustering-only features",
            "Features with lookahead bias",
            "Features that are too unstable"
        ],
        "recommended_count": "30-60 features",
        "priority_categories": ["core_regime", "transition", "cross_asset"]
    },
    
    FeatureUseCase.REGIME_ENSEMBLE_TRAINING: {
        "description": "Features for meta-learner training in ensemble models",
        "key_characteristics": [
            "Features that complement base models",
            "Features that capture regime stability",
            "Features that are robust to overfitting",
            "Features that provide different information than base models"
        ],
        "avoid": [
            "Clustering-only features",
            "Features that are highly correlated with base model features",
            "Features that are too complex for meta-learning"
        ],
        "recommended_count": "20-40 features",
        "priority_categories": ["core_regime", "transition", "live_trading"]
    }
}


__all__ = [
    'RegimeFeatureCategorizer',
    'FeatureUseCase',
    'FeatureCategory',
    'get_hdbscan_features',
    'get_regime_clustering_features',
    'get_regime_models_training_features',
    'get_regime_ensemble_training_features',
    'get_live_trading_features',
    'validate_feature_set',
    'FEATURE_USAGE_GUIDELINES'
]