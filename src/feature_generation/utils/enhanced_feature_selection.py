"""
Enhanced Feature Selection for Regime Clustering

This module combines cluster distinctiveness metrics with the existing
economic relevance system to select optimal features for regime clustering.

Key Features:
- Combines cluster distinctiveness with economic relevance weights
- Uses existing category-based weighting system
- Integrates with economic significance features
- Leverages existing quality scoring for temporal stability
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings

from .cluster_distinctiveness_metrics import (
    ClusterDistinctivenessCalculator, ClusterDistinctivenessConfig
)
from ..integration.feature_bank_integration import FeatureBankCategory


@dataclass
class EnhancedFeatureSelectionConfig:
    """Configuration for enhanced feature selection."""
    
    # Economic relevance weights (from existing system)
    economic_weights: Dict[FeatureBankCategory, float] = None
    
    # Cluster distinctiveness settings
    cluster_distinctiveness_weight: float = 0.4
    economic_relevance_weight: float = 0.4
    temporal_stability_weight: float = 0.2
    
    # Performance settings
    enable_caching: bool = True
    batch_size: int = 1000
    max_features_per_category: int = 20
    
    # Quality thresholds
    min_combined_score: float = 0.3
    min_economic_relevance: float = 0.5
    min_cluster_distinctiveness: float = 0.2
    
    def __post_init__(self):
        """Set default economic weights if not provided."""
        if self.economic_weights is None:
            self.economic_weights = {
                FeatureBankCategory.REGIME: 0.4,
                FeatureBankCategory.VOLUME: 0.2,
                FeatureBankCategory.TREND: 0.2,
                FeatureBankCategory.VOLATILITY: 0.15,
                FeatureBankCategory.MOMENTUM: 0.05
            }


class EnhancedFeatureSelector:
    """
    Enhanced feature selector that combines cluster distinctiveness
    with economic relevance for regime clustering.
    """
    
    def __init__(self, config: Optional[EnhancedFeatureSelectionConfig] = None):
        self.config = config or EnhancedFeatureSelectionConfig()
        
        # Initialize cluster distinctiveness calculator
        cluster_config = ClusterDistinctivenessConfig(
            enable_fast_proxies=True,
            enable_caching=self.config.enable_caching,
            batch_size=self.config.batch_size,
            use_approximate_silhouette=True,
            silhouette_sample_ratio=0.1
        )
        self.cluster_calculator = ClusterDistinctivenessCalculator(cluster_config)
        
        # Cache for feature scores
        self._score_cache = {} if self.config.enable_caching else None
    
    def select_optimal_features(self, 
                              features: Dict[str, np.ndarray], 
                              cluster_labels: np.ndarray,
                              feature_categories: Dict[str, FeatureBankCategory],
                              max_features: int) -> Dict[str, np.ndarray]:
        """
        Select optimal features using combined economic relevance and cluster distinctiveness.
        
        Args:
            features: Dictionary of feature names to feature values
            cluster_labels: Cluster labels for each sample
            feature_categories: Mapping of feature names to their categories
            max_features: Maximum number of features to select
            
        Returns:
            Dictionary of selected features
        """
        if not features or len(cluster_labels) == 0:
            return {}
        
        # 1. Calculate cluster distinctiveness scores
        distinctiveness_metrics = self.cluster_calculator.calculate_feature_distinctiveness(
            features, cluster_labels
        )
        
        # 2. Calculate economic relevance scores
        economic_scores = self._calculate_economic_relevance_scores(
            features, feature_categories
        )
        
        # 3. Calculate temporal stability scores (using existing quality scoring)
        stability_scores = self._calculate_temporal_stability_scores(features)
        
        # 4. Combine scores
        combined_scores = self._combine_scores(
            distinctiveness_metrics, economic_scores, stability_scores
        )
        
        # 5. Select top features
        selected_features = self._select_top_features(
            features, combined_scores, max_features
        )
        
        return selected_features
    
    def _calculate_economic_relevance_scores(self, 
                                           features: Dict[str, np.ndarray],
                                           feature_categories: Dict[str, FeatureBankCategory]) -> Dict[str, float]:
        """Calculate economic relevance scores using existing category weights."""
        economic_scores = {}
        
        for feature_name, feature_values in features.items():
            # Get category for this feature
            category = feature_categories.get(feature_name, FeatureBankCategory.MOMENTUM)
            
            # Get economic weight for this category
            economic_weight = self.config.economic_weights.get(category, 0.1)
            
            # Calculate feature quality score
            quality_score = self._calculate_feature_quality_score(feature_values)
            
            # Combined economic relevance score
            economic_scores[feature_name] = economic_weight * quality_score
        
        return economic_scores
    
    def _calculate_feature_quality_score(self, feature_values: np.ndarray) -> float:
        """Calculate feature quality score (variance, stability, etc.)."""
        # Check for constant values
        if len(np.unique(feature_values)) < 3:
            return 0.0
        
        # Check for excessive NaN values
        nan_ratio = np.isnan(feature_values).sum() / len(feature_values)
        if nan_ratio > 0.5:
            return 0.0
        
        # Calculate variance (normalized)
        variance = np.var(feature_values)
        if variance < 1e-8:
            return 0.0
        
        # Calculate stability (inverse of rolling std)
        if len(feature_values) > 20:
            rolling_std = pd.Series(feature_values).rolling(20).std()
            stability = 1.0 / (1.0 + np.mean(rolling_std.dropna()))
        else:
            stability = 1.0
        
        # Combined quality score
        quality_score = min(1.0, variance * stability)
        
        return float(quality_score)
    
    def _calculate_temporal_stability_scores(self, features: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate temporal stability scores using existing quality scoring approach."""
        stability_scores = {}
        
        for feature_name, feature_values in features.items():
            # Use existing temporal stability calculation approach
            stability_score = self._calculate_temporal_stability(feature_values)
            stability_scores[feature_name] = stability_score
        
        return stability_scores
    
    def _calculate_temporal_stability(self, feature_values: np.ndarray, 
                                    stability_window: int = 20) -> float:
        """Calculate temporal stability of feature values."""
        if len(feature_values) < stability_window * 2:
            return 0.5  # Default moderate stability
        
        # Calculate rolling standard deviation
        rolling_std = pd.Series(feature_values).rolling(stability_window).std()
        
        # Lower rolling std = higher stability
        stability_score = 1.0 / (1.0 + np.mean(rolling_std.dropna()))
        
        # Check for trend stability (avoid features with strong trends)
        if len(feature_values) > 10:
            trend_strength = abs(np.corrcoef(np.arange(len(feature_values)), 
                                           feature_values)[0, 1])
            # Penalize features with strong trends
            stability_score *= (1.0 - trend_strength)
        
        return max(0.0, min(1.0, stability_score))
    
    def _combine_scores(self, 
                       distinctiveness_metrics: Dict[str, Dict[str, float]],
                       economic_scores: Dict[str, float],
                       stability_scores: Dict[str, float]) -> Dict[str, float]:
        """Combine all scores into final feature scores."""
        combined_scores = {}
        
        for feature_name in distinctiveness_metrics.keys():
            # Get distinctiveness score
            distinctiveness_score = distinctiveness_metrics[feature_name].get('combined_score', 0.0)
            
            # Get economic relevance score
            economic_score = economic_scores.get(feature_name, 0.0)
            
            # Get temporal stability score
            stability_score = stability_scores.get(feature_name, 0.0)
            
            # Combine scores with weights
            combined_score = (
                self.config.cluster_distinctiveness_weight * distinctiveness_score +
                self.config.economic_relevance_weight * economic_score +
                self.config.temporal_stability_weight * stability_score
            )
            
            combined_scores[feature_name] = combined_score
        
        return combined_scores
    
    def _select_top_features(self, 
                           features: Dict[str, np.ndarray],
                           combined_scores: Dict[str, float],
                           max_features: int) -> Dict[str, np.ndarray]:
        """Select top features based on combined scores."""
        # Sort features by combined score
        sorted_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Apply quality filters
        filtered_features = []
        for feature_name, score in sorted_features:
            # Check minimum thresholds
            if (score >= self.config.min_combined_score and
                combined_scores[feature_name] >= self.config.min_economic_relevance and
                combined_scores[feature_name] >= self.config.min_cluster_distinctiveness):
                filtered_features.append((feature_name, score))
        
        # Select top N features
        selected_names = [name for name, _ in filtered_features[:max_features]]
        
        # Return selected features
        selected_features = {}
        for name in selected_names:
            if name in features:
                selected_features[name] = features[name]
        
        return selected_features
    
    def get_feature_selection_report(self, 
                                   features: Dict[str, np.ndarray],
                                   cluster_labels: np.ndarray,
                                   feature_categories: Dict[str, FeatureBankCategory]) -> Dict[str, Any]:
        """Generate a detailed report of feature selection process."""
        # Calculate all scores
        distinctiveness_metrics = self.cluster_calculator.calculate_feature_distinctiveness(
            features, cluster_labels
        )
        economic_scores = self._calculate_economic_relevance_scores(
            features, feature_categories
        )
        stability_scores = self._calculate_temporal_stability_scores(features)
        combined_scores = self._combine_scores(
            distinctiveness_metrics, economic_scores, stability_scores
        )
        
        # Generate report
        report = {
            'total_features': len(features),
            'feature_scores': {},
            'category_breakdown': {},
            'selection_summary': {}
        }
        
        # Individual feature scores
        for feature_name in features.keys():
            report['feature_scores'][feature_name] = {
                'distinctiveness': distinctiveness_metrics[feature_name].get('combined_score', 0.0),
                'economic_relevance': economic_scores.get(feature_name, 0.0),
                'temporal_stability': stability_scores.get(feature_name, 0.0),
                'combined_score': combined_scores.get(feature_name, 0.0),
                'category': feature_categories.get(feature_name, 'unknown').value
            }
        
        # Category breakdown
        for category in FeatureBankCategory:
            category_features = [name for name, cat in feature_categories.items() 
                               if cat == category]
            if category_features:
                category_scores = [combined_scores.get(name, 0.0) for name in category_features]
                report['category_breakdown'][category.value] = {
                    'count': len(category_features),
                    'avg_score': np.mean(category_scores),
                    'max_score': np.max(category_scores),
                    'min_score': np.min(category_scores)
                }
        
        # Selection summary
        report['selection_summary'] = {
            'weights': {
                'cluster_distinctiveness': self.config.cluster_distinctiveness_weight,
                'economic_relevance': self.config.economic_relevance_weight,
                'temporal_stability': self.config.temporal_stability_weight
            },
            'thresholds': {
                'min_combined_score': self.config.min_combined_score,
                'min_economic_relevance': self.config.min_economic_relevance,
                'min_cluster_distinctiveness': self.config.min_cluster_distinctiveness
            }
        }
        
        return report


# Convenience functions
def select_features_with_enhanced_selection(features: Dict[str, np.ndarray],
                                          cluster_labels: np.ndarray,
                                          feature_categories: Dict[str, FeatureBankCategory],
                                          max_features: int,
                                          config: Optional[EnhancedFeatureSelectionConfig] = None) -> Dict[str, np.ndarray]:
    """Select features using enhanced selection method."""
    selector = EnhancedFeatureSelector(config)
    return selector.select_optimal_features(features, cluster_labels, feature_categories, max_features)


def generate_feature_selection_report(features: Dict[str, np.ndarray],
                                    cluster_labels: np.ndarray,
                                    feature_categories: Dict[str, FeatureBankCategory],
                                    config: Optional[EnhancedFeatureSelectionConfig] = None) -> Dict[str, Any]:
    """Generate feature selection report."""
    selector = EnhancedFeatureSelector(config)
    return selector.get_feature_selection_report(features, cluster_labels, feature_categories)


__all__ = [
    'EnhancedFeatureSelector',
    'EnhancedFeatureSelectionConfig',
    'select_features_with_enhanced_selection',
    'generate_feature_selection_report'
]