"""
Regime Feature Selection Module

This module implements feature selection based on regime discriminative power,
helping to identify the most important features for regime clustering.

Key Features:
- Mutual Information-based feature selection
- Regime-specific feature importance scoring
- Cross-validation for feature stability
- Economic significance testing
- Integration with HDBSCAN clustering
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class FeatureImportanceMetrics:
    """Metrics for feature importance evaluation."""
    mutual_info_score: float
    regime_discriminative_power: float
    economic_significance: float
    clustering_contribution: float
    stability_score: float
    composite_score: float

@dataclass
class RegimeFeatureSelectorConfig:
    """Configuration for regime feature selection."""
    min_mutual_info: float = 0.01
    min_discriminative_power: float = 0.1
    min_economic_significance: float = 0.05
    min_clustering_contribution: float = 0.1
    min_stability_score: float = 0.7
    max_features: int = 20
    cross_validation_folds: int = 5
    random_state: int = 42

class RegimeFeatureSelector:
    """
    Feature selector optimized for regime clustering.
    
    This class implements multiple feature selection strategies specifically
    designed to identify features that best discriminate between market regimes.
    """
    
    def __init__(self, config: Optional[RegimeFeatureSelectorConfig] = None):
        """Initialize the regime feature selector."""
        self.config = config or RegimeFeatureSelectorConfig()
        self.selected_features = []
        self.feature_importance_scores = {}
        self.feature_metrics = {}
        
    def select_features(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray, 
        method: str = 'composite'
    ) -> Tuple[List[str], Dict[str, FeatureImportanceMetrics]]:
        """
        Select features based on regime discriminative power.
        
        Args:
            X: Feature matrix (samples x features)
            y: Regime labels
            method: Selection method ('mutual_info', 'economic', 'clustering', 'composite')
            
        Returns:
            Tuple of (selected_features, feature_metrics)
        """
        logger.info(f"Starting feature selection with method: {method}")
        
        # Calculate feature importance metrics
        self.feature_metrics = self._calculate_feature_metrics(X, y)
        
        # Select features based on method
        if method == 'mutual_info':
            selected_features = self._select_by_mutual_info()
        elif method == 'economic':
            selected_features = self._select_by_economic_significance()
        elif method == 'clustering':
            selected_features = self._select_by_clustering_contribution()
        elif method == 'composite':
            selected_features = self._select_by_composite_score()
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        # Limit to max_features
        if len(selected_features) > self.config.max_features:
            # Sort by composite score and take top features
            feature_scores = [(f, self.feature_metrics[f].composite_score) for f in selected_features]
            feature_scores.sort(key=lambda x: x[1], reverse=True)
            selected_features = [f for f, _ in feature_scores[:self.config.max_features]]
        
        self.selected_features = selected_features
        logger.info(f"Selected {len(selected_features)} features: {selected_features}")
        
        return selected_features, self.feature_metrics
    
    def _calculate_feature_metrics(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, FeatureImportanceMetrics]:
        """Calculate comprehensive feature importance metrics."""
        metrics = {}
        
        for feature in X.columns:
            try:
                # Mutual information score
                mi_score = self._calculate_mutual_info_score(X[feature], y)
                
                # Regime discriminative power
                discriminative_power = self._calculate_discriminative_power(X[feature], y)
                
                # Economic significance
                economic_significance = self._calculate_economic_significance(X[feature], y)
                
                # Clustering contribution
                clustering_contribution = self._calculate_clustering_contribution(X, feature, y)
                
                # Stability score
                stability_score = self._calculate_stability_score(X[feature], y)
                
                # Composite score
                composite_score = self._calculate_composite_score(
                    mi_score, discriminative_power, economic_significance, 
                    clustering_contribution, stability_score
                )
                
                metrics[feature] = FeatureImportanceMetrics(
                    mutual_info_score=mi_score,
                    regime_discriminative_power=discriminative_power,
                    economic_significance=economic_significance,
                    clustering_contribution=clustering_contribution,
                    stability_score=stability_score,
                    composite_score=composite_score
                )
                
            except Exception as e:
                logger.warning(f"Error calculating metrics for feature {feature}: {e}")
                metrics[feature] = FeatureImportanceMetrics(
                    mutual_info_score=0.0,
                    regime_discriminative_power=0.0,
                    economic_significance=0.0,
                    clustering_contribution=0.0,
                    stability_score=0.0,
                    composite_score=0.0
                )
        
        return metrics
    
    def _calculate_mutual_info_score(self, feature: pd.Series, y: np.ndarray) -> float:
        """Calculate mutual information score between feature and regime labels."""
        try:
            # Handle different data types
            if feature.dtype in ['object', 'category']:
                # For categorical features, use mutual_info_classif
                feature_encoded = pd.Categorical(feature).codes
                mi_score = mutual_info_classif(feature_encoded.reshape(-1, 1), y)[0]
            else:
                # For continuous features, use mutual_info_regression
                mi_score = mutual_info_regression(feature.values.reshape(-1, 1), y)[0]
            
            return float(mi_score)
        except Exception as e:
            logger.warning(f"Error calculating mutual info score: {e}")
            return 0.0
    
    def _calculate_discriminative_power(self, feature: pd.Series, y: np.ndarray) -> float:
        """Calculate how well the feature discriminates between regimes."""
        try:
            unique_regimes = np.unique(y)
            if len(unique_regimes) < 2:
                return 0.0
            
            # Calculate between-regime variance vs within-regime variance
            regime_means = []
            regime_vars = []
            
            for regime in unique_regimes:
                if regime == -1:  # Skip noise
                    continue
                regime_mask = y == regime
                regime_data = feature[regime_mask]
                
                if len(regime_data) > 0:
                    regime_means.append(regime_data.mean())
                    regime_vars.append(regime_data.var())
            
            if len(regime_means) < 2:
                return 0.0
            
            # Between-regime variance
            overall_mean = np.mean(regime_means)
            between_var = np.var(regime_means)
            
            # Within-regime variance
            within_var = np.mean(regime_vars)
            
            # Discriminative power (F-ratio)
            if within_var > 0:
                discriminative_power = between_var / within_var
            else:
                discriminative_power = between_var
            
            return min(1.0, discriminative_power)  # Cap at 1.0
            
        except Exception as e:
            logger.warning(f"Error calculating discriminative power: {e}")
            return 0.0
    
    def _calculate_economic_significance(self, feature: pd.Series, y: np.ndarray) -> float:
        """Calculate economic significance of the feature for regime detection."""
        try:
            # Look for price-related features
            price_related_keywords = ['price', 'return', 'volatility', 'volume', 'close', 'high', 'low', 'open']
            is_price_related = any(keyword in feature.name.lower() for keyword in price_related_keywords)
            
            if not is_price_related:
                return 0.1  # Lower significance for non-price features
            
            # Calculate regime-specific economic characteristics
            unique_regimes = np.unique(y)
            if len(unique_regimes) < 2:
                return 0.0
            
            regime_economic_diffs = []
            for i, regime1 in enumerate(unique_regimes):
                if regime1 == -1:  # Skip noise
                    continue
                for regime2 in unique_regimes[i+1:]:
                    if regime2 == -1:  # Skip noise
                        continue
                    
                    regime1_data = feature[y == regime1]
                    regime2_data = feature[y == regime2]
                    
                    if len(regime1_data) > 0 and len(regime2_data) > 0:
                        # Calculate economic difference (mean difference normalized by std)
                        mean_diff = abs(regime1_data.mean() - regime2_data.mean())
                        combined_std = np.sqrt((regime1_data.var() + regime2_data.var()) / 2)
                        
                        if combined_std > 0:
                            economic_diff = mean_diff / combined_std
                            regime_economic_diffs.append(economic_diff)
            
            if regime_economic_diffs:
                return min(1.0, np.mean(regime_economic_diffs))
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Error calculating economic significance: {e}")
            return 0.0
    
    def _calculate_clustering_contribution(self, X: pd.DataFrame, feature: str, y: np.ndarray) -> float:
        """Calculate how much the feature contributes to clustering quality."""
        try:
            # Calculate clustering metrics with and without the feature
            X_with = X[[feature]]
            X_without = X.drop(columns=[feature])
            
            # Calculate silhouette score with feature
            if len(np.unique(y)) > 1:
                sil_with = silhouette_score(X_with, y)
                sil_without = silhouette_score(X_without, y)
                
                # Contribution is the improvement in silhouette score
                contribution = max(0, sil_with - sil_without)
            else:
                contribution = 0.0
            
            return min(1.0, contribution)
            
        except Exception as e:
            logger.warning(f"Error calculating clustering contribution: {e}")
            return 0.0
    
    def _calculate_stability_score(self, feature: pd.Series, y: np.ndarray) -> float:
        """Calculate stability of the feature across different regime subsets."""
        try:
            # Use cross-validation to assess stability
            if len(np.unique(y)) < 2:
                return 0.0
            
            # Create a simple classifier to test feature stability
            clf = RandomForestClassifier(n_estimators=10, random_state=self.config.random_state)
            
            # Cross-validation scores
            cv_scores = cross_val_score(
                clf, 
                feature.values.reshape(-1, 1), 
                y, 
                cv=self.config.cross_validation_folds,
                scoring='accuracy'
            )
            
            # Stability is the consistency of performance across folds
            stability = 1.0 - np.std(cv_scores) if len(cv_scores) > 1 else 0.0
            return max(0.0, min(1.0, stability))
            
        except Exception as e:
            logger.warning(f"Error calculating stability score: {e}")
            return 0.0
    
    def _calculate_composite_score(
        self, 
        mi_score: float, 
        discriminative_power: float, 
        economic_significance: float, 
        clustering_contribution: float, 
        stability_score: float
    ) -> float:
        """Calculate composite score for feature ranking."""
        # Weighted combination of all metrics
        weights = {
            'mutual_info': 0.25,
            'discriminative_power': 0.30,
            'economic_significance': 0.20,
            'clustering_contribution': 0.15,
            'stability': 0.10
        }
        
        composite = (
            weights['mutual_info'] * mi_score +
            weights['discriminative_power'] * discriminative_power +
            weights['economic_significance'] * economic_significance +
            weights['clustering_contribution'] * clustering_contribution +
            weights['stability'] * stability_score
        )
        
        return min(1.0, composite)
    
    def _select_by_mutual_info(self) -> List[str]:
        """Select features based on mutual information score."""
        features = []
        for feature, metrics in self.feature_metrics.items():
            if metrics.mutual_info_score >= self.config.min_mutual_info:
                features.append(feature)
        return features
    
    def _select_by_economic_significance(self) -> List[str]:
        """Select features based on economic significance."""
        features = []
        for feature, metrics in self.feature_metrics.items():
            if metrics.economic_significance >= self.config.min_economic_significance:
                features.append(feature)
        return features
    
    def _select_by_clustering_contribution(self) -> List[str]:
        """Select features based on clustering contribution."""
        features = []
        for feature, metrics in self.feature_metrics.items():
            if metrics.clustering_contribution >= self.config.min_clustering_contribution:
                features.append(feature)
        return features
    
    def _select_by_composite_score(self) -> List[str]:
        """Select features based on composite score."""
        features = []
        for feature, metrics in self.feature_metrics.items():
            if (metrics.mutual_info_score >= self.config.min_mutual_info and
                metrics.discriminative_power >= self.config.min_discriminative_power and
                metrics.economic_significance >= self.config.min_economic_significance and
                metrics.clustering_contribution >= self.config.min_clustering_contribution and
                metrics.stability_score >= self.config.min_stability_score):
                features.append(feature)
        return features
    
    def get_feature_importance_report(self) -> pd.DataFrame:
        """Generate a comprehensive feature importance report."""
        if not self.feature_metrics:
            return pd.DataFrame()
        
        report_data = []
        for feature, metrics in self.feature_metrics.items():
            report_data.append({
                'feature': feature,
                'mutual_info_score': metrics.mutual_info_score,
                'discriminative_power': metrics.regime_discriminative_power,
                'economic_significance': metrics.economic_significance,
                'clustering_contribution': metrics.clustering_contribution,
                'stability_score': metrics.stability_score,
                'composite_score': metrics.composite_score,
                'selected': feature in self.selected_features
            })
        
        report_df = pd.DataFrame(report_data)
        report_df = report_df.sort_values('composite_score', ascending=False)
        
        return report_df
    
    def apply_feature_selection(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature selection to a dataset."""
        if not self.selected_features:
            logger.warning("No features selected. Returning original dataset.")
            return X
        
        # Select only the chosen features
        selected_X = X[self.selected_features].copy()
        logger.info(f"Applied feature selection: {X.shape[1]} -> {selected_X.shape[1]} features")
        
        return selected_X


def create_regime_feature_selector(config: Optional[RegimeFeatureSelectorConfig] = None) -> RegimeFeatureSelector:
    """Factory function to create a RegimeFeatureSelector instance."""
    return RegimeFeatureSelector(config)
