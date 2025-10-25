"""
Cheap Pruning Pipeline for Feature Selection

Implements 5 sequential pruning methods with category balance tracking:
1. Variance Pruning (~5% reduction)
2. Statistical Significance Pruning (~10% reduction) 
3. Stability Pruning (~10-15% reduction)
4. Mutual Information Pruning (~10% reduction)
5. Correlation Pruning (~10-15% reduction)

Maintains minimum 3 features per category for stages 3-5.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import logging
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import KBinsDiscretizer

from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_error, tprint_warning, tprint_performance
from src.utils.logger import system_logger


@dataclass
class PruningConfig:
    """Configuration for pruning pipeline."""
    variance_threshold: float = 1e-6
    stability_ratio_threshold: float = 0.5
    significance_p_threshold: float = 0.1
    mi_bottom_percentile: float = 10.0
    correlation_threshold: float = 0.9
    min_features_per_category: int = 3
    n_temporal_folds: int = 3
    n_mi_bins: int = 10


class CheapPruningPipeline:
    """
    Sequential feature pruning pipeline with category balance protection.
    
    Applies 5 pruning methods in order of computational cost/effectiveness:
    1. Variance (cheapest, no category protection)
    2. Statistical significance (cheap, no category protection)
    3. Stability (medium cost, category protection)
    4. Mutual information (expensive, category protection)
    5. Correlation (most expensive, category protection)
    """
    
    def __init__(self, config: Optional[PruningConfig] = None):
        """Initialize pruning pipeline."""
        self.config = config or PruningConfig()
        self.logger = system_logger.getChild('CheapPruningPipeline')
        
        # Track statistics
        self.stats = {
            'initial_features': 0,
            'final_features': 0,
            'total_reduction': 0.0,
            'stage_results': {},
            'category_distributions': {},
            'protected_features': [],
            'removed_features': []
        }
    
    def prune_features(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        feature_categories: Dict[str, str],
        composite_scores: Dict[str, float]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply sequential pruning pipeline.
        
        Args:
            features_df: DataFrame with features to prune
            targets_df: DataFrame with target variables
            feature_categories: Dict mapping feature_name -> category
            composite_scores: Dict mapping feature_name -> composite_score
            
        Returns:
            Tuple of (pruned_features_df, statistics)
        """
        self.stats['initial_features'] = len(features_df.columns)
        current_features = features_df.copy()
        
        tprint_info(f"🔧 Starting cheap pruning pipeline on {len(current_features.columns)} features")
        
        # Stage 1: Variance Pruning (no category protection)
        current_features = self._variance_pruning(current_features, "variance")
        
        # Stage 2: Statistical Significance Pruning (no category protection)
        current_features = self._statistical_significance_pruning(
            current_features, targets_df, "significance"
        )
        
        # Stage 3: Stability Pruning (with category protection)
        current_features = self._stability_pruning(
            current_features, feature_categories, composite_scores, "stability"
        )
        
        # Stage 4: Mutual Information Pruning (with category protection)
        current_features = self._mutual_information_pruning(
            current_features, targets_df, feature_categories, composite_scores, "mi"
        )
        
        # Stage 5: Correlation Pruning (with category protection)
        current_features = self._correlation_pruning(
            current_features, feature_categories, composite_scores, "correlation"
        )
        
        # Calculate final statistics
        self.stats['final_features'] = len(current_features.columns)
        self.stats['total_reduction'] = (
            (self.stats['initial_features'] - self.stats['final_features']) / 
            self.stats['initial_features']
        )
        
        tprint_success(f"✅ Pruning completed: {self.stats['initial_features']} → {self.stats['final_features']} features ({self.stats['total_reduction']:.1%} reduction)")
        
        return current_features, self.stats
    
    def _variance_pruning(self, features_df: pd.DataFrame, stage_name: str) -> pd.DataFrame:
        """Remove features with very low variance."""
        tprint_info(f"  📊 Stage 1: Variance pruning...")
        
        # Calculate variances
        variances = features_df.var()
        
        # Identify low variance features
        low_var_mask = variances < self.config.variance_threshold
        low_var_features = variances[low_var_mask].index.tolist()
        
        # Remove low variance features
        remaining_features = features_df.drop(columns=low_var_features)
        
        # Track results
        removed_count = len(low_var_features)
        self.stats['stage_results'][stage_name] = {
            'features_removed': removed_count,
            'features_remaining': len(remaining_features.columns),
            'removed_features': low_var_features
        }
        self.stats['removed_features'].extend(low_var_features)
        
        tprint_info(f"    Removed {removed_count} low-variance features")
        
        return remaining_features
    
    def _statistical_significance_pruning(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        stage_name: str
    ) -> pd.DataFrame:
        """Remove features with low statistical significance."""
        tprint_info(f"  📈 Stage 2: Statistical significance pruning...")
        
        # Use first target for significance testing
        target_col = targets_df.columns[0]
        target = targets_df[target_col].dropna()
        
        # Align features and target
        common_index = features_df.index.intersection(target.index)
        features_aligned = features_df.loc[common_index]
        target_aligned = target.loc[common_index]
        
        removed_features = []
        
        for feature_name in features_aligned.columns:
            try:
                feature = features_aligned[feature_name].dropna()
                target_feature = target_aligned.loc[feature.index]
                
                if len(feature) < 10:  # Need minimum samples
                    removed_features.append(feature_name)
                    continue
                
                # Create quantile groups for t-test
                feature_quantiles = pd.qcut(feature, q=2, duplicates='drop')
                if len(feature_quantiles.cat.categories) < 2:
                    removed_features.append(feature_name)
                    continue
                
                # Split into groups
                group1 = target_feature[feature_quantiles == feature_quantiles.cat.categories[0]]
                group2 = target_feature[feature_quantiles == feature_quantiles.cat.categories[1]]
                
                if len(group1) < 3 or len(group2) < 3:
                    removed_features.append(feature_name)
                    continue
                
                # Perform t-test
                _, p_value = stats.ttest_ind(group1, group2)
                
                if p_value > self.config.significance_p_threshold:
                    removed_features.append(feature_name)
                    
            except Exception as e:
                self.logger.warning(f"Significance test failed for {feature_name}: {e}")
                removed_features.append(feature_name)
        
        # Remove non-significant features
        remaining_features = features_df.drop(columns=removed_features)
        
        # Track results
        self.stats['stage_results'][stage_name] = {
            'features_removed': len(removed_features),
            'features_remaining': len(remaining_features.columns),
            'removed_features': removed_features
        }
        self.stats['removed_features'].extend(removed_features)
        
        tprint_info(f"    Removed {len(removed_features)} non-significant features")
        
        return remaining_features
    
    def _stability_pruning(
        self,
        features_df: pd.DataFrame,
        feature_categories: Dict[str, str],
        composite_scores: Dict[str, float],
        stage_name: str
    ) -> pd.DataFrame:
        """Remove unstable features with category protection."""
        tprint_info(f"  🔄 Stage 3: Stability pruning...")
        
        # Split data into temporal folds
        n_samples = len(features_df)
        fold_size = n_samples // self.config.n_temporal_folds
        
        removed_features = []
        protected_features = []
        
        for feature_name in features_df.columns:
            try:
                feature = features_df[feature_name].dropna()
                
                # Calculate fold means
                fold_means = []
                for i in range(self.config.n_temporal_folds):
                    start_idx = i * fold_size
                    end_idx = (i + 1) * fold_size if i < self.config.n_temporal_folds - 1 else n_samples
                    fold_data = feature.iloc[start_idx:end_idx]
                    if len(fold_data) > 0:
                        fold_means.append(fold_data.mean())
                
                if len(fold_means) < 2:
                    removed_features.append(feature_name)
                    continue
                
                # Calculate stability ratio
                fold_means = np.array(fold_means)
                mean_fold_mean = np.mean(fold_means)
                std_fold_means = np.std(fold_means)
                
                if mean_fold_mean == 0:
                    stability_ratio = float('inf')
                else:
                    stability_ratio = std_fold_means / abs(mean_fold_mean)
                
                # Check if feature should be removed
                if stability_ratio > self.config.stability_ratio_threshold:
                    # Check category protection
                    category = feature_categories.get(feature_name, 'unknown')
                    category_count = self._count_category_features(
                        features_df.columns, feature_categories, category
                    )
                    
                    if category_count > self.config.min_features_per_category:
                        removed_features.append(feature_name)
                    else:
                        protected_features.append(feature_name)
                        self.stats['protected_features'].append(feature_name)
                        
            except Exception as e:
                self.logger.warning(f"Stability test failed for {feature_name}: {e}")
                removed_features.append(feature_name)
        
        # Remove unstable features (except protected ones)
        features_to_remove = [f for f in removed_features if f not in protected_features]
        remaining_features = features_df.drop(columns=features_to_remove)
        
        # Track results
        self.stats['stage_results'][stage_name] = {
            'features_removed': len(features_to_remove),
            'features_remaining': len(remaining_features.columns),
            'removed_features': features_to_remove,
            'protected_features': protected_features
        }
        self.stats['removed_features'].extend(features_to_remove)
        
        tprint_info(f"    Removed {len(features_to_remove)} unstable features, protected {len(protected_features)}")
        
        return remaining_features
    
    def _mutual_information_pruning(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        feature_categories: Dict[str, str],
        composite_scores: Dict[str, float],
        stage_name: str
    ) -> pd.DataFrame:
        """Remove features with low mutual information with category protection."""
        tprint_info(f"  🔗 Stage 4: Mutual information pruning...")
        
        # Use first target for MI calculation
        target_col = targets_df.columns[0]
        target = targets_df[target_col].dropna()
        
        # Align features and target
        common_index = features_df.index.intersection(target.index)
        features_aligned = features_df.loc[common_index]
        target_aligned = target.loc[common_index]
        
        # Calculate MI scores
        mi_scores = {}
        for feature_name in features_aligned.columns:
            try:
                feature = features_aligned[feature_name].dropna()
                target_feature = target_aligned.loc[feature.index]
                
                if len(feature) < 10:
                    mi_scores[feature_name] = 0.0
                    continue
                
                # Discretize feature for MI calculation
                discretizer = KBinsDiscretizer(
                    n_bins=self.config.n_mi_bins,
                    encode='ordinal',
                    strategy='quantile'
                )
                
                feature_discretized = discretizer.fit_transform(feature.values.reshape(-1, 1)).flatten()
                
                # Calculate MI
                mi_score = mutual_info_regression(
                    feature_discretized.reshape(-1, 1),
                    target_feature,
                    discrete_features=True
                )[0]
                
                mi_scores[feature_name] = mi_score
                
            except Exception as e:
                self.logger.warning(f"MI calculation failed for {feature_name}: {e}")
                mi_scores[feature_name] = 0.0
        
        # Sort by MI score
        sorted_features = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate threshold (bottom percentile)
        n_features = len(sorted_features)
        threshold_idx = int(n_features * (100 - self.config.mi_bottom_percentile) / 100)
        threshold_score = sorted_features[threshold_idx][1] if threshold_idx < n_features else 0.0
        
        # Identify features to remove
        removed_features = []
        protected_features = []
        
        for feature_name, mi_score in mi_scores.items():
            if mi_score < threshold_score:
                # Check category protection
                category = feature_categories.get(feature_name, 'unknown')
                category_count = self._count_category_features(
                    features_df.columns, feature_categories, category
                )
                
                if category_count > self.config.min_features_per_category:
                    removed_features.append(feature_name)
                else:
                    protected_features.append(feature_name)
                    self.stats['protected_features'].append(feature_name)
        
        # Remove low MI features (except protected ones)
        features_to_remove = [f for f in removed_features if f not in protected_features]
        remaining_features = features_df.drop(columns=features_to_remove)
        
        # Track results
        self.stats['stage_results'][stage_name] = {
            'features_removed': len(features_to_remove),
            'features_remaining': len(remaining_features.columns),
            'removed_features': features_to_remove,
            'protected_features': protected_features,
            'mi_threshold': threshold_score
        }
        self.stats['removed_features'].extend(features_to_remove)
        
        tprint_info(f"    Removed {len(features_to_remove)} low-MI features, protected {len(protected_features)}")
        
        return remaining_features
    
    def _correlation_pruning(
        self,
        features_df: pd.DataFrame,
        feature_categories: Dict[str, str],
        composite_scores: Dict[str, float],
        stage_name: str
    ) -> pd.DataFrame:
        """Remove highly correlated features with category protection."""
        tprint_info(f"  🔗 Stage 5: Correlation pruning...")
        
        # Calculate correlation matrix
        corr_matrix = features_df.corr().abs()
        
        # Find highly correlated pairs
        upper_tri = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        high_corr_pairs = np.where((corr_matrix > self.config.correlation_threshold) & upper_tri)
        
        # Track features to remove
        to_remove = set()
        protected_features = []
        
        for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]):
            feature_i = corr_matrix.columns[i]
            feature_j = corr_matrix.columns[j]
            
            # Keep feature with higher composite score
            score_i = composite_scores.get(feature_i, 0.0)
            score_j = composite_scores.get(feature_j, 0.0)
            
            feature_to_remove = feature_i if score_i < score_j else feature_j
            feature_to_keep = feature_j if score_i < score_j else feature_i
            
            # Check category protection for feature to remove
            category = feature_categories.get(feature_to_remove, 'unknown')
            category_count = self._count_category_features(
                features_df.columns, feature_categories, category
            )
            
            if category_count > self.config.min_features_per_category:
                to_remove.add(feature_to_remove)
            else:
                protected_features.append(feature_to_remove)
                self.stats['protected_features'].append(feature_to_remove)
        
        # Remove correlated features (except protected ones)
        features_to_remove = [f for f in to_remove if f not in protected_features]
        remaining_features = features_df.drop(columns=features_to_remove)
        
        # Track results
        self.stats['stage_results'][stage_name] = {
            'features_removed': len(features_to_remove),
            'features_remaining': len(remaining_features.columns),
            'removed_features': features_to_remove,
            'protected_features': protected_features,
            'correlation_pairs_found': len(high_corr_pairs[0])
        }
        self.stats['removed_features'].extend(features_to_remove)
        
        tprint_info(f"    Removed {len(features_to_remove)} correlated features, protected {len(protected_features)}")
        
        return remaining_features
    
    def _count_category_features(
        self,
        feature_names: List[str],
        feature_categories: Dict[str, str],
        category: str
    ) -> int:
        """Count features in a specific category."""
        return sum(1 for f in feature_names if feature_categories.get(f, 'unknown') == category)
    
    def get_category_distribution(
        self,
        feature_names: List[str],
        feature_categories: Dict[str, str]
    ) -> Dict[str, int]:
        """Get distribution of features by category."""
        distribution = defaultdict(int)
        for feature_name in feature_names:
            category = feature_categories.get(feature_name, 'unknown')
            distribution[category] += 1
        return dict(distribution)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pruning statistics."""
        return self.stats.copy()


def apply_cheap_pruning(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    feature_categories: Dict[str, str],
    composite_scores: Dict[str, float],
    config: Optional[PruningConfig] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply cheap pruning pipeline to features.
    
    Args:
        features_df: DataFrame with features to prune
        targets_df: DataFrame with target variables
        feature_categories: Dict mapping feature_name -> category
        composite_scores: Dict mapping feature_name -> composite_score
        config: Optional pruning configuration
        
    Returns:
        Tuple of (pruned_features_df, statistics)
    """
    pipeline = CheapPruningPipeline(config)
    return pipeline.prune_features(features_df, targets_df, feature_categories, composite_scores)
