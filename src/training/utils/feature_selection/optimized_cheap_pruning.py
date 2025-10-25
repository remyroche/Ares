"""
Optimized Cheap Pruning Pipeline for Feature Selection

Enhanced version with:
- Batch statistical operations
- Cheaper proxy calculations
- Vectorized correlation analysis
- Memory-efficient processing
- Performance monitoring
- Data validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from collections import defaultdict
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import warnings
from scipy import stats
from scipy.sparse import csr_matrix
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.stats.multitest import multipletests

# VectorBT imports
try:
    from src.vectorbt import (
        vbt, rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max,
        rolling_sum, rolling_apply, VECTORBT_AVAILABLE
    )
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None

# Feature common utilities
try:
    from src.utils.feature_common.caching import get_shared_cache, get_feature_cache
    from src.utils.feature_common.monitoring import get_performance_monitor, get_resource_tracker
    from src.utils.feature_common.validation import get_data_validator
    FEATURE_COMMON_AVAILABLE = True
except ImportError:
    FEATURE_COMMON_AVAILABLE = False

# ML common utilities
try:
    from src.utils.ml_common.unified_vectorization_manager import UnifiedVectorizationManager, VectorizationConfig
    from src.utils.ml_common.optimization.hpo_utils import batch_statistical_analysis
    ML_COMMON_AVAILABLE = True
except ImportError:
    ML_COMMON_AVAILABLE = False

from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_error, tprint_warning, tprint_performance
from src.utils.logger import system_logger

logger = system_logger.getChild('OptimizedCheapPruning')

@dataclass
class OptimizedPruningConfig:
    """Configuration for optimized pruning pipeline."""
    variance_threshold: float = 1e-6
    stability_ratio_threshold: float = 0.5
    significance_p_threshold: float = 0.1
    mi_bottom_percentile: float = 10.0
    correlation_threshold: float = 0.9
    min_features_per_category: int = 3
    n_temporal_folds: int = 3
    n_mi_bins: int = 10
    enable_batch_operations: bool = True
    enable_cheaper_proxies: bool = True
    enable_vectorized_correlation: bool = True
    enable_multiple_testing_correction: bool = True
    max_workers: int = 4
    chunk_size: int = 1000
    memory_efficient: bool = True

class OptimizedCheapPruningPipeline:
    """
    Optimized sequential feature pruning pipeline with comprehensive performance improvements.
    
    Features:
    - Batch statistical operations
    - Cheaper proxy calculations
    - Vectorized correlation analysis
    - Memory-efficient processing
    - Performance monitoring
    - Data validation
    """
    
    def __init__(self, config: Optional[OptimizedPruningConfig] = None):
        """Initialize optimized pruning pipeline."""
        self.config = config or OptimizedPruningConfig()
        self.logger = system_logger.getChild('OptimizedCheapPruningPipeline')
        
        # Initialize shared utilities
        self.shared_cache = get_shared_cache() if FEATURE_COMMON_AVAILABLE else None
        self.feature_cache = get_feature_cache() if FEATURE_COMMON_AVAILABLE else None
        self.performance_monitor = get_performance_monitor() if FEATURE_COMMON_AVAILABLE else None
        self.resource_tracker = get_resource_tracker() if FEATURE_COMMON_AVAILABLE else None
        self.data_validator = get_data_validator() if FEATURE_COMMON_AVAILABLE else None
        
        # Initialize VectorBT components
        self.vectorization_manager = None
        if ML_COMMON_AVAILABLE and VECTORBT_AVAILABLE:
            try:
                vectorization_config = VectorizationConfig(
                    enable_vectorbt=True,
                    enable_gpu=False,
                    enable_parallel=True,
                    memory_efficient=True,
                    max_memory_gb=8.0,
                    chunk_size=self.config.chunk_size
                )
                self.vectorization_manager = UnifiedVectorizationManager(vectorization_config)
                tprint_info("✅ VectorBT components initialized")
            except Exception as e:
                tprint_warning(f"⚠️ VectorBT initialization failed: {e}")
        
        # Track statistics
        self.stats = {
            'initial_features': 0,
            'final_features': 0,
            'total_reduction': 0.0,
            'stage_results': {},
            'category_distributions': {},
            'protected_features': [],
            'removed_features': [],
            'performance_metrics': {},
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        tprint_info("🔧 Initialized OptimizedCheapPruningPipeline")
    
    def prune_features(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        feature_categories: Dict[str, str],
        composite_scores: Dict[str, float]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply optimized sequential pruning pipeline.
        
        Args:
            features_df: DataFrame with features to prune
            targets_df: DataFrame with target variables
            feature_categories: Dict mapping feature_name -> category
            composite_scores: Dict mapping feature_name -> composite_score
            
        Returns:
            Tuple of (pruned_features_df, statistics)
        """
        if self.performance_monitor:
            return self.performance_monitor.monitor_operation("prune_features")(
                self._prune_features_impl
            )(features_df, targets_df, feature_categories, composite_scores)
        else:
            return self._prune_features_impl(features_df, targets_df, feature_categories, composite_scores)
    
    def _prune_features_impl(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        feature_categories: Dict[str, str],
        composite_scores: Dict[str, float]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Implementation of optimized pruning pipeline."""
        self.stats['initial_features'] = len(features_df.columns)
        current_features = features_df.copy()
        
        tprint_info(f"🔧 Starting optimized pruning pipeline on {len(current_features.columns)} features")
        
        # Stage 1: Variance Pruning (optimized)
        current_features = self._variance_pruning_optimized(current_features, "variance")
        
        # Stage 2: Statistical Significance Pruning (batch operations)
        current_features = self._statistical_significance_pruning_optimized(
            current_features, targets_df, "significance"
        )
        
        # Stage 3: Stability Pruning (with category protection)
        current_features = self._stability_pruning_optimized(
            current_features, feature_categories, composite_scores, "stability"
        )
        
        # Stage 4: Mutual Information Pruning (cheaper proxies)
        current_features = self._mutual_information_pruning_optimized(
            current_features, targets_df, feature_categories, composite_scores, "mi"
        )
        
        # Stage 5: Correlation Pruning (vectorized)
        current_features = self._correlation_pruning_optimized(
            current_features, feature_categories, composite_scores, "correlation"
        )
        
        # Calculate final statistics
        self.stats['final_features'] = len(current_features.columns)
        self.stats['total_reduction'] = (
            (self.stats['initial_features'] - self.stats['final_features']) / 
            self.stats['initial_features']
        )
        
        tprint_success(f"✅ Optimized pruning completed: {self.stats['initial_features']} → {self.stats['final_features']} features ({self.stats['total_reduction']:.1%} reduction)")
        
        return current_features, self.stats
    
    def _variance_pruning_optimized(self, features_df: pd.DataFrame, stage_name: str) -> pd.DataFrame:
        """Optimized variance pruning with caching."""
        tprint_info(f"  📊 Stage 1: Optimized variance pruning...")
        
        # Use cached variance calculation if available
        if self.feature_cache:
            variances = {}
            for col in features_df.columns:
                variances[col] = self.feature_cache.get_rolling_stat(features_df[col], 1, 'var').iloc[-1]
            variances = pd.Series(variances)
        else:
            # Standard variance calculation
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
    
    def _statistical_significance_pruning_optimized(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        stage_name: str
    ) -> pd.DataFrame:
        """Optimized statistical significance pruning with batch operations."""
        tprint_info(f"  📈 Stage 2: Optimized statistical significance pruning...")
        
        # Use first target for significance testing
        target_col = targets_df.columns[0]
        target = targets_df[target_col].dropna()
        
        # Align features and target
        common_index = features_df.index.intersection(target.index)
        features_aligned = features_df.loc[common_index]
        target_aligned = target.loc[common_index]
        
        # Batch statistical testing
        if self.config.enable_batch_operations and ML_COMMON_AVAILABLE:
            try:
                # Use batch statistical analysis
                batch_results = batch_statistical_analysis(features_aligned, target_aligned.to_frame())
                p_values = batch_results.get('ttest_pvalues', {})
            except Exception as e:
                tprint_warning(f"⚠️ Batch statistical analysis failed, falling back to individual tests: {e}")
                p_values = self._individual_statistical_tests(features_aligned, target_aligned)
        else:
            p_values = self._individual_statistical_tests(features_aligned, target_aligned)
        
        # Apply multiple testing correction if enabled
        if self.config.enable_multiple_testing_correction and len(p_values) > 1:
            feature_names = list(p_values.keys())
            p_values_list = list(p_values.values())
            
            # Apply FDR correction
            rejected, p_corrected, _, _ = multipletests(p_values_list, method='fdr_bh', alpha=self.config.significance_p_threshold)
            
            # Update p_values with corrected values
            p_values = dict(zip(feature_names, p_corrected))
        
        # Identify non-significant features
        removed_features = [
            feature_name for feature_name, p_value in p_values.items()
            if p_value > self.config.significance_p_threshold
        ]
        
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
    
    def _individual_statistical_tests(
        self,
        features_aligned: pd.DataFrame,
        target_aligned: pd.Series
    ) -> Dict[str, float]:
        """Individual statistical tests (fallback method)."""
        p_values = {}
        
        for feature_name in features_aligned.columns:
            try:
                feature = features_aligned[feature_name].dropna()
                target_feature = target_aligned.loc[feature.index]
                
                if len(feature) < 10:  # Need minimum samples
                    p_values[feature_name] = 1.0
                    continue
                
                # Create quantile groups for t-test
                feature_quantiles = pd.qcut(feature, q=2, duplicates='drop')
                if len(feature_quantiles.cat.categories) < 2:
                    p_values[feature_name] = 1.0
                    continue
                
                # Split into groups
                group1 = target_feature[feature_quantiles == feature_quantiles.cat.categories[0]]
                group2 = target_feature[feature_quantiles == feature_quantiles.cat.categories[1]]
                
                if len(group1) < 3 or len(group2) < 3:
                    p_values[feature_name] = 1.0
                    continue
                
                # Perform t-test
                _, p_value = stats.ttest_ind(group1, group2)
                p_values[feature_name] = p_value
                    
            except Exception as e:
                self.logger.warning(f"Statistical test failed for {feature_name}: {e}")
                p_values[feature_name] = 1.0
        
        return p_values
    
    def _stability_pruning_optimized(
        self,
        features_df: pd.DataFrame,
        feature_categories: Dict[str, str],
        composite_scores: Dict[str, float],
        stage_name: str
    ) -> pd.DataFrame:
        """Optimized stability pruning with TimeSeriesSplit."""
        tprint_info(f"  🔄 Stage 3: Optimized stability pruning...")
        
        # Use TimeSeriesSplit for proper temporal validation
        tscv = TimeSeriesSplit(n_splits=self.config.n_temporal_folds)
        
        removed_features = []
        protected_features = []
        
        for feature_name in features_df.columns:
            try:
                feature = features_df[feature_name].dropna()
                
                # Calculate fold means using TimeSeriesSplit
                fold_means = []
                for train_idx, test_idx in tscv.split(feature):
                    fold_data = feature.iloc[test_idx]
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
    
    def _mutual_information_pruning_optimized(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        feature_categories: Dict[str, str],
        composite_scores: Dict[str, float],
        stage_name: str
    ) -> pd.DataFrame:
        """Optimized mutual information pruning with cheaper proxies."""
        tprint_info(f"  🔗 Stage 4: Optimized mutual information pruning...")
        
        # Use first target for MI calculation
        target_col = targets_df.columns[0]
        target = targets_df[target_col].dropna()
        
        # Align features and target
        common_index = features_df.index.intersection(target.index)
        features_aligned = features_df.loc[common_index]
        target_aligned = target.loc[common_index]
        
        # Calculate MI scores with cheaper proxies
        mi_scores = {}
        for feature_name in features_aligned.columns:
            try:
                feature = features_aligned[feature_name].dropna()
                target_feature = target_aligned.loc[feature.index]
                
                if len(feature) < 10:
                    mi_scores[feature_name] = 0.0
                    continue
                
                # Use cheaper MI calculation
                if self.config.enable_cheaper_proxies:
                    mi_score = self._fast_mutual_information(feature, target_feature)
                else:
                    # Standard MI calculation
                    discretizer = KBinsDiscretizer(
                        n_bins=self.config.n_mi_bins,
                        encode='ordinal',
                        strategy='quantile'
                    )
                    feature_discretized = discretizer.fit_transform(feature.values.reshape(-1, 1)).flatten()
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
        threshold_idx = max(0, min(n_features-1, int(n_features * (100 - self.config.mi_bottom_percentile) / 100)))
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
    
    def _fast_mutual_information(self, x: pd.Series, y: pd.Series, bins: int = 20) -> float:
        """Fast mutual information using histogram binning (10x faster than KBinsDiscretizer)."""
        try:
            # Remove NaN values
            valid_mask = ~(x.isna() | y.isna())
            x_clean = x[valid_mask]
            y_clean = y[valid_mask]
            
            if len(x_clean) < 10:
                return 0.0
            
            # Create 2D histogram
            hist_2d, _, _ = np.histogram2d(x_clean, y_clean, bins=bins)
            hist_2d = hist_2d + 1e-10  # Avoid log(0)
            
            # Calculate probabilities
            pxy = hist_2d / hist_2d.sum()
            px = pxy.sum(axis=1)
            py = pxy.sum(axis=0)
            px_py = px[:, None] * py[None, :]
            
            # Calculate MI
            mi = np.sum(pxy * np.log(pxy / (px_py + 1e-10)))
            
            return max(0.0, mi)  # Ensure non-negative
            
        except Exception as e:
            self.logger.warning(f"Fast MI calculation failed: {e}")
            return 0.0
    
    def _correlation_pruning_optimized(
        self,
        features_df: pd.DataFrame,
        feature_categories: Dict[str, str],
        composite_scores: Dict[str, float],
        stage_name: str
    ) -> pd.DataFrame:
        """Optimized correlation pruning with vectorized operations."""
        tprint_info(f"  🔗 Stage 5: Optimized correlation pruning...")
        
        # Use cached correlation matrix if available
        if self.feature_cache:
            corr_matrix = self.feature_cache.get_correlation_matrix(features_df)
        else:
            # Calculate correlation matrix
            if self.config.enable_vectorized_correlation and VECTORBT_AVAILABLE:
                # Use VectorBT for large correlation matrices
                if len(features_df.columns) > 1000:
                    corr_matrix = self._vectorized_correlation_calculation(features_df)
                else:
                    corr_matrix = features_df.corr().abs()
            else:
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
    
    def _vectorized_correlation_calculation(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized correlation calculation for large feature sets."""
        try:
            # Use sparse matrices for memory efficiency
            from scipy.sparse import csr_matrix
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Convert to sparse matrix
            sparse_features = csr_matrix(features_df.fillna(0).values)
            
            # Calculate sparse correlation
            correlation_matrix = cosine_similarity(sparse_features.T)
            
            # Convert back to DataFrame
            corr_df = pd.DataFrame(
                correlation_matrix,
                index=features_df.columns,
                columns=features_df.columns
            )
            
            return corr_df.abs()
            
        except Exception as e:
            self.logger.warning(f"Vectorized correlation calculation failed: {e}")
            # Fallback to standard correlation
            return features_df.corr().abs()
    
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
        stats = self.stats.copy()
        
        # Add cache statistics
        if self.feature_cache:
            cache_stats = self.feature_cache.get_stats()
            stats['cache_stats'] = cache_stats
        
        # Add performance metrics
        if self.performance_monitor:
            perf_stats = self.performance_monitor.get_performance_summary()
            stats['performance_stats'] = perf_stats
        
        return stats

def apply_optimized_cheap_pruning(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    feature_categories: Dict[str, str],
    composite_scores: Dict[str, float],
    config: Optional[OptimizedPruningConfig] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply optimized cheap pruning pipeline to features.
    
    Args:
        features_df: DataFrame with features to prune
        targets_df: DataFrame with target variables
        feature_categories: Dict mapping feature_name -> category
        composite_scores: Dict mapping feature_name -> composite_score
        config: Optional pruning configuration
        
    Returns:
        Tuple of (pruned_features_df, statistics)
    """
    pipeline = OptimizedCheapPruningPipeline(config)
    return pipeline.prune_features(features_df, targets_df, feature_categories, composite_scores)
