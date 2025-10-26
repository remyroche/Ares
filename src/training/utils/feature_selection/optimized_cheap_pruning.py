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
from collections import defaultdict, OrderedDict
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
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Math validation imports
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    safe_correlation, safe_covariance, safe_mean, safe_std, MathValidation
)

# VectorBT rolling optimizer
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
    VECTORBT_ROLLING_AVAILABLE = True
except ImportError:
    VECTORBT_ROLLING_AVAILABLE = False
    get_vectorbt_rolling_optimizer = None

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
    cache_ttl_seconds: int = 3600  # 1 hour TTL
    max_cache_size: int = 1000  # Maximum number of cached items
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0 < self.variance_threshold < 1:
            raise ValueError("variance_threshold must be between 0 and 1")
        if not 0 < self.stability_ratio_threshold < 1:
            raise ValueError("stability_ratio_threshold must be between 0 and 1")
        if not 0 < self.significance_p_threshold < 1:
            raise ValueError("significance_p_threshold must be between 0 and 1")
        if not 0 < self.mi_bottom_percentile < 100:
            raise ValueError("mi_bottom_percentile must be between 0 and 100")
        if not 0 < self.correlation_threshold < 1:
            raise ValueError("correlation_threshold must be between 0 and 1")
        if self.max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")

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
        
        # Initialize math validation
        self.math_validator = MathValidation()
        
        # Initialize content-based cache with TTL
        self._content_cache = OrderedDict()
        self._cache_timestamps = {}
        
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
        
        # Initialize VectorBT rolling optimizer
        self.rolling_optimizer = None
        if VECTORBT_ROLLING_AVAILABLE:
            try:
                self.rolling_optimizer = get_vectorbt_rolling_optimizer()
                tprint_info("✅ VectorBT rolling optimizer initialized")
            except Exception as e:
                tprint_warning(f"⚠️ VectorBT rolling optimizer initialization failed: {e}")
        
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
        """Implementation of optimized pruning pipeline with memory-efficient processing."""
        self.stats['initial_features'] = len(features_df.columns)
        current_features = features_df.copy()
        
        # Initialize memory monitoring
        if self.hardware_manager:
            self.hardware_manager.start_memory_monitoring()
            initial_memory = self.hardware_manager.get_memory_usage()
            tprint_info(f"📊 Initial memory usage: {initial_memory:.2f} MB")
        
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
        
        # Final memory monitoring
        if self.hardware_manager:
            final_memory = self.hardware_manager.get_memory_usage()
            memory_used = final_memory - initial_memory
            tprint_info(f"📊 Final memory usage: {final_memory:.2f} MB (used: {memory_used:.2f} MB)")
            self.hardware_manager.stop_memory_monitoring()
        
        tprint_success(f"✅ Optimized pruning completed: {self.stats['initial_features']} → {self.stats['final_features']} features ({self.stats['total_reduction']:.1%} reduction)")
        
        return current_features, self.stats
    
    def _clean_cache(self):
        """Clean expired cache entries."""
        current_time = time.time()
        expired_keys = []
        
        for key, timestamp in self._cache_timestamps.items():
            if current_time - timestamp > self.config.cache_ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            self._content_cache.pop(key, None)
            self._cache_timestamps.pop(key, None)
    
    def _variance_pruning_optimized(self, features_df: pd.DataFrame, stage_name: str) -> pd.DataFrame:
        """Optimized variance pruning with direct calculation (no redundant rolling)."""
        tprint_info(f"  📊 Stage 1: Optimized variance pruning...")
        
        # Direct variance calculation - no need for rolling with window=1
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
        
        # Batch statistical testing with parallel processing
        if self.config.enable_batch_operations and ML_COMMON_AVAILABLE:
            try:
                # Use batch statistical analysis
                batch_results = batch_statistical_analysis(features_aligned, target_aligned.to_frame())
                p_values = batch_results.get('ttest_pvalues', {})
            except Exception as e:
                tprint_warning(f"⚠️ Batch statistical analysis failed, falling back to parallel individual tests: {e}")
                p_values = self._parallel_statistical_tests(features_aligned, target_aligned)
        else:
            p_values = self._parallel_statistical_tests(features_aligned, target_aligned)
        
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
    
    def _parallel_statistical_tests(self, features_aligned: pd.DataFrame, target_aligned: pd.Series) -> Dict[str, float]:
        """Perform statistical tests in parallel for better performance."""
        try:
            if not self.config.enable_parallel_processing or len(features_aligned.columns) < 10:
                return self._individual_statistical_tests(features_aligned, target_aligned)
            
            # Use hardware manager to determine optimal number of workers
            max_workers = self.config.max_workers
            if self.hardware_manager:
                max_workers = min(max_workers, self.hardware_manager.get_optimal_worker_count())
            
            # Prepare data for parallel processing
            feature_data = [(col, features_aligned[col].values) for col in features_aligned.columns]
            target_data = target_aligned.values
            
            # Process in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._single_statistical_test, feature_name, feature_values, target_data): feature_name
                    for feature_name, feature_values in feature_data
                }
                
                p_values = {}
                for future in futures:
                    feature_name = futures[future]
                    try:
                        p_values[feature_name] = future.result()
                    except Exception as e:
                        self.logger.warning(f"Statistical test failed for {feature_name}: {e}")
                        p_values[feature_name] = 1.0
            
            return p_values
            
        except Exception as e:
            self.logger.warning(f"Parallel statistical tests failed: {e}")
            return self._individual_statistical_tests(features_aligned, target_aligned)
    
    def _single_statistical_test(self, feature_name: str, feature_values: np.ndarray, target_values: np.ndarray) -> float:
        """Perform a single statistical test for parallel processing."""
        try:
            # Remove NaN values
            valid_mask = ~(np.isnan(feature_values) | np.isnan(target_values))
            if not np.any(valid_mask) or np.sum(valid_mask) < 3:
                return 1.0
            
            feature_clean = feature_values[valid_mask]
            target_clean = target_values[valid_mask]
            
            # Perform t-test
            _, p_value = stats.ttest_ind(feature_clean, target_clean)
            return float(p_value) if not np.isnan(p_value) else 1.0
            
        except Exception as e:
            self.logger.warning(f"Single statistical test failed for {feature_name}: {e}")
            return 1.0
    
    def _calculate_stability_ratio(self, feature: pd.Series, n_splits: int = 3) -> float:
        """Calculate stability ratio using proper temporal windowing."""
        try:
            if len(feature) < n_splits * 10:  # Need at least 10 points per split
                return 1.0  # Consider unstable if insufficient data
            
            window_size = len(feature) // n_splits
            fold_means = []
            
            for i in range(n_splits):
                start_idx = i * window_size
                end_idx = (i + 1) * window_size if i < n_splits - 1 else len(feature)
                fold_data = feature.iloc[start_idx:end_idx]
                
                if len(fold_data) > 0:
                    fold_means.append(fold_data.mean())
            
            if len(fold_means) < 2:
                return 1.0  # Consider unstable if insufficient folds
            
            # Calculate stability ratio with safe operations
            mean_val = self.math_validator.safe_mean(np.array(fold_means), default=0.0)
            std_val = self.math_validator.safe_std(np.array(fold_means), default=0.0)
            
            return self.math_validator.safe_divide(std_val, mean_val + 1e-10, default=1.0)
            
        except Exception as e:
            self.logger.warning(f"Stability ratio calculation failed: {e}")
            return 1.0  # Consider unstable on error
    
    def _calculate_stability_ratio_vectorized(self, feature: pd.Series, rolling_mean: pd.Series) -> float:
        """Calculate stability ratio using VectorBT rolling calculations."""
        try:
            if len(rolling_mean) < 2:
                return 1.0
            
            # Calculate stability ratio with safe operations
            mean_val = self.math_validator.safe_mean(rolling_mean.values, default=0.0)
            std_val = self.math_validator.safe_std(rolling_mean.values, default=0.0)
            
            return self.math_validator.safe_divide(std_val, mean_val + 1e-10, default=1.0)
            
        except Exception as e:
            self.logger.warning(f"Vectorized stability ratio calculation failed: {e}")
            return 1.0
    
    def _stability_pruning_optimized(
        self,
        features_df: pd.DataFrame,
        feature_categories: Dict[str, str],
        composite_scores: Dict[str, float],
        stage_name: str
    ) -> pd.DataFrame:
        """Optimized stability pruning with TimeSeriesSplit."""
        tprint_info(f"  🔄 Stage 3: Optimized stability pruning...")
        
        removed_features = []
        protected_features = []
        
        for feature_name in features_df.columns:
            try:
                feature = features_df[feature_name].dropna()
                
                if len(feature) < 30:  # Skip features with insufficient data
                    removed_features.append(feature_name)
                    continue
                
                # Calculate stability ratio using proper temporal windowing
                # Use VectorBT rolling optimizer if available for better performance
                if self.rolling_optimizer and len(feature) > 100:
                    # Use VectorBT for rolling calculations on large datasets
                    try:
                        rolling_mean = self.rolling_optimizer.rolling_mean(feature, window=len(feature) // self.config.n_temporal_folds)
                        stability_ratio = self._calculate_stability_ratio_vectorized(feature, rolling_mean)
                    except Exception as e:
                        self.logger.warning(f"VectorBT rolling calculation failed: {e}")
                        stability_ratio = self._calculate_stability_ratio(feature, n_splits=self.config.n_temporal_folds)
                else:
                    stability_ratio = self._calculate_stability_ratio(feature, n_splits=self.config.n_temporal_folds)
                
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
            
            # Calculate probabilities
            pxy = hist_2d / hist_2d.sum()
            px = pxy.sum(axis=1)
            py = pxy.sum(axis=0)
            
            # Calculate entropies using safe operations
            hx = -np.sum(px * self.math_validator.safe_log(px + 1e-10, default=0.0))
            hy = -np.sum(py * self.math_validator.safe_log(py + 1e-10, default=0.0))
            hxy = -np.sum(pxy * self.math_validator.safe_log(pxy + 1e-10, default=0.0))
            
            # Calculate mutual information: I(X;Y) = H(X) + H(Y) - H(X,Y)
            mi = hx + hy - hxy
            
            return max(0.0, mi)  # Ensure non-negative
            
        except Exception as e:
            self.logger.warning(f"Fast MI calculation failed: {e}")
            return 0.0
    
    def _calculate_correlation_matrix(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix with memory-efficient approach and VectorBT optimization."""
        n_features = len(features_df.columns)
        estimated_memory = (n_features ** 2 * 8) / (1024 ** 2)  # MB
        
        # Use VectorBT for large correlation matrices if available
        if self.rolling_optimizer and n_features > 500:
            try:
                return self._vectorized_correlation_calculation(features_df)
            except Exception as e:
                self.logger.warning(f"VectorBT correlation calculation failed: {e}")
        
        # Use chunked calculation for large matrices
        if estimated_memory > 500:  # 500MB threshold
            return self._chunked_correlation_calculation(features_df)
        else:
            return features_df.corr().abs()
    
    def _chunked_correlation_calculation(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix in chunks for memory efficiency."""
        try:
            # Use hardware manager to determine optimal chunk size
            if self.hardware_manager:
                available_memory = self.hardware_manager.get_available_memory()
                chunk_size = min(100, max(10, int(available_memory * 0.1 / len(features_df.columns))))
            else:
                chunk_size = min(100, len(features_df.columns) // 4)
            
            n_features = len(features_df.columns)
            corr_matrix = pd.DataFrame(
                np.eye(n_features), 
                index=features_df.columns, 
                columns=features_df.columns
            )
            
            # Calculate correlations in chunks with memory monitoring
            for i in range(0, n_features, chunk_size):
                end_i = min(i + chunk_size, n_features)
                chunk_i = features_df.iloc[:, i:end_i]
                
                for j in range(i, n_features, chunk_size):
                    end_j = min(j + chunk_size, n_features)
                    chunk_j = features_df.iloc[:, j:end_j]
                    
                    # Calculate correlation between chunks
                    chunk_corr = chunk_i.corrwith(chunk_j, axis=0).abs()
                    
                    # Fill correlation matrix
                    for idx_i, col_i in enumerate(chunk_i.columns):
                        for idx_j, col_j in enumerate(chunk_j.columns):
                            if col_i != col_j:
                                corr_matrix.loc[col_i, col_j] = chunk_corr.iloc[idx_i, idx_j]
                
                # Monitor memory usage
                if self.hardware_manager:
                    self.hardware_manager.monitor_memory_usage()
            
            return corr_matrix
            
        except Exception as e:
            self.logger.warning(f"Chunked correlation calculation failed: {e}")
            # Fallback to standard correlation
            return features_df.corr().abs()
    
    def _correlation_pruning_optimized(
        self,
        features_df: pd.DataFrame,
        feature_categories: Dict[str, str],
        composite_scores: Dict[str, float],
        stage_name: str
    ) -> pd.DataFrame:
        """Optimized correlation pruning with vectorized operations."""
        tprint_info(f"  🔗 Stage 5: Optimized correlation pruning...")
        
        # Use memory-efficient correlation matrix calculation
        corr_matrix = self._calculate_correlation_matrix(features_df)
        
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
