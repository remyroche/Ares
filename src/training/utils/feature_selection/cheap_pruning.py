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

# Hardware manager
try:
    from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager
    HARDWARE_MANAGER_AVAILABLE = True
except ImportError:
    HARDWARE_MANAGER_AVAILABLE = False

from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_error, tprint_warning, tprint_performance
from src.utils.logger import system_logger

logger = system_logger.getChild('OptimizedCheapPruning')

@dataclass
class OptimizedPruningConfig:
    """Configuration for optimized pruning pipeline."""
    # Percentile-based thresholds (reduced to 5% for variance/statistical/stability)
    variance_bottom_percentile: float = 5.0
    stability_bottom_percentile: float = 5.0
    significance_bottom_percentile: float = 5.0
    mi_bottom_percentile: float = 10.0  # Will be disabled
    correlation_threshold: float = 0.8  # Increased to 20% (1.0 - 0.8 = 0.2)
    min_features_per_category: int = 3
    
    # Legacy thresholds for backward compatibility (deprecated)
    variance_threshold: float = 1e-6
    stability_ratio_threshold: float = 0.5
    significance_p_threshold: float = 0.1
    n_temporal_folds: int = 3
    n_mi_bins: int = 10
    enable_batch_operations: bool = True
    enable_cheaper_proxies: bool = True
    enable_vectorized_correlation: bool = True
    enable_multiple_testing_correction: bool = True
    enable_parallel_processing: bool = True
    max_workers: int = 4
    chunk_size: int = 1000
    memory_efficient: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour TTL
    max_cache_size: int = 1000  # Maximum number of cached items
    
    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate percentile-based thresholds
        if not 0 < self.variance_bottom_percentile < 100:
            raise ValueError("variance_bottom_percentile must be between 0 and 100")
        if not 0 < self.stability_bottom_percentile < 100:
            raise ValueError("stability_bottom_percentile must be between 0 and 100")
        if not 0 < self.significance_bottom_percentile < 100:
            raise ValueError("significance_bottom_percentile must be between 0 and 100")
        if not 0 < self.mi_bottom_percentile < 100:
            raise ValueError("mi_bottom_percentile must be between 0 and 100")
        
        # Legacy validation (deprecated but kept for backward compatibility)
        if not 0 < self.variance_threshold < 1:
            raise ValueError("variance_threshold must be between 0 and 1")
        if not 0 < self.stability_ratio_threshold < 1:
            raise ValueError("stability_ratio_threshold must be between 0 and 1")
        if not 0 < self.significance_p_threshold < 1:
            raise ValueError("significance_p_threshold must be between 0 and 1")
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
        
        # Initialize hardware manager
        self.hardware_manager = None
        if HARDWARE_MANAGER_AVAILABLE:
            try:
                self.hardware_manager = UnifiedHardwareManager()
                tprint_info("✅ Hardware manager initialized")
            except Exception as e:
                tprint_warning(f"⚠️ Hardware manager initialization failed: {e}")
        
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
        # Temporarily disable monitoring to debug the Series issue
        # if self.performance_monitor:
        #     return self.performance_monitor.monitor_operation("prune_features")(
        #         self._prune_features_impl
        #     )(features_df, targets_df, feature_categories, composite_scores)
        # else:
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
            try:
                # Try the expected method first
                if hasattr(self.hardware_manager, 'start_memory_monitoring'):
                    self.hardware_manager.start_memory_monitoring()
                elif hasattr(self.hardware_manager, 'start_monitoring'):
                    self.hardware_manager.start_monitoring()
                
                # Get memory usage
                if hasattr(self.hardware_manager, 'get_memory_usage'):
                    initial_memory = self.hardware_manager.get_memory_usage()
                    if isinstance(initial_memory, dict):
                        initial_memory = initial_memory.get('used_mb', 0)
                    tprint_info(f"📊 Initial memory usage: {initial_memory:.2f} MB")
                else:
                    tprint_info("📊 Memory monitoring not available")
            except Exception as e:
                tprint_warning(f"⚠️ Memory monitoring initialization failed: {e}")
        
        tprint_info(f"🔧 Starting correlation clustering pruning pipeline on {len(current_features.columns)} features")
        
        # Primary Method: Correlation Clustering Pruning (keep ~70% of features instead of 50%)
        tprint_info(f"🔍 Starting correlation clustering pruning with {len(current_features.columns)} features")
        current_features = self._correlation_clustering_pruning_primary(
            current_features, feature_categories, composite_scores, targets_df, "correlation_clustering"
        )
        tprint_info(f"🔍 After correlation clustering pruning: {len(current_features.columns)} features remaining")
        
        # Optional: Light variance pruning only if we still have too many features
        if len(current_features.columns) > len(features_df.columns) * 0.9:  # If we kept more than 90%
            tprint_info(f"🔍 Applying light variance pruning to reach ~75% retention")
            current_features = self._light_variance_pruning(
                current_features, feature_categories, composite_scores, "light_variance"
            )
            tprint_info(f"🔍 After light variance pruning: {len(current_features.columns)} features remaining")
        
        # Calculate final statistics
        self.stats['final_features'] = len(current_features.columns)
        self.stats['total_reduction'] = (
            (self.stats['initial_features'] - self.stats['final_features']) / 
            self.stats['initial_features']
        )
        
        # Print final summary
        initial_count = self.stats['initial_features']
        final_count = self.stats['final_features']
        total_removed = initial_count - final_count
        reduction_percent = self.stats['total_reduction'] * 100
        
        tprint_info("🎯 CORRELATION CLUSTERING PRUNING SUMMARY:")
        tprint_info(f"  📊 Initial features: {initial_count}")
        tprint_info(f"  📊 Final features: {final_count}")
        tprint_info(f"  📊 Total removed: {total_removed} ({reduction_percent:.1f}%)")
        tprint_info(f"  📊 Retention ratio: {(final_count/initial_count):.1%}")
        
        if final_count < initial_count * 0.6:
            tprint_info(f"  ⚠️ WARNING: Only {(final_count/initial_count):.1%} features retained. Consider adjusting pruning thresholds.")
        elif final_count >= initial_count * 0.6:
            tprint_info(f"  ✅ SUCCESS: {(final_count/initial_count):.1%} features retained - good for interaction generation")
        
        if final_count == 0:
            tprint_warning("⚠️ WARNING: All features were removed! This may cause downstream issues.")
        elif final_count < 5:
            tprint_warning(f"⚠️ WARNING: Only {final_count} features remaining. Consider adjusting pruning thresholds.")
        else:
            tprint_success(f"✅ Pruning completed successfully with {final_count} features remaining")
        
        # Final memory monitoring
        if self.hardware_manager:
            try:
                if hasattr(self.hardware_manager, 'get_memory_usage'):
                    final_memory = self.hardware_manager.get_memory_usage()
                    if isinstance(final_memory, dict):
                        final_memory = final_memory.get('used_mb', 0)
                    memory_used = final_memory - initial_memory if 'initial_memory' in locals() else 0
                    tprint_info(f"📊 Final memory usage: {final_memory:.2f} MB (used: {memory_used:.2f} MB)")
                
                # Stop monitoring
                if hasattr(self.hardware_manager, 'stop_memory_monitoring'):
                    self.hardware_manager.stop_memory_monitoring()
                elif hasattr(self.hardware_manager, 'stop_monitoring'):
                    self.hardware_manager.stop_monitoring()
            except Exception as e:
                tprint_warning(f"⚠️ Final memory monitoring failed: {e}")
        
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
        """Optimized variance pruning with percentile-based thresholds."""
        initial_count = len(features_df.columns)
        tprint_info(f"  📊 Stage 1: Optimized variance pruning... (starting with {initial_count} features)")
        
        # Direct variance calculation - no need for rolling with window=1
        variances = features_df.var()
        
        # Calculate percentile-based threshold
        variance_threshold = np.percentile(variances, self.config.variance_bottom_percentile)
        
        # Identify low variance features
        low_var_mask = variances < variance_threshold
        low_var_features = variances[low_var_mask].index.tolist()
        
        # Remove low variance features
        remaining_features = features_df.drop(columns=low_var_features)
        
        # Track results
        removed_count = len(low_var_features)
        remaining_count = len(remaining_features.columns)
        
        if removed_count > 0:
            tprint_info(f"    ✅ Removed {removed_count} low-variance features (bottom {self.config.variance_bottom_percentile}%, threshold: {variance_threshold:.6f})")
            tprint_info(f"    📊 Remaining features: {remaining_count}")
        else:
            tprint_info(f"    ✅ No low-variance features found (bottom {self.config.variance_bottom_percentile}%, threshold: {variance_threshold:.6f})")
            tprint_info(f"    📊 Remaining features: {initial_count}")
        
        self.stats['stage_results'][stage_name] = {
            'features_removed': removed_count,
            'features_remaining': remaining_count,
            'removed_features': low_var_features
        }
        self.stats['removed_features'].extend(low_var_features)
        
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
            
            # Apply FDR correction with percentile-based threshold
            significance_threshold = np.percentile(p_values_list, self.config.significance_bottom_percentile)
            rejected, p_corrected, _, _ = multipletests(p_values_list, method='fdr_bh', alpha=significance_threshold)
            
            # Update p_values with corrected values
            p_values = dict(zip(feature_names, p_corrected))
            threshold_used = significance_threshold
        else:
            # Use percentile-based threshold without correction
            p_values_list = list(p_values.values())
            threshold_used = np.percentile(p_values_list, self.config.significance_bottom_percentile)
        
        # Identify non-significant features using percentile-based threshold
        removed_features = [
            feature_name for feature_name, p_value in p_values.items()
            if p_value > threshold_used
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
        
        tprint_info(f"    ✅ Removed {len(removed_features)} non-significant features (bottom {self.config.significance_bottom_percentile}%, threshold: {threshold_used:.6f})")
        
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
                if hasattr(self.hardware_manager, 'get_optimal_worker_count'):
                    max_workers = min(max_workers, self.hardware_manager.get_optimal_worker_count())
                elif hasattr(self.hardware_manager, 'get_system_status'):
                    system_status = self.hardware_manager.get_system_status()
                    cpu_cores = system_status.get('cpu_cores', 4)
                    max_workers = min(max_workers, cpu_cores)
            
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
        """Optimized stability pruning with percentile-based thresholds."""
        tprint_info(f"  🔄 Stage 3: Optimized stability pruning...")
        
        # First pass: calculate stability ratios for all features
        stability_ratios = {}
        removed_features = []
        
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
                
                stability_ratios[feature_name] = stability_ratio
                        
            except Exception as e:
                self.logger.warning(f"Stability test failed for {feature_name}: {e}")
                removed_features.append(feature_name)
        
        # Calculate percentile-based threshold
        if stability_ratios:
            stability_threshold = np.percentile(list(stability_ratios.values()), self.config.stability_bottom_percentile)
            
            # Second pass: apply percentile-based removal with category protection
            protected_features = []
            for feature_name, stability_ratio in stability_ratios.items():
                if stability_ratio > stability_threshold:
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
        
        tprint_info(f"    ✅ Removed {len(features_to_remove)} unstable features (bottom {self.config.stability_bottom_percentile}%, threshold: {stability_threshold:.6f}), protected {len(protected_features)}")
        
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
        initial_count = len(features_df.columns)
        tprint_info(f"  🔗 Stage 4: Optimized mutual information pruning... (starting with {initial_count} features)")
        
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
        removed_count = len(features_to_remove)
        remaining_count = len(remaining_features.columns)
        
        if removed_count > 0:
            tprint_info(f"    ✅ Removed {removed_count} low-MI features (bottom {self.config.mi_bottom_percentile}%, threshold: {threshold_score:.4f})")
            tprint_info(f"    📊 Remaining features: {remaining_count}")
        else:
            tprint_info(f"    ✅ No low-MI features found (bottom {self.config.mi_bottom_percentile}%, threshold: {threshold_score:.4f})")
            tprint_info(f"    📊 Remaining features: {initial_count}")
        
        if len(protected_features) > 0:
            tprint_info(f"    🛡️ Protected {len(protected_features)} features from category constraints")
        
        self.stats['stage_results'][stage_name] = {
            'features_removed': removed_count,
            'features_remaining': remaining_count,
            'removed_features': features_to_remove,
            'protected_features': protected_features,
            'mi_threshold': threshold_score
        }
        self.stats['removed_features'].extend(features_to_remove)
        
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
                if hasattr(self.hardware_manager, 'get_available_memory'):
                    available_memory = self.hardware_manager.get_available_memory()
                    chunk_size = min(100, max(10, int(available_memory * 0.1 / len(features_df.columns))))
                elif hasattr(self.hardware_manager, 'get_memory_usage'):
                    memory_info = self.hardware_manager.get_memory_usage()
                    if isinstance(memory_info, dict):
                        available_memory = memory_info.get('available_mb', 1000)
                        chunk_size = min(100, max(10, int(available_memory * 0.1 / len(features_df.columns))))
                    else:
                        chunk_size = min(100, len(features_df.columns) // 4)
                else:
                    chunk_size = min(100, len(features_df.columns) // 4)
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
                    if hasattr(self.hardware_manager, 'monitor_memory_usage'):
                        self.hardware_manager.monitor_memory_usage()
                    elif hasattr(self.hardware_manager, 'get_memory_usage'):
                        # Just check memory usage without monitoring
                        pass
            
            return corr_matrix
            
        except Exception as e:
            self.logger.warning(f"Chunked correlation calculation failed: {e}")
            # Fallback to standard correlation
            return features_df.corr().abs()
    
    def _create_correlation_clusters(self, corr_matrix: pd.DataFrame, threshold: float) -> Dict[int, List[str]]:
        """Create correlation clusters using hierarchical clustering with NaN/infinite value handling."""
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        
        try:
            # Convert correlation matrix to distance matrix (1 - |correlation|)
            distance_matrix = 1 - corr_matrix.abs()
            
            # Handle infinite and NaN values
            distance_matrix = distance_matrix.replace([np.inf, -np.inf], np.nan)
            distance_matrix = distance_matrix.fillna(1.0)  # Max distance for NaN correlations
            
            # Ensure all values are finite
            if not np.all(np.isfinite(distance_matrix.values)):
                tprint_warning("⚠️ Non-finite values detected in distance matrix, using fallback clustering")
                return self._fallback_correlation_clustering(corr_matrix, threshold)
            
            # Convert to condensed distance matrix for scipy
            condensed_distances = squareform(distance_matrix.values, checks=False)
            
            # Check for non-finite values in condensed matrix
            if not np.all(np.isfinite(condensed_distances)):
                tprint_warning("⚠️ Non-finite values detected in condensed distances, using fallback clustering")
                return self._fallback_correlation_clustering(corr_matrix, threshold)
            
            # Perform hierarchical clustering
            linkage_matrix = linkage(condensed_distances, method='ward')
            
            # Create clusters with distance threshold
            # Convert correlation threshold to distance threshold
            distance_threshold = 1 - threshold
            cluster_labels = fcluster(linkage_matrix, distance_threshold, criterion='distance')
            
            # Group features by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                feature_name = corr_matrix.columns[i]
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(feature_name)
            
            # Filter clusters to only include those with 3-5 members
            filtered_clusters = {}
            for cluster_id, features in clusters.items():
                if 3 <= len(features) <= 5:
                    filtered_clusters[cluster_id] = features
            
            return filtered_clusters
            
        except Exception as e:
            tprint_warning(f"⚠️ Hierarchical clustering failed: {e}, using fallback clustering")
            return self._fallback_correlation_clustering(corr_matrix, threshold)
    
    def _fallback_correlation_clustering(self, corr_matrix: pd.DataFrame, threshold: float) -> Dict[int, List[str]]:
        """Fallback correlation clustering using simple threshold-based grouping."""
        clusters = {}
        cluster_id = 0
        processed_features = set()
        
        for feature in corr_matrix.columns:
            if feature in processed_features:
                continue
                
            # Find features highly correlated with this one
            correlations = corr_matrix[feature].abs()
            highly_correlated = correlations[correlations > threshold].index.tolist()
            
            # Remove self-correlation and already processed features
            highly_correlated = [f for f in highly_correlated if f != feature and f not in processed_features]
            
            if len(highly_correlated) >= 2:  # Need at least 3 total features (including current)
                cluster_features = [feature] + highly_correlated
                if 3 <= len(cluster_features) <= 5:  # Keep clusters of 3-5 features
                    clusters[cluster_id] = cluster_features
                    processed_features.update(cluster_features)
                    cluster_id += 1
        
        return clusters
    
    def _remove_problematic_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Remove features with zero variance, constant values, or infinite values.
        
        More lenient for cross-timeframe features which may have some NaN values.
        """
        features_to_keep = []
        cross_timeframe_features = []
        problematic_cross_timeframe = []
        
        for column in features_df.columns:
            series = features_df[column]
            # Ensure we have a Series, not a DataFrame
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0] if len(series.columns) > 0 else pd.Series(dtype='float64')
            
            # Check if this is a cross-timeframe feature
            is_cross_timeframe = ('cross_timeframe' in column.lower() or 
                                '_ratio_' in column or 
                                '_3x_ratio' in column or 
                                '_9x_ratio' in column or 
                                '_27x_ratio' in column)
            
            if is_cross_timeframe:
                cross_timeframe_features.append(column)
            
            # Check for all NaN values (always remove)
            if series.isna().all():
                if is_cross_timeframe:
                    problematic_cross_timeframe.append(f"{column} (all NaN)")
                continue
            
            # Check for constant values (always remove)
            if series.nunique() <= 1:
                if is_cross_timeframe:
                    problematic_cross_timeframe.append(f"{column} (constant)")
                continue
            
            # Check for zero variance (always remove)
            if series.var() == 0:
                if is_cross_timeframe:
                    problematic_cross_timeframe.append(f"{column} (zero variance)")
                continue
            
            # Check for infinite values (always remove)
            if np.isinf(series).any():
                if is_cross_timeframe:
                    problematic_cross_timeframe.append(f"{column} (infinite values)")
                continue
            
            # For cross-timeframe features, be more lenient with NaN values
            if is_cross_timeframe:
                # Allow up to 20% NaN values for cross-timeframe features
                nan_ratio = series.isna().sum() / len(series)
                if nan_ratio > 0.2:
                    problematic_cross_timeframe.append(f"{column} (too many NaN: {nan_ratio:.1%})")
                    continue
            else:
                # For regular features, be stricter with NaN values
                nan_ratio = series.isna().sum() / len(series)
                if nan_ratio > 0.1:  # Only allow 10% NaN for regular features
                    continue
            
            features_to_keep.append(column)
        
        # Log cross-timeframe feature analysis
        tprint_info(f"🔍 Cross-timeframe feature analysis:")
        tprint_info(f"  📊 Total cross-timeframe features found: {len(cross_timeframe_features)}")
        tprint_info(f"  📊 Cross-timeframe features kept: {len([f for f in features_to_keep if f in cross_timeframe_features])}")
        tprint_info(f"  📊 Cross-timeframe features removed: {len(problematic_cross_timeframe)}")
        
        if problematic_cross_timeframe:
            tprint_info(f"  ❌ Problematic cross-timeframe features:")
            for feature in problematic_cross_timeframe[:10]:  # Show first 10
                tprint_info(f"    - {feature}")
            if len(problematic_cross_timeframe) > 10:
                tprint_info(f"    ... and {len(problematic_cross_timeframe) - 10} more")
        
        return features_df[features_to_keep]
    
    def _correlation_clustering_pruning_primary(
        self,
        features_df: pd.DataFrame,
        feature_categories: Dict[str, str],
        composite_scores: Dict[str, float],
        targets_df: pd.DataFrame,
        stage_name: str
    ) -> pd.DataFrame:
        """Primary correlation clustering pruning method - keeps ~75% of features."""
        initial_count = len(features_df.columns)
        target_retention = 0.15  # Keep 15% of features for interaction generation (100-150 from 876)
        target_count = max(100, int(initial_count * target_retention))  # At least 100 features
        
        # DETAILED CROSS-TIMEFRAME TRACKING - BEFORE CLUSTERING
        cross_timeframe_before = [f for f in features_df.columns if 
                                 ('cross_timeframe' in f.lower() or 
                                  '_3x_ratio' in f or 
                                  '_6x_ratio' in f or 
                                  '_9x_ratio' in f or 
                                  '_27x_ratio' in f)]
        
        tprint_info("="*80)
        tprint_info("📊 CROSS-TIMEFRAME FEATURE TRACKING - PRUNING ENTRY POINT")
        tprint_info("="*80)
        tprint_info(f"  📈 Total features at entry: {initial_count}")
        tprint_info(f"  📈 Cross-timeframe features at entry: {len(cross_timeframe_before)}")
        tprint_info(f"  📈 Cross-timeframe ratio: {len(cross_timeframe_before)/initial_count:.1%}")
        if cross_timeframe_before:
            tprint_info(f"  📈 Sample cross-timeframe features:")
            for f in cross_timeframe_before[:5]:
                score = composite_scores.get(f, 0.0)
                tprint_info(f"      - {f} (score: {score:.4f})")
        
        # Analyze cross-timeframe feature scores
        if cross_timeframe_before:
            ct_scores = [composite_scores.get(f, 0.0) for f in cross_timeframe_before]
            all_scores = [composite_scores.get(f, 0.0) for f in features_df.columns]
            tprint_info(f"  📊 Cross-timeframe score statistics:")
            tprint_info(f"      Min: {min(ct_scores):.4f}, Max: {max(ct_scores):.4f}, Mean: {np.mean(ct_scores):.4f}")
            tprint_info(f"  📊 All features score statistics:")
            tprint_info(f"      Min: {min(all_scores):.4f}, Max: {max(all_scores):.4f}, Mean: {np.mean(all_scores):.4f}")
            
            # Check if cross-timeframe features have lower scores
            ct_mean = np.mean(ct_scores)
            all_mean = np.mean(all_scores)
            if ct_mean < all_mean * 0.8:
                tprint_warning(f"  ⚠️ Cross-timeframe features have significantly lower scores!")
                tprint_warning(f"      CT mean: {ct_mean:.4f} vs All mean: {all_mean:.4f}")
                tprint_warning(f"      This may cause aggressive pruning!")
        
        tprint_info(f"🎯 Target: Keep {target_count} features ({target_retention:.0%} retention)")
        
        # Pre-process features to remove zero variance and constant features
        features_df_clean = self._remove_problematic_features(features_df)
        
        # Track cross-timeframe after problematic removal
        cross_timeframe_after_clean = [f for f in features_df_clean.columns if 
                                      ('cross_timeframe' in f.lower() or 
                                       '_3x_ratio' in f or 
                                       '_6x_ratio' in f or 
                                       '_9x_ratio' in f or 
                                       '_27x_ratio' in f)]
        
        if len(features_df_clean.columns) != len(features_df.columns):
            removed_problematic = len(features_df.columns) - len(features_df_clean.columns)
            ct_removed_problematic = len(cross_timeframe_before) - len(cross_timeframe_after_clean)
            tprint_info(f"  🧹 Removed {removed_problematic} problematic features before clustering")
            tprint_info(f"  🧹 Cross-timeframe features removed: {ct_removed_problematic}")
            if ct_removed_problematic > 0:
                tprint_warning(f"  ⚠️ {ct_removed_problematic} cross-timeframe features were problematic (zero variance/constant)")
                ct_removed = set(cross_timeframe_before) - set(cross_timeframe_after_clean)
                for f in list(ct_removed)[:5]:
                    tprint_warning(f"      - {f}")
        
        # Use memory-efficient correlation matrix calculation
        corr_matrix = self._calculate_correlation_matrix(features_df_clean)
        
        # Create correlation clusters using hierarchical clustering
        clusters = self._create_correlation_clusters(corr_matrix, self.config.correlation_threshold)
        
        # Track features to remove
        features_to_remove = []
        protected_features = []
        
        # Track cross-timeframe features through clustering
        cross_timeframe_in_clusters = []
        cross_timeframe_removed_from_clusters = []
        
        for cluster_id, cluster_features in clusters.items():
            if len(cluster_features) < 3:  # Skip small clusters
                continue
                
            # Check for cross-timeframe features in this cluster
            cluster_cross_timeframe = [f for f in cluster_features if 
                                     ('cross_timeframe' in f.lower() or 
                                      '_ratio_' in f or 
                                      '_3x_ratio' in f or 
                                      '_9x_ratio' in f or 
                                      '_27x_ratio' in f)]
            
            if cluster_cross_timeframe:
                cross_timeframe_in_clusters.extend(cluster_cross_timeframe)
                tprint_info(f"  🔍 Processing cluster {cluster_id}: {len(cluster_features)} features (including {len(cluster_cross_timeframe)} cross-timeframe)")
            else:
                tprint_info(f"  🔍 Processing cluster {cluster_id}: {len(cluster_features)} features")
            
            # Sort features by composite score (MI + stability) - lowest first
            cluster_scores = [(f, composite_scores.get(f, 0.0)) for f in cluster_features]
            cluster_scores.sort(key=lambda x: x[1])  # Sort by score (lowest first)
            
            # Remove bottom 5% from this cluster (very conservative)
            remove_count = max(1, len(cluster_features) // 20)  # At most 5% per cluster
            features_to_remove_from_cluster = [f for f, _ in cluster_scores[:remove_count]]
            
            # Track cross-timeframe features being removed from this cluster
            cross_timeframe_removed_from_cluster = [f for f in features_to_remove_from_cluster if 
                                                   ('cross_timeframe' in f.lower() or 
                                                    '_ratio_' in f or 
                                                    '_3x_ratio' in f or 
                                                    '_9x_ratio' in f or 
                                                    '_27x_ratio' in f)]
            
            if cross_timeframe_removed_from_cluster:
                cross_timeframe_removed_from_clusters.extend(cross_timeframe_removed_from_cluster)
                tprint_info(f"    ❌ Removing {len(cross_timeframe_removed_from_cluster)} cross-timeframe features from cluster {cluster_id}")
                for f in cross_timeframe_removed_from_cluster:
                    tprint_info(f"      - {f}")
            
            # Check category protection for each feature to remove
            for feature in features_to_remove_from_cluster:
                category = feature_categories.get(feature, 'unknown')
                category_count = self._count_category_features(
                    features_df.columns, feature_categories, category
                )
                
                if category_count > self.config.min_features_per_category:
                    features_to_remove.append(feature)
                else:
                    protected_features.append(feature)
                    self.stats['protected_features'].append(feature)
            
            tprint_info(f"  ✅ Cluster {cluster_id}: removing {len(features_to_remove_from_cluster)} features, keeping {len(cluster_features) - len(features_to_remove_from_cluster)}")
        
        # Log cross-timeframe feature clustering summary
        tprint_info(f"🔍 Cross-timeframe clustering summary:")
        tprint_info(f"  📊 Cross-timeframe features found in clusters: {len(cross_timeframe_in_clusters)}")
        tprint_info(f"  📊 Cross-timeframe features removed from clusters: {len(cross_timeframe_removed_from_clusters)}")
        tprint_info(f"  📊 Cross-timeframe features surviving clustering: {len(cross_timeframe_in_clusters) - len(cross_timeframe_removed_from_clusters)}")
        
        # Remove clustered features (except protected ones)
        features_to_remove = [f for f in features_to_remove if f not in protected_features]
        remaining_features = features_df.drop(columns=features_to_remove)
        
        # If we still have too many features, apply additional pruning
        if len(remaining_features.columns) > target_count:
            tprint_info(f"  🔧 Still have {len(remaining_features.columns)} features, need to reduce to {target_count}")
            
            # Track cross-timeframe before final selection
            ct_before_final = [f for f in remaining_features.columns if 
                             ('cross_timeframe' in f.lower() or 
                              '_3x_ratio' in f or 
                              '_6x_ratio' in f or 
                              '_9x_ratio' in f or 
                              '_27x_ratio' in f)]
            
            tprint_info("="*80)
            tprint_info("📊 FINAL SELECTION BY COMPOSITE SCORE")
            tprint_info("="*80)
            tprint_info(f"  📈 Cross-timeframe features before final selection: {len(ct_before_final)}")
            
            # Sort all remaining features by composite score and keep the best ones
            all_scores = [(f, composite_scores.get(f, 0.0)) for f in remaining_features.columns]
            all_scores.sort(key=lambda x: x[1], reverse=True)  # Sort by score (highest first)
            
            # Analyze cross-timeframe ranking
            ct_rankings = []
            for idx, (f, score) in enumerate(all_scores):
                if f in ct_before_final:
                    ct_rankings.append((idx, f, score))
            
            if ct_rankings:
                tprint_info(f"  📊 Cross-timeframe feature rankings:")
                tprint_info(f"      Best ranking: #{ct_rankings[0][0]+1} - {ct_rankings[0][1]} (score: {ct_rankings[0][2]:.4f})")
                tprint_info(f"      Worst ranking: #{ct_rankings[-1][0]+1} - {ct_rankings[-1][1]} (score: {ct_rankings[-1][2]:.4f})")
                tprint_info(f"      Median ranking: #{ct_rankings[len(ct_rankings)//2][0]+1}")
                tprint_info(f"      Cutoff position: #{target_count}")
                
                # Check how many would survive the cut
                ct_above_cutoff = [r for r in ct_rankings if r[0] < target_count]
                ct_below_cutoff = [r for r in ct_rankings if r[0] >= target_count]
                
                tprint_info(f"  📊 Cross-timeframe features above cutoff: {len(ct_above_cutoff)}")
                tprint_info(f"  📊 Cross-timeframe features below cutoff: {len(ct_below_cutoff)}")
                
                if len(ct_above_cutoff) == 0:
                    tprint_warning("  ⚠️ ALL cross-timeframe features are ranked below cutoff!")
                    tprint_warning("  ⚠️ This means they all have lower composite scores than other features")
                    tprint_warning("  ⚠️ Consider:")
                    tprint_warning("      1. Improving cross-timeframe feature calculation")
                    tprint_warning("      2. Adding explicit protection for cross-timeframe category")
                    tprint_warning("      3. Using different scoring metrics")
            
            # Keep the top features up to target count
            features_to_keep = [f for f, _ in all_scores[:target_count]]
            remaining_features = remaining_features[features_to_keep]
            
            # Track cross-timeframe features in final selection
            final_cross_timeframe = [f for f in features_to_keep if 
                                   ('cross_timeframe' in f.lower() or 
                                    '_3x_ratio' in f or 
                                    '_6x_ratio' in f or 
                                    '_9x_ratio' in f or 
                                    '_27x_ratio' in f)]
            
            tprint_info("="*80)
            tprint_info("📊 FINAL SELECTION RESULTS")
            tprint_info("="*80)
            tprint_info(f"  ✅ Selected top {len(features_to_keep)} features by composite score")
            tprint_info(f"  📊 Cross-timeframe features in final selection: {len(final_cross_timeframe)}")
            tprint_info(f"  📊 Cross-timeframe survival rate: {len(final_cross_timeframe)/len(ct_before_final):.1%}")
            
            if final_cross_timeframe:
                tprint_success(f"  ✅ Cross-timeframe features kept:")
                for f in final_cross_timeframe[:10]:  # Show first 10
                    score = composite_scores.get(f, 0.0)
                    tprint_success(f"      - {f} (score: {score:.4f})")
                if len(final_cross_timeframe) > 10:
                    tprint_success(f"      ... and {len(final_cross_timeframe) - 10} more")
            else:
                tprint_error("  ❌ NO cross-timeframe features kept!")
                tprint_error("  ❌ All cross-timeframe features were pruned in final selection")
        
        # Track results
        removed_count = len(features_to_remove)
        remaining_count = len(remaining_features.columns)
        retention_ratio = remaining_count / initial_count
        
        if removed_count > 0:
            tprint_info(f"  ✅ Removed {removed_count} features from correlation clusters")
            tprint_info(f"  📊 Remaining features: {remaining_count} ({retention_ratio:.1%} retention)")
        else:
            tprint_info(f"  ✅ No correlation clusters found for pruning")
            tprint_info(f"  📊 Remaining features: {initial_count}")
        
        clusters_found = len(clusters)
        if clusters_found > 0:
            tprint_info(f"  🔍 Found {clusters_found} correlation clusters (3-5 members each)")
        
        if len(protected_features) > 0:
            tprint_info(f"  🛡️ Protected {len(protected_features)} features from category constraints")
        
        self.stats['stage_results'][stage_name] = {
            'features_removed': removed_count,
            'features_remaining': remaining_count,
            'removed_features': features_to_remove,
            'protected_features': protected_features,
            'clusters_found': clusters_found,
            'retention_ratio': retention_ratio
        }
        self.stats['removed_features'].extend(features_to_remove)
        
        return remaining_features
    
    def _light_variance_pruning(
        self,
        features_df: pd.DataFrame,
        feature_categories: Dict[str, str],
        composite_scores: Dict[str, float],
        stage_name: str
    ) -> pd.DataFrame:
        """Light variance pruning to reach target retention ratio."""
        initial_count = len(features_df.columns)
        target_retention = 0.5  # Keep 50% of original features
        target_count = max(10, int(len(features_df.columns) * target_retention))
        
        if len(features_df.columns) <= target_count:
            return features_df
        
        # Calculate variances
        variances = features_df.var()
        
        # Sort by variance (lowest first) and remove bottom features
        variance_scores = [(f, variances[f]) for f in features_df.columns]
        variance_scores.sort(key=lambda x: x[1])  # Sort by variance (lowest first)
        
        # Calculate how many to remove
        features_to_remove_count = len(features_df.columns) - target_count
        features_to_remove = [f for f, _ in variance_scores[:features_to_remove_count]]
        
        # Apply category protection
        protected_features = []
        final_features_to_remove = []
        
        for feature in features_to_remove:
            category = feature_categories.get(feature, 'unknown')
            category_count = self._count_category_features(
                features_df.columns, feature_categories, category
            )
            
            if category_count > self.config.min_features_per_category:
                final_features_to_remove.append(feature)
            else:
                protected_features.append(feature)
                self.stats['protected_features'].append(feature)
        
        # Remove features
        remaining_features = features_df.drop(columns=final_features_to_remove)
        
        tprint_info(f"  ✅ Light variance pruning: removed {len(final_features_to_remove)} features")
        tprint_info(f"  📊 Remaining features: {len(remaining_features.columns)}")
        
        if len(protected_features) > 0:
            tprint_info(f"  🛡️ Protected {len(protected_features)} features from category constraints")
        
        self.stats['stage_results'][stage_name] = {
            'features_removed': len(final_features_to_remove),
            'features_remaining': len(remaining_features.columns),
            'removed_features': final_features_to_remove,
            'protected_features': protected_features
        }
        self.stats['removed_features'].extend(final_features_to_remove)
        
        return remaining_features
    
    def _correlation_pruning_optimized(
        self,
        features_df: pd.DataFrame,
        feature_categories: Dict[str, str],
        composite_scores: Dict[str, float],
        stage_name: str
    ) -> pd.DataFrame:
        """Optimized correlation pruning using clustering approach - remove bottom 50% within each correlation cluster."""
        initial_count = len(features_df.columns)
        tprint_info(f"  🔗 Stage 5: Optimized correlation clustering pruning... (starting with {initial_count} features)")
        
        # Use memory-efficient correlation matrix calculation
        corr_matrix = self._calculate_correlation_matrix(features_df)
        
        # Create correlation clusters using hierarchical clustering
        clusters = self._create_correlation_clusters(corr_matrix, self.config.correlation_threshold)
        
        # Track features to remove
        features_to_remove = []
        protected_features = []
        
        for cluster_id, cluster_features in clusters.items():
            if len(cluster_features) < 3:  # Skip small clusters
                continue
                
            tprint_info(f"    🔍 Processing cluster {cluster_id}: {len(cluster_features)} features")
            
            # Sort features by composite score (MI + stability) - lowest first
            cluster_scores = [(f, composite_scores.get(f, 0.0)) for f in cluster_features]
            cluster_scores.sort(key=lambda x: x[1])  # Sort by score (lowest first)
            
            # Remove bottom 10% from this cluster (less aggressive)
            remove_count = max(1, len(cluster_features) // 10)  # At most 10% per cluster
            features_to_remove_from_cluster = [f for f, _ in cluster_scores[:remove_count]]
            
            # Check category protection for each feature to remove
            for feature in features_to_remove_from_cluster:
                category = feature_categories.get(feature, 'unknown')
                category_count = self._count_category_features(
                    features_df.columns, feature_categories, category
                )
                
                if category_count > self.config.min_features_per_category:
                    features_to_remove.append(feature)
                else:
                    protected_features.append(feature)
                    self.stats['protected_features'].append(feature)
            
            tprint_info(f"    ✅ Cluster {cluster_id}: removing {len(features_to_remove_from_cluster)} features, keeping {len(cluster_features) - len(features_to_remove_from_cluster)}")
        
        # Remove clustered features (except protected ones)
        features_to_remove = [f for f in features_to_remove if f not in protected_features]
        remaining_features = features_df.drop(columns=features_to_remove)
        
        # Track results
        removed_count = len(features_to_remove)
        remaining_count = len(remaining_features.columns)
        
        if removed_count > 0:
            tprint_info(f"    ✅ Removed {removed_count} features from correlation clusters (bottom 50% per cluster)")
            tprint_info(f"    📊 Remaining features: {remaining_count}")
        else:
            tprint_info(f"    ✅ No correlation clusters found for pruning")
            tprint_info(f"    📊 Remaining features: {initial_count}")
        
        clusters_found = len(clusters)
        if clusters_found > 0:
            tprint_info(f"    🔍 Found {clusters_found} correlation clusters (3-5 members each)")
        
        if len(protected_features) > 0:
            tprint_info(f"    🛡️ Protected {len(protected_features)} features from category constraints")
        
        self.stats['stage_results'][stage_name] = {
            'features_removed': removed_count,
            'features_remaining': remaining_count,
            'removed_features': features_to_remove,
            'protected_features': protected_features,
            'clusters_found': clusters_found
        }
        self.stats['removed_features'].extend(features_to_remove)
        
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
