"""
Compute-Aware Optimizer for Feature Selection

This module provides compute-aware optimization techniques for efficient
feature selection and processing.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import time
import psutil
import warnings

logger = logging.getLogger(__name__)

class ComputeAwareOptimizer:
    """
    Compute-aware optimizer for efficient feature processing.
    """
    
    def __init__(self, 
                 max_memory_gb: float = 8.0,
                 max_compute_time_ms: float = 1000.0,
                 enable_caching: bool = True,
                 enable_parallel: bool = True):
        """
        Initialize compute-aware optimizer.
        
        Args:
            max_memory_gb: Maximum memory usage in GB
            max_compute_time_ms: Maximum compute time per feature in ms
            enable_caching: Whether to enable feature caching
            enable_parallel: Whether to enable parallel processing
        """
        self.max_memory_gb = max_memory_gb
        self.max_compute_time_ms = max_compute_time_ms
        self.enable_caching = enable_caching
        self.enable_parallel = enable_parallel
        
        # Caching
        self.cache = {} if enable_caching else None
        
        # Compute profiles
        self.compute_profiles = {}
        self.memory_usage = {}
        
        # Matrix operations
        self.matrix_ops = None
        if enable_parallel:
            try:
                from src.utils.matrix_operations import get_unified_matrix_operations
                self.matrix_ops = get_unified_matrix_operations(enable_gpu=True, enable_parallel=True)
            except ImportError:
                logger.warning("Matrix operations not available")
    
    def optimize_feature_computation(self, X: pd.DataFrame, y: pd.Series,
                                   feature_families: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        """
        Optimize feature computation with compute-aware techniques.
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_families: Dictionary mapping family names to feature lists
            
        Returns:
            Optimization results
        """
        logger.info("Starting compute-aware optimization...")
        
        if feature_families is None:
            feature_families = self._auto_detect_families(X.columns)
        
        # Phase 1: Hierarchical screening per family
        logger.info("Phase 1: Hierarchical screening per family...")
        family_results = self._hierarchical_screening(X, y, feature_families)
        
        # Phase 2: Cached rolling cores
        logger.info("Phase 2: Setting up cached rolling cores...")
        rolling_cores = self._setup_cached_rolling_cores(X)
        
        # Phase 3: Vectorized operations
        logger.info("Phase 3: Vectorized operations...")
        vectorized_results = self._vectorized_operations(X, y, family_results['selected_features'])
        
        # Phase 4: Compute profiling and optimization
        logger.info("Phase 4: Compute profiling and optimization...")
        optimization_results = self._compute_profiling_and_optimization(
            X, y, vectorized_results['optimized_features']
        )
        
        # Compile results
        results = {
            'family_results': family_results,
            'rolling_cores': rolling_cores,
            'vectorized_results': vectorized_results,
            'optimization_results': optimization_results,
            'compute_profiles': self.compute_profiles,
            'memory_usage': self.memory_usage
        }
        
        logger.info("Compute-aware optimization completed")
        return results
    
    def _hierarchical_screening(self, X: pd.DataFrame, y: pd.Series,
                               feature_families: Dict[str, List[str]]) -> Dict[str, Any]:
        """Hierarchical screening per family to avoid dropping correlated clusters."""
        family_results = {}
        all_selected_features = []
        
        for family_name, family_features in feature_families.items():
            logger.info(f"Processing family: {family_name}")
            
            # Filter existing features
            existing_features = [f for f in family_features if f in X.columns]
            if not existing_features:
                continue
            
            # Process family with compute awareness
            family_X = X[existing_features]
            family_result = self._process_family_with_compute_awareness(family_X, y, family_name)
            
            family_results[family_name] = family_result
            all_selected_features.extend(family_result['selected_features'])
        
        return {
            'family_results': family_results,
            'selected_features': all_selected_features
        }
    
    def _process_family_with_compute_awareness(self, family_X: pd.DataFrame, y: pd.Series,
                                             family_name: str) -> Dict[str, Any]:
        """Process a family with compute awareness."""
        # Check memory usage
        memory_usage = family_X.memory_usage(deep=True).sum() / 1024**3  # GB
        if memory_usage > self.max_memory_gb:
            logger.warning(f"Family {family_name} exceeds memory limit: {memory_usage:.2f}GB")
            # Sample data if too large
            sample_size = int(len(family_X) * self.max_memory_gb / memory_usage)
            family_X = family_X.sample(n=sample_size, random_state=42)
        
        # Compute family metrics with timing
        start_time = time.time()
        family_metrics = self._compute_family_metrics(family_X, y)
        compute_time = (time.time() - start_time) * 1000  # ms
        
        # Check compute time
        if compute_time > self.max_compute_time_ms:
            logger.warning(f"Family {family_name} exceeds compute time limit: {compute_time:.2f}ms")
            # Use faster approximation
            family_metrics = self._compute_family_metrics_fast(family_X, y)
        
        # Select features based on metrics
        selected_features = self._select_family_features(family_metrics, family_name)
        
        # Store compute profile
        self.compute_profiles[family_name] = {
            'compute_time_ms': compute_time,
            'memory_usage_gb': memory_usage,
            'n_features': len(family_X.columns),
            'n_selected': len(selected_features)
        }
        
        return {
            'selected_features': selected_features,
            'metrics': family_metrics,
            'compute_profile': self.compute_profiles[family_name]
        }
    
    def _compute_family_metrics(self, family_X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Compute metrics for a family with full accuracy."""
        metrics = []
        
        for feature in family_X.columns:
            feature_data = family_X[feature].dropna()
            if len(feature_data) == 0:
                continue
            
            # Align with target
            common_idx = feature_data.index.intersection(y.index)
            if len(common_idx) == 0:
                continue
            
            feature_aligned = feature_data.loc[common_idx]
            y_aligned = y.loc[common_idx]
            
            valid_mask = ~(feature_aligned.isna() | y_aligned.isna())
            if valid_mask.sum() < 10:
                continue
            
            feature_clean = feature_aligned[valid_mask]
            y_clean = y_aligned[valid_mask]
            
            # Compute metrics
            feature_metrics = self._compute_single_feature_metrics(feature_clean, y_clean, feature)
            metrics.append(feature_metrics)
        
        return pd.DataFrame(metrics)
    
    def _compute_family_metrics_fast(self, family_X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Compute metrics for a family with fast approximation."""
        # Use matrix operations for speed
        if self.matrix_ops is not None:
            return self._compute_family_metrics_matrix(family_X, y)
        else:
            # Fallback to standard computation
            return self._compute_family_metrics(family_X, y)
    
    def _compute_family_metrics_matrix(self, family_X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Compute family metrics using matrix operations."""
        try:
            # Align data
            common_idx = family_X.index.intersection(y.index)
            if len(common_idx) == 0:
                return pd.DataFrame()
            
            X_aligned = family_X.loc[common_idx]
            y_aligned = y.loc[common_idx]
            
            # Remove NaN values
            valid_mask = ~(X_aligned.isna().any(axis=1) | y_aligned.isna())
            if valid_mask.sum() < 10:
                return pd.DataFrame()
            
            X_clean = X_aligned[valid_mask]
            y_clean = y_aligned[valid_mask]
            
            # Use matrix operations for correlation
            corr_matrix = X_clean.corr()
            
            # Compute MI using vectorized operations
            mi_scores = self._compute_mi_vectorized(X_clean, y_clean)
            
            # Create metrics DataFrame
            metrics = []
            for i, feature in enumerate(X_clean.columns):
                metrics.append({
                    'feature': feature,
                    'mi_score': mi_scores[i] if i < len(mi_scores) else 0.0,
                    'compute_time_ms': 0.1,  # Approximate
                    'memory_usage_mb': X_clean[feature].memory_usage(deep=True) / 1024**2
                })
            
            return pd.DataFrame(metrics)
        except Exception as e:
            logger.warning(f"Matrix computation failed: {e}")
            return self._compute_family_metrics(family_X, y)
    
    def _compute_mi_vectorized(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Compute MI scores using vectorized operations."""
        try:
            from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
            
            is_classification = y.dtype.name == 'category' or y.dtype == 'object'
            
            if is_classification:
                mi_scores = mutual_info_classif(X, y, random_state=42)
            else:
                mi_scores = mutual_info_regression(X, y, random_state=42)
            
            return mi_scores
        except:
            return np.zeros(len(X.columns))
    
    def _compute_single_feature_metrics(self, feature_data: pd.Series, y: pd.Series, 
                                       feature_name: str) -> Dict[str, Any]:
        """Compute metrics for a single feature."""
        start_time = time.time()
        
        # Basic metrics
        mi_score = self._compute_mi_score(feature_data, y)
        cv_score = self._compute_cv_score(feature_data, y)
        
        compute_time = (time.time() - start_time) * 1000
        
        return {
            'feature': feature_name,
            'mi_score': mi_score,
            'cv_score': cv_score,
            'compute_time_ms': compute_time,
            'memory_usage_mb': feature_data.memory_usage(deep=True) / 1024**2
        }
    
    def _compute_mi_score(self, feature_data: pd.Series, y: pd.Series) -> float:
        """Compute MI score."""
        try:
            from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
            
            is_classification = y.dtype.name == 'category' or y.dtype == 'object'
            
            if is_classification:
                return mutual_info_classif(feature_data.values.reshape(-1, 1), y, random_state=42)[0]
            else:
                return mutual_info_regression(feature_data.values.reshape(-1, 1), y, random_state=42)[0]
        except:
            return 0.0
    
    def _compute_cv_score(self, feature_data: pd.Series, y: pd.Series) -> float:
        """Compute CV score."""
        try:
            from sklearn.model_selection import cross_val_score
            from sklearn.linear_model import Ridge, LogisticRegression
            from sklearn.preprocessing import StandardScaler
            
            X_scaled = StandardScaler().fit_transform(feature_data.values.reshape(-1, 1))
            
            is_classification = y.dtype.name == 'category' or y.dtype == 'object'
            
            if is_classification:
                model = LogisticRegression(random_state=42, max_iter=1000)
                scoring = 'roc_auc'
            else:
                model = Ridge(alpha=1.0)
                scoring = 'r2'
            
            scores = cross_val_score(model, X_scaled, y, cv=3, scoring=scoring)
            return np.mean(scores)
        except:
            return 0.0
    
    def _select_family_features(self, family_metrics: pd.DataFrame, family_name: str) -> List[str]:
        """Select top features from a family."""
        if len(family_metrics) == 0:
            return []
        
        # Sort by MI score and take top features
        top_features = family_metrics.nlargest(10, 'mi_score')  # Top 10 per family
        
        return top_features['feature'].tolist()
    
    def _setup_cached_rolling_cores(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Set up cached rolling cores using VectorBT for efficient computation."""
        try:
            import vectorbt as vbt
            from vectorbt.generic import rolling_mean, rolling_std
            
            rolling_cores = {}
            
            # Precompute common rolling operations using VectorBT
            if 'close' in X.columns:
                close_prices = X['close']
                
                # Rolling returns
                rolling_cores['returns'] = close_prices.pct_change()
                
                # Rolling means using VectorBT
                rolling_cores['ma_5'] = rolling_mean(close_prices, window=5)
                rolling_cores['ma_10'] = rolling_mean(close_prices, window=10)
                rolling_cores['ma_20'] = rolling_mean(close_prices, window=20)
                
                # Rolling std using VectorBT
                rolling_cores['std_5'] = rolling_std(close_prices, window=5)
                rolling_cores['std_10'] = rolling_std(close_prices, window=10)
                rolling_cores['std_20'] = rolling_std(close_prices, window=20)
                
                # EWMA using VectorBT
                rolling_cores['ewma_5'] = close_prices.ewm(span=5).mean()
                rolling_cores['ewma_10'] = close_prices.ewm(span=10).mean()
                rolling_cores['ewma_20'] = close_prices.ewm(span=20).mean()
            
            # Store in cache
            if self.cache is not None:
                self.cache['rolling_cores'] = rolling_cores
            
            return rolling_cores
            
        except Exception as e:
            logger.warning(f"VectorBT operations failed, falling back to pandas: {e}")
            # Fallback to pandas operations
            rolling_cores = {}
            
            if 'close' in X.columns:
                close_prices = X['close']
                
                rolling_cores['returns'] = close_prices.pct_change()
                rolling_cores['ma_5'] = close_prices.rolling(5).mean()
                rolling_cores['ma_10'] = close_prices.rolling(10).mean()
                rolling_cores['ma_20'] = close_prices.rolling(20).mean()
                rolling_cores['std_5'] = close_prices.rolling(5).std()
                rolling_cores['std_10'] = close_prices.rolling(10).std()
                rolling_cores['std_20'] = close_prices.rolling(20).std()
                rolling_cores['ewma_5'] = close_prices.ewm(span=5).mean()
                rolling_cores['ewma_10'] = close_prices.ewm(span=10).mean()
                rolling_cores['ewma_20'] = close_prices.ewm(span=20).mean()
            
            if self.cache is not None:
                self.cache['rolling_cores'] = rolling_cores
            
            return rolling_cores
    
    def _vectorized_operations(self, X: pd.DataFrame, y: pd.Series, 
                              selected_features: List[str]) -> Dict[str, Any]:
        """Perform vectorized operations for selected features."""
        if not selected_features:
            return {'optimized_features': [], 'vectorized_metrics': {}}
        
        # Filter to selected features
        feature_data = X[selected_features].dropna()
        
        if len(feature_data) == 0:
            return {'optimized_features': [], 'vectorized_metrics': {}}
        
        # Align with target
        common_idx = feature_data.index.intersection(y.index)
        if len(common_idx) == 0:
            return {'optimized_features': [], 'vectorized_metrics': {}}
        
        X_aligned = feature_data.loc[common_idx]
        y_aligned = y.loc[common_idx]
        
        # Remove NaN values
        valid_mask = ~(X_aligned.isna().any(axis=1) | y_aligned.isna())
        if valid_mask.sum() < 10:
            return {'optimized_features': [], 'vectorized_metrics': {}}
        
        X_clean = X_aligned[valid_mask]
        y_clean = y_aligned[valid_mask]
        
        # Vectorized correlation computation
        corr_matrix = X_clean.corr()
        
        # Vectorized MI computation
        mi_scores = self._compute_mi_vectorized(X_clean, y_clean)
        
        # Create feature metrics
        vectorized_metrics = {}
        for i, feature in enumerate(X_clean.columns):
            vectorized_metrics[feature] = {
                'mi_score': mi_scores[i] if i < len(mi_scores) else 0.0,
                'max_correlation': corr_matrix[feature].drop(feature).max() if feature in corr_matrix.columns else 0.0,
                'compute_time_ms': 0.1  # Approximate for vectorized
            }
        
        # Filter features based on vectorized metrics
        optimized_features = self._filter_by_vectorized_metrics(vectorized_metrics)
        
        return {
            'optimized_features': optimized_features,
            'vectorized_metrics': vectorized_metrics
        }
    
    def _filter_by_vectorized_metrics(self, vectorized_metrics: Dict[str, Any]) -> List[str]:
        """Filter features based on vectorized metrics."""
        # Filter by MI score and correlation
        qualified_features = []
        
        for feature, metrics in vectorized_metrics.items():
            mi_score = metrics['mi_score']
            max_corr = metrics['max_correlation']
            
            # Keep features with good MI and low correlation
            if mi_score > 0.01 and max_corr < 0.95:
                qualified_features.append(feature)
        
        return qualified_features
    
    def _compute_profiling_and_optimization(self, X: pd.DataFrame, y: pd.Series,
                                          selected_features: List[str]) -> Dict[str, Any]:
        """Compute profiling and optimization results."""
        if not selected_features:
            return {'final_features': [], 'optimization_summary': {}}
        
        # Profile each feature
        feature_profiles = {}
        for feature in selected_features:
            if feature not in X.columns:
                continue
            
            profile = self._profile_feature(X[feature], y, feature)
            feature_profiles[feature] = profile
        
        # Filter based on profiles
        final_features = self._filter_by_profiles(feature_profiles)
        
        # Compute optimization summary
        optimization_summary = self._compute_optimization_summary(feature_profiles, final_features)
        
        return {
            'final_features': final_features,
            'feature_profiles': feature_profiles,
            'optimization_summary': optimization_summary
        }
    
    def _profile_feature(self, feature_data: pd.Series, y: pd.Series, feature_name: str) -> Dict[str, Any]:
        """Profile a single feature."""
        start_time = time.time()
        
        # Compute time
        compute_time = (time.time() - start_time) * 1000
        
        # Memory usage
        memory_usage = feature_data.memory_usage(deep=True) / 1024**2  # MB
        
        # Performance metrics
        mi_score = self._compute_mi_score(feature_data, y)
        cv_score = self._compute_cv_score(feature_data, y)
        
        # Efficiency score
        efficiency_score = mi_score / (compute_time + 1e-8)  # MI per ms
        
        return {
            'feature': feature_name,
            'compute_time_ms': compute_time,
            'memory_usage_mb': memory_usage,
            'mi_score': mi_score,
            'cv_score': cv_score,
            'efficiency_score': efficiency_score
        }
    
    def _filter_by_profiles(self, feature_profiles: Dict[str, Any]) -> List[str]:
        """Filter features based on profiles."""
        # Sort by efficiency score
        sorted_features = sorted(
            feature_profiles.items(),
            key=lambda x: x[1]['efficiency_score'],
            reverse=True
        )
        
        # Take top features
        final_features = [feature for feature, _ in sorted_features[:50]]  # Top 50
        
        return final_features
    
    def _compute_optimization_summary(self, feature_profiles: Dict[str, Any], 
                                    final_features: List[str]) -> Dict[str, Any]:
        """Compute optimization summary."""
        if not feature_profiles:
            return {}
        
        # Overall statistics
        all_profiles = list(feature_profiles.values())
        compute_times = [p['compute_time_ms'] for p in all_profiles]
        memory_usage = [p['memory_usage_mb'] for p in all_profiles]
        efficiency_scores = [p['efficiency_score'] for p in all_profiles]
        
        summary = {
            'total_features_profiled': len(all_profiles),
            'final_features_selected': len(final_features),
            'mean_compute_time_ms': np.mean(compute_times),
            'mean_memory_usage_mb': np.mean(memory_usage),
            'mean_efficiency_score': np.mean(efficiency_scores),
            'total_compute_time_ms': np.sum(compute_times),
            'total_memory_usage_mb': np.sum(memory_usage)
        }
        
        return summary
    
    def _auto_detect_families(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """Auto-detect feature families."""
        families = {
            'returns': [],
            'vwap': [],
            'volatility': [],
            'volume': [],
            'momentum': [],
            'technical': [],
            'other': []
        }
        
        for feature in feature_names:
            if any(pattern in feature.lower() for pattern in ['ret_', 'return']):
                families['returns'].append(feature)
            elif any(pattern in feature.lower() for pattern in ['vwap', 'vw_']):
                families['vwap'].append(feature)
            elif any(pattern in feature.lower() for pattern in ['vol_', 'volatility']):
                families['volatility'].append(feature)
            elif any(pattern in feature.lower() for pattern in ['volume', 'vol_']):
                families['volume'].append(feature)
            elif any(pattern in feature.lower() for pattern in ['mom_', 'momentum', 'acc_']):
                families['momentum'].append(feature)
            elif any(pattern in feature.lower() for pattern in ['rsi', 'macd', 'bollinger']):
                families['technical'].append(feature)
            else:
                families['other'].append(feature)
        
        return {k: v for k, v in families.items() if v}
    
    def get_compute_summary(self) -> Dict[str, Any]:
        """Get compute usage summary."""
        if not self.compute_profiles:
            return {}
        
        total_compute_time = sum(p['compute_time_ms'] for p in self.compute_profiles.values())
        total_memory_usage = sum(p['memory_usage_gb'] for p in self.compute_profiles.values())
        
        return {
            'total_compute_time_ms': total_compute_time,
            'total_memory_usage_gb': total_memory_usage,
            'family_profiles': self.compute_profiles,
            'optimization_enabled': self.enable_caching and self.enable_parallel
        }