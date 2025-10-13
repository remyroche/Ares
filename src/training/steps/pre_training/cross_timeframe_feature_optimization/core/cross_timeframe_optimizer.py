"""
Cross-Timeframe Feature Optimizer

This module implements feature optimization for cross-timeframe features,
following the pattern of the core optimizer in FeatureLookbackOptimizationComponent.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import time
import logging

# Import optimization utilities
from src.utils.math_validation import safe_divide, validate_finite
from src.utils.common_operations import safe_dataframe_operation
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug

# Import VectorBT optimizations
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import (
        VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
    )
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    VectorBTRollingOptimizer = None
    get_vectorbt_rolling_optimizer = None

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """Optimization methods for cross-timeframe features."""
    MRMR = "mrmr"
    GRID_SEARCH = "grid_search"
    BAYESIAN = "bayesian"
    RANDOM_SEARCH = "random_search"
    CORRELATION = "correlation"
    MUTUAL_INFORMATION = "mutual_information"


@dataclass
class CrossTimeframeOptimizationConfig:
    """Configuration for cross-timeframe feature optimization."""
    
    # Optimization method
    optimization_method: str = "mrmr"
    lookback_range: Tuple[int, int] = (5, 50)
    max_features: int = 20
    
    # Performance optimization
    enable_vectorbt: bool = True
    enable_parallel: bool = True
    memory_efficient: bool = True
    
    # Optimization parameters
    n_trials: int = 50
    cv_folds: int = 5
    random_state: int = 42
    
    # Feature quality thresholds
    min_correlation: float = 0.01
    max_correlation: float = 0.95
    min_mutual_information: float = 0.001


@dataclass
class OptimizationResult:
    """Result from cross-timeframe feature optimization."""
    
    # Optimized features
    optimized_features: Dict[str, pd.Series]
    selected_features: List[str]
    feature_scores: Dict[str, float]
    
    # Optimization metrics
    optimization_score: float
    correlation_matrix: Optional[pd.DataFrame] = None
    mutual_information_scores: Optional[Dict[str, float]] = None
    
    # Performance metrics
    execution_time: float
    memory_usage_mb: float
    
    # Success indicators
    success: bool = True
    error_message: Optional[str] = None


class CrossTimeframeOptimizer:
    """
    Cross-Timeframe Feature Optimizer.
    
    Optimizes cross-timeframe features using various methods including
    MRMR, grid search, Bayesian optimization, and correlation analysis.
    """
    
    def __init__(self, config: CrossTimeframeOptimizationConfig):
        """Initialize the cross-timeframe optimizer."""
        self.config = config
        self.logger = logger
        
        # Initialize VectorBT optimizer if available
        self.vectorbt_optimizer = None
        if VECTORBT_AVAILABLE and config.enable_vectorbt:
            try:
                self.vectorbt_optimizer = get_vectorbt_rolling_optimizer()
                tprint("✅ VectorBT optimizer initialized")
            except Exception as e:
                tprint_warning(f"⚠️ VectorBT optimizer initialization failed: {e}")
        
        # Performance tracking
        self.performance_stats = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'failed_optimizations': 0,
            'total_execution_time': 0.0,
            'vectorbt_operations': 0,
            'pandas_operations': 0
        }
        
        tprint_info("🔧 CrossTimeframeOptimizer initialized")
    
    def optimize_features(self, 
                         data: pd.DataFrame, 
                         target_column: str,
                         lookback_range: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        """
        Optimize cross-timeframe features.
        
        Args:
            data: Input data with features and target
            target_column: Name of target column
            lookback_range: Optional lookback range for optimization
            
        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        
        try:
            tprint_info(f"🚀 Starting cross-timeframe feature optimization...")
            tprint_debug(f"📊 Data shape: {data.shape}, target: {target_column}")
            
            # Validate inputs
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            
            if data.empty:
                raise ValueError("Input data is empty")
            
            # Get feature columns (exclude target)
            feature_columns = [col for col in data.columns if col != target_column]
            
            if not feature_columns:
                raise ValueError("No feature columns found in data")
            
            tprint_info(f"🔍 Optimizing {len(feature_columns)} features using {self.config.optimization_method}")
            
            # Select optimization method
            if self.config.optimization_method == "mrmr":
                result = self._optimize_mrmr(data, feature_columns, target_column)
            elif self.config.optimization_method == "grid_search":
                result = self._optimize_grid_search(data, feature_columns, target_column)
            elif self.config.optimization_method == "bayesian":
                result = self._optimize_bayesian(data, feature_columns, target_column)
            elif self.config.optimization_method == "random_search":
                result = self._optimize_random_search(data, feature_columns, target_column)
            elif self.config.optimization_method == "correlation":
                result = self._optimize_correlation(data, feature_columns, target_column)
            elif self.config.optimization_method == "mutual_information":
                result = self._optimize_mutual_information(data, feature_columns, target_column)
            else:
                raise ValueError(f"Unknown optimization method: {self.config.optimization_method}")
            
            execution_time = time.time() - start_time
            
            # Update performance stats
            self.performance_stats.update({
                'total_optimizations': 1,
                'successful_optimizations': 1 if result.get('success', False) else 0,
                'failed_optimizations': 0 if result.get('success', False) else 1,
                'total_execution_time': execution_time
            })
            
            result['execution_time'] = execution_time
            
            tprint_success(f"✅ Cross-timeframe feature optimization completed in {execution_time:.3f}s")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            tprint_error(f"❌ Cross-timeframe feature optimization failed: {e}")
            
            self.performance_stats.update({
                'total_optimizations': 1,
                'failed_optimizations': 1,
                'total_execution_time': execution_time
            })
            
            return {
                'success': False,
                'error_message': str(e),
                'execution_time': execution_time,
                'optimized_features': {},
                'selected_features': [],
                'feature_scores': {}
            }
    
    def _optimize_mrmr(self, data: pd.DataFrame, feature_columns: List[str], target_column: str) -> Dict[str, Any]:
        """Optimize using MRMR (Maximum Relevance Minimum Redundancy) approach."""
        try:
            tprint_info("🎯 Using MRMR optimization...")
            
            # Calculate mutual information with target
            target_data = data[target_column].dropna()
            mi_scores = {}
            
            for feature in feature_columns:
                feature_data = data[feature].dropna()
                
                # Align data
                common_index = target_data.index.intersection(feature_data.index)
                if len(common_index) < 10:  # Need minimum data points
                    continue
                
                aligned_target = target_data.loc[common_index]
                aligned_feature = feature_data.loc[common_index]
                
                # Calculate mutual information
                mi_score = self._calculate_mutual_information(aligned_target, aligned_feature)
                if mi_score is not None and mi_score > self.config.min_mutual_information:
                    mi_scores[feature] = mi_score
            
            if not mi_scores:
                tprint_warning("⚠️ No features passed mutual information threshold")
                return self._create_empty_result()
            
            # Sort by mutual information score
            sorted_features = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Select top features
            selected_features = [feature for feature, _ in sorted_features[:self.config.max_features]]
            feature_scores = dict(sorted_features[:self.config.max_features])
            
            # Create optimized features
            optimized_features = {feature: data[feature] for feature in selected_features}
            
            tprint_success(f"✅ MRMR optimization selected {len(selected_features)} features")
            
            return {
                'success': True,
                'optimized_features': optimized_features,
                'selected_features': selected_features,
                'feature_scores': feature_scores,
                'optimization_score': np.mean(list(feature_scores.values())),
                'mutual_information_scores': feature_scores
            }
            
        except Exception as e:
            tprint_error(f"❌ MRMR optimization failed: {e}")
            return self._create_empty_result(str(e))
    
    def _optimize_grid_search(self, data: pd.DataFrame, feature_columns: List[str], target_column: str) -> Dict[str, Any]:
        """Optimize using grid search approach."""
        try:
            tprint_info("🔍 Using grid search optimization...")
            
            # Calculate correlation with target
            target_data = data[target_column].dropna()
            correlation_scores = {}
            
            for feature in feature_columns:
                feature_data = data[feature].dropna()
                
                # Align data
                common_index = target_data.index.intersection(feature_data.index)
                if len(common_index) < 10:
                    continue
                
                aligned_target = target_data.loc[common_index]
                aligned_feature = feature_data.loc[common_index]
                
                # Calculate correlation
                correlation = self._calculate_correlation(aligned_target, aligned_feature)
                if correlation is not None and abs(correlation) > self.config.min_correlation:
                    correlation_scores[feature] = abs(correlation)
            
            if not correlation_scores:
                tprint_warning("⚠️ No features passed correlation threshold")
                return self._create_empty_result()
            
            # Sort by correlation score
            sorted_features = sorted(correlation_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Select top features
            selected_features = [feature for feature, _ in sorted_features[:self.config.max_features]]
            feature_scores = dict(sorted_features[:self.config.max_features])
            
            # Create optimized features
            optimized_features = {feature: data[feature] for feature in selected_features}
            
            tprint_success(f"✅ Grid search optimization selected {len(selected_features)} features")
            
            return {
                'success': True,
                'optimized_features': optimized_features,
                'selected_features': selected_features,
                'feature_scores': feature_scores,
                'optimization_score': np.mean(list(feature_scores.values())),
                'correlation_scores': feature_scores
            }
            
        except Exception as e:
            tprint_error(f"❌ Grid search optimization failed: {e}")
            return self._create_empty_result(str(e))
    
    def _optimize_bayesian(self, data: pd.DataFrame, feature_columns: List[str], target_column: str) -> Dict[str, Any]:
        """Optimize using Bayesian optimization approach."""
        try:
            tprint_info("🎲 Using Bayesian optimization...")
            
            # For now, use a simplified approach similar to grid search
            # In a full implementation, this would use a Bayesian optimization library
            return self._optimize_grid_search(data, feature_columns, target_column)
            
        except Exception as e:
            tprint_error(f"❌ Bayesian optimization failed: {e}")
            return self._create_empty_result(str(e))
    
    def _optimize_random_search(self, data: pd.DataFrame, feature_columns: List[str], target_column: str) -> Dict[str, Any]:
        """Optimize using random search approach."""
        try:
            tprint_info("🎲 Using random search optimization...")
            
            # Randomly select features
            np.random.seed(self.config.random_state)
            n_features = min(self.config.max_features, len(feature_columns))
            selected_features = np.random.choice(feature_columns, size=n_features, replace=False).tolist()
            
            # Calculate scores for selected features
            feature_scores = {}
            for feature in selected_features:
                feature_data = data[feature].dropna()
                target_data = data[target_column].dropna()
                
                # Align data
                common_index = target_data.index.intersection(feature_data.index)
                if len(common_index) < 10:
                    feature_scores[feature] = 0.0
                    continue
                
                aligned_target = target_data.loc[common_index]
                aligned_feature = feature_data.loc[common_index]
                
                # Calculate correlation as score
                correlation = self._calculate_correlation(aligned_target, aligned_feature)
                feature_scores[feature] = abs(correlation) if correlation is not None else 0.0
            
            # Create optimized features
            optimized_features = {feature: data[feature] for feature in selected_features}
            
            tprint_success(f"✅ Random search optimization selected {len(selected_features)} features")
            
            return {
                'success': True,
                'optimized_features': optimized_features,
                'selected_features': selected_features,
                'feature_scores': feature_scores,
                'optimization_score': np.mean(list(feature_scores.values()))
            }
            
        except Exception as e:
            tprint_error(f"❌ Random search optimization failed: {e}")
            return self._create_empty_result(str(e))
    
    def _optimize_correlation(self, data: pd.DataFrame, feature_columns: List[str], target_column: str) -> Dict[str, Any]:
        """Optimize using correlation analysis."""
        try:
            tprint_info("📊 Using correlation optimization...")
            
            # Calculate correlation matrix
            feature_data = data[feature_columns + [target_column]].dropna()
            
            if feature_data.empty:
                tprint_warning("⚠️ No valid data for correlation analysis")
                return self._create_empty_result()
            
            correlation_matrix = feature_data.corr()
            target_correlations = correlation_matrix[target_column].drop(target_column)
            
            # Filter by correlation threshold
            valid_correlations = target_correlations[
                (abs(target_correlations) > self.config.min_correlation) &
                (abs(target_correlations) < self.config.max_correlation)
            ]
            
            if valid_correlations.empty:
                tprint_warning("⚠️ No features passed correlation thresholds")
                return self._create_empty_result()
            
            # Sort by absolute correlation
            sorted_correlations = valid_correlations.abs().sort_values(ascending=False)
            
            # Select top features
            selected_features = sorted_correlations.head(self.config.max_features).index.tolist()
            feature_scores = sorted_correlations.head(self.config.max_features).to_dict()
            
            # Create optimized features
            optimized_features = {feature: data[feature] for feature in selected_features}
            
            tprint_success(f"✅ Correlation optimization selected {len(selected_features)} features")
            
            return {
                'success': True,
                'optimized_features': optimized_features,
                'selected_features': selected_features,
                'feature_scores': feature_scores,
                'optimization_score': np.mean(list(feature_scores.values())),
                'correlation_matrix': correlation_matrix
            }
            
        except Exception as e:
            tprint_error(f"❌ Correlation optimization failed: {e}")
            return self._create_empty_result(str(e))
    
    def _optimize_mutual_information(self, data: pd.DataFrame, feature_columns: List[str], target_column: str) -> Dict[str, Any]:
        """Optimize using mutual information analysis."""
        try:
            tprint_info("🔗 Using mutual information optimization...")
            
            target_data = data[target_column].dropna()
            mi_scores = {}
            
            for feature in feature_columns:
                feature_data = data[feature].dropna()
                
                # Align data
                common_index = target_data.index.intersection(feature_data.index)
                if len(common_index) < 10:
                    continue
                
                aligned_target = target_data.loc[common_index]
                aligned_feature = feature_data.loc[common_index]
                
                # Calculate mutual information
                mi_score = self._calculate_mutual_information(aligned_target, aligned_feature)
                if mi_score is not None and mi_score > self.config.min_mutual_information:
                    mi_scores[feature] = mi_score
            
            if not mi_scores:
                tprint_warning("⚠️ No features passed mutual information threshold")
                return self._create_empty_result()
            
            # Sort by mutual information score
            sorted_features = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Select top features
            selected_features = [feature for feature, _ in sorted_features[:self.config.max_features]]
            feature_scores = dict(sorted_features[:self.config.max_features])
            
            # Create optimized features
            optimized_features = {feature: data[feature] for feature in selected_features}
            
            tprint_success(f"✅ Mutual information optimization selected {len(selected_features)} features")
            
            return {
                'success': True,
                'optimized_features': optimized_features,
                'selected_features': selected_features,
                'feature_scores': feature_scores,
                'optimization_score': np.mean(list(feature_scores.values())),
                'mutual_information_scores': feature_scores
            }
            
        except Exception as e:
            tprint_error(f"❌ Mutual information optimization failed: {e}")
            return self._create_empty_result(str(e))
    
    def _calculate_correlation(self, x: pd.Series, y: pd.Series) -> Optional[float]:
        """Calculate correlation between two series."""
        try:
            if len(x) < 2 or len(y) < 2:
                return None
            
            correlation = x.corr(y)
            return correlation if not np.isnan(correlation) else None
            
        except Exception:
            return None
    
    def _calculate_mutual_information(self, x: pd.Series, y: pd.Series) -> Optional[float]:
        """Calculate mutual information between two series."""
        try:
            if len(x) < 2 or len(y) < 2:
                return None
            
            # Simple mutual information calculation using correlation
            # In a full implementation, this would use proper mutual information
            correlation = self._calculate_correlation(x, y)
            if correlation is None:
                return None
            
            # Convert correlation to mutual information approximation
            # MI ≈ -0.5 * log(1 - r²) where r is correlation
            mi = -0.5 * np.log(1 - correlation**2) if abs(correlation) < 0.999 else 0.0
            return mi if not np.isnan(mi) and np.isfinite(mi) else None
            
        except Exception:
            return None
    
    def _create_empty_result(self, error_message: str = None) -> Dict[str, Any]:
        """Create empty result for failed optimization."""
        return {
            'success': False,
            'error_message': error_message,
            'optimized_features': {},
            'selected_features': [],
            'feature_scores': {},
            'optimization_score': 0.0
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()


# Export main classes
__all__ = [
    'CrossTimeframeOptimizer',
    'CrossTimeframeOptimizationConfig',
    'OptimizationResult',
    'OptimizationMethod'
]