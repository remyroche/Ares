"""
Feature Selector for Cross-Timeframe Features

This module implements feature selection for cross-timeframe features,
following the pattern of FeatureLookbackOptimizationComponent.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
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


@dataclass
class SelectionConfig:
    """Configuration for feature selection."""
    
    # Selection method
    selection_method: str = "mutual_information"  # mutual_information, correlation, variance, recursive
    
    # Selection thresholds
    selection_threshold: float = 0.01
    max_correlation: float = 0.95
    min_variance: float = 0.001
    
    # Performance optimization
    enable_vectorbt: bool = True
    enable_parallel: bool = True
    memory_efficient: bool = True
    
    # Selection parameters
    max_features: int = 20
    min_features: int = 1
    random_state: int = 42


@dataclass
class SelectionResult:
    """Result from feature selection."""
    
    # Selected features
    selected_features: List[str]
    feature_scores: Dict[str, float]
    selection_scores: Dict[str, float]
    
    # Selection metrics
    total_features: int
    selected_count: int
    selection_ratio: float
    
    # Performance metrics
    execution_time: float
    memory_usage_mb: float
    
    # Success indicators
    success: bool = True
    error_message: Optional[str] = None


class FeatureSelector:
    """
    Feature Selector for Cross-Timeframe Features.
    
    Selects the best features based on various criteria including
    mutual information, correlation, variance, and recursive selection.
    """
    
    def __init__(self, config: SelectionConfig):
        """Initialize the feature selector."""
        self.config = config
        self.logger = logger
        
        # Initialize VectorBT optimizer if available
        self.vectorbt_optimizer = None
        if VECTORBT_AVAILABLE and config.enable_vectorbt:
            try:
                self.vectorbt_optimizer = get_vectorbt_rolling_optimizer()
                tprint("✅ VectorBT selector initialized")
            except Exception as e:
                tprint_warning(f"⚠️ VectorBT selector initialization failed: {e}")
        
        # Performance tracking
        self.performance_stats = {
            'total_selections': 0,
            'successful_selections': 0,
            'failed_selections': 0,
            'total_execution_time': 0.0,
            'vectorbt_operations': 0,
            'pandas_operations': 0
        }
        
        tprint_info("🔧 FeatureSelector initialized")
    
    def select_features(self, 
                       data: pd.DataFrame, 
                       target_column: str,
                       max_features: Optional[int] = None) -> Dict[str, Any]:
        """
        Select best features based on the configured method.
        
        Args:
            data: Input data with features and target
            target_column: Name of target column
            max_features: Maximum number of features to select
            
        Returns:
            Dictionary with selection results
        """
        start_time = time.time()
        
        try:
            tprint_info(f"🎯 Starting feature selection using {self.config.selection_method}...")
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
            
            # Use configured max_features if not provided
            if max_features is None:
                max_features = self.config.max_features
            
            max_features = min(max_features, len(feature_columns))
            
            tprint_info(f"🔍 Selecting from {len(feature_columns)} features, max: {max_features}")
            
            # Select features based on method
            if self.config.selection_method == "mutual_information":
                result = self._select_by_mutual_information(data, feature_columns, target_column, max_features)
            elif self.config.selection_method == "correlation":
                result = self._select_by_correlation(data, feature_columns, target_column, max_features)
            elif self.config.selection_method == "variance":
                result = self._select_by_variance(data, feature_columns, max_features)
            elif self.config.selection_method == "recursive":
                result = self._select_recursive(data, feature_columns, target_column, max_features)
            else:
                raise ValueError(f"Unknown selection method: {self.config.selection_method}")
            
            execution_time = time.time() - start_time
            
            # Update performance stats
            self.performance_stats.update({
                'total_selections': 1,
                'successful_selections': 1 if result.get('success', False) else 0,
                'failed_selections': 0 if result.get('success', False) else 1,
                'total_execution_time': execution_time
            })
            
            result['execution_time'] = execution_time
            
            tprint_success(f"✅ Feature selection completed in {execution_time:.3f}s")
            tprint_info(f"📊 Selected {len(result.get('selected_features', []))} features")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            tprint_error(f"❌ Feature selection failed: {e}")
            
            self.performance_stats.update({
                'total_selections': 1,
                'failed_selections': 1,
                'total_execution_time': execution_time
            })
            
            return {
                'success': False,
                'error_message': str(e),
                'execution_time': execution_time,
                'selected_features': [],
                'feature_scores': {}
            }
    
    def _select_by_mutual_information(self, 
                                    data: pd.DataFrame, 
                                    feature_columns: List[str], 
                                    target_column: str,
                                    max_features: int) -> Dict[str, Any]:
        """Select features using mutual information with target."""
        try:
            tprint_info("🔗 Using mutual information selection...")
            
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
                if mi_score is not None and mi_score > self.config.selection_threshold:
                    mi_scores[feature] = mi_score
            
            if not mi_scores:
                tprint_warning("⚠️ No features passed mutual information threshold")
                return self._create_empty_result()
            
            # Sort by mutual information score
            sorted_features = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Select top features
            selected_features = [feature for feature, _ in sorted_features[:max_features]]
            feature_scores = dict(sorted_features[:max_features])
            
            tprint_success(f"✅ Mutual information selection: {len(selected_features)} features")
            
            return {
                'success': True,
                'selected_features': selected_features,
                'feature_scores': feature_scores,
                'selection_scores': feature_scores,
                'total_features': len(feature_columns),
                'selected_count': len(selected_features),
                'selection_ratio': len(selected_features) / len(feature_columns)
            }
            
        except Exception as e:
            tprint_error(f"❌ Mutual information selection failed: {e}")
            return self._create_empty_result(str(e))
    
    def _select_by_correlation(self, 
                              data: pd.DataFrame, 
                              feature_columns: List[str], 
                              target_column: str,
                              max_features: int) -> Dict[str, Any]:
        """Select features using correlation with target."""
        try:
            tprint_info("📊 Using correlation selection...")
            
            # Calculate correlation matrix
            feature_data = data[feature_columns + [target_column]].dropna()
            
            if feature_data.empty:
                tprint_warning("⚠️ No valid data for correlation analysis")
                return self._create_empty_result()
            
            correlation_matrix = feature_data.corr()
            target_correlations = correlation_matrix[target_column].drop(target_column)
            
            # Filter by correlation threshold
            valid_correlations = target_correlations[
                (abs(target_correlations) > self.config.selection_threshold) &
                (abs(target_correlations) < self.config.max_correlation)
            ]
            
            if valid_correlations.empty:
                tprint_warning("⚠️ No features passed correlation thresholds")
                return self._create_empty_result()
            
            # Sort by absolute correlation
            sorted_correlations = valid_correlations.abs().sort_values(ascending=False)
            
            # Select top features
            selected_features = sorted_correlations.head(max_features).index.tolist()
            feature_scores = sorted_correlations.head(max_features).to_dict()
            
            tprint_success(f"✅ Correlation selection: {len(selected_features)} features")
            
            return {
                'success': True,
                'selected_features': selected_features,
                'feature_scores': feature_scores,
                'selection_scores': feature_scores,
                'total_features': len(feature_columns),
                'selected_count': len(selected_features),
                'selection_ratio': len(selected_features) / len(feature_columns)
            }
            
        except Exception as e:
            tprint_error(f"❌ Correlation selection failed: {e}")
            return self._create_empty_result(str(e))
    
    def _select_by_variance(self, 
                           data: pd.DataFrame, 
                           feature_columns: List[str],
                           max_features: int) -> Dict[str, Any]:
        """Select features using variance threshold."""
        try:
            tprint_info("📈 Using variance selection...")
            
            # Calculate variance for each feature
            variances = data[feature_columns].var()
            
            # Filter by variance threshold
            valid_variances = variances[variances > self.config.min_variance]
            
            if valid_variances.empty:
                tprint_warning("⚠️ No features passed variance threshold")
                return self._create_empty_result()
            
            # Sort by variance
            sorted_variances = valid_variances.sort_values(ascending=False)
            
            # Select top features
            selected_features = sorted_variances.head(max_features).index.tolist()
            feature_scores = sorted_variances.head(max_features).to_dict()
            
            tprint_success(f"✅ Variance selection: {len(selected_features)} features")
            
            return {
                'success': True,
                'selected_features': selected_features,
                'feature_scores': feature_scores,
                'selection_scores': feature_scores,
                'total_features': len(feature_columns),
                'selected_count': len(selected_features),
                'selection_ratio': len(selected_features) / len(feature_columns)
            }
            
        except Exception as e:
            tprint_error(f"❌ Variance selection failed: {e}")
            return self._create_empty_result(str(e))
    
    def _select_recursive(self, 
                         data: pd.DataFrame, 
                         feature_columns: List[str], 
                         target_column: str,
                         max_features: int) -> Dict[str, Any]:
        """Select features using recursive feature elimination."""
        try:
            tprint_info("🔄 Using recursive selection...")
            
            # For now, use a simplified recursive approach
            # In a full implementation, this would use sklearn's RFE
            
            target_data = data[target_column].dropna()
            remaining_features = feature_columns.copy()
            selected_features = []
            feature_scores = {}
            
            while len(selected_features) < max_features and remaining_features:
                # Calculate mutual information for remaining features
                mi_scores = {}
                
                for feature in remaining_features:
                    feature_data = data[feature].dropna()
                    
                    # Align data
                    common_index = target_data.index.intersection(feature_data.index)
                    if len(common_index) < 10:
                        continue
                    
                    aligned_target = target_data.loc[common_index]
                    aligned_feature = feature_data.loc[common_index]
                    
                    # Calculate mutual information
                    mi_score = self._calculate_mutual_information(aligned_target, aligned_feature)
                    if mi_score is not None and mi_score > self.config.selection_threshold:
                        mi_scores[feature] = mi_score
                
                if not mi_scores:
                    break
                
                # Select best feature
                best_feature = max(mi_scores.items(), key=lambda x: x[1])
                selected_features.append(best_feature[0])
                feature_scores[best_feature[0]] = best_feature[1]
                
                # Remove selected feature from remaining
                remaining_features.remove(best_feature[0])
            
            if not selected_features:
                tprint_warning("⚠️ No features selected in recursive selection")
                return self._create_empty_result()
            
            tprint_success(f"✅ Recursive selection: {len(selected_features)} features")
            
            return {
                'success': True,
                'selected_features': selected_features,
                'feature_scores': feature_scores,
                'selection_scores': feature_scores,
                'total_features': len(feature_columns),
                'selected_count': len(selected_features),
                'selection_ratio': len(selected_features) / len(feature_columns)
            }
            
        except Exception as e:
            tprint_error(f"❌ Recursive selection failed: {e}")
            return self._create_empty_result(str(e))
    
    def _calculate_mutual_information(self, x: pd.Series, y: pd.Series) -> Optional[float]:
        """Calculate mutual information between two series."""
        try:
            if len(x) < 2 or len(y) < 2:
                return None
            
            # Simple mutual information calculation using correlation
            # In a full implementation, this would use proper mutual information
            correlation = x.corr(y)
            if correlation is None or np.isnan(correlation):
                return None
            
            # Convert correlation to mutual information approximation
            # MI ≈ -0.5 * log(1 - r²) where r is correlation
            mi = -0.5 * np.log(1 - correlation**2) if abs(correlation) < 0.999 else 0.0
            return mi if not np.isnan(mi) and np.isfinite(mi) else None
            
        except Exception:
            return None
    
    def _create_empty_result(self, error_message: str = None) -> Dict[str, Any]:
        """Create empty result for failed selection."""
        return {
            'success': False,
            'error_message': error_message,
            'selected_features': [],
            'feature_scores': {},
            'selection_scores': {},
            'total_features': 0,
            'selected_count': 0,
            'selection_ratio': 0.0
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()


# Export main classes
__all__ = [
    'FeatureSelector',
    'SelectionConfig',
    'SelectionResult'
]