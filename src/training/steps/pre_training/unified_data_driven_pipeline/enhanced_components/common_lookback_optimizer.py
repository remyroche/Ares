"""
Common Lookback Optimization Logic for UnifiedDataDrivenPipeline

This module provides common logic for lookback optimization across all feature types,
implementing the requirement to generate 2-3 informative but non-redundant periods
for each feature, using the most informative one for single features and multiple
for cross timeframe features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass
from enum import Enum
import logging
from collections import defaultdict
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
import warnings

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)


class OptimizationStrategy(Enum):
    """Strategy for lookback optimization."""
    SINGLE_BEST = "single_best"  # For single features - select most informative
    MULTIPLE_DIVERSE = "multiple_diverse"  # For cross timeframe - select multiple diverse
    ADAPTIVE = "adaptive"  # Automatically choose based on feature type


class InformativeMetric(Enum):
    """Metrics for measuring feature informativeness."""
    CORRELATION = "correlation"
    MUTUAL_INFORMATION = "mutual_information"
    SHARPE_RATIO = "sharpe_ratio"
    STABILITY = "stability"
    DIVERSITY = "diversity"
    COMBINED = "combined"


@dataclass
class LookbackOptimizationConfig:
    """Configuration for lookback optimization."""
    
    # Lookback period settings
    min_lookback: int = 5
    max_lookback: int = 100
    lookback_step: int = 5
    num_candidate_periods: int = 10  # Number of candidate periods to evaluate
    
    # Optimization settings
    num_informative_periods: int = 3  # Number of informative periods to select
    redundancy_threshold: float = 0.8  # Threshold for considering periods redundant
    informativeness_threshold: float = 0.1  # Minimum informativeness score
    
    # Metrics to use
    primary_metric: InformativeMetric = InformativeMetric.COMBINED
    secondary_metrics: List[InformativeMetric] = None
    
    # Cross timeframe specific settings
    cross_timeframe_min_periods: int = 2  # Minimum periods for cross timeframe
    cross_timeframe_max_periods: int = 5  # Maximum periods for cross timeframe
    
    # Single feature specific settings
    single_feature_strategy: OptimizationStrategy = OptimizationStrategy.SINGLE_BEST
    
    def __post_init__(self):
        if self.secondary_metrics is None:
            self.secondary_metrics = [
                InformativeMetric.CORRELATION,
                InformativeMetric.STABILITY,
                InformativeMetric.DIVERSITY
            ]


@dataclass
class LookbackOptimizationResult:
    """Result of lookback optimization."""
    
    optimized_periods: List[int]
    informativeness_scores: Dict[int, float]
    redundancy_matrix: np.ndarray
    selected_features: List[Dict[str, Any]]
    optimization_metadata: Dict[str, Any]


class CommonLookbackOptimizer:
    """Common lookback optimization logic for all feature types."""
    
    def __init__(self, config: LookbackOptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def optimize_lookback_periods(
        self, 
        features: List[Dict[str, Any]], 
        targets: Optional[pd.Series] = None,
        feature_type: str = "unknown"
    ) -> LookbackOptimizationResult:
        """
        Optimize lookback periods for features.
        
        For single features: select the most informative period
        For cross timeframe features: select multiple informative but non-redundant periods
        
        Args:
            features: List of feature dictionaries with different lookback periods
            targets: Target series for informativeness calculation
            feature_type: Type of features being optimized
            
        Returns:
            LookbackOptimizationResult with optimized periods and metadata
        """
        if not features:
            return LookbackOptimizationResult(
                optimized_periods=[],
                informativeness_scores={},
                redundancy_matrix=np.array([]),
                selected_features=[],
                optimization_metadata={}
            )
        
        try:
            tprint_debug(f"Optimizing lookback periods for {len(features)} {feature_type} features")
            
            # Calculate informativeness scores for all features
            informativeness_scores = self._calculate_informativeness_scores(features, targets)
            
            # Calculate redundancy matrix
            redundancy_matrix = self._calculate_redundancy_matrix(features)
            
            # Select optimal periods based on feature type
            if feature_type == "cross_timeframe":
                selected_features, optimized_periods = self._optimize_cross_timeframe_periods(
                    features, informativeness_scores, redundancy_matrix
                )
            else:
                selected_features, optimized_periods = self._optimize_single_feature_periods(
                    features, informativeness_scores, redundancy_matrix
                )
            
            # Create optimization metadata
            optimization_metadata = self._create_optimization_metadata(
                features, selected_features, informativeness_scores, redundancy_matrix
            )
            
            result = LookbackOptimizationResult(
                optimized_periods=optimized_periods,
                informativeness_scores=informativeness_scores,
                redundancy_matrix=redundancy_matrix,
                selected_features=selected_features,
                optimization_metadata=optimization_metadata
            )
            
            tprint_debug(f"Optimized to {len(optimized_periods)} periods: {optimized_periods}")
            return result
            
        except Exception as e:
            tprint_error(f"Error optimizing lookback periods: {e}")
            return LookbackOptimizationResult(
                optimized_periods=[],
                informativeness_scores={},
                redundancy_matrix=np.array([]),
                selected_features=[],
                optimization_metadata={"error": str(e)}
            )
    
    def _calculate_informativeness_scores(
        self, 
        features: List[Dict[str, Any]], 
        targets: Optional[pd.Series] = None
    ) -> Dict[int, float]:
        """Calculate informativeness scores for all features."""
        scores = {}
        
        try:
            for i, feature in enumerate(features):
                lookback_period = feature.get('lookback_period', i)
                feature_series = feature.get('series')
                
                if feature_series is None:
                    scores[lookback_period] = 0.0
                    continue
                
                # Calculate combined informativeness score
                score = self._calculate_combined_informativeness_score(feature_series, targets)
                scores[lookback_period] = score
            
            return scores
            
        except Exception as e:
            tprint_debug(f"Error calculating informativeness scores: {e}")
            return {i: 0.0 for i in range(len(features))}
    
    def _calculate_combined_informativeness_score(
        self, 
        feature_series: pd.Series, 
        targets: Optional[pd.Series] = None
    ) -> float:
        """Calculate combined informativeness score using multiple metrics."""
        try:
            scores = []
            
            # Correlation score
            if targets is not None:
                corr_score = self._calculate_correlation_score(feature_series, targets)
                scores.append(corr_score)
            
            # Mutual information score
            if targets is not None:
                mi_score = self._calculate_mutual_information_score(feature_series, targets)
                scores.append(mi_score)
            
            # Stability score
            stability_score = self._calculate_stability_score(feature_series)
            scores.append(stability_score)
            
            # Diversity score (variance)
            diversity_score = self._calculate_diversity_score(feature_series)
            scores.append(diversity_score)
            
            # Combine scores (weighted average)
            if scores:
                weights = [0.3, 0.2, 0.25, 0.25]  # Adjust based on importance
                combined_score = np.average(scores[:len(weights)], weights=weights[:len(scores)])
                return float(combined_score)
            else:
                return 0.0
                
        except Exception as e:
            tprint_debug(f"Error calculating combined informativeness score: {e}")
            return 0.0
    
    def _calculate_correlation_score(self, feature_series: pd.Series, targets: pd.Series) -> float:
        """Calculate correlation-based informativeness score."""
        try:
            # Align series
            aligned_feature = feature_series.dropna()
            aligned_targets = targets.reindex(aligned_feature.index).dropna()
            
            if len(aligned_feature) < 10 or len(aligned_targets) < 10:
                return 0.0
            
            correlation = np.corrcoef(aligned_feature, aligned_targets)[0, 1]
            
            if np.isnan(correlation):
                return 0.0
            
            return abs(correlation)
            
        except Exception as e:
            tprint_debug(f"Error calculating correlation score: {e}")
            return 0.0
    
    def _calculate_mutual_information_score(self, feature_series: pd.Series, targets: pd.Series) -> float:
        """Calculate mutual information-based informativeness score."""
        try:
            # Align series
            aligned_feature = feature_series.dropna()
            aligned_targets = targets.reindex(aligned_feature.index).dropna()
            
            if len(aligned_feature) < 10 or len(aligned_targets) < 10:
                return 0.0
            
            # Discretize for mutual information calculation
            feature_discrete = pd.cut(aligned_feature, bins=10, labels=False, duplicates='drop')
            targets_discrete = pd.cut(aligned_targets, bins=10, labels=False, duplicates='drop')
            
            # Remove NaN values
            valid_mask = ~(np.isnan(feature_discrete) | np.isnan(targets_discrete))
            feature_discrete = feature_discrete[valid_mask]
            targets_discrete = targets_discrete[valid_mask]
            
            if len(feature_discrete) < 10:
                return 0.0
            
            # Calculate mutual information
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mi_score = mutual_info_regression(
                    feature_discrete.reshape(-1, 1), 
                    targets_discrete,
                    discrete_features=True
                )[0]
            
            return float(mi_score)
            
        except Exception as e:
            tprint_debug(f"Error calculating mutual information score: {e}")
            return 0.0
    
    def _calculate_stability_score(self, feature_series: pd.Series) -> float:
        """Calculate stability score (inverse of coefficient of variation)."""
        try:
            if len(feature_series) < 10:
                return 0.0
            
            # Calculate rolling mean and std
            rolling_mean = feature_series.rolling(window=min(20, len(feature_series)//4)).mean()
            rolling_std = feature_series.rolling(window=min(20, len(feature_series)//4)).std()
            
            # Calculate coefficient of variation
            cv = rolling_std / (rolling_mean + 1e-8)
            cv_mean = cv.mean()
            
            # Stability is inverse of CV (higher is more stable)
            stability = 1.0 / (1.0 + cv_mean) if not np.isnan(cv_mean) else 0.0
            
            return float(stability)
            
        except Exception as e:
            tprint_debug(f"Error calculating stability score: {e}")
            return 0.0
    
    def _calculate_diversity_score(self, feature_series: pd.Series) -> float:
        """Calculate diversity score (normalized variance)."""
        try:
            if len(feature_series) < 10:
                return 0.0
            
            # Calculate normalized variance
            variance = feature_series.var()
            mean_abs = feature_series.abs().mean()
            
            if mean_abs == 0:
                return 0.0
            
            # Normalized variance (coefficient of variation squared)
            diversity = variance / (mean_abs ** 2)
            
            return float(diversity)
            
        except Exception as e:
            tprint_debug(f"Error calculating diversity score: {e}")
            return 0.0
    
    def _calculate_redundancy_matrix(self, features: List[Dict[str, Any]]) -> np.ndarray:
        """Calculate redundancy matrix between features."""
        try:
            n_features = len(features)
            redundancy_matrix = np.zeros((n_features, n_features))
            
            for i in range(n_features):
                for j in range(i+1, n_features):
                    series_i = features[i].get('series')
                    series_j = features[j].get('series')
                    
                    if series_i is None or series_j is None:
                        redundancy_matrix[i, j] = 1.0  # Consider as redundant
                        redundancy_matrix[j, i] = 1.0
                        continue
                    
                    # Calculate correlation between features
                    try:
                        aligned_i = series_i.dropna()
                        aligned_j = series_j.reindex(aligned_i.index).dropna()
                        
                        if len(aligned_i) < 10 or len(aligned_j) < 10:
                            redundancy_matrix[i, j] = 1.0
                            redundancy_matrix[j, i] = 1.0
                            continue
                        
                        correlation = np.corrcoef(aligned_i, aligned_j)[0, 1]
                        
                        if np.isnan(correlation):
                            redundancy_matrix[i, j] = 1.0
                            redundancy_matrix[j, i] = 1.0
                        else:
                            redundancy_matrix[i, j] = abs(correlation)
                            redundancy_matrix[j, i] = abs(correlation)
                            
                    except Exception as e:
                        tprint_debug(f"Error calculating correlation between features {i} and {j}: {e}")
                        redundancy_matrix[i, j] = 1.0
                        redundancy_matrix[j, i] = 1.0
            
            return redundancy_matrix
            
        except Exception as e:
            tprint_debug(f"Error calculating redundancy matrix: {e}")
            return np.ones((len(features), len(features)))
    
    def _optimize_single_feature_periods(
        self, 
        features: List[Dict[str, Any]], 
        informativeness_scores: Dict[int, float],
        redundancy_matrix: np.ndarray
    ) -> Tuple[List[Dict[str, Any]], List[int]]:
        """Optimize periods for single features (select most informative)."""
        try:
            # Sort features by informativeness score
            sorted_features = sorted(
                features, 
                key=lambda x: informativeness_scores.get(x.get('lookback_period', 0), 0),
                reverse=True
            )
            
            # Select the most informative feature
            best_feature = sorted_features[0]
            selected_features = [best_feature]
            optimized_periods = [best_feature.get('lookback_period', 0)]
            
            # Add metadata
            best_feature['optimization_type'] = 'single_best'
            best_feature['informativeness_score'] = informativeness_scores.get(
                best_feature.get('lookback_period', 0), 0
            )
            best_feature['total_candidates'] = len(features)
            
            return selected_features, optimized_periods
            
        except Exception as e:
            tprint_debug(f"Error optimizing single feature periods: {e}")
            return features[:1], [features[0].get('lookback_period', 0)]
    
    def _optimize_cross_timeframe_periods(
        self, 
        features: List[Dict[str, Any]], 
        informativeness_scores: Dict[int, float],
        redundancy_matrix: np.ndarray
    ) -> Tuple[List[Dict[str, Any]], List[int]]:
        """Optimize periods for cross timeframe features (select multiple diverse)."""
        try:
            # Sort features by informativeness score
            sorted_features = sorted(
                features, 
                key=lambda x: informativeness_scores.get(x.get('lookback_period', 0), 0),
                reverse=True
            )
            
            selected_features = []
            optimized_periods = []
            selected_indices = set()
            
            # Select features ensuring diversity
            for feature in sorted_features:
                if len(selected_features) >= self.config.cross_timeframe_max_periods:
                    break
                
                feature_idx = features.index(feature)
                lookback_period = feature.get('lookback_period', 0)
                
                # Check if this feature is too redundant with already selected features
                is_redundant = False
                for selected_idx in selected_indices:
                    if redundancy_matrix[feature_idx, selected_idx] > self.config.redundancy_threshold:
                        is_redundant = True
                        break
                
                # Add feature if not redundant and meets informativeness threshold
                informativeness_score = informativeness_scores.get(lookback_period, 0)
                if not is_redundant and informativeness_score >= self.config.informativeness_threshold:
                    selected_features.append(feature)
                    optimized_periods.append(lookback_period)
                    selected_indices.add(feature_idx)
            
            # Ensure we have at least minimum number of periods
            if len(selected_features) < self.config.cross_timeframe_min_periods:
                # Add more features even if somewhat redundant
                for feature in sorted_features:
                    if len(selected_features) >= self.config.cross_timeframe_min_periods:
                        break
                    
                    feature_idx = features.index(feature)
                    if feature_idx not in selected_indices:
                        selected_features.append(feature)
                        optimized_periods.append(feature.get('lookback_period', 0))
                        selected_indices.add(feature_idx)
            
            # Add metadata to selected features
            for i, feature in enumerate(selected_features):
                feature['optimization_type'] = 'cross_timeframe_multiple'
                feature['selection_rank'] = i + 1
                feature['informativeness_score'] = informativeness_scores.get(
                    feature.get('lookback_period', 0), 0
                )
                feature['total_candidates'] = len(features)
            
            return selected_features, optimized_periods
            
        except Exception as e:
            tprint_debug(f"Error optimizing cross timeframe periods: {e}")
            return features[:min(3, len(features))], [
                f.get('lookback_period', 0) for f in features[:min(3, len(features))]
            ]
    
    def _create_optimization_metadata(
        self, 
        original_features: List[Dict[str, Any]], 
        selected_features: List[Dict[str, Any]],
        informativeness_scores: Dict[int, float],
        redundancy_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """Create metadata about the optimization process."""
        try:
            metadata = {
                'total_candidates': len(original_features),
                'selected_count': len(selected_features),
                'reduction_ratio': len(selected_features) / len(original_features) if original_features else 0,
                'avg_informativeness': np.mean(list(informativeness_scores.values())) if informativeness_scores else 0,
                'max_informativeness': max(informativeness_scores.values()) if informativeness_scores else 0,
                'min_informativeness': min(informativeness_scores.values()) if informativeness_scores else 0,
                'avg_redundancy': np.mean(redundancy_matrix[np.triu_indices_from(redundancy_matrix, k=1)]) if redundancy_matrix.size > 0 else 0,
                'optimization_timestamp': pd.Timestamp.now().isoformat()
            }
            
            return metadata
            
        except Exception as e:
            tprint_debug(f"Error creating optimization metadata: {e}")
            return {'error': str(e)}


def create_common_lookback_optimizer(config: Optional[LookbackOptimizationConfig] = None) -> CommonLookbackOptimizer:
    """Create a common lookback optimizer instance."""
    if config is None:
        config = LookbackOptimizationConfig()
    return CommonLookbackOptimizer(config)