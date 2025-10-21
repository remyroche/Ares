"""
Data-Driven Regime Merging Threshold Optimization

This module provides optimization of regime merging thresholds using various
strategies including Bayesian TPE, hierarchical clustering, and statistical validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
import logging
from dataclasses import dataclass
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ks_2samp, ttest_ind, mannwhitneyu
import warnings

# Import optimization utilities
try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
        BayesianTPEOptimizer, OptimizationConfig
    )
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    logging.warning("Bayesian TPE optimizer not available")

from ..config.data_driven_config import (
    RegimeMergingThresholdConfig, ValidationMetric, OptimizationStrategy
)

logger = logging.getLogger(__name__)


@dataclass
class RegimeMergingThresholdResult:
    """Result of regime merging threshold optimization."""
    optimal_thresholds: Dict[str, float]
    optimization_score: float
    validation_scores: Dict[str, float]
    merging_statistics: Dict[str, Any]
    optimization_history: List[Dict[str, Any]]
    convergence_info: Dict[str, Any]
    metadata: Dict[str, Any]


class DataDrivenMergingThresholdOptimizer:
    """
    Data-driven optimizer for regime merging thresholds.
    
    Replaces hardcoded thresholds (similarity_threshold=0.8, distance_threshold=0.2, 
    p_value_threshold=0.05) with data-driven optimization based on clustering quality.
    """
    
    def __init__(self, config: RegimeMergingThresholdConfig):
        """
        Initialize the merging threshold optimizer.
        
        Args:
            config: Configuration for merging threshold optimization
        """
        self.config = config
        self.optimization_history = []
        self.best_thresholds = None
        self.best_score = -np.inf
        
    def optimize_thresholds(self, 
                           cluster_labels: np.ndarray,
                           features: np.ndarray,
                           merging_func: Callable) -> RegimeMergingThresholdResult:
        """
        Optimize merging thresholds using the specified strategy.
        
        Args:
            cluster_labels: Initial cluster labels
            features: Feature matrix
            merging_func: Function that performs merging given thresholds
            
        Returns:
            RegimeMergingThresholdResult with optimal thresholds and metadata
        """
        try:
            logger.info("🔗 Starting data-driven merging threshold optimization...")
            
            # Validate input
            if not self._validate_input(cluster_labels, features):
                raise ValueError("Invalid input data for threshold optimization")
            
            # Optimize thresholds based on strategy
            if self.config.optimization_strategy == OptimizationStrategy.BAYESIAN_TPE:
                optimal_thresholds, optimization_info = self._optimize_with_tpe(
                    cluster_labels, features, merging_func
                )
            elif self.config.optimization_strategy == OptimizationStrategy.GRID_SEARCH:
                optimal_thresholds, optimization_info = self._optimize_with_grid_search(
                    cluster_labels, features, merging_func
                )
            elif self.config.optimization_strategy == OptimizationStrategy.RANDOM_SEARCH:
                optimal_thresholds, optimization_info = self._optimize_with_random_search(
                    cluster_labels, features, merging_func
                )
            elif self.config.optimization_strategy == OptimizationStrategy.ADAPTIVE:
                optimal_thresholds, optimization_info = self._optimize_adaptively(
                    cluster_labels, features, merging_func
                )
            else:
                raise ValueError(f"Unknown optimization strategy: {self.config.optimization_strategy}")
            
            # Validate optimal thresholds
            validation_scores = self._validate_thresholds(
                optimal_thresholds, cluster_labels, features, merging_func
            )
            
            # Calculate merging statistics
            merging_stats = self._calculate_merging_statistics(
                optimal_thresholds, cluster_labels, features, merging_func
            )
            
            # Create result
            result = RegimeMergingThresholdResult(
                optimal_thresholds=optimal_thresholds,
                optimization_score=optimization_info.get('best_score', 0.0),
                validation_scores=validation_scores,
                merging_statistics=merging_stats,
                optimization_history=self.optimization_history,
                convergence_info=optimization_info,
                metadata={
                    'config': self.config.__dict__,
                    'n_samples': features.shape[0],
                    'n_features': features.shape[1],
                    'initial_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                }
            )
            
            logger.info(f"✅ Merging threshold optimization completed. Best score: {result.optimization_score:.4f}")
            logger.info(f"📈 Optimal thresholds: {optimal_thresholds}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Merging threshold optimization failed: {e}")
            raise
    
    def _validate_input(self, cluster_labels: np.ndarray, features: np.ndarray) -> bool:
        """Validate input data for threshold optimization."""
        try:
            # Check for sufficient samples
            if len(cluster_labels) < 10:
                logger.warning("⚠️ Insufficient samples for threshold optimization")
                return False
            
            # Check for sufficient clusters
            unique_labels = np.unique(cluster_labels)
            unique_labels = unique_labels[unique_labels != -1]  # Remove noise
            if len(unique_labels) < 2:
                logger.warning("⚠️ Insufficient clusters for threshold optimization")
                return False
            
            # Check for NaN or infinite values
            if np.isnan(features).any() or np.isinf(features).any():
                logger.warning("⚠️ Found NaN or infinite values in features")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Input validation failed: {e}")
            return False
    
    def _optimize_with_tpe(self, 
                          cluster_labels: np.ndarray,
                          features: np.ndarray,
                          merging_func: Callable) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Optimize thresholds using Bayesian TPE."""
        if not OPTIMIZATION_AVAILABLE:
            raise ImportError("Bayesian TPE optimizer not available")
        
        def objective(trial):
            # Sample thresholds
            similarity_threshold = trial.suggest_float(
                'similarity_threshold',
                self.config.similarity_threshold_range[0],
                self.config.similarity_threshold_range[1]
            )
            distance_threshold = trial.suggest_float(
                'distance_threshold',
                self.config.distance_threshold_range[0],
                self.config.distance_threshold_range[1]
            )
            p_value_threshold = trial.suggest_float(
                'p_value_threshold',
                self.config.p_value_threshold_range[0],
                self.config.p_value_threshold_range[1]
            )
            
            thresholds = {
                'similarity_threshold': similarity_threshold,
                'distance_threshold': distance_threshold,
                'p_value_threshold': p_value_threshold
            }
            
            # Apply merging and evaluate
            try:
                merged_labels = merging_func(cluster_labels, features, thresholds)
                score = self._calculate_quality_score(features, merged_labels)
                
                # Store trial info
                self.optimization_history.append({
                    'trial': len(self.optimization_history),
                    'thresholds': thresholds.copy(),
                    'score': score,
                    'timestamp': pd.Timestamp.now()
                })
                
                return score
                
            except Exception as e:
                logger.debug(f"Trial failed: {e}")
                return -np.inf
        
        # Create optimization config
        opt_config = OptimizationConfig(
            n_trials=self.config.n_trials,
            timeout=self.config.timeout_seconds,
            n_startup_trials=self.config.n_startup_trials,
            direction='maximize',
            metric_name='quality_score'
        )
        
        # Run optimization
        optimizer = BayesianTPEOptimizer(opt_config)
        best_params, best_score = optimizer.optimize(objective)
        
        # Extract optimal thresholds
        optimal_thresholds = {
            'similarity_threshold': best_params.get('similarity_threshold', 0.8),
            'distance_threshold': best_params.get('distance_threshold', 0.2),
            'p_value_threshold': best_params.get('p_value_threshold', 0.05)
        }
        
        return optimal_thresholds, {'best_score': best_score, 'n_trials': len(self.optimization_history)}
    
    def _optimize_with_grid_search(self, 
                                  cluster_labels: np.ndarray,
                                  features: np.ndarray,
                                  merging_func: Callable) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Optimize thresholds using grid search."""
        # Create grid of threshold combinations
        similarity_values = np.linspace(
            self.config.similarity_threshold_range[0],
            self.config.similarity_threshold_range[1],
            5
        )
        distance_values = np.linspace(
            self.config.distance_threshold_range[0],
            self.config.distance_threshold_range[1],
            5
        )
        p_value_values = np.linspace(
            self.config.p_value_threshold_range[0],
            self.config.p_value_threshold_range[1],
            5
        )
        
        best_score = -np.inf
        best_thresholds = None
        
        # Generate all combinations
        for sim_thresh, dist_thresh, p_thresh in itertools.product(similarity_values, distance_values, p_value_values):
            thresholds = {
                'similarity_threshold': sim_thresh,
                'distance_threshold': dist_thresh,
                'p_value_threshold': p_thresh
            }
            
            # Apply merging and evaluate
            try:
                merged_labels = merging_func(cluster_labels, features, thresholds)
                score = self._calculate_quality_score(features, merged_labels)
                
                if score > best_score:
                    best_score = score
                    best_thresholds = thresholds.copy()
                
                self.optimization_history.append({
                    'trial': len(self.optimization_history),
                    'thresholds': thresholds.copy(),
                    'score': score,
                    'timestamp': pd.Timestamp.now()
                })
                
            except Exception as e:
                logger.debug(f"Grid search trial failed: {e}")
                continue
        
        return best_thresholds or {
            'similarity_threshold': 0.8,
            'distance_threshold': 0.2,
            'p_value_threshold': 0.05
        }, {'best_score': best_score}
    
    def _optimize_with_random_search(self, 
                                   cluster_labels: np.ndarray,
                                   features: np.ndarray,
                                   merging_func: Callable) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Optimize thresholds using random search."""
        best_score = -np.inf
        best_thresholds = None
        
        for trial in range(self.config.n_trials):
            # Sample random thresholds
            thresholds = {
                'similarity_threshold': np.random.uniform(*self.config.similarity_threshold_range),
                'distance_threshold': np.random.uniform(*self.config.distance_threshold_range),
                'p_value_threshold': np.random.uniform(*self.config.p_value_threshold_range)
            }
            
            # Apply merging and evaluate
            try:
                merged_labels = merging_func(cluster_labels, features, thresholds)
                score = self._calculate_quality_score(features, merged_labels)
                
                if score > best_score:
                    best_score = score
                    best_thresholds = thresholds.copy()
                
                self.optimization_history.append({
                    'trial': trial,
                    'thresholds': thresholds.copy(),
                    'score': score,
                    'timestamp': pd.Timestamp.now()
                })
                
            except Exception as e:
                logger.debug(f"Random search trial failed: {e}")
                continue
        
        return best_thresholds or {
            'similarity_threshold': 0.8,
            'distance_threshold': 0.2,
            'p_value_threshold': 0.05
        }, {'best_score': best_score}
    
    def _optimize_adaptively(self, 
                           cluster_labels: np.ndarray,
                           features: np.ndarray,
                           merging_func: Callable) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Optimize thresholds adaptively based on data characteristics."""
        try:
            # Calculate cluster characteristics
            cluster_centers = self._calculate_cluster_centers(cluster_labels, features)
            cluster_distances = self._calculate_cluster_distances(cluster_centers)
            
            # Estimate optimal similarity threshold based on cluster distances
            median_distance = np.median(cluster_distances)
            similarity_threshold = np.clip(1.0 - median_distance, 
                                         self.config.similarity_threshold_range[0],
                                         self.config.similarity_threshold_range[1])
            
            # Estimate optimal distance threshold based on feature variance
            feature_variance = np.var(features, axis=0)
            mean_variance = np.mean(feature_variance)
            distance_threshold = np.clip(mean_variance * 0.1,
                                       self.config.distance_threshold_range[0],
                                       self.config.distance_threshold_range[1])
            
            # Estimate optimal p-value threshold based on sample size
            n_samples = len(cluster_labels)
            if n_samples < 100:
                p_value_threshold = 0.1
            elif n_samples < 1000:
                p_value_threshold = 0.05
            else:
                p_value_threshold = 0.01
            
            p_value_threshold = np.clip(p_value_threshold,
                                      self.config.p_value_threshold_range[0],
                                      self.config.p_value_threshold_range[1])
            
            thresholds = {
                'similarity_threshold': similarity_threshold,
                'distance_threshold': distance_threshold,
                'p_value_threshold': p_value_threshold
            }
            
            # Fine-tune with local optimization
            def objective(thresholds_array):
                sim_thresh, dist_thresh, p_thresh = thresholds_array
                temp_thresholds = {
                    'similarity_threshold': sim_thresh,
                    'distance_threshold': dist_thresh,
                    'p_value_threshold': p_thresh
                }
                
                try:
                    merged_labels = merging_func(cluster_labels, features, temp_thresholds)
                    score = self._calculate_quality_score(features, merged_labels)
                    return -score  # Minimize negative score
                except:
                    return np.inf
            
            # Initial thresholds
            initial_thresholds = np.array([similarity_threshold, distance_threshold, p_value_threshold])
            
            # Bounds
            bounds = [
                self.config.similarity_threshold_range,
                self.config.distance_threshold_range,
                self.config.p_value_threshold_range
            ]
            
            # Optimize
            result = minimize(objective, initial_thresholds, method='L-BFGS-B', bounds=bounds)
            
            optimal_thresholds = {
                'similarity_threshold': result.x[0],
                'distance_threshold': result.x[1],
                'p_value_threshold': result.x[2]
            }
            
            return optimal_thresholds, {'best_score': -result.fun, 'converged': result.success}
            
        except Exception as e:
            logger.warning(f"Adaptive optimization failed: {e}")
            return {
                'similarity_threshold': 0.8,
                'distance_threshold': 0.2,
                'p_value_threshold': 0.05
            }, {'best_score': 0.0, 'converged': False}
    
    def _calculate_cluster_centers(self, cluster_labels: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Calculate cluster centers."""
        try:
            unique_labels = np.unique(cluster_labels)
            unique_labels = unique_labels[unique_labels != -1]  # Remove noise
            
            centers = []
            for label in unique_labels:
                mask = cluster_labels == label
                if mask.sum() > 0:
                    center = np.mean(features[mask], axis=0)
                    centers.append(center)
            
            return np.array(centers)
            
        except Exception as e:
            logger.debug(f"Cluster center calculation failed: {e}")
            return np.array([])
    
    def _calculate_cluster_distances(self, cluster_centers: np.ndarray) -> np.ndarray:
        """Calculate pairwise distances between cluster centers."""
        try:
            if len(cluster_centers) < 2:
                return np.array([1.0])
            
            distances = pdist(cluster_centers, metric='euclidean')
            return distances
            
        except Exception as e:
            logger.debug(f"Cluster distance calculation failed: {e}")
            return np.array([1.0])
    
    def _calculate_quality_score(self, features: np.ndarray, cluster_labels: np.ndarray) -> float:
        """Calculate combined quality score for merging evaluation."""
        try:
            # Remove noise points
            valid_mask = cluster_labels != -1
            if valid_mask.sum() < 2:
                return -np.inf
            
            valid_labels = cluster_labels[valid_mask]
            valid_features = features[valid_mask]
            
            if len(set(valid_labels)) < 2:
                return -np.inf
            
            # Calculate primary metric
            if self.config.primary_metric == ValidationMetric.SILHOUETTE:
                primary_score = silhouette_score(valid_features, valid_labels)
            elif self.config.primary_metric == ValidationMetric.DAVIES_BOULDIN:
                primary_score = -davies_bouldin_score(valid_features, valid_labels)  # Negative because lower is better
            elif self.config.primary_metric == ValidationMetric.CALINSKI_HARABASZ:
                primary_score = calinski_harabasz_score(valid_features, valid_labels)
            else:
                primary_score = silhouette_score(valid_features, valid_labels)
            
            # Calculate secondary metrics
            secondary_scores = []
            for metric in self.config.secondary_metrics:
                if metric == ValidationMetric.SILHOUETTE:
                    secondary_scores.append(silhouette_score(valid_features, valid_labels))
                elif metric == ValidationMetric.DAVIES_BOULDIN:
                    secondary_scores.append(-davies_bouldin_score(valid_features, valid_labels))
                elif metric == ValidationMetric.CALINSKI_HARABASZ:
                    secondary_scores.append(calinski_harabasz_score(valid_features, valid_labels))
            
            # Combine scores
            combined_score = primary_score
            if secondary_scores:
                combined_score += 0.3 * np.mean(secondary_scores)
            
            # Add stability bonus for reasonable cluster count
            n_clusters = len(set(valid_labels))
            if self.config.min_clusters_after_merge <= n_clusters <= self.config.max_clusters_after_merge:
                combined_score += 0.1
            
            return combined_score
            
        except Exception as e:
            logger.debug(f"Quality score calculation failed: {e}")
            return -np.inf
    
    def _validate_thresholds(self, 
                           thresholds: Dict[str, float],
                           cluster_labels: np.ndarray,
                           features: np.ndarray,
                           merging_func: Callable) -> Dict[str, float]:
        """Validate optimal thresholds."""
        try:
            # Apply thresholds
            merged_labels = merging_func(cluster_labels, features, thresholds)
            
            # Calculate validation metrics
            valid_mask = merged_labels != -1
            if valid_mask.sum() < 2:
                return {'silhouette': -1.0, 'davies_bouldin': 10.0, 'calinski_harabasz': 0.0}
            
            valid_labels = merged_labels[valid_mask]
            valid_features = features[valid_mask]
            
            if len(set(valid_labels)) < 2:
                return {'silhouette': -1.0, 'davies_bouldin': 10.0, 'calinski_harabasz': 0.0}
            
            return {
                'silhouette': silhouette_score(valid_features, valid_labels),
                'davies_bouldin': davies_bouldin_score(valid_features, valid_labels),
                'calinski_harabasz': calinski_harabasz_score(valid_features, valid_labels)
            }
            
        except Exception as e:
            logger.warning(f"Threshold validation failed: {e}")
            return {'silhouette': -1.0, 'davies_bouldin': 10.0, 'calinski_harabasz': 0.0}
    
    def _calculate_merging_statistics(self, 
                                    thresholds: Dict[str, float],
                                    cluster_labels: np.ndarray,
                                    features: np.ndarray,
                                    merging_func: Callable) -> Dict[str, Any]:
        """Calculate merging statistics."""
        try:
            # Apply thresholds
            merged_labels = merging_func(cluster_labels, features, thresholds)
            
            # Calculate statistics
            n_samples = len(cluster_labels)
            original_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            merged_clusters = len(set(merged_labels)) - (1 if -1 in merged_labels else 0)
            
            return {
                'n_samples': n_samples,
                'original_clusters': original_clusters,
                'merged_clusters': merged_clusters,
                'clusters_merged': original_clusters - merged_clusters,
                'merging_rate': (original_clusters - merged_clusters) / max(original_clusters, 1),
                'thresholds_used': thresholds
            }
            
        except Exception as e:
            logger.warning(f"Merging statistics calculation failed: {e}")
            return {'error': str(e)}