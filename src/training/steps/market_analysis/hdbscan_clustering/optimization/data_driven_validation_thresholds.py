"""
Data-Driven Cluster Validation Threshold Optimization

This module provides optimization of cluster validation thresholds using various
strategies including Bayesian TPE, permutation testing, and bootstrap validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
import logging
from dataclasses import dataclass
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans
from scipy.stats import percentileofscore
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
    ClusterValidationThresholdConfig, ValidationMetric, OptimizationStrategy
)

logger = logging.getLogger(__name__)


@dataclass
class ClusterValidationThresholdResult:
    """Result of cluster validation threshold optimization."""
    optimal_thresholds: Dict[str, float]
    optimization_score: float
    validation_scores: Dict[str, float]
    statistical_validation: Dict[str, Any]
    bootstrap_validation: Dict[str, Any]
    optimization_history: List[Dict[str, Any]]
    convergence_info: Dict[str, Any]
    metadata: Dict[str, Any]


class DataDrivenValidationThresholdOptimizer:
    """
    Data-driven optimizer for cluster validation thresholds.
    
    Replaces hardcoded thresholds (min_silhouette=0.2, max_dbi=2.5) with
    data-driven optimization based on statistical significance and bootstrap validation.
    """
    
    def __init__(self, config: ClusterValidationThresholdConfig):
        """
        Initialize the validation threshold optimizer.
        
        Args:
            config: Configuration for validation threshold optimization
        """
        self.config = config
        self.optimization_history = []
        self.best_thresholds = None
        self.best_score = -np.inf
        
    def optimize_thresholds(self, 
                           features: np.ndarray,
                           clustering_func: Callable,
                           economic_validation_func: Optional[Callable] = None) -> ClusterValidationThresholdResult:
        """
        Optimize validation thresholds using the specified strategy.
        
        Args:
            features: Feature matrix for clustering
            clustering_func: Function that performs clustering
            economic_validation_func: Optional function for economic validation
            
        Returns:
            ClusterValidationThresholdResult with optimal thresholds and metadata
        """
        try:
            logger.info("📊 Starting data-driven validation threshold optimization...")
            
            # Calculate baseline metrics
            baseline_metrics = self._calculate_baseline_metrics(features, clustering_func)
            logger.info(f"📈 Baseline metrics: {baseline_metrics}")
            
            # Calculate null distributions for statistical validation
            null_distributions = self._calculate_null_distributions(features)
            
            # Optimize thresholds based on strategy
            if self.config.optimization_strategy == OptimizationStrategy.BAYESIAN_TPE:
                optimal_thresholds, optimization_info = self._optimize_with_tpe(
                    features, baseline_metrics, null_distributions, clustering_func, economic_validation_func
                )
            elif self.config.optimization_strategy == OptimizationStrategy.GRID_SEARCH:
                optimal_thresholds, optimization_info = self._optimize_with_grid_search(
                    features, baseline_metrics, null_distributions, clustering_func, economic_validation_func
                )
            elif self.config.optimization_strategy == OptimizationStrategy.RANDOM_SEARCH:
                optimal_thresholds, optimization_info = self._optimize_with_random_search(
                    features, baseline_metrics, null_distributions, clustering_func, economic_validation_func
                )
            elif self.config.optimization_strategy == OptimizationStrategy.ADAPTIVE:
                optimal_thresholds, optimization_info = self._optimize_adaptively(
                    features, baseline_metrics, null_distributions, clustering_func, economic_validation_func
                )
            else:
                raise ValueError(f"Unknown optimization strategy: {self.config.optimization_strategy}")
            
            # Validate optimal thresholds
            validation_scores = self._validate_thresholds(
                optimal_thresholds, features, clustering_func
            )
            
            # Calculate statistical validation
            statistical_validation = self._calculate_statistical_validation(
                optimal_thresholds, null_distributions
            )
            
            # Calculate bootstrap validation
            bootstrap_validation = self._calculate_bootstrap_validation(
                optimal_thresholds, features, clustering_func
            )
            
            # Create result
            result = ClusterValidationThresholdResult(
                optimal_thresholds=optimal_thresholds,
                optimization_score=optimization_info.get('best_score', 0.0),
                validation_scores=validation_scores,
                statistical_validation=statistical_validation,
                bootstrap_validation=bootstrap_validation,
                optimization_history=self.optimization_history,
                convergence_info=optimization_info,
                metadata={
                    'config': self.config.__dict__,
                    'n_samples': features.shape[0],
                    'n_features': features.shape[1],
                    'baseline_metrics': baseline_metrics
                }
            )
            
            logger.info(f"✅ Validation threshold optimization completed. Best score: {result.optimization_score:.4f}")
            logger.info(f"📈 Optimal thresholds: {optimal_thresholds}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Validation threshold optimization failed: {e}")
            raise
    
    def _calculate_baseline_metrics(self, features: np.ndarray, clustering_func: Callable) -> Dict[str, float]:
        """Calculate baseline clustering metrics."""
        try:
            # Perform clustering
            cluster_labels = clustering_func(features)
            
            # Remove noise points
            valid_mask = cluster_labels != -1
            if valid_mask.sum() < 2:
                return {'silhouette': -1.0, 'davies_bouldin': 10.0, 'calinski_harabasz': 0.0}
            
            valid_labels = cluster_labels[valid_mask]
            valid_features = features[valid_mask]
            
            if len(set(valid_labels)) < 2:
                return {'silhouette': -1.0, 'davies_bouldin': 10.0, 'calinski_harabasz': 0.0}
            
            # Calculate metrics
            silhouette = silhouette_score(valid_features, valid_labels)
            davies_bouldin = davies_bouldin_score(valid_features, valid_labels)
            calinski_harabasz = calinski_harabasz_score(valid_features, valid_labels)
            
            return {
                'silhouette': silhouette,
                'davies_bouldin': davies_bouldin,
                'calinski_harabasz': calinski_harabasz,
                'n_clusters': len(set(valid_labels)),
                'n_samples': len(valid_labels)
            }
            
        except Exception as e:
            logger.warning(f"Baseline metrics calculation failed: {e}")
            return {'silhouette': -1.0, 'davies_bouldin': 10.0, 'calinski_harabasz': 0.0}
    
    def _calculate_null_distributions(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate null distributions for statistical validation."""
        try:
            null_distributions = {}
            
            if self.config.enable_permutation_testing:
                # Generate random clusterings
                n_samples = features.shape[0]
                silhouette_scores = []
                davies_bouldin_scores = []
                calinski_harabasz_scores = []
                
                for _ in range(self.config.permutation_samples):
                    # Generate random labels
                    n_clusters = np.random.randint(2, min(8, n_samples // 10))
                    random_labels = np.random.randint(0, n_clusters, n_samples)
                    
                    # Calculate metrics
                    try:
                        silhouette = silhouette_score(features, random_labels)
                        davies_bouldin = davies_bouldin_score(features, random_labels)
                        calinski_harabasz = calinski_harabasz_score(features, random_labels)
                        
                        silhouette_scores.append(silhouette)
                        davies_bouldin_scores.append(davies_bouldin)
                        calinski_harabasz_scores.append(calinski_harabasz)
                    except:
                        continue
                
                null_distributions['silhouette'] = np.array(silhouette_scores)
                null_distributions['davies_bouldin'] = np.array(davies_bouldin_scores)
                null_distributions['calinski_harabasz'] = np.array(calinski_harabasz_scores)
            
            return null_distributions
            
        except Exception as e:
            logger.warning(f"Null distribution calculation failed: {e}")
            return {}
    
    def _optimize_with_tpe(self, 
                          features: np.ndarray,
                          baseline_metrics: Dict[str, float],
                          null_distributions: Dict[str, np.ndarray],
                          clustering_func: Callable,
                          economic_validation_func: Optional[Callable]) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Optimize thresholds using Bayesian TPE."""
        if not OPTIMIZATION_AVAILABLE:
            raise ImportError("Bayesian TPE optimizer not available")
        
        def objective(trial):
            # Sample thresholds
            min_silhouette = trial.suggest_float(
                'min_silhouette',
                self.config.min_silhouette_range[0],
                self.config.min_silhouette_range[1]
            )
            max_dbi = trial.suggest_float(
                'max_dbi',
                self.config.max_dbi_range[0],
                self.config.max_dbi_range[1]
            )
            min_stability = trial.suggest_float(
                'min_stability',
                self.config.min_stability_range[0],
                self.config.min_stability_range[1]
            )
            
            thresholds = {
                'min_silhouette': min_silhouette,
                'max_dbi': max_dbi,
                'min_stability': min_stability
            }
            
            # Evaluate thresholds
            try:
                score = self._evaluate_thresholds(thresholds, features, baseline_metrics, 
                                                null_distributions, clustering_func, economic_validation_func)
                
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
            'min_silhouette': best_params.get('min_silhouette', 0.2),
            'max_dbi': best_params.get('max_dbi', 2.5),
            'min_stability': best_params.get('min_stability', 0.7)
        }
        
        return optimal_thresholds, {'best_score': best_score, 'n_trials': len(self.optimization_history)}
    
    def _optimize_with_grid_search(self, 
                                  features: np.ndarray,
                                  baseline_metrics: Dict[str, float],
                                  null_distributions: Dict[str, np.ndarray],
                                  clustering_func: Callable,
                                  economic_validation_func: Optional[Callable]) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Optimize thresholds using grid search."""
        # Create grid of threshold combinations
        silhouette_values = np.linspace(
            self.config.min_silhouette_range[0],
            self.config.min_silhouette_range[1],
            5
        )
        dbi_values = np.linspace(
            self.config.max_dbi_range[0],
            self.config.max_dbi_range[1],
            5
        )
        stability_values = np.linspace(
            self.config.min_stability_range[0],
            self.config.min_stability_range[1],
            5
        )
        
        best_score = -np.inf
        best_thresholds = None
        
        # Generate all combinations
        for sil_thresh, dbi_thresh, stab_thresh in itertools.product(silhouette_values, dbi_values, stability_values):
            thresholds = {
                'min_silhouette': sil_thresh,
                'max_dbi': dbi_thresh,
                'min_stability': stab_thresh
            }
            
            # Evaluate thresholds
            try:
                score = self._evaluate_thresholds(thresholds, features, baseline_metrics,
                                                null_distributions, clustering_func, economic_validation_func)
                
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
            'min_silhouette': 0.2,
            'max_dbi': 2.5,
            'min_stability': 0.7
        }, {'best_score': best_score}
    
    def _optimize_with_random_search(self, 
                                   features: np.ndarray,
                                   baseline_metrics: Dict[str, float],
                                   null_distributions: Dict[str, np.ndarray],
                                   clustering_func: Callable,
                                   economic_validation_func: Optional[Callable]) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Optimize thresholds using random search."""
        best_score = -np.inf
        best_thresholds = None
        
        for trial in range(self.config.n_trials):
            # Sample random thresholds
            thresholds = {
                'min_silhouette': np.random.uniform(*self.config.min_silhouette_range),
                'max_dbi': np.random.uniform(*self.config.max_dbi_range),
                'min_stability': np.random.uniform(*self.config.min_stability_range)
            }
            
            # Evaluate thresholds
            try:
                score = self._evaluate_thresholds(thresholds, features, baseline_metrics,
                                                null_distributions, clustering_func, economic_validation_func)
                
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
            'min_silhouette': 0.2,
            'max_dbi': 2.5,
            'min_stability': 0.7
        }, {'best_score': best_score}
    
    def _optimize_adaptively(self, 
                           features: np.ndarray,
                           baseline_metrics: Dict[str, float],
                           null_distributions: Dict[str, np.ndarray],
                           clustering_func: Callable,
                           economic_validation_func: Optional[Callable]) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Optimize thresholds adaptively based on data characteristics."""
        try:
            # Estimate thresholds based on null distributions
            thresholds = {}
            
            if 'silhouette' in null_distributions and len(null_distributions['silhouette']) > 0:
                # Set silhouette threshold based on null distribution
                null_silhouette = null_distributions['silhouette']
                percentile_95 = np.percentile(null_silhouette, 95)
                thresholds['min_silhouette'] = max(percentile_95, self.config.min_silhouette_floor)
            else:
                thresholds['min_silhouette'] = baseline_metrics.get('silhouette', 0.2) * 0.8
            
            if 'davies_bouldin' in null_distributions and len(null_distributions['davies_bouldin']) > 0:
                # Set DBI threshold based on null distribution
                null_dbi = null_distributions['davies_bouldin']
                percentile_5 = np.percentile(null_dbi, 5)
                thresholds['max_dbi'] = min(percentile_5, self.config.max_dbi_ceiling)
            else:
                thresholds['max_dbi'] = baseline_metrics.get('davies_bouldin', 2.5) * 1.2
            
            # Set stability threshold based on data size
            n_samples = features.shape[0]
            if n_samples < 100:
                thresholds['min_stability'] = 0.5
            elif n_samples < 1000:
                thresholds['min_stability'] = 0.6
            else:
                thresholds['min_stability'] = 0.7
            
            # Apply constraints
            thresholds['min_silhouette'] = np.clip(
                thresholds['min_silhouette'],
                self.config.min_silhouette_floor,
                self.config.min_silhouette_range[1]
            )
            thresholds['max_dbi'] = np.clip(
                thresholds['max_dbi'],
                self.config.max_dbi_range[0],
                self.config.max_dbi_ceiling
            )
            thresholds['min_stability'] = np.clip(
                thresholds['min_stability'],
                self.config.min_stability_floor,
                self.config.min_stability_range[1]
            )
            
            # Fine-tune with local optimization
            def objective(thresholds_array):
                min_sil, max_dbi, min_stab = thresholds_array
                temp_thresholds = {
                    'min_silhouette': min_sil,
                    'max_dbi': max_dbi,
                    'min_stability': min_stab
                }
                
                try:
                    score = self._evaluate_thresholds(temp_thresholds, features, baseline_metrics,
                                                    null_distributions, clustering_func, economic_validation_func)
                    return -score  # Minimize negative score
                except:
                    return np.inf
            
            # Initial thresholds
            initial_thresholds = np.array([
                thresholds['min_silhouette'],
                thresholds['max_dbi'],
                thresholds['min_stability']
            ])
            
            # Bounds
            bounds = [
                (self.config.min_silhouette_floor, self.config.min_silhouette_range[1]),
                (self.config.max_dbi_range[0], self.config.max_dbi_ceiling),
                (self.config.min_stability_floor, self.config.min_stability_range[1])
            ]
            
            # Optimize
            result = minimize(objective, initial_thresholds, method='L-BFGS-B', bounds=bounds)
            
            optimal_thresholds = {
                'min_silhouette': result.x[0],
                'max_dbi': result.x[1],
                'min_stability': result.x[2]
            }
            
            return optimal_thresholds, {'best_score': -result.fun, 'converged': result.success}
            
        except Exception as e:
            logger.warning(f"Adaptive optimization failed: {e}")
            return {
                'min_silhouette': 0.2,
                'max_dbi': 2.5,
                'min_stability': 0.7
            }, {'best_score': 0.0, 'converged': False}
    
    def _evaluate_thresholds(self, 
                           thresholds: Dict[str, float],
                           features: np.ndarray,
                           baseline_metrics: Dict[str, float],
                           null_distributions: Dict[str, np.ndarray],
                           clustering_func: Callable,
                           economic_validation_func: Optional[Callable]) -> float:
        """Evaluate threshold configuration."""
        try:
            # Perform clustering
            cluster_labels = clustering_func(features)
            
            # Remove noise points
            valid_mask = cluster_labels != -1
            if valid_mask.sum() < 2:
                return -np.inf
            
            valid_labels = cluster_labels[valid_mask]
            valid_features = features[valid_mask]
            
            if len(set(valid_labels)) < 2:
                return -np.inf
            
            # Calculate metrics
            silhouette = silhouette_score(valid_features, valid_labels)
            davies_bouldin = davies_bouldin_score(valid_features, valid_labels)
            
            # Check if clustering meets thresholds
            if silhouette < thresholds['min_silhouette']:
                return -np.inf
            if davies_bouldin > thresholds['max_dbi']:
                return -np.inf
            
            # Calculate stability (simplified)
            stability = self._calculate_stability(valid_labels)
            if stability < thresholds['min_stability']:
                return -np.inf
            
            # Calculate combined score
            score = silhouette - 0.1 * davies_bouldin + 0.2 * stability
            
            # Add statistical significance bonus
            if self.config.enable_permutation_testing:
                stat_bonus = self._calculate_statistical_bonus(
                    silhouette, davies_bouldin, null_distributions
                )
                score += stat_bonus
            
            # Add economic validation if available
            if economic_validation_func and self.config.enable_economic_validation:
                try:
                    economic_score = economic_validation_func(features, cluster_labels)
                    score += self.config.economic_weight * economic_score
                except Exception as e:
                    logger.debug(f"Economic validation failed: {e}")
            
            return score
            
        except Exception as e:
            logger.debug(f"Threshold evaluation failed: {e}")
            return -np.inf
    
    def _calculate_stability(self, cluster_labels: np.ndarray) -> float:
        """Calculate clustering stability."""
        try:
            # Calculate regime persistence
            label_changes = np.sum(np.diff(cluster_labels) != 0)
            total_periods = len(cluster_labels) - 1
            
            if total_periods == 0:
                return 1.0
            
            change_rate = label_changes / total_periods
            stability = 1.0 - change_rate
            
            return np.clip(stability, 0.0, 1.0)
            
        except Exception as e:
            logger.debug(f"Stability calculation failed: {e}")
            return 0.0
    
    def _calculate_statistical_bonus(self, 
                                   silhouette: float,
                                   davies_bouldin: float,
                                   null_distributions: Dict[str, np.ndarray]) -> float:
        """Calculate bonus for statistical significance."""
        try:
            bonus = 0.0
            
            if 'silhouette' in null_distributions and len(null_distributions['silhouette']) > 0:
                null_silhouette = null_distributions['silhouette']
                percentile = percentileofscore(null_silhouette, silhouette)
                if percentile > 95:
                    bonus += 0.1
                elif percentile > 90:
                    bonus += 0.05
            
            if 'davies_bouldin' in null_distributions and len(null_distributions['davies_bouldin']) > 0:
                null_dbi = null_distributions['davies_bouldin']
                percentile = percentileofscore(null_dbi, davies_bouldin)
                if percentile < 5:  # Lower DBI is better
                    bonus += 0.1
                elif percentile < 10:
                    bonus += 0.05
            
            return bonus
            
        except Exception as e:
            logger.debug(f"Statistical bonus calculation failed: {e}")
            return 0.0
    
    def _validate_thresholds(self, 
                           thresholds: Dict[str, float],
                           features: np.ndarray,
                           clustering_func: Callable) -> Dict[str, float]:
        """Validate optimal thresholds."""
        try:
            # Perform clustering
            cluster_labels = clustering_func(features)
            
            # Remove noise points
            valid_mask = cluster_labels != -1
            if valid_mask.sum() < 2:
                return {'silhouette': -1.0, 'davies_bouldin': 10.0, 'stability': 0.0}
            
            valid_labels = cluster_labels[valid_mask]
            valid_features = features[valid_mask]
            
            if len(set(valid_labels)) < 2:
                return {'silhouette': -1.0, 'davies_bouldin': 10.0, 'stability': 0.0}
            
            # Calculate metrics
            silhouette = silhouette_score(valid_features, valid_labels)
            davies_bouldin = davies_bouldin_score(valid_features, valid_labels)
            stability = self._calculate_stability(valid_labels)
            
            return {
                'silhouette': silhouette,
                'davies_bouldin': davies_bouldin,
                'stability': stability
            }
            
        except Exception as e:
            logger.warning(f"Threshold validation failed: {e}")
            return {'silhouette': -1.0, 'davies_bouldin': 10.0, 'stability': 0.0}
    
    def _calculate_statistical_validation(self, 
                                        thresholds: Dict[str, float],
                                        null_distributions: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Calculate statistical validation information."""
        try:
            validation_info = {}
            
            if 'silhouette' in null_distributions and len(null_distributions['silhouette']) > 0:
                null_silhouette = null_distributions['silhouette']
                threshold_percentile = percentileofscore(null_silhouette, thresholds['min_silhouette'])
                validation_info['silhouette_percentile'] = threshold_percentile
                validation_info['silhouette_significant'] = threshold_percentile > 95
            
            if 'davies_bouldin' in null_distributions and len(null_distributions['davies_bouldin']) > 0:
                null_dbi = null_distributions['davies_bouldin']
                threshold_percentile = percentileofscore(null_dbi, thresholds['max_dbi'])
                validation_info['dbi_percentile'] = threshold_percentile
                validation_info['dbi_significant'] = threshold_percentile < 5  # Lower is better
            
            return validation_info
            
        except Exception as e:
            logger.warning(f"Statistical validation calculation failed: {e}")
            return {}
    
    def _calculate_bootstrap_validation(self, 
                                      thresholds: Dict[str, float],
                                      features: np.ndarray,
                                      clustering_func: Callable) -> Dict[str, Any]:
        """Calculate bootstrap validation information."""
        try:
            if not self.config.enable_bootstrap_validation:
                return {}
            
            bootstrap_scores = []
            
            for _ in range(self.config.bootstrap_samples):
                # Bootstrap sample
                n_samples = features.shape[0]
                bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
                bootstrap_features = features[bootstrap_indices]
                
                # Perform clustering
                try:
                    cluster_labels = clustering_func(bootstrap_features)
                    
                    # Calculate metrics
                    valid_mask = cluster_labels != -1
                    if valid_mask.sum() < 2:
                        continue
                    
                    valid_labels = cluster_labels[valid_mask]
                    valid_features = bootstrap_features[valid_mask]
                    
                    if len(set(valid_labels)) < 2:
                        continue
                    
                    silhouette = silhouette_score(valid_features, valid_labels)
                    davies_bouldin = davies_bouldin_score(valid_features, valid_labels)
                    
                    # Check if meets thresholds
                    if (silhouette >= thresholds['min_silhouette'] and 
                        davies_bouldin <= thresholds['max_dbi']):
                        bootstrap_scores.append(1.0)
                    else:
                        bootstrap_scores.append(0.0)
                        
                except Exception as e:
                    logger.debug(f"Bootstrap trial failed: {e}")
                    continue
            
            if bootstrap_scores:
                success_rate = np.mean(bootstrap_scores)
                confidence_interval = np.percentile(bootstrap_scores, 
                                                  [50 - self.config.confidence_level/2 * 100,
                                                   50 + self.config.confidence_level/2 * 100])
                
                return {
                    'success_rate': success_rate,
                    'confidence_interval': confidence_interval,
                    'n_bootstrap_samples': len(bootstrap_scores),
                    'thresholds_met': success_rate > 0.8
                }
            else:
                return {'success_rate': 0.0, 'n_bootstrap_samples': 0}
            
        except Exception as e:
            logger.warning(f"Bootstrap validation calculation failed: {e}")
            return {}