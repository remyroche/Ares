"""
Data-Driven Feature Group Weight Optimization

This module provides optimization of feature group weights using various
strategies including Bayesian TPE, feature importance analysis, and
economic validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
import logging
from dataclasses import dataclass
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score
import warnings
from scipy.optimize import minimize
from scipy.stats import spearmanr

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
    FeatureGroupWeightConfig, ValidationMetric, OptimizationStrategy
)

logger = logging.getLogger(__name__)


@dataclass
class FeatureGroupWeightResult:
    """Result of feature group weight optimization."""
    optimal_weights: Dict[str, float]
    optimization_score: float
    validation_scores: Dict[str, float]
    feature_importance_scores: Dict[str, float]
    economic_scores: Dict[str, float]
    optimization_history: List[Dict[str, Any]]
    convergence_info: Dict[str, Any]
    metadata: Dict[str, Any]


class DataDrivenFeatureWeightOptimizer:
    """
    Data-driven optimizer for feature group weights in clustering.
    
    Replaces hardcoded weights (w_returns=0.50, w_vol=0.30, w_volume=0.20)
    with data-driven optimization based on clustering quality and economic metrics.
    """
    
    def __init__(self, config: FeatureGroupWeightConfig):
        """
        Initialize the feature weight optimizer.
        
        Args:
            config: Configuration for feature weight optimization
        """
        self.config = config
        self.optimization_history = []
        self.best_weights = None
        self.best_score = -np.inf
        
    def optimize_weights(self, 
                        features: np.ndarray,
                        feature_names: List[str],
                        market_data: pd.DataFrame,
                        clustering_func: Callable,
                        economic_validation_func: Optional[Callable] = None) -> FeatureGroupWeightResult:
        """
        Optimize feature group weights using the specified strategy.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            feature_names: List of feature names
            market_data: Market data for economic validation
            clustering_func: Function that performs clustering given features
            economic_validation_func: Optional function for economic validation
            
        Returns:
            FeatureGroupWeightResult with optimal weights and metadata
        """
        try:
            logger.info("🔍 Starting data-driven feature weight optimization...")
            
            # Categorize features into groups
            feature_groups = self._categorize_features(feature_names)
            logger.info(f"📊 Feature groups: {feature_groups}")
            
            # Calculate feature importance scores
            importance_scores = self._calculate_feature_importance(features, feature_names, market_data)
            
            # Optimize weights based on strategy
            if self.config.optimization_strategy == OptimizationStrategy.BAYESIAN_TPE:
                optimal_weights, optimization_info = self._optimize_with_tpe(
                    features, feature_groups, clustering_func, economic_validation_func
                )
            elif self.config.optimization_strategy == OptimizationStrategy.GRID_SEARCH:
                optimal_weights, optimization_info = self._optimize_with_grid_search(
                    features, feature_groups, clustering_func, economic_validation_func
                )
            elif self.config.optimization_strategy == OptimizationStrategy.RANDOM_SEARCH:
                optimal_weights, optimization_info = self._optimize_with_random_search(
                    features, feature_groups, clustering_func, economic_validation_func
                )
            elif self.config.optimization_strategy == OptimizationStrategy.ADAPTIVE:
                optimal_weights, optimization_info = self._optimize_adaptively(
                    features, feature_groups, importance_scores, clustering_func, economic_validation_func
                )
            else:
                raise ValueError(f"Unknown optimization strategy: {self.config.optimization_strategy}")
            
            # Validate optimal weights
            validation_scores = self._validate_weights(
                optimal_weights, features, feature_groups, clustering_func
            )
            
            # Calculate economic scores if validation function provided
            economic_scores = {}
            if economic_validation_func:
                economic_scores = self._calculate_economic_scores(
                    optimal_weights, features, feature_groups, market_data, economic_validation_func
                )
            
            # Create result
            result = FeatureGroupWeightResult(
                optimal_weights=optimal_weights,
                optimization_score=optimization_info.get('best_score', 0.0),
                validation_scores=validation_scores,
                feature_importance_scores=importance_scores,
                economic_scores=economic_scores,
                optimization_history=self.optimization_history,
                convergence_info=optimization_info,
                metadata={
                    'config': self.config.__dict__,
                    'n_features': features.shape[1],
                    'n_samples': features.shape[0],
                    'feature_groups': feature_groups
                }
            )
            
            logger.info(f"✅ Feature weight optimization completed. Best score: {result.optimization_score:.4f}")
            logger.info(f"📈 Optimal weights: {optimal_weights}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Feature weight optimization failed: {e}")
            raise
    
    def _categorize_features(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """Categorize features into groups based on naming patterns."""
        groups = {group: [] for group in self.config.feature_groups}
        
        for name in feature_names:
            name_lower = name.lower()
            
            # Returns group
            if any(term in name_lower for term in ['return', 'log_return', 'close_return', 'pct_change']):
                groups['returns'].append(name)
            # Volatility group
            elif any(term in name_lower for term in ['volatility', 'vol_', 'atr', 'std', 'boll', 'bb']):
                groups['volatility'].append(name)
            # Volume group
            elif any(term in name_lower for term in ['volume', 'vwap', 'obv', 'accumulation', 'distribution']):
                groups['volume'].append(name)
            # Other group
            else:
                groups['other'].append(name)
        
        # Remove empty groups
        groups = {k: v for k, v in groups.items() if v}
        
        return groups
    
    def _calculate_feature_importance(self, 
                                    features: np.ndarray, 
                                    feature_names: List[str],
                                    market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate feature importance scores using various methods."""
        try:
            importance_scores = {}
            
            if self.config.importance_method == 'mutual_info':
                # Use mutual information with returns as target
                if 'close' in market_data.columns:
                    returns = market_data['close'].pct_change().dropna()
                    if len(returns) == features.shape[0]:
                        mi_scores = mutual_info_regression(features, returns, random_state=42)
                        importance_scores = dict(zip(feature_names, mi_scores))
            
            elif self.config.importance_method == 'l1_regularization':
                # Use L1 regularization to find important features
                if 'close' in market_data.columns:
                    returns = market_data['close'].pct_change().dropna()
                    if len(returns) == features.shape[0]:
                        lasso = LassoCV(cv=3, random_state=42)
                        lasso.fit(features, returns)
                        importance_scores = dict(zip(feature_names, np.abs(lasso.coef_)))
            
            elif self.config.importance_method == 'permutation':
                # Use permutation importance (simplified version)
                if 'close' in market_data.columns:
                    returns = market_data['close'].pct_change().dropna()
                    if len(returns) == features.shape[0]:
                        # Calculate baseline score
                        baseline_score = np.corrcoef(returns, np.mean(features, axis=1))[0, 1]
                        
                        # Calculate permutation importance
                        for i, name in enumerate(feature_names):
                            features_perm = features.copy()
                            np.random.shuffle(features_perm[:, i])
                            perm_score = np.corrcoef(returns, np.mean(features_perm, axis=1))[0, 1]
                            importance_scores[name] = baseline_score - perm_score
            
            # Normalize importance scores
            if importance_scores:
                max_importance = max(importance_scores.values())
                if max_importance > 0:
                    importance_scores = {k: v / max_importance for k, v in importance_scores.items()}
            
            return importance_scores
            
        except Exception as e:
            logger.warning(f"Feature importance calculation failed: {e}")
            return {name: 1.0 for name in feature_names}
    
    def _optimize_with_tpe(self, 
                          features: np.ndarray,
                          feature_groups: Dict[str, List[str]],
                          clustering_func: Callable,
                          economic_validation_func: Optional[Callable]) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Optimize weights using Bayesian TPE."""
        if not OPTIMIZATION_AVAILABLE:
            raise ImportError("Bayesian TPE optimizer not available")
        
        def objective(trial):
            # Sample weights for each group
            weights = {}
            for group in self.config.feature_groups:
                if group in feature_groups:
                    weights[group] = trial.suggest_float(
                        f'weight_{group}', 
                        self.config.min_weight, 
                        self.config.max_weight
                    )
            
            # Normalize weights to sum to 1
            if self.config.weight_sum_constraint:
                total_weight = sum(weights.values())
                if total_weight > 0:
                    weights = {k: v / total_weight for k, v in weights.items()}
            
            # Apply weights to features
            weighted_features = self._apply_weights_to_features(features, feature_groups, weights)
            
            # Perform clustering
            try:
                cluster_labels = clustering_func(weighted_features)
                
                # Calculate quality metrics
                quality_score = self._calculate_quality_score(
                    weighted_features, cluster_labels, economic_validation_func
                )
                
                # Store trial info
                self.optimization_history.append({
                    'trial': len(self.optimization_history),
                    'weights': weights.copy(),
                    'score': quality_score,
                    'timestamp': pd.Timestamp.now()
                })
                
                return quality_score
                
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
        
        # Extract optimal weights
        optimal_weights = {}
        for group in self.config.feature_groups:
            if group in feature_groups:
                optimal_weights[group] = best_params.get(f'weight_{group}', 1.0 / len(feature_groups))
        
        # Normalize weights
        if self.config.weight_sum_constraint:
            total_weight = sum(optimal_weights.values())
            if total_weight > 0:
                optimal_weights = {k: v / total_weight for k, v in optimal_weights.items()}
        
        return optimal_weights, {'best_score': best_score, 'n_trials': len(self.optimization_history)}
    
    def _optimize_with_grid_search(self, 
                                  features: np.ndarray,
                                  feature_groups: Dict[str, List[str]],
                                  clustering_func: Callable,
                                  economic_validation_func: Optional[Callable]) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Optimize weights using grid search."""
        # Create grid of weight combinations
        weight_values = np.linspace(self.config.min_weight, self.config.max_weight, 5)
        
        best_score = -np.inf
        best_weights = None
        
        # Generate all combinations
        for weights_tuple in itertools.product(weight_values, repeat=len(feature_groups)):
            weights = dict(zip(feature_groups.keys(), weights_tuple))
            
            # Normalize weights
            if self.config.weight_sum_constraint:
                total_weight = sum(weights.values())
                if total_weight > 0:
                    weights = {k: v / total_weight for k, v in weights.items()}
            
            # Apply weights and evaluate
            try:
                weighted_features = self._apply_weights_to_features(features, feature_groups, weights)
                cluster_labels = clustering_func(weighted_features)
                score = self._calculate_quality_score(weighted_features, cluster_labels, economic_validation_func)
                
                if score > best_score:
                    best_score = score
                    best_weights = weights.copy()
                
                self.optimization_history.append({
                    'trial': len(self.optimization_history),
                    'weights': weights.copy(),
                    'score': score,
                    'timestamp': pd.Timestamp.now()
                })
                
            except Exception as e:
                logger.debug(f"Grid search trial failed: {e}")
                continue
        
        return best_weights or {group: 1.0 / len(feature_groups) for group in feature_groups}, {'best_score': best_score}
    
    def _optimize_with_random_search(self, 
                                   features: np.ndarray,
                                   feature_groups: Dict[str, List[str]],
                                   clustering_func: Callable,
                                   economic_validation_func: Optional[Callable]) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Optimize weights using random search."""
        best_score = -np.inf
        best_weights = None
        
        for trial in range(self.config.n_trials):
            # Sample random weights
            weights = {}
            for group in feature_groups:
                weights[group] = np.random.uniform(self.config.min_weight, self.config.max_weight)
            
            # Normalize weights
            if self.config.weight_sum_constraint:
                total_weight = sum(weights.values())
                if total_weight > 0:
                    weights = {k: v / total_weight for k, v in weights.items()}
            
            # Apply weights and evaluate
            try:
                weighted_features = self._apply_weights_to_features(features, feature_groups, weights)
                cluster_labels = clustering_func(weighted_features)
                score = self._calculate_quality_score(weighted_features, cluster_labels, economic_validation_func)
                
                if score > best_score:
                    best_score = score
                    best_weights = weights.copy()
                
                self.optimization_history.append({
                    'trial': trial,
                    'weights': weights.copy(),
                    'score': score,
                    'timestamp': pd.Timestamp.now()
                })
                
            except Exception as e:
                logger.debug(f"Random search trial failed: {e}")
                continue
        
        return best_weights or {group: 1.0 / len(feature_groups) for group in feature_groups}, {'best_score': best_score}
    
    def _optimize_adaptively(self, 
                           features: np.ndarray,
                           feature_groups: Dict[str, List[str]],
                           importance_scores: Dict[str, float],
                           clustering_func: Callable,
                           economic_validation_func: Optional[Callable]) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Optimize weights adaptively based on feature importance."""
        # Calculate group importance scores
        group_importance = {}
        for group, group_features in feature_groups.items():
            group_scores = [importance_scores.get(f, 0.0) for f in group_features]
            group_importance[group] = np.mean(group_scores) if group_scores else 0.0
        
        # Normalize group importance
        total_importance = sum(group_importance.values())
        if total_importance > 0:
            group_importance = {k: v / total_importance for k, v in group_importance.items()}
        else:
            group_importance = {group: 1.0 / len(feature_groups) for group in feature_groups}
        
        # Apply constraints
        for group in group_importance:
            group_importance[group] = np.clip(
                group_importance[group], 
                self.config.min_weight, 
                self.config.max_weight
            )
        
        # Normalize to sum to 1
        if self.config.weight_sum_constraint:
            total_weight = sum(group_importance.values())
            if total_weight > 0:
                group_importance = {k: v / total_weight for k, v in group_importance.values()}
        
        # Fine-tune with local optimization
        def objective(weights_array):
            weights = dict(zip(feature_groups.keys(), weights_array))
            
            # Apply weights and evaluate
            try:
                weighted_features = self._apply_weights_to_features(features, feature_groups, weights)
                cluster_labels = clustering_func(weighted_features)
                score = self._calculate_quality_score(weighted_features, cluster_labels, economic_validation_func)
                return -score  # Minimize negative score
            except:
                return np.inf
        
        # Initial weights
        initial_weights = np.array([group_importance[group] for group in feature_groups])
        
        # Constraints: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
        
        # Bounds: each weight between min and max
        bounds = [(self.config.min_weight, self.config.max_weight) for _ in feature_groups]
        
        # Optimize
        result = minimize(objective, initial_weights, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        optimal_weights = dict(zip(feature_groups.keys(), result.x))
        
        return optimal_weights, {'best_score': -result.fun, 'converged': result.success}
    
    def _apply_weights_to_features(self, 
                                  features: np.ndarray,
                                  feature_groups: Dict[str, List[str]],
                                  weights: Dict[str, float]) -> np.ndarray:
        """Apply group weights to features."""
        weighted_features = features.copy()
        
        for group, group_features in feature_groups.items():
            if group in weights:
                # Find indices of features in this group
                feature_indices = [i for i, name in enumerate(features.columns) if name in group_features]
                
                # Apply weight (sqrt because we're scaling variance)
                weight = np.sqrt(weights[group])
                weighted_features[:, feature_indices] *= weight
        
        return weighted_features
    
    def _calculate_quality_score(self, 
                                features: np.ndarray,
                                cluster_labels: np.ndarray,
                                economic_validation_func: Optional[Callable]) -> float:
        """Calculate combined quality score."""
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
            
            # Add economic validation if available
            if economic_validation_func and self.config.enable_economic_validation:
                try:
                    economic_score = economic_validation_func(features, cluster_labels)
                    combined_score += self.config.economic_weight * economic_score
                except Exception as e:
                    logger.debug(f"Economic validation failed: {e}")
            
            return combined_score
            
        except Exception as e:
            logger.debug(f"Quality score calculation failed: {e}")
            return -np.inf
    
    def _validate_weights(self, 
                         weights: Dict[str, float],
                         features: np.ndarray,
                         feature_groups: Dict[str, List[str]],
                         clustering_func: Callable) -> Dict[str, float]:
        """Validate optimal weights."""
        try:
            # Apply weights
            weighted_features = self._apply_weights_to_features(features, feature_groups, weights)
            cluster_labels = clustering_func(weighted_features)
            
            # Calculate validation metrics
            valid_mask = cluster_labels != -1
            if valid_mask.sum() < 2:
                return {'silhouette': -1.0, 'davies_bouldin': 10.0, 'calinski_harabasz': 0.0}
            
            valid_labels = cluster_labels[valid_mask]
            valid_features = weighted_features[valid_mask]
            
            if len(set(valid_labels)) < 2:
                return {'silhouette': -1.0, 'davies_bouldin': 10.0, 'calinski_harabasz': 0.0}
            
            return {
                'silhouette': silhouette_score(valid_features, valid_labels),
                'davies_bouldin': davies_bouldin_score(valid_features, valid_labels),
                'calinski_harabasz': calinski_harabasz_score(valid_features, valid_labels)
            }
            
        except Exception as e:
            logger.warning(f"Weight validation failed: {e}")
            return {'silhouette': -1.0, 'davies_bouldin': 10.0, 'calinski_harabasz': 0.0}
    
    def _calculate_economic_scores(self, 
                                 weights: Dict[str, float],
                                 features: np.ndarray,
                                 feature_groups: Dict[str, List[str]],
                                 market_data: pd.DataFrame,
                                 economic_validation_func: Callable) -> Dict[str, float]:
        """Calculate economic validation scores."""
        try:
            # Apply weights
            weighted_features = self._apply_weights_to_features(features, feature_groups, weights)
            cluster_labels = economic_validation_func(weighted_features, market_data)
            
            return cluster_labels  # Assuming the function returns a dict of scores
            
        except Exception as e:
            logger.warning(f"Economic score calculation failed: {e}")
            return {'sharpe_ratio': 0.0, 'return': 0.0, 'volatility': 1.0}