"""
Multi-Objective Optimization for Data-Driven Clustering

This module provides multi-objective optimization capabilities that balance
clustering quality metrics with economic performance indicators.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass
import logging
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.optimize import minimize
import warnings

# Import economic validator
from .economic_validator import EconomicValidator, EconomicValidationConfig

logger = logging.getLogger(__name__)

@dataclass
class MultiObjectiveConfig:
    """Configuration for multi-objective optimization."""
    # Clustering quality weights
    silhouette_weight: float = 0.2
    calinski_harabasz_weight: float = 0.15
    davies_bouldin_weight: float = 0.15
    
    # Economic performance weights
    return_separation_weight: float = 0.2
    volatility_discrimination_weight: float = 0.15
    risk_discrimination_weight: float = 0.1
    strategy_performance_weight: float = 0.05
    
    # Optimization parameters
    max_iterations: int = 100
    convergence_tolerance: float = 1e-6
    population_size: int = 50
    
    # Economic validation
    enable_economic_validation: bool = True
    economic_validation_config: Optional[EconomicValidationConfig] = None
    
    # Multi-objective strategy
    optimization_strategy: str = 'weighted_sum'  # 'weighted_sum', 'pareto_frontier', 'lexicographic'
    
    # Constraint handling
    enable_constraints: bool = True
    min_cluster_size: int = 5
    max_clusters: int = 20
    min_silhouette_threshold: float = 0.1
    max_dbi_threshold: float = 3.0

class MultiObjectiveOptimizer:
    """
    Multi-objective optimizer for clustering parameters.
    
    Balances clustering quality metrics with economic performance indicators
    to find optimal parameters that maximize both clustering quality and
    financial performance.
    """
    
    def __init__(self, config: Optional[MultiObjectiveConfig] = None):
        """Initialize multi-objective optimizer."""
        self.config = config or MultiObjectiveConfig()
        self.economic_validator = EconomicValidator(self.config.economic_validation_config)
        self.optimization_history = []
        self.pareto_frontier = []
        
    def optimize_parameters(self, 
                          parameter_ranges: Dict[str, Tuple[float, float]],
                          clustering_func: Callable,
                          market_data: pd.DataFrame,
                          features: np.ndarray,
                          feature_names: List[str],
                          initial_parameters: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Optimize parameters using multi-objective optimization.
        
        Args:
            parameter_ranges: Dictionary of parameter names to (min, max) ranges
            clustering_func: Function that performs clustering
            market_data: Market data for economic validation
            features: Feature matrix
            feature_names: List of feature names
            initial_parameters: Optional initial parameter values
            
        Returns:
            Dictionary with optimization results
        """
        try:
            logger.info("🚀 Starting multi-objective parameter optimization...")
            
            # Initialize parameters
            if initial_parameters is None:
                initial_parameters = self._initialize_parameters(parameter_ranges)
            
            # Choose optimization strategy
            if self.config.optimization_strategy == 'weighted_sum':
                result = self._optimize_weighted_sum(
                    parameter_ranges, clustering_func, market_data, features, feature_names, initial_parameters
                )
            elif self.config.optimization_strategy == 'pareto_frontier':
                result = self._optimize_pareto_frontier(
                    parameter_ranges, clustering_func, market_data, features, feature_names, initial_parameters
                )
            elif self.config.optimization_strategy == 'lexicographic':
                result = self._optimize_lexicographic(
                    parameter_ranges, clustering_func, market_data, features, feature_names, initial_parameters
                )
            else:
                raise ValueError(f"Unknown optimization strategy: {self.config.optimization_strategy}")
            
            logger.info(f"✅ Multi-objective optimization completed")
            logger.info(f"📊 Final score: {result.get('overall_score', 0):.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Multi-objective optimization failed: {e}")
            raise
    
    def _initialize_parameters(self, parameter_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """Initialize parameters within specified ranges."""
        try:
            parameters = {}
            for param_name, (min_val, max_val) in parameter_ranges.items():
                # Use middle of range as initial value
                parameters[param_name] = (min_val + max_val) / 2
            return parameters
        except Exception as e:
            logger.warning(f"Parameter initialization failed: {e}")
            return {}
    
    def _optimize_weighted_sum(self, 
                             parameter_ranges: Dict[str, Tuple[float, float]],
                             clustering_func: Callable,
                             market_data: pd.DataFrame,
                             features: np.ndarray,
                             feature_names: List[str],
                             initial_parameters: Dict[str, float]) -> Dict[str, Any]:
        """Optimize using weighted sum approach."""
        try:
            # Define objective function
            def objective_function(params):
                return self._evaluate_parameters(
                    params, parameter_ranges, clustering_func, market_data, features, feature_names
                )
            
            # Convert parameters to array for optimization
            param_names = list(parameter_ranges.keys())
            param_bounds = [parameter_ranges[name] for name in param_names]
            initial_values = [initial_parameters.get(name, 0.5) for name in param_names]
            
            # Perform optimization
            result = minimize(
                objective_function,
                initial_values,
                method='L-BFGS-B',
                bounds=param_bounds,
                options={'maxiter': self.config.max_iterations}
            )
            
            # Convert result back to parameter dictionary
            optimal_parameters = dict(zip(param_names, result.x))
            
            # Calculate final scores
            final_scores = self._evaluate_parameters(
                result.x, parameter_ranges, clustering_func, market_data, features, feature_names, return_detailed=True
            )
            
            return {
                'optimal_parameters': optimal_parameters,
                'overall_score': -result.fun,  # Negative because minimize returns negative
                'detailed_scores': final_scores,
                'optimization_success': result.success,
                'n_iterations': result.nit,
                'convergence_info': {
                    'success': result.success,
                    'message': result.message,
                    'n_iterations': result.nit,
                    'n_function_evaluations': result.nfev
                }
            }
            
        except Exception as e:
            logger.error(f"Weighted sum optimization failed: {e}")
            raise
    
    def _optimize_pareto_frontier(self, 
                                parameter_ranges: Dict[str, Tuple[float, float]],
                                clustering_func: Callable,
                                market_data: pd.DataFrame,
                                features: np.ndarray,
                                feature_names: List[str],
                                initial_parameters: Dict[str, float]) -> Dict[str, Any]:
        """Optimize using Pareto frontier approach."""
        try:
            # This is a simplified implementation
            # In practice, you'd use NSGA-II or similar multi-objective algorithms
            
            logger.info("🔍 Exploring Pareto frontier...")
            
            # Sample parameter space
            n_samples = 100
            pareto_solutions = []
            
            for i in range(n_samples):
                # Sample random parameters
                sample_params = {}
                for param_name, (min_val, max_val) in parameter_ranges.items():
                    sample_params[param_name] = np.random.uniform(min_val, max_val)
                
                # Evaluate solution
                scores = self._evaluate_parameters(
                    list(sample_params.values()), parameter_ranges, clustering_func, 
                    market_data, features, feature_names, return_detailed=True
                )
                
                if scores is not None:
                    pareto_solutions.append({
                        'parameters': sample_params,
                        'scores': scores
                    })
            
            # Find Pareto optimal solutions
            pareto_optimal = self._find_pareto_optimal(pareto_solutions)
            
            # Select best solution (highest overall score)
            best_solution = max(pareto_optimal, key=lambda x: x['scores'].get('overall_score', 0))
            
            self.pareto_frontier = pareto_optimal
            
            return {
                'optimal_parameters': best_solution['parameters'],
                'overall_score': best_solution['scores'].get('overall_score', 0),
                'detailed_scores': best_solution['scores'],
                'pareto_frontier': pareto_optimal,
                'n_pareto_solutions': len(pareto_optimal),
                'optimization_success': True
            }
            
        except Exception as e:
            logger.error(f"Pareto frontier optimization failed: {e}")
            raise
    
    def _optimize_lexicographic(self, 
                              parameter_ranges: Dict[str, Tuple[float, float]],
                              clustering_func: Callable,
                              market_data: pd.DataFrame,
                              features: np.ndarray,
                              feature_names: List[str],
                              initial_parameters: Dict[str, float]) -> Dict[str, Any]:
        """Optimize using lexicographic approach."""
        try:
            logger.info("🔍 Using lexicographic optimization...")
            
            # Define objective priorities
            objectives = [
                ('clustering_quality', 1.0),
                ('economic_performance', 0.8),
                ('stability', 0.6)
            ]
            
            current_parameters = initial_parameters.copy()
            optimization_results = {}
            
            for obj_name, weight in objectives:
                logger.info(f"Optimizing {obj_name} with weight {weight}")
                
                # Define objective function for this priority
                def objective_function(params):
                    scores = self._evaluate_parameters(
                        params, parameter_ranges, clustering_func, market_data, features, feature_names, return_detailed=True
                    )
                    
                    if scores is None:
                        return 1e6  # Large penalty for invalid solutions
                    
                    # Weight the objective
                    if obj_name == 'clustering_quality':
                        return -(scores.get('clustering_score', 0) * weight)
                    elif obj_name == 'economic_performance':
                        return -(scores.get('economic_score', 0) * weight)
                    elif obj_name == 'stability':
                        return -(scores.get('stability_score', 0) * weight)
                    else:
                        return 1e6
                
                # Optimize this objective
                param_names = list(parameter_ranges.keys())
                param_bounds = [parameter_ranges[name] for name in param_names]
                initial_values = [current_parameters.get(name, 0.5) for name in param_names]
                
                result = minimize(
                    objective_function,
                    initial_values,
                    method='L-BFGS-B',
                    bounds=param_bounds,
                    options={'maxiter': self.config.max_iterations // len(objectives)}
                )
                
                if result.success:
                    current_parameters = dict(zip(param_names, result.x))
                    optimization_results[obj_name] = {
                        'success': True,
                        'score': -result.fun,
                        'parameters': current_parameters.copy()
                    }
                else:
                    optimization_results[obj_name] = {
                        'success': False,
                        'score': 0,
                        'parameters': current_parameters.copy()
                    }
            
            # Calculate final scores
            final_scores = self._evaluate_parameters(
                list(current_parameters.values()), parameter_ranges, clustering_func, 
                market_data, features, feature_names, return_detailed=True
            )
            
            return {
                'optimal_parameters': current_parameters,
                'overall_score': final_scores.get('overall_score', 0) if final_scores else 0,
                'detailed_scores': final_scores,
                'optimization_results': optimization_results,
                'optimization_success': any(result['success'] for result in optimization_results.values())
            }
            
        except Exception as e:
            logger.error(f"Lexicographic optimization failed: {e}")
            raise
    
    def _evaluate_parameters(self, 
                           params: List[float],
                           parameter_ranges: Dict[str, Tuple[float, float]],
                           clustering_func: Callable,
                           market_data: pd.DataFrame,
                           features: np.ndarray,
                           feature_names: List[str],
                           return_detailed: bool = False) -> Union[float, Dict[str, Any]]:
        """Evaluate parameters and return scores."""
        try:
            # Convert parameters to dictionary
            param_names = list(parameter_ranges.keys())
            param_dict = dict(zip(param_names, params))
            
            # Apply parameters to clustering function
            # This is a simplified example - in practice, you'd modify the clustering function
            # to accept and use these parameters
            
            # Perform clustering
            cluster_labels = clustering_func(features)
            
            # Calculate clustering quality metrics
            clustering_scores = self._calculate_clustering_scores(cluster_labels, features)
            
            # Calculate economic performance metrics
            economic_scores = self._calculate_economic_scores(cluster_labels, market_data, features, feature_names)
            
            # Calculate stability metrics
            stability_scores = self._calculate_stability_scores(cluster_labels, features)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(
                clustering_scores, economic_scores, stability_scores
            )
            
            if return_detailed:
                return {
                    'overall_score': overall_score,
                    'clustering_score': clustering_scores.get('overall', 0),
                    'economic_score': economic_scores.get('overall', 0),
                    'stability_score': stability_scores.get('overall', 0),
                    'clustering_metrics': clustering_scores,
                    'economic_metrics': economic_scores,
                    'stability_metrics': stability_scores
                }
            else:
                return -overall_score  # Negative for minimization
                
        except Exception as e:
            logger.warning(f"Parameter evaluation failed: {e}")
            return 1e6 if not return_detailed else None
    
    def _calculate_clustering_scores(self, cluster_labels: np.ndarray, features: np.ndarray) -> Dict[str, float]:
        """Calculate clustering quality scores."""
        try:
            # Remove noise points
            valid_mask = cluster_labels != -1
            if valid_mask.sum() < 2:
                return {'overall': 0.0}
            
            valid_features = features[valid_mask]
            valid_labels = cluster_labels[valid_mask]
            
            if len(set(valid_labels)) < 2:
                return {'overall': 0.0}
            
            # Calculate metrics
            silhouette = silhouette_score(valid_features, valid_labels)
            calinski_harabasz = calinski_harabasz_score(valid_features, valid_labels)
            davies_bouldin = davies_bouldin_score(valid_features, valid_labels)
            
            # Normalize DBI (lower is better)
            normalized_dbi = 1 / (1 + davies_bouldin)
            
            # Calculate weighted overall score
            overall_score = (
                self.config.silhouette_weight * silhouette +
                self.config.calinski_harabasz_weight * (calinski_harabasz / 1000) +  # Normalize
                self.config.davies_bouldin_weight * normalized_dbi
            )
            
            return {
                'overall': overall_score,
                'silhouette': silhouette,
                'calinski_harabasz': calinski_harabasz,
                'davies_bouldin': davies_bouldin,
                'normalized_dbi': normalized_dbi
            }
            
        except Exception as e:
            logger.warning(f"Clustering scores calculation failed: {e}")
            return {'overall': 0.0}
    
    def _calculate_economic_scores(self, 
                                 cluster_labels: np.ndarray, 
                                 market_data: pd.DataFrame,
                                 features: np.ndarray,
                                 feature_names: List[str]) -> Dict[str, float]:
        """Calculate economic performance scores."""
        try:
            if not self.config.enable_economic_validation:
                return {'overall': 0.0}
            
            # Use economic validator
            economic_result = self.economic_validator.validate_clustering(
                cluster_labels, market_data, features, feature_names
            )
            
            return {
                'overall': economic_result.overall_economic_score,
                'return_separation': economic_result.return_separation_score,
                'volatility_discrimination': economic_result.volatility_discrimination_score,
                'risk_discrimination': economic_result.risk_discrimination_score,
                'drawdown_discrimination': economic_result.drawdown_discrimination_score,
                'volume_discrimination': economic_result.volume_discrimination_score,
                'strategy_performance': economic_result.strategy_performance_score
            }
            
        except Exception as e:
            logger.warning(f"Economic scores calculation failed: {e}")
            return {'overall': 0.0}
    
    def _calculate_stability_scores(self, cluster_labels: np.ndarray, features: np.ndarray) -> Dict[str, float]:
        """Calculate stability scores."""
        try:
            # Calculate cluster size distribution
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            unique_labels = unique_labels[unique_labels != -1]  # Remove noise
            counts = counts[unique_labels != -1]
            
            if len(counts) < 2:
                return {'overall': 0.0}
            
            # Calculate size balance (lower std is better)
            size_balance = 1 / (1 + np.std(counts) / np.mean(counts))
            
            # Calculate cluster count stability
            n_clusters = len(unique_labels)
            cluster_count_stability = 1 / (1 + abs(n_clusters - 10) / 10)  # Prefer around 10 clusters
            
            # Calculate overall stability
            overall_stability = (size_balance + cluster_count_stability) / 2
            
            return {
                'overall': overall_stability,
                'size_balance': size_balance,
                'cluster_count_stability': cluster_count_stability,
                'n_clusters': n_clusters
            }
            
        except Exception as e:
            logger.warning(f"Stability scores calculation failed: {e}")
            return {'overall': 0.0}
    
    def _calculate_overall_score(self, 
                               clustering_scores: Dict[str, float],
                               economic_scores: Dict[str, float],
                               stability_scores: Dict[str, float]) -> float:
        """Calculate overall optimization score."""
        try:
            # Weighted combination of all scores
            overall_score = (
                0.4 * clustering_scores.get('overall', 0) +
                0.4 * economic_scores.get('overall', 0) +
                0.2 * stability_scores.get('overall', 0)
            )
            
            return overall_score
            
        except Exception as e:
            logger.warning(f"Overall score calculation failed: {e}")
            return 0.0
    
    def _find_pareto_optimal(self, solutions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find Pareto optimal solutions."""
        try:
            if not solutions:
                return []
            
            pareto_optimal = []
            
            for i, solution in enumerate(solutions):
                is_pareto = True
                scores_i = solution['scores']
                
                for j, other_solution in enumerate(solutions):
                    if i == j:
                        continue
                    
                    scores_j = other_solution['scores']
                    
                    # Check if other solution dominates this one
                    if self._dominates(scores_j, scores_i):
                        is_pareto = False
                        break
                
                if is_pareto:
                    pareto_optimal.append(solution)
            
            return pareto_optimal
            
        except Exception as e:
            logger.warning(f"Pareto optimal finding failed: {e}")
            return solutions
    
    def _dominates(self, scores1: Dict[str, float], scores2: Dict[str, float]) -> bool:
        """Check if scores1 dominates scores2."""
        try:
            # Check if scores1 is better in all objectives
            better_in_all = True
            better_in_some = False
            
            for key in ['overall_score', 'clustering_score', 'economic_score', 'stability_score']:
                if key in scores1 and key in scores2:
                    if scores1[key] < scores2[key]:
                        better_in_all = False
                        break
                    elif scores1[key] > scores2[key]:
                        better_in_some = True
            
            return better_in_all and better_in_some
            
        except Exception:
            return False
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history."""
        return self.optimization_history.copy()
    
    def get_pareto_frontier(self) -> List[Dict[str, Any]]:
        """Get Pareto frontier solutions."""
        return self.pareto_frontier.copy()