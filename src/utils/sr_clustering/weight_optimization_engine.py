"""
Weight Optimization Engine for SR Quality Score Parameters

This module implements backtesting-based optimization of quality score parameter weights
to maximize the predictive power of the quality scoring system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize
import itertools

from ..logger import system_logger
from .sr_backtesting_engine import SRBacktestingEngine, BacktestResult, SRLevel

@dataclass
class WeightOptimizationConfig:
    """Configuration for weight optimization."""
    # Optimization parameters
    optimization_method: str = 'scipy_minimize'  # 'scipy_minimize', 'grid_search', 'genetic_algorithm'
    max_iterations: int = 100
    convergence_tolerance: float = 1e-6
    
    # Cross-validation parameters
    n_splits: int = 5
    test_size: float = 0.2
    
    # Weight constraints
    min_weight: float = 0.0
    max_weight: float = 1.0
    weight_sum_constraint: bool = True  # Whether weights should sum to 1.0
    
    # Feature groups for optimization
    primary_features: List[str] = field(default_factory=lambda: [
        'success_rate', 'avg_bounce_strength', 'total_volume_at_level', 
        'time_persistence', 'touch_frequency'
    ])
    penetration_features: List[str] = field(default_factory=lambda: [
        'penetration_depth', 'penetration_frequency'
    ])
    pattern_features: List[str] = field(default_factory=lambda: [
        'pattern_consistency', 'pattern_strength', 'order_flow_confirmation'
    ])
    
    # Optimization objectives
    primary_objective: str = 'r2_score'  # 'r2_score', 'mse', 'mae', 'correlation'
    secondary_objective: str = 'stability'  # 'stability', 'generalization', 'interpretability'

class WeightOptimizationEngine:
    """Engine for optimizing quality score parameter weights through backtesting."""
    
    def __init__(self, config: Optional[WeightOptimizationConfig] = None):
        self.config = config or WeightOptimizationConfig()
        self.logger = system_logger.getChild('WeightOptimizationEngine')
        self.optimization_history: List[Dict[str, Any]] = []
        self.best_weights: Dict[str, float] = {}
        self.best_score: float = 0.0
        
    def optimize_weights(self, backtest_results: List[BacktestResult], 
                        market_data: pd.DataFrame) -> Dict[str, Any]:
        """Optimize quality score parameter weights using backtesting."""
        try:
            self.logger.info(f"Starting weight optimization for {len(backtest_results)} backtest results")
            
            # Prepare data for optimization
            optimization_data = self._prepare_optimization_data(backtest_results, market_data)
            
            if not optimization_data:
                self.logger.warning("No valid data for optimization")
                return {}
            
            # Run optimization based on method
            if self.config.optimization_method == 'scipy_minimize':
                result = self._optimize_with_scipy(optimization_data)
            elif self.config.optimization_method == 'grid_search':
                result = self._optimize_with_grid_search(optimization_data)
            elif self.config.optimization_method == 'genetic_algorithm':
                result = self._optimize_with_genetic_algorithm(optimization_data)
            else:
                raise ValueError(f"Unknown optimization method: {self.config.optimization_method}")
            
            # Store results
            self.best_weights = result['best_weights']
            self.best_score = result['best_score']
            self.optimization_history.append(result)
            
            self.logger.info(f"Weight optimization completed. Best score: {self.best_score:.4f}")
            self.logger.info(f"Best weights: {self.best_weights}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Weight optimization failed: {e}")
            return {}
    
    def _prepare_optimization_data(self, backtest_results: List[BacktestResult], 
                                 market_data: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for weight optimization."""
        try:
            if not backtest_results:
                return {}
            
            # Extract features and target
            all_features = (self.config.primary_features + 
                          self.config.penetration_features + 
                          self.config.pattern_features)
            
            # Build feature matrix
            feature_data = {}
            for feature in all_features:
                feature_values = []
                for result in backtest_results:
                    value = getattr(result, feature, 0.0)
                    feature_values.append(value)
                feature_data[feature] = np.array(feature_values)
            
            # Target variable (actual quality scores from backtesting)
            target_scores = np.array([result.quality_score for result in backtest_results])
            
            # Market context features (if available)
            market_context = self._extract_market_context(backtest_results, market_data)
            
            return {
                'feature_data': feature_data,
                'target_scores': target_scores,
                'market_context': market_context,
                'backtest_results': backtest_results,
                'feature_names': all_features
            }
            
        except Exception as e:
            self.logger.error(f"Failed to prepare optimization data: {e}")
            return {}
    
    def _extract_market_context(self, backtest_results: List[BacktestResult], 
                              market_data: pd.DataFrame) -> Dict[str, Any]:
        """Extract market context for optimization."""
        try:
            # Calculate market regime features
            if len(market_data) > 0:
                # Volatility regime
                returns = market_data['close'].pct_change().dropna()
                volatility = returns.rolling(20).std()
                volatility_regime = np.mean(volatility) if len(volatility) > 0 else 0.0
                
                # Trend strength
                sma_short = market_data['close'].rolling(10).mean()
                sma_long = market_data['close'].rolling(50).mean()
                trend_strength = abs(np.mean((sma_short - sma_long) / sma_long)) if len(sma_short) > 0 else 0.0
                
                # Volume regime
                volume_avg = market_data['volume'].mean() if 'volume' in market_data.columns else 1.0
                volume_regime = volume_avg / 1000000  # Normalize
                
                return {
                    'volatility_regime': volatility_regime,
                    'trend_strength': trend_strength,
                    'volume_regime': volume_regime,
                    'market_periods': len(market_data)
                }
            else:
                return {
                    'volatility_regime': 0.0,
                    'trend_strength': 0.0,
                    'volume_regime': 0.0,
                    'market_periods': 0
                }
                
        except Exception as e:
            self.logger.warning(f"Failed to extract market context: {e}")
            return {}
    
    def _optimize_with_scipy(self, optimization_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize weights using scipy minimize."""
        try:
            feature_data = optimization_data['feature_data']
            target_scores = optimization_data['target_scores']
            feature_names = optimization_data['feature_names']
            
            # Initial weights (equal weights)
            n_features = len(feature_names)
            initial_weights = np.ones(n_features) / n_features
            
            # Define objective function
            def objective(weights):
                return -self._evaluate_weights(weights, feature_data, target_scores, feature_names)
            
            # Define constraints
            constraints = []
            if self.config.weight_sum_constraint:
                constraints.append({
                    'type': 'eq',
                    'fun': lambda w: np.sum(w) - 1.0
                })
            
            # Define bounds
            bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(n_features)]
            
            # Optimize
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': self.config.max_iterations, 'ftol': self.config.convergence_tolerance}
            )
            
            if result.success:
                best_weights = dict(zip(feature_names, result.x))
                best_score = -result.fun
                
                return {
                    'method': 'scipy_minimize',
                    'best_weights': best_weights,
                    'best_score': best_score,
                    'optimization_success': True,
                    'iterations': result.nit,
                    'convergence_message': result.message
                }
            else:
                self.logger.warning(f"Scipy optimization failed: {result.message}")
                return self._get_default_weights(feature_names)
                
        except Exception as e:
            self.logger.error(f"Scipy optimization failed: {e}")
            return self._get_default_weights(optimization_data['feature_names'])
    
    def _optimize_with_grid_search(self, optimization_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize weights using grid search."""
        try:
            feature_data = optimization_data['feature_data']
            target_scores = optimization_data['target_scores']
            feature_names = optimization_data['feature_names']
            
            # Define weight grid
            weight_values = np.linspace(self.config.min_weight, self.config.max_weight, 11)  # 0.0 to 1.0 in steps of 0.1
            
            best_score = -np.inf
            best_weights = {}
            
            # Generate all possible weight combinations
            weight_combinations = itertools.product(weight_values, repeat=len(feature_names))
            
            total_combinations = len(weight_values) ** len(feature_names)
            self.logger.info(f"Grid search: evaluating {total_combinations} weight combinations")
            
            evaluated = 0
            for weights in weight_combinations:
                weights = np.array(weights)
                
                # Apply weight sum constraint
                if self.config.weight_sum_constraint:
                    weights = weights / np.sum(weights)
                
                # Evaluate weights
                score = self._evaluate_weights(weights, feature_data, target_scores, feature_names)
                
                if score > best_score:
                    best_score = score
                    best_weights = dict(zip(feature_names, weights))
                
                evaluated += 1
                if evaluated % 1000 == 0:
                    self.logger.info(f"Evaluated {evaluated}/{total_combinations} combinations")
            
            return {
                'method': 'grid_search',
                'best_weights': best_weights,
                'best_score': best_score,
                'optimization_success': True,
                'combinations_evaluated': evaluated
            }
            
        except Exception as e:
            self.logger.error(f"Grid search optimization failed: {e}")
            return self._get_default_weights(optimization_data['feature_names'])
    
    def _optimize_with_genetic_algorithm(self, optimization_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize weights using genetic algorithm (simplified implementation)."""
        try:
            # This is a simplified genetic algorithm implementation
            # In practice, you might want to use DEAP or similar library
            
            feature_data = optimization_data['feature_data']
            target_scores = optimization_data['target_scores']
            feature_names = optimization_data['feature_names']
            
            n_features = len(feature_names)
            population_size = 50
            generations = 20
            
            # Initialize population
            population = []
            for _ in range(population_size):
                weights = np.random.random(n_features)
                if self.config.weight_sum_constraint:
                    weights = weights / np.sum(weights)
                population.append(weights)
            
            best_score = -np.inf
            best_weights = {}
            
            for generation in range(generations):
                # Evaluate population
                scores = []
                for weights in population:
                    score = self._evaluate_weights(weights, feature_data, target_scores, feature_names)
                    scores.append(score)
                    
                    if score > best_score:
                        best_score = score
                        best_weights = dict(zip(feature_names, weights))
                
                # Selection (keep top 50%)
                sorted_indices = np.argsort(scores)[::-1]
                elite_size = population_size // 2
                elite = [population[i] for i in sorted_indices[:elite_size]]
                
                # Create new generation
                new_population = elite.copy()
                
                # Crossover and mutation
                while len(new_population) < population_size:
                    parent1 = elite[np.random.randint(elite_size)]
                    parent2 = elite[np.random.randint(elite_size)]
                    
                    # Crossover
                    child = (parent1 + parent2) / 2
                    
                    # Mutation
                    mutation_rate = 0.1
                    for i in range(n_features):
                        if np.random.random() < mutation_rate:
                            child[i] = np.random.random()
                    
                    # Apply constraints
                    if self.config.weight_sum_constraint:
                        child = child / np.sum(child)
                    
                    new_population.append(child)
                
                population = new_population
                
                self.logger.info(f"Generation {generation + 1}: Best score = {best_score:.4f}")
            
            return {
                'method': 'genetic_algorithm',
                'best_weights': best_weights,
                'best_score': best_score,
                'optimization_success': True,
                'generations': generations,
                'population_size': population_size
            }
            
        except Exception as e:
            self.logger.error(f"Genetic algorithm optimization failed: {e}")
            return self._get_default_weights(optimization_data['feature_names'])
    
    def _evaluate_weights(self, weights: np.ndarray, feature_data: Dict[str, np.ndarray], 
                         target_scores: np.ndarray, feature_names: List[str]) -> float:
        """Evaluate a set of weights using cross-validation."""
        try:
            # Build weighted quality scores
            weighted_scores = np.zeros(len(target_scores))
            
            for i, feature in enumerate(feature_names):
                if feature in feature_data:
                    weighted_scores += weights[i] * feature_data[feature]
            
            # Normalize to 0-1 range
            weighted_scores = np.clip(weighted_scores, 0.0, 1.0)
            
            # Calculate performance metric
            if self.config.primary_objective == 'r2_score':
                score = r2_score(target_scores, weighted_scores)
            elif self.config.primary_objective == 'mse':
                score = -mean_squared_error(target_scores, weighted_scores)  # Negative because we want to minimize MSE
            elif self.config.primary_objective == 'correlation':
                correlation = np.corrcoef(target_scores, weighted_scores)[0, 1]
                score = correlation if not np.isnan(correlation) else 0.0
            else:
                score = r2_score(target_scores, weighted_scores)  # Default to R²
            
            # Add stability penalty if requested
            if self.config.secondary_objective == 'stability':
                # Penalize extreme weights
                weight_penalty = -np.sum(np.abs(weights - np.mean(weights))) * 0.1
                score += weight_penalty
            
            return score
            
        except Exception as e:
            self.logger.warning(f"Weight evaluation failed: {e}")
            return 0.0
    
    def _get_default_weights(self, feature_names: List[str]) -> Dict[str, Any]:
        """Get default weights when optimization fails."""
        n_features = len(feature_names)
        default_weights = {feature: 1.0 / n_features for feature in feature_names}
        
        return {
            'method': 'default',
            'best_weights': default_weights,
            'best_score': 0.0,
            'optimization_success': False,
            'error': 'Optimization failed, using default weights'
        }
    
    def get_optimized_weights(self) -> Dict[str, float]:
        """Get the best optimized weights."""
        return self.best_weights.copy() if self.best_weights else {}
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get a summary of the optimization process."""
        if not self.optimization_history:
            return {'status': 'No optimization performed yet'}
        
        latest_result = self.optimization_history[-1]
        
        return {
            'status': 'Optimization completed',
            'method': latest_result.get('method', 'unknown'),
            'best_score': self.best_score,
            'best_weights': self.best_weights,
            'optimization_success': latest_result.get('optimization_success', False),
            'total_optimizations': len(self.optimization_history)
        }
    
    def validate_weights(self, weights: Dict[str, float], backtest_results: List[BacktestResult]) -> Dict[str, Any]:
        """Validate optimized weights on new data."""
        try:
            if not backtest_results or not weights:
                return {'validation_score': 0.0, 'status': 'No data for validation'}
            
            # Extract features
            feature_data = {}
            for feature in weights.keys():
                feature_values = [getattr(result, feature, 0.0) for result in backtest_results]
                feature_data[feature] = np.array(feature_values)
            
            # Calculate weighted scores
            target_scores = np.array([result.quality_score for result in backtest_results])
            weighted_scores = np.zeros(len(target_scores))
            
            for feature, weight in weights.items():
                if feature in feature_data:
                    weighted_scores += weight * feature_data[feature]
            
            # Normalize
            weighted_scores = np.clip(weighted_scores, 0.0, 1.0)
            
            # Calculate validation metrics
            r2 = r2_score(target_scores, weighted_scores)
            mse = mean_squared_error(target_scores, weighted_scores)
            correlation = np.corrcoef(target_scores, weighted_scores)[0, 1]
            
            return {
                'validation_score': r2,
                'r2_score': r2,
                'mse': mse,
                'correlation': correlation if not np.isnan(correlation) else 0.0,
                'status': 'Validation completed',
                'samples_validated': len(backtest_results)
            }
            
        except Exception as e:
            self.logger.error(f"Weight validation failed: {e}")
            return {'validation_score': 0.0, 'status': f'Validation failed: {e}'}

def get_weight_optimization_engine(config: Optional[WeightOptimizationConfig] = None) -> WeightOptimizationEngine:
    """Get a weight optimization engine instance."""
    return WeightOptimizationEngine(config)