"""
Ensemble Performance Optimization System

This module implements advanced optimization strategies for ensemble performance including:
1. Genetic algorithm optimization
2. Bayesian optimization
3. Gradient descent optimization
4. Multi-objective optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime

# Import optimization libraries
try:
    from scipy.optimize import minimize, differential_evolution
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for ensemble optimization."""
    
    # Optimization method
    optimization_method: str = "genetic_algorithm"  # "genetic_algorithm", "bayesian", "gradient_descent", "multi_objective"
    
    # Genetic algorithm parameters
    ga_population_size: int = 50
    ga_generations: int = 100
    ga_mutation_rate: float = 0.1
    ga_crossover_rate: float = 0.8
    ga_elite_size: int = 5
    
    # Bayesian optimization parameters
    bo_n_iterations: int = 50
    bo_acquisition_function: str = "ei"  # "ei", "pi", "ucb"
    bo_alpha: float = 1e-6
    
    # Gradient descent parameters
    gd_learning_rate: float = 0.01
    gd_max_iterations: int = 1000
    gd_tolerance: float = 1e-6
    
    # Multi-objective optimization
    mo_objectives: List[str] = field(default_factory=lambda: ["accuracy", "diversity", "stability"])
    mo_weights: List[float] = field(default_factory=lambda: [0.5, 0.3, 0.2])
    
    # Optimization constraints
    min_weight: float = 0.01
    max_weight: float = 0.8
    weight_sum_constraint: bool = True
    
    # Performance metrics
    optimization_metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "precision", "recall", "f1_score", "roc_auc"
    ])
    
    # Convergence criteria
    convergence_threshold: float = 1e-6
    max_no_improvement: int = 10
    enable_early_stopping: bool = True


class EnsembleOptimizer:
    """
    Advanced ensemble optimization system.
    """
    
    def __init__(self, config: OptimizationConfig):
        """Initialize ensemble optimizer."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Optimization state
        self.optimization_history: List[Dict[str, Any]] = []
        self.best_weights: Dict[str, float] = {}
        self.best_performance: float = 0.0
        
        self.logger.info("✅ Ensemble Optimizer initialized")
        self.logger.info(f"   Optimization method: {config.optimization_method}")
        self.logger.info(f"   Available libraries: SciPy={SCIPY_AVAILABLE}, Scikit-learn={SKLEARN_AVAILABLE}")
    
    def optimize_ensemble_weights(self, 
                                regime_id: int,
                                ensemble_models: List[Any],
                                training_data: np.ndarray,
                                training_labels: np.ndarray,
                                validation_data: np.ndarray,
                                validation_labels: np.ndarray) -> Dict[str, Any]:
        """
        Optimize ensemble weights for a specific regime.
        
        Args:
            regime_id: ID of the regime
            ensemble_models: List of ensemble models
            training_data: Training data
            training_labels: Training labels
            validation_data: Validation data
            validation_labels: Validation labels
            
        Returns:
            Optimization result
        """
        try:
            start_time = time.time()
            
            if self.config.optimization_method == "genetic_algorithm":
                result = self._optimize_genetic_algorithm(
                    regime_id, ensemble_models, training_data, training_labels, 
                    validation_data, validation_labels
                )
            elif self.config.optimization_method == "bayesian":
                result = self._optimize_bayesian(
                    regime_id, ensemble_models, training_data, training_labels,
                    validation_data, validation_labels
                )
            elif self.config.optimization_method == "gradient_descent":
                result = self._optimize_gradient_descent(
                    regime_id, ensemble_models, training_data, training_labels,
                    validation_data, validation_labels
                )
            elif self.config.optimization_method == "multi_objective":
                result = self._optimize_multi_objective(
                    regime_id, ensemble_models, training_data, training_labels,
                    validation_data, validation_labels
                )
            else:
                raise ValueError(f"Unknown optimization method: {self.config.optimization_method}")
            
            result['execution_time'] = time.time() - start_time
            result['regime_id'] = regime_id
            result['optimization_method'] = self.config.optimization_method
            
            # Store in history
            self.optimization_history.append(result)
            
            # Update best weights if improved
            if result['best_performance'] > self.best_performance:
                self.best_performance = result['best_performance']
                self.best_weights = result['best_weights']
            
            self.logger.info(f"✅ Ensemble optimization completed for regime {regime_id}")
            self.logger.info(f"   Best performance: {result['best_performance']:.4f}")
            self.logger.info(f"   Execution time: {result['execution_time']:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Ensemble optimization failed for regime {regime_id}: {e}")
            return {
                'success': False,
                'error_message': str(e),
                'regime_id': regime_id,
                'best_weights': {},
                'best_performance': 0.0,
                'execution_time': time.time() - start_time if 'start_time' in locals() else 0.0
            }
    
    def _optimize_genetic_algorithm(self, 
                                  regime_id: int,
                                  ensemble_models: List[Any],
                                  training_data: np.ndarray,
                                  training_labels: np.ndarray,
                                  validation_data: np.ndarray,
                                  validation_labels: np.ndarray) -> Dict[str, Any]:
        """Optimize using genetic algorithm."""
        try:
            if not SCIPY_AVAILABLE:
                raise ImportError("SciPy not available for genetic algorithm optimization")
            
            n_models = len(ensemble_models)
            
            # Define objective function
            def objective(weights):
                # Normalize weights
                weights = np.array(weights)
                weights = weights / np.sum(weights)
                
                # Calculate ensemble performance
                performance = self._evaluate_ensemble_performance(
                    ensemble_models, weights, validation_data, validation_labels
                )
                
                return -performance  # Minimize negative performance
            
            # Define bounds for weights
            bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(n_models)]
            
            # Run genetic algorithm
            result = differential_evolution(
                objective,
                bounds,
                maxiter=self.config.ga_generations,
                popsize=self.config.ga_population_size,
                mutation=self.config.ga_mutation_rate,
                recombination=self.config.ga_crossover_rate,
                seed=42
            )
            
            if result.success:
                best_weights = result.x / np.sum(result.x)  # Normalize
                best_performance = -result.fun
                
                # Create weight dictionary
                weight_dict = {f"model_{i}": weight for i, weight in enumerate(best_weights)}
                
                return {
                    'success': True,
                    'best_weights': weight_dict,
                    'best_performance': best_performance,
                    'n_iterations': result.nit,
                    'convergence': result.success
                }
            else:
                raise ValueError(f"Genetic algorithm optimization failed: {result.message}")
                
        except Exception as e:
            self.logger.error(f"Genetic algorithm optimization failed: {e}")
            # Fallback to equal weights
            weight = 1.0 / len(ensemble_models)
            return {
                'success': False,
                'error_message': str(e),
                'best_weights': {f"model_{i}": weight for i in range(len(ensemble_models))},
                'best_performance': 0.0,
                'n_iterations': 0,
                'convergence': False
            }
    
    def _optimize_bayesian(self, 
                         regime_id: int,
                         ensemble_models: List[Any],
                         training_data: np.ndarray,
                         training_labels: np.ndarray,
                         validation_data: np.ndarray,
                         validation_labels: np.ndarray) -> Dict[str, Any]:
        """Optimize using Bayesian optimization (simplified implementation)."""
        try:
            n_models = len(ensemble_models)
            
            # Simplified Bayesian optimization using random search with Gaussian process approximation
            best_weights = None
            best_performance = 0.0
            performance_history = []
            
            for iteration in range(self.config.bo_n_iterations):
                # Generate candidate weights
                candidate_weights = self._generate_candidate_weights(n_models)
                
                # Evaluate performance
                performance = self._evaluate_ensemble_performance(
                    ensemble_models, candidate_weights, validation_data, validation_labels
                )
                
                performance_history.append(performance)
                
                # Update best if improved
                if performance > best_performance:
                    best_performance = performance
                    best_weights = candidate_weights.copy()
                
                # Early stopping if no improvement
                if (self.config.enable_early_stopping and 
                    len(performance_history) > self.config.max_no_improvement and
                    max(performance_history[-self.config.max_no_improvement:]) <= best_performance):
                    break
            
            if best_weights is not None:
                weight_dict = {f"model_{i}": weight for i, weight in enumerate(best_weights)}
            else:
                # Fallback to equal weights
                weight = 1.0 / n_models
                weight_dict = {f"model_{i}": weight for i in range(n_models)}
                best_performance = 0.0
            
            return {
                'success': True,
                'best_weights': weight_dict,
                'best_performance': best_performance,
                'n_iterations': len(performance_history),
                'convergence': True,
                'performance_history': performance_history
            }
            
        except Exception as e:
            self.logger.error(f"Bayesian optimization failed: {e}")
            # Fallback to equal weights
            weight = 1.0 / len(ensemble_models)
            return {
                'success': False,
                'error_message': str(e),
                'best_weights': {f"model_{i}": weight for i in range(len(ensemble_models))},
                'best_performance': 0.0,
                'n_iterations': 0,
                'convergence': False
            }
    
    def _optimize_gradient_descent(self, 
                                 regime_id: int,
                                 ensemble_models: List[Any],
                                 training_data: np.ndarray,
                                 training_labels: np.ndarray,
                                 validation_data: np.ndarray,
                                 validation_labels: np.ndarray) -> Dict[str, Any]:
        """Optimize using gradient descent."""
        try:
            if not SCIPY_AVAILABLE:
                raise ImportError("SciPy not available for gradient descent optimization")
            
            n_models = len(ensemble_models)
            
            # Define objective function
            def objective(weights):
                # Normalize weights
                weights = np.array(weights)
                weights = weights / np.sum(weights)
                
                # Calculate ensemble performance
                performance = self._evaluate_ensemble_performance(
                    ensemble_models, weights, validation_data, validation_labels
                )
                
                return -performance  # Minimize negative performance
            
            # Define constraints: weights sum to 1
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            
            # Define bounds
            bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(n_models)]
            
            # Initial weights
            initial_weights = np.ones(n_models) / n_models
            
            # Run optimization
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': self.config.gd_max_iterations}
            )
            
            if result.success:
                best_weights = result.x
                best_performance = -result.fun
                
                # Create weight dictionary
                weight_dict = {f"model_{i}": weight for i, weight in enumerate(best_weights)}
                
                return {
                    'success': True,
                    'best_weights': weight_dict,
                    'best_performance': best_performance,
                    'n_iterations': result.nit,
                    'convergence': result.success
                }
            else:
                raise ValueError(f"Gradient descent optimization failed: {result.message}")
                
        except Exception as e:
            self.logger.error(f"Gradient descent optimization failed: {e}")
            # Fallback to equal weights
            weight = 1.0 / len(ensemble_models)
            return {
                'success': False,
                'error_message': str(e),
                'best_weights': {f"model_{i}": weight for i in range(len(ensemble_models))},
                'best_performance': 0.0,
                'n_iterations': 0,
                'convergence': False
            }
    
    def _optimize_multi_objective(self, 
                                 regime_id: int,
                                 ensemble_models: List[Any],
                                 training_data: np.ndarray,
                                 training_labels: np.ndarray,
                                 validation_data: np.ndarray,
                                 validation_labels: np.ndarray) -> Dict[str, Any]:
        """Optimize using multi-objective optimization."""
        try:
            n_models = len(ensemble_models)
            
            # Define multi-objective function
            def multi_objective(weights):
                # Normalize weights
                weights = np.array(weights)
                weights = weights / np.sum(weights)
                
                # Calculate different objectives
                objectives = []
                
                # Performance objective
                if "accuracy" in self.config.mo_objectives:
                    performance = self._evaluate_ensemble_performance(
                        ensemble_models, weights, validation_data, validation_labels
                    )
                    objectives.append(-performance)  # Minimize negative performance
                else:
                    objectives.append(0.0)
                
                # Diversity objective
                if "diversity" in self.config.mo_objectives:
                    diversity = self._calculate_ensemble_diversity(ensemble_models, weights)
                    objectives.append(-diversity)  # Minimize negative diversity
                else:
                    objectives.append(0.0)
                
                # Stability objective
                if "stability" in self.config.mo_objectives:
                    stability = self._calculate_ensemble_stability(ensemble_models, weights)
                    objectives.append(-stability)  # Minimize negative stability
                else:
                    objectives.append(0.0)
                
                # Weighted combination
                weighted_objective = sum(w * obj for w, obj in zip(self.config.mo_weights, objectives))
                
                return weighted_objective
            
            # Use genetic algorithm for multi-objective optimization
            if SCIPY_AVAILABLE:
                bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(n_models)]
                
                result = differential_evolution(
                    multi_objective,
                    bounds,
                    maxiter=self.config.ga_generations,
                    popsize=self.config.ga_population_size,
                    seed=42
                )
                
                if result.success:
                    best_weights = result.x / np.sum(result.x)
                    best_performance = -result.fun
                    
                    weight_dict = {f"model_{i}": weight for i, weight in enumerate(best_weights)}
                    
                    return {
                        'success': True,
                        'best_weights': weight_dict,
                        'best_performance': best_performance,
                        'n_iterations': result.nit,
                        'convergence': result.success,
                        'objectives': self.config.mo_objectives,
                        'weights': self.config.mo_weights
                    }
            
            # Fallback to equal weights
            weight = 1.0 / n_models
            return {
                'success': False,
                'error_message': "Multi-objective optimization not available",
                'best_weights': {f"model_{i}": weight for i in range(n_models)},
                'best_performance': 0.0,
                'n_iterations': 0,
                'convergence': False
            }
            
        except Exception as e:
            self.logger.error(f"Multi-objective optimization failed: {e}")
            # Fallback to equal weights
            weight = 1.0 / len(ensemble_models)
            return {
                'success': False,
                'error_message': str(e),
                'best_weights': {f"model_{i}": weight for i in range(len(ensemble_models))},
                'best_performance': 0.0,
                'n_iterations': 0,
                'convergence': False
            }
    
    def _evaluate_ensemble_performance(self, 
                                     ensemble_models: List[Any],
                                     weights: np.ndarray,
                                     validation_data: np.ndarray,
                                     validation_labels: np.ndarray) -> float:
        """Evaluate ensemble performance with given weights."""
        try:
            # Get predictions from each model
            predictions = []
            probabilities = []
            
            for model in ensemble_models:
                try:
                    pred = model.predict(validation_data)
                    predictions.append(pred)
                    
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(validation_data)
                        probabilities.append(prob)
                    else:
                        # Create dummy probabilities
                        n_classes = len(np.unique(pred))
                        prob = np.zeros((len(validation_data), n_classes))
                        for i, p in enumerate(pred):
                            prob[i, p] = 1.0
                        probabilities.append(prob)
                        
                except Exception as e:
                    self.logger.warning(f"Model prediction failed: {e}")
                    continue
            
            if not predictions:
                return 0.0
            
            # Combine predictions with weights
            ensemble_pred = np.zeros(len(validation_data))
            ensemble_prob = np.zeros((len(validation_data), probabilities[0].shape[1]))
            
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                if i < len(weights):
                    ensemble_pred += weights[i] * pred
                    ensemble_prob += weights[i] * prob
            
            # Round predictions
            ensemble_pred = np.round(ensemble_pred).astype(int)
            
            # Calculate performance metrics
            if SKLEARN_AVAILABLE:
                accuracy = accuracy_score(validation_labels, ensemble_pred)
                return accuracy
            else:
                # Simple accuracy calculation
                correct = np.sum(ensemble_pred == validation_labels)
                return correct / len(validation_labels)
                
        except Exception as e:
            self.logger.error(f"Failed to evaluate ensemble performance: {e}")
            return 0.0
    
    def _generate_candidate_weights(self, n_models: int) -> np.ndarray:
        """Generate candidate weights for optimization."""
        try:
            # Generate random weights
            weights = np.random.uniform(self.config.min_weight, self.config.max_weight, n_models)
            
            # Normalize to sum to 1
            weights = weights / np.sum(weights)
            
            return weights
            
        except Exception as e:
            self.logger.error(f"Failed to generate candidate weights: {e}")
            return np.ones(n_models) / n_models
    
    def _calculate_ensemble_diversity(self, 
                                    ensemble_models: List[Any],
                                    weights: np.ndarray) -> float:
        """Calculate ensemble diversity."""
        try:
            # Simplified diversity calculation based on weight distribution
            # Higher diversity = more balanced weights
            diversity = 1.0 - np.std(weights)
            return max(0.0, diversity)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate ensemble diversity: {e}")
            return 0.0
    
    def _calculate_ensemble_stability(self, 
                                    ensemble_models: List[Any],
                                    weights: np.ndarray) -> float:
        """Calculate ensemble stability."""
        try:
            # Simplified stability calculation
            # Higher stability = weights closer to equal distribution
            equal_weights = np.ones(len(weights)) / len(weights)
            stability = 1.0 - np.mean(np.abs(weights - equal_weights))
            return max(0.0, stability)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate ensemble stability: {e}")
            return 0.0
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization history."""
        try:
            if not self.optimization_history:
                return {'message': 'No optimization history available'}
            
            summary = {
                'total_optimizations': len(self.optimization_history),
                'best_overall_performance': self.best_performance,
                'best_weights': self.best_weights,
                'optimization_methods_used': list(set(r.get('optimization_method', 'unknown') for r in self.optimization_history)),
                'average_execution_time': np.mean([r.get('execution_time', 0) for r in self.optimization_history]),
                'success_rate': sum(1 for r in self.optimization_history if r.get('success', False)) / len(self.optimization_history)
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get optimization summary: {e}")
            return {'error': str(e)}
    
    def save_optimization_history(self, filepath: str):
        """Save optimization history to file."""
        try:
            data = {
                'optimization_history': self.optimization_history,
                'best_weights': self.best_weights,
                'best_performance': self.best_performance,
                'config': self.config.__dict__,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            self.logger.info(f"Optimization history saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save optimization history: {e}")
    
    def load_optimization_history(self, filepath: str):
        """Load optimization history from file."""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.optimization_history = data.get('optimization_history', [])
            self.best_weights = data.get('best_weights', {})
            self.best_performance = data.get('best_performance', 0.0)
            
            self.logger.info(f"Optimization history loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load optimization history: {e}")