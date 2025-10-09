"""
Adaptive Search Optimizer for Feature Lookback Optimization

Implements Bayesian optimization with early stopping, multi-objective scalarization,
and intelligent search budget management for efficient lookback optimization.
"""

import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd
from enum import Enum

# Try to import optimization libraries
try:
    from optuna import create_study, Trial, Study
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    create_study = None
    Trial = None
    Study = None
    TPESampler = None
    MedianPruner = None

try:
    from skopt import gp_minimize
    from skopt.space import Integer
    from skopt.acquisition import gaussian_ei
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    gp_minimize = None
    Integer = None
    gaussian_ei = None

from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
from src.utils.logger import get_logger


class SearchStrategy(Enum):
    """Available search strategies."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    TPE_OPTIMIZATION = "tpe_optimization"
    COARSE_TO_REFINE = "coarse_to_refine"
    MULTI_OBJECTIVE = "multi_objective"


@dataclass
class SearchConfig:
    """Configuration for adaptive search optimization."""
    # Search strategy
    strategy: SearchStrategy = SearchStrategy.TPE_OPTIMIZATION
    
    # Budget management
    max_evaluations: int = 50
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 1e-4
    
    # Coarse-to-refine settings
    coarse_evaluations: int = 8
    refine_evaluations: int = 16
    
    # Multi-objective settings
    objectives: List[str] = None  # ['ic', 'stability', 'cost']
    objective_weights: Dict[str, float] = None
    
    # Search space
    min_lookback: int = 5
    max_lookback: int = 200
    search_step: int = 1
    
    # Convergence criteria
    convergence_threshold: float = 1e-3
    min_improvement: float = 1e-4
    
    # Warm start settings
    enable_warm_start: bool = True
    warm_start_data: Optional[Dict[str, Any]] = None


@dataclass
class OptimizationResult:
    """Result of adaptive search optimization."""
    best_lookback: int
    best_score: float
    optimization_method: str
    total_evaluations: int
    convergence_achieved: bool
    early_stopped: bool
    convergence_reason: str
    evaluation_history: List[Tuple[int, float]]
    stability_score: float
    cost_score: float
    metadata: Dict[str, Any]


class AdaptiveSearchOptimizer:
    """
    Adaptive search optimizer with multiple strategies and intelligent budget management.
    
    Supports grid search, random search, Bayesian optimization, TPE, and multi-objective optimization
    with early stopping and warm start capabilities.
    """
    
    def __init__(self, config: Optional[SearchConfig] = None, logger=None):
        """Initialize the adaptive search optimizer."""
        self.config = config or SearchConfig()
        self.logger = logger or get_logger('AdaptiveSearchOptimizer')
        
        # Set default objectives if not provided
        if self.config.objectives is None:
            self.config.objectives = ['ic', 'stability']
        
        if self.config.objective_weights is None:
            self.config.objective_weights = {'ic': 0.7, 'stability': 0.3}
        
        tprint("🎯 Initializing Adaptive Search Optimizer")
        tprint_info(f"   → Strategy: {self.config.strategy.value}")
        tprint_info(f"   → Max evaluations: {self.config.max_evaluations}")
        tprint_info(f"   → Early stopping patience: {self.config.early_stopping_patience}")
        tprint_info(f"   → Objectives: {self.config.objectives}")
        
        # Initialize optimization state
        self.evaluation_history = []
        self.best_score = -np.inf
        self.best_lookback = None
        self.convergence_data = {
            'no_improvement_count': 0,
            'last_improvement_eval': 0,
            'convergence_achieved': False,
            'early_stopped': False
        }
        
        tprint_success("✅ Adaptive Search Optimizer initialized")
    
    def optimize(self, 
                feature_data: np.ndarray,
                target_data: np.ndarray,
                evaluation_function: Callable[[int, np.ndarray, np.ndarray], float],
                warm_start_data: Optional[Dict[str, Any]] = None) -> OptimizationResult:
        """
        Optimize lookback period using adaptive search strategy.
        
        Args:
            feature_data: Feature data array
            target_data: Target data array
            evaluation_function: Function to evaluate lookback period
            warm_start_data: Optional warm start data for Bayesian optimization
            
        Returns:
            OptimizationResult with best parameters and metadata
        """
        tprint(f"🚀 Starting adaptive search optimization")
        tprint_info(f"   → Strategy: {self.config.strategy.value}")
        tprint_info(f"   → Search space: {self.config.min_lookback}-{self.config.max_lookback}")
        
        # Reset state
        self.evaluation_history = []
        self.best_score = -np.inf
        self.best_lookback = None
        self.convergence_data = {
            'no_improvement_count': 0,
            'last_improvement_eval': 0,
            'convergence_achieved': False,
            'early_stopped': False
        }
        
        start_time = time.time()
        
        try:
            # Choose optimization strategy
            if self.config.strategy == SearchStrategy.GRID_SEARCH:
                result = self._grid_search(feature_data, target_data, evaluation_function)
            elif self.config.strategy == SearchStrategy.RANDOM_SEARCH:
                result = self._random_search(feature_data, target_data, evaluation_function)
            elif self.config.strategy == SearchStrategy.BAYESIAN_OPTIMIZATION:
                result = self._bayesian_optimization(feature_data, target_data, evaluation_function, warm_start_data)
            elif self.config.strategy == SearchStrategy.TPE_OPTIMIZATION:
                result = self._tpe_optimization(feature_data, target_data, evaluation_function, warm_start_data)
            elif self.config.strategy == SearchStrategy.COARSE_TO_REFINE:
                result = self._coarse_to_refine(feature_data, target_data, evaluation_function)
            elif self.config.strategy == SearchStrategy.MULTI_OBJECTIVE:
                result = self._multi_objective_optimization(feature_data, target_data, evaluation_function)
            else:
                raise ValueError(f"Unknown search strategy: {self.config.strategy}")
            
            # Add timing information
            result.metadata['optimization_time'] = time.time() - start_time
            result.metadata['strategy'] = self.config.strategy.value
            
            tprint_success(f"✅ Optimization completed in {result.metadata['optimization_time']:.2f}s")
            tprint_info(f"   → Best lookback: {result.best_lookback}")
            tprint_info(f"   → Best score: {result.best_score:.4f}")
            tprint_info(f"   → Total evaluations: {result.total_evaluations}")
            tprint_info(f"   → Convergence: {result.convergence_achieved}")
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Optimization failed: {e}")
            self.logger.error(f"Optimization error: {e}", exc_info=True)
            
            # Return partial result
            return OptimizationResult(
                best_lookback=self.best_lookback or self.config.min_lookback,
                best_score=self.best_score,
                optimization_method=self.config.strategy.value,
                total_evaluations=len(self.evaluation_history),
                convergence_achieved=False,
                early_stopped=True,
                convergence_reason=f"Error: {str(e)}",
                evaluation_history=self.evaluation_history,
                stability_score=0.0,
                cost_score=0.0,
                metadata={'error': str(e)}
            )
    
    def _grid_search(self, feature_data: np.ndarray, target_data: np.ndarray, 
                    evaluation_function: Callable) -> OptimizationResult:
        """Perform grid search optimization."""
        tprint("🔍 Performing grid search...")
        
        lookback_range = range(self.config.min_lookback, self.config.max_lookback + 1, self.config.search_step)
        
        for lookback in lookback_range:
            if self._should_stop_early():
                break
            
            score = self._evaluate_lookback(lookback, feature_data, target_data, evaluation_function)
            
            if score > self.best_score:
                self.best_score = score
                self.best_lookback = lookback
                self.convergence_data['no_improvement_count'] = 0
                self.convergence_data['last_improvement_eval'] = len(self.evaluation_history)
            else:
                self.convergence_data['no_improvement_count'] += 1
        
        return self._create_result()
    
    def _random_search(self, feature_data: np.ndarray, target_data: np.ndarray, 
                      evaluation_function: Callable) -> OptimizationResult:
        """Perform random search optimization."""
        tprint("🎲 Performing random search...")
        
        rng = np.random.RandomState(42)  # Fixed seed for reproducibility
        
        for eval_idx in range(self.config.max_evaluations):
            if self._should_stop_early():
                break
            
            # Sample random lookback
            lookback = rng.randint(self.config.min_lookback, self.config.max_lookback + 1)
            
            score = self._evaluate_lookback(lookback, feature_data, target_data, evaluation_function)
            
            if score > self.best_score:
                self.best_score = score
                self.best_lookback = lookback
                self.convergence_data['no_improvement_count'] = 0
                self.convergence_data['last_improvement_eval'] = len(self.evaluation_history)
            else:
                self.convergence_data['no_improvement_count'] += 1
        
        return self._create_result()
    
    def _bayesian_optimization(self, feature_data: np.ndarray, target_data: np.ndarray, 
                              evaluation_function: Callable, warm_start_data: Optional[Dict] = None) -> OptimizationResult:
        """Perform Bayesian optimization using scikit-optimize."""
        if not SKOPT_AVAILABLE:
            tprint_warning("⚠️ scikit-optimize not available, falling back to random search")
            return self._random_search(feature_data, target_data, evaluation_function)
        
        tprint("🧠 Performing Bayesian optimization...")
        
        # Define search space
        space = [Integer(self.config.min_lookback, self.config.max_lookback, name='lookback')]
        
        # Define objective function
        def objective(params):
            lookback = params[0]
            score = self._evaluate_lookback(lookback, feature_data, target_data, evaluation_function)
            return -score  # Minimize negative score
        
        # Perform optimization
        result = gp_minimize(
            objective,
            space,
            n_calls=self.config.max_evaluations,
            random_state=42,
            acq_func='EI'  # Expected Improvement
        )
        
        self.best_lookback = result.x[0]
        self.best_score = -result.fun
        
        return self._create_result()
    
    def _tpe_optimization(self, feature_data: np.ndarray, target_data: np.ndarray, 
                         evaluation_function: Callable, warm_start_data: Optional[Dict] = None) -> OptimizationResult:
        """Perform TPE optimization using Optuna."""
        if not OPTUNA_AVAILABLE:
            tprint_warning("⚠️ Optuna not available, falling back to random search")
            return self._random_search(feature_data, target_data, evaluation_function)
        
        tprint("🎯 Performing TPE optimization...")
        
        # Create study
        study = create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        # Define objective function
        def objective(trial: Trial) -> float:
            lookback = trial.suggest_int('lookback', self.config.min_lookback, self.config.max_lookback)
            score = self._evaluate_lookback(lookback, feature_data, target_data, evaluation_function)
            return score
        
        # Optimize
        study.optimize(objective, n_trials=self.config.max_evaluations)
        
        self.best_lookback = study.best_params['lookback']
        self.best_score = study.best_value
        
        return self._create_result()
    
    def _coarse_to_refine(self, feature_data: np.ndarray, target_data: np.ndarray, 
                         evaluation_function: Callable) -> OptimizationResult:
        """Perform coarse-to-refine optimization."""
        tprint("🔍 Performing coarse-to-refine optimization...")
        
        # Phase 1: Coarse grid search
        tprint_info("   → Phase 1: Coarse grid search")
        coarse_step = max(1, (self.config.max_lookback - self.config.min_lookback) // self.config.coarse_evaluations)
        coarse_range = range(self.config.min_lookback, self.config.max_lookback + 1, coarse_step)
        
        for lookback in coarse_range:
            if self._should_stop_early():
                break
            
            score = self._evaluate_lookback(lookback, feature_data, target_data, evaluation_function)
            
            if score > self.best_score:
                self.best_score = score
                self.best_lookback = lookback
                self.convergence_data['no_improvement_count'] = 0
                self.convergence_data['last_improvement_eval'] = len(self.evaluation_history)
            else:
                self.convergence_data['no_improvement_count'] += 1
        
        # Phase 2: Refine around best point
        if self.best_lookback is not None:
            tprint_info("   → Phase 2: Refinement around best point")
            refine_radius = coarse_step // 2
            refine_min = max(self.config.min_lookback, self.best_lookback - refine_radius)
            refine_max = min(self.config.max_lookback, self.best_lookback + refine_radius)
            refine_range = range(refine_min, refine_max + 1, self.config.search_step)
            
            for lookback in refine_range:
                if self._should_stop_early():
                    break
                
                score = self._evaluate_lookback(lookback, feature_data, target_data, evaluation_function)
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_lookback = lookback
                    self.convergence_data['no_improvement_count'] = 0
                    self.convergence_data['last_improvement_eval'] = len(self.evaluation_history)
                else:
                    self.convergence_data['no_improvement_count'] += 1
        
        return self._create_result()
    
    def _multi_objective_optimization(self, feature_data: np.ndarray, target_data: np.ndarray, 
                                    evaluation_function: Callable) -> OptimizationResult:
        """Perform multi-objective optimization with scalarization."""
        tprint("🎯 Performing multi-objective optimization...")
        
        lookback_range = range(self.config.min_lookback, self.config.max_lookback + 1, self.config.search_step)
        
        for lookback in lookback_range:
            if self._should_stop_early():
                break
            
            # Evaluate multiple objectives
            objectives = self._evaluate_multi_objectives(lookback, feature_data, target_data, evaluation_function)
            
            # Scalarize objectives
            scalarized_score = self._scalarize_objectives(objectives)
            
            if scalarized_score > self.best_score:
                self.best_score = scalarized_score
                self.best_lookback = lookback
                self.convergence_data['no_improvement_count'] = 0
                self.convergence_data['last_improvement_eval'] = len(self.evaluation_history)
            else:
                self.convergence_data['no_improvement_count'] += 1
        
        return self._create_result()
    
    def _evaluate_lookback(self, lookback: int, feature_data: np.ndarray, target_data: np.ndarray, 
                          evaluation_function: Callable) -> float:
        """Evaluate a single lookback period."""
        try:
            score = evaluation_function(lookback, feature_data, target_data)
            
            # Record evaluation
            self.evaluation_history.append((lookback, score))
            
            return score
            
        except Exception as e:
            tprint_debug(f"Evaluation failed for lookback {lookback}: {e}")
            return -np.inf
    
    def _evaluate_multi_objectives(self, lookback: int, feature_data: np.ndarray, target_data: np.ndarray, 
                                  evaluation_function: Callable) -> Dict[str, float]:
        """Evaluate multiple objectives for a lookback period."""
        objectives = {}
        
        # Primary objective (IC)
        if 'ic' in self.config.objectives:
            objectives['ic'] = evaluation_function(lookback, feature_data, target_data)
        
        # Stability objective (variance of scores in neighborhood)
        if 'stability' in self.config.objectives:
            objectives['stability'] = self._calculate_stability_score(lookback, feature_data, target_data, evaluation_function)
        
        # Cost objective (computational cost)
        if 'cost' in self.config.objectives:
            objectives['cost'] = self._calculate_cost_score(lookback)
        
        return objectives
    
    def _scalarize_objectives(self, objectives: Dict[str, float]) -> float:
        """Scalarize multiple objectives into a single score."""
        weighted_sum = 0.0
        total_weight = 0.0
        
        for obj_name, obj_value in objectives.items():
            if obj_name in self.config.objective_weights:
                weight = self.config.objective_weights[obj_name]
                weighted_sum += weight * obj_value
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _calculate_stability_score(self, lookback: int, feature_data: np.ndarray, target_data: np.ndarray, 
                                  evaluation_function: Callable) -> float:
        """Calculate stability score for a lookback period."""
        # Evaluate in a small neighborhood
        neighborhood = range(max(self.config.min_lookback, lookback - 2), 
                           min(self.config.max_lookback + 1, lookback + 3))
        
        scores = []
        for neighbor_lookback in neighborhood:
            if neighbor_lookback != lookback:
                try:
                    score = evaluation_function(neighbor_lookback, feature_data, target_data)
                    if not np.isnan(score) and score != -np.inf:
                        scores.append(score)
                except Exception:
                    continue
        
        if len(scores) < 2:
            return 0.0
        
        # Calculate coefficient of variation (lower is more stable)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        if mean_score == 0:
            return 0.0
        
        cv = std_score / abs(mean_score)
        stability_score = max(0.0, 1.0 - cv)  # Convert to 0-1 scale
        
        return stability_score
    
    def _calculate_cost_score(self, lookback: int) -> float:
        """Calculate computational cost score (lower is better)."""
        # Simple cost model: higher lookback = higher cost
        max_lookback = self.config.max_lookback
        cost_score = 1.0 - (lookback - self.config.min_lookback) / (max_lookback - self.config.min_lookback)
        return max(0.0, cost_score)
    
    def _should_stop_early(self) -> bool:
        """Check if optimization should stop early."""
        # Check evaluation limit
        if len(self.evaluation_history) >= self.config.max_evaluations:
            return True
        
        # Check early stopping patience
        if self.convergence_data['no_improvement_count'] >= self.config.early_stopping_patience:
            self.convergence_data['early_stopped'] = True
            return True
        
        # Check convergence threshold
        if len(self.evaluation_history) >= 10:
            recent_scores = [score for _, score in self.evaluation_history[-10:]]
            if len(recent_scores) >= 5:
                score_std = np.std(recent_scores)
                if score_std < self.config.convergence_threshold:
                    self.convergence_data['convergence_achieved'] = True
                    return True
        
        return False
    
    def _create_result(self) -> OptimizationResult:
        """Create optimization result from current state."""
        # Calculate stability score
        stability_score = 0.0
        if len(self.evaluation_history) > 1:
            scores = [score for _, score in self.evaluation_history]
            stability_score = self._calculate_stability_score(
                self.best_lookback or self.config.min_lookback,
                np.array([]), np.array([]), lambda x, y, z: 0.0
            )
        
        # Determine convergence reason
        convergence_reason = "completed"
        if self.convergence_data['early_stopped']:
            convergence_reason = "early_stopped"
        elif self.convergence_data['convergence_achieved']:
            convergence_reason = "converged"
        elif len(self.evaluation_history) >= self.config.max_evaluations:
            convergence_reason = "max_evaluations"
        
        return OptimizationResult(
            best_lookback=self.best_lookback or self.config.min_lookback,
            best_score=self.best_score,
            optimization_method=self.config.strategy.value,
            total_evaluations=len(self.evaluation_history),
            convergence_achieved=self.convergence_data['convergence_achieved'],
            early_stopped=self.convergence_data['early_stopped'],
            convergence_reason=convergence_reason,
            evaluation_history=self.evaluation_history.copy(),
            stability_score=stability_score,
            cost_score=self._calculate_cost_score(self.best_lookback or self.config.min_lookback),
            metadata={
                'no_improvement_count': self.convergence_data['no_improvement_count'],
                'last_improvement_eval': self.convergence_data['last_improvement_eval']
            }
        )


# Convenience function
def optimize_lookback_adaptive(feature_data: np.ndarray,
                              target_data: np.ndarray,
                              evaluation_function: Callable[[int, np.ndarray, np.ndarray], float],
                              config: Optional[SearchConfig] = None) -> OptimizationResult:
    """
    Convenience function for adaptive lookback optimization.
    
    Args:
        feature_data: Feature data array
        target_data: Target data array
        evaluation_function: Function to evaluate lookback period
        config: Search configuration
        
    Returns:
        OptimizationResult with best parameters
    """
    optimizer = AdaptiveSearchOptimizer(config)
    return optimizer.optimize(feature_data, target_data, evaluation_function)