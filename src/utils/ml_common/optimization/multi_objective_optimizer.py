"""
Multi-Objective Hyperparameter Optimization

This module provides comprehensive multi-objective optimization capabilities
for hyperparameter tuning, including Pareto front analysis, NSGA-II implementation,
and multi-objective early stopping.

Enhancement: Multi-objective optimization for HPO
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import time
from collections import defaultdict
import json
from pathlib import Path

# Try to import optimization libraries
try:
    import optuna
    from optuna.samplers import NSGAIISampler, TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MultiObjectiveConfig:
    """Configuration for multi-objective optimization."""
    
    # Optimization settings
    n_trials: int = 100
    n_objectives: int = 2
    direction: str = 'maximize'  # 'maximize' or 'minimize'
    
    # Multi-objective specific
    pareto_front_size: int = 50
    diversity_weight: float = 0.1
    convergence_weight: float = 0.9
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_threshold: float = 0.001
    
    # NSGA-II settings
    population_size: int = 50
    mutation_probability: float = 0.1
    crossover_probability: float = 0.9
    
    # Performance tracking
    track_pareto_front: bool = True
    save_pareto_history: bool = True
    pareto_history_file: str = "pareto_front_history.json"
    
    # Resource management
    max_evaluations: int = 1000
    timeout_seconds: Optional[float] = None
    
    # Warm starting
    enable_warm_start: bool = True
    warm_start_file: Optional[str] = None


@dataclass
class ObjectiveFunction:
    """Represents a single objective function."""
    
    name: str
    function: Callable
    weight: float = 1.0
    direction: str = 'maximize'  # 'maximize' or 'minimize'
    bounds: Optional[Tuple[float, float]] = None
    
    def evaluate(self, params: Dict[str, Any], **kwargs) -> float:
        """Evaluate the objective function."""
        try:
            result = self.function(params, **kwargs)
            # Apply direction
            if self.direction == 'minimize':
                return -result
            return result
        except Exception as e:
            logger.warning(f"Objective {self.name} evaluation failed: {e}")
            return float('-inf') if self.direction == 'maximize' else float('inf')


@dataclass
class ParetoSolution:
    """Represents a solution on the Pareto front."""
    
    params: Dict[str, Any]
    objectives: List[float]
    rank: int = 0
    crowding_distance: float = 0.0
    dominated_count: int = 0
    dominates: List[int] = field(default_factory=list)
    
    def dominates_solution(self, other: 'ParetoSolution') -> bool:
        """Check if this solution dominates another."""
        if len(self.objectives) != len(other.objectives):
            return False
        
        # At least one objective is better
        better_count = 0
        for i, (self_obj, other_obj) in enumerate(zip(self.objectives, other.objectives)):
            if self_obj > other_obj:
                better_count += 1
            elif self_obj < other_obj:
                return False  # This solution is worse in at least one objective
        
        return better_count > 0
    
    def __lt__(self, other):
        """Comparison for sorting (used in NSGA-II)."""
        if self.rank != other.rank:
            return self.rank < other.rank
        return self.crowding_distance > other.crowding_distance


class ParetoFrontManager:
    """Manages the Pareto front of solutions."""
    
    def __init__(self, config: MultiObjectiveConfig):
        self.config = config
        self.solutions: List[ParetoSolution] = []
        self.pareto_fronts: List[List[ParetoSolution]] = []
        self.history: List[Dict[str, Any]] = []
        
    def add_solution(self, params: Dict[str, Any], objectives: List[float]) -> ParetoSolution:
        """Add a new solution to the Pareto front."""
        solution = ParetoSolution(
            params=params.copy(),
            objectives=objectives.copy()
        )
        
        # Update dominance relationships
        self._update_dominance_relationships(solution)
        
        # Add to solutions list
        self.solutions.append(solution)
        
        # Update Pareto fronts
        self._update_pareto_fronts()
        
        # Track history
        if self.config.track_pareto_front:
            self._track_history(solution)
        
        return solution
    
    def _update_dominance_relationships(self, new_solution: ParetoSolution):
        """Update dominance relationships for all solutions."""
        for i, existing_solution in enumerate(self.solutions):
            if new_solution.dominates_solution(existing_solution):
                new_solution.dominates.append(i)
                existing_solution.dominated_count += 1
            elif existing_solution.dominates_solution(new_solution):
                existing_solution.dominates.append(len(self.solutions))
                new_solution.dominated_count += 1
    
    def _update_pareto_fronts(self):
        """Update the Pareto front hierarchy."""
        self.pareto_fronts = []
        remaining_solutions = self.solutions.copy()
        current_rank = 0
        
        while remaining_solutions:
            # Find non-dominated solutions
            non_dominated = []
            for solution in remaining_solutions:
                if solution.dominated_count == 0:
                    solution.rank = current_rank
                    non_dominated.append(solution)
            
            if not non_dominated:
                break
            
            self.pareto_fronts.append(non_dominated)
            
            # Update dominated counts for remaining solutions
            for solution in non_dominated:
                for dominated_idx in solution.dominates:
                    if dominated_idx < len(self.solutions):
                        self.solutions[dominated_idx].dominated_count -= 1
            
            # Remove non-dominated solutions from remaining
            remaining_solutions = [s for s in remaining_solutions if s not in non_dominated]
            current_rank += 1
    
    def _track_history(self, solution: ParetoSolution):
        """Track Pareto front history."""
        history_entry = {
            'timestamp': time.time(),
            'solution_count': len(self.solutions),
            'pareto_front_size': len(self.pareto_fronts[0]) if self.pareto_fronts else 0,
            'objectives': solution.objectives.copy(),
            'params': solution.params.copy()
        }
        self.history.append(history_entry)
        
        # Save to file if configured
        if self.config.save_pareto_history:
            self._save_history()
    
    def _save_history(self):
        """Save Pareto front history to file."""
        try:
            with open(self.config.pareto_history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save Pareto history: {e}")
    
    def get_pareto_front(self) -> List[ParetoSolution]:
        """Get the first (best) Pareto front."""
        return self.pareto_fronts[0] if self.pareto_fronts else []
    
    def get_diverse_solutions(self, n_solutions: int) -> List[ParetoSolution]:
        """Get diverse solutions from the Pareto front."""
        if not self.pareto_fronts:
            return []
        
        front = self.pareto_fronts[0]
        if len(front) <= n_solutions:
            return front
        
        # Calculate crowding distances
        self._calculate_crowding_distances(front)
        
        # Sort by crowding distance and return top solutions
        sorted_front = sorted(front, key=lambda x: x.crowding_distance, reverse=True)
        return sorted_front[:n_solutions]
    
    def _calculate_crowding_distances(self, solutions: List[ParetoSolution]):
        """Calculate crowding distances for solutions."""
        if len(solutions) <= 2:
            for solution in solutions:
                solution.crowding_distance = float('inf')
            return
        
        n_objectives = len(solutions[0].objectives)
        
        for solution in solutions:
            solution.crowding_distance = 0.0
        
        for obj_idx in range(n_objectives):
            # Sort solutions by this objective
            sorted_solutions = sorted(solutions, key=lambda x: x.objectives[obj_idx])
            
            # Set boundary solutions to infinite distance
            sorted_solutions[0].crowding_distance = float('inf')
            sorted_solutions[-1].crowding_distance = float('inf')
            
            # Calculate distances for intermediate solutions
            obj_range = sorted_solutions[-1].objectives[obj_idx] - sorted_solutions[0].objectives[obj_idx]
            if obj_range > 0:
                for i in range(1, len(sorted_solutions) - 1):
                    distance = (sorted_solutions[i + 1].objectives[obj_idx] - 
                              sorted_solutions[i - 1].objectives[obj_idx]) / obj_range
                    sorted_solutions[i].crowding_distance += distance


class MultiObjectiveOptimizer:
    """Multi-objective hyperparameter optimizer."""
    
    def __init__(self, config: MultiObjectiveConfig):
        self.config = config
        self.objectives: List[ObjectiveFunction] = []
        self.pareto_manager = ParetoFrontManager(config)
        self.optimization_history: List[Dict[str, Any]] = []
        self.best_solutions: List[ParetoSolution] = []
        
        logger.info(f"Multi-objective optimizer initialized with {config.n_objectives} objectives")
    
    def add_objective(self, name: str, function: Callable, 
                     weight: float = 1.0, direction: str = 'maximize',
                     bounds: Optional[Tuple[float, float]] = None):
        """Add an objective function."""
        objective = ObjectiveFunction(
            name=name,
            function=function,
            weight=weight,
            direction=direction,
            bounds=bounds
        )
        self.objectives.append(objective)
        logger.info(f"Added objective: {name} (weight={weight}, direction={direction})")
    
    def optimize(self, search_space: Dict[str, Any], 
                model_factory: Callable,
                X: Any, y: Any,
                warm_start_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform multi-objective optimization.
        
        Args:
            search_space: Parameter search space
            model_factory: Function to create model instances
            X: Training features
            y: Training targets
            warm_start_data: Previous optimization results for warm starting
            
        Returns:
            Optimization results including Pareto front
        """
        if not self.objectives:
            raise ValueError("No objectives defined. Use add_objective() to add objectives.")
        
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for multi-objective optimization")
        
        start_time = time.time()
        logger.info(f"Starting multi-objective optimization with {len(self.objectives)} objectives")
        
        # Warm start if enabled
        if self.config.enable_warm_start and warm_start_data:
            self._warm_start(warm_start_data)
        
        # Create Optuna study
        study = self._create_study()
        
        # Define objective function
        def objective(trial):
            # Sample parameters
            params = self._sample_parameters(trial, search_space)
            
            # Create and evaluate model
            model = model_factory(**params)
            
            # Evaluate all objectives
            objective_values = []
            for obj_func in self.objectives:
                try:
                    value = obj_func.evaluate(params, model=model, X=X, y=y)
                    objective_values.append(value)
                except Exception as e:
                    logger.warning(f"Objective {obj_func.name} evaluation failed: {e}")
                    objective_values.append(float('-inf'))
            
            # Add to Pareto front
            solution = self.pareto_manager.add_solution(params, objective_values)
            
            # Return weighted sum for Optuna (NSGA-II will handle multi-objective)
            weighted_sum = sum(obj.weight * val for obj, val in zip(self.objectives, objective_values))
            return weighted_sum
        
        # Run optimization
        try:
            study.optimize(
                objective, 
                n_trials=self.config.n_trials,
                timeout=self.config.timeout_seconds
            )
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise
        
        # Extract results
        results = self._extract_results(study, time.time() - start_time)
        
        logger.info(f"Multi-objective optimization completed in {results['optimization_time']:.2f}s")
        logger.info(f"Found {len(results['pareto_front'])} solutions on Pareto front")
        
        return results
    
    def _create_study(self) -> optuna.Study:
        """Create Optuna study for multi-objective optimization."""
        # Use NSGA-II sampler for multi-objective optimization
        sampler = NSGAIISampler(
            population_size=self.config.population_size,
            mutation_probability=self.config.mutation_probability,
            crossover_probability=self.config.crossover_probability,
            seed=42
        )
        
        # Create study
        study = optuna.create_study(
            directions=['maximize'] * len(self.objectives),
            sampler=sampler
        )
        
        return study
    
    def _sample_parameters(self, trial, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Sample parameters from search space."""
        params = {}
        
        for param_name, param_config in search_space.items():
            if isinstance(param_config, dict):
                param_type = param_config.get('type', 'float')
                
                if param_type == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        log=param_config.get('log', False)
                    )
                elif param_type == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        log=param_config.get('log', False)
                    )
                elif param_type == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config['choices']
                    )
            else:
                # Handle simple parameter definitions
                if isinstance(param_config, (list, tuple)) and len(param_config) == 2:
                    # Assume (low, high) range
                    params[param_name] = trial.suggest_float(
                        param_name, param_config[0], param_config[1]
                    )
                elif isinstance(param_config, list):
                    # Assume categorical
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config
                    )
        
        return params
    
    def _warm_start(self, warm_start_data: Dict[str, Any]):
        """Warm start optimization with previous results."""
        try:
            if 'pareto_front' in warm_start_data:
                for solution_data in warm_start_data['pareto_front']:
                    params = solution_data.get('params', {})
                    objectives = solution_data.get('objectives', [])
                    if params and objectives:
                        self.pareto_manager.add_solution(params, objectives)
                
                logger.info(f"Warm started with {len(warm_start_data['pareto_front'])} solutions")
        except Exception as e:
            logger.warning(f"Warm start failed: {e}")
    
    def _extract_results(self, study: optuna.Study, optimization_time: float) -> Dict[str, Any]:
        """Extract optimization results."""
        pareto_front = self.pareto_manager.get_pareto_front()
        
        # Get diverse solutions
        diverse_solutions = self.pareto_manager.get_diverse_solutions(
            min(self.config.pareto_front_size, len(pareto_front))
        )
        
        # Calculate metrics
        metrics = self._calculate_metrics(pareto_front)
        
        results = {
            'pareto_front': [
                {
                    'params': solution.params,
                    'objectives': solution.objectives,
                    'rank': solution.rank,
                    'crowding_distance': solution.crowding_distance
                }
                for solution in pareto_front
            ],
            'diverse_solutions': [
                {
                    'params': solution.params,
                    'objectives': solution.objectives,
                    'rank': solution.rank,
                    'crowding_distance': solution.crowding_distance
                }
                for solution in diverse_solutions
            ],
            'metrics': metrics,
            'optimization_time': optimization_time,
            'n_trials': len(study.trials),
            'n_objectives': len(self.objectives),
            'objective_names': [obj.name for obj in self.objectives]
        }
        
        return results
    
    def _calculate_metrics(self, pareto_front: List[ParetoSolution]) -> Dict[str, float]:
        """Calculate Pareto front metrics."""
        if not pareto_front:
            return {}
        
        metrics = {}
        
        # Hypervolume (simplified calculation)
        if len(pareto_front) > 1:
            objectives_matrix = np.array([sol.objectives for sol in pareto_front])
            metrics['hypervolume'] = self._calculate_hypervolume(objectives_matrix)
        
        # Spread (diversity metric)
        metrics['spread'] = self._calculate_spread(pareto_front)
        
        # Number of solutions
        metrics['n_solutions'] = len(pareto_front)
        
        # Objective ranges
        if pareto_front:
            objectives_matrix = np.array([sol.objectives for sol in pareto_front])
            for i, obj_name in enumerate([obj.name for obj in self.objectives]):
                metrics[f'{obj_name}_min'] = float(np.min(objectives_matrix[:, i]))
                metrics[f'{obj_name}_max'] = float(np.max(objectives_matrix[:, i]))
                metrics[f'{obj_name}_range'] = float(np.max(objectives_matrix[:, i]) - np.min(objectives_matrix[:, i]))
        
        return metrics
    
    def _calculate_hypervolume(self, objectives_matrix: np.ndarray) -> float:
        """Calculate hypervolume metric (simplified)."""
        # This is a simplified hypervolume calculation
        # In practice, you'd use a proper hypervolume implementation
        if len(objectives_matrix) < 2:
            return 0.0
        
        # Use the area of the convex hull as a proxy for hypervolume
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(objectives_matrix)
            return hull.volume
        except ImportError:
            # Fallback: use bounding box volume
            return np.prod(np.max(objectives_matrix, axis=0) - np.min(objectives_matrix, axis=0))
    
    def _calculate_spread(self, pareto_front: List[ParetoSolution]) -> float:
        """Calculate spread (diversity) metric."""
        if len(pareto_front) < 2:
            return 0.0
        
        # Calculate average distance between solutions
        distances = []
        for i in range(len(pareto_front)):
            for j in range(i + 1, len(pareto_front)):
                dist = np.linalg.norm(
                    np.array(pareto_front[i].objectives) - 
                    np.array(pareto_front[j].objectives)
                )
                distances.append(dist)
        
        return float(np.mean(distances)) if distances else 0.0
    
    def save_results(self, filepath: str, results: Dict[str, Any]):
        """Save optimization results to file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """Load optimization results from file."""
        try:
            with open(filepath, 'r') as f:
                results = json.load(f)
            logger.info(f"Results loaded from {filepath}")
            return results
        except Exception as e:
            logger.error(f"Failed to load results: {e}")
            return {}


# Convenience functions
def create_multi_objective_optimizer(
    n_objectives: int = 2,
    n_trials: int = 100,
    **kwargs
) -> MultiObjectiveOptimizer:
    """Create a multi-objective optimizer with default settings."""
    config = MultiObjectiveConfig(
        n_objectives=n_objectives,
        n_trials=n_trials,
        **kwargs
    )
    return MultiObjectiveOptimizer(config)


def create_accuracy_efficiency_objectives() -> List[ObjectiveFunction]:
    """Create common accuracy and efficiency objectives."""
    def accuracy_objective(params, model, X, y, **kwargs):
        """Accuracy objective."""
        try:
            if hasattr(model, 'fit'):
                model.fit(X, y)
                if hasattr(model, 'score'):
                    return model.score(X, y)
                elif hasattr(model, 'predict'):
                    from sklearn.metrics import accuracy_score
                    y_pred = model.predict(X)
                    return accuracy_score(y, y_pred)
            return 0.0
        except Exception:
            return 0.0
    
    def efficiency_objective(params, model, X, y, **kwargs):
        """Efficiency objective (inverse of training time)."""
        try:
            start_time = time.time()
            if hasattr(model, 'fit'):
                model.fit(X, y)
            training_time = time.time() - start_time
            return 1.0 / (training_time + 1e-6)  # Avoid division by zero
        except Exception:
            return 0.0
    
    return [
        ObjectiveFunction('accuracy', accuracy_objective, direction='maximize'),
        ObjectiveFunction('efficiency', efficiency_objective, direction='maximize')
    ]


def create_performance_robustness_objectives() -> List[ObjectiveFunction]:
    """Create performance and robustness objectives."""
    def performance_objective(params, model, X, y, **kwargs):
        """Performance objective (cross-validation score)."""
        try:
            if SKLEARN_AVAILABLE and hasattr(model, 'fit'):
                scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error')
                return float(np.mean(scores))
            return 0.0
        except Exception:
            return 0.0
    
    def robustness_objective(params, model, X, y, **kwargs):
        """Robustness objective (stability across CV folds)."""
        try:
            if SKLEARN_AVAILABLE and hasattr(model, 'fit'):
                scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
                return -float(np.std(scores))  # Lower std is better
            return 0.0
        except Exception:
            return 0.0
    
    return [
        ObjectiveFunction('performance', performance_objective, direction='maximize'),
        ObjectiveFunction('robustness', robustness_objective, direction='maximize')
    ]