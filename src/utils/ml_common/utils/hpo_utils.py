"""
Hyperparameter Optimization Utilities

This module provides utilities for hyperparameter optimization with memory-aware operations.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Callable
from functools import partial
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class HyperparameterOptimization:
    """Hyperparameter optimization utilities with memory management."""

    def __init__(self):
        """Initialize HPO utilities."""
        self.logger = logger.getChild('HyperparameterOptimization')
        self.logger.info("🚀 Initializing HyperparameterOptimization utilities")

    def multi_objective_optimization(
        self,
        param_space: Dict[str, Any],
        objective_function: Callable,
        n_trials: int = 100,
        n_objectives: int = 2,
        direction: str = 'minimize',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform multi-objective hyperparameter optimization.

        Args:
            param_space: Dictionary defining the parameter space
            objective_function: Function to optimize
            n_trials: Number of optimization trials
            n_objectives: Number of objectives
            direction: Optimization direction ('minimize' or 'maximize')

        Returns:
            Dictionary containing optimization results
        """
        self.logger.info(f"🔍 Starting multi-objective optimization with {n_trials} trials")

        start_time = time.time()

        # Simulate optimization trials
        trials = []
        for i in range(n_trials):
            # Generate random parameters from param_space
            trial_params = self._generate_random_params(param_space)
            trial_params['trial_id'] = i

            # Evaluate objective function
            try:
                objectives = objective_function(**trial_params)
                if not isinstance(objectives, (list, tuple)):
                    objectives = [objectives]

                trial_result = {
                    'params': trial_params,
                    'objectives': objectives,
                    'trial_number': i,
                    'success': True
                }
                trials.append(trial_result)

            except Exception as e:
                self.logger.warning(f"⚠️ Trial {i} failed: {e}")
                trial_result = {
                    'params': trial_params,
                    'objectives': [float('inf')] * n_objectives,
                    'trial_number': i,
                    'success': False,
                    'error': str(e)
                }
                trials.append(trial_result)

        # Find Pareto front (simplified)
        pareto_front = self._find_pareto_front(trials, n_objectives, direction)

        optimization_time = time.time() - start_time

        result = {
            'trials': trials,
            'pareto_front': pareto_front,
            'best_params': pareto_front[0]['params'] if pareto_front else None,
            'n_trials': n_trials,
            'n_objectives': n_objectives,
            'direction': direction,
            'optimization_time': optimization_time,
            'success': len(pareto_front) > 0
        }

        self.logger.info(f"✅ Multi-objective optimization completed in {optimization_time:.2f}s")
        return result

    def early_stopping_optimization(
        self,
        param_space: Dict[str, Any],
        objective_function: Callable,
        early_stopping_rounds: int = 10,
        max_trials: int = 1000,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter optimization with early stopping.

        Args:
            param_space: Dictionary defining the parameter space
            objective_function: Function to optimize
            early_stopping_rounds: Number of rounds for early stopping
            max_trials: Maximum number of trials

        Returns:
            Dictionary containing optimization results
        """
        self.logger.info(f"🔍 Starting early stopping optimization (max {max_trials} trials)")

        start_time = time.time()

        trials = []
        best_score = float('inf') if kwargs.get('direction', 'minimize') == 'minimize' else float('-inf')
        no_improvement_count = 0

        for i in range(max_trials):
            # Generate random parameters
            trial_params = self._generate_random_params(param_space)
            trial_params['trial_id'] = i

            # Evaluate objective function
            try:
                score = objective_function(**trial_params)
                trial_result = {
                    'params': trial_params,
                    'score': score,
                    'trial_number': i,
                    'success': True
                }
                trials.append(trial_result)

                # Check for improvement
                improved = False
                if kwargs.get('direction', 'minimize') == 'minimize':
                    if score < best_score:
                        best_score = score
                        improved = True
                else:
                    if score > best_score:
                        best_score = score
                        improved = True

                if improved:
                    no_improvement_count = 0
                    self.logger.debug(f"📈 New best score: {best_score:.4f} at trial {i}")
                else:
                    no_improvement_count += 1

                # Early stopping check
                if no_improvement_count >= early_stopping_rounds:
                    self.logger.info(f"🛑 Early stopping at trial {i} (no improvement for {early_stopping_rounds} rounds)")
                    break

            except Exception as e:
                self.logger.warning(f"⚠️ Trial {i} failed: {e}")
                trial_result = {
                    'params': trial_params,
                    'score': float('inf'),
                    'trial_number': i,
                    'success': False,
                    'error': str(e)
                }
                trials.append(trial_result)

        optimization_time = time.time() - start_time

        # Find best result
        successful_trials = [t for t in trials if t['success']]
        if successful_trials:
            if kwargs.get('direction', 'minimize') == 'minimize':
                best_trial = min(successful_trials, key=lambda x: x['score'])
            else:
                best_trial = max(successful_trials, key=lambda x: x['score'])
        else:
            best_trial = None

        result = {
            'trials': trials,
            'best_trial': best_trial,
            'best_params': best_trial['params'] if best_trial else None,
            'best_score': best_trial['score'] if best_trial else None,
            'n_trials': len(trials),
            'early_stopping_rounds': early_stopping_rounds,
            'optimization_time': optimization_time,
            'success': best_trial is not None
        }

        self.logger.info(f"✅ Early stopping optimization completed in {optimization_time:.2f}s")
        return result

    def _generate_random_params(self, param_space: Dict[str, Any]) -> Dict[str, Any]:
        """Generate random parameters from parameter space."""
        params = {}

        for param_name, param_config in param_space.items():
            if isinstance(param_config, dict):
                param_type = param_config.get('type', 'float')

                if param_type == 'int':
                    low = param_config.get('low', 0)
                    high = param_config.get('high', 100)
                    params[param_name] = np.random.randint(low, high)
                elif param_type == 'float':
                    low = param_config.get('low', 0.0)
                    high = param_config.get('high', 1.0)
                    params[param_name] = np.random.uniform(low, high)
                elif param_type == 'categorical':
                    choices = param_config.get('choices', [])
                    if choices:
                        params[param_name] = np.random.choice(choices)
                else:
                    # Default to float
                    params[param_name] = np.random.uniform(0, 1)
            elif isinstance(param_config, list):
                # Categorical from list
                params[param_name] = np.random.choice(param_config)
            else:
                # Single value
                params[param_name] = param_config

        return params

    def _find_pareto_front(self, trials: List[Dict], n_objectives: int, direction: str) -> List[Dict]:
        """Find Pareto front from multi-objective optimization trials."""
        if not trials:
            return []

        # Filter successful trials
        successful_trials = [t for t in trials if t['success']]

        if not successful_trials:
            return []

        # Simple Pareto front calculation (can be optimized)
        pareto_front = []

        for trial in successful_trials:
            objectives = trial['objectives']
            dominated = False

            for other_trial in successful_trials:
                if trial == other_trial:
                    continue

                other_objectives = other_trial['objectives']

                # Check if this trial is dominated
                dominates = True
                dominated_by_other = True

                for i in range(n_objectives):
                    if direction == 'minimize':
                        if objectives[i] > other_objectives[i]:
                            dominates = False
                        if objectives[i] < other_objectives[i]:
                            dominated_by_other = False
                    else:
                        if objectives[i] < other_objectives[i]:
                            dominates = False
                        if objectives[i] > other_objectives[i]:
                            dominated_by_other = False

                if dominated_by_other and not dominates:
                    dominated = True
                    break

            if not dominated:
                pareto_front.append(trial)

        # Sort by first objective
        pareto_front.sort(key=lambda x: x['objectives'][0])

        return pareto_front


# Global instance for easy access
_hpo_instance = None

def get_hyperparameter_optimizer() -> HyperparameterOptimization:
    """Get global hyperparameter optimizer instance."""
    global _hpo_instance
    if _hpo_instance is None:
        _hpo_instance = HyperparameterOptimization()
    return _hpo_instance

# Export key classes and functions
__all__ = ['HyperparameterOptimization', 'get_hyperparameter_optimizer']
