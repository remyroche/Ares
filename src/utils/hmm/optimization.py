#!/usr/bin/env python3
"""
HMM Optimization Module

This module contains Bayesian optimization and parameter tuning functionality
for HMM models, extracted from the monolithic hmm_composite_manager.py file.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..logger import system_logger

# Optional imports for optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class BayesianOptimizationConfig:
    """Configuration for Bayesian optimization."""
    n_trials: int = 50
    timeout_minutes: int = 15
    n_jobs: int = 1
    sampler: str = "TPE"  # TPE, Random, CmaEs
    pruner: str = "Median"  # Median, Hyperband, None
    direction: str = "maximize"  # maximize, minimize
    
    # Parameter ranges
    n_components_range: Tuple[int, int] = (2, 8)
    n_iter_range: Tuple[int, int] = (50, 200)
    tol_range: Tuple[float, float] = (1e-6, 1e-2)
    min_covar_range: Tuple[float, float] = (1e-6, 1e-1)


class HMMBayesianOptimizer:
    """Bayesian optimizer for HMM parameters."""
    
    def __init__(self, config: Optional[BayesianOptimizationConfig] = None):
        """Initialize the Bayesian optimizer."""
        self.config = config or BayesianOptimizationConfig()
        self.logger = system_logger.getChild('HMMBayesianOptimizer')
        self.study = None
        self.best_params = None
        
        if not OPTUNA_AVAILABLE:
            self.logger.warning("Optuna not available - optimization will be limited")
    
    def optimize_hmm_parameters(
        self,
        data: pd.DataFrame,
        objective_function: Callable,
        fixed_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Optimize HMM parameters using Bayesian optimization.
        
        Args:
            data: Training data
            objective_function: Function to optimize (should return score to maximize)
            fixed_params: Parameters to keep fixed during optimization
            
        Returns:
            Best parameters found
        """
        if not OPTUNA_AVAILABLE:
            return self._get_default_hmm_parameters()
        
        try:
            self.logger.info("Starting Bayesian optimization for HMM parameters")
            
            # Create study
            study_name = f"hmm_optimization_{int(pd.Timestamp.now().timestamp())}"
            
            sampler = self._create_sampler()
            pruner = self._create_pruner()
            
            self.study = optuna.create_study(
                study_name=study_name,
                direction=self.config.direction,
                sampler=sampler,
                pruner=pruner
            )
            
            # Define objective wrapper
            def objective(trial):
                return self._objective_wrapper(trial, data, objective_function, fixed_params)
            
            # Optimize
            self.study.optimize(
                objective,
                n_trials=self.config.n_trials,
                timeout=self.config.timeout_minutes * 60,
                n_jobs=self.config.n_jobs
            )
            
            self.best_params = self.study.best_params
            
            self.logger.info(f"Optimization completed. Best score: {self.study.best_value}")
            self.logger.info(f"Best parameters: {self.best_params}")
            
            return self.best_params
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            return self._get_default_hmm_parameters()
    
    def _objective_wrapper(
        self,
        trial,
        data: pd.DataFrame,
        objective_function: Callable,
        fixed_params: Optional[Dict[str, Any]]
    ) -> float:
        """Wrapper for the objective function."""
        try:
            # Suggest parameters
            params = self._suggest_parameters(trial)
            
            # Add fixed parameters
            if fixed_params:
                params.update(fixed_params)
            
            # Evaluate objective
            score = objective_function(data, params)
            
            # Handle invalid scores
            if np.isnan(score) or np.isinf(score):
                return -1e6 if self.config.direction == "maximize" else 1e6
            
            return float(score)
            
        except Exception as e:
            self.logger.warning(f"Trial failed: {e}")
            return -1e6 if self.config.direction == "maximize" else 1e6
    
    def _suggest_parameters(self, trial) -> Dict[str, Any]:
        """Suggest parameters for a trial."""
        params = {}
        
        # Core HMM parameters
        params['n_components'] = trial.suggest_int(
            'n_components',
            self.config.n_components_range[0],
            self.config.n_components_range[1]
        )
        
        params['n_iter'] = trial.suggest_int(
            'n_iter',
            self.config.n_iter_range[0],
            self.config.n_iter_range[1]
        )
        
        params['tol'] = trial.suggest_float(
            'tol',
            self.config.tol_range[0],
            self.config.tol_range[1],
            log=True
        )
        
        params['min_covar'] = trial.suggest_float(
            'min_covar',
            self.config.min_covar_range[0],
            self.config.min_covar_range[1],
            log=True
        )
        
        params['covariance_type'] = trial.suggest_categorical(
            'covariance_type',
            ['spherical', 'diag', 'full', 'tied']
        )
        
        params['algorithm'] = trial.suggest_categorical(
            'algorithm',
            ['viterbi', 'map']
        )
        
        return params
    
    def _create_sampler(self):
        """Create optuna sampler based on configuration."""
        if self.config.sampler == "TPE":
            return optuna.samplers.TPESampler()
        elif self.config.sampler == "Random":
            return optuna.samplers.RandomSampler()
        elif self.config.sampler == "CmaEs":
            return optuna.samplers.CmaEsSampler()
        else:
            return optuna.samplers.TPESampler()
    
    def _create_pruner(self):
        """Create optuna pruner based on configuration."""
        if self.config.pruner == "Median":
            return optuna.pruners.MedianPruner()
        elif self.config.pruner == "Hyperband":
            return optuna.pruners.HyperbandPruner()
        elif self.config.pruner == "None":
            return optuna.pruners.NopPruner()
        else:
            return optuna.pruners.MedianPruner()
    
    def _get_default_hmm_parameters(self) -> Dict[str, Any]:
        """Get default HMM parameters when optimization is not available."""
        return {
            'n_components': 3,
            'covariance_type': 'full',
            'n_iter': 100,
            'tol': 1e-4,
            'min_covar': 1e-3,
            'algorithm': 'viterbi',
            'random_state': 42
        }
    
    def get_optimization_history(self) -> Optional[pd.DataFrame]:
        """Get optimization history as DataFrame."""
        if self.study is None:
            return None
        
        try:
            trials_df = self.study.trials_dataframe()
            return trials_df
        except Exception as e:
            self.logger.error(f"Error getting optimization history: {e}")
            return None
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """Plot optimization history."""
        if self.study is None:
            self.logger.warning("No study available for plotting")
            return
        
        try:
            import optuna.visualization as vis
            import plotly.graph_objects as go
            
            # Create plots
            fig_history = vis.plot_optimization_history(self.study)
            fig_importance = vis.plot_param_importances(self.study)
            
            if save_path:
                fig_history.write_html(f"{save_path}_history.html")
                fig_importance.write_html(f"{save_path}_importance.html")
                self.logger.info(f"Optimization plots saved to {save_path}")
            else:
                fig_history.show()
                fig_importance.show()
                
        except ImportError:
            self.logger.warning("Plotly not available for visualization")
        except Exception as e:
            self.logger.error(f"Error plotting optimization history: {e}")


class HMMParameterTuner:
    """Simple parameter tuner for HMM models when Bayesian optimization is not available."""
    
    def __init__(self):
        """Initialize the parameter tuner."""
        self.logger = system_logger.getChild('HMMParameterTuner')
    
    def grid_search_parameters(
        self,
        data: pd.DataFrame,
        objective_function: Callable,
        param_grid: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """
        Perform grid search over HMM parameters.
        
        Args:
            data: Training data
            objective_function: Function to optimize
            param_grid: Dictionary of parameter values to try
            
        Returns:
            Best parameters found
        """
        best_score = -np.inf
        best_params = {}
        
        # Generate all parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)
        
        self.logger.info(f"Testing {len(param_combinations)} parameter combinations")
        
        for i, params in enumerate(param_combinations):
            try:
                score = objective_function(data, params)
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Tested {i + 1}/{len(param_combinations)} combinations")
                    
            except Exception as e:
                self.logger.warning(f"Parameter combination {i} failed: {e}")
                continue
        
        self.logger.info(f"Grid search completed. Best score: {best_score}")
        self.logger.info(f"Best parameters: {best_params}")
        
        return best_params
    
    def _generate_param_combinations(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all combinations of parameters."""
        import itertools
        
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        combinations = []
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))
            combinations.append(param_dict)
        
        return combinations