#!/usr/bin/env python3
"""
Parameter Optimization for HMM Regime Discovery

This module implements dynamic parameter search for HMM clustering:
- HMM State Count Optimization
- Covariance Type Optimization  
- Comprehensive Parameter Grid Search

Author: AI Assistant
Date: 2024-01-XX
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import time
import logging
from pathlib import Path
import json

# HMM imports
try:
    from hmmlearn import hmm
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False

# Sklearn imports
try:
    from sklearn.model_selection import cross_val_score, ParameterGrid
    # Note: Removed silhouette_score, calinski_harabasz_score, davies_bouldin_score 
    # as these traditional clustering metrics are not relevant for HMMs
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Optuna imports for advanced optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from src.utils.logger import system_logger

@dataclass
class OptimizationResult:
    """Result of parameter optimization"""
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Dict[str, Any]]
    optimization_time: float
    method: str

class ParameterOptimizer:
    """Dynamic parameter optimization for HMM clustering"""
    
    def __init__(self, logger=None):
        self.logger = logger or system_logger.getChild('ParameterOptimizer')
        self.optimization_history = []
        
        # Initialize hardware optimizations
        self._initialize_hardware_optimizations()
    
    def _initialize_hardware_optimizations(self):
        """Initialize hardware optimization components."""
        self.hardware_optimizations = {
            'cpu_optimizer': None,
            'memory_optimizer': None,
            'available': False
        }
        
        try:
            from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
            from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
            
            self.hardware_optimizations['cpu_optimizer'] = get_m1_cpu_optimizer()
            self.hardware_optimizations['memory_optimizer'] = get_m1_memory_optimizer()
            self.hardware_optimizations['available'] = True
            self.logger.info("✅ Hardware optimizations initialized for parameter optimization")
        except ImportError:
            self.logger.info("ℹ️ Hardware optimizations not available for parameter optimization")
    
    def optimize_hmm_states(self, features: np.ndarray, 
                          state_range: Tuple[int, int] = (2, 8), 
                          cv_folds: int = 5,
                          covariance_type: str = 'full') -> OptimizationResult:
        """
        Dynamically find optimal number of HMM states using cross-validation
        
        Args:
            features: Input features for HMM training
            state_range: Range of states to test (min, max)
            cv_folds: Number of cross-validation folds
            covariance_type: Covariance type to use
            
        Returns:
            OptimizationResult with best parameters and scores
        """
        if not HMMLEARN_AVAILABLE:
            raise ImportError("hmmlearn not available")
        
        start_time = time.time()
        self.logger.info(f"🔍 Optimizing HMM states in range {state_range}")
        
        best_score = -np.inf
        best_n_states = state_range[0]
        all_results = []
        
        # Use parallel processing if hardware optimizations are available
        if self.hardware_optimizations['available']:
            all_results = self._optimize_hmm_states_parallel(features, state_range, cv_folds, covariance_type)
        else:
            all_results = self._optimize_hmm_states_sequential(features, state_range, cv_folds, covariance_type)
        
        # Find best result
        for result in all_results:
            if result['mean_score'] > best_score:
                best_score = result['mean_score']
                best_n_states = result['n_components']
        
        optimization_time = time.time() - start_time
        
        best_params = {
            'n_components': best_n_states,
            'covariance_type': covariance_type
        }
        
        result = OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            optimization_time=optimization_time,
            method='hmm_states_optimization'
        )
        
        self.optimization_history.append(result)
        self.logger.info(f"✅ Best number of states: {best_n_states} (score: {best_score:.4f})")
        
        return result
    
    def _optimize_hmm_states_parallel(self, features: np.ndarray, state_range: Tuple[int, int], 
                                    cv_folds: int, covariance_type: str) -> List[Dict[str, Any]]:
        """Optimize HMM states using parallel processing."""
        self.logger.info("🚀 Using parallel processing for HMM state optimization...")
        
        cpu_optimizer = self.hardware_optimizations['cpu_optimizer']
        memory_optimizer = self.hardware_optimizations['memory_optimizer']
        
        # Create optimized thread pool
        with cpu_optimizer.create_optimized_thread_pool() as executor:
            # Submit optimization tasks
            futures = []
            for n_states in range(state_range[0], state_range[1] + 1):
                future = executor.submit(self._evaluate_hmm_states, features, n_states, cv_folds, covariance_type)
                futures.append(future)
            
            # Collect results
            all_results = []
            for future in futures:
                try:
                    result = future.result(timeout=300)  # 5 minute timeout
                    all_results.append(result)
                except Exception as e:
                    self.logger.warning(f"Optimization task failed: {e}")
        
        return all_results
    
    def _optimize_hmm_states_sequential(self, features: np.ndarray, state_range: Tuple[int, int], 
                                      cv_folds: int, covariance_type: str) -> List[Dict[str, Any]]:
        """Optimize HMM states using sequential processing."""
        self.logger.info("🔄 Using sequential processing for HMM state optimization...")
        
        all_results = []
        for n_states in range(state_range[0], state_range[1] + 1):
            result = self._evaluate_hmm_states(features, n_states, cv_folds, covariance_type)
            all_results.append(result)
        
        return all_results
    
    def _evaluate_hmm_states(self, features: np.ndarray, n_states: int, cv_folds: int,
                           covariance_type: str) -> Dict[str, Any]:
        """Evaluate HMM with specific number of states."""
        try:
            # Create HMM model
            model = hmm.GaussianHMM(
                n_components=n_states,
                covariance_type=covariance_type,
                random_state=42,
                n_iter=100
            )

            # Cross-validation scoring
            scores = cross_val_score(
                model, features,
                cv=cv_folds,
                scoring='neg_log_likelihood'
            )

            mean_score = scores.mean()
            std_score = scores.std()

            result = {
                'n_components': n_states,
                'covariance_type': covariance_type,
                'mean_score': mean_score,
                'std_score': std_score,
                'scores': scores.tolist()
            }

            self.logger.info(f"   States {n_states}: {mean_score:.4f} ± {std_score:.4f}")

            return result

        except Exception as e:
            self.logger.warning(f"Error with {n_states} states: {e}")
            return {
                'n_components': n_states,
                'covariance_type': covariance_type,
                'mean_score': float('-inf'),
                'std_score': 0.0,
                'scores': [],
                'error': str(e)
            }
    
    def optimize_covariance_type(self, features: np.ndarray, 
                               n_states: int,
                               covariance_types: List[str] = ['full', 'tied', 'diag', 'spherical']) -> OptimizationResult:
        """
        Find optimal covariance type for given number of states
        
        Args:
            features: Input features for HMM training
            n_states: Number of HMM states
            covariance_types: List of covariance types to test
            
        Returns:
            OptimizationResult with best parameters and scores
        """
        if not HMMLEARN_AVAILABLE:
            raise ImportError("hmmlearn not available")
        
        start_time = time.time()
        self.logger.info(f"🔍 Optimizing covariance type for {n_states} states")
        
        best_score = -np.inf
        best_cov_type = covariance_types[0]
        all_results = []
        
        for cov_type in covariance_types:
            try:
                model = hmm.GaussianHMM(
                    n_components=n_states,
                    covariance_type=cov_type,
                    random_state=42,
                    n_iter=100
                )
                
                # Fit and score
                model.fit(features)
                score = model.score(features)
                
                result = {
                    'n_components': n_states,
                    'covariance_type': cov_type,
                    'score': score
                }
                all_results.append(result)
                
                self.logger.info(f"   {cov_type}: {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_cov_type = cov_type
                    
            except Exception as e:
                self.logger.warning(f"Error with {cov_type}: {e}")
                continue
        
        optimization_time = time.time() - start_time
        
        best_params = {
            'n_components': n_states,
            'covariance_type': best_cov_type
        }
        
        result = OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            optimization_time=optimization_time,
            method='covariance_type_optimization'
        )
        
        self.optimization_history.append(result)
        self.logger.info(f"✅ Best covariance type: {best_cov_type} (score: {best_score:.4f})")
        
        return result
    
    def comprehensive_parameter_optimization(self, features: np.ndarray, 
                                           param_grid: Optional[Dict[str, List]] = None,
                                           use_optuna: bool = False,
                                           n_trials: int = 100) -> OptimizationResult:
        """
        Comprehensive parameter optimization using grid search or Optuna
        
        Args:
            features: Input features for HMM training
            param_grid: Parameter grid for grid search
            use_optuna: Whether to use Optuna for optimization
            n_trials: Number of trials for Optuna optimization
            
        Returns:
            OptimizationResult with best parameters and scores
        """
        if not HMMLEARN_AVAILABLE:
            raise ImportError("hmmlearn not available")
        
        if use_optuna and OPTUNA_AVAILABLE:
            return self._optuna_optimization(features, n_trials)
        else:
            return self._grid_search_optimization(features, param_grid)
    
    def _grid_search_optimization(self, features: np.ndarray, 
                                 param_grid: Optional[Dict[str, List]] = None) -> OptimizationResult:
        """Grid search optimization"""
        if param_grid is None:
            param_grid = {
                'n_components': [2, 3, 4, 5, 6, 7, 8],
                'covariance_type': ['full', 'tied', 'diag', 'spherical'],
                'n_iter': [50, 100, 200],
                'tol': [1e-6, 1e-4, 1e-2]
            }
        
        start_time = time.time()
        self.logger.info("🔍 Starting comprehensive grid search optimization")
        
        best_score = -np.inf
        best_params = {}
        all_results = []
        
        # Generate all parameter combinations
        param_combinations = list(ParameterGrid(param_grid))
        total_combinations = len(param_combinations)
        
        self.logger.info(f"   Testing {total_combinations} parameter combinations")
        
        for i, params in enumerate(param_combinations):
            try:
                # Create model with current parameters
                model = hmm.GaussianHMM(
                    n_components=params['n_components'],
                    covariance_type=params['covariance_type'],
                    n_iter=params['n_iter'],
                    tol=params['tol'],
                    random_state=42
                )
                
                # Cross-validation
                scores = cross_val_score(model, features, cv=3, scoring='neg_log_likelihood')
                mean_score = scores.mean()
                
                result = {
                    'params': params.copy(),
                    'mean_score': mean_score,
                    'std_score': scores.std(),
                    'scores': scores.tolist()
                }
                all_results.append(result)
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"   Progress: {i+1}/{total_combinations} combinations tested")
                
                # Update best
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params.copy()
                    
            except Exception as e:
                self.logger.warning(f"Error with params {params}: {e}")
                continue
        
        optimization_time = time.time() - start_time
        
        result = OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            optimization_time=optimization_time,
            method='grid_search_optimization'
        )
        
        self.optimization_history.append(result)
        self.logger.info(f"✅ Best parameters: {best_params} (score: {best_score:.4f})")
        
        return result
    
    def _optuna_optimization(self, features: np.ndarray, n_trials: int, config=None) -> OptimizationResult:
        """Optuna-based optimization"""
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not available")

        start_time = time.time()
        self.logger.info(f"🔍 Starting Optuna optimization with {n_trials} trials")

        # Auto-detect HMM mode from configuration
        param_ranges = self._get_hmm_parameter_ranges()
        hmm_mode = self._auto_detect_hmm_mode(config)
        self.logger.info(f"🔧 Using {hmm_mode} mode: {param_ranges[hmm_mode]['description']}")

        def objective(trial):
            # Suggest parameters based on mode
            n_components = trial.suggest_int('n_components',
                param_ranges[hmm_mode]['n_components_min'],
                param_ranges[hmm_mode]['n_components_max'])
            covariance_type = trial.suggest_categorical('covariance_type',
                param_ranges[hmm_mode]['covariance_types'])
            n_iter = trial.suggest_int('n_iter',
                param_ranges[hmm_mode]['n_iter_min'],
                param_ranges[hmm_mode]['n_iter_max'])
            tol = trial.suggest_float('tol',
                param_ranges[hmm_mode]['tol_min'],
                param_ranges[hmm_mode]['tol_max'], log=True)
            
            try:
                # Create and fit model
                model = hmm.GaussianHMM(
                    n_components=n_components,
                    covariance_type=covariance_type,
                    n_iter=n_iter,
                    tol=tol,
                    random_state=42
                )
                
                # Cross-validation
                scores = cross_val_score(model, features, cv=3, scoring='neg_log_likelihood')
                return scores.mean()
                
            except Exception:
                return -np.inf
        
        # Create study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        optimization_time = time.time() - start_time
        
        best_params = study.best_params
        best_score = study.best_value
        
        # Convert trials to results format
        all_results = []
        for trial in study.trials:
            if trial.value is not None:
                all_results.append({
                    'params': trial.params,
                    'score': trial.value,
                    'state': trial.state.name
                })
        
        result = OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            optimization_time=optimization_time,
            method='optuna_optimization'
        )
        
        self.optimization_history.append(result)
        self.logger.info(f"✅ Best parameters: {best_params} (score: {best_score:.4f})")
        
        return result
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of all optimization runs"""
        if not self.optimization_history:
            return {"message": "No optimization runs recorded"}
        
        summary = {
            "total_runs": len(self.optimization_history),
            "methods_used": list(set(r.method for r in self.optimization_history)),
            "best_overall_score": max(r.best_score for r in self.optimization_history),
            "total_optimization_time": sum(r.optimization_time for r in self.optimization_history),
            "runs": []
        }
        
        for i, result in enumerate(self.optimization_history):
            summary["runs"].append({
                "run_id": i,
                "method": result.method,
                "best_params": result.best_params,
                "best_score": result.best_score,
                "optimization_time": result.optimization_time
            })
        
        return summary
    
    def save_optimization_results(self, filepath: str) -> None:
        """Save optimization results to file"""
        results = {
            "optimization_history": [
                {
                    "best_params": r.best_params,
                    "best_score": r.best_score,
                    "optimization_time": r.optimization_time,
                    "method": r.method,
                    "all_results": r.all_results
                }
                for r in self.optimization_history
            ],
            "summary": self.get_optimization_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"💾 Optimization results saved to {filepath}")

    def _auto_detect_hmm_mode(self, config) -> str:
        """Auto-detect HMM optimization mode based on launcher configuration.

        Maps launcher modes to HMM optimization modes:
        - Launcher FULL → HMM FULL (comprehensive optimization)
        - Launcher LIGHT → HMM BLANK (moderate speedup)
        - Launcher BLANK → HMM LIGHT (maximum speedup)
        """
        if not config:
            return 'BLANK'

        # Check if config has a mode attribute
        if hasattr(config, 'mode'):
            launcher_mode = str(config.mode).upper()

            # Map launcher modes to HMM modes
            mode_mapping = {
                'FULL': 'FULL',      # Launcher FULL → HMM FULL
                'LIGHT': 'BLANK',    # Launcher LIGHT → HMM BLANK
                'BLANK': 'LIGHT'     # Launcher BLANK → HMM LIGHT
            }

            hmm_mode = mode_mapping.get(launcher_mode, 'BLANK')
            self.logger.info(f"🔄 Auto-detected launcher mode '{launcher_mode}' → HMM mode '{hmm_mode}'")
            return hmm_mode

        # Check if config has a bayesian_optimization attribute (for future compatibility)
        elif hasattr(config, 'bayesian_optimization'):
            # For now, use BLANK mode as default when bayesian_optimization is present
            self.logger.info("🔄 Detected bayesian_optimization config → using HMM BLANK mode")
            return 'BLANK'

        # Default fallback
        self.logger.info("🔄 No mode detected, using default HMM BLANK mode")
        return 'BLANK'

    def _get_hmm_parameter_ranges(self) -> Dict[str, Dict[str, Any]]:
        """Get HMM parameter ranges based on optimization mode.

        Returns:
            Dictionary with parameter ranges for each mode:
            - FULL: Regular parameters (comprehensive optimization)
            - BLANK: Lighter parameters (moderate speedup)
            - LIGHT: Ultra-light parameters (maximum speedup)
        """
        return {
            'FULL': {
                'n_components_min': 3,
                'n_components_max': 20,  # Increased to allow more regimes to prevent overlap
                'covariance_types': ['diag', 'spherical', 'tied', 'full'],  # Add more covariance types for better cluster separation
                'n_iter_min': 50,
                'n_iter_max': 150,
                'tol_min': 1e-4,  # More reasonable convergence tolerance
                'tol_max': 1e-3,
                'description': 'Expanded parameters for better regime separation (3-20 regimes, multiple covariance types)'
            },
            'BLANK': {
                'n_components_min': 3,
                'n_components_max': 10,
                'covariance_types': ['diag', 'spherical', 'tied'],  # Include tied for better structure
                'n_iter_min': 5,  # Reduced for BLANK mode as requested
                'n_iter_max': 5,  # Fixed at 5 iterations for BLANK mode
                'tol_min': 1e-4,
                'tol_max': 1e-3,
                'description': 'Balanced parameters with tied covariance (3-10 regimes, 5 iterations)'
            },
            'LIGHT': {
                'n_components_min': 3,
                'n_components_max': 6,
                'covariance_types': ['diag', 'spherical'],  # Keep stable types for LIGHT
                'n_iter_min': 2,  # Reduced to 2 iterations for LIGHT mode as requested
                'n_iter_max': 2,  # Fixed at 2 iterations for LIGHT mode
                'tol_min': 1e-4,
                'tol_max': 5e-4,
                'description': 'Minimal parameters for fastest execution (3-6 regimes, 2 iterations)'
            }
        }

# Example usage and testing
def test_parameter_optimization():
    """Test the parameter optimization functionality"""
    # Generate sample data
    np.random.seed(42)
    n_samples, n_features = 1000, 10
    features = np.random.randn(n_samples, n_features)
    
    # Add some structure to make it more realistic
    features[:n_samples//3] += 2  # First third has different mean
    features[n_samples//3:2*n_samples//3] -= 1  # Second third has different mean
    
    optimizer = ParameterOptimizer()
    
    # Test HMM state optimization
    print("Testing HMM state optimization...")
    result1 = optimizer.optimize_hmm_states(features, state_range=(2, 6))
    print(f"Best states: {result1.best_params['n_components']}")
    
    # Test covariance type optimization
    print("\nTesting covariance type optimization...")
    result2 = optimizer.optimize_covariance_type(features, n_states=4)
    print(f"Best covariance type: {result2.best_params['covariance_type']}")
    
    # Test comprehensive optimization
    print("\nTesting comprehensive optimization...")
    result3 = optimizer.comprehensive_parameter_optimization(features, use_optuna=False)
    print(f"Best parameters: {result3.best_params}")
    
    # Print summary
    print("\nOptimization Summary:")
    summary = optimizer.get_optimization_summary()
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    test_parameter_optimization()