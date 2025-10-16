"""
Bayesian Optimization for MSM Parameters.

This module provides Bayesian optimization for Markov State Model parameters
including transition matrix, lag times, clustering parameters, and attention networks.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import logging
import time
from dataclasses import dataclass
from enum import Enum

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    gp_minimize = None

try:
    from optuna import create_study, Trial
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from src.training.steps.market_analysis.hmm_clustering.core.msm_clustering import MSMClusterer, MSMConfig

logger = logging.getLogger(__name__)


class MSMOptimizationObjective(Enum):
    """Objectives for MSM optimization."""
    MSM_SCORE = "msm_score"
    SILHOUETTE = "silhouette"
    CONNECTIVITY = "connectivity"
    STATIONARITY = "stationarity"
    IMPLIED_TIMESCALE = "implied_timescale"


@dataclass
class MSMOptimizationConfig:
    """Configuration for MSM Bayesian optimization - optimized for efficiency."""
    n_trials: int = 15  # Reduced for computational efficiency
    timeout: int = 300  # 5 minutes max - much more efficient
    random_state: int = 42
    n_jobs: int = 2  # Use 2 cores instead of all available
    optimization_objective: str = "msm_score"
    use_skopt: bool = True  # Use scikit-optimize instead of Optuna
    early_stopping_patience: int = 5  # Reduced patience for faster convergence
    early_stopping_min_delta: float = 0.01  # Larger delta for faster stopping

    # Two-step optimization: grid search first, then Bayesian
    use_two_step_optimization: bool = True
    grid_search_n_points: int = 8  # Grid search points before Bayesian

    # MSM parameter bounds - optimized ranges
    n_states_min: int = 8
    n_states_max: int = 25  # Reduced range for efficiency
    lag_time_min: int = 1
    lag_time_max: int = 10  # Reduced range for efficiency
    connectivity_threshold_min: float = 0.05
    connectivity_threshold_max: float = 0.3  # Narrower range for efficiency
    ergodic_cutoff_min: float = 1e-6
    ergodic_cutoff_max: float = 1e-4  # Narrower range for efficiency


class MSMBayesianOptimizer:
    """Bayesian optimization for MSM parameters - computationally efficient implementation."""

    def __init__(self, config: MSMOptimizationConfig):
        """Initialize MSM Bayesian optimizer.

        Args:
            config: MSM optimization configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        if not SKOPT_AVAILABLE and not OPTUNA_AVAILABLE:
            raise ImportError("Either scikit-optimize or Optuna must be installed for Bayesian optimization")

        # Track optimization history - limit memory usage
        self.optimization_history = []
        self.best_params = None
        self.best_score = -np.inf
        self.early_stopping_counter = 0
        self.early_stopping_best_score = -np.inf

        self.logger.info(f"✅ MSM Bayesian Optimizer initialized with {config.n_trials} trials, timeout {config.timeout}s")

    def optimize(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                evaluation_function: Optional[Callable] = None) -> Dict[str, Any]:
        """Perform two-step optimization: grid search first, then Bayesian optimization.

        Args:
            X: Feature matrix
            y: Optional target values (for supervised evaluation)
            evaluation_function: Custom evaluation function

        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()

        try:
            # Check if we should use subsampled data for efficiency
            if len(X) > 10000:
                X_sampled, y_sampled = self._subsample_data(X, y, max_samples=5000)
                self.logger.info(f"📊 Using subsampled data: {len(X_sampled)} samples for efficiency")
            else:
                X_sampled, y_sampled = X, y

            # Two-step optimization: grid search first, then Bayesian
            if self.config.use_two_step_optimization:
                results = self._optimize_two_step(X_sampled, y_sampled, evaluation_function)
            elif self.config.use_skopt and SKOPT_AVAILABLE:
                results = self._optimize_skopt(X_sampled, y_sampled, evaluation_function)
            elif OPTUNA_AVAILABLE:
                results = self._optimize_optuna(X_sampled, y_sampled, evaluation_function)
            else:
                raise RuntimeError("Neither scikit-optimize nor Optuna is available")

            execution_time = time.time() - start_time

            # Store optimization results - limit history size for memory efficiency
            if len(self.optimization_history) > 100:
                self.optimization_history = self.optimization_history[-50:]  # Keep last 50

            results['execution_time'] = execution_time
            results['optimization_history'] = self.optimization_history

            self.logger.info(f"✅ MSM optimization completed in {execution_time:.2f}s")
            self.logger.info(f"📊 Best score: {results['best_score']:.4f}")
            self.logger.info(f"🔧 Best params: {results['best_params']}")

            return results

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ MSM optimization failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': execution_time,
                'best_score': None,
                'best_params': None,
                'optimization_history': self.optimization_history
            }

    def _optimize_skopt(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                       evaluation_function: Optional[Callable] = None) -> Dict[str, Any]:
        """Optimize using scikit-optimize."""
        # Define parameter space
        dimensions = [
            Integer(low=self.config.n_states_min, high=self.config.n_states_max, name='n_states'),
            Integer(low=self.config.lag_time_min, high=self.config.lag_time_max, name='lag_time'),
            Categorical(['kmeans', 'agglomerative'], name='clustering_method'),
            Categorical(['euclidean', 'mahalanobis', 'correlation'], name='distance_metric'),
            Real(low=self.config.connectivity_threshold_min,
                 high=self.config.connectivity_threshold_max, name='connectivity_threshold'),
            Real(low=self.config.ergodic_cutoff_min,
                 high=self.config.ergodic_cutoff_max, name='ergodic_cutoff')
        ]

        # Define objective function
        @use_named_args(dimensions)
        def objective(**params):
            return self._evaluate_msm_params(X, y, params, evaluation_function)

        # Perform optimization
        result = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=self.config.n_trials,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs
        )

        # Extract best parameters
        best_params = dict(zip([dim.name for dim in dimensions], result.x))
        best_score = -result.fun  # gp_minimize minimizes, we want to maximize

        return {
            'success': True,
            'best_score': best_score,
            'best_params': best_params,
            'optimization_results': result
        }

    def _optimize_optuna(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                        evaluation_function: Optional[Callable] = None) -> Dict[str, Any]:
        """Optimize using Optuna."""
        study = create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.config.random_state)
        )

        def objective(trial: Trial):
            params = {
                'n_states': trial.suggest_int('n_states', self.config.n_states_min, self.config.n_states_max),
                'lag_time': trial.suggest_int('lag_time', self.config.lag_time_min, self.config.lag_time_max),
                'clustering_method': trial.suggest_categorical('clustering_method', ['kmeans', 'agglomerative']),
                'distance_metric': trial.suggest_categorical('distance_metric', ['euclidean', 'mahalanobis', 'correlation']),
                'connectivity_threshold': trial.suggest_float('connectivity_threshold',
                                                            self.config.connectivity_threshold_min,
                                                            self.config.connectivity_threshold_max),
                'ergodic_cutoff': trial.suggest_float('ergodic_cutoff',
                                                    self.config.ergodic_cutoff_min,
                                                    self.config.ergodic_cutoff_max)
            }

            score = self._evaluate_msm_params(X, y, params, evaluation_function)
            return score

        study.optimize(objective, n_trials=self.config.n_trials, timeout=self.config.timeout)

        best_params = study.best_params
        best_score = study.best_value

        return {
            'success': True,
            'best_score': best_score,
            'best_params': best_params,
            'optimization_results': study
        }

    def _evaluate_msm_params(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                           params: Optional[Dict[str, Any]] = None, evaluation_function: Optional[Callable] = None) -> float:
        """Evaluate MSM parameters and return score.

        Args:
            X: Feature matrix
            y: Optional target values
            params: MSM parameters to evaluate
            evaluation_function: Custom evaluation function

        Returns:
            Score (higher is better)
        """
        try:
            # Create MSM configuration
            msm_config = MSMConfig(
                n_states=params['n_states'],
                lag_time=params['lag_time'],
                clustering_method=params['clustering_method'],
                distance_metric=params['distance_metric'],
                connectivity_threshold=params['connectivity_threshold'],
                ergodic_cutoff=params['ergodic_cutoff']
            )

            # Create MSM clusterer
            msm_clusterer = MSMClusterer(msm_config)

            # Perform clustering
            result = msm_clusterer.cluster(X)

            if not result.success:
                return -1.0  # Return negative score for failed clustering

            # Calculate score based on objective
            if evaluation_function is not None:
                score = evaluation_function(result, X, y)
            else:
                score = self._calculate_msm_score(result, X, y)

            # Track optimization history
            self.optimization_history.append({
                'params': params,
                'score': score,
                'msm_score': result.msm_score if hasattr(result, 'msm_score') else None
            })

            return score

        except Exception as e:
            self.logger.warning(f"⚠️ MSM parameter evaluation failed: {e}")
            return -1.0

    def _subsample_data(self, X: np.ndarray, y: Optional[np.ndarray],
                       max_samples: int = 5000) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Subsample data for computational efficiency.

        Args:
            X: Feature matrix
            y: Target values
            max_samples: Maximum number of samples to use

        Returns:
            Tuple of (X_subsampled, y_subsampled)
        """
        n_samples = min(len(X), max_samples)
        indices = np.random.choice(len(X), n_samples, replace=False)

        X_subsampled = X[indices]

        if y is not None:
            y_subsampled = y[indices] if len(y) == len(X) else None
        else:
            y_subsampled = None

        return X_subsampled, y_subsampled

    def _optimize_two_step(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                          evaluation_function: Optional[Callable] = None) -> Dict[str, Any]:
        """Two-step optimization: coarse grid search first, then Bayesian around best."""

        # Step 1: Coarse grid search
        self.logger.info("🔍 Step 1: Coarse grid search")
        grid_results = self._grid_search_step(X, y, evaluation_function)

        if not grid_results['success']:
            self.logger.warning("⚠️ Grid search failed, falling back to Bayesian optimization")
            if self.config.use_skopt and SKOPT_AVAILABLE:
                return self._optimize_skopt(X, y, evaluation_function)
            else:
                raise RuntimeError("Grid search failed and Bayesian optimization unavailable")

        # Step 2: Bayesian optimization around best grid search result
        self.logger.info("🔍 Step 2: Bayesian optimization around best grid search result")
        best_grid_params = grid_results['best_params']
        self.logger.info(f"📊 Best grid search score: {grid_results['best_score']:.4f}")
        self.logger.info(f"🔧 Best grid search params: {best_grid_params}")

        # Use Bayesian optimization with tighter bounds around best grid result
        bayesian_results = self._optimize_bayesian_around_best(X, y, best_grid_params, evaluation_function)

        return {
            'success': True,
            'best_score': bayesian_results['best_score'],
            'best_params': bayesian_results['best_params'],
            'grid_search_results': grid_results,
            'bayesian_results': bayesian_results,
            'two_step_used': True
        }

    def _grid_search_step(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                         evaluation_function: Optional[Callable] = None) -> Dict[str, Any]:
        """Perform coarse grid search to find promising parameter regions."""

        from src.utils.ml_common.optimization.grid_utils import build_coarse_grid_from_search_space

        # Define search space for grid search
        search_space = {
            'n_states': {'type': 'int', 'low': self.config.n_states_min, 'high': self.config.n_states_max},
            'lag_time': {'type': 'int', 'low': self.config.lag_time_min, 'high': self.config.lag_time_max},
            'clustering_method': {'type': 'categorical', 'choices': ['kmeans', 'agglomerative']},
            'distance_metric': {'type': 'categorical', 'choices': ['euclidean', 'mahalanobis', 'correlation']},
            'connectivity_threshold': {'type': 'float', 'low': self.config.connectivity_threshold_min, 'high': self.config.connectivity_threshold_max},
            'ergodic_cutoff': {'type': 'float', 'low': self.config.ergodic_cutoff_min, 'high': self.config.ergodic_cutoff_max}
        }

        # Build coarse grid
        grid_params = build_coarse_grid_from_search_space(search_space, self.config.grid_search_n_points)

        if not grid_params:
            return {'success': False, 'error': 'No grid parameters generated'}

        self.logger.info(f"🔍 Evaluating {len(grid_params)} grid search points")

        # Evaluate each grid point
        grid_scores = []
        for i, params in enumerate(grid_params):
            try:
                score = self._evaluate_msm_params(X, y, params, evaluation_function)
                grid_scores.append((params, score))

                if i % 5 == 0:  # Log progress every 5 evaluations
                    self.logger.debug(f"📊 Grid search progress: {i+1}/{len(grid_params)}, best score: {max([s for _, s in grid_scores]):.4f}")

            except Exception as e:
                self.logger.warning(f"⚠️ Grid search evaluation {i+1} failed: {e}")
                continue

        if not grid_scores:
            return {'success': False, 'error': 'All grid search evaluations failed'}

        # Find best grid search result
        best_params, best_score = max(grid_scores, key=lambda x: x[1])

        return {
            'success': True,
            'best_score': best_score,
            'best_params': best_params,
            'grid_scores': grid_scores
        }

    def _optimize_bayesian_around_best(self, X: np.ndarray, best_grid_params: Dict[str, Any],
                                      y: Optional[np.ndarray] = None,
                                      evaluation_function: Optional[Callable] = None) -> Dict[str, Any]:
        """Perform Bayesian optimization around the best grid search result."""

        # Use scikit-optimize for Bayesian optimization around best
        if not (self.config.use_skopt and SKOPT_AVAILABLE):
            return {'success': False, 'error': 'scikit-optimize not available for Bayesian step'}

        # Define tighter parameter space around best grid result
        dimensions = [
            Integer(low=max(self.config.n_states_min, best_grid_params['n_states'] - 3),
                    high=min(self.config.n_states_max, best_grid_params['n_states'] + 3),
                    name='n_states'),
            Integer(low=max(self.config.lag_time_min, best_grid_params['lag_time'] - 1),
                    high=min(self.config.lag_time_max, best_grid_params['lag_time'] + 1),
                    name='lag_time'),
            Categorical([best_grid_params['clustering_method']], name='clustering_method'),
            Categorical([best_grid_params['distance_metric']], name='distance_metric'),
            Real(low=max(self.config.connectivity_threshold_min, best_grid_params['connectivity_threshold'] * 0.8),
                 high=min(self.config.connectivity_threshold_max, best_grid_params['connectivity_threshold'] * 1.2),
                 name='connectivity_threshold'),
            Real(low=max(self.config.ergodic_cutoff_min, best_grid_params['ergodic_cutoff'] * 0.5),
                 high=min(self.config.ergodic_cutoff_max, best_grid_params['ergodic_cutoff'] * 2.0),
                 name='ergodic_cutoff')
        ]

        @use_named_args(dimensions)
        def objective(**params):
            return self._evaluate_msm_params(X, y, params, evaluation_function)

        # Perform Bayesian optimization
        result = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=self.config.n_trials,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs
        )

        # Extract best parameters
        best_params = dict(zip([dim.name for dim in dimensions], result.x))
        best_score = -result.fun  # gp_minimize minimizes, we want to maximize

        return {
            'success': True,
            'best_score': best_score,
            'best_params': best_params,
            'optimization_results': result
        }

    def _calculate_msm_score(self, result: Any, X: np.ndarray, y: Optional[np.ndarray] = None) -> float:
        """Calculate MSM score based on optimization objective.

        Args:
            result: MSM clustering result
            X: Feature matrix
            y: Optional target values

        Returns:
            Score (higher is better)
        """
        try:
            # Base MSM score
            base_score = getattr(result, 'msm_score', 0.0)

            # Add additional metrics based on objective
            if self.config.optimization_objective == 'silhouette':
                # Calculate silhouette score
                from sklearn.metrics import silhouette_score
                if hasattr(result, 'labels') and len(np.unique(result.labels)) > 1:
                    try:
                        sil_score = silhouette_score(X, result.labels)
                        return sil_score
                    except Exception:
                        pass

            elif self.config.optimization_objective == 'connectivity':
                # Use connectivity as primary score
                if hasattr(result, 'transition_matrix'):
                    connectivity = np.mean(result.transition_matrix > self.config.connectivity_threshold_min)
                    return connectivity

            elif self.config.optimization_objective == 'stationarity':
                # Use stationary distribution properties
                if hasattr(result, 'stationary_distribution'):
                    stationarity_score = 1.0 / (1.0 + np.var(result.stationary_distribution))
                    return stationarity_score

            elif self.config.optimization_objective == 'implied_timescale':
                # Use implied timescale properties
                if hasattr(result, 'implied_timescales') and len(result.implied_timescales) > 0:
                    timescale_score = np.mean(result.implied_timescales)
                    return timescale_score

            # Default to MSM score
            return base_score

        except Exception as e:
            self.logger.warning(f"⚠️ MSM score calculation failed: {e}")
            return 0.0


class AttentionNetworkOptimizer:
    """Bayesian optimization for attention network parameters."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize attention network optimizer.

        Args:
            config: Optimization configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        if not SKOPT_AVAILABLE and not OPTUNA_AVAILABLE:
            raise ImportError("Either scikit-optimize or Optuna must be installed for Bayesian optimization")

    def optimize(self, X: np.ndarray, y: np.ndarray, base_model: Any) -> Dict[str, Any]:
        """Optimize attention network parameters.

        Args:
            X: Training features
            y: Target values
            base_model: Base model to enhance with attention

        Returns:
            Dictionary with optimization results
        """
        try:
            if SKOPT_AVAILABLE:
                return self._optimize_skopt(X, y, base_model)
            elif OPTUNA_AVAILABLE:
                return self._optimize_optuna(X, y, base_model)
            else:
                raise RuntimeError("Neither scikit-optimize nor Optuna is available")

        except Exception as e:
            self.logger.error(f"❌ Attention network optimization failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'best_params': None,
                'best_score': None
            }

    def _optimize_skopt(self, X: np.ndarray, y: np.ndarray, base_model: Any) -> Dict[str, Any]:
        """Optimize using scikit-optimize."""
        # Define parameter space
        dimensions = [
            Integer(low=16, high=256, name='attention_dim'),
            Integer(low=2, high=16, name='attention_heads'),
            Real(low=1e-5, high=1e-2, name='learning_rate'),
            Real(low=1e-6, high=1e-3, name='weight_decay'),
            Real(low=0.0, high=0.5, name='dropout'),
            Categorical([True, False], name='use_temporal_attention')
        ]

        @use_named_args(dimensions)
        def objective(**params):
            return self._evaluate_attention_params(X, y, base_model, params)

        # Perform optimization
        result = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=self.config.get('n_trials', 30),
            random_state=self.config.get('random_state', 42)
        )

        # Extract best parameters
        best_params = dict(zip([dim.name for dim in dimensions], result.x))
        best_score = -result.fun  # gp_minimize minimizes, we want to maximize

        return {
            'success': True,
            'best_score': best_score,
            'best_params': best_params,
            'optimization_results': result
        }

    def _optimize_optuna(self, X: np.ndarray, y: np.ndarray, base_model: Any) -> Dict[str, Any]:
        """Optimize using Optuna."""
        study = create_study(direction='maximize')

        def objective(trial: Trial):
            params = {
                'attention_dim': trial.suggest_int('attention_dim', 16, 256),
                'attention_heads': trial.suggest_int('attention_heads', 2, 16),
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3),
                'dropout': trial.suggest_float('dropout', 0.0, 0.5),
                'use_temporal_attention': trial.suggest_categorical('use_temporal_attention', [True, False])
            }

            score = self._evaluate_attention_params(X, y, base_model, params)
            return score

        study.optimize(objective, n_trials=self.config.get('n_trials', 30))

        return {
            'success': True,
            'best_score': study.best_value,
            'best_params': study.best_params,
            'optimization_results': study
        }

    def _evaluate_attention_params(self, X: np.ndarray, y: np.ndarray,
                                 base_model: Any, params: Dict[str, Any]) -> float:
        """Evaluate attention parameters.

        Args:
            X: Feature matrix
            y: Target values
            base_model: Base model
            params: Attention parameters

        Returns:
            Score (higher is better)
        """
        try:
            from src.training.steps.model_training.attention_enhanced_models import (
                create_attention_model
            )

            # Create attention-enhanced model
            attention_model = create_attention_model(
                model_type=type(base_model).__name__.lower().replace('regressor', '').replace('classifier', ''),
                attention_dim=params['attention_dim'],
                attention_heads=params['attention_heads'],
                model_params={
                    'learning_rate': params['learning_rate'],
                    'weight_decay': params['weight_decay']
                },
                use_temporal_attention=params['use_temporal_attention'],
                dropout=params['dropout']
            )

            # Simple evaluation (in practice, use cross-validation)
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(attention_model, X, y, cv=3, scoring='r2')
            return np.mean(scores)

        except Exception as e:
            self.logger.warning(f"⚠️ Attention parameter evaluation failed: {e}")
            return -1.0


class MetaLearnerOptimizer:
    """Bayesian optimization for meta-learner hyperparameters."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize meta-learner optimizer.

        Args:
            config: Optimization configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def optimize(self, base_models: List[Any], X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Optimize meta-learner hyperparameters.

        Args:
            base_models: List of base models for stacking
            X: Feature matrix
            y: Target values

        Returns:
            Dictionary with optimization results
        """
        try:
            if SKOPT_AVAILABLE:
                return self._optimize_skopt(base_models, X, y)
            elif OPTUNA_AVAILABLE:
                return self._optimize_optuna(base_models, X, y)
            else:
                raise RuntimeError("Neither scikit-optimize nor Optuna is available")

        except Exception as e:
            self.logger.error(f"❌ Meta-learner optimization failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'best_params': None,
                'best_score': None
            }

    def _optimize_skopt(self, base_models: List[Any], X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Optimize using scikit-optimize."""
        # Define parameter space for meta-learner (using your actual meta-learners)
        dimensions = [
            Categorical(['advanced_mamba_hybrid', 'financial_resnet'], name='meta_learner_type'),
            Integer(low=32, high=256, name='hidden_units'),
            Integer(low=2, high=8, name='num_layers'),
            Real(low=1e-4, high=1e-2, name='learning_rate'),
            Real(low=1e-6, high=1e-3, name='weight_decay'),
            Real(low=0.0, high=0.5, name='dropout_rate'),
            Integer(low=16, high=128, name='attention_dim'),
            Integer(low=2, high=16, name='attention_heads')
        ]

        @use_named_args(dimensions)
        def objective(**params):
            return self._evaluate_meta_learner(base_models, X, y, params)

        # Perform optimization
        result = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=self.config.get('n_trials', 30),
            random_state=self.config.get('random_state', 42)
        )

        # Extract best parameters
        best_params = dict(zip([dim.name for dim in dimensions], result.x))
        best_score = -result.fun

        return {
            'success': True,
            'best_score': best_score,
            'best_params': best_params,
            'optimization_results': result
        }

    def _optimize_optuna(self, base_models: List[Any], X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Optimize using Optuna."""
        study = create_study(direction='maximize')

        def objective(trial: Trial):
            params = {
                'meta_learner_type': trial.suggest_categorical('meta_learner_type', ['advanced_mamba_hybrid', 'financial_resnet']),
                'hidden_units': trial.suggest_int('hidden_units', 32, 256),
                'num_layers': trial.suggest_int('num_layers', 2, 8),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.5),
                'attention_dim': trial.suggest_int('attention_dim', 16, 128),
                'attention_heads': trial.suggest_int('attention_heads', 2, 16)
            }

            score = self._evaluate_meta_learner(base_models, X, y, params)
            return score

        study.optimize(objective, n_trials=self.config.get('n_trials', 30))

        return {
            'success': True,
            'best_score': study.best_value,
            'best_params': study.best_params,
            'optimization_results': study
        }

    def _evaluate_meta_learner(self, base_models: List[Any], X: np.ndarray,
                              y: np.ndarray, params: Dict[str, Any]) -> float:
        """Evaluate meta-learner with given parameters.

        Args:
            base_models: List of base models
            X: Feature matrix
            y: Target values
            params: Meta-learner parameters

        Returns:
            Score (higher is better)
        """
        try:
            from sklearn.model_selection import cross_val_score
            from sklearn.ensemble import StackingRegressor
            from xgboost import XGBRegressor

            # Create base estimators (simplified for efficiency)
            estimators = [(f'model_{i}', model) for i, model in enumerate(base_models)]

            # Create meta-learner based on type - FAST FAIL if not available
            if params['meta_learner_type'] == 'advanced_mamba_hybrid':
                # Use AdvancedMambaHybrid - FAST FAIL if not available
                try:
                    from src.utils.ml_common.models.model_factory import ModelType
                    meta_learner = self._create_advanced_mamba_hybrid(params)
                except Exception as e:
                    raise RuntimeError(f"AdvancedMambaHybrid meta-learner not available. "
                                     f"Please ensure your environment supports AdvancedMambaHybrid models. "
                                     f"Error: {e}")
            elif params['meta_learner_type'] == 'financial_resnet':
                # Use FinancialResNet - FAST FAIL if not available
                try:
                    from src.utils.ml_common.models.model_factory import ModelType
                    meta_learner = self._create_financial_resnet(params)
                except Exception as e:
                    raise RuntimeError(f"FinancialResNet meta-learner not available. "
                                     f"Please ensure your environment supports FinancialResNet models. "
                                     f"Error: {e}")
            else:  # xgboost - only used when explicitly requested
                raise RuntimeError(f"Unsupported meta-learner type: {params['meta_learner_type']}. "
                                 f"Expected: 'advanced_mamba_hybrid' or 'financial_resnet'")

            # Create stacking ensemble
            stacking_ensemble = StackingRegressor(
                estimators=estimators,
                final_estimator=meta_learner,
                cv=3  # Reduced for efficiency
            )

            # Evaluate using cross-validation
            scores = cross_val_score(stacking_ensemble, X, y, cv=3, scoring='r2')
            return np.mean(scores)

        except Exception as e:
            self.logger.warning(f"⚠️ Meta-learner evaluation failed: {e}")
            return -1.0

    def _create_advanced_mamba_hybrid(self, params: Dict[str, Any]) -> Any:
        """Create AdvancedMambaHybrid model with given parameters."""
        try:
            from src.utils.ml_common.models.model_factory import EnhancedModelFactory, ModelType, ModelConfig

            factory = EnhancedModelFactory()
            model_config = ModelConfig(
                model_type=ModelType.ADVANCED_MAMBA_HYBRID,
                model_name="optimized_mamba",
                model_params={
                    'hidden_units': params['hidden_units'],
                    'num_layers': params['num_layers'],
                    'learning_rate': params['learning_rate'],
                    'weight_decay': params['weight_decay'],
                    'dropout_rate': params['dropout_rate'],
                    'attention_dim': params['attention_dim'],
                    'attention_heads': params['attention_heads']
                }
            )

            return factory.create_model(model_config)

        except Exception as e:
            self.logger.warning(f"⚠️ AdvancedMambaHybrid creation failed: {e}")
            raise

    def _create_financial_resnet(self, params: Dict[str, Any]) -> Any:
        """Create FinancialResNet model with given parameters."""
        try:
            from src.utils.ml_common.models.model_factory import EnhancedModelFactory, ModelType, ModelConfig

            factory = EnhancedModelFactory()
            model_config = ModelConfig(
                model_type=ModelType.FINANCIAL_RESNET,
                model_name="optimized_resnet",
                model_params={
                    'hidden_units': params['hidden_units'],
                    'num_layers': params['num_layers'],
                    'learning_rate': params['learning_rate'],
                    'weight_decay': params['weight_decay'],
                    'dropout_rate': params['dropout_rate'],
                    'attention_dim': params['attention_dim'],
                    'attention_heads': params['attention_heads']
                }
            )

            return factory.create_model(model_config)

        except Exception as e:
            self.logger.warning(f"⚠️ FinancialResNet creation failed: {e}")
            raise


# Convenience functions
def optimize_msm_parameters(X: np.ndarray, config: Optional[MSMOptimizationConfig] = None) -> Dict[str, Any]:
    """Optimize MSM parameters using Bayesian optimization.

    Args:
        X: Feature matrix
        config: Optimization configuration

    Returns:
        Dictionary with optimization results
    """
    if config is None:
        config = MSMOptimizationConfig()

    optimizer = MSMBayesianOptimizer(config)
    return optimizer.optimize(X)


def optimize_attention_network(X: np.ndarray, y: np.ndarray, base_model: Any,
                              config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Optimize attention network parameters.

    Args:
        X: Training features
        y: Target values
        base_model: Base model
        config: Optimization configuration

    Returns:
        Dictionary with optimization results
    """
    if config is None:
        config = {'n_trials': 30, 'random_state': 42}

    optimizer = AttentionNetworkOptimizer(config)
    return optimizer.optimize(X, y, base_model)


def optimize_meta_learner(base_models: List[Any], X: np.ndarray, y: np.ndarray,
                         config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Optimize meta-learner hyperparameters using Bayesian optimization.

    Args:
        base_models: List of base models for stacking ensemble
        X: Feature matrix
        y: Target values
        config: Optimization configuration

    Returns:
        Dictionary with optimization results
    """
    if config is None:
        config = {'n_trials': 30, 'random_state': 42}

    optimizer = MetaLearnerOptimizer(config)
    return optimizer.optimize(base_models, X, y)


def optimize_deepscaler_parameters(X: np.ndarray, y: np.ndarray,
                                  config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Optimize DeepScaler parameters using Bayesian optimization.

    Args:
        X: Training features
        y: Target values
        config: Optimization configuration

    Returns:
        Dictionary with optimization results
    """
    if config is None:
        config = {'n_trials': 30, 'random_state': 42}

    optimizer = DeepScalerOptimizer(config)
    return optimizer.optimize(X, y)


def optimize_attention_network(X: np.ndarray, y: np.ndarray, base_model: Any,
                              config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Optimize attention network parameters.

    Args:
        X: Training features
        y: Target values
        base_model: Base model
        config: Optimization configuration

    Returns:
        Dictionary with optimization results
    """
    if config is None:
        config = {'n_trials': 30, 'random_state': 42}

    optimizer = AttentionNetworkOptimizer(config)
    return optimizer.optimize(X, y, base_model)


class DeepScalerOptimizer:
    """Bayesian optimization for DeepScaler hyperparameters."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize DeepScaler optimizer.

        Args:
            config: Optimization configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def optimize(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Optimize DeepScaler parameters.

        Args:
            X: Training features
            y: Target values

        Returns:
            Dictionary with optimization results
        """
        try:
            if SKOPT_AVAILABLE:
                return self._optimize_skopt(X, y)
            elif OPTUNA_AVAILABLE:
                return self._optimize_optuna(X, y)
            else:
                raise RuntimeError("Neither scikit-optimize nor Optuna is available")

        except Exception as e:
            self.logger.error(f"❌ DeepScaler optimization failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'best_params': None,
                'best_score': None
            }

    def _optimize_skopt(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Optimize using scikit-optimize."""
        # Define parameter space
        dimensions = [
            Integer(low=32, high=256, name='hidden_units'),
            Integer(low=2, high=10, name='num_layers'),
            Real(low=1e-4, high=1e-2, name='learning_rate'),
            Real(low=1e-6, high=1e-3, name='weight_decay'),
            Real(low=0.0, high=0.5, name='dropout_rate'),
            Categorical(['adam', 'adamw', 'sgd'], name='optimizer'),
            Real(low=0.1, high=1.0, name='feature_dropout')
        ]

        @use_named_args(dimensions)
        def objective(**params):
            return self._evaluate_deepscaler_params(X, y, params)

        # Perform optimization
        result = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=self.config.get('n_trials', 30),
            random_state=self.config.get('random_state', 42)
        )

        # Extract best parameters
        best_params = dict(zip([dim.name for dim in dimensions], result.x))
        best_score = -result.fun  # gp_minimize minimizes, we want to maximize

        return {
            'success': True,
            'best_score': best_score,
            'best_params': best_params,
            'optimization_results': result
        }

    def _optimize_optuna(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Optimize using Optuna."""
        study = create_study(direction='maximize')

        def objective(trial: Trial):
            params = {
                'hidden_units': trial.suggest_int('hidden_units', 32, 256),
                'num_layers': trial.suggest_int('num_layers', 2, 10),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.5),
                'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd']),
                'feature_dropout': trial.suggest_float('feature_dropout', 0.1, 1.0)
            }

            score = self._evaluate_deepscaler_params(X, y, params)
            return score

        study.optimize(objective, n_trials=self.config.get('n_trials', 30))

        return {
            'success': True,
            'best_score': study.best_value,
            'best_params': study.best_params,
            'optimization_results': study
        }

    def _evaluate_deepscaler_params(self, X: np.ndarray, y: np.ndarray, params: Dict[str, Any]) -> float:
        """Evaluate DeepScaler parameters.

        Args:
            X: Feature matrix
            y: Target values
            params: DeepScaler parameters

        Returns:
            Score (higher is better)
        """
        try:
            # Import DeepScaler - this might not be available in all environments
            try:
                from deepscaler import DeepScaler
            except ImportError:
                self.logger.warning("⚠️ DeepScaler not available, using fallback evaluation")
                return self._fallback_deepscaler_evaluation(X, y, params)

            # Create DeepScaler with given parameters
            deepscaler = DeepScaler(
                hidden_units=params['hidden_units'],
                num_layers=params['num_layers'],
                learning_rate=params['learning_rate'],
                weight_decay=params['weight_decay'],
                dropout_rate=params['dropout_rate'],
                optimizer=params['optimizer'],
                feature_dropout=params['feature_dropout']
            )

            # Simple evaluation (in practice, use cross-validation)
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(deepscaler, X, y, cv=3, scoring='r2')
            return np.mean(scores)

        except Exception as e:
            self.logger.warning(f"⚠️ DeepScaler parameter evaluation failed: {e}")
            return -1.0

    def _fallback_deepscaler_evaluation(self, X: np.ndarray, y: np.ndarray, params: Dict[str, Any]) -> float:
        """Fallback evaluation when DeepScaler is not available."""
        try:
            # Use a simple MLP as fallback
            from sklearn.neural_network import MLPRegressor
            from sklearn.model_selection import cross_val_score

            mlp = MLPRegressor(
                hidden_layer_sizes=[params['hidden_units']] * params['num_layers'],
                learning_rate_init=params['learning_rate'],
                alpha=params['weight_decay'],
                random_state=42,
                max_iter=200
            )

            scores = cross_val_score(mlp, X, y, cv=3, scoring='r2')
            return np.mean(scores)

        except Exception as e:
            self.logger.warning(f"⚠️ Fallback DeepScaler evaluation failed: {e}")
            return -1.0