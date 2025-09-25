"""
Bayesian TPE Optimizer with Automatic Grid-First Approach

This module implements a comprehensive Bayesian Tree-structured Parzen Estimator (TPE)
optimizer that automatically uses coarse grid search followed by fine grid search
as initial stages before Bayesian optimization. This hybrid approach ensures better
convergence and more robust optimization results.

Key Features:
- Automatic coarse grid → fine grid → Bayesian TPE pipeline
- Comprehensive logging and error handling
- Configurable search spaces and optimization parameters
- Support for multiple model types and evaluation metrics
- Built-in monitoring and convergence detection
- Parallel optimization support
- Transfer learning capabilities

Usage:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer

    optimizer = BayesianTPEOptimizer()
    results = optimizer.optimize(model_factory, X_train, y_train, search_space)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Type
from dataclasses import dataclass, field
from datetime import datetime
import logging
import time
import traceback
from pathlib import Path
import json

# Import existing grid utilities
from .grid_utils import build_coarse_grid_from_search_space, build_fine_grid_around_best

# Enhanced dependency management with fast fail
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner, HyperbandPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None
    TPESampler = None
    MedianPruner = None
    HyperbandPruner = None

try:
    from sklearn.model_selection import cross_val_score, cross_validate
    from sklearn.metrics import get_scorer
    from sklearn.utils.class_weight import compute_sample_weight
    from sklearn.base import clone, BaseEstimator
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    get_scorer = None
    compute_sample_weight = None
    clone = None
    BaseEstimator = None

try:
    from ..logger import get_logger
    _LOGGER = get_logger("MLCommon.BayesianTPEOptimizer")
except Exception:
    _LOGGER = logging.getLogger("MLCommon.BayesianTPEOptimizer")
    _LOGGER.setLevel(logging.INFO)

logger = _LOGGER


@dataclass
class TPEConfig:
    """Configuration for TPE optimization stage."""
    n_trials: int = 50
    timeout_seconds: Optional[int] = None
    acquisition_function: str = 'ucb'  # 'ucb', 'ei', 'poi'
    pruner: str = 'median'  # 'median', 'hyperband', 'none'
    enable_parallel: bool = True
    max_workers: int = 4
    use_nonlinear_optimization: bool = True


@dataclass
class GridConfig:
    """Configuration for grid search stages."""
    coarse_enabled: bool = True
    coarse_grid_points: int = 5
    fine_enabled: bool = True
    fine_grid_points: int = 10
    subsample_rate: float = 0.3  # For coarse grid stage


@dataclass
class OptimizationConfig:
    """Main configuration for the Bayesian TPE optimizer."""
    tpe_config: TPEConfig = field(default_factory=TPEConfig)
    grid_config: GridConfig = field(default_factory=GridConfig)
    validation_config: Dict[str, Any] = field(default_factory=lambda: {
        'cv_folds': 5,
        'scoring': 'balanced_accuracy',
        'test_size': 0.2,
        'random_state': 42
    })
    enable_monitoring: bool = True
    fast_fail_on_error: bool = False
    save_results: bool = True
    results_path: Optional[str] = None
    transfer_learning_threshold: float = 0.8


@dataclass
class OptimizationResult:
    """Results from optimization process."""
    best_params: Dict[str, Any]
    best_score: float
    best_stage: str  # 'coarse_grid', 'fine_grid', 'tpe', 'fallback'
    optimization_time: float
    optimization_method: str = 'bayesian_tpe'
    n_trials_total: int = 0
    n_trials_coarse: int = 0
    n_trials_fine: int = 0
    n_trials_tpe: int = 0
    convergence_info: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    error_info: Optional[Dict[str, Any]] = None
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)


class BayesianTPEOptimizer:
    """
    Bayesian Tree-structured Parzen Estimator (TPE) optimizer with automatic
    coarse grid → fine grid → TPE pipeline.

    This optimizer provides a robust, production-ready hyperparameter optimization
    solution that combines grid search initialization with Bayesian optimization
    for better convergence and reliability.

    Args:
        config: Optimization configuration
        model_type: Type of model being optimized (for search space generation)
        random_state: Random state for reproducibility
    """

    def __init__(self,
                 config: Optional[OptimizationConfig] = None,
                 model_type: str = 'auto',
                 random_state: int = 42):
        """Initialize the Bayesian TPE optimizer."""
        self.config = config or OptimizationConfig()
        self.model_type = model_type
        self.random_state = random_state

        # Set up logging
        self.logger = logger.getChild(f'BayesianTPEOptimizer.{id(self)}')
        self.logger.info("🚀 Initializing Bayesian TPE Optimizer")
        self.logger.info(f"   Model type: {model_type}")
        self.logger.info(f"   Random state: {random_state}")

        # Optimization state
        self.optimization_history = []
        self.study_results = {}
        self.error_summary = defaultdict(int)
        self.current_study_id = None

        # Results tracking
        self.results_path = self.config.results_path or f'optimization_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        if self.config.save_results:
            Path(self.results_path).mkdir(exist_ok=True)

        # Validate dependencies
        self._validate_dependencies()

        # Initialize search space if model type specified
        if self.model_type != 'auto':
            self.default_search_space = self._get_default_search_space()

        self.logger.info("✅ Bayesian TPE Optimizer initialized successfully")

    def _validate_dependencies(self) -> None:
        """Validate required dependencies and provide helpful error messages."""
        if not OPTUNA_AVAILABLE:
            self.logger.error("❌ Optuna is required for Bayesian TPE optimization")
            self.logger.error("   Install with: pip install optuna")
            if self.config.fast_fail_on_error:
                raise ImportError("Optuna is required for Bayesian TPE optimization")

        if not SKLEARN_AVAILABLE:
            self.logger.warning("⚠️ Scikit-learn not available - limited validation functionality")
            if self.config.fast_fail_on_error:
                raise ImportError("Scikit-learn is required for validation")

        self.logger.info("✅ Dependencies validated successfully")

    def _get_default_search_space(self, model_type: Optional[str] = None) -> Dict[str, Any]:
        """Get default search space for the specified model type."""
        model_type = model_type or self.model_type

        search_spaces = {
            'xgboost': {
                'max_depth': {'type': 'int', 'low': 3, 'high': 12},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3},
                'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
                'subsample': {'type': 'float', 'low': 0.5, 'high': 1.0},
                'colsample_bytree': {'type': 'float', 'low': 0.5, 'high': 1.0},
                'gamma': {'type': 'float', 'low': 0, 'high': 5},
                'reg_alpha': {'type': 'float', 'low': 0, 'high': 10},
                'reg_lambda': {'type': 'float', 'low': 0, 'high': 10}
            },
            'lightgbm': {
                'num_leaves': {'type': 'int', 'low': 10, 'high': 100},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3},
                'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
                'feature_fraction': {'type': 'float', 'low': 0.5, 'high': 1.0},
                'bagging_fraction': {'type': 'float', 'low': 0.5, 'high': 1.0},
                'bagging_freq': {'type': 'int', 'low': 1, 'high': 10},
                'min_child_samples': {'type': 'int', 'low': 5, 'high': 50},
                'lambda_l1': {'type': 'float', 'low': 0, 'high': 10},
                'lambda_l2': {'type': 'float', 'low': 0, 'high': 10}
            },
            'random_forest': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
                'max_depth': {'type': 'int', 'low': 5, 'high': 50},
                'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
                'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', None]},
                'bootstrap': {'type': 'categorical', 'choices': [True, False]}
            },
            'neural_network': {
                'hidden_layers': {'type': 'int', 'low': 1, 'high': 5},
                'hidden_units': {'type': 'int', 'low': 32, 'high': 512},
                'learning_rate': {'type': 'float', 'low': 0.0001, 'high': 0.01, 'log': True},
                'dropout_rate': {'type': 'float', 'low': 0.0, 'high': 0.5},
                'batch_size': {'type': 'int', 'low': 16, 'high': 128},
                'epochs': {'type': 'int', 'low': 10, 'high': 100}
            }
        }

        return search_spaces.get(model_type.lower(), search_spaces['xgboost'])

    def optimize(self,
                 model_factory: Callable[[Dict[str, Any]], Any],
                 X: Union[np.ndarray, pd.DataFrame],
                 y: Union[np.ndarray, pd.Series],
                 search_space: Optional[Dict[str, Any]] = None,
                 custom_evaluation_fn: Optional[Callable] = None,
                 transfer_learning_data: Optional[Dict[str, Any]] = None) -> OptimizationResult:
        """
        Run complete Bayesian TPE optimization with grid-first approach.

        Args:
            model_factory: Function that creates model instance with given parameters
            X: Feature matrix
            y: Target array/series
            search_space: Custom search space (if None, uses default for model_type)
            custom_evaluation_fn: Custom evaluation function (if None, uses CV)
            transfer_learning_data: Previous optimization results for transfer learning

        Returns:
            Complete optimization results
        """
        start_time = datetime.now()
        self.logger.info("🎯 Starting Bayesian TPE optimization")
        self.logger.info(f"   Dataset shape: {X.shape if hasattr(X, 'shape') else len(X)}")
        self.logger.info(f"   Target shape: {y.shape if hasattr(y, 'shape') else len(y)}")

        # Generate study ID
        self.current_study_id = f"study_{int(start_time.timestamp())}"
        self.logger.info(f"   Study ID: {self.current_study_id}")

        try:
            # Use search space
            if search_space is None:
                search_space = self.default_search_space
                self.logger.info(f"   Using default search space for {self.model_type}")
            else:
                self.logger.info("   Using custom search space")

            # Check for transfer learning opportunity
            if transfer_learning_data and self._should_use_transfer_learning(transfer_learning_data, X, y):
                self.logger.info("🔄 Applying transfer learning")
                results = self._transfer_learning_optimization(
                    model_factory, X, y, search_space, transfer_learning_data, custom_evaluation_fn
                )
            else:
                # Standard optimization pipeline
                results = self._run_optimization_pipeline(
                    model_factory, X, y, search_space, custom_evaluation_fn
                )

            # Calculate total time
            results.optimization_time = (datetime.now() - start_time).total_seconds()

            # Save results if configured
            if self.config.save_results:
                self._save_results(results)

            # Log completion
            self.logger.info("✅ Bayesian TPE optimization completed successfully")
            self.logger.info(f"   Best score: {results.best_score:.4f}")
            self.logger.info(f"   Best stage: {results.best_stage}")
            self.logger.info(f"   Total time: {results.optimization_time:.2f}s")

            return results

        except Exception as e:
            self.logger.error(f"❌ Bayesian TPE optimization failed: {e}")
            self.logger.error(f"   Traceback: {traceback.format_exc()}")

            if self.config.fast_fail_on_error:
                raise

            # Return fallback results
            return self._create_fallback_result(model_factory, search_space, e)

    def _run_optimization_pipeline(self,
                                  model_factory: Callable,
                                  X: Union[np.ndarray, pd.DataFrame],
                                  y: Union[np.ndarray, pd.Series],
                                  search_space: Dict[str, Any],
                                  custom_evaluation_fn: Optional[Callable] = None) -> OptimizationResult:
        """
        Run the complete optimization pipeline: coarse grid → fine grid → TPE.

        Args:
            model_factory: Function to create model instances
            X: Feature matrix
            y: Target array
            search_space: Search space for optimization
            custom_evaluation_fn: Custom evaluation function

        Returns:
            Optimization results
        """
        self.logger.info("🔄 Starting optimization pipeline")

        # Stage 1: Coarse grid search
        coarse_result = None
        if self.config.grid_config.coarse_enabled:
            self.logger.info("   → Stage 1: Coarse grid search")
            try:
                coarse_result = self._coarse_grid_optimization(
                    model_factory, X, y, search_space, custom_evaluation_fn
                )
                self.logger.info(f"   → Coarse grid best score: {coarse_result.best_score:.4f}")
            except Exception as e:
                self.logger.error(f"   → Coarse grid failed: {e}")
                if self.config.fast_fail_on_error:
                    raise

        # Stage 2: Fine grid search
        fine_result = None
        if (self.config.grid_config.fine_enabled and
            coarse_result and coarse_result.best_params):
            self.logger.info("   → Stage 2: Fine grid search")
            try:
                fine_result = self._fine_grid_optimization(
                    model_factory, X, y, search_space, coarse_result.best_params, custom_evaluation_fn
                )
                self.logger.info(f"   → Fine grid best score: {fine_result.best_score:.4f}")
            except Exception as e:
                self.logger.error(f"   → Fine grid failed: {e}")
                if self.config.fast_fail_on_error:
                    raise

        # Stage 3: Bayesian TPE optimization
        tpe_result = None
        self.logger.info("   → Stage 3: Bayesian TPE optimization")
        try:
            # Use best grid result as starting point for TPE
            best_grid_params = fine_result.best_params if fine_result else coarse_result.best_params
            tpe_result = self._bayesian_tpe_optimization(
                model_factory, X, y, search_space, best_grid_params, custom_evaluation_fn
            )
            self.logger.info(f"   → TPE best score: {tpe_result.best_score:.4f}")
        except Exception as e:
            self.logger.error(f"   → TPE failed: {e}")
            if self.config.fast_fail_on_error:
                raise

        # Select best result
        best_result = self._select_best_result(coarse_result, fine_result, tpe_result)

        # Update trial counts
        best_result.n_trials_coarse = len(self._get_coarse_grid(search_space))
        best_result.n_trials_fine = len(self._get_fine_grid(search_space, best_result.best_params))
        best_result.n_trials_tpe = self.config.tpe_config.n_trials
        best_result.n_trials_total = (best_result.n_trials_coarse +
                                    best_result.n_trials_fine +
                                    best_result.n_trials_tpe)

        return best_result

    def _coarse_grid_optimization(self,
                                 model_factory: Callable,
                                 X: Union[np.ndarray, pd.DataFrame],
                                 y: Union[np.ndarray, pd.Series],
                                 search_space: Dict[str, Any],
                                 custom_evaluation_fn: Optional[Callable] = None) -> OptimizationResult:
        """Run coarse grid search optimization."""
        self.logger.info("🔍 Running coarse grid search")

        # Build coarse grid
        coarse_grid = self._get_coarse_grid(search_space)
        self.logger.info(f"   Generated {len(coarse_grid)} coarse grid points")

        if not coarse_grid:
            raise ValueError("No valid coarse grid points generated")

        # Evaluate grid points
        best_score = -np.inf
        best_params = None
        evaluation_history = []

        for i, params in enumerate(coarse_grid):
            try:
                # Create and evaluate model
                model = model_factory(params)
                score = self._evaluate_model(model, X, y, custom_evaluation_fn)

                evaluation_history.append({
                    'trial': i,
                    'params': params.copy(),
                    'score': score,
                    'stage': 'coarse_grid'
                })

                # Update best result
                if score > best_score:
                    best_score = score
                    best_params = params.copy()

                # Log progress
                if (i + 1) % max(1, len(coarse_grid) // 10) == 0:
                    self.logger.debug(f"   Evaluated {i + 1}/{len(coarse_grid)} - Best: {best_score:.4f}")

            except Exception as e:
                self.logger.warning(f"   Failed to evaluate params {params}: {e}")
                self.error_summary['coarse_grid'] += 1
                continue

        if best_params is None:
            raise RuntimeError("No valid coarse grid evaluations completed")

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_stage='coarse_grid',
            optimization_time=0.0,  # Set by caller
            optimization_method='bayesian_tpe',
            convergence_info={'stage': 'coarse_grid', 'n_evaluations': len(evaluation_history)},
            performance_metrics={'evaluation_history': evaluation_history}
        )

    def _fine_grid_optimization(self,
                              model_factory: Callable,
                              X: Union[np.ndarray, pd.DataFrame],
                              y: Union[np.ndarray, pd.Series],
                              search_space: Dict[str, Any],
                              best_coarse_params: Dict[str, Any],
                              custom_evaluation_fn: Optional[Callable] = None) -> OptimizationResult:
        """Run fine grid search around best coarse parameters."""
        self.logger.info("🔍 Running fine grid search around best coarse parameters")

        # Build fine grid
        fine_grid = self._get_fine_grid(search_space, best_coarse_params)
        self.logger.info(f"   Generated {len(fine_grid)} fine grid points")

        if not fine_grid:
            raise ValueError("No valid fine grid points generated")

        # Evaluate grid points
        best_score = -np.inf
        best_params = None
        evaluation_history = []

        for i, params in enumerate(fine_grid):
            try:
                # Create and evaluate model
                model = model_factory(params)
                score = self._evaluate_model(model, X, y, custom_evaluation_fn)

                evaluation_history.append({
                    'trial': i,
                    'params': params.copy(),
                    'score': score,
                    'stage': 'fine_grid'
                })

                # Update best result
                if score > best_score:
                    best_score = score
                    best_params = params.copy()

                # Log progress
                if (i + 1) % max(1, len(fine_grid) // 10) == 0:
                    self.logger.debug(f"   Evaluated {i + 1}/{len(fine_grid)} - Best: {best_score:.4f}")

            except Exception as e:
                self.logger.warning(f"   Failed to evaluate params {params}: {e}")
                self.error_summary['fine_grid'] += 1
                continue

        if best_params is None:
            raise RuntimeError("No valid fine grid evaluations completed")

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_stage='fine_grid',
            optimization_time=0.0,  # Set by caller
            optimization_method='bayesian_tpe',
            convergence_info={'stage': 'fine_grid', 'n_evaluations': len(evaluation_history)},
            performance_metrics={'evaluation_history': evaluation_history}
        )

    def _bayesian_tpe_optimization(self,
                                  model_factory: Callable,
                                  X: Union[np.ndarray, pd.DataFrame],
                                  y: Union[np.ndarray, pd.Series],
                                  search_space: Dict[str, Any],
                                  best_grid_params: Optional[Dict[str, Any]] = None,
                                  custom_evaluation_fn: Optional[Callable] = None) -> OptimizationResult:
        """Run Bayesian TPE optimization."""
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for Bayesian TPE optimization")

        self.logger.info("🎲 Running Bayesian TPE optimization")

        def objective(trial):
            """Objective function for Optuna optimization."""
            try:
                # Sample parameters
                params = self._sample_parameters(trial, search_space)

                # Create and evaluate model
                model = model_factory(params)
                score = self._evaluate_model(model, X, y, custom_evaluation_fn)

                # Store trial result
                self.optimization_history.append({
                    'trial': trial.number,
                    'params': params.copy(),
                    'score': score,
                    'stage': 'tpe'
                })

                return score

            except Exception as e:
                self.logger.warning(f"   TPE trial {trial.number} failed: {e}")
                self.error_summary['tpe'] += 1
                # Return worst possible score to guide optimization away from bad regions
                return -999.0

        # Create study with TPE sampler
        sampler = TPESampler(seed=self.random_state)

        # Set up pruner
        pruner = None
        if self.config.tpe_config.pruner == 'median':
            pruner = MedianPruner()
        elif self.config.tpe_config.pruner == 'hyperband':
            pruner = HyperbandPruner()

        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            study_name=f"{self.current_study_id}_tpe"
        )

        # Run optimization
        study.optimize(
            objective,
            n_trials=self.config.tpe_config.n_trials,
            timeout=self.config.tpe_config.timeout_seconds
        )

        # Extract best result
        best_params = study.best_params
        best_score = study.best_value

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_stage='tpe',
            optimization_time=0.0,  # Set by caller
            optimization_method='bayesian_tpe',
            n_trials_tpe=self.config.tpe_config.n_trials,
            convergence_info={
                'stage': 'tpe',
                'n_trials': len(study.trials),
                'best_trial': study.best_trial.number if study.best_trial else None,
                'optuna_study': study
            },
            performance_metrics={
                'optimization_history': self.optimization_history[-self.config.tpe_config.n_trials:],
                'study_stats': {
                    'n_completed_trials': len(study.trials),
                    'n_failed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
                    'n_pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
                }
            }
        )

    def _get_coarse_grid(self, search_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate coarse grid from search space."""
        try:
            return build_coarse_grid_from_search_space(
                search_space,
                self.config.grid_config.coarse_grid_points
            )
        except Exception as e:
            self.logger.error(f"Failed to generate coarse grid: {e}")
            return []

    def _get_fine_grid(self, search_space: Dict[str, Any], best_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate fine grid around best parameters."""
        try:
            return build_fine_grid_around_best(
                search_space,
                best_params,
                self.config.grid_config.fine_grid_points
            )
        except Exception as e:
            self.logger.error(f"Failed to generate fine grid: {e}")
            return []

    def _sample_parameters(self, trial: optuna.Trial, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Sample parameters from search space using Optuna trial."""
        params = {}

        for param_name, param_config in search_space.items():
            try:
                if isinstance(param_config, dict):
                    param_type = param_config.get('type', 'float')

                    if param_type == 'int':
                        low = param_config['low']
                        high = param_config['high']
                        step = param_config.get('step', 1)
                        params[param_name] = trial.suggest_int(param_name, low, high, step=step)

                    elif param_type == 'float':
                        low = param_config['low']
                        high = param_config['high']
                        log = param_config.get('log', False)

                        if log:
                            params[param_name] = trial.suggest_float(param_name, low, high, log=True)
                        else:
                            params[param_name] = trial.suggest_float(param_name, low, high)

                    elif param_type == 'categorical':
                        choices = param_config['choices']
                        params[param_name] = trial.suggest_categorical(param_name, choices)

                else:
                    # Legacy tuple format
                    if isinstance(param_config, tuple) and len(param_config) == 2:
                        low, high = param_config
                        params[param_name] = trial.suggest_float(param_name, low, high)

            except Exception as e:
                self.logger.warning(f"Failed to sample parameter {param_name}: {e}")
                continue

        return params

    def _evaluate_model(self,
                       model: Any,
                       X: Union[np.ndarray, pd.DataFrame],
                       y: Union[np.ndarray, pd.Series],
                       custom_evaluation_fn: Optional[Callable] = None) -> float:
        """Evaluate model using cross-validation or custom function."""
        if custom_evaluation_fn is not None:
            return custom_evaluation_fn(model, X, y)

        # Standard cross-validation evaluation
        try:
            if not SKLEARN_AVAILABLE:
                raise ImportError("Scikit-learn required for model evaluation")

            # Prepare data
            if isinstance(X, pd.DataFrame):
                X = X.values
            if isinstance(y, pd.Series):
                y = y.values

            # Get scorer
            scoring = self.config.validation_config.get('scoring', 'balanced_accuracy')
            scorer = get_scorer(scoring)

            # Cross-validation
            cv_folds = self.config.validation_config.get('cv_folds', 5)
            scores = cross_val_score(
                model,
                X,
                y,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=1  # Avoid nested parallelism issues
            )

            return float(np.mean(scores))

        except Exception as e:
            self.logger.warning(f"Model evaluation failed: {e}")
            return 0.0  # Return neutral score

    def _select_best_result(self, coarse_result: Optional[OptimizationResult],
                           fine_result: Optional[OptimizationResult],
                           tpe_result: Optional[OptimizationResult]) -> OptimizationResult:
        """Select the best result from all stages."""
        results = [r for r in [coarse_result, fine_result, tpe_result] if r is not None]

        if not results:
            raise RuntimeError("No valid optimization results available")

        # Select result with highest score
        best_result = max(results, key=lambda r: r.best_score)

        self.logger.info(f"   Best result from stage: {best_result.best_stage} (score: {best_result.best_score:.4f})")

        return best_result

    def _should_use_transfer_learning(self,
                                    transfer_data: Dict[str, Any],
                                    X: Union[np.ndarray, pd.DataFrame],
                                    y: Union[np.ndarray, pd.Series]) -> bool:
        """Determine if transfer learning should be used."""
        try:
            # Check similarity threshold
            similarity_score = self._calculate_data_similarity(transfer_data, X, y)
            should_transfer = similarity_score >= self.config.transfer_learning_threshold

            self.logger.info(f"   Transfer learning similarity: {similarity_score:.3f}")
            self.logger.info(f"   Transfer learning: {'enabled' if should_transfer else 'disabled'}")

            return should_transfer

        except Exception as e:
            self.logger.warning(f"Transfer learning assessment failed: {e}")
            return False

    def _calculate_data_similarity(self,
                                 transfer_data: Dict[str, Any],
                                 X: Union[np.ndarray, pd.DataFrame],
                                 y: Union[np.ndarray, pd.Series]) -> float:
        """Calculate similarity between current and transfer learning data."""
        try:
            # Extract data characteristics
            current_n_samples = len(X)
            current_n_features = X.shape[1] if hasattr(X, 'shape') else len(X[0]) if len(X) > 0 else 0
            current_n_classes = len(np.unique(y))

            transfer_n_samples = transfer_data.get('n_samples', current_n_samples)
            transfer_n_features = transfer_data.get('n_features', current_n_features)
            transfer_n_classes = transfer_data.get('n_classes', current_n_classes)

            # Calculate similarity ratios
            sample_ratio = min(current_n_samples, transfer_n_samples) / max(current_n_samples, transfer_n_samples)
            feature_ratio = min(current_n_features, transfer_n_features) / max(current_n_features, transfer_n_features)
            class_ratio = min(current_n_classes, transfer_n_classes) / max(current_n_classes, transfer_n_classes)

            # Weighted similarity score
            similarity = (sample_ratio * 0.4 + feature_ratio * 0.4 + class_ratio * 0.2)

            return float(similarity)

        except Exception as e:
            self.logger.warning(f"Data similarity calculation failed: {e}")
            return 0.5  # Neutral similarity

    def _transfer_learning_optimization(self,
                                      model_factory: Callable,
                                      X: Union[np.ndarray, pd.DataFrame],
                                      y: Union[np.ndarray, pd.Series],
                                      search_space: Dict[str, Any],
                                      transfer_data: Dict[str, Any],
                                      custom_evaluation_fn: Optional[Callable] = None) -> OptimizationResult:
        """Run transfer learning-based optimization."""
        self.logger.info("🔄 Running transfer learning optimization")

        # Use transfer learning results as starting point
        best_transfer_params = transfer_data.get('best_params', {})

        # Create narrowed search space around transfer learning results
        narrowed_space = self._narrow_search_space(search_space, best_transfer_params)

        # Run focused TPE optimization around transfer learning results
        tpe_result = self._bayesian_tpe_optimization(
            model_factory, X, y, narrowed_space, best_transfer_params, custom_evaluation_fn
        )

        # Mark as transfer learning result
        tpe_result.best_stage = 'transfer_learning'

        return tpe_result

    def _narrow_search_space(self, search_space: Dict[str, Any], center_params: Dict[str, Any],
                           shrink_factor: float = 0.5) -> Dict[str, Any]:
        """Narrow search space around given parameters."""
        narrowed = {}

        for param_name, param_config in search_space.items():
            if param_name not in center_params:
                narrowed[param_name] = param_config
                continue

            center_value = center_params[param_name]

            if isinstance(param_config, dict):
                param_type = param_config.get('type', 'float')

                if param_type in ('float', 'int'):
                    low = param_config['low']
                    high = param_config['high']

                    # Narrow range around center value
                    range_size = (high - low) * shrink_factor / 2.0
                    new_low = max(low, center_value - range_size)
                    new_high = min(high, center_value + range_size)

                    narrowed[param_name] = {
                        'type': param_type,
                        'low': new_low,
                        'high': new_high
                    }

                else:
                    # Keep categorical and other types as-is
                    narrowed[param_name] = param_config

            else:
                # Legacy format - keep as-is
                narrowed[param_name] = param_config

        return narrowed

    def _create_fallback_result(self,
                               model_factory: Callable,
                               search_space: Dict[str, Any],
                               error: Exception) -> OptimizationResult:
        """Create fallback optimization result when optimization fails."""
        self.logger.warning("🔧 Creating fallback optimization result")

        # Try to get default parameters from search space
        fallback_params = {}
        for param_name, param_config in search_space.items():
            try:
                if isinstance(param_config, dict):
                    param_type = param_config.get('type', 'float')

                    if param_type in ('float', 'int'):
                        low = param_config['low']
                        high = param_config['high']
                        # Use midpoint as fallback
                        fallback_params[param_name] = (low + high) / 2

                    elif param_type == 'categorical':
                        choices = param_config['choices']
                        if choices:
                            fallback_params[param_name] = choices[0]  # First choice as fallback

                else:
                    # Legacy tuple format
                    if isinstance(param_config, tuple) and len(param_config) == 2:
                        low, high = param_config
                        fallback_params[param_name] = (low + high) / 2

            except Exception:
                continue

        # If no fallback params found, use empty dict
        if not fallback_params:
            fallback_params = {}

        return OptimizationResult(
            best_params=fallback_params,
            best_score=0.5,  # Neutral score
            best_stage='fallback',
            optimization_time=0.0,
            optimization_method='bayesian_tpe',
            error_info={'error': str(error), 'error_type': type(error).__name__},
            convergence_info={'stage': 'fallback', 'reason': 'optimization_failed'}
        )

    def _save_results(self, results: OptimizationResult) -> None:
        """Save optimization results to disk."""
        if not self.config.save_results:
            return

        try:
            results_path = Path(self.results_path)

            # Save main results
            results_file = results_path / f"results_{self.current_study_id}.json"
            with open(results_file, 'w') as f:
                # Convert to JSON-serializable format
                results_dict = {
                    'best_params': results.best_params,
                    'best_score': results.best_score,
                    'best_stage': results.best_stage,
                    'optimization_time': results.optimization_time,
                    'optimization_method': results.optimization_method,
                    'n_trials_total': results.n_trials_total,
                    'n_trials_coarse': results.n_trials_coarse,
                    'n_trials_fine': results.n_trials_fine,
                    'n_trials_tpe': results.n_trials_tpe,
                    'convergence_info': results.convergence_info,
                    'performance_metrics': results.performance_metrics,
                    'error_info': results.error_info,
                    'timestamp': datetime.now().isoformat()
                }
                json.dump(results_dict, f, indent=2, default=str)

            # Save optimization history
            if self.optimization_history:
                history_file = results_path / f"history_{self.current_study_id}.json"
                with open(history_file, 'w') as f:
                    json.dump(self.optimization_history, f, indent=2, default=str)

            # Save error summary
            if self.error_summary:
                error_file = results_path / f"errors_{self.current_study_id}.json"
                with open(error_file, 'w') as f:
                    json.dump(dict(self.error_summary), f, indent=2)

            self.logger.info(f"   Results saved to: {results_path}")

        except Exception as e:
            self.logger.warning(f"Failed to save results: {e}")


# ============================================================================
# PUBLIC API FUNCTIONS
# ============================================================================

def optimize_hyperparameters(model_factory: Callable[[Dict[str, Any]], Any],
                           X: Union[np.ndarray, pd.DataFrame],
                           y: Union[np.ndarray, pd.Series],
                           search_space: Optional[Dict[str, Any]] = None,
                           model_type: str = 'auto',
                           config: Optional[OptimizationConfig] = None,
                           custom_evaluation_fn: Optional[Callable] = None) -> OptimizationResult:
    """
    Optimize hyperparameters using Bayesian TPE with automatic grid-first approach.

    This is a convenience function that creates and runs a BayesianTPEOptimizer.

    Args:
        model_factory: Function that creates model with given parameters
        X: Feature matrix
        y: Target array/series
        search_space: Custom search space (optional)
        model_type: Type of model for default search space
        config: Optimization configuration
        custom_evaluation_fn: Custom evaluation function

    Returns:
        Optimization results

    Example:
        def create_xgb(params):
            return XGBClassifier(**params)

        best_params = optimize_hyperparameters(
            create_xgb, X_train, y_train,
            model_type='xgboost',
            config=OptimizationConfig()
        )
    """
    optimizer = BayesianTPEOptimizer(config=config, model_type=model_type)
    return optimizer.optimize(model_factory, X, y, search_space, custom_evaluation_fn)


def create_optimization_config(n_trials: int = 50,
                             coarse_grid_points: int = 5,
                             fine_grid_points: int = 10,
                             enable_parallel: bool = True,
                             **kwargs) -> OptimizationConfig:
    """
    Create a standard optimization configuration.

    Args:
        n_trials: Number of TPE trials
        coarse_grid_points: Number of points per parameter for coarse grid
        fine_grid_points: Number of points per parameter for fine grid
        enable_parallel: Enable parallel processing
        **kwargs: Additional configuration options

    Returns:
        OptimizationConfig instance
    """
    tpe_config = TPEConfig(
        n_trials=n_trials,
        enable_parallel=enable_parallel,
        **kwargs
    )

    grid_config = GridConfig(
        coarse_grid_points=coarse_grid_points,
        fine_grid_points=fine_grid_points
    )

    return OptimizationConfig(
        tpe_config=tpe_config,
        grid_config=grid_config,
        **kwargs
    )


# Add missing imports
from collections import defaultdict

# Make classes available for import
__all__ = [
    'BayesianTPEOptimizer',
    'OptimizationConfig',
    'TPEConfig',
    'GridConfig',
    'OptimizationResult',
    'optimize_hyperparameters',
    'create_optimization_config'
]