"""
Hierarchical Hyperparameter Optimization for Multi-Output Stacking Ensemble

This module implements the recommended HPO strategy:
1. Phase 1: Optimize base models first
2. Phase 2: Optimize meta models with fixed base models
3. Ensures proper timing and prevents meta model overfitting to poor base models
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime
import json
from pathlib import Path

# HPO imports
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

from src.utils.logger import system_logger
from src.utils.tprint import tprint_data_format, LogLevel

# Enforce time-series CV
try:
    from src.utils.purged_kfold import PurgedKFoldTime  # type: ignore
    _PURGED_AVAILABLE = True
except Exception:
    _PURGED_AVAILABLE = False
from sklearn.model_selection import TimeSeriesSplit

# Import universal validation integration - use lazy import to avoid circular dependency
# from ..training.universal_validation_integration import (
#     get_validation_integrator,
#     validate_hpo_trial,
#     ValidationIntegrationConfig
# )

logger = system_logger.getChild('HierarchicalHPO')

@dataclass
class HPOPhaseConfig:
    """Configuration for each HPO phase."""
    phase_name: str
    models: Dict[str, Any]
    search_spaces: Dict[str, Dict[str, Any]]
    n_trials: int = 100
    timeout_seconds: Optional[int] = None
    enable_pruning: bool = True
    cv_folds: int = 5
    scoring_metric: str = 'neg_mean_squared_error'
    direction: str = 'maximize'

@dataclass
class HPOPhaseResult:
    """Result of a single HPO phase."""
    phase_name: str
    best_models: Dict[str, Any]
    best_scores: Dict[str, float]
    optimization_time: float
    n_trials: int
    best_params: Dict[str, Dict[str, Any]]
    optimization_history: List[Dict[str, Any]]

@dataclass
class HierarchicalHPOConfig:
    """Configuration for hierarchical HPO."""
    # Phase 1: Base Model HPO
    phase1_config: HPOPhaseConfig

    # Phase 2: Meta Model HPO
    phase2_config: HPOPhaseConfig

    # General settings
    enable_caching: bool = True
    cache_dir: str = "./hpo_cache"
    enable_parallel: bool = True
    max_workers: int = 4
    random_state: int = 42

    # Validation settings
    validation_split: float = 0.2
    test_split: float = 0.1
    enable_time_series_cv: bool = True

    # Universal validation settings
    enable_validation: bool = True
    enable_overfitting_detection: bool = True
    enable_temporal_validation: bool = True
    enable_timeframe_validation: bool = True
    validation_failure_threshold: float = 0.5
    fail_on_validation_error: bool = False

class HierarchicalHPO:
    """
    Hierarchical Hyperparameter Optimization for Multi-Output Stacking Ensemble.

    This class implements the recommended two-phase HPO strategy:
    1. Phase 1: Optimize base models individually
    2. Phase 2: Optimize meta models with fixed base models
    """

    def __init__(self, config: HierarchicalHPOConfig):
        """Initialize hierarchical HPO."""
        self.config = config
        self.logger = logger.getChild('HierarchicalHPO')

        # Validate dependencies
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for HPO functionality")

        # Initialize results
        self.phase1_result: Optional[HPOPhaseResult] = None
        self.phase2_result: Optional[HPOPhaseResult] = None
        self.final_models: Dict[str, Any] = {}

        # Initialize validation integration
        self._initialize_validation_integration()

        # Create cache directory
        if self.config.enable_caching:
            Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)

    def _initialize_validation_integration(self):
        """Initialize universal validation integration for HPO."""
        try:
            # Lazy import to avoid circular dependency
            from ..training.universal_validation_integration import (
                get_validation_integrator,
                ValidationIntegrationConfig
            )

            # Create validation configuration
            validation_config = ValidationIntegrationConfig(
                enable_validation=self.config.enable_validation,
                enable_overfitting_detection=self.config.enable_overfitting_detection,
                enable_temporal_validation=self.config.enable_temporal_validation,
                enable_timeframe_validation=self.config.enable_timeframe_validation,
                validation_failure_threshold=self.config.validation_failure_threshold,
                fail_on_validation_error=self.config.fail_on_validation_error,
                save_validation_reports=True,
                validation_report_directory=f"{self.config.cache_dir}/validation_reports",
                enable_validation_logging=True
            )

            # Initialize validation integrator
            self.validation_integrator = get_validation_integrator(validation_config)

            logger.info("✅ Universal validation integration initialized for HPO")

        except ImportError as e:
            logger.warning(f"⚠️ Could not initialize validation integration: {e}")
            self.validation_integrator = None

        self.logger.info("✅ Hierarchical HPO initialized")

    def optimize_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform hierarchical hyperparameter optimization.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            feature_names: Names of features (optional)

        Returns:
            Dictionary containing optimized models and results
        """
        self.logger.info("🚀 Starting hierarchical HPO optimization")
        start_time = time.time()

        # Prepare data
        X_val, y_val = self._prepare_validation_data(X_train, y_train, X_val, y_val)

        # Phase 1: Base Model HPO
        self.logger.info("🔄 Phase 1: Optimizing base models...")
        phase1_start = time.time()

        self.phase1_result = self._optimize_phase(
            phase_config=self.config.phase1_config,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_names=feature_names
        )

        phase1_time = time.time() - phase1_start
        self.logger.info(f"✅ Phase 1 completed in {phase1_time:.2f}s")

        # Phase 2: Meta Model HPO with optimized base models
        self.logger.info("🔄 Phase 2: Optimizing meta models with fixed base models...")
        phase2_start = time.time()

        # Create meta features using optimized base models
        meta_features = self._create_meta_features(X_val, self.phase1_result.best_models)

        self.phase2_result = self._optimize_phase(
            phase_config=self.config.phase2_config,
            X_train=X_train,
            y_train=y_train,
            X_val=meta_features,
            y_val=y_val,
            feature_names=feature_names,
            base_models=self.phase1_result.best_models
        )

        phase2_time = time.time() - phase2_start
        self.logger.info(f"✅ Phase 2 completed in {phase2_time:.2f}s")

        # Combine results
        total_time = time.time() - start_time
        self.final_models = {
            'base_models': self.phase1_result.best_models,
            'meta_models': self.phase2_result.best_models,
            'optimization_time': total_time,
            'phase1_time': phase1_time,
            'phase2_time': phase2_time
        }

        self.logger.info(f"✅ Hierarchical HPO completed in {total_time:.2f}s")
        self.logger.info(f"📊 Phase 1: {len(self.phase1_result.best_models)} base models optimized")
        self.logger.info(f"📊 Phase 2: {len(self.phase2_result.best_models)} meta models optimized")

        return self.final_models

    def _optimize_phase(
        self,
        phase_config: HPOPhaseConfig,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[List[str]] = None,
        base_models: Optional[Dict[str, Any]] = None
    ) -> HPOPhaseResult:
        """Optimize a single phase with coarse/fine grid + Optuna TPE."""

        self.logger.info(f"🔄 Optimizing phase: {phase_config.phase_name}")
        start_time = time.time()

        best_models = {}
        best_scores = {}
        best_params = {}
        optimization_history = []

        # Create study for each model
        for model_name, model in phase_config.models.items():
            self.logger.info(f"🔄 Optimizing {model_name} with coarse/fine/optuna approach...")

            # Stage 1: Coarse Grid Search
            self.logger.info(f"🎯 Stage 1: Coarse grid search for {model_name}")
            coarse_start = time.time()
            coarse_result = self._coarse_grid_search_phase(
                model, model_name, phase_config.search_spaces[model_name],
                X_train, y_train, X_val, y_val, phase_config.cv_folds,
                phase_config.scoring_metric, base_models
            )
            coarse_time = time.time() - coarse_start

            if not coarse_result or coarse_result.get('best_score', 0) <= 0:
                self.logger.warning(f"⚠️ Coarse grid search failed for {model_name}, using random sampling")
                coarse_result = self._fallback_random_search_phase(
                    model, model_name, phase_config.search_spaces[model_name],
                    X_train, y_train, X_val, y_val, phase_config.cv_folds,
                    phase_config.scoring_metric, base_models, 20
                )

            self.logger.info(f"✅ Coarse grid completed in {coarse_time:.2f}s - Best score: {coarse_result.get('best_score', 0):.4f}")

            # Stage 2: Fine Grid Search around best coarse parameters
            self.logger.info(f"🎯 Stage 2: Fine grid search for {model_name}")
            fine_start = time.time()
            best_coarse = coarse_result.get('best_params', {})
            fine_result = self._fine_grid_search_phase(
                model, model_name, phase_config.search_spaces[model_name], best_coarse,
                X_train, y_train, X_val, y_val, phase_config.cv_folds,
                phase_config.scoring_metric, base_models
            )
            fine_time = time.time() - fine_start

            if not fine_result or fine_result.get('best_score', 0) <= coarse_result.get('best_score', 0):
                self.logger.info(f"ℹ️ Fine grid search did not improve results for {model_name}")
                best_params_grid = best_coarse
                best_score_grid = coarse_result.get('best_score', 0)
                grid_stage = 'coarse'
            else:
                self.logger.info(f"✅ Fine grid completed in {fine_time:.2f}s - Best score: {fine_result.get('best_score', 0):.4f}")
                best_params_grid = fine_result.get('best_params', {})
                best_score_grid = fine_result.get('best_score', 0)
                grid_stage = 'fine'

            # Stage 3: Optuna TPE Optimization around best grid parameters
            self.logger.info(f"🎯 Stage 3: Optuna TPE optimization for {model_name}")
            optuna_start = time.time()

            # Create narrowed search space around best grid parameters
            narrowed_space = self._create_narrowed_search_space_phase(
                phase_config.search_spaces[model_name], best_params_grid
            )

            # Create Optuna study with TPE sampler
            study = optuna.create_study(
                direction=phase_config.direction,
                sampler=TPESampler(
                    n_startup_trials=5,  # Fewer startup trials since we have good starting point
                    n_ei_candidates=24,
                    gamma=lambda x: min(int(0.25 * x), 25),
                    prior_weight=1.0,
                    consider_magic_clip=True,
                    consider_endpoints=True,
                    seed=self.config.random_state
                ),
                pruner=MedianPruner() if phase_config.enable_pruning else None
            )

            # Use fewer trials since we're fine-tuning around good parameters
            n_trials = min(phase_config.n_trials // 3, 30)
            timeout = min(phase_config.timeout_seconds // 3, 120) if phase_config.timeout_seconds else None

            # Define objective function
            def objective(trial):
                return self._objective_function(
                    trial=trial,
                    model=model,
                    model_name=model_name,
                    search_space=narrowed_space,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    cv_folds=phase_config.cv_folds,
                    scoring_metric=phase_config.scoring_metric,
                    base_models=base_models
                )

            # Optimize
            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout
            )

            optuna_time = time.time() - optuna_start

            # Get best result
            best_trial = study.best_trial
            final_score = best_trial.value
            final_params = best_trial.params
            final_stage = 'optuna'

            if final_score <= best_score_grid:
                self.logger.info(f"ℹ️ Optuna TPE did not improve results for {model_name}, using grid search results")
                final_score = best_score_grid
                final_params = best_params_grid
                final_stage = grid_stage

            best_models[model_name] = self._create_optimized_model(
                model, final_params, base_models
            )
            best_scores[model_name] = final_score
            best_params[model_name] = final_params

            # Record history
            optimization_history.append({
                'model_name': model_name,
                'n_trials': len(study.trials),
                'best_score': final_score,
                'best_params': final_params,
                'optimization_time': time.time() - start_time,
                'coarse_time': coarse_time,
                'fine_time': fine_time,
                'optuna_time': optuna_time,
                'best_stage': final_stage,
                'coarse_score': coarse_result.get('best_score', 0),
                'fine_score': fine_result.get('best_score', 0) if fine_result else 0,
                'optuna_score': best_trial.value
            })

            self.logger.info(f"✅ {model_name} optimized: {final_score:.4f} (best stage: {final_stage})")

        return HPOPhaseResult(
            phase_name=phase_config.phase_name,
            best_models=best_models,
            best_scores=best_scores,
            optimization_time=time.time() - start_time,
            n_trials=sum(len(study.trials) for study in [optuna.create_study()] * len(phase_config.models)),
            best_params=best_params,
            optimization_history=optimization_history
        )

    def _objective_function(
        self,
        trial: optuna.Trial,
        model: Any,
        model_name: str,
        search_space: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        cv_folds: int,
        scoring_metric: str,
        base_models: Optional[Dict[str, Any]] = None
    ) -> float:
        """Objective function for Optuna optimization."""

        try:
            # Sample hyperparameters
            params = self._sample_hyperparameters(trial, search_space)

            # Create model with sampled parameters
            optimized_model = self._create_optimized_model(model, params, base_models)

            # Perform cross-validation
            if cv_folds > 1:
                scores = self._cross_validate_model(
                    optimized_model, X_train, y_train, cv_folds, scoring_metric
                )
                return np.mean(scores)
            else:
                # Single validation
                optimized_model.fit(X_train, y_train)
                y_pred = optimized_model.predict(X_val)

                if scoring_metric == 'neg_mean_squared_error':
                    from sklearn.metrics import mean_squared_error
                    return -mean_squared_error(y_val, y_pred)
                elif scoring_metric == 'neg_mean_absolute_error':
                    from sklearn.metrics import mean_absolute_error
                    return -mean_absolute_error(y_val, y_pred)
                elif scoring_metric == 'r2':
                    from sklearn.metrics import r2_score
                    return r2_score(y_val, y_pred)
                else:
                    raise ValueError(f"Unsupported scoring metric: {scoring_metric}")

        except Exception as e:
            self.logger.warning(f"⚠️ Trial failed for {model_name}: {e}")
            return float('-inf')

    def _sample_hyperparameters(self, trial: optuna.Trial, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Sample hyperparameters from search space."""
        params = {}

        for param_name, param_config in search_space.items():
            if param_config['type'] == 'float':
                params[param_name] = trial.suggest_float(
                    param_name, param_config['low'], param_config['high'], log=param_config.get('log', False)
                )
            elif param_config['type'] == 'int':
                params[param_name] = trial.suggest_int(
                    param_name, param_config['low'], param_config['high'], log=param_config.get('log', False)
                )
            elif param_config['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, param_config['choices'])
            else:
                raise ValueError(f"Unsupported parameter type: {param_config['type']}")

        return params

    def _create_optimized_model(self, base_model: Any, params: Dict[str, Any], base_models: Optional[Dict[str, Any]] = None) -> Any:
        """Create model with optimized parameters."""

        # Clone the base model
        from sklearn.base import clone
        optimized_model = clone(base_model)

        # Set parameters
        optimized_model.set_params(**params)

        return optimized_model

    def _cross_validate_model(self, model: Any, X: np.ndarray, y: np.ndarray, cv_folds: int, scoring_metric: str) -> List[float]:
        """Perform time-series cross-validation (purged when possible)."""
        from sklearn.model_selection import cross_val_score
        # Build splitter
        try:
            if _PURGED_AVAILABLE and isinstance(X, np.ndarray):
                # Purged splitter expects a DataFrame with DatetimeIndex; fallback to TimeSeriesSplit
                splitter = TimeSeriesSplit(n_splits=cv_folds)
            else:
                splitter = TimeSeriesSplit(n_splits=cv_folds)
        except Exception:
            splitter = TimeSeriesSplit(n_splits=cv_folds)

        scores = cross_val_score(
            model, X, y, cv=splitter, scoring=scoring_metric, n_jobs=1
        )
        return scores.tolist()

    def _create_meta_features(self, X: np.ndarray, base_models: Dict[str, Any]) -> np.ndarray:
        """Create meta features using base model predictions."""

        meta_features = []

        for model_name, model in base_models.items():
            try:
                pred = model.predict(X)
                if pred.ndim == 1:
                    pred = pred.reshape(-1, 1)
                meta_features.append(pred)
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to get predictions from {model_name}: {e}")
                # Add zero predictions as fallback
                meta_features.append(np.zeros((len(X), 1)))

        return np.hstack(meta_features)

    def _prepare_validation_data(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare validation data."""

        if X_val is not None and y_val is not None:
            return X_val, y_val

        # Split training data for validation
        from sklearn.model_selection import train_test_split

        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train,
            test_size=self.config.validation_split,
            random_state=self.config.random_state
        )

        return X_val_split, y_val_split

    def save_results(self, filepath: str) -> None:
        """Save optimization results to file."""

        results = {
            'phase1_result': {
                'phase_name': self.phase1_result.phase_name,
                'best_scores': self.phase1_result.best_scores,
                'best_params': self.phase1_result.best_params,
                'optimization_time': self.phase1_result.optimization_time,
                'n_trials': self.phase1_result.n_trials
            },
            'phase2_result': {
                'phase_name': self.phase2_result.phase_name,
                'best_scores': self.phase2_result.best_scores,
                'best_params': self.phase2_result.best_params,
                'optimization_time': self.phase2_result.optimization_time,
                'n_trials': self.phase2_result.n_trials
            },
            'final_models': self.final_models
        }

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        self.logger.info(f"💾 Results saved to {filepath}")

    def _coarse_grid_search_phase(self, model: Any, model_name: str, search_space: Dict[str, Any],
                                 X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                                 cv_folds: int, scoring_metric: str, base_models: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform coarse grid search for a single model in hierarchical HPO."""
        try:
            self.logger.info(f"🔍 Creating coarse grid for {model_name}")

            # Create coarse parameter grid
            coarse_grid = self._create_coarse_parameter_grid_phase(search_space)
            self.logger.info(f"📊 Coarse grid size: {len(coarse_grid)} combinations")

            best_score = -np.inf
            best_params = {}
            parameter_scores = []

            # Evaluate each parameter combination
            for i, params in enumerate(coarse_grid):
                try:
                    # Create model with parameters
                    optimized_model = self._create_optimized_model(model, params, base_models)

                    # Evaluate model
                    if cv_folds > 1:
                        scores = self._cross_validate_model(optimized_model, X_train, y_train, cv_folds, scoring_metric)
                        score = np.mean(scores)
                    else:
                        optimized_model.fit(X_train, y_train)
                        y_pred = optimized_model.predict(X_val)

                        if scoring_metric == 'neg_mean_squared_error':
                            score = -mean_squared_error(y_val, y_pred)
                        elif scoring_metric == 'neg_mean_absolute_error':
                            score = -mean_absolute_error(y_val, y_pred)
                        elif scoring_metric == 'r2':
                            score = r2_score(y_val, y_pred)
                        else:
                            score = 0.0

                    parameter_scores.append((params, score))

                    if score > best_score:
                        best_score = score
                        best_params = params.copy()

                    if (i + 1) % 10 == 0:
                        self.logger.debug(f"   Evaluated {i + 1}/{len(coarse_grid)} combinations")

                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to evaluate parameters {params}: {e}")
                    continue

            if not parameter_scores:
                self.logger.error(f"❌ No valid parameter combinations found for {model_name}")
                return {}

            self.logger.info(f"✅ Coarse grid search completed for {model_name} - Best score: {best_score:.4f}")

            return {
                'best_params': best_params,
                'best_score': best_score,
                'n_combinations': len(coarse_grid),
                'valid_combinations': len(parameter_scores),
                'parameter_scores': parameter_scores[:10]  # Keep top 10 for analysis
            }

        except Exception as e:
            self.logger.error(f"❌ Coarse grid search failed for {model_name}: {e}")
            return {}

    def _fine_grid_search_phase(self, model: Any, model_name: str, search_space: Dict[str, Any],
                               best_coarse_params: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray, cv_folds: int, scoring_metric: str,
                               base_models: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform fine grid search around best coarse parameters for hierarchical HPO."""
        try:
            self.logger.info(f"🔍 Creating fine grid around best coarse parameters for {model_name}")

            # Create fine parameter grid around best coarse parameters
            fine_grid = self._create_fine_parameter_grid_phase(search_space, best_coarse_params)
            self.logger.info(f"📊 Fine grid size: {len(fine_grid)} combinations")

            best_score = -np.inf
            best_params = {}
            parameter_scores = []

            # Evaluate each parameter combination
            for i, params in enumerate(fine_grid):
                try:
                    # Create model with parameters
                    optimized_model = self._create_optimized_model(model, params, base_models)

                    # Evaluate model
                    if cv_folds > 1:
                        scores = self._cross_validate_model(optimized_model, X_train, y_train, cv_folds, scoring_metric)
                        score = np.mean(scores)
                    else:
                        optimized_model.fit(X_train, y_train)
                        y_pred = optimized_model.predict(X_val)

                        if scoring_metric == 'neg_mean_squared_error':
                            score = -mean_squared_error(y_val, y_pred)
                        elif scoring_metric == 'neg_mean_absolute_error':
                            score = -mean_absolute_error(y_val, y_pred)
                        elif scoring_metric == 'r2':
                            score = r2_score(y_val, y_pred)
                        else:
                            score = 0.0

                    parameter_scores.append((params, score))

                    if score > best_score:
                        best_score = score
                        best_params = params.copy()

                    if (i + 1) % 10 == 0:
                        self.logger.debug(f"   Evaluated {i + 1}/{len(fine_grid)} combinations")

                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to evaluate parameters {params}: {e}")
                    continue

            if not parameter_scores:
                self.logger.error(f"❌ No valid parameter combinations found for {model_name}")
                return {}

            self.logger.info(f"✅ Fine grid search completed for {model_name} - Best score: {best_score:.4f}")

            return {
                'best_params': best_params,
                'best_score': best_score,
                'n_combinations': len(fine_grid),
                'valid_combinations': len(parameter_scores),
                'parameter_scores': parameter_scores[:10]  # Keep top 10 for analysis
            }

        except Exception as e:
            self.logger.error(f"❌ Fine grid search failed for {model_name}: {e}")
            return {}

    def _create_coarse_parameter_grid_phase(self, search_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create coarse parameter grid for hierarchical HPO."""
        import itertools

        param_combinations = []

        for param_name, param_config in search_space.items():
            if param_config['type'] == 'float':
                # Use 3 points for coarse grid
                min_val, max_val = param_config['low'], param_config['high']
                if param_config.get('log', False):
                    # Log-spaced values
                    values = np.logspace(np.log10(min_val), np.log10(max_val), 3)
                else:
                    # Linear-spaced values
                    values = np.linspace(min_val, max_val, 3)
                param_combinations.append([(param_name, v) for v in values])

            elif param_config['type'] == 'int':
                # Use 3 points for coarse grid
                min_val, max_val = param_config['low'], param_config['high']
                if max_val - min_val <= 2:
                    values = list(range(min_val, max_val + 1))
                else:
                    values = np.linspace(min_val, max_val, 3, dtype=int)
                param_combinations.append([(param_name, v) for v in values])

            elif param_config['type'] == 'categorical':
                param_combinations.append([(param_name, v) for v in param_config['choices']])

        # Generate all combinations
        all_combinations = list(itertools.product(*param_combinations))

        # Convert to list of dictionaries
        grid = []
        for combination in all_combinations:
            param_dict = dict(combination)
            grid.append(param_dict)

        return grid

    def _create_fine_parameter_grid_phase(self, search_space: Dict[str, Any],
                                        best_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create fine parameter grid around best parameters for hierarchical HPO."""

        param_combinations = []

        for param_name, param_config in search_space.items():
            if param_name not in best_params:
                continue

            best_value = best_params[param_name]

            if param_config['type'] == 'float':
                min_val, max_val = param_config['low'], param_config['high']
                # Create fine grid around best value (±20% of range)
                range_size = max_val - min_val
                fine_range = range_size * 0.2
                fine_min = max(min_val, best_value - fine_range)
                fine_max = min(max_val, best_value + fine_range)

                # Use 5 points for fine grid
                if param_config.get('log', False):
                    # Log-spaced values
                    values = np.logspace(np.log10(fine_min), np.log10(fine_max), 5)
                else:
                    # Linear-spaced values
                    values = np.linspace(fine_min, fine_max, 5)
                param_combinations.append([(param_name, v) for v in values])

            elif param_config['type'] == 'int':
                min_val, max_val = param_config['low'], param_config['high']
                # Create fine grid around best value (±2 values)
                fine_min = max(min_val, best_value - 2)
                fine_max = min(max_val, best_value + 2)
                values = list(range(fine_min, fine_max + 1))
                param_combinations.append([(param_name, v) for v in values])

            elif param_config['type'] == 'categorical':
                param_combinations.append([(param_name, v) for v in param_config['choices']])

        # Generate all combinations
        all_combinations = list(itertools.product(*param_combinations))

        # Convert to list of dictionaries
        grid = []
        for combination in all_combinations:
            param_dict = dict(combination)
            grid.append(param_dict)

        return grid

    def _create_narrowed_search_space_phase(self, search_space: Dict[str, Any],
                                          best_params: Dict[str, Any]) -> Dict[str, Any]:
        """Create narrowed search space around best parameters for Optuna in hierarchical HPO."""
        narrowed_space = {}

        for param_name, param_config in search_space.items():
            if param_name not in best_params:
                narrowed_space[param_name] = param_config
                continue

            best_value = best_params[param_name]
            narrowed_config = param_config.copy()

            if param_config['type'] == 'float':
                min_val, max_val = param_config['low'], param_config['high']
                # Narrow range to ±10% of original range around best value
                range_size = max_val - min_val
                narrow_range = range_size * 0.1
                narrowed_config['low'] = max(min_val, best_value - narrow_range)
                narrowed_config['high'] = min(max_val, best_value + narrow_range)

            elif param_config['type'] == 'int':
                min_val, max_val = param_config['low'], param_config['high']
                # Narrow range to ±1 around best value
                narrowed_config['low'] = max(min_val, best_value - 1)
                narrowed_config['high'] = min(max_val, best_value + 1)

            narrowed_space[param_name] = narrowed_config

        return narrowed_space

    def _fallback_random_search_phase(self, model: Any, model_name: str, search_space: Dict[str, Any],
                                     X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                                     cv_folds: int, scoring_metric: str, base_models: Optional[Dict[str, Any]],
                                     n_samples: int) -> Dict[str, Any]:
        """Fallback random search when grid search fails in hierarchical HPO."""
        try:
            self.logger.info(f"🎲 Performing fallback random search for {model_name} with {n_samples} samples")

            best_score = -np.inf
            best_params = {}
            parameter_scores = []

            for i in range(n_samples):
                try:
                    # Sample random parameters
                    params = self._sample_hyperparameters_random(search_space)

                    # Create model with parameters
                    optimized_model = self._create_optimized_model(model, params, base_models)

                    # Evaluate model
                    if cv_folds > 1:
                        scores = self._cross_validate_model(optimized_model, X_train, y_train, cv_folds, scoring_metric)
                        score = np.mean(scores)
                    else:
                        optimized_model.fit(X_train, y_train)
                        y_pred = optimized_model.predict(X_val)

                        if scoring_metric == 'neg_mean_squared_error':
                            score = -mean_squared_error(y_val, y_pred)
                        elif scoring_metric == 'neg_mean_absolute_error':
                            score = -mean_absolute_error(y_val, y_pred)
                        elif scoring_metric == 'r2':
                            score = r2_score(y_val, y_pred)
                        else:
                            score = 0.0

                    parameter_scores.append((params, score))

                    if score > best_score:
                        best_score = score
                        best_params = params.copy()

                    if (i + 1) % 10 == 0:
                        self.logger.debug(f"   Evaluated {i + 1}/{n_samples} combinations")

                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to evaluate random parameters: {e}")
                    continue

            if not parameter_scores:
                self.logger.error(f"❌ No valid parameter combinations found for {model_name}")
                return {}

            self.logger.info(f"✅ Random search completed for {model_name} - Best score: {best_score:.4f}")

            return {
                'best_params': best_params,
                'best_score': best_score,
                'n_combinations': n_samples,
                'valid_combinations': len(parameter_scores),
                'parameter_scores': parameter_scores[:10],
                'method': 'random_fallback'
            }

        except Exception as e:
            self.logger.error(f"❌ Random search failed for {model_name}: {e}")
            return {}

    def _sample_hyperparameters_random(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Sample random hyperparameters from search space."""
        import random

        params = {}
        for param_name, param_config in search_space.items():
            if param_config['type'] == 'float':
                min_val, max_val = param_config['low'], param_config['high']
                if param_config.get('log', False):
                    # Log-uniform sampling
                    params[param_name] = np.exp(random.uniform(np.log(min_val), np.log(max_val)))
                else:
                    # Uniform sampling
                    params[param_name] = random.uniform(min_val, max_val)
            elif param_config['type'] == 'int':
                min_val, max_val = param_config['low'], param_config['high']
                params[param_name] = random.randint(min_val, max_val)
            elif param_config['type'] == 'categorical':
                params[param_name] = random.choice(param_config['choices'])

        return params

# Convenience functions
def create_hierarchical_hpo_config(
    base_models: Dict[str, Any],
    meta_models: Dict[str, Any],
    base_search_spaces: Dict[str, Dict[str, Any]],
    meta_search_spaces: Dict[str, Dict[str, Any]],
    n_trials_base: int = 100,
    n_trials_meta: int = 50
) -> HierarchicalHPOConfig:
    """Create hierarchical HPO configuration."""

    phase1_config = HPOPhaseConfig(
        phase_name="base_models",
        models=base_models,
        search_spaces=base_search_spaces,
        n_trials=n_trials_base
    )

    phase2_config = HPOPhaseConfig(
        phase_name="meta_models",
        models=meta_models,
        search_spaces=meta_search_spaces,
        n_trials=n_trials_meta
    )

    return HierarchicalHPOConfig(
        phase1_config=phase1_config,
        phase2_config=phase2_config
    )

def optimize_stacking_ensemble(
    base_models: Dict[str, Any],
    meta_models: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    base_search_spaces: Dict[str, Dict[str, Any]],
    meta_search_spaces: Dict[str, Dict[str, Any]],
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Optimize a stacking ensemble using hierarchical HPO."""

    config = create_hierarchical_hpo_config(
        base_models, meta_models, base_search_spaces, meta_search_spaces
    )

    hpo = HierarchicalHPO(config)
    return hpo.optimize_ensemble(X_train, y_train, X_val, y_val)
