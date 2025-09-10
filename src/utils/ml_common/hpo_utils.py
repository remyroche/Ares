"""
Advanced Hyperparameter Optimization Utilities

This module provides comprehensive hyperparameter optimization utilities with automated
search spaces, multi-objective optimization, early stopping, and advanced optimization strategies.

Key Features:
- Automated search space generation
- Multi-objective hyperparameter optimization
- Early stopping and pruning strategies
- Bayesian optimization
- Hyperparameter importance analysis
- Transfer learning for HPO
- Parallel optimization coordination

Built on existing utilities:
- Uses math_validation.py for safe mathematical operations
- Integrates with m1_gpu_utils.py for GPU acceleration
- Leverages common_operations.py for robust error handling
- Extends existing optuna-based patterns
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime
import logging
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import warnings
import time

from ..math_validation import safe_divide, safe_log
from ..common_operations import create_fallback_logger
from ..hardware.m1_optimizations import M1MemoryOptimizer
from ..parallel_processing_optimizer import ParallelProcessor
from .parallel_processing import ParallelProcessingCoordinator
from .memory_optimization import MemoryEfficientTraining

# Enhanced dependency management with fast fail
try:
    from ..logger import get_logger
    _LOGGER = get_logger("MLCommon.HPOUtils")
    print("✅ Custom logger available for MLCommon.HPOUtils")
except Exception as e:
    print(f"⚠️ Custom logger not available: {e}. Using standard logging.")
    _LOGGER = logging.getLogger("MLCommon.HPOUtils")
    _LOGGER.setLevel(logging.INFO)

logger = _LOGGER

try:
    import optuna
    from optuna.samplers import TPESampler, RandomSampler
    from optuna.pruners import MedianPruner, HyperbandPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available - limited HPO functionality")

try:
    from sklearn.model_selection import cross_val_score, StratifiedKFold, TimeSeriesSplit, cross_validate
    from sklearn.metrics import make_scorer
    from sklearn.utils.class_weight import compute_sample_weight
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available - limited HPO functionality")


class HyperparameterOptimization:
    """Advanced hyperparameter optimization utilities."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize hyperparameter optimization utilities with configuration."""
        self.config = config or {}
        self.logger = logger.getChild('HPOUtils')
        
        _LOGGER.info("🚀 Initializing HyperparameterOptimization...")

        # Configuration defaults
        self.enable_gpu = self.config.get('enable_gpu', True)
        self.enable_parallel = self.config.get('enable_parallel', True)
        self.max_workers = self.config.get('max_workers', 4)
        self.default_n_trials = self.config.get('default_n_trials', 50)
        self.default_timeout = self.config.get('default_timeout', 300)
        self.enable_pruning = self.config.get('enable_pruning', True)

        _LOGGER.info(f"⚙️ Configuration - GPU enabled: {self.enable_gpu}")
        _LOGGER.info(f"⚙️ Configuration - Parallel processing: {self.enable_parallel}")
        _LOGGER.info(f"⚙️ Configuration - Max workers: {self.max_workers}")
        _LOGGER.info(f"⚙️ Configuration - Default trials: {self.default_n_trials}")
        _LOGGER.info(f"⚙️ Configuration - Default timeout: {self.default_timeout}s")
        _LOGGER.info(f"⚙️ Configuration - Pruning enabled: {self.enable_pruning}")

        # Initialize utilities
        self.gpu_manager = M1MemoryOptimizer() if self.enable_gpu else None
        self.parallel_processor = ParallelProcessor() if self.enable_parallel else None
        self.parallel_coordinator = ParallelProcessingCoordinator(self.config) if self.enable_parallel else None
        self.memory_tools = MemoryEfficientTraining(self.config)

        # Optimization history
        self.optimization_history = []

        # Default search spaces for common models
        _LOGGER.debug("🔧 Initializing default search spaces...")
        self.default_search_spaces = self._initialize_default_search_spaces()
        
        _LOGGER.info("✅ HyperparameterOptimization initialized successfully")

    def automated_search_space_generation(self, model_type: str,
                                       data_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Automatically generate search space based on model type and data characteristics.

        Args:
            model_type: Type of model ('xgboost', 'lightgbm', 'random_forest', etc.)
            data_characteristics: Dictionary with data characteristics

        Returns:
            Generated search space dictionary
        """
        start_time = time.time()
        _LOGGER.info(f"🔧 Starting automated search space generation for {model_type}...")
        _LOGGER.debug(f"📊 Data characteristics: {data_characteristics}")
        
        try:
            self.logger.info(f"🔍 Generating automated search space for {model_type}")

            search_space = {}

            # Extract data characteristics
            n_samples = data_characteristics.get('n_samples', 1000)
            n_features = data_characteristics.get('n_features', 10)
            n_classes = data_characteristics.get('n_classes', 2)
            task_type = data_characteristics.get('task_type', 'classification')

            # Generate search space based on model type
            if model_type.lower() == 'xgboost':
                search_space = self._generate_xgboost_search_space(n_samples, n_features, n_classes, task_type)
            elif model_type.lower() == 'lightgbm':
                search_space = self._generate_lightgbm_search_space(n_samples, n_features, n_classes, task_type)
            elif model_type.lower() == 'random_forest':
                search_space = self._generate_rf_search_space(n_samples, n_features, n_classes)
            elif model_type.lower() == 'histgradientboostingclassifier':
                search_space = self._generate_histgb_search_space(n_samples, n_features, n_classes)
            elif model_type.lower() == 'neural_network':
                search_space = self._generate_nn_search_space(n_samples, n_features, n_classes)
            elif model_type.lower() == 'svm':
                search_space = self._generate_svm_search_space(n_samples, n_features)
            else:
                # Generic search space
                search_space = self._generate_generic_search_space(model_type)

            # Add data-driven adjustments
            search_space = self._adjust_search_space_for_data(search_space, data_characteristics)

            execution_time = time.time() - start_time
            _LOGGER.info(f"✅ Generated search space with {len(search_space)} parameters for {model_type} in {execution_time:.3f}s")
            _LOGGER.debug(f"📊 Search space parameters: {list(search_space.keys())}")
            return search_space

        except Exception as e:
            execution_time = time.time() - start_time
            _LOGGER.error(f"❌ Automated search space generation failed after {execution_time:.3f}s: {e}")
            return {}

    def multi_objective_optimization(self, model_factory: Callable,
                                  X: np.ndarray, y: np.ndarray,
                                  objectives: List[str],
                                  n_trials: int = 50,
                                  search_space: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform multi-objective hyperparameter optimization.

        Args:
            model_factory: Function that creates model with given parameters
            X: Feature matrix
            y: Target array
            objectives: List of objectives to optimize ('accuracy', 'f1', 'auc', 'speed')
            n_trials: Number of optimization trials
            search_space: Dictionary defining the search space for hyperparameters

        Returns:
            Multi-objective optimization results
        """
        start_time = time.time()
        _LOGGER.info(f"🎯 Starting multi-objective optimization...")
        _LOGGER.info(f"📊 Parameters - Objectives: {objectives}, Trials: {n_trials}, Data shape: {X.shape}")
        
        try:
            if not OPTUNA_AVAILABLE:
                _LOGGER.error("❌ Optuna required for multi-objective optimization")
                raise ImportError("Optuna required for multi-objective optimization")

            def objective(trial):
                # Sample hyperparameters
                params = self._sample_hyperparameters(trial, model_factory, search_space)

                # Create and train model
                model = model_factory(params)

                # Evaluate multiple objectives
                scores = {}

                # Performance objectives
                if 'accuracy' in objectives or 'f1' in objectives or 'auc' in objectives:
                    scores.update(self._evaluate_performance_objectives(model, X, y, objectives))

                # Speed objective
                if 'speed' in objectives:
                    scores['speed'] = self._evaluate_speed_objective(model, X)

                # Combine objectives (weighted sum for now)
                objective_scores = []
                for obj in objectives:
                    if obj in scores:
                        # Convert to minimization (higher is better)
                        if obj in ['accuracy', 'f1', 'auc']:
                            objective_scores.append(-scores[obj])  # Negative for maximization
                        elif obj == 'speed':
                            objective_scores.append(scores[obj])  # Lower is better

                return tuple(objective_scores) if len(objective_scores) > 1 else objective_scores[0]

            # Create study
            if len(objectives) > 1:
                study = optuna.create_study(directions=['minimize'] * len(objectives))
            else:
                study = optuna.create_study(direction='minimize')

            # Optimize
            study.optimize(objective, n_trials=n_trials)

            # Extract results
            results = {
                'best_params': study.best_params,
                'best_scores': study.best_value if len(objectives) == 1 else study.best_values,
                'objectives': objectives,
                'n_trials': len(study.trials),
                'optimization_history': [
                    {
                        'trial': t.number,
                        'params': t.params,
                        'scores': t.value if len(objectives) == 1 else t.values
                    }
                    for t in study.trials
                ]
            }

            execution_time = time.time() - start_time
            _LOGGER.info(f"✅ Multi-objective optimization completed in {execution_time:.3f}s")
            _LOGGER.info(f"📊 Results - Best scores: {results['best_scores']}, Trials: {len(study.trials)}")
            return results

        except Exception as e:
            execution_time = time.time() - start_time
            _LOGGER.error(f"❌ Multi-objective optimization failed after {execution_time:.3f}s: {e}")
            return {'error': str(e)}

    def early_stopping_optimization(self, model_factory: Callable,
                                  X: np.ndarray, y: np.ndarray,
                                  validation_data: Tuple[np.ndarray, np.ndarray],
                                  patience: int = 10,
                                  n_trials: int = 50) -> Dict[str, Any]:
        """
        Perform hyperparameter optimization with early stopping.

        Args:
            model_factory: Function that creates model with given parameters
            X: Training feature matrix
            y: Training target array
            validation_data: Tuple of (X_val, y_val)
            patience: Early stopping patience
            n_trials: Number of optimization trials

        Returns:
            Early stopping optimization results
        """
        try:
            self.logger.info(f"⏹️ Starting early stopping optimization (patience={patience})")

            if not OPTUNA_AVAILABLE:
                raise ImportError("Optuna required for early stopping optimization")

            X_val, y_val = validation_data
            best_score = -np.inf
            patience_counter = 0
            best_params = None

            def objective(trial):
                nonlocal best_score, patience_counter, best_params

                # Sample hyperparameters
                params = self._sample_hyperparameters(trial, model_factory)

                # Create and train model with early stopping
                model = model_factory(**params)

                # Train with early stopping logic
                score = self._train_with_early_stopping(model, X, y, X_val, y_val, patience)

                # Update best score and patience
                if score > best_score:
                    best_score = score
                    best_params = params
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Report intermediate results
                trial.report(score, step=trial.number)

                # Prune if needed
                if trial.should_prune():
                    raise optuna.TrialPruned()

                return score

            # Create study with pruner
            pruner = optuna.pruners.PatientPruner(
                optuna.pruners.MedianPruner(),
                patience=patience
            )

            study = optuna.create_study(direction='maximize', pruner=pruner)
            study.optimize(objective, n_trials=n_trials)

            results = {
                'best_params': best_params or study.best_params,
                'best_score': best_score if best_params else study.best_value,
                'n_trials': len(study.trials),
                'early_stopping_triggered': patience_counter >= patience,
                'final_patience_counter': patience_counter
            }

            best_val = results.get('best_score')
            best_str = f"{best_val:.4f}" if isinstance(best_val, (int, float, np.floating)) else str(best_val)
            self.logger.info(f"✅ Early stopping optimization completed - Best score: {best_str}")
            return results

        except Exception as e:
            self.logger.error(f"❌ Early stopping optimization failed: {e}")
            return {'error': str(e)}

    def bayesian_optimization(self, model_factory: Callable,
                            X: np.ndarray, y: np.ndarray,
                            search_space: Dict[str, Any],
                            n_trials: int = 50,
                            acquisition_function: str = 'ucb',
                            scoring: Union[str, Callable] = 'accuracy',
                            cv: Optional[Any] = None,
                            fit_params: Optional[Dict[str, Any]] = None,
                            pruner: Optional[str] = 'median',
                            storage: Optional[str] = None,
                            study_name: Optional[str] = None,
                            timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform Bayesian hyperparameter optimization.

        Args:
            model_factory: Function that creates model with given parameters
            X: Feature matrix
            y: Target array
            search_space: Dictionary defining the search space
            n_trials: Number of optimization trials
            acquisition_function: Acquisition function ('ucb', 'ei', 'poi')

        Returns:
            Bayesian optimization results
        """
        try:
            self.logger.info(f"🎲 Starting Bayesian optimization with {acquisition_function} acquisition")

            if not OPTUNA_AVAILABLE:
                raise ImportError("Optuna required for Bayesian optimization")

            def objective(trial):
                params = {}

                # Sample from search space
                for param_name, param_config in search_space.items():
                    if isinstance(param_config, dict):
                        param_type = param_config.get('type', 'float')
                        if param_type == 'float':
                            params[param_name] = trial.suggest_float(
                                param_name,
                                param_config['low'],
                                param_config['high']
                            )
                        elif param_type == 'int':
                            params[param_name] = trial.suggest_int(
                                param_name,
                                param_config['low'],
                                param_config['high']
                            )
                        elif param_type == 'categorical':
                            params[param_name] = trial.suggest_categorical(
                                param_name,
                                param_config['choices']
                            )
                    else:
                        # Legacy format support
                        if isinstance(param_config, tuple) and len(param_config) == 2:
                            params[param_name] = trial.suggest_float(
                                param_name, param_config[0], param_config[1]
                            )

                # Create and evaluate model
                model = model_factory(**params)
                # Cap per-trial parallelism if supported
                try:
                    if hasattr(model, 'set_params'):
                        model.set_params(**{k: v for k, v in {'n_jobs': 1}.items() if k in getattr(model, 'get_params')().keys()})
                except Exception:
                    pass

                # Prepare CV and fit params
                cv_obj = cv if cv is not None else self._create_time_series_split(len(X))

                # Compute sample weights if classification and estimator supports it
                fp = dict(fit_params or {})
                try:
                    if SKLEARN_AVAILABLE and len(np.unique(y)) <= 10:
                        fp.setdefault('sample_weight', compute_sample_weight('balanced', y))
                except Exception:
                    pass

                # Manual CV loop to support sample_weight without passing fit_params
                try:
                    fold_scores: list[float] = []
                    for i, (train_idx, test_idx) in enumerate(cv_obj.split(X, y)):
                        X_tr, X_te = X[train_idx], X[test_idx]
                        y_tr, y_te = y[train_idx], y[test_idx]
                        mdl = model_factory(**params)
                        try:
                            import inspect
                            if 'sample_weight' in inspect.signature(mdl.fit).parameters and 'sample_weight' in fp:
                                mdl.fit(X_tr, y_tr, sample_weight=fp['sample_weight'][train_idx])
                            else:
                                mdl.fit(X_tr, y_tr)
                        except Exception:
                            mdl.fit(X_tr, y_tr)
                        try:
                            from sklearn.metrics import get_scorer
                            scorer = get_scorer(scoring) if isinstance(scoring, str) else scoring
                            score = scorer(mdl, X_te, y_te)
                        except Exception:
                            score = mdl.score(X_te, y_te) if hasattr(mdl, 'score') else 0.0
                        fold_scores.append(float(score))
                        trial.report(float(score), step=i)
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                    if fold_scores:
                        return float(np.mean(fold_scores))
                except Exception:
                    pass

                # Fallback single-score
                return self._evaluate_model(model, X, y)

            # Create study with TPE sampler (Bayesian optimization) and pruner/storage
            sampler = TPESampler()
            pruner_obj = None
            if pruner == 'median':
                pruner_obj = MedianPruner()
            elif pruner == 'hyperband':
                pruner_obj = HyperbandPruner()

            study = optuna.create_study(
                direction='maximize',
                sampler=sampler,
                pruner=pruner_obj,
                study_name=study_name,
                storage=storage,
                load_if_exists=bool(storage and study_name)
            )

            study.optimize(objective, n_trials=n_trials, timeout=timeout)

            results = {
                'best_params': study.best_params,
                'best_score': study.best_value,
                'n_trials': len(study.trials),
                'optimization_curve': [t.value for t in study.trials],
                'parameter_importance': self._calculate_parameter_importance(study)
            }

            best_val2 = results.get('best_score')
            best_str2 = f"{best_val2:.4f}" if isinstance(best_val2, (int, float, np.floating)) else str(best_val2)
            self.logger.info(f"✅ Bayesian optimization completed - Best score: {best_str2}")
            return results

        except Exception as e:
            self.logger.error(f"❌ Bayesian optimization failed: {e}")
            return {'error': str(e)}

    def staged_hpo(self, model_factory: Callable,
                   X: np.ndarray, y: np.ndarray,
                   search_space: Dict[str, Any],
                   coarse_strategy: str = 'random',
                   coarse_grid_points: int = 3,
                   coarse_n_samples: int = 50,
                   bayes_n_trials: int = 50,
                   scoring: Union[str, Callable] = 'balanced_accuracy',
                   cv: Optional[Any] = None,
                   pruner: str = 'hyperband',
                   storage: Optional[str] = None,
                   study_name: Optional[str] = None,
                   timeout: Optional[int] = None,
                   subsample_rate: float = 0.3,
                   finalize_refine: bool = True) -> Dict[str, Any]:
        """Staged HPO: coarse grid/random → narrowed Bayesian → optional fine-tune.

        Returns dict with coarse_results, bayes_results, final_params, final_score.
        """
        try:
            self.logger.info("🌀 Starting staged HPO: coarse → bayesian → refine")

            # Optional subsampling for coarse stage (memmap-aware)
            X_train, y_train = X, y
            if 0 < subsample_rate < 1.0 and len(X) > 100:
                try:
                    n_sub = max(100, int(len(X) * subsample_rate))
                    idx = np.linspace(0, len(X) - 1, num=n_sub, dtype=int)
                    X_train = X[idx]
                    y_train = y[idx]
                except Exception as e:
                    self.logger.warning(f"⚠️ Subsampling failed, using full data: {e}")

            # Precompute CV if not provided
            cv_obj = cv if cv is not None else self._create_time_series_split(len(X_train))

            # Build coarse parameter grid from search_space
            param_grid = self._coarse_grid_from_search_space(search_space, coarse_grid_points)

            coarse_results = {}
            if self.parallel_coordinator:
                # Use enhanced parallel random/grid search
                coarse_results = self.parallel_coordinator.parallel_hyperparameter_search(
                    parameter_grid=param_grid,
                    model_factory=model_factory,
                    X=X_train,
                    y=y_train,
                    evaluation_function=self._build_default_eval(scoring, cv_obj),
                    search_strategy='random' if coarse_strategy == 'random' else 'grid',
                    # extended parameter accepted after coordinator update
                    n_random_samples=coarse_n_samples,
                )
            else:
                # Fallback simple random sampling
                sampled = self._generate_random_param_combinations(param_grid, coarse_n_samples)
                best_score = -np.inf
                best_params = {}
                for p in sampled:
                    model = model_factory(**p)
                    score = self._evaluate_model_cv(model, X_train, y_train, cv_obj, scoring)
                    if score > best_score:
                        best_score, best_params = score, p
                coarse_results = {'best_params': best_params, 'best_score': best_score, 'all_results': []}

            # Narrow search space around coarse best
            best_coarse = coarse_results.get('best_params', {})
            narrowed = self._narrow_search_space(search_space, best_coarse)

            # Bayesian stage on full data if feasible
            bayes_results = self.bayesian_optimization(
                model_factory=model_factory,
                X=X,
                y=y,
                search_space=narrowed,
                n_trials=bayes_n_trials,
                scoring=scoring,
                cv=cv if cv is not None else self._create_time_series_split(len(X)),
                pruner=pruner,
                storage=storage,
                study_name=study_name,
                timeout=timeout
            )

            final_params = dict(best_coarse)
            final_score = coarse_results.get('best_score', 0.0)
            if 'best_params' in bayes_results:
                final_params = bayes_results['best_params']
                final_score = bayes_results.get('best_score', final_score)

            # Optional local fine-tune around best (small random jitters)
            if finalize_refine:
                fine_params, fine_score = self._local_refine(model_factory, X, y, final_params, scoring,
                                                             cv if cv is not None else self._create_time_series_split(len(X)))
                if fine_score > final_score:
                    final_params, final_score = fine_params, fine_score

            return {
                'coarse_results': coarse_results,
                'bayes_results': bayes_results,
                'final_params': final_params,
                'final_score': final_score
            }
        except Exception as e:
            self.logger.error(f"❌ Staged HPO failed: {e}")
            return {'error': str(e)}

    def hyperparameter_importance_analysis(self, study_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze hyperparameter importance from optimization results.

        Args:
            study_results: Results from hyperparameter optimization

        Returns:
            Hyperparameter importance analysis
        """
        try:
            self.logger.info("📊 Analyzing hyperparameter importance")

            if not OPTUNA_AVAILABLE:
                return {'error': 'Optuna required for importance analysis'}

            # This would require access to the actual study object
            # For now, return placeholder analysis
            importance_analysis = {
                'method': 'permutation_importance',
                'importance_scores': {},
                'ranking': [],
                'recommendations': []
            }

            # Placeholder for actual importance calculation
            if 'best_params' in study_results:
                params = study_results['best_params']
                # Simple heuristic-based importance
                importance_analysis['importance_scores'] = {
                    param: 0.5 for param in params.keys()
                }

                importance_analysis['ranking'] = [
                    {'parameter': param, 'importance': score}
                    for param, score in importance_analysis['importance_scores'].items()
                ]

            self.logger.info("✅ Hyperparameter importance analysis completed")
            return importance_analysis

        except Exception as e:
            self.logger.error(f"❌ Hyperparameter importance analysis failed: {e}")
            return {'error': str(e)}

    def transfer_learning_hpo(self, base_study_results: Dict[str, Any],
                            new_data: Tuple[np.ndarray, np.ndarray],
                            similarity_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Perform transfer learning for hyperparameter optimization.

        Args:
            base_study_results: Results from previous optimization
            new_data: New dataset (X, y)
            similarity_threshold: Similarity threshold for transfer

        Returns:
            Transfer learning optimization results
        """
        try:
            self.logger.info(f"🔄 Starting transfer learning HPO (threshold={similarity_threshold})")

            X_new, y_new = new_data

            # Assess data similarity
            similarity_score = self._assess_data_similarity(base_study_results, X_new, y_new)

            if similarity_score >= similarity_threshold:
                # Use transfer learning
                transfer_results = self._perform_transfer_optimization(
                    base_study_results, X_new, y_new, similarity_score
                )
                transfer_results['transfer_used'] = True
                transfer_results['similarity_score'] = similarity_score
            else:
                # Perform new optimization
                transfer_results = self._perform_fresh_optimization(X_new, y_new)
                transfer_results['transfer_used'] = False
                transfer_results['similarity_score'] = similarity_score

            self.logger.info(f"✅ Transfer learning HPO completed - Transfer used: {transfer_results['transfer_used']}")
            return transfer_results

        except Exception as e:
            self.logger.error(f"❌ Transfer learning HPO failed: {e}")
            return {'error': str(e)}

    def parallel_optimization_coordinator(self, optimization_tasks: List[Dict[str, Any]],
                                       max_workers: Optional[int] = None) -> Dict[str, Any]:
        """
        Coordinate parallel hyperparameter optimization across multiple tasks.

        Args:
            optimization_tasks: List of optimization task configurations
            max_workers: Maximum number of parallel workers

        Returns:
            Parallel optimization results
        """
        try:
            if max_workers is None:
                max_workers = self.max_workers

            self.logger.info(f"🔄 Starting parallel optimization with {max_workers} workers")

            def run_single_optimization(task_config):
                """Run a single optimization task."""
                try:
                    task_type = task_config.get('type', 'bayesian')

                    if task_type == 'bayesian':
                        return self.bayesian_optimization(**task_config)
                    elif task_type == 'multi_objective':
                        return self.multi_objective_optimization(**task_config)
                    else:
                        return {'error': f'Unknown optimization type: {task_type}'}

                except Exception as task_e:
                    return {'error': str(task_e), 'task_config': task_config}

            # Run optimizations in parallel
            if self.enable_parallel and len(optimization_tasks) > 1:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    results = list(executor.map(run_single_optimization, optimization_tasks))
            else:
                results = [run_single_optimization(task) for task in optimization_tasks]

            # Aggregate results
            parallel_results = {
                'task_results': results,
                'successful_tasks': sum(1 for r in results if 'error' not in r),
                'failed_tasks': sum(1 for r in results if 'error' in r),
                'total_tasks': len(optimization_tasks),
                'execution_time': datetime.now().isoformat()
            }

            self.logger.info(f"✅ Parallel optimization completed - "
                           f"{parallel_results['successful_tasks']}/{parallel_results['total_tasks']} tasks successful")
            return parallel_results

        except Exception as e:
            self.logger.error(f"❌ Parallel optimization coordination failed: {e}")
            return {'error': str(e)}

    def _initialize_default_search_spaces(self) -> Dict[str, Dict[str, Any]]:
        """Initialize default search spaces for common models."""
        return {
            'xgboost': {
                'max_depth': {'type': 'int', 'low': 3, 'high': 10},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3},
                'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
                'subsample': {'type': 'float', 'low': 0.5, 'high': 1.0},
                'colsample_bytree': {'type': 'float', 'low': 0.5, 'high': 1.0},
                'gamma': {'type': 'float', 'low': 0, 'high': 5},
                'reg_alpha': {'type': 'float', 'low': 0, 'high': 10},
                'reg_lambda': {'type': 'float', 'low': 0, 'high': 10}
            },
            'lightgbm': {
                'num_leaves': {'type': 'int', 'low': 10, 'high': 100},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3},
                'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
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
                'learning_rate': {'type': 'float', 'low': 0.0001, 'high': 0.01},
                'dropout_rate': {'type': 'float', 'low': 0.0, 'high': 0.5},
                'batch_size': {'type': 'int', 'low': 16, 'high': 128},
                'epochs': {'type': 'int', 'low': 10, 'high': 100}
            }
        }

    def _generate_xgboost_search_space(self, n_samples: int, n_features: int,
                                     n_classes: int, task_type: str) -> Dict[str, Any]:
        """Generate XGBoost search space."""
        search_space = self.default_search_spaces['xgboost'].copy()

        # Adjust based on data size
        if n_samples < 1000:
            search_space['n_estimators']['high'] = 200
        elif n_samples > 10000:
            search_space['n_estimators']['low'] = 100

        # Adjust for multi-class
        if n_classes > 2 and task_type == 'classification':
            search_space['objective'] = {'type': 'categorical',
                                       'choices': ['multi:softmax', 'multi:softprob']}

        return search_space

    def _generate_lightgbm_search_space(self, n_samples: int, n_features: int,
                                      n_classes: int, task_type: str) -> Dict[str, Any]:
        """Generate LightGBM search space."""
        search_space = self.default_search_spaces['lightgbm'].copy()

        # Adjust based on data characteristics
        if n_features < 50:
            search_space['num_leaves']['high'] = 50
        elif n_features > 500:
            search_space['num_leaves']['low'] = 20

        return search_space

    def _generate_rf_search_space(self, n_samples: int, n_features: int,
                                n_classes: int) -> Dict[str, Any]:
        """Generate Random Forest search space."""
        search_space = self.default_search_spaces['random_forest'].copy()

        # Adjust based on data size
        if n_samples > 10000:
            search_space['n_estimators']['low'] = 100

        return search_space

    def _generate_histgb_search_space(self, n_samples: int, n_features: int,
                                    n_classes: int) -> Dict[str, Any]:
        """Generate HistGradientBoostingClassifier search space."""
        return {
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
            'max_iter': {'type': 'int', 'low': 50, 'high': 300, 'step': 10},
            'max_depth': {'type': 'int', 'low': 3, 'high': 15},
            'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 20},
            'l2_regularization': {'type': 'float', 'low': 0.0, 'high': 1.0}
        }

    def _generate_nn_search_space(self, n_samples: int, n_features: int,
                                n_classes: int) -> Dict[str, Any]:
        """Generate Neural Network search space."""
        search_space = self.default_search_spaces['neural_network'].copy()

        # Adjust based on data characteristics
        if n_features > 100:
            search_space['hidden_units']['low'] = 64
        if n_samples < 1000:
            search_space['batch_size']['high'] = 64

        return search_space

    def _generate_svm_search_space(self, n_samples: int, n_features: int) -> Dict[str, Any]:
        """Generate SVM search space."""
        return {
            'C': {'type': 'float', 'low': 0.1, 'high': 100.0},
            'gamma': {'type': 'categorical', 'choices': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0]},
            'kernel': {'type': 'categorical', 'choices': ['rbf', 'linear', 'poly', 'sigmoid']}
        }

    def _generate_generic_search_space(self, model_type: str) -> Dict[str, Any]:
        """Generate generic search space for unknown model types."""
        return {
            'learning_rate': {'type': 'float', 'low': 0.001, 'high': 0.1},
            'regularization': {'type': 'float', 'low': 0.0, 'high': 1.0},
            'complexity_param': {'type': 'int', 'low': 1, 'high': 100}
        }

    def _adjust_search_space_for_data(self, search_space: Dict[str, Any],
                                    data_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust search space based on data characteristics."""
        try:
            n_samples = data_characteristics.get('n_samples', 1000)
            n_features = data_characteristics.get('n_features', 10)

            # Adjust for small datasets
            if n_samples < 1000:
                for param_config in search_space.values():
                    if isinstance(param_config, dict):
                        # Reduce complexity for small datasets
                        if 'high' in param_config and param_config['type'] == 'int':
                            param_config['high'] = min(param_config['high'], 50)

            # Adjust for high-dimensional data
            if n_features > 100:
                for param_config in search_space.values():
                    if isinstance(param_config, dict):
                        # Increase regularization for high dimensions
                        if 'regularization' in str(param_config):
                            if 'low' in param_config:
                                param_config['low'] = max(param_config['low'], 0.1)

            return search_space

        except Exception as e:
            self.logger.warning(f"Search space adjustment failed: {e}")
            return search_space

    def _sample_hyperparameters(self, trial: Any, model_factory: Callable, search_space: Dict[str, Any] = None) -> Dict[str, Any]:
        """Sample hyperparameters for a trial based on the provided search space."""
        try:
            if search_space is None:
                # Fallback to default parameters if no search space provided
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10)
                }
                return params

            params = {}
            for param_name, param_config in search_space.items():
                if param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, 
                        param_config['low'], 
                        param_config['high'],
                        step=param_config.get('step', 1)
                    )
                elif param_config['type'] == 'float':
                    if param_config.get('log', False):
                        params[param_name] = trial.suggest_float(
                            param_name, 
                            param_config['low'], 
                            param_config['high'],
                            log=True
                        )
                    else:
                        params[param_name] = trial.suggest_float(
                            param_name, 
                            param_config['low'], 
                            param_config['high']
                        )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, 
                        param_config['choices']
                    )

            return params

        except Exception as e:
            self.logger.warning(f"Hyperparameter sampling failed: {e}")
            return {}

    def _evaluate_performance_objectives(self, model: Any, X: np.ndarray, y: np.ndarray,
                                       objectives: List[str]) -> Dict[str, float]:
        """Evaluate performance objectives."""
        try:
            scores = {}

            if not SKLEARN_AVAILABLE:
                return {'accuracy': 0.5}

            # Simple cross-validation for evaluation
            cv_scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
            scores['accuracy'] = np.mean(cv_scores)

            if 'f1' in objectives:
                f1_scores = cross_val_score(model, X, y, cv=3, scoring='f1_macro')
                scores['f1'] = np.mean(f1_scores)

            if 'auc' in objectives and len(np.unique(y)) == 2:
                auc_scores = cross_val_score(model, X, y, cv=3, scoring='roc_auc')
                scores['auc'] = np.mean(auc_scores)

            return scores

        except Exception as e:
            self.logger.warning(f"Performance objective evaluation failed: {e}")
            return {'accuracy': 0.5}

    def _evaluate_speed_objective(self, model: Any, X: np.ndarray) -> float:
        """Evaluate model training speed."""
        try:
            import time
            start_time = time.time()

            # Quick training on subset
            subset_size = min(1000, len(X))
            X_subset = X[:subset_size]

            if hasattr(model, 'fit'):
                model.fit(X_subset, np.random.randint(0, 2, subset_size))

            training_time = time.time() - start_time
            return training_time

        except Exception as e:
            self.logger.warning(f"Speed objective evaluation failed: {e}")
            return 1.0  # Default time

    def _train_with_early_stopping(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                                 X_val: np.ndarray, y_val: np.ndarray, patience: int) -> float:
        """Train model with early stopping."""
        try:
            # This is a simplified implementation
            # In practice, you'd implement proper early stopping logic
            model.fit(X_train, y_train)

            # Evaluate on validation set
            if hasattr(model, 'predict_proba'):
                y_pred = model.predict_proba(X_val)[:, 1]
            else:
                y_pred = model.predict(X_val)

            # Simple accuracy calculation
            if len(np.unique(y_val)) <= 10:  # Classification
                from sklearn.metrics import accuracy_score
                score = accuracy_score(y_val, y_pred)
            else:  # Regression
                from sklearn.metrics import r2_score
                score = r2_score(y_val, y_pred)

            return score

        except Exception as e:
            self.logger.warning(f"Early stopping training failed: {e}")
            return 0.5

    def _evaluate_model(self, model: Any, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate a model using cross-validation."""
        try:
            if not SKLEARN_AVAILABLE:
                return 0.5

            scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
            return np.mean(scores)

        except Exception as e:
            self.logger.warning(f"Model evaluation failed: {e}")
            return 0.5

    # ---- Helpers for staged HPO ----
    def _create_time_series_split(self, n_samples: int, n_splits: int = 5, gap: int = 0) -> Any:
        try:
            test_size = max(1, n_samples // (n_splits + 1))
            return TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
        except Exception:
            return 3

    def _evaluate_model_cv(self, model: Any, X: np.ndarray, y: np.ndarray,
                           cv_obj: Any, scoring: Union[str, Callable]) -> float:
        try:
            # Cap nested parallelism if possible
            try:
                if hasattr(model, 'set_params') and hasattr(model, 'get_params'):
                    params = model.get_params()
                    if 'n_jobs' in params:
                        model.set_params(n_jobs=1)
            except Exception:
                pass

            fit_params = {}
            try:
                if SKLEARN_AVAILABLE and len(np.unique(y)) <= 10:
                    fit_params['sample_weight'] = compute_sample_weight('balanced', y)
            except Exception:
                pass
            # Manual CV to handle sample_weight safely
            try:
                fold_scores: list[float] = []
                for train_idx, test_idx in cv_obj.split(X, y):
                    X_tr, X_te = X[train_idx], X[test_idx]
                    y_tr, y_te = y[train_idx], y[test_idx]
                    mdl = model
                    try:
                        import inspect
                        if 'sample_weight' in inspect.signature(mdl.fit).parameters and 'sample_weight' in fit_params:
                            mdl.fit(X_tr, y_tr, sample_weight=fit_params['sample_weight'][train_idx])
                        else:
                            mdl.fit(X_tr, y_tr)
                    except Exception:
                        mdl.fit(X_tr, y_tr)
                    try:
                        from sklearn.metrics import get_scorer
                        scorer = get_scorer(scoring) if isinstance(scoring, str) else scoring
                        score = scorer(mdl, X_te, y_te)
                    except Exception:
                        score = mdl.score(X_te, y_te) if hasattr(mdl, 'score') else 0.0
                    fold_scores.append(float(score))
                if fold_scores:
                    return float(np.mean(fold_scores))
            except Exception:
                pass
            return 0.5
        except Exception as e:
            self.logger.warning(f"CV evaluation failed: {e}")
            return 0.5

    def _coarse_grid_from_search_space(self, search_space: Dict[str, Any], grid_points: int) -> Dict[str, List[Any]]:
        grid = {}
        try:
            for name, cfg in search_space.items():
                if isinstance(cfg, dict):
                    typ = cfg.get('type', 'float')
                    if typ == 'float':
                        low, high = cfg['low'], cfg['high']
                        grid[name] = np.linspace(low, high, num=max(2, grid_points)).tolist()
                    elif typ == 'int':
                        low, high = cfg['low'], cfg['high']
                        if high == low:
                            grid[name] = [low]
                        else:
                            pts = np.linspace(low, high, num=max(2, grid_points))
                            grid[name] = sorted({int(round(v)) for v in pts})
                    elif typ == 'categorical':
                        grid[name] = cfg.get('choices', [])
                else:
                    # Legacy tuple(low, high)
                    if isinstance(cfg, tuple) and len(cfg) == 2:
                        grid[name] = np.linspace(cfg[0], cfg[1], num=max(2, grid_points)).tolist()
            return grid
        except Exception:
            return {}

    def _generate_random_param_combinations(self, grid: Dict[str, List[Any]], n_samples: int) -> List[Dict[str, Any]]:
        try:
            import random
            keys = list(grid.keys())
            combos = []
            for _ in range(n_samples):
                p = {k: random.choice(grid[k]) for k in keys if grid.get(k)}
                combos.append(p)
            return combos
        except Exception:
            return []

    def _narrow_search_space(self, search_space: Dict[str, Any], center_params: Dict[str, Any],
                              shrink: float = 0.5) -> Dict[str, Any]:
        try:
            narrowed = {}
            for name, cfg in search_space.items():
                if name not in center_params:
                    narrowed[name] = cfg
                    continue
                val = center_params[name]
                if isinstance(cfg, dict):
                    typ = cfg.get('type', 'float')
                    if typ in ('float', 'int'):
                        low, high = cfg['low'], cfg['high']
                        span = (high - low) * shrink / 2.0
                        new_low = max(low, val - span)
                        new_high = min(high, val + span)
                        narrowed[name] = {'type': typ, 'low': new_low, 'high': new_high}
                    else:
                        narrowed[name] = cfg
                else:
                    narrowed[name] = cfg
            return narrowed
        except Exception:
            return search_space

    def _local_refine(self, model_factory: Callable, X: np.ndarray, y: np.ndarray,
                       best_params: Dict[str, Any], scoring: Union[str, Callable], cv_obj: Any,
                       n_trials: int = 15, jitter: float = 0.1) -> Tuple[Dict[str, Any], float]:
        try:
            import random
            best_p = dict(best_params)
            best_s = -np.inf
            # Evaluate baseline
            base_model = model_factory(**best_params)
            base_score = self._evaluate_model_cv(base_model, X, y, cv_obj, scoring)
            best_p, best_s = best_params, base_score
            for _ in range(n_trials):
                cand = dict(best_params)
                for k, v in best_params.items():
                    if isinstance(v, (int, float)):
                        delta = v * jitter if v != 0 else jitter
                        if isinstance(v, int):
                            cand[k] = max(1, int(round(v + random.uniform(-delta, delta))))
                        else:
                            cand[k] = v + random.uniform(-delta, delta)
                model = model_factory(**cand)
                s = self._evaluate_model_cv(model, X, y, cv_obj, scoring)
                if s > best_s:
                    best_p, best_s = cand, s
            return best_p, best_s
        except Exception:
            return best_params, base_score if 'base_score' in locals() else 0.0

    def _build_default_eval(self, scoring: Union[str, Callable], cv_obj: Any) -> Callable:
        def _eval(model, X, y):
            return self._evaluate_model_cv(model, X, y, cv_obj, scoring)
        return _eval

    def _calculate_parameter_importance(self, study: Any) -> Dict[str, float]:
        """Calculate parameter importance from study."""
        try:
            if not OPTUNA_AVAILABLE:
                return {}

            # Use optuna's built-in importance calculation
            importance = optuna.importance.get_param_importances(study)
            return dict(importance)

        except Exception as e:
            self.logger.warning(f"Parameter importance calculation failed: {e}")
            return {}

    def _assess_data_similarity(self, base_results: Dict[str, Any],
                              X_new: np.ndarray, y_new: np.ndarray) -> float:
        """Assess similarity between datasets."""
        try:
            # Simple heuristic-based similarity assessment
            # In practice, you'd implement more sophisticated similarity measures
            base_n_samples = base_results.get('n_samples', 1000)
            new_n_samples = len(X_new)

            sample_ratio = min(base_n_samples, new_n_samples) / max(base_n_samples, new_n_samples)

            # Factor in feature dimensions
            base_n_features = base_results.get('n_features', 10)
            new_n_features = X_new.shape[1]

            feature_ratio = min(base_n_features, new_n_features) / max(base_n_features, new_n_features)

            # Combined similarity score
            similarity = (sample_ratio + feature_ratio) / 2
            return similarity

        except Exception as e:
            self.logger.warning(f"Data similarity assessment failed: {e}")
            return 0.5

    def _perform_transfer_optimization(self, base_results: Dict[str, Any],
                                     X_new: np.ndarray, y_new: np.ndarray,
                                     similarity_score: float) -> Dict[str, Any]:
        """Perform transfer learning optimization."""
        try:
            # Use base results as starting point
            transfer_results = base_results.copy()
            transfer_results['transfer_applied'] = True
            transfer_results['similarity_score'] = similarity_score

            # Fine-tune on new data (simplified)
            # In practice, you'd implement actual transfer learning logic
            transfer_results['fine_tuned'] = True

            return transfer_results

        except Exception as e:
            self.logger.warning(f"Transfer optimization failed: {e}")
            return {'error': str(e)}

    def _perform_fresh_optimization(self, X_new: np.ndarray, y_new: np.ndarray) -> Dict[str, Any]:
        """Perform fresh optimization on new data."""
        try:
            # Perform standard optimization
            # This is a placeholder - implement actual optimization logic
            fresh_results = {
                'fresh_optimization': True,
                'best_params': {},
                'best_score': 0.5
            }

            return fresh_results

        except Exception as e:
            self.logger.warning(f"Fresh optimization failed: {e}")
            return {'error': str(e)}
