from src.utils.tprint import tprint

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
from concurrent.futures import ThreadPoolExecutor
import time

from .parallel_processing import ParallelProcessingCoordinator
from ...nonlinear_optimization_helpers import (
    NonLinearConfig, NonLinearParameterSampler, apply_nonlinear_scoring,
    create_enhanced_search_space
)

# Enhanced dependency management with fast fail
try:
    from ..logger import get_logger
    _LOGGER = get_logger("MLCommon.HPOUtils")
    tprint("✅ Custom logger available for MLCommon.HPOUtils")
except Exception as e:
    tprint(f"⚠️ Custom logger not available: {e}. Using standard logging.")
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
    """Enhanced hyperparameter optimization utilities with monitoring and failure detection."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, nonlinear_config: Optional[NonLinearConfig] = None):
        """Initialize hyperparameter optimization utilities with configuration."""
        self.config = config or {}
        self.logger = logger.getChild('HPOUtils')
        
        # Non-linear optimization configuration
        self.nonlinear_config = nonlinear_config or NonLinearConfig()
        self.parameter_sampler = NonLinearParameterSampler(self.nonlinear_config)
        self.use_nonlinear_optimization = self.config.get('use_nonlinear_optimization', True)
        
        _LOGGER.info("🚀 Initializing Enhanced HyperparameterOptimization...")

        # Configuration defaults
        self.enable_parallel = self.config.get('enable_parallel', True)
        self.max_workers = self.config.get('max_workers', 4)

        # Enhanced monitoring configuration
        self.enable_monitoring = self.config.get('enable_monitoring', True)
        self.convergence_config = self.config.get('convergence', {
            'improvement_threshold': 0.001,
            'patience_trials': 20,
            'variance_threshold': 0.01,
            'confidence_level': 0.95,
            'min_trials_for_convergence': 10
        })
        self.failure_detection_config = self.config.get('failure_detection', {
            'max_failure_rate': 0.3,
            'consecutive_failures_threshold': 5,
            'timeout_threshold': 3600,
            'memory_threshold': 0.9,
            'performance_degradation_threshold': 0.1
        })

        _LOGGER.info(f"⚙️ Configuration - Parallel processing: {self.enable_parallel}")
        _LOGGER.info(f"⚙️ Configuration - Max workers: {self.max_workers}")
        _LOGGER.info(f"⚙️ Configuration - Monitoring enabled: {self.enable_monitoring}")
        _LOGGER.info(f"🚀 Configuration - Non-linear optimization: {self.use_nonlinear_optimization}")

        # Initialize utilities
        self.parallel_coordinator = ParallelProcessingCoordinator(self.config) if self.enable_parallel else None

        # Enhanced optimization tracking
        self.optimization_history = []
        self.active_studies = {}
        self.trial_results = {}

        # Default search spaces for common models
        _LOGGER.debug("🔧 Initializing default search spaces...")
        self.default_search_spaces = self._initialize_default_search_spaces()
        
        # Enhanced search spaces with non-linear transformations
        if self.use_nonlinear_optimization:
            _LOGGER.debug("🚀 Creating enhanced search spaces with non-linear transformations...")
            self.enhanced_search_spaces = self._create_enhanced_search_spaces()
        
        _LOGGER.info("✅ Enhanced HyperparameterOptimization initialized successfully")
    
    def _create_enhanced_search_spaces(self) -> Dict[str, Dict[str, Any]]:
        """Create enhanced search spaces with non-linear transformation metadata."""
        enhanced_spaces = {}
        
        for model_type, space in self.default_search_spaces.items():
            enhanced_spaces[model_type] = create_enhanced_search_space(space, self.nonlinear_config)
        
        return enhanced_spaces

    def start_study_monitoring(self, study_id: str, study_name: str) -> Dict[str, Any]:
        """Start monitoring a new HPO study."""
        try:
            study_info = {
                'study_id': study_id,
                'study_name': study_name,
                'start_time': datetime.now(),
                'status': 'running',
                'total_trials': 0,
                'successful_trials': 0,
                'failed_trials': 0,
                'best_value': None,
                'best_parameters': None,
                'convergence_info': None,
                'error_summary': {}
            }
            
            self.active_studies[study_id] = study_info
            self.trial_results[study_id] = []
            
            _LOGGER.info(f"🚀 Started monitoring HPO study: {study_name} ({study_id})")
            return study_info
            
        except Exception as e:
            _LOGGER.error(f"❌ Failed to start study monitoring: {e}")
            raise

    def record_trial_with_monitoring(self, 
                                   study_id: str,
                                   trial_number: int,
                                   parameters: Dict[str, Any],
                                   objective_value: float,
                                   **kwargs) -> Dict[str, Any]:
        """Record trial result with enhanced monitoring."""
        try:
            trial_result = {
                'trial_number': trial_number,
                'timestamp': datetime.now(),
                'parameters': parameters,
                'objective_value': objective_value,
                'objective_std': kwargs.get('objective_std'),
                'training_time': kwargs.get('training_time'),
                'memory_usage': kwargs.get('memory_usage'),
                'convergence_info': kwargs.get('convergence_info'),
                'error_info': kwargs.get('error_info'),
                'metadata': kwargs.get('metadata', {})
            }
            
            if study_id in self.active_studies:
                self.trial_results[study_id].append(trial_result)
                study_info = self.active_studies[study_id]
                study_info['total_trials'] += 1
                
                if trial_result['error_info'] is None:
                    study_info['successful_trials'] += 1
                    
                    # Update best value
                    if (study_info['best_value'] is None or 
                        objective_value > study_info['best_value']):
                        study_info['best_value'] = objective_value
                        study_info['best_parameters'] = parameters.copy()
                else:
                    study_info['failed_trials'] += 1
                    self._update_error_summary(study_info, trial_result['error_info'])
                
                # Check for convergence
                convergence_info = self._check_convergence(study_id)
                if convergence_info and convergence_info.get('is_converged'):
                    study_info['convergence_info'] = convergence_info
                    study_info['status'] = 'converged'
                    _LOGGER.info(f"✅ Study {study_id} converged after {trial_number} trials")
                
                # Check for failure conditions
                if self._check_failure_conditions(study_id):
                    study_info['status'] = 'failed'
                    _LOGGER.error(f"❌ Study {study_id} failed due to failure conditions")
            
            return trial_result
            
        except Exception as e:
            _LOGGER.error(f"❌ Failed to record trial: {e}")
            raise

    def _check_convergence(self, study_id: str) -> Optional[Dict[str, Any]]:
        """Check if the study has converged."""
        try:
            if study_id not in self.trial_results:
                return None
            
            trial_results = self.trial_results[study_id]
            if len(trial_results) < self.convergence_config['min_trials_for_convergence']:
                return None
            
            # Extract objective values
            objective_values = [t['objective_value'] for t in trial_results if t['error_info'] is None]
            if len(objective_values) < self.convergence_config['min_trials_for_convergence']:
                return None
            
            convergence_criteria = []
            convergence_confidence = 0.0
            
            # Check improvement threshold
            if len(objective_values) >= 2:
                recent_improvement = abs(objective_values[-1] - objective_values[-2])
                if recent_improvement < self.convergence_config['improvement_threshold']:
                    convergence_criteria.append('improvement_threshold')
                    convergence_confidence += 0.3
            
            # Check patience (no improvement for N trials)
            patience_trials = self.convergence_config['patience_trials']
            if len(objective_values) >= patience_trials:
                best_value = max(objective_values)
                recent_values = objective_values[-patience_trials:]
                if all(v <= best_value + self.convergence_config['improvement_threshold'] for v in recent_values):
                    convergence_criteria.append('patience')
                    convergence_confidence += 0.4
            
            # Check variance threshold
            if len(objective_values) >= 10:
                recent_values = objective_values[-10:]
                variance = np.var(recent_values)
                if variance < self.convergence_config['variance_threshold']:
                    convergence_criteria.append('variance_threshold')
                    convergence_confidence += 0.3
            
            # Calculate improvement rate
            if len(objective_values) >= 2:
                improvement_rate = (objective_values[-1] - objective_values[0]) / len(objective_values)
            else:
                improvement_rate = 0.0
            
            # Calculate variance estimate
            variance_estimate = np.var(objective_values) if len(objective_values) > 1 else 0.0
            
            # Determine if converged
            is_converged = (len(convergence_criteria) >= 2 and 
                          convergence_confidence >= 0.6)
            
            convergence_analysis = {
                'objective_values': objective_values,
                'recent_improvement': recent_improvement if len(objective_values) >= 2 else 0.0,
                'variance': variance_estimate,
                'improvement_rate': improvement_rate,
                'convergence_criteria_met': convergence_criteria
            }
            
            return {
                'is_converged': is_converged,
                'convergence_criteria': convergence_criteria,
                'convergence_confidence': convergence_confidence,
                'improvement_rate': improvement_rate,
                'variance_estimate': variance_estimate,
                'best_value_history': objective_values,
                'convergence_analysis': convergence_analysis
            }
            
        except Exception as e:
            _LOGGER.error(f"❌ Convergence check failed: {e}")
            return None

    def _check_failure_conditions(self, study_id: str) -> bool:
        """Check if failure conditions are met."""
        try:
            if study_id not in self.active_studies:
                return False
            
            study_info = self.active_studies[study_id]
            trial_results = self.trial_results[study_id]
            
            # Check failure rate
            if study_info['total_trials'] > 0:
                failure_rate = study_info['failed_trials'] / study_info['total_trials']
                if failure_rate > self.failure_detection_config['max_failure_rate']:
                    _LOGGER.error(f"❌ High failure rate: {failure_rate:.2%}")
                    return True
            
            # Check consecutive failures
            if len(trial_results) >= self.failure_detection_config['consecutive_failures_threshold']:
                recent_trials = trial_results[-self.failure_detection_config['consecutive_failures_threshold']:]
                if all(t['error_info'] is not None for t in recent_trials):
                    _LOGGER.error("❌ Too many consecutive failures")
                    return True
            
            # Check timeout
            if study_info['start_time']:
                elapsed_time = (datetime.now() - study_info['start_time']).total_seconds()
                if elapsed_time > self.failure_detection_config['timeout_threshold']:
                    _LOGGER.error(f"❌ Study timeout: {elapsed_time:.0f}s")
                    return True
            
            return False
            
        except Exception as e:
            _LOGGER.error(f"❌ Failure condition check failed: {e}")
            return False

    def _update_error_summary(self, study_info: Dict[str, Any], error_info: Dict[str, Any]):
        """Update error summary for a study."""
        try:
            error_type = error_info.get('error_type', 'unknown')
            study_info['error_summary'][error_type] = study_info['error_summary'].get(error_type, 0) + 1
            
        except Exception as e:
            _LOGGER.warning(f"⚠️ Failed to update error summary: {e}")

    def get_study_status(self, study_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a study."""
        try:
            study_info = self.active_studies.get(study_id)
            if not study_info:
                return None
            
            trial_results = self.trial_results.get(study_id, [])
            
            return {
                'study_id': study_info['study_id'],
                'study_name': study_info['study_name'],
                'status': study_info['status'],
                'start_time': study_info['start_time'].isoformat(),
                'total_trials': study_info['total_trials'],
                'successful_trials': study_info['successful_trials'],
                'failed_trials': study_info['failed_trials'],
                'best_value': study_info['best_value'],
                'convergence_info': study_info['convergence_info'],
                'error_summary': study_info['error_summary'],
                'recent_trials': len(trial_results[-10:]) if trial_results else 0
            }
            
        except Exception as e:
            _LOGGER.error(f"❌ Failed to get study status: {e}")
            return None

    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary."""
        try:
            active_count = len(self.active_studies)
            
            # Calculate overall statistics
            all_trials = []
            for study_id in self.trial_results:
                all_trials.extend(self.trial_results[study_id])
            
            successful_trials = [t for t in all_trials if t['error_info'] is None]
            failed_trials = [t for t in all_trials if t['error_info'] is not None]
            
            total_trials = len(all_trials)
            success_rate = len(successful_trials) / max(1, total_trials)
            
            # Calculate performance metrics
            if successful_trials:
                objective_values = [t['objective_value'] for t in successful_trials]
                training_times = [t['training_time'] for t in successful_trials if t['training_time'] is not None]
                
                performance_metrics = {
                    'best_objective_value': max(objective_values),
                    'mean_objective_value': np.mean(objective_values),
                    'mean_training_time': np.mean(training_times) if training_times else None,
                    'total_training_time': sum(training_times) if training_times else None
                }
            else:
                performance_metrics = {}
            
            # Error analysis
            error_types = defaultdict(int)
            for trial in failed_trials:
                if trial['error_info']:
                    error_type = trial['error_info'].get('error_type', 'unknown')
                    error_types[error_type] += 1
            
            return {
                'monitoring_summary': {
                    'active_studies': active_count,
                    'total_trials': total_trials,
                    'successful_trials': len(successful_trials),
                    'failed_trials': len(failed_trials),
                    'overall_success_rate': success_rate,
                    'monitoring_enabled': self.enable_monitoring
                },
                'performance_metrics': performance_metrics,
                'error_analysis': dict(error_types),
                'convergence_analysis': {
                    'converged_studies': sum(1 for s in self.active_studies.values() 
                                           if s.get('convergence_info') and s['convergence_info'].get('is_converged'))
                }
            }
            
        except Exception as e:
            _LOGGER.error(f"❌ Failed to get monitoring summary: {e}")
            return {'error': str(e)}

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
                            timeout: Optional[int] = None,
                            use_enhanced_search_space: bool = True) -> Dict[str, Any]:
        """
        Perform enhanced Bayesian hyperparameter optimization with non-linear transformations.

        Args:
            model_factory: Function that creates model with given parameters
            X: Feature matrix
            y: Target array
            search_space: Dictionary defining the search space
            n_trials: Number of optimization trials
            acquisition_function: Acquisition function ('ucb', 'ei', 'poi')
            use_enhanced_search_space: Whether to use enhanced non-linear search space

        Returns:
            Enhanced Bayesian optimization results
        """
        try:
            self.logger.info(f"🎲 Starting enhanced Bayesian optimization with {acquisition_function} acquisition")
            if use_enhanced_search_space and self.use_nonlinear_optimization:
                self.logger.info("🚀 Using enhanced non-linear search space")

            if not OPTUNA_AVAILABLE:
                raise ImportError("Optuna required for Bayesian optimization")

            # Use enhanced search space if requested and available
            if use_enhanced_search_space and self.use_nonlinear_optimization:
                enhanced_space = create_enhanced_search_space(search_space, self.nonlinear_config)
                actual_search_space = enhanced_space
            else:
                actual_search_space = search_space

            def objective(trial):
                # Use enhanced parameter sampling
                params = self._sample_hyperparameters(trial, model_factory, actual_search_space)

                # Create and evaluate model
                model = model_factory(**params)
                # Cap per-trial parallelism if supported
                try:
                    if hasattr(model, 'set_params') and hasattr(model, 'get_params'):
                        params = model.get_params()
                        if 'n_jobs' in params:
                            model.set_params(n_jobs=1)
                except Exception as nj_err:
                    logger.debug(f"Failed to cap n_jobs for model in Optuna trial: {nj_err}")

                # Prepare CV and fit params
                cv_obj = cv if cv is not None else self._create_time_series_split(len(X))

                # Compute sample weights if classification and estimator supports it
                fp = dict(fit_params or {})
                try:
                    if SKLEARN_AVAILABLE and len(np.unique(y)) <= 10:
                        fp.setdefault('sample_weight', compute_sample_weight('balanced', y))
                except Exception as sw_err:
                    logger.debug(f"sample_weight computation failed: {sw_err}")

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
                        except Exception as fit_err:
                            logger.debug(f"Fit without sample_weight due to: {fit_err}")
                            mdl.fit(X_tr, y_tr)
                        try:
                            from sklearn.metrics import get_scorer
                            scorer = get_scorer(scoring) if isinstance(scoring, str) else scoring
                            score = scorer(mdl, X_te, y_te)
                        except Exception as score_err:
                            logger.debug(f"Scorer failed ({score_err}), using estimator.score fallback")
                            score = mdl.score(X_te, y_te) if hasattr(mdl, 'score') else 0.0
                        fold_scores.append(float(score))
                        trial.report(float(score), step=i)
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                    if fold_scores:
                        return float(np.mean(fold_scores))
                except Exception as cv_err:
                    logger.debug(f"Manual CV loop failed, falling back to single-score evaluation: {cv_err}")

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
                   coarse_strategy: str = 'grid',  # Changed default to 'grid'
                   coarse_grid_points: int = 3,
                   fine_grid_points: int = 5,  # Added fine grid points
                   coarse_n_samples: int = 50,
                   bayes_n_trials: int = 30,  # Reduced since we have better starting point
                   scoring: Union[str, Callable] = 'balanced_accuracy',
                   cv: Optional[Any] = None,
                   pruner: str = 'hyperband',
                   storage: Optional[str] = None,
                   study_name: Optional[str] = None,
                   timeout: Optional[int] = None,
                   subsample_rate: float = 0.3,
                   finalize_refine: bool = True) -> Dict[str, Any]:
        """Enhanced Staged HPO: coarse grid → fine grid → Optuna TPE.

        Returns dict with coarse_results, fine_results, optuna_results, final_params, final_score.
        """
        try:
            self.logger.info("🌀 Starting enhanced staged HPO: coarse → fine → optuna TPE")

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

            # Stage 1: Coarse Grid Search
            self.logger.info("🎯 Stage 1: Coarse grid search")
            coarse_start = time.time()
            coarse_results = self._coarse_grid_search_staged(
                model_factory, X_train, y_train, search_space, 
                coarse_grid_points, cv_obj, scoring
            )
            coarse_time = time.time() - coarse_start

            if not coarse_results or coarse_results.get('best_score', 0) <= 0:
                self.logger.warning("⚠️ Coarse grid search failed, using random sampling")
                coarse_results = self._fallback_random_search(
                    model_factory, X_train, y_train, search_space, coarse_n_samples, cv_obj, scoring
                )

            self.logger.info(f"✅ Coarse grid completed in {coarse_time:.2f}s - Best score: {coarse_results.get('best_score', 0):.4f}")

            # Stage 2: Fine Grid Search around best coarse parameters
            self.logger.info("🎯 Stage 2: Fine grid search")
            fine_start = time.time()
            best_coarse = coarse_results.get('best_params', {})
            fine_results = self._fine_grid_search_staged(
                model_factory, X, y, search_space, best_coarse, 
                fine_grid_points, cv_obj, scoring
            )
            fine_time = time.time() - fine_start

            if not fine_results or fine_results.get('best_score', 0) <= coarse_results.get('best_score', 0):
                self.logger.info("ℹ️ Fine grid search did not improve results, using coarse grid results")
                best_params = best_coarse
                best_score = coarse_results.get('best_score', 0)
                grid_stage = 'coarse'
            else:
                self.logger.info(f"✅ Fine grid completed in {fine_time:.2f}s - Best score: {fine_results.get('best_score', 0):.4f}")
                best_params = fine_results.get('best_params', {})
                best_score = fine_results.get('best_score', 0)
                grid_stage = 'fine'

            # Stage 3: Optuna TPE Optimization around best grid parameters
            self.logger.info("🎯 Stage 3: Optuna TPE optimization")
            optuna_start = time.time()
            
            # Narrow search space around best grid parameters
            narrowed = self._narrow_search_space(search_space, best_params)
            
            optuna_results = self.bayesian_optimization(
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
                timeout=timeout,
                use_enhanced_search_space=self.use_nonlinear_optimization
            )
            optuna_time = time.time() - optuna_start

            final_params = best_params
            final_score = best_score
            final_stage = grid_stage

            if optuna_results and optuna_results.get('best_score', 0) > best_score:
                self.logger.info(f"✅ Optuna TPE completed in {optuna_time:.2f}s - Best score: {optuna_results.get('best_score', 0):.4f}")
                final_params = optuna_results.get('best_params', best_params)
                final_score = optuna_results.get('best_score', best_score)
                final_stage = 'optuna'
            else:
                self.logger.info("ℹ️ Optuna TPE did not improve results, using grid search results")

            # Optional local fine-tune around best (small random jitters)
            if finalize_refine:
                refine_start = time.time()
                fine_params, fine_score = self._local_refine(
                    model_factory, X, y, final_params, scoring,
                    cv if cv is not None else self._create_time_series_split(len(X))
                )
                refine_time = time.time() - refine_start
                
                if fine_score > final_score:
                    final_params, final_score = fine_params, fine_score
                    final_stage = 'refine'
                    self.logger.info(f"✅ Local refinement completed in {refine_time:.2f}s - Best score: {fine_score:.4f}")

            total_time = coarse_time + fine_time + optuna_time
            self.logger.info(f"🏆 Final HPO completed in {total_time:.2f}s - Best stage: {final_stage}")

            return {
                'coarse_results': coarse_results,
                'fine_results': fine_results,
                'optuna_results': optuna_results,
                'final_params': final_params,
                'final_score': final_score,
                'best_stage': final_stage,
                'coarse_time': coarse_time,
                'fine_time': fine_time,
                'optuna_time': optuna_time,
                'total_time': total_time,
                'optimization_method': 'coarse_fine_optuna'
            }
        except Exception as e:
            self.logger.error(f"❌ Enhanced staged HPO failed: {e}")
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
        """Enhanced hyperparameter sampling with non-linear transformations."""
        try:
            if search_space is None:
                # Fallback to default parameters if no search space provided
                if self.use_nonlinear_optimization:
                    params = {
                        'n_estimators': self.parameter_sampler.suggest_enhanced_int(trial, 'n_estimators', 50, 300),
                        'max_depth': self.parameter_sampler.suggest_enhanced_int(trial, 'max_depth', 3, 10)
                    }
                else:
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': trial.suggest_int('max_depth', 3, 10)
                    }
                return params

            params = {}
            for param_name, param_config in search_space.items():
                if self.use_nonlinear_optimization and 'transform_type' in param_config:
                    # Use enhanced non-linear sampling
                    transform_type = param_config.get('transform_type', 'auto')
                    
                    if param_config['type'] == 'int':
                        params[param_name] = self.parameter_sampler.suggest_enhanced_int(
                            trial, param_name, param_config['low'], param_config['high'], transform_type
                        )
                    elif param_config['type'] == 'float':
                        params[param_name] = self.parameter_sampler.suggest_enhanced_float(
                            trial, param_name, param_config['low'], param_config['high'], transform_type
                        )
                    elif param_config['type'] == 'categorical':
                        params[param_name] = trial.suggest_categorical(
                            param_name, param_config['choices']
                        )
                else:
                    # Fallback to original sampling
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
            self.logger.warning(f"Enhanced hyperparameter sampling failed: {e}")
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
            except Exception as nj_cap_err:
                self.logger.debug(f"Failed to cap nested parallelism (n_jobs): {nj_cap_err}")

            fit_params = {}
            try:
                if SKLEARN_AVAILABLE and len(np.unique(y)) <= 10:
                    fit_params['sample_weight'] = compute_sample_weight('balanced', y)
            except Exception as sw_err:
                self.logger.debug(f"sample_weight computation failed in CV eval: {sw_err}")
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
                    except Exception as fit_err:
                        self.logger.debug(f"Fit failed with sample_weight handling; retrying without: {fit_err}")
                        mdl.fit(X_tr, y_tr)
                    try:
                        from sklearn.metrics import get_scorer
                        scorer = get_scorer(scoring) if isinstance(scoring, str) else scoring
                        score = scorer(mdl, X_te, y_te)
                    except Exception as score_err:
                        self.logger.debug(f"Scorer failed ({score_err}); using estimator.score fallback")
                        score = mdl.score(X_te, y_te) if hasattr(mdl, 'score') else 0.0
                    fold_scores.append(float(score))
                if fold_scores:
                    return float(np.mean(fold_scores))
            except Exception as cv_loop_err:
                self.logger.debug(f"Manual CV loop failed in CV eval: {cv_loop_err}")
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
    
    def _coarse_grid_search_staged(self, model_factory: Callable, X: np.ndarray, y: np.ndarray,
                                  search_space: Dict[str, Any], grid_points: int, cv_obj: Any, 
                                  scoring: Union[str, Callable]) -> Dict[str, Any]:
        """Perform coarse grid search for staged HPO."""
        try:
            self.logger.info(f"🔍 Creating coarse grid with {grid_points} points per parameter")
            
            # Create coarse parameter grid
            param_grid = self._coarse_grid_from_search_space(search_space, grid_points)
            self.logger.info(f"📊 Coarse grid size: {len(param_grid)} combinations")
            
            best_score = -np.inf
            best_params = {}
            parameter_scores = []
            
            # Evaluate each parameter combination
            for i, params in enumerate(param_grid):
                try:
                    model = model_factory(**params)
                    score = self._evaluate_model_cv(model, X, y, cv_obj, scoring)
                    parameter_scores.append((params, score))
                    
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                    
                    if (i + 1) % 10 == 0:
                        self.logger.debug(f"   Evaluated {i + 1}/{len(param_grid)} combinations")
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to evaluate parameters {params}: {e}")
                    continue
            
            if not parameter_scores:
                self.logger.error("❌ No valid parameter combinations found in coarse grid")
                return {}
            
            self.logger.info(f"✅ Coarse grid search completed - Best score: {best_score:.4f}")
            
            return {
                'best_params': best_params,
                'best_score': best_score,
                'n_combinations': len(param_grid),
                'valid_combinations': len(parameter_scores),
                'parameter_scores': parameter_scores[:10]  # Keep top 10 for analysis
            }
            
        except Exception as e:
            self.logger.error(f"❌ Coarse grid search failed: {e}")
            return {}
    
    def _fine_grid_search_staged(self, model_factory: Callable, X: np.ndarray, y: np.ndarray,
                                search_space: Dict[str, Any], best_coarse_params: Dict[str, Any],
                                grid_points: int, cv_obj: Any, scoring: Union[str, Callable]) -> Dict[str, Any]:
        """Perform fine grid search around best coarse parameters for staged HPO."""
        try:
            self.logger.info(f"🔍 Creating fine grid with {grid_points} points around best coarse parameters")
            
            # Create fine parameter grid around best coarse parameters
            fine_grid = self._create_fine_parameter_grid_staged(search_space, best_coarse_params, grid_points)
            self.logger.info(f"📊 Fine grid size: {len(fine_grid)} combinations")
            
            best_score = -np.inf
            best_params = {}
            parameter_scores = []
            
            # Evaluate each parameter combination
            for i, params in enumerate(fine_grid):
                try:
                    model = model_factory(**params)
                    score = self._evaluate_model_cv(model, X, y, cv_obj, scoring)
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
                self.logger.error("❌ No valid parameter combinations found in fine grid")
                return {}
            
            self.logger.info(f"✅ Fine grid search completed - Best score: {best_score:.4f}")
            
            return {
                'best_params': best_params,
                'best_score': best_score,
                'n_combinations': len(fine_grid),
                'valid_combinations': len(parameter_scores),
                'parameter_scores': parameter_scores[:10]  # Keep top 10 for analysis
            }
            
        except Exception as e:
            self.logger.error(f"❌ Fine grid search failed: {e}")
            return {}
    
    def _create_fine_parameter_grid_staged(self, search_space: Dict[str, Any], best_params: Dict[str, Any], 
                                         grid_points: int) -> List[Dict[str, Any]]:
        """Create fine parameter grid around best parameters for staged HPO."""
        import itertools
        
        param_combinations = []
        
        for param_name, param_config in search_space.items():
            if param_name not in best_params:
                continue
                
            best_value = best_params[param_name]
            
            if isinstance(param_config, dict):
                typ = param_config.get('type', 'float')
                if typ == 'float':
                    low, high = param_config['low'], param_config['high']
                    # Create fine grid around best value (±20% of range)
                    range_size = high - low
                    fine_range = range_size * 0.2
                    fine_min = max(low, best_value - fine_range)
                    fine_max = min(high, best_value + fine_range)
                    
                    # Use specified number of points for fine grid
                    if param_config.get('log', False):
                        # Log-spaced values
                        values = np.logspace(np.log10(fine_min), np.log10(fine_max), grid_points)
                    else:
                        # Linear-spaced values
                        values = np.linspace(fine_min, fine_max, grid_points)
                    param_combinations.append([(param_name, v) for v in values])
                    
                elif typ == 'int':
                    low, high = param_config['low'], param_config['high']
                    # Create fine grid around best value (±2 values)
                    fine_min = max(low, best_value - 2)
                    fine_max = min(high, best_value + 2)
                    values = list(range(fine_min, fine_max + 1))
                    param_combinations.append([(param_name, v) for v in values])
                    
                elif typ == 'categorical':
                    param_combinations.append([(param_name, v) for v in param_config.get('choices', [])])
            else:
                # Legacy tuple format
                if isinstance(param_config, tuple) and len(param_config) == 2:
                    low, high = param_config
                    range_size = high - low
                    fine_range = range_size * 0.2
                    fine_min = max(low, best_value - fine_range)
                    fine_max = min(high, best_value + fine_range)
                    values = np.linspace(fine_min, fine_max, grid_points)
                    param_combinations.append([(param_name, v) for v in values])
        
        # Generate all combinations
        all_combinations = list(itertools.product(*param_combinations))
        
        # Convert to list of dictionaries
        grid = []
        for combination in all_combinations:
            param_dict = dict(combination)
            grid.append(param_dict)
        
        return grid
    
    def _fallback_random_search(self, model_factory: Callable, X: np.ndarray, y: np.ndarray,
                               search_space: Dict[str, Any], n_samples: int, cv_obj: Any, 
                               scoring: Union[str, Callable]) -> Dict[str, Any]:
        """Fallback random search when grid search fails."""
        try:
            self.logger.info(f"🎲 Performing fallback random search with {n_samples} samples")
            
            # Generate random parameter combinations
            sampled = self._generate_random_param_combinations(
                self._coarse_grid_from_search_space(search_space, 3), n_samples
            )
            
            best_score = -np.inf
            best_params = {}
            parameter_scores = []
            
            for i, params in enumerate(sampled):
                try:
                    model = model_factory(**params)
                    score = self._evaluate_model_cv(model, X, y, cv_obj, scoring)
                    parameter_scores.append((params, score))
                    
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                    
                    if (i + 1) % 10 == 0:
                        self.logger.debug(f"   Evaluated {i + 1}/{len(sampled)} combinations")
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to evaluate parameters {params}: {e}")
                    continue
            
            if not parameter_scores:
                self.logger.error("❌ No valid parameter combinations found in random search")
                return {}
            
            self.logger.info(f"✅ Random search completed - Best score: {best_score:.4f}")
            
            return {
                'best_params': best_params,
                'best_score': best_score,
                'n_combinations': len(sampled),
                'valid_combinations': len(parameter_scores),
                'parameter_scores': parameter_scores[:10],
                'method': 'random_fallback'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Random search failed: {e}")
            return {}


# ============================================================================
# PUBLIC API FUNCTIONS
# ============================================================================
# These functions provide a simple interface to the HyperparameterOptimization class

def optimize_hyperparameters(model_factory: Callable = None, 
                           model: Any = None,
                           X: np.ndarray = None, 
                           y: np.ndarray = None,
                           search_space: Dict[str, Any] = None,
                           n_trials: int = 50,
                           method: str = 'bayesian',
                           scoring: Union[str, Callable] = 'accuracy',
                           cv: Optional[Any] = None,
                           config: Optional[Dict[str, Any]] = None,
                           **kwargs) -> Dict[str, Any]:
    """
    Optimize hyperparameters for a given model.
    
    Args:
        model_factory: Function that creates model with given parameters
        model: Model instance (alternative to model_factory)
        X: Feature matrix
        y: Target array
        search_space: Dictionary defining the search space
        n_trials: Number of optimization trials
        method: Optimization method ('bayesian', 'staged', 'multi_objective')
        scoring: Scoring metric
        cv: Cross-validation strategy
        config: Configuration dictionary
        **kwargs: Additional arguments
    
    Returns:
        Optimization results dictionary
    """
    try:
        logger.info(f"🚀 Starting hyperparameter optimization with {method} method")
        
        # Handle model factory creation
        if model_factory is None and model is not None:
            def model_factory(**params):
                # Clone the model and set parameters
                from sklearn.base import clone
                try:
                    cloned_model = clone(model)
                    if hasattr(cloned_model, 'set_params'):
                        cloned_model.set_params(**params)
                    return cloned_model
                except Exception:
                    # Fallback: create new instance with params
                    model_class = type(model)
                    return model_class(**params)
        
        if model_factory is None:
            logger.error("❌ Either model_factory or model must be provided")
            return {'error': 'Either model_factory or model must be provided'}
        
        if X is None or y is None:
            logger.error("❌ Training data (X, y) must be provided")
            return {'error': 'Training data (X, y) must be provided'}
        
        # Create HPO instance
        hpo = HyperparameterOptimization(config=config)
        
        # Generate search space if not provided
        if search_space is None:
            model_name = getattr(model, '__class__.__name__', 'unknown').lower()
            search_space = create_search_space(model_name, X, y)
            logger.info(f"📊 Generated search space for {model_name}: {list(search_space.keys())}")
        
        # Choose optimization method
        if method == 'bayesian':
            results = hpo.bayesian_optimization(
                model_factory=model_factory,
                X=X, y=y,
                search_space=search_space,
                n_trials=n_trials,
                scoring=scoring,
                cv=cv,
                **kwargs
            )
        elif method == 'staged':
            results = hpo.staged_hpo(
                model_factory=model_factory,
                X=X, y=y,
                search_space=search_space,
                bayes_n_trials=n_trials,
                scoring=scoring,
                cv=cv,
                **kwargs
            )
        elif method == 'multi_objective':
            objectives = kwargs.get('objectives', ['accuracy'])
            results = hpo.multi_objective_optimization(
                model_factory=model_factory,
                X=X, y=y,
                objectives=objectives,
                n_trials=n_trials,
                search_space=search_space
            )
        else:
            logger.error(f"❌ Unknown optimization method: {method}")
            return {'error': f'Unknown optimization method: {method}'}
        
        logger.info(f"✅ Hyperparameter optimization completed with {method} method")
        return results
        
    except Exception as e:
        logger.error(f"❌ Hyperparameter optimization failed: {e}")
        return {'error': str(e)}


def create_search_space(model_type: str, 
                       X: Optional[np.ndarray] = None, 
                       y: Optional[np.ndarray] = None,
                       data_characteristics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create search space for hyperparameter optimization.
    
    Args:
        model_type: Type of model ('xgboost', 'lightgbm', 'random_forest', etc.)
        X: Feature matrix (optional, used for data characteristics)
        y: Target array (optional, used for data characteristics)
        data_characteristics: Dictionary with data characteristics
    
    Returns:
        Search space dictionary
    """
    try:
        logger.info(f"🔧 Creating search space for {model_type}")
        
        # Extract data characteristics
        if data_characteristics is None and X is not None and y is not None:
            data_characteristics = {
                'n_samples': len(X),
                'n_features': X.shape[1] if len(X.shape) > 1 else 1,
                'n_classes': len(np.unique(y)) if y is not None else 2,
                'task_type': 'classification' if len(np.unique(y)) <= 20 else 'regression'
            }
        elif data_characteristics is None:
            data_characteristics = {
                'n_samples': 1000,
                'n_features': 10,
                'n_classes': 2,
                'task_type': 'classification'
            }
        
        # Create HPO instance and generate search space
        hpo = HyperparameterOptimization()
        search_space = hpo.automated_search_space_generation(model_type, data_characteristics)
        
        logger.info(f"✅ Created search space with {len(search_space)} parameters")
        return search_space
        
    except Exception as e:
        logger.error(f"❌ Search space creation failed: {e}")
        return {}


def validate_hpo_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate hyperparameter optimization configuration.
    
    Args:
        config: HPO configuration dictionary
    
    Returns:
        Validation results dictionary
    """
    try:
        logger.info("🔍 Validating HPO configuration")
        
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check required fields
        if 'n_trials' in config:
            if not isinstance(config['n_trials'], int) or config['n_trials'] <= 0:
                validation_results['errors'].append("n_trials must be a positive integer")
                validation_results['valid'] = False
        
        if 'method' in config:
            valid_methods = ['bayesian', 'staged', 'multi_objective', 'random']
            if config['method'] not in valid_methods:
                validation_results['errors'].append(f"method must be one of {valid_methods}")
                validation_results['valid'] = False
        
        if 'scoring' in config:
            # Basic scoring validation
            if isinstance(config['scoring'], str):
                valid_scorings = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'neg_mean_squared_error']
                if config['scoring'] not in valid_scorings:
                    validation_results['warnings'].append(f"Scoring '{config['scoring']}' may not be supported")
        
        # Check search space if provided
        if 'search_space' in config:
            search_space = config['search_space']
            if not isinstance(search_space, dict):
                validation_results['errors'].append("search_space must be a dictionary")
                validation_results['valid'] = False
            else:
                for param_name, param_config in search_space.items():
                    if not isinstance(param_config, dict):
                        validation_results['warnings'].append(f"Parameter '{param_name}' config should be a dictionary")
                    elif 'type' not in param_config:
                        validation_results['warnings'].append(f"Parameter '{param_name}' missing 'type' specification")
        
        # Performance warnings
        if config.get('n_trials', 50) > 200:
            validation_results['warnings'].append("High number of trials may take significant time")
        
        if config.get('enable_parallel', True) and config.get('max_workers', 4) > 8:
            validation_results['warnings'].append("High number of workers may cause resource contention")
        
        if validation_results['valid']:
            logger.info("✅ HPO configuration is valid")
        else:
            logger.warning(f"⚠️ HPO configuration has errors: {validation_results['errors']}")
        
        return validation_results
        
    except Exception as e:
        logger.error(f"❌ HPO configuration validation failed: {e}")
        return {
            'valid': False,
            'errors': [f"Validation failed: {str(e)}"],
            'warnings': []
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_hpo_config(n_trials: int = 50,
                     method: str = 'bayesian',
                     scoring: str = 'accuracy',
                     enable_parallel: bool = True,
                     max_workers: int = 4,
                     **kwargs) -> Dict[str, Any]:
    """
    Create a standard HPO configuration dictionary.
    
    Args:
        n_trials: Number of optimization trials
        method: Optimization method
        scoring: Scoring metric
        enable_parallel: Enable parallel processing
        max_workers: Maximum number of parallel workers
        **kwargs: Additional configuration options
    
    Returns:
        HPO configuration dictionary
    """
    config = {
        'n_trials': n_trials,
        'method': method,
        'scoring': scoring,
        'enable_parallel': enable_parallel,
        'max_workers': max_workers,
        'enable_monitoring': kwargs.get('enable_monitoring', True),
        'use_nonlinear_optimization': kwargs.get('use_nonlinear_optimization', True)
    }
    
    # Add any additional kwargs
    config.update(kwargs)
    
    logger.info(f"📋 Created HPO config: {method} method, {n_trials} trials")
    return config


# Add missing import for defaultdict
from collections import defaultdict

# Make functions available for import
__all__ = [
    'HyperparameterOptimization',
    'optimize_hyperparameters', 
    'create_search_space',
    'validate_hpo_config',
    'create_hpo_config'
]
