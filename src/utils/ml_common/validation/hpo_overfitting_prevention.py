"""
HPO with Overfitting Prevention for ML Common

Safe hyperparameter optimization with built-in safeguards, nested cross-validation,
regularization parameter tuning, and overfitting prevention.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from pathlib import Path
import json
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, f1_score
import optuna
import warnings

logger = logging.getLogger(__name__)

@dataclass
class HPOOverfittingPreventionConfig:
    """Configuration for HPO with overfitting prevention."""

    # Basic HPO settings
    n_trials: int = 100
    timeout_minutes: int = 60
    enable_pruning: bool = True
    pruner_type: str = "median"  # median, hyperband, successiveshalving
    sampler_type: str = "tpe"  # tpe, random, cmaes

    # Grid search + Bayesian TPE integration
    enable_staged_hpo: bool = True
    coarse_strategy: str = "grid"  # grid, random
    coarse_grid_points: int = 3
    fine_grid_points: int = 5
    coarse_n_samples: int = 50
    bayes_n_trials: int = 30  # Bayesian TPE trials after grid search
    finalize_refine: bool = True

    # Nested cross-validation
    enable_nested_cv: bool = True
    outer_cv_folds: int = 5
    inner_cv_folds: int = 3
    nested_cv_random_state: int = 42

    # Regularization tuning
    enable_regularization_tuning: bool = True
    regularization_methods: List[str] = None  # l1, l2, dropout, early_stopping
    regularization_ranges: Dict[str, Tuple[float, float]] = None

    # Overfitting prevention
    enable_overfitting_prevention: bool = True
    overfitting_check_interval: int = 10
    max_overfitting_trials: int = 3
    overfitting_threshold: float = 0.1

    # Safety constraints
    enable_safety_constraints: bool = True
    max_model_complexity: float = 1.0
    min_training_stability: float = 0.8
    max_parameter_range: float = 10.0

    # Evaluation
    primary_metric: str = "accuracy"
    additional_metrics: List[str] = None
    evaluation_strategy: str = "nested_cv"

    # Reporting
    save_hpo_reports: bool = True
    report_directory: str = "reports/hpo_overfitting_prevention"
    enable_detailed_logging: bool = True

    def __post_init__(self):
        """Initialize default values."""
        if self.regularization_methods is None:
            self.regularization_methods = ["l2", "dropout", "early_stopping"]
        if self.regularization_ranges is None:
            self.regularization_ranges = {
                "l2": (1e-5, 1e-2),
                "dropout": (0.1, 0.5),
                "early_stopping_patience": (5, 20)
            }
        if self.additional_metrics is None:
            self.additional_metrics = ["f1", "precision", "recall"]

@dataclass
class HPOTrialReport:
    """Report for a single HPO trial with overfitting analysis."""

    # Basic trial information
    trial_number: int = 0
    trial_params: Dict[str, Any] = None
    model_type: str = "unknown"
    training_time: float = 0.0

    # Performance metrics
    primary_score: float = 0.0
    additional_scores: Dict[str, float] = None
    cv_scores: List[float] = None
    cv_mean: float = 0.0
    cv_std: float = 0.0

    # Overfitting analysis
    overfitting_detected: bool = False
    overfitting_severity: str = "none"
    train_val_gap: float = 0.0
    regularization_strength: float = 0.0

    # Safety checks
    safety_violations: List[str] = None
    complexity_score: float = 0.0
    stability_score: float = 0.0

    # Trial status
    trial_status: str = "completed"  # completed, failed, pruned, safety_violated
    error_message: str = ""
    pruning_reason: str = ""

    # Metadata
    trial_timestamp: str = None
    config_used: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize default collections."""
        if self.trial_params is None:
            self.trial_params = {}
        if self.additional_scores is None:
            self.additional_scores = {}
        if self.cv_scores is None:
            self.cv_scores = []
        if self.safety_violations is None:
            self.safety_violations = []
        if self.trial_timestamp is None:
            self.trial_timestamp = datetime.now().isoformat()
        if self.config_used is None:
            self.config_used = {}

@dataclass
class HPOOptimizationReport:
    """Comprehensive HPO optimization report with overfitting prevention analysis."""

    # Basic information
    optimization_id: str = None
    model_type: str = "unknown"
    dataset_name: str = "unknown"
    start_time: str = None
    end_time: str = None
    total_duration: float = 0.0

    # Optimization results
    best_params: Dict[str, Any] = None
    best_score: float = 0.0
    best_trial_number: int = 0
    total_trials: int = 0
    successful_trials: int = 0
    failed_trials: int = 0
    pruned_trials: int = 0

    # Overfitting prevention results
    overfitting_prevention_enabled: bool = True
    overfitting_trials_detected: int = 0
    safety_violations_detected: int = 0
    regularization_effectiveness: float = 0.0

    # Performance analysis
    score_progression: List[float] = None
    overfitting_progression: List[bool] = None
    complexity_progression: List[float] = None

    # Final model assessment
    final_model_score: float = 0.0
    final_model_stability: float = 0.0
    final_model_robustness: float = 0.0
    final_overfitting_risk: str = "low"

    # Staged HPO metrics
    staged_hpo_metrics: Dict[str, Any] = None

    # Recommendations
    recommendations: List[str] = None
    warnings: List[str] = None
    optimization_quality: str = "unknown"

    def __post_init__(self):
        """Initialize default collections."""
        if self.score_progression is None:
            self.score_progression = []
        if self.overfitting_progression is None:
            self.overfitting_progression = []
        if self.complexity_progression is None:
            self.complexity_progression = []
        if self.recommendations is None:
            self.recommendations = []
        if self.warnings is None:
            self.warnings = []
        if self.optimization_id is None:
            self.optimization_id = datetime.now().strftime("%Y%m%d_%H%M%S")

class HPOWithOverfittingPrevention:
    """HPO system with built-in overfitting prevention and safety safeguards."""

    def __init__(self, config: Optional[HPOOverfittingPreventionConfig] = None):
        """
        Initialize HPO with overfitting prevention.

        Args:
            config: Configuration for HPO
        """
        self.config = config or HPOOverfittingPreventionConfig()
        self.optimization_history = []
        self.active_optimizations = {}

        # Create report directory
        if self.config.save_hpo_reports:
            Path(self.config.report_directory).mkdir(parents=True, exist_ok=True)

        logger.info("✅ HPO with Overfitting Prevention initialized")

    def optimize_hyperparameters(self,
                                model_class: Any,
                                X: np.ndarray,
                                y: np.ndarray,
                                model_name: str = "model",
                                model_type: str = "unknown",
                                param_space: Optional[Dict[str, Any]] = None,
                                is_classification: bool = True,
                                random_state: int = 42) -> HPOOptimizationReport:
        """
        Perform safe hyperparameter optimization with overfitting prevention.

        Args:
            model_class: Model class to optimize
            X: Feature matrix
            y: Target vector
            model_name: Name for the model
            model_type: Type of model
            param_space: Parameter search space
            is_classification: Whether it's classification
            random_state: Random state for reproducibility

        Returns:
            HPOOptimizationReport with optimization results
        """
        report = HPOOptimizationReport(
            model_type=model_type,
            dataset_name=f"{model_name}_dataset"
        )

        try:
            # Generate optimization ID
            optimization_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            report.optimization_id = optimization_id

            # Store in active optimizations
            self.active_optimizations[optimization_id] = {
                'start_time': datetime.now(),
                'trials': [],
                'best_score': 0.0,
                'best_params': {}
            }

            # Create parameter space with regularization
            if param_space is None:
                param_space = self._create_default_param_space(model_type)

            # Add regularization parameters if enabled
            if self.config.enable_regularization_tuning:
                param_space = self._add_regularization_params(param_space, model_type)

            # Create study with appropriate pruner
            study = self._create_optimization_study()

            # Define objective function with safety checks
            def objective(trial):
                return self._safe_objective_function(
                    trial, model_class, X, y, is_classification,
                    optimization_id, model_type, random_state
                )

            # Optimize
            logger.info(f"🚀 Starting HPO for {model_name} with {self.config.n_trials} trials")
            study.optimize(objective, n_trials=self.config.n_trials)

            # Extract results
            report.best_params = study.best_params
            report.best_score = study.best_value
            report.best_trial_number = study.best_trial.number
            report.total_trials = len(study.trials)

            # Count trial outcomes
            report.successful_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            report.failed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
            report.pruned_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])

            # Analyze optimization results
            report = self._analyze_optimization_results(report, study, optimization_id)

            # Final model evaluation
            report = self._evaluate_final_model(report, model_class, X, y, is_classification, random_state)

            # Generate recommendations
            report = self._generate_hpo_recommendations(report)

            # Store report
            self.optimization_history.append(report)

            # Clean up active optimizations
            if optimization_id in self.active_optimizations:
                del self.active_optimizations[optimization_id]

            # Log results
            self._log_hpo_report(report)

            return report

        except Exception as e:
            logger.error(f"HPO failed: {e}")
            report.warnings.append(f"Optimization failed: {str(e)}")
            report.optimization_quality = "failed"
            return report

    def optimize_with_staged_hpo(self,
                                model_class: Any,
                                X: np.ndarray,
                                y: np.ndarray,
                                model_name: str = "model",
                                model_type: str = "unknown",
                                param_space: Optional[Dict[str, Any]] = None,
                                is_classification: bool = True,
                                random_state: int = 42) -> HPOOptimizationReport:
        """
        Perform staged HPO: Grid Search -> Fine Grid -> Bayesian TPE.

        Args:
            model_class: Model class to optimize
            X: Feature matrix
            y: Target vector
            model_name: Name for the model
            model_type: Type of model
            param_space: Parameter search space
            is_classification: Whether it's classification
            random_state: Random state for reproducibility

        Returns:
            HPOOptimizationReport with optimization results
        """
        if not self.config.enable_staged_hpo:
            # Fallback to regular HPO
            return self.optimize_hyperparameters(
                model_class, X, y, model_name, model_type, param_space, is_classification, random_state
            )

        try:
            from ..optimization.hpo_utils import StagedHPO

            # Initialize staged HPO
            staged_hpo = StagedHPO()

            # Use default param space if not provided
            if param_space is None:
                param_space = self._create_default_param_space(model_type)

            # Create model factory
            def model_factory(**params):
                return model_class(**params)

            # Run staged HPO
            staged_results = staged_hpo.staged_hpo(
                model_factory=model_factory,
                X=X,
                y=y,
                search_space=param_space,
                coarse_strategy=self.config.coarse_strategy,
                coarse_grid_points=self.config.coarse_grid_points,
                fine_grid_points=self.config.fine_grid_points,
                coarse_n_samples=self.config.coarse_n_samples,
                bayes_n_trials=self.config.bayes_n_trials,
                scoring='accuracy' if is_classification else 'r2',
                cv=None,  # Will use time series split
                pruner='hyperband',
                finalize_refine=self.config.finalize_refine
            )

            # Create report from staged results
            report = HPOOptimizationReport(
                model_type=model_type,
                dataset_name=f"{model_name}_dataset",
                optimization_id=f"staged_hpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                best_params=staged_results.get('best_params', {}),
                best_score=staged_results.get('best_score', 0.0),
                total_trials=staged_results.get('total_trials', 0),
                successful_trials=staged_results.get('successful_trials', 0),
                failed_trials=staged_results.get('failed_trials', 0),
                pruned_trials=staged_results.get('pruned_trials', 0),
                optimization_time=staged_results.get('optimization_time', 0.0),
                final_overfitting_risk=staged_results.get('overfitting_risk', 'unknown')
            )

            # Add staged HPO specific metrics
            report.staged_hpo_metrics = {
                'coarse_grid_score': staged_results.get('coarse_grid_score', 0.0),
                'fine_grid_score': staged_results.get('fine_grid_score', 0.0),
                'bayesian_score': staged_results.get('bayesian_score', 0.0),
                'grid_stage': staged_results.get('grid_stage', 'unknown'),
                'final_stage': staged_results.get('final_stage', 'unknown')
            }

            logger.info(f"✅ Staged HPO completed for {model_name}")
            logger.info(f"📊 Best score: {report.best_score:.4f}")
            logger.info(f"📊 Final stage: {report.staged_hpo_metrics.get('final_stage', 'unknown')}")

            return report

        except Exception as e:
            logger.error(f"❌ Staged HPO failed: {e}")
            # Fallback to regular HPO
            return self.optimize_hyperparameters(
                model_class, X, y, model_name, model_type, param_space, is_classification, random_state
            )

    def _create_default_param_space(self, model_type: str) -> Dict[str, Any]:
        """Create default parameter space for common model types."""
        param_spaces = {
            'xgboost': {
                'n_estimators': optuna.distributions.IntDistribution(50, 500),
                'max_depth': optuna.distributions.IntDistribution(3, 10),
                'learning_rate': optuna.distributions.LogUniformDistribution(0.01, 0.3),
                'subsample': optuna.distributions.UniformDistribution(0.6, 1.0),
                'colsample_bytree': optuna.distributions.UniformDistribution(0.6, 1.0),
                'reg_alpha': optuna.distributions.LogUniformDistribution(1e-5, 1.0),
                'reg_lambda': optuna.distributions.LogUniformDistribution(1e-5, 1.0)
            },
            'lightgbm': {
                'n_estimators': optuna.distributions.IntDistribution(50, 500),
                'max_depth': optuna.distributions.IntDistribution(3, 10),
                'learning_rate': optuna.distributions.LogUniformDistribution(0.01, 0.3),
                'subsample': optuna.distributions.UniformDistribution(0.6, 1.0),
                'colsample_bytree': optuna.distributions.UniformDistribution(0.6, 1.0),
                'reg_alpha': optuna.distributions.LogUniformDistribution(1e-5, 1.0),
                'reg_lambda': optuna.distributions.LogUniformDistribution(1e-5, 1.0)
            },
            'catboost': {
                'iterations': optuna.distributions.IntDistribution(50, 500),
                'depth': optuna.distributions.IntDistribution(3, 10),
                'learning_rate': optuna.distributions.LogUniformDistribution(0.01, 0.3),
                'l2_leaf_reg': optuna.distributions.LogUniformDistribution(1, 10),
                'subsample': optuna.distributions.UniformDistribution(0.6, 1.0),
                'colsample_bylevel': optuna.distributions.UniformDistribution(0.6, 1.0)
            },
            'random_forest': {
                'n_estimators': optuna.distributions.IntDistribution(50, 300),
                'max_depth': optuna.distributions.IntDistribution(5, 20),
                'min_samples_split': optuna.distributions.IntDistribution(2, 10),
                'min_samples_leaf': optuna.distributions.IntDistribution(1, 5),
                'max_features': optuna.distributions.CategoricalDistribution(['sqrt', 'log2'])
            }
        }

        return param_spaces.get(model_type.lower(), param_spaces['xgboost'])

    def _add_regularization_params(self, param_space: Dict[str, Any], model_type: str) -> Dict[str, Any]:
        """Add regularization parameters to search space."""
        if not self.config.enable_regularization_tuning:
            return param_space

        regularization_params = {}

        for method in self.config.regularization_methods:
            if method == "l2" and "reg_lambda" not in param_space:
                regularization_params["reg_lambda"] = optuna.distributions.LogUniformDistribution(
                    self.config.regularization_ranges["l2"][0],
                    self.config.regularization_ranges["l2"][1]
                )
            elif method == "dropout":
                # Add dropout for neural networks or tree-based models with dropout support
                if model_type.lower() in ['xgboost', 'lightgbm']:
                    regularization_params["dropout"] = optuna.distributions.UniformDistribution(
                        self.config.regularization_ranges["dropout"][0],
                        self.config.regularization_ranges["dropout"][1]
                    )

        param_space.update(regularization_params)
        return param_space

    def _create_optimization_study(self) -> optuna.Study:
        """Create Optuna study with appropriate pruner and sampler."""
        # Create pruner
        if self.config.pruner_type == "median":
            pruner = optuna.pruners.MedianPruner(n_startup_trials=10)
        elif self.config.pruner_type == "hyperband":
            pruner = optuna.pruners.HyperbandPruner()
        elif self.config.pruner_type == "successiveshalving":
            pruner = optuna.pruners.SuccessiveHalvingPruner()
        else:
            pruner = optuna.pruners.MedianPruner()

        # Create sampler
        if self.config.sampler_type == "tpe":
            sampler = optuna.samplers.TPESampler(seed=self.config.nested_cv_random_state)
        elif self.config.sampler_type == "cmaes":
            sampler = optuna.samplers.CmaEsSampler(seed=self.config.nested_cv_random_state)
        else:
            sampler = optuna.samplers.RandomSampler(seed=self.config.nested_cv_random_state)

        return optuna.create_study(direction="maximize", pruner=pruner, sampler=sampler)

    def _safe_objective_function(self,
                                trial: optuna.Trial,
                                model_class: Any,
                                X: np.ndarray,
                                y: np.ndarray,
                                is_classification: bool,
                                optimization_id: str,
                                model_type: str,
                                random_state: int) -> float:
        """Safe objective function with overfitting prevention."""
        try:
            # Get trial parameters
            trial_params = self._sample_trial_params(trial, model_class, model_type)

            # Create model with trial parameters
            model = model_class(**trial_params)

            # Perform nested cross-validation
            if self.config.enable_nested_cv:
                score = self._nested_cross_validation(
                    model, X, y, is_classification, random_state
                )
            else:
                score = self._simple_cross_validation(
                    model, X, y, is_classification, random_state
                )

            # Check for overfitting
            if self.config.enable_overfitting_prevention:
                overfitting_check = self._check_trial_overfitting(
                    trial.number, model, X, y, is_classification
                )

                if overfitting_check['overfitting_detected']:
                    # Reduce score for overfitting models
                    score *= (1 - self.config.overfitting_threshold)

                    # Store overfitting information
                    if optimization_id in self.active_optimizations:
                        self.active_optimizations[optimization_id]['trials'].append({
                            'trial_number': trial.number,
                            'overfitting_detected': True,
                            'overfitting_severity': overfitting_check['severity']
                        })

            # Safety checks
            if self.config.enable_safety_constraints:
                safety_violations = self._check_safety_constraints(trial_params, model_type)

                if safety_violations:
                    # Reduce score for unsafe parameters
                    score *= 0.5

                    if optimization_id in self.active_optimizations:
                        self.active_optimizations[optimization_id]['trials'].append({
                            'trial_number': trial.number,
                            'safety_violations': safety_violations
                        })

            # Store trial result
            if optimization_id in self.active_optimizations:
                self.active_optimizations[optimization_id]['trials'].append({
                    'trial_number': trial.number,
                    'score': score,
                    'params': trial_params
                })

            return score

        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}")
            return 0.0  # Return poor score for failed trials

    def _sample_trial_params(self, trial: optuna.Trial, model_class: Any, model_type: str) -> Dict[str, Any]:
        """Sample parameters for trial."""
        # This would need to be implemented based on the specific model type
        # For now, return empty dict - would need model-specific parameter sampling
        return {}

    def _nested_cross_validation(self,
                                model: Any,
                                X: np.ndarray,
                                y: np.ndarray,
                                is_classification: bool,
                                random_state: int) -> float:
        """Perform nested cross-validation via unified CV API."""
        try:
            from src.utils.ml_common.validation.unified_cv import nested_cross_validation as unified_nested
            score = unified_nested(
                model,
                X,
                y,
                outer_folds=self.config.outer_cv_folds,
                inner_folds=self.config.inner_cv_folds,
                scoring='accuracy' if is_classification else 'r2',
                random_state=random_state,
                stratified=is_classification,
            )
            return float(score)
        except Exception as e:
            logger.error(f"Nested CV failed: {e}")
            return 0.0

    def _simple_cross_validation(self,
                                model: Any,
                                X: np.ndarray,
                                y: np.ndarray,
                                is_classification: bool,
                                random_state: int) -> float:
        """Perform simple cross-validation via unified CV API."""
        try:
            from src.utils.ml_common.validation.unified_cv import perform_cross_validation as unified_perform_cv
            result = unified_perform_cv(
                model,
                X,
                y,
                strategy='standard',
                cv_folds=self.config.outer_cv_folds,
                scoring='accuracy' if is_classification else 'r2',
                random_state=random_state,
                stratified=is_classification,
            )
            mean_val = result.get('mean')
            if mean_val is None:
                scores = result.get('scores', []) or []
                return float(np.mean(scores)) if len(scores) else 0.0
            return float(mean_val)
        except Exception as e:
            logger.error(f"Simple CV failed: {e}")
            return 0.0

    def _check_trial_overfitting(self,
                                trial_number: int,
                                model: Any,
                                X: np.ndarray,
                                y: np.ndarray,
                                is_classification: bool) -> Dict[str, Any]:
        """Check if trial exhibits overfitting."""
        try:
            # Split data for overfitting check
            # ⚠️ CRITICAL: For time series data, ALWAYS use TimeSeriesSplit!
            from sklearn.model_selection import TimeSeriesSplit
            cv = TimeSeriesSplit(n_splits=3)

            train_scores = []
            val_scores = []

            for train_idx, val_idx in cv.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Train model
                trial_model = type(model)()
                trial_model.fit(X_train, y_train)

                # Calculate scores
                if is_classification:
                    train_pred = trial_model.predict(X_train)
                    val_pred = trial_model.predict(X_val)
                    train_score = accuracy_score(y_train, train_pred)
                    val_score = accuracy_score(y_val, val_pred)
                else:
                    train_pred = trial_model.predict(X_train)
                    val_pred = trial_model.predict(X_val)
                    train_score = 1 - np.mean((y_train - train_pred) ** 2) / np.var(y_train)
                    val_score = 1 - np.mean((y_val - val_pred) ** 2) / np.var(y_val)

                train_scores.append(train_score)
                val_scores.append(val_score)

            train_mean = np.mean(train_scores)
            val_mean = np.mean(val_scores)
            gap = train_mean - val_mean

            result = {
                'overfitting_detected': gap > self.config.overfitting_threshold,
                'severity': 'high' if gap > 0.2 else 'medium' if gap > 0.1 else 'low',
                'train_val_gap': gap
            }

            return result

        except Exception as e:
            logger.error(f"Overfitting check failed: {e}")
            return {
                'overfitting_detected': False,
                'severity': 'unknown',
                'train_val_gap': 0.0
            }

    def _check_safety_constraints(self, params: Dict[str, Any], model_type: str) -> List[str]:
        """Check if parameters violate safety constraints."""
        violations = []

        try:
            # Check parameter ranges
            for param_name, param_value in params.items():
                if isinstance(param_value, (int, float)):
                    if abs(param_value) > self.config.max_parameter_range:
                        violations.append(f"Parameter {param_name} exceeds maximum range: {param_value}")

            # Model-specific safety checks
            if model_type.lower() == 'xgboost':
                if params.get('max_depth', 0) > 15:
                    violations.append("XGBoost max_depth too high - may cause overfitting")
                if params.get('n_estimators', 0) > 1000:
                    violations.append("XGBoost n_estimators too high - may cause overfitting")

        except Exception as e:
            logger.error(f"Safety constraint check failed: {e}")
            violations.append(f"Safety check failed: {str(e)}")

        return violations

    def _analyze_optimization_results(self, report: HPOOptimizationReport, study: optuna.Study, optimization_id: str) -> HPOOptimizationReport:
        """Analyze optimization results."""
        try:
            # Extract score progression
            report.score_progression = [t.value for t in study.trials if t.value is not None]

            # Count overfitting and safety violations
            if optimization_id in self.active_optimizations:
                trials_data = self.active_optimizations[optimization_id]['trials']
                report.overfitting_trials_detected = len([
                    t for t in trials_data if t.get('overfitting_detected', False)
                ])
                report.safety_violations_detected = len([
                    t for t in trials_data if t.get('safety_violations')
                ])

            # Calculate regularization effectiveness
            if report.overfitting_trials_detected > 0:
                report.regularization_effectiveness = 1 - (report.overfitting_trials_detected / report.total_trials)
            else:
                report.regularization_effectiveness = 1.0

            # Assess optimization quality
            if len(report.score_progression) > 10:
                final_score = report.score_progression[-1]
                initial_score = report.score_progression[0]
                improvement = final_score - initial_score

                if improvement > 0.1:
                    report.optimization_quality = "excellent"
                elif improvement > 0.05:
                    report.optimization_quality = "good"
                elif improvement > 0.01:
                    report.optimization_quality = "fair"
                else:
                    report.optimization_quality = "poor"
            else:
                report.optimization_quality = "insufficient_trials"

        except Exception as e:
            logger.error(f"Optimization analysis failed: {e}")

        return report

    def _evaluate_final_model(self,
                             report: HPOOptimizationReport,
                             model_class: Any,
                             X: np.ndarray,
                             y: np.ndarray,
                             is_classification: bool,
                             random_state: int) -> HPOOptimizationReport:
        """Evaluate the final optimized model."""
        try:
            # Create final model with best parameters
            final_model = model_class(**report.best_params)
            final_model.fit(X, y)

            # Perform comprehensive evaluation
            if is_classification:
                y_pred = final_model.predict(X)
                report.final_model_score = accuracy_score(y, y_pred)
            else:
                y_pred = final_model.predict(X)
                report.final_model_score = 1 - np.mean((y - y_pred) ** 2) / np.var(y)

            # Stability assessment
            try:
                cv_res = unified_perform_cv(final_model, X, y, strategy='standard', cv_folds=5, scoring='accuracy' if is_classification else 'r2')
                cv_scores = np.array(cv_res.get('scores', []) or [])
                report.final_model_stability = 1 - float(np.std(cv_scores)) if cv_scores.size else 0.0
            except Exception:
                report.final_model_stability = 0.0

            # Robustness assessment
            report.final_model_robustness = self._assess_model_robustness(
                final_model, X, y, is_classification
            )

            # Final overfitting risk
            if report.final_model_stability > 0.9 and report.final_model_robustness > 0.8:
                report.final_overfitting_risk = "low"
            elif report.final_model_stability > 0.7 and report.final_model_robustness > 0.6:
                report.final_overfitting_risk = "medium"
            else:
                report.final_overfitting_risk = "high"

        except Exception as e:
            logger.error(f"Final model evaluation failed: {e}")
            report.warnings.append(f"Final evaluation failed: {str(e)}")

        return report

    def _assess_model_robustness(self, model: Any, X: np.ndarray, y: np.ndarray, is_classification: bool) -> float:
        """Assess model robustness."""
        try:
            base_score = self._evaluate_model(model, X, y, is_classification)
            robustness_score = base_score

            # Test with small noise
            X_noisy = X + np.random.normal(0, 0.01, X.shape)
            noisy_score = self._evaluate_model(model, X_noisy, y, is_classification)
            robustness_score -= abs(base_score - noisy_score)

            return max(0, min(1, robustness_score))

        except Exception as e:
            logger.error(f"Robustness assessment failed: {e}")
            return 0.5

    def _evaluate_model(self, model: Any, X: np.ndarray, y: np.ndarray, is_classification: bool) -> float:
        """Evaluate model performance."""
        try:
            if is_classification:
                y_pred = model.predict(X)
                return accuracy_score(y, y_pred)
            else:
                y_pred = model.predict(X)
                return 1 - np.mean((y - y_pred) ** 2) / np.var(y)
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return 0.0

    def _generate_hpo_recommendations(self, report: HPOOptimizationReport) -> HPOOptimizationReport:
        """Generate HPO recommendations."""
        try:
            if report.final_overfitting_risk == "high":
                report.recommendations.append("High overfitting risk - consider stronger regularization")
                report.recommendations.append("Increase training data size if possible")

            if report.optimization_quality == "poor":
                report.recommendations.append("Optimization quality was poor - try different hyperparameters")
                report.recommendations.append("Consider increasing the number of trials")

            if report.safety_violations_detected > 0:
                report.recommendations.append(f"Detected {report.safety_violations_detected} safety violations")
                report.recommendations.append("Review parameter constraints for safety")

            if report.overfitting_trials_detected > report.total_trials * 0.3:
                report.recommendations.append("High percentage of overfitting trials detected")
                report.recommendations.append("Consider using stronger regularization in search space")

            if report.final_model_stability < 0.8:
                report.warnings.append("Final model shows low stability across cross-validation folds")

            if report.final_model_robustness < 0.7:
                report.warnings.append("Final model shows low robustness to perturbations")

        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")

        return report

    def _log_hpo_report(self, report: HPOOptimizationReport):
        """Log HPO report."""
        if not self.config.enable_detailed_logging:
            return

        logger.info(f"HPO Report for {report.model_type}:")
        logger.info(f"  Best Score: {report.best_score:.4f}")
        logger.info(f"  Total Trials: {report.total_trials}")
        logger.info(f"  Success Rate: {report.successful_trials}/{report.total_trials}")
        logger.info(f"  Overfitting Prevention: {report.regularization_effectiveness:.2f}")
        logger.info(f"  Final Overfitting Risk: {report.final_overfitting_risk}")
        logger.info(f"  Optimization Quality: {report.optimization_quality}")

        if report.warnings:
            for warning in report.warnings:
                logger.warning(f"  Warning: {warning}")

        if report.recommendations:
            logger.info(f"  Recommendations: {len(report.recommendations)}")
            for rec in report.recommendations[:3]:
                logger.info(f"    - {rec}")

    def save_hpo_report(self, report: HPOOptimizationReport, filename: Optional[str] = None):
        """Save HPO report to file."""
        if not self.config.save_hpo_reports:
            return

        if filename is None:
            filename = f"hpo_report_{report.optimization_id}.json"

        filepath = Path(self.config.report_directory) / filename

        try:
            report_dict = asdict(report)
            with open(filepath, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)
            logger.info(f"HPO report saved: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save HPO report: {e}")

    def get_optimization_history(self) -> List[HPOOptimizationReport]:
        """Get HPO optimization history."""
        return self.optimization_history.copy()

# Global instance
DEFAULT_HPO_PREVENTION = HPOWithOverfittingPrevention()

def get_hpo_with_overfitting_prevention(config: Optional[HPOOverfittingPreventionConfig] = None) -> HPOWithOverfittingPrevention:
    """Get HPO with overfitting prevention instance."""
    if config is None:
        return DEFAULT_HPO_PREVENTION
    return HPOWithOverfittingPrevention(config)

def optimize_hyperparameters_safely(model_class: Any,
                                   X: np.ndarray,
                                   y: np.ndarray,
                                   model_name: str = "model",
                                   model_type: str = "unknown",
                                   param_space: Optional[Dict[str, Any]] = None,
                                   is_classification: bool = True,
                                   random_state: int = 42) -> HPOOptimizationReport:
    """Convenience function to perform safe HPO with overfitting prevention."""
    hpo_system = get_hpo_with_overfitting_prevention()
    return hpo_system.optimize_hyperparameters(
        model_class, X, y, model_name, model_type, param_space, is_classification, random_state
    )
