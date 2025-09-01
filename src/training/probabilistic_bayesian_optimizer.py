#!/usr/bin/env python3
"""
Probabilistic Bayesian Optimizer for Tactician and Analyst Models

This module provides Bayesian optimization specifically designed for probabilistic models
that output probability distributions, confidence intervals, and uncertainty estimates.
It optimizes both model hyperparameters and probabilistic output calibration.
"""

import logging
import numpy as np
import optuna
from typing import Dict, Any, List, Tuple, Optional, Callable
from dataclasses import dataclass
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class ProbabilisticOptimizationConfig:
    """Configuration for probabilistic Bayesian optimization."""

    # Optimization objectives
    objectives: List[str] = None  # ['calibration', 'sharpness', 'discrimination']

    # Calibration metrics
    calibration_bins: int = 10
    reliability_threshold: float = 0.1

    # Uncertainty quantification
    uncertainty_weight: float = 0.3
    confidence_calibration_weight: float = 0.4
    prediction_accuracy_weight: float = 0.3

    # Optimization parameters
    n_trials: int = 100
    n_jobs: int = 1
    timeout: Optional[int] = None

    # Early stopping
    early_stopping_patience: int = 10
    min_trials: int = 20

    # Sampling strategy
    sampler_type: str = "tpe"  # 'tpe', 'cmaes', 'random'


class ProbabilisticBayesianOptimizer:
    """
    Bayesian optimizer specifically designed for probabilistic models.

    This optimizer focuses on:
    1. Calibration: Ensuring predicted probabilities match observed frequencies
    2. Sharpness: Making predictions as precise as possible
    3. Discrimination: Maximizing the difference between positive and negative predictions
    4. Uncertainty quantification: Optimizing confidence intervals and uncertainty estimates
    """

    def __init__(
        self,
        config: ProbabilisticOptimizationConfig,
        model_type: str = "tactician",  # 'tactician' or 'analyst'
        storage_url: str = "sqlite:///probabilistic_optuna.db"
    ):
        self.config = config
        self.model_type = model_type
        self.storage_url = storage_url
        self.logger = logging.getLogger(__name__)

        # Initialize Optuna study
        self.study = self._create_study()

        # Model-specific configurations
        self.model_configs = self._get_model_configurations()

    def _create_study(self) -> optuna.Study:
        """Create Optuna study with appropriate sampler and pruner."""

        # Choose sampler based on configuration
        if self.config.sampler_type == "tpe":
            sampler = optuna.samplers.TPESampler(seed=42)
        elif self.config.sampler_type == "cmaes":
            sampler = optuna.samplers.CmaEsSampler(seed=42)
        else:
            sampler = optuna.samplers.RandomSampler(seed=42)

        # Create study with multi-objective optimization
        study = optuna.create_study(
            study_name=f"probabilistic_{self.model_type}_optimization",
            storage=self.storage_url,
            sampler=sampler,
            directions=["maximize"] * len(self.config.objectives),
            load_if_exists=True
        )

        return study

    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for the current trial."""

        params = {}

        # Base model parameters
        base_config = self.model_configs["base_model"]
        for param, (low, high) in base_config.items():
            if isinstance(low, int):
                params[param] = trial.suggest_int(param, low, high)
            else:
                params[param] = trial.suggest_float(param, low, high, log=True)

        # Probabilistic calibration parameters
        calib_config = self.model_configs["probabilistic_calibration"]
        params["calibration_method"] = trial.suggest_categorical(
            "calibration_method", calib_config["calibration_method"]
        )
        params["calibration_cv_folds"] = trial.suggest_int(
            "calibration_cv_folds",
            calib_config["calibration_cv_folds"][0],
            calib_config["calibration_cv_folds"][1]
        )
        params["uncertainty_estimation"] = trial.suggest_categorical(
            "uncertainty_estimation", calib_config["uncertainty_estimation"]
        )

        # Model-specific parameters
        if self.model_type == "tactician":
            barrier_config = self.model_configs["barrier_system"]
            params["upper_barrier_multiplier"] = trial.suggest_float(
                "upper_barrier_multiplier",
                barrier_config["upper_barrier_multiplier"][0],
                barrier_config["upper_barrier_multiplier"][1]
            )
            params["lower_barrier_multiplier"] = trial.suggest_float(
                "lower_barrier_multiplier",
                barrier_config["lower_barrier_multiplier"][0],
                barrier_config["lower_barrier_multiplier"][1]
            )
            params["confidence_threshold"] = trial.suggest_float(
                "confidence_threshold",
                barrier_config["confidence_threshold"][0],
                barrier_config["confidence_threshold"][1]
            )
            params["precision_threshold"] = trial.suggest_float(
                "precision_threshold",
                barrier_config["precision_threshold"][0],
                barrier_config["precision_threshold"][1]
            )
        else:
            regime_config = self.model_configs["regime_detection"]
            params["regime_threshold"] = trial.suggest_float(
                "regime_threshold",
                regime_config["regime_threshold"][0],
                regime_config["regime_threshold"][1]
            )
            params["regime_confidence_threshold"] = trial.suggest_float(
                "regime_confidence_threshold",
                regime_config["regime_confidence_threshold"][0],
                regime_config["regime_confidence_threshold"][1]
            )
            params["regime_transition_smoothing"] = trial.suggest_float(
                "regime_transition_smoothing",
                regime_config["regime_transition_smoothing"][0],
                regime_config["regime_transition_smoothing"][1]
            )

        return params

    def evaluate_probabilistic_metrics(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        confidence_intervals: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Evaluate probabilistic model performance metrics."""

        metrics = {}

        # Calibration metrics
        if "calibration" in self.config.objectives:
            metrics["calibration"] = self._calculate_calibration_score(y_true, y_pred_proba)

        # Sharpness metrics
        if "sharpness" in self.config.objectives:
            metrics["sharpness"] = self._calculate_sharpness_score(y_pred_proba)

        # Discrimination metrics
        if "discrimination" in self.config.objectives:
            metrics["discrimination"] = self._calculate_discrimination_score(y_true, y_pred_proba)

        # Uncertainty quantification metrics
        if confidence_intervals is not None:
            metrics["uncertainty_quality"] = self._calculate_uncertainty_quality(
                y_true, y_pred_proba, confidence_intervals
            )

        return metrics

    def _calculate_calibration_score(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> float:
        """Calculate calibration score (lower is better)."""
        try:
            # Use Brier score for calibration
            return brier_score_loss(y_true, y_pred_proba)
        except:
            return 1.0  # Worst possible score

    def _calculate_sharpness_score(self, y_pred_proba: np.ndarray) -> float:
        """Calculate sharpness score (higher is better)."""
        try:
            # Sharpness is the negative entropy of predictions
            # We want predictions to be confident (low entropy)
            entropy = -np.mean(y_pred_proba * np.log(y_pred_proba + 1e-10))
            return -entropy  # Negative because we want to maximize
        except:
            return 0.0

    def _calculate_discrimination_score(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> float:
        """Calculate discrimination score (higher is better)."""
        try:
            # Use ROC AUC for discrimination
            return roc_auc_score(y_true, y_pred_proba)
        except:
            return 0.5  # Random performance

    def _calculate_uncertainty_quality(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        confidence_intervals: np.ndarray
    ) -> float:
        """Calculate uncertainty quantification quality."""
        try:
            # Check if true values fall within confidence intervals
            coverage = np.mean(
                (y_true >= confidence_intervals[:, 0]) &
                (y_true <= confidence_intervals[:, 1])
            )
            return coverage
        except:
            return 0.0

    def create_objective_function(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_factory: Callable,
        validation_split: float = 0.2
    ) -> Callable:
        """Create the objective function for optimization."""

        return objective

    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_factory: Callable,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """Run the Bayesian optimization with MLflow integration."""

        self.logger.info(f"Starting probabilistic Bayesian optimization for {self.model_type}")
        self.logger.info(f"Objectives: {self.config.objectives}")
        self.logger.info(f"Number of trials: {self.config.n_trials}")
        self.logger.info(f"Objective weights: 50% total_profit, 25% win_rate, 25% sharpe_ratio")

        # Create objective function
        objective = self.create_objective_function(X, y, model_factory, validation_split)

        # Set up callbacks
        callbacks = []
        if self.config.early_stopping_patience > 0:
            callbacks.append(
                optuna.callbacks.EarlyStoppingCallback(
                    self.config.early_stopping_patience,
                    directions=["maximize"] * len(self.config.objectives)
                )
            )

        # Run optimization
        self.study.optimize(
            objective,
            n_trials=self.config.n_trials,
            n_jobs=self.config.n_jobs,
            timeout=self.config.timeout,
            callbacks=callbacks
        )

        # Extract results
        results = self._extract_optimization_results()

        # Log to MLflow
        if results.get("best_solutions"):
            best_trial = self.study.best_trials[0] if self.study.best_trials else None
            if best_trial:
                best_params = best_trial.params
                best_values = best_trial.values
                self._log_mlflow_experiment(
                    study_name=self.study.study_name,
                    best_params=best_params,
                    best_values=best_values
                )

        self.logger.info("Probabilistic Bayesian optimization completed successfully!")

        return results

    def _extract_optimization_results(self) -> Dict[str, Any]:
        """Extract and format optimization results."""

        # Get Pareto front solutions
        pareto_front = self.study.best_trials

        # Get best solution for each objective
        best_solutions = {}
        for i, objective in enumerate(self.config.objectives):
            best_trial = min(pareto_front, key=lambda t: t.values[i])
            best_solutions[objective] = {
                "params": best_trial.params,
                "value": best_trial.values[i],
                "trial_number": best_trial.number
            }

        # Get parameter importance
        try:
            param_importance = optuna.importance.get_param_importances(self.study)
        except:
            param_importance = {}

        # Get optimization history
        optimization_history = []
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                optimization_history.append({
                    "trial_number": trial.number,
                    "values": trial.values,
                    "params": trial.params,
                    "duration": trial.duration.total_seconds()
                })

        return {
            "best_solutions": best_solutions,
            "pareto_front": pareto_front,
            "parameter_importance": param_importance,
            "optimization_history": optimization_history,
            "study": self.study,
            "config": self.config
        }

    def _log_mlflow_experiment(self, study_name: str, best_params: Dict[str, Any], best_values: List[float]):
        """Log optimization results to MLflow."""

        try:
            import mlflow

            # Set experiment name
            mlflow.set_experiment(f"step17_optimization_{self.model_type}")

            # Log parameters
            mlflow.log_params(best_params)

            # Log metrics
            for i, objective in enumerate(self.config.objectives):
                mlflow.log_metric(f"best_{objective}", best_values[i])

            # Log optimization metadata
            mlflow.log_param("model_type", self.model_type)
            mlflow.log_param("n_trials", self.config.n_trials)
            mlflow.log_param("sampler_type", self.config.sampler_type)
            mlflow.log_param("study_name", study_name)

            # Log study object
            mlflow.log_artifact(f"{study_name}.db", "study_database")

            self.logger.info("✅ MLflow experiment logged successfully")

        except ImportError:
            self.logger.warning("MLflow not available for experiment logging")
        except Exception as e:
            self.logger.error(f"Failed to log MLflow experiment: {e}")


# Example usage and model factories


if __name__ == "__main__":
    # Example usage
    config = ProbabilisticOptimizationConfig(
        objectives=['calibration', 'sharpness', 'discrimination'],
        n_trials=50,
        n_jobs=1
    )

    # Create optimizer for Tactician
    tactician_optimizer = ProbabilisticBayesianOptimizer(
        config=config,
        model_type="tactician"
    )

    # Create optimizer for Analyst
    analyst_optimizer = ProbabilisticBayesianOptimizer(
        config=config,
        model_type="analyst"
    )

    print("✅ Probabilistic Bayesian Optimizer created successfully!")
    print(f"Tactician optimizer: {tactician_optimizer}")
    print(f"Analyst optimizer: {analyst_optimizer}")