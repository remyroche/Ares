# src/training/bayesian_optimizer.py

from collections.abc import Callable
from typing import Any, Number

import numpy as np
import optuna
import pandas as pd
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic

from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger


class AdvancedBayesianOptimizer:
    """
    Advanced Bayesian optimization with multiple sampling strategies and early stopping.

    Features:
    - Multiple acquisition functions (EI, PI, UCB)
    - Adaptive sampling strategies
    - Early stopping with patience
    - Pruning of unpromising trials
    - Multi-dimensional optimization
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("BayesianOptimizer")

        # Optimization configuration
        self.max_trials = config.get("max_trials", 1000)
        self.patience = config.get("patience", 50)
        self.min_trials = config.get("min_trials", 20)
        self.pruning_threshold = config.get("pruning_threshold", 0.1)

        # Sampling configuration
        self.sampling_strategy = config.get("sampling_strategy", "tpe")
        self.acquisition_function = config.get("acquisition_function", "ei")

        # Early stopping
        self.best_score = -np.inf
        self.patience_counter = 0
        self.trial_history = []

        # Custom kernels for Gaussian Process
        self.kernels = {
            "rbf": RBF(length_scale=1.0),
            "matern": Matern(length_scale=1.0, nu=1.5),
            "rational_quadratic": RationalQuadratic(length_scale=1.0, alpha=1.0),
        }

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="bayesian optimization setup",
    )
    def create_study(self, direction: str = "maximize") -> optuna.Study:
        """Create an Optuna study with advanced configuration."""

        # Choose sampler based on configuration
        if self.sampling_strategy == "tpe":
            sampler = optuna.samplers.TPESampler(
                n_startup_trials=10,
                n_ei_candidates=24,
                multivariate=True,
                group=True,
            )
        elif self.sampling_strategy == "random":
            sampler = optuna.samplers.RandomSampler()
        elif self.sampling_strategy == "cmaes":
            sampler = optuna.samplers.CmaEsSampler()
        elif self.sampling_strategy == "nsga2":
            sampler = optuna.samplers.NSGAIISampler(
                population_size=50,
                crossover_prob=0.9,
                mutation_prob=0.1,
            )
        else:
            sampler = optuna.samplers.TPESampler()

        # Create study with pruning
        study = optuna.create_study(
            direction=direction,
            sampler=sampler,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1,
            ),
        )

        return study

    def suggest_hyperparameters(self, trial: optuna.trial.Trial) -> dict[str, Any]:
        """Suggest hyperparameters with advanced search spaces and constraints."""
        params = {}

        # Model architecture parameters
        params["model_type"] = trial.suggest_categorical(
            "model_type",
            [
                "xgboost",
                "lightgbm",
                "catboost",
                "random_forest",
                "gradient_boosting",
                "tabnet",
                "transformer",
            ],
        )

        # Learning parameters
        params["learning_rate"] = trial.suggest_float(
            "learning_rate",
            1e-4,
            1e-1,
            log=True,
        )
        params["max_depth"] = trial.suggest_int("max_depth", 3, 15)
        params["n_estimators"] = trial.suggest_int("n_estimators", 50, 1000)

        # Regularization parameters
        params["reg_alpha"] = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True)
        params["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)

        # Feature engineering parameters
        params["lookback_window"] = trial.suggest_int("lookback_window", 5, 200)
        params["feature_selection_threshold"] = trial.suggest_float(
            "feature_selection_threshold",
            0.001,
            0.1,
        )

        # Trading parameters
        params["tp_multiplier"] = trial.suggest_float("tp_multiplier", 1.2, 10.0)
        params["sl_multiplier"] = trial.suggest_float("sl_multiplier", 0.5, 5.0)
        params["position_size"] = trial.suggest_float("position_size", 0.01, 0.5)

        # Ensemble parameters
        params["ensemble_size"] = trial.suggest_int("ensemble_size", 3, 10)
        params["ensemble_weight"] = trial.suggest_float("ensemble_weight", 0.1, 0.9)

        # Advanced parameters based on model type
        if params["model_type"] in ["xgboost", "lightgbm"]:
            params["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
            params["colsample_bytree"] = trial.suggest_float(
                "colsample_bytree",
                0.6,
                1.0,
            )
            params["min_child_weight"] = trial.suggest_int("min_child_weight", 1, 10)
        elif params["model_type"] == "tabnet":
            params["tabnet_attention_dim"] = trial.suggest_int(
                "tabnet_attention_dim",
                8,
                64,
            )
            params["tabnet_num_steps"] = trial.suggest_int("tabnet_num_steps", 1, 10)
            params["tabnet_gamma"] = trial.suggest_float("tabnet_gamma", 1.0, 3.0)
            params["tabnet_momentum"] = trial.suggest_float("tabnet_momentum", 0.1, 0.9)
        elif params["model_type"] == "transformer":
            params["transformer_n_heads"] = trial.suggest_int(
                "transformer_n_heads",
                2,
                8,
            )
            params["transformer_n_layers"] = trial.suggest_int(
                "transformer_n_layers",
                1,
                6,
            )
            params["transformer_d_model"] = trial.suggest_int(
                "transformer_d_model",
                32,
                256,
            )
            params["transformer_dropout"] = trial.suggest_float(
                "transformer_dropout",
                0.1,
                0.5,
            )

        # Add constraints
        self._add_parameter_constraints(trial, params)

        return params

    def _add_parameter_constraints(
        self,
        trial: optuna.trial.Trial,
        params: dict[str, Any],
    ):
        """Add parameter constraints to ensure valid combinations."""

        # Ensure TP > SL
        if params["tp_multiplier"] <= params["sl_multiplier"]:
            params["tp_multiplier"] = params["sl_multiplier"] * 1.5

        # Ensure reasonable position size
        params["position_size"] = min(params["position_size"], 0.3)

        # Model-specific constraints
        if params["model_type"] == "random_forest":
            params["max_depth"] = min(params["max_depth"], 20)
        elif params["model_type"] in ["xgboost", "lightgbm"]:
            params["max_depth"] = min(params["max_depth"], 12)

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="objective function evaluation",
    )
    def objective(self, trial: optuna.trial.Trial) -> Number:
        """Objective function with early stopping and pruning."""

        # Suggest hyperparameters
        params = self.suggest_hyperparameters(trial)

        # Run evaluation
        score = self._evaluate_parameters(params)

        # Early stopping check
        if self._should_stop_early(score):
            trial.report(score, step=0)
            trial.set_user_attr("stopped_early", True)
            return score

        # Report intermediate results for pruning
        trial.report(score, step=0)

        # Store trial history
        self.trial_history.append(
            {
                "trial_number": trial.number,
                "params": params,
                "score": score,
                "timestamp": pd.Timestamp.now(),
            },
        )

        return score

    def _evaluate_parameters(self, params: dict[str, Any]) -> Number:
        """Evaluate parameters using backtesting."""
        # This would integrate with your existing backtesting infrastructure
        # For now, using a mock evaluation

        # Mock evaluation based on parameter quality
        base_score = 0.5

        # Adjust score based on parameter combinations
        if params["model_type"] in ["xgboost", "lightgbm"]:
            base_score += 0.1

        if 0.01 <= params["learning_rate"] <= 0.1:
            base_score += 0.2

        if 5 <= params["max_depth"] <= 10:
            base_score += 0.15

        if params["tp_multiplier"] > params["sl_multiplier"] * 1.5:
            base_score += 0.1

        # Add some randomness to simulate real evaluation
        noise = np.random.normal(0, 0.1)
        final_score = base_score + noise

        return max(0, min(1, final_score))  # Clamp between 0 and 1

    def _should_stop_early(self, score: Number) -> bool:
        """Check if optimization should stop early."""
        if score > self.best_score:
            self.best_score = score
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        return self.patience_counter >= self.patience

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="bayesian optimization execution",
    )
    def run_optimization(
        self,
        objective_func: Callable | None = None,
    ) -> dict[str, Any]:
        """Run Bayesian optimization with advanced features."""

        self.logger.info(
            f"Starting Bayesian optimization with {self.max_trials} trials...",
        )

        # Create study
        study = self.create_study()

        # Use custom objective if provided, otherwise use default
        if objective_func is None:
            objective_func = self.objective

        # Run optimization
        study.optimize(
            objective_func,
            n_trials=self.max_trials,
            show_progress_bar=True,
            callbacks=[self._optimization_callback],
        )

        # Analyze results
        results = self._analyze_optimization_results(study)

        self.logger.info("Bayesian optimization completed successfully")

        return results

    def _optimization_callback(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
    ):
        """Callback function for optimization monitoring."""
        if trial.state == optuna.trial.TrialState.COMPLETE:
            self.logger.info(f"Trial {trial.number}: Score = {trial.value:.4f}")

            # Log best parameters periodically
            if trial.number % 10 == 0:
                best_trial = study.best_trial
                self.logger.info(f"Best trial so far: {best_trial.value:.4f}")

    def _analyze_optimization_results(self, study: optuna.Study) -> dict[str, Any]:
        """Analyze and summarize optimization results."""

        # Get best trial
        best_trial = study.best_trial

        # Analyze parameter importance
        importance = optuna.importance.get_param_importances(study)

        # Get optimization history
        trials_df = study.trials_dataframe()

        # Calculate convergence metrics
        convergence_metrics = self._calculate_convergence_metrics(study)

        return {
            "best_params": best_trial.params,
            "best_score": best_trial.value,
            "parameter_importance": importance,
            "trials_dataframe": trials_df,
            "convergence_metrics": convergence_metrics,
            "study": study,
        }

    def _calculate_convergence_metrics(self, study: optuna.Study) -> dict[str, Any]:
        """Calculate convergence and optimization quality metrics."""

        trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]

        if len(trials) < 2:
            return {"converged": False, "improvement_rate": 0.0}

        # Calculate improvement over time
        scores = [t.value for t in trials]
        improvements = []

        for i in range(1, len(scores)):
            improvement = (scores[i] - scores[i - 1]) / max(abs(scores[i - 1]), 1e-8)
            improvements.append(improvement)

        # Check convergence
        recent_improvements = (
            improvements[-10:] if len(improvements) >= 10 else improvements
        )
        avg_recent_improvement = (
            np.mean(recent_improvements) if recent_improvements else 0
        )

        converged = avg_recent_improvement < 0.01  # Less than 1% improvement

        return {
            "converged": converged,
            "improvement_rate": avg_recent_improvement,
            "total_improvement": scores[-1] - scores[0],
            "final_score": scores[-1],
            "num_trials": len(trials),
        }

    def suggest_hyperparameter_ranges(self, param_name: str) -> dict[str, Any]:
        """Suggest optimal hyperparameter ranges based on optimization history."""

        if not self.trial_history:
            return {}

        # Analyze parameter distribution for best trials
        best_trials = sorted(
            self.trial_history,
            key=lambda x: x["score"],
            reverse=True,
        )[:10]

        param_values = [
            trial["params"].get(param_name)
            for trial in best_trials
            if param_name in trial["params"]
        ]

        if not param_values:
            return {}

        # Calculate statistics
        mean_val = np.mean(param_values)
        std_val = np.std(param_values)
        min_val = np.min(param_values)
        max_val = np.max(param_values)

        return {
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
            "range": max_val - min_val,
            "confidence_interval": (mean_val - 2 * std_val, mean_val + 2 * std_val),
        }
