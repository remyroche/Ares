# src/training/bayesian_optimizer.py

from collections.abc import Callable
from typing import Any, Number

import numpy as np
import optuna
import pandas as pd

from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger


class AdvancedHyperparameterOptimizer:
    """
    Advanced hyperparameter optimization with decomposed search spaces and proper constraints.

    Features:
    - Decomposed optimization (feature engineering → model → trading strategy)
    - Proper parameter constraints with pruning
    - Multiple sampling strategies
    - Early stopping with optuna callbacks
    - Multi-dimensional optimization
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("HyperparameterOptimizer")

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

        # Store results from each optimization stage
        self.feature_engineering_results = None
        self.model_optimization_results = None
        self.trading_strategy_results = None

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="hyperparameter optimization setup",
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
        return optuna.create_study(
            direction=direction,
            sampler=sampler,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1,
            ),
        )

    def suggest_feature_engineering_params(
        self,
        trial: optuna.trial.Trial,
    ) -> dict[str, Any]:
        """Suggest hyperparameters for feature engineering optimization."""
        params = {}

        # Feature engineering parameters only
        params["lookback_window"] = trial.suggest_int("lookback_window", 5, 200)
        params["feature_selection_threshold"] = trial.suggest_float(
            "feature_selection_threshold",
            0.001,
            0.1,
        )

        # Add feature engineering constraints
        self._add_feature_constraints(trial, params)

        return params

    def suggest_model_params(self, trial: optuna.trial.Trial) -> dict[str, Any]:
        """Suggest hyperparameters for model optimization."""
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

        # Add model constraints
        self._add_model_constraints(trial, params)

        return params

    def suggest_trading_params(self, trial: optuna.trial.Trial) -> dict[str, Any]:
        """Suggest hyperparameters for trading strategy optimization."""
        params = {}

        # Trading parameters
        params["tp_multiplier"] = trial.suggest_float("tp_multiplier", 1.2, 10.0)
        params["sl_multiplier"] = trial.suggest_float("sl_multiplier", 0.5, 5.0)
        params["position_size"] = trial.suggest_float("position_size", 0.01, 0.5)

        # Add trading constraints
        self._add_trading_constraints(trial, params)

        return params

    def _add_feature_constraints(
        self,
        trial: optuna.trial.Trial,
        params: dict[str, Any],
    ):
        """Add constraints for feature engineering parameters."""

        # Ensure lookback window is reasonable for the data
        if params["lookback_window"] > 100:
            # Prune trials with very large lookback windows
            msg = "Lookback window too large"
            raise optuna.TrialPruned(msg)

        # Ensure feature selection threshold is not too restrictive
        if params["feature_selection_threshold"] > 0.05:
            # Prune trials with very high thresholds
            msg = "Feature selection threshold too high"
            raise optuna.TrialPruned(msg)

    def _add_model_constraints(self, trial: optuna.trial.Trial, params: dict[str, Any]):
        """Add constraints for model parameters."""

        # Model-specific constraints
        if params["model_type"] == "random_forest":
            if params["max_depth"] > 20:
                msg = "Random forest max_depth too high"
                raise optuna.TrialPruned(msg)
        elif params["model_type"] in ["xgboost", "lightgbm"]:
            if params["max_depth"] > 12:
                msg = "XGBoost/LightGBM max_depth too high"
                raise optuna.TrialPruned(msg)

        # Ensure reasonable learning rate
        if params["learning_rate"] < 1e-5:
            msg = "Learning rate too low"
            raise optuna.TrialPruned(msg)

        # Ensure reasonable regularization
        if params["reg_alpha"] > 100 or params["reg_lambda"] > 100:
            msg = "Regularization too high"
            raise optuna.TrialPruned(msg)

    def _add_trading_constraints(
        self,
        trial: optuna.trial.Trial,
        params: dict[str, Any],
    ):
        """Add constraints for trading parameters."""

        # Ensure TP > SL (proper constraint with pruning)
        if params["tp_multiplier"] <= params["sl_multiplier"]:
            msg = "TP multiplier must be greater than SL multiplier"
            raise optuna.TrialPruned(msg)

        # Ensure reasonable position size
        if params["position_size"] > 0.3:
            msg = "Position size too high"
            raise optuna.TrialPruned(msg)

        # Ensure reasonable risk-reward ratio
        risk_reward_ratio = params["tp_multiplier"] / params["sl_multiplier"]
        if risk_reward_ratio < 1.5:
            msg = "Risk-reward ratio too low"
            raise optuna.TrialPruned(msg)

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="objective function evaluation",
    )
    def feature_engineering_objective(self, trial: optuna.trial.Trial) -> Number:
        """Objective function for feature engineering optimization."""

        # Suggest feature engineering parameters
        params = self.suggest_feature_engineering_params(trial)

        # Run evaluation with fixed model and trading parameters
        score = self._evaluate_feature_engineering(params)

        # Report intermediate results for pruning
        trial.report(score, step=0)

        # Store trial history
        self.trial_history.append(
            {
                "trial_number": trial.number,
                "params": params,
                "score": score,
                "timestamp": pd.Timestamp.now(),
                "stage": "feature_engineering",
            },
        )

        return score

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="objective function evaluation",
    )
    def model_optimization_objective(self, trial: optuna.trial.Trial) -> Number:
        """Objective function for model optimization."""

        # Suggest model parameters
        params = self.suggest_model_params(trial)

        # Combine with best feature engineering results
        if self.feature_engineering_results:
            params.update(self.feature_engineering_results["best_params"])

        # Run evaluation with fixed trading parameters
        score = self._evaluate_model_optimization(params)

        # Report intermediate results for pruning
        trial.report(score, step=0)

        # Store trial history
        self.trial_history.append(
            {
                "trial_number": trial.number,
                "params": params,
                "score": score,
                "timestamp": pd.Timestamp.now(),
                "stage": "model_optimization",
            },
        )

        return score

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="objective function evaluation",
    )
    def trading_strategy_objective(self, trial: optuna.trial.Trial) -> Number:
        """Objective function for trading strategy optimization."""

        # Suggest trading parameters
        params = self.suggest_trading_params(trial)

        # Combine with best results from previous stages
        if self.feature_engineering_results:
            params.update(self.feature_engineering_results["best_params"])
        if self.model_optimization_results:
            params.update(self.model_optimization_results["best_params"])

        # Run evaluation
        score = self._evaluate_trading_strategy(params)

        # Report intermediate results for pruning
        trial.report(score, step=0)

        # Store trial history
        self.trial_history.append(
            {
                "trial_number": trial.number,
                "params": params,
                "score": score,
                "timestamp": pd.Timestamp.now(),
                "stage": "trading_strategy",
            },
        )

        return score

    def _evaluate_feature_engineering(self, params: dict[str, Any]) -> Number:
        """Evaluate feature engineering parameters."""
        # Mock evaluation - replace with actual feature engineering evaluation
        base_score = 0.5

        # Adjust score based on feature engineering quality
        if 10 <= params["lookback_window"] <= 50:
            base_score += 0.2

        if 0.01 <= params["feature_selection_threshold"] <= 0.03:
            base_score += 0.15

        # Add some randomness to simulate real evaluation
        noise = np.random.normal(0, 0.1)
        final_score = base_score + noise

        return max(0, min(1, final_score))

    def _evaluate_model_optimization(self, params: dict[str, Any]) -> Number:
        """Evaluate model optimization parameters."""
        # Mock evaluation - replace with actual model evaluation
        base_score = 0.5

        # Adjust score based on model quality
        if params["model_type"] in ["xgboost", "lightgbm"]:
            base_score += 0.1

        if 0.01 <= params["learning_rate"] <= 0.1:
            base_score += 0.2

        if 5 <= params["max_depth"] <= 10:
            base_score += 0.15

        # Add some randomness to simulate real evaluation
        noise = np.random.normal(0, 0.1)
        final_score = base_score + noise

        return max(0, min(1, final_score))

    def _evaluate_trading_strategy(self, params: dict[str, Any]) -> Number:
        """Evaluate trading strategy parameters."""
        # Mock evaluation - replace with actual trading strategy evaluation
        base_score = 0.5

        # Adjust score based on trading strategy quality
        if params["tp_multiplier"] > params["sl_multiplier"] * 1.5:
            base_score += 0.1

        if 0.05 <= params["position_size"] <= 0.2:
            base_score += 0.15

        # Add some randomness to simulate real evaluation
        noise = np.random.normal(0, 0.1)
        final_score = base_score + noise

        return max(0, min(1, final_score))

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="hyperparameter optimization execution",
    )
    def run_decomposed_optimization(
        self,
        objective_func: Callable | None = None,
    ) -> dict[str, Any]:
        """Run decomposed hyperparameter optimization with three focused stages."""

        self.logger.info("Starting decomposed hyperparameter optimization...")

        # Step 1: Optimize Feature Engineering
        self.logger.info("Stage 1: Optimizing feature engineering...")
        feature_study = self.create_study()
        feature_study.optimize(
            self.feature_engineering_objective,
            n_trials=self.max_trials // 3,
            show_progress_bar=True,
            callbacks=[self._optimization_callback],
        )
        self.feature_engineering_results = self._analyze_optimization_results(
            feature_study,
        )

        # Step 2: Optimize Model Hyperparameters
        self.logger.info("Stage 2: Optimizing model hyperparameters...")
        model_study = self.create_study()
        model_study.optimize(
            self.model_optimization_objective,
            n_trials=self.max_trials // 3,
            show_progress_bar=True,
            callbacks=[self._optimization_callback],
        )
        self.model_optimization_results = self._analyze_optimization_results(
            model_study,
        )

        # Step 3: Optimize Trading Strategy
        self.logger.info("Stage 3: Optimizing trading strategy...")
        trading_study = self.create_study()
        trading_study.optimize(
            self.trading_strategy_objective,
            n_trials=self.max_trials // 3,
            show_progress_bar=True,
            callbacks=[self._optimization_callback],
        )
        self.trading_strategy_results = self._analyze_optimization_results(
            trading_study,
        )

        # Combine results
        final_results = self._combine_optimization_results()

        self.logger.info(
            "Decomposed hyperparameter optimization completed successfully",
        )

        return final_results

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

    def _combine_optimization_results(self) -> dict[str, Any]:
        """Combine results from all optimization stages."""

        # Combine best parameters from all stages
        combined_params = {}
        if self.feature_engineering_results:
            combined_params.update(self.feature_engineering_results["best_params"])
        if self.model_optimization_results:
            combined_params.update(self.model_optimization_results["best_params"])
        if self.trading_strategy_results:
            combined_params.update(self.trading_strategy_results["best_params"])

        # Calculate overall score (weighted average)
        scores = []
        weights = []

        if self.feature_engineering_results:
            scores.append(self.feature_engineering_results["best_score"])
            weights.append(0.3)

        if self.model_optimization_results:
            scores.append(self.model_optimization_results["best_score"])
            weights.append(0.4)

        if self.trading_strategy_results:
            scores.append(self.trading_strategy_results["best_score"])
            weights.append(0.3)

        overall_score = np.average(scores, weights=weights) if scores else 0.0

        return {
            "combined_params": combined_params,
            "overall_score": overall_score,
            "feature_engineering_results": self.feature_engineering_results,
            "model_optimization_results": self.model_optimization_results,
            "trading_strategy_results": self.trading_strategy_results,
            "trial_history": self.trial_history,
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
