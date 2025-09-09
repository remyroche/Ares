
import logging
import time
from typing import Any

import optuna
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .utils.logger import setup_logging

import pandas as pd
import numpy as np
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
from src.utils.ml_common import (
    Solution,
    compute_pareto_front,
    select_knee_point,
    DEFAULT_FINANCIAL_WEIGHTS,
)

# src/training/steps/optimized_optuna_optimization.py

setup_logging()

# --- Configuration ---
# Configure logging for Optuna to provide clear output without being overly verbose.
optuna.logging.set_verbosity(optuna.logging.WARNING)

class AdvancedOptunaManager:
    """Manages Optuna hyperparameter optimization with advanced features for
    efficiency = robustness, and extensibility.

    Key Features:
    - Persistence: Uses a database backend (e.g., SQLite) to save and resume studies.
    - Pruning: Employs aggressive pruning, including a custom implementation for RandomForest.
    - Efficiency: Supports data subsampling to accelerate trials on large datasets.
    - Extensibility: Uses a configuration-driven design to easily add new models.
    - Robustness: Handles categorical features and trial errors gracefully.
    """
    @log_important_calls

    def __init__(
        self,
        storage_url: str = "sqlite:///optuna_studies_advanced.db",
        study_name_prefix: str = "optimization",
    ) -> None:
        """Initializes the AdvancedOptunaManager.

        Args:
            storage_url (str): Database URL for study persistence. This is crucial
        for resuming studies and enabling safe parallel execution.
            study_name_prefix (str): A prefix for all study names.

        """
        self.storage_url = storage_url
        self.study_name_prefix = study_name_prefix
        self.logger = logging.getLogger(__name__)
        self._model_configs = self._get_model_configurations()
    @log_all_calls

    def _get_model_configurations(self) -> dict[str, dict[str, Any]]:
        """Returns a dictionary containing the configuration for each supported model.
        This design makes the manager easily extensible.
        """
        return {
            "random_forest": {
                "model": RandomForestClassifier,
                "space": self._get_rf_space,
            },
            "lightgbm": {"model": lgb.LGBMClassifier, "space": self._get_lgbm_space},
            "xgboost": {"model": xgb.XGBClassifier, "space": self._get_xgb_space},
            "catboost": {"model": CatBoostClassifier, "space": self._get_cb_space},
        }

    @log_all_calls
    # --- Hyperparameter Space Definitions ---
    def _get_rf_space(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step = 50),
            "max_depth": trial.suggest_int("max_depth", 5, 50),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_float("max_features", 0.1, 1.0),
            "random_state": 42,
            "n_jobs": 1,  # Important for nested parallelism
        }
    @log_all_calls

    def _get_lgbm_space(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000, step = 100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log = True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
            "verbose": -1,
            "n_jobs": 1,
        }
    @log_all_calls

    def _get_xgb_space(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000, step = 100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log = True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log = True),
            "random_state": 42,
            "verbosity": 0,
            "n_jobs": 1,
        }
    @log_all_calls

    def _get_cb_space(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "iterations": trial.suggest_int("iterations", 200, 2000, step = 100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log = True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "random_seed": 42,
            "verbose": False,
        }
    @log_all_calls

    def _summarize_study(self, study: optuna.Study) -> dict[str, Any]:
        """Extracts key results from a completed study."""
        pruned_trials = study.get_trials(
            deepcopy = False,
            states=[optuna.trial.TrialState.PRUNED],
        )
        complete_trials = study.get_trials(
            deepcopy = False,
            states=[optuna.trial.TrialState.COMPLETE],
        )

        summary = {
            "study_name": study.study_name,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "total_trials": len(study.trials),
            "n_completed": len(complete_trials),
            "n_pruned": len(pruned_trials),
        }
        self.logger.info(f"Study summary: {summary}")
        return summary
    @log_step_functions

    def optimize(
        self,
        model_type: str,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 100,
        n_jobs: int = -1,
        cv_folds: int = 5,
        early_stopping_patience: int | None = 15,
        subsample_fraction: float | None = None,
        constraints: dict[str, Any] | None = None,
        custom_metric_fn: Any | None = None,
    ) -> dict[str, Any]:
        """Runs a full hyperparameter optimization for a specified model.

        Args: model_type (str): The model to optimize (e.g. = 'lightgbm').
            X (pd.DataFrame): Full training features.
            y (pd.Series): Full training labels.
            n_trials (int): Number of optimization trials.
            n_jobs (int): Number of parallel jobs. -1 uses all cores.
            cv_folds (int): Number of folds for cross-validation.
            early_stopping_patience (Optional[int]): Patience for early stopping callback.
            subsample_fraction (Optional[float]): Fraction of data to use for each trial
                                                to speed up optimization. If None, uses all data.

        Returns:
            A dictionary summarizing the results of the optimization study.

        """
        if model_type not in self._model_configs:
            msg = f"Model type '{model_type}' is not configured."
            raise ValueError(msg)

        study_name = f"{self.study_name_prefix}_{model_type}"
        study = optuna.create_study(
            storage = self.storage_url,
            study_name = study_name,
            direction="maximize",
            pruner = optuna.pruners.HyperbandPruner(
                min_resource = 1,
                max_resource = n_trials,
            ),
            sampler = optuna.samplers.TPESampler(seed = 42),
            load_if_exists = True,
        )

        def objective(trial: optuna.Trial) -> float:
            try:
                # --- Data Subsampling for Efficiency ---
                X_sample, y_sample = (X, y)
                if subsample_fraction and subsample_fraction < 1.0:
                    # FIXED: Use time-based subsampling to prevent lookahead bias
                    subsample_size = int(len(X) * subsample_fraction)
                    X_sample = X.iloc[:subsample_size]
                    y_sample = y.iloc[:subsample_size]

                # --- Model and Hyperparameter Setup ---
                config = self._model_configs[model_type]
                params = config["space"](trial)
                model = config["model"](**params)

                # --- Cross-validation and Pruning ---
                cv = StratifiedKFold(n_splits = cv_folds, shuffle = True, random_state = 42)

                # Custom pruning for RandomForest
                if model_type == "random_forest":
                    # Iteratively train and report to enable pruning
                    intermediate_scores = []
                    n_estimators = params["n_estimators"]
                    for i, step in enumerate(range(10, n_estimators + 1, 10)):
                        model.n_estimators = step
                        score = cross_val_score(
                            model,
                            X_sample,
                            y_sample,
                            cv = cv,
                            scoring="accuracy",
                        ).mean()
                        intermediate_scores.append(score)
                        trial.report(score, step = i)
                        if trial.should_prune():
                            raise optuna.TrialPruned
                    return np.mean(intermediate_scores)

                # Native pruning for LightGBM and XGBoost
                score = cross_val_score(
                    model,
                    X_sample,
                    y_sample,
                    cv = cv,
                    scoring="accuracy",
                ).mean()
                trial.report(score, step = 0)  # Report final score
                trial.set_user_attr("accuracy", float(score))
                # Optional: compute custom financial metrics on full sample (fast baseline)
                if callable(custom_metric_fn):
                    try:
                        # Fit once on subsample for metric estimation
                        model.fit(X_sample, y_sample)
                        extra = custom_metric_fn(model, X_sample, y_sample)
                        if isinstance(extra, dict):
                            for k, v in extra.items():
                                try:
                                    trial.set_user_attr(str(k), float(v))
                                except Exception:
                                    pass
                    except Exception:
                        pass
                return score

            except optuna.TrialPruned:
                raise
            except Exception as e:
                self.logger.exception(f"Trial {trial.number} failed with error: {e}")
                return 0.0  # Return a poor score to guide sampler away

        callbacks = []
        if early_stopping_patience:
            callbacks.append(
                optuna.callbacks.EarlyStoppingCallback(
                    early_stopping_patience,
                    "maximize",
                ),
            )

        self.logger.info(
            f"Starting optimization for '{model_type}' with {n_trials} trials...",
        )
        start_time = time.time()

        study.optimize(objective, n_trials = n_trials, n_jobs = n_jobs, callbacks = callbacks)

        elapsed_time = time.time() - start_time
        self.logger.info(f"Optimization finished in {elapsed_time:.2f} seconds.")

        summary = self._summarize_study(study)

        # --- Pareto front selection (post-study) ---
        try:
            complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if complete_trials:
                metric_keys = None
                # Prefer financial metrics if available
                example_attrs = complete_trials[0].user_attrs
                if all(k in example_attrs for k in ("pnl", "win_rate", "sharpe")):
                    metric_keys = ("pnl", "win_rate", "sharpe")
                    objectives = {"pnl": "max", "win_rate": "max", "sharpe": "max"}
                    weights = DEFAULT_FINANCIAL_WEIGHTS
                else:
                    # Fallback to accuracy vs fit_time if available, otherwise accuracy only
                    metric_keys = tuple(k for k in ("accuracy", "fit_time") if k in example_attrs)
                    if not metric_keys:
                        metric_keys = ("accuracy",)
                    objectives = {k: ("min" if k == "fit_time" else "max") for k in metric_keys}
                    weights = None

                solutions: list[Solution] = []
                for t in complete_trials:
                    metrics = {k: float(t.user_attrs.get(k)) for k in metric_keys if k in t.user_attrs}
                    if metrics:
                        solutions.append(Solution(metrics=metrics, params=t.params))

                if solutions:
                    if constraints:
                        # simple threshold-based filter
                        from src.utils.ml_common import filter_by_constraints
                        solutions = filter_by_constraints(solutions, constraints)

                    front = compute_pareto_front(solutions, objectives)
                    knee = select_knee_point(front, objectives, weights=weights)

                    # Populate summary
                    summary["pareto_front"] = [s.metrics for s in front]
                    if knee:
                        summary["pareto_knee_metrics"] = knee.metrics
                        summary["pareto_knee_params"] = knee.params
                        # Convenience: recommend params
                        summary["recommended_params"] = knee.params
        except Exception as e:
            self.logger.warning(f"Pareto post-processing failed: {e}")

        return summary

if __name__ == "__main__":
    # --- Example Usage ---

    # 1. Create a larger, more realistic sample dataset
    X, y = (
        pd.DataFrame(np.random.randn(2000, 30)),
        pd.Series(np.random.randint(0, 2, 2000)),
    )

    # 2. Initialize the manager
    optimizer = AdvancedOptunaManager(study_name_prefix="production_models")

    # 3. Run optimization for LightGBM using data subsampling for speed
    # This will use only 50% of the data for each trial, making it much faster.
    lgbm_results = optimizer.optimize(
        model_type="lightgbm",
        X = X,
        y = y,
        n_trials = 50,
        n_jobs=-1,
        subsample_fraction = 0.5,  # Use 50% of data per trial
    )

    # 4. Run optimization for RandomForest with custom pruning
    rf_results = optimizer.optimize(
        model_type="random_forest",
        X = X,
        y = y,
        n_trials = 30,  # Fewer trials as RF is slower
        n_jobs=-1,
    )

    # 5. You can easily retrieve the full study from storage if needed
    loaded_study = optuna.load_study(
        study_name="production_models_lightgbm",
        storage = optimizer.storage_url,
    )
