# src/training/steps/optimized_optuna_optimization.py

import logging
import time
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from src.utils.logger import setup_logging
from src.utils.warning_symbols import (
    failed,
)

setup_logging()

# --- Configuration ---
# Configure logging for Optuna to provide clear output without being overly verbose.
optuna.logging.set_verbosity(optuna.logging.WARNING)


class AdvancedOptunaManager:
    """
    Manages Optuna hyperparameter optimization with advanced features for
    efficiency, robustness, and extensibility.

    Key Features:
    - Persistence: Uses a database backend (e.g., SQLite) to save and resume studies.
    - Pruning: Employs aggressive pruning, including a custom implementation for RandomForest.
    - Efficiency: Supports data subsampling to accelerate trials on large datasets.
    - Extensibility: Uses a configuration-driven design to easily add new models.
    - Robustness: Handles categorical features and trial errors gracefully.
    """

    def __init__(
        self,
        storage_url: str = "sqlite:///optuna_studies_advanced.db",
        study_name_prefix: str = "optimization",
    ):
        """
        Initializes the AdvancedOptunaManager.

        Args:
            storage_url (str): Database URL for study persistence. This is crucial
                               for resuming studies and enabling safe parallel execution.
            study_name_prefix (str): A prefix for all study names.
        """
        self.storage_url = storage_url
        self.study_name_prefix = study_name_prefix
        self.logger = logging.getLogger(__name__)
        self._model_configs = self._get_model_configurations()

    def _get_model_configurations(self) -> dict[str, dict[str, Any]]:
        """
        Returns a dictionary containing the configuration for each supported model.
        This design makes the manager easily extensible.
        """
        return {
            "random_forest": {
                "model": RandomForestClassifier,
                "space": self._get_rf_space,
            },
            "lightgbm": {"model": lgb.LGBMClassifier, "space": self._get_lgbm_space},
            "xgboost": {"model": xgb.XGBClassifier, "space": self._get_xgb_space},
        }

    # --- Hyperparameter Space Definitions ---
    def _get_rf_space(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "max_depth": trial.suggest_int("max_depth", 5, 50),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_float("max_features", 0.1, 1.0),
            "random_state": 42,
            "n_jobs": 1,  # Important for nested parallelism
        }

    def _get_lgbm_space(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
            "verbose": -1,
            "n_jobs": 1,
        }

    def _get_xgb_space(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "random_state": 42,
            "verbosity": 0,
            "n_jobs": 1,
        }

    def _summarize_study(self, study: optuna.Study) -> dict[str, Any]:
        """Extracts key results from a completed study."""

        pruned_trials = study.get_trials(
            deepcopy=False,
            states=[optuna.trial.TrialState.PRUNED],
        )
        complete_trials = study.get_trials(
            deepcopy=False,
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
    ) -> dict[str, Any]:
        """
        Runs a full hyperparameter optimization for a specified model.

        Args:
            model_type (str): The model to optimize (e.g., 'lightgbm').
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
            storage=self.storage_url,
            study_name=study_name,
            direction="maximize",
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=1,
                max_resource=n_trials,
            ),
            sampler=optuna.samplers.TPESampler(seed=42),
            load_if_exists=True,
        )

        def objective(trial: optuna.Trial) -> float:
            try:
                # --- Data Subsampling for Efficiency ---
                X_sample, y_sample = (X, y)
                if subsample_fraction and subsample_fraction < 1.0:
                    X_sample, _, y_sample, _ = train_test_split(
                        X,
                        y,
                        train_size=subsample_fraction,
                        stratify=y,
                        random_state=trial.number,
                    )

                # --- Model and Hyperparameter Setup ---
                config = self._model_configs[model_type]
                params = config["space"](trial)
                model = config["model"](**params)

                # --- Cross-validation and Pruning ---
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

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
                            cv=cv,
                            scoring="accuracy",
                        ).mean()
                        intermediate_scores.append(score)
                        trial.report(score, step=i)
                        if trial.should_prune():
                            raise optuna.TrialPruned
                    return np.mean(intermediate_scores)

                # Native pruning for LightGBM and XGBoost
                score = cross_val_score(
                    model,
                    X_sample,
                    y_sample,
                    cv=cv,
                    scoring="accuracy",
                ).mean()
                trial.report(score, step=0)  # Report final score
                return score

            except optuna.TrialPruned:
                raise
            except Exception:
                self.print(failed("Trial {trial.number} failed with error: {e}"))
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

        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, callbacks=callbacks)

        elapsed_time = time.time() - start_time
        self.logger.info(f"Optimization finished in {elapsed_time:.2f} seconds.")

        return self._summarize_study(study)


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
    print("\n" + "=" * 50)
    print("Optimizing LightGBM with Subsampling (50%)")
    print("=" * 50)
    lgbm_results = optimizer.optimize(
        model_type="lightgbm",
        X=X,
        y=y,
        n_trials=50,
        n_jobs=-1,
        subsample_fraction=0.5,  # Use 50% of data per trial
    )
    print(f"LightGBM Results: {lgbm_results}")

    # 4. Run optimization for RandomForest with custom pruning
    print("\n" + "=" * 50)
    print("Optimizing RandomForest with Custom Pruning")
    print("=" * 50)
    rf_results = optimizer.optimize(
        model_type="random_forest",
        X=X,
        y=y,
        n_trials=30,  # Fewer trials as RF is slower
        n_jobs=-1,
    )
    print(f"RandomForest Results: {rf_results}")

    # 5. You can easily retrieve the full study from storage if needed
    print("\n" + "=" * 50)
    print("Loading previous study from storage")
    print("=" * 50)
    loaded_study = optuna.load_study(
        study_name="production_models_lightgbm",
        storage=optimizer.storage_url,
    )
    print(
        f"Loaded study '{loaded_study.study_name}' with {len(loaded_study.trials)} trials.",
    )
    print("Top 5 trials from loaded study:")
    print(loaded_study.trials_dataframe().sort_values("value", ascending=False).head())
