# src/training/steps/optimized_optuna_optimization.py

import logging
import time
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.utils.logger import setup_logging
from src.utils.warning_symbols import (
    failed,
)

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

    # --- Hyperparameter Space Definitions ---
    def _summarize_study(self, study: optuna.Study) -> dict[str, Any]:
        """Extracts key results from a completed study."""
        pruned_trials = study.get_trials(
            deepcopy=False,
            states=[optuna.trial.TrialState.PRUNED]
        )
        complete_trials = study.get_trials(
            deepcopy=False,
            states=[optuna.trial.TrialState.COMPLETE]
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
        self, model_type: str, X: pd.DataFrame, y: pd.Series, n_trials: int = 100, n_jobs: int = -1, cv_folds: int = 5, early_stopping_patience: int | None = 15, subsample_fraction: float | None = None
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
            storage=self.storage_url,
            study_name=study_name,
            direction="maximize",
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=1,
                max_resource=n_trials
            ),
            sampler=optuna.samplers.TPESampler(seed=42),
            load_if_exists=True
        )

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
    lgbm_results = optimizer.optimize(
        model_type="lightgbm",
        X=X,
        y=y,
        n_trials=50,
        n_jobs=-1,
        subsample_fraction=0.5,  # Use 50% of data per trial
    )

    # 4. Run optimization for RandomForest with custom pruning
    rf_results = optimizer.optimize(
        model_type="random_forest",
        X=X,
        y=y,
        n_trials=30,  # Fewer trials as RF is slower
        n_jobs=-1,
    )

    # 5. You can easily retrieve the full study from storage if needed
    loaded_study = optuna.load_study(
        study_name="production_models_lightgbm",
        storage=optimizer.storage_url
    )