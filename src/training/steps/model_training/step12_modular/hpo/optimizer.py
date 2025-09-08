from ..standardized_parquet_handler import standardized_parquet_handler
"""
Step 12 Modular: Hyperparameter Optimization

This module contains hyperparameter optimization logic for Step 12.
"""

import os
import sys
import signal
import contextlib
import warnings
from io import StringIO
from typing import Dict, Any, Tuple, Optional

try:
    import optuna
    import pandas as pd
    from sklearn.metrics import accuracy_score, log_loss
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from ..base.logger import setup_step12_logger
from ..base.utils import error, failed, timeout, warning

logger = setup_step12_logger()

class HyperparameterOptimizer:
    """Hyperparameter optimization engine for Step 12."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the hyperparameter optimizer.

        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.logger = logger

    async def optimize_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Tuple[Dict[str, Any], float]:
        """Perform hyperparameter optimization using Optuna.

        Args:
            model_name: Name of the model to optimize.
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.

        Returns:
            Tuple of (best_params, best_score).
        """
        if not OPTUNA_AVAILABLE:
            self.logger.warning("Optuna not available, returning default parameters")
            return {}, 0.0

        self.logger.info(f'🚀 Running Optuna HPO with pruning for {model_name}...')

        try:
            # Determine trial count based on model
            model_trial_mapping = {
                'lightgbm': self.config.get('lightgbm_trials', 50),
                'xgboost': self.config.get('xgboost_trials', 50),
                'svm': self.config.get('svm_trials', 30),
                'random_forest': self.config.get('random_forest_trials', 40),
                'neural_network': self.config.get('neural_network_trials', 25)
            }
            total_trials = model_trial_mapping.get(model_name, self.config.get('n_trials', 50))

            # Create study
            study_direction = 'maximize'
            study = optuna.create_study(
                direction=study_direction,
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
            )

            # Define objective function
            def objective(trial):
                return self._objective_function(
                    trial, model_name, X_train, y_train, X_val, y_val
                )

            # Run optimization
            study.optimize(objective, n_trials=total_trials)

            best_params = study.best_params
            best_score = study.best_value

            self.logger.info(f'✅ HPO completed for {model_name}: {best_score:.4f}')
            return best_params, best_score

        except Exception as e:
            self.logger.error(error(f'HPO failed for {model_name}: {e}'))
            return {}, 0.0

    def _objective_function(
        self,
        trial: optuna.trial.Trial,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> float:
        """Objective function for Optuna optimization."""
        try:
            # Skip if insufficient target diversity
            if y_train.nunique() <= 1:
                self.logger.warning(f'Target has only {y_train.nunique()} unique values, skipping optimization')
                return 0.0

            # Get model parameters based on model type
            params = self._get_model_params(trial, model_name)

            # Create and fit model
            model = self._get_model_instance(model_name, params)
            score = self._fit_and_score_model(
                model, model_name, params, X_train, y_train, X_val, y_val
            )

            return score

        except Exception as e:
            self.logger.error(f'Objective function failed: {e}')
            return 0.0

    def _get_model_params(self, trial: optuna.trial.Trial, model_name: str) -> Dict[str, Any]:
        """Get model parameters for the given trial."""
        if model_name == 'lightgbm':
            return self._get_lightgbm_params(trial)
        elif model_name == 'xgboost':
            return self._get_xgboost_params(trial)
        elif model_name == 'svm':
            return self._get_svm_params(trial)
        elif model_name == 'neural_network':
            return self._get_neural_network_params(trial)
        else:
            return self._get_default_params(trial)

    def _get_lightgbm_params(self, trial: optuna.trial.Trial) -> Dict[str, Any]:
        """Get LightGBM parameters for optimization."""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-08, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-08, 10.0, log=True),
            'early_stopping_rounds': 50
        }

    def _get_xgboost_params(self, trial: optuna.trial.Trial) -> Dict[str, Any]:
        """Get XGBoost parameters for optimization."""
        return {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'verbosity': 0
        }

    def _get_svm_params(self, trial: optuna.trial.Trial) -> Dict[str, Any]:
        """Get SVM parameters for optimization."""
        return {
            'C': trial.suggest_float('C', 0.1, 100.0, log=True),
            'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly']),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto'])
        }

    def _get_neural_network_params(self, trial: optuna.trial.Trial) -> Dict[str, Any]:
        """Get Neural Network parameters for optimization."""
        return {
            'hidden_layer_sizes': trial.suggest_categorical(
                'hidden_layer_sizes', [(50,), (100,), (50, 25), (100, 50), (100, 50, 25)]
            ),
            'alpha': trial.suggest_float('alpha', 1e-05, 0.1, log=True),
            'learning_rate_init': trial.suggest_float('learning_rate_init', 0.0001, 0.1, log=True),
            'max_iter': trial.suggest_int('max_iter', 200, 1000)
        }

    def _get_default_params(self, trial: optuna.trial.Trial) -> Dict[str, Any]:
        """Get default parameters for Random Forest."""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20)
        }

    def _get_model_instance(self, model_name: str, params: Dict[str, Any]):
        """Get model instance based on model name and parameters."""
        if model_name == 'lightgbm':
            import lightgbm as lgb
            return lgb.LGBMClassifier(**params)
        elif model_name == 'xgboost':
            import xgboost as xgb
            return xgb.XGBClassifier(**params)
        elif model_name == 'svm':
            from sklearn.svm import SVC
            return SVC(**params)
        elif model_name == 'neural_network':
            from sklearn.neural_network import MLPClassifier
            return MLPClassifier(**params)
        else:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**params)

    def _fit_and_score_model(
        self,
        model,
        model_name: str,
        params: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> float:
        """Fit model and return score."""
        try:
            if model_name == 'lightgbm':
                score = self._fit_lightgbm_model(model, params, X_train, y_train, X_val, y_val)
            elif model_name == 'xgboost':
                score = self._fit_xgboost_model(model, X_train, y_train, X_val, y_val)
            else:
                score = self._fit_default_model(model, model_name, X_train, y_train, X_val, y_val)

            return score

        except Exception as e:
            self.logger.error(f'Failed to fit {model_name}: {e}')
            return 0.0

    def _fit_lightgbm_model(
        self,
        model,
        params: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> float:
        """Fit LightGBM model with timeout protection."""
        def timeout_handler(signum, frame):
            raise TimeoutError('LightGBM training timed out')

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(300)

        try:
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        finally:
            sys.stdout = old_stdout
            signal.alarm(0)

        # Calculate score
        y_proba = model.predict_proba(X_val)
        labels_sorted = sorted(pd.unique(pd.concat([y_train, y_val])))

        try:
            loss = log_loss(y_val, y_proba, labels=labels_sorted)
        except Exception:
            loss = log_loss(y_val, y_proba)

        return float(loss)

    def _fit_xgboost_model(self, model, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> float:
        """Fit XGBoost model."""
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        preds = model.predict(X_val)
        return accuracy_score(y_val, preds)

    def _fit_default_model(self, model, model_name: str, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> float:
        """Fit default model (Random Forest, SVM, etc.)."""
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return accuracy_score(y_val, preds)

__all__ = ['HyperparameterOptimizer']
