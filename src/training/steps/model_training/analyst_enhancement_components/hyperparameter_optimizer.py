"""Hyperparameter optimization component for analyst enhancement."""
import asyncio
from typing import Any, Dict, Optional
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from src.core.decorators import handles_errors, log_execution_time
from src.utils.logger import system_logger
from typing import Dict, List, Optional, Union, Any, Tuple
from src.core.decorators.errors import handles_errors

class HyperparameterOptimizer:
    """Handles hyperparameter optimization for analyst models."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the hyperparameter optimizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get('hyperparameter_optimization', {})
        self.logger = system_logger.getChild('hyperparameter_optimizer')
        self.n_trials = self.config.get('n_trials', 50)
        self.timeout = self.config.get('timeout', 300)
        self.n_jobs = self.config.get('n_jobs', -1)
        self.pruning = self.config.get('pruning', True)
        self.search_spaces = self._initialize_search_spaces()

    def _initialize_search_spaces(self) -> Dict[str, Dict[str, Any]]:
        """Initialize model-specific hyperparameter search spaces."""
        return {'lightgbm': {'num_leaves': (10, 100), 'learning_rate': (0.01, 0.3), 'feature_fraction': (0.5, 1.0), 'bagging_fraction': (0.5, 1.0), 'bagging_freq': (1, 10), 'min_child_samples': (5, 50), 'lambda_l1': (0, 10), 'lambda_l2': (0, 10)}, 'xgboost': {'max_depth': (3, 10), 'learning_rate': (0.01, 0.3), 'n_estimators': (50, 300), 'subsample': (0.5, 1.0), 'colsample_bytree': (0.5, 1.0), 'gamma': (0, 5), 'reg_alpha': (0, 10), 'reg_lambda': (0, 10)}, 'random_forest': {'n_estimators': (50, 500), 'max_depth': (5, 50), 'min_samples_split': (2, 20), 'min_samples_leaf': (1, 10), 'max_features': ['sqrt', 'log2', None], 'bootstrap': [True, False]}, 'neural_network': {'hidden_layers': (1, 5), 'hidden_units': (32, 512), 'learning_rate': (0.0001, 0.01), 'dropout_rate': (0.0, 0.5), 'batch_size': (16, 128), 'epochs': (10, 100)}}

    @handles_errors(exceptions=(Exception,), default_return={}, context='hyperparameter optimization')
    async def optimize(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, regime_id: str) -> Dict[str, Any]:
        """Optimize hyperparameters for a model.
        
        Args:
            model: Model to optimize
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            regime_id: Regime identifier
            
        Returns:
            Dictionary of optimized hyperparameters
        """
        model_type = self._get_model_type(model)
        if model_type not in self.search_spaces:
            self.logger.warning(f'No search space defined for model type: {model_type}')
            return {}
        self.logger.info(f'Starting HPO for {model_type} in regime {regime_id}')
        study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner() if self.pruning else None, sampler=optuna.samplers.TPESampler(seed=42))

        def objective(trial: Any) -> float:
            params = self._suggest_params(trial, model_type)
            model_with_params = self._create_model_with_params(model, params)
            try:
                scores = cross_val_score(model_with_params, X_train, y_train, cv=3, scoring='accuracy', n_jobs=1)
                return scores.mean()
            except Exception as e:
                self.logger.warning(f'Trial failed: {str(e)}')
                return 0.0
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout, n_jobs=1, show_progress_bar=False)
        best_params = study.best_params
        best_score = study.best_value
        self.logger.info(f'HPO completed for {model_type}: Best score = {best_score:.4f}, Trials = {len(study.trials)}')
        return best_params

    def _get_model_type(self, model: Any) -> str:
        """Determine the type of model."""
        model_class = model.__class__.__name__.lower()
        if 'lightgbm' in model_class or 'lgb' in model_class:
            return 'lightgbm'
        elif 'xgboost' in model_class or 'xgb' in model_class:
            return 'xgboost'
        elif 'randomforest' in model_class:
            return 'random_forest'
        elif 'neural' in model_class or 'nn' in model_class:
            return 'neural_network'
        else:
            return 'unknown'

    def _suggest_params(self, trial: optuna.Trial, model_type: str) -> Dict[str, Any]:
        """Suggest parameters for a specific model type."""
        search_space = self.search_spaces[model_type]
        params = {}
        for param_name, param_range in search_space.items():
            if isinstance(param_range, tuple) and len(param_range) == 2:
                if isinstance(param_range[0], int):
                    params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                else:
                    params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
            elif isinstance(param_range, list):
                params[param_name] = trial.suggest_categorical(param_name, param_range)
        return params

    def _create_model_with_params(self, base_model: Any, params: Dict[str, Any]) -> Any:
        """Create a new model instance with suggested parameters."""
        try:
            model_class = base_model.__class__
            if hasattr(base_model, 'get_params'):
                base_params = base_model.get_params()
            else:
                base_params = {}
            base_params.update(params)
            return model_class(**base_params)
        except Exception as e:
            self.logger.warning(f'Failed to create model with params: {str(e)}')
            return base_model