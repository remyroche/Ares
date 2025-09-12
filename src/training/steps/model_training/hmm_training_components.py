
import numpy as np
from src.utils.logger import system_logger
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
from src.utils.intensity_scaler import (
    get_intensity_from_environment, get_scaled_hpo_trials, 
    get_scaled_hpo_timeout, log_intensity_info, apply_intensity_scaling
)
from .regime_data_integration import RegimeDataIntegrator

"""HMM training components for model training.

This module contains specialized components for HMM-based model training,
including regime-specific training, multi-output models, and optimization.
"""
from typing import Any, Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from src.utils.logger import system_logger

import optuna

import lightgbm as lgb
import logging

class HMMModelTrainer:
    """Trains HMM-based models that predict regime membership probabilities."""
    @log_important_calls

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize HMM model trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.logger = system_logger.getChild('HMMModelTrainer')
        self.model_types = config.get('model_types', ['random_forest', 'lightgbm'])
        
        # Apply intensity scaling to config
        intensity_pct = get_intensity_from_environment()
        if intensity_pct < 1.0:
            self.config = apply_intensity_scaling(self.config, intensity_pct)
            self.logger.info(f"🔧 Applied intensity scaling ({intensity_pct*100:.0f}%) to HMM training config")

    async def train_models(self, prepared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train models that predict regime membership probabilities.
        
        Args:
            prepared_data: Prepared training data with regime information
            
        Returns:
            Training results with regime probabilities and confidence scores
        """
        results = {
            'models': {}, 
            'performance': {}, 
            'feature_importance': {},
            'regime_probabilities': {},
            'regime_confidence': {}
        }
        
        if 'train' not in prepared_data or 'val' not in prepared_data:
            self.logger.error('Missing train or validation data')
            return results
            
        train_data = prepared_data['train']
        val_data = prepared_data['val']
        
        # Check for regime information
        if 'regime_labels' not in train_data:
            self.logger.error('Missing regime labels in training data')
            return results
            
        for model_type in self.model_types:
            self.logger.info(f'Training {model_type} model for regime probability prediction...')
            try:
                if model_type == 'lightgbm':
                    model_results = await self._train_lightgbm_regime(train_data, val_data)
                elif model_type == 'random_forest':
                    model_results = await self._train_random_forest_regime(train_data, val_data)
                elif model_type == 'xgboost':
                    model_results = await self._train_xgboost_regime(train_data, val_data)
                else:
                    self.logger.warning(f'Unknown model type: {model_type}')
                    continue
                    
                results['models'][model_type] = model_results['model']
                results['performance'][model_type] = model_results['performance']
                results['feature_importance'][model_type] = model_results.get('feature_importance', {})
                results['regime_probabilities'][model_type] = model_results.get('regime_probabilities', [])
                results['regime_confidence'][model_type] = model_results.get('regime_confidence', [])
                
            except Exception as e:
                self.logger.error(f'Failed to train {model_type}: {e}')
                
        return results

    async def _train_lightgbm_regime(self, train_data: Dict[str, Any], val_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train LightGBM model for regime probability prediction."""
        unique_labels = np.unique(train_data['regime_labels'])
        is_classification = len(unique_labels) < 50  # Support up to 40 regimes
        
        params = {
            'objective': 'multiclass' if is_classification and len(unique_labels) > 2 else 'binary' if is_classification else 'regression',
            'metric': 'multi_logloss' if is_classification and len(unique_labels) > 2 else 'binary_logloss' if is_classification else 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'num_threads': 4
        }
        
        if is_classification and len(unique_labels) > 2:
            params['num_class'] = len(unique_labels)
            
        train_dataset = lgb.Dataset(train_data['features'], label=train_data['regime_labels'], feature_name=train_data['feature_names'])
        val_dataset = lgb.Dataset(val_data['features'], label=val_data['regime_labels'], reference=train_dataset)
        
        model = lgb.train(params, train_dataset, valid_sets=[val_dataset], num_boost_round=100, callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)])
        
        # Get predictions and probabilities
        val_pred = model.predict(val_data['features'], num_iteration=model.best_iteration)
        val_pred_proba = model.predict_proba(val_data['features'], num_iteration=model.best_iteration) if hasattr(model, 'predict_proba') else val_pred
        
        if is_classification:
            if len(unique_labels) > 2:
                val_pred_class = np.argmax(val_pred, axis=1)
            else:
                val_pred_class = (val_pred > 0.5).astype(int)
            performance = {
                'accuracy': accuracy_score(val_data['regime_labels'], val_pred_class),
                'f1_score': f1_score(val_data['regime_labels'], val_pred_class, average='weighted')
            }
        else:
            performance = {
                'mse': mean_squared_error(val_data['regime_labels'], val_pred),
                'r2_score': r2_score(val_data['regime_labels'], val_pred)
            }
            
        # Calculate regime confidence
        regime_confidence = np.max(val_pred_proba, axis=1) if len(val_pred_proba.shape) > 1 else np.abs(val_pred_proba)
        
        importance = model.feature_importance(importance_type='gain')
        feature_importance = {train_data['feature_names'][i]: float(importance[i]) for i in range(len(train_data['feature_names']))}
        
        return {
            'model': model,
            'performance': performance,
            'feature_importance': feature_importance,
            'regime_probabilities': val_pred_proba,
            'regime_confidence': regime_confidence
        }

    async def _train_random_forest_regime(self, train_data: Dict[str, Any], val_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train Random Forest model for regime probability prediction."""
        unique_labels = np.unique(train_data['regime_labels'])
        is_classification = len(unique_labels) < 50  # Support up to 40 regimes
        
        if is_classification:
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=4
            )
        else:
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=4
            )
            
        model.fit(train_data['features'], train_data['regime_labels'])
        
        # Get predictions and probabilities
        val_pred = model.predict(val_data['features'])
        val_pred_proba = model.predict_proba(val_data['features']) if hasattr(model, 'predict_proba') else val_pred
        
        if is_classification:
            performance = {
                'accuracy': accuracy_score(val_data['regime_labels'], val_pred),
                'f1_score': f1_score(val_data['regime_labels'], val_pred, average='weighted')
            }
        else:
            performance = {
                'mse': mean_squared_error(val_data['regime_labels'], val_pred),
                'r2_score': r2_score(val_data['regime_labels'], val_pred)
            }
            
        # Calculate regime confidence
        regime_confidence = np.max(val_pred_proba, axis=1) if len(val_pred_proba.shape) > 1 else np.abs(val_pred_proba)
        
        feature_importance = {train_data['feature_names'][i]: float(model.feature_importances_[i]) for i in range(len(train_data['feature_names']))}
        
        return {
            'model': model,
            'performance': performance,
            'feature_importance': feature_importance,
            'regime_probabilities': val_pred_proba,
            'regime_confidence': regime_confidence
        }

    async def _train_xgboost_regime(self, train_data: Dict[str, Any], val_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train XGBoost model for regime probability prediction."""
        try:
            import xgboost as xgb
            
            unique_labels = np.unique(train_data['regime_labels'])
            is_classification = len(unique_labels) < 50  # Support up to 40 regimes
            
            if is_classification:
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=4
                )
            else:
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=4
                )
                
            model.fit(train_data['features'], train_data['regime_labels'])
            
            # Get predictions and probabilities
            val_pred = model.predict(val_data['features'])
            val_pred_proba = model.predict_proba(val_data['features']) if hasattr(model, 'predict_proba') else val_pred
            
            if is_classification:
                performance = {
                    'accuracy': accuracy_score(val_data['regime_labels'], val_pred),
                    'f1_score': f1_score(val_data['regime_labels'], val_pred, average='weighted')
                }
            else:
                performance = {
                    'mse': mean_squared_error(val_data['regime_labels'], val_pred),
                    'r2_score': r2_score(val_data['regime_labels'], val_pred)
                }
                
            # Calculate regime confidence
            regime_confidence = np.max(val_pred_proba, axis=1) if len(val_pred_proba.shape) > 1 else np.abs(val_pred_proba)
            
            feature_importance = {train_data['feature_names'][i]: float(model.feature_importances_[i]) for i in range(len(train_data['feature_names']))}
            
            return {
                'model': model,
                'performance': performance,
                'feature_importance': feature_importance,
                'regime_probabilities': val_pred_proba,
                'regime_confidence': regime_confidence
            }
            
        except ImportError:
            self.logger.warning('XGBoost not available, using fallback')
            return {
                'model': None,
                'performance': {},
                'feature_importance': {},
                'regime_probabilities': [],
                'regime_confidence': []
            }



class HyperparameterOptimizer:
    """Optimizes model hyperparameters."""
    @log_important_calls

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize hyperparameter optimizer.
        
        Args:
            config: Optimization configuration
        """
        self.config = config
        self.logger = system_logger.getChild('HyperparameterOptimizer')
        self.n_trials = config.get('n_trials', 50)
        self.cv_folds = config.get('cv_folds', 5)
        
        # Apply intensity scaling
        intensity_pct = get_intensity_from_environment()
        if intensity_pct < 1.0:
            self.n_trials = get_scaled_hpo_trials(self.n_trials, intensity_pct)
            self.logger.info(f"🔧 Applied intensity scaling ({intensity_pct*100:.0f}%): HPO trials={self.n_trials}")

    async def optimize_hyperparameters(self, model_type: str, train_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize hyperparameters for a model type.
        
        Args:
            model_type: Type of model to optimize
            train_data: Training data
            
        Returns:
            Optimal hyperparameters
        """
        try:

            def objective(trial: Any) -> float:
                if model_type == 'lightgbm':
                    params = {'num_leaves': trial.suggest_int('num_leaves', 10, 100), 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log = True), 'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0), 'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0), 'bagging_freq': trial.suggest_int('bagging_freq', 1, 10), 'min_child_samples': trial.suggest_int('min_child_samples', 5, 50)}
                elif model_type == 'random_forest':
                    params = {'n_estimators': trial.suggest_int('n_estimators', 50, 300), 'max_depth': trial.suggest_int('max_depth', 5, 30), 'min_samples_split': trial.suggest_int('min_samples_split', 2, 20), 'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)}
                else:
                    return 0.0
                return self._evaluate_params(model_type, params, train_data)
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials = self.n_trials)
            return {'best_params': study.best_params, 'best_score': study.best_value}
        except ImportError:
            self.logger.warning('Optuna not available, using default parameters')
            return {'best_params': {}, 'best_score': 0.0}
    @log_all_calls

    def _evaluate_params(self, model_type: str, params: Dict[str, Any], train_data: Dict[str, Any]) -> float:
        """Evaluate parameters using cross-validation.
        
        Args:
            model_type: Type of model
            params: Parameters to evaluate
            train_data: Training data
            
        Returns:
            Cross-validation score
        """
        return np.random.random()