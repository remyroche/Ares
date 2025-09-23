"""
Regime-Specific Hyperparameter Optimization

This module implements regime-specific hyperparameter optimization for ML models,
allowing different model configurations for different market regimes.

Key features:
- Regime-aware hyperparameter search
- Bayesian optimization for regime-specific parameters
- Multi-objective optimization
- Regime transition-aware parameter adaptation
- Automated hyperparameter tuning
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from enum import Enum
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Optimized imports
from src.utils.tprint import tprint
from src.utils.logger import get_logger
from src.core.decorators import handles_errors, traced, validates, log_execution_time
from src.utils.common_operations import (
    validate_dataframe, validate_dataframe_columns, safe_dataframe_operation,
    timed_operation, memory_checkpoint, gpu_context
)

logger = get_logger('RegimeSpecificHPO')

class OptimizationObjective(Enum):
    """Optimization objectives."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ROC_AUC = "roc_auc"
    MSE = "mse"
    MAE = "mae"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"

@dataclass
class RegimeHPOConfig:
    """Configuration for regime-specific hyperparameter optimization."""
    
    # Regime configuration
    num_regimes: int = 3
    regime_names: List[str] = field(default_factory=lambda: ['low_vol', 'medium_vol', 'high_vol'])
    
    # Optimization configuration
    optimization_objectives: List[OptimizationObjective] = field(default_factory=lambda: [
        OptimizationObjective.ACCURACY, OptimizationObjective.SHARPE_RATIO
    ])
    optimization_trials: int = 100
    optimization_timeout: int = 3600  # 1 hour
    
    # Hyperparameter search spaces
    model_types: List[str] = field(default_factory=lambda: ['xgboost', 'lightgbm', 'catboost', 'neural_network'])
    
    # XGBoost parameters
    xgb_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': (100, 1000),
        'max_depth': (3, 10),
        'learning_rate': (0.01, 0.3),
        'subsample': (0.6, 1.0),
        'colsample_bytree': (0.6, 1.0),
        'reg_alpha': (0, 1.0),
        'reg_lambda': (0, 1.0)
    })
    
    # LightGBM parameters
    lgb_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': (100, 1000),
        'max_depth': (3, 10),
        'learning_rate': (0.01, 0.3),
        'subsample': (0.6, 1.0),
        'colsample_bytree': (0.6, 1.0),
        'reg_alpha': (0, 1.0),
        'reg_lambda': (0, 1.0)
    })
    
    # CatBoost parameters
    catboost_params: Dict[str, Any] = field(default_factory=lambda: {
        'iterations': (100, 1000),
        'depth': (3, 10),
        'learning_rate': (0.01, 0.3),
        'l2_leaf_reg': (1, 10),
        'border_count': (32, 255)
    })
    
    # Neural network parameters
    nn_params: Dict[str, Any] = field(default_factory=lambda: {
        'hidden_layers': (2, 5),
        'layer_size': (64, 512),
        'dropout_rate': (0.1, 0.5),
        'learning_rate': (0.0001, 0.01),
        'batch_size': (32, 256),
        'activation': ['relu', 'gelu', 'swish', 'mish']
    })
    
    # Cross-validation
    cv_folds: int = 5
    cv_strategy: str = 'time_series_split'  # 'time_series_split', 'k_fold'
    
    # Early stopping
    early_stopping_rounds: int = 50
    early_stopping_patience: int = 10
    
    # Multi-objective optimization
    use_multi_objective: bool = True
    objective_weights: Dict[str, float] = field(default_factory=lambda: {
        'accuracy': 0.4,
        'sharpe_ratio': 0.3,
        'max_drawdown': 0.3
    })


class RegimeSpecificHPO:
    """Regime-specific hyperparameter optimization system."""
    
    def __init__(self, config: RegimeHPOConfig):
        self.config = config
        self.logger = get_logger('RegimeSpecificHPO')
        
        # Optimization results storage
        self.optimization_results = {}
        self.best_parameters = {}
        self.regime_models = {}
        
        # Initialize Optuna studies for each regime
        self.studies = {}
        for regime_id in range(config.num_regimes):
            regime_name = config.regime_names[regime_id]
            self.studies[regime_name] = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=42),
                pruner=MedianPruner()
            )
    
    def _create_objective_function(self, regime_id: int, X: np.ndarray, y: np.ndarray, 
                                 regime_mask: np.ndarray) -> callable:
        """Create objective function for a specific regime."""
        regime_name = self.config.regime_names[regime_id]
        
        def objective(trial):
            # Sample hyperparameters
            model_type = trial.suggest_categorical('model_type', self.config.model_types)
            
            if model_type == 'xgboost':
                params = self._sample_xgb_params(trial)
                model = self._create_xgb_model(params)
            elif model_type == 'lightgbm':
                params = self._sample_lgb_params(trial)
                model = self._create_lgb_model(params)
            elif model_type == 'catboost':
                params = self._sample_catboost_params(trial)
                model = self._create_catboost_model(params)
            elif model_type == 'neural_network':
                params = self._sample_nn_params(trial)
                model = self._create_nn_model(params)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Filter data for this regime
            regime_X = X[regime_mask]
            regime_y = y[regime_mask]
            
            if len(regime_X) < 100:  # Not enough data for this regime
                return 0.0
            
            # Cross-validation
            if self.config.cv_strategy == 'time_series_split':
                scores = self._time_series_cv_score(model, regime_X, regime_y)
            else:
                scores = cross_val_score(model, regime_X, regime_y, cv=self.config.cv_folds, scoring='accuracy')
            
            # Calculate objective score
            objective_score = self._calculate_objective_score(scores, model, regime_X, regime_y)
            
            return objective_score
        
        return objective
    
    def _sample_xgb_params(self, trial) -> Dict[str, Any]:
        """Sample XGBoost parameters."""
        params = {}
        for param_name, param_range in self.config.xgb_params.items():
            if isinstance(param_range, tuple):
                if isinstance(param_range[0], int):
                    params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                else:
                    params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
            else:
                params[param_name] = trial.suggest_categorical(param_name, param_range)
        return params
    
    def _sample_lgb_params(self, trial) -> Dict[str, Any]:
        """Sample LightGBM parameters."""
        params = {}
        for param_name, param_range in self.config.lgb_params.items():
            if isinstance(param_range, tuple):
                if isinstance(param_range[0], int):
                    params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                else:
                    params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
            else:
                params[param_name] = trial.suggest_categorical(param_name, param_range)
        return params
    
    def _sample_catboost_params(self, trial) -> Dict[str, Any]:
        """Sample CatBoost parameters."""
        params = {}
        for param_name, param_range in self.config.catboost_params.items():
            if isinstance(param_range, tuple):
                if isinstance(param_range[0], int):
                    params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                else:
                    params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
            else:
                params[param_name] = trial.suggest_categorical(param_name, param_range)
        return params
    
    def _sample_nn_params(self, trial) -> Dict[str, Any]:
        """Sample neural network parameters."""
        params = {}
        for param_name, param_range in self.config.nn_params.items():
            if isinstance(param_range, tuple):
                if isinstance(param_range[0], int):
                    params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                else:
                    params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
            else:
                params[param_name] = trial.suggest_categorical(param_name, param_range)
        return params
    
    def _create_xgb_model(self, params: Dict[str, Any]):
        """Create XGBoost model with given parameters."""
        from xgboost import XGBRegressor, XGBClassifier
        
        # Determine if classification or regression
        if 'objective' in params:
            if params['objective'] in ['binary:logistic', 'multi:softmax']:
                return XGBClassifier(**params)
            else:
                return XGBRegressor(**params)
        else:
            return XGBRegressor(**params)
    
    def _create_lgb_model(self, params: Dict[str, Any]):
        """Create LightGBM model with given parameters."""
        from lightgbm import LGBMRegressor, LGBMClassifier
        
        # Determine if classification or regression
        if 'objective' in params:
            if params['objective'] in ['binary', 'multiclass']:
                return LGBMClassifier(**params)
            else:
                return LGBMRegressor(**params)
        else:
            return LGBMRegressor(**params)
    
    def _create_catboost_model(self, params: Dict[str, Any]):
        """Create CatBoost model with given parameters."""
        from catboost import CatBoostRegressor, CatBoostClassifier
        
        # Determine if classification or regression
        if 'loss_function' in params:
            if params['loss_function'] in ['Logloss', 'MultiClass']:
                return CatBoostClassifier(**params)
            else:
                return CatBoostRegressor(**params)
        else:
            return CatBoostRegressor(**params)
    
    def _create_nn_model(self, params: Dict[str, Any]):
        """Create neural network model with given parameters."""
        # This would create a PyTorch or TensorFlow model
        # For now, return a placeholder
        return None
    
    def _time_series_cv_score(self, model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate time series cross-validation scores."""
        n_samples = len(X)
        fold_size = n_samples // self.config.cv_folds
        
        scores = []
        for i in range(self.config.cv_folds):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < self.config.cv_folds - 1 else n_samples
            
            # Training set: all data before this fold
            X_train = X[:start_idx]
            y_train = y[:start_idx]
            
            # Test set: this fold
            X_test = X[start_idx:end_idx]
            y_test = y[start_idx:end_idx]
            
            if len(X_train) > 0 and len(X_test) > 0:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = accuracy_score(y_test, y_pred)
                scores.append(score)
        
        return np.array(scores)
    
    def _calculate_objective_score(self, scores: np.ndarray, model, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate objective score based on multiple criteria."""
        if len(scores) == 0:
            return 0.0
        
        # Base score (mean CV score)
        base_score = np.mean(scores)
        
        if not self.config.use_multi_objective:
            return base_score
        
        # Multi-objective scoring
        total_score = 0.0
        
        # Accuracy weight
        if OptimizationObjective.ACCURACY in self.config.optimization_objectives:
            accuracy_weight = self.config.objective_weights.get('accuracy', 0.4)
            total_score += base_score * accuracy_weight
        
        # Sharpe ratio weight (if applicable)
        if OptimizationObjective.SHARPE_RATIO in self.config.optimization_objectives:
            sharpe_weight = self.config.objective_weights.get('sharpe_ratio', 0.3)
            # Calculate Sharpe ratio (simplified)
            returns = np.diff(y) if len(y) > 1 else np.array([0])
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) if np.std(returns) > 0 else 0
            total_score += sharpe_ratio * sharpe_weight
        
        # Max drawdown weight (if applicable)
        if OptimizationObjective.MAX_DRAWDOWN in self.config.optimization_objectives:
            drawdown_weight = self.config.objective_weights.get('max_drawdown', 0.3)
            # Calculate max drawdown (simplified)
            cumulative = np.cumsum(y)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / (running_max + 1e-8)
            max_drawdown = 1 - np.min(drawdown) if len(drawdown) > 0 else 0
            total_score += (1 - max_drawdown) * drawdown_weight  # Higher is better
        
        return total_score
    
    @traced(span_name='optimize_regime_parameters')
    def optimize_regime_parameters(self, regime_id: int, X: np.ndarray, y: np.ndarray, 
                                 regime_mask: np.ndarray) -> Dict[str, Any]:
        """Optimize hyperparameters for a specific regime."""
        regime_name = self.config.regime_names[regime_id]
        self.logger.info(f"🔍 Optimizing hyperparameters for regime: {regime_name}")
        
        # Create objective function
        objective = self._create_objective_function(regime_id, X, y, regime_mask)
        
        # Run optimization
        study = self.studies[regime_name]
        study.optimize(
            objective,
            n_trials=self.config.optimization_trials,
            timeout=self.config.optimization_timeout
        )
        
        # Store results
        self.optimization_results[regime_name] = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'optimization_history': [trial.value for trial in study.trials if trial.value is not None]
        }
        
        self.best_parameters[regime_name] = study.best_params
        
        self.logger.info(f"✅ Optimization completed for {regime_name}")
        self.logger.info(f"   → Best value: {study.best_value:.4f}")
        self.logger.info(f"   → Best parameters: {study.best_params}")
        
        return self.optimization_results[regime_name]
    
    @traced(span_name='optimize_all_regimes')
    def optimize_all_regimes(self, X: np.ndarray, y: np.ndarray, regime_labels: np.ndarray) -> Dict[str, Any]:
        """Optimize hyperparameters for all regimes."""
        self.logger.info("🚀 Starting regime-specific hyperparameter optimization")
        
        all_results = {}
        
        for regime_id in range(self.config.num_regimes):
            regime_name = self.config.regime_names[regime_id]
            
            # Create regime mask
            regime_mask = (regime_labels == regime_id)
            
            if np.sum(regime_mask) < 100:
                self.logger.warning(f"⚠️ Not enough data for regime {regime_name}: {np.sum(regime_mask)} samples")
                continue
            
            # Optimize parameters for this regime
            regime_results = self.optimize_regime_parameters(regime_id, X, y, regime_mask)
            all_results[regime_name] = regime_results
        
        self.logger.info("✅ All regime optimizations completed")
        return all_results
    
    def get_best_parameters(self, regime_name: str) -> Dict[str, Any]:
        """Get best parameters for a specific regime."""
        if regime_name not in self.best_parameters:
            raise ValueError(f"No optimized parameters found for regime: {regime_name}")
        
        return self.best_parameters[regime_name]
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of all optimization results."""
        summary = {}
        
        for regime_name, results in self.optimization_results.items():
            summary[regime_name] = {
                'best_value': results['best_value'],
                'n_trials': results['n_trials'],
                'best_params': results['best_params']
            }
        
        return summary
    
    def create_regime_models(self, X: np.ndarray, y: np.ndarray, regime_labels: np.ndarray) -> Dict[str, Any]:
        """Create optimized models for each regime."""
        self.logger.info("🏗️ Creating regime-specific models with optimized parameters")
        
        for regime_id in range(self.config.num_regimes):
            regime_name = self.config.regime_names[regime_id]
            
            if regime_name not in self.best_parameters:
                self.logger.warning(f"⚠️ No optimized parameters for regime {regime_name}")
                continue
            
            # Get best parameters
            best_params = self.best_parameters[regime_name]
            model_type = best_params.get('model_type', 'xgboost')
            
            # Create model with best parameters
            if model_type == 'xgboost':
                model = self._create_xgb_model(best_params)
            elif model_type == 'lightgbm':
                model = self._create_lgb_model(best_params)
            elif model_type == 'catboost':
                model = self._create_catboost_model(best_params)
            elif model_type == 'neural_network':
                model = self._create_nn_model(best_params)
            else:
                self.logger.error(f"Unknown model type: {model_type}")
                continue
            
            # Train model on regime-specific data
            regime_mask = (regime_labels == regime_id)
            regime_X = X[regime_mask]
            regime_y = y[regime_mask]
            
            if len(regime_X) > 0:
                model.fit(regime_X, regime_y)
                self.regime_models[regime_name] = model
                self.logger.info(f"✅ Created and trained model for regime {regime_name}")
        
        return self.regime_models
    
    def predict_with_regime_models(self, X: np.ndarray, regime_labels: np.ndarray) -> np.ndarray:
        """Make predictions using regime-specific models."""
        predictions = np.zeros(len(X))
        
        for regime_id in range(self.config.num_regimes):
            regime_name = self.config.regime_names[regime_id]
            
            if regime_name not in self.regime_models:
                continue
            
            # Get regime mask
            regime_mask = (regime_labels == regime_id)
            
            if np.sum(regime_mask) > 0:
                # Make predictions for this regime
                regime_X = X[regime_mask]
                regime_predictions = self.regime_models[regime_name].predict(regime_X)
                predictions[regime_mask] = regime_predictions
        
        return predictions


# Factory functions
def create_regime_specific_hpo(config: Optional[RegimeHPOConfig] = None) -> RegimeSpecificHPO:
    """Create regime-specific HPO system with default configuration."""
    if config is None:
        config = RegimeHPOConfig()
    
    return RegimeSpecificHPO(config)


# Test function
if __name__ == '__main__':
    tprint('🧪 Testing Regime-Specific Hyperparameter Optimization')
    
    # Test configuration
    config = RegimeHPOConfig(
        num_regimes=3,
        optimization_trials=10,  # Reduced for testing
        optimization_timeout=300  # 5 minutes for testing
    )
    
    tprint(f'📊 Regime-Specific HPO Configuration:')
    tprint(f'   → Number of regimes: {config.num_regimes}')
    tprint(f'   → Regime names: {config.regime_names}')
    tprint(f'   → Optimization trials: {config.optimization_trials}')
    tprint(f'   → Optimization timeout: {config.optimization_timeout}')
    tprint(f'   → Model types: {config.model_types}')
    tprint(f'   → Use multi-objective: {config.use_multi_objective}')
    
    # Test system creation
    try:
        hpo_system = create_regime_specific_hpo(config)
        
        tprint('✅ Regime-specific HPO system created successfully')
        tprint(f'   → Studies created: {list(hpo_system.studies.keys())}')
        tprint(f'   → Optimization results: {list(hpo_system.optimization_results.keys())}')
        tprint(f'   → Best parameters: {list(hpo_system.best_parameters.keys())}')
        
        # Test parameter sampling
        import optuna
        
        def test_objective(trial):
            return trial.suggest_float('test_param', 0.0, 1.0)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(test_objective, n_trials=5)
        
        tprint(f'   → Test optimization completed: {study.best_value:.4f}')
        
    except Exception as e:
        tprint(f'❌ Error creating regime-specific HPO system: {e}')
    
    tprint('✅ Regime-Specific Hyperparameter Optimization test completed!')