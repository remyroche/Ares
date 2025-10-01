"""
Comprehensive Overfitting Prevention for Multi-Output Stacking Ensemble

This module provides comprehensive overfitting prevention strategies for all models
in the multi-output stacking ensemble system, including regularization, validation,
early stopping, and ensemble diversity measures.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime

from src.utils.logger import system_logger

logger = system_logger.getChild('OverfittingPrevention')

@dataclass
class OverfittingPreventionConfig:
    """Configuration for overfitting prevention strategies."""
    
    # General settings
    enable_early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    early_stopping_monitor: str = 'val_loss'
    
    # Cross-validation settings
    enable_cross_validation: bool = True
    cv_folds: int = 5
    cv_strategy: str = 'time_series_split'  # time_series_split, kfold, stratified_kfold
    
    # Regularization settings
    enable_regularization: bool = True
    l1_regularization: float = 0.01
    l2_regularization: float = 0.01
    dropout_rate: float = 0.2
    
    # Ensemble diversity
    enable_ensemble_diversity: bool = True
    diversity_threshold: float = 0.7
    enable_bagging: bool = True
    bagging_fraction: float = 0.8
    
    # Model-specific settings
    model_specific_configs: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'TCN': {
            'dropout': 0.2,
            'recurrent_dropout': 0.1,
            'l2_regularization': 0.01,
            'early_stopping_patience': 15
        },
        'DeepScaler': {
            'dropout': 0.2,
            'l2_regularization': 0.01,
            'early_stopping_patience': 15,
            'use_batch_norm': True,
            'use_residual_connections': True
        },
        'NBEATS': {
            'dropout': 0.1,
            'l2_regularization': 0.001,
            'early_stopping_patience': 20,
            'regime_aware_training': True,
            'regime_feature_integration': True
        },
        'FinancialResNet': {
            'dropout': 0.15,
            'l2_regularization': 0.01,
            'early_stopping_patience': 25,
            'regime_aware': True,
            'use_batch_norm': True,
            'use_layer_norm': True
        },
        'AdvancedMambaHybrid': {
            'dropout': 0.1,
            'l2_regularization': 0.01,
            'early_stopping_patience': 20,
            'multi_timeframe_fusion': True,
            'execution_optimization': False,
            'micro_timing_attention': False,
            'latency_aware': False
        },
        'DeepScaler1m': {
            'dropout': 0.1,
            'l2_regularization': 0.005,
            'early_stopping_patience': 30,
            'precision_focused': True,
            'micro_timing_aware': True
        },
        'NODE': {
            'lambda_sparse': 1e-3,
            'gamma': 1.5,
            'dropout': 0.1,
            'l2_regularization': 0.01
        },
        'CatBoostRegressor': {
            'l2_leaf_reg': 3.0,
            'bagging_temperature': 1.0,
            'subsample': 0.8,
            'colsample_bylevel': 0.8,
            'early_stopping_rounds': 50
        },
        'LGBMRegressor': {
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_samples': 20,
            'early_stopping_rounds': 50
        },
        'XGBRegressor': {
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'max_depth': 6,
            'early_stopping_rounds': 50
        },
        'RandomForestRegressor': {
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True
        },
        'Ridge': {
            'alpha': 1.0,
            'solver': 'auto'
        }
    })
    
    # Validation settings
    validation_split: float = 0.2
    test_split: float = 0.1
    enable_holdout_validation: bool = True
    
    # Performance monitoring
    enable_performance_monitoring: bool = True
    overfitting_threshold: float = 0.1  # Max difference between train and val performance
    enable_learning_curves: bool = True

class OverfittingPrevention:
    """
    Comprehensive overfitting prevention for multi-output stacking ensemble.
    
    This class provides various strategies to prevent overfitting across all models:
    1. Regularization techniques
    2. Early stopping
    3. Cross-validation
    4. Ensemble diversity
    5. Model-specific optimizations
    """
    
    def __init__(self, config: Optional[OverfittingPreventionConfig] = None):
        """Initialize overfitting prevention."""
        self.config = config or OverfittingPreventionConfig()
        self.logger = logger.getChild('OverfittingPrevention')
        
        # Initialize monitoring
        self.performance_history = []
        self.overfitting_detected = False
        
        self.logger.info("✅ Overfitting Prevention initialized")
    
    def apply_regularization(self, model: Any, model_type: str) -> Any:
        """Apply appropriate regularization to model based on type."""
        
        self.logger.debug(f"🔄 Applying regularization to {model_type}")
        
        # Get model-specific config
        model_config = self.config.model_specific_configs.get(model_type, {})
        
        try:
            if hasattr(model, 'set_params'):
                # Apply regularization parameters
                regularization_params = self._get_regularization_params(model_type, model_config)
                model.set_params(**regularization_params)
                self.logger.debug(f"✅ Applied regularization to {model_type}: {regularization_params}")
            else:
                self.logger.warning(f"⚠️ Model {model_type} does not support parameter setting")
        
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to apply regularization to {model_type}: {e}")
        
        return model
    
    def _get_regularization_params(self, model_type: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get regularization parameters for specific model type."""
        
        params = {}
        
        if model_type == 'TCN':
            params.update({
                'dropout': model_config.get('dropout', self.config.dropout_rate),
                'recurrent_dropout': model_config.get('recurrent_dropout', 0.1),
                'kernel_regularizer': f"l2({model_config.get('l2_regularization', self.config.l2_regularization)})"
            })
        
        elif model_type == 'NODE':
            params.update({
                'lambda_sparse': model_config.get('lambda_sparse', 1e-3),
                'gamma': model_config.get('gamma', 1.5),
                'dropout': model_config.get('dropout', self.config.dropout_rate)
            })
        
        elif model_type == 'CatBoostRegressor':
            params.update({
                'l2_leaf_reg': model_config.get('l2_leaf_reg', 3.0),
                'bagging_temperature': model_config.get('bagging_temperature', 1.0),
                'subsample': model_config.get('subsample', self.config.bagging_fraction),
                'colsample_bylevel': model_config.get('colsample_bylevel', 0.8),
                'early_stopping_rounds': model_config.get('early_stopping_rounds', 50)
            })
        
        elif model_type == 'LGBMRegressor':
            params.update({
                'reg_alpha': model_config.get('reg_alpha', self.config.l1_regularization),
                'reg_lambda': model_config.get('reg_lambda', self.config.l2_regularization),
                'subsample': model_config.get('subsample', self.config.bagging_fraction),
                'colsample_bytree': model_config.get('colsample_bytree', 0.8),
                'min_child_samples': model_config.get('min_child_samples', 20),
                'early_stopping_rounds': model_config.get('early_stopping_rounds', 50)
            })
        
        elif model_type == 'RandomForestRegressor':
            params.update({
                'max_depth': model_config.get('max_depth', 10),
                'min_samples_split': model_config.get('min_samples_split', 5),
                'min_samples_leaf': model_config.get('min_samples_leaf', 2),
                'max_features': model_config.get('max_features', 'sqrt'),
                'bootstrap': model_config.get('bootstrap', True)
            })
        
        elif model_type == 'Ridge':
            params.update({
                'alpha': model_config.get('alpha', 1.0),
                'solver': model_config.get('solver', 'auto')
            })

        elif model_type == 'DeepScaler':
            params.update({
                'dropout': model_config.get('dropout', self.config.dropout_rate),
                'l2_regularization': model_config.get('l2_regularization', self.config.l2_regularization),
                'early_stopping_patience': model_config.get('early_stopping_patience', 15),
                'use_batch_norm': model_config.get('use_batch_norm', True),
                'use_residual_connections': model_config.get('use_residual_connections', True)
            })

        elif model_type == 'NBEATS':
            params.update({
                'dropout': model_config.get('dropout', self.config.dropout_rate),
                'l2_regularization': model_config.get('l2_regularization', 0.001),
                'early_stopping_patience': model_config.get('early_stopping_patience', 20),
                'regime_aware_training': model_config.get('regime_aware_training', True),
                'regime_feature_integration': model_config.get('regime_feature_integration', True)
            })

        elif model_type == 'FinancialResNet':
            params.update({
                'dropout': model_config.get('dropout', self.config.dropout_rate),
                'l2_regularization': model_config.get('l2_regularization', self.config.l2_regularization),
                'early_stopping_patience': model_config.get('early_stopping_patience', 25),
                'regime_aware': model_config.get('regime_aware', True),
                'use_batch_norm': model_config.get('use_batch_norm', True),
                'use_layer_norm': model_config.get('use_layer_norm', True)
            })

        elif model_type == 'AdvancedMambaHybrid':
            params.update({
                'dropout': model_config.get('dropout', self.config.dropout_rate),
                'l2_regularization': model_config.get('l2_regularization', self.config.l2_regularization),
                'early_stopping_patience': model_config.get('early_stopping_patience', 20),
                'multi_timeframe_fusion': model_config.get('multi_timeframe_fusion', True),
                'execution_optimization': model_config.get('execution_optimization', False),
                'micro_timing_attention': model_config.get('micro_timing_attention', False),
                'latency_aware': model_config.get('latency_aware', False)
            })

        elif model_type == 'DeepScaler1m':
            params.update({
                'dropout': model_config.get('dropout', self.config.dropout_rate),
                'l2_regularization': model_config.get('l2_regularization', 0.005),
                'early_stopping_patience': model_config.get('early_stopping_patience', 30),
                'precision_focused': model_config.get('precision_focused', True),
                'micro_timing_aware': model_config.get('micro_timing_aware', True)
            })

        elif model_type == 'XGBRegressor':
            params.update({
                'reg_alpha': model_config.get('reg_alpha', self.config.l1_regularization),
                'reg_lambda': model_config.get('reg_lambda', self.config.l2_regularization),
                'subsample': model_config.get('subsample', self.config.bagging_fraction),
                'colsample_bytree': model_config.get('colsample_bytree', 0.8),
                'max_depth': model_config.get('max_depth', 6),
                'early_stopping_rounds': model_config.get('early_stopping_rounds', 50)
            })

        return params
    
    def setup_early_stopping(self, model: Any, model_type: str) -> Dict[str, Any]:
        """Setup early stopping for model."""
        
        if not self.config.enable_early_stopping:
            return {}
        
        self.logger.debug(f"🔄 Setting up early stopping for {model_type}")
        
        early_stopping_config = {
            'patience': self.config.early_stopping_patience,
            'min_delta': self.config.early_stopping_min_delta,
            'monitor': self.config.early_stopping_monitor,
            'restore_best_weights': True,
            'verbose': 1
        }
        
        # Model-specific early stopping
        model_config = self.config.model_specific_configs.get(model_type, {})
        if 'early_stopping_patience' in model_config:
            early_stopping_config['patience'] = model_config['early_stopping_patience']
        
        return early_stopping_config
    
    def perform_cross_validation(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str
    ) -> Dict[str, Any]:
        """Perform cross-validation to assess overfitting."""
        
        if not self.config.enable_cross_validation:
            return {}
        
        self.logger.info(f"🔄 Performing cross-validation for {model_name}")
        
        try:
            from sklearn.model_selection import cross_val_score, TimeSeriesSplit, KFold, StratifiedKFold
            from sklearn.metrics import mean_squared_error, r2_score
            
            # Select CV strategy
            if self.config.cv_strategy == 'time_series_split':
                cv = TimeSeriesSplit(n_splits=self.config.cv_folds)
            elif self.config.cv_strategy == 'kfold':
                # ⚠️ Using TimeSeriesSplit instead to prevent data leakage
                cv = TimeSeriesSplit(n_splits=self.config.cv_folds)
            elif self.config.cv_strategy == 'stratified_kfold':
                # ⚠️ WARNING: For time series, use 'time_series' strategy instead!
                cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=False)
            else:
                cv = TimeSeriesSplit(n_splits=self.config.cv_folds)
            
            # Perform cross-validation
            try:
                from src.utils.ml_common.validation.unified_cv import perform_cross_validation as unified_perform_cv
                cv_res = unified_perform_cv(model, X, y, strategy='temporal', cv_folds=self.config.cv_folds, scoring='neg_mean_squared_error')
                cv_scores = np.array(cv_res.get('scores', []) or [])
            except Exception:
                cv_scores = np.array([])
            
            # Calculate metrics
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            cv_scores_positive = -cv_scores  # Convert to positive MSE
            
            # Check for overfitting indicators
            overfitting_indicators = self._check_overfitting_indicators(cv_scores_positive)
            
            cv_results = {
                'cv_scores': cv_scores.tolist(),
                'cv_mean': float(cv_mean),
                'cv_std': float(cv_std),
                'cv_mean_mse': float(np.mean(cv_scores_positive)),
                'cv_std_mse': float(np.std(cv_scores_positive)),
                'overfitting_indicators': overfitting_indicators,
                'cv_strategy': self.config.cv_strategy,
                'n_folds': self.config.cv_folds
            }
            
            self.logger.info(f"✅ Cross-validation completed for {model_name}: "
                           f"CV MSE = {cv_results['cv_mean_mse']:.4f} ± {cv_results['cv_std_mse']:.4f}")
            
            return cv_results
            
        except Exception as e:
            self.logger.error(f"❌ Cross-validation failed for {model_name}: {e}")
            return {'error': str(e)}
    
    def _check_overfitting_indicators(self, cv_scores: np.ndarray) -> Dict[str, Any]:
        """Check for overfitting indicators in CV scores."""
        
        indicators = {
            'high_variance': False,
            'decreasing_performance': False,
            'unstable_performance': False
        }
        
        # Check for high variance (CV std > 20% of mean)
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        if cv_std > 0.2 * cv_mean:
            indicators['high_variance'] = True
        
        # Check for decreasing performance over folds
        if len(cv_scores) > 2:
            trend = np.polyfit(range(len(cv_scores)), cv_scores, 1)[0]
            if trend > 0.1 * cv_mean:  # Increasing MSE (decreasing performance)
                indicators['decreasing_performance'] = True
        
        # Check for unstable performance (large jumps)
        if len(cv_scores) > 1:
            max_jump = np.max(np.abs(np.diff(cv_scores)))
            if max_jump > 0.5 * cv_mean:
                indicators['unstable_performance'] = True
        
        return indicators
    
    def monitor_performance(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_name: str
    ) -> Dict[str, Any]:
        """Monitor model performance for overfitting detection."""
        
        if not self.config.enable_performance_monitoring:
            return {}
        
        self.logger.debug(f"🔄 Monitoring performance for {model_name}")
        
        try:
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            
            # Calculate metrics
            
            train_mse = mean_squared_error(y_train, y_train_pred)
            val_mse = mean_squared_error(y_val, y_val_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            val_r2 = r2_score(y_val, y_val_pred)
            
            # Calculate performance gap
            mse_gap = val_mse - train_mse
            r2_gap = train_r2 - val_r2
            
            # Check for overfitting
            overfitting_detected = (
                mse_gap > self.config.overfitting_threshold * train_mse or
                r2_gap > self.config.overfitting_threshold
            )
            
            if overfitting_detected:
                self.overfitting_detected = True
                self.logger.warning(f"⚠️ Overfitting detected in {model_name}: "
                                  f"MSE gap = {mse_gap:.4f}, R² gap = {r2_gap:.4f}")
            
            # Record performance
            performance_record = {
                'model_name': model_name,
                'timestamp': datetime.now(),
                'train_mse': float(train_mse),
                'val_mse': float(val_mse),
                'train_r2': float(train_r2),
                'val_r2': float(val_r2),
                'mse_gap': float(mse_gap),
                'r2_gap': float(r2_gap),
                'overfitting_detected': overfitting_detected
            }
            
            self.performance_history.append(performance_record)
            
            return performance_record
            
        except Exception as e:
            self.logger.error(f"❌ Performance monitoring failed for {model_name}: {e}")
            return {'error': str(e)}
    
    def ensure_ensemble_diversity(
        self,
        base_models: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """Ensure ensemble diversity to prevent overfitting."""
        
        if not self.config.enable_ensemble_diversity:
            return {}
        
        self.logger.info("🔄 Ensuring ensemble diversity")
        
        try:
            # Get predictions from all base models
            model_predictions = {}
            for model_name, model in base_models.items():
                try:
                    pred = model.predict(X)
                    if pred.ndim == 1:
                        pred = pred.reshape(-1, 1)
                    model_predictions[model_name] = pred
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to get predictions from {model_name}: {e}")
                    continue
            
            if len(model_predictions) < 2:
                self.logger.warning("⚠️ Not enough models for diversity analysis")
                return {'diversity_score': 0.0, 'diverse_models': []}
            
            # Calculate pairwise correlations
            model_names = list(model_predictions.keys())
            correlations = np.zeros((len(model_names), len(model_names)))
            
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names):
                    if i != j:
                        corr = np.corrcoef(
                            model_predictions[model1].flatten(),
                            model_predictions[model2].flatten()
                        )[0, 1]
                        correlations[i, j] = corr if not np.isnan(corr) else 0.0
            
            # Calculate diversity metrics
            avg_correlation = np.mean(correlations[correlations != 0])
            diversity_score = 1 - avg_correlation
            
            # Identify diverse models (low correlation with others)
            diverse_models = []
            for i, model_name in enumerate(model_names):
                model_avg_corr = np.mean(correlations[i][correlations[i] != 0])
                if model_avg_corr < self.config.diversity_threshold:
                    diverse_models.append(model_name)
            
            diversity_results = {
                'diversity_score': float(diversity_score),
                'avg_correlation': float(avg_correlation),
                'diverse_models': diverse_models,
                'correlation_matrix': correlations.tolist(),
                'model_names': model_names
            }
            
            self.logger.info(f"✅ Ensemble diversity analysis completed: "
                           f"diversity score = {diversity_score:.3f}, "
                           f"diverse models = {len(diverse_models)}")
            
            return diversity_results
            
        except Exception as e:
            self.logger.error(f"❌ Ensemble diversity analysis failed: {e}")
            return {'error': str(e)}
    
    def apply_bagging(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply bagging to reduce overfitting."""
        
        if not self.config.enable_bagging:
            return X, y
        
        self.logger.debug("🔄 Applying bagging")
        
        try:
            # Random sampling with replacement
            n_samples = len(X)
            bag_size = int(n_samples * self.config.bagging_fraction)
            
            # Sample indices
            bag_indices = np.random.choice(n_samples, size=bag_size, replace=True)
            
            # Create bagged dataset
            X_bagged = X[bag_indices]
            y_bagged = y[bag_indices]
            
            self.logger.debug(f"✅ Bagging applied: {bag_size} samples from {n_samples}")
            
            return X_bagged, y_bagged
            
        except Exception as e:
            self.logger.error(f"❌ Bagging failed: {e}")
            return X, y
    
    def get_overfitting_summary(self) -> Dict[str, Any]:
        """Get summary of overfitting prevention measures."""
        
        summary = {
            'overfitting_detected': self.overfitting_detected,
            'total_models_monitored': len(self.performance_history),
            'overfitting_models': [
                record['model_name'] for record in self.performance_history
                if record.get('overfitting_detected', False)
            ],
            'avg_performance_gap': 0.0,
            'recommendations': []
        }
        
        if self.performance_history:
            # Calculate average performance gap
            mse_gaps = [record.get('mse_gap', 0) for record in self.performance_history]
            r2_gaps = [record.get('r2_gap', 0) for record in self.performance_history]
            
            summary['avg_performance_gap'] = float(np.mean(mse_gaps + r2_gaps))
            
            # Generate recommendations
            if self.overfitting_detected:
                summary['recommendations'].extend([
                    "Increase regularization strength",
                    "Reduce model complexity",
                    "Increase training data",
                    "Use early stopping more aggressively"
                ])
            
            if summary['avg_performance_gap'] > 0.05:
                summary['recommendations'].append("Consider ensemble methods")
            
            if len(summary['overfitting_models']) > len(self.performance_history) * 0.5:
                summary['recommendations'].append("Review overall model architecture")
        
        return summary

# Convenience functions
def create_overfitting_prevention(config: Optional[OverfittingPreventionConfig] = None) -> OverfittingPrevention:
    """Create overfitting prevention instance."""
    return OverfittingPrevention(config)

def apply_comprehensive_regularization(
    model: Any,
    model_type: str,
    config: Optional[OverfittingPreventionConfig] = None
) -> Any:
    """Apply comprehensive regularization to model."""
    prevention = create_overfitting_prevention(config)
    return prevention.apply_regularization(model, model_type)