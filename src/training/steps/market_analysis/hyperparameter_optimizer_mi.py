"""
Hyperparameter Optimization Framework for MI Improvement

This framework provides:
- MI-focused hyperparameter optimization
- Bayesian optimization for MI > 0.02 target
- Feature selection based on MI contribution
- Automated model selection and validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.optimize import minimize
import optuna

logger = logging.getLogger(__name__)


class OptimizationStatus(Enum):
    """Optimization status enumeration."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    STOPPED = "STOPPED"


@dataclass
class OptimizationResult:
    """Optimization result data structure."""
    specialist_name: str
    best_params: Dict[str, Any]
    best_mi: float
    best_auc: float
    best_accuracy: float
    n_trials: int
    optimization_time: float
    feature_importance: Dict[str, float]
    status: OptimizationStatus
    trial_history: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'specialist_name': self.specialist_name,
            'best_params': self.best_params,
            'best_mi': self.best_mi,
            'best_auc': self.best_auc,
            'best_accuracy': self.best_accuracy,
            'n_trials': self.n_trials,
            'optimization_time': self.optimization_time,
            'feature_importance': self.feature_importance,
            'status': self.status.value,
            'trial_history': self.trial_history
        }


class MIHyperparameterOptimizer:
    """Hyperparameter optimizer focused on MI improvement."""
    
    def __init__(self, target_mi: float = 0.02, max_trials: int = 100, 
                 timeout: int = 3600, cv_folds: int = 3):
        self.target_mi = target_mi
        self.max_trials = max_trials
        self.timeout = timeout
        self.cv_folds = cv_folds
        self.logger = logging.getLogger(self.__class__.__name__)
        self.optimization_history: List[OptimizationResult] = []
        
    def optimize_lightgbm_mi(self, X: pd.DataFrame, y: pd.Series, 
                           specialist_name: str) -> OptimizationResult:
        """Optimize LightGBM hyperparameters for MI improvement."""
        
        self.logger.info(f"🔧 Starting LightGBM MI optimization for {specialist_name}")
        
        def objective(trial):
            # Define parameter search space
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'num_leaves': trial.suggest_int('num_leaves', 15, 127),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 200),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'verbose': -1,
                'random_state': 42
            }
            
            try:
                # Time series cross-validation
                tscv = TimeSeriesSplit(n_splits=self.cv_folds)
                mi_scores = []
                auc_scores = []
                
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    # Train LightGBM
                    import lightgbm as lgb
                    model = lgb.LGBMClassifier(**params)
                    
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                             callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
                    
                    # Predictions
                    val_pred = model.predict_proba(X_val)[:, 1]
                    
                    # Compute MI
                    mi_score = mutual_info_regression(
                        val_pred.reshape(-1, 1), y_val.values
                    )[0]
                    mi_scores.append(mi_score)
                    
                    # Compute AUC
                    auc_score = roc_auc_score(y_val, val_pred)
                    auc_scores.append(auc_score)
                
                avg_mi = np.mean(mi_scores)
                avg_auc = np.mean(auc_scores)
                
                # Store trial information
                trial.set_user_attr('mi_score', avg_mi)
                trial.set_user_attr('auc_score', avg_auc)
                trial.set_user_attr('params', params)
                
                # Optimize for MI with AUC as secondary objective
                if avg_mi < self.target_mi:
                    # Penalize low MI heavily
                    return -(avg_mi * 1000 + avg_auc)
                else:
                    # Reward high MI
                    return -(avg_mi * 1000 + avg_auc)
                
            except Exception as e:
                self.logger.warning(f"Trial failed: {e}")
                return -1.0  # Return worst possible score
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            study_name=f'{specialist_name}_lgb_mi_optimization'
        )
        
        # Optimize
        start_time = datetime.utcnow()
        
        study.optimize(
            objective,
            n_trials=self.max_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        optimization_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Get best result
        best_trial = study.best_trial
        best_params = best_trial.user_attrs['params']
        best_mi = best_trial.user_attrs['mi_score']
        best_auc = best_trial.user_attrs['auc_score']
        
        # Compute feature importance
        feature_importance = self._compute_feature_importance_lgb(X, y, best_params)
        
        # Compile trial history
        trial_history = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                trial_history.append({
                    'trial_number': trial.number,
                    'mi_score': trial.user_attrs.get('mi_score', 0.0),
                    'auc_score': trial.user_attrs.get('auc_score', 0.0),
                    'params': trial.user_attrs.get('params', {}),
                    'value': trial.value
                })
        
        result = OptimizationResult(
            specialist_name=specialist_name,
            best_params=best_params,
            best_mi=best_mi,
            best_auc=best_auc,
            best_accuracy=0.0,  # Not computed for this optimization
            n_trials=len(study.trials),
            optimization_time=optimization_time,
            feature_importance=feature_importance,
            status=OptimizationStatus.COMPLETED,
            trial_history=trial_history
        )
        
        self.optimization_history.append(result)
        
        self.logger.info(f"✅ LightGBM MI optimization completed for {specialist_name}")
        self.logger.info(f"   Best MI: {best_mi:.4f} (target: {self.target_mi})")
        self.logger.info(f"   Best AUC: {best_auc:.3f}")
        self.logger.info(f"   Trials: {len(study.trials)}")
        
        return result
    
    def optimize_xgboost_mi(self, X: pd.DataFrame, y: pd.Series, 
                          specialist_name: str) -> OptimizationResult:
        """Optimize XGBoost hyperparameters for MI improvement."""
        
        self.logger.info(f"🔧 Starting XGBoost MI optimization for {specialist_name}")
        
        def objective(trial):
            # Define parameter search space
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0.0, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 100),
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'use_label_encoder': False,
                'random_state': 42
            }
            
            try:
                # Time series cross-validation
                tscv = TimeSeriesSplit(n_splits=self.cv_folds)
                mi_scores = []
                auc_scores = []
                
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    # Train XGBoost
                    import xgboost as xgb
                    model = xgb.XGBClassifier(**params)
                    
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                             early_stopping_rounds=30, verbose=False)
                    
                    # Predictions
                    val_pred = model.predict_proba(X_val)[:, 1]
                    
                    # Compute MI
                    mi_score = mutual_info_regression(
                        val_pred.reshape(-1, 1), y_val.values
                    )[0]
                    mi_scores.append(mi_score)
                    
                    # Compute AUC
                    auc_score = roc_auc_score(y_val, val_pred)
                    auc_scores.append(auc_score)
                
                avg_mi = np.mean(mi_scores)
                avg_auc = np.mean(auc_scores)
                
                # Store trial information
                trial.set_user_attr('mi_score', avg_mi)
                trial.set_user_attr('auc_score', avg_auc)
                trial.set_user_attr('params', params)
                
                # Optimize for MI with AUC as secondary objective
                if avg_mi < self.target_mi:
                    return -(avg_mi * 1000 + avg_auc)
                else:
                    return -(avg_mi * 1000 + avg_auc)
                
            except Exception as e:
                self.logger.warning(f"Trial failed: {e}")
                return -1.0
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            study_name=f'{specialist_name}_xgb_mi_optimization'
        )
        
        # Optimize
        start_time = datetime.utcnow()
        
        study.optimize(
            objective,
            n_trials=self.max_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        optimization_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Get best result
        best_trial = study.best_trial
        best_params = best_trial.user_attrs['params']
        best_mi = best_trial.user_attrs['mi_score']
        best_auc = best_trial.user_attrs['auc_score']
        
        # Compute feature importance
        feature_importance = self._compute_feature_importance_xgb(X, y, best_params)
        
        # Compile trial history
        trial_history = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                trial_history.append({
                    'trial_number': trial.number,
                    'mi_score': trial.user_attrs.get('mi_score', 0.0),
                    'auc_score': trial.user_attrs.get('auc_score', 0.0),
                    'params': trial.user_attrs.get('params', {}),
                    'value': trial.value
                })
        
        result = OptimizationResult(
            specialist_name=specialist_name,
            best_params=best_params,
            best_mi=best_mi,
            best_auc=best_auc,
            best_accuracy=0.0,
            n_trials=len(study.trials),
            optimization_time=optimization_time,
            feature_importance=feature_importance,
            status=OptimizationStatus.COMPLETED,
            trial_history=trial_history
        )
        
        self.optimization_history.append(result)
        
        self.logger.info(f"✅ XGBoost MI optimization completed for {specialist_name}")
        self.logger.info(f"   Best MI: {best_mi:.4f} (target: {self.target_mi})")
        self.logger.info(f"   Best AUC: {best_auc:.3f}")
        self.logger.info(f"   Trials: {len(study.trials)}")
        
        return result
    
    def _compute_feature_importance_lgb(self, X: pd.DataFrame, y: pd.Series, 
                                    params: Dict[str, Any]) -> Dict[str, float]:
        """Compute feature importance for LightGBM."""
        try:
            import lightgbm as lgb
            
            # Train model on full data
            model = lgb.LGBMClassifier(**params)
            model.fit(X, y)
            
            # Get feature importance
            importance = model.feature_importances_
            feature_names = X.columns
            
            return dict(zip(feature_names, importance))
            
        except Exception as e:
            self.logger.warning(f"Feature importance computation failed: {e}")
            return {}
    
    def _compute_feature_importance_xgb(self, X: pd.DataFrame, y: pd.Series, 
                                    params: Dict[str, Any]) -> Dict[str, float]:
        """Compute feature importance for XGBoost."""
        try:
            import xgboost as xgb
            
            # Train model on full data
            model = xgb.XGBClassifier(**params)
            model.fit(X, y)
            
            # Get feature importance
            importance = model.feature_importances_
            feature_names = X.columns
            
            return dict(zip(feature_names, importance))
            
        except Exception as e:
            self.logger.warning(f"Feature importance computation failed: {e}")
            return {}
    
    def select_features_by_mi(self, X: pd.DataFrame, y: pd.Series, 
                           top_k: int = 50, mi_threshold: float = 0.01) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Select features based on MI contribution."""
        
        self.logger.info(f"🔍 Selecting top {top_k} features by MI contribution")
        
        feature_mi_scores = {}
        
        for col in X.select_dtypes(include=[np.number]).columns:
            try:
                mi_score = mutual_info_regression(
                    X[col].values.reshape(-1, 1), y.values
                )[0]
                feature_mi_scores[col] = mi_score
            except Exception:
                feature_mi_scores[col] = 0.0
        
        # Sort by MI score
        sorted_features = sorted(feature_mi_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select top features
        selected_features = [col for col, mi in sorted_features[:top_k] if mi >= mi_threshold]
        
        if not selected_features:
            self.logger.warning("No features met MI threshold, selecting top 10")
            selected_features = [col for col, mi in sorted_features[:10]]
        
        selected_df = X[selected_features]
        selected_mi_scores = {col: feature_mi_scores[col] for col in selected_features}
        
        self.logger.info(f"   Selected {len(selected_features)} features")
        self.logger.info(f"   Average MI: {np.mean(list(selected_mi_scores.values())):.4f}")
        
        return selected_df, selected_mi_scores
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of all optimizations."""
        
        if not self.optimization_history:
            return {'error': 'No optimization history available'}
        
        summary = {
            'total_optimizations': len(self.optimization_history),
            'target_mi': self.target_mi,
            'successful_optimizations': 0,
            'average_mi': 0.0,
            'best_mi': 0.0,
            'average_trials': 0.0,
            'average_time': 0.0,
            'specialists': {}
        }
        
        mi_scores = []
        trial_counts = []
        times = []
        
        for result in self.optimization_history:
            if result.status == OptimizationStatus.COMPLETED:
                summary['successful_optimizations'] += 1
                mi_scores.append(result.best_mi)
                trial_counts.append(result.n_trials)
                times.append(result.optimization_time)
                
                summary['specialists'][result.specialist_name] = {
                    'best_mi': result.best_mi,
                    'best_auc': result.best_auc,
                    'n_trials': result.n_trials,
                    'optimization_time': result.optimization_time,
                    'target_met': result.best_mi >= self.target_mi
                }
        
        if mi_scores:
            summary['average_mi'] = np.mean(mi_scores)
            summary['best_mi'] = np.max(mi_scores)
            summary['average_trials'] = np.mean(trial_counts)
            summary['average_time'] = np.mean(times)
        
        return summary
    
    def export_results(self, filepath: str, format: str = 'json'):
        """Export optimization results to file."""
        
        export_data = {
            'summary': self.get_optimization_summary(),
            'optimizations': [result.to_dict() for result in self.optimization_history],
            'export_timestamp': datetime.utcnow().isoformat()
        }
        
        if format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Optimization results exported to {filepath}")


# Global optimizer instance
mi_optimizer = MIHyperparameterOptimizer()


def get_mi_optimizer() -> MIHyperparameterOptimizer:
    """Get the global MI optimizer instance."""
    return mi_optimizer


def optimize_specialist_mi(X: pd.DataFrame, y: pd.Series, specialist_name: str, 
                          model_type: str = 'lightgbm') -> OptimizationResult:
    """Convenience function to optimize a specialist for MI."""
    optimizer = get_mi_optimizer()
    
    if model_type.lower() == 'lightgbm':
        return optimizer.optimize_lightgbm_mi(X, y, specialist_name)
    elif model_type.lower() == 'xgboost':
        return optimizer.optimize_xgboost_mi(X, y, specialist_name)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
