"""
Layer 4 Multi-Model Position Sizing Implementation

Implements LGBM, CatBoost, and XGBoost models for position sizing
with Huber teacher constraints and PnL optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
import lightgbm as lgb
import xgboost as xgb
import optuna
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit

# Import CatBoost if available
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
from src.utils.huber_regressor_for_trees import prepare_huber_teacher_outputs

def train_layer4_lgbm(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: Optional[np.ndarray] = None,
    huber_outputs: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    n_trials: int = 50
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train LightGBM model for Layer 4 position sizing.
    """
    tprint_info("🌳 Training Layer 4 LightGBM model...")
    
    # Extract Huber constraints if available
    monotonic_cst = None
    if huber_outputs and 'monotonic_constraints' in huber_outputs:
        huber_monotonic_constraints = huber_outputs['monotonic_constraints']
        monotonic_cst = np.array([huber_monotonic_constraints.get(col, 0) for col in X.columns])
    
    # Add Huber warm start if available
    if huber_outputs and 'warm_start' in huber_outputs and huber_outputs['warm_start'] is not None:
        huber_warm_start = huber_outputs['warm_start']
        if len(huber_warm_start) == len(X):
            X = X.copy()
            X['huber_baseline'] = huber_warm_start
            tprint_info("🎓 Added Huber warm start baseline as feature")
    
    # LightGBM parameters
    default_params = {
        "learning_rate": 0.05,              # Standard
        "num_leaves": 31,                   # Balanced
        "min_child_samples": 20,            # Regularization
        "subsample": 0.7,                   # Row subsampling
        "subsample_freq": 1,                 # Frequent subsampling
        "colsample_bytree": 0.7,            # Column subsampling
        "colsample_bynode": 0.7,            # Node-level subsampling
        "reg_lambda": 10.0,                 # L2 regularization
        "min_split_gain": 0.005,            # Regularization
        "linear_tree": True,                # Linear trees
        "path_smooth": 20,                  # Smoothing
        "extra_trees": True,                 # Random Forest behavior
        "boosting_type": "dart",            # DART boosting
        "n_jobs": -1,
        "verbosity": -1,
        "objective": "binary",
        "metric": "binary_logloss"
    }
    
    # Add constraints if available
    if monotonic_cst is not None:
        default_params["monotone_constraints"] = monotonic_cst
    
    # Hyperparameter optimization
    def objective(trial):
        params = default_params.copy()
        
        # Suggest hyperparameters
        params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.1)
        params["num_leaves"] = trial.suggest_int("num_leaves", 15, 63)
        params["min_child_samples"] = trial.suggest_int("min_child_samples", 10, 50)
        params["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.5, 0.9)
        params["reg_lambda"] = trial.suggest_float("reg_lambda", 1.0, 50.0)
        
        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        pnl_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Create datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # Train model
            model = lgb.train(
                params,
                train_data,
                num_boost_round=500,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(30, verbose=False)]
            )
            
            # Predict and calculate PnL
            probas = model.predict(X_val, num_iteration=model.best_iteration)
            positions = np.where(probas > 0.6, 1, np.where(probas < 0.4, -1, 0))
            # Simple PnL calculation (would use actual returns in practice)
            pnl = positions * np.sign(y_val - 0.5)  # Simplified
            
            # Calculate Sortino ratio
            if len(pnl) > 1:
                downside_returns = pnl[pnl < 0]
                if len(downside_returns) > 0:
                    downside_std = np.std(downside_returns)
                    sortino = pnl.mean() / (downside_std + 1e-8) * np.sqrt(365 * 24 * 4)
                else:
                    sortino = pnl.mean() * np.sqrt(365 * 24 * 4)
            else:
                sortino = 0
            
            pnl_scores.append(sortino)
        
        return np.mean(pnl_scores)
    
    # Optimize
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=n_trials, timeout=300)
    
    # Train final model
    best_params = study.best_params
    final_params = default_params.copy()
    final_params.update(best_params)
    
    train_data = lgb.Dataset(X, label=y, weight=sample_weight)
    model = lgb.train(
        final_params,
        train_data,
        num_boost_round=500,
        callbacks=[lgb.log_evaluation(0)]
    )
    
    metadata = {
        'model_type': 'lgbm',
        'best_params': best_params,
        'best_score': study.best_value,
        'n_features': len(X.columns),
        'constraints_applied': monotonic_cst is not None
    }
    
    tprint_success(f"✅ LightGBM trained successfully (Score: {study.best_value:.4f})")
    
    return model, metadata

def train_layer4_catboost(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: Optional[np.ndarray] = None,
    huber_outputs: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    n_trials: int = 50
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train CatBoost model for Layer 4 position sizing.
    """
    if not CATBOOST_AVAILABLE:
        tprint_warning("⚠️ CatBoost not available, skipping")
        return None, {}
    
    tprint_info("🐈 Training Layer 4 CatBoost model...")
    
    # Extract Huber constraints
    monotonic_cst = None
    if huber_outputs and 'monotonic_constraints' in huber_outputs:
        huber_monotonic_constraints = huber_outputs['monotonic_constraints']
        monotonic_cst = np.array([huber_monotonic_constraints.get(col, 0) for col in X.columns])
    
    # Add Huber warm start
    if huber_outputs and 'warm_start' in huber_outputs and huber_outputs['warm_start'] is not None:
        huber_warm_start = huber_outputs['warm_start']
        if len(huber_warm_start) == len(X):
            X = X.copy()
            X['huber_baseline'] = huber_warm_start
            tprint_info("🎓 Added Huber warm start baseline as feature")
    
    # CatBoost parameters
    default_params = {
        "iterations": 1000,                 # Boosting rounds
        "learning_rate": 0.05,              # Standard
        "depth": 6,                         # Moderate depth
        "l2_leaf_reg": 20.0,                # L2 regularization
        "subsample": 0.6,                   # Row subsampling
        "rsm": 0.8,                         # Random subspace method
        "bagging_temperature": 1,            # Bagging randomness
        "random_strength": 5.0,              # Feature randomness
        "early_stopping_rounds": 30,        # Native early stopping
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": -1,
        "random_seed": 42,
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "boosting_type": "Dart"             # DART boosting
    }
    
    # Add constraints
    if monotonic_cst is not None and np.any(monotonic_cst != 0):
        default_params["monotone_constraints"] = monotonic_cst
    
    # Hyperparameter optimization
    def objective(trial):
        params = default_params.copy()
        
        params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.1)
        params["depth"] = trial.suggest_int("depth", 4, 10)
        params["l2_leaf_reg"] = trial.suggest_float("l2_leaf_reg", 1.0, 50.0)
        params["subsample"] = trial.suggest_float("subsample", 0.5, 0.9)
        params["rsm"] = trial.suggest_float("rsm", 0.5, 0.9)
        
        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        pnl_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Create pools
            train_pool = cb.Pool(X_train, label=y_train, weight=sample_weight[train_idx] if sample_weight is not None else None)
            val_pool = cb.Pool(X_val, label=y_val)
            
            # Train model
            model = cb.CatBoost(**params)
            model.fit(train_pool, eval_set=val_pool, verbose=False)
            
            # Predict and calculate PnL
            probas = model.predict_proba(X_val)[:, 1]
            positions = np.where(probas > 0.6, 1, np.where(probas < 0.4, -1, 0))
            pnl = positions * np.sign(y_val - 0.5)
            
            # Calculate Sortino
            if len(pnl) > 1:
                downside_returns = pnl[pnl < 0]
                if len(downside_returns) > 0:
                    downside_std = np.std(downside_returns)
                    sortino = pnl.mean() / (downside_std + 1e-8) * np.sqrt(365 * 24 * 4)
                else:
                    sortino = pnl.mean() * np.sqrt(365 * 24 * 4)
            else:
                sortino = 0
            
            pnl_scores.append(sortino)
        
        return np.mean(pnl_scores)
    
    # Optimize
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=n_trials, timeout=300)
    
    # Train final model
    best_params = study.best_params
    final_params = default_params.copy()
    final_params.update(best_params)
    
    train_pool = cb.Pool(X, label=y, weight=sample_weight)
    model = cb.CatBoost(**final_params)
    model.fit(train_pool, verbose=False)
    
    metadata = {
        'model_type': 'catboost',
        'best_params': best_params,
        'best_score': study.best_value,
        'n_features': len(X.columns),
        'constraints_applied': monotonic_cst is not None
    }
    
    tprint_success(f"✅ CatBoost trained successfully (Score: {study.best_value:.4f})")
    
    return model, metadata

def train_layer4_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: Optional[np.ndarray] = None,
    huber_outputs: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    n_trials: int = 50
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train XGBoost model for Layer 4 position sizing.
    """
    tprint_info("🌲 Training Layer 4 XGBoost model...")
    
    # Extract Huber constraints
    monotonic_cst = None
    if huber_outputs and 'monotonic_constraints' in huber_outputs:
        huber_monotonic_constraints = huber_outputs['monotonic_constraints']
        monotonic_cst = np.array([huber_monotonic_constraints.get(col, 0) for col in X.columns])
    
    # Add Huber warm start
    if huber_outputs and 'warm_start' in huber_outputs and huber_outputs['warm_start'] is not None:
        huber_warm_start = huber_outputs['warm_start']
        if len(huber_warm_start) == len(X):
            X = X.copy()
            X['huber_baseline'] = huber_warm_start
            tprint_info("🎓 Added Huber warm start baseline as feature")
    
    # XGBoost parameters
    default_params = {
        'n_estimators': 500,
        'learning_rate': 0.05,             # Conservative
        'max_depth': 4,                     # Shallower than LGBM
        'min_child_weight': 10,            # Regularization
        'gamma': 0.5,                      # High regularization
        'subsample': 0.6,                  # Row subsampling
        'colsample_bytree': 0.6,           # Column subsampling
        'colsample_bynode': 0.4,           # Node-level subsampling
        'reg_alpha': 0.3,                   # L1 regularization
        'reg_lambda': 30,                   # Strong L2
        'num_parallel_tree': 7,             # Random Forest behavior
        'n_jobs': -1,
        'verbosity': 0,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'early_stopping_rounds': 30
    }
    
    # Add constraints
    if monotonic_cst is not None:
        default_params['monotone_constraints'] = tuple(monotonic_cst)
    
    # Hyperparameter optimization
    def objective(trial):
        params = default_params.copy()
        
        params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.1)
        params["max_depth"] = trial.suggest_int("max_depth", 3, 8)
        params["min_child_weight"] = trial.suggest_int("min_child_weight", 5, 25)
        params["subsample"] = trial.suggest_float("subsample", 0.5, 0.9)
        params["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.5, 0.9)
        params["reg_lambda"] = trial.suggest_float("reg_lambda", 1.0, 50.0)
        params["reg_alpha"] = trial.suggest_float("reg_alpha", 0.0, 5.0)
        
        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        pnl_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                sample_weight=sample_weight[train_idx] if sample_weight is not None else None,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # Predict and calculate PnL
            probas = model.predict_proba(X_val)[:, 1]
            positions = np.where(probas > 0.6, 1, np.where(probas < 0.4, -1, 0))
            pnl = positions * np.sign(y_val - 0.5)
            
            # Calculate Sortino
            if len(pnl) > 1:
                downside_returns = pnl[pnl < 0]
                if len(downside_returns) > 0:
                    downside_std = np.std(downside_returns)
                    sortino = pnl.mean() / (downside_std + 1e-8) * np.sqrt(365 * 24 * 4)
                else:
                    sortino = pnl.mean() * np.sqrt(365 * 24 * 4)
            else:
                sortino = 0
            
            pnl_scores.append(sortino)
        
        return np.mean(pnl_scores)
    
    # Optimize
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=n_trials, timeout=300)
    
    # Train final model
    best_params = study.best_params
    final_params = default_params.copy()
    final_params.update(best_params)
    
    model = xgb.XGBClassifier(**final_params)
    model.fit(X, y, sample_weight=sample_weight, verbose=False)
    
    metadata = {
        'model_type': 'xgboost',
        'best_params': best_params,
        'best_score': study.best_value,
        'n_features': len(X.columns),
        'constraints_applied': monotonic_cst is not None
    }
    
    tprint_success(f"✅ XGBoost trained successfully (Score: {study.best_value:.4f})")
    
    return model, metadata

def train_layer4_elasticnet(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: Optional[np.ndarray] = None,
    huber_outputs: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    n_trials: int = 30
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train ElasticNet model for Layer 4 position sizing.
    """
    tprint_info("🔗 Training Layer 4 ElasticNet model...")
    
    # Add Huber warm start if available
    if huber_outputs and 'warm_start' in huber_outputs and huber_outputs['warm_start'] is not None:
        huber_warm_start = huber_outputs['warm_start']
        if len(huber_warm_start) == len(X):
            X = X.copy()
            X['huber_baseline'] = huber_warm_start
            tprint_info("🎓 Added Huber warm start baseline as feature")
    
    # Scale features for ElasticNet
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # ElasticNet parameters (fixed as requested)
    elastic_params = {
        'loss': 'log_loss',
        'penalty': 'elasticnet',
        'l1_ratio': 0.3,
        'alpha': 2.5,
        'max_iter': 5000,
        'tol': 1e-4,
        'fit_intercept': True,
        'random_state': 42,
        'class_weight': 'balanced',
        'n_jobs': -1
    }
    
    # Train model
    model = SGDClassifier(**elastic_params)
    model.fit(X_scaled, y, sample_weight=sample_weight)
    
    # Cross-validation for scoring
    tscv = TimeSeriesSplit(n_splits=3)
    pnl_scores = []
    
    for train_idx, val_idx in tscv.split(X_scaled):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model
        cv_model = SGDClassifier(**elastic_params)
        cv_model.fit(X_train, y_train, sample_weight=sample_weight[train_idx] if sample_weight is not None else None)
        
        # Predict and calculate PnL
        probas = cv_model.predict_proba(X_val)[:, 1]
        positions = np.where(probas > 0.6, 1, np.where(probas < 0.4, -1, 0))
        pnl = positions * np.sign(y_val - 0.5)
        
        # Calculate Sortino
        if len(pnl) > 1:
            downside_returns = pnl[pnl < 0]
            if len(downside_returns) > 0:
                downside_std = np.std(downside_returns)
                sortino = pnl.mean() / (downside_std + 1e-8) * np.sqrt(365 * 24 * 4)
            else:
                sortino = pnl.mean() * np.sqrt(365 * 24 * 4)
        else:
            sortino = 0
        
        pnl_scores.append(sortino)
    
    avg_score = np.mean(pnl_scores)
    
    metadata = {
        'model_type': 'elasticnet',
        'best_params': elastic_params,
        'best_score': avg_score,
        'n_features': len(X.columns),
        'constraints_applied': False
    }
    
    tprint_success(f"✅ ElasticNet trained successfully (Score: {avg_score:.4f})")
    
    return model, metadata

def train_layer4_ensemble(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: Optional[np.ndarray] = None,
    huber_outputs: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    n_trials: int = 30
) -> Dict[str, Any]:
    """
    Train ensemble of Layer 4 models and compare performance.
    """
    tprint_info("🚀 Training Layer 4 Multi-Model Ensemble...")
    
    models = {}
    results = {}
    
    # Train LightGBM
    try:
        lgbm_model, lgbm_meta = train_layer4_lgbm(
            X, y, sample_weight, huber_outputs, config, n_trials
        )
        if lgbm_model is not None:
            models['lgbm'] = lgbm_model
            results['lgbm'] = lgbm_meta
    except Exception as e:
        tprint_warning(f"⚠️ LightGBM training failed: {e}")
    
    # Train CatBoost
    try:
        catboost_model, catboost_meta = train_layer4_catboost(
            X, y, sample_weight, huber_outputs, config, n_trials
        )
        if catboost_model is not None:
            models['catboost'] = catboost_model
            results['catboost'] = catboost_meta
    except Exception as e:
        tprint_warning(f"⚠️ CatBoost training failed: {e}")
    
    # Train XGBoost
    try:
        xgb_model, xgb_meta = train_layer4_xgboost(
            X, y, sample_weight, huber_outputs, config, n_trials
        )
        if xgb_model is not None:
            models['xgboost'] = xgb_model
            results['xgboost'] = xgb_meta
    except Exception as e:
        tprint_warning(f"⚠️ XGBoost training failed: {e}")
    
    # Train ElasticNet
    try:
        elastic_model, elastic_meta = train_layer4_elasticnet(
            X, y, sample_weight, huber_outputs, config, n_trials
        )
        if elastic_model is not None:
            models['elasticnet'] = elastic_model
            results['elasticnet'] = elastic_meta
    except Exception as e:
        tprint_warning(f"⚠️ ElasticNet training failed: {e}")
    
    # Compare models
    tprint_info("📊 Layer 4 Model Comparison:")
    for model_name, metadata in results.items():
        score = metadata.get('best_score', 0)
        features = metadata.get('n_features', 0)
        constraints = metadata.get('constraints_applied', False)
        tprint_info(f"   {model_name.upper()}: Score={score:.4f}, Features={features}, Constraints={constraints}")
    
    # Select best model
    if results:
        best_model = max(results.items(), key=lambda x: x[1].get('best_score', 0))
        tprint_success(f"🏆 Best model: {best_model[0].upper()} (Score: {best_model[1]['best_score']:.4f})")
    
    return {
        'models': models,
        'results': results,
        'best_model': best_model[0] if results else None
    }
