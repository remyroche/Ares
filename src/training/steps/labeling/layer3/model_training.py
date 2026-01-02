"""
Layer 3 Model Training

Handles dual-head model training with De Prado compliance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LogisticRegression
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, log_loss
from scipy.stats import spearmanr
import logging
from numba import njit
from joblib import Parallel, delayed

# Import tprint functions
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    # Fallback print functions
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_success(msg): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")
    def tprint_error(msg): print(f"[ERROR] {msg}")

logger = logging.getLogger(__name__)

def train_dual_head_models(
    X: pd.DataFrame,
    y_alpha: np.ndarray,
    y_prob: np.ndarray,
    w_alpha: np.ndarray,
    w_prob: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    config: Optional[Dict[str, Any]] = None,
    fast_mode: bool = False
) -> Dict[str, Any]:
    """
    Train dual-head models (Alpha generation + Probability calibration).
    """
    tprint_info("🤖 Training Dual-Head Models...")
    tprint_info(f"📊 Training data: {X.shape[0]} samples, {X.shape[1]} features")
    tprint_info(f"🔄 Cross-validation: {len(cv_splits)} folds")
    
    cfg = config or {}
    
    # Train Alpha Head (Regression)
    tprint_info("📈 Training Alpha Head (Regression)...")
    alpha_results = train_alpha_head(
        X, y_alpha, w_alpha, cv_splits, cfg.get('alpha_config', {}), fast_mode
    )
    
    # Train Probability Head (Classification)
    tprint_info("🎯 Training Probability Head (Classification)...")
    prob_results = train_probability_head(
        X, y_prob, w_prob, cv_splits, cfg.get('prob_config', {}), fast_mode
    )
    
    results = {
        'alpha_models': alpha_results['models'],
        'alpha_oof': alpha_results['oof_predictions'],
        'alpha_metrics': alpha_results['metrics'],
        'prob_models': prob_results['models'],
        'prob_oof': prob_results['oof_predictions'],
        'prob_metrics': prob_results['metrics'],
        'calibrated_models': prob_results['calibrated_models']
    }
    
    tprint_success("✅ Dual-Head Training Complete!")
    tprint_success(f"📈 Alpha models: {len(alpha_results['models'])}")
    tprint_success(f"🎯 Probability models: {len(prob_results['models'])}")
    
    return results

def train_alpha_head(
    X: pd.DataFrame,
    y: np.ndarray,
    w: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    config: Dict[str, Any],
    fast_mode: bool = False
) -> Dict[str, Any]:
    """
    Train Alpha generation models with De Prado compliance.
    """
    tprint_info("📈 Alpha Head: Racing Model Candidates...")
    
    # De Prado compliant model candidates - reduced for fast mode
    if fast_mode:
        alpha_candidates = [
            {'name': 'Ridge_MSE', 'type': 'alpha_ridge', 'obj': 'mse'},
            {'name': 'LGBM_MSE', 'type': 'alpha_lgbm', 'obj': 'mse'}
        ]
    else:
        alpha_candidates = [
            {'name': 'Ridge_MSE', 'type': 'alpha_ridge', 'obj': 'mse'},
            {'name': 'LGBM_MSE', 'type': 'alpha_lgbm', 'obj': 'mse'},
            {'name': 'LGBM_Huber', 'type': 'alpha_lgbm', 'obj': 'huber'},
            {'name': 'LGBM_AsymMSE', 'type': 'alpha_lgbm', 'obj': 'asymmetric_mse'}
        ]
    
    tprint_info(f"🏁 Racing {len(alpha_candidates)} candidates:")
    for cand in alpha_candidates:
        tprint_info(f"   - {cand['name']} ({cand['type']}, {cand['obj']})")
    
    # Race candidates
    alpha_scores = {}
    alpha_oof_predictions = {}
    
    for cand in alpha_candidates:
        tprint_info(f"   🏃 Racing {cand['name']}...")
        fold_ics = []
        # Use proper OOF array with NaN for missing predictions
        model_oof = np.full(len(X), np.nan)
        
        # Process folds sequentially to ensure correct index ordering
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            w_tr = w[train_idx]
            
            try:
                if cand['type'] == 'alpha_ridge':
                    # StandardScaler for Ridge (de Prado recommendation)
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    X_tr_scaled = scaler.fit_transform(X_tr)
                    X_val_scaled = scaler.transform(X_val)
                    
                    model = Ridge(alpha=10.0)  # Higher alpha for event-driven training
                    model.fit(X_tr_scaled, y_tr, sample_weight=w_tr)
                    preds = model.predict(X_val_scaled)
                else:
                    # LGBM with De Prado compliant parameters
                    n_estimators = 50 if fast_mode else 200
                    params = {
                        'n_estimators': n_estimators,
                        'max_depth': 3 if fast_mode else 5,
                        'learning_rate': 0.1 if fast_mode else 0.05,
                        'verbose': -1,
                        'n_jobs': 1,
                        'min_child_samples': 20,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8
                    }
                    
                    if cand['obj'] == 'huber':
                        params['objective'] = 'huber'
                        params['alpha'] = 0.9
                    elif cand['obj'] == 'tweedie':
                        params['objective'] = 'tweedie'
                        params['tweedie_variance_power'] = 1.2  # Between 1.1-1.5
                    elif cand['obj'] == 'asymmetric_mse':
                        params['objective'] = _asymmetric_mse_objective
                    else:
                        params['objective'] = 'mse'
                    
                    model = lgb.LGBMRegressor(**params)
                    model.fit(
                        X_tr, y_tr,
                        sample_weight=w_tr,
                        eval_set=[(X_val, y_val)],  # Use val set for early stopping
                        callbacks=[lgb.early_stopping(10, verbose=False)]
                    )
                    preds = model.predict(X_val)
                
                # Store predictions at correct indices
                model_oof[val_idx] = preds
                
                # Evaluate IC (Information Coefficient)
                ic, _ = spearmanr(y_val, preds)
                if np.isfinite(ic):
                    fold_ics.append(ic)
                    
            except Exception as e:
                tprint_warning(f"     Fold {fold_idx+1} failed: {e}")
                # Fill with median prediction for failed folds
                model_oof[val_idx] = 0.0
        
        # Compute ScoreIC (De Prado metric)
        if fold_ics:
            mean_ic = np.mean(fold_ics)
            std_ic = np.std(fold_ics) + 1e-6
            score_ic = 100 * mean_ic + 50 * (mean_ic / std_ic)
        else:
            mean_ic = 0.0
            score_ic = -999.0
        
        alpha_scores[cand['name']] = score_ic
        # Always store OOF predictions (NaN for missing)
        alpha_oof_predictions[cand['name']] = model_oof
        
        ic_stats = f"mean={mean_ic:.4f}" if fold_ics else "N/A"
        tprint_info(f"     ScoreIC: {score_ic:.4f} (IC: {ic_stats})")
    
    # Select top uncorrelated models (De Prado ensemble selection)
    tprint_info("🔄 Selecting Top Uncorrelated Models...")
    selected_alpha_names = select_uncorrelated_models(alpha_scores, alpha_oof_predictions, top_k=2)
    
    tprint_success(f"✅ Selected Alpha Models: {selected_alpha_names}")
    
    # Train final ensemble
    ensemble_models = []
    ensemble_oof = None
    
    for name in selected_alpha_names:
        tprint_info(f"🎯 Training final {name}...")
        cand = next(c for c in alpha_candidates if c['name'] == name)
        
        # HPO parameters (simplified)
        if cand['type'] == 'alpha_ridge':
            best_params = {'alpha': 1.0}
        else:
            best_params = {
                'n_estimators': 300,
                'max_depth': 6,
                'learning_rate': 0.03,
                'verbose': -1,
                'n_jobs': 1,
                'min_child_samples': 25
            }
        
        # OOF predictions
        model_oof = np.full(len(X), np.nan)
        for train_idx, val_idx in cv_splits:
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            w_tr = w[train_idx]
            
            if cand['type'] == 'alpha_ridge':
                model = Ridge(**best_params)
            else:
                model = lgb.LGBMRegressor(**best_params)
            
            model.fit(X_tr, y_tr, sample_weight=w_tr)
            model_oof[val_idx] = model.predict(X_val)
        
        # Final model
        if cand['type'] == 'alpha_ridge':
            final_model = Ridge(**best_params)
        else:
            final_model = lgb.LGBMRegressor(**best_params)
        
        final_model.fit(X, y, sample_weight=w)
        
        ensemble_models.append(final_model)
        
        if ensemble_oof is None:
            ensemble_oof = model_oof
        else:
            ensemble_oof = np.mean([ensemble_oof, model_oof], axis=0)
    
    # Calculate metrics
    final_ic, _ = spearmanr(y, ensemble_oof)
    
    tprint_success(f"✅ Alpha Head Training Complete!")
    tprint_info(f"📈 Final IC: {final_ic:.4f}")
    
    return {
        'models': ensemble_models,
        'oof_predictions': ensemble_oof,
        'metrics': {
            'final_ic': final_ic,
            'selected_models': selected_alpha_names,
            'model_scores': {name: alpha_scores[name] for name in selected_alpha_names}
        }
    }

def train_probability_head(
    X: pd.DataFrame,
    y: np.ndarray,
    w: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    config: Dict[str, Any],
    fast_mode: bool = False
) -> Dict[str, Any]:
    """
    Train Probability calibration models with De Prado compliance.
    """
    tprint_info("🎯 Probability Head: Racing Model Candidates...")
    
    # De Prado compliant model candidates - simplified for fast mode
    if fast_mode:
        prob_candidates = [
            {'name': 'LGBM_LogLoss', 'type': 'classifier', 'obj': 'binary_logloss'},
            {'name': 'Logistic_Reg', 'type': 'logistic_regression', 'obj': 'binary'}
        ]
    else:
        prob_candidates = [
            {'name': 'LGBM_LogLoss', 'type': 'classifier', 'obj': 'binary_logloss'},
            {'name': 'LGBM_Focal', 'type': 'classifier', 'obj': 'focal'},
            {'name': 'Logistic_Reg', 'type': 'logistic_regression', 'obj': 'binary'}
        ]
    
    tprint_info(f"🏁 Racing {len(prob_candidates)} candidates:")
    for cand in prob_candidates:
        tprint_info(f"   - {cand['name']} ({cand['type']}, {cand['obj']})")
    
    # Race candidates
    prob_scores = {}
    prob_oof_predictions = {}
    
    for cand in prob_candidates:
        tprint_info(f"   🏃 Racing {cand['name']}...")
        fold_scores = []
        # Use proper OOF array with NaN for missing predictions
        model_oof = np.full(len(X), np.nan)
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            w_tr = w[train_idx]
            
            # Skip fold if single-class
            if len(np.unique(y_tr)) < 2:
                tprint_warning(f"     Fold {fold_idx+1}: Single class in train, skipping")
                model_oof[val_idx] = 0.5
                continue
            if len(np.unique(y_val)) < 2:
                tprint_warning(f"     Fold {fold_idx+1}: Single class in val, skipping")
                model_oof[val_idx] = 0.5
                continue
            
            try:
                if cand['type'] == 'logistic_regression':
                    model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
                    model.fit(X_tr, y_tr, sample_weight=w_tr)
                else:
                    # LGBM with De Prado compliant parameters
                    n_estimators = 50 if fast_mode else 200
                    params = {
                        'n_estimators': n_estimators,
                        'max_depth': 3 if fast_mode else 5,
                        'learning_rate': 0.1 if fast_mode else 0.05,
                        'verbose': -1,
                        'n_jobs': 1,
                        'min_child_samples': 20,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8
                    }
                    
                    if cand['obj'] == 'focal':
                        params['objective'] = _focal_loss_objective
                    else:
                        params['objective'] = 'binary'
                    
                    model = lgb.LGBMClassifier(**params)
                    model.fit(
                        X_tr, y_tr,
                        sample_weight=w_tr,
                        eval_set=[(X_val, y_val)],
                        callbacks=[lgb.early_stopping(10, verbose=False)]
                    )
                
                # Handle predict_proba output shape
                prob_output = model.predict_proba(X_val)
                if prob_output.ndim == 2 and prob_output.shape[1] >= 2:
                    probs = prob_output[:, 1]
                elif prob_output.ndim == 2 and prob_output.shape[1] == 1:
                    probs = prob_output[:, 0]
                else:
                    probs = prob_output
                
                # For custom objectives (focal), apply sigmoid and clip to [0, 1]
                if cand['obj'] == 'focal':
                    # Custom objectives return raw logits, convert to probabilities
                    probs = 1 / (1 + np.exp(-probs))  # Sigmoid
                probs = np.clip(probs, 0.0, 1.0)  # Ensure valid probability range
                
                # Store predictions at correct indices
                model_oof[val_idx] = probs
                
                # ScoreL3 (De Prado metric)
                auc = roc_auc_score(y_val, probs)
                # Standard log lossll = log_loss(y_val, probs, labels=[0, 1])
                
                # Enhanced weighted log loss with absolute return weighting
                try:

                    if len(returns_val) == len(probs):

                        weighted_ll = enhanced_weighted_logloss(

                            y_val, probs, returns_val,

                            sample_weights=w_val,

                            alpha=0.5, beta=0.3

                        )

                        tprint_info(f"   📊 Weighted LogLoss: {weighted_ll:.4f} (vs {ll:.4f} standard)")

                    else:

                        tprint_warning("   ⚠️ Mismatched lengths, skipping weighted loss")

                except Exception as e:

                    tprint_warning(f"   ⚠️ Weighted loss calculation failed: {e}")
                score = 100 * (auc - 0.5) + 50 * (0.693 - ll)
                fold_scores.append(score)
                
            except Exception as e:
                tprint_warning(f"     Fold {fold_idx+1} failed: {e}")
                model_oof[val_idx] = 0.5
        
        prob_scores[cand['name']] = np.mean(fold_scores) if fold_scores else -999.0
        # Always store OOF predictions
        prob_oof_predictions[cand['name']] = model_oof
        
        score_stats = f"mean={np.mean(fold_scores):.4f}" if fold_scores else "N/A"
        tprint_info(f"     ScoreL3: {prob_scores[cand['name']]:.4f} ({score_stats})")
    
    # Select top uncorrelated models
    tprint_info("🔄 Selecting Top Uncorrelated Models...")
    selected_prob_names = select_uncorrelated_models(prob_scores, prob_oof_predictions, top_k=2)
    
    tprint_success(f"✅ Selected Probability Models: {selected_prob_names}")
    
    # Train final ensemble with calibration
    ensemble_models = []
    calibrated_models = []
    ensemble_oof = None
    
    for name in selected_prob_names:
        tprint_info(f"🎯 Training final calibrated {name}...")
        cand = next(c for c in prob_candidates if c['name'] == name)
        
        # HPO parameters (simplified)
        if cand['type'] == 'logistic_regression':
            best_params = {'C': 1.0}
        else:
            best_params = {
                'n_estimators': 300,
                'max_depth': 6,
                'learning_rate': 0.03,
                'verbose': -1,
                'n_jobs': 1,
                'min_child_samples': 25
            }
        
        # OOF with calibration
        model_oof = np.full(len(X), np.nan)
        for train_idx, val_idx in cv_splits:
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            w_tr = w[train_idx]
            
            # Skip fold if single-class in train or val
            if len(np.unique(y_tr)) < 2:
                tprint_warning(f"⚠️ Skipping fold: Single class in training set")
                model_oof[val_idx] = 0.5  # Neutral prediction
                continue
            if len(np.unique(y_val)) < 2:
                tprint_warning(f"⚠️ Skipping fold: Single class in validation set")
                model_oof[val_idx] = 0.5  # Neutral prediction
                continue
            
            try:
                if cand['type'] == 'logistic_regression':
                    base_model = LogisticRegression(**best_params, solver='lbfgs', max_iter=1000)
                else:
                    base_model = lgb.LGBMClassifier(**best_params)
                
                # Apply calibration
                cal_model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
                cal_model.fit(X_tr, y_tr, sample_weight=w_tr)
                # predict_proba might return 1D array if only 1 class in training
                prob = cal_model.predict_proba(X_val)
                if prob.ndim == 2 and prob.shape[1] >= 2:
                    model_oof[val_idx] = prob[:, 1]
                elif prob.ndim == 2 and prob.shape[1] == 1:
                    model_oof[val_idx] = prob[:, 0]  # Single class, use that probability
                else:
                    model_oof[val_idx] = prob
            except Exception as e:
                tprint_warning(f"⚠️ Fold training failed: {e}")
                model_oof[val_idx] = 0.5  # Neutral prediction
        
        # Final calibrated model
        if cand['type'] == 'logistic_regression':
            base_model = LogisticRegression(**best_params, solver='lbfgs', max_iter=1000)
        else:
            base_model = lgb.LGBMClassifier(**best_params)
        
        final_calibrated = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
        final_calibrated.fit(X, y, sample_weight=w)
        
        ensemble_models.append(base_model)
        calibrated_models.append(final_calibrated)
        
        if ensemble_oof is None:
            ensemble_oof = model_oof
        else:
            ensemble_oof = np.mean([ensemble_oof, model_oof], axis=0)
    
    # Calculate metrics (handle single-class case)
    try:
        final_auc = roc_auc_score(y, ensemble_oof)
    except ValueError:
        tprint_warning("⚠️ Could not compute AUC (single class in y_true)")
        final_auc = 0.5
    try:
        final_ll = log_loss(y, ensemble_oof, labels=[0, 1])
    except ValueError:
        tprint_warning("⚠️ Could not compute log_loss (single class in y_true)")
        final_ll = 1.0
    
    tprint_success(f"✅ Probability Head Training Complete!")
    tprint_info(f"🎯 Final AUC: {final_auc:.4f}")
    tprint_info(f"🎯 Final LogLoss: {final_ll:.4f}")
    
    return {
        'models': ensemble_models,
        'calibrated_models': calibrated_models,
        'oof_predictions': ensemble_oof,
        'metrics': {
            'final_auc': final_auc,
            'final_logloss': final_ll,
            'selected_models': selected_prob_names,
            'model_scores': {name: prob_scores[name] for name in selected_prob_names}
        }
    }

def select_uncorrelated_models(
    scores: Dict[str, float],
    oof_predictions: Dict[str, np.ndarray],
    top_k: int = 2,
    correlation_threshold: float = 0.9
) -> List[str]:
    """
    Select top uncorrelated models based on scores and correlations.
    """
    tprint_info(f"🔄 Selecting {top_k} uncorrelated models (threshold={correlation_threshold})")
    
    if len(oof_predictions) < top_k:
        tprint_info(f"📊 Only {len(oof_predictions)} models available, selecting all")
        return list(scores.keys())[:top_k]
    
    # Sort by score
    sorted_models = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    tprint_info(f"📊 Model ranking by score: {sorted_models}")
    
    selected = []
    selected_predictions = []
    
    for model_name in sorted_models:
        if len(selected) >= top_k:
            break
        
        if model_name not in oof_predictions:
            continue
        
        current_pred = oof_predictions[model_name]
        
        # Check correlation with already selected models
        is_correlated = False
        for selected_pred in selected_predictions:
            corr = np.corrcoef(current_pred, selected_pred)[0, 1]
            if corr > correlation_threshold:
                is_correlated = True
                tprint_info(f"   ❌ {model_name} correlated with selected model (corr={corr:.3f})")
                break
        
        if not is_correlated:
            selected.append(model_name)
            selected_predictions.append(current_pred)
            tprint_info(f"   ✅ Selected {model_name} (score={scores[model_name]:.4f})")
    
    return selected

# Custom objective functions for De Prado compliance
@njit
def _asymmetric_mse_objective(y_true, y_pred):
    """Asymmetric MSE objective for alpha models. JIT compiled for performance."""
    residual = y_true - y_pred
    grad = -2 * residual * (residual > 0).astype(np.float64)  # Penalize negative residuals more
    hess = 2 * np.ones_like(residual)
    return grad, hess

@njit
def _focal_loss_objective(y_true, y_pred):
    """Focal loss objective for probability models. JIT compiled for performance."""
    gamma = 2.0  # Focusing parameter
    p = 1 / (1 + np.exp(-y_pred))
    
    grad = -gamma * (1 - p) ** gamma * np.log(p) * (y_true - p) + (p - y_true)
    hess = p * (1 - p)
    
    return grad, hess
