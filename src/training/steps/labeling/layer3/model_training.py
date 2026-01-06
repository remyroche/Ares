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

    Supports Regime-Conditional Mixture-of-Experts.
    """
    tprint_info("🤖 Training Dual-Head Models (Mixture-of-Experts)...")
    tprint_info(f"📊 Training data: {X.shape[0]} samples, {X.shape[1]} features")
    tprint_info(f"🔄 Cross-validation: {len(cv_splits)} folds")
    
    cfg = config or {}
    
    # Check for regime labels
    regime_col = 'regime_label'
    regimes = ['Global']

    if regime_col in X.columns:
        # Use existing regime labels
        regimes = X[regime_col].unique().tolist()
        regimes = [r for r in regimes if isinstance(r, str) and r != "Unknown"]
        tprint_info(f"   🏷️  Regime-Conditional Mode: Training experts for {regimes}")
    else:
        tprint_info("   🌍 Global Mode: Training single expert")

    # Initialize results containers
    alpha_models_dict = {}
    prob_models_dict = {}

    # Initialize OOF arrays (NaN filled)
    alpha_oof_combined = np.full(len(X), np.nan)
    prob_oof_combined = np.full(len(X), np.nan)

    # Soft Gating containers (Accumulate weighted predictions)
    # We will accumulate weighted predictions and normalize by sum of weights later if needed,
    # but regime probabilities sum to 1.0, so direct weighted sum is fine.
    # However, for OOF, we need to generate predictions for ALL samples using EACH expert.
    
    alpha_soft_oof = np.zeros(len(X))
    prob_soft_oof = np.zeros(len(X))
    
    # Check if we have regime probabilities
    has_soft_probs = any(c.startswith('prob_') and c != 'prob_oof' for c in X.columns)

    # Train Expert per Regime
    for regime in regimes:
        tprint_info(f"   👉 Training Expert for: {regime}")

        # Determine mask
        if regime == 'Global':
            mask = np.ones(len(X), dtype=bool)
        else:
            mask = (X[regime_col] == regime).values

        if mask.sum() < 50:
            tprint_warning(f"      ⚠️ Skipping {regime}: Insufficient samples ({mask.sum()})")
            continue

        # Subset data for TRAINING
        indices = np.where(mask)[0]

        X_regime = X.iloc[indices]
        y_alpha_regime = y_alpha[indices]
        y_prob_regime = y_prob[indices]
        w_alpha_regime = w_alpha[indices]
        w_prob_regime = w_prob[indices]

        # Create regime-specific CV splits
        regime_cv_splits = []
        for train_idx, val_idx in cv_splits:
            # Intersection with regime mask
            train_mask_local = np.isin(train_idx, indices)
            val_mask_local = np.isin(val_idx, indices)

            if not np.any(val_mask_local): continue

            # Efficient mapping using searchsorted
            regime_train_global = train_idx[train_mask_local]
            regime_val_global = val_idx[val_mask_local]

            train_local = np.searchsorted(indices, regime_train_global)
            val_local = np.searchsorted(indices, regime_val_global)

            if len(train_local) > 10 and len(val_local) > 0:
                regime_cv_splits.append((train_local, val_local))

        if not regime_cv_splits:
            tprint_warning(f"      ⚠️ No valid CV splits for {regime}")
            continue

        # Prepare numeric features for this regime
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        # Exclude prob_ columns from expert training inputs to avoid leakage/noise
        train_cols = [c for c in numeric_cols if not c.startswith('prob_')]
        X_regime_numeric = X_regime[train_cols]

        # Train Alpha Head
        tprint_info("      📈 Training Alpha Expert...")
        alpha_res = train_alpha_head(
            X_regime_numeric, y_alpha_regime, w_alpha_regime,
            regime_cv_splits, cfg.get('alpha_config', {}), fast_mode
        )
        alpha_models_dict[regime] = alpha_res['models']

        # Train Probability Head
        tprint_info("      🎯 Training Probability Expert...")
        prob_res = train_probability_head(
            X_regime_numeric, y_prob_regime, w_prob_regime,
            regime_cv_splits, cfg.get('prob_config', {}), fast_mode
        )
        prob_models_dict[regime] = prob_res['calibrated_models']

        # --- Soft Gating OOF Generation ---
        # We need OOF predictions for the FULL dataset using this expert.
        # Logic: Iterate global CV splits. Train expert on (Train \cap Regime). Predict on (Val_Full).
        # Optimization: We already have trained models from `train_alpha_head` which corresponds to folds.
        # But `train_alpha_head` returns models trained on X_regime.
        # If the number of regime folds != global folds, alignment is tricky.
        # Fallback to Hard Gating for filling holes?
        # Ideally, we loop over GLOBAL splits.

        # To support Soft Gating properly, we need to predict on the FULL validation fold.
        # `train_alpha_head` logic doesn't support this out of the box (it predicts on X_val passed to it).
        # We will use the trained models (from `alpha_res['models']`) to predict on the rest of the fold?
        # But `alpha_res['models']` is a list of ensemble models (one per run? No, it returns final ensemble trained on FULL X_regime).
        # Wait, `train_alpha_head` returns `ensemble_models` which is a list of final models trained on ALL X_regime.
        # It does NOT return the fold-specific models.
        # OOF predictions in `alpha_res['oof_predictions']` are only for X_regime.

        # CRITICAL: We cannot do true Soft Gating OOF without retraining or accessing fold models.
        # Compromise: Hard Gating for Training OOF (since we know the true regime), Soft Gating for Inference.
        # During training, if we are in 'Quiet' regime, the 'Quiet' expert is the only one that matters locally.
        # 'Trending' expert trained on 'Trending' data will likely be garbage on 'Quiet' data.
        # So hard gating for OOF evaluation is actually statistically sounder for 'specialist' validation.
        # HOWEVER, the user asked for Soft Gating.
        # "OOF prediction generation ... uses Hard Gating ... misses requested smooth transition".
        # If we have `prob_Quiet` = 0.9, `prob_Trending` = 0.1 for a sample.
        # We want `0.9 * Pred_Quiet + 0.1 * Pred_Trending`.
        # `Pred_Quiet` is available (it's the expert for this sample).
        # `Pred_Trending` is NOT available (because that expert wasn't trained on this sample).
        # BUT `Pred_Trending` expert exists (trained on OTHER samples).
        # We can use the Trending Expert to predict on this Quiet sample.

        # Implementation:
        # We use the final trained expert (alpha_res['models'][0]) to predict on ALL X.
        # This is leaky (it saw the sample during training if the sample was in Regime).
        # But for Soft Gating, we need predictions where the sample was NOT in Regime.
        # If sample was in Regime, we use OOF prediction.
        # If sample was NOT in Regime, we use Full Model prediction (it wasn't in training set for that model!).

        # 1. Get OOF predictions for in-regime samples
        oof_local_alpha = alpha_res['oof_predictions'] # aligned to indices
        oof_local_prob = prob_res['oof_predictions']

        # 2. Get Full Model predictions for out-of-regime samples
        # Prepare full X (numeric only)
        X_full_numeric = X[train_cols]

        # Alpha Expert Prediction (Full)
        # Use simple average of ensemble models
        alpha_preds_full = np.mean([m.predict(X_full_numeric) for m in alpha_res['models']], axis=0)

        # Probability Expert Prediction (Full)
        prob_preds_full = np.mean([m.predict_proba(X_full_numeric)[:, 1] for m in prob_res['calibrated_models']], axis=0)

        # 3. Combine: Use OOF where available (in-regime), Full Model elsewhere (out-regime)
        # Create a full-size array for this expert
        expert_alpha_full = alpha_preds_full.copy()
        expert_prob_full = prob_preds_full.copy()

        # Overwrite in-regime samples with OOF predictions (unbiased)
        # Map local OOF to global indices
        valid_local_alpha = ~np.isnan(oof_local_alpha)
        global_idx_alpha = indices[valid_local_alpha]
        expert_alpha_full[global_idx_alpha] = oof_local_alpha[valid_local_alpha]

        valid_local_prob = ~np.isnan(oof_local_prob)
        global_idx_prob = indices[valid_local_prob]
        expert_prob_full[global_idx_prob] = oof_local_prob[valid_local_prob]

        # 4. Weight by Regime Probability
        if has_soft_probs:
            # Find probability column for this regime
            prob_col = f"prob_{regime}"
            if prob_col in X.columns:
                weights = X[prob_col].values
                # Accumulate weighted predictions
                alpha_soft_oof += expert_alpha_full * weights
                prob_soft_oof += expert_prob_full * weights
            else:
                # Fallback: Hard mask if prob col missing
                tprint_warning(f"      ⚠️ Missing {prob_col}, using hard mask for weighting")
                alpha_soft_oof[indices] += oof_local_alpha
                prob_soft_oof[indices] += oof_local_prob
        else:
            # Fallback to Hard Gating if no soft probs
            alpha_soft_oof[indices] = oof_local_alpha
            prob_soft_oof[indices] = oof_local_prob

    # Finalize OOF arrays
    if not has_soft_probs:
        # If hard gating, we populated indices directly.
        alpha_oof_combined = alpha_soft_oof
        prob_oof_combined = prob_soft_oof
    else:
        # Soft gating accumulation complete
        alpha_oof_combined = alpha_soft_oof
        prob_oof_combined = prob_soft_oof

    # Store metrics (approximate global metrics from combined OOF)
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr

    valid_alpha = ~np.isnan(alpha_oof_combined)
    final_ic = 0.0
    if valid_alpha.sum() > 0:
        final_ic, _ = spearmanr(y_alpha[valid_alpha], alpha_oof_combined[valid_alpha])

    valid_prob = ~np.isnan(prob_oof_combined)
    final_auc = 0.5
    if valid_prob.sum() > 0:
        try:
            final_auc = roc_auc_score(y_prob[valid_prob], prob_oof_combined[valid_prob])
        except: pass

    # If gaps in OOF (e.g. Regime 'Chaos' had no training data), fill with mean/0.5
    # or interpolate? 0.5 is safer.
    prob_oof_combined[np.isnan(prob_oof_combined)] = 0.5
    alpha_oof_combined[np.isnan(alpha_oof_combined)] = 0.0

    results = {
        'alpha_models': alpha_models_dict, # Dict[Regime, List[Models]]
        'alpha_oof': alpha_oof_combined,
        'alpha_metrics': {'final_ic': final_ic},
        'prob_models': prob_models_dict, # Dict[Regime, List[Models]]
        'prob_oof': prob_oof_combined,
        'prob_metrics': {'final_auc': final_auc},
        'calibrated_models': prob_models_dict # Redundant but for compatibility
    }
    
    tprint_success("✅ Dual-Head Training Complete (MoE)!")
    tprint_success(f"   📈 Experts trained: {list(alpha_models_dict.keys())}")
    
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
