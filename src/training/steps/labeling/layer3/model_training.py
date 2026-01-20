"""
Layer 3 Model Training - Multi-Horizon Implementation
(ExtraTrees + LGBM + XGBoost with Monotonic & Interaction Constraints)

Replaces ORF with specific constrained tree models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
import lightgbm as lgb
import xgboost as xgb
import optuna
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, mean_squared_error, log_loss
from sklearn.linear_model import HuberRegressor, Ridge
from scipy.special import expit

# Import CatBoost if available
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from src.utils.huber_regressor_for_trees import prepare_huber_teacher_outputs
from src.training.steps.labeling.layer3.feature_engineering import downcast_float
from src.training.steps.labeling.focal_loss_utils import RobustFocalLoss, XGBFocalLoss
from src.training.steps.labeling.probability_calibration import ProbabilityCalibrator
from src.training.steps.labeling.irm_regime_pipeline import IRMLinearClassifier, IRMLinearRegressor

# Import tprint functions
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_success(msg): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")
    def tprint_error(msg): print(f"[ERROR] {msg}")

logger = logging.getLogger(__name__)

def apply_huber_rotation_logic(X: pd.DataFrame, huber_coeffs: pd.Series, top_n: int = 3) -> pd.DataFrame:
    """
    Applies Pseudo-Oblique rotation by adding Sum and Difference of top high-signal feature pairs.
    Expected X to be Scaled.
    Optimized to use vectorized operations.
    """
    # Ensure working with float32 if not already
    X_rotated = downcast_float(X.copy())

    # 1. Identify top feature pairs based on Huber Absolute Coefficients
    valid_coeffs = huber_coeffs[huber_coeffs.index.isin(X.columns)]
    if valid_coeffs.empty:
        return X_rotated

    # Select features and ensure indices align
    top_features = valid_coeffs.abs().sort_values(ascending=False).index[:top_n*2]

    # 2. Create Sum and Difference for high-signal pairs
    # Vectorized approach: select columns, perform operation
    new_cols = {}

    for i in range(0, len(top_features) - 1, 2):
        f1, f2 = top_features[i], top_features[i+1]

        # Using numpy arrays for speed
        val1 = X_rotated[f1].values
        val2 = X_rotated[f2].values

        new_cols[f"{f1}_{f2}_sum"] = val1 + val2
        new_cols[f"{f1}_{f2}_diff"] = val1 - val2

    if new_cols:
        new_df = pd.DataFrame(new_cols, index=X_rotated.index)
        X_rotated = pd.concat([X_rotated, new_df], axis=1)

    return X_rotated

def train_lgbm_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    model_name: str,
    task_type: str,
    huber_output: Dict[str, Any],
    sample_weight: Optional[np.ndarray] = None,
    config: Optional[Dict[str, Any]] = None,
    fast_mode: bool = False
) -> Dict[str, Any]:
    """
    Trains LightGBM with Pseudo-Oblique Linear Trees, Monotonic & Interaction Constraints, and Warm Start.
    Uses 'huber' objective for regression tasks (Robustness) and 'RobustFocalLoss' (mix=1.0) for classification (Calibration).
    """
    tprint_info(f"   🍁 Training LGBM ({task_type}): {model_name}...")
    
    # Feature Selection & Scaling
    selected_features = huber_output['selected_features']
    # Downcast and copy
    X_t = X_train[selected_features].copy().astype(np.float32)
    
    scaler = StandardScaler()
    X_scaled_np = scaler.fit_transform(X_t).astype(np.float32)
    X_scaled = pd.DataFrame(X_scaled_np, columns=X_t.columns, index=X_t.index)

    # Prepare constraints
    # Huber output returns dict {feat: sign}, convert to list matching columns
    mono_dict = huber_output['monotonic_constraints']
    monotone_constraints = [mono_dict.get(c, 0) for c in X_t.columns]

    interaction_constraints = huber_output['interaction_constraints']

    # Warm start
    init_score = huber_output['warm_start']['train']

    # Configure objective and params aligned with Layer 2
    if task_type == 'classification':
        # Use RobustFocalLoss with mix=1.0 (LogLoss) for optimal calibration + Smoothing
        objective = RobustFocalLoss(gamma_pos=1.0, gamma_neg=2.5, mix=1.0, label_smoothing=0.02, verbose=False)
        metric = 'binary_logloss' # Track calibration
        # Layer 2 classification params
        params = {
            'boosting_type': 'dart', # Use DART for classification
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 6,
            'min_data_in_leaf': 20,
            'feature_fraction': 0.6,
            'lambda_l1': 0.5,
            'lambda_l2': 1.0,
            'bagging_fraction': 1.0, # Disable bagging for DART
            'bagging_freq': 0
        }
    else:
        # Regression: Huber for Robust Statistics (Fat Tails)
        objective = 'huber'
        metric = 'l2' # Monitor MSE/IC
        # Robust regression params
        params = {
            'boosting_type': 'dart', # Use DART for regression too
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 6,
            'min_data_in_leaf': 100, # More robust for regression
            'feature_fraction': 0.6,
            'lambda_l1': 1.0,
            'lambda_l2': 2.0, # Higher reg for regression
            'alpha': 1.35 # Huber delta (tuning param)
        }

    # Common params
    params.update({
        'verbosity': -1,
        'linear_tree': True, # Keep linear tree
        'n_estimators': 100 if fast_mode else 500, # Aligned with Layer 2 loop limits (usually 200-500)
        'monotone_constraints': monotone_constraints,
        'interaction_constraints': interaction_constraints if interaction_constraints else "",
        'early_stopping_round': 30,
        'seed': 42
    })

    if config and 'lgbm_params' in config:
        params.update(config['lgbm_params'])

    # Data Splitting for Validation
    split_idx = int(len(X_scaled) * 0.9)
    X_tr, X_val = X_scaled.iloc[:split_idx], X_scaled.iloc[split_idx:]
    y_tr, y_val = y_train[:split_idx], y_train[split_idx:]
    w_tr = sample_weight[:split_idx] if sample_weight is not None else None
    w_val = sample_weight[split_idx:] if sample_weight is not None else None
    init_tr = init_score[:split_idx]
    init_val = init_score[split_idx:]

    dtrain_split = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, init_score=init_tr)
    dval_split = lgb.Dataset(X_val, label=y_val, weight=w_val, init_score=init_val, reference=dtrain_split)

    callbacks = [
        lgb.early_stopping(stopping_rounds=params.pop('early_stopping_round')),
        lgb.log_evaluation(period=0)
    ]

    # Handle custom objective
    if not isinstance(objective, str):
        # Pass callable objective via params if fobj arg is not supported
        params['objective'] = objective
        params['metric'] = metric
        model = lgb.train(
            params,
            dtrain_split,
            valid_sets=[dval_split],
            callbacks=callbacks
        )
    else:
        params['objective'] = objective
        params['metric'] = metric
        model = lgb.train(
            params,
            dtrain_split,
            valid_sets=[dval_split],
            callbacks=callbacks
        )
    
    # Prediction (Raw Score = Margin)
    raw_margin = model.predict(X_scaled, raw_score=True)
    final_margin = raw_margin + init_score

    if task_type == 'classification':
        final_preds = expit(final_margin)

        # Post-hoc Calibration (De Prado Compliant)
        try:
            # We need OOF predictions to fit calibration, but simple split here:
            # Fit calibration on validation set predictions (unbiased)
            val_margin = model.predict(X_val, raw_score=True)
            val_preds = expit(val_margin + init_val)

            calibrator = ProbabilityCalibrator(method='isotonic', min_samples=50, plot_calibration=False, save_plots=False)
            cal_res = calibrator.fit(y_val, val_preds, sample_weights=w_val)

            # Apply to final predictions
            final_preds_cal = calibrator.predict(final_preds)
            tprint_info(f"   ⚖️  LGBM Calibrated: Brier improvement {cal_res['metrics'].get('brier_improvement', 0):.4f}")
            final_preds = final_preds_cal
        except Exception as e:
            tprint_warning(f"   ⚠️ Calibration failed: {e}")

    else:
        final_preds = final_margin

    return {
        'model': model,
        'cate': final_preds,
        'se': np.zeros(len(final_preds)), # Placeholder
        'scaler': scaler
    }

def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    model_name: str,
    task_type: str,
    huber_output: Dict[str, Any],
    sample_weight: Optional[np.ndarray] = None,
    config: Optional[Dict[str, Any]] = None,
    fast_mode: bool = False
) -> Dict[str, Any]:
    """
    Trains XGBoost with Feature-Rotation, Monotonic & Interaction Constraints, and Warm Start.
    Aligned with Layer 2 parameters.
    """
    tprint_info(f"   🚀 Training XGBoost ({task_type}): {model_name}...")

    selected_features = huber_output['selected_features']
    # Downcast
    X_t = X_train[selected_features].copy().astype(np.float32)

    # Scaling BEFORE Rotation
    scaler = StandardScaler()
    X_scaled_np = scaler.fit_transform(X_t).astype(np.float32)
    X_scaled = pd.DataFrame(X_scaled_np, columns=X_t.columns, index=X_t.index)

    # 1. Apply Huber Rotation Logic
    # Use the median model (first in list)
    huber_model = huber_output['huber_models'][0]

    # Map coefficients to original columns (X_train) before selection
    # X_t is already subsetted, so we need to map to X_train first
    all_coeffs = pd.Series(huber_model.coef_, index=X_train.columns)

    # Filter coefficients for selected features only
    coeffs_series = all_coeffs[selected_features]

    X_rotated = apply_huber_rotation_logic(X_scaled, coeffs_series)
    X_rotated = X_rotated.astype(np.float32)

    # 2. Constraints (Only on original features)
    mono_dict = huber_output['monotonic_constraints']
    final_constraints = []
    for col in X_rotated.columns:
        # Constraints only apply to original columns, new rotated cols get 0
        val = mono_dict.get(col, 0)
        # Ensure standard python int for XGBoost compatibility
        final_constraints.append(int(val))

    # 3. Warm Start
    base_margin = huber_output['warm_start']['train']

    # 4. Parameters Aligned with Layer 2
    params = {
        'n_estimators': 100 if fast_mode else 500,
        'learning_rate': 0.05,
        'max_depth': 4 if task_type == 'classification' else 5,
        'min_child_weight': 10,
        'gamma': 0.5, # High regularization
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'colsample_bynode': 0.4,
        'reg_alpha': 0.3, # L1
        'reg_lambda': 30, # Strong L2 (Reduced from 50)
        'num_parallel_tree': 7,
        'monotone_constraints': tuple(final_constraints),
        'interaction_constraints': huber_output['interaction_constraints'] if huber_output['interaction_constraints'] else None,
        'n_jobs': -1,
        'verbosity': 0,
        'early_stopping_rounds': 30
    }

    # Internal split for early stopping
    split_idx = int(len(X_rotated) * 0.9)
    X_tr, X_val = X_rotated.iloc[:split_idx], X_rotated.iloc[split_idx:]
    y_tr, y_val = y_train[:split_idx], y_train[split_idx:]
    w_tr = sample_weight[:split_idx] if sample_weight is not None else None
    w_val = sample_weight[split_idx:] if sample_weight is not None else None
    bm_tr = base_margin[:split_idx]
    bm_val = base_margin[split_idx:]

    model = None

    if task_type == 'classification':
        # Use Focal Loss for classification
        params.pop('eval_metric', None) # Use default or set explicitly
        params['disable_default_eval_metric'] = 1

        # Instantiate model with custom objective
        # Note: XGBClassifier with custom objective needs obj argument in init or fit
        # We pass it in init to keep sklearn interface consistency if possible, but
        # for custom obj in XGB, it's safer to use 'objective' param as function

        focal_loss = XGBFocalLoss(gamma_pos=1.0, gamma_neg=2.5) # Similar to LGBM RobustFocal

        model = xgb.XGBClassifier(objective=focal_loss, eval_metric='logloss', **params)
    else:
        # Robust regression (Pseudo-Huber)
        params['objective'] = 'reg:pseudohubererror'
        params['eval_metric'] = 'rmse'
        model = xgb.XGBRegressor(**params)

    if config and 'xgb_params' in config:
        params.update(config['xgb_params'])

    # Safe Fit with retry on constraints failure
    try:
        model.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            base_margin=bm_tr,
            eval_set=[(X_val, y_val)],
            sample_weight_eval_set=[w_val],
            base_margin_eval_set=[bm_val],
            verbose=False
        )
    except (ValueError, xgb.core.XGBoostError):
        tprint_warning("   ⚠️ XGBoost constraints failed, retrying without interaction constraints.")
        params['interaction_constraints'] = None
        if task_type == 'classification':
             focal_loss = XGBFocalLoss(gamma_pos=1.0, gamma_neg=2.5)
             model = xgb.XGBClassifier(objective=focal_loss, eval_metric='logloss', **params)
        else:
             model = xgb.XGBRegressor(**params)

        model.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            base_margin=bm_tr,
            eval_set=[(X_val, y_val)],
            sample_weight_eval_set=[w_val],
            base_margin_eval_set=[bm_val],
            verbose=False
        )

    # Prediction
    if task_type == 'regression':
        preds = model.predict(X_rotated, base_margin=base_margin)
    else:
        try:
             # Predict proba with custom objective might return margins or transformed
             # XGBClassifier predict_proba usually applies sigmoid if objective is binary
             # But with custom obj, we must check.
             # Usually we rely on model.predict(output_margin=True) then sigmoid
             # But sklearn wrapper might handle it?
             # Let's use margin + sigmoid manually for safety with custom objective
             margin_preds = model.predict(X_rotated, output_margin=True, base_margin=base_margin)
             preds = expit(margin_preds)
        except TypeError:
            dmat = xgb.DMatrix(X_rotated, base_margin=base_margin)
            margin_preds = model.get_booster().predict(dmat, output_margin=True)
            preds = expit(margin_preds)

        # Post-hoc Calibration
        try:
            # Fit calibration on validation set
            val_margin = model.predict(X_val, output_margin=True, base_margin=bm_val)
            val_preds = expit(val_margin)

            calibrator = ProbabilityCalibrator(method='isotonic', min_samples=50, plot_calibration=False, save_plots=False)
            cal_res = calibrator.fit(y_val, val_preds, sample_weights=w_val)

            preds_cal = calibrator.predict(preds)
            tprint_info(f"   ⚖️  XGB Calibrated: Brier improvement {cal_res['metrics'].get('brier_improvement', 0):.4f}")
            preds = preds_cal
        except Exception as e:
            tprint_warning(f"   ⚠️ Calibration failed: {e}")

    return {
        'model': model,
        'cate': preds,
        'se': np.zeros(len(preds)),
        'scaler': scaler
    }

def train_extratrees_constrained(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    model_name: str,
    task_type: str,
    huber_output: Dict[str, Any],
    sample_weight: Optional[np.ndarray] = None,
    config: Optional[Dict[str, Any]] = None,
    fast_mode: bool = False
) -> Dict[str, Any]:
    """
    Trains ExtraTrees model with Monotonic Constraints and IRM-style Robustness.
    """
    tprint_info(f"   🌳 Training ExtraTrees ({task_type}): {model_name} with Constraints...")

    cfg = config or {}
    et_params = cfg.get('et_params', {
        'n_estimators': 100 if fast_mode else 500,
        'max_depth': 6 if fast_mode else 12, # Constrained depth like Layer 2
        'min_samples_leaf': 20,
        'max_features': 0.8, # Feature subsampling
        'bootstrap': True,
        'n_jobs': -1,
        'random_state': 42
    })

    selected_features = huber_output['selected_features']
    X_t = X_train[selected_features].copy().astype(np.float32)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_t).astype(np.float32)

    # Convert dict to array matching columns
    mono_dict = huber_output['monotonic_constraints']
    constraints = np.array([mono_dict.get(c, 0) for c in X_t.columns])

    # Attempt to use monotonic_cst (sklearn 1.4+)
    try:
        if task_type == 'regression':
            # Use MAE for robustness if possible, but standard is MSE
            et_model = ExtraTreesRegressor(monotonic_cst=constraints, criterion='squared_error', **et_params)
            et_model.fit(X_scaled, y_train, sample_weight=sample_weight)
            preds = et_model.predict(X_scaled)
        else:
            et_model = ExtraTreesClassifier(monotonic_cst=constraints, criterion='log_loss', **et_params)
            y_int = (y_train > 0).astype(int)
            et_model.fit(X_scaled, y_int, sample_weight=sample_weight)
            preds = et_model.predict_proba(X_scaled)[:, 1]

    except TypeError:
        # Fallback if version mismatch
        tprint_warning(f"   ⚠️ ExtraTrees constraint fallback.")
        if task_type == 'regression':
            et_model = ExtraTreesRegressor(**et_params)
            et_model.fit(X_scaled, y_train, sample_weight=sample_weight)
            preds = et_model.predict(X_scaled)
        else:
            et_model = ExtraTreesClassifier(**et_params)
            y_int = (y_train > 0).astype(int)
            et_model.fit(X_scaled, y_int, sample_weight=sample_weight)
            preds = et_model.predict_proba(X_scaled)[:, 1]

    # Post-hoc Calibration for ExtraTrees (Vote-based probs are often uncalibrated)
    if task_type == 'classification':
        try:
            # ExtraTrees doesn't have an internal val set in this flow, but we can do a quick KFold calibration
            # Or simpler: Split training data here since ET is fast
            from sklearn.model_selection import train_test_split
            X_cal_tr, X_cal_val, y_cal_tr, y_cal_val, w_cal_tr, w_cal_val = train_test_split(
                X_scaled, y_int, sample_weight, test_size=0.2, random_state=42
            )

            # Re-fit a small ET for calibration reference (or use OOB if available, but manual split is safer)
            # Actually, using the fitted model on train data is biased.
            # Best is to use CalibratedClassifierCV, but that changes the 'model' object structure.
            # Let's stick to ProbabilityCalibrator using the full X_scaled predictions (Biased!) -> NO.
            # Correct approach: Pre-calibration using CV is best, but here we just fit calibrator on the
            # OOB estimates if bootstrap=True!

            if et_model.bootstrap and hasattr(et_model, 'oob_decision_function_'):
                # Use OOB predictions for calibration! Perfect for Random Forests.
                oob_preds = et_model.oob_decision_function_
                # oob_decision_function_ shape is (n_samples, n_classes)
                if oob_preds.ndim > 1:
                    oob_pos_preds = oob_preds[:, 1]
                else:
                    oob_pos_preds = oob_preds

                # Check for NaNs (unsampled points)
                mask = ~np.isnan(oob_pos_preds)
                if mask.sum() > 50:
                    calibrator = ProbabilityCalibrator(method='isotonic', min_samples=50, plot_calibration=False, save_plots=False)
                    cal_res = calibrator.fit(y_int[mask], oob_pos_preds[mask], sample_weights=sample_weight[mask] if sample_weight is not None else None)

                    preds_cal = calibrator.predict(preds)
                    tprint_info(f"   ⚖️  ET Calibrated (OOB): Brier improvement {cal_res['metrics'].get('brier_improvement', 0):.4f}")
                    preds = preds_cal

        except Exception as e:
            tprint_warning(f"   ⚠️ ET Calibration failed: {e}")

    # Calculate approximate SE
    if hasattr(et_model, 'estimators_'):
        if task_type == 'regression':
            tree_preds = np.array([tree.predict(X_scaled) for tree in et_model.estimators_])
        else:
            tree_preds = np.array([tree.predict_proba(X_scaled)[:, 1] for tree in et_model.estimators_])
        se = np.std(tree_preds, axis=0)
    else:
        se = np.ones(len(preds)) * 0.1

    return {
        'model': et_model,
        'cate': preds,
        'se': se,
        'scaler': scaler
    }

def train_catboost_model(
    X: pd.DataFrame,
    y_train: np.ndarray,
    model_name: str,
    task_type: str,
    huber_output: Dict[str, Any],
    sample_weight: Optional[np.ndarray] = None,
    config: Optional[Dict[str, Any]] = None,
    fast_mode: bool = False
) -> Dict[str, Any]:
    """
    Train CatBoost model with Huber constraints and DART boosting.
    """
    if not CATBOOST_AVAILABLE:
        tprint_warning(f"   ⚠️ CatBoost not available, skipping {model_name}")
        return None
    
    cfg = config or {}
    
    # Extract Huber constraints
    mono_dict = huber_output['monotonic_constraints']
    interaction_constraints = huber_output['interaction_constraints']
    
    # Prepare features
    selected_features = huber_output['selected_features']
    X_t = X[selected_features].copy().astype(np.float32)
    
    # Scale features for CatBoost
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_t).astype(np.float32)
    
    # Convert constraints to CatBoost format
    constraints = np.array([mono_dict.get(c, 0) for c in X_t.columns])
    
    # CatBoost parameters
    cb_params = {
        'iterations': 100 if fast_mode else 1000,
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 20.0,
        'subsample': 0.6,
        'rsm': 0.8,  # Random subspace method
        'bagging_temperature': 1,
        'random_strength': 5.0,
        'verbose': False,
        'allow_writing_files': False,
        'early_stopping_rounds': 30,
        'thread_count': -1,
        'random_seed': 42
    }
    
    # Add DART boosting for better performance
    cb_params['boosting_type'] = 'Dart'
    cb_params['dart_wait_time'] = 1
    
    # Set objective based on task type
    if task_type == 'classification':
        cb_params['loss_function'] = 'Logloss'
        cb_params['eval_metric'] = 'Logloss'
        y_int = (y_train > 0).astype(int)
    else:
        cb_params['loss_function'] = 'MAE'
        cb_params['eval_metric'] = 'MAE'
        y_int = y_train
    
    # Add monotonic constraints if available
    if np.any(constraints != 0):
        cb_params['monotone_constraints'] = constraints
    
    # Split data for validation
    split_idx = int(len(X_scaled) * 0.9)
    X_tr, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
    y_tr, y_val = y_int[:split_idx], y_int[split_idx:]
    w_tr = sample_weight[:split_idx] if sample_weight is not None else None
    w_val = sample_weight[split_idx:] if sample_weight is not None else None
    
    # Create CatBoost pools
    train_pool = cb.Pool(X_tr, label=y_tr, weight=w_tr)
    val_pool = cb.Pool(X_val, label=y_val, weight=w_val)
    
    try:
        # Train model
        model = cb.CatBoost(**cb_params)
        model.fit(train_pool, eval_set=val_pool)
        
        # Make predictions
        if task_type == 'classification':
            preds = model.predict_proba(X_scaled)[:, 1]
        else:
            preds = model.predict(X_scaled)
        
        # Calculate standard error approximation
        if hasattr(model, 'get_feature_importance'):
            # Use feature importance as proxy for uncertainty
            importance = model.get_feature_importance()
            se = np.std(importance) / np.sqrt(len(importance)) * np.ones(len(preds))
        else:
            se = np.ones(len(preds)) * 0.1
        
        tprint_success(f"   ✅ {model_name}: CatBoost trained successfully")
        tprint_info(f"      📊 Best iteration: {model.get_best_iteration()}")
        tprint_info(f"      🎯 Features: {len(selected_features)}")
        
        return {
            'model': model,
            'cate': preds,
            'se': se,
            'scaler': scaler
        }
        
    except Exception as e:
        tprint_warning(f"   ⚠️ {model_name}: CatBoost training failed: {e}")
        return None

def train_hubber_regression_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    model_name: str,
    task_type: str,
    huber_output: Dict[str, Any],
    sample_weight: Optional[np.ndarray] = None,
    config: Optional[Dict[str, Any]] = None,
    fast_mode: bool = False
) -> Dict[str, Any]:
    """
    Trains Huber Regression model with robust loss function.
    """
    tprint_info(f"   📊 Training Huber Regression ({task_type}): {model_name}...")
    
    selected_features = huber_output['selected_features']
    X_t = X_train[selected_features].copy().astype(np.float32)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_t).astype(np.float32)
    
    if task_type == 'regression':
        # Huber Regression for robust regression
        model = HuberRegressor(epsilon=1.35, alpha=0.1, max_iter=1000)
        model.fit(X_scaled, y_train, sample_weight=sample_weight)
        preds = model.predict(X_scaled)
    else:
        # For classification, use Huber as feature extractor + logistic calibration
        model = HuberRegressor(epsilon=1.35, alpha=0.1, max_iter=1000)
        model.fit(X_scaled, y_train, sample_weight=sample_weight)
        raw_preds = model.predict(X_scaled)
        # Apply sigmoid for classification
        preds = expit(raw_preds)
    
    # Calculate standard error approximation
    n_samples = len(preds)
    se = np.ones(n_samples) * 0.1  # Placeholder for Huber SE
    
    return {
        'model': model,
        'cate': preds,
        'se': se,
        'scaler': scaler
    }

def train_ridge_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    model_name: str,
    task_type: str,
    huber_output: Dict[str, Any],
    sample_weight: Optional[np.ndarray] = None,
    config: Optional[Dict[str, Any]] = None,
    fast_mode: bool = False
) -> Dict[str, Any]:
    """
    Trains Ridge models with different alpha values and selects the best.
    Also tries ElasticNet as replacement for Ridge alpha=10.
    Alpha candidates: 1, 5 (Ridge) + ElasticNet (l1_ratio=0.3, alpha=2.5)
    """
    tprint_info(f"   🏔️ Training Ridge + ElasticNet Models ({task_type}): {model_name}...")
    
    selected_features = huber_output['selected_features']
    X_t = X_train[selected_features].copy().astype(np.float32)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_t).astype(np.float32)
    
    # Alpha candidates to test
    alphas = [1.0, 5.0]  # Removed 10.0, will add ElasticNet instead
    best_score = float('-inf') if task_type == 'classification' else float('inf')
    best_model = None
    best_alpha = None
    best_preds = None
    
    for alpha in alphas:
        try:
            if task_type == 'regression':
                model = Ridge(alpha=alpha, random_state=42)
                model.fit(X_scaled, y_train, sample_weight=sample_weight)
                preds = model.predict(X_scaled)
                # Use IC as score for regression
                valid_mask = ~np.isnan(y_train) & ~np.isnan(preds)
                if np.sum(valid_mask) > 1:
                    ic = np.corrcoef(y_train[valid_mask], preds[valid_mask])[0, 1]
                    score = ic if not np.isnan(ic) else 0.0
                else:
                    score = 0.0
                
                if score > best_score:  # Higher IC is better
                    best_score = score
                    best_model = model
                    best_alpha = alpha
                    best_preds = preds
                    
            else:
                # Classification with Ridge + sigmoid calibration
                model = Ridge(alpha=alpha, random_state=42)
                model.fit(X_scaled, y_train, sample_weight=sample_weight)
                raw_preds = model.predict(X_scaled)
                preds = expit(raw_preds)  # Sigmoid for probability
                
                # Use AUC as score for classification
                try:
                    y_binary = (y_train > 0).astype(int)
                    auc = roc_auc_score(y_binary, preds)
                    score = auc
                except:
                    score = 0.5
                
                if score > best_score:  # Higher AUC is better
                    best_score = score
                    best_model = model
                    best_alpha = alpha
                    best_preds = preds
                    
        except Exception as e:
            tprint_warning(f"   ⚠️ Ridge alpha={alpha} failed: {e}")
            continue
    
    # Try ElasticNet as replacement for Ridge alpha=10
    try:
        from sklearn.linear_model import SGDClassifier
        if task_type == 'regression':
            # ElasticNet for regression via SGDRegressor
            from sklearn.linear_model import SGDRegressor
            elastic_model = SGDRegressor(
                loss='squared_error',
                penalty='elasticnet',
                l1_ratio=0.3,
                alpha=2.5,
                max_iter=5000,
                tol=1e-4,
                fit_intercept=True,
                random_state=42
            )
            elastic_model.fit(X_scaled, y_train, sample_weight=sample_weight)
            elastic_preds = elastic_model.predict(X_scaled)
            
            # Use IC as score for regression
            valid_mask = ~np.isnan(y_train) & ~np.isnan(elastic_preds)
            if np.sum(valid_mask) > 1:
                ic = np.corrcoef(y_train[valid_mask], elastic_preds[valid_mask])[0, 1]
                elastic_score = ic if not np.isnan(ic) else 0.0
            else:
                elastic_score = 0.0
                
        else:
            # ElasticNet for classification via SGDClassifier
            elastic_model = SGDClassifier(
                loss='log_loss',
                penalty='elasticnet',
                l1_ratio=0.3,
                alpha=2.5,
                max_iter=5000,
                tol=1e-4,
                fit_intercept=True,
                random_state=42,
                class_weight='balanced'
            )
            elastic_model.fit(X_scaled, y_train, sample_weight=sample_weight)
            elastic_raw_preds = elastic_model.decision_function(X_scaled)
            elastic_preds = expit(elastic_raw_preds)  # Sigmoid for probability
            
            # Use AUC as score for classification
            try:
                y_binary = (y_train > 0).astype(int)
                elastic_score = roc_auc_score(y_binary, elastic_preds)
            except:
                elastic_score = 0.5
        
        # Compare ElasticNet with best Ridge
        if elastic_score > best_score:
            tprint_info(f"   🎯 ElasticNet beats Ridge: {elastic_score:.4f} > {best_score:.4f}")
            best_score = elastic_score
            best_model = elastic_model
            best_alpha = "ElasticNet"
            best_preds = elastic_preds
        else:
            tprint_info(f"   📊 Ridge beats ElasticNet: {best_score:.4f} > {elastic_score:.4f}")
            
    except Exception as e:
        tprint_warning(f"   ⚠️ ElasticNet failed: {e}")
    
    if best_model is None:
        tprint_warning(f"   ⚠️ All Ridge models failed, using fallback")
        # Fallback to alpha=1.0
        best_model = Ridge(alpha=1.0, random_state=42)
        best_alpha = 1.0
        best_model.fit(X_scaled, y_train, sample_weight=sample_weight)
        if task_type == 'regression':
            best_preds = best_model.predict(X_scaled)
        else:
            best_preds = expit(best_model.predict(X_scaled))
        best_score = 0.0
    
    # Calculate standard error approximation
    n_samples = len(best_preds)
    se = np.ones(n_samples) * 0.1  # Placeholder for Ridge SE
    
    tprint_info(f"   ✅ Best Ridge alpha={best_alpha}, score={best_score:.4f}")
    
    return {
        'model': best_model,
        'cate': best_preds,
        'se': se,
        'scaler': scaler,
        'best_alpha': best_alpha,
        'score': best_score
    }

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
    Orchestrates the training of ExtraTrees, LGBM, XGBoost, Huber Regression, and Ridge models.
    Now with Regime-Aware Features, Alignment, and comprehensive model race.
    """
    cfg = config or {}
    base_model_cols = cfg.get('base_model_cols', [])
    if not base_model_cols:
        base_model_cols = [c for c in X.columns if c.startswith('prob_') and not c.endswith('_oof')]
    
    # Context X (Meta Features)
    context_cols = [c for c in X.columns if c not in base_model_cols and c != 'regime_label']
    X_context = X[context_cols].fillna(0).replace([np.inf, -np.inf], 0)
    X_context = X_context.astype(np.float32)
    
    # Horizon outcomes
    y_alpha_48 = cfg.get('y_alpha_48', y_alpha * 1.5)
    y_prob_48 = cfg.get('y_prob_48', y_prob)

    models_store = {}
    irm_env_indices = cfg.get('irm_env_indices', [])
    irm_lambda = cfg.get('irm_lambda', 1.0)
    
    tasks = [
        ('12', 'alpha', y_alpha, w_alpha, 'regression'),
        ('12', 'prob', y_prob, w_prob, 'classification'),
        ('48', 'alpha', y_alpha_48, w_alpha, 'regression'),
        ('48', 'prob', y_prob_48, w_prob, 'classification')
    ]

    for horizon, target_name, y_target, w_target, task_type in tasks:
        suffix = f"{horizon}_{'reg' if task_type == 'regression' else 'cls'}"

        # 1. Prepare Huber Teacher (Fold-Local & Stability Gated)
        tprint_info(f"🎓 Running Robust Huber Teacher for {suffix}...")

        # Use existing utility
        huber_out = prepare_huber_teacher_outputs(
            X_context,
            pd.Series(y_target, index=X_context.index),
            pruning_percentile=15,
            n_time_splits=5, # Stability check
            use_irm=bool(irm_env_indices),
            irm_env_indices=irm_env_indices,
            irm_lambda=irm_lambda
        )

        # 2. Train ExtraTrees
        et_res = train_extratrees_constrained(
            X_context, y_target, f"ET_{suffix}", task_type, huber_out, w_target, cfg, fast_mode
        )
        models_store[f"et_{suffix}"] = et_res

        # 3. Train LGBM
        lgbm_res = train_lgbm_model(
            X_context, y_target, f"LGBM_{suffix}", task_type, huber_out, w_target, cfg, fast_mode
        )
        models_store[f"lgbm_{suffix}"] = lgbm_res

        # 4. Train XGBoost
        xgb_res = train_xgboost_model(
            X_context, y_target, f"XGB_{suffix}", task_type, huber_out, w_target, cfg, fast_mode
        )
        models_store[f"xgb_{suffix}"] = xgb_res

        if irm_env_indices and len(irm_env_indices) > 1:
            if task_type == 'regression':
                ridge_irm = IRMLinearRegressor(
                    loss_type='ridge',
                    alpha=cfg.get('irm_linear_alpha', 1.0),
                    irm_lambda=irm_lambda
                )
                ridge_irm.fit(X_context.values, y_target, irm_env_indices, sample_weight=w_target)
                models_store[f"irm_ridge_{suffix}"] = {
                    'model': ridge_irm,
                    'cate': ridge_irm.predict(X_context.values),
                    'se': np.zeros(len(y_target)),
                    'scaler': None
                }

                elastic_irm = IRMLinearRegressor(
                    loss_type='elasticnet',
                    alpha=cfg.get('irm_elastic_alpha', 0.01),
                    l1_ratio=cfg.get('irm_elastic_l1_ratio', 0.5),
                    irm_lambda=irm_lambda
                )
                elastic_irm.fit(X_context.values, y_target, irm_env_indices, sample_weight=w_target)
                models_store[f"irm_elasticnet_{suffix}"] = {
                    'model': elastic_irm,
                    'cate': elastic_irm.predict(X_context.values),
                    'se': np.zeros(len(y_target)),
                    'scaler': None
                }
            else:
                # Use LogLoss for proper probabilistic output in classification
                ridge_irm = IRMLinearClassifier(
                    loss_type='logloss',
                    alpha=cfg.get('irm_linear_alpha', 1.0),
                    irm_lambda=irm_lambda
                )
                ridge_irm.fit(X_context.values, y_target, irm_env_indices, sample_weight=w_target)
                models_store[f"irm_ridge_{suffix}"] = {
                    'model': ridge_irm,
                    'cate': ridge_irm.predict_proba(X_context.values)[:, 1],
                    'se': np.zeros(len(y_target)),
                    'scaler': None
                }

                elastic_irm = IRMLinearClassifier(
                    loss_type='logloss',
                    alpha=cfg.get('irm_elastic_alpha', 0.01),
                    l1_ratio=cfg.get('irm_elastic_l1_ratio', 0.5),
                    irm_lambda=irm_lambda
                )
                elastic_irm.fit(X_context.values, y_target, irm_env_indices, sample_weight=w_target)
                models_store[f"irm_elasticnet_{suffix}"] = {
                    'model': elastic_irm,
                    'cate': elastic_irm.predict_proba(X_context.values)[:, 1],
                    'se': np.zeros(len(y_target)),
                    'scaler': None
                }

        # 5. Train CatBoost
        catboost_res = train_catboost_model(
            X_context, y_target, f"CatBoost_{suffix}", task_type, huber_out, w_target, cfg, fast_mode
        )
        if catboost_res is not None:
            models_store[f"catboost_{suffix}"] = catboost_res

        # 6. Train Huber Regression
        huber_res = train_hubber_regression_model(
            X_context, y_target, f"Huber_{suffix}", task_type, huber_out, w_target, cfg, fast_mode
        )
        models_store[f"huber_{suffix}"] = huber_res

        # 7. Train Ridge (best alpha selected internally)
        ridge_res = train_ridge_models(
            X_context, y_target, f"Ridge_{suffix}", task_type, huber_out, w_target, cfg, fast_mode
        )
        models_store[f"ridge_{suffix}"] = ridge_res

    all_results = {
        'models': models_store
    }
    
    return all_results
