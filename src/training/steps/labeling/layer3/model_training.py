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
from scipy.special import expit

from src.utils.huber_regressor_for_trees import prepare_huber_teacher_outputs
from src.training.steps.labeling.layer3.feature_engineering import downcast_float

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
    Uses 'quantile' objective for regression tasks.
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
    monotone_constraints = list(huber_output['monotonic_constraints'])
    interaction_constraints = huber_output['interaction_constraints']

    # Warm start
    init_score = huber_output['warm_start']['train']

    # Configure objective
    if task_type == 'classification':
        objective = 'binary'
        metric = 'binary_logloss'
    else:
        objective = 'quantile'
        metric = 'quantile'

    params = {
        'objective': objective,
        'metric': metric,
        'verbosity': -1,
        'linear_tree': True,
        'path_smooth': 20 if fast_mode else 50,
        'min_data_in_leaf': 100 if fast_mode else 300,
        'num_leaves': 7 if fast_mode else 12,
        'lambda_l1': 1.0,
        'learning_rate': 0.05,
        'n_estimators': 100 if fast_mode else 500,
        'monotone_constraints': monotone_constraints,
        'interaction_constraints': interaction_constraints if interaction_constraints else "",
        'early_stopping_round': 20,
        'seed': 42
    }

    if objective == 'quantile':
        params['alpha'] = 0.5

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

    # Optuna Integration
    if config and 'optuna_trial' in config:
        trial = config['optuna_trial']
        metric_name = params['metric']
        callbacks.append(optuna.integration.LightGBMPruningCallback(trial, metric_name, valid_name='valid_0'))

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
    Using objective='reg:quantileerror' for robust regression.
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
    huber_model = huber_output['huber_model']
    coeffs_series = pd.Series(huber_model.coef_, index=X_t.columns)
    X_rotated = apply_huber_rotation_logic(X_scaled, coeffs_series)
    # Ensure rotated is float32
    X_rotated = X_rotated.astype(np.float32)

    # 2. Constraints (Only on original features)
    orig_constraints = huber_output['monotonic_constraints']
    cst_dict = dict(zip(selected_features, orig_constraints))

    final_constraints = []
    for col in X_rotated.columns:
        final_constraints.append(cst_dict.get(col, 0))

    # 3. Warm Start
    base_margin = huber_output['warm_start']['train']

    params = {
        'objective': 'reg:quantileerror' if task_type == 'regression' else 'binary:logistic',
        'quantile_alpha': 0.5,
        'n_estimators': 100 if fast_mode else 300,
        'learning_rate': 0.05,
        'max_depth': 3 if fast_mode else 4,
        'min_child_weight': 10,
        'gamma': 0.5,
        'colsample_bynode': 0.4,
        'reg_lambda': 50,
        'num_parallel_tree': 7,
        'monotone_constraints': tuple(final_constraints),
        'n_jobs': -1,
        'verbosity': 0
    }

    if config and 'xgb_params' in config:
        params.update(config['xgb_params'])

    # Internal split for early stopping
    split_idx = int(len(X_rotated) * 0.9)
    X_tr, X_val = X_rotated.iloc[:split_idx], X_rotated.iloc[split_idx:]
    y_tr, y_val = y_train[:split_idx], y_train[split_idx:]
    w_tr = sample_weight[:split_idx] if sample_weight is not None else None
    w_val = sample_weight[split_idx:] if sample_weight is not None else None
    bm_tr = base_margin[:split_idx]
    bm_val = base_margin[split_idx:]

    model = xgb.XGBRegressor(**params) if task_type == 'regression' else xgb.XGBClassifier(**params)

    callbacks = []
    if config and 'optuna_trial' in config:
        trial = config['optuna_trial']
        callbacks.append(optuna.integration.XGBoostPruningCallback(trial, "validation_0-" + ("quantile" if task_type == 'regression' else "logloss")))

    model.fit(
        X_tr, y_tr,
        sample_weight=w_tr,
        base_margin=bm_tr,
        eval_set=[(X_val, y_val)],
        sample_weight_eval_set=[w_val],
        base_margin_eval_set=[bm_val],
        early_stopping_rounds=20,
        verbose=False,
        callbacks=callbacks
    )

    # Prediction
    if task_type == 'regression':
        preds = model.predict(X_rotated, base_margin=base_margin)
    else:
        # Predict Probabilities for Classification
        try:
             preds = model.predict_proba(X_rotated, base_margin=base_margin)[:, 1]
        except TypeError:
            # Fallback if base_margin not supported in predict_proba
            dmat = xgb.DMatrix(X_rotated, base_margin=base_margin)
            preds = model.get_booster().predict(dmat)

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
    Trains ExtraTrees model with Monotonic Constraints and Feature Pruning from Huber Teacher.
    """
    tprint_info(f"   🌳 Training ExtraTrees ({task_type}): {model_name} with Constraints...")

    cfg = config or {}
    et_params = cfg.get('et_params', {
        'n_estimators': 100 if fast_mode else 300,
        'max_depth': 10 if fast_mode else 20,
        'min_samples_leaf': 20,
        'bootstrap': True,
        'n_jobs': -1,
        'random_state': 42
    })

    selected_features = huber_output['selected_features']
    # Downcast
    X_t = X_train[selected_features].copy().astype(np.float32)

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_t).astype(np.float32)

    constraints = np.array(huber_output['monotonic_constraints'])

    try:
        if task_type == 'regression':
            et_model = ExtraTreesRegressor(
                monotonic_cst=constraints,
                **et_params
            )
            et_model.fit(X_scaled, y_train, sample_weight=sample_weight)
            preds = et_model.predict(X_scaled)
        else:
            et_model = ExtraTreesClassifier(
                monotonic_cst=constraints,
                **et_params
            )
            y_int = (y_train > 0).astype(int)
            et_model.fit(X_scaled, y_int, sample_weight=sample_weight)
            preds = et_model.predict_proba(X_scaled)[:, 1]

        # Calculate approximate SE
        if hasattr(et_model, 'estimators_'):
            if task_type == 'regression':
                tree_preds = np.array([tree.predict(X_scaled) for tree in et_model.estimators_])
            else:
                tree_preds = np.array([tree.predict_proba(X_scaled)[:, 1] for tree in et_model.estimators_])
            se = np.std(tree_preds, axis=0)
        else:
            se = np.ones(len(preds))

        return {
            'model': et_model,
            'cate': preds,
            'se': se,
            'scaler': scaler
        }

    except TypeError as e:
        if "unexpected keyword argument 'monotonic_cst'" in str(e):
             tprint_warning(f"   ⚠️ Scikit-learn version does not support monotonic_cst for ExtraTrees. Training without constraints.")
             if task_type == 'regression':
                et_model = ExtraTreesRegressor(**et_params)
                et_model.fit(X_scaled, y_train, sample_weight=sample_weight)
                preds = et_model.predict(X_scaled)
             else:
                et_model = ExtraTreesClassifier(**et_params)
                y_int = (y_train > 0).astype(int)
                et_model.fit(X_scaled, y_int, sample_weight=sample_weight)
                preds = et_model.predict_proba(X_scaled)[:, 1]

             return {
                'model': et_model,
                'cate': preds,
                'se': np.zeros(len(preds)),
                'scaler': scaler
             }
        else:
            raise e

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
    Orchestrates the training of ExtraTrees, LGBM, and XGBoost models.
    """
    cfg = config or {}
    base_model_cols = cfg.get('base_model_cols', [])
    if not base_model_cols:
        base_model_cols = [c for c in X.columns if c.startswith('prob_') and not c.endswith('_oof')]
    
    # Context X
    context_cols = [c for c in X.columns if c not in base_model_cols and c != 'regime_label']
    X_context = X[context_cols].fillna(0).replace([np.inf, -np.inf], 0)
    # Ensure float32
    X_context = X_context.astype(np.float32)
    
    # Horizon outcomes
    y_alpha_48 = cfg.get('y_alpha_48', y_alpha * 1.5)
    y_prob_48 = cfg.get('y_prob_48', y_prob)

    models_store = {}
    
    tasks = [
        ('12', 'alpha', y_alpha, w_alpha, 'regression'),
        ('12', 'prob', y_prob, w_prob, 'classification'),
        ('48', 'alpha', y_alpha_48, w_alpha, 'regression'),
        ('48', 'prob', y_prob_48, w_prob, 'classification')
    ]

    for horizon, target_name, y_target, w_target, task_type in tasks:
        suffix = f"{horizon}_{'reg' if task_type == 'regression' else 'cls'}"

        # 1. Prepare Huber Teacher
        tprint_info(f"🎓 Running Huber Teacher for {suffix}...")
        huber_out = prepare_huber_teacher_outputs(X_context, y_target, pruning_percentile=15)

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

    all_results = {
        'models': models_store
    }
    
    return all_results
