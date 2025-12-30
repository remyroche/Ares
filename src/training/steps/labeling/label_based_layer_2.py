"""
Label-Based Layer 2: Regime-Conditional Geometry Optimization & Meta-Labeling
-----------------------------------------------------------------------------
This module implements the Layer 2 architecture for meta-labeling, focusing on
regime-conditional barrier optimization and ML learnability probes.

It performs:
1. Event generation based on volatility-scaled returns.
2. Regime-conditional barrier family assignment.
3. Independent optimization of barrier geometries (Kappa/Horizon) per family using Optuna.
4. MFE/MAE Dominance Labeling: Label = 1 if MFE > Kappa * MAE.
5. Stability checks (Time-Flip) and Learnability probes.
6. Bagged output generation with family-level cap checks.
7. Enhanced LGBM training with Robust Focal Loss and Tree Variance calculation.
"""

import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
import xgboost as xgb
import catboost
import os
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import spearmanr, rankdata
from scipy.special import expit, ndtri
from scipy.spatial.distance import euclidean, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from joblib import Parallel, delayed
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict, field
import logging
import copy
import warnings
import sys

try:
    from joblib.externals.loky.process_executor import BrokenProcessPool, TimeoutError as LokyTimeoutError
except Exception:  # pragma: no cover - older joblib versions
    class LokyTimeoutError(Exception):
        """Fallback timeout exception."""

    class BrokenProcessPool(Exception):
        """Fallback broken pool exception."""

# Import tprint for enhanced logging
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success

# Suppress LightGBM verbose warnings for clean output
warnings.filterwarnings("ignore")

# Import compute_realized_returns from the existing module
from src.training.steps.labeling.feature_generation_meta_labeling_step import (
    compute_realized_returns,
    generate_primary_signals,
)
from src.training.steps.labeling.mtf_feature_generation import (
    create_meta_features,
    get_efficiency_ratio
)
from src.training.steps.labeling.generate_weights_per_label import finalize_sample_weights

from src.utils.purged_kfold import PurgedKFoldTime

# Import selection logic
from src.training.steps.labeling.label_geometry_selection import (
    select_geometries,
    Event,
    Geometry,
    MIN_SL_PCT,
    MIN_TP_SL_RATIO
)

from src.training.steps.labeling.regime_leaf_feature_extractor import (
    extract_regime_leaf_onehot_features,
)

# Configure logging
logger = logging.getLogger(__name__)
_lgb_logger = logging.getLogger("lightgbm")
_lgb_logger.setLevel(logging.ERROR)
_lgb_logger.propagate = False

# Constants for Layer 2 Model Training (defaults/fixed) - Less Regularized
LAYER2_MODEL_CONSTANTS = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'max_depth': -1,
    'learning_rate': 0.03,  # Reduced from 0.05 for better convergence
    'lambda_l1': 0.01,
    'lambda_l2': 0.05,       # Reduced from 0.1 to allow more learning  
    'num_leaves': 31,
    'min_data_in_leaf': 5, # Reduced to 5 (Hunter Mode)
    'min_sum_hessian_in_leaf': 1e-3,
    'feature_fraction': 0.95, # Increased from 0.9 to use more features
    'bagging_fraction': 0.95,  # Increased from 0.9 to use more data
    'bagging_freq': 1,       # Reduced from 3 for more frequent updates
    'verbose': -1,
    'random_state': 42,
    'n_jobs': 1,
    'is_unbalance': False,
    'scale_pos_weight': 1,
    'min_gain_to_split': 0.001, # Reduced from 0.003 to allow more splits
    'min_child_weight': 0.0001, # Reduced from 0.0005 to allow more growth
}

class RobustFocalLoss:
    """
    Production-grade Focal Loss for LightGBM in Financial ML.

    De Prado 1.3: Preference vs. Outcome Separation
    - Labels encode objective outcomes (Dominance).
    - Loss encodes preferences (Utility Shaping).

    This class handles the Utility Shaping via asymmetric gammas.

    Enhancements over standard Focal Loss:
    1. Asymmetric Gamma: Penalize False Positives (Traps, gamma_fp) harder than Missed Opportunities (gamma_fn).
    2. Label Smoothing: Prevents the model from becoming over-confident on noisy labels.
    3. Gradient Capping & Mixing: Stabilizes training against outliers.
    4. Guardrails: w_cap prevents the loss from exploding on 'impossible' examples.
    """

    def __init__(
        self,
        gamma_pos=1.0, # gamma_fn: Preference for Opportunity (Missed Upside)
        gamma_neg=2.5, # gamma_fp: Preference for Safety (Traps)
        alpha=None,
        grad_clip=5.0,
        w_cap=3.0,
        mix=0.25,
        label_smoothing=0.02,
        verbose=True
    ):
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.grad_clip = grad_clip
        self.w_cap = w_cap
        self.mix = mix
        self.label_smoothing = label_smoothing
        self.alpha = alpha
        self.verbose = verbose
        self._is_init = False

    def _init_alpha(self, labels):
        """Auto-compute alpha based on prevalence if not provided."""
        if self.alpha is None:
            n_pos = np.sum(labels > 0.5)
            n_total = len(labels)
            if n_total > 0:
                # Standard inverse frequency: High alpha for rare positives
                self.alpha = 1.0 - (n_pos / n_total)
            else:
                self.alpha = 0.5

        # Clamp alpha for safety
        self.alpha = np.clip(self.alpha, 0.05, 0.95)

        if self.verbose:
            logger.info(f"[LGBM Focal] Gamma(+):{self.gamma_pos} Gamma(-):{self.gamma_neg} | Alpha:{self.alpha:.4f}")

        self._is_init = True

    def __call__(self, preds, train_data):
        if hasattr(train_data, 'get_label'):
             labels = train_data.get_label()
        else:
             labels = train_data

        # Lazy init alpha on first call to handle data loading
        if not self._is_init:
            self._init_alpha(labels)

        # 1. Label Smoothing (Crucial for Finance)
        # Softens hard 0/1 to e.g. 0.02/0.98. Reduces overfitting to noise.
        y_smooth = labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # 2. Robust Sigmoid
        p = expit(preds)
        p = np.clip(p, 1e-7, 1 - 1e-7)

        # 3. Vectorized Asymmetric Gamma
        # If label is positive, use gamma_pos (usually lower, e.g. 1.0)
        # If label is negative, use gamma_neg (usually higher, e.g. 2.5) to filter traps
        gamma_arr = np.where(labels > 0.5, self.gamma_pos, self.gamma_neg)

        # 4. Focal Weights with Capping
        # For pos: (1-p)^g | For neg: p^g
        focal_weight = np.where(labels > 0.5, (1 - p), p) ** gamma_arr

        # Guardrail: Cap the weight. If an example is "impossible" (p=0.0001, y=1),
        # don't let the gradient explode (which would destroy the tree structure).
        focal_weight = np.minimum(focal_weight, self.w_cap)

        # 5. Gradient & Hessian Calculation
        # We use the "Modulated Cross Entropy" approximation for stability.
        # It has the same root properties but is numerically safer than the full derivative.

        # Standard LogLoss Gradient: (p - y)
        grad_bce = p - y_smooth

        # Focal Gradient: alpha * weight * (p - y)
        # We apply alpha asymmetry manually
        alpha_factor = np.where(labels > 0.5, self.alpha, (1 - self.alpha))
        grad_focal = alpha_factor * focal_weight * grad_bce

        # Standard LogLoss Hessian: p * (1-p)
        hess_bce = p * (1 - p)

        # Focal Hessian: Scaled by weight.
        # Note: We do NOT use the complex 2nd derivative term involving logs.
        # In GBDT, a positive-definite diagonal approximation (like this) works better.
        hess_focal = alpha_factor * focal_weight * hess_bce

        # 6. Mixing (Stability Anchor)
        # Blend pure Focal Loss with standard BCE to ensure convergence
        grad = self.mix * grad_focal + (1 - self.mix) * grad_bce
        hess = self.mix * hess_focal + (1 - self.mix) * hess_bce

        # 7. Clipping & Safety
        if self.grad_clip:
            grad = np.clip(grad, -self.grad_clip, self.grad_clip)

        hess = np.maximum(hess, 1e-6) # Prevent divide-by-zero

        return grad, hess


class XGBFocalLoss:
    """
    Focal Loss for XGBoost (custom objective function).
    Fully matches RobustFocalLoss behavior (LGBM) including asymmetric gamma.
    """

    def __init__(
        self,
        gamma_pos=1.0,
        gamma_neg=2.5,
        alpha=None,
        grad_clip=5.0,
        w_cap=3.0,
        mix=0.25,
        label_smoothing=0.02,
        verbose=True
    ):
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.grad_clip = grad_clip
        self.w_cap = w_cap
        self.mix = mix
        self.label_smoothing = label_smoothing
        self.alpha = alpha
        self.verbose = verbose
        self._is_init = False

    def _init_alpha(self, labels):
        """Auto-compute alpha based on prevalence if not provided."""
        if self.alpha is None:
            n_pos = np.sum(labels > 0.5)
            n_total = len(labels)
            if n_total > 0:
                # Standard inverse frequency
                self.alpha = 1.0 - (n_pos / n_total)
            else:
                self.alpha = 0.5

        # Clamp alpha for safety
        self.alpha = np.clip(self.alpha, 0.05, 0.95)
        self._is_init = True

    def __call__(self, preds, dtrain):
        """
        Args:
            preds: Raw predictions (logits) from XGBoost
            dtrain: xgb.DMatrix with labels

        Returns:
            grad, hess: Gradient and hessian arrays
        """
        # Detect if called via SKLearn (y_true, y_pred) or Native (preds, dtrain)
        is_sklearn = False
        try:
            # Native: dtrain is DMatrix
            if hasattr(dtrain, 'get_label'):
                labels = dtrain.get_label()
                logits = preds
            # SKLearn: dtrain is actually y_pred (logits), preds is y_true
            elif isinstance(dtrain, np.ndarray):
                labels = preds
                logits = dtrain
                is_sklearn = True
            else:
                # Fallback assuming Native/Standard (preds, labels-as-array)
                labels = dtrain
                logits = preds
        except Exception:
             labels = dtrain
             logits = preds

        # Lazy init alpha
        if not self._is_init:
            self._init_alpha(labels)

        # 1. Label Smoothing
        y_smooth = labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # 2. Robust Sigmoid (preds are logits)
        # Using 1/(1+exp(-x))
        p = 1.0 / (1.0 + np.exp(-logits))
        p = np.clip(p, 1e-7, 1 - 1e-7)

        # 3. Vectorized Asymmetric Gamma
        gamma_arr = np.where(labels > 0.5, self.gamma_pos, self.gamma_neg)

        # 4. Focal Weights with Capping
        # For pos: (1-p)^g | For neg: p^g
        focal_weight = np.where(labels > 0.5, (1 - p), p) ** gamma_arr
        focal_weight = np.minimum(focal_weight, self.w_cap)

        # 5. Gradient & Hessian Calculation

        # Standard LogLoss Gradient: (p - y)
        grad_bce = p - y_smooth

        # Focal Gradient
        alpha_factor = np.where(labels > 0.5, self.alpha, (1 - self.alpha))
        grad_focal = alpha_factor * focal_weight * grad_bce

        # Standard LogLoss Hessian: p * (1-p)
        hess_bce = p * (1 - p)

        # Focal Hessian
        hess_focal = alpha_factor * focal_weight * hess_bce

        # 6. Mixing
        grad = self.mix * grad_focal + (1 - self.mix) * grad_bce
        hess = self.mix * hess_focal + (1 - self.mix) * hess_bce

        # 7. Clipping & Safety
        if self.grad_clip:
            grad = np.clip(grad, -self.grad_clip, self.grad_clip)

        hess = np.maximum(hess, 1e-6)

        return grad, hess


# ==============================================================================
# Multi-Output Model Functions (Cross-Geometry Learning)
# ==============================================================================

def generate_probe_features(price: pd.Series, volume: pd.Series) -> pd.DataFrame:
    """
    Generates a standardized 'Basis Set' of features for Geometry Validation.
    These use fixed industry-standard lookbacks to serve as a robust benchmark.
    """
    df = pd.DataFrame(index=price.index)

    # 1. Momentum (Immediate & Short-term)
    df['ret_1'] = np.log(price).diff(1)
    df['ret_12'] = np.log(price).diff(12) # Context momentum

    # 2. Oscillator (RSI 14)
    # Simple pandas implementation
    delta = price.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    # Avoid div/0
    rs = gain / (loss + 1e-9)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # 3. Volatility Regime (20 vs 100)
    # Is recent vol expanding relative to history?
    vol_20 = df['ret_1'].rolling(20).std()
    vol_100 = df['ret_1'].rolling(100).std()
    df['vol_ratio'] = vol_20 / (vol_100 + 1e-6)

    # 4. Trend Distance (50 bar MA)
    # Are we far from the mean?
    ma_50 = price.rolling(50).mean()
    df['trend_dist'] = (price / (ma_50 + 1e-9)) - 1

    # 5. Liquidity Shock (Volume vs 20 bar avg)
    vol_ma_20 = volume.rolling(20).mean()
    df['vol_shock'] = volume / (vol_ma_20 + 1e-6)

    # Clean up (Probe models hate NaNs)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    return df


def _build_multi_target_matrix(
    events_df: pd.DataFrame,
    geometries: List,
    all_geometry_labels: Dict[str, pd.Series],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build multi-target matrix for all geometries + compute class-aware sample weights.
    
    Args:
        events_df: Event DataFrame
        geometries: List of GeometryTrial objects
        all_geometry_labels: Dict of {geometry_uuid: labels_series}
    
    Returns:
        Y_multi: (n_samples, n_geometries) target matrix
        sample_weights: (n_samples,) weights (upweight rows with many positive labels)
    """
    n_samples = len(events_df)
    n_geometries = len(geometries)
    
    Y_multi = np.zeros((n_samples, n_geometries), dtype=float)
    
    # Fill target matrix
    for i, geometry in enumerate(geometries):
        geo_id = geometry.uuid
        if geo_id in all_geometry_labels:
            labels = all_geometry_labels[geo_id]
            # Align to events_df index
            Y_multi[:, i] = labels.reindex(events_df.index, fill_value=0.0).values
        else:
            Y_multi[:, i] = 0.0
    
    # Compute sample weights (upweight samples with positive labels)
    # Strategy: weight = 1.0 + 3.0 * (ratio of positive geometries)
    positive_ratio = np.mean(Y_multi > 0.5, axis=1)
    sample_weights = 1.0 + 3.0 * positive_ratio
    sample_weights = np.clip(sample_weights, 1.0, 4.0)
    
    return Y_multi, sample_weights


def _train_extratrees_multioutput(
    X_events: pd.DataFrame,
    Y_multi: np.ndarray,
    sample_weights: np.ndarray,
    random_state: int = 42,
) -> Optional[np.ndarray]:
    """
    Train ExtraTreesRegressor on all geometries with class weighting.
    
    Returns:
        Predictions array (n_samples, n_geometries) or None if failed
    """
    try:
        logger.info("Training ExtraTreesRegressor (multi-output) with class weighting...")
        
        model = ExtraTreesRegressor(
            n_estimators=800,
            max_depth=None,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            bootstrap=False,
            n_jobs=-1,
            random_state=random_state,
        )
        
        # Fit with sample weights
        model.fit(X_events, Y_multi, sample_weight=sample_weights)
        
        # Predict
        preds = model.predict(X_events)
        
        # Convert to probabilities (sigmoid)
        probs = expit(preds)
        
        logger.info(f"   ExtraTrees trained on {Y_multi.shape[1]} geometries, {Y_multi.shape[0]} samples")
        
        return probs
        
    except Exception as e:
        logger.warning(f"ExtraTreesRegressor multi-output failed: {e}")
        return None


def _train_pls_multioutput(
    X_events: pd.DataFrame,
    Y_multi: np.ndarray,
    sample_weights: np.ndarray,
    n_components: int = 10,
    random_state: int = 42,
) -> Optional[np.ndarray]:
    """
    Train PLSRegression on all geometries with class weighting (via replication).
    
    Returns:
        Predictions array (n_samples, n_geometries) or None if failed
    """
    try:
        logger.info("Training PLSRegression (multi-output) with class weighting...")
        
        # PLS doesn't support sample_weight, so replicate samples
        replication_counts = np.round(sample_weights).astype(int)
        
        # Build replicated dataset
        X_replicated = []
        Y_replicated = []
        
        for i in range(len(X_events)):
            count = replication_counts[i]
            for _ in range(count):
                X_replicated.append(X_events.iloc[i].values)
                Y_replicated.append(Y_multi[i])
        
        X_replicated = np.array(X_replicated)
        Y_replicated = np.array(Y_replicated)
        
        # Scale
        scaler_X = StandardScaler()
        scaler_Y = StandardScaler()
        
        X_scaled = scaler_X.fit_transform(X_replicated)
        Y_scaled = scaler_Y.fit_transform(Y_replicated)
        
        # Determine n_components
        max_components = min(X_scaled.shape[0], X_scaled.shape[1], n_components)
        
        # Train PLS
        pls = PLSRegression(n_components=max_components, scale=False)
        pls.fit(X_scaled, Y_scaled)
        
        # Predict on original (unscaled) data
        X_orig_scaled = scaler_X.transform(X_events)
        Y_pred_scaled = pls.predict(X_orig_scaled)
        Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)
        
        # Convert to probabilities
        probs = expit(Y_pred)
        
        logger.info(f"   PLS trained on {Y_multi.shape[1]} geometries, {len(X_replicated)} replicated samples")
        
        return probs
        
    except Exception as e:
        logger.warning(f"PLSRegression multi-output failed: {e}")
        return None


def _quick_5model_race(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    random_state: int = 42,
) -> Tuple[str, Dict[str, float]]:
    """
    Fast 5-model race to determine best model type for a geometry.
    Includes Linear Tree variants to capture simple linear relationships.
    
    Returns:
        Tuple of (winning_model_type, scores_dict)
    """
    from sklearn.metrics import roc_auc_score
    
    scores = {}
    
    tprint_info(f"    Running Model Race with Custom Focal Loss (LGBM/XGB)...")
    
    # --- Model 1: LGBM Standard (Non-linear) ---
    try:
        focal_lgbm = RobustFocalLoss(gamma_pos=1.5, gamma_neg=3.0, alpha=None, verbose=False)
        
        params_lgbm = {
            'n_estimators': 500,
            'learning_rate': 0.03,
            'num_leaves': 63, # Regularized from 127
            'max_depth': 7,
            'min_data_in_leaf': 20,
            'feature_fraction': 0.8, # Added regularization
            'verbosity': -1,
            'random_state': random_state,
            'metric': 'average_precision',
            'objective': focal_lgbm,
        }
        
        train_ds = lgb.Dataset(X_train, label=y_train)
        val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)
        
        model_lgbm = lgb.train(
            params_lgbm,
            train_ds,
            valid_sets=[val_ds],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        
        preds_lgbm = model_lgbm.predict(X_val)
        preds_lgbm = expit(preds_lgbm)
        scores['lgbm'] = roc_auc_score(y_val, preds_lgbm)
    except Exception as e:
        logger.warning(f"LGBM race failed: {e}")
        scores['lgbm'] = 0.0
        
    # --- Model 2: LGBM Linear (Extra Trees / Shallow) ---
    try:
        # Use ExtraTrees-like splits and shallower depth for "linear-ish" behavior
        params_lgbm_lin = {
            'n_estimators': 500,
            'learning_rate': 0.05,
            'num_leaves': 63,
            'max_depth': 4, # Shallower
            'extra_trees': True, # randomized splits
            'min_data_in_leaf': 20,
            'verbosity': -1,
            'random_state': random_state,
            'metric': 'average_precision',
            'objective': focal_lgbm,
            'feature_fraction': 0.8,
        }
        
        model_lgbm_lin = lgb.train(
            params_lgbm_lin,
            train_ds,
            valid_sets=[val_ds],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        
        preds_lgbm_lin = model_lgbm_lin.predict(X_val)
        scores['lgbm_linear'] = roc_auc_score(y_val, preds_lgbm_lin)
    except Exception as e:
        logger.warning(f"LGBM Linear race failed: {e}")
        scores['lgbm_linear'] = 0.0
    
    # --- Model 3: XGBoost Standard ---
    try:
        focal_xgb = XGBFocalLoss(gamma_pos=1.5, gamma_neg=3.0, alpha=None)
        
        model_xgb = xgb.XGBClassifier(
            n_estimators=400,
            learning_rate=0.04,
            max_depth=6,
            min_child_weight=10,
            subsample=0.8,
            colsample_bytree=0.8,
            objective=focal_xgb,
            eval_metric='aucpr', # PR-AUC for imbalanced data
            early_stopping_rounds=30,
            verbosity=0,
            random_state=random_state,
            n_jobs=1,
        )
        
        model_xgb.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        
        preds_xgb = model_xgb.predict_proba(X_val)[:, 1]
        scores['xgb'] = roc_auc_score(y_val, preds_xgb)
    except Exception as e:
        logger.warning(f"XGBoost race failed: {e}")
        scores['xgb'] = 0.0
        
    # --- Model 4: XGBoost Linear (gblinear) ---
    try:
        # Note: gblinear uses its own objective logic, custom objective might not work well with it directly 
        # or require gradients. Our XGBFocalLoss provides gradients, so it should work.
        # But gblinear expects simple gradients.
        
        model_xgb_lin = xgb.XGBClassifier(
            booster='gblinear',
            n_estimators=100,
            learning_rate=0.1,
            objective='binary:logistic', # Use standard objective for linear to be safe/stable
            eval_metric=['auc', 'aucpr'],
            early_stopping_rounds=30,
            verbosity=0,
            random_state=random_state,
            n_jobs=1,
        )
        
        model_xgb_lin.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        
        preds_xgb_lin = model_xgb_lin.predict_proba(X_val)[:, 1]
        scores['xgb_linear'] = roc_auc_score(y_val, preds_xgb_lin)
    except Exception as e:
        logger.warning(f"XGBoost Linear race failed: {e}")
        scores['xgb_linear'] = 0.0
    
    # --- Model 5: CatBoost ---
    try:
        model_cat = catboost.CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            class_weights={0: 1.0, 1: 3.0},
            verbose=False,
            random_seed=random_state,
            thread_count=1,
        )
        
        model_cat.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=30,
        )
        
        preds_cat = model_cat.predict_proba(X_val)[:, 1]
        scores['catboost'] = roc_auc_score(y_val, preds_cat)
    except Exception as e:
        logger.warning(f"CatBoost race failed: {e}")
        scores['catboost'] = 0.0

    
    # Pick winner
    winner = max(scores, key=scores.get)
    
    logger.info(f"   Model race scores: LGBM={scores.get('lgbm', 0):.4f}, LGBM_Linear={scores.get('lgbm_linear', 0):.4f}, XGB={scores.get('xgb', 0):.4f}, XGB_Linear={scores.get('xgb_linear', 0):.4f}, CatBoost={scores.get('catboost', 0):.4f}")
    logger.info(f"   Winner: {winner.upper()}")
    return winner, scores


def _calculate_tree_variance(booster, X) -> np.ndarray:
    """
    Calculate the variance of predictions across all trees in the ensemble (Tree Variation).

    1. Get leaf indices for each sample.
    2. Retrieve leaf values from the model dump.
    3. Look up values for indices.
    4. Compute variance across trees for each sample.
    """
    if booster is None:
        return np.zeros(X.shape[0])

    try:
        # Unwrap CalibratedClassifierCV if necessary
        if hasattr(booster, 'calibrated_classifiers_'):
            if len(booster.calibrated_classifiers_) > 0:
                # Use the first base estimator for structure analysis
                booster = booster.calibrated_classifiers_[0].base_estimator

        # Unwrap LGBMClassifier / XGBClassifier wrapper to get booster if needed
        raw_booster = None
        if hasattr(booster, 'booster_'):
            raw_booster = booster.booster_
        elif hasattr(booster, 'get_booster'):
            raw_booster = booster.get_booster()
        else:
            raw_booster = booster

        # 1. Get leaf indices: (n_samples, n_trees)
        leaf_indices_raw = None

        # Try raw booster first (LightGBM)
        if hasattr(raw_booster, 'predict'):
            try:
                leaf_indices_raw = raw_booster.predict(X, pred_leaf=True)
            except Exception:
                pass

        # Fallback to wrapper if raw failed
        if leaf_indices_raw is None:
            try:
                leaf_indices_raw = booster.predict(X, pred_leaf=True)
            except Exception:
                pass

        if leaf_indices_raw is None:
            return np.zeros(X.shape[0])
        
        # Ensure 2D (n_samples, n_trees)
        if leaf_indices_raw.ndim == 1:
            # If 1D, it could be (n_samples,) if 1 tree, or (n_trees,) if 1 sample.
            # predict(pred_leaf=True) usually returns (N, T).
            # If 1D, assume it's (N,) for 1 tree.
            leaf_indices = leaf_indices_raw.reshape(-1, 1)
        else:
            leaf_indices = leaf_indices_raw

        # 2. Parse model to get leaf values
        # We need a lookup table: tree_index -> leaf_index -> leaf_value
        model_dump = None
        if hasattr(raw_booster, 'dump_model'):
            model_dump = raw_booster.dump_model()
        elif hasattr(booster, 'dump_model'):
            model_dump = booster.dump_model()

        if model_dump is None:
             return np.zeros(X.shape[0])

        trees = model_dump.get('tree_info', [])

        # Build lookup table: values[tree_idx][leaf_idx] = value
        # Note: leaf indices in predict() output are local to the tree

        # Determine max leaf index to size the array correctly
        # This might be sparse if not all leaves are present, but usually dense 0..num_leaves-1
        max_leaf_idx = 0
        for tree in trees:
            if 'tree_structure' in tree:
                nodes = [tree['tree_structure']]
                while nodes:
                    node = nodes.pop()
                    if 'leaf_index' in node:
                        max_leaf_idx = max(max_leaf_idx, node['leaf_index'])
                    if 'left_child' in node:
                        nodes.append(node['left_child'])
                    if 'right_child' in node:
                        nodes.append(node['right_child'])

        n_trees = len(trees)
        # Create a lookup array (n_trees, max_leaf_idx + 1) filled with NaN
        # Using dictionary might be safer if indices are sparse, but array is faster
        leaf_values_lookup = np.full((n_trees, max_leaf_idx + 1), np.nan)

        for i, tree in enumerate(trees):
            if 'tree_structure' in tree:
                nodes = [tree['tree_structure']]
                while nodes:
                    node = nodes.pop()
                    if 'leaf_index' in node:
                        idx = node['leaf_index']
                        val = node.get('leaf_value', 0.0)
                        if idx <= max_leaf_idx:
                            leaf_values_lookup[i, idx] = val
                    if 'left_child' in node:
                        nodes.append(node['left_child'])
                    if 'right_child' in node:
                        nodes.append(node['right_child'])

        # 3. Vectorized lookup
        # leaf_indices shape: (n_samples, n_trees)
        # We want result shape: (n_samples, n_trees) containing values

        n_samples = leaf_indices.shape[0]
        n_trees_pred = leaf_indices.shape[1]

        # Ensure we don't go out of bounds if predict returns more/less trees than dump
        # (e.g. early stopping)
        limit_trees = min(n_trees, n_trees_pred)

        # Use numpy advanced indexing
        # row indices: broadcast to (n_samples, limit_trees) -> 0..limit_trees-1
        tree_indices = np.arange(limit_trees)

        # Gather values
        # collected_values[sample_i, tree_j] = leaf_values_lookup[tree_j, leaf_indices[sample_i, tree_j]]

        subset_indices = leaf_indices[:, :limit_trees]
        # Clip indices to be safe against weird dump/predict mismatches
        subset_indices = np.clip(subset_indices, 0, max_leaf_idx)

        collected_values = leaf_values_lookup[tree_indices, subset_indices]

        # 4. Calculate Variance
        # collected_values shape: (n_samples, limit_trees)
        # Variance across trees (axis 1)
        variance = np.nanvar(collected_values, axis=1)

        return variance

    except Exception as e:
        logger.warning(f"Failed to calculate tree variance: {e}")
        return np.zeros(X.shape[0])

@dataclass
class GeometryTrial:
    family: str
    params: Dict[str, Any]  # Kappa, Horizon, sl_sigma, alpha, beta, min_ratio
    final_score: float
    learnability: float
    robust_magnitude: float
    stability: float
    balance: float
    raw_metrics: Dict[str, float]
    uuid: str
    model_params: Optional[Dict[str, Any]] = None
    selected_features: Optional[List[str]] = field(default=None)
    race_score: Optional[float] = None

class LabelBasedLayer2:
    """
    Layer 2: Regime-Conditional Geometry Optimization & Meta-Labeling.
    """

    def __init__(
        self,
        transaction_cost: Optional[float] = None,  # round-trip cost
        n_trials: int = 60,
        n_splits: int = 3,
        random_state: int = 42,
        verbose: bool = True,
        force_hpo: bool = False  # Bypass caching when force-hpo is used
    ):
        """
        Initialize Layer 2.

        Args:
            transaction_cost: Trading cost (slippage + fees) per side.
            n_trials: Number of Optuna trials per barrier family.
            n_splits: Number of TimeSeriesSplit folds for ML probes.
            random_state: Seed for reproducibility.
            verbose: Logging verbosity.
            force_hpo: Bypass caching when force-hpo is used.
        """
        if transaction_cost is None:
             try:
                 from src.utils.ml_common.transaction_costs import DEFAULT_TRANSACTION_COST
                 transaction_cost = DEFAULT_TRANSACTION_COST
             except ImportError:
                 transaction_cost = 0.003

        self.transaction_cost = float(transaction_cost)
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.random_state = random_state
        self.verbose = verbose
        self.force_hpo = force_hpo

        # Internal state
        self.selected_geometries: List[GeometryTrial] = []
        self.family_weights: Dict[str, float] = {}

        self._labels_cache: Dict[Any, Tuple[pd.Series, pd.Series, pd.Series, pd.Series]] = {}
        self._signals_cache: Dict[Any, pd.DataFrame] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._global_probe_features: List[str] = []
        self._current_param_bounds: Dict[str, Dict[str, Any]] = {}
        self._primary_signals = None
        self._rfe_stats = []  # Store RFE statistics for reporting
        self._geometry_label_cache: Dict[str, pd.Series] = {}
        self._feature_selection_cache: Dict[str, List[str]] = {}
        self._dataset_token: str = "unset"

        cpu_guess = max(1, (os.cpu_count() or 4) - 1)
        self._parallel_n_jobs = max(1, min(cpu_guess, 4))
        self._parallel_prefer = "threads"

        # Suppress Optuna logging if not verbose
        if not self.verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

    def execute(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._current_config = dict(config or {})
        self._update_parallel_settings_from_config(self._current_config)
        return self.run(df)

    # ------------------------------------------------------------------
    # Internal helpers for caching-heavy steps
    # ------------------------------------------------------------------
    def _make_geometry_label_cache_key(self, family: str, params: Dict[str, Any]) -> str:
        try:
            param_tuple = tuple(sorted((k, float(v)) if isinstance(v, (int, float)) else (k, str(v)) for k, v in (params or {}).items()))
        except Exception:
            param_tuple = tuple(sorted((k, str(v)) for k, v in (params or {}).items()))
        dataset_token = getattr(self, "_dataset_token", "global")
        return f"{dataset_token}|{family}|{hash(param_tuple)}"

    def _get_cached_geometry_labels(
        self,
        df: pd.DataFrame,
        fam_events: pd.DataFrame,
        family: str,
        params: Dict[str, Any],
    ) -> pd.Series:
        dataset_fingerprint = self._fingerprint_dataframe(df)
        cache_key = f"{dataset_fingerprint}::{self._make_geometry_label_cache_key(family, params)}"
        if cache_key in self._geometry_label_cache:
            return self._geometry_label_cache[cache_key]

        lbls, _, _, _, _ = self._compute_dominance_labels(df, fam_events, family=family, **params)
        lbls = pd.to_numeric(lbls, errors='coerce').astype(float).reindex(fam_events.index)
        self._geometry_label_cache[cache_key] = lbls
        return lbls

    def _make_feature_cache_key(
        self,
        feature_cols: List[str],
        n_rows: int,
        target_n: int,
        extra_token: Optional[str] = None,
    ) -> str:
        cols_token = hash(tuple(feature_cols))
        dataset_token = getattr(self, "_dataset_token", "global")
        return f"{dataset_token}|{cols_token}_{n_rows}_{target_n}_{extra_token or ''}"

    def _update_parallel_settings_from_config(self, cfg: Optional[Dict[str, Any]]) -> None:
        if not isinstance(cfg, dict):
            return
        prefer = str(cfg.get('layer2_parallel_prefer', self._parallel_prefer)).lower()
        if prefer not in ('threads', 'processes'):
            prefer = 'threads'
        self._parallel_prefer = prefer
        try:
            n_jobs = int(cfg.get('layer2_parallel_n_jobs', self._parallel_n_jobs))
        except Exception:
            n_jobs = self._parallel_n_jobs
        if n_jobs == -1:
            n_jobs = max(1, (os.cpu_count() or 4) - 1)
        self._parallel_n_jobs = max(1, n_jobs)

    def _get_parallel_kwargs(self) -> Dict[str, Any]:
        kwargs = {'n_jobs': int(self._parallel_n_jobs)}
        prefer = getattr(self, '_parallel_prefer', 'threads')
        if prefer in ('threads', 'processes'):
            kwargs['prefer'] = prefer
        return kwargs

    def _run_parallel_with_timeout(
        self,
        task_fn: Callable[[], Any],
        context: str,
        timeout_seconds: Optional[float] = None
    ) -> Any:
        cfg = getattr(self, "_current_config", {}) or {}
        if timeout_seconds is None:
            try:
                timeout_seconds = float(cfg.get('layer2_parallel_timeout_seconds', 900.0))
            except Exception:
                timeout_seconds = 900.0
        if (timeout_seconds is None) or (not np.isfinite(timeout_seconds)) or timeout_seconds <= 0:
            timeout_seconds = 900.0

        start_time = time.time()
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(task_fn)
                result = future.result(timeout=timeout_seconds)
        except FuturesTimeoutError:
            elapsed = time.time() - start_time
            error_msg = (
                f"Layer2 parallel section '{context}' timed out after "
                f"{elapsed:.1f}s (limit={timeout_seconds:.1f}s)."
            )
            logger.error(error_msg)
            raise TimeoutError(error_msg)
        except Exception as exc:
            elapsed = time.time() - start_time
            logger.error(f"Layer2 parallel section '{context}' failed after {elapsed:.1f}s: {exc}")
            raise

        elapsed = time.time() - start_time
        logger.info(f"Layer2 parallel section '{context}' completed in {elapsed:.2f}s.")
        return result

    def _fingerprint_dataframe(self, df: pd.DataFrame) -> str:
        if df is None or getattr(df, "empty", True):
            return "empty_df"
        try:
            idx = df.index
            first = idx[0] if len(idx) else "none"
            last = idx[-1] if len(idx) else "none"
            marker_cols = [c for c in ["close", "open", "volume"] if c in df.columns]
            marker_vals = tuple(round(float(df[c].iloc[0]), 6) if len(df[c]) else 0.0 for c in marker_cols) if marker_cols else ()
            return f"{len(df)}_{hash((first, last, marker_cols, marker_vals))}"
        except Exception:
            return f"{len(df)}_{hash(tuple(idx[:5])) if len(idx) else 0}"

    def _hash_series_signature(self, series: Optional[pd.Series]) -> str:
        if series is None or len(series) == 0:
            return "empty_series"
        try:
            sig = (
                int(len(series)),
                float(np.nanmean(series)),
                float(np.nanstd(series)),
                float(series.iloc[0]),
                float(series.iloc[-1]),
            )
        except Exception:
            sig = (len(series), 0.0, 0.0, 0.0, 0.0)
        return str(hash(sig))

    def _maybe_sample_indices(self, index: pd.Index, max_rows: int) -> pd.Index:
        if max_rows <= 0 or len(index) <= max_rows:
            return index
        rng = np.random.default_rng(self.random_state)
        sample_idx = rng.choice(len(index), size=max_rows, replace=False)
        return index.take(np.sort(sample_idx))

    def _passes_surrogate_gate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[pd.Series],
        cfg: Dict[str, Any],
    ) -> bool:
        try:
            gate_min_samples = int(cfg.get('layer2_surrogate_min_samples', 400))
        except Exception:
            gate_min_samples = 400
        if len(y) < gate_min_samples:
            return True

        try:
            gate_max_rows = int(cfg.get('layer2_surrogate_max_rows', 2000))
        except Exception:
            gate_max_rows = 2000
        idx = self._maybe_sample_indices(X.index, gate_max_rows)
        X_gate = X.loc[idx]
        y_gate = y.loc[idx]
        if sample_weight is not None:
            sw_gate = sample_weight.reindex(idx).fillna(1.0)
        else:
            sw_gate = None

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_gate)

        tscv = TimeSeriesSplit(n_splits=2)
        aucs = []
        clf = LogisticRegression(
            max_iter=200,
            solver='lbfgs',
            n_jobs=1 if hasattr(LogisticRegression, 'n_jobs') else None,
        )
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y_gate.iloc[train_idx], y_gate.iloc[val_idx]
            if y_train.nunique() < 2 or y_val.nunique() < 2:
                continue
            if sw_gate is not None:
                clf.fit(X_train, y_train, sample_weight=sw_gate.iloc[train_idx])
            else:
                clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_val)[:, 1]
            aucs.append(roc_auc_score(y_val, probs))

        if not aucs:
            return True
        avg_auc = float(np.nanmean(aucs))
        try:
            gate_threshold = float(cfg.get('layer2_surrogate_auc_threshold', 0.54))
        except Exception:
            gate_threshold = 0.54
        return avg_auc >= gate_threshold

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute the Layer 2 pipeline. Orquestrates independent steps.
        """
        tprint_info("Starting Layer 2 Pipeline...")

        # 1. Prepare
        df, events_df, X_events, global_probe_features = self.prepare_data_and_events(df)
        if events_df.empty:
            tprint_warning("No events generated in Layer 2. Skipping.")
            return {}

        # 2. Train (Production)
        production_geometries, production_selected_features = self.optimize_production_geometries(
            df, events_df, global_probe_features=global_probe_features
        )

        # 3. Validate (OOF)
        oof_results = self.run_oof_analytics(
            df, events_df, production_geometries,
            global_probe_features=global_probe_features,
            production_selected_features=production_selected_features
        )

        # 4. Report
        self.generate_reports(df, events_df, production_geometries, oof_results)

        # Combine
        return {
            **oof_results,
            "events_df": events_df,
            "selected_trials": [asdict(t) for t in production_geometries],
            "production_selected_features": list(getattr(self, '_production_selected_features', []) or []),
        }

    def prepare_data_and_events(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
        """Step 1: Stateless data preparation and event generation."""
        tprint_info(">>> Layer 2: Step 1 - Prepare Data and Events...")

        self._labels_cache = {}
        self._signals_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._global_probe_features = []
        self._current_param_bounds = {}
        self._primary_signals = None
        self._rfe_stats = []
        self._geometry_label_cache = {}
        self._feature_selection_cache = {}
        self._dataset_token = self._fingerprint_dataframe(df)

        df = self._validate_inputs(df)
        df = self._precompute_geometry_base_features(df)
        events_df = self._generate_events(df)

        if not events_df.empty:
            events_df['family'] = self._assign_barrier_families(events_df)
            try:
                # Use Probe Basis set for validation
                X_probe_events = self._build_geometry_independent_event_features(df, events_df, mode='probe')
                # No selection needed for basis set
                self._global_probe_features = list(X_probe_events.columns)
            except Exception:
                self._global_probe_features = []
                X_probe_events = pd.DataFrame(index=events_df.index)
        else:
            X_probe_events = pd.DataFrame()

        return df, events_df, X_probe_events, self._global_probe_features

    def optimize_production_geometries(
        self,
        df: pd.DataFrame,
        events_df: pd.DataFrame,
        global_probe_features: Optional[List[str]] = None
    ) -> Tuple[List[GeometryTrial], List[str]]:
        """Step 2: Full Optimization (Production Artifacts)."""
        tprint_info(">>> Layer 2: Step 2 - Optimize Production Geometries...")

        if global_probe_features:
            self._global_probe_features = list(global_probe_features)

        if events_df.empty:
            return [], []

        full_results = self._optimize_families(df, events_df)

        try:
            full_counts = {str(k): int(len(v)) for k, v in (full_results or {}).items()}
            tprint_info(f"Layer2 Full Optimization: extracted_trials_per_family={full_counts}")
        except Exception:
            pass

        production_geometries = self._select_best_geometries(df, events_df, full_results, require_passed=True)
        if not production_geometries:
            fallback_enabled = True
            try:
                cfg_prod = getattr(self, "_current_config", {})
                if isinstance(cfg_prod, dict):
                    fallback_enabled = bool(cfg_prod.get('layer2_production_fallback_enabled', True))
            except Exception:
                pass
            if fallback_enabled:
                production_geometries = self._select_best_geometries(df, events_df, full_results, require_passed=False)

        # Enforce Max 10 Geometries
        # Tier 2: Model Race Screening (Filter 12 proposals down to Top 2 per Horizon)
        if production_geometries:
            tprint_info(f">>> Layer 2: Tier 2 Screening - Running Model Race on {len(production_geometries)} proposals...")
            for i, g in enumerate(production_geometries):
                tprint_info(f"    Race probing geometry {g.uuid} ({i+1}/{len(production_geometries)})...")
                # Run only the race part
                self._tune_geometry_model_params(df, events_df, g, skip_hpo=True)

            # Group by horizon and select top 2 per horizon
            by_horizon = {}
            for g in production_geometries:
                # Horizon is in params
                h = g.params.get('horizon', 0)
                if h not in by_horizon: by_horizon[h] = []
                by_horizon[h].append(g)

            surviving_tier2 = []
            for h in sorted(by_horizon.keys()):
                # Sort by race_score descending
                geoms_h = sorted(by_horizon[h], key=lambda x: getattr(x, 'race_score', 0) or 0, reverse=True)
                # Keep top 2 from this horizon
                kept_h = geoms_h[:2]
                for g in kept_h:
                    score = getattr(g, 'race_score', 0) or 0
                    if score >= 0.52:
                        surviving_tier2.append(g)
                    else:
                        tprint_warning(f"    Geometry {g.uuid} (H={h}) rejected: Race Score {score:.4f} < 0.52")
            
            production_geometries = surviving_tier2
            tprint_info(f"Tier 2 Screening Complete: {len(production_geometries)} geometries proceeding to Tier 3 Execution.")

        # Tier 3: Full Execution (HPO, RFE, Regime Extraction)
        if production_geometries:
            tprint_info(">>> Layer 2: Tier 3 Execution - Full HPO & Feature Selection for Survivors...")

            # Build shared data for RFE (Full Feature Set)
            X_events_full = self._build_geometry_independent_event_features(df, events_df, mode='full')
            w_l1_prod = self._get_target_sample_weight_for_events(df, events_df)
            vol_prod = df['volatility_1d'].reindex(events_df.index).fillna(0.0)

            surviving_tier3 = []
            for i, g in enumerate(production_geometries):
                tprint_info(f"    Tier 3 Processing {g.uuid} ({i+1}/{len(production_geometries)})...")

                # 1. Full HPO on winning model type
                best_params = self._tune_geometry_model_params(df, events_df, g, skip_hpo=False)
                if not best_params:
                    tprint_warning(f"    Geometry {g.uuid} failed HPO phase. Skipping.")
                    continue

                g.model_params = best_params
                tprint_info(f"    Found production params for {g.uuid}: {best_params}")
                surviving_tier3.append(g)

                # 2. Per-Geometry Feature Selection (Titan RFE)
                try:
                    cfg_prod_fs = getattr(self, "_current_config", {})
                    if not isinstance(cfg_prod_fs, dict): cfg_prod_fs = {}
                    enable_prod_fs = bool(cfg_prod_fs.get('layer2_production_supervised_feature_selection_enabled', True))

                    if enable_prod_fs:
                        tprint_info(f"    Selecting features for {g.uuid}...")
                        fam = str(getattr(g, 'family', ''))
                        fam_events = events_df[events_df['family'] == fam]

                        if not fam_events.empty:
                            lbls_local = self._get_cached_geometry_labels(df, fam_events, fam, g.params)
                            lbls = pd.to_numeric(lbls_local, errors='coerce').astype(float).reindex(events_df.index)

                            valid_idx = lbls.dropna().index
                            if len(valid_idx) > 50:
                                y_target = lbls.loc[valid_idx]
                                X_target = X_events_full.reindex(valid_idx).fillna(0.0)
                                w_target = w_l1_prod.reindex(valid_idx) if w_l1_prod is not None else None
                                vol_target = vol_prod.reindex(valid_idx)

                                initial_feat_count = X_target.shape[1]
                                tprint_info(f"    Starting Titan RFE with {initial_feat_count} features...")

                                sel_feats = self._select_supervised_features_for_events(
                                    X_target, y_target, w_target, volatility_series=vol_target
                                )

                                if sel_feats:
                                    g.selected_features = list(sel_feats)
                                    tprint_success(f"    Selected {len(sel_feats)} features for {g.uuid}")
                                    
                                    # Regime Leaf Feature Extraction
                                    try:
                                        self._extract_and_save_regime_leaves(df, g)
                                    except Exception as re_err:
                                        tprint_warning(f"    Regime leaf extraction failed for {g.uuid}: {re_err}")
                except Exception as e:
                    tprint_warning(f"    Tier 3 Feature selection failed for {g.uuid}: {e}")

            # Update to surviving set
            production_geometries = surviving_tier3

        # FAST-FAIL
        if not production_geometries:
            tprint_error(
                "Layer2 CRITICAL: Zero production geometries passed all gates! "
                "Pipeline cannot continue. Consider relaxing gates via config."
            )
            raise ValueError(
                "Layer2 failed: No production geometries passed validation gates."
            )

        self.selected_geometries = production_geometries

        # Production Feature Selection (Optional)
        self._run_production_feature_selection(df, events_df, production_geometries)

        return production_geometries, list(getattr(self, '_production_selected_features', []) or [])

    def _run_production_feature_selection(self, df, events_df, production_geometries):
        try:
            self._production_selected_features = []
            cfg_prod_fs = getattr(self, "_current_config", {})
            if not isinstance(cfg_prod_fs, dict): cfg_prod_fs = {}
            enable_prod_fs = bool(cfg_prod_fs.get('layer2_production_supervised_feature_selection_enabled', True))

            if enable_prod_fs:
                X_events_full = self._build_geometry_independent_event_features(df, events_df)
                y_fs_prod = self._aggregate_geometry_labels_for_feature_selection(df, events_df, production_geometries)
                w_l1_prod = self._get_target_sample_weight_for_events(df, events_df)
                vol_prod = df['volatility_1d'].reindex(events_df.index).fillna(0.0)
                prod_feats = self._select_supervised_features_for_events(X_events_full, y_fs_prod, w_l1_prod, volatility_series=vol_prod)
                if prod_feats:
                    self._production_selected_features = list(prod_feats)
                    # Persist logic moved to generate_reports if needed, or done here
                    try:
                        outcomes_dir = Path("outcomes")
                        outcomes_dir.mkdir(parents=True, exist_ok=True)
                        ts = getattr(self, "_current_config", {}).get("run_timestamp") or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                        symbol = getattr(self, "_current_config", {}).get("symbol", "")
                        timeframe = getattr(self, "_current_config", {}).get("timeframe", "")
                        pd.Series(self._production_selected_features, name='feature').to_csv(
                            outcomes_dir / f"layer2_selected_features_supervised_{symbol}_{timeframe}_{ts}.csv",
                            index=False,
                        )
                    except Exception:
                        pass
        except Exception as e:
            tprint_error(f"Error in production feature selection: {e}")

    def _extract_and_save_regime_leaves(self, df: pd.DataFrame, geometry: GeometryTrial) -> None:
        """
        Extract regime leaf features for a specific geometry and save as an artifact.
        
        This logic is moved from Layer 3 to Layer 2 to ensure geometry-level 
        completeness before meta-labeling. Only the most significant leaves are kept.
        """
        try:
            cfg = getattr(self, "_current_config", {})
            symbol = cfg.get("symbol", "UNKNOWN")
            tf = cfg.get("timeframe", "15m")
            ts = cfg.get("run_timestamp") or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            market_data = df[['open', 'high', 'low', 'close', 'volume']].copy()
            
            # Geometry-specific horizon scaling (similar to L3 logic)
            # Default H=24 (6h in 15m), scale relative to that
            geom_params = geometry.params or {}
            h = float(geom_params.get('horizon', 24))
            scale = h / 24.0
            scaled_horizons = [max(4, min(96, int(base_h * scale))) for base_h in [8, 16, 24]]
            
            # Extractor Configuration (Production Grade)
            extractor_cfg = {
                "targets": {
                    "macro_trend_horizons": scaled_horizons,
                    "trend_efficiency_horizons": scaled_horizons,
                    "trend_efficiency_window": int(h),
                },
                "inputs": {"input_source": "ohlcv_only"},
                "onehot": {"enabled": True},
                "interaction_feature": {"enabled": True, "include_base": False},
                # Importance-based Pruning (De Prado 1.4: Signal vs. Noise)
                "leaf_pruning": {
                    "min_support": 0.05,    # Leaf must cover at least 5% of samples
                    "max_support": 0.35,    # Leaf shouldn't be too dominant (high entropy)
                    "min_effect_z": 1.5,    # Mean outcome in leaf must be >1.5 StdDev from global mean
                    "max_pairs": 48         # Cap total leaf features per geometry
                },
                "reporting": {"enabled": False},
                "walk_forward": {"mode": "cross_fit", "cross_fit": {"n_splits": 4}}
            }
            
            tprint_info(f"    Extracting regime leaves for {geometry.uuid} (H={h})...")
            
            # Call the extractor - use ohlcv_only input source which builds features internally
            # Note: X must have the same index as market_data for alignment
            rl_features = extract_regime_leaf_onehot_features(
                X=market_data.copy(),  # Pass market_data as X with input_source='ohlcv_only'
                market_data=market_data,
                config=extractor_cfg,
                random_state=self.random_state,
                verbose=False
            )
            
            if rl_features is not None and not rl_features.empty:
                # Align and Clean
                rl_features = rl_features.reindex(df.index).fillna(0.0)
                
                # Save artifact
                outcomes_dir = Path("outcomes") / "regime_leaves"
                outcomes_dir.mkdir(parents=True, exist_ok=True)
                
                artifact_name = f"regime_leaves_{geometry.uuid}_{symbol}_{tf}_{ts}.parquet"
                artifact_path = outcomes_dir / artifact_name
                
                rl_features.to_parquet(artifact_path)
                tprint_success(f"    Saved {rl_features.shape[1]} regime leaves to {artifact_name}")
                
                # Attach metadata to geometry object for L3 awareness
                geometry.raw_metrics['regime_leaves_path'] = str(artifact_path)
                geometry.raw_metrics['n_regime_leaves'] = int(rl_features.shape[1])
            else:
                tprint_warning(f"    No important regime leaves found for {geometry.uuid}")
                
        except Exception as e:
            logger.error(f"Failed to extract regime leaves for {geometry.uuid}: {e}", exc_info=True)
            raise

    def run_oof_analytics(
        self,
        df: pd.DataFrame,
        events_df: pd.DataFrame,
        production_geometries: Optional[List[GeometryTrial]] = None,
        global_probe_features: Optional[List[str]] = None,
        production_selected_features: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Step 3: OOF Optimization (Analytics Artifacts)."""
        tprint_info(">>> Layer 2: Step 3 - Running OOF Optimization (Analytics)...")

        if global_probe_features:
            self._global_probe_features = list(global_probe_features)

        if production_selected_features:
            self._production_selected_features = list(production_selected_features)

        # Initialize storage for OOF results
        indices = events_df.index
        oof_scores = pd.Series(np.nan, index=indices)
        oof_labels = pd.Series(np.nan, index=indices)
        oof_confidence = pd.Series(np.nan, index=indices)
        oof_returns = pd.Series(np.nan, index=indices)
        oof_weights = pd.Series(np.nan, index=indices)
        oof_mean_probs = pd.Series(np.nan, index=indices)

        # Storage for Tree Diagnostics
        self._all_tree_stats = [] # Store on instance for report usage

        # Initialize storage for individual geometries
        oof_geo_preds = {}
        oof_geo_vars = {} # Store variances

        try:
            cfg_oof = getattr(self, "_current_config", {})
            if not isinstance(cfg_oof, dict):
                cfg_oof = {}
        except Exception:
            cfg_oof = {}
        self._update_parallel_settings_from_config(cfg_oof)

        try:
            n_oof_splits = int(cfg_oof.get("layer2_oof_splits", 3))
        except Exception:
            n_oof_splits = 3
        n_oof_splits = int(max(2, min(n_oof_splits, int(len(df)))))

        try:
            purge_bars = int(cfg_oof.get("layer2_oof_purge_bars", 0))
        except Exception:
            purge_bars = 0
        if purge_bars <= 0:
            try:
                purge_bars = int(cfg_oof.get("layer3_max_lookahead_bars", 100))
            except Exception:
                purge_bars = 100
        purge_bars = int(max(0, purge_bars))

        n_samples = int(len(df))
        fold_sizes = np.full(n_oof_splits, n_samples // n_oof_splits, dtype=int)
        fold_sizes[: n_samples % n_oof_splits] += 1
        current = 0
        folds = []
        for fold_size in fold_sizes:
            start, stop = (current, current + int(fold_size))
            folds.append((int(start), int(stop)))
            current = int(stop)

        fold_idx = 0
        for (val_start, val_stop) in folds:
            fold_idx += 1
            test_idx = np.arange(int(val_start), int(val_stop))
            
            # Purged K-Fold: Use all data EXCEPT test fold and purge windows
            train_idx_list = []
            if val_start > purge_bars:
                train_idx_list.extend(range(0, int(val_start - purge_bars)))
            if val_stop + purge_bars < n_samples:
                train_idx_list.extend(range(int(val_stop + purge_bars), n_samples))
            train_idx = np.array(train_idx_list, dtype=int)

            tprint_info(f"   > Processing Fold {fold_idx}/{int(len(folds))}...")

            try:
                t0 = str(df.index[int(val_start)]) if int(val_start) < len(df.index) else ""
                t1 = str(df.index[int(val_stop - 1)]) if int(val_stop - 1) < len(df.index) else ""
                tprint_info(
                    f"Layer2 OOF Fold {fold_idx}: purged k-fold val_start={val_start}, val_stop={val_stop}, "
                    f"purge_bars={purge_bars}, test_start_time={t0}, test_end_time={t1}, n_train={len(train_idx)}"
                )
            except Exception:
                pass

            # Create Train Slice
            df_train = df.iloc[train_idx]

            # Subset events
            events_train = events_df.loc[events_df.index.intersection(df_train.index)]
            events_test = events_df.loc[events_df.index.intersection(df.index[test_idx])]

            try:
                logger.info(
                    f"Layer2 OOF Fold {fold_idx}: n_train_bars={int(len(df_train))}, n_train_events={int(len(events_train))}, "
                    f"n_test_events={int(len(events_test))}"
                )
            except Exception:
                pass

            if events_train.empty:
                logger.warning(f"Fold {fold_idx}: No training events. Skipping.")
                continue

            # Optimize on Train
            fold_results = self._optimize_families(df_train, events_train)
            if not fold_results:
                continue

            try:
                fold_counts = {str(k): int(len(v)) for k, v in (fold_results or {}).items()}
                logger.info(f"Layer2 OOF Fold {fold_idx}: extracted_trials_per_family={fold_counts}")
            except Exception:
                pass

            fold_geometries = self._select_best_geometries(df_train, events_train, fold_results, require_passed=False)
            if not fold_geometries:
                continue

            # Enforce Max 10 Geometries (Fold)
            if len(fold_geometries) > 10:
                fold_geometries = fold_geometries[:10]

            try:
                by_fam_fold: Dict[str, int] = {}
                for g in list(fold_geometries or []):
                    try:
                        by_fam_fold[str(getattr(g, 'family', ''))] = by_fam_fold.get(str(getattr(g, 'family', '')), 0) + 1
                    except Exception:
                        continue
                logger.info(
                    f"Layer2 OOF Fold {fold_idx}: selected_geometries={int(len(fold_geometries or []))}, by_family={by_fam_fold}"
                )
            except Exception:
                pass

            # Rename/Standardize Geometries for consistent channels
            geo_by_fam = {}
            for g in fold_geometries:
                geo_by_fam.setdefault(g.family, []).append(g)

            standardized_geos = []
            for fam, geos in geo_by_fam.items():
                # Sort by final_score descending
                geos_sorted = sorted(geos, key=lambda x: x.final_score, reverse=True)
                for rank, g in enumerate(geos_sorted):
                    # Assign standardized UUID
                    g_copy = copy.deepcopy(g)
                    g_copy.uuid = f"{fam}_Rank{rank}"
                    standardized_geos.append(g_copy)

            # OOF Fix: fold-local probe feature selection (train slice only)
            fold_probe_features: List[str] = []
            X_train_events = None
            X_test_events = None

            try:
                # OOF Models are production-like, so they use the full feature set (or selected subset)
                # But here we are building the full feature set to potentially select from or train on.
                X_train_events_full = self._build_geometry_independent_event_features(df_train, events_train, mode='full')
                # If using selected features from production (monolithic), we might not need probe features here
                # But if we are re-selecting per fold, we need full features.
                fold_probe_features = self._select_global_probe_features(X_train_events_full)
            except Exception:
                X_train_events_full = None
                fold_probe_features = []

            try:
                cfg_fs = getattr(self, '_current_config', {})
                if not isinstance(cfg_fs, dict):
                    cfg_fs = {}
            except Exception:
                cfg_fs = {}

            try:
                use_supervised_fs = bool(cfg_fs.get('layer2_supervised_feature_selection_enabled', True))
            except Exception:
                use_supervised_fs = True

            # Per-Geometry Selection (Fold-Local)
            if use_supervised_fs and X_train_events_full is not None and not getattr(X_train_events_full, 'empty', True):
                try:
                    w_l1 = self._get_target_sample_weight_for_events(df_train, events_train)
                    vol_train = df_train['volatility_1d'].reindex(events_train.index).fillna(0.0)

                    for g in standardized_geos:
                         try:
                             fam = str(getattr(g, 'family', ''))
                             fam_events = events_train[events_train['family'] == fam]
                             if fam_events.empty: continue

                             lbls, _, _, _, _ = self._compute_dominance_labels(df_train, fam_events, family=fam, **g.params)
                             lbls = pd.to_numeric(lbls, errors='coerce').astype(float).reindex(events_train.index)
                             valid_idx = lbls.dropna().index
                             if len(valid_idx) > 50:
                                 y_t = lbls.loc[valid_idx]
                                 X_t = X_train_events_full.reindex(valid_idx).fillna(0.0)
                                 w_t = w_l1.reindex(valid_idx) if w_l1 is not None else None
                                 v_t = vol_train.reindex(valid_idx)
                                 sel = self._select_supervised_features_for_events(X_t, y_t, w_t, volatility_series=v_t)
                                 if sel:
                                     g.selected_features = list(sel)
                         except Exception:
                             pass
                except Exception:
                    pass

            feature_cols_for_models: List[str] = []
            if X_train_events_full is not None and not getattr(X_train_events_full, 'empty', True):
                if fold_probe_features:
                    # IMPORTANT: keep a stable column list for BOTH train and test.
                    # Reindex fills missing columns with 0.0 so shapes always match.
                    feature_cols_for_models = [str(c) for c in list(fold_probe_features)]
                else:
                    feature_cols_for_models = [str(c) for c in list(X_train_events_full.columns)]

                X_train_events = X_train_events_full.reindex(columns=feature_cols_for_models).fillna(0.0)

            # Train models on Train Split
            trained_models = None
            if X_train_events is not None and not getattr(X_train_events, 'empty', True):
                try:
                    trained_models = self._train_geometry_models(
                        df=df_train,
                        X_events=X_train_events,
                        events_df=events_train,
                        geometries=standardized_geos,
                        X_events_full=X_train_events_full
                    )
                except Exception:
                    trained_models = None

            # Collect Tree Diagnostics
            if trained_models:
                for uuid, model in trained_models.items():
                    if model is not None:
                        try:
                            stats = self._extract_tree_diagnostics(model)
                            self._all_tree_stats.append(stats)
                        except Exception:
                            pass

            # Predict on Test (Bagged Labeling)
            if not events_test.empty:
                try:
                    max_h = int(
                        max(
                            int(g.params.get("horizon", 0))
                            for g in standardized_geos
                            if isinstance(g, GeometryTrial) and isinstance(getattr(g, "params", None), dict)
                        )
                    )
                except Exception:
                    max_h = 0

                try:
                    lookahead_scale = float(getattr(self, "_current_config", {}).get("layer2_oof_lookahead_scale", 2.0))
                except Exception:
                    lookahead_scale = 2.0
                if (not np.isfinite(lookahead_scale)) or float(lookahead_scale) <= 0.0:
                    lookahead_scale = 2.0

                try:
                    fixed_lookahead = getattr(self, "_current_config", {}).get("layer2_oof_lookahead_bars")
                    fixed_lookahead = int(fixed_lookahead) if fixed_lookahead is not None else None
                except Exception:
                    fixed_lookahead = None

                if fixed_lookahead is not None and fixed_lookahead > 0:
                    lookahead_bars = int(fixed_lookahead)
                else:
                    lookahead_bars = int(np.ceil(float(max_h) * float(lookahead_scale))) + 1
                    lookahead_bars = int(max(1, lookahead_bars))

                try:
                    test_end_pos = int(np.max(np.asarray(test_idx, dtype=int)))
                except Exception:
                    test_end_pos = int(test_idx[-1])
                label_end_pos = int(min(len(df) - 1, test_end_pos + lookahead_bars))
                df_label = df.iloc[: label_end_pos + 1]

                try:
                    # Test set also uses full features for OOF predictions
                    X_test_events_full = self._build_geometry_independent_event_features(df_label, events_test, mode='full')
                    if X_test_events_full is not None and not getattr(X_test_events_full, 'empty', True):
                        cols = feature_cols_for_models or [str(c) for c in list(X_test_events_full.columns)]
                        X_test_events = X_test_events_full.reindex(columns=cols).fillna(0.0)
                except Exception:
                    X_test_events = None

                fold_output = self._bagged_labeling(
                    df_label, 
                    events_test, 
                    standardized_geos,
                    trained_models=trained_models,
                    X_events=X_test_events
                )

                try:
                    lbl = fold_output.get('l2_label')
                    n_lbl = int(lbl.notna().sum()) if isinstance(lbl, pd.Series) else 0
                    n_geo = int(len(fold_output.get('individual_geometries') or {}))
                    logger.info(
                        f"Layer2 OOF Fold {fold_idx}: labeled_events={n_lbl}/{int(len(events_test))}, geometry_channels={n_geo}"
                    )
                except Exception:
                    pass

                # Assign to OOF arrays
                target_idx = events_test.index

                oof_scores.loc[target_idx] = fold_output.get('l2_score', fold_output.get('oof_labels')).reindex(target_idx)
                oof_labels.loc[target_idx] = fold_output.get('l2_label', pd.Series(np.nan, index=target_idx)).reindex(target_idx)
                oof_confidence.loc[target_idx] = fold_output.get('l2_confidence', pd.Series(np.nan, index=target_idx)).reindex(target_idx)
                oof_returns.loc[target_idx] = fold_output['oof_returns'].reindex(target_idx)
                oof_weights.loc[target_idx] = fold_output['weights'].reindex(target_idx)

                # Capture diagnostics mean prob
                if 'diagnostics' in fold_output and 'mean_consensus_prob' in fold_output['diagnostics']:
                    mp = fold_output['diagnostics']['mean_consensus_prob']
                    if isinstance(mp, pd.Series):
                        oof_mean_probs.loc[target_idx] = mp.reindex(target_idx)

                # Assign individual geometry preds and variances
                for uuid, series in fold_output['individual_geometries'].items():
                    if uuid in oof_geo_preds:
                        oof_geo_preds[uuid].loc[target_idx] = series.reindex(target_idx)

                for uuid, series in fold_output['individual_variances'].items():
                    if uuid in oof_geo_vars:
                        oof_geo_vars[uuid].loc[target_idx] = series.reindex(target_idx)

        final_geo_preds = {k: v for k, v in oof_geo_preds.items() if v.notna().any()}
        final_geo_vars = {k: v for k, v in oof_geo_vars.items() if v.notna().any()}

        # We need composite weights for Layer 3, and quality weights
        # Recalculate global quality weights on the final OOF composite return
        try:
            c_ret = oof_returns.fillna(0.0)
            c_vol = df['volatility_1d'].reindex(oof_returns.index).ffill().fillna(0.0)
            safe_v = np.where(c_vol > 1e-9, c_vol, 1e-9)
            z = c_ret / safe_v
            sig = 1.0 / (1.0 + np.exp(-1.0 * z))
            quality_weights = pd.Series(0.5 + 1.5 * sig, index=oof_returns.index)
        except Exception:
            quality_weights = pd.Series(1.0, index=oof_returns.index)

        # Calculate Diagnostics
        n_base = int(len(events_df))
        n_bagged = int((oof_labels == 1.0).sum())
        inflation_ratio = n_bagged / max(1, n_base)

        return {
            "oof_labels": oof_scores,
            "oof_returns": oof_returns,
            "weights": oof_weights,
            "l2_score": oof_scores,
            "l2_label": oof_labels,
            "l2_confidence": oof_confidence,
            "individual_geometries": final_geo_preds,
            "individual_variances": final_geo_vars,
            "quality_weights": quality_weights,
            "tree_stats": self._all_tree_stats,
            "diagnostics": {
                "signal_inflation_ratio": inflation_ratio,
                "n_bagged_signals": n_bagged,
                "n_base_events": n_base,
                "mean_consensus_prob": oof_mean_probs
            }
        }

    def _calculate_ranking_metrics(self, y_true: pd.Series, y_score: pd.Series) -> Dict[str, float]:
        """
        Calculate Lift, Precision, and Recall at various top-k thresholds.
        """
        metrics = {}
        try:
            y_true_arr = pd.to_numeric(y_true, errors='coerce').fillna(0.0).values
            y_score_arr = pd.to_numeric(y_score, errors='coerce').fillna(0.0).values

            mask = np.isfinite(y_true_arr) & np.isfinite(y_score_arr)
            y_true_clean = y_true_arr[mask]
            y_score_clean = y_score_arr[mask]

            n_total = len(y_true_clean)
            if n_total < 10:
                return {}

            n_pos = np.sum(y_true_clean)
            global_pos_rate = n_pos / n_total if n_total > 0 else 0.0

            # Sort descending by score
            sorted_indices = np.argsort(y_score_clean)[::-1]
            y_true_sorted = y_true_clean[sorted_indices]

            # K-levels (deciles + top 5%)
            k_levels = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

            for k in k_levels:
                cutoff_idx = int(n_total * k)
                if cutoff_idx < 1: continue

                # Top K predictions
                top_k_true = y_true_sorted[:cutoff_idx]

                # Metrics
                tp = np.sum(top_k_true)
                prec_at_k = tp / cutoff_idx
                recall_at_k = tp / n_pos if n_pos > 0 else 0.0
                lift_at_k = prec_at_k / global_pos_rate if global_pos_rate > 0 else 0.0

                metrics[f"Lift@{int(k*100)}"] = float(lift_at_k)
                metrics[f"Precision@{int(k*100)}"] = float(prec_at_k)
                metrics[f"Recall@{int(k*100)}"] = float(recall_at_k)

        except Exception as e:
            logger.warning(f"Failed to calculate ranking metrics: {e}")

        return metrics

    def _calculate_portfolio_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate financial metrics: Cumulative Gain, Max Drawdown.
        Assumes returns are per-event or period returns.
        """
        metrics = {}
        try:
            # Drop NaNs
            r = pd.to_numeric(returns, errors='coerce').dropna()
            if r.empty:
                return {}

            # Equity Curve (Simple sum for log returns or cumprod for simple)
            # Assuming simple returns here
            equity = (1 + r).cumprod()

            # Total Return / Cumulative Gain
            total_ret = equity.iloc[-1] - 1.0 if not equity.empty else 0.0
            metrics["Cumulative_Gain"] = float(total_ret)

            # Max Drawdown
            running_max = equity.cummax()
            drawdown = (equity - running_max) / running_max
            max_dd = drawdown.min()
            metrics["Max_Drawdown"] = float(max_dd)

            # Win Rate & Expectancy
            wins = r > 0
            win_rate = wins.mean()
            avg_win = r[wins].mean() if wins.any() else 0.0
            avg_loss = r[~wins].mean() if (~wins).any() else 0.0

            metrics["Win_Rate"] = float(win_rate)
            metrics["Avg_Win"] = float(avg_win)
            metrics["Avg_Loss"] = float(avg_loss)

            if abs(avg_loss) > 1e-9:
                metrics["Profit_Factor"] = float((avg_win * wins.sum()) / (abs(avg_loss) * (~wins).sum()))
            else:
                metrics["Profit_Factor"] = float('nan')

            # Sharpe Ratio (Annualized proxy, assuming ~daily periods or scaling manually)
            # Standard simple Sharpe = mean / std
            r_std = r.std()
            if r_std > 1e-9:
                metrics["Sharpe_Ratio"] = float(r.mean() / r_std)
            else:
                metrics["Sharpe_Ratio"] = 0.0

        except Exception as e:
            logger.warning(f"Failed to calculate portfolio metrics: {e}")

        return metrics

    def generate_reports(self, df, events_df, production_geometries, oof_results):
        """Step 4: Generate Reports."""
        tprint_info(">>> Layer 2: Step 4 - Generate Reports...")

        oof_scores = oof_results['l2_score']
        oof_labels = oof_results['l2_label']
        oof_returns = oof_results.get('oof_returns') # May not be present in minimal dict
        if oof_returns is None and 'l2_returns' in oof_results:
             oof_returns = oof_results['l2_returns']

        oof_weights = oof_results['weights']
        final_geo_preds = oof_results['individual_geometries']

        # Extract Validation Diagnostics
        diagnostics = oof_results.get('diagnostics', {})
        signal_inflation = diagnostics.get('signal_inflation_ratio', 0.0)
        n_bagged = diagnostics.get('n_bagged_signals', 0)
        n_base = diagnostics.get('n_base_events', 0)
        mean_probs = diagnostics.get('mean_consensus_prob', None)

        # Compute divergence metrics
        divergence_mean = 0.0
        divergence_std = 0.0
        corr_max_mean = 0.0
        coverage_diff_06 = 0.0

        if mean_probs is not None and oof_scores is not None:
            try:
                # Align
                s_max = pd.to_numeric(oof_scores, errors='coerce')
                s_mean = pd.to_numeric(mean_probs.reindex(s_max.index), errors='coerce')

                valid = s_max.notna() & s_mean.notna()
                if valid.sum() > 10:
                    diff = s_max[valid] - s_mean[valid]
                    divergence_mean = float(diff.mean())
                    divergence_std = float(diff.std())
                    corr_max_mean = float(s_max[valid].corr(s_mean[valid]))

                    cov_max_06 = (s_max[valid] > 0.6).mean()
                    cov_mean_06 = (s_mean[valid] > 0.6).mean()
                    coverage_diff_06 = float(cov_max_06 - cov_mean_06)
            except Exception:
                pass

        # --- Extended Metrics Calculation ---
        ranking_metrics = {}
        portfolio_metrics = {}
        global_metrics = {}

        # 1. Global (Threshold-Independent)
        try:
            valid_mask = oof_labels.notna() & oof_scores.notna()
            if valid_mask.sum() > 10:
                y_true = oof_labels[valid_mask]
                y_score = oof_scores[valid_mask]

                # Check if binary
                if len(np.unique(y_true)) > 1:
                    global_metrics["ROC_AUC"] = float(roc_auc_score(y_true, y_score))
                    global_metrics["PR_AUC"] = float(average_precision_score(y_true, y_score)) # Average Precision
        except Exception as e:
            logger.warning(f"Global metrics failed: {e}")

        # 2. Ranking (Threshold-Dependent)
        if oof_scores is not None and oof_labels is not None:
            ranking_metrics = self._calculate_ranking_metrics(oof_labels, oof_scores)

        # 3. Financial (Risk-Adjusted)
        # Calculate returns of the strategy (e.g. taking trades where score > 0.5)
        if oof_returns is not None and oof_scores is not None:
            # Filter for traded events (e.g. score > 0.5)
            # This is a basic proxy for "Business-Centric" outcome of the ensemble
            traded_mask = oof_scores > 0.5
            traded_returns = oof_returns[traded_mask]
            portfolio_metrics = self._calculate_portfolio_metrics(traded_returns)

            # Add Expected Profit (Mean Return of Traded)
            portfolio_metrics["Expected_Profit_Per_Trade"] = float(traded_returns.mean()) if not traded_returns.empty else 0.0

        # ... logic for reports ...
        # (Copied from original run method)

        try:
            cfg = getattr(self, "_current_config", {})
            if not isinstance(cfg, dict):
                cfg = {}
        except Exception:
            cfg = {}

        try:
            ts = str(cfg.get("run_timestamp") or datetime.utcnow().strftime("%Y%m%d_%H%M%S"))
        except Exception:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        try:
            symbol = str(cfg.get("symbol", ""))
        except Exception:
            symbol = ""
        try:
            timeframe = str(cfg.get("timeframe", ""))
        except Exception:
            timeframe = ""

        try:
            outcomes_dir = Path("outcomes")
            outcomes_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            outcomes_dir = Path("outcomes")

        try:
            n_bars = int(len(df))
        except Exception:
            n_bars = 0
        try:
            n_events = int(len(events_df))
        except Exception:
            n_events = 0

        # Note: full_results not available here unless passed.
        # We can reconstruct counts from production_geometries or skip
        extracted_trials_counts = {}

        try:
            oof_labeled = int(pd.to_numeric(oof_labels, errors="coerce").notna().sum())
        except Exception:
            oof_labeled = 0
        try:
            oof_weight_nonzero = int((pd.to_numeric(oof_weights, errors="coerce").fillna(0.0).astype(float) > 0.0).sum())
        except Exception:
            oof_weight_nonzero = 0
        try:
            n_geo_channels = int(len(final_geo_preds or {}))
        except Exception:
            n_geo_channels = 0

        # --- Diagnostics Calculation ---
        try:
            # 1. Signal Coverage
            n_total_events = len(oof_scores)
            n_signals = (oof_scores > 0.5).sum()
            coverage_pct = (n_signals / n_total_events * 100.0) if n_total_events > 0 else 0.0

            # 2. Entropy
            # Clip to avoid log(0)
            p_safe = oof_scores.clip(1e-9, 1.0 - 1e-9)
            entropy_vals = -(p_safe * np.log(p_safe) + (1.0 - p_safe) * np.log(1.0 - p_safe))
            entropy_mean = entropy_vals.mean()
            entropy_std = entropy_vals.std()

            # 3. Tree Stats
            # Recovered from oof_results if passed (substep mode) or self._all_tree_stats (monolithic mode)
            all_tree_stats = oof_results.get('tree_stats')
            if not all_tree_stats:
                all_tree_stats = getattr(self, '_all_tree_stats', [])

            avg_feats_used = 0.0
            avg_depth = 0.0
            if all_tree_stats:
                avg_feats_used = np.mean([s['n_features_used'] for s in all_tree_stats])
                avg_depth = np.mean([s['avg_depth'] for s in all_tree_stats])
        except Exception as e:
            logger.warning(f"Error calculating diagnostics: {e}")
            coverage_pct = 0.0
            entropy_mean = 0.0
            entropy_std = 0.0
            avg_feats_used = 0.0
            avg_depth = 0.0

        try:
            md_path = outcomes_dir / f"layer2_report_{symbol}_{timeframe}_{ts}.md"
            lines = [
                "# Layer2 Report\n",
                f"- timestamp: {ts}\n",
                f"- symbol: {symbol}\n",
                f"- timeframe: {timeframe}\n",
                f"- n_bars: {n_bars}\n",
                f"- n_events: {n_events}\n",
                f"- cache_hits: {int(getattr(self, '_cache_hits', 0))}\n",
                f"- cache_misses: {int(getattr(self, '_cache_misses', 0))}\n",
                f"- extracted_trials_per_family: {extracted_trials_counts}\n",
                f"- production_geometries_n: {int(len(production_geometries or []))}\n",
                f"- oof_labeled_events: {oof_labeled}\n",
                f"- oof_nonzero_weight_events: {oof_weight_nonzero}\n",
                f"- oof_geometry_channels: {n_geo_channels}\n",
                "\n## Diagnostics\n",
                "### 1. Signal Coverage (First-Order Test)\n",
                f"- **Coverage**: {coverage_pct:.2f}%\n",
                "- **Diagnosis**:\n",
                "  - < 5-10%: Under-hunting (over-regularised)\n",
                "  - 20-50%: Healthy hunting regime\n",
                "  - > 70%: Likely noise saturation\n",
                "\n### 2. Prediction Entropy Distribution\n",
                f"- **Mean Entropy**: {entropy_mean:.4f} (Max ~0.693)\n",
                f"- **Entropy Std**: {entropy_std:.4f}\n",
                "- **Diagnosis**:\n",
                "  - Mass near 0 or 1: Over-confident / brittle\n",
                "  - Mass near 0.5: Under-hunting\n",
                "  - Wide distribution: Healthy\n",
                "\n### 3. Feature Utilisation / Split Diversity\n",
                f"- **Avg Features Used**: {avg_feats_used:.1f}\n",
                f"- **Avg Leaf Depth**: {avg_depth:.2f}\n",
                "- **Diagnosis**:\n",
                "  - Few features, shallow: Over-regularised\n",
                "  - Many features, deep: Expressive (desired)\n",
                "\n### 4. Bagging Logic Validation (De Prado 1.2)\n",
                f"- **Signal Inflation Ratio**: {signal_inflation:.2f}x (Target < 1.8x)\n",
                f"  - Base Events: {n_base}\n",
                f"  - Bagged Signals: {n_bagged}\n",
                f"- **Max vs Mean Consensus**:\n",
                f"  - Mean Divergence (Max - Mean): {divergence_mean:.4f} (std {divergence_std:.4f})\n",
                f"  - Correlation: {corr_max_mean:.4f}\n",
                f"  - Coverage Delta @ 0.6: {coverage_diff_06*100:.1f}pp\n",
                "\n### 5. Ranking Quality (Lift & Precision)\n",
            ]

            # Append Ranking Metrics
            if ranking_metrics:
                lines.append("| Metric | Value |\n|---|---|\n")
                for k in sorted(ranking_metrics.keys()):
                    lines.append(f"| {k} | {ranking_metrics[k]:.4f} |\n")

            lines.append("\n### 6. Financial / Business Metrics (Score > 0.5)\n")
            if portfolio_metrics:
                lines.append("| Metric | Value |\n|---|---|\n")
                for k in sorted(portfolio_metrics.keys()):
                    lines.append(f"| {k} | {portfolio_metrics[k]:.6f} |\n")

            if global_metrics:
                lines.append("\n### 7. Global Model Quality\n")
                for k, v in global_metrics.items():
                    lines.append(f"- **{k}**: {v:.4f}\n")

            if production_geometries:
                lines.append("\n### 7. Winning Geometries Details\n")
                lines.append("| UUID | Family | Model Type | Race Score | Selected Features |\n|---|---|---|---|---|\n")
                for g in production_geometries:
                    m_type = "N/A"
                    race_score = "N/A"
                    if isinstance(g.model_params, dict):
                        m_type = g.model_params.get('model_type', 'lgbm')
                        # Extract score of winner from race_scores
                        r_scores = g.model_params.get('race_scores', {})
                        if r_scores and m_type in r_scores:
                             race_score = f"{r_scores[m_type]:.4f}"

                    n_feats = len(g.selected_features) if g.selected_features else 0
                    lines.append(f"| {g.uuid} | {g.family} | {m_type} | {race_score} | {n_feats} |\n")

            md_path.write_text("".join(lines))
            tprint_success(f"Generated Layer 2 Report: {md_path}")

            # Save RFE Stats CSV
            if self._rfe_stats:
                rfe_df = pd.DataFrame(self._rfe_stats)
                rfe_csv_path = outcomes_dir / f"titan_rfe_stats_{ts}.csv"
                rfe_df.to_csv(rfe_csv_path, index=False)
                tprint_success(f"Saved Titan RFE summary to {rfe_csv_path}")

        except Exception as e:
            tprint_error(f"Report generation failed: {e}")
            pass

        try:
            summary_row: Dict[str, Any] = {
                "timestamp": ts,
                "symbol": symbol,
                "timeframe": timeframe,
                "n_bars": n_bars,
                "n_events": n_events,
                "cache_hits": int(getattr(self, "_cache_hits", 0)),
                "cache_misses": int(getattr(self, "_cache_misses", 0)),
                "production_geometries_n": int(len(production_geometries or [])),
                "oof_labeled_events": int(oof_labeled),
                "oof_nonzero_weight_events": int(oof_weight_nonzero),
                "oof_geometry_channels": int(n_geo_channels),
            }
            # Merge new metrics into summary row
            summary_row.update(ranking_metrics)
            summary_row.update(portfolio_metrics)
            summary_row.update(global_metrics)

            for fam, cnt in extracted_trials_counts.items():
                summary_row[f"extracted_trials_{fam}"] = int(cnt)
            csv_path = outcomes_dir / f"layer2_summary_{symbol}_{timeframe}_{ts}.csv"
            pd.DataFrame([summary_row]).to_csv(csv_path, index=False)
        except Exception:
            pass

        try:
            rows = []
            for g in list(production_geometries or []):
                try:
                    fam = str(getattr(g, "family", ""))
                    params = getattr(g, "params", None)
                    kappa = None
                    sl_mult = None
                    horizon = None
                    if isinstance(params, dict):
                        kappa = params.get("kappa")
                        sl_mult = params.get("sl_mult")
                        horizon = params.get("horizon")

                    mean_return = float("nan")
                    ret_std = float("nan")
                    sharpe_proxy = float("nan")
                    win_rate = float("nan")
                    n_geom_events = 0
                    pos_ratio = float("nan")

                    try:
                        fam_events = events_df[events_df.get('family') == fam] if 'family' in events_df.columns else events_df
                        if kappa is not None and horizon is not None:
                            _lbl, _ret, _, _ = self._compute_dominance_labels(
                                df=df,
                                events_df=fam_events,
                                kappa=float(kappa),
                                horizon=int(horizon),
                                family=fam,
                                sl_mult=(float(sl_mult) if sl_mult is not None else None),
                            )
                            _ret_s = pd.to_numeric(_ret, errors='coerce').astype(float)
                            _lbl_s = pd.to_numeric(_lbl, errors='coerce').astype(float)
                            _ret_s = _ret_s.replace([np.inf, -np.inf], np.nan)
                            _lbl_s = _lbl_s.replace([np.inf, -np.inf], np.nan)

                            n_geom_events = int(_ret_s.notna().sum())
                            mean_return = float(_ret_s.mean()) if n_geom_events > 0 else float('nan')
                            ret_std = float(_ret_s.std()) if n_geom_events > 1 else float('nan')
                            sharpe_proxy = float(mean_return) / (float(ret_std) + 1e-12) if np.isfinite(mean_return) and np.isfinite(ret_std) else float('nan')

                            try:
                                win_rate = float((_ret_s.dropna() > 0.0).mean()) if n_geom_events > 0 else float('nan')
                            except Exception:
                                win_rate = float('nan')

                            try:
                                pos_ratio = float((_lbl_s.dropna() == 1.0).mean()) if int(_lbl_s.notna().sum()) > 0 else float('nan')
                            except Exception:
                                pos_ratio = float('nan')
                    except Exception:
                        pass

                    row = {
                        "timestamp": ts,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "uuid": str(getattr(g, "uuid", "")),
                        "family": fam,
                        "final_score": float(getattr(g, "final_score", np.nan)),
                        "learnability": float(getattr(g, "learnability", np.nan)),
                        "robust_magnitude": float(getattr(g, "robust_magnitude", np.nan)),
                        "stability": float(getattr(g, "stability", np.nan)),
                        "balance": float(getattr(g, "balance", np.nan)),
                        "mean_return": float(mean_return),
                        "count": int(n_geom_events),
                        "win_rate": float(win_rate),
                        "return_std": float(ret_std),
                        "sharpe_proxy": float(sharpe_proxy),
                        "pos_ratio": float(pos_ratio),
                    }
                    if isinstance(params, dict):
                        for k, v in params.items():
                            row[f"param_{k}"] = v

                    raw_metrics = getattr(g, 'raw_metrics', None)
                    if isinstance(raw_metrics, dict):
                        for k, v in raw_metrics.items():
                            row[f"raw_{k}"] = v

                    rows.append(row)
                except Exception:
                    continue
            if rows:
                df_geos = pd.DataFrame(rows)
                df_geos.to_csv(
                    outcomes_dir / f"layer2_production_geometries_{symbol}_{timeframe}_{ts}.csv",
                    index=False,
                )
                df_geos.to_csv(
                    outcomes_dir / f"layer2_geometry_metrics_{symbol}_{timeframe}_{ts}.csv",
                    index=False,
                )
        except Exception:
            pass

    def _extract_events_for_selection(
        self,
        df: pd.DataFrame,
        events_df: pd.DataFrame,
        max_horizon: int = 120
    ) -> List[Event]:
        """
        Convert DataFrame and events into the list of Event objects required by label_geometry_selection.
        Extracts future return paths.
        """
        events_list = []

        # Get integer locations
        # We need a quick way to slice df
        # Ensure df is sorted by index
        df = df.sort_index()

        # Map event timestamps to integer locations in df
        # get_indexer returns -1 for missing
        idx_locs = df.index.get_indexer(events_df.index)

        # Pre-fetch numpy arrays for speed
        close_arr = df['close'].to_numpy()
        vol_arr = df['volatility_1d'].to_numpy()

        # Directions
        if 'event_consensus' in events_df.columns:
            directions = np.sign(events_df['event_consensus'].fillna(0).to_numpy())
        else:
            # Fallback to config direction
            try:
                dir_raw = str(getattr(self, "_current_config", {}).get("direction", "long")).lower()
            except Exception:
                dir_raw = "long"
            default_dir = 1.0
            if dir_raw in {"short", "sell", "-1", "-1.0", "s"}:
                default_dir = -1.0
            directions = np.full(len(events_df), default_dir)

        for i, (ts, row) in enumerate(events_df.iterrows()):
            loc = idx_locs[i]
            if loc == -1: continue

            # Horizon
            start_loc = loc
            end_loc = min(len(df), loc + max_horizon)

            # Extract path
            price_path = close_arr[start_loc:end_loc]
            if len(price_path) < 2: continue

            # Cumulative returns relative to entry
            entry_price = price_path[0]
            if entry_price <= 0: continue

            # path: (P_t - P_0) / P_0
            # Note: label_geometry_selection logic expects 'returns_path' to be cumulative return from entry
            returns_path = (price_path - entry_price) / entry_price

            sigma = vol_arr[loc]
            if np.isnan(sigma) or sigma <= 0: sigma = 0.01 # Fallback

            # ID: use integer index for simplicity or hash timestamp
            event_id = i

            e = Event(
                id=event_id,
                # FIX: Use ABSOLUTE bar positions for proper uniqueness/duration calculation
                entry_idx=start_loc,  # Absolute position in df
                exit_idx=end_loc,     # Absolute position in df
                direction=int(directions[i]) if directions[i] != 0 else 1,
                returns_path=returns_path,
                sigma=float(sigma)
            )
            events_list.append(e)

        return events_list

    def _extract_tree_diagnostics(self, model_or_booster) -> Dict[str, float]:
        """
        Extract diagnostics from a trained LGBM booster or sklearn wrapper.
        Returns:
            - n_features_used: Count of features with importance > 0
            - avg_depth: Average depth of leaves across all trees
            - max_depth: Maximum depth found
        """
        if model_or_booster is None:
            return {'n_features_used': 0.0, 'avg_depth': 0.0, 'max_depth': 0.0}

        try:
            # Unwrap sklearn-API models
            booster = model_or_booster
            if hasattr(model_or_booster, 'booster_'):
                booster = model_or_booster.booster_

            # 1. Feature Usage
            imp = booster.feature_importance(importance_type='split')
            n_features = int(np.sum(imp > 0))

            # 2. Tree Depth
            # dump_model returns a dict with 'tree_info' list
            dump = booster.dump_model()
            trees = dump.get('tree_info', [])

            depths = []

            for tree in trees:
                if 'tree_structure' not in tree:
                    continue

                # Traverse tree to find leaf depths
                # Stack: (node, depth)
                stack = [(tree['tree_structure'], 0)]
                while stack:
                    node, d = stack.pop()
                    if 'leaf_index' in node:
                        # It's a leaf
                        depths.append(d)
                    else:
                        if 'left_child' in node:
                            stack.append((node['left_child'], d + 1))
                        if 'right_child' in node:
                            stack.append((node['right_child'], d + 1))

            avg_depth = float(np.mean(depths)) if depths else 0.0
            max_depth = float(np.max(depths)) if depths else 0.0

            return {
                'n_features_used': float(n_features),
                'avg_depth': avg_depth,
                'max_depth': max_depth
            }
        except Exception as e:
            logger.warning(f"Failed to extract tree diagnostics: {e}")
            return {'n_features_used': 0.0, 'avg_depth': 0.0, 'max_depth': 0.0}

    def _precompute_geometry_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pre-compute rolling metrics needed for geometry-specific features.
        Adds columns to df (on a copy) if they don't exist.
        """
        df_out = df.copy()

        # 1. ATR-14
        if 'geo_atr_14' not in df_out.columns:
            try:
                high = df_out['high'] if 'high' in df_out.columns else df_out['close']
                low = df_out['low'] if 'low' in df_out.columns else df_out['close']
                close = df_out['close']
                prev_close = close.shift(1)
                tr1 = (high - low).abs()
                tr2 = (high - prev_close).abs()
                tr3 = (low - prev_close).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                df_out['geo_atr_14'] = tr.rolling(14).mean()
            except Exception:
                df_out['geo_atr_14'] = np.nan

        # 2. Recent Returns (10, 20 bars)
        if 'geo_ret_10' not in df_out.columns:
            df_out['geo_ret_10'] = df_out['close'].pct_change(10).abs() # Magnitude
        if 'geo_ret_20' not in df_out.columns:
            df_out['geo_ret_20'] = df_out['close'].pct_change(20).abs() # Magnitude

        # 3. Range-50, Min-50, Max-50
        if 'geo_range_50' not in df_out.columns:
            try:
                high = df_out['high'] if 'high' in df_out.columns else df_out['close']
                low = df_out['low'] if 'low' in df_out.columns else df_out['close']
                h50 = high.rolling(50).max()
                l50 = low.rolling(50).min()
                df_out['geo_max_50'] = h50
                df_out['geo_min_50'] = l50
                df_out['geo_range_50'] = h50 - l50
            except Exception:
                df_out['geo_range_50'] = np.nan
                df_out['geo_max_50'] = np.nan
                df_out['geo_min_50'] = np.nan

        return df_out

    def _compute_specific_geometry_features(
        self,
        df: pd.DataFrame,
        events_index: pd.Index,
        params: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Compute geometry-specific features for a given set of events and parameters.
        These features depend on kappa/sl_mult and thus vary per geometry.

        Features:
        - Volatility / Stop Size
        - Volatility / Target Size
        - Return(10/20) / Stop Size
        - Return(10/20) / Target Size
        - ATR / Stop Size
        - ATR / Target Size
        - Range / Stop Size
        - Range / Target Size
        - Normalized Dist from Min/Max
        """
        if events_index.empty:
            return pd.DataFrame()

        # Extract params
        kappa = params.get('kappa')
        sl_mult = params.get('sl_mult')

        # New params support
        sl_sigma = params.get('sl_sigma')
        alpha = params.get('alpha')
        beta = params.get('beta')
        min_ratio = params.get('min_ratio')

        # Resolve SL
        if sl_sigma is not None:
            eff_sl = float(sl_sigma)
        elif sl_mult is not None:
            eff_sl = float(sl_mult)
        else:
            eff_sl = 1.0

        # Resolve Target (Kappa)
        if kappa is not None:
            eff_kappa = float(kappa)
        elif alpha is not None and beta is not None and min_ratio is not None and sl_sigma is not None:
            # Calculate effective kappa implied by the score condition at the stop distance
            # score = (mfe_norm ^ beta) / (mae_norm ^ alpha) >= min_ratio
            # Assuming mae_norm hits stop (sl_sigma)
            # mfe_norm >= (min_ratio * sl_sigma^alpha) ^ (1/beta)
            try:
                eff_kappa = (float(min_ratio) * (float(sl_sigma) ** float(alpha))) ** (1.0 / float(beta))
            except:
                eff_kappa = 2.0
        else:
            eff_kappa = 2.0

        # Get subset of DF aligned with events
        # We need historical context for rolling features, but _precompute_geometry_base_features
        # already put them in df. So we just need values at events_index.

        try:
            subset = df.reindex(events_index)

            vol = subset['volatility_1d'].fillna(0.0)

            # ATR Handling: ensure we have Price-unit ATR
            close = subset['close']
            atr_price = subset.get('geo_atr_14')
            if atr_price is None or atr_price.isna().all():
                # Fallback: estimate ATR as Vol * Close
                atr_price = vol * close
            atr_price = atr_price.fillna(0.0)

            # ATR Percentage (for ratios against stop_size)
            atr_pct = atr_price / close
            atr_pct = atr_pct.replace([np.inf, -np.inf], 0.0).fillna(0.0)

            # Compute Stop and Target Sizes (Percentage distance)
            # Enforce min profit logic from _compute_dominance_labels
            min_profit = self.transaction_cost * 1.1

            # ECONOMIC CONSTRAINTS
            min_sl_dist = 0.004
            max_tp_dist = 0.03

            # Target Size (Percentage)
            raw_target = eff_kappa * vol
            target_size = np.maximum(raw_target, min_profit)
            # Apply Max TP ceiling
            target_size = np.minimum(target_size, max_tp_dist)

            target_size = target_size.replace(0.0, np.nan) # Avoid div/0

            # Stop Size (Percentage)
            raw_stop = eff_sl * vol
            # Apply Min SL floor
            stop_size = np.maximum(raw_stop, min_sl_dist)

            stop_size = stop_size.replace(0.0, np.nan)

            # Features
            feats = pd.DataFrame(index=events_index)

            # 1. Volatility Ratios
            feats['geo_vol_to_stop'] = vol / stop_size
            feats['geo_vol_to_target'] = vol / target_size

            # 2. Recent Return Ratios
            ret10 = subset.get('geo_ret_10', pd.Series(0, index=events_index)).fillna(0.0)
            ret20 = subset.get('geo_ret_20', pd.Series(0, index=events_index)).fillna(0.0)

            feats['geo_ret10_to_stop'] = ret10 / stop_size
            feats['geo_ret20_to_stop'] = ret20 / stop_size
            feats['geo_ret10_to_target'] = ret10 / target_size
            feats['geo_ret20_to_target'] = ret20 / target_size

            # 3. ATR Ratios (Normalized: Percentage / Percentage)
            feats['geo_atr_to_stop'] = atr_pct / stop_size
            feats['geo_atr_to_target'] = atr_pct / target_size

            # 4. Range Ratios
            rng50 = subset.get('geo_range_50', atr_price * 3.0).fillna(0.0)
            # Normalize range to percentage (range / close) to match stop_size units
            rng50_pct = rng50 / close

            feats['geo_range_to_stop'] = rng50_pct / stop_size
            feats['geo_range_to_target'] = rng50_pct / target_size

            # 5. Normalized Distance from Local Extremum
            # (Close - Min) / ATR_Price  -> Price / Price = Ratio (Stationary)
            min50 = subset.get('geo_min_50', close)
            max50 = subset.get('geo_max_50', close)

            safe_atr = atr_price.replace(0.0, np.nan)

            feats['geo_dist_from_min'] = (close - min50) / safe_atr
            feats['geo_dist_from_max'] = (max50 - close) / safe_atr

            # Fill NaNs/Infs
            feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)

            return feats

        except Exception as e:
            logger.warning(f"Failed to compute specific geometry features: {e}")
            return pd.DataFrame(index=events_index)

    def _validate_inputs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure the dataframe contains the full OHLCV + volatility context expected by MTF feature generation.
        Returns (potentially modified) copy of df.
        """
        required_price_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_price = [c for c in required_price_cols if c not in df.columns]
        if missing_price:
            raise ValueError(
                "Layer2 expects raw OHLCV inputs (Kalman-smoothed price frame) but received a dataframe "
                f"missing {missing_price}. Pass the market_data dataframe instead of derived feature matrices."
            )

        numeric_cols = df[required_price_cols].select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) != len(required_price_cols):
            raise ValueError(
                "One or more OHLCV columns are non-numeric. Ensure Layer2 receives the raw OHLCV dataframe "
                "before derived features are attached."
            )

        if 'volatility_1d' not in df.columns:
            raise ValueError("Missing required column 'volatility_1d' in df.")

        # Check for regime columns, if missing create dummies (on a copy if needed)
        df_out = df
        if 'trend_regime' not in df.columns:
            logger.warning("'trend_regime' missing. Creating dummy 'Low' regime.")
            if df_out is df: df_out = df.copy()
            df_out['trend_regime'] = 'Low'
        if 'vol_regime' not in df.columns:
            logger.warning("'vol_regime' missing. Creating dummy 'Low' regime.")
            if df_out is df: df_out = df.copy()
            df_out['vol_regime'] = 'Low'
        
        return df_out

    def _generate_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 0: Generate events using CUSUM filter.
        Returns a DataFrame of event timestamps.
        """
        config = getattr(self, '_current_config', {})
        if not isinstance(config, dict):
            config = {}

        # Call generate_primary_signals which uses CUSUM
        # We pass the config to allow tuning CUSUM params
        try:
            cfg_signals = dict(config)
            try:
                if 'k' not in cfg_signals:
                    k_override = cfg_signals.get('layer2_signal_k')
                    if k_override is None:
                        k_override = cfg_signals.get('layer2_default_k', 0.12)
                    cfg_signals['k'] = float(k_override)

                # Expose advanced CUSUM params if configured
                if 'alpha' not in cfg_signals:
                    alpha_override = cfg_signals.get('layer2_signal_alpha')
                    if alpha_override is not None:
                        cfg_signals['alpha'] = float(alpha_override)

                if 'beta' not in cfg_signals:
                    beta_override = cfg_signals.get('layer2_signal_beta')
                    if beta_override is not None:
                        cfg_signals['beta'] = float(beta_override)

                if 'er_min' not in cfg_signals:
                    er_min_override = cfg_signals.get('layer2_signal_er_min')
                    if er_min_override is not None:
                        cfg_signals['er_min'] = float(er_min_override)

            except Exception:
                pass
            signals = generate_primary_signals(
                df,
                **cfg_signals
            )

            try:
                consensus = pd.to_numeric(signals.get('consensus'), errors='coerce').astype(float)
            except Exception:
                consensus = pd.Series(0.0, index=df.index, dtype=float)
            consensus = consensus.reindex(df.index).fillna(0.0)
            self._primary_signals = pd.DataFrame({'consensus': consensus}, index=df.index)

            trigger_mask = consensus != 0.0

            try:
                dir_raw = str(config.get('direction', 'long')).lower()
            except Exception:
                dir_raw = 'long'

            if dir_raw in {'long', 'buy', '1', '1.0', '+1', 'l'}:
                trigger_mask = trigger_mask & (consensus > 0.0)
            elif dir_raw in {'short', 'sell', '-1', '-1.0', 's'}:
                trigger_mask = trigger_mask & (consensus < 0.0)

            events = df.index[trigger_mask]
        except Exception as e:
            logger.warning(f"Error in CUSUM event generation: {e}. Falling back to basic events.")
            # Fallback to absolute returns threshold if CUSUM fails
            returns = df['close'].pct_change().abs()
            trigger_mask = (returns > 0.004).fillna(False)
            events = df.index[trigger_mask]

            try:
                consensus = pd.to_numeric(df['close'].pct_change().shift(1), errors='coerce').astype(float)
                consensus = np.sign(consensus).reindex(df.index).fillna(0.0)
            except Exception:
                consensus = pd.Series(0.0, index=df.index, dtype=float)
            self._primary_signals = pd.DataFrame({'consensus': consensus}, index=df.index)

            try:
                dir_raw = str(config.get('direction', 'long')).lower()
            except Exception:
                dir_raw = 'long'
            if dir_raw in {'long', 'buy', '1', '1.0', '+1', 'l'}:
                events = df.index[trigger_mask & (consensus > 0.0)]
            elif dir_raw in {'short', 'sell', '-1', '-1.0', 's'}:
                events = df.index[trigger_mask & (consensus < 0.0)]

        logger.info(f"Generated {len(events)} events from {len(df)} bars using CUSUM filter.")

        # Create events dataframe (index=timestamp)
        # We store regime info here for easy lookup
        events_df = df.loc[events, ['trend_regime', 'vol_regime', 'volatility_1d']].copy()

        try:
            if self._primary_signals is not None and 'consensus' in self._primary_signals.columns:
                evt_cons = pd.to_numeric(self._primary_signals['consensus'].reindex(events_df.index), errors='coerce').astype(float)
                evt_cons = np.sign(evt_cons).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                events_df['event_consensus'] = evt_cons.astype(float)
        except Exception:
            pass

        return events_df

    def _events_cache_key(self, events_index: pd.Index) -> Tuple[Any, ...]:
        try:
            idx = pd.DatetimeIndex(events_index)
        except Exception:
            idx = events_index

        n = int(len(idx))
        if n <= 0:
            return (0, None, None, None, None)
        first = idx[0]
        last = idx[-1]
        mid1 = idx[1] if n > 1 else None
        mid2 = idx[-2] if n > 1 else None
        return (n, first, last, mid1, mid2)

    def _df_cache_key(self, df: pd.DataFrame) -> Tuple[Any, ...]:
        idx = df.index
        n = int(len(idx))
        if n <= 0:
            return (0, None, None)
        return (n, idx[0], idx[-1])

    def _select_global_probe_features(self, X_events: pd.DataFrame) -> List[str]:
        try:
            target_n = int(getattr(self, '_current_config', {}).get('layer2_probe_feature_count', 70))
        except Exception:
            target_n = 70
        try:
            corr_threshold = float(getattr(self, '_current_config', {}).get('layer2_probe_corr_threshold', 0.95))
        except Exception:
            corr_threshold = 0.95
        try:
            max_rows = int(getattr(self, '_current_config', {}).get('layer2_probe_corr_rows', 2000))
        except Exception:
            max_rows = 2000

        ranked = [str(c) for c in list(X_events.columns)]
        try:
            selected = self._cheap_corr_prune(
                X_events,
                ranked_features=ranked,
                target_n=int(target_n),
                corr_threshold=float(corr_threshold),
                max_rows=int(max_rows),
            )
        except Exception:
            selected = ranked[: int(target_n)]
        return [c for c in selected if c in X_events.columns]

    def _get_or_build_signals(self, df: pd.DataFrame, events_df: pd.DataFrame, family: str) -> pd.DataFrame:
        try:
            dir_raw = str(getattr(self, "_current_config", {}).get("direction", "long")).lower()
        except Exception:
            dir_raw = "long"
        default_dir = 1.0
        if dir_raw in {"short", "sell", "-1", "-1.0", "s"}:
            default_dir = -1.0

        direction_mode = "primary"
        key = (
            self._df_cache_key(df),
            self._events_cache_key(events_df.index),
            str(family),
            str(direction_mode),
            float(default_dir),
        )

        cached = self._signals_cache.get(key)
        if cached is not None:
            return cached

        base_cons = None
        try:
            if 'event_consensus' in events_df.columns:
                base_cons = pd.to_numeric(events_df['event_consensus'], errors='coerce').astype(float)
        except Exception:
            base_cons = None
        if base_cons is None:
            try:
                if self._primary_signals is not None and 'consensus' in self._primary_signals.columns:
                    base_cons = pd.to_numeric(self._primary_signals['consensus'].reindex(events_df.index), errors='coerce').astype(float)
            except Exception:
                base_cons = None
        if base_cons is None:
            base_cons = pd.Series(float(default_dir), index=events_df.index, dtype=float)

        directions = np.sign(base_cons.to_numpy(dtype=float, copy=False))
        directions = np.where(np.isfinite(directions), directions, float(default_dir))
        directions[directions == 0.0] = float(default_dir)

        try:
            mr_flip = bool(getattr(self, "_current_config", {}).get("layer2_mean_reversion_flip_direction", False))
        except Exception:
            mr_flip = False
        if mr_flip and family == 'Mean Reversion':
            directions = -directions

        idx = df.index
        consensus_arr = np.zeros(len(idx), dtype=float)
        pos = idx.get_indexer(events_df.index)
        valid_pos = pos >= 0
        if np.any(valid_pos):
            consensus_arr[pos[valid_pos]] = directions[valid_pos]

        signals = pd.DataFrame({'consensus': consensus_arr}, index=idx)
        self._signals_cache[key] = signals
        return signals

    def _assign_barrier_families(self, events_df: pd.DataFrame) -> pd.Series:
        """
        Assign barrier families.
        Refactored: Returns 'Unified' for all events (no family distinction).
        """
        return pd.Series('Unified', index=events_df.index, dtype=object)

    def _compute_dominance_labels(
        self,
        df: pd.DataFrame,
        events_df: pd.DataFrame,
        # Accepted params:
        kappa: float = None,
        horizon: int = 120,
        family: str = '',
        sl_mult: float = None,
        # New params
        sl_sigma: float = None,
        alpha: float = None,
        beta: float = None,
        min_ratio: float = None,
        events_shift: int = 0,
        **kwargs
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Compute TP/SL(+optional trailing) exit-model labels and related metrics.
        Label = 1 if the trade exits via profit barrier (or trailing), else 0.

        De Prado 1.3: Objective Outcome Generation
        This function encodes objective trade outcomes only (Dominance).
        No preference, asymmetry, or utility shaping occurs here.
        Preferences are handled exclusively in the loss function (RobustFocalLoss).

        Args:
            df: Market data
            events_df: Events to label
            kappa: Dominance ratio threshold
            horizon: Window size
            family: Geometry family (defines direction)
            events_shift: Shift event timestamps by N bars (for stability check)
            sl_mult: Optional stop loss multiplier

        Returns:
            Tuple: (labels, returns, mfe, mae, exit_reasons)
        """
        # Determine logic mode
        is_new_logic = (alpha is not None) and (beta is not None)

        # 1. Horizon & Data Prep
        if horizon is None: horizon = 120
        horizon = int(horizon)

        # Prepare caching key (extended)
        cache_key = (
            self._df_cache_key(df),
            self._events_cache_key(events_df.index),
            str(family),
            float(kappa) if kappa else None,
            float(sl_mult) if sl_mult else None,
            float(alpha) if alpha else None,
            float(beta) if beta else None,
            float(min_ratio) if min_ratio else None,
            float(sl_sigma) if sl_sigma else None,
            int(horizon),
            int(events_shift),
            "new_logic_v1"
        )

        cached = self._labels_cache.get(cache_key)
        if cached is not None and not self.force_hpo:
            self._cache_hits += 1
            return cached
        self._cache_misses += 1

        # Resolve params
        eff_sl_sigma = sl_sigma if sl_sigma is not None else (sl_mult if sl_mult is not None else 1.0)

        vol_series = pd.to_numeric(df.get('volatility_1d'), errors='coerce').astype(float).reindex(events_df.index).fillna(0.0)

        stop_threshold = None
        profit_threshold = None

        # ECONOMIC CONSTRAINTS
        # Min SL: 0.4% floor (Hard economic constraint for noise filtering)
        min_sl_dist = 0.004
        # Max TP: 3% ceiling (Hard economic constraint for realistic targets)
        max_tp_dist = 0.03

        if is_new_logic:
            # New logic (Dominance Score)
            # stop_threshold = vol * multiplier, but clamped
            raw_stop = eff_sl_sigma * vol_series
            stop_threshold = np.maximum(raw_stop, min_sl_dist)
        else:
            # Legacy logic with adjustment (Triple Barrier)
            vol_median = vol_series.median()
            vol_adj_factor = 1.0 + 0.3 * ((vol_series - vol_median) / (vol_median + 1e-9))
            vol_adj_factor = vol_adj_factor.clip(lower=0.7, upper=1.3)

            raw_stop = float(eff_sl_sigma) * vol_series * vol_adj_factor
            # Apply Min SL Floor
            stop_threshold = np.maximum(raw_stop, min_sl_dist)

            # For Legacy, we also have profit_threshold
            eff_kappa = kappa if kappa is not None else 2.0
            min_profit = self.transaction_cost * 1.1

            raw_target = float(eff_kappa) * vol_series * vol_adj_factor
            # Apply Min Profit Floor (existing) and Max TP Ceiling (new)
            profit_threshold = np.maximum(raw_target, min_profit)
            profit_threshold = np.minimum(profit_threshold, max_tp_dist)

        signals = self._get_or_build_signals(df, events_df, family)

        # Handle Shift
        target_events_idx = events_df.index
        calc_signals = signals
        calc_events_idx = target_events_idx
        valid_locs = None

        if events_shift != 0:
            df_idx_locs = df.index.get_indexer(target_events_idx)
            shifted_locs = df_idx_locs + events_shift
            valid_locs = (shifted_locs >= 0) & (shifted_locs < len(df))

            if not np.any(valid_locs):
                 empty_s = pd.Series(np.nan, index=target_events_idx)
                 return empty_s, empty_s, empty_s, empty_s, empty_s

            shifted_timestamps = df.index[shifted_locs[valid_locs]]
            orig_signals = signals.loc[target_events_idx[valid_locs]]

            temp_signals = pd.DataFrame(0.0, index=df.index, columns=['consensus'])
            temp_signals.loc[shifted_timestamps, 'consensus'] = orig_signals['consensus'].values

            calc_signals = temp_signals
            calc_events_idx = shifted_timestamps

            # Re-align thresholds for shifted events if necessary (or assume thresholds valid at shift time?)
            # Usually thresholds (vol based) should be taken at entry time.
            # Here we are shifting entry time.
            # vol_series should be reindexed to shifted timestamps.
            if is_new_logic:
                # Re-calculate thresholds for shifted times
                vol_shifted = pd.to_numeric(df.get('volatility_1d'), errors='coerce').astype(float).reindex(calc_events_idx).fillna(0.0)
                stop_threshold = eff_sl_sigma * vol_shifted
            else:
                # Legacy re-calc
                vol_shifted = pd.to_numeric(df.get('volatility_1d'), errors='coerce').astype(float).reindex(calc_events_idx).fillna(0.0)
                vol_median = vol_shifted.median()
                vol_adj_factor = 1.0 + 0.3 * ((vol_shifted - vol_median) / (vol_median + 1e-9))
                vol_adj_factor = vol_adj_factor.clip(lower=0.7, upper=1.3)
                stop_threshold = float(eff_sl_sigma) * vol_shifted * vol_adj_factor
                eff_kappa = kappa if kappa is not None else 2.0
                min_profit = self.transaction_cost * 1.1
                profit_threshold = np.maximum(float(eff_kappa) * vol_shifted * vol_adj_factor, min_profit)
        
        if is_new_logic:
            # 1. Run with STOP only
            # Note: For new logic (Dominance), profit_threshold is effectively None (infinite)
            # But the 'score' calculation uses normalized MFE/MAE.
            # We still need to respect stop_threshold (clamped above).
            (
                realized_returns, _, exit_reasons, _,
                mfe_series, mae_series, _, _
            ) = compute_realized_returns(
                df=df,
                signals=calc_signals,
                profit_threshold=None, # Infinite profit
                stop_threshold=stop_threshold, # Fixed SL
                horizon=horizon,
                transaction_cost=self.transaction_cost,
                min_event_spacing=0
            )

            # 2. Check Profit Condition
            # Condition: (norm_mfe ** beta) / (norm_mae ** alpha) >= min_ratio
            # AND NOT Stop Hit (exit_reasons != 'stop')

            # Normalize
            # MFE/MAE from function are absolute.
            # norm = abs / vol
            # Handle alignment
            mfe_aligned = mfe_series.reindex(calc_events_idx)
            mae_aligned = mae_series.reindex(calc_events_idx)
            exit_aligned = exit_reasons.reindex(calc_events_idx)

            # Use volatility at entry time
            vol_at_entry = pd.to_numeric(df.get('volatility_1d'), errors='coerce').astype(float).reindex(calc_events_idx).fillna(0.0)

            # Apply Max TP constraint logic for scoring
            # If max_tp_dist is enforced, mfe should be capped at max_tp_dist for scoring?
            # Let's cap the effective MFE used for score calculation.
            mfe_capped = np.minimum(mfe_aligned, max_tp_dist)

            norm_mfe = mfe_capped / vol_at_entry
            norm_mae = mae_aligned / vol_at_entry

            # Avoid div/0
            norm_mae_safe = norm_mae.replace(0.0, 1e-6)

            score = (norm_mfe ** float(beta)) / (norm_mae_safe ** float(alpha))

            # Logic:
            # If Stop Hit (exit_reasons == 'stop') -> Label 0
            # Else If Score >= min_ratio -> Label 1
            # Else -> Label 0

            is_stop = exit_aligned == 'stop'
            is_profit = (score >= float(min_ratio)) & (~is_stop)

            binary_labels = is_profit.astype(float)

            # Mask out NaNs
            binary_labels[realized_returns.reindex(calc_events_idx).isna()] = np.nan

            # Construct subset returns/mfe/mae
            subset_returns = realized_returns.reindex(calc_events_idx)
            # If profit condition met, return = MFE (best case capture assumption for dominance)
            subset_returns[is_profit] = mfe_aligned[is_profit]

            subset_mfe = mfe_series.reindex(calc_events_idx)
            subset_mae = mae_series.reindex(calc_events_idx)
            subset_exit = exit_aligned  # FIX: Was missing in new logic block

        else:
            # Legacy Logic
            (
                realized_returns, _, exit_reasons, _,
                mfe_series, mae_series, _, _
            ) = compute_realized_returns(
                df=df,
                signals=calc_signals,
                profit_threshold=profit_threshold,
                stop_threshold=stop_threshold,
                horizon=horizon,
                transaction_cost=self.transaction_cost,
                min_event_spacing=0,
                volatility_series=None,
                atr_series=None,
                trail_distance_atr_mult=None,
                use_multiclass_labels=False,
                use_soft_labels=False,
            )

            subset_returns = realized_returns.reindex(calc_events_idx)
            subset_mfe = mfe_series.reindex(calc_events_idx)
            subset_mae = mae_series.reindex(calc_events_idx)
            subset_exit = exit_reasons.reindex(calc_events_idx)

            binary_labels = subset_exit.astype(str).isin(['profit', 'trailing']).astype(float)
            binary_labels = binary_labels.where(subset_returns.notna())

        if events_shift != 0:
            final_labels = pd.Series(np.nan, index=target_events_idx)
            final_returns = pd.Series(np.nan, index=target_events_idx)
            final_mfe = pd.Series(np.nan, index=target_events_idx)
            final_mae = pd.Series(np.nan, index=target_events_idx)
            final_exit = pd.Series(np.nan, index=target_events_idx, dtype=object)

            final_labels.iloc[valid_locs] = binary_labels.values
            final_returns.iloc[valid_locs] = subset_returns.values
            final_mfe.iloc[valid_locs] = subset_mfe.values
            final_mae.iloc[valid_locs] = subset_mae.values
            final_exit.iloc[valid_locs] = subset_exit.values
        else:
            final_labels = binary_labels
            final_returns = subset_returns
            final_mfe = subset_mfe
            final_mae = subset_mae
            final_exit = subset_exit

        result = (final_labels, final_returns, final_mfe, final_mae, final_exit)
        self._labels_cache[cache_key] = result
        return result

    def _build_geometry_independent_event_features(
        self,
        df: pd.DataFrame,
        events_df: pd.DataFrame,
        mode: str = 'full'
    ) -> pd.DataFrame:
        """
        Build feature matrix for all events.

        Args:
            df: Market Data (OHLCV)
            events_df: Events
            mode: 'full' (MTF Meta Features) or 'probe' (Basis Set only)
        """
        if mode == 'probe':
            # Use lightweight basis set for validation
            close = df['close'] if 'close' in df.columns else df['Close']

            # Check volume availability
            vol_col = None
            if 'volume' in df.columns: vol_col = 'volume'
            elif 'Volume' in df.columns: vol_col = 'Volume'

            if vol_col:
                volume = pd.to_numeric(df[vol_col], errors='coerce').fillna(0.0)
            else:
                volume = pd.Series(1.0, index=close.index)

            # Generate features (Basis Set)
            probe_feats = generate_probe_features(close, volume)

            # Align to events
            X_events = probe_feats.reindex(events_df.index).fillna(0.0)
            return X_events

        # --- FULL MODE (Production / Titan RFE) ---
        signals = pd.DataFrame(index=df.index)
        try:
            if self._primary_signals is not None and 'consensus' in self._primary_signals.columns:
                consensus = pd.to_numeric(self._primary_signals['consensus'].reindex(df.index), errors='coerce').astype(float)
                consensus = consensus.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            else:
                consensus = np.sign(df['close'].pct_change()).fillna(0.0)
                consensus = consensus.replace([np.inf, -np.inf], 0.0)
            signals['consensus'] = consensus.astype(float)
        except Exception as e:
            logger.warning(f"Error building consensus signal: {e}")
            signals['consensus'] = 0.0

        try:
            volume_available = ('volume' in df.columns) and bool(pd.to_numeric(df['volume'], errors='coerce').notna().any())
        except Exception as e:
            logger.warning(f"Error checking volume availability: {e}")
            volume_available = False

        meta_features = create_meta_features(
            df=df,
            signals=signals,
            volume_available=volume_available,
            include_raw_signals=False,
            use_kalman=True,
        )

        try:
            meta_features = meta_features.replace([np.inf, -np.inf], np.nan)
            meta_features = meta_features.apply(pd.to_numeric, errors='coerce')
        except Exception as e:
            logger.debug(f"Meta features cleanup failed: {e}")

        # 1) Align base features to events
        X_events = meta_features.reindex(events_df.index)
        try:
            X_events = X_events.replace([np.inf, -np.inf], np.nan)
        except Exception as e:
            logger.debug(f"X_events cleanup failed: {e}")
        # 2) Merge Titan-selected features (if available) before any filtering
        titan_cols = getattr(self, "_production_selected_features", None) or getattr(self, "_global_probe_features", None) or []
        titan_cols = [c for c in titan_cols if c in df.columns]
        if titan_cols:
            titan_df = df[titan_cols].reindex(events_df.index)
            titan_df = titan_df.replace([np.inf, -np.inf], np.nan)
            X_events = pd.concat([X_events, titan_df], axis=1)

        # 3) Enrich with event-specific risk geometry proxies (same as before)
        try:
            vol_event = pd.to_numeric(events_df.get('volatility_1d'), errors='coerce').astype(float).fillna(0.0)
        except Exception:
            vol_event = pd.Series(0.0, index=events_df.index, dtype=float)
        stop_sigma = vol_event * MIN_SL_PCT
        target_sigma = stop_sigma * MIN_TP_SL_RATIO

        close_event = pd.to_numeric(df['close'].reindex(events_df.index), errors='coerce').astype(float)
        high_event = pd.to_numeric(df.get('high', df['close']).reindex(events_df.index), errors='coerce').astype(float)
        low_event = pd.to_numeric(df.get('low', df['close']).reindex(events_df.index), errors='coerce').astype(float)
        price_range = (high_event - low_event).abs().replace(0.0, np.nan)

        X_events['event_stop_sigma'] = stop_sigma
        X_events['event_target_sigma'] = target_sigma
        X_events['event_stop_abs'] = (stop_sigma * close_event).fillna(0.0)
        X_events['event_target_abs'] = (target_sigma * close_event).fillna(0.0)
        X_events['stop_to_range_ratio'] = (X_events['event_stop_abs'] / (price_range + 1e-9)).fillna(0.0)
        X_events['target_to_range_ratio'] = (X_events['event_target_abs'] / (price_range + 1e-9)).fillna(0.0)

        rolling_vol = vol_event.rolling(50, min_periods=5)
        vol_pct = rolling_vol.mean() / (rolling_vol.std() + 1e-9)
        X_events['event_volatility_ratio_50'] = vol_pct.fillna(0.0)

        rolling_range = price_range.rolling(50, min_periods=5).mean()
        X_events['event_range_to_stop_ratio_50'] = (rolling_range / (X_events['event_stop_abs'] + 1e-9)).fillna(0.0)

        # 4) Standardize columns to preserve variance before filtering
        # Only standardize float columns to avoid object dtypes
        float_cols = X_events.select_dtypes(include=[np.number]).columns
        if len(float_cols) > 0:
            scaler = StandardScaler(with_mean=True, with_std=True)
            try:
                scaled = scaler.fit_transform(X_events[float_cols])
                X_events[float_cols] = scaled
            except Exception as e:
                logger.warning(f"Standardization failed: {e}")

        # 5) Fill remaining NaNs with 0 after scaling
        X_events = X_events.fillna(0.0)

        return X_events

    def _get_target_sample_weight_for_events(
        self,
        df: pd.DataFrame,
        events_df: pd.DataFrame,
    ) -> Optional[pd.Series]:
        """Align config-provided target_sample_weight (diagnostic column) to events."""
        cfg = getattr(self, '_current_config', {})
        raw = None
        try:
            raw = cfg.get('target_sample_weight') if isinstance(cfg, dict) else None
        except Exception:
            raw = None

        if raw is None:
            return None

        try:
            if isinstance(raw, pd.Series):
                w_full = raw.reindex(df.index)
            else:
                arr = np.asarray(raw, dtype=float).reshape(-1)
                if arr.shape[0] == len(df.index):
                    w_full = pd.Series(arr, index=df.index)
                elif arr.shape[0] > len(df.index):
                    w_full = pd.Series(arr[: len(df.index)], index=df.index)
                else:
                    padded = np.ones(len(df.index), dtype=float)
                    if arr.shape[0] > 0:
                        padded[: arr.shape[0]] = arr
                    w_full = pd.Series(padded, index=df.index)

            w_events = w_full.reindex(events_df.index)
            w_events = pd.to_numeric(w_events, errors='coerce').astype(float)
            w_events = w_events.replace([np.inf, -np.inf], np.nan).fillna(1.0)
            w_events = w_events.clip(lower=0.0)
            return w_events
        except Exception:
            return None

    def _robust_normalize(self, series: pd.Series) -> pd.Series:
        """
        Log-Robust Normalization: Log1p -> (x - Median) / IQR.
        """
        s_log = np.log1p(series)
        q1, q3 = s_log.quantile(0.25), s_log.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0: iqr = 1.0
        scaled = (s_log - s_log.median()) / iqr
        return scaled.clip(lower=0)

    def _compute_root_dispersion(self, df: pd.DataFrame, feature_names: List[str], decay: float = 0.7) -> pd.Series:
        """
        Cheap proxy for uniformity: cross-tree root dispersion.
        """
        if df.empty:
            return pd.Series(0.0, index=feature_names)

        # Only need tree_index, split_feature, node_depth
        splits = df[['tree_index', 'split_feature', 'node_depth']]

        # Keep only earliest appearance per tree
        min_depth = (
            splits
            .groupby(['tree_index', 'split_feature'])['node_depth']
            .min()
            .reset_index()
        )

        # Depth-weighted presence
        min_depth['w'] = np.exp(-decay * min_depth['node_depth'])

        # Aggregate per feature
        rds = min_depth.groupby('split_feature')['w'].mean()

        return rds.reindex(feature_names).fillna(0.0)

    def _select_optimal_k(
        self,
        rfe_history_df: pd.DataFrame,
        effective_n_samples: int,
        tree_depth: int = 6,
        signal_threshold: float = 0.95,
        shadow_percentile: float = 75,
        marginal_eps: float = 0.005
    ) -> List[str]:
        """
        Trident stopping rule for optimal K selection.
        """
        # --- 1. Separate & Sort (CRITICAL FIX) ---
        # Ensure shadow doesn't pollute the sorting
        shadow_rows = rfe_history_df[rfe_history_df['feature'] == 'SHADOW_NOISE']

        if shadow_rows.empty:
            # Fallback if shadow missing
            shadow_cutoff = 0.0
        else:
            # Get Shadow Threshold (Robust against single or multiple shadow entries)
            shadow_scores = shadow_rows['hafsr_score'].values
            shadow_cutoff = np.percentile(shadow_scores, shadow_percentile)

        # Filter Real Features and FORCE SORT descending
        clean_df = rfe_history_df[rfe_history_df['feature'] != 'SHADOW_NOISE'].copy()
        clean_df = clean_df.sort_values('hafsr_score', ascending=False)

        # --- 2. Prepare Scores ---
        # Clip negative scores to 0 (anti-signal shouldn't reduce cumulative sum)
        scores = clean_df['hafsr_score'].clip(lower=0).values
        total_signal = scores.sum()

        if total_signal == 0:
            logger.warning("Warning: No positive signal detected across all features.")
            return []

        # Calculate Cumulative and Marginal Signal
        cumulative_signal = np.cumsum(scores) / total_signal
        # Normalized marginal contribution of each feature
        marginal_signal = scores / total_signal

        # --- 3. Capacity Constraint (De Prado) ---
        # Formula: N / (Depth * 8) is a conservative estimate for degrees of freedom
        # We ensure a minimum floor of 5 features to prevent over-pruning on small datasets
        calculated_cap = int(effective_n_samples / (tree_depth * 8))
        max_k_capacity = max(5, calculated_cap)

        # --- 4. Trident Evaluation ---
        optimal_k = len(clean_df)
        reason = "Exhausted (Kept All)"

        for k in range(1, len(clean_df) + 1):
            idx = k - 1 # 0-based index

            current_score = scores[idx]
            current_cumulative = cumulative_signal[idx]
            current_marginal = marginal_signal[idx]

            # STOP CONDITION 1: Capacity Limit
            if k > max_k_capacity:
                optimal_k = k - 1
                reason = f"Capacity Limit (Max {max_k_capacity})"
                break

            # STOP CONDITION 2: Noise Dominance (Shadow Gap)
            # If current feature is weaker than the shadow baseline
            if current_score <= shadow_cutoff:
                optimal_k = k - 1
                reason = "Hit Noise Floor (Shadow Dominance)"
                break

            # STOP CONDITION 3: Signal Saturation (The Elbow)
            # We stop if we have enough signal (95%)
            # OR if the new feature adds virtually nothing (< 0.5%)
            # Note: We check k > 5 to ensure we don't stop too early on the "Head"
            if k > 5:
                if current_cumulative >= signal_threshold:
                    optimal_k = k
                    reason = f"Signal Saturation (>{signal_threshold:.0%})"
                    break

                if current_marginal < marginal_eps:
                    optimal_k = k - 1
                    reason = f"Marginal Decay (<{marginal_eps:.1%})"
                    break

        # Final Safety: Ensure we select at least 1 feature if signal exists
        if optimal_k < 1 and total_signal > 0:
            optimal_k = 1
            reason = "Forced Minimum (1)"

        logger.info(f"Selected K={optimal_k} | Reason: {reason}")
        if optimal_k > 0:
            logger.info(f" Cumulative Signal: {cumulative_signal[optimal_k-1]:.4f}")

        return clean_df.iloc[:optimal_k]['feature'].tolist()

    def _calculate_dynamic_score(self, model, feature_names: List[str], weights: Dict[str, float]) -> pd.Series:
        """
        Calculates feature importance using 4 weighted components:
        1. Total Gain (Volume)
        2. Avg Gain (Efficiency)
        3. Structural Gain (Primacy - Depth Weighted)
        4. Uniformity (Consistency - Gini)
        """
        try:
            # 1. Extract Tree Data (LGBM)
            # Handle both direct LGBMClassifier and wrapped objects if any
            booster = model.booster_ if hasattr(model, 'booster_') else model
            df = booster.trees_to_dataframe()
            # Filter leaves (LGBM leaves have no split feature)
            df = df[df['split_feature'].notna()]

            # 2. Basic Metrics (Volume)
            groupby = df.groupby('split_feature')['split_gain']
            total_gain = groupby.sum().reindex(feature_names).fillna(0)

            # AvgGain (Efficiency) - Only calc if needed
            if weights.get('avg', 0) > 0:
                split_count = groupby.count().reindex(feature_names).fillna(0)
                avg_gain = total_gain / (split_count + 1e-9)
            else:
                avg_gain = pd.Series(0, index=feature_names)

            # 3. Structural Gain (Primacy) - Depth Weighted
            # Filter strictly for depth <= 6 (Hard Cap)
            df_struct = df[df['node_depth'] <= 6].copy()

            if weights.get('struct', 0) > 0 and not df_struct.empty:
                # Decay rate 0.5 ^ Depth
                df_struct['w_gain'] = df_struct['split_gain'] * (0.5 ** df_struct['node_depth'])
                struct_gain = df_struct.groupby('split_feature')['w_gain'].sum().reindex(feature_names).fillna(0)
            else:
                struct_gain = pd.Series(0, index=feature_names)

            # 4. Uniformity (Consistency) - Root Dispersion Proxy
            if weights.get('uni', 0) > 0:
                uniformity = self._compute_root_dispersion(df, feature_names, decay=0.7)
            else:
                uniformity = pd.Series(0, index=feature_names)

            # 5. Normalize & Combine
            n_total = self._robust_normalize(total_gain)
            n_avg = self._robust_normalize(avg_gain)
            n_struct = self._robust_normalize(struct_gain)
            n_uni = self._robust_normalize(uniformity)

            # Scale to [0,1] range for weighted sum
            scores_df = pd.DataFrame({'T': n_total, 'A': n_avg, 'S': n_struct, 'U': n_uni})
            scaler = MinMaxScaler()
            scaled = pd.DataFrame(scaler.fit_transform(scores_df),
                                index=scores_df.index, columns=scores_df.columns)

            final_score = (weights['total'] * scaled['T'] +
                        weights['avg'] * scaled['A'] +
                        weights['struct'] * scaled['S'] +
                        weights['uni'] * scaled['U'])

            return final_score
        except Exception as e:
            logger.warning(f"Dynamic Score failed: {e}")
            return pd.Series(0.0, index=feature_names)

    def _calculate_hafsr_dynamic(self, fold_scores_df: pd.DataFrame, shadow_vals: Optional[np.ndarray], n_cv: int) -> pd.Series:
        """
        Calculates stability.
        If CV=0: Returns (RawScore - MedianShadow).
        If CV>0: Returns HAFSR (ExcessReturn / DownsideRisk).
        """
        if fold_scores_df.empty:
            return pd.Series()

        shadow_median = np.median(shadow_vals) if shadow_vals is not None else 0.0

        if n_cv == 0:
            # Single Fold Mode: Simple Hurdle
            return fold_scores_df.iloc[:, 0] - shadow_median

        # Multi-Fold Mode: Full HAFSR
        mu = fold_scores_df.median(axis=1)

        # Robust Downside MAD
        def robust_mad(row):
            med = row.median()
            neg = row[row < med] - med
            if len(neg) == 0: return 1e-6
            return np.median(np.abs(neg))

        downside = fold_scores_df.apply(robust_mad, axis=1)

        # Hurdles based on Shadow Distribution
        hurdles = np.percentile(shadow_vals, [50, 75, 90]) if shadow_vals is not None else [0]
        weights = [0.3, 0.4, 0.3]

        final = []
        for h, w in zip(hurdles, weights):
            ratio = (mu - h) / (downside + 1e-6)
            final.append(ratio * w)

        return pd.Series(np.sum(final, axis=0), index=fold_scores_df.index)

    def _calculate_absolute_rolling_correlation(self, X: pd.DataFrame, n_subsamples: int = 5) -> pd.DataFrame:
        """
        Approximates 'Absolute Rolling Correlation' by averaging absolute correlation
        matrices across N contiguous subsamples. Vectorized implementation.
        """
        if X.empty:
            return pd.DataFrame()

        n_rows, n_cols = X.shape
        if n_rows < 50:
             return X.corr().abs()

        chunk_size = n_rows // n_subsamples
        if chunk_size < 10:
             return X.corr().abs()

        # 1. Prepare 3D Tensor: (n_subsamples, chunk_size, n_features)
        # Truncate to fit perfectly
        n_truncated = chunk_size * n_subsamples
        if n_truncated < n_rows:
            X_mat = X.iloc[:n_truncated].to_numpy()
        else:
            X_mat = X.to_numpy()

        # Reshape: (S, T, F)
        try:
            X_3d = X_mat.reshape(n_subsamples, chunk_size, n_cols)

            # 2. Vectorized Correlation Calculation
            # Center: (S, T, F)
            means = X_3d.mean(axis=1, keepdims=True)
            X_centered = X_3d - means

            # Covariance: (S, F, F)
            cov = np.matmul(X_centered.transpose(0, 2, 1), X_centered) / (chunk_size - 1)

            # Std Devs: (S, F)
            stds = X_3d.std(axis=1, ddof=1)

            # Outer product of stds: (S, F, F)
            stds_outer = stds[:, :, None] * stds[:, None, :]

            # Avoid division by zero
            stds_outer[stds_outer == 0] = 1e-9

            # Correlation: (S, F, F)
            corrs = cov / stds_outer

            # Clip to valid range
            corrs = np.clip(corrs, -1.0, 1.0)

            # 3. Absolute and Average
            avg_abs_corr_mat = np.mean(np.abs(corrs), axis=0)

            # Fill NaNs
            avg_abs_corr_mat = np.nan_to_num(avg_abs_corr_mat, nan=0.0)

            return pd.DataFrame(avg_abs_corr_mat, index=X.columns, columns=X.columns)

        except Exception as e:
            logger.warning(f"Vectorized rolling correlation failed: {e}. Falling back to Pandas.")
            return X.corr().abs()

    def _cluster_and_deduplicate(
        self,
        X: pd.DataFrame,
        feature_scores: pd.Series,
        top_n: int = 150
    ) -> List[str]:
        """
        Cluster features using Average Absolute Rolling Correlation and select representatives.
        """
        if X.empty:
            return []

        # 1. Compute Distance Matrix using Absolute Rolling Correlation on subsamples
        try:
            # Subsample for correlation speedup (Use recent history to preserve time structure)
            if len(X) > 2000:
                X_corr = X.iloc[-2000:]
            else:
                X_corr = X

            avg_abs_corr = self._calculate_absolute_rolling_correlation(X_corr, n_subsamples=5)
            dist_matrix = 1.0 - avg_abs_corr.values
            dist_matrix = np.clip(dist_matrix, 0.0, 1.0)
            np.fill_diagonal(dist_matrix, 0.0)
            condensed_dist = squareform(dist_matrix, checks=False)

            # 2. Hierarchical Clustering
            Z = linkage(condensed_dist, method='average')
            cluster_labels = fcluster(Z, t=0.15, criterion='distance')
        except Exception:
            # Fallback to simple correlation
            try:
                dist_matrix = 1.0 - X.corr().abs().fillna(0.0).values
                condensed_dist = squareform(np.clip(dist_matrix, 0.0, 1.0), checks=False)
                Z = linkage(condensed_dist, method='average')
                cluster_labels = fcluster(Z, t=0.15, criterion='distance')
            except Exception:
                # Total fallback
                return list(X.columns)[:top_n]

        # 3. Select Representatives based on feature_scores (HAFSR)
        clusters: Dict[int, List[str]] = {}
        features = list(X.columns)
        for i, label in enumerate(cluster_labels):
            clusters.setdefault(label, []).append(features[i])

        selected_features = []

        for label, cluster_feats in clusters.items():
            if len(cluster_feats) == 1:
                selected_features.append(cluster_feats[0])
            else:
                # Multiple features: sort by score descending
                feats_with_scores = [
                    (f, feature_scores.get(f, -float('inf')))
                    for f in cluster_feats
                ]
                feats_with_scores.sort(key=lambda x: x[1], reverse=True)

                # Keep top 1 as representative
                winners = [feats_with_scores[0][0]]
                selected_features.extend(winners)

        # 4. Global Top N
        if len(selected_features) > top_n:
            final_scores = [
                (f, feature_scores.get(f, -float('inf')))
                for f in selected_features
            ]
            final_scores.sort(key=lambda x: x[1], reverse=True)
            selected_features = [f for f, s in final_scores[:top_n]]

        return selected_features

    def _run_titan_rfe(self, X: pd.DataFrame, y: pd.Series, cv_splits, volatility_series: pd.Series, min_features=70) -> List[str]:

        current_features = list(X.columns)

        cfg_rfe = getattr(self, "_current_config", {}) or {}
        try:
            sfi_threshold = int(cfg_rfe.get('layer2_rfe_sfi_threshold', 150))
        except Exception:
            sfi_threshold = 150
        try:
            sfi_row_frac = float(cfg_rfe.get('layer2_rfe_sfi_row_frac', 0.6))
        except Exception:
            sfi_row_frac = 0.6
        sfi_row_frac = float(min(max(sfi_row_frac, 0.1), 1.0))
        sfi_enabled = bool(cfg_rfe.get('layer2_rfe_enable_sfi', True))

        # Inject Shadow Feature (Gaussian Noise)
        X_work = X.copy()
        X_work['SHADOW_NOISE'] = np.random.normal(0, 1, size=len(X))
        current_features.append('SHADOW_NOISE')

        # State Tracking
        # 1.0 = Shadow is worse than 100% of features (Ideal state)
        # We maintain 1.0 until CV is active to avoid noisy switching.
        prev_shadow_rank_pct = 1.0

        def _num_leaves_from_depth(depth: int, buffer: int = 0) -> int:
            depth = int(max(1, depth))
            leaves = 2 ** depth
            return max(2, leaves + buffer)

        while len(current_features) > min_features:

            n_feats = len(current_features)

            # --- A. DYNAMIC CV SWITCHING ---
            # Switch to 3-Fold CV if:
            # 1. Shadow feature is beating > 40% of real features (Rank < 0.60)
            # 2. We are in the final fine-tuning stage (approaching min_features)
            if (prev_shadow_rank_pct < 0.60) or (n_feats <= 2 * min_features):
                n_cv_active = 3
            else:
                n_cv_active = 0

            # --- B. DYNAMIC WEIGHTS & CONFIG ---

            # Determine if we are in Phase 2 (Fine-tuning with SFI)
            # Use SFI only when feature count drops to <= 250
            use_sfi = sfi_enabled and (n_feats <= int(sfi_threshold))

            if use_sfi:
                # Phase 2 Weights (Rebalanced: 70% Tree / 30% SFI)
                # Tree Component (0.70 total): Total 0.30, Avg 0.15, Struct 0.15, Uni 0.10
                w = {'total': 0.30, 'avg': 0.15, 'struct': 0.15, 'uni': 0.10}
            elif n_cv_active == 0:
                # Phase 1 Early Stage (Speed Mode)
                w = {'total': 0.50, 'avg': 0.30, 'struct': 0.20, 'uni': 0.00}
            else:
                # Phase 1 Standard
                w = {'total': 0.40, 'avg': 0.25, 'struct': 0.20, 'uni': 0.15}

            # Estimators (Scale with feature count) - Reduced for speed
            n_est = max(30, int(1.5 * n_feats))

            logger.info(f"RFE: {n_feats} feats | CV: {n_cv_active} | Est: {n_est} | SFI: {use_sfi} | ShadowRank: {prev_shadow_rank_pct:.2%}")

            # --- C. TRAINING ---
            # Define Focal Loss locally for pickling safety in Parallel
            gamma_pos, gamma_neg, f_alpha = 0.5, 1.25, 0.65
            focal_obj = RobustFocalLoss(gamma_pos=gamma_pos, gamma_neg=gamma_neg, alpha=f_alpha, verbose=False)

            def lgbm_focal_obj(y_true, y_pred):
                return focal_obj(y_pred, y_true)

            # --- PRE-COMPUTE SFI ONCE (Not per fold) ---
            # This is a significant speedup: train 1 model per feature instead of N_folds * N_features
            cached_sfi_norm = None
            if use_sfi:
                logger.info(f"RFE: Pre-computing SFI for {len(current_features)} features (once, not per-fold)...")
                # Use first CV split for SFI computation
                if len(cv_splits) > 0:
                    sfi_train_idx, sfi_val_idx = cv_splits[0]
                    X_sfi_train = X_work.iloc[sfi_train_idx]
                    y_sfi_train = y.iloc[sfi_train_idx]
                    X_sfi_val = X_work.iloc[sfi_val_idx]
                    y_sfi_val = y.iloc[sfi_val_idx]

                    if sfi_row_frac < 0.999:
                        train_sample = X_sfi_train.sample(frac=sfi_row_frac, random_state=42)
                        X_sfi_train = train_sample
                        y_sfi_train = y_sfi_train.loc[train_sample.index]
                        val_sample = X_sfi_val.sample(frac=sfi_row_frac, random_state=42)
                        X_sfi_val = val_sample
                        y_sfi_val = y_sfi_val.loc[val_sample.index]

                    # Inverse Volatility Weighting for SFI
                    vol_sfi = volatility_series.iloc[sfi_train_idx]
                    if sfi_row_frac < 0.999:
                        vol_sfi = vol_sfi.loc[y_sfi_train.index]
                    weights_sfi = 1.0 / (vol_sfi + 1e-5)
                    weights_sfi = weights_sfi.clip(upper=weights_sfi.quantile(0.99))

                    sfi_params = {
                        'objective': lgbm_focal_obj,
                        'n_estimators': 50,  # Reduced from 100 for speed
                        'max_depth': 2,
                        'num_leaves': 4,
                        'learning_rate': 0.05,
                        'subsample': 0.8,
                        'colsample_bytree': 1.0,
                        'colsample_bynode': 1.0,
                        'reg_alpha': 0.5,
                        'reg_lambda': 0.5,
                        'n_jobs': 1,
                        'verbose': -1,
                        'random_state': 42
                    }
                    sfi_model = lgb.LGBMClassifier(**sfi_params)
                    
                    sfi_scores = []
                    for feat in current_features:
                        try:
                            feat_train = X_sfi_train[[feat]]
                            sfi_model.fit(feat_train, y_sfi_train, sample_weight=weights_sfi)
                            feat_val = X_sfi_val[[feat]]
                            raw_preds = sfi_model.predict(feat_val, raw_score=True)
                            probs = expit(raw_preds)
                            loss = log_loss(y_sfi_val, probs, labels=[0, 1], eps=1e-15)
                            sfi_scores.append(-loss)
                        except:
                            sfi_scores.append(-10.0)
                    
                    sfi_series = pd.Series(sfi_scores, index=current_features).fillna(-10.0)
                    sfi_gain = np.exp(sfi_series)
                    sfi_robust = self._robust_normalize(sfi_gain)
                    scaler_sfi = MinMaxScaler()
                    sfi_norm_vals = scaler_sfi.fit_transform(sfi_robust.values.reshape(-1, 1)).flatten()
                    cached_sfi_norm = pd.Series(sfi_norm_vals, index=current_features)

            def run_fold_process(train_idx, val_idx, features, sfi_norm_cache):
                X_fold = X_work.iloc[train_idx]
                y_fold = y.iloc[train_idx]

                # 1. Subsampling (Speed optimization for large sets)
                if n_feats > 3 * min_features:
                    X_fold = X_fold.sample(frac=0.5, random_state=42)
                    y_fold = y_fold.loc[X_fold.index]

                # 2. Inverse Volatility Weighting
                vol = volatility_series.loc[X_fold.index]
                weights = 1.0 / (vol + 1e-5)
                weights = weights.clip(upper=weights.quantile(0.99))

                # 3. Model Fit (LGBM Optimized with RobustFocalLoss)
                max_depth = 6
                model = lgb.LGBMClassifier(
                    objective=lgbm_focal_obj,
                    n_estimators=n_est,
                    max_depth=max_depth,
                    num_leaves=_num_leaves_from_depth(max_depth),
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    colsample_bynode=0.8,
                    reg_alpha=0.2,
                    reg_lambda=0.05,
                    n_jobs=1,
                    verbose=-1,
                    random_state=42
                )

                model.fit(X_fold[features], y_fold, sample_weight=weights)

                # Get Tree-based Score (MDI-based)
                tree_score = self._calculate_dynamic_score(model, features, w)

                # Combine with cached SFI (no retraining per fold)
                if sfi_norm_cache is not None and use_sfi:
                    # Align SFI to current features (some may have been dropped)
                    sfi_aligned = sfi_norm_cache.reindex(features).fillna(0.0)
                    final_combined = tree_score + 0.3 * sfi_aligned
                else:
                    # Phase 1: Only Tree Score (weights sum to 1.0)
                    final_combined = tree_score

                return final_combined

            # Execute
            if n_cv_active == 0:
                # Single Split (Fast)
                if len(cv_splits) > 0:
                    tr, val = cv_splits[0]
                    fold_res = [run_fold_process(tr, val, current_features, cached_sfi_norm)]
                else:
                    logger.warning("No CV splits available for RFE")
                    break
            else:
                # Parallel 3-Fold with timeout guard
                parallel_kwargs = self._get_parallel_kwargs()

                def _rfe_parallel_task():
                    return Parallel(**parallel_kwargs)(
                        delayed(run_fold_process)(tr, val, current_features, cached_sfi_norm)
                        for tr, val in cv_splits[:3]
                    )

                fold_res = self._run_parallel_with_timeout(
                    _rfe_parallel_task,
                    context="titan_rfe_parallel"
                )

            fold_df = pd.concat(fold_res, axis=1)

            # --- D. STABILITY & SELECTION (CORRECTED) ---

            # Extract Shadow
            if 'SHADOW_NOISE' in fold_df.index:
                # Explicit asarray to handle potential scalar return
                shadow_vals = np.asarray(fold_df.loc['SHADOW_NOISE'].values)
                candidates = fold_df.drop('SHADOW_NOISE')
            else:
                shadow_vals = np.array([0.0])
                candidates = fold_df

            # Calc Stability (HAFSR or Raw Score)
            stability = self._calculate_hafsr_dynamic(
                candidates,
                shadow_vals,
                n_cv_active
            )

            # UPDATE SHADOW RANK (Only when CV is active)
            if n_cv_active > 0:
                shadow_score = np.median(shadow_vals)

                # Percent of real features that are WORSE than shadow
                # Note: We compare directly against shadow_score per user specs
                worse_than_shadow = (stability < shadow_score).sum()
                prev_shadow_rank_pct = worse_than_shadow / len(stability)
            # Else: keep previous shadow rank (do NOT update in noisy CV=0 mode)

            # Geometric Drop
            ranked = stability.sort_values(ascending=False)
            # RFE: remove 50% features per round
            n_keep = max(min_features, int(len(ranked) * 0.50))

            keep_feats = ranked.index[:n_keep].tolist()
            keep_feats.append('SHADOW_NOISE')
            current_features = keep_feats

        # --- E. TRIDENT OPTIMAL K SELECTION ---
        # At this point, we have ~min_features left (e.g. 70).
        # We need to pick the "optimal" subset from these survivors.

        # Reconstruct history dataframe from the final stability scores
        if 'stability' in locals():
            rfe_history = stability.reset_index()
            rfe_history.columns = ['feature', 'hafsr_score']

            # Run Trident
            selected_features = self._select_optimal_k(
                rfe_history,
                effective_n_samples=len(y),
                tree_depth=6
            )
            return selected_features
        else:
            return [f for f in current_features if f != 'SHADOW_NOISE']

    def _aggregate_geometry_labels_for_feature_selection(
        self,
        df: pd.DataFrame,
        events_df: pd.DataFrame,
        geometries: List[GeometryTrial],
    ) -> pd.Series:
        if events_df is None or getattr(events_df, 'empty', True) or not geometries:
            return pd.Series(np.nan, index=getattr(events_df, 'index', pd.Index([])), dtype=float)

        events_local = events_df
        if 'family' not in events_local.columns:
            try:
                events_local = events_local.copy()
                events_local['family'] = self._assign_barrier_families(events_local)
            except Exception:
                events_local = events_df

        sum_w = pd.Series(0.0, index=events_local.index, dtype=float)
        sum_lbl = pd.Series(0.0, index=events_local.index, dtype=float)

        for g in list(geometries):
            try:
                fam = str(getattr(g, 'family', ''))
                if 'family' in events_local.columns:
                    fam_events = events_local[events_local['family'] == fam]
                else:
                    fam_events = events_local
                if fam_events.empty:
                    continue

                lbls = self._get_cached_geometry_labels(df, fam_events, fam, getattr(g, 'params', {}))
                valid = lbls.notna()
                if not bool(valid.any()):
                    continue

                w_g = float(getattr(g, 'final_score', 1.0))
                if (not np.isfinite(w_g)) or w_g <= 0.0:
                    w_g = 1.0

                idx = lbls.index[valid]
                sum_lbl.loc[idx] = sum_lbl.loc[idx] + (w_g * lbls.loc[idx])
                sum_w.loc[idx] = sum_w.loc[idx] + float(w_g)
            except Exception:
                continue

        y_soft = pd.Series(np.nan, index=events_local.index, dtype=float)
        valid_w = sum_w > 0.0
        if bool(valid_w.any()):
            y_soft.loc[valid_w] = (sum_lbl.loc[valid_w] / sum_w.loc[valid_w]).astype(float)

        y_bin = pd.Series(np.nan, index=events_local.index, dtype=float)
        try:
            y_bin.loc[valid_w] = (y_soft.loc[valid_w] >= 0.5).astype(float)
        except Exception:
            pass
        return y_bin

    def _select_supervised_features_for_events(
        self,
        X_events_full: pd.DataFrame,
        y_target: pd.Series,
        layer1_weight_events: Optional[pd.Series],
        volatility_series: Optional[pd.Series] = None
    ) -> List[str]:
        if X_events_full is None or getattr(X_events_full, 'empty', True) or y_target is None:
            return []

        valid = y_target.notna()
        try:
            n_valid = int(valid.sum())
        except Exception:
            n_valid = 0
        if n_valid < 100:
            return []

        y_clean = pd.to_numeric(y_target.loc[valid], errors='coerce').astype(float)
        if int(y_clean.nunique()) < 2:
            return []

        X_clean = X_events_full.loc[valid].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        cfg = getattr(self, '_current_config', {})
        if not isinstance(cfg, dict):
            cfg = {}

        try:
            target_n = int(cfg.get('layer2_supervised_feature_count', 50))
        except Exception:
            target_n = 50

        try:
            max_rows = int(cfg.get('layer2_supervised_feature_max_rows', 4000))  # Reduced from 8000 for speed
        except Exception:
            max_rows = 4000
        if max_rows > 0 and len(X_clean) > max_rows:
            sampled_idx = self._maybe_sample_indices(X_clean.index, max_rows)
            X_clean = X_clean.loc[sampled_idx]
            y_clean = y_clean.loc[sampled_idx]

        w_series = None
        if layer1_weight_events is not None:
            try:
                w_series = pd.to_numeric(layer1_weight_events.reindex(X_clean.index), errors='coerce').astype(float)
                w_series = w_series.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(lower=0.0)
            except Exception:
                w_series = None
        w_arr = w_series.to_numpy(dtype=float, copy=False) if w_series is not None else None

        # Volatility for Inverse Vol Weighting
        if volatility_series is not None:
             vol_s = volatility_series.reindex(X_clean.index).fillna(0.0)
        else:
             vol_s = pd.Series(1.0, index=X_clean.index)

        # 1. HAFSR Pre-Ranking for Clustering
        n_splits = 2 # Reduced from 3 for speed
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Robust Focal Loss for consistency with trading models
        gamma_pos, gamma_neg, f_alpha = 0.5, 1.25, 0.65
        focal_obj = RobustFocalLoss(gamma_pos=gamma_pos, gamma_neg=gamma_neg, alpha=f_alpha, verbose=False)

        def lgbm_focal_obj(y_true, y_pred):
            return focal_obj(y_pred, y_true)

        # We need a base model for scoring
        base_model = lgb.LGBMClassifier(
            objective=lgbm_focal_obj,
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            reg_alpha=0.2, # L1
            reg_lambda=0.05, # L2
            colsample_bytree=0.85,
            random_state=self.random_state,
            n_jobs=1,
            verbose=-1
        )

        # Calculate initial HAFSR scores to pick cluster winners
        # Reuse process_fold logic from Titan but just one pass
        def process_fold_initial(tr_idx, val_idx, feats):
            X_tr = X_clean.iloc[tr_idx]
            y_tr = y_clean.iloc[tr_idx]

            # Inv Vol Weights
            vol_tr = vol_s.iloc[tr_idx]
            inv_vol = 1.0 / (vol_tr + 1e-5)
            inv_vol = np.clip(inv_vol, 0.0, np.quantile(inv_vol, 0.99))

            w_tr = inv_vol
            if w_arr is not None:
                w_tr = w_arr[tr_idx] * inv_vol

            m = clone(base_model)
            m.fit(X_tr, y_tr, sample_weight=w_tr)
            # Default weights for initial ranking (Balanced)
            w_score = {'total': 0.40, 'avg': 0.25, 'struct': 0.20, 'uni': 0.15}
            return self._calculate_dynamic_score(m, feats, w_score)

        current_features = list(X_clean.columns)

        # Cache key (reusable when dataset/signature repeats)
        cache_key = self._make_feature_cache_key(
            current_features,
            len(X_clean),
            target_n,
            extra_token=self._hash_series_signature(y_clean)
        )
        if cache_key in self._feature_selection_cache:
            cached = self._feature_selection_cache[cache_key]
            return [c for c in cached if c in X_events_full.columns]

        # Lightweight surrogate gate to avoid expensive Titan runs on weak targets
        surrogate_passed = self._passes_surrogate_gate(X_clean, y_clean, w_series, cfg)

        try:
            fold_results = Parallel(n_jobs=getattr(self, 'n_jobs', -1), prefer="threads")(
                delayed(process_fold_initial)(tr, val, current_features) for tr, val in tscv.split(X_clean)
            )
            fold_df = pd.concat(fold_results, axis=1)
            hafsr_scores = self._calculate_hafsr_dynamic(fold_df, shadow_vals=None, n_cv=n_splits)
        except Exception as e:
            logger.warning(f"Initial HAFSR calculation failed: {e}")
            hafsr_scores = pd.Series(0.0, index=current_features)

        # 2. Cluster and Deduplicate (Using Avg Abs Rolling Corr + HAFSR)
        # We select ~ 2*target_n to pass to Titan RFE, or just reduce redundancy?
        # "final selection: use the code below ... until Top 70"
        # The prompt says "2/ within each cluster: choose the feature... 3/ final selection: Titan"
        # So Clustering is a filter.
        # Let's keep 150 features from Clustering, then Titan reduces to 70.

        try:
            cluster_multiplier = float(cfg.get('layer2_cluster_multiplier', 1.5))
        except Exception:
            cluster_multiplier = 1.5
        try:
            cluster_cap = int(cfg.get('layer2_cluster_top_n', 140))
        except Exception:
            cluster_cap = 140
        intermediate_n = int(
            min(
                len(current_features),
                max(target_n, int(target_n * cluster_multiplier), cluster_cap)
            )
        )

        selected_from_clusters = self._cluster_and_deduplicate(
            X_clean,
            hafsr_scores,
            top_n=int(intermediate_n)
        )
        if not selected_from_clusters:
            return []

        # Optional fast exit when surrogate fails: keep tight, cheap subset
        if not surrogate_passed:
            fast_subset = self._cheap_corr_prune(
                X_clean[selected_from_clusters],
                selected_from_clusters,
                target_n=target_n,
                corr_threshold=0.90,
                max_rows=1500
            )
            self._feature_selection_cache[cache_key] = list(fast_subset)
            return [c for c in fast_subset if c in X_events_full.columns]

        try:
            pre_rfe_pool = int(cfg.get('layer2_rfe_initial_pool', 80))  # Reduced from 110 for speed
        except Exception:
            pre_rfe_pool = 80
        pre_rfe_candidates = self._cheap_corr_prune(
            X_clean[selected_from_clusters],
            selected_from_clusters,
            target_n=min(pre_rfe_pool, max(target_n, 100)),
            corr_threshold=float(cfg.get('layer2_rfe_corr_threshold', 0.92)),
            max_rows=int(cfg.get('layer2_rfe_corr_max_rows', 2500))
        )
        if not pre_rfe_candidates:
            pre_rfe_candidates = selected_from_clusters

        # 3. Run Titan RFE
        # Convert generator to list for reuse in dynamic CV
        cv_splits = list(tscv.split(X_clean))

        final_features = self._run_titan_rfe(
             X_clean[pre_rfe_candidates],
             y_clean,
             cv_splits,
             vol_s,
             min_features=target_n
        )

        final_features = [c for c in final_features if c in X_events_full.columns]
        self._feature_selection_cache[cache_key] = list(final_features)
        return final_features

    def _subsample_rows_for_proxy(self, df: pd.DataFrame, max_rows: int, seed: int = 42) -> pd.DataFrame:
        if max_rows <= 0:
            return df
        n_rows = len(df)
        if n_rows <= max_rows:
            return df
        rng = np.random.default_rng(seed)
        sample_idx = rng.choice(n_rows, size=max_rows, replace=False)
        return df.iloc[sample_idx]

    def _cheap_corr_prune(
        self,
        X: pd.DataFrame,
        ranked_features: List[str],
        target_n: int = 70,
        corr_threshold: float = 0.85,
        max_rows: int = 2000,
    ) -> List[str]:
        sorted_cols = [c for c in ranked_features if c in X.columns]
        if not sorted_cols:
            return []

        df_valid = X[sorted_cols].copy().fillna(0.0)
        df_sample = self._subsample_rows_for_proxy(df_valid, max_rows=max_rows, seed=42)
        try:
            corr_matrix = df_sample.corr().abs()
        except Exception:
            return sorted_cols[:target_n]

        cols = list(corr_matrix.columns)
        corr_arr = corr_matrix.to_numpy(copy=False)
        col_to_idx = {c: i for i, c in enumerate(cols)}

        selected_idx: List[int] = []
        selected_features: List[str] = []
        for col in sorted_cols:
            if len(selected_features) >= target_n:
                break

            i = col_to_idx.get(col)
            if i is None:
                continue

            if selected_idx:
                if bool(np.any(corr_arr[i, selected_idx] > float(corr_threshold))):
                    continue

            selected_idx.append(i)
            selected_features.append(col)

        return selected_features

    def _train_probes(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
        trial: Optional[optuna.Trial] = None,
    ) -> Dict[str, float]:
        """
        Step 4: Cheap ML learnability probes.
        Train Shallow LGBM and Linear Model.
        """
        valid = y.notna()
        X_clean = X.loc[valid].fillna(0.0)
        y_clean = y.loc[valid].astype(int)

        w_clean = None
        if sample_weight is not None:
            try:
                w_arr = np.asarray(sample_weight, dtype=float).reshape(-1)
                if w_arr.shape[0] == int(valid.sum()):
                    w_clean = w_arr
            except Exception:
                w_clean = None

        if len(y_clean) < 50 or y_clean.nunique() < 2:
            return {'auc': 0.5, 'log_loss': 1.0, 'ic': 0.0, 'passed': False}

        # --- OPTIMIZATION: Sampling ---
        try:
            sample_rate = float(getattr(self, '_current_config', {}).get('layer2_probe_sampling_rate', 1.0))
        except Exception:
            sample_rate = 1.0
            
        if sample_rate < 1.0 and len(y_clean) > 200:
            step = int(1.0 / sample_rate)
            X_clean = X_clean.iloc[::step]
            y_clean = y_clean.iloc[::step]
            if w_clean is not None:
                w_clean = w_clean[::step]
                
        # --- OPTIMIZATION: Feature Limit ---
        try:
             feat_limit = int(getattr(self, '_current_config', {}).get('layer2_probe_feature_limit', 0))
        except Exception:
             feat_limit = 0
             
        if feat_limit > 0 and X_clean.shape[1] > feat_limit:
             X_clean = X_clean.iloc[:, :feat_limit]

        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        # Models
        lgbm = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            num_leaves=31,
            learning_rate=0.1,
            verbose=-1,
            random_state=self.random_state,
            n_jobs=1
        )

        linear = LinearRegression(n_jobs=1)

        scaler = StandardScaler()

        metrics = {
            'lgbm_auc': [], 'lgbm_ic': [], 'lgbm_ll': [], 'lgbm_pr': [],
            'lin_auc': [], 'lin_ic': [], 'lin_ll': [], 'lin_pr': []
        }

        fold_idx = 0
        try:
            # --- OPTIMIZATION CONFIG ---
            try:
                linear_only_auc = float(getattr(self, '_current_config', {}).get('layer2_probe_linear_only_auc', 0.65))
            except Exception:
                linear_only_auc = 0.65
            
            for train_index, test_index in tscv.split(X_clean):
                X_train, X_test = X_clean.iloc[train_index], X_clean.iloc[test_index]
                y_train, y_test = y_clean.iloc[train_index], y_clean.iloc[test_index]

                if y_train.nunique() < 2 or y_test.nunique() < 2:
                    continue

                # --- OPTIMIZATION: Linear First ---
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                if w_clean is not None:
                    linear.fit(X_train_scaled, y_train, sample_weight=w_clean[train_index])
                else:
                    linear.fit(X_train_scaled, y_train)
                
                raw_scores = linear.predict(X_test_scaled)
                raw_scores = np.asarray(raw_scores, dtype=float)
                raw_scores = np.clip(raw_scores, -20.0, 20.0)
                p_linear = expit(raw_scores)
                p_linear = np.clip(np.asarray(p_linear, dtype=float), 1e-6, 1.0 - 1e-6)

                sw_te = w_clean[test_index] if w_clean is not None else None
                try:
                    auc_lin = roc_auc_score(y_test, p_linear, sample_weight=sw_te) if sw_te is not None else roc_auc_score(y_test, p_linear)
                    metrics['lin_auc'].append(float(auc_lin))
                except Exception:
                    auc_lin = 0.5

                try:
                    ll_lin = log_loss(y_test, p_linear, sample_weight=sw_te) if sw_te is not None else log_loss(y_test, p_linear)
                    metrics['lin_ll'].append(float(ll_lin))
                except Exception:
                    pass

                try:
                    ic_lin, _ = spearmanr(y_test, p_linear)
                    metrics['lin_ic'].append(float(ic_lin) if np.isfinite(ic_lin) else 0.0)
                except Exception:
                    pass

                try:
                    pr_lin = average_precision_score(y_test, p_linear, sample_weight=sw_te) if sw_te is not None else average_precision_score(y_test, p_linear)
                    metrics['lin_pr'].append(float(pr_lin))
                except Exception:
                    pass

                # Skip LGBM if Linear is already very good OR if it's very bad in first fold
                skip_lgbm = (auc_lin >= linear_only_auc) or (fold_idx == 0 and auc_lin < 0.48)

                if not skip_lgbm:
                    n_train = int(len(X_train))
                    val_n = int(max(10, min(int(np.floor(0.2 * n_train)), n_train - 1)))
                    use_es = bool(val_n >= 10 and n_train - val_n >= 10)

                    if use_es:
                        X_tr2 = X_train.iloc[:-val_n]
                        y_tr2 = y_train.iloc[:-val_n]
                        X_val2 = X_train.iloc[-val_n:]
                        y_val2 = y_train.iloc[-val_n:]
                        if y_tr2.nunique() < 2 or y_val2.nunique() < 2:
                            use_es = False

                    if w_clean is not None and use_es:
                        w_tr2 = w_clean[train_index][:-val_n]
                        lgbm.fit(
                            X_tr2, y_tr2,
                            sample_weight=w_tr2,
                            eval_set=[(X_val2, y_val2)],
                            callbacks=[lgb.early_stopping(10, verbose=False)]
                        )
                    elif w_clean is not None:
                        lgbm.fit(
                            X_train, y_train,
                            sample_weight=w_clean[train_index],
                        )
                    elif use_es:
                        lgbm.fit(
                            X_tr2, y_tr2,
                            eval_set=[(X_val2, y_val2)],
                            callbacks=[lgb.early_stopping(10, verbose=False)]
                        )
                    else:
                        lgbm.fit(
                            X_train, y_train,
                        )
                    p_lgbm = lgbm.predict_proba(X_test)[:, 1]
                    p_lgbm = np.clip(np.asarray(p_lgbm, dtype=float), 1e-6, 1.0 - 1e-6)

                    try:
                        auc_val = roc_auc_score(y_test, p_lgbm, sample_weight=sw_te) if sw_te is not None else roc_auc_score(y_test, p_lgbm)
                        metrics['lgbm_auc'].append(auc_val)
                        metrics['lgbm_ll'].append(log_loss(y_test, p_lgbm, sample_weight=sw_te) if sw_te is not None else log_loss(y_test, p_lgbm))
                        ic, _ = spearmanr(y_test, p_lgbm)
                        metrics['lgbm_ic'].append(ic if not np.isnan(ic) else 0.0)

                        try:
                            pr_val = average_precision_score(y_test, p_lgbm, sample_weight=sw_te) if sw_te is not None else average_precision_score(y_test, p_lgbm)
                            metrics['lgbm_pr'].append(float(pr_val))
                        except Exception:
                            pass
                        
                        if trial is not None:
                            trial.report(auc_val, step=fold_idx)
                            if trial.should_prune():
                                raise optuna.TrialPruned()
                    except optuna.TrialPruned:
                        raise
                    except Exception:
                        pass
                else:
                    # Sync metrics or report Linear AUC to Optuna if skipping LGBM
                    if trial is not None:
                        trial.report(auc_lin, step=fold_idx)
                        if trial.should_prune():
                            raise optuna.TrialPruned()

                # --- OPTIMIZATION: Tiered Folding (Early Exit) ---
                # If after 2 folds the performance is clearly not promising, stop.
                if fold_idx == 1:
                    current_avg = np.mean(metrics['lgbm_auc'] if metrics['lgbm_auc'] else metrics['lin_auc'])
                    if current_avg < 0.515:
                        break

                fold_idx += 1

        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.warning(f"Probe failure: {e}")
            return {'auc': 0.5, 'log_loss': 1.0, 'ic': 0.0, 'passed': False}

        if not metrics['lgbm_auc'] and not metrics['lin_auc']:
            return {'auc': 0.5, 'log_loss': 1.0, 'ic': 0.0, 'passed': False}

        auc_lgbm = np.asarray(metrics['lgbm_auc'], dtype=float)
        auc_lin = np.asarray(metrics['lin_auc'], dtype=float)
        pr_lgbm = np.asarray(metrics.get('lgbm_pr') or [], dtype=float)
        pr_lin = np.asarray(metrics.get('lin_pr') or [], dtype=float)

        avg_auc_lgbm = float(np.mean(auc_lgbm)) if auc_lgbm.size else float('nan')
        avg_auc_linear = float(np.mean(auc_lin)) if auc_lin.size else float('nan')

        avg_ic_lgbm = float(np.mean(np.asarray(metrics.get('lgbm_ic') or [], dtype=float))) if metrics.get('lgbm_ic') else float('nan')
        avg_ic_linear = float(np.mean(np.asarray(metrics.get('lin_ic') or [], dtype=float))) if metrics.get('lin_ic') else float('nan')

        avg_ll_lgbm = float(np.mean(np.asarray(metrics.get('lgbm_ll') or [], dtype=float))) if metrics.get('lgbm_ll') else float('nan')
        avg_ll_linear = float(np.mean(np.asarray(metrics.get('lin_ll') or [], dtype=float))) if metrics.get('lin_ll') else float('nan')

        auc_pool = []
        if np.isfinite(avg_auc_lgbm):
            auc_pool.append(float(avg_auc_lgbm))
        if np.isfinite(avg_auc_linear):
            auc_pool.append(float(avg_auc_linear))
        final_auc = float(np.median(auc_pool)) if auc_pool else 0.5
        auc_std = float(np.std(np.concatenate([auc_lgbm, auc_lin])) if (auc_lgbm.size + auc_lin.size) > 0 else float('nan'))

        # PR-AUC baseline is the positive class rate.
        try:
            pos_rate = float(y_clean.mean())
        except Exception:
            pos_rate = float('nan')
        pr_baseline = float(pos_rate) if np.isfinite(pos_rate) else float('nan')
        pr_best = float('nan')
        try:
            pr_pool = []
            if pr_lgbm.size:
                pr_pool.append(float(np.mean(pr_lgbm)))
            if pr_lin.size:
                pr_pool.append(float(np.mean(pr_lin)))
            if pr_pool:
                pr_best = float(np.median(pr_pool))
        except Exception:
            pr_best = float('nan')

        try:
            auc_thr = 0.52
        except Exception:
            auc_thr = 0.53
        try:
            pr_margin = float(getattr(self, '_current_config', {}).get('layer2_probe_pr_margin', 0.01))
        except Exception:
            pr_margin = 0.01
        pr_thr = float(pr_baseline + pr_margin) if np.isfinite(pr_baseline) else float('nan')

        passed_auc = bool(np.isfinite(final_auc) and (final_auc >= float(auc_thr)))
        passed_pr = bool((not np.isfinite(pr_thr)) or (np.isfinite(pr_best) and (pr_best >= pr_thr)))
        passed = bool(passed_auc and passed_pr)

        ic_pool = []
        if np.isfinite(avg_ic_lgbm):
            ic_pool.append(float(avg_ic_lgbm))
        if np.isfinite(avg_ic_linear):
            ic_pool.append(float(avg_ic_linear))
        ll_pool = []
        if np.isfinite(avg_ll_lgbm):
            ll_pool.append(float(avg_ll_lgbm))
        if np.isfinite(avg_ll_linear):
            ll_pool.append(float(avg_ll_linear))

        return {
            'auc': final_auc,
            'auc_std': auc_std,
            'pr_auc': pr_best,
            'pr_auc_baseline': pr_baseline,
            'ic': float(np.mean(ic_pool)) if ic_pool else 0.0,
            'log_loss': float(np.mean(ll_pool)) if ll_pool else 1.0,
            'auc_lgbm': float(avg_auc_lgbm) if np.isfinite(avg_auc_lgbm) else float('nan'),
            'auc_lgbm_light': float(avg_auc_lgbm) if np.isfinite(avg_auc_lgbm) else float('nan'),
            'auc_linear': float(avg_auc_linear) if np.isfinite(avg_auc_linear) else float('nan'),
            'pr_auc_lgbm': float(np.mean(pr_lgbm)) if pr_lgbm.size else float('nan'),
            'pr_auc_linear': float(np.mean(pr_lin)) if pr_lin.size else float('nan'),
            'passed': passed,
        }

    def _train_full_lgbm_probe(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        valid = y.notna()
        X_clean = X.loc[valid].fillna(0.0)
        y_clean = y.loc[valid].astype(int)

        w_clean = None
        if sample_weight is not None:
            try:
                w_arr = np.asarray(sample_weight, dtype=float).reshape(-1)
                if w_arr.shape[0] == int(valid.sum()):
                    w_clean = w_arr
            except Exception:
                w_clean = None

        if len(y_clean) < 100 or y_clean.nunique() < 2:
            return {'auc_full': 0.5, 'auc_std_full': float('nan'), 'pr_auc_full': float('nan'), 'ic_full': 0.0, 'log_loss_full': 1.0}

        try:
            cfg = getattr(self, '_current_config', {})
            if not isinstance(cfg, dict):
                cfg = {}
        except Exception:
            cfg = {}

        try:
            sample_rate = float(cfg.get('layer2_full_probe_sampling_rate', 1.0))
        except Exception:
            sample_rate = 1.0

        if sample_rate < 1.0 and len(y_clean) > 400:
            step = int(max(1, np.floor(1.0 / max(1e-9, sample_rate))))
            X_clean = X_clean.iloc[::step]
            y_clean = y_clean.iloc[::step]
            if w_clean is not None:
                w_clean = w_clean[::step]

        try:
            feat_limit = int(cfg.get('layer2_full_probe_feature_limit', 0))
        except Exception:
            feat_limit = 0
        if feat_limit > 0 and X_clean.shape[1] > feat_limit:
            X_clean = X_clean.iloc[:, :feat_limit]

        params_default = {
            'n_estimators': 2000,
            'learning_rate': 0.05,
            'max_depth': 7,
            'num_leaves': 63,
            'min_data_in_leaf': 20,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.9,
            'bagging_freq': 1,
            'lambda_l1': 0.1,
            'lambda_l2': 1.0,
            'min_gain_to_split': 0.005,
            'verbose': -1,
            'random_state': int(getattr(self, 'random_state', 42)),
            'n_jobs': 1,
        }
        try:
            params_cfg = cfg.get('layer2_full_probe_params')
            if isinstance(params_cfg, dict) and params_cfg:
                params_default.update({k: v for k, v in params_cfg.items()})
        except Exception:
            pass

        tscv = TimeSeriesSplit(n_splits=int(max(2, getattr(self, 'n_splits', 3))))

        aucs: List[float] = []
        prs: List[float] = []
        ics: List[float] = []
        lls: List[float] = []

        for train_index, test_index in tscv.split(X_clean):
            X_train, X_test = X_clean.iloc[train_index], X_clean.iloc[test_index]
            y_train, y_test = y_clean.iloc[train_index], y_clean.iloc[test_index]
            if y_train.nunique() < 2 or y_test.nunique() < 2:
                continue

            sw_te = w_clean[test_index] if w_clean is not None else None

            n_train = int(len(X_train))
            val_n = int(max(20, min(int(np.floor(0.2 * n_train)), n_train - 1)))
            use_es = bool(val_n >= 20 and n_train - val_n >= 20)

            if use_es:
                X_tr2 = X_train.iloc[:-val_n]
                y_tr2 = y_train.iloc[:-val_n]
                X_val2 = X_train.iloc[-val_n:]
                y_val2 = y_train.iloc[-val_n:]
                if y_tr2.nunique() < 2 or y_val2.nunique() < 2:
                    use_es = False

            model = lgb.LGBMClassifier(**params_default)

            if w_clean is not None and use_es:
                w_tr2 = w_clean[train_index][:-val_n]
                model.fit(
                    X_tr2, y_tr2,
                    sample_weight=w_tr2,
                    eval_set=[(X_val2, y_val2)],
                    eval_metric='auc',
                    callbacks=[lgb.early_stopping(30, verbose=False)],
                )
            elif w_clean is not None:
                model.fit(X_train, y_train, sample_weight=w_clean[train_index])
            elif use_es:
                model.fit(
                    X_tr2, y_tr2,
                    eval_set=[(X_val2, y_val2)],
                    eval_metric='auc',
                    callbacks=[lgb.early_stopping(30, verbose=False)],
                )
            else:
                model.fit(X_train, y_train)

            p = model.predict_proba(X_test)[:, 1]
            p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)

            try:
                aucs.append(float(roc_auc_score(y_test, p, sample_weight=sw_te) if sw_te is not None else roc_auc_score(y_test, p)))
            except Exception:
                pass
            try:
                lls.append(float(log_loss(y_test, p, sample_weight=sw_te) if sw_te is not None else log_loss(y_test, p)))
            except Exception:
                pass
            try:
                pr = average_precision_score(y_test, p, sample_weight=sw_te) if sw_te is not None else average_precision_score(y_test, p)
                prs.append(float(pr))
            except Exception:
                pass
            try:
                ic, _ = spearmanr(y_test, p)
                ics.append(float(ic) if np.isfinite(ic) else 0.0)
            except Exception:
                pass

        if not aucs:
            return {'auc_full': 0.5, 'auc_std_full': float('nan'), 'pr_auc_full': float('nan'), 'ic_full': 0.0, 'log_loss_full': 1.0}

        auc_arr = np.asarray(aucs, dtype=float)
        pr_arr = np.asarray(prs, dtype=float)

        return {
            'auc_full': float(np.mean(auc_arr)),
            'auc_std_full': float(np.std(auc_arr)) if auc_arr.size else float('nan'),
            'pr_auc_full': float(np.mean(pr_arr)) if pr_arr.size else float('nan'),
            'ic_full': float(np.mean(np.asarray(ics, dtype=float))) if ics else 0.0,
            'log_loss_full': float(np.mean(np.asarray(lls, dtype=float))) if lls else 1.0,
        }

    def _check_stability(
        self,
        df: pd.DataFrame,
        events_df: pd.DataFrame,
        trial_params: Dict[str, Any],
        base_score: float,
        family: str
    ) -> bool:
        """
        Stability check (Time-Flip):
        Perturb start time by ±1 bar.
        If labels flip frequently, discard.
        """
        # 1. Base Labels
        base_labels, _, _, _, _ = self._compute_dominance_labels(df, events_df, family=family, **trial_params)

        # 2. Shifted Labels (+1 bar)
        # Using events_shift=1
        shift1_labels, _, _, _, _ = self._compute_dominance_labels(
            df, events_df, family=family, events_shift=1, **trial_params
        )

        # 3. Shifted Labels (-1 bar)
        # Using events_shift=-1
        shift_neg1_labels, _, _, _, _ = self._compute_dominance_labels(
             df, events_df, family=family, events_shift=-1, **trial_params
        )

        # Align
        idx = base_labels.dropna().index

        b = base_labels.reindex(idx)
        s1 = shift1_labels.reindex(idx)
        sn1 = shift_neg1_labels.reindex(idx)

        valid = b.notna() & s1.notna() & sn1.notna()
        if valid.sum() < 10:
             return False # Not enough data to verify stability

        b_v = b[valid]
        s1_v = s1[valid]
        sn1_v = sn1[valid]

        # Agreement Rate
        agree1 = (b_v == s1_v).mean()
        agree2 = (b_v == sn1_v).mean()
        avg_agreement = (agree1 + agree2) / 2.0

        # Threshold: Configurable (default 0.55 - heavily relaxed to ensure flow)
        try:
            stability_threshold = float(getattr(self, '_current_config', {}).get('layer2_stability_threshold', 0.55))
        except Exception:
            stability_threshold = 0.55

        if avg_agreement < stability_threshold:
             logger.debug(f"Stability failed: Flip rate too high (agreement={avg_agreement:.2f} < {stability_threshold})")
             return False

        return True

    def _tune_geometry_model_params(
        self,
        df: pd.DataFrame,
        events_df: pd.DataFrame,
        geometry: GeometryTrial,
        skip_hpo: bool = False
    ) -> Dict[str, Any]:
        """
        Run small HPO for model hyperparameters on a specific geometry.
        Scope: Production geometries only.
        """
        try:
            # 1. Prepare Data
            fam_events = events_df[events_df['family'] == geometry.family]
            if fam_events.empty:
                return {}

            lbls, _, _, _, _ = self._compute_dominance_labels(df, fam_events, family=geometry.family, **geometry.params)
            valid_lbls = lbls.dropna()

            # Subsample for HPO (30% or 2000, whichever is smaller, but at least 400)
            n_total = len(valid_lbls)
            if n_total < 100:
                return {} # Too small to tune

            n_sub = min(2000, int(n_total * 0.3))  # Reduced from 0.5 to 0.3
            n_sub = max(n_sub, min(n_total, 400))  # Reduced from 500 to 400

            # Deterministic subsample for stability
            rng = np.random.default_rng(self.random_state)
            indices = rng.choice(valid_lbls.index, size=n_sub, replace=False)
            indices = np.sort(indices) # keep temporal order mostly? No, random is fine for distribution check, but for time series...
            # Actually, standard shuffle is better for quick HPO unless we strictly need time series split.
            # Given we use simple Train/Val split inside objective, random subsample is risky for leakage?
            # Better to take a contiguous chunk or just use random if we assume stationarity for HPO.
            # Let's use the last N samples to be safe/relevant? Or stratified?
            # User said "Subsample (we need relative performance)". Random is likely fine.

            y_sub = valid_lbls.loc[indices]

            # Need features (Use Probe Basis if we want fast HPO)
            # Actually, `tune_geometry_model_params` is Tier 3 (Production) optimization.
            # It should ideally use the *full* feature set or the *selected* feature set for that geometry.
            # Since selection happens *after* or *during* Tier 3, we often tune on full features or a large subset.
            # However, for consistency with the initial screening, we can start with full.
            # BUT, the user explicitly asked for probe features for the "weak" probe.
            # This function runs "Small HPO" for the WINNING geometries.
            # It should probably use the full set to find optimal params for the production model.
            X_events = self._build_geometry_independent_event_features(df, fam_events.loc[indices], mode='full')

            # If we have probe features defined (from Tier 1), we could use them, but this is Tier 3 HPO.
            # We usually tune on the full set to get the best model.
            # No filtering to probe features here.

            X_sub = X_events.fillna(0.0)

            # PHASE 1: Quick Model Race (5 min)
            tprint_info(f"Running quick model race for {geometry.uuid} ({geometry.family})...")
            
            split_idx = int(len(X_sub) * 0.8)
            X_train_race = X_sub.iloc[:split_idx]
            X_val_race = X_sub.iloc[split_idx:]
            y_train_race = y_sub.iloc[:split_idx]
            y_val_race = y_sub.iloc[split_idx:]
            
            if len(np.unique(y_train_race)) < 2 or len(np.unique(y_val_race)) < 2:
                tprint_warning("Insufficient classes in race split. Using LGBM.")
                winning_model_type = 'lgbm'
                race_scores = {}
            else:
                winning_model_type, race_scores = _quick_5model_race(
                    X_train_race, y_train_race,
                    X_val_race, y_val_race,
                    self.random_state
                )

            tprint_info(f"Model Race for {geometry.uuid} ({geometry.family}): Winner={winning_model_type.upper()}")
            for m, s in race_scores.items():
                 tprint_info(f"  - {m}: {s:.4f}")

            # Pruning: If best model is barely better than random, skip expensive HPO
            winner_score = race_scores.get(winning_model_type, 0.5)
            geometry.race_score = winner_score

            if winner_score < 0.52:
                tprint_warning(f"Geometry {geometry.family} pruned: Score {winner_score:.4f} < 0.52. Skipping HPO/RFE.")
                return {}

            if skip_hpo:
                return {'race_score': winner_score, 'model_type': winning_model_type}

            # PHASE 2: Full HPO on Winner (20 min)
            tprint_info(f"Running HPO for {winning_model_type.upper()} on {geometry.family} geometry...")
            
            
            if winning_model_type == 'lgbm':
                # LGBM objective wrapped in function
                def objective(trial):
                    # Optimized Focal Loss Parameters
                    focal_alpha = trial.suggest_float('focal_alpha', 0.1, 0.9)
                    gamma_pos = trial.suggest_float('gamma_pos', 0.0, 5.0)
                    gamma_neg = trial.suggest_float('gamma_neg', 0.0, 5.0)

                    num_leaves = trial.suggest_int('num_leaves', 16, 64)
                    n_estimators = trial.suggest_int('n_estimators', 100, 500)
                    
                    params = LAYER2_MODEL_CONSTANTS.copy()
                    params.update({
                        'num_leaves': num_leaves,
                        'n_estimators': n_estimators,
                        'metric': ['binary_logloss', 'auc'],
                    })

                    # Split for Early Stopping
                    split_idx = int(len(X_sub) * 0.8)
                    X_tr, X_val = X_sub.iloc[:split_idx], X_sub.iloc[split_idx:]
                    y_tr, y_val = y_sub.iloc[:split_idx], y_sub.iloc[split_idx:]

                    if len(np.unique(y_tr)) < 2:
                        return 10.0

                    train_ds = lgb.Dataset(X_tr, label=y_tr)
                    val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)

                    # Focal Loss
                    focal_obj = RobustFocalLoss(gamma_pos=gamma_pos, gamma_neg=gamma_neg, alpha=focal_alpha, verbose=False)
                    params['objective'] = focal_obj
                    params['metric'] = 'auc'

                    # Callback for pruning - using AUC for pruning (average_precision str not supported by LGBM)
                    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "auc")

                    model = lgb.train(
                        params,
                        train_ds,
                        valid_sets=[val_ds],
                        callbacks=[
                            lgb.early_stopping(30, verbose=False),
                            # pruning_callback # Disabled to avoid direction mismatch errors
                        ]
                    )

                    # Score
                    preds = model.predict(X_val)
                    preds = expit(preds)
                    # Use (1 - Average Precision) for minimization direction
                    score = 1.0 - average_precision_score(y_val, preds)
                    return score

            elif winning_model_type == 'xgb':
                # XGBoost HPO objective
                def objective(trial):
                    # Optimized Focal Loss Parameters
                    focal_alpha = trial.suggest_float('focal_alpha', 0.1, 0.9)
                    gamma_pos = trial.suggest_float('gamma_pos', 0.0, 5.0)
                    gamma_neg = trial.suggest_float('gamma_neg', 0.0, 5.0)

                    n_estimators = trial.suggest_int('n_estimators', 100, 400)
                    max_depth = trial.suggest_int('max_depth', 3, 6)
                    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.05)

                    split_idx = int(len(X_sub) * 0.8)
                    X_tr, X_val = X_sub.iloc[:split_idx], X_sub.iloc[split_idx:]
                    y_tr, y_val = y_sub.iloc[:split_idx], y_sub.iloc[split_idx:]

                    if len(np.unique(y_tr)) < 2:
                        return 10.0

                    focal_obj = XGBFocalLoss(gamma_pos=gamma_pos, gamma_neg=gamma_neg, alpha=focal_alpha)

                    model = xgb.XGBClassifier(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        objective=focal_obj,
                        eval_metric=['logloss', 'aucpr'], # De Prado: Monitor PR-AUC
                        early_stopping_rounds=30,
                        verbosity=0,
                        random_state=self.random_state,
                        n_jobs=1,
                    )

                    model.fit(
                        X_tr, y_tr,
                        eval_set=[(X_val, y_val)],
                        verbose=False,
                    )

                    preds = model.predict_proba(X_val)[:, 1]
                    score = 1.0 - average_precision_score(y_val, preds)
                    return score

            elif winning_model_type == 'xgb_linear':
                # XGBoost (linear booster) HPO objective
                def objective(trial):
                    n_estimators = trial.suggest_int('n_estimators', 200, 800)
                    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1)
                    reg_alpha = trial.suggest_float('reg_alpha', 0.0, 5.0)
                    reg_lambda = trial.suggest_float('reg_lambda', 0.0, 5.0)

                    split_idx = int(len(X_sub) * 0.8)
                    X_tr, X_val = X_sub.iloc[:split_idx], X_sub.iloc[split_idx:]
                    y_tr, y_val = y_sub.iloc[:split_idx], y_sub.iloc[split_idx:]

                    if len(np.unique(y_tr)) < 2:
                        return 10.0

                    model = xgb.XGBClassifier(
                        booster='gblinear',
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        reg_alpha=reg_alpha,
                        reg_lambda=reg_lambda,
                        objective='binary:logistic',
                        eval_metric=['logloss', 'aucpr'],
                        early_stopping_rounds=30,
                        verbosity=0,
                        random_state=self.random_state,
                        n_jobs=1,
                    )

                    model.fit(
                        X_tr, y_tr,
                        eval_set=[(X_val, y_val)],
                        verbose=False,
                    )

                    preds = model.predict_proba(X_val)[:, 1]
                    score = 1.0 - average_precision_score(y_val, preds)
                    return score

            elif winning_model_type == 'catboost':
                # CatBoost HPO objective
                def objective(trial):
                    iterations = trial.suggest_int('iterations', 100, 400)
                    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.05)
                    depth = trial.suggest_int('depth', 3, 5)

                    # Optimize Class Weights for Imbalance
                    class_weight_ratio = trial.suggest_float('class_weight_ratio', 1.0, 10.0)
                    # CatBoost lacks focal_gamma, so we can't couple it directly.
                    # We rely on class_weight_ratio roughly behaving like alpha/(1-alpha).

                    split_idx = int(len(X_sub) * 0.8)
                    X_tr, X_val = X_sub.iloc[:split_idx], X_sub.iloc[split_idx:]
                    y_tr, y_val = y_sub.iloc[:split_idx], y_sub.iloc[split_idx:]

                    if len(np.unique(y_tr)) < 2:
                        return 10.0

                    model = catboost.CatBoostClassifier(
                        iterations=iterations,
                        learning_rate=learning_rate,
                        depth=depth,
                        class_weights={0: 1.0, 1: class_weight_ratio},
                        verbose=False,
                        random_seed=self.random_state,
                        thread_count=1,
                    )

                    model.fit(
                        X_tr, y_tr,
                        eval_set=(X_val, y_val),
                        early_stopping_rounds=30,
                    )

                    preds = model.predict_proba(X_val)[:, 1]
                    score = 1.0 - average_precision_score(y_val, preds)
                    return score
            
            else:
                # Fallback to LGBM if unknown model type
                logger.warning(f"Unknown model type {winning_model_type}, using LGBM")
                winning_model_type = 'lgbm'
                # Define default LGBM objective
                def objective(trial):
                    focal_alpha = trial.suggest_float('focal_alpha', 0.4, 1.0)
                    gamma_ratio = trial.suggest_float('gamma_ratio', 0.5, 3.0)
                    focal_gamma = gamma_ratio / focal_alpha
                    num_leaves = trial.suggest_int('num_leaves', 63, 255)
                    n_estimators = trial.suggest_int('n_estimators', 750, 1500)
                    
                    params = LAYER2_MODEL_CONSTANTS.copy()
                    params.update({
                        'num_leaves': num_leaves,
                        'n_estimators': n_estimators,
                        'metric': ['binary_logloss', 'average_precision'],
                    })
                    
                    split_idx = int(len(X_sub) * 0.8)
                    X_tr, X_val = X_sub.iloc[:split_idx], X_sub.iloc[split_idx:]
                    y_tr, y_val = y_sub.iloc[:split_idx], y_sub.iloc[split_idx:]
                    
                    if len(np.unique(y_tr)) < 2:
                        return 10.0
                    
                    train_ds = lgb.Dataset(X_tr, label=y_tr)
                    val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)
                    
                    focal_obj = RobustFocalLoss(gamma_pos=focal_gamma, gamma_neg=focal_gamma * 2.5, alpha=focal_alpha, verbose=False)
                    params['objective'] = focal_obj
                    
                    model = lgb.train(
                        params,
                        train_ds,
                        valid_sets=[val_ds],
                        callbacks=[lgb.early_stopping(30, verbose=False)]
                    )
                    
                    preds = model.predict(X_val)
                    preds = expit(preds)
                    score = 1.0 - average_precision_score(y_val, preds)
                    return score

            # 3. Optimize
            sampler = optuna.samplers.TPESampler(seed=self.random_state)
            pruner = optuna.pruners.HyperbandPruner()
            study = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)
            study.optimize(objective, n_trials=40, n_jobs=1) # 40 trials as requested

            best = study.best_params
            # Extract params and add model type + race scores
            result = best.copy()
            result['model_type'] = winning_model_type
            result['race_scores'] = race_scores
            return result

        except Exception as e:
            logger.warning(f"Geometry model HPO failed for {geometry.uuid}: {e}")
            return {}

    def _optimize_families(
        self,
        df: pd.DataFrame,
        events_df: pd.DataFrame
    ) -> Dict[str, List[GeometryTrial]]:
        """
        Replaces Optuna optimization with label_geometry_selection logic.
        
        REFACTORED: No longer processes per-family. Geometry selection now uses
        geometric characteristics (sl_sigma, alpha, beta, min_ratio, horizon)
        for diversity, not semantic family names. All events are processed together.
        """
        results: Dict[str, List[GeometryTrial]] = {}

        events_df = events_df.copy()
        # Keep family assignment for backward compatibility and logging
        events_df['family'] = self._assign_barrier_families(events_df)

        # Build feature matrix once for ALL events (Probe Basis)
        X_events_all = self._build_geometry_independent_event_features(df, events_df, mode='probe')
        
        # Check we have enough events total
        if len(events_df) < 50:
            tprint_warning(f"Not enough events total ({len(events_df)}). Skipping geometry selection.")
            return {}

        tprint_info(f"Optimizing geometries: {len(events_df)} total events (no family split)")

        # Extract events for ALL data - no family filtering
        selection_events = self._extract_events_for_selection(df, events_df)

        # Reset features index to match event IDs (0..N-1)
        X_events_reset = X_events_all.reset_index(drop=True)

        # Run unified Selection with ALL events
        selected_raw = select_geometries(selection_events, {}, X_events_reset)

        # Guard: Handle empty selection gracefully
        if not selected_raw:
            tprint_warning("Geometry selection returned 0 geometries - all candidates failed gates. "
                          "Consider relaxing thresholds or increasing data.")
            return {}

        # Convert to GeometryTrial - use 'Unified' as family name
        # (keeps backward compatibility with dict structure)
        unified_family = 'Unified'
        trials = []
        
        for i, (geom, survivors) in enumerate(selected_raw):
            survival_rate = len(survivors) / len(selection_events) if selection_events else 0.0

            params = {
                'sl_sigma': geom.sl_sigma,
                'alpha': geom.alpha,
                'beta': geom.beta,
                'min_ratio': geom.min_ratio,
                'horizon': geom.horizon
            }

            t_obj = GeometryTrial(
                family=unified_family,  # Single unified family
                params=params,
                final_score=survival_rate * 100.0,
                learnability=0.5,
                robust_magnitude=0.0,
                stability=1.0,
                balance=1.0,
                raw_metrics={'passed': True, 'survivors': len(survivors)},
                uuid=f"Geo_Sel{i}"
            )
            trials.append(t_obj)

        results[unified_family] = trials
        tprint_info(f"Geometry selection complete: {len(trials)} geometries selected")

        return results

    def _optimization_objective(
        self,
        study: optuna.Study,
        trial: optuna.Trial,
        df: pd.DataFrame,
        family: str,
        family_events: pd.DataFrame,
        X_events: pd.DataFrame,
        probe_features: List[str],
        target_sample_weight_events: Optional[pd.Series]
    ) -> float:
        """Extracted optimization objective to avoid nested function re-definition."""
        bounds = self._current_param_bounds.get(str(family)) if isinstance(getattr(self, '_current_param_bounds', None), dict) else None

        # Parameter Space: Kappa and Horizon (De Prado 1.1: Economically Constrained)
        # Use discrete grids for Kappa and Horizon to reduce overfitting/snooping.
        kappa_grid = [1.25, 1.6, 2.0, 2.5, 3.2, 4.0]  # Log-spaced grid
        horizon_grid = [12, 24, 48]  # Discrete horizons (12, 24, 48 bars)

        # Use family-specific bounds if available
        if isinstance(bounds, dict) and all(k in bounds for k in ('k_low', 'k_high', 'h_low', 'h_high')):
             kappa = trial.suggest_float('kappa', float(bounds['k_low']), float(bounds['k_high']))
             horizon = trial.suggest_int('horizon', int(bounds['h_low']), int(bounds['h_high']))
        else:
             kappa = trial.suggest_categorical('kappa', kappa_grid)
             horizon = trial.suggest_categorical('horizon', horizon_grid)

        try:
            sl_low = float(getattr(self, '_current_config', {}).get('layer2_sl_mult_low', 0.3))
        except Exception:
            sl_low = 0.3
        try:
            sl_high = float(getattr(self, '_current_config', {}).get('layer2_sl_mult_high', 2.0))
        except Exception:
            sl_high = 2.0
        if (not np.isfinite(sl_low)) or sl_low <= 0.0:
            sl_low = 0.5
        if (not np.isfinite(sl_high)) or sl_high <= float(sl_low):
            sl_high = float(max(float(sl_low) + 0.5, 3.0))

        # Enforce Proportionality Constraint: 1.0 <= TP / SL <= 4.0
        # TP ~ kappa * vol, SL ~ sl_mult * vol  => 1.0 <= kappa / sl_mult <= 4.0
        # => sl_mult <= kappa AND sl_mult >= kappa / 4.0

        eff_sl_low = max(sl_low, kappa / 4.0)
        eff_sl_high = min(sl_high, kappa)

        if eff_sl_low > eff_sl_high:
            # Impossible to satisfy constraints with this kappa and config bounds
            # Return -1.0 to prune
            return -1.0

        sl_mult = trial.suggest_float('sl_mult', eff_sl_low, eff_sl_high)

        # Distance-based pruning to avoid similar geometries
        # Note: Normalization logic adapted for discrete grids
        # Kappa range roughly 1.25-4.0
        # Horizon range 12-48

        params_vector = [kappa, sl_mult, horizon]

        k_min, k_max = 1.25, 4.0
        sl_min, sl_max = 0.3, 4.0
        h_min, h_max = 12, 48

        normalized_params = [
            (kappa - k_min) / (k_max - k_min + 1e-9),
            (sl_mult - sl_min) / (sl_max - sl_min + 1e-9),
            (horizon - h_min) / (h_max - h_min + 1e-9)
        ]
        threshold = 0.05
        
        # Weights for distance: Kappa (1.5) > SL (1.0) > Horizon (0.5)
        # Normalized to sum to 3.0 (dimension count) to keep threshold scale similar
        dist_weights = np.array([1.5, 1.0, 0.5])

        for prev_trial in study.trials:
            if prev_trial.value is None or prev_trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            # Handle potential missing params in previous trials if schema changed
            p_k = prev_trial.params.get('kappa', k_min)
            p_sl = prev_trial.params.get('sl_mult', sl_min)
            p_h = prev_trial.params.get('horizon', h_min)

            prev_norm = [
                (p_k - k_min) / (k_max - k_min + 1e-9),
                (p_sl - sl_min) / (sl_max - sl_min + 1e-9),
                (p_h - h_min) / (h_max - h_min + 1e-9)
            ]
            if euclidean(normalized_params, prev_norm, w=dist_weights) < threshold:
                return -1.0

        # Compute labels
        labels, returns, _, _, exit_reasons = self._compute_dominance_labels(df, family_events, kappa, horizon, family, sl_mult=sl_mult)

        # Metrics
        mean_ret = returns.mean()
        if np.isnan(mean_ret):
            mean_ret = -1.0

        # Profitability (trade-conditional): only the trades you would take (label==1)
        try:
            trade_mask = labels == 1
            pos_count = int(trade_mask.sum()) if hasattr(trade_mask, 'sum') else 0
            mean_trade_ret = float(pd.to_numeric(returns[trade_mask], errors='coerce').astype(float).mean()) if pos_count > 0 else float('nan')
        except Exception:
            pos_count = 0
            mean_trade_ret = float('nan')

        # Positive Rate Filter (10-40%)
        count = labels.notna().sum()
        if count < 20:
            return -1.0 # Too few samples

        pos_rate = labels.mean()

        # --- NEW GATES: Time Limit, Frequency, Gini, Sharpe ---
        # 1. Time Limit Hit Rate < 50%
        # exit_reasons contains strings like 'timeout', 'profit', 'stop', 'trailing'
        n_timeout = (exit_reasons == 'timeout').sum()
        time_limit_hit_rate = float(n_timeout) / float(count) if count > 0 else 1.0

        if time_limit_hit_rate >= 0.5:
             logger.info(f"[GATE_REJECT] trial={trial.number}, family={family}, reason=time_limit_hit_rate, rate={time_limit_hit_rate:.3f}")
             return -1.0

        # 2. Frequency >= 0.75 events/day
        if len(returns) > 1 and returns.index[-1] > returns.index[0]:
            duration_days = (returns.index[-1] - returns.index[0]).total_seconds() / 86400.0
            events_per_day = float(len(returns)) / max(1.0, duration_days)
        else:
            events_per_day = 0.0

        if events_per_day < 0.75:
             logger.info(f"[GATE_REJECT] trial={trial.number}, family={family}, reason=low_frequency, rate={events_per_day:.3f}/day")
             return -1.0

        # 3. Gini of Signals > 0.8 (Burstiness)
        # Using event timestamps differences
        try:
            # Convert index diffs to float seconds
            ts_diffs = pd.Series(returns.index).diff().dropna().dt.total_seconds().values
            gini_signals = _gini_coefficient(ts_diffs)
        except Exception:
            gini_signals = 0.0

        # Requirement: Gini > 0.8. Reject if <= 0.8
        # Wait, usually high Gini means bursty/clustered.
        # Requirement text: "Gini of signals > 0.8".
        if gini_signals <= 0.8:
             logger.info(f"[GATE_REJECT] trial={trial.number}, family={family}, reason=low_gini_signals, gini={gini_signals:.3f}")
             return -1.0

        # 4. Sharpe of Labels < 0.5 (Base Strategy Quality - should be weak)
        ret_std = returns.std()
        sharpe_base = (returns.mean() / ret_std) if ret_std > 1e-9 else 0.0

        if sharpe_base >= 0.5:
             logger.info(f"[GATE_REJECT] trial={trial.number}, family={family}, reason=high_base_sharpe, sharpe={sharpe_base:.3f}")
             return -1.0

        # 5. Perfect Information Sharpe > 2.0 (Potential)
        # Sharpe of returns where label == 1
        pos_rets = returns[labels == 1]
        if len(pos_rets) > 1:
             pos_std = pos_rets.std()
             perfect_sharpe = (pos_rets.mean() / pos_std) if pos_std > 1e-9 else 0.0
        else:
             perfect_sharpe = 0.0

        if perfect_sharpe <= 2.0:
             logger.info(f"[GATE_REJECT] trial={trial.number}, family={family}, reason=low_perfect_sharpe, sharpe={perfect_sharpe:.3f}")
             return -1.0

        # --- OPTIMIZATION: Tighter Pre-Filters ---
        # If the geometry is fundamentally poor in terms of base statistics, don't waste time on probes.
        # LOOSENED GATES: Defaults relaxed to [0.001, 0.999] to prioritize learnability.
        try:
            min_rate = float(getattr(self, '_current_config', {}).get('layer2_min_pos_rate', 0.001))
            max_rate = float(getattr(self, '_current_config', {}).get('layer2_max_pos_rate', 0.999))
        except Exception:
             min_rate, max_rate = 0.001, 0.999

        if pos_rate < min_rate or pos_rate > max_rate: 
             logger.info(f"[GATE_REJECT] trial={trial.number}, family={family}, reason=pos_rate_limit, pos_rate={pos_rate:.3f}, range=[{min_rate}, {max_rate}]")
             t_obj = GeometryTrial(
                family=family,
                params={'kappa': kappa, 'sl_mult': sl_mult, 'horizon': horizon},
                final_score=-1.0,
                learnability=0.0,
                robust_magnitude=0.0,
                stability=0.0,
                balance=0.0,
                raw_metrics={'passed': False, 'pos_rate': pos_rate, 'reason': 'pos_rate_limit'},
                uuid=f"{family}_{trial.number}"
            )
             trial.set_user_attr("geometry_object", t_obj)
             return -1.0

        # Strategic Profitability Gate (allow break-even with risk filters)
        try:
            profit_mode = str(getattr(self, '_current_config', {}).get('layer2_profitability_mode', 'intelligent'))
        except Exception:
            profit_mode = 'intelligent'
        profit_mode = str(profit_mode).strip().lower()

        try:
            min_pos_trades = int(getattr(self, '_current_config', {}).get('layer2_min_positive_trades', 1))
        except Exception:
            min_pos_trades = 1

        if profit_mode == 'intelligent':
            # Allow small losses but require risk compensation
            # LOOSENED GATES: Defaults set to -1.0 to disable strict PnL requirements.
            min_trade_ret = float(getattr(self, '_current_config', {}).get('layer2_min_mean_trade_return', -1.0))
            max_acceptable_loss = float(getattr(self, '_current_config', {}).get('layer2_max_acceptable_loss', -1.0))
            min_sharpe_proxy = float(getattr(self, '_current_config', {}).get('layer2_min_sharpe_proxy', 0.0))  # Disable Sharpe gate temporarily
        else:
            # Original strict mode
            min_trade_ret = float(getattr(self, '_current_config', {}).get('layer2_min_mean_trade_return', self.transaction_cost))
            max_acceptable_loss = float('inf')
            min_sharpe_proxy = -float('inf')

        is_profitable = True
        if profit_mode in {'trade', 'trade_mean', 'conditional', 'intelligent'}:
            if int(pos_count) < int(min_pos_trades):
                is_profitable = False
            elif (not np.isfinite(mean_trade_ret)) or (float(mean_trade_ret) < float(min_trade_ret)):
                is_profitable = False
            elif profit_mode == 'intelligent':
                # Additional risk filters for intelligent mode
                # Fix sign logic: defaults are negative, so check strictly less than
                limit = max_acceptable_loss
                if limit > 0: limit = -limit # Ensure it is a floor (negative return)

                if float(mean_ret) < limit:  # Don't allow large losses
                    is_profitable = False
                # Calculate Sharpe proxy if return data available
                try:
                    return_std = returns.std() if len(returns.dropna()) > 1 else float('inf')
                    sharpe_proxy = float(mean_ret) / float(return_std) if return_std > 0 else -float('inf')
                    if sharpe_proxy < min_sharpe_proxy:
                        is_profitable = False
                except Exception:
                    pass  # If Sharpe calculation fails, continue
        else:
            if float(mean_ret) < float(self.transaction_cost):
                is_profitable = False

        if not is_profitable:
             logger.info(f"[GATE_REJECT] trial={trial.number}, family={family}, reason=unprofitable, mean_ret={mean_ret:.5f}")
             t_obj = GeometryTrial(
                family=family,
                params={'kappa': kappa, 'sl_mult': sl_mult, 'horizon': horizon},
                final_score=-1.0,
                learnability=0.0,
                robust_magnitude=0.0,
                stability=0.0,
                balance=0.0,
                raw_metrics={
                    'passed': False,
                    'pos_rate': pos_rate,
                    'pos_count': float(pos_count),
                    'mean_trade_ret': float(mean_trade_ret) if np.isfinite(mean_trade_ret) else float('nan'),
                    'mean_ret': float(mean_ret) if np.isfinite(mean_ret) else float('nan'),
                    'reason': 'unprofitable'
                },
                uuid=f"{family}_{trial.number}"
            )
             trial.set_user_attr("geometry_object", t_obj)
             return -1.0

        # Stability Check (Time-Flip)
        # Configurable frequency + optional subsampling to reduce compute.
        try:
            stability_every = int(getattr(self, '_current_config', {}).get('layer2_stability_every_n_trials', 3))
        except Exception:
            stability_every = 3
        if stability_every <= 0:
            stability_every = 1

        try:
            stability_sample_frac = float(getattr(self, '_current_config', {}).get('layer2_stability_sample_frac', 0.7))
        except Exception:
            stability_sample_frac = 0.7
        if (not np.isfinite(stability_sample_frac)) or stability_sample_frac <= 0.0:
            stability_sample_frac = 1.0
        stability_sample_frac = float(min(1.0, stability_sample_frac))

        do_stability = (trial.number % int(stability_every)) == 0
        is_stable = True
        fam_events_for_checks = family_events
        if do_stability:
            if stability_sample_frac < 1.0 and int(len(family_events)) > 50:
                try:
                    fam_events_for_checks = family_events.sample(frac=stability_sample_frac, random_state=int(self.random_state))
                except Exception:
                    fam_events_for_checks = family_events
            is_stable = self._check_stability(
                df,
                fam_events_for_checks,
                {'kappa': kappa, 'sl_mult': sl_mult, 'horizon': horizon},
                0.0,
                family,
            )
            if not is_stable:
                 logger.info(f"[GATE_REJECT] trial={trial.number}, family={family}, reason=unstable")
                 t_obj = GeometryTrial(
                    family=family,
                    params={'kappa': kappa, 'sl_mult': sl_mult, 'horizon': horizon},
                    final_score=-1.0,
                    learnability=0.0,
                    robust_magnitude=0.0,
                    stability=0.0,
                    balance=0.0,
                    raw_metrics={'passed': False, 'pos_rate': pos_rate, 'reason': 'unstable'},
                    uuid=f"{family}_{trial.number}"
                )
                 trial.set_user_attr("geometry_object", t_obj)
                 return -1.0

        # --- Noise Metrics ---

        # 1. Flip Rate (Barrier Perturbation)
        # Configurable frequency + optional subsampling to reduce compute.
        try:
            perturb_every = int(getattr(self, '_current_config', {}).get('layer2_perturb_every_n_trials', 1))
        except Exception:
            perturb_every = 1
        if perturb_every <= 0:
            perturb_every = 1

        if (trial.number % int(perturb_every)) == 0:
            perturb_labels_k, _, _, _, _ = self._compute_dominance_labels(df, fam_events_for_checks, kappa * 1.05, horizon, family, sl_mult=sl_mult)
            perturb_labels_sl, _, _, _, _ = self._compute_dominance_labels(df, fam_events_for_checks, kappa, horizon, family, sl_mult=sl_mult * 1.05)
            perturb_labels_h, _, _, _, _ = self._compute_dominance_labels(df, fam_events_for_checks, kappa, int(horizon * 1.05), family, sl_mult=sl_mult)

            base_lbl = labels.reindex(fam_events_for_checks.index)
            agree_k = (base_lbl == perturb_labels_k).mean()
            agree_sl = (base_lbl == perturb_labels_sl).mean()
            agree_h = (base_lbl == perturb_labels_h).mean()
            flip_rate = 1.0 - ((agree_k + agree_sl + agree_h) / 3.0)
        else:
            flip_rate = 0.0

        # 2. Directional Entropy
        # H = -p log p - (1-p) log(1-p)
        p_safe = np.clip(pos_rate, 1e-9, 1.0 - 1e-9)
        dir_entropy = -(p_safe * np.log(p_safe) + (1.0 - p_safe) * np.log(1.0 - p_safe))

        # 3. Conditional IC (IC | ER bucket)
        # Since we haven't trained a model yet for this specific geometry inside the loop (only probing next),
        # we can't calculate IC of predictions yet.
        # However, we can use the IC from the probe model if it passes.
        # We will compute it AFTER probe training.
        
        if not is_stable:
             logger.info(f"[GATE_REJECT] trial={trial.number}, family={family}, reason=unstable_recheck")
             t_obj = GeometryTrial(
                family=family,
                params={'kappa': kappa, 'sl_mult': sl_mult, 'horizon': horizon},
                final_score=-1.0,
                learnability=0.0,
                robust_magnitude=0.0,
                stability=0.0,
                balance=0.0,
                raw_metrics={
                    'passed': False,
                    'pos_rate': pos_rate,
                    'stable': False,
                    'flip_rate': flip_rate,
                    'entropy': dir_entropy
                },
                uuid=f"{family}_{trial.number}"
            )
             trial.set_user_attr("geometry_object", t_obj)
             return -1.0

        # Align features to events
        try:
            X_geom = X_events.loc[labels.index]
        except Exception:
            X_geom = X_events.reindex(labels.index)

        global_feats = [f for f in (probe_features or []) if f in X_geom.columns]
        X_probe = X_geom[global_feats] if global_feats else X_geom

        probe_weight = None
        if target_sample_weight_events is not None:
             # ... weight loading logic ...
             try:
                w_probe = target_sample_weight_events.reindex(labels.index)
                w_probe = pd.to_numeric(w_probe, errors='coerce').astype(float)
                w_probe = w_probe.replace([np.inf, -np.inf], np.nan).fillna(1.0)
                w_probe = w_probe.clip(lower=0.0)
                w_probe = w_probe.reindex(labels.dropna().index)
                probe_weight = w_probe.values
             except Exception:
                probe_weight = None

        probe_res = self._train_probes(X_probe, labels, sample_weight=probe_weight, trial=trial)

        try:
            # De Prado Fix: AUC-Excess (only reward edge over random 0.5)
            raw_auc = float(probe_res.get('auc', 0.5))
            learnability = max(0.0, raw_auc - 0.5) * 2.0
        except Exception:
            learnability = 0.0
        if not np.isfinite(learnability):
            learnability = 0.0

        # Conditional IC Calculation (approximate using probe results if available)
        # We don't have per-sample predictions from _train_probes easily without refactoring.
        # _train_probes uses K-Fold internally and returns aggregated metrics.
        # We will use the 'ic' from probe_res as a proxy for global IC.
        # Calculating IC conditioned on ER buckets requires predictions aligned with events.
        # Since _train_probes doesn't return OOF preds, we skip detailed conditional IC
        # and just store the global IC in raw_metrics.
        global_ic = probe_res.get('ic', 0.0)

        # Degeneracy guardrail
        entropy_norm = _normalized_binary_entropy(pos_rate)
        degeneracy_floor = 0.25 + 0.75 * entropy_norm

        # Magnitude bonus (using mean return of successful trades vs volatility)
        ret_std = float(returns.std())
        sharpe_proxy = float(mean_ret) / (ret_std + 1e-9)
        mag_component = float(np.clip(sharpe_proxy, 0.0, 3.0))

        final_score = (1.0 + mag_component) * np.log1p(count) * degeneracy_floor * learnability

        t_obj = GeometryTrial(
            family=family,
            params={'kappa': kappa, 'sl_mult': sl_mult, 'horizon': horizon},
            final_score=final_score,
            learnability=learnability,
            robust_magnitude=float(mean_ret) * 10000,
            stability=1.0, # Passed stability check
            balance=degeneracy_floor,
            raw_metrics=dict(probe_res, **{
                'pos_rate': pos_rate,
                'flip_rate': flip_rate,
                'entropy': dir_entropy,
                'ic_global': global_ic
            }),
            uuid=f"{family}_{trial.number}"
        )
        
        trial.set_user_attr("geometry_object", t_obj)
        
        return final_score

    def _select_best_geometries(
        self,
        df: pd.DataFrame,
        events_df: pd.DataFrame,
        family_results: Dict[str, List[GeometryTrial]],
        require_passed: bool = True,
    ) -> List[GeometryTrial]:
        """Step 3.2 & 3.3: Prune families and select diverse geometries.

        When require_passed=True (production mode), only geometries with
        raw_metrics['passed']==True and final_score>0 are eligible.
        When require_passed=False (OOF analytics mode), we allow selecting from
        all trials if nothing passes, so that labels/returns can still be produced.
        """
        # Ensure family column exists
        if 'family' not in events_df.columns:
            events_df = events_df.copy()
            events_df['family'] = self._assign_barrier_families(events_df)

        def _is_passed_trial(t: Any) -> bool:
            # Relaxed gate: just check if final_score is positive
            try:
                score = float(getattr(t, 'final_score', -1.0))
            except Exception:
                return False
            return np.isfinite(score) and score > 0.0

        # 3.2 Discard poorer barrier families
        family_medians = {}
        for fam, trials in family_results.items():
            trials_all = list(trials or [])
            trials_use = [t for t in trials_all if _is_passed_trial(t)] if require_passed else trials_all

            # In OOF mode, if nothing passed but we have trials, we still want a ranking
            # so we can pick a stable/diverse subset for analytics.
            if (not require_passed) and (not trials_use) and trials_all:
                trials_use = trials_all

            trials_sorted = sorted(trials_use, key=lambda x: float(getattr(x, 'final_score', -1.0)), reverse=True)
            top_k = trials_sorted[:10]
            if not top_k:
                continue
            median_score = np.median([t.final_score for t in top_k])
            family_medians[fam] = median_score

        sorted_families = sorted(family_medians.items(), key=lambda x: x[1], reverse=True)
        keep_families = [f[0] for f in sorted_families[:3]]
        if not keep_families:
            keep_families = [str(k) for k in family_results.keys()]

        keep_families = [
            fam
            for fam in keep_families
            if fam in family_results and isinstance(family_results.get(fam), list) and len(family_results.get(fam)) > 0
        ]

        selected = []

        # 3.3 Keep diverse geometries per family
        for fam in keep_families:
            trials_all = list(family_results.get(fam) or [])

            if require_passed:
                trials = [t for t in trials_all if _is_passed_trial(t)]
                if not trials:
                    continue
            else:
                trials = [t for t in trials_all if np.isfinite(float(getattr(t, 'final_score', -1.0)))]
                if not trials and trials_all:
                    trials = trials_all

            # Improved Sorting Logic:
            # 1. Final Score (Profitability)
            # 2. Learnability (AUC) - explicit tie-breaker for negative scores
            # 3. Stability (lower is better, so negate)
            def _sort_key(x):
                fs = float(getattr(x, 'final_score', -1.0))
                lrn = float(getattr(x, 'learnability', 0.0))
                stab = float(getattr(x, 'stability', 0.0))
                return (fs, lrn, -stab)

            trials.sort(key=_sort_key, reverse=True)
            # Keep all candidates since upstream selection already pruned to top 10/20
            # Previously: n_top = max(2, int(len(trials) * 0.2))
            n_top = len(trials)
            top_tier = trials[:n_top]

            try:
                cfg_hs = getattr(self, '_current_config', {})
                if not isinstance(cfg_hs, dict):
                    cfg_hs = {}
            except Exception:
                cfg_hs = {}

            try:
                hs_enabled = bool(cfg_hs.get('layer2_hierarchical_selection_enabled', True))
            except Exception:
                hs_enabled = True

            try:
                hs_full_enabled = bool(cfg_hs.get('layer2_hs_full_enabled', True))
            except Exception:
                hs_full_enabled = True

            try:
                hs_full_in_oof = bool(cfg_hs.get('layer2_hs_full_in_oof', False))
            except Exception:
                hs_full_in_oof = False

            do_full = bool(hs_full_enabled and (bool(require_passed) or bool(hs_full_in_oof)))

            if hs_enabled and len(top_tier) > 2:
                try:
                    k0 = int(cfg_hs.get('layer2_hs_k0_linear', 20))
                except Exception:
                    k0 = 20
                try:
                    k1 = int(cfg_hs.get('layer2_hs_k1_light', 8))
                except Exception:
                    k1 = 8

                k0 = int(max(2, min(int(k0), int(len(top_tier)))))
                k1 = int(max(2, min(int(k1), int(k0))))

                def _safe_rm_auc(t_obj: GeometryTrial, key: str) -> float:
                    try:
                        rm = getattr(t_obj, 'raw_metrics', None)
                        if not isinstance(rm, dict):
                            return 0.0
                        v = rm.get(key)
                        return float(v) if v is not None and np.isfinite(float(v)) else 0.0
                    except Exception:
                        return 0.0

                stage0 = sorted(top_tier, key=lambda t: _safe_rm_auc(t, 'auc_linear'), reverse=True)[:k0]

                def _safe_light_auc(t_obj: GeometryTrial) -> float:
                    v = _safe_rm_auc(t_obj, 'auc_lgbm_light')
                    if v > 0.0:
                        return v
                    return _safe_rm_auc(t_obj, 'auc_lgbm')

                stage1 = sorted(stage0, key=lambda t: _safe_light_auc(t), reverse=True)[:k1]

                if do_full:
                    try:
                        fam_events_local = events_df[events_df['family'] == fam]
                    except Exception:
                        fam_events_local = events_df

                    try:
                        X_events_full = self._build_geometry_independent_event_features(df, fam_events_local)
                    except Exception:
                        X_events_full = None

                    try:
                        probe_features = [f for f in (getattr(self, '_global_probe_features', []) or []) if X_events_full is not None and f in X_events_full.columns]
                    except Exception:
                        probe_features = []

                    if X_events_full is not None and not getattr(X_events_full, 'empty', True):
                        X_probe_full = X_events_full[probe_features] if probe_features else X_events_full
                    else:
                        X_probe_full = None

                    try:
                        target_sample_weight_events = self._get_target_sample_weight_for_events(df, fam_events_local)
                    except Exception:
                        target_sample_weight_events = None

                    if X_probe_full is not None and not getattr(X_probe_full, 'empty', True):
                        for cand in stage1:
                            try:
                                lbls, _, _, _, _ = self._compute_dominance_labels(df, fam_events_local, family=fam, **cand.params)
                            except Exception:
                                continue

                            if lbls is None or getattr(lbls, 'empty', True):
                                continue

                            w_full = None
                            if target_sample_weight_events is not None:
                                try:
                                    w_s = target_sample_weight_events.reindex(lbls.dropna().index)
                                    w_s = pd.to_numeric(w_s, errors='coerce').astype(float)
                                    w_s = w_s.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(lower=0.0)
                                    w_full = w_s.values
                                except Exception:
                                    w_full = None

                            try:
                                X_cand = X_probe_full.reindex(lbls.index)
                                full_res = self._train_full_lgbm_probe(X_cand, lbls, sample_weight=w_full)
                            except Exception:
                                continue

                            try:
                                if not isinstance(getattr(cand, 'raw_metrics', None), dict):
                                    cand.raw_metrics = {}
                                cand.raw_metrics.update(full_res)
                            except Exception:
                                pass

                        stage1 = sorted(stage1, key=lambda t: _safe_rm_auc(t, 'auc_full'), reverse=True)

                top_tier = stage1

            if not top_tier:
                continue

            fam_selected = []

            # Helper to normalize params for distance calculation
            # Support both legacy (kappa, sl_mult) and new (sl_sigma, alpha) formats
            k_vals = [t.params.get('kappa') or t.params.get('alpha') for t in top_tier]
            sl_vals = [t.params.get('sl_mult') or t.params.get('sl_sigma') for t in top_tier]
            h_vals = [t.params.get('horizon') for t in top_tier]

            k_vals_f = [float(v) for v in k_vals if v is not None and np.isfinite(float(v))]
            sl_vals_f = [float(v) for v in sl_vals if v is not None and np.isfinite(float(v))]
            h_vals_f = [float(v) for v in h_vals if v is not None and np.isfinite(float(v))]

            # Skip only if truly no valid params
            if (not k_vals_f) or (not h_vals_f):
                logger.warning(f"Family {fam}: Missing kappa/alpha or horizon params, skipping")
                continue

            if not sl_vals_f:
                sl_vals_f = [1.0]

            k_range = max(k_vals_f) - min(k_vals_f) + 1e-6
            sl_range = max(sl_vals_f) - min(sl_vals_f) + 1e-6
            h_range = max(h_vals_f) - min(h_vals_f) + 1e-6

            def get_norm_vec(t):
                # Support both legacy (kappa, sl_mult) and new (alpha, sl_sigma) formats
                k_val = t.params.get('kappa') or t.params.get('alpha', 0.0)
                sl_val = t.params.get('sl_mult') or t.params.get('sl_sigma', 1.0)
                return np.array([
                    (float(k_val) - min(k_vals_f)) / k_range,
                    (float(sl_val) - min(sl_vals_f)) / sl_range,
                    (float(t.params.get('horizon', 0.0)) - min(h_vals_f)) / h_range,
                ])

            # Outcome-space diversification: avoid selecting highly correlated return series
            try:
                # Lowered default correlation threshold to 0.95 to ensure diversity but keep high counts
                corr_thr = float(getattr(self, '_current_config', {}).get('layer2_outcome_corr_threshold', 0.95))
            except Exception:
                corr_thr = 0.95

            ret_cache: Dict[str, pd.Series] = {}
            def _get_ret_series(t_obj: GeometryTrial) -> pd.Series:
                key = str(getattr(t_obj, 'uuid', ''))
                if key in ret_cache:
                    return ret_cache[key]
                try:
                    fam_events_local = events_df[events_df['family'] == fam]
                    _lbl, _ret, _, _, _ = self._compute_dominance_labels(df, fam_events_local, family=fam, **t_obj.params)
                    s = pd.to_numeric(_ret, errors='coerce').astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                except Exception:
                    s = pd.Series(0.0, index=events_df[events_df['family'] == fam].index)
                ret_cache[key] = s
                return s

            # Pick best first (stable)
            for cand in top_tier:
                fam_events = events_df[events_df['family'] == fam]
                # Already checked stability in optimization loop for passed trials
                # But double check if coming from fallback
                if self._check_stability(df, fam_events, cand.params, cand.final_score, fam):
                    fam_selected.append(cand)
                    break

            if not fam_selected:
                # Production mode should never fall back to a failing/unstable geometry.
                if require_passed:
                    continue
                try:
                    fam_selected.append(top_tier[0])
                except Exception:
                    continue

            # Pick others maximizing normalized distance
            candidate_pool = [t for t in top_tier if t not in fam_selected]
            dist_weights = np.array([1.5, 1.0, 0.5]) # Kappa > SL > Horizon

            # Increased limit from 4 to 10 to allow more diversity
            while len(fam_selected) < 10 and candidate_pool:
                best_cand = None
                max_dist = -1.0

                for cand in candidate_pool:
                    # Use Weighted Euclidean Distance
                    dists = [euclidean(get_norm_vec(cand), get_norm_vec(s), w=dist_weights) for s in fam_selected]
                    min_d = min(dists)

                    if min_d > max_dist:
                        max_dist = min_d
                        best_cand = cand

                if best_cand:
                    # Stability check
                    fam_events = events_df[events_df['family'] == fam]
                    # Correlation filter vs already-selected
                    drop_reason = None
                    try:
                        ok_corr = True
                        cand_ret = _get_ret_series(best_cand)
                        for s_obj in fam_selected:
                            s_ret = _get_ret_series(s_obj)
                            c = float(pd.Series(cand_ret).corr(pd.Series(s_ret)))
                            if np.isfinite(c) and abs(c) >= float(corr_thr):
                                ok_corr = False
                                drop_reason = f"Correlation {c:.3f} >= {corr_thr}"
                                break
                        
                        if not ok_corr:
                            logger.info(f"Dropped candidate (Score={best_cand.final_score:.3f}): {drop_reason}")
                        elif self._check_stability(df, fam_events, best_cand.params, best_cand.final_score, fam):
                            fam_selected.append(best_cand)
                            logger.info(f"Selected candidate (Score={best_cand.final_score:.3f})")
                        else:
                            logger.info(f"Dropped candidate (Score={best_cand.final_score:.3f}): Stability check failed")
                            
                    except Exception as e:
                        logger.warning(f"Error checking candidate {best_cand.uuid}: {e}")
                        # Fallback try stability
                        if self._check_stability(df, fam_events, best_cand.params, best_cand.final_score, fam):
                            fam_selected.append(best_cand)

                    candidate_pool.remove(best_cand)
                else:
                    break

            selected.extend(fam_selected)

            # CLUSTER-BASED SELECTION (Global Diversification)
            # Replaces greedy filtering with Hierarchical Clustering to ensure structural diversity.

            # Helper to get return series for a trial
            # Cache stores dict: {'ret': pd.Series, 'dd': pd.Series, 'ratio': float}
            cache_global: Dict[str, Dict[str, Any]] = {}
            all_events_idx = events_df.index

            def _get_metrics_global(t_obj: GeometryTrial) -> Dict[str, Any]:
                key = str(getattr(t_obj, 'uuid', ''))
                if key in cache_global:
                    return cache_global[key]

                try:
                    fam_local = str(getattr(t_obj, 'family', ''))
                    fam_events_local = events_df[events_df['family'] == fam_local]

                    if fam_events_local.empty:
                        # Fallback empty
                        zeros = pd.Series(0.0, index=all_events_idx)
                        res = {'ret': zeros, 'dd': zeros, 'ratio': 0.0}
                        cache_global[key] = res
                        return res

                    # Compute Labels & Metrics
                    _lbl, _ret, _mfe, _mae, _ = self._compute_dominance_labels(df, fam_events_local, family=fam_local, **t_obj.params)

                    # 1. Returns Series (Aligned)
                    s_evt = pd.to_numeric(_ret, errors='coerce').astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                    s_glob = pd.Series(0.0, index=all_events_idx, dtype=float)
                    common = s_evt.index.intersection(all_events_idx)
                    s_glob.loc[common] = s_evt.loc[common]

                    # 2. Drawdown Series
                    # Construct Equity Curve -> Drawdowns
                    equity = s_glob.cumsum()
                    running_max = equity.cummax()
                    drawdown = equity - running_max
                    # Fill NaN at start if any
                    drawdown = drawdown.fillna(0.0)

                    # 3. MFE/MAE Ratio (Behavioral)
                    # We compute mean MFE / mean MAE
                    # Use aligned local series for stats to avoid zeros bias
                    mfe_local = pd.to_numeric(_mfe, errors='coerce').fillna(0.0)
                    mae_local = pd.to_numeric(_mae, errors='coerce').fillna(0.0)

                    # Only consider trades that happened
                    valid_mask = s_evt.abs() > 1e-9
                    if valid_mask.sum() > 0:
                        avg_mfe = mfe_local[valid_mask].mean()
                        avg_mae = mae_local[valid_mask].mean()
                        if avg_mae > 1e-9:
                            ratio = avg_mfe / avg_mae
                        else:
                            ratio = 10.0 # High cap if no MAE
                    else:
                        ratio = 1.0

                    res = {
                        'ret': s_glob,
                        'dd': drawdown,
                        'ratio': float(min(ratio, 20.0)) # Cap at 20
                    }

                except Exception as e:
                    logger.warning(f"Error computing global metrics for {key}: {e}")
                    zeros = pd.Series(0.0, index=all_events_idx)
                    res = {'ret': zeros, 'dd': zeros, 'ratio': 0.0}

                cache_global[key] = res
                return res

            # -----------------------------------------------------------------
            # Final global (cross-family) diversification pass
            # -----------------------------------------------------------------
            # Re-apply clustering globally to ensure total diversity
            try:
                cfg_global = getattr(self, '_current_config', {})
                if not isinstance(cfg_global, dict):
                    cfg_global = {}
            except Exception:
                cfg_global = {}

            try:
                global_div_enabled = bool(cfg_global.get('layer2_global_diversification_enabled', True))
            except Exception:
                global_div_enabled = True

            if global_div_enabled and len(selected) > 1:
                try:
                    # Global Max
                    max_keep = int(cfg_global.get('layer2_global_max_geometries', 10))
                except Exception:
                    max_keep = 10

                if len(selected) <= max_keep:
                    kept = selected
                else:
                    # Global Clustering
                    all_candidates = list(selected)
                    n_glob = len(all_candidates)

                    # Gather metrics
                    metrics_list = [_get_metrics_global(c) for c in all_candidates]

                    # Distance Matrix Components
                    # 1. Return Correlation (Primary)
                    corr_mat_ret = np.zeros((n_glob, n_glob))
                    # 2. Drawdown Correlation (Risk)
                    corr_mat_dd = np.zeros((n_glob, n_glob))
                    # 3. Behavioral Diff (MFE/MAE Ratio)
                    dist_mat_beh = np.zeros((n_glob, n_glob))

                    # Extract ratio array for vectorized diff
                    ratios = np.array([m['ratio'] for m in metrics_list])
                    # Normalize ratios for distance calculation?
                    # Simple absolute difference is okay if scale is reasonable. Ratios are ~1.0-5.0 usually.

                    for i in range(n_glob):
                        ret_i = metrics_list[i]['ret']
                        dd_i = metrics_list[i]['dd']

                        for j in range(i, n_glob):
                            if i == j:
                                c_ret = 1.0
                                c_dd = 1.0
                                d_beh = 0.0
                            else:
                                ret_j = metrics_list[j]['ret']
                                dd_j = metrics_list[j]['dd']

                                c_ret = ret_i.corr(ret_j)
                                c_dd = dd_i.corr(dd_j)
                                d_beh = abs(ratios[i] - ratios[j])

                            corr_mat_ret[i, j] = c_ret
                            corr_mat_ret[j, i] = c_ret

                            corr_mat_dd[i, j] = c_dd
                            corr_mat_dd[j, i] = c_dd

                            dist_mat_beh[i, j] = d_beh
                            dist_mat_beh[j, i] = d_beh

                    # Clean NaNs
                    corr_mat_ret = np.nan_to_num(corr_mat_ret, nan=0.0)
                    corr_mat_dd = np.nan_to_num(corr_mat_dd, nan=0.0)

                    # Distances
                    d_ret = 1.0 - np.abs(corr_mat_ret)
                    d_dd = 1.0 - np.abs(corr_mat_dd)

                    # Normalize behavioral distance to [0, 1] roughly
                    max_beh = np.max(dist_mat_beh)
                    if max_beh > 1e-9:
                        d_beh_norm = dist_mat_beh / max_beh
                    else:
                        d_beh_norm = dist_mat_beh

                    # Composite Distance
                    # D = 0.6 * Return + 0.3 * Drawdown + 0.1 * Behavior
                    dist_mat = (0.6 * d_ret) + (0.3 * d_dd) + (0.1 * d_beh_norm)

                    np.fill_diagonal(dist_mat, 0.0)

                    try:
                        Z = linkage(squareform(dist_mat, checks=False), method='average')
                        labels = fcluster(Z, t=max_keep, criterion='maxclust')

                        kept = []
                        for clust_id in np.unique(labels):
                            indices = np.where(labels == clust_id)[0]
                            cluster_cands = [all_candidates[i] for i in indices]
                            # Pick best by final score within cluster
                            best = max(cluster_cands, key=lambda x: float(getattr(x, 'final_score', -1.0)))
                            kept.append(best)

                        logger.info(f"Global Clustering reduced {len(selected)} -> {len(kept)} geometries using Composite Distance (Ret+DD+Beh).")
                    except Exception as e:
                        logger.warning(f"Global clustering failed: {e}. Using score sort.")
                        kept = sorted(selected, key=lambda x: float(getattr(x, 'final_score', -1.0)), reverse=True)[:max_keep]

                selected = kept

        return selected

    def _train_geometry_models(
        self,
        df: pd.DataFrame,
        X_events: pd.DataFrame,
        events_df: pd.DataFrame,
        geometries: List[GeometryTrial],
        X_events_full: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Train simple LGBM models for each geometry on the provided training set
        to allow Out-Of-Sample prediction generation on the test set.

        Updated to use RobustFocalLoss and specified hyperparameters.
        """
        models = {}
        for g in geometries:
            try:
                lbls, _, _, _, _ = self._compute_dominance_labels(df, events_df, family=g.family, **g.params)
                valid_lbls = lbls.dropna()

                # Determine feature set: Use per-geometry selection if available, else fallback to X_events
                if getattr(g, 'selected_features', None) and X_events_full is not None:
                     cols = [c for c in g.selected_features if c in X_events_full.columns]
                     if cols:
                         X_base = X_events_full.loc[valid_lbls.index.intersection(X_events_full.index), cols]
                     else:
                         X_base = X_events.loc[valid_lbls.index.intersection(X_events.index)]
                else:
                     X_base = X_events.loc[valid_lbls.index.intersection(X_events.index)]

                common_idx = X_base.index
                
                if len(common_idx) < 20: 
                     models[g.uuid] = None
                     continue

                # Generate specific geometry features
                geo_features = self._compute_specific_geometry_features(df, common_idx, g.params)

                X_train = X_base

                # Append geometry features
                if not geo_features.empty:
                    # Align index just in case
                    geo_features = geo_features.reindex(common_idx).fillna(0.0)
                    X_train = pd.concat([X_train, geo_features], axis=1)

                y_train = valid_lbls.loc[common_idx]
                
                if len(y_train.unique()) < 2:
                    models[g.uuid] = None
                    continue

                # Base params from constants
                params = LAYER2_MODEL_CONSTANTS.copy()

                # Check for geometry-specific model params (HPO result)
                tuned_params = getattr(g, 'model_params', None)

                # Default Focal Loss params (Hunter Mode: Favor Positives)
                gamma_pos = 0.5
                gamma_neg = 1.25 # Derived from 2.5 * 0.5 default logic
                f_alpha = 0.65  # Increased from 0.25 (Favor positives > 50%)
                
                # Determine model type
                model_type = 'lgbm'  # Default
                if isinstance(tuned_params, dict) and tuned_params:
                    model_type = tuned_params.get('model_type', 'lgbm')
                    # Extract Focal params if present
                    if 'gamma_pos' in tuned_params:
                        gamma_pos = float(tuned_params['gamma_pos'])
                    elif 'focal_gamma' in tuned_params: # Fallback/Legacy
                         gamma_pos = float(tuned_params['focal_gamma'])

                    if 'gamma_neg' in tuned_params:
                        gamma_neg = float(tuned_params['gamma_neg'])
                    elif 'focal_gamma' in tuned_params: # Fallback/Legacy
                         gamma_neg = float(tuned_params['focal_gamma']) * 2.5

                    if 'focal_alpha' in tuned_params:
                        f_alpha = float(tuned_params['focal_alpha'])

                # Split for early stopping
                X_tr_inner, X_val_inner = X_train.iloc[:int(len(X_train)*0.9)], X_train.iloc[int(len(X_train)*0.9):]
                y_tr_inner, y_val_inner = y_train.iloc[:int(len(y_train)*0.9)], y_train.iloc[int(len(y_train)*0.9):]
                has_val = len(y_val_inner) >= 10

                # Train based on model type
                if model_type == 'lgbm':
                    # LGBM with Focal Loss + Calibration
                    if isinstance(tuned_params, dict):
                        params.update({k: v for k, v in tuned_params.items() if k in params})
                    
                    # Ensure n_estimators is integer
                    n_estimators = int(params.get('n_estimators', 500))

                    focal_obj = RobustFocalLoss(gamma_pos=gamma_pos, gamma_neg=gamma_neg, alpha=f_alpha)
                    
                    # Helper wrapper for objective compatibility with LGBMClassifier
                    def lgbm_focal_obj(y_true, y_pred):
                        # SKLearn API: func(y_true, y_pred)
                        # RobustFocalLoss expectations: func(preds, train_data)
                        return focal_obj(y_pred, y_true)

                    clf = lgb.LGBMClassifier(
                        objective=lgbm_focal_obj,
                        n_estimators=n_estimators,
                        num_leaves=int(params.get('num_leaves', 31)),
                        learning_rate=float(params.get('learning_rate', 0.05)),
                        # class_weight='balanced', # Disabled for Focal Loss
                        random_state=self.random_state,
                        n_jobs=1,
                        verbosity=-1,
                        metric='auc'  # Required for early_stopping with custom objective
                    )
                    
                    if has_val:
                        # Require both train and validation to have both classes for AUC metric
                        if y_tr_inner.nunique() < 2 or y_val_inner.nunique() < 2:
                            has_val = False

                    # Note: Early stopping with custom objectives (Focal Loss) can cause 
                    # 'tuple index out of range' errors in LightGBM's internal callback handling.
                    # Solution: Use a fixed number of estimators without early stopping when using
                    # custom objectives. The fallback is already handled but we minimize warnings.
                    if has_val:
                        # Fit without early stopping for custom objective to avoid callback errors
                        # but use fewer estimators as a conservative approach
                        clf.set_params(n_estimators=min(n_estimators, 300))
                        clf.fit(X_train, y_train)
                        # Calibrate on inner val (safe because eval data already purged)
                        try:
                            calibrated = CalibratedClassifierCV(clf, method='sigmoid', cv='prefit')
                            calibrated.fit(X_val_inner, y_val_inner)
                            models[g.uuid] = calibrated
                        except Exception as cal_e:
                            logger.warning(f"Calibration failed for {g.uuid}: {cal_e}. Using uncalibrated model.")
                            models[g.uuid] = clf
                    else:
                        clf.fit(X_train, y_train)
                        models[g.uuid] = clf

                elif model_type == 'xgb':
                    # XGBoost with Focal Loss
                    focal_obj = XGBFocalLoss(gamma_pos=gamma_pos, gamma_neg=gamma_neg, alpha=f_alpha)
                    
                    model_xgb = xgb.XGBClassifier(
                        n_estimators=tuned_params.get('n_estimators', 500) if tuned_params else 500,
                        learning_rate=tuned_params.get('learning_rate', 0.03) if tuned_params else 0.03,
                        max_depth=tuned_params.get('max_depth', 6) if tuned_params else 6,
                        objective=focal_obj,
                        eval_metric=['logloss', 'aucpr'], # De Prado: Monitor PR-AUC
                        early_stopping_rounds=50 if has_val else None,
                        verbosity=0,
                        random_state=self.random_state,
                        n_jobs=1,
                    )
                    
                    if has_val:
                        model_xgb.fit(
                            X_tr_inner, y_tr_inner,
                            eval_set=[(X_val_inner, y_val_inner)],
                            verbose=False,
                        )
                        # Calibrate
                        calibrated = CalibratedClassifierCV(model_xgb, method='sigmoid', cv='prefit')
                        calibrated.fit(X_val_inner, y_val_inner)
                        models[g.uuid] = calibrated
                    else:
                        model_xgb.fit(X_train, y_train)
                        models[g.uuid] = model_xgb

                elif model_type == 'catboost':
                    # CatBoost with Class Weights
                    class_weight_ratio = tuned_params.get('class_weight_ratio', 3.0) if tuned_params else 3.0
                    
                    model_cat = catboost.CatBoostClassifier(
                        iterations=tuned_params.get('iterations', 400) if tuned_params else 400,
                        learning_rate=tuned_params.get('learning_rate', 0.05) if tuned_params else 0.05,
                        depth=tuned_params.get('depth', 6) if tuned_params else 6,
                        class_weights={0: 1.0, 1: class_weight_ratio},
                        verbose=False,
                        random_seed=self.random_state,
                        thread_count=1,
                    )
                    
                    if has_val:
                        model_cat.fit(
                            X_tr_inner, y_tr_inner,
                            eval_set=(X_val_inner, y_val_inner),
                            early_stopping_rounds=30,
                        )
                        # Calibrate
                        calibrated = CalibratedClassifierCV(model_cat, method='sigmoid', cv='prefit')
                        calibrated.fit(X_val_inner, y_val_inner)
                        models[g.uuid] = calibrated
                    else:
                        model_cat.fit(X_train, y_train)
                        models[g.uuid] = model_cat

                else:
                    # Fallback to LGBM Classifier + Calibration
                    logger.warning(f"Unknown model type {model_type} for {g.uuid}, using LGBM")
                    
                    # Re-instantiate focal obj for wrapper
                    focal_obj = RobustFocalLoss(gamma_pos=f_gamma, gamma_neg=f_gamma * 2.5, alpha=f_alpha)
                    def lgbm_focal_obj_fb(y_true, y_pred):
                         return focal_obj(y_pred, y_true)

                    clf = lgb.LGBMClassifier(
                        objective=lgbm_focal_obj_fb,
                        n_estimators=500,
                        class_weight='balanced',
                        random_state=self.random_state,
                        n_jobs=1,
                        verbosity=-1,
                        metric='auc'
                    )

                    if has_val and y_val_inner.nunique() < 2:
                         has_val = False

                    if has_val:
                         clf.fit(
                             X_tr_inner, y_tr_inner, 
                             eval_set=[(X_val_inner, y_val_inner)], 
                             eval_metric='auc',
                             callbacks=[lgb.early_stopping(50, verbose=False)]
                         )
                         calibrated = CalibratedClassifierCV(clf, method='sigmoid', cv='prefit')
                         calibrated.fit(X_val_inner, y_val_inner)
                         models[g.uuid] = calibrated
                    else:
                         clf.fit(X_train, y_train)
                         models[g.uuid] = clf
            except Exception as e:
                logger.warning(f"Failed to train geometry model for {g.uuid}: {e}")
                models[g.uuid] = None
        return models

    def _bagged_labeling(
        self,
        df: pd.DataFrame,
        events_df: pd.DataFrame,
        geometries: List[GeometryTrial],
        trained_models: Optional[Dict[str, Any]] = None,
        X_events: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Step 3.4: Generate final bagged outputs with advanced weighting checks.

        Outputs:
        - Weighted Consensus Labels
        - Weighted Consensus Returns
        - Event Weights (capped and normalized)
        - Individual OOF Predictions (Probabilities)
        - Individual OOF Variances (Tree Variance)
        """

        # Ensure family assignment is up to date
        events_df = events_df.copy()
        events_df['family'] = self._assign_barrier_families(events_df)

        # Organize geometries by family
        geo_by_fam = {}
        for g in geometries:
            geo_by_fam.setdefault(g.family, []).append(g)

        # Storage for aggregation
        composite_labels = pd.Series(index=events_df.index, dtype=float)
        composite_prob = pd.Series(index=events_df.index, dtype=float)
        composite_mean_prob = pd.Series(index=events_df.index, dtype=float)
        composite_returns = pd.Series(index=events_df.index, dtype=float)
        composite_weights = pd.Series(index=events_df.index, dtype=float)
        oof_preds = {} # Store individual geometry predictions (probabilities)
        oof_vars = {} # Store individual geometry variances

        def _predict_probs(model, X_block: pd.DataFrame) -> np.ndarray:
            """Return class-1 probabilities for binary classifiers."""
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_block)
                proba_arr = np.asarray(proba)
                if proba_arr.ndim == 2:
                    if proba_arr.shape[1] == 1:
                        return proba_arr[:, 0]
                    return proba_arr[:, 1]
                return proba_arr.reshape(-1)
            margins = model.predict(X_block)
            margins_arr = np.asarray(margins).reshape(-1)
            return 1.0 / (1.0 + np.exp(-margins_arr))

        # Iterate by family (since events are disjoint by family)
        for family, fam_geos in geo_by_fam.items():
            fam_mask = events_df['family'] == family
            fam_events = events_df[fam_mask]

            if fam_events.empty: continue

            # Temporary storage for this family's calculations
            # Dimensions: (n_events, n_geometries)
            n_events = len(fam_events)
            n_geos = len(fam_geos)

            geo_labels_mat = np.zeros((n_events, n_geos))
            geo_returns_mat = np.zeros((n_events, n_geos))
            geo_probs_mat = np.zeros((n_events, n_geos))
            geo_scores_mat = np.zeros((n_events, n_geos))
            valid_mask_mat = np.zeros((n_events, n_geos), dtype=bool)

            # Pre-compute Efficiency Ratio for structure confidence
            try:
                # Use a standard window for ER or derive from config if possible
                er_window = 50
                # We need close prices for ER. df has 'close'.
                er_series = get_efficiency_ratio(df['close'], window=er_window)
                er_events = er_series.reindex(fam_events.index).fillna(0.0)

                # Define ER min/max for normalization
                er_min = 0.2
                er_max = 0.8

                w_structure_conf = np.clip((er_events.values - er_min) / (er_max - er_min), 0.0, 1.0)
            except Exception:
                w_structure_conf = np.ones(n_events)

            # Accumulator for Wsignalgate across geometries
            w_signalgate_accum = np.zeros(n_events)
            w_signalgate_count = np.zeros(n_events)

            for i, g in enumerate(fam_geos):
                # Compute labels/returns for this geometry
                lbls, rets, mfe, mae, _ = self._compute_dominance_labels(df, fam_events, family=family, **g.params)

                # Generate specific geometry features for all events in this family block
                geo_features = self._compute_specific_geometry_features(df, fam_events.index, g.params)

                # Store individual geometry output
                # OOF Fix: Use trained model if available and X_events provided
                pred_done = False

                # Initialize container
                oof_preds[g.uuid] = pd.Series(np.nan, index=fam_events.index)
                oof_vars[g.uuid] = pd.Series(np.nan, index=fam_events.index)

                if trained_models is not None and X_events is not None and g.uuid in trained_models:
                     booster = trained_models[g.uuid]
                     if booster is not None:
                         # Predict on fam_events
                         fam_indices = fam_events.index.intersection(X_events.index)
                         if not fam_indices.empty:
                             try:
                                 X_subset = X_events.loc[fam_indices]

                                 # Append geometry features
                                 if not geo_features.empty:
                                     geo_subset = geo_features.reindex(fam_indices).fillna(0.0)
                                     X_subset = pd.concat([X_subset, geo_subset], axis=1)

                                 probs = _predict_probs(booster, X_subset)
                                 probs = np.asarray(probs, dtype=float).reshape(-1)
                                 probs = np.clip(probs, 0.0, 1.0)
                                 probs = np.maximum(probs, 0.05)

                                 variances = _calculate_tree_variance(booster, X_subset)

                                 # Store
                                 oof_preds[g.uuid].loc[fam_indices] = probs
                                 oof_vars[g.uuid].loc[fam_indices] = variances
                             except Exception as e:
                                 logger.warning(f"OOF prediction failed for {g.uuid}: {e}")
                
                # If prediction failed or not available, leave as NaN (Layer 3 will handle fillna if needed, but for now we leave explicit NaN)

                # Align to fam_events index
                lbls_aligned = lbls.reindex(fam_events.index)
                rets_aligned = rets.reindex(fam_events.index)
                mfe_aligned = mfe.reindex(fam_events.index).fillna(0.0)
                mae_aligned = mae.reindex(fam_events.index).fillna(0.0)

                # Identify valid labels (not NaN)
                not_na = lbls_aligned.notna()

                # Fill matrices
                geo_labels_mat[not_na, i] = lbls_aligned[not_na]
                geo_returns_mat[not_na, i] = rets_aligned[not_na]
                geo_scores_mat[not_na, i] = g.final_score
                valid_mask_mat[not_na, i] = True

                try:
                    prob_s = oof_preds.get(g.uuid)
                    if isinstance(prob_s, pd.Series):
                        prob_aligned = pd.to_numeric(prob_s.reindex(fam_events.index), errors='coerce').astype(float)
                    else:
                        prob_aligned = pd.Series(np.nan, index=fam_events.index, dtype=float)
                    prob_vals = prob_aligned.to_numpy(dtype=float, copy=False)
                    fill_mask = (~np.isfinite(prob_vals)) & not_na.to_numpy(dtype=bool, copy=False)
                    if np.any(fill_mask):
                        prob_vals = prob_vals.copy()
                        # Do NOT fill with labels. Use 0.5 (neutral/uncertain) or 0.0.
                        # Using 0.5 implies we don't know, which is safer than leaking truth.
                        prob_vals[fill_mask] = 0.5
                    prob_vals = np.where(np.isfinite(prob_vals), prob_vals, 0.5)
                    prob_vals = np.clip(prob_vals, 0.0, 1.0)
                    geo_probs_mat[:, i] = prob_vals
                except Exception:
                    geo_probs_mat[not_na, i] = lbls_aligned[not_na].astype(float)

                # --- Compute Wsignalgate for this geometry ---
                # Wmagnitude = ln(1 + MFE)
                w_magnitude = np.log1p(np.maximum(0.0, mfe_aligned.values))

                # Wsmoothness = ln(1 + MFE/MAE)
                safe_mae = np.where(mae_aligned.values > 1e-9, mae_aligned.values, 1e-9)
                w_smoothness = np.log1p(np.maximum(0.0, mfe_aligned.values / safe_mae))

                # Wsignalgate_i
                w_sig_i = w_magnitude * w_smoothness * w_structure_conf

                # Accumulate for average
                # Only accumulate where valid
                w_signalgate_accum[not_na] += w_sig_i[not_na]
                w_signalgate_count[not_na] += 1

            if geo_labels_mat.shape != geo_returns_mat.shape or geo_labels_mat.shape != geo_scores_mat.shape:
                raise ValueError("Layer2 bagging: geometry matrices have inconsistent shapes")
            if geo_labels_mat.shape != valid_mask_mat.shape:
                raise ValueError("Layer2 bagging: valid mask has inconsistent shape")

            # --- Per-Geometry Capping Logic ---
            # Raw total score per event
            score_base_mat = np.maximum(geo_scores_mat, 0.0)
            score_base_mat[~valid_mask_mat] = 0.0
            all_zero_scores = bool(np.all(score_base_mat <= 0.0))
            if all_zero_scores:
                score_base_mat = valid_mask_mat.astype(float)

            event_total_score = np.sum(score_base_mat, axis=1)

            # Max contribution per geometry: 30% of event total
            max_contrib = 0.3 * event_total_score

            # Broadcast max_contrib to match geometry dimension
            max_contrib_mat = max_contrib[:, np.newaxis]

            # Cap the weights: min(score, max_contrib)
            capped_weights_mat = np.minimum(score_base_mat, max_contrib_mat)
            capped_weights_mat[~valid_mask_mat] = 0.0

            # Final Event Weight (sum of capped weights) - used for consensus averaging
            final_event_weights_consensus = np.sum(capped_weights_mat, axis=1)

            # Safety: ensure non-negative event weights
            final_event_weights_consensus = np.where(np.isfinite(final_event_weights_consensus), final_event_weights_consensus, 0.0)
            final_event_weights_consensus = np.maximum(final_event_weights_consensus, 0.0)

            if final_event_weights_consensus.shape[0] != n_events:
                raise ValueError("Layer2 bagging: final_event_weights_consensus shape mismatch")

            # Avoid division by zero
            safe_weights = final_event_weights_consensus.copy()
            safe_weights[safe_weights == 0] = 1.0 # arbitrary, will be 0 in result anyway

            # Weighted Consensus Calculation
            # Aggregation Logic: "At Least One" (Max) for Labels/Probs to prevent signal dilution.
            # Weighted Average is too conservative for diverse specialist geometries.

            # Configurable aggregation mode (De Prado 1.2 Future-proofing)
            agg_mode = str(getattr(self, '_current_config', {}).get('layer2_aggregation_mode', 'max')).lower()

            # For probs: Max probability (Default "max")
            # We also compute Mean for diagnostics (De Prado 1.2)

            valid_counts = np.sum(valid_mask_mat, axis=1)
            safe_counts = np.maximum(valid_counts, 1.0)

            sum_probs = np.sum(geo_probs_mat * valid_mask_mat.astype(float), axis=1)
            mean_probs = sum_probs / safe_counts

            max_probs = np.max(geo_probs_mat * valid_mask_mat.astype(float), axis=1)

            if agg_mode == 'mean':
                consensus_prob = mean_probs
                consensus_labels = (mean_probs >= 0.5).astype(float)
            elif agg_mode.startswith('vote'):
                # vote_k logic (e.g. vote_0.33)
                try:
                    k_vote = float(agg_mode.split('_')[1])
                except:
                    k_vote = 0.33

                # Count positive labels
                pos_votes = np.sum((geo_probs_mat >= 0.5) & valid_mask_mat, axis=1)
                vote_ratio = pos_votes / safe_counts
                consensus_labels = (vote_ratio >= k_vote).astype(float)
                # Keep consensus_prob = max_probs to preserve signal strength, but gate with labels
                consensus_prob = max_probs
                # Dampen probability if vote failed
                consensus_prob[consensus_labels == 0.0] = np.minimum(consensus_prob[consensus_labels == 0.0], 0.49)
            else:
                # Default: MAX (Logical OR)
                consensus_prob = max_probs
                consensus_labels = np.max(geo_labels_mat * valid_mask_mat.astype(float), axis=1)

            # Weighted Average Return (Keep conservative for PnL estimation)
            w_returns_sum = np.sum(geo_returns_mat * capped_weights_mat, axis=1)
            consensus_returns = w_returns_sum / safe_weights

            # Handle events with no valid geometries
            # Fix: Use fallback values instead of NaN to prevent Layer 3 data loss
            no_valid_geo = final_event_weights_consensus == 0
            consensus_labels[no_valid_geo] = 0.5  # Neutral label (uncertain)
            consensus_returns[no_valid_geo] = 0.0  # Zero return fallback
            consensus_prob[no_valid_geo] = 0.5  # Neutral probability
            mean_probs[no_valid_geo] = 0.5 # Safe value for diagnostics

            # --- Final Weight Logic: Wsignalgate ---
            # Average Wsignalgate across valid geometries for this event
            avg_w_signalgate = np.divide(
                w_signalgate_accum,
                w_signalgate_count,
                out=np.zeros_like(w_signalgate_accum),
                where=w_signalgate_count > 0
            )

            # --- Trade Quality Soft Weighting (Soft Labels) ---
            # Calculate w_quality based on consensus return magnitude and volatility.
            # Scaling: 0.5 to 2.0 based on z-score of return.
            # w = 0.5 + 1.5 * sigmoid( return / (vol * 2.0) )
            # This boosts high-quality trades (high sharpe/return) and dampens weak ones.
            try:
                ret_vals = consensus_returns.fillna(0.0).values
                vol_vals = df['volatility_1d'].reindex(fam_events.index).ffill().fillna(0.0).values

                # Z-score proxy
                safe_vol = np.where(vol_vals > 1e-9, vol_vals, 1e-9)
                z_score = ret_vals / safe_vol

                # Sigmoid scaling to [0.5, 2.0]
                # Center around z=0 (neutral) -> sigmoid(0)=0.5 -> w=0.5+0.75=1.25 (base)
                # If z huge -> sigmoid(10)=1 -> w=0.5+1.5=2.0
                # If z neg -> sigmoid(-10)=0 -> w=0.5+0=0.5
                sig = 1.0 / (1.0 + np.exp(-1.0 * z_score))
                w_quality = 0.5 + 1.5 * sig
            except Exception:
                w_quality = np.ones_like(avg_w_signalgate)

            # Multiply pre-existing weights (w_signalgate) by w_quality
            weighted_raw = avg_w_signalgate * w_quality

            # Apply MAD-based scaling to the aggregated weights to ensure comparability
            # and robustness, consistent with Layer 0 scaling.
            # finalize_sample_weights performs MAD clipping -> Mean centering (mean=1.0)
            if np.sum(weighted_raw) > 0:
                final_event_weights = finalize_sample_weights(weighted_raw)
            else:
                final_event_weights = np.zeros_like(weighted_raw)

            # Assign to main storage
            composite_labels.loc[fam_events.index] = consensus_labels
            composite_prob.loc[fam_events.index] = consensus_prob
            composite_mean_prob.loc[fam_events.index] = mean_probs
            composite_returns.loc[fam_events.index] = consensus_returns
            composite_weights.loc[fam_events.index] = final_event_weights

        # --- Global Fallback for Orphan Events (Families with No Passing Geometries) ---
        # If an event's family had all trials gate-rejected, it was never visited in the loop above.
        # Fill these orphan events with neutral defaults to prevent NaN propagation to Layer 3.
        orphan_mask = composite_returns.isna()
        if orphan_mask.any():
            n_orphans = int(orphan_mask.sum())
            logger.info(f"Layer2 Bagging: {n_orphans} orphan events (no geo coverage) filled with neutral defaults.")
            composite_labels.loc[orphan_mask] = 0.5  # Neutral label (uncertain)
            composite_prob.loc[orphan_mask] = 0.5  # Neutral probability
            composite_returns.loc[orphan_mask] = 0.0  # Zero return (no assumed PnL)
            composite_weights.loc[orphan_mask] = 0.1  # Low weight (low-information events)

        # --- Add Multi-Output Models (Cross-Geometry Learning) ---
        logger.info("\n>>> Training multi-output ensemble members...")
        
#        extratrees_preds = None
        pls_preds = None

        # Fill NaNs in weights with 0
        composite_weights = composite_weights.fillna(0.0)
        composite_weights = composite_weights.clip(lower=0.0)

        try:
            uniq_enabled = bool(getattr(self, '_current_config', {}).get('layer2_uniqueness_enabled', True))
        except Exception:
            uniq_enabled = True
        if uniq_enabled and int(len(events_df.index)) > 0 and int(len(df.index)) > 1:
            try:
                max_h = 0
                for g in list(geometries or []):
                    try:
                        if isinstance(getattr(g, 'params', None), dict):
                            h = int(g.params.get('horizon', 0))
                            if h > max_h:
                                max_h = h
                    except Exception:
                        continue
                horizon = int(getattr(self, '_current_config', {}).get('layer2_uniqueness_horizon', max_h))
                horizon = int(max(1, horizon))

                idx = df.index
                pos = idx.get_indexer(events_df.index)
                valid_pos = pos >= 0
                pos_v = pos[valid_pos]
                if pos_v.size > 0:
                    end_pos = np.minimum(pos_v + horizon, int(len(idx) - 1))
                    diff = np.zeros(int(len(idx)) + 1, dtype=float)
                    diff[pos_v] += 1.0
                    diff[end_pos + 1] -= 1.0
                    conc = np.cumsum(diff)[:-1]
                    conc = np.maximum(conc, 1.0)
                    inv = 1.0 / conc
                    inv_cum = np.cumsum(inv)
                    start = pos_v
                    end = end_pos
                    prev = np.zeros_like(start, dtype=float)
                    mask_prev = start > 0
                    if np.any(mask_prev):
                        prev[mask_prev] = inv_cum[start[mask_prev] - 1]
                    sum_inv = inv_cum[end] - prev
                    lengths = (end - start + 1).astype(float)
                    uniq = np.divide(sum_inv, lengths, out=np.ones_like(sum_inv), where=lengths > 0)

                    uniq_series = pd.Series(1.0, index=events_df.index, dtype=float)
                    uniq_series.iloc[np.where(valid_pos)[0]] = uniq

                    try:
                        alpha = float(getattr(self, '_current_config', {}).get('layer2_uniqueness_alpha', 1.0))
                    except Exception:
                        alpha = 1.0
                    if (not np.isfinite(alpha)) or float(alpha) < 0.0:
                        alpha = 1.0
                    mult = np.power(np.clip(uniq_series.values, 0.0, 1.0), float(alpha))
                    composite_weights *= pd.Series(mult, index=events_df.index)
            except Exception:
                pass

        # If everything is zero (can happen when all geometries fail), fall back to unit weights on labeled events.
        try:
            labeled_mask_global = composite_labels.notna()
        except Exception:
            labeled_mask_global = None
        if float(composite_weights.sum()) <= 0.0 and labeled_mask_global is not None:
            try:
                composite_weights.loc[labeled_mask_global] = 1.0
            except Exception:
                pass

        total_weight_global = composite_weights.sum()
        # Family capping logic removed.

        # Normalize final weights to mean=1.0 for stability
        mean_weight = composite_weights.mean()
        if (mean_weight is not None) and float(mean_weight) > 0:
            composite_weights /= float(mean_weight)

        score_thr = 0.5
        try:
            score_thr = float(getattr(self, '_current_config', {}).get('layer2_score_threshold', 0.5))
        except Exception:
            score_thr = 0.5
        if (not np.isfinite(score_thr)) or float(score_thr) <= 0.0 or float(score_thr) >= 1.0:
            score_thr = 0.5

        l2_score = composite_prob
        l2_label = pd.Series(np.nan, index=l2_score.index, dtype=float)
        try:
            valid = composite_labels.notna()
            l2_label.loc[valid] = (
                pd.to_numeric(composite_labels[valid], errors='coerce').astype(float) >= float(score_thr)
            ).astype(float)
        except Exception:
            pass

        l2_confidence = pd.Series(np.nan, index=l2_score.index, dtype=float)
        try:
            s = pd.to_numeric(l2_score, errors='coerce').astype(float).clip(lower=0.0, upper=1.0)
            p = np.clip(s.to_numpy(dtype=float, copy=False), 1e-12, 1.0 - 1e-12)
            h = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
            conf = 1.0 - (h / float(np.log(2.0)))
            l2_confidence.loc[:] = np.where(np.isfinite(conf), conf, np.nan)
            l2_confidence = l2_confidence.clip(lower=0.0, upper=1.0).where(l2_score.notna())
        except Exception:
            pass

        # Calculate global quality weights (for all events) for Layer 3 usage
        try:
            # Reconstruct global quality weight series
            # We computed w_quality per family block above, need to stitch it or recompute global.
            # Recomputing global is cleaner.
            c_ret = composite_returns.fillna(0.0)
            c_vol = df['volatility_1d'].reindex(composite_returns.index).ffill().fillna(0.0)
            safe_v = np.where(c_vol > 1e-9, c_vol, 1e-9)
            z = c_ret / safe_v
            sig = 1.0 / (1.0 + np.exp(-1.0 * z))
            quality_weights = pd.Series(0.5 + 1.5 * sig, index=composite_returns.index)
        except Exception:
            quality_weights = pd.Series(1.0, index=composite_returns.index)

        # De Prado 1.2: Validation Diagnostics
        n_base = int(len(events_df))
        n_bagged = int((l2_label == 1.0).sum())
        inflation_ratio = n_bagged / max(1, n_base)

        return {
            "oof_labels": l2_score,
            "oof_returns": composite_returns,
            "weights": composite_weights,
            "quality_weights": quality_weights,
            "l2_score": l2_score,
            "l2_label": l2_label,
            "l2_confidence": l2_confidence,
            "individual_geometries": oof_preds,
            "individual_variances": oof_vars,
            "selected_trials": [asdict(t) for t in geometries],
            "diagnostics": {
                "signal_inflation_ratio": inflation_ratio,
                "n_bagged_signals": n_bagged,
                "n_base_events": n_base,
                "mean_consensus_prob": composite_mean_prob
            }
        }
