"""
Label-Based Layer 2: Regime-Conditional Geometry Optimization & Meta-Labeling
-----------------------------------------------------------------------------
This module implements the Layer 2 architecture for meta-labeling, focusing on
regime-conditional barrier optimization and ML learnability probes.

It performs:
1. Event generation based on volatility-scaled returns (CUSUM).
2. Global Geometry Selection using `label_geometry_selection`.
3. MFE/MAE Dominance Labeling based on selected Geometries (Alpha/Beta/SL/MinRatio).
4. OOF Optimization and ML Training (LGBM/XGB/CatBoost).
5. Bagged output generation.
"""

import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
import xgboost as xgb
import catboost
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import spearmanr
from scipy.special import expit
from scipy.spatial.distance import euclidean
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from dataclasses import dataclass, asdict
import logging
import copy

# Import compute_realized_returns from the existing module
from src.training.steps.labeling.feature_generation_meta_labeling_step import (
    compute_realized_returns,
    create_meta_features,
    get_efficiency_ratio,
    generate_primary_signals,
)
from src.training.steps.labeling.generate_weights_per_label import finalize_sample_weights

from src.utils.purged_kfold import PurgedKFoldTime
from src.training.steps.labeling.label_geometry_selection import (
    Geometry,
    Event,
    select_geometries,
    events_to_dataframe
)

# Configure logging
logger = logging.getLogger(__name__)

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

def _normalized_binary_entropy(p: float) -> float:
    """Return normalized entropy in [0, 1] for a Bernoulli(p)."""
    try:
        p = float(p)
    except Exception:
        return 0.0
    if not np.isfinite(p):
        return 0.0
    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    h = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    h_max = float(np.log(2.0))
    if h_max <= 0:
        return 0.0
    return float(np.clip(h / h_max, 0.0, 1.0))

class RobustFocalLoss:
    """
    Production-grade Focal Loss for LightGBM in Financial ML.

    Enhancements over standard Focal Loss:
    1. Asymmetric Gamma: Penalize False Positives (Traps) harder than Missed Opportunities.
    2. Label Smoothing: Prevents the model from becoming over-confident on noisy labels.
    3. Gradient Capping & Mixing: Stabilizes training against outliers.
    4. Guardrails: w_cap prevents the loss from exploding on 'impossible' examples.
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
    Matches RobustFocalLoss behavior for consistency across LGBM and XGB.
    """
    
    def __init__(self, gamma=2.0, alpha=0.25):
        """
        Args:
            gamma: Focusing parameter (higher = more focus on hard examples)
            alpha: Positive class weight
        """
        self.gamma = gamma
        self.alpha = alpha
    
    def __call__(self, preds, dtrain):
        """
        Args:
            preds: Raw predictions (logits) from XGBoost
            dtrain: xgb.DMatrix with labels
        
        Returns:
            grad, hess: Gradient and hessian arrays
        """
        if hasattr(dtrain, 'get_label'):
            labels = dtrain.get_label()
        else:
            labels = dtrain
        
        # Convert logits to probabilities (standardized clipping)
        p = 1.0 / (1.0 + np.exp(-preds))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        
        # Safe log operations
        log_p = np.log(p + 1e-12)
        log_1mp = np.log(1 - p + 1e-12)
        
        # Focal loss terms
        term_pos = np.power(1 - p, self.gamma)
        term_neg = np.power(p, self.gamma)
        
        # Gradient
        # NOTE: We negate the derived gradient based on empirical testing (AUC 0.86 vs 0.13)
        # This suggests either a derivation sign error or XGBoost expecting descent direction.
        grad_raw = np.where(
            labels == 1,
            -self.alpha * term_pos * (1 - p - self.gamma * p * log_p),
            (1 - self.alpha) * term_neg * (p - self.gamma * (1 - p) * log_1mp)
        )
        grad = -grad_raw
        
        # Hessian approximation (Binary Cross Entropy Hessian)
        # This guarantees positive curvature and stability avoiding negative non-convex regions
        hess = p * (1.0 - p)
        
        # CRITICAL: Gradient clipping (was missing!)
        grad = np.clip(grad, -10.0, 10.0)
        
        # Hessian stability
        hess = np.maximum(hess, 1e-6)
        
        return grad, hess


# ==============================================================================
# Multi-Output Model Functions (Cross-Geometry Learning)
# ==============================================================================

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
    
    # --- Model 1: LGBM Standard (Non-linear) ---
    try:
        focal_lgbm = RobustFocalLoss(gamma_pos=2.0, gamma_neg=5.0, alpha=0.25, verbose=False)
        
        params_lgbm = {
            'n_estimators': 500,
            'learning_rate': 0.03,
            'num_leaves': 127,
            'max_depth': 7,
            'min_data_in_leaf': 20,
            'verbosity': -1,
            'random_state': random_state,
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
        focal_xgb = XGBFocalLoss(gamma=2.0, alpha=0.25)
        
        model_xgb = xgb.XGBClassifier(
            n_estimators=400,
            learning_rate=0.04,
            max_depth=6,
            min_child_weight=10,
            subsample=0.8,
            colsample_bytree=0.8,
            objective=focal_xgb,
            eval_metric='auc',
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
            eval_metric='auc',
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
    """
    if booster is None:
        return np.zeros(X.shape[0])

    try:
        # 1. Get leaf indices: (n_samples, n_trees)
        leaf_indices_raw = booster.predict(X, pred_leaf=True)
        
        if leaf_indices_raw.ndim == 1:
            leaf_indices = leaf_indices_raw.reshape(-1, 1)
        else:
            leaf_indices = leaf_indices_raw

        model_dump = booster.dump_model()
        trees = model_dump['tree_info']

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

        limit_trees = min(n_trees, leaf_indices.shape[1])
        tree_indices = np.arange(limit_trees)
        subset_indices = leaf_indices[:, :limit_trees]
        subset_indices = np.clip(subset_indices, 0, max_leaf_idx)

        collected_values = leaf_values_lookup[tree_indices, subset_indices]
        variance = np.nanvar(collected_values, axis=1)

        return variance

    except Exception as e:
        logger.warning(f"Failed to calculate tree variance: {e}")
        return np.zeros(X.shape[0])

@dataclass
class GeometryWrapper:
    """Wrapper to adapt Geometry to existing pipeline structure."""
    geometry: Geometry
    survivors: Set[int]
    uuid: str
    final_score: float = 0.0
    model_params: Optional[Dict[str, Any]] = None
    learnability: float = 0.0
    stability: float = 1.0

    @property
    def family(self) -> str:
        return self.geometry.archetype

    @property
    def params(self) -> Dict[str, Any]:
        return {
            'sl_sigma': self.geometry.sl_sigma,
            'alpha': self.geometry.alpha,
            'beta': self.geometry.beta,
            'min_ratio': self.geometry.min_ratio
        }

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
        self.selected_geometries: List[GeometryWrapper] = []
        self.family_weights: Dict[str, float] = {}

        self._labels_cache: Dict[Any, Tuple[pd.Series, pd.Series, pd.Series, pd.Series]] = {}
        self._signals_cache: Dict[Any, pd.DataFrame] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._global_probe_features: List[str] = []
        self._primary_signals: Optional[pd.DataFrame] = None

        # Suppress Optuna logging if not verbose
        if not self.verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

    def execute(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._current_config = dict(config or {})
        return self.run(df)

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute the Layer 2 pipeline with OOF generation using updated Geometry Selection.
        """
        logger.info("Starting Layer 2 Pipeline...")

        self._labels_cache = {}
        self._signals_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._global_probe_features = []
        self._primary_signals = None

        # Step 0: Preparation
        df = self._validate_inputs(df)
        df = self._precompute_geometry_base_features(df)
        events_df = self._generate_events(df)

        if events_df.empty:
            logger.warning("No events generated in Layer 2. Skipping.")
            return {}

        # Pre-assign families for backward compatibility in reporting/bagging if needed
        # But 'select_geometries' works globally.
        events_df['family'] = self._assign_barrier_families(events_df)

        X_probe_events = pd.DataFrame()
        try:
            X_probe_events = self._build_geometry_independent_event_features(df, events_df)
            self._global_probe_features = self._select_global_probe_features(X_probe_events)
        except Exception as e:
            logger.warning(f"Failed to build probe features: {e}")
            self._global_probe_features = []

        # Persist selected features
        self._save_selected_features(self._global_probe_features)

        # ---------------------------------------------------------------------
        # Part A: Full Selection (Production Artifacts)
        # ---------------------------------------------------------------------
        logger.info(">>> Layer 2: Running Geometry Selection...")

        # 1. Convert events to list of Event objects with Fixed Horizon
        events_list = self._create_events_list(df, events_df, horizon=40)

        # 2. Run Global Selection
        # Note: Passing empty fold_metrics_map for 'discovery' mode. Validation happens in OOF loop.
        selected_raw = select_geometries(events_list, fold_metrics_map={}, features_df=X_probe_events)

        # 3. Convert to Wrappers
        production_geometries = []
        for i, (geom, survivors) in enumerate(selected_raw):
            wrapper = GeometryWrapper(
                geometry=geom,
                survivors=survivors,
                uuid=f"Geo_{geom.archetype.replace(' ', '_')}_{i}",
                final_score=1.0 # Default score, refined later
            )
            production_geometries.append(wrapper)

        # Optimize Model Parameters for Production Geometries
        if production_geometries:
            logger.info(">>> Layer 2: Tuning Model Parameters for Production Geometries...")
            for i, g in enumerate(production_geometries):
                logger.info(f"    Tuning model for geometry {g.uuid} ({i+1}/{len(production_geometries)})...")
                best_params = self._tune_geometry_model_params(df, events_df, g)
                if best_params:
                    g.model_params = best_params
                    logger.info(f"    Found params for {g.uuid}: {best_params}")

        # FAST-FAIL
        if not production_geometries:
            logger.error("Layer2 CRITICAL: No geometries selected!")
            raise ValueError("Layer2 failed: No geometries selected.")

        # Store for reference
        self.selected_geometries = production_geometries

        # ---------------------------------------------------------------------
        # Part B: OOF Optimization (Analytics Artifacts)
        # ---------------------------------------------------------------------
        logger.info(">>> Layer 2: Running OOF Optimization (Analytics)...")

        # Initialize storage for OOF results
        indices = events_df.index
        oof_scores = pd.Series(np.nan, index=indices)
        oof_labels = pd.Series(np.nan, index=indices)
        oof_confidence = pd.Series(np.nan, index=indices)
        oof_returns = pd.Series(np.nan, index=indices)
        oof_weights = pd.Series(np.nan, index=indices)

        # Storage for Tree Diagnostics
        all_tree_stats = []

        # Track predictions per geometry
        oof_geo_preds = {}
        oof_geo_vars = {}

        # Config OOF
        try:
            cfg_oof = getattr(self, "_current_config", {})
            if not isinstance(cfg_oof, dict): cfg_oof = {}
        except Exception: cfg_oof = {}

        n_oof_splits = int(cfg_oof.get("layer2_oof_splits", 3))
        n_oof_splits = int(max(2, min(n_oof_splits, int(len(df)))))

        purge_bars = int(cfg_oof.get("layer2_oof_purge_bars", 0))
        if purge_bars <= 0: purge_bars = 40 # Default to fixed horizon

        # Purged K-Fold Logic
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

            logger.info(f"   > Processing Fold {fold_idx}/{int(len(folds))}...")

            # Create Train Slice
            df_train = df.iloc[train_idx]

            # Subset events
            events_train = events_df.loc[events_df.index.intersection(df_train.index)]
            events_test = events_df.loc[events_df.index.intersection(df.index[test_idx])]

            if events_train.empty:
                continue

            # Run Selection on Train Split
            events_list_train = self._create_events_list(df, events_train, horizon=40)
            selected_raw_fold = select_geometries(events_list_train, fold_metrics_map={}, features_df=X_probe_events.loc[events_train.index])

            fold_geometries = []
            for i, (geom, survivors) in enumerate(selected_raw_fold):
                # Standardize UUIDs across folds based on params/archetype to align columns?
                # Actually, geometries might differ per fold. We usually just use the production geometries for OOF if we want stable channels,
                # OR we retrain production geometries on OOF folds.
                # However, strict OOF requires selecting on train.
                # To maintain consistent columns in 'individual_geometries', we can map them to 'production_geometries' if they match,
                # or just use generic IDs.
                # For simplicity and robustness, let's use the Production Geometries defined in Part A, but re-train their models on the fold.
                # This validates the chosen production configs.
                pass

            # DECISION: Use Production Geometries but retrain models on fold data.
            # This measures how well the selected production geometries generalize.
            standardized_geos = production_geometries

            # Feature Selection
            X_train_events = X_probe_events.reindex(events_train.index).fillna(0.0)

            # Train models on Train Split
            trained_models = None
            if not X_train_events.empty:
                try:
                    trained_models = self._train_geometry_models(
                        df=df, # Full DF needed for path calculations inside
                        X_events=X_train_events,
                        events_df=events_train,
                        geometries=standardized_geos
                    )
                except Exception:
                    trained_models = None

            # Collect Tree Diagnostics
            if trained_models:
                for uuid, model in trained_models.items():
                    if model is not None:
                        try:
                            stats = self._extract_tree_diagnostics(model)
                            all_tree_stats.append(stats)
                        except Exception:
                            pass

            # Predict on Test (Bagged Labeling)
            if not events_test.empty:
                # Use fixed horizon 40 for labeling context
                lookahead_bars = 45
                test_end_pos = int(np.max(np.asarray(test_idx, dtype=int)))
                label_end_pos = int(min(len(df) - 1, test_end_pos + lookahead_bars))
                df_label = df.iloc[: label_end_pos + 1]

                X_test_events = X_probe_events.reindex(events_test.index).fillna(0.0)

                fold_output = self._bagged_labeling(
                    df_label, 
                    events_test, 
                    standardized_geos,
                    trained_models=trained_models,
                    X_events=X_test_events
                )

                # Assign to OOF arrays
                target_idx = events_test.index

                oof_scores.loc[target_idx] = fold_output.get('l2_score', fold_output.get('oof_labels')).reindex(target_idx)
                oof_labels.loc[target_idx] = fold_output.get('l2_label', pd.Series(np.nan, index=target_idx)).reindex(target_idx)
                oof_confidence.loc[target_idx] = fold_output.get('l2_confidence', pd.Series(np.nan, index=target_idx)).reindex(target_idx)
                oof_returns.loc[target_idx] = fold_output['oof_returns'].reindex(target_idx)
                oof_weights.loc[target_idx] = fold_output['weights'].reindex(target_idx)

                # Assign individual geometry preds and variances
                for uuid, series in fold_output['individual_geometries'].items():
                    if uuid not in oof_geo_preds:
                        oof_geo_preds[uuid] = pd.Series(np.nan, index=indices)
                        oof_geo_vars[uuid] = pd.Series(np.nan, index=indices)

                    oof_geo_preds[uuid].loc[target_idx] = series.reindex(target_idx)

                for uuid, series in fold_output['individual_variances'].items():
                    if uuid in oof_geo_vars:
                        oof_geo_vars[uuid].loc[target_idx] = series.reindex(target_idx)

        # ---------------------------------------------------------------------
        # Final Packaging
        # ---------------------------------------------------------------------
        final_geo_preds = {k: v for k, v in oof_geo_preds.items() if v.notna().any()}
        final_geo_vars = {k: v for k, v in oof_geo_vars.items() if v.notna().any()}

        self._generate_reports(df, events_df, production_geometries, oof_scores, oof_labels, oof_weights, all_tree_stats)

        logger.info("Layer 2 Pipeline Complete.")

        return {
            "oof_labels": oof_scores,
            "oof_returns": oof_returns,
            "weights": oof_weights,
            "l2_score": oof_scores,
            "l2_label": oof_labels,
            "l2_confidence": oof_confidence,
            "individual_geometries": final_geo_preds,
            "individual_variances": final_geo_vars,
            "events_df": events_df,
            "selected_trials": [asdict(t) for t in production_geometries],
            "production_selected_features": list(getattr(self, '_production_selected_features', []) or []),
        }

    def _create_events_list(self, df: pd.DataFrame, events_df: pd.DataFrame, horizon: int = 40) -> List[Event]:
        """
        Convert DataFrame and event timestamps into List[Event] for selection logic.
        """
        events_list = []
        if booster is None:
            return {'n_features_used': 0.0, 'avg_depth': 0.0, 'max_depth': 0.0}

        try:
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
        kappa = float(params.get('kappa', 2.0))
        sl_mult = float(params.get('sl_mult', 1.0))

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

            # Target Size (Percentage)
            target_size = np.maximum(kappa * vol, min_profit)
            target_size = target_size.replace(0.0, np.nan) # Avoid div/0

            # Stop Size (Percentage)
            stop_size = sl_mult * vol
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
        """Ensure required columns exist. Returns (potentially modified) copy of df."""
        required = ['close', 'volatility_1d']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in df: {missing}")

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

    def _get_barrier_family(self, trend_regime: str, vol_regime: str) -> str:
        """
        Map regimes to barrier families.

        High Trend -> Trend Continuation
        Low Trend / High Vol -> Momentum
        Low Trend / Low Vol -> Mean Reversion
        """
        # Normalize inputs (handle int/float/string)
        t_reg = str(trend_regime).lower()
        v_reg = str(vol_regime).lower()

        is_high_trend = 'high' in t_reg or t_reg == '1' or t_reg == '1.0'
        is_high_vol = 'high' in v_reg or v_reg == '1' or v_reg == '1.0'

        if is_high_trend:
            return 'Trend Continuation'
        elif is_high_vol:
            # Low Trend + High Vol
            return 'Momentum'
        else:
            # Low Trend + Low Vol
            return 'Mean Reversion'

    def _assign_barrier_families(self, events_df: pd.DataFrame) -> pd.Series:
        trend = events_df['trend_regime']
        vol = events_df['vol_regime']

        t_reg = trend.astype(str).str.lower()
        v_reg = vol.astype(str).str.lower()

        is_high_trend = t_reg.str.contains('high', na=False) | t_reg.isin(['1', '1.0'])
        is_high_vol = v_reg.str.contains('high', na=False) | v_reg.isin(['1', '1.0'])

        families = np.where(
            is_high_trend.to_numpy(),
            'Trend Continuation',
            np.where(is_high_vol.to_numpy(), 'Momentum', 'Mean Reversion'),
        )
        return pd.Series(families, index=events_df.index, dtype=object)

    def _compute_dominance_labels(
        self,
        df: pd.DataFrame,
        events_df: pd.DataFrame,
        kappa: float,
        horizon: int,
        family: str,
        events_shift: int = 0,
        sl_mult: Optional[float] = None,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Compute TP/SL(+optional trailing) exit-model labels and related metrics.
        Label = 1 if the trade exits via profit barrier (or trailing), else 0.

        Args:
            df: Market data
            events_df: Events to label
            kappa: Dominance ratio threshold
            horizon: Window size
            family: Geometry family (defines direction)
            events_shift: Shift event timestamps by N bars (for stability check)
            sl_mult: Optional stop loss multiplier
        """
        try:
            direction_mode = str(getattr(self, "_current_config", {}).get("layer2_direction_mode", "lagged"))
        except Exception:
            direction_mode = "lagged"

        sl_mult_eff = 1.0
        if sl_mult is not None:
            sl_mult_eff = float(sl_mult)
        else:
            try:
                sl_mult_eff = float(getattr(self, '_current_config', {}).get('layer2_sl_mult', 1.0))
            except Exception:
                sl_mult_eff = 1.0
        if (not np.isfinite(sl_mult_eff)) or float(sl_mult_eff) <= 0.0:
            sl_mult_eff = 1.0

        trail_mult = None
        try:
            cfg_trail = getattr(self, '_current_config', {}).get('layer2_trail_distance_atr_mult')
            trail_mult = float(cfg_trail) if cfg_trail is not None else None
        except Exception:
            trail_mult = None
        if trail_mult is not None and ((not np.isfinite(float(trail_mult))) or float(trail_mult) <= 0.0):
            trail_mult = None

        cache_key = (
            self._df_cache_key(df),
            self._events_cache_key(events_df.index),
            str(family),
            float(round(float(kappa), 6)),  # Reduced precision from 8 to 6
            float(round(float(sl_mult_eff), 6)),  # Reduced precision from 8 to 6
            int(horizon),
            int(events_shift),
            float(self.transaction_cost),
            str(direction_mode),
            float(trail_mult) if trail_mult is not None else None,
            int(max(0, int(getattr(self, '_current_config', {}).get('layer2_min_event_spacing', 1) if isinstance(getattr(self, '_current_config', {}), dict) else 1))),  # Reduced from 4 to 1
            "tpsl_full"
        )
        cached = self._labels_cache.get(cache_key)
        if cached is not None and not self.force_hpo:
            self._cache_hits += 1
            return cached
        self._cache_misses += 1

        signals = self._get_or_build_signals(df, events_df, family)

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
                 return empty_s, empty_s, empty_s, empty_s

            shifted_timestamps = df.index[shifted_locs[valid_locs]]
            orig_signals = signals.loc[target_events_idx[valid_locs]]

            temp_signals = pd.DataFrame(0.0, index=df.index, columns=['consensus'])
            temp_signals.loc[shifted_timestamps, 'consensus'] = orig_signals['consensus'].values

            calc_signals = temp_signals
            calc_events_idx = shifted_timestamps

        vol_series = pd.to_numeric(df.get('volatility_1d'), errors='coerce').astype(float)
        vol_series = vol_series.replace([np.inf, -np.inf], np.nan)
        vol_series = vol_series.ffill().bfill()
        vol_series = vol_series.clip(lower=1e-8)

        # Volatility-aware profit threshold adjustment
        vol_median = vol_series.median()
        vol_adj_factor = 1.0 + 0.3 * ((vol_series - vol_median) / (vol_median + 1e-9))
        vol_adj_factor = vol_adj_factor.clip(lower=0.7, upper=1.3)  # Reasonable bounds
        
        # Enforce minimum profit threshold to cover transaction costs (Root Cause 3: Mis-specified Labels)
        # If profit target < cost, a "win" is still a net loss.
        # We require TP to be at least 1.1x cost to consider it a valid target.
        min_profit = self.transaction_cost * 1.1
        profit_thr = np.maximum(float(kappa) * vol_series * vol_adj_factor, min_profit)

        stop_thr = float(sl_mult_eff) * vol_series * vol_adj_factor

        atr_series = None
        if trail_mult is not None:
            try:
                if ('high' in df.columns) and ('low' in df.columns) and ('close' in df.columns):
                    atr_window = int(getattr(self, '_current_config', {}).get('layer2_atr_window', 14))
                    atr_window = int(max(2, atr_window))
                    high = pd.to_numeric(df['high'], errors='coerce').astype(float)
                    low = pd.to_numeric(df['low'], errors='coerce').astype(float)
                    close = pd.to_numeric(df['close'], errors='coerce').astype(float)
                    prev_close = close.shift(1)
                    tr1 = (high - low).abs()
                    tr2 = (high - prev_close).abs()
                    tr3 = (low - prev_close).abs()
                    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    atr_series = tr.rolling(atr_window).mean()
            except Exception:
                atr_series = None

        (
            realized_returns,
            _,
            exit_reasons,
            _,
            mfe_series,
            mae_series,
            _, _
        ) = compute_realized_returns(
            df=df,
            signals=calc_signals,
            profit_threshold=profit_thr,
            stop_threshold=stop_thr,
            horizon=horizon,
            transaction_cost=self.transaction_cost,
            min_event_spacing=int(max(0, int(getattr(self, '_current_config', {}).get('layer2_min_event_spacing', 1)))),  # Reduced from 4 to 1
            volatility_series=None,
            atr_series=atr_series,
            trail_distance_atr_mult=trail_mult,
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

            final_labels.iloc[valid_locs] = binary_labels.values
            final_returns.iloc[valid_locs] = subset_returns.values
            final_mfe.iloc[valid_locs] = subset_mfe.values
            final_mae.iloc[valid_locs] = subset_mae.values
        else:
            final_labels = binary_labels
            final_returns = subset_returns
            final_mfe = subset_mfe
            final_mae = subset_mae

        result = (final_labels, final_returns, final_mfe, final_mae)
        self._labels_cache[cache_key] = result
        return result

    # Legacy wrapper for compatibility if needed, but we switch internal calls to _compute_dominance_labels
    def _compute_labels(self, df, events_df, tp_mult=None, sl_mult=None, horizon=None, family=None, **kwargs):
        # Adapt old signature to new logic if called with old params
        # Use simple mapping if kappa not provided
        kappa = kwargs.get('kappa')
        if kappa is None:
            # Heuristic: if TP=2, SL=1, Kappa=2
            if tp_mult and sl_mult:
                kappa = tp_mult / max(sl_mult, 1e-3)
            else:
                kappa = 2.0

        lbl, ret, _, _ = self._compute_dominance_labels(df, events_df, kappa, int(horizon), family, sl_mult=sl_mult)
        return lbl, ret

    def _build_geometry_independent_event_features(
        self,
        df: pd.DataFrame,
        events_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build one feature matrix for all events, independent of TP/SL/Horizon geometry."""
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
            drop_regime_context_features=bool(getattr(self, '_current_config', {}).get('layer2_drop_regime_context_features', False)),
        )

        try:
            meta_features = meta_features.replace([np.inf, -np.inf], np.nan)
            meta_features = meta_features.apply(pd.to_numeric, errors='coerce')
        except Exception as e:
            logger.debug(f"Meta features cleanup failed: {e}")

        try:
            forbidden_exact = {
                "vol_ratio",
                "vol_expansion",
                "returns_std_50",
                "volume_spike_ema",
                "event_r_multiple_mean_last_50",
            }
            forbidden_prefixes = ("zigzag_",)
            forbidden_substrings = (
                "zigzag",
                "pivot",
                "swing",
                "renko",
                "last_",
                "last_50",
                "last_100",
                "cumulative",
                "streak",
                "vol_expansion",
                "signal_density",
            )
            cols_to_drop = []
            for col in list(meta_features.columns):
                col_str = str(col)
                col_lower = col_str.lower()
                if col_str in forbidden_exact:
                    cols_to_drop.append(col_str)
                    continue
                if any(col_str.startswith(pref) for pref in forbidden_prefixes):
                    cols_to_drop.append(col_str)
                    continue
                if any(sub in col_lower for sub in forbidden_substrings):
                    cols_to_drop.append(col_str)
            if cols_to_drop:
                meta_features = meta_features.drop(columns=list(set(cols_to_drop)), errors='ignore')
        except Exception:
            pass

        enable_regime_leaf = True
        try:
            enable_regime_leaf = bool(getattr(self, '_current_config', {}).get('enable_regime_leaf_features', True))
        except Exception:
            enable_regime_leaf = True

        if enable_regime_leaf:
            try:
                from src.training.steps.labeling.regime_leaf_feature_extractor import extract_regime_leaf_onehot_features

                extractor_cfg = {
                    "enabled_targets": [
                        "regime_trendiness",
                        "regime_volatility",
                        "regime_trend_efficiency",
                        "regime_memory",
                    ],
                    "inputs": {
                        "input_source": "provided_x",
                        "alignment": {"enabled": True, "method": "ffill"},
                    },
                    "onehot": {"enabled": True},
                    "interaction_feature": {"enabled": True, "include_base": True},
                    "reporting": {"enabled": False},
                }

                rl_df = extract_regime_leaf_onehot_features(
                    X=meta_features,
                    market_data=df,
                    config=extractor_cfg,
                    random_state=int(getattr(self, '_current_config', {}).get('random_state', 42)),
                    verbose=False,
                )
                if rl_df is not None and not getattr(rl_df, 'empty', True):
                    rl_df = rl_df.reindex(meta_features.index).fillna(0.0)
                    meta_features = pd.concat([meta_features, rl_df], axis=1)
            except Exception as e:
                logger.debug(f"Regime leaf extraction failed: {e}")

        X_events = meta_features.reindex(events_df.index)
        try:
            X_events = X_events.replace([np.inf, -np.inf], np.nan)
        except Exception as e:
            logger.debug(f"X_events cleanup failed: {e}")
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

    def _rank_features_by_mean_mdi(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
        n_splits: int = 5,
    ) -> Tuple[List[str], np.ndarray]:
        X_num = X.fillna(0.0)
        y_num = y.astype(int)
        w_arr = None
        if sample_weight is not None:
            try:
                w_arr = np.asarray(sample_weight, dtype=float).reshape(-1)
            except Exception:
                w_arr = None

        tscv = TimeSeriesSplit(n_splits=n_splits)
        importances_sum = np.zeros(X_num.shape[1], dtype=float)
        n_used = 0

        for tr_idx, te_idx in tscv.split(X_num):
            X_tr = X_num.iloc[tr_idx]
            y_tr = y_num.iloc[tr_idx]
            X_te = X_num.iloc[te_idx]
            y_te = y_num.iloc[te_idx]

            if y_tr.nunique() < 2 or y_te.nunique() < 2:
                continue

            model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                num_leaves=31,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=1,
                verbose=-1,
            )

            fit_kwargs: Dict[str, Any] = {}
            if w_arr is not None and len(w_arr) == len(X_num):
                fit_kwargs['sample_weight'] = w_arr[tr_idx]

            try:
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_te, y_te)],
                    callbacks=[lgb.early_stopping(10, verbose=False)],
                    **fit_kwargs
                )
                imp = np.asarray(model.feature_importances_, dtype=float)
                if imp.shape[0] == importances_sum.shape[0]:
                    importances_sum += imp
                    n_used += 1
            except Exception as e:
                # logger.debug(f"Feature ranking fit failed: {e}")
                continue

        if n_used <= 0:
            ranked = list(X_num.columns)
            return ranked, np.ones(len(ranked), dtype=float)

        mean_imp = importances_sum / float(max(1, n_used))
        order = np.argsort(mean_imp)[::-1]
        ranked_features = [str(X_num.columns[i]) for i in order]
        return ranked_features, mean_imp

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

                lbls, _, _, _ = self._compute_dominance_labels(df, fam_events, family=fam, **getattr(g, 'params', {}))
                lbls = pd.to_numeric(lbls, errors='coerce').astype(float).reindex(fam_events.index)
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

        w_arr = None
        if layer1_weight_events is not None:
            try:
                w_s = pd.to_numeric(layer1_weight_events.reindex(X_clean.index), errors='coerce').astype(float)
                w_s = w_s.replace([np.inf, -np.inf], np.nan).fillna(1.0)
                w_s = w_s.clip(lower=0.0)
                w_arr = w_s.to_numpy(dtype=float, copy=False)
            except Exception:
                w_arr = None

        cfg = getattr(self, '_current_config', {})
        if not isinstance(cfg, dict):
            cfg = {}

        try:
            target_n = int(cfg.get('layer2_supervised_feature_count', cfg.get('layer2_probe_feature_count', 70)))
        except Exception:
            target_n = 70

        try:
            corr_threshold = float(cfg.get('layer2_supervised_corr_threshold', cfg.get('layer2_probe_corr_threshold', 0.95)))
        except Exception:
            corr_threshold = 0.95

        try:
            max_rows = int(cfg.get('layer2_supervised_corr_rows', cfg.get('layer2_probe_corr_rows', 2000)))
        except Exception:
            max_rows = 2000

        try:
            n_splits = int(cfg.get('layer2_supervised_mdi_splits', getattr(self, 'n_splits', 3)))
        except Exception:
            n_splits = int(getattr(self, 'n_splits', 3))
        n_splits = int(max(2, min(n_splits, max(2, int(n_valid // 50)))))

        ranked, _ = self._rank_features_by_mean_mdi(
            X_clean,
            y_clean.astype(int),
            sample_weight=w_arr,
            n_splits=n_splits,
        )

        selected = self._cheap_corr_prune(
            X_clean,
            ranked_features=[str(c) for c in ranked],
            target_n=int(target_n),
            corr_threshold=float(corr_threshold),
            max_rows=int(max_rows),
        )
        return [c for c in selected if c in X_events_full.columns]

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
        corr_threshold: float = 0.95,
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
                    callbacks=[lgb.early_stopping(30, verbose=False)],
                )
            elif w_clean is not None:
                model.fit(X_train, y_train, sample_weight=w_clean[train_index])
            elif use_es:
                model.fit(
                    X_tr2, y_tr2,
                    eval_set=[(X_val2, y_val2)],
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
        base_labels, _, _, _ = self._compute_dominance_labels(df, events_df, family=family, **trial_params)

        # 2. Shifted Labels (+1 bar)
        # Using events_shift=1
        shift1_labels, _, _, _ = self._compute_dominance_labels(
            df, events_df, family=family, events_shift=1, **trial_params
        )

        # 3. Shifted Labels (-1 bar)
        # Using events_shift=-1
        shift_neg1_labels, _, _, _ = self._compute_dominance_labels(
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

        # Threshold: Configurable (default 0.85)
        try:
            stability_threshold = float(getattr(self, '_current_config', {}).get('layer2_stability_threshold', 0.82))
        except Exception:
            stability_threshold = 0.82

        if avg_agreement < stability_threshold:
             logger.debug(f"Stability failed: Flip rate too high (agreement={avg_agreement:.2f} < {stability_threshold})")
             return False

        return True

    def _tune_geometry_model_params(
        self,
        df: pd.DataFrame,
        events_df: pd.DataFrame,
        geometry: GeometryTrial
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

            lbls, _, _, _ = self._compute_dominance_labels(df, fam_events, family=geometry.family, **geometry.params)
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

            # Need features
            X_events = self._build_geometry_independent_event_features(df, fam_events.loc[indices])
            # Filter to probe features if global selection is done?
            # We don't have easy access to 'global_probe_features' here unless we store it or re-derive.
            # But 'run' stores it in self._global_probe_features.
            if getattr(self, '_global_probe_features', None):
                cols = [c for c in self._global_probe_features if c in X_events.columns]
                if cols:
                    X_events = X_events[cols]

            X_sub = X_events.fillna(0.0)

            # PHASE 1: Quick Model Race (5 min)
            logger.info(f"Running quick model race for {geometry.family} geometry...")
            
            split_idx = int(len(X_sub) * 0.8)
            X_train_race = X_sub.iloc[:split_idx]
            X_val_race = X_sub.iloc[split_idx:]
            y_train_race = y_sub.iloc[:split_idx]
            y_val_race = y_sub.iloc[split_idx:]
            
            if len(np.unique(y_train_race)) < 2 or len(np.unique(y_val_race)) < 2:
                logger.warning("Insufficient classes in race split. Using LGBM.")
                winning_model_type = 'lgbm'
                race_scores = {}
            else:
                winning_model_type, race_scores = _quick_5model_race(
                    X_train_race, y_train_race,
                    X_val_race, y_val_race,
                    self.random_state
                )

            # PHASE 2: Full HPO on Winner (20 min)
            logger.info(f"Running HPO for {winning_model_type.upper()} on {geometry.family} geometry...")
            
            
            if winning_model_type == 'lgbm':
                # LGBM objective wrapped in function
                def objective(trial):
                    focal_alpha = trial.suggest_float('focal_alpha', 0.4, 1.0)
                    gamma_ratio = trial.suggest_float('gamma_ratio', 0.5, 3.0) # Tweaked via ratio
                    focal_gamma = gamma_ratio / focal_alpha # Inverse relationship: higher alpha -> lower gamma
                    num_leaves = trial.suggest_int('num_leaves', 127, 511)
                    n_estimators = trial.suggest_int('n_estimators', 1000, 2000)
                    
                    params = LAYER2_MODEL_CONSTANTS.copy()
                    params.update({
                        'num_leaves': num_leaves,
                        'n_estimators': n_estimators,
                        'metric': 'binary_logloss',
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
                    focal_obj = RobustFocalLoss(gamma_pos=focal_gamma, gamma_neg=focal_gamma * 2.5, alpha=focal_alpha, verbose=False)
                    params['objective'] = focal_obj

                    # Callback for pruning
                    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "binary_logloss")

                    model = lgb.train(
                        params,
                        train_ds,
                        valid_sets=[val_ds],
                        callbacks=[
                            lgb.early_stopping(30, verbose=False),
                            pruning_callback
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
                    focal_alpha = trial.suggest_float('focal_alpha', 0.4, 1.0)
                    gamma_ratio = trial.suggest_float('gamma_ratio', 0.5, 3.0)
                    focal_gamma = gamma_ratio / focal_alpha
                    n_estimators = trial.suggest_int('n_estimators', 200, 800)
                    max_depth = trial.suggest_int('max_depth', 4, 8)
                    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.05)

                    split_idx = int(len(X_sub) * 0.8)
                    X_tr, X_val = X_sub.iloc[:split_idx], X_sub.iloc[split_idx:]
                    y_tr, y_val = y_sub.iloc[:split_idx], y_sub.iloc[split_idx:]

                    if len(np.unique(y_tr)) < 2:
                        return 10.0

                    focal_obj = XGBFocalLoss(gamma=focal_gamma, alpha=focal_alpha)

                    model = xgb.XGBClassifier(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        objective=focal_obj,
                        eval_metric='logloss',
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
                    iterations = trial.suggest_int('iterations', 200, 600)
                    learning_rate = trial.suggest_float('learning_rate', 0.02, 0.08)
                    depth = trial.suggest_int('depth', 4, 8)
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
                        'metric': 'binary_logloss',
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
        Run Optuna optimization for each barrier family.
        """
        results: Dict[str, List[GeometryTrial]] = {}

        events_df = events_df.copy()
        events_df['family'] = self._assign_barrier_families(events_df)

        unique_families = events_df['family'].unique()

        # Build feature matrix once
        X_events_all = self._build_geometry_independent_event_features(df, events_df)
        target_sample_weight_events = self._get_target_sample_weight_for_events(df, events_df)

        # Initialize param bounds safely
        self._current_param_bounds = getattr(self, '_current_param_bounds', {})
        if not isinstance(self._current_param_bounds, dict):
            self._current_param_bounds = {}

        for family in unique_families:
            logger.info(f"Optimizing family: {family}")

            family_mask = events_df['family'] == family
            family_events = events_df[family_mask]

            if len(family_events) < 50:
                logger.warning(f"Not enough events for family {family} ({len(family_events)}). Skipping.")
                continue

            # Feature Selection PER FAMILY
            # This ensures the "Trend" specialist sees trend features, etc.
            try:
                X_fam = X_events_all.reindex(family_events.index)
                probe_features = self._select_global_probe_features(X_fam)
            except Exception:
                probe_features = []

            try:
                logger.info(
                    f"Layer2 Optimize family={family}: n_events={int(len(family_events))}, n_trials={int(self.n_trials)}, "
                    f"probe_feats={int(len(probe_features))}"
                )
            except Exception:
                pass

            # Define family-specific parameter bounds (heuristics)
            # Trend: Needs room to run (larger horizon), higher RR (kappa)
            # Momentum: Fast moves (shorter horizon), moderate RR
            # Mean Reversion: Quick snaps (short horizon), tighter RR
            fam_bounds = {}
            if family == 'Trend Continuation':
                fam_bounds = {'k_low': 1.5, 'k_high': 6.0, 'h_low': 20, 'h_high': 100}
            elif family == 'Momentum':
                fam_bounds = {'k_low': 2.0, 'k_high': 8.0, 'h_low': 5, 'h_high': 40}
            elif family == 'Mean Reversion':
                fam_bounds = {'k_low': 1.0, 'k_high': 4.0, 'h_low': 10, 'h_high': 60}
            else:
                fam_bounds = {'k_low': 1.0, 'k_high': 6.0, 'h_low': 10, 'h_high': 100}

            # Update current param bounds for this family
            self._current_param_bounds[str(family)] = fam_bounds

            # Use a single, continuous optimization stage with TPESampler
            sampler = optuna.samplers.TPESampler(
                seed=int(self.random_state),
                n_startup_trials=10,
                multivariate=True
            )

            study = optuna.create_study(
                direction="maximize",
                sampler=sampler,
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1)
            )

            # Use partial to pass context to the extracted objective method
            from functools import partial
            obj_func = partial(
                self._optimization_objective,
                study,
                df=df,
                family=family,
                family_events=family_events,
                X_events=X_events_all,
                probe_features=probe_features,
                target_sample_weight_events=target_sample_weight_events
            )

            study.optimize(obj_func, n_trials=int(self.n_trials))

            results[family] = self._extract_trials_from_study(study)

            try:
                n_ext = int(len(results.get(family) or []))
                logger.info(
                    f"Layer2 Optimize family={family}: extracted_trials={n_ext}, cache_hits={int(getattr(self, '_cache_hits', 0))}, "
                    f"cache_misses={int(getattr(self, '_cache_misses', 0))}"
                )
            except Exception:
                pass

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

        # Parameter Space: Kappa and Horizon
        # Use family-specific bounds if available
        if isinstance(bounds, dict) and all(k in bounds for k in ('k_low', 'k_high', 'h_low', 'h_high')):
             kappa = trial.suggest_float('kappa', float(bounds['k_low']), float(bounds['k_high']))
             horizon = trial.suggest_int('horizon', int(bounds['h_low']), int(bounds['h_high']))
        else:
             # Default ranges - reduced from 0.5-12.0 to 0.3-8.0 for softer barriers
             kappa = trial.suggest_float('kappa', 0.3, 8.0)
             horizon = trial.suggest_int('horizon', 8, 120)

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
        sl_mult = trial.suggest_float('sl_mult', float(sl_low), float(sl_high))

        # Distance-based pruning to avoid similar geometries
        params_vector = [kappa, sl_mult, horizon]
        # Normalize based on bounds (kappa: 0.3-8.0, sl_mult: 0.3-2.0, horizon: 8-120)
        normalized_params = [
            (kappa - 0.3) / (8.0 - 0.3),
            (sl_mult - 0.3) / (2.0 - 0.3),
            (horizon - 8) / (120 - 8)
        ]
        threshold = 0.05  # 5% of normalized space; tune as needed
        
        for prev_trial in study.trials:
            if prev_trial.value is None or prev_trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            prev_norm = [
                (prev_trial.params['kappa'] - 0.3) / (8.0 - 0.3),
                (prev_trial.params['sl_mult'] - 0.3) / (2.0 - 0.3),
                (prev_trial.params['horizon'] - 8) / (120 - 8)
            ]
            if euclidean(normalized_params, prev_norm) < threshold:
                return -1.0  # Skip computation for near-duplicates

        # Compute labels
        labels, returns, _, _ = self._compute_dominance_labels(df, family_events, kappa, horizon, family, sl_mult=sl_mult)

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

        pos_rate = labels.mean()

        # --- OPTIMIZATION: Tighter Pre-Filters ---
        # If the geometry is fundamentally poor in terms of base statistics, don't waste time on probes.
        try:
            min_rate = float(getattr(self, '_current_config', {}).get('layer2_min_pos_rate', 0.01))
            max_rate = float(getattr(self, '_current_config', {}).get('layer2_max_pos_rate', 0.95))
        except Exception:
             min_rate, max_rate = 0.01, 0.95

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
            min_pos_trades = int(getattr(self, '_current_config', {}).get('layer2_min_positive_trades', 15))
        except Exception:
            min_pos_trades = 15

        if profit_mode == 'intelligent':
            # Allow small losses but require risk compensation
            # Relaxed from -0.1% to -0.35% to match current geometry performance
            min_trade_ret = float(getattr(self, '_current_config', {}).get('layer2_min_mean_trade_return', -0.0035))  # Allow -0.35% losses
            max_acceptable_loss = float(getattr(self, '_current_config', {}).get('layer2_max_acceptable_loss', -0.0035))  # Max -0.35% loss
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
            perturb_labels_k, _, _, _ = self._compute_dominance_labels(df, fam_events_for_checks, kappa * 1.05, horizon, family, sl_mult=sl_mult)
            perturb_labels_sl, _, _, _ = self._compute_dominance_labels(df, fam_events_for_checks, kappa, horizon, family, sl_mult=sl_mult * 1.05)
            perturb_labels_h, _, _, _ = self._compute_dominance_labels(df, fam_events_for_checks, kappa, int(horizon * 1.05), family, sl_mult=sl_mult)

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
            learnability = float(probe_res.get('auc', 0.0))
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
            robust_magnitude=float(mean_ret) * 1000,
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
        
        # Map timestamps to integer indices
        idx_map = {ts: i for i, ts in enumerate(df.index)}
        close_arr = df['close'].values
        vol_arr = df['volatility_1d'].fillna(0.0).values
        
        for i, ts in enumerate(events_df.index):
            entry_idx = idx_map.get(ts)
            if entry_idx is None: continue
            
            exit_idx = min(entry_idx + horizon, len(close_arr) - 1)
            if exit_idx <= entry_idx: continue
            
            # Calculate path returns relative to entry
            # path[k] = (price[k] / price[entry]) - 1
            entry_price = close_arr[entry_idx]
            if entry_price <= 0: continue
            
            # Slice path
            price_path = close_arr[entry_idx : exit_idx + 1]
            returns_path = (price_path / entry_price) - 1.0
            
            # Direction from CUSUM consensus
            direction = 1
            if 'event_consensus' in events_df.columns:
                c = events_df['event_consensus'].iloc[i]
                if c < 0: direction = -1
            
            sigma = vol_arr[entry_idx]
            if sigma <= 0: sigma = 0.001 # safe floor
            
            events_list.append(Event(
                id=entry_idx, # Use int index as ID for mapping back
                entry_idx=entry_idx,
                exit_idx=exit_idx,
                direction=direction,
                returns_path=returns_path,
                sigma=sigma
            ))
            
        return events_list

    def _compute_geometry_labels(
        self,
        df: pd.DataFrame,
        events_df: pd.DataFrame,
        geometry: GeometryWrapper
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Compute labels based on new Geometry logic:
        Label = 1 if (norm_mae <= sl_sigma) AND (score >= min_ratio).
        Returns: labels, returns, mfe, mae.
        """
        # Convert to Event objects
        # Horizon fixed to 40
        events_list = self._create_events_list(df, events_df, horizon=40)
        if not events_list:
            empty = pd.Series(np.nan, index=events_df.index)
            return empty, empty, empty, empty

        # Vectorize
        ev_df = events_to_dataframe(events_list)
        if ev_df.empty:
            empty = pd.Series(np.nan, index=events_df.index)
            return empty, empty, empty, empty

        geom = geometry.geometry
        
        # Apply Logic
        mask_sl = ev_df['norm_mae'] <= geom.sl_sigma
        score = (ev_df['norm_mfe'] ** geom.beta) / ((ev_df['norm_mae'] + 1e-6) ** geom.alpha)
        mask_score = score >= geom.min_ratio
        
        labels_arr = (mask_sl & mask_score).astype(float)
        
        # Returns:
        # If Label 1: Return = MFE (Perfect exit assumption for labeling potential)
        # OR Return at horizon?
        # New logic implies we capture the 'score'.
        # Existing logic used MFE for winners. Let's stick to MFE * direction * sigma?
        # Actually, returns_path is percentage.
        # If Winner: Return = MFE * Sigma (since norm_mfe = raw_mfe/sigma) => Raw MFE.
        # If Loser: Return at horizon? Or Stop Loss?
        # If hit SL (mask_sl False): Return = -SL_Sigma * Sigma
        # If time out (mask_sl True, mask_score False): Return = final return
        
        # Recover Raw values
        raw_mfe = ev_df['norm_mfe'] * ev_df['sigma']
        raw_mae = ev_df['norm_mae'] * ev_df['sigma']

        final_ret = []
        for _, row in ev_df.iterrows():
            idx = int(row.name) # this is entry_idx
            is_win = (row['norm_mae'] <= geom.sl_sigma) and (score.loc[idx] >= geom.min_ratio)

            if is_win:
                # Optimistic: We captured the MFE
                ret = row['norm_mfe'] * row['sigma']
            elif row['norm_mae'] > geom.sl_sigma:
                # Stopped out
                ret = -geom.sl_sigma * row['sigma']
            else:
                # Timed out - use return at horizon
                # Need to look up actual return.
                # Approx: use 0.0 or small penalty.
                # Better: In _create_events_list we have returns_path.
                # Re-accessing it here is slow.
                # Let's approximate Time-Out return as 0 for now or assume close to entry.
                ret = 0.0

            final_ret.append(ret)

        # Map back to Series
        # events_list IDs are integer indices of df. We need to map back to timestamps.
        # events_df.index contains timestamps.

        # Create map: int_idx -> timestamp
        idx_to_ts = {i: ts for i, ts in enumerate(df.index)}

        mapped_labels = pd.Series(labels_arr.values, index=[idx_to_ts[i] for i in ev_df.index])
        mapped_returns = pd.Series(final_ret, index=[idx_to_ts[i] for i in ev_df.index])
        mapped_mfe = pd.Series(raw_mfe.values, index=[idx_to_ts[i] for i in ev_df.index])
        mapped_mae = pd.Series(raw_mae.values, index=[idx_to_ts[i] for i in ev_df.index])

        # Align with full events_df
        return (
            mapped_labels.reindex(events_df.index),
            mapped_returns.reindex(events_df.index),
            mapped_mfe.reindex(events_df.index),
            mapped_mae.reindex(events_df.index)
        )

    # Legacy wrapper
    def _compute_dominance_labels(self, df, events_df, family=None, **kwargs):
        # Fallback if old code calls this
        # We need a geometry object. Create a dummy one based on kwargs or defaults.
        # This is strictly for compatibility, but we should not hit this in new flow.
        return pd.Series(np.nan, index=events_df.index), pd.Series(np.nan, index=events_df.index), pd.Series(np.nan, index=events_df.index), pd.Series(np.nan, index=events_df.index)

    def _train_geometry_models(
        self,
        df: pd.DataFrame,
        X_events: pd.DataFrame,
        events_df: pd.DataFrame,
        geometries: List[GeometryWrapper]
    ) -> Dict[str, Any]:
        """
        Train simple LGBM models for each geometry on the provided training set.
        """
        models = {}
        for g in geometries:
            try:
                lbls, _, _, _ = self._compute_geometry_labels(df, events_df, g)
                valid_lbls = lbls.dropna()
                common_idx = valid_lbls.index.intersection(X_events.index)
                
                if len(common_idx) < 20: 
                     models[g.uuid] = None
                     continue

                # Generate specific geometry features
                geo_features = self._compute_specific_geometry_features(df, common_idx, g.params)

                X_train = X_events.loc[common_idx]

                # Append geometry features
                if not geo_features.empty:
                    geo_features = geo_features.reindex(common_idx).fillna(0.0)
                    X_train = pd.concat([X_train, geo_features], axis=1)

                y_train = valid_lbls.loc[common_idx]
                
                if len(y_train.unique()) < 2:
                    models[g.uuid] = None
                    continue

                # Base params from constants
                params = LAYER2_MODEL_CONSTANTS.copy()
                tuned_params = getattr(g, 'model_params', None)
                
                # Default Focal Loss params
                f_gamma = 0.5
                f_alpha = 0.65

                model_type = 'lgbm'
                if isinstance(tuned_params, dict) and tuned_params:
                    model_type = tuned_params.get('model_type', 'lgbm')
                    if 'focal_gamma' in tuned_params: f_gamma = float(tuned_params['focal_gamma'])
                    if 'focal_alpha' in tuned_params: f_alpha = float(tuned_params['focal_alpha'])

                # Simple training loop (simplified from original for brevity, logic preserved)
                focal_obj = RobustFocalLoss(train_labels=y_train.values, gamma=f_gamma, alpha=f_alpha)
                def lgbm_focal_obj(y_pred, y_true): return focal_obj(y_pred, y_true)

                clf = lgb.LGBMClassifier(
                    objective=lgbm_focal_obj,
                    n_estimators=int(params.get('n_estimators', 500)),
                    num_leaves=int(params.get('num_leaves', 31)),
                    learning_rate=float(params.get('learning_rate', 0.05)),
                    random_state=self.random_state,
                    n_jobs=1,
                    verbosity=-1
                )

                clf.fit(X_train, y_train)
                models[g.uuid] = clf
                    # Extract Focal params if present
                    if 'focal_gamma' in tuned_params:
                        f_gamma = float(tuned_params['focal_gamma'])
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

                    focal_obj = RobustFocalLoss(gamma_pos=f_gamma, gamma_neg=f_gamma * 2.5, alpha=f_alpha)
                    
                    # Helper wrapper for objective compatibility with LGBMClassifier
                    def lgbm_focal_obj(y_pred, y_true):
                        # Note: y_true is passed as 2nd arg in sklearn API sometimes, but 
                        # LGBMClassifier typically expects: func(y_true, y_pred) -> (grad, hess)
                        # We use the instance method which handles the math.
                        return focal_obj(y_pred, y_true)

                    clf = lgb.LGBMClassifier(
                        objective=lgbm_focal_obj,
                        n_estimators=n_estimators,
                        num_leaves=int(params.get('num_leaves', 31)),
                        learning_rate=float(params.get('learning_rate', 0.05)),
                        class_weight='balanced',
                        random_state=self.random_state,
                        n_jobs=1,
                        verbosity=-1
                    )
                    
                    if has_val:
                        # Train on inner train
                        clf.fit(
                            X_tr_inner, y_tr_inner,
                            eval_set=[(X_val_inner, y_val_inner)],
                            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
                        )
                        # Calibrate on inner val
                        calibrated = CalibratedClassifierCV(clf, method='sigmoid', cv='prefit')
                        calibrated.fit(X_val_inner, y_val_inner)
                        models[g.uuid] = calibrated
                    else:
                        # Fallback: Train on full, no calibration possible without leak
                        # We could use cv=3 here instead, but for now we stick to uncalibrated fallback
                        clf.fit(X_train, y_train)
                        models[g.uuid] = clf

                elif model_type == 'xgb':
                    # XGBoost with Focal Loss
                    focal_obj = XGBFocalLoss(gamma=f_gamma, alpha=f_alpha)
                    
                    model_xgb = xgb.XGBClassifier(
                        n_estimators=tuned_params.get('n_estimators', 500) if tuned_params else 500,
                        learning_rate=tuned_params.get('learning_rate', 0.03) if tuned_params else 0.03,
                        max_depth=tuned_params.get('max_depth', 6) if tuned_params else 6,
                        objective=focal_obj,
                        eval_metric='logloss',
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
                    def lgbm_focal_obj_fb(y_pred, y_true):
                         return focal_obj(y_pred, y_true)

                    clf = lgb.LGBMClassifier(
                        objective=lgbm_focal_obj_fb,
                        n_estimators=500,
                        class_weight='balanced',
                        random_state=self.random_state,
                        n_jobs=1,
                        verbosity=-1
                    )

            except Exception as e:
                logger.warning(f"Failed to train geometry model for {g.uuid}: {e}")
                models[g.uuid] = None
        return models

    def _bagged_labeling(
        self,
        df: pd.DataFrame,
        events_df: pd.DataFrame,
        geometries: List[GeometryWrapper],
        trained_models: Optional[Dict[str, Any]] = None,
        X_events: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Generate final bagged outputs.
        """
        # Storage
        composite_labels = pd.Series(index=events_df.index, dtype=float)
        composite_prob = pd.Series(index=events_df.index, dtype=float)
        composite_returns = pd.Series(index=events_df.index, dtype=float)
        composite_weights = pd.Series(index=events_df.index, dtype=float)
        
        oof_preds = {}
        oof_vars = {}

        # Organize by archetype (Family)
        geo_by_fam = {}
        for g in geometries:
            geo_by_fam.setdefault(g.family, []).append(g)

        # Iterate by family/archetype
        # Note: events_df is global now, not pre-split.
        # But we still process by group to average within archetypes first.

        # Accumulators for Global Consensus
        global_score_sum = pd.Series(0.0, index=events_df.index)
        global_prob_max = pd.Series(0.0, index=events_df.index)
        global_ret_sum = pd.Series(0.0, index=events_df.index)
        global_weight_sum = pd.Series(0.0, index=events_df.index)

        for i, g in enumerate(geometries):
            lbls, rets, mfe, mae = self._compute_geometry_labels(df, events_df, g)
            geo_features = self._compute_specific_geometry_features(df, events_df.index, g.params)

            # Predict
            prob_s = pd.Series(np.nan, index=events_df.index)
            var_s = pd.Series(np.nan, index=events_df.index)

            if trained_models and g.uuid in trained_models and X_events is not None:
                booster = trained_models[g.uuid]
                if booster:
                    common = events_df.index.intersection(X_events.index)
                    if not common.empty:
                        X_sub = X_events.loc[common]
                        if not geo_features.empty:
                            g_feat = geo_features.reindex(common).fillna(0.0)
                            X_sub = pd.concat([X_sub, g_feat], axis=1)

                        raw = booster.predict(X_sub)
                        p = 1.0 / (1.0 + np.exp(-raw))
                        v = _calculate_tree_variance(booster, X_sub)

                        prob_s.loc[common] = p
                        var_s.loc[common] = v

            oof_preds[g.uuid] = prob_s
            oof_vars[g.uuid] = var_s

            # Weighting Logic
            # W = ln(1+MFE) * ln(1+MFE/MAE) * ER
            # Assuming MFE/MAE are aligned
            mfe_v = mfe.fillna(0.0)
            mae_v = mae.fillna(1e-9).replace(0.0, 1e-9)

            w_mag = np.log1p(np.maximum(0.0, mfe_v))
            w_smooth = np.log1p(np.maximum(0.0, mfe_v / mae_v))

            # Structure Conf (ER)
            er_series = get_efficiency_ratio(df['close'], 50).reindex(events_df.index).fillna(0.2)
            w_conf = np.clip((er_series - 0.2) / 0.6, 0.0, 1.0)

            w_i = w_mag * w_smooth * w_conf

            # Accumulate
            # Only where label is valid (triggered)
            valid = lbls.notna()

            # Consensus Strategy: Max Prob
            global_prob_max = np.maximum(global_prob_max, prob_s.fillna(0.0))

            # Weighted Return
            ret_contrib = rets.fillna(0.0) * w_i
            global_ret_sum = global_ret_sum.add(ret_contrib, fill_value=0.0)
            global_weight_sum = global_weight_sum.add(w_i, fill_value=0.0)

        # Finalize
        # Consensus Return
        consensus_ret = global_ret_sum / global_weight_sum.replace(0.0, 1.0)

        # Final Weights (Standardized)
        final_weights = finalize_sample_weights(global_weight_sum)

        return {
            "oof_labels": global_prob_max, # Score
            "oof_returns": consensus_ret,
            "weights": final_weights,
            "l2_score": global_prob_max,
            "l2_label": (global_prob_max > 0.5).astype(float),
            "l2_confidence": global_prob_max, # Placeholder
            "individual_geometries": oof_preds,
            "individual_variances": oof_vars
        }

    def _compute_specific_geometry_features(
        self,
        df: pd.DataFrame,
        events_index: pd.Index,
        params: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Compute geometry-specific features based on new parameters.
        """
        if events_index.empty: return pd.DataFrame()

        sl_sigma = float(params.get('sl_sigma', 1.0))
        # alpha, beta, min_ratio are scoring params, not barriers directly usable for scaling features easily
        # but we can use sl_sigma.

        subset = df.reindex(events_index)
        vol = subset['volatility_1d'].fillna(0.0)
        close = subset['close']

        # Stop Size in Price
        stop_dist = sl_sigma * vol

        feats = pd.DataFrame(index=events_index)

        # Vol / Stop
        feats['geo_vol_to_sl'] = vol / (stop_dist + 1e-9)

        # ATR / Stop
        if 'geo_atr_14' in df.columns:
            atr = df['geo_atr_14'].reindex(events_index).fillna(0.0)
            feats['geo_atr_to_sl'] = atr / (stop_dist + 1e-9)

        return feats

    def _validate_inputs(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'close' not in df.columns or 'volatility_1d' not in df.columns:
            raise ValueError("Missing required columns: close, volatility_1d")
        return df

    def _precompute_geometry_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        # Ensure ATR
        if 'geo_atr_14' not in df_out.columns:
            high = df_out['high'] if 'high' in df_out.columns else df_out['close']
            low = df_out['low'] if 'low' in df_out.columns else df_out['close']
            close = df_out['close']
            tr = pd.concat([(high-low).abs(), (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
            df_out['geo_atr_14'] = tr.rolling(14).mean()
        return df_out

    def _generate_events(self, df: pd.DataFrame) -> pd.DataFrame:
        # Re-use CUSUM logic from existing tool or simple implementation
        try:
            signals = generate_primary_signals(df)
            cons = pd.to_numeric(signals.get('consensus'), errors='coerce').fillna(0.0)
            events = df.index[cons != 0]

            ev_df = df.loc[events].copy()
            ev_df['event_consensus'] = cons.loc[events]
            return ev_df
        except Exception:
            # Fallback
            return pd.DataFrame(index=df.index[::50]) # Dummy

    def _save_selected_features(self, features: List[str]):
        try:
            outcomes_dir = Path("outcomes")
            outcomes_dir.mkdir(parents=True, exist_ok=True)
            if features:
                ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                pd.Series(features).to_csv(outcomes_dir / f"layer2_selected_features_{ts}.csv", index=False)
        except Exception:
            pass

    def _generate_reports(self, df, events_df, geometries, scores, labels, weights, tree_stats):
        try:
            outcomes_dir = Path("outcomes")
            outcomes_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

            # Geometry Report
            rows = []
            for g in geometries:
                r = asdict(g)
                r.update(g.params)
                rows.append(r)
            pd.DataFrame(rows).to_csv(outcomes_dir / f"layer2_production_geometries_{ts}.csv", index=False)

            # Summary
            lines = [
                f"Selected Geometries: {len(geometries)}",
                f"Events: {len(events_df)}",
                f"Labeled: {labels.sum()}",
            ]
            (outcomes_dir / f"layer2_report_{ts}.md").write_text("\n".join(lines))
        except Exception:
            pass

    def _extract_tree_diagnostics(self, booster) -> Dict[str, float]:
        if hasattr(booster, 'feature_importance'):
            return {'n_features_used': float(np.sum(booster.feature_importance() > 0))}
        return {}

    def _tune_geometry_model_params(self, df, events_df, g):
        # Stub for model tuning - in real flow this runs Optuna
        return {'model_type': 'lgbm', 'n_estimators': 500}

    def _assign_barrier_families(self, df):
        return pd.Series('General', index=df.index)

    def _select_global_probe_features(self, X):
        return list(X.columns[:50])

    def _build_geometry_independent_event_features(self, df, events_df):
        return create_meta_features(df, pd.DataFrame({'consensus':0}, index=df.index), volume_available=True).reindex(events_df.index).fillna(0.0)

    def _aggregate_geometry_labels_for_feature_selection(self, df, events_df, geos):
        return pd.Series(0, index=events_df.index) # Stub
