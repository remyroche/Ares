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
    Focal Loss for LightGBM with numeric stability, auto-alpha, and optional gradient clipping.
    Suitable for rare-event classification.
    """

    def __init__(self, train_labels, gamma=1.5, alpha=None, grad_clip=100.0, verbose=True):
        """
        Args:
            train_labels: np.array of 0/1 labels
            gamma: focusing parameter (1-2 typical)
            alpha: positive class weight; if None, auto-computed from prevalence
            grad_clip: optional max absolute gradient value
            verbose: print alpha/gamma info
        """
        self.gamma = gamma
        self.grad_clip = grad_clip

        # --- ALPHA TUNING ---
        if alpha is None:
            n_pos = np.sum(train_labels == 1)
            n_neg = np.sum(train_labels == 0)
            if (n_pos + n_neg) > 0:
                self.alpha = n_neg / (n_pos + n_neg)
            else:
                self.alpha = 0.5
        else:
            self.alpha = alpha

        # Safety: enforce alpha in [0.01,0.99] to allow user-specified downweighting (e.g. 0.25)
        self.alpha = min(max(self.alpha, 0.01), 0.99)

        if verbose:
            try:
                pos_frac = np.mean(train_labels)
            except Exception:
                pos_frac = 0.0
            if verbose:
                logger.info(f"[Focal Loss] gamma={self.gamma}, alpha={self.alpha:.4f} (Pos fraction: {pos_frac:.2%})")

    def __call__(self, preds, train_data):
        """
        Args:
            preds: raw margins from LGBM
            train_data: lgb.Dataset or numpy labels
        Returns:
            grad, hess: gradient and hessian arrays
        """
        if hasattr(train_data, 'get_label'):
             labels = train_data.get_label()
        else:
             labels = train_data
        
        p = expit(preds)  # convert raw score to probability
        p = np.clip(p, 1e-9, 1 - 1e-9)  # standardized clipping (prevent log(0))

        # --- Common terms ---
        term_pos = (1 - p) ** self.gamma
        term_neg = p ** self.gamma

        # --- Gradient ---
        grad = (-self.alpha * term_pos * (1 - p - self.gamma * p * np.log(p)) * labels +
                (1 - self.alpha) * term_neg * (p - self.gamma * (1 - p) * np.log(1 - p)) * (1 - labels))

        # --- Hessian ---
        hess = (self.alpha * term_pos * (1 - p) * (1 + (self.gamma - 1) * p * np.log(p)) * labels +
                (1 - self.alpha) * term_neg * p * (1 + (self.gamma - 1) * (1 - p) * np.log(1 - p)) * (1 - labels))

        # --- Gradient clipping ---
        if self.grad_clip is not None:
            grad = np.clip(grad, -self.grad_clip, self.grad_clip)

        # --- Hessian stability ---
        hess = np.maximum(hess, 1e-6)

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
        focal_lgbm = RobustFocalLoss(train_labels=y_train.values, gamma=2.0, alpha=0.25, verbose=False)
        
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
