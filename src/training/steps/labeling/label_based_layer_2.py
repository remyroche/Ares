"""
Label-Based Layer 2: Regime-Conditional Geometry Optimization & Meta-Labeling
-----------------------------------------------------------------------------
This module implements the Layer 2 architecture for meta-labeling, focusing on
geometry optimization and ML learnability probes.

It performs:
1. Event generation based on volatility-scaled returns.
2. Independent optimization of barrier geometries (Kappa/Horizon) using Optuna (Unified).
3. MFE/MAE Dominance Labeling: Label = 1 if MFE > Kappa * MAE.
4. Stability checks (Time-Flip) and Learnability probes.
5. Bagged output generation.
6. Enhanced LGBM training with Robust Focal Loss and Tree Variance calculation.
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import spearmanr
from scipy.special import expit
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from joblib import Parallel, delayed
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
import logging
import copy
import warnings

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

from src.training.steps.labeling.label_geometry_selection import (
    select_geometries,
    Event,
    Geometry,
    MIN_SL_PCT,
    MIN_TP_SL_RATIO
)

# Configure logging
logger = logging.getLogger(__name__)

# Constants for Layer 2 Model Training (defaults/fixed) - Less Regularized
LAYER2_MODEL_CONSTANTS = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'max_depth': -1,
    'learning_rate': 0.03,
    'lambda_l1': 0.01,
    'lambda_l2': 0.05,
    'num_leaves': 31,
    'min_data_in_leaf': 5,
    'min_sum_hessian_in_leaf': 1e-3,
    'feature_fraction': 0.95,
    'bagging_fraction': 0.95,
    'bagging_freq': 1,
    'verbose': -1,
    'random_state': 42,
    'n_jobs': 1,
    'is_unbalance': False,
    'scale_pos_weight': 1,
    'min_gain_to_split': 0.001,
    'min_child_weight': 0.0001,
}

class RobustFocalLoss:
    """
    Production-grade Focal Loss for LightGBM in Financial ML.
    """

    def __init__(
        self,
        gamma_pos=1.0, # gamma_fn
        gamma_neg=2.5, # gamma_fp
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
        if self.alpha is None:
            n_pos = np.sum(labels > 0.5)
            n_total = len(labels)
            if n_total > 0:
                self.alpha = 1.0 - (n_pos / n_total)
            else:
                self.alpha = 0.5

        self.alpha = np.clip(self.alpha, 0.05, 0.95)
        self._is_init = True

    def __call__(self, preds, train_data):
        if hasattr(train_data, 'get_label'):
             labels = train_data.get_label()
        else:
             labels = train_data

        if not self._is_init:
            self._init_alpha(labels)

        y_smooth = labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        p = expit(preds)
        p = np.clip(p, 1e-7, 1 - 1e-7)

        gamma_arr = np.where(labels > 0.5, self.gamma_pos, self.gamma_neg)
        focal_weight = np.where(labels > 0.5, (1 - p), p) ** gamma_arr
        focal_weight = np.minimum(focal_weight, self.w_cap)

        grad_bce = p - y_smooth
        alpha_factor = np.where(labels > 0.5, self.alpha, (1 - self.alpha))
        grad_focal = alpha_factor * focal_weight * grad_bce

        hess_bce = p * (1 - p)
        hess_focal = alpha_factor * focal_weight * hess_bce

        grad = self.mix * grad_focal + (1 - self.mix) * grad_bce
        hess = self.mix * hess_focal + (1 - self.mix) * hess_bce

        if self.grad_clip:
            grad = np.clip(grad, -self.grad_clip, self.grad_clip)

        hess = np.maximum(hess, 1e-6)

        return grad, hess


class XGBFocalLoss:
    """
    Focal Loss for XGBoost (custom objective function).
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
        if self.alpha is None:
            n_pos = np.sum(labels > 0.5)
            n_total = len(labels)
            if n_total > 0:
                self.alpha = 1.0 - (n_pos / n_total)
            else:
                self.alpha = 0.5
        self.alpha = np.clip(self.alpha, 0.05, 0.95)
        self._is_init = True

    def __call__(self, preds, dtrain):
        if hasattr(dtrain, 'get_label'):
            labels = dtrain.get_label()
        else:
            labels = dtrain

        if not self._is_init:
            self._init_alpha(labels)

        y_smooth = labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        p = 1.0 / (1.0 + np.exp(-preds))
        p = np.clip(p, 1e-7, 1 - 1e-7)

        gamma_arr = np.where(labels > 0.5, self.gamma_pos, self.gamma_neg)
        focal_weight = np.where(labels > 0.5, (1 - p), p) ** gamma_arr
        focal_weight = np.minimum(focal_weight, self.w_cap)

        grad_bce = p - y_smooth
        alpha_factor = np.where(labels > 0.5, self.alpha, (1 - self.alpha))
        grad_focal = alpha_factor * focal_weight * grad_bce

        hess_bce = p * (1 - p)
        hess_focal = alpha_factor * focal_weight * hess_bce

        grad = self.mix * grad_focal + (1 - self.mix) * grad_bce
        hess = self.mix * hess_focal + (1 - self.mix) * hess_bce

        if self.grad_clip:
            grad = np.clip(grad, -self.grad_clip, self.grad_clip)

        hess = np.maximum(hess, 1e-6)

        return grad, hess


def _quick_5model_race(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    random_state: int = 42,
) -> Tuple[str, Dict[str, float]]:
    """
    Fast 5-model race to determine best model type for a geometry.
    """
    scores = {}
    
    # --- Model 1: LGBM Standard ---
    try:
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
        
    # --- Model 2: LGBM Linear ---
    try:
        params_lgbm_lin = {
            'n_estimators': 500,
            'learning_rate': 0.05,
            'num_leaves': 63,
            'max_depth': 4,
            'extra_trees': True,
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
        focal_xgb = XGBFocalLoss(gamma_pos=2.0, gamma_neg=5.0, alpha=0.25)
        model_xgb = xgb.XGBClassifier(
            n_estimators=400,
            learning_rate=0.04,
            max_depth=6,
            min_child_weight=10,
            subsample=0.8,
            colsample_bytree=0.8,
            objective=focal_xgb,
            eval_metric=['auc', 'aucpr'],
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
        
    # --- Model 4: XGBoost Linear ---
    try:
        model_xgb_lin = xgb.XGBClassifier(
            booster='gblinear',
            n_estimators=100,
            learning_rate=0.1,
            objective='binary:logistic',
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

    winner = max(scores, key=scores.get)
    logger.info(f"   Model race winner: {winner.upper()}")
    return winner, scores


def _calculate_tree_variance(booster, X) -> np.ndarray:
    """
    Calculate the variance of predictions across all trees in the ensemble.
    """
    if booster is None:
        return np.zeros(X.shape[0])

    try:
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

        n_samples = leaf_indices.shape[0]
        n_trees_pred = leaf_indices.shape[1]
        limit_trees = min(n_trees, n_trees_pred)
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
class GeometryTrial:
    family: str
    params: Dict[str, Any]
    final_score: float
    learnability: float
    robust_magnitude: float
    stability: float
    balance: float
    raw_metrics: Dict[str, float]
    uuid: str
    model_params: Optional[Dict[str, Any]] = None
    selected_features: Optional[List[str]] = field(default=None)

class LabelBasedLayer2:
    """
    Layer 2: Geometry Optimization & Meta-Labeling.
    """

    def __init__(
        self,
        transaction_cost: Optional[float] = None,
        n_trials: int = 60,
        n_splits: int = 3,
        random_state: int = 42,
        verbose: bool = True,
        force_hpo: bool = False
    ):
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

        self.selected_geometries: List[GeometryTrial] = []
        self._labels_cache: Dict[Any, Tuple[pd.Series, pd.Series, pd.Series, pd.Series]] = {}
        self._signals_cache: Dict[Any, pd.DataFrame] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._global_probe_features: List[str] = []
        self._current_param_bounds: Dict[str, Dict[str, Any]] = {}
        self._primary_signals: Optional[pd.DataFrame] = None
        self._rfe_stats = []

        if not self.verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

    def execute(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._current_config = dict(config or {})
        return self.run(df)

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute the Layer 2 pipeline.
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

        df = self._validate_inputs(df)
        df = self._precompute_geometry_base_features(df)
        events_df = self._generate_events(df)

        if not events_df.empty:
            events_df['family'] = 'Unified'
            try:
                X_probe_events = self._build_geometry_independent_event_features(df, events_df)
                self._global_probe_features = self._select_global_probe_features(X_probe_events)
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
             count = len(full_results.get('Unified', []))
             tprint_info(f"Layer2 Full Optimization: extracted_trials={count}")
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
        if production_geometries and len(production_geometries) > 10:
            tprint_info(f"Capping production geometries to 10 (from {len(production_geometries)})")
            production_geometries = production_geometries[:10]

        # Optimize Model Parameters and Features for Production Geometries
        if production_geometries:
            tprint_info(">>> Layer 2: Tuning Model Parameters & Selecting Features for Production Geometries...")

            X_events_full = self._build_geometry_independent_event_features(df, events_df)
            w_l1_prod = self._get_target_sample_weight_for_events(df, events_df)
            vol_prod = df['volatility_1d'].reindex(events_df.index).fillna(0.0)

            for i, g in enumerate(production_geometries):
                tprint_info(f"    Processing geometry {g.uuid} ({i+1}/{len(production_geometries)})...")

                # 1. Parameter Tuning
                best_params = self._tune_geometry_model_params(df, events_df, g)
                if best_params:
                    g.model_params = best_params
                    tprint_info(f"    Found params for {g.uuid}: {best_params}")

                # 2. Per-Geometry Feature Selection
                try:
                    cfg_prod_fs = getattr(self, "_current_config", {})
                    if not isinstance(cfg_prod_fs, dict): cfg_prod_fs = {}
                    enable_prod_fs = bool(cfg_prod_fs.get('layer2_production_supervised_feature_selection_enabled', True))

                    if enable_prod_fs:
                        tprint_info(f"    Selecting features for {g.uuid}...")

                        lbls, _, _, _, _ = self._compute_dominance_labels(df, events_df, **g.params)
                        lbls = pd.to_numeric(lbls, errors='coerce').astype(float).reindex(events_df.index)

                        valid_idx = lbls.dropna().index
                        if len(valid_idx) > 50:
                            y_target = lbls.loc[valid_idx]
                            X_target = X_events_full.reindex(valid_idx).fillna(0.0)
                            w_target = w_l1_prod.reindex(valid_idx) if w_l1_prod is not None else None
                            vol_target = vol_prod.reindex(valid_idx)

                            initial_feat_count = X_target.shape[1]
                            sel_feats = self._select_supervised_features_for_events(
                                X_target, y_target, w_target, volatility_series=vol_target
                            )
                            final_feat_count = len(sel_feats) if sel_feats else 0

                            self._rfe_stats.append({
                                'timestamp': datetime.utcnow().isoformat(),
                                'uuid': g.uuid,
                                'initial_features': initial_feat_count,
                                'final_features': final_feat_count,
                                'retention_pct': (final_feat_count / initial_feat_count * 100) if initial_feat_count else 0,
                                'selected_features_list': str(sel_feats)
                            })

                            if sel_feats:
                                g.selected_features = list(sel_feats)
                                tprint_success(f"    Selected {len(sel_feats)} features for {g.uuid}")
                except Exception as e:
                    tprint_warning(f"    Feature selection failed for {g.uuid}: {e}")

        if not production_geometries:
            tprint_error("Layer2 CRITICAL: Zero production geometries passed all gates!")
            raise ValueError("Layer2 failed: No production geometries passed validation gates.")

        self.selected_geometries = production_geometries
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
        except Exception:
            pass

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

        indices = events_df.index
        oof_scores = pd.Series(np.nan, index=indices)
        oof_labels = pd.Series(np.nan, index=indices)
        oof_confidence = pd.Series(np.nan, index=indices)
        oof_returns = pd.Series(np.nan, index=indices)
        oof_weights = pd.Series(np.nan, index=indices)

        self._all_tree_stats = []

        families = ['Unified']
        max_rank = 10
        oof_geo_preds = {}
        oof_geo_vars = {}
        for fam in families:
            for r in range(max_rank):
                key = f"{fam}_Rank{r}"
                oof_geo_preds[key] = pd.Series(np.nan, index=indices)
                oof_geo_vars[key] = pd.Series(np.nan, index=indices)

        try:
            cfg_oof = getattr(self, "_current_config", {})
            if not isinstance(cfg_oof, dict): cfg_oof = {}
        except Exception:
            cfg_oof = {}

        n_oof_splits = int(cfg_oof.get("layer2_oof_splits", 3))
        n_oof_splits = int(max(2, min(n_oof_splits, int(len(df)))))

        purge_bars = int(cfg_oof.get("layer2_oof_purge_bars", 0))
        if purge_bars <= 0:
            purge_bars = int(cfg_oof.get("layer3_max_lookahead_bars", 100))
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
            
            train_idx_list = []
            if val_start > purge_bars:
                train_idx_list.extend(range(0, int(val_start - purge_bars)))
            if val_stop + purge_bars < n_samples:
                train_idx_list.extend(range(int(val_stop + purge_bars), n_samples))
            train_idx = np.array(train_idx_list, dtype=int)

            tprint_info(f"   > Processing Fold {fold_idx}/{int(len(folds))}...")

            df_train = df.iloc[train_idx]
            events_train = events_df.loc[events_df.index.intersection(df_train.index)]
            events_test = events_df.loc[events_df.index.intersection(df.index[test_idx])]

            if events_train.empty:
                logger.warning(f"Fold {fold_idx}: No training events. Skipping.")
                continue

            fold_results = self._optimize_families(df_train, events_train)
            if not fold_results:
                continue

            fold_geometries = self._select_best_geometries(df_train, events_train, fold_results, require_passed=False)
            if not fold_geometries:
                continue

            if len(fold_geometries) > 10:
                fold_geometries = fold_geometries[:10]

            standardized_geos = []
            geos_sorted = sorted(fold_geometries, key=lambda x: x.final_score, reverse=True)
            for rank, g in enumerate(geos_sorted):
                g_copy = copy.deepcopy(g)
                g_copy.uuid = f"Unified_Rank{rank}"
                standardized_geos.append(g_copy)

            X_train_events = None
            X_test_events = None
            feature_cols_for_models = []

            try:
                X_train_events_full = self._build_geometry_independent_event_features(df_train, events_train)
                fold_probe_features = self._select_global_probe_features(X_train_events_full)
            except Exception:
                X_train_events_full = None
                fold_probe_features = []

            if X_train_events_full is not None:
                if fold_probe_features:
                    feature_cols_for_models = [str(c) for c in list(fold_probe_features)]
                else:
                    feature_cols_for_models = [str(c) for c in list(X_train_events_full.columns)]
                X_train_events = X_train_events_full.reindex(columns=feature_cols_for_models).fillna(0.0)

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

            if trained_models:
                for uuid, model in trained_models.items():
                    if model is not None:
                        try:
                            stats = self._extract_tree_diagnostics(model)
                            self._all_tree_stats.append(stats)
                        except Exception:
                            pass

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
                lookahead_bars = int(np.ceil(float(max_h) * float(lookahead_scale))) + 1

                try:
                    test_end_pos = int(np.max(np.asarray(test_idx, dtype=int)))
                except Exception:
                    test_end_pos = int(test_idx[-1])
                label_end_pos = int(min(len(df) - 1, test_end_pos + lookahead_bars))
                df_label = df.iloc[: label_end_pos + 1]

                try:
                    X_test_events_full = self._build_geometry_independent_event_features(df_label, events_test)
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

                target_idx = events_test.index

                oof_scores.loc[target_idx] = fold_output.get('l2_score', fold_output.get('oof_labels')).reindex(target_idx)
                oof_labels.loc[target_idx] = fold_output.get('l2_label', pd.Series(np.nan, index=target_idx)).reindex(target_idx)
                oof_confidence.loc[target_idx] = fold_output.get('l2_confidence', pd.Series(np.nan, index=target_idx)).reindex(target_idx)
                oof_returns.loc[target_idx] = fold_output['oof_returns'].reindex(target_idx)
                oof_weights.loc[target_idx] = fold_output['weights'].reindex(target_idx)

                for uuid, series in fold_output['individual_geometries'].items():
                    if uuid in oof_geo_preds:
                        oof_geo_preds[uuid].loc[target_idx] = series.reindex(target_idx)

                for uuid, series in fold_output['individual_variances'].items():
                    if uuid in oof_geo_vars:
                        oof_geo_vars[uuid].loc[target_idx] = series.reindex(target_idx)

        final_geo_preds = {k: v for k, v in oof_geo_preds.items() if v.notna().any()}
        final_geo_vars = {k: v for k, v in oof_geo_vars.items() if v.notna().any()}

        try:
            c_ret = oof_returns.fillna(0.0)
            c_vol = df['volatility_1d'].reindex(oof_returns.index).ffill().fillna(0.0)
            safe_v = np.where(c_vol > 1e-9, c_vol, 1e-9)
            z = c_ret / safe_v
            sig = 1.0 / (1.0 + np.exp(-1.0 * z))
            quality_weights = pd.Series(0.5 + 1.5 * sig, index=oof_returns.index)
        except Exception:
            quality_weights = pd.Series(1.0, index=oof_returns.index)

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
            "tree_stats": self._all_tree_stats
        }

    def _calculate_ranking_metrics(self, y_true: pd.Series, y_score: pd.Series) -> Dict[str, float]:
        metrics = {}
        try:
            y_true_arr = pd.to_numeric(y_true, errors='coerce').fillna(0.0).values
            y_score_arr = pd.to_numeric(y_score, errors='coerce').fillna(0.0).values
            mask = np.isfinite(y_true_arr) & np.isfinite(y_score_arr)
            y_true_clean = y_true_arr[mask]
            y_score_clean = y_score_arr[mask]

            n_total = len(y_true_clean)
            if n_total < 10: return {}
            n_pos = np.sum(y_true_clean)
            global_pos_rate = n_pos / n_total if n_total > 0 else 0.0

            sorted_indices = np.argsort(y_score_clean)[::-1]
            y_true_sorted = y_true_clean[sorted_indices]

            for k in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
                cutoff_idx = int(n_total * k)
                if cutoff_idx < 1: continue
                top_k_true = y_true_sorted[:cutoff_idx]
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
        metrics = {}
        try:
            r = pd.to_numeric(returns, errors='coerce').dropna()
            if r.empty: return {}
            equity = (1 + r).cumprod()
            metrics["Cumulative_Gain"] = float(equity.iloc[-1] - 1.0) if not equity.empty else 0.0

            running_max = equity.cummax()
            drawdown = (equity - running_max) / running_max
            metrics["Max_Drawdown"] = float(drawdown.min())

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

            r_std = r.std()
            metrics["Sharpe_Ratio"] = float(r.mean() / r_std) if r_std > 1e-9 else 0.0
        except Exception as e:
            logger.warning(f"Failed to calculate portfolio metrics: {e}")
        return metrics

    def generate_reports(self, df, events_df, production_geometries, oof_results):
        """Step 4: Generate Reports."""
        tprint_info(">>> Layer 2: Step 4 - Generate Reports...")

        oof_scores = oof_results['l2_score']
        oof_labels = oof_results['l2_label']
        oof_returns = oof_results.get('oof_returns')
        if oof_returns is None and 'l2_returns' in oof_results:
             oof_returns = oof_results['l2_returns']
        oof_weights = oof_results['weights']
        final_geo_preds = oof_results['individual_geometries']

        diagnostics = oof_results.get('diagnostics', {})
        signal_inflation = diagnostics.get('signal_inflation_ratio', 0.0)
        n_bagged = diagnostics.get('n_bagged_signals', 0)
        n_base = diagnostics.get('n_base_events', 0)
        mean_probs = diagnostics.get('mean_consensus_prob', None)

        divergence_mean = 0.0
        divergence_std = 0.0
        corr_max_mean = 0.0
        coverage_diff_06 = 0.0

        if mean_probs is not None and oof_scores is not None:
            try:
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

        ranking_metrics = {}
        portfolio_metrics = {}
        global_metrics = {}

        try:
            valid_mask = oof_labels.notna() & oof_scores.notna()
            if valid_mask.sum() > 10:
                y_true = oof_labels[valid_mask]
                y_score = oof_scores[valid_mask]
                if len(np.unique(y_true)) > 1:
                    global_metrics["ROC_AUC"] = float(roc_auc_score(y_true, y_score))
                    global_metrics["PR_AUC"] = float(average_precision_score(y_true, y_score))
        except Exception:
            pass

        if oof_scores is not None and oof_labels is not None:
            ranking_metrics = self._calculate_ranking_metrics(oof_labels, oof_scores)

        if oof_returns is not None and oof_scores is not None:
            traded_mask = oof_scores > 0.5
            traded_returns = oof_returns[traded_mask]
            portfolio_metrics = self._calculate_portfolio_metrics(traded_returns)
            portfolio_metrics["Expected_Profit_Per_Trade"] = float(traded_returns.mean()) if not traded_returns.empty else 0.0

        try:
            cfg = getattr(self, "_current_config", {})
            if not isinstance(cfg, dict): cfg = {}
        except Exception: cfg = {}

        ts = str(cfg.get("run_timestamp") or datetime.utcnow().strftime("%Y%m%d_%H%M%S"))
        symbol = str(cfg.get("symbol", ""))
        timeframe = str(cfg.get("timeframe", ""))
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(parents=True, exist_ok=True)

        n_bars = int(len(df))
        n_events = int(len(events_df))

        prod_count = int(len(production_geometries or []))
        oof_labeled = int(pd.to_numeric(oof_labels, errors="coerce").notna().sum())
        oof_weight_nonzero = int((pd.to_numeric(oof_weights, errors="coerce").fillna(0.0).astype(float) > 0.0).sum())
        n_geo_channels = int(len(final_geo_preds or {}))

        try:
            n_total_events = len(oof_scores)
            n_signals = (oof_scores > 0.5).sum()
            coverage_pct = (n_signals / n_total_events * 100.0) if n_total_events > 0 else 0.0

            p_safe = oof_scores.clip(1e-9, 1.0 - 1e-9)
            entropy_vals = -(p_safe * np.log(p_safe) + (1.0 - p_safe) * np.log(1.0 - p_safe))
            entropy_mean = entropy_vals.mean()
            entropy_std = entropy_vals.std()

            all_tree_stats = oof_results.get('tree_stats') or getattr(self, '_all_tree_stats', [])
            avg_feats_used = np.mean([s['n_features_used'] for s in all_tree_stats]) if all_tree_stats else 0.0
            avg_depth = np.mean([s['avg_depth'] for s in all_tree_stats]) if all_tree_stats else 0.0
        except Exception:
            coverage_pct = 0.0
            entropy_mean = 0.0
            entropy_std = 0.0
            avg_feats_used = 0.0
            avg_depth = 0.0

        try:
            md_path = outcomes_dir / f"layer2_report_{symbol}_{timeframe}_{ts}.md"
            lines = [
                "# Layer2 Report (Unified)\n",
                f"- timestamp: {ts}\n",
                f"- symbol: {symbol}\n",
                f"- timeframe: {timeframe}\n",
                f"- n_bars: {n_bars}\n",
                f"- n_events: {n_events}\n",
                f"- production_geometries_n: {prod_count}\n",
                f"- oof_labeled_events: {oof_labeled}\n",
                f"- oof_nonzero_weight_events: {oof_weight_nonzero}\n",
                f"- oof_geometry_channels: {n_geo_channels}\n",
                "\n## Diagnostics\n",
                "### 1. Signal Coverage\n",
                f"- **Coverage**: {coverage_pct:.2f}%\n",
                "\n### 2. Prediction Entropy\n",
                f"- **Mean Entropy**: {entropy_mean:.4f}\n",
                f"- **Entropy Std**: {entropy_std:.4f}\n",
                "\n### 3. Feature Utilisation\n",
                f"- **Avg Features Used**: {avg_feats_used:.1f}\n",
                f"- **Avg Leaf Depth**: {avg_depth:.2f}\n",
                "\n### 4. Bagging Logic\n",
                f"- **Signal Inflation Ratio**: {signal_inflation:.2f}x\n",
                f"- **Max vs Mean Consensus**:\n",
                f"  - Mean Divergence: {divergence_mean:.4f} (std {divergence_std:.4f})\n",
                f"  - Correlation: {corr_max_mean:.4f}\n",
                "\n### 5. Ranking Quality\n",
            ]

            if ranking_metrics:
                lines.append("| Metric | Value |\n|---|---|\n")
                for k in sorted(ranking_metrics.keys()):
                    lines.append(f"| {k} | {ranking_metrics[k]:.4f} |\n")

            lines.append("\n### 6. Financial Metrics (Score > 0.5)\n")
            if portfolio_metrics:
                lines.append("| Metric | Value |\n|---|---|\n")
                for k in sorted(portfolio_metrics.keys()):
                    lines.append(f"| {k} | {portfolio_metrics[k]:.6f} |\n")

            if global_metrics:
                lines.append("\n### 7. Global Model Quality\n")
                for k, v in global_metrics.items():
                    lines.append(f"- **{k}**: {v:.4f}\n")

            if production_geometries:
                lines.append("\n### 8. Winning Geometries\n")
                lines.append("| UUID | Model | Race Score | Feats |\n|---|---|---|---|\n")
                for g in production_geometries:
                    m_type = "N/A"
                    race_score = "N/A"
                    if isinstance(g.model_params, dict):
                        m_type = g.model_params.get('model_type', 'lgbm')
                        r_scores = g.model_params.get('race_scores', {})
                        if r_scores and m_type in r_scores:
                             race_score = f"{r_scores[m_type]:.4f}"
                    n_feats = len(g.selected_features) if g.selected_features else 0
                    lines.append(f"| {g.uuid} | {m_type} | {race_score} | {n_feats} |\n")

            md_path.write_text("".join(lines))
            tprint_success(f"Generated Layer 2 Report: {md_path}")

            if self._rfe_stats:
                rfe_df = pd.DataFrame(self._rfe_stats)
                rfe_csv_path = outcomes_dir / f"titan_rfe_stats_{ts}.csv"
                rfe_df.to_csv(rfe_csv_path, index=False)

        except Exception as e:
            tprint_error(f"Report generation failed: {e}")

        try:
            summary_row: Dict[str, Any] = {
                "timestamp": ts,
                "symbol": symbol,
                "timeframe": timeframe,
                "n_bars": n_bars,
                "n_events": n_events,
                "production_geometries_n": prod_count,
                "oof_labeled_events": oof_labeled,
            }
            summary_row.update(ranking_metrics)
            summary_row.update(portfolio_metrics)
            summary_row.update(global_metrics)
            csv_path = outcomes_dir / f"layer2_summary_{symbol}_{timeframe}_{ts}.csv"
            pd.DataFrame([summary_row]).to_csv(csv_path, index=False)
        except Exception:
            pass

    def _extract_events_for_selection(
        self,
        df: pd.DataFrame,
        events_df: pd.DataFrame,
        max_horizon: int = 120
    ) -> List[Event]:
        """
        Convert DataFrame and events into the list of Event objects.
        """
        events_list = []
        df = df.sort_index()
        idx_locs = df.index.get_indexer(events_df.index)
        close_arr = df['close'].to_numpy()
        vol_arr = df['volatility_1d'].to_numpy()

        if 'event_consensus' in events_df.columns:
            directions = np.sign(events_df['event_consensus'].fillna(0).to_numpy())
        else:
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

            start_loc = loc
            end_loc = min(len(df), loc + max_horizon)
            price_path = close_arr[start_loc:end_loc]
            if len(price_path) < 2: continue

            entry_price = price_path[0]
            if entry_price <= 0: continue

            returns_path = (price_path - entry_price) / entry_price
            sigma = vol_arr[loc]
            if np.isnan(sigma) or sigma <= 0: sigma = 0.01

            e = Event(
                id=i,
                entry_idx=start_loc,
                exit_idx=end_loc,
                direction=int(directions[i]) if directions[i] != 0 else 1,
                returns_path=returns_path,
                sigma=float(sigma)
            )
            events_list.append(e)

        return events_list

    def _extract_tree_diagnostics(self, booster) -> Dict[str, float]:
        """
        Extract diagnostics from a trained LGBM booster.
        """
        if booster is None:
            return {'n_features_used': 0.0, 'avg_depth': 0.0, 'max_depth': 0.0}

        try:
            imp = booster.feature_importance(importance_type='split')
            n_features = int(np.sum(imp > 0))

            dump = booster.dump_model()
            trees = dump.get('tree_info', [])
            depths = []

            for tree in trees:
                if 'tree_structure' not in tree: continue
                stack = [(tree['tree_structure'], 0)]
                while stack:
                    node, d = stack.pop()
                    if 'leaf_index' in node:
                        depths.append(d)
                    else:
                        if 'left_child' in node: stack.append((node['left_child'], d + 1))
                        if 'right_child' in node: stack.append((node['right_child'], d + 1))

            avg_depth = float(np.mean(depths)) if depths else 0.0
            max_depth = float(np.max(depths)) if depths else 0.0

            return {'n_features_used': float(n_features), 'avg_depth': avg_depth, 'max_depth': max_depth}
        except Exception as e:
            logger.warning(f"Failed to extract tree diagnostics: {e}")
            return {'n_features_used': 0.0, 'avg_depth': 0.0, 'max_depth': 0.0}

    def _precompute_geometry_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()

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

        if 'geo_ret_10' not in df_out.columns:
            df_out['geo_ret_10'] = df_out['close'].pct_change(10).abs()
        if 'geo_ret_20' not in df_out.columns:
            df_out['geo_ret_20'] = df_out['close'].pct_change(20).abs()

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
        if events_index.empty:
            return pd.DataFrame()

        kappa = params.get('kappa')
        sl_mult = params.get('sl_mult')
        sl_sigma = params.get('sl_sigma')
        alpha = params.get('alpha')
        beta = params.get('beta')
        min_ratio = params.get('min_ratio')

        if sl_sigma is not None:
            eff_sl = float(sl_sigma)
        elif sl_mult is not None:
            eff_sl = float(sl_mult)
        else:
            eff_sl = 1.0

        if kappa is not None:
            eff_kappa = float(kappa)
        elif alpha is not None and beta is not None and min_ratio is not None and sl_sigma is not None:
            try:
                eff_kappa = (float(min_ratio) * (float(sl_sigma) ** float(alpha))) ** (1.0 / float(beta))
            except:
                eff_kappa = 2.0
        else:
            eff_kappa = 2.0

        try:
            subset = df.reindex(events_index)
            vol = subset['volatility_1d'].fillna(0.0)

            close = subset['close']
            atr_price = subset.get('geo_atr_14')
            if atr_price is None or atr_price.isna().all():
                atr_price = vol * close
            atr_price = atr_price.fillna(0.0)

            atr_pct = atr_price / close
            atr_pct = atr_pct.replace([np.inf, -np.inf], 0.0).fillna(0.0)

            min_profit = self.transaction_cost * 1.1
            min_sl_dist = 0.004
            max_tp_dist = 0.03

            raw_target = eff_kappa * vol
            target_size = np.maximum(raw_target, min_profit)
            target_size = np.minimum(target_size, max_tp_dist)
            target_size = target_size.replace(0.0, np.nan)

            raw_stop = eff_sl * vol
            stop_size = np.maximum(raw_stop, min_sl_dist)
            stop_size = stop_size.replace(0.0, np.nan)

            feats = pd.DataFrame(index=events_index)

            feats['geo_vol_to_stop'] = vol / stop_size
            feats['geo_vol_to_target'] = vol / target_size

            ret10 = subset.get('geo_ret_10', pd.Series(0, index=events_index)).fillna(0.0)
            ret20 = subset.get('geo_ret_20', pd.Series(0, index=events_index)).fillna(0.0)

            feats['geo_ret10_to_stop'] = ret10 / stop_size
            feats['geo_ret20_to_stop'] = ret20 / stop_size
            feats['geo_ret10_to_target'] = ret10 / target_size
            feats['geo_ret20_to_target'] = ret20 / target_size

            feats['geo_atr_to_stop'] = atr_pct / stop_size
            feats['geo_atr_to_target'] = atr_pct / target_size

            rng50 = subset.get('geo_range_50', atr_price * 3.0).fillna(0.0)
            rng50_pct = rng50 / close
            feats['geo_range_to_stop'] = rng50_pct / stop_size
            feats['geo_range_to_target'] = rng50_pct / target_size

            min50 = subset.get('geo_min_50', close)
            max50 = subset.get('geo_max_50', close)
            safe_atr = atr_price.replace(0.0, np.nan)

            feats['geo_dist_from_min'] = (close - min50) / safe_atr
            feats['geo_dist_from_max'] = (max50 - close) / safe_atr

            feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            return feats

        except Exception as e:
            logger.warning(f"Failed to compute specific geometry features: {e}")
            return pd.DataFrame(index=events_index)

    def _validate_inputs(self, df: pd.DataFrame) -> pd.DataFrame:
        required = ['close', 'volatility_1d']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in df: {missing}")
        return df

    def _generate_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 0: Generate events using CUSUM filter.
        """
        config = getattr(self, '_current_config', {})
        if not isinstance(config, dict): config = {}

        try:
            cfg_signals = dict(config)
            try:
                if 'k' not in cfg_signals:
                    k_override = cfg_signals.get('layer2_signal_k') or cfg_signals.get('layer2_default_k', 0.12)
                    cfg_signals['k'] = float(k_override)
            except Exception: pass

            signals = generate_primary_signals(df, **cfg_signals)
            consensus = pd.to_numeric(signals.get('consensus'), errors='coerce').reindex(df.index).fillna(0.0)
            self._primary_signals = pd.DataFrame({'consensus': consensus}, index=df.index)

            trigger_mask = consensus != 0.0
            dir_raw = str(config.get('direction', 'long')).lower()
            if dir_raw in {'long', 'buy', '1', '1.0', '+1', 'l'}:
                trigger_mask = trigger_mask & (consensus > 0.0)
            elif dir_raw in {'short', 'sell', '-1', '-1.0', 's'}:
                trigger_mask = trigger_mask & (consensus < 0.0)

            events = df.index[trigger_mask]
        except Exception as e:
            logger.warning(f"Error in CUSUM event generation: {e}. Fallback.")
            returns = df['close'].pct_change().abs()
            trigger_mask = (returns > 0.004).fillna(False)
            events = df.index[trigger_mask]

            consensus = pd.to_numeric(df['close'].pct_change().shift(1), errors='coerce').fillna(0.0)
            consensus = np.sign(consensus).reindex(df.index).fillna(0.0)
            self._primary_signals = pd.DataFrame({'consensus': consensus}, index=df.index)

        events_df = df.loc[events, ['volatility_1d']].copy()
        try:
            if self._primary_signals is not None and 'consensus' in self._primary_signals.columns:
                evt_cons = pd.to_numeric(self._primary_signals['consensus'].reindex(events_df.index), errors='coerce')
                evt_cons = np.sign(evt_cons).fillna(0.0)
                events_df['event_consensus'] = evt_cons.astype(float)
        except Exception:
            pass

        return events_df

    def _events_cache_key(self, events_index: pd.Index) -> Tuple[Any, ...]:
        n = int(len(events_index))
        if n <= 0: return (0, None, None, None, None)
        return (n, events_index[0], events_index[-1])

    def _df_cache_key(self, df: pd.DataFrame) -> Tuple[Any, ...]:
        n = int(len(df.index))
        if n <= 0: return (0, None, None)
        return (n, df.index[0], df.index[-1])

    def _select_global_probe_features(self, X_events: pd.DataFrame) -> List[str]:
        try:
            target_n = int(getattr(self, '_current_config', {}).get('layer2_probe_feature_count', 70))
        except Exception: target_n = 70
        try:
            corr_threshold = float(getattr(self, '_current_config', {}).get('layer2_probe_corr_threshold', 0.95))
        except Exception: corr_threshold = 0.95
        try:
            max_rows = int(getattr(self, '_current_config', {}).get('layer2_probe_corr_rows', 2000))
        except Exception: max_rows = 2000

        ranked = [str(c) for c in list(X_events.columns)]
        try:
            selected = self._cheap_corr_prune(X_events, ranked, target_n, corr_threshold, max_rows)
        except Exception:
            selected = ranked[: int(target_n)]
        return [c for c in selected if c in X_events.columns]

    def _get_or_build_signals(self, df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
        try:
            dir_raw = str(getattr(self, "_current_config", {}).get("direction", "long")).lower()
        except Exception: dir_raw = "long"
        default_dir = 1.0
        if dir_raw in {"short", "sell", "-1", "-1.0", "s"}: default_dir = -1.0

        key = (self._df_cache_key(df), self._events_cache_key(events_df.index), "Unified", float(default_dir))
        cached = self._signals_cache.get(key)
        if cached is not None: return cached

        base_cons = None
        try:
            if 'event_consensus' in events_df.columns:
                base_cons = pd.to_numeric(events_df['event_consensus'], errors='coerce').astype(float)
        except Exception: pass
        if base_cons is None and self._primary_signals is not None:
             try:
                 base_cons = pd.to_numeric(self._primary_signals['consensus'].reindex(events_df.index), errors='coerce').astype(float)
             except Exception: pass
        if base_cons is None:
            base_cons = pd.Series(float(default_dir), index=events_df.index)

        directions = np.sign(base_cons.to_numpy(dtype=float, copy=False))
        directions = np.where(np.isfinite(directions), directions, float(default_dir))
        directions[directions == 0.0] = float(default_dir)

        idx = df.index
        consensus_arr = np.zeros(len(idx), dtype=float)
        pos = idx.get_indexer(events_df.index)
        valid_pos = pos >= 0
        if np.any(valid_pos):
            consensus_arr[pos[valid_pos]] = directions[valid_pos]

        signals = pd.DataFrame({'consensus': consensus_arr}, index=idx)
        self._signals_cache[key] = signals
        return signals

    def _compute_dominance_labels(
        self,
        df: pd.DataFrame,
        events_df: pd.DataFrame,
        kappa: float = None,
        horizon: int = 120,
        sl_mult: float = None,
        sl_sigma: float = None,
        alpha: float = None,
        beta: float = None,
        min_ratio: float = None,
        events_shift: int = 0,
        **kwargs
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Compute TP/SL labels.
        """
        is_new_logic = (alpha is not None) and (beta is not None)
        if horizon is None: horizon = 120
        horizon = int(horizon)

        cache_key = (
            self._df_cache_key(df),
            self._events_cache_key(events_df.index),
            "Unified",
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

        eff_sl_sigma = sl_sigma if sl_sigma is not None else (sl_mult if sl_mult is not None else 1.0)
        vol_series = pd.to_numeric(df.get('volatility_1d'), errors='coerce').astype(float).reindex(events_df.index).fillna(0.0)

        min_sl_dist = 0.004
        max_tp_dist = 0.03
        stop_threshold = None
        profit_threshold = None

        if is_new_logic:
            raw_stop = eff_sl_sigma * vol_series
            stop_threshold = np.maximum(raw_stop, min_sl_dist)
        else:
            vol_median = vol_series.median()
            vol_adj_factor = 1.0 + 0.3 * ((vol_series - vol_median) / (vol_median + 1e-9))
            vol_adj_factor = vol_adj_factor.clip(lower=0.7, upper=1.3)
            raw_stop = float(eff_sl_sigma) * vol_series * vol_adj_factor
            stop_threshold = np.maximum(raw_stop, min_sl_dist)

            eff_kappa = kappa if kappa is not None else 2.0
            min_profit = self.transaction_cost * 1.1
            raw_target = float(eff_kappa) * vol_series * vol_adj_factor
            profit_threshold = np.maximum(raw_target, min_profit)
            profit_threshold = np.minimum(profit_threshold, max_tp_dist)

        signals = self._get_or_build_signals(df, events_df)

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

            vol_shifted = pd.to_numeric(df.get('volatility_1d'), errors='coerce').astype(float).reindex(calc_events_idx).fillna(0.0)
            if is_new_logic:
                stop_threshold = eff_sl_sigma * vol_shifted
            else:
                vol_median = vol_shifted.median()
                vol_adj_factor = 1.0 + 0.3 * ((vol_shifted - vol_median) / (vol_median + 1e-9))
                vol_adj_factor = vol_adj_factor.clip(lower=0.7, upper=1.3)
                stop_threshold = float(eff_sl_sigma) * vol_shifted * vol_adj_factor
                min_profit = self.transaction_cost * 1.1
                profit_threshold = np.maximum(float(eff_kappa if kappa else 2.0) * vol_shifted * vol_adj_factor, min_profit)
        
        if is_new_logic:
            (
                realized_returns, _, exit_reasons, _,
                mfe_series, mae_series, _, _
            ) = compute_realized_returns(
                df=df,
                signals=calc_signals,
                profit_threshold=None,
                stop_threshold=stop_threshold,
                horizon=horizon,
                transaction_cost=self.transaction_cost,
                min_event_spacing=0
            )
            mfe_aligned = mfe_series.reindex(calc_events_idx)
            mae_aligned = mae_series.reindex(calc_events_idx)
            exit_aligned = exit_reasons.reindex(calc_events_idx)
            vol_at_entry = pd.to_numeric(df.get('volatility_1d'), errors='coerce').astype(float).reindex(calc_events_idx).fillna(0.0)

            mfe_capped = np.minimum(mfe_aligned, max_tp_dist)
            norm_mfe = mfe_capped / vol_at_entry
            norm_mae = mae_aligned / vol_at_entry
            norm_mae_safe = norm_mae.replace(0.0, 1e-6)

            score = (norm_mfe ** float(beta)) / (norm_mae_safe ** float(alpha))
            is_stop = exit_aligned == 'stop'
            is_profit = (score >= float(min_ratio)) & (~is_stop)
            binary_labels = is_profit.astype(float)
            binary_labels[realized_returns.reindex(calc_events_idx).isna()] = np.nan

            subset_returns = realized_returns.reindex(calc_events_idx)
            subset_returns[is_profit] = mfe_aligned[is_profit]
            subset_mfe = mfe_series.reindex(calc_events_idx)
            subset_mae = mae_series.reindex(calc_events_idx)
            subset_exit = exit_aligned
        else:
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
                min_event_spacing=0
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

    def _build_geometry_independent_event_features(self, df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=df.index)
        try:
            if self._primary_signals is not None and 'consensus' in self._primary_signals.columns:
                consensus = pd.to_numeric(self._primary_signals['consensus'].reindex(df.index), errors='coerce').fillna(0.0)
            else:
                consensus = np.sign(df['close'].pct_change()).fillna(0.0)
            signals['consensus'] = consensus.astype(float)
        except Exception:
            signals['consensus'] = 0.0

        volume_available = ('volume' in df.columns)
        meta_features = create_meta_features(
            df=df,
            signals=signals,
            volume_available=volume_available,
            include_raw_signals=False,
            use_kalman=True,
        )
        meta_features = meta_features.apply(pd.to_numeric, errors='coerce')

        try:
            forbidden = {"vol_ratio", "vol_expansion", "returns_std_50"}
            cols = [c for c in meta_features.columns if str(c) not in forbidden and not str(c).startswith("zigzag_")]
            meta_features = meta_features[cols]
        except Exception: pass

        X_events = meta_features.reindex(events_df.index).replace([np.inf, -np.inf], np.nan)
        
        vol_event = pd.to_numeric(events_df.get('volatility_1d'), errors='coerce').fillna(0.0)
        close_event = pd.to_numeric(df['close'].reindex(events_df.index), errors='coerce')
        stop_sigma = vol_event * MIN_SL_PCT
        target_sigma = stop_sigma * MIN_TP_SL_RATIO
        
        X_events['event_stop_sigma'] = stop_sigma
        X_events['event_target_sigma'] = target_sigma
        
        return X_events

    def _get_target_sample_weight_for_events(self, df: pd.DataFrame, events_df: pd.DataFrame) -> Optional[pd.Series]:
        cfg = getattr(self, '_current_config', {})
        raw = cfg.get('target_sample_weight') if isinstance(cfg, dict) else None
        if raw is None: return None

        try:
            if isinstance(raw, pd.Series): w_full = raw.reindex(df.index)
            else: w_full = pd.Series(raw, index=df.index)
            w_events = w_full.reindex(events_df.index).fillna(1.0).clip(lower=0.0)
            return w_events
        except Exception:
            return None

    def _robust_normalize(self, series: pd.Series) -> pd.Series:
        s_log = np.log1p(series)
        q1, q3 = s_log.quantile(0.25), s_log.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0: iqr = 1.0
        return ((s_log - s_log.median()) / iqr).clip(lower=0)

    def _compute_root_dispersion(self, df: pd.DataFrame, feature_names: List[str], decay: float = 0.7) -> pd.Series:
        if df.empty: return pd.Series(0.0, index=feature_names)
        splits = df[['tree_index', 'split_feature', 'node_depth']]
        min_depth = splits.groupby(['tree_index', 'split_feature'])['node_depth'].min().reset_index()
        min_depth['w'] = np.exp(-decay * min_depth['node_depth'])
        return min_depth.groupby('split_feature')['w'].mean().reindex(feature_names).fillna(0.0)

    def _select_optimal_k(self, rfe_history_df, effective_n_samples, tree_depth=6):
        shadow_rows = rfe_history_df[rfe_history_df['feature'] == 'SHADOW_NOISE']
        shadow_cutoff = np.percentile(shadow_rows['hafsr_score'], 75) if not shadow_rows.empty else 0.0

        clean_df = rfe_history_df[rfe_history_df['feature'] != 'SHADOW_NOISE'].sort_values('hafsr_score', ascending=False)
        scores = clean_df['hafsr_score'].clip(lower=0).values
        total_signal = scores.sum()
        if total_signal == 0: return []

        cumulative = np.cumsum(scores) / total_signal
        max_k = max(5, int(effective_n_samples / (tree_depth * 8)))

        optimal_k = len(clean_df)
        for k in range(1, len(clean_df) + 1):
            if k > max_k:
                optimal_k = k - 1
                break
            if scores[k-1] <= shadow_cutoff:
                optimal_k = k - 1
                break
            if k > 5 and cumulative[k-1] >= 0.95:
                optimal_k = k
                break

        return clean_df.iloc[:max(1, optimal_k)]['feature'].tolist()

    def _calculate_dynamic_score(self, model, feature_names, weights):
        try:
            booster = model.booster_ if hasattr(model, 'booster_') else model
            df = booster.trees_to_dataframe()
            df = df[df['split_feature'].notna()]

            total_gain = df.groupby('split_feature')['split_gain'].sum().reindex(feature_names).fillna(0)
            avg_gain = total_gain / (df.groupby('split_feature')['split_gain'].count().reindex(feature_names).fillna(0) + 1e-9)

            df_struct = df[df['node_depth'] <= 6].copy()
            df_struct['w_gain'] = df_struct['split_gain'] * (0.5 ** df_struct['node_depth'])
            struct_gain = df_struct.groupby('split_feature')['w_gain'].sum().reindex(feature_names).fillna(0)

            uniformity = self._compute_root_dispersion(df, feature_names)

            scores = pd.DataFrame({
                'T': self._robust_normalize(total_gain),
                'A': self._robust_normalize(avg_gain),
                'S': self._robust_normalize(struct_gain),
                'U': self._robust_normalize(uniformity)
            })
            scaled = MinMaxScaler().fit_transform(scores)

            return (weights['total'] * scaled[:,0] + weights['avg'] * scaled[:,1] +
                    weights['struct'] * scaled[:,2] + weights['uni'] * scaled[:,3])
        except Exception:
            return pd.Series(0, index=feature_names)

    def _calculate_hafsr_dynamic(self, fold_df, shadow_vals, n_cv):
        means = fold_df.mean(axis=1)
        stds = fold_df.std(axis=1)
        # HAFSR = Mean / (1 + CV) = Mean^2 / (Mean + Std)
        hafsr = (means ** 2) / (means + stds + 1e-9)
        return hafsr

    def _cluster_and_deduplicate(self, X, scores, top_n):
        if X.shape[1] < top_n: return list(X.columns)
        try:
            corr = X.corr().abs().fillna(0)
            dist = 1 - corr
            Z = linkage(squareform(dist), method='average')
            labels = fcluster(Z, t=top_n, criterion='maxclust')
            selected = []
            features = list(X.columns)
            for i in range(1, max(labels) + 1):
                cluster_feats = [features[j] for j in range(len(features)) if labels[j] == i]
                best = max(cluster_feats, key=lambda f: scores.get(f, 0))
                selected.append(best)
            return selected
        except: return list(X.columns)[:top_n]

    def _run_titan_rfe(self, X_work, y, cv_splits, volatility_series, min_features=70):
        current_features = list(X_work.columns)
        n_cv_active = len(cv_splits)

        while len(current_features) > min_features:
            n_feats = len(current_features)
            use_sfi = (n_feats <= 250)
            w = {'total': 0.30, 'avg': 0.15, 'struct': 0.15, 'uni': 0.10} if use_sfi else \
                {'total': 0.40, 'avg': 0.25, 'struct': 0.20, 'uni': 0.15}

            focal_obj = RobustFocalLoss(gamma_pos=0.5, gamma_neg=1.25, alpha=0.65, verbose=False)
            def lgbm_focal_obj(y_t, y_p): return focal_obj(y_p, y_t)

            def run_fold(tr, val, feats):
                X_tr = X_work.iloc[tr]
                y_tr = y.iloc[tr]
                vol = volatility_series.iloc[tr]
                wts = np.clip(1.0/(vol + 1e-5), 0, np.quantile(1.0/(vol+1e-5), 0.99))

                model = lgb.LGBMClassifier(objective=lgbm_focal_obj, n_estimators=max(50, int(2*len(feats))), verbose=-1)
                model.fit(X_tr[feats], y_tr, sample_weight=wts)
                score = self._calculate_dynamic_score(model, feats, w)

                if use_sfi:
                    sfi_mod = lgb.LGBMClassifier(objective=lgbm_focal_obj, n_estimators=100, max_depth=2, verbose=-1)
                    sfi = []
                    for f in feats:
                        try:
                            sfi_mod.fit(X_tr[[f]], y_tr, sample_weight=wts)
                            p = expit(sfi_mod.predict(X_work.iloc[val][[f]], raw_score=True))
                            sfi.append(-log_loss(y.iloc[val], p))
                        except: sfi.append(-10.0)
                    score += 0.3 * MinMaxScaler().fit_transform(self._robust_normalize(np.exp(sfi)).reshape(-1,1)).flatten()
                return score

            res = Parallel(n_jobs=1)(delayed(run_fold)(tr, val, current_features) for tr, val in cv_splits[:2])
            fold_df = pd.concat([pd.Series(r, index=current_features) for r in res], axis=1)

            stability = self._calculate_hafsr_dynamic(fold_df, None, n_cv_active)
            ranked = stability.sort_values(ascending=False)
            current_features = ranked.index[:max(min_features, int(len(ranked)*0.5))].tolist()

        return current_features

    def _aggregate_geometry_labels_for_feature_selection(
        self,
        df: pd.DataFrame,
        events_df: pd.DataFrame,
        geometries: List[GeometryTrial],
    ) -> pd.Series:
        if events_df is None or getattr(events_df, 'empty', True) or not geometries:
            return pd.Series(np.nan, index=getattr(events_df, 'index', pd.Index([])), dtype=float)

        sum_w = pd.Series(0.0, index=events_df.index, dtype=float)
        sum_lbl = pd.Series(0.0, index=events_df.index, dtype=float)

        for g in list(geometries):
            try:
                lbls, _, _, _, _ = self._compute_dominance_labels(df, events_df, **getattr(g, 'params', {}))
                lbls = pd.to_numeric(lbls, errors='coerce').astype(float).reindex(events_df.index)
                valid = lbls.notna()
                if not bool(valid.any()): continue

                w_g = float(getattr(g, 'final_score', 1.0))
                idx = lbls.index[valid]
                sum_lbl.loc[idx] += (w_g * lbls.loc[idx])
                sum_w.loc[idx] += w_g
            except Exception: continue

        y_soft = pd.Series(np.nan, index=events_df.index, dtype=float)
        valid_w = sum_w > 0.0
        y_soft.loc[valid_w] = (sum_lbl.loc[valid_w] / sum_w.loc[valid_w]).astype(float)
        return (y_soft >= 0.5).astype(float)

    def _select_supervised_features_for_events(
        self,
        X_events_full: pd.DataFrame,
        y_target: pd.Series,
        layer1_weight_events: Optional[pd.Series],
        volatility_series: Optional[pd.Series] = None
    ) -> List[str]:
        if X_events_full is None or y_target is None: return []
        valid = y_target.notna()
        if int(valid.sum()) < 100: return []

        y_clean = pd.to_numeric(y_target.loc[valid], errors='coerce').astype(int)
        if y_clean.nunique() < 2: return []
        X_clean = X_events_full.loc[valid].fillna(0.0)

        tscv = TimeSeriesSplit(n_splits=2)
        focal_obj = RobustFocalLoss(verbose=False)
        def lgbm_focal_obj(y_t, y_p): return focal_obj(y_p, y_t)

        base_model = lgb.LGBMClassifier(objective=lgbm_focal_obj, n_estimators=100, verbose=-1)

        def process_fold_initial(tr, val, feats):
            w = (1.0 / (volatility_series.iloc[tr] + 1e-5)).clip(0, 1e2) if volatility_series is not None else None
            m = clone(base_model)
            m.fit(X_clean.iloc[tr], y_clean.iloc[tr], sample_weight=w)
            return self._calculate_dynamic_score(m, feats, {'total': 0.4, 'avg': 0.25, 'struct': 0.2, 'uni': 0.15})

        curr_feats = list(X_clean.columns)
        res = Parallel(n_jobs=1)(delayed(process_fold_initial)(tr, val, curr_feats) for tr, val in tscv.split(X_clean))
        hafsr = self._calculate_hafsr_dynamic(pd.concat(res, axis=1), None, 2)

        selected = self._cluster_and_deduplicate(X_clean, hafsr, top_n=150)
        final = self._run_titan_rfe(X_clean[selected], y_clean, list(tscv.split(X_clean)), volatility_series, min_features=70)

        return [c for c in final if c in X_events_full.columns]

    def _subsample_rows_for_proxy(self, df: pd.DataFrame, max_rows: int, seed: int = 42) -> pd.DataFrame:
        if len(df) <= max_rows: return df
        rng = np.random.default_rng(seed)
        return df.iloc[rng.choice(len(df), size=max_rows, replace=False)]

    def _cheap_corr_prune(self, X, ranked, target_n, corr_threshold, max_rows):
        sorted_cols = [c for c in ranked if c in X.columns]
        if not sorted_cols: return []

        df_sample = self._subsample_rows_for_proxy(X[sorted_cols].fillna(0), max_rows)
        try: corr_matrix = df_sample.corr().abs()
        except: return sorted_cols[:target_n]

        corr_arr = corr_matrix.to_numpy()
        col_to_idx = {c: i for i, c in enumerate(corr_matrix.columns)}
        selected = []
        selected_idx = []

        for col in sorted_cols:
            if len(selected) >= target_n: break
            i = col_to_idx.get(col)
            if i is None: continue

            if not selected_idx or not np.any(corr_arr[i, selected_idx] > corr_threshold):
                selected.append(col)
                selected_idx.append(i)

        return selected

    def _train_probes(self, X, y, sample_weight=None, trial=None):
        valid = y.notna()
        X_clean = X.loc[valid].fillna(0.0)
        y_clean = y.loc[valid].astype(int)
        if len(y_clean) < 50 or y_clean.nunique() < 2:
            return {'auc': 0.5, 'passed': False}

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        lgbm = lgb.LGBMClassifier(n_estimators=100, verbose=-1, n_jobs=1)
        linear = LinearRegression(n_jobs=1)
        scaler = StandardScaler()

        scores_lgbm, scores_lin = [], []

        for tr, te in tscv.split(X_clean):
            X_tr, X_te = X_clean.iloc[tr], X_clean.iloc[te]
            y_tr, y_te = y_clean.iloc[tr], y_clean.iloc[te]
            if y_tr.nunique() < 2: continue
            
            w_tr = sample_weight[tr] if sample_weight is not None else None

            linear.fit(scaler.fit_transform(X_tr), y_tr, sample_weight=w_tr)
            p_lin = expit(np.clip(linear.predict(scaler.transform(X_te)), -20, 20))
            try: scores_lin.append(roc_auc_score(y_te, p_lin))
            except: pass

            lgbm.fit(X_tr, y_tr, sample_weight=w_tr)
            try: scores_lgbm.append(roc_auc_score(y_te, lgbm.predict_proba(X_te)[:,1]))
            except: pass

        final_auc = np.mean(scores_lgbm + scores_lin) if (scores_lgbm or scores_lin) else 0.5
        return {'auc': final_auc, 'passed': final_auc > 0.52}

    def _train_full_lgbm_probe(self, X, y, sample_weight=None):
        valid = y.notna()
        X_clean = X.loc[valid].fillna(0.0)
        y_clean = y.loc[valid].astype(int)
        if len(y_clean) < 100 or y_clean.nunique() < 2:
            return {'auc_full': 0.5}

        tscv = TimeSeriesSplit(n_splits=3)
        lgbm = lgb.LGBMClassifier(n_estimators=2000, verbose=-1, n_jobs=1)
        aucs = []

        for tr, te in tscv.split(X_clean):
            if y_clean.iloc[tr].nunique() < 2: continue
            lgbm.fit(X_clean.iloc[tr], y_clean.iloc[tr], sample_weight=sample_weight[tr] if sample_weight is not None else None)
            try: aucs.append(roc_auc_score(y_clean.iloc[te], lgbm.predict_proba(X_clean.iloc[te])[:,1]))
            except: pass

        return {'auc_full': np.mean(aucs) if aucs else 0.5}

    def _check_stability(self, df, events_df, params, base_score, family):
        # 1. Base Labels
        base_labels, _, _, _, _ = self._compute_dominance_labels(df, events_df, **params)
        # 2. Shifted Labels (+1 bar)
        shift1_labels, _, _, _, _ = self._compute_dominance_labels(df, events_df, events_shift=1, **params)
        # 3. Shifted Labels (-1 bar)
        shift_neg1_labels, _, _, _, _ = self._compute_dominance_labels(df, events_df, events_shift=-1, **params)

        idx = base_labels.dropna().index
        b = base_labels.reindex(idx)
        s1 = shift1_labels.reindex(idx)
        sn1 = shift_neg1_labels.reindex(idx)

        valid = b.notna() & s1.notna() & sn1.notna()
        if valid.sum() < 10: return False

        agree1 = (b[valid] == s1[valid]).mean()
        agree2 = (b[valid] == sn1[valid]).mean()

        return ((agree1 + agree2) / 2.0) >= 0.82

    def _tune_geometry_model_params(self, df, events_df, geometry):
        try:
            lbls, _, _, _, _ = self._compute_dominance_labels(df, events_df, **geometry.params)
            valid_lbls = lbls.dropna()
            if len(valid_lbls) < 100: return {}

            n_sub = min(2000, max(400, int(len(valid_lbls)*0.3)))
            idx = np.random.choice(valid_lbls.index, n_sub, replace=False)
            y_sub = valid_lbls.loc[idx]
            X_sub = self._build_geometry_independent_event_features(df, events_df.loc[idx]).fillna(0.0)

            split = int(len(X_sub)*0.8)
            winner, _ = _quick_5model_race(X_sub.iloc[:split], y_sub.iloc[:split], X_sub.iloc[split:], y_sub.iloc[split:])
            
            def objective(trial):
                focal_alpha = trial.suggest_float('focal_alpha', 0.1, 0.9)
                n_est = trial.suggest_int('n_estimators', 200, 1000)

                tr_idx, val_idx = idx[:split], idx[split:]
                focal_obj = RobustFocalLoss(alpha=focal_alpha, verbose=False)

                model = lgb.LGBMClassifier(objective=lambda y_t, y_p: focal_obj(y_p, y_t), n_estimators=n_est, verbose=-1)
                model.fit(X_sub.iloc[:split], y_sub.iloc[:split])
                return 1.0 - average_precision_score(y_sub.iloc[split:], expit(model.predict(X_sub.iloc[split:])))

            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=40, n_jobs=1)
            
            res = study.best_params
            res['model_type'] = winner
            return res
        except: return {}

    def _optimize_families(self, df: pd.DataFrame, events_df: pd.DataFrame) -> Dict[str, List[GeometryTrial]]:
        """
        Replaces Optuna optimization with label_geometry_selection logic (Unified).
        """
        if len(events_df) < 50: return {}

        X_events_all = self._build_geometry_independent_event_features(df, events_df)
        selection_events = self._extract_events_for_selection(df, events_df)
        X_events_reset = X_events_all.reset_index(drop=True)

        selected_raw = select_geometries(selection_events, {}, X_events_reset)
        if not selected_raw: return {}

        trials = []
        for i, (geom, survivors) in enumerate(selected_raw):
            survival_rate = len(survivors) / len(selection_events) if selection_events else 0.0
            t_obj = GeometryTrial(
                family='Unified',
                params={'sl_sigma': geom.sl_sigma, 'alpha': geom.alpha, 'beta': geom.beta,
                       'min_ratio': geom.min_ratio, 'horizon': geom.horizon},
                final_score=survival_rate * 100.0,
                learnability=0.5, robust_magnitude=0.0, stability=1.0, balance=1.0,
                raw_metrics={'passed': True, 'survivors': len(survivors)},
                uuid=f"Geo_Sel{i}"
            )
            trials.append(t_obj)

        return {'Unified': trials}

    def _select_best_geometries(
        self,
        df: pd.DataFrame,
        events_df: pd.DataFrame,
        family_results: Dict[str, List[GeometryTrial]],
        require_passed: bool = True,
    ) -> List[GeometryTrial]:

        all_trials = []
        for trials in family_results.values():
            all_trials.extend(trials)

        if require_passed:
            trials_use = [t for t in all_trials if getattr(t, 'final_score', 0) > 0]
        else:
            trials_use = all_trials

        trials_use.sort(key=lambda x: float(getattr(x, 'final_score', -1)), reverse=True)
        top_tier = trials_use[:20]

        if not top_tier: return []

        # Diversity selection
        selected = []

        # 1. Best Score
        selected.append(top_tier[0])

        # 2. Maximize Param Distance
        pool = top_tier[1:]

        while len(selected) < 10 and pool:
            best_cand = None
            max_dist = -1

            for cand in pool:
                # Simple param distance
                d = 0
                for s in selected:
                    d += abs(float(cand.params.get('sl_sigma', 0)) - float(s.params.get('sl_sigma', 0)))
                if d > max_dist:
                    max_dist = d
                    best_cand = cand

            if best_cand:
                selected.append(best_cand)
                pool.remove(best_cand)
            else:
                break

        return selected

    def _train_geometry_models(
        self,
        df: pd.DataFrame,
        X_events: pd.DataFrame,
        events_df: pd.DataFrame,
        geometries: List[GeometryTrial],
        X_events_full: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:

        models = {}
        for g in geometries:
            try:
                lbls, _, _, _, _ = self._compute_dominance_labels(df, events_df, **g.params)
                valid_lbls = lbls.dropna()
                if len(valid_lbls) < 20 or valid_lbls.nunique() < 2:
                    models[g.uuid] = None
                    continue

                X_base = X_events.loc[valid_lbls.index]
                geo_features = self._compute_specific_geometry_features(df, valid_lbls.index, g.params)
                X_train = pd.concat([X_base, geo_features], axis=1).fillna(0.0)
                y_train = valid_lbls

                focal_obj = RobustFocalLoss(gamma_pos=0.5, gamma_neg=1.25, alpha=0.65)
                clf = lgb.LGBMClassifier(
                    objective=lambda y_t, y_p: focal_obj(y_p, y_t),
                    n_estimators=500,
                    verbose=-1,
                    n_jobs=1
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
        geometries: List[GeometryTrial],
        trained_models: Optional[Dict[str, Any]] = None,
        X_events: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Step 3.4: Generate final bagged outputs (Unified).
        """
        n_events = len(events_df)
        n_geos = len(geometries)

        geo_probs = np.zeros((n_events, n_geos))
        geo_labels = np.zeros((n_events, n_geos))
        geo_returns = np.zeros((n_events, n_geos))
        weights = np.zeros((n_events, n_geos))

        oof_preds = {}
        oof_vars = {}

        for i, g in enumerate(geometries):
            lbls, rets, mfe, mae, _ = self._compute_dominance_labels(df, events_df, **g.params)

            geo_feats = self._compute_specific_geometry_features(df, events_df.index, g.params)

            # Predict
            probs = np.full(n_events, 0.5)
            if trained_models and g.uuid in trained_models and trained_models[g.uuid]:
                try:
                    X_sub = pd.concat([X_events.reindex(events_df.index), geo_feats.reindex(events_df.index)], axis=1).fillna(0.0)
                    raw = trained_models[g.uuid].predict(X_sub)
                    probs = 1.0 / (1.0 + np.exp(-raw))
                    oof_preds[g.uuid] = pd.Series(probs, index=events_df.index)
                    oof_vars[g.uuid] = pd.Series(_calculate_tree_variance(trained_models[g.uuid].booster_, X_sub), index=events_df.index)
                except: pass

            geo_probs[:, i] = probs
            geo_labels[:, i] = lbls.fillna(0).values
            geo_returns[:, i] = rets.fillna(0).values

            # Weighting
            w_mag = np.log1p(np.maximum(0, mfe.fillna(0).values))
            w_smooth = np.log1p(np.maximum(0, mfe.fillna(0).values / (mae.fillna(0).values + 1e-9)))
            weights[:, i] = w_mag * w_smooth

        # Aggregate (Max/Consensus)
        consensus_prob = np.max(geo_probs, axis=1)
        consensus_ret = np.average(geo_returns, weights=np.maximum(weights, 1e-9), axis=1)
        final_weights = np.sum(weights, axis=1)
        
        # Normalize weights
        if final_weights.sum() > 0:
            final_weights = finalize_sample_weights(final_weights)

        l2_score = pd.Series(consensus_prob, index=events_df.index)
        l2_label = (l2_score > 0.5).astype(float)

        return {
            "l2_score": l2_score,
            "l2_label": l2_label,
            "oof_returns": pd.Series(consensus_ret, index=events_df.index),
            "weights": pd.Series(final_weights, index=events_df.index),
            "individual_geometries": oof_preds,
            "individual_variances": oof_vars
        }
