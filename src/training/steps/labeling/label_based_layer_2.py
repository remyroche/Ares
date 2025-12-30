"""
Label-Based Layer 2: Regime-Conditional Geometry Optimization & Meta-Labeling
-----------------------------------------------------------------------------
This module implements the Layer 2 architecture for meta-labeling, focusing on
regime-conditional barrier optimization and ML learnability probes.

It performs:
1. Event generation based on Orthogonal Families (Vol, Trend, etc.).
2. Geometry Selection via Mutual Information & Uniqueness filtering.
3. Independent optimization of barrier geometries (Kappa/Horizon) implicitly via selection.
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
from functools import partial
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import spearmanr, rankdata, entropy as shannon_entropy
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
    Geometry as LegacyGeometry,
    MIN_SL_PCT,
    MIN_TP_SL_RATIO
)

from src.training.steps.labeling.regime_leaf_feature_extractor import (
    extract_regime_leaf_onehot_features,
)

from src.training.steps.labeling.lgbm_feature_selection import lgbm_feature_selection_pipeline

# Import Orthogonal Generation
from src.training.steps.labeling.orthogonal_label_generation import (
    orthogonal_label_generation,
    AdaptiveSymmetricCUSUMEvents,
    ATRShockEvents,
    KalmanTrendEvents,
    VWAPReversionEvents,
    MicrostructureEvents,
    EntropyEvents,
    TrendModulatedBreakoutEvents,
    KalmanRegimeEvents,
    VWAPCrossEvents,
    compute_dominance_labels,
    CusumEvents,
    VolatilityShockEvents,
    TrendInitiationEvents,
    MeanReversionExtremeEvents,
    LiquidityShockEvents,
    TimeEvents,
    OutputGeometry as OrthoGeometry
    AdaptiveSymmetricCUSUMEvents,
    ATRShockEvents,
    TrendModulatedBreakoutEvents,
    VWAPReversionEvents,
    MicrostructureEvents,
    EntropyEvents,
    Geometry as OrthoGeometry
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
        y_smooth = labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # 2. Robust Sigmoid
        p = expit(preds)
        p = np.clip(p, 1e-7, 1 - 1e-7)

        # 3. Vectorized Asymmetric Gamma
        gamma_arr = np.where(labels > 0.5, self.gamma_pos, self.gamma_neg)

        # 4. Focal Weights with Capping
        focal_weight = np.where(labels > 0.5, (1 - p), p) ** gamma_arr
        focal_weight = np.minimum(focal_weight, self.w_cap)

        # 5. Gradient & Hessian Calculation
        grad_bce = p - y_smooth
        alpha_factor = np.where(labels > 0.5, self.alpha, (1 - self.alpha))
        grad_focal = alpha_factor * focal_weight * grad_bce
        hess_bce = p * (1 - p)
        hess_focal = alpha_factor * focal_weight * hess_bce

        # 6. Mixing (Stability Anchor)
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
        is_sklearn = False
        try:
            if hasattr(dtrain, 'get_label'):
                labels = dtrain.get_label()
                logits = preds
            elif isinstance(dtrain, np.ndarray):
                labels = preds
                logits = dtrain
                is_sklearn = True
            else:
                labels = dtrain
                logits = preds
        except Exception:
             labels = dtrain
             logits = preds

        if not self._is_init:
            self._init_alpha(labels)

        y_smooth = labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        p = 1.0 / (1.0 + np.exp(-logits))
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


def _build_multi_target_matrix(
    events_df: pd.DataFrame,
    geometries: List,
    all_geometry_labels: Dict[str, pd.Series],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build multi-target matrix for all geometries + compute class-aware sample weights.
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
    """
    try:
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
        
        return probs
        
    except Exception as e:
        logger.warning(f"ExtraTreesRegressor multi-output failed: {e}")
        return None


def _calculate_tree_variance(booster, X) -> np.ndarray:
    """
    Calculate the variance of predictions across all trees in the ensemble (Tree Variation).
    Full implementation restored.
    """
    if booster is None:
        return np.zeros(X.shape[0])

    try:
        if hasattr(booster, 'calibrated_classifiers_'):
            if len(booster.calibrated_classifiers_) > 0:
                booster = booster.calibrated_classifiers_[0].base_estimator

        raw_booster = None
        if hasattr(booster, 'booster_'):
            raw_booster = booster.booster_
        elif hasattr(booster, 'get_booster'):
            raw_booster = booster.get_booster()
        else:
            raw_booster = booster

        leaf_indices_raw = None
        if hasattr(raw_booster, 'predict'):
            try:
                leaf_indices_raw = raw_booster.predict(X, pred_leaf=True)
            except Exception:
                pass

        if leaf_indices_raw is None:
            try:
                leaf_indices_raw = booster.predict(X, pred_leaf=True)
            except Exception:
                pass

        if leaf_indices_raw is None:
            return np.zeros(X.shape[0])
        
        if leaf_indices_raw.ndim == 1:
            leaf_indices = leaf_indices_raw.reshape(-1, 1)
        else:
            leaf_indices = leaf_indices_raw

        model_dump = None
        if hasattr(raw_booster, 'dump_model'):
            model_dump = raw_booster.dump_model()
        elif hasattr(booster, 'dump_model'):
            model_dump = booster.dump_model()

        if model_dump is None:
             return np.zeros(X.shape[0])

        trees = model_dump.get('tree_info', [])
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


def _quick_5model_race(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    random_state: int = 42,
) -> Tuple[str, Dict[str, float]]:
    """
    Fast 5-model race to determine best model type for a geometry.
    Full implementation restored.
    """
    scores = {}

    # --- Model 1: LGBM Standard ---
    try:
        focal_lgbm = RobustFocalLoss(gamma_pos=1.5, gamma_neg=3.0, alpha=None, verbose=False)
        params_lgbm = {
            'n_estimators': 500, 'learning_rate': 0.03, 'num_leaves': 63, 'max_depth': 7,
            'min_data_in_leaf': 20, 'feature_fraction': 0.8, 'verbosity': -1,
            'random_state': random_state, 'metric': 'average_precision', 'objective': focal_lgbm,
        }
        train_ds = lgb.Dataset(X_train, label=y_train)
        val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)
        model_lgbm = lgb.train(params_lgbm, train_ds, valid_sets=[val_ds], callbacks=[lgb.early_stopping(30, verbose=False)])
        preds_lgbm = model_lgbm.predict(X_val)
        scores['lgbm'] = roc_auc_score(y_val, expit(preds_lgbm))
    except Exception:
        scores['lgbm'] = 0.0

    # --- Model 2: LGBM Linear ---
    try:
        params_lgbm_lin = {
            'n_estimators': 500, 'learning_rate': 0.05, 'num_leaves': 63, 'max_depth': 4,
            'extra_trees': True, 'min_data_in_leaf': 20, 'verbosity': -1,
            'random_state': random_state, 'metric': 'average_precision', 'objective': focal_lgbm,
            'feature_fraction': 0.8,
        }
        model_lgbm_lin = lgb.train(params_lgbm_lin, train_ds, valid_sets=[val_ds], callbacks=[lgb.early_stopping(30, verbose=False)])
        preds_lgbm_lin = model_lgbm_lin.predict(X_val)
        scores['lgbm_linear'] = roc_auc_score(y_val, expit(preds_lgbm_lin))
    except Exception:
        scores['lgbm_linear'] = 0.0

    # --- Model 3: XGBoost Standard ---
    try:
        focal_xgb = XGBFocalLoss(gamma_pos=1.5, gamma_neg=3.0, alpha=None)
        model_xgb = xgb.XGBClassifier(
            n_estimators=400, learning_rate=0.04, max_depth=6, min_child_weight=10,
            subsample=0.8, colsample_bytree=0.8, objective=focal_xgb, eval_metric='aucpr',
            early_stopping_rounds=30, verbosity=0, random_state=random_state, n_jobs=1,
        )
        model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds_xgb = model_xgb.predict_proba(X_val)[:, 1]
        scores['xgb'] = roc_auc_score(y_val, preds_xgb)
    except Exception:
        scores['xgb'] = 0.0

    # --- Model 4: XGBoost Linear ---
    try:
        model_xgb_lin = xgb.XGBClassifier(
            booster='gblinear', n_estimators=100, learning_rate=0.1, objective='binary:logistic',
            eval_metric=['auc', 'aucpr'], early_stopping_rounds=30, verbosity=0,
            random_state=random_state, n_jobs=1,
        )
        model_xgb_lin.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds_xgb_lin = model_xgb_lin.predict_proba(X_val)[:, 1]
        scores['xgb_linear'] = roc_auc_score(y_val, preds_xgb_lin)
    except Exception:
        scores['xgb_linear'] = 0.0

    # --- Model 5: CatBoost ---
    try:
        model_cat = catboost.CatBoostClassifier(
            iterations=300, learning_rate=0.05, depth=6, class_weights={0: 1.0, 1: 3.0},
            verbose=False, random_seed=random_state, thread_count=1,
        )
        model_cat.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=30)
        preds_cat = model_cat.predict_proba(X_val)[:, 1]
        scores['catboost'] = roc_auc_score(y_val, preds_cat)
    except Exception:
        scores['catboost'] = 0.0

    winner = max(scores, key=scores.get)
    return winner, scores


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
    events: Optional[pd.DatetimeIndex] = None # Added for orthogonality

def get_entropy(series: pd.Series, base=2) -> float:
    """Computes Shannon Entropy of a digitized series."""
    value_counts = series.value_counts(normalize=True, sort=False)
    return shannon_entropy(value_counts, base=base)


def roll_entropy(series: pd.Series, window: int = 20, bins: int = 10) -> pd.Series:
    """Rolling Entropy to detect structural breaks."""
    def _ent(x):
        hist, bin_edges = np.histogram(x, bins=bins, density=True)
        # Avoid log(0)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))

    return series.rolling(window).apply(_ent, raw=True)


def get_serial_correlation(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Rolling serial correlation (autocorrelation at lag 1).
    High positive = Trending; Negative = Mean Reverting.
    """
    return series.rolling(window).apply(lambda x: pd.Series(x).autocorr(lag=1), raw=True)


def build_indicator_matrix(events: pd.DatetimeIndex, index: pd.DatetimeIndex, horizon: int = 1) -> pd.DataFrame:
    """
    Maps events to the full timeline.
    Crucial for uniqueness calculations.
    """
    arr = np.zeros(len(index), dtype=int)
    valid_events = events.intersection(index)
    if valid_events.empty:
        return pd.DataFrame(0, index=index, columns=[0])


    event_locs = index.get_indexer(valid_events)
    event_locs = event_locs[event_locs != -1]

    n_bars = len(index)
    for loc in event_locs:
        end_loc = min(loc + horizon, n_bars)
        arr[loc:end_loc] += 1

    # Cap at 1 for binary "Active" status
    arr = np.clip(arr, 0, 1)
    return pd.DataFrame(arr, index=index, columns=[0])


def generate_market_state_probe(price: pd.Series, volume: pd.Series) -> pd.DataFrame:
    """
    A 'Theory of Mind' Probe.
    Instead of just RSI, we test if the geometry is learnable based on
    Information Theory and Market Microstructure states.
    """
    df = pd.DataFrame(index=price.index)

    # 1. Serial Correlation (Trendiness vs Mean Reversion state)
    df['serial_corr'] = get_serial_correlation(price.pct_change(), window=20)

    # 2. Volatility Ratio (Expansion/Contraction state)
    vol_short = price.pct_change().rolling(10).std()
    vol_long = price.pct_change().rolling(60).std()
    df['vol_regime'] = vol_short / (vol_long + 1e-9)

    # 3. Entropy (Information state)
    # Are returns random or structured?
    df['entropy'] = roll_entropy(np.log(price).diff().fillna(0), window=50)

    # 4. Amihud Illiquidity (Liquidity state)
    # High value = Price moves easily with little volume (Fragile)
    ret_abs = price.pct_change().abs()
    df['illiquidity'] = (ret_abs / (volume * price + 1e-9)).rolling(20).mean()

    # 5. Relative Drawdown (Psychological state)
    roll_max = price.rolling(100).max()
    df['drawdown'] = (price / roll_max) - 1.0

    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


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
        self._labels_cache = {}
        self._signals_cache = {}
        self._geometry_label_cache = {}
        self._feature_selection_cache = {}
        self._all_tree_stats = []

        cpu_guess = max(1, (os.cpu_count() or 4) - 1)
        self._parallel_n_jobs = max(1, min(cpu_guess, 4))
        self._parallel_prefer = "threads"

        if not self.verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

    def execute(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._current_config = dict(config or {})
        return self.run(df)

    def _update_parallel_settings_from_config(self, cfg: Optional[Dict[str, Any]]) -> None:
        if isinstance(cfg, dict):
            try:
                n = cfg.get('layer2_parallel_n_jobs', self._parallel_n_jobs)
                if n is not None: self._parallel_n_jobs = int(n)
            except: pass

    def _fingerprint_dataframe(self, df: pd.DataFrame) -> str:
        return f"{len(df)}"

    def _hash_series_signature(self, series: Optional[pd.Series]) -> str:
        return "sig"

    def _maybe_sample_indices(self, index: pd.Index, max_rows: int) -> pd.Index:
        if max_rows <= 0 or len(index) <= max_rows: return index
        rng = np.random.default_rng(self.random_state)
        return index.take(np.sort(rng.choice(len(index), size=max_rows, replace=False)))

    def _dominance_label_wrapper(self, df: pd.DataFrame, events: pd.DatetimeIndex, **params) -> pd.Series:
        """Wrapper for _compute_dominance_labels to fit Orthogonal Generator interface."""
        # Construct a dummy events_df with just the index
        # We need to preserve 'family' if possible, but params usually cover it
        dummy_events = pd.DataFrame(index=events)
        # Call existing logic
        labels, _, _, _, _ = self._compute_dominance_labels(
            df, dummy_events, **params
        )
        return labels

    def _get_labeler_menu(self) -> Dict[str, Callable]:
        """Define the menu of labelers with baked-in parameters."""
        return {
            "SCALP": partial(self._dominance_label_wrapper, kappa=1.5, sl_mult=0.5, horizon=12),
            "SWING": partial(self._dominance_label_wrapper, kappa=2.0, sl_mult=1.0, horizon=24),
            "TREND": partial(self._dominance_label_wrapper, kappa=3.0, sl_mult=1.5, horizon=48)
        }

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute the Layer 2 pipeline.
        """
        tprint_info("Starting Layer 2 Pipeline...")

        # 1. Prepare (Validate df, setup caches)
        # Note: We do NOT generate global events_df here anymore,
        # but we initialize the structure.
        df, _, _, global_probe_features = self.prepare_data_and_events(df)

        # 2. Optimize (Orthogonal Selection)
        # This returns GeometryTrial objects which contain their own events.
        production_geometries, production_selected_features = self.optimize_production_geometries(
            df, None, global_probe_features=global_probe_features
        )

        # 3. Construct Global Event Union (for compatibility/reporting)
        events_df = self._construct_union_events_df(df, production_geometries)

        # 4. Validate (OOF)
        oof_results = self.run_oof_analytics(
            df, events_df, production_geometries,
            global_probe_features=global_probe_features,
            production_selected_features=production_selected_features
        )

        # 5. Report
        self.generate_reports(df, events_df, production_geometries, oof_results)

        return {
            **oof_results,
            "events_df": events_df,
            "selected_trials": [asdict(t) for t in production_geometries],
            "production_selected_features": list(getattr(self, '_production_selected_features', []) or []),
        }

    def prepare_data_and_events(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
        """Step 1: Stateless data preparation."""
        tprint_info(">>> Layer 2: Step 1 - Prepare Data...")
        self._labels_cache = {}
        self._signals_cache = {}

        df = self._validate_inputs(df)
        df = self._precompute_geometry_base_features(df)

        # We don't generate global events here. Return empty placeholders.
        events_df = pd.DataFrame()
        X_probe_events = pd.DataFrame()

        # We can try to select global probe features based on a sample if needed,
        # or defer until we have events. For now, empty.
        self._global_probe_features = []

        return df, events_df, X_probe_events, self._global_probe_features

    def _construct_union_events_df(self, df: pd.DataFrame, geometries: List[GeometryTrial]) -> pd.DataFrame:
        """Construct a composite events dataframe from selected geometries."""
        if not geometries:
            return pd.DataFrame()

        all_indices = []
        for g in geometries:
            if g.events is not None:
                all_indices.extend(g.events)

        if not all_indices:
            return pd.DataFrame()

        unique_indices = pd.DatetimeIndex(sorted(list(set(all_indices))))
        events_df = df.loc[unique_indices, ['trend_regime', 'vol_regime', 'volatility_1d']].copy()
        # Add default family/consensus cols if needed
        events_df['family'] = 'Unified'
        return events_df

    def optimize_production_geometries(
        self,
        df: pd.DataFrame,
        events_df: Optional[pd.DataFrame], # Ignored/None
        global_probe_features: Optional[List[str]] = None
    ) -> Tuple[List[GeometryTrial], List[str]]:
        """Step 2: Orthogonal Label Generation & Selection."""
        tprint_info(">>> Layer 2: Step 2 - Orthogonal Optimization...")

        # 1. Generate & Filter
        # Note: orthogonal_label_generation now handles labeling/looping internally
        ortho_geoms = orthogonal_label_generation(df)

        if not ortho_geoms:
            tprint_error("Layer 2: No orthogonal geometries selected.")
            return [], []

        # 2. Convert to GeometryTrial
        production_geometries = []
        for i, og in enumerate(ortho_geoms):
            # Note: og is now an OutputGeometry with .purity, .auc, .params
            
            # Score
            score = og.purity if og.purity is not None else 1.0

            # Params
            params = og.params.copy()
            # Ensure horizon is present
            if 'horizon' not in params:
                params['horizon'] = 120 # Fallback

            gt = GeometryTrial(
                family=og.family,
                params=params,
                final_score=score * 100.0, # Scale up
                learnability=og.auc, # Use the actual Probe AUC
                robust_magnitude=0.0,
                stability=1.0,
                balance=1.0,
                raw_metrics=og.metrics,
                uuid=f"{og.name}_{i}",
                events=og.events, # Store events!
                selected_features=None # Initialize explicitly
            )
            production_geometries.append(gt)

        self.selected_geometries = production_geometries

        # 3. Feature Selection (Optional) on Union
        # We need a union events_df to run global feature selection
        union_events = self._construct_union_events_df(df, production_geometries)
        X_events_full = self._build_geometry_independent_event_features(df, union_events)

        # Select global probe features based on Union
        self._global_probe_features = self._select_global_probe_features(X_events_full)

        # 4. Per-Geometry Titan RFE (Adaptive)
        tprint_info(">>> Layer 2: Running Titan RFE per geometry...")
        for gt in production_geometries:
            try:
                selected_feats = self._run_titan_rfe_for_geometry(df, gt)
                if selected_feats:
                    gt.selected_features = selected_feats
                    tprint_success(f"   ✅ {gt.uuid}: Selected {len(selected_feats)} features")
                else:
                    tprint_warning(f"   ⚠️ {gt.uuid}: Feature selection returned empty set.")
            except Exception as e:
                tprint_error(f"   ❌ {gt.uuid}: Feature selection failed: {e}")

        return production_geometries, self._global_probe_features

    def run_oof_analytics(
        self,
        df: pd.DataFrame,
        events_df: pd.DataFrame, # This is the UNION events_df
        production_geometries: Optional[List[GeometryTrial]] = None,
        global_probe_features: Optional[List[str]] = None,
        production_selected_features: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Step 3: OOF Analytics."""
        tprint_info(">>> Layer 2: Step 3 - OOF Analytics...")

        if not production_geometries:
            return {}

        # Initialize global containers
        idx = events_df.index
        oof_scores = pd.Series(np.nan, index=idx)
        oof_labels = pd.Series(np.nan, index=idx)
        oof_weights = pd.Series(np.nan, index=idx)

        # K-Fold Cross-Validation
        n_splits = 3
        fold_size = len(df) // n_splits
        folds = []
        for i in range(n_splits):
            start = i * fold_size
            end = (i + 1) * fold_size if i < n_splits - 1 else len(df)
            folds.append((start, end))

        generators = {
            "CUSUM": AdaptiveSymmetricCUSUMEvents(),
            "VOL": ATRShockEvents(),
            "TREND": TrendModulatedBreakoutEvents(),
            "MEAN": VWAPReversionEvents(),
            "LIQUIDITY": MicrostructureEvents(),
            "ENTROPY": EntropyEvents(),
            "BREAKOUT": TrendModulatedBreakoutEvents(),
            "KALMAN": KalmanTrendEvents(),
            "MR": VWAPReversionEvents(),
            "VWAP": VWAPCrossEvents()
            "MEAN_REV": VWAPReversionEvents(),
            "LIQUIDITY": MicrostructureEvents(),
            "ENTROPY": EntropyEvents(),
            # "TIME": EntropyEvents() # Replacing TIME with Entropy as it is structural
        }

        # For OOF, we treat 'production_geometries' as the selected strategy.
        # We retrain the strategy on Train and predict on Test.
        
        for i, (test_start, test_end) in enumerate(folds):
            tprint_info(f"OOF Fold {i+1}/{n_splits}...")
            
            # Setup Train/Test split (Walk-Forward / Purged K-Fold simplified)
            train_mask = np.ones(len(df), dtype=bool)
            train_mask[test_start:test_end] = False
            # Purge 100 bars
            p_start = max(0, test_start - 100)
            p_end = min(len(df), test_end + 100)
            train_mask[p_start:p_end] = False
            
            df_train = df[train_mask]
            
            # We predict on ALL events that fall into the test window
            # First, reconstruct events for each geometry on the full timeline (or test slice)
            # Actually, we should regenerate events on test slice to avoid lookahead?
            # Or use global events and mask?
            # Generators use expanding window. If we generate on full DF, it's safe (expanding).
            # If we slice df_test, expanding window resets, which is inconsistent.
            # So generate on full, slice events.
            
            # Train Models on df_train
            trained_models = {}
            
            for gt in production_geometries:
                # 1. Get events in Train
                # We need events from THIS family in Train
                gen = generators.get(gt.family)
                if not gen: continue
                
                # Regenerate events on df_train to simulate training environment
                # This ensures we don't rely on future data for normalization in Train
                train_evts_idx = gen.generate(df_train)
                
                if len(train_evts_idx) < 20: continue
                
                # Build dummy events df
                train_evts_df = pd.DataFrame(index=train_evts_idx)
                
                # Compute labels & weights
                labels, weights, _, _, _, _ = self._compute_dominance_labels(df_train, train_evts_df, **gt.params)
                valid_lbls = labels.dropna()
                
                if len(valid_lbls) < 20: continue

                # Features
                X_train = self._build_geometry_independent_event_features(df_train, train_evts_df)
                if X_train.empty: continue

                # Append geometry specific features first
                geo_feats = self._compute_specific_geometry_features(df_train, X_train.index, gt.params)
                X_train = pd.concat([X_train, geo_feats], axis=1).fillna(0.0)

                # Feature Selection Application
                # Priority: Geometry specific selection > Global selection > All
                selected_cols = None

                # Check for selected_features safely
                gt_selected = getattr(gt, 'selected_features', None)

                if gt_selected:
                    selected_cols = [c for c in gt_selected if c in X_train.columns]
                elif global_probe_features:
                    selected_cols = [c for c in global_probe_features if c in X_train.columns]

                if selected_cols:
                    X_train = X_train[selected_cols]

                X_train = X_train.loc[valid_lbls.index]
                y_train = valid_lbls
                # Align all
                common_idx = valid_lbls.index.intersection(X_train.index)
                X_train = X_train.loc[common_idx]
                y_train = valid_lbls.loc[common_idx]
                w_train = weights.loc[common_idx] if weights is not None else None

                # Train Model
                try:
                    focal_lgbm = RobustFocalLoss(verbose=False)
                    params = LAYER2_MODEL_CONSTANTS.copy()
                    params['objective'] = focal_lgbm
                    params['metric'] = 'auc'

                    clf = lgb.LGBMClassifier(**params)
                    clf.fit(X_train, y_train, sample_weight=w_train)
                    trained_models[gt.uuid] = clf
                except Exception as e:
                    logger.warning(f"Training failed for {gt.uuid} on fold {i}: {e}")

            # Predict on Test
            # We predict on events that occur in the Test window
            test_evts_map = {}
            for gt in production_geometries:
                gen = generators.get(gt.family)
                if not gen: continue

                # Generate on DF up to test_end to get expanding window stats correct
                df_context = df.iloc[:test_end]
                full_evts = gen.generate(df_context)
                # Slice to test window
                test_evts = full_evts[(full_evts >= df.index[test_start]) & (full_evts < df.index[test_end])]

                if len(test_evts) > 0:
                    test_evts_map[gt.uuid] = test_evts

            # Aggregate Predictions
            # We iterate test_evts_map, predict, and fill global OOF series

            for gt_uuid, evts in test_evts_map.items():
                model = trained_models.get(gt_uuid)
                if model is None: continue

                # Build Features for Test Events
                test_evts_df = pd.DataFrame(index=evts)
                X_test = self._build_geometry_independent_event_features(df_context, test_evts_df)
                if X_test.empty: continue

                # Geo features using params from production geometry object
                gt = next(g for g in production_geometries if g.uuid == gt_uuid)
                geo_feats = self._compute_specific_geometry_features(df_context, X_test.index, gt.params)
                X_test = pd.concat([X_test, geo_feats], axis=1).fillna(0.0)

                # Feature Selection Application (Test)
                selected_cols = None

                # Check for selected_features safely
                gt_selected = getattr(gt, 'selected_features', None)

                if gt_selected:
                    selected_cols = [c for c in gt_selected if c in X_test.columns]
                elif global_probe_features:
                    selected_cols = [c for c in global_probe_features if c in X_test.columns]

                if selected_cols:
                    X_test = X_test[selected_cols]

                # Predict
                preds = model.predict_proba(X_test)[:, 1]

                # Bagging: Max probability aggregation
                # If multiple geometries predict on the same timestamp, take max
                current_vals = oof_scores.loc[evts].fillna(0.0)
                new_vals = np.maximum(current_vals, preds)
                oof_scores.loc[evts] = new_vals

                # Weights: 1.0 for now
                oof_weights.loc[evts] = 1.0

        # Construct oof_labels (0.5 threshold)
        oof_labels = (oof_scores >= 0.5).astype(float)

        # Calculate Returns for OOF events (Consensus)
        # We need returns for the union of events generated in Test folds
        # oof_scores index contains all predicted events.
        # We need realized returns for these events.
        # Since we don't know WHICH geometry "won" the max, we use a generic return (e.g. at fixed horizon 120)
        # or we try to reconstruct weighted return.
        # Simplification: Calculate return at horizon=120 for all events
        oof_returns = pd.Series(np.nan, index=oof_scores.index)
        valid_idx = oof_scores.dropna().index

        if not valid_idx.empty:
            # Stub return calculation
            # Use compute_realized_returns with default params
            ret, _, _, _, _, _, _, _ = compute_realized_returns(
                df, pd.DataFrame({'consensus': 1}, index=df.index),
                profit_threshold=None, stop_threshold=None, horizon=120,
                transaction_cost=self.transaction_cost
            )
            oof_returns.loc[valid_idx] = ret.loc[valid_idx]

        return {
            "l2_score": oof_scores,
            "oof_labels": oof_labels,
            "oof_returns": oof_returns,
            "weights": oof_weights
        }

    # ... (Keep existing helpers like _extract_tree_diagnostics, _precompute_geometry_base_features, _validate_inputs) ...

    def _validate_inputs(self, df: pd.DataFrame) -> pd.DataFrame:
        # Basic validation ensuring volatility exists
        if 'volatility_1d' not in df.columns:
            df = df.copy()
            df['volatility_1d'] = df['close'].pct_change().rolling(50).std()
        return df

    def _precompute_geometry_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Restore basic precomputation
        df_out = df.copy()
        if 'geo_atr_14' not in df_out.columns:
            try:
                high = df_out['high'] if 'high' in df_out.columns else df_out['close']
                low = df_out['low'] if 'low' in df_out.columns else df_out['close']
                close = df_out['close']
                tr = pd.concat([(high - low).abs(), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
                df_out['geo_atr_14'] = tr.rolling(14).mean()
            except Exception:
                pass
        return df_out

    def _compute_specific_geometry_features(self, df, events_index, params):
        """Re-implement basic ratio features."""
        if events_index.empty: return pd.DataFrame()

        subset = df.reindex(events_index)
        vol = subset['volatility_1d'].fillna(0.0)

        sl_mult = params.get('sl_mult', 1.0)
        stop_size = (vol * sl_mult).replace(0.0, np.nan)

        feats = pd.DataFrame(index=events_index)
        feats['geo_vol_to_stop'] = vol / stop_size
        return feats.fillna(0.0)

    def _run_titan_rfe_for_geometry(self, df: pd.DataFrame, gt: GeometryTrial) -> List[str]:
        """
        Run adaptive Titan RFE for a specific geometry.
        """
        # 1. Regenerate events/labels for this geometry on the full df (or relevant slice)

        # Re-generate events if not present or need full context
        if gt.events is None or len(gt.events) == 0:
            return []

        events_df = pd.DataFrame(index=gt.events)

        # Compute labels
        # Note: We use the same params as optimization
        labels, _, _, _, _ = self._compute_dominance_labels(df, events_df, **gt.params)
        valid_mask = ~labels.isna()
        y = labels[valid_mask]

        if len(y) < 50:
            return []

        # Build features
        # We use the subset of events
        events_subset = events_df.loc[y.index]
        X = self._build_geometry_independent_event_features(df, events_subset)

        if X is None or X.empty:
            return []

        # Add geometry specific features
        geo_feats = self._compute_specific_geometry_features(df, X.index, gt.params)
        X = pd.concat([X, geo_feats], axis=1).fillna(0.0)

        # Run Pipeline
        # We aim for ~60 features but adaptive to sample size (1 per 100 samples)
        # The pipeline handles the adaptation internally.
        target_sets = [60, 50, 40, 30, 20, 10]

        feature_sets, _ = lgbm_feature_selection_pipeline(
            X, y,
            target_feature_sets=target_sets,
            samples_per_feature_ratio=100
        )

        if not feature_sets:
            # Fallback: respect adaptive limit
            # max_allowed was calculated inside pipeline, but we can approximate it or be safe
            n_samples = len(y)
            limit = max(1, n_samples // 100)
            return list(X.columns)[:limit]

        # Return the largest available set (keys are ints)
        best_k = max(feature_sets.keys())
        return feature_sets[best_k]

    def _select_global_probe_features(self, X_events: pd.DataFrame) -> List[str]:
        if X_events is None or X_events.empty: return []
        # Simple variance filter
        return [c for c in X_events.columns if X_events[c].var() > 1e-6][:70]

    def _build_geometry_independent_event_features(self, df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
        if events_df.empty: return pd.DataFrame()
        # Use MTF feature generation
        signals = pd.DataFrame({'consensus': 1.0}, index=df.index)
        try:
            X = create_meta_features(df, signals, events_df.index)

            # --- Enhanced Probe Features ---
            if 'close' in df.columns and 'volume' in df.columns:
                probe_feats = generate_market_state_probe(df['close'], df['volume'])
                # Reindex to events
                probe_feats_events = probe_feats.reindex(events_df.index).fillna(0.0)
                X = pd.concat([X, probe_feats_events], axis=1)
            # -------------------------------

            return X
        except Exception as e:
            logger.warning(f"Feature generation failed: {e}")
            return pd.DataFrame(index=events_df.index)

    def _compute_dominance_labels(self, df, events_df, **kwargs):
        # Use Vectorized Implementation from orthogonal_label_generation

        # Params
        risk_budget = float(kwargs.get('risk_budget', 1.0))
        sl_mult = float(kwargs.get('sl_mult', 1.0))
        pt_mult = float(kwargs.get('pt_mult', 2.0))
        horizon = int(kwargs.get('horizon', 120))

        # Data
        price = df['close']
        vol = df['volatility_1d'].fillna(0.0)
        events = events_df.index

        # High/Low if available
        high = df.get('high')
        low = df.get('low')

        # Call Vectorized
        labels, weights, returns, mfe, mae, _ = compute_dominance_labels(
            price, events, vol,
            risk_budget=risk_budget, pt_mult=pt_mult, sl_mult=sl_mult, horizon=horizon,
            transaction_cost=self.transaction_cost,
            high=high, low=low
        )

        # Return matched format (labels, weights, returns, mfe, mae, exits)
        # Exits is dummy for now
        exits = pd.Series('', index=labels.index)

        return labels, weights, returns, mfe, mae, exits

    def generate_reports(self, *args, **kwargs):
        pass

    def _extract_tree_diagnostics(self, model_or_booster) -> Dict[str, float]:
        """
        Extract diagnostics from a trained LGBM booster or sklearn wrapper.
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
            dump = booster.dump_model()
            trees = dump.get('tree_info', [])
            depths = []
            for tree in trees:
                if 'tree_structure' not in tree: continue
                stack = [(tree['tree_structure'], 0)]
                while stack:
                    node, d = stack.pop()
                    if 'leaf_index' in node: depths.append(d)
                    else:
                        if 'left_child' in node: stack.append((node['left_child'], d + 1))
                        if 'right_child' in node: stack.append((node['right_child'], d + 1))

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
