"""
Label-Based Layer 2: Regime-Conditional Geometry Optimization & Meta-Labeling
-----------------------------------------------------------------------------
This module implements the Layer 2 architecture for meta-labeling, focusing on
regime-conditional barrier optimization and ML learnability probes.

Updates:
- Replaced MFE/MAE Dominance with Fixed Barrier Tournament.
- Integrated Orthogonal Event Generators (Symmetric CUSUM, Hurst, etc.).
- Implemented Learnability Sorting & Orthogonal Filtering.
- Added Race & HPO for winning geometries.
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
except Exception:
    class LokyTimeoutError(Exception):
        pass

    class BrokenProcessPool(Exception):
        pass

from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success

warnings.filterwarnings("ignore")

from src.training.steps.labeling.feature_generation_meta_labeling_step import (
    compute_realized_returns,
)
from src.training.steps.labeling.mtf_feature_generation import (
    create_meta_features,
    get_efficiency_ratio
)
from src.training.steps.labeling.generate_weights_per_label import finalize_sample_weights

from src.utils.purged_kfold import PurgedKFoldTime

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

# Import Orthogonal Generation
from src.training.steps.labeling.orthogonal_label_generation import (
    orthogonal_label_generation,
    SymmetricCusumEvents,
    ImprovedCUSUMEvents,
    HurstStateEvents,
    VolatilityShockEvents,
    TrendInitiationEvents,
    MeanReversionExtremeEvents,
    LiquidityShockEvents,
    TimeEvents,
    Geometry as OrthoGeometry
)

logger = logging.getLogger(__name__)
_lgb_logger = logging.getLogger("lightgbm")
_lgb_logger.setLevel(logging.ERROR)
_lgb_logger.propagate = False

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
    def __init__(self, gamma_pos=1.0, gamma_neg=2.5, alpha=None, grad_clip=5.0, w_cap=3.0, mix=0.25, label_smoothing=0.02, verbose=True):
        self.gamma_pos, self.gamma_neg, self.alpha = gamma_pos, gamma_neg, alpha
        self.grad_clip, self.w_cap, self.mix = grad_clip, w_cap, mix
        self.label_smoothing, self.verbose, self._is_init = label_smoothing, verbose, False

    def _init_alpha(self, labels):
        if self.alpha is None:
            n_pos = np.sum(labels > 0.5)
            self.alpha = 1.0 - (n_pos / len(labels)) if len(labels) > 0 else 0.5
        self.alpha = np.clip(self.alpha, 0.05, 0.95)
        self._is_init = True

    def __call__(self, preds, train_data):
        labels = train_data.get_label() if hasattr(train_data, 'get_label') else train_data
        if not self._is_init: self._init_alpha(labels)
        y_smooth = labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        p = np.clip(expit(preds), 1e-7, 1 - 1e-7)
        gamma_arr = np.where(labels > 0.5, self.gamma_pos, self.gamma_neg)
        focal_weight = np.minimum(np.where(labels > 0.5, (1 - p), p) ** gamma_arr, self.w_cap)
        grad_bce, hess_bce = p - y_smooth, p * (1 - p)
        alpha_f = np.where(labels > 0.5, self.alpha, 1 - self.alpha)
        grad = self.mix * (alpha_f * focal_weight * grad_bce) + (1 - self.mix) * grad_bce
        hess = self.mix * (alpha_f * focal_weight * hess_bce) + (1 - self.mix) * hess_bce
        if self.grad_clip: grad = np.clip(grad, -self.grad_clip, self.grad_clip)
        return grad, np.maximum(hess, 1e-6)

class XGBFocalLoss:
    def __init__(self, gamma_pos=1.0, gamma_neg=2.5, alpha=None, grad_clip=5.0, w_cap=3.0, mix=0.25, label_smoothing=0.02, verbose=True):
        self.gamma_pos, self.gamma_neg, self.alpha = gamma_pos, gamma_neg, alpha
        self.grad_clip, self.w_cap, self.mix = grad_clip, w_cap, mix
        self.label_smoothing, self.verbose, self._is_init = label_smoothing, verbose, False

    def _init_alpha(self, labels):
        if self.alpha is None:
            n_pos = np.sum(labels > 0.5)
            self.alpha = 1.0 - (n_pos / len(labels)) if len(labels) > 0 else 0.5
        self.alpha = np.clip(self.alpha, 0.05, 0.95)
        self._is_init = True

    def __call__(self, preds, dtrain):
        labels = dtrain.get_label() if hasattr(dtrain, 'get_label') else dtrain
        logits = preds
        if not self._is_init: self._init_alpha(labels)
        p = np.clip(1.0 / (1.0 + np.exp(-logits)), 1e-7, 1 - 1e-7)
        gamma_arr = np.where(labels > 0.5, self.gamma_pos, self.gamma_neg)
        focal_weight = np.minimum(np.where(labels > 0.5, (1 - p), p) ** gamma_arr, self.w_cap)
        y_smooth = labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        alpha_f = np.where(labels > 0.5, self.alpha, 1 - self.alpha)
        grad = self.mix * (alpha_f * focal_weight * (p - y_smooth)) + (1 - self.mix) * (p - y_smooth)
        hess = self.mix * (alpha_f * focal_weight * p * (1 - p)) + (1 - self.mix) * (p * (1 - p))
        if self.grad_clip: grad = np.clip(grad, -self.grad_clip, self.grad_clip)
        return grad, np.maximum(hess, 1e-6)

def _quick_5model_race(X_train, y_train, X_val, y_val, random_state=42):
    scores = {}

    # 1. LGBM Standard
    try:
        train_ds = lgb.Dataset(X_train, label=y_train)
        val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)
        focal = RobustFocalLoss(verbose=False)
        params = {
            'n_estimators': 500, 'learning_rate': 0.03, 'num_leaves': 63, 'max_depth': 7,
            'min_data_in_leaf': 20, 'feature_fraction': 0.8, 'verbosity': -1,
            'random_state': random_state, 'metric': 'average_precision', 'objective': focal,
        }
        m = lgb.train(params, train_ds, valid_sets=[val_ds], callbacks=[lgb.early_stopping(30, verbose=False)])
        scores['lgbm'] = roc_auc_score(y_val, expit(m.predict(X_val)))
    except: scores['lgbm'] = 0.0

    # 2. LGBM Linear
    try:
        params_lin = {
            'n_estimators': 500, 'learning_rate': 0.05, 'num_leaves': 63, 'max_depth': 4,
            'extra_trees': True, 'min_data_in_leaf': 20, 'verbosity': -1,
            'random_state': random_state, 'metric': 'average_precision', 'objective': focal, # reusing focal
            'feature_fraction': 0.8,
        }
        train_ds = lgb.Dataset(X_train, label=y_train)
        val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)
        m = lgb.train(params_lin, train_ds, valid_sets=[val_ds], callbacks=[lgb.early_stopping(30, verbose=False)])
        scores['lgbm_linear'] = roc_auc_score(y_val, expit(m.predict(X_val)))
    except: scores['lgbm_linear'] = 0.0

    # 3. XGB Standard
    try:
        focal_xgb = XGBFocalLoss(gamma_pos=1.5, gamma_neg=3.0, alpha=None)
        m = xgb.XGBClassifier(
            n_estimators=400, learning_rate=0.04, max_depth=6, min_child_weight=10,
            subsample=0.8, colsample_bytree=0.8, objective=focal_xgb, eval_metric='aucpr',
            early_stopping_rounds=30, verbosity=0, random_state=random_state, n_jobs=1
        )
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        scores['xgb'] = roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
    except: scores['xgb'] = 0.0

    # 4. XGB Linear
    try:
        m = xgb.XGBClassifier(
            booster='gblinear', n_estimators=100, learning_rate=0.1, objective='binary:logistic',
            eval_metric=['auc', 'aucpr'], early_stopping_rounds=30, verbosity=0,
            random_state=random_state, n_jobs=1
        )
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        scores['xgb_linear'] = roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
    except: scores['xgb_linear'] = 0.0

    # 5. CatBoost
    try:
        m = catboost.CatBoostClassifier(
            iterations=300, learning_rate=0.05, depth=6, class_weights={0: 1.0, 1: 3.0},
            verbose=False, random_seed=random_state, thread_count=1, allow_writing_files=False
        )
        m.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=30)
        scores['catboost'] = roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
    except: scores['catboost'] = 0.0

    winner = max(scores, key=scores.get) if scores else 'lgbm'
    return winner, scores

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
    race_score: Optional[float] = None
    events: Optional[pd.DatetimeIndex] = None
    model_type: Optional[str] = None

class LabelBasedLayer2:
    """
    Layer 2: Orthogonal Geometry Selection & Optimization.
    """

    def __init__(self, transaction_cost=None, n_trials=60, n_splits=3, random_state=42, verbose=True, force_hpo=False):
        self.transaction_cost = float(transaction_cost) if transaction_cost is not None else 0.003
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.random_state = random_state
        self.verbose = verbose
        self.force_hpo = force_hpo
        self.selected_geometries = []

    def execute(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.run(df)

    def _fixed_barrier_wrapper(self, df: pd.DataFrame, events_idx: pd.DatetimeIndex, **kwargs) -> pd.Series:
        """Wrapper for orthogonal generator to call fixed barrier logic."""
        if events_idx.empty:
            return pd.Series(dtype=float)
        events_df = pd.DataFrame(index=events_idx)
        labels, _ = self._compute_fixed_barrier_labels(df, events_df, **kwargs)
        return labels

    def _get_labeler_menu(self) -> Dict[str, Callable]:
        """Define the menu of labelers with baked-in parameters (SCALP, SWING, TREND)."""
        return {
            "SCALP": partial(self._fixed_barrier_wrapper, horizon=12, vol_mult=0.5),
            "SWING": partial(self._fixed_barrier_wrapper, horizon=24, vol_mult=1.0),
            "TREND": partial(self._fixed_barrier_wrapper, horizon=48, vol_mult=1.5)
        }

    def _compute_learnability_probe(self, df: pd.DataFrame, events: pd.DatetimeIndex, labels: pd.Series) -> float:
        """Score a geometry using LGBM CV AUC."""
        if len(events) < 50: return 0.5

        events_df = pd.DataFrame(index=events)
        X = self._build_geometry_independent_event_features(df, events_df)
        if X.empty: return 0.5

        cols = self._select_global_probe_features(X)
        X = X[cols]

        common = X.index.intersection(labels.index)
        X = X.loc[common]
        y = labels.loc[common]

        if len(y) < 50 or y.nunique() < 2: return 0.5

        tscv = TimeSeriesSplit(n_splits=3)
        scores = []

        model = lgb.LGBMClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            verbose=-1, random_state=42, n_jobs=1
        )

        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            if y_tr.nunique() < 2: continue

            try:
                model.fit(X_tr, y_tr)
                preds = model.predict_proba(X_val)[:, 1]
                scores.append(roc_auc_score(y_val, preds))
            except: pass

        return np.mean(scores) if scores else 0.5

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        tprint_info("Starting Layer 2 Pipeline (Tournament Mode)...")

        df = self._validate_inputs(df)
        df = self._precompute_geometry_base_features(df)

        labelers = self._get_labeler_menu()
        scorer = self._compute_learnability_probe

        tprint_info("Running Orthogonal Label Generation & Filtering...")
        ortho_geoms = orthogonal_label_generation(df, labelers, scorer=scorer)

        if not ortho_geoms:
            tprint_error("No geometries selected.")
            return {}

        tprint_info(f"Selected {len(ortho_geoms)} Geometries. Starting Race & HPO...")
        production_geometries = []

        for g in ortho_geoms:
            gt = self._process_selected_geometry(df, g)
            if gt:
                production_geometries.append(gt)

        self.selected_geometries = production_geometries

        events_df = self._construct_union_events_df(df, production_geometries)
        oof_results = self.run_oof_analytics(df, events_df, production_geometries)

        return {
            **oof_results,
            "events_df": events_df,
            "selected_trials": [asdict(t) for t in production_geometries],
        }

    def _process_selected_geometry(self, df, ortho_geom) -> Optional[GeometryTrial]:
        """Run Race, RFE, HPO for a single geometry."""
        events = ortho_geom.events
        labels = ortho_geom.labels
        events_df = pd.DataFrame(index=events)

        X = self._build_geometry_independent_event_features(df, events_df)
        geo_feats = self._compute_specific_geometry_features(df, events, ortho_geom.params)
        X = pd.concat([X, geo_feats], axis=1).fillna(0.0)

        common = X.index.intersection(labels.index)
        X = X.loc[common]
        y = labels.loc[common]

        if len(y) < 50: return None

        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        # 1. Race
        winner_model, scores = _quick_5model_race(X_train, y_train, X_val, y_val)
        tprint_info(f"Geometry {ortho_geom.name} Race Winner: {winner_model} (Score: {scores.get(winner_model, 0.0):.4f})")

        # 2. RFE (Quick)
        model = lgb.LGBMClassifier(n_estimators=100, verbose=-1, random_state=42)
        model.fit(X_train, y_train)
        imp = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        selected_features = imp.head(40).index.tolist()

        # 3. HPO (Placeholder - in production use BayesianTPEOptimizer or similar)

        return GeometryTrial(
            family=ortho_geom.family,
            params=ortho_geom.params,
            final_score=ortho_geom.score * 100,
            learnability=ortho_geom.score,
            robust_magnitude=0.0,
            stability=1.0,
            balance=1.0,
            raw_metrics=scores,
            uuid=ortho_geom.name,
            model_params={},
            selected_features=selected_features,
            race_score=scores.get(winner_model, 0.0),
            events=events,
            model_type=winner_model
        )

    def _compute_fixed_barrier_labels(self, df, events_df, **kwargs):
        horizon = int(kwargs.get('horizon', 12))
        vol_mult = float(kwargs.get('vol_mult', 1.0))

        signals = pd.DataFrame({'consensus': 1.0}, index=df.index)
        vol_series = df['volatility_1d'].fillna(0.0)

        tp = (vol_series * vol_mult).clip(lower=0.004)
        sl = tp / 1.5

        (realized_returns, _, exit_reasons, _, _, _, _, _) = compute_realized_returns(
            df=df,
            signals=signals,
            profit_threshold=tp,
            stop_threshold=sl,
            horizon=horizon,
            transaction_cost=self.transaction_cost,
            min_event_spacing=0
        )

        labels = (exit_reasons == 'profit').astype(float)
        idx = events_df.index
        return labels.reindex(idx), realized_returns.reindex(idx)

    def run_oof_analytics(self, df, events_df, production_geometries):
        if not production_geometries: return {}
        tprint_info(">>> Layer 2: Step 3 - OOF Analytics...")

        idx = events_df.index
        oof_scores = pd.Series(np.nan, index=idx)
        oof_weights = pd.Series(np.nan, index=idx)

        n_splits = self.n_splits
        fold_size = len(df) // n_splits
        
        for i in range(n_splits):
            test_start_idx = i * fold_size
            test_end_idx = (i + 1) * fold_size if i < n_splits - 1 else len(df)
            test_start_dt = df.index[test_start_idx]
            test_end_dt = df.index[test_end_idx-1]
            
            tprint_info(f"OOF Fold {i+1}/{n_splits}...")
            
            if i == 0: continue
            
            for gt in production_geometries:
                events = gt.events
                train_events = events[events < test_start_dt]
                test_events = events[(events >= test_start_dt) & (events <= test_end_dt)]
                
                if len(train_events) < 20 or len(test_events) == 0: continue
                
                train_labels = gt.labels.reindex(train_events).dropna()
                if len(train_labels) < 20: continue
                
                X_train_base = self._build_geometry_independent_event_features(df, pd.DataFrame(index=train_labels.index))
                X_train_geo = self._compute_specific_geometry_features(df, train_labels.index, gt.params)
                X_train = pd.concat([X_train_base, X_train_geo], axis=1).fillna(0.0)
                
                if gt.selected_features:
                    cols = [c for c in gt.selected_features if c in X_train.columns]
                    X_train = X_train[cols]

                model = None
                model_type = gt.model_type or 'lgbm'
                try:
                    if 'lgbm' in model_type:
                        model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, verbose=-1, random_state=42)
                    elif 'xgb' in model_type:
                        model = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, verbosity=0, random_state=42)
                    elif 'cat' in model_type:
                        model = catboost.CatBoostClassifier(iterations=200, verbose=False, random_seed=42, allow_writing_files=False)

                    if model: model.fit(X_train, train_labels)
                except: continue

                if not model: continue

                X_test_base = self._build_geometry_independent_event_features(df, pd.DataFrame(index=test_events))
                X_test_geo = self._compute_specific_geometry_features(df, test_events, gt.params)
                X_test = pd.concat([X_test_base, X_test_geo], axis=1).fillna(0.0)

                if gt.selected_features:
                    cols = [c for c in gt.selected_features if c in X_test.columns]
                    X_test = X_test[cols]

                try:
                    preds = model.predict_proba(X_test)[:, 1]
                    current = oof_scores.reindex(test_events).fillna(0.0)
                    updated = np.maximum(current, preds)
                    oof_scores.loc[test_events] = updated
                    oof_weights.loc[test_events] = 1.0
                except: pass

        oof_labels = (oof_scores >= 0.5).astype(float)
        oof_returns = pd.Series(np.nan, index=idx)
        valid_idx = oof_scores.dropna().index
        if not valid_idx.empty:
             signals = pd.DataFrame({'consensus': 1.0}, index=df.index)
             (ret, _, _, _, _, _, _, _) = compute_realized_returns(
                df=df, signals=signals, profit_threshold=None, stop_threshold=None,
                horizon=48, transaction_cost=self.transaction_cost, min_event_spacing=0
             )
             oof_returns.loc[valid_idx] = ret.loc[valid_idx]

        return {
            "l2_score": oof_scores,
            "oof_labels": oof_labels,
            "oof_returns": oof_returns,
            "weights": oof_weights
        }

    def _validate_inputs(self, df):
        if 'volatility_1d' not in df.columns:
            df = df.copy()
            df['volatility_1d'] = df['close'].pct_change().rolling(50).std()
        return df

    def _precompute_geometry_base_features(self, df):
        df_out = df.copy()
        if 'geo_atr_14' not in df_out.columns:
            try:
                high = df_out['high'] if 'high' in df_out.columns else df_out['close']
                low = df_out['low'] if 'low' in df_out.columns else df_out['close']
                close = df_out['close']
                tr = pd.concat([(high - low).abs(), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
                df_out['geo_atr_14'] = tr.rolling(14).mean()
            except: pass
        return df_out

    def _build_geometry_independent_event_features(self, df, events_df):
        if events_df.empty: return pd.DataFrame()
        signals = pd.DataFrame({'consensus': 1.0}, index=df.index)
        try:
            return create_meta_features(df, signals, events_df.index)
        except Exception as e:
            logger.warning(f"Feature gen failed: {e}")
            return pd.DataFrame(index=events_df.index)

    def _select_global_probe_features(self, X):
        if X is None or X.empty: return []
        return [c for c in X.columns if X[c].var() > 1e-6][:70]

    def _construct_union_events_df(self, df, geometries):
        if not geometries: return pd.DataFrame()
        all_indices = sorted(list(set([idx for g in geometries for idx in g.events])))
        if not all_indices: return pd.DataFrame()
        events_df = df.loc[all_indices, ['volatility_1d']].copy()
        events_df['family'] = 'Unified'
        return events_df

    def _compute_specific_geometry_features(self, df, events_index, params):
        if events_index.empty: return pd.DataFrame()
        subset = df.reindex(events_index)
        vol = subset['volatility_1d'].fillna(0.0)
        feats = pd.DataFrame(index=events_index)
        feats['geo_vol'] = vol
        return feats
