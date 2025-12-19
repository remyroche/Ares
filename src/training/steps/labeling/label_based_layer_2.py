"""
Label-Based Layer 2: Regime-Conditional Geometry Optimization & Meta-Labeling
-----------------------------------------------------------------------------
This module implements the Layer 2 architecture for meta-labeling, focusing on
regime-conditional barrier optimization and ML learnability probes.

It performs:
1. Event generation based on volatility-scaled returns.
2. Regime-conditional barrier family assignment.
3. Independent optimization of barrier geometries (TP/SL/Dominance/Horizon) per family using Optuna.
4. MFE/MAE Dominance Labeling: Label = 1 if (Exit==Profit/Trail) AND (MFE / MAE >= DominanceRatio).
5. Stability checks (Time-Flip) and Learnability probes.
6. Bagged output generation with family-level cap checks.
7. Enhanced LGBM training with Robust Focal Loss and Tree Variance calculation.
"""

import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from scipy.special import expit
from typing import Dict, List, Tuple, Optional, Any, Union
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

# Configure logging
logger = logging.getLogger(__name__)


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

        # Safety: enforce alpha in [0.5,0.95] to avoid extreme weighting
        self.alpha = min(max(self.alpha, 0.5), 0.95)

    def __call__(self, preds, train_data):
        """
        Args:
            preds: raw margins from LGBM
            train_data: lgb.Dataset
        Returns:
            grad, hess: gradient and hessian arrays
        """
        labels = train_data.get_label()
        p = expit(preds)  # convert raw score to probability
        p = np.clip(p, 1e-15, 1 - 1e-15)  # prevent log(0)

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

def _calculate_tree_variance(booster, X) -> np.ndarray:
    """
    Calculate the variance of predictions across all trees in the ensemble (Tree Variation).
    """
    if booster is None:
        return np.zeros(X.shape[0])

    try:
        # 1. Get leaf indices: (n_samples, n_trees)
        leaf_indices_raw = booster.predict(X, pred_leaf=True)
        
        # Ensure 2D (n_samples, n_trees)
        if leaf_indices_raw.ndim == 1:
            leaf_indices = leaf_indices_raw.reshape(-1, 1)
        else:
            leaf_indices = leaf_indices_raw

        # 2. Parse model to get leaf values
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

        # 3. Vectorized lookup
        n_trees_pred = leaf_indices.shape[1]
        limit_trees = min(n_trees, n_trees_pred)
        tree_indices = np.arange(limit_trees)
        subset_indices = leaf_indices[:, :limit_trees]
        subset_indices = np.clip(subset_indices, 0, max_leaf_idx)

        collected_values = leaf_values_lookup[tree_indices, subset_indices]

        # 4. Calculate Variance
        variance = np.nanvar(collected_values, axis=1)

        return variance

    except Exception as e:
        logger.warning(f"Failed to calculate tree variance: {e}")
        return np.zeros(X.shape[0])

@dataclass
class GeometryTrial:
    family: str
    params: Dict[str, Any]  # tp_mult, dominance_ratio, sl_mult, horizon
    final_score: float
    learnability: float
    robust_magnitude: float
    stability: float
    balance: float
    raw_metrics: Dict[str, float]
    uuid: str

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
        verbose: bool = True
    ):
        """
        Initialize Layer 2.

        Args:
            transaction_cost: Trading cost (slippage + fees) per side.
            n_trials: Number of Optuna trials per barrier family.
            n_splits: Number of TimeSeriesSplit folds for ML probes.
            random_state: Seed for reproducibility.
            verbose: Logging verbosity.
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

        # Internal state
        self.selected_geometries: List[GeometryTrial] = []
        self.family_weights: Dict[str, float] = {}

        self._labels_cache: Dict[Any, Tuple[pd.Series, pd.Series, pd.Series, pd.Series]] = {}
        self._signals_cache: Dict[Any, pd.DataFrame] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._global_probe_features: List[str] = []
        self._current_param_bounds: Dict[str, Dict[str, Any]] = {}
        self._primary_signals: Optional[pd.DataFrame] = None

        # Suppress Optuna logging if not verbose
        if not self.verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

    def execute(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._current_config = dict(config or {})
        return self.run(df)

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute the Layer 2 pipeline with OOF generation.

        This method performs:
        1. Full Optimization (to get production geometries).
        2. K-Fold OOF Optimization (to get unbiased analytics/artifacts).

        Args:
            df: Input DataFrame containing 'close', 'vwap', 'volatility_1d',
                'trend_regime', 'vol_regime', etc.

        Returns:
            Dict containing:
            - 'oof_labels': OOF Weighted Consensus Labels (Series)
            - 'oof_returns': OOF Weighted Consensus Returns (Series)
            - 'weights': OOF Weights (Series)
            - 'individual_geometries': OOF predictions per geometry channel (Dict[str, Series])
            - 'individual_variances': OOF variance per geometry channel (Dict[str, Series])
            - 'events_df': Events DataFrame
            - 'selected_trials': List[Dict] (Production geometries from full fit)
        """
        logger.info("Starting Layer 2 Pipeline...")

        self._labels_cache = {}
        self._signals_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._global_probe_features = []
        self._current_param_bounds = {}
        self._primary_signals = None

        # Step 0: Preparation
        df = self._validate_inputs(df)
        events_df = self._generate_events(df)

        if events_df.empty:
            logger.warning("No events generated in Layer 2. Skipping.")
            return {}

        try:
            X_probe_events = self._build_geometry_independent_event_features(df, events_df)
            self._global_probe_features = self._select_global_probe_features(X_probe_events)
        except Exception:
            self._global_probe_features = []

        # Persist selected features
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
            if self._global_probe_features:
                pd.Series(self._global_probe_features, name='feature').to_csv(
                    outcomes_dir / f"layer2_selected_features_{symbol}_{timeframe}_{ts}.csv",
                    index=False
                )
        except Exception as e:
            logger.warning(f"Failed to persist layer2 selected features: {e}")

        # ---------------------------------------------------------------------
        # Part A: Full Optimization (Production Artifacts)
        # ---------------------------------------------------------------------
        logger.info(">>> Layer 2: Running Full Optimization (Production)...")
        full_results = self._optimize_families(df, events_df)

        try:
            full_counts = {str(k): int(len(v)) for k, v in (full_results or {}).items()}
            logger.info(f"Layer2 Full Optimization: extracted_trials_per_family={full_counts}")
        except Exception:
            pass

        # Strictly select best geometries. If none pass, we fail (no fallback).
        production_geometries = self._select_best_geometries(df, events_df, full_results, require_passed=True)

        try:
            by_fam: Dict[str, int] = {}
            for g in list(production_geometries or []):
                try:
                    by_fam[str(getattr(g, 'family', ''))] = by_fam.get(str(getattr(g, 'family', '')), 0) + 1
                except Exception:
                    continue
            logger.info(
                f"Layer2 Production Geometries: n={int(len(production_geometries or []))}, by_family={by_fam}"
            )
        except Exception:
            pass

        # FAST-FAIL: If no production geometries passed, the pipeline cannot continue
        if not production_geometries:
            logger.error(
                "Layer2 CRITICAL: Zero production geometries passed all gates! "
                "Pipeline cannot continue. Consider relaxing gates via config: "
                "layer2_probe_auc_threshold, layer2_stability_threshold, "
                "layer2_min_pos_rate, layer2_max_pos_rate."
            )
            raise ValueError(
                "Layer2 failed: No production geometries passed validation gates. "
                "Strict quality control prevented fallback."
            )

        # Store for reference
        self.selected_geometries = production_geometries

        try:
            self._production_selected_features = []
        except Exception:
            pass

        try:
            cfg_prod_fs = getattr(self, "_current_config", {})
            if not isinstance(cfg_prod_fs, dict):
                cfg_prod_fs = {}
        except Exception:
            cfg_prod_fs = {}

        try:
            enable_prod_fs = bool(cfg_prod_fs.get('layer2_production_supervised_feature_selection_enabled', True))
        except Exception:
            enable_prod_fs = True

        if enable_prod_fs:
            try:
                X_events_full = self._build_geometry_independent_event_features(df, events_df)
                y_fs_prod = self._aggregate_geometry_labels_for_feature_selection(df, events_df, production_geometries)
                w_l1_prod = self._get_target_sample_weight_for_events(df, events_df)
                prod_feats = self._select_supervised_features_for_events(X_events_full, y_fs_prod, w_l1_prod)
                if prod_feats:
                    self._production_selected_features = list(prod_feats)
                    try:
                        pd.Series(self._production_selected_features, name='feature').to_csv(
                            outcomes_dir / f"layer2_selected_features_supervised_{symbol}_{timeframe}_{ts}.csv",
                            index=False,
                        )
                    except Exception:
                        pass
            except Exception:
                pass

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

        # Derive families dynamically to avoid hardcoding
        families = ['Trend Continuation', 'Momentum', 'Mean Reversion']
        max_rank = 4
        oof_geo_preds = {}
        oof_geo_vars = {} # Store variances
        for fam in families:
            for r in range(max_rank):
                key = f"{fam}_Rank{r}"
                oof_geo_preds[key] = pd.Series(np.nan, index=indices)
                oof_geo_vars[key] = pd.Series(np.nan, index=indices)

        try:
            cfg_oof = getattr(self, "_current_config", {})
            if not isinstance(cfg_oof, dict):
                cfg_oof = {}
        except Exception:
            cfg_oof = {}

        try:
            n_oof_splits = int(cfg_oof.get("layer2_oof_splits", 5))
        except Exception:
            n_oof_splits = 5
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
            train_end = int(max(0, int(val_start) - int(purge_bars)))
            train_idx = np.arange(0, int(train_end))

            logger.info(f"   > Processing Fold {fold_idx}/{int(len(folds))}...")

            try:
                t0 = str(df.index[int(val_start)]) if int(val_start) < len(df.index) else ""
                t1 = str(df.index[int(val_stop - 1)]) if int(val_stop - 1) < len(df.index) else ""
                te = str(df.index[int(train_end - 1)]) if int(train_end - 1) >= 0 and int(train_end - 1) < len(df.index) else ""
                logger.info(
                    f"Layer2 OOF Fold {fold_idx}: walkforward train_end={train_end}, val_start={val_start}, val_stop={val_stop}, "
                    f"purge_bars={purge_bars}, train_end_time={te}, test_start_time={t0}, test_end_time={t1}"
                )
            except Exception:
                pass

            # Create Train Slice (strictly past only)
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

            # STRICT SELECTION IN OOF TOO
            fold_geometries = self._select_best_geometries(df_train, events_train, fold_results, require_passed=True)
            if not fold_geometries:
                # If nothing passed in this fold, we skip the fold. Layer 3 will see NaNs which is correct.
                logger.warning(f"Fold {fold_idx}: No geometries passed strict selection. Skipping.")
                continue

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
                X_train_events_full = self._build_geometry_independent_event_features(df_train, events_train)
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

            if use_supervised_fs and X_train_events_full is not None and not getattr(X_train_events_full, 'empty', True):
                try:
                    y_fs = self._aggregate_geometry_labels_for_feature_selection(df_train, events_train, standardized_geos)
                    w_l1 = self._get_target_sample_weight_for_events(df_train, events_train)
                    fs_feats = self._select_supervised_features_for_events(X_train_events_full, y_fs, w_l1)
                    if fs_feats:
                        fold_probe_features = fs_feats
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
                        geometries=standardized_geos
                    )
                except Exception:
                    trained_models = None

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

                try:
                    lbl = fold_output.get('oof_labels')
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

                # Assign individual geometry preds and variances
                for uuid, series in fold_output['individual_geometries'].items():
                    if uuid in oof_geo_preds:
                        oof_geo_preds[uuid].loc[target_idx] = series.reindex(target_idx)

                for uuid, series in fold_output['individual_variances'].items():
                    if uuid in oof_geo_vars:
                        oof_geo_vars[uuid].loc[target_idx] = series.reindex(target_idx)

        # ---------------------------------------------------------------------
        # Final Packaging
        # ---------------------------------------------------------------------
        final_geo_preds = {k: v for k, v in oof_geo_preds.items() if v.notna().any()}
        final_geo_vars = {k: v for k, v in oof_geo_vars.items() if v.notna().any()}

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

        try:
            extracted_trials_counts = {str(k): int(len(v)) for k, v in (full_results or {}).items()}
        except Exception:
            extracted_trials_counts = {}

        try:
            prod_by_family: Dict[str, int] = {}
            for g in list(production_geometries or []):
                try:
                    prod_by_family[str(getattr(g, "family", ""))] = prod_by_family.get(str(getattr(g, "family", "")), 0) + 1
                except Exception:
                    continue
        except Exception:
            prod_by_family = {}

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
                f"- production_geometries_by_family: {prod_by_family}\n",
                f"- production_geometries_n: {int(len(production_geometries or []))}\n",
                f"- oof_labeled_events: {oof_labeled}\n",
                f"- oof_nonzero_weight_events: {oof_weight_nonzero}\n",
                f"- oof_geometry_channels: {n_geo_channels}\n",
            ]
            md_path.write_text("".join(lines))
        except Exception:
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
            for fam, cnt in extracted_trials_counts.items():
                summary_row[f"extracted_trials_{fam}"] = int(cnt)
            for fam, cnt in prod_by_family.items():
                summary_row[f"production_geos_{fam}"] = int(cnt)
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
                    tp_mult = None
                    dominance_ratio = None
                    sl_mult = None
                    horizon = None
                    if isinstance(params, dict):
                        tp_mult = params.get("tp_mult")
                        dominance_ratio = params.get("dominance_ratio")
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
                        if tp_mult is not None and horizon is not None:
                            _lbl, _ret, _, _ = self._compute_dominance_labels(
                                df=df,
                                events_df=fam_events,
                                tp_mult=float(tp_mult),
                                dominance_ratio=float(dominance_ratio) if dominance_ratio is not None else 1.0,
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

        logger.info("Layer 2 Pipeline Complete.")

        return {
            "oof_labels": oof_scores,
            "oof_returns": oof_returns,
            "weights": oof_weights,
            "l2_score": l2_score,
            "l2_label": l2_label,
            "l2_confidence": l2_confidence,
            "individual_geometries": final_geo_preds,
            "individual_variances": final_geo_vars,
            "events_df": events_df,
            "selected_trials": [asdict(t) for t in production_geometries],
            "production_selected_features": list(getattr(self, '_production_selected_features', []) or []),
        }

    def _extract_trials_from_study(self, study: optuna.Study) -> List[GeometryTrial]:
        """Extract GeometryTrial objects from study user attrs."""
        trials = []
        for t in study.trials:
            if t.state == optuna.trial.TrialState.COMPLETE:
                # We saved the object in user_attrs
                g_obj = t.user_attrs.get("geometry_object")
                if g_obj:
                    trials.append(g_obj)
        return trials

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
        tp_mult: float,
        dominance_ratio: float,
        horizon: int,
        family: str,
        events_shift: int = 0,
        sl_mult: Optional[float] = None,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Compute TP/SL(+optional trailing) exit-model labels and related metrics.
        Label = 1 if (Exit == Profit OR Exit == Trailing) AND (MFE / MAE >= dominance_ratio).

        Args:
            df: Market data
            events_df: Events to label
            tp_mult: Take-Profit multiplier (of volatility)
            dominance_ratio: MFE/MAE threshold
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
            float(round(float(tp_mult), 8)),
            float(round(float(dominance_ratio), 8)),
            float(round(float(sl_mult_eff), 8)),
            int(horizon),
            int(events_shift),
            float(self.transaction_cost),
            str(direction_mode),
            float(trail_mult) if trail_mult is not None else None,
            int(max(0, int(getattr(self, '_current_config', {}).get('layer2_min_event_spacing', 4) if isinstance(getattr(self, '_current_config', {}), dict) else 4))),
            "tpsl_full_dominance"
        )
        cached = self._labels_cache.get(cache_key)
        if cached is not None:
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
        vol_series = vol_series.fillna(method='ffill').fillna(method='bfill')
        vol_series = vol_series.clip(lower=1e-8)

        profit_thr = float(tp_mult) * vol_series
        stop_thr = float(sl_mult_eff) * vol_series

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
            min_event_spacing=int(max(0, int(getattr(self, '_current_config', {}).get('layer2_min_event_spacing', 4)))),
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

        # Labels: Profit (or trailing) Exit AND MFE/MAE >= dominance_ratio
        is_profit_exit = subset_exit.astype(str).isin(['profit', 'trailing'])

        # Calculate Dominance (MFE / MAE)
        mae_safe = subset_mae.replace(0.0, 1e-9).abs()
        dominance = subset_mfe.abs() / mae_safe

        is_dominant = (dominance >= float(dominance_ratio))

        binary_labels = (is_profit_exit & is_dominant).astype(float)
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
        kappa = kwargs.get('kappa')

        if tp_mult is None and kappa is not None:
            tp_mult = float(kappa)
        if tp_mult is None:
            tp_mult = 2.0
            
        # Use kappa as dominance ratio if present, else default 1.0 or heuristic
        dominance_ratio = float(kappa) if kappa is not None else 1.0

        lbl, ret, _, _ = self._compute_dominance_labels(
            df=df,
            events_df=events_df,
            tp_mult=float(tp_mult),
            dominance_ratio=float(dominance_ratio),
            horizon=int(horizon),
            family=family,
            sl_mult=sl_mult
        )
        return lbl, ret

    def _optimization_objective(
        self,
        trial: optuna.Trial,
        df: pd.DataFrame,
        family: str,
        family_events: pd.DataFrame,
        X_events: pd.DataFrame,
        probe_features: List[str],
        target_sample_weight_events: Optional[pd.Series]
    ) -> float:
        """Extracted optimization objective to avoid nested function re-definition."""
        # Parameter Suggestion
        # tp_mult (1.0 - 6.0)
        tp_mult = trial.suggest_float('tp_mult', 1.0, 6.0)
        # dominance_ratio (1.0 - 4.0)
        dominance_ratio = trial.suggest_float('dominance_ratio', 1.0, 4.0)
        # sl_mult (0.5 - 3.0)
        sl_mult = trial.suggest_float('sl_mult', 0.5, 3.0)
        # horizon (10 - 50)
        horizon = trial.suggest_int('horizon', 10, 50)

        # Enforce Constraint: tp_mult >= 1.5 * sl_mult
        min_tp = 1.5 * sl_mult
        if tp_mult < min_tp:
            # Adjust tp_mult to meet constraint rather than reject trial
            # This allows Optuna to learn the constraint region or just work with valid params
            tp_mult = min_tp
            if tp_mult > 6.0:
                # If adjustment pushes it out of bounds, clip and reduce SL instead
                tp_mult = 6.0
                sl_mult = tp_mult / 1.5
                # if sl_mult < 0.5... strict constraint might be tricky.
                # but let's assume valid regions exist.

        # Compute labels
        labels, returns, _, _ = self._compute_dominance_labels(
            df,
            family_events,
            tp_mult=tp_mult,
            dominance_ratio=dominance_ratio,
            horizon=horizon,
            family=family,
            sl_mult=sl_mult
        )

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

        # --- OPTIMIZATION: Tighter Pre-Filters ---
        # If the geometry is fundamentally poor in terms of base statistics, don't waste time on probes.
        try:
             min_rate = float(getattr(self, '_current_config', {}).get('layer2_min_pos_rate', 0.08))
             max_rate = float(getattr(self, '_current_config', {}).get('layer2_max_pos_rate', 0.65))
        except Exception:
             min_rate, max_rate = 0.08, 0.65

        if pos_rate < min_rate or pos_rate > max_rate: 
             logger.info(f"[GATE_REJECT] trial={trial.number}, family={family}, reason=pos_rate_limit, pos_rate={pos_rate:.3f}, range=[{min_rate}, {max_rate}]")
             t_obj = GeometryTrial(
                family=family,
                params={'tp_mult': tp_mult, 'dominance_ratio': dominance_ratio, 'sl_mult': sl_mult, 'horizon': horizon},
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

        try:
            profit_mode = str(getattr(self, '_current_config', {}).get('layer2_profitability_mode', 'trade_mean'))
        except Exception:
            profit_mode = 'trade_mean'
        profit_mode = str(profit_mode).strip().lower()

        try:
            min_pos_trades = int(getattr(self, '_current_config', {}).get('layer2_min_positive_trades', 15))
        except Exception:
            min_pos_trades = 15

        try:
            min_trade_ret = float(getattr(self, '_current_config', {}).get('layer2_min_mean_trade_return', self.transaction_cost))
        except Exception:
            min_trade_ret = float(self.transaction_cost)

        is_profitable = True
        if profit_mode in {'trade', 'trade_mean', 'conditional'}:
            if int(pos_count) < int(min_pos_trades):
                is_profitable = False
            elif (not np.isfinite(mean_trade_ret)) or (float(mean_trade_ret) < float(min_trade_ret)):
                is_profitable = False
        else:
            if float(mean_ret) < float(self.transaction_cost):
                is_profitable = False

        if not is_profitable:
             logger.info(f"[GATE_REJECT] trial={trial.number}, family={family}, reason=unprofitable, mean_ret={mean_ret:.5f}")
             t_obj = GeometryTrial(
                family=family,
                params={'tp_mult': tp_mult, 'dominance_ratio': dominance_ratio, 'sl_mult': sl_mult, 'horizon': horizon},
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
            stability_every = int(getattr(self, '_current_config', {}).get('layer2_stability_every_n_trials', 1))
        except Exception:
            stability_every = 1
        if stability_every <= 0:
            stability_every = 1

        try:
            stability_sample_frac = float(getattr(self, '_current_config', {}).get('layer2_stability_sample_frac', 1.0))
        except Exception:
            stability_sample_frac = 1.0
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
                {'tp_mult': tp_mult, 'dominance_ratio': dominance_ratio, 'sl_mult': sl_mult, 'horizon': horizon},
                0.0,
                family,
            )
            if not is_stable:
                 logger.info(f"[GATE_REJECT] trial={trial.number}, family={family}, reason=unstable")
                 t_obj = GeometryTrial(
                    family=family,
                    params={'tp_mult': tp_mult, 'dominance_ratio': dominance_ratio, 'sl_mult': sl_mult, 'horizon': horizon},
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
        # Configurable frequency to reduce compute.
        try:
            perturb_every = int(getattr(self, '_current_config', {}).get('layer2_perturb_every_n_trials', 1))
        except Exception:
            perturb_every = 1
        if perturb_every <= 0:
            perturb_every = 1

        if (trial.number % int(perturb_every)) == 0:
            # Small perturbation to params
            perturb_labels_k, _, _, _ = self._compute_dominance_labels(df, fam_events_for_checks, tp_mult * 1.05, dominance_ratio, horizon, family, sl_mult=sl_mult)
            perturb_labels_sl, _, _, _ = self._compute_dominance_labels(df, fam_events_for_checks, tp_mult, dominance_ratio, horizon, family, sl_mult=sl_mult * 1.05)
            perturb_labels_h, _, _, _ = self._compute_dominance_labels(df, fam_events_for_checks, tp_mult, dominance_ratio, int(horizon * 1.05), family, sl_mult=sl_mult)

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
                params={'tp_mult': tp_mult, 'dominance_ratio': dominance_ratio, 'sl_mult': sl_mult, 'horizon': horizon},
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
            params={'tp_mult': tp_mult, 'dominance_ratio': dominance_ratio, 'sl_mult': sl_mult, 'horizon': horizon},
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
            try:
                score = float(getattr(t, 'final_score', -1.0))
            except Exception:
                return False
            if (not np.isfinite(score)) or score <= 0.0:
                return False
            rm = getattr(t, 'raw_metrics', None)
            return bool(isinstance(rm, dict) and bool(rm.get('passed', False)))

        # 3.2 Discard poorer barrier families
        family_medians = {}
        for fam, trials in family_results.items():
            trials_all = list(trials or [])
            # Filter logic: if require_passed is True, filter.
            # If False (OOF), we also want to filter by passed to maintain consistency/quality if possible.
            # However, if nothing passed in OOF, we might want to return nothing (as per strict request).
            trials_use = [t for t in trials_all if _is_passed_trial(t)]

            # Note: Removed the fallback "if not trials_use then trials_use = trials_all"
            # as we want strict acceptance.

            if not trials_use:
                continue

            trials_sorted = sorted(trials_use, key=lambda x: float(getattr(x, 'final_score', -1.0)), reverse=True)
            top_k = trials_sorted[:10]
            if not top_k:
                continue
            median_score = np.median([t.final_score for t in top_k])
            family_medians[fam] = median_score

        sorted_families = sorted(family_medians.items(), key=lambda x: x[1], reverse=True)
        keep_families = [f[0] for f in sorted_families[:3]]
        if not keep_families:
            # If no family has any passed trial, we return empty list
            return []

        selected = []

        # 3.3 Keep diverse geometries per family
        for fam in keep_families:
            trials_all = list(family_results.get(fam) or [])

            # Strict filtering
            trials = [t for t in trials_all if _is_passed_trial(t)]
            if not trials:
                continue

            trials.sort(key=lambda x: float(getattr(x, 'final_score', -1.0)), reverse=True)
            n_top = max(2, int(len(trials) * 0.2))
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
                                lbls, _, _, _ = self._compute_dominance_labels(df, fam_events_local, family=fam, **cand.params)
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
            tp_vals = [float(t.params.get('tp_mult', 0.0)) for t in top_tier]
            dom_vals = [float(t.params.get('dominance_ratio', 0.0)) for t in top_tier]
            sl_vals = [float(t.params.get('sl_mult', 1.0)) for t in top_tier]
            h_vals = [float(t.params.get('horizon', 0.0)) for t in top_tier]

            # Avoid empty ranges
            def _get_range(vals):
                if not vals: return 1.0
                return max(vals) - min(vals) + 1e-6

            tp_range = _get_range(tp_vals)
            dom_range = _get_range(dom_vals)
            sl_range = _get_range(sl_vals)
            h_range = _get_range(h_vals)

            tp_min = min(tp_vals) if tp_vals else 0.0
            dom_min = min(dom_vals) if dom_vals else 0.0
            sl_min = min(sl_vals) if sl_vals else 0.0
            h_min = min(h_vals) if h_vals else 0.0

            def get_norm_vec(t):
                return np.array([
                    (float(t.params.get('tp_mult', 0.0)) - tp_min) / tp_range,
                    (float(t.params.get('dominance_ratio', 0.0)) - dom_min) / dom_range,
                    (float(t.params.get('sl_mult', 1.0)) - sl_min) / sl_range,
                    (float(t.params.get('horizon', 0.0)) - h_min) / h_range,
                ])

            # Outcome-space diversification: avoid selecting highly correlated return series
            try:
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
                    _lbl, _ret, _, _ = self._compute_dominance_labels(df, fam_events_local, family=fam, **t_obj.params)
                    s = pd.to_numeric(_ret, errors='coerce').astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                except Exception:
                    s = pd.Series(0.0, index=events_df[events_df['family'] == fam].index)
                ret_cache[key] = s
                return s

            # Pick best first (stable)
            for cand in top_tier:
                fam_events = events_df[events_df['family'] == fam]
                # Already checked stability in optimization loop for passed trials
                if self._check_stability(df, fam_events, cand.params, cand.final_score, fam):
                    fam_selected.append(cand)
                    break

            if not fam_selected:
                # No stable geometry in top tier? Strict check => skip family
                continue

            # Pick others maximizing normalized distance
            candidate_pool = [t for t in top_tier if t not in fam_selected]

            while len(fam_selected) < 4 and candidate_pool:
                best_cand = None
                max_dist = -1.0

                for cand in candidate_pool:
                    dists = [np.linalg.norm(get_norm_vec(cand) - get_norm_vec(s)) for s in fam_selected]
                    min_d = min(dists)

                    if min_d > max_dist:
                        max_dist = min_d
                        best_cand = cand

                if best_cand:
                    # Stability check
                    fam_events = events_df[events_df['family'] == fam]
                    # Correlation filter vs already-selected
                    try:
                        ok_corr = True
                        cand_ret = _get_ret_series(best_cand)
                        for s_obj in fam_selected:
                            s_ret = _get_ret_series(s_obj)
                            c = float(pd.Series(cand_ret).corr(pd.Series(s_ret)))
                            if np.isfinite(c) and abs(c) >= float(corr_thr):
                                ok_corr = False
                                break
                        if ok_corr and self._check_stability(df, fam_events, best_cand.params, best_cand.final_score, fam):
                            fam_selected.append(best_cand)
                    except Exception:
                        if self._check_stability(df, fam_events, best_cand.params, best_cand.final_score, fam):
                            fam_selected.append(best_cand)

                    candidate_pool.remove(best_cand)
                else:
                    break

            selected.extend(fam_selected)

        # -----------------------------------------------------------------
        # Final global (cross-family) diversification pass
        # -----------------------------------------------------------------
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

        if global_div_enabled and int(len(selected)) > 1:
            try:
                global_corr_thr = float(cfg_global.get('layer2_global_outcome_corr_threshold', cfg_global.get('layer2_outcome_corr_threshold', 0.95)))
            except Exception:
                global_corr_thr = 0.95

            try:
                max_keep = int(cfg_global.get('layer2_global_max_geometries', 0))
            except Exception:
                max_keep = 0

            try:
                all_events_idx = pd.Index(events_df.index)
            except Exception:
                all_events_idx = events_df.index

            ret_cache_global: Dict[str, pd.Series] = {}

            def _series_corr(a: pd.Series, b: pd.Series) -> float:
                try:
                    a_v = pd.to_numeric(a, errors='coerce').astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                    b_v = pd.to_numeric(b, errors='coerce').astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                    if a_v.std() < 1e-12 and b_v.std() < 1e-12:
                        return 1.0 if bool(np.allclose(a_v.values, b_v.values)) else 0.0
                    c = float(a_v.corr(b_v))
                    return float(c) if np.isfinite(c) else 0.0
                except Exception:
                    return 0.0

            def _get_ret_series_global(t_obj: GeometryTrial) -> pd.Series:
                key = str(getattr(t_obj, 'uuid', ''))
                if key in ret_cache_global:
                    return ret_cache_global[key]

                try:
                    fam_local = str(getattr(t_obj, 'family', ''))
                except Exception:
                    fam_local = ''

                try:
                    fam_events_local = events_df[events_df['family'] == fam_local]
                except Exception:
                    fam_events_local = events_df

                try:
                    _lbl, _ret, _, _ = self._compute_dominance_labels(df, fam_events_local, family=fam_local, **t_obj.params)
                    s_evt = pd.to_numeric(_ret, errors='coerce').astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                except Exception:
                    s_evt = pd.Series(0.0, index=fam_events_local.index)

                try:
                    s_glob = pd.Series(0.0, index=all_events_idx, dtype=float)
                    s_glob.loc[s_evt.index.intersection(all_events_idx)] = s_evt.reindex(all_events_idx).fillna(0.0).loc[s_evt.index.intersection(all_events_idx)].values
                except Exception:
                    try:
                        s_glob = s_evt.reindex(all_events_idx).fillna(0.0)
                    except Exception:
                        s_glob = pd.Series(0.0, index=all_events_idx, dtype=float)

                ret_cache_global[key] = s_glob
                return s_glob

            def _quality_key(t_obj: GeometryTrial) -> Tuple[float, float, float, float]:
                rm = getattr(t_obj, 'raw_metrics', None)
                rm = rm if isinstance(rm, dict) else {}
                try:
                    auc_full = float(rm.get('auc_full')) if rm.get('auc_full') is not None and np.isfinite(float(rm.get('auc_full'))) else float('-inf')
                except Exception:
                    auc_full = float('-inf')
                try:
                    auc_light = float(rm.get('auc_lgbm_light')) if rm.get('auc_lgbm_light') is not None and np.isfinite(float(rm.get('auc_lgbm_light'))) else float('-inf')
                except Exception:
                    auc_light = float('-inf')
                try:
                    auc_lin = float(rm.get('auc_linear')) if rm.get('auc_linear') is not None and np.isfinite(float(rm.get('auc_linear'))) else float('-inf')
                except Exception:
                    auc_lin = float('-inf')
                try:
                    score = float(getattr(t_obj, 'final_score', -1.0))
                except Exception:
                    score = -1.0
                return (auc_full, auc_light, auc_lin, score)

            ordered = sorted(list(selected), key=_quality_key, reverse=True)

            kept: List[GeometryTrial] = []
            for cand in ordered:
                if max_keep > 0 and len(kept) >= max_keep:
                    break

                cand_ret = _get_ret_series_global(cand)
                ok = True
                for k in kept:
                    k_ret = _get_ret_series_global(k)
                    c = _series_corr(cand_ret, k_ret)
                    if np.isfinite(c) and abs(float(c)) >= float(global_corr_thr):
                        ok = False
                        break
                if ok:
                    kept.append(cand)

            try:
                if len(kept) < len(selected):
                    logger.info(
                        f"Layer2 Global Diversification: kept={int(len(kept))}/{int(len(selected))}, corr_thr={float(global_corr_thr):.3f}"
                    )
            except Exception:
                pass

            selected = kept

        return selected
