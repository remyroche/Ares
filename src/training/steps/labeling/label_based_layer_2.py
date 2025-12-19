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

        if verbose:
            try:
                pos_frac = np.mean(train_labels)
            except Exception:
                pos_frac = 0.0
            # logger.info(f"[Focal Loss] gamma={self.gamma}, alpha={self.alpha:.4f} (Pos fraction: {pos_frac:.2%})")

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

    1. Get leaf indices for each sample.
    2. Retrieve leaf values from the model dump.
    3. Look up values for indices.
    4. Compute variance across trees for each sample.
    """
    if booster is None:
        return np.zeros(X.shape[0])

    try:
        # 1. Get leaf indices: (n_samples, n_trees)
        leaf_indices_raw = booster.predict(X, pred_leaf=True)
        
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
        model_dump = booster.dump_model()
        trees = model_dump['tree_info']

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
    params: Dict[str, Any]  # Kappa, Horizon
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

        production_geometries = self._select_best_geometries(df, events_df, full_results, require_passed=True)
        if not production_geometries:
            try:
                cfg_prod = getattr(self, "_current_config", {})
                if not isinstance(cfg_prod, dict):
                    cfg_prod = {}
            except Exception:
                cfg_prod = {}
            try:
                fallback_enabled = bool(cfg_prod.get('layer2_production_fallback_enabled', True))
            except Exception:
                fallback_enabled = True
            if fallback_enabled:
                production_geometries = self._select_best_geometries(df, events_df, full_results, require_passed=False)

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
                "Check logs for [GATE_REJECT] messages. Gates: "
                "pos_rate 10-40%, mean_ret > transaction_cost, stability >=85%, probe AUC >=0.52"
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

            fold_geometries = self._select_best_geometries(df_train, events_train, fold_results, require_passed=False)
            if not fold_geometries:
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
            float(round(float(kappa), 8)),
            float(round(float(sl_mult_eff), 8)),
            int(horizon),
            int(events_shift),
            float(self.transaction_cost),
            str(direction_mode),
            float(trail_mult) if trail_mult is not None else None,
            int(max(0, int(getattr(self, '_current_config', {}).get('layer2_min_event_spacing', 4) if isinstance(getattr(self, '_current_config', {}), dict) else 4))),
            "tpsl_full"
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

        profit_thr = float(kappa) * vol_series
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
            auc_thr = float(getattr(self, '_current_config', {}).get('layer2_probe_auc_threshold', 0.515))
        except Exception:
            auc_thr = 0.515
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

        X_events = self._build_geometry_independent_event_features(df, events_df)
        try:
            probe_features = self._select_global_probe_features(X_events)
        except Exception:
            probe_features = []
        target_sample_weight_events = self._get_target_sample_weight_for_events(df, events_df)

        for family in unique_families:
            logger.info(f"Optimizing family: {family}")

            family_mask = events_df['family'] == family
            family_events = events_df[family_mask]

            if len(family_events) < 50:
                logger.warning(f"Not enough events for family {family} ({len(family_events)}). Skipping.")
                continue

            try:
                logger.info(
                    f"Layer2 Optimize family={family}: n_events={int(len(family_events))}, n_trials={int(self.n_trials)}, "
                    f"probe_feats={int(len(getattr(self, '_global_probe_features', []) or []))}"
                )
            except Exception:
                pass

            # Use a single, continuous optimization stage with TPESampler
            # This allows Optuna to explore the full space naturally ('do it on its own')
            # without artificial bounds narrowing.
            # We add n_startup_trials to ensure good initial coverage.
            sampler = optuna.samplers.TPESampler(
                seed=int(self.random_state),
                n_startup_trials=10,
                multivariate=True  # beneficial if kappa/horizon interact
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
                df=df,
                family=family,
                family_events=family_events,
                X_events=X_events,
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
        if isinstance(bounds, dict) and all(k in bounds for k in ('k_low', 'k_high', 'h_low', 'h_high')):
             kappa = trial.suggest_float('kappa', float(bounds['k_low']), float(bounds['k_high']))
             horizon = trial.suggest_int('horizon', int(bounds['h_low']), int(bounds['h_high']))
        else:
             # Default ranges
             kappa = trial.suggest_float('kappa', 1.0, 6.0)
             horizon = trial.suggest_int('horizon', 10, 100)

        try:
            sl_low = float(getattr(self, '_current_config', {}).get('layer2_sl_mult_low', 0.5))
        except Exception:
            sl_low = 0.5
        try:
            sl_high = float(getattr(self, '_current_config', {}).get('layer2_sl_mult_high', 3.0))
        except Exception:
            sl_high = 3.0
        if (not np.isfinite(sl_low)) or sl_low <= 0.0:
            sl_low = 0.5
        if (not np.isfinite(sl_high)) or sl_high <= float(sl_low):
            sl_high = float(max(float(sl_low) + 0.5, 3.0))
        sl_mult = trial.suggest_float('sl_mult', float(sl_low), float(sl_high))

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
             min_rate = float(getattr(self, '_current_config', {}).get('layer2_min_pos_rate', 0.08))
             max_rate = float(getattr(self, '_current_config', {}).get('layer2_max_pos_rate', 0.65))
        except Exception:
             min_rate, max_rate = 0.08, 0.65

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
        # Configurable frequency to reduce compute.
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
            k_vals = [t.params.get('kappa') for t in top_tier]
            sl_vals = [t.params.get('sl_mult') for t in top_tier]
            h_vals = [t.params.get('horizon') for t in top_tier]

            k_vals_f = [float(v) for v in k_vals if v is not None and np.isfinite(float(v))]
            sl_vals_f = [float(v) for v in sl_vals if v is not None and np.isfinite(float(v))]
            h_vals_f = [float(v) for v in h_vals if v is not None and np.isfinite(float(v))]

            if (not k_vals_f) or (not h_vals_f):
                continue

            if not sl_vals_f:
                sl_vals_f = [1.0]

            k_range = max(k_vals_f) - min(k_vals_f) + 1e-6
            sl_range = max(sl_vals_f) - min(sl_vals_f) + 1e-6
            h_range = max(h_vals_f) - min(h_vals_f) + 1e-6

            def get_norm_vec(t):
                return np.array([
                    (float(t.params.get('kappa', 0.0)) - min(k_vals_f)) / k_range,
                    (float(t.params.get('sl_mult', 1.0)) - min(sl_vals_f)) / sl_range,
                    (float(t.params.get('horizon', 0.0)) - min(h_vals_f)) / h_range,
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

    def _train_geometry_models(
        self,
        df: pd.DataFrame,
        X_events: pd.DataFrame,
        events_df: pd.DataFrame,
        geometries: List[GeometryTrial]
    ) -> Dict[str, Any]:
        """
        Train simple LGBM models for each geometry on the provided training set
        to allow Out-Of-Sample prediction generation on the test set.

        Updated to use RobustFocalLoss and specified hyperparameters.
        """
        models = {}
        for g in geometries:
            try:
                lbls, _, _, _ = self._compute_dominance_labels(df, events_df, family=g.family, **g.params)
                valid_lbls = lbls.dropna()
                common_idx = valid_lbls.index.intersection(X_events.index)
                
                if len(common_idx) < 20: 
                     models[g.uuid] = None
                     continue

                X_train = X_events.loc[common_idx]
                y_train = valid_lbls.loc[common_idx]
                
                if len(y_train.unique()) < 2:
                    models[g.uuid] = None
                    continue

                # --- New Hyperparameters ---
                # 'boosting_type': 'gbdt',
                # 'objective': 'binary', (overridden by fobj)
                # 'metric': 'auc',
                # 'max_depth': 5,
                # 'num_leaves': 31,
                # 'lambda_l1': 0.5,
                # 'lambda_l2': 5.0,
                # 'min_data_in_leaf': 50,
                # 'feature_fraction': 0.8,
                # 'bagging_fraction': 0.8,
                # 'bagging_freq': 1,
                # 'learning_rate': 0.02,
                # 'n_estimators': 2000,
                # 'is_unbalance': False,
                # 'scale_pos_weight': 1

                params = {
                    'boosting_type': 'gbdt',
                    'objective': 'binary',
                    'metric': 'auc',
                    'max_depth': 5,
                    'num_leaves': 31,
                    'lambda_l1': 1.0,
                    'lambda_l2': 5.0,
                    'min_data_in_leaf': 30,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 1,
                    'learning_rate': 0.02,
                    'n_estimators': 1000,
                    'verbose': -1,
                    'random_state': 42,
                    'n_jobs': 1,
                    # Disable internal imbalance handling as Focal Loss handles it
                    'is_unbalance': False,
                    'scale_pos_weight': 1,
                    # Add min_gain_to_split to prevent -inf gain issues
                    'min_gain_to_split': 0.005,
                    # Add regularization to prevent overfitting
                    'min_child_weight': 0.001,
                }

                # Instantiate Focal Loss
                focal_obj = RobustFocalLoss(train_labels=y_train.values, gamma=1.5)
                params['objective'] = focal_obj

                # Create Dataset for training
                train_ds = lgb.Dataset(X_train, label=y_train)

                # Train with callbacks
                # Note: We don't have a separate validation set here for early stopping in this specific function flow
                # (it's OOF training on the full 'train' fold passed by the caller).
                # To enable early stopping, we should split X_train internally or rely on n_estimators.
                # Given the instruction to use early stopping (rounds=100), we'll do an internal split 90/10.

                X_tr_inner, X_val_inner = X_train.iloc[:int(len(X_train)*0.9)], X_train.iloc[int(len(X_train)*0.9):]
                y_tr_inner, y_val_inner = y_train.iloc[:int(len(y_train)*0.9)], y_train.iloc[int(len(y_train)*0.9):]

                if len(y_val_inner) < 10:
                    # Too small for split, train on full without early stopping
                    booster = lgb.train(
                        params,
                        train_ds,
                        num_boost_round=params.get('n_estimators', 500)
                    )
                else:
                    train_ds_inner = lgb.Dataset(X_tr_inner, label=y_tr_inner)
                    val_ds_inner = lgb.Dataset(X_val_inner, label=y_val_inner, reference=train_ds_inner)

                    booster = lgb.train(
                        params,
                        train_ds_inner,
                        valid_sets=[train_ds_inner, val_ds_inner],
                        num_boost_round=params.get('n_estimators', 500),
                        callbacks=[
                            lgb.early_stopping(stopping_rounds=100, verbose=False),
                        ]
                    )

                models[g.uuid] = booster
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
        composite_returns = pd.Series(index=events_df.index, dtype=float)
        composite_weights = pd.Series(index=events_df.index, dtype=float)
        oof_preds = {} # Store individual geometry predictions (probabilities)
        oof_vars = {} # Store individual geometry variances

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
                lbls, rets, mfe, mae = self._compute_dominance_labels(df, fam_events, family=family, **g.params)

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

                                 # 1. Prediction (Raw Margins -> Sigmoid)
                                 raw_margins = booster.predict(X_subset)
                                 # Sigmoid is mandatory for Focal Loss output!
                                 probs = 1.0 / (1.0 + np.exp(-raw_margins))

                                 # 2. Variance (Tree Variance)
                                 variances = _calculate_tree_variance(booster, X_subset)

                                 # Store
                                 oof_preds[g.uuid].loc[fam_indices] = probs
                                 oof_vars[g.uuid].loc[fam_indices] = variances

                                 pred_done = True
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
                        prob_vals[fill_mask] = pd.to_numeric(lbls_aligned, errors='coerce').astype(float).fillna(0.0).to_numpy(dtype=float, copy=False)[fill_mask]
                    prob_vals = np.where(np.isfinite(prob_vals), prob_vals, 0.0)
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

            # Use max(probability) to capture the strongest signal
            # Mask out invalid geometries first (0.0 prob is valid, but nan/masked should be ignored)
            # geo_probs_mat is already filled with 0.0 or valid probs.
            # We want max over valid geometries.

            # For labels (0/1): Max is equivalent to Logical OR
            consensus_labels = np.max(geo_labels_mat * valid_mask_mat.astype(float), axis=1)

            # For probs: Max probability
            consensus_prob = np.max(geo_probs_mat * valid_mask_mat.astype(float), axis=1)

            # Weighted Average Return (Keep conservative for PnL estimation)
            w_returns_sum = np.sum(geo_returns_mat * capped_weights_mat, axis=1)
            consensus_returns = w_returns_sum / safe_weights

            # Handle events with no valid geometries
            no_valid_geo = final_event_weights_consensus == 0
            consensus_labels[no_valid_geo] = np.nan
            consensus_returns[no_valid_geo] = np.nan
            consensus_prob[no_valid_geo] = np.nan

            # --- Final Weight Logic: Wsignalgate ---
            # Average Wsignalgate across valid geometries for this event
            avg_w_signalgate = np.divide(
                w_signalgate_accum,
                w_signalgate_count,
                out=np.zeros_like(w_signalgate_accum),
                where=w_signalgate_count > 0
            )

            # Apply MAD-based scaling to the aggregated weights to ensure comparability
            # and robustness, consistent with Layer 0 scaling.
            # finalize_sample_weights performs MAD clipping -> Mean centering (mean=1.0)
            if np.sum(avg_w_signalgate) > 0:
                final_event_weights = finalize_sample_weights(avg_w_signalgate)
            else:
                final_event_weights = np.zeros_like(avg_w_signalgate)

            # Assign to main storage
            composite_labels.loc[fam_events.index] = consensus_labels
            composite_prob.loc[fam_events.index] = consensus_prob
            composite_returns.loc[fam_events.index] = consensus_returns
            composite_weights.loc[fam_events.index] = final_event_weights

        # --- Global Family Normalization (Max 60% of total mass) ---
        # "weights[event.family == fam] = np.minimum(weights[event.family == fam], family_cap)"

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

        if total_weight_global > 0:
            for family in geo_by_fam.keys():
                fam_mask = events_df['family'] == family
                fam_total_weight = composite_weights[fam_mask].sum()

                # Cap at 60% of GLOBAL total
                family_cap = 0.6 * total_weight_global

                if fam_total_weight > family_cap:
                    scale_factor = family_cap / fam_total_weight
                    logger.info(f"Scaling down family {family} by {scale_factor:.4f} (Total: {fam_total_weight:.2f} > Cap: {family_cap:.2f})")
                    composite_weights.loc[fam_mask] *= scale_factor

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

        return {
            "oof_labels": l2_score,
            "oof_returns": composite_returns,
            "weights": composite_weights,
            "l2_score": l2_score,
            "l2_label": l2_label,
            "l2_confidence": l2_confidence,
            "individual_geometries": oof_preds,
            "individual_variances": oof_vars,
            "selected_trials": [asdict(t) for t in geometries]
        }
