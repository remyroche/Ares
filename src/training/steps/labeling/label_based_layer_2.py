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
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import logging
import copy

# Import compute_realized_returns from the existing module
from src.training.steps.labeling.feature_generation_meta_labeling_step import (
    compute_realized_returns,
    create_meta_features,
)

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

        self._labels_cache: Dict[Any, Tuple[pd.Series, pd.Series]] = {}
        self._signals_cache: Dict[Any, pd.DataFrame] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._global_probe_features: List[str] = []
        self._current_param_bounds: Dict[str, Dict[str, Any]] = {}

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

        # Store for reference
        self.selected_geometries = production_geometries

        # ---------------------------------------------------------------------
        # Part B: OOF Optimization (Analytics Artifacts)
        # ---------------------------------------------------------------------
        logger.info(">>> Layer 2: Running OOF Optimization (Analytics)...")

        # Initialize storage for OOF results
        indices = events_df.index
        oof_labels = pd.Series(np.nan, index=indices)
        oof_returns = pd.Series(np.nan, index=indices)
        oof_weights = pd.Series(np.nan, index=indices)

        # Derive families dynamically to avoid hardcoding
        families = ['Trend Continuation', 'Momentum', 'Mean Reversion']
        max_rank = 4
        oof_geo_preds = {}
        for fam in families:
            for r in range(max_rank):
                key = f"{fam}_Rank{r}"
                oof_geo_preds[key] = pd.Series(np.nan, index=indices)

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

            # OOF Fix: Train models on Train Split
            trained_models = None

            # Prepare X_oof (filtered features)
            X_oof = X_probe_events
            if X_oof is not None and not X_oof.empty and self._global_probe_features:
                 valid_feats = [f for f in self._global_probe_features if f in X_oof.columns]
                 if valid_feats:
                     X_oof = X_oof[valid_feats]

            if X_oof is not None and not X_oof.empty:
                try:
                    trained_models = self._train_geometry_models(
                        df=df_train,
                        X_events=X_oof,
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

                fold_output = self._bagged_labeling(
                    df_label, 
                    events_test, 
                    standardized_geos,
                    trained_models=trained_models,
                    X_events=X_oof
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

                oof_labels.loc[target_idx] = fold_output['oof_labels'].reindex(target_idx)
                oof_returns.loc[target_idx] = fold_output['oof_returns'].reindex(target_idx)
                oof_weights.loc[target_idx] = fold_output['weights'].reindex(target_idx)

                # Assign individual geometry preds
                for uuid, series in fold_output['individual_geometries'].items():
                    if uuid in oof_geo_preds:
                        oof_geo_preds[uuid].loc[target_idx] = series.reindex(target_idx)

        # ---------------------------------------------------------------------
        # Final Packaging
        # ---------------------------------------------------------------------
        final_geo_preds = {k: v for k, v in oof_geo_preds.items() if v.notna().any()}

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
                    horizon = None
                    if isinstance(params, dict):
                        kappa = params.get("kappa")
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
                            _lbl, _ret = self._compute_dominance_labels(
                                df=df,
                                events_df=fam_events,
                                kappa=float(kappa),
                                horizon=int(horizon),
                                family=fam,
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
            "oof_labels": oof_labels,
            "oof_returns": oof_returns,
            "weights": oof_weights,
            "individual_geometries": final_geo_preds,
            "events_df": events_df,
            "selected_trials": [asdict(t) for t in production_geometries]
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
        Step 0: Generate events where |r_t| / sigma_t > 0.5.
        Returns a DataFrame of event timestamps.
        """
        returns = df['close'].pct_change()
        # Avoid division by zero
        vol = df['volatility_1d'].replace(0, np.nan)

        # Signal to Noise Ratio
        snr = returns.abs() / vol

        # Event Trigger
        try:
            snr_thr = float(getattr(self, '_current_config', {}).get('layer2_event_snr_threshold', 0.5))
        except Exception:
            snr_thr = 0.5
        if (not np.isfinite(snr_thr)) or float(snr_thr) <= 0.0:
            snr_thr = 0.5
        trigger_mask = snr > float(snr_thr)

        events = df.index[trigger_mask]
        logger.info(f"Generated {len(events)} events from {len(df)} bars.")

        # Create events dataframe (index=timestamp)
        # We can store regime info here for easy lookup
        events_df = df.loc[events, ['trend_regime', 'vol_regime', 'volatility_1d']].copy()

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
        returns = df['close'].pct_change()
        try:
            direction_mode = str(getattr(self, "_current_config", {}).get("layer2_direction_mode", "lagged"))
        except Exception:
            direction_mode = "lagged"

        try:
            lookback = int(getattr(self, "_current_config", {}).get("layer2_direction_lookback", 20))
        except Exception:
            lookback = 20
        lookback = int(max(2, lookback))

        try:
            dir_raw = str(getattr(self, "_current_config", {}).get("direction", "long")).lower()
        except Exception:
            dir_raw = "long"
        default_dir = 1.0
        if dir_raw in {"short", "sell", "-1", "s"}:
            default_dir = -1.0

        if direction_mode.lower() in ("same_bar", "same", "current"):
            dir_src = returns
        else:
            dir_src = returns.shift(1)

        trend_src = dir_src.rolling(lookback).mean()

        key = (
            self._df_cache_key(df),
            self._events_cache_key(events_df.index),
            str(family),
            str(direction_mode),
            int(lookback),
            float(default_dir),
        )

        cached = self._signals_cache.get(key)
        if cached is not None:
            return cached

        directions = np.sign(trend_src.reindex(events_df.index).to_numpy(dtype=float, na_value=0.0))
        directions = np.where(np.isfinite(directions), directions, float(default_dir))
        directions[directions == 0.0] = float(default_dir)

        if family == 'Mean Reversion':
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
        events_shift: int = 0
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Compute MFE/MAE dominance labels.
        Label = 1 if MFE > Kappa * MAE (Directional Dominance).

        Args:
            df: Market data
            events_df: Events to label
            kappa: Dominance ratio threshold
            horizon: Window size
            family: Geometry family (defines direction)
            events_shift: Shift event timestamps by N bars (for stability check)
        """
        try:
            direction_mode = str(getattr(self, "_current_config", {}).get("layer2_direction_mode", "lagged"))
        except Exception:
            direction_mode = "lagged"

        cache_key = (
            self._df_cache_key(df),
            self._events_cache_key(events_df.index),
            str(family),
            float(round(float(kappa), 8)),
            int(horizon),
            int(events_shift),
            float(self.transaction_cost),
            str(direction_mode),
            "dominance"
        )
        cached = self._labels_cache.get(cache_key)
        if cached is not None:
            self._cache_hits += 1
            return cached
        self._cache_misses += 1

        signals = self._get_or_build_signals(df, events_df, family)

        # Handle event shifting for stability check
        # We need to calculate outcomes for the trade entered at t + shift
        # but using the direction signal from t (persistence check)
        # However, compute_realized_returns looks up signal at the entry time.
        # So we must create a signals frame where the signal is present at t + shift.

        target_events_idx = events_df.index
        if events_shift != 0:
            # Shift event timestamps by bars
            # We map current event index position to new position
            df_idx_locs = df.index.get_indexer(target_events_idx)
            shifted_locs = df_idx_locs + events_shift

            # Filter valid
            valid_locs = (shifted_locs >= 0) & (shifted_locs < len(df))

            if not np.any(valid_locs):
                 # All shifted out of bounds
                 return pd.Series(np.nan, index=target_events_idx), pd.Series(np.nan, index=target_events_idx)

            # New timestamps
            shifted_timestamps = df.index[shifted_locs[valid_locs]]

            # We need signals at these new timestamps to match original direction
            # Copy original signals to new timestamps
            orig_signals = signals.loc[target_events_idx[valid_locs]]

            # Create a temporary signals dataframe for the shifted calculation
            # We assume no conflicting signals at the shifted times for simplicity of this check
            temp_signals = pd.DataFrame(0.0, index=df.index, columns=['consensus'])
            temp_signals.loc[shifted_timestamps, 'consensus'] = orig_signals['consensus'].values

            calc_signals = temp_signals
            calc_events_idx = shifted_timestamps
        else:
            calc_signals = signals
            calc_events_idx = target_events_idx

        # Use Infinite Barriers to capture full window path
        # TP/SL = Infinity implies exit only on timeout (end of horizon)
        # compute_realized_returns will compute MFE/MAE over the full horizon
        inf_threshold = pd.Series(float('inf'), index=df.index)

        (
            realized_returns,
            _, _, _,
            mfe_series,
            mae_series,
            _, _
        ) = compute_realized_returns(
            df=df,
            signals=calc_signals,
            profit_threshold=inf_threshold,
            stop_threshold=inf_threshold,
            horizon=horizon,
            transaction_cost=self.transaction_cost,
            min_event_spacing=0, # No spacing check for meta-labeling candidates
            volatility_series=None # Fixed horizon
        )

        # Filter to events
        subset_mfe = mfe_series.reindex(calc_events_idx)
        subset_mae = mae_series.reindex(calc_events_idx)
        subset_returns = realized_returns.reindex(calc_events_idx)

        # Apply Dominance Logic
        # Label = 1 if MFE > Kappa * MAE
        # Also enforce a minimum noise floor: MFE > epsilon
        # Epsilon = 2 * transaction_cost (ensure we beat costs significantly)
        epsilon = 2.0 * self.transaction_cost

        # Ensure MAE is non-zero to avoid division by zero (though < implies non-neg)
        # MAE is absolute value in compute_realized_returns
        safe_mae = subset_mae.replace(0.0, 1e-9)

        dominance_ratio = subset_mfe / safe_mae

        # Logic:
        # 1. Dominance met: ratio > kappa
        # 2. Significant move: mfe > epsilon
        # 3. Direction confirmed (mfe is positive by definition, but net return check?)
        #    Dominance implies we *could* have exited for profit.
        #    We label "1" if the PATH allowed for a dominant win.

        labels = (dominance_ratio > kappa) & (subset_mfe > epsilon)
        binary_labels = labels.astype(float)

        # For those that failed dominance, label is 0

        # Map back to original event indices if shifted
        if events_shift != 0:
            final_labels = pd.Series(np.nan, index=target_events_idx)
            final_returns = pd.Series(np.nan, index=target_events_idx)

            final_labels.iloc[valid_locs] = binary_labels.values
            final_returns.iloc[valid_locs] = subset_returns.values # This is "timeout" return
        else:
            final_labels = binary_labels
            final_returns = subset_returns

        # For Label=1, we assume we captured the move.
        # Assign return = MFE (optimistic) or MFE - cost?
        # Let's use MFE - cost to represent the "potential" captured.
        # For Label=0, we assign the actual realized return at timeout (likely small or negative)
        # subset_returns contains the return at horizon end (timeout).
        # We update returns where label is 1

        # Actually, let's keep it simple: Return = Net Return at Horizon.
        # The Label indicates if it was a "Dominant" path.
        # BUT: If we use this for weighting, a Label=1 with negative horizon return might be confusing.
        # If Dominance is true, it means there existed a point `t < T` where `P_t` was good.
        # We assign `MFE - cost` as the proxy return for successful dominance trades.

        if events_shift == 0: # Only update returns for the primary calculation
            success_mask = (final_labels == 1.0)
            final_returns.loc[success_mask] = subset_mfe.loc[success_mask] - self.transaction_cost

        result = (final_labels, final_returns)
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

        return self._compute_dominance_labels(df, events_df, kappa, int(horizon), family)

    def _build_geometry_independent_event_features(
        self,
        df: pd.DataFrame,
        events_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build one feature matrix for all events, independent of TP/SL/Horizon geometry."""
        signals = pd.DataFrame(index=df.index)
        try:
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
                max_depth=3,
                num_leaves=8,
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

        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        # Models
        lgbm = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=3,
            num_leaves=8,
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
            for train_index, test_index in tscv.split(X_clean):
                X_train, X_test = X_clean.iloc[train_index], X_clean.iloc[test_index]
                y_train, y_test = y_clean.iloc[train_index], y_clean.iloc[test_index]

                if y_train.nunique() < 2 or y_test.nunique() < 2:
                    continue

                # LGBM
                if w_clean is not None:
                    lgbm.fit(
                        X_train, y_train,
                        sample_weight=w_clean[train_index],
                        eval_set=[(X_test, y_test)],
                        callbacks=[lgb.early_stopping(10, verbose=False)]
                    )
                else:
                    lgbm.fit(
                        X_train, y_train,
                        eval_set=[(X_test, y_test)],
                        callbacks=[lgb.early_stopping(10, verbose=False)]
                    )
                p_lgbm = lgbm.predict_proba(X_test)[:, 1]
                p_lgbm = np.clip(np.asarray(p_lgbm, dtype=float), 1e-6, 1.0 - 1e-6)

                try:
                    sw_te = w_clean[test_index] if w_clean is not None else None
                    auc_val = roc_auc_score(y_test, p_lgbm, sample_weight=sw_te) if sw_te is not None else roc_auc_score(y_test, p_lgbm)
                    metrics['lgbm_auc'].append(auc_val)
                    metrics['lgbm_ll'].append(log_loss(y_test, p_lgbm, sample_weight=sw_te) if sw_te is not None else log_loss(y_test, p_lgbm))
                    try:
                        pr_val = average_precision_score(y_test, p_lgbm, sample_weight=sw_te) if sw_te is not None else average_precision_score(y_test, p_lgbm)
                    except Exception:
                        pr_val = float('nan')
                    if np.isfinite(pr_val):
                        metrics['lgbm_pr'].append(float(pr_val))
                    ic, _ = spearmanr(y_test, p_lgbm)
                    metrics['lgbm_ic'].append(ic if not np.isnan(ic) else 0.0)
                    
                    # --- Optuna Pruning ---
                    if trial is not None:
                        trial.report(auc_val, step=fold_idx)
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                    # ----------------------
                    
                except optuna.TrialPruned:
                    raise
                except Exception:
                    pass

                # Linear
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                if w_clean is not None:
                    linear.fit(X_train_scaled, y_train, sample_weight=w_clean[train_index])
                else:
                    linear.fit(X_train_scaled, y_train)
                raw_scores = linear.predict(X_test_scaled)
                raw_scores = np.asarray(raw_scores, dtype=float)
                raw_scores = np.clip(raw_scores, -20.0, 20.0)
                p_linear = 1.0 / (1.0 + np.exp(-raw_scores))
                p_linear = np.clip(np.asarray(p_linear, dtype=float), 1e-6, 1.0 - 1e-6)

                try:
                    sw_te = w_clean[test_index] if w_clean is not None else None
                    metrics['lin_auc'].append(roc_auc_score(y_test, p_linear, sample_weight=sw_te) if sw_te is not None else roc_auc_score(y_test, p_linear))
                    metrics['lin_ll'].append(log_loss(y_test, p_linear, sample_weight=sw_te) if sw_te is not None else log_loss(y_test, p_linear))
                    try:
                        pr_val = average_precision_score(y_test, p_linear, sample_weight=sw_te) if sw_te is not None else average_precision_score(y_test, p_linear)
                    except Exception:
                        pr_val = float('nan')
                    if np.isfinite(pr_val):
                        metrics['lin_pr'].append(float(pr_val))
                    ic, _ = spearmanr(y_test, p_linear)
                    metrics['lin_ic'].append(ic if not np.isnan(ic) else 0.0)
                except:
                    pass
                
                fold_idx += 1

        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.warning(f"Probe failure: {e}")
            return {'auc': 0.5, 'log_loss': 1.0, 'ic': 0.0, 'passed': False}

        if not metrics['lgbm_auc']:
            return {'auc': 0.5, 'log_loss': 1.0, 'ic': 0.0, 'passed': False}

        auc_lgbm = np.asarray(metrics['lgbm_auc'], dtype=float)
        auc_lin = np.asarray(metrics['lin_auc'], dtype=float)
        pr_lgbm = np.asarray(metrics['lgbm_pr'], dtype=float) if metrics.get('lgbm_pr') else np.asarray([], dtype=float)
        pr_lin = np.asarray(metrics['lin_pr'], dtype=float) if metrics.get('lin_pr') else np.asarray([], dtype=float)

        avg_auc_lgbm = float(np.mean(auc_lgbm))
        avg_auc_linear = float(np.mean(auc_lin))

        avg_ic_lgbm = np.mean(metrics['lgbm_ic'])
        avg_ic_linear = np.mean(metrics['lin_ic'])

        avg_ll_lgbm = np.mean(metrics['lgbm_ll'])
        avg_ll_linear = np.mean(metrics['lin_ll'])

        final_auc = float(np.median([avg_auc_lgbm, avg_auc_linear]))
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
            auc_thr = float(getattr(self, '_current_config', {}).get('layer2_probe_auc_threshold', 0.52))
        except Exception:
            auc_thr = 0.52
        try:
            pr_margin = float(getattr(self, '_current_config', {}).get('layer2_probe_pr_margin', 0.02))
        except Exception:
            pr_margin = 0.02
        pr_thr = float(pr_baseline + pr_margin) if np.isfinite(pr_baseline) else float('nan')

        passed_auc = bool(np.isfinite(final_auc) and (final_auc >= float(auc_thr)))
        passed_pr = bool((not np.isfinite(pr_thr)) or (np.isfinite(pr_best) and (pr_best >= pr_thr)))
        passed = bool(passed_auc and passed_pr)

        return {
            'auc': final_auc,
            'auc_std': auc_std,
            'pr_auc': pr_best,
            'pr_auc_baseline': pr_baseline,
            'ic': float(np.mean([avg_ic_lgbm, avg_ic_linear])),
            'log_loss': float(np.mean([avg_ll_lgbm, avg_ll_linear])),
            'auc_lgbm': float(avg_auc_lgbm),
            'auc_linear': float(avg_auc_linear),
            'passed': passed,
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
        base_labels, _ = self._compute_dominance_labels(df, events_df, family=family, **trial_params)

        # 2. Shifted Labels (+1 bar)
        # Using events_shift=1
        shift1_labels, _ = self._compute_dominance_labels(
            df, events_df, family=family, events_shift=1, **trial_params
        )

        # 3. Shifted Labels (-1 bar)
        # Using events_shift=-1
        shift_neg1_labels, _ = self._compute_dominance_labels(
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

        # Threshold: 85% agreement required
        if avg_agreement < 0.85:
             logger.debug(f"Stability failed: Flip rate too high (agreement={avg_agreement:.2f})")
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

            # Split n_trials into stage1 (broad) and stage2 (fine)
            n_stage1 = int(max(5, int(self.n_trials * 0.6)))
            n_stage2 = int(max(0, int(self.n_trials) - int(n_stage1)))

            # Add MedianPruner to optimize computational resources
            study = optuna.create_study(
                direction="maximize",
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1)
            )
            trial_results = []

            # Use partial to pass context to the extracted objective method
            from functools import partial
            obj_func = partial(
                self._optimization_objective,
                df=df,
                family=family,
                family_events=family_events,
                X_events=X_events,
                target_sample_weight_events=target_sample_weight_events
            )

            study.optimize(obj_func, n_trials=int(n_stage1))

            if n_stage2 > 0:
                # Narrow bounds around best params
                best_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                best_trials = sorted(best_trials, key=lambda t: float(t.value) if t.value is not None else -1e9, reverse=True)
                best = best_trials[0] if best_trials else None

                if best is not None:
                    try:
                        k0 = float(best.params.get('kappa'))
                        h0 = int(best.params.get('horizon'))
                    except Exception:
                        k0 = None
                        h0 = None

                    if k0 is not None and h0 is not None:
                        shrink = float(self._current_config.get('layer2_stage2_shrink', 0.25)) if isinstance(self._current_config, dict) else 0.25
                        shrink = float(np.clip(shrink, 0.05, 0.75))
                        b0 = self._current_param_bounds.get(str(family), {})
                        self._current_param_bounds[str(family)] = {
                            'k_low': float(max(b0.get('k_low', 1.0), k0 * (1.0 - shrink))),
                            'k_high': float(min(b0.get('k_high', k0 * (1.0 + shrink)), k0 * (1.0 + shrink))),
                            'h_low': int(max(b0.get('h_low', 1), int(max(1, round(h0 * (1.0 - shrink)))))),
                            'h_high': int(min(b0.get('h_high', int(max(2, round(h0 * (1.0 + shrink))))), int(max(2, round(h0 * (1.0 + shrink)))))),
                        }

                refine = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=int(self.random_state)))
                refine.optimize(obj_func, n_trials=int(n_stage2))
                # Merge refine trials into the first study for extraction
                for t in refine.trials:
                    try:
                        study.add_trial(t)
                    except Exception:
                        pass

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

        # Compute labels
        labels, returns = self._compute_dominance_labels(df, family_events, kappa, horizon, family)

        # Metrics
        mean_ret = returns.mean()
        if np.isnan(mean_ret): mean_ret = -1.0

        # Positive Rate Filter (10-40%)
        count = labels.notna().sum()
        if count < 20:
            return -1.0 # Too few samples

        pos_rate = labels.mean()

        if pos_rate < 0.05 or pos_rate > 0.40: # Relaxed slightly for initial discovery, but strict on target
             t_obj = GeometryTrial(
                family=family,
                params={'kappa': kappa, 'horizon': horizon},
                final_score=-1.0,
                learnability=0.0,
                robust_magnitude=0.0,
                stability=0.0,
                balance=0.0,
                raw_metrics={'passed': False, 'pos_rate': pos_rate},
                uuid=f"{family}_{trial.number}"
            )
             trial.set_user_attr("geometry_object", t_obj)
             return -1.0

        # Stability Check (Time-Flip)
        # This is expensive, so maybe do it only if passed?
        # But we need it for scoring.
        # "Pick the geometry that is robust"

        # We perform stability check here
        # (This is cheaper than training ML probes, so good to do before)
        is_stable = self._check_stability(df, family_events, {'kappa': kappa, 'horizon': horizon}, 0.0, family)

        if not is_stable:
             t_obj = GeometryTrial(
                family=family,
                params={'kappa': kappa, 'horizon': horizon},
                final_score=-1.0,
                learnability=0.0,
                robust_magnitude=0.0,
                stability=0.0,
                balance=0.0,
                raw_metrics={'passed': False, 'pos_rate': pos_rate, 'stable': False},
                uuid=f"{family}_{trial.number}"
            )
             trial.set_user_attr("geometry_object", t_obj)
             return -1.0

        # Align features to events
        try:
            X_geom = X_events.loc[labels.index]
        except Exception:
            X_geom = X_events.reindex(labels.index)

        global_feats = [f for f in getattr(self, '_global_probe_features', []) if f in X_geom.columns]
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

        if not probe_res['passed']:
            learnability = 0.0
        else:
            learnability = probe_res['auc']

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
            params={'kappa': kappa, 'horizon': horizon},
            final_score=final_score,
            learnability=learnability,
            robust_magnitude=float(mean_ret) * 1000,
            stability=1.0, # Passed stability check
            balance=degeneracy_floor,
            raw_metrics=dict(probe_res, **{'pos_rate': pos_rate}),
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

            if not top_tier:
                continue

            fam_selected = []

            # Helper to normalize params for distance calculation
            k_vals = [t.params['kappa'] for t in top_tier]
            h_vals = [t.params['horizon'] for t in top_tier]

            if not k_vals or not h_vals:
                continue

            k_range = max(k_vals) - min(k_vals) + 1e-6
            h_range = max(h_vals) - min(h_vals) + 1e-6

            def get_norm_vec(t):
                return np.array([
                    (t.params['kappa'] - min(k_vals)) / k_range,
                    (t.params['horizon'] - min(h_vals)) / h_range
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
                    _lbl, _ret = self._compute_dominance_labels(df, fam_events_local, family=fam, **t_obj.params)
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
        """
        models = {}
        for g in geometries:
            try:
                lbls, _ = self._compute_dominance_labels(df, events_df, family=g.family, **g.params)
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

                clf = lgb.LGBMClassifier(
                    n_estimators=50, 
                    max_depth=2,
                    num_leaves=4,
                    learning_rate=0.05,
                    n_jobs=1,
                    verbose=-1,
                    random_state=42
                )
                clf.fit(X_train, y_train)
                models[g.uuid] = clf
            except Exception:
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
        composite_returns = pd.Series(index=events_df.index, dtype=float)
        composite_weights = pd.Series(index=events_df.index, dtype=float)
        oof_preds = {} # Store individual geometry predictions

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
            geo_scores_mat = np.zeros((n_events, n_geos))
            valid_mask_mat = np.zeros((n_events, n_geos), dtype=bool)

            for i, g in enumerate(fam_geos):
                # Compute labels/returns for this geometry
                lbls, rets = self._compute_dominance_labels(df, fam_events, family=family, **g.params)

                # Store individual geometry output
                # Store individual geometry output
                # OOF Fix: Use trained model if available and X_events provided
                pred_done = False
                if trained_models is not None and X_events is not None and g.uuid in trained_models:
                     clf = trained_models[g.uuid]
                     if clf is not None:
                         # Predict on fam_events
                         fam_indices = fam_events.index.intersection(X_events.index)
                         if not fam_indices.empty:
                             try:
                                 probs = clf.predict_proba(X_events.loc[fam_indices])[:, 1]
                                 oof_preds[g.uuid] = pd.Series(probs, index=fam_indices)
                                 # Align to full fam_events if missing some
                                 oof_preds[g.uuid] = oof_preds[g.uuid].reindex(fam_events.index)
                                 pred_done = True
                             except Exception:
                                 logger.warning(f"OOF prediction failed for {g.uuid}")
                
                if not pred_done:
                    # CRITICAL FIX: Do NOT fall back to ground truth labels (lbls) for OOF features.
                    # Use NaN (uncertainty) which will be filled with 0.5 in Layer 3.
                    oof_preds[g.uuid] = pd.Series(np.nan, index=fam_events.index)

                # Align to fam_events index
                lbls_aligned = lbls.reindex(fam_events.index)
                rets_aligned = rets.reindex(fam_events.index)

                # Identify valid labels (not NaN)
                not_na = lbls_aligned.notna()

                # Fill matrices
                geo_labels_mat[not_na, i] = lbls_aligned[not_na]
                geo_returns_mat[not_na, i] = rets_aligned[not_na]
                geo_scores_mat[not_na, i] = g.final_score
                valid_mask_mat[not_na, i] = True

            if geo_labels_mat.shape != geo_returns_mat.shape or geo_labels_mat.shape != geo_scores_mat.shape:
                raise ValueError("Layer2 bagging: geometry matrices have inconsistent shapes")
            if geo_labels_mat.shape != valid_mask_mat.shape:
                raise ValueError("Layer2 bagging: valid mask has inconsistent shape")

            # --- Per-Geometry Capping Logic ---
            # Raw total score per event
            # Sum of scores of VALID geometries for each event
            # Scores must never be negative: negative scores lead to negative event weights and break downstream.
            # If all selected geometries have non-positive scores (e.g. all failed probes), fall back to
            # uniform weights over valid geometries.
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
            # Only apply to valid geometries (invalid have score 0 anyway in calculation,
            # but let's be explicit: capped_weight should be 0 if invalid)
            capped_weights_mat = np.minimum(score_base_mat, max_contrib_mat)
            capped_weights_mat[~valid_mask_mat] = 0.0

            # Final Event Weight (sum of capped weights)
            final_event_weights = np.sum(capped_weights_mat, axis=1)

            # Safety: ensure non-negative event weights
            final_event_weights = np.where(np.isfinite(final_event_weights), final_event_weights, 0.0)
            final_event_weights = np.maximum(final_event_weights, 0.0)

            if final_event_weights.shape[0] != n_events:
                raise ValueError("Layer2 bagging: final_event_weights shape mismatch")

            # Avoid division by zero
            safe_weights = final_event_weights.copy()
            safe_weights[safe_weights == 0] = 1.0 # arbitrary, will be 0 in result anyway

            # Weighted Consensus Calculation
            # Weighted Average Label
            w_labels_sum = np.sum(geo_labels_mat * capped_weights_mat, axis=1)
            consensus_labels = w_labels_sum / safe_weights

            # Weighted Average Return
            w_returns_sum = np.sum(geo_returns_mat * capped_weights_mat, axis=1)
            consensus_returns = w_returns_sum / safe_weights

            # Handle events with no valid geometries
            no_valid_geo = final_event_weights == 0
            consensus_labels[no_valid_geo] = np.nan
            consensus_returns[no_valid_geo] = np.nan

            # Assign to main storage
            composite_labels.loc[fam_events.index] = consensus_labels
            composite_returns.loc[fam_events.index] = consensus_returns
            composite_weights.loc[fam_events.index] = final_event_weights

        # --- Global Family Normalization (Max 60% of total mass) ---
        # "weights[event.family == fam] = np.minimum(weights[event.family == fam], family_cap)"

        # Fill NaNs in weights with 0
        composite_weights = composite_weights.fillna(0.0)
        composite_weights = composite_weights.clip(lower=0.0)

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

        return {
            "oof_labels": composite_labels,
            "oof_returns": composite_returns,
            "weights": composite_weights,
            "individual_geometries": oof_preds,
            "selected_trials": [asdict(t) for t in geometries]
        }
