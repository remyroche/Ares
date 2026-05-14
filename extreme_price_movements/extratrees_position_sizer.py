"""
extratrees_position_sizer.py

A variant of simple_position_sizer.py using ExtraTrees with high regularization
+ post-calibration instead of Ridge regression.

Provides head-to-head comparison metrics between ET-based and Ridge-based models.
"""

from __future__ import annotations

import logging
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, linregress
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from extreme_price_movements.data_store import read_parquet_projected

# Import from simple_position_sizer to reuse common functions
from extreme_price_movements.simple_position_sizer import (
    detect_meta_head_keys,
    clean_and_standardize,
    walk_forward_temporal_splits,
    evaluate_signal,
    compute_period_aggregated_stats,
    run_stage_1_diagnostics,
    build_combo_candidates,
    run_stage_2_combo_race,
    evaluate_selection_profit_proxy,
    collect_ridge_head_columns,
    _strategy_params_path,
    _frozen_strategy_thresholds_path,
    _extract_strategy_params_payload,
    _save_strategy_params_payload,
    _to_float_or_nan,
    _stable_equity_and_drawdown,
)
from extreme_price_movements.metrics import _stable_equity_and_drawdown
from extreme_price_movements.position_sizer_v2_metrics import (
    compute_bucket_monotonicity,
    compute_false_safe_rate,
    compute_top_slice_metrics,
)
from extreme_price_movements.periods_symbols_management import (
    EventSchema,
    SlicePlanner,
    SlicePlannerConfig,
)
from extreme_price_movements.offline_optimisers.params_store import (
    load_inference_candidate_mask_params_per_bucket,
)
from extreme_price_movements.run_ridge_sizer import (
    load_base_oof_predictions,
    load_meta_oof_predictions,
    load_trade_outcomes,
    find_latest_run_id,
)

logger = logging.getLogger(__name__)


class SimpleHeadExtraTreesSizer:
    """
    A compact experimental component that tests if an ExtraTrees model using
    only meta heads can beat fixed formulas and Ridge regression.
    
    Uses high regularization parameters:
    - Limited max_depth (6-8)
    - High min_samples_leaf (20-50)
    - Limited max_features (sqrt or 0.5)
    - Moderate n_estimators (100-200)
    
    Includes post-calibration via IsotonicRegression.
    """
    
    def __init__(self, model=None, calibration_method: str = "isotonic"):
        if model is None:
            self.model = ExtraTreesRegressor(
                n_estimators=150,
                max_depth=5,
                min_samples_leaf=30,
                min_samples_split=60,
                max_features="sqrt",
                bootstrap=True,
                oob_score=True,
                random_state=42,
                n_jobs=-1,
                verbose=0,
            )
        else:
            self.model = model
        self.calibration_method = calibration_method
        self.calibrator = None
        self.fold_importances = []
        self.feature_names = []
        self.scaler = None
        
    def fit_predict_oof(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        splits: List[Tuple[np.ndarray, np.ndarray]], 
        feature_names: List[str] = None
    ) -> np.ndarray:
        """
        Fits locally on each train fold and predicts the next test fold.
        Stores feature importances for interpretability.
        Applies post-calibration on OOF predictions.
        """
        n_samples = len(y)
        oof_preds = np.zeros(n_samples)
        self.fold_importances = []
        self.feature_names = feature_names or [f"head_{i}" for i in range(X.shape[1])]
        
        for tr_idx, te_idx in splits:
            if len(tr_idx) == 0 or len(te_idx) == 0:
                continue
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_te = X[te_idx]
            
            if X_tr.shape[0] == 0 or X_te.shape[0] == 0:
                continue
                
            # Fold-local scaling and NaN cleaning
            X_tr_clean, medians, scaler, center_1d, scale_1d = clean_and_standardize(X_tr)
            X_te_clean, _, _, _, _ = clean_and_standardize(
                X_te, fit_medians=medians, scaler=scaler, center_1d=center_1d, scale_1d=scale_1d
            )
            
            # Fit ExtraTrees
            self.model.fit(X_tr_clean, y_tr)
            
            # Store feature importances
            self.fold_importances.append(self.model.feature_importances_)
            
            # Predict on test fold
            preds = self.model.predict(X_te_clean)
            oof_preds[te_idx] = preds
            
        # Post-calibration using IsotonicRegression on OOF predictions
        if self.calibration_method == "isotonic" and np.any(oof_preds != 0):
            try:
                valid = np.isfinite(oof_preds) & np.isfinite(y)
                if valid.sum() >= 20:
                    oof_valid = oof_preds[valid]
                    y_valid = y[valid]
                    sort_idx = np.argsort(oof_valid)
                    self.calibrator = IsotonicRegression(out_of_bounds="clip")
                    self.calibrator.fit(oof_valid[sort_idx].reshape(-1, 1), y_valid[sort_idx])
                    oof_preds = self.calibrator.predict(oof_preds.reshape(-1, 1))
            except Exception as e:
                logger.warning(f"Calibration failed: {e}. Using raw predictions.")
                
        return oof_preds
        
    def get_feature_importance(self) -> pd.DataFrame:
        """Returns the mean feature importance across folds."""
        if not self.fold_importances:
            return pd.DataFrame()
            
        importance_matrix = np.array(self.fold_importances)
        mean_importance = np.mean(importance_matrix, axis=0)
        std_importance = np.std(importance_matrix, axis=0)
        
        df = pd.DataFrame({
            "head_name": self.feature_names,
            "mean_importance": mean_importance,
            "std_importance": std_importance,
            "importance_rank": pd.Series(mean_importance).rank(ascending=False).values
        })
        return df.sort_values("mean_importance", ascending=False)


def run_extratrees_position_sizer(
    feature_dict: Dict[str, np.ndarray],
    trade_outcomes: pd.DataFrame,
    y_raw_net_return: np.ndarray,
    y_downside: np.ndarray,
    timestamps: np.ndarray,
    bucket_labels: Optional[np.ndarray] = None,
    sample_weight: Optional[np.ndarray] = None,
    start_equity: float = 100000.0,
    cost_pct: float = 0.003,
    lambda_grid: Optional[List[float]] = None,
    top_fracs: Tuple[float, ...] = (0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2),
    use_extratrees_head_sizer: bool = True,
    calibration_method: str = "isotonic",
    # ExtraTrees hyperparameters for tuning
    et_n_estimators: int = 150,
    et_max_depth: int = 5,
    et_min_samples_leaf: int = 30,
    et_min_samples_split: int = 60,
) -> Dict[str, Any]:
    """
    Main orchestrator for the ExtraTrees position sizer diagnostic framework.
    Mirrors run_simple_position_sizer but uses ExtraTrees instead of Ridge.
    """
    if lambda_grid is None:
        lambda_grid = [0.25, 0.5, 1.0, 2.0]
        
    # 1. Detect Meta Heads (same as Ridge version)
    detected_heads = detect_meta_head_keys(feature_dict)
    used_keys = [k for k in detected_heads.keys() if k in feature_dict]
    missing_keys = [k for k in detected_heads.keys() if k not in feature_dict]
    
    feature_coverage_report = {
        "detected_candidates": list(detected_heads.keys()),
        "used_heads": used_keys,
        "missing_heads": missing_keys,
        "head_classification": detected_heads
    }
    
    # 2. Stage 1 Diagnostics (same)
    stage_1_df = run_stage_1_diagnostics(feature_dict, detected_heads, y_raw_net_return, y_downside)
    
    # Use strict walk-forward splits
    n_samples = len(y_raw_net_return)
    splits = walk_forward_temporal_splits(timestamps, n_samples, n_splits=5)
    
    # 3. Stage 2 Combo Race (same)
    combo_candidates = build_combo_candidates(feature_dict, detected_heads, lambda_grid)
    stage_2_df, best_combo = run_stage_2_combo_race(combo_candidates, y_raw_net_return, y_downside, splits)
    
    # Track the best score
    best_simple_score = None
    best_simple_score_name = None
    
    if not stage_2_df.empty:
        best_simple_score_name = best_combo["combo_name"]
        best_simple_score = combo_candidates[best_simple_score_name]
        
    # 4. ExtraTrees Sizer (replaces Ridge)
    results = {}
    et_sizer_eval = {}
    et_importance_df = pd.DataFrame()
    et_profit_proxy_df = pd.DataFrame()
    
    if use_extratrees_head_sizer and used_keys:
        # Assemble X from used heads
        X_heads = np.column_stack([feature_dict[k] for k in used_keys])
        
        # Create ExtraTrees model with specified hyperparameters
        et_model = ExtraTreesRegressor(
            n_estimators=et_n_estimators,
            max_depth=et_max_depth,
            min_samples_leaf=et_min_samples_leaf,
            min_samples_split=et_min_samples_split,
            max_features="sqrt",
            bootstrap=True,
            oob_score=False,
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )
        
        sizer = SimpleHeadExtraTreesSizer(
            model=et_model,
            calibration_method=calibration_method
        )
        
        # Fit OOF across temporal splits
        et_oof_preds = sizer.fit_predict_oof(X_heads, y_raw_net_return, splits, feature_names=used_keys)
        et_importance_df = sizer.get_feature_importance()
        
        et_metrics = evaluate_signal(
            "ExtraTrees_Head_Sizer", 
            et_oof_preds, 
            y_raw_net_return, 
            y_downside, 
            directionality="return-like"
        )
        et_sizer_eval = et_metrics
        
        # Evaluate Profit Proxy for ET Scores
        t_diff = np.max(timestamps) - np.min(timestamps)
        if hasattr(t_diff, 'astype') and not isinstance(t_diff, float):
            n_days = float(t_diff / np.timedelta64(1, 'D'))
        else:
            n_days = float(t_diff) / 86400.0 if len(timestamps) > 1 else 0.0
            
        et_profit_proxy_df, et_opt_rets, et_opt_ts = evaluate_selection_profit_proxy(
            et_oof_preds,
            y_raw_net_return,
            timestamps=timestamps,
            top_fracs=list(top_fracs),
            cost_pct=cost_pct,
            n_days=n_days,
        )
        
        results["extratrees_sizer_scores_"] = et_oof_preds
        results["extratrees_importance_table_"] = et_importance_df
        results["extratrees_profit_proxy_table_"] = et_profit_proxy_df
        results["extratrees_opt_rets_"] = et_opt_rets
        results["extratrees_opt_ts_"] = et_opt_ts
        
        # Compare ET vs Best Combo
        if not best_combo or et_metrics.get("utility_score", 0) > best_combo.get("utility_score", -9999):
            best_simple_score = et_oof_preds
            best_simple_score_name = "ExtraTrees_Head_Sizer"
            
    # 5. Profit Proxy on Best Score (same logic as Ridge)
    profit_proxy_df = pd.DataFrame()
    best_opt_rets = np.array([])
    best_opt_ts = np.array([])
    if best_simple_score is not None:
        profit_proxy_df, best_opt_rets, best_opt_ts = evaluate_selection_profit_proxy(
            best_simple_score,
            y_raw_net_return,
            timestamps=timestamps,
            top_fracs=list(top_fracs) + [0.3],
            start_equity=start_equity,
            cost_pct=cost_pct
        )
        
    return {
        "feature_coverage_report_": feature_coverage_report,
        "head_diagnostics_table_": stage_1_df,
        "combo_race_table_": stage_2_df,
        "best_combo_": best_combo,
        "extratrees_sizer_eval_": et_sizer_eval,
        "extratrees_importance_table_": et_importance_df,
        "extratrees_profit_proxy_table_": et_profit_proxy_df,
        "best_simple_score_": best_simple_score,
        "best_simple_score_name_": best_simple_score_name,
        "profit_proxy_table_": profit_proxy_df if not profit_proxy_df.empty else pd.DataFrame(),
        "opt_rets_": best_opt_rets,
        "opt_ts_": best_opt_ts
    }


def run_bucketed_extratrees_position_sizer(
    feature_dict: Dict[str, np.ndarray],
    trade_outcomes: pd.DataFrame,
    y_raw_net_return: np.ndarray,
    y_downside: np.ndarray,
    timestamps: np.ndarray,
    bucket_labels: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    min_bucket_samples: int = 50,
    **kwargs
) -> Dict[str, Any]:
    """
    Runs the ExtraTrees position sizer independently per bucket.
    """
    # Run global first
    global_results = run_extratrees_position_sizer(
        feature_dict, trade_outcomes, y_raw_net_return, y_downside, timestamps,
        bucket_labels=None, sample_weight=sample_weight, **kwargs
    )
    
    bucket_results = {}
    summary_rows = []
    
    unique_buckets = np.unique(bucket_labels[~pd.isna(bucket_labels)])
    
    for b in unique_buckets:
        mask = (bucket_labels == b)
        if np.sum(mask) < min_bucket_samples:
            continue
            
        b_feature_dict = {k: v[mask] for k, v in feature_dict.items()}
        b_trade_outcomes = trade_outcomes.iloc[mask].reset_index(drop=True)
        b_y_raw_net_return = y_raw_net_return[mask]
        b_y_downside = y_downside[mask]
        b_timestamps = timestamps[mask]
        b_sample_weight = sample_weight[mask] if sample_weight is not None else None
        
        b_res = run_extratrees_position_sizer(
            b_feature_dict, b_trade_outcomes, b_y_raw_net_return, b_y_downside, b_timestamps,
            bucket_labels=None, sample_weight=b_sample_weight, **kwargs
        )
        bucket_results[b] = b_res
        
        # Build summary row
        summary_rows.append({
            "bucket": b,
            "samples": np.sum(mask),
            "best_model_name": b_res.get("best_simple_score_name_"),
            "best_utility": b_res.get("best_combo_", {}).get("utility_score", 0.0)
        })
        
    global_results["bucket_results"] = bucket_results
    global_results["bucket_summary_table_"] = pd.DataFrame(summary_rows)
    
    return global_results


def run_extratrees_position_sizer_from_artifacts(
    data_root: str,
    run_id: str,
    top_fracs: Tuple[float, ...] = (0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2),
    use_extratrees_head_sizer: bool = True,
    top_n_strategies: int = 4,
    # ExtraTrees hyperparameters
    et_n_estimators: int = 150,
    et_max_depth: int = 5,
    et_min_samples_leaf: int = 30,
    et_min_samples_split: int = 60,
) -> Dict[str, Any]:
    """
    Runs the ExtraTrees position sizer directly on pipeline artifacts.
    Mirrors run_simple_position_sizer_from_artifacts but uses ExtraTrees.
    """
    from extreme_price_movements.run_ridge_sizer import (
        load_trade_outcomes,
        load_meta_oof_predictions,
    )
    from extreme_price_movements.offline_optimisers.params_store import (
        load_inference_candidate_mask_params_per_bucket,
    )
    import re as _re_sizer
    
    # Load dynamic strategies
    _pool = load_inference_candidate_mask_params_per_bucket(top_n=99, ranking_metric="score_for_best_params")
    
    if not _pool:
        logger.warning("No strategies loaded from params_store.")
        return {}
        
    # Deduplicate by strategy_id and take global top-N
    _seen_ids: set = set()
    strategies = []
    for s in _pool:
        sid = s.get("strategy_id", "")
        if sid and sid not in _seen_ids:
            _seen_ids.add(sid)
            strategies.append(s)
    strategies = strategies[:top_n_strategies]
    
    logger.info(f"Loaded {len(strategies)} strategies (global top-{top_n_strategies}).")
    
    # Load base and meta OOFs
    base_oofs = load_base_oof_predictions(data_root, run_id)
    try:
        meta_oofs = load_meta_oof_predictions(data_root, run_id)
    except Exception as e:
        logger.warning(f"Could not load meta OOFs: {e}. Falling back to base-only.")
        meta_oofs = {}
        
    if not base_oofs and not meta_oofs:
        logger.warning(f"No OOFs found in {data_root}/artifacts/{run_id}.")
        return {}
        
    # Supplement with OOF-derived strategies
    _known_ids = {s.get("strategy_id", "") for s in strategies}
    _all_oof_keys = set(base_oofs.keys()) | set(meta_oofs.keys())
    for _oof_key in sorted(_all_oof_keys):
        _stripped = _re_sizer.sub(r'^(long|short)_', '', _oof_key)
        _side = "long" if _oof_key.startswith("long_") else ("short" if _oof_key.startswith("short_") else "")
        if _stripped not in _known_ids and _oof_key not in _known_ids:
            strategies.append({"strategy_id": _stripped, "trade_side": _side})
            _known_ids.add(_stripped)
            
    strategy_results = {}
    
    for strategy in strategies:
        strategy_id = strategy.get("strategy_id", "")
        if not strategy_id:
            continue
            
        # Load labels (same logic as Ridge version)
        labels_dir = Path(data_root) / "artifacts" / run_id / "labels"
        full_df = pd.DataFrame()
        label_file = None
        
        if labels_dir.exists():
            all_label_files = list(labels_dir.glob("train_*.parquet"))
            
            def normalize(s):
                return _re_sizer.sub(r'[^a-z0-9]', '', s.lower())
                
            target_norm = normalize(strategy_id)
            
            for f in all_label_files:
                if "_tight" in f.name or "_wide" in f.name or "_balanced" in f.name:
                    continue
                    
                f_name_norm = normalize(f.stem.replace("train_", ""))
                if target_norm in f_name_norm or f_name_norm in target_norm:
                    label_file = f
                    break
                    
            if not label_file:
                tokens = set(_re_sizer.split(r'[^a-z0-9]', strategy_id.lower()))
                tokens.discard('')
                max_overlap = 0
                best_match = None
                for f in all_label_files:
                    if "_tight" in f.name or "_wide" in f.name or "_balanced" in f.name:
                        continue
                    f_tokens = set(_re_sizer.split(r'[^a-z0-9]', f.stem.lower().replace("train_", "")))
                    f_tokens.discard('')
                    overlap = len(tokens.intersection(f_tokens))
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_match = f
                        
                min_recall = 0.6
                if best_match and len(tokens) > 0 and (max_overlap / len(tokens)) >= min_recall:
                    label_file = best_match
                    
        if label_file and label_file.exists():
            full_df = read_parquet_projected(
                label_file,
                [
                    "__ts__",
                    "__symbol__",
                    "__index__",
                    "__y_ret__",
                    "timestamp",
                    "symbol",
                    "index",
                    "return",
                ],
            )
            if "__y_ret__" in full_df.columns:
                full_df["return"] = full_df["__y_ret__"]
            if "__ts__" in full_df.columns:
                full_df["timestamp"] = full_df["__ts__"]
            if "__symbol__" in full_df.columns:
                full_df["symbol"] = full_df["__symbol__"]
            if "__index__" in full_df.columns:
                full_df["index"] = full_df["__index__"]
                
        trade_side = strategy.get('trade_side', '')
        
        # Resolve OOF bucket
        oof_df = pd.DataFrame()
        resolved_meta_key = None
        
        prefixed = f"{trade_side}_{strategy_id}" if trade_side else strategy_id
        if prefixed in meta_oofs:
            resolved_meta_key = prefixed
        else:
            def _strip_side(k):
                return _re_sizer.sub(r'^(long|short)_', '', k)
                
            strat_norm = _re_sizer.sub(r'[^a-z0-9]', '', strategy_id.lower())
            best_key, best_score = None, 0
            for mk in meta_oofs.keys():
                mk_norm = _re_sizer.sub(r'[^a-z0-9]', '', _strip_side(mk).lower())
                plen = 0
                for a, b in zip(strat_norm, mk_norm):
                    if a == b:
                        plen += 1
                    else:
                        break
                if plen > best_score:
                    best_score = plen
                    best_key = mk
            if best_key and best_score >= 20:
                resolved_meta_key = best_key
                
        if resolved_meta_key:
            oof_df = meta_oofs[resolved_meta_key]
            
        # Join OOF onto Full Labels
        if not full_df.empty:
            if not oof_df.empty and "index" in oof_df.columns:
                join_cols = [c for c in ["timestamp", "symbol"] if c in full_df.columns and c in oof_df.columns]
                if join_cols:
                    oof_clean = oof_df.drop(columns=[c for c in ["return", "y_ret", "y_bin"] if c in full_df.columns], errors="ignore")
                    if "timestamp" in join_cols:
                        for _df in [full_df, oof_clean]:
                            if "timestamp" in _df.columns and hasattr(_df["timestamp"].dtype, "tz") and _df["timestamp"].dt.tz is not None:
                                _df["timestamp"] = _df["timestamp"].dt.tz_localize(None)
                    if "symbol" in join_cols and "symbol" in oof_clean.columns:
                        oof_syms = set(oof_clean["symbol"].dropna().unique())
                        if len(oof_syms) == 1:
                            oof_sym = next(iter(oof_syms))
                            def _norm_sym(s): return str(s).replace("/", "").replace(" ", "").upper()
                            oof_sym_norm = _norm_sym(oof_sym)
                            label_sym_col = full_df["symbol"] if "symbol" in full_df.columns else (full_df["__symbol__"] if "__symbol__" in full_df.columns else None)
                            if hasattr(label_sym_col, "map"):
                                sym_mask = label_sym_col.map(_norm_sym) == oof_sym_norm
                                if sym_mask.sum() > 0:
                                    full_df = full_df[sym_mask].copy()
                            join_cols = ["timestamp"]
                    active_df = pd.merge(full_df, oof_clean, on=join_cols, how="left", suffixes=("", "_oof"))
                else:
                    oof_clean = oof_df.drop(columns=[c for c in ["return", "y_ret", "y_bin"] if c in full_df.columns], errors="ignore")
                    active_df = full_df.copy()
                    for col in oof_clean.columns:
                        if col not in active_df.columns:
                            vals = np.full(len(active_df), np.nan)
                            vals[:min(len(oof_clean), len(active_df))] = oof_clean[col].values[:len(active_df)]
                            active_df[col] = vals
            else:
                active_df = full_df.copy()
        else:
            active_df = oof_df
            
        if active_df.empty:
            logger.warning(f"Could not resolve data for strategy {strategy_id[:60]}. Skipping.")
            continue
            
        active_joined_df = active_df
        
        # Load trade outcomes
        trade_outcomes = load_trade_outcomes(data_root, run_id, active_joined_df)
        if trade_outcomes is None or "return" not in trade_outcomes.columns or len(trade_outcomes) == 0:
            logger.info(f"Skipping strategy {strategy_id}: could not load trade outcomes.")
            continue
            
        # Filter to scored rows
        _oof_score_cols = [c for c in active_joined_df.columns if c in ("oof_prob", "oof_pred", "reg", "clf") or c.startswith(("tbm_", "mae_h", "mfe_h"))]
        if _oof_score_cols:
            n_scorable = int(active_joined_df[_oof_score_cols[0]].notna().sum())
            scorable_mask = active_joined_df[_oof_score_cols[0]].notna()
            active_scored_df = active_joined_df[scorable_mask].copy()
        else:
            n_scorable = len(active_joined_df)
            scorable_mask = np.ones(len(active_joined_df), dtype=bool)
            active_scored_df = active_joined_df.copy()
            
        if len(active_scored_df) < 30:
            logger.warning(f"Strategy {strategy_id[:50]}: only {len(active_scored_df)} scored rows. Skipping.")
            continue
            
        y_raw_net_return = trade_outcomes["return"].values
        if y_raw_net_return.size == 0:
            continue
            
        if "downside" in trade_outcomes.columns:
            y_downside = trade_outcomes["downside"].values
        elif "mae" in active_scored_df.columns:
            y_downside = active_scored_df["mae"].values
        else:
            y_downside = np.zeros_like(y_raw_net_return)
            
        timestamps = active_scored_df["timestamp"].values if "timestamp" in active_scored_df.columns else np.zeros(len(y_raw_net_return))
        
        # Filter head columns (excluding hindsight)
        _HINDSIGHT_COLS = {
            "return", "is_long", "y_ret", "y_bin", "exit_code", "bars_to_mfe",
            "mae_ret", "mfe_ret", "u_policy", "u_policy_net",
        }
        def _is_hindsight(col: str) -> bool:
            return col in _HINDSIGHT_COLS or (col.startswith("__") and col.endswith("__"))
            
        head_cols = [c for c in collect_ridge_head_columns(active_scored_df) if not _is_hindsight(c)]
        head_cols = [c for c in head_cols if active_scored_df[c].notna().any()]
        
        if not head_cols:
            logger.warning(f"Strategy {strategy_id[:50]}: no OOF prediction columns found. Skipping.")
            continue
            
        feature_dict = {col: active_scored_df[col].values for col in head_cols}
        
        # Run ExtraTrees pipeline
        res = run_extratrees_position_sizer(
            feature_dict=feature_dict,
            trade_outcomes=trade_outcomes,
            y_raw_net_return=y_raw_net_return,
            y_downside=y_downside,
            timestamps=timestamps,
            bucket_labels=None,
            top_fracs=top_fracs,
            use_extratrees_head_sizer=use_extratrees_head_sizer,
            et_n_estimators=et_n_estimators,
            et_max_depth=et_max_depth,
            et_min_samples_leaf=et_min_samples_leaf,
            et_min_samples_split=et_min_samples_split,
        )
        
        res["_strategy_meta_"] = {
            "trade_side": strategy.get("trade_side", ""),
            "source_target": strategy.get("source_target", ""),
            "source_horizon": strategy.get("source_horizon", np.nan),
        }
        strategy_results[strategy_id] = res
        
    # Save strategy params
    strategy_params_path = _save_strategy_params_payload(
        data_root=data_root,
        run_id=run_id,
        cost_pct=0.003,
        strategy_results=strategy_results,
    )
    if strategy_params_path is not None:
        logger.info(f"Saved strategy params to {strategy_params_path}")
        
    # Print Strategy Leaderboard
    if strategy_results:
        _print_extratrees_leaderboard(strategy_results)
        
    return strategy_results


def _print_extratrees_leaderboard(strategy_results: Dict[str, Any]) -> None:
    """Prints a formatted leaderboard for ExtraTrees results."""
    leaderboard_rows = []
    
    for sid, res in strategy_results.items():
        opt_table = res.get("extratrees_profit_proxy_table_", pd.DataFrame())
        if opt_table.empty:
            opt_table = res.get("profit_proxy_table_", pd.DataFrame())
            
        if not opt_table.empty:
            if "is_optimal" in opt_table.columns:
                opt_row = opt_table[opt_table["is_optimal"]].iloc[0]
            else:
                opt_row = opt_table.iloc[0]
                
            leaderboard_rows.append({
                "strategy_id": sid[:40] + "...",
                "threshold": opt_row["threshold_pct"],
                "wallet_pnl": opt_row["wallet_pnl"],
                "net_pnl": opt_row["net_pnl"],
                "pnl/trade(bps)": opt_row["pnl_per_trade"],
                "trades/day": opt_row["trades_per_day"],
                "hit_rate": opt_row["hit_rate"],
                "pf": opt_row["profit_factor"],
                "weekly_sortino": opt_row.get("weekly_sortino", np.nan),
                "monthly_sortino": opt_row.get("monthly_sortino", np.nan),
                "stability": opt_row["stability"],
                "mdd": opt_row["max_drawdown"]
            })
            
    if leaderboard_rows:
        print("\n" + "=" * 110)
        print(" EXTRATREES STRATEGY LEADERBOARD")
        print("=" * 110)
        leaderboard_df = pd.DataFrame(leaderboard_rows)
        leaderboard_df = leaderboard_df.sort_values("net_pnl", ascending=False)
        print(leaderboard_df.to_string(index=False))
        print("=" * 110 + "\n")


if __name__ == "__main__":
    import os
    import sys
    import argparse
    from extreme_price_movements.src_utils_tprint import tprint
    
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s:%(name)s:%(message)s'
    )
    
    parser = argparse.ArgumentParser(description="Run ExtraTrees Position Sizer Diagnostics")
    parser.add_argument("--run-id", type=str, help="Run ID to analyze (defaults to latest)")
    parser.add_argument("--data-root", type=str, default=".", help="Root directory for data/artifacts")
    parser.add_argument("--cost-pct", type=float, default=0.003, help="Cost per trade in decimal")
    parser.add_argument("--top-n", type=int, default=4, help="Top N strategies to evaluate")
    parser.add_argument("--n-estimators", type=int, default=150, help="ExtraTrees n_estimators")
    parser.add_argument("--max-depth", type=int, default=5, help="ExtraTrees max_depth")
    parser.add_argument("--min-samples-leaf", type=int, default=30, help="ExtraTrees min_samples_leaf")
    parser.add_argument("--calibration", type=str, default="isotonic", choices=["isotonic", "none"])
    
    args = parser.parse_args()
    
    data_root = args.data_root
    run_id = args.run_id
    
    if not run_id:
        try:
            run_id = find_latest_run_id(data_root)
            tprint(f"Detected latest run: {run_id}")
        except Exception as e:
            tprint(f"Error detecting latest run: {e}")
            sys.exit(1)
            
    tprint(f"Starting ExtraTrees Position Sizer for run: {run_id}")
    tprint(f"Hyperparameters: n_estimators={args.n_estimators}, max_depth={args.max_depth}, min_samples_leaf={args.min_samples_leaf}")
    
    try:
        results = run_extratrees_position_sizer_from_artifacts(
            data_root=data_root,
            run_id=run_id,
            top_n_strategies=args.top_n,
            et_n_estimators=args.n_estimators,
            et_max_depth=args.max_depth,
            et_min_samples_leaf=args.min_samples_leaf,
        )
        
        if not results:
            tprint("No strategy results produced.")
            sys.exit(0)
            
        for strategy_id, res in results.items():
            print("-" * 60)
            print(f"\n STRATEGY: {strategy_id}")
            print("-" * 60)
            
            if "extratrees_importance_table_" in res:
                print(f"\nMeta-Head Importance (ExtraTrees - Walk-Forward OOF):")
                print(res["extratrees_importance_table_"].to_string(index=False))
                
            print(f"\nBest Combo Found: {res.get('best_simple_score_name_', 'N/A')}")
            print(f"  Utility Score: {res.get('best_combo_', {}).get('utility_score', 0.0):.4f}")
            
            print(f"\nExtraTrees Profit Proxy (with Isotonic Calibration):")
            print(res.get("extratrees_profit_proxy_table_", pd.DataFrame()).to_string(index=False))
            print("-" * 60)
            
    except KeyboardInterrupt:
        tprint("Execution interrupted by user.")
    except Exception as e:
        tprint(f"CRITICAL ERROR: {e}")
        import traceback
        tprint(traceback.format_exc())
        sys.exit(1)
