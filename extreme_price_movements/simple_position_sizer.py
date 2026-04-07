"""
simple_position_sizer.py

A lightweight, diagnostic-first module that evaluates whether upstream meta-model heads
are economically sufficient before any complex policy optimization. It answers:
"Can these upstream inputs be used to generate a profit proxy?"

Focuses on:
1. Stage 1: head-level diagnostics
2. Stage 2: small-combo race
3. Simple Ridge-based sizer using only meta-model heads as inputs.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

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
)

logger = logging.getLogger(__name__)

def detect_meta_head_keys(feature_dict: Dict[str, np.ndarray], config_overrides: Optional[List[str]] = None) -> Dict[str, str]:
    """Detects likely meta-model heads from the feature dictionary and classifies them."""
    if config_overrides:
        keys = [k for k in config_overrides if k in feature_dict]
    else:
        keys = list(feature_dict.keys())

    heads = {}
    for k in keys:
        kl = k.lower()
        if "edge" in kl or "expected_return" in kl or "regressor" in kl or "reg_head" in kl:
            heads[k] = "return-like"
        elif "mae" in kl or "downside" in kl or "risk" in kl:
            heads[k] = "risk-like"
        elif "mfe" in kl or "upside" in kl:
            heads[k] = "upside-like"
        elif "asym" in kl:
            heads[k] = "asymmetry-like"
        elif "uncert" in kl or "confid" in kl:
            heads[k] = "uncertainty-like"
        elif "prob" in kl or "logit" in kl or "class" in kl or "meta_clf" in kl or "multi_obj" in kl:
            heads[k] = "classification-like"

    # Include keys that were requested via override but missed the heuristic (if any).
    if config_overrides:
        for k in config_overrides:
            if k in feature_dict and k not in heads:
                heads[k] = "unknown"

    return heads

def clean_and_standardize(X: np.ndarray, fit_medians: Optional[np.ndarray] = None, scaler: Optional[StandardScaler] = None, mean_1d: Optional[float] = None, std_1d: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, Any, Any, Any]:
    """Standardizes features safely handling NaNs and Infs."""
    X_clean = X.copy()
    X_clean[np.isinf(X_clean)] = np.nan

    if fit_medians is None:
        fit_medians = np.nanmedian(X_clean, axis=0)
        if np.isscalar(fit_medians):
            if np.isnan(fit_medians):
                fit_medians = 0.0
        else:
            fit_medians[np.isnan(fit_medians)] = 0.0

    if X_clean.ndim == 1:
        inds = np.isnan(X_clean)
        X_clean[inds] = fit_medians

        if mean_1d is None or std_1d is None:
            mean_1d = np.mean(X_clean)
            std_1d = np.std(X_clean)

        if std_1d > 1e-9:
            X_clean = (X_clean - mean_1d) / std_1d
        else:
            X_clean = X_clean - mean_1d
    else:
        inds = np.where(np.isnan(X_clean))
        X_clean[inds] = np.take(fit_medians, inds[1])

        if scaler is None:
            scaler = StandardScaler()
            X_clean = scaler.fit_transform(X_clean)
        else:
            X_clean = scaler.transform(X_clean)

    return X_clean, fit_medians, scaler, mean_1d, std_1d

def simple_temporal_splits(
    timestamps: Optional[np.ndarray],
    n_samples: int,
    n_splits: int = 3
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generates simple temporal cross-validation splits."""
    if timestamps is None or len(timestamps) == 0:
        # Fallback to simple chunking if no timestamps
        indices = np.arange(n_samples)
        chunk_size = max(1, n_samples // n_splits)
        splits = []
        for i in range(n_splits):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < n_splits - 1 else n_samples
            test_idx = indices[start:end]
            train_idx = np.concatenate([indices[:start], indices[end:]]) if n_splits > 1 else test_idx
            if len(train_idx) > 0 and len(test_idx) > 0:
                splits.append((train_idx, test_idx))
        return splits

    # Try to use SlicePlanner for honest temporal splits if available
    try:
        ts = pd.to_datetime(pd.Series(timestamps), unit="s", utc=True, errors="coerce")
        if ts.isna().all():
            ts = pd.to_datetime(pd.Series(timestamps), utc=True, errors="coerce")
        ts = ts.ffill().bfill()
        events = pd.DataFrame({
            "event_id": np.arange(n_samples, dtype=np.int64),
            "symbol": np.repeat("ALL", n_samples),
            "t0": ts.to_numpy(),
            "t1": (ts + pd.Timedelta(seconds=1)).to_numpy(),
        })
        cfg = SlicePlannerConfig.fast_defaults(schema=EventSchema())
        cfg = cfg.__class__(**{
            **cfg.__dict__,
            "preset": cfg.preset.__class__(
                preset_name=cfg.preset.preset_name,
                outer=cfg.preset.outer,
                inner=cfg.preset.inner.__class__(n_splits=max(1, int(n_splits))),
                sampling=cfg.preset.sampling,
                symbol_policy=cfg.preset.symbol_policy,
                purge_policy=cfg.preset.purge_policy,
            ),
            "silent": True,
            "min_rows_per_fold": 1,
            "min_symbols_per_fold": 1,
        })
        bundle = SlicePlanner(cfg).build(events)
        plans = bundle["consumer_plans"]["ridge_sizer_fit"]
        splits = []
        for plan in plans:
            if plan.tag != "predict_outer_test":
                continue
            tr = np.asarray(plan.fit_idx, dtype=np.int64)
            te = np.asarray(plan.predict_idx, dtype=np.int64)
            if tr.size > 0 and te.size > 0:
                splits.append((tr, te))
        if splits:
            return splits
    except Exception as e:
        logger.warning(f"SlicePlanner failed in simple splits: {e}. Falling back to simple chunking.")

    # Fallback to simple temporal sort splitting
    idx = np.argsort(timestamps)
    chunk_size = max(1, n_samples // n_splits)
    splits = []
    for i in range(n_splits):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < n_splits - 1 else n_samples
        test_idx = idx[start:end]
        train_idx = np.concatenate([idx[:start], idx[end:]]) if n_splits > 1 else test_idx
        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))
    return splits


def evaluate_signal(
    name: str,
    scores: np.ndarray,
    y_raw_net_return: np.ndarray,
    y_downside: np.ndarray,
    directionality: str
) -> Dict[str, Any]:
    """
    Evaluates a single signal or formula.
    Inverts the signal if it's risk-like so that higher score always means better expected outcome.
    """
    eval_scores = scores.copy()
    if directionality == "risk-like":
        # Lower risk is better, so invert it for evaluation
        eval_scores = -eval_scores

    metrics = {"head_name": name, "directionality": directionality}

    # Spearman with returns
    try:
        corr, _ = spearmanr(eval_scores, y_raw_net_return, nan_policy="omit")
        metrics["spearman_ret"] = float(corr) if pd.notna(corr) else 0.0
    except Exception:
        metrics["spearman_ret"] = 0.0

    # Top slice metrics
    top_metrics = compute_top_slice_metrics(eval_scores, y_raw_net_return, top_fracs=(0.1, 0.2))
    metrics.update(top_metrics)

    # Bucket monotonicity
    metrics["monotonicity"] = compute_bucket_monotonicity(eval_scores, y_raw_net_return, n_buckets=10)

    # Downside false safe
    # Note: For false safe, we want to know if "safe" predictions (high eval_score) lead to high downside.
    # We pass -eval_scores so that lower values mean "predicted safe" for the helper logic
    # which assumes 'lower predicted downside == safer'.
    metrics["false_safe_rate"] = compute_false_safe_rate(-eval_scores, y_downside, low_q=0.2, high_q=0.8)

    # Calculate simple utility score for ranking:
    # Reward high top 10% returns, high monotonicity, low false safe rate.
    # Normalizing top 10% returns heuristically
    top10_ret = metrics.get("top_10_mean_net", 0.0)
    mono = max(0.0, metrics["monotonicity"])
    fs_penalty = metrics["false_safe_rate"]

    # Very simple empirical utility proxy:
    utility = (np.sign(top10_ret) * (np.abs(top10_ret)**0.5) * 10) + mono - fs_penalty
    metrics["utility_score"] = float(utility)

    return metrics

def run_stage_1_diagnostics(
    feature_dict: Dict[str, np.ndarray],
    detected_heads: Dict[str, str],
    y_raw_net_return: np.ndarray,
    y_downside: np.ndarray
) -> pd.DataFrame:
    """Runs single-head diagnostics for all detected meta heads."""
    results = []
    for head_key, head_type in detected_heads.items():
        if head_key not in feature_dict:
            continue
        scores = feature_dict[head_key]
        if len(scores) != len(y_raw_net_return):
            continue

        metrics = evaluate_signal(head_key, scores, y_raw_net_return, y_downside, head_type)
        results.append(metrics)

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by="utility_score", ascending=False).reset_index(drop=True)
    return df


def build_combo_candidates(
    feature_dict: Dict[str, np.ndarray],
    detected_heads: Dict[str, str],
    lambda_grid: List[float] = [0.25, 0.5, 1.0, 2.0]
) -> Dict[str, np.ndarray]:
    """
    Generates a small family of fixed-form score combinations from available heads.
    Uses basic normalization to combine disparate scales safely.
    """
    candidates = {}

    # Organize heads by type
    edge_heads = [k for k, v in detected_heads.items() if v == "return-like" and k in feature_dict]
    mae_heads = [k for k, v in detected_heads.items() if v == "risk-like" and k in feature_dict]
    mfe_heads = [k for k, v in detected_heads.items() if v == "upside-like" and k in feature_dict]
    clf_heads = [k for k, v in detected_heads.items() if v == "classification-like" and k in feature_dict]
    asym_heads = [k for k, v in detected_heads.items() if v == "asymmetry-like" and k in feature_dict]

    def _norm(x):
        x_c, _, _, _, _ = clean_and_standardize(x)
        return x_c

    # 1. Base edge heads (always include if present)
    for eh in edge_heads:
        candidates[f"base_{eh}"] = _norm(feature_dict[eh])

    # 2. Edge - lambda * MAE
    if edge_heads and mae_heads:
        for eh in edge_heads:
            for mh in mae_heads:
                n_eh = _norm(feature_dict[eh])
                n_mh = _norm(feature_dict[mh])
                for lam in lambda_grid:
                    # Note: MAE is risk-like, so higher MAE score means higher predicted risk.
                    # We subtract lambda * MAE to penalize risk.
                    candidates[f"combo: {eh} - {lam}*{mh}"] = n_eh - lam * n_mh

    # 3. MFE - lambda * MAE
    if mfe_heads and mae_heads:
        for mfe_h in mfe_heads:
            for mae_h in mae_heads:
                n_mfe = _norm(feature_dict[mfe_h])
                n_mae = _norm(feature_dict[mae_h])
                for lam in lambda_grid:
                    candidates[f"combo: {mfe_h} - {lam}*{mae_h}"] = n_mfe - lam * n_mae

    # 4. Edge / (MAE + eps) style ratio (using standardized but shifted positive)
    if edge_heads and mae_heads:
        for eh in edge_heads:
            for mh in mae_heads:
                n_eh = _norm(feature_dict[eh])
                n_mh = _norm(feature_dict[mh])
                # Shift mae to be positive so ratio makes sense
                pos_mh = n_mh - np.min(n_mh) + 1.0
                candidates[f"combo: {eh} / ({mh}+eps)"] = n_eh / pos_mh

    # 5. Edge + alpha * classification
    if edge_heads and clf_heads:
        for eh in edge_heads:
            for ch in clf_heads:
                n_eh = _norm(feature_dict[eh])
                n_ch = _norm(feature_dict[ch])
                for lam in lambda_grid:
                    candidates[f"combo: {eh} + {lam}*{ch}"] = n_eh + lam * n_ch

    # 6. Include asym combinations if present
    if edge_heads and asym_heads:
        for eh in edge_heads:
            for ah in asym_heads:
                n_eh = _norm(feature_dict[eh])
                n_ah = _norm(feature_dict[ah])
                for lam in lambda_grid:
                    # Assuming higher asym prediction is better
                    candidates[f"combo: {eh} + {lam}*{ah}"] = n_eh + lam * n_ah

    return candidates

def run_stage_2_combo_race(
    candidates: Dict[str, np.ndarray],
    y_raw_net_return: np.ndarray,
    y_downside: np.ndarray,
    splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Runs race evaluation across all combinations."""
    results = []

    for name, scores in candidates.items():
        if len(scores) != len(y_raw_net_return):
            continue

        # Combos are pre-aligned so higher score = better expected outcome.
        # We pass directionality "return-like" because we built them that way.
        metrics = evaluate_signal(name, scores, y_raw_net_return, y_downside, directionality="return-like")

        # Calculate fold-level stability
        if splits:
            fold_spearmans = []
            for tr_idx, te_idx in splits:
                if len(te_idx) > 0:
                    corr, _ = spearmanr(scores[te_idx], y_raw_net_return[te_idx], nan_policy="omit")
                    if pd.notna(corr):
                        fold_spearmans.append(float(corr))
            if fold_spearmans:
                metrics["fold_spearman_mean"] = float(np.mean(fold_spearmans))
                metrics["fold_spearman_std"] = float(np.std(fold_spearmans))
            else:
                metrics["fold_spearman_mean"] = 0.0
                metrics["fold_spearman_std"] = 0.0

        # Rename head_name to combo_name for clarity
        metrics["combo_name"] = metrics.pop("head_name")
        results.append(metrics)

    df = pd.DataFrame(results)
    best_combo = {}
    if not df.empty:
        df = df.sort_values(by="utility_score", ascending=False).reset_index(drop=True)
        best_combo = df.iloc[0].to_dict()

    return df, best_combo


class SimpleHeadRidgeSizer:
    """
    A compact experimental component that tests if a linear model using
    only meta heads can beat fixed formulas.
    """
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha, fit_intercept=True)

    def fit_predict_oof(
        self,
        X: np.ndarray,
        y: np.ndarray,
        splits: List[Tuple[np.ndarray, np.ndarray]]
    ) -> np.ndarray:
        """Runs Out-Of-Fold predictions using temporal splits."""
        n_samples = len(y)
        oof_preds = np.zeros(n_samples)

        for tr_idx, te_idx in splits:
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_te = X[te_idx]

            # Fold-local scaling and NaN cleaning
            X_tr_clean, medians, scaler, mean_1d, std_1d = clean_and_standardize(X_tr)
            X_te_clean, _, _, _, _ = clean_and_standardize(X_te, fit_medians=medians, scaler=scaler, mean_1d=mean_1d, std_1d=std_1d)

            # Fit & predict
            self.model.fit(X_tr_clean, y_tr)
            oof_preds[te_idx] = self.model.predict(X_te_clean)

        return oof_preds

def evaluate_selection_profit_proxy(
    scores: np.ndarray,
    y_raw_net_return: np.ndarray,
    top_fracs: List[float] = [0.1, 0.2, 0.3],
    start_equity: float = 100000.0,
    cost_pct: float = 0.002
) -> pd.DataFrame:
    """
    Evaluates "Could this generate profit?" by applying a simple top-fraction selection rule.
    """
    results = []
    n_samples = len(scores)

    if n_samples == 0:
        return pd.DataFrame()

    for frac in top_fracs:
        k = max(1, int(n_samples * frac))
        idx = np.argpartition(scores, -k)[-k:]

        selected_rets = y_raw_net_return[idx]

        hit_rate = float(np.mean(selected_rets > 0)) if len(selected_rets) > 0 else 0.0

        # Simple sizing: equal size for selected trades
        # Assuming y_raw_net_return is already net of standard costs, but we apply an extra
        # cost proxy if requested or just use it as is.
        # Let's assume y_raw_net_return is exactly what we get per unit trade.
        sized_rets = selected_rets - cost_pct

        _, dd_series = _stable_equity_and_drawdown(sized_rets)
        mdd_pct = float(np.max(dd_series)) if len(dd_series) > 0 else 0.0
        net_pnl = float(np.sum(sized_rets))

        gross_profit = float(np.sum(sized_rets[sized_rets > 0]))
        gross_loss = float(np.abs(np.sum(sized_rets[sized_rets < 0])))
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float(gross_profit)

        results.append({
            "selection_frac": frac,
            "net_pnl": net_pnl,
            "hit_rate": hit_rate,
            "profit_factor": profit_factor,
            "max_drawdown": mdd_pct,
            "trades_selected": len(selected_rets)
        })

    return pd.DataFrame(results)

def run_simple_position_sizer(
    feature_dict: Dict[str, np.ndarray],
    trade_outcomes: pd.DataFrame,
    y_raw_net_return: np.ndarray,
    y_downside: np.ndarray,
    timestamps: np.ndarray,
    bucket_labels: Optional[np.ndarray] = None,
    sample_weight: Optional[np.ndarray] = None,
    start_equity: float = 100000.0,
    cost_pct: float = 0.002,
    lambda_grid: Optional[List[float]] = None,
    top_fracs: Tuple[float, ...] = (0.1, 0.2),
    use_ridge_head_sizer: bool = True
) -> Dict[str, Any]:
    """
    Main orchestrator for the simple position sizer diagnostic framework.
    """
    if lambda_grid is None:
        lambda_grid = [0.25, 0.5, 1.0, 2.0]

    # 1. Detect Meta Heads
    detected_heads = detect_meta_head_keys(feature_dict)
    used_keys = [k for k in detected_heads.keys() if k in feature_dict]
    missing_keys = [k for k in detected_heads.keys() if k not in feature_dict]

    feature_coverage_report = {
        "detected_candidates": list(detected_heads.keys()),
        "used_heads": used_keys,
        "missing_heads": missing_keys,
        "head_classification": detected_heads
    }

    # 2. Stage 1 Diagnostics
    stage_1_df = run_stage_1_diagnostics(feature_dict, detected_heads, y_raw_net_return, y_downside)

    # Determine temporal splits for stability checking and OOF Ridge
    n_samples = len(y_raw_net_return)
    splits = simple_temporal_splits(timestamps, n_samples)

    # 3. Stage 2 Combo Race
    combo_candidates = build_combo_candidates(feature_dict, detected_heads, lambda_grid)
    stage_2_df, best_combo = run_stage_2_combo_race(combo_candidates, y_raw_net_return, y_downside, splits)

    # Track the best score
    best_simple_score = None
    best_simple_score_name = None

    if not stage_2_df.empty:
        best_simple_score_name = best_combo["combo_name"]
        best_simple_score = combo_candidates[best_simple_score_name]

    # 4. Optional Ridge Sizer
    ridge_sizer_eval = {}
    if use_ridge_head_sizer and used_keys:
        n_samples = len(y_raw_net_return)
        # Assemble X from used heads
        X_heads = np.column_stack([feature_dict[k] for k in used_keys])

        # We also want to invert risk-like heads before fitting so Ridge
        # doesn't have to learn negative coefficients as hard, but Ridge can handle it.
        # However, to keep it simple, we just feed the standardized features.

        ridge = SimpleHeadRidgeSizer(alpha=1.0)
        ridge_oof_preds = ridge.fit_predict_oof(X_heads, y_raw_net_return, splits)

        ridge_metrics = evaluate_signal("Ridge_Head_Sizer", ridge_oof_preds, y_raw_net_return, y_downside, directionality="return-like")
        ridge_sizer_eval = ridge_metrics

        # Compare Ridge vs Best Combo
        if not best_combo or ridge_metrics.get("utility_score", 0) > best_combo.get("utility_score", -9999):
            best_simple_score = ridge_oof_preds
            best_simple_score_name = "Ridge_Head_Sizer"

    # 5. Profit Proxy on Best Score
    profit_proxy_df = pd.DataFrame()
    if best_simple_score is not None:
        profit_proxy_df = evaluate_selection_profit_proxy(
            best_simple_score,
            y_raw_net_return,
            top_fracs=list(top_fracs) + [0.3],
            start_equity=start_equity,
            cost_pct=cost_pct
        )

    return {
        "feature_coverage_report_": feature_coverage_report,
        "head_diagnostics_table_": stage_1_df,
        "combo_race_table_": stage_2_df,
        "best_combo_": best_combo,
        "ridge_sizer_eval_": ridge_sizer_eval,
        "best_simple_score_": best_simple_score,
        "best_simple_score_name_": best_simple_score_name,
        "profit_proxy_table_": profit_proxy_df
    }

def run_bucketed_simple_position_sizer(
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
    Runs the simple position sizer independently per bucket.
    """
    # Run global first
    global_results = run_simple_position_sizer(
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

        b_res = run_simple_position_sizer(
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


def run_simple_position_sizer_from_artifacts(
    data_root: str,
    run_id: str,
    top_fracs: Tuple[float, ...] = (0.1, 0.2),
    use_ridge_head_sizer: bool = True
) -> Dict[str, Any]:
    """
    Runs the simple position sizer directly on pipeline artifacts.
    Loads base model OOF predictions and filters strictly to the exact strategy mask
    (as optimized per-bucket) before running diagnostics.
    """
    from extreme_price_movements.run_ridge_sizer import load_trade_outcomes

    # Load dynamic strategies (which rules are active per bucket)
    strategies = load_inference_candidate_mask_params_per_bucket(top_n=1, ranking_metric="score_for_best_params")

    if not strategies:
        logger.warning("No strategies loaded from params_store.")
        return {}

    # Load base OOFs
    base_oofs = load_base_oof_predictions(data_root, run_id)
    if not base_oofs:
        logger.warning(f"No base OOFs found in {data_root}/artifacts/{run_id}/oof.")
        return {}

    # Gather data across buckets, specifically filtering to the allowed rule/mask
    all_returns = []
    all_downside = []
    all_timestamps = []
    all_bucket_labels = []

    global_feature_dict = {}

    bucket_dataframes = []

    for bucket, oof_df in base_oofs.items():
        # Find strategy corresponding to this bucket
        bucket_strategies = [s for s in strategies if f"{s['trade_side']}_{s['base_event_trigger']}" == bucket or s.get("strategy_id", "").startswith(bucket)]

        if not bucket_strategies:
            logger.info(f"Skipping bucket {bucket} (no matching strategy in registry).")
            continue

        strategy = bucket_strategies[0]
        strategy_id = strategy.get("strategy_id", "")

        # OOF DFS usually contain a mask column named mask_{strategy_id} or just 'mask'
        # Let's filter to rows where the mask is active if the column exists
        mask_col = f"mask_{strategy_id}"

        if mask_col in oof_df.columns:
            active_df = oof_df[oof_df[mask_col] == 1].copy()
        elif "mask" in oof_df.columns:
            # Fallback
            active_df = oof_df[oof_df["mask"] == 1].copy()
        else:
            # If no mask explicitly provided, use all (assumes pre-filtered or pure bucket logic)
            active_df = oof_df.copy()

        if active_df.empty:
            continue

        # Get target outcomes
        trade_outcomes = load_trade_outcomes(data_root, run_id, active_df)
        if trade_outcomes is None or "return" not in trade_outcomes.columns:
            continue

        # Identify columns to use as heads
        # Base models usually output things like base_H2, base_H4, etc. We will add them to feature_dict
        # We must filter by strategy_id if present to satisfy: "uses ONLY the outputs from models trained under the same strategy_id"
        head_cols = []
        for c in active_df.columns:
            # Typical columns we want to evaluate
            if c.startswith("base_") or "pred" in c.lower() or "score" in c.lower() or "mae" in c.lower() or "mfe" in c.lower():
                # If the column has a strategy_id appended (e.g. base_H2_StratX), we MUST match it
                # If there's no strategy suffix in the column name, we allow it (e.g., standard base_H2)
                if strategy_id and strategy_id in c:
                    head_cols.append(c)
                elif not any(s.get("strategy_id", "") in c for s in strategies if s.get("strategy_id")):
                    # It's a generic column not tied to *any* other strategy
                    head_cols.append(c)

        bucket_dataframes.append((bucket, active_df, trade_outcomes, head_cols))

    if not bucket_dataframes:
        logger.warning("No valid bucket dataframes constructed.")
        return {}

    # We now evaluate per bucket rather than globally mapping mismatched columns.
    # Since run_bucketed_simple_position_sizer normally takes a globally concatenated dataset,
    # let's construct it.

    # First, find all unique head columns across all active buckets
    all_head_cols = set()
    for _, _, _, head_cols in bucket_dataframes:
        all_head_cols.update(head_cols)

    # Concatenate
    combined_dfs = []
    combined_outcomes = []
    combined_buckets = []
    combined_timestamps = []

    for bucket, active_df, trade_outcomes, _ in bucket_dataframes:
        n_rows = len(active_df)
        combined_buckets.extend([bucket] * n_rows)
        combined_outcomes.append(trade_outcomes)

        ts = active_df["timestamp"].values if "timestamp" in active_df.columns else np.zeros(n_rows)
        combined_timestamps.append(ts)

        # Ensure all head cols exist, fill with NaN if missing for this bucket
        for hc in all_head_cols:
            if hc not in active_df.columns:
                active_df[hc] = np.nan
        combined_dfs.append(active_df[list(all_head_cols)])

    # Build global structures
    global_df = pd.concat(combined_dfs, ignore_index=True)
    global_outcomes = pd.concat(combined_outcomes, ignore_index=True)

    y_raw_net_return = global_outcomes["return"].values

    # Note: Downside might not be in trade_outcomes natively if not modeled. We proxy with 0.0 or actual if present.
    if "downside" in global_outcomes.columns:
        y_downside = global_outcomes["downside"].values
    elif "mae" in global_df.columns:
         y_downside = global_df["mae"].values
    else:
        # Fallback empty downside array
        y_downside = np.zeros_like(y_raw_net_return)

    timestamps = np.concatenate(combined_timestamps)
    bucket_labels = np.array(combined_buckets)

    feature_dict = {col: global_df[col].values for col in all_head_cols}

    return run_bucketed_simple_position_sizer(
        feature_dict=feature_dict,
        trade_outcomes=global_outcomes,
        y_raw_net_return=y_raw_net_return,
        y_downside=y_downside,
        timestamps=timestamps,
        bucket_labels=bucket_labels,
        top_fracs=top_fracs,
        use_ridge_head_sizer=use_ridge_head_sizer
    )
