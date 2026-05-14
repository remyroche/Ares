"""
compare_ridge_vs_extratrees.py

Head-to-head comparison script for Ridge vs ExtraTrees position sizers.
Runs both models on a basket of 10 assets and generates detailed comparison metrics.
"""

from __future__ import annotations

import logging
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_rel, wilcoxon

from extreme_price_movements.data_store import read_parquet_projected

# Import both sizers
from extreme_price_movements.simple_position_sizer import (
    run_simple_position_sizer_from_artifacts,
    SimpleHeadRidgeSizer,
    run_simple_position_sizer,
)
from extreme_price_movements.extratrees_position_sizer import (
    run_extratrees_position_sizer_from_artifacts,
    SimpleHeadExtraTreesSizer,
    run_extratrees_position_sizer,
)
from extreme_price_movements.run_ridge_sizer import (
    load_base_oof_predictions,
    load_meta_oof_predictions,
    load_trade_outcomes,
)
from extreme_price_movements.offline_optimisers.params_store import (
    load_inference_candidate_mask_params_per_bucket,
)
from extreme_price_movements.src_utils_tprint import tprint

logger = logging.getLogger(__name__)


# Default basket of 10 liquid crypto assets for testing
DEFAULT_BASKET_10 = [
    "BTCUSDT",
    "ETHUSDT", 
    "SOLUSDT",
    "ADAUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "DOTUSDT",
    "LINKUSDT",
    "AVAXUSDT",
    "MATICUSDT",
]


@dataclass
class ModelComparisonMetrics:
    """Metrics for comparing Ridge vs ExtraTrees performance."""
    # Identification
    strategy_id: str
    symbol: str
    
    # Spearman correlation metrics
    ridge_spearman_ret: float
    extratrees_spearman_ret: float
    spearman_diff: float
    spearman_winner: str
    
    # Top slice metrics (top 10% mean return)
    ridge_top10_mean: float
    extratrees_top10_mean: float
    top10_diff: float
    top10_winner: str
    
    # Monotonicity
    ridge_monotonicity: float
    extratrees_monotonicity: float
    mono_diff: float
    mono_winner: str
    
    # Utility score
    ridge_utility: float
    extratrees_utility: float
    utility_diff: float
    utility_winner: str
    
    # Profit proxy metrics
    ridge_wallet_pnl: float
    extratrees_wallet_pnl: float
    wallet_pnl_diff: float
    wallet_pnl_winner: str
    
    ridge_hit_rate: float
    extratrees_hit_rate: float
    hit_rate_diff: float
    hit_rate_winner: str
    
    ridge_profit_factor: float
    extratrees_profit_factor: float
    pf_diff: float
    pf_winner: str
    
    ridge_sortino: float
    extratrees_sortino: float
    sortino_diff: float
    sortino_winner: str
    
    # Model characteristics
    ridge_n_features: int
    extratrees_n_features: int
    
    # Overall assessment
    overall_winner: str
    win_count_ridge: int
    win_count_et: int


def extract_profit_proxy_metrics(profit_proxy_df: pd.DataFrame) -> Dict[str, float]:
    """Extract key metrics from profit proxy table."""
    if profit_proxy_df.empty:
        return {
            "wallet_pnl": 0.0,
            "net_pnl": 0.0,
            "hit_rate": 0.0,
            "profit_factor": 0.0,
            "sortino": 0.0,
            "trades_per_day": 0.0,
        }
    
    # Get optimal row
    if "is_optimal" in profit_proxy_df.columns:
        opt_row = profit_proxy_df[profit_proxy_df["is_optimal"]].iloc[0]
    else:
        opt_row = profit_proxy_df.iloc[0]
        
    return {
        "wallet_pnl": opt_row.get("wallet_pnl", 0.0),
        "net_pnl": opt_row.get("net_pnl", 0.0),
        "hit_rate": opt_row.get("hit_rate", 0.0),
        "profit_factor": opt_row.get("profit_factor", 0.0),
        "sortino": opt_row.get("sortino", 0.0),
        "trades_per_day": opt_row.get("trades_per_day", 0.0),
    }


def run_comparison_on_strategy(
    data_root: str,
    run_id: str,
    strategy: Dict[str, Any],
    base_oofs: Dict[str, pd.DataFrame],
    meta_oofs: Dict[str, pd.DataFrame],
    et_hyperparams: Dict[str, int],
) -> Optional[ModelComparisonMetrics]:
    """
    Runs both Ridge and ExtraTrees on a single strategy and returns comparison metrics.
    """
    import re as _re
    
    strategy_id = strategy.get("strategy_id", "")
    if not strategy_id:
        return None
        
    # Determine symbol from data (will extract from OOF or labels after loading)
    symbol = "PENDING"
        
    # Load labels (same logic as both sizers)
    labels_dir = Path(data_root) / "artifacts" / run_id / "labels"
    full_df = pd.DataFrame()
    label_file = None
    
    if labels_dir.exists():
        all_label_files = list(labels_dir.glob("train_*.parquet"))
        
        def normalize(s):
            return _re.sub(r'[^a-z0-9]', '', s.lower())
            
        target_norm = normalize(strategy_id)
        
        for f in all_label_files:
            if "_tight" in f.name or "_wide" in f.name or "_balanced" in f.name:
                continue
            f_name_norm = normalize(f.stem.replace("train_", ""))
            if target_norm in f_name_norm or f_name_norm in target_norm:
                label_file = f
                break
                
        if not label_file:
            tokens = set(_re.split(r'[^a-z0-9]', strategy_id.lower()))
            tokens.discard('')
            max_overlap = 0
            best_match = None
            for f in all_label_files:
                if "_tight" in f.name or "_wide" in f.name or "_balanced" in f.name:
                    continue
                f_tokens = set(_re.split(r'[^a-z0-9]', f.stem.lower().replace("train_", "")))
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
            # Extract symbol from data
            unique_symbols = full_df["__symbol__"].dropna().unique()
            if len(unique_symbols) == 1:
                symbol = unique_symbols[0]
            elif len(unique_symbols) > 1:
                symbol = str(len(unique_symbols)) + "_assets"
        if "__index__" in full_df.columns:
            full_df["index"] = full_df["__index__"]
            
    trade_side = strategy.get('trade_side', '')
    
    # Resolve OOF
    oof_df = pd.DataFrame()
    resolved_meta_key = None
    
    prefixed = f"{trade_side}_{strategy_id}" if trade_side else strategy_id
    if prefixed in meta_oofs:
        resolved_meta_key = prefixed
    else:
        def _strip_side(k):
            return _re.sub(r'^(long|short)_', '', k)
        strat_norm = _re.sub(r'[^a-z0-9]', '', strategy_id.lower())
        best_key, best_score = None, 0
        for mk in meta_oofs.keys():
            mk_norm = _re.sub(r'[^a-z0-9]', '', _strip_side(mk).lower())
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
        
    # Join OOF onto labels
    if not full_df.empty:
        if not oof_df.empty and "index" in oof_df.columns:
            join_cols = [c for c in ["timestamp", "symbol"] if c in full_df.columns and c in oof_df.columns]
            if join_cols:
                oof_clean = oof_df.drop(columns=[c for c in ["return", "y_ret", "y_bin"] if c in full_df.columns], errors="ignore")
                if "timestamp" in join_cols:
                    for _df in [full_df, oof_clean]:
                        if "timestamp" in _df.columns and hasattr(_df["timestamp"].dtype, "tz") and _df["timestamp"].dt.tz is not None:
                            _df["timestamp"] = _df["timestamp"].dt.tz_localize(None)
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
        return None
    
    # Extract symbol from OOF data if still pending
    if symbol == "PENDING" and not oof_df.empty and "symbol" in oof_df.columns:
        unique_symbols = oof_df["symbol"].dropna().unique()
        if len(unique_symbols) == 1:
            symbol = unique_symbols[0]
        elif len(unique_symbols) > 1:
            symbol = str(len(unique_symbols)) + "_assets"
    if symbol == "PENDING":
        symbol = "UNKNOWN"
        
    # Load trade outcomes
    trade_outcomes = load_trade_outcomes(data_root, run_id, active_df)
    if trade_outcomes is None or "return" not in trade_outcomes.columns:
        return None
        
    # Filter to scored rows
    from extreme_price_movements.simple_position_sizer import collect_ridge_head_columns
    
    _oof_score_cols = [c for c in active_df.columns if c in ("oof_prob", "oof_pred", "reg", "clf") or c.startswith(("tbm_", "mae_h", "mfe_h"))]
    if _oof_score_cols:
        scorable_mask = active_df[_oof_score_cols[0]].notna()
        active_scored_df = active_df[scorable_mask].copy()
    else:
        scorable_mask = np.ones(len(active_df), dtype=bool)
        active_scored_df = active_df.copy()
        
    if len(active_scored_df) < 30:
        return None
        
    y_raw_net_return = trade_outcomes["return"].values
    if y_raw_net_return.size == 0:
        return None
        
    if "downside" in trade_outcomes.columns:
        y_downside = trade_outcomes["downside"].values
    elif "mae" in active_scored_df.columns:
        y_downside = active_scored_df["mae"].values
    else:
        y_downside = np.zeros_like(y_raw_net_return)
        
    timestamps = active_scored_df["timestamp"].values if "timestamp" in active_scored_df.columns else np.zeros(len(y_raw_net_return))
    
    # Extract feature dict
    _HINDSIGHT_COLS = {
        "return", "is_long", "y_ret", "y_bin", "exit_code", "bars_to_mfe",
        "mae_ret", "mfe_ret", "u_policy", "u_policy_net",
    }
    def _is_hindsight(col: str) -> bool:
        return col in _HINDSIGHT_COLS or (col.startswith("__") and col.endswith("__"))
        
    head_cols = [c for c in collect_ridge_head_columns(active_scored_df) if not _is_hindsight(c)]
    head_cols = [c for c in head_cols if active_scored_df[c].notna().any()]
    
    if not head_cols:
        return None
        
    feature_dict = {col: active_scored_df[col].values for col in head_cols}
    
    # Get temporal splits
    from extreme_price_movements.simple_position_sizer import walk_forward_temporal_splits
    n_samples = len(y_raw_net_return)
    splits = walk_forward_temporal_splits(timestamps, n_samples, n_splits=5)
    
    # --- Run Ridge ---
    from extreme_price_movements.simple_position_sizer import SimpleHeadRidgeSizer, evaluate_signal
    from sklearn.linear_model import Ridge
    
    X_heads = np.column_stack([feature_dict[k] for k in head_cols])
    ridge_sizer = SimpleHeadRidgeSizer(model=Ridge(alpha=1.0))
    ridge_oof_preds = ridge_sizer.fit_predict_oof(X_heads, y_raw_net_return, splits, feature_names=head_cols)
    ridge_metrics = evaluate_signal("Ridge", ridge_oof_preds, y_raw_net_return, y_downside, directionality="return-like")
    
    # --- Run ExtraTrees ---
    from sklearn.ensemble import ExtraTreesRegressor
    from extreme_price_movements.extratrees_position_sizer import SimpleHeadExtraTreesSizer
    
    et_model = ExtraTreesRegressor(
        n_estimators=et_hyperparams.get("n_estimators", 150),
        max_depth=et_hyperparams.get("max_depth", 7),
        min_samples_leaf=et_hyperparams.get("min_samples_leaf", 30),
        min_samples_split=et_hyperparams.get("min_samples_split", 60),
        max_features="sqrt",
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
    )
    et_sizer = SimpleHeadExtraTreesSizer(model=et_model, calibration_method="isotonic")
    et_oof_preds = et_sizer.fit_predict_oof(X_heads, y_raw_net_return, splits, feature_names=head_cols)
    et_metrics = evaluate_signal("ExtraTrees", et_oof_preds, y_raw_net_return, y_downside, directionality="return-like")
    
    # --- Extract profit proxy metrics ---
    from extreme_price_movements.simple_position_sizer import evaluate_selection_profit_proxy
    
    t_diff = np.max(timestamps) - np.min(timestamps)
    n_days = float(t_diff) / 86400.0 if len(timestamps) > 1 else 0.0
    
    ridge_pp_df, _, _ = evaluate_selection_profit_proxy(
        ridge_oof_preds, y_raw_net_return, timestamps=timestamps,
        top_fracs=[0.05, 0.1, 0.15, 0.2], cost_pct=0.003, n_days=n_days
    )
    et_pp_df, _, _ = evaluate_selection_profit_proxy(
        et_oof_preds, y_raw_net_return, timestamps=timestamps,
        top_fracs=[0.05, 0.1, 0.15, 0.2], cost_pct=0.003, n_days=n_days
    )
    
    ridge_pp = extract_profit_proxy_metrics(ridge_pp_df)
    et_pp = extract_profit_proxy_metrics(et_pp_df)
    
    # --- Calculate comparison ---
    ridge_top10 = ridge_metrics.get("top_10_mean_net", 0.0)
    et_top10 = et_metrics.get("top_10_mean_net", 0.0)
    
    ridge_mono = ridge_metrics.get("monotonicity", 0.0)
    et_mono = et_metrics.get("monotonicity", 0.0)
    
    ridge_util = ridge_metrics.get("utility_score", 0.0)
    et_util = et_metrics.get("utility_score", 0.0)
    
    ridge_spear = ridge_metrics.get("spearman_ret", 0.0)
    et_spear = et_metrics.get("spearman_ret", 0.0)
    
    # Determine winners
    spearman_winner = "Ridge" if ridge_spear > et_spear else "ExtraTrees"
    top10_winner = "Ridge" if ridge_top10 > et_top10 else "ExtraTrees"
    mono_winner = "Ridge" if ridge_mono > et_mono else "ExtraTrees"
    utility_winner = "Ridge" if ridge_util > et_util else "ExtraTrees"
    wallet_winner = "Ridge" if ridge_pp["wallet_pnl"] > et_pp["wallet_pnl"] else "ExtraTrees"
    hit_winner = "Ridge" if ridge_pp["hit_rate"] > et_pp["hit_rate"] else "ExtraTrees"
    pf_winner = "Ridge" if ridge_pp["profit_factor"] > et_pp["profit_factor"] else "ExtraTrees"
    sortino_winner = "Ridge" if ridge_pp["sortino"] > et_pp["sortino"] else "ExtraTrees"
    
    # Count wins
    win_count_ridge = sum([
        spearman_winner == "Ridge",
        top10_winner == "Ridge",
        mono_winner == "Ridge",
        utility_winner == "Ridge",
        wallet_winner == "Ridge",
        hit_winner == "Ridge",
        pf_winner == "Ridge",
        sortino_winner == "Ridge",
    ])
    win_count_et = 8 - win_count_ridge
    
    overall_winner = "Ridge" if win_count_ridge > win_count_et else ("ExtraTrees" if win_count_et > win_count_ridge else "Tie")
    
    return ModelComparisonMetrics(
        strategy_id=strategy_id[:60],
        symbol=symbol,
        ridge_spearman_ret=ridge_spear,
        extratrees_spearman_ret=et_spear,
        spearman_diff=et_spear - ridge_spear,
        spearman_winner=spearman_winner,
        ridge_top10_mean=ridge_top10,
        extratrees_top10_mean=et_top10,
        top10_diff=et_top10 - ridge_top10,
        top10_winner=top10_winner,
        ridge_monotonicity=ridge_mono,
        extratrees_monotonicity=et_mono,
        mono_diff=et_mono - ridge_mono,
        mono_winner=mono_winner,
        ridge_utility=ridge_util,
        extratrees_utility=et_util,
        utility_diff=et_util - ridge_util,
        utility_winner=utility_winner,
        ridge_wallet_pnl=ridge_pp["wallet_pnl"],
        extratrees_wallet_pnl=et_pp["wallet_pnl"],
        wallet_pnl_diff=et_pp["wallet_pnl"] - ridge_pp["wallet_pnl"],
        wallet_pnl_winner=wallet_winner,
        ridge_hit_rate=ridge_pp["hit_rate"],
        extratrees_hit_rate=et_pp["hit_rate"],
        hit_rate_diff=et_pp["hit_rate"] - ridge_pp["hit_rate"],
        hit_rate_winner=hit_winner,
        ridge_profit_factor=ridge_pp["profit_factor"],
        extratrees_profit_factor=et_pp["profit_factor"],
        pf_diff=et_pp["profit_factor"] - ridge_pp["profit_factor"],
        pf_winner=pf_winner,
        ridge_sortino=ridge_pp["sortino"],
        extratrees_sortino=et_pp["sortino"],
        sortino_diff=et_pp["sortino"] - ridge_pp["sortino"],
        sortino_winner=sortino_winner,
        ridge_n_features=len(head_cols),
        extratrees_n_features=len(head_cols),
        overall_winner=overall_winner,
        win_count_ridge=win_count_ridge,
        win_count_et=win_count_et,
    )


def run_comparison_on_basket(
    data_root: str,
    run_id: str,
    basket: List[str] = None,
    top_n_strategies: int = 4,
    et_hyperparams: Dict[str, int] = None,
) -> pd.DataFrame:
    """
    Runs Ridge vs ExtraTrees comparison on a basket of assets.
    
    Args:
        data_root: Path to data directory
        run_id: Pipeline run ID to analyze
        basket: List of symbols (defaults to DEFAULT_BASKET_10)
        top_n_strategies: Number of top strategies to evaluate per asset
        et_hyperparams: ExtraTrees hyperparameters
        
    Returns:
        DataFrame with comparison metrics for all strategies
    """
    if basket is None:
        basket = DEFAULT_BASKET_10
    if et_hyperparams is None:
        et_hyperparams = {
            "n_estimators": 150,
            "max_depth": 7,
            "min_samples_leaf": 30,
            "min_samples_split": 60,
        }
        
    tprint(f"Running Ridge vs ExtraTrees comparison")
    tprint(f"Basket (reference): {basket}")
    tprint(f"Run ID: {run_id}")
    tprint(f"ExtraTrees hyperparameters: {et_hyperparams}")
    
    # Load strategies and OOFs once
    _pool = load_inference_candidate_mask_params_per_bucket(top_n=99, ranking_metric="score_for_best_params")
    
    if not _pool:
        tprint("No strategies loaded from params_store.")
        return pd.DataFrame()
        
    # Deduplicate and select top strategies
    _seen_ids: set = set()
    all_strategies = []
    for s in _pool:
        sid = s.get("strategy_id", "")
        if sid and sid not in _seen_ids:
            _seen_ids.add(sid)
            all_strategies.append(s)
    
    # Take top-N strategies overall (or top-N per side if available)
    long_strategies = [s for s in all_strategies if s.get('trade_side', '') == 'long']
    short_strategies = [s for s in all_strategies if s.get('trade_side', '') == 'short']
    
    selected_strategies = []
    selected_strategies.extend(long_strategies[:top_n_strategies])
    selected_strategies.extend(short_strategies[:top_n_strategies])
    
    # If no side info, just take top overall
    if not selected_strategies:
        selected_strategies = all_strategies[:top_n_strategies * 2]
        
    tprint(f"Selected {len(selected_strategies)} strategies for comparison")
    
    # Load OOFs
    base_oofs = load_base_oof_predictions(data_root, run_id)
    try:
        meta_oofs = load_meta_oof_predictions(data_root, run_id)
    except Exception as e:
        tprint(f"Could not load meta OOFs: {e}")
        meta_oofs = {}
        
    # Run comparison for each strategy
    results: List[ModelComparisonMetrics] = []
    
    for i, strategy in enumerate(selected_strategies):
        tprint(f"[{i+1}/{len(selected_strategies)}] Comparing: {strategy.get('strategy_id', '')[:50]}...")
        
        try:
            comparison = run_comparison_on_strategy(
                data_root=data_root,
                run_id=run_id,
                strategy=strategy,
                base_oofs=base_oofs,
                meta_oofs=meta_oofs,
                et_hyperparams=et_hyperparams,
            )
            if comparison:
                results.append(comparison)
        except Exception as e:
            tprint(f"  Error processing strategy: {e}")
            continue
            
    if not results:
        tprint("No valid comparison results produced.")
        return pd.DataFrame()
        
    # Convert to DataFrame
    results_df = pd.DataFrame([asdict(r) for r in results])
    
    return results_df


def print_comparison_report(results_df: pd.DataFrame) -> None:
    """Prints a formatted comparison report."""
    if results_df.empty:
        print("No results to report.")
        return
        
    print("\n" + "=" * 120)
    print(" RIDGE vs EXTRATREES POSITION SIZER - COMPARISON REPORT")
    print("=" * 120)
    
    # Summary statistics
    n_strategies = len(results_df)
    ridge_wins = results_df[results_df["overall_winner"] == "Ridge"].shape[0]
    et_wins = results_df[results_df["overall_winner"] == "ExtraTrees"].shape[0]
    ties = n_strategies - ridge_wins - et_wins
    
    print(f"\nOverall Results:")
    print(f"  Total Strategies Evaluated: {n_strategies}")
    print(f"  Ridge Wins: {ridge_wins} ({100*ridge_wins/n_strategies:.1f}%)")
    print(f"  ExtraTrees Wins: {et_wins} ({100*et_wins/n_strategies:.1f}%)")
    print(f"  Ties: {ties} ({100*ties/n_strategies:.1f}%)")
    
    # Metric-by-metric breakdown
    metrics = [
        ("Spearman Correlation", "spearman_winner"),
        ("Top 10% Mean Return", "top10_winner"),
        ("Monotonicity", "mono_winner"),
        ("Utility Score", "utility_winner"),
        ("Wallet PnL", "wallet_pnl_winner"),
        ("Hit Rate", "hit_rate_winner"),
        ("Profit Factor", "pf_winner"),
        ("Sortino Ratio", "sortino_winner"),
    ]
    
    print(f"\nMetric-by-Metric Wins:")
    for metric_name, col_name in metrics:
        ridge_metric_wins = (results_df[col_name] == "Ridge").sum()
        et_metric_wins = (results_df[col_name] == "ExtraTrees").sum()
        print(f"  {metric_name:25s}: Ridge {ridge_metric_wins:2d} vs ExtraTrees {et_metric_wins:2d}")
        
    # Mean differences
    print(f"\nMean Differences (ExtraTrees - Ridge):")
    print(f"  Spearman Correlation: {results_df['spearman_diff'].mean():+.4f}")
    print(f"  Top 10% Mean Return:  {results_df['top10_diff'].mean():+.4f}")
    print(f"  Monotonicity:         {results_df['mono_diff'].mean():+.4f}")
    print(f"  Utility Score:        {results_df['utility_diff'].mean():+.4f}")
    print(f"  Wallet PnL:           {results_df['wallet_pnl_diff'].mean():+.4f}")
    print(f"  Hit Rate:             {results_df['hit_rate_diff'].mean():+.4f}")
    print(f"  Profit Factor:        {results_df['pf_diff'].mean():+.4f}")
    print(f"  Sortino Ratio:        {results_df['sortino_diff'].mean():+.4f}")
    
    # Top performers
    print(f"\nTop 5 Ridge Performers (by Wallet PnL):")
    top_ridge = results_df.nlargest(5, "ridge_wallet_pnl")[["strategy_id", "symbol", "ridge_wallet_pnl", "ridge_hit_rate"]]
    print(top_ridge.to_string(index=False))
    
    print(f"\nTop 5 ExtraTrees Performers (by Wallet PnL):")
    top_et = results_df.nlargest(5, "extratrees_wallet_pnl")[["strategy_id", "symbol", "extratrees_wallet_pnl", "extratrees_hit_rate"]]
    print(top_et.to_string(index=False))
    
    # Per-symbol summary
    print(f"\nPer-Symbol Summary:")
    symbol_summary = results_df.groupby("symbol").agg({
        "overall_winner": lambda x: x.value_counts().index[0],  # Most common winner
        "ridge_wallet_pnl": "mean",
        "extratrees_wallet_pnl": "mean",
        "wallet_pnl_diff": "mean",
        "ridge_hit_rate": "mean",
        "extratrees_hit_rate": "mean",
    }).round(4)
    print(symbol_summary.to_string())
    
    print("\n" + "=" * 120)


def save_comparison_results(results_df: pd.DataFrame, output_path: Path) -> None:
    """Saves comparison results to CSV and JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save full results as CSV
    csv_path = output_path.with_suffix(".csv")
    results_df.to_csv(csv_path, index=False)
    tprint(f"Saved comparison CSV to: {csv_path}")
    
    # Save summary as JSON
    summary = {
        "n_strategies": len(results_df),
        "ridge_wins": int((results_df["overall_winner"] == "Ridge").sum()),
        "extratrees_wins": int((results_df["overall_winner"] == "ExtraTrees").sum()),
        "ties": int((results_df["overall_winner"] == "Tie").sum()),
        "mean_spearman_diff": float(results_df["spearman_diff"].mean()),
        "mean_wallet_pnl_diff": float(results_df["wallet_pnl_diff"].mean()),
        "mean_hit_rate_diff": float(results_df["hit_rate_diff"].mean()),
        "mean_pf_diff": float(results_df["pf_diff"].mean()),
        "mean_sortino_diff": float(results_df["sortino_diff"].mean()),
    }
    
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    tprint(f"Saved summary JSON to: {json_path}")


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s:%(name)s:%(message)s'
    )
    
    parser = argparse.ArgumentParser(description="Compare Ridge vs ExtraTrees Position Sizers")
    parser.add_argument("--run-id", type=str, help="Run ID to analyze (defaults to latest)")
    parser.add_argument("--data-root", type=str, default=".", help="Root directory for data/artifacts")
    parser.add_argument("--basket", type=str, nargs="+", default=None, 
                        help=f"List of symbols to analyze (default: {DEFAULT_BASKET_10})")
    parser.add_argument("--top-n", type=int, default=4, help="Top N strategies per asset")
    parser.add_argument("--output-dir", type=str, default="./comparison_output", help="Output directory")
    
    # ExtraTrees hyperparameters
    parser.add_argument("--et-n-estimators", type=int, default=150)
    parser.add_argument("--et-max-depth", type=int, default=7)
    parser.add_argument("--et-min-samples-leaf", type=int, default=30)
    
    args = parser.parse_args()
    
    data_root = args.data_root
    run_id = args.run_id
    
    if not run_id:
        from extreme_price_movements.run_ridge_sizer import find_latest_run_id
        try:
            run_id = find_latest_run_id(data_root)
            tprint(f"Detected latest run: {run_id}")
        except Exception as e:
            tprint(f"Error detecting latest run: {e}")
            sys.exit(1)
            
    basket = args.basket if args.basket else DEFAULT_BASKET_10
    
    et_hyperparams = {
        "n_estimators": args.et_n_estimators,
        "max_depth": args.et_max_depth,
        "min_samples_leaf": args.et_min_samples_leaf,
        "min_samples_split": 60,
    }
    
    try:
        results_df = run_comparison_on_basket(
            data_root=data_root,
            run_id=run_id,
            basket=basket,
            top_n_strategies=args.top_n,
            et_hyperparams=et_hyperparams,
        )
        
        if results_df.empty:
            tprint("No comparison results produced.")
            sys.exit(0)
            
        # Print report
        print_comparison_report(results_df)
        
        # Save results
        output_path = Path(args.output_dir) / f"ridge_vs_extratrees_{run_id}"
        save_comparison_results(results_df, output_path)
        
    except KeyboardInterrupt:
        tprint("Execution interrupted by user.")
    except Exception as e:
        tprint(f"CRITICAL ERROR: {e}")
        import traceback
        tprint(traceback.format_exc())
        sys.exit(1)
