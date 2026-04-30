"""
simple_policy_optimiser.py

A simple script to optimise Trailing Profit and Position sizing parameters using OOF predictions.
Requirements:
1. Use OOF preds / meta clf head from train_meta.
2. Optimise Trailing Profit (Optuna, 400 trials) and Position Sizing (Grid search).
3. Filter by top 15% preds by rank for each strategy_id.
4. Use unseen samples: divide 2/3 for optimisation, 1/3 for OOS validation, and load models to generate OOS predictions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import optuna

# NO EXTERNAL IMPORTS ALLOWED
# from extreme_price_movements.policy_optimiser import ...

logger = logging.getLogger(__name__)

# Parameter grids (moved to Optuna suggest variables directly below)

def compute_position_size(rank_pct: np.ndarray, size_power: float) -> np.ndarray:
    """Position size formula: size = 0.05 + (0.15 - 0.05) * rank_pct ** size_power"""
    return 0.05 + 0.10 * (rank_pct ** size_power)

def simulate_and_score(
    df_sub: pd.DataFrame, 
    f_opens: np.ndarray, f_highs: np.ndarray, f_lows: np.ndarray, f_closes: np.ndarray,
    cost_pct: float = 0.0015,
    size_power: float = 1.0,
    sl_mult: float = 1.0,
    trailing_activation_mult: float = 1.0,
    trailing_power: float = 1.5,
    trailing_squash_divisor: float = 2.0,
    giveback_beta: float = 0.5
) -> Dict[str, float]:
    """
    Fully self-contained, vectorized, bar-by-bar simulator.
    Checks TP/SL pessimistically, computes fees properly per trade.
    """
    n_trades, max_bars = f_opens.shape
    
    # 1. Entry
    entry_prices = f_opens[:, 0].copy() 
    
    # 2. Position sizing (dynamically scaled)
    sizes = compute_position_size(df_sub["rank_pct"].values, size_power)
    
    # 3. Side & Barriers
    side = np.ones(n_trades, dtype=np.float32)
    if "side" in df_sub.columns:
        side = df_sub["side"].values
        
    barrier = np.maximum(df_sub.get("barrier_pct", pd.Series(np.full(n_trades, 0.02))).values, 1e-4)
    barrier_price_dist = entry_prices * barrier
    
    is_long_arr = (side == 1)
    is_short_arr = (side == -1)
    
    sl_dist = barrier_price_dist * sl_mult
    tp_act = barrier_price_dist * trailing_activation_mult
    
    active = np.ones(n_trades, dtype=bool)
    exit_rets = np.zeros(n_trades, dtype=np.float32)
    max_favorable = np.zeros(n_trades, dtype=np.float32)
    
    # 4. Bar by Bar Simulation Loop
    for j in range(1, max_bars):
        active_idx = np.where(active)[0]
        if len(active_idx) == 0:
            break
            
        entry = entry_prices[active_idx]
        is_long_mask = is_long_arr[active_idx]
        is_short_mask = is_short_arr[active_idx]
        
        # 1. Check SL (Pessimistic: happens first)
        sl_hit_long = is_long_mask & (f_lows[active_idx, j] <= (entry - sl_dist[active_idx]))
        sl_hit_short = is_short_mask & (f_highs[active_idx, j] >= (entry + sl_dist[active_idx]))
        sl_hit = sl_hit_long | sl_hit_short
        
        # Update exits for hits
        hit_indices = active_idx[sl_hit]
        exit_rets[hit_indices] = - (sl_dist[hit_indices] / entry_prices[hit_indices])
        active[hit_indices] = False
        
        # Re-filter active
        active_idx = np.where(active)[0]
        if len(active_idx) == 0: break
        
        entry = entry_prices[active_idx]
        
        # 2. Check Trailing
        trail_active = max_favorable[active_idx] > tp_act[active_idx]
        
        dynamic_giveback = (max_favorable[active_idx] / (barrier_price_dist[active_idx] * trailing_squash_divisor)) ** trailing_power
        dynamic_giveback = np.clip(dynamic_giveback, 0.0, 1.0)
        trail_amount = max_favorable[active_idx] * giveback_beta * (1.0 - dynamic_giveback)
        
        trail_level_long = entry + (max_favorable[active_idx] - trail_amount)
        trail_level_short = entry - (max_favorable[active_idx] - trail_amount)
        
        trail_hit_long = is_long_arr[active_idx] & trail_active & (f_lows[active_idx, j] <= trail_level_long)
        trail_hit_short = is_short_arr[active_idx] & trail_active & (f_highs[active_idx, j] >= trail_level_short)
        trail_hit = trail_hit_long | trail_hit_short
        
        exit_rets[active_idx[trail_hit_long]] = (trail_level_long[trail_hit_long] - entry[trail_hit_long]) / entry[trail_hit_long]
        exit_rets[active_idx[trail_hit_short]] = (entry[trail_hit_short] - trail_level_short[trail_hit_short]) / entry[trail_hit_short]
        active[active_idx[trail_hit]] = False
        
        # 4. Update max_favorable
        cur_fav_long = f_highs[:, j] - entry_prices
        cur_fav_short = entry_prices - f_lows[:, j]
        cur_fav = np.where(is_long_arr, cur_fav_long, cur_fav_short)
        max_favorable = np.maximum(max_favorable, np.where(active, cur_fav, 0.0))

    # 5. Force exit remaining at max bars
    active_end = active
    if np.any(active_end):
        end_idx = np.flatnonzero(active_end)
        b_close = f_closes[end_idx, -1]
        v_ent = entry_prices[end_idx]
        v_s = side[end_idx]
        exit_rets[end_idx] = v_s * (b_close / v_ent - 1.0)

    # 6. Apply fees and compute net
    fees = sizes * cost_pct + sizes * (1 + exit_rets) * cost_pct
    gross_gain = sizes * exit_rets
    net_gain = gross_gain - fees

    return {
        "net_pnl": float(np.sum(net_gain)),
        "mean_net_trade": float(np.mean(net_gain)),
        "win_rate": float(np.mean(net_gain > 0)),
        "total_trades": len(net_gain),
        "raw_gains": net_gain,
        "sizes": sizes
    }

def optimise_position_sizing(
    df_sub: pd.DataFrame,
    f_opens: np.ndarray, f_highs: np.ndarray, f_lows: np.ndarray, f_closes: np.ndarray,
    cost_pct: float,
    best_trailing_params: dict
) -> Tuple[float, float, Dict[str, float]]:
    best_size_power = 1.0
    best_pnl = float("-inf")
    best_metrics = {}

    SIZE_POWER_GRID = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    for size_power in SIZE_POWER_GRID:
        metrics = simulate_and_score(
            df_sub, f_opens, f_highs, f_lows, f_closes,
            cost_pct=cost_pct,
            size_power=size_power,
            **best_trailing_params
        )
        if metrics["net_pnl"] > best_pnl:
            best_pnl = metrics["net_pnl"]
            best_size_power = size_power
            best_metrics = metrics

    return best_size_power, best_pnl, best_metrics

def calculate_advanced_metrics(df_sub: pd.DataFrame, raw_gains: np.ndarray, sizes: np.ndarray) -> dict:
    if len(raw_gains) == 0:
        return {}
    
    df_trades = pd.DataFrame({
        "timestamp": pd.to_datetime(df_sub["timestamp"].values),
        "net_gain": raw_gains,
        "size": sizes
    })
    df_trades = df_trades[np.isfinite(df_trades["net_gain"])]
    if len(df_trades) == 0:
        return {}

    df_trades = df_trades.sort_values("timestamp")
    df_trades.set_index("timestamp", inplace=True)
    
    start_date = df_trades.index.min()
    end_date = df_trades.index.max()
    n_trades = len(df_trades)
    
    avg_pnl_bankroll = df_trades["net_gain"].mean()
    df_trades["rop"] = df_trades["net_gain"] / df_trades["size"]
    avg_pnl_sized = df_trades["rop"].mean()
    
    hit_rate = (df_trades["net_gain"] > 0).mean()
    
    winning_trades = df_trades[df_trades["rop"] > 0]["rop"]
    losing_trades = df_trades[df_trades["rop"] < 0]["rop"]
    avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0.0
    avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0.0
    
    w_pnl = df_trades["net_gain"].resample("W").sum().fillna(0.0)
    m_pnl = df_trades["net_gain"].resample("ME").sum().fillna(0.0)
    
    w_std = w_pnl.std()
    m_std = m_pnl.std()
    
    def sortino(pnl_series):
        downside = pnl_series[pnl_series < 0]
        if len(downside) == 0 or downside.std(ddof=0) == 0:
            return 100.0 if pnl_series.mean() > 0 else 0.0
        return pnl_series.mean() / np.sqrt(np.mean(downside**2))
        
    w_sortino = sortino(w_pnl)
    m_sortino = sortino(m_pnl)
    
    cum_pnl = df_trades["net_gain"].cumsum()
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    max_dd = drawdown.min()
    
    tuw_max = pd.Timedelta(seconds=0)
    is_high = (drawdown == 0)
    if not is_high.all():
        high_dates = df_trades.index[is_high]
        if len(high_dates) > 0:
            all_highs = list(high_dates) + [df_trades.index[-1]]
            for i in range(1, len(all_highs)):
                dur = all_highs[i] - all_highs[i-1]
                if dur > tuw_max:
                    tuw_max = dur
        else:
            tuw_max = df_trades.index[-1] - df_trades.index[0]
            
    return {
        "start_date": str(start_date.date()),
        "end_date": str(end_date.date()),
        "n_trades": n_trades,
        "avg_pnl_bankroll": avg_pnl_bankroll,
        "avg_pnl_sized": avg_pnl_sized,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "hit_rate": hit_rate,
        "w_sortino": w_sortino,
        "m_sortino": m_sortino,
        "w_std": w_std,
        "m_std": m_std,
        "max_dd": max_dd,
        "tuw_days": tuw_max.total_seconds() / 86400.0
    }

def run_simple_policy_optimisation(
    data_root: str,
    run_id: str,
    cost_pct: float = 0.0015,
):
    meta_oof_dir = Path(data_root) / "artifacts" / run_id / "meta_oof"
    if not meta_oof_dir.exists():
        logger.error(f"Directory not found: {meta_oof_dir}")
        return

    meta_oof = {}
    for pq_file in meta_oof_dir.glob("*_tbm_clf.parquet"):
        strategy_id = pq_file.stem.replace("meta_oof_", "").replace("_tbm_clf", "")
        meta_oof[strategy_id] = pd.read_parquet(pq_file)

    if not meta_oof:
        logger.error(f"No _tbm_clf.parquet files found in {meta_oof_dir}")
        return

    from extreme_price_movements.data_store import PartitionedOHLCVStore
    ds = PartitionedOHLCVStore(data_root, timeframe="15m")

    
    results_json = {}

    for strategy_id, df in meta_oof.items():
        logger.info(f"Optimising strategy: {strategy_id}")

        if "clf" not in df.columns and "oof_p_tp" in df.columns:
            df["clf"] = df["oof_p_tp"]
        elif "clf" not in df.columns and "oof_pred" in df.columns:
            df["clf"] = df["oof_pred"]

        if "clf" not in df.columns:
            logger.warning(f"Strategy {strategy_id} has no valid clf or oof_p_tp score. Skipping.")
            continue

        df["rank_pct"] = df["clf"].rank(pct=True)
        
        if "side" not in df.columns:
            if strategy_id.startswith("short"):
                df["side"] = -1
            else:
                df["side"] = 1
        
        # Only evaluate on top 5% to speed up and focus on high-conviction trades
        df_top = df[df["rank_pct"] >= 0.95].copy()
        
        if "timestamp" in df_top.columns:
            df_top = df_top.sort_values("timestamp").reset_index(drop=True)
        else:
            df_top = df_top.sort_index().reset_index(drop=True)

        n = len(df_top)
        if n < 10:
            continue

        split_idx = int(n * 2 / 3)
        df_opt = df_top.iloc[:split_idx].copy()
        df_val = df_top.iloc[split_idx:].copy()

        logger.info(f"Optimisation set size: {len(df_opt)}, Validation set size: {len(df_val)}")

        def _fetch_paths(df_subset):
            n_events = len(df_subset)
            path_len = 96 # 24 hours at 15m resolution
            f_op = np.full((n_events, path_len), np.nan, dtype=np.float32)
            f_hi = np.full((n_events, path_len), np.nan, dtype=np.float32)
            f_lo = np.full((n_events, path_len), np.nan, dtype=np.float32)
            f_cl = np.full((n_events, path_len), np.nan, dtype=np.float32)
            
            for symbol, group in df_subset.groupby("symbol"):
                klines = ds.load(symbol)
                if klines is None or len(klines) == 0: continue
                klines = klines.reset_index()
                if "ts" not in klines.columns and "index" in klines.columns:
                    klines = klines.rename(columns={"index": "ts"})
                
                k_ts = klines["ts"].astype("int64").values // 10**6
                for df_idx, row in group.iterrows():
                    rel_idx = np.where(df_subset.index == df_idx)[0][0]
                    event_ts = int(pd.Timestamp(row["timestamp"]).timestamp() * 1000)
                    
                    idx_arr = np.searchsorted(k_ts, event_ts)
                    if idx_arr < len(k_ts):
                        end_idx = min(idx_arr + path_len, len(k_ts))
                        actual_len = end_idx - idx_arr
                        if actual_len > 0:
                            f_op[rel_idx, :actual_len] = klines["open"].values[idx_arr:end_idx]
                            f_hi[rel_idx, :actual_len] = klines["high"].values[idx_arr:end_idx]
                            f_lo[rel_idx, :actual_len] = klines["low"].values[idx_arr:end_idx]
                            f_cl[rel_idx, :actual_len] = klines["close"].values[idx_arr:end_idx]
            return f_op, f_hi, f_lo, f_cl

        opt_paths = _fetch_paths(df_opt)
        
        def objective(trial: optuna.Trial) -> Tuple[float, float]:
            sl_mult = trial.suggest_categorical("sl_mult", [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
            trailing_activation_mult = trial.suggest_categorical("trailing_activation_mult", [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5])
            trailing_power = trial.suggest_categorical("trailing_power", [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
            trailing_squash_divisor = trial.suggest_categorical("trailing_squash_divisor", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0])
            giveback_beta = trial.suggest_categorical("giveback_beta", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
            
            metrics = simulate_and_score(
                df_opt, *opt_paths, 
                cost_pct=cost_pct, size_power=1.0, 
                sl_mult=sl_mult, trailing_activation_mult=trailing_activation_mult,
                trailing_power=trailing_power, trailing_squash_divisor=trailing_squash_divisor, 
                giveback_beta=giveback_beta
            )
            adv = calculate_advanced_metrics(df_opt, metrics["raw_gains"], metrics["sizes"])
            w_std = adv.get("w_std", 10.0)
            m_std = adv.get("m_std", 10.0)
            avg_std = 0.5 * w_std + 0.5 * m_std
            
            return metrics["net_pnl"], avg_std

        study = optuna.create_study(directions=["maximize", "minimize"])
        study.optimize(objective, n_trials=400, show_progress_bar=False)

        best_trials = study.best_trials
        if not best_trials:
            logger.warning(f"Strategy {strategy_id} has no best trials.")
            continue
            
        # Top 5 by PnL
        best_trials_sorted_by_pnl = sorted(best_trials, key=lambda t: t.values[0], reverse=True)
        top_5_trials = best_trials_sorted_by_pnl[:min(5, len(best_trials_sorted_by_pnl))]
        
        pnls = np.array([t.values[0] for t in top_5_trials])
        stds = np.array([t.values[1] for t in top_5_trials])
        
        pnl_range = np.ptp(pnls) if np.ptp(pnls) > 0 else 1.0
        std_range = np.ptp(stds) if np.ptp(stds) > 0 else 1.0
        
        norm_pnls = (pnls - np.min(pnls)) / pnl_range
        norm_stds = (stds - np.min(stds)) / std_range
        
        scores = 0.5 * norm_pnls - 0.5 * norm_stds
        best_idx = np.argmax(scores)
        best_trial = top_5_trials[best_idx]
        
        best_params = best_trial.params
        logger.info(f"[{strategy_id}] Best Trailing Params (Pareto chosen): {best_params} with PnL {best_trial.values[0]:.2f} and Std {best_trial.values[1]:.4f}")

        # Validation Set Path Fetch & Evaluation
        val_paths = _fetch_paths(df_val)
        best_size_power, best_pnl, best_metrics = optimise_position_sizing(
            df_val, *val_paths, cost_pct=cost_pct, best_trailing_params=best_params
        )
        logger.info(f"[{strategy_id}] Best Size Power: {best_size_power}, Net PnL OOS: {best_pnl:.4f}")
        
        strategy_results = {
            "best_params": best_params,
            "best_size_power": float(best_size_power),
            "metrics": {}
        }
        
        # --- Advanced Metrics Generation ---
        for subset_name, subset_df, paths in [("train", df_opt, opt_paths), ("val", df_val, val_paths)]:
            strategy_results["metrics"][subset_name] = {}
            for top_pct, rank_thresh in [("top_10", 0.90), ("top_5", 0.95), ("top_1", 0.99)]:
                mask = subset_df["rank_pct"] >= rank_thresh
                if not mask.any():
                    continue
                
                # Filter subset and paths
                sub_filtered = subset_df[mask].copy()
                f_op, f_hi, f_lo, f_cl = paths
                mask_idx = np.where(mask)[0]
                
                f_op_f = f_op[mask_idx]
                f_hi_f = f_hi[mask_idx]
                f_lo_f = f_lo[mask_idx]
                f_cl_f = f_cl[mask_idx]
                
                # Re-simulate with best params
                metrics = simulate_and_score(
                    sub_filtered, f_op_f, f_hi_f, f_lo_f, f_cl_f,
                    cost_pct=cost_pct,
                    size_power=best_size_power,
                    **best_params
                )
                
                adv_metrics = calculate_advanced_metrics(sub_filtered, metrics.get("raw_gains", np.array([])), metrics.get("sizes", np.array([])))
                if adv_metrics:
                    strategy_results["metrics"][subset_name][top_pct] = adv_metrics
                    
                    logger.info(f"\n--- {strategy_id} | {subset_name} | {top_pct} ---")
                    logger.info(f"Period: {adv_metrics['start_date']} to {adv_metrics['end_date']}")
                    logger.info(f"Trades: {adv_metrics['n_trades']}")
                    logger.info(f"Net PnL/Trade (Bankroll): {adv_metrics['avg_pnl_bankroll'] * 100:.2f}%")
                    logger.info(f"Net PnL/Trade (Sized): {adv_metrics['avg_pnl_sized'] * 100:.2f}%")
                    logger.info(f"Avg Win: {adv_metrics['avg_win'] * 100:.2f}%, Avg Loss: {adv_metrics['avg_loss'] * 100:.2f}%")
                    logger.info(f"Hit Rate: {adv_metrics['hit_rate'] * 100:.1f}%")
                    logger.info(f"Sortino (W / M): {adv_metrics['w_sortino']:.2f} / {adv_metrics['m_sortino']:.2f}")
                    logger.info(f"Max DD: {adv_metrics['max_dd'] * 100:.2f}%")
                    logger.info(f"Time Under Water: {adv_metrics['tuw_days']:.1f} days")
                    logger.info(f"PnL Std (W / M): {adv_metrics['w_std'] * 100:.2f}% / {adv_metrics['m_std'] * 100:.2f}%")
                    
        results_json[strategy_id] = strategy_results

    output_path = meta_oof_dir.parent / "policy_optimisation.json"
    with open(output_path, "w") as f:
        json.dump(results_json, f, indent=4)
    logger.info(f"Saved policy optimisation results to {output_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/Users/remyroche/Documents/Ares/extreme_price_movements/data")
    parser.add_argument("--run_id", type=str, default=None)
    args = parser.parse_args()
    
    run_simple_policy_optimisation(args.data_root, args.run_id)


