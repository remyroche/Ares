import numpy as np
import pandas as pd


def _bucket3(x: pd.Series) -> pd.Series:
    q = x.rank(pct=True, method='first')
    return pd.cut(q, bins=[0, 1/3, 2/3, 1], labels=["low", "mid", "high"], include_lowest=True)


def compute_comprehensive_metrics(trades: pd.DataFrame, fee_pct: float = 0.005, initial_capital: float = 100000.0) -> dict:
    """Computes comprehensive performance metrics for a set of trades.

    Args:
        trades: DataFrame with columns: 'entry_price', 'exit_price', 'is_long', 'confidence', 'timestamp', 'bucket', 'exit_reason' (optional).
        fee_pct: Per-trade fee (e.g., 0.005 for 0.5%).
        initial_capital: Starting capital for equity curve/drawdown.

    Returns:
        Dictionary of metrics (scalars).
    """
    if trades.empty:
        return {}

    # Calculate raw returns if not present
    if "raw_return" not in trades.columns:
        is_long = trades["is_long"].astype(int).to_numpy()
        raw_ret = np.where(is_long == 1,
                           (trades["exit_price"] - trades["entry_price"]) / trades["entry_price"],
                           (trades["entry_price"] - trades["exit_price"]) / trades["entry_price"])
        trades = trades.assign(raw_return=raw_ret)
    else:
        raw_ret = trades["raw_return"].to_numpy()

    # Calculate position size if not present (default to 1.0)
    if "pos_size" not in trades.columns:
        pos_size = np.ones(len(trades))
        trades = trades.assign(pos_size=pos_size)
    else:
        pos_size = trades["pos_size"].to_numpy()

    # Calculate net returns
    net_ret = (raw_ret * pos_size) - fee_pct
    trades = trades.assign(net_return=net_ret)

    # --- A) Core Performance (Net) ---
    pnl_net = float(np.sum(net_ret))

    # Sortino
    neg_ret = net_ret[net_ret < 0]
    std_neg = np.std(neg_ret) if len(neg_ret) > 0 else 0.0
    sortino = float(np.mean(net_ret) / std_neg) if std_neg > 1e-9 else 0.0

    # Gain-to-Pain
    gross_gains = np.sum(net_ret[net_ret > 0])
    gross_losses = np.abs(np.sum(net_ret[net_ret < 0]))
    gtp = float(gross_gains / gross_losses) if gross_losses > 1e-9 else 0.0
    profit_factor = gtp  # Same calculation for simple PnL based

    # PnL/day
    duration_days = 1.0
    if "timestamp" in trades.columns and len(trades) > 1:
        duration_days = (trades["timestamp"].max() - trades["timestamp"].min()).days
        pnl_per_day_pct = float(pnl_net / max(1, duration_days)) * 100  # percentage
    else:
        pnl_per_day_pct = 0.0

    # --- B) Tail Risk ---
    # Max Drawdown
    equity_curve = np.concatenate(([initial_capital], initial_capital * np.cumprod(1 + net_ret)))
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    max_dd = float(np.max(drawdown))

    # CVaR (95%)
    cvar_95 = float(np.mean(np.sort(net_ret)[:max(1, int(0.05 * len(net_ret)))]))

    # Worst Week
    worst_week = 0.0
    if "timestamp" in trades.columns and len(trades) > 1:
        trades_wk = trades.set_index("timestamp").resample("W-MON")["net_return"].sum()
        if not trades_wk.empty:
            worst_week = float(trades_wk.min())

    # --- C) Stability ---
    # 14-day block stats
    block_mean, block_std, block_iqr, block_hit_rate = 0.0, 0.0, 0.0, 0.0
    if "timestamp" in trades.columns and len(trades) > 1:
        block_14d = trades.set_index("timestamp").resample("14D")["net_return"].sum()
        if not block_14d.empty:
            block_mean = float(block_14d.mean())
            block_std = float(block_14d.std())
            block_iqr = float(block_14d.quantile(0.75) - block_14d.quantile(0.25))
            block_hit_rate = float((block_14d > 0).mean())

    # Performance Decay (slope of cumsum)
    slope = 0.0
    if len(net_ret) > 1:
        x = np.arange(len(net_ret))
        y = np.cumsum(net_ret)
        slope = float(np.polyfit(x, y, 1)[0])

    # --- D) Trading Microstructure ---
    trades_per_day = float(len(trades) / max(1, duration_days))

    # Holding Time
    holding_time_median_h = 0.0
    if "exit_timestamp" in trades.columns and "timestamp" in trades.columns:
        ht = (trades["exit_timestamp"] - trades["timestamp"]).dt.total_seconds() / 3600.0
        holding_time_median_h = float(ht.median())

    # Exit Reasons
    exit_counts = {}
    if "exit_reason" in trades.columns:
        counts = trades["exit_reason"].value_counts(normalize=True)
        for reason, freq in counts.items():
            exit_counts[f"exit_{reason}_pct"] = float(freq)

    # --- E) Payoff Distribution ---
    payoff_mean = float(np.mean(net_ret))
    payoff_median = float(np.median(net_ret))
    p5 = float(np.percentile(net_ret, 5))
    p25 = float(np.percentile(net_ret, 25))
    p75 = float(np.percentile(net_ret, 75))
    p95 = float(np.percentile(net_ret, 95))
    win_rate = float(np.mean(net_ret > 0))

    avg_win = np.mean(net_ret[net_ret > 0]) if np.any(net_ret > 0) else 0.0
    avg_loss = np.abs(np.mean(net_ret[net_ret < 0])) if np.any(net_ret < 0) else 1e-9
    win_loss_ratio = float(avg_win / avg_loss)

    # --- F) Confidence Calibration ---
    lift_top10, lift_bot10 = 0.0, 0.0
    if "confidence" in trades.columns and len(trades) > 10:
        top_10 = trades[trades["confidence"] >= trades["confidence"].quantile(0.9)]
        bot_10 = trades[trades["confidence"] <= trades["confidence"].quantile(0.1)]
        lift_top10 = float(top_10["net_return"].mean()) if not top_10.empty else 0.0
        lift_bot10 = float(bot_10["net_return"].mean()) if not bot_10.empty else 0.0

    metrics = {
        "net_pnl": pnl_net,
        "pnl_per_day_pct": pnl_per_day_pct,
        "sortino": sortino,
        "profit_factor": profit_factor,
        "max_dd": max_dd,
        "cvar_95": cvar_95,
        "worst_week": worst_week,
        "block_14d_mean": block_mean,
        "block_14d_std": block_std,
        "block_14d_iqr": block_iqr,
        "block_hit_rate": block_hit_rate,
        "perf_decay_slope": slope,
        "trades_per_day": trades_per_day,
        "holding_time_median_h": holding_time_median_h,
        "payoff_mean": payoff_mean,
        "payoff_median": payoff_median,
        "p5": p5,
        "p25": p25,
        "p75": p75,
        "p95": p95,
        "win_rate": win_rate,
        "win_loss_ratio": win_loss_ratio,
        "lift_top10": lift_top10,
        "lift_bot10": lift_bot10,
    }
    metrics.update(exit_counts)

    # --- G) Regime-conditional performance ---
    # Bucket regimes if not present
    if "regime_vol" not in trades.columns:
        if "realized_vol_12" in trades.columns:
            trades = trades.assign(regime_vol=_bucket3(trades["realized_vol_12"]))
        elif "net_return" in trades.columns and len(trades) > 12:
             vol_proxy = trades["net_return"].rolling(12, min_periods=1).std().abs()
             trades = trades.assign(regime_vol=_bucket3(vol_proxy))

    if "regime_trend" not in trades.columns:
        if "trend_12" in trades.columns:
            trades = trades.assign(regime_trend=_bucket3(trades["trend_12"]))
        elif "net_return" in trades.columns and len(trades) > 12:
             trend_proxy = trades["net_return"].rolling(12, min_periods=1).mean()
             trades = trades.assign(regime_trend=_bucket3(trend_proxy))

    # Add 'liquidity' and 'entropy' dummies if not present, to match request
    # Since we can't calculate them from scratch without features, we check for columns
    if "liquidity" in trades.columns and "regime_liq" not in trades.columns:
        trades = trades.assign(regime_liq=_bucket3(trades["liquidity"]))

    if "entropy" in trades.columns and "regime_ent" not in trades.columns:
         trades = trades.assign(regime_ent=_bucket3(trades["entropy"]))

    regime_cols = [c for c in trades.columns if c.startswith("regime_")]
    for rc in regime_cols:
        # For each regime column (e.g. regime_vol), compute stats per bucket
        try:
            for bucket_val, g in trades.groupby(rc):
                suffix = f"{rc}_{bucket_val}"
                g_pnl = float(g["net_return"].sum())
                g_count = len(g)
                g_days = duration_days # simple approx
                g_pnl_day = (g_pnl / max(1, g_days)) * 100

                # Sharpe (daily)
                g_mean = g["net_return"].mean()
                g_std = g["net_return"].std()
                g_sharpe = float(g_mean / g_std) if g_std > 1e-9 else 0.0

                g_dd = 0.0 # simplified maxdd per regime

                metrics[f"{suffix}_pnl_day"] = g_pnl_day
                metrics[f"{suffix}_sharpe"] = g_sharpe
                metrics[f"{suffix}_count"] = g_count
        except Exception:
            pass # skip failed groupings

    return metrics
