from __future__ import annotations

import numpy as np
import pandas as pd


def _bucket3(x: pd.Series) -> pd.Series:
    q = x.rank(pct=True)
    return pd.cut(q, bins=[0, 1/3, 2/3, 1], labels=["low", "mid", "high"], include_lowest=True)


def evaluate_holdout(trades: pd.DataFrame, net_returns: np.ndarray) -> dict:
    n = len(trades)
    split = max(1, int(n * 0.30))
    test_idx = np.arange(split, n)
    test_ret = net_returns[test_idx]
    df = trades.iloc[test_idx].copy()
    df["net_return"] = test_ret

    out = {
        "holdout_trades": int(len(df)),
        "holdout_pnl_net": float(df["net_return"].sum()),
        "holdout_win_rate": float((df["net_return"] > 0).mean()) if len(df) else 0.0,
    }

    # Decile Metrics (Top 10, 20, 30%)
    if not df.empty and "confidence" in df.columns:
        decile_rows = []
        for pct in [0.90, 0.80, 0.70]: # Top 10%, 20%, 30%
            threshold = df["confidence"].quantile(pct)
            subset = df[df["confidence"] >= threshold]

            # Use round to ensure "Top10%" instead of "Top9%"
            decile_val = int(round((1.0 - pct) * 100))
            label = f"Top{decile_val}%"
            decile_rows.append({
                "decile": label,
                "n_trades": int(len(subset)),
                "pnl_net": float(subset["net_return"].sum()),
                "win_rate": float((subset["net_return"] > 0).mean()) if len(subset) else 0.0,
            })
        out["decile_metrics"] = decile_rows

    vol = df.get("realized_vol_12", pd.Series(np.abs(df["net_return"].rolling(12, min_periods=1).std())))
    volume = df.get("volume_12", pd.Series(np.ones(len(df))))
    trend = df.get("trend_12", pd.Series(df["net_return"].rolling(12, min_periods=1).mean()))

    df["regime_vol"] = _bucket3(vol)
    df["regime_volume"] = _bucket3(volume)
    df["regime_trend"] = _bucket3(trend)

    regime_rows = []
    for keys, g in df.groupby(["regime_vol", "regime_volume", "regime_trend"], dropna=False):
        regime_rows.append({
            "regime": "/".join([str(k) for k in keys]),
            "n_trades": int(len(g)),
            "pnl_net": float(g["net_return"].sum()),
            "win_rate": float((g["net_return"] > 0).mean()) if len(g) else 0.0,
        })

    out["regime_breakdown"] = regime_rows
    return out
