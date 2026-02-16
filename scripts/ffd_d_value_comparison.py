#!/usr/bin/env python3
"""Lightweight FFD d-value comparison across assets.

Runs a purged time-series CV Ridge probe on the 2-feature FFD template:
- ffd_diff_1_d
- ffd_diff_4_d

Outputs:
- CSV with per-asset/per-d metrics
- Markdown report with rankings and interpretation aids
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from extreme_price_movements.config import CFG
from extreme_price_movements.data_store import PartitionedOHLCVStore
from extreme_price_movements.frac_diff_adaptive import (
    compute_weight_window_sizes,
    frac_diff_ffd,
)


def _safe_log(series: pd.Series, eps: float = 1e-9) -> pd.Series:
    return np.log(np.maximum(series.astype(np.float64), eps))


def _ewma(series: pd.Series, span: int = 5) -> pd.Series:
    alpha = 2.0 / (span + 1.0)
    return series.ewm(alpha=alpha, adjust=False).mean()


def _compute_atr_ln(log_h: pd.Series, log_l: pd.Series, log_c: pd.Series, span: int = 14) -> pd.Series:
    prev_c = log_c.shift(1)
    tr = pd.concat(
        [
            (log_h - log_l),
            (log_h - prev_c).abs(),
            (log_l - prev_c).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / span, adjust=False).mean().clip(lower=1e-6)


def _zscore_train_apply(x_train: pd.DataFrame, x_val: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mu = x_train.mean(axis=0)
    sd = x_train.std(axis=0).replace(0.0, 1.0)
    return (x_train - mu) / sd, (x_val - mu) / sd


def _agg(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 0.0
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=0))


def _purged_cv_metrics(
    x: pd.DataFrame,
    y: pd.Series,
    atr_ln: pd.Series,
    range_over_atr: pd.Series,
    n_folds: int,
    purge_gap: int,
    ridge_alpha: float,
) -> dict:
    n = len(x)
    if n < 300:
        return {
            "ic_overall_mean": 0.0,
            "ic_overall_std": 0.0,
            "ic_event_mean": 0.0,
            "ic_event_std": 0.0,
            "ic_nonevent_mean": 0.0,
            "ic_nonevent_std": 0.0,
            "ic_ir_event": 0.0,
            "n_folds_used": 0,
        }

    fold_size = n // n_folds
    ic_overall: list[float] = []
    ic_event: list[float] = []
    ic_nonevent: list[float] = []

    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = n if fold == n_folds - 1 else min((fold + 1) * fold_size, n)

        train_end = val_start - purge_gap
        if train_end <= 100:
            continue

        x_train = x.iloc[:train_end]
        y_train = y.iloc[:train_end]
        x_val = x.iloc[val_start:val_end]
        y_val = y.iloc[val_start:val_end]

        atr_train = atr_ln.iloc[:train_end]
        atr_val = atr_ln.iloc[val_start:val_end]
        roa_train = range_over_atr.iloc[:train_end]
        roa_val = range_over_atr.iloc[val_start:val_end]

        tr_valid = (~x_train.isna().any(axis=1)) & y_train.notna()
        x_train = x_train.loc[tr_valid]
        y_train = y_train.loc[tr_valid]
        atr_train = atr_train.loc[tr_valid]
        roa_train = roa_train.loc[tr_valid]
        if len(x_train) < 100:
            continue

        x_train_z, x_val_z = _zscore_train_apply(x_train, x_val)
        val_valid = (~x_val_z.isna().any(axis=1)) & y_val.notna()
        if val_valid.sum() < 30:
            continue

        atr_thr = float(atr_train.quantile(0.8))
        roa_thr = float(roa_train.quantile(0.9))
        evt_val = (atr_val > atr_thr) | (roa_val > roa_thr)

        model = Ridge(alpha=ridge_alpha)
        model.fit(x_train_z, y_train)

        pred = pd.Series(np.nan, index=x_val_z.index, dtype=float)
        pred.loc[val_valid] = model.predict(x_val_z.loc[val_valid])

        all_mask = val_valid & pred.notna()
        if all_mask.sum() >= 30:
            ic_overall.append(float(spearmanr(pred[all_mask], y_val[all_mask]).correlation or 0.0))

        evt_mask = all_mask & evt_val.astype(bool)
        if evt_mask.sum() >= 20:
            ic_event.append(float(spearmanr(pred[evt_mask], y_val[evt_mask]).correlation or 0.0))

        ne_mask = all_mask & (~evt_val.astype(bool))
        if ne_mask.sum() >= 20:
            ic_nonevent.append(float(spearmanr(pred[ne_mask], y_val[ne_mask]).correlation or 0.0))

    o_mu, o_sd = _agg(ic_overall)
    e_mu, e_sd = _agg(ic_event)
    n_mu, n_sd = _agg(ic_nonevent)
    return {
        "ic_overall_mean": o_mu,
        "ic_overall_std": o_sd,
        "ic_event_mean": e_mu,
        "ic_event_std": e_sd,
        "ic_nonevent_mean": n_mu,
        "ic_nonevent_std": n_sd,
        "ic_ir_event": (e_mu / (e_sd + 1e-9)) if e_sd > 0 else (1e6 if e_mu > 0 else 0.0),
        "n_folds_used": len(ic_overall),
    }


def _discover_symbols(store: PartitionedOHLCVStore) -> list[str]:
    symbols = []
    if not os.path.exists(store.ohlcv_dir):
        return symbols
    for fname in os.listdir(store.ohlcv_dir):
        if not fname.endswith(".meta.json"):
            continue
        sym = fname.replace(".meta.json", "").replace("_", "/")
        symbols.append(sym)
    return sorted(set(symbols))


def _build_report(results: pd.DataFrame, k_info: dict, output_md: str) -> None:
    if results.empty:
        with open(output_md, "w", encoding="utf-8") as f:
            f.write("# FFD d-value Comparison Report\n\nNo results generated.\n")
        return

    summary_event = (
        results.groupby("d", as_index=False)
        .agg(
            ic_event_mean=("ic_event_mean", "mean"),
            ic_event_med=("ic_event_mean", "median"),
            ic_event_std_cross_asset=("ic_event_mean", "std"),
            ic_ir_event_mean=("ic_ir_event", "mean"),
            ic_overall_mean=("ic_overall_mean", "mean"),
            ic_nonevent_mean=("ic_nonevent_mean", "mean"),
            assets=("asset", "nunique"),
        )
        .sort_values("ic_ir_event_mean", ascending=False)
    )

    best_by_asset = (
        results.sort_values(["asset", "ic_ir_event"], ascending=[True, False])
        .groupby("asset", as_index=False)
        .first()[["asset", "d", "ic_event_mean", "ic_ir_event", "ic_overall_mean"]]
        .sort_values("ic_ir_event", ascending=False)
    )

    def _df_to_md(df: pd.DataFrame, floatfmt: str = ".6f") -> str:
        if df.empty:
            return "(empty)"
        cols = list(df.columns)
        rows = [cols]
        for _, row in df.iterrows():
            vals = []
            for c in cols:
                v = row[c]
                if isinstance(v, (float, np.floating)):
                    vals.append(format(float(v), floatfmt))
                else:
                    vals.append(str(v))
            rows.append(vals)

        widths = [0] * len(cols)
        for r in rows:
            for i, v in enumerate(r):
                widths[i] = max(widths[i], len(v))

        def _fmt_row(r: list[str]) -> str:
            return "| " + " | ".join(v.ljust(widths[i]) for i, v in enumerate(r)) + " |"

        header = _fmt_row(rows[0])
        sep = "| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |"
        body = "\n".join(_fmt_row(r) for r in rows[1:])
        return header + "\n" + sep + ("\n" + body if body else "")

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("# FFD d-value Comparison Report\n\n")
        f.write("## Weight Window Sizes K(d)\n\n")
        k_df = pd.DataFrame(
            [{"d": d, **vals} for d, vals in sorted(k_info.items(), key=lambda x: x[0])]
        )
        f.write(_df_to_md(k_df))
        f.write("\n\n")

        f.write("## Cross-Asset d Ranking (Event Regime Priority)\n\n")
        f.write(_df_to_md(summary_event, floatfmt=".6f"))
        f.write("\n\n")

        f.write("## Best d per Asset (by Event IC IR)\n\n")
        f.write(_df_to_md(best_by_asset, floatfmt=".6f"))
        f.write("\n\n")

        f.write("## Interpretation\n\n")
        f.write("- Higher `ic_event_mean` and `ic_ir_event_mean` suggest better discrimination in high-vol/high-range regimes.\n")
        f.write("- Compare `ic_event_mean` vs `ic_nonevent_mean` to decide whether a d is event-specialized or broad.\n")
        f.write("- Larger K(d) means longer memory and higher compute/warmup costs; include this in production trade-offs.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight FFD d-value comparison")
    parser.add_argument("--data-root", default=CFG.get("data_root", "data"))
    parser.add_argument("--timeframe", default=CFG.get("timeframe", "1h"))
    parser.add_argument("--symbols", default="", help="Comma-separated symbols, e.g. BTC/USDT,ETH/USDT")
    parser.add_argument("--max-symbols", type=int, default=30)
    parser.add_argument("--min-rows", type=int, default=3000)
    parser.add_argument("--lookback-days", type=int, default=365 * 2)
    parser.add_argument("--d-values", default="0.2,0.3,0.4,0.5,0.6")
    parser.add_argument("--label-horizon", type=int, default=24)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--out-dir", default="extreme_price_movements/reports")
    args = parser.parse_args()

    d_values = [float(x.strip()) for x in args.d_values.split(",") if x.strip()]
    d_values = sorted(set(d_values))

    store = PartitionedOHLCVStore(root_dir=args.data_root, timeframe=args.timeframe)
    if args.symbols.strip():
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = _discover_symbols(store)

    if args.max_symbols > 0:
        symbols = symbols[: args.max_symbols]

    now = pd.Timestamp.now(tz="UTC")
    start_ts = now - pd.Timedelta(days=args.lookback_days)

    k_info = compute_weight_window_sizes(d_values=d_values, thres=CFG.get("ffd_thres", 1e-5))
    warmup = max(v["warmup_bars"] for v in k_info.values()) if k_info else 0

    rows = []
    for sym in symbols:
        df = store.load(sym, start_ts=start_ts, end_ts=now)
        if df.empty or len(df) < args.min_rows:
            continue
        req_cols = {"high", "low", "close"}
        if not req_cols.issubset(df.columns):
            continue

        log_h = _safe_log(df["high"])
        log_l = _safe_log(df["low"])
        log_c = _safe_log(df["close"])
        log_c_ewm = _ewma(log_c, span=5)

        atr_ln = _compute_atr_ln(log_h, log_l, log_c)
        roa = (log_h - log_l) / (atr_ln + 1e-12)
        target = (log_c.shift(-args.label_horizon) - log_c).astype(np.float64)

        for d in d_values:
            ffd_c = frac_diff_ffd(log_c_ewm, d=d, thres=CFG.get("ffd_thres", 1e-5))
            x = pd.DataFrame(
                {
                    "ffd_diff_1": ffd_c.diff(1),
                    "ffd_diff_4": ffd_c.diff(4),
                },
                index=ffd_c.index,
            )

            x = x.iloc[warmup:]
            y = target.reindex(x.index)
            atr_a = atr_ln.reindex(x.index)
            roa_a = roa.reindex(x.index)

            valid = y.notna()
            x = x.loc[valid]
            y = y.loc[valid]
            atr_a = atr_a.loc[valid]
            roa_a = roa_a.loc[valid]

            if len(x) < args.min_rows:
                continue

            metrics = _purged_cv_metrics(
                x=x,
                y=y,
                atr_ln=atr_a,
                range_over_atr=roa_a,
                n_folds=args.n_folds,
                purge_gap=args.label_horizon,
                ridge_alpha=args.ridge_alpha,
            )
            rows.append({
                "asset": sym,
                "d": d,
                "n_rows": len(x),
                "warmup_bars": warmup,
                "K_d": int(k_info[d]["K"]),
                **metrics,
            })

    out_df = pd.DataFrame(rows)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, ts)
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "ffd_d_value_comparison.csv")
    md_path = os.path.join(out_dir, "ffd_d_value_comparison_report.md")
    k_csv_path = os.path.join(out_dir, "ffd_weight_window_sizes.csv")

    out_df.to_csv(csv_path, index=False)
    pd.DataFrame([{"d": d, **vals} for d, vals in sorted(k_info.items(), key=lambda x: x[0])]).to_csv(
        k_csv_path, index=False
    )
    _build_report(out_df, k_info, md_path)

    print(f"Saved comparison CSV: {csv_path}")
    print(f"Saved weight windows CSV: {k_csv_path}")
    print(f"Saved markdown report: {md_path}")


if __name__ == "__main__":
    main()
