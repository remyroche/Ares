"""
Per-bucket / per-horizon detailed reports for each pipeline step.

Saves to extreme_price_movements/reports/<run_id>/bucket_report_<step>.md
and a companion CSV for machine consumption.

Steps covered:
  - compare_tbm   : geometry grid quality per cell
  - labels        : label distribution per (bucket, horizon)
  - base_training : alpha model AUC/IC per (bucket, horizon)
  - meta_training : meta model IC/AUC per (bucket, horizon)
  - ridge_sizer   : ridge weights + IC per bucket
  - optimise      : backtest PnL/Sharpe/WR per bucket
"""

from __future__ import annotations

import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

REPORTS_DIR = Path(__file__).parent

_BUCKETS = ["MR_long", "MR_short", "TF_long", "TF_short"]
_HORIZONS = [2, 4, 8]
_CELL_KEYS = [f"{b}_H{h}" for b in _BUCKETS for h in _HORIZONS]


def _ensure_dir(run_id: str) -> Path:
    d = REPORTS_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fmt(v, decimals=4):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def _pct(v, decimals=1):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v * 100:.{decimals}f}%"


def _md_table(headers: List[str], rows: List[List[str]]) -> List[str]:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return lines


def _save(run_id: str, step: str, lines: List[str], df: Optional[pd.DataFrame] = None) -> str:
    out = _ensure_dir(run_id)
    md_path = out / f"bucket_report_{step}.md"
    md_path.write_text("\n".join(lines))
    if df is not None and not df.empty:
        csv_path = out / f"bucket_report_{step}.csv"
        df.to_csv(csv_path, index=False)
    return str(md_path)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: compare_tbm — geometry grid quality
# ─────────────────────────────────────────────────────────────────────────────
def report_compare_tbm(run_id: str, grid_csv_path: str) -> str:
    """Generate per-cell geometry quality report from tbm_geometry_grid.csv."""
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"# TBM Geometry Grid Report — {run_id}", f"Generated: {ts}\n"]

    try:
        df = pd.read_csv(grid_csv_path)
    except Exception as e:
        lines.append(f"ERROR: Could not load grid CSV: {e}")
        return _save(run_id, "compare_tbm", lines)

    lines.append(f"**Grid CSV**: `{grid_csv_path}`")
    lines.append(f"**Total rows**: {len(df)}  |  **Cells covered**: {df['cell_key'].nunique() if 'cell_key' in df.columns else '?'} / 12\n")

    # Per-cell summary table
    lines.append("## Per-Cell Geometry Quality")
    headers = ["Cell", "N configs", "k_tp range", "sl range", "Best AUC", "Best TP-sep", "Best timeout", "Best bind", "Fallback?"]
    rows = []
    for ck in _CELL_KEYS:
        sub = df[df["cell_key"] == ck] if "cell_key" in df.columns else pd.DataFrame()
        if sub.empty:
            rows.append([ck, "0", "—", "—", "—", "—", "—", "—", "⚠️ MISSING"])
            continue
        n = len(sub)
        ktp_vals = sub["k_tp"].dropna()
        sl_vals = sub["sl_as_tp_pct"].dropna()
        ktp_rng = f"{ktp_vals.min():.2f}–{ktp_vals.max():.2f}" if len(ktp_vals) else "—"
        sl_rng = f"{sl_vals.min():.2f}–{sl_vals.max():.2f}" if len(sl_vals) else "—"
        best_auc = _fmt(sub["cell_auc"].max() if "cell_auc" in sub.columns else float("nan"))
        best_sep = _fmt(sub["cell_tp_sep"].max() if "cell_tp_sep" in sub.columns else float("nan"))
        best_to = _fmt(sub["cell_timeout"].min() if "cell_timeout" in sub.columns else float("nan"))
        best_bind = _fmt(sub["cell_bind"].max() if "cell_bind" in sub.columns else float("nan"))
        is_fallback = "✓ fallback" if (sub["rank"] == 99).any() else ""
        rows.append([ck, n, ktp_rng, sl_rng, best_auc, best_sep, best_to, best_bind, is_fallback])
    lines.extend(_md_table(headers, rows))
    lines.append("")

    # Per-bucket summary
    lines.append("## Per-Bucket Summary (across H2/H4/H8)")
    bkt_headers = ["Bucket", "Cells populated", "Median AUC", "Median TP-sep", "Median bind"]
    bkt_rows = []
    for bkt in _BUCKETS:
        bsub = df[df["bucket"] == bkt] if "bucket" in df.columns else pd.DataFrame()
        cells_pop = bsub["cell_key"].nunique() if not bsub.empty else 0
        med_auc = _fmt(bsub["cell_auc"].median() if not bsub.empty and "cell_auc" in bsub.columns else float("nan"))
        med_sep = _fmt(bsub["cell_tp_sep"].median() if not bsub.empty and "cell_tp_sep" in bsub.columns else float("nan"))
        med_bind = _fmt(bsub["cell_bind"].median() if not bsub.empty and "cell_bind" in bsub.columns else float("nan"))
        bkt_rows.append([bkt, f"{cells_pop}/3", med_auc, med_sep, med_bind])
    lines.extend(_md_table(bkt_headers, bkt_rows))
    lines.append("")

    return _save(run_id, "compare_tbm", lines, df)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: labels — label distribution per (bucket, horizon)
# ─────────────────────────────────────────────────────────────────────────────
def report_labels(run_id: str, data_root: str, cfg: Dict[str, Any]) -> str:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"# Label Generation Report — {run_id}", f"Generated: {ts}\n"]
    horizons = cfg.get("label_horizons_hours", [2, 4, 8])
    sides = ["long", "short"]
    kinds = ["mr", "tf"]

    rows_data = []
    table_rows = []
    headers = ["Dataset (side_kind_H)", "N rows", "TP%", "SL%", "Timeout%", "Bind%", "Balance", "TP/SL ratio"]

    for side in sides:
        for kind in kinds:
            for H in horizons:
                name = f"train_{side}_{kind}_{H}"
                from extreme_price_movements.data_store import load_artifact_df
                try:
                    df = load_artifact_df(data_root, run_id, "labels", name)
                except Exception:
                    df = None
                if df is None or df.empty:
                    table_rows.append([name, "0", "—", "—", "—", "—", "—", "—"])
                    continue
                n = len(df)
                if "label" not in df.columns:
                    table_rows.append([name, str(n), "no label col", "—", "—", "—", "—", "—"])
                    continue
                tp = float((df["label"] == 1).mean())
                sl = float((df["label"] == -1).mean())
                to = float((df["label"] == 0).mean())
                bind = tp + sl
                bal = min(tp, sl) / max(max(tp, sl), 1e-9)
                tpsl = tp / max(sl, 1e-9)
                table_rows.append([name, f"{n:,}", _pct(tp), _pct(sl), _pct(to), _pct(bind), _fmt(bal, 3), _fmt(tpsl, 3)])
                rows_data.append({"dataset": name, "side": side, "kind": kind, "H": H, "n": n,
                                   "tp_pct": tp, "sl_pct": sl, "timeout_pct": to, "bind": bind,
                                   "balance": bal, "tp_over_sl": tpsl})

    lines.append("## Label Distribution per (side, kind, horizon)")
    lines.extend(_md_table(headers, table_rows))
    lines.append("")

    # Per-bucket aggregation (MR_long = long_mr, TF_long = long_tf, etc.)
    _bucket_map = {"MR_long": ("long", "mr"), "MR_short": ("short", "mr"),
                   "TF_long": ("long", "tf"), "TF_short": ("short", "tf")}
    lines.append("## Per-Bucket Summary (median across horizons)")
    bkt_headers = ["Bucket", "Total N", "Median TP%", "Median SL%", "Median Timeout%", "Median Bind%"]
    bkt_rows = []
    for bkt, (s, k) in _bucket_map.items():
        bkt_data = [r for r in rows_data if r["side"] == s and r["kind"] == k]
        if not bkt_data:
            bkt_rows.append([bkt, "0", "—", "—", "—", "—"])
            continue
        total_n = sum(r["n"] for r in bkt_data)
        med_tp = float(np.median([r["tp_pct"] for r in bkt_data]))
        med_sl = float(np.median([r["sl_pct"] for r in bkt_data]))
        med_to = float(np.median([r["timeout_pct"] for r in bkt_data]))
        med_bind = float(np.median([r["bind"] for r in bkt_data]))
        bkt_rows.append([bkt, f"{total_n:,}", _pct(med_tp), _pct(med_sl), _pct(med_to), _pct(med_bind)])
    lines.extend(_md_table(bkt_headers, bkt_rows))
    lines.append("")

    out_df = pd.DataFrame(rows_data) if rows_data else pd.DataFrame()
    return _save(run_id, "labels", lines, out_df)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: base_training — alpha model AUC/IC per (bucket, horizon)
# ─────────────────────────────────────────────────────────────────────────────
def report_base_training(run_id: str, bundle: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"# Base Training Report — {run_id}", f"Generated: {ts}\n"]
    horizons = cfg.get("label_horizons_hours", [2, 4, 8])
    sides = ["long", "short"]
    kinds = ["mr", "tf"]

    alpha_models = bundle.get("alpha_models", {}) if bundle else {}
    alpha_metrics = bundle.get("alpha_oof_metrics", {}) if bundle else {}

    rows_data = []
    table_rows = []
    headers = ["Bucket", "H", "Winner algo", "AUC (raw)", "AUC (weighted)", "IC", "Prec@10", "N features"]

    for side in sides:
        for kind in kinds:
            bkt_label = f"{'MR' if kind == 'mr' else 'TF'}_{'long' if side == 'long' else 'short'}"
            side_models = alpha_models.get(side, {})
            kind_model = side_models.get(kind, {})
            for H in horizons:
                cell_key = f"{bkt_label}_H{H}"
                # Alpha metrics are keyed by (side, kind, H)
                m_key = f"{side}/{kind}/H={H}"
                m = alpha_metrics.get(m_key, {})
                winner = m.get("winner", kind_model.get("winner_algo", "—"))
                auc_raw = _fmt(m.get("auc_raw", float("nan")))
                auc_w = _fmt(m.get("auc_weighted", float("nan")))
                ic = _fmt(m.get("ic", float("nan")))
                prec10 = _fmt(m.get("prec_at_10", float("nan")))
                n_feats = m.get("n_features", kind_model.get("n_features", "—"))
                table_rows.append([bkt_label, H, winner, auc_raw, auc_w, ic, prec10, n_feats])
                rows_data.append({"bucket": bkt_label, "H": H, "winner": winner,
                                   "auc_raw": m.get("auc_raw"), "auc_weighted": m.get("auc_weighted"),
                                   "ic": m.get("ic"), "prec_at_10": m.get("prec_at_10")})

    lines.append("## Alpha Model Performance per (Bucket, Horizon)")
    lines.extend(_md_table(headers, table_rows))
    lines.append("")

    # Per-bucket summary
    lines.append("## Per-Bucket Summary (median across horizons)")
    bkt_headers = ["Bucket", "Deployed Hs", "Primary H", "Median AUC (weighted)", "Median IC"]
    bkt_rows = []
    _bucket_map = {"MR_long": ("long", "mr"), "MR_short": ("short", "mr"),
                   "TF_long": ("long", "tf"), "TF_short": ("short", "tf")}
    for bkt, (s, k) in _bucket_map.items():
        side_models = alpha_models.get(s, {})
        kind_model = side_models.get(k, {})
        deployed = kind_model.get("deployed_horizons", horizons)
        primary_h = kind_model.get("H", "—")
        bkt_data = [r for r in rows_data if r["bucket"] == bkt and r["auc_weighted"] is not None]
        med_auc = _fmt(float(np.median([r["auc_weighted"] for r in bkt_data]))) if bkt_data else "—"
        med_ic = _fmt(float(np.median([r["ic"] for r in bkt_data if r["ic"] is not None]))) if bkt_data else "—"
        bkt_rows.append([bkt, str(deployed), str(primary_h), med_auc, med_ic])
    lines.extend(_md_table(bkt_headers, bkt_rows))
    lines.append("")

    out_df = pd.DataFrame(rows_data) if rows_data else pd.DataFrame()
    return _save(run_id, "base_training", lines, out_df)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: meta_training — meta model IC/AUC per (bucket, horizon)
# ─────────────────────────────────────────────────────────────────────────────
def report_meta_training(run_id: str, data_root: str, bundle: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"# Meta Training Report — {run_id}", f"Generated: {ts}\n"]

    meta_oof_dir = Path(data_root) / "artifacts" / run_id / "meta_oof"
    rows_data = []

    # Load all meta OOF parquet files
    oof_files = sorted(meta_oof_dir.glob("meta_oof_*.parquet")) if meta_oof_dir.exists() else []
    lines.append(f"**Meta OOF files found**: {len(oof_files)}\n")

    table_rows = []
    headers = ["Model", "N samples", "IC (payoff)", "AUC", "Prec@10", "Mean pred", "Std pred"]

    for pf in oof_files:
        model_name = pf.stem.replace("meta_oof_", "")
        try:
            df = pd.read_parquet(pf)
        except Exception:
            table_rows.append([model_name, "ERR", "—", "—", "—", "—", "—"])
            continue
        n = len(df)
        if "oof_pred" not in df.columns:
            table_rows.append([model_name, str(n), "no oof_pred", "—", "—", "—", "—"])
            continue
        pred = df["oof_pred"].values.astype(float)
        mean_p = _fmt(float(np.nanmean(pred)))
        std_p = _fmt(float(np.nanstd(pred)))
        # IC on return if available
        ic = float("nan")
        auc = float("nan")
        prec10 = float("nan")
        if "return" in df.columns:
            ret = df["return"].values.astype(float)
            valid = np.isfinite(pred) & np.isfinite(ret)
            if valid.sum() >= 10:
                from scipy.stats import spearmanr
                ic = float(spearmanr(pred[valid], ret[valid]).correlation)
        if "label" in df.columns:
            y = (df["label"].values == 1).astype(float)
            n_pos = int(y.sum()); n_neg = len(y) - n_pos
            if n_pos > 0 and n_neg > 0:
                ranks = pd.Series(pred).rank(method="average").to_numpy(float)
                u = ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0
                auc = float(u / (n_pos * n_neg))
            if len(pred) >= 10:
                top10 = pred >= np.quantile(pred, 0.90)
                prec10 = float(y[top10].mean()) if top10.any() else float("nan")
        table_rows.append([model_name, f"{n:,}", _fmt(ic), _fmt(auc), _fmt(prec10), mean_p, std_p])
        rows_data.append({"model": model_name, "n": n, "ic": ic, "auc": auc,
                           "prec_at_10": prec10, "mean_pred": float(np.nanmean(pred))})

    lines.append("## Meta OOF Predictions per Model")
    lines.extend(_md_table(headers, table_rows))
    lines.append("")

    # Per-bucket summary: group by base bucket (strip _H{n} / _clf suffix)
    import re
    _h_pat = re.compile(r'^(.+)_H\d+$')
    _clf_pat = re.compile(r'^(.+)_clf$')
    bucket_groups: Dict[str, List[Dict]] = {}
    for r in rows_data:
        m = _h_pat.match(r["model"])
        if m:
            bkt = m.group(1)
        elif _clf_pat.match(r["model"]):
            bkt = _clf_pat.match(r["model"]).group(1)
        else:
            bkt = r["model"]
        bucket_groups.setdefault(bkt, []).append(r)

    lines.append("## Per-Bucket Summary")
    bkt_headers = ["Bucket", "Models", "Median IC", "Median AUC", "Median Prec@10"]
    bkt_rows = []
    for bkt in sorted(bucket_groups.keys()):
        grp = bucket_groups[bkt]
        ics = [r["ic"] for r in grp if r["ic"] is not None and not math.isnan(r["ic"])]
        aucs = [r["auc"] for r in grp if r["auc"] is not None and not math.isnan(r["auc"])]
        precs = [r["prec_at_10"] for r in grp if r["prec_at_10"] is not None and not math.isnan(r["prec_at_10"])]
        bkt_rows.append([bkt, len(grp),
                         _fmt(float(np.median(ics))) if ics else "—",
                         _fmt(float(np.median(aucs))) if aucs else "—",
                         _fmt(float(np.median(precs))) if precs else "—"])
    lines.extend(_md_table(bkt_headers, bkt_rows))
    lines.append("")

    out_df = pd.DataFrame(rows_data) if rows_data else pd.DataFrame()
    return _save(run_id, "meta_training", lines, out_df)


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: ridge_sizer — weights + IC per bucket
# ─────────────────────────────────────────────────────────────────────────────
def report_ridge_sizer(run_id: str, result: Dict[str, Any]) -> str:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"# Ridge Position Sizer Report — {run_id}", f"Generated: {ts}\n"]

    if not result:
        lines.append("No ridge sizer results available.")
        return _save(run_id, "ridge_sizer", lines)

    rows_data = []
    directions = result.get("directions", {})
    all_metrics = result.get("metrics", {})
    all_weights = result.get("weights", {})

    for direction, dir_res in directions.items():
        buckets = dir_res if isinstance(dir_res, list) else dir_res.get("buckets", [])
        lines.append(f"## Direction: {direction.upper()}")
        dir_metrics = all_metrics.get(direction, {})

        table_rows = []
        headers = ["Bucket", "IC (train)", "IC (val)", "Sharpe", "N weights", "Top weight"]
        for bkt in buckets:
            m = dir_metrics.get(bkt, {})
            ic_train = _fmt(m.get("ic_train", float("nan")))
            ic_val = _fmt(m.get("ic_val", float("nan")))
            sharpe = _fmt(m.get("sharpe", float("nan")))
            bkt_weights = {k: v for k, v in all_weights.items() if k.startswith(bkt + "_")}
            n_w = len(bkt_weights)
            top_w = max(bkt_weights.items(), key=lambda x: abs(x[1]), default=("—", float("nan")))
            top_w_str = f"{top_w[0].replace(bkt+'_', '')}={top_w[1]:.4f}" if top_w[0] != "—" else "—"
            table_rows.append([bkt, ic_train, ic_val, sharpe, n_w, top_w_str])
            rows_data.append({"direction": direction, "bucket": bkt,
                               "ic_train": m.get("ic_train"), "ic_val": m.get("ic_val"),
                               "sharpe": m.get("sharpe"), "n_weights": n_w})
        lines.extend(_md_table(headers, table_rows))
        lines.append("")

        # Weight breakdown per bucket
        lines.append(f"### Weight Breakdown — {direction.upper()}")
        for bkt in buckets:
            bkt_weights = {k.replace(bkt + "_", ""): v for k, v in all_weights.items() if k.startswith(bkt + "_")}
            if bkt_weights:
                lines.append(f"**{bkt}**: " + ", ".join(f"`{k}`={v:.4f}" for k, v in sorted(bkt_weights.items())))
        lines.append("")

    out_df = pd.DataFrame(rows_data) if rows_data else pd.DataFrame()
    return _save(run_id, "ridge_sizer", lines, out_df)


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: optimise — backtest PnL/Sharpe/WR per bucket
# ─────────────────────────────────────────────────────────────────────────────
def report_optimise(run_id: str, data_root: str) -> str:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"# Optimise Step Report — {run_id}", f"Generated: {ts}\n"]

    backtest_path = Path(data_root) / "artifacts" / run_id / "backtest_results.csv"
    bucket_params_path = Path(data_root) / "artifacts" / run_id / "models" / "bucket_params.json"

    if not backtest_path.exists():
        lines.append(f"No backtest results found at `{backtest_path}`.")
        return _save(run_id, "optimise", lines)

    try:
        trades = pd.read_csv(backtest_path)
    except Exception as e:
        lines.append(f"ERROR loading backtest results: {e}")
        return _save(run_id, "optimise", lines)

    lines.append(f"**Total trades**: {len(trades):,}\n")

    # Determine bucket column
    bkt_col = None
    for c in ["bucket", "strategy", "side_kind", "kind"]:
        if c in trades.columns:
            bkt_col = c
            break

    rows_data = []

    if bkt_col and "pnl" in trades.columns:
        lines.append("## Per-Bucket Performance")
        headers = ["Bucket", "N trades", "Total PnL", "Win Rate", "Avg PnL", "Sharpe (approx)", "Max DD"]
        table_rows = []
        for bkt, grp in trades.groupby(bkt_col):
            n = len(grp)
            total_pnl = float(grp["pnl"].sum())
            wr = float((grp["pnl"] > 0).mean())
            avg_pnl = float(grp["pnl"].mean())
            pnl_std = float(grp["pnl"].std()) if n > 1 else 0.0
            sharpe = avg_pnl / max(pnl_std, 1e-9) * math.sqrt(n)
            cum = grp["pnl"].cumsum()
            max_dd = float((cum - cum.cummax()).min())
            table_rows.append([bkt, f"{n:,}", _fmt(total_pnl, 4), _pct(wr), _fmt(avg_pnl, 5),
                                _fmt(sharpe, 3), _fmt(max_dd, 4)])
            rows_data.append({"bucket": bkt, "n": n, "total_pnl": total_pnl, "win_rate": wr,
                               "avg_pnl": avg_pnl, "sharpe": sharpe, "max_dd": max_dd})
        lines.extend(_md_table(headers, table_rows))
        lines.append("")
    else:
        # Global summary only
        if "pnl" in trades.columns:
            total_pnl = float(trades["pnl"].sum())
            wr = float((trades["pnl"] > 0).mean())
            lines.append(f"**Total PnL**: {total_pnl:.4f}  |  **Win Rate**: {_pct(wr)}  |  **N**: {len(trades):,}")
            lines.append("")

    # Bucket params if available
    if bucket_params_path.exists():
        import json
        try:
            with open(bucket_params_path) as f:
                bp = json.load(f)
            buckets_bp = bp.get("buckets", bp)
            lines.append("## Optimised Bucket Parameters")
            bp_headers = ["Bucket", "TP mult", "SL mult", "Trail mult", "Max hold h"]
            bp_rows = []
            for bkt, params in buckets_bp.items():
                if isinstance(params, dict):
                    bp_rows.append([bkt,
                                    _fmt(params.get("tp_mult", float("nan")), 3),
                                    _fmt(params.get("sl_mult", float("nan")), 3),
                                    _fmt(params.get("trail_mult", float("nan")), 3),
                                    str(params.get("max_hold_hours", "—"))])
            lines.extend(_md_table(bp_headers, bp_rows))
            lines.append("")
        except Exception as e:
            lines.append(f"WARNING: Could not load bucket params: {e}\n")

    out_df = pd.DataFrame(rows_data) if rows_data else pd.DataFrame()
    return _save(run_id, "optimise", lines, out_df)
