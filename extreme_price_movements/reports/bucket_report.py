"""
Per-bucket / per-horizon detailed reports for each pipeline step.

Saves to extreme_price_movements/reports/<run_id>/bucket_report_<step>.md
and a companion CSV for machine consumption.

Steps covered:
  - compare_tbm   : geometry grid quality per cell
  - labels        : label distribution per (bucket, horizon)
  - base_training : alpha model AUC/IC per (bucket, horizon)
  - meta_training : meta model IC/AUC per (bucket, horizon)
  - ev_decomposition : p(win), win quantile, loss quantile metrics per bucket
  - ridge_sizer   : ridge weights + IC per bucket
  - optimise      : backtest PnL/Sharpe/WR per bucket
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from extreme_price_movements.path_utils import resolve_reports_dir

DEFAULT_REPORTS_DIR = Path(__file__).parent

from extreme_price_movements.config import CANON_BUCKETS, CANON_HORIZONS, CANON_CELLS

_BUCKETS = CANON_BUCKETS
_HORIZONS = CANON_HORIZONS
# Remove H8 from cell keys - use CANON_CELLS


def _ensure_dir(run_id: str, base_dir: str | Path | None = None) -> Path:
    reports_dir = resolve_reports_dir(base_dir)
    d = reports_dir / run_id
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


def _save(run_id: str, step: str, lines: List[str], df: Optional[pd.DataFrame] = None, base_dir: str | Path | None = None) -> str:
    out = _ensure_dir(run_id, base_dir=base_dir)
    md_path = out / f"bucket_report_{step}.md"
    md_path.write_text("\n".join(lines))
    if df is not None and not df.empty:
        csv_path = out / f"bucket_report_{step}.csv"
        df.to_csv(csv_path, index=False)
    return str(md_path)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: compare_tbm — geometry grid quality
# ─────────────────────────────────────────────────────────────────────────────
def report_compare_tbm(run_id: str, grid_csv_path: str, base_dir: str | Path | None = None) -> str:
    """Generate per-cell geometry quality report from tbm_geometry_grid.csv."""
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"# TBM Geometry Grid Report — {run_id}", f"Generated: {ts}\n"]

    try:
        df = pd.read_csv(grid_csv_path)
    except Exception as e:
        lines.append(f"ERROR: Could not load grid CSV: {e}")
        return _save(run_id, "compare_tbm", lines, base_dir=base_dir)

    lines.append(f"**Grid CSV**: `{grid_csv_path}`")
    lines.append(f"**Total rows**: {len(df)}  |  **Cells covered**: {df['cell_key'].nunique() if 'cell_key' in df.columns else '?'} / 12\n")

    # Per-cell summary table
    lines.append("## Per-Cell Geometry Quality")
    headers = ["Cell", "N configs", "k_tp range", "sl range", "Best AUC", "Best TP-sep", "Best timeout", "Best bind", "Fallback?"]
    rows = []
    for ck in CANON_CELLS:
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
    lines.append("## Per-Bucket Summary (across H1/H2/H4)")
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

    return _save(run_id, "compare_tbm", lines, df, base_dir=base_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: labels — label distribution per (bucket, horizon)
# ─────────────────────────────────────────────────────────────────────────────
def report_labels(run_id: str, data_root: str, cfg: Dict[str, Any], base_dir: str | Path | None = None) -> str:
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
                # Resolve outcome columns
                if "__y_outcome__" in df.columns:
                    # Outcomes: 2=TP, 1=TIMEOUT, 0=SL
                    tp = float((df["__y_outcome__"] == 2).mean())
                    sl = float((df["__y_outcome__"] == 0).mean())
                    to = float((df["__y_outcome__"] == 1).mean())
                elif "label" in df.columns:
                    # Legacy: 1=TP, -1=SL, 0=TO
                    tp = float((df["label"] == 1).mean())
                    sl = float((df["label"] == -1).mean())
                    to = float((df["label"] == 0).mean())
                elif "__y_bin__" in df.columns:
                    # Partial: 1=TP, 0=Others (SL+TO combined)
                    tp = float((df["__y_bin__"] == 1).mean())
                    sl = float((df["__y_bin__"] == 0).mean()) # Mixed
                    to = 0.0 # Unknown
                else:
                    table_rows.append([name, str(n), "no label col", "—", "—", "—", "—", "—"])
                    continue

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
    return _save(run_id, "labels", lines, out_df, base_dir=base_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: base_training — alpha model AUC/IC per (bucket, horizon)
# ─────────────────────────────────────────────────────────────────────────────
def report_base_training(run_id: str, bundle: Dict[str, Any], cfg: Dict[str, Any], base_dir: str | Path | None = None) -> str:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"# Base Training Report — {run_id}", f"Generated: {ts}\n"]
    horizons = cfg.get("label_horizons_hours", [2, 4, 8])
    sides = ["long", "short"]
    kinds = ["mr", "tf"]

    alpha_models = bundle.get("alpha_models", {}) if bundle else {}
    alpha_metrics = bundle.get("alpha_oof_metrics", {}) if bundle else {}

    rows_data = []
    table_rows = []
    headers = ["Bucket", "H", "Winner algo", "AUC (raw)", "AUC (weighted)", "IC", "Prec@10", "Prec@30", "N features"]

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
                prec30 = _fmt(m.get("prec_at_30", float("nan")))
                n_feats = m.get("n_features", kind_model.get("n_features", "—"))
                table_rows.append([bkt_label, H, winner, auc_raw, auc_w, ic, prec10, prec30, n_feats])
                rows_data.append({"bucket": bkt_label, "H": H, "winner": winner,
                                   "auc_raw": m.get("auc_raw"), "auc_weighted": m.get("auc_weighted"),
                                   "ic": m.get("ic"), "prec_at_10": m.get("prec_at_10"), "prec_at_30": m.get("prec_at_30")})

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
    return _save(run_id, "base_training", lines, out_df, base_dir=base_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: meta_training — meta model IC/AUC per (bucket, horizon)
# ─────────────────────────────────────────────────────────────────────────────
def report_meta_training(run_id: str, data_root: str, bundle: Dict[str, Any], cfg: Dict[str, Any], base_dir: str | Path | None = None) -> str:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"# Meta Training Report — {run_id}", f"Generated: {ts}\n"]

    meta_oof_dir = Path(data_root) / "artifacts" / run_id / "meta_oof"
    rows_data = []

    # Load downstream meta heads only (exclude legacy _reg/_clf and per-horizon files).
    import re as _re
    _exclude_pat = _re.compile(r'(_H\d+|_dur|_reg|_clf)$')
    _include_pat = _re.compile(r'(_utility|_mae_q70|_mfe|_early_inval)$')
    oof_files = sorted([
        f for f in (meta_oof_dir.glob("meta_oof_*.parquet") if meta_oof_dir.exists() else [])
        if (not _exclude_pat.search(f.stem.replace("meta_oof_", "")))
        and _include_pat.search(f.stem.replace("meta_oof_", ""))
    ])
    lines.append(f"**Meta OOF files found**: {len(oof_files)}\n")

    table_rows = []
    headers = ["Model", "N samples", "IC (payoff)", "AUC", "Prec@10", "U_IC", "MAE_IC", "MFE_IC"]

    for pf in oof_files:
        model_name = pf.stem.replace("meta_oof_", "")
        try:
            df = pd.read_parquet(pf)
        except Exception:
            table_rows.append([model_name, "ERR", "—", "—", "—", "—", "—", "—", "—"])
            continue
        n = len(df)
        
        # Payoff IC
        pred = df["oof_pred"].values.astype(float) if "oof_pred" in df.columns else (df["oof_ev"].values.astype(float) if "oof_ev" in df.columns else None)
        ic = float("nan")
        if pred is not None and "return" in df.columns:
            ret = df["return"].values.astype(float)
            valid = np.isfinite(pred) & np.isfinite(ret)
            if valid.sum() >= 10:
                from scipy.stats import spearmanr
                ic = float(spearmanr(pred[valid], ret[valid]).correlation)
        
        # AUC / Prec@10
        auc = float("nan")
        prec10 = float("nan")
        if pred is not None and "label" in df.columns:
            y = (df["label"].values == 1).astype(float)
            n_pos = int(y.sum()); n_neg = len(y) - n_pos
            if n_pos > 0 and n_neg > 0:
                ranks = pd.Series(pred).rank(method="average").to_numpy(float)
                u = ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0
                auc = float(u / (n_pos * n_neg))
            if len(pred) >= 10:
                top10 = pred >= np.quantile(pred, 0.90)
                prec10 = float(y[top10].mean()) if top10.any() else float("nan")

        # --- early_inval: score as AUC vs binary label (IC vs return is structurally negative) ---
        if model_name.endswith("_early_inval") and pred is not None and "early_inval" in df.columns:
            y_ei = df["early_inval"].values.astype(float)
            valid_ei = np.isfinite(pred) & np.isfinite(y_ei)
            if valid_ei.sum() >= 10:
                n_pos_ei = int(y_ei[valid_ei].sum())
                n_neg_ei = int(valid_ei.sum()) - n_pos_ei
                if n_pos_ei > 0 and n_neg_ei > 0:
                    ranks_ei = pd.Series(pred[valid_ei]).rank(method="average").to_numpy(float)
                    u_ei = ranks_ei[y_ei[valid_ei] == 1].sum() - n_pos_ei * (n_pos_ei + 1) / 2.0
                    auc = float(u_ei / (n_pos_ei * n_neg_ei))
                    top10_ei = pred[valid_ei] >= np.quantile(pred[valid_ei], 0.90)
                    prec10 = float(y_ei[valid_ei][top10_ei].mean()) if top10_ei.any() else float("nan")
                    ic = float("nan")  # IC vs return is structurally misleading here

        # Aux Heads ICs
        # MAE and MFE are stored as absolute magnitudes (always positive)
        # The model predictions are also positive (log of absolute values)
        # Sign = 1.0 for both long and short (no flip needed)
        mae_sign = 1.0
        mfe_sign = 1.0
        
        u_ic = float("nan")
        mae_ic = float("nan")
        mfe_ic = float("nan")
        from scipy.stats import spearmanr
        def _sic(a, b):
            mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() < 10: return float("nan")
            return float(spearmanr(a[mask], b[mask]).correlation)

        if "oof_u_hat" in df.columns and "u_policy_net" in df.columns:
            u_ic = _sic(df["oof_u_hat"].values, df["u_policy_net"].values)
        if "oof_log_mae_q70_hat" in df.columns and "mae_ret" in df.columns:
            u_mae = df["mae_ret"].values / np.clip(df["__barrier_pct__"].values if "__barrier_pct__" in df.columns else 1.0, 1e-6, None)
            mae_ic = _sic(df["oof_log_mae_q70_hat"].values, np.log1p(np.clip(mae_sign * u_mae, 0, None)))
        if "oof_log_mfe_hat" in df.columns and "mfe_ret" in df.columns:
            u_mfe = df["mfe_ret"].values / np.clip(df["__barrier_pct__"].values if "__barrier_pct__" in df.columns else 1.0, 1e-6, None)
            mfe_ic = _sic(df["oof_log_mfe_hat"].values, np.log1p(np.clip(mfe_sign * u_mfe, 0, None)))

        table_rows.append([model_name, f"{n:,}", _fmt(ic), _fmt(auc), _fmt(prec10), _fmt(u_ic), _fmt(mae_ic), _fmt(mfe_ic)])
        rows_data.append({"model": model_name, "n": n, "ic": ic, "auc": auc, "prec_at_10": prec10,
                           "u_ic": u_ic, "mae_ic": mae_ic, "mfe_ic": mfe_ic})

    lines.append("## Meta OOF Predictions per Model")
    lines.extend(_md_table(headers, table_rows))
    lines.append("")

    # Per-bucket summary: group by base bucket (strip all known model-type suffixes)
    import re
    _suffix_pat = re.compile(r'^(.+?)(_H\d+|_reg|_clf|_early_inval|_utility|_mae_q70|_mfe|_dur)$')
    bucket_groups: Dict[str, List[Dict]] = {}
    for r in rows_data:
        m = _suffix_pat.match(r["model"])
        bkt = m.group(1) if m else r["model"]
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
    return _save(run_id, "meta_training", lines, out_df, base_dir=base_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4b: ev_decomposition — p(win), win quantile, loss quantile metrics per bucket
# ─────────────────────────────────────────────────────────────────────────────
def report_ev_decomposition(run_id: str, data_root: str, base_dir: str | Path | None = None) -> str:
    """Generate EV decomposition diagnostics report from training diagnostics JSON."""
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"# EV Decomposition Report — {run_id}", f"Generated: {ts}\n"]

    diag_path = Path(data_root) / "artifacts" / run_id / "ev_decomposition" / "ev_decomposition_training_diagnostics.json"
    
    if not diag_path.exists():
        lines.append(f"**Diagnostics file not found**: `{diag_path}`")
        lines.append("\nEV decomposition may not have been trained (check cfg: ev_decomposition_enabled)")
        return _save(run_id, "ev_decomposition", lines, base_dir=base_dir)

    try:
        import json
        with open(diag_path) as f:
            diag = json.load(f)
    except Exception as e:
        lines.append(f"ERROR: Could not load diagnostics JSON: {e}")
        return _save(run_id, "ev_decomposition", lines, base_dir=base_dir)

    training_config = diag.get("training_config", {})
    diagnostics = diag.get("diagnostics", {})

    lines.append("## Training Configuration")
    lines.append(f"- **p(win) base engine**: {training_config.get('pwin_base_engine', '—')}")
    lines.append(f"- **Quantile base engine**: {training_config.get('quantile_base_engine', '—')}")
    lines.append(f"- **Regularization level**: {training_config.get('regularization_level', '—')}")
    lines.append(f"- **Calibrator method**: {training_config.get('calibrator_method', '—')}")
    lines.append(f"- **Quantile delta**: {training_config.get('quantile_delta', '—')}")
    lines.append("")

    # Overall p(win) metrics
    pwin = diagnostics.get("pwin", {})
    lines.append("## p(win) Model — Overall")
    lines.append(f"- **Mean prediction**: {_fmt(pwin.get('pwin_mean_pred', float('nan')))}")
    lines.append(f"- **Mean target**: {_fmt(pwin.get('pwin_mean_target', float('nan')))}")
    if "auc" in pwin:
        lines.append(f"- **AUC**: {_fmt(pwin.get('auc', float('nan')))}")
    if "ece_top10" in pwin:
        lines.append(f"- **ECE@10**: {_fmt(pwin.get('ece_top10', float('nan')))}")
    if "prec_at_10" in pwin:
        lines.append(f"- **Prec@10**: {_fmt(pwin.get('prec_at_10', float('nan')))}")
    lines.append("")

    # Win quantile metrics
    wq = diagnostics.get("win_quantiles", {})
    lines.append("## Win Quantiles — Overall")
    win_headers = ["Metric", "Value"]
    win_rows = [
        ["N", wq.get("n", "—")],
        ["Pinball Q50", _fmt(wq.get("pinball_q50", float('nan')))],
        ["Pinball Q80", _fmt(wq.get("pinball_qh", float('nan')))],
        ["Coverage Q50", _fmt(wq.get("coverage_q50", float('nan')))],
        ["Coverage Q80", _fmt(wq.get("coverage_qh", float('nan')))],
        ["Interval Width (mean)", _fmt(wq.get("interval_evaluation", float('nan')))],
        ["Mean actual", _fmt(wq.get("mean_y", float('nan')))],
        ["Mean Q50 pred", _fmt(wq.get("mean_q50", float('nan')))],
        ["Mean Q80 pred", _fmt(wq.get("mean_qh", float('nan')))],
    ]
    lines.extend(_md_table(win_headers, win_rows))
    lines.append("")

    # Loss quantile metrics
    lq = diagnostics.get("loss_quantiles", {})
    lines.append("## Loss Quantiles — Overall")
    loss_headers = ["Metric", "Value"]
    loss_rows = [
        ["N", lq.get("n", "—")],
        ["Pinball Q50", _fmt(lq.get("pinball_q50", float('nan')))],
        ["Pinball Q90", _fmt(lq.get("pinball_qh", float('nan')))],
        ["Coverage Q50", _fmt(lq.get("coverage_q50", float('nan')))],
        ["Coverage Q90", _fmt(lq.get("coverage_qh", float('nan')))],
        ["Interval Width (mean)", _fmt(lq.get("interval_evaluation", float('nan')))],
        ["Mean actual", _fmt(lq.get("mean_y", float('nan')))],
        ["Mean Q50 pred", _fmt(lq.get("mean_q50", float('nan')))],
        ["Mean Q90 pred", _fmt(lq.get("mean_qh", float('nan')))],
    ]
    lines.extend(_md_table(loss_headers, loss_rows))
    lines.append("")

    # Per-bucket breakdown
    per_bucket = diagnostics.get("per_bucket", {})
    if per_bucket:
        lines.append("## Per-Bucket Breakdown")
        bkt_headers = ["Bucket", "N", "p(win) pred", "p(win) target", "Win Q50 pinball", "Win Q80 pinball", "Loss Q50 pinball", "Loss Q90 pinball"]
        bkt_rows = []
        for bkt in sorted(per_bucket.keys()):
            bb = per_bucket[bkt]
            pw = bb.get("pwin", {})
            wq_b = bb.get("win_quantiles", {})
            lq_b = bb.get("loss_quantiles", {})
            bkt_rows.append([
                bkt,
                pw.get("n", "—"),
                _fmt(pw.get("mean_pred", float('nan'))),
                _fmt(pw.get("mean_target", float('nan'))),
                _fmt(wq_b.get("pinball_q50", float('nan'))),
                _fmt(wq_b.get("pinball_qh", float('nan'))),
                _fmt(lq_b.get("pinball_q50", float('nan'))),
                _fmt(lq_b.get("pinball_qh", float('nan'))),
            ])
        lines.extend(_md_table(bkt_headers, bkt_rows))
        lines.append("")

    return _save(run_id, "ev_decomposition", lines, base_dir=base_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: ridge_sizer — weights + IC per bucket
# ─────────────────────────────────────────────────────────────────────────────
def report_ridge_sizer(run_id: str, result: Dict[str, Any], base_dir: str | Path | None = None) -> str:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"# Ridge Position Sizer Report — {run_id}", f"Generated: {ts}\n"]

    if not result:
        lines.append("No ridge sizer results available.")
        return _save(run_id, "ridge_sizer", lines, base_dir=base_dir)

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
            
            # Build flat row with all weights + metrics
            row = {
                "direction": direction,
                "bucket": bkt,
                "ic_train": m.get("ic_train"),
                "ic_val": m.get("ic_val"),
                "sharpe": m.get("sharpe"),
                "n_weights": n_w,
            }
            # Add all weights as JSON string (flat columns can have name collisions across buckets)
            row["weights_json"] = json.dumps(bkt_weights) if bkt_weights else "{}"
            # Add key metrics from internal dict
            row["n_trades"] = m.get("n_trades")
            row["n_models"] = m.get("n_models")
            row["best_target_name"] = m.get("best_target_name")
            row["cv_best_pnl_total"] = m.get("cv_best_pnl_total")
            row["cv_best_sortino"] = m.get("cv_best_sortino")
            row["cv_best_ic"] = m.get("cv_best_ic")
            row["cv_best_winrate"] = m.get("cv_best_winrate")
            row["cv_best_maxdd"] = m.get("cv_best_maxdd")
            row["utility_policy_model_family"] = m.get("utility_policy_model_family")
            row["utility_smoother_family"] = m.get("utility_smoother_family")
            row["offset_model_family"] = m.get("offset_model_family")
            row["offset_smoother_family"] = m.get("offset_smoother_family")
            
            # Add sizing impact if present
            sizing_imp = m.get("sizing_impact")
            if sizing_imp:
                row["sizing_best_mode"] = sizing_imp.get("best", {}).get("mode")
                row["sizing_best_pnl"] = sizing_imp.get("best", {}).get("pnl")
                row["sizing_best_sortino"] = sizing_imp.get("best", {}).get("sortino")
                row["sizing_worst_mode"] = sizing_imp.get("worst", {}).get("mode")
                row["sizing_worst_pnl"] = sizing_imp.get("worst", {}).get("pnl")
            
            # Add weight diagnostics
            wdiag = m.get("weight_diagnostics", {})
            row["weight_l1"] = wdiag.get("weight_l1")
            row["weight_l2"] = wdiag.get("weight_l2")
            row["weight_max_abs"] = wdiag.get("weight_max_abs")
            row["weight_top1_share"] = wdiag.get("weight_top1_share")
            row["weight_effective_n_models"] = wdiag.get("weight_effective_n_models")
            
            # Add OOF rank-based metrics (top30, top20, top10)
            for prefix in ["oof_top30", "oof_top20", "oof_top10"]:
                row[f"{prefix}_pnl_total"] = m.get(f"{prefix}_pnl_total")
                row[f"{prefix}_pnl_per_trade"] = m.get(f"{prefix}_pnl_per_trade")
                row[f"{prefix}_trades_per_day"] = m.get(f"{prefix}_trades_per_day")
                row[f"{prefix}_ulcer"] = m.get(f"{prefix}_ulcer")
                row[f"{prefix}_sortino"] = m.get(f"{prefix}_sortino")
                row[f"{prefix}_maxdd"] = m.get(f"{prefix}_maxdd")
                row[f"{prefix}_time_under_water"] = m.get(f"{prefix}_time_under_water")
                row[f"{prefix}_n_trades"] = m.get(f"{prefix}_n_trades")
            
            rows_data.append(row)
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
    return _save(run_id, "ridge_sizer", lines, out_df, base_dir=base_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: optimise — backtest PnL/Sharpe/WR per bucket
# ─────────────────────────────────────────────────────────────────────────────
def report_optimise(run_id: str, data_root: str, base_dir: str | Path | None = None) -> str:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"# Optimise Step Report — {run_id}", f"Generated: {ts}\n"]

    bucket_params_path = Path(data_root) / "artifacts" / run_id / "models" / "bucket_params.json"

    rows_data = []

    # Bucket params if available
    if bucket_params_path.exists():
        import json
        try:
            with open(bucket_params_path) as f:
                bp = json.load(f)
            buckets_bp = bp.get("buckets", bp) if isinstance(bp, dict) else {}
            if not isinstance(buckets_bp, dict) or not buckets_bp:
                lines.append(f"No bucket payload in `{bucket_params_path}`.")
                return _save(run_id, "optimise", lines, base_dir=base_dir)

            # Per-bucket holdout metrics from optimiser evaluation (source of truth).
            table_rows = []
            all_ledger_rows = 0
            starts = []
            ends = []
            for bkt, payload in buckets_bp.items():
                if not isinstance(payload, dict):
                    continue
                ev = payload.get("evaluation", {}) if isinstance(payload.get("evaluation", {}), dict) else {}
                tpsl = payload.get("tp_sl", {}) if isinstance(payload.get("tp_sl", {}), dict) else {}
                holdout_n = int(ev.get("holdout_trades", 0) or 0)
                holdout_pnl = float(ev.get("holdout_pnl_net", 0.0) or 0.0)
                holdout_wr = float(ev.get("holdout_win_rate", 0.0) or 0.0)
                avg_pnl = (holdout_pnl / holdout_n) if holdout_n > 0 else 0.0
                pnl_per_day = float("nan")
                ledger_path = ev.get("holdout_ledger_path")
                if isinstance(ledger_path, str) and ledger_path:
                    lp = Path(ledger_path)
                    if not lp.is_absolute():
                        lp = (Path(data_root) / "artifacts" / run_id / "models" / lp.name).resolve()
                    if lp.exists():
                        try:
                            ldf = pd.read_csv(lp, usecols=["t_entry"])
                            if not ldf.empty:
                                ts = pd.to_datetime(ldf["t_entry"], unit="ns", errors="coerce")
                                starts.append(ts.min())
                                ends.append(ts.max())
                                all_ledger_rows += int(len(ldf))
                                _s = ts.min()
                                _e = ts.max()
                                if pd.notna(_s) and pd.notna(_e):
                                    _days = max((pd.Timestamp(_e) - pd.Timestamp(_s)).total_seconds() / 86400.0, 1.0 / 24.0)
                                    pnl_per_day = holdout_pnl / float(_days)
                        except Exception:
                            pass

                table_rows.append([
                    bkt,
                    f"{holdout_n:,}",
                    _fmt(holdout_pnl, 5),
                    _fmt(pnl_per_day, 5),
                    _pct(holdout_wr),
                    _fmt(avg_pnl, 5),
                    _fmt(float(tpsl.get("tp_mult", float("nan"))), 3),
                    _fmt(float(tpsl.get("sl_mult", float("nan"))), 3),
                ])
                rows_data.append({
                    "bucket": bkt,
                    "n": holdout_n,
                    "total_pnl": holdout_pnl,
                    "pnl_per_day": pnl_per_day,
                    "win_rate": holdout_wr,
                    "avg_pnl": avg_pnl,
                    "tp_mult": float(tpsl.get("tp_mult", float("nan"))),
                    "sl_mult": float(tpsl.get("sl_mult", float("nan"))),
                })

            total_n = int(sum(r["n"] for r in rows_data)) if rows_data else 0
            total_pnl = float(sum(r["total_pnl"] for r in rows_data)) if rows_data else 0.0
            wr_num = float(sum(r["win_rate"] * r["n"] for r in rows_data)) if rows_data else 0.0
            wr_den = float(sum(r["n"] for r in rows_data)) if rows_data else 0.0
            wr_w = (wr_num / wr_den) if wr_den > 0 else 0.0
            lines.append(f"**Total holdout trades**: {total_n:,}  |  **Total holdout PnL (net)**: {_fmt(total_pnl, 5)}  |  **Weighted holdout WR**: {_pct(wr_w)}")
            if starts and ends:
                _s = min(starts)
                _e = max(ends)
                if pd.notna(_s) and pd.notna(_e):
                    days = max(1, int((pd.Timestamp(_e) - pd.Timestamp(_s)).total_seconds() / 86400))
                    total_pnl_per_day = total_pnl / max((pd.Timestamp(_e) - pd.Timestamp(_s)).total_seconds() / 86400.0, 1.0 / 24.0)
                    lines.append(f"**Holdout period**: `{pd.Timestamp(_s).strftime('%Y-%m-%d %H:%M')} → {pd.Timestamp(_e).strftime('%Y-%m-%d %H:%M')}` (~{days} days)")
                    lines.append(f"**Aggregate holdout PnL/Day**: {_fmt(total_pnl_per_day, 5)}")
            if all_ledger_rows:
                lines.append(f"**Ledger rows across buckets**: {all_ledger_rows:,}")
            lines.append("")
            lines.append("## Per-Bucket Holdout Performance (from bucket_params evaluation)")
            headers = ["Bucket", "N trades", "Total PnL", "PnL/Day", "Win Rate", "Avg PnL", "TP mult", "SL mult"]
            lines.extend(_md_table(headers, table_rows))
            lines.append("")
        except Exception as e:
            lines.append(f"WARNING: Could not load bucket params: {e}\n")
    else:
        lines.append(f"No bucket params found at `{bucket_params_path}`.")

    out_df = pd.DataFrame(rows_data) if rows_data else pd.DataFrame()
    return _save(run_id, "optimise", lines, out_df, base_dir=base_dir)
