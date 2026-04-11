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

    # Per-bucket aggregation
    from extreme_price_movements.strategy_registry import get_strategies
    strategies = get_strategies()
    _bucket_map = {f"{strat['trade_side']}_{strat['strategy_id']}": (strat["trade_side"], strat["strategy_id"]) for strat in strategies}

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
    alpha_models = bundle.get("alpha_models", {}) if bundle else {}
    base_variant_models = bundle.get("base_variant_models", {}) if bundle else {}
    quality_gate = bundle.get("quality_gate_report", {}) if bundle else {}
    base_rows = [
        r
        for r in list(quality_gate.get("base_models", []) or [])
        if str(r.get("variant", "primary")) != "primary"
    ]

    rows_data = []
    table_rows = []
    if base_rows:
        headers = [
            "Strategy ID",
            "Side",
            "H",
            "Variant",
            "Model",
            "Winner",
            "AUC",
            "IC(bin)",
            "IC(ret)",
            "LogLoss",
            "PR-AUC",
            "Lift@20",
            "Prec@10",
            "Prec@30",
            "N features",
        ]

        def _feature_count(side: str, strategy_id: str, horizon: int, variant: str) -> int | str:
            if variant:
                v_conf = base_variant_models.get((side, strategy_id, int(horizon), variant), {})
                feats = v_conf.get("selected_features") or v_conf.get("feat_cols") or []
                return len(feats) if feats else "—"
            conf = alpha_models.get(side, {}).get(strategy_id, {})
            h_conf = (conf.get("models_by_h", {}) or {}).get(int(horizon), {})
            feats = h_conf.get("selected_features") or h_conf.get("feat_cols") or []
            return len(feats) if feats else "—"

        for r in base_rows:
            metrics = r.get("metrics", {}) or {}
            side = str(r.get("side", ""))
            strategy_id = str(r.get("kind", ""))
            horizon = int(r.get("H", 0) or 0)
            variant = str(r.get("variant", "primary") or "primary")
            model_name = str(r.get("model", ""))
            candidate = model_name.split(":", 1)[-1] if ":" in model_name else model_name
            winner = "✓" if bool(r.get("is_winner", False)) else ""
            auc = _fmt(metrics.get("auc"))
            ic_bin = _fmt(metrics.get("ic"))
            ic_ret = _fmt(metrics.get("ic_ret"))
            logloss = _fmt(metrics.get("logloss"))
            pr_auc = _fmt(metrics.get("pr_auc"))
            lift20 = _fmt(metrics.get("lift_at_20pct"))
            prec10 = _fmt(metrics.get("prec_at_10pct"))
            prec30 = _fmt(metrics.get("prec_at_30pct"))
            n_feats = _feature_count(side, strategy_id, horizon, variant)
            table_rows.append(
                [
                    strategy_id,
                    side,
                    horizon,
                    variant,
                    candidate,
                    winner,
                    auc,
                    ic_bin,
                    ic_ret,
                    logloss,
                    pr_auc,
                    lift20,
                    prec10,
                    prec30,
                    n_feats,
                ]
            )
            rows_data.append(
                {
                    "strategy_id": strategy_id,
                    "side": side,
                    "H": horizon,
                    "variant": variant,
                    "model": candidate,
                    "is_winner": bool(r.get("is_winner", False)),
                    "auc": metrics.get("auc"),
                    "ic_bin": metrics.get("ic"),
                    "ic_ret": metrics.get("ic_ret"),
                    "logloss": metrics.get("logloss"),
                    "pr_auc": metrics.get("pr_auc"),
                    "lift_at_20pct": metrics.get("lift_at_20pct"),
                    "prec_at_10pct": metrics.get("prec_at_10pct"),
                    "prec_at_30pct": metrics.get("prec_at_30pct"),
                    "n_features": n_feats,
                }
            )

        lines.append("## Alpha Model Performance per Strategy / Horizon")
        lines.extend(_md_table(headers, table_rows))
        lines.append("")

        lines.append("## Per-Strategy Summary")
        bkt_headers = ["Strategy ID", "Side", "Variant", "Deployed Hs", "Primary H", "Median AUC", "Median IC", "Median PR-AUC"]
        bkt_rows = []
        for side in sorted({r["side"] for r in rows_data}):
            side_rows = [r for r in rows_data if r["side"] == side]
            for strategy_id in sorted({r["strategy_id"] for r in side_rows}):
                sub_all = [r for r in side_rows if r["strategy_id"] == strategy_id]
                for variant in sorted({str(r.get("variant", "primary")) for r in sub_all}):
                    sub = [r for r in sub_all if str(r.get("variant", "primary")) == variant]
                    if not sub:
                        continue
                    deployed = sorted({int(r["H"]) for r in sub if r.get("H") is not None})
                    winners = [r for r in sub if r.get("is_winner")]
                    primary_h = winners[0]["H"] if winners else deployed[0]
                    auc_vals = [r["auc"] for r in sub if r.get("auc") is not None]
                    ic_vals = [r["ic_bin"] for r in sub if r.get("ic_bin") is not None]
                    pr_vals = [r["pr_auc"] for r in sub if r.get("pr_auc") is not None]
                    bkt_rows.append(
                        [
                            strategy_id,
                            side,
                            variant,
                            str(deployed),
                            str(primary_h),
                            _fmt(float(np.median(auc_vals))) if auc_vals else "—",
                            _fmt(float(np.median(ic_vals))) if ic_vals else "—",
                            _fmt(float(np.median(pr_vals))) if pr_vals else "—",
                        ]
                    )
        lines.extend(_md_table(bkt_headers, bkt_rows))
        lines.append("")
    else:
        horizons = cfg.get("label_horizons_hours", [5, 10])
        sides = ["long", "short"]
        kinds = ["mr", "tf"]
        headers = ["Bucket", "H", "Winner algo", "AUC (raw)", "AUC (weighted)", "IC", "Prec@10", "Prec@30", "N features"]
        for side in sides:
            for kind in kinds:
                bkt_label = f"{'MR' if kind == 'mr' else 'TF'}_{'long' if side == 'long' else 'short'}"
                side_models = alpha_models.get(side, {})
                kind_model = side_models.get(kind, {})
                for H in horizons:
                    m = {}
                    winner = kind_model.get("winner_algo", "—")
                    auc_raw = _fmt(float("nan"))
                    auc_w = _fmt(float("nan"))
                    ic = _fmt(float("nan"))
                    prec10 = _fmt(float("nan"))
                    prec30 = _fmt(float("nan"))
                    n_feats = kind_model.get("n_features", "—")
                    table_rows.append([bkt_label, H, winner, auc_raw, auc_w, ic, prec10, prec30, n_feats])
        lines.append("## Alpha Model Performance per (Bucket, Horizon)")
        lines.extend(_md_table(headers, table_rows))
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

    import re as _re
    oof_files = sorted(
        meta_oof_dir.glob("meta_oof_*.parquet") if meta_oof_dir.exists() else []
    )
    lines.append(f"**Meta OOF files found**: {len(oof_files)}\n")

    for pf in oof_files:
        model_name = pf.stem.replace("meta_oof_", "")
        try:
            df = pd.read_parquet(pf)
        except Exception:
            rows_data.append({"model": model_name, "load_error": True})
            continue
        n = len(df)
        from scipy.stats import spearmanr

        def _sic(a, b):
            mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() < 10:
                return float("nan")
            return float(spearmanr(a[mask], b[mask]).correlation)

        pred = (
            df["oof_pred"].values.astype(float)
            if "oof_pred" in df.columns
            else (df["oof_ev"].values.astype(float) if "oof_ev" in df.columns else None)
        )
        ic_return = (
            _sic(pred, df["return"].values.astype(float))
            if pred is not None and "return" in df.columns
            else float("nan")
        )
        ic_u = (
            _sic(pred, df["u_policy_net"].values.astype(float))
            if pred is not None and "u_policy_net" in df.columns
            else float("nan")
        )

        head_type = "other"
        horizon = None
        geom = None
        strategy = model_name
        m = _re.match(
            r"^(?P<strategy>.+)_(?P<geom>tbm_\d+_\d+)_h(?P<h>\d+)$",
            model_name,
        )
        if m:
            head_type = "tbm_clf"
            strategy = m.group("strategy")
            geom = m.group("geom")
            horizon = int(m.group("h"))
        else:
            m = _re.match(r"^(?P<strategy>.+)_(?P<kind>mae|mfe|asym)_h(?P<h>\d+)$", model_name)
            if m:
                head_type = m.group("kind")
                strategy = m.group("strategy")
                horizon = int(m.group("h"))
            elif model_name.endswith("_clf"):
                head_type = "clf"
                strategy = model_name[: -len("_clf")]
            elif model_name.endswith("_reg"):
                head_type = "reg"
                strategy = model_name[: -len("_reg")]

        row = {
            "model": model_name,
            "strategy": strategy,
            "head_type": head_type,
            "geometry": geom,
            "horizon": horizon,
            "n": n,
            "ic_return": ic_return,
            "ic_u_policy": ic_u,
        }

        if head_type == "tbm_clf" and {"oof_p_sl", "oof_p_to", "oof_p_tp", "exit_code"}.issubset(df.columns):
            y = df["exit_code"].values.astype(int)
            p = df[["oof_p_sl", "oof_p_to", "oof_p_tp"]].values.astype(float)
            valid = np.isfinite(p).all(axis=1) & np.isfinite(y)
            y = y[valid]
            p = p[valid]
            if len(y) >= 10 and len(np.unique(y)) >= 2:
                p = np.clip(p, 1e-9, 1.0)
                p = p / np.clip(p.sum(axis=1, keepdims=True), 1e-9, None)
                row["logloss"] = float((-np.log(p[np.arange(len(y)), y])).mean())
                row["acc"] = float((np.argmax(p, axis=1) == y).mean())
                y_tp = (y == 2).astype(float)
                if y_tp.sum() > 0 and y_tp.sum() < len(y_tp):
                    ranks = pd.Series(p[:, 2]).rank(method="average").to_numpy(float)
                    n_pos = int(y_tp.sum())
                    n_neg = int(len(y_tp) - n_pos)
                    u = ranks[y_tp == 1].sum() - n_pos * (n_pos + 1) / 2.0
                    row["auc_tp"] = float(u / (n_pos * n_neg))
                    k10 = max(1, int(np.ceil(0.10 * len(y_tp))))
                    idx10 = np.argsort(p[:, 2])[-k10:]
                    row["prec10_tp"] = float(y_tp[idx10].mean())
        elif head_type == "mae" and pred is not None and "mae_ret" in df.columns:
            row["ic_target"] = _sic(pred, df["mae_ret"].values.astype(float))
            if "mae_ret" in df.columns and len(df) >= 10:
                k30 = max(1, int(np.ceil(0.30 * len(df))))
                idx30 = np.argsort(pred)[-k30:]
                row["top30_target_mean"] = float(np.nanmean(df["mae_ret"].values[idx30].astype(float)))
        elif head_type == "mfe" and pred is not None and "mfe_ret" in df.columns:
            row["ic_target"] = _sic(pred, df["mfe_ret"].values.astype(float))
            if "mfe_ret" in df.columns and len(df) >= 10:
                k30 = max(1, int(np.ceil(0.30 * len(df))))
                idx30 = np.argsort(pred)[-k30:]
                row["top30_target_mean"] = float(np.nanmean(df["mfe_ret"].values[idx30].astype(float)))
        elif head_type == "reg" and pred is not None:
            if "return" in df.columns and len(df) >= 10:
                k30 = max(1, int(np.ceil(0.30 * len(df))))
                idx30 = np.argsort(pred)[-k30:]
                row["top30_return_mean"] = float(np.nanmean(df["return"].values[idx30].astype(float)))

        rows_data.append(row)

    detail_df = pd.DataFrame(rows_data) if rows_data else pd.DataFrame()

    requested_order = ["tbm_500_250", "tbm_250_125", "mae", "mfe", "asym", "clf", "reg"]
    detail_df = detail_df[
        detail_df["head_type"].isin(["tbm_clf", "mae", "mfe", "reg"])
    ].copy()

    strategies = sorted(
        {
            str(v)
            for v in detail_df.get("strategy", pd.Series(dtype=str)).dropna().tolist()
            if v
        }
    )
    for strategy in strategies:
        sub = detail_df[detail_df["strategy"] == strategy].copy()
        if sub.empty:
            continue
        lines.append(f"## {strategy}")
        headers = [
            "Head",
            "H",
            "N",
            "LogLoss",
            "Acc",
            "AUC_TP",
            "Prec@10_TP",
            "IC_target",
            "IC_return",
            "IC_u",
            "Top30_target",
            "Top30_return",
        ]
        rows = []
        for key in requested_order:
            if key.startswith("tbm_"):
                rows_sub = sub[(sub["head_type"] == "tbm_clf") & (sub["geometry"] == key)]
            else:
                rows_sub = sub[sub["head_type"] == key]
            if rows_sub.empty:
                rows.append([key, "—", "0", "—", "—", "—", "—", "—", "—", "—", "—", "—"])
                continue
            for _, r in rows_sub.sort_values(["horizon", "model"]).iterrows():
                rows.append([
                    key,
                    _fmt(r.get("horizon"), 0),
                    f"{int(r.get('n', 0)):,}",
                    _fmt(r.get("logloss")),
                    _fmt(r.get("acc")),
                    _fmt(r.get("auc_tp")),
                    _fmt(r.get("prec10_tp")),
                    _fmt(r.get("ic_target")),
                    _fmt(r.get("ic_return")),
                    _fmt(r.get("ic_u_policy")),
                    _fmt(r.get("top30_target_mean")),
                    _fmt(r.get("top30_return_mean")),
                ])
        lines.extend(_md_table(headers, rows))
        lines.append("")

    lines.append("## Strategy Summary")
    summary_headers = ["Strategy", "Heads", "Median IC_return", "Median IC_u", "Median AUC_TP"]
    summary_rows = []
    for strategy in strategies:
        sub = detail_df[detail_df["strategy"] == strategy]
        ic_ret = sub["ic_return"].dropna()
        ic_u = sub["ic_u_policy"].dropna()
        auc_tp = sub["auc_tp"].dropna() if "auc_tp" in sub.columns else pd.Series(dtype=float)
        summary_rows.append([
            strategy,
            int(len(sub)),
            _fmt(float(ic_ret.median())) if len(ic_ret) else "—",
            _fmt(float(ic_u.median())) if len(ic_u) else "—",
            _fmt(float(auc_tp.median())) if len(auc_tp) else "—",
        ])
    lines.extend(_md_table(summary_headers, summary_rows))
    lines.append("")

    return _save(run_id, "meta_training", lines, detail_df, base_dir=base_dir)


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
