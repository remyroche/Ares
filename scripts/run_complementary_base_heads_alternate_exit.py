#!/usr/bin/env python3
"""Replay the frozen 15-minute trailing exit on the causal base-head OOS panel.

The model predictions are imported unchanged from the complementary-head
monthly expanding replay.  Only the economic outcome is replaced by the
frozen alternate exit: entry at the decision bar's 15-minute open, 48 bars,
SL=3 ATR, trailing activation=.5 ATR, giveback=.25 ATR, and one 100-bps cost.
This keeps model fitting untouched while providing a matched alternate-exit
confirmation against the single retained base head.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.complementary_base_heads import global_tail_metrics
from extreme_price_movements.trailing_exit_grid import net_bps, simulate_h12_stop_trailing_grid

HORIZON_BARS = 48
COST_BPS = 100.0
STOP_ATR = 3.0
TRAIL_ACTIVATION_ATR = 0.5
TRAIL_GIVEBACK_ATR = 0.25
TAILS = (0.01, 0.02, 0.05)


def _policy_labels(pred: pd.DataFrame, path_root: Path, bars_root: Path) -> pd.DataFrame:
    out = pred[["candidate_id", "__ts__", "side_name"]].copy()
    for c in ("alternate_policy_net_bps", "alternate_policy_gross_bps", "alternate_policy_atr_bps"):
        out[c] = np.nan
    out["alternate_policy_valid"] = False
    out["alternate_policy_entry_ts"] = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    out["alternate_policy_label_available_ts"] = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    out["alternate_policy_symbol"] = ""
    out["alternate_policy_path_resolution_minutes"] = 15
    out["alternate_policy_entry_convention"] = "decision_ts_15m_open"
    index = out.set_index("candidate_id").index
    work = out[["candidate_id", "__ts__", "side_name"]].copy()
    work["symbol"] = work["candidate_id"].astype(str).str.split("|").str[0]
    for symbol, group in work.groupby("symbol", sort=True):
        path_file = path_root / f"symbol={symbol}.parquet"
        bar_file = bars_root / (str(symbol).lower().replace("_", "") + "_15m.parquet")
        if not path_file.exists() or not bar_file.exists():
            continue
        ids = set(group.candidate_id.astype(str))
        path = pd.read_parquet(path_file, columns=[
            "candidate_id", "__decision_ts__", "entry_price", "atr_bps", "label_valid",
        ])
        path = path[path.candidate_id.astype(str).isin(ids)].copy()
        if path.empty:
            continue
        path["candidate_id"] = path["candidate_id"].astype(str)
        if path.candidate_id.duplicated().any():
            raise ValueError(f"duplicate path candidate in {path_file}")
        z = group.merge(path, on="candidate_id", how="inner", validate="one_to_one")
        bars = pd.read_parquet(bar_file)
        time_col = next((c for c in ("ts", "timestamp", "__index_level_0__") if c in bars.columns), None)
        if time_col is not None:
            bars = bars.set_index(time_col)
        if not isinstance(bars.index, pd.DatetimeIndex):
            raise ValueError(f"15m source lacks timestamps: {bar_file}")
        bars.index = pd.to_datetime(bars.index, utc=True)
        bars = bars.loc[:, ["high", "low", "close"]]
        bars = bars[~bars.index.duplicated(keep="last")].sort_index()
        decision = pd.to_datetime(z["__decision_ts__"], utc=True)
        starts = bars.index.get_indexer(decision)
        entry = pd.to_numeric(z.entry_price, errors="coerce").to_numpy(float)
        atr_bps = pd.to_numeric(z.atr_bps, errors="coerce").to_numpy(float)
        valid = (
            z.label_valid.fillna(False).to_numpy(bool)
            & (starts >= 0)
            & (starts + HORIZON_BARS <= len(bars))
            & np.isfinite(entry) & (entry > 0.0)
            & np.isfinite(atr_bps) & (atr_bps > 0.0)
        )
        if not valid.any():
            continue
        zv = z.loc[valid]
        starts_v = starts[valid].astype(np.int64)
        entry_v = entry[valid].astype(np.float32)
        atr_bps_v = atr_bps[valid].astype(np.float32)
        atr_v = entry_v * atr_bps_v / 10_000.0
        side = np.where(zv.side_name.astype(str).str.lower().eq("long"), 1.0, -1.0).astype(np.float32)
        gross_atr = simulate_h12_stop_trailing_grid(
            bars.high.to_numpy(float), bars.low.to_numpy(float), bars.close.to_numpy(float),
            starts_v, entry_v, atr_v, side,
            np.asarray([STOP_ATR], dtype=np.float32),
            np.asarray([TRAIL_ACTIVATION_ATR], dtype=np.float32),
            np.asarray([TRAIL_GIVEBACK_ATR], dtype=np.float32),
            horizon_bars=HORIZON_BARS,
        ).reshape(-1)
        policy_net = net_bps(gross_atr.reshape(-1, 1, 1, 1), atr_bps_v, cost_bps=COST_BPS).reshape(-1)
        for cid, ts, net, atrv in zip(zv.candidate_id, decision.loc[zv.index], policy_net, atr_bps_v):
            loc = index.get_loc(str(cid))
            out.loc[loc, "alternate_policy_net_bps"] = float(net)
            out.loc[loc, "alternate_policy_gross_bps"] = float(net + COST_BPS)
            out.loc[loc, "alternate_policy_atr_bps"] = float(atrv)
            out.loc[loc, "alternate_policy_valid"] = bool(np.isfinite(net))
            out.loc[loc, "alternate_policy_entry_ts"] = pd.Timestamp(ts)
            out.loc[loc, "alternate_policy_label_available_ts"] = pd.Timestamp(ts) + pd.Timedelta(hours=12)
            out.loc[loc, "alternate_policy_symbol"] = symbol
    out["alternate_policy_entry_ts"] = pd.to_datetime(out["alternate_policy_entry_ts"], utc=True)
    out["alternate_policy_label_available_ts"] = pd.to_datetime(out["alternate_policy_label_available_ts"], utc=True)
    valid = out.alternate_policy_valid
    if valid.any():
        offsets = out.loc[valid, "alternate_policy_entry_ts"] - pd.to_datetime(out.loc[valid, "__ts__"], utc=True)
        if not (offsets == pd.Timedelta(hours=1)).all():
            raise AssertionError("alternate policy entry is not exactly +1h after feature timestamp")
    return out


def _tail(frame: pd.DataFrame, score: str, net: str, gross: str) -> dict[str, float]:
    return global_tail_metrics(frame, score_column=score, net_column=net, gross_column=gross, tails=TAILS)


def run(pred_path: Path, out: Path, path_root: Path, bars_root: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    pred = pd.read_parquet(pred_path)
    pred["__ts__"] = pd.to_datetime(pred["__ts__"], utc=True)
    required = {"candidate_id", "__ts__", "side_name", "fold", "score", "base_committee_expected_net_bps"}
    missing = sorted(required.difference(pred.columns))
    if missing:
        raise ValueError(f"prediction panel missing required columns: {missing}")
    labels = _policy_labels(pred, path_root, bars_root)
    panel = pred.merge(labels.drop(columns=["__ts__", "side_name"]), on="candidate_id", how="left", validate="one_to_one")
    if not panel.alternate_policy_valid.all():
        bad = panel.loc[~panel.alternate_policy_valid, ["candidate_id", "__ts__", "side_name"]]
        bad.to_parquet(out / "alternate_policy_unresolved.parquet", index=False)
        raise ValueError(f"alternate policy incomplete: {len(bad)} rows; see alternate_policy_unresolved.parquet")
    panel["month"] = panel["__ts__"].dt.strftime("%Y-%m")
    panel.to_parquet(out / "alternate_policy_oos_predictions.parquet", index=False, compression="zstd")

    # Persist the exact monthly expanding cutoffs used by the source replay.
    # This is an audit artifact, not a new fit: the source runner already
    # generated each held month with (__ts__ < month_start) and
    # (label_available_ts < month_start), while its residual consumed only
    # development OOF plus earlier final months.
    source_manifest = pred_path.parent / "run_manifest.json"
    source_payload = json.loads(source_manifest.read_text()) if source_manifest.exists() else {}
    ledger_path = Path(source_payload.get("ledger", "")) if source_payload.get("ledger") else None
    windows = []
    dev_rows = 0
    dev_path = pred_path.parent / "head_01_opportunity" / "development_oof_predictions.parquet"
    if dev_path.exists():
        dev_rows = int(len(pd.read_parquet(dev_path, columns=["candidate_id"])))
    ledger = None
    if ledger_path is not None and ledger_path.exists():
        ledger = pd.read_parquet(ledger_path, columns=["__ts__", "label_available_ts", "side_name"])
        ledger["__ts__"] = pd.to_datetime(ledger["__ts__"], utc=True)
        ledger["label_available_ts"] = pd.to_datetime(ledger["label_available_ts"], utc=True)
    for month in sorted(panel.month.unique()):
        start = pd.Timestamp(f"{month}-01", tz="UTC")
        if ledger is not None:
            fit = ledger[(ledger.__ts__ < start) & (ledger.label_available_ts < start)]
            base_rows = int(len(fit))
            base_long = int((fit.side_name.astype(str).str.lower() == "long").sum())
            base_short = int((fit.side_name.astype(str).str.lower() == "short").sum())
        else:
            base_rows = base_long = base_short = -1
        prior_final_rows = int((panel.month < month).sum())
        windows.append({
            "month": month, "base_fit_cutoff": str(start),
            "base_train_rows_matured_before_month": base_rows,
            "base_train_long_rows": base_long, "base_train_short_rows": base_short,
            "residual_fit_rows": int(dev_rows + prior_final_rows),
            "residual_development_rows": int(dev_rows), "residual_prior_final_rows": prior_final_rows,
            "held_rows": int((panel.month == month).sum()),
        })
    pd.DataFrame(windows).to_parquet(out / "monthly_training_windows.parquet", index=False)
    rows = []
    for month, group in panel.groupby("month", sort=True):
        for arm, score in (("single_baseline", "base_committee_expected_net_bps"), ("final_stack", "score")):
            m = _tail(group, score, "alternate_policy_net_bps", "alternate_policy_gross_bps")
            rows.append({"month": month, "arm": arm, "rows": len(group), **m})
        final = _tail(group, "score", "alternate_policy_net_bps", "alternate_policy_gross_bps")
        base = _tail(group, "base_committee_expected_net_bps", "alternate_policy_net_bps", "alternate_policy_gross_bps")
        rows.append({
            "month": month, "arm": "delta_final_minus_single", "rows": len(group),
            **{k: final[k] - base[k] for k in final if k.endswith("_net_bps") or k.endswith("_gross_bps")},
            **{k: np.nan for k in final if k.endswith("_rows")},
        })
    monthly = pd.DataFrame(rows)
    monthly.to_parquet(out / "alternate_policy_monthly_metrics.parquet", index=False)
    global_rows = []
    for arm, score in (("single_baseline", "base_committee_expected_net_bps"), ("final_stack", "score")):
        m = _tail(panel, score, "alternate_policy_net_bps", "alternate_policy_gross_bps")
        global_rows.append({"arm": arm, "rows": len(panel), **m})
    final = _tail(panel, "score", "alternate_policy_net_bps", "alternate_policy_gross_bps")
    base = _tail(panel, "base_committee_expected_net_bps", "alternate_policy_net_bps", "alternate_policy_gross_bps")
    global_rows.append({"arm": "delta_final_minus_single", "rows": len(panel), **{k: final[k] - base[k] for k in final if k.endswith("_net_bps") or k.endswith("_gross_bps")}, **{k: np.nan for k in final if k.endswith("_rows")}})
    global_metrics = pd.DataFrame(global_rows)
    global_metrics.to_parquet(out / "alternate_policy_global_metrics.parquet", index=False)
    manifest = {
        "schema": "complementary_base_heads_alternate_exit_v1",
        "prediction_source": str(pred_path),
        "prediction_source_manifest": str(pred_path.parent / "run_manifest.json"),
        "path_root": str(path_root), "bars_root": str(bars_root),
        "months": sorted(panel.month.unique().tolist()), "rows": int(len(panel)),
        "policy": {"entry": "__ts__ + 1h decision bar 15m open", "horizon_bars": HORIZON_BARS, "stop_atr": STOP_ATR, "trailing_activation_atr": TRAIL_ACTIVATION_ATR, "giveback_atr": TRAIL_GIVEBACK_ATR, "cost_bps_once": COST_BPS},
        "ranking": "global within month and pooled; no per-timestamp quota",
        "baseline": "base_committee_expected_net_bps (single retained head in the source replay)",
        "stack": "score (causal residual correction fit on development and prior months only)",
        "monthly_training_windows": str(out / "monthly_training_windows.parquet"),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lines = [
        "# Complementary base-head alternate-exit confirmation", "",
        "The model predictions are the frozen complementary-head replay. The economic outcome is the alternate frozen 15-minute policy, recomputed for every Jan–Aug 2024 candidate from the immutable path grid and 15-minute bars.", "",
        "## Policy", "",
        f"- Entry: decision timestamp + 1 hour, at the corresponding 15-minute open; horizon: {HORIZON_BARS} bars (12 hours).",
        f"- Stop: {STOP_ATR:g} ATR; trailing activation: {TRAIL_ACTIVATION_ATR:g} ATR; giveback: {TRAIL_GIVEBACK_ATR:g} ATR; cost: {COST_BPS:g} bps exactly once.",
        "- Global ranking is performed separately within each month and on the pooled panel; there is no per-timestamp quota.",
        "",
        "## Monthly results", "",
        "| Month | Arm | Rows | Top-1 net | Top-2 net | Top-5 net | Δ vs single |", "|---|---|---:|---:|---:|---:|---:|",
    ]
    for month in sorted(panel.month.unique()):
        b = monthly[(monthly.month == month) & (monthly.arm == "single_baseline")].iloc[0]
        f = monthly[(monthly.month == month) & (monthly.arm == "final_stack")].iloc[0]
        d = monthly[(monthly.month == month) & (monthly.arm == "delta_final_minus_single")].iloc[0]
        lines.append(f"| {month} | single baseline | {int(b.rows)} | {b.top1_net_bps:.2f} | {b.top2_net_bps:.2f} | {b.top5_net_bps:.2f} | — |")
        lines.append(f"| {month} | final stack | {int(f.rows)} | {f.top1_net_bps:.2f} | {f.top2_net_bps:.2f} | {f.top5_net_bps:.2f} | {d.top1_net_bps:.2f} / {d.top2_net_bps:.2f} / {d.top5_net_bps:.2f} |")
    lines += ["", "## Pooled results", "", "| Arm | Rows | Top-1 net | Top-2 net | Top-5 net |", "|---|---:|---:|---:|---:|"]
    for row in global_metrics.itertuples(index=False):
        lines.append(f"| {row.arm} | {int(row.rows)} | {row.top1_net_bps:.2f} | {row.top2_net_bps:.2f} | {row.top5_net_bps:.2f} |")
    lines += ["", "## Causality and integrity", "", "- Each final month’s base ranker is fitted on rows strictly before that month and with label availability before the month.", "- The residual is fitted only on development OOF rows and already-resolved prior final months; the held month is never used in its fit or mapping.", "- The alternate policy is evaluation-only and applies cost exactly once; it does not alter the H12 target used to fit the base head.", "- Entry offsets and label completeness are asserted for every scored row."]
    lines += ["", "## Expanding training windows", "", "| Month | Matured base rows | Residual rows | Held rows |", "|---|---:|---:|---:|"]
    for row in windows:
        lines.append(f"| {row['month']} | {row['base_train_rows_matured_before_month']} | {row['residual_fit_rows']} | {row['held_rows']} |")
    (out / "ALTERNATE_EXIT_CONFIRMATION_REPORT.md").write_text("\n".join(lines) + "\n")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, default=ROOT / "data_perp/artifacts/complementary_base_heads_20260808_v2/final_oos_predictions.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "data_perp/artifacts/complementary_base_heads_alternate_exit_20260808_v1")
    ap.add_argument("--path-root", type=Path, default=ROOT / "data_perp/artifacts/h12_query_path_grid_20260805_v2")
    ap.add_argument("--bars-root", type=Path, default=ROOT / "15m_ohlcv_perp")
    args = ap.parse_args()
    print(run(args.predictions, args.out, args.path_root, args.bars_root))
