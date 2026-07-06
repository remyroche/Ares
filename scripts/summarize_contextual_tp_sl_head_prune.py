#!/usr/bin/env python3
"""Summarize head-pruned contextual TP/SL materialized replays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _load_manifest(path: Path) -> Dict[str, Any]:
    manifest_path = path / "combo_replay_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _weekly(path: Path) -> pd.DataFrame:
    p = path / "combo_replay_weekly_metrics.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing weekly metrics: {p}")
    frame = pd.read_csv(p)
    if "period_type" in frame.columns:
        frame = frame.loc[frame["period_type"].eq("week")].copy()
    return frame


def _global_row(label: str, path: Path) -> Dict[str, Any]:
    manifest = _load_manifest(path)
    metrics = manifest.get("metrics", {})
    return {
        "label": label,
        "path": str(path),
        "combo_id": manifest.get("combo_id"),
        "active_heads": ",".join(manifest.get("active_heads", [])),
        "candidate_rows": manifest.get("candidate_rows"),
        "candidate_start": manifest.get("candidate_start"),
        "candidate_end": manifest.get("candidate_end"),
        "net_pnl": metrics.get("net_pnl"),
        "gross_pnl": metrics.get("gross_pnl"),
        "trade_count": metrics.get("trade_count"),
        "full_sl_rate": metrics.get("full_sl_rate"),
        "timeout_rate": metrics.get("timeout_rate"),
        "max_drawdown": metrics.get("max_drawdown"),
        "strategy_concentration": metrics.get("strategy_concentration"),
    }


def _pair_row(candidate_label: str, candidate_path: Path, baseline_label: str, baseline_path: Path) -> Dict[str, Any]:
    cand = _global_row(candidate_label, candidate_path)
    base = _global_row(baseline_label, baseline_path)
    c_week = _weekly(candidate_path)[["week", "net_pnl"]].rename(columns={"net_pnl": "candidate_week_pnl"})
    b_week = _weekly(baseline_path)[["week", "net_pnl"]].rename(columns={"net_pnl": "baseline_week_pnl"})
    merged = c_week.merge(b_week, on="week", how="inner")
    delta = pd.to_numeric(merged["candidate_week_pnl"], errors="coerce") - pd.to_numeric(
        merged["baseline_week_pnl"], errors="coerce"
    )
    delta = delta.dropna().to_numpy(dtype=float)
    rec = {
        "candidate": candidate_label,
        "matched_baseline": baseline_label,
        "active_heads": cand["active_heads"],
        "candidate_net_pnl": cand["net_pnl"],
        "baseline_net_pnl": base["net_pnl"],
        "delta_net_pnl": float(cand["net_pnl"] - base["net_pnl"]),
        "candidate_trade_count": cand["trade_count"],
        "baseline_trade_count": base["trade_count"],
        "delta_trade_count": int(cand["trade_count"] - base["trade_count"]),
        "candidate_full_sl_rate": cand["full_sl_rate"],
        "baseline_full_sl_rate": base["full_sl_rate"],
        "delta_full_sl_rate": float(cand["full_sl_rate"] - base["full_sl_rate"]),
        "candidate_max_drawdown": cand["max_drawdown"],
        "baseline_max_drawdown": base["max_drawdown"],
        "delta_max_drawdown": float(cand["max_drawdown"] - base["max_drawdown"]),
        "week_count": int(delta.size),
        "delta_week_sum_pnl": float(delta.sum()) if delta.size else np.nan,
        "delta_week_q10_pnl": float(np.nanpercentile(delta, 10)) if delta.size else np.nan,
        "delta_week_q20_pnl": float(np.nanpercentile(delta, 20)) if delta.size else np.nan,
        "delta_worst_week_pnl": float(np.nanmin(delta)) if delta.size else np.nan,
        "positive_week_delta_share": float(np.nanmean(delta > 0.0)) if delta.size else np.nan,
    }
    rec["economics_tail_pass"] = bool(
        rec["delta_net_pnl"] >= 0.0
        and rec["delta_full_sl_rate"] <= 0.0
        and rec["delta_max_drawdown"] >= 0.0
        and rec["delta_week_q10_pnl"] >= -1000.0
        and rec["delta_week_q20_pnl"] >= 0.0
        and rec["positive_week_delta_share"] >= 0.60
    )
    return rec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, help="label=path. Repeatable.")
    parser.add_argument("--pair", action="append", required=True, help="candidate=baseline label pair. Repeatable.")
    parser.add_argument("--reference-label", default="")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs: Dict[str, Path] = {}
    for raw in args.run:
        if "=" not in raw:
            raise ValueError(f"Invalid --run {raw!r}; expected label=path")
        label, path_s = raw.split("=", 1)
        runs[label] = Path(path_s)
    global_rows = [_global_row(label, path) for label, path in runs.items()]
    global_df = pd.DataFrame(global_rows)
    if args.reference_label:
        ref = global_df.loc[global_df["label"].eq(args.reference_label)]
        if not ref.empty:
            ref_row = ref.iloc[0]
            for col in ("net_pnl", "trade_count", "full_sl_rate", "max_drawdown"):
                global_df[f"delta_vs_{args.reference_label}_{col}"] = (
                    pd.to_numeric(global_df[col], errors="coerce") - float(ref_row[col])
                )

    pair_rows: List[Dict[str, Any]] = []
    for raw in args.pair:
        if "=" not in raw:
            raise ValueError(f"Invalid --pair {raw!r}; expected candidate=baseline")
        candidate, baseline = raw.split("=", 1)
        pair_rows.append(_pair_row(candidate, runs[candidate], baseline, runs[baseline]))
    pair_df = pd.DataFrame(pair_rows)

    global_df.to_csv(args.out_dir / "head_prune_global_summary.csv", index=False)
    pair_df.to_csv(args.out_dir / "head_prune_matched_pair_summary.csv", index=False)
    payload = {
        "generated_by": "summarize_contextual_tp_sl_head_prune",
        "out_dir": str(args.out_dir),
        "runs": {label: str(path) for label, path in runs.items()},
        "pairs": list(args.pair),
        "reference_label": str(args.reference_label),
    }
    (args.out_dir / "head_prune_summary_manifest.json").write_text(
        json.dumps(_json_safe(payload), indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Contextual TP/SL Head-Prune Summary",
        "",
        "Replay type: development/proxy replay with costs included.",
        "",
        "## Global Runs",
        "",
        global_df.to_markdown(index=False),
        "",
        "## Matched Candidate vs Static Baseline Pairs",
        "",
        pair_df.to_markdown(index=False),
    ]
    (args.out_dir / "head_prune_summary_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(_json_safe({"out_dir": str(args.out_dir), "runs": len(runs), "pairs": len(pair_df)}), indent=2))


if __name__ == "__main__":
    main()
