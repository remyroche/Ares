#!/usr/bin/env python3
"""Bootstrap weekly replay deltas for contextual TP/SL candidates.

The materialized comparison already reports point estimates versus a baseline.
This script resamples complete weeks with replacement to estimate how robust
the PnL and weekly-tail improvements are under block bootstrap uncertainty.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def _bootstrap_metrics(values: np.ndarray, *, n_boot: int, seed: int) -> Dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(int(n_boot), values.size), endpoint=False)
    samples = values[idx]
    sum_delta = samples.sum(axis=1)
    mean_delta = samples.mean(axis=1)
    q10_delta = np.percentile(samples, 10, axis=1)
    q20_delta = np.percentile(samples, 20, axis=1)
    worst_delta = samples.min(axis=1)
    positive_share = (samples > 0.0).mean(axis=1)

    def ci(arr: np.ndarray, pct: float) -> float:
        return float(np.percentile(arr, pct))

    all_gate = (
        (sum_delta >= 0.0)
        & (q10_delta >= -1000.0)
        & (q20_delta >= 0.0)
        & (positive_share >= 0.60)
    )
    return {
        "week_count": int(values.size),
        "point_sum_delta_net_pnl": float(values.sum()),
        "point_mean_delta_net_pnl": float(values.mean()),
        "point_q10_delta_week_pnl": float(np.percentile(values, 10)),
        "point_q20_delta_week_pnl": float(np.percentile(values, 20)),
        "point_worst_delta_week_pnl": float(values.min()),
        "point_positive_week_delta_share": float((values > 0.0).mean()),
        "boot_sum_delta_p05": ci(sum_delta, 5),
        "boot_sum_delta_p50": ci(sum_delta, 50),
        "boot_sum_delta_p95": ci(sum_delta, 95),
        "boot_mean_delta_p05": ci(mean_delta, 5),
        "boot_mean_delta_p50": ci(mean_delta, 50),
        "boot_mean_delta_p95": ci(mean_delta, 95),
        "boot_q10_delta_p05": ci(q10_delta, 5),
        "boot_q10_delta_p50": ci(q10_delta, 50),
        "boot_q10_delta_p95": ci(q10_delta, 95),
        "boot_q20_delta_p05": ci(q20_delta, 5),
        "boot_q20_delta_p50": ci(q20_delta, 50),
        "boot_q20_delta_p95": ci(q20_delta, 95),
        "boot_positive_share_p05": ci(positive_share, 5),
        "boot_positive_share_p50": ci(positive_share, 50),
        "boot_positive_share_p95": ci(positive_share, 95),
        "prob_sum_delta_positive": float((sum_delta > 0.0).mean()),
        "prob_q10_delta_ge_minus_1000": float((q10_delta >= -1000.0).mean()),
        "prob_q20_delta_positive": float((q20_delta >= 0.0).mean()),
        "prob_positive_share_ge_60": float((positive_share >= 0.60).mean()),
        "prob_all_tail_gates": float(all_gate.mean()),
    }


def _bootstrap_frame(
    frame: pd.DataFrame,
    *,
    baseline_label: str,
    group_cols: List[str],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    group_keys = ["label", *group_cols]
    for key, group in frame.groupby(group_keys, sort=False, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        key_values = dict(zip(group_keys, key))
        if str(key_values["label"]) == str(baseline_label):
            continue
        values = pd.to_numeric(group["delta_net_pnl"], errors="coerce").to_numpy(dtype=np.float64)
        rec: Dict[str, Any] = dict(key_values)
        rec.update(_bootstrap_metrics(values, n_boot=int(n_boot), seed=int(seed) + len(rows)))
        rows.append(rec)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        [*group_cols, "prob_all_tail_gates", "boot_sum_delta_p50", "point_sum_delta_net_pnl"],
        ascending=[True for _ in group_cols] + [False, False, False],
    )


def _write_markdown_table(path: Path, title: str, body: List[str], frame: Optional[pd.DataFrame], columns: List[str]) -> None:
    existing = [col for col in columns if frame is not None and col in frame.columns]
    lines = [
        f"# {title}",
        "",
        *body,
        "",
        frame[existing].to_markdown(index=False) if frame is not None and not frame.empty else "_No candidate rows._",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--baseline-label", default="static")
    parser.add_argument("--n-bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    week_path = args.comparison_dir / "materialized_replay_week_comparison.csv"
    head_week_path = args.comparison_dir / "materialized_replay_head_week_comparison.csv"
    global_path = args.comparison_dir / "materialized_replay_global_comparison.csv"
    if not week_path.exists():
        raise FileNotFoundError(f"Missing weekly comparison: {week_path}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    weekly = pd.read_csv(week_path)
    global_df = pd.read_csv(global_path) if global_path.exists() else pd.DataFrame()
    out = _bootstrap_frame(
        weekly,
        baseline_label=str(args.baseline_label),
        group_cols=[],
        n_boot=int(args.n_bootstrap),
        seed=int(args.seed),
    )
    if not out.empty:
        if not global_df.empty:
            out = out.merge(
                global_df[
                    [
                        "label",
                        "delta_net_pnl",
                        "delta_full_sl_rate",
                        "delta_max_drawdown",
                    ]
                ].rename(
                    columns={
                        "delta_net_pnl": "global_delta_net_pnl",
                        "delta_full_sl_rate": "global_delta_full_sl_rate",
                        "delta_max_drawdown": "global_delta_max_drawdown",
                    }
                ),
                on="label",
                how="left",
            )
    out.to_csv(args.out_dir / "weekly_delta_bootstrap.csv", index=False)

    head_out = pd.DataFrame()
    if head_week_path.exists():
        head_week = pd.read_csv(head_week_path)
        head_out = _bootstrap_frame(
            head_week,
            baseline_label=str(args.baseline_label),
            group_cols=["head"],
            n_boot=int(args.n_bootstrap),
            seed=int(args.seed) + 10000,
        )
        head_out.to_csv(args.out_dir / "head_week_delta_bootstrap.csv", index=False)
    payload = {
        "generated_by": "bootstrap_contextual_tp_sl_weekly_deltas",
        "comparison_dir": str(args.comparison_dir),
        "out_dir": str(args.out_dir),
        "baseline_label": str(args.baseline_label),
        "n_bootstrap": int(args.n_bootstrap),
        "seed": int(args.seed),
        "candidate_count": int(len(out)),
        "head_candidate_count": int(len(head_out)),
    }
    (args.out_dir / "weekly_delta_bootstrap_manifest.json").write_text(
        json.dumps(_json_safe(payload), indent=2),
        encoding="utf-8",
    )
    columns = [
        "label",
        "week_count",
        "point_sum_delta_net_pnl",
        "boot_sum_delta_p05",
        "boot_sum_delta_p50",
        "boot_sum_delta_p95",
        "point_q10_delta_week_pnl",
        "boot_q10_delta_p05",
        "boot_q10_delta_p50",
        "point_q20_delta_week_pnl",
        "boot_q20_delta_p05",
        "boot_q20_delta_p50",
        "point_positive_week_delta_share",
        "boot_positive_share_p05",
        "prob_sum_delta_positive",
        "prob_q10_delta_ge_minus_1000",
        "prob_q20_delta_positive",
        "prob_positive_share_ge_60",
        "prob_all_tail_gates",
        "global_delta_full_sl_rate",
        "global_delta_max_drawdown",
    ]
    body = [
        f"Comparison directory: `{args.comparison_dir}`",
        f"Baseline label: `{args.baseline_label}`",
        f"Bootstrap samples: `{args.n_bootstrap}`",
        "",
        "This is a block bootstrap over complete weekly PnL deltas from a development/proxy replay.",
        "It is robustness evidence, not untouched OOS evidence.",
    ]
    _write_markdown_table(
        args.out_dir / "weekly_delta_bootstrap_report.md",
        "Contextual TP/SL Weekly Delta Bootstrap",
        body,
        out,
        columns,
    )
    head_columns = ["label", "head", *columns[1:]]
    _write_markdown_table(
        args.out_dir / "head_week_delta_bootstrap_report.md",
        "Contextual TP/SL Head-Week Delta Bootstrap",
        body,
        head_out,
        head_columns,
    )
    print(json.dumps(_json_safe(payload), indent=2))


if __name__ == "__main__":
    main()
