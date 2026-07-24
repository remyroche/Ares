#!/usr/bin/env python3
"""Run the bounded recency and leaf-support ablation for side residual experts.

The underlying expert remains the canonical base-correctness architecture:
it learns realized net EV minus a train-only side x archetype expected-EV map
of the frozen base score.  This wrapper deliberately does not re-run feature
selection or HPO.  Every arm reuses one frozen side-local feature/parameter
contract and differs only in either causal observation recency or the
full-fit leaf-support scaling requested for the experiment.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_meta_v9_ev_mapped_side_residual_ablation.py"
DEFAULT_SOURCE_ROOT = ROOT / (
    "data_perp/reports/"
    "s59_h5_signalclose_causal_stagec_packb_residual_only_hpo150_wf30_20260721_v1"
)
DEFAULT_SELECTION = DEFAULT_SOURCE_ROOT / "staged_selection_hpo_manifest.json"
DEFAULT_HANDOFF = ROOT / (
    "data_perp/reports/"
    "s59_h5_signalclose_causal_stagec_packb_sliding365_wf30_20260721_v1/"
    "meta_handoff_top30_resolved_v3/train_meta_regime_handoff.parquet"
)
DEFAULT_LEDGER = DEFAULT_HANDOFF.with_name("s52_trailing_regime_scored_ledger.parquet")
DEFAULT_FEATURE_DIR = ROOT / "data_perp/features/20260711_070000"
DEFAULT_OUT = ROOT / "data_perp/reports/meta_residual_recency_leaf_ablation_20260722_v1"


def _resolved_end_exclusive(path: Path) -> pd.Timestamp:
    schema = pq.read_schema(path)
    required = {"__ts__", "__label_path_end_ts__"}
    missing = required - set(schema.names)
    if missing:
        raise ValueError(f"{path} lacks causal label columns: {sorted(missing)}")
    frame = pd.read_parquet(path, columns=["__ts__", "__label_path_end_ts__"])
    signal = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    resolved = pd.to_datetime(frame["__label_path_end_ts__"], utc=True, errors="coerce")
    valid = signal.notna() & resolved.notna() & resolved.ge(signal)
    if not valid.any():
        raise ValueError(f"{path} has no resolved labels")
    # The scored ledger itself contains only rows with a materialized outcome;
    # its final signal timestamp is therefore the authoritative evaluation end.
    return signal.loc[valid].max() + pd.Timedelta(hours=1)


def _metric_row(out_dir: Path, *, arm: str, kind: str, value: float) -> dict[str, object]:
    metric_path = out_dir / "metrics.csv"
    if not metric_path.exists():
        raise RuntimeError(f"Missing metrics from completed arm: {metric_path}")
    metrics = pd.read_csv(metric_path)
    score = "score_base_ev_residual_expert_hier_mapped"
    overall = metrics.loc[
        (metrics["scope"] == "overall") & (metrics["model"] == score)
    ]
    if len(overall) != 1:
        raise RuntimeError(f"Missing unique overall residual score metric in {metric_path}")
    row = overall.iloc[0].to_dict()
    row.update({"arm": arm, "ablation_kind": kind, "value": value, "out_dir": str(out_dir)})
    return row


def _run_arm(args: argparse.Namespace, *, arm: str, kind: str, value: float, half_life: float, alpha: float) -> dict[str, object]:
    out = args.out_dir / arm
    # Parameter-only sweeps are resumable.  A completed arm already contains
    # an immutable prediction/metric artifact, so rerunning it only wastes
    # the same full one-year fit and can obscure which configuration produced
    # the reported value.
    if (out / "metrics.csv").is_file() and (out / "manifest.json").is_file():
        print(f"[arm-resume] {arm} using completed artifact {out}", flush=True)
        return _metric_row(out, arm=arm, kind=kind, value=value)
    command = [
        sys.executable,
        str(RUNNER),
        "--handoff", str(args.handoff),
        "--scored-ledger", str(args.scored_ledger),
        "--feature-dir", str(args.feature_dir),
        "--out-dir", str(out),
        "--reuse-selection-manifest", str(args.selection_manifest),
        "--source-mode", "current_handoff",
        "--backbone-score", "base",
        "--calibration-month", args.calibration_month,
        "--eval-start", args.eval_start,
        "--eval-end", args.eval_end,
        "--oos-fit-mode", "frozen_pre_eval",
        "--max-train-days", str(args.max_train_days),
        "--sample-weight-half-life-months", str(half_life),
        "--min-leaf-scaling-alpha", str(alpha),
        "--hpo-reference-rows", str(args.hpo_reference_rows),
        "--skip-final-refit",
    ]
    print("[arm]", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)
    return _metric_row(out, arm=arm, kind=kind, value=value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--scored-ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--selection-manifest", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--calibration-month", default="2026-06")
    parser.add_argument("--eval-start", default="2026-07-01")
    parser.add_argument("--eval-end", default="2026-07-21")
    parser.add_argument("--max-train-days", type=int, default=365)
    parser.add_argument("--hpo-reference-rows", type=int, default=45_000)
    parser.add_argument("--run", action="store_true", help="Execute arms; otherwise write the matrix plan only.")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    requested_end = pd.Timestamp(args.eval_end, tz="UTC")
    available_end = _resolved_end_exclusive(args.handoff)
    matrix = [{"arm": "baseline", "kind": "half_life_months", "value": 0.0, "half_life_months": 0.0, "min_leaf_scaling_alpha": 0.0}]
    matrix += [
        {"arm": f"half_life_{month}m", "kind": "half_life_months", "value": float(month), "half_life_months": float(month), "min_leaf_scaling_alpha": 0.0}
        for month in (2, 3, 4, 5, 6)
    ]
    matrix += [
        {"arm": f"leaf_alpha_{alpha:.1f}", "kind": "leaf_scaling_alpha", "value": float(alpha), "half_life_months": 0.0, "min_leaf_scaling_alpha": float(alpha)}
        for alpha in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    ]
    plan = {
        "runner": str(RUNNER),
        "architecture": "base_correctness_residual_net_ev_after_1pct",
        "selection_manifest": str(args.selection_manifest),
        "handoff": str(args.handoff),
        "scored_ledger": str(args.scored_ledger),
        "max_train_days": int(args.max_train_days),
        "calibration_month": args.calibration_month,
        "eval_start": args.eval_start,
        "eval_end": args.eval_end,
        "resolved_outcome_end_exclusive": available_end.isoformat(),
        "matrix": matrix,
    }
    (args.out_dir / "ablation_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    if not args.run:
        print(json.dumps(plan, indent=2))
        return
    if requested_end > available_end:
        raise RuntimeError(
            "Cannot evaluate the requested OOS range on resolved outcomes: "
            f"requested end {requested_end.isoformat()}, but the current scored ledger only "
            f"resolves through {available_end.isoformat()}. Materialize causal outcomes first; "
            "the runner intentionally refuses a partial July 1-20 claim."
        )
    rows = []
    for spec in matrix:
        rows.append(_run_arm(
            args,
            arm=str(spec["arm"]),
            kind=str(spec["kind"]),
            value=float(spec["value"]),
            half_life=float(spec["half_life_months"]),
            alpha=float(spec["min_leaf_scaling_alpha"]),
        ))
    report = pd.DataFrame(rows).sort_values(["ablation_kind", "value"], kind="stable")
    report.to_csv(args.out_dir / "ablation_metrics.csv", index=False)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
