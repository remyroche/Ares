#!/usr/bin/env python3
"""Run frozen apply -> replay -> boundary -> promotion-gate workflow.

This is an orchestration helper for contextual TP/SL A/B candidates.  It does
not fit new models or select new thresholds; it applies already-frozen bundles
to one candidate ledger and evaluates whether the resulting replay has enough
binding evidence for promotion.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_READINESS = (
    ROOT / "data_perp/reports/contextual_tp_sl_wfrecent_smooth_penalty_combo_readiness_v3_materialized_20260701"
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        val = float(value)
        return val if np.isfinite(val) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _parse_bundle(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        path = Path(raw)
        return path.name, path
    label, path = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Empty bundle label in {raw!r}")
    return label, Path(path.strip())


def _run(cmd: list[str], log_rows: list[dict[str, Any]]) -> None:
    print(" ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=ROOT, check=True)
    log_rows.append({"cmd": cmd, "returncode": int(completed.returncode)})


def _candidate_args(apply_audit_path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    audit = pd.read_csv(apply_audit_path)
    args: list[str] = []
    rows: list[dict[str, Any]] = []
    for _, row in audit.iterrows():
        combo = str(row["combo"])
        output = str(row["output"])
        args.extend(["--candidate", f"{combo}={output}"])
        rows.append(
            {
                "combo": combo,
                "output": output,
                "candidate_rows": int(row.get("candidate_rows", 0)),
                "penalized_rows": int(row.get("penalized_rows", 0)),
                "penalized_share": float(row.get("penalized_share", 0.0)),
            }
        )
    return args, rows


def _rule_family(rule_name: Any) -> str:
    text = str(rule_name).lower()
    if text == "__combined__":
        return "combined"
    if "recent_hr" in text or "recent_perf" in text:
        return "recent_hit_rate_surprise"
    if "uncertainty" in text:
        return "uncertainty"
    if "drift" in text:
        return "drift"
    if "ood" in text:
        return "OOD"
    if "q85" in text or "aggressive" in text:
        return "rank_tail"
    return "other"


def _summarize_component_families(label: str, apply_dir: Path, out_dir: Path) -> pd.DataFrame:
    path = apply_dir / "smooth_penalty_combo_apply_component_audit.csv"
    if not path.exists():
        return pd.DataFrame()
    audit = pd.read_csv(path)
    if audit.empty or "rule_name" not in audit.columns:
        return pd.DataFrame()
    audit = audit[~audit["rule_name"].astype(str).eq("__combined__")].copy()
    if audit.empty:
        return pd.DataFrame()
    audit["bundle"] = label
    audit["feature_family"] = audit["rule_name"].map(_rule_family)
    grouped = (
        audit.groupby(["bundle", "feature_family"], dropna=False)
        .agg(
            rule_count=("rule_name", "nunique"),
            combo_count=("combo", "nunique"),
            penalized_rows_max=("penalized_rows", "max"),
            penalized_rows_sum=("penalized_rows", "sum"),
            penalized_share_max=("penalized_share", "max"),
            mean_penalty_min=("mean_penalty", "min"),
            min_penalty_min=("min_penalty", "min"),
        )
        .reset_index()
        .sort_values(["bundle", "feature_family"])
    )
    grouped.to_csv(out_dir / f"{label}_component_family_summary.csv", index=False)
    return grouped


def _summarize_result(label: str, apply_dir: Path, dual_dir: Path, boundary_dir: Path, gate_dir: Path) -> dict[str, Any]:
    summary_path = dual_dir / "dual_scoring_summary.csv"
    overlap_path = dual_dir / "dual_scoring_accepted_overlap.csv"
    adjustment_path = dual_dir / "dual_scoring_adjustment_summary.csv"
    boundary_path = boundary_dir / "boundary_summary.csv"
    gate_path = gate_dir / "dual_scoring_promotion_gate.csv"
    rows: dict[str, Any] = {"bundle": label}
    family = _summarize_component_families(label, apply_dir, gate_dir.parent)
    if not family.empty:
        rows["tested_feature_families"] = ",".join(sorted(family["feature_family"].dropna().astype(str).unique()))
        for fam in ("recent_hit_rate_surprise", "drift", "OOD", "uncertainty"):
            fam_rows = family[family["feature_family"].eq(fam)]
            rows[f"{fam}_penalized_rows_max"] = (
                int(fam_rows["penalized_rows_max"].max()) if not fam_rows.empty else 0
            )

    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        baseline = summary[summary["variant"].eq("baseline")]
        rows["baseline_trade_count"] = int(baseline["trade_count"].iloc[0]) if not baseline.empty else 0
        variants = summary[~summary["variant"].eq("baseline")].copy()
        if not variants.empty:
            best = variants.sort_values("delta_net_pnl", ascending=False, na_position="last").iloc[0]
            rows["best_delta_pnl_variant"] = str(best["variant"])
            rows["best_delta_net_pnl"] = float(best.get("delta_net_pnl", np.nan))
            rows["best_delta_full_sl_rate"] = float(best.get("delta_full_sl_rate", np.nan))
    if adjustment_path.exists():
        adj = pd.read_csv(adjustment_path)
        adj = adj[~adj["variant"].eq("baseline")]
        rows["max_adjusted_rows"] = int(adj["adjusted_rows"].max()) if not adj.empty else 0
        rows["max_adjusted_share"] = float(adj["adjusted_share"].max()) if not adj.empty else 0.0
    if overlap_path.exists():
        overlap = pd.read_csv(overlap_path)
        rows["min_accepted_jaccard"] = float(overlap["jaccard"].min()) if not overlap.empty else np.nan
        rows["total_entrants"] = int(overlap["entrants"].sum()) if not overlap.empty else 0
        rows["total_removed"] = int(overlap["removed"].sum()) if not overlap.empty else 0
    if boundary_path.exists():
        boundary = pd.read_csv(boundary_path)
        rows["max_adjusted_acceptance_changed"] = (
            int(boundary["adjusted_acceptance_changed"].max()) if not boundary.empty else 0
        )
        rows["max_adjusted_candidate_accepted"] = (
            int(boundary["adjusted_candidate_accepted"].max()) if not boundary.empty else 0
        )
        rows["max_adjusted_near_threshold_0p010"] = (
            int(boundary["adjusted_within_0p010_of_threshold"].max()) if not boundary.empty else 0
        )
    if gate_path.exists():
        gate = pd.read_csv(gate_path)
        rows["promotion_ready"] = bool(gate["passed_promotion_gate"].any()) if not gate.empty else False
        rows["failed_checks"] = ";".join(sorted(set(gate.get("failed_checks", pd.Series(dtype=str)).dropna().astype(str))))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Flat baseline candidate ledger parquet.")
    parser.add_argument("--bundle", action="append", required=True, help="label=path or path. Repeatable.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-start", required=True)
    parser.add_argument("--eval-end", default="")
    parser.add_argument("--readiness-dir", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--market-mode", default="perps", choices=["spot", "perps"])
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    bundles = [_parse_bundle(raw) for raw in args.bundle]
    command_log: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []

    for label, bundle_dir in bundles:
        bundle_dir = bundle_dir.resolve()
        work = out_dir / label
        apply_dir = work / "apply"
        dual_dir = work / "dual_scoring"
        boundary_dir = work / "boundary"
        gate_dir = work / "promotion_gate"
        for cur in (apply_dir, dual_dir, boundary_dir, gate_dir):
            cur.mkdir(parents=True, exist_ok=True)

        _run(
            [
                sys.executable,
                "scripts/freeze_apply_wfrecent_smooth_penalty_combo_bundle.py",
                "apply",
                "--bundle-dir",
                str(bundle_dir),
                "--candidates",
                str(args.baseline),
                "--output-dir",
                str(apply_dir),
            ],
            command_log,
        )
        candidate_args, apply_rows = _candidate_args(apply_dir / "smooth_penalty_combo_apply_audit.csv")

        replay_cmd = [
            sys.executable,
            "scripts/replay_frozen_smooth_penalty_dual_scoring.py",
            "--baseline",
            str(args.baseline),
            *candidate_args,
            "--output-dir",
            str(dual_dir),
            "--eval-start",
            str(args.eval_start),
            "--market-mode",
            str(args.market_mode),
        ]
        if args.eval_end:
            replay_cmd.extend(["--eval-end", str(args.eval_end)])
        _run(replay_cmd, command_log)

        _run(
            [
                sys.executable,
                "scripts/analyze_frozen_dual_scoring_boundary.py",
                "--dual-dir",
                str(dual_dir),
                "--output-dir",
                str(boundary_dir),
            ],
            command_log,
        )
        _run(
            [
                sys.executable,
                "scripts/build_frozen_dual_scoring_promotion_gate.py",
                "--dual-dir",
                str(dual_dir),
                "--boundary-dir",
                str(boundary_dir),
                "--readiness-dir",
                str(args.readiness_dir),
                "--output-dir",
                str(gate_dir),
            ],
            command_log,
        )
        row = _summarize_result(label, apply_dir, dual_dir, boundary_dir, gate_dir)
        row["bundle_dir"] = str(bundle_dir)
        row["applied_combos"] = len(apply_rows)
        result_rows.append(row)

    results = pd.DataFrame(result_rows)
    results.to_csv(out_dir / "frozen_dual_scoring_gate_summary.csv", index=False)
    manifest = {
        "generated_by": Path(__file__).name,
        "baseline": str(args.baseline),
        "eval_start": str(args.eval_start),
        "eval_end": str(args.eval_end),
        "readiness_dir": str(args.readiness_dir),
        "bundles": [{"label": label, "path": str(path)} for label, path in bundles],
        "commands": command_log,
    }
    (out_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n")

    lines = [
        "# Frozen Dual-Scoring Gate Run",
        "",
        f"Baseline: `{args.baseline}`",
        f"Evaluation start: `{args.eval_start}`",
        f"Evaluation end: `{args.eval_end or 'open'}`",
        "",
        "## Summary",
        "",
        results.to_markdown(index=False) if not results.empty else "_No bundles processed._",
        "",
        "## Interpretation",
        "",
        "- `promotion_ready=False` means the replay did not yet provide enough binding evidence.",
        "- This script only applies frozen bundles; it does not refit thresholds or tune candidates.",
    ]
    (out_dir / "frozen_dual_scoring_gate_run_report.md").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
