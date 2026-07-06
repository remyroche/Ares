#!/usr/bin/env python3
"""Build a promotion gate for frozen smooth-penalty dual-scoring evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_GATES = {
    "min_eval_rows": 1000,
    "min_baseline_accepted": 100,
    "min_adjusted_rows": 25,
    "min_adjusted_accepted_or_near_boundary": 5,
    "min_acceptance_changes": 1,
}


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
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _safe_float(row: pd.Series, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    return float(value) if pd.notna(value) else default


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dual-dir", type=Path, required=True)
    parser.add_argument("--boundary-dir", type=Path, required=True)
    parser.add_argument("--readiness-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dual_manifest = json.loads((args.dual_dir / "dual_scoring_manifest.json").read_text())
    summary = pd.read_csv(args.dual_dir / "dual_scoring_summary.csv")
    overlaps = pd.read_csv(args.dual_dir / "dual_scoring_accepted_overlap.csv")
    boundary = pd.read_csv(args.boundary_dir / "boundary_summary.csv")
    selected = json.loads((args.readiness_dir / "selected_candidate_manifest.json").read_text())
    gates = dict(DEFAULT_GATES)

    baseline = summary.loc[summary["variant"].eq("baseline")]
    baseline_accepted = int(baseline["trade_count"].iloc[0]) if not baseline.empty and "trade_count" in baseline else 0
    eval_rows = int(dual_manifest.get("eval_rows", 0))
    rows: list[dict[str, Any]] = []
    for _, row in boundary.iterrows():
        variant = str(row["variant"])
        overlap = overlaps.loc[overlaps["variant"].astype(str).eq(variant)]
        accepted_overlap = float(overlap["jaccard"].iloc[0]) if not overlap.empty else np.nan
        adjusted_touch = int(row.get("adjusted_candidate_accepted", 0)) + int(row.get("adjusted_within_0p010_of_threshold", 0))
        checks = {
            "enough_eval_rows": eval_rows >= gates["min_eval_rows"],
            "enough_accepted_trades": baseline_accepted >= gates["min_baseline_accepted"],
            "enough_adjusted_rows": int(row["adjusted_rows"]) >= gates["min_adjusted_rows"],
            "touches_accepted_or_boundary": adjusted_touch >= gates["min_adjusted_accepted_or_near_boundary"],
            "changes_acceptance": int(row["adjusted_acceptance_changed"]) >= gates["min_acceptance_changes"],
        }
        rows.append(
            {
                "variant": variant,
                "passed_promotion_gate": bool(all(checks.values())),
                "failed_checks": ",".join(name for name, ok in checks.items() if not ok),
                "eval_rows": eval_rows,
                "baseline_accepted": baseline_accepted,
                "accepted_overlap": accepted_overlap,
                "adjusted_rows": int(row["adjusted_rows"]),
                "adjusted_share": _safe_float(row, "adjusted_share"),
                "adjusted_candidate_accepted": int(row["adjusted_candidate_accepted"]),
                "adjusted_within_0p010_of_threshold": int(row["adjusted_within_0p010_of_threshold"]),
                "adjusted_acceptance_changed": int(row["adjusted_acceptance_changed"]),
                **checks,
            }
        )
    decision = pd.DataFrame(rows)
    decision.to_csv(args.output_dir / "dual_scoring_promotion_gate.csv", index=False)
    manifest = {
        "generated_by": "build_frozen_dual_scoring_promotion_gate",
        "dual_dir": str(args.dual_dir),
        "boundary_dir": str(args.boundary_dir),
        "readiness_dir": str(args.readiness_dir),
        "eval_start": dual_manifest.get("eval_start"),
        "eval_end": dual_manifest.get("eval_end"),
        "strict_candidate": selected.get("selected_variant"),
        "aggressive_candidate": selected.get("pnl_dominant_variant"),
        "gates": gates,
        "status": "promotion_ready" if bool(decision["passed_promotion_gate"].any()) else "insufficient_binding_evidence",
    }
    (args.output_dir / "dual_scoring_promotion_gate_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Frozen Dual-Scoring Promotion Gate",
        "",
        f"Status: `{manifest['status']}`",
        f"Dual replay: `{args.dual_dir}`",
        f"Boundary report: `{args.boundary_dir}`",
        "",
        "## Decision",
        "",
        decision.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- A candidate can have good historical replay metrics but still fail this gate if current prospective rows are too few or non-binding.",
        "- `accepted_overlap == 1.0` with zero acceptance changes means the current dual-scoring evidence is operationally a smoke test, not a PnL/tail comparison.",
    ]
    (args.output_dir / "dual_scoring_promotion_gate_report.md").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
