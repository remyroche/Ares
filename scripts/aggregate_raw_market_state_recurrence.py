#!/usr/bin/env python3
"""Combine independently replayed weekly raw-state recurrence diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATTERN = "raw_market_state_backward_recurrence_20260726_v1_w*/summary.json"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/raw_market_state_backward_recurrence_20260726_v1"


def run(args: argparse.Namespace) -> dict[str, Path]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    reports = sorted((ROOT / "data_perp/artifacts").glob(args.pattern))
    if not reports:
        raise FileNotFoundError(f"no reports match {args.pattern}")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in reports]
    blocks = [block for payload in payloads for block in payload["blocks"]]
    blocks.sort(key=lambda block: block["start"])
    recurrence = [pd.read_csv(payload["outputs"]["recurrence"]) for payload in payloads]
    recurrence = [frame for frame in recurrence if not frame.empty]
    args.output_dir.mkdir(parents=True)
    recurrence_path = args.output_dir / "recurrence_by_prior_block_state.csv"
    (pd.concat(recurrence, ignore_index=True) if recurrence else pd.DataFrame()).to_csv(recurrence_path, index=False)
    gate_rows: list[dict[str, object]] = []
    for block in blocks:
        for side, details in block["sides"].items():
            if details.get("status") != "evaluated":
                continue
            gate = details["specialist_eligibility"]
            gate_rows.append(
                {
                    "evaluation_block": block["block"],
                    "start": block["start"],
                    "side_name": side,
                    "gate_side_eligible": bool(gate.get("eligible", False)),
                    "gate_reason": gate.get("reason"),
                    "recurring_states": json.dumps(gate.get("recurring_states", [])),
                    "rank_correlation": gate.get("rank_correlation"),
                    "sign_consistency": gate.get("sign_consistency"),
                    "minimum_within_block_effect_range": gate.get("minimum_within_block_effect_range"),
                    "resolved_prior_rows": details["resolved_prior_rows"],
                }
            )
    gates = pd.DataFrame(gate_rows)
    gates_path = args.output_dir / "specialist_eligibility_by_week.csv"
    gates.to_csv(gates_path, index=False)
    first = payloads[0]
    summary = {
        "schema": "raw_market_state_backward_recurrence_aggregate_v1",
        "status": "completed_diagnostic_not_specialist_promotion",
        "contract": first["contract"],
        "sources": first["sources"],
        "transition_v2_coordinate": first["transition_v2_coordinate"],
        "joined_rows": first["joined_rows"],
        "date_range": first["date_range"],
        "raw_features": first["raw_features"],
        "feature_coverage_threshold": first["feature_coverage_threshold"],
        "weekly_reports": [str(path.resolve()) for path in reports],
        "blocks": blocks,
        "gate_summary": {
            "evaluated_side_blocks": int(len(gates)),
            "eligible_side_blocks": int(gates["gate_side_eligible"].sum()) if len(gates) else 0,
            "eligible_blocks_both_sides": int(gates.groupby("evaluation_block")["gate_side_eligible"].all().sum()) if len(gates) else 0,
        },
        "outputs": {"recurrence": str(recurrence_path.resolve()), "gates": str(gates_path.resolve())},
    }
    report_path = args.output_dir / "summary.json"
    report_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    return {"report": report_path, "recurrence": recurrence_path, "gates": gates_path}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


if __name__ == "__main__":
    print(json.dumps({key: str(value) for key, value in run(_parser().parse_args()).items()}, indent=2))
