#!/usr/bin/env python3
"""Run the canonical post-score Cell-day, R5, portfolio and report funnel.

This thin orchestrator exists to prevent manual contract drift between the
four immutable stages.  It never fits upstream/base/conversion models and it
never overwrites artifacts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "data_perp/artifacts/strict_r3_schema_v2_simple_policy_targetfree_long_pre2025_20260809_v3/winner.json"
INTEGRATION = ROOT / "config/strict_r3_cell_day_residual_trust_posterior_28d_challenger_v1.json"


def _run(command: list[str]) -> None:
    print(json.dumps({"event": "stage_begin", "command": command}), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lockstep-dir", type=Path, required=True)
    parser.add_argument("--prequential-ledger", type=Path, required=True)
    parser.add_argument("--policy-outcomes", type=Path, required=True)
    parser.add_argument("--evaluation-start", required=True)
    parser.add_argument("--evaluation-end", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--initial-wallet", type=float, default=1000.0)
    args = parser.parse_args()
    lockstep = args.lockstep_dir.resolve()
    prequential = args.prequential_ledger.resolve()
    outcomes = args.policy_outcomes.resolve()
    output = args.out_dir.resolve()
    if output.exists():
        raise FileExistsError(f"immutable canonical postprocessing output exists: {output}")
    scored = lockstep / "walkforward_scored_label_ledger.parquet"
    reference = lockstep / "immediate_calibration_reference_scores.parquet"
    for path in (scored, reference, prequential, outcomes, POLICY, INTEGRATION):
        if not path.exists():
            raise FileNotFoundError(path)
    output.mkdir(parents=True)
    map_dir = output / "cell_day_map"
    r5_dir = output / "r5_walkforward"
    portfolio_dir = output / "portfolio"
    report_dir = output / "waterfall"

    _run([
        sys.executable, "scripts/ablate_strict_r3_cell_day_bayesian_ev_mapping.py",
        "--reference-scores", str(reference),
        "--policy-outcomes", str(outcomes),
        "--held-ledger", str(scored),
        "--window-days", "28",
        "--out-dir", str(map_dir),
    ])
    _run([
        sys.executable, "scripts/run_strict_r3_cell_day_residual_trust_walkforward.py",
        "--scored-ledger", str(scored),
        "--cell-day-provenance", str(map_dir / "cell_day_bayesian_selection.parquet"),
        "--integration-contract", str(INTEGRATION),
        "--evaluation-start", args.evaluation_start,
        "--evaluation-end", args.evaluation_end,
        "--out-dir", str(r5_dir),
    ])
    _run([
        sys.executable, "scripts/replay_strict_r3_forward_portfolio.py",
        "--schema", "current-v5",
        "--scored-label-ledger", str(scored),
        "--geometry-mode", "frozen",
        "--admission-provenance", str(
            map_dir / "score_and_cell_day_admission_provenance.parquet"
        ),
        "--cell-day-trust-oof-predictions", str(
            r5_dir / "cell_day_residual_trust_oof_predictions.parquet"
        ),
        "--cell-day-trust-integration", str(INTEGRATION),
        "--evaluation-start", args.evaluation_start,
        "--evaluation-end", args.evaluation_end,
        "--out-dir", str(portfolio_dir),
        "--initial-wallet", str(args.initial_wallet),
        "--perp-leverage", "7",
        "--margin-slot-wallet-fraction", "0.10",
        "--policy-json", str(POLICY),
        "--disable-canonical-n5",
    ])
    _run([
        sys.executable, "scripts/report_strict_r3_r5_canonical_waterfall.py",
        "--scored-ledger", str(scored),
        "--prequential-ledger", str(prequential),
        "--cell-day-provenance", str(
            map_dir / "score_and_cell_day_admission_provenance.parquet"
        ),
        "--r5-predictions", str(
            r5_dir / "cell_day_residual_trust_oof_predictions.parquet"
        ),
        "--portfolio-dir", str(portfolio_dir),
        "--evaluation-start", args.evaluation_start,
        "--evaluation-end", args.evaluation_end,
        "--out-dir", str(report_dir),
    ])
    manifest = {
        "schema": "strict_r3_r5_canonical_postprocessing_v1",
        "lockstep_dir": str(lockstep),
        "prequential_ledger": str(prequential),
        "policy_outcomes": str(outcomes),
        "evaluation_start": args.evaluation_start,
        "evaluation_end_exclusive": args.evaluation_end,
        "stages": ["cell_day_map", "r5_walkforward", "portfolio", "waterfall"],
        "policy": str(POLICY),
        "integration": str(INTEGRATION),
        "canonical_n5_active": False,
        "status": "complete",
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "out_dir": str(output), **manifest}))


if __name__ == "__main__":
    main()
