#!/usr/bin/env python3
"""Evaluate sequential-funnel trial predictions and publish immutable tables."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from extreme_price_movements.sequential_funnel_evaluation import (  # noqa: E402
    SequentialFunnelEvaluationError, evaluate_funnel_trials, render_trial_report, validate_nested_oof_provenance,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--trial-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unit", choices=("return", "bps"), default="return")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    frame, manifest = pd.read_parquet(args.predictions), json.loads(args.trial_manifest.read_text())
    try:
        tables = evaluate_funnel_trials(frame, manifest, unit=args.unit)
        status = "COMPLETED_RESEARCH_ONLY_SEQUENTIAL_FUNNEL_EVALUATION"
    except SequentialFunnelEvaluationError as exc:
        tables = {"correctness_checks": validate_nested_oof_provenance(frame, manifest)}
        status = f"BLOCKED_SEQUENTIAL_FUNNEL_PROVENANCE: {exc}"
    stage = Path(tempfile.mkdtemp(dir=args.output.parent, prefix=f".{args.output.name}.staging-"))
    try:
        hashes = {}
        for name, table in tables.items():
            path = stage / f"{name}.parquet"; table.to_parquet(path, index=False, compression="zstd"); hashes[path.name] = _sha(path)
        if status.startswith("COMPLETED"):
            (stage / "FINAL_TARGET_STACK_REPORT.md").write_text(render_trial_report(tables, manifest))
        contract = {"schema": "sequential_funnel_evaluation_v1", "status": status, "input_rows": len(frame),
                    "predictions_sha256": _sha(args.predictions), "trial_manifest_sha256": _sha(args.trial_manifest),
                    "selection": "one pooled-global top-k per trial after common-unit mapping; no quotas or portfolio constraints",
                    "outputs_sha256": hashes}
        (stage / "run_manifest.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
        os.replace(stage, args.output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True); raise
    print(json.dumps(contract, sort_keys=True))


if __name__ == "__main__":
    main()
