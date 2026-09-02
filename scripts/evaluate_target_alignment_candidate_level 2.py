#!/usr/bin/env python3
"""Evaluate strict candidate-level T0--T4/supportive score arms; never fit."""

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
from extreme_price_movements.target_alignment_candidate_evaluator import AlignmentEvaluationError, Columns, evaluate_target_arms


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--entry-threshold", type=float, default=0.0)
    parser.add_argument("--score-unit", choices=("return", "bps"), default="return")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    frame = pd.read_parquet(args.predictions)
    try:
        tables = evaluate_target_arms(frame, columns=Columns(), score_unit=args.score_unit, entry_threshold=args.entry_threshold)
        status = "COMPLETED_STRICT_CANDIDATE_LEVEL_EVALUATION_RESEARCH_ONLY"
    except AlignmentEvaluationError as error:
        # Fail closed rather than falling back to a blocked/non-causal OOF
        # artifact.  The readiness record tells the caller exactly what must
        # be materialised before any economics are claimed.
        tables = {"correctness_checks": pd.DataFrame([{"check": "strict_prequential_candidate_oof", "passed": False, "value": str(error)}])}
        status = "BLOCKED_MISSING_STRICT_PREQUENTIAL_CANDIDATE_LINEAGE"
    stage = Path(tempfile.mkdtemp(dir=args.output.parent, prefix=f".{args.output.name}.staging-"))
    try:
        hashes = {}
        for name, table in tables.items():
            if "value" in table.columns:
                table = table.copy()
                table["value"] = table["value"].map(str)
            path = stage / f"{name}.parquet"; table.to_parquet(path, index=False, compression="zstd"); hashes[path.name] = _sha256(path)
        manifest = {"schema": "target_alignment_candidate_evaluation_v1", "status": status, "input": {"path": str(args.predictions), "sha256": _sha256(args.predictions), "rows": len(frame)}, "selection": "one pooled global tail per target/supportive arm; candidate-id ties; no timestamp/side/asset/portfolio constraint", "policy": "candidate score > entry threshold only; optional clean gate supported by library; no portfolio logic", "outputs_sha256": hashes}
        (stage / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(stage, args.output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True); raise
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
