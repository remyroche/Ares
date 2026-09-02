#!/usr/bin/env python3
"""Fit strict-prequential MC1 from sealed isolated router score folds.

This is the bounded operational counterpart to the main downstream runner.
It consumes already written target-free Current/BCF score panels and a
separately supplied immutable target-free feature root.  It performs no
consensus fitting and never writes labels into the score root.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_strict_r3_router_downstream as downstream  # noqa: E402


def run(*, target_free_root: Path, score_root: Path, policy_path: Path, out: Path) -> None:
    if out.exists():
        raise FileExistsError(out)
    target_free = target_free_root / "target_free_monthly"
    if not target_free.exists():
        raise FileNotFoundError(target_free)
    if not (score_root / "consensus_fold_audit.parquet").exists():
        raise FileNotFoundError(score_root / "consensus_fold_audit.parquet")
    policy = downstream._restrict_policy_to_source(downstream._load_policy(policy_path), target_free)
    out.mkdir(parents=True)
    combined, current_audit, bcf_audit = downstream._score_mc1(
        score_root, out, policy, downstream.EVALUATION_MONTHS,
    )
    (out / "run_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_economic_router_downstream_mc1_v1",
        "scope": "offline research only; strict-prequential MC1 from sealed score folds",
        "target_free_root": str(target_free_root), "score_root": str(score_root),
        "policy_path": str(policy_path),
        "evaluation_months": [f"{month:%Y-%m}" for month in downstream.EVALUATION_MONTHS],
        "current_fit_rows": int(len(current_audit)), "bcf_fit_rows": int(len(bcf_audit)),
        "combined_rows": int(len(combined)),
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-free-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(target_free_root=args.target_free_root, score_root=args.score_root,
        policy_path=args.policy_path, out=args.out)


if __name__ == "__main__":
    main()
