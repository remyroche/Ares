#!/usr/bin/env python3
"""Recompute the strict-R3 long-only decision chain on hourly :00 only.

This intentionally excludes the retired :15/:30/:45 research experiment.
It re-scores the frozen target-free :00 feature contract, fits the two
strict-prequential MC1 maps using the immutable canonical policy substrate,
then runs one causal dual-admission portfolio replay.  No exchange I/O or
live artifacts are touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = ROOT / (
    "data_perp/artifacts/strict_r3_full_stack_phase_h1_mayjul_20260818_v2"
)
PHASE = 0
START = "2026-05-01T00:00:00Z"
END = "2026-08-01T00:00:00Z"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(*, stage: str, command: list[str], receipt: Path) -> None:
    """Run one immutable stage and leave a local, non-model progress receipt."""
    receipt.write_text(json.dumps({"stage": stage, "status": "running"}, sort_keys=True) + "\n")
    completed = subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True)
    (receipt.parent / f"{stage}.stdout.log").write_text(completed.stdout)
    (receipt.parent / f"{stage}.stderr.log").write_text(completed.stderr)
    receipt.write_text(json.dumps({"stage": stage, "status": "complete"}, sort_keys=True) + "\n")


def _assert_target_free(raw_root: Path) -> None:
    manifest = json.loads((raw_root / "run_manifest.json").read_text())
    if manifest.get("outcome_columns_consumed") not in ([], None):
        raise RuntimeError("raw score ledger consumed policy/outcome columns")
    if manifest.get("phases") != [PHASE]:
        raise RuntimeError("raw score ledger is not phase-00 only")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    source = args.source_root
    out = args.out_dir
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")

    policy = source / "source_aligned_parent_policy_contract_v1/canonical_policy_contract.parquet"
    required = [
        source / "warmup_grid_phase0/target_free_candidate_population.parquet",
        source / "grid_phase0/target_free_candidate_population.parquet",
        source / "phase0_streamed_v2/checkpoint.json",
        policy,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"phase-00 source contract is incomplete: {missing}")

    out.mkdir(parents=True)
    progress = out / "progress.json"
    raw_history = out / "raw_history_phase00"
    raw_live = out / "raw_mayjul_phase00"
    _run(stage="raw_history", receipt=progress, command=[
        sys.executable, str(ROOT / "scripts/score_strict_r3_phase_h1_full_stack.py"),
        "--feature-root", str(source), "--candidate-root", str(source),
        "--out-dir", str(raw_history), "--phases", "0", "--phase-stream-tag", "v2",
        "--historical-native-ledger", "--score-end-exclusive", START,
    ])
    _assert_target_free(raw_history)
    _run(stage="raw_mayjul", receipt=progress, command=[
        sys.executable, str(ROOT / "scripts/score_strict_r3_phase_h1_full_stack.py"),
        "--feature-root", str(source), "--candidate-root", str(source),
        "--out-dir", str(raw_live), "--phases", "0", "--phase-stream-tag", "v2",
    ])
    _assert_target_free(raw_live)

    panels = out / "mc1_panels_phase00"
    _run(stage="assemble_mc1_panels", receipt=progress, command=[
        sys.executable, str(ROOT / "scripts/assemble_strict_r3_phase_h1_mc1_panels.py"),
        "--current-native-history-root", str(raw_history),
        "--bcf-native-history-root", str(raw_history),
        "--current-raw-root", str(raw_live), "--bcf-raw-root", str(raw_live),
        "--phase", str(PHASE), "--start", START, "--end", END, "--out-dir", str(panels),
    ])
    mc1 = out / "mc1_phase00_prequential"
    _run(stage="fit_prequential_mc1", receipt=progress, command=[
        sys.executable, str(ROOT / "scripts/replay_strict_r3_score_family_mc1_canonical_policy.py"),
        "--bcf-scores", str(panels / "bcf_scores_target_free.parquet"),
        "--current-scores", str(panels / "current_scores_target_free.parquet"),
        "--canonical-policy", str(policy), "--start", START, "--end", END,
        "--out-dir", str(mc1),
    ])
    pooled = out / "pooled_phase00"
    _run(stage="portfolio", receipt=progress, command=[
        sys.executable, str(ROOT / "scripts/replay_strict_r3_phase_h1_pooled_dual_portfolio.py"),
        "--phase", str(PHASE),
        str(mc1 / "predictions_current_v5_mc1_d2.parquet"),
        str(mc1 / "predictions_bcf_mc1_d2.parquet"),
        "--out-dir", str(pooled),
    ])

    manifest = {
        "schema": "strict_r3_phase00_full_pipeline_v1",
        "scope": "offline research only; phase 00 only; no exchange I/O",
        "phase_minutes": PHASE,
        "feature_contract": "frozen target-free phase0_streamed_v2",
        "policy_substrate": {"path": str(policy), "sha256": _sha(policy)},
        "raw_history": str(raw_history),
        "raw_mayjul": str(raw_live),
        "mc1": str(mc1),
        "pooled_replay": str(pooled),
        "target_free_until_policy_join": True,
        "status": "complete",
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
