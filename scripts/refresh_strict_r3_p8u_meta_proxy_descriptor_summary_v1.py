#!/usr/bin/env python3
"""Refresh an immutable strict-OOF descriptor summary from its fold panels.

This is a narrow lineage-preserving repair: a prior descriptor summary omitted
some diagnostics that were already present in its strict-OOF fold panel.  It
never reads scores, policy outcomes, path outcomes, MC1 outputs, or live data.
The output retains the source fold panel through a hard link and records the
source receipt, so it cannot change the underlying experiment population.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

import build_strict_r3_p8u_meta_downstream_proxy_descriptors_v1 as descriptors


SCHEMA = "strict_r3_p8u_meta_proxy_descriptor_summary_refresh_v1"
REQUIRED = {
    "ic_base_5_10", "ic_base_10_20", "ic_base_20_30",
    "useful_upgrade_ev", "false_upgrade_ev",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=500)
    parser.add_argument("--bootstrap-seed", type=int, default=1729)
    args = parser.parse_args()
    source = args.source_root.resolve()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    if args.bootstrap_iterations < 100:
        raise ValueError("bootstrap iterations must be at least 100")
    for name in ("run_manifest.json", "correctness_report.json", "trial_fold_descriptors.parquet", "trial_weekly_descriptors.parquet"):
        if not (source / name).exists():
            raise FileNotFoundError(source / name)
    source_correctness = json.loads((source / "correctness_report.json").read_text())
    if not all(value is True for value in source_correctness.values()):
        raise AssertionError("source descriptor correctness receipt is not clean")
    fold = pd.read_parquet(source / "trial_fold_descriptors.parquet")
    weekly = pd.read_parquet(source / "trial_weekly_descriptors.parquet")
    missing = sorted(REQUIRED.difference(fold.columns))
    if missing:
        raise AssertionError(f"source fold descriptor fields missing: {missing}")
    if fold.duplicated(["score_root", "trial", "held_month"]).any():
        raise AssertionError("source fold descriptor identities are not unique")
    summary = descriptors._bootstrap_summary(fold, iterations=args.bootstrap_iterations, seed=args.bootstrap_seed)
    summary = descriptors._attach_cross_fold_stability(summary, fold, weekly)
    missing = sorted(REQUIRED.difference(summary.columns))
    if missing:
        raise AssertionError(f"refreshed descriptor fields missing: {missing}")
    out.mkdir(parents=True)
    # A hard link guarantees byte-identical strict-OOF fold/weekly inputs while
    # avoiding needless copies of immutable experiment data.
    os.link(source / "trial_fold_descriptors.parquet", out / "trial_fold_descriptors.parquet")
    os.link(source / "trial_weekly_descriptors.parquet", out / "trial_weekly_descriptors.parquet")
    summary.to_parquet(out / "trial_descriptor_summary.parquet", index=False, compression="zstd")
    _once(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "strict-OOF descriptor-summary refresh only; no score/outcome/MC1/live read or mutation",
        "source_root": str(source),
        "source_manifest_sha256": _sha(source / "run_manifest.json"),
        "source_correctness_sha256": _sha(source / "correctness_report.json"),
        "source_fold_sha256": _sha(source / "trial_fold_descriptors.parquet"),
        "source_weekly_sha256": _sha(source / "trial_weekly_descriptors.parquet"),
        "summary_fields_added": sorted(REQUIRED),
        "bootstrap": {"iterations": int(args.bootstrap_iterations), "seed": int(args.bootstrap_seed)},
        "selection_authority": "none; this only restores already-computed strict-OOF diagnostics",
    })
    _once(out / "correctness_report.json", {
        "source_strict_oof_receipt_clean": True,
        "fold_and_weekly_panels_are_hard_linked_byte_identical": True,
        "summary_includes_all_required_proxy_diagnostics": True,
        "no_score_policy_path_mc1_or_live_read": True,
    })
    print(out)


if __name__ == "__main__":
    main()
