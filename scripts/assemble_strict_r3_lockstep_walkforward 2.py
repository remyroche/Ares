#!/usr/bin/env python3
"""Assemble immutable per-block lock-step scores for immediate EV replay.

The block workers persist target-free score checkpoints independently.  This
assembler joins their policy outcomes only after all scores are frozen, making
the aggregate suitable for the exact-producer reserve calibrator and replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks-root", type=Path, required=True)
    parser.add_argument("--policy-outcomes", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable assembled output exists: {args.out_dir}")
    manifest_paths = sorted(args.blocks_root.glob("bundles/cutoff=*/scores/block_manifest.json"))
    if not manifest_paths:
        raise FileNotFoundError("no completed lock-step block manifests")
    manifests = [json.loads(path.read_text()) for path in manifest_paths]
    geometry = {str(manifest["geometry_bundle_sha256"]) for manifest in manifests}
    if len(geometry) != 1:
        raise ValueError("lock-step blocks use different geometry/K9 identities")
    if not all(bool(manifest.get("shared_upstream_conversion_cutoff")) for manifest in manifests):
        raise ValueError("a block did not use a shared upstream/conversion cutoff")
    if not all(int(manifest.get("reserve_days", 0)) == 42 for manifest in manifests):
        raise ValueError("a block did not use the full 42-day reserve")
    reference_paths = [path.parent / "reserve_target_free_scores.parquet" for path in manifest_paths]
    held_paths = [path.parent / "held_target_free_scores.parquet" for path in manifest_paths]
    if any(not path.is_file() for path in [*reference_paths, *held_paths]):
        raise FileNotFoundError("a completed lock-step block lacks target-free scores")
    reference = pd.concat([pd.read_parquet(path) for path in reference_paths], ignore_index=True)
    held = pd.concat([pd.read_parquet(path) for path in held_paths], ignore_index=True)
    reference_identity = [
        "candidate_id", "conversion_bundle_sha256", "upstream_bundle_sha256", "calibration_activation_ts",
    ]
    if reference.duplicated(reference_identity).any():
        raise AssertionError("assembled calibration reserve duplicates a producer identity")
    if held["candidate_id"].duplicated().any():
        raise AssertionError("assembled held candidate population is not unique")
    outcomes = pd.read_parquet(args.policy_outcomes)
    if outcomes["candidate_id"].duplicated().any():
        raise ValueError("policy outcome ledger has duplicate candidate identities")
    outcome_columns = [
        "policy_path_valid", "policy_gross_bps", "policy_net_bps",
        "policy_exit_bar_15m", "policy_exit_reason", "policy_entry_price",
        "policy_exit_price", "policy_label_available_ts", "policy_outcome_source",
    ]
    available = [column for column in outcome_columns if column in outcomes]
    required = {"policy_path_valid", "policy_net_bps", "policy_label_available_ts"}
    if missing := sorted(required.difference(available)):
        raise ValueError(f"policy outcome ledger lacks: {missing}")
    labelled = held.merge(
        outcomes.loc[:, ["candidate_id", *available]], on="candidate_id", how="left", validate="one_to_one",
    )
    args.out_dir.mkdir(parents=True)
    reference.to_parquet(args.out_dir / "immediate_calibration_reference_scores.parquet", index=False, compression="zstd")
    held.to_parquet(args.out_dir / "walkforward_predictions.parquet", index=False, compression="zstd")
    labelled.to_parquet(args.out_dir / "walkforward_scored_label_ledger.parquet", index=False, compression="zstd")
    pd.DataFrame(manifests).sort_values("cutoff").to_parquet(args.out_dir / "block_audit.parquet", index=False)
    manifest = {
        "schema": "strict_r3_lockstep_walkforward_assembly_v1",
        "blocks_root": str(args.blocks_root),
        "policy_outcomes": str(args.policy_outcomes),
        "policy_outcomes_sha256": _sha(args.policy_outcomes),
        "blocks": len(manifests),
        "geometry_bundle_sha256": next(iter(geometry)),
        "target_free_reference_rows": int(len(reference)),
        "target_free_held_rows": int(len(held)),
        "outcomes_consumed_during_scoring": [],
        "outcomes_joined_after_scoring": True,
        "contract": (
            "per-refit shared 42-day OOS reserve; exact producer map; frozen geometry; "
            "point-in-time candidates; post-score policy-outcome join"
        ),
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), **manifest}))


if __name__ == "__main__":
    main()
