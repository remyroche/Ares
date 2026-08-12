#!/usr/bin/env python3
"""Attach a policy-outcome ledger to already-scored strict-R3 candidates.

The score input must be target-free.  This utility exists for fair historical
replays when a more complete, candidate-keyed policy-path backfill becomes
available after scores have been frozen.  It never trains or alters a score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


POLICY_COLUMNS = (
    "policy_path_valid",
    "policy_gross_bps",
    "policy_net_bps",
    "policy_exit_bar_15m",
    "policy_exit_reason",
    "policy_entry_price",
    "policy_exit_price",
    "policy_label_available_ts",
    "policy_outcome_source",
    "policy_cost_bps",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-free-predictions", type=Path, required=True)
    parser.add_argument("--policy-outcomes", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")

    predictions = pd.read_parquet(args.target_free_predictions)
    outcomes = pd.read_parquet(args.policy_outcomes)
    if "candidate_id" not in predictions or predictions["candidate_id"].duplicated().any():
        raise ValueError("target-free prediction ledger requires unique candidate_id")
    if "candidate_id" not in outcomes or outcomes["candidate_id"].duplicated().any():
        raise ValueError("policy outcome ledger requires unique candidate_id")
    available = [column for column in POLICY_COLUMNS if column in outcomes.columns]
    required = {"policy_path_valid", "policy_net_bps", "policy_label_available_ts"}
    if required.difference(available):
        raise ValueError(f"policy outcome ledger lacks {sorted(required.difference(available))}")
    output = predictions.merge(
        outcomes.loc[:, ["candidate_id", *available]],
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    if len(output) != len(predictions) or output["candidate_id"].duplicated().any():
        raise AssertionError("outcome join changed target-free candidate identities")
    missing = int(output["policy_label_available_ts"].isna().sum())
    if missing:
        raise ValueError(f"policy outcome backfill does not cover {missing} scored candidates")
    args.out_dir.mkdir(parents=True)
    output.to_parquet(args.out_dir / "walkforward_scored_label_ledger.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "strict_r3_post_score_policy_outcome_join_v1",
        "target_free_predictions": str(args.target_free_predictions),
        "target_free_predictions_sha256": _sha(args.target_free_predictions),
        "policy_outcomes": str(args.policy_outcomes),
        "policy_outcomes_sha256": _sha(args.policy_outcomes),
        "rows": int(len(output)),
        "policy_path_valid_rows": int(output["policy_path_valid"].fillna(False).astype(bool).sum()),
        "score_mutation": "none; outcomes joined only after all scores were generated",
        "identity": "candidate_id one-to-one exact coverage",
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), **manifest}))


if __name__ == "__main__":
    main()
