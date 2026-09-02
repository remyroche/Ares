#!/usr/bin/env python3
"""Join frozen family predictions to side and exact label-availability lineage."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--sidecar", type=Path, required=True)
    ap.add_argument("--execution-labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    pred = pd.read_parquet(args.predictions)
    side = pd.read_parquet(args.sidecar, columns=["fold", "candidate_id", "side_name"])
    labels = pd.read_parquet(
        args.execution_labels,
        columns=["fold", "candidate_id", "policy_label_available_ts"],
    )
    keys = ["fold", "candidate_id"]
    if pred.duplicated(keys).any() or side.duplicated(keys).any() or labels.duplicated(keys).any():
        raise ValueError("candidate identity is not unique in one of the input artifacts")
    out = pred.merge(side, on=keys, how="left", validate="one_to_one")
    out = out.merge(labels, on=keys, how="left", validate="one_to_one")
    if out["side_name"].isna().any() or out["policy_label_available_ts"].isna().any():
        raise ValueError("admission lineage join left missing side or label availability")
    if not out["side_name"].astype(str).str.lower().eq("long").all():
        raise ValueError("family admission input is not long-only")
    out["policy_label_available_ts"] = pd.to_datetime(out["policy_label_available_ts"], utc=True)
    # Candidate IDs recur across outer folds; the admission identity must not.
    out["admission_identity"] = out["fold"].astype(str) + "::" + out["candidate_id"].astype(str)
    if out["admission_identity"].duplicated().any():
        raise ValueError("fold-qualified admission identities are not unique")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False, compression="zstd")
    print({"rows": int(len(out)), "columns": int(len(out.columns)), "out": str(args.out)})


if __name__ == "__main__":
    main()
