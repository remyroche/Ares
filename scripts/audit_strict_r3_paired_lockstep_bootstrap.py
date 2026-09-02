#!/usr/bin/env python3
"""Exact parity audit for independent paired lock-step bootstrap runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    keys = ["candidate_id", "__decision_ts__", "__score_role__"]
    left = pd.read_parquet(args.reference / "score_decomposition.parquet").sort_values(keys, kind="stable").reset_index(drop=True)
    right = pd.read_parquet(args.replay / "score_decomposition.parquet").sort_values(keys, kind="stable").reset_index(drop=True)
    keys_exact = len(left) == len(right) and left.loc[:, keys].equals(right.loc[:, keys])
    numeric = [
        column for column in left.columns if column in right.columns
        and pd.api.types.is_numeric_dtype(left[column])
        and not pd.api.types.is_bool_dtype(left[column])
    ]
    maximum = 0.0
    for column in numeric:
        a = pd.to_numeric(left[column], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(right[column], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.any():
            maximum = max(maximum, float(np.max(np.abs(a[mask] - b[mask]))))
    state_files = ["causal_geometry_k9_history.parquet", "same_model_prior28_cdf_values.npy"]
    state_exact = all(
        _sha(args.reference / "geometry_k9_state" / name)
        == _sha(args.replay / "geometry_k9_state" / name)
        for name in state_files
    )
    manifest_left = json.loads((args.reference / "run_manifest.json").read_text())
    manifest_right = json.loads((args.replay / "run_manifest.json").read_text())
    lineage_exact = all(
        manifest_left.get(key) == manifest_right.get(key)
        for key in ("conversion_bundle_sha256", "upstream_bundle_sha256", "source_hashes")
    )
    passed = bool(keys_exact and maximum == 0.0 and state_exact and lineage_exact)
    receipt = {
        "schema": "strict_r3_paired_lockstep_bootstrap_parity_v1",
        "passed": passed,
        "reference": str(args.reference),
        "replay": str(args.replay),
        "rows": int(len(left)),
        "candidate_identity_exact": keys_exact,
        "numeric_columns_compared": len(numeric),
        "max_abs_numeric_delta": maximum,
        "geometry_state_exact": state_exact,
        "producer_lineage_exact": lineage_exact,
        "order_submission": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise SystemExit(json.dumps(receipt, sort_keys=True))
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
