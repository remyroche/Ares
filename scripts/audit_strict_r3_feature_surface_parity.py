#!/usr/bin/env python3
"""Compare a target-free inference probe to its same-lineage training surface."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _base_fields(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    fields = [str(name) for name in payload["base_fields_by_side"]["long"]]
    if len(fields) != 120 or len(set(fields)) != 120:
        raise ValueError("parity audit requires the frozen 120-field long contract")
    return fields


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-features", type=Path, required=True)
    parser.add_argument("--inference-features", type=Path, required=True)
    parser.add_argument("--feature-contract", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    fields = _base_fields(args.feature_contract)
    columns = ["candidate_id", "__decision_ts__", *fields]
    inference = pd.read_parquet(args.inference_features, columns=columns)
    ids = inference["candidate_id"].astype(str).tolist()
    if inference.empty or len(ids) != len(set(ids)):
        raise ValueError("inference feature probe must have unique nonempty candidates")
    dataset = ds.dataset(args.training_features, format="parquet")
    training = dataset.to_table(
        columns=columns,
        filter=ds.field("candidate_id").isin(ids),
    ).to_pandas()
    if len(training) != len(ids) or training["candidate_id"].duplicated().any():
        raise ValueError("training feature surface does not contain the exact probe candidate identity set")
    inference["__decision_ts__"] = pd.to_datetime(inference["__decision_ts__"], utc=True)
    training["__decision_ts__"] = pd.to_datetime(training["__decision_ts__"], utc=True)
    paired = inference.merge(
        training, on="candidate_id", suffixes=("__inference", "__training"), validate="one_to_one")
    time_match = paired["__decision_ts____inference"].eq(paired["__decision_ts____training"])
    if not time_match.all():
        raise ValueError("training and inference decision timestamps disagree")
    rows: list[dict[str, object]] = []
    candidate_max = np.zeros(len(paired), dtype=float)
    for field in fields:
        left = pd.to_numeric(paired[f"{field}__inference"], errors="coerce").to_numpy(float)
        right = pd.to_numeric(paired[f"{field}__training"], errors="coerce").to_numpy(float)
        missing_equal = np.array_equal(np.isnan(left), np.isnan(right))
        finite = np.isfinite(left) & np.isfinite(right)
        delta = np.zeros(len(left), dtype=float)
        delta[finite] = np.abs(left[finite] - right[finite])
        candidate_max = np.maximum(candidate_max, delta)
        exact = bool(missing_equal and np.array_equal(left[finite], right[finite]))
        rows.append({
            "field": field,
            "missingness_equal": missing_equal,
            "exact_numeric": exact,
            "max_abs_delta": float(delta.max(initial=0.0)),
            "mean_abs_delta": float(delta.mean()),
            "nonexact_rows": int((delta > 0.0).sum()) + int(np.isnan(left).sum() if not missing_equal else 0),
        })
    audit = pd.DataFrame(rows)
    candidate = paired.loc[:, ["candidate_id"]].copy()
    candidate["max_abs_delta"] = candidate_max
    args.out_dir.mkdir(parents=True)
    audit.to_parquet(args.out_dir / "field_audit.parquet", index=False)
    candidate.to_parquet(args.out_dir / "candidate_audit.parquet", index=False)
    manifest = {
        "schema": "strict_r3_same_lineage_feature_parity_audit_v1",
        "training_features": {"path": str(args.training_features), "sha256": _sha(args.training_features)},
        "inference_features": {"path": str(args.inference_features), "sha256": _sha(args.inference_features)},
        "feature_contract": {"path": str(args.feature_contract), "sha256": _sha(args.feature_contract)},
        "candidate_rows": int(len(paired)),
        "candidate_identity_exact": True,
        "decision_timestamp_exact": True,
        "fields": int(len(fields)),
        "fields_exact": int(audit["exact_numeric"].sum()),
        "fields_nonexact": int((~audit["exact_numeric"]).sum()),
        "global_max_abs_delta": float(audit["max_abs_delta"].max()),
        "numeric_tolerance": 1e-12,
        "passes_exact": bool(audit["exact_numeric"].all()),
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", **manifest}))


if __name__ == "__main__":
    main()
