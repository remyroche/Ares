#!/usr/bin/env python3
"""Audit an early target-free F72/Router panel against a frozen reference.

This is an offline, read-only comparison used before an earlier feature panel
is admitted to a strict OOF history.  It proves candidate identity, ordered
feature availability, finite-value alignment, and numeric parity for the
feature union declared by the early panel manifest.  It never opens policy
labels, outcomes, models, admission, portfolio, live, or exchange state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


IDENTITY = ("candidate_id", "__decision_ts__", "side_name")
PROHIBITED = frozenset({
    "policy_net_bps", "policy_gross_bps", "policy_path_valid",
    "policy_label_available_ts", "exact_net_bps", "target", "label",
})
RTOL = 1e-6
ATOL = 1e-7


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _audit(
    probe: pd.DataFrame,
    reference: pd.DataFrame,
    fields: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, bool]]:
    if probe["candidate_id"].duplicated().any() or reference["candidate_id"].duplicated().any():
        raise AssertionError("candidate identities must be unique before parity audit")
    joined = probe.merge(
        reference,
        on=list(IDENTITY),
        how="outer",
        suffixes=("_probe", "_reference"),
        indicator=True,
        validate="one_to_one",
    )
    identity = pd.DataFrame([{
        "probe_rows": int(len(probe)),
        "reference_rows": int(len(reference)),
        "both_rows": int(joined["_merge"].eq("both").sum()),
        "probe_only_rows": int(joined["_merge"].eq("left_only").sum()),
        "reference_only_rows": int(joined["_merge"].eq("right_only").sum()),
    }])
    common = joined.loc[joined["_merge"].eq("both")]
    rows: list[dict[str, object]] = []
    for field in fields:
        left = pd.to_numeric(common[f"{field}_probe"], errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(common[f"{field}_reference"], errors="coerce").to_numpy(dtype=float)
        left_finite = np.isfinite(left)
        right_finite = np.isfinite(right)
        finite = left_finite & right_finite
        delta = np.abs(left[finite] - right[finite])
        equal = np.isclose(left[finite], right[finite], rtol=RTOL, atol=ATOL)
        rows.append({
            "field": field,
            "finite_pairs": int(finite.sum()),
            "finite_mismatch_rows": int(np.sum(left_finite != right_finite)),
            "equal_fraction": float(equal.mean()) if len(equal) else np.nan,
            "median_abs_delta": float(np.median(delta)) if len(delta) else np.nan,
            "p95_abs_delta": float(np.quantile(delta, .95)) if len(delta) else np.nan,
            "max_abs_delta": float(delta.max()) if len(delta) else np.nan,
        })
    field = pd.DataFrame(rows).sort_values(["equal_fraction", "p95_abs_delta"], ascending=[True, False], kind="stable")
    strict = {
        "identity_exact": bool(identity.loc[0, "probe_only_rows"] == 0 and identity.loc[0, "reference_only_rows"] == 0),
        "all_finite_masks_match": bool(field["finite_mismatch_rows"].eq(0).all()),
        "all_values_within_tolerance": bool(field["equal_fraction"].ge(1.0).all()),
    }
    return identity, field, strict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    manifest = json.loads(args.probe_manifest.read_text())
    fields = tuple(str(value) for value in manifest.get("feature_contract", ()))
    if not fields or len(fields) != len(set(fields)):
        raise AssertionError("probe manifest has no unique ordered feature contract")
    probe_columns = set(pq.ParquetFile(args.probe).schema.names)
    reference_columns = set(pq.ParquetFile(args.reference).schema.names)
    required = set(IDENTITY) | set(fields)
    if required - probe_columns:
        raise AssertionError("probe lacks declared identity or feature field")
    if required - reference_columns:
        raise AssertionError("reference lacks declared identity or feature field")
    if PROHIBITED & probe_columns:
        raise AssertionError("target-free probe contains prohibited outcome column")
    if PROHIBITED & reference_columns:
        raise AssertionError("target-free reference contains prohibited outcome column")
    probe = pd.read_parquet(args.probe, columns=[*IDENTITY, *fields])
    reference = pd.read_parquet(args.reference, columns=[*IDENTITY, *fields])
    identity, field, strict = _audit(probe, reference, fields)
    args.out.mkdir(parents=True)
    identity.to_parquet(args.out / "identity_parity.parquet", index=False, compression="zstd")
    field.to_parquet(args.out / "field_parity.parquet", index=False, compression="zstd")
    _once(args.out / "correctness_report.json", {
        "target_free_probe": True,
        "target_free_reference": True,
        "ordered_feature_contract_from_probe_manifest": True,
        **strict,
    })
    _once(args.out / "run_manifest.json", {
        "schema": "strict_r3_f72_early_feature_parity_v1",
        "scope": "offline read-only target-free feature parity audit only",
        "probe": {"path": str(args.probe.resolve()), "sha256": _sha256(args.probe)},
        "reference": {"path": str(args.reference.resolve()), "sha256": _sha256(args.reference)},
        "probe_manifest": {"path": str(args.probe_manifest.resolve()), "sha256": _sha256(args.probe_manifest)},
        "feature_count": len(fields),
        "numeric_tolerance": {"rtol": RTOL, "atol": ATOL},
        "strict_result": strict,
    })
    print(json.dumps({"event": "complete", "out": str(args.out), **strict}), flush=True)


if __name__ == "__main__":
    main()
