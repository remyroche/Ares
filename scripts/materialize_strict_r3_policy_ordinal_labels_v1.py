#!/usr/bin/env python3
"""Materialise the policy-ordinal B-head label sidecar from a policy ledger.

This deliberately narrow producer avoids a dependency on the auxiliary TBM
families.  It is appropriate when only the frozen B head is being rebuilt:
the B contract consumes the rich-policy ordinal target, not auxiliary path
grades.  Candidate identities are read target-free from the path ledger and
missing policy outcomes remain invalid rather than becoming economic failures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


IDENTITY = ["candidate_id", "__decision_ts__", "side_name"]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    for item in paths:
        digest.update(str(item).encode())
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _ordinal(net: pd.Series) -> np.ndarray:
    return np.searchsorted([0.0, 50.0, 100.0, 200.0, 400.0], pd.to_numeric(net, errors="coerce").to_numpy(float), side="right").astype(np.int8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-root", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start-month", default="2024-02")
    parser.add_argument("--end-month", default="2026-07")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    policy = pd.read_parquet(
        args.policy_path,
        columns=["candidate_id", "policy_path_valid", "policy_net_bps", "policy_label_available_ts"],
    )
    policy["policy_path_valid"] = policy["policy_path_valid"].fillna(False).astype(bool)
    policy["policy_net_bps"] = pd.to_numeric(policy["policy_net_bps"], errors="coerce")
    policy["policy_label_available_ts"] = pd.to_datetime(policy["policy_label_available_ts"], utc=True, errors="coerce")
    if policy.candidate_id.duplicated().any():
        raise AssertionError("policy ledger has duplicate candidate IDs")
    args.out.mkdir(parents=True)
    audit: list[dict[str, object]] = []
    for source in sorted(args.path_root.glob("month=*/side=long.parquet")):
        token = source.parent.name.split("=", 1)[1]
        if token < args.start_month or token > args.end_month:
            continue
        candidates = pd.read_parquet(source, columns=IDENTITY)
        candidates["__decision_ts__"] = pd.to_datetime(candidates["__decision_ts__"], utc=True, errors="raise")
        if candidates.candidate_id.duplicated().any():
            raise AssertionError(f"{source}: duplicate candidate IDs")
        frame = candidates.merge(policy, on="candidate_id", how="left", validate="one_to_one")
        valid = frame["policy_path_valid"].fillna(False).astype(bool) & np.isfinite(frame["policy_net_bps"])
        output = frame.loc[:, IDENTITY].copy()
        output["label_available_ts"] = frame["policy_label_available_ts"].where(
            frame["policy_label_available_ts"].notna(), output["__decision_ts__"] + pd.Timedelta(hours=12),
        )
        output["policy_ordinal_base_valid"] = valid
        output["policy_ordinal_base_grade"] = _ordinal(frame["policy_net_bps"])
        output["policy_net_bps"] = frame["policy_net_bps"]
        if not (output["label_available_ts"] - output["__decision_ts__"]).eq(pd.Timedelta(hours=12)).all():
            raise AssertionError(f"{token}: B labels do not resolve exactly at H12")
        destination = args.out / f"month={token}"
        destination.mkdir()
        output.to_parquet(destination / "b0_replacement_targets.parquet", index=False, compression="zstd")
        audit.append({
            "month": token, "rows": int(len(output)), "policy_valid": int(valid.sum()),
            "missing_policy_rows": int(frame["policy_path_valid"].isna().sum()),
        })
    if not audit:
        raise ValueError("no requested path-label partitions found")
    pd.DataFrame(audit).to_parquet(args.out / "coverage_and_grade_audit.parquet", index=False, compression="zstd")
    payload = {
        "schema": "strict_r3_policy_ordinal_b_labels_v1",
        "scope": "offline B-head labels only; never inference features or candidate filtering",
        "target": "policy_ordinal_base_grade",
        "valid": "policy_path_valid and finite policy_net_bps",
        "bins": [0.0, 50.0, 100.0, 200.0, 400.0],
        "label_availability": "exactly decision plus 12 hours",
        "path_root": str(args.path_root), "path_root_sha256": _sha(args.path_root),
        "policy_path": str(args.policy_path), "policy_sha256": _sha(args.policy_path),
        "months": [row["month"] for row in audit],
        "candidate_contract": "all path-ledger candidates are retained; missing policy outcomes are invalid labels",
    }
    descriptor = os.open(args.out / "run_manifest.json", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"event": "complete", "out": str(args.out), "months": len(audit), "rows": sum(row["rows"] for row in audit)}))


if __name__ == "__main__":
    main()
