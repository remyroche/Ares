#!/usr/bin/env python3
"""Score an offline BCF challenger ledger with a frozen BCF MC1 bundle.

This is intentionally an audit-only tool.  It applies the native mapper
chronologically, using only policy rows whose outcomes were available before
each decision.  It neither creates an execution bundle nor permits a
zero-support structural-prior fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_bcf_mc1_mapper import BCFMC1D2Bundle, FEATURES


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--bcf-mc1-bundle", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)

    bundle = BCFMC1D2Bundle.load(args.bcf_mc1_bundle)
    columns = [
        "candidate_id", "__decision_ts__", "side_name", *FEATURES,
        "policy_label_available_ts", "policy_path_valid", "policy_net_bps",
    ]
    ledger = pd.read_parquet(args.ledger, columns=columns)
    ledger["__decision_ts__"] = pd.to_datetime(ledger["__decision_ts__"], utc=True)
    ledger["policy_label_available_ts"] = pd.to_datetime(
        ledger["policy_label_available_ts"], utc=True,
    )
    if ledger["candidate_id"].duplicated().any():
        raise ValueError("ledger candidate identities must be unique")
    side = ledger["side_name"].astype(str).str.lower()
    if ledger.empty or not side.eq(bundle.side).all():
        raise ValueError("ledger side does not match frozen BCF MC1 bundle")

    rows: list[pd.DataFrame] = []
    for decision, current in ledger.groupby("__decision_ts__", sort=True):
        mapped = bundle.score(
            current,
            resolved_history=ledger,
            decision_ts=decision,
        )
        mapped["__decision_ts__"] = decision
        rows.append(mapped)
    output = pd.concat(rows, ignore_index=True)
    if len(output) != len(ledger) or output["candidate_id"].duplicated().any():
        raise AssertionError("mapper changed target-free candidate identity")

    args.out_dir.mkdir(parents=True)
    output.to_parquet(args.out_dir / "mc1_mapping.parquet", index=False, compression="zstd")
    timestamp = output.groupby("__decision_ts__", sort=True).agg(
        candidates=("candidate_id", "size"),
        available=("bcf_mc1_available", "sum"),
        admitted_ge30=("bcf_mc1_admitted_ge_30bps", "sum"),
        support_rows=("bcf_mc1_recent_support_rows", "max"),
        support_days=("bcf_mc1_recent_support_days", "max"),
        shift_bps=("bcf_mc1_recent_global_shift_bps", "max"),
    ).reset_index()
    timestamp.to_parquet(args.out_dir / "per_timestamp.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "strict_r3_bcf_mc1_challenger_audit_v1",
        "purpose": "offline_only_frozen_mapper_compatibility_diagnostic",
        "rows": int(len(output)),
        "timestamps": int(output["__decision_ts__"].nunique()),
        "available_rows": int(output["bcf_mc1_available"].sum()),
        "admitted_ge30_rows": int(output["bcf_mc1_admitted_ge_30bps"].sum()),
        "zero_support_timestamps": int((timestamp["support_rows"] == 0).sum()),
        "ledger": {"path": str(args.ledger), "sha256": _sha(args.ledger)},
        "bcf_mc1_bundle": {
            "path": str(args.bcf_mc1_bundle),
            "model_sha256": str(bundle.manifest["sha256"]["model_bundle"]),
            "bundle_id": str(bundle.manifest["bundle_id"]),
            "fit_cutoff": str(bundle.fit_cutoff),
        },
        "no_structural_prior_fallback": True,
        "not_promotable_without_train_to_inference_feature_parity": True,
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", **manifest}))


if __name__ == "__main__":
    main()
