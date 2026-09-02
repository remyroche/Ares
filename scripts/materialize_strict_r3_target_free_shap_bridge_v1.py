#!/usr/bin/env python3
"""Link an audited contiguous target-free F72-SHAP ledger without rewriting it.

This is deliberately a source-lineage utility.  It accepts completed strict
OOF SHAP roots, verifies their contract/schema/month ownership, then makes
immutable hard-link receipts for a contiguous period.  It never opens labels,
paths, outcomes, model scores beyond the source receipt, MC1, or any live
component.  The resulting directory is only a target-free ``base_root`` for
the existing Meta scorer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


SCHEMA = "strict_r3_p8u_target_free_shap_bridge_v1"
PROHIBITED = frozenset({
    "policy_path_valid", "policy_gross_bps", "policy_net_bps", "policy_label_available_ts",
    "policy_entry_price", "policy_exit_price", "policy_exit_reason", "policy_cost_bps",
    "path_arch_peak_mfe_atr", "supportive_path_valid", "supportive_label_available_ts",
})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _once_json(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _months(text: str) -> tuple[pd.Timestamp, ...]:
    values = tuple(pd.Timestamp(f"{item.strip()}-01", tz="UTC") for item in text.split(",") if item.strip())
    if not values or tuple(sorted(values)) != values or len(values) != len(set(values)):
        raise ValueError("--months must contain unique chronological YYYY-MM values")
    expected = tuple(pd.date_range(values[0], values[-1], freq="MS", tz="UTC"))
    if values != expected:
        raise ValueError("--months must be contiguous")
    return values


def _source_manifest(root: Path) -> dict[str, Any]:
    manifest = root / "run_manifest.json"
    payload = json.loads(manifest.read_text())
    if payload.get("schema") != "strict_r3_p8u_f72_shapderived_oof_v1":
        raise AssertionError(f"{root}: not a strict F72 SHAP receipt")
    if int(payload.get("feature_count", -1)) != 72:
        raise AssertionError(f"{root}: expected F72 contract")
    if not isinstance(payload.get("feature_contract_sha256"), str):
        raise AssertionError(f"{root}: missing F72 contract hash")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--months", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    months = _months(args.months)
    roots = tuple(root.resolve() for root in args.source_roots)
    manifests = {root: _source_manifest(root) for root in roots}
    contracts = {str(value["feature_contract_sha256"]) for value in manifests.values()}
    if len(contracts) != 1:
        raise AssertionError("source F72 contract hashes differ")
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    out.mkdir(parents=True)

    schema: Any | None = None
    receipts: list[dict[str, object]] = []
    for month in months:
        candidates = [root / "target_free_shap_features" / f"month={month:%Y-%m}.parquet" for root in roots]
        owners = [path for path in candidates if path.is_file()]
        if len(owners) != 1:
            raise AssertionError(f"{month:%Y-%m}: expected exactly one source receipt, found {len(owners)}")
        source = owners[0]
        source_schema = pq.ParquetFile(source).schema_arrow
        leaked = sorted(set(source_schema.names).intersection(PROHIBITED))
        if leaked:
            raise AssertionError(f"{source}: outcome/path fields in target-free source {leaked}")
        if schema is None:
            schema = source_schema
        elif not schema.equals(source_schema, check_metadata=False):
            raise AssertionError(f"{month:%Y-%m}: source parquet schema differs")
        target = out / source.name
        os.link(source, target)
        receipts.append({
            "month": f"{month:%Y-%m}", "source": str(source), "source_sha256": _sha256(source),
            "linked_path": str(target), "rows": int(pq.ParquetFile(source).metadata.num_rows),
            "target_free": True, "identity_schema": ["candidate_id", "__decision_ts__", "side_name"],
        })
    _once_json(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline target-free F72 SHAP source bridge only; no labels/models/MC1/admission/portfolio/live/exchange mutation",
        "months": [f"{month:%Y-%m}" for month in months],
        "source_roots": [str(root) for root in roots],
        "source_manifests": {str(root): _sha256(root / "run_manifest.json") for root in roots},
        "feature_contract_sha256": next(iter(contracts)), "source_receipts": receipts,
        "correctness": {
            "contiguous_single_owner_months": True,
            "f72_contract_hash_exact_across_sources": True,
            "parquet_schema_exact_across_sources": True,
            "target_free_sources_only": True,
            "hardlinks_preserve_source_bytes": True,
            "no_labels_models_mc1_admission_portfolio_live_or_exchange_mutation": True,
        },
    })


if __name__ == "__main__":
    main()
