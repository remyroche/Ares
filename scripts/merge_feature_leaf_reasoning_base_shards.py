#!/usr/bin/env python3
"""Merge disjoint F0--F3 transport shards without rerunning or retuning them.

The portability funnel can be resource-heavy because every arm must fit two
side-local models over a long chronological population.  Sharding is allowed
only across predeclared development transport/arm pairs.  This utility checks
that all shards reused the exact Stage-A feature audit and frozen source
contract, rejects duplicated evaluations, then performs the one permitted
development-only feature selection on their immutable union.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_feature_leaf_reasoning_portability import (  # noqa: E402
    DEVELOPMENT_TRANSPORTS,
    select_base_feature_winner,
)


REQUIRED_TABLES = (
    "base_feature_ablation_results.parquet",
    "base_feature_transport_gates.parquet",
    "base_feature_contract_coverage.parquet",
    "base_feature_rejected_arms.parquet",
)
STAGE_A_FILES = (
    "feature_portability_audit.parquet",
    "feature_portability_era_audit.parquet",
    "feature_portability_dispositions.parquet",
    "feature_portability_role_disposition_manifest.json",
    "feature_role_manifest.csv",
    "feature_role_manifest.yaml",
    "portable_feature_manifest.json",
)
ALL_ARMS = frozenset(("F0_current_frozen", "F1_portable_raw", "F2_portable_plus_atr", "F3_plus_relative"))


class BaseShardMergeError(RuntimeError):
    """A supposedly identical base-ablation shard disagreed with the contract."""


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_shard(path: Path) -> None:
    if not path.is_dir():
        raise BaseShardMergeError(f"shard is not a directory: {path}")
    for name in (*REQUIRED_TABLES, "base_feature_arm_lineage.json", "run_manifest.json", "feature_portability_dispositions.parquet"):
        if not (path / name).exists():
            raise BaseShardMergeError(f"shard lacks required artifact: {path / name}")


def _identity_pairs(results: pd.DataFrame, rejected: pd.DataFrame) -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for table in (results, rejected):
        if table.empty:
            continue
        if not {"transport", "arm"}.issubset(table.columns):
            raise BaseShardMergeError("base shard tables require transport and arm columns")
        found.update((str(row.transport), str(row.arm)) for row in table.loc[:, ["transport", "arm"]].drop_duplicates().itertuples(index=False))
    return found


def _unique_metric_rows(table: pd.DataFrame) -> None:
    if table.empty:
        return
    keys = ["transport", "arm", "scope", "period", "side_name", "top_fraction"]
    missing = [name for name in keys if name not in table]
    if missing:
        raise BaseShardMergeError(f"base metric schema missing merge keys: {missing}")
    if table.duplicated(keys, keep=False).any():
        duplicate = table.loc[table.duplicated(keys, keep=False), keys].head(4).to_dict("records")
        raise BaseShardMergeError(f"the same transport/arm metric appears in multiple shards: {duplicate}")


def merge_shards(*, destination: Path, shards: Iterable[Path]) -> dict[str, object]:
    shard_paths = [Path(path).resolve() for path in shards]
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite experiment destination: {destination}")
    if not shard_paths:
        raise BaseShardMergeError("at least one immutable shard is required")
    for path in shard_paths:
        _require_shard(path)

    first_manifest = _read_json(shard_paths[0] / "run_manifest.json")
    first_contract = first_manifest.get("source_contract")
    audit_hash = _sha256(shard_paths[0] / "feature_portability_dispositions.parquet")
    result_parts: list[pd.DataFrame] = []
    gate_parts: list[pd.DataFrame] = []
    coverage_parts: list[pd.DataFrame] = []
    rejected_parts: list[pd.DataFrame] = []
    lineage: list[object] = []
    seen_pairs: set[tuple[str, str]] = set()
    for path in shard_paths:
        manifest = _read_json(path / "run_manifest.json")
        if manifest.get("source_contract") != first_contract:
            raise BaseShardMergeError(f"frozen source contract differs in shard: {path}")
        if _sha256(path / "feature_portability_dispositions.parquet") != audit_hash:
            raise BaseShardMergeError(f"Stage-A portability audit differs in shard: {path}")
        results = pd.read_parquet(path / "base_feature_ablation_results.parquet")
        gates = pd.read_parquet(path / "base_feature_transport_gates.parquet")
        coverage = pd.read_parquet(path / "base_feature_contract_coverage.parquet")
        rejected = pd.read_parquet(path / "base_feature_rejected_arms.parquet")
        pairs = _identity_pairs(results, rejected)
        overlap = seen_pairs.intersection(pairs)
        if overlap:
            raise BaseShardMergeError(f"duplicate immutable transport/arm shard pair(s): {sorted(overlap)}")
        seen_pairs.update(pairs)
        result_parts.append(results); gate_parts.append(gates); coverage_parts.append(coverage); rejected_parts.append(rejected)
        lineage.extend(_read_json(path / "base_feature_arm_lineage.json"))

    results = pd.concat(result_parts, ignore_index=True)
    gates = pd.concat(gate_parts, ignore_index=True)
    coverage = pd.concat(coverage_parts, ignore_index=True)
    rejected = pd.concat(rejected_parts, ignore_index=True)
    _unique_metric_rows(results)
    expected_pairs = {(run.name, arm) for run in DEVELOPMENT_TRANSPORTS for arm in ALL_ARMS}
    missing = sorted(expected_pairs.difference(seen_pairs))

    destination.mkdir(parents=True)
    for name in STAGE_A_FILES:
        source = shard_paths[0] / name
        if source.exists():
            shutil.copy2(source, destination / name)
    results.to_parquet(destination / "base_feature_ablation_results.parquet", index=False, compression="zstd")
    gates.to_parquet(destination / "base_feature_transport_gates.parquet", index=False, compression="zstd")
    coverage.to_parquet(destination / "base_feature_contract_coverage.parquet", index=False, compression="zstd")
    rejected.to_parquet(destination / "base_feature_rejected_arms.parquet", index=False, compression="zstd")
    (destination / "base_feature_arm_lineage.json").write_text(json.dumps(lineage, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    if missing:
        decision = {
            "status": "BASE_FEATURE_SHARDS_INCOMPLETE_NO_SELECTION",
            "winner": None,
            "missing_transport_arm_pairs": [f"{transport}:{arm}" for transport, arm in missing],
        }
        (destination / "base_feature_selection_decision.json").write_text(
            json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    else:
        decision = select_base_feature_winner(results, destination=destination)
    manifest = {
        "schema": "feature_leaf_reasoning_portability_base_shard_merge_v1",
        "status": "BASE_FEATURE_FUNNEL_MERGED_AND_SELECTED" if not missing else "BASE_FEATURE_SHARDS_MERGED_INCOMPLETE",
        "source_contract": first_contract,
        "stage_a_disposition_sha256": audit_hash,
        "shards": [str(path) for path in shard_paths],
        "completed_transport_arm_pairs": [f"{transport}:{arm}" for transport, arm in sorted(seen_pairs)],
        "missing_transport_arm_pairs": [f"{transport}:{arm}" for transport, arm in missing],
        "base_feature_decision": decision,
    }
    (destination / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    args = parser.parse_args()
    manifest = merge_shards(destination=args.out, shards=args.shard)
    print(json.dumps({"status": manifest["status"], "decision": manifest["base_feature_decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
