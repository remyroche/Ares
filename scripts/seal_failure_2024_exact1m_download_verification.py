#!/usr/bin/env python3
"""Seal four read-only full-2024 exact-1m coverage verification partitions."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUEST_ROOT = (
    ROOT
    / "data_perp/artifacts/failure_2024_transition_exact1m_request_stage_20260730_v2"
)
OUTPUT = (
    ROOT / "data_perp/artifacts/failure_2024_exact1m_download_verify_20260730_v1"
)
PARTITIONS = [
    Path(f"/private/tmp/failure_2024_verify_partition_{partition}.json")
    for partition in range(4)
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(output: Path = OUTPUT) -> dict[str, object]:
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    request_manifest_path = REQUEST_ROOT / "manifest.json"
    request_manifest = json.loads(request_manifest_path.read_text())
    candidate_path = REQUEST_ROOT / "download_candidates.parquet"
    expected_candidate_hash = request_manifest["outputs"]["download_candidates"][
        "sha256"
    ]
    if sha256(candidate_path) != expected_candidate_hash:
        raise ValueError("frozen request candidate hash mismatch")

    payloads: list[dict[str, object]] = []
    symbols: set[str] = set()
    for expected_partition, path in enumerate(PARTITIONS):
        payload = json.loads(path.read_text())
        if (
            payload.get("generated_by") != "download_policy_execution_1m"
            or payload.get("candidate_sha256") != expected_candidate_hash
            or payload.get("partition_count") != 4
            or payload.get("partition_id") != expected_partition
            or payload.get("verify_only") is not True
            or payload.get("horizon_minutes") != 720
        ):
            raise ValueError(f"partition {expected_partition} contract mismatch")
        stage = payload.get("stage_manifest") or {}
        if stage.get("sha256") != sha256(request_manifest_path):
            raise ValueError(f"partition {expected_partition} stage binding mismatch")
        summary = payload["summary"]
        if (
            summary["required_minutes"] != summary["covered_minutes"]
            or summary["incomplete_symbols"] != 0
            or summary["failed_symbols"] != 0
            or summary["ok_symbols"] != payload["symbols"]
        ):
            raise ValueError(f"partition {expected_partition} is not complete")
        local_symbols = {str(row["symbol"]) for row in payload["results"]}
        if len(local_symbols) != payload["symbols"] or symbols.intersection(local_symbols):
            raise ValueError("partition symbol ownership overlaps or duplicates")
        if any(
            row["status"] != "ok"
            or row["required_minutes"] != row["covered_after"]
            or row["coverage_after"] != 1.0
            for row in payload["results"]
        ):
            raise ValueError(f"partition {expected_partition} has an incomplete row")
        symbols.update(local_symbols)
        payloads.append(payload)

    required = sum(int(payload["summary"]["required_minutes"]) for payload in payloads)
    covered = sum(int(payload["summary"]["covered_minutes"]) for payload in payloads)
    if len(symbols) != int(request_manifest["distinct_symbols"]) or required != covered:
        raise ValueError("aggregate verification does not cover the frozen request")

    output.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(
        tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.staging-")
    )
    try:
        partition_outputs: dict[str, dict[str, object]] = {}
        for partition, source in enumerate(PARTITIONS):
            destination = stage_dir / f"partition_{partition}.json"
            shutil.copy2(source, destination)
            partition_outputs[str(partition)] = {
                "path": destination.name,
                "sha256": sha256(destination),
                "symbols": int(payloads[partition]["summary"]["ok_symbols"]),
                "required_minutes": int(
                    payloads[partition]["summary"]["required_minutes"]
                ),
                "covered_minutes": int(
                    payloads[partition]["summary"]["covered_minutes"]
                ),
            }
        manifest = {
            "schema": "failure_2024_exact1m_download_verification_v1",
            "status": "SEALED_COMPLETE",
            "request_manifest": {
                "path": str(request_manifest_path),
                "sha256": sha256(request_manifest_path),
            },
            "candidate_request": {
                "path": str(candidate_path),
                "sha256": expected_candidate_hash,
                "rows": int(
                    request_manifest["outputs"]["download_candidates"]["rows"]
                ),
            },
            "partition_count": 4,
            "symbols": len(symbols),
            "required_minutes": required,
            "covered_minutes": covered,
            "coverage_fraction": covered / required,
            "incomplete_symbols": 0,
            "failed_symbols": 0,
            "verification_only": True,
            "immutable_store_contract": (
                "canonical_kraken_execution_1m_immutable_read_only_v1"
            ),
            "partitions": partition_outputs,
            "unlocks": [
                "full_2024_candidate_local_policy_label_replay",
                "full_2024_physical_path_and_multitask_label_replay",
                "2022_2026_calendar_and_regime_report_regeneration",
            ],
        }
        manifest_path = stage_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (stage_dir / "manifest.sha256").write_text(
            f"{sha256(manifest_path)}  manifest.json\n"
        )
        os.replace(stage_dir, output)
        return manifest
    except Exception:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
