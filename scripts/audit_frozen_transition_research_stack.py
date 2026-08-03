#!/usr/bin/env python3
"""Verify the frozen transition/context stack and publish a fail-closed audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / (
    "configs/frozen_transition_research_stack_20260729_v1.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    registry = json.loads(Path(path).read_text(encoding="utf-8"))
    if registry.get("status") != "FROZEN_CONTEXT_ONLY_NO_DIRECT_POLICY_CONTROL":
        raise ValueError("transition registry is not in the frozen context-only state")
    return registry


def validate_frozen_source(
    registry: dict[str, Any],
    key: str,
    supplied_path: Path,
    root: Path = ROOT,
    require_model_input_eligible: bool = False,
) -> Path:
    if key not in registry["sources"]:
        raise KeyError(f"unknown frozen transition source: {key}")
    specification = registry["sources"][key]
    expected = (root / specification["path"]).resolve()
    actual = Path(supplied_path).resolve()
    if actual != expected:
        raise ValueError(f"{key} path is not the frozen registry path")
    if require_model_input_eligible and not specification.get(
        "model_input_eligible", False
    ):
        raise ValueError(f"{key} is frozen descriptive evidence, not a model input")
    if not actual.exists():
        raise FileNotFoundError(actual)
    if sha256(actual) != specification["sha256"]:
        raise ValueError(f"{key} hash differs from the frozen registry")
    return actual


def _source_columns(path: Path) -> list[str]:
    if path.suffix == ".parquet":
        return pq.ParquetFile(path).schema_arrow.names
    if path.suffix == ".csv":
        return pd.read_csv(path, nrows=0).columns.tolist()
    return []


def audit(registry_path: Path, output_dir: Path) -> dict[str, Any]:
    registry = load_registry(registry_path)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    rows = []
    for key, specification in registry["sources"].items():
        path = validate_frozen_source(registry, key, ROOT / specification["path"])
        columns = _source_columns(path)
        declared_fields = specification.get(
            "fields", specification.get("descriptive_packet_fields", [])
        )
        missing = sorted(set(declared_fields).difference(columns))
        if missing:
            raise ValueError(f"{key} misses declared frozen fields: {missing}")
        forbidden_declared = [
            field
            for field in declared_fields
            if field.startswith("target__")
            or field.startswith("fold__")
            or "outcome" in field.lower()
            or "realized" in field.lower()
        ]
        if forbidden_declared:
            raise ValueError(f"{key} declares outcome fields: {forbidden_declared}")
        rows.append(
            {
                "source": key,
                "path": str(path),
                "sha256": specification["sha256"],
                "model_input_eligible": bool(
                    specification.get("model_input_eligible", False)
                ),
                "declared_field_count": len(declared_fields),
                "hash_verified": True,
                "field_contract_verified": True,
            }
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    audit_path = output_dir / "frozen_source_audit.parquet"
    pd.DataFrame(rows).to_parquet(audit_path, index=False, compression="zstd")
    report = {
        "schema": "frozen_transition_research_stack_audit_v1",
        "status": "FROZEN_STACK_VERIFIED",
        "registry": {
            "path": str(registry_path.resolve()),
            "sha256": sha256(registry_path),
        },
        "source_count": len(rows),
        "all_hashes_verified": True,
        "all_field_contracts_verified": True,
        "consumer_contract": registry["consumer_contract"],
        "paused_workstreams": registry["paused_workstreams"],
        "output": {
            "path": str(audit_path.resolve()),
            "sha256": sha256(audit_path),
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
    }
    manifest = output_dir / "manifest.json"
    manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (output_dir / "manifest.sha256").write_text(sha256(manifest) + "\n")
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    print(
        json.dumps(
            audit(args.registry, args.output_dir), indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
