#!/usr/bin/env python3
"""Deterministically seal and verify the accepted Stage-C Stage-0 artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT / "data_perp/artifacts/stage_c_continuation_feature_panel_20260731_v2"
MANIFEST_NAME = "run_manifest.json"
CODE_PATHS = {
    "continuation_features.py": ROOT / "extreme_price_movements/continuation_features.py",
    "materializer": ROOT / "scripts/materialize_stage_c_continuation_feature_panel.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_outputs(artifact: Path) -> dict[str, Path]:
    return {
        path.name: path
        for path in sorted(artifact.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.name != MANIFEST_NAME
    }


def compute_output_hashes(artifact: Path) -> dict[str, str]:
    """Hash every regular artifact output except the self-referential manifest."""
    return {name: sha256(path) for name, path in _regular_outputs(artifact).items()}


def verify_output_hashes(artifact: Path, expected: Mapping[str, str]) -> dict[str, Any]:
    """Fail closed unless both the output file set and every byte hash match."""
    actual_paths = _regular_outputs(artifact)
    expected_names = set(expected)
    actual_names = set(actual_paths)
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    mismatched = sorted(
        name
        for name in expected_names & actual_names
        if sha256(actual_paths[name]) != expected[name]
    )
    if missing or unexpected or mismatched:
        raise ValueError(
            "Stage-0 output verification failed closed: "
            f"missing={missing}, unexpected={unexpected}, mismatched={mismatched}"
        )
    return {
        "verified": True,
        "output_count": len(expected),
        "correctness_report_included": "correctness_test_report.json" in expected,
        "manifest_excluded": MANIFEST_NAME not in expected,
    }


def _verify_named_hashes(
    namespace: str, expected: Mapping[str, str], paths: Mapping[str, Path]
) -> int:
    missing_keys = sorted(set(expected) - set(paths))
    unexpected_keys = sorted(set(paths) - set(expected))
    missing_files = sorted(key for key, path in paths.items() if not path.is_file())
    mismatched = sorted(
        key
        for key, path in paths.items()
        if path.is_file() and expected.get(key) != sha256(path)
    )
    if missing_keys or unexpected_keys or missing_files or mismatched:
        raise ValueError(
            f"Stage-0 {namespace} verification failed closed: "
            f"missing_keys={missing_keys}, unexpected_keys={unexpected_keys}, "
            f"missing_files={missing_files}, mismatched={mismatched}"
        )
    return len(expected)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, sort_keys=True, indent=2, default=str) + "\n"
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def seal(artifact: Path) -> dict[str, Any]:
    manifest_path = artifact / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    inputs = {str(path): Path(path) for path in manifest["inputs"]}
    sources = {str(path): Path(path) for path in manifest["source_file_sha256"]}
    verified_inputs = _verify_named_hashes("inputs", manifest["inputs"], inputs)
    verified_sources = _verify_named_hashes(
        "source files", manifest["source_file_sha256"], sources
    )
    verified_code = _verify_named_hashes("code", manifest["code_sha256"], CODE_PATHS)

    outputs = compute_output_hashes(artifact)
    manifest["outputs"] = outputs
    manifest["seal"] = {
        "algorithm": "sha256",
        "code_hashes_verified": verified_code,
        "correctness_report_included": "correctness_test_report.json" in outputs,
        "input_hashes_verified": verified_inputs,
        "manifest_excluded": MANIFEST_NAME,
        "output_count": len(outputs),
        "schema": "stage_c_stage0_output_seal_v1",
        "source_hashes_verified": verified_sources,
    }
    _atomic_write_json(manifest_path, manifest)

    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = verify_output_hashes(artifact, persisted["outputs"])
    result.update(
        {
            "artifact": str(artifact),
            "input_hashes_verified": verified_inputs,
            "source_hashes_verified": verified_sources,
            "code_hashes_verified": verified_code,
            "manifest_sha256": sha256(manifest_path),
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()
    print(json.dumps(seal(args.artifact.resolve()), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
