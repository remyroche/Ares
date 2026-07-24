"""Fail-closed runtime contract for the frozen short-default uncertainty challenger."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_schema_hash(feature_order: Sequence[str]) -> str:
    payload = json.dumps([str(feature) for feature in feature_order], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_frozen_challenger(
    challenger_dir: Path,
    *,
    runtime_feature_order: Sequence[str],
    runtime_transform_schema: str,
    parent_model_path: Path,
    diagnostic_source_path: Path,
    neighbor_training_index_path: Path,
) -> dict[str, Any]:
    """Validate the entire frozen contract before allowing an overlay adjustment."""

    challenger_dir = Path(challenger_dir)
    manifest_path = challenger_dir / "manifest.json"
    failures: list[str] = []
    if not manifest_path.exists():
        return {"valid": False, "failures": ["missing_manifest"], "action": "revert_to_v11"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen_research_challenger_not_live":
        failures.append("unexpected_challenger_status")
    expected = manifest.get("provenance_hashes") or {}
    runtime_hashes = {
        "feature_schema_hash": feature_schema_hash(runtime_feature_order),
        "parent_model_hash": sha256_file(parent_model_path) if Path(parent_model_path).exists() else None,
        "diagnostic_source_hash": sha256_file(diagnostic_source_path) if Path(diagnostic_source_path).exists() else None,
        "neighbor_training_index_hash": sha256_file(neighbor_training_index_path)
        if Path(neighbor_training_index_path).exists() else None,
        "normalization_array_hash": sha256_file(challenger_dir / "normalization_references.npz")
        if (challenger_dir / "normalization_references.npz").exists() else None,
    }
    for name, observed in runtime_hashes.items():
        if not expected.get(name):
            failures.append(f"missing_expected_{name}")
        elif observed != expected[name]:
            failures.append(f"{name}_mismatch")
    feature_contract = manifest.get("feature_schema") or {}
    if runtime_transform_schema != feature_contract.get("transform_schema"):
        failures.append("transform_schema_mismatch")
    return {
        "valid": not failures,
        "failures": failures,
        "action": "apply_challenger" if not failures else "revert_to_v11",
        "candidate_id": manifest.get("candidate_id"),
        "candidate": manifest.get("candidate") or {},
    }


def apply_or_revert_to_v11(
    parent_rank: np.ndarray,
    uncertainty_score: np.ndarray,
    short_default_mask: np.ndarray,
    contract: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the frozen penalty only after a valid contract; otherwise return V11."""

    rank = np.asarray(parent_rank, dtype=np.float32)
    if not bool(contract.get("valid")):
        return rank.copy(), {"applied": False, "reason": "contract_invalid_reverted_to_v11"}
    score = np.asarray(uncertainty_score, dtype=np.float32)
    mask = np.asarray(short_default_mask, dtype=bool)
    result = rank.copy()
    # Values are frozen in the manifest; callers should supply them from that
    # artifact, not perform a new policy search.
    threshold = float(contract["candidate"]["threshold"])
    alpha = float(contract["candidate"]["alpha"])
    intensity = np.clip((score - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0)
    result[mask] -= np.float32(alpha) * intensity[mask]
    return np.clip(result, 0.0, 1.0), {"applied": True, "reason": "frozen_challenger"}


def load_runtime_contract(challenger_dir: Path) -> dict[str, Any]:
    """Load the immutable parameters needed by the fail-closed rank wrapper."""

    manifest = json.loads((Path(challenger_dir) / "manifest.json").read_text(encoding="utf-8"))
    return {"candidate": manifest["candidate"], "candidate_id": manifest["candidate_id"]}


__all__ = [
    "apply_or_revert_to_v11",
    "feature_schema_hash",
    "load_runtime_contract",
    "sha256_file",
    "validate_frozen_challenger",
]
