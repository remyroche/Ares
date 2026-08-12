"""Canonical common-row contract for joint Stage-I finalist comparisons.

Different target geometries can legitimately have different label-valid
populations.  A final R3/S/O comparison must not silently exploit those
different denominators.  This module freezes the side-qualified intersection
of all declared finalist target populations *before* final OOS fitting and
causal mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd


SCHEMA = "stage_i_joint_finalist_shared_population_v3"
LEGACY_SCHEMA = "stage_i_joint_finalist_shared_population_v1"
V2_SCHEMA = "stage_i_joint_finalist_shared_population_v2"
SUPPORTED_SCHEMAS = frozenset((LEGACY_SCHEMA, V2_SCHEMA, SCHEMA))
# Candidate IDs are side-qualified but symbol remains part of the immutable
# source identity used by the final feature/contract joins.  Omitting it made
# the v1/v2 shared universe insufficient for a no-fit Stage-I OOS preflight.
IDENTITY = (
    "candidate_id", "side_name", "__symbol__", "__ts__", "decision_ts", "label_available_ts",
)


class SharedPopulationError(ValueError):
    """Raised when finalist target populations cannot be compared honestly."""


def file_sha256(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _identity_records(frame: pd.DataFrame) -> list[dict[str, str]]:
    """Return the canonical side-qualified candidate/time identity surface."""
    columns = [*IDENTITY, "candidate_key"]
    ordered = frame.loc[:, columns].copy().sort_values(
        ["side_name", "decision_ts", "candidate_key"], kind="stable",
    )
    return ordered.astype(str).to_dict(orient="records")


def _per_side_audit(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    audit: dict[str, dict[str, Any]] = {}
    for side in ("long", "short"):
        part = frame.loc[frame.side_name.eq(side)].copy()
        if part.empty:
            raise SharedPopulationError(f"common evaluation universe has no {side} candidates")
        audit[side] = {
            "rows": int(len(part)),
            "candidate_ids_sha256": canonical_sha256(
                sorted(part.candidate_id.astype(str).tolist())
            ),
            "identity_sha256": canonical_sha256(_identity_records(part)),
        }
    return audit


@dataclass(frozen=True)
class SharedPopulationSpec:
    r3_base_selection_dir: Path
    scalar_winner_dir: Path
    ordinal_winner_dir: Path
    output_dir: Path


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise SharedPopulationError(f"expected JSON object: {path}")
    return value


def _normalise(frame: pd.DataFrame, *, name: str, r3: bool) -> pd.DataFrame:
    required = set(IDENTITY) | ({"exact_net_bps"} if r3 else {"target_valid"})
    if missing := required.difference(frame.columns):
        raise SharedPopulationError(f"{name}: missing population fields: {sorted(missing)}")
    output = frame.loc[:, list(IDENTITY)].copy()
    output["candidate_id"] = output.candidate_id.astype(str)
    output["side_name"] = output.side_name.astype(str).str.lower()
    output["__symbol__"] = output["__symbol__"].astype(str)
    output["__ts__"] = pd.to_datetime(output["__ts__"], utc=True, errors="coerce")
    output["decision_ts"] = pd.to_datetime(output.decision_ts, utc=True, errors="coerce")
    output["label_available_ts"] = pd.to_datetime(output.label_available_ts, utc=True, errors="coerce")
    if output.loc[:, ["__ts__", "decision_ts", "label_available_ts"]].isna().any().any():
        raise SharedPopulationError(f"{name}: non-UTC/invalid timing")
    if not output.side_name.isin(("long", "short")).all():
        raise SharedPopulationError(f"{name}: invalid side")
    if output["__symbol__"].str.strip().eq("").any() or output["__symbol__"].eq("nan").any():
        raise SharedPopulationError(f"{name}: invalid symbol identity")
    if r3:
        output["target_valid"] = np.isfinite(pd.to_numeric(frame.exact_net_bps, errors="coerce"))
    else:
        output["target_valid"] = frame.target_valid.astype(bool)
    output["candidate_key"] = output.side_name + "::" + output.candidate_id
    if output.candidate_key.duplicated().any():
        raise SharedPopulationError(f"{name}: duplicate qualified candidate identity")
    return output


def _winner_population(root: Path, *, name: str) -> tuple[pd.DataFrame, dict[str, str]]:
    manifest_path, handoff_path = root / "manifest.json", root / "winner_target_handoff.parquet"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "complete":
        raise SharedPopulationError(f"{name}: incomplete winner bundle")
    observed = file_sha256(handoff_path)
    if str((manifest.get("artifact_sha256") or {}).get(handoff_path.name, "")) != observed:
        raise SharedPopulationError(f"{name}: handoff checksum drift")
    return _normalise(pd.read_parquet(handoff_path), name=name, r3=False), {
        f"{name}_manifest": file_sha256(manifest_path), f"{name}_handoff": observed,
    }


def _r3_population(root: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    frames = []
    hashes: dict[str, str] = {}
    for side in ("long", "short"):
        manifest_path, oof_path = root / side / "manifest.json", root / side / "selector_base_oof.parquet"
        manifest = _read_json(manifest_path)
        observed = file_sha256(oof_path)
        if manifest.get("status") != "complete" or str(manifest.get("selector_base_oof_sha256", "")) != observed:
            raise SharedPopulationError(f"R3/{side}: frozen base OOF checksum drift")
        frames.append(_normalise(pd.read_parquet(oof_path), name=f"R3/{side}", r3=True))
        hashes[f"r3_{side}_manifest"] = file_sha256(manifest_path)
        hashes[f"r3_{side}_oof"] = observed
    return pd.concat(frames, ignore_index=True), hashes


def validate_shared_population(
    root: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and verify a signed common finalist evaluation universe.

    The validator is deliberately usable by both the materializer and any
    no-fit preflight.  It proves that the persisted candidate list is the
    exact side-qualified R3 ∩ scalar ∩ ordinal valid universe, rather than
    merely trusting a directory name or a single manifest hash.
    """
    root = Path(root)
    manifest_path, population_path = root / "manifest.json", root / "shared_population.parquet"
    manifest = _read_json(manifest_path)
    if manifest.get("schema") not in SUPPORTED_SCHEMAS or manifest.get("status") != "complete":
        raise SharedPopulationError("shared population is incomplete or has an unsupported schema")
    expected_contract = str(manifest.get("contract_sha256", ""))
    if expected_contract != canonical_sha256({key: value for key, value in manifest.items() if key != "contract_sha256"}):
        raise SharedPopulationError("shared population contract checksum drift")
    if not population_path.is_file() or str((manifest.get("files") or {}).get(population_path.name, "")) != file_sha256(population_path):
        raise SharedPopulationError("shared population parquet checksum drift")
    frame = pd.read_parquet(population_path)
    required = {*IDENTITY, "candidate_key"}
    if missing := required.difference(frame.columns):
        raise SharedPopulationError(f"shared population lacks fields: {sorted(missing)}")
    normal = _normalise(frame.assign(target_valid=True), name="shared_population", r3=False)
    supplied_key = frame["candidate_key"].astype(str).reset_index(drop=True)
    if not supplied_key.equals(normal["candidate_key"].astype(str).reset_index(drop=True)):
        raise SharedPopulationError("shared population candidate_key drift")
    if len(normal) != len(frame):
        raise SharedPopulationError("shared population normalization changed rows")
    normal = normal.loc[:, [*IDENTITY, "candidate_key"]]
    if int(manifest.get("rows", -1)) != len(normal):
        raise SharedPopulationError("shared population row count drift")
    if str(manifest.get("population_sha256", "")) != canonical_sha256(_identity_records(normal)):
        raise SharedPopulationError("shared population identity checksum drift")
    if manifest.get("schema") == SCHEMA:
        expected_per_side = _per_side_audit(normal)
        if dict(manifest.get("per_side", {})) != expected_per_side:
            raise SharedPopulationError("shared population per-side identity audit drift")
        if tuple(manifest.get("finalist_families", ())) != ("R3_control", "scalar_S", "ordinal_O"):
            raise SharedPopulationError("shared population finalist family contract drift")
    return normal, manifest


def shared_population_contract_reference(root: str | Path) -> dict[str, Any]:
    """Return the immutable reference embedded in the joint-finalist contract."""
    root = Path(root).resolve()
    _frame, manifest = validate_shared_population(root)
    manifest_path, population_path = root / "manifest.json", root / "shared_population.parquet"
    return {
        "schema": "stage_i_joint_finalist_shared_population_reference_v1",
        "path": str(root),
        "manifest_sha256": file_sha256(manifest_path),
        "population_file_sha256": file_sha256(population_path),
        "contract_sha256": str(manifest["contract_sha256"]),
        "population_sha256": str(manifest["population_sha256"]),
        "rows": int(manifest["rows"]),
        "per_side": dict(manifest.get("per_side", {})),
    }


def materialize_shared_population(spec: SharedPopulationSpec) -> Mapping[str, Any]:
    """Atomically write the valid common R3/scalar/ordinal population."""
    if spec.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite shared population: {spec.output_dir}")
    r3, hashes = _r3_population(spec.r3_base_selection_dir)
    scalar, scalar_hashes = _winner_population(spec.scalar_winner_dir, name="scalar_S")
    ordinal, ordinal_hashes = _winner_population(spec.ordinal_winner_dir, name="ordinal_O")
    hashes.update(scalar_hashes)
    hashes.update(ordinal_hashes)
    populations = {"R3": r3, "scalar_S": scalar, "ordinal_O": ordinal}
    valid_keys = {
        name: set(frame.loc[frame.target_valid.astype(bool), "candidate_key"])
        for name, frame in populations.items()
    }
    common = set.intersection(*valid_keys.values())
    if not common:
        raise SharedPopulationError("finalist target populations have no common valid rows")
    canonical = r3.loc[r3.candidate_key.isin(common), list(IDENTITY) + ["candidate_key"]].copy()
    canonical = canonical.sort_values(["decision_ts", "side_name", "candidate_key"], kind="stable").reset_index(drop=True)
    for name, frame in populations.items():
        candidate = frame.loc[frame.candidate_key.isin(common), list(IDENTITY) + ["candidate_key"]]
        candidate = candidate.sort_values("candidate_key", kind="stable").reset_index(drop=True)
        reference = canonical.sort_values("candidate_key", kind="stable").reset_index(drop=True)
        if not reference.equals(candidate):
            raise SharedPopulationError(f"{name}: common candidates have timing/identity drift")
    counts = {
        name: {
            "source_rows": int(len(frame)), "valid_rows": int(frame.target_valid.sum()),
            "common_valid_rows": int(len(common)), "excluded_valid_rows": int(len(valid_keys[name] - common)),
        }
        for name, frame in populations.items()
    }
    spec.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{spec.output_dir.name}.", dir=spec.output_dir.parent))
    try:
        population_path = temporary / "shared_population.parquet"
        canonical.to_parquet(population_path, index=False, compression="zstd")
        per_side = _per_side_audit(canonical)
        manifest = {
            "schema": SCHEMA, "status": "complete", "rows": int(len(canonical)),
            "identity_columns": list(IDENTITY), "population_sha256": canonical_sha256(_identity_records(canonical)),
            "source_lineage_sha256": hashes, "counts": counts,
            "per_side": per_side,
            "finalist_families": ["R3_control", "scalar_S", "ordinal_O"],
            "selection": "per-side side-qualified intersection of target_valid candidate IDs across R3, scalar_S and ordinal_O before final OOS fitting/mapping",
            "files": {population_path.name: file_sha256(population_path)},
        }
        manifest["contract_sha256"] = canonical_sha256({key: value for key, value in manifest.items() if key != "contract_sha256"})
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, spec.output_dir)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


__all__ = [
    "SCHEMA", "LEGACY_SCHEMA", "SUPPORTED_SCHEMAS", "SharedPopulationError",
    "SharedPopulationSpec", "materialize_shared_population", "validate_shared_population",
    "shared_population_contract_reference",
]
