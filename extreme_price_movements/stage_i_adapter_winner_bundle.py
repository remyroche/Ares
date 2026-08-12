"""Immutable winner bundle for the explicit Stage-I target adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

from .stage_i_target_adapter import (
    StageITargetContract,
    canonical_sha256,
    file_sha256,
    training_objectives,
)


SCHEMA = "stage_i_production_target_adapter_winner_v2"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FINALIST_FAMILIES = frozenset(("R3_control", "scalar_S", "ordinal_O"))


def _validated_joint_finalist_authorization(
    raw: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Validate the handoff that binds a bundle to one authorized finalist.

    Old v2 bundles intentionally remain readable for historical diagnostics.
    New joint-finalist publications carry this immutable signed reference so a
    bundle cannot be repointed at a different population, family, or arm after
    selector completion.
    """
    if raw is None:
        return None
    value = dict(raw)
    required = {
        "schema", "target_finalist_contract_sha256", "target_finalist_contract_file_sha256",
        "family", "arm", "shared_population_contract_sha256",
        "shared_population_manifest_sha256", "shared_population_file_sha256",
        "shared_population_per_side",
    }
    if missing := required.difference(value):
        raise ValueError(f"joint finalist authorization lacks fields: {sorted(missing)}")
    if value["schema"] != "stage_i_adapter_joint_finalist_authorization_v1":
        raise ValueError("unsupported joint finalist authorization schema")
    if str(value["family"]) not in _FINALIST_FAMILIES or not str(value["arm"]).strip():
        raise ValueError("joint finalist authorization family/arm is invalid")
    for key in (
        "target_finalist_contract_sha256", "target_finalist_contract_file_sha256",
        "shared_population_contract_sha256", "shared_population_manifest_sha256",
        "shared_population_file_sha256",
    ):
        if not _SHA256.fullmatch(str(value[key])):
            raise ValueError(f"joint finalist authorization {key} must be a SHA-256")
    per_side = value["shared_population_per_side"]
    if not isinstance(per_side, Mapping) or set(per_side) != {"long", "short"}:
        raise ValueError("joint finalist authorization needs long/short universe audits")
    normalized: dict[str, dict[str, Any]] = {}
    for side in ("long", "short"):
        item = per_side[side]
        if not isinstance(item, Mapping) or int(item.get("rows", 0)) < 1:
            raise ValueError(f"joint finalist authorization {side} universe is invalid")
        candidate_hash, identity_hash = str(item.get("candidate_ids_sha256", "")), str(item.get("identity_sha256", ""))
        if not _SHA256.fullmatch(candidate_hash) or not _SHA256.fullmatch(identity_hash):
            raise ValueError(f"joint finalist authorization {side} universe checksums are invalid")
        normalized[side] = {
            "rows": int(item["rows"]), "candidate_ids_sha256": candidate_hash,
            "identity_sha256": identity_hash,
        }
    value["shared_population_per_side"] = normalized
    return value


@dataclass(frozen=True)
class StageIAdapterWinnerCell:
    side: str
    base_features: tuple[str, ...]
    meta_features: tuple[str, ...]
    base_params: Mapping[str, Any]
    meta_params: Mapping[str, Any]
    base_target_contract: StageITargetContract
    meta_target_contract: StageITargetContract
    base_selector_manifest_sha256: str
    meta_selector_manifest_sha256: str
    required_same_side_base_handoff_features: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.side not in {"long", "short"}:
            raise ValueError("winner cell side must be long/short")
        if not self.base_features or not self.meta_features:
            raise ValueError("winner cell needs exact selected feature lists")
        if self.base_target_contract.layer != "base" or self.meta_target_contract.layer != "meta":
            raise ValueError("winner target layers are explicit")
        if self.base_target_contract.geometry != self.meta_target_contract.geometry:
            raise ValueError("base/meta winner geometry drift")
        for params, contract in (
            (self.base_params, self.base_target_contract),
            (self.meta_params, self.meta_target_contract),
        ):
            expected = training_objectives(contract.family)[0]
            if str(params.get("objective", "")) != str(expected["objective"]):
                raise ValueError(f"{self.side}/{contract.family}: runtime objective drift")
            if "num_class" in expected and int(params.get("num_class", -1)) != int(expected["num_class"]):
                raise ValueError("winner num_class drift")
        if any(name not in self.meta_features for name in self.required_same_side_base_handoff_features):
            raise ValueError("meta winner dropped a direct same-side base handoff")

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side, "base_features": list(self.base_features),
            "meta_features": list(self.meta_features), "base_params": dict(self.base_params),
            "meta_params": dict(self.meta_params),
            "base_target_contract": self.base_target_contract.to_dict(),
            "meta_target_contract": self.meta_target_contract.to_dict(),
            "base_selector_manifest_sha256": self.base_selector_manifest_sha256,
            "meta_selector_manifest_sha256": self.meta_selector_manifest_sha256,
            "required_same_side_base_handoff_features": list(self.required_same_side_base_handoff_features),
        }


@dataclass(frozen=True)
class StageIAdapterWinnerBundle:
    cells: tuple[StageIAdapterWinnerCell, ...]
    code_revision: str
    run_id: str = "stage_i_target_adapter_oos_2024_2026"
    joint_finalist_authorization: Mapping[str, Any] | None = None
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SCHEMA or {cell.side for cell in self.cells} != {"long", "short"} or len(self.cells) != 2:
            raise ValueError("target-adapter winner must contain exactly long/short cells")
        if not self.code_revision or not self.run_id:
            raise ValueError("winner bundle needs immutable code/run lineage")
        _validated_joint_finalist_authorization(self.joint_finalist_authorization)

    def cell(self, side: str) -> StageIAdapterWinnerCell:
        return next(cell for cell in self.cells if cell.side == side)

    def to_dict(self) -> dict[str, Any]:
        output = {
            "schema": self.schema, "run_id": self.run_id, "code_revision": self.code_revision,
            "cells": [cell.to_dict() for cell in self.cells],
        }
        authorization = _validated_joint_finalist_authorization(self.joint_finalist_authorization)
        if authorization is not None:
            output["joint_finalist_authorization"] = authorization
        return output

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "StageIAdapterWinnerBundle":
        cells = []
        for item in raw["cells"]:
            cells.append(StageIAdapterWinnerCell(
                side=str(item["side"]), base_features=tuple(item["base_features"]),
                meta_features=tuple(item["meta_features"]),
                base_params=dict(item["base_params"]), meta_params=dict(item["meta_params"]),
                base_target_contract=StageITargetContract.from_dict(item["base_target_contract"]),
                meta_target_contract=StageITargetContract.from_dict(item["meta_target_contract"]),
                base_selector_manifest_sha256=str(item["base_selector_manifest_sha256"]),
                meta_selector_manifest_sha256=str(item["meta_selector_manifest_sha256"]),
                required_same_side_base_handoff_features=tuple(item["required_same_side_base_handoff_features"]),
            ))
        return cls(
            cells=tuple(cells), code_revision=str(raw["code_revision"]), run_id=str(raw["run_id"]),
            joint_finalist_authorization=_validated_joint_finalist_authorization(raw.get("joint_finalist_authorization")),
            schema=str(raw["schema"]),
        )


def build_stage_i_adapter_winner_bundle(
    *, base_selection_dir: str | Path, meta_selection_dir: str | Path,
    code_revision: str, run_id: str = "stage_i_target_adapter_oos_2024_2026",
    joint_finalist_authorization: Mapping[str, Any] | None = None,
) -> StageIAdapterWinnerBundle:
    def _meta_manifest_path(root: Path, side: str) -> Path:
        candidate = root / side / "manifest.json"
        if candidate.is_file():
            return candidate
        side_root = root / side
        pointer_path = side_root / "resume_complete.json"
        if not pointer_path.is_file():
            return candidate
        pointer = json.loads(pointer_path.read_text())
        if pointer.get("schema") != "stage_i_direct_fq3_resume_complete_v1" or pointer.get("side") != side:
            raise ValueError(f"{side}: direct-FQ3 resume pointer is invalid")
        relative = pointer.get("attempt_relative_path")
        if not isinstance(relative, str):
            raise ValueError(f"{side}: direct-FQ3 resume pointer lacks attempt path")
        resolved = (side_root / relative / "manifest.json").resolve()
        attempts = (side_root / "_resume_attempts").resolve()
        if attempts not in resolved.parents or resolved.parent.parent != attempts:
            raise ValueError(f"{side}: direct-FQ3 resume pointer escapes attempt root")
        if not resolved.is_file() or file_sha256(resolved) != pointer.get("attempt_manifest_sha256"):
            raise ValueError(f"{side}: direct-FQ3 resume manifest hash drift")
        return resolved

    cells = []
    for side in ("long", "short"):
        base_path = Path(base_selection_dir) / side / "manifest.json"
        meta_path = _meta_manifest_path(Path(meta_selection_dir), side)
        base, meta = json.loads(base_path.read_text()), json.loads(meta_path.read_text())
        if base.get("status") != "complete" or meta.get("status") != "complete":
            raise ValueError(f"{side}: selectors are incomplete")
        if base.get("schema") not in {"stage_i_base_feature_selection_v1", "stage_i_base_feature_selection_v2"}:
            raise ValueError(f"{side}: base selector is not a supported target-adapter selector")
        if meta.get("schema") != "stage_i_adapter_meta_feature_selection_v2":
            raise ValueError(f"{side}: meta selector is not target-adapter v2")
        base_contract = StageITargetContract.from_dict(base["target_contract"])
        meta_contract = StageITargetContract.from_dict(meta["target_contract"])
        if (
            base.get("schema") == "stage_i_base_feature_selection_v1"
            and base_contract.family != "legacy_R3_multiclass3_control"
        ):
            raise ValueError(f"{side}: selector-v1 compatibility is restricted to frozen R3")
        if base.get("target_contract_sha256") != base_contract.sha256 or meta.get("target_contract_sha256") != meta_contract.sha256:
            raise ValueError(f"{side}: selector target contract hash drift")
        if meta.get("base_target_contract", {}).get("target_sha256") != base_contract.target_sha256:
            raise ValueError(f"{side}: meta selector does not bind its base target")
        # Schema-v2 originally covered the historical mapped-bps residual
        # selector too.  A schema check alone is therefore insufficient for a
        # promotable bundle: bind the newer direct-correctness semantics and
        # reject the pre-map EV handoff explicitly.
        meta_metadata = dict(meta_contract.metadata)
        if meta_contract.family == "fold_quantile_residual3":
            if meta_metadata.get("meta_target_semantics") != "same_side_direct_base_output_correctness_q33_v1":
                raise ValueError(f"{side}: meta selector is not direct FQ3 correctness")
            if meta_metadata.get("base_input_semantics") != "same_side_direct_base_output_without_bps_conversion_v1":
                raise ValueError(f"{side}: meta selector permits a converted base input")
            selected_meta = tuple(map(str, meta["selected_feature_contract"]))
            if "prequential_base_expected_net_bps" in selected_meta:
                raise ValueError(f"{side}: direct FQ3 meta selector contains a pre-mapped expected-net feature")
        cells.append(StageIAdapterWinnerCell(
            side=side, base_features=tuple(base["selected_feature_contract"]),
            meta_features=tuple(meta["selected_feature_contract"]),
            base_params=dict(base["best_params"]), meta_params=dict(meta["best_params"]),
            base_target_contract=base_contract, meta_target_contract=meta_contract,
            base_selector_manifest_sha256=file_sha256(base_path),
            meta_selector_manifest_sha256=file_sha256(meta_path),
            required_same_side_base_handoff_features=tuple(meta["required_same_side_base_oof_handoff_features"]),
        ))
    authorization = _validated_joint_finalist_authorization(joint_finalist_authorization)
    if authorization is not None:
        expected_family = {
            "R3_control": "legacy_R3_multiclass3_control",
            "scalar_S": "soft_scalar_S",
            "ordinal_O": "cumulative_ordinal5_O",
        }[str(authorization["family"])]
        observed = {cell.base_target_contract.family for cell in cells}
        if observed != {expected_family}:
            raise ValueError(
                f"joint finalist authorization base family drift: expected {expected_family}, got {sorted(observed)}"
            )
    return StageIAdapterWinnerBundle(
        cells=tuple(cells), code_revision=code_revision, run_id=run_id,
        joint_finalist_authorization=authorization,
    )


def freeze_stage_i_adapter_winner_bundle(bundle: StageIAdapterWinnerBundle, output_path: str | Path) -> str:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(bundle.to_dict(), sort_keys=True, indent=2).encode() + b"\n"
    if destination.exists():
        if destination.read_bytes() == payload:
            return "reused_verified_immutable_bundle"
        raise FileExistsError(f"conflicting target-adapter bundle: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, destination)
        os.chmod(destination, 0o444)
    finally:
        temporary.unlink(missing_ok=True)
    return "created_immutable_bundle"


__all__ = [
    "SCHEMA", "StageIAdapterWinnerCell", "StageIAdapterWinnerBundle",
    "build_stage_i_adapter_winner_bundle", "freeze_stage_i_adapter_winner_bundle",
]
