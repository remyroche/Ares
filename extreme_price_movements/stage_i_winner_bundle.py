"""Fail-closed freezer for the four completed Stage-I selector cells.

This module performs no model fitting, feature materialisation, or OOS
generation.  It converts already-complete side-local base/meta selector
artifacts and the immutable production-input contract into the sole winner
bundle accepted by :mod:`stage_i_production_oos`.
"""

from __future__ import annotations

import json
import os
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .stage_i_feature_selection import (
    STAGE_I_ACTIVE_CONTRACTS,
    STAGE_I_META_BASE_OOF_HANDOFF_FEATURES,
    StageIHeadContract,
)
from .stage_i_production_oos import (
    StageIFeatureSelectionReuseException,
    StageIOOSCalendar,
    StageIProductionOOSError,
    StageIProductionWinnerBundle,
    StageIWinnerCell,
)


SOURCE_BINDING_SCHEMA = "stage_i_production_input_source_binding_v1"
CALENDAR_START_UTC = "2024-01-01T00:00:00Z"
CALENDAR_END_DAY = pd.Timestamp("2026-07-10T00:00:00Z")


class StageIWinnerBundleFreezeError(StageIProductionOOSError):
    """Raised when completed selection artifacts cannot be frozen exactly."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: Any) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise StageIWinnerBundleFreezeError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StageIWinnerBundleFreezeError(f"{label} is not readable complete JSON: {path}") from exc
    if not isinstance(value, dict):
        raise StageIWinnerBundleFreezeError(f"{label} must contain one JSON object: {path}")
    return value


def _exact_exposed_values(
    manifest: Mapping[str, Any], keys: Sequence[str], *, label: str
) -> tuple[str, ...]:
    exposed: list[tuple[str, ...]] = []
    for key in keys:
        if key not in manifest:
            continue
        raw = manifest[key]
        if not isinstance(raw, (list, tuple)):
            raise StageIWinnerBundleFreezeError(f"{label} {key} must be an ordered JSON list")
        values = tuple(str(value) for value in raw)
        if not values or any(not value.strip() for value in values) or len(set(values)) != len(values):
            raise StageIWinnerBundleFreezeError(
                f"{label} {key} must be non-empty, ordered, unique feature names"
            )
        exposed.append(values)
    if not exposed:
        raise StageIWinnerBundleFreezeError(f"{label} has no frozen selected-feature contract")
    if any(value != exposed[0] for value in exposed[1:]):
        raise StageIWinnerBundleFreezeError(f"{label} selected-feature fields disagree")
    return exposed[0]


def _normalise_runtime_params(
    manifest: Mapping[str, Any], *, contract: StageIHeadContract
) -> dict[str, Any]:
    exposed: list[dict[str, Any]] = []
    for key in ("best_params", "lgbm_params", "frozen_lgbm_params", "params"):
        if key not in manifest:
            continue
        raw = manifest[key]
        if not isinstance(raw, Mapping) or not raw:
            raise StageIWinnerBundleFreezeError(f"{contract.artifact_key} {key} must be non-empty parameters")
        exposed.append(dict(raw))
    if not exposed:
        raise StageIWinnerBundleFreezeError(f"{contract.artifact_key} has no frozen LGBM parameters")

    expected_objective = "multiclass" if contract.layer == "base" else "huber"
    normalised: list[dict[str, Any]] = []
    for params in exposed:
        if "objective" in params and str(params["objective"]).lower() != expected_objective:
            raise StageIWinnerBundleFreezeError(
                f"{contract.artifact_key} selector objective disagrees with {expected_objective} runtime semantics"
            )
        params["objective"] = expected_objective
        if contract.layer == "base":
            if "num_class" in params and int(params["num_class"]) != 3:
                raise StageIWinnerBundleFreezeError(
                    f"{contract.artifact_key} selector num_class disagrees with the three-state R3 target"
                )
            params["num_class"] = 3
        normalised.append(params)
    if any(_canonical_bytes(value) != _canonical_bytes(normalised[0]) for value in normalised[1:]):
        raise StageIWinnerBundleFreezeError(f"{contract.artifact_key} parameter fields disagree")
    return normalised[0]


def load_stage_i_production_source_binding(input_contract_dir: str | Path) -> dict[str, Any]:
    """Load and hash-bind both files that define the production input contract."""
    root = Path(input_contract_dir)
    manifest_path = root / "manifest.json"
    feature_path = root / "frozen_feature_contract.json"
    manifest = _load_json(manifest_path, label="Stage-I production input manifest")
    features = _load_json(feature_path, label="Stage-I frozen feature contract")
    if manifest.get("schema") != "stage_i_production_input_contract_v1" or manifest.get("status") != "complete":
        raise StageIWinnerBundleFreezeError("Stage-I production input contract is not complete or has the wrong schema")
    if not isinstance(features.get("feature_columns"), list) or not features["feature_columns"]:
        raise StageIWinnerBundleFreezeError("frozen feature contract has no ordered feature_columns")
    if manifest.get("feature_contract_sha256") != features.get("feature_contract_sha256"):
        raise StageIWinnerBundleFreezeError("input manifest and frozen feature contract hashes disagree")

    start = pd.to_datetime(manifest.get("min_signal_ts"), utc=True, errors="coerce")
    end = pd.to_datetime(manifest.get("max_signal_ts"), utc=True, errors="coerce")
    if pd.isna(start) or pd.isna(end) or start >= pd.Timestamp(CALENDAR_START_UTC):
        raise StageIWinnerBundleFreezeError("production source must contain pre-2024 fitting history")
    if pd.Timestamp(end).normalize() != CALENDAR_END_DAY:
        raise StageIWinnerBundleFreezeError("production source must end on the frozen 2026-07-10 calendar day")

    return {
        "schema": SOURCE_BINDING_SCHEMA,
        "production_input_manifest": manifest,
        "production_input_manifest_sha256": _digest(manifest),
        "frozen_feature_contract": features,
        "frozen_feature_contract_sha256": _digest(features),
    }


def _selector_manifest_path(root: Path, contract: StageIHeadContract) -> Path:
    # Base and meta roots are intentionally separate.  Each has exactly one
    # directory per side, matching the checkpointed selector CLIs.
    return root / contract.side / "manifest.json"


def _build_cell(
    contract: StageIHeadContract,
    *,
    selector_root: Path,
    source_binding: Mapping[str, Any],
) -> StageIWinnerCell:
    path = _selector_manifest_path(selector_root, contract)
    manifest = _load_json(path, label=f"{contract.artifact_key} selector manifest")
    expected_schema = f"stage_i_{contract.layer}_feature_selection_v1"
    if manifest.get("schema") != expected_schema or manifest.get("status") != "complete":
        raise StageIWinnerBundleFreezeError(f"{contract.artifact_key} selector cell is partial or has the wrong schema")
    if str(manifest.get("side", "")).lower() != contract.side:
        raise StageIWinnerBundleFreezeError(f"{contract.artifact_key} selector side disagrees with its directory")
    if int(manifest.get("rows", 0)) <= 0:
        raise StageIWinnerBundleFreezeError(f"{contract.artifact_key} selector has no completed rows")

    features = _exact_exposed_values(
        manifest,
        ("selected_features", "selected_feature_names", "stage_i_selected_feature_contract", "selected_feature_contract"),
        label=contract.artifact_key,
    )
    if "selected_feature_count" in manifest and int(manifest["selected_feature_count"]) != len(features):
        raise StageIWinnerBundleFreezeError(f"{contract.artifact_key} selected_feature_count disagrees")
    params = _normalise_runtime_params(manifest, contract=contract)

    raw_contract = tuple(str(value) for value in source_binding["frozen_feature_contract"]["feature_columns"])
    raw_available = set(raw_contract)
    permitted_generated = set(STAGE_I_META_BASE_OOF_HANDOFF_FEATURES) if contract.layer == "meta" else set()
    missing = [feature for feature in features if feature not in raw_available and feature not in permitted_generated]
    if missing:
        raise StageIWinnerBundleFreezeError(
            f"{contract.artifact_key} selected fields are absent from its frozen production source: {missing[:12]}"
        )
    if contract.layer == "meta":
        handoffs = tuple(STAGE_I_META_BASE_OOF_HANDOFF_FEATURES)
        if any(feature not in features for feature in handoffs):
            raise StageIWinnerBundleFreezeError(f"{contract.artifact_key} lacks the exact same-side base OOF handoff")
        declared = manifest.get("required_same_side_base_oof_handoff_features")
        if declared is not None and tuple(map(str, declared)) != handoffs:
            raise StageIWinnerBundleFreezeError(f"{contract.artifact_key} handoff declaration disagrees")

    # Preserve the selector file exactly as completed, but add only the fixed
    # runtime fields when the HPO artifact omitted them. StageIWinnerCell then
    # independently checks the resulting exact agreement.
    frozen_selector = dict(manifest)
    frozen_selector["best_params"] = params
    frozen_selector["selected_feature_contract"] = list(features)
    return StageIWinnerCell(
        contract=contract,
        selected_feature_names=features,
        lgbm_params=params,
        selector_manifest=frozen_selector,
        selector_manifest_sha256=_digest(frozen_selector),
        source_manifest=dict(source_binding),
        source_manifest_sha256=_digest(source_binding),
    )


def build_stage_i_winner_bundle(
    *,
    base_selection_dir: str | Path,
    meta_selection_dir: str | Path,
    input_contract_dir: str | Path,
    code_revision: str,
    run_id: str = "stage_i_production_oos_2024_2026",
) -> StageIProductionWinnerBundle:
    """Build the exact four-cell winner bundle without writing or fitting."""
    source = load_stage_i_production_source_binding(input_contract_dir)
    roots = {"base": Path(base_selection_dir), "meta": Path(meta_selection_dir)}
    cells = tuple(
        _build_cell(contract, selector_root=roots[contract.layer], source_binding=source)
        for contract in STAGE_I_ACTIVE_CONTRACTS
    )
    source_end = source["production_input_manifest"]["max_signal_ts"]
    return StageIProductionWinnerBundle(
        cells=cells,
        code_revision=str(code_revision),
        calendar=StageIOOSCalendar(CALENDAR_START_UTC, str(source_end)),
        feature_selection_exception=StageIFeatureSelectionReuseException(
            approved=True,
            selection_reference_start_utc=str(source["production_input_manifest"]["min_signal_ts"]),
            selection_reference_end_utc=str(source_end),
            rationale=(
                "User-approved exception: select features once on the full available production reference "
                "population, with the frozen per-side/per-layer list reused backward across the 2024-2026 "
                "strict chronological OOF calendar. This is not claimed as fold-local feature selection."
            ),
        ),
        run_id=run_id,
    )


def freeze_stage_i_winner_bundle(
    *,
    base_selection_dir: str | Path,
    meta_selection_dir: str | Path,
    input_contract_dir: str | Path,
    output_path: str | Path,
    code_revision: str,
    run_id: str = "stage_i_production_oos_2024_2026",
) -> tuple[StageIProductionWinnerBundle, str]:
    """Atomically publish an immutable JSON bundle or reuse an exact match."""
    bundle = build_stage_i_winner_bundle(
        base_selection_dir=base_selection_dir,
        meta_selection_dir=meta_selection_dir,
        input_contract_dir=input_contract_dir,
        code_revision=code_revision,
        run_id=run_id,
    )
    # Detect source mutation between cell construction and publication.  This
    # also makes the intended source-hash check explicit at the write boundary.
    current_source = load_stage_i_production_source_binding(input_contract_dir)
    if any(cell.source_manifest_sha256 != _digest(current_source) for cell in bundle.cells):
        raise StageIWinnerBundleFreezeError("production input contract changed while freezing the winner bundle")
    payload = _canonical_bytes(bundle.to_dict()) + b"\n"
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        if destination.is_file() and destination.read_bytes() == payload:
            return bundle, "reused_verified_immutable_bundle"
        raise FileExistsError(f"refusing to overwrite conflicting immutable Stage-I winner bundle: {destination}")

    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # Atomic no-replace publication. A racing writer can never be
            # overwritten; an exact racing artifact is merely reused.
            os.link(temporary, destination)
        except FileExistsError:
            if not destination.is_file() or destination.read_bytes() != payload:
                raise FileExistsError(
                    f"refusing to overwrite conflicting immutable Stage-I winner bundle: {destination}"
                )
            return bundle, "reused_verified_immutable_bundle"
        os.chmod(destination, 0o444)
        return bundle, "created_immutable_bundle"
    finally:
        temporary.unlink(missing_ok=True)
