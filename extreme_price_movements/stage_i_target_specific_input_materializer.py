"""Materialise immutable inputs for the Stage-I direct-FQ3 OOS evaluator.

The selector feature matrix is a causal source, while target winner handoffs
are the authoritative label/economic source.  This module joins them only by
the exact candidate identity and emits the deliberately narrow per-side input
layout consumed by :mod:`stage_i_target_specific_oos`.

Training target contracts remain bound to the rows used by selector/HPO.  New
evaluation contracts bind the materialised rows and are required to have the
same target semantics.  This distinction is essential: an OOS population
cannot legitimately reproduce a training population's content hashes.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .stage_i_adapter_winner_bundle import StageIAdapterWinnerBundle
from .stage_i_target_adapter import (
    FOLD_QUANTILE_RESIDUAL3,
    LEGACY_R3_MULTICLASS3,
    StageITargetContract,
    bind_target_contract,
    canonical_sha256,
    file_sha256,
)
from .stage_i_target_specific_oos import (
    _EVALUATION_MONTHS,
    _EVALUATION_TARGET_CONTRACT_SCHEMA,
    _MONTH_CONTRACT_SCHEMA,
    _is_reserved_source_feature,
    _ROLE_CONTRACT_SCHEMA,
    _TRUST_FIELDS,
    _target_semantic_signature,
)
from .stage_i_shared_population import (
    SCHEMA as SHARED_POPULATION_SCHEMA,
    SharedPopulationError,
    validate_shared_population,
)


SCHEMA = "stage_i_target_specific_input_materialization_v1"
IDENTITY = ("candidate_id", "__ts__", "__symbol__")
_GENERATED = {"base_raw_score", *_TRUST_FIELDS}
_STATE = re.compile(r"^base_state_p\d+$")


class TargetSpecificInputMaterializationError(ValueError):
    """Raised when a source cannot prove an immutable causal handoff."""


@dataclass(frozen=True)
class TargetSpecificInputMaterializationSpec:
    selector_dir: Path
    base_selector_dir: Path
    meta_selector_dir: Path
    winner_bundle_path: Path
    output_dir: Path
    target_winner_dir: Path | None = None
    shared_population_dir: Path | None = None
    n_validation_folds: int = 4
    min_train_rows: int = 500


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TargetSpecificInputMaterializationError(f"{path}: expected a JSON object")
    return value


def _require_file_hash(manifest: Mapping[str, Any], key: str, path: Path) -> str:
    expected = str(manifest.get(key, ""))
    observed = file_sha256(path)
    if len(expected) != 64 or expected != observed:
        raise TargetSpecificInputMaterializationError(f"source artifact hash drift: {path}")
    return observed


def _selector_sources(
    root: Path, *, feature_columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, str]]:
    manifest_path = root / "manifest.json"
    features_path = root / "selector_features.parquet"
    ledger_path = root / "selector_ledger.parquet"
    feature_contract_path = root / "selector_feature_contract.json"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "complete" or manifest.get("schema") != "stage_i_selector_sample_v1":
        raise TargetSpecificInputMaterializationError("selector sample must be completed schema v1")
    integrity = manifest.get("artifact_integrity")
    if not isinstance(integrity, Mapping) or integrity.get("schema") != "stage_i_selector_artifact_integrity_v1":
        raise TargetSpecificInputMaterializationError("selector sample lacks immutable artifact integrity")
    hashes = {
        "selector_manifest": file_sha256(manifest_path),
        "selector_features": _require_file_hash(integrity, "selector_features_sha256", features_path),
        "selector_ledger": _require_file_hash(integrity, "selector_ledger_sha256", ledger_path),
        "selector_feature_contract": file_sha256(feature_contract_path),
    }
    if str(manifest.get("feature_contract_sha256", "")) != canonical_sha256(
        _read_json(feature_contract_path)
    ) and str(manifest.get("feature_contract_sha256", "")) != _read_json(feature_contract_path).get("feature_contract_sha256"):
        raise TargetSpecificInputMaterializationError("selector feature-contract lineage drift")
    columns = None
    if feature_columns is not None:
        columns = list(dict.fromkeys([*IDENTITY, *map(str, feature_columns)]))
    try:
        features = pd.read_parquet(features_path, columns=columns)
    except (KeyError, ValueError) as exc:
        raise TargetSpecificInputMaterializationError(
            "selected source feature is absent from selector feature parquet"
        ) from exc
    ledger = pd.read_parquet(ledger_path)
    if not features.loc[:, list(IDENTITY)].equals(ledger.loc[:, list(IDENTITY)]):
        raise TargetSpecificInputMaterializationError("selector feature/ledger identity order drift")
    if features.candidate_id.duplicated().any() or ledger.candidate_id.duplicated().any():
        raise TargetSpecificInputMaterializationError("selector identities must be unique")
    return features, ledger, manifest, hashes


def _selector_manifest(root: Path, side: str, *, expected_sha: str) -> tuple[dict[str, Any], str]:
    path = root / side / "manifest.json"
    if not path.is_file():
        side_root = root / side
        pointer_path = side_root / "resume_complete.json"
        if pointer_path.is_file():
            pointer = _read_json(pointer_path)
            if pointer.get("schema") != "stage_i_direct_fq3_resume_complete_v1" or pointer.get("side") != side:
                raise TargetSpecificInputMaterializationError(f"{side}: invalid direct-FQ3 resume pointer")
            relative = pointer.get("attempt_relative_path")
            if not isinstance(relative, str):
                raise TargetSpecificInputMaterializationError(f"{side}: direct-FQ3 resume pointer lacks attempt path")
            resolved = (side_root / relative / "manifest.json").resolve()
            attempts = (side_root / "_resume_attempts").resolve()
            if attempts not in resolved.parents or resolved.parent.parent != attempts:
                raise TargetSpecificInputMaterializationError(f"{side}: direct-FQ3 resume pointer escapes attempt root")
            if not resolved.is_file() or file_sha256(resolved) != pointer.get("attempt_manifest_sha256"):
                raise TargetSpecificInputMaterializationError(f"{side}: direct-FQ3 resume manifest hash drift")
            path = resolved
    manifest = _read_json(path)
    observed = file_sha256(path)
    if observed != expected_sha or manifest.get("status") != "complete" or str(manifest.get("side", "")).lower() != side:
        raise TargetSpecificInputMaterializationError(f"{side}: selector manifest lineage drift")
    selected = tuple(map(str, manifest.get("selected_feature_contract", manifest.get("selected_features", ()))))
    if not selected or len(set(selected)) != len(selected):
        raise TargetSpecificInputMaterializationError(f"{side}: selected feature contract is empty/ambiguous")
    return manifest, observed


def _align(source: pd.DataFrame, target: pd.DataFrame, *, label: str) -> np.ndarray:
    left = pd.MultiIndex.from_frame(source.loc[:, list(IDENTITY)].astype(str))
    right = pd.MultiIndex.from_frame(target.loc[:, list(IDENTITY)].astype(str))
    positions = left.get_indexer(right)
    if (positions < 0).any() or len(np.unique(positions)) != len(positions):
        raise TargetSpecificInputMaterializationError(f"{label}: exact identity join failed")
    return positions.astype(np.int64)


def _training_universe(
    manifest: Mapping[str, Any], selected: Sequence[str], *, side: str, layer: str,
) -> dict[str, Any]:
    declared_values = manifest.get("input_feature_contract", manifest.get("stage_i_input_features", ()))
    declared = set(map(str, declared_values))
    generated = {name for name in selected if name in _GENERATED or _STATE.match(name)}
    raw = set(map(str, selected)).difference(generated)
    source = "selector_manifest_exact_input_feature_contract"
    if not declared:
        # Early direct-FQ3 selector manifests persisted the resolver lineage
        # but accidentally omitted its exact input list. Re-resolve the same
        # named config keys and bind the resulting hash in this new artifact;
        # never treat the full physical parquet schema as the allowed pool.
        from .config import CFG
        from .stage_i_feature_selection import resolve_stage_i_feature_universe

        head = FOLD_QUANTILE_RESIDUAL3 if layer == "meta" else None
        declared = set(resolve_stage_i_feature_universe(CFG, layer=layer, side=side, head=head))
        source = "materialization_time_config_key_resolution_for_legacy_missing_list"
    if not raw.issubset(declared):
        raise TargetSpecificInputMaterializationError(
            f"{side}/{layer}: selected raw fields are not bound to that layer's config-derived input universe: "
            f"{sorted(raw.difference(declared))[:8]}"
        )
    lineage = manifest.get("feature_universe_lineage", manifest.get("stage_i_feature_universe_lineage"))
    if not isinstance(lineage, Mapping) or str(lineage.get("layer", "")) != layer:
        raise TargetSpecificInputMaterializationError(f"{side}/{layer}: missing config feature-universe lineage")
    return {
        "source": source, "allowed_feature_count": len(declared),
        "allowed_feature_sha256": canonical_sha256(sorted(declared)),
        "selected_raw_feature_sha256": canonical_sha256(sorted(raw)),
        "selector_feature_universe_lineage": dict(lineage),
    }


def _winner_handoff(root: Path, *, side: str) -> tuple[pd.DataFrame, dict[str, Any], dict[str, str]]:
    manifest_path = root / "manifest.json"
    handoff_path = root / "winner_target_handoff.parquet"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "complete":
        raise TargetSpecificInputMaterializationError("target winner bundle is incomplete")
    inventory = dict(manifest.get("artifact_sha256", {}))
    expected = str(inventory.get(handoff_path.name, ""))
    observed = file_sha256(handoff_path)
    if expected != observed:
        raise TargetSpecificInputMaterializationError("winner target handoff hash drift")
    frame = pd.read_parquet(handoff_path)
    frame = frame.loc[frame.side_name.astype(str).str.lower().eq(side)].reset_index(drop=True)
    if frame.empty:
        raise TargetSpecificInputMaterializationError(f"{side}: winner target handoff is empty")
    return frame, manifest, {
        "target_winner_manifest": file_sha256(manifest_path),
        "target_winner_handoff": observed,
    }


def _shared_population(root: Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, str]]:
    """Load a predeclared R3/S/O common valid universe; never infer it here."""
    manifest_path = root / "manifest.json"
    population_path = root / "shared_population.parquet"
    try:
        frame, manifest = validate_shared_population(root)
    except SharedPopulationError as exc:
        raise TargetSpecificInputMaterializationError(str(exc)) from exc
    observed = file_sha256(population_path)
    required = {*IDENTITY, "side_name", "candidate_key"}
    if missing := required.difference(frame.columns):
        raise TargetSpecificInputMaterializationError(f"shared population lacks {sorted(missing)}")
    if frame.candidate_key.duplicated().any() or not frame.side_name.astype(str).str.lower().isin(("long", "short")).all():
        raise TargetSpecificInputMaterializationError("shared population identity/side contract is invalid")
    return frame, manifest, {
        "shared_population_manifest": file_sha256(manifest_path),
        "shared_population": observed,
        "shared_population_contract": str(manifest.get("contract_sha256", "")),
    }


def _validate_bundle_joint_finalist_authorization(
    bundle: StageIAdapterWinnerBundle, shared_manifest: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """Require a signed common universe whenever a published winner binds one."""
    authorization = bundle.joint_finalist_authorization
    if authorization is None:
        return None
    if shared_manifest is None:
        raise TargetSpecificInputMaterializationError(
            "authorized joint finalist requires --shared-population-dir"
        )
    if str(authorization["shared_population_contract_sha256"]) != str(shared_manifest.get("contract_sha256", "")):
        raise TargetSpecificInputMaterializationError(
            "winner bundle and shared population contract disagree"
        )
    expected_per_side = authorization["shared_population_per_side"]
    if dict(shared_manifest.get("per_side", {})) != dict(expected_per_side):
        raise TargetSpecificInputMaterializationError(
            "winner bundle and shared population per-side universe disagree"
        )
    return authorization


def _r3_contract_source(
    *, side: str, base_root: Path, ledger: pd.DataFrame,
    base_contract: StageITargetContract, meta_contract: StageITargetContract,
) -> tuple[pd.DataFrame, Path, dict[str, str]]:
    oof_path = base_root / side / "selector_base_oof.parquet"
    manifest = _read_json(base_root / side / "manifest.json")
    observed = file_sha256(oof_path)
    if str(manifest.get("selector_base_oof_sha256", "")) != observed:
        raise TargetSpecificInputMaterializationError(f"{side}: frozen R3 OOF hash drift")
    frame = pd.read_parquet(oof_path)
    if not frame.side_name.astype(str).str.lower().eq(side).all():
        raise TargetSpecificInputMaterializationError(f"{side}: frozen R3 OOF is cross-side")
    side_ledger = ledger.loc[ledger.side_name.astype(str).str.lower().eq(side)].reset_index(drop=True)
    positions = _align(side_ledger, frame, label=f"{side}/frozen-R3 ledger")
    support = side_ledger.iloc[positions].reset_index(drop=True)
    contract = frame.loc[:, [*IDENTITY, "side_name", "decision_ts", "label_available_ts"]].copy()
    if len(base_contract.target_columns) != 1 or len(meta_contract.target_columns) != 1:
        raise TargetSpecificInputMaterializationError("frozen R3/FQ3 contracts require one target column")
    contract[base_contract.target_columns[0]] = pd.to_numeric(
        support["r3_class"], errors="raise",
    ).to_numpy(np.float32)
    contract["target_valid"] = np.isfinite(pd.to_numeric(frame["exact_net_bps"], errors="coerce"))
    contract["gross_bps"] = pd.to_numeric(frame["exact_gross_bps"], errors="coerce").to_numpy(np.float32)
    contract["net_bps"] = pd.to_numeric(frame["exact_net_bps"], errors="coerce").to_numpy(np.float32)
    contract[meta_contract.target_columns[0]] = contract["net_bps"].to_numpy(np.float32)
    contract[base_contract.weight_column] = 1.0
    if meta_contract.weight_column not in contract:
        contract[meta_contract.weight_column] = 1.0
    return contract, oof_path.resolve(), {"frozen_r3_oof": observed}


def _winner_contract_source(
    *, side: str, target_root: Path, base_contract: StageITargetContract,
    meta_contract: StageITargetContract,
) -> tuple[pd.DataFrame, None, dict[str, str]]:
    handoff, _manifest, hashes = _winner_handoff(target_root, side=side)
    required = {*IDENTITY, "side_name", "decision_ts", "label_available_ts", "target_value", "target_valid", "gross_bps", "net_bps"}
    if missing := required.difference(handoff.columns):
        raise TargetSpecificInputMaterializationError(f"{side}: winner target handoff lacks {sorted(missing)}")
    contract = handoff.loc[:, [*IDENTITY, "side_name", "decision_ts", "label_available_ts", "target_valid", "gross_bps", "net_bps"]].copy()
    if len(base_contract.target_columns) != 1 or len(meta_contract.target_columns) != 1:
        raise TargetSpecificInputMaterializationError("direct S/O and FQ3 contracts require one target column")
    contract[base_contract.target_columns[0]] = pd.to_numeric(handoff.target_value, errors="coerce").to_numpy(np.float32)
    contract[meta_contract.target_columns[0]] = pd.to_numeric(handoff.net_bps, errors="coerce").to_numpy(np.float32)
    base_weight = base_contract.weight_column
    if base_weight in handoff:
        contract[base_weight] = pd.to_numeric(handoff[base_weight], errors="raise").to_numpy(np.float32)
    elif "sample_weight_base_component" in handoff:
        contract[base_weight] = pd.to_numeric(handoff.sample_weight_base_component, errors="raise").to_numpy(np.float32)
    else:
        raise TargetSpecificInputMaterializationError(f"{side}: winner handoff lacks base weight {base_weight}")
    # Contract certainty is deliberately not an inference feature: it is
    # training-only label/geometry information.  It must nevertheless travel
    # with the immutable evaluation contract so every strict OOF fold can
    # recompute the authorised certainty weights from *its own* prior-resolved
    # rows.  Previously the scalar S handoff kept the base component but
    # dropped this source field, making the documented ``contract_certainty``
    # weight mode impossible to replay.
    weight_mode = str(
        (base_contract.metadata.get("training_weight_contract") or {}).get(
            "mode", "uniform"
        )
    )
    if weight_mode in {"contract_certainty", "hybrid"}:
        if "contract_certainty" not in handoff:
            raise TargetSpecificInputMaterializationError(
                f"{side}: {weight_mode} base contract lacks training-only contract_certainty"
            )
        certainty = pd.to_numeric(handoff["contract_certainty"], errors="coerce")
        if not np.isfinite(certainty).all() or ((certainty < 0.0) | (certainty > 1.0)).any():
            raise TargetSpecificInputMaterializationError(
                f"{side}: training-only contract_certainty is invalid"
            )
        contract["contract_certainty"] = certainty.to_numpy(np.float32)
    if meta_contract.weight_column not in contract:
        contract[meta_contract.weight_column] = 1.0
    return contract, None, hashes


def _month_contract(frame: pd.DataFrame, *, side: str) -> dict[str, Any]:
    timestamps = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise TargetSpecificInputMaterializationError(f"{side}: signal timestamps are not finite UTC")
    month_counts = timestamps.dt.strftime("%Y-%m").value_counts().to_dict()
    available_months = sorted(month for month in month_counts if month in _EVALUATION_MONTHS)
    first = min(available_months) if available_months else None
    last = max(available_months) if available_months else None
    availability: dict[str, dict[str, Any]] = {}
    for month in _EVALUATION_MONTHS:
        rows = int(month_counts.get(month, 0))
        if rows:
            availability[month] = {"source_available": True, "candidate_rows": rows}
        else:
            if first is None:
                reason = "supplied immutable source contains no 2024-2026 candidate rows"
                kind = "source_unavailable"
            elif month < first:
                reason = f"supplied immutable source begins in {first}; compatible earlier feature rows are not materialised"
                kind = "historical_source_gap"
            elif month > last:
                reason = f"supplied immutable source ends in {last}; later feature/label rows are not materialised"
                kind = "future_or_later_source_gap"
            else:
                reason = "supplied immutable source has no candidate rows for this intervening month"
                kind = "candidate_population_gap"
            availability[month] = {
                "source_available": False, "candidate_rows": 0,
                "source_gap_reason": reason, "source_gap_kind": kind,
                "allow_zero_strict_coverage": True, "zero_coverage_reason": reason,
            }
    return {
        "schema": _MONTH_CONTRACT_SCHEMA, "side": side,
        "expected_months": list(_EVALUATION_MONTHS),
        "source_availability": availability,
        "available_months": available_months,
        "first_available_month": first, "last_available_month": last,
    }


def _bind_evaluation_contract(
    frame: pd.DataFrame, training: StageITargetContract,
) -> StageITargetContract:
    return bind_target_contract(
        frame, family=training.family, layer=training.layer,
        target_name=training.target_name, geometry=training.geometry,
        target_columns=training.target_columns,
        economics_columns=training.economics_columns,
        validity_column=training.validity_column,
        weight_column=training.weight_column,
        metadata=training.metadata,
    )


def _materialise_side(
    *, spec: TargetSpecificInputMaterializationSpec, side: str,
    features_all: pd.DataFrame, ledger: pd.DataFrame,
    selector_manifest: Mapping[str, Any], selector_hashes: Mapping[str, str],
    bundle: StageIAdapterWinnerBundle, temporary_root: Path,
    shared_population: pd.DataFrame | None,
    shared_population_manifest: Mapping[str, Any] | None,
    shared_population_hashes: Mapping[str, str],
) -> dict[str, Any]:
    cell = bundle.cell(side)
    base_manifest, base_manifest_sha = _selector_manifest(
        spec.base_selector_dir, side, expected_sha=cell.base_selector_manifest_sha256,
    )
    meta_manifest, meta_manifest_sha = _selector_manifest(
        spec.meta_selector_dir, side, expected_sha=cell.meta_selector_manifest_sha256,
    )
    base_universe_evidence = _training_universe(
        base_manifest, cell.base_features, side=side, layer="base",
    )
    meta_universe_evidence = _training_universe(
        meta_manifest, cell.meta_features, side=side, layer="meta",
    )
    if cell.meta_target_contract.family != FOLD_QUANTILE_RESIDUAL3:
        raise TargetSpecificInputMaterializationError(f"{side}: materializer requires direct FQ3 meta")
    if cell.base_target_contract.family == LEGACY_R3_MULTICLASS3:
        contract, frozen_path, label_hashes = _r3_contract_source(
            side=side, base_root=spec.base_selector_dir, ledger=ledger,
            base_contract=cell.base_target_contract, meta_contract=cell.meta_target_contract,
        )
    else:
        if spec.target_winner_dir is None:
            raise TargetSpecificInputMaterializationError("S/O materialization requires --target-winner-dir")
        contract, frozen_path, label_hashes = _winner_contract_source(
            side=side, target_root=spec.target_winner_dir,
            base_contract=cell.base_target_contract, meta_contract=cell.meta_target_contract,
        )
    if contract.candidate_id.duplicated().any() or not contract.side_name.astype(str).str.lower().eq(side).all():
        raise TargetSpecificInputMaterializationError(f"{side}: contract identities are invalid")
    if shared_population is not None:
        common = shared_population.loc[
            shared_population.side_name.astype(str).str.lower().eq(side)
        ].copy()
        contract_keys = contract.side_name.astype(str).str.lower() + "::" + contract.candidate_id.astype(str)
        positions = pd.Index(contract_keys).get_indexer(common.candidate_key.astype(str))
        if (positions < 0).any() or len(np.unique(positions)) != len(positions):
            raise TargetSpecificInputMaterializationError(f"{side}: shared population is not a strict subset of this target contract")
        contract = contract.iloc[positions].reset_index(drop=True)
        if not contract.loc[:, list(IDENTITY)].astype(str).reset_index(drop=True).equals(
            common.loc[:, list(IDENTITY)].astype(str).reset_index(drop=True)
        ):
            raise TargetSpecificInputMaterializationError(f"{side}: shared population timing/identity drift")
        if not contract.target_valid.astype(bool).all():
            raise TargetSpecificInputMaterializationError(
                f"{side}: shared population contains a non-valid target row"
            )
    positions = _align(features_all, contract, label=f"{side}/feature source")
    generated = {name for name in cell.meta_features if name in _GENERATED or _STATE.match(name)}
    base_raw = tuple(map(str, cell.base_features))
    meta_raw = tuple(name for name in map(str, cell.meta_features) if name not in generated)
    raw_union = tuple(dict.fromkeys((*base_raw, *meta_raw)))
    if forbidden := sorted(name for name in raw_union if _is_reserved_source_feature(name)):
        raise TargetSpecificInputMaterializationError(f"{side}: selected source feature uses reserved namespace: {forbidden}")
    if missing := set(raw_union).difference(features_all.columns):
        raise TargetSpecificInputMaterializationError(f"{side}: source features absent: {sorted(missing)[:8]}")
    feature_frame = features_all.iloc[positions].loc[:, [*IDENTITY, *raw_union]].reset_index(drop=True)
    if any(name in feature_frame for name in generated):
        raise TargetSpecificInputMaterializationError(f"{side}: generated base/meta handoff leaked into source features")
    if not feature_frame.loc[:, list(IDENTITY)].astype(str).equals(contract.loc[:, list(IDENTITY)].astype(str)):
        raise TargetSpecificInputMaterializationError(f"{side}: final feature/contract identity drift")
    decision = pd.to_datetime(contract.decision_ts, utc=True, errors="coerce")
    signal = pd.to_datetime(contract["__ts__"], utc=True, errors="coerce")
    available = pd.to_datetime(contract.label_available_ts, utc=True, errors="coerce")
    if decision.isna().any() or available.isna().any() or not (decision - signal).eq(pd.Timedelta(hours=1)).all() or not (available - decision).eq(pd.Timedelta(hours=12)).all():
        raise TargetSpecificInputMaterializationError(f"{side}: timing is not close -> +1h entry -> +12h label")
    valid = contract.target_valid.astype(bool).to_numpy()
    gross = pd.to_numeric(contract.gross_bps, errors="coerce").to_numpy(float)
    net = pd.to_numeric(contract.net_bps, errors="coerce").to_numpy(float)
    if not np.isfinite(gross[valid]).all() or not np.isfinite(net[valid]).all() or not np.allclose(gross[valid] - 100.0, net[valid], atol=2e-3, rtol=0.0):
        raise TargetSpecificInputMaterializationError(f"{side}: economics do not apply exactly one 100bps cost")
    evaluation_base = _bind_evaluation_contract(contract, cell.base_target_contract)
    evaluation_meta = _bind_evaluation_contract(contract, cell.meta_target_contract)
    evaluation_contracts = {
        "schema": _EVALUATION_TARGET_CONTRACT_SCHEMA, "side": side,
        "training_base_target_contract_sha256": cell.base_target_contract.sha256,
        "training_meta_target_contract_sha256": cell.meta_target_contract.sha256,
        "base": evaluation_base.to_dict(), "meta": evaluation_meta.to_dict(),
    }
    role_contract = {
        "schema": _ROLE_CONTRACT_SCHEMA, "side": side,
        "base_source_features": list(base_raw), "meta_source_features": list(meta_raw),
        "generated_meta_features_excluded_from_source": sorted(generated),
        "base_config_feature_universe_evidence": base_universe_evidence,
        "meta_config_feature_universe_evidence": meta_universe_evidence,
    }
    months = _month_contract(contract, side=side)
    destination = temporary_root / side
    destination.mkdir(parents=True)
    feature_path, contract_path = destination / "features.parquet", destination / "contract.parquet"
    feature_frame.to_parquet(feature_path, index=False, compression="zstd")
    contract.to_parquet(contract_path, index=False, compression="zstd")
    source_lineage = {
        **dict(selector_hashes), **label_hashes, **dict(shared_population_hashes),
        "base_selector_manifest": base_manifest_sha,
        "meta_selector_manifest": meta_manifest_sha,
        "winner_bundle": file_sha256(spec.winner_bundle_path),
    }
    manifest: dict[str, Any] = {
        "schema": SCHEMA, "status": "complete", "side": side, "rows": len(contract),
        "base_target_column": cell.base_target_contract.target_columns[0],
        "meta_target_column": cell.meta_target_contract.target_columns[0],
        "n_validation_folds": int(spec.n_validation_folds), "min_train_rows": int(spec.min_train_rows),
        "artifact_sha256": {
            feature_path.name: file_sha256(feature_path), contract_path.name: file_sha256(contract_path),
        },
        "source_lineage_sha256": source_lineage,
        "selector_source_manifest": dict(selector_manifest),
        "winner_bundle_sha256": bundle.sha256,
        "training_target_semantics": {
            "base": _target_semantic_signature(cell.base_target_contract),
            "meta": _target_semantic_signature(cell.meta_target_contract),
        },
        "evaluation_target_contracts": evaluation_contracts,
        "evaluation_target_contracts_sha256": canonical_sha256(evaluation_contracts),
        "causal_feature_role_contract": role_contract,
        "causal_feature_role_contract_sha256": canonical_sha256(role_contract),
        "evaluation_month_contract": months,
        "evaluation_month_contract_sha256": canonical_sha256(months),
        "identity_contract": {
            "columns": list(IDENTITY), "unique": True, "ordered_join": True,
            "sha256": evaluation_base.identity_sha256,
        },
        "timing_contract": {
            "signal_column": "__ts__", "decision_column": "decision_ts",
            "label_available_column": "label_available_ts",
            "signal_to_decision_hours": 1, "decision_to_label_available_hours": 12,
            "utc_verified": True,
        },
        "economics_contract": {
            "gross_column": "gross_bps", "net_column": "net_bps",
            "cost_bps": 100.0, "cost_application_count": 1,
            "valid_rows": int(valid.sum()), "invalid_rows": int((~valid).sum()),
            "gross_sha256": canonical_sha256(contract.gross_bps.tolist()),
            "net_sha256": canonical_sha256(contract.net_bps.tolist()),
        },
        "weight_contract": {
            "base_column": evaluation_base.weight_column,
            "base_sha256": evaluation_base.weight_sha256,
            "meta_column": evaluation_meta.weight_column,
            "meta_sha256": evaluation_meta.weight_sha256,
            "weights_are_training_only_not_features": True,
        },
        "source_feature_contract": {
            "columns": list(raw_union), "column_count": len(raw_union),
            "sha256": canonical_sha256(list(raw_union)),
            "generated_fields_present": False, "reserved_fields_present": False,
        },
        "frozen_base_oof_path": None if frozen_path is None else str(frozen_path),
        "frozen_base_oof_lineage": None if frozen_path is None else {
            "path": str(frozen_path), "sha256": label_hashes["frozen_r3_oof"],
            "selector_manifest_sha256": base_manifest_sha,
        },
        "shared_population_contract_sha256": None if shared_population_manifest is None else shared_population_manifest.get("contract_sha256"),
        "shared_population_rows": None if shared_population is None else int(len(contract)),
        "joint_finalist_authorization": bundle.joint_finalist_authorization,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "side": side, "rows": len(contract), "features": len(raw_union),
        "available_months": months["available_months"],
        "frozen_r3_oof": frozen_path is not None,
    }


def materialize_stage_i_target_specific_inputs(spec: TargetSpecificInputMaterializationSpec) -> dict[str, Any]:
    """Write both side inputs atomically; never fit or score a model."""
    if spec.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite target-specific inputs: {spec.output_dir}")
    if spec.n_validation_folds < 2 or spec.min_train_rows < 1:
        raise TargetSpecificInputMaterializationError("invalid OOS fold settings")
    bundle_payload = _read_json(spec.winner_bundle_path)
    bundle = StageIAdapterWinnerBundle.from_dict(bundle_payload)
    if canonical_sha256(bundle_payload) != bundle.sha256:
        raise TargetSpecificInputMaterializationError("winner bundle canonical hash drift")
    selected_raw = []
    for cell in bundle.cells:
        selected_raw.extend(cell.base_features)
        selected_raw.extend(
            name for name in cell.meta_features
            if name not in _GENERATED and not _STATE.match(str(name))
        )
    features, ledger, selector_manifest, selector_hashes = _selector_sources(
        spec.selector_dir, feature_columns=tuple(dict.fromkeys(map(str, selected_raw))),
    )
    shared_population: pd.DataFrame | None = None
    shared_manifest: dict[str, Any] | None = None
    shared_hashes: dict[str, str] = {}
    if spec.shared_population_dir is not None:
        shared_population, shared_manifest, shared_hashes = _shared_population(spec.shared_population_dir)
    authorization = _validate_bundle_joint_finalist_authorization(bundle, shared_manifest)
    spec.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{spec.output_dir.name}.", dir=spec.output_dir.parent))
    try:
        cells = [
            _materialise_side(
                spec=spec, side=side, features_all=features, ledger=ledger,
                selector_manifest=selector_manifest, selector_hashes=selector_hashes,
                bundle=bundle, temporary_root=temporary,
                shared_population=shared_population,
                shared_population_manifest=shared_manifest,
                shared_population_hashes=shared_hashes,
            )
            for side in ("long", "short")
        ]
        root_manifest = {
            "schema": SCHEMA, "status": "complete", "cells": cells,
            "winner_bundle_sha256": bundle.sha256,
            "shared_population_contract_sha256": None if shared_manifest is None else shared_manifest.get("contract_sha256"),
            "joint_finalist_authorization": authorization,
            "publication": "atomic_directory_rename",
            "model_fit_performed": False,
        }
        (temporary / "manifest.json").write_text(json.dumps(root_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.rename(temporary, spec.output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return root_manifest


__all__ = [
    "SCHEMA", "TargetSpecificInputMaterializationError",
    "TargetSpecificInputMaterializationSpec", "materialize_stage_i_target_specific_inputs",
]
