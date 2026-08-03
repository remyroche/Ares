#!/usr/bin/env python3
"""Append frozen side-local AE/GMM features to retrospective Pack-B inputs.

The primary input is the manifest-bound *pre-score* candidate ledger.  The
frozen base long model consumes AE/GMM columns, so those learned columns must
exist before ``score_packb_final_refits_forward.py`` runs.  A scored Pack-B
context remains supported for downstream-only reconstruction.  Both modes are
retrospective, non-promotable research infrastructure and never read outcomes
or labels.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.base_candidate_population import (
    candidate_identity_sha256,
    deterministic_candidate_ids,
)
from extreme_price_movements.packb_static_point_feature_loader import (
    FrozenFeatureContract,
    _provenance_backed_raw_allowlist,
    iter_point_in_time_feature_batches,
    point_feature_matrix_sha256,
)
from extreme_price_movements.training_resource_guard import (
    TrainingResourceGuard,
    TrainingResourceLimits,
)
from scripts.materialize_packb_downstream_representation import (
    DownstreamRepresentationError,
    append_side_representation,
)
from scripts.run_packb_pre_march_side_ae import DEFAULT_FEATURE_STORE
from scripts.run_packb_pre_march_side_fs_hpo import (
    SideRepresentationFeatureLoader,
    _active_ae_gmm_columns,
    _load_loader_contract,
    _load_side_ae_state,
)
from scripts.run_packb_side_local_residual_oof import _side_loader


SCHEMA = "packb_final_refit_retrospective_side_representation_v1"
SIDES = ("long", "short")
IDENTITY_COLUMNS = ("__ts__", "__symbol__", "side_name", "candidate_id")
DEFAULT_CONTEXT_ROOT = ROOT / "data_perp/artifacts/execution_ev_july20_23_retrospective_20260730_v1/packb"
DEFAULT_CANDIDATE_ROOT = ROOT / "data_perp/artifacts/execution_ev_july20_23_retrospective_20260730_v1/candidates"
DEFAULT_AE_ROOT = ROOT / "data_perp/artifacts/packb_side_local_ae_20260724_v1"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/execution_ev_july20_23_retrospective_20260730_v1/packb_with_frozen_representation"
DEFAULT_HISTORICAL_REPRESENTATION_ROOT = (
    ROOT / "data_perp/artifacts/packb_downstream_representation_july20_20260726_v1_31_8"
)
RESOURCE_TELEMETRY_FILENAME = "training_resource_telemetry.jsonl"
# These are the frozen learned outputs found in the July execution-EV model
# contracts.  ``gmm_representation_available`` is created by the append step;
# the other ten must be supplied by each side's frozen AE/GMM state.
FROZEN_EXECUTION_EV_REPRESENTATION = (
    "dae_b16_00",
    "dae_b16_02",
    "dae_b16_04",
    "dae_b16_08",
    "dae_b16_14",
    "expected_mahalanobis",
    "gmm_cluster_posterior_4",
    "gmm_dist_center_4",
    "gmm_dist_center_9",
    "gmm_ood_score",
    "gmm_representation_available",
)
FROZEN_EXECUTION_EV_GENERATED = tuple(
    feature
    for feature in FROZEN_EXECUTION_EV_REPRESENTATION
    if feature != "gmm_representation_available"
)
FROZEN_EXECUTION_EV_GENERATED_SET = frozenset(FROZEN_EXECUTION_EV_GENERATED)


class RetrospectiveRepresentationError(RuntimeError):
    """Raised if a final-refit context cannot be safely augmented."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime, Path)):
        return str(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(_jsonable(dict(payload)), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _source_revision() -> tuple[str, bool]:
    """Record the local source revision without claiming a clean release tree."""

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    if revision.returncode != 0 or not revision.stdout.strip():
        return "unavailable", False
    clean = subprocess.run(
        ["git", "diff", "--quiet"], cwd=ROOT, check=False
    ).returncode == 0
    return revision.stdout.strip(), clean


def _manifest_path(value: object, *, root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _validate_context(
    *, context_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any], Path, Path]:
    """Validate the scored, selected stream and its immutable provenance."""

    context_path = context_root / "packb_forward_context.parquet"
    manifest_path = context_root / "manifest.json"
    if not context_path.is_file() or not manifest_path.is_file():
        raise RetrospectiveRepresentationError(
            "manifest-bound Pack-B final-refit context is required"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = manifest.get("output", {})
    contract = manifest.get("contract", {})
    if (
        manifest.get("schema") != "packb_final_refits_forward_v1"
        or manifest.get("status") != "frozen_final_refit_preentry_context_not_oos_metrics"
        or output.get("sha256") != _sha256(context_path)
        or contract.get("outcomes_used") is not False
    ):
        raise RetrospectiveRepresentationError(
            "Pack-B final-refit context manifest is not an exact pre-entry binding"
        )
    candidate_record = manifest.get("inputs", {}).get("candidate_features", {})
    candidate_path = _manifest_path(candidate_record.get("path", ""), root=ROOT)
    if (
        not candidate_path.is_file()
        or candidate_record.get("sha256") != _sha256(candidate_path)
    ):
        raise RetrospectiveRepresentationError(
            "Pack-B final-refit candidate-source provenance changed"
        )
    frame = _validate_identity_and_timing(
        pd.read_parquet(context_path), require_selected_top40=True
    )
    if set(frame["prediction_source"].astype(str)) != {"frozen_final_refit"}:
        raise RetrospectiveRepresentationError(
            "Pack-B final-refit context prediction-source contract changed"
        )
    return frame, manifest, context_path, manifest_path


def _validate_identity_and_timing(
    frame: pd.DataFrame,
    *,
    require_selected_top40: bool,
) -> pd.DataFrame:
    """Normalize and fail-close the identity/timing shared by both input modes."""

    required = {
        *IDENTITY_COLUMNS,
        "side",
        "execution_decision_utc",
        "feature_available_at",
    }
    if require_selected_top40:
        required.add("selected_top40")
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RetrospectiveRepresentationError(
            f"Pack-B final-refit context is missing: {missing}"
        )
    frame = frame.copy()
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    frame["execution_decision_utc"] = pd.to_datetime(
        frame["execution_decision_utc"], utc=True, errors="raise"
    )
    frame["feature_available_at"] = pd.to_datetime(
        frame["feature_available_at"], utc=True, errors="raise"
    )
    frame["__symbol__"] = frame["__symbol__"].astype(str)
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["side_name"] = frame["side_name"].astype(str).str.lower()
    numeric_side = pd.to_numeric(frame["side"], errors="coerce")
    expected_numeric_side = frame["side_name"].map({"long": 1.0, "short": -1.0})
    expected_decision = frame["__ts__"] + pd.Timedelta(hours=1)
    if (
        frame["candidate_id"].duplicated().any()
        or frame.duplicated(list(IDENTITY_COLUMNS)).any()
        or set(frame["side_name"]) != set(SIDES)
        or (require_selected_top40 and not frame["selected_top40"].astype(bool).all())
        or not np.isfinite(numeric_side.to_numpy(dtype=float)).all()
        or not numeric_side.eq(expected_numeric_side).all()
        or not frame["execution_decision_utc"].eq(expected_decision).all()
        or (frame["feature_available_at"] > frame["execution_decision_utc"]).any()
        or not frame["candidate_id"].eq(
            deterministic_candidate_ids(frame, timeframe="1h").astype(str)
        ).all()
    ):
        raise RetrospectiveRepresentationError(
            "candidate identity, side, selection, or timing contract changed"
        )
    return frame


def _validate_prescore_candidates(
    *,
    candidate_features: Path,
    source_manifest_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate the immutable retrospective raw-candidate surface."""

    if not candidate_features.is_file() or not source_manifest_path.is_file():
        raise RetrospectiveRepresentationError(
            "manifest-bound pre-score candidate_features/source_manifest is required"
        )
    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    output = manifest.get("output", {})
    if (
        manifest.get("schema") != "execution_ev_july_retrospective_candidate_surface_v1"
        or manifest.get("status") != "materialized_retrospective_non_promotable"
        or manifest.get("outcomes_used") is not False
        or manifest.get("candidates_written") is not True
        or output.get("sha256") != _sha256(candidate_features)
    ):
        raise RetrospectiveRepresentationError(
            "pre-score candidate manifest is not an exact outcome-free binding"
        )
    frame = _validate_identity_and_timing(
        pd.read_parquet(candidate_features), require_selected_top40=False
    )
    if (
        int(output.get("rows", -1)) != len(frame)
        or int(output.get("columns", -1)) != len(frame.columns)
    ):
        raise RetrospectiveRepresentationError(
            "pre-score candidate manifest row/column binding changed"
        )
    return frame, manifest


def _validate_ae_provenance(ae_root: Path) -> tuple[dict[str, Any], Path]:
    summary_path = ae_root / "summary.json"
    if not summary_path.is_file():
        raise RetrospectiveRepresentationError("frozen side-local AE/GMM summary is required")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "FROZEN_LONG_AND_SHORT_AE_GMM":
        raise RetrospectiveRepresentationError("AE/GMM summary is not frozen for both sides")
    state_hashes: set[str] = set()
    for side in SIDES:
        record = summary.get("sides", {}).get(side, {}).get("ae_gmm", {})
        state_path = _manifest_path(record.get("state_path", ""), root=ROOT)
        state_hash = str(record.get("state_sha256") or "")
        if (
            record.get("status") != "FROZEN_SIDE_LOCAL_AE_GMM_STATE"
            or record.get("side") != side
            or not state_path.is_file()
            or not state_hash
            or _sha256(state_path) != state_hash
        ):
            raise RetrospectiveRepresentationError(
                f"{side} frozen AE/GMM state provenance changed"
            )
        state_hashes.add(state_hash)
    if len(state_hashes) != len(SIDES):
        raise RetrospectiveRepresentationError("long and short must use distinct frozen AE/GMM states")
    return summary, summary_path


def _validate_frozen_registry_replay_binding(
    *,
    loader_root: Path,
    contract: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, str]:
    """Authorize one audited historical-contract replay after registry drift.

    This is intentionally narrower than ``verify_frozen_schema=False``.  It
    accepts a changed registry *only* when the frozen raw-universe snapshot,
    feature contract and loader evidence still bind each other, and the current
    registry derives exactly the same raw allowlist.  Exact reads and a value
    hash are still required for the requested ledger.
    """

    universe_path = loader_root / "raw_feature_universe.json"
    contract_path = loader_root / "frozen_feature_contract.json"
    evidence_path = loader_root / "loader_evidence.json"
    if not all(path.is_file() for path in (universe_path, contract_path, evidence_path)):
        raise RetrospectiveRepresentationError(
            "frozen-registry replay requires the complete immutable loader-evidence snapshot"
        )
    universe = json.loads(universe_path.read_text(encoding="utf-8"))
    on_disk_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    on_disk_evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    frozen = FrozenFeatureContract.from_mapping(contract)
    if (
        FrozenFeatureContract.from_mapping(on_disk_contract).feature_contract_sha256
        != frozen.feature_contract_sha256
        or universe.get("universe_sha256") != frozen.candidate_universe_sha256
        or universe.get("source_schema_sha256") != frozen.source_schema_sha256
        or universe.get("raw_allowlist_sha256") != frozen.raw_allowlist_sha256
        or universe.get("generator_registry_sha256") != frozen.generator_registry_sha256
        or universe.get("store_scan_manifest_sha256") != frozen.store_scan_manifest_sha256
        or not set(frozen.feature_columns).issubset(
            set(map(str, universe.get("feature_columns", ())))
        )
        or evidence.get("feature_contract_sha256") != frozen.feature_contract_sha256
        or evidence.get("raw_universe_sha256") != frozen.candidate_universe_sha256
        or evidence.get("source_schema_sha256") != frozen.source_schema_sha256
        or on_disk_evidence.get("feature_contract_sha256")
        != frozen.feature_contract_sha256
        or on_disk_evidence.get("raw_universe_sha256")
        != frozen.candidate_universe_sha256
        or on_disk_evidence.get("source_schema_sha256")
        != frozen.source_schema_sha256
    ):
        raise RetrospectiveRepresentationError(
            "frozen-registry replay evidence does not bind the frozen raw contract"
        )
    _current_allowlist, _rejected, current_allowlist_sha256, current_registry_sha256 = (
        _provenance_backed_raw_allowlist()
    )
    if current_allowlist_sha256 != frozen.raw_allowlist_sha256:
        raise RetrospectiveRepresentationError(
            "current raw allowlist differs from the frozen contract; rebuild is required"
        )
    frozen_revision = str(evidence.get("source_revision") or "")
    if len(frozen_revision) != 40 or any(
        value not in "0123456789abcdef" for value in frozen_revision.lower()
    ):
        raise RetrospectiveRepresentationError(
            "frozen-registry replay has no immutable source revision"
        )

    def source_at_revision(relative: str) -> bytes:
        result = subprocess.run(
            ["git", "show", f"{frozen_revision}:{relative}"],
            cwd=ROOT,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0 or not result.stdout:
            raise RetrospectiveRepresentationError(
                f"frozen-registry replay cannot recover {relative} at {frozen_revision}"
            )
        return result.stdout

    def function_ast_hash(source: bytes, name: str) -> str:
        tree = ast.parse(source.decode("utf-8"))
        nodes = [
            node for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ]
        if len(nodes) != 1:
            raise RetrospectiveRepresentationError(
                f"cannot uniquely recover frozen {name} implementation"
            )
        return hashlib.sha256(
            ast.dump(nodes[0], annotate_fields=True, include_attributes=False).encode(
                "utf-8"
            )
        ).hexdigest()

    frozen_config = source_at_revision("extreme_price_movements/config.py")
    frozen_features = source_at_revision("extreme_price_movements/features.py")
    frozen_pipeline = source_at_revision("extreme_price_movements/pipeline_steps.py")
    current_config = (ROOT / "extreme_price_movements/config.py").read_bytes()
    current_features = (ROOT / "extreme_price_movements/features.py").read_bytes()
    current_pipeline = (ROOT / "extreme_price_movements/pipeline_steps.py").read_bytes()
    frozen_expected_key_logic = function_ast_hash(
        frozen_pipeline, "_expected_feature_keys_from_cfg"
    )
    current_expected_key_logic = function_ast_hash(
        current_pipeline, "_expected_feature_keys_from_cfg"
    )
    if (
        hashlib.sha256(current_config).digest() != hashlib.sha256(frozen_config).digest()
        or hashlib.sha256(current_features).digest()
        != hashlib.sha256(frozen_features).digest()
        or current_expected_key_logic != frozen_expected_key_logic
    ):
        raise RetrospectiveRepresentationError(
            "config/features or expected-key generation changed; frozen replay requires rebuild"
        )
    from extreme_price_movements import config as epm_config
    from extreme_price_movements import pipeline_steps

    current_expected_keys = sorted(
        str(value)
        for value in pipeline_steps._expected_feature_keys_from_cfg(epm_config.CFG)
        if isinstance(value, str) and value
    )
    current_expected_keys_sha256 = hashlib.sha256(
        json.dumps(
            {"expected_feature_keys": current_expected_keys},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "mode": "frozen_snapshot_same_raw_allowlist_exact_value_hash",
        "frozen_generator_registry_sha256": frozen.generator_registry_sha256,
        "current_generator_registry_sha256": current_registry_sha256,
        "raw_allowlist_sha256": current_allowlist_sha256,
        "frozen_config_py_sha256": hashlib.sha256(frozen_config).hexdigest(),
        "current_config_py_sha256": hashlib.sha256(current_config).hexdigest(),
        "frozen_features_py_sha256": hashlib.sha256(frozen_features).hexdigest(),
        "current_features_py_sha256": hashlib.sha256(current_features).hexdigest(),
        "frozen_expected_key_logic_sha256": frozen_expected_key_logic,
        "current_expected_key_logic_sha256": current_expected_key_logic,
        "current_expected_feature_keys_sha256": current_expected_keys_sha256,
        "frozen_pipeline_steps_py_sha256": hashlib.sha256(frozen_pipeline).hexdigest(),
        "current_pipeline_steps_py_sha256": hashlib.sha256(current_pipeline).hexdigest(),
        "frozen_store_scan_manifest_sha256": frozen.store_scan_manifest_sha256,
        "raw_universe_snapshot_sha256": _sha256(universe_path),
        "frozen_contract_snapshot_sha256": _sha256(contract_path),
        "loader_evidence_snapshot_sha256": _sha256(evidence_path),
    }


def _frozen_registry_replay_side_loader(
    *,
    side: str,
    ae_root: Path,
    feature_store: Path,
    guard: TrainingResourceGuard,
) -> tuple[SideRepresentationFeatureLoader, list[str], dict[str, Any]]:
    """Build a one-run, snapshot-bound raw loader for registry-drift replay."""

    ae_summary = json.loads((ae_root / "summary.json").read_text(encoding="utf-8"))
    ae_revision = str(ae_summary.get("source_revision") or "")
    loader_root = ae_root / side / "loader_evidence"
    contract_mapping, evidence_bundle, loader_hashes = _load_loader_contract(
        loader_root, source_revision=ae_revision
    )
    replay_evidence = _validate_frozen_registry_replay_binding(
        loader_root=loader_root,
        contract=contract_mapping,
        evidence=evidence_bundle.to_dict(),
    )
    contract = FrozenFeatureContract.from_mapping(contract_mapping)
    raw_features = tuple(contract.feature_columns)
    matrix_evidence: dict[str, Any] = {}

    def raw_loader(ledger: pd.DataFrame, requested_features: Sequence[str]) -> pd.DataFrame:
        requested = tuple(map(str, requested_features))
        if (
            not requested
            or len(set(requested)) != len(requested)
            or not set(requested).issubset(raw_features)
        ):
            raise RetrospectiveRepresentationError(
                "frozen-registry replay requested an invalid raw feature subset"
            )
        matrix = np.empty((len(ledger), len(raw_features)), dtype=np.float32)
        matched = np.zeros(len(ledger), dtype=bool)
        for batch in iter_point_in_time_feature_batches(
            ledger,
            feature_store_dir=feature_store,
            feature_contract=contract,
            verify_frozen_schema=False,
            max_rows_per_batch=8_000,
            max_columns_per_read=64,
            coverage_discovery=False,
            resource_guard=guard,
        ):
            matrix[batch.ledger_row_positions, :] = batch.features.to_numpy(
                dtype=np.float32, copy=False
            )
            matched[batch.ledger_row_positions] = batch.matched_exact_keys
        finite = np.isfinite(matrix)
        if not matched.all() or not finite.all():
            nonfinite_columns = [
                name
                for name, complete in zip(raw_features, finite.all(axis=0))
                if not complete
            ]
            raise RetrospectiveRepresentationError(
                "frozen-registry replay has missing exact keys or non-finite raw "
                f"values (missing_exact_rows={int((~matched).sum())}, "
                f"nonfinite_features={len(nonfinite_columns)}, "
                f"sample={nonfinite_columns[:12]})"
            )
        frame = pd.DataFrame(matrix, columns=list(raw_features))
        matrix_evidence.update(
            {
                "rows": int(len(ledger)),
                "exact_key_rows": int(matched.sum()),
                "exact_key_fraction": float(matched.mean()) if len(matched) else 0.0,
                "point_feature_matrix_sha256": point_feature_matrix_sha256(
                    ledger, frame, feature_contract=contract
                ),
            }
        )
        return frame.loc[:, list(requested)].reset_index(drop=True)

    ae_manifest_path = ae_root / side / "ae_gmm" / "side_stage_manifest.json"
    ae_manifest = json.loads(ae_manifest_path.read_text(encoding="utf-8"))
    state_path = ae_root / side / "ae_gmm" / str(ae_manifest["artifact"]["path"])
    state = _load_side_ae_state(
        state_path,
        expected_side=side,
        expected_sha256=str(ae_manifest["artifact"]["sha256"]),
        raw_features=raw_features,
    )
    generated = list(_active_ae_gmm_columns(state))
    loader = SideRepresentationFeatureLoader(
        raw_loader=raw_loader,
        raw_features=raw_features,
        state=state,
        generated_features=generated,
    )
    evidence = {
        **loader_hashes,
        "ae_state_sha256": str(ae_manifest["artifact"]["sha256"]),
        "ae_manifest_sha256": _sha256(ae_manifest_path),
        "raw_candidate_features": len(raw_features),
        "generated_candidate_features": len(generated),
        "frozen_registry_replay": replay_evidence,
        "exact_value_hash": matrix_evidence,
    }
    return loader, [*raw_features, *generated], evidence


def _embedded_candidate_side_loader(
    *,
    side: str,
    candidate_context: pd.DataFrame,
    ae_root: Path,
) -> tuple[SideRepresentationFeatureLoader, list[str], dict[str, Any]]:
    """Use only the manifest-bound pre-score ledger for accepted inference.

    No feature-store schema bypass is permitted here.  The candidate surface
    must already contain the exact ordered frozen 256-column raw matrix for
    this side, with finite values for every candidate.
    """

    ae_summary = json.loads((ae_root / "summary.json").read_text(encoding="utf-8"))
    ae_revision = str(ae_summary.get("source_revision") or "")
    loader_root = ae_root / side / "loader_evidence"
    contract_mapping, _bundle, loader_hashes = _load_loader_contract(
        loader_root, source_revision=ae_revision
    )
    contract = FrozenFeatureContract.from_mapping(contract_mapping)
    raw_features = tuple(contract.feature_columns)
    local = candidate_context.loc[candidate_context["side_name"].eq(side)].copy()
    if local["candidate_id"].duplicated().any() or set(raw_features).difference(local.columns):
        missing = sorted(set(raw_features).difference(local.columns))
        raise RetrospectiveRepresentationError(
            f"{side} candidate ledger lacks the complete frozen AE raw contract: {missing[:12]}"
        )
    raw_values = local.loc[:, list(raw_features)].apply(pd.to_numeric, errors="coerce")
    values = raw_values.to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        bad = [
            name for name, complete in zip(raw_features, np.isfinite(values).all(axis=0))
            if not complete
        ]
        raise RetrospectiveRepresentationError(
            f"{side} candidate ledger frozen AE raw contract is non-finite: {bad[:12]}"
        )
    source_identity = local.loc[:, list(IDENTITY_COLUMNS)].copy()
    source_identity["candidate_id"] = source_identity["candidate_id"].astype(str)
    source_by_id = source_identity.set_index("candidate_id", drop=False)
    source_values = pd.DataFrame(values, columns=list(raw_features), index=source_by_id.index)
    matrix_evidence = {
        "rows": int(len(local)),
        "exact_key_rows": int(len(local)),
        "exact_key_fraction": 1.0,
        "point_feature_matrix_sha256": point_feature_matrix_sha256(
            source_identity, raw_values, feature_contract=contract
        ),
    }

    def raw_loader(ledger: pd.DataFrame, requested_features: Sequence[str]) -> pd.DataFrame:
        requested = tuple(map(str, requested_features))
        if (
            not requested
            or len(set(requested)) != len(requested)
            or not set(requested).issubset(raw_features)
            or ledger["candidate_id"].astype(str).duplicated().any()
        ):
            raise RetrospectiveRepresentationError(
                "embedded candidate replay requested an invalid raw feature subset"
            )
        requested_identity = ledger.loc[:, list(IDENTITY_COLUMNS)].copy()
        requested_identity["candidate_id"] = requested_identity["candidate_id"].astype(str)
        aligned_identity = source_by_id.reindex(requested_identity["candidate_id"])
        exact = (
            aligned_identity["candidate_id"].astype(str).to_numpy()
            == requested_identity["candidate_id"].to_numpy()
        ) & (
            pd.to_datetime(aligned_identity["__ts__"], utc=True).to_numpy()
            == pd.to_datetime(requested_identity["__ts__"], utc=True).to_numpy()
        ) & (
            aligned_identity["__symbol__"].astype(str).to_numpy()
            == requested_identity["__symbol__"].astype(str).to_numpy()
        ) & (
            aligned_identity["side_name"].astype(str).str.lower().to_numpy()
            == side
        )
        if not bool(np.all(exact)):
            raise RetrospectiveRepresentationError(
                "embedded candidate replay identity or side binding changed"
            )
        return source_values.reindex(requested_identity["candidate_id"]).loc[:, list(requested)].reset_index(drop=True)

    ae_manifest_path = ae_root / side / "ae_gmm" / "side_stage_manifest.json"
    ae_manifest = json.loads(ae_manifest_path.read_text(encoding="utf-8"))
    state_path = ae_root / side / "ae_gmm" / str(ae_manifest["artifact"]["path"])
    state = _load_side_ae_state(
        state_path,
        expected_side=side,
        expected_sha256=str(ae_manifest["artifact"]["sha256"]),
        raw_features=raw_features,
    )
    generated = list(_active_ae_gmm_columns(state))
    loader = SideRepresentationFeatureLoader(
        raw_loader=raw_loader,
        raw_features=raw_features,
        state=state,
        generated_features=generated,
    )
    evidence = {
        **loader_hashes,
        "ae_state_sha256": str(ae_manifest["artifact"]["sha256"]),
        "ae_manifest_sha256": _sha256(ae_manifest_path),
        "raw_candidate_features": len(raw_features),
        "generated_candidate_features": len(generated),
        "embedded_candidate_raw_matrix": matrix_evidence,
    }
    return loader, [*raw_features, *generated], evidence


def _verify_historical_representation_overlap(
    *,
    representation_root: Path,
    ae_summary_path: Path,
    ae_root: Path,
    feature_store: Path,
    guard: TrainingResourceGuard,
    rows_per_side: int = 16,
) -> dict[str, Any]:
    """Prove current exact replay reproduces a frozen historical sidecar."""

    manifest_path = representation_root / "manifest.json"
    reference_path = representation_root / "context.parquet"
    if not manifest_path.is_file() or not reference_path.is_file():
        raise RetrospectiveRepresentationError(
            "frozen-registry replay requires a historical frozen representation sidecar"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "packb_downstream_frozen_side_representation_v1"
        or manifest.get("status")
        != "MATERIALIZED_CANONICAL_CONTEXT_WITH_FROZEN_SIDE_AE_GMM"
        or manifest.get("output", {}).get("sha256") != _sha256(reference_path)
        or manifest.get("ae_gmm", {}).get("summary_sha256") != _sha256(ae_summary_path)
    ):
        raise RetrospectiveRepresentationError(
            "historical frozen representation sidecar provenance does not match the frozen AE/GMM source"
        )
    generated = list(manifest.get("representation", {}).get("generated_features", ()))
    if not FROZEN_EXECUTION_EV_GENERATED_SET.issubset(generated):
        raise RetrospectiveRepresentationError(
            "historical representation sidecar lacks required frozen outputs"
        )
    needed = [
        *IDENTITY_COLUMNS,
        "gmm_representation_available",
        *generated,
    ]
    reference = pd.read_parquet(reference_path, columns=needed)
    report: dict[str, Any] = {}
    for side in SIDES:
        sample = (
            reference.loc[
                reference["side_name"].astype(str).str.lower().eq(side)
                & reference["gmm_representation_available"].eq(1.0),
                list(IDENTITY_COLUMNS),
            ]
            .sort_values("candidate_id", kind="mergesort")
            .head(int(rows_per_side))
            .reset_index(drop=True)
        )
        if len(sample) != int(rows_per_side):
            raise RetrospectiveRepresentationError(
                f"historical representation sidecar has insufficient complete {side} overlap rows"
            )
        expected = reference.merge(
            sample.loc[:, list(IDENTITY_COLUMNS)],
            on=list(IDENTITY_COLUMNS),
            how="inner",
            validate="one_to_one",
            sort=False,
        ).sort_values("candidate_id", kind="mergesort")
        loader, candidates, evidence = _frozen_registry_replay_side_loader(
            side=side,
            ae_root=ae_root,
            feature_store=feature_store,
            guard=guard,
        )
        raw_count = int(evidence["raw_candidate_features"])
        replay_generated = list(candidates[raw_count:])
        if replay_generated != generated:
            raise RetrospectiveRepresentationError(
                f"historical {side} generated feature contract changed"
            )
        actual = loader(sample, replay_generated).assign(
            candidate_id=sample["candidate_id"].astype(str).to_numpy()
        ).sort_values("candidate_id", kind="mergesort")
        left = expected.loc[:, replay_generated].to_numpy(dtype=np.float32)
        right = actual.loc[:, replay_generated].to_numpy(dtype=np.float32)
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise RetrospectiveRepresentationError(
                f"historical {side} overlap contains non-finite representation values"
            )
        max_abs_error = float(np.max(np.abs(left.astype(np.float64) - right)))
        if max_abs_error > 1e-6:
            raise RetrospectiveRepresentationError(
                f"historical {side} frozen representation replay mismatch: {max_abs_error}"
            )
        report[side] = {
            "rows": int(len(sample)),
            "max_abs_error": max_abs_error,
            "point_feature_matrix_sha256": evidence["exact_value_hash"].get(
                "point_feature_matrix_sha256"
            ),
        }
    return {
        "root": str(representation_root),
        "manifest_sha256": _sha256(manifest_path),
        "output_sha256": _sha256(reference_path),
        "generated_feature_count": len(generated),
        "selection": f"lowest candidate_id, complete representation, {rows_per_side} rows per side",
        "tolerance_max_abs": 1e-6,
        "by_side": report,
    }


def run(
    *,
    context_root: Path | None = None,
    candidate_features: Path | None = None,
    source_manifest_path: Path | None = None,
    ae_root: Path,
    feature_store: Path,
    destination: Path,
) -> dict[str, Any]:
    """Materialize finite frozen AE/GMM outputs on a bound input ledger."""

    if destination.exists():
        raise FileExistsError(f"refusing to overwrite representation output: {destination}")
    using_context = context_root is not None
    using_candidates = candidate_features is not None or source_manifest_path is not None
    if using_context == using_candidates:
        raise ValueError(
            "provide exactly one input mode: context_root or candidate_features plus source_manifest_path"
        )
    if using_context:
        assert context_root is not None
        context, input_manifest, input_path, input_manifest_path = _validate_context(
            context_root=context_root
        )
        input_kind = "post_score_selected_context"
        output_name = "packb_forward_context_with_representation.parquet"
    else:
        if candidate_features is None or source_manifest_path is None:
            raise ValueError(
                "pre-score mode requires both candidate_features and source_manifest_path"
            )
        context, input_manifest = _validate_prescore_candidates(
            candidate_features=candidate_features,
            source_manifest_path=source_manifest_path,
        )
        input_path = candidate_features
        input_manifest_path = source_manifest_path
        input_kind = "pre_score_candidate_surface"
        output_name = "candidate_features_with_representation.parquet"
    ae_summary, ae_summary_path = _validate_ae_provenance(ae_root)
    stage = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    stage.mkdir(parents=True)
    guard = TrainingResourceGuard(
        limits=TrainingResourceLimits(
            min_free_ram_bytes=2 * 1024**3,
            max_process_rss_bytes=12 * 1024**3,
            min_free_disk_bytes=10 * 1024**3,
            check_interval_seconds=1.0,
        ),
        disk_path=destination.parent,
        telemetry_path=stage / RESOURCE_TELEMETRY_FILENAME,
    )
    side_frames: dict[str, pd.DataFrame] = {}
    generated_by_side: dict[str, list[str]] = {}
    loader_evidence: dict[str, dict[str, Any]] = {}
    try:
        guard.preflight("packb_final_refit_retrospective_representation:preflight")
        for side in SIDES:
            guard.checkpoint(f"packb_final_refit_retrospective_representation:{side}:start")
            local = context.loc[context["side_name"].eq(side)].reset_index(drop=True)
            if input_kind == "pre_score_candidate_surface":
                loader, candidates, evidence = _embedded_candidate_side_loader(
                    side=side,
                    candidate_context=context,
                    ae_root=ae_root,
                )
            else:
                loader, candidates, evidence = _side_loader(
                    side=side,
                    ae_root=ae_root,
                    feature_store=feature_store,
                    guard=guard,
                )
            raw_count = int(evidence.get("raw_candidate_features", -1))
            generated = list(map(str, candidates[raw_count:]))
            if (
                raw_count < 1
                or not generated
                or len(generated) != int(evidence.get("generated_candidate_features", -1))
                or len(set(generated)) != len(generated)
                or not FROZEN_EXECUTION_EV_GENERATED_SET.issubset(generated)
            ):
                raise RetrospectiveRepresentationError(
                    f"{side} generated representation contract is invalid or lacks a required frozen output"
                )
            generated_frame = loader(local, generated)
            if (
                list(generated_frame.columns) != generated
                or len(generated_frame) != len(local)
                or not np.isfinite(generated_frame.to_numpy(dtype=np.float32)).all()
            ):
                raise RetrospectiveRepresentationError(
                    f"{side} frozen representation has incomplete/non-finite exact coverage"
                )
            side_frames[side] = generated_frame.reset_index(drop=True)
            generated_by_side[side] = generated
            loader_evidence[side] = dict(evidence)
            guard.checkpoint(f"packb_final_refit_retrospective_representation:{side}:loaded")
            gc.collect()
        try:
            output, representation = append_side_representation(
                context,
                side_frames=side_frames,
                generated_features_by_side=generated_by_side,
                minimum_joint_finite_fraction=1.0,
                minimum_monthly_joint_finite_fraction=1.0,
            )
        except DownstreamRepresentationError as exc:
            raise RetrospectiveRepresentationError(str(exc)) from exc
        generated_features = list(representation["generated_features"])
        if (
            len(output) != len(context)
            or candidate_identity_sha256(output, columns=IDENTITY_COLUMNS)
            != candidate_identity_sha256(context, columns=IDENTITY_COLUMNS)
            or not np.isfinite(output.loc[:, generated_features].to_numpy(dtype=np.float32)).all()
            or not output["gmm_representation_available"].eq(1.0).all()
            or not set(FROZEN_EXECUTION_EV_REPRESENTATION).issubset(output.columns)
        ):
            raise RetrospectiveRepresentationError(
                "representation append changed identity or did not have complete finite coverage"
            )
        guard.checkpoint("packb_final_refit_retrospective_representation:write")
        output_path = stage / output_name
        output.to_parquet(output_path, index=False, compression="zstd", compression_level=5)
        source_revision, source_worktree_clean = _source_revision()
        result = {
            "schema": SCHEMA,
            "status": "materialized_retrospective_final_refit_context_with_frozen_side_ae_gmm_non_promotable",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_revision": source_revision,
            "source_worktree_clean": source_worktree_clean,
            "outcomes_used": False,
            "promotion_status": "retrospective_final_refit_input_not_oos_metrics",
            "input": {
                "kind": input_kind,
                "path": str(input_path),
                "sha256": _sha256(input_path),
                "manifest_path": str(input_manifest_path),
                "manifest_sha256": _sha256(input_manifest_path),
                "source_status": input_manifest["status"],
                "candidate_identity_sha256": candidate_identity_sha256(
                    context, columns=IDENTITY_COLUMNS
                ),
            },
            "ae_gmm": {
                "root": str(ae_root),
                "summary_sha256": _sha256(ae_summary_path),
                "source_revision": ae_summary.get("source_revision"),
                "side_local_states_required": True,
                "loader_evidence_by_side": loader_evidence,
                "accepted_input_policy": (
                    "pre-score: complete finite frozen raw AE contract embedded in "
                    "the manifest-bound candidate ledger"
                    if input_kind == "pre_score_candidate_surface"
                    else "post-score retrospective context only"
                ),
            },
            "feature_store": {"path": str(feature_store), "immutable_point_lookup": True},
            "representation": representation,
            "output": {
                "path": str(destination / output_path.name),
                "sha256": _sha256(output_path),
                "rows": int(len(output)),
                "columns": int(len(output.columns)),
                "candidate_identity_sha256": candidate_identity_sha256(
                    output, columns=IDENTITY_COLUMNS
                ),
            },
            "resource_guard": {
                "max_process_rss_bytes": guard.limits.max_process_rss_bytes,
                "min_free_ram_bytes": guard.limits.min_free_ram_bytes,
                "min_free_disk_bytes": guard.limits.min_free_disk_bytes,
                "telemetry": str(destination / RESOURCE_TELEMETRY_FILENAME),
            },
            "outcome_columns_added": [],
        }
        _write_json(stage / "manifest.json", result)
        guard.checkpoint("packb_final_refit_retrospective_representation:complete")
        os.replace(stage, destination)
        return result
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _parser(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--context-root",
        type=Path,
        help="Post-score Pack-B directory containing packb_forward_context.parquet and manifest.json.",
    )
    inputs.add_argument(
        "--candidate-root",
        type=Path,
        help="Pre-score candidate directory containing candidate_features.parquet and source_manifest.json.",
    )
    parser.add_argument("--candidate-features", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--ae-root", type=Path, default=DEFAULT_AE_ROOT)
    parser.add_argument("--feature-store", type=Path, default=DEFAULT_FEATURE_STORE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser(argv)
    if args.candidate_root is not None:
        candidate_features = args.candidate_root / "candidate_features.parquet"
        source_manifest_path = args.candidate_root / "source_manifest.json"
    else:
        candidate_features = args.candidate_features
        source_manifest_path = args.source_manifest
    if args.context_root is None and (
        candidate_features is None or source_manifest_path is None
    ):
        raise ValueError(
            "pre-score CLI mode requires --candidate-root or both --candidate-features and --source-manifest"
        )
    print(
        json.dumps(
            _jsonable(
                run(
                    context_root=args.context_root,
                    candidate_features=candidate_features,
                    source_manifest_path=source_manifest_path,
                    ae_root=args.ae_root,
                    feature_store=args.feature_store,
                    destination=args.output_dir,
                )
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
