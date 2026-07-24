"""Fail-closed, side-local AE/GMM reference-state preparation for Pack-B.

The component is deliberately a small post-preflight stage rather than a
production feature-store runner.  It receives a single already-authorized
side ledger and a caller-provided feature loader.  Before the loader is called
it proves that every ledger row belongs to that side and that every reference
row is in the locked Jan--Oct 2025 signal interval with the actual Pack-B
decision and label-resolution timing.  It then freezes one side-local,
outcome-free AE/GMM state and emits immutable stage evidence.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from extreme_price_movements import packb_side_stage_manifest as stage_manifest
from extreme_price_movements.features_gmm_ae import (
    ae_gmm_cycle_reference_indices,
    ae_gmm_cycle_sample_identity_hash,
    ae_gmm_input_feature_order_hash,
    ae_gmm_learned_transform_hash,
    fit_ae_gmm_state,
)
from extreme_price_movements.training_resource_guard import TrainingResourceGuard

AE_STAGE_SCHEMA = "packb_side_local_ae_stage_v1"
AE_REFERENCE_START_UTC = pd.Timestamp("2025-01-01T00:00:00Z")
AE_REFERENCE_END_UTC = pd.Timestamp("2025-11-01T00:00:00Z")
RESOLUTION_CUTOFF_UTC = pd.Timestamp("2026-03-01T00:00:00Z")
DECISION_LAG = pd.Timedelta(hours=1)
LABEL_RESOLUTION_HORIZON = pd.Timedelta(hours=24)
CANONICAL_SIDES = ("long", "short")
REQUIRED_LEDGER_COLUMNS = (
    "candidate_id",
    "side_name",
    "__ts__",
    "__decision_ts__",
    "__label_resolution_ts__",
    "__symbol__",
)

FeatureLoader = Callable[[pd.DataFrame, Sequence[str]], pd.DataFrame]


class PackBSideLocalAEStageError(ValueError):
    """Raised when side-local AE/GMM preparation cannot prove its contract."""


def _utc_series(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if values.isna().any():
        raise PackBSideLocalAEStageError(
            f"ledger column {column!r} has invalid UTC timestamps"
        )
    return values


def _require_sha256(value: Any, *, name: str) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 64 or any(
        char not in "0123456789abcdef" for char in normalized
    ):
        raise PackBSideLocalAEStageError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


def _validate_source_evidence(
    *,
    source_hashes: Mapping[str, str],
    source_revision: str,
    fixed_calendar_sha256: str,
) -> tuple[dict[str, str], str, str]:
    required = (
        "dec09_decisions_sha256",
        "canonical_shard_inventory_sha256",
        "causal_audit_sha256",
        "population_preflight_sha256",
        "authorized_population_ledger_sha256",
        "feature_store_inventory_sha256",
        "feature_store_inventory_evidence_sha256",
    )
    if set(source_hashes) != set(required):
        raise PackBSideLocalAEStageError(
            "source_hashes must contain exactly the locked stage-manifest source hashes"
        )
    normalized = {
        key: _require_sha256(source_hashes[key], name=f"source_hashes.{key}")
        for key in required
    }
    revision = str(source_revision or "").strip().lower()
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise PackBSideLocalAEStageError(
            "source_revision must be a 40-character Git SHA"
        )
    return (
        normalized,
        revision,
        _require_sha256(fixed_calendar_sha256, name="fixed_calendar_sha256"),
    )


def _validate_bound_ledger(
    path: Path,
    *,
    name: str,
    expected_sha256: str | None = None,
) -> tuple[Path, str]:
    ledger_path = Path(path)
    if not ledger_path.is_file():
        raise PackBSideLocalAEStageError(f"{name} does not exist: {ledger_path}")
    actual = stage_manifest.sha256_file(ledger_path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise PackBSideLocalAEStageError(
            f"{name} SHA-256 does not match its authorized source hash"
        )
    return ledger_path, actual


def _validate_input_features(input_features: Sequence[str]) -> list[str]:
    features = [str(value) for value in input_features]
    if len(features) < 2 or len(set(features)) != len(features):
        raise PackBSideLocalAEStageError(
            "AE/GMM input_features must contain at least two unique columns"
        )
    forbidden_exact = set(REQUIRED_LEDGER_COLUMNS) | {"side", "__side__"}
    forbidden_tokens = (
        "target",
        "label",
        "future",
        "outcome",
        "first_touch",
        "full_path",
        "realized",
        "pnl",
    )
    blocked = [
        feature
        for feature in features
        if feature in forbidden_exact
        or feature.lower().startswith("side_")
        or any(token in feature.lower() for token in forbidden_tokens)
    ]
    if blocked:
        raise PackBSideLocalAEStageError(
            "AE/GMM input features contain identity, side, or outcome-derived columns: "
            + ", ".join(blocked[:8])
        )
    return features


def _validate_and_select_reference(
    cohort_ledger: pd.DataFrame, *, side: str
) -> pd.DataFrame:
    if not isinstance(cohort_ledger, pd.DataFrame):
        raise PackBSideLocalAEStageError("cohort_ledger must be a pandas DataFrame")
    missing = sorted(set(REQUIRED_LEDGER_COLUMNS) - set(cohort_ledger.columns))
    if missing:
        raise PackBSideLocalAEStageError(
            "cohort ledger misses required columns: " + ", ".join(missing)
        )
    if cohort_ledger.empty:
        raise PackBSideLocalAEStageError("authorized side cohort ledger is empty")
    ledger = cohort_ledger.loc[:, list(REQUIRED_LEDGER_COLUMNS)].copy()
    candidate_ids = ledger["candidate_id"].astype("string")
    invalid_id = (
        candidate_ids.isna()
        | candidate_ids.str.strip().eq("")
        | candidate_ids.ne(candidate_ids.str.strip())
        | candidate_ids.duplicated(keep=False)
    )
    if invalid_id.any():
        raise PackBSideLocalAEStageError(
            "authorized side cohort has null, malformed, or duplicate candidate_id"
        )
    sides = ledger["side_name"].astype("string").str.strip().str.lower()
    if sides.isna().any() or not sides.eq(side).all():
        raise PackBSideLocalAEStageError(
            f"authorized side cohort must contain exactly {side!r} rows"
        )
    signal = _utc_series(ledger, "__ts__")
    decision = _utc_series(ledger, "__decision_ts__")
    resolution = _utc_series(ledger, "__label_resolution_ts__")
    if not decision.eq(signal + DECISION_LAG).all():
        raise PackBSideLocalAEStageError(
            "authorized side cohort violates decision_timestamp = signal_timestamp + 1h"
        )
    if not resolution.eq(decision + LABEL_RESOLUTION_HORIZON).all():
        raise PackBSideLocalAEStageError(
            "authorized side cohort violates label_resolution = decision_timestamp + 24h"
        )
    if not resolution.lt(RESOLUTION_CUTOFF_UTC).all():
        raise PackBSideLocalAEStageError(
            "authorized side cohort contains a label resolved at/after the pre-March cutoff"
        )
    reference = ledger.loc[
        signal.ge(AE_REFERENCE_START_UTC)
        & signal.lt(AE_REFERENCE_END_UTC)
        & resolution.lt(AE_REFERENCE_END_UTC)
    ].copy()
    if reference.empty:
        raise PackBSideLocalAEStageError(
            "authorized side cohort has no rows in the locked pre-Nov AE reference interval"
        )
    return reference


def _candidate_stream_evidence(frame: pd.DataFrame) -> dict[str, Any]:
    ordered = frame.copy()
    signal = _utc_series(ordered, "__ts__")
    decision = _utc_series(ordered, "__decision_ts__")
    resolution = _utc_series(ordered, "__label_resolution_ts__")
    ordered["__ts__"] = signal
    ordered["__decision_ts__"] = decision
    ordered["__label_resolution_ts__"] = resolution
    ordered = ordered.sort_values(
        ["__ts__", "__symbol__", "candidate_id"], kind="mergesort"
    ).reset_index(drop=True)
    digest = hashlib.sha256()
    for candidate_id, side_name, symbol, ts, decision_ts, resolution_ts in zip(
        ordered["candidate_id"],
        ordered["side_name"],
        ordered["__symbol__"],
        ordered["__ts__"],
        ordered["__decision_ts__"],
        ordered["__label_resolution_ts__"],
        strict=True,
    ):
        digest.update(
            (
                f"{candidate_id}\x1f{side_name}\x1f{symbol}\x1f"
                f"{pd.Timestamp(ts).isoformat()}\x1f"
                f"{pd.Timestamp(decision_ts).isoformat()}\x1f"
                f"{pd.Timestamp(resolution_ts).isoformat()}\n"
            ).encode("utf-8")
        )
    return {
        "count": int(len(ordered)),
        "sha256": digest.hexdigest(),
        "signal_min_utc": signal.min().isoformat(),
        "signal_max_utc": signal.max().isoformat(),
        "decision_min_utc": decision.min().isoformat(),
        "decision_max_utc": decision.max().isoformat(),
        "label_resolution_min_utc": resolution.min().isoformat(),
        "label_resolution_max_utc": resolution.max().isoformat(),
    }


def _ledger_identity_sha256(frame: pd.DataFrame) -> str:
    """Hash the identity/timing fields that bind an in-memory cohort to Parquet."""

    return str(_candidate_stream_evidence(frame)["sha256"])


def _validate_feature_loader_binding(
    *,
    feature_loader: FeatureLoader,
    input_features: Sequence[str],
    source_revision: str,
) -> tuple[dict[str, Any], Callable[[pd.DataFrame, pd.DataFrame], str]]:
    """Require the canonical bounded point loader and immutable evidence."""

    evidence = getattr(feature_loader, "packb_static_feature_loader_evidence", None)
    contract = getattr(feature_loader, "packb_static_feature_contract", None)
    matrix_hasher = getattr(feature_loader, "packb_static_feature_matrix_sha256", None)
    if not isinstance(evidence, Mapping) or not isinstance(contract, Mapping):
        raise PackBSideLocalAEStageError(
            "feature_loader must expose the frozen Pack-B static-loader evidence contract"
        )
    if not callable(matrix_hasher):
        raise PackBSideLocalAEStageError(
            "feature_loader must expose the canonical Pack-B feature-matrix hasher"
        )
    contract_features = contract.get("feature_columns")
    if not isinstance(contract_features, list) or list(input_features) != [
        str(value) for value in contract_features
    ]:
        raise PackBSideLocalAEStageError(
            "AE/GMM input_features must equal the complete frozen static feature contract"
        )
    required_hashes = (
        "raw_universe_sha256",
        "coverage_profile_sha256",
        "feature_contract_sha256",
        "loader_contract_sha256",
        "loader_module_sha256",
        "source_schema_sha256",
    )
    normalized: dict[str, Any] = {
        key: _require_sha256(evidence.get(key), name=f"feature_loader_evidence.{key}")
        for key in required_hashes
    }
    contract_sha256 = _require_sha256(
        contract.get("feature_contract_sha256"),
        name="feature_loader_contract.feature_contract_sha256",
    )
    if normalized["feature_contract_sha256"] != contract_sha256:
        raise PackBSideLocalAEStageError(
            "feature-loader evidence does not bind its frozen feature contract"
        )
    loader_revision = str(evidence.get("source_revision") or "").strip().lower()
    if loader_revision != source_revision:
        raise PackBSideLocalAEStageError(
            "feature-loader source revision does not match the AE stage source revision"
        )
    evidence_path = Path(str(evidence.get("evidence_path") or ""))
    if not evidence_path.is_file():
        raise PackBSideLocalAEStageError(
            "feature-loader immutable evidence file does not exist"
        )
    try:
        persisted = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackBSideLocalAEStageError(
            "feature-loader immutable evidence file is unreadable"
        ) from exc
    for key in required_hashes:
        if persisted.get(key) != normalized[key]:
            raise PackBSideLocalAEStageError(
                f"feature-loader immutable evidence disagrees on {key}"
            )
    if persisted.get("source_revision") != loader_revision:
        raise PackBSideLocalAEStageError(
            "feature-loader immutable evidence has a different source revision"
        )
    normalized.update(
        {
            "source_revision": loader_revision,
            "evidence_path": str(evidence_path),
            "evidence_file_sha256": stage_manifest.sha256_file(evidence_path),
            "requested_feature_policy": str(
                evidence.get("requested_feature_policy") or ""
            ),
        }
    )
    return normalized, matrix_hasher


def _load_reference_matrix(
    *,
    feature_loader: FeatureLoader,
    sampled_ledger: pd.DataFrame,
    input_features: Sequence[str],
    matrix_hasher: Callable[[pd.DataFrame, pd.DataFrame], str],
) -> tuple[pd.DataFrame, dict[str, float], str]:
    loaded = feature_loader(sampled_ledger.copy(), list(input_features))
    if not isinstance(loaded, pd.DataFrame):
        raise PackBSideLocalAEStageError(
            "feature_loader must return a pandas DataFrame"
        )
    if len(loaded) != len(sampled_ledger):
        raise PackBSideLocalAEStageError(
            "feature_loader row count does not match the sampled authorized ledger"
        )
    if list(loaded.columns) != list(input_features):
        raise PackBSideLocalAEStageError(
            "feature_loader must return exactly the ordered AE/GMM input feature columns"
        )
    matrix = loaded.replace([np.inf, -np.inf], np.nan)
    try:
        matrix = matrix.astype(np.float32, copy=False)
    except (TypeError, ValueError) as exc:
        raise PackBSideLocalAEStageError(
            "feature_loader returned non-numeric AE/GMM input values"
        ) from exc
    matrix_sha256 = _require_sha256(
        matrix_hasher(sampled_ledger.copy(), matrix.copy()),
        name="feature_loader matrix SHA-256",
    )
    fill_values = matrix.median(numeric_only=True).reindex(input_features).fillna(0.0)
    matrix = matrix.fillna(fill_values).astype(np.float32, copy=False)
    values = matrix.to_numpy(dtype=np.float32, copy=False)
    if not np.isfinite(values).all():
        raise PackBSideLocalAEStageError(
            "AE/GMM matrix remains non-finite after frozen fill"
        )
    return (
        matrix,
        {name: float(fill_values[name]) for name in input_features},
        matrix_sha256,
    )


def _write_pickle_once(path: Path, value: Any) -> None:
    if path.exists():
        raise PackBSideLocalAEStageError(
            f"refusing to overwrite frozen AE/GMM state: {path}"
        )
    with path.open("xb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise PackBSideLocalAEStageError(
            f"refusing to overwrite immutable AE stage evidence: {path}"
        )
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")


def _write_parquet_once(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise PackBSideLocalAEStageError(
            f"refusing to overwrite immutable AE candidate evidence: {path}"
        )
    frame.to_parquet(path, index=False)


def fit_side_local_ae_gmm_stage(
    *,
    side: str,
    cohort_ledger: pd.DataFrame,
    cohort_ledger_path: Path,
    authorized_population_ledger_path: Path,
    feature_loader: FeatureLoader,
    input_features: Sequence[str],
    output_dir: Path,
    source_hashes: Mapping[str, str],
    source_revision: str,
    fixed_calendar_sha256: str,
    seed: int = 41,
    max_train_rows: int = 50_000,
    gmm_max_train_rows: int = 50_000,
    ae_max_iter: int = 80,
    min_reference_rows: int = 200,
    resource_guard: TrainingResourceGuard | Any | None = None,
) -> dict[str, Any]:
    """Fit one frozen outcome-free AE/GMM state from one side-local cohort.

    ``feature_loader`` sees only the deterministically sampled pre-Nov identity
    ledger.  It must return exactly ``input_features`` in that order.  The
    caller must supply hashes from the already-authorized DEC-09 population;
    no labels or production feature-store paths are opened by this component.
    """

    normalized_side = str(side or "").strip().lower()
    if normalized_side not in CANONICAL_SIDES:
        raise PackBSideLocalAEStageError("side must be exactly long or short")
    features = _validate_input_features(input_features)
    source_evidence, revision, calendar_hash = _validate_source_evidence(
        source_hashes=source_hashes,
        source_revision=source_revision,
        fixed_calendar_sha256=fixed_calendar_sha256,
    )
    if int(max_train_rows) < 1 or int(gmm_max_train_rows) < 1:
        raise PackBSideLocalAEStageError("AE/GMM row caps must be positive")
    if int(min_reference_rows) < 2:
        raise PackBSideLocalAEStageError("min_reference_rows must be at least two")
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise PackBSideLocalAEStageError(
            f"output directory must be empty: {destination}"
        )
    guard = resource_guard or TrainingResourceGuard(disk_path=destination.parent)
    guard.preflight(f"packb_side_local_ae:{normalized_side}:preflight")

    cohort_path, cohort_sha256 = _validate_bound_ledger(
        cohort_ledger_path,
        name="authorized side cohort ledger",
    )
    population_path, population_sha256 = _validate_bound_ledger(
        authorized_population_ledger_path,
        name="authorized population ledger",
        expected_sha256=source_evidence["authorized_population_ledger_sha256"],
    )
    try:
        cohort_on_disk = pd.read_parquet(
            cohort_path, columns=list(REQUIRED_LEDGER_COLUMNS)
        )
    except Exception as exc:
        raise PackBSideLocalAEStageError(
            "cannot read the bound authorized side cohort ledger"
        ) from exc
    if _ledger_identity_sha256(cohort_on_disk) != _ledger_identity_sha256(
        cohort_ledger
    ):
        raise PackBSideLocalAEStageError(
            "in-memory cohort ledger does not match its bound cohort ledger file"
        )
    reference = _validate_and_select_reference(cohort_ledger, side=normalized_side)
    if len(reference) < int(min_reference_rows):
        raise PackBSideLocalAEStageError(
            f"pre-Nov {normalized_side} AE reference has only {len(reference)} rows; "
            f"requires {int(min_reference_rows)}"
        )
    guard.checkpoint(f"packb_side_local_ae:{normalized_side}:authorized_reference")
    reference_cap = max(int(max_train_rows), int(gmm_max_train_rows))
    sampled_positions = ae_gmm_cycle_reference_indices(
        reference["__ts__"],
        symbols=reference["__symbol__"],
        sides=reference["side_name"],
        max_rows=reference_cap,
    )
    if len(sampled_positions) < int(min_reference_rows):
        raise PackBSideLocalAEStageError(
            "deterministic AE/GMM sample is below minimum support"
        )
    sampled = reference.iloc[sampled_positions].reset_index(drop=True)
    loader_evidence, matrix_hasher = _validate_feature_loader_binding(
        feature_loader=feature_loader,
        input_features=features,
        source_revision=revision,
    )
    guard.checkpoint(f"packb_side_local_ae:{normalized_side}:before_feature_load")
    matrix, fill_values, feature_matrix_sha256 = _load_reference_matrix(
        feature_loader=feature_loader,
        sampled_ledger=sampled,
        input_features=features,
        matrix_hasher=matrix_hasher,
    )
    guard.checkpoint(f"packb_side_local_ae:{normalized_side}:before_fit")
    state = fit_ae_gmm_state(
        matrix,
        timestamps=sampled["__ts__"],
        economic_targets={},
        random_state=int(seed),
        max_train_rows=int(max_train_rows),
        gmm_max_train_rows=int(gmm_max_train_rows),
        ae_max_iter=int(ae_max_iter),
        require_both_sides=False,
        path_aware_hpo=False,
        temporal_concentration_hpo=False,
        temporal_stability_hpo=False,
        smooth_lambda_candidates=(0.0,),
        final_refit_all_rows=False,
        enhanced_search=False,
        outcome_free=True,
        temporal_feature_contract="row_independent_v1",
    )
    if not bool(state.get("enabled", False)):
        raise PackBSideLocalAEStageError(
            "side-local AE/GMM state is disabled: "
            + str(state.get("reason", "unknown"))
        )
    sample_hash = ae_gmm_cycle_sample_identity_hash(
        sampled["__ts__"],
        symbols=sampled["__symbol__"],
        sides=sampled["side_name"],
    )
    input_hash = ae_gmm_input_feature_order_hash(features)
    reference_stream = _candidate_stream_evidence(reference)
    sampled_stream = _candidate_stream_evidence(sampled)
    stage_config = {
        "schema": AE_STAGE_SCHEMA,
        "side": normalized_side,
        "reference_signal_interval": [
            AE_REFERENCE_START_UTC.isoformat(),
            AE_REFERENCE_END_UTC.isoformat(),
        ],
        "reference_label_resolution_end_exclusive": AE_REFERENCE_END_UTC.isoformat(),
        "input_feature_order_sha256": input_hash,
        "feature_contract_sha256": loader_evidence["feature_contract_sha256"],
        "feature_matrix_sha256": feature_matrix_sha256,
        "feature_loader_evidence": loader_evidence,
        "seed": int(seed),
        "max_train_rows": int(max_train_rows),
        "gmm_max_train_rows": int(gmm_max_train_rows),
        "ae_max_iter": int(ae_max_iter),
        "sample_policy": "canonical_timestamp_symbol_side_beginning_middle_end",
        "require_both_sides": False,
        "economic_targets": [],
        "outcome_free": True,
        "path_aware_hpo": False,
        "temporal_concentration_hpo": False,
        "temporal_stability_hpo": False,
        "temporal_feature_contract": "row_independent_v1",
    }
    stage_config_serialized = json.dumps(stage_config, sort_keys=True, indent=2) + "\n"
    stage_config_sha256 = hashlib.sha256(
        stage_config_serialized.encode("utf-8")
    ).hexdigest()
    state.update(
        {
            "packb_side_local_ae_stage_schema": AE_STAGE_SCHEMA,
            "packb_side_scope": normalized_side,
            "cycle_reference_signal_start": AE_REFERENCE_START_UTC.isoformat(),
            "cycle_reference_signal_end_exclusive": AE_REFERENCE_END_UTC.isoformat(),
            "cycle_reference_rows_available": int(len(reference)),
            "cycle_reference_rows_sampled": int(len(sampled)),
            "cycle_reference_sample_policy": stage_config["sample_policy"],
            "cycle_reference_sample_identity_hash": sample_hash,
            "cycle_reference_candidate_stream": reference_stream,
            "cycle_reference_sampled_candidate_stream": sampled_stream,
            "cycle_input_fill_values": fill_values,
            "cycle_input_matrix_sha256": feature_matrix_sha256,
            "feature_loader_evidence": loader_evidence,
            "input_feature_order_hash": input_hash,
            "representation_selection_outcome_free": True,
            "representation_selection_context_keys": [],
            "representation_selection_outcome_keys": [],
            "stage_config_sha256": stage_config_sha256,
        }
    )
    state["cycle_state_hash"] = ae_gmm_learned_transform_hash(state)
    guard.checkpoint(f"packb_side_local_ae:{normalized_side}:before_persist")
    if not destination.exists():
        destination.mkdir(parents=True, exist_ok=False)
    candidate_stream_path = destination / "reference_candidates.parquet"
    _write_parquet_once(
        candidate_stream_path, sampled.loc[:, list(REQUIRED_LEDGER_COLUMNS)]
    )
    candidate_stream_sha256 = stage_manifest.sha256_file(candidate_stream_path)
    stage_config_path = destination / "stage_config.json"
    _write_json_once(stage_config_path, stage_config)
    if stage_manifest.sha256_file(stage_config_path) != stage_config_sha256:
        raise PackBSideLocalAEStageError("persisted AE stage config hash is unstable")
    state_path = destination / "ae_gmm_state.pkl"
    _write_pickle_once(state_path, state)
    state_sha256 = stage_manifest.sha256_file(state_path)
    metadata = {
        "schema": AE_STAGE_SCHEMA,
        "side": normalized_side,
        "state_path": str(state_path),
        "state_sha256": state_sha256,
        "cycle_state_hash": str(state["cycle_state_hash"]),
        "input_feature_order_sha256": input_hash,
        "feature_contract_sha256": loader_evidence["feature_contract_sha256"],
        "feature_matrix_sha256": feature_matrix_sha256,
        "feature_loader_evidence": loader_evidence,
        "cycle_input_fill_values": fill_values,
        "cycle_reference_sample_identity_sha256": sample_hash,
        "cycle_reference_candidate_stream": reference_stream,
        "cycle_reference_sampled_candidate_stream": sampled_stream,
        "authorized_side_cohort_ledger": {
            "path": str(cohort_path),
            "sha256": cohort_sha256,
            "identity_sha256": _ledger_identity_sha256(cohort_ledger),
        },
        "authorized_population_ledger": {
            "path": str(population_path),
            "sha256": population_sha256,
        },
        "candidate_stream_evidence": {
            "path": str(candidate_stream_path),
            "sha256": candidate_stream_sha256,
        },
        "source_hashes": source_evidence,
        "source_revision": revision,
        "fixed_calendar_sha256": calendar_hash,
        "stage_config_path": str(stage_config_path),
        "stage_config_sha256": str(state["stage_config_sha256"]),
    }
    metadata_path = destination / "ae_gmm_state_metadata.json"
    _write_json_once(metadata_path, metadata)
    manifest_payload = {
        "schema": stage_manifest.SIDE_STAGE_MANIFEST_SCHEMA,
        "source_revision": revision,
        "side": normalized_side,
        "stage": "ae_gmm",
        "resolution_cutoff_utc": RESOLUTION_CUTOFF_UTC.isoformat(),
        "actual_label_resolution_contract": stage_manifest.ACTUAL_LABEL_RESOLUTION_CONTRACT,
        "source_hashes": source_evidence,
        "authorized_population_ledger": {
            "path": str(population_path),
            "sha256": population_sha256,
        },
        "candidate_stream": {
            "path": candidate_stream_path.name,
            **sampled_stream,
            "sha256": candidate_stream_sha256,
        },
        "fixed_calendar_sha256": calendar_hash,
        "stage_config": {
            "path": stage_config_path.name,
            "sha256": stage_config_sha256,
        },
        "artifact": {
            "kind": "ae_gmm_state",
            "path": state_path.name,
            "sha256": state_sha256,
            "scope": normalized_side,
        },
    }
    manifest_path = destination / "side_stage_manifest.json"
    manifest_sha256 = stage_manifest.write_immutable_side_stage_manifest(
        manifest_path, manifest_payload
    )
    guard.checkpoint(f"packb_side_local_ae:{normalized_side}:complete")
    return {
        "schema": AE_STAGE_SCHEMA,
        "status": "FROZEN_SIDE_LOCAL_AE_GMM_STATE",
        "side": normalized_side,
        "state_path": str(state_path),
        "state_sha256": state_sha256,
        "metadata_path": str(metadata_path),
        "side_stage_manifest_path": str(manifest_path),
        "side_stage_manifest_sha256": manifest_sha256,
        "cycle_state_hash": str(state["cycle_state_hash"]),
        "reference_rows_available": int(len(reference)),
        "reference_rows_sampled": int(len(sampled)),
        "sample_identity_sha256": sample_hash,
        "source_candidate_stream_sha256": reference_stream["sha256"],
        "sampled_candidate_stream_sha256": sampled_stream["sha256"],
        "candidate_stream_evidence_path": str(candidate_stream_path),
        "candidate_stream_evidence_sha256": candidate_stream_sha256,
        "input_feature_order_sha256": input_hash,
        "feature_contract_sha256": loader_evidence["feature_contract_sha256"],
        "feature_matrix_sha256": feature_matrix_sha256,
        "feature_loader_evidence_sha256": loader_evidence["evidence_file_sha256"],
        "stage_config_sha256": str(state["stage_config_sha256"]),
    }
