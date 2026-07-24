"""Fail-closed, fixed-calendar side-local Pack-B feature selection and HPO.

This module is an orchestration boundary, not a generic model trainer.  It
consumes *already materialized* DEC-09 side-local cohort ledgers and caller
supplied feature/target/weight/model callbacks.  The boundary does four things
that the callbacks are not allowed to weaken:

* prove the long/short ledger, decision, resolution, purge, and fixed-calendar
  contracts before a feature, target, or weight callback is reached;
* require a real side-local MDA-containing feature-selection result and an
  explicit parameter choice from a multi-arm, three-fold chronological search;
* call the resource guard before every material load, HPO trial/fold, and
  publication boundary; and
* publish immutable Parquet candidate evidence, JSON configs/artifacts, and
  current ``packb_side_stage_manifest`` records only after all computation
  succeeds.

There is deliberately no production feature-store loader here.  A bounded,
causal loader belongs at the caller boundary; keeping it a callback makes the
calendar and provenance checks testable without accidentally loading the full
feature store into memory.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from extreme_price_movements import packb_side_stage_manifest as stage_manifest
from extreme_price_movements.training_resource_guard import TrainingResourceGuard

FS_HPO_STAGE_SCHEMA = "packb_side_local_fs_hpo_stage_v1"
CANONICAL_SIDES = ("long", "short")
REQUIRED_LEDGER_COLUMNS = (
    "candidate_id",
    "side_name",
    "__ts__",
    "__decision_ts__",
    "__label_resolution_ts__",
    "__symbol__",
)
DECISION_LAG = pd.Timedelta(hours=1)
LABEL_RESOLUTION_HORIZON = pd.Timedelta(hours=24)
SIGNAL_PURGE = pd.Timedelta(hours=25)
RESOLUTION_CUTOFF_UTC = pd.Timestamp("2026-03-01T00:00:00Z")
FS_VALIDATION = (
    pd.Timestamp("2025-11-01T00:00:00Z"),
    pd.Timestamp("2025-12-01T00:00:00Z"),
)
HPO_VALIDATIONS = (
    (pd.Timestamp("2025-12-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-02-01T00:00:00Z")),
    (pd.Timestamp("2026-02-01T00:00:00Z"), pd.Timestamp("2026-03-01T00:00:00Z")),
)
_SOURCE_HASH_FIELDS = (
    "dec09_decisions_sha256",
    "canonical_shard_inventory_sha256",
    "causal_audit_sha256",
    "population_preflight_sha256",
    "authorized_population_ledger_sha256",
    "feature_store_inventory_sha256",
    "feature_store_inventory_evidence_sha256",
)
_FORBIDDEN_FEATURE_TOKENS = (
    "target",
    "label",
    "future",
    "outcome",
    "realized",
    "first_touch",
    "full_path",
    "pnl",
)
MIN_PER_FEATURE_FINITE_FRACTION = 0.98
MIN_JOINT_COMPLETE_FRACTION = 0.95
DEFAULT_FS_TRAIN_MAX_ROWS = 60_000
DEFAULT_FS_VALID_MAX_ROWS = 20_000
DEFAULT_HPO_TRAIN_MAX_ROWS = 10_000
DEFAULT_HPO_VALID_MAX_ROWS = 10_000


class PackBSideLocalFSHPOStageError(ValueError):
    """Raised when the fixed side-local FS/HPO contract cannot be proved."""


@dataclass(frozen=True)
class HPOFoldLedger:
    """One named, side-local fixed-calendar HPO train/validation pair."""

    name: str
    train_ledger: pd.DataFrame
    train_ledger_path: Path
    valid_ledger: pd.DataFrame
    valid_ledger_path: Path


@dataclass(frozen=True)
class HPOTrial:
    """One explicit HPO arm.  Parameters must be side-local and non-empty."""

    trial_id: str
    params: Mapping[str, Any]


@dataclass(frozen=True)
class StageDataset:
    """One already-filtered callback input; no unfiltered ledger is exposed."""

    ledger: pd.DataFrame
    features: pd.DataFrame
    target: pd.Series
    weights: pd.Series


@dataclass(frozen=True)
class FeatureSelectionInput:
    """The exact November train/validation datasets passed to the selector."""

    side: str
    candidate_features: tuple[str, ...]
    train: StageDataset
    validation: StageDataset


@dataclass(frozen=True)
class HPOFoldInput:
    """One trial on one chronological fold, loaded only for this invocation."""

    side: str
    fold_name: str
    validation_start_utc: str
    validation_end_utc: str
    trial: HPOTrial
    selected_features: tuple[str, ...]
    train: StageDataset
    validation: StageDataset


@dataclass(frozen=True)
class HPOTrialEvaluation:
    """A normalised evaluator result retained in the final HPO artifact."""

    trial_id: str
    params: Mapping[str, Any]
    fold_name: str
    result: Mapping[str, Any]


FeatureLoader = Callable[[pd.DataFrame, Sequence[str]], pd.DataFrame]
TargetLoader = Callable[[pd.DataFrame], Any]
WeightLoader = Callable[[pd.DataFrame, pd.Series], Any]
FeatureSelectionCallback = Callable[[FeatureSelectionInput], Mapping[str, Any]]
HPOTrialEvaluator = Callable[[HPOFoldInput], Any]
HPOSelectionCallback = Callable[[Sequence[HPOTrialEvaluation]], Mapping[str, Any]]


def locked_calendar() -> dict[str, Any]:
    """Return the literal DEC-09 calendar bound into both published stages."""

    return {
        "resolution_cutoff_utc": RESOLUTION_CUTOFF_UTC.isoformat(),
        "decision_contract": "__decision_ts__ = __ts__ + 1h",
        "label_resolution_contract": "__label_resolution_ts__ = __decision_ts__ + 24h",
        "train_rule": (
            "__ts__ < validation_start - 25h AND "
            "__label_resolution_ts__ < validation_start"
        ),
        "validation_rule": (
            "validation_start <= __ts__ < validation_end AND "
            "__label_resolution_ts__ < 2026-03-01T00:00:00+00:00"
        ),
        "feature_selection_validation": [item.isoformat() for item in FS_VALIDATION],
        "hpo_validations": [
            [item.isoformat() for item in interval] for interval in HPO_VALIDATIONS
        ],
        "fallback": "FORBIDDEN_NO_GLOBAL_POOLED_UNIVARIATE_OR_DEFAULT_FALLBACK",
    }


def _utc_series(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if values.isna().any():
        raise PackBSideLocalFSHPOStageError(
            f"ledger column {column!r} has invalid UTC timestamps"
        )
    return values


def _require_sha256(value: Any, *, name: str) -> str:
    normalised = str(value or "").strip().lower()
    if len(normalised) != 64 or any(
        char not in "0123456789abcdef" for char in normalised
    ):
        raise PackBSideLocalFSHPOStageError(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return normalised


def _canonical_json(value: Any, *, name: str) -> str:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
    except (TypeError, ValueError) as exc:
        raise PackBSideLocalFSHPOStageError(
            f"{name} must be JSON-serialisable with deterministic keys"
        ) from exc


def _json_safe(value: Any, *, name: str) -> Any:
    """Round-trip a callback value through canonical JSON, rejecting NaN/Inf."""

    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        return json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PackBSideLocalFSHPOStageError(
            f"{name} must be JSON-serialisable and contain no NaN/Inf"
        ) from exc


def _validate_source_evidence(
    *,
    source_hashes: Mapping[str, str],
    source_revision: str,
    fixed_calendar_sha256: str,
) -> tuple[dict[str, str], str, str]:
    if set(source_hashes) != set(_SOURCE_HASH_FIELDS):
        raise PackBSideLocalFSHPOStageError(
            "source_hashes must contain exactly the locked stage-manifest source hashes"
        )
    normalised = {
        key: _require_sha256(source_hashes[key], name=f"source_hashes.{key}")
        for key in _SOURCE_HASH_FIELDS
    }
    revision = str(source_revision or "").strip().lower()
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise PackBSideLocalFSHPOStageError(
            "source_revision must be a 40-character Git SHA"
        )
    return (
        normalised,
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
        raise PackBSideLocalFSHPOStageError(f"{name} does not exist: {ledger_path}")
    actual = stage_manifest.sha256_file(ledger_path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise PackBSideLocalFSHPOStageError(
            f"{name} SHA-256 does not match its authorized source hash"
        )
    return ledger_path, actual


def _candidate_stream_evidence(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute a logical identity/timing summary independent of Parquet layout."""

    ordered = frame.loc[:, list(REQUIRED_LEDGER_COLUMNS)].copy()
    ordered["__ts__"] = _utc_series(ordered, "__ts__")
    ordered["__decision_ts__"] = _utc_series(ordered, "__decision_ts__")
    ordered["__label_resolution_ts__"] = _utc_series(ordered, "__label_resolution_ts__")
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
    signal = ordered["__ts__"]
    decision = ordered["__decision_ts__"]
    resolution = ordered["__label_resolution_ts__"]
    return {
        "count": int(len(ordered)),
        "identity_sha256": digest.hexdigest(),
        "signal_min_utc": signal.min().isoformat(),
        "signal_max_utc": signal.max().isoformat(),
        "decision_min_utc": decision.min().isoformat(),
        "decision_max_utc": decision.max().isoformat(),
        "label_resolution_min_utc": resolution.min().isoformat(),
        "label_resolution_max_utc": resolution.max().isoformat(),
    }


def _validate_candidate_ids(ledger: pd.DataFrame, *, name: str) -> None:
    candidate_ids = ledger["candidate_id"].astype("string")
    invalid = (
        candidate_ids.isna()
        | candidate_ids.str.strip().eq("")
        | candidate_ids.ne(candidate_ids.str.strip())
        | candidate_ids.duplicated(keep=False)
    )
    if invalid.any():
        raise PackBSideLocalFSHPOStageError(
            f"{name} has null, malformed, or duplicate candidate_id"
        )


def _validate_ledger_frame(
    ledger: pd.DataFrame,
    *,
    side: str,
    name: str,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
    role: str,
) -> pd.DataFrame:
    if not isinstance(ledger, pd.DataFrame):
        raise PackBSideLocalFSHPOStageError(f"{name} must be a pandas DataFrame")
    missing = sorted(set(REQUIRED_LEDGER_COLUMNS) - set(ledger.columns))
    if missing:
        raise PackBSideLocalFSHPOStageError(
            f"{name} misses required columns: " + ", ".join(missing)
        )
    if ledger.empty:
        raise PackBSideLocalFSHPOStageError(f"{name} is empty")
    local = ledger.loc[:, list(REQUIRED_LEDGER_COLUMNS)].copy()
    _validate_candidate_ids(local, name=name)
    sides = local["side_name"].astype("string").str.strip().str.lower()
    if sides.isna().any() or not sides.eq(side).all():
        raise PackBSideLocalFSHPOStageError(
            f"{name} must contain exactly {side!r} rows"
        )
    local["side_name"] = sides.astype(str)
    signal = _utc_series(local, "__ts__")
    decision = _utc_series(local, "__decision_ts__")
    resolution = _utc_series(local, "__label_resolution_ts__")
    if not decision.eq(signal + DECISION_LAG).all():
        raise PackBSideLocalFSHPOStageError(
            f"{name} violates decision_timestamp = signal_timestamp + 1h"
        )
    if not resolution.eq(decision + LABEL_RESOLUTION_HORIZON).all():
        raise PackBSideLocalFSHPOStageError(
            f"{name} violates label_resolution = decision_timestamp + 24h"
        )
    if not resolution.lt(RESOLUTION_CUTOFF_UTC).all():
        raise PackBSideLocalFSHPOStageError(
            f"{name} contains a label resolved at/after the pre-March cutoff"
        )
    if role == "train":
        allowed = signal.lt(validation_start - SIGNAL_PURGE) & resolution.lt(
            validation_start
        )
        rule = "signal < validation_start - 25h and resolution < validation_start"
    elif role == "validation":
        allowed = signal.ge(validation_start) & signal.lt(validation_end)
        rule = "validation_start <= signal < validation_end"
    else:  # pragma: no cover - all callers use the literals above.
        raise AssertionError(f"unsupported ledger role: {role}")
    if not allowed.all():
        raise PackBSideLocalFSHPOStageError(
            f"{name} violates its locked {role} calendar ({rule})"
        )
    return local


def _validate_bound_frame_identity(
    ledger: pd.DataFrame, *, path: Path, name: str
) -> None:
    try:
        on_disk = pd.read_parquet(path, columns=list(REQUIRED_LEDGER_COLUMNS))
    except Exception as exc:
        raise PackBSideLocalFSHPOStageError(
            f"cannot read bound {name} ledger file"
        ) from exc
    if (
        _candidate_stream_evidence(on_disk)["identity_sha256"]
        != _candidate_stream_evidence(ledger)["identity_sha256"]
    ):
        raise PackBSideLocalFSHPOStageError(
            f"in-memory {name} ledger does not match its bound ledger file"
        )


def _require_population_membership(
    *,
    candidate_path: Path,
    population_path: Path,
    name: str,
) -> None:
    """Use a bounded on-disk join rather than loading the population ledger."""

    try:
        import duckdb

        missing = duckdb.execute(
            """
            SELECT count(*)::BIGINT
            FROM read_parquet(?) AS candidate
            LEFT JOIN read_parquet(?) AS population
            USING (
                candidate_id,
                side_name,
                __symbol__,
                __ts__,
                __decision_ts__,
                __label_resolution_ts__
            )
            WHERE population.candidate_id IS NULL
            """,
            [str(candidate_path), str(population_path)],
        ).fetchone()
    except Exception as exc:
        raise PackBSideLocalFSHPOStageError(
            f"cannot prove {name} membership in the authorized population ledger: {exc}"
        ) from exc
    if missing is None or int(missing[0]):
        raise PackBSideLocalFSHPOStageError(
            f"{name} contains rows absent from the authorized population ledger"
        )


def _validate_candidate_features(features: Sequence[str]) -> tuple[str, ...]:
    values = tuple(str(value) for value in features)
    if not values or len(set(values)) != len(values):
        raise PackBSideLocalFSHPOStageError(
            "candidate_features must be a non-empty sequence of unique columns"
        )
    forbidden_exact = set(REQUIRED_LEDGER_COLUMNS) | {"side", "__side__"}
    blocked = [
        feature
        for feature in values
        if not feature.strip()
        or feature in forbidden_exact
        or feature.lower().startswith("side_")
        or any(token in feature.lower() for token in _FORBIDDEN_FEATURE_TOKENS)
    ]
    if blocked:
        raise PackBSideLocalFSHPOStageError(
            "candidate_features contain identity, side, or outcome-derived columns: "
            + ", ".join(blocked[:8])
        )
    return values


def _validate_feature_provenance(
    value: Mapping[str, Mapping[str, str]], *, features: Sequence[str]
) -> dict[str, dict[str, str]]:
    """Require a real per-feature causal/inference provenance registry.

    Name-based deny lists are merely a backstop.  A training feature is
    admitted only when its causal definition, inference availability, and
    units/normalisation contract are each explicitly bound by a hash.
    """

    if not isinstance(value, Mapping) or set(value) != set(features):
        raise PackBSideLocalFSHPOStageError(
            "feature_provenance must contain exactly one registry entry per candidate feature"
        )
    required = {
        "causal_definition_sha256",
        "inference_availability_sha256",
        "units_contract_sha256",
    }
    normalised: dict[str, dict[str, str]] = {}
    for feature in features:
        entry = value[feature]
        if not isinstance(entry, Mapping) or set(entry) != required:
            raise PackBSideLocalFSHPOStageError(
                f"feature_provenance[{feature!r}] must contain exactly "
                "causal_definition_sha256, inference_availability_sha256, and units_contract_sha256"
            )
        normalised[feature] = {
            key: _require_sha256(entry[key], name=f"feature_provenance.{feature}.{key}")
            for key in sorted(required)
        }
    return normalised


def _validate_extra_provenance_hashes(
    value: Mapping[str, str] | None,
) -> dict[str, str]:
    """Bind optional real raw-universe/loader/coverage artifacts in configs.

    The current side-stage-manifest schema intentionally has exactly seven
    source hashes.  Additional source artifacts therefore live in the bound
    stage config rather than weakening that manifest's exact key contract.
    """

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise PackBSideLocalFSHPOStageError("extra_provenance_hashes must be a mapping")
    result: dict[str, str] = {}
    for raw_name, raw_hash in value.items():
        name = str(raw_name).strip()
        if (
            not name
            or name in _SOURCE_HASH_FIELDS
            or name in result
            or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_" for char in name)
        ):
            raise PackBSideLocalFSHPOStageError(
                "extra_provenance_hashes keys must be unique lowercase artifact identifiers "
                "and may not replace the current seven source hashes"
            )
        result[name] = _require_sha256(raw_hash, name=f"extra_provenance_hashes.{name}")
    return result


def _bounded_beginning_middle_end_sample(
    ledger: pd.DataFrame, *, max_rows: int, name: str
) -> pd.DataFrame:
    """Deterministically cap a legal cohort without outcomes or random draws."""

    if not isinstance(max_rows, int) or max_rows < 1:
        raise PackBSideLocalFSHPOStageError(
            f"{name} max_rows must be a positive integer"
        )
    ordered = ledger.sort_values(
        ["__ts__", "__symbol__", "candidate_id"], kind="mergesort"
    ).reset_index(drop=True)
    if len(ordered) <= max_rows:
        return ordered
    # Equal-position selection deterministically covers early, middle, and late
    # periods without allowing target, weight, or score information to choose a
    # row.  ``linspace`` yields exactly max_rows distinct ascending locations.
    positions = np.linspace(0, len(ordered) - 1, num=max_rows, dtype=np.int64)
    positions = np.unique(positions)
    if len(positions) != max_rows:  # pragma: no cover - impossible with n > cap.
        raise AssertionError("deterministic B/M/E sampler lost a requested position")
    return ordered.iloc[positions].reset_index(drop=True)


def _raw_feature_matrix(
    value: Any, *, rows: int, columns: Sequence[str], name: str
) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise PackBSideLocalFSHPOStageError(f"{name} must return a pandas DataFrame")
    if len(value) != rows:
        raise PackBSideLocalFSHPOStageError(f"{name} returned a different row count")
    if list(value.columns) != list(columns):
        raise PackBSideLocalFSHPOStageError(
            f"{name} must return exactly the requested ordered feature columns"
        )
    try:
        matrix = value.reset_index(drop=True).astype(np.float32, copy=False)
    except (TypeError, ValueError) as exc:
        raise PackBSideLocalFSHPOStageError(
            f"{name} returned non-numeric feature values"
        ) from exc
    # Missingness is assessed by the per-window coverage gate below.  It is
    # never imputed: rows are admitted only after the joint-complete contract.
    return matrix.replace([np.inf, -np.inf], np.nan)


def _vector(value: Any, *, rows: int, name: str, numeric: bool) -> pd.Series:
    if isinstance(value, pd.DataFrame):
        if value.shape[1] != 1:
            raise PackBSideLocalFSHPOStageError(f"{name} must be one-dimensional")
        value = value.iloc[:, 0]
    if isinstance(value, pd.Series):
        series = value.reset_index(drop=True)
    elif isinstance(value, (list, tuple, np.ndarray)):
        series = pd.Series(value)
    else:
        raise PackBSideLocalFSHPOStageError(
            f"{name} must return a Series or one-dimensional sequence"
        )
    if len(series) != rows or series.isna().any():
        raise PackBSideLocalFSHPOStageError(
            f"{name} must have one non-null value for every filtered row"
        )
    if numeric:
        try:
            series = pd.to_numeric(series, errors="raise").astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise PackBSideLocalFSHPOStageError(f"{name} must be numeric") from exc
        if not np.isfinite(series.to_numpy(dtype=np.float64, copy=False)).all():
            raise PackBSideLocalFSHPOStageError(f"{name} contains non-finite values")
    return series


def _finalize_dataset(
    *,
    ledger: pd.DataFrame,
    matrix: pd.DataFrame,
    target_loader: TargetLoader,
    weight_loader: WeightLoader,
    name: str,
) -> StageDataset:
    """Attach target/weights only after calendar and coverage row filtering."""

    local = ledger.reset_index(drop=True).copy()
    target = _vector(
        target_loader(local.copy()),
        rows=len(local),
        name=f"target_loader for {name}",
        numeric=False,
    )
    weights = _vector(
        weight_loader(local.copy(), target.copy()),
        rows=len(local),
        name=f"weight_loader for {name}",
        numeric=True,
    )
    if (weights < 0).any() or float(weights.sum()) <= 0.0:
        raise PackBSideLocalFSHPOStageError(
            f"weight_loader for {name} must return non-negative weights with positive sum"
        )
    return StageDataset(ledger=local, features=matrix, target=target, weights=weights)


def _stage_dataset_sha256(value: StageDataset) -> str:
    """Bind callback inputs so one trial cannot mutate another trial's data."""

    digest = hashlib.sha256()
    for name, frame in (
        ("ledger", value.ledger),
        ("features", value.features),
        ("target", value.target.to_frame("__target__")),
        ("weights", value.weights.to_frame("__weight__")),
    ):
        schema = {
            "name": name,
            "columns": [str(column) for column in frame.columns],
            "dtypes": [str(dtype) for dtype in frame.dtypes],
            "rows": int(len(frame)),
        }
        digest.update(_canonical_json(schema, name=f"{name} dataset schema").encode())
        try:
            row_hashes = pd.util.hash_pandas_object(
                frame, index=True, categorize=True
            ).to_numpy(dtype=np.uint64, copy=False)
        except (TypeError, ValueError) as exc:
            raise PackBSideLocalFSHPOStageError(
                f"{name} callback dataset cannot be deterministically hashed"
            ) from exc
        digest.update(row_hashes.tobytes(order="C"))
    return digest.hexdigest()


def _require_callback_inputs_unchanged(
    *,
    train: StageDataset,
    validation: StageDataset,
    expected_train_sha256: str,
    expected_validation_sha256: str,
    callback_name: str,
) -> None:
    if (
        _stage_dataset_sha256(train) != expected_train_sha256
        or _stage_dataset_sha256(validation) != expected_validation_sha256
    ):
        raise PackBSideLocalFSHPOStageError(
            f"{callback_name} mutated its read-only side-local dataset inputs"
        )


def _finite_fractions(matrix: pd.DataFrame) -> dict[str, float]:
    finite = np.isfinite(matrix.to_numpy(dtype=np.float32, copy=False))
    return {
        column: float(finite[:, index].mean())
        for index, column in enumerate(matrix.columns)
    }


def _joint_complete_mask(matrix: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    if not columns:
        raise PackBSideLocalFSHPOStageError("no feature columns remain after coverage")
    return np.isfinite(
        matrix.loc[:, list(columns)].to_numpy(dtype=np.float32, copy=False)
    ).all(axis=1)


def _prepare_dataset_pair(
    *,
    train_ledger: pd.DataFrame,
    valid_ledger: pd.DataFrame,
    features: Sequence[str],
    feature_loader: FeatureLoader,
    target_loader: TargetLoader,
    weight_loader: WeightLoader,
    name: str,
    allow_feature_pruning: bool,
) -> tuple[StageDataset, StageDataset, tuple[str, ...], dict[str, Any]]:
    """Apply the side/window-specific 98% / 95% raw-admission contract.

    For the November selector, weak raw columns can be deterministically
    rejected before supervised callbacks.  Once selection is frozen, HPO may
    not silently alter that feature contract: every selected feature must meet
    coverage in *each* of its own train/validation windows.
    """

    requested = tuple(features)
    train_raw = _raw_feature_matrix(
        feature_loader(train_ledger.copy(), list(requested)),
        rows=len(train_ledger),
        columns=requested,
        name=f"feature_loader for {name} train",
    )
    valid_raw = _raw_feature_matrix(
        feature_loader(valid_ledger.copy(), list(requested)),
        rows=len(valid_ledger),
        columns=requested,
        name=f"feature_loader for {name} validation",
    )
    train_fraction = _finite_fractions(train_raw)
    valid_fraction = _finite_fractions(valid_raw)
    admissible = [
        column
        for column in requested
        if train_fraction[column] >= MIN_PER_FEATURE_FINITE_FRACTION
        and valid_fraction[column] >= MIN_PER_FEATURE_FINITE_FRACTION
    ]
    rejected_per_feature = [column for column in requested if column not in admissible]
    if rejected_per_feature and not allow_feature_pruning:
        raise PackBSideLocalFSHPOStageError(
            f"{name} selected feature coverage is below {MIN_PER_FEATURE_FINITE_FRACTION:.0%} "
            "in its own side-local train/validation window; fallback is forbidden"
        )
    if not admissible:
        raise PackBSideLocalFSHPOStageError(
            f"{name} has no features with >= {MIN_PER_FEATURE_FINITE_FRACTION:.0%} "
            "finite coverage in both side-local slices"
        )
    active = list(admissible)
    joint_pruned: list[str] = []
    while True:
        train_mask = _joint_complete_mask(train_raw, active)
        valid_mask = _joint_complete_mask(valid_raw, active)
        train_joint = float(train_mask.mean())
        valid_joint = float(valid_mask.mean())
        if (
            train_joint >= MIN_JOINT_COMPLETE_FRACTION
            and valid_joint >= MIN_JOINT_COMPLETE_FRACTION
        ):
            break
        if not allow_feature_pruning or len(active) <= 1:
            raise PackBSideLocalFSHPOStageError(
                f"{name} joint-complete retention is below "
                f"{MIN_JOINT_COMPLETE_FRACTION:.0%} in a side-local window; "
                "no global/default coverage fallback is allowed"
            )
        removal_scores: list[tuple[float, float, str]] = []
        for column in active:
            remaining = [item for item in active if item != column]
            candidate_train = float(_joint_complete_mask(train_raw, remaining).mean())
            candidate_valid = float(_joint_complete_mask(valid_raw, remaining).mean())
            # Larger worst-slice recovery is better; lower standalone coverage
            # is then pruned first; lexical order makes ties deterministic.
            removal_scores.append(
                (
                    min(candidate_train, candidate_valid),
                    min(train_fraction[column], valid_fraction[column]),
                    column,
                )
            )
        best_worst = max(item[0] for item in removal_scores)
        contenders = [item for item in removal_scores if item[0] == best_worst]
        best_coverage = min(item[1] for item in contenders)
        to_remove = sorted(item[2] for item in contenders if item[1] == best_coverage)[
            0
        ]
        active.remove(to_remove)
        joint_pruned.append(to_remove)
    if not train_mask.all() and float(train_mask.mean()) < MIN_JOINT_COMPLETE_FRACTION:
        raise AssertionError("joint coverage loop returned an invalid train mask")
    if not valid_mask.all() and float(valid_mask.mean()) < MIN_JOINT_COMPLETE_FRACTION:
        raise AssertionError("joint coverage loop returned an invalid validation mask")
    train_filtered_ledger = train_ledger.loc[train_mask].reset_index(drop=True)
    valid_filtered_ledger = valid_ledger.loc[valid_mask].reset_index(drop=True)
    train_matrix = train_raw.loc[train_mask, active].reset_index(drop=True)
    valid_matrix = valid_raw.loc[valid_mask, active].reset_index(drop=True)
    # The row masks are complete by construction; this is a proof check rather
    # than an imputation path.
    if (
        not np.isfinite(train_matrix.to_numpy(dtype=np.float32, copy=False)).all()
        or not np.isfinite(valid_matrix.to_numpy(dtype=np.float32, copy=False)).all()
    ):
        raise AssertionError("joint-complete row filtering left non-finite values")
    train_data = _finalize_dataset(
        ledger=train_filtered_ledger,
        matrix=train_matrix,
        target_loader=target_loader,
        weight_loader=weight_loader,
        name=f"{name} train",
    )
    valid_data = _finalize_dataset(
        ledger=valid_filtered_ledger,
        matrix=valid_matrix,
        target_loader=target_loader,
        weight_loader=weight_loader,
        name=f"{name} validation",
    )
    coverage = {
        "policy": {
            "per_feature_min_finite_fraction": MIN_PER_FEATURE_FINITE_FRACTION,
            "joint_complete_min_fraction": MIN_JOINT_COMPLETE_FRACTION,
            "scope": "per_side_per_fixed_train_validation_window",
            "global_fallback": "FORBIDDEN",
        },
        "requested_features": list(requested),
        "admitted_features": list(active),
        "rejected_per_feature_coverage": rejected_per_feature,
        "rejected_for_joint_complete_pruning": joint_pruned,
        "train": {
            "raw_rows": int(len(train_ledger)),
            "joint_complete_rows": int(train_mask.sum()),
            "joint_complete_fraction": float(train_mask.mean()),
            "finite_fraction_by_feature": train_fraction,
        },
        "validation": {
            "raw_rows": int(len(valid_ledger)),
            "joint_complete_rows": int(valid_mask.sum()),
            "joint_complete_fraction": float(valid_mask.mean()),
            "finite_fraction_by_feature": valid_fraction,
        },
    }
    return train_data, valid_data, tuple(active), coverage


def _require_no_overlap(left: pd.DataFrame, right: pd.DataFrame, *, name: str) -> None:
    overlap = set(left["candidate_id"]).intersection(right["candidate_id"])
    if overlap:
        raise PackBSideLocalFSHPOStageError(
            f"{name} train/validation candidate_id overlap is forbidden"
        )


def _normalise_feature_selection_result(
    value: Mapping[str, Any], *, side: str, candidates: tuple[str, ...]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PackBSideLocalFSHPOStageError(
            "feature_selection_callback must return a mapping"
        )
    result = _json_safe(dict(value), name="feature_selection_callback result")
    required = {
        "side",
        "selected_features",
        "selection_scope",
        "fallback_used",
        "selection_methods",
        "search_breadth",
    }
    missing = sorted(required - set(result))
    if missing:
        raise PackBSideLocalFSHPOStageError(
            "feature_selection_callback result misses: " + ", ".join(missing)
        )
    if str(result["side"]).strip().lower() != side:
        raise PackBSideLocalFSHPOStageError(
            "feature selection result is not side-local"
        )
    if str(result["selection_scope"]).strip().lower() != "side_local":
        raise PackBSideLocalFSHPOStageError(
            "feature selection result must declare selection_scope=side_local"
        )
    if result["fallback_used"] is not False:
        raise PackBSideLocalFSHPOStageError("feature selection fallback is forbidden")
    methods = result["selection_methods"]
    if not isinstance(methods, list) or not methods:
        raise PackBSideLocalFSHPOStageError(
            "feature selection result must declare non-empty selection_methods"
        )
    normalised_methods = [str(item).strip().lower() for item in methods]
    if "mda" not in normalised_methods:
        raise PackBSideLocalFSHPOStageError(
            "feature selection must include side-local MDA; univariate-only fallback is forbidden"
        )
    if any(
        token in method
        for method in normalised_methods
        for token in ("global", "pooled", "default", "fallback")
    ):
        raise PackBSideLocalFSHPOStageError(
            "feature selection methods declare a forbidden global/pooled/default/fallback path"
        )
    try:
        breadth = int(result["search_breadth"])
    except (TypeError, ValueError) as exc:
        raise PackBSideLocalFSHPOStageError(
            "feature selection search_breadth must be a positive integer"
        ) from exc
    if isinstance(result["search_breadth"], bool) or breadth < 1:
        raise PackBSideLocalFSHPOStageError(
            "feature selection search_breadth must be a positive integer"
        )
    selected = result["selected_features"]
    if not isinstance(selected, list) or not selected:
        raise PackBSideLocalFSHPOStageError(
            "feature selection must return non-empty explicit selected_features"
        )
    selected_features = [str(item) for item in selected]
    if len(set(selected_features)) != len(selected_features) or any(
        item not in candidates for item in selected_features
    ):
        raise PackBSideLocalFSHPOStageError(
            "selected_features must be unique members of candidate_features"
        )
    result["side"] = side
    result["selection_scope"] = "side_local"
    result["selection_methods"] = normalised_methods
    result["search_breadth"] = breadth
    result["selected_features"] = selected_features
    return result


def _normalise_trials(trials: Sequence[HPOTrial]) -> tuple[HPOTrial, ...]:
    if not isinstance(trials, Sequence) or isinstance(trials, (str, bytes)):
        raise PackBSideLocalFSHPOStageError("hpo_trials must be a sequence")
    if len(trials) < 2:
        raise PackBSideLocalFSHPOStageError(
            "HPO requires at least two explicit trial arms; default fallback is forbidden"
        )
    result: list[HPOTrial] = []
    seen_ids: set[str] = set()
    seen_params: set[str] = set()
    for item in trials:
        if not isinstance(item, HPOTrial):
            raise PackBSideLocalFSHPOStageError(
                "hpo_trials must contain HPOTrial values"
            )
        trial_id = str(item.trial_id).strip()
        if not trial_id or trial_id in seen_ids:
            raise PackBSideLocalFSHPOStageError(
                "HPO trial_id values must be non-empty and unique"
            )
        if not isinstance(item.params, Mapping) or not item.params:
            raise PackBSideLocalFSHPOStageError(
                f"HPO trial {trial_id!r} must have non-empty explicit params"
            )
        params = _json_safe(dict(item.params), name=f"HPO trial {trial_id!r} params")
        if not isinstance(params, dict) or not params:
            raise PackBSideLocalFSHPOStageError(
                f"HPO trial {trial_id!r} params must be an object"
            )
        param_key = _canonical_json(params, name=f"HPO trial {trial_id!r} params")
        if param_key in seen_params:
            raise PackBSideLocalFSHPOStageError(
                "HPO trial params must be distinct; duplicate/default arms are forbidden"
            )
        seen_ids.add(trial_id)
        seen_params.add(param_key)
        result.append(HPOTrial(trial_id=trial_id, params=params))
    return tuple(result)


def _normalise_trial_result(
    value: Any, *, trial_id: str, fold_name: str
) -> dict[str, Any]:
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, bool
    ):
        objective = float(value)
        if not math.isfinite(objective):
            raise PackBSideLocalFSHPOStageError(
                f"HPO evaluator result for {trial_id}/{fold_name} is non-finite"
            )
        return {"objective": objective}
    if not isinstance(value, Mapping):
        raise PackBSideLocalFSHPOStageError(
            f"HPO evaluator result for {trial_id}/{fold_name} must be numeric or a mapping"
        )
    result = _json_safe(
        dict(value), name=f"HPO evaluator result {trial_id}/{fold_name}"
    )
    if "objective" not in result or isinstance(result["objective"], bool):
        raise PackBSideLocalFSHPOStageError(
            f"HPO evaluator result for {trial_id}/{fold_name} must include numeric objective"
        )
    try:
        objective = float(result["objective"])
    except (TypeError, ValueError) as exc:
        raise PackBSideLocalFSHPOStageError(
            f"HPO evaluator objective for {trial_id}/{fold_name} must be numeric"
        ) from exc
    if not math.isfinite(objective):
        raise PackBSideLocalFSHPOStageError(
            f"HPO evaluator objective for {trial_id}/{fold_name} is non-finite"
        )
    result["objective"] = objective
    return result


def _normalise_hpo_selection(
    value: Mapping[str, Any], *, side: str, trials: Sequence[HPOTrial]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PackBSideLocalFSHPOStageError(
            "hpo_selection_callback must return a mapping"
        )
    result = _json_safe(dict(value), name="hpo_selection_callback result")
    required = {
        "side",
        "selected_trial_id",
        "selected_params",
        "selection_scope",
        "fallback_used",
        "selection_metric",
    }
    missing = sorted(required - set(result))
    if missing:
        raise PackBSideLocalFSHPOStageError(
            "hpo_selection_callback result misses: " + ", ".join(missing)
        )
    if str(result["side"]).strip().lower() != side:
        raise PackBSideLocalFSHPOStageError("HPO selection result is not side-local")
    if str(result["selection_scope"]).strip().lower() != "side_local":
        raise PackBSideLocalFSHPOStageError(
            "HPO selection result must declare selection_scope=side_local"
        )
    if result["fallback_used"] is not False:
        raise PackBSideLocalFSHPOStageError("HPO fallback is forbidden")
    trial_id = str(result["selected_trial_id"]).strip()
    by_id = {trial.trial_id: trial for trial in trials}
    if trial_id not in by_id:
        raise PackBSideLocalFSHPOStageError(
            "HPO selected_trial_id is not one of the evaluated explicit trials"
        )
    params = result["selected_params"]
    if not isinstance(params, dict) or not params:
        raise PackBSideLocalFSHPOStageError(
            "HPO must return non-empty explicit selected_params"
        )
    if _canonical_json(params, name="selected_params") != _canonical_json(
        by_id[trial_id].params, name="selected trial params"
    ):
        raise PackBSideLocalFSHPOStageError(
            "HPO selected_params do not match the declared evaluated selected trial"
        )
    metric = str(result["selection_metric"]).strip()
    if not metric:
        raise PackBSideLocalFSHPOStageError("HPO selection_metric must be non-empty")
    result["side"] = side
    result["selection_scope"] = "side_local"
    result["selected_trial_id"] = trial_id
    result["selected_params"] = dict(by_id[trial_id].params)
    result["selection_metric"] = metric
    return result


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise PackBSideLocalFSHPOStageError(
            f"refusing to overwrite immutable stage evidence: {path}"
        )
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _write_parquet_once(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise PackBSideLocalFSHPOStageError(
            f"refusing to overwrite immutable candidate evidence: {path}"
        )
    frame.to_parquet(path, index=False)


def _stage_config(
    *,
    side: str,
    stage: str,
    source_hashes: Mapping[str, str],
    source_revision: str,
    fixed_calendar_sha256: str,
    population_path: Path,
    population_sha256: str,
    ledger_bindings: Mapping[str, Mapping[str, Any]],
    candidate_evidence: Mapping[str, Any],
    feature_provenance: Mapping[str, Mapping[str, str]],
    extra_provenance_hashes: Mapping[str, str],
    details: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": FS_HPO_STAGE_SCHEMA,
        "stage": stage,
        "side": side,
        "source_revision": source_revision,
        "source_hashes": dict(source_hashes),
        "fixed_calendar_sha256": fixed_calendar_sha256,
        "locked_calendar": locked_calendar(),
        "authorized_population_ledger": {
            "path": str(population_path),
            "sha256": population_sha256,
        },
        "input_ledgers": dict(ledger_bindings),
        "candidate_evidence": dict(candidate_evidence),
        "feature_provenance": {
            key: dict(value) for key, value in feature_provenance.items()
        },
        "extra_provenance_hashes": dict(extra_provenance_hashes),
        "details": dict(details),
    }


def _emit_stage(
    *,
    destination: Path,
    side: str,
    stage: str,
    source_hashes: Mapping[str, str],
    source_revision: str,
    fixed_calendar_sha256: str,
    population_path: Path,
    population_sha256: str,
    candidate_frame: pd.DataFrame,
    config: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> dict[str, str]:
    candidate_path = destination / f"{stage}_candidates.parquet"
    _write_parquet_once(
        candidate_path, candidate_frame.loc[:, list(REQUIRED_LEDGER_COLUMNS)]
    )
    candidate_sha256 = stage_manifest.sha256_file(candidate_path)
    candidate_summary = _candidate_stream_evidence(candidate_frame)
    config_path = destination / f"{stage}_stage_config.json"
    _write_json_once(config_path, config)
    config_sha256 = stage_manifest.sha256_file(config_path)
    artifact_path = destination / (
        "feature_contract.json"
        if stage == "feature_selection"
        else "hpo_parameters.json"
    )
    _write_json_once(artifact_path, artifact)
    artifact_sha256 = stage_manifest.sha256_file(artifact_path)
    manifest_payload = {
        "schema": stage_manifest.SIDE_STAGE_MANIFEST_SCHEMA,
        "source_revision": source_revision,
        "side": side,
        "stage": stage,
        "resolution_cutoff_utc": RESOLUTION_CUTOFF_UTC.isoformat(),
        "actual_label_resolution_contract": stage_manifest.ACTUAL_LABEL_RESOLUTION_CONTRACT,
        "source_hashes": dict(source_hashes),
        "authorized_population_ledger": {
            "path": str(population_path),
            "sha256": population_sha256,
        },
        "candidate_stream": {
            "path": candidate_path.name,
            "count": int(candidate_summary["count"]),
            "sha256": candidate_sha256,
            **{
                key: candidate_summary[key]
                for key in (
                    "signal_min_utc",
                    "signal_max_utc",
                    "decision_min_utc",
                    "decision_max_utc",
                    "label_resolution_min_utc",
                    "label_resolution_max_utc",
                )
            },
        },
        "fixed_calendar_sha256": fixed_calendar_sha256,
        "stage_config": {"path": config_path.name, "sha256": config_sha256},
        "artifact": {
            "kind": "feature_contract" if stage == "feature_selection" else "parameter",
            "path": artifact_path.name,
            "sha256": artifact_sha256,
            "scope": side,
        },
    }
    manifest_path = destination / f"{stage}_side_stage_manifest.json"
    manifest_sha256 = stage_manifest.write_immutable_side_stage_manifest(
        manifest_path, manifest_payload
    )
    return {
        "candidate_path": str(candidate_path),
        "candidate_sha256": candidate_sha256,
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "artifact_path": str(artifact_path),
        "artifact_sha256": artifact_sha256,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
    }


def fit_side_local_fs_hpo_stages(
    *,
    side: str,
    fs_train_ledger: pd.DataFrame,
    fs_train_ledger_path: Path,
    fs_valid_ledger: pd.DataFrame,
    fs_valid_ledger_path: Path,
    hpo_folds: Sequence[HPOFoldLedger],
    authorized_population_ledger_path: Path,
    feature_loader: FeatureLoader,
    target_loader: TargetLoader,
    weight_loader: WeightLoader,
    candidate_features: Sequence[str],
    feature_provenance: Mapping[str, Mapping[str, str]],
    feature_selection_callback: FeatureSelectionCallback,
    hpo_trials: Sequence[HPOTrial],
    hpo_trial_evaluator: HPOTrialEvaluator,
    hpo_selection_callback: HPOSelectionCallback,
    output_dir: Path,
    source_hashes: Mapping[str, str],
    source_revision: str,
    fixed_calendar_sha256: str,
    extra_provenance_hashes: Mapping[str, str] | None = None,
    fs_train_max_rows: int = DEFAULT_FS_TRAIN_MAX_ROWS,
    fs_valid_max_rows: int = DEFAULT_FS_VALID_MAX_ROWS,
    hpo_train_max_rows: int = DEFAULT_HPO_TRAIN_MAX_ROWS,
    hpo_valid_max_rows: int = DEFAULT_HPO_VALID_MAX_ROWS,
    resource_guard: TrainingResourceGuard | Any | None = None,
) -> dict[str, Any]:
    """Run one side's strict November FS and Dec--Feb three-fold HPO stages.

    The callback sequence is intentionally fixed.  Feature-selection callbacks
    receive the one legal November pair.  Every declared HPO arm is then
    evaluated once on each of the three legal chronological folds, with fresh
    bounded callback inputs per arm/fold.  A selector may choose only a
    declared evaluated arm.  There is no historical, pooled, global,
    univariate-only, or default fallback path.
    """

    normalised_side = str(side or "").strip().lower()
    if normalised_side not in CANONICAL_SIDES:
        raise PackBSideLocalFSHPOStageError("side must be exactly long or short")
    features = _validate_candidate_features(candidate_features)
    feature_registry = _validate_feature_provenance(
        feature_provenance, features=features
    )
    extra_provenance = _validate_extra_provenance_hashes(extra_provenance_hashes)
    source_evidence, revision, calendar_hash = _validate_source_evidence(
        source_hashes=source_hashes,
        source_revision=source_revision,
        fixed_calendar_sha256=fixed_calendar_sha256,
    )
    trials = _normalise_trials(hpo_trials)
    for name, value in (
        ("fs_train_max_rows", fs_train_max_rows),
        ("fs_valid_max_rows", fs_valid_max_rows),
        ("hpo_train_max_rows", hpo_train_max_rows),
        ("hpo_valid_max_rows", hpo_valid_max_rows),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise PackBSideLocalFSHPOStageError(f"{name} must be a positive integer")
    destination = Path(output_dir)
    if destination.exists():
        raise PackBSideLocalFSHPOStageError(
            f"refusing to overwrite or reuse output directory: {destination}"
        )
    guard = resource_guard or TrainingResourceGuard(disk_path=destination.parent)
    guard.preflight(f"packb_side_local_fs_hpo:{normalised_side}:preflight")

    population_path, population_sha256 = _validate_bound_ledger(
        authorized_population_ledger_path,
        name="authorized population ledger",
        expected_sha256=source_evidence["authorized_population_ledger_sha256"],
    )
    fs_train_path, fs_train_sha256 = _validate_bound_ledger(
        fs_train_ledger_path, name="feature-selection train ledger"
    )
    fs_valid_path, fs_valid_sha256 = _validate_bound_ledger(
        fs_valid_ledger_path, name="feature-selection validation ledger"
    )
    fs_train = _validate_ledger_frame(
        fs_train_ledger,
        side=normalised_side,
        name="feature-selection train ledger",
        validation_start=FS_VALIDATION[0],
        validation_end=FS_VALIDATION[1],
        role="train",
    )
    fs_valid = _validate_ledger_frame(
        fs_valid_ledger,
        side=normalised_side,
        name="feature-selection validation ledger",
        validation_start=FS_VALIDATION[0],
        validation_end=FS_VALIDATION[1],
        role="validation",
    )
    _validate_bound_frame_identity(
        fs_train, path=fs_train_path, name="feature-selection train"
    )
    _validate_bound_frame_identity(
        fs_valid, path=fs_valid_path, name="feature-selection validation"
    )
    _require_no_overlap(fs_train, fs_valid, name="feature-selection")
    _require_population_membership(
        candidate_path=fs_train_path,
        population_path=population_path,
        name="feature-selection train ledger",
    )
    _require_population_membership(
        candidate_path=fs_valid_path,
        population_path=population_path,
        name="feature-selection validation ledger",
    )
    guard.checkpoint(f"packb_side_local_fs_hpo:{normalised_side}:fs_calendar_verified")

    expected_fold_names = tuple(f"hpo_{index}" for index in range(1, 4))
    if (
        not isinstance(hpo_folds, Sequence)
        or any(not isinstance(fold, HPOFoldLedger) for fold in hpo_folds)
        or tuple(fold.name for fold in hpo_folds) != expected_fold_names
    ):
        raise PackBSideLocalFSHPOStageError(
            "hpo_folds must be exactly ordered hpo_1, hpo_2, hpo_3"
        )
    validated_folds: list[dict[str, Any]] = []
    validation_ids: set[str] = set()
    for index, (fold, interval) in enumerate(
        zip(hpo_folds, HPO_VALIDATIONS, strict=True), start=1
    ):
        train_path, train_sha256 = _validate_bound_ledger(
            fold.train_ledger_path, name=f"{fold.name} train ledger"
        )
        valid_path, valid_sha256 = _validate_bound_ledger(
            fold.valid_ledger_path, name=f"{fold.name} validation ledger"
        )
        train = _validate_ledger_frame(
            fold.train_ledger,
            side=normalised_side,
            name=f"{fold.name} train ledger",
            validation_start=interval[0],
            validation_end=interval[1],
            role="train",
        )
        valid = _validate_ledger_frame(
            fold.valid_ledger,
            side=normalised_side,
            name=f"{fold.name} validation ledger",
            validation_start=interval[0],
            validation_end=interval[1],
            role="validation",
        )
        _validate_bound_frame_identity(
            train, path=train_path, name=f"{fold.name} train"
        )
        _validate_bound_frame_identity(
            valid, path=valid_path, name=f"{fold.name} validation"
        )
        _require_no_overlap(train, valid, name=fold.name)
        overlap = validation_ids.intersection(set(valid["candidate_id"]))
        if overlap:
            raise PackBSideLocalFSHPOStageError(
                "HPO validation candidate_id overlap across fixed folds is forbidden"
            )
        validation_ids.update(valid["candidate_id"])
        _require_population_membership(
            candidate_path=train_path,
            population_path=population_path,
            name=f"{fold.name} train ledger",
        )
        _require_population_membership(
            candidate_path=valid_path,
            population_path=population_path,
            name=f"{fold.name} validation ledger",
        )
        validated_folds.append(
            {
                "name": f"hpo_{index}",
                "start": interval[0],
                "end": interval[1],
                "train": train,
                "valid": valid,
                "train_path": train_path,
                "train_sha256": train_sha256,
                "valid_path": valid_path,
                "valid_sha256": valid_sha256,
            }
        )
    guard.checkpoint(
        f"packb_side_local_fs_hpo:{normalised_side}:hpo_calendars_verified"
    )

    ledger_bindings: dict[str, dict[str, Any]] = {
        "feature_selection_train": {
            "path": str(fs_train_path),
            "sha256": fs_train_sha256,
            "identity_sha256": _candidate_stream_evidence(fs_train)["identity_sha256"],
        },
        "feature_selection_valid": {
            "path": str(fs_valid_path),
            "sha256": fs_valid_sha256,
            "identity_sha256": _candidate_stream_evidence(fs_valid)["identity_sha256"],
        },
    }
    for fold in validated_folds:
        ledger_bindings[f"{fold['name']}_train"] = {
            "path": str(fold["train_path"]),
            "sha256": fold["train_sha256"],
            "identity_sha256": _candidate_stream_evidence(fold["train"])[
                "identity_sha256"
            ],
        }
        ledger_bindings[f"{fold['name']}_valid"] = {
            "path": str(fold["valid_path"]),
            "sha256": fold["valid_sha256"],
            "identity_sha256": _candidate_stream_evidence(fold["valid"])[
                "identity_sha256"
            ],
        }

    fs_train_sample = _bounded_beginning_middle_end_sample(
        fs_train, max_rows=fs_train_max_rows, name="feature-selection train"
    )
    fs_valid_sample = _bounded_beginning_middle_end_sample(
        fs_valid, max_rows=fs_valid_max_rows, name="feature-selection validation"
    )
    guard.checkpoint(f"packb_side_local_fs_hpo:{normalised_side}:before_fs_window_load")
    fs_train_data, fs_valid_data, admitted_features, fs_coverage = (
        _prepare_dataset_pair(
            train_ledger=fs_train_sample,
            valid_ledger=fs_valid_sample,
            features=features,
            feature_loader=feature_loader,
            target_loader=target_loader,
            weight_loader=weight_loader,
            name="feature-selection November",
            allow_feature_pruning=True,
        )
    )
    guard.checkpoint(
        f"packb_side_local_fs_hpo:{normalised_side}:before_feature_selection"
    )
    fs_train_input_sha256 = _stage_dataset_sha256(fs_train_data)
    fs_valid_input_sha256 = _stage_dataset_sha256(fs_valid_data)
    fs_dataset_sha256 = {
        "train": fs_train_input_sha256,
        "validation": fs_valid_input_sha256,
    }
    fs_result = _normalise_feature_selection_result(
        feature_selection_callback(
            FeatureSelectionInput(
                side=normalised_side,
                candidate_features=admitted_features,
                train=fs_train_data,
                validation=fs_valid_data,
            )
        ),
        side=normalised_side,
        candidates=admitted_features,
    )
    _require_callback_inputs_unchanged(
        train=fs_train_data,
        validation=fs_valid_data,
        expected_train_sha256=fs_train_input_sha256,
        expected_validation_sha256=fs_valid_input_sha256,
        callback_name="feature_selection_callback",
    )
    selected_features = tuple(fs_result["selected_features"])
    fs_candidate = pd.concat(
        [fs_train_data.ledger, fs_valid_data.ledger], ignore_index=True
    )
    # Drop large candidate matrices as soon as selection has frozen the feature set.
    del fs_train_data, fs_valid_data

    evaluations: list[HPOTrialEvaluation] = []
    hpo_coverage_by_fold: dict[str, dict[str, Any]] = {}
    hpo_dataset_sha256_by_fold: dict[str, dict[str, str]] = {}
    hpo_valid_evidence_frames: list[pd.DataFrame] = []
    # Feature-store I/O is fold-scoped, not trial-scoped.  This bounds a 150
    # arm sweep to three loads while retaining guard checks before every trial.
    for fold in validated_folds:
        fold_train_sample = _bounded_beginning_middle_end_sample(
            fold["train"], max_rows=hpo_train_max_rows, name=f"{fold['name']} train"
        )
        fold_valid_sample = _bounded_beginning_middle_end_sample(
            fold["valid"],
            max_rows=hpo_valid_max_rows,
            name=f"{fold['name']} validation",
        )
        fold_prefix = f"packb_side_local_fs_hpo:{normalised_side}:{fold['name']}"
        guard.checkpoint(fold_prefix + ":before_hpo_fold_load")
        train_data, valid_data, hpo_features, fold_coverage = _prepare_dataset_pair(
            train_ledger=fold_train_sample,
            valid_ledger=fold_valid_sample,
            features=selected_features,
            feature_loader=feature_loader,
            target_loader=target_loader,
            weight_loader=weight_loader,
            name=fold["name"],
            allow_feature_pruning=False,
        )
        if hpo_features != selected_features:  # pragma: no cover - no-pruning proof.
            raise AssertionError("HPO changed the frozen selected feature contract")
        hpo_coverage_by_fold[fold["name"]] = fold_coverage
        hpo_valid_evidence_frames.append(valid_data.ledger.copy())
        train_input_sha256 = _stage_dataset_sha256(train_data)
        valid_input_sha256 = _stage_dataset_sha256(valid_data)
        hpo_dataset_sha256_by_fold[fold["name"]] = {
            "train": train_input_sha256,
            "validation": valid_input_sha256,
        }
        for trial in trials:
            stage_prefix = f"{fold_prefix}:{trial.trial_id}"
            guard.checkpoint(stage_prefix + ":before_hpo_trial_evaluation")
            evaluation = _normalise_trial_result(
                hpo_trial_evaluator(
                    HPOFoldInput(
                        side=normalised_side,
                        fold_name=fold["name"],
                        validation_start_utc=fold["start"].isoformat(),
                        validation_end_utc=fold["end"].isoformat(),
                        trial=trial,
                        selected_features=selected_features,
                        train=train_data,
                        validation=valid_data,
                    )
                ),
                trial_id=trial.trial_id,
                fold_name=fold["name"],
            )
            evaluations.append(
                HPOTrialEvaluation(
                    trial_id=trial.trial_id,
                    params=dict(trial.params),
                    fold_name=fold["name"],
                    result=evaluation,
                )
            )
            _require_callback_inputs_unchanged(
                train=train_data,
                validation=valid_data,
                expected_train_sha256=train_input_sha256,
                expected_validation_sha256=valid_input_sha256,
                callback_name=(
                    f"hpo_trial_evaluator for {fold['name']}/{trial.trial_id}"
                ),
            )
        del train_data, valid_data
    guard.checkpoint(f"packb_side_local_fs_hpo:{normalised_side}:before_hpo_selection")
    hpo_result = _normalise_hpo_selection(
        hpo_selection_callback(tuple(evaluations)), side=normalised_side, trials=trials
    )

    if fs_candidate["candidate_id"].duplicated().any():  # Defensive; checked above.
        raise PackBSideLocalFSHPOStageError("feature-selection evidence is not unique")
    hpo_candidate = pd.concat(hpo_valid_evidence_frames, ignore_index=True)
    if hpo_candidate["candidate_id"].duplicated().any():  # Defensive; checked above.
        raise PackBSideLocalFSHPOStageError("HPO validation evidence is not unique")
    fs_candidate_evidence = _candidate_stream_evidence(fs_candidate)
    hpo_candidate_evidence = _candidate_stream_evidence(hpo_candidate)

    fs_artifact = {
        "schema": FS_HPO_STAGE_SCHEMA,
        "artifact_kind": "feature_contract",
        "side": normalised_side,
        "selection": fs_result,
        "raw_candidate_features": list(features),
        "admitted_candidate_features": list(admitted_features),
        "selected_features": list(selected_features),
        "coverage": fs_coverage,
        "dataset_sha256": fs_dataset_sha256,
        "feature_provenance": feature_registry,
        "extra_provenance_hashes": extra_provenance,
        "source_revision": revision,
        "fixed_calendar_sha256": calendar_hash,
    }
    hpo_artifact = {
        "schema": FS_HPO_STAGE_SCHEMA,
        "artifact_kind": "parameter",
        "side": normalised_side,
        "selected_features": list(selected_features),
        "selection": hpo_result,
        "trial_arms": [
            {"trial_id": trial.trial_id, "params": dict(trial.params)}
            for trial in trials
        ],
        "evaluations": [
            {
                "trial_id": item.trial_id,
                "params": dict(item.params),
                "fold_name": item.fold_name,
                "result": dict(item.result),
            }
            for item in evaluations
        ],
        "coverage_by_fold": hpo_coverage_by_fold,
        "dataset_sha256_by_fold": hpo_dataset_sha256_by_fold,
        "feature_provenance": {
            feature: feature_registry[feature] for feature in selected_features
        },
        "extra_provenance_hashes": extra_provenance,
        "source_revision": revision,
        "fixed_calendar_sha256": calendar_hash,
    }
    fs_config = _stage_config(
        side=normalised_side,
        stage="feature_selection",
        source_hashes=source_evidence,
        source_revision=revision,
        fixed_calendar_sha256=calendar_hash,
        population_path=population_path,
        population_sha256=population_sha256,
        ledger_bindings={
            key: value
            for key, value in ledger_bindings.items()
            if key.startswith("feature_selection")
        },
        candidate_evidence={
            "published_sample": fs_candidate_evidence,
            "source_train": _candidate_stream_evidence(fs_train),
            "source_validation": _candidate_stream_evidence(fs_valid),
            "sampled_before_coverage_train": _candidate_stream_evidence(
                fs_train_sample
            ),
            "sampled_before_coverage_validation": _candidate_stream_evidence(
                fs_valid_sample
            ),
        },
        feature_provenance=feature_registry,
        extra_provenance_hashes=extra_provenance,
        details={
            "raw_candidate_feature_order": list(features),
            "admitted_candidate_feature_order": list(admitted_features),
            "selection_methods": fs_result["selection_methods"],
            "search_breadth": fs_result["search_breadth"],
            "selected_features": list(selected_features),
            "coverage": fs_coverage,
            "coverage_sha256": stage_manifest.canonical_json_sha256(fs_coverage),
            "dataset_sha256": fs_dataset_sha256,
            "sample_caps": {
                "train_max_rows": fs_train_max_rows,
                "validation_max_rows": fs_valid_max_rows,
                "policy": "deterministic_timestamp_symbol_candidate_beginning_middle_end",
            },
            "fallback": "FORBIDDEN",
        },
    )
    hpo_config = _stage_config(
        side=normalised_side,
        stage="hpo",
        source_hashes=source_evidence,
        source_revision=revision,
        fixed_calendar_sha256=calendar_hash,
        population_path=population_path,
        population_sha256=population_sha256,
        ledger_bindings={
            key: value
            for key, value in ledger_bindings.items()
            if key.startswith("hpo_")
        },
        candidate_evidence={
            "published_validation_samples": hpo_candidate_evidence,
            "source_validation_hpo_1": _candidate_stream_evidence(
                validated_folds[0]["valid"]
            ),
            "source_validation_hpo_2": _candidate_stream_evidence(
                validated_folds[1]["valid"]
            ),
            "source_validation_hpo_3": _candidate_stream_evidence(
                validated_folds[2]["valid"]
            ),
        },
        feature_provenance={
            feature: feature_registry[feature] for feature in selected_features
        },
        extra_provenance_hashes=extra_provenance,
        details={
            "selected_features": list(selected_features),
            "trials_declared": len(trials),
            "folds_evaluated_per_trial": len(validated_folds),
            "evaluations": len(evaluations),
            "selected_trial_id": hpo_result["selected_trial_id"],
            "coverage_by_fold": hpo_coverage_by_fold,
            "coverage_sha256": stage_manifest.canonical_json_sha256(
                hpo_coverage_by_fold
            ),
            "dataset_sha256_by_fold": hpo_dataset_sha256_by_fold,
            "sample_caps": {
                "train_max_rows": hpo_train_max_rows,
                "validation_max_rows": hpo_valid_max_rows,
                "policy": "deterministic_timestamp_symbol_candidate_beginning_middle_end",
            },
            "fallback": "FORBIDDEN",
        },
    )

    stage = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    if stage.exists():  # pragma: no cover - UUID collision guard.
        raise PackBSideLocalFSHPOStageError(f"staging path already exists: {stage}")
    try:
        guard.checkpoint(f"packb_side_local_fs_hpo:{normalised_side}:before_persist")
        stage.mkdir(parents=True, exist_ok=False)
        fs_paths = _emit_stage(
            destination=stage,
            side=normalised_side,
            stage="feature_selection",
            source_hashes=source_evidence,
            source_revision=revision,
            fixed_calendar_sha256=calendar_hash,
            population_path=population_path,
            population_sha256=population_sha256,
            candidate_frame=fs_candidate,
            config=fs_config,
            artifact=fs_artifact,
        )
        hpo_paths = _emit_stage(
            destination=stage,
            side=normalised_side,
            stage="hpo",
            source_hashes=source_evidence,
            source_revision=revision,
            fixed_calendar_sha256=calendar_hash,
            population_path=population_path,
            population_sha256=population_sha256,
            candidate_frame=hpo_candidate,
            config=hpo_config,
            artifact=hpo_artifact,
        )
        summary = {
            "schema": FS_HPO_STAGE_SCHEMA,
            "status": "FROZEN_SIDE_LOCAL_FEATURE_SELECTION_AND_HPO",
            "side": normalised_side,
            "selected_features": list(selected_features),
            "selected_hpo_trial_id": hpo_result["selected_trial_id"],
            "selected_params": hpo_result["selected_params"],
            "hpo_trials_evaluated": len(trials),
            "hpo_fold_evaluations": len(evaluations),
            "feature_selection": fs_paths,
            "hpo": hpo_paths,
            "source_revision": revision,
            "fixed_calendar_sha256": calendar_hash,
        }
        persisted_summary = {
            **summary,
            "feature_selection": {
                key: Path(value).name if key.endswith("_path") else value
                for key, value in fs_paths.items()
            },
            "hpo": {
                key: Path(value).name if key.endswith("_path") else value
                for key, value in hpo_paths.items()
            },
        }
        _write_json_once(stage / "summary.json", persisted_summary)
        guard.checkpoint(f"packb_side_local_fs_hpo:{normalised_side}:complete")
        os.replace(stage, destination)
        return {
            **summary,
            "summary_path": str(destination / "summary.json"),
            "feature_selection": {
                key: str(destination / Path(value).name)
                if key.endswith("_path")
                else value
                for key, value in fs_paths.items()
            },
            "hpo": {
                key: str(destination / Path(value).name)
                if key.endswith("_path")
                else value
                for key, value in hpo_paths.items()
            },
        }
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise
