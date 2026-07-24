"""Bounded, exact point-in-time reads from the canonical static feature store.

Pack-B stages receive an identity ledger rather than a pre-joined feature
matrix.  This module is the only bridge from that ledger to the canonical
per-symbol feature store.  It deliberately has a narrow contract:

* feature rows are joined on exactly ``(__symbol__, __ts__)``;
* reads use :func:`data_store.read_symbol_features`, including the canonical
  Parquet base, repair parts, and DuckDB delta view;
* there is no as-of, forward-fill, or nearest-neighbour lookup;
* schemas are freshly discovered from the requested store, never from an old
  feature-importance artifact; and
* normal training reads fail closed on duplicate identities, missing store
  keys, changed schemas, or non-numeric model inputs.

The iterator is the primary API for large cohorts.  It reads one symbol and a
bounded number of requested ledger rows at a time, then releases that source
frame before moving on.  ``load_point_in_time_features`` is a convenience for
small, bounded consumers such as the deterministic AE/GMM reference sample.
It refuses output matrices larger than an explicit cap instead of risking an
unobserved process-level OOM.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from extreme_price_movements.data_store import (
    _feature_schema_names,
    _symbol_alias_candidates,
    read_symbol_features,
)
from extreme_price_movements.training_resource_guard import TrainingResourceGuard

POINT_FEATURE_LOADER_SCHEMA = "packb_static_point_feature_loader_v1"
IDENTITY_COLUMNS = ("candidate_id", "__ts__", "__symbol__")
DEFAULT_MAX_ROWS_PER_BATCH = 8_000
DEFAULT_MAX_COLUMNS_PER_READ = 64
DEFAULT_MAX_OUTPUT_BYTES = 1 * 1024**3


class PackBStaticPointFeatureLoaderError(ValueError):
    """Raised when an exact, causal feature read cannot be proven."""


def _canonical_json_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, *, name: str) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
        raise PackBStaticPointFeatureLoaderError(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return normalized


def _require_git_sha(value: Any, *, name: str) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 40 or any(c not in "0123456789abcdef" for c in normalized):
        raise PackBStaticPointFeatureLoaderError(
            f"{name} must be a 40-character lowercase Git SHA"
        )
    return normalized


def _as_utc_series(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if values.isna().any():
        raise PackBStaticPointFeatureLoaderError(
            f"identity ledger column {column!r} has invalid timestamps"
        )
    return values


def _normalise_identity_ledger(identity_ledger: pd.DataFrame) -> pd.DataFrame:
    """Validate stable point identities while retaining their supplied order."""

    if not isinstance(identity_ledger, pd.DataFrame):
        raise PackBStaticPointFeatureLoaderError(
            "identity_ledger must be a pandas DataFrame"
        )
    missing = sorted(set(IDENTITY_COLUMNS) - set(identity_ledger.columns))
    if missing:
        raise PackBStaticPointFeatureLoaderError(
            "identity ledger misses required columns: " + ", ".join(missing)
        )
    if identity_ledger.empty:
        raise PackBStaticPointFeatureLoaderError("identity ledger is empty")

    ledger = identity_ledger.loc[:, list(IDENTITY_COLUMNS)].copy()
    candidate_ids = ledger["candidate_id"].astype("string")
    symbols = ledger["__symbol__"].astype("string")
    invalid_candidate = (
        candidate_ids.isna()
        | candidate_ids.str.strip().eq("")
        | candidate_ids.ne(candidate_ids.str.strip())
    )
    invalid_symbol = (
        symbols.isna() | symbols.str.strip().eq("") | symbols.ne(symbols.str.strip())
    )
    if invalid_candidate.any():
        raise PackBStaticPointFeatureLoaderError(
            "identity ledger has null, blank, or whitespace-padded candidate_id"
        )
    if invalid_symbol.any():
        raise PackBStaticPointFeatureLoaderError(
            "identity ledger has null, blank, or whitespace-padded __symbol__"
        )
    if candidate_ids.duplicated(keep=False).any():
        raise PackBStaticPointFeatureLoaderError(
            "identity ledger has duplicate candidate_id values"
        )

    ledger["candidate_id"] = candidate_ids.astype(str)
    ledger["__symbol__"] = symbols.astype(str)
    ledger["__ts__"] = _as_utc_series(ledger, "__ts__")
    exact_key = pd.MultiIndex.from_frame(ledger[["__symbol__", "__ts__"]])
    if exact_key.duplicated(keep=False).any():
        raise PackBStaticPointFeatureLoaderError(
            "identity ledger has duplicate exact (__symbol__, __ts__) keys"
        )
    ledger.insert(0, "__ledger_row__", np.arange(len(ledger), dtype=np.int64))
    return ledger


def _identity_stream_sha256(ledger: pd.DataFrame) -> str:
    normalized = _normalise_identity_ledger(ledger)
    payload = [
        [
            str(candidate_id),
            pd.Timestamp(timestamp).isoformat(),
            str(symbol),
        ]
        for candidate_id, timestamp, symbol in normalized.loc[
            :, list(IDENTITY_COLUMNS)
        ].itertuples(index=False, name=None)
    ]
    return _canonical_json_digest({"rows": payload})


def _feature_store_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise PackBStaticPointFeatureLoaderError(
            f"feature_store_dir must be an existing directory: {root}"
        )
    return root


def _symbol_feature_path(
    feature_store_dir: Path, symbol: str
) -> tuple[Path, bool] | None:
    """Resolve one canonical per-symbol file without guessing an arbitrary file."""

    exact = feature_store_dir / f"symbol={str(symbol).replace('/', '_')}.parquet"
    if exact.is_file():
        return exact, False
    candidates: list[Path] = []
    for alias in _symbol_alias_candidates(symbol):
        path = feature_store_dir / f"symbol={str(alias).replace('/', '_')}.parquet"
        if path.is_file() and path not in candidates:
            candidates.append(path)
    if not candidates:
        return None
    if len(candidates) != 1:
        raise PackBStaticPointFeatureLoaderError(
            "ambiguous canonical feature-store files for symbol "
            f"{symbol!r}: {', '.join(str(path.name) for path in candidates)}"
        )
    # Alias files are accepted only when a later exact read can prove their
    # stored ``__symbol__`` payload equals the requested ledger symbol.  This
    # prevents BTC/USD:USD from accidentally landing in a similarly named
    # spot/quote cache solely because it shares a convenient filename alias.
    return candidates[0], True


def _name_tokens(name: str) -> tuple[str, ...]:
    return tuple(token for token in re.split(r"[^a-z0-9]+", name.lower()) if token)


def _feature_rejection_reason(name: str) -> str | None:
    """Return a fail-closed reason for a non-causal/model-derived name.

    This is intentionally name-based and conservative.  The canonical store
    does not provide a complete machine-readable feature ontology today, so a
    column which looks like an identity, outcome, learned representation, or
    side feature is not eligible merely because it appears in a static Parquet
    schema.
    """

    raw = str(name)
    lowered = raw.lower()
    tokens = set(_name_tokens(raw))
    if raw in {"ts", "timestamp", "__ts__", "__symbol__"} or raw.startswith(
        "__index_level_"
    ):
        return "storage_identity"
    if (
        raw.startswith("__")
        or "candidate" in tokens
        or tokens
        & {
            "id",
            "uuid",
            "symbol",
            "instrument",
            "ticker",
            "exchange",
            "venue",
        }
    ):
        return "identifier"
    if tokens & {"side", "direction", "long", "short"} or lowered.startswith("side_"):
        return "side"
    # This deny list is deliberately narrow.  Generator/config provenance is
    # the primary safe allow-list; generic causal fields such as asset_* or
    # first-difference features must not be rejected on a substring guess.
    if tokens & {"target", "label", "outcome", "future", "realized", "pnl"}:
        return "outcome_or_future"
    # Compound names which do not split cleanly are still disallowed when they
    # identify known forward path targets.  Do not block ordinary historical
    # return, slope, volume, or volatility features.
    if any(
        fragment in lowered
        for fragment in (
            "future_",
            "label_",
            "target_",
            "realized_",
            "peak_mfe",
            "meaningful_mfe",
            "mae_before_",
        )
    ):
        return "outcome_or_future"
    return None


def _module_file_sha256(module: Any) -> str:
    path = Path(str(getattr(module, "__file__", "")))
    if not path.is_file():
        raise PackBStaticPointFeatureLoaderError(
            f"cannot bind feature-generator source for module {module!r}"
        )
    return _sha256_file(path)


def _provenance_backed_raw_allowlist(
    cfg: Mapping[str, Any] | None = None,
) -> tuple[frozenset[str], dict[str, str], str, str]:
    """Return raw market columns permitted by the current generator registry.

    Static Parquet schemas contain historical model outputs and old repair
    columns.  They are evidence of storage, not evidence that a field is a raw
    causal feature.  The current config/generator registry is the authority;
    this helper binds both the resulting allow-list and the source-file hashes
    used to derive it.
    """

    # Delayed imports avoid loading the broad training stack for callers which
    # only deserialize an already-frozen contract.
    from extreme_price_movements import config as epm_config
    from extreme_price_movements import features as epm_features
    from extreme_price_movements import pipeline_steps
    from extreme_price_movements.features_gmm_ae import AE_GMM_FEATURE_COLUMNS

    active_cfg = epm_config.CFG if cfg is None else cfg
    if not isinstance(active_cfg, Mapping):
        raise PackBStaticPointFeatureLoaderError("cfg must be a mapping when supplied")
    expected = {
        str(name)
        for name in pipeline_steps._expected_feature_keys_from_cfg(active_cfg)
        if isinstance(name, str) and name
    }
    model_derived = {
        str(name) for name in epm_config.MODEL_DERIVED_META_PERFORMANCE_FEATURE_KEYS
    }
    model_derived.update(str(name) for name in AE_GMM_FEATURE_COLUMNS)
    allowed: set[str] = set()
    rejected: dict[str, str] = {}
    for name in sorted(expected):
        if name in model_derived:
            rejected[name] = "model_derived_config_key"
            continue
        if name.startswith(("base_lgbm_", "meta_lgbm_", "resid_event_")):
            rejected[name] = "prior_model_output_prefix"
            continue
        if epm_config.is_non_portable_feature_key(name):
            rejected[name] = "non_portable_config_key"
            continue
        requirements = epm_features._feature_source_requirements(name)
        if "deleted" in requirements:
            rejected[name] = "deleted_or_live_impossible_generator_key"
            continue
        reason = _feature_rejection_reason(name)
        if reason is not None:
            rejected[name] = reason
            continue
        allowed.add(name)
    provenance = {
        "config_py_sha256": _module_file_sha256(epm_config),
        "features_py_sha256": _module_file_sha256(epm_features),
        "pipeline_steps_py_sha256": _module_file_sha256(pipeline_steps),
        "expected_feature_keys_sha256": _canonical_json_digest(
            {"expected_feature_keys": sorted(expected)}
        ),
        "raw_allowlist_sha256": _canonical_json_digest(
            {"raw_allowlist": sorted(allowed)}
        ),
        "model_derived_exclusion_sha256": _canonical_json_digest(
            {"model_derived": sorted(model_derived)}
        ),
    }
    provenance["generator_registry_sha256"] = _canonical_json_digest(provenance)
    return (
        frozenset(allowed),
        rejected,
        provenance["raw_allowlist_sha256"],
        provenance["generator_registry_sha256"],
    )


@dataclass(frozen=True)
class CandidateFeatureUniverse:
    """Fresh provenance-backed causal candidate universe for one ledger surface."""

    feature_store_dir: str
    symbols: tuple[str, ...]
    feature_columns: tuple[str, ...]
    rejected_columns: tuple[tuple[str, str], ...]
    missing_schema_symbols: tuple[str, ...]
    per_symbol_schema_sha256: tuple[tuple[str, str], ...]
    source_schema_sha256: str
    raw_allowlist_sha256: str
    generator_registry_sha256: str
    store_scan_manifest_sha256: str
    schema_evidence_sha256: str | None
    universe_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": POINT_FEATURE_LOADER_SCHEMA,
            "feature_store_dir": self.feature_store_dir,
            "symbols": list(self.symbols),
            "feature_columns": list(self.feature_columns),
            "rejected_columns": [list(value) for value in self.rejected_columns],
            "missing_schema_symbols": list(self.missing_schema_symbols),
            "per_symbol_schema_sha256": [
                list(value) for value in self.per_symbol_schema_sha256
            ],
            "source_schema_sha256": self.source_schema_sha256,
            "raw_allowlist_sha256": self.raw_allowlist_sha256,
            "generator_registry_sha256": self.generator_registry_sha256,
            "store_scan_manifest_sha256": self.store_scan_manifest_sha256,
            "schema_evidence_sha256": self.schema_evidence_sha256,
            "universe_sha256": self.universe_sha256,
            "selection_provenance": (
                "current_generator_registry_allowlist_intersected_with_"
                "fresh_store_schema_union"
            ),
        }


@dataclass(frozen=True)
class FeatureCoverageSegment:
    """Availability and learnability diagnostics for one named causal slice."""

    name: str
    sample_identity_sha256: str
    sampled_rows: int
    matched_exact_rows: int
    missing_exact_rows: int
    joint_complete_rows: int
    joint_complete_fraction: float
    feature_non_null_counts: tuple[tuple[str, int], ...]
    feature_non_null_fractions: tuple[tuple[str, float], ...]
    feature_variances: tuple[tuple[str, float], ...]
    feature_unique_count_capped: tuple[tuple[str, int], ...]
    feature_binary_prevalence: tuple[tuple[str, float | None], ...]
    missing_symbols: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sample_identity_sha256": self.sample_identity_sha256,
            "sampled_rows": self.sampled_rows,
            "matched_exact_rows": self.matched_exact_rows,
            "missing_exact_rows": self.missing_exact_rows,
            "joint_complete_rows": self.joint_complete_rows,
            "joint_complete_fraction": self.joint_complete_fraction,
            "feature_non_null_counts": [
                list(value) for value in self.feature_non_null_counts
            ],
            "feature_non_null_fractions": [
                [name, fraction] for name, fraction in self.feature_non_null_fractions
            ],
            "feature_variances": [list(value) for value in self.feature_variances],
            "feature_unique_count_capped": [
                list(value) for value in self.feature_unique_count_capped
            ],
            "feature_binary_prevalence": [
                list(value) for value in self.feature_binary_prevalence
            ],
            "missing_symbols": list(self.missing_symbols),
        }


@dataclass(frozen=True)
class FeatureCoverageProfile:
    """Deterministic availability evidence, not target-based feature selection."""

    sample_identity_sha256: str
    sampled_rows: int
    matched_exact_rows: int
    missing_exact_rows: int
    feature_non_null_counts: tuple[tuple[str, int], ...]
    feature_non_null_fractions: tuple[tuple[str, float], ...]
    missing_symbols: tuple[str, ...]
    segments: tuple[FeatureCoverageSegment, ...]
    profile_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": POINT_FEATURE_LOADER_SCHEMA,
            "sample_identity_sha256": self.sample_identity_sha256,
            "sampled_rows": self.sampled_rows,
            "matched_exact_rows": self.matched_exact_rows,
            "missing_exact_rows": self.missing_exact_rows,
            "feature_non_null_counts": [
                list(value) for value in self.feature_non_null_counts
            ],
            "feature_non_null_fractions": [
                [name, fraction] for name, fraction in self.feature_non_null_fractions
            ],
            "missing_symbols": list(self.missing_symbols),
            "segments": [segment.to_dict() for segment in self.segments],
            "profile_sha256": self.profile_sha256,
            "selection_provenance": "deterministic_coverage_only_no_targets_or_importance",
        }


@dataclass(frozen=True)
class FrozenFeatureContract:
    """Ordered training input contract derived only from schema and coverage."""

    feature_columns: tuple[str, ...]
    candidate_universe_sha256: str
    source_schema_sha256: str
    raw_allowlist_sha256: str
    generator_registry_sha256: str
    store_scan_manifest_sha256: str
    coverage_profile_sha256: str | None
    min_exact_key_coverage: float
    min_non_null_feature_coverage: float
    max_feature_columns: int | None
    coverage_admission_rejections: tuple[tuple[str, str], ...]
    feature_contract_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": POINT_FEATURE_LOADER_SCHEMA,
            "feature_columns": list(self.feature_columns),
            "candidate_universe_sha256": self.candidate_universe_sha256,
            "source_schema_sha256": self.source_schema_sha256,
            "raw_allowlist_sha256": self.raw_allowlist_sha256,
            "generator_registry_sha256": self.generator_registry_sha256,
            "store_scan_manifest_sha256": self.store_scan_manifest_sha256,
            "coverage_profile_sha256": self.coverage_profile_sha256,
            "min_exact_key_coverage": self.min_exact_key_coverage,
            "min_non_null_feature_coverage": self.min_non_null_feature_coverage,
            "max_feature_columns": self.max_feature_columns,
            "coverage_admission_rejections": [
                list(value) for value in self.coverage_admission_rejections
            ],
            "feature_contract_sha256": self.feature_contract_sha256,
            "selection_provenance": (
                "current_generator_registry_allowlist_and_deterministic_coverage_only"
            ),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FrozenFeatureContract":
        if value.get("schema") != POINT_FEATURE_LOADER_SCHEMA:
            raise PackBStaticPointFeatureLoaderError(
                "frozen feature contract has an unsupported schema"
            )
        columns_value = value.get("feature_columns")
        if not isinstance(columns_value, list) or not columns_value:
            raise PackBStaticPointFeatureLoaderError(
                "frozen feature contract has no ordered feature_columns"
            )
        columns = tuple(str(column) for column in columns_value)
        if tuple(sorted(columns)) != columns or len(set(columns)) != len(columns):
            raise PackBStaticPointFeatureLoaderError(
                "frozen feature contract columns must be unique and sorted"
            )
        contract = cls(
            feature_columns=columns,
            candidate_universe_sha256=_require_sha256(
                value.get("candidate_universe_sha256"),
                name="candidate_universe_sha256",
            ),
            source_schema_sha256=_require_sha256(
                value.get("source_schema_sha256"), name="source_schema_sha256"
            ),
            raw_allowlist_sha256=_require_sha256(
                value.get("raw_allowlist_sha256"), name="raw_allowlist_sha256"
            ),
            generator_registry_sha256=_require_sha256(
                value.get("generator_registry_sha256"),
                name="generator_registry_sha256",
            ),
            store_scan_manifest_sha256=_require_sha256(
                value.get("store_scan_manifest_sha256"),
                name="store_scan_manifest_sha256",
            ),
            coverage_profile_sha256=(
                _require_sha256(
                    value.get("coverage_profile_sha256"),
                    name="coverage_profile_sha256",
                )
                if value.get("coverage_profile_sha256") is not None
                else None
            ),
            min_exact_key_coverage=float(value.get("min_exact_key_coverage")),
            min_non_null_feature_coverage=float(
                value.get("min_non_null_feature_coverage")
            ),
            max_feature_columns=(
                int(value.get("max_feature_columns"))
                if value.get("max_feature_columns") is not None
                else None
            ),
            coverage_admission_rejections=tuple(
                (str(item[0]), str(item[1]))
                for item in value.get("coverage_admission_rejections", [])
                if isinstance(item, (list, tuple)) and len(item) == 2
            ),
            feature_contract_sha256=_require_sha256(
                value.get("feature_contract_sha256"), name="feature_contract_sha256"
            ),
        )
        if (
            contract.max_feature_columns is not None
            and contract.max_feature_columns < 1
        ):
            raise PackBStaticPointFeatureLoaderError(
                "max_feature_columns must be positive or null"
            )
        expected = _feature_contract_digest(
            feature_columns=contract.feature_columns,
            candidate_universe_sha256=contract.candidate_universe_sha256,
            source_schema_sha256=contract.source_schema_sha256,
            raw_allowlist_sha256=contract.raw_allowlist_sha256,
            generator_registry_sha256=contract.generator_registry_sha256,
            store_scan_manifest_sha256=contract.store_scan_manifest_sha256,
            coverage_profile_sha256=contract.coverage_profile_sha256,
            min_exact_key_coverage=contract.min_exact_key_coverage,
            min_non_null_feature_coverage=contract.min_non_null_feature_coverage,
            max_feature_columns=contract.max_feature_columns,
            coverage_admission_rejections=contract.coverage_admission_rejections,
        )
        if expected != contract.feature_contract_sha256:
            raise PackBStaticPointFeatureLoaderError(
                "frozen feature contract SHA-256 does not match its content"
            )
        return contract


@dataclass(frozen=True)
class PointFeatureBatch:
    """One bounded exact join result from :func:`iter_point_in_time_feature_batches`."""

    ledger_row_positions: np.ndarray
    identity: pd.DataFrame
    features: pd.DataFrame
    matched_exact_keys: np.ndarray


def _schema_payload(
    *,
    feature_store_dir: Path,
    symbol_schemas: Mapping[str, Sequence[str]],
    evidence_sha256: str | None,
    store_scan_manifest_sha256: str,
    raw_allowlist_sha256: str,
    generator_registry_sha256: str,
) -> dict[str, Any]:
    return {
        "feature_store_dir": str(feature_store_dir),
        "symbol_schemas": {
            str(symbol): list(sorted(str(column) for column in columns))
            for symbol, columns in sorted(symbol_schemas.items())
        },
        "schema_evidence_sha256": evidence_sha256,
        "store_scan_manifest_sha256": store_scan_manifest_sha256,
        "raw_allowlist_sha256": raw_allowlist_sha256,
        "generator_registry_sha256": generator_registry_sha256,
    }


def _bound_store_scan_manifest_sha256(feature_store_dir: Path) -> str:
    """Bind the canonical cache scan evidence to the raw-universe contract."""

    path = feature_store_dir / "_feature_cache_scan_manifest.json"
    if not path.is_file():
        raise PackBStaticPointFeatureLoaderError(
            f"canonical feature store has no _feature_cache_scan_manifest.json: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackBStaticPointFeatureLoaderError(
            f"cannot parse canonical feature-store scan manifest: {path}"
        ) from exc
    if not isinstance(payload, dict) or not isinstance(
        payload.get("input_signature"), dict
    ):
        raise PackBStaticPointFeatureLoaderError(
            "canonical feature-store scan manifest has no valid input_signature"
        )
    return _sha256_file(path)


def discover_causal_feature_universe(
    identity_ledger: pd.DataFrame,
    *,
    feature_store_dir: str | Path,
    schema_evidence_path: str | Path | None = None,
    cfg: Mapping[str, Any] | None = None,
    coverage_discovery: bool = False,
    resource_guard: TrainingResourceGuard | Any | None = None,
) -> CandidateFeatureUniverse:
    """Discover a fresh provenance-backed causal feature universe for this ledger.

    ``coverage_discovery=False`` is the production default: every required
    symbol needs exactly one canonical feature file.  In discovery mode a
    missing symbol is retained as coverage evidence, allowing callers to
    inspect coverage before refusing to freeze a trainable contract.
    """

    ledger = _normalise_identity_ledger(identity_ledger)
    root = _feature_store_root(feature_store_dir)
    guard = resource_guard or TrainingResourceGuard(disk_path=root)
    guard.preflight("packb_static_point_features:discover_schema")
    (
        raw_allowlist,
        configured_rejections,
        raw_allowlist_sha256,
        generator_registry_sha256,
    ) = _provenance_backed_raw_allowlist(cfg)
    store_scan_manifest_sha256 = _bound_store_scan_manifest_sha256(root)
    evidence_sha256: str | None = None
    if schema_evidence_path is not None:
        evidence_path = Path(schema_evidence_path)
        if not evidence_path.is_file():
            raise PackBStaticPointFeatureLoaderError(
                f"schema_evidence_path is not a file: {evidence_path}"
            )
        evidence_sha256 = _sha256_file(evidence_path)

    symbols = tuple(sorted(ledger["__symbol__"].unique().tolist()))
    schemas: dict[str, tuple[str, ...]] = {}
    missing_symbols: list[str] = []
    for position, symbol in enumerate(symbols, start=1):
        resolved = _symbol_feature_path(root, symbol)
        if resolved is None:
            missing_symbols.append(symbol)
            if not coverage_discovery:
                raise PackBStaticPointFeatureLoaderError(
                    f"no canonical feature-store file exists for symbol {symbol!r}"
                )
            continue
        path, _is_alias = resolved
        schema = tuple(
            sorted(str(column) for column in _feature_schema_names(str(path)))
        )
        if not schema:
            raise PackBStaticPointFeatureLoaderError(
                f"canonical feature-store schema is unavailable for {path}"
            )
        if "__symbol__" not in schema:
            raise PackBStaticPointFeatureLoaderError(
                "canonical feature-store schema must retain __symbol__ for exact "
                f"symbol verification: {path}"
            )
        schemas[symbol] = schema
        if position == 1 or position % 25 == 0:
            guard.checkpoint("packb_static_point_features:discover_schema")
    guard.checkpoint("packb_static_point_features:discover_schema_complete")
    if not schemas:
        raise PackBStaticPointFeatureLoaderError(
            "no requested symbol has a readable canonical feature-store schema"
        )

    schema_union = set().union(*(set(values) for values in schemas.values()))
    accepted: list[str] = []
    rejected: dict[str, str] = dict(configured_rejections)
    # Store fields are accepted only when the current generator/config registry
    # proves them raw and causal.  Do not take a schema intersection here:
    # variable listing histories should be surfaced by point-in-time coverage,
    # not silently erase a useful causal feature from the candidate universe.
    for column in sorted(schema_union):
        reason = _feature_rejection_reason(column)
        if reason is not None:
            rejected[column] = reason
        elif column.startswith(("base_lgbm_", "meta_lgbm_", "resid_event_")):
            rejected[column] = "prior_model_output_prefix"
        elif column in raw_allowlist:
            accepted.append(column)
        else:
            rejected[column] = "not_current_generator_registry_allowlist"
    if not accepted:
        raise PackBStaticPointFeatureLoaderError(
            "fresh feature-store schema has no causal non-identity candidate columns"
        )
    source_payload = _schema_payload(
        feature_store_dir=root,
        symbol_schemas=schemas,
        evidence_sha256=evidence_sha256,
        store_scan_manifest_sha256=store_scan_manifest_sha256,
        raw_allowlist_sha256=raw_allowlist_sha256,
        generator_registry_sha256=generator_registry_sha256,
    )
    source_schema_sha256 = _canonical_json_digest(source_payload)
    universe_payload = {
        "schema": POINT_FEATURE_LOADER_SCHEMA,
        "source_schema_sha256": source_schema_sha256,
        "symbols": list(symbols),
        "feature_columns": accepted,
        "rejected_columns": sorted(rejected.items()),
        "missing_schema_symbols": sorted(missing_symbols),
        "raw_allowlist_sha256": raw_allowlist_sha256,
        "generator_registry_sha256": generator_registry_sha256,
        "store_scan_manifest_sha256": store_scan_manifest_sha256,
    }
    return CandidateFeatureUniverse(
        feature_store_dir=str(root),
        symbols=symbols,
        feature_columns=tuple(accepted),
        rejected_columns=tuple(sorted(rejected.items())),
        missing_schema_symbols=tuple(sorted(missing_symbols)),
        per_symbol_schema_sha256=tuple(
            (symbol, _canonical_json_digest({"columns": list(columns)}))
            for symbol, columns in sorted(schemas.items())
        ),
        source_schema_sha256=source_schema_sha256,
        raw_allowlist_sha256=raw_allowlist_sha256,
        generator_registry_sha256=generator_registry_sha256,
        store_scan_manifest_sha256=store_scan_manifest_sha256,
        schema_evidence_sha256=evidence_sha256,
        universe_sha256=_canonical_json_digest(universe_payload),
    )


def _deterministic_coverage_sample(
    ledger: pd.DataFrame, *, max_rows: int
) -> pd.DataFrame:
    if int(max_rows) < 1:
        raise PackBStaticPointFeatureLoaderError(
            "coverage_sample_rows must be positive"
        )
    if len(ledger) <= int(max_rows):
        return ledger.drop(columns=["__ledger_row__"]).copy()
    ordering = pd.DataFrame(
        {
            "candidate_id": ledger["candidate_id"].astype(str),
            "digest": ledger["candidate_id"]
            .astype(str)
            .map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()),
            "row": ledger["__ledger_row__"].to_numpy(dtype=np.int64, copy=False),
        }
    )
    positions = (
        ordering.sort_values(["digest", "candidate_id", "row"], kind="stable")
        .head(int(max_rows))["row"]
        .to_numpy(dtype=np.int64, copy=False)
    )
    # The sampled ledger keeps source order; this prevents sampling itself from
    # creating a hidden time/symbol reordering for downstream diagnostics.
    return ledger.iloc[np.sort(positions)].drop(columns=["__ledger_row__"]).copy()


def _require_contract(
    value: FrozenFeatureContract | Mapping[str, Any],
) -> FrozenFeatureContract:
    if isinstance(value, FrozenFeatureContract):
        # Round-trip through the content validation to catch manually-created
        # dataclasses with a stale digest too.
        return FrozenFeatureContract.from_mapping(value.to_dict())
    if isinstance(value, Mapping):
        return FrozenFeatureContract.from_mapping(value)
    raise PackBStaticPointFeatureLoaderError(
        "feature_contract must be FrozenFeatureContract or a validated mapping"
    )


def _validate_contract_against_universe(
    contract: FrozenFeatureContract,
    universe: CandidateFeatureUniverse,
) -> None:
    if contract.source_schema_sha256 != universe.source_schema_sha256:
        raise PackBStaticPointFeatureLoaderError(
            "canonical feature-store schema changed after the frozen contract was created"
        )
    if contract.candidate_universe_sha256 != universe.universe_sha256:
        raise PackBStaticPointFeatureLoaderError(
            "fresh causal candidate universe changed after the frozen contract was created"
        )
    for name in (
        "raw_allowlist_sha256",
        "generator_registry_sha256",
        "store_scan_manifest_sha256",
    ):
        if getattr(contract, name) != getattr(universe, name):
            raise PackBStaticPointFeatureLoaderError(
                f"frozen feature contract {name} no longer matches its raw-universe evidence"
            )
    missing = sorted(set(contract.feature_columns) - set(universe.feature_columns))
    if missing:
        raise PackBStaticPointFeatureLoaderError(
            "frozen feature contract contains columns absent from the fresh causal universe: "
            + ", ".join(missing[:8])
        )
    blocked = [
        name for name in contract.feature_columns if _feature_rejection_reason(name)
    ]
    if blocked:
        raise PackBStaticPointFeatureLoaderError(
            "frozen feature contract contains prohibited identity, side, model-derived, "
            "or outcome columns: " + ", ".join(blocked[:8])
        )


def _numeric_feature_frame(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    symbol: str,
) -> pd.DataFrame:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise PackBStaticPointFeatureLoaderError(
            f"canonical feature read for {symbol!r} misses contract columns: "
            + ", ".join(missing[:8])
        )
    out = frame.loc[:, list(columns)].copy()
    invalid: list[str] = []
    for column in columns:
        values = out[column]
        if not (
            pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values)
        ):
            # A model input may not acquire semantics from a lossy string
            # coercion.  Categorical/string payloads require an explicit,
            # frozen upstream transform and are therefore rejected here.
            if values.notna().any():
                invalid.append(str(column))
            out[column] = np.nan
    if invalid:
        raise PackBStaticPointFeatureLoaderError(
            f"canonical feature read for {symbol!r} has non-numeric model columns: "
            + ", ".join(invalid[:8])
        )
    try:
        return out.astype(np.float32, copy=False)
    except (TypeError, ValueError) as exc:
        raise PackBStaticPointFeatureLoaderError(
            f"canonical feature read for {symbol!r} cannot be represented as float32"
        ) from exc


def _column_blocks(
    columns: Sequence[str], *, max_columns: int
) -> Iterator[tuple[str, ...]]:
    if int(max_columns) < 1 or int(max_columns) > DEFAULT_MAX_COLUMNS_PER_READ:
        raise PackBStaticPointFeatureLoaderError(
            f"max_columns_per_read must be in [1, {DEFAULT_MAX_COLUMNS_PER_READ}]"
        )
    for offset in range(0, len(columns), int(max_columns)):
        yield tuple(columns[offset : offset + int(max_columns)])


def _monthly_row_blocks(
    symbol_ledger: pd.DataFrame, *, max_rows: int
) -> Iterator[pd.DataFrame]:
    """Bound source scans to one UTC calendar month and a row cap."""

    if int(max_rows) < 1:
        raise PackBStaticPointFeatureLoaderError("max_rows_per_batch must be positive")
    keyed = symbol_ledger.copy()
    keyed["__month__"] = keyed["__ts__"].dt.strftime("%Y-%m")
    for _month, month_rows in keyed.groupby("__month__", sort=True, observed=True):
        month_rows = month_rows.drop(columns=["__month__"])
        for offset in range(0, len(month_rows), int(max_rows)):
            yield month_rows.iloc[offset : offset + int(max_rows)].copy()


def _verify_store_symbol_payload(
    source: pd.DataFrame, *, symbol: str, path: Path
) -> None:
    if "__symbol__" not in source.columns:
        raise PackBStaticPointFeatureLoaderError(
            "canonical feature file does not retain __symbol__ for verification: "
            f"{path}"
        )
    observed = source["__symbol__"].dropna().astype(str).str.strip()
    if observed.empty or not observed.eq(str(symbol)).all():
        preview = sorted(observed.unique().tolist())[:3]
        raise PackBStaticPointFeatureLoaderError(
            "canonical feature-store file has a mismatched stored "
            f"__symbol__ for {symbol!r}: {preview} ({path.name})"
        )


def iter_point_in_time_feature_batches(
    identity_ledger: pd.DataFrame,
    *,
    feature_store_dir: str | Path,
    feature_contract: FrozenFeatureContract | Mapping[str, Any],
    max_rows_per_batch: int = DEFAULT_MAX_ROWS_PER_BATCH,
    max_columns_per_read: int = DEFAULT_MAX_COLUMNS_PER_READ,
    coverage_discovery: bool = False,
    verify_frozen_schema: bool = True,
    resource_guard: TrainingResourceGuard | Any | None = None,
) -> Iterator[PointFeatureBatch]:
    """Yield exact joins in bounded symbol/batch reads.

    In production mode each requested ledger key must have one exact feature
    row.  ``coverage_discovery=True`` is the sole exception: it produces
    explicit ``matched_exact_keys`` masks and NaNs for unavailable rows so a
    caller can quantify availability before freezing a contract.  It never
    fills from a past or future timestamp.
    """

    if int(max_rows_per_batch) < 1:
        raise PackBStaticPointFeatureLoaderError("max_rows_per_batch must be positive")
    # Validate before any reads, even when a zero-column contract is rejected
    # by the frozen-contract parser above.
    tuple(_column_blocks(("_probe",), max_columns=max_columns_per_read))
    ledger = _normalise_identity_ledger(identity_ledger)
    root = _feature_store_root(feature_store_dir)
    contract = _require_contract(feature_contract)
    guard = resource_guard or TrainingResourceGuard(disk_path=root)
    guard.preflight("packb_static_point_features:point_load")
    if verify_frozen_schema and not coverage_discovery:
        current_universe = discover_causal_feature_universe(
            ledger.drop(columns=["__ledger_row__"]),
            feature_store_dir=root,
            coverage_discovery=coverage_discovery,
            resource_guard=guard,
        )
        _validate_contract_against_universe(contract, current_universe)

    symbol_groups = [
        (str(symbol), group.copy())
        for symbol, group in ledger.groupby("__symbol__", sort=True, observed=True)
    ]
    for symbol_position, (symbol, group) in enumerate(symbol_groups, start=1):
        resolved = _symbol_feature_path(root, symbol)
        for chunk in _monthly_row_blocks(group, max_rows=int(max_rows_per_batch)):
            requested_ts = pd.DatetimeIndex(chunk["__ts__"], name="ts")
            matched = np.zeros(len(chunk), dtype=bool)
            values = pd.DataFrame(
                np.nan,
                index=np.arange(len(chunk)),
                columns=list(contract.feature_columns),
                dtype=np.float32,
            )
            if resolved is None:
                if not coverage_discovery:
                    raise PackBStaticPointFeatureLoaderError(
                        f"no canonical feature-store file exists for symbol {symbol!r}"
                    )
            else:
                path, _is_alias = resolved
                for block_number, block in enumerate(
                    _column_blocks(
                        contract.feature_columns,
                        max_columns=int(max_columns_per_read),
                    )
                ):
                    # ``__symbol__`` is a narrow canonical key anchor.  It
                    # makes exact-key coverage independent of a feature which
                    # first appeared in a later repair delta.
                    source = read_symbol_features(
                        str(path),
                        columns=["__symbol__", *block],
                        start_ts=requested_ts.min(),
                        end_ts=requested_ts.max(),
                    )
                    if source.empty:
                        current_matched = np.zeros(len(chunk), dtype=bool)
                    else:
                        source.index = pd.to_datetime(
                            source.index, utc=True, errors="coerce"
                        )
                        source = source.loc[source.index.notna()]
                        if not source.index.is_unique:
                            raise PackBStaticPointFeatureLoaderError(
                                f"canonical feature read for {symbol!r} has duplicate timestamps"
                            )
                        source = source.sort_index()
                        _verify_store_symbol_payload(source, symbol=symbol, path=path)
                        current_matched = np.asarray(
                            requested_ts.isin(source.index), dtype=bool
                        )
                    if block_number == 0:
                        # First column block establishes store-key coverage.
                        matched = current_matched
                    elif not np.array_equal(matched, current_matched):
                        raise PackBStaticPointFeatureLoaderError(
                            "canonical __symbol__ key anchor changed across column "
                            f"blocks for {symbol!r}"
                        )
                    if not bool(current_matched.any()) or source.empty:
                        continue
                    available = [column for column in block if column in source.columns]
                    if not available:
                        continue
                    exact = source.reindex(requested_ts[current_matched])
                    values.loc[current_matched, available] = _numeric_feature_frame(
                        exact,
                        available,
                        symbol=symbol,
                    ).to_numpy(dtype=np.float32, copy=False)
            if not coverage_discovery and not bool(matched.all()):
                absent = chunk.loc[~matched, ["candidate_id", "__ts__", "__symbol__"]]
                preview = ", ".join(
                    f"{candidate_id}@{pd.Timestamp(timestamp).isoformat()}"
                    for candidate_id, timestamp, _symbol in absent.head(3).itertuples(
                        index=False, name=None
                    )
                )
                raise PackBStaticPointFeatureLoaderError(
                    "missing exact canonical feature-store keys for "
                    f"{symbol!r}; no as-of/future fallback is permitted: {preview}"
                )
            identity = chunk.loc[:, list(IDENTITY_COLUMNS)].reset_index(drop=True)
            yield PointFeatureBatch(
                ledger_row_positions=chunk["__ledger_row__"].to_numpy(
                    dtype=np.int64, copy=True
                ),
                identity=identity,
                features=values.reset_index(drop=True),
                matched_exact_keys=matched,
            )
            guard.checkpoint("packb_static_point_features:point_load_batch")
        if symbol_position == 1 or symbol_position % 25 == 0:
            guard.checkpoint("packb_static_point_features:point_load_symbol")
    guard.checkpoint("packb_static_point_features:point_load_complete")


def _profile_coverage_segment(
    name: str,
    ledger: pd.DataFrame,
    *,
    feature_store_dir: str | Path,
    contract: FrozenFeatureContract,
    coverage_sample_rows: int,
    max_rows_per_batch: int,
    max_columns_per_read: int,
    resource_guard: TrainingResourceGuard | Any | None,
) -> FeatureCoverageSegment:
    """Compute bounded coverage/variance/binary diagnostics for one slice."""

    if not str(name).strip():
        raise PackBStaticPointFeatureLoaderError(
            "coverage segment names must be non-empty"
        )
    normalized = _normalise_identity_ledger(ledger)
    sample = _deterministic_coverage_sample(normalized, max_rows=coverage_sample_rows)
    count = len(contract.feature_columns)
    finite_counts = np.zeros(count, dtype=np.int64)
    sums = np.zeros(count, dtype=np.float64)
    squared_sums = np.zeros(count, dtype=np.float64)
    binary_possible = np.ones(count, dtype=bool)
    binary_positive = np.zeros(count, dtype=np.int64)
    unique_values: list[set[float]] = [set() for _ in range(count)]
    matched_total = 0
    joint_complete_rows = 0
    missing_symbols: set[str] = set()
    for batch in iter_point_in_time_feature_batches(
        sample,
        feature_store_dir=feature_store_dir,
        feature_contract=contract,
        max_rows_per_batch=max_rows_per_batch,
        max_columns_per_read=max_columns_per_read,
        coverage_discovery=True,
        resource_guard=resource_guard,
    ):
        matched_total += int(batch.matched_exact_keys.sum())
        if not bool(batch.matched_exact_keys.all()):
            missing_symbols.update(
                batch.identity.loc[~batch.matched_exact_keys, "__symbol__"].astype(str)
            )
        matrix = batch.features.to_numpy(dtype=np.float32, copy=False)
        finite = np.isfinite(matrix)
        joint_complete_rows += int(finite.all(axis=1).sum())
        finite_counts += finite.sum(axis=0, dtype=np.int64)
        safe = np.where(finite, matrix, 0.0).astype(np.float64, copy=False)
        sums += safe.sum(axis=0, dtype=np.float64)
        squared_sums += np.square(safe, dtype=np.float64).sum(axis=0, dtype=np.float64)
        for position in range(count):
            values = matrix[finite[:, position], position]
            if not len(values):
                continue
            if binary_possible[position] and not np.isin(values, (0.0, 1.0)).all():
                binary_possible[position] = False
            if binary_possible[position]:
                binary_positive[position] += int(np.count_nonzero(values == 1.0))
            if len(unique_values[position]) < 3:
                unique_values[position].update(
                    float(value) for value in np.unique(values)[:3]
                )
                if len(unique_values[position]) > 3:
                    unique_values[position] = set(sorted(unique_values[position])[:3])
    fractions = finite_counts.astype(float) / float(len(sample))
    variances = np.full(count, np.nan, dtype=np.float64)
    present = finite_counts > 0
    variances[present] = squared_sums[present] / finite_counts[present] - np.square(
        sums[present] / finite_counts[present]
    )
    variances[present] = np.maximum(variances[present], 0.0)
    binary_prevalence: list[float | None] = []
    for position in range(count):
        if binary_possible[position] and finite_counts[position] > 0:
            binary_prevalence.append(
                float(binary_positive[position] / finite_counts[position])
            )
        else:
            binary_prevalence.append(None)
    return FeatureCoverageSegment(
        name=str(name),
        sample_identity_sha256=_identity_stream_sha256(sample),
        sampled_rows=int(len(sample)),
        matched_exact_rows=int(matched_total),
        missing_exact_rows=int(len(sample) - matched_total),
        joint_complete_rows=int(joint_complete_rows),
        joint_complete_fraction=float(joint_complete_rows / len(sample)),
        feature_non_null_counts=tuple(
            (column, int(value))
            for column, value in zip(
                contract.feature_columns, finite_counts, strict=True
            )
        ),
        feature_non_null_fractions=tuple(
            (column, float(value))
            for column, value in zip(contract.feature_columns, fractions, strict=True)
        ),
        feature_variances=tuple(
            (column, float(value))
            for column, value in zip(contract.feature_columns, variances, strict=True)
        ),
        feature_unique_count_capped=tuple(
            (column, min(3, len(values)))
            for column, values in zip(
                contract.feature_columns, unique_values, strict=True
            )
        ),
        feature_binary_prevalence=tuple(
            (column, value)
            for column, value in zip(
                contract.feature_columns, binary_prevalence, strict=True
            )
        ),
        missing_symbols=tuple(sorted(missing_symbols)),
    )


def profile_point_feature_coverage(
    identity_ledger: pd.DataFrame,
    *,
    feature_store_dir: str | Path,
    feature_contract: FrozenFeatureContract | Mapping[str, Any],
    coverage_sample_rows: int = 20_000,
    coverage_segments: Mapping[str, pd.DataFrame] | None = None,
    max_rows_per_batch: int = DEFAULT_MAX_ROWS_PER_BATCH,
    max_columns_per_read: int = DEFAULT_MAX_COLUMNS_PER_READ,
    resource_guard: TrainingResourceGuard | Any | None = None,
) -> FeatureCoverageProfile:
    """Profile exact keys and learnability in global and named causal slices.

    ``coverage_segments`` is intended for the side-local AE beginning/middle/end
    reference slices or FS/HPO train/validation slices.  Segment names and
    input ledgers are caller-supplied, so this loader never infers them from
    outcomes.  Each segment gets its own deterministic identity-hash sample.
    """

    contract = _require_contract(feature_contract)
    overall = _profile_coverage_segment(
        "all",
        identity_ledger,
        feature_store_dir=feature_store_dir,
        contract=contract,
        coverage_sample_rows=coverage_sample_rows,
        max_rows_per_batch=max_rows_per_batch,
        max_columns_per_read=max_columns_per_read,
        resource_guard=resource_guard,
    )
    named_segments: list[FeatureCoverageSegment] = []
    for name, segment_ledger in sorted((coverage_segments or {}).items()):
        if str(name) == "all":
            raise PackBStaticPointFeatureLoaderError(
                "coverage_segments cannot redefine reserved segment name 'all'"
            )
        named_segments.append(
            _profile_coverage_segment(
                str(name),
                segment_ledger,
                feature_store_dir=feature_store_dir,
                contract=contract,
                coverage_sample_rows=coverage_sample_rows,
                max_rows_per_batch=max_rows_per_batch,
                max_columns_per_read=max_columns_per_read,
                resource_guard=resource_guard,
            )
        )
    profile_payload = {
        "sample_identity_sha256": overall.sample_identity_sha256,
        "sampled_rows": overall.sampled_rows,
        "matched_exact_rows": overall.matched_exact_rows,
        "missing_exact_rows": overall.missing_exact_rows,
        "feature_non_null_counts": [
            list(value) for value in overall.feature_non_null_counts
        ],
        "feature_non_null_fractions": [
            list(value) for value in overall.feature_non_null_fractions
        ],
        "missing_symbols": list(overall.missing_symbols),
        "segments": [segment.to_dict() for segment in named_segments],
    }
    return FeatureCoverageProfile(
        sample_identity_sha256=overall.sample_identity_sha256,
        sampled_rows=overall.sampled_rows,
        matched_exact_rows=overall.matched_exact_rows,
        missing_exact_rows=overall.missing_exact_rows,
        feature_non_null_counts=overall.feature_non_null_counts,
        feature_non_null_fractions=overall.feature_non_null_fractions,
        missing_symbols=overall.missing_symbols,
        segments=tuple(named_segments),
        profile_sha256=_canonical_json_digest(profile_payload),
    )


def _validate_coverage_threshold(value: float, *, name: str) -> float:
    normalized = float(value)
    if not 0.0 <= normalized <= 1.0:
        raise PackBStaticPointFeatureLoaderError(f"{name} must be in [0, 1]")
    return normalized


def validate_feature_coverage_gates(
    profile: FeatureCoverageProfile,
    *,
    feature_columns: Sequence[str],
    min_segment_exact_key_coverage: float | None = None,
    min_segment_non_null_feature_coverage: float | None = None,
    min_segment_joint_complete_coverage: float | None = None,
    min_variance: float | None = None,
    binary_prevalence_bounds: tuple[float, float] | None = None,
    required_segment_names: Sequence[str] = (),
) -> None:
    """Fail closed on named-slice availability and basic learnability gates.

    Typical calls are ``0.99`` coverage / ``1e-6`` variance / ``[.005, .995]``
    for AE beginning-middle-end reference slices and ``0.98`` coverage for
    FS/HPO train-validation slices.  These checks use only feature values and
    point identities; no label or future-path information enters the gate.
    """

    selected = tuple(str(value) for value in feature_columns)
    if not selected:
        raise PackBStaticPointFeatureLoaderError(
            "coverage gates require feature columns"
        )
    exact_threshold = (
        _validate_coverage_threshold(
            min_segment_exact_key_coverage, name="min_segment_exact_key_coverage"
        )
        if min_segment_exact_key_coverage is not None
        else None
    )
    non_null_threshold = (
        _validate_coverage_threshold(
            min_segment_non_null_feature_coverage,
            name="min_segment_non_null_feature_coverage",
        )
        if min_segment_non_null_feature_coverage is not None
        else None
    )
    joint_threshold = (
        _validate_coverage_threshold(
            min_segment_joint_complete_coverage,
            name="min_segment_joint_complete_coverage",
        )
        if min_segment_joint_complete_coverage is not None
        else None
    )
    if min_variance is not None and float(min_variance) < 0.0:
        raise PackBStaticPointFeatureLoaderError("min_variance must be non-negative")
    bounds: tuple[float, float] | None = None
    if binary_prevalence_bounds is not None:
        lower, upper = map(float, binary_prevalence_bounds)
        if not 0.0 <= lower <= upper <= 1.0:
            raise PackBStaticPointFeatureLoaderError(
                "binary_prevalence_bounds must be ordered values in [0, 1]"
            )
        bounds = (lower, upper)
    segments = {segment.name: segment for segment in profile.segments}
    missing_segments = sorted(set(map(str, required_segment_names)) - set(segments))
    if missing_segments:
        raise PackBStaticPointFeatureLoaderError(
            "coverage profile lacks required named slices: "
            + ", ".join(missing_segments)
        )
    errors: list[str] = []
    for name, segment in sorted(segments.items()):
        if segment.sampled_rows < 1:
            errors.append(f"{name}: no sampled rows")
            continue
        exact_fraction = segment.matched_exact_rows / segment.sampled_rows
        if exact_threshold is not None and exact_fraction < exact_threshold:
            errors.append(
                f"{name}: exact coverage {exact_fraction:.6f} < {exact_threshold:.6f}"
            )
        if (
            joint_threshold is not None
            and segment.joint_complete_fraction < joint_threshold
        ):
            errors.append(
                f"{name}: joint complete coverage {segment.joint_complete_fraction:.6f} "
                f"< {joint_threshold:.6f}"
            )
        fractions = dict(segment.feature_non_null_fractions)
        variances = dict(segment.feature_variances)
        binary = dict(segment.feature_binary_prevalence)
        for column in selected:
            if (
                non_null_threshold is not None
                and float(fractions.get(column, 0.0)) < non_null_threshold
            ):
                errors.append(
                    f"{name}/{column}: non-null coverage "
                    f"{float(fractions.get(column, 0.0)):.6f} < {non_null_threshold:.6f}"
                )
            if min_variance is not None and (
                not np.isfinite(float(variances.get(column, np.nan)))
                or float(variances[column]) <= float(min_variance)
            ):
                errors.append(
                    f"{name}/{column}: variance {variances.get(column)!r} "
                    f"<= {float(min_variance):.6g}"
                )
            prevalence = binary.get(column)
            if (
                bounds is not None
                and prevalence is not None
                and not (bounds[0] <= float(prevalence) <= bounds[1])
            ):
                errors.append(
                    f"{name}/{column}: binary prevalence {float(prevalence):.6f} "
                    f"outside [{bounds[0]:.6f}, {bounds[1]:.6f}]"
                )
    if errors:
        raise PackBStaticPointFeatureLoaderError(
            "coverage/learnability gates failed: " + "; ".join(errors[:12])
        )


def _feature_contract_digest(
    *,
    feature_columns: Sequence[str],
    candidate_universe_sha256: str,
    source_schema_sha256: str,
    raw_allowlist_sha256: str,
    generator_registry_sha256: str,
    store_scan_manifest_sha256: str,
    coverage_profile_sha256: str | None,
    min_exact_key_coverage: float,
    min_non_null_feature_coverage: float,
    max_feature_columns: int | None,
    coverage_admission_rejections: Sequence[tuple[str, str]],
) -> str:
    return _canonical_json_digest(
        {
            "schema": POINT_FEATURE_LOADER_SCHEMA,
            "feature_columns": list(feature_columns),
            "candidate_universe_sha256": candidate_universe_sha256,
            "source_schema_sha256": source_schema_sha256,
            "raw_allowlist_sha256": raw_allowlist_sha256,
            "generator_registry_sha256": generator_registry_sha256,
            "store_scan_manifest_sha256": store_scan_manifest_sha256,
            "coverage_profile_sha256": coverage_profile_sha256,
            "min_exact_key_coverage": min_exact_key_coverage,
            "min_non_null_feature_coverage": min_non_null_feature_coverage,
            "max_feature_columns": max_feature_columns,
            "coverage_admission_rejections": [
                list(value) for value in coverage_admission_rejections
            ],
            "selection_provenance": (
                "current_generator_registry_allowlist_and_deterministic_coverage_only"
            ),
        }
    )


def _coverage_admitted_feature_columns(
    profile: FeatureCoverageProfile,
    columns: Sequence[str],
    *,
    min_segment_non_null_feature_coverage: float | None,
    min_segment_variance: float | None,
    binary_prevalence_bounds: tuple[float, float] | None,
    required_segment_names: Sequence[str],
) -> tuple[list[str], dict[str, str]]:
    """Deterministically prune weak columns using outcome-free slice evidence."""

    selected = [str(column) for column in columns]
    segments = {segment.name: segment for segment in profile.segments}
    required = tuple(map(str, required_segment_names))
    missing_segments = sorted(set(required) - set(segments))
    if missing_segments:
        raise PackBStaticPointFeatureLoaderError(
            "coverage profile lacks required named slices: "
            + ", ".join(missing_segments)
        )
    checked = (
        [segments[name] for name in required] if required else list(segments.values())
    )
    if not checked:
        return selected, {}
    non_null_threshold = (
        _validate_coverage_threshold(
            min_segment_non_null_feature_coverage,
            name="min_segment_non_null_feature_coverage",
        )
        if min_segment_non_null_feature_coverage is not None
        else None
    )
    if min_segment_variance is not None and float(min_segment_variance) < 0:
        raise PackBStaticPointFeatureLoaderError(
            "min_segment_variance must be non-negative"
        )
    bounds: tuple[float, float] | None = None
    if binary_prevalence_bounds is not None:
        lower, upper = map(float, binary_prevalence_bounds)
        if not 0.0 <= lower <= upper <= 1.0:
            raise PackBStaticPointFeatureLoaderError(
                "binary_prevalence_bounds must be ordered values in [0, 1]"
            )
        bounds = lower, upper
    survivors: list[str] = []
    rejected: dict[str, str] = {}
    for column in selected:
        reason: str | None = None
        for segment in checked:
            fractions = dict(segment.feature_non_null_fractions)
            variances = dict(segment.feature_variances)
            binary = dict(segment.feature_binary_prevalence)
            if (
                non_null_threshold is not None
                and float(fractions.get(column, 0.0)) < non_null_threshold
            ):
                reason = (
                    f"{segment.name}:non_null_coverage_"
                    f"{float(fractions.get(column, 0.0)):.6f}_below_{non_null_threshold:.6f}"
                )
                break
            variance = float(variances.get(column, np.nan))
            if min_segment_variance is not None and (
                not np.isfinite(variance) or variance <= float(min_segment_variance)
            ):
                reason = (
                    f"{segment.name}:variance_{variance!r}_below_or_equal_"
                    f"{float(min_segment_variance):.6g}"
                )
                break
            prevalence = binary.get(column)
            if (
                bounds is not None
                and prevalence is not None
                and not (bounds[0] <= float(prevalence) <= bounds[1])
            ):
                reason = (
                    f"{segment.name}:binary_prevalence_{float(prevalence):.6f}_outside_"
                    f"[{bounds[0]:.6f},{bounds[1]:.6f}]"
                )
                break
        if reason is None:
            survivors.append(column)
        else:
            rejected[column] = reason
    return survivors, rejected


def _cap_coverage_admitted_columns(
    profile: FeatureCoverageProfile,
    columns: Sequence[str],
    *,
    max_feature_columns: int | None,
    required_segment_names: Sequence[str],
) -> tuple[list[str], dict[str, str]]:
    """Apply the deterministic outcome-free AE input cap after admission."""

    if max_feature_columns is None:
        return list(columns), {}
    cap = int(max_feature_columns)
    if cap < 1:
        raise PackBStaticPointFeatureLoaderError(
            "max_feature_columns must be positive or null"
        )
    if len(columns) <= cap:
        return list(columns), {}
    segments = {segment.name: segment for segment in profile.segments}
    required = tuple(map(str, required_segment_names))
    missing = sorted(set(required) - set(segments))
    if missing:
        raise PackBStaticPointFeatureLoaderError(
            "coverage profile lacks required named slices for feature cap: "
            + ", ".join(missing)
        )
    checked = (
        [segments[name] for name in required] if required else list(segments.values())
    )
    if not checked:
        # No named slices means global coverage supplied the deterministic rank.
        global_fractions = dict(profile.feature_non_null_fractions)
        scores = {
            str(column): float(global_fractions.get(column, 0.0)) for column in columns
        }
    else:
        scores = {
            str(column): min(
                float(dict(segment.feature_non_null_fractions).get(column, 0.0))
                for segment in checked
            )
            for column in columns
        }
    ranked = sorted(
        (str(column) for column in columns), key=lambda name: (-scores[name], name)
    )
    kept = ranked[:cap]
    rejected = {
        name: f"coverage_rank_{rank + 1}_outside_deterministic_cap_{cap}"
        for rank, name in enumerate(ranked[cap:], start=cap)
    }
    return sorted(kept), rejected


def freeze_feature_contract(
    universe: CandidateFeatureUniverse,
    *,
    coverage_profile: FeatureCoverageProfile | None = None,
    min_exact_key_coverage: float = 1.0,
    min_non_null_feature_coverage: float = 0.99,
    max_feature_columns: int | None = None,
    min_segment_exact_key_coverage: float | None = None,
    min_segment_non_null_feature_coverage: float | None = None,
    min_segment_joint_complete_coverage: float | None = None,
    min_segment_variance: float | None = None,
    binary_prevalence_bounds: tuple[float, float] | None = None,
    required_segment_names: Sequence[str] = (),
    prior_coverage_admission_rejections: Mapping[str, str] | None = None,
) -> FrozenFeatureContract:
    """Freeze ordered inputs using only explicit availability thresholds.

    This function performs no target, return, label, feature-importance, or
    HPO-based selection.  It may remove a schema column only because that
    column is unavailable on the deterministic sample below the supplied
    inference-availability threshold.
    """

    exact_threshold = _validate_coverage_threshold(
        min_exact_key_coverage, name="min_exact_key_coverage"
    )
    feature_threshold = _validate_coverage_threshold(
        min_non_null_feature_coverage, name="min_non_null_feature_coverage"
    )
    if universe.missing_schema_symbols:
        raise PackBStaticPointFeatureLoaderError(
            "cannot freeze a feature contract while store schemas are missing for: "
            + ", ".join(universe.missing_schema_symbols)
        )
    selected = list(universe.feature_columns)
    admission_rejections: dict[str, str] = {
        str(name): str(reason)
        for name, reason in (prior_coverage_admission_rejections or {}).items()
    }
    profile_sha256: str | None = None
    if coverage_profile is not None:
        if coverage_profile.sampled_rows < 1:
            raise PackBStaticPointFeatureLoaderError(
                "coverage profile has no sampled rows"
            )
        exact_fraction = (
            coverage_profile.matched_exact_rows / coverage_profile.sampled_rows
        )
        if exact_fraction < exact_threshold:
            raise PackBStaticPointFeatureLoaderError(
                "exact feature-key coverage is below the frozen contract threshold: "
                f"{exact_fraction:.6f} < {exact_threshold:.6f}"
            )
        fractions = dict(coverage_profile.feature_non_null_fractions)
        global_selected: list[str] = []
        for name in selected:
            coverage = float(fractions.get(name, 0.0))
            if coverage >= feature_threshold:
                global_selected.append(name)
            else:
                admission_rejections.setdefault(
                    name,
                    (
                        f"all:non_null_coverage_{coverage:.6f}_below_{feature_threshold:.6f}"
                    ),
                )
        selected, segment_rejections = _coverage_admitted_feature_columns(
            coverage_profile,
            global_selected,
            min_segment_non_null_feature_coverage=min_segment_non_null_feature_coverage,
            min_segment_variance=min_segment_variance,
            binary_prevalence_bounds=binary_prevalence_bounds,
            required_segment_names=required_segment_names,
        )
        for name, reason in segment_rejections.items():
            admission_rejections.setdefault(name, reason)
        selected, cap_rejections = _cap_coverage_admitted_columns(
            coverage_profile,
            selected,
            max_feature_columns=max_feature_columns,
            required_segment_names=required_segment_names,
        )
        for name, reason in cap_rejections.items():
            admission_rejections.setdefault(name, reason)
        profile_sha256 = coverage_profile.profile_sha256
        validate_feature_coverage_gates(
            coverage_profile,
            feature_columns=selected,
            min_segment_exact_key_coverage=min_segment_exact_key_coverage,
            min_segment_non_null_feature_coverage=min_segment_non_null_feature_coverage,
            min_segment_joint_complete_coverage=min_segment_joint_complete_coverage,
            min_variance=min_segment_variance,
            binary_prevalence_bounds=binary_prevalence_bounds,
            required_segment_names=required_segment_names,
        )
    if not selected:
        raise PackBStaticPointFeatureLoaderError(
            "coverage-only filtering leaves no causal trainable feature columns"
        )
    selected = sorted(selected)
    digest = _feature_contract_digest(
        feature_columns=selected,
        candidate_universe_sha256=universe.universe_sha256,
        source_schema_sha256=universe.source_schema_sha256,
        raw_allowlist_sha256=universe.raw_allowlist_sha256,
        generator_registry_sha256=universe.generator_registry_sha256,
        store_scan_manifest_sha256=universe.store_scan_manifest_sha256,
        coverage_profile_sha256=profile_sha256,
        min_exact_key_coverage=exact_threshold,
        min_non_null_feature_coverage=feature_threshold,
        max_feature_columns=max_feature_columns,
        coverage_admission_rejections=tuple(sorted(admission_rejections.items())),
    )
    return FrozenFeatureContract(
        feature_columns=tuple(selected),
        candidate_universe_sha256=universe.universe_sha256,
        source_schema_sha256=universe.source_schema_sha256,
        raw_allowlist_sha256=universe.raw_allowlist_sha256,
        generator_registry_sha256=universe.generator_registry_sha256,
        store_scan_manifest_sha256=universe.store_scan_manifest_sha256,
        coverage_profile_sha256=profile_sha256,
        min_exact_key_coverage=exact_threshold,
        min_non_null_feature_coverage=feature_threshold,
        max_feature_columns=max_feature_columns,
        coverage_admission_rejections=tuple(sorted(admission_rejections.items())),
        feature_contract_sha256=digest,
    )


def build_fresh_causal_feature_contract(
    identity_ledger: pd.DataFrame,
    *,
    feature_store_dir: str | Path,
    schema_evidence_path: str | Path | None = None,
    cfg: Mapping[str, Any] | None = None,
    coverage_sample_rows: int = 20_000,
    min_exact_key_coverage: float = 1.0,
    min_non_null_feature_coverage: float = 0.99,
    max_feature_columns: int | None = 256,
    coverage_segments: Mapping[str, pd.DataFrame] | None = None,
    min_segment_exact_key_coverage: float | None = None,
    min_segment_non_null_feature_coverage: float | None = None,
    min_segment_joint_complete_coverage: float | None = None,
    min_segment_variance: float | None = None,
    binary_prevalence_bounds: tuple[float, float] | None = None,
    required_segment_names: Sequence[str] = (),
    max_rows_per_batch: int = DEFAULT_MAX_ROWS_PER_BATCH,
    max_columns_per_read: int = DEFAULT_MAX_COLUMNS_PER_READ,
    resource_guard: TrainingResourceGuard | Any | None = None,
) -> tuple[CandidateFeatureUniverse, FeatureCoverageProfile, FrozenFeatureContract]:
    """Discover, profile, and freeze a fresh causal contract in one call."""

    universe = discover_causal_feature_universe(
        identity_ledger,
        feature_store_dir=feature_store_dir,
        schema_evidence_path=schema_evidence_path,
        cfg=cfg,
        resource_guard=resource_guard,
    )
    preliminary = freeze_feature_contract(
        universe,
        min_exact_key_coverage=0.0,
        min_non_null_feature_coverage=0.0,
        max_feature_columns=None,
    )
    profile = profile_point_feature_coverage(
        identity_ledger,
        feature_store_dir=feature_store_dir,
        feature_contract=preliminary,
        coverage_sample_rows=coverage_sample_rows,
        coverage_segments=coverage_segments,
        max_rows_per_batch=max_rows_per_batch,
        max_columns_per_read=max_columns_per_read,
        resource_guard=resource_guard,
    )
    interim = freeze_feature_contract(
        universe,
        coverage_profile=profile,
        min_exact_key_coverage=min_exact_key_coverage,
        min_non_null_feature_coverage=min_non_null_feature_coverage,
        max_feature_columns=max_feature_columns,
        min_segment_exact_key_coverage=min_segment_exact_key_coverage,
        min_segment_non_null_feature_coverage=min_segment_non_null_feature_coverage,
        # This profile was evaluated over the broad raw universe.  Its joint
        # complete fraction is intentionally *not* a survivor gate.
        min_segment_joint_complete_coverage=None,
        min_segment_variance=min_segment_variance,
        binary_prevalence_bounds=binary_prevalence_bounds,
        required_segment_names=required_segment_names,
    )
    survivor_profile = profile_point_feature_coverage(
        identity_ledger,
        feature_store_dir=feature_store_dir,
        feature_contract=interim,
        coverage_sample_rows=coverage_sample_rows,
        coverage_segments=coverage_segments,
        max_rows_per_batch=max_rows_per_batch,
        max_columns_per_read=max_columns_per_read,
        resource_guard=resource_guard,
    )
    frozen = freeze_feature_contract(
        universe,
        coverage_profile=survivor_profile,
        min_exact_key_coverage=min_exact_key_coverage,
        min_non_null_feature_coverage=min_non_null_feature_coverage,
        max_feature_columns=max_feature_columns,
        min_segment_exact_key_coverage=min_segment_exact_key_coverage,
        min_segment_non_null_feature_coverage=min_segment_non_null_feature_coverage,
        min_segment_joint_complete_coverage=min_segment_joint_complete_coverage,
        min_segment_variance=min_segment_variance,
        binary_prevalence_bounds=binary_prevalence_bounds,
        required_segment_names=required_segment_names,
        prior_coverage_admission_rejections=dict(interim.coverage_admission_rejections),
    )
    return universe, survivor_profile, frozen


def load_point_in_time_features(
    identity_ledger: pd.DataFrame,
    *,
    feature_store_dir: str | Path,
    feature_contract: FrozenFeatureContract | Mapping[str, Any],
    max_rows_per_batch: int = DEFAULT_MAX_ROWS_PER_BATCH,
    max_columns_per_read: int = DEFAULT_MAX_COLUMNS_PER_READ,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    include_identity: bool = False,
    resource_guard: TrainingResourceGuard | Any | None = None,
) -> pd.DataFrame:
    """Return one order-preserving exact feature matrix under a memory cap."""

    normalized = _normalise_identity_ledger(identity_ledger)
    contract = _require_contract(feature_contract)
    output_bytes = (
        len(normalized) * len(contract.feature_columns) * np.dtype(np.float32).itemsize
    )
    if int(max_output_bytes) < 1:
        raise PackBStaticPointFeatureLoaderError("max_output_bytes must be positive")
    if output_bytes > int(max_output_bytes):
        raise PackBStaticPointFeatureLoaderError(
            "requested point-in-time output exceeds the explicit memory cap "
            f"({output_bytes} > {int(max_output_bytes)} bytes); use "
            "iter_point_in_time_feature_batches instead"
        )
    matrix = np.empty(
        (len(normalized), len(contract.feature_columns)), dtype=np.float32
    )
    for batch in iter_point_in_time_feature_batches(
        normalized.drop(columns=["__ledger_row__"]),
        feature_store_dir=feature_store_dir,
        feature_contract=contract,
        max_rows_per_batch=max_rows_per_batch,
        max_columns_per_read=max_columns_per_read,
        coverage_discovery=False,
        resource_guard=resource_guard,
    ):
        matrix[batch.ledger_row_positions, :] = batch.features.to_numpy(
            dtype=np.float32, copy=False
        )
    features = pd.DataFrame(matrix, columns=list(contract.feature_columns))
    if not include_identity:
        return features
    return pd.concat(
        [
            normalized.loc[:, list(IDENTITY_COLUMNS)].reset_index(drop=True),
            features,
        ],
        axis=1,
        copy=False,
    )


@dataclass(frozen=True)
class LoaderEvidenceBundle:
    """Immutable provenance linking a callable loader to its frozen inputs."""

    raw_universe_sha256: str
    coverage_profile_sha256: str | None
    feature_contract_sha256: str
    loader_contract_sha256: str
    loader_module_sha256: str
    source_schema_sha256: str
    source_revision: str
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": POINT_FEATURE_LOADER_SCHEMA,
            "raw_universe_sha256": self.raw_universe_sha256,
            "coverage_profile_sha256": self.coverage_profile_sha256,
            "feature_contract_sha256": self.feature_contract_sha256,
            "loader_contract_sha256": self.loader_contract_sha256,
            "loader_module_sha256": self.loader_module_sha256,
            "source_schema_sha256": self.source_schema_sha256,
            "source_revision": self.source_revision,
            "path": self.path,
        }


def _loader_contract_digest(
    contract: FrozenFeatureContract,
    *,
    max_rows_per_batch: int,
    max_columns_per_read: int,
    max_output_bytes: int,
) -> str:
    loader_module_sha256 = _sha256_file(Path(__file__).resolve())
    return _canonical_json_digest(
        {
            "schema": POINT_FEATURE_LOADER_SCHEMA,
            "feature_contract_sha256": contract.feature_contract_sha256,
            "exact_join": "__symbol__+__ts__",
            "no_asof_or_future_fallback": True,
            "time_block": "utc_calendar_month",
            "max_rows_per_batch": int(max_rows_per_batch),
            "max_columns_per_read": int(max_columns_per_read),
            "max_output_bytes": int(max_output_bytes),
            "storage_reader": "data_store.read_symbol_features_base_delta_view",
            "loader_module_sha256": loader_module_sha256,
        }
    )


def write_loader_evidence_bundle(
    *,
    output_dir: str | Path,
    universe: CandidateFeatureUniverse,
    feature_contract: FrozenFeatureContract | Mapping[str, Any],
    coverage_profile: FeatureCoverageProfile | None = None,
    source_revision: str,
    max_rows_per_batch: int = DEFAULT_MAX_ROWS_PER_BATCH,
    max_columns_per_read: int = DEFAULT_MAX_COLUMNS_PER_READ,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
) -> LoaderEvidenceBundle:
    """Publish immutable raw-universe, coverage, and loader-contract JSONs.

    The target directory must not already exist.  This deliberately avoids
    replacing evidence consumed by a later AE/FS/HPO stage.
    """

    contract = _require_contract(feature_contract)
    revision = _require_git_sha(source_revision, name="source_revision")
    if contract.candidate_universe_sha256 != universe.universe_sha256:
        raise PackBStaticPointFeatureLoaderError(
            "feature contract does not bind the supplied raw-universe evidence"
        )
    if contract.source_schema_sha256 != universe.source_schema_sha256:
        raise PackBStaticPointFeatureLoaderError(
            "feature contract does not bind the supplied store-schema evidence"
        )
    if contract.coverage_profile_sha256 != (
        coverage_profile.profile_sha256 if coverage_profile is not None else None
    ):
        raise PackBStaticPointFeatureLoaderError(
            "feature contract coverage hash does not match supplied coverage evidence"
        )
    destination = Path(output_dir)
    if destination.exists():
        raise PackBStaticPointFeatureLoaderError(
            f"refusing to overwrite loader evidence directory: {destination}"
        )
    loader_hash = _loader_contract_digest(
        contract,
        max_rows_per_batch=max_rows_per_batch,
        max_columns_per_read=max_columns_per_read,
        max_output_bytes=max_output_bytes,
    )
    destination.mkdir(parents=True, exist_ok=False)

    def _write_once(path: Path, value: Mapping[str, Any]) -> None:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")

    raw_path = destination / "raw_feature_universe.json"
    contract_path = destination / "frozen_feature_contract.json"
    coverage_path = destination / "coverage_profile.json"
    _write_once(raw_path, universe.to_dict())
    _write_once(contract_path, contract.to_dict())
    if coverage_profile is not None:
        _write_once(coverage_path, coverage_profile.to_dict())
    bundle = LoaderEvidenceBundle(
        raw_universe_sha256=universe.universe_sha256,
        coverage_profile_sha256=(
            coverage_profile.profile_sha256 if coverage_profile is not None else None
        ),
        feature_contract_sha256=contract.feature_contract_sha256,
        loader_contract_sha256=loader_hash,
        loader_module_sha256=_sha256_file(Path(__file__).resolve()),
        source_schema_sha256=universe.source_schema_sha256,
        source_revision=revision,
        path=str(destination / "loader_evidence.json"),
    )
    evidence_payload = {
        **bundle.to_dict(),
        "raw_feature_universe_file_sha256": _sha256_file(raw_path),
        "frozen_feature_contract_file_sha256": _sha256_file(contract_path),
        "coverage_profile_file_sha256": (
            _sha256_file(coverage_path) if coverage_profile is not None else None
        ),
    }
    _write_once(destination / "loader_evidence.json", evidence_payload)
    return bundle


def point_feature_matrix_sha256(
    identity_ledger: pd.DataFrame,
    feature_matrix: pd.DataFrame | np.ndarray,
    *,
    feature_contract: FrozenFeatureContract | Mapping[str, Any],
) -> str:
    """Hash identities, ordered inputs, and canonical float32 feature values."""

    ledger = _normalise_identity_ledger(identity_ledger)
    contract = _require_contract(feature_contract)
    if isinstance(feature_matrix, pd.DataFrame):
        if tuple(map(str, feature_matrix.columns)) != contract.feature_columns:
            raise PackBStaticPointFeatureLoaderError(
                "matrix columns do not equal the frozen ordered feature contract"
            )
        values = feature_matrix.to_numpy(dtype=np.float32, copy=True)
    else:
        values = np.asarray(feature_matrix, dtype=np.float32)
        values = np.array(values, dtype=np.float32, copy=True)
    if values.shape != (len(ledger), len(contract.feature_columns)):
        raise PackBStaticPointFeatureLoaderError(
            "matrix shape does not match identity rows and frozen feature columns"
        )
    if np.isinf(values).any():
        raise PackBStaticPointFeatureLoaderError(
            "matrix contains non-finite infinity values"
        )
    # Canonicalise NaN payload and signed zero so equal float32 matrices have a
    # stable cross-process digest even when Arrow/pandas allocated different
    # IEEE payload bits.
    bits = values.view(np.uint32)
    bits[np.isnan(values)] = np.uint32(0x7FC00000)
    bits[values == 0.0] = np.uint32(0)
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "schema": POINT_FEATURE_LOADER_SCHEMA,
                "feature_contract_sha256": contract.feature_contract_sha256,
                "identity_stream_sha256": _identity_stream_sha256(ledger),
                "shape": list(values.shape),
                "dtype": "float32_le",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(np.ascontiguousarray(values.astype("<f4", copy=False)).tobytes())
    return digest.hexdigest()


def make_packb_static_feature_loader(
    *,
    feature_store_dir: str | Path,
    feature_contract: FrozenFeatureContract | Mapping[str, Any],
    max_rows_per_batch: int = DEFAULT_MAX_ROWS_PER_BATCH,
    max_columns_per_read: int = DEFAULT_MAX_COLUMNS_PER_READ,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    evidence_bundle: LoaderEvidenceBundle | None = None,
    resource_guard: TrainingResourceGuard | Any | None = None,
):
    """Return the strict callback shape required by ``packb_side_local_ae_stage``.

    The callback refuses a changed/reordered feature list, so the AE/GMM stage
    cannot accidentally use a caller-selected subset that bypasses the frozen
    store contract.
    """

    contract = _require_contract(feature_contract)
    loader_hash = _loader_contract_digest(
        contract,
        max_rows_per_batch=max_rows_per_batch,
        max_columns_per_read=max_columns_per_read,
        max_output_bytes=max_output_bytes,
    )
    if evidence_bundle is not None and (
        evidence_bundle.feature_contract_sha256 != contract.feature_contract_sha256
        or evidence_bundle.loader_contract_sha256 != loader_hash
        or evidence_bundle.raw_universe_sha256 != contract.candidate_universe_sha256
        or evidence_bundle.source_schema_sha256 != contract.source_schema_sha256
        or evidence_bundle.coverage_profile_sha256 != contract.coverage_profile_sha256
        or evidence_bundle.loader_module_sha256
        != _sha256_file(Path(__file__).resolve())
    ):
        raise PackBStaticPointFeatureLoaderError(
            "loader evidence bundle does not bind this frozen loader contract"
        )
    if evidence_bundle is not None:
        _require_git_sha(
            evidence_bundle.source_revision, name="evidence_bundle.source_revision"
        )

    def _loader(ledger: pd.DataFrame, input_features: Sequence[str]) -> pd.DataFrame:
        requested = tuple(str(column) for column in input_features)
        if requested != contract.feature_columns:
            raise PackBStaticPointFeatureLoaderError(
                "Pack-B static feature loader requires the complete frozen ordered "
                "feature contract"
            )
        return load_point_in_time_features(
            ledger,
            feature_store_dir=feature_store_dir,
            feature_contract=contract,
            max_rows_per_batch=max_rows_per_batch,
            max_columns_per_read=max_columns_per_read,
            max_output_bytes=max_output_bytes,
            include_identity=False,
            resource_guard=resource_guard,
        )

    # The AE stage can bind these attributes into its side-stage manifest
    # without trusting a look-alike arbitrary callback.
    _loader.packb_static_feature_loader_evidence = {
        "raw_universe_sha256": contract.candidate_universe_sha256,
        "coverage_profile_sha256": contract.coverage_profile_sha256,
        "feature_contract_sha256": contract.feature_contract_sha256,
        "loader_contract_sha256": loader_hash,
        "loader_module_sha256": _sha256_file(Path(__file__).resolve()),
        "source_schema_sha256": contract.source_schema_sha256,
        "source_revision": evidence_bundle.source_revision
        if evidence_bundle is not None
        else None,
        "evidence_path": evidence_bundle.path if evidence_bundle is not None else None,
    }
    _loader.packb_static_feature_contract = contract.to_dict()
    _loader.packb_static_feature_matrix_sha256 = (
        lambda ledger, matrix: point_feature_matrix_sha256(
            ledger, matrix, feature_contract=contract
        )
    )
    return _loader
