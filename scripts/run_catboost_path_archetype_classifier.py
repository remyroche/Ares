#!/usr/bin/env python3
"""Train the leakage-safe future-path CatBoost archetype classifier.

The materialized input contains canonical causal ``path_arch_*`` summaries used
solely as outcome labels. Raw array-only paths are rejected because they cannot
preserve asset ATR, cost, activation-distance, and OHLC path semantics. The
classifier input universe is resolved from configured base/meta feature keys and
is rejected if it contains any realised path field.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements import (
    catboost_archetype_classifier as archetype,  # noqa: E402
)
from extreme_price_movements.base_candidate_population import (  # noqa: E402
    candidate_identity_sha256,
)
from extreme_price_movements.class_balance_oof_economics import (  # noqa: E402
    BalanceArmOOF,
    EconomicOOFConfig,
    score_class_balance_oof_economics,
)
from extreme_price_movements.features_gmm_ae import AE_GMM_FEATURE_COLUMNS  # noqa: E402
from extreme_price_movements.path_archetype_labels import (  # noqa: E402
    PATH_ARCHETYPE_RULE_VERSION,
    PATH_ARCHETYPE_TYPES,
    PATH_REALIZATION_STRENGTH_TYPES,
    PATH_SHAPE_TYPES,
    deterministic_path_archetype,
)
from extreme_price_movements.path_archetype_support import (  # noqa: E402
    FAST_REALIZATION_WINNER,
    LEGACY_FAST_CLASSES,
    MERGED_PATH_ARCHETYPE_CLASSES,
    merge_fast_realization_winner,
)
from extreme_price_movements.side_aware import candidate_id_series  # noqa: E402
from extreme_price_movements.static_feature_store import (  # noqa: E402
    STATIC_FEATURE_ENDPOINT_VERSION,
    read_static_features,
)
from extreme_price_movements.training_resource_guard import (  # noqa: E402
    GIB,
    TrainingResourceGuard,
    TrainingResourceLimits,
)

RUNNER_SCHEMA = "run_catboost_path_archetype_classifier_v10_side_local_staged"
DEVELOPMENT_OOS_ROUTING_SCHEMA = "catboost_path_archetype_development_oos_routing_v1"
FEATURE_SELECTION_HPO_CONTRACT_SCHEMA = "catboost_path_archetype_feature_selection_hpo_contract_v3_structural_hpo_balance_sweep"
FEATURE_SELECTION_HPO_CONTRACT_FILENAME = "feature_selection_hpo_contract.json"
DEFAULT_HPO_NO_IMPROVEMENT_TRIALS = 15
HPO_SAMPLING_CONTRACT_VERSION = (
    "chronological_regions_time_spread_side_class_stratified_v1"
)
DEFAULT_TIMESTAMP_COLUMN = "__ts__"
DEFAULT_LABEL_END_COLUMN = "__label_end_ts__"
SMOKE_MAX_ROWS = 240
SMOKE_HPO_TRIALS = 1
SMOKE_MAX_FEATURES = 32
SMOKE_MAX_ITERATIONS = 128
SMOKE_PERMUTATION_STAGES = (32, 16)
GEOMETRY_SEARCH_MODEL_PARAMS = {
    # Geometry compares target definitions, never class-balancing arms.  Keep
    # this fixed, bounded and CatBoost-valid until the post-geometry HPO stage.
    "loss_function": "MultiClass",
    "eval_metric": "MultiClass",
    "iterations": 1_000,
    "od_wait": 100,
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 30.0,
    "random_strength": 1.0,
    "bagging_temperature": 1.0,
    "rsm": 0.8,
    "border_count": 64,
    "auto_class_weights": None,
    "bootstrap_type": "Bayesian",
    "grow_policy": "SymmetricTree",
    "random_seed": 20260722,
    "verbose": False,
    "allow_writing_files": False,
    "thread_count": 1,
}
PROXY_HPO_ROWS = 8_000
PROXY_HPO_TRIALS = 20
PROXY_HPO_FOLDS = 2
PROXY_HPO_ITERATIONS = 400
PROXY_HPO_OD_WAIT = 40
PROXY_SELECTION_ITERATIONS = 500
PROXY_SELECTION_OD_WAIT = 50
HPO_STUDY_FILENAME = "hpo_study.sqlite3"
HPO_PROGRESS_FILENAME = "hpo_progress.json"
MDA_PROGRESS_FILENAME = "mda_progress.json"
RESOURCE_TELEMETRY_FILENAME = "training_resource_telemetry.jsonl"
LOGGER = logging.getLogger(__name__)
IDENTITY_SYMBOL_COLUMN = "__symbol__"
IDENTITY_SIDE_COLUMNS = ("side", "side_name", "__side__")
CANONICAL_CANDIDATE_KEY_COLUMNS = ("__ts__", "__symbol__", "side", "candidate_id")
FROZEN_AE_GMM_KEY_COLUMNS = ("__ts__", "__symbol__", "side")
REPRESENTATION_AVAILABLE_FEATURE = "gmm_representation_available"
PATH_TARGET_PROVENANCE_COLUMNS = {
    "__decision_ts__",
    "candidate_id",
    "__barrier_pct__",
    "barrier_pct",
    "__path_auxiliary_atr_fraction__",
    "atr_fraction",
    "path_cost_return",
    "round_trip_cost_return",
    "execution_cost_return",
    "cost_return",
    "activation_distance_return",
    "trailing_activation_distance_return",
}
MIN_SIDE_CLASS_SHARE = 0.01
MIN_SIDE_MONTH_CLASS_SHARE = 0.005

FUTURE_TRAINING_TAXONOMY_CONTRACT = {
    "version": "merged_fast_realization_winner_v1",
    "ordered_classes": list(MERGED_PATH_ARCHETYPE_CLASSES),
    "merge": {
        "source_classes": list(LEGACY_FAST_CLASSES),
        "target_class": FAST_REALIZATION_WINNER,
    },
    "probability_contract": {
        "calibration": "none_raw_catboost_probabilities",
        "sample_weights": "uniform",
        "adverse_classes": [
            "immediate_adverse_path",
            "early_mfe_full_reversal",
            "dead_timeout",
        ],
        "favorable_classes": [
            FAST_REALIZATION_WINNER,
            "late_breakout",
            "slow_grinder",
        ],
        "neutral_classes": ["noisy_timeout_usable_mfe"],
        "derived_fields": [
            "probability_entropy",
            "max_probability",
            "normalized_entropy=entropy/log(7)",
            "top2_probability_margin",
            "adverse_probability_mass",
            "favorable_probability_mass",
        ],
    },
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace JSON artifacts so monitors never read partial output."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = json.dumps(
        _json_safe(payload), indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256_json(payload: Any) -> str:
    """Hash JSON-compatible data with a canonical serialization."""
    encoded = json.dumps(
        _json_safe(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _signed_manifest_hash(payload: Mapping[str, Any]) -> str:
    return _sha256_json(
        {
            key: value
            for key, value in payload.items()
            if key != "prediction_role_manifest_sha256"
        }
    )


def _catboost_oof_provenance(
    frame: pd.DataFrame,
    oof: archetype.OOFPathArchetypeResult,
    *,
    timestamp_column: str,
    label_end_column: str,
    config: archetype.PathArchetypeConfig,
) -> pd.DataFrame:
    """Materialize actual purged-fold information available to each OOF row."""

    timestamps = pd.to_datetime(frame[timestamp_column], utc=True, errors="raise")
    label_end = pd.to_datetime(frame[label_end_column], utc=True, errors="raise")
    output = pd.DataFrame(index=frame.index)
    for column in (
        "validation_start",
        "latest_train_decision_ts",
        "label_resolution_available_at",
        "train_decision_cutoff",
    ):
        output[column] = pd.Series(
            pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]"
        )
    fold_lookup = {int(fold.fold_id): fold for fold in oof.folds}
    for fold_id in sorted(
        int(value) for value in np.unique(oof.fold_ids) if value >= 0
    ):
        validation_rows = np.flatnonzero(oof.fold_ids == fold_id)
        if not len(validation_rows):
            continue
        validation_start = timestamps.iloc[validation_rows].min()
        fitted_fold = fold_lookup.get(fold_id)
        if fitted_fold is not None:
            train_rows = np.asarray(fitted_fold.train_indices, dtype=np.int64)
        else:
            # This is the same explicit purge/embargo rule as the fitter and is
            # used only for adapters/tests that retain fold IDs but not objects.
            prior = np.flatnonzero((timestamps < validation_start).to_numpy())
            train_rows = prior[
                (label_end.iloc[prior] < validation_start).to_numpy()
                & (
                    timestamps.iloc[prior] < validation_start - config.embargo
                ).to_numpy()
            ]
        if not len(train_rows):
            raise ValueError(f"CatBoost OOF fold {fold_id} has no purged training rows")
        latest_decision = timestamps.iloc[train_rows].max()
        latest_resolution = label_end.iloc[train_rows].max()
        information_cutoff = max(latest_decision, latest_resolution)
        if not information_cutoff < validation_start:
            raise ValueError(
                f"CatBoost OOF fold {fold_id} training information reaches validation"
            )
        index = frame.index[validation_rows]
        output.loc[index, "validation_start"] = validation_start
        output.loc[index, "latest_train_decision_ts"] = latest_decision
        output.loc[index, "label_resolution_available_at"] = latest_resolution
        output.loc[index, "train_decision_cutoff"] = information_cutoff
    return output


def _frame_contract_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    """Hash an ordered target contract without serializing a large CSV."""
    selected = tuple(map(str, columns))
    missing = sorted(set(selected).difference(frame.columns))
    if missing:
        raise ValueError("contract columns are missing: " + ", ".join(missing[:8]))
    digest = hashlib.sha256()
    digest.update("\x1f".join(selected).encode("utf-8"))
    digest.update(b"\n")
    values = pd.util.hash_pandas_object(
        frame.loc[:, list(selected)], index=False, categorize=True
    ).to_numpy(dtype=np.uint64, copy=False)
    digest.update(values.tobytes())
    return digest.hexdigest()


def _read_optional_list(path: Path | None) -> list[str]:
    if path is None:
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if isinstance(payload, Mapping):
        payload = payload.get("mandatory_features", payload.get("features"))
    if not isinstance(payload, list) or not all(
        isinstance(item, str) for item in payload
    ):
        raise ValueError(
            "mandatory feature file must be a JSON string list, {mandatory_features: [...]}, or text list"
        )
    return list(dict.fromkeys(payload))


def _read_config_mapping(path: Path | None) -> Mapping[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("config JSON must contain an object")
    return payload


def _gib_to_bytes(value: float, *, name: str) -> int:
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative GiB value")
    return int(value * GIB)


def _resource_disk_path(output_dir: Path) -> Path:
    """Use an existing ancestor on the output filesystem for disk telemetry."""

    path = Path(output_dir)
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def _build_resource_guard(
    *,
    output_dir: Path,
    min_free_ram_gib: float,
    max_process_rss_gib: float,
    min_free_disk_gib: float,
    check_interval_seconds: float,
    telemetry_path: Path | None,
) -> TrainingResourceGuard:
    if not np.isfinite(check_interval_seconds) or check_interval_seconds < 0:
        raise ValueError(
            "resource_check_interval_seconds must be finite and non-negative"
        )
    limits = TrainingResourceLimits(
        min_free_ram_bytes=_gib_to_bytes(
            min_free_ram_gib, name="resource_min_free_ram_gib"
        ),
        max_process_rss_bytes=_gib_to_bytes(
            max_process_rss_gib, name="resource_max_process_rss_gib"
        ),
        min_free_disk_bytes=_gib_to_bytes(
            min_free_disk_gib, name="resource_min_free_disk_gib"
        ),
        check_interval_seconds=float(check_interval_seconds),
    )
    return TrainingResourceGuard(
        limits=limits,
        disk_path=_resource_disk_path(output_dir),
        telemetry_path=telemetry_path or output_dir / RESOURCE_TELEMETRY_FILENAME,
    )


def _resource_guard_contract(guard: TrainingResourceGuard) -> dict[str, Any]:
    return {
        "limits": {
            "min_free_ram_bytes": guard.limits.min_free_ram_bytes,
            "max_process_rss_bytes": guard.limits.max_process_rss_bytes,
            "min_free_disk_bytes": guard.limits.min_free_disk_bytes,
            "check_interval_seconds": guard.limits.check_interval_seconds,
        },
        "disk_path": str(guard.disk_path),
        "telemetry_path": (
            str(guard.telemetry_path) if guard.telemetry_path is not None else None
        ),
        "contract": "fail_closed_preflight_and_boundary_checkpoints_v1",
    }


def _load_frame(input_data: pd.DataFrame | Path) -> tuple[pd.DataFrame, str]:
    if isinstance(input_data, pd.DataFrame):
        return input_data.copy(), "in_memory_dataframe"
    path = Path(input_data)
    if path.suffix.lower() not in {".parquet", ".pq"}:
        raise ValueError("input must be a materialized parquet file")
    return pd.read_parquet(path), str(path)


def _utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )


def _feature_store_location(feature_dir: Path) -> tuple[pd.Timestamp, Path]:
    """Resolve one shared static-store directory without guessing a run ID."""
    feature_dir = Path(feature_dir)
    try:
        feature_store_ts = pd.to_datetime(
            feature_dir.name, format="%Y%m%d_%H%M%S", utc=True
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "feature directory must be a timestamped shared static store, e.g. "
            "data_perp/features/20260711_070000"
        ) from exc
    if feature_dir.parent.name != "features":
        raise ValueError(
            f"feature directory is outside the shared static-store layout: {feature_dir}"
        )
    if not feature_dir.is_dir():
        raise FileNotFoundError(f"feature directory does not exist: {feature_dir}")
    return pd.Timestamp(feature_store_ts), feature_dir.parent.parent


def _schema_names(path: Path) -> set[str]:
    """Read parquet metadata only; never materialize a feature-store column here."""
    try:
        from extreme_price_movements.data_store import _feature_schema_names

        return {str(value) for value in _feature_schema_names(str(path))}
    except Exception:
        try:
            import pyarrow.parquet as pq

            return {str(value) for value in pq.read_schema(path).names}
        except Exception:
            return set()


def _feature_store_schemas(feature_dir: Path) -> tuple[set[str], dict[str, set[str]]]:
    """Return the schema union and exact-file schema coverage for audit output."""
    union: set[str] = set()
    by_file: dict[str, set[str]] = {}
    for path in sorted(feature_dir.glob("symbol=*.parquet")):
        names = {
            name
            for name in _schema_names(path)
            if name not in {"ts", "__symbol__"}
            and not name.startswith("__index_level_")
        }
        by_file[path.name] = names
        union.update(names)
    if not by_file:
        raise ValueError(f"feature store has no symbol parquet files: {feature_dir}")
    return union, by_file


def _config_feature_names(config_mapping: Mapping[str, Any] | None) -> tuple[str, ...]:
    """Give the classifier's existing resolver every config-declared string."""
    mapping: Mapping[str, Any]
    if config_mapping is None:
        from extreme_price_movements.config import CFG

        mapping = CFG
    else:
        mapping = config_mapping
    names: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, str):
            names.append(value)
        elif isinstance(value, Mapping):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, (list, tuple, set)):
            for nested in value:
                visit(nested)

    visit(mapping)
    return tuple(dict.fromkeys(names))


def _canonical_identity_columns(
    frame: pd.DataFrame, *, timestamp_column: str
) -> tuple[str, ...]:
    if IDENTITY_SYMBOL_COLUMN not in frame:
        raise ValueError(
            "canonical feature-dir mode requires __symbol__ for timestamp/symbol/side alignment"
        )
    side_column = next((name for name in IDENTITY_SIDE_COLUMNS if name in frame), None)
    if side_column is None:
        raise ValueError(
            "canonical feature-dir mode requires side, side_name, or __side__ for exact alignment"
        )
    columns = (timestamp_column, IDENTITY_SYMBOL_COLUMN, side_column)
    if frame.loc[:, columns].isna().any().any():
        raise ValueError("canonical timestamp/symbol/side identity has missing values")
    if frame.duplicated(list(columns)).any():
        raise ValueError("canonical timestamp/symbol/side identity must be one-to-one")
    return columns


def _sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _quote_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _canonical_side_values(values: pd.Series) -> pd.Series:
    """Normalize the label identity side to the frozen sidecar's int8 contract."""
    numeric = pd.to_numeric(values, errors="coerce")
    text = values.astype("string").str.strip().str.lower()
    numeric = numeric.where(
        numeric.notna(), text.map({"long": 1, "short": -1, "buy": 1, "sell": -1})
    )
    if numeric.isna().any() or not numeric.isin((-1, 1)).all():
        raise ValueError(
            "frozen AE/GMM sidecar requires canonical long/short side values"
        )
    return numeric.astype(np.int8)


def _frozen_sidecar_identity(
    frame: pd.DataFrame, *, timestamp_column: str, side_column: str
) -> pd.DataFrame:
    identity = pd.DataFrame(
        {
            "__row_id__": np.arange(len(frame), dtype=np.int64),
            "__ts__": pd.to_datetime(frame[timestamp_column], utc=True, errors="raise"),
            "__symbol__": frame[IDENTITY_SYMBOL_COLUMN].astype(str).to_numpy(),
            "side": _canonical_side_values(frame[side_column]).to_numpy(),
        }
    )
    if identity.loc[:, list(FROZEN_AE_GMM_KEY_COLUMNS)].duplicated().any():
        raise ValueError("frozen AE/GMM label identity must be one-to-one")
    return identity


def _read_frozen_ae_gmm_manifest(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("frozen AE/GMM manifest must contain a JSON object")
    expected = list(map(str, AE_GMM_FEATURE_COLUMNS))
    manifest_features = payload.get("output_features")
    if manifest_features is None and payload.get("schema") == (
        "packb_downstream_frozen_side_representation_v1"
    ):
        representation = payload.get("representation")
        if not isinstance(representation, dict):
            raise ValueError(
                "frozen AE/GMM context manifest is missing representation contract"
            )
        manifest_features = representation.get("generated_features")
        if representation.get("availability_feature") != (
            REPRESENTATION_AVAILABLE_FEATURE
        ):
            raise ValueError(
                "frozen AE/GMM context manifest has the wrong availability feature"
            )
        if (
            not isinstance(manifest_features, list)
            or not manifest_features
            or len(manifest_features) != len(set(manifest_features))
            or not set(map(str, manifest_features)).issubset(expected)
        ):
            raise ValueError(
                "frozen AE/GMM context manifest has invalid generated features"
            )
    elif manifest_features != expected:
        raise ValueError(
            "frozen AE/GMM manifest generated features do not match "
            "AE_GMM_FEATURE_COLUMNS"
        )
    return payload


def _sidecar_side_sql(alias: str, column: str = "side") -> str:
    qualified = f"{alias}.{_quote_identifier(column)}"
    return (
        f"CASE lower(trim(CAST({qualified} AS VARCHAR))) "
        f"WHEN 'long' THEN 1 WHEN 'short' THEN -1 "
        f"ELSE try_cast({qualified} AS TINYINT) END"
    )


def _development_oos_routing(
    frame: pd.DataFrame,
    *,
    timestamp_column: str,
    label_end_column: str,
    development_end: str | pd.Timestamp | None,
) -> tuple[pd.Series, dict[str, Any]]:
    """Return the training-development mask and an auditable OOS partition.

    A supplied cutoff is deliberately based on *label resolution*, not merely
    decision time: a decision before the boundary whose path resolves later
    cannot influence feature selection or HPO.  It is also not silently
    relabelled as an OOS decision, so the boundary spill is explicit in the
    manifest.  The later outer-OOF stage remains responsible for scoring its
    fixed evaluation windows with its own fold-local train sets.
    """

    decision = pd.to_datetime(frame[timestamp_column], utc=True, errors="raise")
    label_end = pd.to_datetime(frame[label_end_column], utc=True, errors="raise")
    if development_end is None:
        development = pd.Series(True, index=frame.index)
        return development, {
            "schema": DEVELOPMENT_OOS_ROUTING_SCHEMA,
            "enabled": False,
            "development_end_exclusive": None,
            "development_rows": int(len(frame)),
            "oos_decision_rows": 0,
            "boundary_unassigned_rows": 0,
            "development_contract": "legacy_full_population_compatibility_mode",
        }
    cutoff = _utc_timestamp(development_end)
    development = label_end.lt(cutoff)
    oos = decision.ge(cutoff)
    boundary = ~(development | oos)
    if not development.any():
        raise ValueError(
            "strict development/OOS routing leaves no label-resolved development rows"
        )
    return development, {
        "schema": DEVELOPMENT_OOS_ROUTING_SCHEMA,
        "enabled": True,
        "development_end_exclusive": cutoff,
        "development_rows": int(development.sum()),
        "oos_decision_rows": int(oos.sum()),
        "boundary_unassigned_rows": int(boundary.sum()),
        "development_contract": (
            "feature_selection_hpo_and_any_reusable_selection_state_use_only_rows "
            "whose_label_end_is_strictly_before_development_end"
        ),
        "oos_contract": (
            "decisions_at_or_after_development_end_are_excluded_from_feature_selection "
            "and_hpo_and_are_reserved_for_outer_oof_evaluation"
        ),
        "boundary_contract": (
            "pre_cutoff_decisions_with_unresolved_labels_are_excluded_from_both "
            "development_fitting_and_oos_decision_reporting"
        ),
    }


def _validate_frozen_ae_gmm_sidecar(
    frame: pd.DataFrame,
    *,
    sidecar_path: Path,
    manifest_path: Path | None,
    timestamp_column: str,
    side_column: str,
) -> dict[str, Any]:
    """Validate the frozen representation before selection can see any rows."""
    try:
        import duckdb
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - production dependency.
        raise ImportError(
            "duckdb and pyarrow are required for frozen AE/GMM sidecars"
        ) from exc

    sidecar_path = Path(sidecar_path)
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"frozen AE/GMM sidecar does not exist: {sidecar_path}")
    manifest = _read_frozen_ae_gmm_manifest(manifest_path)
    if manifest is not None and manifest.get("schema") == (
        "packb_downstream_frozen_side_representation_v1"
    ):
        generated_features = tuple(
            map(str, manifest["representation"]["generated_features"])
        )
    else:
        generated_features = tuple(map(str, AE_GMM_FEATURE_COLUMNS))
    expected_features = (*generated_features, REPRESENTATION_AVAILABLE_FEATURE)
    schema_names = tuple(map(str, pq.ParquetFile(sidecar_path).schema_arrow.names))
    required_columns = set(FROZEN_AE_GMM_KEY_COLUMNS) | set(expected_features)
    missing_columns = sorted(required_columns.difference(schema_names))
    if missing_columns:
        raise ValueError(
            "frozen AE/GMM sidecar is missing required columns: "
            + ", ".join(missing_columns[:8])
        )
    identity = _frozen_sidecar_identity(
        frame, timestamp_column=timestamp_column, side_column=side_column
    )
    sidecar_sql = _sql_literal(sidecar_path)
    normalized_side = _sidecar_side_sql("s")
    join = (
        "epoch_ns(l.__ts__) = epoch_ns(s.__ts__) "
        f"AND l.__symbol__ = s.__symbol__ AND l.side = ({normalized_side})"
    )
    con = duckdb.connect()
    try:
        con.execute("SET TimeZone='UTC'")
        con.register("label_keys", identity)
        duckdb_columns = tuple(
            con.execute(f"SELECT * FROM read_parquet({sidecar_sql}) LIMIT 0")
            .fetchdf()
            .columns
        )
        if len(duckdb_columns) != len(schema_names):
            raise RuntimeError("frozen AE/GMM sidecar schema changed while opening it")
        resolved_columns = dict(zip(schema_names, duckdb_columns, strict=True))
        native_missing_predicate = " OR ".join(
            f"s.{_quote_identifier(resolved_columns[name])} IS NULL OR "
            "coalesce(isnan(CAST("
            f"s.{_quote_identifier(resolved_columns[name])} AS DOUBLE)), false)"
            for name in generated_features
        )
        infinite_predicate = " OR ".join(
            "coalesce(isinf(CAST("
            f"s.{_quote_identifier(resolved_columns[name])} AS DOUBLE)), false)"
            for name in generated_features
        )
        availability_column = _quote_identifier(
            resolved_columns[REPRESENTATION_AVAILABLE_FEATURE]
        )
        availability_sql = f"CAST(s.{availability_column} AS DOUBLE)"
        invalid_availability_predicate = (
            f"s.{availability_column} IS NULL OR "
            f"coalesce(isnan({availability_sql}), false) OR "
            f"coalesce(isinf({availability_sql}), false) OR "
            f"{availability_sql} NOT IN (0.0, 1.0)"
        )
        duplicate = con.execute(
            f"""
            SELECT 1
            FROM read_parquet({sidecar_sql}) AS s
            GROUP BY epoch_ns(s.__ts__), s.__symbol__, ({normalized_side})
            HAVING count(*) > 1
            LIMIT 1
            """
        ).fetchone()
        if duplicate is not None:
            raise ValueError(
                "frozen AE/GMM sidecar has duplicate timestamp/symbol/side keys"
            )
        coverage = con.execute(
            f"""
            SELECT
                count(*) AS label_rows,
                count(s.__ts__) AS matched_rows,
                count(*) - count(s.__ts__) AS missing_rows,
                sum(CASE WHEN s.__ts__ IS NOT NULL AND ({native_missing_predicate}) THEN 1 ELSE 0 END)
                    AS native_missing_rows,
                sum(CASE WHEN s.__ts__ IS NOT NULL AND ({infinite_predicate}) THEN 1 ELSE 0 END)
                    AS infinite_rows,
                sum(CASE WHEN s.__ts__ IS NOT NULL AND ({invalid_availability_predicate})
                    THEN 1 ELSE 0 END) AS invalid_availability_rows,
                sum(CASE WHEN s.__ts__ IS NOT NULL
                              AND {availability_sql} = 1.0
                              AND ({native_missing_predicate})
                    THEN 1 ELSE 0 END) AS available_with_missing_rows
            FROM label_keys AS l
            LEFT JOIN read_parquet({sidecar_sql}) AS s ON {join}
            """
        ).fetchone()
    finally:
        con.close()
    assert coverage is not None
    (
        label_rows,
        matched_rows,
        missing_rows,
        native_missing_rows,
        infinite_rows,
        invalid_availability_rows,
        available_with_missing_rows,
    ) = map(int, coverage)
    if missing_rows:
        raise ValueError(
            "frozen AE/GMM sidecar does not cover every label key: "
            f"missing={missing_rows}, labels={label_rows}"
        )
    if infinite_rows:
        raise ValueError(
            "frozen AE/GMM sidecar has infinite generated outputs: "
            f"rows={infinite_rows}"
        )
    if invalid_availability_rows:
        raise ValueError(
            "frozen AE/GMM sidecar has an invalid representation-availability "
            f"flag: rows={invalid_availability_rows}"
        )
    if available_with_missing_rows:
        raise ValueError(
            "frozen AE/GMM sidecar marks rows available despite missing generated "
            f"outputs: rows={available_with_missing_rows}"
        )
    return {
        "path": str(sidecar_path),
        "sidecar_sha256": _sha256_file(sidecar_path),
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "manifest_sha256": (
            _sha256_file(manifest_path) if manifest_path is not None else None
        ),
        "manifest": manifest,
        "key_columns": list(FROZEN_AE_GMM_KEY_COLUMNS),
        "output_features": list(expected_features),
        "duckdb_columns": {name: resolved_columns[name] for name in expected_features},
        "label_rows": label_rows,
        "matched_rows": matched_rows,
        "missing_rows": missing_rows,
        "native_missing_rows": native_missing_rows,
        "infinite_rows": infinite_rows,
        "invalid_availability_rows": invalid_availability_rows,
        "available_with_missing_rows": available_with_missing_rows,
        "availability_feature": REPRESENTATION_AVAILABLE_FEATURE,
        "missing_value_policy": (
            "preserve_native_nan_only_when_availability_is_zero_reject_infinite"
        ),
        "join_contract": (
            "exact UTC timestamp, symbol, and canonical long/short side "
            "normalized to int8"
        ),
    }


def _load_frozen_ae_gmm_matrix(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    sidecar_contract: Mapping[str, Any],
    timestamp_column: str,
    side_column: str,
) -> pd.DataFrame:
    """Load a column-pruned, exact-key sidecar matrix in caller row order."""
    requested = tuple(dict.fromkeys(map(str, feature_columns)))
    if not requested:
        return pd.DataFrame(index=frame.index)
    allowed = set(map(str, sidecar_contract["output_features"]))
    unexpected = sorted(set(requested).difference(allowed))
    if unexpected:
        raise ValueError(
            "requested fields are not frozen AE/GMM outputs: "
            + ", ".join(unexpected[:8])
        )
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - production dependency.
        raise ImportError("duckdb is required for frozen AE/GMM sidecars") from exc
    identity = _frozen_sidecar_identity(
        frame, timestamp_column=timestamp_column, side_column=side_column
    )
    source_columns = dict(sidecar_contract.get("duckdb_columns", {}))
    aliases = tuple(f"__frozen_feature_{index}" for index in range(len(requested)))
    projection = ", ".join(
        f"s.{_quote_identifier(str(source_columns.get(name, name)))} "
        f"AS {_quote_identifier(alias)}"
        for name, alias in zip(requested, aliases, strict=True)
    )
    sidecar_sql = _sql_literal(Path(str(sidecar_contract["path"])))
    con = duckdb.connect()
    try:
        con.execute("SET TimeZone='UTC'")
        con.register("label_keys", identity)
        loaded = con.execute(
            f"""
            SELECT l.__row_id__, {projection}
            FROM label_keys AS l
            JOIN read_parquet({sidecar_sql}) AS s
              ON epoch_ns(l.__ts__) = epoch_ns(s.__ts__)
             AND l.__symbol__ = s.__symbol__
             AND l.side = ({_sidecar_side_sql("s")})
            ORDER BY l.__row_id__
            """
        ).fetchdf()
    finally:
        con.close()
    if len(loaded) != len(frame) or not np.array_equal(
        loaded["__row_id__"].to_numpy(dtype=np.int64),
        np.arange(len(frame), dtype=np.int64),
    ):
        raise RuntimeError(
            "frozen AE/GMM sidecar exact join changed label row alignment"
        )
    matrix = loaded.loc[:, list(aliases)].apply(pd.to_numeric, errors="coerce")
    matrix.columns = requested
    values = matrix.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise RuntimeError("frozen AE/GMM sidecar load produced infinite values")
    if REPRESENTATION_AVAILABLE_FEATURE in matrix:
        availability = matrix[REPRESENTATION_AVAILABLE_FEATURE].to_numpy(dtype=float)
        if (
            not np.isfinite(availability).all()
            or not np.isin(availability, (0.0, 1.0)).all()
        ):
            raise RuntimeError(
                "frozen AE/GMM sidecar load produced an invalid availability flag"
            )
    matrix.index = frame.index
    return matrix.astype(np.float32, copy=False)


def _load_model_feature_matrix(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    static_feature_columns: set[str],
    frozen_sidecar_contract: Mapping[str, Any] | None,
    timestamp_column: str,
    side_column: str,
    feature_store_ts: pd.Timestamp,
    data_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Combine static and frozen representation inputs without changing row order."""
    requested = tuple(dict.fromkeys(map(str, feature_columns)))
    static_columns = tuple(
        column for column in requested if column in static_feature_columns
    )
    frozen_columns = tuple(
        column for column in requested if column not in static_feature_columns
    )
    if frozen_columns and frozen_sidecar_contract is None:
        raise ValueError(
            "frozen AE/GMM fields were requested without a sidecar contract"
        )
    if static_columns:
        static_matrix, static_report = _load_static_feature_matrix(
            frame,
            static_columns,
            feature_store_ts=feature_store_ts,
            data_root=data_root,
            timestamp_column=timestamp_column,
        )
    else:
        static_matrix = pd.DataFrame(index=frame.index)
        static_report = {"requested_features": 0, "alignment": "not_requested"}
    frozen_matrix = (
        _load_frozen_ae_gmm_matrix(
            frame,
            frozen_columns,
            sidecar_contract=frozen_sidecar_contract,
            timestamp_column=timestamp_column,
            side_column=side_column,
        )
        if frozen_columns
        else pd.DataFrame(index=frame.index)
    )
    matrix = pd.concat([static_matrix, frozen_matrix], axis=1).reindex(
        columns=requested
    )
    return matrix, {
        "static": static_report,
        "frozen_ae_gmm_requested_features": list(frozen_columns),
        "frozen_ae_gmm_rows": int(len(frozen_matrix)),
    }


def _beginning_middle_end_positions(rows: int, sample_rows: int) -> np.ndarray:
    """Take bounded contiguous beginning/middle/end blocks for store pushdown."""
    if rows <= 0:
        return np.empty(0, dtype=np.int64)
    count = min(int(rows), max(1, int(sample_rows)))
    allocations = np.full(3, count // 3, dtype=np.int64)
    allocations[: count % 3] += 1
    # The full selection population must never force a wide full-history read.
    # Keep each sample in a short contiguous time block: beginning, middle, end.
    first, middle, last = map(int, allocations)
    middle_start = max(first, (rows - middle) // 2)
    last_start = max(middle_start + middle, rows - last)
    positions = np.concatenate(
        (
            np.arange(0, first, dtype=np.int64),
            np.arange(middle_start, middle_start + middle, dtype=np.int64),
            np.arange(last_start, rows, dtype=np.int64),
        )
    )
    return np.unique(positions)


def _support_records(
    frame: pd.DataFrame,
    labels: pd.Series,
    *,
    side_column: str,
) -> dict[str, list[dict[str, Any]]]:
    """Return stable class, side, and side-by-class support for an audit."""
    support = pd.DataFrame(
        {
            "side": frame[side_column].astype("string").fillna("<missing>").astype(str),
            "class": labels.astype("string").fillna("<missing>").astype(str),
        },
        index=frame.index,
    )
    return {
        "class_support": [
            {"class": str(name), "rows": int(rows)}
            for name, rows in support["class"]
            .value_counts(sort=False)
            .sort_index()
            .items()
        ],
        "side_support": [
            {"side": str(name), "rows": int(rows)}
            for name, rows in support["side"]
            .value_counts(sort=False)
            .sort_index()
            .items()
        ],
        "side_class_support": [
            {"side": str(side), "class": str(label), "rows": int(rows)}
            for (side, label), rows in (
                support.value_counts(sort=False).sort_index().items()
            )
        ],
    }


def _stratified_time_spread_positions(
    frame: pd.DataFrame,
    labels: pd.Series,
    *,
    sample_rows: int,
    side_column: str,
) -> np.ndarray:
    """Sample every side/class stratum proportionally across its local time span."""
    if len(frame) != len(labels):
        raise ValueError("HPO sampling frame and labels must have equal rows")
    target_rows = min(int(sample_rows), len(frame))
    if target_rows <= 0:
        return np.empty(0, dtype=np.int64)
    strata = pd.DataFrame(
        {
            "side": frame[side_column]
            .astype("string")
            .fillna("<missing>")
            .astype(str)
            .to_numpy(),
            "class": labels.astype("string").fillna("<missing>").astype(str).to_numpy(),
        }
    )
    groups = [
        positions.to_numpy(dtype=np.int64, copy=False)
        for _key, positions in strata.groupby(
            ["side", "class"], sort=True
        ).groups.items()
    ]
    if target_rows < len(groups):
        raise ValueError(
            "HPO sample rows cannot preserve every observed side/class stratum: "
            f"rows={target_rows}, strata={len(groups)}"
        )
    counts = np.asarray([len(group) for group in groups], dtype=np.int64)
    quotas = np.ones(len(groups), dtype=np.int64)
    remaining = int(target_rows - len(groups))
    capacity = counts - quotas
    if remaining:
        weights = capacity.astype(float)
        raw = remaining * weights / weights.sum()
        additions = np.floor(raw).astype(np.int64)
        quotas += additions
        capacity -= additions
        remaining -= int(additions.sum())
        order = sorted(
            np.flatnonzero(capacity > 0),
            key=lambda index: (
                -float(raw[index] - additions[index]),
                -int(counts[index]),
                int(index),
            ),
        )
        for index in order[:remaining]:
            quotas[index] += 1
        remaining = 0
    if remaining:
        raise RuntimeError("HPO stratified sampler could not allocate requested rows")
    selected = [
        group[np.linspace(0, len(group) - 1, int(quota), dtype=np.int64)]
        for group, quota in zip(groups, quotas, strict=True)
    ]
    return np.sort(np.concatenate(selected).astype(np.int64, copy=False))


def _stratified_hpo_sample(
    frame: pd.DataFrame,
    labels: pd.Series,
    *,
    sample_rows: int,
    validation_folds: int,
    timestamp_column: str,
    label_end_column: str,
    side_column: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build chronological HPO regions with deterministic side/class coverage."""
    if not frame[timestamp_column].is_monotonic_increasing:
        raise ValueError("HPO sampling requires chronologically ordered rows")
    region_count = int(validation_folds) + 1
    target_rows = min(int(sample_rows), len(frame))
    if target_rows < region_count:
        raise ValueError("HPO sample needs at least one row per chronological region")
    allocations = np.full(region_count, target_rows // region_count, dtype=np.int64)
    allocations[: target_rows % region_count] += 1
    full_regions = np.array_split(np.arange(len(frame), dtype=np.int64), region_count)
    selected_regions: list[np.ndarray] = []
    report_regions: list[dict[str, Any]] = []
    for region_id, (full_positions, allocation) in enumerate(
        zip(full_regions, allocations, strict=True)
    ):
        if not len(full_positions):
            raise ValueError("HPO chronological region is empty")
        region_frame = frame.iloc[full_positions]
        region_labels = labels.iloc[full_positions]
        local = _stratified_time_spread_positions(
            region_frame,
            region_labels,
            sample_rows=int(allocation),
            side_column=side_column,
        )
        positions = full_positions[local]
        sampled_frame = frame.iloc[positions]
        sampled_labels = labels.iloc[positions]
        selected_regions.append(positions)
        report_regions.append(
            {
                "region_id": int(region_id),
                "eligible_rows": int(len(region_frame)),
                "sample_rows": int(len(sampled_frame)),
                "eligible_start_timestamp_utc": region_frame[timestamp_column].min(),
                "eligible_end_timestamp_utc": region_frame[timestamp_column].max(),
                "sample_start_timestamp_utc": sampled_frame[timestamp_column].min(),
                "sample_end_timestamp_utc": sampled_frame[timestamp_column].max(),
                "sample_max_label_end_utc": sampled_frame[label_end_column].max(),
                "eligible_support": _support_records(
                    region_frame, region_labels, side_column=side_column
                ),
                "sample_support": _support_records(
                    sampled_frame, sampled_labels, side_column=side_column
                ),
            }
        )
    positions = np.concatenate(selected_regions)
    if not np.all(np.diff(positions) > 0):
        raise RuntimeError("HPO chronological regions did not preserve strict ordering")
    return positions, {
        "version": HPO_SAMPLING_CONTRACT_VERSION,
        "sampling_method": (
            "equal chronological regions; deterministic time-spread proportional "
            "sampling within each side_x_class stratum"
        ),
        "eligible_rows": int(len(frame)),
        "sample_rows": int(len(positions)),
        "validation_folds": int(validation_folds),
        "regions": report_regions,
    }


def _load_static_feature_matrix(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    feature_store_ts: pd.Timestamp,
    data_root: Path,
    timestamp_column: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read exact point-in-time feature values without an as-of join or wide panels."""
    columns = tuple(dict.fromkeys(map(str, feature_columns)))
    matrix = pd.DataFrame(index=frame.index, columns=columns, dtype=np.float32)
    timestamps = pd.DatetimeIndex(pd.to_datetime(frame[timestamp_column], utc=True))
    symbols = [str(value) for value in frame[IDENTITY_SYMBOL_COLUMN].unique()]
    static = read_static_features(
        feature_store_ts=feature_store_ts,
        data_root=data_root,
        feature_keys=columns,
        symbols=symbols,
        start_ts=timestamps.min(),
        end_ts=timestamps.max(),
    )
    loaded_symbols = 0
    read_errors: list[str] = []
    available_by_feature = {column: 0 for column in columns}
    if static is None:
        read_errors.append("static_reader_returned_none")
    else:
        for symbol, positions in frame.groupby(
            IDENTITY_SYMBOL_COLUMN, sort=False
        ).indices.items():
            rows = np.asarray(positions, dtype=np.int64)
            try:
                if hasattr(static, "symbol_frame"):
                    symbol_frame = static.symbol_frame(str(symbol), keys=columns)
                else:
                    # The production endpoint returns LazyFeatureDict. Retain a
                    # narrow compatibility path for simple test doubles.
                    symbol_frame = pd.DataFrame(
                        {
                            column: static[column][str(symbol)]
                            for column in columns
                            if column in static and str(symbol) in static[column]
                        }
                    )
                if symbol_frame.empty:
                    read_errors.append(f"{symbol}:no_static_rows")
                    continue
                symbol_frame = symbol_frame.reindex(columns=columns)
                symbol_frame.index = pd.DatetimeIndex(
                    pd.to_datetime(symbol_frame.index, utc=True, errors="coerce")
                )
                if not symbol_frame.index.is_unique:
                    raise ValueError("static feature timestamps are not one-to-one")
                aligned = symbol_frame.reindex(timestamps[rows])
                matrix.iloc[rows, :] = aligned.to_numpy(dtype=np.float32, copy=False)
                loaded_symbols += 1
                for column in columns:
                    if column in symbol_frame:
                        available_by_feature[column] += 1
            except Exception as exc:
                read_errors.append(f"{symbol}:{type(exc).__name__}:{exc}")
    finite_fraction = {
        column: float(np.isfinite(matrix[column].to_numpy(dtype=float)).mean())
        for column in columns
    }
    return matrix, {
        "static_feature_endpoint_version": STATIC_FEATURE_ENDPOINT_VERSION,
        "requested_features": len(columns),
        "requested_symbols": len(symbols),
        "loaded_symbols": loaded_symbols,
        "features_with_symbol_schema": available_by_feature,
        "finite_fraction": finite_fraction,
        "read_error_count": len(read_errors),
        "read_error_sample": read_errors[:20],
        "alignment": "exact UTC timestamp and symbol reindex; side retained in source identity",
    }


def _normalise_input(
    frame: pd.DataFrame,
    *,
    timestamp_column: str,
    label_end_column: str,
    max_rows: int,
) -> pd.DataFrame:
    missing = {timestamp_column, label_end_column}.difference(frame.columns)
    if missing:
        raise ValueError(
            f"input is missing required UTC time columns: {sorted(missing)}"
        )
    out = frame.copy()
    if "path_arch_complete_24h" in out:
        complete = (
            pd.to_numeric(out["path_arch_complete_24h"], errors="coerce")
            .fillna(0)
            .astype(bool)
        )
        out = out.loc[complete].copy()
    if "path_archetype" in out:
        target = out["path_archetype"].astype("string").str.strip()
        out = out.loc[target.notna() & target.ne("")].copy()
    out[timestamp_column] = pd.to_datetime(
        out[timestamp_column], utc=True, errors="coerce"
    )
    out[label_end_column] = pd.to_datetime(
        out[label_end_column], utc=True, errors="coerce"
    )
    invalid = out[timestamp_column].isna() | out[label_end_column].isna()
    if invalid.any():
        raise ValueError(
            f"input has {int(invalid.sum())} invalid timestamp/label-end rows"
        )
    if (out[label_end_column] < out[timestamp_column]).any():
        raise ValueError("label-end timestamps must be at or after decision timestamps")
    out = out.sort_values(timestamp_column, kind="mergesort").reset_index(drop=True)
    if max_rows > 0:
        out = out.iloc[: int(max_rows)].reset_index(drop=True)
    if out.empty:
        raise ValueError("input is empty after caps")
    return out


def _normalise_training_side(value: str) -> str:
    side = str(value).strip().lower()
    if side not in {"long", "short"}:
        raise ValueError("side must be exactly 'long' or 'short'")
    return side


def _validate_merged_class_support_gate(
    labels: pd.Series,
    timestamps: pd.Series,
    *,
    side: str,
) -> dict[str, Any]:
    """Require all seven classes to be learnable in this side-local cohort."""

    expected = tuple(map(str, MERGED_PATH_ARCHETYPE_CLASSES))
    values = labels.astype(str)
    months = pd.to_datetime(timestamps, utc=True, errors="raise").dt.strftime("%Y-%m")
    total = max(1, len(values))
    report: dict[str, Any] = {
        "contract": "merged_7class_side_month_minimum_support_v1",
        "side": side,
        "minimum_overall_share": MIN_SIDE_CLASS_SHARE,
        "minimum_monthly_share": MIN_SIDE_MONTH_CLASS_SHARE,
        "rows": int(len(values)),
        "months": sorted(months.unique().tolist()),
        "classes": {},
    }
    failures: list[str] = []
    for class_name in expected:
        overall_rows = int(values.eq(class_name).sum())
        overall_share = overall_rows / total
        monthly: dict[str, dict[str, float | int]] = {}
        for month in report["months"]:
            month_mask = months.eq(month)
            month_rows = int(month_mask.sum())
            class_rows = int((month_mask & values.eq(class_name)).sum())
            share = class_rows / max(1, month_rows)
            monthly[month] = {"rows": class_rows, "share": share}
            if share < MIN_SIDE_MONTH_CLASS_SHARE:
                failures.append(f"{class_name}/{month}={share:.4%}")
        report["classes"][class_name] = {
            "overall_rows": overall_rows,
            "overall_share": overall_share,
            "monthly": monthly,
        }
        if overall_share < MIN_SIDE_CLASS_SHARE:
            failures.append(f"{class_name}/overall={overall_share:.4%}")
    report["passed"] = not failures
    if failures:
        raise ValueError(
            "side-local merged-class support gate failed (requires >=1% overall and "
            ">=0.5% per UTC month): " + ", ".join(failures[:12])
        )
    return report


def _canonical_side_series(values: pd.Series, *, source: str) -> pd.Series:
    """Normalise one side field without accepting a mixed/unknown cohort."""

    text = values.astype("string").str.strip().str.lower()
    normalized = text.map(
        {
            "long": "long",
            "buy": "long",
            "1": "long",
            "short": "short",
            "sell": "short",
            "-1": "short",
        }
    )
    if normalized.isna().any():
        invalid = sorted(set(text.loc[normalized.isna()].dropna().astype(str)))
        raise ValueError(f"{source} contains noncanonical side values: {invalid[:8]}")
    return normalized.astype(str)


def _economic_scoring_frame(
    frame: pd.DataFrame,
    *,
    side_column: str,
    canonical_side: str | None,
) -> pd.DataFrame:
    """Restore the semantic side label after numeric feature encoding."""

    if canonical_side is None:
        return frame
    economic_frame = frame.copy()
    economic_frame[side_column] = pd.Series(
        canonical_side,
        index=economic_frame.index,
        dtype="string",
    )
    return economic_frame


def _side_candidate_identity_sha256(
    frame: pd.DataFrame,
    *,
    timestamp_column: str,
    symbol_column: str,
    side_column: str,
) -> str:
    """Hash side-local identities using canonical long/short side strings."""

    identity = frame.loc[:, [timestamp_column, symbol_column, side_column]].copy()
    identity[side_column] = _canonical_side_series(
        identity[side_column], source="candidate identity"
    )
    return candidate_identity_sha256(
        identity, columns=(timestamp_column, symbol_column, side_column)
    )


def _canonical_candidate_id_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    require_selected_top40: bool,
) -> pd.DataFrame:
    """Return exact canonical keys, rejecting synthetic or duplicate identities."""

    required = {"__ts__", "__symbol__", "candidate_id", "side"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing canonical identity columns: {missing}")
    if require_selected_top40:
        if "selected_top40" not in frame:
            raise ValueError(f"{source} is missing selected_top40")
        selected = pd.to_numeric(frame["selected_top40"], errors="coerce")
        if selected.isna().any() or not selected.isin((0, 1, True, False)).all():
            raise ValueError(f"{source} has invalid selected_top40 values")
        frame = frame.loc[selected.astype(bool)].copy()
    keys = frame.loc[:, list(CANONICAL_CANDIDATE_KEY_COLUMNS)].copy()
    keys["__ts__"] = pd.to_datetime(keys["__ts__"], utc=True, errors="coerce")
    keys["__symbol__"] = keys["__symbol__"].astype(str)
    keys["side"] = _canonical_side_series(keys["side"], source=source)
    keys["candidate_id"] = keys["candidate_id"].astype("string").str.strip()
    if (
        keys.isna().any().any()
        or keys["__symbol__"].eq("").any()
        or keys["candidate_id"].eq("").any()
    ):
        raise ValueError(f"{source} has null or blank canonical identity values")
    expected = candidate_id_series(
        keys["__ts__"], keys["__symbol__"], "1h", keys["side"]
    )
    if not np.array_equal(
        keys["candidate_id"].astype(str).to_numpy(), expected.astype(str).to_numpy()
    ):
        raise ValueError(f"{source} candidate_id does not match UTC/symbol/1h/side")
    if keys.duplicated(list(CANONICAL_CANDIDATE_KEY_COLUMNS), keep=False).any():
        raise ValueError(f"{source} has duplicate canonical candidate keys")
    if keys["candidate_id"].duplicated(keep=False).any():
        raise ValueError(f"{source} has duplicate candidate_id values")
    return keys.sort_values(
        list(CANONICAL_CANDIDATE_KEY_COLUMNS), kind="mergesort"
    ).reset_index(drop=True)


def _read_canonical_identity_parquet(path: Path) -> pd.DataFrame:
    """Read narrow canonical keys while accepting the declared side alias."""

    available = _schema_names(Path(path))
    side_source = next(
        (column for column in IDENTITY_SIDE_COLUMNS if column in available), None
    )
    if side_source is None:
        raise ValueError(
            f"{path} is missing a canonical side identity column "
            f"({', '.join(IDENTITY_SIDE_COLUMNS)})"
        )
    required = {"__ts__", "__symbol__", "candidate_id", "selected_top40"}
    missing = sorted(required.difference(available))
    if missing:
        raise ValueError(f"{path} is missing canonical identity columns: {missing}")
    columns = [*sorted(required), side_source]
    frame = pd.read_parquet(path, columns=columns)
    if side_source != "side":
        frame["side"] = frame[side_source]
    return frame


def _read_manifest_binding(path: Path, *, artifact: Path, field: str) -> dict[str, Any]:
    """Read an explicit manifest and require its advertised file digest."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"canonical {field} manifest does not exist: {path}")
    payload = _read_json_object(path, artifact_name=f"canonical {field} manifest")
    advertised: str | None = None
    for key in ("output_sha256", "sha256"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            advertised = value
            break
    output = payload.get("output")
    if advertised is None and isinstance(output, Mapping):
        value = output.get("sha256")
        if isinstance(value, str) and value:
            advertised = value
    if advertised is None:
        raise ValueError(f"canonical {field} manifest has no advertised output SHA-256")
    actual = _sha256_file(artifact)
    if advertised != actual:
        raise ValueError(
            f"canonical {field} parquet SHA-256 does not match its manifest"
        )
    return {
        "path": str(artifact.resolve()),
        "sha256": actual,
        "manifest_path": str(path.resolve()),
        "manifest_sha256": _sha256_file(path),
        "manifest": payload,
    }


def _validate_side_local_canonical_inputs(
    labels: pd.DataFrame,
    *,
    side: str,
    candidate_path: Path,
    candidate_manifest: Path,
    context_path: Path,
    context_manifest: Path,
    ae_gmm_state: Path | None,
    ae_gmm_state_manifest: Path | None,
) -> dict[str, Any]:
    """Bind one side's labels to immutable candidate/context/AE artifacts.

    This check intentionally happens before feature selection.  It prevents a
    pooled candidate file, another side's frozen representation, or a context
    row that is merely close in time from entering any learned stage.
    """

    side = _normalise_training_side(side)
    candidate_path, context_path = Path(candidate_path), Path(context_path)
    if not candidate_path.is_file() or not context_path.is_file():
        raise FileNotFoundError(
            "canonical candidate and context inputs must be parquet files"
        )
    candidate_binding = _read_manifest_binding(
        candidate_manifest, artifact=candidate_path, field="candidate"
    )
    context_binding = _read_manifest_binding(
        context_manifest, artifact=context_path, field="context"
    )
    label_keys = _canonical_candidate_id_frame(
        labels, source="path labels", require_selected_top40=False
    )
    candidate_keys = _canonical_candidate_id_frame(
        _read_canonical_identity_parquet(candidate_path),
        source=str(candidate_path),
        require_selected_top40=True,
    )
    context_keys = _canonical_candidate_id_frame(
        _read_canonical_identity_parquet(context_path),
        source=str(context_path),
        require_selected_top40=True,
    )
    label_keys = label_keys.loc[label_keys["side"].eq(side)].reset_index(drop=True)
    candidate_keys = candidate_keys.loc[candidate_keys["side"].eq(side)].reset_index(
        drop=True
    )
    context_keys = context_keys.loc[context_keys["side"].eq(side)].reset_index(
        drop=True
    )
    if label_keys.empty or candidate_keys.empty or context_keys.empty:
        raise ValueError(
            f"canonical side={side} has empty label, candidate, or context support"
        )
    candidate_index = pd.MultiIndex.from_frame(candidate_keys)
    context_index = pd.MultiIndex.from_frame(context_keys)
    label_index = pd.MultiIndex.from_frame(label_keys)
    if not candidate_index.isin(context_index).all():
        raise ValueError(
            "canonical candidate population contains rows outside the canonical context"
        )
    if not label_index.isin(candidate_index).all():
        raise ValueError(
            "path labels contain rows outside the canonical candidate population"
        )
    if not label_index.isin(context_index).all():
        raise ValueError(
            "path labels contain rows outside the canonical context population"
        )
    context_manifest_payload = context_binding["manifest"]
    ae_evidence = (
        context_manifest_payload.get("ae_gmm", {})
        .get("loader_evidence_by_side", {})
        .get(side)
    )
    if not isinstance(ae_evidence, Mapping):
        raise ValueError(
            f"canonical context manifest has no AE/GMM binding for side={side}"
        )
    ae_root = context_manifest_payload.get("ae_gmm", {}).get("root")
    default_state = Path(str(ae_root)) / side / "ae_gmm" / "ae_gmm_state.pkl"
    default_manifest = Path(str(ae_root)) / side / "ae_gmm" / "side_stage_manifest.json"
    state_path = Path(ae_gmm_state) if ae_gmm_state is not None else default_state
    state_manifest_path = (
        Path(ae_gmm_state_manifest)
        if ae_gmm_state_manifest is not None
        else default_manifest
    )
    if not state_path.is_file() or not state_manifest_path.is_file():
        raise FileNotFoundError("side-local AE/GMM state and side manifest must exist")
    expected_state_sha = ae_evidence.get("ae_state_sha256")
    expected_manifest_sha = ae_evidence.get("ae_manifest_sha256")
    if _sha256_file(state_path) != expected_state_sha:
        raise ValueError(
            "AE/GMM state hash does not match canonical context side binding"
        )
    if _sha256_file(state_manifest_path) != expected_manifest_sha:
        raise ValueError(
            "AE/GMM manifest hash does not match canonical context side binding"
        )
    side_manifest_payload = _read_json_object(
        state_manifest_path, artifact_name="side-local AE/GMM manifest"
    )
    if side_manifest_payload.get("side") != side:
        raise ValueError("AE/GMM state manifest side does not match requested side")
    if side_manifest_payload.get("artifact", {}).get("sha256") != expected_state_sha:
        raise ValueError("AE/GMM state manifest does not bind the selected state hash")
    return {
        "side": side,
        "candidate": {
            key: value for key, value in candidate_binding.items() if key != "manifest"
        },
        "context": {
            key: value for key, value in context_binding.items() if key != "manifest"
        },
        "label_rows": int(len(label_keys)),
        "candidate_rows_side": int(len(candidate_keys)),
        "context_rows_side": int(len(context_keys)),
        "exact_key_contract": "candidate_id + UTC timestamp + symbol + canonical side; label rows are exact subsets of candidate and context",
        "ae_gmm": {
            "state_path": str(state_path.resolve()),
            "state_sha256": expected_state_sha,
            "manifest_path": str(state_manifest_path.resolve()),
            "manifest_sha256": expected_manifest_sha,
            "side": side,
        },
    }


def _side_local_geometry_provenance(
    canonical_input_contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Promote immutable context/AE digests to simple geometry-facing fields."""

    if canonical_input_contract is None:
        return {
            "canonical_context_sha256": None,
            "side_ae_state_sha256": None,
            "geometry_search_model_params": dict(GEOMETRY_SEARCH_MODEL_PARAMS),
            "geometry_search_model_params_sha256": _sha256_json(
                GEOMETRY_SEARCH_MODEL_PARAMS
            ),
        }
    context = canonical_input_contract.get("context", {})
    ae_gmm = canonical_input_contract.get("ae_gmm", {})
    context_sha = context.get("sha256") if isinstance(context, Mapping) else None
    ae_sha = ae_gmm.get("state_sha256") if isinstance(ae_gmm, Mapping) else None
    if not isinstance(context_sha, str) or not isinstance(ae_sha, str):
        raise ValueError("canonical side input contract is missing context/AE hashes")
    return {
        "canonical_context_sha256": context_sha,
        "side_ae_state_sha256": ae_sha,
        "geometry_search_model_params": dict(GEOMETRY_SEARCH_MODEL_PARAMS),
        "geometry_search_model_params_sha256": _sha256_json(
            GEOMETRY_SEARCH_MODEL_PARAMS
        ),
    }


def _read_side_geometry_contract(
    path: Path | None,
    *,
    side: str | None,
    candidate_identity: str,
    selected_features: Sequence[str],
    selection_fingerprint: str,
    geometry_prerequisite_path: Path,
    canonical_input_contract: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Accept only a geometry artifact bound to this side and frozen selection.

    Geometry is intentionally external to this runner.  This narrow adapter is
    the stage-order seam: it makes model-HPO impossible to run against pooled or
    stale geometry, without reimplementing the geometry sweep here.
    """

    if path is None:
        return None
    path = Path(path)
    if path.name != "geometry_contract.json":
        raise ValueError("model-HPO/final stage requires geometry_contract.json")
    payload = _read_json_object(path, artifact_name="side-local geometry contract")
    if side is None:
        raise ValueError(
            "geometry contracts are only valid for explicit side-local runs"
        )
    if payload.get("side") != side:
        raise ValueError("geometry contract side does not match requested side")
    if payload.get("candidate_identity_sha256") != candidate_identity:
        raise ValueError(
            "geometry contract candidate identity does not match this side"
        )
    if payload.get("selection_fingerprint") != selection_fingerprint:
        raise ValueError("geometry contract selection fingerprint does not match")
    geometry_features = payload.get("selected_features")
    if list(geometry_features or []) != list(selected_features):
        raise ValueError(
            "geometry contract selected features do not match frozen selection"
        )
    if payload.get("status") not in {"geometry_complete", "selected"}:
        raise ValueError(
            "geometry contract is not a completed side-local geometry selection"
        )
    prerequisite_path = Path(geometry_prerequisite_path)
    prerequisite = _read_json_object(
        prerequisite_path, artifact_name="geometry prerequisite"
    )
    expected_prerequisite_sha = _sha256_file(prerequisite_path)
    if payload.get("geometry_prerequisite_sha256") != expected_prerequisite_sha:
        raise ValueError("geometry contract does not bind the exact selection handoff")
    expected_provenance = _side_local_geometry_provenance(canonical_input_contract)
    for key in (
        "canonical_context_sha256",
        "side_ae_state_sha256",
        "geometry_search_model_params_sha256",
    ):
        if prerequisite.get(key) != expected_provenance[key]:
            raise ValueError(f"geometry prerequisite {key} does not match this run")
        if payload.get(key) != expected_provenance[key]:
            raise ValueError(f"geometry contract {key} does not match this run")
    if prerequisite.get("candidate_identity_sha256") != candidate_identity:
        raise ValueError("geometry prerequisite candidate identity does not match")
    if prerequisite.get("selection_fingerprint") != selection_fingerprint:
        raise ValueError("geometry prerequisite selection fingerprint does not match")
    if list(prerequisite.get("selected_features") or []) != list(selected_features):
        raise ValueError("geometry prerequisite selected features do not match")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "schema": payload.get("schema"),
        "status": payload.get("status"),
        "side": side,
        "candidate_identity_sha256": candidate_identity,
        "selection_fingerprint": selection_fingerprint,
        "selected_features": list(selected_features),
        "geometry_prerequisite_path": str(prerequisite_path.resolve()),
        "geometry_prerequisite_sha256": expected_prerequisite_sha,
        **expected_provenance,
    }


def _path_summaries(
    frame: pd.DataFrame, *, future_path_column: str | None
) -> tuple[pd.DataFrame, str]:
    existing = [
        column
        for column in frame.columns
        if str(column).startswith(archetype.PATH_SUMMARY_PREFIX)
        and column != "path_arch_complete_24h"
    ]
    if existing:
        expected = set(archetype.path_summary_columns())
        missing = expected.difference(existing)
        if missing:
            raise ValueError(
                f"materialized path summaries are incomplete: missing {sorted(missing)}"
            )
        return frame.loc[:, existing].copy(), "materialized_path_arch_summaries"
    if not future_path_column:
        raise ValueError(
            "input needs complete path_arch_* summaries or --future-path-column"
        )
    raise ValueError(
        "array-only future paths are not valid for the current CatBoost archetype "
        "contract because they do not preserve asset ATR, cost, activation-distance, "
        "or causal OHLC path semantics; materialize canonical path_arch_* summaries"
    )


def _assert_preentry_only(columns: Sequence[str]) -> None:
    archetype.validate_preentry_features(columns)
    realized = [
        column
        for column in columns
        if str(column).lower().startswith(archetype.PATH_SUMMARY_PREFIX)
    ]
    if realized:
        raise ValueError(
            "realized path summaries must never be classifier inputs: "
            + ", ".join(map(str, realized[:8]))
        )


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
    return -np.sum(values * np.log(values), axis=1)


def _aligned_oof_probabilities(
    probabilities: np.ndarray,
    class_names: Sequence[str],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Align OOF output to the frozen future-training class order."""

    observed = tuple(map(str, class_names))
    aligned = np.zeros(
        (len(probabilities), len(MERGED_PATH_ARCHETYPE_CLASSES)), dtype=float
    )
    positions = {
        name: index for index, name in enumerate(MERGED_PATH_ARCHETYPE_CLASSES)
    }
    for source, name in enumerate(observed):
        if name not in positions:
            raise ValueError(f"OOF emitted a class outside the merged taxonomy: {name}")
        aligned[:, positions[name]] = probabilities[:, source]
    return aligned, MERGED_PATH_ARCHETYPE_CLASSES


def _consolidate_sparse_supervised_classes(
    raw_labels: pd.Series,
    discovery_mask: pd.Series,
    *,
    min_class_rows: int,
) -> tuple[pd.Series, pd.DataFrame]:
    """Consolidate sparse strength bands from discovery-train support only."""
    if min_class_rows < 1:
        raise ValueError("min_class_rows must be at least one")
    labels = raw_labels.astype("string").str.strip()
    mask = pd.Series(discovery_mask, index=labels.index).fillna(False).astype(bool)
    discovery = labels.loc[mask].astype(str)
    if discovery.empty:
        raise ValueError("sparse-class consolidation has no discovery rows")
    counts = discovery.value_counts().to_dict()
    strength_order = {
        strength: index
        for index, strength in enumerate(PATH_REALIZATION_STRENGTH_TYPES)
    }
    mapping: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for raw_class in sorted(set(map(str, labels.unique()))):
        shape, strength = raw_class.rsplit("__", 1)
        if strength not in strength_order:
            raise ValueError(f"unknown path strength in class {raw_class!r}")
        raw_support = int(counts.get(raw_class, 0))
        candidates = [
            f"{shape}__{candidate}"
            for candidate in PATH_REALIZATION_STRENGTH_TYPES
            if int(counts.get(f"{shape}__{candidate}", 0)) >= min_class_rows
        ]
        if raw_support >= min_class_rows:
            effective = raw_class
        else:
            if not candidates:
                candidates = [
                    f"{shape}__{candidate}"
                    for candidate in PATH_REALIZATION_STRENGTH_TYPES
                    if int(counts.get(f"{shape}__{candidate}", 0)) > 0
                ]
            if not candidates:
                raise ValueError(
                    f"path shape {shape!r} has no discovery support for consolidation"
                )
            effective = min(
                candidates,
                key=lambda candidate: (
                    abs(
                        strength_order[candidate.rsplit("__", 1)[1]]
                        - strength_order[strength]
                    ),
                    -int(counts.get(candidate, 0)),
                    candidate,
                ),
            )
        mapping[raw_class] = effective
        rows.append(
            {
                "raw_path_archetype": raw_class,
                "effective_path_archetype": effective,
                "raw_discovery_rows": raw_support,
                "effective_discovery_rows": int(counts.get(effective, 0)),
                "was_consolidated": bool(effective != raw_class),
                "min_class_rows": int(min_class_rows),
            }
        )
    effective_labels = labels.astype(str).map(mapping).astype("string")
    if effective_labels.isna().any():
        raise ValueError("sparse-class consolidation left unmapped labels")
    return effective_labels, pd.DataFrame(rows)


def _validate_oof_class_support(
    labels: pd.Series,
    timestamps: pd.Series,
    label_end: pd.Series,
    *,
    config: archetype.PathArchetypeConfig,
) -> None:
    """Fail clearly when CatBoost cannot evaluate a chronological fold."""
    folds = archetype.purged_chronological_folds(
        timestamps,
        label_end=label_end,
        n_splits=config.oof_folds,
        embargo=config.embargo,
    )
    viable = 0
    for fold in folds:
        train_classes = set(labels.iloc[fold.train_indices].astype(str))
        if len(train_classes) < 2:
            continue
        viable += 1
        valid_classes = set(labels.iloc[fold.validation_indices].astype(str))
        unseen = sorted(valid_classes.difference(train_classes))
        if unseen:
            raise ValueError(
                "purged OOF fold has validation archetypes absent from its train period; "
                f"fold={fold.fold_id}, unseen={unseen}. Increase chronological support, "
                "reduce cluster count, or use a later discovery boundary."
            )
    if not viable:
        raise ValueError(
            "purge/embargo leaves no chronological OOF fold with at least two train archetypes"
        )


def _validate_hpo_sample_class_support(
    frame: pd.DataFrame,
    labels: pd.Series,
    *,
    timestamp_column: str,
    label_end_column: str,
    side_column: str,
    config: archetype.PathArchetypeConfig,
) -> list[dict[str, Any]]:
    """Require each sampled validation class to be trainable after purge/embargo."""
    folds = archetype.purged_chronological_folds(
        frame[timestamp_column],
        label_end=frame[label_end_column],
        n_splits=config.oof_folds,
        embargo=config.embargo,
    )
    reports: list[dict[str, Any]] = []
    for fold in folds:
        train_frame = frame.iloc[fold.train_indices]
        validation_frame = frame.iloc[fold.validation_indices]
        train_labels = labels.iloc[fold.train_indices]
        validation_labels = labels.iloc[fold.validation_indices]
        unseen = sorted(
            set(validation_labels.astype(str)).difference(train_labels.astype(str))
        )
        report = {
            "fold_id": int(fold.fold_id),
            "train_rows": int(len(train_frame)),
            "validation_rows": int(len(validation_frame)),
            "train_start_timestamp_utc": train_frame[timestamp_column].min(),
            "train_end_timestamp_utc": train_frame[timestamp_column].max(),
            "train_max_label_end_utc": train_frame[label_end_column].max(),
            "validation_start_timestamp_utc": validation_frame[timestamp_column].min(),
            "validation_end_timestamp_utc": validation_frame[timestamp_column].max(),
            "unseen_validation_classes": unseen,
            "train_support": _support_records(
                train_frame, train_labels, side_column=side_column
            ),
            "validation_support": _support_records(
                validation_frame, validation_labels, side_column=side_column
            ),
        }
        reports.append(report)
        if unseen:
            raise ValueError(
                "HPO sampled fold has validation classes absent from prior sampled "
                f"training support after purge/embargo; fold={fold.fold_id}, "
                f"unseen={unseen}"
            )
    if not reports:
        raise ValueError("HPO sample has no viable purged chronological folds")
    return reports


def _fit_final_classifier(
    features: pd.DataFrame,
    target: pd.Series,
    selected_features: Sequence[str],
    *,
    config: archetype.PathArchetypeConfig,
    params: Mapping[str, Any],
) -> archetype.PathArchetypeClassifier:
    CatBoostClassifier = archetype._require_catboost()
    final_params = archetype._catboost_params(config, params)
    final_params["use_best_model"] = False
    categories = archetype._categorical_target(target, features.index, config=config)
    model = CatBoostClassifier(**final_params)
    model.fit(
        archetype._finite_matrix(features, selected_features),
        categories.cat.codes.to_numpy(),
    )
    return archetype.PathArchetypeClassifier(
        feature_columns=tuple(selected_features),
        class_names=tuple(map(str, categories.cat.categories)),
        model=model,
        config=config,
    )


def _selector_manifest(result: archetype.FastSelectorResult) -> dict[str, Any]:
    return {
        "selected_features": list(result.selected_features),
        "candidate_features": list(result.candidate_features),
        "mandatory_features": list(result.mandatory_features),
        "availability": dict(result.availability),
        "correlation_clusters": [
            list(cluster) for cluster in result.correlation_clusters
        ],
        "proxy_backend": result.proxy_backend,
        "scores": result.scores.reset_index(names="feature").to_dict(orient="records"),
    }


_PERMUTATION_STAGE_METRIC_COLUMNS = (
    "stage_acceleration_algorithm_version",
    "stage_input_feature_count",
    "stage_keep_count",
    "stage_full_mda_candidate_count",
    "stage_screened_out_count",
    "stage_screening_used",
    "stage_screen_fold_ids",
    "stage_screen_fold_count",
    "stage_screen_aggregation",
    "stage_screen_cutoff_loss",
    "stage_selection_semantics",
    "stage_reused_oof_models",
    "stage_fit_calls",
    "stage_baseline_predict_calls",
    "stage_permutation_predict_calls",
    "stage_max_permutation_batch_size",
    "stage_matrix_cache_bytes",
    "stage_validation_matrix_cache_bytes",
    "stage_validation_matrix_cache_used",
    "stage_matrix_dtype",
    "stage_fit_seconds",
    "stage_baseline_predict_seconds",
    "stage_screen_seconds",
    "stage_permutation_predict_seconds",
    "stage_total_seconds",
)


def _permutation_stage_metrics(
    permutation: pd.DataFrame | Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Persist one timing/candidate record per MDA stage, not per feature."""
    frame = (
        permutation.copy()
        if isinstance(permutation, pd.DataFrame)
        else pd.DataFrame(list(permutation))
    )
    if "stage" not in frame:
        return []
    available = [
        column for column in _PERMUTATION_STAGE_METRIC_COLUMNS if column in frame
    ]
    if not available:
        return []
    return (
        frame.loc[:, ["stage", *available]]
        .drop_duplicates(subset=["stage"], keep="first")
        .sort_values("stage", kind="stable")
        .to_dict(orient="records")
    )


def _target_geometry_contract(
    frame: pd.DataFrame,
    summaries: pd.DataFrame,
    effective_labels: pd.Series,
    *,
    timestamp_column: str,
    side_column: str,
    taxonomy_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the target values and geometry that supervised selection consumed."""
    cost_and_meaningful_mfe_columns = (
        "path_arch_cost_atr",
        "path_arch_meaningful_mfe_threshold_atr",
        "path_arch_mfe_to_cost",
        "path_arch_peak_mfe_atr",
        "path_arch_peak_mfe_minus_cost_atr",
        "path_arch_peak_mfe_div_cost",
        "path_arch_reaches_meaningful_mfe",
        "path_arch_time_to_first_meaningful_mfe_h",
        "path_arch_bars_to_meaningful_mfe",
        "path_arch_stop_before_meaningful_mfe",
    )
    target_columns = tuple(
        dict.fromkeys(
            (
                timestamp_column,
                IDENTITY_SYMBOL_COLUMN,
                side_column,
                "path_archetype_rule_version",
                "path_archetype",
                "path_shape_archetype",
                *(
                    ("path_realization_strength",)
                    if "path_realization_strength" in frame.columns
                    else ()
                ),
                *sorted(map(str, summaries.columns)),
                *sorted(
                    column
                    for column in PATH_TARGET_PROVENANCE_COLUMNS
                    if column in frame.columns
                ),
            )
        )
    )
    target_values = frame.loc[:, list(target_columns)].copy()
    target_values["classifier_path_archetype"] = effective_labels.to_numpy()
    return {
        "target": "classifier_path_archetype",
        "rule_version": PATH_ARCHETYPE_RULE_VERSION,
        "path_shape_types": list(PATH_SHAPE_TYPES),
        "path_realization_strength_types": list(PATH_REALIZATION_STRENGTH_TYPES),
        "path_archetype_types": list(PATH_ARCHETYPE_TYPES),
        "path_summary_columns": sorted(map(str, summaries.columns)),
        "cost_and_meaningful_mfe_columns": [
            column for column in cost_and_meaningful_mfe_columns if column in frame
        ],
        "target_value_columns": [*target_columns, "classifier_path_archetype"],
        "target_value_sha256": _frame_contract_sha256(
            target_values, [*target_columns, "classifier_path_archetype"]
        ),
        "future_training_taxonomy": dict(taxonomy_contract),
    }


def _feature_selection_hpo_fingerprint(
    *,
    frame: pd.DataFrame,
    summaries: pd.DataFrame,
    effective_labels: pd.Series,
    timestamp_column: str,
    side_column: str,
    frozen_sidecar_contract: Mapping[str, Any] | None,
    universe: Sequence[str],
    model_universe: Sequence[str],
    mandatory_features: Sequence[str],
    config: archetype.PathArchetypeConfig,
    effective_trials: int,
    hpo_rows: int,
    hpo_folds: int,
    hpo_iterations: int,
    hpo_od_wait: int,
    hpo_no_improvement_trials: int,
    selection_iterations: int,
    selection_od_wait: int,
    smoke: bool,
    hpo_sample_contract: Mapping[str, Any],
    structural_hpo_contract: Mapping[str, Any],
    taxonomy_contract: Mapping[str, Any],
    side: str | None,
    canonical_input_contract: Mapping[str, Any] | None,
    geometry_contract_path: Path | None,
    development_oos_routing: Mapping[str, Any],
) -> dict[str, Any]:
    target_contract = _target_geometry_contract(
        frame,
        summaries,
        effective_labels,
        timestamp_column=timestamp_column,
        side_column=side_column,
        taxonomy_contract=taxonomy_contract,
    )
    sidecar_contract = (
        {
            "sidecar_sha256": frozen_sidecar_contract.get("sidecar_sha256"),
            "manifest_sha256": frozen_sidecar_contract.get("manifest_sha256"),
            "output_features": frozen_sidecar_contract.get("output_features"),
            "key_columns": frozen_sidecar_contract.get("key_columns"),
            "join_contract": frozen_sidecar_contract.get("join_contract"),
        }
        if frozen_sidecar_contract is not None
        else None
    )
    inputs = {
        "side_local_training": {
            "side": side,
            "development_oos_routing": dict(development_oos_routing),
            "canonical_input_contract": (
                dict(canonical_input_contract)
                if canonical_input_contract is not None
                else None
            ),
            "geometry_contract_sha256": (
                _sha256_file(geometry_contract_path)
                if geometry_contract_path is not None
                else None
            ),
        },
        "candidate_identity_sha256": _side_candidate_identity_sha256(
            frame,
            timestamp_column=timestamp_column,
            symbol_column=IDENTITY_SYMBOL_COLUMN,
            side_column=side_column,
        ),
        "frozen_ae_gmm_sidecar_contract_sha256": _sha256_json(sidecar_contract),
        "eligible_initial_feature_contract": {
            "configured_universe": list(map(str, universe)),
            "eligible_model_universe": list(map(str, model_universe)),
            "mandatory_features": list(map(str, mandatory_features)),
        },
        "target_archetype_geometry_contract": target_contract,
        "selection_hpo_settings": {
            "selector_sample_rows": int(config.selector_sample_rows),
            "permutation_stages": list(config.permutation_stages),
            "permutation_execution_contract": (
                archetype.staged_permutation_acceleration_contract(config)
            ),
            "random_state": int(config.random_state),
            "oof_folds": int(config.oof_folds),
            "embargo_hours": float(config.embargo / pd.Timedelta(hours=1)),
            "catboost_resource_contract": archetype.catboost_resource_contract(config),
            "hpo_trials": int(effective_trials),
            "hpo_rows": int(hpo_rows),
            "hpo_folds": int(hpo_folds),
            "hpo_search_iterations": int(hpo_iterations),
            "hpo_search_od_wait": int(hpo_od_wait),
            "selection_proxy_iterations": int(selection_iterations),
            "selection_proxy_od_wait": int(selection_od_wait),
            "hpo_no_improvement_patience_trials": int(hpo_no_improvement_trials),
            "hpo_sampling_contract": dict(hpo_sample_contract),
            # This is deliberately HPO-only: balance-arm selection must
            # invalidate a reusable HPO result, but must not force the
            # already-leakage-safe feature-selection stage to rerun.
            "hpo_structural_contract": dict(structural_hpo_contract),
            "hpo_structural_contract_sha256": _sha256_json(structural_hpo_contract),
            "smoke": bool(smoke),
        },
    }
    selection_fingerprint = _selection_only_fingerprint(inputs)
    return {
        "schema": FEATURE_SELECTION_HPO_CONTRACT_SCHEMA,
        "fingerprint": _sha256_json(inputs),
        "fingerprint_inputs": inputs,
        "selection_fingerprint": selection_fingerprint,
    }


def _selection_only_fingerprint(inputs: Mapping[str, Any]) -> str:
    """Hash only the candidate/target/feature-selection inputs, never HPO inputs."""
    settings = dict(inputs.get("selection_hpo_settings", {}))
    selection_settings = {
        key: value for key, value in settings.items() if not key.startswith("hpo_")
    }
    selection_inputs = {
        key: value for key, value in inputs.items() if key != "selection_hpo_settings"
    }
    # Geometry is deliberately selected after feature selection.  It belongs to
    # the downstream model-HPO fingerprint, never the reusable selector state.
    side_local = dict(selection_inputs.get("side_local_training", {}))
    side_local.pop("geometry_contract_sha256", None)
    selection_inputs["side_local_training"] = side_local
    selection_inputs["selection_settings"] = selection_settings
    return _sha256_json(selection_inputs)


def _class_balance_artifact_provenance(
    *,
    structural_hpo_contract: Mapping[str, Any],
    mini_sweep: archetype.CatBoostClassBalanceMiniSweepResult,
    economic_report: Mapping[str, Any],
    economic_config: EconomicOOFConfig,
    structural_fingerprint: str,
    feature_fingerprint: str,
    geometry_fingerprint: str,
    mini_sweep_report_path: Path,
    economic_report_path: Path,
    economic_evidence_sha256: str,
) -> dict[str, Any]:
    """Bind economic arm selection to its matched, full-population OOF sweep."""

    selection = economic_report.get("selection_provenance")
    if not isinstance(selection, Mapping):
        raise ValueError("economic balance scorer did not return selection provenance")
    selected_arm = selection.get("arm")
    if not isinstance(selected_arm, str) or not selected_arm:
        raise ValueError("economic balance scorer did not select a declared arm")
    selected = {arm.arm: arm for arm in mini_sweep.arms}.get(selected_arm)
    if selected is None or selected.oof is None:
        raise ValueError(
            "economic balance scorer selected an arm without full OOF output"
        )
    guard = dict(selected.guard)
    return {
        "structural_hpo_contract": dict(structural_hpo_contract),
        "structural_hpo_contract_sha256": _sha256_json(structural_hpo_contract),
        "structural_fingerprint": structural_fingerprint,
        "selected_feature_fingerprint": feature_fingerprint,
        "geometry_fingerprint": geometry_fingerprint,
        "mini_sweep_contract": dict(mini_sweep.contract),
        "mini_sweep_contract_sha256": _sha256_json(mini_sweep.contract),
        "mini_sweep_report": mini_sweep_report_path.name,
        "mini_sweep_report_sha256": _sha256_file(mini_sweep_report_path),
        "economic_oof_report": economic_report_path.name,
        "economic_oof_report_sha256": economic_evidence_sha256,
        "economic_oof_report_file_sha256": _sha256_file(economic_report_path),
        "economic_oof_config": dict(economic_config.__dict__),
        "economic_oof_config_sha256": _sha256_json(economic_config.__dict__),
        "selected_arm": selected_arm
        if bool(selection.get("promotion_eligible"))
        else None,
        "provisional_arm": None
        if bool(selection.get("promotion_eligible"))
        else selected_arm,
        "promotion_eligible": bool(selection.get("promotion_eligible", False)),
        "selection_status": selection.get("selection_status"),
        "selection_provenance": dict(selection),
        "selected_arm_oof_guard": guard,
        "selected_arm_oof_guard_sha256": _sha256_json(guard),
        "selected_arm_fold_balance_provenance": selected.fold_balance_provenance,
        "selected_arm_fold_balance_provenance_sha256": _sha256_json(
            selected.fold_balance_provenance
        ),
        "final_refit_used_for_selection": False,
    }


def _read_feature_selection_hpo_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"feature-selection/HPO checkpoint is not an object: {path}")
    required = {
        "schema",
        "status",
        "fingerprint",
        "selected_features",
        "effective_model_params",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(
            "feature-selection/HPO checkpoint is incomplete: " + ", ".join(missing)
        )
    if payload["schema"] != FEATURE_SELECTION_HPO_CONTRACT_SCHEMA:
        raise ValueError(
            f"feature-selection/HPO checkpoint has unsupported schema: {path}"
        )
    if payload["status"] != "feature_selection_hpo_complete":
        raise ValueError(f"feature-selection/HPO checkpoint is not complete: {path}")
    if not isinstance(payload["selected_features"], list) or not all(
        isinstance(value, str) for value in payload["selected_features"]
    ):
        raise ValueError(
            f"feature-selection/HPO checkpoint has invalid selected features: {path}"
        )
    if not isinstance(payload["effective_model_params"], Mapping):
        raise ValueError(
            f"feature-selection/HPO checkpoint has invalid model parameters: {path}"
        )
    return payload


def _read_resumable_feature_selection_checkpoint(
    path: Path, expected_selection_fingerprint: str
) -> dict[str, Any]:
    """Read only this runner's exact pre-HPO selection checkpoint."""
    payload = _read_json_object(path, artifact_name="feature-selection checkpoint")
    required = {
        "schema",
        "status",
        "fingerprint",
        "selected_features",
        "selection",
        "permutation",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(
            "feature-selection checkpoint is incomplete and cannot be resumed: "
            + ", ".join(missing)
        )
    if (
        payload["schema"] != RUNNER_SCHEMA
        or payload["status"] != "feature_selection_complete"
    ):
        raise ValueError(
            "feature-selection checkpoint is not a resumable current-run checkpoint"
        )
    stored_fingerprint = payload.get("selection_fingerprint", payload["fingerprint"])
    if stored_fingerprint != expected_selection_fingerprint:
        raise ValueError(
            "feature-selection checkpoint fingerprint does not match the current exact contract for selection"
        )
    if not isinstance(payload["selected_features"], list) or not all(
        isinstance(value, str) for value in payload["selected_features"]
    ):
        raise ValueError("feature-selection checkpoint has invalid selected features")
    if not isinstance(payload["selection"], Mapping) or not isinstance(
        payload["permutation"], list
    ):
        raise ValueError("feature-selection checkpoint has invalid selection evidence")
    return payload


def _read_resumable_mda_progress(
    path: Path, expected_selection_fingerprint: str
) -> dict[str, Any] | None:
    """Read only an exact in-progress MDA stage checkpoint for this run."""
    payload = _read_json_object(path, artifact_name="MDA progress checkpoint")
    if payload.get("status") == "mda_complete":
        return None
    required = {
        "schema",
        "status",
        "fingerprint",
        "initial_selected_features",
        "selection",
        "completed_stages",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError("MDA progress checkpoint is incomplete: " + ", ".join(missing))
    if payload["schema"] != RUNNER_SCHEMA or payload["status"] != "mda_running":
        raise ValueError("MDA progress checkpoint is not resumable for this runner")
    stored_fingerprint = payload.get("selection_fingerprint", payload["fingerprint"])
    if stored_fingerprint != expected_selection_fingerprint:
        raise ValueError(
            "MDA progress checkpoint does not match the current exact selection contract"
        )
    if not isinstance(payload["initial_selected_features"], list) or not all(
        isinstance(value, str) for value in payload["initial_selected_features"]
    ):
        raise ValueError(
            "MDA progress checkpoint has invalid initial selected features"
        )
    if not isinstance(payload["selection"], Mapping) or not isinstance(
        payload["completed_stages"], list
    ):
        raise ValueError("MDA progress checkpoint has invalid stage state")
    return payload


def _checkpoint_file(path: Path) -> Path:
    path = Path(path)
    return path / FEATURE_SELECTION_HPO_CONTRACT_FILENAME if path.is_dir() else path


def _read_json_object(path: Path, *, artifact_name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{artifact_name} must contain a JSON object: {path}")
    return payload


def _adopt_explicit_legacy_feature_selection_hpo_contract(
    run_dir: Path, expected: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert one explicitly named legacy run into the current strict contract."""
    run_dir = Path(run_dir)
    artifact_paths = {
        "feature_selection_checkpoint": run_dir / "feature_selection_checkpoint.json",
        "hpo_checkpoint": run_dir / "hpo_checkpoint.json",
        "run_manifest": run_dir / "run_manifest.json",
    }
    missing = [name for name, path in artifact_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "legacy feature-selection/HPO adoption requires: " + ", ".join(missing)
        )
    feature_selection = _read_json_object(
        artifact_paths["feature_selection_checkpoint"],
        artifact_name="legacy feature-selection checkpoint",
    )
    hpo = _read_json_object(
        artifact_paths["hpo_checkpoint"], artifact_name="legacy HPO checkpoint"
    )
    run_manifest = _read_json_object(
        artifact_paths["run_manifest"], artifact_name="legacy run manifest"
    )
    if feature_selection.get("status") != "feature_selection_complete":
        raise ValueError("legacy feature-selection checkpoint is not complete")
    if hpo.get("status") != "hpo_complete":
        raise ValueError("legacy HPO checkpoint is not complete")

    expected_candidate_hash = str(
        expected.get("fingerprint_inputs", {}).get("candidate_identity_sha256", "")
    )
    legacy_candidate_hash = run_manifest.get("candidate_identity_sha256")
    if not expected_candidate_hash:
        raise ValueError("current strict contract is missing candidate identity")
    if not isinstance(legacy_candidate_hash, str) or not legacy_candidate_hash:
        raise ValueError(
            "legacy run manifest is missing candidate_identity_sha256; refusing adoption"
        )
    if legacy_candidate_hash != expected_candidate_hash:
        raise ValueError(
            "legacy run candidate_identity_sha256 does not match the current candidate population"
        )

    selected = feature_selection.get("selected_features")
    hpo_selected = hpo.get("selected_features")
    params = hpo.get("effective_model_params")
    if not isinstance(selected, list) or not all(
        isinstance(value, str) for value in selected
    ):
        raise ValueError(
            "legacy feature-selection checkpoint has invalid selected features"
        )
    if hpo_selected != selected:
        raise ValueError(
            "legacy feature-selection and HPO checkpoints disagree on selected features"
        )
    if not isinstance(params, Mapping):
        raise ValueError("legacy HPO checkpoint has invalid effective model parameters")

    provenance = {
        "mode": "explicit_legacy_adoption",
        "reused": True,
        "path": str(run_dir),
        "legacy_candidate_identity_sha256": legacy_candidate_hash,
        "legacy_artifact_sha256": {
            name: _sha256_file(path) for name, path in artifact_paths.items()
        },
    }
    contract = {
        **expected,
        "status": "feature_selection_hpo_complete",
        "selected_features": list(selected),
        "effective_model_params": dict(params),
        "selection": feature_selection.get("selection", {}),
        "permutation": feature_selection.get("permutation", []),
        "hpo": hpo.get("hpo", {}),
        "legacy_adoption": provenance,
    }
    return contract, provenance


def _find_reusable_feature_selection_hpo_contract(
    expected: Mapping[str, Any],
    *,
    output_dir: Path,
    checkpoint_path: Path | None,
    registry_root: Path | None,
    force: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Find only an exact completed contract; explicit mismatch is an error."""
    expected_fingerprint = str(expected["fingerprint"])
    if force:
        return None, {"mode": "forced_rerun", "reused": False}
    if checkpoint_path is not None:
        requested_path = Path(checkpoint_path)
        path = _checkpoint_file(requested_path)
        if requested_path.is_dir() and not path.is_file():
            return _adopt_explicit_legacy_feature_selection_hpo_contract(
                requested_path, expected
            )
        if not path.is_file():
            raise FileNotFoundError(
                f"feature-selection/HPO checkpoint does not exist: {path}"
            )
        contract = _read_feature_selection_hpo_contract(path)
        if contract["fingerprint"] != expected_fingerprint:
            raise ValueError(
                "explicit feature-selection/HPO checkpoint fingerprint does not match "
                "the current candidate, sidecar, target, feature, and HPO contract"
            )
        return contract, {
            "mode": "explicit_checkpoint",
            "reused": True,
            "path": str(path),
        }

    root = Path(registry_root) if registry_root is not None else output_dir.parent
    if not root.is_dir():
        return None, {
            "mode": "automatic_registry",
            "reused": False,
            "registry_root": str(root),
            "candidates": 0,
            "mismatched_candidates": 0,
        }
    candidates: list[tuple[float, Path, dict[str, Any]]] = []
    mismatched_candidates = 0
    for path in root.rglob(FEATURE_SELECTION_HPO_CONTRACT_FILENAME):
        if path.parent == output_dir:
            continue
        try:
            contract = _read_feature_selection_hpo_contract(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if contract["fingerprint"] == expected_fingerprint:
            candidates.append((path.stat().st_mtime, path, contract))
        else:
            mismatched_candidates += 1
    if not candidates:
        return None, {
            "mode": "automatic_registry",
            "reused": False,
            "registry_root": str(root),
            "candidates": 0,
            "mismatched_candidates": mismatched_candidates,
        }
    _, path, contract = max(candidates, key=lambda item: (item[0], str(item[1])))
    return contract, {
        "mode": "automatic_registry",
        "reused": True,
        "registry_root": str(root),
        "path": str(path),
        "candidates": len(candidates),
        "mismatched_candidates": mismatched_candidates,
    }


def _read_reusable_selection_checkpoint(path: Path) -> dict[str, Any]:
    """Read completed MDA evidence after a parent contract proves its identity."""
    payload = _read_json_object(
        path, artifact_name="reusable feature-selection checkpoint"
    )
    required = {"schema", "status", "selected_features", "selection", "permutation"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(
            "reusable feature-selection checkpoint is incomplete: " + ", ".join(missing)
        )
    if (
        payload["schema"] != RUNNER_SCHEMA
        or payload["status"] != "feature_selection_complete"
    ):
        raise ValueError("reusable feature-selection checkpoint is not complete")
    if not isinstance(payload["selected_features"], list) or not all(
        isinstance(value, str) for value in payload["selected_features"]
    ):
        raise ValueError(
            "reusable feature-selection checkpoint has invalid selected features"
        )
    if not isinstance(payload["selection"], Mapping) or not isinstance(
        payload["permutation"], list
    ):
        raise ValueError(
            "reusable feature-selection checkpoint has invalid selection evidence"
        )
    return payload


def _selection_fingerprint_from_completed_contract(
    contract: Mapping[str, Any],
) -> str | None:
    value = contract.get("selection_fingerprint")
    if isinstance(value, str) and value:
        return value
    inputs = contract.get("fingerprint_inputs")
    if not isinstance(inputs, Mapping):
        return None
    return _selection_only_fingerprint(inputs)


def _find_reusable_feature_selection_checkpoint(
    expected_selection_fingerprint: str,
    *,
    output_dir: Path,
    checkpoint_path: Path | None,
    registry_root: Path | None,
    force: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Reuse MDA only when the completed parent proves an exact selection contract."""
    if force:
        return None, {"mode": "forced_rerun", "reused": False}

    def candidate(directory: Path) -> tuple[dict[str, Any], Path] | None:
        contract_path = directory / FEATURE_SELECTION_HPO_CONTRACT_FILENAME
        selection_path = directory / "feature_selection_checkpoint.json"
        if not contract_path.is_file() or not selection_path.is_file():
            return None
        try:
            contract = _read_feature_selection_hpo_contract(contract_path)
            selection = _read_reusable_selection_checkpoint(selection_path)
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        if (
            _selection_fingerprint_from_completed_contract(contract)
            != expected_selection_fingerprint
        ):
            return None
        return selection, contract_path

    if checkpoint_path is not None:
        requested = Path(checkpoint_path)
        directory = requested if requested.is_dir() else requested.parent
        found = candidate(directory)
        if found is None:
            return None, {"mode": "explicit_selection_checkpoint", "reused": False}
        selection, contract_path = found
        return selection, {
            "mode": "explicit_selection_checkpoint",
            "reused": True,
            "path": str(contract_path),
            "selection_only": True,
        }

    root = Path(registry_root) if registry_root is not None else output_dir.parent
    if not root.is_dir():
        return None, {
            "mode": "automatic_selection_registry",
            "reused": False,
            "registry_root": str(root),
            "candidates": 0,
        }
    candidates: list[tuple[float, Path, dict[str, Any]]] = []
    for contract_path in root.rglob(FEATURE_SELECTION_HPO_CONTRACT_FILENAME):
        if contract_path.parent == output_dir:
            continue
        found = candidate(contract_path.parent)
        if found is None:
            continue
        selection, path = found
        candidates.append((path.stat().st_mtime, path, selection))
    if not candidates:
        return None, {
            "mode": "automatic_selection_registry",
            "reused": False,
            "registry_root": str(root),
            "candidates": 0,
        }
    _, path, selection = max(candidates, key=lambda item: (item[0], str(item[1])))
    return selection, {
        "mode": "automatic_selection_registry",
        "reused": True,
        "registry_root": str(root),
        "path": str(path),
        "candidates": len(candidates),
        "selection_only": True,
    }


def _validate_reused_selected_features(
    selected_features: Sequence[str], model_universe: Sequence[str]
) -> list[str]:
    selected = list(dict.fromkeys(map(str, selected_features)))
    if not selected:
        raise ValueError("reused feature-selection/HPO checkpoint selected no features")
    unavailable = sorted(set(selected).difference(model_universe))
    if unavailable:
        raise ValueError(
            "reused feature-selection/HPO checkpoint selected features unavailable "
            "from the current eligible feature contract: " + ", ".join(unavailable[:8])
        )
    _assert_preentry_only(selected)
    return selected


def run_pipeline(
    input_data: pd.DataFrame | Path,
    output_dir: Path,
    *,
    discovery_end: str | pd.Timestamp,
    development_end: str | pd.Timestamp | None = None,
    side: str | None = None,
    feature_dir: Path | None = None,
    canonical_candidate_path: Path | None = None,
    canonical_candidate_manifest: Path | None = None,
    canonical_context_path: Path | None = None,
    canonical_context_manifest: Path | None = None,
    ae_gmm_state: Path | None = None,
    ae_gmm_state_manifest: Path | None = None,
    geometry_contract: Path | None = None,
    stage: str = "model_hpo_final",
    frozen_ae_gmm_sidecar: Path | None = None,
    frozen_ae_gmm_manifest: Path | None = None,
    timestamp_column: str = DEFAULT_TIMESTAMP_COLUMN,
    label_end_column: str = DEFAULT_LABEL_END_COLUMN,
    future_path_column: str | None = None,
    mandatory_features: Sequence[str] = (),
    config_mapping: Mapping[str, Any] | None = None,
    hpo_trials: int = 75,
    oof_folds: int = 5,
    selection_rows: int = 45_000,
    hpo_rows: int = 24_000,
    hpo_folds: int = 3,
    hpo_iterations: int = 1_500,
    hpo_od_wait: int = 100,
    hpo_no_improvement_trials: int = DEFAULT_HPO_NO_IMPROVEMENT_TRIALS,
    selection_iterations: int = PROXY_SELECTION_ITERATIONS,
    selection_od_wait: int = PROXY_SELECTION_OD_WAIT,
    catboost_threads: int = 4,
    catboost_os_reserve_gib: float = 4.0,
    unsafe_allow_catboost_threads: bool = False,
    embargo_hours: float = 24.0,
    min_class_rows: int = 400,
    random_state: int = 20260722,
    max_rows: int = 0,
    smoke: bool = False,
    stop_after_hpo: bool = False,
    feature_selection_hpo_checkpoint: Path | None = None,
    feature_selection_hpo_registry_root: Path | None = None,
    force_feature_selection_hpo: bool = False,
    hpo_study_path: Path | None = None,
    hpo_progress_path: Path | None = None,
    hpo_proxy: bool = False,
    resource_min_free_ram_gib: float = 2.0,
    resource_max_process_rss_gib: float = 12.0,
    resource_min_free_disk_gib: float = 10.0,
    resource_check_interval_seconds: float = 60.0,
    resource_telemetry_path: Path | None = None,
    resource_guard: TrainingResourceGuard | None = None,
) -> dict[str, Any]:
    """Run deterministic-label feature selection, HPO, OOF, and final fit."""
    if stage not in {"selection_only", "model_hpo_final"}:
        raise ValueError("stage must be 'selection_only' or 'model_hpo_final'")
    if hpo_trials < 1:
        raise ValueError("hpo_trials must be at least one; HPO is a required stage")
    if selection_rows < 100:
        raise ValueError("selection_rows must be at least 100")
    if hpo_rows < 100:
        raise ValueError("hpo_rows must be at least 100")
    if hpo_folds < 2:
        raise ValueError("hpo_folds must be at least two")
    if hpo_no_improvement_trials < 1:
        raise ValueError("HPO no-improvement patience must be positive")
    if selection_iterations < 1 or selection_od_wait < 1:
        raise ValueError("selection proxy iterations and od wait must be positive")
    if hpo_proxy:
        hpo_trials = PROXY_HPO_TRIALS
        hpo_rows = PROXY_HPO_ROWS
        hpo_folds = PROXY_HPO_FOLDS
        hpo_iterations = PROXY_HPO_ITERATIONS
        hpo_od_wait = PROXY_HPO_OD_WAIT
    resource_guard = resource_guard or _build_resource_guard(
        output_dir=output_dir,
        min_free_ram_gib=resource_min_free_ram_gib,
        max_process_rss_gib=resource_max_process_rss_gib,
        min_free_disk_gib=resource_min_free_disk_gib,
        check_interval_seconds=resource_check_interval_seconds,
        telemetry_path=resource_telemetry_path,
    )
    selection_checkpoint_path = output_dir / "feature_selection_checkpoint.json"
    mda_progress_path = output_dir / MDA_PROGRESS_FILENAME
    hpo_study_path = Path(hpo_study_path or output_dir / HPO_STUDY_FILENAME)
    hpo_progress_path = Path(hpo_progress_path or output_dir / HPO_PROGRESS_FILENAME)
    resumable_names = {
        selection_checkpoint_path.name,
        mda_progress_path.name,
        hpo_progress_path.name,
        "geometry_prerequisite.json",
        FEATURE_SELECTION_HPO_CONTRACT_FILENAME,
        "hpo_checkpoint.json",
        "class_balance_mini_sweep_report.json",
    }
    guard_telemetry_path = getattr(resource_guard, "telemetry_path", None)
    if (
        guard_telemetry_path is not None
        and Path(guard_telemetry_path).parent.resolve() == output_dir.resolve()
    ):
        resumable_names.add(Path(guard_telemetry_path).name)
    if hpo_study_path.parent == output_dir:
        resumable_names.update(
            {
                hpo_study_path.name,
                f"{hpo_study_path.name}-wal",
                f"{hpo_study_path.name}-shm",
                f"{hpo_study_path.name}-journal",
            }
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        existing = {path.name for path in output_dir.iterdir()}
        if "run_manifest.json" in existing:
            try:
                prior_manifest = _read_json_object(
                    output_dir / "run_manifest.json", artifact_name="run manifest"
                )
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                raise ValueError("existing run manifest is unreadable") from exc
            if (
                prior_manifest.get("status")
                != "stopped_after_side_local_feature_selection"
            ):
                raise FileExistsError(
                    f"refusing to overwrite completed or incompatible output directory: {output_dir}"
                )
            resumable_names.update({"run_manifest.json", "geometry_contract.json"})
        if not existing.issubset(resumable_names):
            raise FileExistsError(
                f"refusing to overwrite non-empty output directory: {output_dir}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    resource_guard.preflight("input_load")

    effective_max_rows = (
        min(max_rows, SMOKE_MAX_ROWS)
        if smoke and max_rows > 0
        else (SMOKE_MAX_ROWS if smoke else max_rows)
    )
    effective_trials = (
        min(int(hpo_trials), SMOKE_HPO_TRIALS) if smoke else int(hpo_trials)
    )
    stages = (
        SMOKE_PERMUTATION_STAGES
        if smoke
        else archetype.PathArchetypeConfig().permutation_stages
    )
    effective_folds = min(int(oof_folds), 2) if smoke else int(oof_folds)
    config = archetype.PathArchetypeConfig(
        timestamp_col=timestamp_column,
        label_end_col=label_end_column,
        random_state=int(random_state),
        embargo=pd.Timedelta(hours=float(embargo_hours)),
        oof_folds=effective_folds,
        catboost_thread_count=max(1, int(catboost_threads)),
        catboost_os_reserve_gib=float(catboost_os_reserve_gib),
        unsafe_allow_catboost_threads=bool(unsafe_allow_catboost_threads),
        selector_sample_rows=(
            min(int(selection_rows), effective_max_rows)
            if smoke and effective_max_rows > 0
            else int(selection_rows)
        ),
        permutation_stages=tuple(stages),
        class_order=MERGED_PATH_ARCHETYPE_CLASSES,
        legacy_allow_class_weights=False,
    )
    raw, source = _load_frame(input_data)
    canonical_store_mode = feature_dir is not None
    canonical_side: str | None = None
    canonical_input_contract: dict[str, Any] | None = None
    if frozen_ae_gmm_manifest is not None and frozen_ae_gmm_sidecar is None:
        raise ValueError("--frozen-ae-gmm-manifest requires --frozen-ae-gmm-sidecar")
    if frozen_ae_gmm_sidecar is not None and not canonical_store_mode:
        raise ValueError("frozen AE/GMM sidecars require canonical --feature-dir mode")
    if isinstance(input_data, Path) and not canonical_store_mode:
        raise ValueError(
            "parquet input requires --feature-dir; only in-memory DataFrame mode "
            "may supply pre-entry features directly"
        )
    if canonical_store_mode and "path_arch_complete_24h" not in raw:
        raise ValueError(
            "canonical path-label training requires explicit path_arch_complete_24h; "
            "incomplete paths must never enter selection, HPO, or OOF"
        )
    # A canonical path-label file may contain both sides.  Normalise and route
    # it *before* every cap, selection sample, HPO sample, and OOF split.
    frame = _normalise_input(
        raw,
        timestamp_column=timestamp_column,
        label_end_column=label_end_column,
        max_rows=0,
    )
    if canonical_store_mode:
        if side is None:
            raise ValueError(
                "canonical --feature-dir mode requires explicit --side long|short"
            )
        canonical_side = _normalise_training_side(side)
        canonical_side_column = next(
            (name for name in IDENTITY_SIDE_COLUMNS if name in frame), None
        )
        if canonical_side_column is None:
            raise ValueError("canonical path labels require an explicit side column")
        normalized_labels_side = _canonical_side_series(
            frame[canonical_side_column], source="path labels"
        )
        frame = frame.loc[normalized_labels_side.eq(canonical_side)].copy()
        if frame.empty:
            raise ValueError(
                f"canonical path labels have no rows for side={canonical_side}"
            )
        if effective_max_rows > 0:
            frame = frame.iloc[: int(effective_max_rows)].reset_index(drop=True)
        required_bindings = {
            "canonical_candidate_path": canonical_candidate_path,
            "canonical_candidate_manifest": canonical_candidate_manifest,
            "canonical_context_path": canonical_context_path,
            "canonical_context_manifest": canonical_context_manifest,
        }
        missing_bindings = [
            name for name, value in required_bindings.items() if value is None
        ]
        if missing_bindings:
            raise ValueError(
                "canonical side-local CatBoost requires exact candidate/context parquet "
                "inputs and manifests: " + ", ".join(missing_bindings)
            )
        canonical_input_contract = _validate_side_local_canonical_inputs(
            frame,
            side=canonical_side,
            candidate_path=Path(canonical_candidate_path),
            candidate_manifest=Path(canonical_candidate_manifest),
            context_path=Path(canonical_context_path),
            context_manifest=Path(canonical_context_manifest),
            ae_gmm_state=ae_gmm_state,
            ae_gmm_state_manifest=ae_gmm_state_manifest,
        )
        if stage == "model_hpo_final" and geometry_contract is None:
            raise ValueError(
                "canonical model-HPO/final stage requires a side-local geometry contract "
                "between feature selection and HPO"
            )
    elif side is not None:
        # Tests and legacy in-memory callers can opt into side filtering, but a
        # canonical parquet run never reaches this compatibility branch.
        canonical_side = _normalise_training_side(side)
        side_column = next(
            (name for name in IDENTITY_SIDE_COLUMNS if name in frame), None
        )
        if side_column is None:
            raise ValueError("side was requested but input has no side column")
        frame = frame.loc[
            _canonical_side_series(frame[side_column], source="input").eq(
                canonical_side
            )
        ].copy()
        if frame.empty:
            raise ValueError(f"input has no rows for side={canonical_side}")
        if effective_max_rows > 0:
            frame = frame.iloc[: int(effective_max_rows)].reset_index(drop=True)
    elif effective_max_rows > 0:
        frame = frame.iloc[: int(effective_max_rows)].reset_index(drop=True)
    discovery_cutoff = _utc_timestamp(discovery_end)
    discovery_mask = frame[label_end_column] < discovery_cutoff

    summaries, summary_source = _path_summaries(
        frame, future_path_column=future_path_column
    )
    if "path_archetype" not in frame or "path_shape_archetype" not in frame:
        raise ValueError(
            "CatBoost training requires the frozen deterministic path shape and "
            "shape-strength targets; discovery clusters are diagnostic only"
        )
    if "path_archetype_rule_version" not in frame:
        raise ValueError(
            "deterministic path target requires path_archetype_rule_version"
        )
    raw_labels = frame["path_archetype"].astype("string").str.strip()
    shape_labels = frame["path_shape_archetype"].astype("string").str.strip()
    labels = merge_fast_realization_winner(shape_labels).astype("string")
    if raw_labels.isna().any() or raw_labels.eq("").any():
        raise ValueError("deterministic path_archetype diagnostic must be complete")
    if labels.isna().any() or labels.eq("").any():
        raise ValueError("deterministic path_shape_archetype target must be complete")
    versions = frame["path_archetype_rule_version"].astype("string").str.strip()
    if not versions.eq(PATH_ARCHETYPE_RULE_VERSION).all():
        found = sorted(versions.dropna().unique().tolist())
        raise ValueError(
            "stale path_archetype_rule_version: "
            f"expected {PATH_ARCHETYPE_RULE_VERSION!r}, found {found[:5]}"
        )
    invalid_raw_labels = sorted(
        set(raw_labels.astype(str)).difference(PATH_ARCHETYPE_TYPES)
    )
    if invalid_raw_labels:
        raise ValueError(
            f"unknown deterministic path archetypes: {invalid_raw_labels[:8]}"
        )
    expected_classes = MERGED_PATH_ARCHETYPE_CLASSES
    invalid_labels = sorted(set(labels.astype(str)).difference(expected_classes))
    if invalid_labels:
        raise ValueError(f"unknown deterministic path shapes: {invalid_labels[:8]}")
    if labels.nunique() > len(expected_classes):
        raise ValueError("CatBoost path target exceeds its declared taxonomy contract")
    development_mask, development_oos_routing = _development_oos_routing(
        frame,
        timestamp_column=timestamp_column,
        label_end_column=label_end_column,
        development_end=development_end,
    )
    if canonical_store_mode and not smoke:
        required_development_end = pd.Timestamp("2026-05-01T00:00:00Z")
        actual_development_end = development_oos_routing.get(
            "development_end_exclusive"
        )
        if (
            not development_oos_routing["enabled"]
            or pd.Timestamp(actual_development_end) != required_development_end
        ):
            raise ValueError(
                "canonical production CatBoost requires --development-end "
                "2026-05-01T00:00:00Z"
            )
        if config.embargo != pd.Timedelta(hours=24):
            raise ValueError(
                "canonical production CatBoost requires an exact 24h embargo"
            )
    development_frame = frame.loc[development_mask].copy()
    development_labels = labels.loc[development_mask].copy()
    development_summaries = summaries.loc[development_mask].copy()
    class_support_gate: dict[str, Any] | None = None
    if canonical_store_mode and not smoke:
        assert canonical_side is not None
        class_support_gate = _validate_merged_class_support_gate(
            development_labels,
            development_frame[timestamp_column],
            side=canonical_side,
        )
    recomputed = summaries.apply(deterministic_path_archetype, axis=1).astype("string")
    effective_recomputed = merge_fast_realization_winner(recomputed).astype("string")
    mismatch = effective_recomputed.isna() | effective_recomputed.ne(labels)
    if mismatch.any():
        raise ValueError(
            "effective deterministic path archetype does not match realized path summaries: "
            f"{int(mismatch.sum())} mismatches"
        )
    effective_min_class_rows = 1
    class_consolidation = (
        pd.DataFrame(
            {
                "raw_path_archetype": raw_labels.loc[discovery_mask].astype(str),
                "effective_path_archetype": labels.loc[discovery_mask].astype(str),
            }
        )
        .value_counts(sort=False)
        .rename("rows")
        .reset_index()
    )
    taxonomy_contract = FUTURE_TRAINING_TAXONOMY_CONTRACT
    target_source = "merged_deterministic_path_shape_archetype_7class"
    static_store_manifest: dict[str, Any] | None = None
    frozen_sidecar_contract: dict[str, Any] | None = None
    static_feature_columns: set[str] = set()
    side_column = "side"
    if canonical_store_mode:
        if future_path_column is not None:
            raise ValueError(
                "canonical feature-dir mode requires materialized path_arch_* summaries; "
                "future path arrays are not accepted"
            )
        identity_columns = _canonical_identity_columns(
            frame, timestamp_column=timestamp_column
        )
        side_column = identity_columns[-1]
        allowed_input_columns = {
            timestamp_column,
            label_end_column,
            IDENTITY_SYMBOL_COLUMN,
            *IDENTITY_SIDE_COLUMNS,
            *PATH_TARGET_PROVENANCE_COLUMNS,
            *[
                column
                for column in frame.columns
                if str(column).startswith(archetype.PATH_SUMMARY_PREFIX)
            ],
            "path_arch_complete_24h",
            "path_shape_archetype",
            "path_realization_strength",
            "path_archetype",
            "path_archetype_rule_version",
            "discovery_cluster_id",
        }
        unexpected_input_columns = sorted(
            set(frame.columns).difference(allowed_input_columns)
        )
        if unexpected_input_columns:
            raise ValueError(
                "canonical feature-dir input may contain only UTC identity, "
                "label-end, and path_arch_* summary columns; unexpected: "
                + ", ".join(unexpected_input_columns[:8])
            )
        feature_store_ts, data_root = _feature_store_location(Path(feature_dir))
        schema_columns, schema_by_file = _feature_store_schemas(Path(feature_dir))
        static_configured_declared = (
            archetype.configured_base_meta_preselection_universe(
                _config_feature_names(config_mapping), config_mapping=config_mapping
            )
        )
        if frozen_ae_gmm_sidecar is not None:
            frozen_sidecar_contract = _validate_frozen_ae_gmm_sidecar(
                frame,
                sidecar_path=Path(frozen_ae_gmm_sidecar),
                manifest_path=(
                    Path(frozen_ae_gmm_manifest)
                    if frozen_ae_gmm_manifest is not None
                    else None
                ),
                timestamp_column=timestamp_column,
                side_column=side_column,
            )
        configured_declared = archetype.configured_base_meta_preselection_universe(
            [
                *_config_feature_names(config_mapping),
                *(
                    tuple(frozen_sidecar_contract["output_features"])
                    if frozen_sidecar_contract is not None
                    else ()
                ),
            ],
            config_mapping=config_mapping,
            frozen_representation_features=(
                AE_GMM_FEATURE_COLUMNS if frozen_sidecar_contract is not None else ()
            ),
        )
        if frozen_sidecar_contract is not None:
            configured_declared = tuple(
                dict.fromkeys([*configured_declared, REPRESENTATION_AVAILABLE_FEATURE])
            )
        # A canonical path-label parquet is deliberately narrow. Catch an
        # accidental wide feature export before the static store is consulted.
        embedded_features = sorted(
            set(frame.columns).intersection(static_configured_declared)
        )
        if embedded_features:
            raise ValueError(
                "canonical feature-dir input must not embed pre-entry feature columns: "
                + ", ".join(embedded_features[:8])
            )
        static_feature_columns = set(static_configured_declared).intersection(
            schema_columns
        )
        static_universe = tuple(
            feature
            for feature in static_configured_declared
            if feature in schema_columns
        )
        frozen_universe = tuple(
            map(str, frozen_sidecar_contract["output_features"])
            if frozen_sidecar_contract is not None
            else ()
        )
        universe = tuple(dict.fromkeys([*static_universe, *frozen_universe]))
        static_store_manifest = {
            "mode": "canonical_static_feature_store",
            "feature_dir": str(Path(feature_dir)),
            "data_root": str(data_root),
            "feature_store_timestamp_utc": feature_store_ts,
            "schema_symbol_files": len(schema_by_file),
            "schema_union_feature_count": len(schema_columns),
            "configured_declared_features": list(static_configured_declared),
            "configured_and_frozen_declared_features": list(configured_declared),
            "configured_features_absent_from_schema": sorted(
                set(static_configured_declared).difference(schema_columns)
            ),
            "configured_features_resolved_from_schema": list(static_universe),
            "frozen_ae_gmm_sidecar": frozen_sidecar_contract,
            "frozen_ae_gmm_features": list(frozen_universe),
        }
    else:
        universe = archetype.configured_base_meta_preselection_universe(
            frame.columns, config_mapping=config_mapping
        )
        static_feature_columns = set(universe)
    if not universe:
        raise ValueError(
            "no configured base/meta pre-entry features are present in input"
        )
    _assert_preentry_only(universe)
    mandatory = tuple(
        dict.fromkeys(
            [
                *map(str, mandatory_features),
                *(
                    (REPRESENTATION_AVAILABLE_FEATURE,)
                    if frozen_sidecar_contract is not None
                    else ()
                ),
            ]
        )
    )
    _assert_preentry_only(mandatory)
    missing_mandatory = set(mandatory).difference(universe)
    if missing_mandatory:
        raise ValueError(
            "mandatory features must be in configured base/meta universe: "
            + ", ".join(sorted(missing_mandatory))
        )
    model_universe = tuple(universe)
    if smoke and len(model_universe) > SMOKE_MAX_FEATURES:
        frozen_required = tuple(
            map(str, frozen_sidecar_contract["output_features"])
            if frozen_sidecar_contract is not None
            else ()
        )
        model_universe = tuple(
            dict.fromkeys(
                [
                    *mandatory,
                    *frozen_required,
                    *model_universe[
                        : max(
                            0,
                            SMOKE_MAX_FEATURES - len(mandatory) - len(frozen_required),
                        )
                    ],
                ]
            )
        )
    effective_hpo_rows = min(int(hpo_rows), len(development_frame))
    effective_hpo_folds = min(int(hpo_folds), effective_folds)
    hpo_config = archetype.PathArchetypeConfig(
        **{**config.__dict__, "oof_folds": effective_hpo_folds}
    )
    structural_hpo_contract = archetype.catboost_structural_hpo_contract(hpo_config)
    hpo_positions, hpo_sampling_contract = _stratified_hpo_sample(
        development_frame,
        development_labels,
        sample_rows=effective_hpo_rows,
        validation_folds=effective_hpo_folds,
        timestamp_column=timestamp_column,
        label_end_column=label_end_column,
        side_column=side_column,
    )
    hpo_frame = development_frame.iloc[hpo_positions]
    hpo_labels = development_labels.iloc[hpo_positions]
    hpo_sampling_contract["purged_fold_support"] = _validate_hpo_sample_class_support(
        hpo_frame,
        hpo_labels,
        timestamp_column=timestamp_column,
        label_end_column=label_end_column,
        side_column=side_column,
        config=hpo_config,
    )
    selection_hpo_contract = _feature_selection_hpo_fingerprint(
        frame=development_frame,
        summaries=development_summaries,
        effective_labels=development_labels,
        timestamp_column=timestamp_column,
        side_column=side_column,
        frozen_sidecar_contract=frozen_sidecar_contract,
        universe=universe,
        model_universe=model_universe,
        mandatory_features=mandatory,
        config=config,
        effective_trials=effective_trials,
        hpo_rows=effective_hpo_rows,
        hpo_folds=effective_hpo_folds,
        hpo_iterations=int(hpo_iterations),
        hpo_od_wait=int(hpo_od_wait),
        hpo_no_improvement_trials=int(hpo_no_improvement_trials),
        selection_iterations=int(selection_iterations),
        selection_od_wait=int(selection_od_wait),
        smoke=smoke,
        hpo_sample_contract=hpo_sampling_contract,
        structural_hpo_contract=structural_hpo_contract,
        taxonomy_contract=taxonomy_contract,
        side=canonical_side,
        canonical_input_contract=canonical_input_contract,
        geometry_contract_path=geometry_contract,
        development_oos_routing=development_oos_routing,
    )
    selection_fingerprint = str(selection_hpo_contract["selection_fingerprint"])
    resumed_selection_checkpoint = (
        _read_resumable_feature_selection_checkpoint(
            selection_checkpoint_path, selection_fingerprint
        )
        if selection_checkpoint_path.is_file() and not force_feature_selection_hpo
        else None
    )
    resumed_mda_progress = (
        _read_resumable_mda_progress(mda_progress_path, selection_fingerprint)
        if mda_progress_path.is_file()
        and not selection_checkpoint_path.is_file()
        and not force_feature_selection_hpo
        else None
    )
    full_contract_error: ValueError | None = None
    try:
        reused_checkpoint, reuse_provenance = (
            _find_reusable_feature_selection_hpo_contract(
                selection_hpo_contract,
                output_dir=output_dir,
                checkpoint_path=feature_selection_hpo_checkpoint,
                registry_root=feature_selection_hpo_registry_root,
                force=force_feature_selection_hpo,
            )
        )
    except ValueError as exc:
        full_contract_error = exc
        reused_checkpoint = None
        reuse_provenance = {"mode": "full_hpo_contract_mismatch", "reused": False}
    reused_selection_checkpoint: dict[str, Any] | None = None
    if reused_checkpoint is None and resumed_selection_checkpoint is None:
        reused_selection_checkpoint, selection_reuse_provenance = (
            _find_reusable_feature_selection_checkpoint(
                selection_fingerprint,
                output_dir=output_dir,
                checkpoint_path=feature_selection_hpo_checkpoint,
                registry_root=feature_selection_hpo_registry_root,
                force=force_feature_selection_hpo,
            )
        )
        if reused_selection_checkpoint is not None:
            reuse_provenance = selection_reuse_provenance
    if (
        full_contract_error is not None
        and feature_selection_hpo_checkpoint is not None
        and reused_selection_checkpoint is None
    ):
        raise full_contract_error
    _validate_oof_class_support(
        development_labels,
        development_frame[timestamp_column],
        development_frame[label_end_column],
        config=config,
    )
    selector: archetype.FastSelectorResult | None = None
    bound_geometry_contract: dict[str, Any] | None = None
    pre_permutation_oof: archetype.OOFPathArchetypeResult | None = None
    selection_features: pd.DataFrame | None = None
    selection_labels: pd.Series | None = None
    selection_availability: dict[str, Any] | None = None
    permutation_stage_metrics: list[dict[str, Any]] = []
    if reused_checkpoint is not None:
        selected = _validate_reused_selected_features(
            reused_checkpoint["selected_features"], model_universe
        )
        selection_frame = (
            development_frame.iloc[
                _beginning_middle_end_positions(
                    len(development_frame), config.selector_sample_rows
                )
            ]
            if canonical_store_mode
            else development_frame
        )
        selector_manifest = dict(reused_checkpoint.get("selection", {}))
        permutation_records = list(reused_checkpoint.get("permutation", []))
        permutation_stage_metrics = list(
            reused_checkpoint.get(
                "permutation_stage_metrics",
                _permutation_stage_metrics(permutation_records),
            )
        )
        hpo_report = dict(reused_checkpoint.get("hpo", {}))
        params = dict(reused_checkpoint["effective_model_params"])
    elif (
        resumed_selection_checkpoint is not None
        or reused_selection_checkpoint is not None
    ):
        selection_checkpoint = (
            resumed_selection_checkpoint
            if resumed_selection_checkpoint is not None
            else reused_selection_checkpoint
        )
        assert selection_checkpoint is not None
        selected = _validate_reused_selected_features(
            selection_checkpoint["selected_features"], model_universe
        )
        selection_frame = (
            development_frame.iloc[
                _beginning_middle_end_positions(
                    len(development_frame), config.selector_sample_rows
                )
            ].copy()
            if canonical_store_mode
            else development_frame
        )
        selector_manifest = dict(selection_checkpoint["selection"])
        permutation_records = list(selection_checkpoint["permutation"])
        permutation_stage_metrics = list(
            selection_checkpoint.get(
                "permutation_stage_metrics",
                _permutation_stage_metrics(permutation_records),
            )
        )
        hpo_report = None
        params = {}
        if resumed_selection_checkpoint is not None:
            reuse_provenance = {
                "mode": "interrupted_hpo_resume",
                "reused": False,
                "path": str(selection_checkpoint_path),
            }
    elif resumed_mda_progress is not None:
        selected = _validate_reused_selected_features(
            resumed_mda_progress["initial_selected_features"], model_universe
        )
        selection_frame = (
            development_frame.iloc[
                _beginning_middle_end_positions(
                    len(development_frame), config.selector_sample_rows
                )
            ].copy()
            if canonical_store_mode
            else development_frame
        )
        selector_manifest = dict(resumed_mda_progress["selection"])
        permutation_records = [
            record
            for stage in resumed_mda_progress["completed_stages"]
            for record in stage.get("records", [])
        ]
        permutation_stage_metrics = _permutation_stage_metrics(permutation_records)
        hpo_report = None
        params = {}
        reuse_provenance = {
            "mode": "interrupted_mda_resume",
            "reused": False,
            "path": str(mda_progress_path),
        }
    else:
        if canonical_store_mode:
            sample_positions = _beginning_middle_end_positions(
                len(development_frame), config.selector_sample_rows
            )
            selection_frame = development_frame.iloc[sample_positions].copy()
            selection_labels = development_labels.iloc[sample_positions]
            split_at = np.flatnonzero(np.diff(sample_positions) > 1) + 1
            selection_parts: list[pd.DataFrame] = []
            selection_reads: list[dict[str, Any]] = []
            for block_positions in np.split(sample_positions, split_at):
                block_frame = development_frame.iloc[block_positions]
                block_features, block_read = _load_model_feature_matrix(
                    block_frame,
                    model_universe,
                    static_feature_columns=static_feature_columns,
                    frozen_sidecar_contract=frozen_sidecar_contract,
                    side_column=side_column,
                    feature_store_ts=feature_store_ts,
                    data_root=data_root,
                    timestamp_column=timestamp_column,
                )
                selection_parts.append(block_features)
                selection_reads.append(
                    {
                        "rows": int(len(block_frame)),
                        "start_timestamp_utc": block_frame[timestamp_column].min(),
                        "end_timestamp_utc": block_frame[timestamp_column].max(),
                        **block_read,
                    }
                )
            selection_features = pd.concat(selection_parts, axis=0).reindex(
                selection_frame.index
            )
            selection_availability = {
                "sample_rows": int(len(selection_frame)),
                "blocks": selection_reads,
            }
        else:
            selection_frame = development_frame
            selection_labels = development_labels
            selection_features = development_frame.loc[:, model_universe].copy()
        selector = archetype.fast_select_preentry_features(
            selection_features,
            selection_labels,
            mandatory_features=mandatory,
            config=config,
        )
        selected = list(selector.selected_features)

    if canonical_store_mode:
        if selection_labels is None:
            selection_labels = development_labels.loc[selection_frame.index]
        if selection_features is None:
            selection_features, selection_read = _load_model_feature_matrix(
                selection_frame,
                selected,
                static_feature_columns=static_feature_columns,
                frozen_sidecar_contract=frozen_sidecar_contract,
                side_column=side_column,
                feature_store_ts=feature_store_ts,
                data_root=data_root,
                timestamp_column=timestamp_column,
            )
            selection_availability = {
                "sample_rows": int(len(selection_frame)),
                "blocks": [selection_read],
                "resume_read": True,
            }
        else:
            selection_features = selection_features.loc[:, selected]
        assert static_store_manifest is not None
        static_store_manifest.update(
            {
                "selection_sample_rows": int(len(selection_frame)),
                "selection_sample_contract": "deterministic beginning/middle/end timestamp rows",
                "selection_read": selection_availability,
                "full_selected_feature_read": "deferred_until_final_oof_and_refit",
            }
        )
    else:
        selection_features = (
            development_frame.loc[:, selected].copy()
            if (
                reused_checkpoint is not None
                or resumed_selection_checkpoint is not None
                or reused_selection_checkpoint is not None
                or resumed_mda_progress is not None
            )
            else selection_features
        )
        selection_labels = development_labels.loc[selection_frame.index]

    if selection_features is None or selection_labels is None:
        raise RuntimeError("selection sample was not materialized")

    if reused_checkpoint is None:
        selection_params: dict[str, Any] = {
            "iterations": int(selection_iterations),
            "od_wait": int(selection_od_wait),
        }
        if smoke:
            selection_params["iterations"] = min(
                int(selection_params["iterations"]), SMOKE_MAX_ITERATIONS
            )
            selection_params["od_wait"] = min(
                int(selection_params["od_wait"]),
                32,
                int(selection_params["iterations"]),
            )
        if resumed_selection_checkpoint is None and reused_selection_checkpoint is None:
            if selector is not None:
                selector_manifest = _selector_manifest(selector)
            mda_columns = archetype.staged_permutation_feature_order(
                selected, mandatory
            )
            mda_features = selection_features.loc[:, mda_columns]
            mda_cache = archetype.build_staged_permutation_matrix_cache(
                mda_features,
                selection_frame[timestamp_column],
                label_end=selection_frame[label_end_column],
                config=config,
            )
            resource_guard.checkpoint("feature_selection_oof")
            pre_permutation_oof = archetype.fit_purged_chronological_oof_catboost(
                mda_features,
                selection_labels,
                selection_frame[timestamp_column],
                label_end=selection_frame[label_end_column],
                config=config,
                params=selection_params,
                staged_matrix_cache=mda_cache,
                force_classes_count=False,
                fold_callback=lambda fold_index, _probabilities, _fold_ids: (
                    resource_guard.checkpoint(
                        f"feature_selection_oof_fold:{fold_index}"
                    )
                ),
            )
            mda_completed_stages = (
                list(resumed_mda_progress["completed_stages"])
                if resumed_mda_progress is not None
                else []
            )

            def write_mda_progress(stage: Mapping[str, Any]) -> None:
                resource_guard.checkpoint(
                    f"permutation_mda_stage:{int(stage['stage_index'])}"
                )
                completed = [*mda_completed_stages, dict(stage)]
                mda_completed_stages[:] = completed
                _write_json(
                    mda_progress_path,
                    {
                        "schema": RUNNER_SCHEMA,
                        "status": "mda_running",
                        "fingerprint": selection_fingerprint,
                        "selection_fingerprint": selection_fingerprint,
                        "initial_selected_features": list(mda_columns),
                        "selection": selector_manifest,
                        "selection_proxy_params": selection_params,
                        "completed_stages": completed,
                        "permutation_acceleration_contract": (
                            archetype.staged_permutation_acceleration_contract(config)
                        ),
                    },
                )

            if resumed_mda_progress is None:
                _write_json(
                    mda_progress_path,
                    {
                        "schema": RUNNER_SCHEMA,
                        "status": "mda_running",
                        "fingerprint": selection_fingerprint,
                        "selection_fingerprint": selection_fingerprint,
                        "initial_selected_features": list(mda_columns),
                        "selection": selector_manifest,
                        "selection_proxy_params": selection_params,
                        "completed_stages": [],
                        "permutation_acceleration_contract": (
                            archetype.staged_permutation_acceleration_contract(config)
                        ),
                    },
                )
            selected, permutation = archetype.staged_permutation_selection(
                mda_features,
                selection_labels,
                pre_permutation_oof,
                mandatory_features=mandatory,
                stages=config.permutation_stages,
                random_state=config.random_state,
                config=config,
                params=selection_params,
                completed_stages=mda_completed_stages,
                stage_callback=write_mda_progress,
            )
            _assert_preentry_only(selected)
            if selector is not None:
                selector_manifest = _selector_manifest(selector)
            permutation_records = permutation.to_dict(orient="records")
            permutation_stage_metrics = _permutation_stage_metrics(permutation)
            _write_json(
                mda_progress_path,
                {
                    "schema": RUNNER_SCHEMA,
                    "status": "mda_complete",
                    "fingerprint": selection_fingerprint,
                    "selection_fingerprint": selection_fingerprint,
                    "initial_selected_features": list(mda_columns),
                    "selected_features": list(selected),
                    "selection": selector_manifest,
                    "selection_proxy_params": selection_params,
                    "completed_stages": mda_completed_stages,
                    "permutation_acceleration_contract": (
                        archetype.staged_permutation_acceleration_contract(config)
                    ),
                },
            )
            _write_json(
                selection_checkpoint_path,
                {
                    "schema": RUNNER_SCHEMA,
                    "status": "feature_selection_complete",
                    "fingerprint": selection_fingerprint,
                    "selection_fingerprint": selection_fingerprint,
                    "selected_features": list(selected),
                    "selected_feature_count": int(len(selected)),
                    "selection": selector_manifest,
                    "permutation": permutation_records,
                    "permutation_stage_metrics": permutation_stage_metrics,
                    "selection_proxy_params": selection_params,
                    "permutation_acceleration_contract": (
                        archetype.staged_permutation_acceleration_contract(config)
                    ),
                    "catboost_resource_contract": archetype.catboost_resource_contract(
                        config
                    ),
                },
            )
        side_candidate_identity = _side_candidate_identity_sha256(
            frame,
            timestamp_column=timestamp_column,
            symbol_column=IDENTITY_SYMBOL_COLUMN,
            side_column=side_column,
        )
        geometry_provenance = _side_local_geometry_provenance(canonical_input_contract)
        preserve_geometry_handoff = (
            canonical_store_mode
            and stage == "model_hpo_final"
            and geometry_contract is not None
        )
        if preserve_geometry_handoff:
            # The external geometry sweep cryptographically binds these files.
            # Verify the frozen handoff before any write and never regenerate it
            # during model HPO merely because reuse provenance changed.
            bound_geometry_contract = _read_side_geometry_contract(
                geometry_contract,
                side=canonical_side,
                candidate_identity=side_candidate_identity,
                selected_features=selected,
                selection_fingerprint=selection_fingerprint,
                geometry_prerequisite_path=output_dir / "geometry_prerequisite.json",
                canonical_input_contract=canonical_input_contract,
            )
        else:
            # Feature selection is a durable stage in its own right. Geometry is
            # intentionally external and must be selected from this exact
            # side-only feature contract before model HPO is allowed to begin.
            _write_json(
                selection_checkpoint_path,
                {
                    "schema": RUNNER_SCHEMA,
                    "status": "feature_selection_complete",
                    "fingerprint": selection_fingerprint,
                    "selection_fingerprint": selection_fingerprint,
                    "side": canonical_side,
                    "model_side_scope": "per_side" if canonical_side else None,
                    "canonical_input_contract": canonical_input_contract,
                    "class_support_gate": class_support_gate,
                    **_side_local_geometry_provenance(canonical_input_contract),
                    "selected_features": list(selected),
                    "selected_feature_count": int(len(selected)),
                    "selection": selector_manifest,
                    "permutation": permutation_records,
                    "permutation_stage_metrics": permutation_stage_metrics,
                    "selection_proxy_params": {
                        "iterations": int(selection_iterations),
                        "od_wait": int(selection_od_wait),
                    },
                    "permutation_acceleration_contract": (
                        archetype.staged_permutation_acceleration_contract(config)
                    ),
                    "reuse_provenance": reuse_provenance,
                    "catboost_resource_contract": archetype.catboost_resource_contract(
                        config
                    ),
                },
            )
            geometry_prerequisite = {
                "schema": "catboost_path_archetype_geometry_prerequisite_v1",
                "status": "selection_complete_pending_geometry",
                "side": canonical_side,
                "model_side_scope": "per_side",
                "candidate_identity_sha256": side_candidate_identity,
                "selection_fingerprint": selection_fingerprint,
                "selected_features": list(selected),
                "selected_features_sha256": _sha256_json(list(selected)),
                "feature_selection_checkpoint": str(
                    selection_checkpoint_path.resolve()
                ),
                "feature_selection_checkpoint_sha256": _sha256_file(
                    selection_checkpoint_path
                ),
                "canonical_input_contract": canonical_input_contract,
                "class_support_gate": class_support_gate,
                "canonical_input_contract_sha256": _sha256_json(
                    canonical_input_contract
                ),
                "required_next_stage": "side_local_geometry_sweep",
                **geometry_provenance,
            }
            _write_json(
                output_dir / "geometry_prerequisite.json", geometry_prerequisite
            )
        if stage == "selection_only":
            checkpoint_manifest = {
                "schema": RUNNER_SCHEMA,
                "status": "stopped_after_side_local_feature_selection",
                "stage": "selection_only",
                "side": canonical_side,
                "model_side_scope": "per_side" if canonical_side else None,
                "rows": int(len(frame)),
                "candidate_identity_sha256": side_candidate_identity,
                "canonical_input_contract": canonical_input_contract,
                "class_support_gate": class_support_gate,
                "selected_features": list(selected),
                "selected_features_sha256": _sha256_json(list(selected)),
                "feature_selection_checkpoint": "feature_selection_checkpoint.json",
                "geometry_prerequisite": "geometry_prerequisite.json",
                "feature_selection_fingerprint": selection_fingerprint,
                "geometry_prerequisite_sha256": _sha256_file(
                    output_dir / "geometry_prerequisite.json"
                ),
                **geometry_provenance,
                "full_oof_and_final_refit_complete": False,
            }
            _write_json(output_dir / "run_manifest.json", checkpoint_manifest)
            return checkpoint_manifest
        if canonical_store_mode:
            if bound_geometry_contract is None:
                bound_geometry_contract = _read_side_geometry_contract(
                    geometry_contract,
                    side=canonical_side,
                    candidate_identity=side_candidate_identity,
                    selected_features=selected,
                    selection_fingerprint=selection_fingerprint,
                    geometry_prerequisite_path=output_dir
                    / "geometry_prerequisite.json",
                    canonical_input_contract=canonical_input_contract,
                )
            hpo_features, _ = _load_model_feature_matrix(
                hpo_frame,
                selected,
                static_feature_columns=static_feature_columns,
                frozen_sidecar_contract=frozen_sidecar_contract,
                side_column=side_column,
                feature_store_ts=feature_store_ts,
                data_root=data_root,
                timestamp_column=timestamp_column,
            )
        else:
            hpo_features = development_frame.iloc[hpo_positions].loc[:, selected]
        resource_guard.checkpoint("hpo")
        hpo = archetype.optimize_purged_catboost_hpo(
            hpo_features.loc[:, selected],
            hpo_labels,
            hpo_frame[timestamp_column],
            label_end=hpo_frame[label_end_column],
            config=hpo_config,
            n_trials=effective_trials,
            study_name=(f"path_archetype_{selection_hpo_contract['fingerprint'][:16]}"),
            storage=f"sqlite:///{hpo_study_path.resolve()}",
            search_iterations=int(hpo_iterations),
            search_od_wait=int(hpo_od_wait),
            no_improvement_trials=int(hpo_no_improvement_trials),
            progress_path=hpo_progress_path,
            structural_only_hpo=True,
        )
        resource_guard.checkpoint("hpo_complete")
        params = dict(hpo.best_params)
        # HPO uses a bounded proxy fit on the frozen final feature contract.
        params["iterations"] = 3_000
        params["od_wait"] = 150
        if smoke:
            params["iterations"] = min(
                int(params.get("iterations", SMOKE_MAX_ITERATIONS)),
                SMOKE_MAX_ITERATIONS,
            )
            params["od_wait"] = min(int(params.get("od_wait", 32)), 32)
        hpo_report = {
            **hpo.report(),
            "sampling_contract": hpo_sampling_contract,
            "geometry_contract": bound_geometry_contract,
        }

    if canonical_store_mode and bound_geometry_contract is None:
        bound_geometry_contract = _read_side_geometry_contract(
            geometry_contract,
            side=canonical_side,
            candidate_identity=_side_candidate_identity_sha256(
                frame,
                timestamp_column=timestamp_column,
                symbol_column=IDENTITY_SYMBOL_COLUMN,
                side_column=side_column,
            ),
            selected_features=selected,
            selection_fingerprint=selection_fingerprint,
            geometry_prerequisite_path=output_dir / "geometry_prerequisite.json",
            canonical_input_contract=canonical_input_contract,
        )
    class_balance_provenance: dict[str, Any] = {
        "structural_hpo_contract": dict(structural_hpo_contract),
        "structural_hpo_contract_sha256": _sha256_json(structural_hpo_contract),
        "status": "structural_hpo_complete_pending_development_balance_sweep",
        "promotion_eligible": False,
        "final_refit_used_for_selection": False,
    }
    completed_selection_hpo_contract = {
        **selection_hpo_contract,
        "status": "feature_selection_hpo_complete",
        "side": canonical_side,
        "model_side_scope": "per_side" if canonical_side else None,
        "development_oos_routing": development_oos_routing,
        "candidate_identity_sha256": _side_candidate_identity_sha256(
            frame,
            timestamp_column=timestamp_column,
            symbol_column=IDENTITY_SYMBOL_COLUMN,
            side_column=side_column,
        ),
        **_side_local_geometry_provenance(canonical_input_contract),
        "selected_features": list(selected),
        "effective_model_params": params,
        "selection": selector_manifest,
        "selection_proxy_params": {
            "iterations": int(selection_iterations),
            "od_wait": int(selection_od_wait),
        },
        "permutation": permutation_records,
        "permutation_stage_metrics": permutation_stage_metrics,
        "permutation_acceleration_contract": (
            archetype.staged_permutation_acceleration_contract(config)
        ),
        "hpo": hpo_report,
        "class_balance": class_balance_provenance,
        "hpo_sampling_contract": hpo_sampling_contract,
        "hpo_study_path": hpo_study_path,
        "hpo_progress_path": hpo_progress_path,
        "geometry_contract": bound_geometry_contract,
        "reuse_provenance": reuse_provenance,
        "catboost_resource_contract": archetype.catboost_resource_contract(config),
    }
    if not (
        canonical_store_mode
        and stage == "model_hpo_final"
        and geometry_contract is not None
    ):
        _write_json(
            output_dir / "feature_selection_checkpoint.json",
            {
                "schema": RUNNER_SCHEMA,
                "status": "feature_selection_complete",
                "fingerprint": selection_fingerprint,
                "selection_fingerprint": selection_fingerprint,
                "selected_features": list(selected),
                "selected_feature_count": int(len(selected)),
                "selection": selector_manifest,
                "permutation": permutation_records,
                "permutation_stage_metrics": permutation_stage_metrics,
                "selection_proxy_params": {
                    "iterations": int(selection_iterations),
                    "od_wait": int(selection_od_wait),
                },
                "permutation_acceleration_contract": (
                    archetype.staged_permutation_acceleration_contract(config)
                ),
                "reuse_provenance": reuse_provenance,
                "catboost_resource_contract": archetype.catboost_resource_contract(
                    config
                ),
            },
        )
    _write_json(
        output_dir / "hpo_checkpoint.json",
        {
            "schema": RUNNER_SCHEMA,
            "status": "hpo_complete",
            "fingerprint": selection_hpo_contract["fingerprint"],
            "selected_features": list(selected),
            "effective_model_params": params,
            "hpo": hpo_report,
            "class_balance": class_balance_provenance,
            "hpo_sampling_contract": hpo_sampling_contract,
            "selection_proxy_params": {
                "iterations": int(selection_iterations),
                "od_wait": int(selection_od_wait),
            },
            "hpo_study_path": hpo_study_path,
            "hpo_progress_path": hpo_progress_path,
            "reuse_provenance": reuse_provenance,
            "catboost_resource_contract": archetype.catboost_resource_contract(config),
        },
    )
    _write_json(
        output_dir / FEATURE_SELECTION_HPO_CONTRACT_FILENAME,
        completed_selection_hpo_contract,
    )
    if stop_after_hpo:
        checkpoint_manifest = {
            "schema": RUNNER_SCHEMA,
            "status": "stopped_after_structural_hpo_pending_balance_sweep",
            "future_training_taxonomy": taxonomy_contract,
            "source": source,
            "rows": int(len(frame)),
            "candidate_identity_sha256": _side_candidate_identity_sha256(
                frame,
                timestamp_column=timestamp_column,
                symbol_column=IDENTITY_SYMBOL_COLUMN,
                side_column=side_column,
            ),
            "selected_features": list(selected),
            "effective_model_params": params,
            "catboost_resource_contract": archetype.catboost_resource_contract(config),
            "feature_selection_checkpoint": "feature_selection_checkpoint.json",
            "hpo_checkpoint": "hpo_checkpoint.json",
            "feature_selection_hpo_contract": FEATURE_SELECTION_HPO_CONTRACT_FILENAME,
            "feature_selection_hpo_fingerprint": selection_hpo_contract["fingerprint"],
            "feature_selection_fingerprint": selection_fingerprint,
            "feature_selection_hpo_reuse": reuse_provenance,
            "hpo_sampling_contract": hpo_sampling_contract,
            "class_balance": class_balance_provenance,
            "next_required_stage": (
                "development_only_fixed_parameter_balance_mini_sweep"
            ),
            "permutation_stage_metrics": permutation_stage_metrics,
            "permutation_acceleration_contract": (
                archetype.staged_permutation_acceleration_contract(config)
            ),
            "full_oof_and_final_refit_complete": False,
        }
        _write_json(output_dir / "run_manifest.json", checkpoint_manifest)
        return checkpoint_manifest
    if canonical_store_mode:
        resource_guard.checkpoint("final_feature_load")
        features, full_availability = _load_model_feature_matrix(
            frame,
            selected,
            static_feature_columns=static_feature_columns,
            frozen_sidecar_contract=frozen_sidecar_contract,
            side_column=side_column,
            feature_store_ts=feature_store_ts,
            data_root=data_root,
            timestamp_column=timestamp_column,
        )
        assert static_store_manifest is not None
        static_store_manifest["full_selected_feature_read"] = full_availability
    else:
        features = frame.loc[:, selected].copy()
    if "candidate_id" not in frame:
        raise ValueError(
            "class-balance economic OOF sweep requires persisted candidate_id row IDs"
        )
    row_ids = frame["candidate_id"].astype("string")
    if row_ids.isna().any() or row_ids.duplicated().any():
        raise ValueError(
            "class-balance economic OOF sweep requires unique non-null candidate_id row IDs"
        )
    observed_sides = _canonical_side_series(
        frame[side_column], source="class-balance population"
    )
    if observed_sides.isna().any() or observed_sides.nunique() != 1:
        raise ValueError(
            "class-balance economic OOF sweep requires one exact training side"
        )
    if canonical_side is not None and observed_sides.iloc[0] != canonical_side:
        raise ValueError(
            "class-balance economic OOF sweep side no longer matches the canonical side"
        )
    structural_fingerprint = _sha256_json(
        {
            "structural_hpo_contract": structural_hpo_contract,
            "frozen_structural_params": params,
        }
    )
    feature_fingerprint = _sha256_json(
        {
            "selected_features": list(selected),
            "selection_fingerprint": selection_fingerprint,
        }
    )
    geometry_fingerprint = _sha256_json(bound_geometry_contract)
    balance_frame = development_frame
    balance_labels = development_labels
    balance_features = features.loc[development_mask, selected]
    balance_row_ids = row_ids.loc[development_mask]
    resource_guard.checkpoint("development_only_balance_mini_sweep")
    mini_sweep = archetype.sweep_purged_catboost_class_balance_arms(
        balance_features,
        balance_labels,
        balance_frame[timestamp_column],
        structural_params=params,
        label_end=balance_frame[label_end_column],
        config=config,
        arm_callback=lambda arm: resource_guard.checkpoint(
            f"development_only_balance_mini_sweep:{arm.arm}"
        ),
        arm_fold_callback=lambda arm, fold_index, _probabilities, _fold_ids: (
            resource_guard.checkpoint(
                f"development_only_balance_mini_sweep:{arm}:fold={fold_index}"
            )
        ),
    )
    mini_sweep_report_path = output_dir / "class_balance_mini_sweep_report.json"
    # Persist the complete compact arm evidence before fail-closing on a
    # missing arm, so an interrupted/unsafe sweep remains auditable.
    _write_json(mini_sweep_report_path, mini_sweep.report())
    mini_sweep_report_sha256 = _sha256_file(mini_sweep_report_path)
    sweep_arms: dict[str, BalanceArmOOF] = {}
    for arm in mini_sweep.arms:
        if arm.oof is None:
            raise RuntimeError(
                "class-balance mini-sweep arm has no OOF output for economic scoring: "
                + arm.arm
            )
        sweep_arms[arm.arm] = BalanceArmOOF(
            probabilities=arm.oof.probabilities,
            fold_ids=arm.oof.fold_ids,
            folds=arm.oof.folds,
            classes=arm.oof.classes,
            structural_fingerprint=structural_fingerprint,
            feature_fingerprint=feature_fingerprint,
            geometry_fingerprint=geometry_fingerprint,
            oof_guard=arm.guard,
            row_ids=balance_row_ids.to_numpy(),
        )
    encoded_final_target = archetype._categorical_target(
        balance_labels, balance_features.index, config=config
    )
    frozen_class_order = tuple(map(str, encoded_final_target.cat.categories))
    if tuple(map(str, sweep_arms["uniform"].classes)) != frozen_class_order:
        raise ValueError(
            "class-balance mini-sweep OOF class order does not match the frozen target order"
        )
    economic_config = EconomicOOFConfig(
        timestamp_col=timestamp_column,
        side_col=side_column,
        label_end_col=label_end_column,
        identity_col="candidate_id",
        embargo=config.embargo,
    )
    economic_balance_frame = _economic_scoring_frame(
        balance_frame,
        side_column=side_column,
        canonical_side=canonical_side,
    )
    economic_report = score_class_balance_oof_economics(
        economic_balance_frame,
        encoded_final_target.cat.codes.to_numpy(),
        sweep_arms,
        config=economic_config,
    )
    selection_provenance = economic_report.get("selection_provenance")
    if not isinstance(selection_provenance, Mapping):
        raise ValueError("economic class-balance scorer returned no arm selection")
    selection_provenance = dict(selection_provenance)
    economic_evidence_sha256 = _sha256_json(
        {
            "schema": economic_report.get("schema"),
            "contract": economic_report.get("contract"),
            "per_arm": economic_report.get("per_arm"),
        }
    )
    economic_contract = economic_report.get("contract")
    scorer_config_digest = (
        economic_contract.get("selector_config_sha256")
        if isinstance(economic_contract, Mapping)
        else None
    )
    if not isinstance(scorer_config_digest, str) or not scorer_config_digest:
        scorer_config_digest = _sha256_json(economic_config.__dict__)
    selection_provenance.update(
        {
            "structural_fingerprint": structural_fingerprint,
            "feature_fingerprint": feature_fingerprint,
            "geometry_fingerprint": geometry_fingerprint,
            "mini_sweep_contract_sha256": _sha256_json(mini_sweep.contract),
            "mini_sweep_report_sha256": mini_sweep_report_sha256,
            "economic_oof_config_sha256": scorer_config_digest,
            "economic_oof_report_sha256": economic_evidence_sha256,
            "candidate_order_sha256": _sha256_json(
                balance_row_ids.astype(str).tolist()
            ),
            "selection_scope": (
                "pre_first_outer_validation_month_development_only"
                if development_oos_routing["enabled"]
                else "legacy_full_population_compatibility_mode"
            ),
            "development_label_end_exclusive": (
                development_oos_routing["development_end_exclusive"]
            ),
        }
    )
    if smoke:
        # A smoke sweep checks plumbing only; it cannot promote a production
        # balance arm even when the tiny fixture happens to cover all gates.
        selection_provenance.update(
            {
                "arm": archetype.CATBOOST_CLASS_BALANCE_ARM_UNIFORM,
                "promotion_eligible": False,
                "mandatory_initial_coverage_complete": False,
                "selection_status": "smoke_nonpromotable_uniform_default",
                "promotion_reason": "explicit_smoke_only_nonpromotable_override",
            }
        )
    economic_report = {**economic_report, "selection_provenance": selection_provenance}
    selected_arm = selection_provenance.get("arm")
    if not isinstance(selected_arm, str) or selected_arm not in sweep_arms:
        raise ValueError("economic class-balance scorer selected an unavailable arm")
    economic_report_path = output_dir / "class_balance_economic_oof_report.json"
    _write_json(economic_report_path, economic_report)
    class_balance_provenance = _class_balance_artifact_provenance(
        structural_hpo_contract=structural_hpo_contract,
        mini_sweep=mini_sweep,
        economic_report=economic_report,
        economic_config=economic_config,
        structural_fingerprint=structural_fingerprint,
        feature_fingerprint=feature_fingerprint,
        geometry_fingerprint=geometry_fingerprint,
        mini_sweep_report_path=mini_sweep_report_path,
        economic_report_path=economic_report_path,
        economic_evidence_sha256=economic_evidence_sha256,
    )
    params = {
        **dict(mini_sweep.structural_params),
        "class_balance_arm": selected_arm,
        "class_balance_selection_provenance": selection_provenance,
    }
    balance_selection_oof = {arm.arm: arm.oof for arm in mini_sweep.arms}[selected_arm]
    assert balance_selection_oof is not None
    outer_oof_report: dict[str, Any] | None = None
    if canonical_store_mode and not smoke:
        frozen_params_provenance = {
            "schema": archetype.FIXED_MONTHLY_OUTER_OOF_FROZEN_PARAMS_SCHEMA,
            "selection_scope": ("pre_first_outer_validation_month_development_only"),
            "development_label_end_exclusive": (
                development_oos_routing["development_end_exclusive"]
            ),
            "params_sha256": _sha256_json(params),
            "final_refit_used_for_selection": False,
            "class_balance_selection_scope": (
                "pre_first_outer_validation_month_development_only"
            ),
        }
        resource_guard.checkpoint("fixed_monthly_outer_oof")
        outer_result = archetype.fit_fixed_monthly_outer_oof_catboost(
            features.loc[:, selected],
            labels,
            frame[timestamp_column],
            label_end=frame[label_end_column],
            params=params,
            frozen_params_provenance=frozen_params_provenance,
            row_ids=row_ids.to_numpy(),
            config=config,
            fold_callback=lambda window, _probabilities, _fold_ids: (
                resource_guard.checkpoint(
                    "fixed_monthly_outer_oof:"
                    + window.validation_start.strftime("%Y-%m")
                )
            ),
        )
        oof = outer_result.oof
        outer_oof_report = outer_result.report()
        _write_json(
            output_dir / "fixed_monthly_outer_oof_report.json", outer_oof_report
        )
    elif development_oos_routing["enabled"]:
        # Smoke/in-memory compatibility still evaluates the frozen choices on
        # rows outside the development selector instead of reusing the
        # development balance-selection predictions as reported OOF.
        oof = archetype.fit_purged_chronological_oof_catboost(
            features.loc[:, selected],
            labels,
            frame[timestamp_column],
            label_end=frame[label_end_column],
            config=config,
            params=params,
        )
    else:
        oof = balance_selection_oof
    resource_guard.checkpoint("final_refit")
    if "class_balance_selection_provenance" in params and (
        not smoke or bool(selection_provenance.get("promotion_eligible", False))
    ):
        # The arm was selected solely from purged OOF rows above.  Derive its
        # bounded weights again from the actual full final-label population;
        # no HPO-sample weight vector may leak into the final fit.
        params = archetype.rematerialize_final_class_balance_params(
            params,
            labels,
            config=config,
            allow_nonpromotable_selection=bool(smoke),
        )
        final_balance = params.get("class_balance_provenance")
        if not isinstance(final_balance, Mapping):
            raise ValueError("final class-balance rematerialisation lacks provenance")
        class_balance_provenance["final_refit_weight_provenance"] = dict(final_balance)
        class_balance_provenance["final_refit_weight_provenance_sha256"] = _sha256_json(
            final_balance
        )
        class_balance_provenance["final_refit_weights_sha256"] = _sha256_json(
            params.get("class_balance_final_weights")
        )
    elif smoke:
        class_balance_provenance["final_refit_weight_materialisation"] = (
            "skipped_smoke_nonpromotable_uniform"
        )
    classifier = _fit_final_classifier(
        features, labels, selected, config=config, params=params
    )
    if selector is not None:
        classifier.selector = selector
    classifier.training_report = {
        "configured_universe": list(universe),
        "model_universe": list(model_universe),
        "selection_sample_rows": int(len(selection_frame)),
        "selected_features": list(selected),
        "training_phase_order": [
            "fast_feature_selection",
            "permutation_feature_selection",
            "side_local_geometry_selected_between_selection_and_hpo",
            "structural_hpo_on_frozen_selected_features",
            "four_arm_development_only_oof_economic_balance_sweep",
            "fixed_monthly_may_june_july_outer_oof_with_frozen_choices",
            "final_refit_after_outer_oof",
        ],
        "hpo_feature_count": int(len(selected)),
        "future_training_taxonomy": taxonomy_contract,
        "class_support_gate": class_support_gate,
        "effective_model_params": params,
        "catboost_resource_contract": archetype.catboost_resource_contract(config),
        "selector_backend": selector.proxy_backend
        if selector is not None
        else "reused_checkpoint",
        "pre_permutation_oof_diagnostics": (
            pre_permutation_oof.diagnostics if pre_permutation_oof is not None else None
        ),
        "oof_diagnostics": oof.diagnostics,
        "fixed_monthly_outer_oof": outer_oof_report,
        "hpo": hpo_report,
        "class_balance": class_balance_provenance,
        "geometry_contract": bound_geometry_contract,
        "hpo_sampling_contract": hpo_sampling_contract,
        "permutation_stages": permutation_records,
        "permutation_stage_metrics": permutation_stage_metrics,
        "feature_selection_hpo_reuse": reuse_provenance,
    }

    valid_oof = oof.fold_ids >= 0
    oof_frame = frame.loc[valid_oof, [timestamp_column, label_end_column]].copy()
    for column in ("candidate_id", "__symbol__", "symbol", "side", "side_name"):
        if column in frame and column not in oof_frame:
            oof_frame[column] = frame.loc[valid_oof, column].to_numpy()
    if "candidate_id" not in oof_frame:
        raise ValueError(
            "CatBoost OOF source requires persisted candidate_id for strict execution-EV joins"
        )
    oof_frame["path_archetype_raw"] = raw_labels.loc[valid_oof].to_numpy()
    oof_frame["path_shape_archetype_raw"] = shape_labels.loc[valid_oof].to_numpy()
    oof_frame["path_archetype"] = labels.loc[valid_oof].to_numpy()
    oof_frame["path_shape_archetype"] = labels.loc[valid_oof].to_numpy()
    realization_strength = (
        frame["path_realization_strength"].astype("string")
        if "path_realization_strength" in frame
        else raw_labels.str.rsplit("__", n=1).str[-1]
    )
    oof_frame["path_realization_strength"] = realization_strength.loc[
        valid_oof
    ].to_numpy()
    oof_frame["oof_fold_id"] = oof.fold_ids[valid_oof]
    provenance = _catboost_oof_provenance(
        frame,
        oof,
        timestamp_column=timestamp_column,
        label_end_column=label_end_column,
        config=config,
    )
    for column in provenance:
        oof_frame[column] = provenance.loc[valid_oof, column].to_numpy()
    oof_frame["available_at"] = pd.to_datetime(
        oof_frame[timestamp_column], utc=True, errors="raise"
    )
    probabilities, probability_classes = _aligned_oof_probabilities(
        oof.probabilities[valid_oof],
        oof.classes,
    )
    for index, class_name in enumerate(probability_classes):
        oof_frame[f"probability__{class_name}"] = probabilities[:, index]
    oof_frame["predicted_path_archetype"] = np.asarray(probability_classes, dtype=str)[
        np.argmax(probabilities, axis=1)
    ]
    oof_frame["predicted_path_shape_archetype"] = oof_frame["predicted_path_archetype"]
    probability_contract = archetype.path_archetype_probability_contract(
        probabilities, probability_classes, index=oof_frame.index
    )
    oof_frame["probability_entropy"] = _entropy(probabilities)
    for column in (
        "max_probability",
        "normalized_entropy",
        "top2_probability_margin",
        "adverse_probability_mass",
        "favorable_probability_mass",
    ):
        oof_frame[column] = probability_contract[column].to_numpy()
    oof_path = output_dir / "oof_probabilities.parquet"
    oof_frame.to_parquet(oof_path, index=False, compression="zstd")
    prediction_columns = {
        column: {"role": "pre_entry_path_archetype_oof_prediction", "target": False}
        for column in oof_frame.columns
        if column.startswith("probability__")
        or column
        in {
            "predicted_path_archetype",
            "predicted_path_shape_archetype",
            "probability_entropy",
            "max_probability",
            "normalized_entropy",
            "top2_probability_margin",
            "adverse_probability_mass",
            "favorable_probability_mass",
        }
    }
    role_manifest = {
        "schema": "path_archetype_oof_prediction_role_v1",
        "prediction_role": "path_archetype_oof",
        "development_oos_routing": development_oos_routing,
        "fixed_monthly_outer_oof": outer_oof_report,
        "future_training_taxonomy": taxonomy_contract,
        "source_artifact_sha256": _sha256_file(oof_path),
        "source_artifact": str(oof_path),
        "prediction_columns": prediction_columns,
        "identity_columns": [
            timestamp_column,
            "__symbol__",
            "side_name",
            "candidate_id",
        ],
        "fold_provenance_columns": {
            "fold": "oof_fold_id",
            "validation_start": "validation_start",
            "latest_train_decision": "latest_train_decision_ts",
            "training_information_cutoff": "train_decision_cutoff",
            "latest_resolved_training_label": "label_resolution_available_at",
            "prediction_available_at": "available_at",
        },
        "class_balance": class_balance_provenance,
    }
    role_manifest["prediction_role_manifest_sha256"] = _signed_manifest_hash(
        role_manifest
    )
    _write_json(output_dir / "oof_probabilities.role_manifest.json", role_manifest)

    joblib.dump(classifier, output_dir / "path_archetype_classifier.joblib", compress=3)
    class_consolidation.to_csv(
        output_dir / "supervised_class_consolidation.csv", index=False
    )
    _write_json(
        output_dir / "discovery_manifest.json",
        {
            "schema": RUNNER_SCHEMA,
            "discovery_end_exclusive": discovery_cutoff,
            "development_oos_routing": development_oos_routing,
            "label_source": summary_source,
            "supervised_target_source": target_source,
            "future_training_taxonomy": taxonomy_contract,
            "supervised_class_min_discovery_rows": None,
            "raw_path_archetype_count": int(raw_labels.nunique()),
            "effective_path_archetype_count": int(labels.nunique()),
            "frozen_rule_manifest": None,
        },
    )
    _write_json(
        output_dir / "feature_selection_manifest.json",
        {
            "schema": RUNNER_SCHEMA,
            "future_training_taxonomy": taxonomy_contract,
            "development_oos_routing": development_oos_routing,
            "configured_universe": list(universe),
            "model_universe": list(model_universe),
            "selection": selector_manifest,
            "permutation": permutation_records,
            "permutation_stage_metrics": permutation_stage_metrics,
            "permutation_acceleration_contract": (
                archetype.staged_permutation_acceleration_contract(config)
            ),
            "final_selected_features": list(selected),
            "feature_selection_hpo_fingerprint": selection_hpo_contract["fingerprint"],
            "feature_selection_fingerprint": selection_fingerprint,
            "feature_selection_hpo_reuse": reuse_provenance,
            "catboost_resource_contract": archetype.catboost_resource_contract(config),
        },
    )
    if static_store_manifest is not None:
        _write_json(
            output_dir / "feature_availability_manifest.json",
            {
                "schema": RUNNER_SCHEMA,
                **static_store_manifest,
                "development_oos_routing": development_oos_routing,
                "selector_availability": dict(
                    selector_manifest.get("availability", {})
                ),
                "selected_features": list(selected),
            },
        )
    _write_json(
        output_dir / "hpo_manifest.json",
        {
            "schema": RUNNER_SCHEMA,
            "future_training_taxonomy": taxonomy_contract,
            "feature_contract_frozen_before_hpo": True,
            "development_oos_routing": development_oos_routing,
            "hpo_feature_count": int(len(selected)),
            "hpo_features": list(selected),
            "no_improvement_patience_trials": int(hpo_no_improvement_trials),
            "hpo": hpo_report,
            "class_balance": class_balance_provenance,
            "selection_proxy_params": {
                "iterations": int(selection_iterations),
                "od_wait": int(selection_od_wait),
            },
            "hpo_study_path": hpo_study_path,
            "hpo_progress_path": hpo_progress_path,
            "catboost_resource_contract": archetype.catboost_resource_contract(config),
            "feature_selection_hpo_fingerprint": selection_hpo_contract["fingerprint"],
            "feature_selection_fingerprint": selection_fingerprint,
            "feature_selection_hpo_reuse": reuse_provenance,
            "hpo_sampling_contract": hpo_sampling_contract,
        },
    )
    _write_json(
        output_dir / "training_report.json",
        {
            "schema": RUNNER_SCHEMA,
            "future_training_taxonomy": taxonomy_contract,
            "development_oos_routing": development_oos_routing,
            "pre_permutation_oof_diagnostics": (
                pre_permutation_oof.diagnostics
                if pre_permutation_oof is not None
                else None
            ),
            "oof_diagnostics": oof.diagnostics,
            "fixed_monthly_outer_oof": outer_oof_report,
            "class_balance": class_balance_provenance,
            "class_names": list(classifier.class_names),
            "class_semantics": "merged seven-class future path taxonomy",
            "oof_rows": int(valid_oof.sum()),
            "oof_total_rows": int(len(frame)),
            "selected_features": list(selected),
            "effective_model_params": params,
            "catboost_resource_contract": archetype.catboost_resource_contract(config),
            "feature_selection_hpo_fingerprint": selection_hpo_contract["fingerprint"],
            "feature_selection_fingerprint": selection_fingerprint,
            "feature_selection_hpo_reuse": reuse_provenance,
            "training_phase_order": [
                "fast_feature_selection",
                "permutation_feature_selection",
                "side_local_geometry_selected_between_selection_and_hpo",
                "structural_hpo_on_frozen_selected_features",
                "four_arm_development_only_oof_economic_balance_sweep",
                "fixed_monthly_may_june_july_outer_oof_with_frozen_choices",
                "final_refit_after_outer_oof",
            ],
            "hpo_sampling_contract": hpo_sampling_contract,
            "permutation_stage_metrics": permutation_stage_metrics,
            "permutation_acceleration_contract": (
                archetype.staged_permutation_acceleration_contract(config)
            ),
        },
    )
    manifest = {
        "schema": RUNNER_SCHEMA,
        "stage": "model_hpo_final",
        "side": canonical_side,
        "source": source,
        "rows": int(len(frame)),
        "candidate_identity_sha256": _side_candidate_identity_sha256(
            frame,
            timestamp_column=timestamp_column,
            symbol_column=IDENTITY_SYMBOL_COLUMN,
            side_column=side_column,
        ),
        "candidate_population_contract": (
            "exact canonical OOF base top-fraction population; identity hash "
            "must match auxiliary and residual-alpha handoffs"
        ),
        "canonical_input_contract": canonical_input_contract,
        "geometry_contract": bound_geometry_contract,
        "class_support_gate": class_support_gate,
        "discovery_rows": int(discovery_mask.sum()),
        "discovery_end_exclusive": discovery_cutoff,
        "development_oos_routing": development_oos_routing,
        "fixed_monthly_outer_oof": outer_oof_report,
        "timestamp_column": timestamp_column,
        "label_end_column": label_end_column,
        "utc_timestamp_contract": "naive timestamps interpreted as UTC on ingest",
        "label_source": summary_source,
        "supervised_target_source": target_source,
        "future_training_taxonomy": taxonomy_contract,
        "supervised_class_min_discovery_rows": None,
        "raw_path_archetype_count": int(raw_labels.nunique()),
        "effective_path_archetype_count": int(labels.nunique()),
        "supervised_path_shape_class_count": int(len(expected_classes)),
        "classifier_input_contract": "configured pre-entry base/meta features only; path_arch_* summaries are labels only",
        "frozen_ae_gmm_sidecar": frozen_sidecar_contract,
        "feature_loading_contract": (
            "static store: deterministic beginning/middle/end selection sample and chronological side_x_class-stratified HPO sample, then selected features only for full OOF/final fit"
            if canonical_store_mode
            else "in-memory pre-entry feature frame"
        ),
        "purge_embargo_hours": float(embargo_hours),
        "oof_folds_requested": int(effective_folds),
        "hpo_trials_requested": int(effective_trials),
        "hpo_rows": int(effective_hpo_rows),
        "hpo_sample_contract": hpo_sampling_contract,
        "hpo_folds": int(hpo_config.oof_folds),
        "hpo_search_iterations": int(hpo_iterations),
        "hpo_search_od_wait": int(hpo_od_wait),
        "selection_proxy_iterations": int(selection_iterations),
        "selection_proxy_od_wait": int(selection_od_wait),
        "hpo_study_path": hpo_study_path,
        "hpo_progress_path": hpo_progress_path,
        "no_wall_clock_timeout": True,
        "hpo_no_improvement_patience_trials": int(hpo_no_improvement_trials),
        "feature_selection_hpo_fingerprint": selection_hpo_contract["fingerprint"],
        "feature_selection_fingerprint": selection_fingerprint,
        "feature_selection_hpo_reuse": reuse_provenance,
        "training_phase_order": [
            "fast_feature_selection",
            "permutation_feature_selection",
            "side_local_geometry_selected_between_selection_and_hpo",
            "structural_hpo_on_frozen_selected_features",
            "four_arm_development_only_oof_economic_balance_sweep",
            "fixed_monthly_may_june_july_outer_oof_with_frozen_choices",
            "final_refit_after_outer_oof",
        ],
        "catboost_resource_contract": archetype.catboost_resource_contract(config),
        "training_resource_guard": _resource_guard_contract(resource_guard),
        "selection_rows_requested": int(selection_rows),
        "selection_rows_effective": int(len(selection_frame)),
        "permutation_stage_metrics": permutation_stage_metrics,
        "permutation_acceleration_contract": (
            archetype.staged_permutation_acceleration_contract(config)
        ),
        "smoke_caps": {
            "max_rows": SMOKE_MAX_ROWS if smoke else None,
            "max_features": SMOKE_MAX_FEATURES if smoke else None,
            "max_iterations_after_hpo": SMOKE_MAX_ITERATIONS if smoke else None,
        },
        "smoke": bool(smoke),
        "artifacts": {
            "classifier": "path_archetype_classifier.joblib",
            "supervised_class_consolidation": "supervised_class_consolidation.csv",
            "oof_probabilities": "oof_probabilities.parquet",
            "oof_prediction_role_manifest": "oof_probabilities.role_manifest.json",
            "feature_selection": "feature_selection_manifest.json",
            "hpo": "hpo_manifest.json",
            "class_balance_mini_sweep": "class_balance_mini_sweep_report.json",
            "class_balance_economic_oof": "class_balance_economic_oof_report.json",
            "mda_progress": MDA_PROGRESS_FILENAME,
            "hpo_progress": hpo_progress_path.name,
            "hpo_study": hpo_study_path.name,
            "feature_selection_hpo_contract": FEATURE_SELECTION_HPO_CONTRACT_FILENAME,
            "report": "training_report.json",
            **(
                {"feature_availability": "feature_availability_manifest.json"}
                if canonical_store_mode
                else {}
            ),
        },
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--side", required=True, choices=("long", "short"))
    parser.add_argument(
        "--feature-dir",
        required=True,
        type=Path,
        help="Timestamped shared static feature store used for all classifier inputs",
    )
    parser.add_argument(
        "--canonical-candidate-path",
        required=True,
        type=Path,
        help="Exact canonical top-40 candidate parquet; validated by key and SHA-256.",
    )
    parser.add_argument(
        "--canonical-candidate-manifest",
        required=True,
        type=Path,
        help="Manifest that advertises the candidate parquet SHA-256.",
    )
    parser.add_argument(
        "--canonical-context-path",
        required=True,
        type=Path,
        help="Exact canonical frozen side-context parquet; validated by key and SHA-256.",
    )
    parser.add_argument(
        "--canonical-context-manifest",
        required=True,
        type=Path,
        help="Manifest that advertises the context parquet SHA-256 and side AE/GMM binding.",
    )
    parser.add_argument(
        "--ae-gmm-state",
        type=Path,
        default=None,
        help="Optional explicit side-local AE/GMM state; defaults to the context manifest binding.",
    )
    parser.add_argument(
        "--ae-gmm-state-manifest",
        type=Path,
        default=None,
        help="Optional explicit side-local AE/GMM manifest; defaults to the context manifest binding.",
    )
    parser.add_argument(
        "--geometry-contract",
        type=Path,
        default=None,
        help="Completed side-local geometry contract required for model-HPO/final OOF.",
    )
    parser.add_argument(
        "--stage",
        choices=("selection_only", "model_hpo_final"),
        default="model_hpo_final",
        help="Run only frozen side-local feature selection, or resume after geometry for HPO/OOF/final refit.",
    )
    parser.add_argument(
        "--frozen-ae-gmm-sidecar",
        type=Path,
        default=None,
        help="Exact frozen AE/GMM output sidecar keyed by UTC timestamp, symbol, and side",
    )
    parser.add_argument(
        "--frozen-ae-gmm-manifest",
        type=Path,
        default=None,
        help="Optional manifest proving the frozen AE/GMM output contract",
    )
    parser.add_argument(
        "--discovery-end",
        required=True,
        help="Exclusive UTC boundary for train-only path discovery",
    )
    parser.add_argument(
        "--development-end",
        default=None,
        help=(
            "Optional exclusive UTC boundary for strict feature-selection/HPO "
            "development routing. Rows resolved at or after it are excluded "
            "from development fitting and reserved for outer-OOF evaluation."
        ),
    )
    parser.add_argument("--timestamp-column", default=DEFAULT_TIMESTAMP_COLUMN)
    parser.add_argument("--label-end-column", default=DEFAULT_LABEL_END_COLUMN)
    parser.add_argument("--mandatory-features", type=Path, default=None)
    parser.add_argument("--config-json", type=Path, default=None)
    parser.add_argument("--hpo-trials", type=int, default=75)
    parser.add_argument("--oof-folds", type=int, default=5)
    parser.add_argument("--selection-rows", type=int, default=45_000)
    parser.add_argument("--hpo-rows", type=int, default=24_000)
    parser.add_argument("--hpo-folds", type=int, default=3)
    parser.add_argument("--hpo-iterations", type=int, default=3_000)
    parser.add_argument("--hpo-od-wait", type=int, default=150)
    parser.add_argument(
        "--hpo-no-improvement-trials",
        type=int,
        default=DEFAULT_HPO_NO_IMPROVEMENT_TRIALS,
        help=(
            "Stop the Optuna study after this many consecutive terminal trials "
            "without a new best objective; no wall-clock timeout is used."
        ),
    )
    parser.add_argument(
        "--selection-iterations",
        type=int,
        default=PROXY_SELECTION_ITERATIONS,
        help="Bounded CatBoost iterations for pre-MDA OOF and MDA folds only.",
    )
    parser.add_argument(
        "--selection-od-wait",
        type=int,
        default=PROXY_SELECTION_OD_WAIT,
        help="Early-stopping wait for pre-MDA OOF and MDA folds only.",
    )
    parser.add_argument(
        "--hpo-proxy",
        action="store_true",
        help=(
            "Use the bounded HPO proxy: 20 trials, 8k rows, 2 folds, "
            "400 iterations, and od_wait 40; no wall-clock timeout."
        ),
    )
    parser.add_argument(
        "--hpo-study-path",
        type=Path,
        default=None,
        help="Durable Optuna SQLite path; defaults to OUTPUT_DIR/hpo_study.sqlite3.",
    )
    parser.add_argument(
        "--hpo-progress-path",
        type=Path,
        default=None,
        help="Atomic trial progress JSON path; defaults to OUTPUT_DIR/hpo_progress.json.",
    )
    parser.add_argument("--catboost-threads", type=int, default=4)
    parser.add_argument(
        "--catboost-os-reserve-gib",
        type=float,
        default=4.0,
        help="Physical RAM reserved for the OS before CatBoost thread and RAM caps.",
    )
    parser.add_argument(
        "--unsafe-allow-catboost-threads",
        action="store_true",
        help="Unsafe: bypass the RAM-derived CatBoost thread cap for this run.",
    )
    parser.add_argument("--embargo-hours", type=float, default=24.0)
    parser.add_argument("--min-class-rows", type=int, default=400)
    parser.add_argument("--random-state", type=int, default=20260722)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--resource-min-free-ram-gib",
        type=float,
        default=2.0,
        help="Fail closed when available RAM is below this threshold (default: 2 GiB).",
    )
    parser.add_argument(
        "--resource-max-process-rss-gib",
        type=float,
        default=12.0,
        help="Fail closed when this process exceeds this RSS threshold (default: 12 GiB).",
    )
    parser.add_argument(
        "--resource-min-free-disk-gib",
        type=float,
        default=10.0,
        help="Fail closed when free output-filesystem disk is below this threshold (default: 10 GiB).",
    )
    parser.add_argument(
        "--resource-check-interval-seconds",
        type=float,
        default=60.0,
        help="Minimum interval between boundary resource samples (default: 60 seconds).",
    )
    parser.add_argument(
        "--resource-telemetry-path",
        type=Path,
        default=None,
        help=(
            "Append guard events as JSONL here; defaults to "
            "OUTPUT_DIR/training_resource_telemetry.jsonl."
        ),
    )
    parser.add_argument(
        "--feature-selection-hpo-checkpoint",
        type=Path,
        default=None,
        help=(
            "Completed strict contract file/directory, or an explicit legacy run "
            "directory containing both checkpoints plus run_manifest.json."
        ),
    )
    parser.add_argument(
        "--feature-selection-hpo-registry-root",
        type=Path,
        default=None,
        help="Root searched recursively for exact reusable feature-selection/HPO contracts.",
    )
    parser.add_argument(
        "--force-feature-selection-hpo",
        action="store_true",
        help="Ignore reusable contracts and rerun feature selection plus HPO.",
    )
    parser.add_argument(
        "--stop-after-hpo",
        action="store_true",
        help="Persist the frozen feature/parameter contract without fitting final OOF models.",
    )
    args = parser.parse_args()
    manifest = run_pipeline(
        args.input,
        args.output_dir,
        discovery_end=args.discovery_end,
        development_end=args.development_end,
        side=args.side,
        feature_dir=args.feature_dir,
        canonical_candidate_path=args.canonical_candidate_path,
        canonical_candidate_manifest=args.canonical_candidate_manifest,
        canonical_context_path=args.canonical_context_path,
        canonical_context_manifest=args.canonical_context_manifest,
        ae_gmm_state=args.ae_gmm_state,
        ae_gmm_state_manifest=args.ae_gmm_state_manifest,
        geometry_contract=args.geometry_contract,
        stage=args.stage,
        frozen_ae_gmm_sidecar=args.frozen_ae_gmm_sidecar,
        frozen_ae_gmm_manifest=args.frozen_ae_gmm_manifest,
        timestamp_column=args.timestamp_column,
        label_end_column=args.label_end_column,
        future_path_column=None,
        mandatory_features=_read_optional_list(args.mandatory_features),
        config_mapping=_read_config_mapping(args.config_json),
        hpo_trials=args.hpo_trials,
        oof_folds=args.oof_folds,
        selection_rows=args.selection_rows,
        hpo_rows=args.hpo_rows,
        hpo_folds=args.hpo_folds,
        hpo_iterations=args.hpo_iterations,
        hpo_od_wait=args.hpo_od_wait,
        hpo_no_improvement_trials=args.hpo_no_improvement_trials,
        selection_iterations=args.selection_iterations,
        selection_od_wait=args.selection_od_wait,
        catboost_threads=args.catboost_threads,
        catboost_os_reserve_gib=args.catboost_os_reserve_gib,
        unsafe_allow_catboost_threads=args.unsafe_allow_catboost_threads,
        embargo_hours=args.embargo_hours,
        min_class_rows=args.min_class_rows,
        random_state=args.random_state,
        max_rows=args.max_rows,
        smoke=args.smoke,
        stop_after_hpo=args.stop_after_hpo,
        feature_selection_hpo_checkpoint=args.feature_selection_hpo_checkpoint,
        feature_selection_hpo_registry_root=args.feature_selection_hpo_registry_root,
        force_feature_selection_hpo=args.force_feature_selection_hpo,
        hpo_study_path=args.hpo_study_path,
        hpo_progress_path=args.hpo_progress_path,
        hpo_proxy=args.hpo_proxy,
        resource_min_free_ram_gib=args.resource_min_free_ram_gib,
        resource_max_process_rss_gib=args.resource_max_process_rss_gib,
        resource_min_free_disk_gib=args.resource_min_free_disk_gib,
        resource_check_interval_seconds=args.resource_check_interval_seconds,
        resource_telemetry_path=args.resource_telemetry_path,
    )
    print(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
