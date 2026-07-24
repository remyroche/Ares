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
from extreme_price_movements.static_feature_store import (  # noqa: E402
    STATIC_FEATURE_ENDPOINT_VERSION,
    read_static_features,
)

RUNNER_SCHEMA = "run_catboost_path_archetype_classifier_v9_merged_raw_probability"
FEATURE_SELECTION_HPO_CONTRACT_SCHEMA = (
    "catboost_path_archetype_feature_selection_hpo_contract_v1"
)
FEATURE_SELECTION_HPO_CONTRACT_FILENAME = "feature_selection_hpo_contract.json"
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
LOGGER = logging.getLogger(__name__)
IDENTITY_SYMBOL_COLUMN = "__symbol__"
IDENTITY_SIDE_COLUMNS = ("side", "side_name", "__side__")
FROZEN_AE_GMM_KEY_COLUMNS = ("__ts__", "__symbol__", "side")
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
        output[column] = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    fold_lookup = {int(fold.fold_id): fold for fold in oof.folds}
    for fold_id in sorted(int(value) for value in np.unique(oof.fold_ids) if value >= 0):
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
                    timestamps.iloc[prior]
                    < validation_start - config.embargo
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
    if payload.get("output_features") != expected:
        raise ValueError(
            "frozen AE/GMM manifest output_features do not match AE_GMM_FEATURE_COLUMNS"
        )
    return payload


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
    expected_features = tuple(map(str, AE_GMM_FEATURE_COLUMNS))
    schema_names = tuple(map(str, pq.ParquetFile(sidecar_path).schema_arrow.names))
    required_columns = set(FROZEN_AE_GMM_KEY_COLUMNS) | set(expected_features)
    missing_columns = sorted(required_columns.difference(schema_names))
    if missing_columns:
        raise ValueError(
            "frozen AE/GMM sidecar is missing required columns: "
            + ", ".join(missing_columns[:8])
        )
    manifest = _read_frozen_ae_gmm_manifest(manifest_path)
    identity = _frozen_sidecar_identity(
        frame, timestamp_column=timestamp_column, side_column=side_column
    )
    sidecar_sql = _sql_literal(sidecar_path)
    join = (
        "epoch_ns(l.__ts__) = epoch_ns(s.__ts__) "
        "AND l.__symbol__ = s.__symbol__ AND l.side = CAST(s.side AS TINYINT)"
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
        finite_predicate = " OR ".join(
            "NOT coalesce(isfinite(CAST("
            f"s.{_quote_identifier(resolved_columns[name])} AS DOUBLE)), false)"
            for name in expected_features
        )
        duplicate = con.execute(
            f"""
            SELECT 1
            FROM read_parquet({sidecar_sql})
            GROUP BY epoch_ns(__ts__), __symbol__, CAST(side AS TINYINT)
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
                sum(CASE WHEN s.__ts__ IS NOT NULL AND ({finite_predicate}) THEN 1 ELSE 0 END)
                    AS nonfinite_rows
            FROM label_keys AS l
            LEFT JOIN read_parquet({sidecar_sql}) AS s ON {join}
            """
        ).fetchone()
    finally:
        con.close()
    assert coverage is not None
    label_rows, matched_rows, missing_rows, nonfinite_rows = map(int, coverage)
    if missing_rows:
        raise ValueError(
            "frozen AE/GMM sidecar does not cover every label key: "
            f"missing={missing_rows}, labels={label_rows}"
        )
    if nonfinite_rows:
        raise ValueError(
            "frozen AE/GMM sidecar has non-finite generated outputs: "
            f"rows={nonfinite_rows}"
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
        "nonfinite_rows": nonfinite_rows,
        "join_contract": "exact UTC timestamp, symbol, and canonical int8 side",
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
             AND l.side = CAST(s.side AS TINYINT)
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
    if not np.isfinite(matrix.to_numpy(dtype=float)).all():
        raise RuntimeError("frozen AE/GMM sidecar load produced non-finite values")
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
            for name, rows in support["class"].value_counts(sort=False).sort_index().items()
        ],
        "side_support": [
            {"side": str(name), "rows": int(rows)}
            for name, rows in support["side"].value_counts(sort=False).sort_index().items()
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
        for _key, positions in strata.groupby(["side", "class"], sort=True).groups.items()
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
            key=lambda index: (-float(raw[index] - additions[index]), -int(counts[index]), int(index)),
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
    aligned = np.zeros((len(probabilities), len(MERGED_PATH_ARCHETYPE_CLASSES)), dtype=float)
    positions = {name: index for index, name in enumerate(MERGED_PATH_ARCHETYPE_CLASSES)}
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
    available = [column for column in _PERMUTATION_STAGE_METRIC_COLUMNS if column in frame]
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
    taxonomy_contract: Mapping[str, Any],
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
        "candidate_identity_sha256": candidate_identity_sha256(
            frame,
            columns=(timestamp_column, IDENTITY_SYMBOL_COLUMN, side_column),
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
            "hpo_no_improvement_patience_trials": int(
                hpo_no_improvement_trials
            ),
            "hpo_sampling_contract": dict(hpo_sample_contract),
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
    selection_inputs["selection_settings"] = selection_settings
    return _sha256_json(selection_inputs)


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
    required = {"schema", "status", "fingerprint", "selected_features", "selection", "permutation"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(
            "feature-selection checkpoint is incomplete and cannot be resumed: "
            + ", ".join(missing)
        )
    if payload["schema"] != RUNNER_SCHEMA or payload["status"] != "feature_selection_complete":
        raise ValueError("feature-selection checkpoint is not a resumable current-run checkpoint")
    stored_fingerprint = payload.get("selection_fingerprint", payload["fingerprint"])
    if stored_fingerprint != expected_selection_fingerprint:
        raise ValueError(
            "feature-selection checkpoint fingerprint does not match the current exact contract for selection"
        )
    if not isinstance(payload["selected_features"], list) or not all(
        isinstance(value, str) for value in payload["selected_features"]
    ):
        raise ValueError("feature-selection checkpoint has invalid selected features")
    if not isinstance(payload["selection"], Mapping) or not isinstance(payload["permutation"], list):
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
        "schema", "status", "fingerprint", "initial_selected_features",
        "selection", "completed_stages",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError("MDA progress checkpoint is incomplete: " + ", ".join(missing))
    if payload["schema"] != RUNNER_SCHEMA or payload["status"] != "mda_running":
        raise ValueError("MDA progress checkpoint is not resumable for this runner")
    stored_fingerprint = payload.get("selection_fingerprint", payload["fingerprint"])
    if stored_fingerprint != expected_selection_fingerprint:
        raise ValueError("MDA progress checkpoint does not match the current exact selection contract")
    if not isinstance(payload["initial_selected_features"], list) or not all(
        isinstance(value, str) for value in payload["initial_selected_features"]
    ):
        raise ValueError("MDA progress checkpoint has invalid initial selected features")
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
    payload = _read_json_object(path, artifact_name="reusable feature-selection checkpoint")
    required = {"schema", "status", "selected_features", "selection", "permutation"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(
            "reusable feature-selection checkpoint is incomplete: " + ", ".join(missing)
        )
    if payload["schema"] != RUNNER_SCHEMA or payload["status"] != "feature_selection_complete":
        raise ValueError("reusable feature-selection checkpoint is not complete")
    if not isinstance(payload["selected_features"], list) or not all(
        isinstance(value, str) for value in payload["selected_features"]
    ):
        raise ValueError("reusable feature-selection checkpoint has invalid selected features")
    if not isinstance(payload["selection"], Mapping) or not isinstance(payload["permutation"], list):
        raise ValueError("reusable feature-selection checkpoint has invalid selection evidence")
    return payload


def _selection_fingerprint_from_completed_contract(contract: Mapping[str, Any]) -> str | None:
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
        if _selection_fingerprint_from_completed_contract(contract) != expected_selection_fingerprint:
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
    feature_dir: Path | None = None,
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
    hpo_no_improvement_trials: int = 30,
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
) -> dict[str, Any]:
    """Run deterministic-label feature selection, HPO, OOF, and final fit."""
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
    selection_checkpoint_path = output_dir / "feature_selection_checkpoint.json"
    mda_progress_path = output_dir / MDA_PROGRESS_FILENAME
    hpo_study_path = Path(hpo_study_path or output_dir / HPO_STUDY_FILENAME)
    hpo_progress_path = Path(hpo_progress_path or output_dir / HPO_PROGRESS_FILENAME)
    resumable_names = {
        selection_checkpoint_path.name,
        mda_progress_path.name,
        hpo_progress_path.name,
    }
    if hpo_study_path.parent == output_dir:
        resumable_names.update({
            hpo_study_path.name,
            f"{hpo_study_path.name}-wal",
            f"{hpo_study_path.name}-shm",
            f"{hpo_study_path.name}-journal",
        })
    if output_dir.exists() and any(output_dir.iterdir()):
        existing = {path.name for path in output_dir.iterdir()}
        if not existing.issubset(resumable_names):
            raise FileExistsError(
                f"refusing to overwrite non-empty output directory: {output_dir}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)

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
    if frozen_ae_gmm_manifest is not None and frozen_ae_gmm_sidecar is None:
        raise ValueError("--frozen-ae-gmm-manifest requires --frozen-ae-gmm-sidecar")
    if frozen_ae_gmm_sidecar is not None and not canonical_store_mode:
        raise ValueError("frozen AE/GMM sidecars require canonical --feature-dir mode")
    if isinstance(input_data, Path) and not canonical_store_mode:
        raise ValueError(
            "parquet input requires --feature-dir; only in-memory DataFrame mode "
            "may supply pre-entry features directly"
        )
    frame = _normalise_input(
        raw,
        timestamp_column=timestamp_column,
        label_end_column=label_end_column,
        max_rows=effective_max_rows,
    )
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
                    AE_GMM_FEATURE_COLUMNS
                    if frozen_sidecar_contract is not None
                    else ()
                ),
            ],
            config_mapping=config_mapping,
            frozen_representation_features=(
                AE_GMM_FEATURE_COLUMNS if frozen_sidecar_contract is not None else ()
            ),
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
            map(str, AE_GMM_FEATURE_COLUMNS)
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
    mandatory = tuple(dict.fromkeys(map(str, mandatory_features)))
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
            map(str, AE_GMM_FEATURE_COLUMNS)
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
    effective_hpo_rows = min(int(hpo_rows), len(frame))
    effective_hpo_folds = min(int(hpo_folds), effective_folds)
    hpo_config = archetype.PathArchetypeConfig(
        **{**config.__dict__, "oof_folds": effective_hpo_folds}
    )
    hpo_positions, hpo_sampling_contract = _stratified_hpo_sample(
        frame,
        labels,
        sample_rows=effective_hpo_rows,
        validation_folds=effective_hpo_folds,
        timestamp_column=timestamp_column,
        label_end_column=label_end_column,
        side_column=side_column,
    )
    hpo_frame = frame.iloc[hpo_positions]
    hpo_labels = labels.iloc[hpo_positions]
    hpo_sampling_contract["purged_fold_support"] = _validate_hpo_sample_class_support(
        hpo_frame,
        hpo_labels,
        timestamp_column=timestamp_column,
        label_end_column=label_end_column,
        side_column=side_column,
        config=hpo_config,
    )
    selection_hpo_contract = _feature_selection_hpo_fingerprint(
        frame=frame,
        summaries=summaries,
        effective_labels=labels,
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
        taxonomy_contract=taxonomy_contract,
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
        _read_resumable_mda_progress(
            mda_progress_path, selection_fingerprint
        )
        if mda_progress_path.is_file()
        and not selection_checkpoint_path.is_file()
        and not force_feature_selection_hpo
        else None
    )
    full_contract_error: ValueError | None = None
    try:
        reused_checkpoint, reuse_provenance = _find_reusable_feature_selection_hpo_contract(
            selection_hpo_contract,
            output_dir=output_dir,
            checkpoint_path=feature_selection_hpo_checkpoint,
            registry_root=feature_selection_hpo_registry_root,
            force=force_feature_selection_hpo,
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
        labels,
        frame[timestamp_column],
        frame[label_end_column],
        config=config,
    )
    selector: archetype.FastSelectorResult | None = None
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
            frame.iloc[
                _beginning_middle_end_positions(len(frame), config.selector_sample_rows)
            ]
            if canonical_store_mode
            else frame
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
    elif resumed_selection_checkpoint is not None or reused_selection_checkpoint is not None:
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
            frame.iloc[
                _beginning_middle_end_positions(
                    len(frame), config.selector_sample_rows
                )
            ].copy()
            if canonical_store_mode else frame
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
            frame.iloc[
                _beginning_middle_end_positions(
                    len(frame), config.selector_sample_rows
                )
            ].copy()
            if canonical_store_mode else frame
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
                len(frame), config.selector_sample_rows
            )
            selection_frame = frame.iloc[sample_positions].copy()
            selection_labels = labels.iloc[sample_positions]
            split_at = np.flatnonzero(np.diff(sample_positions) > 1) + 1
            selection_parts: list[pd.DataFrame] = []
            selection_reads: list[dict[str, Any]] = []
            for block_positions in np.split(sample_positions, split_at):
                block_frame = frame.iloc[block_positions]
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
            selection_frame = frame
            selection_labels = labels
            selection_features = frame.loc[:, model_universe].copy()
        selector = archetype.fast_select_preentry_features(
            selection_features,
            selection_labels,
            mandatory_features=mandatory,
            config=config,
        )
        selected = list(selector.selected_features)

    if canonical_store_mode:
        if selection_labels is None:
            selection_labels = labels.loc[selection_frame.index]
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
            frame.loc[:, selected].copy()
            if (
                reused_checkpoint is not None
                or resumed_selection_checkpoint is not None
                or reused_selection_checkpoint is not None
                or resumed_mda_progress is not None
            )
            else selection_features
        )
        selection_labels = labels.loc[selection_frame.index]

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
            pre_permutation_oof = archetype.fit_purged_chronological_oof_catboost(
                mda_features,
                selection_labels,
                selection_frame[timestamp_column],
                label_end=selection_frame[label_end_column],
                config=config,
                params=selection_params,
                staged_matrix_cache=mda_cache,
                force_classes_count=False,
            )
            mda_completed_stages = (
                list(resumed_mda_progress["completed_stages"])
                if resumed_mda_progress is not None else []
            )

            def write_mda_progress(stage: Mapping[str, Any]) -> None:
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
                    "catboost_resource_contract": archetype.catboost_resource_contract(config),
                },
            )
        if canonical_store_mode:
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
            hpo_features = frame.iloc[hpo_positions].loc[:, selected]
        hpo = archetype.optimize_purged_catboost_hpo(
            hpo_features.loc[:, selected],
            hpo_labels,
            hpo_frame[timestamp_column],
            label_end=hpo_frame[label_end_column],
            config=hpo_config,
            n_trials=effective_trials,
            study_name=(
                "path_archetype_"
                f"{selection_hpo_contract['fingerprint'][:16]}"
            ),
            storage=f"sqlite:///{hpo_study_path.resolve()}",
            search_iterations=int(hpo_iterations),
            search_od_wait=int(hpo_od_wait),
            no_improvement_trials=int(hpo_no_improvement_trials),
            progress_path=hpo_progress_path,
        )
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
        }

    completed_selection_hpo_contract = {
        **selection_hpo_contract,
        "status": "feature_selection_hpo_complete",
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
        "hpo_sampling_contract": hpo_sampling_contract,
        "hpo_study_path": hpo_study_path,
        "hpo_progress_path": hpo_progress_path,
        "reuse_provenance": reuse_provenance,
        "catboost_resource_contract": archetype.catboost_resource_contract(config),
    }
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
            "catboost_resource_contract": archetype.catboost_resource_contract(config),
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
            "status": "stopped_after_feature_selection_and_hpo",
            "future_training_taxonomy": taxonomy_contract,
            "source": source,
            "rows": int(len(frame)),
            "candidate_identity_sha256": candidate_identity_sha256(
                frame,
                columns=(timestamp_column, IDENTITY_SYMBOL_COLUMN, side_column),
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
            "permutation_stage_metrics": permutation_stage_metrics,
            "permutation_acceleration_contract": (
                archetype.staged_permutation_acceleration_contract(config)
            ),
            "full_oof_and_final_refit_complete": False,
        }
        _write_json(output_dir / "run_manifest.json", checkpoint_manifest)
        return checkpoint_manifest
    if canonical_store_mode:
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
    oof = archetype.fit_purged_chronological_oof_catboost(
        features.loc[:, selected],
        labels,
        frame[timestamp_column],
        label_end=frame[label_end_column],
        config=config,
        params=params,
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
            "hpo_on_frozen_selected_features",
            "final_oof_and_refit",
        ],
        "hpo_feature_count": int(len(selected)),
        "future_training_taxonomy": taxonomy_contract,
        "effective_model_params": params,
        "catboost_resource_contract": archetype.catboost_resource_contract(config),
        "selector_backend": selector.proxy_backend
        if selector is not None
        else "reused_checkpoint",
        "pre_permutation_oof_diagnostics": (
            pre_permutation_oof.diagnostics if pre_permutation_oof is not None else None
        ),
        "oof_diagnostics": oof.diagnostics,
        "hpo": hpo_report,
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
        "future_training_taxonomy": taxonomy_contract,
        "source_artifact_sha256": _sha256_file(oof_path),
        "source_artifact": str(oof_path),
        "prediction_columns": prediction_columns,
        "identity_columns": [timestamp_column, "__symbol__", "side_name", "candidate_id"],
        "fold_provenance_columns": {
            "fold": "oof_fold_id",
            "validation_start": "validation_start",
            "latest_train_decision": "latest_train_decision_ts",
            "training_information_cutoff": "train_decision_cutoff",
            "latest_resolved_training_label": "label_resolution_available_at",
            "prediction_available_at": "available_at",
        },
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
            "hpo_feature_count": int(len(selected)),
            "hpo_features": list(selected),
            "no_improvement_patience_trials": int(
                hpo_no_improvement_trials
            ),
            "hpo": hpo_report,
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
            "pre_permutation_oof_diagnostics": (
                pre_permutation_oof.diagnostics
                if pre_permutation_oof is not None
                else None
            ),
            "oof_diagnostics": oof.diagnostics,
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
            "hpo_sampling_contract": hpo_sampling_contract,
            "permutation_stage_metrics": permutation_stage_metrics,
            "permutation_acceleration_contract": (
                archetype.staged_permutation_acceleration_contract(config)
            ),
        },
    )
    manifest = {
        "schema": RUNNER_SCHEMA,
        "source": source,
        "rows": int(len(frame)),
        "candidate_identity_sha256": candidate_identity_sha256(
            frame,
            columns=(timestamp_column, IDENTITY_SYMBOL_COLUMN, side_column),
        ),
        "candidate_population_contract": (
            "exact canonical OOF base top-fraction population; identity hash "
            "must match auxiliary and residual-alpha handoffs"
        ),
        "discovery_rows": int(discovery_mask.sum()),
        "discovery_end_exclusive": discovery_cutoff,
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
        "hpo_no_improvement_patience_trials": int(
            hpo_no_improvement_trials
        ),
        "feature_selection_hpo_fingerprint": selection_hpo_contract["fingerprint"],
        "feature_selection_fingerprint": selection_fingerprint,
        "feature_selection_hpo_reuse": reuse_provenance,
        "training_phase_order": [
            "fast_feature_selection",
            "permutation_feature_selection",
            "hpo_on_frozen_selected_features",
            "final_oof_and_refit",
        ],
        "catboost_resource_contract": archetype.catboost_resource_contract(config),
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
    parser.add_argument(
        "--feature-dir",
        required=True,
        type=Path,
        help="Timestamped shared static feature store used for all classifier inputs",
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
        default=30,
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
        feature_dir=args.feature_dir,
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
    )
    print(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
