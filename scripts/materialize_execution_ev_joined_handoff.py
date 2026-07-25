#!/usr/bin/env python3
"""Materialize the leakage-safe execution-EV direct/residual joined handoff.

The input artifacts remain deliberately independent while upstream jobs are
running.  This adapter standardizes their decision keys and joins only exact,
one-to-one OOF rows.  It is a data-contract builder, not a trainer: the only
realized columns it emits are the 12-hour execution target and its label end.

Example (override the column names to match the finished upstream artifacts)::

  python scripts/materialize_execution_ev_joined_handoff.py \
    --alpha alpha_oof.parquet --time-oof time_oof.parquet \
    --peak-oof peak_oof.parquet --mae-oof mae_oof.parquet \
    --turn-oof turn_oof.parquet --slope-oof slope_oof.parquet \
    --catboost-oof path_oof.parquet \
    --execution-labels execution_labels.parquet --output joined.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.execution_ev_meta import (  # noqa: E402
    catboost_class_order_sha256,
)
from extreme_price_movements.path_archetype_support import (  # noqa: E402
    MERGED_PATH_ARCHETYPE_CLASSES,
)

HANDOFF_SCHEMA = "execution_ev_joined_handoff_v2"
BASE_JOIN_KEYS = ("__ts__", "__symbol__", "side_name")
DEFAULT_CANDIDATE_ID_COLUMN = "candidate_id"
JOIN_KEYS = (*BASE_JOIN_KEYS, DEFAULT_CANDIDATE_ID_COLUMN)
MIN_COMMON_INTERSECTION_ROWS = 2
LEGACY_ALPHA_COST_RETURN = 0.01
BASE_ARCHETYPE_FEATURE_PREFIX = "base_archetype_label__"
FORBIDDEN_FEATURE_TOKENS = (
    "realized",
    "future_",
    "label",
    "target",
    "outcome",
    "y_exec",
    "ev_after",
    "ret_net",
    "exec_margin",
    "actual_",
    "execution_",
    "gross_ev",
    "net_ev",
)
CATBOOST_ADVERSE_CLASSES = (
    "immediate_adverse_path",
    "early_mfe_full_reversal",
    "dead_timeout",
)
CATBOOST_FAVORABLE_CLASSES = (
    "fast_realization_winner",
    "late_breakout",
    "slow_grinder",
)
TIMING_CDF_PREDICTION_COLUMNS = (
    "prediction_p_hit_by_2h",
    "prediction_p_hit_by_4h",
    "prediction_p_hit_by_8h",
    "prediction_p_hit_by_12h",
)
TIMING_CDF_JOINED_FEATURE_COLUMNS = {
    source_column: f"pred_time_to_meaningful_mfe_p_hit_by_{hours}h"
    for source_column, hours in zip(TIMING_CDF_PREDICTION_COLUMNS, (2, 4, 8, 12))
}


@dataclass(frozen=True)
class SourceSpec:
    name: str
    path: Path
    timestamp_col: str
    symbol_col: str
    side_col: str
    candidate_id_col: str
    available_at_col: str
    fold_col: str | None
    validation_start_col: str | None
    train_decision_cutoff_col: str | None
    label_resolution_available_at_col: str
    manifest_path: Path
    prediction_role: str
    output_columns: Mapping[str, str]
    require_oof_fold: bool


@dataclass
class LoadedSource:
    spec: SourceSpec
    frame: pd.DataFrame
    metadata: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(dict(payload)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _canonical_json_hash(
    payload: Mapping[str, Any], *, excluded: Sequence[str] = ()
) -> str:
    canonical = {
        str(key): _json_safe(value)
        for key, value in payload.items()
        if key not in set(excluded)
    }
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _load_signed_prediction_manifest(
    spec: SourceSpec, *, verify_artifact_hash: bool = True
) -> dict[str, Any]:
    if not spec.manifest_path.is_file():
        raise ValueError(f"{spec.name}: signed prediction-role manifest is required")
    try:
        payload = json.loads(spec.manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"{spec.name}: cannot read signed prediction-role manifest"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"{spec.name}: signed prediction-role manifest must be a JSON object"
        )
    signed_hash = payload.get("prediction_role_manifest_sha256")
    if not isinstance(signed_hash, str) or not signed_hash:
        raise ValueError(f"{spec.name}: missing signed prediction-role manifest hash")
    expected_hash = _canonical_json_hash(
        payload, excluded=("prediction_role_manifest_sha256",)
    )
    if not hmac.compare_digest(signed_hash, expected_hash):
        raise ValueError(
            f"{spec.name}: signed prediction-role manifest hash does not verify"
        )
    if payload.get("prediction_role") != spec.prediction_role:
        raise ValueError(
            f"{spec.name}: prediction role must be {spec.prediction_role!r} in its signed manifest"
        )
    artifact_hash = payload.get("source_artifact_sha256", payload.get("output_sha256"))
    if verify_artifact_hash and (
        not isinstance(artifact_hash, str)
        or not hmac.compare_digest(artifact_hash, _sha256(spec.path))
    ):
        raise ValueError(
            f"{spec.name}: signed manifest does not bind this parquet artifact hash"
        )
    class_order: tuple[str, ...] | None = None
    class_order_source: str | None = None
    declared_orders: list[tuple[str, ...]] = []
    declared_sources: list[str] = []
    for key in ("path_shape_types", "class_names"):
        value = payload.get(key)
        if value is None:
            continue
        if not isinstance(value, list):
            raise ValueError(f"{spec.name}: {key!r} must be an ordered class list")
        order = tuple(str(name).strip() for name in value)
        catboost_class_order_sha256(order)
        declared_orders.append(order)
        declared_sources.append(key)
    probability_classes: tuple[str, ...] = ()
    if spec.name == "catboost":
        prediction_columns = payload.get("prediction_columns")
        if prediction_columns is not None and not isinstance(
            prediction_columns, Mapping
        ):
            raise ValueError("catboost: prediction_columns must be a mapping")
        probability_columns = (
            [
                str(name)
                for name in prediction_columns
                if str(name).startswith("probability__")
            ]
            if isinstance(prediction_columns, Mapping)
            else []
        )
        if probability_columns:
            probability_classes = tuple(
                name.removeprefix("probability__") for name in probability_columns
            )
    if declared_orders:
        if len(set(declared_orders)) != 1:
            raise ValueError(f"{spec.name}: signed manifest class orders disagree")
        class_order = declared_orders[0]
        class_order_source = "+".join(declared_sources)
        if probability_classes and set(probability_classes) != set(class_order):
            raise ValueError(
                "catboost: signed probability columns do not match the declared class order"
            )
    elif probability_classes:
        if set(probability_classes) == set(MERGED_PATH_ARCHETYPE_CLASSES):
            class_order = tuple(MERGED_PATH_ARCHETYPE_CLASSES)
            class_order_source = "merged_path_archetype_classes_from_probability_names"
        else:
            # Object-key order is not preserved by the signed canonical JSON
            # hash, so a promoted taxonomy without an explicit list uses a
            # deterministic order derived from its signed class names.
            class_order = tuple(sorted(probability_classes))
            class_order_source = "canonical_sorted_probability_names"
    if spec.name == "catboost" and class_order is None:
        class_order = tuple(MERGED_PATH_ARCHETYPE_CLASSES)
        class_order_source = "default_merged_path_archetype_classes"
    return {
        "path": str(spec.manifest_path.resolve()),
        "sha256": _sha256(spec.manifest_path),
        "signed_prediction_role_manifest_sha256": signed_hash,
        "prediction_role": spec.prediction_role,
        "prediction_columns": payload.get("prediction_columns"),
        "class_contract": (
            {
                "class_order": list(class_order),
                "class_order_sha256": catboost_class_order_sha256(class_order),
                "source": class_order_source,
            }
            if class_order is not None
            else None
        ),
        "alpha_cost_basis": (
            _alpha_cost_basis(payload) if spec.name == "alpha" else None
        ),
    }


def _alpha_cost_basis(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a signed alpha cost basis or fail closed.

    Execution-EV reconciliation is valid only when the alpha source declares
    the cost already included in its OOF EV.  Legacy alpha adapters publish
    this exact block after proving their residual target against the source
    model manifest.
    """

    raw = payload.get("alpha_cost_basis", payload.get("cost_basis"))
    if not isinstance(raw, Mapping):
        raise ValueError("alpha: signed manifest requires an explicit alpha_cost_basis")
    amount_raw = raw.get("deducted_cost_return", raw.get("source_cost_return"))
    try:
        amount = float(amount_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("alpha: alpha_cost_basis has no finite deducted cost") from exc
    if not np.isfinite(amount) or not np.isclose(
        amount, LEGACY_ALPHA_COST_RETURN, atol=1e-12, rtol=0.0
    ):
        raise ValueError(
            "alpha: alpha_cost_basis must prove exactly a 0.01 deducted return"
        )
    if str(raw.get("cost_unit", "")).strip().lower() != "return":
        raise ValueError("alpha: alpha_cost_basis cost_unit must be 'return'")
    if str(raw.get("target_semantics", "")).strip().lower() not in {
        "residual_net_ev_after_1pct",
        "ev_after_1pct",
    }:
        raise ValueError(
            "alpha: alpha_cost_basis must declare residual_net_ev_after_1pct semantics"
        )
    source_manifest = raw.get("source_manifest")
    evidence = raw.get("source_manifest_evidence")
    if (
        not isinstance(source_manifest, Mapping)
        or not isinstance(source_manifest.get("sha256"), str)
        or not source_manifest["sha256"]
        or not isinstance(evidence, list)
        or not evidence
    ):
        raise ValueError(
            "alpha: alpha_cost_basis requires source-manifest hash and target evidence"
        )
    for item in evidence:
        if not isinstance(item, Mapping):
            raise ValueError("alpha: alpha_cost_basis target evidence is malformed")
        field = item.get("field")
        value = item.get("value")
        if not isinstance(field, str) or not isinstance(value, str):
            raise ValueError("alpha: alpha_cost_basis target evidence is malformed")
        normalized_field = field.strip().lower()
        normalized_value = value.strip().lower()
        if normalized_field in {"target_mode", "params.target_mode"}:
            if normalized_value not in {
                "residual_net_ev_after_1pct",
                "ev_after_1pct",
            }:
                raise ValueError(
                    "alpha: alpha_cost_basis target-mode evidence is inconsistent"
                )
        elif normalized_field == "residual_expert_target":
            if "ev_after_1pct" not in normalized_value or not any(
                token in normalized_value for token in ("residual", "-", "minus")
            ):
                raise ValueError(
                    "alpha: alpha_cost_basis residual-target evidence is inconsistent"
                )
        else:
            raise ValueError(
                "alpha: alpha_cost_basis target evidence has an unsupported source field"
            )
    return {
        "deducted_cost_return": amount,
        "cost_unit": "return",
        "target_semantics": "residual_net_ev_after_1pct",
        "source_manifest": dict(source_manifest),
        "source_manifest_evidence": [dict(item) for item in evidence],
    }


def _assert_alpha_ev_manifest_binding(
    spec: SourceSpec, manifest: Mapping[str, Any]
) -> None:
    alpha_input_columns = [
        input_column
        for input_column, output_column in spec.output_columns.items()
        if output_column == "existing_alpha_ev"
    ]
    if len(alpha_input_columns) != 1:
        raise ValueError(
            "alpha: exactly one signed alpha EV prediction column is required"
        )
    source_column = alpha_input_columns[0]
    declared = manifest.get("prediction_columns")
    if not isinstance(declared, Mapping):
        raise ValueError("alpha: signed manifest must declare prediction columns")
    record = declared.get(source_column)
    if not isinstance(record, Mapping):
        raise ValueError(
            f"alpha: signed manifest has no prediction-role declaration for {source_column!r}"
        )
    if (
        record.get("role") != "pre_entry_alpha_ev_oof_prediction"
        or record.get("target") is not False
    ):
        raise ValueError(
            "alpha: existing_alpha_ev must be bound to a signed pre-entry alpha OOF prediction role"
        )


def _timing_cdf_output_columns(manifest: Mapping[str, Any]) -> dict[str, str]:
    """Return signed timing-CDF inputs, retaining legacy scalar-only artifacts."""

    declared = manifest.get("prediction_columns")
    if declared is None:
        return {}
    if not isinstance(declared, Mapping):
        raise ValueError("time_to_mfe: prediction_columns must be a mapping")
    cdf_like_columns = {
        str(column)
        for column in declared
        if str(column).startswith("prediction_p_hit_by_")
    }
    if not cdf_like_columns:
        return {}
    if cdf_like_columns != set(TIMING_CDF_PREDICTION_COLUMNS):
        raise ValueError(
            "time_to_mfe: signed timing CDF vector must declare exactly "
            "p_hit_by_2h, p_hit_by_4h, p_hit_by_8h, and p_hit_by_12h"
        )
    for output_column in TIMING_CDF_PREDICTION_COLUMNS:
        record = declared.get(output_column)
        expected_source = f"pred_{output_column.removeprefix('prediction_')}"
        if not isinstance(record, Mapping) or (
            record.get("role") != "pre_entry_auxiliary_timing_cdf_probability_oof"
            or record.get("target") is not False
            or record.get("head") != "timing"
            or record.get("source_prediction_column") != expected_source
        ):
            raise ValueError(
                f"time_to_mfe: signed timing CDF declaration for {output_column!r} "
                "is not a target-free timing OOF prediction"
            )
    return dict(TIMING_CDF_JOINED_FEATURE_COLUMNS)


def _parse_columns(value: str) -> list[str]:
    columns = [item.strip() for item in value.split(",") if item.strip()]
    if not columns:
        raise argparse.ArgumentTypeError("at least one column is required")
    if len(columns) != len(set(columns)):
        raise argparse.ArgumentTypeError("column names must be unique")
    return columns


def _none_if_blank(value: str | None) -> str | None:
    value = (value or "").strip()
    return value or None


def _utc(values: pd.Series, *, source: str, column: str) -> pd.Series:
    converted = pd.to_datetime(values, utc=True, errors="coerce")
    if converted.isna().any():
        raise ValueError(f"{source}: {column!r} contains null or invalid timestamps")
    return converted


def _canonical_side(values: pd.Series, *, source: str, column: str) -> pd.Series:
    sides = values.astype("string").str.strip().str.lower()
    if sides.isna().any() or not sides.isin(["long", "short"]).all():
        invalid = sorted(set(sides.dropna()) - {"long", "short"})
        raise ValueError(
            f"{source}: {column!r} must contain canonical long/short sides; "
            f"invalid={invalid[:5]!r}"
        )
    return sides.astype(str)


def _nonempty_strings(values: pd.Series, *, source: str, column: str) -> pd.Series:
    result = values.astype("string").str.strip()
    if result.isna().any() or (result == "").any():
        raise ValueError(f"{source}: {column!r} contains null or blank identity values")
    return result.astype(str)


def _assert_unique(frame: pd.DataFrame, keys: Sequence[str], *, source: str) -> None:
    duplicate_count = int(frame.duplicated(list(keys), keep=False).sum())
    if duplicate_count:
        raise ValueError(
            f"{source}: duplicate rows violate exact one-to-one identity on {list(keys)!r}; "
            f"duplicate_rows={duplicate_count}"
        )


def _assert_feature_name_safe(column: str, *, source: str, role: str) -> None:
    name = str(column).lower()
    if any(token in name for token in FORBIDDEN_FEATURE_TOKENS):
        raise ValueError(
            f"{source}: target leakage in feature {column!r} for {role}; "
            "join a pre-entry OOF/frozen prediction instead"
        )


def _assert_auxiliary_prediction_name(column: str, *, source: str, role: str) -> None:
    _assert_feature_name_safe(column, source=source, role=role)
    name = str(column).lower()
    if not any(token in name for token in ("pred", "oof", "frozen", "score")):
        raise ValueError(
            f"{source}: {role} column {column!r} is not explicitly a prediction/OOF output; "
            "refuse to use a raw future-path target as a feature"
        )


def _source_key_value(args: argparse.Namespace, source: str, key: str) -> str | None:
    override = _none_if_blank(getattr(args, f"{source}_{key}_col", None))
    return (
        override
        if override is not None
        else _none_if_blank(getattr(args, f"{key}_col"))
    )


def _load_source(spec: SourceSpec) -> LoadedSource:
    if not spec.path.is_file() or spec.path.suffix.lower() not in {".parquet", ".pq"}:
        raise ValueError(
            f"{spec.name}: expected an existing parquet source, got {spec.path}"
        )
    manifest = _load_signed_prediction_manifest(spec, verify_artifact_hash=False)
    output_columns = dict(spec.output_columns)
    timing_cdf_output_columns: dict[str, str] = {}
    if spec.name == "time_to_mfe":
        timing_cdf_output_columns = _timing_cdf_output_columns(manifest)
        output_columns.update(timing_cdf_output_columns)
    raw = pd.read_parquet(spec.path)
    if raw.empty:
        raise ValueError(f"{spec.name}: source parquet is empty")
    required = {
        spec.timestamp_col,
        spec.symbol_col,
        spec.side_col,
        spec.candidate_id_col,
        spec.available_at_col,
        spec.label_resolution_available_at_col,
        *output_columns.keys(),
    }
    if spec.require_oof_fold:
        if not all(
            (spec.fold_col, spec.validation_start_col, spec.train_decision_cutoff_col)
        ):
            raise ValueError(
                f"{spec.name}: OOF fold, validation start, and train decision cutoff columns are required"
            )
        required.update(
            (spec.fold_col, spec.validation_start_col, spec.train_decision_cutoff_col)
        )
    missing = sorted(column for column in required if column not in raw.columns)
    if missing:
        provenance_columns = {
            spec.candidate_id_col,
            spec.available_at_col,
            spec.label_resolution_available_at_col,
            spec.fold_col,
            spec.validation_start_col,
            spec.train_decision_cutoff_col,
        }
        absent_evidence = sorted(
            column for column in missing if column in provenance_columns
        )
        if absent_evidence:
            raise ValueError(
                f"{spec.name}: strict joined execution-EV provenance is unavailable; "
                f"missing evidence columns: {', '.join(absent_evidence)}. Regenerate the "
                "upstream OOF artifact from resolved training rows; do not infer these values "
                "from a validation boundary."
            )
        raise ValueError(f"{spec.name}: missing required columns: {', '.join(missing)}")

    work = pd.DataFrame(
        {
            "__ts__": _utc(
                raw[spec.timestamp_col], source=spec.name, column=spec.timestamp_col
            ),
            "__symbol__": _nonempty_strings(
                raw[spec.symbol_col], source=spec.name, column=spec.symbol_col
            ),
            "side_name": _canonical_side(
                raw[spec.side_col], source=spec.name, column=spec.side_col
            ),
        }
    )
    work["candidate_id"] = _nonempty_strings(
        raw[spec.candidate_id_col], source=spec.name, column=spec.candidate_id_col
    )
    for input_column, output_column in output_columns.items():
        if output_column in work.columns:
            raise ValueError(
                f"{spec.name}: output column conflicts with identity: {output_column!r}"
            )
        work[output_column] = raw[input_column].to_numpy()
    availability_output = f"{spec.name}_available_at"
    work[availability_output] = _utc(
        raw[spec.available_at_col], source=spec.name, column=spec.available_at_col
    )
    if spec.require_oof_fold and (work[availability_output] > work["__ts__"]).any():
        raise ValueError(
            f"{spec.name}: feature availability is after the decision timestamp"
        )
    resolution_output = f"{spec.name}_label_resolution_available_at"
    work[resolution_output] = _utc(
        raw[spec.label_resolution_available_at_col],
        source=spec.name,
        column=spec.label_resolution_available_at_col,
    )

    fold_output: str | None = None
    fold_ids: list[str] | None = None
    if spec.require_oof_fold:
        assert (
            spec.fold_col
            and spec.validation_start_col
            and spec.train_decision_cutoff_col
        )
        try:
            folds = _nonempty_strings(
                raw[spec.fold_col], source=spec.name, column=spec.fold_col
            )
        except ValueError as exc:
            raise ValueError(
                f"{spec.name}: missing or invalid required OOF fold IDs in {spec.fold_col!r}"
            ) from exc
        fold_output = f"{spec.name}_oof_fold"
        work[fold_output] = folds
        validation_output = f"{spec.name}_validation_start"
        cutoff_output = f"{spec.name}_train_decision_cutoff"
        work[validation_output] = _utc(
            raw[spec.validation_start_col],
            source=spec.name,
            column=spec.validation_start_col,
        )
        work[cutoff_output] = _utc(
            raw[spec.train_decision_cutoff_col],
            source=spec.name,
            column=spec.train_decision_cutoff_col,
        )
        if not (work[cutoff_output] < work[validation_output]).all():
            raise ValueError(
                f"{spec.name}: train decision cutoff must be strictly before validation start"
            )
        if not (work[validation_output] <= work["__ts__"]).all():
            raise ValueError(
                f"{spec.name}: validation start is after an OOF decision timestamp"
            )
        if not (work[cutoff_output] < work["__ts__"]).all():
            raise ValueError(
                f"{spec.name}: train decision cutoff must be strictly before decision"
            )
        if not (work[resolution_output] <= work[cutoff_output]).all():
            raise ValueError(
                f"{spec.name}: training labels must resolve before train decision cutoff availability"
            )
        fold_ids = sorted(str(value) for value in pd.unique(work[fold_output]))
    elif not (work["execution_label_end_utc"] <= work[resolution_output]).all():
        raise ValueError(
            f"{spec.name}: execution label end must be resolved before its label availability timestamp"
        )

    identity = list(JOIN_KEYS)
    _assert_unique(work, identity, source=spec.name)
    manifest = _load_signed_prediction_manifest(spec)
    if spec.name == "alpha":
        _assert_alpha_ev_manifest_binding(spec, manifest)
    metadata: dict[str, Any] = {
        "path": str(spec.path.resolve()),
        "sha256": _sha256(spec.path),
        "input_rows": int(len(raw)),
        "input_identity_columns": {
            "timestamp": spec.timestamp_col,
            "symbol": spec.symbol_col,
            "side": spec.side_col,
            "candidate_id": spec.candidate_id_col,
        },
        "normalized_identity_columns": identity,
        "selected_columns": output_columns,
        "availability": {
            "input_column": spec.available_at_col,
            "materialized_column": availability_output,
            "rule": "source availability <= decision timestamp",
        },
        "label_resolution": {
            "input_column": spec.label_resolution_available_at_col,
            "materialized_column": resolution_output,
            "rule": (
                "training-label resolution <= train decision cutoff < validation decision"
                if spec.require_oof_fold
                else "execution label end <= execution label resolution availability"
            ),
        },
        "signed_prediction_role_manifest": manifest,
    }
    if spec.name == "time_to_mfe":
        metadata["timing_cdf_vector"] = {
            "status": (
                "signed_complete_timing_oof_vector"
                if timing_cdf_output_columns
                else "legacy_scalar_only_no_timing_cdf_vector_declared"
            ),
            "source_columns": list(timing_cdf_output_columns),
            "joined_feature_columns": list(timing_cdf_output_columns.values()),
            "contract": (
                "all four p(hit by horizon) probabilities are target-free, signed timing OOF predictions"
                if timing_cdf_output_columns
                else "no timing CDF vector declared; retain the scalar compatibility input only"
            ),
        }
    if spec.require_oof_fold:
        metadata["oof"] = {
            "source_fold_column": spec.fold_col,
            "materialized_fold_column": fold_output,
            "fold_ids": fold_ids,
            "fold_count": len(fold_ids or []),
        }
    return LoadedSource(spec=spec, frame=work, metadata=metadata)


def _common_identity_hash(frame: pd.DataFrame) -> str:
    records = [
        [
            pd.Timestamp(row["__ts__"]).isoformat(),
            str(row["__symbol__"]),
            str(row["side_name"]),
            str(row["candidate_id"]),
        ]
        for _, row in frame.loc[:, list(JOIN_KEYS)].iterrows()
    ]
    encoded = json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _retain_common_intersection(
    sources: Sequence[LoadedSource], *, min_rows: int
) -> dict[str, Any]:
    """Restrict independent OOF artifacts to their explicit common identity set."""

    if min_rows < MIN_COMMON_INTERSECTION_ROWS:
        raise ValueError(
            f"min_common_rows must be at least {MIN_COMMON_INTERSECTION_ROWS}"
        )
    if not sources:  # pragma: no cover - source specs are fixed by the parser.
        raise ValueError("at least one joined source is required")
    keys = list(JOIN_KEYS)
    alpha_keys = sources[0].frame.loc[:, keys].copy()
    common = alpha_keys.copy()
    for source in sources[1:]:
        source_keys = source.frame.loc[:, keys]
        common = common.merge(
            source_keys, on=keys, how="inner", validate="one_to_one", sort=False
        )
    common = common.sort_values(keys, kind="stable").reset_index(drop=True)
    if len(common) < min_rows:
        raise ValueError(
            "common OOF identity intersection is too small; "
            f"retained_rows={len(common)}, min_common_rows={min_rows}"
        )

    source_reports: dict[str, dict[str, int]] = {}
    for source in sources:
        source_keys = source.frame.loc[:, keys]
        alpha_coverage = alpha_keys.merge(
            source_keys,
            on=keys,
            how="outer",
            indicator=True,
            validate="one_to_one",
            sort=False,
        )
        retained = source.frame.merge(
            common, on=keys, how="inner", validate="one_to_one", sort=False
        )
        dropped = int(len(source.frame) - len(retained))
        report = {
            "input_rows": int(len(source.frame)),
            "retained_common_rows": int(len(retained)),
            "dropped_not_in_common_rows": dropped,
            "missing_vs_alpha_rows": int(
                alpha_coverage["_merge"].eq("left_only").sum()
            ),
            "source_only_vs_alpha_rows": int(
                alpha_coverage["_merge"].eq("right_only").sum()
            ),
        }
        source.metadata["common_intersection"] = report
        source.frame = retained
        source_reports[source.spec.name] = report
    return {
        "mode": "explicit_common_identity_intersection_one_to_one",
        "keys": keys,
        "minimum_required_rows": int(min_rows),
        "common_rows": int(len(common)),
        "common_identity_sha256": _common_identity_hash(common),
        "sources": source_reports,
    }


def _join(base: pd.DataFrame, source: LoadedSource) -> pd.DataFrame:
    keys = list(JOIN_KEYS)
    _assert_unique(
        base, keys, source="retained joined handoff before " + source.spec.name
    )
    _assert_unique(source.frame, keys, source=source.spec.name)
    source_columns = [
        *keys,
        *[column for column in source.frame.columns if column not in keys],
    ]
    joined = base.merge(
        source.frame.loc[:, source_columns],
        on=keys,
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    if joined.empty:
        raise ValueError(
            f"{source.spec.name}: non-overlapping availability; exact inner join on {keys!r} produced zero rows"
        )
    return joined


def _numeric_features(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(
                f"joined handoff has non-finite required feature {column!r}"
            )
        frame[column] = values.astype("float64")


def _source_specs(args: argparse.Namespace) -> list[SourceSpec]:
    key = lambda source, name: _source_key_value(args, source, name)
    candidate = lambda source: key(source, "candidate_id")

    def required(value: str | None, *, source: str, field: str) -> str:
        if value is None:
            raise ValueError(f"{source}: {field} column is required")
        return value

    def spec(
        name: str,
        path: Path,
        *,
        source_arg: str,
        output_columns: Mapping[str, str],
        require_oof_fold: bool,
        prediction_role: str,
    ) -> SourceSpec:
        manifest_path = getattr(args, f"{source_arg}_manifest", None)
        if not isinstance(manifest_path, Path):
            raise ValueError(
                f"{name}: signed prediction-role manifest path is required"
            )
        return SourceSpec(
            name=name,
            path=path,
            timestamp_col=required(
                key(source_arg, "timestamp"), source=name, field="timestamp"
            ),
            symbol_col=required(key(source_arg, "symbol"), source=name, field="symbol"),
            side_col=required(key(source_arg, "side"), source=name, field="side"),
            candidate_id_col=required(
                candidate(source_arg), source=name, field="candidate_id"
            ),
            available_at_col=required(
                _none_if_blank(getattr(args, f"{source_arg}_available_at_col", None)),
                source=name,
                field="availability",
            ),
            fold_col=(
                required(
                    _none_if_blank(getattr(args, f"{source_arg}_fold_col", None)),
                    source=name,
                    field="OOF fold",
                )
                if require_oof_fold
                else None
            ),
            validation_start_col=(
                required(
                    _none_if_blank(
                        getattr(args, f"{source_arg}_validation_start_col", None)
                    ),
                    source=name,
                    field="validation_start",
                )
                if require_oof_fold
                else None
            ),
            train_decision_cutoff_col=(
                required(
                    _none_if_blank(
                        getattr(args, f"{source_arg}_train_decision_cutoff_col", None)
                    ),
                    source=name,
                    field="train_decision_cutoff",
                )
                if require_oof_fold
                else None
            ),
            label_resolution_available_at_col=required(
                _none_if_blank(
                    getattr(
                        args, f"{source_arg}_label_resolution_available_at_col", None
                    )
                ),
                source=name,
                field="label resolution availability",
            ),
            manifest_path=manifest_path,
            prediction_role=prediction_role,
            output_columns=output_columns,
            require_oof_fold=require_oof_fold,
        )

    alpha_columns = {
        args.alpha_ev_col: "existing_alpha_ev",
        args.alpha_uncertainty_col: "alpha_prediction_uncertainty",
        args.alpha_leaf_support_col: "alpha_leaf_support",
    }
    if len(alpha_columns) != 3:
        raise ValueError(
            "alpha EV, uncertainty, and leaf-support columns must be distinct"
        )
    alpha_archetype_columns = sorted(
        column
        for column in pq.read_schema(args.alpha).names
        if column.startswith(BASE_ARCHETYPE_FEATURE_PREFIX)
    )
    if not alpha_archetype_columns:
        raise ValueError(
            "alpha OOF must contain frozen non-CatBoost base archetype-label features"
        )
    alpha_columns.update({column: column for column in alpha_archetype_columns})
    for column, role in alpha_columns.items():
        if not column.startswith(BASE_ARCHETYPE_FEATURE_PREFIX):
            _assert_feature_name_safe(column, source="alpha", role=role)
    _assert_auxiliary_prediction_name(
        args.time_prediction_col, source="time_oof", role="time_to_mfe"
    )
    _assert_auxiliary_prediction_name(
        args.peak_prediction_col, source="peak_oof", role="peak_mfe"
    )
    _assert_auxiliary_prediction_name(
        args.mae_prediction_col, source="mae_oof", role="mae_before_mfe"
    )
    _assert_auxiliary_prediction_name(
        args.turn_prediction_col, source="turn_oof", role="adverse_turn_bars"
    )
    _assert_auxiliary_prediction_name(
        args.slope_prediction_col, source="slope_oof", role="favorable_path_slope"
    )
    _assert_auxiliary_prediction_name(
        args.catboost_archetype_col,
        source="catboost_oof",
        role="predicted path archetype",
    )
    for column in [
        *args.catboost_prob_cols,
        args.catboost_entropy_col,
        args.catboost_max_probability_col,
        args.catboost_normalized_entropy_col,
        args.catboost_top2_margin_col,
        args.catboost_adverse_mass_col,
        args.catboost_favorable_mass_col,
    ]:
        _assert_feature_name_safe(
            column, source="catboost_oof", role="CatBoost prediction"
        )
    return [
        spec(
            "alpha",
            args.alpha,
            source_arg="alpha",
            output_columns=alpha_columns,
            require_oof_fold=True,
            prediction_role="alpha_ev_oof",
        ),
        spec(
            "time_to_mfe",
            args.time_oof,
            source_arg="time",
            output_columns={
                args.time_prediction_col: "pred_time_to_first_meaningful_MFE"
            },
            require_oof_fold=True,
            prediction_role="time_to_mfe_oof",
        ),
        spec(
            "peak_mfe",
            args.peak_oof,
            source_arg="peak",
            output_columns={args.peak_prediction_col: "pred_peak_MFE_12h_ATR"},
            require_oof_fold=True,
            prediction_role="peak_mfe_oof",
        ),
        spec(
            "mae_before_mfe",
            args.mae_oof,
            source_arg="mae",
            output_columns={
                args.mae_prediction_col: "pred_mae_before_meaningful_mfe_atr"
            },
            require_oof_fold=True,
            prediction_role="mae_before_mfe_oof",
        ),
        spec(
            "adverse_turn",
            args.turn_oof,
            source_arg="turn",
            output_columns={
                args.turn_prediction_col: "pred_bars_before_price_stops_decreasing"
            },
            require_oof_fold=True,
            prediction_role="adverse_turn_oof",
        ),
        spec(
            "path_slope",
            args.slope_oof,
            source_arg="slope",
            output_columns={
                args.slope_prediction_col: "pred_favorable_path_slope_atr_per_hour"
            },
            require_oof_fold=True,
            prediction_role="path_slope_oof",
        ),
        spec(
            "catboost",
            args.catboost_oof,
            source_arg="catboost",
            output_columns={
                **{
                    column: f"catboost_p_{index}"
                    for index, column in enumerate(args.catboost_prob_cols)
                },
                args.catboost_entropy_col: "catboost_entropy",
                args.catboost_max_probability_col: "catboost_max_probability",
                args.catboost_normalized_entropy_col: "catboost_normalized_entropy",
                args.catboost_top2_margin_col: "catboost_top2_probability_margin",
                args.catboost_adverse_mass_col: "catboost_adverse_probability_mass",
                args.catboost_favorable_mass_col: "catboost_favorable_probability_mass",
                args.catboost_archetype_col: "catboost_archetype",
            },
            require_oof_fold=True,
            prediction_role="path_archetype_oof",
        ),
        spec(
            "execution_labels",
            args.execution_labels,
            source_arg="labels",
            output_columns={
                args.execution_decision_ts_col: "execution_decision_utc",
                args.execution_gross_ev_col: "execution_gross_ev_12h",
                args.execution_cost_return_col: "execution_cost_return",
                args.execution_net_ev_col: "execution_net_ev_12h",
                args.execution_label_end_col: "execution_label_end_utc",
                args.execution_exit_reason_col: "execution_exit_reason",
                args.execution_exit_hour_col: "execution_exit_hour",
                args.execution_mfe_col: "execution_mfe_return_12h",
                args.execution_mae_col: "execution_mae_return_12h",
            },
            require_oof_fold=False,
            prediction_role="execution_ev_12h_labels",
        ),
    ]


def _provenance(
    joined: pd.DataFrame,
    sources: Sequence[LoadedSource],
    *,
    join_keys: Sequence[str],
    catboost_class_contract: Mapping[str, Any],
    population_alignment: Mapping[str, Any],
    alpha_cost_basis: Mapping[str, Any],
) -> dict[str, Any]:
    availability = {
        source.spec.name: f"{source.spec.name}_available_at"
        for source in sources
        if source.spec.name != "execution_labels"
    }
    features: dict[str, dict[str, Any]] = {
        "existing_alpha_ev": {
            "family": "alpha_score",
            "source": "alpha OOF/candidate stream",
            "pre_entry": True,
            "oof_or_frozen": True,
            "available_at_col": availability["alpha"],
            "model_input": True,
        },
        "alpha_prediction_uncertainty": {
            "family": "prediction_uncertainty",
            "source": "alpha OOF/candidate stream",
            "pre_entry": True,
            "oof_or_frozen": True,
            "available_at_col": availability["alpha"],
            "model_input": True,
        },
        "alpha_leaf_support": {
            "family": "leaf_support",
            "source": "alpha OOF/candidate stream",
            "pre_entry": True,
            "oof_or_frozen": True,
            "available_at_col": availability["alpha"],
            "model_input": True,
        },
        "pred_time_to_first_meaningful_MFE": {
            "family": "time_to_mfe",
            "source": "auxiliary LGBM OOF time-to-first-meaningful-MFE head",
            "pre_entry": True,
            "oof_or_frozen": True,
            "available_at_col": availability["time_to_mfe"],
            "model_input": True,
        },
        "pred_peak_MFE_12h_ATR": {
            "family": "peak_mfe",
            "source": "auxiliary LGBM OOF peak-MFE-12h-ATR head",
            "pre_entry": True,
            "oof_or_frozen": True,
            "available_at_col": availability["peak_mfe"],
            "model_input": True,
        },
        "pred_mae_before_meaningful_mfe_atr": {
            "family": "mae_before_meaningful_mfe",
            "source": "auxiliary LGBM OOF adverse-depth-before-meaningful-MFE head",
            "pre_entry": True,
            "oof_or_frozen": True,
            "available_at_col": availability["mae_before_mfe"],
            "model_input": True,
        },
        "pred_bars_before_price_stops_decreasing": {
            "family": "adverse_turn_timing",
            "source": "auxiliary LGBM OOF adverse-turn timing head",
            "pre_entry": True,
            "oof_or_frozen": True,
            "available_at_col": availability["adverse_turn"],
            "model_input": True,
        },
        "pred_favorable_path_slope_atr_per_hour": {
            "family": "favorable_path_slope",
            "source": "auxiliary LGBM OOF favorable accumulation-rate head",
            "pre_entry": True,
            "oof_or_frozen": True,
            "available_at_col": availability["path_slope"],
            "model_input": True,
        },
        "catboost_entropy": {
            "family": "catboost_entropy",
            "source": "CatBoost OOF path-archetype classifier",
            "pre_entry": True,
            "oof_or_frozen": True,
            "available_at_col": availability["catboost"],
            "model_input": True,
            **catboost_class_contract,
        },
        "catboost_max_probability": {
            "family": "catboost_probability_confidence",
            "source": "CatBoost OOF maximum raw archetype probability",
            "pre_entry": True,
            "oof_or_frozen": True,
            "available_at_col": availability["catboost"],
            "model_input": True,
            **catboost_class_contract,
        },
        "catboost_normalized_entropy": {
            "family": "catboost_probability_uncertainty",
            "source": "CatBoost OOF raw probability entropy normalized by log(class_count)",
            "pre_entry": True,
            "oof_or_frozen": True,
            "available_at_col": availability["catboost"],
            "model_input": True,
            **catboost_class_contract,
        },
        "catboost_top2_probability_margin": {
            "family": "catboost_probability_confidence",
            "source": "CatBoost OOF top-one minus top-two raw probability margin",
            "pre_entry": True,
            "oof_or_frozen": True,
            "available_at_col": availability["catboost"],
            "model_input": True,
            **catboost_class_contract,
        },
        "catboost_adverse_probability_mass": {
            "family": "catboost_path_role_mass",
            "source": "CatBoost OOF raw probability mass over adverse path archetypes",
            "pre_entry": True,
            "oof_or_frozen": True,
            "available_at_col": availability["catboost"],
            "model_input": True,
            "class_group": list(CATBOOST_ADVERSE_CLASSES),
            **catboost_class_contract,
        },
        "catboost_favorable_probability_mass": {
            "family": "catboost_path_role_mass",
            "source": "CatBoost OOF raw probability mass over favorable path archetypes",
            "pre_entry": True,
            "oof_or_frozen": True,
            "available_at_col": availability["catboost"],
            "model_input": True,
            "class_group": list(CATBOOST_FAVORABLE_CLASSES),
            **catboost_class_contract,
        },
        "catboost_archetype": {
            "family": "predicted_path_archetype",
            "source": "CatBoost OOF predicted effective path archetype",
            "pre_entry": True,
            "oof_or_frozen": True,
            "available_at_col": availability["catboost"],
            "model_input": False,
            **catboost_class_contract,
        },
    }
    for column in sorted(
        column
        for column in joined.columns
        if column.startswith(BASE_ARCHETYPE_FEATURE_PREFIX)
    ):
        features[column] = {
            "family": "base_archetype_labels",
            "source": "frozen existing base archetype identity; not CatBoost-derived",
            "pre_entry": True,
            "oof_or_frozen": True,
            "available_at_col": availability["alpha"],
            "model_input": True,
        }
    for source_column, feature_column in TIMING_CDF_JOINED_FEATURE_COLUMNS.items():
        if feature_column not in joined.columns:
            continue
        features[feature_column] = {
            "family": "time_to_meaningful_mfe_cdf_probability",
            "source": "auxiliary LGBM OOF timing-CDF head; signed "
            f"{source_column} probability",
            "pre_entry": True,
            "oof_or_frozen": True,
            "available_at_col": availability["time_to_mfe"],
            "model_input": True,
            "timing_cdf_horizon_hours": int(
                source_column.removeprefix("prediction_p_hit_by_").removesuffix("h")
            ),
            "signed_source_prediction_column": source_column,
        }
    for column in sorted(
        column for column in joined.columns if column.startswith("catboost_p_")
    ):
        features[column] = {
            "family": "catboost_probabilities",
            "source": "CatBoost OOF full probability vector",
            "pre_entry": True,
            "oof_or_frozen": True,
            "available_at_col": availability["catboost"],
            "model_input": True,
            **catboost_class_contract,
        }
    return {
        "schema": HANDOFF_SCHEMA,
        "handoff": {
            "join_mode": "exact_inner_one_to_one",
            "join_keys": list(join_keys),
            "row_count": int(len(joined)),
            "source_artifacts": {
                source.spec.name: source.metadata for source in sources
            },
            "population_alignment": dict(population_alignment),
            "catboost_class_contract": dict(catboost_class_contract),
            "availability_contract": "All model features are OOF/frozen pre-entry predictions with source availability at or before the UTC signal timestamp. Realized execution columns are retained only for target accounting and diagnostics.",
            "cost_basis": {
                "source_alpha_cost_return": float(
                    alpha_cost_basis["deducted_cost_return"]
                ),
                "source_alpha_cost_basis": dict(alpha_cost_basis),
                "execution_cost_column": "execution_cost_return",
                "aligned_alpha_formula": "existing_alpha_ev = existing_alpha_ev_source_basis + alpha_source_cost_return - execution_cost_return",
                "execution_net_formula": "execution_net_ev_12h = execution_gross_ev_12h - execution_cost_return",
            },
        },
        "targets": {
            "execution_net_ev_12h": {
                "source": "execution_labels",
                "decision_time_col": "execution_decision_utc",
                "label_end_time_col": "execution_label_end_utc",
                "signal_to_decision_hours": 1.0,
                "horizon_hours": 12.0,
                "role": "supervised_target_only_not_feature",
            },
        },
        "features": features,
        "materializer": {
            "name": Path(__file__).name,
            "schema": "execution_ev_joined_handoff_materializer_v1",
        },
    }


def run(args: argparse.Namespace) -> dict[str, Path]:
    if args.output.exists():
        raise ValueError(
            f"refusing to overwrite existing output parquet: {args.output}"
        )
    provenance_path = args.provenance_json or args.output.with_suffix(
        ".provenance.json"
    )
    if provenance_path.exists():
        raise ValueError(
            f"refusing to overwrite existing provenance JSON: {provenance_path}"
        )
    sources = [_load_source(spec) for spec in _source_specs(args)]
    population_alignment = _retain_common_intersection(
        sources,
        min_rows=int(getattr(args, "min_common_rows", MIN_COMMON_INTERSECTION_ROWS)),
    )
    catboost_source = next(
        source for source in sources if source.spec.name == "catboost"
    )
    catboost_class_contract = catboost_source.metadata[
        "signed_prediction_role_manifest"
    ]["class_contract"]
    if not isinstance(catboost_class_contract, Mapping):
        raise ValueError("catboost: signed manifest did not yield a class contract")
    if len(args.catboost_prob_cols) != len(catboost_class_contract["class_order"]):
        raise ValueError(
            "the complete CatBoost probability vector requires exactly "
            f"{len(catboost_class_contract['class_order'])} columns in declared class order"
        )
    expected_probability_columns = [
        f"probability__{class_name}"
        for class_name in catboost_class_contract["class_order"]
    ]
    if list(args.catboost_prob_cols) != expected_probability_columns:
        raise ValueError(
            "CatBoost probability columns must match the signed manifest class order"
        )
    alpha, *remaining = sources
    alpha_cost_basis = alpha.metadata["signed_prediction_role_manifest"].get(
        "alpha_cost_basis"
    )
    if not isinstance(
        alpha_cost_basis, Mapping
    ):  # pragma: no cover - validated at load.
        raise ValueError("alpha: signed manifest did not yield an alpha_cost_basis")
    override = getattr(args, "alpha_source_cost_return", None)
    if override is not None:
        try:
            override_value = float(override)
        except (TypeError, ValueError) as exc:
            raise ValueError("alpha source cost override must be finite") from exc
        if not np.isfinite(override_value) or not np.isclose(
            override_value,
            float(alpha_cost_basis["deducted_cost_return"]),
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError(
                "alpha source cost override must exactly match the signed alpha_cost_basis"
            )
    joined = alpha.frame.copy()
    for source in remaining:
        joined = _join(joined, source)
    join_keys = list(JOIN_KEYS)
    _assert_unique(joined, join_keys, source="final joined handoff")
    if joined.empty:
        raise ValueError("joined handoff is empty")

    _numeric_features(
        joined,
        [
            "existing_alpha_ev",
            "alpha_prediction_uncertainty",
            "alpha_leaf_support",
            "pred_time_to_first_meaningful_MFE",
            "pred_peak_MFE_12h_ATR",
            "pred_mae_before_meaningful_mfe_atr",
            "pred_bars_before_price_stops_decreasing",
            "pred_favorable_path_slope_atr_per_hour",
            "catboost_entropy",
            "catboost_max_probability",
            "catboost_normalized_entropy",
            "catboost_top2_probability_margin",
            "catboost_adverse_probability_mass",
            "catboost_favorable_probability_mass",
            "execution_gross_ev_12h",
            "execution_cost_return",
            "execution_net_ev_12h",
            "execution_exit_hour",
            "execution_mfe_return_12h",
            "execution_mae_return_12h",
            *[
                column
                for column in TIMING_CDF_JOINED_FEATURE_COLUMNS.values()
                if column in joined.columns
            ],
            *[
                column
                for column in joined.columns
                if column.startswith(BASE_ARCHETYPE_FEATURE_PREFIX)
            ],
            *[column for column in joined.columns if column.startswith("catboost_p_")],
        ],
    )
    timing_cdf_features = [
        column
        for column in TIMING_CDF_JOINED_FEATURE_COLUMNS.values()
        if column in joined.columns
    ]
    if timing_cdf_features:
        timing_cdf = joined.loc[:, timing_cdf_features].to_numpy(dtype=float)
        if (timing_cdf < -1e-6).any() or (timing_cdf > 1.0 + 1e-6).any():
            raise ValueError(
                "timing CDF OOF probabilities must be finite and bounded in [0, 1]"
            )
        if (np.diff(timing_cdf, axis=1) < -1e-6).any():
            raise ValueError(
                "timing CDF OOF probabilities must be non-decreasing by horizon"
            )
    probabilities = joined.loc[
        :,
        sorted(column for column in joined.columns if column.startswith("catboost_p_")),
    ].to_numpy(dtype=float)
    if (
        (probabilities < -1e-6).any()
        or (probabilities > 1.0 + 1e-6).any()
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4, rtol=1e-4)
    ):
        raise ValueError(
            "CatBoost OOF probability vector must be finite, bounded, and normalized"
        )
    expected_entropy = -np.sum(
        np.clip(probabilities, 1e-12, 1.0) * np.log(np.clip(probabilities, 1e-12, 1.0)),
        axis=1,
    )
    if not np.allclose(
        joined["catboost_entropy"].to_numpy(dtype=float),
        expected_entropy,
        atol=1e-4,
        rtol=1e-4,
    ):
        raise ValueError(
            "CatBoost OOF entropy does not match its full probability vector"
        )
    expected_max_probability = probabilities.max(axis=1)
    ordered_probabilities = np.sort(probabilities, axis=1)
    expected_top2_margin = ordered_probabilities[:, -1] - ordered_probabilities[:, -2]
    expected_normalized_entropy = expected_entropy / np.log(probabilities.shape[1])
    class_indices = {
        str(class_name): index
        for index, class_name in enumerate(catboost_class_contract["class_order"])
    }
    missing_role_classes = sorted(
        set((*CATBOOST_ADVERSE_CLASSES, *CATBOOST_FAVORABLE_CLASSES)).difference(
            class_indices
        )
    )
    if missing_role_classes:
        raise ValueError(
            "CatBoost signed class contract is missing required adverse/favorable "
            f"classes: {missing_role_classes!r}"
        )
    expected_adverse_mass = probabilities[
        :, [class_indices[name] for name in CATBOOST_ADVERSE_CLASSES]
    ].sum(axis=1)
    expected_favorable_mass = probabilities[
        :, [class_indices[name] for name in CATBOOST_FAVORABLE_CLASSES]
    ].sum(axis=1)
    derived_checks = {
        "catboost_max_probability": expected_max_probability,
        "catboost_normalized_entropy": expected_normalized_entropy,
        "catboost_top2_probability_margin": expected_top2_margin,
        "catboost_adverse_probability_mass": expected_adverse_mass,
        "catboost_favorable_probability_mass": expected_favorable_mass,
    }
    for column, expected in derived_checks.items():
        if not np.allclose(
            joined[column].to_numpy(dtype=float),
            expected,
            atol=1e-6,
            rtol=1e-6,
        ):
            raise ValueError(
                f"CatBoost OOF {column} does not match its full probability vector"
            )
    if (
        joined["catboost_archetype"].isna().any()
        or (joined["catboost_archetype"].astype("string").str.strip() == "").any()
    ):
        raise ValueError(
            "CatBoost predicted effective path archetype must be explicit for every row"
        )
    joined["catboost_archetype"] = (
        joined["catboost_archetype"].astype("string").astype(str)
    )
    expected_archetype = np.asarray(
        catboost_class_contract["class_order"], dtype=object
    )[np.argmax(probabilities, axis=1)]
    if not np.array_equal(
        joined["catboost_archetype"].to_numpy(dtype=object), expected_archetype
    ):
        raise ValueError(
            "CatBoost predicted archetype does not match argmax of its full probability vector"
        )
    decision = _utc(
        joined["execution_decision_utc"],
        source="execution_labels",
        column="execution_decision_utc",
    )
    label_end = _utc(
        joined["execution_label_end_utc"],
        source="execution_labels",
        column="execution_label_end_utc",
    )
    if not (decision == joined["__ts__"] + pd.Timedelta(hours=1)).all():
        raise ValueError(
            "execution decision timestamp must equal signal timestamp + one hour"
        )
    if not (label_end == decision + pd.Timedelta(hours=12)).all():
        raise ValueError(
            "execution label-end timestamp must equal decision timestamp + 12 hours"
        )
    accounting_error = (
        joined["execution_gross_ev_12h"]
        - joined["execution_cost_return"]
        - joined["execution_net_ev_12h"]
    ).abs()
    if float(accounting_error.max()) > 1e-6:
        raise ValueError("execution gross-cost-net accounting identity is inconsistent")
    joined["existing_alpha_ev_source_basis"] = joined["existing_alpha_ev"].astype(
        "float64"
    )
    joined["alpha_source_cost_return"] = float(alpha_cost_basis["deducted_cost_return"])
    joined["existing_alpha_ev"] = (
        joined["existing_alpha_ev_source_basis"]
        + joined["alpha_source_cost_return"]
        - joined["execution_cost_return"]
    )
    joined["execution_decision_utc"] = decision
    joined["execution_label_end_utc"] = label_end
    joined = joined.sort_values(join_keys, kind="stable").reset_index(drop=True)

    provenance = _provenance(
        joined,
        sources,
        join_keys=join_keys,
        catboost_class_contract=catboost_class_contract,
        population_alignment=population_alignment,
        alpha_cost_basis=alpha_cost_basis,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(args.output, index=False, compression="zstd")
    _write_json(provenance_path, provenance)
    return {"handoff": args.output, "provenance": provenance_path}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for argument, help_text in (
        ("alpha", "Existing alpha OOF/candidate stream parquet."),
        ("time-oof", "Auxiliary LGBM OOF time-to-first-meaningful-MFE parquet."),
        ("peak-oof", "Auxiliary LGBM OOF peak-MFE-12h-ATR parquet."),
        ("mae-oof", "Auxiliary LGBM OOF MAE-before-meaningful-MFE parquet."),
        ("turn-oof", "Auxiliary LGBM OOF adverse-turn timing parquet."),
        ("slope-oof", "Auxiliary LGBM OOF favorable path-slope parquet."),
        ("catboost-oof", "CatBoost OOF path-archetype parquet."),
        ("execution-labels", "Execution 12h net-label parquet."),
    ):
        parser.add_argument(f"--{argument}", type=Path, required=True, help=help_text)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance-json", type=Path, default=None)
    parser.add_argument("--timestamp-col", default="__ts__")
    parser.add_argument("--symbol-col", default="__symbol__")
    parser.add_argument("--side-col", default="side_name")
    parser.add_argument(
        "--candidate-id-col",
        default=DEFAULT_CANDIDATE_ID_COLUMN,
        help="Required stable candidate identity column in every joined source.",
    )
    for source in (
        "alpha",
        "time",
        "peak",
        "mae",
        "turn",
        "slope",
        "catboost",
        "labels",
    ):
        for key in ("timestamp", "symbol", "side", "candidate_id"):
            parser.add_argument(f"--{source}-{key.replace('_', '-')}-col", default=None)
        parser.add_argument(f"--{source}-manifest", type=Path, required=True)
        parser.add_argument(
            f"--{source}-available-at-col",
            default="execution_label_available_at"
            if source == "labels"
            else "available_at",
        )
        parser.add_argument(
            f"--{source}-label-resolution-available-at-col",
            default="execution_label_available_at"
            if source == "labels"
            else "label_resolution_available_at",
        )
        if source != "labels":
            parser.add_argument(
                f"--{source}-validation-start-col", default="validation_start"
            )
            parser.add_argument(
                f"--{source}-train-decision-cutoff-col", default="train_decision_cutoff"
            )
    parser.add_argument("--alpha-ev-col", default="existing_alpha_ev")
    parser.add_argument(
        "--alpha-uncertainty-col", default="base_prediction_uncertainty"
    )
    parser.add_argument("--alpha-leaf-support-col", default="meta_leaf_support_log1p")
    parser.add_argument("--alpha-fold-col", default="oof_fold")
    parser.add_argument("--time-prediction-col", default="prediction")
    parser.add_argument("--time-fold-col", default="oof_fold")
    parser.add_argument("--peak-prediction-col", default="prediction")
    parser.add_argument("--peak-fold-col", default="oof_fold")
    parser.add_argument("--mae-prediction-col", default="prediction")
    parser.add_argument("--mae-fold-col", default="oof_fold")
    parser.add_argument("--turn-prediction-col", default="prediction")
    parser.add_argument("--turn-fold-col", default="oof_fold")
    parser.add_argument("--slope-prediction-col", default="prediction")
    parser.add_argument("--slope-fold-col", default="oof_fold")
    parser.add_argument(
        "--catboost-prob-cols",
        type=_parse_columns,
        default=[f"probability__{shape}" for shape in MERGED_PATH_ARCHETYPE_CLASSES],
    )
    parser.add_argument("--catboost-entropy-col", default="probability_entropy")
    parser.add_argument("--catboost-max-probability-col", default="max_probability")
    parser.add_argument(
        "--catboost-normalized-entropy-col", default="normalized_entropy"
    )
    parser.add_argument("--catboost-top2-margin-col", default="top2_probability_margin")
    parser.add_argument(
        "--catboost-adverse-mass-col", default="adverse_probability_mass"
    )
    parser.add_argument(
        "--catboost-favorable-mass-col", default="favorable_probability_mass"
    )
    parser.add_argument("--catboost-archetype-col", default="predicted_path_archetype")
    parser.add_argument("--catboost-fold-col", default="oof_fold_id")
    parser.add_argument("--execution-net-ev-col", default="execution_net_ev_12h")
    parser.add_argument("--execution-decision-ts-col", default="__decision_ts__")
    parser.add_argument("--execution-gross-ev-col", default="execution_gross_ev_12h")
    parser.add_argument("--execution-cost-return-col", default="execution_cost_return")
    parser.add_argument("--execution-label-end-col", default="execution_label_end_utc")
    parser.add_argument("--execution-exit-reason-col", default="execution_exit_reason")
    parser.add_argument("--execution-exit-hour-col", default="execution_exit_hour")
    parser.add_argument("--execution-mfe-col", default="execution_mfe_return_12h")
    parser.add_argument("--execution-mae-col", default="execution_mae_return_12h")
    parser.add_argument(
        "--alpha-source-cost-return",
        type=float,
        default=None,
        help=(
            "Optional audit assertion. When provided it must exactly match the "
            "signed alpha_cost_basis; it never overrides manifest evidence."
        ),
    )
    parser.add_argument(
        "--min-common-rows",
        type=int,
        default=MIN_COMMON_INTERSECTION_ROWS,
        help="Fail when the explicit all-source common identity intersection is smaller.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        paths = run(args)
    except (OSError, ValueError) as exc:
        raise SystemExit(
            f"execution-EV joined-handoff materialization failed: {exc}"
        ) from exc
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
