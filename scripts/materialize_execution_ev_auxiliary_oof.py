#!/usr/bin/env python3
"""Normalize completed path-auxiliary LGBM OOF predictions for execution EV.

The side-local auxiliary trainer changed its OOF schema after completed runs
already existed.  This adapter retains only the pre-entry OOF prediction and
its audit identity; it deliberately does not propagate the realized path
target or other outcome-derived columns from those artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.base_candidate_population import candidate_identity_sha256

SCHEMA = "execution_ev_auxiliary_oof_adapter_v3_canonical_head_bundle"
IDENTITY_COLUMNS = ("__ts__", "__symbol__", "side_name")
CANDIDATE_ID_COLUMN = "candidate_id"
EVIDENCE_COLUMNS = (
    "oof_fold",
    "validation_start",
    "train_decision_cutoff",
    "label_resolution_available_at",
)
OUTPUT_COLUMNS = (
    *IDENTITY_COLUMNS,
    CANDIDATE_ID_COLUMN,
    "prediction",
    *EVIDENCE_COLUMNS,
    "available_at",
)


class TargetSpec(NamedTuple):
    kind: str
    aliases: tuple[str, ...]
    lower: float
    upper: float
    unit: str
    natural_prediction_columns: tuple[str, ...]
    canonical_prediction_column: str


TARGET_SPECS = (
    TargetSpec(
        kind="timing",
        aliases=(
            "timing",
            "time",
            "time_to_peak_mfe",
            "time_to_first_meaningful_mfe",
            "time_to_mfe",
        ),
        lower=0.0,
        upper=12.0,
        unit="hours",
        natural_prediction_columns=(
            "pred_time_to_first_meaningful_mfe_12h",
            "pred_time_to_peak_mfe_12h",
            "pred_time_to_mfe_12h",
        ),
        canonical_prediction_column="pred_expected_censored_time_hours",
    ),
    TargetSpec(
        kind="peak_mfe",
        aliases=("peak", "peak_mfe", "peak_mfe_12h_atr"),
        lower=0.0,
        upper=10.0,
        unit="ATR",
        natural_prediction_columns=(
            "pred_peak_mfe_12h_atr",
            "pred_peak_mfe_12h",
        ),
        canonical_prediction_column="pred_expected_peak_mfe_atr",
    ),
    TargetSpec(
        kind="mae_before_meaningful_mfe",
        aliases=(
            "mae_before_meaningful_mfe",
            "mae_before_meaningful_mfe_atr",
            "adverse_depth",
        ),
        lower=0.0,
        upper=10.0,
        unit="ATR",
        natural_prediction_columns=("pred_mae_before_meaningful_mfe_atr_12h",),
        canonical_prediction_column="pred_expected_mae_atr",
    ),
    TargetSpec(
        kind="bars_before_price_stops_decreasing",
        aliases=(
            "bars_before_price_stops_decreasing",
            "adverse_turn_bars",
            "turning_bars",
        ),
        lower=0.0,
        upper=12.0,
        unit="bars",
        natural_prediction_columns=("pred_bars_before_price_stops_decreasing_12h",),
        canonical_prediction_column="pred_confirmed_adverse_trough_bars",
    ),
    TargetSpec(
        kind="future_slope",
        aliases=(
            "future_slope",
            "future_slope_atr_per_hour",
            "path_slope",
        ),
        lower=0.0,
        upper=10.0,
        unit="ATR/hour",
        natural_prediction_columns=("pred_future_slope_atr_per_hour_12h",),
        canonical_prediction_column="diag_pred_future_slope_atr_per_hour",
    ),
)
CANONICAL_HEAD_NAMES = {
    "timing": "time_to_first_meaningful_mfe",
    "peak_mfe": "peak_mfe_12h_atr",
    "mae_before_meaningful_mfe": "mae_before_meaningful_mfe_atr",
    "bars_before_price_stops_decreasing": "bars_before_price_stops_decreasing",
    "future_slope": "future_slope_atr_per_hour",
}
CANONICAL_OOF_MONTHS = ("2026-05", "2026-06", "2026-07")
PREDICTION_ROLES = {
    "timing": "time_to_mfe_oof",
    "peak_mfe": "peak_mfe_oof",
    "mae_before_meaningful_mfe": "mae_before_mfe_oof",
    "bars_before_price_stops_decreasing": "adverse_turn_oof",
    "future_slope": "path_slope_oof",
}
LOG_PREDICTION_COLUMNS = (
    "oof_prediction_log1p",
    "oof_prediction",
    "prediction_log1p",
)
LEAKAGE_TOKENS = (
    "target",
    "label",
    "realized",
    "outcome",
    "actual_",
    "future_",
    "y_",
)


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


def _target_spec(target_kind: str) -> TargetSpec:
    normalized = str(target_kind).strip().lower().replace("-", "_")
    for spec in TARGET_SPECS:
        if normalized in spec.aliases:
            return spec
    allowed = sorted(alias for spec in TARGET_SPECS for alias in spec.aliases)
    raise ValueError(f"unknown target kind {target_kind!r}; expected one of {allowed}")


def _first_present(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    present = set(map(str, columns))
    return next((column for column in candidates if column in present), None)


def _utc(values: pd.Series, *, source: str, column: str) -> pd.Series:
    converted = pd.to_datetime(values, utc=True, errors="coerce")
    if converted.isna().any():
        raise ValueError(f"{source}: {column!r} contains null or invalid timestamps")
    return converted


def _nonempty_strings(values: pd.Series, *, source: str, column: str) -> pd.Series:
    result = values.astype("string").str.strip()
    if result.isna().any() or result.eq("").any():
        raise ValueError(f"{source}: {column!r} contains null or blank identity values")
    return result.astype(str)


def _canonical_side(values: pd.Series, *, source: str, column: str) -> pd.Series:
    side = values.astype("string").str.strip().str.lower()
    aliases = {
        "long": "long",
        "buy": "long",
        "1": "long",
        "1.0": "long",
        "short": "short",
        "sell": "short",
        "-1": "short",
        "-1.0": "short",
    }
    normalized = side.map(aliases)
    if normalized.isna().any():
        invalid = sorted(set(side[normalized.isna()].dropna()))
        raise ValueError(
            f"{source}: {column!r} must contain canonical long/short sides; "
            f"invalid={invalid[:5]!r}"
        )
    return normalized.astype(str)


def _assert_unique(frame: pd.DataFrame, keys: Sequence[str], *, source: str) -> None:
    duplicate_rows = int(frame.duplicated(list(keys), keep=False).sum())
    if duplicate_rows:
        raise ValueError(
            f"{source}: duplicate rows violate exact one-to-one identity on "
            f"{list(keys)!r}; duplicate_rows={duplicate_rows}"
        )


def _parse_folds(values: pd.Series, *, source: str, column: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = (
        numeric.notna()
        & np.isfinite(numeric)
        & (numeric >= 0)
        & (numeric == np.floor(numeric))
    )
    if not bool(valid.all()):
        raise ValueError(f"{source}: {column!r} has missing or invalid OOF fold IDs")
    return numeric.astype("int64")


def _prediction_source(
    raw: pd.DataFrame,
    spec: TargetSpec,
    *,
    source: str,
) -> tuple[str, np.ndarray, str]:
    log_column = _first_present(raw.columns, LOG_PREDICTION_COLUMNS)
    if log_column is not None:
        numeric = pd.to_numeric(raw[log_column], errors="coerce").to_numpy(dtype=float)
        # expm1 is performed only on finite values below, avoiding an overflow
        # warning from intentionally unavailable historical OOF rows.
        natural = np.full(len(numeric), np.nan, dtype=np.float64)
        finite = np.isfinite(numeric)
        natural[finite] = np.expm1(numeric[finite])
        return log_column, natural, "expm1_log1p"

    natural_column = _first_present(raw.columns, spec.natural_prediction_columns)
    if natural_column is None:
        expected = [*LOG_PREDICTION_COLUMNS, *spec.natural_prediction_columns]
        raise ValueError(
            f"{source}: missing an OOF prediction column; expected one of {expected}"
        )
    # Natural columns are admitted only through the explicit per-target
    # prediction allowlist above. A name such as ``pred_future_slope...`` is a
    # model output, despite containing the otherwise outcome-sensitive token
    # ``future_``.
    return (
        natural_column,
        pd.to_numeric(raw[natural_column], errors="coerce").to_numpy(dtype=float),
        "already_natural",
    )


def _read_json_object(path: Path, *, source: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{source}: missing required JSON artifact: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{source}: cannot read JSON artifact: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{source}: JSON artifact must contain an object: {path}")
    return dict(payload)


def _verified_artifact_path(
    record: Any,
    *,
    expected_kind: str,
    source: str,
) -> Path:
    if not isinstance(record, Mapping):
        raise ValueError(f"{source}: missing {expected_kind!r} artifact record")
    if record.get("kind") != expected_kind:
        raise ValueError(
            f"{source}: artifact kind must be {expected_kind!r}, got {record.get('kind')!r}"
        )
    path = Path(str(record.get("path", "")))
    digest = record.get("sha256")
    if not path.is_file() or not isinstance(digest, str) or digest != _sha256(path):
        raise ValueError(
            f"{source}: artifact record is missing or hash-mismatched: {path}"
        )
    return path


def _canonical_head_bundle(
    head_dir: Path,
    spec: TargetSpec,
) -> tuple[pd.DataFrame, Path, dict[str, Any]] | None:
    """Load and validate the canonical composed-head OOF contract when present.

    Canonical bundles intentionally retain every candidate, including reference
    rows and unavailable rows.  Only their explicit ``oof_available`` subset is
    eligible for an execution-EV feature.  The corresponding head manifest and
    promotion gate are both hash-bound before any prediction is read.
    """

    bundle_path = head_dir / "oof_bundle.parquet"
    if not bundle_path.is_file():
        return None
    source = str(bundle_path)
    manifest_path = head_dir / "manifest.json"
    manifest = _read_json_object(manifest_path, source=source)
    expected_head = CANONICAL_HEAD_NAMES[spec.kind]
    if manifest.get("head_name") != expected_head:
        raise ValueError(
            f"{source}: head manifest binds {manifest.get('head_name')!r}, "
            f"not requested canonical head {expected_head!r}"
        )
    bound_bundle = _verified_artifact_path(
        manifest.get("oof_bundle"), expected_kind="head_oof_bundle", source=source
    )
    if bound_bundle.resolve() != bundle_path.resolve():
        raise ValueError(f"{source}: manifest binds a different oof_bundle path")
    if tuple(manifest.get("oof_months", ())) != CANONICAL_OOF_MONTHS:
        raise ValueError(
            f"{source}: canonical execution-EV input requires exact OOF months "
            f"{list(CANONICAL_OOF_MONTHS)!r}"
        )
    if manifest.get("target_columns_are_audit_only") is not True:
        raise ValueError(f"{source}: head manifest must declare targets audit-only")
    if manifest.get("final_refit_excluded_from_oof") is not True:
        raise ValueError(f"{source}: head manifest must exclude final refit from OOF")
    declared_predictions = manifest.get("prediction_columns")
    if (
        not isinstance(declared_predictions, list)
        or spec.canonical_prediction_column not in declared_predictions
    ):
        raise ValueError(
            f"{source}: head manifest does not declare canonical prediction "
            f"{spec.canonical_prediction_column!r}"
        )

    gate_path = _verified_artifact_path(
        manifest.get("promotion_gate"), expected_kind="promotion_gate", source=source
    )
    gate = _read_json_object(gate_path, source=source)
    promotion_status = manifest.get("promotion_status")
    if promotion_status != gate.get("status"):
        raise ValueError(f"{source}: head manifest and promotion gate status disagree")
    declared_deployable = manifest.get("deployable_prediction_columns")
    gated_deployable = gate.get("deployable_prediction_columns")
    if (
        not isinstance(declared_deployable, list)
        or declared_deployable != gated_deployable
    ):
        raise ValueError(
            f"{source}: promotion gate deployable columns disagree with manifest"
        )
    if promotion_status != "ELIGIBLE_FOR_EXECUTION_EV_OOF_CONSUMER":
        raise ValueError(
            f"{source}: canonical head {expected_head!r} is not promotable for "
            f"execution EV ({promotion_status!r}); complete its required identical-row ablation first"
        )
    if spec.canonical_prediction_column not in declared_deployable:
        raise ValueError(
            f"{source}: canonical prediction {spec.canonical_prediction_column!r} "
            "is not promotion-gated as deployable"
        )

    raw = pd.read_parquet(bundle_path)
    if raw.empty:
        raise ValueError(f"{source}: canonical OOF bundle is empty")
    required = {
        "__ts__",
        "__symbol__",
        "side",
        CANDIDATE_ID_COLUMN,
        "oof_available",
        "oof_fold",
        "oof_fold_month",
        "validation_start",
        "train_label_resolution_max",
        "prediction_available_at",
        spec.canonical_prediction_column,
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(
            f"{source}: canonical OOF bundle is missing columns: {missing}"
        )
    availability = raw["oof_available"]
    if availability.isna().any() or not availability.isin((True, False)).all():
        raise ValueError(f"{source}: oof_available must be a finite boolean contract")
    available = availability.astype(bool).to_numpy()
    if not available.any():
        raise ValueError(f"{source}: canonical OOF bundle has no available rows")
    declared_oof_rows = manifest.get("oof_rows")
    if not isinstance(declared_oof_rows, int) or declared_oof_rows != int(
        available.sum()
    ):
        raise ValueError(f"{source}: head manifest OOF row count does not match bundle")
    canonical_side = _canonical_side(raw["side"], source=source, column="side")
    identity = pd.DataFrame(
        {
            "__ts__": _utc(raw["__ts__"], source=source, column="__ts__"),
            "__symbol__": _nonempty_strings(
                raw["__symbol__"], source=source, column="__symbol__"
            ),
            "side": canonical_side,
            CANDIDATE_ID_COLUMN: _nonempty_strings(
                raw[CANDIDATE_ID_COLUMN], source=source, column=CANDIDATE_ID_COLUMN
            ),
        }
    )
    _assert_unique(
        identity, ("__ts__", "__symbol__", "side", CANDIDATE_ID_COLUMN), source=source
    )
    declared_identity = manifest.get("candidate_identity_sha256")
    observed_identity = candidate_identity_sha256(
        identity, columns=("__ts__", "__symbol__", "side", CANDIDATE_ID_COLUMN)
    )
    if not isinstance(declared_identity, str) or declared_identity != observed_identity:
        raise ValueError(
            f"{source}: canonical candidate identity hash does not match head manifest"
        )
    return (
        raw,
        bundle_path,
        {
            "manifest_path": manifest_path,
            "manifest": manifest,
            "promotion_gate_path": gate_path,
            "promotion_gate": gate,
            "available_mask": available,
        },
    )


def _fold_metrics(
    metrics_path: Path,
    *,
    source: str,
) -> dict[str, list[tuple[int, pd.Timestamp, pd.Timestamp, pd.Timestamp]]]:
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"{source}: missing metrics.json required to reconstruct OOF folds: {metrics_path}"
        )
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    raw_metrics = payload.get("fold_metrics")
    if not isinstance(raw_metrics, Mapping):
        raise ValueError(f"{source}: metrics.json lacks side-local fold_metrics")
    result: dict[str, list[tuple[int, pd.Timestamp, pd.Timestamp, pd.Timestamp]]] = {}
    for side in ("long", "short"):
        rows = raw_metrics.get(side)
        if not isinstance(rows, list) or not rows:
            raise ValueError(
                f"{source}: metrics.json lacks fold_metrics for side {side!r}"
            )
        parsed: list[tuple[int, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
        fold_ids: set[int] = set()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(
                    f"{source}: invalid fold_metrics entry for side {side!r}"
                )
            fold = _parse_folds(
                pd.Series([row.get("fold")]), source=source, column="fold"
            ).iloc[0]
            start = _utc(
                pd.Series([row.get("valid_start")]), source=source, column="valid_start"
            ).iloc[0]
            end = _utc(
                pd.Series([row.get("valid_end")]), source=source, column="valid_end"
            ).iloc[0]
            train_end = _utc(
                pd.Series([row.get("train_end")]), source=source, column="train_end"
            ).iloc[0]
            if end < start:
                raise ValueError(
                    f"{source}: fold {fold} for side {side!r} has valid_end < valid_start"
                )
            if train_end >= start:
                raise ValueError(
                    f"{source}: fold {fold} for side {side!r} has a training "
                    "cutoff at or after validation start"
                )
            if int(fold) in fold_ids:
                raise ValueError(
                    f"{source}: duplicate fold ID {fold} for side {side!r}"
                )
            fold_ids.add(int(fold))
            parsed.append((int(fold), start, end, train_end))
        parsed.sort(key=lambda item: item[1])
        for previous, current in zip(parsed, parsed[1:]):
            if current[1] <= previous[2]:
                raise ValueError(
                    f"{source}: inclusive validation intervals overlap for side {side!r}"
                )
        result[side] = parsed
    return result


def _reconstruct_folds(
    frame: pd.DataFrame,
    metrics_path: Path,
    *,
    source: str,
) -> pd.Series:
    metrics = _fold_metrics(metrics_path, source=source)
    folds = np.full(len(frame), -1, dtype=np.int64)
    for side in ("long", "short"):
        positions = np.flatnonzero(frame["side_name"].to_numpy() == side)
        timestamps = frame["__ts__"].iloc[positions]
        matches = np.zeros(len(positions), dtype=np.int8)
        assigned = np.full(len(positions), -1, dtype=np.int64)
        for fold, start, end, _train_end in metrics[side]:
            # Path auxiliary fold manifests record ``valid_end`` as the actual
            # final included timestamp, not an exclusive boundary.
            in_fold = (timestamps >= start).to_numpy() & (timestamps <= end).to_numpy()
            matches += in_fold.astype(np.int8)
            assigned[in_fold] = fold
        if not np.all(matches == 1):
            bad = positions[matches != 1]
            sample = [str(value) for value in frame["__ts__"].iloc[bad[:3]]]
            raise ValueError(
                f"{source}: every finite OOF row must map to exactly one side-local "
                f"validation fold; side={side!r}, unmatched_or_overlapping_rows={len(bad)}, "
                f"sample_timestamps={sample}"
            )
        folds[positions] = assigned
    if (folds < 0).any():  # pragma: no cover - both canonical sides are asserted above.
        raise ValueError(f"{source}: failed to reconstruct OOF folds")
    return pd.Series(folds, index=frame.index, dtype="int64")


def materialize(
    *,
    head_dir: Path,
    target_kind: str,
    output: Path,
    manifest: Path,
) -> dict[str, Path]:
    """Materialize one auxiliary head into the execution-EV OOF contract."""

    spec = _target_spec(target_kind)
    head_dir = Path(head_dir)
    if output.exists():
        raise ValueError(f"refusing to overwrite existing output parquet: {output}")
    if manifest.exists():
        raise ValueError(f"refusing to overwrite existing manifest JSON: {manifest}")
    canonical = _canonical_head_bundle(head_dir, spec)
    canonical_metadata: dict[str, Any] | None = None
    if canonical is not None:
        raw, source_path, canonical_metadata = canonical
        source = str(source_path)
        ts_column, symbol_column, side_column = "__ts__", "__symbol__", "side"
        prediction_column = spec.canonical_prediction_column
        natural_prediction = pd.to_numeric(
            raw[prediction_column], errors="coerce"
        ).to_numpy(dtype=float)
        transform = "already_natural_composed_head_oof"
        available_mask = np.asarray(canonical_metadata["available_mask"], dtype=bool)
        if not np.isfinite(natural_prediction[available_mask]).all():
            raise ValueError(
                f"{source}: canonical oof_available rows require finite "
                f"{prediction_column!r} predictions"
            )
        finite_prediction = available_mask
        # No decision outside the immutable May--July outer-OOF calendar may
        # enter the execution-EV training handoff.
        fold_month = raw.loc[finite_prediction, "oof_fold_month"].astype("string")
        timestamp_month = _utc(
            raw.loc[finite_prediction, ts_column], source=source, column=ts_column
        ).dt.strftime("%Y-%m")
        if set(fold_month) != set(CANONICAL_OOF_MONTHS) or not np.array_equal(
            fold_month.to_numpy(), timestamp_month.to_numpy()
        ):
            raise ValueError(
                f"{source}: available OOF rows must be exactly the May--July "
                "outer-OOF calendar and their fold month must match decision time"
            )
        raw_evidence = {
            "oof_fold": "oof_fold",
            "validation_start": "validation_start",
            # The canonical bundle records the exact latest resolved training
            # label.  It is a conservative, evidence-backed pre-validation
            # bound for the legacy handoff field; we never invent a fold cutoff.
            "train_decision_cutoff": "train_label_resolution_max",
            "label_resolution_available_at": "train_label_resolution_max",
            "available_at": "prediction_available_at",
        }
    else:
        source_path = head_dir / "oof_predictions.parquet"
        if not source_path.is_file():
            raise FileNotFoundError(
                "missing canonical oof_bundle.parquet and legacy auxiliary OOF parquet: "
                f"{source_path}"
            )
        raw = pd.read_parquet(source_path)
        if raw.empty:
            raise ValueError(f"{source_path}: source parquet is empty")
        ts_column = _first_present(raw.columns, ("__ts__", "timestamp"))
        symbol_column = _first_present(raw.columns, ("__symbol__", "symbol"))
        side_column = _first_present(raw.columns, ("side_name", "side"))
        required_provenance = {CANDIDATE_ID_COLUMN, *EVIDENCE_COLUMNS}
        missing_provenance = sorted(required_provenance.difference(raw.columns))
        if ts_column is None or symbol_column is None or side_column is None:
            raise ValueError(
                f"{source_path}: missing identity columns; require timestamp, symbol, and side "
                "under canonical or legacy names"
            )
        if missing_provenance:
            raise ValueError(
                f"{source_path}: strict execution-EV provenance is unavailable; missing evidence columns: "
                f"{', '.join(missing_provenance)}. Regenerate the auxiliary OOF artifact; do not infer these values."
            )
        prediction_column, natural_prediction, transform = _prediction_source(
            raw, spec, source=str(source_path)
        )
        finite_prediction = np.isfinite(natural_prediction)
        raw_evidence = {
            "oof_fold": "oof_fold",
            **{column: column for column in EVIDENCE_COLUMNS},
            "available_at": "available_at",
        }

    if not bool(finite_prediction.any()):
        raise ValueError(f"{source_path}: no finite OOF predictions are available")

    work = pd.DataFrame(
        {
            "__ts__": _utc(raw[ts_column], source=str(source_path), column=ts_column),
            "__symbol__": _nonempty_strings(
                raw[symbol_column], source=str(source_path), column=symbol_column
            ),
            "side_name": _canonical_side(
                raw[side_column], source=str(source_path), column=side_column
            ),
            CANDIDATE_ID_COLUMN: _nonempty_strings(
                raw[CANDIDATE_ID_COLUMN],
                source=str(source_path),
                column=CANDIDATE_ID_COLUMN,
            ),
        }
    )
    # Finished historical artifacts contain leading non-OOF rows with NaN
    # predictions. They are unavailable, not valid OOF observations.
    work = work.loc[finite_prediction].copy()
    work["prediction"] = np.clip(
        natural_prediction[finite_prediction], spec.lower, spec.upper
    )
    if not np.isfinite(work["prediction"].to_numpy(dtype=float)).all():
        raise ValueError(
            f"{source_path}: non-finite predictions cannot enter the output"
        )
    _assert_unique(work, IDENTITY_COLUMNS, source=str(source_path))

    work["oof_fold"] = _parse_folds(
        raw.loc[finite_prediction, raw_evidence["oof_fold"]],
        source=str(source_path),
        column=raw_evidence["oof_fold"],
    ).to_numpy()
    for column in (
        "validation_start",
        "train_decision_cutoff",
        "label_resolution_available_at",
        "available_at",
    ):
        work[column] = _utc(
            raw.loc[finite_prediction, raw_evidence[column]],
            source=str(source_path),
            column=raw_evidence[column],
        ).to_numpy()
    if not (work["train_decision_cutoff"] < work["validation_start"]).all():
        raise ValueError(
            f"{source_path}: train decision cutoff must be strictly before validation start"
        )
    if not (
        work["label_resolution_available_at"] <= work["train_decision_cutoff"]
    ).all():
        raise ValueError(
            f"{source_path}: training-label resolution must be available by its train decision cutoff"
        )
    if not (work["validation_start"] <= work["__ts__"]).all():
        raise ValueError(
            f"{source_path}: validation start is after an OOF decision timestamp"
        )
    if not (work["available_at"] <= work["__ts__"]).all():
        raise ValueError(
            f"{source_path}: OOF prediction availability is after decision timestamp"
        )

    output_frame = work.loc[:, OUTPUT_COLUMNS].copy()
    feature_columns = set(output_frame.columns).difference(
        {*IDENTITY_COLUMNS, CANDIDATE_ID_COLUMN, *EVIDENCE_COLUMNS, "available_at"}
    )
    if any(
        any(token in column.lower() for token in LEAKAGE_TOKENS)
        for column in feature_columns
    ):
        raise ValueError("output column contract includes a target/leakage column")
    _assert_unique(output_frame, IDENTITY_COLUMNS, source="materialized auxiliary OOF")
    if not np.isfinite(output_frame["prediction"].to_numpy(dtype=float)).all():
        raise ValueError("materialized auxiliary OOF contains non-finite predictions")
    output_frame = output_frame.sort_values(
        list(IDENTITY_COLUMNS), kind="stable"
    ).reset_index(drop=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_parquet(output, index=False)
    payload = {
        "schema": SCHEMA,
        "target": {
            "kind": spec.kind,
            "unit": spec.unit,
            "input_prediction_column": prediction_column,
            "transform": transform,
            "natural_unit_clip": [spec.lower, spec.upper],
        },
        "source": {
            "head_dir": str(head_dir.resolve()),
            "oof_predictions_path": str(source_path.resolve()),
            "oof_predictions_sha256": _sha256(source_path),
            "input_rows": int(len(raw)),
            "finite_oof_rows": int(finite_prediction.sum()),
            "dropped_unavailable_rows": int((~finite_prediction).sum()),
            "identity_columns": {
                "timestamp": ts_column,
                "symbol": symbol_column,
                "side": side_column,
            },
            "input_contract": (
                "canonical_composed_head_oof_bundle"
                if canonical_metadata is not None
                else "legacy_row_level_auxiliary_oof"
            ),
        },
        "oof_fold": {
            "mode": "source_row_level_actual_fitted_fold_evidence",
            "columns": list(EVIDENCE_COLUMNS),
            "rule": (
                "canonical source row carries its actual fitted fold ID, validation start, "
                "latest resolved training-label timestamp, and prediction availability; "
                "no fold metadata is inferred"
                if canonical_metadata is not None
                else "source row carries its actual fitted fold ID, validation start, resolved-training cutoff, and label-resolution availability; no fold metadata is inferred"
            ),
        },
        "availability": {
            "source_column": "available_at",
            "rule": "source available_at <= UTC decision timestamp",
        },
        "output": {
            "path": str(output.resolve()),
            "sha256": _sha256(output),
            "rows": int(len(output_frame)),
            "columns": list(OUTPUT_COLUMNS),
            "identity": list(IDENTITY_COLUMNS),
            "utc_timestamp_contract": "UTC only; naive source timestamps are interpreted as UTC",
            "leakage_contract": "Only an OOF/frozen prediction and audit identity are emitted; raw targets and realized path columns are excluded.",
        },
        "materializer": {"name": Path(__file__).name, "schema": SCHEMA},
    }
    if canonical_metadata is not None:
        payload["source"]["canonical_head_manifest"] = {
            "path": str(canonical_metadata["manifest_path"].resolve()),
            "sha256": _sha256(canonical_metadata["manifest_path"]),
            "candidate_identity_sha256": canonical_metadata["manifest"][
                "candidate_identity_sha256"
            ],
            "oof_months": list(CANONICAL_OOF_MONTHS),
        }
        payload["source"]["promotion_gate"] = {
            "path": str(canonical_metadata["promotion_gate_path"].resolve()),
            "sha256": _sha256(canonical_metadata["promotion_gate_path"]),
            "status": canonical_metadata["promotion_gate"]["status"],
            "selected_prediction_column": prediction_column,
        }
    payload["prediction_role"] = PREDICTION_ROLES[spec.kind]
    payload["source_artifact_sha256"] = _sha256(output)
    payload["prediction_columns"] = {
        "prediction": {
            "role": "pre_entry_auxiliary_oof_prediction",
            "target": False,
            "head": spec.kind,
        }
    }
    payload["prediction_role_manifest_sha256"] = _canonical_json_hash(payload)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    _write_json(manifest, payload)
    return {"output": output, "manifest": manifest}


def run(args: argparse.Namespace) -> dict[str, Path]:
    output = args.output or (args.head_dir / "execution_ev_auxiliary_oof.parquet")
    manifest = args.manifest or output.with_suffix(".manifest.json")
    return materialize(
        head_dir=args.head_dir,
        target_kind=args.target_kind,
        output=output,
        manifest=manifest,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-dir", required=True, type=Path)
    parser.add_argument("--target-kind", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def main() -> int:
    paths = run(parse_args())
    print(json.dumps({name: str(path) for name, path in paths.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
