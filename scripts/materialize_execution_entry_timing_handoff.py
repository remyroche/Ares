#!/usr/bin/env python3
"""Materialize the strict OOF post-execution-EV entry-timing handoff.

This is deliberately a contract builder.  It joins the original execution-EV
joined handoff to *row-level OOF* execution-EV predictions, an independently
OOF EV map, and signed one-minute timing-label paths.  It never derives a
prediction, a map, a path, ATR, or execution cost component from another
column.  Missing upstream evidence is therefore a hard, actionable failure.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.path_archetype_labels import PATH_SHAPE_TYPES  # noqa: E402

SCHEMA = "execution_entry_timing_handoff_v1"
HANDOFF_SCHEMA = "execution_ev_joined_handoff_v2"
JOIN_KEYS = ("__ts__", "__symbol__", "side_name", "candidate_id")
TIMING_PATH_SCHEMA = "execution_entry_timing_1m_paths_v1"
TIMING_PATH_ROLE = "execution_entry_timing_1m_paths"
EV_MAP_ROLE = "execution_ev_map_oof"

_AUXILIARY_COLUMNS = {
    "pred_time_to_first_meaningful_MFE": "frozen_aux_time",
    "pred_peak_MFE_12h_ATR": "frozen_aux_peak",
    "pred_mae_before_meaningful_mfe_atr": "frozen_aux_mae",
    "pred_bars_before_price_stops_decreasing": "frozen_aux_turn",
    "pred_favorable_path_slope_atr_per_hour": "frozen_aux_slope",
}
_HANDOFF_OOF_SOURCES = (
    "alpha",
    "time_to_mfe",
    "peak_mfe",
    "mae_before_mfe",
    "adverse_turn",
    "path_slope",
    "catboost",
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


def _canonical_manifest_hash(payload: Mapping[str, Any]) -> str:
    canonical = {
        str(key): _json_safe(value)
        for key, value in payload.items()
        if key != "prediction_role_manifest_sha256"
    }
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path, *, role: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{role}: expected a JSON object at {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{role}: expected a JSON object at {path}")
    return payload


def _load_signed_artifact_manifest(
    path: Path,
    *,
    source: str,
    artifact: Path,
    prediction_role: str,
    schema: str | None = None,
) -> dict[str, Any]:
    payload = _read_json(path, role=source)
    signed = payload.get("prediction_role_manifest_sha256")
    if not isinstance(signed, str) or not signed:
        raise ValueError(f"{source}: signed prediction-role manifest hash is required")
    if not hmac.compare_digest(signed, _canonical_manifest_hash(payload)):
        raise ValueError(f"{source}: signed prediction-role manifest hash does not verify")
    if payload.get("prediction_role") != prediction_role:
        raise ValueError(
            f"{source}: prediction_role must be {prediction_role!r}, got {payload.get('prediction_role')!r}"
        )
    if schema is not None and payload.get("schema") != schema:
        raise ValueError(f"{source}: schema must be {schema!r}")
    artifact_hash = payload.get("source_artifact_sha256", payload.get("output_sha256"))
    if not isinstance(artifact_hash, str) or not hmac.compare_digest(artifact_hash, _sha256(artifact)):
        raise ValueError(f"{source}: signed manifest does not bind this parquet artifact hash")
    return payload


def _utc(values: pd.Series, *, source: str, column: str) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"{source}: {column!r} contains null or invalid UTC timestamps")
    return parsed


def _nonempty(values: pd.Series, *, source: str, column: str) -> pd.Series:
    parsed = values.astype("string").str.strip()
    if parsed.isna().any() or parsed.eq("").any():
        raise ValueError(f"{source}: {column!r} contains null or blank identity values")
    return parsed.astype(str)


def _canonical_frame(path: Path, *, source: str) -> pd.DataFrame:
    if not path.is_file() or path.suffix.lower() not in {".parquet", ".pq"}:
        raise ValueError(f"{source}: expected an existing parquet artifact")
    frame = pd.read_parquet(path)
    if frame.empty:
        raise ValueError(f"{source}: source parquet is empty")
    missing = sorted(set(JOIN_KEYS).difference(frame.columns))
    if missing:
        raise ValueError(f"{source}: missing strict identity columns: {', '.join(missing)}")
    work = frame.copy()
    work["__ts__"] = _utc(work["__ts__"], source=source, column="__ts__")
    work["__symbol__"] = _nonempty(work["__symbol__"], source=source, column="__symbol__")
    work["side_name"] = _nonempty(work["side_name"], source=source, column="side_name").str.lower()
    if not work["side_name"].isin(("long", "short")).all():
        raise ValueError(f"{source}: side_name must contain canonical long/short values")
    work["candidate_id"] = _nonempty(work["candidate_id"], source=source, column="candidate_id")
    _assert_unique(work, source=source)
    return work


def _assert_unique(frame: pd.DataFrame, *, source: str) -> None:
    duplicate_rows = int(frame.duplicated(list(JOIN_KEYS), keep=False).sum())
    if duplicate_rows:
        raise ValueError(
            f"{source}: duplicate rows violate exact identity on {list(JOIN_KEYS)!r}; "
            f"duplicate_rows={duplicate_rows}"
        )


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, source: str, remediation: str = "") -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        suffix = f" {remediation}" if remediation else ""
        raise ValueError(f"{source}: missing required columns: {', '.join(missing)}.{suffix}")


def _finite(frame: pd.DataFrame, columns: Sequence[str], *, source: str) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{source}: {column!r} must contain finite numeric values")
        frame[column] = values


def _require_true_oof_flag(values: pd.Series, *, source: str, column: str) -> None:
    if not pd.api.types.is_bool_dtype(values) or values.isna().any() or not values.eq(True).all():
        raise ValueError(
            f"{source}: {column!r} must be a true boolean for every row; "
            "in-sample/final-refit predictions are rejected"
        )


def _join(base: pd.DataFrame, addition: pd.DataFrame, *, source: str, columns: Sequence[str]) -> pd.DataFrame:
    _assert_unique(base, source="retained timing handoff before " + source)
    _assert_unique(addition, source=source)
    coverage = base.loc[:, JOIN_KEYS].merge(
        addition.loc[:, JOIN_KEYS], on=list(JOIN_KEYS), how="outer", indicator=True, sort=False
    )
    missing = int(coverage["_merge"].eq("left_only").sum())
    unexpected = int(coverage["_merge"].eq("right_only").sum())
    if missing or unexpected:
        raise ValueError(
            f"{source}: exact candidate identity coverage mismatch on {list(JOIN_KEYS)!r}; "
            f"missing_from_source={missing}, unexpected_in_source={unexpected}"
        )
    return base.merge(
        addition.loc[:, [*JOIN_KEYS, *columns]],
        on=list(JOIN_KEYS),
        how="inner",
        validate="one_to_one",
        sort=False,
    )


def _validate_original_handoff(
    handoff: pd.DataFrame, provenance_path: Path
) -> dict[str, Any]:
    provenance = _read_json(provenance_path, role="joined handoff provenance")
    if provenance.get("schema") != HANDOFF_SCHEMA:
        raise ValueError(f"joined handoff provenance: schema must be {HANDOFF_SCHEMA!r}")
    contract = provenance.get("handoff")
    if not isinstance(contract, Mapping):
        raise ValueError("joined handoff provenance: handoff contract is required")
    if contract.get("join_mode") != "exact_inner_one_to_one" or contract.get("join_keys") != list(JOIN_KEYS):
        raise ValueError("joined handoff provenance: requires exact_inner_one_to_one canonical candidate keys")
    required = [
        "__decision_ts__",
        "execution_label_end_utc",
        "existing_alpha_ev",
        "catboost_entropy",
        "catboost_archetype",
        *_AUXILIARY_COLUMNS,
        *[f"catboost_p_{index}" for index in range(len(PATH_SHAPE_TYPES))],
    ]
    _require_columns(
        handoff,
        required,
        source="original execution-EV joined handoff",
        remediation="Regenerate it with scripts/materialize_execution_ev_joined_handoff.py; do not synthesize missing timing features.",
    )
    for source in _HANDOFF_OOF_SOURCES:
        _require_columns(
            handoff,
            (f"{source}_oof_fold", f"{source}_train_decision_cutoff", f"{source}_available_at"),
            source="original execution-EV joined handoff",
            remediation="Regenerate the source OOF adapter with row-level fold, cutoff, and availability evidence.",
        )
    _finite(handoff, ["existing_alpha_ev", "catboost_entropy", *_AUXILIARY_COLUMNS], source="original execution-EV joined handoff")
    _finite(
        handoff,
        [f"catboost_p_{index}" for index in range(len(PATH_SHAPE_TYPES))],
        source="original execution-EV joined handoff",
    )
    probabilities = handoff.loc[:, [f"catboost_p_{index}" for index in range(len(PATH_SHAPE_TYPES))]].to_numpy(dtype=float)
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=2e-5, rtol=2e-5):
        raise ValueError("original execution-EV joined handoff: CatBoost probability vector must sum to one")
    entropy = -np.sum(np.clip(probabilities, 1e-12, 1.0) * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1)
    if not np.allclose(handoff["catboost_entropy"].to_numpy(dtype=float), entropy, atol=2e-5, rtol=2e-5):
        raise ValueError("original execution-EV joined handoff: CatBoost entropy does not match probability vector")
    expected_arch = np.asarray(PATH_SHAPE_TYPES, dtype=object)[np.argmax(probabilities, axis=1)]
    if not np.array_equal(handoff["catboost_archetype"].astype(str).to_numpy(), expected_arch):
        raise ValueError("original execution-EV joined handoff: CatBoost archetype must equal probability argmax")
    base_archetypes = sorted(column for column in handoff if column.startswith("base_archetype_label__"))
    if not base_archetypes:
        raise ValueError("original execution-EV joined handoff: frozen base archetype features are required")
    _finite(handoff, base_archetypes, source="original execution-EV joined handoff")
    return provenance


def _validate_execution_ev_run(
    manifest_path: Path, *, handoff_path: Path, provenance_path: Path, oof_path: Path
) -> dict[str, Any]:
    payload = _read_json(manifest_path, role="execution-EV runner manifest")
    if payload.get("schema") != "execution_ev_meta_runner_v1" or payload.get("status") != "completed":
        raise ValueError("execution-EV runner manifest: completed execution_ev_meta_runner_v1 manifest is required")
    source = payload.get("input")
    provenance = payload.get("provenance")
    if not isinstance(source, Mapping) or not hmac.compare_digest(str(source.get("sha256", "")), _sha256(handoff_path)):
        raise ValueError("execution-EV runner manifest: input hash does not bind the original joined handoff")
    if not isinstance(provenance, Mapping) or not hmac.compare_digest(str(provenance.get("sha256", "")), _sha256(provenance_path)):
        raise ValueError("execution-EV runner manifest: provenance hash does not bind the original joined handoff provenance")
    if Path(str(payload.get("oof_ledger", ""))).name != oof_path.name:
        raise ValueError("execution-EV runner manifest: oof_ledger does not name --execution-ev-oof")
    return payload


def _validate_execution_ev_oof(frame: pd.DataFrame) -> None:
    required = (
        "direct__all_features",
        "residual__all_features",
        "direct__all_features__is_oof",
        "residual__all_features__is_oof",
        "execution_ev_oof_fold",
        "execution_ev_oof_train_decision_cutoff_utc",
        "execution_ev_oof_available_at",
    )
    _require_columns(
        frame,
        required,
        source="execution-EV OOF ledger",
        remediation=(
            "Current scripts/run_execution_ev_meta.py output is insufficient until it carries "
            "row-level OOF availability and no-final-refit evidence; do not use final-fit scores."
        ),
    )
    _finite(frame, ("direct__all_features", "residual__all_features"), source="execution-EV OOF ledger")
    for column in ("direct__all_features__is_oof", "residual__all_features__is_oof"):
        _require_true_oof_flag(frame[column], source="execution-EV OOF ledger", column=column)
    for column in ("execution_ev_oof_fold",):
        if frame[column].isna().any() or frame[column].astype("string").str.strip().eq("").any():
            raise ValueError(f"execution-EV OOF ledger: {column!r} requires row-level OOF provenance")
    cutoff = _utc(frame["execution_ev_oof_train_decision_cutoff_utc"], source="execution-EV OOF ledger", column="execution_ev_oof_train_decision_cutoff_utc")
    available = _utc(frame["execution_ev_oof_available_at"], source="execution-EV OOF ledger", column="execution_ev_oof_available_at")
    if not (cutoff < frame["__ts__"]).all() or not (available <= frame["__ts__"]).all():
        raise ValueError("execution-EV OOF ledger: row-level cutoff/availability violates UTC OOF causality")


def _validate_ev_map(frame: pd.DataFrame, manifest: Mapping[str, Any]) -> None:
    if manifest.get("oof_only") is not True or str(manifest.get("prediction_scope", "oof")).lower() != "oof":
        raise ValueError("execution-EV map manifest: explicit oof_only=true and prediction_scope='oof' are required; final-refit maps are rejected")
    required = (
        "mapped_execution_ev",
        "mapped_execution_ev__is_oof",
        "execution_ev_map_oof_fold",
        "execution_ev_map_train_decision_cutoff_utc",
        "execution_ev_map_available_at",
    )
    _require_columns(
        frame,
        required,
        source="execution-EV map OOF artifact",
        remediation="Materialize a signed row-level OOF EV map; never substitute a final-refit calibration map.",
    )
    _finite(frame, ("mapped_execution_ev",), source="execution-EV map OOF artifact")
    _require_true_oof_flag(
        frame["mapped_execution_ev__is_oof"],
        source="execution-EV map OOF artifact",
        column="mapped_execution_ev__is_oof",
    )
    if frame["execution_ev_map_oof_fold"].isna().any() or frame["execution_ev_map_oof_fold"].astype("string").str.strip().eq("").any():
        raise ValueError("execution-EV map OOF artifact: row-level map OOF fold is required")
    cutoff = _utc(frame["execution_ev_map_train_decision_cutoff_utc"], source="execution-EV map OOF artifact", column="execution_ev_map_train_decision_cutoff_utc")
    available = _utc(frame["execution_ev_map_available_at"], source="execution-EV map OOF artifact", column="execution_ev_map_available_at")
    if not (cutoff < frame["__ts__"]).all() or not (available <= frame["__ts__"]).all():
        raise ValueError("execution-EV map OOF artifact: row-level cutoff/availability violates UTC OOF causality")


def _load_target_manifest(path: Path) -> dict[str, Any]:
    # The target manifest is signed but binds the 12h label parquet rather than
    # itself, so validate the signature and timing contract here.
    payload = _read_json(path, role="execution-EV target manifest")
    signed = payload.get("prediction_role_manifest_sha256")
    if not isinstance(signed, str) or not hmac.compare_digest(signed, _canonical_manifest_hash(payload)):
        raise ValueError("execution-EV target manifest: signed manifest hash does not verify")
    if payload.get("schema") != "execution_ev_12h_hourly_policy_labels_v2" or payload.get("prediction_role") != "execution_ev_12h_labels":
        raise ValueError("execution-EV target manifest: incompatible execution-EV label contract")
    timing = payload.get("timing")
    if not isinstance(timing, Mapping) or timing.get("signal_timestamp") != "__ts__" or timing.get("first_path_timestamp") != "__decision_ts__" or float(timing.get("horizon_hours", -1)) != 12.0:
        raise ValueError("execution-EV target manifest: requires canonical 1h decision and 12h horizon timing")
    return payload


def _validate_timing_labels(
    frame: pd.DataFrame, manifest: Mapping[str, Any], target_manifest: Mapping[str, Any], target_path: Path
) -> None:
    if manifest.get("execution_ev_target_manifest_sha256") != _sha256(target_path):
        raise ValueError("1m timing-label manifest: execution-EV target manifest hash does not match")
    if manifest.get("execution_ev_target_signed_manifest_sha256") != target_manifest.get("prediction_role_manifest_sha256"):
        raise ValueError("1m timing-label manifest: execution-EV target signed identity does not match")
    if manifest.get("cost_accounting") != "fee_once_entry_spread_once_exit_spread_once":
        raise ValueError("1m timing-label manifest: requires decomposed fee/spread contract with costs charged once")
    required = ("execution_future_path", "atr_1h", "fee", "entry_spread", "exit_spread")
    _require_columns(
        frame,
        required,
        source="signed 1m timing labels",
        remediation="Materialize and sign exact 1m paths, ATR, fee, entry spread, and exit spread; do not derive them from execution_cost_return.",
    )
    if frame["execution_future_path"].isna().any() or frame["execution_future_path"].astype("string").str.strip().eq("").any():
        raise ValueError("signed 1m timing labels: execution_future_path is required for every row")
    _finite(frame, ("atr_1h", "fee", "entry_spread", "exit_spread"), source="signed 1m timing labels")
    if (frame[["atr_1h", "fee", "entry_spread", "exit_spread"]] < 0.0).any().any():
        raise ValueError("signed 1m timing labels: ATR, fee, and spreads must be non-negative")


def _feature(
    family: str,
    source: str,
    availability: str,
    fold: str | None = None,
    cutoff: str | None = None,
    *,
    cost_spread_aware: bool = False,
    model_input: bool = True,
    frozen_bundle_id: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "family": family,
        "source": source,
        "pre_entry": True,
        "oof_or_frozen": True,
        "available_at_col": availability,
        "cost_spread_aware": cost_spread_aware,
        "model_input": model_input,
    }
    if fold is not None:
        result["oof_fold_col"] = fold
    if cutoff is not None:
        result["source_train_cutoff_col"] = cutoff
    if frozen_bundle_id is not None:
        result["frozen_bundle_id"] = frozen_bundle_id
    return result


def _provenance(
    *,
    output: Path,
    sources: Mapping[str, Path],
    manifests: Mapping[str, Path],
    base_archetypes: Sequence[str],
) -> dict[str, Any]:
    features: dict[str, dict[str, Any]] = {
        "frozen_execution_ev": _feature("execution_ev_prediction", "execution-EV direct all-features OOF", "execution_ev_oof_available_at", "execution_ev_oof_fold", "execution_ev_oof_train_decision_cutoff_utc", cost_spread_aware=True),
        "frozen_ev_map": _feature("execution_ev_mapping", "signed execution-EV OOF map", "execution_ev_map_available_at", "execution_ev_map_oof_fold", "execution_ev_map_train_decision_cutoff_utc"),
        "frozen_alpha": _feature("alpha_outputs", "original alpha OOF", "alpha_available_at", "alpha_oof_fold", "alpha_train_decision_cutoff"),
        "frozen_residual": _feature("residual_outputs", "execution-EV residual all-features OOF", "execution_ev_oof_available_at", "execution_ev_oof_fold", "execution_ev_oof_train_decision_cutoff_utc"),
        "frozen_aux_time": _feature("auxiliary_heads", "time-to-MFE auxiliary OOF", "time_to_mfe_available_at", "time_to_mfe_oof_fold", "time_to_mfe_train_decision_cutoff"),
        "frozen_aux_peak": _feature("auxiliary_heads", "peak-MFE auxiliary OOF", "peak_mfe_available_at", "peak_mfe_oof_fold", "peak_mfe_train_decision_cutoff"),
        "frozen_aux_mae": _feature("auxiliary_heads", "MAE auxiliary OOF", "mae_before_mfe_available_at", "mae_before_mfe_oof_fold", "mae_before_mfe_train_decision_cutoff"),
        "frozen_aux_turn": _feature("auxiliary_heads", "adverse-turn auxiliary OOF", "adverse_turn_available_at", "adverse_turn_oof_fold", "adverse_turn_train_decision_cutoff"),
        "frozen_aux_slope": _feature("auxiliary_heads", "path-slope auxiliary OOF", "path_slope_available_at", "path_slope_oof_fold", "path_slope_train_decision_cutoff"),
        "frozen_entropy": _feature("catboost_entropy", "CatBoost full-vector OOF entropy", "catboost_available_at", "catboost_oof_fold", "catboost_train_decision_cutoff"),
        "frozen_side_is_long": _feature("side_archetypes", "exact canonical side identity", "alpha_available_at", model_input=True, frozen_bundle_id="joined_handoff_base_archetype_contract"),
        "frozen_side_is_short": _feature("side_archetypes", "exact canonical side identity", "alpha_available_at", model_input=True, frozen_bundle_id="joined_handoff_base_archetype_contract"),
    }
    for index in range(len(PATH_SHAPE_TYPES)):
        features[f"frozen_p_{index}"] = _feature("catboost_probabilities", "CatBoost full-vector OOF probability", "catboost_available_at", "catboost_oof_fold", "catboost_train_decision_cutoff")
    for column in base_archetypes:
        # Retain the original, exact base-archetype columns for audit and
        # downstream routing.  Their legacy names contain "label", so they
        # cannot be timing-model inputs under the realised-feature guard.
        features[column] = _feature("side_archetypes", "exact frozen base archetype identity", "alpha_available_at", model_input=False, frozen_bundle_id="joined_handoff_base_archetype_contract")
    return {
        "schema": SCHEMA,
        "handoff": {"join_mode": "exact_inner_one_to_one", "join_keys": list(JOIN_KEYS), "row_count": None},
        "source_artifacts": {name: {"path": str(path.resolve()), "sha256": _sha256(path)} for name, path in sources.items()},
        "source_manifests": {name: {"path": str(path.resolve()), "sha256": _sha256(path)} for name, path in manifests.items()},
        "features": features,
        "timing_labels": {"path_column": "execution_future_path", "atr_column": "atr_1h", "fee_column": "fee", "entry_spread_bps_column": "entry_spread", "exit_spread_bps_column": "exit_spread", "cost_accounting": "fee_once_entry_spread_once_exit_spread_once"},
        "materializer": {"name": Path(__file__).name, "output_sha256": _sha256(output)},
    }


def run(args: argparse.Namespace) -> dict[str, Path]:
    output = args.output
    provenance_path = args.provenance_json or output.with_suffix(".provenance.json")
    if output.exists() or provenance_path.exists():
        raise ValueError("refusing to overwrite an existing timing handoff or provenance JSON")

    handoff = _canonical_frame(args.joined_handoff, source="original execution-EV joined handoff")
    original_provenance = _validate_original_handoff(handoff, args.joined_handoff_provenance)
    oof = _canonical_frame(args.execution_ev_oof, source="execution-EV OOF ledger")
    _validate_execution_ev_run(args.execution_ev_runner_manifest, handoff_path=args.joined_handoff, provenance_path=args.joined_handoff_provenance, oof_path=args.execution_ev_oof)
    _validate_execution_ev_oof(oof)

    ev_map = _canonical_frame(args.execution_ev_map_oof, source="execution-EV map OOF artifact")
    map_manifest = _load_signed_artifact_manifest(args.execution_ev_map_manifest, source="execution-EV map manifest", artifact=args.execution_ev_map_oof, prediction_role=EV_MAP_ROLE)
    _validate_ev_map(ev_map, map_manifest)

    target_manifest = _load_target_manifest(args.execution_ev_target_manifest)
    timing = _canonical_frame(args.timing_labels, source="signed 1m timing labels")
    timing_manifest = _load_signed_artifact_manifest(args.timing_labels_manifest, source="1m timing-label manifest", artifact=args.timing_labels, prediction_role=TIMING_PATH_ROLE, schema=TIMING_PATH_SCHEMA)
    _validate_timing_labels(timing, timing_manifest, target_manifest, args.execution_ev_target_manifest)

    joined = handoff.copy()
    joined = _join(joined, oof, source="execution-EV OOF ledger", columns=["direct__all_features", "residual__all_features", "execution_ev_oof_fold", "execution_ev_oof_train_decision_cutoff_utc", "execution_ev_oof_available_at"])
    joined = _join(joined, ev_map, source="execution-EV map OOF artifact", columns=["mapped_execution_ev", "execution_ev_map_oof_fold", "execution_ev_map_train_decision_cutoff_utc", "execution_ev_map_available_at"])
    joined = _join(joined, timing, source="signed 1m timing labels", columns=["execution_future_path", "atr_1h", "fee", "entry_spread", "exit_spread"])

    _require_columns(joined, ("__decision_ts__", "execution_label_end_utc"), source="original execution-EV joined handoff")
    decision = _utc(joined["__decision_ts__"], source="original execution-EV joined handoff", column="__decision_ts__")
    label_end = _utc(joined["execution_label_end_utc"], source="original execution-EV joined handoff", column="execution_label_end_utc")
    if not (decision == joined["__ts__"] + pd.Timedelta(hours=1)).all() or not (label_end == decision + pd.Timedelta(hours=12)).all():
        raise ValueError("original execution-EV joined handoff: canonical 1h decision / 12h label-end timing is required")

    base_archetypes = sorted(column for column in joined if column.startswith("base_archetype_label__"))
    output_frame = pd.DataFrame({
        "__ts__": joined["__ts__"], "__symbol__": joined["__symbol__"], "side_name": joined["side_name"], "candidate_id": joined["candidate_id"],
        "__decision_ts__": decision, "execution_label_end_utc": label_end,
        "catboost_archetype": joined["catboost_archetype"].astype(str),
        "frozen_execution_ev": joined["direct__all_features"], "frozen_ev_map": joined["mapped_execution_ev"],
        "frozen_alpha": joined["existing_alpha_ev"], "frozen_residual": joined["residual__all_features"],
        "frozen_entropy": joined["catboost_entropy"],
        "frozen_side_is_long": joined["side_name"].eq("long").astype(np.float32),
        "frozen_side_is_short": joined["side_name"].eq("short").astype(np.float32),
        "execution_ev_oof_fold": joined["execution_ev_oof_fold"], "execution_ev_oof_train_decision_cutoff_utc": joined["execution_ev_oof_train_decision_cutoff_utc"], "execution_ev_oof_available_at": joined["execution_ev_oof_available_at"],
        "execution_ev_map_oof_fold": joined["execution_ev_map_oof_fold"], "execution_ev_map_train_decision_cutoff_utc": joined["execution_ev_map_train_decision_cutoff_utc"], "execution_ev_map_available_at": joined["execution_ev_map_available_at"],
        "execution_future_path": joined["execution_future_path"], "atr_1h": joined["atr_1h"], "fee": joined["fee"], "entry_spread": joined["entry_spread"], "exit_spread": joined["exit_spread"],
    })
    for source in _HANDOFF_OOF_SOURCES:
        for suffix in ("oof_fold", "train_decision_cutoff", "available_at"):
            column = f"{source}_{suffix}"
            output_frame[column] = joined[column]
    for raw, output_name in _AUXILIARY_COLUMNS.items():
        output_frame[output_name] = joined[raw]
    for index in range(len(PATH_SHAPE_TYPES)):
        output_frame[f"frozen_p_{index}"] = joined[f"catboost_p_{index}"]
    for column in base_archetypes:
        output_frame[column] = joined[column]
    _assert_unique(output_frame, source="final entry-timing handoff")
    output_frame = output_frame.sort_values(list(JOIN_KEYS), kind="stable").reset_index(drop=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_parquet(output, index=False, compression="zstd")
    provenance = _provenance(
        output=output,
        sources={"joined_handoff": args.joined_handoff, "execution_ev_oof": args.execution_ev_oof, "execution_ev_map_oof": args.execution_ev_map_oof, "timing_labels": args.timing_labels},
        manifests={"joined_handoff_provenance": args.joined_handoff_provenance, "execution_ev_runner": args.execution_ev_runner_manifest, "execution_ev_map": args.execution_ev_map_manifest, "timing_labels": args.timing_labels_manifest, "execution_ev_target": args.execution_ev_target_manifest},
        base_archetypes=base_archetypes,
    )
    provenance["handoff"]["row_count"] = int(len(output_frame))
    provenance_path.write_text(json.dumps(_json_safe(provenance), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"handoff": output, "provenance": provenance_path}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joined-handoff", type=Path, required=True)
    parser.add_argument("--joined-handoff-provenance", type=Path, required=True)
    parser.add_argument("--execution-ev-oof", type=Path, required=True)
    parser.add_argument("--execution-ev-runner-manifest", type=Path, required=True)
    parser.add_argument("--execution-ev-map-oof", type=Path, required=True)
    parser.add_argument("--execution-ev-map-manifest", type=Path, required=True)
    parser.add_argument("--timing-labels", type=Path, required=True)
    parser.add_argument("--timing-labels-manifest", type=Path, required=True)
    parser.add_argument("--execution-ev-target-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance-json", type=Path, default=None)
    return parser


def main() -> None:
    try:
        paths = run(_parser().parse_args())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"execution entry-timing handoff materialization failed: {exc}") from exc
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
