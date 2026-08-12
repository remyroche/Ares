"""Seal the selected leaf-reasoning stack before one untouched final OOS.

This module is intentionally the only fitting boundary immediately before
``leaf_reasoning_final_oos``.  It consumes a *previously immutable*
development selection, never performs feature selection/HPO/clustering, and
fits only rows whose H12 labels had resolved strictly before 2024-11-01.

The resulting directory contains native LightGBM text models, side-local
class-to-net-bps maps, the exact causal feature contract, and the hash-bound
JSON understood by :mod:`extreme_price_movements.leaf_reasoning_final_oos`.
It does not open the November candidate panel; that is solely the replay
consumer's responsibility.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .leaf_reasoning_final_oos import (
    CLASS_ORDER,
    DEVELOPMENT_CUTOFF,
    FrozenArtifact,
    FinalOOSReplayContract,
    FinalOOSReplayError,
    SIDES,
)
from .leaf_reasoning_meta_funnel import (
    CLUSTER_THRESHOLD_BY_ARM,
    ClusterTaxonomyContract,
    FrozenMetaModelSpec,
    MetaFunnelError,
    reject_raw_leaf_columns,
)
from .tp6_portability_data import LABEL_RESOLUTION_HOURS, TP6_SL4_COST_BPS


SCHEMA = "leaf_reasoning_finalizer_v1"
DEVELOPMENT_SELECTION_SCHEMA = "leaf_reasoning_finalizer_selection_v1"
DEVELOPMENT_SELECTION_STATUS = "DEVELOPMENT_SELECTION_FROZEN_FOR_FINALIZATION"
FINALIZATION_STATUS = "SEALED_DEVELOPMENT_ONLY_FINAL_OOS_CONTRACT"
F0_REPRESENTATION = "F0_current_frozen"

# Exact F0 control parameters from ``run_feature_leaf_reasoning_portability``.
# The only final-fit value supplied by the immutable selection is the side
# seed; every other parameter is frozen here, not exposed as a CLI knob.
F0_BASE_PARAMS: Mapping[str, Any] = {
    "n_estimators": 140,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 350,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "reg_lambda": 8.0,
    "n_jobs": 1,
    "verbosity": -1,
}

_BASE_REQUIRED = (
    "candidate_id", "side_name", "decision_ts", "label_available_ts",
    "gross_bps", "net_bps", "r3_class",
)
_META_REQUIRED = (
    "candidate_id", "side_name", "decision_ts", "label_available_ts",
    "base_expected_bps", "realized_net_bps", "base_same_side_strict_oof",
    "base_oof_fit_end_ts", "base_oof_generated_ts",
)
_RESERVED_FEATURES = {
    "candidate_id", "side_name", "decision_ts", "entry_ts", "label_available_ts",
    "gross_bps", "net_bps", "realized_gross_bps", "realized_cost_bps",
    "realized_net_bps", "r3_class", "base_same_side_strict_oof",
    "base_oof_fit_end_ts", "base_oof_generated_ts", "meta_partition", "transport",
}


class LeafReasoningFinalizerError(ValueError):
    """Raised when a final model cannot be proved development-only."""


def _utc(value: object, *, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    if pd.isna(timestamp):
        raise LeafReasoningFinalizerError(f"{name} must be a finite UTC timestamp")
    return timestamp


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return sha256(_canonical_json(value)).hexdigest()


def _ordered_fields(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise LeafReasoningFinalizerError(f"{name} must be an ordered feature list")
    fields = tuple(map(str, value))
    if not fields or any(not item.strip() for item in fields) or len(set(fields)) != len(fields):
        raise LeafReasoningFinalizerError(f"{name} must be non-empty and contain no duplicate/blank fields")
    forbidden = sorted(set(fields).intersection(_RESERVED_FEATURES))
    if forbidden:
        raise LeafReasoningFinalizerError(f"{name} includes target, identity, or unavailable fields: {forbidden}")
    return fields


def _artifact(raw: object, *, root: Path, role: str) -> FrozenArtifact:
    try:
        return FrozenArtifact.from_dict(raw, root=root, role=role)  # type: ignore[arg-type]
    except FinalOOSReplayError as exc:
        raise LeafReasoningFinalizerError(str(exc)) from exc


def _json_artifact(artifact: FrozenArtifact, *, role: str) -> Mapping[str, Any]:
    try:
        value = json.loads(artifact.path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LeafReasoningFinalizerError(f"{role} must be a readable JSON artifact") from exc
    if not isinstance(value, Mapping):
        raise LeafReasoningFinalizerError(f"{role} JSON artifact must be an object")
    if value.get("final_november_oos_consumed") is True:
        raise LeafReasoningFinalizerError(f"{role} has already consumed final November OOS")
    return dict(value)


def _frame_sha256(frame: pd.DataFrame, *, columns: Sequence[str], sort_columns: Sequence[str]) -> str:
    """Fingerprint only the row values actually admitted to final fitting.

    This deliberately excludes source-file bytes so the finalizer need not
    inspect a future partition merely to produce provenance.  Its callers
    supply a pre-cutoff projection; the sorted typed row hash binds precisely
    the values/labels/features used by the respective learner.
    """
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise LeafReasoningFinalizerError(f"cannot fingerprint absent columns: {missing}")
    ordered = frame.loc[:, list(columns)].sort_values(list(sort_columns), kind="stable").reset_index(drop=True)
    row_hash = pd.util.hash_pandas_object(ordered, index=False, categorize=True).to_numpy(np.uint64)
    digest = sha256()
    digest.update(_canonical_json({"columns": list(columns), "dtypes": [str(ordered[item].dtype) for item in columns], "rows": len(ordered)}))
    digest.update(row_hash.tobytes())
    return digest.hexdigest()


def _numeric_matrix(frame: pd.DataFrame, fields: Sequence[str], *, name: str, allow_nan: bool) -> pd.DataFrame:
    missing = sorted(set(fields).difference(frame.columns))
    if missing:
        raise LeafReasoningFinalizerError(f"{name} is missing frozen fields: {missing[:16]}")
    matrix = frame.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce")
    values = matrix.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise LeafReasoningFinalizerError(f"{name} has infinite feature values")
    if not allow_nan and not np.isfinite(values).all():
        raise LeafReasoningFinalizerError(f"{name} has missing/non-finite frozen feature values")
    return matrix.astype(np.float32)


def _validate_h12_cost(frame: pd.DataFrame, *, gross: str, net: str, decision: str, label_available: str, name: str) -> None:
    gross_values = pd.to_numeric(frame[gross], errors="coerce").to_numpy(float)
    net_values = pd.to_numeric(frame[net], errors="coerce").to_numpy(float)
    if not np.isfinite(gross_values).all() or not np.isfinite(net_values).all():
        raise LeafReasoningFinalizerError(f"{name} has non-finite gross/net labels")
    if not np.allclose(gross_values - net_values, TP6_SL4_COST_BPS, rtol=0.0, atol=0.02):
        raise LeafReasoningFinalizerError(f"{name} must charge the fixed 100-bps cost exactly once")
    hours = (frame[label_available] - frame[decision]).dt.total_seconds().to_numpy(float) / 3600.0
    if not np.allclose(hours, LABEL_RESOLUTION_HOURS, rtol=0.0, atol=1e-6):
        raise LeafReasoningFinalizerError(
            f"{name} must use the H12 next-open label resolving {LABEL_RESOLUTION_HOURS:g}h after decision"
        )


@dataclass(frozen=True)
class DevelopmentFinalizationSelection:
    """Immutable selection evidence which the finalizer is allowed to consume."""

    source_path: Path
    source_sha256: str
    selected_arm: str
    successor: str
    base_features_by_side: Mapping[str, tuple[str, ...]]
    base_final_seed_by_side: Mapping[str, int]
    meta_features_by_side: Mapping[str, tuple[str, ...]]
    development_selection: Mapping[str, Any]
    base_selection_artifact: FrozenArtifact
    causal_state_artifacts: tuple[FrozenArtifact, ...]
    frozen_meta_spec: FrozenMetaModelSpec

    @classmethod
    def from_json_path(cls, path: str | Path) -> "DevelopmentFinalizationSelection":
        source = Path(path).resolve()
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LeafReasoningFinalizerError(f"cannot read immutable development selection: {source}") from exc
        if not isinstance(raw, Mapping):
            raise LeafReasoningFinalizerError("immutable development selection must be a JSON object")
        if raw.get("schema") != DEVELOPMENT_SELECTION_SCHEMA:
            raise LeafReasoningFinalizerError(
                f"development selection schema must be {DEVELOPMENT_SELECTION_SCHEMA!r}"
            )
        if raw.get("status") != DEVELOPMENT_SELECTION_STATUS or raw.get("immutable_output") is not True:
            raise LeafReasoningFinalizerError("development selection is not an immutable finalized development decision")
        if raw.get("final_november_oos_consumed") is not False:
            raise LeafReasoningFinalizerError("development selection must retain untouched November OOS")
        end = raw.get("development_evaluation_end_utc")
        if end is None or _utc(end, name="development selection end") != DEVELOPMENT_CUTOFF:
            raise LeafReasoningFinalizerError("development selection must end exactly at the November OOS boundary")
        base = raw.get("base")
        selection = raw.get("development_selection")
        states = raw.get("causal_state_artifacts")
        if not isinstance(base, Mapping) or not isinstance(selection, Mapping):
            raise LeafReasoningFinalizerError("development selection lacks base or final-meta selection blocks")
        if str(base.get("representation", "")) != F0_REPRESENTATION:
            raise LeafReasoningFinalizerError("finalizer only permits the selected frozen F0 base representation")
        base_features_raw = base.get("feature_columns_by_side")
        meta_features_raw = selection.get("selected_meta_features_by_side")
        if not isinstance(base_features_raw, Mapping) or not isinstance(meta_features_raw, Mapping):
            raise LeafReasoningFinalizerError("development selection lacks side-local base/meta feature contracts")
        base_features = {str(side): _ordered_fields(values, name=f"F0 base features/{side}") for side, values in base_features_raw.items()}
        meta_features = {str(side): _ordered_fields(values, name=f"selected meta features/{side}") for side, values in meta_features_raw.items()}
        if set(base_features) != set(SIDES) or set(meta_features) != set(SIDES):
            raise LeafReasoningFinalizerError("base and meta feature contracts must each cover exactly long and short")
        required_meta = {"p_adverse", "p_weak", "p_clear", "base_expected_bps"}
        for side in SIDES:
            missing = sorted(required_meta.difference(meta_features[side]))
            if missing:
                raise LeafReasoningFinalizerError(
                    f"{side} selected meta contract must directly consume same-side base outputs: {missing}"
                )
        try:
            reject_raw_leaf_columns([field for values in meta_features.values() for field in values])
        except MetaFunnelError as exc:
            raise LeafReasoningFinalizerError(str(exc)) from exc
        seeds_raw = base.get("final_seed_by_side")
        if not isinstance(seeds_raw, Mapping) or set(map(str, seeds_raw)) != set(SIDES):
            raise LeafReasoningFinalizerError("immutable F0 selection must declare exactly one final seed per side")
        try:
            seeds = {side: int(seeds_raw[side]) for side in SIDES}
        except (TypeError, ValueError, KeyError) as exc:
            raise LeafReasoningFinalizerError("F0 final seeds must be integers") from exc
        artifacts = selection
        for key in (
            "selection_artifact", "feature_group_artifact", "taxonomy_artifact",
            "successor_decision_artifact", "frozen_meta_model_spec_artifact",
        ):
            if key not in artifacts:
                raise LeafReasoningFinalizerError(f"development selection lacks {key}")
        root = source.parent
        base_selection = _artifact(base.get("selection_artifact"), root=root, role="F0 base selection")
        base_selection_payload = _json_artifact(base_selection, role="F0 base selection")
        if base_selection_payload.get("development_only") is not True or base_selection_payload.get("final_november_oos_consumed") is not False:
            raise LeafReasoningFinalizerError("F0 base selection must be development-only and retain November")
        base_end = base_selection_payload.get("development_evaluation_end_utc")
        if base_end is None or _utc(base_end, name="F0 base selection end") != DEVELOPMENT_CUTOFF:
            raise LeafReasoningFinalizerError("F0 base selection must end exactly at the November boundary")
        winner = base_selection_payload.get("winner", base_selection_payload.get("selected_representation"))
        if winner is not None and str(winner) != F0_REPRESENTATION:
            raise LeafReasoningFinalizerError("F0 base selection artifact does not bind F0_current_frozen")
        spec_artifact = _artifact(artifacts["frozen_meta_model_spec_artifact"], root=root, role="frozen meta model spec")
        spec_payload = _json_artifact(spec_artifact, role="frozen meta model spec")
        try:
            frozen_spec = FrozenMetaModelSpec(
                family=str(spec_payload["family"]), params=dict(spec_payload["params"]),
                contract_id=str(spec_payload["contract_id"]),
            )
        except (KeyError, TypeError, MetaFunnelError) as exc:
            raise LeafReasoningFinalizerError("selection lacks a valid frozen LightGBM Huber meta spec") from exc
        if not isinstance(states, Sequence) or isinstance(states, (str, bytes)) or not states:
            raise LeafReasoningFinalizerError("development selection must bind non-empty frozen causal state artifacts")
        state_artifacts = tuple(
            _artifact(item, root=root, role=f"causal state {index}") for index, item in enumerate(states)
        )
        # The selection section is consumed verbatim by the final replay after
        # all artifact paths are resolved/hashes checked below.
        expected_transports = {
            "transport_a_2023q4_to_2024h1", "transport_b_2024h1_to_2024h2_to_date",
        }
        transports = selection.get("development_transports")
        if not isinstance(transports, Sequence) or isinstance(transports, (str, bytes)) or set(map(str, transports)) != expected_transports or len(transports) != 2:
            raise LeafReasoningFinalizerError("finalization requires exactly both declared development transports")
        if selection.get("selected_arm") is None or selection.get("successor") is None:
            raise LeafReasoningFinalizerError("development selection must bind selected meta arm and successor")
        if selection.get("development_evaluation_end_utc") is None or _utc(selection["development_evaluation_end_utc"], name="final-meta selection end") != DEVELOPMENT_CUTOFF:
            raise LeafReasoningFinalizerError("final-meta selection must end exactly at the November boundary")
        if selection.get("final_november_oos_consumed") is not False:
            raise LeafReasoningFinalizerError("final-meta selection must explicitly retain November")
        # Validate every development-only decision before any native model is
        # fitted.  The final replay repeats these checks, but waiting until
        # after fitting would allow a malformed selection to create misleading
        # model files even though it cannot be sealed for OOS.
        bound = {
            key: _artifact(selection[key], root=root, role=role)
            for key, role in {
                "selection_artifact": "development selection",
                "feature_group_artifact": "feature group",
                "taxonomy_artifact": "cluster taxonomy",
                "successor_decision_artifact": "successor decision",
            }.items()
        }
        payloads = {key: _json_artifact(value, role=value.role) for key, value in bound.items()}
        for key, payload in payloads.items():
            if payload.get("development_only") is not True or payload.get("final_november_oos_consumed") is not False:
                raise LeafReasoningFinalizerError(f"{key} must be development-only and retain November")
            artifact_end = payload.get("development_evaluation_end_utc", payload.get("evaluation_end_utc"))
            if artifact_end is None or _utc(artifact_end, name=f"{key} evaluation end") != DEVELOPMENT_CUTOFF:
                raise LeafReasoningFinalizerError(f"{key} must end exactly at the November boundary")
        selected_payload = payloads["selection_artifact"]
        if selected_payload.get("selected_arm") != selection.get("selected_arm") or str(selected_payload.get("successor", "")).upper() != str(selection.get("successor", "")).upper():
            raise LeafReasoningFinalizerError("development selection artifact differs from selected meta arm/successor")
        feature_payload = payloads["feature_group_artifact"]
        selected_features = feature_payload.get("selected_meta_features_by_side", feature_payload.get("feature_contract"))
        normalized_selected = {
            str(side): tuple(map(str, values))
            for side, values in selected_features.items()
        } if isinstance(selected_features, Mapping) else {}
        if normalized_selected != meta_features or feature_payload.get("selected_arm") != selection.get("selected_arm"):
            raise LeafReasoningFinalizerError("feature-group artifact differs from the selected side-local meta contract")
        successor_payload = payloads["successor_decision_artifact"]
        if str(successor_payload.get("successor", "")).upper() != str(selection.get("successor", "")).upper() or not str(successor_payload.get("terminal_decision", "")).strip():
            raise LeafReasoningFinalizerError("successor decision artifact does not bind the selected terminal S generation")
        taxonomy_payload = payloads["taxonomy_artifact"]
        try:
            thresholds = {str(key): float(value) for key, value in dict(taxonomy_payload.get("threshold_by_arm", {})).items()}
            if thresholds != dict(CLUSTER_THRESHOLD_BY_ARM):
                raise MetaFunnelError("taxonomy threshold grid differs")
            ClusterTaxonomyContract(
                linkage=str(taxonomy_payload.get("linkage", "")),
                cluster_ids_by_arm=taxonomy_payload["cluster_ids_by_arm"],
                threshold_by_arm=thresholds,
                c5_source_arm=str(taxonomy_payload.get("c5_source_arm", "C1")),
                c6_source_arm=str(taxonomy_payload.get("c6_source_arm", "C5")),
                top_decile_coverage_target=float(taxonomy_payload.get("top_decile_coverage_target", .95)),
                top_decile_coverage_by_arm=taxonomy_payload.get("top_decile_coverage_by_arm", {}),
                portable_top_decile_coverage_by_arm=taxonomy_payload.get("portable_top_decile_coverage_by_arm", {}),
                production_soft_cap=int(taxonomy_payload.get("production_soft_cap", 12)),
                exploratory_hard_cap=int(taxonomy_payload.get("exploratory_hard_cap", 20)),
                c6_best_cross_era_score=taxonomy_payload.get("c6_best_cross_era_score"),
                c6_best_cross_era_standard_error=taxonomy_payload.get("c6_best_cross_era_standard_error"),
                c6_compact_cross_era_score=taxonomy_payload.get("c6_compact_cross_era_score"),
            )
        except (KeyError, TypeError, ValueError, MetaFunnelError) as exc:
            raise LeafReasoningFinalizerError("taxonomy artifact lacks the frozen C1--C6 threshold/coverage/one-SE contract") from exc
        return cls(
            source_path=source,
            source_sha256=_sha256_file(source),
            selected_arm=str(selection["selected_arm"]),
            successor=str(selection["successor"]).upper(),
            base_features_by_side=base_features,
            base_final_seed_by_side=seeds,
            meta_features_by_side=meta_features,
            development_selection=dict(selection),
            base_selection_artifact=base_selection,
            causal_state_artifacts=state_artifacts,
            frozen_meta_spec=frozen_spec,
        )

    def resolved_development_selection(self) -> dict[str, Any]:
        """Return the replay's selected-development block with absolute bindings."""
        selection = dict(self.development_selection)
        role_by_key = {
            "selection_artifact": "development selection",
            "feature_group_artifact": "feature group",
            "taxonomy_artifact": "cluster taxonomy",
            "successor_decision_artifact": "successor decision",
            "frozen_meta_model_spec_artifact": "frozen meta model spec",
        }
        root = self.source_path.parent
        for key, role in role_by_key.items():
            selection[key] = _artifact(selection[key], root=root, role=role).to_dict()
        selection["selected_meta_features_by_side"] = {
            side: list(self.meta_features_by_side[side]) for side in SIDES
        }
        selection["successor"] = self.successor
        return selection


@dataclass(frozen=True)
class FinalizationResult:
    output_dir: Path
    frozen_contract_path: Path
    contract_sha256: str
    manifest: Mapping[str, Any]


def _prepare_base_training(frame: pd.DataFrame, selection: DevelopmentFinalizationSelection) -> pd.DataFrame:
    required = set(_BASE_REQUIRED)
    for fields in selection.base_features_by_side.values():
        required.update(fields)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise LeafReasoningFinalizerError(f"base final-training panel lacks required fields: {missing[:20]}")
    optional_weight_fields = [
        field for field in (
            "robust_clear_event_b0", "robust_clear_event_b25", "robust_clear_event_b50", "f0_sample_weight",
        ) if field in frame.columns
    ]
    work = frame.loc[:, list(dict.fromkeys([
        *_BASE_REQUIRED, *optional_weight_fields,
        *[field for side in SIDES for field in selection.base_features_by_side[side]],
    ]))].copy()
    work["side_name"] = work["side_name"].astype("string").str.lower()
    if not set(work["side_name"].dropna()).issubset(SIDES) or set(work["side_name"].dropna()) != set(SIDES):
        raise LeafReasoningFinalizerError("base final-training panel must contain both canonical sides")
    for field in ("decision_ts", "label_available_ts"):
        work[field] = pd.to_datetime(work[field], utc=True, errors="coerce")
    if work[["decision_ts", "label_available_ts"]].isna().any().any():
        raise LeafReasoningFinalizerError("base final-training panel has invalid UTC timestamps")
    # The dataframe API is deliberately fail-closed: callers must not pass a
    # panel that contains November rows.  The CLI reads a pre-cutoff parquet
    # projection, so no November labels/features are read in the first place.
    if not work["decision_ts"].lt(DEVELOPMENT_CUTOFF).all() or not work["label_available_ts"].lt(DEVELOPMENT_CUTOFF).all():
        raise LeafReasoningFinalizerError("base final-training panel contains rows at/after the untouched November cutoff")
    if not work["label_available_ts"].gt(work["decision_ts"]).all():
        raise LeafReasoningFinalizerError("base labels must resolve after their decision")
    _validate_h12_cost(work, gross="gross_bps", net="net_bps", decision="decision_ts", label_available="label_available_ts", name="base final-training panel")
    classes = pd.to_numeric(work["r3_class"], errors="coerce")
    if classes.isna().any() or not np.isin(classes.to_numpy(int), (0, 1, 2)).all():
        raise LeafReasoningFinalizerError("base final-training panel must carry canonical R3 adverse/weak/clear classes")
    work["r3_class"] = classes.astype(np.int8)
    if work.duplicated(["candidate_id", "side_name", "decision_ts"]).any():
        raise LeafReasoningFinalizerError("base final-training panel has duplicate side-qualified candidate rows")
    for side in SIDES:
        local = work.loc[work["side_name"].eq(side)]
        if local.empty:
            raise LeafReasoningFinalizerError(f"base final-training panel has no {side} rows")
        # F0's native LightGBM missing-value routing is retained.  Infinite
        # values are never accepted; coverage is written into the contract.
        _numeric_matrix(local, selection.base_features_by_side[side], name=f"F0 base/{side}", allow_nan=True)
    return work.sort_values(["decision_ts", "side_name", "candidate_id"], kind="stable").reset_index(drop=True)


def _prepare_meta_training(frame: pd.DataFrame, selection: DevelopmentFinalizationSelection) -> pd.DataFrame:
    required = set(_META_REQUIRED)
    for fields in selection.meta_features_by_side.values():
        required.update(fields)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise LeafReasoningFinalizerError(f"strict meta ledger lacks required fields: {missing[:20]}")
    fields = list(dict.fromkeys([*_META_REQUIRED, *[field for side in SIDES for field in selection.meta_features_by_side[side]]]))
    work = frame.loc[:, fields].copy()
    work["side_name"] = work["side_name"].astype("string").str.lower()
    if set(work["side_name"].dropna()) != set(SIDES):
        raise LeafReasoningFinalizerError("strict meta ledger must contain both canonical sides")
    for field in ("decision_ts", "label_available_ts", "base_oof_fit_end_ts", "base_oof_generated_ts"):
        work[field] = pd.to_datetime(work[field], utc=True, errors="coerce")
    if work[["decision_ts", "label_available_ts", "base_oof_fit_end_ts", "base_oof_generated_ts"]].isna().any().any():
        raise LeafReasoningFinalizerError("strict meta ledger has invalid UTC provenance timestamps")
    if not work["decision_ts"].lt(DEVELOPMENT_CUTOFF).all() or not work["label_available_ts"].lt(DEVELOPMENT_CUTOFF).all():
        raise LeafReasoningFinalizerError("strict meta ledger contains rows at/after the untouched November cutoff")
    if not work["base_same_side_strict_oof"].fillna(False).astype(bool).all():
        raise LeafReasoningFinalizerError("final meta fit requires same-side strict OOF base scores on every row")
    if not work["base_oof_fit_end_ts"].lt(work["decision_ts"]).all():
        raise LeafReasoningFinalizerError("meta ledger base fit must strictly precede each decision")
    if not work["base_oof_generated_ts"].le(work["decision_ts"]).all():
        raise LeafReasoningFinalizerError("meta ledger base score was generated after its decision")
    if not work["base_oof_fit_end_ts"].lt(work["base_oof_generated_ts"]).all():
        raise LeafReasoningFinalizerError("meta ledger base score has invalid fit/generation ordering")
    for field in ("base_expected_bps", "realized_net_bps"):
        values = pd.to_numeric(work[field], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise LeafReasoningFinalizerError(f"strict meta ledger has non-finite {field}")
        work[field] = values.astype(np.float32)
    if work.duplicated(["candidate_id", "side_name", "decision_ts"]).any():
        raise LeafReasoningFinalizerError("strict meta ledger has duplicate side-qualified candidate rows")
    for side in SIDES:
        local = work.loc[work["side_name"].eq(side)]
        if local.empty:
            raise LeafReasoningFinalizerError(f"strict meta ledger has no {side} rows")
        _numeric_matrix(local, selection.meta_features_by_side[side], name=f"final meta/{side}", allow_nan=True)
    return work.sort_values(["decision_ts", "side_name", "candidate_id"], kind="stable").reset_index(drop=True)


def _fit_base_model(train: pd.DataFrame, fields: Sequence[str], *, seed: int):
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise LeafReasoningFinalizerError("LightGBM is required to seal native final base models") from exc
    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=3, random_state=int(seed), **dict(F0_BASE_PARAMS)
    )
    matrix = _numeric_matrix(train, fields, name="F0 final base model", allow_nan=True)
    labels = train["r3_class"].to_numpy(np.int8)
    # This is deliberately the same certainty/class-support weighting used by
    # the frozen F0 base pipeline when its three robust-clear definitions are
    # available.  Finalization can only consume the canonical R3 source; it
    # refuses a silently substituted target/weighting scheme.
    if not {"robust_clear_event_b0", "robust_clear_event_b25", "robust_clear_event_b50"}.issubset(train.columns):
        # The per-row training panel used by F0 normally carries these flags.
        # A selection that materialised a canonical R3-only final panel may
        # bind its precomputed weights explicitly instead.
        if "f0_sample_weight" not in train.columns:
            raise LeafReasoningFinalizerError(
                "F0 final-training panel must retain robust-clear agreement flags or immutable f0_sample_weight"
            )
        weight = pd.to_numeric(train["f0_sample_weight"], errors="coerce").to_numpy(float)
        if not np.isfinite(weight).all() or (weight <= 0.0).any():
            raise LeafReasoningFinalizerError("immutable f0_sample_weight must be finite and positive")
    else:
        agreement = train[["robust_clear_event_b0", "robust_clear_event_b25", "robust_clear_event_b50"]].nunique(axis=1).eq(1).to_numpy(float)
        certainty = 0.5 + 0.5 * agreement
        counts = np.bincount(labels, minlength=3).astype(float)
        class_weight = np.sqrt(len(train) / np.maximum(counts, 1.0))[labels]
        class_weight /= max(float(class_weight.mean()), 1e-12)
        weight = np.clip(certainty * class_weight, 0.25, 4.0)
        weight /= max(float(weight.mean()), 1e-12)
    model.fit(matrix, labels, sample_weight=weight)
    if not np.array_equal(np.asarray(model.classes_, dtype=np.int8), np.array([0, 1, 2], dtype=np.int8)):
        raise LeafReasoningFinalizerError("F0 final base model class order must be adverse=0, weak=1, clear=2")
    return model, weight.astype(np.float32)


def _fit_meta_model(train: pd.DataFrame, fields: Sequence[str], *, spec: FrozenMetaModelSpec):
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise LeafReasoningFinalizerError("LightGBM is required to seal native final meta models") from exc
    matrix = _numeric_matrix(train, fields, name="final Huber residual meta model", allow_nan=True)
    target = train["realized_net_bps"].to_numpy(np.float32) - train["base_expected_bps"].to_numpy(np.float32)
    if not np.isfinite(target).all():
        raise LeafReasoningFinalizerError("final Huber residual target must be finite")
    model = lgb.LGBMRegressor(**dict(spec.params))
    model.fit(matrix, target)
    return model, target


def _model_text(model: Any, path: Path, *, role: str) -> None:
    booster = getattr(model, "booster_", None)
    if booster is None:
        raise LeafReasoningFinalizerError(f"fitted {role} model has no native LightGBM booster")
    booster.save_model(str(path))
    if not path.is_file() or path.stat().st_size <= 0:
        raise LeafReasoningFinalizerError(f"native {role} model was not written")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _artifact_decl(path: Path, *, fit_end: pd.Timestamp, relative_to: Path | None = None) -> dict[str, str]:
    declared_path = path.relative_to(relative_to).as_posix() if relative_to is not None else str(path.resolve())
    return {
        "path": declared_path,
        "sha256": _sha256_file(path),
        "fit_end_utc": _utc(fit_end, name="artifact fit end").isoformat(),
    }


def _feature_coverage(frame: pd.DataFrame, *, side: str, layer: str, fields: Sequence[str]) -> list[dict[str, Any]]:
    matrix = _numeric_matrix(frame, fields, name=f"{layer} coverage/{side}", allow_nan=True)
    finite = np.isfinite(matrix.to_numpy(float))
    return [
        {
            "side_name": side, "layer": layer, "feature": field,
            "rows": int(len(frame)), "finite_rows": int(finite[:, index].sum()),
            "finite_coverage": float(finite[:, index].mean()),
            "unique_finite_values": int(np.unique(matrix.iloc[:, index].to_numpy(float)[finite[:, index]]).size),
        }
        for index, field in enumerate(fields)
    ]


def _checksums(root: Path) -> dict[str, str]:
    return {
        item.relative_to(root).as_posix(): _sha256_file(item)
        for item in sorted(root.rglob("*"))
        if item.is_file() and item.name not in {"checksums.json", "run_manifest.json"}
    }


def finalize_leaf_reasoning_final_oos(
    selection: DevelopmentFinalizationSelection,
    base_training: pd.DataFrame,
    meta_ledger: pd.DataFrame,
    *,
    output_dir: str | Path,
) -> FinalizationResult:
    """Fit once through October and seal a replay-ready final-OOS contract.

    ``selection`` must already have selected all feature fields, arm,
    successor, model specification, clustering/taxonomy evidence, and causal
    state artifacts on development transports.  This function has no knobs
    for any of those decisions and rejects a destination that already exists.
    """
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite final-OOS finalization: {output}")
    base = _prepare_base_training(base_training, selection)
    meta = _prepare_meta_training(meta_ledger, selection)
    # Source data fingerprints intentionally cover only values admitted before
    # the cutoff.  No November row is consulted for selection, fitting, map
    # construction, coverage, or provenance.
    base_fingerprint_columns = list(base.columns)
    meta_fingerprint_columns = list(meta.columns)
    base_sha = _frame_sha256(base, columns=base_fingerprint_columns, sort_columns=("decision_ts", "side_name", "candidate_id"))
    meta_sha = _frame_sha256(meta, columns=meta_fingerprint_columns, sort_columns=("decision_ts", "side_name", "candidate_id"))
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=parent))
    published = False
    try:
        model_root = staging / "models"
        map_root = staging / "value_maps"
        model_root.mkdir()
        map_root.mkdir()
        scoring: dict[str, dict[str, Any]] = {}
        coverage_rows: list[dict[str, Any]] = []
        per_side: dict[str, Any] = {}
        for side in SIDES:
            base_rows = base.loc[base["side_name"].eq(side)].copy()
            meta_rows = meta.loc[meta["side_name"].eq(side)].copy()
            base_model, sample_weight = _fit_base_model(
                base_rows, selection.base_features_by_side[side], seed=selection.base_final_seed_by_side[side]
            )
            meta_model, residual_target = _fit_meta_model(
                meta_rows, selection.meta_features_by_side[side], spec=selection.frozen_meta_spec
            )
            base_model_path = model_root / f"base_{side}.txt"
            meta_model_path = model_root / f"meta_{side}.txt"
            _model_text(base_model, base_model_path, role=f"{side} F0 base")
            _model_text(meta_model, meta_model_path, role=f"{side} Huber meta")
            base_fit_end = base_rows["label_available_ts"].max()
            meta_fit_end = meta_rows["label_available_ts"].max()
            class_counts = base_rows["r3_class"].value_counts().reindex((0, 1, 2), fill_value=0)
            global_mean = float(base_rows["net_bps"].mean())
            class_map = {
                CLASS_ORDER[index]: float(base_rows.loc[base_rows["r3_class"].eq(index), "net_bps"].mean())
                if bool(base_rows["r3_class"].eq(index).any()) else global_mean
                for index in range(3)
            }
            if not np.isfinite(np.asarray(list(class_map.values()), dtype=float)).all():
                raise LeafReasoningFinalizerError(f"{side} class-to-net-bps map is non-finite")
            map_path = map_root / f"base_class_value_map_{side}.json"
            _write_json(map_path, {
                "schema": SCHEMA,
                "status": "FROZEN_SIDE_LOCAL_BASE_CLASS_VALUE_MAP",
                "development_only": True,
                "final_november_oos_consumed": False,
                "side_name": side,
                "fit_end_utc": base_fit_end.isoformat(),
                "label_contract": "TP6/SL4/R3 H12 from next-hourly open; fixed 100-bps cost exactly once",
                "class_order": list(CLASS_ORDER),
                "class_expected_net_bps": class_map,
                "class_counts": {CLASS_ORDER[index]: int(class_counts.iloc[index]) for index in range(3)},
                "base_training_data_sha256": base_sha,
                "base_feature_columns": list(selection.base_features_by_side[side]),
            })
            scoring[side] = {
                "base_model": _artifact_decl(base_model_path, fit_end=base_fit_end, relative_to=staging),
                "base_feature_columns": list(selection.base_features_by_side[side]),
                "base_value_map": _artifact_decl(map_path, fit_end=base_fit_end, relative_to=staging),
                "meta_model": _artifact_decl(meta_model_path, fit_end=meta_fit_end, relative_to=staging),
                "meta_feature_columns": list(selection.meta_features_by_side[side]),
            }
            coverage_rows.extend(_feature_coverage(base_rows, side=side, layer="base", fields=selection.base_features_by_side[side]))
            coverage_rows.extend(_feature_coverage(meta_rows, side=side, layer="meta", fields=selection.meta_features_by_side[side]))
            per_side[side] = {
                "base_training_rows": int(len(base_rows)),
                "meta_training_rows": int(len(meta_rows)),
                "base_fit_end_utc": base_fit_end.isoformat(),
                "meta_fit_end_utc": meta_fit_end.isoformat(),
                "base_class_counts": {CLASS_ORDER[index]: int(class_counts.iloc[index]) for index in range(3)},
                "f0_sample_weight_min": float(sample_weight.min()),
                "f0_sample_weight_max": float(sample_weight.max()),
                "meta_residual_target_mean_bps": float(residual_target.mean()),
                "meta_residual_target_std_bps": float(residual_target.std()),
            }
        feature_contract = {
            "schema": SCHEMA,
            "status": "FROZEN_CAUSAL_FINAL_OOS_FEATURE_CONTRACT",
            "development_only": True,
            "final_november_oos_consumed": False,
            "development_cutoff_utc": DEVELOPMENT_CUTOFF.isoformat(),
            "base_representation": F0_REPRESENTATION,
            "base_feature_columns_by_side": {side: list(selection.base_features_by_side[side]) for side in SIDES},
            "meta_feature_columns_by_side": {side: list(selection.meta_features_by_side[side]) for side in SIDES},
            "base_to_meta": "same-side p_adverse/p_weak/p_clear/base_expected_bps are recomputed from frozen native base model and map",
            "meta_input_policy": "selected compact causal/context/health/cluster fields only; raw leaf identifiers forbidden",
            "causal_state_artifacts": [artifact.to_dict() for artifact in selection.causal_state_artifacts],
            "availability_requirement": "all features and causal state must be available at or before each decision_ts",
            "entry_label_contract": "candidate bar close decision -> next hourly open; H12 label resolves 13h after decision; cost 100 bps once",
        }
        feature_contract_path = staging / "causal_feature_contract.json"
        _write_json(feature_contract_path, feature_contract)
        feature_contract_sha = _sha256_file(feature_contract_path)
        finalization_provenance = {
            "schema": SCHEMA,
            "development_cutoff_utc": DEVELOPMENT_CUTOFF.isoformat(),
            "development_selection_path": str(selection.source_path),
            "development_selection_sha256": selection.source_sha256,
            "base_selection_artifact_sha256": selection.base_selection_artifact.sha256,
            "base_training_data_sha256": base_sha,
            "meta_training_data_sha256": meta_sha,
            "causal_feature_contract_sha256": feature_contract_sha,
            "base_representation": F0_REPRESENTATION,
            "base_model_params": dict(F0_BASE_PARAMS),
            "frozen_meta_model": {
                "family": selection.frozen_meta_spec.family,
                "contract_id": selection.frozen_meta_spec.contract_id,
                "params": dict(selection.frozen_meta_spec.params),
                "params_hash": selection.frozen_meta_spec.params_hash,
            },
            "strict_meta_training": "all rows use same-side strict OOF base scores; base fit < decision and generation <= decision",
            "final_oos_not_opened": True,
            "per_side": per_side,
        }
        provenance_path = staging / "training_provenance.json"
        _write_json(provenance_path, finalization_provenance)
        contract_payload = {
            "schema": "leaf_reasoning_final_oos_replay_v1",
            "status": "DEVELOPMENT_SELECTED_FROZEN_FINAL_OOS_CONTRACT",
            "final_november_oos_consumed": False,
            "development_selection": selection.resolved_development_selection(),
            "scoring": scoring,
            "causal_state_artifacts": [artifact.to_dict() for artifact in selection.causal_state_artifacts],
            "finalization_provenance": finalization_provenance,
        }
        contract_path = staging / "frozen_final_oos_contract.json"
        _write_json(contract_path, contract_payload)
        # Re-read via the actual consumer schema before publishing.  This
        # proves model/map hashes, selection bindings, fixed taxonomy grid,
        # explicit cutoff, and side-local feature equality all at once.
        try:
            contract = FinalOOSReplayContract.from_json_path(contract_path)
        except FinalOOSReplayError as exc:
            raise LeafReasoningFinalizerError(f"sealed final-OOS contract failed replay validation: {exc}") from exc
        # Validate the relative-path contract while the atomic staging tree is
        # still intact.  After publication it is reloaded once more below so
        # its hash is bound to the final destination rather than this temporary
        # staging pathname.
        pd.DataFrame(coverage_rows).to_parquet(staging / "training_feature_coverage.parquet", index=False, compression="zstd")
        os.replace(staging, output)
        published = True
        try:
            contract = FinalOOSReplayContract.from_json_path(output / "frozen_final_oos_contract.json")
        except FinalOOSReplayError as exc:  # pragma: no cover - staging validation already exercised
            raise LeafReasoningFinalizerError(f"published final-OOS contract failed replay validation: {exc}") from exc
        # ``FrozenArtifact`` normalises paths to absolute paths.  Rewrite the
        # final contract only after its directory has its permanent name so
        # the one-time replay registry and manifest share a stable contract
        # hash across reloads.
        _write_json(output / "frozen_final_oos_contract.json", contract.to_dict())
        try:
            contract = FinalOOSReplayContract.from_json_path(output / "frozen_final_oos_contract.json")
        except FinalOOSReplayError as exc:  # pragma: no cover - defensive path-stability check
            raise LeafReasoningFinalizerError(f"permanent final-OOS contract failed replay validation: {exc}") from exc
        manifest = {
            "schema": SCHEMA,
            "status": FINALIZATION_STATUS,
            "immutable_output": True,
            "final_november_oos_consumed": False,
            "development_cutoff_utc": DEVELOPMENT_CUTOFF.isoformat(),
            "final_oos_window": "2024-11-01T00:00:00Z..2024-12-01T00:00:00Z (not opened)",
            "frozen_contract": "frozen_final_oos_contract.json",
            "frozen_contract_sha256": contract.sha256,
            "causal_feature_contract": "causal_feature_contract.json",
            "training_provenance": "training_provenance.json",
            "native_models": {side: {"base": f"models/base_{side}.txt", "meta": f"models/meta_{side}.txt"} for side in SIDES},
            "class_value_maps": {side: f"value_maps/base_class_value_map_{side}.json" for side in SIDES},
            "selection_sha256": selection.source_sha256,
            "base_training_data_sha256": base_sha,
            "meta_training_data_sha256": meta_sha,
            "no_final_oos_labels_or_features_read_for_selection_or_fitting": True,
            "no_hpo_feature_selection_or_policy_tuning_in_finalizer": True,
            "global_ranking": "deferred to final replay: one pooled cross-side common-bps ranking after base map plus meta residual",
        }
        manifest["checksums"] = _checksums(output)
        _write_json(output / "checksums.json", manifest["checksums"])
        _write_json(output / "run_manifest.json", manifest)
        return FinalizationResult(
            output_dir=output,
            frozen_contract_path=output / "frozen_final_oos_contract.json",
            contract_sha256=contract.sha256,
            manifest=manifest,
        )
    except Exception:
        shutil.rmtree(output if published else staging, ignore_errors=True)
        raise


def read_pre_cutoff_parquet(path: str | Path, *, columns: Sequence[str]) -> pd.DataFrame:
    """Read only pre-November, already-resolved rows from a parquet source.

    This is the CLI-safe ingestion boundary.  Parquet predicate pushdown is
    intentional: a source may be physically partitioned beyond October, but
    the finalizer does not materialise November feature/label rows merely to
    reject them.
    """
    source = Path(path)
    if source.suffix.lower() != ".parquet":
        raise LeafReasoningFinalizerError("finalizer accepts parquet training inputs only")
    requested = list(dict.fromkeys([*columns, "decision_ts", "label_available_ts"]))
    try:
        return pd.read_parquet(
            source,
            columns=requested,
            filters=[
                ("decision_ts", "<", DEVELOPMENT_CUTOFF.to_pydatetime()),
                ("label_available_ts", "<", DEVELOPMENT_CUTOFF.to_pydatetime()),
            ],
        )
    except Exception as exc:
        raise LeafReasoningFinalizerError(f"could not read pre-cutoff parquet projection: {source}") from exc


__all__ = [
    "DEVELOPMENT_SELECTION_SCHEMA", "DEVELOPMENT_SELECTION_STATUS", "F0_BASE_PARAMS",
    "F0_REPRESENTATION", "FINALIZATION_STATUS", "LeafReasoningFinalizerError",
    "DevelopmentFinalizationSelection", "FinalizationResult", "finalize_leaf_reasoning_final_oos",
    "read_pre_cutoff_parquet",
]
