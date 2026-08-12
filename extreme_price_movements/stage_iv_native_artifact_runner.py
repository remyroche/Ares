"""Artifact-producing Stage-IV broad-to-tail native-score runner.

This is the active direct-FQ3 architecture.  Broad and tail base models emit
same-side native scores and probability states.  The meta model classifies
fold-local correctness terciles from the tail score, reconstructs a joint
native score, and only then applies causal side-local 21-day common-bps maps.
The historical ``net_bps - tail_score`` residual is neither accepted nor
constructed here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

from .stage_i_causal_admission import (
    Causal21dAdmissionSpec,
    apply_causal_21d_side_admission,
)
from .stage_i_strict_oof import _multiclass_probabilities
from .stage_i_target_specific_oos import (
    DIRECT_BASE_INPUT_SEMANTICS,
    DIRECT_FQ3_SEMANTICS,
    _fit_direct_correctness,
    _reconstruct_direct_correctness,
)
from .stage_iv_broad_to_tail import (
    TAIL_FRACTIONS,
    _design,
    _strict_blocks,
    prequential_tail_handoff,
)


SCHEMA = "stage_iv_native_broad_tail_artifact_runner_v1"
FROZEN_MODEL_SERIALIZATION_FORMAT = "joblib_sha256_verified_v1"
_ROUTES = frozenset({"neither", "tail", "meta", "both"})
_REQUIRED_REPORT_TAILS = (0.01, 0.05, 0.10, 0.20)
_RESERVED_NATIVE_INPUT = re.compile(
    r"^(?:"
    r"(?:exact_)?(?:net|gross)(?:_|$)|"
    r"(?:target|label|event|outcome|pnl|mfe|mae|future|path|barrier|exit)(?:_|$)|"
    r"(?:prequential|causal_21d|mapped|expected_net|meta_direct|meta_p_)(?:_|$)"
    r")",
    flags=re.IGNORECASE,
)


class StageIVNativeRunnerError(ValueError):
    """A native-score, chronology, or publication contract was violated."""


@dataclass(frozen=True)
class NativeBasePrediction:
    score: Sequence[float]
    states: Sequence[Sequence[float]]


class NativeBasePredictor(Protocol):
    def predict_native(self, frame: pd.DataFrame) -> NativeBasePrediction: ...


class DirectMetaPredictor(Protocol):
    def predict_proba(self, frame: pd.DataFrame) -> Any: ...


NativeBaseFitter = Callable[
    [pd.DataFrame, np.ndarray, np.ndarray, str, Mapping[str, Any]],
    NativeBasePredictor,
]
DirectMetaFitter = Callable[
    [pd.DataFrame, np.ndarray, np.ndarray, str, Mapping[str, Any]],
    DirectMetaPredictor,
]


@dataclass(frozen=True)
class StageIVNativePlan:
    side: str
    candidate_ids: Sequence[Any]
    symbols: Sequence[Any]
    frame: pd.DataFrame
    base_target: Sequence[float]
    exact_net_bps: Sequence[float]
    decision_timestamps: Sequence[Any]
    label_available_timestamps: Sequence[Any]
    broad_feature_names: tuple[str, ...]
    tail_feature_names: tuple[str, ...]
    meta_feature_names: tuple[str, ...]
    broad_params: Mapping[str, Any] = field(default_factory=dict)
    tail_params: Mapping[str, Any] = field(default_factory=dict)
    meta_params: Mapping[str, Any] = field(default_factory=dict)
    tail_fraction: float = 0.30
    broad_min_train_rows: int = 500
    tail_min_train_rows: int = 500
    meta_min_train_rows: int = 500
    min_handoff_history_rows: int = 100
    n_validation_folds: int = 4
    # The native runner predates the strict research path.  Preserve its
    # previous no-calendar-burn behaviour unless a caller opts in explicitly;
    # the new broad→tail runner uses the mandatory two-month default instead.
    burn_in_months: int = 0
    broad_output_route: str = "both"
    score_domain: tuple[float, float] = (-1.0, 1.0)
    sample_weight: Sequence[float] | None = None
    cost_bps: float = 100.0


@dataclass(frozen=True)
class StageIVNativeCell:
    cell_id: str
    plans: tuple[StageIVNativePlan, ...]
    source_lineage: Mapping[str, str]


@dataclass(frozen=True)
class StageIVNativeRunnerSpec:
    control_cell_id: str = ""
    admission_spec: Causal21dAdmissionSpec = Causal21dAdmissionSpec()
    top_fractions: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20)
    selection_top_fraction: float = 0.10
    winner_requires_admission: bool = True
    min_selected_rows: int = 20
    min_paired_months: int = 2
    require_tail_fraction_coverage: bool = True

    def validate(self) -> None:
        if self.admission_spec.window_days != 21:
            raise StageIVNativeRunnerError("Stage IV requires the causal 21-day map")
        if not str(self.control_cell_id).strip():
            raise StageIVNativeRunnerError("Stage IV requires a declared control_cell_id")
        if not self.top_fractions or any(not 0.0 < float(x) <= 1.0 for x in self.top_fractions):
            raise StageIVNativeRunnerError("top fractions must lie in (0,1]")
        if float(self.selection_top_fraction) not in set(map(float, self.top_fractions)):
            raise StageIVNativeRunnerError("winner fraction must be one declared top fraction")
        if not np.isclose(float(self.selection_top_fraction), 0.10):
            raise StageIVNativeRunnerError("robust winner gates require pooled-global top-10%")
        if int(self.min_selected_rows) < 1 or int(self.min_paired_months) < 1:
            raise StageIVNativeRunnerError("winner support gates must be positive")


@dataclass(frozen=True)
class StageIVNativeRunResult:
    output_directory: Path
    metrics: pd.DataFrame
    winner: Mapping[str, Any]
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class StageIVNativeFrozenArtifact:
    """Already-fitted Stage-IV models and pre-OOS causal state.

    This contract intentionally contains predictors, never fitters: opening a
    later OOS period cannot refit, reselect a cell, alter FQ3 terciles, or
    rerun HPO.  ``pre_oos_*`` values are frozen development artifacts whose
    timestamps must precede the first OOS decision.
    """

    artifact_id: str
    artifact_sha256: str
    freeze_cutoff_timestamp: Any
    side: str
    broad_model: NativeBasePredictor
    tail_model: NativeBasePredictor
    meta_model: DirectMetaPredictor
    direct_fq3_state: Any
    broad_feature_names: tuple[str, ...]
    tail_feature_names: tuple[str, ...]
    meta_feature_names: tuple[str, ...]
    broad_output_route: str
    tail_fraction: float
    min_handoff_history_rows: int
    score_domain: tuple[float, float]
    pre_oos_handoff_history: pd.DataFrame
    pre_oos_mapping_reference: pd.DataFrame
    # The three already-fitted model files are verified before they are
    # deserialised by the materializer and again immediately before OOS score.
    # No fitter/HPO/selector is represented in this frozen contract.
    model_artifacts: Mapping[str, Mapping[str, str]]
    model_artifact_manifest_sha256: str


@dataclass(frozen=True)
class StageIVNativeFrozenOOSPlan:
    artifact: StageIVNativeFrozenArtifact
    candidate_ids: Sequence[Any]
    symbols: Sequence[Any]
    frame: pd.DataFrame
    exact_net_bps: Sequence[float]
    decision_timestamps: Sequence[Any]
    label_available_timestamps: Sequence[Any]


@dataclass(frozen=True)
class StageIVNativeFrozenOOSResult:
    output_directory: Path
    metrics: pd.DataFrame
    manifest: Mapping[str, Any]


def _json_sha(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _cell_checkpoint_sha256(cell: StageIVNativeCell) -> str:
    plans: list[dict[str, Any]] = []
    for plan in cell.plans:
        feature_names = tuple(dict.fromkeys((
            *map(str, plan.broad_feature_names), *map(str, plan.tail_feature_names),
            *map(str, plan.meta_feature_names),
        )))
        matrix = plan.frame.loc[:, list(feature_names)]
        plans.append({
            "side": plan.side, "candidate_ids": list(map(str, plan.candidate_ids)),
            "symbols": list(map(str, plan.symbols)),
            "decision_timestamps": list(map(str, plan.decision_timestamps)),
            "label_available_timestamps": list(map(str, plan.label_available_timestamps)),
            "base_target_sha256": sha256(np.asarray(plan.base_target).tobytes()).hexdigest(),
            "exact_net_sha256": sha256(np.asarray(plan.exact_net_bps).tobytes()).hexdigest(),
            "feature_values_sha256": sha256(
                pd.util.hash_pandas_object(matrix, index=False).values.tobytes()
            ).hexdigest(),
            "features": {
                "broad": list(plan.broad_feature_names), "tail": list(plan.tail_feature_names),
                "meta": list(plan.meta_feature_names),
            },
            "params": {
                "broad": dict(plan.broad_params), "tail": dict(plan.tail_params),
                "meta": dict(plan.meta_params),
            },
            "tail_fraction": plan.tail_fraction,
            "burns": [plan.broad_min_train_rows, plan.tail_min_train_rows,
                      plan.meta_min_train_rows, plan.min_handoff_history_rows],
            "folds": plan.n_validation_folds, "route": plan.broad_output_route,
            "score_domain": list(plan.score_domain), "cost_bps": plan.cost_bps,
        })
    return _json_sha({
        "schema": SCHEMA, "cell_id": cell.cell_id,
        "source_lineage": dict(sorted(cell.source_lineage.items())), "plans": plans,
    })


def _directory_checksums(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "checksums.json"
    }


def _validate_checkpoint(
    directory: Path, *, cell_id: str, contract_sha256: str,
) -> dict[str, Any]:
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        expected = json.loads((directory / "checksums.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StageIVNativeRunnerError(f"checkpoint {cell_id} is incomplete") from exc
    if (
        manifest.get("status") != "complete" or manifest.get("cell_id") != cell_id
        or manifest.get("cell_contract_sha256") != contract_sha256
    ):
        raise StageIVNativeRunnerError(f"checkpoint {cell_id} contract drift")
    if expected != _directory_checksums(directory):
        raise StageIVNativeRunnerError(f"checkpoint {cell_id} checksum drift")
    return manifest


def _utc(values: Sequence[Any], name: str) -> pd.Series:
    result = pd.to_datetime(pd.Series(values), utc=True, errors="coerce")
    if result.isna().any():
        raise StageIVNativeRunnerError(f"{name} contains invalid timestamps")
    return result


def _array(values: Sequence[Any], n: int, name: str, dtype: Any) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).reshape(-1)
    if len(result) != n:
        raise StageIVNativeRunnerError(f"{name} is not row aligned")
    return result


def _features(frame: pd.DataFrame, values: Sequence[str], layer: str) -> tuple[str, ...]:
    names = tuple(map(str, values))
    if not names or len(set(names)) != len(names):
        raise StageIVNativeRunnerError(f"{layer} features must be unique and non-empty")
    missing = sorted(set(names).difference(frame.columns))
    if missing:
        raise StageIVNativeRunnerError(f"{layer} features are absent: {missing[:8]}")
    forbidden = [name for name in names if _RESERVED_NATIVE_INPUT.match(name)]
    if forbidden:
        raise StageIVNativeRunnerError(
            f"{layer} native model forbids mapped, outcome, or future-path features: {forbidden[:8]}"
        )
    return names


def _validated_plan(plan: StageIVNativePlan) -> tuple[pd.DataFrame, dict[str, Any]]:
    side = str(plan.side).lower()
    route = str(plan.broad_output_route).lower()
    if side not in {"long", "short"} or route not in _ROUTES:
        raise StageIVNativeRunnerError("plan requires a valid isolated side and broad-output route")
    if float(plan.tail_fraction) not in TAIL_FRACTIONS:
        raise StageIVNativeRunnerError("tail fraction must be 20/30/40/50%")
    for name in (
        "broad_min_train_rows", "tail_min_train_rows", "meta_min_train_rows",
        "min_handoff_history_rows", "n_validation_folds",
    ):
        if int(getattr(plan, name)) < 1:
            raise StageIVNativeRunnerError(f"{name} must be positive")
    if int(plan.burn_in_months) != plan.burn_in_months or int(plan.burn_in_months) < 0:
        raise StageIVNativeRunnerError("burn_in_months must be a non-negative integer")
    lower, upper = map(float, plan.score_domain)
    if not np.isfinite([lower, upper]).all() or not lower < upper:
        raise StageIVNativeRunnerError("native score domain must be finite and ordered")
    if not np.isclose(float(plan.cost_bps), 100.0):
        raise StageIVNativeRunnerError("native Stage IV labels must apply the fixed 100-bps cost exactly once")
    frame = plan.frame.copy().reset_index(drop=True)
    frame.columns = frame.columns.astype(str)
    n = len(frame)
    ids = _array(plan.candidate_ids, n, "candidate_ids", object)
    symbols = _array(plan.symbols, n, "symbols", object)
    target = _array(plan.base_target, n, "base_target", float)
    net = _array(plan.exact_net_bps, n, "exact_net_bps", float)
    decision = _utc(plan.decision_timestamps, "decision_timestamps")
    available = _utc(plan.label_available_timestamps, "label_available_timestamps")
    if pd.isna(ids).any() or pd.isna(symbols).any() or len(pd.unique(ids)) != n:
        raise StageIVNativeRunnerError("candidate identities must be unique and complete per side")
    if not np.isfinite(target).all() or not np.isfinite(net).all() or (available <= decision).any():
        raise StageIVNativeRunnerError("targets/economics/timestamps violate the native plan")
    weight = (
        np.ones(n, dtype=np.float32)
        if plan.sample_weight is None
        else _array(plan.sample_weight, n, "sample_weight", float).astype(np.float32)
    )
    if not np.isfinite(weight).all() or (weight < 0.0).any() or weight.sum() <= 0.0:
        raise StageIVNativeRunnerError("sample weights must be finite and non-negative")
    return frame, {
        "side": side, "route": route, "ids": ids, "symbols": symbols,
        "target": target, "net": net, "decision": decision, "available": available,
        "weight": weight, "broad_features": _features(frame, plan.broad_feature_names, "broad"),
        "tail_features": _features(frame, plan.tail_feature_names, "tail"),
        "meta_features": _features(frame, plan.meta_feature_names, "meta"),
    }


def _direct_base_trust(states: np.ndarray) -> dict[str, np.ndarray]:
    clipped = np.clip(states, 1e-12, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    ordered = np.sort(states, axis=1)
    return {
        "base_output_entropy": entropy.astype(np.float32),
        "base_output_top2_margin": (ordered[:, -1] - ordered[:, -2]).astype(np.float32),
        "base_output_max_probability": ordered[:, -1].astype(np.float32),
    }


def _native_prediction(
    model: NativeBasePredictor, design: pd.DataFrame, *, layer: str,
    score_domain: tuple[float, float], state_width: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if not hasattr(model, "predict_native"):
        raise StageIVNativeRunnerError(f"{layer} fitter must emit native score and states")
    prediction = model.predict_native(design)
    score = np.asarray(prediction.score, dtype=np.float32).reshape(-1)
    states = np.asarray(prediction.states, dtype=np.float32)
    if len(score) != len(design) or states.ndim != 2 or len(states) != len(design):
        raise StageIVNativeRunnerError(f"{layer} native prediction is misaligned")
    if states.shape[1] < 2:
        raise StageIVNativeRunnerError(f"{layer} must emit at least two native states")
    if state_width is not None and states.shape[1] != state_width:
        raise StageIVNativeRunnerError("broad and tail native state widths must match")
    lower, upper = map(float, score_domain)
    if (
        not np.isfinite(score).all() or not np.isfinite(states).all()
        or (score < lower - 1e-6).any() or (score > upper + 1e-6).any()
        or (states < 0.0).any() or not np.allclose(states.sum(axis=1), 1.0, atol=1e-5)
    ):
        raise StageIVNativeRunnerError(f"{layer} emitted invalid native scores/states")
    return score, states


def _fit_native_layer(
    *, layer: str, design: pd.DataFrame, target: np.ndarray, weight: np.ndarray,
    candidate_mask: np.ndarray, trainable_mask: np.ndarray, decision: pd.Series,
    available: pd.Series, min_train_rows: int, folds: int, params: Mapping[str, Any],
    fitter: NativeBaseFitter, score_domain: tuple[float, float],
    state_width: int | None, provenance: list[dict[str, Any]], side: str,
    burn_in_months: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score = np.full(len(design), np.nan, dtype=np.float32)
    states: np.ndarray | None = None
    fold_ids = np.full(len(design), -1, dtype=np.int16)
    blocks = _strict_blocks(
        candidate_mask=candidate_mask, trainable_mask=trainable_mask,
        decision=decision, available=available, min_train_rows=min_train_rows,
        n_folds=folds, burn_in_months=int(burn_in_months),
    )
    for fold_id, valid_idx in enumerate(blocks):
        start = decision.iloc[valid_idx].min()
        train_idx = np.flatnonzero(
            trainable_mask & decision.lt(start).to_numpy() & available.lt(start).to_numpy()
        )
        model = fitter(design.iloc[train_idx], target[train_idx], weight[train_idx], layer, params)
        fold_score, fold_states = _native_prediction(
            model, design.iloc[valid_idx], layer=layer, score_domain=score_domain,
            state_width=state_width,
        )
        if states is None:
            states = np.full((len(design), fold_states.shape[1]), np.nan, dtype=np.float32)
        score[valid_idx], states[valid_idx], fold_ids[valid_idx] = fold_score, fold_states, fold_id
        provenance.append({
            "side_name": side, "layer": layer, "fold_id": fold_id,
            "train_rows": len(train_idx), "validation_rows": len(valid_idx),
            "validation_start_ts": start,
            "train_max_label_available_ts": available.iloc[train_idx].max(),
            "strict_prior_resolved": bool(available.iloc[train_idx].lt(start).all()),
            "score_semantics": "same_side_native_score_and_states_no_bps_conversion",
        })
    if states is None:
        width = state_width or 0
        states = np.full((len(design), width), np.nan, dtype=np.float32)
    return score, states, fold_ids


def generate_stage_iv_native_side_oof(
    plan: StageIVNativePlan, *, base_fitter: NativeBaseFitter,
    meta_fitter: DirectMetaFitter,
) -> tuple[pd.DataFrame, pd.DataFrame, Mapping[str, Any]]:
    """Generate one side's strict broad, tail, and direct-FQ3 joint OOF chain."""
    frame, value = _validated_plan(plan)
    n, side, route = len(frame), value["side"], value["route"]
    decision, available = value["decision"], value["available"]
    provenance: list[dict[str, Any]] = []
    broad, broad_states, broad_fold = _fit_native_layer(
        layer="broad", design=_design(frame, value["broad_features"], {}),
        target=value["target"], weight=value["weight"],
        candidate_mask=np.ones(n, bool), trainable_mask=np.ones(n, bool),
        decision=decision, available=available,
        min_train_rows=plan.broad_min_train_rows, folds=plan.n_validation_folds,
        params=plan.broad_params, fitter=base_fitter, score_domain=plan.score_domain,
        state_width=None, provenance=provenance, side=side,
        burn_in_months=int(plan.burn_in_months),
    )
    if not np.isfinite(broad).any() or broad_states.shape[1] < 2:
        raise StageIVNativeRunnerError(f"{side} broad burn-in emitted no native OOF state")
    threshold, eligible = prequential_tail_handoff(
        broad, decision, tail_fraction=plan.tail_fraction,
        min_history_rows=plan.min_handoff_history_rows,
    )
    eligible &= np.isfinite(broad)
    tail_extra = {}
    if route in {"tail", "both"}:
        tail_extra["__stage_iv_broad_native_score"] = broad
    tail_design = _design(frame, value["tail_features"], tail_extra)
    tail, tail_states, tail_fold = _fit_native_layer(
        layer="tail", design=tail_design, target=value["target"], weight=value["weight"],
        candidate_mask=eligible, trainable_mask=eligible, decision=decision,
        available=available, min_train_rows=plan.tail_min_train_rows,
        folds=plan.n_validation_folds, params=plan.tail_params, fitter=base_fitter,
        score_domain=plan.score_domain, state_width=(broad_states.shape[1] or None),
        provenance=provenance, side=side, burn_in_months=int(plan.burn_in_months),
    )
    tail_scored = np.isfinite(tail)
    if not tail_scored.any():
        raise StageIVNativeRunnerError(f"{side} tail burn-in emitted no native OOF state")
    meta_extra: dict[str, np.ndarray] = {"base_raw_score": tail}
    for index in range(tail_states.shape[1]):
        meta_extra[f"base_state_p{index}"] = tail_states[:, index]
    meta_extra.update(_direct_base_trust(tail_states))
    if route in {"meta", "both"}:
        meta_extra["__stage_iv_broad_native_score"] = broad
    meta_design = _design(frame, value["meta_features"], meta_extra)
    probability = np.full((n, 3), np.nan, dtype=np.float32)
    correction = np.full(n, np.nan, dtype=np.float32)
    joint = np.full(n, np.nan, dtype=np.float32)
    meta_fold = np.full(n, -1, dtype=np.int16)
    blocks = _strict_blocks(
        candidate_mask=tail_scored, trainable_mask=tail_scored,
        decision=decision, available=available, min_train_rows=plan.meta_min_train_rows,
        n_folds=plan.n_validation_folds, burn_in_months=int(plan.burn_in_months),
    )
    for fold_id, valid_idx in enumerate(blocks):
        start = decision.iloc[valid_idx].min()
        train_idx = np.flatnonzero(
            tail_scored & decision.lt(start).to_numpy() & available.lt(start).to_numpy()
        )
        labels, state = _fit_direct_correctness(
            value["net"][train_idx], tail[train_idx], score_domain=plan.score_domain,
        )
        model = meta_fitter(
            meta_design.iloc[train_idx], labels, value["weight"][train_idx],
            "meta", plan.meta_params,
        )
        p = _multiclass_probabilities(model, meta_design.iloc[valid_idx])
        delta, combined = _reconstruct_direct_correctness(p, tail[valid_idx], state)
        probability[valid_idx], correction[valid_idx], joint[valid_idx] = p, delta, combined
        meta_fold[valid_idx] = fold_id
        provenance.append({
            "side_name": side, "layer": "meta", "fold_id": fold_id,
            "train_rows": len(train_idx), "validation_rows": len(valid_idx),
            "validation_start_ts": start,
            "train_max_label_available_ts": available.iloc[train_idx].max(),
            "strict_prior_resolved": bool(available.iloc[train_idx].lt(start).all()),
            "target_semantics": DIRECT_FQ3_SEMANTICS,
            "base_input_semantics": DIRECT_BASE_INPUT_SEMANTICS,
            "q33": state.thresholds[0], "q67": state.thresholds[1],
        })
    if not np.isfinite(joint).any():
        raise StageIVNativeRunnerError(f"{side} meta burn-in emitted no direct-FQ3 OOF score")
    output = pd.DataFrame({
        "candidate_id": value["ids"], "symbol": value["symbols"],
        "candidate_key": [f"{side}::{x}" for x in value["ids"]],
        "side_name": side, "decision_ts": decision,
        "label_available_ts": available, "exact_net_bps": value["net"],
        "exact_gross_bps": value["net"] + float(plan.cost_bps),
        "cost_bps": float(plan.cost_bps), "broad_native_score": broad,
        "broad_handoff_threshold": threshold, "tail_prequentially_eligible": eligible,
        "tail_native_score": tail, "meta_direct_correction": correction,
        "joint_meta_native_score": joint, "broad_fold_id": broad_fold,
        "tail_fold_id": tail_fold, "meta_fold_id": meta_fold,
        "broad_strict_oof_available": np.isfinite(broad),
        "tail_strict_oof_available": tail_scored,
        "joint_meta_strict_oof_available": np.isfinite(joint),
        "meta_p_error_tercile_0": probability[:, 0],
        "meta_p_error_tercile_1": probability[:, 1],
        "meta_p_error_tercile_2": probability[:, 2],
    })
    for prefix, states in (("broad", broad_states), ("tail", tail_states)):
        for index in range(states.shape[1]):
            output[f"{prefix}_native_state_p{index}"] = states[:, index]
    summary = {
        "side": side, "tail_fraction": plan.tail_fraction,
        "broad_output_route": route,
        "burn_ins": {
            "broad": plan.broad_min_train_rows, "tail": plan.tail_min_train_rows,
            "meta": plan.meta_min_train_rows,
            "calendar_months": int(plan.burn_in_months),
        },
        "architecture": "broad_native_to_prior_global_tail_native_to_direct_FQ3_joint_native",
        "legacy_mapped_bps_residual": False,
    }
    return output, pd.DataFrame(provenance), summary


def _frozen_sha256(value: str, *, name: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest) or len(set(digest)) == 1:
        raise StageIVNativeRunnerError(f"{name} must be a non-placeholder SHA-256 digest")
    return digest


def _model_artifact_payload(artifacts: Mapping[str, Mapping[str, str]]) -> dict[str, dict[str, str]]:
    required = {"broad_model", "tail_model", "meta_model"}
    if set(map(str, artifacts)) != required:
        raise StageIVNativeRunnerError("frozen artifact must declare exactly broad/tail/meta serialized model files")
    output: dict[str, dict[str, str]] = {}
    for role in sorted(required):
        item = artifacts[role]
        if not isinstance(item, Mapping):
            raise StageIVNativeRunnerError(f"frozen {role} artifact declaration is invalid")
        path = Path(str(item.get("path", ""))).expanduser()
        fmt = str(item.get("format", ""))
        digest = _frozen_sha256(str(item.get("sha256", "")), name=f"frozen {role} model SHA-256")
        if fmt != FROZEN_MODEL_SERIALIZATION_FORMAT:
            raise StageIVNativeRunnerError(f"frozen {role} model uses undeclared serialization format")
        if not path.is_file():
            raise StageIVNativeRunnerError(f"frozen {role} model file is absent: {path}")
        observed = sha256(path.read_bytes()).hexdigest()
        if observed != digest:
            raise StageIVNativeRunnerError(f"frozen {role} model file SHA-256 drift")
        output[role] = {"path": str(path.resolve()), "sha256": digest, "format": fmt}
    return output


def _validate_frozen_model_artifacts(artifact: StageIVNativeFrozenArtifact) -> dict[str, dict[str, str]]:
    payload = _model_artifact_payload(artifact.model_artifacts)
    expected = _frozen_sha256(artifact.model_artifact_manifest_sha256, name="frozen model artifact manifest SHA-256")
    if _json_sha(payload) != expected:
        raise StageIVNativeRunnerError("frozen serialized model manifest SHA-256 drift")
    return payload


def _score_frozen_native_oos_side(
    plan: StageIVNativeFrozenOOSPlan, *, admission_spec: Causal21dAdmissionSpec,
    map_output: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, Mapping[str, Any]]:
    """Score one later period using only the supplied frozen native artifact."""
    artifact = plan.artifact
    side = str(artifact.side).lower()
    if side not in {"long", "short"} or str(artifact.broad_output_route).lower() not in _ROUTES:
        raise StageIVNativeRunnerError("frozen OOS artifact has invalid side or broad-output route")
    _frozen_sha256(artifact.artifact_sha256, name="frozen OOS artifact hash")
    model_files = _validate_frozen_model_artifacts(artifact)
    if not str(artifact.artifact_id).strip() or float(artifact.tail_fraction) not in TAIL_FRACTIONS:
        raise StageIVNativeRunnerError("frozen OOS artifact lacks immutable identity or tail contract")
    if admission_spec.window_days != 21:
        raise StageIVNativeRunnerError("frozen OOS requires the canonical 21-day admission map")
    frame = plan.frame.copy().reset_index(drop=True)
    frame.columns = frame.columns.astype(str)
    n = len(frame)
    ids = _array(plan.candidate_ids, n, "candidate_ids", object)
    symbols = _array(plan.symbols, n, "symbols", object)
    net = _array(plan.exact_net_bps, n, "exact_net_bps", float)
    decision = _utc(plan.decision_timestamps, "decision_timestamps")
    available = _utc(plan.label_available_timestamps, "label_available_timestamps")
    if not n or pd.isna(ids).any() or pd.isna(symbols).any() or len(pd.unique(ids)) != n:
        raise StageIVNativeRunnerError("frozen OOS candidate identities must be complete and unique per side")
    if not np.isfinite(net).all() or (available <= decision).any():
        raise StageIVNativeRunnerError("frozen OOS outcomes/timestamps are invalid")
    route = str(artifact.broad_output_route).lower()
    broad_features = _features(frame, artifact.broad_feature_names, "frozen broad")
    tail_features = _features(frame, artifact.tail_feature_names, "frozen tail")
    meta_features = _features(frame, artifact.meta_feature_names, "frozen meta")
    start = decision.min()
    history = artifact.pre_oos_handoff_history.copy()
    required_history = {"decision_ts", "broad_native_score"}
    if required_history.difference(history.columns):
        raise StageIVNativeRunnerError("frozen OOS handoff history lacks decision_ts/broad_native_score")
    history_decision = _utc(history.decision_ts, "pre_oos_handoff_history.decision_ts")
    history_score = pd.to_numeric(history.broad_native_score, errors="coerce").to_numpy(float)
    if not len(history) or not np.isfinite(history_score).all() or not (history_decision < start).all():
        raise StageIVNativeRunnerError("frozen OOS handoff history must be finite and strictly pre-OOS")
    broad, broad_states = _native_prediction(
        artifact.broad_model, _design(frame, broad_features, {}), layer="frozen_broad",
        score_domain=artifact.score_domain, state_width=None,
    )
    combined_scores = np.concatenate([history_score.astype(np.float32), broad])
    combined_decision = pd.concat([history_decision, decision], ignore_index=True)
    threshold_all, eligible_all = prequential_tail_handoff(
        combined_scores, combined_decision, tail_fraction=artifact.tail_fraction,
        min_history_rows=artifact.min_handoff_history_rows,
    )
    threshold, eligible = threshold_all[-n:], eligible_all[-n:]
    tail = np.full(n, np.nan, dtype=np.float32)
    tail_states = np.full((n, broad_states.shape[1]), np.nan, dtype=np.float32)
    tail_extra: dict[str, np.ndarray] = {}
    if route in {"tail", "both"}:
        tail_extra["__stage_iv_broad_native_score"] = broad
    tail_design = _design(frame, tail_features, tail_extra)
    tail_idx = np.flatnonzero(eligible)
    if len(tail_idx):
        score, state = _native_prediction(
            artifact.tail_model, tail_design.iloc[tail_idx], layer="frozen_tail",
            score_domain=artifact.score_domain, state_width=broad_states.shape[1],
        )
        tail[tail_idx], tail_states[tail_idx] = score, state
    probability = np.full((n, 3), np.nan, dtype=np.float32)
    correction, joint = np.full(n, np.nan, dtype=np.float32), np.full(n, np.nan, dtype=np.float32)
    tail_scored = np.isfinite(tail)
    if tail_scored.any():
        meta_extra: dict[str, np.ndarray] = {"base_raw_score": tail}
        for index in range(tail_states.shape[1]):
            meta_extra[f"base_state_p{index}"] = tail_states[:, index]
        meta_extra.update(_direct_base_trust(tail_states))
        if route in {"meta", "both"}:
            meta_extra["__stage_iv_broad_native_score"] = broad
        meta_design = _design(frame, meta_features, meta_extra)
        p = _multiclass_probabilities(artifact.meta_model, meta_design.iloc[np.flatnonzero(tail_scored)])
        delta, combined = _reconstruct_direct_correctness(
            p, tail[tail_scored], artifact.direct_fq3_state,
        )
        probability[tail_scored], correction[tail_scored], joint[tail_scored] = p, delta, combined
    output = pd.DataFrame({
        "candidate_id": ids, "symbol": symbols,
        "candidate_key": [f"{side}::{value}" for value in ids], "side_name": side,
        "decision_ts": decision, "label_available_ts": available,
        "exact_net_bps": net, "exact_gross_bps": net + 100.0, "cost_bps": 100.0,
        "broad_native_score": broad, "broad_handoff_threshold": threshold,
        "tail_prequentially_eligible": eligible, "tail_native_score": tail,
        "meta_direct_correction": correction, "joint_meta_native_score": joint,
        "broad_strict_oof_available": False, "tail_strict_oof_available": False,
        "joint_meta_strict_oof_available": False,
        "joint_meta_frozen_oos_available": np.isfinite(joint),
        "meta_p_error_tercile_0": probability[:, 0],
        "meta_p_error_tercile_1": probability[:, 1],
        "meta_p_error_tercile_2": probability[:, 2],
        "frozen_artifact_id": artifact.artifact_id,
        "frozen_artifact_sha256": artifact.artifact_sha256,
        "frozen_model_artifact_manifest_sha256": artifact.model_artifact_manifest_sha256,
        "direct_fq3_semantics": DIRECT_FQ3_SEMANTICS,
        "direct_base_input_semantics": DIRECT_BASE_INPUT_SEMANTICS,
    })
    for prefix, states in (("broad", broad_states), ("tail", tail_states)):
        for index in range(states.shape[1]):
            output[f"{prefix}_native_state_p{index}"] = states[:, index]
    reference = _validated_frozen_mapping_reference(artifact, start=start, side=side)
    if not map_output:
        return output, pd.DataFrame(), {
            "artifact_id": artifact.artifact_id, "artifact_sha256": artifact.artifact_sha256,
            "model_artifact_manifest_sha256": artifact.model_artifact_manifest_sha256,
            "model_files": model_files,
            "side": side, "oos_start": str(start), "oos_end": str(decision.max()),
            "refit_forbidden": True, "selection_forbidden": True, "hpo_forbidden": True,
            "frozen_feature_contract": {
                "broad": broad_features, "tail": tail_features, "meta": meta_features,
            },
        }
    map_input = pd.concat([reference, output.assign(net_bps=output.exact_net_bps)], ignore_index=True, sort=False)
    mapped, audit = apply_causal_21d_side_admission(
        map_input, score_column="joint_meta_native_score", net_column="net_bps",
        decision_column="decision_ts", label_available_column="label_available_ts",
        identity_column="candidate_key", spec=admission_spec,
    )
    mapped = mapped.loc[mapped.candidate_key.isin(output.candidate_key)].set_index("candidate_key").reindex(output.candidate_key).reset_index()
    output["joint_meta_causal_21d_common_bps"] = mapped.causal_21d_side_expected_net_bps.to_numpy()
    output["joint_meta_causal_21d_admitted"] = mapped.causal_21d_side_admitted_ge_50bps.to_numpy(bool)
    output["joint_map_is_prequential"] = True
    output["joint_map_source_side"] = side
    output["joint_map_pre_oos_reference_max_label_available_ts"] = reference.label_available_ts.max()
    return output, audit.assign(layer="joint_meta", source="frozen_oos"), {
        "artifact_id": artifact.artifact_id, "artifact_sha256": artifact.artifact_sha256,
        "model_artifact_manifest_sha256": artifact.model_artifact_manifest_sha256,
        "model_files": model_files,
        "side": side, "oos_start": str(start), "oos_end": str(decision.max()),
        "refit_forbidden": True, "selection_forbidden": True, "hpo_forbidden": True,
        "frozen_feature_contract": {
            "broad": broad_features, "tail": tail_features, "meta": meta_features,
        },
    }


def _validated_frozen_mapping_reference(
    artifact: StageIVNativeFrozenArtifact, *, start: pd.Timestamp, side: str,
) -> pd.DataFrame:
    reference = artifact.pre_oos_mapping_reference.copy()
    required_reference = {
        "candidate_key", "side_name", "decision_ts", "label_available_ts",
        "exact_net_bps", "joint_meta_native_score",
    }
    if required_reference.difference(reference.columns):
        raise StageIVNativeRunnerError("frozen OOS mapping reference lacks required causal joint-score fields")
    reference["decision_ts"] = _utc(reference.decision_ts, "pre_oos_mapping_reference.decision_ts")
    reference["label_available_ts"] = _utc(reference.label_available_ts, "pre_oos_mapping_reference.label_available_ts")
    if reference.candidate_key.isna().any() or reference.candidate_key.duplicated().any() or not reference.side_name.astype(str).str.lower().eq(side).all():
        raise StageIVNativeRunnerError("frozen OOS mapping reference identity/side is invalid")
    if not (reference.label_available_ts < start).all():
        raise StageIVNativeRunnerError("frozen OOS mapping reference contains non-prior resolved labels")
    reference["net_bps"] = pd.to_numeric(reference.exact_net_bps, errors="coerce")
    return reference


def _map_layer(
    predictions: pd.DataFrame, *, layer: str, score_column: str,
    spec: Causal21dAdmissionSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = predictions.loc[predictions[score_column].notna()].copy()
    selected["net_bps"] = selected.exact_net_bps
    mapped, audit = apply_causal_21d_side_admission(
        selected, score_column=score_column, net_column="net_bps",
        decision_column="decision_ts", label_available_column="label_available_ts",
        identity_column="candidate_key", spec=spec,
    )
    columns = {
        "causal_21d_side_expected_net_bps": f"{layer}_causal_21d_common_bps",
        "causal_21d_side_admitted_ge_50bps": f"{layer}_causal_21d_admitted",
    }
    output = predictions.merge(
        mapped.loc[:, ["candidate_key", *columns]].rename(columns=columns),
        on="candidate_key", how="left", validate="one_to_one",
    )
    output[f"{layer}_causal_21d_admitted"] = output[
        f"{layer}_causal_21d_admitted"
    ].eq(True)
    audit = audit.assign(layer=layer)
    return output, audit


def _metrics(
    frame: pd.DataFrame, *, cell_id: str, layer: str,
    top_fractions: Sequence[float], diagnostic_only: bool,
    common_population_rows: int,
) -> pd.DataFrame:
    score = f"{layer}_causal_21d_common_bps"
    admitted = f"{layer}_causal_21d_admitted"
    rows: list[dict[str, Any]] = []
    if int(common_population_rows) < 1:
        raise StageIVNativeRunnerError("metrics require a common scored population")
    for scope, population in (
        ("without_admission", frame.loc[frame[score].notna()]),
        ("with_admission", frame.loc[frame[score].notna() & frame[admitted]]),
    ):
        ordered = population.sort_values(
            [score, "candidate_key"], ascending=[False, True], kind="stable"
        )
        for fraction in top_fractions:
            global_k = max(1, int(np.ceil(float(fraction) * int(common_population_rows))))
            selected = ordered.head(global_k)
            common = {
                "cell_id": cell_id, "layer": layer, "admission_scope": scope,
                "top_fraction": float(fraction), "eligible_rows": len(ordered),
                "common_globally_scored_rows": int(common_population_rows),
                "global_top_k_rows": global_k,
                "selected_global_rows": len(selected),
                "ranking": "one_pooled_global_rank_after_side_local_common_bps_mapping",
                "diagnostic_only": bool(diagnostic_only),
                "promotable": not diagnostic_only,
            }
            rows.append({
                **common, "scope": "pooled_global", "side_name": "__all__",
                "month": "__all__", "selected_rows": len(selected),
                "net_bps_per_trade": float(selected.exact_net_bps.mean()) if len(selected) else np.nan,
                "gross_bps_per_trade": float(selected.exact_gross_bps.mean()) if len(selected) else np.nan,
            })
            if len(selected):
                attributed = selected.assign(
                    month=pd.to_datetime(selected.decision_ts, utc=True).dt.strftime("%Y-%m")
                )
                for (side, month), group in attributed.groupby(
                    ["side_name", "month"], observed=True, sort=True
                ):
                    rows.append({
                        **common, "scope": "selected_contribution",
                        "side_name": side, "month": month,
                        "selected_rows": len(group),
                        "net_bps_per_trade": float(group.exact_net_bps.mean()),
                        "gross_bps_per_trade": float(group.exact_gross_bps.mean()),
                    })
    return pd.DataFrame(rows)


def _zero_complete_side_month_report(
    *, predictions_by_cell: Mapping[str, pd.DataFrame], common_by_layer: Mapping[str, set[Any]],
    cells: Sequence[StageIVNativeCell], top_fractions: Sequence[float],
) -> pd.DataFrame:
    """Attribute each pooled book to every available side/month, including zeroes."""
    rows: list[dict[str, Any]] = []
    for layer, keys in common_by_layer.items():
        score, admitted = f"{layer}_causal_21d_common_bps", f"{layer}_causal_21d_admitted"
        reference = next(iter(predictions_by_cell.values())).loc[
            lambda x: x.candidate_key.isin(keys), ["candidate_key", "side_name", "decision_ts"]
        ].copy()
        reference["month"] = pd.to_datetime(reference.decision_ts, utc=True).dt.strftime("%Y-%m")
        grid = reference.loc[:, ["side_name", "month"]].drop_duplicates().sort_values(
            ["side_name", "month"], kind="stable"
        )
        global_rows = len(keys)
        for cell in cells:
            frame = predictions_by_cell[cell.cell_id].loc[
                lambda x: x.candidate_key.isin(keys)
            ].copy()
            frame["month"] = pd.to_datetime(frame.decision_ts, utc=True).dt.strftime("%Y-%m")
            for scope, population in (
                ("without_admission", frame.loc[frame[score].notna()]),
                ("with_admission", frame.loc[frame[score].notna() & frame[admitted]]),
            ):
                ordered = population.sort_values([score, "candidate_key"], ascending=[False, True], kind="stable")
                for fraction in top_fractions:
                    global_k = max(1, int(np.ceil(float(fraction) * global_rows)))
                    selected = ordered.head(global_k)
                    selected_group = selected.groupby(["side_name", "month"], observed=True)
                    eligible_group = population.groupby(["side_name", "month"], observed=True).size()
                    for _, item in grid.iterrows():
                        key = (item.side_name, item.month)
                        group = selected_group.get_group(key) if key in selected_group.groups else selected.iloc[:0]
                        rows.append({
                            "cell_id": cell.cell_id, "layer": layer, "admission_scope": scope,
                            "top_fraction": float(fraction), "side_name": item.side_name,
                            "month": item.month, "common_globally_scored_rows": global_rows,
                            "global_top_k_rows": global_k, "eligible_rows_side_month": int(eligible_group.get(key, 0)),
                            "selected_rows": len(group),
                            "net_bps_per_trade": float(group.exact_net_bps.mean()) if len(group) else np.nan,
                            "gross_bps_per_trade": float(group.exact_gross_bps.mean()) if len(group) else np.nan,
                            "zero_selected": not bool(len(group)),
                        })
    return pd.DataFrame(rows)


def _feature_contract_rows(cells: Sequence[StageIVNativeCell]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cell in cells:
        for plan in cell.plans:
            for layer, features in (
                ("broad_base", plan.broad_feature_names), ("tail_base", plan.tail_feature_names),
                ("joint_meta", plan.meta_feature_names),
            ):
                rows.append({
                    "cell_id": cell.cell_id, "side_name": str(plan.side).lower(), "layer": layer,
                    "feature_names_json": json.dumps(list(map(str, features)), separators=(",", ":")),
                    "feature_count": len(features),
                    "source_lineage_json": json.dumps(dict(cell.source_lineage), sort_keys=True),
                })
    return pd.DataFrame(rows)


def run_stage_iv_native_frozen_oos(
    plans: Sequence[StageIVNativeFrozenOOSPlan], *, output_directory: str | Path,
    admission_spec: Causal21dAdmissionSpec = Causal21dAdmissionSpec(),
) -> StageIVNativeFrozenOOSResult:
    """Evaluate a selected native Stage-IV artifact on later untouched rows.

    This deliberately accepts no fitters, feature selector, HPO configuration,
    or winner selector.  Its only adaptive operation is the declared causal
    21-day map, which uses labels resolved before each later decision.
    """
    choices = tuple(plans)
    if not choices or len({str(plan.artifact.side).lower() for plan in choices}) != len(choices):
        raise StageIVNativeRunnerError("frozen OOS requires exactly one immutable artifact per side")
    output = Path(output_directory)
    if output.exists():
        raise StageIVNativeRunnerError("frozen OOS output already exists and is immutable")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        predictions, contracts, references = [], [], []
        for plan in choices:
            start = _utc(plan.decision_timestamps, "decision_timestamps").min()
            freeze = _utc([plan.artifact.freeze_cutoff_timestamp], "freeze_cutoff_timestamp").iloc[0]
            if not freeze < start:
                raise StageIVNativeRunnerError("frozen OOS artifact cutoff must precede every OOS decision")
            scored, _audit, contract = _score_frozen_native_oos_side(
                plan, admission_spec=admission_spec, map_output=False,
            )
            predictions.append(scored)
            contracts.append(contract)
            references.append(_validated_frozen_mapping_reference(
                plan.artifact, start=start, side=str(plan.artifact.side).lower(),
            ))
        ledger = pd.concat(predictions, ignore_index=True)
        if ledger.candidate_key.duplicated().any():
            raise StageIVNativeRunnerError("frozen OOS candidate keys must be globally side-qualified unique")
        reference = pd.concat(references, ignore_index=True)
        if reference.candidate_key.duplicated().any() or reference.candidate_key.isin(ledger.candidate_key).any():
            raise StageIVNativeRunnerError("frozen OOS mapping reference has colliding side-qualified identities")
        map_input = pd.concat([reference, ledger.assign(net_bps=ledger.exact_net_bps)], ignore_index=True, sort=False)
        mapped, audit = apply_causal_21d_side_admission(
            map_input, score_column="joint_meta_native_score", net_column="net_bps",
            decision_column="decision_ts", label_available_column="label_available_ts",
            identity_column="candidate_key", spec=admission_spec,
        )
        mapped = mapped.loc[mapped.candidate_key.isin(ledger.candidate_key)].set_index("candidate_key").reindex(ledger.candidate_key).reset_index()
        ledger["joint_meta_causal_21d_common_bps"] = mapped.causal_21d_side_expected_net_bps.to_numpy()
        ledger["joint_meta_causal_21d_admitted"] = mapped.causal_21d_side_admitted_ge_50bps.to_numpy(bool)
        ledger["joint_map_is_prequential"] = True
        ledger["joint_map_source_side"] = ledger.side_name.astype(str)
        ledger["joint_map_pre_oos_reference_max_label_available_ts"] = ledger.side_name.map(
            reference.groupby("side_name", observed=True).label_available_ts.max()
        )
        tails = _REQUIRED_REPORT_TAILS
        metrics = _metrics(
            ledger, cell_id="frozen_oos", layer="joint_meta", top_fractions=tails,
            diagnostic_only=True, common_population_rows=len(ledger),
        )
        zero_complete = _zero_complete_side_month_report(
            predictions_by_cell={"frozen_oos": ledger},
            common_by_layer={"joint_meta": set(ledger.candidate_key)},
            cells=(StageIVNativeCell("frozen_oos", tuple(), {}),), top_fractions=tails,
        )
        ledger.to_parquet(stage / "frozen_oos_predictions.parquet", index=False)
        metrics.to_parquet(stage / "frozen_oos_global_top_metrics.parquet", index=False)
        zero_complete.to_parquet(stage / "frozen_oos_side_month_zero_complete_metrics.parquet", index=False)
        audit.assign(layer="joint_meta", source="frozen_oos_joint_side_parent_map").to_parquet(
            stage / "frozen_oos_map_admission_support.parquet", index=False
        )
        contract_frame = pd.DataFrame(contracts)
        contract_frame.to_json(stage / "frozen_oos_feature_contracts_and_source_hashes.json", orient="records", indent=2)
        fold_rows = []
        for plan, contract in zip(choices, contracts):
            reference = plan.artifact.pre_oos_mapping_reference
            fold_rows.append({
                "side_name": contract["side"], "artifact_id": contract["artifact_id"],
                "artifact_sha256": contract["artifact_sha256"],
                "model_artifact_manifest_sha256": contract["model_artifact_manifest_sha256"],
                "model_files_json": json.dumps(contract["model_files"], sort_keys=True),
                "freeze_cutoff_timestamp": str(plan.artifact.freeze_cutoff_timestamp),
                "oos_start_ts": contract["oos_start"], "oos_end_ts": contract["oos_end"],
                "pre_oos_reference_max_label_available_ts": str(pd.to_datetime(reference.label_available_ts, utc=True).max()),
                "refit_forbidden": True, "selection_forbidden": True, "hpo_forbidden": True,
            })
        pd.DataFrame(fold_rows).to_parquet(stage / "frozen_oos_windows_and_label_cutoffs.parquet", index=False)
        manifest = {
            "schema": f"{SCHEMA}_frozen_oos_v1", "status": "complete",
            "architecture": "frozen_native_broad_tail_direct_FQ3_joint_then_causal_21d_common_bps",
            "untouched_oos": True, "refit_forbidden": True, "selection_forbidden": True,
            "hpo_forbidden": True, "global_ranking": "pooled_global_after_side_local_common_bps_mapping",
            "top_k_denominator": "all_frozen_oos_scored_rows_before_admission",
            "reported_global_top_fractions": list(tails), "artifacts": contracts,
            "serialized_model_loading": {
                "format": FROZEN_MODEL_SERIALIZATION_FORMAT,
                "verification": "every declared model file SHA-256 verified immediately before frozen OOS scoring",
                "fit_hpo_reselection_paths": "absent_from_frozen_cli_and_runner",
            },
        }
        (stage / "run_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        checksums = {
            path.relative_to(stage).as_posix(): sha256(path.read_bytes()).hexdigest()
            for path in sorted(stage.rglob("*")) if path.is_file()
        }
        (stage / "checksums.json").write_text(json.dumps(checksums, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        stage.replace(output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return StageIVNativeFrozenOOSResult(output, metrics, manifest)


def _globally_bounded_joint_book(
    frame: pd.DataFrame, *, common_keys: set[Any], global_top_k_rows: int,
    require_admission: bool,
) -> tuple[pd.DataFrame, int]:
    """Select at most the pre-admission global k from one cell's joint book."""

    matched = frame.loc[frame.candidate_key.isin(common_keys)].copy()
    if require_admission:
        matched = matched.loc[matched.joint_meta_causal_21d_admitted].copy()
    ordered = matched.sort_values(
        ["joint_meta_causal_21d_common_bps", "candidate_key"],
        ascending=[False, True], kind="stable",
    )
    return ordered.head(int(global_top_k_rows)), len(ordered)


def run_stage_iv_native_artifact_sweep(
    cells: Sequence[StageIVNativeCell], *, output_directory: str | Path,
    base_fitter: NativeBaseFitter, meta_fitter: DirectMetaFitter,
    spec: StageIVNativeRunnerSpec = StageIVNativeRunnerSpec(),
    checkpoint_directory: str | Path | None = None,
    resume: bool = False,
    launch_manifest: Mapping[str, Any] | None = None,
) -> StageIVNativeRunResult:
    """Run explicit cells sequentially and atomically publish one Stage-IV bundle."""
    spec.validate()
    choices = tuple(cells)
    if not choices or len({cell.cell_id for cell in choices}) != len(choices):
        raise StageIVNativeRunnerError("declare uniquely named sequential cells")
    if spec.control_cell_id not in {cell.cell_id for cell in choices}:
        raise StageIVNativeRunnerError(
            f"declared control cell {spec.control_cell_id!r} is absent"
        )
    canonical_lineage = dict(choices[0].source_lineage)
    if any(dict(cell.source_lineage) != canonical_lineage for cell in choices[1:]):
        raise StageIVNativeRunnerError("all compared cells must share identical frozen source lineage")
    fractions = {float(plan.tail_fraction) for cell in choices for plan in cell.plans}
    if spec.require_tail_fraction_coverage and fractions != set(TAIL_FRACTIONS):
        raise StageIVNativeRunnerError("primary Stage-IV sweep must cover x=20/30/40/50%")
    output = Path(output_directory)
    if output.exists():
        raise StageIVNativeRunnerError("Stage-IV output already exists and is immutable")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    (stage / "checkpoints").mkdir()
    checkpoint_root = (
        stage / "checkpoints"
        if checkpoint_directory is None else Path(checkpoint_directory)
    )
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    if checkpoint_directory is not None and any(checkpoint_root.iterdir()) and not resume:
        raise StageIVNativeRunnerError(
            "checkpoint directory is non-empty; pass resume=True to verify and reuse it"
        )
    expected_checkpoint_names = {
        f"{ordinal:04d}__{cell.cell_id}" for ordinal, cell in enumerate(choices)
    }
    unexpected = {path.name for path in checkpoint_root.iterdir()}.difference(
        expected_checkpoint_names
    )
    if unexpected:
        raise StageIVNativeRunnerError(
            f"checkpoint directory contains undeclared cells/files: {sorted(unexpected)[:8]}"
        )
    predictions_by_cell: dict[str, pd.DataFrame] = {}
    execution: list[dict[str, Any]] = []
    resumed_cells = 0
    for ordinal, cell in enumerate(choices):
        if not cell.plans or len({str(plan.side).lower() for plan in cell.plans}) != len(cell.plans):
            raise StageIVNativeRunnerError("each cell requires unique side plans")
        contracts = {
            (
                float(plan.tail_fraction), int(plan.broad_min_train_rows),
                int(plan.tail_min_train_rows), int(plan.meta_min_train_rows),
                int(plan.min_handoff_history_rows), str(plan.broad_output_route).lower(),
            )
            for plan in cell.plans
        }
        if len(contracts) != 1:
            raise StageIVNativeRunnerError(
                "each cell's side plans must share tail, burn-in, and route settings"
            )
        lineage = [str(value).lower() for value in cell.source_lineage.values()]
        if (
            not lineage
            or any(len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
                   or len(set(value)) == 1 for value in lineage)
        ):
            raise StageIVNativeRunnerError("each cell requires SHA-256 source lineage")
        cell_contract_sha = _cell_checkpoint_sha256(cell)
        checkpoint = checkpoint_root / f"{ordinal:04d}__{cell.cell_id}"
        if checkpoint.exists():
            if not resume:
                raise StageIVNativeRunnerError(
                    f"checkpoint already exists for {cell.cell_id}; explicit resume required"
                )
            cell_manifest = _validate_checkpoint(
                checkpoint, cell_id=cell.cell_id,
                contract_sha256=cell_contract_sha,
            )
            predictions = pd.read_parquet(checkpoint / "oof_predictions.parquet")
            if predictions.candidate_key.duplicated().any():
                raise StageIVNativeRunnerError(
                    f"checkpoint {cell.cell_id} contains duplicate candidate keys"
                )
            predictions_by_cell[cell.cell_id] = predictions
            execution.append({**cell_manifest, "resumed": True})
            resumed_cells += 1
            if checkpoint.resolve() != (stage / "checkpoints" / checkpoint.name).resolve():
                shutil.copytree(checkpoint, stage / "checkpoints" / checkpoint.name)
            continue
        cell_stage = Path(tempfile.mkdtemp(
            prefix=f".{cell.cell_id}.", dir=checkpoint_root.parent
        ))
        side_predictions, folds, summaries = [], [], []
        for plan in cell.plans:
            prediction, fold, summary = generate_stage_iv_native_side_oof(
                plan, base_fitter=base_fitter, meta_fitter=meta_fitter,
            )
            side_predictions.append(prediction)
            folds.append(fold)
            summaries.append(summary)
        predictions = pd.concat(side_predictions, ignore_index=True)
        if predictions.candidate_key.duplicated().any():
            raise StageIVNativeRunnerError("cell candidate keys are not globally unique")
        audits: list[pd.DataFrame] = []
        for layer, score in (
            ("broad_base", "broad_native_score"),
            ("tail_base", "tail_native_score"),
            ("joint_meta", "joint_meta_native_score"),
        ):
            predictions, audit = _map_layer(
                predictions, layer=layer, score_column=score, spec=spec.admission_spec,
            )
            audits.append(audit)
        predictions.to_parquet(cell_stage / "oof_predictions.parquet", index=False)
        pd.concat(folds, ignore_index=True).to_parquet(
            cell_stage / "fold_provenance.parquet", index=False
        )
        pd.concat(audits, ignore_index=True).to_parquet(
            cell_stage / "causal_21d_map_audit.parquet", index=False
        )
        cell_manifest = {
            "schema": SCHEMA, "status": "complete", "ordinal": ordinal,
            "cell_id": cell.cell_id, "cell_contract_sha256": cell_contract_sha,
            "source_lineage": dict(cell.source_lineage), "sides": summaries,
            "joint_meta_only_promotable": True, "resumed": False,
        }
        (cell_stage / "manifest.json").write_text(
            json.dumps(cell_manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        (cell_stage / "checksums.json").write_text(
            json.dumps(_directory_checksums(cell_stage), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        cell_stage.replace(checkpoint)
        if checkpoint.resolve() != (stage / "checkpoints" / checkpoint.name).resolve():
            shutil.copytree(checkpoint, stage / "checkpoints" / checkpoint.name)
        predictions_by_cell[cell.cell_id] = predictions
        execution.append(cell_manifest)
    common_by_layer: dict[str, set[Any]] = {}
    all_metrics: list[pd.DataFrame] = []
    for layer in ("broad_base", "tail_base", "joint_meta"):
        score = f"{layer}_causal_21d_common_bps"
        key_sets = [
            set(frame.loc[frame[score].notna(), "candidate_key"])
            for frame in predictions_by_cell.values()
        ]
        layer_common = set.intersection(*key_sets) if key_sets else set()
        if not layer_common:
            raise StageIVNativeRunnerError(
                f"cells have no common mapped {layer} population"
            )
        common_by_layer[layer] = layer_common
        for cell in choices:
            matched_layer = predictions_by_cell[cell.cell_id].loc[
                predictions_by_cell[cell.cell_id].candidate_key.isin(layer_common)
            ].copy()
            all_metrics.append(_metrics(
                matched_layer, cell_id=cell.cell_id, layer=layer,
                top_fractions=tuple(sorted(set(map(float, spec.top_fractions)).union(_REQUIRED_REPORT_TAILS))),
                diagnostic_only=layer != "joint_meta",
                common_population_rows=len(layer_common),
            ))
    metrics = pd.concat(all_metrics, ignore_index=True)
    common = common_by_layer["joint_meta"]
    if not common:
        raise StageIVNativeRunnerError("cells have no matched mapped joint-meta population")
    invariant_columns = (
        "candidate_key", "side_name", "symbol", "decision_ts",
        "label_available_ts", "exact_net_bps", "exact_gross_bps", "cost_bps",
    )
    reference: pd.DataFrame | None = None
    for cell_id, frame in predictions_by_cell.items():
        current = frame.loc[
            frame.candidate_key.isin(common), list(invariant_columns)
        ].sort_values("candidate_key", kind="stable").reset_index(drop=True)
        if reference is None:
            reference = current
            continue
        if not current.candidate_key.astype(str).equals(reference.candidate_key.astype(str)):
            raise StageIVNativeRunnerError("matched Stage-IV candidate identity drifted")
        for column in invariant_columns[1:]:
            left, right = reference[column], current[column]
            equal = (
                np.allclose(left.to_numpy(float), right.to_numpy(float), atol=1e-6, rtol=0.0)
                if pd.api.types.is_numeric_dtype(left)
                else left.astype(str).equals(right.astype(str))
            )
            if not equal:
                raise StageIVNativeRunnerError(
                    f"matched Stage-IV economics/identity differ in {column} for {cell_id}"
                )
    population_sha = sha256(
        json.dumps(sorted(map(str, common)), separators=(",", ":")).encode()
    ).hexdigest()
    global_k = max(1, int(np.ceil(spec.selection_top_fraction * len(common))))
    selected_books: dict[str, pd.DataFrame] = {}
    eligible_rows: dict[str, int] = {}
    for cell in choices:
        frame = predictions_by_cell[cell.cell_id]
        selected, eligible = _globally_bounded_joint_book(
            frame, common_keys=common, global_top_k_rows=global_k,
            require_admission=spec.winner_requires_admission,
        )
        if selected.empty:
            raise StageIVNativeRunnerError(
                f"cell {cell.cell_id} has no admitted matched joint-meta rows"
            )
        selected_books[cell.cell_id] = selected
        eligible_rows[cell.cell_id] = eligible

    control_book = selected_books[spec.control_cell_id]
    control_month = control_book.assign(
        month=pd.to_datetime(control_book.decision_ts, utc=True).dt.strftime("%Y-%m")
    ).groupby("month", observed=True).exact_net_bps.mean()
    gate_rows: list[dict[str, Any]] = []
    for cell in choices:
        selected = selected_books[cell.cell_id]
        monthly = selected.assign(
            month=pd.to_datetime(selected.decision_ts, utc=True).dt.strftime("%Y-%m")
        ).groupby("month", observed=True).exact_net_bps.mean()
        common_months = control_month.index.intersection(monthly.index)
        month_coverage_complete = set(monthly.index) == set(control_month.index)
        month_delta = monthly.loc[common_months] - control_month.loc[common_months]
        aggregate_delta = float(
            selected.exact_net_bps.mean() - control_book.exact_net_bps.mean()
        )
        worst_delta = float(month_delta.min()) if len(month_delta) else np.nan
        latest_delta = (
            float(month_delta.loc[common_months.max()]) if len(common_months) else np.nan
        )
        adequate_support = bool(
            len(control_book) >= int(spec.min_selected_rows)
            and len(selected) >= int(spec.min_selected_rows)
            and len(common_months) >= int(spec.min_paired_months)
            and month_coverage_complete
        )
        is_control = cell.cell_id == spec.control_cell_id
        passes = bool(
            not is_control and adequate_support and aggregate_delta > 0.0
            and np.isfinite(worst_delta) and worst_delta >= 0.0
            and np.isfinite(latest_delta) and latest_delta >= 0.0
        )
        gate_rows.append({
            "cell_id": cell.cell_id, "is_control": is_control,
            "common_globally_scored_rows": len(common),
            "global_top_k_rows": global_k,
            "admission_eligible_rows": eligible_rows[cell.cell_id],
            "selected_rows": len(selected),
            "net_bps_per_trade": float(selected.exact_net_bps.mean()),
            "gross_bps_per_trade": float(selected.exact_gross_bps.mean()),
            "aggregate_top10_lift_bps": 0.0 if is_control else aggregate_delta,
            "worst_month_top10_lift_bps": 0.0 if is_control else worst_delta,
            "latest_month_top10_lift_bps": 0.0 if is_control else latest_delta,
            "paired_months": len(common_months),
            "month_coverage_matches_control": month_coverage_complete,
            "adequate_selected_support": adequate_support,
            "passes_robust_joint_meta_gate": passes,
        })
    gate_table = pd.DataFrame(gate_rows)
    passing = gate_table.loc[gate_table.passes_robust_joint_meta_gate].copy()
    if passing.empty:
        selected_row = gate_table.loc[
            gate_table.cell_id.eq(spec.control_cell_id)
        ].iloc[0]
        decision = "NO_STAGE_IV_ADVANCE"
    else:
        selected_row = passing.sort_values(
            [
                "aggregate_top10_lift_bps", "worst_month_top10_lift_bps",
                "latest_month_top10_lift_bps", "cell_id",
            ],
            ascending=[False, False, False, True], kind="stable",
        ).iloc[0]
        decision = "STAGE_IV_CHALLENGER_ADVANCES"
    winner = {
        **selected_row.to_dict(),
        "control_cell_id": spec.control_cell_id,
        "decision": decision,
        "selection_layer": "joint_meta_only",
        "admission_scope": (
            "with_causal_21d_admission"
            if spec.winner_requires_admission else "without_admission"
        ),
        "score_column": "joint_meta_causal_21d_common_bps",
        "matched_population_sha256": population_sha,
        "top_fraction": spec.selection_top_fraction,
    }
    report_tails = tuple(sorted(set(map(float, spec.top_fractions)).union(_REQUIRED_REPORT_TAILS)))
    metrics.to_parquet(stage / "per_side_month_base_meta_admission_metrics.parquet", index=False)
    _zero_complete_side_month_report(
        predictions_by_cell=predictions_by_cell, common_by_layer=common_by_layer,
        cells=choices, top_fractions=report_tails,
    ).to_parquet(stage / "per_cell_side_month_zero_complete_metrics.parquet", index=False)
    fold_frames = []
    map_audits = []
    for ordinal, cell in enumerate(choices):
        checkpoint = checkpoint_root / f"{ordinal:04d}__{cell.cell_id}"
        fold_frames.append(pd.read_parquet(checkpoint / "fold_provenance.parquet").assign(cell_id=cell.cell_id))
        map_audits.append(pd.read_parquet(checkpoint / "causal_21d_map_audit.parquet").assign(cell_id=cell.cell_id))
    pd.concat(fold_frames, ignore_index=True).to_parquet(
        stage / "fold_windows_and_label_cutoffs.parquet", index=False
    )
    pd.concat(map_audits, ignore_index=True).to_parquet(
        stage / "map_admission_support.parquet", index=False
    )
    _feature_contract_rows(choices).to_parquet(
        stage / "feature_contracts_and_source_hashes.parquet", index=False
    )
    gate_table.to_csv(stage / "joint_meta_winner_comparison.csv", index=False)
    (stage / "winner.json").write_text(
        json.dumps(winner, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema": SCHEMA,
        "architecture": "native_broad_and_tail_states_direct_FQ3_joint_then_21d_common_bps",
        "legacy_mapped_bps_residual": False,
        "tail_selection": "prior_score_only_global_in_time_top_20_30_40_50",
        "execution": "explicit_cells_sequential_with_atomic_cell_checkpoints",
        "ranking": "pooled_global_only_after_side_local_causal_21d_common_bps_mapping",
        "base_metrics": "diagnostic_only",
        "winner_selection": "joint_meta_only",
        "winner_gate": (
            "positive_aggregate_top10_lift_and_nonnegative_worst_latest_month_lift"
            "_with_adequate_support_vs_declared_control"
        ),
        "top_k_denominator": "common_globally_scored_population_before_admission",
        "reported_global_top_fractions": list(report_tails),
        "matched_population_sha256": population_sha,
        "cells": execution, "winner": winner, "spec": asdict(spec),
        "launch_manifest": dict(launch_manifest or {}),
        "resume": {
            "enabled": bool(resume), "resumed_cell_count": resumed_cells,
            "executed_cell_count": len(choices) - resumed_cells,
            "checkpoint_directory": (
                str(checkpoint_root.resolve()) if checkpoint_directory is not None else None
            ),
        },
    }
    (stage / "run_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    checksums = {
        path.relative_to(stage).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in sorted(stage.rglob("*")) if path.is_file()
    }
    (stage / "checksums.json").write_text(
        json.dumps(checksums, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    stage.replace(output)
    return StageIVNativeRunResult(output, metrics, winner, manifest)


__all__ = [
    "SCHEMA", "FROZEN_MODEL_SERIALIZATION_FORMAT", "DirectMetaFitter", "NativeBaseFitter", "NativeBasePrediction",
    "StageIVNativeCell", "StageIVNativePlan", "StageIVNativeRunResult",
    "StageIVNativeFrozenArtifact", "StageIVNativeFrozenOOSPlan", "StageIVNativeFrozenOOSResult",
    "StageIVNativeRunnerError", "StageIVNativeRunnerSpec",
    "generate_stage_iv_native_side_oof", "run_stage_iv_native_artifact_sweep",
    "run_stage_iv_native_frozen_oos",
]
