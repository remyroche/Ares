"""Immutable, sequential Stage-IV/V experiment orchestration.

The individual Stage-IV and Stage-V modules deliberately do not decide which
experiments to run.  This module supplies the missing *research control
plane*: an explicitly declared Stage-IV cell sequence, matched pooled-global
comparison, a deterministic winner freeze, and a Stage-V feature-only OOD
contract that can be handed to a later Stage-III or Stage-IV run.

It is intentionally in-memory and side-effect free.  Materialisation,
training, artifact publication, and scheduling remain the responsibility of
the experiment runner.  In particular, an OOD ``controller`` here means an
ordered model-input context contract; it is never a post-score re-ranker,
admission rule, or hard regime router.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .stage_iv_broad_to_tail import (
    TAIL_FRACTIONS,
    ModelFitter,
    StageIVPlan,
    StageIVResult,
    pooled_global_stage_iv_metrics,
    run_stage_iv_broad_to_tail_ablation,
)
from .stage_v_drift_ood import STAGE_V_FEATURE_COLUMNS, StageVContract


STAGE_IV_V_SCHEMA = "stage_iv_v_sequential_orchestration_v1"
_META_SCORE = "meta_reconstructed_expected_net_bps"
_CONTROLLERS: Mapping[str, tuple[str, ...]] = {
    # Controls only which causal Stage-V fields are supplied to a model.  It
    # never changes an already-produced score.
    "none": (),
    "soft_ood": (
        "stage_v_reference_ready",
        "stage_v_ood_score",
    ),
    "grouped_ood": (
        "stage_v_reference_ready",
        "stage_v_group_activation_mean",
        "stage_v_group_activation_max",
        "stage_v_group_coactivation_mean",
        "stage_v_group_coactivation_max",
        "stage_v_group_pattern_ood",
        "stage_v_group_drift_mean",
        "stage_v_group_drift_max",
        "stage_v_model_drift",
        "stage_v_ood_score",
    ),
    "full_soft_context": STAGE_V_FEATURE_COLUMNS,
}


class StageIVVOrchestrationError(ValueError):
    """Raised when an immutable sequential experiment contract is violated."""


def _json_digest(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _ordered_feature_digest(feature_names: Sequence[str]) -> str:
    """Deliberately matches the Stage-III ordered feature-list hash."""
    payload = list(dict.fromkeys(str(name) for name in feature_names))
    return sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()


def _finite_unique_names(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    names = tuple(str(value) for value in values)
    if not names or any(not value.strip() for value in names):
        raise StageIVVOrchestrationError(f"{name} must be non-empty ordered feature names")
    if len(set(names)) != len(names):
        raise StageIVVOrchestrationError(f"{name} must not contain duplicates")
    return names


def _plan_snapshot(plan: StageIVPlan) -> dict[str, Any]:
    """A compact input identity, including feature/target/value values.

    The plan object is frozen but contains a mutable DataFrame.  Snapshotting
    all values used by the three layers binds a winning cell to what was
    actually compared, rather than merely to a Python object identity.
    """
    raw = plan.frame.copy()
    raw.columns = raw.columns.astype(str)
    feature_names = tuple(dict.fromkeys(
        [*map(str, plan.broad_feature_names), *map(str, plan.tail_feature_names), *map(str, plan.meta_feature_names)]
    ))
    missing = [name for name in feature_names if name not in raw]
    if missing:
        raise StageIVVOrchestrationError(f"Stage-IV plan has missing frozen features: {missing[:12]}")
    matrix = raw.loc[:, feature_names]
    # The JSON split form is stable for the numeric model inputs accepted by
    # Stage IV and preserves NaN position.  This is metadata only, not an
    # artifact writer.
    return {
        "side": str(plan.side).lower(),
        "candidate_ids": [str(value) for value in plan.candidate_ids],
        "decision_timestamps": [str(value) for value in plan.decision_timestamps],
        "label_available_timestamps": [str(value) for value in plan.label_available_timestamps],
        "base_target": [float(value) for value in plan.base_target],
        "tail_target": None if plan.tail_target is None else [float(value) for value in plan.tail_target],
        "exact_net_bps": [float(value) for value in plan.exact_net_bps],
        "meta_target": None if plan.meta_target is None else [float(value) for value in plan.meta_target],
        "sample_weight": None if plan.sample_weight is None else [float(value) for value in plan.sample_weight],
        "features": {
            "broad": list(map(str, plan.broad_feature_names)),
            "tail": list(map(str, plan.tail_feature_names)),
            "meta": list(map(str, plan.meta_feature_names)),
        },
        "feature_values": matrix.to_json(orient="split", date_format="iso", double_precision=15),
        "params": {
            "broad": dict(plan.broad_params), "tail": dict(plan.tail_params), "meta": dict(plan.meta_params),
        },
        "cost_bps": float(plan.cost_bps),
        "n_validation_folds": int(plan.n_validation_folds),
        "burn_in_months": int(plan.burn_in_months),
    }


@dataclass(frozen=True)
class StageIVCellSpec:
    """One explicitly declared broad-to-tail cell, never a Cartesian product."""

    cell_id: str
    tail_fraction: float
    broad_min_train_rows: int
    tail_min_train_rows: int
    meta_min_train_rows: int
    min_handoff_history_rows: int
    broad_output_route: str

    def validate(self) -> None:
        if not str(self.cell_id).strip():
            raise StageIVVOrchestrationError("Stage-IV cell_id must be non-empty")
        if float(self.tail_fraction) not in TAIL_FRACTIONS:
            raise StageIVVOrchestrationError("Stage-IV tail_fraction must be exactly 20%, 30%, 40%, or 50%")
        if str(self.broad_output_route).lower() not in {"neither", "tail", "meta", "both"}:
            raise StageIVVOrchestrationError("Stage-IV broad_output_route is invalid")
        for name in (
            "broad_min_train_rows", "tail_min_train_rows", "meta_min_train_rows", "min_handoff_history_rows",
        ):
            if int(getattr(self, name)) < 1:
                raise StageIVVOrchestrationError(f"Stage-IV {name} must be positive")

    @property
    def digest(self) -> str:
        self.validate()
        return _json_digest(asdict(self))


@dataclass(frozen=True)
class StageIVCell:
    """Frozen plans for the sides of one predeclared sequential cell."""

    spec: StageIVCellSpec
    plans: tuple[StageIVPlan, ...]
    source_lineage: Mapping[str, str]

    def validate(self) -> None:
        self.spec.validate()
        if not self.plans:
            raise StageIVVOrchestrationError("Stage-IV cell must contain one or two side plans")
        sides = [str(plan.side).lower() for plan in self.plans]
        if len(set(sides)) != len(sides) or any(side not in {"long", "short"} for side in sides):
            raise StageIVVOrchestrationError("Stage-IV cell plans must be isolated long/short plans")
        for plan in self.plans:
            expected = {
                "tail_fraction": float(self.spec.tail_fraction),
                "broad_min_train_rows": int(self.spec.broad_min_train_rows),
                "tail_min_train_rows": int(self.spec.tail_min_train_rows),
                "meta_min_train_rows": int(self.spec.meta_min_train_rows),
                "min_handoff_history_rows": int(self.spec.min_handoff_history_rows),
                "broad_output_route": str(self.spec.broad_output_route).lower(),
            }
            actual = {
                "tail_fraction": float(plan.tail_fraction),
                "broad_min_train_rows": int(plan.broad_min_train_rows),
                "tail_min_train_rows": int(plan.tail_min_train_rows),
                "meta_min_train_rows": int(plan.meta_min_train_rows),
                "min_handoff_history_rows": int(plan.min_handoff_history_rows),
                "broad_output_route": str(plan.broad_output_route).lower(),
            }
            if actual != expected:
                raise StageIVVOrchestrationError(
                    f"Stage-IV cell {self.spec.cell_id!r} plan {plan.side!r} does not match its declared immutable spec"
                )
        lineage = {str(key): str(value) for key, value in self.source_lineage.items()}
        if not lineage or any(len(value) != 64 or len(set(value)) == 1 for value in lineage.values()):
            raise StageIVVOrchestrationError("Stage-IV source_lineage needs non-placeholder SHA-256 digests")

    @property
    def input_digest(self) -> str:
        self.validate()
        return _json_digest({
            "spec": asdict(self.spec),
            "source_lineage": dict(sorted((str(k), str(v)) for k, v in self.source_lineage.items())),
            "plans": [_plan_snapshot(plan) for plan in self.plans],
        })


@dataclass(frozen=True)
class StageIVFrozenWinner:
    cell_id: str
    cell_spec: Mapping[str, Any]
    cell_spec_sha256: str
    cell_input_sha256: str
    matched_population_sha256: str
    score_column: str
    selection_top_fraction: float
    selected_global_rows: int
    selected_net_bps_per_trade: float
    source_lineage: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StageIVSweepResult:
    """Full in-memory results plus a compact, winner-only manifest."""

    cell_results: Mapping[str, StageIVResult]
    matched_metrics: pd.DataFrame
    winner: StageIVFrozenWinner
    manifest: Mapping[str, Any]


def _matched_key_digest(keys: Sequence[Any]) -> str:
    return _json_digest([str(key) for key in sorted(map(str, keys))])


def _validated_matched_ledger(
    cell_results: Mapping[str, StageIVResult],
    *,
    score_column: str,
) -> tuple[dict[str, pd.DataFrame], str]:
    """Intersect complete final-score rows and verify common economics exactly."""
    scored: dict[str, pd.DataFrame] = {}
    common: set[str] | None = None
    for cell_id, result in cell_results.items():
        frame = result.predictions.loc[result.predictions[score_column].notna()].copy()
        if frame.empty or frame.candidate_key.duplicated().any():
            raise StageIVVOrchestrationError(f"Stage-IV cell {cell_id!r} has no unique final-score OOF ledger")
        keys = set(frame.candidate_key.astype(str))
        common = keys if common is None else common & keys
        scored[cell_id] = frame
    if not common:
        raise StageIVVOrchestrationError("Stage-IV cells have no shared final-score candidate population")
    key_digest = _matched_key_digest(tuple(common))
    comparison: pd.DataFrame | None = None
    matched: dict[str, pd.DataFrame] = {}
    invariants = ("side_name", "decision_ts", "label_available_ts", "exact_net_bps", "cost_bps")
    for cell_id, frame in scored.items():
        local = frame.loc[frame.candidate_key.astype(str).isin(common)].copy()
        local = local.sort_values("candidate_key", kind="stable").reset_index(drop=True)
        if len(local) != len(common):
            raise AssertionError("matched Stage-IV intersection lost a candidate")
        if comparison is None:
            comparison = local.loc[:, ["candidate_key", *invariants]].copy()
        else:
            current = local.loc[:, ["candidate_key", *invariants]]
            if not current.candidate_key.astype(str).equals(comparison.candidate_key.astype(str)):
                raise StageIVVOrchestrationError("Stage-IV cells do not agree on matched candidate identity")
            for column in invariants:
                left = comparison[column]
                right = current[column]
                if pd.api.types.is_numeric_dtype(left):
                    equal = np.allclose(left.to_numpy(float), right.to_numpy(float), rtol=0.0, atol=1e-6, equal_nan=False)
                else:
                    equal = left.astype(str).equals(right.astype(str))
                if not equal:
                    raise StageIVVOrchestrationError(
                        f"Stage-IV cells cannot be matched: {column!r} differs for the same candidate"
                    )
        matched[cell_id] = local
    return matched, key_digest


def run_stage_iv_sequential_sweep(
    cells: Sequence[StageIVCell],
    *,
    fitter: ModelFitter | None = None,
    selection_top_fraction: float = 0.10,
    require_all_tail_fractions: bool = True,
) -> StageIVSweepResult:
    """Run an ordered, predeclared Stage-IV sweep and freeze one winner.

    A cell is a complete treatment, including all three independent burn-ins
    and the broad-output route.  The function intentionally never expands
    cells into a factorial.  All cells are compared on the exact intersection
    of final-score candidates and selected once globally after side scores are
    already in a common unit.
    """
    if not cells:
        raise StageIVVOrchestrationError("Stage-IV sweep requires explicit cells")
    if not 0.0 < float(selection_top_fraction) <= 1.0:
        raise StageIVVOrchestrationError("Stage-IV selection_top_fraction must lie in (0, 1]")
    ids = [str(cell.spec.cell_id) for cell in cells]
    if len(set(ids)) != len(ids):
        raise StageIVVOrchestrationError("Stage-IV sweep cell_id values must be unique")
    for cell in cells:
        cell.validate()
    fractions = {float(cell.spec.tail_fraction) for cell in cells}
    if require_all_tail_fractions and fractions != set(TAIL_FRACTIONS):
        raise StageIVVOrchestrationError("Stage-IV primary sweep must explicitly cover x=20/30/40/50%")

    # This is deliberately a serial loop.  The declared order is part of the
    # manifest and makes a stopped/restarted research run auditable.
    results: dict[str, StageIVResult] = {}
    execution: list[dict[str, Any]] = []
    for ordinal, cell in enumerate(cells):
        result = run_stage_iv_broad_to_tail_ablation(cell.plans, fitter=fitter)
        results[cell.spec.cell_id] = result
        execution.append({
            "ordinal": ordinal,
            "cell_id": cell.spec.cell_id,
            "cell_spec_sha256": cell.spec.digest,
            "cell_input_sha256": cell.input_digest,
            "sides": [str(plan.side).lower() for plan in cell.plans],
        })

    matched, population_digest = _validated_matched_ledger(results, score_column=_META_SCORE)
    metric_frames: list[pd.DataFrame] = []
    for cell in cells:
        cell_id = cell.spec.cell_id
        metrics = pooled_global_stage_iv_metrics(
            matched[cell_id], score_column=_META_SCORE,
            layer="meta_residual_reconstructed_matched_population",
            top_fractions=(float(selection_top_fraction),),
        ).copy()
        metrics.insert(0, "cell_id", cell_id)
        metrics["cell_spec_sha256"] = cell.spec.digest
        metrics["cell_input_sha256"] = cell.input_digest
        metrics["matched_population_sha256"] = population_digest
        metrics["comparison_population"] = "intersection_of_final_strict_oof_rows_before_one_pooled_global_rank"
        metric_frames.append(metrics)
    matched_metrics = pd.concat(metric_frames, ignore_index=True)
    pooled = matched_metrics.loc[matched_metrics.scope.eq("pooled_global")].copy()
    if len(pooled) != len(cells):
        raise AssertionError("Stage-IV matched sweep must yield one pooled metric per explicit cell")
    # Stable cell-id tie breaking is predeclared; it avoids post-hoc judgement
    # where identical observed economics occur.
    chosen = pooled.sort_values(
        ["net_bps_per_trade", "gross_bps_per_trade", "cell_id"],
        ascending=[False, False, True], kind="stable",
    ).iloc[0]
    winner_cell = next(cell for cell in cells if cell.spec.cell_id == str(chosen.cell_id))
    winner = StageIVFrozenWinner(
        cell_id=str(chosen.cell_id),
        cell_spec=asdict(winner_cell.spec),
        cell_spec_sha256=winner_cell.spec.digest,
        cell_input_sha256=winner_cell.input_digest,
        matched_population_sha256=population_digest,
        score_column=_META_SCORE,
        selection_top_fraction=float(selection_top_fraction),
        selected_global_rows=int(chosen.selected_global_rows),
        selected_net_bps_per_trade=float(chosen.net_bps_per_trade),
        source_lineage=dict(sorted((str(k), str(v)) for k, v in winner_cell.source_lineage.items())),
    )
    manifest = {
        "schema": STAGE_IV_V_SCHEMA,
        "stage": "IV",
        "execution": "sequential_explicit_cells_no_factorial_expansion",
        "ranking": "pooled_global_once_after_common_bps_scores_no_timestamp_month_or_side_rerank",
        "comparison_population": "intersection_of_final_strict_oof_rows_before_one_pooled_global_rank",
        "matched_population_sha256": population_digest,
        "selection": {
            "score_column": _META_SCORE,
            "top_fraction": float(selection_top_fraction),
            "tie_break": "net_bps_per_trade_then_gross_bps_per_trade_then_cell_id",
        },
        "cells": execution,
        "winner": winner.to_dict(),
        # Compact by construction: no per-arm prediction copies or models are
        # retained in the portable manifest.  The in-memory result has them
        # only for immediate diagnostics/publication by the caller.
        "compact_outputs": ("matched_metrics", "winner", "cell_input_hashes"),
    }
    return StageIVSweepResult(results, matched_metrics, winner, manifest)


@dataclass(frozen=True)
class StageVFrozenFeatureContract:
    """A side/layer-local Stage-V context addition to a raw feature contract."""

    side: str
    layer: str
    controller: str
    raw_feature_names: tuple[str, ...]
    context_feature_names: tuple[str, ...]
    state_sha256: str
    ordered_model_feature_names: tuple[str, ...]
    source_feature_contract_sha256: str
    model_feature_contract_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def contract_sha256(self) -> str:
        return _json_digest(self.to_dict())


def freeze_stage_v_feature_contract(
    *,
    contract: StageVContract,
    raw_feature_names: Sequence[str],
    state: Mapping[str, Any],
    controller: str,
    context_feature_names: Sequence[str] | None = None,
) -> StageVFrozenFeatureContract:
    """Freeze one causal OOD input contract after an ablation decision.

    ``state`` must be fit on training rows only for precisely the requested
    side/layer.  The resulting ordered hash is byte-compatible with
    ``stage_iii_feature_contract_sha256`` and can be used as a Stage-III input
    list or appended to the matching Stage-IV layer list.
    """
    normalized = contract.normalized()
    raw = _finite_unique_names(raw_feature_names, name="raw_feature_names")
    name = str(controller).lower()
    if name not in _CONTROLLERS:
        raise StageIVVOrchestrationError(f"unknown Stage-V controller {controller!r}")
    if not isinstance(state, Mapping) or not bool(state.get("enabled", False)):
        raise StageIVVOrchestrationError("Stage-V feature contract requires an enabled training-only state")
    state_contract = state.get("contract")
    expected = {"side": normalized.side, "layer": normalized.layer}
    if not isinstance(state_contract, Mapping) or {
        "side": str(state_contract.get("side", "")).lower(),
        "layer": str(state_contract.get("layer", "")).lower(),
    } != expected:
        raise StageIVVOrchestrationError("Stage-V state cannot cross side/layer feature contracts")
    if str(state.get("reference_role", "")) != "train_only" or not bool(state.get("soft_context_only", False)):
        raise StageIVVOrchestrationError("Stage-V state must prove a training-only soft context")
    selected = tuple(_CONTROLLERS[name] if context_feature_names is None else context_feature_names)
    if tuple(dict.fromkeys(map(str, selected))) != tuple(map(str, selected)):
        raise StageIVVOrchestrationError("Stage-V context_feature_names must be unique and ordered")
    selected = tuple(map(str, selected))
    invalid = sorted(set(selected) - set(STAGE_V_FEATURE_COLUMNS))
    if invalid:
        raise StageIVVOrchestrationError(f"Stage-V contract has non-causal/unknown context fields: {invalid}")
    # Controller naming is an audit label, but an explicit subset is allowed
    # only for a non-empty controller: this supports a selected ablation while
    # preventing a silently unlabelled custom controller.
    if name == "none" and selected:
        raise StageIVVOrchestrationError("Stage-V controller 'none' cannot inject context features")
    if set(raw) & set(selected):
        raise StageIVVOrchestrationError("Stage-V raw and context feature names must not overlap")
    ordered = (*raw, *selected)
    return StageVFrozenFeatureContract(
        side=normalized.side,
        layer=normalized.layer,
        controller=name,
        raw_feature_names=raw,
        context_feature_names=selected,
        state_sha256=_json_digest(state),
        ordered_model_feature_names=ordered,
        source_feature_contract_sha256=_ordered_feature_digest(raw),
        model_feature_contract_sha256=_ordered_feature_digest(ordered),
    )


def apply_stage_v_contract_to_stage_iv_plan(
    plan: StageIVPlan,
    *,
    base_contract: StageVFrozenFeatureContract | None = None,
    meta_contract: StageVFrozenFeatureContract | None = None,
) -> StageIVPlan:
    """Attach selected Stage-V context to the matching Stage-IV model layers.

    Base context is supplied to both broad and tail base models.  Meta context
    is supplied only to the residual layer.  This preserves side/layer
    isolation; callers cannot, for example, put a short/meta OOD state into a
    long/broad model.  The function only changes frozen *input lists* and
    never modifies scores, admission, or ranking.
    """
    side = str(plan.side).lower()
    def _check(value: StageVFrozenFeatureContract | None, *, expected_layer: str) -> tuple[str, ...]:
        if value is None:
            return ()
        if value.side != side or value.layer != expected_layer:
            raise StageIVVOrchestrationError(
                f"Stage-V {value.side}/{value.layer} context cannot enter Stage-IV {side}/{expected_layer}"
            )
        missing = [name for name in value.context_feature_names if name not in plan.frame]
        if missing:
            raise StageIVVOrchestrationError(f"Stage-IV plan has not materialised frozen Stage-V fields: {missing}")
        return value.context_feature_names
    base_extra = _check(base_contract, expected_layer="base")
    meta_extra = _check(meta_contract, expected_layer="meta")
    return replace(
        plan,
        broad_feature_names=tuple(dict.fromkeys((*map(str, plan.broad_feature_names), *base_extra))),
        tail_feature_names=tuple(dict.fromkeys((*map(str, plan.tail_feature_names), *base_extra))),
        meta_feature_names=tuple(dict.fromkeys((*map(str, plan.meta_feature_names), *meta_extra))),
    )


@dataclass(frozen=True)
class StageVControllerArm:
    """One matched OOD model-input ablation, described rather than executed."""

    arm_id: str
    contracts: tuple[StageVFrozenFeatureContract, ...]

    def validate(self) -> None:
        if not str(self.arm_id).strip() or not self.contracts:
            raise StageIVVOrchestrationError("Stage-V arm needs an id and one or more frozen contracts")
        keys = [(contract.side, contract.layer) for contract in self.contracts]
        if len(set(keys)) != len(keys):
            raise StageIVVOrchestrationError("Stage-V arm has duplicate side/layer contracts")


@dataclass(frozen=True)
class StageVControllerSelection:
    winner_arm_id: str
    matched_population_sha256: str
    selection_top_fraction: float
    selected_net_bps_per_trade: float
    manifest: Mapping[str, Any]


def select_stage_v_controller(
    arm_ledgers: Mapping[str, pd.DataFrame],
    *,
    controller_arms: Mapping[str, StageVControllerArm] | None = None,
    score_column: str,
    selection_top_fraction: float = 0.10,
) -> tuple[pd.DataFrame, StageVControllerSelection]:
    """Compare OOD arms on one shared candidate set without any local rank.

    The caller trains/evaluates each model using the supplied frozen contract
    and provides its OOF ledger.  This function is intentionally unable to
    adjust scores: it verifies unchanged economics/identity across arms,
    intersects scored rows, and performs a single pooled-global rank per arm.
    """
    if not arm_ledgers:
        raise StageIVVOrchestrationError("Stage-V controller selection needs OOF ledgers")
    if not 0.0 < float(selection_top_fraction) <= 1.0:
        raise StageIVVOrchestrationError("Stage-V selection_top_fraction must lie in (0, 1]")
    arm_ids = {str(arm_id) for arm_id in arm_ledgers}
    if controller_arms is None:
        raise StageIVVOrchestrationError(
            "Stage-V controller selection requires frozen controller_arms; unnamed score-ledger comparison is diagnostic only"
        )
    declared = {str(arm_id): arm for arm_id, arm in controller_arms.items()}
    if set(declared) != arm_ids:
        raise StageIVVOrchestrationError("Stage-V controller arms must match scored OOF arm ids exactly")
    for arm_id, arm in declared.items():
        if str(arm.arm_id) != arm_id:
            raise StageIVVOrchestrationError("Stage-V controller arm_id must match its mapping key")
        arm.validate()
    required = {"candidate_key", "side_name", "decision_ts", "exact_net_bps", "cost_bps", score_column}
    scored: dict[str, pd.DataFrame] = {}
    common: set[str] | None = None
    for arm_id, ledger in arm_ledgers.items():
        missing = sorted(required - set(ledger.columns))
        if missing:
            raise StageIVVOrchestrationError(f"Stage-V arm {arm_id!r} lacks OOF ledger columns: {missing}")
        local = ledger.loc[ledger[score_column].notna()].copy()
        if local.empty or local.candidate_key.duplicated().any():
            raise StageIVVOrchestrationError(f"Stage-V arm {arm_id!r} lacks unique scored OOF rows")
        common = set(local.candidate_key.astype(str)) if common is None else common & set(local.candidate_key.astype(str))
        declared_sides = {contract.side for contract in declared[str(arm_id)].contracts}
        ledger_sides = set(local.side_name.astype(str).str.lower())
        if not ledger_sides.issubset(declared_sides):
            raise StageIVVOrchestrationError(
                f"Stage-V arm {arm_id!r} has scored sides without frozen feature contracts: {sorted(ledger_sides - declared_sides)}"
            )
        scored[str(arm_id)] = local
    if not common:
        raise StageIVVOrchestrationError("Stage-V arms have no common scored OOF population")
    population_digest = _matched_key_digest(tuple(common))
    baseline: pd.DataFrame | None = None
    rows: list[pd.DataFrame] = []
    for arm_id, ledger in scored.items():
        local = ledger.loc[ledger.candidate_key.astype(str).isin(common)].copy()
        local = local.sort_values("candidate_key", kind="stable").reset_index(drop=True)
        economics = local.loc[:, ["candidate_key", "side_name", "decision_ts", "exact_net_bps", "cost_bps"]]
        if baseline is None:
            baseline = economics
        elif not economics.astype(str).equals(baseline.astype(str)):
            raise StageIVVOrchestrationError("Stage-V arms must share candidate identity and realised economics")
        metric = pooled_global_stage_iv_metrics(
            local, score_column=score_column, layer="stage_v_ood_controller_matched",
            top_fractions=(float(selection_top_fraction),),
        ).copy()
        metric.insert(0, "arm_id", arm_id)
        metric["matched_population_sha256"] = population_digest
        metric["selection"] = "pooled_global_once_no_timestamp_month_or_side_rerank"
        rows.append(metric)
    metrics = pd.concat(rows, ignore_index=True)
    pooled = metrics.loc[metrics.scope.eq("pooled_global")].sort_values(
        ["net_bps_per_trade", "gross_bps_per_trade", "arm_id"], ascending=[False, False, True], kind="stable",
    )
    best = pooled.iloc[0]
    manifest = {
        "schema": STAGE_IV_V_SCHEMA,
        "stage": "V",
        "selection": "matched_ood_controller_feature_ablation",
        "ranking": "pooled_global_once_after_model_score_no_timestamp_month_or_side_rerank",
        "matched_population_sha256": population_digest,
        "score_column": score_column,
        "top_fraction": float(selection_top_fraction),
        "winner_arm_id": str(best.arm_id),
        "controller_contracts": {
            arm_id: [
                {**contract.to_dict(), "contract_sha256": contract.contract_sha256}
                for contract in arm.contracts
            ]
            for arm_id, arm in sorted(declared.items())
        },
        "compact_outputs": ("matched_metrics", "winner_arm_id", "matched_population_sha256"),
    }
    return metrics, StageVControllerSelection(
        winner_arm_id=str(best.arm_id), matched_population_sha256=population_digest,
        selection_top_fraction=float(selection_top_fraction),
        selected_net_bps_per_trade=float(best.net_bps_per_trade), manifest=manifest,
    )


__all__ = [
    "STAGE_IV_V_SCHEMA", "StageIVVOrchestrationError", "StageIVCellSpec", "StageIVCell",
    "StageIVFrozenWinner", "StageIVSweepResult", "run_stage_iv_sequential_sweep",
    "StageVFrozenFeatureContract", "freeze_stage_v_feature_contract",
    "apply_stage_v_contract_to_stage_iv_plan", "StageVControllerArm", "StageVControllerSelection",
    "select_stage_v_controller",
]
