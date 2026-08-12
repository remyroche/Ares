"""Generic target-adapter strict OOF for the post-ablation Stage-I winner.

This module leaves the schema-v1 R3/Huber generator intact.  A v2 winner uses
this path and must provide both immutable target contracts explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .prequential_r3_value_map import (
    PrequentialR3ValueMapConfig,
    prequential_same_side_r3_value_map,
)
from .stage_i_base_target_ablation import recover_ordinal_simplex
from .stage_i_base_target_ablation import training_weights
from .stage_i_strict_oof import _multiclass_probabilities, _strict_train_mask, _validation_blocks
from .stage_i_ranking import stable_stage_i_topk_positions
from .stage_i_target_adapter import (
    CUMULATIVE_ORDINAL5_O,
    FOLD_QUANTILE_RESIDUAL3,
    LEGACY_HUBER_RESIDUAL,
    LEGACY_R3_MULTICLASS3,
    SOFT_SCALAR_S,
    FoldQuantileResidualState,
    StageITargetContract,
    fit_cumulative_ordinal5_estimator,
    fit_fold_quantile_residual3,
    reconstruct_fold_quantile_residual3,
    verify_target_contract,
)


SCHEMA = "stage_i_target_adapter_strict_oof_v2"


@dataclass(frozen=True)
class StageIAdapterStrictOOFPlan:
    side: str
    frame: pd.DataFrame
    contract_frame: pd.DataFrame
    candidate_ids: Sequence[Any]
    decision_timestamps: Sequence[Any]
    label_available_timestamps: Sequence[Any]
    base_target: Sequence[float]
    exact_gross_bps: Sequence[float]
    exact_net_bps: Sequence[float]
    target_valid: Sequence[bool]
    sample_weight: Sequence[float]
    base_target_contract: StageITargetContract
    meta_target_contract: StageITargetContract
    base_feature_names: Sequence[str]
    meta_feature_names: Sequence[str]
    base_params: Mapping[str, Any]
    meta_params: Mapping[str, Any]
    candidate_selected: Sequence[bool] | None
    candidate_fraction: float = 0.30
    runtime_base_target_contract: StageITargetContract | None = None
    runtime_meta_target_contract: StageITargetContract | None = None
    meta_sample_weight: Sequence[float] | None = None
    n_validation_folds: int = 4
    min_train_rows: int = 500


@dataclass(frozen=True)
class StageIAdapterStrictOOFResult:
    side: str
    predictions: pd.DataFrame
    fold_provenance: pd.DataFrame
    manifest: Mapping[str, Any]


def _vector(values: Sequence[Any], n: int, *, name: str, dtype: Any) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).reshape(-1)
    if len(result) != n:
        raise ValueError(f"{name} must be row-aligned")
    return result


def _utc(values: Sequence[Any], n: int, *, name: str) -> pd.Series:
    result = pd.to_datetime(pd.Series(values), utc=True, errors="coerce")
    if len(result) != n or result.isna().any():
        raise ValueError(f"{name} must be aligned UTC")
    return result


def _clean_params(params: Mapping[str, Any], *, objective: str, num_class: int | None = None) -> dict[str, Any]:
    output = dict(params)
    output["objective"] = objective
    if num_class is None:
        output.pop("num_class", None)
    else:
        output["num_class"] = int(num_class)
    return output


def _fit_base_fold(
    family: str,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    weight: np.ndarray,
    x_valid: pd.DataFrame,
    params: Mapping[str, Any],
    fit_model: Callable[..., Any],
) -> tuple[np.ndarray, np.ndarray | None]:
    if family == CUMULATIVE_ORDINAL5_O:
        model = fit_cumulative_ordinal5_estimator(
            x_train, y_train.astype(np.int8), weight,
            params=_clean_params(params, objective="binary"),
        )
        survival = model.predict_cumulative_probability(x_valid)
        simplex = recover_ordinal_simplex(survival)
        return (simplex @ (np.arange(5) / 4.0)).astype(np.float32), simplex
    if family == SOFT_SCALAR_S:
        model = fit_model(
            x_train, y_train.astype(np.float32), weight, classifier=False,
            params=_clean_params(params, objective="regression_l1"),
            objective_mode="stage_i_soft_scalar_S_oof",
        )
        raw = np.asarray(model.predict(x_valid), dtype=np.float32).reshape(-1)
        return np.clip(raw, 0.0, 1.0), None
    if family == LEGACY_R3_MULTICLASS3:
        model = fit_model(
            x_train, y_train.astype(np.int8), weight, classifier=True,
            params=_clean_params(params, objective="multiclass", num_class=3),
            objective_mode="stage_i_legacy_R3_oof",
        )
        probability = _multiclass_probabilities(model, x_valid)
        return (probability[:, 2] - probability[:, 0]).astype(np.float32), probability
    raise ValueError(f"unsupported explicit base adapter: {family}")


def _map_score_domain(family: str, score: np.ndarray) -> np.ndarray:
    # The current prequential mapper has a fixed [-1,1] domain. S and O retain
    # their raw [0,1] outputs in the ledger; only the mapper coordinate is an
    # affine, order-preserving transform.
    return (2.0 * score - 1.0).astype(np.float32) if family in {SOFT_SCALAR_S, CUMULATIVE_ORDINAL5_O} else score


def generate_stage_i_adapter_strict_oof(
    plan: StageIAdapterStrictOOFPlan,
    *,
    fit_model: Callable[..., Any],
) -> StageIAdapterStrictOOFResult:
    """Generate side-local base and candidate-trained/meta-full-scored OOF."""

    side = str(plan.side).lower()
    if side not in {"long", "short"}:
        raise ValueError("adapter strict OOF must be side-local")
    if plan.base_target_contract.layer != "base" or plan.meta_target_contract.layer != "meta":
        raise ValueError("base/meta target contracts cannot be inferred from layer")
    if plan.meta_target_contract.family not in {FOLD_QUANTILE_RESIDUAL3, LEGACY_HUBER_RESIDUAL}:
        raise ValueError("unsupported explicit meta adapter")
    frame = plan.frame.copy()
    n = len(frame)
    runtime_base = plan.runtime_base_target_contract or plan.base_target_contract
    runtime_meta = plan.runtime_meta_target_contract or plan.meta_target_contract
    if runtime_base.rows != n or runtime_meta.rows != n:
        raise ValueError("adapter target contract row drift")
    if (
        runtime_base.family != plan.base_target_contract.family
        or runtime_base.geometry != plan.base_target_contract.geometry
        or runtime_meta.family != plan.meta_target_contract.family
        or runtime_meta.geometry != plan.meta_target_contract.geometry
    ):
        raise ValueError("runtime target semantics differ from frozen winner")
    verify_target_contract(plan.contract_frame, runtime_base)
    verify_target_contract(plan.contract_frame, runtime_meta)
    ids = _vector(plan.candidate_ids, n, name="candidate_ids", dtype=object)
    if pd.isna(ids).any() or len(pd.unique(ids)) != n:
        raise ValueError("candidate IDs must be unique/non-null")
    decision = _utc(plan.decision_timestamps, n, name="decision_timestamps")
    available = _utc(plan.label_available_timestamps, n, name="label_available_timestamps")
    if not (available - decision).eq(pd.Timedelta(hours=12)).all():
        raise ValueError("adapter labels must resolve at decision+12h")
    valid = _vector(plan.target_valid, n, name="target_valid", dtype=bool)
    candidate = (
        None
        if plan.candidate_selected is None
        else _vector(plan.candidate_selected, n, name="candidate_selected", dtype=bool)
    )
    target = _vector(plan.base_target, n, name="base_target", dtype=np.float32)
    gross = _vector(plan.exact_gross_bps, n, name="exact_gross_bps", dtype=np.float32)
    net = _vector(plan.exact_net_bps, n, name="exact_net_bps", dtype=np.float32)
    weight = _vector(plan.sample_weight, n, name="sample_weight", dtype=np.float32)
    meta_weight = (
        np.ones(n, dtype=np.float32)
        if plan.meta_sample_weight is None
        else _vector(plan.meta_sample_weight, n, name="meta_sample_weight", dtype=np.float32)
    )
    if not np.isfinite(target[valid]).all() or not np.isfinite(gross[valid]).all() or not np.isfinite(net[valid]).all():
        raise ValueError("valid rows need finite target and winner-geometry economics")
    if not np.allclose(gross[valid] - 100.0, net[valid], atol=2e-3, rtol=0):
        raise ValueError("winning-geometry cost must be applied exactly once")
    if not np.isfinite(weight).all() or (weight < 0).any() or not np.any(weight[valid] > 0):
        raise ValueError("adapter weights are invalid")
    base_features = tuple(map(str, plan.base_feature_names))
    meta_features = tuple(map(str, plan.meta_feature_names))
    if not base_features or not meta_features or len(set(base_features)) != len(base_features) or len(set(meta_features)) != len(meta_features):
        raise ValueError("frozen feature contracts must be ordered/non-empty")
    missing_base = set(base_features).difference(frame.columns)
    if missing_base:
        raise ValueError(f"base features are absent: {sorted(missing_base)[:8]}")
    blocks = _validation_blocks(
        decision, available, n_folds=int(plan.n_validation_folds),
        min_train_rows=int(plan.min_train_rows),
    )
    raw_score = np.full(n, np.nan, dtype=np.float32)
    base_probability_width = 5 if plan.base_target_contract.family == CUMULATIVE_ORDINAL5_O else 3
    base_probability = np.full((n, base_probability_width), np.nan, dtype=np.float32)
    has_base_probability = plan.base_target_contract.family != SOFT_SCALAR_S
    base_fold = np.full(n, -1, dtype=np.int16)
    provenance: list[dict[str, Any]] = []
    for fold_id, validation_idx in enumerate(blocks):
        validation_idx = np.asarray(validation_idx, dtype=np.int32)
        start = decision.iloc[validation_idx].min()
        train_idx = np.flatnonzero(available.lt(start).to_numpy() & valid)
        if len(train_idx) < int(plan.min_train_rows):
            raise ValueError("base fold lacks prior valid support")
        weight_mode = str(
            (plan.base_target_contract.metadata.get("training_weight_contract") or {}).get(
                "mode", "uniform"
            )
        )
        regime_column = str(plan.base_target_contract.metadata.get("regime_column", ""))
        fold_weight = training_weights(
            plan.contract_frame.iloc[train_idx], target=target[train_idx],
            mode=weight_mode, regime_column=regime_column,
        )
        score, probability = _fit_base_fold(
            plan.base_target_contract.family,
            frame.iloc[train_idx].loc[:, list(base_features)], target[train_idx],
            fold_weight, frame.iloc[validation_idx].loc[:, list(base_features)],
            plan.base_params, fit_model,
        )
        raw_score[validation_idx] = score
        if probability is not None:
            base_probability[validation_idx] = probability
        base_fold[validation_idx] = fold_id
        provenance.append({
            "side": side, "layer": "base", "fold_id": fold_id,
            "target_family": plan.base_target_contract.family,
            "train_rows": len(train_idx), "validation_rows": len(validation_idx),
            "validation_start_utc": start.isoformat(),
            "train_max_label_available_utc": available.iloc[train_idx].max().isoformat(),
            "strict_prior_resolved": bool(available.iloc[train_idx].lt(start).all()),
        })
    base_scored = np.isfinite(raw_score) & valid
    if candidate is None:
        fraction = float(plan.candidate_fraction)
        if not 0.0 < fraction <= 1.0:
            raise ValueError("base candidate fraction must lie in (0,1]")
        eligible = np.flatnonzero(base_scored)
        count = max(1, int(np.ceil(fraction * len(eligible))))
        chosen = stable_stage_i_topk_positions(
            raw_score, candidate_ids=ids, decision_timestamps=decision,
            side_names=side, count=count, valid_mask=base_scored,
        )
        candidate = np.zeros(n, dtype=bool)
        candidate[chosen] = True
    mapper_score = _map_score_domain(plan.base_target_contract.family, raw_score[base_scored])
    mapped_valid, map_audit, map_manifest = prequential_same_side_r3_value_map(
        exact_net_bps=net[base_scored], decision_timestamps=decision.iloc[base_scored],
        label_available_timestamps=available.iloc[base_scored], side=side,
        score=mapper_score, config=PrequentialR3ValueMapConfig(side=side),
    )
    mapped = np.full(n, np.nan, dtype=np.float32)
    mapped[base_scored] = mapped_valid

    # Direct same-side base outputs are retained without replacing them by the
    # bps map. The map is an additional fixed offset/value feature.
    meta_design = frame.copy()
    meta_design["base_raw_score"] = raw_score
    meta_design["prequential_base_expected_net_bps"] = mapped
    if has_base_probability:
        for index in range(base_probability.shape[1]):
            meta_design[f"base_state_p{index}"] = base_probability[:, index]
    raw_meta_features = [name for name in meta_features if name in meta_design]
    missing_meta = sorted(set(meta_features).difference(raw_meta_features))
    if missing_meta:
        raise ValueError(f"meta features are absent after direct base handoff: {missing_meta[:8]}")

    correction = np.full(n, np.nan, dtype=np.float32)
    reconstructed = np.full(n, np.nan, dtype=np.float32)
    meta_probability = np.full((n, 3), np.nan, dtype=np.float32)
    meta_fold = np.full(n, -1, dtype=np.int16)
    meta_states: list[dict[str, Any]] = []
    for fold_id, validation_idx in enumerate(blocks):
        validation_idx = np.asarray(validation_idx, dtype=np.int32)
        start = decision.iloc[validation_idx].min()
        train_mask = available.lt(start).to_numpy() & valid & base_scored & candidate
        train_idx = np.flatnonzero(train_mask)
        validation_score_idx = validation_idx[valid[validation_idx] & base_scored[validation_idx]]
        if len(train_idx) < int(plan.min_train_rows) or not len(validation_score_idx):
            provenance.append({
                "side": side, "layer": "meta", "fold_id": fold_id,
                "target_family": plan.meta_target_contract.family,
                "train_rows": len(train_idx), "validation_rows": len(validation_score_idx),
                "strict_prior_resolved": True, "skipped": True,
                "skip_reason": "candidate_only_meta_burnin_or_no_valid_reference_rows",
            })
            continue
        if plan.meta_target_contract.family == FOLD_QUANTILE_RESIDUAL3:
            meta_target, state = fit_fold_quantile_residual3(net[train_idx], mapped[train_idx])
            model = fit_model(
                meta_design.iloc[train_idx].loc[:, list(meta_features)], meta_target,
                meta_weight[train_idx], classifier=True,
                params=_clean_params(plan.meta_params, objective="multiclass", num_class=3),
                objective_mode="stage_i_fold_quantile_residual3_oof",
            )
            probability = _multiclass_probabilities(
                model, meta_design.iloc[validation_score_idx].loc[:, list(meta_features)]
            )
            fold_correction, fold_reconstructed = reconstruct_fold_quantile_residual3(
                probability, mapped[validation_score_idx], state,
            )
            meta_probability[validation_score_idx] = probability
            correction[validation_score_idx] = fold_correction
            reconstructed[validation_score_idx] = fold_reconstructed
            meta_states.append({"fold_id": fold_id, **state.to_dict()})
        else:
            residual = net[train_idx] - mapped[train_idx]
            model = fit_model(
                meta_design.iloc[train_idx].loc[:, list(meta_features)], residual,
                meta_weight[train_idx], classifier=False,
                params=_clean_params(plan.meta_params, objective="huber"),
                objective_mode="stage_i_legacy_Huber_oof",
            )
            fold_correction = np.asarray(
                model.predict(meta_design.iloc[validation_score_idx].loc[:, list(meta_features)]),
                dtype=np.float32,
            )
            correction[validation_score_idx] = fold_correction
            reconstructed[validation_score_idx] = mapped[validation_score_idx] + fold_correction
        meta_fold[validation_score_idx] = fold_id
        provenance.append({
            "side": side, "layer": "meta", "fold_id": fold_id,
            "target_family": plan.meta_target_contract.family,
            "train_rows": len(train_idx), "validation_rows": len(validation_score_idx),
            "candidate_only_training": True, "full_valid_reference_rows_scored": True,
            "validation_start_utc": start.isoformat(),
            "train_max_label_available_utc": available.iloc[train_idx].max().isoformat(),
            "strict_prior_resolved": bool(available.iloc[train_idx].lt(start).all()),
            "skipped": False,
        })
    strict = np.isfinite(reconstructed) & valid
    output = pd.DataFrame({
        "candidate_id": ids, "side_name": side, "decision_ts": decision,
        "label_available_ts": available, "target_valid": valid,
        "candidate_selected": candidate, "mapping_reference_eligible": valid & base_scored,
        "mapping_reference_only": valid & base_scored & ~candidate,
        "exact_gross_bps": gross, "exact_net_bps": net,
        "base_target_family": plan.base_target_contract.family,
        "meta_target_family": plan.meta_target_contract.family,
        "base_fold_id": base_fold, "meta_fold_id": meta_fold,
        "base_raw_score": raw_score,
        "prequential_base_expected_net_bps": mapped,
        "meta_correction_bps": correction,
        "reconstructed_expected_net_bps": reconstructed,
        "base_strict_oof_available": base_scored,
        "strict_oof_available": strict,
    })
    if has_base_probability:
        for index in range(base_probability.shape[1]):
            output[f"base_state_p{index}"] = base_probability[:, index]
    if plan.meta_target_contract.family == FOLD_QUANTILE_RESIDUAL3:
        for index in range(3):
            output[f"meta_residual_state_p{index}"] = meta_probability[:, index]
    manifest = {
        "schema": SCHEMA, "status": "complete", "side": side, "rows": n,
        "base_target_contract": plan.base_target_contract.to_dict(),
        "base_target_contract_sha256": plan.base_target_contract.sha256,
        "meta_target_contract": plan.meta_target_contract.to_dict(),
        "meta_target_contract_sha256": plan.meta_target_contract.sha256,
        "runtime_base_target_contract_sha256": runtime_base.sha256,
        "runtime_meta_target_contract_sha256": runtime_meta.sha256,
        "base_score_semantics": (
            "raw_scalar_S" if plan.base_target_contract.family == SOFT_SCALAR_S
            else "monotone_simplex_expected_ordinal" if plan.base_target_contract.family == CUMULATIVE_ORDINAL5_O
            else "P(clear)-P(adverse)"
        ),
        "economics": "winning_geometry_gross_and_net_only",
        "meta_training": "candidate_only_prior_resolved",
        "mapping_reference": "all_valid_prior_resolved_base_scored_rows; noncandidates_reference_only",
        "meta_fold_states": meta_states,
        "value_map": dict(map_manifest),
    }
    return StageIAdapterStrictOOFResult(
        side=side, predictions=output,
        fold_provenance=pd.DataFrame(provenance), manifest=manifest,
    )


__all__ = [
    "SCHEMA", "StageIAdapterStrictOOFPlan", "StageIAdapterStrictOOFResult",
    "generate_stage_i_adapter_strict_oof",
]
