"""Target-v2 automatic/20/30/40/60/full Stage-I count ladders."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .stage_i_nested_feature_challenger import NESTED_SET_NAMES, NestedFeatureChallengePlan
from .stage_i_base_target_ablation import training_weights
from .stage_i_strict_oof import _strict_train_mask, _validation_blocks
from .stage_i_target_adapter import (
    FOLD_QUANTILE_RESIDUAL3,
    StageITargetContract,
    canonical_sha256,
    file_sha256,
    fit_fold_quantile_residual3,
    recover_base_score,
    reconstruct_fold_quantile_residual3,
)


SCHEMA = "stage_i_target_adapter_feature_ladder_v2"
BasePredictor = Callable[[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, str], np.ndarray]
MetaPredictor = Callable[[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, str], np.ndarray]


@dataclass(frozen=True)
class AdapterCountLadderInput:
    side: str
    frame: pd.DataFrame
    target: Sequence[float]
    exact_net_bps: Sequence[float]
    decision_timestamps: Sequence[Any]
    label_available_timestamps: Sequence[Any]
    sample_weight: Sequence[float]
    target_contract: StageITargetContract
    fold_local_weight_frame: pd.DataFrame | None = None
    candidate_selected: Sequence[bool] | None = None
    mapped_base_expected_net_bps: Sequence[float] | None = None


def _validate(data: AdapterCountLadderInput, plan: NestedFeatureChallengePlan) -> tuple[pd.Series, pd.Series, np.ndarray, np.ndarray, np.ndarray]:
    n = len(data.frame)
    if plan.side != data.side or tuple(item.name for item in plan.feature_sets) != NESTED_SET_NAMES:
        raise ValueError("adapter count ladder requires the complete matched side plan")
    if data.target_contract.rows != n or data.target_contract.layer not in {"base", "meta"}:
        raise ValueError("adapter count ladder target contract drift")
    decision = pd.to_datetime(pd.Series(data.decision_timestamps), utc=True, errors="coerce")
    available = pd.to_datetime(pd.Series(data.label_available_timestamps), utc=True, errors="coerce")
    target = np.asarray(data.target).reshape(-1)
    net = np.asarray(data.exact_net_bps, dtype=np.float32).reshape(-1)
    weight = np.asarray(data.sample_weight, dtype=np.float32).reshape(-1)
    if any(len(value) != n for value in (decision, available, target, net, weight)):
        raise ValueError("adapter count ladder arrays are not aligned")
    if decision.isna().any() or available.isna().any() or not (available - decision).eq(pd.Timedelta(hours=12)).all():
        raise ValueError("adapter count ladder timing drift")
    if not np.isfinite(target).all() or not np.isfinite(net).all() or not np.isfinite(weight).all():
        raise ValueError("adapter count ladder target/economics/weight invalid")
    return decision, available, target, net, weight


def _publish(
    root: Path, *, request: Mapping[str, Any], arms: list[dict[str, Any]],
) -> dict[str, Any]:
    request_sha = canonical_sha256(request)
    root.mkdir(parents=True, exist_ok=False)
    inventory: dict[str, Any] = {}
    for arm in arms:
        name = str(arm.pop("feature_set"))
        arm_root = root / "arms" / name
        arm_root.mkdir(parents=True)
        prediction = arm.pop("prediction")
        provenance = arm.pop("provenance")
        prediction.to_parquet(arm_root / "oof_predictions.parquet", index=False, compression="zstd")
        provenance.to_parquet(arm_root / "fold_provenance.parquet", index=False, compression="zstd")
        hpo_request = {
            "schema": "stage_i_target_adapter_count_hpo_refit_request_v2",
            "side": request["side"], "layer": request["layer"], "feature_set": name,
            "features": arm["features"], "target_contract_sha256": request["target_contract_sha256"],
            "required_action": "count_specific_target_HPO_then_strict_OOF_refit",
            "freeze_eligible_now": False,
        }
        hpo_request["request_sha256"] = canonical_sha256(hpo_request)
        (arm_root / "count_specific_hpo_refit_request.json").write_text(json.dumps(hpo_request, indent=2, sort_keys=True) + "\n")
        manifest = {
            "schema": SCHEMA, "status": "complete", "request_sha256": request_sha,
            "feature_set": name, **arm,
            "oof_predictions_sha256": file_sha256(arm_root / "oof_predictions.parquet"),
            "fold_provenance_sha256": file_sha256(arm_root / "fold_provenance.parquet"),
            "hpo_request_sha256": file_sha256(arm_root / "count_specific_hpo_refit_request.json"),
            "freeze_eligible_now": False,
        }
        (arm_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        inventory[name] = {
            "manifest_sha256": file_sha256(arm_root / "manifest.json"),
            "oof_predictions_sha256": manifest["oof_predictions_sha256"],
            "fold_provenance_sha256": manifest["fold_provenance_sha256"],
            "hpo_request_sha256": manifest["hpo_request_sha256"],
        }
    manifest = {
        **request, "request_sha256": request_sha, "status": "complete",
        "schema": SCHEMA, "planned_arm_order": list(NESTED_SET_NAMES),
        "arm_inventory": inventory, "freeze_eligible_now": False,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def run_adapter_base_count_ladder(
    data: AdapterCountLadderInput,
    plan: NestedFeatureChallengePlan,
    *, predictor: BasePredictor, output_dir: str | Path,
    n_validation_folds: int = 4, min_train_rows: int = 500,
) -> dict[str, Any]:
    if data.target_contract.layer != "base":
        raise ValueError("base count ladder needs an explicit base adapter")
    decision, available, target, net, weight = _validate(data, plan)
    weight_contract = data.target_contract.metadata.get("training_weight_contract") or {}
    weight_mode = str(weight_contract.get("mode", ""))
    regime_column = str(data.target_contract.metadata.get("regime_column", ""))
    weight_frame = data.fold_local_weight_frame
    if weight_mode:
        if weight_mode not in {"uniform", "contract_certainty", "hybrid"}:
            raise ValueError("base count ladder target weight mode is unknown")
        if weight_frame is None or len(weight_frame) != len(data.frame):
            raise ValueError(
                "target-v2 base count ladder requires the aligned fold-local weight frame"
            )
        weight_frame = weight_frame.reset_index(drop=True)
        if "decision_ts" not in weight_frame:
            raise ValueError("fold-local count-ladder weights lack decision_ts")
        declared_decision = pd.to_datetime(
            weight_frame.decision_ts, utc=True, errors="coerce"
        ).reset_index(drop=True)
        if declared_decision.isna().any() or not declared_decision.equals(decision.reset_index(drop=True)):
            raise ValueError("fold-local count-ladder weight decisions drift")
    blocks = _validation_blocks(decision, available, n_folds=n_validation_folds, min_train_rows=min_train_rows)
    arms: list[dict[str, Any]] = []
    for feature_set in plan.feature_sets:
        raw = np.full(len(data.frame), np.nan, dtype=np.float32)
        state_probability: np.ndarray | None = None
        lineage = []
        for fold_id, validation_idx in enumerate(blocks):
            start = decision.iloc[validation_idx].min()
            train_idx = np.flatnonzero(_strict_train_mask(available, start))
            fold_weight = (
                weight[train_idx]
                if not weight_mode
                else training_weights(
                    weight_frame.iloc[train_idx], target=target[train_idx],
                    mode=weight_mode, regime_column=regime_column,
                )
            )
            prediction = predictor(
                data.frame.iloc[train_idx].loc[:, list(feature_set.features)], target[train_idx],
                fold_weight, data.frame.iloc[validation_idx].loc[:, list(feature_set.features)],
                data.target_contract.family,
            )
            score, simplex = recover_base_score(data.target_contract.family, prediction)
            raw[validation_idx] = score
            if simplex is not None:
                if state_probability is None:
                    state_probability = np.full((len(data.frame), simplex.shape[1]), np.nan, np.float32)
                state_probability[validation_idx] = simplex
            lineage.append({
                "fold_id": fold_id, "train_rows": len(train_idx), "validation_rows": len(validation_idx),
                "validation_start_utc": start.isoformat(), "strict_prior_resolved": True,
                "training_weight_fit_scope": (
                    "strict_fold_train_only" if weight_mode else "provided_aligned_vector"
                ),
                "training_weight_mode": weight_mode or "legacy_provided",
            })
        output = pd.DataFrame({
            "row_id": np.arange(len(data.frame)), "raw_score": raw,
            "exact_net_bps": net, "strict_oof_available": np.isfinite(raw),
        })
        if state_probability is not None:
            for index in range(state_probability.shape[1]):
                output[f"base_state_p{index}"] = state_probability[:, index]
        arms.append({
            "feature_set": feature_set.name, "features": list(feature_set.features),
            "feature_set_sha256": feature_set.source_hash,
            "target_family": data.target_contract.family,
            "prediction": output, "provenance": pd.DataFrame(lineage),
        })
    request = {
        "side": data.side, "layer": "base", "target_contract": data.target_contract.to_dict(),
        "target_contract_sha256": data.target_contract.sha256,
        "plan_sha256": plan.plan_hash, "winner_economics": "exact_net_bps_from_target_contract_geometry",
        "count_specific_hpo_required": True,
    }
    return _publish(Path(output_dir), request=request, arms=arms)


def run_adapter_meta_count_ladder(
    data: AdapterCountLadderInput,
    plan: NestedFeatureChallengePlan,
    *, predictor: MetaPredictor, output_dir: str | Path,
    n_validation_folds: int = 4, min_train_rows: int = 500,
) -> dict[str, Any]:
    if data.target_contract.family != FOLD_QUANTILE_RESIDUAL3:
        raise ValueError("meta count ladder is frozen to the explicit residual3 adapter")
    decision, available, _target_basis, net, weight = _validate(data, plan)
    candidate = np.asarray(data.candidate_selected, dtype=bool).reshape(-1)
    mapped = np.asarray(data.mapped_base_expected_net_bps, dtype=np.float32).reshape(-1)
    if len(candidate) != len(data.frame) or len(mapped) != len(data.frame) or not np.isfinite(mapped).all():
        raise ValueError("meta count ladder candidate/map contract drift")
    blocks = _validation_blocks(decision[candidate].reset_index(drop=True), available[candidate].reset_index(drop=True), n_folds=n_validation_folds, min_train_rows=min_train_rows)
    candidate_positions = np.flatnonzero(candidate)
    projected_blocks = [candidate_positions[block] for block in blocks]
    arms: list[dict[str, Any]] = []
    for feature_set in plan.feature_sets:
        correction = np.full(len(data.frame), np.nan, np.float32)
        probability = np.full((len(data.frame), 3), np.nan, np.float32)
        lineage = []
        for fold_id, validation_candidate_idx in enumerate(projected_blocks):
            start = decision.iloc[validation_candidate_idx].min()
            train_idx = np.flatnonzero(available.lt(start).to_numpy() & candidate)
            if len(train_idx) < min_train_rows:
                continue
            labels, state = fit_fold_quantile_residual3(net[train_idx], mapped[train_idx])
            valid_idx = np.flatnonzero(decision.ge(start).to_numpy())
            predicted = predictor(
                data.frame.iloc[train_idx].loc[:, list(feature_set.features)], labels,
                weight[train_idx], data.frame.iloc[valid_idx].loc[:, list(feature_set.features)],
                FOLD_QUANTILE_RESIDUAL3,
            )
            predicted = np.asarray(predicted, dtype=np.float32)
            fold_correction, _ = reconstruct_fold_quantile_residual3(predicted, mapped[valid_idx], state)
            # Later folds overwrite later rows with the fit whose declared
            # validation boundary owns them; no training outcome enters score.
            probability[valid_idx] = predicted
            correction[valid_idx] = fold_correction
            lineage.append({
                "fold_id": fold_id, "train_rows": len(train_idx), "scored_full_valid_rows": len(valid_idx),
                "candidate_only_training": True, "full_reference_scoring": True,
                "validation_start_utc": start.isoformat(), "strict_prior_resolved": True,
                "target_state": state.to_dict(),
            })
        output = pd.DataFrame({
            "row_id": np.arange(len(data.frame)), "candidate_selected": candidate,
            "mapped_base_expected_net_bps": mapped, "meta_correction_bps": correction,
            "reconstructed_expected_net_bps": mapped + correction, "exact_net_bps": net,
        })
        for index in range(3):
            output[f"meta_residual_state_p{index}"] = probability[:, index]
        arms.append({
            "feature_set": feature_set.name, "features": list(feature_set.features),
            "feature_set_sha256": feature_set.source_hash,
            "target_family": data.target_contract.family,
            "candidate_only_training": True, "full_valid_reference_scoring": True,
            "prediction": output, "provenance": pd.DataFrame(lineage),
        })
    request = {
        "side": data.side, "layer": "meta", "target_contract": data.target_contract.to_dict(),
        "target_contract_sha256": data.target_contract.sha256,
        "plan_sha256": plan.plan_hash, "candidate_only_training": True,
        "full_valid_rows_reference_only_for_mapping": True,
        "count_specific_hpo_required": True,
    }
    return _publish(Path(output_dir), request=request, arms=arms)


__all__ = [
    "SCHEMA", "AdapterCountLadderInput", "run_adapter_base_count_ladder",
    "run_adapter_meta_count_ladder",
]
