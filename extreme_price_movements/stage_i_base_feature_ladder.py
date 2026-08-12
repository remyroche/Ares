"""Executable, hash-bound Stage-I base feature-count ladder.

This is deliberately separate from the nested base+meta diagnostic.  It holds
the side-local base population, chronological strict-OOF folds, R3 target, and
source base HPO parameters fixed while varying only one of the pre-materialised
base feature sets.  In particular, no candidate handoff or meta burn-in is
allowed to shorten the base evaluation population.

Each count is a fixed-source-HPO diagnostic.  A later count-specific base HPO
and refit artifact is required before any feature count can be frozen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .stage_i_causal_admission import (
    Causal21dAdmissionSpec,
    apply_causal_21d_side_admission,
    pooled_global_admission_comparison,
)
from .stage_i_nested_feature_challenger import (
    IDENTITY_COLUMNS,
    NESTED_SET_NAMES,
    NestedFeatureChallengePlan,
    NestedFeatureChallengerError,
    NestedFeatureSet,
)
from .stage_i_ranking import RANKING_POLICY, stable_stage_i_rank_frame
from .stage_i_timestamp_contract import resolve_stage_i_timestamp_contract
from .stage_i_r3_contract import frame_content_sha256


SCHEMA = "stage_i_base_feature_ladder_execution_v1"
TOP_FRACTIONS: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20)
BasePredictor = Callable[[pd.DataFrame, np.ndarray, pd.DataFrame, NestedFeatureSet], np.ndarray]


class StageIBaseFeatureLadderError(ValueError):
    """Raised when an independently-comparable base count arm is impossible."""


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity_sha(frame: pd.DataFrame) -> str:
    identity = frame.loc[:, list(IDENTITY_COLUMNS)]
    if identity.isna().any().any() or identity.duplicated().any():
        raise StageIBaseFeatureLadderError("base feature ladder needs unique, non-null identities")
    return sha256(
        pd.util.hash_pandas_object(identity, index=False).to_numpy(dtype=np.uint64).tobytes()
    ).hexdigest()


def _strict_folds(
    decision: pd.Series,
    available: pd.Series,
    *, count: int,
    min_train_rows: int,
) -> list[np.ndarray]:
    """Whole-timestamp strict chronological folds, shared by every count."""
    order = np.argsort(decision.to_numpy(dtype="datetime64[ns]"), kind="stable")
    ordered = decision.to_numpy(dtype="datetime64[ns]")[order]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    groups = [
        order[start:stop]
        for start, stop in zip(starts, np.r_[starts[1:], len(order)], strict=True)
    ]
    first = next(
        (
            index
            for index, group in enumerate(groups)
            if int(available.lt(decision.iloc[group].min()).sum()) >= int(min_train_rows)
        ),
        None,
    )
    if first is None:
        raise StageIBaseFeatureLadderError("no strict OOF support after base burn-in")
    remaining = len(groups) - int(first)
    blocks = np.array_split(np.arange(remaining), min(int(count), remaining))
    return [
        np.concatenate([groups[int(first) + int(item)] for item in block]).astype(np.int32)
        for block in blocks
        if len(block)
    ]


def _simplex(value: np.ndarray, rows: int, *, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if (
        result.shape != (rows, 3)
        or not np.isfinite(result).all()
        or (result < 0.0).any()
        or not np.allclose(result.sum(axis=1), 1.0, atol=1e-6)
    ):
        raise StageIBaseFeatureLadderError(f"{label} must emit a finite R3 probability simplex")
    return result


@dataclass(frozen=True)
class BaseFeatureLadderInput:
    """Complete valid side-local base population; no meta fields are required."""

    side: str
    frame: pd.DataFrame
    base_feature_universe: tuple[str, ...]
    target_column: str = "r3_class"
    net_column: str = "exact_net_bps"
    decision_column: str = "decision_ts"
    label_available_column: str = "label_available_ts"
    # The strict OOF frame contains only supervised-valid rows.  Preserve the
    # complete side candidate denominator separately so requested global K is
    # never silently redefined by invalid labels or base burn-in.
    full_candidate_rows: int | None = None
    invalid_or_incomplete_rows: int = 0
    full_candidate_identity_sha256: str | None = None
    full_candidate_validity_sha256: str | None = None
    source_integrity: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BaseFeatureLadderConfig:
    n_validation_folds: int = 4
    min_train_rows: int = 500

    def __post_init__(self) -> None:
        if self.n_validation_folds < 1 or self.min_train_rows < 3:
            raise StageIBaseFeatureLadderError("base fold count/minimum training rows are invalid")


def _validate_input(data: BaseFeatureLadderInput, plan: NestedFeatureChallengePlan) -> pd.DataFrame:
    side = str(data.side).lower()
    if side not in {"long", "short"} or plan.side != side:
        raise StageIBaseFeatureLadderError("base input and feature plan must share canonical long/short side")
    if tuple(item.name for item in plan.feature_sets) != NESTED_SET_NAMES:
        raise StageIBaseFeatureLadderError("base plan must expose the complete automatic/20/30/40/60/full ladder")
    raw = data.frame.copy().reset_index(drop=True)
    required = {
        *IDENTITY_COLUMNS,
        "side_name",
        data.target_column,
        data.net_column,
        data.decision_column,
        data.label_available_column,
        *data.base_feature_universe,
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise StageIBaseFeatureLadderError(f"base ladder input lacks {missing[:12]}")
    if not raw.side_name.astype(str).str.lower().eq(side).all():
        raise StageIBaseFeatureLadderError("base feature ladder input is not side-local")
    timing = resolve_stage_i_timestamp_contract(raw)
    raw[data.decision_column] = timing.decision.to_numpy()
    raw[data.label_available_column] = timing.label_available.to_numpy()
    target = pd.to_numeric(raw[data.target_column], errors="coerce").to_numpy()
    net = pd.to_numeric(raw[data.net_column], errors="coerce").to_numpy(float)
    if not np.isin(target, (0, 1, 2)).all() or not np.isfinite(net).all():
        raise StageIBaseFeatureLadderError("base ladder needs finite exact net and R3 classes 0/1/2")
    for item in plan.feature_sets:
        if not set(item.features).issubset(data.base_feature_universe):
            raise StageIBaseFeatureLadderError(f"{item.name} escapes the declared base feature universe")
    _identity_sha(raw)
    return raw


def _input_value_sha(raw: pd.DataFrame, data: BaseFeatureLadderInput) -> str:
    """Bind every fitted/scored base input value, not just its identities."""

    columns = (
        *IDENTITY_COLUMNS,
        "side_name",
        data.target_column,
        data.net_column,
        data.decision_column,
        data.label_available_column,
        *data.base_feature_universe,
    )
    return frame_content_sha256(raw, columns)


def _denominator_contract(data: BaseFeatureLadderInput, raw: pd.DataFrame) -> dict[str, Any]:
    full = int(data.full_candidate_rows) if data.full_candidate_rows is not None else int(len(raw))
    invalid = int(data.invalid_or_incomplete_rows)
    if full < len(raw) or invalid < 0 or invalid > full:
        raise StageIBaseFeatureLadderError("invalid full-candidate denominator contract")
    if data.full_candidate_identity_sha256 is not None:
        candidate_identity_sha = str(data.full_candidate_identity_sha256)
    else:
        candidate_identity_sha = _identity_sha(raw)
    if data.full_candidate_validity_sha256 is not None:
        validity_sha = str(data.full_candidate_validity_sha256)
    else:
        validity_sha = _canonical_sha({"valid_rows": int(len(raw)), "invalid_rows": invalid})
    return {
        "schema": "stage_i_base_ladder_population_denominator_v1",
        "full_candidate_rows": full,
        "invalid_or_incomplete_label_rows": invalid,
        "valid_complete_candidate_rows": int(full - invalid),
        "full_candidate_identity_sha256": candidate_identity_sha,
        "full_candidate_validity_sha256": validity_sha,
    }


def _multiclass_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y = np.asarray(target, dtype=int)
    p = np.asarray(probability, dtype=float)
    one_hot = np.eye(3)[y]
    return {
        "base_multiclass_log_loss": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)).mean()),
        "base_multiclass_brier": float(np.square(p - one_hot).sum(axis=1).mean() / 3.0),
    }


def _side_raw_metrics(frame: pd.DataFrame) -> list[dict[str, Any]]:
    ordered = stable_stage_i_rank_frame(
        frame, score_column="r3_opportunity_score", candidate_id_column="candidate_key",
        decision_column="decision_ts",
    )
    rows: list[dict[str, Any]] = []
    for fraction in TOP_FRACTIONS:
        count = max(1, int(math.ceil(float(fraction) * len(ordered))))
        selected = ordered.head(count)
        month = selected.decision_ts.dt.strftime("%Y-%m")
        rows.append({
            "scope": "side_local_raw_score_before_common_bps_mapping",
            "side": str(frame.side_name.iloc[0]), "top_fraction": float(fraction),
            "scored_rows": int(len(frame)), "selected_rows": int(len(selected)),
            "net_bps_per_trade": float(selected.exact_net_bps.mean()),
            "worst_month_net_bps_per_trade": float(selected.groupby(month, observed=True).exact_net_bps.mean().min()),
            "ranking_tie_policy": RANKING_POLICY,
            "freeze_eligible_now": False,
        })
    return rows


def _hpo_refit_request(
    *, plan: NestedFeatureChallengePlan, feature_set: NestedFeatureSet, side: str,
    source_base_params: Mapping[str, Any], source_base_manifest_sha256: str,
) -> dict[str, Any]:
    payload = {
        "schema": "stage_i_base_count_specific_hpo_refit_request_v1",
        "side": side,
        "feature_set": feature_set.name,
        "feature_set_sha256": feature_set.source_hash,
        "features": list(feature_set.features),
        "plan_sha256": plan.plan_hash,
        "source_base_manifest_sha256": source_base_manifest_sha256,
        "source_base_hpo_params_sha256": _canonical_sha(dict(source_base_params)),
        "target": "R3_cost_aware_three_state_multiclass",
        "target_contract": "frozen_same_side_R3",
        "shared_oof_contract": "same_side_full_valid_base_population_and_chronological_fold_vector",
        "required_artifact_schema": "stage_i_base_count_specific_hpo_refit_v1",
        "fixed_source_hpo_diagnostic": True,
        "freeze_eligible_now": False,
        "freeze_blocker": "count_specific_base_HPO_and_refit_required",
        "full_input_promotion_eligible": bool(feature_set.promotion_eligible),
    }
    payload["request_sha256"] = _canonical_sha(payload)
    return payload


def _verified_resume(root: Path, request_sha: str) -> dict[str, Any] | None:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "complete" or manifest.get("request_sha256") != request_sha:
        raise StageIBaseFeatureLadderError("completed base feature ladder request/hash drift")
    for relative, key in (
        ("evaluation_population.parquet", "evaluation_population_sha256"),
        ("fold_vector.parquet", "fold_vector_sha256"),
        ("hpo_refit_requests.json", "hpo_refit_requests_sha256"),
        ("side_raw_metrics.parquet", "side_raw_metrics_sha256"),
        ("denominator_audit.parquet", "denominator_audit_sha256"),
    ):
        path = root / relative
        if not path.is_file() or manifest.get(key) != file_sha256(path):
            raise StageIBaseFeatureLadderError("completed base feature ladder checksum drift")
    inventory = manifest.get("planned_arm_inventory")
    if (
        not isinstance(inventory, dict)
        or set(inventory) != set(NESTED_SET_NAMES)
        or manifest.get("planned_arm_order") != list(NESTED_SET_NAMES)
    ):
        raise StageIBaseFeatureLadderError("completed base feature ladder arm inventory drift")
    arms_root = root / "arms"
    if not arms_root.is_dir() or {path.name for path in arms_root.iterdir()} != set(NESTED_SET_NAMES):
        raise StageIBaseFeatureLadderError("completed base feature ladder arms are stale, extra, or partial")
    for name in NESTED_SET_NAMES:
        expected = inventory[name]
        if not isinstance(expected, Mapping):
            raise StageIBaseFeatureLadderError("completed base feature ladder arm manifest inventory is malformed")
        arm_root = arms_root / name
        files = {
            "oof_predictions_sha256": arm_root / "base_oof_predictions.parquet",
            "fold_provenance_sha256": arm_root / "fold_provenance.parquet",
            "count_specific_base_hpo_refit_request_sha256": arm_root / "count_specific_base_hpo_refit_request.json",
            "arm_manifest_sha256": arm_root / "manifest.json",
        }
        if {path.name for path in arm_root.iterdir()} != {path.name for path in files.values()}:
            raise StageIBaseFeatureLadderError("completed base feature ladder arm has stale, extra, or partial files")
        for key, path in files.items():
            if not path.is_file() or expected.get(key) != file_sha256(path):
                raise StageIBaseFeatureLadderError("completed base feature ladder arm checksum drift")
        arm_manifest = json.loads((arm_root / "manifest.json").read_text(encoding="utf-8"))
        if (
            arm_manifest.get("schema") != SCHEMA
            or arm_manifest.get("status") != "complete"
            or arm_manifest.get("feature_set") != name
            or arm_manifest.get("request_sha256") != request_sha
            or arm_manifest.get("feature_set_sha256") != expected.get("feature_set_sha256")
        ):
            raise StageIBaseFeatureLadderError("completed base feature ladder arm manifest drift")
    return manifest


def run_side_base_feature_ladder(
    data: BaseFeatureLadderInput,
    plan: NestedFeatureChallengePlan,
    *,
    base_predictor: BasePredictor,
    source_base_params: Mapping[str, Any],
    source_base_manifest_sha256: str,
    output_dir: str | Path,
    config: BaseFeatureLadderConfig = BaseFeatureLadderConfig(),
    resume: bool = False,
) -> dict[str, Any]:
    """Run all six base sets without candidate/meta conditioning."""
    if not callable(base_predictor) or not isinstance(source_base_params, Mapping):
        raise StageIBaseFeatureLadderError("base predictor and frozen source HPO parameters are required")
    raw = _validate_input(data, plan)
    side, root = str(data.side).lower(), Path(output_dir)
    input_value_sha = _input_value_sha(raw, data)
    denominator = _denominator_contract(data, raw)
    decision = pd.to_datetime(raw[data.decision_column], utc=True, errors="raise")
    available = pd.to_datetime(raw[data.label_available_column], utc=True, errors="raise")
    blocks = _strict_folds(decision, available, count=config.n_validation_folds, min_train_rows=config.min_train_rows)
    fold = np.full(len(raw), -1, dtype=np.int16)
    for index, positions in enumerate(blocks):
        fold[positions] = int(index)
    fold_frame = raw.loc[:, list(IDENTITY_COLUMNS)].copy()
    fold_frame["fold_id"] = fold
    identity_sha = _identity_sha(raw)
    fold_sha = _canonical_sha({"identity_sha256": identity_sha, "fold_id": fold.tolist()})
    request = {
        "schema": SCHEMA, "side": side, "plan_sha256": plan.plan_hash,
        "source_base_manifest_sha256": source_base_manifest_sha256,
        "source_base_hpo_params_sha256": _canonical_sha(dict(source_base_params)),
        "strict_fold_vector_sha256": fold_sha,
        "strict_oof_identity_sha256": identity_sha,
        "base_input_value_sha256": input_value_sha,
        "source_integrity_sha256": _canonical_sha(dict(data.source_integrity)),
        "source_integrity": _jsonable(dict(data.source_integrity)),
        "full_candidate_denominator": denominator,
        "n_validation_folds": int(config.n_validation_folds),
        "min_train_rows": int(config.min_train_rows),
        "feature_sets": list(NESTED_SET_NAMES),
        "base_only": True,
        "meta_dependency": "forbidden",
        "hpo_disposition": "fixed_source_HPO_diagnostic_only; count_specific_base_HPO_and_refit_required",
    }
    request_sha = _canonical_sha(request)
    previous = _verified_resume(root, request_sha) if resume else None
    if previous is not None:
        return {**previous, "restart_status": "reused_verified_complete"}
    if root.exists() and not resume:
        raise FileExistsError(f"base feature ladder output exists without --resume: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        (temporary / "arms").mkdir()
        evaluation = raw.loc[fold >= 0, list(IDENTITY_COLUMNS) + [
            "side_name", data.target_column, data.net_column, data.decision_column,
            data.label_available_column,
        ]].copy()
        evaluation = evaluation.rename(columns={
            data.target_column: "r3_class", data.net_column: "exact_net_bps",
            data.decision_column: "decision_ts", data.label_available_column: "label_available_ts",
        })
        evaluation["candidate_key"] = (
            evaluation.side_name.astype(str) + "::" + evaluation.candidate_id.astype(str)
            + "::" + evaluation["__ts__"].astype(str)
        )
        evaluation["fold_id"] = fold[fold >= 0]
        denominator_row = {
            **denominator,
            "side": side,
            "base_burn_in_unscored_rows": int(len(raw) - len(evaluation)),
            "valid_strict_oof_scored_rows": int(len(evaluation)),
            "strict_oof_identity_sha256": _identity_sha(evaluation),
            "base_input_value_sha256": input_value_sha,
        }
        hpo_requests: list[dict[str, Any]] = []
        metric_rows: list[dict[str, Any]] = []
        for feature_set in plan.feature_sets:
            probability = np.full((len(raw), 3), np.nan, dtype=float)
            lineage: list[dict[str, Any]] = []
            for fold_id, valid_idx in enumerate(blocks):
                start = decision.iloc[valid_idx].min()
                train_idx = np.flatnonzero(available.lt(start).to_numpy())
                if len(train_idx) < int(config.min_train_rows):
                    raise StageIBaseFeatureLadderError("base fold escaped the strict training support gate")
                y_train = pd.to_numeric(raw.iloc[train_idx][data.target_column], errors="raise").to_numpy(int)
                if set(np.unique(y_train)) != {0, 1, 2}:
                    raise StageIBaseFeatureLadderError(f"{feature_set.name}/fold{fold_id}: base training lacks R3 class support")
                prediction = _simplex(
                    base_predictor(
                        raw.iloc[train_idx].loc[:, list(feature_set.features)], y_train,
                        raw.iloc[valid_idx].loc[:, list(feature_set.features)], feature_set,
                    ),
                    len(valid_idx), label=f"{feature_set.name}/fold{fold_id}",
                )
                probability[valid_idx] = prediction
                lineage.append({
                    "side": side, "feature_set": feature_set.name, "fold_id": int(fold_id),
                    "train_rows": int(len(train_idx)), "validation_rows": int(len(valid_idx)),
                    "validation_start_utc": start.isoformat(),
                    "validation_end_utc": decision.iloc[valid_idx].max().isoformat(),
                    "train_max_label_available_utc": available.iloc[train_idx].max().isoformat(),
                    "strict_prior_resolved": True,
                    "base_feature_count": int(len(feature_set.features)),
                    "meta_training_or_candidate_gate_used": False,
                })
            if not np.isfinite(probability[fold >= 0]).all():
                raise StageIBaseFeatureLadderError("incomplete base strict OOF probability output")
            output = evaluation.copy()
            output["r3_p_adverse"] = probability[fold >= 0, 0]
            output["r3_p_weak"] = probability[fold >= 0, 1]
            output["r3_p_clear"] = probability[fold >= 0, 2]
            output["r3_opportunity_score"] = output.r3_p_clear - output.r3_p_adverse
            output["mapping_reference_eligible"] = True
            output["base_feature_set"] = feature_set.name
            output["base_feature_set_sha256"] = feature_set.source_hash
            p = output.loc[:, ["r3_p_adverse", "r3_p_weak", "r3_p_clear"]].to_numpy(float)
            base_metrics = _multiclass_metrics(output.r3_class.to_numpy(int), p)
            hpo = _hpo_refit_request(
                plan=plan, feature_set=feature_set, side=side,
                source_base_params=source_base_params,
                source_base_manifest_sha256=source_base_manifest_sha256,
            )
            hpo_requests.append(hpo)
            arm = temporary / "arms" / feature_set.name
            arm.mkdir()
            output.to_parquet(arm / "base_oof_predictions.parquet", index=False, compression="zstd")
            pd.DataFrame(lineage).to_parquet(arm / "fold_provenance.parquet", index=False, compression="zstd")
            request_path = arm / "count_specific_base_hpo_refit_request.json"
            request_path.write_text(json.dumps(hpo, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            arm_manifest = {
                "schema": SCHEMA, "status": "complete", "request_sha256": request_sha,
                "side": side, "feature_set": feature_set.name,
                "feature_set_sha256": feature_set.source_hash,
                "strict_oof_identity_sha256": identity_sha,
                "strict_fold_vector_sha256": fold_sha,
                "base_feature_count": int(len(feature_set.features)),
                "base_only": True, "meta_dependency": "forbidden",
                "fixed_source_hpo_diagnostic": True, "freeze_eligible_now": False,
                "full_input_promotion_eligible": bool(feature_set.promotion_eligible),
                "count_specific_base_hpo_refit_request_sha256": file_sha256(request_path),
                "oof_predictions_sha256": file_sha256(arm / "base_oof_predictions.parquet"),
                "fold_provenance_sha256": file_sha256(arm / "fold_provenance.parquet"),
            }
            (arm / "manifest.json").write_text(json.dumps(arm_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            metric_rows.extend({
                **row, "feature_set": feature_set.name, **base_metrics,
                "strict_oof_identity_sha256": identity_sha,
                "strict_fold_vector_sha256": fold_sha,
            } for row in _side_raw_metrics(output))
        evaluation.to_parquet(temporary / "evaluation_population.parquet", index=False, compression="zstd")
        fold_frame.to_parquet(temporary / "fold_vector.parquet", index=False, compression="zstd")
        pd.DataFrame(metric_rows).to_parquet(temporary / "side_raw_metrics.parquet", index=False, compression="zstd")
        (temporary / "hpo_refit_requests.json").write_text(json.dumps(hpo_requests, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        pd.DataFrame([denominator_row]).to_parquet(temporary / "denominator_audit.parquet", index=False, compression="zstd")
        planned_arm_inventory = {}
        for feature_set in plan.feature_sets:
            arm = temporary / "arms" / feature_set.name
            planned_arm_inventory[feature_set.name] = {
                "feature_set_sha256": feature_set.source_hash,
                "oof_predictions_sha256": file_sha256(arm / "base_oof_predictions.parquet"),
                "fold_provenance_sha256": file_sha256(arm / "fold_provenance.parquet"),
                "count_specific_base_hpo_refit_request_sha256": file_sha256(arm / "count_specific_base_hpo_refit_request.json"),
                "arm_manifest_sha256": file_sha256(arm / "manifest.json"),
            }
        manifest = {
            **request, "status": "complete", "request_sha256": request_sha,
            "evaluation_rows": int(len(evaluation)), "freeze_eligible_now": False,
            "evaluation_population_sha256": file_sha256(temporary / "evaluation_population.parquet"),
            "fold_vector_sha256": file_sha256(temporary / "fold_vector.parquet"),
            "side_raw_metrics_sha256": file_sha256(temporary / "side_raw_metrics.parquet"),
            "hpo_refit_requests_sha256": file_sha256(temporary / "hpo_refit_requests.json"),
            "denominator_audit_sha256": file_sha256(temporary / "denominator_audit.parquet"),
            "planned_arm_order": list(NESTED_SET_NAMES),
            "planned_arm_inventory": planned_arm_inventory,
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, root)
    finally:
        if temporary.exists():
            import shutil
            shutil.rmtree(temporary)
    return manifest


def _load_arm(root: Path, *, name: str, side_manifest: Mapping[str, Any]) -> pd.DataFrame:
    manifest_path, prediction_path = root / "manifest.json", root / "base_oof_predictions.parquet"
    if not manifest_path.is_file() or not prediction_path.is_file():
        raise StageIBaseFeatureLadderError(f"base count arm is incomplete: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inventory = side_manifest.get("planned_arm_inventory", {})
    expected = inventory.get(name) if isinstance(inventory, Mapping) else None
    required_files = {
        "oof_predictions_sha256": prediction_path,
        "fold_provenance_sha256": root / "fold_provenance.parquet",
        "count_specific_base_hpo_refit_request_sha256": root / "count_specific_base_hpo_refit_request.json",
        "arm_manifest_sha256": manifest_path,
    }
    if {path.name for path in root.iterdir()} != {path.name for path in required_files.values()}:
        raise StageIBaseFeatureLadderError("base count arm is stale, extra, or partial")
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("feature_set") != name
        or manifest.get("oof_predictions_sha256") != file_sha256(prediction_path)
        or not isinstance(expected, Mapping)
        or any(
            not path.is_file() or expected.get(key) != file_sha256(path)
            for key, path in required_files.items()
        )
    ):
        raise StageIBaseFeatureLadderError("base count arm prediction checksum/schema drift")
    return pd.read_parquet(prediction_path)


def _load_denominator_audit(root: Path, manifest: Mapping[str, Any]) -> pd.DataFrame:
    path = root / "denominator_audit.parquet"
    if (
        not path.is_file()
        or manifest.get("denominator_audit_sha256") != file_sha256(path)
    ):
        raise StageIBaseFeatureLadderError("base ladder denominator audit checksum drift")
    audit = pd.read_parquet(path)
    required = {
        "side", "full_candidate_rows", "invalid_or_incomplete_label_rows",
        "valid_complete_candidate_rows", "base_burn_in_unscored_rows",
        "valid_strict_oof_scored_rows",
    }
    missing = sorted(required.difference(audit.columns))
    if len(audit) != 1 or missing:
        raise StageIBaseFeatureLadderError(f"base ladder denominator audit is malformed: {missing}")
    return audit


def run_pooled_base_feature_ladder(
    *, long_dir: str | Path, short_dir: str | Path, output_dir: str | Path,
    admission_spec: Causal21dAdmissionSpec = Causal21dAdmissionSpec(), resume: bool = False,
) -> dict[str, Any]:
    """Map each side causally, then rank every base count once globally."""
    long_root, short_root, root = map(Path, (long_dir, short_dir, output_dir))
    long_manifest = json.loads((long_root / "manifest.json").read_text(encoding="utf-8"))
    short_manifest = json.loads((short_root / "manifest.json").read_text(encoding="utf-8"))
    if (
        long_manifest.get("schema") != SCHEMA
        or short_manifest.get("schema") != SCHEMA
        or long_manifest.get("status") != "complete"
        or short_manifest.get("status") != "complete"
    ):
        raise StageIBaseFeatureLadderError("pooled base ladder needs two completed side ladders")
    long_denominator = _load_denominator_audit(long_root, long_manifest)
    short_denominator = _load_denominator_audit(short_root, short_manifest)
    full_candidate_rows = int(long_denominator.full_candidate_rows.iloc[0] + short_denominator.full_candidate_rows.iloc[0])
    request = {
        "schema": SCHEMA,
        "scope": "side_local_causal_21d_common_bps_mapping_then_one_pooled_global_ranking",
        "long_request_sha256": long_manifest.get("request_sha256"),
        "short_request_sha256": short_manifest.get("request_sha256"),
        "long_denominator_audit_sha256": long_manifest.get("denominator_audit_sha256"),
        "short_denominator_audit_sha256": short_manifest.get("denominator_audit_sha256"),
        "full_candidate_rows": full_candidate_rows,
        "admission_spec": admission_spec.__dict__, "base_only": True,
    }
    request_sha = _canonical_sha(request)
    existing = root / "manifest.json"
    if existing.is_file() and resume:
        value = json.loads(existing.read_text(encoding="utf-8"))
        metrics = root / "pooled_global_metrics.parquet"
        denominator_path = root / "pooled_denominator_audit.parquet"
        if (
            value.get("request_sha256") != request_sha
            or value.get("status") != "complete"
            or not metrics.is_file()
            or value.get("pooled_global_metrics_sha256") != file_sha256(metrics)
            or not denominator_path.is_file()
            or value.get("pooled_denominator_audit_sha256") != file_sha256(denominator_path)
        ):
            raise StageIBaseFeatureLadderError("pooled base feature ladder resume/hash drift")
        return {**value, "restart_status": "reused_verified_complete"}
    if root.exists() and not resume:
        raise FileExistsError(f"pooled base ladder output exists without --resume: {root}")
    long_arms = {item.name: item for item in (long_root / "arms").iterdir() if item.is_dir()}
    short_arms = {item.name: item for item in (short_root / "arms").iterdir() if item.is_dir()}
    if set(long_arms) != set(NESTED_SET_NAMES) or set(short_arms) != set(NESTED_SET_NAMES):
        raise StageIBaseFeatureLadderError("long/short base count arm sets differ from the fixed ladder")
    root.mkdir(parents=True, exist_ok=True)
    (root / "arms").mkdir()
    metrics: list[pd.DataFrame] = []
    for name in NESTED_SET_NAMES:
        combined = pd.concat([
            _load_arm(long_arms[name], name=name, side_manifest=long_manifest),
            _load_arm(short_arms[name], name=name, side_manifest=short_manifest),
        ], ignore_index=True)
        if combined.candidate_key.duplicated().any():
            raise StageIBaseFeatureLadderError("pooled base ladder candidate identities collide")
        mapped, audit = apply_causal_21d_side_admission(
            combined, score_column="r3_opportunity_score", net_column="exact_net_bps",
            decision_column="decision_ts", label_available_column="label_available_ts",
            identity_column="candidate_key", spec=admission_spec,
        )
        arm = root / "arms" / name
        arm.mkdir()
        mapped.to_parquet(arm / "mapped_predictions.parquet", index=False, compression="zstd")
        audit.to_parquet(arm / "admission_audit.parquet", index=False, compression="zstd")
        comparison = pooled_global_admission_comparison(
            mapped, raw_score_column="r3_opportunity_score", net_column="exact_net_bps",
            identity_column="candidate_key", top_fractions=TOP_FRACTIONS,
            original_population_rows=full_candidate_rows,
        )
        comparison["feature_set"] = name
        comparison["base_only"] = True
        comparison["freeze_eligible_now"] = False
        metrics.append(comparison)
    metrics_frame = pd.concat(metrics, ignore_index=True)
    metrics_frame.to_parquet(root / "pooled_global_metrics.parquet", index=False, compression="zstd")
    pooled_denominator = pd.concat([long_denominator, short_denominator], ignore_index=True)
    pooled_denominator.to_parquet(root / "pooled_denominator_audit.parquet", index=False, compression="zstd")
    manifest = {
        **request, "status": "complete", "request_sha256": request_sha,
        "ranking": "one pooled-global ranking only after side-local causal common-bps mapping; never per timestamp or side",
        "hpo_disposition": "fixed_source_HPO_diagnostic_only; count_specific_base_HPO_and_refit_required",
        "pooled_global_metrics_sha256": file_sha256(root / "pooled_global_metrics.parquet"),
        "pooled_denominator_audit_sha256": file_sha256(root / "pooled_denominator_audit.parquet"),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


__all__ = [
    "SCHEMA", "TOP_FRACTIONS", "BaseFeatureLadderInput", "BaseFeatureLadderConfig",
    "StageIBaseFeatureLadderError", "file_sha256", "run_side_base_feature_ladder",
    "run_pooled_base_feature_ladder",
]
