"""Executable, hash-bound Stage-I meta feature-count ladder.

This is deliberately a *sequential* meta experiment.  A completed per-side
base selector and its canonical top-30 handoff are frozen first.  The ladder
then varies only the already-materialised meta feature sets
(``automatic/20/30/40/60/full``), holding the full base OOF population,
candidate identities, chronological folds, target specification and frozen
base HPO lineage fixed.

The meta loss is fitted only on the base-candidate stream, but every strict OOF
validation block is scored.  This is important: side-local 21-day common-bps
mapping needs the complete resolved base population for support; the top-30
candidate condition remains an explicit action gate and is never confused with
the mapping reference population.

Feature-count runs are not production refits.  They publish an immutable
count-specific HPO/refit request for every side/set/target.  A result remains
non-freezable until that request is satisfied by a target-specific HPO/refit
artifact.  ``full_input_control`` remains eligible to make such a request.
"""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .prequential_r3_value_map import (
    PrequentialR3ValueMapConfig,
    base_oof_trust_features,
    prequential_same_side_r3_value_map,
)
from .stage_i_causal_admission import (
    Causal21dAdmissionSpec,
    apply_causal_21d_side_admission,
)
from .stage_i_meta_feature_challenger import MetaFeatureChallengePlan
from .stage_i_meta_target_execution import (
    IDENTITY,
    _canonical_sha,
    _classifier_features_for_spec,
    _full_population_fold_id,
    _validate_inputs,
    file_sha256,
    make_lgbm_predictor,
)
from .stage_i_meta_target_funnel import (
    MetaTargetSpec,
    _evaluation_target,
    fit_meta_target,
    reconstruct_meta_action,
)
from .stage_i_ranking import RANKING_POLICY, stable_stage_i_rank_frame


SCHEMA = "stage_i_meta_feature_ladder_execution_v1"
TOP_FRACTIONS = (0.01, 0.05, 0.10, 0.20)
Predictor = Callable[[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, MetaTargetSpec], np.ndarray]


class StageIMetaFeatureLadderError(ValueError):
    """Raised when a count ladder would mix lineage, targets or OOF rows."""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StageIMetaFeatureLadderError(f"JSON object expected: {path}")
    return value


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values if str(value).strip()))


def _valid_target_rows(ledger: pd.DataFrame) -> np.ndarray:
    """Exclude invalid future paths from fitting *and* map reference support."""
    valid = np.ones(len(ledger), dtype=bool)
    for column, expected in (
        ("target_invalid", False),
        ("label_valid", True),
        ("path_complete", True),
    ):
        if column not in ledger:
            continue
        value = ledger[column]
        if expected:
            valid &= value.astype(bool).to_numpy()
        else:
            valid &= ~value.astype(bool).to_numpy()
    return valid


def prepare_full_meta_ladder_population(
    ledger: pd.DataFrame,
    raw: pd.DataFrame,
    base: pd.DataFrame,
    candidate_handoff: pd.DataFrame,
    *,
    side: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build one full side OOF population and a separate candidate action gate.

    The score map is fit on valid rows in the complete base OOF ledger.  It is
    therefore not starved by the top-30 training/action gate.  The routine is
    intentionally target-free after the prequential map is constructed: the
    candidate handoff was fixed from base score only and is validated upstream.
    """
    original_side_population_rows = int(len(ledger))
    probability = base.loc[:, ["r3_p_adverse", "r3_p_weak", "r3_p_clear"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(np.float32)
    complete = np.isfinite(probability).all(axis=1)
    if np.any(np.isfinite(probability).any(axis=1) & ~complete):
        raise StageIMetaFeatureLadderError("base OOF simplex is partially missing")
    keep = np.flatnonzero(complete)
    ledger = ledger.iloc[keep].reset_index(drop=True).copy()
    raw = raw.iloc[keep].drop(columns=list(IDENTITY), errors="ignore").reset_index(drop=True).copy()
    probability = probability[keep]
    if "selected_base_candidate" not in candidate_handoff or "side_name" not in candidate_handoff:
        raise StageIMetaFeatureLadderError("candidate handoff lacks selected/side identity fields")
    if len(candidate_handoff) != len(ledger):
        raise StageIMetaFeatureLadderError("candidate handoff/full base OOF population differs")
    if not candidate_handoff.loc[:, list(IDENTITY)].reset_index(drop=True).equals(
        ledger.loc[:, list(IDENTITY)].reset_index(drop=True)
    ):
        raise StageIMetaFeatureLadderError(
            "candidate handoff/full base OOF identity order differs"
        )
    if not candidate_handoff.side_name.astype(str).str.lower().eq(str(side).lower()).all():
        raise StageIMetaFeatureLadderError("candidate handoff side does not match full base OOF population")
    selected = candidate_handoff.selected_base_candidate.astype(bool).to_numpy()
    decision = pd.to_datetime(ledger["__ts__"], utc=True, errors="raise") + pd.Timedelta(hours=1)
    available = pd.to_datetime(ledger["label_available_ts"], utc=True, errors="raise")
    if not available.eq(decision + pd.Timedelta(hours=12)).all():
        raise StageIMetaFeatureLadderError("full population must preserve decision+12h label availability")
    resolved = _valid_target_rows(ledger)
    if not np.any(resolved):
        raise StageIMetaFeatureLadderError(
            "full base OOF population contains no valid complete-path target rows"
        )
    score = probability[:, 2] - probability[:, 0]
    mapped = np.full(len(ledger), np.nan, dtype=np.float32)
    map_audit = pd.DataFrame()
    if int(resolved.sum()):
        fitted, map_audit, _ = prequential_same_side_r3_value_map(
            exact_net_bps=ledger.loc[resolved, "exact_net_bps"].to_numpy(np.float32),
            decision_timestamps=decision.loc[resolved],
            label_available_timestamps=available.loc[resolved],
            side=side,
            score=score[resolved],
            config=PrequentialR3ValueMapConfig(side=side),
        )
        mapped[resolved] = np.asarray(fitted, dtype=np.float32)
    raw["r3_p_adverse"], raw["r3_p_weak"], raw["r3_p_clear"] = (
        probability[:, 0], probability[:, 1], probability[:, 2]
    )
    raw["r3_opportunity_score"] = score
    raw["prequential_base_expected_net_bps"] = mapped
    # The prequential audit contains exactly the complete-path rows supplied
    # to the map above.  Construct trust on that same index and place it back
    # into the full frame; invalid rows intentionally remain missing.
    trust = base_oof_trust_features(probability[resolved], map_audit)
    # ``base_oof_trust_features`` is aligned to the mapped valid subset.  Keep
    # invalid rows explicitly missing; they are neither training nor mapping
    # references and must not acquire a synthetic causal feature value.
    for column in trust:
        value = np.full(len(raw), np.nan, dtype=np.float32)
        value[resolved] = trust[column].to_numpy(np.float32)
        raw[column] = value
    model = pd.concat(
        [
            ledger.loc[:, list(IDENTITY) + ["side_name", "exact_net_bps"]].reset_index(drop=True),
            raw.reset_index(drop=True),
        ],
        axis=1,
    )
    model["decision_ts"], model["label_available_ts"] = decision, available
    model["candidate_key"] = model.side_name.astype(str) + "::" + model.candidate_id.astype(str)
    model["candidate_selected"] = selected
    model["valid_resolved_target"] = resolved
    model["mapping_reference_eligible"] = resolved
    model["full_base_oof_population_rows"] = int(len(model))
    # Top-k is always a fraction of the original side population, not merely
    # of the later strict-OOF evaluation suffix.  Preserve this denominator
    # through every count arm so burn-in cannot inflate a reported tail.
    model["original_side_population_rows"] = original_side_population_rows
    model["base_top30_candidate_rows"] = int(selected.sum())
    handoff = candidate_handoff.copy()
    handoff["meta_feature_ladder_action_gate"] = selected
    handoff["mapping_reference_population"] = True
    return model, map_audit, handoff


def _fold_provenance_row(
    *, spec: MetaTargetSpec, side: str, fold_id: int,
    validation: pd.DataFrame, train: pd.DataFrame,
    candidate_train_rows: int, candidate_validation_rows: int,
    target_fit: Any,
) -> dict[str, Any]:
    thresholds = tuple(target_fit.residual_thresholds_bps)
    return {
        "arm_id": spec.arm_id,
        "target_family": spec.family,
        "side": side,
        "fold_id": int(fold_id),
        "train_full_prior_resolved_rows": int(len(train)),
        "train_candidate_rows": int(candidate_train_rows),
        "validation_scored_full_rows": int(len(validation)),
        "validation_action_candidate_rows": int(candidate_validation_rows),
        "validation_start_utc": validation.decision_ts.min().isoformat(),
        "validation_end_utc": validation.decision_ts.max().isoformat(),
        "train_max_label_available_utc": train.label_available_ts.max().isoformat(),
        "strict_prior_resolved": True,
        "mapping_reference_scope": "complete_side_base_oof_population_not_top30_prefilter",
        "residual_q33_bps": thresholds[0] if len(thresholds) == 2 else np.nan,
        "residual_q67_bps": thresholds[1] if len(thresholds) == 2 else np.nan,
        "zero_in_middle_tercile": (
            bool(thresholds[0] < 0.0 <= thresholds[1]) if len(thresholds) == 2 else np.nan
        ),
        "fold_semantic_valid": (
            bool(thresholds[0] < 0.0 <= thresholds[1]) if len(thresholds) == 2 else True
        ),
        "class_0_support": target_fit.class_support[0] if target_fit.class_support else np.nan,
        "class_1_support": target_fit.class_support[1] if target_fit.class_support else np.nan,
        "class_2_support": target_fit.class_support[2] if target_fit.class_support else np.nan,
        "class_0_residual_location_bps": target_fit.class_payoff_bps[0] if len(target_fit.class_payoff_bps) == 3 else np.nan,
        "class_1_residual_location_bps": target_fit.class_payoff_bps[1] if len(target_fit.class_payoff_bps) == 3 else np.nan,
        "class_2_residual_location_bps": target_fit.class_payoff_bps[2] if len(target_fit.class_payoff_bps) == 3 else np.nan,
    }


def run_strict_candidate_meta_feature_arm(
    frame: pd.DataFrame,
    spec: MetaTargetSpec,
    *,
    feature_columns: Sequence[str],
    fold_id: Sequence[int],
    predictor: Predictor,
    min_train_candidate_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit on prior resolved top-30 rows and score every held-out base row.

    Validation labels enter only target metrics.  Mutating a validation label
    cannot alter the prior fit, class thresholds, prediction or score for that
    fold.  The returned action flag adds the pre-frozen candidate gate after
    prediction; it does not reduce the mapping reference population.
    """
    features = _ordered_unique(feature_columns)
    required = {
        "side_name", "decision_ts", "label_available_ts", "exact_net_bps",
        "candidate_selected", "valid_resolved_target", "original_side_population_rows", *features,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise StageIMetaFeatureLadderError(f"meta ladder frame misses required fields: {missing}")
    if frame.side_name.astype(str).str.lower().nunique() != 1:
        raise StageIMetaFeatureLadderError("meta count head must remain side-local")
    original_population = pd.to_numeric(
        frame.original_side_population_rows, errors="coerce"
    ).to_numpy(float)
    if (
        not np.isfinite(original_population).all()
        or not np.allclose(original_population, original_population[0])
        or original_population[0] < len(frame)
    ):
        raise StageIMetaFeatureLadderError(
            "meta count head needs one valid original-side population denominator"
        )
    fold = np.asarray(fold_id, dtype=np.int32).reshape(-1)
    if len(fold) != len(frame) or not np.any(fold >= 0):
        raise StageIMetaFeatureLadderError("strict fold vector is missing evaluation rows")
    decision = pd.to_datetime(frame.decision_ts, utc=True, errors="raise")
    available = pd.to_datetime(frame.label_available_ts, utc=True, errors="raise")
    if not (available > decision).all():
        raise StageIMetaFeatureLadderError("labels must resolve after decisions")
    scores = np.full(len(frame), np.nan, dtype=np.float64)
    action = np.zeros(len(frame), dtype=bool)
    target = np.full(len(frame), np.nan, dtype=np.float64)
    prediction: np.ndarray | None = None
    prior: np.ndarray | None = None
    classes = 3 if spec.family == "quantile_ordinal_residual" else 4 if spec.family == "ordinal_residual" else 0
    if classes:
        prediction = np.full((len(frame), classes), np.nan, dtype=np.float64)
        if spec.family == "quantile_ordinal_residual":
            prior = np.full((len(frame), classes), np.nan, dtype=np.float64)
    else:
        prediction = np.full(len(frame), np.nan, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    side = str(frame.side_name.iloc[0]).lower()
    # Invalid/incomplete paths are deliberately absent from supervised fitting
    # *and* from the map reference.  They must not become a synthetic scored
    # failure simply because a base OOF probability exists for the candidate.
    evaluation_mask = (fold >= 0) & frame.valid_resolved_target.astype(bool).to_numpy()
    if not np.any(evaluation_mask):
        raise StageIMetaFeatureLadderError("strict meta folds contain no valid resolved evaluation rows")
    values = sorted(set(fold[evaluation_mask].tolist()))
    starts = [decision.iloc[np.flatnonzero(fold == value)].min() for value in values]
    if starts != sorted(starts):
        raise StageIMetaFeatureLadderError("strict meta folds are not chronological")
    for value, start in zip(values, starts, strict=True):
        validation_idx = np.flatnonzero((fold == value) & evaluation_mask)
        strict_prior = available.lt(start).to_numpy() & frame.valid_resolved_target.astype(bool).to_numpy()
        train_idx = np.flatnonzero(strict_prior & frame.candidate_selected.astype(bool).to_numpy())
        if len(train_idx) < int(min_train_candidate_rows):
            raise StageIMetaFeatureLadderError(
                f"fold {value} has only {len(train_idx)} prior-resolved candidate rows; "
                f"requires {min_train_candidate_rows}"
            )
        train, validation = frame.iloc[train_idx], frame.iloc[validation_idx]
        # Exact target and fitted calibration state use the candidate training
        # subset only, with the cutoff enforced again by ``fit_meta_target``.
        target_fit = fit_meta_target(train, spec, side=side, fit_before_utc=start)
        raw_prediction = np.asarray(
            predictor(
                train.loc[:, list(features)], target_fit.target,
                target_fit.sample_weight, validation.loc[:, list(features)], spec,
            )
        )
        fold_score, fold_action = reconstruct_meta_action(validation, target_fit, raw_prediction)
        if len(fold_score) != len(validation_idx):
            raise StageIMetaFeatureLadderError(f"fold {value} predictor misalignment")
        scores[validation_idx] = fold_score
        action[validation_idx] = (
            np.asarray(fold_action, dtype=bool)
            & validation.candidate_selected.astype(bool).to_numpy()
            & validation.valid_resolved_target.astype(bool).to_numpy()
        )
        target[validation_idx] = _evaluation_target(validation, spec, target_fit)
        if classes:
            if raw_prediction.shape != (len(validation_idx), classes):
                raise StageIMetaFeatureLadderError(f"fold {value} classifier simplex shape drift")
            assert prediction is not None
            prediction[validation_idx] = raw_prediction
            if prior is not None:
                vector = np.asarray(target_fit.class_support, dtype=float)
                prior[validation_idx] = vector / vector.sum()
        else:
            assert prediction is not None
            vector = raw_prediction[:, 1] if raw_prediction.ndim == 2 else raw_prediction.reshape(-1)
            prediction[validation_idx] = vector
        rows.append(_fold_provenance_row(
            spec=spec, side=side, fold_id=int(value), validation=validation,
            train=frame.loc[strict_prior], candidate_train_rows=len(train_idx),
            candidate_validation_rows=int(validation.candidate_selected.astype(bool).sum()),
            target_fit=target_fit,
        ))
    evaluation = evaluation_mask
    if not np.isfinite(scores[evaluation]).all() or not np.isfinite(prediction[evaluation]).all():
        raise StageIMetaFeatureLadderError("incomplete strict OOF predictions")
    output = frame.loc[evaluation, list(IDENTITY) + [
        "candidate_key", "side_name", "decision_ts", "label_available_ts", "exact_net_bps",
        "r3_opportunity_score", "prequential_base_expected_net_bps", "candidate_selected",
        "valid_resolved_target", "mapping_reference_eligible", "original_side_population_rows",
    ]].copy()
    output["fold_id"] = fold[evaluation]
    output["score"] = scores[evaluation]
    output["action_admitted"] = action[evaluation]
    output["target"] = target[evaluation]
    if prediction.ndim == 2:
        for index in range(prediction.shape[1]):
            output[f"prediction_class_{index}"] = prediction[evaluation, index]
        if prior is not None:
            for index in range(prior.shape[1]):
                output[f"prior_prediction_class_{index}"] = prior[evaluation, index]
    else:
        output["prediction"] = prediction[evaluation]
    output["target_semantic_valid"] = bool(
        all(row["fold_semantic_valid"] for row in rows)
    )
    return output.reset_index(drop=True), pd.DataFrame(rows)


def _hpo_refit_request(
    *, plan: MetaFeatureChallengePlan, feature_set: Any,
    spec: MetaTargetSpec, side: str, source_meta_params: Mapping[str, Any],
    model_features: Sequence[str], ladder_request_sha256: str,
) -> dict[str, Any]:
    """A count winner is explicitly non-freezable until this request is met."""
    payload = {
        "schema": "stage_i_meta_count_specific_hpo_refit_request_v1",
        "side": side,
        "feature_set": feature_set.name,
        "feature_set_sha256": feature_set.source_hash,
        "feature_set_features": list(feature_set.features),
        # T3Q deliberately removes its hidden converted-bps anchor from the
        # classifier matrix.  A count-specific HPO request must bind the
        # matrix actually scored, not merely its parent feature set.
        "model_features": list(model_features),
        "model_features_sha256": _canonical_sha(list(model_features)),
        "target_arm_id": spec.arm_id,
        "target_family": spec.family,
        "plan_sha256": plan.plan_hash,
        "frozen_base_oof_sha256": plan.frozen_base_oof_sha256,
        "candidate_scope": "frozen_same_side_base_top30_for_training; full_base_oof_for_scoring_and_21d_reference",
        "source_meta_hpo_params_sha256": _canonical_sha(dict(source_meta_params)),
        "ladder_request_sha256": ladder_request_sha256,
        "required_artifact_schema": "stage_i_meta_count_specific_hpo_refit_v1",
        "freeze_eligible_now": False,
        "freeze_blocker": "count_specific_target_HPO_and_refit_required",
        "full_input_promotion_eligible": bool(feature_set.promotion_eligible),
    }
    payload["request_sha256"] = _canonical_sha(payload)
    return payload


def _arm_key(item: Mapping[str, Any]) -> tuple[str, str]:
    return str(item["feature_set"]), str(item["target_arm_id"])


def _verified_arm_inventory(
    root: Path, *, request_sha: str, expected_inventory: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Verify the complete planned arm set, never only a root checksum."""
    expected = {_arm_key(item): dict(item) for item in expected_inventory}
    if len(expected) != len(expected_inventory):
        raise StageIMetaFeatureLadderError("duplicate planned meta ladder arm")
    actual_dirs = {
        (path.parent.name, path.name): path
        for path in (root / "arms").glob("*/*") if path.is_dir()
    }
    if set(actual_dirs) != set(expected):
        raise StageIMetaFeatureLadderError(
            "meta feature ladder arm inventory drift "
            f"expected={sorted(expected)} actual={sorted(actual_dirs)}"
        )
    verified: list[dict[str, Any]] = []
    for key in sorted(expected):
        expected_item, arm_root = expected[key], actual_dirs[key]
        manifest_path = arm_root / "manifest.json"
        prediction_path = arm_root / "oof_predictions.parquet"
        provenance_path = arm_root / "fold_provenance.parquet"
        hpo_path = arm_root / "count_specific_hpo_refit_request.json"
        if not all(path.is_file() for path in (manifest_path, prediction_path, provenance_path, hpo_path)):
            raise StageIMetaFeatureLadderError(f"incomplete meta feature ladder arm: {arm_root}")
        manifest, hpo = _json(manifest_path), _json(hpo_path)
        expected_feature_sha = str(expected_item["feature_set_sha256"])
        checks = (
            manifest.get("schema") == SCHEMA,
            manifest.get("status") == "complete",
            manifest.get("request_sha256") == request_sha,
            manifest.get("feature_set") == key[0],
            manifest.get("target_arm_id") == key[1],
            manifest.get("target_family") == expected_item["target_family"],
            manifest.get("feature_set_sha256") == expected_feature_sha,
            manifest.get("classifier_feature_contract_sha256")
            == _canonical_sha(expected_item["classifier_features"]),
            manifest.get("oof_predictions_sha256") == file_sha256(prediction_path),
            manifest.get("fold_provenance_sha256") == file_sha256(provenance_path),
            manifest.get("count_specific_hpo_refit_request_sha256") == file_sha256(hpo_path),
            hpo.get("ladder_request_sha256") == request_sha,
            hpo.get("feature_set") == key[0],
            hpo.get("target_arm_id") == key[1],
            hpo.get("target_family") == expected_item["target_family"],
            hpo.get("feature_set_sha256") == expected_feature_sha,
            hpo.get("model_features_sha256")
            == _canonical_sha(expected_item["classifier_features"]),
            hpo.get("request_sha256") == _canonical_sha(
                {key_: value for key_, value in hpo.items() if key_ != "request_sha256"}
            ),
        )
        if not all(checks):
            raise StageIMetaFeatureLadderError(f"meta feature ladder arm checksum/lineage drift: {arm_root}")
        verified.append({
            "feature_set": key[0],
            "target_arm_id": key[1],
            "arm_manifest_sha256": file_sha256(manifest_path),
            "oof_predictions_sha256": file_sha256(prediction_path),
            "fold_provenance_sha256": file_sha256(provenance_path),
            "count_specific_hpo_refit_request_sha256": file_sha256(hpo_path),
        })
    return verified


def _side_resume_verified(
    root: Path, request_sha: str,
    expected_inventory: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any] | None:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = _json(manifest_path)
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("request_sha256") != request_sha
    ):
        raise StageIMetaFeatureLadderError("completed meta feature ladder request/hash drift")
    for relative, field in (
        ("evaluation_population.parquet", "evaluation_population_sha256"),
        ("full_mapping_audit.parquet", "full_mapping_audit_sha256"),
        ("hpo_refit_requests.json", "hpo_refit_requests_sha256"),
    ):
        path = root / relative
        if not path.is_file() or manifest.get(field) != file_sha256(path):
            raise StageIMetaFeatureLadderError("completed meta feature ladder checksum drift")
    if expected_inventory is not None:
        if manifest.get("planned_arm_inventory") != [dict(item) for item in expected_inventory]:
            raise StageIMetaFeatureLadderError("completed meta feature ladder planned-arm drift")
        verified = _verified_arm_inventory(
            root, request_sha=request_sha, expected_inventory=expected_inventory,
        )
        if manifest.get("arm_inventory") != verified:
            raise StageIMetaFeatureLadderError("completed meta feature ladder arm artifact drift")
        request_path = root / "hpo_refit_requests.json"
        requests = json.loads(request_path.read_text(encoding="utf-8"))
        if not isinstance(requests, list) or len(requests) != len(verified):
            raise StageIMetaFeatureLadderError("completed meta feature ladder HPO request inventory drift")
        if {
            _arm_key(item) for item in requests if isinstance(item, Mapping)
        } != {_arm_key(item) for item in expected_inventory}:
            raise StageIMetaFeatureLadderError("completed meta feature ladder HPO request arms drift")
    return manifest


def run_side_meta_feature_ladder(
    *,
    selector_dir: str | Path,
    base_selection_dir: str | Path,
    meta_selection_dir: str | Path,
    plan: MetaFeatureChallengePlan,
    output_dir: str | Path,
    specs: Sequence[MetaTargetSpec],
    n_validation_folds: int = 4,
    min_train_candidate_rows: int = 500,
    resume: bool = False,
    predictor: Predictor | None = None,
) -> dict[str, Any]:
    """Execute all six fixed meta feature sets for one side on matched OOF rows."""
    side = str(plan.side).lower()
    if side not in {"long", "short"}:
        raise StageIMetaFeatureLadderError("plan side must be long/short")
    if tuple(item.name for item in plan.feature_sets) != (
        "automatic_sparse", "full_input_control", "top20", "top30", "top40", "top60",
    ):
        raise StageIMetaFeatureLadderError("meta feature plan does not expose the complete fixed ladder")
    ordered_specs = tuple(specs)
    if not ordered_specs or len({spec.arm_id for spec in ordered_specs}) != len(ordered_specs):
        raise StageIMetaFeatureLadderError("ordered target specs must be non-empty and unique")
    selector, base_dir, meta_dir, root = map(Path, (selector_dir, base_selection_dir, meta_selection_dir, output_dir))
    ledger, raw, base, handoff, meta_manifest, _selected, params = _validate_inputs(
        selector_dir=selector, base_dir=base_dir, meta_dir=meta_dir, side=side
    )
    base_manifest = _json(base_dir / side / "manifest.json")
    base_policy = str(base_manifest.get("correlation_policy", ""))
    meta_policy = str(meta_manifest.get("correlation_policy", ""))
    lineage_policy = str(
        (meta_manifest.get("base_correlation_lineage") or {}).get(
            "correlation_policy", ""
        )
    )
    if (
        not plan.correlation_policy
        or plan.correlation_policy != plan.base_correlation_policy
        or plan.correlation_policy != base_policy
        or plan.correlation_policy != meta_policy
        or lineage_policy != base_policy
        or str(meta_manifest.get("base_correlation_policy", "")) != base_policy
    ):
        raise StageIMetaFeatureLadderError(
            "meta feature ladder correlation-policy lineage is missing or mismatched"
        )
    # Bind the provided plan to the exact current completed artifacts.  This
    # prevents using an automatic MDA ranking from a different base OOF state.
    expected = {
        "source_manifest_sha256": file_sha256(meta_dir / side / "manifest.json"),
        "selector_manifest_sha256": file_sha256(selector / "manifest.json"),
        "selector_feature_contract_sha256": file_sha256(selector / "selector_feature_contract.json"),
        "frozen_base_manifest_sha256": file_sha256(base_dir / side / "manifest.json"),
        "frozen_base_oof_sha256": file_sha256(base_dir / side / "selector_base_oof.parquet"),
        "candidate_handoff_audit_sha256": file_sha256(meta_dir / side / "base_candidate_handoff_audit.parquet"),
    }
    if any(getattr(plan, key) != value for key, value in expected.items()):
        raise StageIMetaFeatureLadderError("meta feature plan lineage/hash drift")
    full, map_audit, enforced_handoff = prepare_full_meta_ladder_population(
        ledger, raw, base, handoff, side=side
    )
    # Fold boundaries are constructed from the exact frozen top-30 handoff
    # and then projected onto the full OOF population.  Building blocks from
    # all rows would silently change candidate-fold identities merely because
    # the mapper needs a larger reference stream.
    fold = _full_population_fold_id(
        full.assign(action_candidate=full.candidate_selected.astype(bool)),
        n_folds=int(n_validation_folds),
        min_train_rows=int(min_train_candidate_rows),
    )
    evaluation = full.loc[
        (fold >= 0) & full.valid_resolved_target.astype(bool).to_numpy()
    ].reset_index(drop=True)
    planned_arm_inventory = [
        {
            "feature_set": feature_set.name,
            "feature_set_sha256": feature_set.source_hash,
            "target_arm_id": spec.arm_id,
            "target_family": spec.family,
            "classifier_features": list(
                _classifier_features_for_spec(feature_set.features, spec)
            ),
        }
        for feature_set in plan.feature_sets
        for spec in ordered_specs
    ]
    request = {
        "schema": SCHEMA,
        "side": side,
        "plan_sha256": plan.plan_hash,
        "lineage": expected,
        "correlation_policy": plan.correlation_policy,
        "specs": [spec.__dict__ for spec in ordered_specs],
        "n_validation_folds": int(n_validation_folds),
        "min_train_candidate_rows": int(min_train_candidate_rows),
        "candidate_training_scope": "frozen_same_side_global_base_top30; never_per_timestamp",
        "scoring_mapping_scope": "full_same_side_base_oof_population; candidate_gate_applied_after_scoring",
        "full_input_control_promotion_eligible": True,
        "planned_arm_inventory": planned_arm_inventory,
    }
    request_sha = _canonical_sha(request)
    previous = _side_resume_verified(
        root, request_sha, planned_arm_inventory,
    ) if resume else None
    if previous is not None:
        return {**previous, "restart_status": "reused_verified_complete"}
    if root.exists() and not resume:
        raise FileExistsError(f"meta feature ladder output exists without --resume: {root}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "arms").mkdir(exist_ok=True)
    model_predictor = predictor or make_lgbm_predictor(params)
    hpo_requests: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for feature_set in plan.feature_sets:
        for spec in ordered_specs:
            features = _classifier_features_for_spec(feature_set.features, spec)
            # T3Q removes its hidden bps anchor from the classifier matrix, but
            # the count plan still records the original full authorized set.
            prediction, provenance = run_strict_candidate_meta_feature_arm(
                full, spec, feature_columns=features, fold_id=fold,
                predictor=model_predictor,
                min_train_candidate_rows=int(min_train_candidate_rows),
            )
            arm_dir = root / "arms" / feature_set.name / spec.arm_id
            arm_dir.mkdir(parents=True, exist_ok=False)
            prediction.to_parquet(arm_dir / "oof_predictions.parquet", index=False, compression="zstd")
            provenance.to_parquet(arm_dir / "fold_provenance.parquet", index=False, compression="zstd")
            hpo = _hpo_refit_request(
                plan=plan, feature_set=feature_set, spec=spec, side=side,
                source_meta_params=params, model_features=features,
                ladder_request_sha256=request_sha,
            )
            hpo_requests.append(hpo)
            request_path = arm_dir / "count_specific_hpo_refit_request.json"
            request_path.write_text(json.dumps(hpo, indent=2, sort_keys=True) + "\n")
            payload = {
                "schema": SCHEMA,
                "status": "complete",
                "request_sha256": request_sha,
                "side": side,
                "feature_set": feature_set.name,
                "feature_set_sha256": feature_set.source_hash,
                "feature_count": len(feature_set.features),
                "classifier_feature_count": len(features),
                "classifier_feature_contract_sha256": _canonical_sha(list(features)),
                "target_arm_id": spec.arm_id,
                "target_family": spec.family,
                "full_input_promotion_eligible": bool(feature_set.promotion_eligible),
                "freeze_eligible_now": False,
                "count_specific_hpo_refit_request_sha256": file_sha256(request_path),
                "oof_predictions_sha256": file_sha256(arm_dir / "oof_predictions.parquet"),
                "fold_provenance_sha256": file_sha256(arm_dir / "fold_provenance.parquet"),
            }
            (arm_dir / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            for fraction in TOP_FRACTIONS:
                eligible = prediction.loc[prediction.action_admitted.astype(bool)].copy()
                original_population_rows = int(
                    prediction.original_side_population_rows.iloc[0]
                )
                requested_topk_rows = max(
                    1, int(math.ceil(float(fraction) * original_population_rows))
                )
                selected_count = (
                    min(
                        len(eligible),
                        requested_topk_rows,
                    )
                    if len(eligible)
                    else 0
                )
                selected = stable_stage_i_rank_frame(
                    eligible, score_column="score", candidate_id_column="candidate_key",
                    decision_column="decision_ts",
                ).head(selected_count)
                metric_rows.append({
                    "scope": "side_local_raw_score_before_common_bps_mapping",
                    "side": side, "feature_set": feature_set.name, "target_arm_id": spec.arm_id,
                    "top_fraction": float(fraction), "full_oof_rows": int(len(full)),
                    "original_side_population_rows": original_population_rows,
                    "evaluation_full_rows": int(len(prediction)),
                    "action_candidate_rows": int(prediction.action_admitted.sum()),
                    "selected_rows": int(len(selected)),
                    "requested_topk_rows_original_population": requested_topk_rows,
                    "net_bps_per_trade": float(selected.exact_net_bps.mean()) if len(selected) else np.nan,
                    "gross_bps_per_trade": np.nan,
                    "ranking": "side_local_diagnostic_only; pooled ranking occurs only after common_bps_mapping",
                    "freeze_eligible_now": False,
                })
    evaluation.to_parquet(root / "evaluation_population.parquet", index=False, compression="zstd")
    map_audit.to_parquet(root / "full_mapping_audit.parquet", index=False, compression="zstd")
    enforced_handoff.to_parquet(root / "base_candidate_handoff_audit.parquet", index=False, compression="zstd")
    pd.DataFrame(metric_rows).to_parquet(root / "side_raw_metrics.parquet", index=False, compression="zstd")
    (root / "hpo_refit_requests.json").write_text(json.dumps(hpo_requests, indent=2, sort_keys=True) + "\n")
    arm_inventory = _verified_arm_inventory(
        root, request_sha=request_sha, expected_inventory=planned_arm_inventory,
    )
    manifest = {
        **request,
        "status": "complete",
        "request_sha256": request_sha,
        "evaluation_rows": int(len(evaluation)),
        "full_base_oof_rows": int(len(full)),
        "mapping_reference_rows": int(full.mapping_reference_eligible.sum()),
        "candidate_training_rows": int((full.candidate_selected & full.valid_resolved_target).sum()),
        "candidate_action_rows": int(full.candidate_selected.sum()),
        "hpo_disposition": "all feature-count arms require target-specific count-HPO/refit before winner freezing",
        "freeze_eligible_now": False,
        "planned_arm_inventory": planned_arm_inventory,
        "arm_inventory": arm_inventory,
        "evaluation_population_sha256": file_sha256(root / "evaluation_population.parquet"),
        "full_mapping_audit_sha256": file_sha256(root / "full_mapping_audit.parquet"),
        "hpo_refit_requests_sha256": file_sha256(root / "hpo_refit_requests.json"),
        "side_raw_metrics_sha256": file_sha256(root / "side_raw_metrics.parquet"),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _load_prediction(path: Path, *, request_sha: str, expected: Mapping[str, Any]) -> pd.DataFrame:
    manifest = _json(path / "manifest.json")
    prediction_path = path / "oof_predictions.parquet"
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("request_sha256") != request_sha
        or manifest.get("feature_set") != expected["feature_set"]
        or manifest.get("target_arm_id") != expected["target_arm_id"]
        or manifest.get("oof_predictions_sha256") != file_sha256(prediction_path)
    ):
        raise StageIMetaFeatureLadderError(f"meta ladder arm checksum drift: {path}")
    return pd.read_parquet(prediction_path)


def _verified_side_arm_roots(root: Path) -> tuple[dict[str, Any], dict[tuple[str, str], Path]]:
    manifest = _json(root / "manifest.json")
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "complete":
        raise StageIMetaFeatureLadderError("pooled ladder needs two completed side artifacts")
    request_sha = str(manifest.get("request_sha256", ""))
    planned = manifest.get("planned_arm_inventory")
    if not request_sha or not isinstance(planned, list):
        raise StageIMetaFeatureLadderError("pooled ladder side artifact lacks immutable arm inventory")
    verified = _verified_arm_inventory(
        root, request_sha=request_sha, expected_inventory=planned,
    )
    if manifest.get("arm_inventory") != verified:
        raise StageIMetaFeatureLadderError("pooled ladder side arm inventory/hash drift")
    mapping = {
        _arm_key(item): root / "arms" / str(item["feature_set"]) / str(item["target_arm_id"])
        for item in planned
    }
    return manifest, mapping


def run_pooled_meta_feature_ladder(
    *,
    long_dir: str | Path,
    short_dir: str | Path,
    output_dir: str | Path,
    admission_spec: Causal21dAdmissionSpec = Causal21dAdmissionSpec(),
    resume: bool = False,
) -> dict[str, Any]:
    """Apply causal side-local 21d maps, then rank each arm once globally."""
    long_root, short_root, root = map(Path, (long_dir, short_dir, output_dir))
    long_manifest, long_arms = _verified_side_arm_roots(long_root)
    short_manifest, short_arms = _verified_side_arm_roots(short_root)
    request = {
        "schema": SCHEMA,
        "scope": "side_local_causal_common_bps_mapping_then_one_pooled_global_ranking",
        "long_request_sha256": long_manifest.get("request_sha256"),
        "short_request_sha256": short_manifest.get("request_sha256"),
        "admission_spec": admission_spec.__dict__,
    }
    request_sha = _canonical_sha(request)
    existing = root / "manifest.json"
    if existing.is_file() and resume:
        value = _json(existing)
        if value.get("request_sha256") != request_sha or value.get("status") != "complete":
            raise StageIMetaFeatureLadderError("pooled meta ladder resume drift")
        metrics = root / "pooled_global_metrics.parquet"
        if value.get("pooled_global_metrics_sha256") != file_sha256(metrics):
            raise StageIMetaFeatureLadderError("pooled meta ladder metrics drift")
        return {**value, "restart_status": "reused_verified_complete"}
    if root.exists() and not resume:
        raise FileExistsError(f"pooled meta feature ladder output exists without --resume: {root}")
    if set(long_arms) != set(short_arms) or not long_arms:
        raise StageIMetaFeatureLadderError("long/short meta count arms differ")
    root.mkdir(parents=True, exist_ok=True)
    (root / "arms").mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []
    for key in sorted(long_arms):
        feature_set, arm_id = key
        long_expected = next(item for item in long_manifest["planned_arm_inventory"] if _arm_key(item) == key)
        short_expected = next(item for item in short_manifest["planned_arm_inventory"] if _arm_key(item) == key)
        combined = pd.concat([
            _load_prediction(long_arms[key], request_sha=long_manifest["request_sha256"], expected=long_expected),
            _load_prediction(short_arms[key], request_sha=short_manifest["request_sha256"], expected=short_expected),
        ], ignore_index=True)
        if combined.candidate_key.duplicated().any():
            raise StageIMetaFeatureLadderError("pooled meta ladder candidate identities collide")
        mapped, audit = apply_causal_21d_side_admission(
            combined,
            score_column="score", net_column="exact_net_bps",
            decision_column="decision_ts", label_available_column="label_available_ts",
            identity_column="candidate_key", spec=admission_spec,
        )
        mapped["final_action_admitted"] = (
            mapped.action_admitted.astype(bool)
            & mapped.causal_21d_side_admitted_ge_50bps.astype(bool)
        )
        mapped["mapping_reference_scope"] = "full_side_base_oof_scored_population; candidate gate only at action"
        arm_root = root / "arms" / feature_set / arm_id
        arm_root.mkdir(parents=True, exist_ok=False)
        mapped.to_parquet(arm_root / "mapped_predictions.parquet", index=False, compression="zstd")
        audit.to_parquet(arm_root / "admission_audit.parquet", index=False, compression="zstd")
        for fraction in TOP_FRACTIONS:
            raw = mapped.loc[mapped.action_admitted.astype(bool)].copy()
            admitted = mapped.loc[mapped.final_action_admitted.astype(bool)].copy()
            side_denominators = (
                mapped.groupby("side_name", observed=True)["original_side_population_rows"]
                .first()
            )
            if (
                side_denominators.empty
                or side_denominators.isna().any()
                or (side_denominators <= 0).any()
            ):
                raise StageIMetaFeatureLadderError(
                    "pooled meta ladder lacks original-side population denominators"
                )
            original_population_rows = int(side_denominators.sum())
            requested = max(1, int(math.ceil(float(fraction) * original_population_rows)))
            for label, population, score_column in (
                ("without_admission_raw_global", raw, "score"),
                ("with_admission_mapped_pooled_global", admitted, "causal_21d_side_expected_net_bps"),
            ):
                selected = stable_stage_i_rank_frame(
                    population, score_column=score_column, candidate_id_column="candidate_key",
                    decision_column="decision_ts",
                ).head(min(len(population), requested))
                month = pd.to_datetime(selected.decision_ts, utc=True).dt.strftime("%Y-%m") if len(selected) else pd.Series(dtype=str)
                side_net = selected.groupby("side_name", observed=True).exact_net_bps.agg(["size", "mean"]) if len(selected) else pd.DataFrame()
                month_net = selected.groupby(month, observed=True).exact_net_bps.agg(["size", "mean"]) if len(selected) else pd.DataFrame()
                rows.append({
                    "feature_set": feature_set,
                    "target_arm_id": arm_id,
                    "comparison": label,
                    "top_fraction": float(fraction),
                    "full_scored_rows": int(len(mapped)),
                    "original_population_rows": original_population_rows,
                    "scored_population_fraction": float(
                        len(mapped) / original_population_rows
                    ),
                    "mapping_reference_full_rows": int(mapped.mapping_reference_eligible.astype(bool).sum()),
                    "action_candidate_rows": int(mapped.action_admitted.astype(bool).sum()),
                    "mapped_rows": int(mapped.causal_21d_side_expected_net_bps.notna().sum()),
                    "admitted_rows": int(mapped.final_action_admitted.astype(bool).sum()),
                    "requested_topk_rows": requested,
                    "selected_rows": int(len(selected)),
                    "net_bps_per_trade": float(selected.exact_net_bps.mean()) if len(selected) else np.nan,
                    "selected_long_rows": int(side_net.loc["long", "size"]) if "long" in side_net.index else 0,
                    "selected_short_rows": int(side_net.loc["short", "size"]) if "short" in side_net.index else 0,
                    "selected_long_net_bps": float(side_net.loc["long", "mean"]) if "long" in side_net.index else np.nan,
                    "selected_short_net_bps": float(side_net.loc["short", "mean"]) if "short" in side_net.index else np.nan,
                    "worst_month": str(month_net["mean"].idxmin()) if len(month_net) else None,
                    "worst_month_net_bps_per_trade": float(month_net["mean"].min()) if len(month_net) else np.nan,
                    "worst_month_selected_rows": int(month_net.loc[month_net["mean"].idxmin(), "size"]) if len(month_net) else 0,
                    "ranking_tie_policy": RANKING_POLICY,
                    "full_input_promotion_eligible": feature_set == "full_input_control",
                    "freeze_eligible_now": False,
                })
    metrics = pd.DataFrame(rows)
    metrics.to_parquet(root / "pooled_global_metrics.parquet", index=False, compression="zstd")
    manifest = {
        **request,
        "status": "complete",
        "request_sha256": request_sha,
        "ranking": "one pooled-global selection only after side-local causal common-bps maps",
        "hpo_disposition": "no feature-count arm can be frozen until its count-specific target HPO/refit request is satisfied",
        "pooled_global_metrics_sha256": file_sha256(root / "pooled_global_metrics.parquet"),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


__all__ = [
    "SCHEMA",
    "StageIMetaFeatureLadderError",
    "prepare_full_meta_ladder_population",
    "run_strict_candidate_meta_feature_arm",
    "run_side_meta_feature_ladder",
    "run_pooled_meta_feature_ladder",
]
