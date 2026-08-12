"""Checkpointed execution for the guarded Stage-I meta-target funnel."""

from __future__ import annotations

from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import CFG
from .stage_i_causal_admission import (
    Causal21dAdmissionSpec,
    apply_causal_21d_side_admission,
)
from .prequential_r3_value_map import (
    PrequentialR3ValueMapConfig,
    base_oof_trust_features,
    prequential_same_side_r3_value_map,
)
from .stage_i_feature_selection import (
    STAGE_I_META_BASE_OOF_HANDOFF_FEATURES,
    resolve_stage_i_feature_universe,
)
from .stage_i_meta_target_funnel import (
    MetaOOFArm,
    MetaTargetSpec,
    SCHEMA as FUNNEL_SCHEMA,
    TOP_FRACTIONS,
    current_huber_control_arm,
    StageIMetaTargetError,
    default_meta_target_specs,
    focused_quantile_meta_target_specs,
    evaluate_meta_oof_arms,
    mandatory_control_arms,
    quantile_prior_conversion_control_arm,
    fit_meta_target,
    reconstruct_meta_action,
    _evaluation_target,
    select_meta_arm_with_noop_gate,
)
from .stage_i_strict_oof import _validation_blocks
from .stage_i_ranking import stable_stage_i_topk_positions


SCHEMA = "stage_i_meta_target_funnel_execution_v1"
IDENTITY = ("candidate_id", "__ts__", "__symbol__")
TARGET_VALIDITY_COLUMNS = ("target_invalid", "label_valid", "path_complete")
Predictor = Callable[[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, MetaTargetSpec], np.ndarray]


@dataclass(frozen=True)
class _FullPopulationArmResult:
    """One candidate-trained arm scored on every contemporaneous side row."""

    full_arm: MetaOOFArm
    action_arm: MetaOOFArm
    full_evaluation_positions: np.ndarray
    action_evaluation_positions: np.ndarray
    fold_provenance: pd.DataFrame


def _classifier_features_for_spec(
    features: Sequence[str], spec: MetaTargetSpec
) -> tuple[str, ...]:
    ordered = tuple(dict.fromkeys(map(str, features)))
    if spec.family != "quantile_ordinal_residual":
        return ordered
    forbidden = (
        "prequential_", "reconstructed_", "mapped_expected_",
        "expected_net_bps", "converted_ev", "converted_score",
    )
    clean = tuple(
        feature for feature in ordered
        if not any(token in feature.lower() for token in forbidden)
    )
    required_raw = {
        feature for feature in STAGE_I_META_BASE_OOF_HANDOFF_FEATURES
        if feature != "prequential_base_expected_net_bps"
    }
    if not required_raw.issubset(clean):
        raise StageIMetaTargetExecutionError(
            "T3Q classifier must retain direct same-side base simplex/trust inputs"
        )
    if not clean:
        raise StageIMetaTargetExecutionError("T3Q has no unconverted classifier inputs")
    return clean


class StageIMetaTargetExecutionError(StageIMetaTargetError):
    pass


def _valid_complete_target_mask(ledger: pd.DataFrame) -> np.ndarray:
    """Return the only rows allowed to supply supervised economic outcomes.

    A finite placeholder ``exact_net_bps`` is not evidence that a future path
    was labelable.  Stage-I production artifacts must therefore carry all three
    validity fields explicitly; legacy artifacts fail closed rather than
    silently turning missing paths into economic failures.
    """
    missing = sorted(set(TARGET_VALIDITY_COLUMNS).difference(ledger.columns))
    if missing:
        raise StageIMetaTargetExecutionError(
            "Stage-I meta target execution requires explicit target validity "
            f"provenance: {missing}"
        )
    invalid = ledger["target_invalid"]
    label_valid = ledger["label_valid"]
    path_complete = ledger["path_complete"]
    # Nullable booleans and malformed values fail closed.  In particular, do
    # not let ``NaN.astype(bool)`` manufacture a valid supervised row.
    mask = (
        ~invalid.fillna(True).astype(bool).to_numpy()
        & label_valid.fillna(False).astype(bool).to_numpy()
        & path_complete.fillna(False).astype(bool).to_numpy()
    )
    exact = pd.to_numeric(ledger["exact_net_bps"], errors="coerce").to_numpy(float)
    return mask & np.isfinite(exact)


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StageIMetaTargetExecutionError(f"JSON object expected: {path}")
    return value


def _validate_inputs(
    *, selector_dir: Path, base_dir: Path, meta_dir: Path, side: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], tuple[str, ...], dict[str, Any]]:
    selector_manifest_path = selector_dir / "manifest.json"
    selector_contract_path = selector_dir / "selector_feature_contract.json"
    ledger_path, features_path = selector_dir / "selector_ledger.parquet", selector_dir / "selector_features.parquet"
    base_root, meta_root = base_dir / side, meta_dir / side
    base_manifest_path, base_oof_path = base_root / "manifest.json", base_root / "selector_base_oof.parquet"
    meta_manifest_path = meta_root / "manifest.json"
    candidate_handoff_path = meta_root / "base_candidate_handoff_audit.parquet"
    required_paths = (selector_manifest_path, selector_contract_path, ledger_path, features_path, base_manifest_path, base_oof_path, meta_manifest_path, candidate_handoff_path)
    if any(not path.is_file() for path in required_paths):
        raise StageIMetaTargetExecutionError(f"{side}: selector/base/meta input contract is incomplete")
    selector_manifest, base_manifest, meta_manifest = _json(selector_manifest_path), _json(base_manifest_path), _json(meta_manifest_path)
    if selector_manifest.get("status") != "complete":
        raise StageIMetaTargetExecutionError("selector manifest is incomplete")
    if (
        base_manifest.get("schema") != "stage_i_base_feature_selection_v1"
        or base_manifest.get("status") != "complete"
        or str(base_manifest.get("side", "")).lower() != side
        or base_manifest.get("selector_base_oof_sha256") != file_sha256(base_oof_path)
        or base_manifest.get("selector_sample_manifest_sha256") != file_sha256(selector_manifest_path)
        or base_manifest.get("selector_feature_contract_sha256") != file_sha256(selector_contract_path)
    ):
        raise StageIMetaTargetExecutionError(f"{side}: base-selection OOF provenance drift")
    if (
        meta_manifest.get("schema") != "stage_i_meta_feature_selection_v1"
        or meta_manifest.get("status") != "complete"
        or str(meta_manifest.get("side", "")).lower() != side
        or meta_manifest.get("selector_sample_manifest_sha256") != file_sha256(selector_manifest_path)
        or meta_manifest.get("selector_feature_contract_sha256") != file_sha256(selector_contract_path)
        or meta_manifest.get("base_selector_oof_sha256") != file_sha256(base_oof_path)
        or meta_manifest.get("base_selector_manifest_sha256") != file_sha256(base_manifest_path)
        or meta_manifest.get("base_candidate_handoff_audit_sha256")
        != file_sha256(candidate_handoff_path)
        or not np.isclose(float(meta_manifest.get("base_candidate_fraction", -1.0)), 0.30)
        or "never_per_timestamp" not in str(meta_manifest.get("base_candidate_ranking_scope", ""))
    ):
        raise StageIMetaTargetExecutionError(f"{side}: authorized meta-selection provenance drift")
    features = tuple(map(str, meta_manifest.get("selected_features", ())))
    contract = tuple(map(str, meta_manifest.get("selected_feature_contract", features)))
    required_handoff = tuple(
        map(
            str,
            meta_manifest.get(
                "required_same_side_base_oof_handoff_features", ()
            ),
        )
    )
    params = meta_manifest.get("best_params")
    if (
        not features or features != contract or len(set(features)) != len(features)
        or required_handoff != STAGE_I_META_BASE_OOF_HANDOFF_FEATURES
        or not set(required_handoff).issubset(features)
        or not isinstance(params, Mapping) or not params
    ):
        raise StageIMetaTargetExecutionError(f"{side}: meta selector lacks one exact selected feature/parameter contract")
    authorized = set(resolve_stage_i_feature_universe(
        CFG, layer="meta", side=side, head="shared_exact_net_residual",
    ))
    if not set(features).issubset(authorized):
        raise StageIMetaTargetExecutionError(f"{side}: selected meta features escape the authorized layer pool")
    ledger, raw_features, base, candidate_handoff = (
        pd.read_parquet(ledger_path), pd.read_parquet(features_path),
        pd.read_parquet(base_oof_path), pd.read_parquet(candidate_handoff_path),
    )
    if not ledger.loc[:, list(IDENTITY)].reset_index(drop=True).equals(raw_features.loc[:, list(IDENTITY)].reset_index(drop=True)):
        raise StageIMetaTargetExecutionError("selector ledger/features identity order drift")
    local_mask = ledger.side_name.astype(str).str.lower().eq(side).to_numpy()
    local_ledger = ledger.loc[local_mask].reset_index(drop=True)
    local_features = raw_features.loc[local_mask].reset_index(drop=True)
    aligned = [*IDENTITY, "side_name", "label_available_ts", "exact_net_bps"]
    missing = sorted(set(aligned + ["r3_p_adverse", "r3_p_weak", "r3_p_clear", "r3_opportunity_score"]).difference(base.columns))
    if missing or not base.loc[:, aligned].reset_index(drop=True).equals(local_ledger.loc[:, aligned].reset_index(drop=True)):
        raise StageIMetaTargetExecutionError(f"{side}: base OOF is absent or not aligned: {missing}")
    probability = base.loc[:, ["r3_p_adverse", "r3_p_weak", "r3_p_clear"]].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    complete = np.isfinite(probability).all(axis=1)
    if np.any(np.isfinite(probability).any(axis=1) & ~complete) or not complete.any():
        raise StageIMetaTargetExecutionError(f"{side}: partial/absent base probability simplex")
    if (probability[complete] < 0).any() or not np.allclose(probability[complete].sum(axis=1), 1.0, atol=1e-5):
        raise StageIMetaTargetExecutionError(f"{side}: invalid base probability simplex")
    expected_handoff = local_ledger.loc[complete, list(IDENTITY)].reset_index(drop=True)
    expected_score = probability[complete, 2] - probability[complete, 0]
    expected_decision = (
        pd.to_datetime(expected_handoff["__ts__"], utc=True, errors="raise")
        + pd.Timedelta(hours=1)
    )
    if (
        "selected_base_candidate" not in candidate_handoff
        or not candidate_handoff.loc[:, list(IDENTITY)].reset_index(drop=True).equals(expected_handoff)
        or not candidate_handoff.side_name.astype(str).str.lower().eq(side).all()
        or not np.allclose(
            pd.to_numeric(
                candidate_handoff.r3_opportunity_score, errors="coerce"
            ).to_numpy(float),
            expected_score,
            atol=1e-6,
        )
        or not pd.to_datetime(
            candidate_handoff.decision_ts, utc=True, errors="coerce"
        ).reset_index(drop=True).equals(expected_decision.reset_index(drop=True))
        or not np.isclose(
            pd.to_numeric(candidate_handoff.base_candidate_fraction, errors="coerce"),
            0.30,
        ).all()
    ):
        raise StageIMetaTargetExecutionError(
            f"{side}: canonical base top-30 candidate handoff audit is invalid"
        )
    expected_selected = int(math.ceil(0.30 * len(candidate_handoff)))
    if int(candidate_handoff.selected_base_candidate.astype(bool).sum()) != expected_selected:
        raise StageIMetaTargetExecutionError(
            f"{side}: base candidate handoff does not contain exact global top-30 support"
        )
    expected_positions = stable_stage_i_topk_positions(
        candidate_handoff.r3_opportunity_score.to_numpy(float),
        candidate_ids=candidate_handoff.candidate_id.to_numpy(object),
        side_names=candidate_handoff.side_name.to_numpy(object),
        decision_timestamps=candidate_handoff.decision_ts,
        signal_timestamps=candidate_handoff["__ts__"],
        symbols=candidate_handoff["__symbol__"].to_numpy(object),
        count=expected_selected,
    )
    expected_mask = np.zeros(len(candidate_handoff), dtype=bool)
    expected_mask[expected_positions] = True
    if not np.array_equal(
        candidate_handoff.selected_base_candidate.astype(bool).to_numpy(),
        expected_mask,
    ):
        raise StageIMetaTargetExecutionError(
            f"{side}: candidate audit is not the canonical score-only global top-30"
        )
    return local_ledger, local_features, base, candidate_handoff, meta_manifest, features, dict(params)


def _prepare_frame(
    ledger: pd.DataFrame, raw: pd.DataFrame, base: pd.DataFrame,
    candidate_handoff: pd.DataFrame, *, side: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    original_population_rows = int(len(ledger))
    valid_target_source = _valid_complete_target_mask(ledger)
    probability = base.loc[:, ["r3_p_adverse", "r3_p_weak", "r3_p_clear"]].to_numpy(np.float32)
    valid = np.isfinite(probability).all(axis=1)
    keep = np.flatnonzero(valid)
    ledger = ledger.iloc[keep].reset_index(drop=True).copy()
    frame = raw.iloc[keep].drop(columns=list(IDENTITY), errors="ignore").reset_index(drop=True).copy()
    probability = probability[keep]
    valid_resolved_target = valid_target_source[keep]
    decision = pd.to_datetime(ledger["__ts__"], utc=True, errors="raise") + pd.Timedelta(hours=1)
    available = pd.to_datetime(ledger.label_available_ts, utc=True, errors="raise")
    if not available.eq(decision + pd.Timedelta(hours=12)).all():
        raise StageIMetaTargetExecutionError("selector labels must be available exactly decision+12h")
    score = probability[:, 2] - probability[:, 0]
    if not valid_resolved_target.any():
        raise StageIMetaTargetExecutionError(
            f"{side}: base OOF population has no valid complete-path targets"
        )
    # Invalid paths remain available for a coverage audit but are never a
    # value-map observation, a residual target, or a source of trust context.
    mapped = np.full(len(ledger), np.nan, dtype=np.float32)
    valid_mapped, audit, _provenance = prequential_same_side_r3_value_map(
        exact_net_bps=ledger.loc[valid_resolved_target, "exact_net_bps"].to_numpy(np.float32),
        decision_timestamps=decision.loc[valid_resolved_target],
        label_available_timestamps=available.loc[valid_resolved_target],
        side=side, score=score[valid_resolved_target], config=PrequentialR3ValueMapConfig(side=side),
    )
    mapped[valid_resolved_target] = np.asarray(valid_mapped, dtype=np.float32)
    frame["r3_p_adverse"], frame["r3_p_weak"], frame["r3_p_clear"] = probability[:, 0], probability[:, 1], probability[:, 2]
    frame["r3_opportunity_score"] = score
    frame["prequential_base_expected_net_bps"] = np.asarray(mapped, dtype=np.float32)
    trust = base_oof_trust_features(probability[valid_resolved_target], audit)
    for column in trust:
        value = np.full(len(frame), np.nan, dtype=np.float32)
        value[valid_resolved_target] = trust[column].to_numpy(np.float32)
        frame[column] = value
    model = pd.concat([
        ledger.loc[:, list(IDENTITY) + ["side_name", "exact_net_bps"]].reset_index(drop=True),
        frame.reset_index(drop=True),
    ], axis=1)
    model["decision_ts"], model["label_available_ts"] = decision, available
    model["candidate_key"] = model.side_name.astype(str) + "::" + model.candidate_id.astype(str)
    selected = candidate_handoff.selected_base_candidate.astype(bool).to_numpy()
    if len(selected) != len(model):
        raise StageIMetaTargetExecutionError("base candidate handoff/model row drift")
    # The frozen top-30 stream is the *training/action* population only.  The
    # 21-day side mapper must see the complete contemporaneous base population
    # so its reference distribution cannot be conditioned on a hindsight
    # candidate filter.  Keep the tags explicit and mutually exclusive.
    model["original_side_population_rows"] = original_population_rows
    model["base_oof_candidate_population_rows"] = int(len(model))
    model["base_top30_candidate_rows"] = int(selected.sum())
    model["valid_resolved_target"] = valid_resolved_target
    model["mapping_reference_eligible"] = valid_resolved_target
    # An invalid row is neither a training/action candidate nor a mapping
    # reference.  Keeping this distinct from ordinary reference-only rows is
    # essential: the latter are scoreable, while the former have no usable
    # realised economic outcome at all.
    model["action_candidate"] = selected & valid_resolved_target
    model["mapping_reference_only"] = valid_resolved_target & ~selected
    if (model.action_candidate & ~model.mapping_reference_eligible).any():
        raise StageIMetaTargetExecutionError("invalid paths cannot enter the action candidate stream")
    if not np.array_equal(
        model.mapping_reference_only.to_numpy(bool),
        (
            model.mapping_reference_eligible.astype(bool)
            & ~model.action_candidate.astype(bool)
        ).to_numpy(bool),
    ):
        raise StageIMetaTargetExecutionError("candidate/reference validity tags are inconsistent")
    handoff_audit = candidate_handoff.copy()
    handoff_audit["enforced_by_meta_target_funnel"] = True
    handoff_audit["selected_model_row"] = selected
    return model, audit, handoff_audit


def _fold_id(frame: pd.DataFrame, *, n_folds: int, min_train_rows: int) -> np.ndarray:
    decision = pd.to_datetime(frame.decision_ts, utc=True)
    available = pd.to_datetime(frame.label_available_ts, utc=True)
    blocks = _validation_blocks(decision, available, n_folds=n_folds, min_train_rows=min_train_rows)
    fold = np.full(len(frame), -1, dtype=np.int32)
    for index, positions in enumerate(blocks):
        positions = np.asarray(positions, dtype=np.int32)
        if (fold[positions] >= 0).any():
            raise StageIMetaTargetExecutionError("strict meta validation blocks overlap")
        fold[positions] = index
    if not (fold >= 0).any():
        raise StageIMetaTargetExecutionError("strict meta runner produced no evaluation support")
    return fold


def _full_population_fold_id(
    frame: pd.DataFrame, *, n_folds: int, min_train_rows: int,
) -> np.ndarray:
    """Use frozen-candidate blocks to train, but score every row in each block.

    The candidate-only block construction preserves all target-specific class
    support and the originally authorised chronology.  Boundaries are then
    projected onto the full same-side population.  Consequently a fold has one
    identical trained model for both action candidates and mapping references.
    """
    required = {"action_candidate", "valid_resolved_target", "mapping_reference_eligible"}
    if missing := sorted(required.difference(frame.columns)):
        raise StageIMetaTargetExecutionError(
            f"full-population frame lacks target-validity tags: {missing}"
        )
    valid = frame.valid_resolved_target.astype(bool).to_numpy()
    if not np.array_equal(valid, frame.mapping_reference_eligible.astype(bool).to_numpy()):
        raise StageIMetaTargetExecutionError(
            "full-population frame has ambiguous mapping-reference validity"
        )
    candidate_positions = np.flatnonzero(
        frame.action_candidate.to_numpy(bool) & valid
    )
    if not len(candidate_positions):
        raise StageIMetaTargetExecutionError("full-population frame has no action candidates")
    candidate_fold = _fold_id(
        frame.iloc[candidate_positions].reset_index(drop=True),
        n_folds=n_folds, min_train_rows=min_train_rows,
    )
    decision = pd.to_datetime(frame.decision_ts, utc=True, errors="raise")
    full_fold = np.full(len(frame), -1, dtype=np.int32)
    values = sorted(set(candidate_fold[candidate_fold >= 0].tolist()))
    starts = [
        pd.to_datetime(
            frame.iloc[candidate_positions[candidate_fold == value]].decision_ts,
            utc=True, errors="raise",
        ).min()
        for value in values
    ]
    if starts != sorted(starts):
        raise StageIMetaTargetExecutionError("candidate strict folds are not chronological")
    for offset, (value, start) in enumerate(zip(values, starts)):
        stop = starts[offset + 1] if offset + 1 < len(starts) else None
        mask = decision.ge(start).to_numpy() & valid
        if stop is not None:
            mask &= decision.lt(stop).to_numpy()
        if np.any(full_fold[mask] >= 0):
            raise StageIMetaTargetExecutionError("full-population strict folds overlap")
        full_fold[mask] = int(value)
    if not np.array_equal(
        full_fold[candidate_positions], candidate_fold,
    ):
        raise StageIMetaTargetExecutionError(
            "full-population fold projection changed frozen candidate support"
        )
    if not np.any(full_fold >= 0):
        raise StageIMetaTargetExecutionError("full-population strict runner produced no scoring support")
    return full_fold


def make_lgbm_predictor(params: Mapping[str, Any]) -> Predictor:
    frozen = dict(params)

    def predict(train_x: pd.DataFrame, target: np.ndarray, weight: np.ndarray, valid_x: pd.DataFrame, spec: MetaTargetSpec) -> np.ndarray:
        if spec.family in {
            "reliability", "overestimate_risk", "ordinal_residual",
            "quantile_ordinal_residual",
        }:
            import lightgbm as lgb
            local = dict(frozen)
            multiclass = spec.family in {
                "ordinal_residual", "quantile_ordinal_residual"
            }
            local["objective"] = "multiclass" if multiclass else "binary"
            if multiclass:
                local["num_class"] = (
                    4 if spec.family == "ordinal_residual" else 3
                )
            else:
                local.pop("num_class", None)
            model = lgb.LGBMClassifier(**local).fit(train_x, target, sample_weight=weight)
            probability = np.asarray(model.predict_proba(valid_x), dtype=np.float64)
            expected = (
                4 if spec.family == "ordinal_residual"
                else 3 if spec.family == "quantile_ordinal_residual"
                else 2
            )
            if probability.shape != (len(valid_x), expected):
                raise StageIMetaTargetExecutionError(f"{spec.arm_id}: classifier did not emit the full class simplex")
            return probability
        from .lgbm_pipeline import _fit_lgbm_model
        local = dict(frozen)
        local["objective"] = "huber"
        model = _fit_lgbm_model(train_x, target, weight, classifier=False, params=local, objective_mode="stage_i_residual")
        return np.asarray(model.predict(valid_x), dtype=np.float64)
    return predict


def _arm_frame(arm: MetaOOFArm, evaluation: pd.DataFrame) -> pd.DataFrame:
    out = evaluation.loc[:, ["candidate_key", *IDENTITY, "side_name", "decision_ts", "label_available_ts", "exact_net_bps", "r3_opportunity_score", "prequential_base_expected_net_bps"]].copy()
    for column in (
        "original_side_population_rows", "base_oof_candidate_population_rows",
        "base_top30_candidate_rows", "action_candidate", "mapping_reference_only",
        "valid_resolved_target", "mapping_reference_eligible",
    ):
        if column in evaluation:
            out[column] = evaluation[column].to_numpy()
    out["arm_id"], out["target_family"] = arm.arm_id, arm.target_family
    out["score"], out["model_action_admitted"], out["fold_id"] = arm.score, arm.action_admitted, arm.fold_id
    # Keep the legacy name as an arm-local model gate only.  It must never be
    # interpreted as permission for reference-only rows to be selected.
    out["action_admitted"] = arm.action_admitted
    out["target_semantic_valid"] = bool(arm.semantic_valid)
    if arm.target is not None:
        out["target"] = np.asarray(arm.target).reshape(-1)
    if arm.prediction is not None:
        prediction = np.asarray(arm.prediction)
        if prediction.ndim == 1:
            out["prediction"] = prediction
        else:
            for index in range(prediction.shape[1]):
                out[f"prediction_class_{index}"] = prediction[:, index]
            if arm.target_family == "quantile_ordinal_residual" and prediction.shape[1] == 3:
                out["probability_lower_residual_tercile"] = prediction[:, 0]
                out["probability_middle_residual_tercile"] = prediction[:, 1]
                out["probability_upper_residual_tercile"] = prediction[:, 2]
    if arm.prior_prediction is not None:
        prior = np.asarray(arm.prior_prediction)
        if prior.ndim != 2:
            raise StageIMetaTargetExecutionError(
                f"{arm.arm_id}: prior classifier prediction must be a matrix"
            )
        for index in range(prior.shape[1]):
            out[f"prior_prediction_class_{index}"] = prior[:, index]
        if arm.target_family == "quantile_ordinal_residual" and prior.shape[1] == 3:
            out["prior_probability_lower_residual_tercile"] = prior[:, 0]
            out["prior_probability_middle_residual_tercile"] = prior[:, 1]
            out["prior_probability_upper_residual_tercile"] = prior[:, 2]
    return out


def _run_candidate_trained_full_population_arm(
    frame: pd.DataFrame,
    spec: MetaTargetSpec,
    *,
    feature_columns: Sequence[str],
    fold_id: Sequence[int],
    predictor: Predictor,
) -> _FullPopulationArmResult:
    """Strict folds: candidate-only fitting, full-population prediction."""
    features = tuple(dict.fromkeys(map(str, feature_columns)))
    if not features or any(column not in frame for column in features):
        raise StageIMetaTargetExecutionError("strict full-population arm has an incomplete feature matrix")
    required = {"action_candidate", "valid_resolved_target", "mapping_reference_eligible"}
    missing = sorted(required.difference(frame.columns))
    if frame.empty or missing:
        raise StageIMetaTargetExecutionError(
            "strict full-population arm lacks target-validity candidate tags"
            + (f": {missing}" if missing else "")
        )
    sides = frame.side_name.astype(str).str.lower().unique()
    if len(sides) != 1:
        raise StageIMetaTargetExecutionError("strict full-population arm must be side-local")
    side = str(sides[0])
    fold = np.asarray(fold_id, dtype=np.int32).reshape(-1)
    if len(fold) != len(frame) or not np.any(fold >= 0):
        raise StageIMetaTargetExecutionError("strict full-population fold IDs are invalid")
    decision = pd.to_datetime(frame.decision_ts, utc=True, errors="raise")
    available = pd.to_datetime(frame.label_available_ts, utc=True, errors="raise")
    if not available.gt(decision).all():
        raise StageIMetaTargetExecutionError("strict full-population labels must resolve after decisions")
    full_positions = np.flatnonzero(fold >= 0)
    action_positions = full_positions[frame.action_candidate.to_numpy(bool)[full_positions]]
    if not len(action_positions):
        raise StageIMetaTargetExecutionError("strict full-population folds contain no action candidates")
    values = sorted(set(fold[full_positions].tolist()))
    starts = [decision.iloc[np.flatnonzero(fold == value)].min() for value in values]
    if starts != sorted(starts):
        raise StageIMetaTargetExecutionError("full-population folds are not chronological")
    score = np.full(len(frame), np.nan, dtype=np.float64)
    admitted = np.zeros(len(frame), dtype=bool)
    target_oof = np.full(len(frame), np.nan, dtype=np.float64)
    if spec.family in {"ordinal_residual", "quantile_ordinal_residual"}:
        classes = 4 if spec.family == "ordinal_residual" else 3
        prediction_oof: np.ndarray = np.full((len(frame), classes), np.nan, dtype=np.float64)
        prior_prediction_oof: np.ndarray | None = (
            np.full((len(frame), classes), np.nan, dtype=np.float64)
            if spec.family == "quantile_ordinal_residual" else None
        )
    else:
        prediction_oof = np.full(len(frame), np.nan, dtype=np.float64)
        prior_prediction_oof = None
    provenance: list[dict[str, Any]] = []
    valid = frame.valid_resolved_target.astype(bool).to_numpy()
    if not np.array_equal(valid, frame.mapping_reference_eligible.astype(bool).to_numpy()):
        raise StageIMetaTargetExecutionError("full-population arm has invalid mapping reference rows")
    candidate = frame.action_candidate.to_numpy(bool) & valid
    for value, validation_start in zip(values, starts):
        validation_idx = np.flatnonzero(fold == value)
        train_idx = np.flatnonzero(candidate & available.lt(validation_start).to_numpy())
        if not len(train_idx) or not available.iloc[train_idx].lt(validation_start).all():
            raise StageIMetaTargetExecutionError(f"fold {value} has no candidate-only strict prior support")
        train, validation = frame.iloc[train_idx], frame.iloc[validation_idx]
        target_fit = fit_meta_target(train, spec, side=side, fit_before_utc=validation_start)
        prediction = np.asarray(predictor(
            train.loc[:, list(features)], target_fit.target, target_fit.sample_weight,
            validation.loc[:, list(features)], spec,
        ))
        fold_score, fold_admitted = reconstruct_meta_action(validation, target_fit, prediction)
        if len(fold_score) != len(validation_idx):
            raise StageIMetaTargetExecutionError(f"fold {value} full-population prediction is misaligned")
        score[validation_idx], admitted[validation_idx] = fold_score, fold_admitted
        target_oof[validation_idx] = _evaluation_target(validation, spec, target_fit)
        if prediction_oof.ndim == 2:
            if prediction.shape != (len(validation_idx), prediction_oof.shape[1]):
                raise StageIMetaTargetExecutionError(f"fold {value} full-population classifier simplex is invalid")
            prediction_oof[validation_idx] = prediction
            if prior_prediction_oof is not None:
                prior = np.asarray(target_fit.class_support, dtype=np.float64)
                prior_prediction_oof[validation_idx] = prior / prior.sum()
        else:
            vector = prediction[:, 1] if prediction.ndim == 2 and prediction.shape[1] == 2 else prediction.reshape(-1)
            if len(vector) != len(validation_idx):
                raise StageIMetaTargetExecutionError(f"fold {value} full-population prediction is misaligned")
            prediction_oof[validation_idx] = vector
        thresholds = target_fit.residual_thresholds_bps
        provenance.append({
            "arm_id": spec.arm_id, "target_family": spec.family, "side": side,
            "fold_id": int(value), "train_rows": int(len(train_idx)),
            "train_action_candidate_rows": int(len(train_idx)),
            "validation_rows": int(len(validation_idx)),
            "validation_full_population_rows": int(len(validation_idx)),
            "validation_action_candidate_rows": int(candidate[validation_idx].sum()),
            "full_population_scored": True,
            "validation_start_utc": validation_start.isoformat(),
            "validation_end_utc": decision.iloc[validation_idx].max().isoformat(),
            "train_max_label_available_utc": available.iloc[train_idx].max().isoformat(),
            "strict_prior_resolved": True,
            "residual_q33_bps": thresholds[0] if thresholds else np.nan,
            "residual_q67_bps": thresholds[1] if thresholds else np.nan,
            "class_0_support": target_fit.class_support[0] if target_fit.class_support else np.nan,
            "class_1_support": target_fit.class_support[1] if len(target_fit.class_support) > 1 else np.nan,
            "class_2_support": target_fit.class_support[2] if len(target_fit.class_support) > 2 else np.nan,
            "class_0_training_prior": target_fit.class_support[0] / target_fit.fit_rows if target_fit.class_support else np.nan,
            "class_1_training_prior": target_fit.class_support[1] / target_fit.fit_rows if len(target_fit.class_support) > 1 else np.nan,
            "class_2_training_prior": target_fit.class_support[2] / target_fit.fit_rows if len(target_fit.class_support) > 2 else np.nan,
            "class_0_residual_location_bps": target_fit.class_payoff_bps[0] if len(target_fit.class_payoff_bps) > 0 else np.nan,
            "class_1_residual_location_bps": target_fit.class_payoff_bps[1] if len(target_fit.class_payoff_bps) > 1 else np.nan,
            "class_2_residual_location_bps": target_fit.class_payoff_bps[2] if len(target_fit.class_payoff_bps) > 2 else np.nan,
            "class_0_residual_median_bps": target_fit.class_median_bps[0] if len(target_fit.class_median_bps) > 0 else np.nan,
            "class_1_residual_median_bps": target_fit.class_median_bps[1] if len(target_fit.class_median_bps) > 1 else np.nan,
            "class_2_residual_median_bps": target_fit.class_median_bps[2] if len(target_fit.class_median_bps) > 2 else np.nan,
            "class_0_location_uncertainty_bps": target_fit.class_location_uncertainty_bps[0] if len(target_fit.class_location_uncertainty_bps) > 0 else np.nan,
            "class_1_location_uncertainty_bps": target_fit.class_location_uncertainty_bps[1] if len(target_fit.class_location_uncertainty_bps) > 1 else np.nan,
            "class_2_location_uncertainty_bps": target_fit.class_location_uncertainty_bps[2] if len(target_fit.class_location_uncertainty_bps) > 2 else np.nan,
            "residual_winsor_lower_bps": target_fit.residual_winsor_bounds_bps[0] if target_fit.residual_winsor_bounds_bps else np.nan,
            "residual_winsor_upper_bps": target_fit.residual_winsor_bounds_bps[1] if target_fit.residual_winsor_bounds_bps else np.nan,
            "class_location_shrinkage_support": float(spec.shrinkage_support),
            "class_location_method": target_fit.class_location_method,
            "quantile_method": target_fit.quantile_method,
            "zero_in_middle_tercile": bool(thresholds[0] < 0.0 <= thresholds[1]) if len(thresholds) == 2 else None,
            "fold_semantic_valid": bool(thresholds[0] < 0.0 <= thresholds[1]) if len(thresholds) == 2 else True,
            "class_0_name": "lower_residual_tercile", "class_1_name": "middle_residual_tercile",
            "class_2_name": "upper_residual_tercile",
            "economic_class_interpretation": "overestimate|approximately_right|underestimate" if len(thresholds) == 2 and thresholds[0] < 0.0 <= thresholds[1] else "not_authorized",
            "correction_bound_bps": spec.residual_clip_bps if spec.family == "quantile_ordinal_residual" else np.nan,
        })
    if not np.isfinite(score[full_positions]).all() or not np.isfinite(prediction_oof[full_positions]).all():
        raise StageIMetaTargetExecutionError("strict full-population arm emitted incomplete predictions")
    if prior_prediction_oof is not None and not np.isfinite(prior_prediction_oof[full_positions]).all():
        raise StageIMetaTargetExecutionError("strict full-population arm emitted incomplete priors")
    provenance_frame = pd.DataFrame(provenance)
    semantic_valid = bool(provenance_frame.fold_semantic_valid.fillna(False).all() if spec.family == "quantile_ordinal_residual" else True)
    full_arm = MetaOOFArm(
        spec.arm_id, score[full_positions], admitted[full_positions], fold[full_positions],
        spec.family, target_oof[full_positions], prediction_oof[full_positions],
        prior_prediction_oof[full_positions] if prior_prediction_oof is not None else None,
        semantic_valid,
    )
    action_mask = candidate[full_positions]
    action_arm = MetaOOFArm(
        spec.arm_id, full_arm.score[action_mask], full_arm.action_admitted[action_mask],
        full_arm.fold_id[action_mask], spec.family, full_arm.target[action_mask] if full_arm.target is not None else None,
        full_arm.prediction[action_mask] if full_arm.prediction is not None else None,
        full_arm.prior_prediction[action_mask] if full_arm.prior_prediction is not None else None,
        semantic_valid,
    )
    return _FullPopulationArmResult(full_arm, action_arm, full_positions, action_positions, provenance_frame)


def _load_arm(root: Path, spec: MetaTargetSpec, request_sha: str) -> tuple[MetaOOFArm, pd.DataFrame]:
    manifest = _json(root / "manifest.json")
    predictions, full_predictions, provenance = (
        root / "oof_predictions.parquet", root / "full_oof_reference_predictions.parquet",
        root / "fold_provenance.parquet",
    )
    if (
        manifest.get("schema") != SCHEMA or manifest.get("status") != "complete"
        or manifest.get("arm_id") != spec.arm_id or manifest.get("request_sha256") != request_sha
        or not predictions.is_file() or not full_predictions.is_file() or not provenance.is_file()
        or manifest.get("oof_predictions_sha256") != file_sha256(predictions)
        or manifest.get("full_oof_reference_predictions_sha256") != file_sha256(full_predictions)
        or manifest.get("fold_provenance_sha256") != file_sha256(provenance)
    ):
        raise StageIMetaTargetExecutionError(f"{spec.arm_id}: checkpoint drift")
    frame = pd.read_parquet(predictions)
    prediction_cols = sorted((name for name in frame if name.startswith("prediction_class_")), key=lambda name: int(name.rsplit("_", 1)[1]))
    prior_cols = sorted((name for name in frame if name.startswith("prior_prediction_class_")), key=lambda name: int(name.rsplit("_", 1)[1]))
    prediction = frame[prediction_cols].to_numpy(float) if prediction_cols else frame["prediction"].to_numpy(float)
    prior_prediction = frame[prior_cols].to_numpy(float) if prior_cols else None
    arm = MetaOOFArm(
        spec.arm_id, frame.score.to_numpy(float), frame.action_admitted.to_numpy(bool), frame.fold_id.to_numpy(np.int32),
        spec.family, frame.target.to_numpy(float), prediction, prior_prediction,
        bool(frame.target_semantic_valid.astype(bool).all()),
    )
    return arm, pd.read_parquet(provenance)


def _require_exact_action_subset(
    full: pd.DataFrame, action: pd.DataFrame, *, context: str,
) -> None:
    required = {"candidate_key", "action_candidate"}
    if not required.issubset(full.columns) or "candidate_key" not in action:
        raise StageIMetaTargetExecutionError(f"{context}: missing action-subset identity/tags")
    expected_columns = ["candidate_key"]
    if "fold_id" in full and "fold_id" in action:
        expected_columns.append("fold_id")
    elif "meta_fold_id" in full and "meta_fold_id" in action:
        expected_columns.append("meta_fold_id")
    expected = full.loc[full.action_candidate.astype(bool), expected_columns].reset_index(drop=True)
    observed = action.loc[:, expected_columns].reset_index(drop=True)
    if not expected.equals(observed):
        raise StageIMetaTargetExecutionError(f"{context}: action ledger is not the exact full-reference subset")


def run_side_meta_target_funnel(
    *, selector_dir: str | Path, base_selection_dir: str | Path, meta_selection_dir: str | Path,
    output_dir: str | Path, side: str, n_validation_folds: int = 4,
    min_train_rows: int = 500, resume: bool = False,
    predictor: Predictor | None = None, specs: Sequence[MetaTargetSpec] | None = None,
) -> dict[str, Any]:
    side = str(side).lower()
    if side not in {"long", "short"}:
        raise StageIMetaTargetExecutionError("side must be long or short")
    selector, base_dir, meta_dir, root = map(Path, (selector_dir, base_selection_dir, meta_selection_dir, output_dir))
    ledger, raw, base, candidate_handoff, meta_manifest, features, params = _validate_inputs(selector_dir=selector, base_dir=base_dir, meta_dir=meta_dir, side=side)
    frame, map_audit, enforced_handoff_audit = _prepare_frame(
        ledger, raw, base, candidate_handoff, side=side
    )
    missing = sorted(set(features).difference(frame.columns))
    if missing:
        raise StageIMetaTargetExecutionError(f"{side}: selected meta model matrix is incomplete: {missing}")
    fold = _full_population_fold_id(
        frame, n_folds=int(n_validation_folds), min_train_rows=int(min_train_rows)
    )
    full_evaluation_positions = np.flatnonzero(fold >= 0)
    action_evaluation_positions = full_evaluation_positions[
        frame.action_candidate.to_numpy(bool)[full_evaluation_positions]
    ]
    full_evaluation = frame.iloc[full_evaluation_positions].reset_index(drop=True)
    evaluation = frame.iloc[action_evaluation_positions].reset_index(drop=True)
    if not len(evaluation):
        raise StageIMetaTargetExecutionError("strict full-population folds contain no action evaluation support")
    ordered_specs = tuple(specs or default_meta_target_specs())
    if tuple(spec.arm_id for spec in ordered_specs) != tuple(dict.fromkeys(spec.arm_id for spec in ordered_specs)):
        raise StageIMetaTargetExecutionError("predeclared target arm IDs must be unique and ordered")
    arm_features = {
        spec.arm_id: _classifier_features_for_spec(features, spec)
        for spec in ordered_specs
    }
    request = {
        "schema": SCHEMA, "funnel_schema": FUNNEL_SCHEMA, "side": side,
        "selector_manifest_sha256": file_sha256(selector / "manifest.json"),
        "selector_feature_contract_sha256": file_sha256(selector / "selector_feature_contract.json"),
        "base_manifest_sha256": file_sha256(base_dir / side / "manifest.json"),
        "base_oof_sha256": file_sha256(base_dir / side / "selector_base_oof.parquet"),
        "meta_manifest_sha256": file_sha256(meta_dir / side / "manifest.json"),
        "base_candidate_handoff_audit_sha256": file_sha256(
            meta_dir / side / "base_candidate_handoff_audit.parquet"
        ),
        "base_candidate_fraction": 0.30,
        "base_candidate_scope": (
            "side_local_global_over_strict_oof_development_rows; never_per_timestamp"
        ),
        "selected_features": features, "params": params,
        "classifier_features_by_arm": {
            arm_id: list(columns) for arm_id, columns in arm_features.items()
        },
        "T3Q_anchor_semantics": (
            "conversion_head: hidden causal side-local mapped_expected_net_bps anchor; "
            "classifier inputs exclude mapped/prequential/converted values; ranking is "
            "anchor plus prior-centered bounded residual correction; raw-base economic "
            "control retained"
        ),
        "feature_contract_disposition": (
            "fixed_feature_target_isolation_diagnostic; selected originally for "
            "shared_exact_net_residual; if quantile_ordinal_residual advances, rerun "
            "per-side per-layer per-head MDA/feature-selection/HPO before promotion"
        ),
        "target_validity_contract": {
            "required_columns": list(TARGET_VALIDITY_COLUMNS),
            "invalid_paths": "excluded_from_anchor_map_fit_target_and_economic_evaluation",
        },
        "specs": [spec.__dict__ for spec in ordered_specs],
        "n_validation_folds": int(n_validation_folds), "min_train_rows": int(min_train_rows),
    }
    request_sha = _canonical_sha(request)
    final_manifest = root / "manifest.json"
    if final_manifest.is_file() and resume:
        existing = _json(final_manifest)
        if existing.get("request_sha256") != request_sha or existing.get("status") != "complete":
            raise StageIMetaTargetExecutionError("completed meta-target funnel resume contract drift")
        for name, field in (
            ("metrics.parquet", "metrics_sha256"),
            ("fold_provenance.parquet", "fold_provenance_sha256"),
            ("prequential_value_map_audit.parquet", "prequential_value_map_audit_sha256"),
            ("evaluation_ledger.parquet", "evaluation_ledger_sha256"),
            ("full_oof_reference_ledger.parquet", "full_oof_reference_ledger_sha256"),
            ("base_candidate_handoff_audit.parquet", "base_candidate_handoff_audit_output_sha256"),
        ):
            path = root / name
            if not path.is_file() or existing.get(field) != file_sha256(path):
                raise StageIMetaTargetExecutionError("completed meta-target funnel checksum drift")
        for spec in ordered_specs:
            arm_root = root / "arms" / spec.arm_id
            _load_arm(arm_root, spec, request_sha)
            _require_exact_action_subset(
                pd.read_parquet(arm_root / "full_oof_reference_predictions.parquet"),
                pd.read_parquet(arm_root / "oof_predictions.parquet"),
                context=f"{spec.arm_id}: resumed arm",
            )
        _require_exact_action_subset(
            pd.read_parquet(root / "full_oof_reference_ledger.parquet"),
            pd.read_parquet(root / "evaluation_ledger.parquet"),
            context="resumed full reference ledger",
        )
        return {**existing, "restart_status": "reused_verified_complete"}
    if root.exists() and not resume:
        raise FileExistsError(f"meta-target output exists without --resume: {root}")
    root.mkdir(parents=True, exist_ok=True)
    arms_root = root / "arms"
    arms_root.mkdir(exist_ok=True)
    model_predictor = predictor or make_lgbm_predictor(params)
    learned: list[MetaOOFArm] = []
    provenance_frames: list[pd.DataFrame] = []
    for sequence, spec in enumerate(ordered_specs):
        arm_root = arms_root / spec.arm_id
        if arm_root.is_dir() and resume:
            arm, provenance = _load_arm(arm_root, spec, request_sha)
        else:
            if arm_root.exists():
                raise StageIMetaTargetExecutionError(f"{spec.arm_id}: incomplete checkpoint exists; resume cannot trust it")
            result = _run_candidate_trained_full_population_arm(
                frame, spec, feature_columns=arm_features[spec.arm_id],
                fold_id=fold, predictor=model_predictor,
            )
            if not np.array_equal(result.full_evaluation_positions, full_evaluation_positions):
                raise StageIMetaTargetExecutionError(f"{spec.arm_id}: full reference evaluation support drift")
            if not np.array_equal(result.action_evaluation_positions, action_evaluation_positions):
                raise StageIMetaTargetExecutionError(f"{spec.arm_id}: action evaluation support drift")
            staging = Path(tempfile.mkdtemp(prefix=f".{spec.arm_id}-", dir=arms_root))
            try:
                predictions = staging / "oof_predictions.parquet"
                full_predictions = staging / "full_oof_reference_predictions.parquet"
                provenance_path = staging / "fold_provenance.parquet"
                action_prediction_frame = _arm_frame(result.action_arm, evaluation)
                full_prediction_frame = _arm_frame(result.full_arm, full_evaluation)
                expected_action = full_prediction_frame.loc[
                    full_prediction_frame.action_candidate.astype(bool)
                ].reset_index(drop=True)
                if not expected_action.loc[:, ["candidate_key", "fold_id"]].equals(
                    action_prediction_frame.loc[:, ["candidate_key", "fold_id"]].reset_index(drop=True)
                ):
                    raise StageIMetaTargetExecutionError(f"{spec.arm_id}: action ledger is not the exact full-reference subset")
                action_prediction_frame.to_parquet(predictions, index=False, compression="zstd")
                full_prediction_frame.to_parquet(full_predictions, index=False, compression="zstd")
                result.fold_provenance.to_parquet(provenance_path, index=False, compression="zstd")
                payload = {
                    "schema": SCHEMA, "status": "complete", "side": side,
                    "arm_id": spec.arm_id, "target_family": spec.family,
                    "sequence": sequence, "request_sha256": request_sha,
                    "rows": int(len(evaluation)), "full_oof_reference_rows": int(len(full_evaluation)),
                    "selected_features": list(arm_features[spec.arm_id]),
                    "classifier_input_contract": (
                        "direct_same_side_raw_base_simplex_trust_regime_context_only"
                        if spec.family == "quantile_ordinal_residual"
                        else "frozen_meta_selector_contract"
                    ),
                    "oof_predictions_sha256": file_sha256(predictions),
                    "full_oof_reference_predictions_sha256": file_sha256(full_predictions),
                    "fold_provenance_sha256": file_sha256(provenance_path),
                }
                (staging / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
                os.replace(staging, arm_root)
            except Exception:
                shutil.rmtree(staging, ignore_errors=True)
                raise
            arm, provenance = result.action_arm, result.fold_provenance
        learned.append(arm)
        provenance_frames.append(provenance.assign(sequence=sequence))
    controls = list(mandatory_control_arms(evaluation, fold[action_evaluation_positions]))
    huber = next((arm for arm in learned if arm.target_family == "huber_residual"), None)
    if huber is None or huber.prediction is None:
        raise StageIMetaTargetExecutionError("predeclared funnel omitted the current Huber control")
    huber_control = current_huber_control_arm(
        evaluation, fold[action_evaluation_positions], np.asarray(huber.prediction).reshape(-1)
    )
    tercile = next(
        (arm for arm in learned if arm.target_family == "quantile_ordinal_residual"),
        None,
    )
    prior_control = (
        quantile_prior_conversion_control_arm(
            evaluation, fold[action_evaluation_positions], tercile
        )
        if tercile is not None else None
    )
    comparison = [
        *controls, huber_control, *([prior_control] if prior_control is not None else []),
        *(arm for arm in learned if arm.arm_id != "C3_current_map_huber"),
    ]
    metrics = evaluate_meta_oof_arms(evaluation, comparison)
    decision = dict(select_meta_arm_with_noop_gate(metrics))
    tercile_semantic_gate = bool(
        tercile.semantic_valid if tercile is not None else True
    )
    metrics.to_parquet(root / "metrics.parquet", index=False, compression="zstd")
    pd.concat(provenance_frames, ignore_index=True).to_parquet(root / "fold_provenance.parquet", index=False, compression="zstd")
    map_audit.to_parquet(root / "prequential_value_map_audit.parquet", index=False, compression="zstd")
    enforced_handoff_audit.to_parquet(
        root / "base_candidate_handoff_audit.parquet", index=False, compression="zstd"
    )
    full_ledger = full_evaluation.assign(
        meta_fold_id=fold[full_evaluation_positions]
    )
    action_ledger = full_ledger.loc[
        full_ledger.action_candidate.astype(bool)
    ].reset_index(drop=True)
    full_ledger.to_parquet(
        root / "full_oof_reference_ledger.parquet", index=False, compression="zstd"
    )
    action_ledger.to_parquet(
        root / "evaluation_ledger.parquet", index=False, compression="zstd"
    )
    _require_exact_action_subset(
        full_ledger, action_ledger, context="selected action ledger"
    )
    for spec in ordered_specs:
        arm_root = arms_root / spec.arm_id
        arm_metrics = arm_root / "metrics.parquet"
        metrics.loc[metrics.arm_id.eq(spec.arm_id)].to_parquet(arm_metrics, index=False, compression="zstd")
        arm_manifest = _json(arm_root / "manifest.json")
        arm_manifest["metrics_sha256"] = file_sha256(arm_metrics)
        (arm_root / "manifest.json").write_text(
            json.dumps(arm_manifest, indent=2, sort_keys=True) + "\n"
        )
    manifest = {
        **request, "status": "complete", "request_sha256": request_sha,
        "evaluation_rows": int(len(evaluation)),
        "full_oof_reference_rows": int(len(full_evaluation)),
        "burnin_rows": int((fold < 0).sum()),
        "arm_order": [spec.arm_id for spec in ordered_specs], "decision": decision,
        "T3Q_promotion_semantic_gate": {
            "required": "residual_q33_bps < 0 <= residual_q67_bps in every OOF fold",
            "passed": tercile_semantic_gate,
            "class_artifact_names": ["lower", "middle", "upper"],
            "economic_interpretation_allowed": (
                ["overestimate", "approximately_right", "underestimate"]
                if tercile_semantic_gate else []
            ),
        },
        "metrics_sha256": file_sha256(root / "metrics.parquet"),
        "fold_provenance_sha256": file_sha256(root / "fold_provenance.parquet"),
        "prequential_value_map_audit_sha256": file_sha256(root / "prequential_value_map_audit.parquet"),
        "base_candidate_handoff_audit_output_sha256": file_sha256(
            root / "base_candidate_handoff_audit.parquet"
        ),
        "evaluation_ledger_sha256": file_sha256(root / "evaluation_ledger.parquet"),
        "full_oof_reference_ledger_sha256": file_sha256(
            root / "full_oof_reference_ledger.parquet"
        ),
        "admission_reference_status": "available_full_population_strict_oof",
        "admission_reference_contract": (
            "candidate-only prior-resolved training; every contemporaneous "
            "same-side strict held-out row scored by the identical fold model; "
            "action ledger is the identity subset where action_candidate=true"
        ),
        "scope": "side_local_diagnostic_until_two_side_common_bps_mapping",
    }
    final_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    return manifest


def _side_arm_frames(root: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    manifest = _json(root / "manifest.json")
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "complete":
        raise StageIMetaTargetExecutionError(f"incomplete side-local meta-target artifact: {root}")
    ledger_path = root / "evaluation_ledger.parquet"
    full_ledger_path = root / "full_oof_reference_ledger.parquet"
    if (
        manifest.get("admission_reference_status")
        != "available_full_population_strict_oof"
        or not full_ledger_path.is_file()
        or manifest.get("full_oof_reference_ledger_sha256") != file_sha256(full_ledger_path)
    ):
        raise StageIMetaTargetExecutionError(
            f"ADMISSION_REFERENCE_INSUFFICIENT: missing or unverified full OOF reference ledger: {root}"
        )
    if not ledger_path.is_file() or manifest.get("evaluation_ledger_sha256") != file_sha256(ledger_path):
        raise StageIMetaTargetExecutionError(f"side-local evaluation ledger checksum drift: {root}")
    ledger = pd.read_parquet(full_ledger_path)
    action_ledger = pd.read_parquet(ledger_path)
    required_tags = {
        "action_candidate", "mapping_reference_only", "mapping_reference_eligible",
        "valid_resolved_target", "meta_fold_id",
    }
    if not required_tags.issubset(ledger.columns):
        raise StageIMetaTargetExecutionError("ADMISSION_REFERENCE_INSUFFICIENT: full ledger lacks population tags")
    candidate = ledger.action_candidate.astype(bool).to_numpy()
    reference_only = ledger.mapping_reference_only.astype(bool).to_numpy()
    valid_reference = ledger.mapping_reference_eligible.astype(bool).to_numpy()
    if (
        not np.array_equal(valid_reference, ledger.valid_resolved_target.astype(bool).to_numpy())
        or np.any(candidate & ~valid_reference)
        or not np.array_equal(reference_only, valid_reference & ~candidate)
    ):
        raise StageIMetaTargetExecutionError(
            "ADMISSION_REFERENCE_INSUFFICIENT: full ledger target-validity tags are ambiguous"
        )
    try:
        _require_exact_action_subset(
            ledger, action_ledger, context="full OOF reference ledger"
        )
    except StageIMetaTargetExecutionError as error:
        raise StageIMetaTargetExecutionError(
            f"ADMISSION_REFERENCE_INSUFFICIENT: {error}"
        ) from error
    fold = ledger.meta_fold_id.to_numpy(np.int32)
    frames: dict[str, pd.DataFrame] = {}
    controls = {
        "C0_raw_base_exact_noop": ledger.r3_opportunity_score.to_numpy(float),
        "C1_causal_map_only": ledger.prequential_base_expected_net_bps.to_numpy(float),
        "C2_raw_base_bounded_zero": ledger.r3_opportunity_score.to_numpy(float),
    }
    if any(
        str(item.get("family")) == "quantile_ordinal_residual"
        for item in manifest.get("specs", [])
    ):
        controls["C4_T3Q_fold_prior_conversion"] = (
            ledger.prequential_base_expected_net_bps.to_numpy(float)
        )
    for arm_id, score in controls.items():
        frames[arm_id] = pd.DataFrame({
            "candidate_key": ledger.candidate_key.astype(str), "side_name": ledger.side_name.astype(str),
            "__symbol__": ledger["__symbol__"].astype(str),
            "decision_ts": ledger.decision_ts, "label_available_ts": ledger.label_available_ts,
            "exact_net_bps": ledger.exact_net_bps, "fold_id": fold,
            "original_side_population_rows": ledger.original_side_population_rows,
            "action_candidate": ledger.action_candidate.astype(bool),
            "mapping_reference_only": ledger.mapping_reference_only.astype(bool),
            "mapping_reference_eligible": ledger.mapping_reference_eligible.astype(bool),
            "valid_resolved_target": ledger.valid_resolved_target.astype(bool),
            "raw_arm_score": score, "model_action_admitted": True,
        })
    for item in manifest["specs"]:
        spec = MetaTargetSpec(**dict(item))
        arm_root = root / "arms" / spec.arm_id
        arm_manifest = _json(arm_root / "manifest.json")
        path = arm_root / "full_oof_reference_predictions.parquet"
        if (
            not path.is_file()
            or arm_manifest.get("full_oof_reference_predictions_sha256") != file_sha256(path)
        ):
            raise StageIMetaTargetExecutionError(
                f"ADMISSION_REFERENCE_INSUFFICIENT: {spec.arm_id} lacks verified full OOF scores"
            )
        raw = pd.read_parquet(path)
        if not raw.candidate_key.astype(str).reset_index(drop=True).equals(ledger.candidate_key.astype(str).reset_index(drop=True)):
            raise StageIMetaTargetExecutionError(f"ADMISSION_REFERENCE_INSUFFICIENT: {spec.arm_id} full OOF identity drift")
        if not np.array_equal(raw.action_candidate.astype(bool).to_numpy(), candidate):
            raise StageIMetaTargetExecutionError(f"ADMISSION_REFERENCE_INSUFFICIENT: {spec.arm_id} action tags drift")
        action_path = arm_root / "oof_predictions.parquet"
        if (
            not action_path.is_file()
            or arm_manifest.get("oof_predictions_sha256") != file_sha256(action_path)
        ):
            raise StageIMetaTargetExecutionError(
                f"ADMISSION_REFERENCE_INSUFFICIENT: {spec.arm_id} action OOF subset/hash drift"
            )
        try:
            _require_exact_action_subset(
                raw, pd.read_parquet(action_path), context=f"{spec.arm_id} action OOF"
            )
        except StageIMetaTargetExecutionError as error:
            raise StageIMetaTargetExecutionError(
                f"ADMISSION_REFERENCE_INSUFFICIENT: {error}"
            ) from error
        frames[spec.arm_id] = pd.DataFrame({
            "candidate_key": raw.candidate_key.astype(str), "side_name": raw.side_name.astype(str),
            "__symbol__": raw["__symbol__"].astype(str),
            "decision_ts": raw.decision_ts, "label_available_ts": raw.label_available_ts,
            "exact_net_bps": raw.exact_net_bps, "fold_id": raw.fold_id.astype(np.int32),
            "original_side_population_rows": ledger.original_side_population_rows,
            "action_candidate": raw.action_candidate.astype(bool),
            "mapping_reference_only": raw.mapping_reference_only.astype(bool),
            "mapping_reference_eligible": raw.mapping_reference_eligible.astype(bool),
            "valid_resolved_target": raw.valid_resolved_target.astype(bool),
            "raw_arm_score": raw.score.astype(float),
            "model_action_admitted": raw.model_action_admitted.astype(bool),
        })
        if spec.family == "huber_residual" and spec.arm_id != "C3_current_map_huber":
            if "C3_current_map_huber" in frames:
                raise StageIMetaTargetExecutionError("multiple fitted Huber controls are ambiguous")
            # The strict runner already reconstructs this fitted residual as
            # causal-map + residual. Preserve the fitted T0 diagnostic, while
            # also exposing the mandatory canonical C3 control ID exactly once.
            frames["C3_current_map_huber"] = frames[spec.arm_id].copy()
    return ledger, frames, manifest


def run_pooled_global_meta_target_evaluation(
    *, long_dir: str | Path, short_dir: str | Path, output_dir: str | Path,
    admission_spec: Causal21dAdmissionSpec = Causal21dAdmissionSpec(),
    resume: bool = False, min_worst_period_selected_rows: int = 10,
    bootstrap_draws: int = 2_000,
) -> dict[str, Any]:
    """Map each side/arm causally to bps, then rank once over both sides."""
    root = Path(output_dir)
    try:
        long_ledger, long_frames, long_manifest = _side_arm_frames(Path(long_dir))
        short_ledger, short_frames, short_manifest = _side_arm_frames(Path(short_dir))
    except StageIMetaTargetExecutionError as error:
        if not str(error).startswith("ADMISSION_REFERENCE_INSUFFICIENT:"):
            raise
        if root.exists() and not resume:
            raise FileExistsError(f"pooled-global output exists without --resume: {root}")
        root.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": SCHEMA,
            "status": "ADMISSION_REFERENCE_INSUFFICIENT",
            "reason": str(error),
            "admission_reference_required": (
                "verified full-population strict-OOF ledger and per-arm scores; "
                "action-only OOF ledgers are never a mapping fallback"
            ),
        }
        (root / "manifest.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return payload
    if set(long_frames) != set(short_frames):
        raise StageIMetaTargetExecutionError("long/short meta-target arms differ")
    request = {
        "schema": SCHEMA, "scope": "pooled_global_after_side_local_causal_common_bps_mapping",
        "long_request_sha256": long_manifest["request_sha256"],
        "short_request_sha256": short_manifest["request_sha256"],
        "admission_spec": admission_spec.__dict__, "arm_ids": sorted(long_frames),
        "min_worst_period_selected_rows": int(min_worst_period_selected_rows),
        "bootstrap_draws": int(bootstrap_draws),
    }
    request_sha = _canonical_sha(request)
    manifest_path = root / "manifest.json"
    if manifest_path.is_file() and resume:
        existing = _json(manifest_path)
        if existing.get("status") == "ADMISSION_REFERENCE_INSUFFICIENT":
            return {**existing, "restart_status": "reference_still_insufficient"}
        if existing.get("request_sha256") != request_sha or existing.get("status") != "complete":
            raise StageIMetaTargetExecutionError("pooled-global resume contract drift")
        metrics_path = root / "pooled_global_metrics.parquet"
        if not metrics_path.is_file() or existing.get("pooled_global_metrics_sha256") != file_sha256(metrics_path):
            raise StageIMetaTargetExecutionError("pooled-global metrics checksum drift")
        return {**existing, "restart_status": "reused_verified_complete"}
    if root.exists() and not resume:
        raise FileExistsError(f"pooled-global output exists without --resume: {root}")
    root.mkdir(parents=True, exist_ok=True)
    mapped_root = root / "arms"
    mapped_root.mkdir(exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    semantic_by_arm = {arm_id: True for arm_id in long_frames}
    if "T3Q_fold_quantile_ordinal_residual" in semantic_by_arm:
        semantic_by_arm["T3Q_fold_quantile_ordinal_residual"] = bool(
            (long_manifest.get("T3Q_promotion_semantic_gate") or {}).get("passed", False)
            and (short_manifest.get("T3Q_promotion_semantic_gate") or {}).get("passed", False)
        )
    selections: dict[tuple[str, float], pd.DataFrame] = {}
    for arm_id in sorted(long_frames):
        combined = pd.concat([long_frames[arm_id], short_frames[arm_id]], ignore_index=True)
        if combined.candidate_key.duplicated().any():
            raise StageIMetaTargetExecutionError(f"{arm_id}: pooled candidate identity collision")
        mapped, audit = apply_causal_21d_side_admission(
            combined, score_column="raw_arm_score", net_column="exact_net_bps",
            decision_column="decision_ts", label_available_column="label_available_ts",
            identity_column="candidate_key", spec=admission_spec,
        )
        mapped["production_comparable"] = mapped.causal_21d_side_expected_net_bps.notna()
        mapped["final_action_admitted"] = (
            mapped.action_candidate.astype(bool)
            & mapped.model_action_admitted.astype(bool)
            & mapped.causal_21d_side_admitted_ge_50bps.astype(bool)
        )
        arm_root = mapped_root / arm_id
        arm_root.mkdir(exist_ok=True)
        mapped.to_parquet(arm_root / "mapped_predictions.parquet", index=False, compression="zstd")
        audit.to_parquet(arm_root / "admission_audit.parquet", index=False, compression="zstd")
        total = len(mapped)
        original_population_rows = int(
            mapped.groupby("side_name", sort=False).original_side_population_rows.first().sum()
        )
        decision_ts = pd.to_datetime(mapped.decision_ts, utc=True)
        day = decision_ts.dt.strftime("%Y-%m-%d")
        week = decision_ts.dt.tz_localize(None).dt.to_period("W-SUN").astype(str)
        month = decision_ts.dt.strftime("%Y-%m")
        evaluation_days = int(day.nunique())
        # Namespace folds by side for robustness attribution only. Selection
        # remains one pooled-global ordering after common-bps conversion.
        fold_key = mapped.side_name.astype(str) + "::" + mapped.fold_id.astype(str)
        eligible = mapped.loc[mapped.final_action_admitted].copy()
        for fraction in TOP_FRACTIONS:
            requested_count = max(1, int(np.ceil(float(fraction) * original_population_rows)))
            count = min(len(eligible), requested_count) if len(eligible) else 0
            selected = eligible.sort_values(
                ["causal_21d_side_expected_net_bps", "candidate_key"],
                ascending=[False, True], kind="stable",
            ).head(count)
            selections[(arm_id, float(fraction))] = selected.loc[:, [
                "candidate_key", "side_name", "__symbol__", "decision_ts", "fold_id",
                "exact_net_bps",
            ]].copy()
            if len(selected):
                values = selected.exact_net_bps.to_numpy(float)
                selected_day = day.loc[selected.index]
                selected_week = week.loc[selected.index]
                selected_month = month.loc[selected.index]
                selected_fold = fold_key.loc[selected.index]
                symbol_counts = selected["__symbol__"].astype(str).value_counts()
                symbol_shares = symbol_counts.to_numpy(float) / float(len(selected))
                day_counts = selected_day.value_counts().to_numpy(float)
                week_counts = selected_week.value_counts().to_numpy(float)
                week_net = pd.Series(values, index=selected_week.to_numpy()).groupby(level=0).sum()
                month_net = pd.Series(values, index=selected_month.to_numpy()).groupby(level=0).sum()
                month_stats = sorted(
                    (
                        float(np.mean(values[selected_month.to_numpy() == value])),
                        str(value), int(np.sum(selected_month.to_numpy() == value)),
                    )
                    for value in selected_month.unique()
                )
                fold_stats = sorted(
                    (
                        float(np.mean(values[selected_fold.to_numpy() == value])),
                        str(value), int(np.sum(selected_fold.to_numpy() == value)),
                    )
                    for value in selected_fold.unique()
                )
                worst_month, worst_month_key, worst_month_rows = month_stats[0]
                worst_fold, worst_fold_key, worst_fold_rows = fold_stats[0]
                unique_symbols = int(len(symbol_counts))
                max_symbol_share = float(symbol_shares.max())
                symbol_hhi = float(np.square(symbol_shares).sum())
                max_day_share = float(day_counts.max() / len(selected))
                max_week_share = float(week_counts.max() / len(selected))
                selected_active_days = int(selected_day.nunique())
                positive_weeks, negative_weeks = int((week_net > 0).sum()), int((week_net < 0).sum())
                positive_months, negative_months = int((month_net > 0).sum()), int((month_net < 0).sum())
            else:
                values = np.asarray([], dtype=float)
                worst_month = worst_fold = np.nan
                worst_month_key = worst_fold_key = None
                worst_month_rows = worst_fold_rows = 0
                unique_symbols = selected_active_days = 0
                max_symbol_share = symbol_hhi = max_day_share = max_week_share = np.nan
                positive_weeks = negative_weeks = positive_months = negative_months = 0
            metric_rows.append({
                "schema": SCHEMA, "scope": "pooled_global_common_bps_after_21d_admission",
                "arm_id": arm_id, "top_fraction": float(fraction),
                "candidate_rows": int(total),
                "original_population_rows": original_population_rows,
                "candidate_population_fraction": float(total / original_population_rows),
                "target_semantic_valid": bool(semantic_by_arm[arm_id]),
                "common_bps_mapped_rows": int(mapped.production_comparable.sum()),
                "admitted_rows": int(mapped.final_action_admitted.sum()), "selected_rows": int(len(selected)),
                "requested_topk_rows": int(requested_count),
                "topk_saturated_due_admission": bool(len(eligible) < requested_count),
                "net_bps_per_trade": float(np.mean(values)) if len(values) else np.nan,
                "worst_month_net_bps_per_trade": worst_month,
                "worst_fold_net_bps_per_trade": worst_fold,
                "worst_month": worst_month_key,
                "worst_month_selected_rows": int(worst_month_rows),
                "worst_fold": worst_fold_key,
                "worst_fold_selected_rows": int(worst_fold_rows),
                "selected_long_rows": int(selected.side_name.eq("long").sum()),
                "selected_short_rows": int(selected.side_name.eq("short").sum()),
                "selected_long_net_bps": float(selected.loc[selected.side_name.eq("long"), "exact_net_bps"].mean()) if selected.side_name.eq("long").any() else np.nan,
                "selected_short_net_bps": float(selected.loc[selected.side_name.eq("short"), "exact_net_bps"].mean()) if selected.side_name.eq("short").any() else np.nan,
                "unique_symbols": unique_symbols,
                "max_symbol_share": max_symbol_share,
                "symbol_hhi": symbol_hhi,
                "max_day_share": max_day_share,
                "max_week_share": max_week_share,
                "evaluation_days": evaluation_days,
                "selected_active_days": selected_active_days,
                "trades_per_day": (
                    float(len(selected) / evaluation_days) if evaluation_days else np.nan
                ),
                "positive_weeks": positive_weeks,
                "negative_weeks": negative_weeks,
                "positive_months": positive_months,
                "negative_months": negative_months,
            })
    metrics = pd.DataFrame(metric_rows)
    paired_rows: list[dict[str, Any]] = []
    for arm_id in sorted(long_frames):
        for fraction in TOP_FRACTIONS:
            arm_selected = selections[(arm_id, float(fraction))].copy()
            raw_selected = selections[("C0_raw_base_exact_noop", float(fraction))].copy()
            for frame, prefix in ((arm_selected, "arm"), (raw_selected, "raw")):
                frame["week"] = (
                    pd.to_datetime(frame.decision_ts, utc=True)
                    .dt.tz_localize(None)
                    .dt.to_period("W-SUN")
                    .astype(str)
                )
                frame.rename(columns={"exact_net_bps": f"{prefix}_net_bps"}, inplace=True)
            arm_week = arm_selected.groupby("week", sort=True).arm_net_bps.agg(["sum", "count"]).rename(columns={"sum": "arm_net_sum_bps", "count": "arm_selected_rows"})
            raw_week = raw_selected.groupby("week", sort=True).raw_net_bps.agg(["sum", "count"]).rename(columns={"sum": "raw_net_sum_bps", "count": "raw_selected_rows"})
            paired = arm_week.join(raw_week, how="outer").fillna(0.0).reset_index()
            for row in paired.itertuples(index=False):
                paired_rows.append({
                    "arm_id": arm_id, "top_fraction": float(fraction), "week": row.week,
                    "arm_net_sum_bps": float(row.arm_net_sum_bps), "arm_selected_rows": int(row.arm_selected_rows),
                    "raw_net_sum_bps": float(row.raw_net_sum_bps), "raw_selected_rows": int(row.raw_selected_rows),
                })
    paired_columns = [
        "arm_id", "top_fraction", "week", "arm_net_sum_bps",
        "arm_selected_rows", "raw_net_sum_bps", "raw_selected_rows",
    ]
    # A valid strict-OOS replay can have no admitted trades at all.  Preserve
    # the full empty schema so checkpointing and groupby remain deterministic
    # instead of raising ``KeyError: arm_id`` before the no-op decision gate.
    paired_ledger = pd.DataFrame.from_records(
        paired_rows, columns=paired_columns
    )
    paired_ledger.to_parquet(root / "paired_week_arm_vs_raw.parquet", index=False, compression="zstd")
    metrics["paired_week_blocks"] = 0
    metrics["paired_week_bootstrap_draws"] = 0
    metrics["paired_week_delta_mean_bps"] = np.nan
    metrics["paired_week_delta_q025_bps"] = np.nan
    metrics["paired_week_delta_q975_bps"] = np.nan
    for (arm_id, fraction), paired in paired_ledger.groupby(["arm_id", "top_fraction"], sort=True):
        arm_sum = paired.arm_net_sum_bps.to_numpy(float)
        arm_count = paired.arm_selected_rows.to_numpy(float)
        raw_sum = paired.raw_net_sum_bps.to_numpy(float)
        raw_count = paired.raw_selected_rows.to_numpy(float)
        weeks = len(paired)
        if weeks and int(bootstrap_draws) > 0:
            seed = int(sha256(f"{arm_id}|{fraction}|20260803".encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            sampled = rng.integers(0, weeks, size=(int(bootstrap_draws), weeks))
            arm_den = arm_count[sampled].sum(axis=1)
            raw_den = raw_count[sampled].sum(axis=1)
            valid = (arm_den > 0) & (raw_den > 0)
            delta = arm_sum[sampled].sum(axis=1)[valid] / arm_den[valid] - raw_sum[sampled].sum(axis=1)[valid] / raw_den[valid]
        else:
            delta = np.asarray([], dtype=float)
        mask = metrics.arm_id.eq(arm_id) & np.isclose(metrics.top_fraction, fraction)
        metrics.loc[mask, "paired_week_blocks"] = int(weeks)
        metrics.loc[mask, "paired_week_bootstrap_draws"] = int(len(delta))
        metrics.loc[mask, "paired_week_delta_mean_bps"] = float(np.mean(delta)) if len(delta) else np.nan
        metrics.loc[mask, "paired_week_delta_q025_bps"] = float(np.quantile(delta, 0.025)) if len(delta) else np.nan
        metrics.loc[mask, "paired_week_delta_q975_bps"] = float(np.quantile(delta, 0.975)) if len(delta) else np.nan
    raw = metrics.loc[metrics.arm_id.eq("C0_raw_base_exact_noop")].set_index("top_fraction")
    base_map = metrics.loc[metrics.arm_id.eq("C1_causal_map_only")].set_index("top_fraction")
    for index, row in metrics.iterrows():
        metrics.loc[index, "delta_vs_raw_net_bps"] = row.net_bps_per_trade - raw.loc[row.top_fraction, "net_bps_per_trade"]
        metrics.loc[index, "delta_vs_raw_worst_month_bps"] = row.worst_month_net_bps_per_trade - raw.loc[row.top_fraction, "worst_month_net_bps_per_trade"]
        metrics.loc[index, "delta_vs_raw_worst_fold_bps"] = row.worst_fold_net_bps_per_trade - raw.loc[row.top_fraction, "worst_fold_net_bps_per_trade"]
        metrics.loc[index, "delta_vs_map_net_bps"] = row.net_bps_per_trade - base_map.loc[row.top_fraction, "net_bps_per_trade"]
    metrics.to_parquet(root / "pooled_global_metrics.parquet", index=False, compression="zstd")
    top10 = metrics.loc[np.isclose(metrics.top_fraction, 0.10)].copy()
    baseline = top10.loc[top10.arm_id.eq("C0_raw_base_exact_noop")].iloc[0]
    eligible = top10.loc[
        ~top10.arm_id.str.startswith("C")
        & top10.net_bps_per_trade.gt(baseline.net_bps_per_trade)
        & top10.worst_month_net_bps_per_trade.ge(baseline.worst_month_net_bps_per_trade)
        & top10.worst_fold_net_bps_per_trade.ge(baseline.worst_fold_net_bps_per_trade)
        & top10.worst_month_selected_rows.ge(int(min_worst_period_selected_rows))
        & top10.worst_fold_selected_rows.ge(int(min_worst_period_selected_rows))
        & top10.paired_week_delta_q025_bps.ge(0.0)
        & top10.target_semantic_valid.astype(bool)
    ].copy()
    if eligible.empty:
        decision = {
            "winner_arm_id": "C0_raw_base_exact_noop", "deployment_action": "no_op",
            "learned_meta_promoted": False,
            "reason": "no learned arm cleared pooled, worst-period support and paired-week bootstrap gates",
        }
    else:
        winner = eligible.sort_values(
            ["net_bps_per_trade", "worst_month_net_bps_per_trade", "paired_week_delta_q025_bps", "arm_id"],
            ascending=[False, False, False, True], kind="stable",
        ).iloc[0]
        decision = {
            "winner_arm_id": str(winner.arm_id), "deployment_action": "learned_meta",
            "learned_meta_promoted": True,
            "reason": "cleared pooled, worst-period support and paired-week bootstrap gates",
        }
    manifest = {
        **request, "status": "complete", "request_sha256": request_sha,
        "decision": decision,
        "comparability_boundary": "raw side scores are diagnostic only; all pooled ranking uses causal 21d side-local monotone expected-net-bps maps and the 50bps admission floor",
        "ranking": "one pooled-global ordering; side attribution computed only after selection",
        "T3Q_promotion_semantic_gate": {
            "required": "residual_q33_bps < 0 <= residual_q67_bps in every long and short OOF fold",
            "passed": bool(semantic_by_arm.get("T3Q_fold_quantile_ordinal_residual", True)),
        },
        "pooled_global_metrics_sha256": file_sha256(root / "pooled_global_metrics.parquet"),
        "paired_week_arm_vs_raw_sha256": file_sha256(root / "paired_week_arm_vs_raw.parquet"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    return manifest


__all__ = [
    "SCHEMA", "StageIMetaTargetExecutionError", "make_lgbm_predictor",
    "run_pooled_global_meta_target_evaluation", "run_side_meta_target_funnel",
]
