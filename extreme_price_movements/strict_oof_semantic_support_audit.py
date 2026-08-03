"""Strict OOF evaluation contract for conditional/censored supportive labels.

This module sits *after* :mod:`supportive_target_semantics`.  It does not fit
heads or change a frozen target pack.  It joins a hash-bound semantic-label
sidecar to candidate-level OOF predictions only when exact ID/timestamp and
row-level temporal lineage can be proven.  Otherwise it produces explicit
readiness/blocker rows and no performance metric.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SCHEMA = "strict_oof_semantic_support_audit_v1"
IDENTITY = ("candidate_id", "decision_ts", "label_available_ts")
PREDICTION_LINEAGE_COLUMNS = (
    "candidate_id", "__ts__", "__decision_ts__", "__label_available_at__",
    "is_oof", "prediction_fit_end_ts", "prediction_generated_ts",
    "prediction_model_id", "prediction_fold_id",
)


class SemanticSupportAuditError(ValueError):
    """A requested audit input is structurally unreadable."""


@dataclass(frozen=True)
class SemanticHeadSpec:
    name: str
    kind: str
    target_column: str
    valid_column: str
    prediction_aliases: tuple[str, ...]
    semantics: str


@dataclass(frozen=True)
class SemanticSupportAudit:
    status: str
    readiness: pd.DataFrame
    metrics: pd.DataFrame
    joined: pd.DataFrame | None


def _utc(frame: pd.DataFrame, column: str, *, context: str) -> pd.Series:
    if column not in frame:
        raise SemanticSupportAuditError(f"{context} is missing timestamp column {column!r}")
    values = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if values.isna().any():
        raise SemanticSupportAuditError(f"{context}.{column} contains invalid/missing UTC timestamps")
    return values


def _strict_bool(values: pd.Series, *, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not numeric.isin((0.0, 1.0)).all():
        raise SemanticSupportAuditError(f"{name} must contain only boolean or 0/1 values")
    return numeric.astype(bool)


def semantic_head_specs(label_columns: Sequence[str]) -> list[SemanticHeadSpec]:
    """Return the full reach/conditional/retention/recovery target surface.

    Interval hazards and cumulative incidence are generated from the sidecar's
    declared columns, so changing its pre-registered time boundaries cannot
    silently omit a semantic head from this audit.
    """
    required = {
        "target_meaningful_mfe_reached_12h", "target_source_path_valid",
        "target_opportunity_reached", "target_opportunity_valid",
        "target_peak_mfe_atr_given_meaningful_mfe",
        "target_peak_mfe_atr_given_meaningful_mfe_valid",
        "target_mae_before_meaningful_mfe_atr_given_meaningful_mfe",
        "target_mae_before_meaningful_mfe_atr_given_meaningful_mfe_valid",
        "support_adverse", "support_adverse_valid",
        "support_persistence_given_meaningful_mfe",
        "support_persistence_given_meaningful_mfe_valid",
        "support_adverse_recovery_50pct_confirmed",
        "support_adverse_recovery_50pct_confirmed_valid",
    }
    missing = sorted(required.difference(label_columns))
    if missing:
        raise SemanticSupportAuditError(f"semantic sidecar lacks required target columns: {missing}")
    specs = [
        SemanticHeadSpec(
            "meaningful_mfe_reach", "binary", "target_meaningful_mfe_reached_12h", "target_source_path_valid",
            ("semantic_oof__meaningful_mfe_reach", "pred_meaningful_mfe_reach"),
            "meaningful MFE reach by H12",
        ),
        SemanticHeadSpec(
            "opportunity_reach", "binary", "target_opportunity_reached", "target_opportunity_valid",
            ("semantic_oof__opportunity_reach", "pred_opportunity_reach", "support_oof__clean_opportunity"),
            "clean economic favorable-first opportunity by H12",
        ),
        SemanticHeadSpec(
            "conditional_peak_mfe", "regression", "target_peak_mfe_atr_given_meaningful_mfe", "target_peak_mfe_atr_given_meaningful_mfe_valid",
            ("semantic_oof__conditional_peak_mfe", "pred_conditional_peak_mfe", "support_oof__peak_mfe_atr"),
            "Peak MFE conditional on meaningful-MFE reach",
        ),
        SemanticHeadSpec(
            "conditional_mae_before_mfe", "regression", "target_mae_before_meaningful_mfe_atr_given_meaningful_mfe", "target_mae_before_meaningful_mfe_atr_given_meaningful_mfe_valid",
            ("semantic_oof__conditional_mae_before_mfe", "pred_conditional_mae_before_mfe", "support_oof__mae_before_meaningful_mfe_atr"),
            "pre-MFE MAE conditional on meaningful-MFE reach",
        ),
        SemanticHeadSpec(
            "adverse", "binary", "support_adverse", "support_adverse_valid",
            ("semantic_oof__adverse", "pred_adverse"),
            "adverse-first or same-minute adverse/favorable conflict",
        ),
        SemanticHeadSpec(
            "retention_persistence", "regression", "support_persistence_given_meaningful_mfe", "support_persistence_given_meaningful_mfe_valid",
            ("semantic_oof__retention_persistence", "pred_retention_persistence", "semantic_oof__persistence"),
            "post-reach MFE persistence path efficiency",
        ),
        SemanticHeadSpec(
            "adverse_recovery", "binary", "support_adverse_recovery_50pct_confirmed", "support_adverse_recovery_50pct_confirmed_valid",
            ("semantic_oof__adverse_recovery", "pred_adverse_recovery"),
            "confirmed 50% recovery conditional on an adverse trough",
        ),
    ]
    for target in sorted(column for column in label_columns if column.startswith("target_") and "_cumulative_reach_by_" in column):
        prefix = target.removeprefix("target_").removesuffix("_cumulative_reach_by_" + target.split("_cumulative_reach_by_", 1)[1])
        suffix = target.removeprefix("target_")
        valid = f"target_{prefix}_valid"
        if valid not in label_columns:
            raise SemanticSupportAuditError(f"cumulative target {target} lacks validity column {valid}")
        specs.append(SemanticHeadSpec(
            suffix, "binary", target, valid,
            (f"semantic_oof__{suffix}", f"pred_{suffix}"),
            "cumulative incidence at the declared horizon",
        ))
    for target in sorted(column for column in label_columns if column.startswith("target_") and "_hazard_" in column and not column.endswith("_valid")):
        suffix = target.removeprefix("target_")
        valid = f"{target}_valid"
        if valid not in label_columns:
            raise SemanticSupportAuditError(f"hazard target {target} lacks at-risk validity column {valid}")
        specs.append(SemanticHeadSpec(
            suffix, "binary", target, valid,
            (f"semantic_oof__{suffix}", f"pred_{suffix}"),
            "discrete event hazard conditional on being at risk at interval start",
        ))
    return specs


def _blocker(code: str, detail: str) -> dict[str, Any]:
    return {"record_type": "blocker", "head": None, "status": code, "detail": detail}


def _lineage_blockers(
    labels: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    semantic_contract_sha256: str | None,
    oof_manifest: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], pd.DataFrame | None]:
    blockers: list[dict[str, Any]] = []
    required_labels = set(IDENTITY)
    required_predictions = set(PREDICTION_LINEAGE_COLUMNS)
    missing_labels = sorted(required_labels.difference(labels.columns))
    missing_predictions = sorted(required_predictions.difference(predictions.columns))
    if missing_labels:
        blockers.append(_blocker("BLOCKED_LABEL_IDENTITY_COLUMNS_MISSING", repr(missing_labels)))
    if missing_predictions:
        blockers.append(_blocker("BLOCKED_PREDICTION_LINEAGE_COLUMNS_MISSING", repr(missing_predictions)))
    if semantic_contract_sha256 is not None:
        bound = (oof_manifest or {}).get("semantic_target_contract_sha256")
        if bound != semantic_contract_sha256:
            blockers.append(_blocker(
                "BLOCKED_SEMANTIC_CONTRACT_HASH_UNBOUND",
                "OOF manifest semantic_target_contract_sha256 does not exactly match the supplied semantic sidecar contract",
            ))
    if blockers:
        return blockers, None
    if labels.candidate_id.isna().any() or labels.candidate_id.astype(str).duplicated().any():
        blockers.append(_blocker("BLOCKED_LABEL_CANDIDATE_IDENTITY_NOT_ONE_TO_ONE", "semantic label sidecar candidate_id is null or duplicated"))
    if predictions.candidate_id.isna().any() or predictions.candidate_id.astype(str).duplicated().any():
        blockers.append(_blocker("BLOCKED_PREDICTION_CANDIDATE_IDENTITY_NOT_ONE_TO_ONE", "OOF prediction ledger candidate_id is null or duplicated"))
    if blockers:
        return blockers, None
    decision = _utc(labels, "decision_ts", context="labels")
    label_available = _utc(labels, "label_available_ts", context="labels")
    prediction_decision = _utc(predictions, "__decision_ts__", context="predictions")
    prediction_feature = _utc(predictions, "__ts__", context="predictions")
    prediction_available = _utc(predictions, "__label_available_at__", context="predictions")
    fit_end = _utc(predictions, "prediction_fit_end_ts", context="predictions")
    generated = _utc(predictions, "prediction_generated_ts", context="predictions")
    try:
        is_oof = _strict_bool(predictions.is_oof, name="predictions.is_oof")
    except SemanticSupportAuditError as error:
        return [_blocker("BLOCKED_INVALID_OOF_FLAG", str(error))], None
    if not is_oof.all():
        blockers.append(_blocker("BLOCKED_NON_OOF_PREDICTION_ROWS", f"{int((~is_oof).sum())} rows are not strict OOF"))
    if predictions.prediction_model_id.isna().any() or predictions.prediction_fold_id.isna().any():
        blockers.append(_blocker("BLOCKED_PREDICTION_MODEL_OR_FOLD_MISSING", "model and fold lineage must be present on every candidate"))
    if not prediction_feature.le(prediction_decision).all():
        blockers.append(_blocker("BLOCKED_PREDICTION_FEATURE_AFTER_DECISION", "OOF feature timestamp is after decision on one or more candidates"))
    if not fit_end.lt(prediction_decision).all():
        blockers.append(_blocker("BLOCKED_OOF_FIT_END_NOT_BEFORE_DECISION", "fit end must precede every candidate decision"))
    if not generated.le(prediction_decision).all():
        blockers.append(_blocker("BLOCKED_OOF_GENERATED_AFTER_DECISION", "prediction generated timestamp is after decision"))
    if blockers:
        return blockers, None
    left = predictions.loc[:, ["candidate_id"]].copy()
    left["prediction_decision_ts"] = prediction_decision.to_numpy()
    left["prediction_label_available_ts"] = prediction_available.to_numpy()
    right = labels.loc[:, ["candidate_id"]].copy()
    right["semantic_decision_ts"] = decision.to_numpy()
    right["semantic_label_available_ts"] = label_available.to_numpy()
    joined_time = left.merge(right, on="candidate_id", how="left", validate="one_to_one", indicator=True)
    if not joined_time["_merge"].eq("both").all():
        blockers.append(_blocker("BLOCKED_PREDICTION_TO_LABEL_JOIN_INCOMPLETE", "one or more OOF candidates have no semantic-label row"))
        return blockers, None
    if not joined_time.prediction_decision_ts.eq(joined_time.semantic_decision_ts).all():
        blockers.append(_blocker("BLOCKED_PREDICTION_TO_LABEL_DECISION_TIMESTAMP_MISMATCH", "candidate decision timestamps differ"))
    if not joined_time.prediction_label_available_ts.eq(joined_time.semantic_label_available_ts).all():
        blockers.append(_blocker("BLOCKED_PREDICTION_TO_LABEL_AVAILABILITY_TIMESTAMP_MISMATCH", "candidate label-availability timestamps differ"))
    if blockers:
        return blockers, None
    joined = predictions.merge(labels, on="candidate_id", how="inner", validate="one_to_one", suffixes=("", "__semantic"))
    return blockers, joined


def _rank_ic(prediction: np.ndarray, target: np.ndarray) -> float:
    if len(prediction) < 3 or np.unique(prediction).size < 2 or np.unique(target).size < 2:
        return float("nan")
    return float(pd.Series(prediction).rank(method="average").corr(pd.Series(target).rank(method="average")))


def _metrics(spec: SemanticHeadSpec, joined: pd.DataFrame) -> dict[str, Any]:
    prediction_col = next((name for name in spec.prediction_aliases if name in joined.columns), None)
    base = {
        "record_type": "head", "head": spec.name, "kind": spec.kind,
        "target_column": spec.target_column, "valid_column": spec.valid_column,
        "prediction_column": prediction_col, "semantics": spec.semantics,
    }
    if prediction_col is None:
        return {**base, "status": "NOT_RUN_MISSING_SEMANTIC_OOF_PREDICTION", "valid_target_rows": 0, "scored_rows": 0}
    valid_mask = _strict_bool(joined[spec.valid_column], name=f"labels.{spec.valid_column}")
    target = pd.to_numeric(joined[spec.target_column], errors="coerce")
    prediction = pd.to_numeric(joined[prediction_col], errors="coerce")
    valid = valid_mask & target.notna()
    scored = valid & prediction.notna() & np.isfinite(prediction.to_numpy(float))
    if int(valid.sum()) == 0:
        return {**base, "status": "NOT_RUN_NO_VALID_CONDITIONAL_LABEL_ROWS", "valid_target_rows": 0, "scored_rows": 0}
    if not scored.equals(valid):
        return {**base, "status": "INCOMPLETE_OOF_PREDICTION_COVERAGE", "valid_target_rows": int(valid.sum()), "scored_rows": int(scored.sum())}
    y = target[scored].to_numpy(float)
    p = prediction[scored].to_numpy(float)
    result: dict[str, Any] = {
        **base, "status": "STRICT_OOF_METRIC", "valid_target_rows": int(valid.sum()), "scored_rows": int(scored.sum()),
        "rank_ic": _rank_ic(p, y), "mae": float(np.abs(p - y).mean()), "rmse": float(np.sqrt(np.square(p - y).mean())),
    }
    if spec.kind == "binary":
        if ((p < 0.0) | (p > 1.0)).any():
            return {**base, "status": "INVALID_BINARY_PREDICTION_RANGE", "valid_target_rows": int(valid.sum()), "scored_rows": int(scored.sum())}
        positive = y > 0.5
        result["brier"] = float(np.square(p - positive.astype(float)).mean())
        if positive.any() and (~positive).any():
            # AUC from ranks is equivalent to the Mann-Whitney statistic and
            # avoids a hard sklearn dependency in this contract-only module.
            ranks = pd.Series(p).rank(method="average").to_numpy(float)
            result["auc"] = float((ranks[positive].sum() - positive.sum() * (positive.sum() + 1) / 2.0) / (positive.sum() * (~positive).sum()))
        else:
            result["auc"] = float("nan")
    return result


def audit_semantic_support(
    labels: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    semantic_contract_sha256: str | None = None,
    oof_manifest: Mapping[str, Any] | None = None,
) -> SemanticSupportAudit:
    """Join and evaluate semantic OOF heads, or fail closed with blockers."""
    specs = semantic_head_specs(labels.columns)
    blockers, joined = _lineage_blockers(
        labels, predictions,
        semantic_contract_sha256=semantic_contract_sha256,
        oof_manifest=oof_manifest,
    )
    if blockers:
        readiness = pd.DataFrame(blockers)
        return SemanticSupportAudit(
            status="BLOCKED_FAIL_CLOSED_NO_SEMANTIC_METRICS",
            readiness=readiness,
            metrics=pd.DataFrame(),
            joined=None,
        )
    metrics = pd.DataFrame([_metrics(spec, joined) for spec in specs])
    readiness = metrics.loc[:, ["record_type", "head", "status"]].copy()
    missing = metrics.status.ne("STRICT_OOF_METRIC")
    status = "STRICT_OOF_SEMANTIC_AUDIT_COMPLETE" if not missing.any() else "STRICT_OOF_SEMANTIC_AUDIT_PARTIAL_NOT_PROMOTABLE"
    return SemanticSupportAudit(status=status, readiness=readiness, metrics=metrics, joined=joined)
