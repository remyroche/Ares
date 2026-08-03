"""Fail-closed feature admission for the Stage-III shared residual expert.

This module deliberately performs *admission*, not feature selection or model
fitting.  A caller supplies previously materialised, fold-local permutation
evidence.  The gate records exactly why every feature group was admitted or
rejected and emits the ordered live/meta feature contract used by a later
shared-expert experiment.

All MDA quantities in this contract are higher-is-better permutation effects:
the degradation in the declared validation metric after permuting the field.
``false_positive_loss_mda`` is consequently the increase in the false-positive
loss caused by permutation, not the false-positive loss itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SCHEMA = "stage_iii_feature_admission_v1"
_DIGEST = re.compile(r"[0-9a-f]{64}")


class StageIIIFeatureAdmissionError(ValueError):
    """Raised when the evidence cannot support a deployable feature contract."""


@dataclass(frozen=True)
class FeatureAdmissionConfig:
    min_positive_cell_fraction: float = 0.70
    phantom_quantile: float = 0.95
    min_coverage: float = 0.90
    latest_block: str | None = None

    def validate(self) -> None:
        if not 0.0 < self.min_positive_cell_fraction <= 1.0:
            raise StageIIIFeatureAdmissionError("min_positive_cell_fraction must be in (0, 1]")
        if not 0.0 < self.phantom_quantile < 1.0:
            raise StageIIIFeatureAdmissionError("phantom_quantile must be in (0, 1)")
        if not 0.0 < self.min_coverage <= 1.0:
            raise StageIIIFeatureAdmissionError("min_coverage must be in (0, 1]")
        if self.latest_block is None or not str(self.latest_block).strip():
            raise StageIIIFeatureAdmissionError(
                "latest_block must be explicitly preregistered; lexical inference is forbidden"
            )


@dataclass(frozen=True)
class StageIIIFeatureAdmissionArtifact:
    """Serializable feature-admission decision and immutable source evidence."""

    schema: str
    config: Mapping[str, Any]
    source_digests: Mapping[str, str]
    evidence_sha256: Mapping[str, str]
    group_audit: tuple[Mapping[str, Any], ...]
    feature_audit: tuple[Mapping[str, Any], ...]
    admitted_feature_groups: tuple[str, ...]
    admitted_ordered_features: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def artifact_sha256(self) -> str:
        return sha256(self.to_json().encode("utf-8")).hexdigest()


_CELL_REQUIRED = (
    "feature_group", "fold_id", "seed", "train_environment", "test_environment",
    "within_era_mda", "transport_mda", "false_positive_loss_mda",
    "effect_sign", "effect_magnitude", "sign_reversal_explained",
)
_PHANTOM_REQUIRED = ("fold_id", "seed", "phantom_mda")
_FEATURE_REQUIRED = (
    "feature_name", "feature_group", "feature_order", "coverage", "null_fraction",
    "finite_fraction", "live_parity", "meta_allowed_key",
)
_CONDITIONAL_COLUMNS = (
    "conditioned_within_era_mda", "conditioned_transport_mda",
    "conditioned_false_positive_loss_mda", "conditioned_effect_sign",
    "conditioned_effect_magnitude", "conditioned_sign_reversal_explained",
)


def _stable_frame_sha256(frame: pd.DataFrame) -> str:
    work = frame.copy()
    work = work.reindex(sorted(work.columns), axis=1)
    work = work.sort_values(list(work.columns), kind="stable", na_position="first").reset_index(drop=True)
    payload = work.to_json(orient="split", date_format="iso", double_precision=15)
    return sha256(payload.encode("utf-8")).hexdigest()


def _require(frame: pd.DataFrame, columns: Sequence[str], *, name: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise StageIIIFeatureAdmissionError(f"{name} lacks columns: {missing[:12]}")


def _finite(frame: pd.DataFrame, columns: Sequence[str], *, name: str) -> None:
    for column in columns:
        value = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        if not np.isfinite(value).all():
            raise StageIIIFeatureAdmissionError(f"{name}.{column} must be finite")


def _strict_bool(values: pd.Series, *, name: str) -> np.ndarray:
    if values.isna().any() or not values.isin([True, False, 0, 1]).all():
        raise StageIIIFeatureAdmissionError(f"{name} must contain explicit booleans")
    return values.astype(bool).to_numpy()


def _validate_source_digests(source_digests: Mapping[str, str]) -> dict[str, str]:
    if not source_digests:
        raise StageIIIFeatureAdmissionError("source_digests must bind every source artifact")
    result = {str(key): str(value) for key, value in source_digests.items()}
    invalid = [key for key, value in result.items() if not _DIGEST.fullmatch(value) or len(set(value)) == 1]
    if invalid:
        raise StageIIIFeatureAdmissionError(f"source digests must be non-placeholder SHA-256 values: {invalid[:8]}")
    return dict(sorted(result.items()))


def _validated_inputs(
    cell_evidence: pd.DataFrame,
    phantom_evidence: pd.DataFrame,
    feature_evidence: pd.DataFrame,
    *,
    config: FeatureAdmissionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config.validate()
    _require(cell_evidence, _CELL_REQUIRED, name="cell_evidence")
    _require(phantom_evidence, _PHANTOM_REQUIRED, name="phantom_evidence")
    _require(feature_evidence, _FEATURE_REQUIRED, name="feature_evidence")
    cell = cell_evidence.copy()
    phantom = phantom_evidence.copy()
    feature = feature_evidence.copy()
    _finite(cell, ("within_era_mda", "transport_mda", "false_positive_loss_mda", "effect_magnitude"), name="cell_evidence")
    _finite(phantom, ("phantom_mda",), name="phantom_evidence")
    _finite(feature, ("feature_order", "coverage", "null_fraction", "finite_fraction"), name="feature_evidence")
    if cell.duplicated(["feature_group", "fold_id", "seed", "train_environment", "test_environment"]).any():
        raise StageIIIFeatureAdmissionError("cell evidence must be unique by group/fold/seed/train/test")
    if phantom.groupby(["fold_id", "seed"], observed=True).size().lt(2).any():
        raise StageIIIFeatureAdmissionError("each fold/seed needs at least two phantom draws for q95")
    if feature.duplicated("feature_name").any() or feature.feature_name.astype(str).str.strip().eq("").any():
        raise StageIIIFeatureAdmissionError("feature evidence needs unique non-empty feature_name values")
    if feature.feature_order.duplicated().any():
        raise StageIIIFeatureAdmissionError("feature_order must be globally unique and deterministic")
    for column in ("coverage", "null_fraction", "finite_fraction"):
        if ((feature[column] < 0.0) | (feature[column] > 1.0)).any():
            raise StageIIIFeatureAdmissionError(f"feature evidence {column} must lie in [0, 1]")
    if not np.allclose(feature.coverage.to_numpy(float) + feature.null_fraction.to_numpy(float), 1.0, atol=1e-6):
        raise StageIIIFeatureAdmissionError("coverage and null_fraction must reconcile to one")
    _strict_bool(feature.live_parity, name="feature_evidence.live_parity")
    _strict_bool(feature.meta_allowed_key, name="feature_evidence.meta_allowed_key")
    _strict_bool(cell.sign_reversal_explained, name="cell_evidence.sign_reversal_explained")
    sign = pd.to_numeric(cell.effect_sign, errors="coerce").to_numpy(float)
    if not np.isfinite(sign).all() or not np.isin(sign, [-1.0, 0.0, 1.0]).all():
        raise StageIIIFeatureAdmissionError("effect_sign must be one of -1, 0, +1")
    present_conditional = [column for column in _CONDITIONAL_COLUMNS if column in cell]
    if present_conditional and len(present_conditional) != len(_CONDITIONAL_COLUMNS):
        raise StageIIIFeatureAdmissionError("conditional evidence must provide every conditioned metric/sign field")
    if present_conditional:
        _finite(cell, ("conditioned_within_era_mda", "conditioned_transport_mda", "conditioned_false_positive_loss_mda", "conditioned_effect_magnitude"), name="cell_evidence")
        conditioned_sign = pd.to_numeric(cell.conditioned_effect_sign, errors="coerce").to_numpy(float)
        if not np.isfinite(conditioned_sign).all() or not np.isin(conditioned_sign, [-1.0, 0.0, 1.0]).all():
            raise StageIIIFeatureAdmissionError("conditioned_effect_sign must be one of -1, 0, +1")
        _strict_bool(cell.conditioned_sign_reversal_explained, name="cell_evidence.conditioned_sign_reversal_explained")
    return cell, phantom, feature


def _view_summary(
    local: pd.DataFrame,
    *,
    threshold_by_cell: pd.DataFrame,
    config: FeatureAdmissionConfig,
    prefix: str = "",
) -> dict[str, Any]:
    transport = f"{prefix}transport_mda"
    within = f"{prefix}within_era_mda"
    false_positive = f"{prefix}false_positive_loss_mda"
    sign = f"{prefix}effect_sign"
    magnitude = f"{prefix}effect_magnitude"
    explained = f"{prefix}sign_reversal_explained"
    cols = ["fold_id", "seed", transport, within, false_positive, sign, magnitude, explained, "test_environment"]
    work = local.loc[:, cols].merge(threshold_by_cell, on=["fold_id", "seed"], how="left", validate="many_to_one")
    if work.phantom_q95.isna().any():
        raise StageIIIFeatureAdmissionError("feature evidence has no matching fold-local phantom q95")
    passed = (
        work[transport].gt(work.phantom_q95)
        & work[within].ge(0.0)
        & work[false_positive].ge(0.0)
    )
    values = work[transport].to_numpy(float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    latest_name = str(config.latest_block)
    latest = work.loc[work.test_environment.astype(str).eq(latest_name)]
    if latest.empty:
        raise StageIIIFeatureAdmissionError(f"configured latest block {latest_name!r} has no MDA evidence")
    signs = work[sign].to_numpy(float)
    magnitudes = work[magnitude].to_numpy(float)
    positive_magnitude = float(np.max(magnitudes[signs > 0])) if (signs > 0).any() else 0.0
    negative = (signs < 0) & (magnitudes >= positive_magnitude) & ~work[explained].astype(bool).to_numpy()
    severe_reversal = bool(positive_magnitude > 0.0 and negative.any())
    return {
        "cell_count": int(len(work)),
        "positive_cell_count": int(passed.sum()),
        "positive_cell_fraction": float(passed.mean()),
        "transport_mda_median": median,
        "transport_mda_mad": mad,
        "transport_mda": float(median - 0.5 * mad),
        "within_era_mda_median": float(np.median(work[within].to_numpy(float))),
        "false_positive_loss_mda_median": float(np.median(work[false_positive].to_numpy(float))),
        "phantom_q95_median": float(np.median(work.phantom_q95.to_numpy(float))),
        "latest_block": latest_name,
        "latest_transport_mda": float(np.median(latest[transport].to_numpy(float))),
        "latest_phantom_q95": float(np.median(latest.phantom_q95.to_numpy(float))),
        "severe_unexplained_sign_reversal": severe_reversal,
        "passes_transport": bool(
            passed.mean() >= config.min_positive_cell_fraction
            and (median - 0.5 * mad) > float(np.median(work.phantom_q95.to_numpy(float)))
            and float(np.median(latest[transport].to_numpy(float))) >= 0.0
            and not severe_reversal
        ),
    }


def admit_stage_iii_features(
    cell_evidence: pd.DataFrame,
    phantom_evidence: pd.DataFrame,
    feature_evidence: pd.DataFrame,
    *,
    source_digests: Mapping[str, str],
    config: FeatureAdmissionConfig = FeatureAdmissionConfig(),
) -> StageIIIFeatureAdmissionArtifact:
    """Return an immutable, fail-closed shared-expert feature contract.

    ``cell_evidence`` has one non-phantom measurement per
    feature-group/fold/seed/train-era/test-era cell.  ``phantom_evidence`` has
    at least two random phantom measurements per fold/seed.  Conditional
    columns, when present, are evaluated only after the raw group fails; they
    can classify a group as ``REGIME_CONDITIONAL`` but cannot bypass parity,
    coverage or latest-block gates.
    """
    digests = _validate_source_digests(source_digests)
    cell, phantom, feature = _validated_inputs(
        cell_evidence, phantom_evidence, feature_evidence, config=config,
    )
    threshold = (
        phantom.groupby(["fold_id", "seed"], observed=True)["phantom_mda"]
        .quantile(config.phantom_quantile).rename("phantom_q95").reset_index()
    )
    has_conditional = all(column in cell for column in _CONDITIONAL_COLUMNS)
    group_rows: list[dict[str, Any]] = []
    for group, local in cell.groupby("feature_group", sort=True, observed=True):
        raw = _view_summary(local, threshold_by_cell=threshold, config=config)
        conditional: dict[str, Any] | None = None
        if has_conditional:
            # Re-use the exact same evaluator, but make a narrow view rather
            # than ``rename`` in place: retaining the raw columns would create
            # duplicate labels and silently select the wrong one in pandas.
            conditional_view = local.loc[:, ["fold_id", "seed", "test_environment"]].copy()
            for base in (
                "within_era_mda", "transport_mda", "false_positive_loss_mda",
                "effect_sign", "effect_magnitude", "sign_reversal_explained",
            ):
                conditional_view[base] = local[f"conditioned_{base}"].to_numpy()
            conditional = _view_summary(
                conditional_view, threshold_by_cell=threshold, config=config,
            )
        static = feature.loc[feature.feature_group.astype(str).eq(str(group))]
        static_pass = bool(
            not static.empty
            and static.coverage.ge(config.min_coverage).all()
            and static.null_fraction.le(1.0 - config.min_coverage + 1e-12).all()
            and static.finite_fraction.ge(config.min_coverage).all()
            and _strict_bool(static.live_parity, name="feature_evidence.live_parity").all()
            and _strict_bool(static.meta_allowed_key, name="feature_evidence.meta_allowed_key").all()
        )
        if raw["passes_transport"] and static_pass:
            classification = "INVARIANT_CORE"
        elif conditional is not None and conditional["passes_transport"] and static_pass:
            classification = "REGIME_CONDITIONAL"
        elif raw["positive_cell_count"] == 0 and (conditional is None or conditional["positive_cell_count"] == 0):
            classification = "REDUNDANT"
        elif raw["positive_cell_count"] <= 1 and not raw["severe_unexplained_sign_reversal"]:
            classification = "REGIME_LOCAL_DIAGNOSTIC"
        else:
            classification = "UNSTABLE"
        group_rows.append({
            "feature_group": str(group), "classification": classification,
            "static_contract_pass": static_pass, "raw": raw, "conditional": conditional,
            "feature_count": int(len(static)),
        })
    classified = {row["feature_group"]: row["classification"] for row in group_rows}
    static_feature_pass = (
        feature.coverage.ge(config.min_coverage)
        & feature.null_fraction.le(1.0 - config.min_coverage + 1e-12)
        & feature.finite_fraction.ge(config.min_coverage)
        & _strict_bool(feature.live_parity, name="feature_evidence.live_parity")
        & _strict_bool(feature.meta_allowed_key, name="feature_evidence.meta_allowed_key")
    )
    feature_rows: list[dict[str, Any]] = []
    for row, static_pass in zip(feature.itertuples(index=False), static_feature_pass, strict=True):
        classification = classified.get(str(row.feature_group), "UNSTABLE")
        admitted = bool(static_pass and classification in {"INVARIANT_CORE", "REGIME_CONDITIONAL"})
        feature_rows.append({
            "feature_name": str(row.feature_name), "feature_group": str(row.feature_group),
            "feature_order": int(row.feature_order), "coverage": float(row.coverage),
            "null_fraction": float(row.null_fraction), "finite_fraction": float(row.finite_fraction),
            "live_parity": bool(row.live_parity), "meta_allowed_key": bool(row.meta_allowed_key),
            "classification": classification, "admitted": admitted,
        })
    feature_rows.sort(key=lambda row: (row["feature_order"], row["feature_name"]))
    admitted = tuple(row["feature_name"] for row in feature_rows if row["admitted"])
    if not admitted:
        raise StageIIIFeatureAdmissionError("no Stage-III features survived the fail-closed admission gate")
    groups = tuple(row["feature_group"] for row in group_rows if row["classification"] in {"INVARIANT_CORE", "REGIME_CONDITIONAL"})
    return StageIIIFeatureAdmissionArtifact(
        schema=SCHEMA,
        config=asdict(config),
        source_digests=digests,
        evidence_sha256={
            "cell_evidence": _stable_frame_sha256(cell),
            "phantom_evidence": _stable_frame_sha256(phantom),
            "feature_evidence": _stable_frame_sha256(feature),
        },
        group_audit=tuple(group_rows), feature_audit=tuple(feature_rows),
        admitted_feature_groups=groups, admitted_ordered_features=admitted,
    )


__all__ = [
    "SCHEMA", "FeatureAdmissionConfig", "StageIIIFeatureAdmissionArtifact",
    "StageIIIFeatureAdmissionError", "admit_stage_iii_features",
]
