"""Strict-OOS evaluator for the explicit S/O -> direct-correctness FQ3 stack.

This is intentionally separate from the R3/Huber production runner.  The
target-v2 route has a different semantic boundary:

``S or O base direct score -> FQ3 over/right/under correctness -> 21d map``.

The FQ3 model sees the *same-side direct base output* and its probability
states.  It never receives a pre-mapped bps prediction.  A score is converted
to common bps only after the base/meta score has been produced, through the
causal pooled-parent / side-shrunk 21-day map.  Consequently there is no
pre-map global ranking or timestamp-local ranking in this module.

The implementation is a library boundary as well as an evaluator: every
source, selector, target, feature and timing contract is checked before any
fit starts.  It is safe to import while other Stage-I jobs are running.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .stage_i_adapter_winner_bundle import StageIAdapterWinnerBundle
from .stage_i_base_target_ablation import recover_ordinal_simplex, training_weights
from .stage_i_causal_admission import (
    Causal21dAdmissionSpec,
    apply_causal_21d_side_admission,
)
from .stage_i_ranking import RANKING_POLICY, stable_stage_i_rank_frame
from .stage_i_strict_oof import _multiclass_probabilities, _validation_blocks
from .stage_i_target_adapter import (
    CUMULATIVE_ORDINAL5_O,
    FOLD_QUANTILE_RESIDUAL3,
    LEGACY_R3_MULTICLASS3,
    SOFT_SCALAR_S,
    StageITargetContract,
    canonical_sha256,
    file_sha256,
    verify_target_contract,
)


SCHEMA = "stage_i_target_specific_direct_fq3_oos_v1"
FROZEN_R3_NORMALIZER_SCHEMA = "stage_i_frozen_r3_finalist_normalizer_v1"
DIRECT_FQ3_SEMANTICS = "same_side_direct_base_output_correctness_q33_v1"
DIRECT_BASE_INPUT_SEMANTICS = "same_side_direct_base_output_without_bps_conversion_v1"
_SIDES = ("long", "short")
_TAILS = (0.01, 0.05, 0.10, 0.20)
_TRUST_FIELDS = (
    "base_output_entropy", "base_output_top2_margin", "base_output_max_probability",
)
_ROLE_CONTRACT_SCHEMA = "stage_i_target_specific_causal_feature_roles_v1"
_MONTH_CONTRACT_SCHEMA = "stage_i_target_specific_2024_2026_month_coverage_v1"
_EVALUATION_TARGET_CONTRACT_SCHEMA = "stage_i_target_specific_evaluation_target_contracts_v1"
_EVALUATION_START = pd.Timestamp("2024-01-01T00:00:00Z")
_EVALUATION_END_EXCLUSIVE = pd.Timestamp("2027-01-01T00:00:00Z")
_EVALUATION_MONTHS = tuple(
    pd.period_range("2024-01", "2026-12", freq="M").astype(str).tolist()
)
_RESERVED_FEATURE_PATTERN = re.compile(
    r"^(?:"
    r"(?:exact_)?(?:net|gross)(?:_|$)|"
    r"(?:target|label|event|outcome|pnl|mfe|mae|future|path|barrier|exit)(?:_|$)|"
    r"(?:prequential|causal_21d|mapped|expected_net|meta_direct|meta_p_)(?:_|$)|"
    r"(?:base_state_p\d+)(?:_|$)"
    r")",
    flags=re.IGNORECASE,
)
# These are decision-time rolling price-path descriptors from features.py, not
# realised trade-path labels.  Keep the broad ``path_*`` guard for all other
# names, but do not reject this small, source-audited causal subset merely for
# its historical naming convention.
_CAUSAL_PATH_FEATURE_ALLOWLIST = frozenset((
    "path_efficiency_12", "path_efficiency_24", "path_efficiency_24_ts_resid",
    "path_entropy_12", "path_entropy_24",
))


def _is_reserved_source_feature(name: str) -> bool:
    return bool(_RESERVED_FEATURE_PATTERN.match(str(name))) and str(name) not in _CAUSAL_PATH_FEATURE_ALLOWLIST


class TargetSpecificOOSError(ValueError):
    """Raised for any source/semantic mismatch in the direct FQ3 route."""


@dataclass(frozen=True)
class DirectCorrectnessState:
    """Fold-local direct-score correctness state.

    ``outcome_rank`` is the prior-training empirical net-outcome percentile.
    The FQ3 residual is therefore in the direct score's unit, not bps:
    ``outcome_rank - base_direct_score``.  Classes are the lower, middle and
    upper residual terciles.  When the residual distribution straddles zero,
    those terciles also have the intuitive overestimate / approximately-right
    / underestimate interpretation.  A systematic calibration offset is a
    meta-learning target and is audited rather than rejected.
    """

    thresholds: tuple[float, float]
    class_prior: tuple[float, float, float]
    class_locations: tuple[float, float, float]
    class_support: tuple[int, int, int]
    clip: float = 1.0
    score_lower: float = 0.0
    score_upper: float = 1.0
    semantic_gate: str = "finite_ordered_q33_q67_with_zero_straddle_audit"

    def __post_init__(self) -> None:
        q33, q67 = self.thresholds
        if not (np.isfinite(q33) and np.isfinite(q67) and q33 < q67):
            raise TargetSpecificOOSError("direct FQ3 requires finite ordered tercile thresholds")
        if any(value < 1 for value in self.class_support):
            raise TargetSpecificOOSError("direct FQ3 fold lacks a class")
        if not (
            np.isfinite(self.score_lower) and np.isfinite(self.score_upper)
            and self.score_lower < self.score_upper
        ):
            raise TargetSpecificOOSError("direct FQ3 score domain is invalid")
        prior = np.asarray(self.class_prior, dtype=float)
        if (prior <= 0).any() or not np.isclose(prior.sum(), 1.0):
            raise TargetSpecificOOSError("direct FQ3 priors are invalid")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DirectFQ3Estimator:
    """Selector-compatible classifier whose prediction is direct-score correction."""

    def __init__(self, model: Any, state: DirectCorrectnessState, feature_names: Sequence[str]):
        self.model = model
        self.state = state
        self.feature_name_ = tuple(map(str, feature_names))
        self.classes_ = np.asarray([0, 1, 2], dtype=np.int8)
        raw_importance = getattr(model, "feature_importances_", np.zeros(len(self.feature_name_)))
        self.feature_importances_ = np.asarray(raw_importance, dtype=np.float64)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return _multiclass_probabilities(self.model, frame)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if "base_raw_score" not in frame:
            raise TargetSpecificOOSError("direct FQ3 estimator requires the native base_raw_score")
        probability = self.predict_proba(frame)
        correction, _ = _reconstruct_direct_correctness(
            probability, pd.to_numeric(frame.base_raw_score, errors="raise").to_numpy(float), self.state,
        )
        return correction


class _DirectFQ3PriorModel:
    """Neutral nested-resample fallback when a parent-defined class is absent."""

    classes_ = np.asarray([0, 1, 2], dtype=np.int8)

    def __init__(self, prior: Sequence[float], n_features: int):
        self.prior = np.asarray(prior, dtype=np.float32)
        self.feature_importances_ = np.zeros(int(n_features), dtype=np.float64)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.tile(self.prior, (len(frame), 1))


def fit_direct_fq3_estimator(
    frame: pd.DataFrame, exact_net_bps: Sequence[float], sample_weight: Sequence[float] | None,
    *, params: Mapping[str, Any], score_domain: tuple[float, float], fit_model: Callable[..., Any],
) -> DirectFQ3Estimator:
    """Fit FQ3 labels exclusively from this call's training rows.

    This is the reusable boundary needed by MDA/HPO: internal resamples pass
    only their own frame and exact-net labels, so outcome percentiles and
    terciles cannot see validation rows.  ``base_raw_score`` is the native
    same-side OOF scalar and no bps-map field is accepted.
    """
    if "base_raw_score" not in frame:
        raise TargetSpecificOOSError("direct FQ3 fit requires base_raw_score")
    if any(_is_reserved_source_feature(str(name)) for name in frame.columns if "mapped" in str(name).lower() or "expected_net" in str(name).lower()):
        raise TargetSpecificOOSError("direct FQ3 fit forbids pre-mapped expected-net features")
    net = np.asarray(exact_net_bps, dtype=np.float64).reshape(-1)
    base = pd.to_numeric(frame.base_raw_score, errors="coerce").to_numpy(np.float64)
    labels, state = _fit_direct_correctness(net, base, score_domain=score_domain)
    weight = np.ones(len(frame), dtype=np.float32) if sample_weight is None else np.asarray(sample_weight, dtype=np.float32).reshape(-1)
    if len(weight) != len(frame) or not np.isfinite(weight).all() or (weight < 0).any():
        raise TargetSpecificOOSError("direct FQ3 sample weight is invalid")
    model = fit_model(
        frame, labels, weight, classifier=True,
        params=_clean_params(params, objective="multiclass", num_class=3),
        objective_mode="stage_i_direct_correctness_FQ3",
    )
    return DirectFQ3Estimator(model, state, tuple(frame.columns))


@contextmanager
def direct_fq3_selector_fit_context(*, parent_state: DirectCorrectnessState) -> Any:
    """Use one parent-defined FQ3 state across all nested selector resamples.

    The target supplied to each internal fit is already the immutable
    parent-reference class vector. Nested feature permutations therefore
    cannot change labels by perturbing ``base_raw_score``. A child slice with
    fewer than three parent-defined classes is explicitly neutral: it emits
    the parent prior and consequently contributes no spurious MDA lift.
    Final chronological HPO/OOS do not use this context and retain their
    strict fold-local target construction and class-support gate.
    """
    from . import lgbm_pipeline

    original = lgbm_pipeline._fit_lgbm_model

    def _fit(
        frame: pd.DataFrame, target: Sequence[float], sample_weight: Any = None,
        *, params: Mapping[str, Any], **_: Any,
    ) -> DirectFQ3Estimator:
        if "base_raw_score" not in frame:
            raise TargetSpecificOOSError("direct FQ3 selector requires protected base_raw_score")
        labels = np.asarray(target, dtype=np.int8).reshape(-1)
        if len(labels) != len(frame) or not np.isin(labels, [0, 1, 2]).all():
            raise TargetSpecificOOSError("direct FQ3 selector requires immutable parent-defined classes")
        observed = np.unique(labels)
        if len(observed) < 3:
            estimator = DirectFQ3Estimator(
                _DirectFQ3PriorModel(parent_state.class_prior, frame.shape[1]),
                parent_state, tuple(frame.columns),
            )
            estimator.nested_support_audit = {
                "status": "unsupported_child_class_support_neutral_prior",
                "observed_classes": observed.astype(int).tolist(),
                "rows": len(frame),
                "parent_thresholds": list(parent_state.thresholds),
            }
            return estimator
        weight = np.ones(len(frame), dtype=np.float32) if sample_weight is None else np.asarray(sample_weight, dtype=np.float32)
        model = original(
            frame, labels, weight, classifier=True,
            params=_clean_params(params, objective="multiclass", num_class=3),
            objective_mode="stage_i_direct_correctness_FQ3_selector_parent_state",
        )
        estimator = DirectFQ3Estimator(model, parent_state, tuple(frame.columns))
        estimator.nested_support_audit = {
            "status": "supported_parent_defined_three_class_fit",
            "observed_classes": [0, 1, 2], "rows": len(frame),
            "parent_thresholds": list(parent_state.thresholds),
        }
        return estimator

    lgbm_pipeline._fit_lgbm_model = _fit
    try:
        yield
    finally:
        lgbm_pipeline._fit_lgbm_model = original


@dataclass(frozen=True)
class StageITargetSpecificInput:
    """One full side panel and the manifests that bind it.

    ``contract_frame`` must carry the target winner's identity, selected
    S/O label, and winner-geometry gross/net fields.  It is deliberately
    separate from the wide feature frame so the latter cannot redefine labels.
    """

    side: str
    frame: pd.DataFrame
    contract_frame: pd.DataFrame
    source_manifest: Mapping[str, Any]
    source_manifest_sha256: str
    source_file_sha256: Mapping[str, str]
    base_selector_manifest: Mapping[str, Any]
    meta_selector_manifest: Mapping[str, Any]
    # These are byte hashes of the selector manifests, matching the hashes in
    # the immutable winner bundle (rather than a convenient re-serialisation).
    base_selector_manifest_sha256: str
    meta_selector_manifest_sha256: str
    base_target_column: str
    meta_target_column: str
    # R3 is already a completed strict-OOF base. Supplying its immutable
    # ledger prevents an accidental refit. S/O inputs must leave these unset.
    frozen_base_oof: pd.DataFrame | None = None
    frozen_base_oof_manifest: Mapping[str, Any] | None = None
    frozen_base_oof_file_sha256: str = ""
    frozen_base_oof_manifest_sha256: str = ""
    n_validation_folds: int = 4
    min_train_rows: int = 500


@dataclass(frozen=True)
class StageITargetSpecificResult:
    side: str
    predictions: pd.DataFrame
    fold_provenance: pd.DataFrame
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class StageITargetSpecificFinalist:
    """One already-generated finalist ledger for joint-stack comparison.

    The base family may be R3, S, or O.  Comparison deliberately consumes
    only the reconstructed/meta common-bps score, never base economics.
    """

    name: str
    predictions: pd.DataFrame
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class FrozenR3FinalistInput:
    """Immutable inputs for adapting an already-frozen R3 joint ledger.

    This is intentionally a schema adapter, not a model or mapping runner.
    The separate admission ledger must already contain the causal 21-day map;
    the adapter merely verifies, joins, and renames its output.
    """

    strict_oof_predictions: pd.DataFrame
    admission_predictions: pd.DataFrame
    coverage_audit: pd.DataFrame
    strict_oof_manifest: Mapping[str, Any]
    admission_manifest: Mapping[str, Any]
    strict_oof_file_sha256: str
    strict_oof_manifest_sha256: str
    admission_file_sha256: str
    admission_manifest_sha256: str
    coverage_audit_sha256: str
    # The source manifests must name these exact frozen files under either a
    # legacy ``artifacts`` ledger or a newer ``files`` ledger.  The basename
    # also matches a manifest's relative-path key.
    strict_oof_artifact_path: str = "strict_oof_predictions.parquet"
    admission_artifact_path: str = "causal_admission_predictions.parquet"


def _utc(values: Sequence[Any], n: int, *, name: str) -> pd.Series:
    out = pd.to_datetime(pd.Series(values), utc=True, errors="coerce")
    if len(out) != n or out.isna().any():
        raise TargetSpecificOOSError(f"{name} must be a finite row-aligned UTC vector")
    return out


def _array(values: Sequence[Any], n: int, *, name: str, dtype: Any) -> np.ndarray:
    out = np.asarray(values, dtype=dtype).reshape(-1)
    if len(out) != n:
        raise TargetSpecificOOSError(f"{name} must be row-aligned")
    return out


def _clean_params(params: Mapping[str, Any], *, objective: str, num_class: int | None = None) -> dict[str, Any]:
    out = dict(params)
    out["objective"] = objective
    if num_class is None:
        out.pop("num_class", None)
    else:
        out["num_class"] = int(num_class)
    return out


def _base_fold(
    contract: StageITargetContract, x_train: pd.DataFrame, y_train: np.ndarray,
    weight: np.ndarray, x_valid: pd.DataFrame, params: Mapping[str, Any],
    fit_model: Callable[..., Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit an S/O base fold and return a [0,1] direct score plus simplex."""
    if contract.family == SOFT_SCALAR_S:
        model = fit_model(
            x_train, y_train.astype(np.float32), weight, classifier=False,
            params=_clean_params(params, objective="regression_l1"),
            objective_mode="stage_i_target_specific_direct_S",
        )
        score = np.clip(np.asarray(model.predict(x_valid), dtype=np.float32).reshape(-1), 0.0, 1.0)
        return score, np.column_stack([1.0 - score, score]).astype(np.float32)
    if contract.family == CUMULATIVE_ORDINAL5_O:
        # Reuse the existing four-head implementation indirectly through its
        # public selector context would hide the fit.  The ordinal model is
        # deliberately implemented by the adapter's tested function.
        from .stage_i_target_adapter import fit_cumulative_ordinal5_estimator

        model = fit_cumulative_ordinal5_estimator(
            x_train, y_train.astype(np.int8), weight,
            params=_clean_params(params, objective="binary"),
        )
        simplex = recover_ordinal_simplex(model.predict_cumulative_probability(x_valid))
        score = (simplex @ (np.arange(5, dtype=np.float32) / 4.0)).astype(np.float32)
        return np.clip(score, 0.0, 1.0), simplex.astype(np.float32)
    raise TargetSpecificOOSError("direct FQ3 route permits only S or O bases")


def _outcome_percentile(train_net: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Prior-fold empirical CDF in [0,1], with no future label reference."""
    reference = np.sort(np.asarray(train_net, dtype=np.float64))
    value = np.asarray(values, dtype=np.float64)
    if len(reference) < 3 or not np.isfinite(reference).all() or not np.isfinite(value).all():
        raise TargetSpecificOOSError("direct correctness needs finite prior net outcomes")
    # Mid-rank removes the artificial endpoints but retains deterministic ties.
    return np.clip((np.searchsorted(reference, value, side="right") - 0.5) / len(reference), 0.0, 1.0)


def _fit_direct_correctness(
    net: np.ndarray, base_score: np.ndarray, *, score_domain: tuple[float, float] = (0.0, 1.0),
) -> tuple[np.ndarray, DirectCorrectnessState]:
    net = np.asarray(net, dtype=np.float64).reshape(-1)
    base = np.asarray(base_score, dtype=np.float64).reshape(-1)
    if len(net) < 3 or len(net) != len(base) or not np.isfinite(net).all() or not np.isfinite(base).all():
        raise TargetSpecificOOSError("direct FQ3 fit needs aligned finite base score/net")
    score_lower, score_upper = (float(value) for value in score_domain)
    if not (np.isfinite(score_lower) and np.isfinite(score_upper) and score_lower < score_upper):
        raise TargetSpecificOOSError("direct base score domain is invalid")
    if ((base < score_lower - 1e-6) | (base > score_upper + 1e-6)).any():
        raise TargetSpecificOOSError("direct base score lies outside its declared native domain")
    outcome_coordinate = score_lower + (score_upper - score_lower) * _outcome_percentile(net, net)
    residual = outcome_coordinate - base
    q33, q67 = (float(v) for v in np.quantile(residual, (1.0 / 3.0, 2.0 / 3.0), method="linear"))
    labels = np.digitize(residual, (q33, q67), right=True).astype(np.int8)
    support = tuple(int((labels == value).sum()) for value in range(3))
    if not (np.isfinite(q33) and np.isfinite(q67) and q33 < q67) or any(value < 1 for value in support):
        raise TargetSpecificOOSError("direct FQ3 fold lacks finite ordered terciles or class support")
    lower, upper = np.quantile(residual, (0.05, 0.95), method="linear")
    clipped = np.clip(residual, lower, upper)
    global_location = float(clipped.mean())
    locations = tuple(float((clipped[labels == value].sum() + 50.0 * global_location) / (support[value] + 50.0)) for value in range(3))
    state = DirectCorrectnessState(
        thresholds=(q33, q67), class_prior=tuple(float(value / len(labels)) for value in support),
        class_locations=locations, class_support=support,
        clip=score_upper - score_lower, score_lower=score_lower, score_upper=score_upper,
    )
    return labels, state


def _reconstruct_direct_correctness(
    probabilities: np.ndarray, base_score: np.ndarray, state: DirectCorrectnessState,
) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probabilities, dtype=np.float64)
    base = np.asarray(base_score, dtype=np.float64).reshape(-1)
    if p.shape != (len(base), 3) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1.0, atol=1e-5):
        raise TargetSpecificOOSError("direct FQ3 predictions must be a finite three-state simplex")
    correction = (p - np.asarray(state.class_prior, dtype=float)) @ np.asarray(state.class_locations, dtype=float)
    correction = np.clip(correction, -state.clip, state.clip)
    return correction.astype(np.float32), np.clip(
        base + correction, state.score_lower, state.score_upper,
    ).astype(np.float32)


def _direct_trust(simplex: np.ndarray) -> pd.DataFrame:
    p = np.asarray(simplex, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] < 2 or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1.0, atol=1e-5):
        raise TargetSpecificOOSError("base state probabilities must be finite simplexes")
    ordered = np.sort(p, axis=1)
    return pd.DataFrame({
        "base_output_entropy": (-np.sum(p * np.log(np.clip(p, 1e-12, 1.0)), axis=1)).astype(np.float32),
        "base_output_top2_margin": (ordered[:, -1] - ordered[:, -2]).astype(np.float32),
        "base_output_max_probability": ordered[:, -1].astype(np.float32),
    })


def _require_selector(
    manifest: Mapping[str, Any], *, supplied_sha: str, expected_sha: str, side: str, layer: str,
    target: StageITargetContract, features: tuple[str, ...], allow_legacy_r3_rebind: bool = False,
) -> str:
    if str(supplied_sha) != str(expected_sha) or len(str(supplied_sha)) != 64:
        raise TargetSpecificOOSError(f"{side}/{layer}: selector manifest SHA drift")
    if str(manifest.get("status", "")) != "complete" or str(manifest.get("side", "")).lower() != side:
        raise TargetSpecificOOSError(f"{side}/{layer}: selector must be completed and side-local")
    if str(manifest.get("target_contract_sha256", "")) != target.sha256:
        legacy = manifest.get("target_contract", {})
        legacy_ok = (
            allow_legacy_r3_rebind and layer == "base" and isinstance(legacy, Mapping)
            and str(legacy.get("family", "")) == LEGACY_R3_MULTICLASS3
            and bool(legacy.get("metadata", {}).get("schema_v1_compatibility_only", False))
        )
        if not legacy_ok:
            raise TargetSpecificOOSError(f"{side}/{layer}: selector target winner drift")
    selected = tuple(map(str, manifest.get("selected_feature_contract", manifest.get("selected_features", ()))))
    if selected != features:
        raise TargetSpecificOOSError(f"{side}/{layer}: selector selected-feature contract drift")
    policy = str(manifest.get("correlation_policy", ""))
    if not policy:
        raise TargetSpecificOOSError(f"{side}/{layer}: selector lacks correlation policy lineage")
    return policy


def _role_contract(source_manifest: Mapping[str, Any], *, side: str) -> tuple[set[str], set[str], Mapping[str, Any]]:
    """Bind source-approved causal fields before a selected feature is read.

    The contract intentionally lists only source fields.  Direct base states,
    direct score, and trust values are generated in-process after strict base
    OOF and must never be supplied by a parquet source.
    """
    raw = source_manifest.get("causal_feature_role_contract")
    expected = str(source_manifest.get("causal_feature_role_contract_sha256", ""))
    if not isinstance(raw, Mapping) or len(expected) != 64 or canonical_sha256(dict(raw)) != expected:
        raise TargetSpecificOOSError(f"{side}: approved causal feature-role contract hash drift")
    if raw.get("schema") != _ROLE_CONTRACT_SCHEMA:
        raise TargetSpecificOOSError(f"{side}: unsupported causal feature-role contract")
    base = tuple(map(str, raw.get("base_source_features", ())))
    meta = tuple(map(str, raw.get("meta_source_features", ())))
    if not base or not meta or len(set(base)) != len(base) or len(set(meta)) != len(meta):
        raise TargetSpecificOOSError(f"{side}: causal feature-role contract is empty/ambiguous")
    if any(_is_reserved_source_feature(name) for name in (*base, *meta)):
        raise TargetSpecificOOSError(f"{side}: source role contract permits a reserved target/path/map/state field")
    return set(base), set(meta), raw


def _validate_selected_feature_roles(
    *, side: str, frame: pd.DataFrame, base_features: Sequence[str], meta_features: Sequence[str],
    base_source_features: set[str], meta_source_features: set[str], required_generated: set[str],
) -> None:
    generated_prefix = re.compile(r"^base_state_p\d+$")
    # No generated FQ3 handoff may be supplied by the source even if it happens
    # to have the expected name.  This closes the state/god-feature shadowing
    # route before the design matrix is built.
    generated_names = {"base_raw_score", *_TRUST_FIELDS}
    source_generated = generated_names.intersection(frame.columns) | {
        name for name in frame.columns if generated_prefix.match(str(name))
    }
    if source_generated:
        raise TargetSpecificOOSError(f"{side}: source frame illegally supplies generated handoffs: {sorted(source_generated)}")
    forbidden = {
        name for name in (*base_features, *meta_features)
        if _is_reserved_source_feature(str(name))
        and name not in required_generated
    }
    if forbidden:
        raise TargetSpecificOOSError(f"{side}: selected feature uses reserved target/economic/path/map/state namespace: {sorted(forbidden)}")
    unknown_base = set(map(str, base_features)).difference(base_source_features)
    if unknown_base:
        raise TargetSpecificOOSError(f"{side}: selected base feature is absent from approved causal inventory: {sorted(unknown_base)}")
    external_meta = set(map(str, meta_features)).difference(required_generated)
    unknown_meta = external_meta.difference(meta_source_features)
    if unknown_meta:
        raise TargetSpecificOOSError(f"{side}: selected meta feature is absent from approved causal inventory: {sorted(unknown_meta)}")


def _validate_frozen_r3_handoff(
    source: StageITargetSpecificInput, *, side: str, contract: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return immutable native R3 score/simplex/fold without fitting a base.

    The native scalar is exactly ``P(clear)-P(adverse)`` in ``[-1,1]``.  It
    is not an EV estimate and is never mapped before FQ3.  The supplied base
    manifest must be the same byte-bound selector manifest already frozen in
    the winner cell, including its strict prior-resolved fold audit.
    """
    frozen = source.frozen_base_oof
    manifest = source.frozen_base_oof_manifest
    if frozen is None or not isinstance(manifest, Mapping):
        raise TargetSpecificOOSError(f"{side}: frozen R3 requires its immutable strict-OOF ledger and manifest")
    if dict(manifest) != dict(source.base_selector_manifest):
        raise TargetSpecificOOSError(f"{side}: frozen R3 manifest differs from the bound base selector")
    if source.frozen_base_oof_manifest_sha256 != source.base_selector_manifest_sha256:
        raise TargetSpecificOOSError(f"{side}: frozen R3 selector-manifest SHA drift")
    expected_oof_sha = str(manifest.get("selector_base_oof_sha256", ""))
    if len(source.frozen_base_oof_file_sha256) != 64 or source.frozen_base_oof_file_sha256 != expected_oof_sha:
        raise TargetSpecificOOSError(f"{side}: frozen R3 OOF file SHA drift")
    if str(manifest.get("hpo_oof_score_semantics", "")) != "P(clear)-P(adverse)":
        raise TargetSpecificOOSError(f"{side}: frozen R3 native score semantics drift")
    audit = manifest.get("hpo_oof_regeneration_fold_audit")
    if not isinstance(audit, Sequence) or not audit or any(
        not isinstance(item, Mapping) or not bool(item.get("strict_prior_resolved", False))
        for item in audit
    ):
        raise TargetSpecificOOSError(f"{side}: frozen R3 lacks strict prior-resolved fold provenance")
    required = {
        "candidate_id", "__ts__", "__symbol__", "side_name", "decision_ts",
        "label_available_ts", "exact_net_bps", "exact_gross_bps", "r3_p_adverse",
        "r3_p_weak", "r3_p_clear", "r3_opportunity_score", "base_raw_score",
        "base_oof_fold_id",
    }
    if missing := required.difference(frozen.columns):
        raise TargetSpecificOOSError(f"{side}: frozen R3 OOF lacks {sorted(missing)}")
    identity = ["candidate_id", "__ts__", "__symbol__"]
    left = contract.loc[:, identity].copy()
    right = frozen.loc[:, identity].copy()
    left["__ts__"] = pd.to_datetime(left["__ts__"], utc=True, errors="coerce")
    right["__ts__"] = pd.to_datetime(right["__ts__"], utc=True, errors="coerce")
    # A signed joint-finalist universe is allowed to be a strict subset of
    # the frozen base OOF ledger.  Keep the immutable full ledger as the
    # source of truth, then project it into the contract's declared order.
    # This is intentionally an exact identity projection (not a merge), so a
    # shared-population filter can neither invent nor duplicate a base score.
    left_key = pd.MultiIndex.from_frame(left.astype(str))
    right_key = pd.MultiIndex.from_frame(right.astype(str))
    positions = right_key.get_indexer(left_key)
    if (
        left.isna().any().any() or right.isna().any().any()
        or not right_key.is_unique or (positions < 0).any()
        or len(np.unique(positions)) != len(positions)
    ):
        raise TargetSpecificOOSError(f"{side}: frozen R3 OOF identity/order drift")
    frozen = frozen.iloc[positions].reset_index(drop=True)
    if not left.equals(frozen.loc[:, identity].assign(__ts__=pd.to_datetime(frozen["__ts__"], utc=True, errors="coerce"))):
        raise TargetSpecificOOSError(f"{side}: frozen R3 OOF identity/order drift")
    if not frozen.side_name.astype(str).str.lower().eq(side).all():
        raise TargetSpecificOOSError(f"{side}: frozen R3 OOF contains cross-side rows")
    decision = pd.to_datetime(frozen.decision_ts, utc=True, errors="coerce")
    available = pd.to_datetime(frozen.label_available_ts, utc=True, errors="coerce")
    expected_decision = pd.to_datetime(contract.decision_ts, utc=True, errors="coerce")
    expected_available = pd.to_datetime(contract.label_available_ts, utc=True, errors="coerce")
    if decision.isna().any() or available.isna().any() or not decision.equals(expected_decision) or not available.equals(expected_available):
        raise TargetSpecificOOSError(f"{side}: frozen R3 timing drift")
    score = pd.to_numeric(frozen.base_raw_score, errors="coerce").to_numpy(np.float32)
    native = pd.to_numeric(frozen.r3_opportunity_score, errors="coerce").to_numpy(np.float32)
    simplex = frozen.loc[:, ["r3_p_adverse", "r3_p_weak", "r3_p_clear"]].to_numpy(np.float32)
    folds = pd.to_numeric(frozen.base_oof_fold_id, errors="coerce").fillna(-1).to_numpy(np.int16)
    finite = np.isfinite(score)
    if not np.array_equal(finite, folds >= 0):
        raise TargetSpecificOOSError(f"{side}: frozen R3 finite-score/fold availability drift")
    if finite.any():
        p = simplex[finite]
        if (
            not np.isfinite(p).all() or (p < 0).any()
            or not np.allclose(p.sum(axis=1), 1.0, atol=1e-5)
            or not np.allclose(score[finite], native[finite], atol=1e-6)
            or not np.allclose(score[finite], p[:, 2] - p[:, 0], atol=1e-6)
            or (score[finite] < -1.0).any() or (score[finite] > 1.0).any()
        ):
            raise TargetSpecificOOSError(f"{side}: frozen R3 handoff is not the native direct simplex/contrast")
    return score, simplex, folds


def _month_contract(source_manifest: Mapping[str, Any], *, side: str) -> Mapping[str, Any]:
    raw = source_manifest.get("evaluation_month_contract")
    expected_sha = str(source_manifest.get("evaluation_month_contract_sha256", ""))
    if not isinstance(raw, Mapping) or len(expected_sha) != 64 or canonical_sha256(dict(raw)) != expected_sha:
        raise TargetSpecificOOSError(f"{side}: evaluation month coverage contract hash drift")
    if raw.get("schema") != _MONTH_CONTRACT_SCHEMA or tuple(map(str, raw.get("expected_months", ()))) != _EVALUATION_MONTHS:
        raise TargetSpecificOOSError(f"{side}: evaluation month contract must declare every 2024--2026 month")
    availability = raw.get("source_availability")
    if not isinstance(availability, Mapping) or set(map(str, availability)) != set(_EVALUATION_MONTHS):
        raise TargetSpecificOOSError(f"{side}: source availability must cover every 2024--2026 month")
    for month, item in availability.items():
        if not isinstance(item, Mapping) or not isinstance(item.get("source_available"), bool):
            raise TargetSpecificOOSError(f"{side}/{month}: source availability must be explicit boolean")
        if not item["source_available"] and not str(item.get("source_gap_reason", "")).strip():
            raise TargetSpecificOOSError(f"{side}/{month}: declared source gap needs a reason")
        if bool(item.get("allow_zero_strict_coverage", False)) and not str(item.get("zero_coverage_reason", "")).strip():
            raise TargetSpecificOOSError(f"{side}/{month}: declared zero strict coverage needs a reason")
    return raw


def _target_semantic_signature(contract: StageITargetContract) -> dict[str, Any]:
    """Return the immutable target meaning, excluding population content.

    Selector target contracts intentionally bind the exact rows used for
    feature selection/HPO.  Those identity, label, economics, validity and
    weight hashes cannot also describe a later OOS population.  The source
    manifest therefore carries newly bound evaluation contracts, while this
    signature proves that their target meaning is unchanged.
    """
    return {
        "family": contract.family,
        "layer": contract.layer,
        "target_name": contract.target_name,
        "geometry": contract.geometry,
        "target_columns": list(contract.target_columns),
        "economics_columns": list(contract.economics_columns),
        "validity_column": contract.validity_column,
        "weight_column": contract.weight_column,
        "metadata": dict(contract.metadata),
    }


def _evaluation_target_contracts(
    source_manifest: Mapping[str, Any], *, side: str,
    training_base: StageITargetContract, training_meta: StageITargetContract,
) -> tuple[StageITargetContract, StageITargetContract]:
    """Load exact OOS-row contracts without confusing them with training rows."""
    raw = source_manifest.get("evaluation_target_contracts")
    expected = str(source_manifest.get("evaluation_target_contracts_sha256", ""))
    if not isinstance(raw, Mapping) or len(expected) != 64 or canonical_sha256(dict(raw)) != expected:
        raise TargetSpecificOOSError(f"{side}: evaluation target-contract hash drift")
    if raw.get("schema") != _EVALUATION_TARGET_CONTRACT_SCHEMA:
        raise TargetSpecificOOSError(f"{side}: unsupported evaluation target-contract schema")
    if str(raw.get("side", "")).lower() != side:
        raise TargetSpecificOOSError(f"{side}: evaluation target contracts are cross-side")
    try:
        base = StageITargetContract.from_dict(raw["base"])
        meta = StageITargetContract.from_dict(raw["meta"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TargetSpecificOOSError(f"{side}: malformed evaluation target contracts") from exc
    for layer, evaluation, training in (
        ("base", base, training_base), ("meta", meta, training_meta),
    ):
        if _target_semantic_signature(evaluation) != _target_semantic_signature(training):
            raise TargetSpecificOOSError(
                f"{side}/{layer}: evaluation target semantics drift from frozen selector"
            )
        declared_training_sha = str(raw.get(f"training_{layer}_target_contract_sha256", ""))
        if declared_training_sha != training.sha256:
            raise TargetSpecificOOSError(
                f"{side}/{layer}: evaluation contract does not bind frozen training contract"
            )
    return base, meta


def _validate_input(
    bundle: StageIAdapterWinnerBundle, source: StageITargetSpecificInput,
) -> tuple[StageITargetContract, StageITargetContract, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    side = str(source.side).lower()
    if side not in _SIDES:
        raise TargetSpecificOOSError("target-specific input must be side-local long/short")
    cell = bundle.cell(side)
    authorization = bundle.joint_finalist_authorization
    declared_authorization = source.source_manifest.get("joint_finalist_authorization")
    if authorization is not None:
        if not isinstance(declared_authorization, Mapping) or canonical_sha256(dict(declared_authorization)) != canonical_sha256(dict(authorization)):
            raise TargetSpecificOOSError(
                f"{side}: source input is not bound to this authorized joint finalist"
            )
        if str(source.source_manifest.get("shared_population_contract_sha256", "")) != str(authorization["shared_population_contract_sha256"]):
            raise TargetSpecificOOSError(
                f"{side}: source input shared universe differs from the authorized winner"
            )
    training_base_contract, training_meta_contract = (
        cell.base_target_contract, cell.meta_target_contract,
    )
    base_contract, meta_contract = _evaluation_target_contracts(
        source.source_manifest, side=side,
        training_base=training_base_contract, training_meta=training_meta_contract,
    )
    permitted_base = {SOFT_SCALAR_S, CUMULATIVE_ORDINAL5_O, LEGACY_R3_MULTICLASS3}
    if base_contract.family not in permitted_base or meta_contract.family != FOLD_QUANTILE_RESIDUAL3:
        raise TargetSpecificOOSError("target-specific route requires S/O/R3 base and FQ3 meta winner")
    is_frozen_r3 = base_contract.family == LEGACY_R3_MULTICLASS3
    if is_frozen_r3 != (source.frozen_base_oof is not None):
        raise TargetSpecificOOSError("only R3 may use the frozen strict-OOF base handoff")
    if base_contract.geometry != meta_contract.geometry:
        raise TargetSpecificOOSError("base/meta winner geometry differs")
    metadata = dict(meta_contract.metadata)
    if metadata.get("meta_target_semantics") != DIRECT_FQ3_SEMANTICS:
        raise TargetSpecificOOSError("FQ3 winner is not bound to direct correctness semantics")
    if metadata.get("base_input_semantics") != DIRECT_BASE_INPUT_SEMANTICS:
        raise TargetSpecificOOSError("FQ3 winner permits a converted base input")
    if canonical_sha256(dict(source.source_manifest)) != str(source.source_manifest_sha256):
        raise TargetSpecificOOSError(f"{side}: source manifest SHA drift")
    frame, contract = source.frame.copy(), source.contract_frame.copy()
    n = len(contract)
    if n < int(source.min_train_rows) or len(frame) != n:
        raise TargetSpecificOOSError(f"{side}: frame/contract rows are insufficient or drift")
    identity = ["candidate_id", "__ts__", "__symbol__"]
    if missing := set(identity).difference(frame.columns):
        raise TargetSpecificOOSError(f"{side}: feature frame lacks exact identity fields: {sorted(missing)}")
    left = frame.loc[:, identity].copy()
    right = contract.loc[:, identity].copy()
    left["__ts__"] = pd.to_datetime(left["__ts__"], utc=True, errors="coerce")
    right["__ts__"] = pd.to_datetime(right["__ts__"], utc=True, errors="coerce")
    if left.isna().any().any() or right.isna().any().any() or not left.equals(right):
        raise TargetSpecificOOSError(f"{side}: feature/contract identity order drift")
    expected_files = dict(source.source_manifest.get("artifact_sha256", {}))
    observed_files = {str(key): str(value) for key, value in dict(source.source_file_sha256).items()}
    if not expected_files or expected_files != observed_files or any(len(value) != 64 for value in observed_files.values()):
        raise TargetSpecificOOSError(f"{side}: source feature/contract artifact hash drift")
    # Completed v1 R3 selectors predate content-bound target contracts and
    # carry explicit all-zero compatibility hashes. Their immutable OOF file,
    # exact identities/timing/economics, native simplex, and strict fold audit
    # are validated separately below. New S/O and any content-bound R3 remain
    # subject to the ordinary exact contract rebind.
    legacy_r3_placeholder = (
        base_contract.family == LEGACY_R3_MULTICLASS3
        and bool(base_contract.metadata.get("schema_v1_compatibility_only", False))
        and {base_contract.identity_sha256, base_contract.target_sha256,
             base_contract.economics_sha256, base_contract.validity_sha256,
             base_contract.weight_sha256} == {"0" * 64}
    )
    if not legacy_r3_placeholder:
        verify_target_contract(contract, base_contract)
    verify_target_contract(contract, meta_contract)
    base_policy = _require_selector(
        source.base_selector_manifest, expected_sha=cell.base_selector_manifest_sha256,
        supplied_sha=source.base_selector_manifest_sha256, side=side, layer="base", target=training_base_contract,
        features=tuple(cell.base_features),
        allow_legacy_r3_rebind=base_contract.family == LEGACY_R3_MULTICLASS3,
    )
    meta_policy = _require_selector(
        source.meta_selector_manifest, expected_sha=cell.meta_selector_manifest_sha256,
        supplied_sha=source.meta_selector_manifest_sha256, side=side, layer="meta", target=training_meta_contract,
        features=tuple(cell.meta_features),
    )
    for feature_set, layer in ((tuple(cell.base_features), "base"), (tuple(cell.meta_features), "meta")):
        if not feature_set or len(set(feature_set)) != len(feature_set):
            raise TargetSpecificOOSError(f"{side}/{layer}: feature contract is not exact/unique")
    missing = set(cell.base_features).difference(frame.columns)
    if missing:
        raise TargetSpecificOOSError(f"{side}/base fields absent: {sorted(missing)[:8]}")
    # The direct base handoff and derived trust values are generated below;
    # every remaining selected meta field must come from the causal source.
    state_width = 3 if is_frozen_r3 else (2 if base_contract.family == SOFT_SCALAR_S else 5)
    required_states = tuple(f"base_state_p{i}" for i in range(state_width))
    required_meta = {"base_raw_score", *required_states, *_TRUST_FIELDS}
    if missing := required_meta.difference(cell.meta_features):
        raise TargetSpecificOOSError(f"{side}/meta lacks required direct base/trust handoffs: {sorted(missing)}")
    if "prequential_base_expected_net_bps" in cell.meta_features:
        raise TargetSpecificOOSError("direct FQ3 meta must not consume a converted base bps field")
    base_source_features, meta_source_features, _role_contract_payload = _role_contract(
        source.source_manifest, side=side,
    )
    month_contract = _month_contract(source.source_manifest, side=side)
    required_regime = tuple(map(str, metadata.get("required_regime_features", ())))
    required_context = tuple(map(str, metadata.get("required_context_features", ())))
    required_trust = tuple(map(str, metadata.get("required_trust_features", ())))
    if not required_regime or not required_context or not required_trust:
        raise TargetSpecificOOSError("FQ3 winner must explicitly bind regime/context/trust feature requirements")
    for label, required in (("regime", required_regime), ("context", required_context), ("trust", required_trust)):
        absent = set(required).difference(cell.meta_features)
        if absent:
            raise TargetSpecificOOSError(f"{side}/meta lacks bound {label} features: {sorted(absent)}")
    _validate_selected_feature_roles(
        side=side, frame=frame, base_features=cell.base_features, meta_features=cell.meta_features,
        base_source_features=base_source_features, meta_source_features=meta_source_features,
        required_generated=required_meta,
    )
    raw_needed = set(cell.meta_features).difference(required_meta)
    if missing := raw_needed.difference(frame.columns):
        raise TargetSpecificOOSError(f"{side}/meta causal fields absent: {sorted(missing)[:8]}")
    for column in (
        "candidate_id", "__ts__", "__symbol__", "side_name", "target_valid",
        source.base_target_column, source.meta_target_column, "gross_bps", "net_bps",
        base_contract.weight_column, meta_contract.weight_column,
    ):
        if column not in contract:
            raise TargetSpecificOOSError(f"{side}: contract frame lacks {column}")
    if not contract.side_name.astype(str).str.lower().eq(side).all():
        raise TargetSpecificOOSError(f"{side}: contract has cross-side rows")
    if contract[["candidate_id", "__ts__", "__symbol__", "side_name"]].isna().any().any() or contract.candidate_id.duplicated().any():
        raise TargetSpecificOOSError(f"{side}: contract identity is invalid")
    decision = _utc(contract.get("decision_ts", contract["__ts__"] + pd.Timedelta(hours=1)), n, name="decision_ts")
    signal = _utc(contract["__ts__"], n, name="signal_ts")
    available = _utc(contract.get("label_available_ts", decision + pd.Timedelta(hours=12)), n, name="label_available_ts")
    if not (decision - signal).eq(pd.Timedelta(hours=1)).all() or not (available - decision).eq(pd.Timedelta(hours=12)).all():
        raise TargetSpecificOOSError("target-specific route requires close -> +1h entry -> +12h availability")
    valid = _array(contract.target_valid, n, name="target_valid", dtype=bool)
    gross = _array(contract.gross_bps, n, name="gross_bps", dtype=np.float32)
    net = _array(contract.net_bps, n, name="net_bps", dtype=np.float32)
    if not np.isfinite(gross[valid]).all() or not np.isfinite(net[valid]).all() or not np.allclose(gross[valid] - 100.0, net[valid], atol=2e-3, rtol=0):
        raise TargetSpecificOOSError("winner geometry economics must apply exactly one 100bps cost")
    weight = _array(contract[base_contract.weight_column], n, name="base sample_weight", dtype=np.float32)
    if not np.isfinite(weight).all() or (weight < 0).any() or not np.any(weight[valid] > 0):
        raise TargetSpecificOOSError("base sample weights are invalid")
    meta_weight = _array(
        contract[meta_contract.weight_column], n, name="meta sample_weight", dtype=np.float32,
    )
    if not np.isfinite(meta_weight).all() or (meta_weight < 0).any() or not np.any(meta_weight[valid] > 0):
        raise TargetSpecificOOSError("meta sample weights are invalid")
    if is_frozen_r3:
        _validate_frozen_r3_handoff(source, side=side, contract=contract)
    return base_contract, meta_contract, frame, contract, {
        "decision": decision, "available": available, "valid": valid, "gross": gross,
        "net": net, "weight": weight, "meta_weight": meta_weight,
        "base_correlation_policy": base_policy, "meta_correlation_policy": meta_policy,
        "causal_feature_role_contract_sha256": source.source_manifest["causal_feature_role_contract_sha256"],
        "evaluation_month_contract": month_contract,
        "evaluation_month_contract_sha256": source.source_manifest["evaluation_month_contract_sha256"],
        "required_regime_features": required_regime, "required_context_features": required_context,
        "required_trust_features": required_trust,
        "base_input_family": base_contract.family,
    }


def preflight_strict_meta_availability(
    bundle: StageIAdapterWinnerBundle,
    inputs: Sequence[StageITargetSpecificInput],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Project strict base/meta availability without fitting either model.

    This is a structural preflight: rows are admitted only if the configured
    chronological blocks have prior-resolved valid support for the meta fit.
    It lets R3/S/O finalists prove identical strict-meta denominators before
    any expensive inference, while keeping all model-dependent economics out
    of the check.
    """
    if len(inputs) != 2 or {str(item.side).lower() for item in inputs} != set(_SIDES):
        raise TargetSpecificOOSError("strict-meta preflight requires exactly long and short inputs")
    records: list[pd.DataFrame] = []
    per_side: dict[str, dict[str, int]] = {}
    shared_contracts = set()
    for source in inputs:
        base_contract, _meta_contract, _frame, contract, vector = _validate_input(bundle, source)
        side = str(source.side).lower()
        decision, available, valid = vector["decision"], vector["available"], vector["valid"]
        blocks = _validation_blocks(
            decision, available, n_folds=int(source.n_validation_folds),
            min_train_rows=int(source.min_train_rows),
        )
        if base_contract.family == LEGACY_R3_MULTICLASS3:
            score, _states, _folds = _validate_frozen_r3_handoff(source, side=side, contract=contract)
            base_available = valid & np.isfinite(score)
        else:
            base_available = np.zeros(len(contract), dtype=bool)
            for validation_idx in blocks:
                base_available[validation_idx] = valid[validation_idx]
        meta_available = np.zeros(len(contract), dtype=bool)
        for validation_idx in blocks:
            start = decision.iloc[validation_idx].min()
            train_idx = np.flatnonzero(available.lt(start).to_numpy() & valid & base_available)
            score_idx = validation_idx[valid[validation_idx] & base_available[validation_idx]]
            if len(train_idx) >= int(source.min_train_rows) and len(score_idx):
                meta_available[score_idx] = True
        projected = contract.loc[:, ["candidate_id", "side_name", "decision_ts"]].copy()
        projected["candidate_id"] = projected.candidate_id.astype(str)
        projected["side_name"] = projected.side_name.astype(str).str.lower()
        projected["decision_ts"] = pd.to_datetime(projected.decision_ts, utc=True, errors="coerce")
        projected["target_valid"] = valid
        projected["base_strict_oof_available"] = base_available
        projected["strict_oof_available"] = meta_available
        if projected.decision_ts.isna().any() or projected.duplicated(["side_name", "candidate_id"]).any():
            raise TargetSpecificOOSError(f"{side}: invalid candidate identity in strict-meta preflight")
        records.append(projected)
        per_side[side] = {
            "rows": int(len(projected)), "valid_rows": int(valid.sum()),
            "base_strict_oof_rows": int(base_available.sum()),
            "strict_meta_rows": int(meta_available.sum()),
        }
        shared_contracts.add(str(source.source_manifest.get("shared_population_contract_sha256", "")))
    if len(shared_contracts) != 1:
        raise TargetSpecificOOSError("strict-meta preflight long/short shared universe drift")
    projection = pd.concat(records, ignore_index=True).sort_values(
        ["side_name", "decision_ts", "candidate_id"], kind="stable",
    ).reset_index(drop=True)
    identity_hash = canonical_sha256(
        projection.astype(str).to_dict(orient="records")
    )
    return projection, {
        "schema": "stage_i_target_specific_strict_meta_preflight_v1",
        "shared_population_contract_sha256": next(iter(shared_contracts)),
        "rows": int(len(projection)), "per_side": per_side,
        "availability_sha256": identity_hash,
        "model_fit_performed": False,
    }


def validate_preflight_strict_meta_availability_equality(
    preflights: Mapping[str, tuple[pd.DataFrame, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Fail closed unless every R3/S/O finalist has the same projected rows."""
    if len(preflights) < 2 or len(set(preflights)) != len(preflights):
        raise TargetSpecificOOSError("strict-meta equality needs at least two uniquely named finalists")
    canonical: pd.DataFrame | None = None
    shared_contract: str | None = None
    hashes: dict[str, str] = {}
    required = {
        "candidate_id", "side_name", "decision_ts", "target_valid",
        "base_strict_oof_available", "strict_oof_available",
    }
    for name, (frame, receipt) in preflights.items():
        if missing := required.difference(frame.columns):
            raise TargetSpecificOOSError(f"{name}: strict-meta preflight lacks {sorted(missing)}")
        declared = str(receipt.get("availability_sha256", ""))
        recomputed = canonical_sha256(
            frame.loc[:, sorted(required)].sort_values(
                ["side_name", "decision_ts", "candidate_id"], kind="stable",
            ).astype(str).to_dict(orient="records")
        )
        # ``preflight_strict_meta_availability`` signs the same canonical
        # fields in stable order; this direct recompute also catches callers
        # attempting to pass an unrelated dataframe/receipt pair.
        if declared != recomputed:
            raise TargetSpecificOOSError(f"{name}: strict-meta preflight receipt checksum drift")
        current_shared = str(receipt.get("shared_population_contract_sha256", ""))
        if shared_contract is None:
            shared_contract = current_shared
        elif current_shared != shared_contract:
            raise TargetSpecificOOSError("strict-meta preflight shared universe differs across finalists")
        comparable = frame.loc[:, sorted(required)].copy()
        comparable["decision_ts"] = pd.to_datetime(comparable.decision_ts, utc=True, errors="coerce")
        comparable = comparable.sort_values(["side_name", "decision_ts", "candidate_id"], kind="stable").reset_index(drop=True)
        if canonical is None:
            canonical = comparable
        elif not canonical.equals(comparable):
            raise TargetSpecificOOSError(
                "finalist preflight must have identical rows/base/meta strict availability"
            )
        hashes[str(name)] = declared
    return {
        "schema": "stage_i_target_specific_strict_meta_equality_v1",
        "status": "complete", "finalists": sorted(map(str, preflights)),
        "shared_population_contract_sha256": shared_contract,
        "availability_sha256": next(iter(hashes.values())),
        "per_finalist_availability_sha256": hashes,
        "model_fit_performed": False,
    }


def generate_stage_i_target_specific_strict_oof(
    bundle: StageIAdapterWinnerBundle, source: StageITargetSpecificInput, *, fit_model: Callable[..., Any],
) -> StageITargetSpecificResult:
    """Create strict chronological direct-base/FQ3 OOF for a single side."""
    base_contract, meta_contract, frame, contract, vector = _validate_input(bundle, source)
    side, n = str(source.side).lower(), len(contract)
    decision, available = vector["decision"], vector["available"]
    valid, gross, net, weight, meta_weight = (
        vector["valid"], vector["gross"], vector["net"], vector["weight"], vector["meta_weight"],
    )
    ids = _array(contract.candidate_id, n, name="candidate_id", dtype=object)
    base_target = _array(contract[source.base_target_column], n, name="base target", dtype=np.float32)
    if not np.isfinite(base_target[valid]).all():
        raise TargetSpecificOOSError("valid base target must be finite")
    blocks = _validation_blocks(decision, available, n_folds=int(source.n_validation_folds), min_train_rows=int(source.min_train_rows))
    is_frozen_r3 = base_contract.family == LEGACY_R3_MULTICLASS3
    state_width = 3 if is_frozen_r3 else (2 if base_contract.family == SOFT_SCALAR_S else 5)
    provenance: list[dict[str, Any]] = []
    if is_frozen_r3:
        base_score, base_states, base_fold = _validate_frozen_r3_handoff(
            source, side=side, contract=contract,
        )
        for item in source.frozen_base_oof_manifest["hpo_oof_regeneration_fold_audit"]:
            provenance.append({
                "side_name": side, "layer": "base", "fold_id": int(item["fold_id"]),
                "train_rows": int(item["train_rows"]), "validation_rows": int(item["validation_rows"]),
                "validation_start_utc": pd.to_datetime(item["validation_start_utc"], utc=True),
                "train_max_label_available_utc": pd.to_datetime(item["train_max_label_available_utc"], utc=True),
                "strict_prior_resolved": True, "target_family": base_contract.family,
                "score_semantics": "native_P(clear)-P(adverse)_no_bps_conversion",
                "frozen_completed_base": True,
            })
    else:
        base_score = np.full(n, np.nan, dtype=np.float32)
        base_states = np.full((n, state_width), np.nan, dtype=np.float32)
        base_fold = np.full(n, -1, dtype=np.int16)
        for fold_id, validation_idx in enumerate(blocks):
            start = decision.iloc[validation_idx].min()
            train_idx = np.flatnonzero(available.lt(start).to_numpy() & valid)
            if len(train_idx) < int(source.min_train_rows):
                raise TargetSpecificOOSError("base fold lacks prior-resolved valid support")
            weight_cfg = dict(base_contract.metadata.get("training_weight_contract") or {})
            fold_weight = training_weights(
                contract.iloc[train_idx], target=base_target[train_idx],
                mode=str(weight_cfg.get("mode", "uniform")), regime_column=str(base_contract.metadata.get("regime_column", "")),
            )
            score, states = _base_fold(
                base_contract, frame.iloc[train_idx].loc[:, list(bundle.cell(side).base_features)], base_target[train_idx], fold_weight,
                frame.iloc[validation_idx].loc[:, list(bundle.cell(side).base_features)], bundle.cell(side).base_params, fit_model,
            )
            base_score[validation_idx] = score
            base_states[validation_idx] = states
            base_fold[validation_idx] = fold_id
            provenance.append({
                "side_name": side, "layer": "base", "fold_id": fold_id, "train_rows": len(train_idx),
                "validation_rows": len(validation_idx), "validation_start_utc": start,
                "train_max_label_available_utc": available.iloc[train_idx].max(), "strict_prior_resolved": bool(available.iloc[train_idx].lt(start).all()),
                "target_family": base_contract.family, "score_semantics": "direct_S_or_O_score_no_bps_conversion",
            })
    base_scored = valid & np.isfinite(base_score)
    if not base_scored.any():
        raise TargetSpecificOOSError("base emitted no strict OOF scores")
    # Generated handoffs only use each row's direct base OOF prediction.
    design = frame.copy()
    design["base_raw_score"] = base_score
    for index in range(state_width):
        design[f"base_state_p{index}"] = base_states[:, index]
    trust = _direct_trust(base_states[base_scored])
    for name in _TRUST_FIELDS:
        values = np.full(n, np.nan, dtype=np.float32)
        values[base_scored] = trust[name].to_numpy(np.float32)
        design[name] = values
    meta_features = tuple(bundle.cell(side).meta_features)
    if missing := set(meta_features).difference(design.columns):
        raise TargetSpecificOOSError(f"meta features absent after direct handoff: {sorted(missing)[:8]}")
    # Detect null/constant selected source features before fit.  Derived score
    # state features have legitimate base burn-in and are assessed only scored.
    for feature in meta_features:
        values = pd.to_numeric(design.loc[base_scored, feature], errors="coerce")
        if values.notna().mean() < 0.90 or values.dropna().nunique() <= 1:
            raise TargetSpecificOOSError(f"{side}/meta selected feature lacks >=90% finite nonconstant OOF coverage: {feature}")
    correction = np.full(n, np.nan, dtype=np.float32)
    combined = np.full(n, np.nan, dtype=np.float32)
    meta_state = np.full((n, 3), np.nan, dtype=np.float32)
    meta_fold = np.full(n, -1, dtype=np.int16)
    states: list[dict[str, Any]] = []
    for fold_id, validation_idx in enumerate(blocks):
        start = decision.iloc[validation_idx].min()
        train_idx = np.flatnonzero(available.lt(start).to_numpy() & valid & base_scored)
        score_idx = validation_idx[valid[validation_idx] & base_scored[validation_idx]]
        if len(train_idx) < int(source.min_train_rows) or not len(score_idx):
            provenance.append({
                "side_name": side, "layer": "meta", "fold_id": fold_id, "train_rows": len(train_idx),
                "validation_rows": len(score_idx), "validation_start_utc": start, "strict_prior_resolved": True,
                "target_family": FOLD_QUANTILE_RESIDUAL3, "skipped": True, "skip_reason": "insufficient_prior_direct_base_oof_support",
            })
            continue
        labels, state = _fit_direct_correctness(
            net[train_idx], base_score[train_idx], score_domain=(-1.0, 1.0) if is_frozen_r3 else (0.0, 1.0),
        )
        model = fit_model(
            design.iloc[train_idx].loc[:, list(meta_features)], labels, meta_weight[train_idx], classifier=True,
            params=_clean_params(bundle.cell(side).meta_params, objective="multiclass", num_class=3),
            objective_mode="stage_i_target_specific_direct_FQ3",
        )
        probability = _multiclass_probabilities(
            model, design.iloc[score_idx].loc[:, list(meta_features)]
        )
        fold_correction, fold_combined = _reconstruct_direct_correctness(probability, base_score[score_idx], state)
        correction[score_idx], combined[score_idx], meta_state[score_idx], meta_fold[score_idx] = fold_correction, fold_combined, probability, fold_id
        states.append({"fold_id": fold_id, **state.to_dict()})
        provenance.append({
            "side_name": side, "layer": "meta", "fold_id": fold_id, "train_rows": len(train_idx),
            "validation_rows": len(score_idx), "validation_start_utc": start,
            "train_max_label_available_utc": available.iloc[train_idx].max(), "strict_prior_resolved": bool(available.iloc[train_idx].lt(start).all()),
            "target_family": FOLD_QUANTILE_RESIDUAL3, "target_semantics": DIRECT_FQ3_SEMANTICS,
            "base_input_semantics": DIRECT_BASE_INPUT_SEMANTICS, "skipped": False,
        })
    strict = valid & np.isfinite(combined)
    output = contract.loc[:, ["candidate_id", "__ts__", "__symbol__", "side_name"]].copy()
    output["decision_ts"], output["label_available_ts"] = decision, available
    output["target_valid"], output["exact_gross_bps"], output["exact_net_bps"] = valid, gross, net
    output["base_fold_id"], output["meta_fold_id"] = base_fold, meta_fold
    output["base_strict_oof_available"], output["strict_oof_available"] = base_scored, strict
    output["base_direct_score"], output["meta_direct_correction"], output["meta_direct_score"] = base_score, correction, combined
    for index in range(state_width):
        output[f"base_state_p{index}"] = base_states[:, index]
    output[["meta_p_overestimating", "meta_p_approximately_right", "meta_p_underestimating"]] = meta_state
    output[["meta_p_error_tercile_0", "meta_p_error_tercile_1", "meta_p_error_tercile_2"]] = meta_state
    manifest = {
        "schema": SCHEMA, "status": "complete", "side": side, "rows": n,
        "base_target_contract_sha256": base_contract.sha256, "meta_target_contract_sha256": meta_contract.sha256,
        "winner_geometry": base_contract.geometry,
        "base_correlation_policy": vector["base_correlation_policy"],
        "meta_correlation_policy": vector["meta_correlation_policy"],
        "causal_feature_role_contract_sha256": vector["causal_feature_role_contract_sha256"],
        "evaluation_month_contract_sha256": vector["evaluation_month_contract_sha256"],
        "evaluation_month_contract": vector["evaluation_month_contract"],
        "base_score": (
            "same-side native R3 P(clear)-P(adverse) in [-1,1]; frozen strict OOF; no bps conversion before meta"
            if is_frozen_r3 else "same-side direct S/O output in [0,1]; no bps conversion before meta"
        ),
        "base_only_economics_disposition": "diagnostic_only_never_terminal",
        "terminal_economics_layer": "joint_reconstructed_meta_after_causal_common_bps_mapping",
        "meta_target": DIRECT_FQ3_SEMANTICS, "meta_input": DIRECT_BASE_INPUT_SEMANTICS,
        "meta_probability_semantics": (
            "neutral_error_terciles_0_1_2; per-fold thresholds determine whether legacy "
            "overestimating/approximately_right/underestimating labels are literal"
        ),
        "meta_fold_states": states, "strict_meta_rows": int(strict.sum()),
        "source_manifest_sha256": source.source_manifest_sha256,
    }
    return StageITargetSpecificResult(side=side, predictions=output, fold_provenance=pd.DataFrame(provenance), manifest=manifest)


def _side_month_raw_metrics(frame: pd.DataFrame, *, layer: str, score: str) -> pd.DataFrame:
    work = frame.loc[frame[score].notna() & frame.target_valid.astype(bool)].copy()
    if work.empty:
        return pd.DataFrame()
    work["month"] = pd.to_datetime(work.decision_ts, utc=True).dt.strftime("%Y-%m")
    rows: list[dict[str, Any]] = []
    for (side, month), group in work.groupby(["side_name", "month"], observed=True):
        rows.append({
            "layer": layer, "score_stage": "raw_direct", "selection": "unranked_side_month_diagnostic",
            "side_name": side, "month": month, "rows": len(group),
            "mean_gross_bps": float(group.exact_gross_bps.mean()), "mean_net_bps": float(group.exact_net_bps.mean()),
            "score_mean": float(group[score].mean()), "score_std": float(group[score].std(ddof=0)),
        })
    rows.append({
        "layer": layer, "score_stage": "raw_direct", "selection": "unranked_pooled_diagnostic",
        "side_name": "pooled_global", "month": "all", "rows": len(work),
        "mean_gross_bps": float(work.exact_gross_bps.mean()), "mean_net_bps": float(work.exact_net_bps.mean()),
        "score_mean": float(work[score].mean()), "score_std": float(work[score].std(ddof=0)),
    })
    return pd.DataFrame(rows)


def _mapped_global_metrics(frame: pd.DataFrame, *, layer: str, map_prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Global only after common-bps mapping; side/month entries are attribution."""
    mapped = f"{map_prefix}expected_net_bps"
    admitted = f"{map_prefix}admitted"
    work = frame.loc[frame[mapped].notna() & frame.target_valid.astype(bool)].copy()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame()
    work["month"] = pd.to_datetime(work.decision_ts, utc=True).dt.strftime("%Y-%m")
    records: list[dict[str, Any]] = []
    worst: list[dict[str, Any]] = []
    for mode, population in (("mapped_no_admission", work), ("mapped_21d_admitted", work.loc[work[admitted].astype(bool)])):
        for tail in _TAILS:
            requested = max(1, int(math.ceil(tail * len(work))))
            selected = stable_stage_i_rank_frame(population, score_column=mapped).head(min(requested, len(population)))
            records.append({
                "layer": layer, "score_stage": "causal_21d_common_bps", "selection": mode,
                "selection_scope": "pooled_global_after_side_local_common_bps_mapping",
                "side_name": "pooled_global", "month": "all", "top_fraction_of_full_scored_population": tail,
                "requested_rows": requested, "selected_rows": len(selected),
                "mean_gross_bps": float(selected.exact_gross_bps.mean()) if len(selected) else np.nan,
                "mean_net_bps": float(selected.exact_net_bps.mean()) if len(selected) else np.nan,
            })
            for (side, month), part in selected.groupby(["side_name", "month"], observed=True):
                records.append({
                    "layer": layer, "score_stage": "causal_21d_common_bps", "selection": mode,
                    "selection_scope": "pooled_global_after_side_local_common_bps_mapping",
                    "side_name": side, "month": month, "top_fraction_of_full_scored_population": tail,
                    "requested_rows": requested, "selected_rows": len(part),
                    "mean_gross_bps": float(part.exact_gross_bps.mean()), "mean_net_bps": float(part.exact_net_bps.mean()),
                })
            monthly = selected.groupby("month", observed=True).exact_net_bps.agg(["mean", "size"])
            if len(monthly):
                worst_month = monthly["mean"].idxmin()
                worst.append({
                    "layer": layer, "selection": mode, "top_fraction_of_full_scored_population": tail,
                    "worst_month": worst_month, "worst_month_net_bps": float(monthly.loc[worst_month, "mean"]),
                    "worst_month_selected_rows": int(monthly.loc[worst_month, "size"]),
                    "eligible_rows": len(population), "requested_rows": requested, "selected_rows": len(selected),
                })
    return pd.DataFrame(records), pd.DataFrame(worst)


def _coverage_audit(
    evaluation: pd.DataFrame, *, results: Sequence[StageITargetSpecificResult],
) -> pd.DataFrame:
    """Audit all 36 months × sides before any stack can be promoted.

    Source availability is a frozen contract, not inferred from a silently
    absent parquet partition.  A declared available month with zero candidate,
    valid, base-OOF, or meta-OOF rows is a promotion failure unless the source
    explicitly documents the zero-coverage exception and its reason.
    """
    contracts = {item.side: item.manifest["evaluation_month_contract"] for item in results}
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    work = evaluation.copy()
    work["month"] = pd.to_datetime(work.decision_ts, utc=True).dt.strftime("%Y-%m")
    for side in _SIDES:
        availability = contracts[side]["source_availability"]
        side_rows = work.loc[work.side_name.eq(side)]
        for month in _EVALUATION_MONTHS:
            part = side_rows.loc[side_rows.month.eq(month)]
            available = bool(availability[month]["source_available"])
            allowed_zero = bool(availability[month].get("allow_zero_strict_coverage", False))
            candidate_rows = int(len(part))
            valid_rows = int(part.target_valid.astype(bool).sum())
            base_rows = int(part.base_strict_oof_available.astype(bool).sum())
            meta_rows = int(part.strict_oof_available.astype(bool).sum())
            for layer, mapped_col, admitted_col in (
                ("base", "base_causal_21d_expected_net_bps", "base_causal_21d_admitted"),
                ("meta", "meta_causal_21d_expected_net_bps", "meta_causal_21d_admitted"),
            ):
                mapped_rows = int(part[mapped_col].notna().sum())
                admitted_rows = int(part[admitted_col].astype(bool).sum())
                status = "pass"
                reason = ""
                if not available and candidate_rows:
                    status, reason = "fail", "rows_present_in_declared_source_gap"
                elif available and not candidate_rows:
                    status, reason = "fail", "missing_declared_available_candidate_month"
                elif available and not allowed_zero and min(valid_rows, base_rows, meta_rows) == 0:
                    status, reason = "fail", "undeclared_zero_valid_or_strict_oof_coverage"
                if status == "fail":
                    failures.append({"side_name": side, "month": month, "layer": layer, "reason": reason})
                records.append({
                    "side_name": side, "month": month, "layer": layer,
                    "source_available": available, "source_gap_reason": availability[month].get("source_gap_reason", ""),
                    "allow_zero_strict_coverage": allowed_zero,
                    "zero_coverage_reason": availability[month].get("zero_coverage_reason", ""),
                    "candidate_rows": candidate_rows, "valid_rows": valid_rows,
                    "base_strict_oof_rows": base_rows, "meta_strict_oof_rows": meta_rows,
                    "mapped_rows": mapped_rows, "admitted_rows": admitted_rows,
                    "promotion_coverage_status": status, "promotion_coverage_reason": reason,
                })
    if failures:
        raise TargetSpecificOOSError(f"promotion coverage gate failed: {failures[:8]}")
    return pd.DataFrame(records)


def compare_target_specific_finalists(
    finalists: Sequence[StageITargetSpecificFinalist],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare finalist **joint stacks** on identical evaluation rows.

    This helper accepts an R3, S, or O base finalist as long as its published
    ledger has the same reconstructed meta common-bps coordinates.  It does
    not read, rank, or gate any base-only score.  Candidate identities,
    decision timestamps, target validity, and strict-meta availability must
    match exactly.  Realised economics may legitimately differ across R3/S/O
    exit geometries, so each finalist is validated against its own exactly-once
    cost contract rather than against another finalist's path outcome.
    """
    if not finalists or len({item.name for item in finalists}) != len(finalists):
        raise TargetSpecificOOSError("finalist scoring requires uniquely named joint stacks")
    if len(finalists) > 1:
        shared_contracts = {
            str(item.manifest.get("shared_population_contract_sha256", ""))
            for item in finalists
        }
        if len(shared_contracts) != 1 or not re.fullmatch(r"[0-9a-f]{64}", next(iter(shared_contracts))):
            raise TargetSpecificOOSError(
                "multi-finalist comparison requires one identical signed shared-population contract"
            )
    required = {
        "candidate_id", "side_name", "decision_ts", "target_valid", "exact_gross_bps", "exact_net_bps",
        "strict_oof_available", "meta_causal_21d_expected_net_bps", "meta_causal_21d_admitted",
    }
    canonical: pd.DataFrame | None = None
    rows: list[dict[str, Any]] = []
    attribution: list[dict[str, Any]] = []
    for finalist in finalists:
        frame = finalist.predictions.copy()
        if missing := required.difference(frame.columns):
            raise TargetSpecificOOSError(f"{finalist.name}: missing joint-stack comparison fields: {sorted(missing)}")
        frame["candidate_key"] = frame.side_name.astype(str) + "::" + frame.candidate_id.astype(str)
        if frame.candidate_key.duplicated().any():
            raise TargetSpecificOOSError(f"{finalist.name}: duplicate qualified candidate identity")
        frame["decision_ts"] = pd.to_datetime(frame.decision_ts, utc=True, errors="coerce")
        if frame.decision_ts.isna().any():
            raise TargetSpecificOOSError(f"{finalist.name}: invalid decision timestamps")
        frame = frame.loc[frame.decision_ts.ge(_EVALUATION_START) & frame.decision_ts.lt(_EVALUATION_END_EXCLUSIVE)].copy()
        valid = frame.target_valid.astype(bool)
        gross = pd.to_numeric(frame.exact_gross_bps, errors="coerce")
        net = pd.to_numeric(frame.exact_net_bps, errors="coerce")
        if not np.isfinite(gross.loc[valid]).all() or not np.isfinite(net.loc[valid]).all() or not np.allclose(
            gross.loc[valid].to_numpy(float) - 100.0,
            net.loc[valid].to_numpy(float), atol=2e-3, rtol=0.0,
        ):
            raise TargetSpecificOOSError(
                f"{finalist.name}: own declared economics do not apply 100bps cost exactly once"
            )
        key_cols = ["candidate_key", "decision_ts", "target_valid", "strict_oof_available"]
        comparable = frame.loc[:, key_cols].sort_values("candidate_key", kind="stable").reset_index(drop=True)
        if canonical is None:
            canonical = comparable
        elif not canonical.equals(comparable):
            raise TargetSpecificOOSError("finalist stacks must use identical rows/strict-meta availability")
        eligible = frame.loc[
            frame.target_valid.astype(bool)
            & frame.strict_oof_available.astype(bool)
            & frame.meta_causal_21d_expected_net_bps.notna()
            & frame.meta_causal_21d_admitted.astype(bool)
        ].copy()
        if eligible.empty:
            rows.append({
                "finalist": finalist.name, "joint_stack_only": True, "promotion_eligible": False,
                "promotion_reason": "no_admitted_reconstructed_meta_rows", "joint_promotion_score_bps": np.nan,
                "joint_top10_net_bps": np.nan, "joint_worst_month_top10_net_bps": np.nan,
            })
            continue
        ordered = stable_stage_i_rank_frame(eligible, score_column="meta_causal_21d_expected_net_bps")
        tail_net: dict[float, float] = {}
        tail_selection: dict[float, pd.DataFrame] = {}
        denominator = int(
            (frame.target_valid.astype(bool) & frame.strict_oof_available.astype(bool)).sum()
        )
        for tail in _TAILS:
            selected = ordered.head(min(max(1, int(math.ceil(tail * denominator))), len(ordered)))
            tail_selection[tail] = selected
            tail_net[tail] = float(selected.exact_net_bps.mean()) if len(selected) else np.nan
            for (side, month), part in selected.assign(month=selected.decision_ts.dt.strftime("%Y-%m")).groupby(["side_name", "month"], observed=True):
                attribution.append({
                    "finalist": finalist.name, "joint_stack_only": True, "tail": tail,
                    "side_name": side, "month": month, "selected_rows": len(part),
                    "mean_net_bps": float(part.exact_net_bps.mean()), "mean_gross_bps": float(part.exact_gross_bps.mean()),
                })
        top10 = tail_selection[0.10]
        monthly = top10.assign(month=top10.decision_ts.dt.strftime("%Y-%m")).groupby("month", observed=True).exact_net_bps.mean()
        worst = float(monthly.min()) if len(monthly) else np.nan
        # Score and gate are intentionally constructed only from reconstructed
        # meta economics.  The coefficients are frozen reporting weights, not
        # an optimisation surface: breadth (10%) dominates, with tail and
        # worst-period penalties preventing a one-month-only promotion.
        score = 0.20 * tail_net[0.01] + 0.30 * tail_net[0.05] + 0.50 * tail_net[0.10] - 0.25 * max(0.0, -worst)
        promote = bool(np.isfinite(score) and tail_net[0.10] > 0.0 and worst >= 0.0)
        rows.append({
            "finalist": finalist.name, "joint_stack_only": True, "strict_meta_rows": denominator,
            "admitted_reconstructed_meta_rows": len(eligible), "joint_top1_net_bps": tail_net[0.01],
            "joint_top5_net_bps": tail_net[0.05], "joint_top10_net_bps": tail_net[0.10],
            "joint_top20_net_bps": tail_net[0.20], "joint_worst_month_top10_net_bps": worst,
            "joint_promotion_score_bps": score, "promotion_eligible": promote,
            "promotion_reason": "joint_top10_and_worst_month_nonnegative" if promote else "joint_meta_gate_not_met",
            "ranking": "pooled global after causal common-bps mapping; never base-only or per timestamp",
        })
    return pd.DataFrame(rows), pd.DataFrame(attribution)


def _require_sha256(value: Any, *, label: str) -> str:
    text = str(value)
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise TargetSpecificOOSError(f"{label} must be a SHA-256 hex digest")
    return text


def _manifest_declared_artifact_hash(
    manifest: Mapping[str, Any], *, artifact_path: str, label: str,
) -> str:
    """Return one manifest-bound checksum for an exact frozen source file.

    Older artifacts use ``artifacts`` while newer ones use ``files``.  Both
    formats may use relative paths or just basenames; an unrelated checksum is
    never accepted simply because its value happens to equal the input's hash.
    """
    requested = str(artifact_path).replace("\\", "/").lstrip("./")
    basename = Path(requested).name
    matches: list[str] = []

    def path_matches(value: Any) -> bool:
        path = str(value).replace("\\", "/").lstrip("./")
        return path == requested or Path(path).name == basename

    def value_hash(value: Any) -> str | None:
        if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
            return value
        if isinstance(value, Mapping):
            for key in ("sha256", "checksum", "hash", "file_sha256"):
                candidate = value.get(key)
                if isinstance(candidate, str) and re.fullmatch(r"[0-9a-f]{64}", candidate):
                    return candidate
        return None

    for container_name in ("files", "artifacts"):
        container = manifest.get(container_name)
        if not isinstance(container, Mapping):
            continue
        for key, value in container.items():
            matched = path_matches(key)
            if isinstance(value, Mapping) and "path" in value:
                matched = matched or path_matches(value["path"])
            if matched:
                declared = value_hash(value)
                if declared is None:
                    raise TargetSpecificOOSError(
                        f"{label}: manifest declaration for {artifact_path} lacks SHA-256"
                    )
                matches.append(declared)
    if not matches:
        raise TargetSpecificOOSError(
            f"{label}: source manifest has no declared hash for {artifact_path}"
        )
    if len(matches) != 1:
        raise TargetSpecificOOSError(
            f"{label}: source manifest has ambiguous declarations for {artifact_path}"
        )
    return _require_sha256(matches[0], label=f"{label} declared artifact hash")


def _require_canonical_21d_mapping(manifest: Mapping[str, Any], *, name: str) -> None:
    """Reject comparator inputs that were not mapped/admitted causally first."""
    mapping = str(manifest.get("mapping", "")).lower()
    ranking = str(manifest.get("ranking", manifest.get("global_ranking", ""))).lower()
    spec = manifest.get("admission_spec", {})
    window = spec.get("window_days") if isinstance(spec, Mapping) else None
    # Frozen legacy R3 admission artifacts called this a "21-day trailing"
    # map and record no dataclass payload.  They remain eligible only when the
    # normalizer records the canonical explicit 21-day contract below.
    if window != 21:
        raise TargetSpecificOOSError(f"{name}: manifest does not bind canonical 21-day admission")
    if not all(term in mapping for term in ("causal", "21-day", "side")):
        raise TargetSpecificOOSError(f"{name}: manifest mapping is not causal side-local 21-day semantics")
    if "common-bps" not in mapping and "common bps" not in mapping:
        raise TargetSpecificOOSError(f"{name}: manifest mapping does not bind common-bps output")
    if "pooled global after" not in ranking or "never per timestamp" not in ranking:
        raise TargetSpecificOOSError(f"{name}: manifest does not bind post-map pooled-global ranking")


def _declared_target_geometries(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    """Extract only explicit target/exit geometry declarations from lineage."""
    values: list[str] = []
    declared = manifest.get("declared_target_geometry")
    if isinstance(declared, Sequence) and not isinstance(declared, (str, bytes)):
        values.extend(str(candidate).strip() for candidate in declared if str(candidate).strip())
    for candidate in (declared, manifest.get("geometry")):
        if isinstance(candidate, str) and candidate.strip():
            values.append(candidate.strip())
    contract = manifest.get("contract")
    if isinstance(contract, Mapping):
        candidate = contract.get("geometry")
        if isinstance(candidate, str) and candidate.strip():
            values.append(candidate.strip())
    winner = manifest.get("winner_bundle")
    if isinstance(winner, Mapping):
        cells = winner.get("cells", ())
        if isinstance(cells, Sequence) and not isinstance(cells, (str, bytes)):
            for cell in cells:
                if not isinstance(cell, Mapping):
                    continue
                for target_key in ("base_target_contract", "meta_target_contract"):
                    target = cell.get(target_key)
                    if isinstance(target, Mapping) and isinstance(target.get("geometry"), str):
                        values.append(str(target["geometry"]).strip())
    return tuple(sorted({value for value in values if value}))


def validate_target_specific_coverage_audit(
    coverage: pd.DataFrame, *, name: str,
) -> None:
    """Validate the frozen 2024--26 side/month coverage proof.

    A joint-meta finalist needs one meta row for every side/month (72 rows).
    The native direct-FQ3 evaluator additionally emits base rows, yielding
    144 rows.  Both forms are accepted, but a partial layer can never hide an
    unavailable month.
    """
    required = {
        "side_name", "month", "promotion_coverage_status", "source_available",
        "source_gap_reason",
    }
    missing = required.difference(coverage.columns)
    if missing:
        raise TargetSpecificOOSError(f"{name}: coverage audit lacks fields: {sorted(missing)}")
    frame = coverage.copy()
    frame["side_name"] = frame.side_name.astype(str).str.lower()
    frame["month"] = frame.month.astype(str)
    if not set(frame.side_name).issubset(set(_SIDES)) or not set(frame.month).issubset(set(_EVALUATION_MONTHS)):
        raise TargetSpecificOOSError(f"{name}: coverage audit has out-of-contract side/month values")
    if not frame.promotion_coverage_status.astype(str).eq("pass").all():
        raise TargetSpecificOOSError(f"{name}: coverage audit contains a promotion failure")
    unavailable = ~frame.source_available.astype(bool)
    if unavailable.any() and frame.loc[unavailable, "source_gap_reason"].astype(str).str.strip().eq("").any():
        raise TargetSpecificOOSError(f"{name}: declared source gaps need explicit reasons")
    if "layer" not in frame:
        if len(frame) != 72 or frame.duplicated(["side_name", "month"]).any():
            raise TargetSpecificOOSError(f"{name}: coverage audit must contain exactly 72 side/month cells")
        expected = {(side, month) for side in _SIDES for month in _EVALUATION_MONTHS}
        if set(map(tuple, frame[["side_name", "month"]].to_numpy())) != expected:
            raise TargetSpecificOOSError(f"{name}: coverage audit is incomplete")
        return
    frame["layer"] = frame.layer.astype(str).str.lower()
    if not set(frame.layer).issubset({"base", "meta"}):
        raise TargetSpecificOOSError(f"{name}: coverage audit has an unsupported layer")
    meta = frame.loc[frame.layer.eq("meta")]
    expected = {(side, month) for side in _SIDES for month in _EVALUATION_MONTHS}
    if len(meta) != 72 or meta.duplicated(["side_name", "month"]).any() or set(map(tuple, meta[["side_name", "month"]].to_numpy())) != expected:
        raise TargetSpecificOOSError(f"{name}: coverage audit must contain all 72 meta side/month cells")
    if "base" in set(frame.layer):
        base = frame.loc[frame.layer.eq("base")]
        if len(base) != 72 or base.duplicated(["side_name", "month"]).any() or set(map(tuple, base[["side_name", "month"]].to_numpy())) != expected:
            raise TargetSpecificOOSError(f"{name}: coverage audit base layer is incomplete")
    if len(frame) not in (72, 144):
        raise TargetSpecificOOSError(f"{name}: coverage audit has unexpected row count {len(frame)}")


def load_target_specific_finalist_artifact(
    root: str | Path, *, name: str,
) -> StageITargetSpecificFinalist:
    """Load a complete immutable finalist after checking its file lineage."""
    directory = Path(root)
    manifest_path = directory / "manifest.json"
    prediction_path = directory / "strict_oof_predictions.parquet"
    coverage_path = directory / "2024_2026_side_month_coverage_audit.parquet"
    if not manifest_path.is_file() or not prediction_path.is_file() or not coverage_path.is_file():
        raise FileNotFoundError(
            f"{name}: expected manifest, strict OOF ledger and 2024--26 coverage audit under {directory}"
        )
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or str(raw.get("status", "")) != "complete":
        raise TargetSpecificOOSError(f"{name}: finalist artifact is incomplete")
    files = raw.get("files")
    if not isinstance(files, Mapping):
        raise TargetSpecificOOSError(f"{name}: manifest lacks immutable output file hashes")
    for path in (prediction_path, coverage_path):
        expected = _require_sha256(files.get(path.name, ""), label=f"{name}:{path.name} manifest hash")
        actual = file_sha256(path)
        if actual != expected:
            raise TargetSpecificOOSError(f"{name}: manifest checksum drift for {path.name}")
    _require_canonical_21d_mapping(raw, name=name)
    if not _declared_target_geometries(raw):
        raise TargetSpecificOOSError(f"{name}: manifest does not declare its own target/exit geometry")
    coverage = pd.read_parquet(coverage_path)
    validate_target_specific_coverage_audit(coverage, name=name)
    return StageITargetSpecificFinalist(name=name, predictions=pd.read_parquet(prediction_path), manifest=raw)


def _normalizer_column(frame: pd.DataFrame, names: Sequence[str], *, name: str, artifact: str) -> str:
    for candidate in names:
        if candidate in frame.columns:
            return candidate
    raise TargetSpecificOOSError(f"{artifact}: missing required {name}; tried {list(names)}")


def normalize_frozen_r3_finalist(
    source: FrozenR3FinalistInput,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Adapt frozen R3 OOF + already-mapped admission into comparator schema.

    No model, calibration map, admission threshold, target, or feature is
    fitted here.  Every transformed value is a copied identity/economic value
    or a direct rename from the supplied frozen artifacts.
    """
    for label, value in (
        ("strict OOF file", source.strict_oof_file_sha256),
        ("strict OOF manifest", source.strict_oof_manifest_sha256),
        ("admission file", source.admission_file_sha256),
        ("admission manifest", source.admission_manifest_sha256),
        ("coverage audit", source.coverage_audit_sha256),
    ):
        _require_sha256(value, label=label)
    if str(source.strict_oof_manifest.get("status", "")) != "complete":
        raise TargetSpecificOOSError("frozen R3 strict OOF manifest is incomplete")
    if str(source.admission_manifest.get("status", "complete")) not in {"complete", ""}:
        raise TargetSpecificOOSError("frozen R3 admission manifest is incomplete")
    strict_declared_hash = _manifest_declared_artifact_hash(
        source.strict_oof_manifest,
        artifact_path=source.strict_oof_artifact_path,
        label="frozen R3 strict OOF",
    )
    if strict_declared_hash != source.strict_oof_file_sha256:
        raise TargetSpecificOOSError("frozen R3 strict OOF manifest hash does not match supplied ledger")
    admission_declared_hash = _manifest_declared_artifact_hash(
        source.admission_manifest,
        artifact_path=source.admission_artifact_path,
        label="frozen R3 admission",
    )
    if admission_declared_hash != source.admission_file_sha256:
        raise TargetSpecificOOSError("frozen R3 admission manifest hash does not match supplied ledger")
    if not _declared_target_geometries(source.strict_oof_manifest):
        raise TargetSpecificOOSError("frozen R3 strict OOF manifest does not declare target/exit geometry")
    legacy_mapping = str(source.admission_manifest.get("mapping", "")).lower()
    if not all(term in legacy_mapping for term in ("21-day", "side-local")):
        raise TargetSpecificOOSError("frozen R3 admission is not a side-local 21-day map")
    threshold = source.admission_manifest.get("threshold_bps", 50.0)
    if not np.isclose(float(threshold), 50.0):
        raise TargetSpecificOOSError("frozen R3 admission threshold must be 50bps")
    strict = source.strict_oof_predictions.copy()
    admission = source.admission_predictions.copy()
    strict_candidate = _normalizer_column(strict, ("candidate_id",), name="candidate identity", artifact="frozen R3 strict OOF")
    strict_side = _normalizer_column(strict, ("side_name",), name="side", artifact="frozen R3 strict OOF")
    strict_time = _normalizer_column(strict, ("decision_ts", "__ts__"), name="decision/signal timestamp", artifact="frozen R3 strict OOF")
    strict_gross = _normalizer_column(strict, ("exact_gross_bps", "gross_bps"), name="gross economics", artifact="frozen R3 strict OOF")
    strict_net = _normalizer_column(strict, ("exact_net_bps", "net_bps"), name="net economics", artifact="frozen R3 strict OOF")
    strict_valid = _normalizer_column(strict, ("target_valid",), name="target-valid flag", artifact="frozen R3 strict OOF")
    strict_oof = _normalizer_column(strict, ("strict_oof_available",), name="strict-OOF flag", artifact="frozen R3 strict OOF")
    admission_candidate = _normalizer_column(admission, ("candidate_id",), name="candidate identity", artifact="frozen R3 admission")
    admission_side = _normalizer_column(admission, ("side_name",), name="side", artifact="frozen R3 admission")
    expected = _normalizer_column(admission, ("causal_21d_side_expected_net_bps", "causal_expected_net_bps"), name="causal expected-net map", artifact="frozen R3 admission")
    admitted = _normalizer_column(admission, ("causal_21d_side_admitted_ge_50bps", "admitted"), name="causal admission flag", artifact="frozen R3 admission")
    strict = pd.DataFrame({
        "candidate_id": strict[strict_candidate].astype(str), "side_name": strict[strict_side].astype(str).str.lower(),
        "decision_ts": pd.to_datetime(strict[strict_time], utc=True, errors="coerce"),
        "target_valid": strict[strict_valid].astype(bool), "strict_oof_available": strict[strict_oof].astype(bool),
        "exact_gross_bps": pd.to_numeric(strict[strict_gross], errors="coerce"), "exact_net_bps": pd.to_numeric(strict[strict_net], errors="coerce"),
    })
    if strict.decision_ts.isna().any() or strict.side_name.isin(_SIDES).eq(False).any():
        raise TargetSpecificOOSError("frozen R3 strict OOF has invalid decision timestamps or sides")
    strict["candidate_key"] = strict.side_name + "::" + strict.candidate_id
    if strict.candidate_key.duplicated().any():
        raise TargetSpecificOOSError("frozen R3 strict OOF has duplicate qualified identities")
    admission = pd.DataFrame({
        "candidate_id": admission[admission_candidate].astype(str), "side_name": admission[admission_side].astype(str).str.lower(),
        "meta_causal_21d_expected_net_bps": pd.to_numeric(admission[expected], errors="coerce"),
        "meta_causal_21d_admitted": admission[admitted].astype(bool),
    })
    admission["candidate_key"] = admission.side_name + "::" + admission.candidate_id
    if admission.candidate_key.duplicated().any() or set(admission.candidate_key) != set(strict.candidate_key):
        raise TargetSpecificOOSError("frozen R3 admission identities do not exactly match strict OOF")
    output = strict.merge(admission.loc[:, ["candidate_key", "meta_causal_21d_expected_net_bps", "meta_causal_21d_admitted"]], on="candidate_key", how="left", validate="one_to_one")
    if output.loc[~output.meta_causal_21d_expected_net_bps.notna(), "meta_causal_21d_admitted"].any():
        raise TargetSpecificOOSError("frozen R3 admission accepts an unmapped row")
    output = output.drop(columns="candidate_key")
    validate_target_specific_coverage_audit(source.coverage_audit, name="frozen R3 source")
    lineage = {
        "schema": FROZEN_R3_NORMALIZER_SCHEMA,
        "operation": "schema_only_adapter_no_fitting_no_mapping_no_tuning",
        "strict_oof_file_sha256": source.strict_oof_file_sha256,
        "strict_oof_manifest_sha256": source.strict_oof_manifest_sha256,
        "admission_file_sha256": source.admission_file_sha256,
        "admission_manifest_sha256": source.admission_manifest_sha256,
        "coverage_audit_sha256": source.coverage_audit_sha256,
        "strict_oof_artifact_path": source.strict_oof_artifact_path,
        "strict_oof_manifest_declared_sha256": strict_declared_hash,
        "admission_artifact_path": source.admission_artifact_path,
        "admission_manifest_declared_sha256": admission_declared_hash,
        "declared_target_geometry": _declared_target_geometries(source.strict_oof_manifest),
    }
    return output, source.coverage_audit.copy(), lineage


def write_frozen_r3_finalist_normalizer(
    *, source: FrozenR3FinalistInput, output_dir: str | Path, finalist_name: str = "R3",
) -> Mapping[str, Any]:
    """Materialize a new immutable comparator-ready R3 adapter artifact."""
    root = Path(output_dir)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite frozen R3 normalizer output: {root}")
    predictions, coverage, lineage = normalize_frozen_r3_finalist(source)
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.tmp-", dir=root.parent))
    try:
        prediction_path = temporary / "strict_oof_predictions.parquet"
        coverage_path = temporary / "2024_2026_side_month_coverage_audit.parquet"
        predictions.to_parquet(prediction_path, index=False, compression="zstd")
        coverage.to_parquet(coverage_path, index=False, compression="zstd")
        manifest = {
            "schema": FROZEN_R3_NORMALIZER_SCHEMA, "status": "complete", "finalist_name": str(finalist_name),
            "base_family": "R3", "source_lineage": lineage,
            "declared_target_geometry": list(lineage["declared_target_geometry"]),
            "mapping": "causal prior-resolved 21-day side-local map with pooled-parent common-bps output",
            "ranking": "only pooled global after side-local causal common-bps mapping; never per timestamp",
            "admission_spec": {"window_days": 21, "net_floor_bps": 50.0, "frozen_source": True},
            "normalizer": "schema-only; no fitting, no map refit, no threshold tuning, no target tuning",
            "files": {
                prediction_path.name: file_sha256(prediction_path),
                coverage_path.name: file_sha256(coverage_path),
            },
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, root)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _prefix_admission_columns(mapped: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    rename = {
        "causal_21d_side_expected_net_bps": f"{prefix}expected_net_bps",
        "causal_21d_side_admitted_ge_50bps": f"{prefix}admitted",
    }
    return mapped.rename(columns=rename)


def run_stage_i_target_specific_oos(
    *, bundle: StageIAdapterWinnerBundle, inputs: Sequence[StageITargetSpecificInput], output_dir: str | Path,
    fit_model: Callable[..., Any], admission_spec: Causal21dAdmissionSpec = Causal21dAdmissionSpec(),
) -> Mapping[str, Any]:
    """Run the target-specific 2024--26 strict OOF path and write diagnostics."""
    if len(inputs) != 2 or {str(item.side).lower() for item in inputs} != set(_SIDES):
        raise TargetSpecificOOSError("target-specific OOS requires exactly long and short inputs")
    if admission_spec.window_days != 21:
        raise TargetSpecificOOSError("target-specific OOS requires the canonical 21-day mapping")
    root = Path(output_dir)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite target-specific OOS artifact: {root}")
    input_shared_contracts = {
        item.source_manifest.get("shared_population_contract_sha256") for item in inputs
    }
    if len(input_shared_contracts) != 1:
        raise TargetSpecificOOSError("long/short target-specific inputs disagree on shared-population contract")
    shared_population_contract = next(iter(input_shared_contracts))
    strict_meta_preflight, strict_meta_preflight_receipt = preflight_strict_meta_availability(
        bundle, inputs,
    )
    results = [generate_stage_i_target_specific_strict_oof(bundle, item, fit_model=fit_model) for item in inputs]
    prediction = pd.concat([item.predictions for item in results], ignore_index=True)
    if prediction.duplicated(["side_name", "candidate_id"]).any():
        raise TargetSpecificOOSError("pooled sides do not have unique qualified identities")
    prediction["candidate_key"] = prediction.side_name.astype(str) + "::" + prediction.candidate_id.astype(str)
    # Each mapping sees both sides for the pooled parent but constructs its
    # own side-local rank coordinate.  These are the only common-bps scores.
    base_input = prediction.loc[prediction.base_strict_oof_available.astype(bool)].copy()
    base_input["net_bps"] = base_input.exact_net_bps
    base_mapped, base_audit = apply_causal_21d_side_admission(
        base_input, score_column="base_direct_score", net_column="net_bps", decision_column="decision_ts",
        label_available_column="label_available_ts", identity_column="candidate_key", spec=admission_spec,
    )
    meta_input = prediction.loc[prediction.strict_oof_available.astype(bool)].copy()
    meta_input["net_bps"] = meta_input.exact_net_bps
    meta_mapped, meta_audit = apply_causal_21d_side_admission(
        meta_input, score_column="meta_direct_score", net_column="net_bps", decision_column="decision_ts",
        label_available_column="label_available_ts", identity_column="candidate_key", spec=admission_spec,
    )
    base_mapped = _prefix_admission_columns(base_mapped, prefix="base_causal_21d_")
    meta_mapped = _prefix_admission_columns(meta_mapped, prefix="meta_causal_21d_")
    for mapped, prefix in ((base_mapped, "base_causal_21d_"), (meta_mapped, "meta_causal_21d_")):
        columns = ["candidate_key", f"{prefix}expected_net_bps", f"{prefix}admitted"]
        prediction = prediction.merge(mapped.loc[:, columns], on="candidate_key", how="left", validate="one_to_one")
    # Earlier rows remain available exclusively as fold/map history.  The
    # published economics are exactly the requested 2024--26 period and may
    # never be pooled with the historical warm-up population.
    evaluation = prediction.loc[
        prediction.decision_ts.ge(_EVALUATION_START) & prediction.decision_ts.lt(_EVALUATION_END_EXCLUSIVE)
    ].copy()
    if evaluation.empty or set(evaluation.side_name.astype(str).str.lower()) != set(_SIDES):
        raise TargetSpecificOOSError("target-specific OOS requires an evaluable 2024--2026 row population on both sides")
    coverage = _coverage_audit(evaluation, results=results)
    # This is a *joint reconstructed* gate candidate.  It provides a frozen
    # score for this stack and uses no base-only economics.  Promotion between
    # alternatives must call the same comparator with all finalist ledgers on
    # their identical-row contract.
    joint_score, joint_attribution = compare_target_specific_finalists((
        StageITargetSpecificFinalist("current_stack", evaluation, {"winner_bundle_sha256": bundle.sha256}),
    ))
    raw = pd.concat([
        _side_month_raw_metrics(evaluation, layer="base", score="base_direct_score"),
        _side_month_raw_metrics(evaluation, layer="meta", score="meta_direct_score"),
    ], ignore_index=True)
    base_global, base_worst = _mapped_global_metrics(evaluation, layer="base", map_prefix="base_causal_21d_")
    meta_global, meta_worst = _mapped_global_metrics(evaluation, layer="meta", map_prefix="meta_causal_21d_")
    metrics = pd.concat([raw, base_global, meta_global], ignore_index=True, sort=False)
    worst = pd.concat([base_worst, meta_worst], ignore_index=True, sort=False)
    source_lineage = {
        item.side: {
            "source_manifest_sha256": item.source_manifest_sha256,
            "base_selector_manifest_sha256": bundle.cell(item.side).base_selector_manifest_sha256,
            "meta_selector_manifest_sha256": bundle.cell(item.side).meta_selector_manifest_sha256,
            "base_target_contract_sha256": bundle.cell(item.side).base_target_contract.sha256,
            "meta_target_contract_sha256": bundle.cell(item.side).meta_target_contract.sha256,
            "base_correlation_policy": item.base_selector_manifest.get("correlation_policy"),
            "meta_correlation_policy": item.meta_selector_manifest.get("correlation_policy"),
            "causal_feature_role_contract_sha256": item.source_manifest.get("causal_feature_role_contract_sha256"),
            "evaluation_month_contract_sha256": item.source_manifest.get("evaluation_month_contract_sha256"),
        }
        for item in inputs
    }
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.tmp-", dir=root.parent))
    try:
        # Keep historical warm-up explicitly separate from the evaluation
        # ledger.  The familiar strict_oof_predictions name always means the
        # requested OOS calendar in this target-specific route.
        prediction.to_parquet(temporary / "full_history_strict_oof_predictions.parquet", index=False, compression="zstd")
        evaluation.to_parquet(temporary / "strict_oof_predictions.parquet", index=False, compression="zstd")
        pd.concat([item.fold_provenance for item in results], ignore_index=True).to_parquet(temporary / "fold_provenance.parquet", index=False, compression="zstd")
        metrics.to_parquet(temporary / "per_side_month_base_meta_metrics.parquet", index=False, compression="zstd")
        worst.to_parquet(temporary / "worst_period_diagnostics.parquet", index=False, compression="zstd")
        coverage.to_parquet(temporary / "2024_2026_side_month_coverage_audit.parquet", index=False, compression="zstd")
        joint_score.to_parquet(temporary / "joint_stack_promotion_score.parquet", index=False, compression="zstd")
        joint_attribution.to_parquet(temporary / "joint_stack_promotion_attribution.parquet", index=False, compression="zstd")
        base_audit.assign(layer="base").to_parquet(temporary / "base_causal_21d_map_audit.parquet", index=False, compression="zstd")
        meta_audit.assign(layer="meta").to_parquet(temporary / "meta_causal_21d_map_audit.parquet", index=False, compression="zstd")
        files = {path.name: file_sha256(path) for path in temporary.iterdir() if path.is_file()}
        manifest = {
            "schema": SCHEMA, "status": "complete", "winner_bundle_sha256": bundle.sha256,
            "winner_bundle": bundle.to_dict(), "source_lineage": source_lineage,
            "shared_population_contract_sha256": shared_population_contract,
            "strict_meta_preflight": strict_meta_preflight_receipt,
            "strict_meta_preflight_rows": int(len(strict_meta_preflight)),
            "full_history_rows": len(prediction), "rows": len(evaluation),
            "evaluation_window": "2024-01-01T00:00:00Z <= decision_ts < 2027-01-01T00:00:00Z",
            "base_strict_oof_rows": int(evaluation.base_strict_oof_available.sum()),
            "meta_strict_oof_rows": int(evaluation.strict_oof_available.sum()),
            "timing": "signal close -> +1h decision/entry -> H12 label; label available at decision+12h",
            "meta_target": DIRECT_FQ3_SEMANTICS, "meta_input": DIRECT_BASE_INPUT_SEMANTICS,
            "mapping": "causal prior-resolved 21-day side-local rank maps with pooled-parent common-bps shrinkage",
            "ranking": "only pooled global after side-local causal common-bps mapping; never per timestamp",
            "raw_metrics": "unranked side/month diagnostics only; raw score is never a global decision score",
            "admission_spec": asdict(admission_spec), "ranking_tie_policy": RANKING_POLICY,
            "promotion_rule": "joint reconstructed base+meta stack only; base-only economics are diagnostics and are never terminal",
            "joint_stack_gate": joint_score.to_dict(orient="records"),
            "files": files,
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
        os.replace(temporary, root)
        return manifest
    except Exception:
        # One temporary directory is created for the atomic publish.  A single
        # recursive cleanup avoids duplicate creation/raise paths and remains
        # harmless if publication itself failed before any file was written.
        shutil.rmtree(temporary, ignore_errors=True)
        raise


__all__ = [
    "SCHEMA", "FROZEN_R3_NORMALIZER_SCHEMA", "DIRECT_FQ3_SEMANTICS", "DIRECT_BASE_INPUT_SEMANTICS", "TargetSpecificOOSError",
    "DirectCorrectnessState", "DirectFQ3Estimator", "fit_direct_fq3_estimator",
    "direct_fq3_selector_fit_context", "StageITargetSpecificInput", "StageITargetSpecificResult",
    "StageITargetSpecificFinalist", "FrozenR3FinalistInput", "generate_stage_i_target_specific_strict_oof",
    "run_stage_i_target_specific_oos", "compare_target_specific_finalists",
    "preflight_strict_meta_availability", "validate_preflight_strict_meta_availability_equality",
    "validate_target_specific_coverage_audit", "load_target_specific_finalist_artifact",
    "normalize_frozen_r3_finalist", "write_frozen_r3_finalist_normalizer",
]
