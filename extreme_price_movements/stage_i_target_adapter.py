"""Explicit target/objective adapters for the Stage-I stack.

The layer name is deliberately *not* an objective selector.  A caller must
carry one hash-bound :class:`StageITargetContract` from target materialisation
through MDA, HPO, strict OOF, winner freezing and production replay.

The promoted adapters are:

``soft_scalar_S``
    One bounded scalar regressor.  Its raw model output is the base score.
``cumulative_ordinal5_O``
    Four cumulative binary heads.  Their probabilities are monotonically
    projected to a five-state simplex and the expected ordinal value is the
    base score.
``fold_quantile_residual3``
    A three-class residual classifier.  Terciles, winsorisation and residual
    locations are fitted on each fold's prior-resolved candidate-only training
    stream.  Reconstruction keeps the fixed mapped base EV and adds the
    centred class correction.

Frozen R3 and Huber adapters remain named controls.  They are never inferred
from ``layer`` in the v2 contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from contextlib import contextmanager
from hashlib import sha256
import json
from typing import Any, Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .stage_i_base_target_ablation import (
    cumulative_ordinal_targets,
    recover_ordinal_simplex,
)


SCHEMA = "stage_i_target_adapter_v2"
CONTRACT_SCHEMA = "stage_i_target_economics_contract_v2"
IDENTITY_COLUMNS: tuple[str, ...] = (
    "candidate_id", "__ts__", "__symbol__", "side_name",
)

SOFT_SCALAR_S = "soft_scalar_S"
CUMULATIVE_ORDINAL5_O = "cumulative_ordinal5_O"
FOLD_QUANTILE_RESIDUAL3 = "fold_quantile_residual3"
LEGACY_R3_MULTICLASS3 = "legacy_R3_multiclass3_control"
LEGACY_HUBER_RESIDUAL = "legacy_Huber_residual_control"

_ALIASES = {
    "scalar_S": SOFT_SCALAR_S,
    "soft_scalar": SOFT_SCALAR_S,
    "ordinal_O": CUMULATIVE_ORDINAL5_O,
    "cumulative_ordinal5": CUMULATIVE_ORDINAL5_O,
    "quantile_ordinal_residual": FOLD_QUANTILE_RESIDUAL3,
    "T3Q_fold_quantile_ordinal_residual": FOLD_QUANTILE_RESIDUAL3,
    "R3_control": LEGACY_R3_MULTICLASS3,
    "huber_residual": LEGACY_HUBER_RESIDUAL,
}
_FAMILIES = frozenset({
    SOFT_SCALAR_S,
    CUMULATIVE_ORDINAL5_O,
    FOLD_QUANTILE_RESIDUAL3,
    LEGACY_R3_MULTICLASS3,
    LEGACY_HUBER_RESIDUAL,
})


class StageITargetAdapterError(ValueError):
    """Raised when target lineage or objective reconstruction is ambiguous."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return _jsonable(value.item())
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        # Invalid-path target/economic cells are deliberately retained for
        # coverage lineage. Bind their missingness without asking JSON to
        # encode non-standard NaN/Infinity tokens.
        return {"__nonfinite__": str(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha(values: Sequence[Any], *, dtype: Any | None = None) -> str:
    array = np.asarray(values, dtype=dtype).reshape(-1)
    # JSON is slower than bytes but stable across endian/platform differences
    # and preserves object/string identity values unambiguously.
    return canonical_sha256(array.tolist())


def normalize_target_family(value: str) -> str:
    family = _ALIASES.get(str(value), str(value))
    if family not in _FAMILIES:
        raise StageITargetAdapterError(f"unsupported explicit Stage-I target family: {value!r}")
    return family


@dataclass(frozen=True)
class StageITargetContract:
    """Immutable identity-aligned target/economic training contract."""

    family: str
    layer: str
    target_name: str
    geometry: str
    identity_sha256: str
    target_sha256: str
    economics_sha256: str
    validity_sha256: str
    weight_sha256: str
    rows: int
    target_columns: tuple[str, ...]
    economics_columns: tuple[str, ...] = ("gross_bps", "net_bps")
    validity_column: str = "target_valid"
    weight_column: str = "sample_weight"
    source_manifest_sha256: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        family = normalize_target_family(self.family)
        object.__setattr__(self, "family", family)
        if self.schema != CONTRACT_SCHEMA:
            raise StageITargetAdapterError("target contract schema drift")
        if self.layer not in {"base", "meta"}:
            raise StageITargetAdapterError("target contract layer must be base/meta")
        if family == FOLD_QUANTILE_RESIDUAL3 and self.layer != "meta":
            raise StageITargetAdapterError("fold-quantile residual target is meta-only")
        if family in {SOFT_SCALAR_S, CUMULATIVE_ORDINAL5_O, LEGACY_R3_MULTICLASS3} and self.layer != "base":
            raise StageITargetAdapterError(f"{family} is a base target")
        if family == LEGACY_HUBER_RESIDUAL and self.layer != "meta":
            raise StageITargetAdapterError("legacy Huber residual is meta-only")
        if int(self.rows) < 1 or not self.target_name or not self.geometry:
            raise StageITargetAdapterError("target contract requires rows/name/geometry")
        if not self.target_columns or len(set(self.target_columns)) != len(self.target_columns):
            raise StageITargetAdapterError("target columns must be non-empty and unique")
        for label, value in (
            ("identity", self.identity_sha256), ("target", self.target_sha256),
            ("economics", self.economics_sha256), ("validity", self.validity_sha256),
            ("weight", self.weight_sha256),
        ):
            if len(str(value)) != 64:
                raise StageITargetAdapterError(f"{label} contract SHA256 is malformed")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "StageITargetContract":
        value = dict(raw)
        value["target_columns"] = tuple(value["target_columns"])
        value["economics_columns"] = tuple(value.get("economics_columns", ("gross_bps", "net_bps")))
        return cls(**value)


def bind_target_contract(
    frame: pd.DataFrame,
    *,
    family: str,
    layer: str,
    target_name: str,
    geometry: str,
    target_columns: Sequence[str],
    identity_columns: Sequence[str] = IDENTITY_COLUMNS,
    economics_columns: Sequence[str] = ("gross_bps", "net_bps"),
    validity_column: str = "target_valid",
    weight_column: str = "sample_weight",
    source_manifest_sha256: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> StageITargetContract:
    """Bind aligned target vectors; no positional or geometry fallback exists."""

    columns = tuple(dict.fromkeys([
        *map(str, identity_columns), *map(str, target_columns),
        *map(str, economics_columns), str(validity_column), str(weight_column),
    ]))
    if missing := sorted(set(columns).difference(frame.columns)):
        raise StageITargetAdapterError(f"target contract frame lacks {missing}")
    if frame.empty:
        raise StageITargetAdapterError("target contract cannot bind an empty frame")
    identity = frame.loc[:, list(identity_columns)].copy()
    if identity.isna().any().any() or identity.duplicated().any():
        raise StageITargetAdapterError("target contract identities must be non-null and unique")
    validity = frame[validity_column]
    if not validity.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise StageITargetAdapterError("target validity must be explicit booleans")
    weight = pd.to_numeric(frame[weight_column], errors="coerce").to_numpy(np.float64)
    if not np.isfinite(weight).all() or (weight < 0).any():
        raise StageITargetAdapterError("target weights must be finite and non-negative")
    economics = frame.loc[:, list(economics_columns)].apply(pd.to_numeric, errors="coerce")
    valid = validity.to_numpy(bool)
    if not np.isfinite(economics.to_numpy(np.float64)[valid]).all():
        raise StageITargetAdapterError("winning-geometry economics must be finite on valid rows")
    targets = frame.loc[:, list(target_columns)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(targets.to_numpy(np.float64)[valid]).all():
        raise StageITargetAdapterError("targets must be finite on valid rows")
    return StageITargetContract(
        family=family, layer=str(layer), target_name=str(target_name), geometry=str(geometry),
        identity_sha256=canonical_sha256(identity.astype(str).to_dict("records")),
        target_sha256=canonical_sha256(targets.to_dict("list")),
        economics_sha256=canonical_sha256(economics.to_dict("list")),
        validity_sha256=_array_sha(validity.to_numpy(bool), dtype=bool),
        weight_sha256=_array_sha(weight, dtype=np.float64), rows=int(len(frame)),
        target_columns=tuple(map(str, target_columns)),
        economics_columns=tuple(map(str, economics_columns)),
        validity_column=str(validity_column), weight_column=str(weight_column),
        source_manifest_sha256=str(source_manifest_sha256), metadata=dict(metadata or {}),
    )


def verify_target_contract(
    frame: pd.DataFrame,
    contract: StageITargetContract,
    *, identity_columns: Sequence[str] = IDENTITY_COLUMNS,
) -> None:
    rebound = bind_target_contract(
        frame, family=contract.family, layer=contract.layer,
        target_name=contract.target_name, geometry=contract.geometry,
        target_columns=contract.target_columns, identity_columns=identity_columns,
        economics_columns=contract.economics_columns,
        validity_column=contract.validity_column, weight_column=contract.weight_column,
        source_manifest_sha256=contract.source_manifest_sha256,
        metadata=contract.metadata,
    )
    if rebound.sha256 != contract.sha256:
        raise StageITargetAdapterError("identity-aligned target/economics/validity/weight contract drift")


@dataclass(frozen=True)
class FoldQuantileResidualState:
    thresholds_bps: tuple[float, float]
    winsor_bounds_bps: tuple[float, float]
    class_prior: tuple[float, float, float]
    class_locations_bps: tuple[float, float, float]
    class_support: tuple[int, int, int]
    shrinkage_support: float = 50.0
    correction_clip_bps: float = 200.0
    semantic_gate: str = "q33<0<=q67"

    def __post_init__(self) -> None:
        q33, q67 = self.thresholds_bps
        if not (np.isfinite(q33) and np.isfinite(q67) and q33 < 0.0 <= q67):
            raise StageITargetAdapterError(
                "fold residual semantic gate failed: expected q33 < 0 <= q67"
            )
        if any(value < 1 for value in self.class_support):
            raise StageITargetAdapterError("fold residual target lacks a class")
        prior = np.asarray(self.class_prior, dtype=np.float64)
        if (prior <= 0).any() or not np.isclose(prior.sum(), 1.0):
            raise StageITargetAdapterError("fold residual priors are invalid")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


def fit_fold_quantile_residual3(
    exact_net_bps: Sequence[float],
    mapped_base_expected_net_bps: Sequence[float],
    *,
    shrinkage_support: float = 50.0,
    correction_clip_bps: float = 200.0,
) -> tuple[np.ndarray, FoldQuantileResidualState]:
    """Fit fold-local terciles and shrunk winsorised residual locations."""

    net = np.asarray(exact_net_bps, dtype=np.float64).reshape(-1)
    mapped = np.asarray(mapped_base_expected_net_bps, dtype=np.float64).reshape(-1)
    if len(net) < 3 or len(net) != len(mapped) or not np.isfinite(net).all() or not np.isfinite(mapped).all():
        raise StageITargetAdapterError("fold residual fit needs aligned finite net/base EV")
    residual = net - mapped
    q33, q67 = (float(value) for value in np.quantile(residual, (1 / 3, 2 / 3), method="linear"))
    labels = np.digitize(residual, (q33, q67), right=True).astype(np.int8)
    support = tuple(int(np.sum(labels == value)) for value in range(3))
    if not (q33 < 0.0 <= q67):
        raise StageITargetAdapterError(
            "fold residual semantic gate failed: q33<0<=q67 is required"
        )
    if any(value < 1 for value in support):
        raise StageITargetAdapterError("fold residual target lacks a class")
    lower, upper = (float(value) for value in np.quantile(residual, (0.05, 0.95), method="linear"))
    winsor = np.clip(residual, lower, upper)
    global_location = float(winsor.mean())
    locations: list[float] = []
    for value, n in enumerate(support):
        local = winsor[labels == value]
        locations.append(float(
            (local.sum() + float(shrinkage_support) * global_location)
            / (n + float(shrinkage_support))
        ))
    prior = tuple(float(value / len(labels)) for value in support)
    state = FoldQuantileResidualState(
        thresholds_bps=(q33, q67), winsor_bounds_bps=(lower, upper),
        class_prior=prior, class_locations_bps=tuple(locations),
        class_support=support, shrinkage_support=float(shrinkage_support),
        correction_clip_bps=float(correction_clip_bps),
    )
    return labels, state


def reconstruct_fold_quantile_residual3(
    probabilities: np.ndarray,
    mapped_base_expected_net_bps: Sequence[float],
    state: FoldQuantileResidualState,
) -> tuple[np.ndarray, np.ndarray]:
    """Return centred correction and fixed-base-EV reconstruction."""

    probability = np.asarray(probabilities, dtype=np.float64)
    mapped = np.asarray(mapped_base_expected_net_bps, dtype=np.float64).reshape(-1)
    if probability.shape != (len(mapped), 3) or not np.isfinite(probability).all():
        raise StageITargetAdapterError("meta probabilities must be a finite Nx3 simplex")
    if (probability < 0).any() or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-5):
        raise StageITargetAdapterError("meta probabilities must sum to one")
    correction = (
        (probability - np.asarray(state.class_prior, dtype=np.float64))
        @ np.asarray(state.class_locations_bps, dtype=np.float64)
    )
    correction = np.clip(correction, -state.correction_clip_bps, state.correction_clip_bps)
    return correction.astype(np.float32), (mapped + correction).astype(np.float32)


def recover_base_score(family: str, raw_prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Canonical raw-score reconstruction used by MDA, HPO and OOS."""

    family = normalize_target_family(family)
    raw = np.asarray(raw_prediction, dtype=np.float64)
    if family == SOFT_SCALAR_S:
        score = np.clip(raw.reshape(-1), 0.0, 1.0)
        return score.astype(np.float32), None
    if family == CUMULATIVE_ORDINAL5_O:
        simplex = recover_ordinal_simplex(raw)
        score = simplex @ (np.arange(5, dtype=np.float64) / 4.0)
        return score.astype(np.float32), simplex.astype(np.float32)
    if family == LEGACY_R3_MULTICLASS3:
        if raw.ndim != 2 or raw.shape[1] != 3:
            raise StageITargetAdapterError("legacy R3 prediction must be Nx3")
        if not np.isfinite(raw).all() or not np.allclose(raw.sum(axis=1), 1.0, atol=1e-5):
            raise StageITargetAdapterError("legacy R3 prediction is not a simplex")
        return (raw[:, 2] - raw[:, 0]).astype(np.float32), raw.astype(np.float32)
    raise StageITargetAdapterError(f"{family} does not reconstruct a base score")


def training_objectives(family: str) -> tuple[dict[str, Any], ...]:
    """Return explicit head objectives; never consult a layer name."""

    family = normalize_target_family(family)
    if family == SOFT_SCALAR_S:
        return ({"head": "scalar", "objective": "regression_l1", "classifier": False},)
    if family == CUMULATIVE_ORDINAL5_O:
        return tuple(
            {"head": f"P(Y>{boundary})", "objective": "binary", "classifier": True}
            for boundary in range(4)
        )
    if family in {FOLD_QUANTILE_RESIDUAL3, LEGACY_R3_MULTICLASS3}:
        return ({"head": "simplex3", "objective": "multiclass", "num_class": 3, "classifier": True},)
    if family == LEGACY_HUBER_RESIDUAL:
        return ({"head": "residual_bps", "objective": "huber", "classifier": False},)
    raise AssertionError(family)


class CumulativeOrdinal5Estimator:
    """Small sklearn-compatible four-head estimator used by selector MDA."""

    def __init__(self, models: Sequence[Any], constants: Sequence[float], feature_names: Sequence[str]):
        self.models = tuple(models)
        self.constants = tuple(float(value) for value in constants)
        self.feature_name_ = tuple(map(str, feature_names))
        importances = []
        for model in self.models:
            raw = getattr(model, "feature_importances_", np.zeros(len(self.feature_name_)))
            importances.append(np.asarray(raw, dtype=np.float64))
        self.feature_importances_ = (
            np.mean(importances, axis=0) if importances
            else np.zeros(len(self.feature_name_), dtype=np.float64)
        )

    def predict_cumulative_probability(self, frame: pd.DataFrame) -> np.ndarray:
        output = np.empty((len(frame), 4), dtype=np.float32)
        model_index = 0
        for boundary, constant in enumerate(self.constants):
            if np.isfinite(constant):
                output[:, boundary] = constant
            else:
                model = self.models[model_index]
                model_index += 1
                output[:, boundary] = np.asarray(model.predict_proba(frame), dtype=np.float32)[:, 1]
        return output

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        score, _ = recover_base_score(
            CUMULATIVE_ORDINAL5_O, self.predict_cumulative_probability(frame)
        )
        return score


def fit_cumulative_ordinal5_estimator(
    frame: pd.DataFrame,
    target: Sequence[int],
    sample_weight: Sequence[float] | None,
    *, params: Mapping[str, Any],
) -> CumulativeOrdinal5Estimator:
    """Fit the four true cumulative heads with identical row/weight support."""

    from lightgbm import LGBMClassifier

    classes = np.asarray(target, dtype=np.int8).reshape(-1)
    if len(classes) != len(frame) or not np.isin(classes, [0, 1, 2, 3, 4]).all():
        raise StageITargetAdapterError("cumulative ordinal fitting target must be aligned classes 0..4")
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float32).reshape(-1)
    if weights is not None and (len(weights) != len(frame) or not np.isfinite(weights).all()):
        raise StageITargetAdapterError("cumulative ordinal weights are invalid")
    cumulative = cumulative_ordinal_targets(classes)
    models: list[Any] = []
    constants: list[float] = []
    for boundary in range(4):
        local = cumulative[:, boundary]
        if np.unique(local).size < 2:
            constants.append(float(local[0]))
            continue
        constants.append(float("nan"))
        local_params = dict(params)
        local_params.pop("num_class", None)
        local_params["objective"] = "binary"
        local_params["random_state"] = int(local_params.get("random_state", 42)) + boundary
        model = LGBMClassifier(**local_params)
        fit_kwargs = {"sample_weight": weights} if weights is not None else {}
        models.append(model.fit(frame, local, **fit_kwargs))
    return CumulativeOrdinal5Estimator(models, constants, frame.columns)


class FoldQuantileResidual3Estimator:
    """Regression-shaped wrapper around the fold-local three-class head."""

    def __init__(self, model: Any, state: FoldQuantileResidualState, feature_names: Sequence[str]):
        self.model = model
        self.state = state
        self.classes_ = np.asarray([0, 1, 2], dtype=np.int8)
        self.feature_name_ = tuple(map(str, feature_names))
        self.feature_importances_ = np.asarray(
            getattr(model, "feature_importances_", np.zeros(len(self.feature_name_))),
            dtype=np.float64,
        )

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        probability = np.asarray(self.model.predict_proba(frame), dtype=np.float32)
        if probability.shape != (len(frame), 3):
            raise StageITargetAdapterError("fold residual model did not emit Nx3")
        return probability

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        correction, _ = reconstruct_fold_quantile_residual3(
            self.predict_proba(frame), np.zeros(len(frame), dtype=np.float32), self.state,
        )
        return correction


def fit_fold_quantile_residual3_estimator(
    frame: pd.DataFrame,
    residual_target: Sequence[float],
    sample_weight: Sequence[float] | None,
    *, params: Mapping[str, Any],
) -> FoldQuantileResidual3Estimator:
    from lightgbm import LGBMClassifier

    residual = np.asarray(residual_target, dtype=np.float64).reshape(-1)
    labels, state = fit_fold_quantile_residual3(
        residual, np.zeros(len(residual), dtype=np.float64),
    )
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float32).reshape(-1)
    local_params = dict(params)
    local_params.update({"objective": "multiclass", "num_class": 3})
    model = LGBMClassifier(**local_params)
    fit_kwargs = {"sample_weight": weights} if weights is not None else {}
    model.fit(frame, labels, **fit_kwargs)
    return FoldQuantileResidual3Estimator(model, state, frame.columns)


@contextmanager
def cumulative_ordinal5_selector_fit_context() -> Any:
    """Temporarily route the existing selector's model fits to four O heads.

    The selector still owns coverage, univariate/Relief, Spearman grouping,
    stability cohorts and signed-economic permutation logic.  Only its model
    adapter changes, so every permutation is ranked by the recovered monotone
    expected ordinal score rather than by an ordinal-class surrogate.
    """

    from . import lgbm_pipeline

    original = lgbm_pipeline._fit_lgbm_model

    def _fit(frame: pd.DataFrame, target: Sequence[int], sample_weight: Any = None, *, params: Mapping[str, Any], **_: Any) -> Any:
        return fit_cumulative_ordinal5_estimator(
            frame, target, sample_weight, params=params,
        )

    lgbm_pipeline._fit_lgbm_model = _fit
    try:
        yield
    finally:
        lgbm_pipeline._fit_lgbm_model = original


@contextmanager
def fold_quantile_residual3_selector_fit_context() -> Any:
    """Route every selector/MDA refit through fold-local residual terciles."""

    from . import lgbm_pipeline

    original = lgbm_pipeline._fit_lgbm_model

    def _fit(frame: pd.DataFrame, target: Sequence[float], sample_weight: Any = None, *, params: Mapping[str, Any], **_: Any) -> Any:
        return fit_fold_quantile_residual3_estimator(
            frame, target, sample_weight, params=params,
        )

    lgbm_pipeline._fit_lgbm_model = _fit
    try:
        yield
    finally:
        lgbm_pipeline._fit_lgbm_model = original


@contextmanager
def normalized_selector_sample_weight_context() -> Any:
    """Normalize the weights presented to every internal selector model fit.

    Contract-certainty weights have fixed row-relative values, but their
    declared scale is the mean of the current permitted training slice.  The
    selector owns several nested resamples, so normalization belongs at this
    final fit boundary rather than in the globally aligned handoff vector.
    """

    from . import lgbm_pipeline

    original = lgbm_pipeline._fit_lgbm_model

    def _fit(frame: pd.DataFrame, target: Sequence[Any], sample_weight: Any = None, **kwargs: Any) -> Any:
        normalized = sample_weight
        if sample_weight is not None:
            normalized = np.asarray(sample_weight, dtype=np.float32).reshape(-1)
            if (
                len(normalized) != len(frame)
                or not np.isfinite(normalized).all()
                or (normalized < 0.0).any()
                or float(normalized.mean()) <= 0.0
            ):
                raise StageITargetAdapterError("selector target weights are invalid")
            normalized = normalized / np.float32(normalized.mean())
        return original(frame, target, normalized, **kwargs)

    lgbm_pipeline._fit_lgbm_model = _fit
    try:
        yield
    finally:
        lgbm_pipeline._fit_lgbm_model = original


def adapter_manifest(contract: StageITargetContract) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "contract": contract.to_dict(),
        "contract_sha256": contract.sha256,
        "training_objectives": list(training_objectives(contract.family)),
        "score_reconstruction": {
            SOFT_SCALAR_S: "clip(raw_scalar,0,1)",
            CUMULATIVE_ORDINAL5_O: "monotone_five_simplex_then_expected_ordinal",
            FOLD_QUANTILE_RESIDUAL3: "mapped_base_ev_plus_clip((p-prior)@locations,+/-200bps)",
            LEGACY_R3_MULTICLASS3: "P(clear)-P(adverse)",
            LEGACY_HUBER_RESIDUAL: "mapped_base_ev_plus_predicted_residual",
        }[contract.family],
    }


def generic_base_trust_features(
    raw_score: Sequence[float],
    probability_simplex: np.ndarray | None,
    value_map_audit: pd.DataFrame,
) -> pd.DataFrame:
    """Target-neutral trust fields from direct OOF output and causal map support."""

    score = np.asarray(raw_score, dtype=np.float64).reshape(-1)
    if not np.isfinite(score).all():
        raise StageITargetAdapterError("base trust raw score must be finite")
    if probability_simplex is None:
        if ((score < 0.0) | (score > 1.0)).any():
            raise StageITargetAdapterError("scalar base trust score must lie in [0,1]")
        probability = np.column_stack([1.0 - score, score])
    else:
        probability = np.asarray(probability_simplex, dtype=np.float64)
        if probability.ndim != 2 or len(probability) != len(score):
            raise StageITargetAdapterError("base trust simplex must be aligned")
    if (probability < 0).any() or not np.isfinite(probability).all() or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-5):
        raise StageITargetAdapterError("base trust probabilities are not a simplex")
    audit = value_map_audit.reset_index(drop=True)
    required = {"prior_resolved_global_support", "prior_resolved_bin_support", "value_map_fallback"}
    if len(audit) != len(score) or not required.issubset(audit.columns):
        raise StageITargetAdapterError("base trust map audit is incomplete/misaligned")
    ordered = np.sort(probability, axis=1)
    second = ordered[:, -2] if probability.shape[1] > 1 else np.zeros(len(score))
    entropy = -np.sum(probability * np.log(np.clip(probability, 1e-12, 1.0)), axis=1)
    return pd.DataFrame({
        "base_output_entropy": entropy.astype(np.float32),
        "base_output_top2_margin": (ordered[:, -1] - second).astype(np.float32),
        "base_output_max_probability": ordered[:, -1].astype(np.float32),
        "base_value_map_prior_global_support_log1p": np.log1p(
            pd.to_numeric(audit.prior_resolved_global_support, errors="raise").to_numpy(np.float32)
        ),
        "base_value_map_prior_bin_support_log1p": np.log1p(
            pd.to_numeric(audit.prior_resolved_bin_support, errors="raise").to_numpy(np.float32)
        ),
        "base_value_map_neutral_fallback": audit.value_map_fallback.astype(str).eq(
            "neutral_no_prior_resolved_support"
        ).to_numpy(np.float32),
    })


def load_base_target_winner_bundle(
    bundle_dir: str | Path,
    *,
    side: str,
) -> tuple[pd.DataFrame, StageITargetContract, dict[str, Any]]:
    """Load one reviewed target winner and bind its exact side-local bytes.

    ``sample_weight_base_component`` is the only row-level weight input.  The
    training-weight JSON remains separately hash-bound because hybrid weights
    must be recomputed inside each permitted fold and may never be globally
    prefit in this loader.
    """

    root = Path(bundle_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise StageITargetAdapterError(f"target winner manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "stage_i_base_target_winner_bundle_v1"
        or manifest.get("status") != "complete"
    ):
        raise StageITargetAdapterError("target winner bundle is incomplete/wrong schema")
    inventory = manifest.get("artifact_sha256")
    if not isinstance(inventory, Mapping) or not inventory:
        raise StageITargetAdapterError("target winner lacks artifact hashes")
    for relative, expected in inventory.items():
        path = root / str(relative)
        if not path.is_file() or file_sha256(path) != str(expected):
            raise StageITargetAdapterError(f"target winner artifact drift: {relative}")
    target_path = root / "winner_target_handoff.parquet"
    weight_path = root / "training_weight_contract.json"
    target = pd.read_parquet(target_path)
    side_value = str(side).lower()
    target = target.loc[
        target.side_name.astype(str).str.lower().eq(side_value)
    ].reset_index(drop=True)
    required = {
        *IDENTITY_COLUMNS, "decision_ts", "label_available_ts", "target_valid",
        "gross_bps", "net_bps", "target_value", "sample_weight_base_component",
        "contract_certainty",
        "sample_weight_requires_fold_local_fit", "target_family", "target_name",
        "geometry", "weight_mode",
    }
    if missing := sorted(required.difference(target.columns)):
        raise StageITargetAdapterError(f"target winner handoff lacks {missing}")
    if target.empty or not target.target_valid.astype(bool).all():
        raise StageITargetAdapterError("winner handoff has no valid side-local target rows")
    family_raw = str(target.target_family.iloc[0])
    family = normalize_target_family(family_raw)
    if target.target_family.astype(str).nunique() != 1 or target.target_name.astype(str).nunique() != 1:
        raise StageITargetAdapterError("target winner side mixes target identities")
    if target.geometry.astype(str).nunique() != 1 or target.weight_mode.astype(str).nunique() != 1:
        raise StageITargetAdapterError("target winner side mixes geometry/weight contracts")
    weight_mode = str(target.weight_mode.iloc[0])
    if weight_mode not in {"uniform", "contract_certainty", "hybrid"}:
        raise StageITargetAdapterError("target winner declares an unknown weight mode")
    expected_fold_local = weight_mode == "hybrid"
    observed_fold_local = target.sample_weight_requires_fold_local_fit.astype(bool)
    if not observed_fold_local.eq(expected_fold_local).all():
        raise StageITargetAdapterError(
            "winner handoff fold-local weight flag disagrees with its weight mode"
        )
    weight_contract = json.loads(weight_path.read_text(encoding="utf-8"))
    if (
        weight_contract.get("schema") != "stage_i_base_target_training_weight_contract_v1"
        or str(weight_contract.get("mode", "")) != weight_mode
    ):
        raise StageITargetAdapterError("winner training-weight contract drift")
    declared_weight_sha = str(weight_contract.get("contract_sha256", ""))
    unhashed_weight_contract = dict(weight_contract)
    unhashed_weight_contract.pop("contract_sha256", None)
    if declared_weight_sha != canonical_sha256(unhashed_weight_contract):
        raise StageITargetAdapterError("winner training-weight contract self-hash drift")
    certainty = pd.to_numeric(target["contract_certainty"], errors="coerce").to_numpy(float)
    component = pd.to_numeric(
        target["sample_weight_base_component"], errors="coerce"
    ).to_numpy(float)
    if not np.isfinite(certainty).all() or ((certainty < 0.0) | (certainty > 1.0)).any():
        raise StageITargetAdapterError("winner contract-certainty values are invalid")
    expected_component = (
        np.ones(len(target), dtype=float)
        if weight_mode == "uniform" else 0.5 + 0.5 * certainty
    )
    if not np.allclose(component, expected_component, atol=1e-6, rtol=0.0):
        raise StageITargetAdapterError("winner base weight component drift")
    excluded = required | {
        "label_valid", "event", "event_minute", "favorable_progress",
        "adverse_progress", "dominance", "upper_fraction", "lower_fraction",
        "upper_floor_bound", "upper_cap_bound", "ordinal_alpha",
        "sample_weight_base_component", "sample_weight_requires_fold_local_fit",
    }
    regime_candidates = [
        column for column in target.columns
        if column not in excluded and "regime" in str(column).lower()
    ]
    metadata = {
        "winner_bundle_manifest_sha256": file_sha256(manifest_path),
        "winner_bundle_sha256": manifest.get("bundle_sha256", ""),
        "winner_target_handoff_sha256": file_sha256(target_path),
        "training_weight_contract_sha256": file_sha256(weight_path),
        "training_weight_contract": weight_contract,
        # The selector performs additional internal resampling fits whose
        # fold identities are intentionally hidden from target adapters.  A
        # hybrid winner therefore uses only its causal row-local certainty
        # component for MDA.  The full hybrid remains the declared training
        # arm and is recomputed by the chronological HPO routine, which does
        # expose fold-train identities.  Keeping both modes explicit prevents
        # a silent downgrade while avoiding globally prefit regime/class
        # weights inside MDA.
        "mda_selection_weight_mode": (
            "contract_certainty" if weight_mode == "hybrid" else weight_mode
        ),
        "hpo_training_weight_mode": weight_mode,
        "regime_column": regime_candidates[0] if len(regime_candidates) == 1 else "",
        "sample_weight_semantics": (
            "MDA uses only the row-local contract-certainty component; hybrid "
            "chronology/regime/class components are recomputed on each permitted HPO fold"
        ),
    }
    contract = bind_target_contract(
        target,
        family=family, layer="base", target_name=str(target.target_name.iloc[0]),
        geometry=str(target.geometry.iloc[0]), target_columns=("target_value",),
        economics_columns=("gross_bps", "net_bps"),
        validity_column="target_valid", weight_column="sample_weight_base_component",
        source_manifest_sha256=file_sha256(manifest_path), metadata=metadata,
    )
    audit = {
        "manifest": manifest, "manifest_sha256": file_sha256(manifest_path),
        "target_contract_sha256": contract.sha256,
        "side": side_value, "rows": len(target),
    }
    return target, contract, audit


__all__ = [
    "SCHEMA", "CONTRACT_SCHEMA", "SOFT_SCALAR_S", "CUMULATIVE_ORDINAL5_O",
    "FOLD_QUANTILE_RESIDUAL3", "LEGACY_R3_MULTICLASS3",
    "LEGACY_HUBER_RESIDUAL", "StageITargetAdapterError",
    "StageITargetContract", "FoldQuantileResidualState", "normalize_target_family",
    "canonical_sha256", "bind_target_contract", "verify_target_contract",
    "fit_fold_quantile_residual3", "reconstruct_fold_quantile_residual3",
    "recover_base_score", "training_objectives", "adapter_manifest",
    "CumulativeOrdinal5Estimator", "fit_cumulative_ordinal5_estimator",
    "cumulative_ordinal5_selector_fit_context",
    "normalized_selector_sample_weight_context",
    "FoldQuantileResidual3Estimator", "fit_fold_quantile_residual3_estimator",
    "fold_quantile_residual3_selector_fit_context",
    "file_sha256", "load_base_target_winner_bundle",
    "generic_base_trust_features",
]
