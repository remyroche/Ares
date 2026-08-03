"""Stage-VI causal and realised-path archetype ablation contracts.

This module is deliberately a *representation* experiment, not an expert
router.  It provides two separate, side-local workstreams:

``causal_feature``
    clusters compact pre-entry views.  Its memberships are inference-safe.

``realised_path``
    clusters post-entry paths for diagnosis, then learns a causal recogniser
    for the resulting *soft* memberships.  The realised memberships never
    leave diagnostics; only strict-OOF recogniser probabilities may be used as
    model features.

Both states reject outcome/path fields from inference transforms, are fitted
per side, use positive-label rows only for discovery, and never fit local
trading experts or route trades by a hard cluster id.  The compact design is
intended to make the small prescribed K/method/view ablations reproducible
without silently falling back to the legacy AE/GMM outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler


STAGE_VI_SCHEMA = "stage_vi_archetype_ablations_v1"
CAUSAL_PREFIX = "stage_vi_causal_arch_"
PATH_PREFIX = "stage_vi_path_arch_"
UNKNOWN_SUFFIX = "prob__unknown"

# These are intentionally token-level checks rather than a catalogue of
# current columns.  An explicit false positive is preferable to quietly
# accepting a new outcome-derived field in a causal input view.
FORBIDDEN_CAUSAL_TOKENS = (
    "label", "outcome", "realised", "realized", "post_entry", "postentry",
    "future", "mfe", "mae", "pnl", "first_touch", "timeout", "giveback",
    "retention", "time_to", "terminal_return", "path_arch", "exit",
    "path_", "event_",
)
CURRENT_ARCHETYPE_TOKENS = (
    "gmm_", "ae_", "dae_", "cluster_", "meta_conversion_arch_",
    "stage_ii_meta", "archetype_",
)


@dataclass(frozen=True)
class ArchetypeView:
    """A compact, explicitly named input view.

    The caller owns the feature assignment for CF0--CF4/PF0--PF4.  This keeps
    the representation honest when source schemas evolve: the module will not
    infer a huge catch-all matrix from a dataframe.
    """

    name: str
    columns: tuple[str, ...]
    kind: Literal["causal", "path"]

    def validate(self) -> None:
        if not self.name or not self.columns:
            raise ValueError("an archetype view needs a name and non-empty columns")
        if self.kind not in {"causal", "path"}:
            raise ValueError("view kind must be causal or path")
        if len(set(self.columns)) != len(self.columns):
            raise ValueError(f"view {self.name} contains duplicate columns")


@dataclass(frozen=True)
class ArchetypeWeightConfig:
    """AW0--AW5 sampling-weight contracts for representation discovery."""

    mode: Literal[
        "uniform", "time_balanced", "symbol_balanced", "path_certainty",
        "economic_diversity",
    ] = "uniform"
    timestamp_col: str = "decision_ts"
    symbol_col: str = "symbol"
    path_certainty_col: str | None = None
    economic_bucket_col: str | None = None
    minimum_weight: float = 0.25
    maximum_weight: float = 4.0

    def validate(self) -> None:
        if self.mode not in {
            "uniform", "time_balanced", "symbol_balanced", "path_certainty",
            "economic_diversity",
        }:
            raise ValueError(f"unknown archetype weight mode: {self.mode}")
        if self.minimum_weight <= 0 or self.maximum_weight < self.minimum_weight:
            raise ValueError("weight bounds must be positive and ordered")
        if self.mode == "path_certainty" and not self.path_certainty_col:
            raise ValueError("path_certainty requires path_certainty_col")
        if self.mode == "economic_diversity" and not self.economic_bucket_col:
            raise ValueError("economic_diversity requires economic_bucket_col")


@dataclass(frozen=True)
class ArchetypeConfig:
    """A bounded side-local clustering/recognition contract."""

    view: ArchetypeView
    method: Literal[
        "kmeans", "gmm_diag", "gmm_full", "gmm_pca_diag", "gmm_pca_full",
        "ae_gmm_diag", "ae_gmm_full",
    ] = "gmm_diag"
    components: int = 4
    side_col: str = "side_name"
    decision_ts_col: str = "decision_ts"
    label_available_ts_col: str = "label_available_ts"
    positive_label_col: str = "exact_net_bps"
    min_positive_value: float = 0.0
    min_side_rows: int = 250
    min_component_rows: int = 20
    classifier_c: float = 0.5
    reg_covar: float = 1e-4
    embedding_dimensions: int = 3
    ae_hidden_units: int = 8
    ae_max_iter: int = 200
    ae_alpha: float = 1e-3
    random_state: int = 20260803
    weights: ArchetypeWeightConfig = field(default_factory=ArchetypeWeightConfig)

    def validate(self) -> None:
        self.view.validate()
        if self.method not in {
            "kmeans", "gmm_diag", "gmm_full", "gmm_pca_diag", "gmm_pca_full",
            "ae_gmm_diag", "ae_gmm_full",
        }:
            raise ValueError("unsupported bounded archetype clustering method")
        if not 3 <= int(self.components) <= 8:
            raise ValueError("components must be in the bounded range [3, 8]")
        if self.min_side_rows < self.components * self.min_component_rows:
            raise ValueError("min_side_rows must support each requested component")
        if self.min_component_rows < 2 or self.reg_covar <= 0:
            raise ValueError("invalid support or covariance regularisation")
        if not 1 <= int(self.embedding_dimensions) <= 16:
            raise ValueError("embedding_dimensions must be in [1, 16]")
        if not 2 <= int(self.ae_hidden_units) <= 32:
            raise ValueError("ae_hidden_units must be in [2, 32]")
        if not 20 <= int(self.ae_max_iter) <= 500 or self.ae_alpha <= 0:
            raise ValueError("AE must use bounded positive regularisation and iterations")
        self.weights.validate()


@dataclass
class ArchetypeOOFResult:
    """Strict OOF inference-safe memberships plus diagnostic truth."""

    features: pd.DataFrame
    diagnostic_truth_memberships: pd.DataFrame
    fold_audit: pd.DataFrame
    catalog: pd.DataFrame
    manifest: dict[str, Any]


@dataclass
class ArchetypePredictiveValidation:
    """Strict-OOF recogniser diagnostics for realised-path memberships.

    ``prior_cluster_economic_bps`` is deliberately supplied by the caller: it
    must have been estimated from the resolving training rows, never from this
    evaluation block.  That makes the economic-confusion calculation a proper
    forecast diagnostic rather than a hindsight cluster-payoff lookup.
    """

    summary: pd.DataFrame
    per_membership: pd.DataFrame


@dataclass(frozen=True)
class MultiViewObjectiveConfig:
    """Predeclared composite rather than a silhouette-only cluster score."""

    path_separation_weight: float = 1.0
    economic_separation_weight: float = 1.0
    causal_predictability_weight: float = 1.0
    temporal_stability_weight: float = 1.0
    concentration_penalty: float = 1.0

    def validate(self) -> None:
        values = (
            self.path_separation_weight, self.economic_separation_weight,
            self.causal_predictability_weight, self.temporal_stability_weight,
            self.concentration_penalty,
        )
        if any(not np.isfinite(value) or value < 0 for value in values):
            raise ValueError("multi-view objective weights must be finite and non-negative")
        if sum(values[:-1]) <= 0:
            raise ValueError("multi-view objective requires a positive benefit weight")


@dataclass(frozen=True)
class ArchetypeDecisionConfig:
    """Conservative gates for the Stage-VI decision matrix."""

    minimum_economic_separation: float = 0.05
    minimum_causal_predictability: float = 0.05
    minimum_temporal_stability: float = 0.50
    minimum_incremental_bps: float = 0.0
    maximum_concentration: float = 0.75

    def validate(self) -> None:
        if not 0 <= self.minimum_economic_separation <= 1:
            raise ValueError("minimum_economic_separation must be in [0, 1]")
        if not -1 <= self.minimum_causal_predictability <= 1:
            raise ValueError("minimum_causal_predictability must be in [-1, 1]")
        if not 0 <= self.minimum_temporal_stability <= 1:
            raise ValueError("minimum_temporal_stability must be in [0, 1]")
        if not 0 <= self.maximum_concentration <= 1:
            raise ValueError("maximum_concentration must be in [0, 1]")


@dataclass
class CompactMultiViewState:
    """Train-only robust-scaled PCA embeddings for CF4/PF4-style views.

    This is deliberately linear and small.  It prevents the multi-view arm
    from accidentally concatenating hundreds of raw columns while keeping each
    source view auditable.  A caller must fit it inside an allowed training
    fold and can then materialise the returned low-dimensional causal columns
    for an ``ArchetypeView``.
    """

    views: Mapping[str, tuple[str, ...]]
    dimensions_per_view: int = 2
    kind: Literal["causal", "path"] = "causal"
    random_state: int = 20260803
    _models: dict[str, tuple[RobustScaler, PCA]] = field(default_factory=dict, init=False)

    def fit(self, frame: pd.DataFrame) -> "CompactMultiViewState":
        if not self.views or self.dimensions_per_view < 1:
            raise ValueError("compact multiview needs views and positive dimensions_per_view")
        for name, columns in self.views.items():
            view = ArchetypeView(str(name), tuple(columns), self.kind)
            view.validate()
            if self.kind == "causal":
                illegal = [column for column in columns if any(token in column.lower() for token in FORBIDDEN_CAUSAL_TOKENS)]
                if illegal:
                    raise ValueError(f"causal multiview contains outcome/path fields: {illegal[:8]}")
            legacy = [column for column in columns if any(token in column.lower() for token in CURRENT_ARCHETYPE_TOKENS)]
            if legacy:
                raise ValueError(f"Stage-VI multiview cannot overlap legacy AE/GMM/archetype outputs: {legacy[:8]}")
            missing = [column for column in columns if column not in frame]
            if missing:
                raise KeyError(f"multiview {name} missing fields: {missing[:8]}")
            raw = _matrix(frame, columns)
            scaler = RobustScaler().fit(raw)
            scaled = scaler.transform(np.where(np.isfinite(raw), raw, scaler.center_))
            dimensions = min(int(self.dimensions_per_view), scaled.shape[1], max(1, scaled.shape[0] - 1))
            self._models[str(name)] = (
                scaler,
                PCA(n_components=dimensions, random_state=int(self.random_state)).fit(scaled),
            )
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self._models:
            raise RuntimeError("compact multiview must be fit before transform")
        output: dict[str, np.ndarray] = {}
        for name, (scaler, pca) in self._models.items():
            columns = self.views[name]
            missing = [column for column in columns if column not in frame]
            if missing:
                raise KeyError(f"multiview transform {name} missing fields: {missing[:8]}")
            raw = _matrix(frame, columns)
            scaled = scaler.transform(np.where(np.isfinite(raw), raw, scaler.center_))
            embedding = pca.transform(scaled).astype(np.float32)
            for dimension in range(embedding.shape[1]):
                output[f"stage_vi_mv__{name}__pc{dimension}"] = embedding[:, dimension]
        return pd.DataFrame(output, index=frame.index, dtype=np.float32)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": STAGE_VI_SCHEMA,
            "kind": self.kind,
            "method": "per_view_train_only_robust_pca",
            "views": {name: list(columns) for name, columns in self.views.items()},
            "dimensions_per_view_requested": int(self.dimensions_per_view),
            "fitted": bool(self._models),
            "nonlinear_embedding": False,
        }


@dataclass
class _SideState:
    side: str
    # Discovery and recogniser transforms differ for realised-path states.
    # Keeping both in explicit state avoids an unsafe private mutation of an
    # sklearn estimator and makes direct ``SideLocalArchetypeState.fit`` safe.
    discovery_scaler: RobustScaler
    discovery_embedder: PCA | "_SmallAutoEncoder" | None
    scaler: RobustScaler
    clusterer: GaussianMixture | KMeans
    classifier: LogisticRegression | None
    classes: np.ndarray
    # Original cluster ID -> rank.  Ranks are assigned by training-only
    # centroid lexicographic order, not arbitrary sklearn numbering.
    original_to_rank: dict[int, int]
    support_by_rank: np.ndarray
    centroid_by_rank: np.ndarray


@dataclass
class _SmallAutoEncoder:
    """A bounded deterministic ReLU encoder extracted from an sklearn AE.

    It is deliberately small and regularised.  We use it only as a discovery
    transform before a regularised GMM; its reconstruction fit never receives
    future/path data at causal inference time.  Keeping the encoder weights
    explicit avoids a private sklearn-estimator mutation at transform time.
    """

    first_weight: np.ndarray
    first_bias: np.ndarray
    second_weight: np.ndarray
    second_bias: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, weights: np.ndarray, config: ArchetypeConfig) -> "_SmallAutoEncoder":
        latent = min(int(config.embedding_dimensions), values.shape[1], max(1, values.shape[0] - 1))
        hidden = max(latent, min(int(config.ae_hidden_units), 32))
        # MLPRegressor does not provide a stable sample-weight API.  This
        # deterministic weighted bootstrap is the same bounded treatment as
        # the GMM path and is fitting-only, never an inference feature.
        rng = np.random.default_rng(int(config.random_state) + 7919)
        positions = rng.choice(len(values), size=len(values), replace=True, p=weights / weights.sum())
        estimator = MLPRegressor(
            hidden_layer_sizes=(hidden, latent, hidden), activation="relu",
            solver="lbfgs", alpha=float(config.ae_alpha),
            max_iter=int(config.ae_max_iter), random_state=int(config.random_state),
        ).fit(values[positions], values[positions])
        return cls(
            first_weight=np.asarray(estimator.coefs_[0], dtype=np.float32),
            first_bias=np.asarray(estimator.intercepts_[0], dtype=np.float32),
            second_weight=np.asarray(estimator.coefs_[1], dtype=np.float32),
            second_bias=np.asarray(estimator.intercepts_[1], dtype=np.float32),
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        hidden = np.maximum(values @ self.first_weight + self.first_bias, 0.0)
        return np.maximum(hidden @ self.second_weight + self.second_bias, 0.0).astype(np.float32)


def archetype_feature_names(prefix: str, components: int) -> list[str]:
    return [
        *[f"{prefix}prob__{i}" for i in range(int(components))],
        f"{prefix}{UNKNOWN_SUFFIX}",
        f"{prefix}entropy",
        f"{prefix}confidence",
        f"{prefix}available",
    ]


def remove_current_archetype_columns(columns: Iterable[str]) -> list[str]:
    """Remove current AE/GMM/Stage-II outputs for a clean Stage-VI control.

    This deliberately does not remove ordinary feature names containing a
    generic word such as ``state``.  The returned list preserves the supplied
    order and can be recorded in the ablation manifest.
    """

    return [
        str(column) for column in columns
        if not any(token in str(column).lower() for token in CURRENT_ARCHETYPE_TOKENS)
    ]


def stage_vi_ablation_grid(
    views: Sequence[ArchetypeView],
    *,
    methods: Sequence[str] = ("kmeans", "gmm_diag", "gmm_full", "gmm_pca_diag", "ae_gmm_diag"),
    components: Sequence[int] = (3, 4, 5, 6, 8),
    weight_modes: Sequence[str] = ("uniform", "time_balanced", "symbol_balanced"),
    **base: Any,
) -> list[ArchetypeConfig]:
    """Build the intentionally small method × K × view experiment grid."""

    result: list[ArchetypeConfig] = []
    for view, method, k, weight in product(views, methods, components, weight_modes):
        cfg = ArchetypeConfig(
            view=view,
            method=str(method),  # type: ignore[arg-type]
            components=int(k),
            weights=ArchetypeWeightConfig(mode=str(weight)),  # type: ignore[arg-type]
            **base,
        )
        cfg.validate()
        result.append(cfg)
    return result


def stage_vi_path_ablation_grid(
    views: Sequence[ArchetypeView],
    *,
    methods: Sequence[str] = ("kmeans", "gmm_diag", "gmm_full", "gmm_pca_diag", "ae_gmm_diag"),
    components: Sequence[int] = (3, 4, 5, 6, 8),
    path_certainty_col: str,
    economic_bucket_col: str,
    **base: Any,
) -> list[ArchetypeConfig]:
    """Materialise the declared PF grid, including AW4/AW5 explicitly.

    AW3 is intentionally absent: discovery is already fitted separately by
    side, which is the preferred anti-side-collapse mechanism.  AW4 and AW5
    demand named path-quality/economic-diversity sources rather than silently
    using a profit-ranked surrogate.
    """

    if any(view.kind != "path" for view in views):
        raise ValueError("stage_vi_path_ablation_grid accepts only realised-path views")
    modes = ("uniform", "time_balanced", "symbol_balanced", "path_certainty", "economic_diversity")
    result: list[ArchetypeConfig] = []
    for view, method, k, mode in product(views, methods, components, modes):
        weights = ArchetypeWeightConfig(
            mode=mode,  # type: ignore[arg-type]
            path_certainty_col=path_certainty_col if mode == "path_certainty" else None,
            economic_bucket_col=economic_bucket_col if mode == "economic_diversity" else None,
        )
        config = ArchetypeConfig(view=view, method=str(method), components=int(k), weights=weights, **base)  # type: ignore[arg-type]
        config.validate()
        result.append(config)
    return result


def _timestamps(values: pd.Series, name: str) -> pd.Series:
    result = pd.to_datetime(values, utc=True, errors="coerce")
    if result.isna().any():
        raise ValueError(f"{name} must contain valid UTC timestamps")
    return result


def _validate_columns(frame: pd.DataFrame, config: ArchetypeConfig) -> list[str]:
    config.validate()
    required = [
        config.side_col, config.decision_ts_col, config.label_available_ts_col,
        config.positive_label_col, *config.view.columns,
    ]
    missing = [name for name in dict.fromkeys(required) if name not in frame]
    if missing:
        raise KeyError(f"archetype input is missing columns: {missing[:12]}")
    columns = list(config.view.columns)
    legacy = [name for name in columns if any(token in name.lower() for token in CURRENT_ARCHETYPE_TOKENS)]
    if legacy:
        raise ValueError(f"Stage-VI views cannot overlap legacy AE/GMM/archetype outputs: {legacy[:8]}")
    if config.view.kind == "causal":
        illegal = [name for name in columns if any(token in name.lower() for token in FORBIDDEN_CAUSAL_TOKENS)]
        if illegal:
            raise ValueError(f"causal archetype view contains outcome/path fields: {illegal[:8]}")
    non_numeric = [name for name in columns if not pd.api.types.is_numeric_dtype(frame[name])]
    if non_numeric:
        raise TypeError(f"archetype view must contain numeric fields: {non_numeric[:8]}")
    decision = _timestamps(frame[config.decision_ts_col], config.decision_ts_col)
    available = _timestamps(frame[config.label_available_ts_col], config.label_available_ts_col)
    if (available <= decision).any():
        raise ValueError("archetype labels must resolve strictly after the decision")
    return columns


def _matrix(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    return frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64, copy=True)


def _impute_fit_apply(train: np.ndarray, apply: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    medians = np.nanmedian(train, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    return (
        np.where(np.isfinite(train), train, medians).astype(np.float32),
        np.where(np.isfinite(apply), apply, medians).astype(np.float32),
    )


def _positive_mask(frame: pd.DataFrame, config: ArchetypeConfig) -> np.ndarray:
    values = pd.to_numeric(frame[config.positive_label_col], errors="coerce").to_numpy(dtype=float)
    return np.isfinite(values) & (values > float(config.min_positive_value))


def archetype_sample_weights(frame: pd.DataFrame, config: ArchetypeConfig) -> np.ndarray:
    """Compute normalized, bounded AW0--AW5 discovery weights.

    Weighting is used only for cluster fitting/sample selection and is never
    emitted as an inference feature.  KMeans supports it natively; GMM does
    not, so :func:`_fit_clusterer` uses deterministic weighted resampling.
    """

    rule = config.weights
    rule.validate()
    weights = np.ones(len(frame), dtype=np.float64)
    if rule.mode == "time_balanced":
        if rule.timestamp_col not in frame:
            raise KeyError(f"time-balanced weights require {rule.timestamp_col}")
        month = _timestamps(frame[rule.timestamp_col], rule.timestamp_col).dt.to_period("M").astype(str)
        counts = month.value_counts(dropna=False)
        weights = month.map(lambda value: 1.0 / float(counts[value])).to_numpy(dtype=float)
    elif rule.mode == "symbol_balanced":
        if rule.symbol_col not in frame:
            raise KeyError(f"symbol-balanced weights require {rule.symbol_col}")
        symbol = frame[rule.symbol_col].astype("string").fillna("__missing__")
        counts = symbol.value_counts(dropna=False)
        weights = symbol.map(lambda value: 1.0 / np.sqrt(float(counts[value]))).to_numpy(dtype=float)
    elif rule.mode == "path_certainty":
        values = pd.to_numeric(frame[str(rule.path_certainty_col)], errors="coerce").to_numpy(dtype=float)
        weights = np.where(np.isfinite(values), np.clip(values, 0.0, 1.0), 0.0)
    elif rule.mode == "economic_diversity":
        bucket = frame[str(rule.economic_bucket_col)].astype("string").fillna("__missing__")
        counts = bucket.value_counts(dropna=False)
        weights = bucket.map(lambda value: 1.0 / float(counts[value])).to_numpy(dtype=float)
    finite = np.isfinite(weights) & (weights > 0)
    if not finite.any():
        raise ValueError("archetype weighting produced no positive finite weight")
    weights = np.where(finite, weights, 0.0)
    weights *= float(finite.sum()) / weights.sum()
    return np.clip(weights, rule.minimum_weight, rule.maximum_weight).astype(np.float64)


def _fit_clusterer(
    values: np.ndarray,
    weights: np.ndarray,
    config: ArchetypeConfig,
) -> GaussianMixture | KMeans:
    if config.method == "kmeans":
        return KMeans(
            n_clusters=int(config.components), n_init=10,
            random_state=int(config.random_state),
        ).fit(values, sample_weight=weights)
    # sklearn GaussianMixture has no stable sample_weight API.  A deterministic
    # weighted bootstrap preserves bounded AW weights without treating them as
    # an inference input.  Cap at the native sample size to keep the search
    # bounded and reproducible.
    rng = np.random.default_rng(int(config.random_state))
    probabilities = weights / weights.sum()
    positions = rng.choice(len(values), size=len(values), replace=True, p=probabilities)
    return GaussianMixture(
        n_components=int(config.components),
        covariance_type="diag" if config.method.endswith("diag") else "full",
        reg_covar=float(config.reg_covar), n_init=2,
        random_state=int(config.random_state),
    ).fit(values[positions])


def _fit_discovery_embedding(
    scaled: np.ndarray,
    weights: np.ndarray,
    config: ArchetypeConfig,
) -> tuple[PCA | _SmallAutoEncoder | None, np.ndarray]:
    """Fit only the pre-declared bounded discovery embedding, if requested."""

    if config.method in {"kmeans", "gmm_diag", "gmm_full"}:
        return None, scaled
    if config.method.startswith("gmm_pca_"):
        dimensions = min(int(config.embedding_dimensions), scaled.shape[1], max(1, scaled.shape[0] - 1))
        pca = PCA(n_components=dimensions, svd_solver="full", random_state=int(config.random_state)).fit(scaled)
        return pca, pca.transform(scaled).astype(np.float32)
    if config.method.startswith("ae_gmm_"):
        encoder = _SmallAutoEncoder.fit(scaled, weights, config)
        return encoder, encoder.transform(scaled)
    raise AssertionError(f"unhandled method {config.method!r}")


def _apply_discovery_embedding(
    embedder: PCA | _SmallAutoEncoder | None,
    scaled: np.ndarray,
) -> np.ndarray:
    if embedder is None:
        return scaled
    return embedder.transform(scaled).astype(np.float32)


def _soft_membership(clusterer: GaussianMixture | KMeans, values: np.ndarray) -> np.ndarray:
    if isinstance(clusterer, GaussianMixture):
        return clusterer.predict_proba(values).astype(np.float32)
    distances = clusterer.transform(values).astype(np.float64)
    # A soft KMeans membership is a calibrated-free geometry descriptor, not a
    # predicted probability of an outcome.  Its temperature is train-only.
    finite = distances[np.isfinite(distances)]
    temperature = max(float(np.median(finite)) if finite.size else 1.0, 1e-6)
    logits = -distances / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)


def _centroids(clusterer: GaussianMixture | KMeans) -> np.ndarray:
    return np.asarray(clusterer.means_ if isinstance(clusterer, GaussianMixture) else clusterer.cluster_centers_, dtype=np.float32)


def _rank_clusters(centroids: np.ndarray) -> dict[int, int]:
    # Stable enough for a catalogue: each scaled centroid is ordered
    # lexicographically, with original component as deterministic tie-breaker.
    order = sorted(range(len(centroids)), key=lambda value: tuple(np.round(centroids[value], 6)) + (value,))
    return {int(original): int(rank) for rank, original in enumerate(order)}


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-8, 1.0)
    return (-np.sum(clipped * np.log(clipped), axis=1) / np.log(max(probabilities.shape[1], 2))).astype(np.float32)


class SideLocalArchetypeState:
    """A frozen side-local archetype state with only soft inference outputs."""

    def __init__(self, config: ArchetypeConfig, *, causal_recogniser_columns: Sequence[str] | None = None) -> None:
        config.validate()
        self.config = config
        self.causal_recogniser_columns = list(causal_recogniser_columns or ())
        self.side_models: dict[str, _SideState] = {}
        self.catalog_: pd.DataFrame = pd.DataFrame()

    @property
    def prefix(self) -> str:
        return CAUSAL_PREFIX if self.config.view.kind == "causal" else PATH_PREFIX

    @property
    def inference_columns(self) -> list[str]:
        return list(self.config.view.columns) if self.config.view.kind == "causal" else list(self.causal_recogniser_columns)

    def fit(self, train: pd.DataFrame) -> "SideLocalArchetypeState":
        view_cols = _validate_columns(train, self.config)
        if self.config.view.kind == "path":
            if not self.causal_recogniser_columns:
                raise ValueError("path archetypes require explicit causal_recogniser_columns")
            # Reuse causal validation with a temporary causal view to prevent
            # a path descriptor from accidentally becoming recogniser input.
            causal_cfg = ArchetypeConfig(
                view=ArchetypeView("path_recogniser", tuple(self.causal_recogniser_columns), "causal"),
                method=self.config.method, components=self.config.components,
                side_col=self.config.side_col, decision_ts_col=self.config.decision_ts_col,
                label_available_ts_col=self.config.label_available_ts_col,
                positive_label_col=self.config.positive_label_col,
                min_positive_value=self.config.min_positive_value,
                min_side_rows=self.config.min_side_rows, min_component_rows=self.config.min_component_rows,
                classifier_c=self.config.classifier_c, reg_covar=self.config.reg_covar,
                embedding_dimensions=self.config.embedding_dimensions,
                ae_hidden_units=self.config.ae_hidden_units,
                ae_max_iter=self.config.ae_max_iter, ae_alpha=self.config.ae_alpha,
                random_state=self.config.random_state, weights=self.config.weights,
            )
            recogniser_cols = _validate_columns(train, causal_cfg)
        else:
            recogniser_cols = view_cols
        positive = _positive_mask(train, self.config)
        side = train[self.config.side_col].astype(str).str.lower()
        rows: list[dict[str, Any]] = []
        for side_name, idx in side.groupby(side, sort=True).groups.items():
            subset = train.loc[idx]
            subset_positive = positive[train.index.get_indexer(subset.index)]
            fit_rows = subset.loc[subset_positive]
            if len(fit_rows) < int(self.config.min_side_rows):
                continue
            raw = _matrix(fit_rows, view_cols)
            finite = np.isfinite(raw).any(axis=1)
            if int(finite.sum()) < int(self.config.min_side_rows):
                continue
            raw = raw[finite]
            scaler = RobustScaler().fit(raw)
            scaled = scaler.transform(np.where(np.isfinite(raw), raw, scaler.center_)).astype(np.float32)
            weights = archetype_sample_weights(fit_rows.iloc[np.flatnonzero(finite)], self.config)
            discovery_embedder, discovered = _fit_discovery_embedding(scaled, weights, self.config)
            clusterer = _fit_clusterer(discovered, weights, self.config)
            hard = clusterer.predict(discovered).astype(np.int32)
            support = np.bincount(hard, minlength=int(self.config.components))
            if (support < int(self.config.min_component_rows)).any():
                continue
            centroid = _centroids(clusterer)
            rank_map = _rank_clusters(centroid)
            classes = np.asarray([rank_map[int(value)] for value in hard], dtype=np.int32)
            rec_raw = _matrix(fit_rows.iloc[np.flatnonzero(finite)], recogniser_cols)
            rec_scaler = RobustScaler().fit(rec_raw)
            rec_values = rec_scaler.transform(np.where(np.isfinite(rec_raw), rec_raw, rec_scaler.center_)).astype(np.float32)
            unique = np.unique(classes)
            classifier: LogisticRegression | None
            if len(unique) < 2:
                classifier = None
            else:
                classifier = LogisticRegression(
                    C=float(self.config.classifier_c), max_iter=400,
                    random_state=int(self.config.random_state),
                ).fit(rec_values, classes, sample_weight=weights)
            # The recogniser's scaler is stored in the generic scaler field;
            # path views do not need their fitted path scaler at transform.
            state = _SideState(
                side=str(side_name), discovery_scaler=scaler, discovery_embedder=discovery_embedder,
                scaler=rec_scaler,
                clusterer=clusterer, classifier=classifier,
                classes=unique.astype(np.int32), original_to_rank=rank_map,
                support_by_rank=np.asarray([support[original] for original in sorted(rank_map, key=rank_map.get)], dtype=np.float32),
                centroid_by_rank=np.asarray([centroid[original] for original in sorted(rank_map, key=rank_map.get)], dtype=np.float32),
            )
            self.side_models[str(side_name)] = state
            for original, rank in rank_map.items():
                rows.append({
                    "side": str(side_name), "rank": int(rank), "original_component": int(original),
                    "support_rows": int(support[original]), "view": self.config.view.name,
                    "view_kind": self.config.view.kind, "method": self.config.method,
                    "components": int(self.config.components), "centroid": centroid[original].tolist(),
                    "discovery_embedding": (
                        "pca" if isinstance(discovery_embedder, PCA)
                        else "small_ae" if isinstance(discovery_embedder, _SmallAutoEncoder) else "none"
                    ),
                })
        self.catalog_ = pd.DataFrame(rows).sort_values(["side", "rank"], kind="stable") if rows else pd.DataFrame()
        return self

    def diagnostic_realised_memberships(self, labelled: pd.DataFrame) -> pd.DataFrame:
        """Return cluster memberships from the discovery view for diagnostics.

        For path states this is explicitly realised future information and must
        never be joined into an inference model matrix.
        """

        _validate_columns(labelled, self.config)
        names = [f"{self.prefix}prob__{i}" for i in range(self.config.components)]
        out = pd.DataFrame(np.nan, index=labelled.index, columns=names, dtype=np.float32)
        side = labelled[self.config.side_col].astype(str).str.lower()
        for side_name, idx in side.groupby(side, sort=False).groups.items():
            state = self.side_models.get(str(side_name))
            if state is None:
                continue
            subset = labelled.loc[idx]
            values = _matrix(subset, self.config.view.columns)
            # Path diagnostic centroids were fit in a path scaler which is
            # intentionally not used at inference.  Use its explicit state
            # here only for retrospective truth memberships.
            discovery_scaler = state.discovery_scaler
            values = discovery_scaler.transform(np.where(np.isfinite(values), values, discovery_scaler.center_))
            values = _apply_discovery_embedding(state.discovery_embedder, values)
            original = _soft_membership(state.clusterer, values)
            ranked = np.zeros((len(subset), self.config.components), dtype=np.float32)
            for old, rank in state.original_to_rank.items():
                ranked[:, rank] = original[:, old]
            out.loc[subset.index] = ranked
        return out

    def transform(self, oos_without_outcomes: pd.DataFrame) -> pd.DataFrame:
        """Emit only causal soft memberships; reject realised-path leakage."""

        if self.config.view.kind == "path":
            # Path labels and every outcome-like column are forbidden in the
            # recogniser transform.  The positive-label column is included
            # explicitly because an innocuous name such as ``exact_net_bps``
            # does not necessarily contain one of the broad forbidden tokens.
            leaked = sorted(
                set(self.config.view.columns)
                .union({self.config.positive_label_col})
                .intersection(oos_without_outcomes.columns)
            )
            leaked.extend(
                name for name in oos_without_outcomes.columns
                if any(token in str(name).lower() for token in FORBIDDEN_CAUSAL_TOKENS)
                and str(name) not in leaked
            )
            if leaked:
                raise ValueError(f"path archetype transform received realised path columns: {leaked[:8]}")
        # Validate required identity/recogniser fields without requiring the
        # discovery view or positive label, both absent by design at inference.
        needed = [self.config.side_col, *self.inference_columns]
        missing = [name for name in needed if name not in oos_without_outcomes]
        if missing:
            raise KeyError(f"archetype transform is missing causal columns: {missing[:8]}")
        if self.config.view.kind == "causal":
            illegal = [name for name in self.inference_columns if any(token in name.lower() for token in FORBIDDEN_CAUSAL_TOKENS)]
            if illegal:
                raise ValueError(f"causal transform has forbidden fields: {illegal[:8]}")
        names = archetype_feature_names(self.prefix, self.config.components)
        out = pd.DataFrame(0.0, index=oos_without_outcomes.index, columns=names, dtype=np.float32)
        out[f"{self.prefix}{UNKNOWN_SUFFIX}"] = 1.0
        side = oos_without_outcomes[self.config.side_col].astype(str).str.lower()
        for side_name, idx in side.groupby(side, sort=False).groups.items():
            state = self.side_models.get(str(side_name))
            if state is None:
                continue
            subset = oos_without_outcomes.loc[idx]
            raw = _matrix(subset, self.inference_columns)
            values = state.scaler.transform(np.where(np.isfinite(raw), raw, state.scaler.center_)).astype(np.float32)
            probabilities = np.zeros((len(subset), self.config.components), dtype=np.float32)
            if state.classifier is None:
                # Degenerate support is normally rejected, but retaining this
                # path makes a small fold explicit rather than accidentally NaN.
                rank = int(state.classes[0]) if len(state.classes) else 0
                probabilities[:, rank] = 1.0
            else:
                learned = state.classifier.predict_proba(values)
                for col, rank in enumerate(state.classifier.classes_):
                    probabilities[:, int(rank)] = learned[:, col]
            for rank in range(self.config.components):
                out.loc[subset.index, f"{self.prefix}prob__{rank}"] = probabilities[:, rank]
            out.loc[subset.index, f"{self.prefix}{UNKNOWN_SUFFIX}"] = 0.0
            out.loc[subset.index, f"{self.prefix}entropy"] = _entropy(probabilities)
            out.loc[subset.index, f"{self.prefix}confidence"] = probabilities.max(axis=1)
            out.loc[subset.index, f"{self.prefix}available"] = 1.0
        return out.astype(np.float32)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": STAGE_VI_SCHEMA,
            "view": self.config.view.name,
            "view_kind": self.config.view.kind,
            "side_local_construction": True,
            "positive_label_only_for_discovery": True,
            "soft_memberships_only": True,
            "hard_routing": False,
            "local_trading_experts": False,
            "path_memberships_diagnostic_until_strict_oof_recogniser": self.config.view.kind == "path",
            "inference_columns": self.inference_columns,
            "discovery_columns": list(self.config.view.columns),
            "excluded_current_archetype_tokens": list(CURRENT_ARCHETYPE_TOKENS),
            "discovery_embedding": (
                "pca" if self.config.method.startswith("gmm_pca_")
                else "small_ae" if self.config.method.startswith("ae_gmm_") else "none"
            ),
            "regularized_covariance": self.config.method != "kmeans",
            "ae_bounded_deterministic": self.config.method.startswith("ae_gmm_"),
            "fitted_sides": sorted(self.side_models),
        }


def fit_side_local_archetypes(
    train: pd.DataFrame,
    *,
    config: ArchetypeConfig,
    causal_recogniser_columns: Sequence[str] | None = None,
) -> SideLocalArchetypeState:
    """Fit a state with explicit train-only discovery/recogniser scalers."""

    return SideLocalArchetypeState(
        config, causal_recogniser_columns=causal_recogniser_columns
    ).fit(train)


def _strict_boundaries(decision: pd.Series, folds: int) -> list[pd.Timestamp]:
    unique = np.sort(pd.unique(decision.to_numpy(dtype="datetime64[ns]")))
    if len(unique) < 2:
        return []
    cuts = np.linspace(0, len(unique), int(folds) + 2, dtype=np.int64)[1:-1]
    return [pd.Timestamp(unique[cut], tz="UTC") for cut in np.unique(cuts) if 0 < cut < len(unique)]


def _soft_scores(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(truth).all(axis=1)
    if not valid.any():
        return {"membership_log_loss": np.nan, "membership_brier": np.nan, "membership_rps": np.nan, "diagnostic_rows": 0}
    p = np.clip(predicted[valid], 1e-8, 1.0)
    y = truth[valid]
    log_loss = float(np.mean(-np.sum(y * np.log(p), axis=1)))
    brier = float(np.mean(np.mean((p - y) ** 2, axis=1)))
    # Ranked probability score is reported with train-ranked cluster slots;
    # it is a diagnostic only, not an assumption of a path ordering.
    rps = float(np.mean(np.sum((np.cumsum(p, axis=1) - np.cumsum(y, axis=1)) ** 2, axis=1) / max(p.shape[1] - 1, 1)))
    return {"membership_log_loss": log_loss, "membership_brier": brier, "membership_rps": rps, "diagnostic_rows": int(valid.sum())}


def _membership_arrays(
    predicted: pd.DataFrame | np.ndarray,
    truth: pd.DataFrame | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = predicted.to_numpy(dtype=float) if isinstance(predicted, pd.DataFrame) else np.asarray(predicted, dtype=float)
    y = truth.to_numpy(dtype=float) if isinstance(truth, pd.DataFrame) else np.asarray(truth, dtype=float)
    if p.ndim != 2 or y.ndim != 2 or p.shape != y.shape:
        raise ValueError("predicted and truth memberships must be equally shaped rows × components matrices")
    valid = np.isfinite(p).all(axis=1) & np.isfinite(y).all(axis=1)
    if valid.any():
        p = np.clip(p[valid], 1e-8, 1.0)
        p /= p.sum(axis=1, keepdims=True)
        y = np.clip(y[valid], 0.0, 1.0)
        total = y.sum(axis=1, keepdims=True)
        keep = total[:, 0] > 0
        p, y = p[keep], y[keep] / total[keep]
        source_positions = np.flatnonzero(valid)
        valid = np.zeros(len(valid), dtype=bool)
        valid[source_positions[keep]] = True
    return p, y, valid


def _class_calibration(predicted: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    intercept = np.full(predicted.shape[1], np.nan, dtype=float)
    slope = np.full(predicted.shape[1], np.nan, dtype=float)
    ece = np.full(predicted.shape[1], np.nan, dtype=float)
    for component in range(predicted.shape[1]):
        p, y = predicted[:, component], truth[:, component]
        if len(p) < 2 or np.nanstd(p) <= 1e-12:
            continue
        design = np.column_stack([np.ones(len(p)), p])
        fitted = np.linalg.lstsq(design, y, rcond=None)[0]
        intercept[component], slope[component] = fitted
        # Fixed bins make comparisons deterministic across folds/arms.
        bins = np.clip((p * 10).astype(int), 0, 9)
        ece[component] = sum(
            abs(float(y[bins == bucket].mean()) - float(p[bins == bucket].mean())) * float((bins == bucket).mean())
            for bucket in range(10) if (bins == bucket).any()
        )
    return intercept, slope, ece


def _finite_mean(values: Sequence[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    return float(array[np.isfinite(array)].mean()) if np.isfinite(array).any() else np.nan


def archetype_membership_validation(
    predicted: pd.DataFrame | np.ndarray,
    truth: pd.DataFrame | np.ndarray,
    *,
    prior_cluster_economic_bps: Sequence[float],
    realised_net_bps: Sequence[float] | np.ndarray | pd.Series | None = None,
    top_fraction: float = 0.10,
) -> ArchetypePredictiveValidation:
    """Evaluate causal path-membership forecasts without outcome remapping.

    The reported top-decile enrichment is per membership (not a local trading
    tail): it asks whether the rows most likely to belong to an archetype
    actually carry more of that realised soft membership.  Economic confusion
    is the bps loss induced by replacing truth memberships with causal
    memberships against a *prior-resolved* cluster payoff vector.
    """

    if not 0 < float(top_fraction) <= 1:
        raise ValueError("top_fraction must be in (0, 1]")
    p, y, valid = _membership_arrays(predicted, truth)
    values = np.asarray(prior_cluster_economic_bps, dtype=float)
    if values.ndim != 1 or values.shape[0] != (p.shape[1] if p.ndim == 2 else len(values)) or not np.isfinite(values).all():
        raise ValueError("prior_cluster_economic_bps must be finite and match membership components")
    if len(p) == 0:
        empty = pd.DataFrame(columns=["metric", "value"])
        return ArchetypePredictiveValidation(empty, pd.DataFrame())
    intercept, slope, ece = _class_calibration(p, y)
    rows: list[dict[str, Any]] = []
    count = max(1, int(np.ceil(len(p) * float(top_fraction))))
    for component in range(p.shape[1]):
        correlation = np.nan
        if np.std(p[:, component]) > 1e-12 and np.std(y[:, component]) > 1e-12:
            correlation = float(np.corrcoef(p[:, component], y[:, component])[0, 1])
        order = np.argsort(-p[:, component], kind="stable")[:count]
        baseline = float(y[:, component].mean())
        top = float(y[order, component].mean())
        rows.append({
            "component": component,
            "rows": int(len(p)),
            "calibration_intercept": float(intercept[component]),
            "calibration_slope": float(slope[component]),
            "calibration_ece": float(ece[component]),
            "membership_correlation": correlation,
            "top_decile_truth_membership": top,
            "overall_truth_membership": baseline,
            "top_decile_enrichment": top / baseline if baseline > 1e-12 else np.nan,
        })
    scores = _soft_scores(p, y)
    economic_confusion = np.abs((p - y) @ values)
    summary: dict[str, Any] = {
        **scores,
        "rows": int(len(p)),
        "mean_calibration_intercept": _finite_mean(intercept),
        "mean_calibration_slope": _finite_mean(slope),
        "mean_calibration_ece": _finite_mean(ece),
        "mean_membership_correlation": _finite_mean([row["membership_correlation"] for row in rows]),
        "mean_top_decile_enrichment": _finite_mean([row["top_decile_enrichment"] for row in rows]),
        "economic_confusion_cost_bps": float(np.mean(economic_confusion)),
        "economic_confusion_p90_bps": float(np.quantile(economic_confusion, 0.90)),
        "prior_payoff_map_only": True,
    }
    if realised_net_bps is not None:
        realised = np.asarray(realised_net_bps, dtype=float)
        if realised.ndim != 1 or len(realised) != len(valid):
            raise ValueError("realised_net_bps must align to the original membership rows")
        realised = realised[valid]
        if np.isfinite(realised).any():
            summary["causal_membership_net_mae_bps"] = float(np.mean(np.abs(realised[np.isfinite(realised)] - (p @ values)[np.isfinite(realised)])))
    return ArchetypePredictiveValidation(pd.DataFrame([summary]), pd.DataFrame(rows))


def _prior_cluster_economic_map(
    memberships: pd.DataFrame,
    net_bps: pd.Series,
) -> np.ndarray:
    """Soft cluster payoff map from a caller's already-prior training rows."""

    matrix = memberships.to_numpy(dtype=float)
    values = pd.to_numeric(net_bps, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(matrix).all(axis=1) & np.isfinite(values)
    if not valid.any():
        return np.full(matrix.shape[1], 0.0, dtype=float)
    numerator = matrix[valid].T @ values[valid]
    denominator = matrix[valid].sum(axis=0)
    fallback = float(np.mean(values[valid]))
    return np.where(denominator > 1e-8, numerator / denominator, fallback).astype(float)


def strict_oof_archetype_features(
    frame: pd.DataFrame,
    *,
    config: ArchetypeConfig,
    causal_recogniser_columns: Sequence[str] | None = None,
    folds: int = 4,
) -> ArchetypeOOFResult:
    """Build strict chronological OOF memberships for either Stage-VI view.

    A validation block uses only labels resolved before its first decision.  A
    causal-feature cluster is also trained only on prior *positive* outcomes;
    it then transforms every candidate in the valid block.  A path cluster is
    fitted only on prior realised paths and exposes the causal recogniser—not
    the realised membership—to the output feature frame.
    """

    _validate_columns(frame, config)
    if config.view.kind == "path" and not causal_recogniser_columns:
        raise ValueError("strict path archetypes require causal_recogniser_columns")
    if causal_recogniser_columns:
        # Give causal recogniser inputs the same strong no-outcome contract.
        temporary = ArchetypeConfig(
            view=ArchetypeView("strict_recogniser", tuple(causal_recogniser_columns), "causal"),
            method=config.method, components=config.components, side_col=config.side_col,
            decision_ts_col=config.decision_ts_col, label_available_ts_col=config.label_available_ts_col,
            positive_label_col=config.positive_label_col, min_positive_value=config.min_positive_value,
            min_side_rows=config.min_side_rows, min_component_rows=config.min_component_rows,
            classifier_c=config.classifier_c, reg_covar=config.reg_covar,
            embedding_dimensions=config.embedding_dimensions, ae_hidden_units=config.ae_hidden_units,
            ae_max_iter=config.ae_max_iter, ae_alpha=config.ae_alpha,
            random_state=config.random_state,
            weights=config.weights,
        )
        _validate_columns(frame, temporary)
    decision = _timestamps(frame[config.decision_ts_col], config.decision_ts_col)
    available = _timestamps(frame[config.label_available_ts_col], config.label_available_ts_col)
    names = archetype_feature_names(CAUSAL_PREFIX if config.view.kind == "causal" else PATH_PREFIX, config.components)
    features = pd.DataFrame(0.0, index=frame.index, columns=names, dtype=np.float32)
    prefix = CAUSAL_PREFIX if config.view.kind == "causal" else PATH_PREFIX
    features[f"{prefix}{UNKNOWN_SUFFIX}"] = 1.0
    truth = pd.DataFrame(np.nan, index=frame.index, columns=[f"{prefix}prob__{i}" for i in range(config.components)], dtype=np.float32)
    audit: list[dict[str, Any]] = []
    catalogues: list[pd.DataFrame] = []
    boundaries = _strict_boundaries(decision, folds)
    for fold, start in enumerate(boundaries):
        end = boundaries[fold + 1] if fold + 1 < len(boundaries) else None
        valid_mask = decision.ge(start) if end is None else decision.ge(start) & decision.lt(end)
        train_mask = available.lt(start) & _positive_mask(frame, config)
        train = frame.loc[train_mask]
        valid = frame.loc[valid_mask]
        if valid.empty:
            continue
        if len(train) < int(config.min_side_rows):
            audit.append({"fold": fold, "valid_start": start, "valid_end": end, "train_rows": int(len(train)), "valid_rows": int(len(valid)), "status": "insufficient_prior_positive_rows", "train_max_label_available_ts": available.loc[train_mask].max() if train_mask.any() else pd.NaT})
            continue
        state = fit_side_local_archetypes(train, config=config, causal_recogniser_columns=causal_recogniser_columns)
        safe = valid.drop(columns=list(config.view.columns), errors="ignore") if config.view.kind == "path" else valid.copy()
        # Positive label and timestamps are allowed to remain for causal view
        # only because transform reads an explicit causal column list.  Remove
        # them nevertheless to make accidental future use impossible.
        # A source ledger may carry several unused realised outcomes alongside
        # the requested path coordinates.  Strip every recognised outcome-like
        # field before the inference transform so path recognisers cannot pick
        # it up through a future refactor.
        unsafe = [
            name for name in safe.columns
            if name in {config.positive_label_col, config.label_available_ts_col}
            or any(token in str(name).lower() for token in FORBIDDEN_CAUSAL_TOKENS)
        ]
        safe = safe.drop(columns=unsafe, errors="ignore")
        transformed = state.transform(safe)
        features.loc[valid.index, transformed.columns] = transformed
        diagnostic = state.diagnostic_realised_memberships(valid)
        truth.loc[valid.index, diagnostic.columns] = diagnostic
        score = _soft_scores(transformed.loc[:, diagnostic.columns].to_numpy(dtype=float), diagnostic.to_numpy(dtype=float))
        if config.view.kind == "path":
            # The payoff maps are fit separately by side from *this fold's
            # prior-resolved rows*.  Evaluation labels only score the map;
            # they never alter it or the recogniser.
            validation_parts: list[pd.DataFrame] = []
            train_side = train[config.side_col].astype(str).str.lower()
            valid_side = valid[config.side_col].astype(str).str.lower()
            train_truth = state.diagnostic_realised_memberships(train)
            for side_name in sorted(set(valid_side)):
                train_idx = train_side.eq(side_name)
                valid_idx = valid_side.eq(side_name)
                if not train_idx.any() or not valid_idx.any():
                    continue
                prior_map = _prior_cluster_economic_map(
                    train_truth.loc[train_idx], train.loc[train_idx, config.positive_label_col],
                )
                checked = archetype_membership_validation(
                    transformed.loc[valid_idx, diagnostic.columns], diagnostic.loc[valid_idx],
                    prior_cluster_economic_bps=prior_map,
                    realised_net_bps=valid.loc[valid_idx, config.positive_label_col],
                )
                if not checked.summary.empty:
                    validation_parts.append(checked.summary)
            if validation_parts:
                checked_summary = pd.concat(validation_parts, ignore_index=True)
                for column in (
                    "mean_calibration_intercept", "mean_calibration_slope", "mean_calibration_ece", "mean_membership_correlation",
                    "mean_top_decile_enrichment", "economic_confusion_cost_bps",
                    "economic_confusion_p90_bps", "causal_membership_net_mae_bps",
                ):
                    if column in checked_summary:
                        score[column] = float(checked_summary[column].mean())
        catalogue = state.catalog_.copy()
        if not catalogue.empty:
            catalogue.insert(0, "fold", fold)
            catalogue.insert(1, "valid_start", start)
            catalogues.append(catalogue)
        audit.append({
            "fold": fold, "valid_start": start, "valid_end": end, "train_rows": int(len(train)), "valid_rows": int(len(valid)),
            "status": "scored", "train_max_label_available_ts": available.loc[train_mask].max(),
            "fitted_sides": ",".join(sorted(state.side_models)), **score,
        })
    fold_audit = pd.DataFrame(audit)
    return ArchetypeOOFResult(
        features=features,
        diagnostic_truth_memberships=truth,
        fold_audit=fold_audit,
        catalog=pd.concat(catalogues, ignore_index=True) if catalogues else pd.DataFrame(),
        manifest={
            "schema": STAGE_VI_SCHEMA,
            "strict_oof": True,
            "prior_resolution_rule": f"{config.label_available_ts_col} < validation_decision_ts",
            "side_local_construction": True,
            "positive_label_only_for_discovery": True,
            "soft_memberships_only": True,
            "hard_routing": False,
            "local_trading_experts": False,
            "path_truth_memberships": "diagnostic_only_not_model_features",
            "scored_rows": int(features[f"{prefix}available"].sum()),
            "unknown_rows": int(features[f"{prefix}{UNKNOWN_SUFFIX}"].sum()),
            "view": config.view.name,
            "view_kind": config.view.kind,
        },
    )


def align_archetype_catalogues(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    centroid_column: str = "centroid",
    side_column: str = "side",
    rank_column: str = "rank",
) -> pd.DataFrame:
    """Align independently fit fold clusters with optimal centroid assignment."""

    required = {centroid_column, side_column, rank_column}
    if missing := required.difference(reference.columns):
        raise KeyError(f"reference catalogue missing {sorted(missing)}")
    if missing := required.difference(candidate.columns):
        raise KeyError(f"candidate catalogue missing {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for side in sorted(set(reference[side_column].astype(str)).intersection(candidate[side_column].astype(str))):
        left = reference.loc[reference[side_column].astype(str).eq(side)].sort_values(rank_column, kind="stable")
        right = candidate.loc[candidate[side_column].astype(str).eq(side)].sort_values(rank_column, kind="stable")
        if left.empty or right.empty:
            continue
        a = np.asarray(left[centroid_column].tolist(), dtype=float)
        b = np.asarray(right[centroid_column].tolist(), dtype=float)
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
            raise ValueError("catalogue centroids must be equal-width numeric vectors")
        distance = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))
        ii, jj = linear_sum_assignment(distance)
        for i, j in zip(ii, jj):
            rows.append({
                "side": side, "reference_rank": int(left.iloc[i][rank_column]),
                "candidate_rank": int(right.iloc[j][rank_column]),
                "centroid_distance": float(distance[i, j]),
            })
    return pd.DataFrame(rows)


def archetype_alignment_switching(
    reference_catalog: pd.DataFrame,
    candidate_catalog: pd.DataFrame,
    reference_economics: pd.DataFrame,
    candidate_economics: pd.DataFrame,
    *,
    economic_columns: Sequence[str],
    side_column: str = "side",
    rank_column: str = "rank",
) -> pd.DataFrame:
    """Report whether geometric and economic semantic alignment disagree.

    Component numbers are arbitrary.  We therefore calculate two independent
    optimal assignments: centroid geometry and declared realised-economic
    semantics.  A switch means the candidate component paired by geometry is
    not the one paired by economics, a direct instability signal rather than
    an accidental sklearn label permutation.
    """

    if not economic_columns:
        raise ValueError("economic alignment requires predeclared economics columns")
    required = {side_column, rank_column, *economic_columns}
    for name, table in (("reference economics", reference_economics), ("candidate economics", candidate_economics)):
        missing = required.difference(table.columns)
        if missing:
            raise KeyError(f"{name} missing {sorted(missing)}")
    centroid_assignment = align_archetype_catalogues(
        reference_catalog, candidate_catalog, side_column=side_column, rank_column=rank_column,
    )
    rows: list[dict[str, Any]] = []
    for side in sorted(set(reference_economics[side_column].astype(str)).intersection(candidate_economics[side_column].astype(str))):
        left = reference_economics.loc[reference_economics[side_column].astype(str).eq(side)].sort_values(rank_column, kind="stable")
        right = candidate_economics.loc[candidate_economics[side_column].astype(str).eq(side)].sort_values(rank_column, kind="stable")
        if left.empty or right.empty:
            continue
        a = left.loc[:, list(economic_columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        b = right.loc[:, list(economic_columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("economic semantic inputs must be finite")
        combined = np.vstack([a, b])
        scale = np.nanstd(combined, axis=0)
        scale = np.where(scale > 1e-8, scale, 1.0)
        economic_distance = np.sqrt((((a[:, None, :] - b[None, :, :]) / scale) ** 2).sum(axis=2))
        ii, jj = linear_sum_assignment(economic_distance)
        centroid_side = centroid_assignment.loc[centroid_assignment.side.eq(side)]
        centroid_map = dict(zip(centroid_side.reference_rank.astype(int), centroid_side.candidate_rank.astype(int)))
        for i, j in zip(ii, jj):
            reference_rank = int(left.iloc[i][rank_column])
            economic_rank = int(right.iloc[j][rank_column])
            geometry_rank = centroid_map.get(reference_rank)
            rows.append({
                "side": side,
                "reference_rank": reference_rank,
                "geometry_candidate_rank": geometry_rank,
                "economic_candidate_rank": economic_rank,
                "semantic_alignment_switch": bool(geometry_rank != economic_rank),
                "economic_semantic_distance": float(economic_distance[i, j]),
            })
    return pd.DataFrame(rows)


def archetype_fold_stability(
    catalog: pd.DataFrame,
    *,
    fold_column: str = "fold",
    side_column: str = "side",
    economic_catalog: pd.DataFrame | None = None,
    economic_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Align adjacent fold catalogues and report a compact stability surface.

    Cluster identifiers have no inherent cross-fold identity.  This function
    always uses optimal centroid assignment before aggregating distance, so an
    apparent change cannot be caused merely by sklearn permuting component
    numbers.  Economic separation is intentionally reported elsewhere from
    held-out rows; it is never used to choose this alignment.
    """

    if catalog.empty:
        return pd.DataFrame(columns=["reference_fold", "candidate_fold", "side", "matched_components", "mean_centroid_distance", "max_centroid_distance"])
    required = {fold_column, side_column, "rank", "centroid"}
    if missing := required.difference(catalog.columns):
        raise KeyError(f"catalogue missing stability fields: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    folds = sorted(pd.unique(catalog[fold_column]))
    for previous, current in zip(folds[:-1], folds[1:]):
        aligned = align_archetype_catalogues(
            catalog.loc[catalog[fold_column].eq(previous)],
            catalog.loc[catalog[fold_column].eq(current)],
            side_column=side_column,
        )
        if aligned.empty:
            continue
        for side, subset in aligned.groupby("side", observed=True, sort=True):
            row: dict[str, Any] = {
                "reference_fold": previous, "candidate_fold": current, "side": str(side),
                "matched_components": int(len(subset)),
                "mean_centroid_distance": float(subset["centroid_distance"].mean()),
                "max_centroid_distance": float(subset["centroid_distance"].max()),
            }
            if economic_catalog is not None:
                if not economic_columns:
                    raise ValueError("economic_columns are required with economic_catalog")
                ref_economics = economic_catalog.loc[economic_catalog[fold_column].eq(previous)]
                candidate_economics = economic_catalog.loc[economic_catalog[fold_column].eq(current)]
                switching = archetype_alignment_switching(
                    catalog.loc[catalog[fold_column].eq(previous)], catalog.loc[catalog[fold_column].eq(current)],
                    ref_economics, candidate_economics, economic_columns=economic_columns,
                    side_column=side_column,
                )
                switching = switching.loc[switching.side.eq(str(side))]
                row["semantic_alignment_switch_rate"] = float(switching["semantic_alignment_switch"].mean()) if not switching.empty else np.nan
                row["mean_economic_semantic_distance"] = float(switching["economic_semantic_distance"].mean()) if not switching.empty else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def archetype_economic_separation(
    frame: pd.DataFrame,
    memberships: pd.DataFrame | np.ndarray,
    *,
    outcome_columns: Sequence[str],
    timestamp_col: str = "decision_ts",
    side_col: str = "side_name",
    symbol_col: str = "symbol",
    by_month: bool = True,
) -> pd.DataFrame:
    """Report support, future economics, time/side/symbol concentration.

    This is evaluation-only: it deliberately receives realised outcomes after
    cluster probabilities are frozen.  It supports gross/net/event/MFE/MAE and
    any further numeric path column without baking an outcome schema into the
    representation state.
    """

    missing = [name for name in [timestamp_col, side_col, *outcome_columns] if name not in frame]
    if missing:
        raise KeyError(f"economic separation frame missing {missing[:8]}")
    matrix = memberships.to_numpy(dtype=float) if isinstance(memberships, pd.DataFrame) else np.asarray(memberships, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != len(frame):
        raise ValueError("memberships must be a rows × components matrix aligned to frame")
    hard = np.nanargmax(np.where(np.isfinite(matrix), matrix, -np.inf), axis=1)
    work = frame.copy()
    work["__cluster__"] = hard.astype(np.int32)
    work["__month__"] = _timestamps(work[timestamp_col], timestamp_col).dt.strftime("%Y-%m")
    transition_by_side: dict[str, float] = {}
    for side, side_rows in work.groupby(side_col, observed=True, sort=False):
        sequence = side_rows.sort_values(timestamp_col, kind="stable")["__cluster__"]
        transition_by_side[str(side)] = float(sequence.diff().ne(0).iloc[1:].mean()) if len(sequence) > 1 else np.nan
    rows: list[dict[str, Any]] = []
    groups = [side_col, "__cluster__", "__month__"] if by_month else [side_col, "__cluster__"]
    for keys, subset in work.groupby(groups, observed=True, sort=True):
        side, cluster = keys[:2] if isinstance(keys, tuple) else (keys, 0)
        record: dict[str, Any] = {
            "side": str(side), "cluster": int(cluster), "rows": int(len(subset)),
            "transition_rate": transition_by_side.get(str(side), np.nan),
        }
        if by_month:
            record["month"] = str(keys[2])
        if symbol_col in subset:
            counts = subset[symbol_col].astype("string").value_counts(normalize=True, dropna=False)
            record["top_symbol_share"] = float(counts.iloc[0]) if len(counts) else np.nan
        for column in outcome_columns:
            values = pd.to_numeric(subset[column], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            record[f"{column}__mean"] = float(finite.mean()) if len(finite) else np.nan
            record[f"{column}__q10"] = float(np.quantile(finite, 0.10)) if len(finite) else np.nan
            record[f"{column}__q50"] = float(np.quantile(finite, 0.50)) if len(finite) else np.nan
            record[f"{column}__q90"] = float(np.quantile(finite, 0.90)) if len(finite) else np.nan
        rows.append(record)
    return pd.DataFrame(rows).sort_values(["side", "cluster"], kind="stable").reset_index(drop=True) if rows else pd.DataFrame()


def run_matched_incremental_archetype_comparison(
    ledger: pd.DataFrame,
    *,
    arm_score_columns: Mapping[str, str],
    net_bps_col: str,
    gross_bps_col: str | None = None,
    identity_columns: Sequence[str] = ("candidate_id", "symbol", "decision_ts", "side_name"),
    top_fractions: Sequence[float] = (0.01, 0.05, 0.10),
) -> pd.DataFrame:
    """Materialise the matched control/base/meta/both incremental test.

    One ledger is intentional: every arm sees the exact same candidate IDs,
    resolved outcomes, cost convention and chronological evaluation rows.  The
    function selects globally by score (never per timestamp), so changes are
    attributable to the added archetype probabilities rather than a silently
    changed sample or local ranking rule.
    """

    required_arms = {"control", "base", "meta", "both"}
    if set(arm_score_columns) != required_arms:
        raise ValueError("arm_score_columns must contain exactly control, base, meta, and both")
    identities = list(identity_columns)
    required = [*identities, net_bps_col, *arm_score_columns.values()]
    if gross_bps_col:
        required.append(gross_bps_col)
    missing = [name for name in dict.fromkeys(required) if name not in ledger]
    if missing:
        raise KeyError(f"matched archetype comparison missing {missing[:8]}")
    if ledger.loc[:, identities].isna().any().any() or ledger.duplicated(identities).any():
        raise ValueError("matched archetype comparison requires unique non-null immutable identities")
    if not top_fractions or any(not 0 < float(value) <= 1 for value in top_fractions):
        raise ValueError("top_fractions must be non-empty values in (0, 1]")
    work = ledger.copy()
    outcome = pd.to_numeric(work[net_bps_col], errors="coerce")
    if outcome.isna().any():
        raise ValueError("matched archetype comparison cannot silently fill unresolved net outcomes")
    score_values = {
        arm: pd.to_numeric(work[column], errors="coerce") for arm, column in arm_score_columns.items()
    }
    if any(values.isna().any() or not np.isfinite(values).all() for values in score_values.values()):
        raise ValueError("every matched arm needs a finite score for every candidate")
    gross = pd.to_numeric(work[gross_bps_col], errors="coerce") if gross_bps_col else None
    if gross is not None and gross.isna().any():
        raise ValueError("gross outcomes must be resolved when supplied")
    rows: list[dict[str, Any]] = []
    for fraction in sorted(set(float(value) for value in top_fractions)):
        count = max(1, int(np.ceil(len(work) * fraction)))
        selected_by_arm: dict[str, pd.Index] = {
            arm: scores.sort_values(ascending=False, kind="stable").index[:count]
            for arm, scores in score_values.items()
        }
        control_mean = float(outcome.loc[selected_by_arm["control"]].mean())
        for arm, selected in selected_by_arm.items():
            record = {
                "arm": arm, "tail_fraction": fraction, "selected_rows": int(len(selected)),
                "net_bps_per_trade": float(outcome.loc[selected].mean()),
                "net_bps_sum": float(outcome.loc[selected].sum()),
                "net_positive_rate": float((outcome.loc[selected] > 0).mean()),
                "delta_net_bps_per_trade_vs_control": float(outcome.loc[selected].mean() - control_mean),
                "selection_jaccard_vs_control": float(
                    len(set(selected).intersection(selected_by_arm["control"]))
                    / len(set(selected).union(selected_by_arm["control"]))
                ),
                "global_ranking": True,
                "matched_candidate_population": True,
            }
            if gross is not None:
                record["gross_bps_per_trade"] = float(gross.loc[selected].mean())
                record["gross_bps_sum"] = float(gross.loc[selected].sum())
            rows.append(record)
    return pd.DataFrame(rows).sort_values(["tail_fraction", "arm"], kind="stable").reset_index(drop=True)


def materialize_multiview_composite_objective(
    metrics: pd.DataFrame,
    *,
    config: MultiViewObjectiveConfig = MultiViewObjectiveConfig(),
) -> pd.DataFrame:
    """Score multi-view candidates on path/economics/predictability/stability.

    Inputs are pre-normalised [0, 1] diagnostics.  Explicitly refusing raw
    silhouette or bps scale avoids an arbitrary geometric metric dominating
    causal predictability or economic separation.
    """

    config.validate()
    needed = [
        "path_separation", "economic_separation", "causal_predictability",
        "temporal_stability", "concentration",
    ]
    missing = [name for name in needed if name not in metrics]
    if missing:
        raise KeyError(f"multi-view objective missing {missing}")
    result = metrics.copy()
    values = result.loc[:, needed].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any() or ((values < 0) | (values > 1)).any().any():
        raise ValueError("multi-view metrics must be finite values in [0, 1]")
    result["cluster_score"] = (
        config.path_separation_weight * values["path_separation"]
        + config.economic_separation_weight * values["economic_separation"]
        + config.causal_predictability_weight * values["causal_predictability"]
        + config.temporal_stability_weight * values["temporal_stability"]
        - config.concentration_penalty * values["concentration"]
    )
    result["composite_objective"] = "path+economics+causal_predictability+temporal_stability-concentration"
    return result


def materialize_archetype_decision_matrix(
    metrics: pd.DataFrame,
    *,
    config: ArchetypeDecisionConfig = ArchetypeDecisionConfig(),
) -> pd.DataFrame:
    """Turn predeclared Stage-VI evidence into a non-routing disposition."""

    config.validate()
    required = [
        "economic_separation", "causal_predictability", "temporal_stability", "concentration",
        "base_incremental_bps", "meta_incremental_bps",
    ]
    missing = [name for name in required if name not in metrics]
    if missing:
        raise KeyError(f"decision matrix missing {missing}")
    output = metrics.copy()
    dispositions: list[str] = []
    for _, row in output.iterrows():
        economic = float(row["economic_separation"]) >= config.minimum_economic_separation
        predictable = float(row["causal_predictability"]) >= config.minimum_causal_predictability
        stable = float(row["temporal_stability"]) >= config.minimum_temporal_stability
        concentrated = float(row["concentration"]) > config.maximum_concentration
        base = float(row["base_incremental_bps"]) > config.minimum_incremental_bps
        meta = float(row["meta_incremental_bps"]) > config.minimum_incremental_bps
        soft_only = bool(row.get("hard_label_value", 0.0) <= 0 and row.get("soft_membership_value", 0.0) > 0)
        side_specific = bool(row.get("side_specific_only", False))
        if not stable or concentrated:
            disposition = "Reject or reduce K"
        elif economic and not predictable:
            disposition = "Diagnostic/path research only"
        elif predictable and not economic:
            disposition = "Reject"
        elif side_specific and (base or meta):
            disposition = "Side-specific research"
        elif base:
            disposition = "Retained base context"
        elif meta:
            disposition = "Retained trust/residual context"
        elif soft_only:
            disposition = "Retain soft memberships only"
        else:
            disposition = "Diagnostic only"
        dispositions.append(disposition)
    output["disposition"] = dispositions
    output["hard_routing"] = False
    output["local_trading_experts"] = False
    return output
