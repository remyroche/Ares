"""Robust mixture models for descriptive failure-episode taxonomy."""

from __future__ import annotations

from dataclasses import dataclass
from math import lgamma
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPRegressor

from extreme_price_movements.alternative_latent_encoders import (
    AlternativeLatentEncoder,
    EncoderConfig,
)

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - scipy is a sklearn dependency in production.
    linear_sum_assignment = None


@dataclass(frozen=True)
class FailureTaxonomyModelConfig:
    latent_dims: tuple[int, ...] = (2, 4, 8)
    cluster_counts: tuple[int, ...] = (2, 3, 4, 5, 6, 8)
    methods: tuple[str, ...] = (
        "pca_student_t",
        "pca_gmm",
        "small_dae_gmm",
        "vicreg_gmm",
    )
    student_df: float = 5.0
    max_iter: int = 250
    tolerance: float = 1e-5
    reg_covar: float = 1e-3
    min_cluster_episodes: int = 3
    stability_seeds: tuple[int, ...] = (17, 29, 43, 71)
    multi_view_enabled: bool = True
    min_view_features: int = 1
    episode_bootstrap_repeats: int = 3
    episode_bootstrap_fraction: float = 0.80
    learned_encoder_epochs: int = 12
    learned_encoder_weight_decay: float = 5e-4
    learned_encoder_noise_std: float = 0.08
    recovery_horizon_days: int = 14
    outcome_resolution_days: int = 1
    random_state: int = 20260719


class DiagonalStudentTMixture:
    """Small diagonal Student-t mixture suited to heavy-tailed episodes."""

    def __init__(
        self,
        n_components: int,
        *,
        degrees_of_freedom: float = 5.0,
        reg_covar: float = 1e-3,
        max_iter: int = 250,
        tolerance: float = 1e-5,
        random_state: int = 0,
    ) -> None:
        self.n_components = int(n_components)
        self.degrees_of_freedom = float(degrees_of_freedom)
        self.reg_covar = float(reg_covar)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        self.random_state = int(random_state)

    def _log_density(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nu = self.degrees_of_freedom
        dimensions = matrix.shape[1]
        delta = np.empty((len(matrix), self.n_components), dtype=np.float64)
        log_density = np.empty_like(delta)
        constant = lgamma((nu + dimensions) / 2.0) - lgamma(nu / 2.0)
        constant -= 0.5 * dimensions * np.log(nu * np.pi)
        for component in range(self.n_components):
            variance = np.maximum(self.variances_[component], self.reg_covar)
            centered = matrix - self.means_[component]
            delta[:, component] = np.sum(centered * centered / variance, axis=1)
            log_density[:, component] = (
                constant
                - 0.5 * np.log(variance).sum()
                - 0.5 * (nu + dimensions) * np.log1p(delta[:, component] / nu)
                + np.log(max(self.weights_[component], 1e-12))
            )
        return log_density, delta

    def fit(self, matrix: np.ndarray) -> "DiagonalStudentTMixture":
        values = np.asarray(matrix, dtype=np.float64)
        if values.ndim != 2 or len(values) < self.n_components:
            raise ValueError("Student-t mixture needs a non-empty 2D matrix")
        initializer = GaussianMixture(
            n_components=self.n_components,
            covariance_type="diag",
            reg_covar=self.reg_covar,
            n_init=1,
            max_iter=100,
            random_state=self.random_state,
        ).fit(values)
        self.weights_ = initializer.weights_.astype(np.float64)
        self.means_ = initializer.means_.astype(np.float64)
        self.variances_ = initializer.covariances_.astype(np.float64)
        previous = -np.inf
        nu = self.degrees_of_freedom
        dimensions = values.shape[1]
        for iteration in range(self.max_iter):
            log_density, delta = self._log_density(values)
            maximum = log_density.max(axis=1, keepdims=True)
            exp_values = np.exp(log_density - maximum)
            responsibilities = exp_values / np.maximum(
                exp_values.sum(axis=1, keepdims=True), 1e-12
            )
            scales = (nu + dimensions) / np.maximum(nu + delta, 1e-12)
            effective = responsibilities * scales
            component_mass = responsibilities.sum(axis=0).clip(min=1e-8)
            weighted_mass = effective.sum(axis=0).clip(min=1e-8)
            means = (effective.T @ values) / weighted_mass[:, None]
            variances = np.empty_like(self.variances_)
            for component in range(self.n_components):
                centered = values - means[component]
                variances[component] = (
                    effective[:, component, None] * centered * centered
                ).sum(axis=0) / component_mass[component]
            self.weights_ = component_mass / len(values)
            self.means_ = means
            self.variances_ = np.maximum(variances, self.reg_covar)
            objective = float(
                np.sum(
                    maximum[:, 0] + np.log(np.maximum(exp_values.sum(axis=1), 1e-12))
                )
            )
            self.n_iter_ = iteration + 1
            if np.isfinite(previous) and abs(objective - previous) <= self.tolerance * (
                1.0 + abs(previous)
            ):
                break
            previous = objective
        self.lower_bound_ = previous / max(len(values), 1)
        return self

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        log_density, _ = self._log_density(np.asarray(matrix, dtype=np.float64))
        maximum = log_density.max(axis=1, keepdims=True)
        values = np.exp(log_density - maximum)
        return values / np.maximum(values.sum(axis=1, keepdims=True), 1e-12)

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(matrix), axis=1).astype(np.int16)


def _robust_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    values = (
        frame.loc[:, list(columns)]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(np.float64)
    )
    available = np.isfinite(values).any(axis=0)
    median = np.zeros(values.shape[1], dtype=np.float64)
    scale = np.ones(values.shape[1], dtype=np.float64)
    if available.any():
        present = values[:, available]
        median[available] = np.nanmedian(present, axis=0)
        scale[available] = np.maximum(
            np.nanquantile(present, 0.75, axis=0)
            - np.nanquantile(present, 0.25, axis=0),
            1e-4,
        )
    return np.clip(np.nan_to_num((values - median) / scale, nan=0.0), -8.0, 8.0)


def _fit_labels(
    matrix: np.ndarray,
    *,
    method: str,
    clusters: int,
    seed: int,
    config: FailureTaxonomyModelConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if method == "pca_student_t":
        model: Any = DiagonalStudentTMixture(
            clusters,
            degrees_of_freedom=config.student_df,
            reg_covar=config.reg_covar,
            max_iter=config.max_iter,
            tolerance=config.tolerance,
            random_state=seed,
        ).fit(matrix)
    elif method in {"pca_gmm", "small_dae_gmm", "vicreg_gmm"}:
        model = GaussianMixture(
            n_components=clusters,
            covariance_type="diag",
            reg_covar=config.reg_covar,
            max_iter=config.max_iter,
            n_init=2,
            random_state=seed,
        ).fit(matrix)
    else:
        raise ValueError(f"Unsupported failure taxonomy method: {method}")
    return model.predict(matrix), model.predict_proba(matrix)


def _mixture_method(method: str) -> str:
    return "pca_student_t" if method == "pca_student_t" else "pca_gmm"


def _learned_representation_embeddings(
    matrix: np.ndarray,
    *,
    method: str,
    latent_dim: int,
    config: FailureTaxonomyModelConfig,
) -> dict[int, np.ndarray]:
    """Refit a compact outcome-agnostic encoder for every stability seed.

    The production DAE requires at least 200 rows, which is intentionally too
    strict for many local episode groups. The DAE arm therefore uses a compact,
    strongly regularized sklearn denoising bottleneck local to this descriptive
    taxonomy. VICReg reuses the repository encoder. Each embedding is fitted
    independently; downstream ARI measures representation plus mixture.
    """

    if method == "small_dae_gmm":
        kind = "small_dae"
    elif method == "vicreg_gmm":
        kind = "vicreg"
    else:
        return {}
    output: dict[int, np.ndarray] = {}
    sides = np.repeat("local", len(matrix))
    for seed in config.stability_seeds:
        if kind == "small_dae":
            try:
                output[int(seed)] = _fit_small_denoising_episode_embedding(
                    matrix,
                    latent_dim=int(latent_dim),
                    seed=int(seed),
                    config=config,
                )
            except (RuntimeError, ValueError, FloatingPointError):
                return {}
            continue
        encoder = AlternativeLatentEncoder(
            EncoderConfig(
                kind=kind,
                latent_dim=int(latent_dim),
                hidden_dim=max(24, 6 * int(latent_dim)),
                residual_blocks=1,
                epochs=int(config.learned_encoder_epochs),
                batch_size=min(128, max(16, len(matrix))),
                learning_rate=7.5e-4,
                weight_decay=float(config.learned_encoder_weight_decay),
                corruption_rate=0.15,
                element_mask_rate=0.10,
                additive_noise_std=float(config.learned_encoder_noise_std),
                ssl_objective="vicreg" if kind == "vicreg" else "masked_reconstruction",
                ssl_view_pair="weak_strong",
                reconstruction_weight=0.5 if kind == "vicreg" else 1.0,
                vicreg_variance_weight=20.0,
                vicreg_covariance_weight=2.0,
                random_state=int(seed),
                device="cpu",
                dae_max_train_rows=max(64, len(matrix)),
            )
        )
        try:
            embedding = encoder.fit_transform(matrix, sides=sides)
        except (RuntimeError, ValueError, FloatingPointError):
            return {}
        if embedding.ndim != 2 or len(embedding) != len(matrix):
            return {}
        output[int(seed)] = np.asarray(embedding, dtype=np.float32)
    return output


def _fit_small_denoising_episode_embedding(
    matrix: np.ndarray,
    *,
    latent_dim: int,
    seed: int,
    config: FailureTaxonomyModelConfig,
) -> np.ndarray:
    """Fit a conservative nonlinear denoising bottleneck on episode rows."""

    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2 or len(values) < 8 or values.shape[1] < 2:
        raise ValueError("Small episode DAE requires at least eight rows")
    rng = np.random.default_rng(seed)
    corrupted = values + rng.normal(
        0.0,
        float(config.learned_encoder_noise_std),
        size=values.shape,
    ).astype(np.float32)
    masked = rng.random(values.shape) < 0.10
    corrupted[masked] = 0.0
    hidden = max(12, 4 * int(latent_dim))
    model = MLPRegressor(
        hidden_layer_sizes=(hidden, int(latent_dim), hidden),
        activation="tanh",
        solver="adam",
        alpha=max(float(config.learned_encoder_weight_decay), 1e-3),
        batch_size=max(4, min(32, len(values) // 2)),
        learning_rate_init=7.5e-4,
        max_iter=max(40, 5 * int(config.learned_encoder_epochs)),
        early_stopping=len(values) >= 30,
        validation_fraction=0.20,
        n_iter_no_change=8,
        random_state=int(seed),
    )
    model.fit(corrupted, values)
    hidden_values = np.tanh(values @ model.coefs_[0] + model.intercepts_[0])
    latent = np.tanh(hidden_values @ model.coefs_[1] + model.intercepts_[1])
    return np.asarray(latent, dtype=np.float32)


def _fit_representation_seed_stability(
    embeddings: dict[int, np.ndarray],
    *,
    method: str,
    clusters: int,
    config: FailureTaxonomyModelConfig,
) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    labels_by_seed: list[np.ndarray] = []
    probabilities_by_seed: list[np.ndarray] = []
    for seed in config.stability_seeds:
        embedding = embeddings.get(int(seed))
        if embedding is None:
            return None
        try:
            labels, probabilities = _fit_labels(
                embedding,
                method=_mixture_method(method),
                clusters=clusters,
                seed=int(seed),
                config=config,
            )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            return None
        labels_by_seed.append(labels)
        probabilities_by_seed.append(probabilities)
    reference_labels = labels_by_seed[0]
    reference_probabilities = probabilities_by_seed[0]
    seed_ari = float(
        np.mean(
            [
                adjusted_rand_score(reference_labels, labels)
                for labels in labels_by_seed[1:]
            ]
        )
    )
    posterior_js = float(
        np.mean(
            [
                _posterior_js(reference_probabilities, probabilities)
                for probabilities in probabilities_by_seed[1:]
            ]
        )
    )
    return reference_labels, reference_probabilities, seed_ari, posterior_js


_ERROR_TRAJECTORY_TOKENS = (
    "error",
    "expost",
    "residual",
    "calibration",
    "brier",
    "log_loss",
    "false_positive",
    "false_negative",
    "ranking",
    "base_meta_disagreement",
    "bad_mae",
    "timeout",
    "stop",
    "adverse",
    "dirty",
    "clean_hit",
)


def _split_view_feature_columns(
    episodes: pd.DataFrame,
    feature_columns: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Split descriptive outcomes from observable state without using target columns.

    Episode construction materializes ex-post trajectory summaries under
    ``family__error*`` in newer ledgers. Older artifacts use explicit outcome
    names such as ``family__residual`` or ``family__timeout``. Everything else
    remains the observable market-state view.
    """

    family_error_columns = [
        name
        for name in feature_columns
        if name.casefold().startswith("family__error")
        or any(token in name.casefold() for token in _ERROR_TRAJECTORY_TOKENS)
    ]
    # Older episode artifacts retain realized trajectory summaries directly
    # under the protected ex-post namespace rather than under family__error.
    # These columns are descriptive only and deliberately never join the
    # observable market-state view.
    expost_columns = [name for name in episodes if name.startswith("expost__")]
    error_columns = [*family_error_columns, *expost_columns]
    error_columns = [
        name
        for name in error_columns
        if pd.to_numeric(episodes[name], errors="coerce").notna().any()
    ]
    error_set = set(error_columns)
    return error_columns, [name for name in feature_columns if name not in error_set]


def _pca_embedding(
    matrix: np.ndarray, requested_dim: int, random_state: int
) -> np.ndarray:
    dimensions = min(int(requested_dim), matrix.shape[1], max(1, len(matrix) - 1))
    return PCA(
        n_components=dimensions, whiten=True, random_state=random_state
    ).fit_transform(matrix)


def _align_probability_columns(
    reference: np.ndarray, candidate: np.ndarray
) -> np.ndarray:
    """Align mixture components before comparing posterior distributions."""

    if reference.shape != candidate.shape or reference.shape[1] < 2:
        return candidate
    similarity = reference.T @ candidate
    if linear_sum_assignment is None:
        order = np.argmax(similarity, axis=1)
    else:
        rows, cols = linear_sum_assignment(-similarity)
        order = np.empty(reference.shape[1], dtype=np.int64)
        order[rows] = cols
    return candidate[:, order]


def _posterior_js(reference: np.ndarray, candidate: np.ndarray) -> float:
    aligned = _align_probability_columns(reference, candidate)
    midpoint = 0.5 * (reference + aligned)
    left = np.sum(
        reference * np.log(np.maximum(reference, 1e-12) / np.maximum(midpoint, 1e-12)),
        axis=1,
    )
    right = np.sum(
        aligned * np.log(np.maximum(aligned, 1e-12) / np.maximum(midpoint, 1e-12)),
        axis=1,
    )
    return float(np.mean(0.5 * (left + right) / np.log(2.0)))


def _fit_seed_stability(
    embedding: np.ndarray,
    *,
    method: str,
    clusters: int,
    config: FailureTaxonomyModelConfig,
) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    labels_by_seed: list[np.ndarray] = []
    probabilities_by_seed: list[np.ndarray] = []
    for seed in config.stability_seeds:
        try:
            labels, probabilities = _fit_labels(
                embedding,
                method=method,
                clusters=clusters,
                seed=int(seed),
                config=config,
            )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            return None
        labels_by_seed.append(labels)
        probabilities_by_seed.append(probabilities)
    reference_labels = labels_by_seed[0]
    reference_probabilities = probabilities_by_seed[0]
    seed_ari = (
        float(
            np.mean(
                [
                    adjusted_rand_score(reference_labels, labels)
                    for labels in labels_by_seed[1:]
                ]
            )
        )
        if len(labels_by_seed) > 1
        else 1.0
    )
    posterior_js = (
        float(
            np.mean(
                [
                    _posterior_js(reference_probabilities, probabilities)
                    for probabilities in probabilities_by_seed[1:]
                ]
            )
        )
        if len(probabilities_by_seed) > 1
        else 0.0
    )
    return reference_labels, reference_probabilities, seed_ari, posterior_js


def _episode_bootstrap_stability(
    embedding: np.ndarray,
    reference_labels: np.ndarray,
    reference_probabilities: np.ndarray,
    *,
    method: str,
    clusters: int,
    config: FailureTaxonomyModelConfig,
) -> tuple[float, float]:
    """Measure whether a taxonomy survives resampling entire episode rows."""

    repeats = max(0, int(config.episode_bootstrap_repeats))
    if repeats == 0 or len(embedding) < clusters:
        return np.nan, np.nan
    size = max(
        clusters,
        int(np.ceil(len(embedding) * float(config.episode_bootstrap_fraction))),
    )
    rng = np.random.default_rng(config.random_state + 991 * clusters)
    aris: list[float] = []
    divergences: list[float] = []
    for repeat in range(repeats):
        sample_index = rng.integers(0, len(embedding), size=size)
        try:
            # Fit on the resampled episodes, then score the original episode set.
            if method == "pca_student_t":
                model: Any = DiagonalStudentTMixture(
                    clusters,
                    degrees_of_freedom=config.student_df,
                    reg_covar=config.reg_covar,
                    max_iter=config.max_iter,
                    tolerance=config.tolerance,
                    random_state=config.random_state + repeat,
                ).fit(embedding[sample_index])
            else:
                model = GaussianMixture(
                    n_components=clusters,
                    covariance_type="diag",
                    reg_covar=config.reg_covar,
                    max_iter=config.max_iter,
                    n_init=1,
                    random_state=config.random_state + repeat,
                ).fit(embedding[sample_index])
            labels = model.predict(embedding)
            probabilities = model.predict_proba(embedding)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            continue
        aris.append(float(adjusted_rand_score(reference_labels, labels)))
        divergences.append(_posterior_js(reference_probabilities, probabilities))
    return (
        float(np.mean(aris)) if aris else np.nan,
        float(np.mean(divergences)) if divergences else np.nan,
    )


def _fit_view_assignments(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    latent_dim: int,
    method: str,
    clusters: int,
    config: FailureTaxonomyModelConfig,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float] | None:
    if len(columns) < int(config.min_view_features):
        return None
    embedding = _pca_embedding(
        _robust_matrix(frame, columns), latent_dim, config.random_state
    )
    fitted = _fit_seed_stability(
        embedding,
        method=method,
        clusters=clusters,
        config=config,
    )
    if fitted is None:
        return None
    labels, probabilities, seed_ari, seed_js = fitted
    bootstrap_ari, bootstrap_js = _episode_bootstrap_stability(
        embedding,
        labels,
        probabilities,
        method=method,
        clusters=clusters,
        config=config,
    )
    return labels, probabilities, seed_ari, seed_js, bootstrap_ari, bootstrap_js


def fit_failure_taxonomy_models(
    episodes: pd.DataFrame,
    *,
    config: FailureTaxonomyModelConfig = FailureTaxonomyModelConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare robust local mixtures and return assignments plus diagnostics."""

    feature_columns = [name for name in episodes if name.startswith("family__")]
    if not feature_columns:
        raise ValueError("Failure episode table has no family__ trajectory columns")
    error_columns, market_columns = _split_view_feature_columns(
        episodes, feature_columns
    )
    assignments: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    group_keys = ["side_name", "archetype_policy_key"]
    for keys, local in episodes.groupby(group_keys, observed=True, sort=True):
        local = local.reset_index().rename(columns={"index": "source_index"})
        if len(local) < 2 * int(config.min_cluster_episodes):
            continue
        raw = _robust_matrix(local, feature_columns)
        maximum_dim = min(raw.shape[1], max(1, len(raw) - 1))
        for latent_dim in config.latent_dims:
            dimensions = min(int(latent_dim), maximum_dim)
            if dimensions < 1:
                continue
            for method in config.methods:
                learned_embeddings = _learned_representation_embeddings(
                    raw,
                    method=method,
                    latent_dim=dimensions,
                    config=config,
                )
                if method in {"small_dae_gmm", "vicreg_gmm"}:
                    if not learned_embeddings:
                        continue
                    embedding = learned_embeddings[int(config.stability_seeds[0])]
                    representation_seed_refit = True
                else:
                    embedding = _pca_embedding(raw, dimensions, config.random_state)
                    representation_seed_refit = False
                for clusters in config.cluster_counts:
                    if clusters < 2 or clusters * config.min_cluster_episodes > len(
                        local
                    ):
                        continue
                    fitted = (
                        _fit_representation_seed_stability(
                            learned_embeddings,
                            method=method,
                            clusters=int(clusters),
                            config=config,
                        )
                        if representation_seed_refit
                        else _fit_seed_stability(
                            embedding,
                            method=method,
                            clusters=int(clusters),
                            config=config,
                        )
                    )
                    if fitted is None:
                        continue
                    labels, probability, stability, seed_posterior_js = fitted
                    support = np.bincount(labels, minlength=int(clusters))
                    if int(support.min()) < int(config.min_cluster_episodes):
                        continue
                    bootstrap_ari, bootstrap_posterior_js = (
                        _episode_bootstrap_stability(
                            embedding,
                            labels,
                            probability,
                            method=_mixture_method(method),
                            clusters=int(clusters),
                            config=config,
                        )
                    )
                    silhouette = float(silhouette_score(embedding, labels))
                    concentration = float(support.max() / support.sum())
                    error_view = None
                    market_view = None
                    if config.multi_view_enabled:
                        error_view = _fit_view_assignments(
                            local,
                            error_columns,
                            latent_dim=dimensions,
                            method=_mixture_method(method),
                            clusters=int(clusters),
                            config=config,
                        )
                        market_view = _fit_view_assignments(
                            local,
                            market_columns,
                            latent_dim=dimensions,
                            method=_mixture_method(method),
                            clusters=int(clusters),
                            config=config,
                        )
                    if error_view is not None and market_view is not None:
                        (
                            error_labels,
                            error_probabilities,
                            error_seed_ari,
                            error_seed_js,
                            error_bootstrap_ari,
                            error_bootstrap_js,
                        ) = error_view
                        (
                            market_labels,
                            market_probabilities,
                            market_seed_ari,
                            market_seed_js,
                            market_bootstrap_ari,
                            market_bootstrap_js,
                        ) = market_view
                        pair_codes, pair_values = pd.factorize(
                            pd.MultiIndex.from_arrays([error_labels, market_labels]),
                            sort=True,
                        )
                        pairing_ami = float(
                            adjusted_mutual_info_score(error_labels, market_labels)
                        )
                        pair_support = np.bincount(pair_codes)
                        pairing_concentration = float(
                            pair_support.max() / len(pair_codes)
                        )
                    else:
                        error_labels = market_labels = None
                        error_probabilities = market_probabilities = None
                        error_seed_ari = error_seed_js = error_bootstrap_ari = (
                            error_bootstrap_js
                        ) = np.nan
                        market_seed_ari = market_seed_js = market_bootstrap_ari = (
                            market_bootstrap_js
                        ) = np.nan
                        pair_codes = None
                        pair_values = []
                        pairing_ami = pairing_concentration = np.nan
                    finite_bootstrap_ari = (
                        float(bootstrap_ari) if np.isfinite(bootstrap_ari) else 0.0
                    )
                    finite_seed_js = (
                        float(seed_posterior_js)
                        if np.isfinite(seed_posterior_js)
                        else 1.0
                    )
                    finite_bootstrap_js = (
                        float(bootstrap_posterior_js)
                        if np.isfinite(bootstrap_posterior_js)
                        else 1.0
                    )
                    view_stability = (
                        float(
                            np.nanmean(
                                [
                                    error_seed_ari,
                                    market_seed_ari,
                                    error_bootstrap_ari,
                                    market_bootstrap_ari,
                                ]
                            )
                        )
                        if error_view is not None and market_view is not None
                        else 0.0
                    )
                    objective = (
                        silhouette
                        + 0.35 * stability
                        + 0.20 * finite_bootstrap_ari
                        + 0.10 * view_stability
                        - 0.15 * finite_seed_js
                        - 0.10 * finite_bootstrap_js
                        - 0.20 * concentration
                    )
                    diagnostics.append(
                        {
                            "side_name": keys[0],
                            "archetype_policy_key": keys[1],
                            "method": method,
                            "latent_dim": dimensions,
                            "clusters": int(clusters),
                            "representation_seed_refit": representation_seed_refit,
                            "effective_embedding_dim": int(embedding.shape[1]),
                            "episodes": int(len(local)),
                            "min_cluster_support": int(support.min()),
                            "max_cluster_fraction": concentration,
                            "silhouette": silhouette,
                            "seed_ari": stability,
                            "seed_posterior_js": seed_posterior_js,
                            "episode_bootstrap_ari": bootstrap_ari,
                            "episode_bootstrap_posterior_js": bootstrap_posterior_js,
                            "error_view_feature_count": int(len(error_columns)),
                            "market_state_view_feature_count": int(len(market_columns)),
                            "error_seed_ari": error_seed_ari,
                            "error_seed_posterior_js": error_seed_js,
                            "error_episode_bootstrap_ari": error_bootstrap_ari,
                            "error_episode_bootstrap_posterior_js": error_bootstrap_js,
                            "market_state_seed_ari": market_seed_ari,
                            "market_state_seed_posterior_js": market_seed_js,
                            "market_state_episode_bootstrap_ari": market_bootstrap_ari,
                            "market_state_episode_bootstrap_posterior_js": market_bootstrap_js,
                            "error_market_adjusted_mi": pairing_ami,
                            "consensus_pair_count": int(len(pair_values)),
                            "consensus_pair_max_fraction": pairing_concentration,
                            "selection_objective": objective,
                        }
                    )
                    candidate = local.loc[
                        :,
                        [
                            "source_index",
                            *group_keys,
                            "event_block",
                            "event_start",
                            "event_end",
                        ],
                    ].copy()
                    candidate["method"] = method
                    candidate["latent_dim"] = dimensions
                    candidate["clusters"] = int(clusters)
                    candidate["cluster_id"] = labels
                    candidate["cluster_posterior_max"] = probability.max(axis=1)
                    entropy = -np.sum(
                        probability * np.log(np.maximum(probability, 1e-12)), axis=1
                    ) / np.log(max(int(clusters), 2))
                    candidate["cluster_entropy"] = entropy
                    if error_view is not None and market_view is not None:
                        candidate["error_cluster_id"] = error_labels.astype(np.int16)
                        candidate["error_cluster_posterior_max"] = (
                            error_probabilities.max(axis=1)
                        )
                        candidate["error_cluster_entropy"] = -np.sum(
                            error_probabilities
                            * np.log(np.maximum(error_probabilities, 1e-12)),
                            axis=1,
                        ) / np.log(max(int(clusters), 2))
                        candidate["market_state_cluster_id"] = market_labels.astype(
                            np.int16
                        )
                        candidate["market_state_cluster_posterior_max"] = (
                            market_probabilities.max(axis=1)
                        )
                        candidate["market_state_cluster_entropy"] = -np.sum(
                            market_probabilities
                            * np.log(np.maximum(market_probabilities, 1e-12)),
                            axis=1,
                        ) / np.log(max(int(clusters), 2))
                        candidate["consensus_pair_id"] = pair_codes.astype(np.int16)
                        candidate["consensus_pair_posterior"] = error_probabilities.max(
                            axis=1
                        ) * market_probabilities.max(axis=1)
                    assignments.append(candidate)
    diagnostic_frame = pd.DataFrame(diagnostics)
    if diagnostic_frame.empty:
        return pd.DataFrame(), diagnostic_frame
    ranked_diagnostics = diagnostic_frame.sort_values(
        [*group_keys, "selection_objective"],
        ascending=[True, True, False],
        kind="stable",
    ).copy()
    ranked_diagnostics["winner_rank"] = (
        ranked_diagnostics.groupby(group_keys, observed=True).cumcount() + 1
    )
    ranked_diagnostics["is_winner"] = ranked_diagnostics["winner_rank"].eq(1)
    winners = ranked_diagnostics.groupby(
        group_keys, observed=True, as_index=False
    ).head(1)
    all_assignments = pd.concat(assignments, ignore_index=True)
    winner_keys = winners.loc[:, [*group_keys, "method", "latent_dim", "clusters"]]
    selected = all_assignments.merge(
        winner_keys,
        on=[*group_keys, "method", "latent_dim", "clusters"],
        how="inner",
        validate="many_to_one",
    )
    selected = selected.merge(
        winners.loc[
            :, [*group_keys, "method", "latent_dim", "clusters", "selection_objective"]
        ].rename(columns={"selection_objective": "winner_selection_objective"}),
        on=[*group_keys, "method", "latent_dim", "clusters"],
        how="left",
        validate="many_to_one",
    )
    return selected, ranked_diagnostics


def failure_taxonomy_nonredundancy(
    episodes: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    """Audit whether selected modes merely reproduce month or loss magnitude.

    Side and archetype are intentionally fixed by the local clustering design,
    so they are not treated as redundancy defects here. Symbol concentration is
    reported upstream where row-level episode membership is available.
    """
    if episodes.empty or assignments.empty:
        return pd.DataFrame()
    source = episodes.reset_index().rename(columns={"index": "source_index"})
    columns = [
        "source_index",
        "side_name",
        "archetype_policy_key",
        "event_start",
        "event_end",
        "calendar_mean_ev",
    ]
    joined = assignments.merge(
        source.loc[:, [name for name in columns if name in source.columns]],
        on=[
            "source_index",
            "side_name",
            "archetype_policy_key",
            "event_start",
            "event_end",
        ],
        how="left",
        validate="one_to_one",
    )
    joined["event_month"] = pd.to_datetime(
        joined["event_start"], utc=True, errors="coerce"
    ).dt.strftime("%Y-%m")
    joined["event_duration_days"] = (
        pd.to_datetime(joined["event_end"], utc=True, errors="coerce")
        - pd.to_datetime(joined["event_start"], utc=True, errors="coerce")
    ).dt.days.add(1)
    joined["severity_bucket"] = "unavailable"
    for _, positions in joined.groupby(
        ["side_name", "archetype_policy_key"], observed=True, sort=False
    ).indices.items():
        index = np.asarray(positions, dtype=np.int64)
        severity = pd.to_numeric(
            joined.iloc[index]["calendar_mean_ev"], errors="coerce"
        )
        if severity.notna().sum() < 2:
            continue
        ranks = severity.rank(method="average", pct=True)
        bucket = np.minimum((ranks.fillna(0.5) * 4).astype(int), 3)
        joined.iloc[index, joined.columns.get_loc("severity_bucket")] = (
            "q" + bucket.astype(str)
        ).to_numpy()

    rows: list[dict[str, Any]] = []
    keys = [
        "side_name",
        "archetype_policy_key",
        "method",
        "latent_dim",
        "clusters",
        "cluster_id",
    ]
    for values, group in joined.groupby(keys, observed=True, sort=True):
        month_share = group["event_month"].value_counts(normalize=True, dropna=False)
        severity_share = group["severity_bucket"].value_counts(
            normalize=True, dropna=False
        )
        max_month = float(month_share.max()) if len(month_share) else np.nan
        max_severity = float(severity_share.max()) if len(severity_share) else np.nan
        rows.append(
            {
                **dict(zip(keys, values, strict=True)),
                "episodes": int(len(group)),
                "active_months": int(group["event_month"].nunique(dropna=True)),
                "month_max_fraction": max_month,
                "dominant_month": str(month_share.index[0]) if len(month_share) else "",
                "severity_bucket_max_fraction": max_severity,
                "dominant_severity_bucket": (
                    str(severity_share.index[0]) if len(severity_share) else ""
                ),
                "mean_duration_days": float(group["event_duration_days"].mean()),
                "calendar_redundancy_warning": bool(
                    (np.isfinite(max_month) and max_month >= 0.80)
                    or (np.isfinite(max_severity) and max_severity >= 0.80)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["calendar_redundancy_warning", "episodes"],
        ascending=[True, False],
        kind="stable",
    )


def failure_taxonomy_temporal_stability(
    episodes: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    """Measure month-to-month support and qualitative profile persistence."""

    if episodes.empty or assignments.empty:
        return pd.DataFrame()
    source = episodes.reset_index().rename(columns={"index": "source_index"})
    identity = [
        "source_index",
        "side_name",
        "archetype_policy_key",
        "event_start",
        "event_end",
    ]
    profile_columns = [
        name
        for name in source.columns
        if name == "calendar_mean_ev"
        or name.startswith("expost__")
        or name.startswith("family__error_")
    ][:32]
    joined = assignments.merge(
        source.loc[:, [*identity, *profile_columns]],
        on=identity,
        how="left",
        validate="one_to_one",
    )
    joined["event_month"] = pd.to_datetime(
        joined["event_start"], utc=True, errors="coerce"
    ).dt.strftime("%Y-%m")
    group_keys = ["side_name", "archetype_policy_key"]
    mode_keys = [*group_keys, "method", "latent_dim", "clusters", "cluster_id"]
    rows: list[dict[str, Any]] = []
    for (side, archetype), local in joined.groupby(
        group_keys, observed=True, sort=True
    ):
        valid_months = local["event_month"].dropna()
        if valid_months.empty:
            continue
        month_index = pd.period_range(
            pd.Period(valid_months.min(), freq="M"),
            pd.Period(valid_months.max(), freq="M"),
            freq="M",
        ).astype(str)
        monthly_total = local.groupby("event_month", observed=True).size().reindex(
            month_index, fill_value=0
        )
        numeric_profiles = (
            local.loc[:, profile_columns].apply(pd.to_numeric, errors="coerce")
            if profile_columns
            else pd.DataFrame(index=local.index)
        )
        profile_scale = (
            numeric_profiles.quantile(0.75) - numeric_profiles.quantile(0.25)
        ).clip(lower=1e-4)
        for values, mode in local.groupby(mode_keys[2:], observed=True, sort=True):
            monthly_count = mode.groupby("event_month", observed=True).size().reindex(
                month_index, fill_value=0
            )
            support = monthly_count.div(monthly_total.replace(0, np.nan)).fillna(0.0)
            active = monthly_count.gt(0)
            support_mean = float(support.mean())
            support_std = float(support.std(ddof=0))
            profile_drift = np.nan
            if profile_columns and int(active.sum()) >= 2:
                mode_numeric = numeric_profiles.loc[mode.index]
                global_centroid = mode_numeric.median()
                monthly_centroids = mode_numeric.groupby(
                    mode["event_month"], observed=True
                ).median()
                normalized = monthly_centroids.sub(global_centroid).div(profile_scale)
                finite = (
                    normalized.replace([np.inf, -np.inf], np.nan)
                    .abs()
                    .clip(upper=8.0)
                )
                if finite.notna().any().any():
                    profile_drift = float(finite.mean(axis=1).mean())
            rows.append(
                {
                    "side_name": str(side),
                    "archetype_policy_key": str(archetype),
                    **dict(
                        zip(
                            ["method", "latent_dim", "clusters", "cluster_id"],
                            values,
                            strict=True,
                        )
                    ),
                    "episodes": int(len(mode)),
                    "calendar_span_months": int(len(month_index)),
                    "active_months": int(active.sum()),
                    "month_presence_rate": float(active.mean()),
                    "monthly_support_mean": support_mean,
                    "monthly_support_std": support_std,
                    "monthly_support_cv": support_std / max(support_mean, 1e-6),
                    "monthly_profile_l1_drift": profile_drift,
                    "temporal_stability_warning": bool(
                        int(active.sum()) < 2
                        or float(active.mean()) < 0.10
                        or support_std > max(2.0 * support_mean, 0.25)
                        or (np.isfinite(profile_drift) and profile_drift > 2.0)
                    ),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["temporal_stability_warning", "episodes"],
        ascending=[True, False],
        kind="stable",
    )


def fit_frozen_consensus_taxonomy(
    episodes: pd.DataFrame,
    *,
    reference_end: pd.Timestamp,
    config: FailureTaxonomyModelConfig = FailureTaxonomyModelConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit failure modes before ``reference_end`` and freeze their prototypes.

    The descriptive model search may use error and market trajectories from the
    reference episodes. Later episodes receive a ground-truth mode assignment
    from fixed reference centroids; no later episode can alter the taxonomy.
    These assignments remain ex-post labels, never inference features.
    """

    cutoff = pd.Timestamp(reference_end)
    if cutoff.tzinfo is None:
        raise ValueError("reference_end must be timezone-aware")
    cutoff = cutoff.tz_convert("UTC")
    family_columns = [name for name in episodes if name.startswith("family__")]
    if not family_columns:
        raise ValueError("Failure episode table has no family__ trajectory columns")
    assignments: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    state_groups: dict[str, Any] = {}
    keys = ["side_name", "archetype_policy_key"]
    for values, local in episodes.groupby(keys, observed=True, sort=True):
        local = local.copy()
        event_end = pd.to_datetime(local["event_end"], utc=True, errors="coerce")
        mode_available = event_end + pd.Timedelta(
            days=max(0, int(config.recovery_horizon_days))
            + max(0, int(config.outcome_resolution_days))
        )
        reference = local.loc[mode_available.lt(cutoff)].copy()
        if len(reference) < 2 * int(config.min_cluster_episodes):
            continue
        reference_assignments, reference_diagnostics = fit_failure_taxonomy_models(
            reference,
            config=config,
        )
        if reference_assignments.empty:
            continue
        target_column = (
            "consensus_pair_id"
            if "consensus_pair_id" in reference_assignments
            else "cluster_id"
        )
        reference_indices = reference_assignments["source_index"].astype(int)
        labels = reference_assignments[target_column].astype(int).to_numpy()
        support = pd.Series(labels).value_counts().sort_index()
        retained = support.loc[support.ge(int(config.min_cluster_episodes))].index
        keep = np.isin(labels, retained.to_numpy(dtype=int))
        if int(keep.sum()) < 2 * int(config.min_cluster_episodes):
            continue
        reference_indices = reference_indices.loc[keep]
        labels = labels[keep]
        reference_matrix_raw = (
            episodes.loc[reference_indices, family_columns]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(np.float64)
        )
        present = np.isfinite(reference_matrix_raw).any(axis=0)
        median = np.zeros(reference_matrix_raw.shape[1], dtype=np.float64)
        scale = np.ones(reference_matrix_raw.shape[1], dtype=np.float64)
        if present.any():
            reference_present = reference_matrix_raw[:, present]
            median[present] = np.nanmedian(reference_present, axis=0)
            scale[present] = np.maximum(
                np.nanquantile(reference_present, 0.75, axis=0)
                - np.nanquantile(reference_present, 0.25, axis=0),
                1e-4,
            )

        def normalize(frame: pd.DataFrame) -> np.ndarray:
            raw = (
                frame.loc[:, family_columns]
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy(np.float64)
            )
            return np.clip(
                np.nan_to_num((raw - median) / scale, nan=0.0), -8.0, 8.0
            )

        reference_matrix = normalize(episodes.loc[reference_indices])
        label_values = sorted(set(map(int, labels)))
        centroids = np.vstack(
            [reference_matrix[labels == label].mean(axis=0) for label in label_values]
        )
        all_matrix = normalize(local)
        distances = np.sqrt(
            np.maximum(
                ((all_matrix[:, None, :] - centroids[None, :, :]) ** 2).mean(axis=2),
                0.0,
            )
        )
        nearest = np.argmin(distances, axis=1)
        assigned_labels = np.asarray(label_values, dtype=np.int16)[nearest]
        logits = -distances
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
        result = local.loc[
            :, [*keys, "event_block", "event_start", "event_end"]
        ].copy()
        result.insert(0, "source_index", local.index.to_numpy(dtype=np.int64))
        result["method"] = "frozen_consensus_prototype"
        result["latent_dim"] = 0
        result["clusters"] = len(label_values)
        result["cluster_id"] = assigned_labels
        result["cluster_posterior_max"] = probabilities.max(axis=1)
        result["cluster_entropy"] = -np.sum(
            probabilities * np.log(np.maximum(probabilities, 1e-12)), axis=1
        ) / np.log(max(len(label_values), 2))
        result["taxonomy_reference_end"] = cutoff
        result["assignment_is_reference"] = mode_available.lt(cutoff).to_numpy()
        assignments.append(result)
        reference_predicted = assigned_labels[
            local.index.get_indexer(reference_indices.to_numpy())
        ]
        fidelity = float(np.mean(reference_predicted == labels))
        diagnostics.append(
            {
                "side_name": str(values[0]),
                "archetype_policy_key": str(values[1]),
                "reference_end": cutoff,
                "reference_label_availability_horizon_days": int(
                    config.recovery_horizon_days + config.outcome_resolution_days
                ),
                "reference_episodes": int(len(reference)),
                "retained_reference_episodes": int(len(labels)),
                "frozen_modes": int(len(label_values)),
                "reference_assignment_fidelity": fidelity,
                "descriptive_winner_method": str(
                    reference_assignments["method"].iloc[0]
                ),
                "descriptive_winner_latent_dim": int(
                    reference_assignments["latent_dim"].iloc[0]
                ),
                "descriptive_winner_clusters": int(
                    reference_assignments["clusters"].iloc[0]
                ),
            }
        )
        state_key = f"{values[0]}::{values[1]}"
        state_groups[state_key] = {
            "side_name": str(values[0]),
            "archetype_policy_key": str(values[1]),
            "feature_columns": family_columns,
            "median": median.astype(np.float32).tolist(),
            "scale": scale.astype(np.float32).tolist(),
            "labels": label_values,
            "centroids": centroids.astype(np.float32).tolist(),
            "reference_assignment_fidelity": fidelity,
        }
        if not reference_diagnostics.empty:
            winner = reference_diagnostics.loc[reference_diagnostics["is_winner"]]
            if len(winner):
                state_groups[state_key]["winner_diagnostics"] = winner.iloc[0].to_dict()
    state = {
        "schema": "frozen_failure_consensus_taxonomy_v1",
        "reference_end": cutoff.isoformat(),
        "reference_recovery_horizon_days": int(config.recovery_horizon_days),
        "reference_outcome_resolution_days": int(config.outcome_resolution_days),
        "reference_label_availability_horizon_days": int(
            config.recovery_horizon_days + config.outcome_resolution_days
        ),
        "feature_columns": family_columns,
        "groups": state_groups,
        "leakage_contract": (
            "Taxonomy search, scaling and prototypes use only episodes whose full "
            "recovery trajectory and outcome-resolution delay end before reference_end. "
            "Later episode modes are ex-post labels assigned by frozen reference "
            "prototypes and are never inference inputs."
        ),
    }
    return (
        pd.concat(assignments, ignore_index=True) if assignments else pd.DataFrame(),
        pd.DataFrame(diagnostics),
        state,
    )
