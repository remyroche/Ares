"""Services for NAS/TAS label fusion and regime optimization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

import numpy as np
from sklearn.mixture import GaussianMixture

from src.utils.tprint import tprint


if TYPE_CHECKING:  # pragma: no cover - type checking only
    from hmmlearn import hmm


def _default_logger(message: str, level: str = "INFO") -> None:
    tprint(message, level)


@dataclass
class LabelFusionResult:
    assignments: np.ndarray
    metadata: Dict[str, Any]


class LabelFusionService:
    """Service responsible for aligning NAS/TAS labels and running Dawid–Skene."""

    def __init__(self, logger: Callable[[str, str], None] = _default_logger):
        self._logger = logger

    def map_labels_to_k_space(
        self,
        tas_assignments: np.ndarray,
        nas_assignments: np.ndarray,
        target_k: int,
        features: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Map NAS/TAS labels to the shared ``0..K-1`` space when possible."""

        tas_unique = set(tas_assignments.tolist())
        nas_unique = set(nas_assignments.tolist())

        if (
            tas_unique
            and nas_unique
            and max(tas_unique) < target_k
            and max(nas_unique) < target_k
            and min(tas_unique) >= 0
            and min(nas_unique) >= 0
        ):
            self._logger("Labels already aligned with target space", "INFO")
            return tas_assignments, nas_assignments, {"mapping_needed": False}

        self._logger(
            f"Mapping labels to shared K={target_k} space (TAS={len(tas_unique)}, NAS={len(nas_unique)})",
            "INFO",
        )

        if features is not None:
            tas_mapped, nas_mapped, tas_mapping, nas_mapping = self._map_using_gmm(
                tas_assignments, nas_assignments, target_k, features
            )
            method = "gmm_centroid"
            mapping_details: Dict[str, Any] = {}
        else:
            tas_mapped, nas_mapped, tas_mapping, nas_mapping, abstain_value = self._create_abstain_mapping(
                tas_assignments, nas_assignments, target_k
            )
            method = "abstain_column"
            mapping_details = {"abstain_value": abstain_value}

        mapping_info = {
            "mapping_needed": True,
            "tas_mapping": tas_mapping,
            "nas_mapping": nas_mapping,
            "method": method,
            **mapping_details,
        }

        self._logger(
            f"Label mapping completed – TAS unique: {len(set(tas_mapped))}, NAS unique: {len(set(nas_mapped))}",
            "SUCCESS",
        )
        return tas_mapped, nas_mapped, mapping_info

    def run_dawid_skene(
        self,
        tas_assignments: np.ndarray,
        nas_assignments: np.ndarray,
        target_k: int,
        features: Optional[np.ndarray] = None,
        max_iterations: int = 50,
        tolerance: float = 1e-6,
    ) -> LabelFusionResult:
        """Run Dawid–Skene EM to fuse NAS and TAS labels."""

        self._logger(f"Starting Dawid–Skene fusion with K={target_k}", "INFO")

        (
            tas_mapped,
            nas_mapped,
            mapping_info,
        ) = self.map_labels_to_k_space(tas_assignments, nas_assignments, target_k, features)

        n_classes = target_k
        n_samples = len(tas_mapped)

        tas_confusion, nas_confusion = self._initialize_confusion_matrices(n_classes)
        class_priors = np.ones(n_classes) / n_classes

        posteriors = np.zeros((n_samples, n_classes))
        log_likelihoods = []

        for iteration in range(max_iterations):
            self._e_step(
                tas_mapped,
                nas_mapped,
                tas_confusion,
                nas_confusion,
                class_priors,
                posteriors,
                mapping_info,
            )

            log_likelihoods.append(
                float(np.sum(np.log(np.clip(posteriors.sum(axis=1), 1e-10, None))))
            )

            old_tas_confusion = tas_confusion.copy()
            old_nas_confusion = nas_confusion.copy()
            old_priors = class_priors.copy()

            class_priors = self._update_class_priors(posteriors)
            tas_confusion = self._update_confusion_matrix(posteriors, tas_mapped, n_classes)
            nas_confusion = self._update_confusion_matrix(posteriors, nas_mapped, n_classes)

            if self._has_converged(
                old_tas_confusion,
                old_nas_confusion,
                old_priors,
                tas_confusion,
                nas_confusion,
                class_priors,
                tolerance,
            ):
                self._logger(f"Dawid–Skene converged after {iteration + 1} iterations", "SUCCESS")
                break

        fused_assignments = np.argmax(posteriors, axis=1)

        metadata: Dict[str, Any] = {
            "iterations": len(log_likelihoods),
            "converged": len(log_likelihoods) < max_iterations,
            "log_likelihoods": log_likelihoods,
            "tas_confusion_matrix": tas_confusion.tolist(),
            "nas_confusion_matrix": nas_confusion.tolist(),
            "class_priors": class_priors.tolist(),
            "mapping_info": mapping_info,
            "posteriors": posteriors.tolist(),
            "tas_row_sums": tas_confusion.sum(axis=1).tolist(),
            "nas_row_sums": nas_confusion.sum(axis=1).tolist(),
        }

        self._logger(
            f"Dawid–Skene fusion completed: {n_samples} samples, {n_classes} classes",
            "SUCCESS",
        )

        return LabelFusionResult(assignments=fused_assignments, metadata=metadata)

    def _map_using_gmm(
        self,
        tas_assignments: np.ndarray,
        nas_assignments: np.ndarray,
        target_k: int,
        features: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[int, int], Dict[int, int]]:
        gmm = GaussianMixture(n_components=target_k, random_state=42)
        gmm.fit(features)
        centroids = gmm.means_

        tas_mapping: Dict[int, int] = {}
        nas_mapping: Dict[int, int] = {}

        for label in set(tas_assignments.tolist()):
            mapped_label = self._nearest_centroid_label(label, tas_assignments, features, centroids, target_k)
            tas_mapping[label] = mapped_label

        for label in set(nas_assignments.tolist()):
            mapped_label = self._nearest_centroid_label(label, nas_assignments, features, centroids, target_k)
            nas_mapping[label] = mapped_label

        tas_mapped = np.array([tas_mapping.get(label, label % target_k) for label in tas_assignments])
        nas_mapped = np.array([nas_mapping.get(label, label % target_k) for label in nas_assignments])

        return tas_mapped, nas_mapped, tas_mapping, nas_mapping

    def _nearest_centroid_label(
        self,
        label: int,
        assignments: np.ndarray,
        features: np.ndarray,
        centroids: np.ndarray,
        target_k: int,
    ) -> int:
        mask = assignments == label
        if not np.any(mask):
            return int(label % target_k)

        label_features = features[mask]
        distances = np.linalg.norm(label_features[:, np.newaxis] - centroids, axis=2)
        return int(np.argmin(distances.mean(axis=0)))

    def _create_abstain_mapping(
        self,
        tas_assignments: np.ndarray,
        nas_assignments: np.ndarray,
        target_k: int,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[int, int], Dict[int, int], int]:
        tas_mapping: Dict[int, int] = {}
        nas_mapping: Dict[int, int] = {}
        abstain_value = target_k

        for label in set(tas_assignments.tolist()):
            tas_mapping[label] = label if 0 <= label < target_k else abstain_value

        for label in set(nas_assignments.tolist()):
            nas_mapping[label] = label if 0 <= label < target_k else abstain_value

        tas_mapped = np.array([tas_mapping.get(label, abstain_value) for label in tas_assignments])
        nas_mapped = np.array([nas_mapping.get(label, abstain_value) for label in nas_assignments])

        self._logger(
            f"Abstain mapping applied – TAS unique: {len(set(tas_mapped))}, NAS unique: {len(set(nas_mapped))}",
            "INFO",
        )

        return tas_mapped, nas_mapped, tas_mapping, nas_mapping, abstain_value

    def _initialize_confusion_matrices(self, n_classes: int) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(42)
        alpha = 0.5
        tas_confusion = rng.dirichlet([alpha] * n_classes, size=n_classes)
        nas_confusion = rng.dirichlet([alpha] * n_classes, size=n_classes)
        return tas_confusion, nas_confusion

    def _e_step(
        self,
        tas_mapped: np.ndarray,
        nas_mapped: np.ndarray,
        tas_confusion: np.ndarray,
        nas_confusion: np.ndarray,
        class_priors: np.ndarray,
        posteriors: np.ndarray,
        mapping_info: Dict[str, Any],
    ) -> None:
        n_samples, n_classes = posteriors.shape
        abstain_value = mapping_info.get("abstain_value")

        for i in range(n_samples):
            tas_observation = tas_mapped[i]
            nas_observation = nas_mapped[i]

            for true_class in range(n_classes):
                tas_factor = (
                    tas_confusion[true_class, tas_observation]
                    if tas_observation < n_classes
                    else 1.0
                )
                nas_factor = (
                    nas_confusion[true_class, nas_observation]
                    if nas_observation < n_classes
                    else 1.0
                )

                if abstain_value is not None and (
                    tas_observation == abstain_value or nas_observation == abstain_value
                ):
                    tas_factor = 1.0 if tas_observation == abstain_value else tas_factor
                    nas_factor = 1.0 if nas_observation == abstain_value else nas_factor

                posteriors[i, true_class] = class_priors[true_class] * tas_factor * nas_factor

        row_sums = posteriors.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        posteriors /= row_sums

    def _update_class_priors(self, posteriors: np.ndarray) -> np.ndarray:
        priors = posteriors.mean(axis=0)
        priors_sum = priors.sum()
        if priors_sum == 0:
            return np.ones_like(priors) / len(priors)
        return priors / priors_sum

    def _update_confusion_matrix(
        self, posteriors: np.ndarray, mapped_assignments: np.ndarray, n_classes: int
    ) -> np.ndarray:
        confusion = np.zeros((n_classes, n_classes))
        for true_class in range(n_classes):
            for observed_class in range(n_classes):
                mask = mapped_assignments == observed_class
                if np.any(mask):
                    confusion[true_class, observed_class] = posteriors[mask, true_class].sum()

            row_sum = confusion[true_class].sum()
            if row_sum == 0:
                confusion[true_class] = np.ones(n_classes) / n_classes
            else:
                confusion[true_class] /= row_sum

        return confusion

    def _has_converged(
        self,
        old_tas: np.ndarray,
        old_nas: np.ndarray,
        old_priors: np.ndarray,
        new_tas: np.ndarray,
        new_nas: np.ndarray,
        new_priors: np.ndarray,
        tolerance: float,
    ) -> bool:
        tas_change = np.abs(new_tas - old_tas).max()
        nas_change = np.abs(new_nas - old_nas).max()
        prior_change = np.abs(new_priors - old_priors).max()
        return max(tas_change, nas_change, prior_change) < tolerance


class RegimeOptimizationService:
    """Service responsible for regime optimization, scoring and smoothing."""

    def __init__(
        self,
        label_fusion_service: LabelFusionService,
        score_calculator: Callable[[np.ndarray, np.ndarray], float],
        logger: Callable[[str, str], None] = _default_logger,
    ) -> None:
        self._label_fusion_service = label_fusion_service
        self._score_calculator = score_calculator
        self._logger = logger

    async def progressive_regime_optimization_with_k(
        self,
        features: np.ndarray,
        tas_assignments: np.ndarray,
        nas_assignments: np.ndarray,
        market_data: Optional[np.ndarray],
        optimal_k: int,
    ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
        """Run Dawid–Skene fusion then score the resulting assignments."""

        self._logger("Starting progressive regime optimization", "INFO")
        fusion_result = self._label_fusion_service.run_dawid_skene(
            tas_assignments,
            nas_assignments,
            optimal_k,
            features,
        )

        _ = market_data  # retained for signature compatibility

        mapped_assignments = self._map_to_optimal_k(fusion_result.assignments, features, optimal_k)

        initial_score = self._score_calculator(features, mapped_assignments)

        optimization_metrics = {
            "initial_score": initial_score,
            "final_score": initial_score,
            "improvement": 0.0,
            "iterations": 1,
            "optimal_k": optimal_k,
            "method": "data_driven_optimization",
            "fusion_metadata": fusion_result.metadata,
        }

        self._logger(
            f"Progressive optimization completed – Score: {initial_score:.3f}",
            "SUCCESS",
        )

        return mapped_assignments, optimization_metrics, fusion_result.metadata

    def apply_hmm_smoothing(
        self, features: np.ndarray, assignments: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply HMM-based smoothing with simple fallback."""

        try:
            model = self._initialize_hmm(features, assignments)
            model.fit(features)
            smoothed_assignments = model.predict(features)
            metadata = self._build_smoothing_metadata(model, assignments, smoothed_assignments)
            self._logger(
                f"HMM smoothing completed – {model.n_components} clusters", "SUCCESS"
            )
            return smoothed_assignments, metadata
        except Exception as exc:  # pragma: no cover - safety fallback
            self._logger(
                f"HMM smoothing failed ({exc}), using simple smoothing fallback",
                "WARNING",
            )
            smoothed = self._simple_temporal_smoothing(assignments)
            return smoothed, {"method": "simple_fallback", "error": str(exc)}

    def _map_to_optimal_k(
        self, assignments: np.ndarray, features: np.ndarray, optimal_k: int
    ) -> np.ndarray:
        if len(set(assignments.tolist())) == optimal_k:
            self._logger(
                f"Assignments already match optimal K={optimal_k}", "SUCCESS"
            )
            return assignments

        gmm = GaussianMixture(n_components=optimal_k, random_state=42)
        gmm.fit(features)
        mapped = gmm.predict(features)
        self._logger(
            f"Assignments remapped via GMM – clusters: {len(set(mapped.tolist()))}",
            "SUCCESS",
        )
        return mapped

    def _initialize_hmm(self, features: np.ndarray, assignments: np.ndarray) -> "hmm.GaussianHMM":
        from hmmlearn import hmm

        n_clusters = len(set(assignments.tolist()))
        if n_clusters <= 0:
            raise ValueError("At least one cluster is required for HMM smoothing")

        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm.fit(features)

        model = hmm.GaussianHMM(
            n_components=n_clusters,
            random_state=42,
            n_iter=50,
            init_params="stmc",
        )

        model.means_ = gmm.means_
        model.covars_ = gmm.covariances_
        model.startprob_ = np.ones(n_clusters) / n_clusters
        model.transmat_ = self._learn_transition_matrix(assignments, n_clusters)
        return model

    def _learn_transition_matrix(
        self, assignments: np.ndarray, n_clusters: int
    ) -> np.ndarray:
        transition_matrix = np.zeros((n_clusters, n_clusters))
        for current, nxt in zip(assignments[:-1], assignments[1:]):
            transition_matrix[current, nxt] += 1

        transition_matrix += 0.1 * np.eye(n_clusters)
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        return transition_matrix / row_sums

    def _build_smoothing_metadata(
        self,
        model: hmm.GaussianHMM,
        original_assignments: np.ndarray,
        smoothed_assignments: np.ndarray,
    ) -> Dict[str, Any]:
        expected_durations = []
        low_persistence_regimes = []
        for idx in range(model.n_components):
            p_kk = model.transmat_[idx, idx]
            if p_kk >= 0.99:
                expected_durations.append(float("inf"))
            else:
                expected_duration = 1.0 / max(1e-6, (1 - p_kk))
                expected_durations.append(expected_duration)
                if p_kk < 0.6:
                    low_persistence_regimes.append(idx)

        metadata = {
            "method": "hmm",
            "expected_durations": expected_durations,
            "low_persistence_regimes": low_persistence_regimes,
            "transmat": model.transmat_.tolist(),
            "changed_points": np.nonzero(original_assignments != smoothed_assignments)[0].tolist(),
        }

        if low_persistence_regimes:
            self._logger(
                f"Low persistence regimes detected: {low_persistence_regimes}",
                "WARNING",
            )

        return metadata

    def _simple_temporal_smoothing(self, assignments: np.ndarray) -> np.ndarray:
        smoothed = assignments.copy()
        for idx in range(1, len(assignments) - 1):
            if assignments[idx] != assignments[idx - 1] and assignments[idx] != assignments[idx + 1]:
                window = assignments[max(0, idx - 2) : min(len(assignments), idx + 3)]
                values, counts = np.unique(window, return_counts=True)
                smoothed[idx] = int(values[np.argmax(counts)])
        return smoothed

