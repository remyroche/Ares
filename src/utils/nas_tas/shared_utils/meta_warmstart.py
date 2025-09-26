"""Meta-feature driven warm-start utilities for NAS/TAS searches.

"""Meta-feature driven warm-start utilities for NAS/TAS searches.

This module provides a light-weight meta-learning helper that embeds market
context descriptors (volatility, liquidity, macro regime, etc.) and retrieves
historically successful architecture blueprints using a k-nearest neighbour
approach.  The output can be used to seed search populations for both neural
(NAS) and tree (TAS) architecture searches so that early iterations explore
high-probability regions of the search space without collapsing to a single
objective.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MetaWarmStartConfig:
    """Configuration for the :class:`MetaWarmStarter` utility."""

    k_neighbors: int = 5
    descriptor_keys: Sequence[str] = ("volatility", "liquidity", "macro_regime")
    descriptor_normalization: str = "zscore"
    min_similarity: float = 0.1
    exploration_weight: float = 0.2
    historical_blueprints: Sequence[Dict[str, Any]] = field(default_factory=list)
    descriptor_provider: Optional[Callable[[], Dict[str, float]]] = None
    blueprint_provider: Optional[Callable[[], Sequence[Dict[str, Any]]]] = None

    def copy_with_overrides(self, overrides: Optional[Dict[str, Any]] = None) -> "MetaWarmStartConfig":
        """Return a new config instance applying optional user overrides."""

        if not overrides:
            return MetaWarmStartConfig(
                k_neighbors=self.k_neighbors,
                descriptor_keys=tuple(self.descriptor_keys),
                descriptor_normalization=self.descriptor_normalization,
                min_similarity=self.min_similarity,
                exploration_weight=self.exploration_weight,
                historical_blueprints=list(self.historical_blueprints),
                descriptor_provider=self.descriptor_provider,
                blueprint_provider=self.blueprint_provider,
            )

        params = {
            "k_neighbors": overrides.get("k_neighbors", self.k_neighbors),
            "descriptor_keys": tuple(overrides.get("descriptor_keys", self.descriptor_keys)),
            "descriptor_normalization": overrides.get(
                "descriptor_normalization", self.descriptor_normalization
            ),
            "min_similarity": overrides.get("min_similarity", self.min_similarity),
            "exploration_weight": overrides.get("exploration_weight", self.exploration_weight),
            "historical_blueprints": overrides.get(
                "historical_blueprints", self.historical_blueprints
            ),
            "descriptor_provider": overrides.get("descriptor_provider", self.descriptor_provider),
            "blueprint_provider": overrides.get("blueprint_provider", self.blueprint_provider),
        }
        return MetaWarmStartConfig(**params)


class MetaWarmStarter:
    """Meta-learning assisted warm-start helper for NAS/TAS searches."""

    def __init__(self, config: Optional[MetaWarmStartConfig] = None) -> None:
        self.config = config or MetaWarmStartConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def warm_start(
        self,
        search_space: Dict[str, Any],
        population_size: int,
        fallback_sampler: Callable[[Dict[str, Any]], Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Generate an initial population guided by market meta-features.

        Parameters
        ----------
        search_space:
            Parameter search space to sample from when the blueprint does not
            contain a value for a given parameter.
        population_size:
            Number of architectures requested.
        fallback_sampler:
            Callable that produces a random sample from ``search_space`` when no
            meta-learning information is available.
        """

        descriptors = self._resolve_descriptors()
        blueprints = list(self._resolve_blueprints())

        if not blueprints:
            self.logger.debug("No historical blueprints available – falling back to random sampling")
            return [fallback_sampler(search_space) for _ in range(population_size)]

        descriptor_matrix = self._build_descriptor_matrix(blueprints)
        query_vector = self._embed_descriptors(descriptors)
        similarities = self._cosine_similarity(descriptor_matrix, query_vector)

        ranked_indices = np.argsort(similarities)[::-1]
        selected: List[Dict[str, Any]] = []

        for idx in ranked_indices:
            if len(selected) >= population_size:
                break
            similarity = similarities[idx]
            if similarity < self.config.min_similarity:
                continue
            candidate = self._project_blueprint(blueprints[idx], search_space)
            candidate["meta_similarity"] = float(similarity)
            selected.append(candidate)

        # Exploration bonus – fill remaining slots with random samples.
        remaining = population_size - len(selected)
        if remaining > 0:
            exploratory = [fallback_sampler(search_space) for _ in range(remaining)]
            for sample in exploratory:
                sample["meta_similarity"] = 0.0
            selected.extend(exploratory)

        # Blend exploitation/exploration by perturbing blueprint parameters.
        return [self._inject_exploration(candidate, search_space) for candidate in selected]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_descriptors(self) -> Dict[str, float]:
        if self.config.descriptor_provider:
            try:
                descriptors = self.config.descriptor_provider() or {}
                self.logger.debug("Resolved descriptors from provider: %s", descriptors)
                return descriptors
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.warning("Descriptor provider failed: %s", exc)
        return {}

    def _resolve_blueprints(self) -> Iterable[Dict[str, Any]]:
        if self.config.blueprint_provider:
            try:
                blueprints = self.config.blueprint_provider() or []
                self.logger.debug("Resolved %d blueprints from provider", len(blueprints))
                return blueprints
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.warning("Blueprint provider failed: %s", exc)
        return self.config.historical_blueprints

    def _build_descriptor_matrix(self, blueprints: Sequence[Dict[str, Any]]) -> np.ndarray:
        vectors = []
        for blueprint in blueprints:
            descriptors = blueprint.get("meta_features", {})
            vectors.append(self._embed_descriptors(descriptors))
        if not vectors:
            return np.zeros((0, len(self.config.descriptor_keys)), dtype=float)
        return np.vstack(vectors)

    def _embed_descriptors(self, descriptors: Dict[str, float]) -> np.ndarray:
        vector = np.array([float(descriptors.get(key, 0.0)) for key in self.config.descriptor_keys])
        if not vector.size:
            return vector
        if self.config.descriptor_normalization == "zscore":
            mean = np.mean(vector)
            std = np.std(vector) or 1.0
            vector = (vector - mean) / std
        elif self.config.descriptor_normalization == "minmax":
            min_val = np.min(vector)
            max_val = np.max(vector)
            span = max(max_val - min_val, 1e-8)
            vector = (vector - min_val) / span
        return vector

    def _cosine_similarity(self, matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        if matrix.size == 0 or vector.size == 0:
            return np.zeros(len(matrix), dtype=float)
        norm_vector = np.linalg.norm(vector) or 1.0
        norm_matrix = np.linalg.norm(matrix, axis=1)
        norm_matrix[norm_matrix == 0] = 1.0
        similarity = matrix @ vector / (norm_matrix * norm_vector)
        return np.clip(similarity, -1.0, 1.0)

    def _project_blueprint(
        self, blueprint: Dict[str, Any], search_space: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Project blueprint parameters into the active search space."""

        params = dict(blueprint.get("parameters", {}))
        projected = {}
        for key, definition in search_space.items():
            if key in params:
                projected[key] = params[key]
                continue
            projected[key] = self._sample_from_definition(definition)
        return projected

    def _sample_from_definition(self, definition: Any) -> Any:
        if isinstance(definition, dict):
            if "choices" in definition:
                return np.random.choice(definition["choices"])
            low = definition.get("low")
            high = definition.get("high")
            if low is not None and high is not None:
                return float(np.random.uniform(low, high))
        if isinstance(definition, (list, tuple)):
            if len(definition) == 2 and all(isinstance(v, (int, float)) for v in definition):
                return float(np.random.uniform(definition[0], definition[1]))
            return np.random.choice(definition)
        return definition

    def _inject_exploration(self, candidate: Dict[str, Any], search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Blend exploitation with exploration noise to avoid mode collapse."""

        exploratory_candidate = dict(candidate)
        for key, definition in search_space.items():
            if not isinstance(candidate.get(key), (int, float)):
                continue
            noise_scale = self.config.exploration_weight
            if isinstance(definition, dict) and {"low", "high"}.issubset(definition):
                span = definition["high"] - definition["low"]
                noise_scale *= span
            elif isinstance(definition, (list, tuple)) and len(definition) == 2:
                span = definition[1] - definition[0]
                noise_scale *= span
            else:
                noise_scale *= abs(candidate[key]) or 1.0
            exploratory_candidate[key] = float(candidate[key] + np.random.normal(0, noise_scale))
        return exploratory_candidate


__all__ = [
    "MetaWarmStartConfig",
    "MetaWarmStarter",
]
