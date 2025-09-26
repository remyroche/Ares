"""Bayesian-ensemble surrogate modelling utilities for NAS/TAS searches."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _flatten_params(params: Dict[str, Any]) -> np.ndarray:
    """Project heterogeneous parameter dictionaries into a numeric vector."""

    vector: List[float] = []
    for key in sorted(params.keys()):
        value = params[key]
        if isinstance(value, (int, float)):
            vector.append(float(value))
        elif isinstance(value, (list, tuple)):
            vector.append(float(np.mean(value) if value else 0.0))
        else:
            vector.append(float(hash(str(value)) % 10_000) / 10_000.0)
    return np.asarray(vector, dtype=float)


@dataclass
class SurrogateObservation:
    features: np.ndarray
    score: float


@dataclass
class BayesianEnsembleConfig:
    """Configuration for the surrogate ensemble."""

    max_observations: int = 256
    exploration_beta: float = 2.0
    min_samples_for_prediction: int = 8
    jitter: float = 1e-6


class BayesianEnsembleSurrogate:
    """Simple bootstrap-based surrogate with epistemic uncertainty estimates."""

    def __init__(self, config: Optional[BayesianEnsembleConfig] = None) -> None:
        self.config = config or BayesianEnsembleConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._observations: List[SurrogateObservation] = []
        self._feature_cache: Dict[Tuple[Any, ...], np.ndarray] = {}

    # ------------------------------------------------------------------
    # Observation management
    # ------------------------------------------------------------------
    def update(self, params: Dict[str, Any], score: float) -> None:
        features = self._encode(params)
        self._observations.append(SurrogateObservation(features=features, score=float(score)))
        if len(self._observations) > self.config.max_observations:
            self._observations.pop(0)

    def _encode(self, params: Dict[str, Any]) -> np.ndarray:
        key = tuple(sorted(params.items()))
        cached = self._feature_cache.get(key)
        if cached is not None:
            return cached
        encoded = _flatten_params(params)
        self._feature_cache[key] = encoded
        return encoded

    # ------------------------------------------------------------------
    # Prediction interface
    # ------------------------------------------------------------------
    def predict(self, params: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        """Return (mean, std) for the provided parameter set if possible."""

        if len(self._observations) < self.config.min_samples_for_prediction:
            return None

        features = self._encode(params)
        distances = self._pairwise_distance(features)
        weights = self._distance_to_weight(distances)
        scores = np.array([obs.score for obs in self._observations], dtype=float)
        mean = float(np.sum(weights * scores))
        variance = float(np.sum(weights * (scores - mean) ** 2)) + self.config.jitter
        std = float(np.sqrt(max(variance, self.config.jitter)))
        return mean, std

    def compute_ucb(self, params: Dict[str, Any]) -> Optional[float]:
        prediction = self.predict(params)
        if prediction is None:
            return None
        mean, std = prediction
        return float(mean + self.config.exploration_beta * std)

    # ------------------------------------------------------------------
    # Distance helpers
    # ------------------------------------------------------------------
    def _pairwise_distance(self, features: np.ndarray) -> np.ndarray:
        matrix = np.vstack([obs.features for obs in self._observations])
        return np.linalg.norm(matrix - features, axis=1)

    def _distance_to_weight(self, distances: np.ndarray) -> np.ndarray:
        inverse = 1.0 / (distances + self.config.jitter)
        weights = inverse / np.sum(inverse)
        return weights


__all__ = [
    "BayesianEnsembleSurrogate",
    "BayesianEnsembleConfig",
]
