"""Entropy stability filtering utilities for feature selection."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class EntropyBalancerConfig:
    """Configuration for entropy stability filtering."""

    num_slices: int = 12
    min_slice_size: int = 100
    max_entropy_variance: float = 0.12
    max_bins: int = 15
    min_unique_values: int = 5
    use_time_index: bool = True
    normalize_entropy: bool = True

@dataclass
class EntropyFilterResult:
    """Result of entropy stability filtering."""

    selected_features: List[str] = field(default_factory=list)
    dropped_features: Dict[str, float] = field(default_factory=dict)
    entropy_history: Dict[str, List[float]] = field(default_factory=dict)
    entropy_variance: Dict[str, float] = field(default_factory=dict)
    stability_scores: Dict[str, float] = field(default_factory=dict)

class EntropyStabilityFilter:
    """Evaluate and filter features based on entropy stability across time slices."""

    def __init__(self, config: Optional[EntropyBalancerConfig] = None):
        self.config = config or EntropyBalancerConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

    def filter(self, features: pd.DataFrame) -> EntropyFilterResult:
        """Filter features whose entropy fluctuates excessively across time slices."""

        if features.empty:
            return EntropyFilterResult(selected_features=[], stability_scores={}, entropy_variance={})

        if self.config.use_time_index and isinstance(features.index, pd.DatetimeIndex):
            features = features.sort_index()

        if self.config.num_slices < 2 or len(features) < self.config.min_slice_size:
            self.logger.debug(
                "Skipping entropy filtering – insufficient data (rows=%s, slices=%s)",
                len(features),
                self.config.num_slices,
            )
            return EntropyFilterResult(
                selected_features=features.columns.tolist(),
                stability_scores={col: 1.0 for col in features.columns},
                entropy_variance={col: 0.0 for col in features.columns},
                entropy_history={col: [] for col in features.columns},
            )

        slice_boundaries = self._compute_slice_boundaries(len(features))

        result = EntropyFilterResult()

        for column in features.columns:
            column_values = features[column].to_numpy(copy=False)
            entropies: List[float] = []

            for start, end in slice_boundaries:
                slice_values = column_values[start:end]
                if len(slice_values) < self.config.min_slice_size:
                    continue
                entropy = self._compute_entropy(slice_values)
                entropies.append(entropy)

            result.entropy_history[column] = entropies

            if not entropies:
                variance = 0.0
            else:
                variance = float(np.nanvar(entropies))

            stability = 1.0 / (1.0 + variance)

            result.entropy_variance[column] = variance
            result.stability_scores[column] = stability

            if variance > self.config.max_entropy_variance:
                result.dropped_features[column] = variance
            else:
                result.selected_features.append(column)

        return result

    def _compute_slice_boundaries(self, length: int) -> List[tuple[int, int]]:
        """Compute start and end indices for each time slice."""

        indices = np.linspace(0, length, num=self.config.num_slices + 1, dtype=int)
        indices = np.unique(indices)
        slices: List[tuple[int, int]] = []

        for start, end in zip(indices[:-1], indices[1:]):
            if end - start >= self.config.min_slice_size:
                slices.append((start, end))

        return slices

    def _compute_entropy(self, values: np.ndarray) -> float:
        """Compute normalized entropy for a slice of values."""

        values = values[np.isfinite(values)]
        if values.size < self.config.min_slice_size or len(np.unique(values)) < self.config.min_unique_values:
            return 0.0

        bins = min(self.config.max_bins, len(np.unique(values)))
        if bins < 2:
            return 0.0

        hist, _ = np.histogram(values, bins=bins, density=False)
        total = hist.sum()
        if total == 0:
            return 0.0

        probs = hist / total
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs))

        if not self.config.normalize_entropy:
            return float(entropy)

        max_entropy = np.log(bins)
        return float(entropy / max_entropy) if max_entropy > 0 else 0.0
