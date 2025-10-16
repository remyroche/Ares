"""Utility helpers for injecting low-amplitude cyclic noise into feature matrices."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ArrayLike = Union[np.ndarray, pd.DataFrame]

@dataclass
class CyclicNoiseConfig:
    """Configuration for cyclic noise injection."""

    noise_scale: float = 1e-3
    cycle_length: int = 512
    random_state: Optional[int] = None
    min_feature_scale: float = 1e-12

def add_cyclic_noise(
    features: ArrayLike,
    config: Optional[CyclicNoiseConfig] = None,
) -> ArrayLike:
    """Add small deterministic cyclic noise to break tree split ties.

    The function supports both :class:`numpy.ndarray` and :class:`pandas.DataFrame`
    inputs and returns a copy of the same type with noise added. The injected noise
    follows a repeating pattern so repeated training calls remain reproducible when
    a ``random_state`` is supplied.
    """

    if config is None:
        config = CyclicNoiseConfig()

    if config.noise_scale <= 0:
        return features.copy() if isinstance(features, (pd.DataFrame, np.ndarray)) else features

    data, to_dataframe = _as_ndarray(features)

    if data.size == 0:
        return features.copy() if isinstance(features, (pd.DataFrame, np.ndarray)) else features

    noise = _generate_cyclic_noise(data.shape, config)

    feature_scale = np.nanstd(data, axis=0, ddof=0)
    feature_scale = np.where(feature_scale < config.min_feature_scale, config.min_feature_scale, feature_scale)
    scaled_noise = noise * (feature_scale * config.noise_scale)

    noisy = data + scaled_noise

    return to_dataframe(noisy)

def _generate_cyclic_noise(shape: Tuple[int, int], config: CyclicNoiseConfig) -> np.ndarray:
    """Generate a deterministic cyclic noise pattern for the requested shape."""

    n_samples, n_features = shape

    if n_samples == 0 or n_features == 0:
        return np.zeros(shape, dtype=float)

    cycle_length = max(1, int(config.cycle_length))
    repeats = int(np.ceil(n_samples / cycle_length))

    rng = np.random.default_rng(config.random_state)
    base_cycle = rng.standard_normal((cycle_length, n_features))

    cycle_std = np.std(base_cycle, axis=0, ddof=0)
    cycle_std = np.where(cycle_std == 0, 1.0, cycle_std)
    normalized_cycle = base_cycle / cycle_std

    tiled = np.tile(normalized_cycle, (repeats, 1))[:n_samples]

    return tiled

def _as_ndarray(features: ArrayLike) -> Tuple[np.ndarray, callable]:
    """Return array representation and converter back to the original type."""

    if isinstance(features, pd.DataFrame):
        columns = features.columns
        index = features.index
        data = features.to_numpy(dtype=float, copy=True)

        def to_dataframe(array: np.ndarray) -> pd.DataFrame:
            return pd.DataFrame(array, index=index, columns=columns)

        return data, to_dataframe

    data = np.asarray(features, dtype=float)

    def to_array(array: np.ndarray) -> np.ndarray:
        return np.array(array, copy=True)

    return data.copy(), to_array
