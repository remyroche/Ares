"""Common dataset-aware constants for NAS/TAS neural models."""

from __future__ import annotations

# Approximate number of samples from four years of 15-minute bars.
FOUR_YEARS_15M_SAMPLES: int = 4 * 365 * 24 * 4

# Recommended neural architecture limits calibrated to the available data volume.
RECOMMENDED_MIN_LAYERS: int = 2
RECOMMENDED_MAX_LAYERS: int = 6
RECOMMENDED_MIN_UNITS: int = 32
RECOMMENDED_MAX_UNITS: int = 128
RECOMMENDED_HIDDEN_SIZE_OPTIONS = (32, 48, 64, 96, 128)

# Rough estimate of input feature dimensionality for parameter calculations.
ESTIMATED_INPUT_FEATURES: int = 64

# Conservative parameter capacity so model size stays proportional to data volume.
DATA_AWARE_PARAMETER_CAPACITY: int = max(int(FOUR_YEARS_15M_SAMPLES * 0.5), 1)
