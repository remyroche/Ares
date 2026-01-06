import pywt
import numpy as np
from typing import Dict, List, Union

def wavelet_energy_ratios(series: Union[List[float], np.ndarray], wavelet: str = 'db4', level: int = 3) -> float:
    """
    Returns high/low frequency energy ratio for a 1D series.
    Pre-event only.

    Args:
        series: Input time series
        wavelet: Wavelet name (default 'db4')
        level: Decomposition level (default 3)

    Returns:
        noise_ratio: Ratio of high frequency energy to total energy
    """
    if len(series) < 2**level:
        return 0.5 # Default fallback for short series

    coeffs = pywt.wavedec(series, wavelet, level=level)

    # i=0 -> Approximation (low freq), i>0 -> Details (high freq)
    energy = [np.sum(c**2) for c in coeffs]

    low_freq_energy = energy[0]  # Approximation
    high_freq_energy = sum(energy[1:])  # All detail coefficients

    # Noise ratio (higher = more noise)
    noise_ratio = high_freq_energy / (high_freq_energy + low_freq_energy + 1e-8)

    return float(noise_ratio)

def wavelet_gate(noise_ratio: float) -> float:
    """
    Convert noise_ratio into a discrete sample weight.

    Args:
        noise_ratio: Calculated noise ratio

    Returns:
        weight: 1.0 (low noise), 0.5 (medium), 0.2 (high)
    """
    if noise_ratio < 0.3:
        return 1.0  # Low noise → full weight
    elif noise_ratio < 0.6:
        return 0.5  # Medium noise → downweight
    else:
        return 0.2  # High noise → suppress / discard

def get_wavelet_features(series: Union[List[float], np.ndarray], wavelet: str = 'db4', level: int = 4) -> Dict[str, float]:
    """
    Compute multi-scale wavelet features for a 1D pre-event series.

    Returns a dict with:
    - Energy per scale
    - Relative energy per scale
    - Entropy per scale
    - HF/LF energy ratio

    Args:
        series: Input time series
        wavelet: Wavelet name (default 'db4')
        level: Decomposition level (default 4)
    """
    features = {}
    if len(series) < 2**level:
        # Fallback for short series
        for i in range(level + 1):
            features[f'energy_lvl_{i}'] = 0.0
            features[f'rel_energy_lvl_{i}'] = 0.0
            features[f'entropy_lvl_{i}'] = 0.0
        features['hf_lf_ratio'] = 0.0
        return features

    coeffs = pywt.wavedec(series, wavelet, level=level)

    # Compute energy per scale
    energy = [np.sum(c**2) for c in coeffs]  # coeffs[0]=low freq, 1..=details
    total_energy = sum(energy) + 1e-8

    for i, e in enumerate(energy):
        features[f'energy_lvl_{i}'] = float(e)
        features[f'rel_energy_lvl_{i}'] = float(e / total_energy)

    # HF/LF ratio: sum of all detail energies / approximation energy
    high_freq_energy = sum(energy[1:])
    low_freq_energy = energy[0]
    features['hf_lf_ratio'] = float(high_freq_energy / (low_freq_energy + 1e-8))

    # Entropy per scale
    for i, c in enumerate(coeffs):
        p = np.square(c) / (np.sum(np.square(c)) + 1e-8)
        features[f'entropy_lvl_{i}'] = float(-np.sum(p * np.log(p + 1e-8)))

    return features
