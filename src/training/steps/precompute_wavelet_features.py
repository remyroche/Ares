"""
Wavelet feature precomputation utilities with high-performance optimizations.

This module provides optimized wavelet feature extraction with:
- Numba JIT compilation for compute-intensive operations
- Vectorized processing for better performance
- Memory-efficient implementations
- Comprehensive error handling and logging
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
import time

# Try to import Numba for JIT compilation
try:
    from numba import jit, prange, float64
    import numba as nb
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# Try to import psutil for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from src.utils.comprehensive_function_logger import (
    log_step_functions, log_important_calls, log_all_calls,
    log_internal_call, log_step_progress, log_data_operation
)
from src.utils.logger import system_logger

# Optimized wavelet feature computation functions
if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, fastmath=True)
    def numba_wavelet_energy(coeffs_array: np.ndarray, n_levels: int) -> np.ndarray:
        """Numba-optimized wavelet energy calculation with parallel processing."""
        n_samples = coeffs_array.shape[0]
        energy = np.zeros(n_samples)

        for i in prange(n_samples):
            energy_sum = 0.0
            for level in range(n_levels):
                if level < coeffs_array.shape[1]:
                    coeff_val = coeffs_array[i, level]
                    energy_sum += coeff_val * coeff_val
            energy[i] = energy_sum

        return energy

    @jit(nopython=True, parallel=True, fastmath=True)
    def numba_wavelet_entropy(coeffs_array: np.ndarray, n_levels: int) -> np.ndarray:
        """Numba-optimized wavelet entropy calculation with parallel processing."""
        n_samples = coeffs_array.shape[0]
        entropy = np.zeros(n_samples)
        eps = 1e-10

        for i in prange(n_samples):
            entropy_sum = 0.0
            for level in range(n_levels):
                if level < coeffs_array.shape[1]:
                    coeff_val = np.abs(coeffs_array[i, level])
                    if coeff_val > eps:
                        entropy_sum += coeff_val * np.log(coeff_val + eps)
            entropy[i] = -entropy_sum

        return entropy

    @jit(nopython=True)
    def numba_pad_coefficients(coeff: np.ndarray, target_length: int) -> np.ndarray:
        """Numba-optimized coefficient padding."""
        current_length = len(coeff)
        if current_length >= target_length:
            return coeff[:target_length]
        else:
            # Pad with zeros
            padded = np.zeros(target_length)
            padded[:current_length] = coeff
            return padded

class WaveletFeaturePrecomputer:
    """Precomputes wavelet features for training with high-performance optimizations."""
    @log_important_calls

    def __init__(self) -> None:
        self.logger = system_logger.getChild('WaveletFeaturePrecomputer')
        self.numba_available = NUMBA_AVAILABLE
        self.psutil_available = PSUTIL_AVAILABLE

        if self.numba_available:
            self.logger.info("🚀 Wavelet feature precomputation initialized with Numba JIT optimizations")
        else:
            self.logger.info("📊 Wavelet feature precomputation initialized (Numba not available)")

        if self.psutil_available:
            self.logger.info("📈 Memory monitoring enabled for wavelet operations")

    def precompute_features(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> None:
        """Precompute wavelet features."""
        return data

    def extract_wavelet_features(self, data: pd.DataFrame, price_column: str = 'close', symbol: str = 'SYMBOL', timeframe: str = '30m') -> pd.DataFrame:
        """Extract wavelet features from price data with high-performance optimizations."""
        start_time = time.time()
        start_memory = 0

        if self.psutil_available:
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024

        try:
            import pywt
        except ImportError:
            self.logger.warning("⚠️ PyWavelets not available, returning empty features")
            return pd.DataFrame(index=data.index)

        features = pd.DataFrame(index=data.index)

        if price_column not in data.columns:
            self.logger.warning(f"⚠️ Price column '{price_column}' not found in data")
            return features

        price_data = data[price_column].fillna(method='ffill').fillna(0).values
        n_samples = len(price_data)

        self.logger.info(f"🔍 Extracting wavelet features for {n_samples:,} data points using {'Numba-optimized' if self.numba_available else 'standard'} processing")

        try:
            # Perform wavelet decomposition
            coeffs = pywt.wavedec(price_data, 'db1', level=4)
            n_levels = len(coeffs)

            # Process coefficients with optimized padding
            coeff_arrays = []
            for i, coeff in enumerate(coeffs):
                if len(coeff) > 0:
                    if self.numba_available:
                        # Use Numba-optimized padding
                        padded_coeff = numba_pad_coefficients(coeff, n_samples)
                    else:
                        # Standard numpy padding
                        if len(coeff) != n_samples:
                            if len(coeff) < n_samples:
                                padded_coeff = np.pad(coeff, (0, n_samples - len(coeff)), 'constant')
                            else:
                                padded_coeff = coeff[:n_samples]
                        else:
                            padded_coeff = coeff

                    features[f'wavelet_level_{i}'] = padded_coeff
                    coeff_arrays.append(padded_coeff)

            # Compute wavelet energy and entropy using optimized functions
            if coeff_arrays:
                coeffs_matrix = np.column_stack(coeff_arrays)

                if self.numba_available:
                    # Use Numba-optimized calculations
                    energy_values = numba_wavelet_energy(coeffs_matrix, n_levels)
                    entropy_values = numba_wavelet_entropy(coeffs_matrix, n_levels)
                else:
                    # Fallback to numpy operations
                    energy_values = np.sum(coeffs_matrix ** 2, axis=1)
                    abs_coeffs = np.abs(coeffs_matrix)
                    entropy_values = -np.sum(abs_coeffs * np.log(abs_coeffs + 1e-10), axis=1)

                features['wavelet_energy'] = energy_values
                features['wavelet_entropy'] = entropy_values

            # Performance monitoring
            end_time = time.time()
            execution_time = end_time - start_time

            memory_info = ""
            if self.psutil_available:
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_delta = end_memory - start_memory
                memory_info = f", memory delta: {memory_delta:+.1f}MB"

            optimization_info = " (Numba accelerated)" if self.numba_available else ""
            self.logger.info(f"✅ Wavelet feature extraction completed in {execution_time:.3f}s{memory_info}{optimization_info}")

        except Exception as e:
            self.logger.error(f"❌ Wavelet feature extraction failed: {e}")
            return pd.DataFrame(index=data.index)

        return features