"""
Spectral and Wavelet Features Module

This module provides comprehensive spectral and wavelet analysis features
for quantitative finance, including frequency domain analysis, cycle detection,
and fractal dimension calculations.

Key Features:
- Wavelet energy analysis and decomposition
- Band-limited volatility using spectral analysis
- Cycle length detection and frequency analysis
- Fractal dimension and complexity measures
- Detrended Fluctuation Analysis (DFA) slopes
- Full VectorBT integration for optimal performance
"""

# Standard library imports
import warnings
from typing import Any, Dict, List, Optional, Union, Tuple

# Third-party imports
import numpy as np
import pandas as pd

# Optional third-party imports
try:
    from scipy import stats
    from scipy.signal import find_peaks, welch
    from scipy.fft import fft, fftfreq
    from scipy.stats import skew, kurtosis, jarque_bera
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None
    find_peaks = None
    welch = None
    fft = None
    fftfreq = None
    skew = None
    kurtosis = None
    jarque_bera = None
    warnings.warn("SciPy not available. Some spectral features may not work properly")

try:
    import pywt
    PYWAVELETS_AVAILABLE = True
except ImportError:
    PYWAVELETS_AVAILABLE = False
    pywt = None
    warnings.warn("PyWavelets not available. Wavelet features will be disabled")

try:
    import vectorbt as vbt
    from src.utils.vectorbt_compat import (
        rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max,
        rolling_sum, rolling_apply, rolling_corr, rolling_cov,
        rolling_skew, rolling_kurt, rolling_quantile
    )
    from src.utils.vectorbt_compat import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    rolling_skew = None
    rolling_kurt = None
    rolling_quantile = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Local imports
from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

# Optimization utilities
try:
    from src.feature_generation.utils.vectorization_optimizer import get_vectorization_optimizer
    from src.feature_generation.utils.optimized_feature_pipeline import get_optimized_feature_pipeline
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    get_vectorization_optimizer = None
    get_optimized_feature_pipeline = None

try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer, VectorBTRollingOptimizer
    VECTORBT_ROLLING_OPTIMIZER_AVAILABLE = True
except ImportError:
    VECTORBT_ROLLING_OPTIMIZER_AVAILABLE = False
    VectorBTRollingOptimizer = None

# Import tprint for consistent logging
try:
    from src.utils.tprint import tprint
except ImportError:
    def tprint(*args, **kwargs):
        print(*args, **kwargs)

class WaveletEnergyGenerator(VectorizedFeatureGenerator):
    """
    Generator for wavelet energy features.

    Wavelet energy analysis provides insights into the frequency content of
    financial time series, helping identify dominant cycles and patterns.

    Parameters:
    - window: Lookback window for calculation (default: 64)
    - wavelet: Wavelet type for decomposition (default: 'db4')

    Returns:
    - pd.Series: Wavelet energy values

    Example:
        >>> generator = WaveletEnergyGenerator(window=32)
        >>> energy = generator._generate_feature(data)
        >>> print(f"Average wavelet energy: {energy.mean():.3f}")
    """

    def __init__(self, window: int = 64, wavelet: str = 'db4'):
        config = FeatureConfig(
            name="wavelet_energy",
            category=FeatureCategory.SPECTRAL_WAVELET,
            description="Wavelet energy analysis for frequency domain features",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=32,
            max_lookback=256,
            parameters={"window": window, "wavelet": wavelet}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.wavelet = wavelet

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate wavelet energy feature."""
        close = data['close']

        if PYWAVELETS_AVAILABLE:
            try:
                # Use rolling apply for wavelet energy calculation
                energy_series = close.rolling(window=self.window).apply(
                    lambda x: self._calculate_wavelet_energy(x),
                    raw=False
                )
                return energy_series
            except Exception as e:
                warnings.warn(f"Wavelet energy calculation failed: {e}, using fallback")
                return self._calculate_wavelet_energy_fallback(close, self.window, data.index)
        else:
            warnings.warn("PyWavelets not available, using FFT-based approximation")
            return self._calculate_fft_energy_fallback(close, self.window, data.index)

    def _calculate_wavelet_energy(self, prices: pd.Series) -> float:
        """Calculate wavelet energy for a price series."""
        try:
            if len(prices) < 8:
                return 0.0

            # Remove NaN values
            prices = prices.dropna()
            if len(prices) < 8:
                return 0.0

            # Perform wavelet decomposition
            coeffs = pywt.wavedec(prices.values, self.wavelet, level=3)

            # Calculate energy for each level
            energy = 0.0
            for coeff in coeffs:
                energy += np.sum(coeff ** 2)

            return energy / len(prices)

        except Exception:
            return 0.0

    def _calculate_wavelet_energy_fallback(self, close: pd.Series, window: int, index: pd.Index) -> pd.Series:
        """Fallback wavelet energy calculation using pandas rolling."""
        energy_values = []
        for i in range(len(close)):
            if i < window - 1:
                energy_values.append(np.nan)
            else:
                segment = close.iloc[i-window+1:i+1]
                energy = self._calculate_wavelet_energy(segment)
                energy_values.append(energy)

        return pd.Series(energy_values, index=index)

    def _calculate_fft_energy_fallback(self, close: pd.Series, window: int, index: pd.Index) -> pd.Series:
        """FFT-based energy calculation as fallback."""
        energy_values = []
        for i in range(len(close)):
            if i < window - 1:
                energy_values.append(np.nan)
            else:
                segment = close.iloc[i-window+1:i+1].dropna()
                if len(segment) >= 8:
                    # Use FFT as approximation
                    fft_values = np.abs(np.fft.fft(segment.values))
                    energy = np.sum(fft_values ** 2) / len(segment)
                else:
                    energy = 0.0
                energy_values.append(energy)

        return pd.Series(energy_values, index=index)

class BandLimitedVolatilityGenerator(VectorizedFeatureGenerator):
    """
    Generator for band-limited volatility features.

    Band-limited volatility uses spectral analysis to focus on specific
    frequency ranges, providing cleaner volatility measures.

    Parameters:
    - window: Lookback window for calculation (default: 32)
    - low_freq: Lower frequency bound (default: 0.1)
    - high_freq: Upper frequency bound (default: 0.5)

    Returns:
    - pd.Series: Band-limited volatility values

    Example:
        >>> generator = BandLimitedVolatilityGenerator(window=20)
        >>> vol = generator._generate_feature(data)
        >>> print(f"Average band-limited volatility: {vol.mean():.3f}")
    """

    def __init__(self, window: int = 32, low_freq: float = 0.1, high_freq: float = 0.5):
        config = FeatureConfig(
            name="band_limited_volatility",
            category=FeatureCategory.SPECTRAL_WAVELET,
            description="Band-limited volatility using spectral analysis",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=16,
            max_lookback=72,
            parameters={"window": window, "low_freq": low_freq, "high_freq": high_freq}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.low_freq = low_freq
        self.high_freq = high_freq

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate band-limited volatility feature."""
        close = data['close']
        returns = close.pct_change().dropna()

        if SCIPY_AVAILABLE:
            try:
                # Use rolling apply for band-limited volatility calculation
                volatility_series = returns.rolling(window=self.window).apply(
                    lambda x: self._calculate_band_limited_volatility(x),
                    raw=False
                )
                return volatility_series
            except Exception as e:
                warnings.warn(f"Band-limited volatility calculation failed: {e}, using fallback")
                return self._calculate_band_limited_volatility_fallback(returns, self.window, data.index)
        else:
            return self._calculate_band_limited_volatility_fallback(returns, self.window, data.index)

    def _calculate_band_limited_volatility(self, returns: pd.Series) -> float:
        """Calculate band-limited volatility for a returns series."""
        try:
            if len(returns) < 16:
                return 0.0

            # Remove NaN values
            returns = returns.dropna()
            if len(returns) < 16:
                return 0.0

            # Calculate power spectral density
            freqs, psd = welch(returns.values, nperseg=min(16, len(returns)//2))

            # Find frequency band
            freq_mask = (freqs >= self.low_freq) & (freqs <= self.high_freq)

            # Calculate band-limited volatility
            band_limited_vol = np.sqrt(np.sum(psd[freq_mask]))

            return band_limited_vol

        except Exception:
            return 0.0

    def _calculate_band_limited_volatility_fallback(self, returns: pd.Series, window: int, index: pd.Index) -> pd.Series:
        """Fallback band-limited volatility calculation using pandas rolling."""
        volatility_values = []
        for i in range(len(returns)):
            if i < window - 1:
                volatility_values.append(np.nan)
            else:
                segment = returns.iloc[i-window+1:i+1]
                volatility = self._calculate_band_limited_volatility(segment)
                volatility_values.append(volatility)

        return pd.Series(volatility_values, index=index)

class CycleLengthGenerator(VectorizedFeatureGenerator):
    """
    Generator for cycle length detection features.

    Detects dominant cycles in price data using spectral analysis,
    helping identify recurring patterns and market cycles.

    Parameters:
    - window: Lookback window for calculation (default: 64)
    - min_cycle: Minimum cycle length (default: 4)
    - max_cycle: Maximum cycle length (default: 32)

    Returns:
    - pd.Series: Detected cycle lengths

    Example:
        >>> generator = CycleLengthGenerator(window=40)
        >>> cycles = generator._generate_feature(data)
        >>> print(f"Average cycle length: {cycles.mean():.1f}")
    """

    def __init__(self, window: int = 64, min_cycle: int = 4, max_cycle: int = 32):
        config = FeatureConfig(
            name="cycle_length",
            category=FeatureCategory.SPECTRAL_WAVELET,
            description="Cycle length detection using spectral analysis",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=32,
            max_lookback=256,
            parameters={"window": window, "min_cycle": min_cycle, "max_cycle": max_cycle}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.min_cycle = min_cycle
        self.max_cycle = max_cycle

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate cycle length feature."""
        close = data['close']

        if SCIPY_AVAILABLE:
            try:
                # Use rolling apply for cycle length calculation
                cycle_series = close.rolling(window=self.window).apply(
                    lambda x: self._detect_cycle_length(x),
                    raw=False
                )
                return cycle_series
            except Exception as e:
                warnings.warn(f"Cycle length detection failed: {e}, using fallback")
                return self._detect_cycle_length_fallback(close, self.window, data.index)
        else:
            return self._detect_cycle_length_fallback(close, self.window, data.index)

    def _detect_cycle_length(self, prices: pd.Series) -> float:
        """Detect cycle length in a price series."""
        try:
            if len(prices) < 16:
                return 0.0

            # Remove NaN values
            prices = prices.dropna()
            if len(prices) < 16:
                return 0.0

            # Calculate FFT
            fft_values = np.abs(fft(prices.values))
            freqs = fftfreq(len(prices))

            # Find dominant frequency
            # Exclude DC component and negative frequencies
            positive_freqs = freqs[1:len(freqs)//2]
            positive_fft = fft_values[1:len(fft_values)//2]

            if len(positive_fft) == 0:
                return 0.0

            # Find peak frequency
            peak_idx = np.argmax(positive_fft)
            peak_freq = positive_freqs[peak_idx]

            # Convert to cycle length
            if peak_freq > 0:
                cycle_length = 1.0 / peak_freq
                # Clamp to reasonable range
                cycle_length = max(self.min_cycle, min(self.max_cycle, cycle_length))
            else:
                cycle_length = 0.0

            return cycle_length

        except Exception:
            return 0.0

    def _detect_cycle_length_fallback(self, close: pd.Series, window: int, index: pd.Index) -> pd.Series:
        """Fallback cycle length detection using pandas rolling."""
        cycle_values = []
        for i in range(len(close)):
            if i < window - 1:
                cycle_values.append(np.nan)
            else:
                segment = close.iloc[i-window+1:i+1]
                cycle_length = self._detect_cycle_length(segment)
                cycle_values.append(cycle_length)

        return pd.Series(cycle_values, index=index)

class FractalDimensionGenerator(VectorizedFeatureGenerator):
    """
    Generator for fractal dimension features.

    Fractal dimension measures the complexity and irregularity of price
    movements, providing insights into market behavior and patterns.

    Parameters:
    - window: Lookback window for calculation (default: 32)

    Returns:
    - pd.Series: Fractal dimension values (1.0 to 2.0)

    Example:
        >>> generator = FractalDimensionGenerator(window=20)
        >>> fractal = generator._generate_feature(data)
        >>> print(f"Average fractal dimension: {fractal.mean():.3f}")
    """

    def __init__(self, window: int = 32):
        config = FeatureConfig(
            name="fractal_dimension",
            category=FeatureCategory.SPECTRAL_WAVELET,
            description="Fractal dimension analysis for complexity measurement",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=16,
            max_lookback=72,
            parameters={"window": window}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate fractal dimension feature."""
        close = data['close']

        try:
            # Use rolling apply for fractal dimension calculation
            fractal_series = close.rolling(window=self.window).apply(
                lambda x: self._calculate_fractal_dimension(x),
                raw=False
            )
            return fractal_series
        except Exception as e:
            warnings.warn(f"Fractal dimension calculation failed: {e}, using fallback")
            return self._calculate_fractal_dimension_fallback(close, self.window, data.index)

    def _calculate_fractal_dimension(self, prices: pd.Series) -> float:
        """Calculate fractal dimension using box-counting method."""
        try:
            if len(prices) < 8:
                return 1.0

            # Remove NaN values
            prices = prices.dropna()
            if len(prices) < 8:
                return 1.0

            # Normalize prices
            prices_norm = (prices - prices.min()) / (prices.max() - prices.min())

            # Box-counting method
            scales = [2, 4, 8, 16]
            counts = []

            for scale in scales:
                if scale >= len(prices_norm):
                    continue

                # Create boxes
                box_size = len(prices_norm) // scale
                if box_size == 0:
                    continue

                count = 0
                for i in range(scale):
                    start_idx = i * box_size
                    end_idx = min((i + 1) * box_size, len(prices_norm))

                    if start_idx < end_idx:
                        box_data = prices_norm.iloc[start_idx:end_idx]
                        if len(box_data) > 0:
                            count += 1

                counts.append(count)

            if len(counts) < 2:
                return 1.0

            # Calculate fractal dimension
            scales = scales[:len(counts)]
            log_scales = np.log(scales)
            log_counts = np.log(counts)

            # Linear regression
            if len(log_scales) > 1:
                slope, _ = np.polyfit(log_scales, log_counts, 1)
                fractal_dim = -slope
            else:
                fractal_dim = 1.0

            # Clamp to reasonable range
            return max(1.0, min(2.0, fractal_dim))

        except Exception:
            return 1.0

    def _calculate_fractal_dimension_fallback(self, close: pd.Series, window: int, index: pd.Index) -> pd.Series:
        """Fallback fractal dimension calculation using pandas rolling."""
        fractal_values = []
        for i in range(len(close)):
            if i < window - 1:
                fractal_values.append(np.nan)
            else:
                segment = close.iloc[i-window+1:i+1]
                fractal_dim = self._calculate_fractal_dimension(segment)
                fractal_values.append(fractal_dim)

        return pd.Series(fractal_values, index=index)

class DFASlopesGenerator(VectorizedFeatureGenerator):
    """
    Generator for Detrended Fluctuation Analysis (DFA) slopes.

    DFA slopes measure long-range correlations in time series data,
    helping identify persistent or anti-persistent behavior.

    Parameters:
    - window: Lookback window for calculation (default: 64)
    - min_scale: Minimum scale for DFA (default: 4)
    - max_scale: Maximum scale for DFA (default: 32)

    Returns:
    - pd.Series: DFA slope values

    Example:
        >>> generator = DFASlopesGenerator(window=40)
        >>> dfa = generator._generate_feature(data)
        >>> print(f"Average DFA slope: {dfa.mean():.3f}")
    """

    def __init__(self, window: int = 64, min_scale: int = 4, max_scale: int = 32):
        config = FeatureConfig(
            name="dfa_slopes",
            category=FeatureCategory.SPECTRAL_WAVELET,
            description="Detrended Fluctuation Analysis slopes for long-range correlation",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=32,
            max_lookback=256,
            parameters={"window": window, "min_scale": min_scale, "max_scale": max_scale}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.min_scale = min_scale
        self.max_scale = max_scale

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate DFA slopes feature."""
        close = data['close']
        returns = close.pct_change().dropna()

        try:
            # Use rolling apply for DFA calculation
            dfa_series = returns.rolling(window=self.window).apply(
                lambda x: self._calculate_dfa_slope(x),
                raw=False
            )
            return dfa_series
        except Exception as e:
            warnings.warn(f"DFA calculation failed: {e}, using fallback")
            return self._calculate_dfa_slope_fallback(returns, self.window, data.index)

    def _calculate_dfa_slope(self, returns: pd.Series) -> float:
        """Calculate DFA slope for a returns series."""
        try:
            # Remove NaN values early
            returns = returns.dropna()

            if len(returns) < 16:
                return np.nan

            # Center the cumulative sum to reduce drift effects
            centered_returns = returns.values - np.mean(returns.values)
            y = np.cumsum(centered_returns)

            max_scale = min(self.max_scale, len(y) // 4)
            if max_scale <= self.min_scale:
                return np.nan

            # Generate logarithmically spaced scales
            scales = np.logspace(
                np.log10(self.min_scale),
                np.log10(max_scale),
                num=8
            ).astype(int)
            scales = np.unique(scales)

            fluctuations: List[float] = []
            valid_scales: List[int] = []

            for scale in scales:
                if scale < 4:
                    continue

                n_segments = len(y) // scale
                if n_segments < 2:
                    continue

                segment_fluctuations: List[float] = []
                for i in range(n_segments):
                    start_idx = i * scale
                    end_idx = start_idx + scale
                    segment = y[start_idx:end_idx]

                    if len(segment) < 4:
                        continue

                    x = np.arange(len(segment))
                    coeffs = np.polyfit(x, segment, 1)
                    trend = np.polyval(coeffs, x)
                    detrended = segment - trend

                    rms = np.sqrt(np.mean(detrended ** 2))
                    if np.isfinite(rms) and rms > 0:
                        segment_fluctuations.append(rms)

                if segment_fluctuations:
                    fluctuations.append(float(np.mean(segment_fluctuations)))
                    valid_scales.append(scale)

            if len(fluctuations) < 2:
                return np.nan

            log_scales = np.log(valid_scales)
            log_fluctuations = np.log(fluctuations)

            slope = np.polyfit(log_scales, log_fluctuations, 1)[0]
            return float(np.clip(slope, 0.0, 2.0))

        except Exception as e:
            warnings.warn(f"DFA slope calculation failed: {e}")
            return np.nan

    def _calculate_dfa_slope_fallback(self, returns: pd.Series, window: int, index: pd.Index) -> pd.Series:
        """Fallback DFA calculation using pandas rolling."""
        dfa_values = []
        for i in range(len(returns)):
            if i < window - 1:
                dfa_values.append(np.nan)
            else:
                segment = returns.iloc[i-window+1:i+1]
                dfa_slope = self._calculate_dfa_slope(segment)
                dfa_values.append(dfa_slope)

        return pd.Series(dfa_values, index=index)

class VectorBTSpectralWaveletBatchGenerator(VectorizedFeatureGenerator):
    """
    Batch generator for spectral and wavelet features using VectorBT.

    This generator efficiently processes multiple spectral and wavelet
    features in batch using VectorBT optimization.

    Parameters:
    - window: Lookback window for calculation (default: 64)

    Returns:
    - Dict[str, np.ndarray]: Dictionary of spectral and wavelet features

    Example:
        >>> generator = VectorBTSpectralWaveletBatchGenerator(window=32)
        >>> features = generator.generate_features(data)
        >>> print(f"Generated {len(features)} spectral features")
    """

    def __init__(self, window: int = 64):
        config = FeatureConfig(
            name="vectorbt_spectral_wavelet_batch",
            category=FeatureCategory.SPECTRAL_WAVELET,
            description="Batch spectral and wavelet features using VectorBT optimization",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=32,
            max_lookback=256,
            parameters={"window": window}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window

        # Initialize VectorBT optimizer
        self.vectorbt_rolling_optimizer = None
        if VECTORBT_ROLLING_OPTIMIZER_AVAILABLE:
            try:
                self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer()
            except Exception as e:
                tprint(f"⚠️ VectorBT optimizer initialization failed: {e}")

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate batch spectral and wavelet features."""
        features = self.generate_features(data, **kwargs)

        # Return the first feature as representative
        if features:
            first_feature_name = list(features.keys())[0]
            return pd.Series(features[first_feature_name], index=data.index[:len(features[first_feature_name])])
        else:
            return pd.Series(np.zeros(len(data)), index=data.index)

    def generate_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """Generate all spectral and wavelet features in batch."""
        features = {}

        try:
            close = data['close']

            # Spectral features
            if SCIPY_AVAILABLE:
                try:
                    # Power spectral density
                    returns = close.pct_change().dropna()
                    if len(returns) >= 16:
                        freqs, psd = welch(returns.values, nperseg=min(16, len(returns)//2))
                        features['spectral_power'] = np.tile(np.sum(psd), len(close))
                        features['spectral_centroid'] = np.tile(np.sum(freqs * psd) / np.sum(psd), len(close))
                except Exception as e:
                    tprint(f"⚠️ Spectral features failed: {e}")

            # Wavelet features
            if PYWAVELETS_AVAILABLE:
                try:
                    # Wavelet energy
                    wavelet_energy = close.rolling(window=self.window).apply(
                        lambda x: self._calculate_wavelet_energy_batch(x),
                        raw=False
                    )
                    features['wavelet_energy'] = wavelet_energy.values
                except Exception as e:
                    tprint(f"⚠️ Wavelet features failed: {e}")

            # Fractal dimension
            try:
                fractal_dim = close.rolling(window=self.window).apply(
                    lambda x: self._calculate_fractal_dimension_batch(x),
                    raw=False
                )
                features['fractal_dimension'] = fractal_dim.values
            except Exception as e:
                tprint(f"⚠️ Fractal dimension failed: {e}")

        except Exception as e:
            tprint(f"⚠️ Spectral/wavelet batch generation failed: {e}")

        return features

    def _calculate_wavelet_energy_batch(self, prices: pd.Series) -> float:
        """Calculate wavelet energy for batch processing."""
        try:
            if len(prices) < 8:
                return 0.0

            prices = prices.dropna()
            if len(prices) < 8:
                return 0.0

            coeffs = pywt.wavedec(prices.values, 'db4', level=3)
            energy = 0.0
            for coeff in coeffs:
                energy += np.sum(coeff ** 2)

            return energy / len(prices)

        except Exception:
            return 0.0

    def _calculate_fractal_dimension_batch(self, prices: pd.Series) -> float:
        """Calculate fractal dimension for batch processing."""
        try:
            if len(prices) < 8:
                return 1.0

            prices = prices.dropna()
            if len(prices) < 8:
                return 1.0

            prices_norm = (prices - prices.min()) / (prices.max() - prices.min())

            scales = [2, 4, 8, 16]
            counts = []

            for scale in scales:
                if scale >= len(prices_norm):
                    continue

                box_size = len(prices_norm) // scale
                if box_size == 0:
                    continue

                count = 0
                for i in range(scale):
                    start_idx = i * box_size
                    end_idx = min((i + 1) * box_size, len(prices_norm))

                    if start_idx < end_idx:
                        box_data = prices_norm.iloc[start_idx:end_idx]
                        if len(box_data) > 0:
                            count += 1

                counts.append(count)

            if len(counts) < 2:
                return 1.0

            scales = scales[:len(counts)]
            log_scales = np.log(scales)
            log_counts = np.log(counts)

            if len(log_scales) > 1:
                slope, _ = np.polyfit(log_scales, log_counts, 1)
                fractal_dim = -slope
            else:
                fractal_dim = 1.0

            return max(1.0, min(2.0, fractal_dim))

        except Exception:
            return 1.0

# ============================================================================
# SPECTRAL FEATURE GENERATOR CLASS
# ============================================================================

class SpectralFeatureGenerator(VectorizedFeatureGenerator):
    """
    Main spectral feature generator that combines multiple spectral analysis techniques.
    
    This generator provides a unified interface for spectral analysis features
    including wavelet energy, band-limited volatility, cycle detection, and more.
    """
    
    def __init__(self, window: int = 64):
        config = FeatureConfig(
            name="spectral_features",
            category=FeatureCategory.SPECTRAL_WAVELET,
            description="Comprehensive spectral analysis features",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=32,
            max_lookback=256,
            parameters={"window": window}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        
        # Initialize sub-generators
        self.wavelet_generator = WaveletEnergyGenerator(window)
        self.volatility_generator = BandLimitedVolatilityGenerator(window)
        self.cycle_generator = CycleLengthGenerator(window)
        self.fractal_generator = FractalDimensionGenerator(window)
        self.dfa_generator = DFASlopesGenerator(window)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate comprehensive spectral features."""
        try:
            # Generate wavelet energy as primary feature
            wavelet_energy = self.wavelet_generator._generate_feature(data, **kwargs)
            return wavelet_energy
        except Exception as e:
            warnings.warn(f"Spectral feature generation failed: {e}")
            return pd.Series(np.zeros(len(data)), index=data.index)

class WaveletFeatureGenerator(VectorizedFeatureGenerator):
    """
    Wavelet feature generator for frequency domain analysis.
    
    This generator focuses specifically on wavelet-based features
    for time-frequency analysis of financial time series.
    """
    
    def __init__(self, window: int = 64, wavelet: str = 'db4'):
        config = FeatureConfig(
            name="wavelet_features",
            category=FeatureCategory.SPECTRAL_WAVELET,
            description="Wavelet-based frequency domain features",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=32,
            max_lookback=256,
            parameters={"window": window, "wavelet": wavelet}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.wavelet = wavelet
        
        # Initialize wavelet generator
        self.wavelet_energy_generator = WaveletEnergyGenerator(window, wavelet)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate wavelet features."""
        try:
            # Generate wavelet energy as primary feature
            wavelet_energy = self.wavelet_energy_generator._generate_feature(data, **kwargs)
            return wavelet_energy
        except Exception as e:
            warnings.warn(f"Wavelet feature generation failed: {e}")
            return pd.Series(np.zeros(len(data)), index=data.index)

class DetrendedFluctuationAnalysisGenerator(VectorizedFeatureGenerator):
    """
    Detrended Fluctuation Analysis (DFA) generator.
    
    This generator provides DFA-based features for analyzing
    long-range correlations in financial time series.
    """
    
    def __init__(self, window: int = 64, min_scale: int = 4, max_scale: int = 32):
        config = FeatureConfig(
            name="detrended_fluctuation_analysis",
            category=FeatureCategory.SPECTRAL_WAVELET,
            description="Detrended Fluctuation Analysis for long-range correlation",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=32,
            max_lookback=256,
            parameters={"window": window, "min_scale": min_scale, "max_scale": max_scale}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.min_scale = min_scale
        self.max_scale = max_scale
        
        # Initialize DFA generator
        self.dfa_generator = DFASlopesGenerator(window, min_scale, max_scale)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate DFA features."""
        try:
            # Generate DFA slopes as primary feature
            dfa_slopes = self.dfa_generator._generate_feature(data, **kwargs)
            return dfa_slopes
        except Exception as e:
            warnings.warn(f"DFA feature generation failed: {e}")
            return pd.Series(np.zeros(len(data)), index=data.index)

class VectorBTSpectralFeatureGenerator(VectorizedFeatureGenerator):
    """
    VectorBT-optimized spectral feature generator.
    
    This generator provides VectorBT-optimized spectral analysis features
    for high-performance frequency domain analysis.
    """
    
    def __init__(self, window: int = 64):
        config = FeatureConfig(
            name="vectorbt_spectral_features",
            category=FeatureCategory.SPECTRAL_WAVELET,
            description="VectorBT-optimized spectral analysis features",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=32,
            max_lookback=256,
            parameters={"window": window}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        
        # Initialize VectorBT batch generator
        self.batch_generator = VectorBTSpectralWaveletBatchGenerator(window)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate VectorBT-optimized spectral features."""
        try:
            # Generate batch features and return the first one
            features = self.batch_generator.generate_features(data, **kwargs)
            if features:
                first_feature_name = list(features.keys())[0]
                return pd.Series(features[first_feature_name], index=data.index[:len(features[first_feature_name])])
            else:
                return pd.Series(np.zeros(len(data)), index=data.index)
        except Exception as e:
            warnings.warn(f"VectorBT spectral feature generation failed: {e}")
            return pd.Series(np.zeros(len(data)), index=data.index)

# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_default_spectral_generators() -> List[FeatureGenerator]:
    """Create default spectral feature generators."""
    generators = []

    for window in [32, 64, 72]:
        generators.append(WaveletEnergyGenerator(window))
        generators.append(BandLimitedVolatilityGenerator(window))
        generators.append(CycleLengthGenerator(window))
        generators.append(FractalDimensionGenerator(window))
        generators.append(DFASlopesGenerator(window))
        generators.append(VectorBTSpectralWaveletBatchGenerator(window))

    return generators

def create_default_wavelet_generators() -> List[FeatureGenerator]:
    """Create default wavelet feature generators."""
    generators = []

    for window in [32, 64, 72]:
        generators.append(WaveletEnergyGenerator(window))
        generators.append(WaveletFeatureGenerator(window))

    return generators

def create_default_fractal_generators() -> List[FeatureGenerator]:
    """Create default fractal dimension generators."""
    generators = []

    for window in [32, 64, 72]:
        generators.append(FractalDimensionGenerator(window))

    return generators

def create_default_dfa_generators() -> List[FeatureGenerator]:
    """Create default DFA generators."""
    generators = []

    for window in [32, 64, 72]:
        generators.append(DFASlopesGenerator(window))
        generators.append(DetrendedFluctuationAnalysisGenerator(window))

    return generators

def create_spectral_feature_generators() -> List[FeatureGenerator]:
    """Create all spectral feature generators."""
    return create_default_spectral_generators()

def create_default_spectral_wavelet_generators() -> List[FeatureGenerator]:
    """Create default spectral and wavelet feature generators."""
    return create_default_spectral_generators()

def process_spectral_features_batch(data: pd.DataFrame,
                                  generators: Optional[List[FeatureGenerator]] = None,
                                  use_vectorbt: bool = True,
                                  **kwargs) -> pd.DataFrame:
    """
    Process spectral features in batch using VectorBT optimizations.

    Args:
        data: Input OHLCV data
        generators: List of feature generators (uses default if None)
        use_vectorbt: Whether to use VectorBT batch processing
        **kwargs: Additional parameters

    Returns:
        DataFrame with generated spectral features
    """
    if generators is None:
        generators = create_spectral_feature_generators()

    if use_vectorbt and OPTIMIZATION_AVAILABLE:
        try:
            # Use unified optimization system for batch processing
            from src.feature_generation.utils.unified_optimization_system import get_unified_optimization_system
            unified_optimizer = get_unified_optimization_system()

            # Process features in batch
            result = unified_optimizer.process_features_batch(data, generators, **kwargs)
            return result

        except Exception as e:
            warnings.warn(f"VectorBT batch processing failed: {e}, using sequential processing")
            return _process_spectral_features_sequential(data, generators, **kwargs)
    else:
        return _process_spectral_features_sequential(data, generators, **kwargs)

def _process_spectral_features_sequential(data: pd.DataFrame,
                                        generators: List[FeatureGenerator],
                                        **kwargs) -> pd.DataFrame:
    """Process spectral features sequentially (fallback)."""
    results = []

    for generator in generators:
        try:
            feature_result = generator._generate_feature(data, **kwargs)
            if not feature_result.empty:
                results.append(feature_result)
        except Exception as e:
            warnings.warn(f"Generator {generator.__class__.__name__} failed: {e}")
            continue

    if results:
        return pd.concat(results, axis=1)
    else:
        return pd.DataFrame(index=data.index)

__all__ = [
    'SpectralFeatureGenerator',
    'WaveletFeatureGenerator',
    'DetrendedFluctuationAnalysisGenerator',
    'VectorBTSpectralFeatureGenerator',
    'WaveletEnergyGenerator',
    'BandLimitedVolatilityGenerator',
    'CycleLengthGenerator',
    'FractalDimensionGenerator',
    'DFASlopesGenerator',
    'VectorBTSpectralWaveletBatchGenerator',
    'create_default_spectral_generators',
    'create_default_wavelet_generators',
    'create_default_fractal_generators',
    'create_default_dfa_generators',
    'create_spectral_feature_generators',
    'create_default_spectral_wavelet_generators',
    'process_spectral_features_batch'
]
