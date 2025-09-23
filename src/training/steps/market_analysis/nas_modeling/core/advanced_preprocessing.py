"""
Advanced Time Series Preprocessing for NAS

This module provides advanced preprocessing techniques for time series data:
- Wavelet transforms for multi-resolution analysis
- Fourier analysis for frequency domain features
- Technical indicators and feature engineering
- Data augmentation for time series
- Noise reduction and filtering
- Seasonal decomposition
- Anomaly detection and handling
"""

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import butter, filtfilt, detrend
import pywt
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class PreprocessingConfig:
    """Configuration for advanced preprocessing."""
    use_wavelet_transform: bool = True
    wavelet_type: str = "db1"
    wavelet_level: int = 3
    use_fourier_features: bool = True
    n_fourier_components: int = 10
    use_technical_indicators: bool = True
    technical_indicators: List[str] = field(default_factory=lambda: ["sma", "ema", "rsi", "macd", "bollinger"])
    use_seasonal_decomposition: bool = True
    decomposition_period: int = 24
    use_data_augmentation: bool = True
    augmentation_methods: List[str] = field(default_factory=lambda: ["noise", "scaling", "time_warp", "magnitude_warp"])
    use_noise_reduction: bool = True
    noise_filter_type: str = "butterworth"
    filter_cutoff: float = 0.1
    use_anomaly_detection: bool = True
    anomaly_threshold: float = 3.0
    use_normalization: bool = True
    normalization_method: str = "robust"
    sequence_length: int = 100

class WaveletTransformer:
    """
    Wavelet transform for multi-resolution time series analysis.

    Decomposes time series into different frequency components.
    """

    def __init__(self, config: PreprocessingConfig):
        """Initialize wavelet transformer.

        Args:
            config: Preprocessing configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def wavelet_decomposition(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Perform wavelet decomposition on time series data.

        Args:
            data: Input time series data

        Returns:
            Dictionary of wavelet coefficients at different levels
        """
        if not self.config.use_wavelet_transform:
            return {"original": data}

        try:
            # Perform wavelet decomposition
            coeffs = pywt.wavedec(data, self.config.wavelet_type, level=self.config.wavelet_level)

            # Reconstruct approximations and details
            results = {"original": data}

            # Approximation (low-frequency)
            approx = pywt.waverec(coeffs[:-self.config.wavelet_level], self.config.wavelet_type)
            results["approximation"] = approx

            # Details (high-frequency components)
            for i in range(self.config.wavelet_level):
                detail = pywt.waverec(coeffs[-(i+1):], self.config.wavelet_type)
                results[f"detail_{i+1}"] = detail

            # Wavelet energy
            results["wavelet_energy"] = self._compute_wavelet_energy(coeffs)

            self.logger.info(f"✅ Wavelet decomposition completed with {len(results)} components")
            return results

        except Exception as e:
            self.logger.warning(f"⚠️ Wavelet decomposition failed: {e}")
            return {"original": data}

    def _compute_wavelet_energy(self, coeffs: List[np.ndarray]) -> np.ndarray:
        """Compute wavelet energy from coefficients."""
        energy = np.zeros(len(coeffs))
        for i, coeff in enumerate(coeffs):
            energy[i] = np.sum(coeff ** 2)
        return energy / np.sum(energy)  # Normalized energy

class FourierAnalyzer:
    """
    Fourier analysis for frequency domain features.

    Extracts frequency domain features from time series data.
    """

    def __init__(self, config: PreprocessingConfig):
        """Initialize Fourier analyzer.

        Args:
            config: Preprocessing configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def extract_fourier_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract Fourier features from time series.

        Args:
            data: Input time series data

        Returns:
            Dictionary of Fourier features
        """
        if not self.config.use_fourier_features:
            return {"original": data}

        try:
            results = {"original": data}

            # Compute FFT
            fft_values = fft(data)
            frequencies = fftfreq(len(data))

            # Power spectrum
            power_spectrum = np.abs(fft_values) ** 2
            results["power_spectrum"] = power_spectrum

            # Dominant frequencies
            dominant_freqs = frequencies[np.argsort(power_spectrum)[-self.config.n_fourier_components:]]
            results["dominant_frequencies"] = dominant_freqs

            # Spectral centroid
            spectral_centroid = np.sum(frequencies * power_spectrum) / np.sum(power_spectrum)
            results["spectral_centroid"] = np.array([spectral_centroid])

            # Spectral rolloff
            cumulative_power = np.cumsum(power_spectrum)
            rolloff_point = np.where(cumulative_power >= 0.85 * cumulative_power[-1])[0][0]
            results["spectral_rolloff"] = np.array([frequencies[rolloff_point]])

            # Phase information
            phase = np.angle(fft_values)
            results["phase"] = phase

            self.logger.info("✅ Fourier analysis completed")
            return results

        except Exception as e:
            self.logger.warning(f"⚠️ Fourier analysis failed: {e}")
            return {"original": data}

class TechnicalIndicatorExtractor:
    """
    Technical indicators for financial time series.

    Computes various technical indicators used in trading.
    """

    def __init__(self, config: PreprocessingConfig):
        """Initialize technical indicator extractor.

        Args:
            config: Preprocessing configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def compute_technical_indicators(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Compute technical indicators from OHLCV data.

        Args:
            data: OHLCV DataFrame

        Returns:
            Dictionary of technical indicators
        """
        if not self.config.use_technical_indicators:
            return {"original": data.values}

        try:
            results = {}

            close_prices = data['close'].values
            high_prices = data['high'].values
            low_prices = data['low'].values
            volume = data['volume'].values

            for indicator in self.config.technical_indicators:
                if indicator == "sma":
                    sma = self._simple_moving_average(close_prices, window=20)
                    results["sma_20"] = sma
                    sma = self._simple_moving_average(close_prices, window=50)
                    results["sma_50"] = sma

                elif indicator == "ema":
                    ema = self._exponential_moving_average(close_prices, span=12)
                    results["ema_12"] = ema
                    ema = self._exponential_moving_average(close_prices, span=26)
                    results["ema_26"] = ema

                elif indicator == "rsi":
                    rsi = self._relative_strength_index(close_prices, period=14)
                    results["rsi_14"] = rsi

                elif indicator == "macd":
                    macd, signal, hist = self._macd(close_prices)
                    results["macd"] = macd
                    results["macd_signal"] = signal
                    results["macd_histogram"] = hist

                elif indicator == "bollinger":
                    upper, middle, lower = self._bollinger_bands(close_prices)
                    results["bb_upper"] = upper
                    results["bb_middle"] = middle
                    results["bb_lower"] = lower
                    results["bb_width"] = upper - lower

            self.logger.info(f"✅ Computed {len(results)} technical indicators")
            return results

        except Exception as e:
            self.logger.warning(f"⚠️ Technical indicator computation failed: {e}")
            return {"original": data.values}

    def _simple_moving_average(self, data: np.ndarray, window: int) -> np.ndarray:
        """Compute simple moving average."""
        return pd.Series(data).rolling(window=window, min_periods=1).mean().values

    def _exponential_moving_average(self, data: np.ndarray, span: int) -> np.ndarray:
        """Compute exponential moving average."""
        return pd.Series(data).ewm(span=span, adjust=False).mean().values

    def _relative_strength_index(self, data: np.ndarray, period: int = 14) -> np.ndarray:
        """Compute RSI."""
        delta = np.diff(data)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = pd.Series(gain).rolling(window=period, min_periods=1).mean()
        avg_loss = pd.Series(loss).rolling(window=period, min_periods=1).mean()

        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))

        return rsi.values

    def _macd(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute MACD."""
        ema12 = self._exponential_moving_average(data, span=12)
        ema26 = self._exponential_moving_average(data, span=26)
        macd = ema12 - ema26

        signal = self._exponential_moving_average(macd, span=9)
        histogram = macd - signal

        return macd, signal, histogram

    def _bollinger_bands(self, data: np.ndarray, window: int = 20, num_std: float = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute Bollinger Bands."""
        sma = self._simple_moving_average(data, window)
        std = pd.Series(data).rolling(window=window, min_periods=1).std().values

        upper = sma + (std * num_std)
        lower = sma - (std * num_std)

        return upper, sma, lower

class SeasonalDecomposer:
    """
    Seasonal decomposition for time series.

    Separates time series into trend, seasonal, and residual components.
    """

    def __init__(self, config: PreprocessingConfig):
        """Initialize seasonal decomposer.

        Args:
            config: Preprocessing configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def decompose_series(self, data: pd.Series) -> Dict[str, np.ndarray]:
        """
        Decompose time series into components.

        Args:
            data: Time series data

        Returns:
            Dictionary of decomposed components
        """
        if not self.config.use_seasonal_decomposition:
            return {"original": data.values}

        try:
            from statsmodels.tsa.seasonal import seasonal_decompose

            # Perform seasonal decomposition
            decomposition = seasonal_decompose(
                data, period=self.config.decomposition_period, model='additive'
            )

            results = {
                "original": data.values,
                "trend": decomposition.trend.values,
                "seasonal": decomposition.seasonal.values,
                "residual": decomposition.resid.values
            }

            # Fill NaN values
            for key in results:
                if key != "original":
                    results[key] = self._fill_nan(results[key])

            self.logger.info("✅ Seasonal decomposition completed")
            return results

        except Exception as e:
            self.logger.warning(f"⚠️ Seasonal decomposition failed: {e}")
            return {"original": data.values}

    def _fill_nan(self, data: np.ndarray) -> np.ndarray:
        """Fill NaN values using interpolation."""
        series = pd.Series(data)
        filled = series.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
        return filled.values

class DataAugmenter:
    """
    Data augmentation for time series data.

    Applies various augmentation techniques to increase dataset diversity.
    """

    def __init__(self, config: PreprocessingConfig):
        """Initialize data augmenter.

        Args:
            config: Preprocessing configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def augment_time_series(self, data: np.ndarray) -> List[np.ndarray]:
        """
        Apply data augmentation to time series.

        Args:
            data: Original time series data

        Returns:
            List of augmented time series
        """
        if not self.config.use_data_augmentation:
            return [data]

        augmented_data = [data.copy()]  # Include original

        for method in self.config.augmentation_methods:
            try:
                if method == "noise":
                    augmented = self._add_noise(data)
                    augmented_data.append(augmented)

                elif method == "scaling":
                    augmented = self._scale_amplitude(data)
                    augmented_data.append(augmented)

                elif method == "time_warp":
                    augmented = self._time_warp(data)
                    augmented_data.append(augmented)

                elif method == "magnitude_warp":
                    augmented = self._magnitude_warp(data)
                    augmented_data.append(augmented)

                elif method == "jitter":
                    augmented = self._add_jitter(data)
                    augmented_data.append(augmented)

                elif method == "spawner":
                    augmented = self._spawner_augmentation(data)
                    augmented_data.extend(augmented)

            except Exception as e:
                self.logger.warning(f"⚠️ Augmentation method {method} failed: {e}")

        self.logger.info(f"✅ Generated {len(augmented_data)} augmented versions")
        return augmented_data

    def _add_noise(self, data: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
        """Add Gaussian noise to time series."""
        noise = np.random.normal(0, noise_level, data.shape)
        return data + noise

    def _scale_amplitude(self, data: np.ndarray, scale_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """Scale amplitude of time series."""
        scale_factor = np.random.uniform(scale_range[0], scale_range[1])
        return data * scale_factor

    def _time_warp(self, data: np.ndarray, sigma: float = 0.2) -> np.ndarray:
        """Apply time warping to time series."""
        from scipy.interpolate import interp1d

        time_steps = np.arange(len(data))
        warped_time = time_steps * (1 + np.random.normal(0, sigma, len(data)))
        warped_time = np.clip(warped_time, 0, len(data) - 1)

        f = interp1d(time_steps, data, kind='linear')
        warped_data = f(warped_time)

        return warped_data

    def _magnitude_warp(self, data: np.ndarray, sigma: float = 0.2) -> np.ndarray:
        """Apply magnitude warping to time series."""
        from scipy.interpolate import interp1d

        magnitude = np.random.normal(1, sigma, len(data))
        warped_data = data * magnitude

        return warped_data

    def _add_jitter(self, data: np.ndarray, jitter_level: float = 0.05) -> np.ndarray:
        """Add random jitter to time series."""
        jitter = np.random.normal(0, jitter_level, data.shape)
        return data + jitter

    def _spawner_augmentation(self, data: np.ndarray, num_samples: int = 3) -> List[np.ndarray]:
        """SPAWNER augmentation - generates new samples."""
        augmented_samples = []

        for _ in range(num_samples):
            # Random combination of augmentations
            augmented = data.copy()

            # Apply random augmentations
            if np.random.random() > 0.5:
                augmented = self._add_noise(augmented, 0.05)
            if np.random.random() > 0.5:
                augmented = self._scale_amplitude(augmented, (0.9, 1.1))
            if np.random.random() > 0.5:
                augmented = self._add_jitter(augmented, 0.02)

            augmented_samples.append(augmented)

        return augmented_samples

class NoiseReducer:
    """
    Noise reduction and filtering for time series.

    Applies various filtering techniques to reduce noise.
    """

    def __init__(self, config: PreprocessingConfig):
        """Initialize noise reducer.

        Args:
            config: Preprocessing configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def reduce_noise(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Apply noise reduction techniques.

        Args:
            data: Noisy time series data

        Returns:
            Dictionary of filtered data
        """
        if not self.config.use_noise_reduction:
            return {"original": data}

        try:
            results = {"original": data}

            if self.config.noise_filter_type == "butterworth":
                filtered = self._butterworth_filter(data)
                results["butterworth"] = filtered

            elif self.config.noise_filter_type == "savitzky_golay":
                filtered = self._savitzky_golay_filter(data)
                results["savitzky_golay"] = filtered

            elif self.config.noise_filter_type == "median":
                filtered = self._median_filter(data)
                results["median"] = filtered

            # Detrending
            detrended = detrend(data)
            results["detrended"] = detrended

            self.logger.info("✅ Noise reduction completed")
            return results

        except Exception as e:
            self.logger.warning(f"⚠️ Noise reduction failed: {e}")
            return {"original": data}

    def _butterworth_filter(self, data: np.ndarray, order: int = 4) -> np.ndarray:
        """Apply Butterworth filter."""
        b, a = butter(order, self.config.filter_cutoff, btype='low')
        return filtfilt(b, a, data)

    def _savitzky_golay_filter(self, data: np.ndarray, window: int = 15, polyorder: int = 2) -> np.ndarray:
        """Apply Savitzky-Golay filter."""
        return signal.savgol_filter(data, window, polyorder)

    def _median_filter(self, data: np.ndarray, window: int = 5) -> np.ndarray:
        """Apply median filter."""
        return signal.medfilt(data, kernel_size=window)

class AnomalyDetector:
    """
    Anomaly detection and handling for time series.

    Detects and handles outliers and anomalies in the data.
    """

    def __init__(self, config: PreprocessingConfig):
        """Initialize anomaly detector.

        Args:
            config: Preprocessing configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def detect_anomalies(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Detect anomalies in time series data.

        Args:
            data: Time series data

        Returns:
            Dictionary with anomaly information
        """
        if not self.config.use_anomaly_detection:
            return {"original": data, "anomalies": np.zeros(len(data))}

        try:
            # Z-score based anomaly detection
            mean_val = np.mean(data)
            std_val = np.std(data)
            z_scores = np.abs((data - mean_val) / std_val)

            # Detect anomalies
            anomalies = z_scores > self.config.anomaly_threshold

            # Handle anomalies
            cleaned_data = self._handle_anomalies(data, anomalies)

            results = {
                "original": data,
                "cleaned": cleaned_data,
                "anomalies": anomalies.astype(int),
                "z_scores": z_scores,
                "anomaly_ratio": np.mean(anomalies)
            }

            self.logger.info(f"✅ Detected {np.sum(anomalies)} anomalies ({np.mean(anomalies)*100:.2f}%)")
            return results

        except Exception as e:
            self.logger.warning(f"⚠️ Anomaly detection failed: {e}")
            return {"original": data, "anomalies": np.zeros(len(data))}

    def _handle_anomalies(self, data: np.ndarray, anomalies: np.ndarray) -> np.ndarray:
        """Handle detected anomalies."""
        cleaned_data = data.copy()

        # Replace anomalies with interpolated values
        anomaly_indices = np.where(anomalies)[0]

        for idx in anomaly_indices:
            # Use neighboring values for interpolation
            left_idx = max(0, idx - 1)
            right_idx = min(len(data) - 1, idx + 1)

            if left_idx < right_idx:
                # Linear interpolation
                cleaned_data[idx] = (data[left_idx] + data[right_idx]) / 2
            else:
                # Use nearest neighbor
                cleaned_data[idx] = data[left_idx]

        return cleaned_data

class AdvancedPreprocessor:
    """
    Advanced preprocessor combining all preprocessing techniques.

    Provides a unified interface for comprehensive time series preprocessing.
    """

    def __init__(self, config: PreprocessingConfig):
        """Initialize advanced preprocessor.

        Args:
            config: Preprocessing configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize components
        self.wavelet_transformer = WaveletTransformer(config)
        self.fourier_analyzer = FourierAnalyzer(config)
        self.technical_extractor = TechnicalIndicatorExtractor(config)
        self.seasonal_decomposer = SeasonalDecomposer(config)
        self.data_augmenter = DataAugmenter(config)
        self.noise_reducer = NoiseReducer(config)
        self.anomaly_detector = AnomalyDetector(config)

    def preprocess(self, data: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """
        Apply comprehensive preprocessing to time series data.

        Args:
            data: Input time series data

        Returns:
            Dictionary with all preprocessed features
        """
        logger.info("🚀 Starting comprehensive preprocessing")

        if isinstance(data, pd.DataFrame):
            df_data = data
            array_data = data.values
        else:
            df_data = pd.DataFrame(data)
            array_data = data

        results = {"original": array_data}

        # 1. Anomaly detection and cleaning
        if self.config.use_anomaly_detection:
            anomaly_results = self.anomaly_detector.detect_anomalies(array_data.flatten())
            results.update({f"anomaly_{k}": v for k, v in anomaly_results.items() if k != "original"})
            # Use cleaned data for further processing
            array_data = anomaly_results["cleaned"].reshape(-1, 1)

        # 2. Noise reduction
        if self.config.use_noise_reduction:
            noise_results = self.noise_reducer.reduce_noise(array_data.flatten())
            results.update({f"noise_{k}": v.reshape(-1, 1) for k, v in noise_results.items() if k != "original"})

        # 3. Technical indicators
        if self.config.use_technical_indicators and isinstance(df_data, pd.DataFrame):
            tech_results = self.technical_extractor.compute_technical_indicators(df_data)
            results.update({f"tech_{k}": v.reshape(-1, 1) for k, v in tech_results.items() if k != "original"})

        # 4. Wavelet decomposition
        if self.config.use_wavelet_transform:
            wavelet_results = self.wavelet_transformer.wavelet_decomposition(array_data.flatten())
            results.update({f"wavelet_{k}": v.reshape(-1, 1) for k, v in wavelet_results.items() if k != "original"})

        # 5. Fourier analysis
        if self.config.use_fourier_features:
            fourier_results = self.fourier_analyzer.extract_fourier_features(array_data.flatten())
            results.update({f"fourier_{k}": v.reshape(-1, 1) for k, v in fourier_results.items() if k != "original"})

        # 6. Seasonal decomposition
        if self.config.use_seasonal_decomposition and isinstance(df_data, pd.DataFrame):
            seasonal_results = self.seasonal_decomposer.decompose_series(df_data['close'])
            results.update({f"seasonal_{k}": v.reshape(-1, 1) for k, v in seasonal_results.items() if k != "original"})

        # 7. Data augmentation
        if self.config.use_data_augmentation:
            augmented_data = self.data_augmenter.augment_time_series(array_data)
            for i, aug_data in enumerate(augmented_data[1:], 1):  # Skip original
                results[f"augmented_{i}"] = aug_data.reshape(-1, 1)

        # 8. Normalization
        if self.config.use_normalization:
            normalized_data = self._normalize_data(results["original"])
            results["normalized"] = normalized_data

        self.logger.info(f"✅ Comprehensive preprocessing completed with {len(results)} feature sets")
        return results

    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Apply normalization to data."""
        if self.config.normalization_method == "standard":
            mean_val = np.mean(data, axis=0)
            std_val = np.std(data, axis=0)
            return (data - mean_val) / (std_val + 1e-8)

        elif self.config.normalization_method == "minmax":
            min_val = np.min(data, axis=0)
            max_val = np.max(data, axis=0)
            return (data - min_val) / (max_val - min_val + 1e-8)

        elif self.config.normalization_method == "robust":
            median_val = np.median(data, axis=0)
            mad_val = np.median(np.abs(data - median_val), axis=0)
            return (data - median_val) / (mad_val + 1e-8)

        else:
            return data  # No normalization

# Utility functions
def create_advanced_preprocessor(config: PreprocessingConfig) -> AdvancedPreprocessor:
    """Create advanced preprocessor with given configuration."""
    return AdvancedPreprocessor(config)

def preprocess_market_data(data: pd.DataFrame, config: PreprocessingConfig) -> Dict[str, Any]:
    """Preprocess market data with advanced techniques."""
    preprocessor = AdvancedPreprocessor(config)
    return preprocessor.preprocess(data)

def augment_time_series_data(data: np.ndarray, methods: List[str] = None) -> List[np.ndarray]:
    """Augment time series data with specified methods."""
    config = PreprocessingConfig(
        use_data_augmentation=True,
        augmentation_methods=methods or ["noise", "scaling", "time_warp"]
    )
    augmenter = DataAugmenter(config)
    return augmenter.augment_time_series(data)