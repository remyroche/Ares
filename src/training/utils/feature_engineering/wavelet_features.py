"""Wavelet transform features for financial time series.

This module provides wavelet-based feature extraction capabilities.
"""
from pathlib import Path
from typing import Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
import pywt
from src.core.decorators import handles_errors
from src.utils.logger import system_logger

class WaveletFeatureCache:
    """Cache for wavelet features to avoid recomputation."""

    def __init__(self, cache_dir: str='data_cache/wavelet_features') -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = system_logger.getChild('WaveletFeatureCache')
        self.memory_cache = {}

    def _get_cache_key(self, symbol: str, timeframe: str, wavelet: str, level: int, data_hash: str) -> str:
        """Generate cache key for wavelet features."""
        return f'{symbol}_{timeframe}_{wavelet}_L{level}_{data_hash}'

    def get(self, key: str) -> Optional[pd.DataFrame]:
        """Get cached wavelet features."""
        if key in self.memory_cache:
            return self.memory_cache[key]
        cache_file = self.cache_dir / f'{key}.parquet'
        if cache_file.exists():
            try:
                features = pd.read_parquet(cache_file)
                self.memory_cache[key] = features
                return features
            except Exception as e:
                self.logger.warning(f'Failed to load cache {key}: {e}')
        return None

    def set(self, key: str, features: pd.DataFrame) -> None:
        """Cache wavelet features."""
        self.memory_cache[key] = features
        cache_file = self.cache_dir / f'{key}.parquet'
        try:
            features.to_parquet(cache_file)
        except Exception as e:
            self.logger.warning(f'Failed to save cache {key}: {e}')

class WaveletTransformAnalyzer:
    """Analyzes price data using wavelet transforms."""

    def __init__(self, cache_enabled: bool=True) -> None:
        self.logger = system_logger.getChild('WaveletTransformAnalyzer')
        self.cache = WaveletFeatureCache() if cache_enabled else None
        self.wavelets = ['db4', 'sym4', 'coif2']
        self.levels = [3, 4, 5]

    @handles_errors(exceptions=(Exception,), default_return=pd.DataFrame(), context='wavelet transform')
    def extract_wavelet_features(self, data: pd.DataFrame, price_column: str='close', symbol: str='unknown', timeframe: str='1m') -> pd.DataFrame:
        """Extract wavelet-based features from price data.
        
        Args:
            data: DataFrame with price data
            price_column: Column to analyze
            symbol: Trading symbol for caching
            timeframe: Data timeframe for caching
            
        Returns:
            DataFrame with wavelet features
        """
        data_hash = joblib.hash(data[price_column].values)[:8]
        if self.cache:
            cache_key = self.cache._get_cache_key(symbol, timeframe, 'all', max(self.levels), data_hash)
            cached = self.cache.get(cache_key)
            if cached is not None:
                self.logger.info(f'Using cached wavelet features for {symbol}')
                return cached
        self.logger.info(f'Computing wavelet features for {symbol} ({timeframe})')
        features = pd.DataFrame(index=data.index)
        signal = data[price_column].values
        for wavelet in self.wavelets:
            for level in self.levels:
                coeffs = self._wavelet_decompose(signal, wavelet, level)
                wavelet_features = self._extract_coefficient_features(coeffs, data.index, wavelet, level)
                for name, values in wavelet_features.items():
                    features[name] = values
        features = self._add_wavelet_statistics(features, signal)
        if self.cache:
            self.cache.set(cache_key, features)
        return features

    def _wavelet_decompose(self, signal: np.ndarray, wavelet: str, level: int) -> List[np.ndarray]:
        """Perform wavelet decomposition."""
        pad_len = 2 ** level
        padded_signal = np.pad(signal, pad_len, mode='edge')
        coeffs = pywt.wavedec(padded_signal, wavelet, level=level)
        coeffs[0] = self._remove_padding(coeffs[0], len(signal), pad_len)
        return coeffs

    def _remove_padding(self, coeff: np.ndarray, original_len: int, pad_len: int) -> np.ndarray:
        """Remove padding from wavelet coefficients."""
        expected_len = original_len
        if len(coeff) > expected_len:
            excess = len(coeff) - expected_len
            start = excess // 2
            end = start + expected_len
            return coeff[start:end]
        return coeff

    def _extract_coefficient_features(self, coeffs: List[np.ndarray], index: pd.DatetimeIndex, wavelet: str, level: int) -> Dict[str, np.ndarray]:
        """Extract features from wavelet coefficients."""
        features = {}
        approx = coeffs[0]
        approx_interp = np.interp(np.linspace(0, len(approx) - 1, len(index)), np.arange(len(approx)), approx)
        features[f'{wavelet}_L{level}_trend'] = approx_interp
        for i, detail in enumerate(coeffs[1:], 1):
            detail_interp = np.interp(np.linspace(0, len(detail) - 1, len(index)), np.arange(len(detail)), detail)
            features[f'{wavelet}_L{level}_D{i}'] = detail_interp
            energy = np.convolve(detail_interp ** 2, np.ones(20) / 20, mode='same')
            features[f'{wavelet}_L{level}_D{i}_energy'] = energy
        return features

    def _add_wavelet_statistics(self, features: pd.DataFrame, signal: np.ndarray) -> pd.DataFrame:
        """Add statistical features derived from wavelets."""
        for wavelet in self.wavelets:
            detail_cols = [col for col in features.columns if wavelet in col and '_D' in col and ('_energy' not in col)]
            if len(detail_cols) >= 2:
                for i in range(len(detail_cols) - 1):
                    corr = features[detail_cols[i]].rolling(50).corr(features[detail_cols[i + 1]])
                    features[f'{wavelet}_scale_coherence_{i}_{i + 1}'] = corr
        for wavelet in self.wavelets:
            trend_col = f'{wavelet}_L{max(self.levels)}_trend'
            if trend_col in features:
                noise = signal - features[trend_col].values
                snr = np.abs(features[trend_col]) / (np.abs(noise) + 1e-10)
                features[f'{wavelet}_snr'] = snr
        return features

    def get_multi_scale_trends(self, data: pd.DataFrame, price_column: str='close') -> Dict[str, pd.Series]:
        """Extract trends at multiple scales using wavelets.
        
        Args:
            data: DataFrame with price data
            price_column: Column to analyze
            
        Returns:
            Dictionary mapping scale to trend series
        """
        signal = data[price_column].values
        trends = {}
        for level in self.levels:
            coeffs = pywt.wavedec(signal, 'db4', level=level)
            trend = pywt.waverec([coeffs[0]] + [None] * (len(coeffs) - 1), 'db4')
            if len(trend) > len(signal):
                trend = trend[:len(signal)]
            elif len(trend) < len(signal):
                trend = np.pad(trend, (0, len(signal) - len(trend)), mode='edge')
            trends[f'scale_{level}'] = pd.Series(trend, index=data.index)
        return trends