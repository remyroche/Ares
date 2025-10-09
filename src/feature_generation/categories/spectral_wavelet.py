"""
Spectral/Wavelet Feature Generator

This module provides feature generators for spectral and wavelet analysis,
including wavelet energy, cycle detection, fractal dimension, and other
frequency-domain features for quantitative finance.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union

from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

# Optimization utilities
try:
    from ..utils.vectorization_optimizer import get_vectorization_optimizer
    from ..utils.optimized_feature_pipeline import get_optimized_feature_pipeline
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

class WaveletEnergyGenerator(VectorizedFeatureGenerator):
    """Generator for wavelet energy by scale (MODWT levels)."""
    
    def __init__(self, window: int = 20, levels: int = 5):
        config = FeatureConfig(
            name=f"wavelet_energy_{window}_{levels}",
            category=FeatureCategory.SPECTRAL_WAVELET,
            description=f"Wavelet energy by scale over {window} periods (levels {levels})",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'levels': levels},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.levels = levels
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        close = data['close'].values
        if len(close) < self.window:
            return pd.Series(np.full(len(close), np.nan), index=data.index)
        
        # Calculate wavelet energy
        wavelet_energy = np.full(len(close), np.nan)
        for i in range(self.window - 1, len(close)):
            window_prices = close[i - self.window + 1:i + 1]
            energy = self._calculate_wavelet_energy(window_prices, self.levels)
            wavelet_energy[i] = energy
        
        return pd.Series(wavelet_energy, index=data.index)
    
    def _calculate_wavelet_energy(self, prices: np.ndarray, levels: int) -> float:
        """Calculate wavelet energy using simplified DWT."""
        try:
            # Simplified wavelet transform (Haar wavelet)
            def haar_transform(data):
                if len(data) <= 1:
                    return data, []
                
                # Downsample and calculate differences
                even = data[::2]
                odd = data[1::2]
                
                # Approximation and detail coefficients
                approx = (even + odd) / np.sqrt(2)
                detail = (even - odd) / np.sqrt(2)
                
                return approx, detail
            
            # Apply wavelet transform
            current_data = prices.copy()
            total_energy = 0
            
            for level in range(min(levels, int(np.log2(len(prices))))):
                if len(current_data) <= 1:
                    break
                
                approx, detail = haar_transform(current_data)
                # Energy is sum of squared detail coefficients
                level_energy = np.sum(detail ** 2)
                total_energy += level_energy
                
                current_data = approx
            
            return total_energy
        except:
            return 0.0

class BandLimitedVolatilityGenerator(VectorizedFeatureGenerator):
    """Generator for band-limited volatility (power in low vs high frequency bands)."""
    
    def __init__(self, window: int = 20, low_freq_cutoff: float = 0.1):
        config = FeatureConfig(
            name=f"band_limited_volatility_{window}_{low_freq_cutoff}",
            category=FeatureCategory.SPECTRAL_WAVELET,
            description=f"Band-limited volatility over {window} periods (low freq cutoff {low_freq_cutoff})",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'low_freq_cutoff': low_freq_cutoff},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.low_freq_cutoff = low_freq_cutoff
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        close = data['close'].values
        if len(close) < self.window + 1:
            return pd.Series(np.full(len(close), np.nan), index=data.index)
        
        # Calculate returns
        returns = np.diff(close) / close[:-1]
        returns = np.concatenate([[np.nan], returns])
        
        # Calculate band-limited volatility
        band_volatility = np.full(len(close), np.nan)
        for i in range(self.window, len(close)):
            window_returns = returns[i - self.window + 1:i + 1]
            valid_returns = window_returns[np.isfinite(window_returns)]
            
            if len(valid_returns) > 4:  # Need enough data for FFT
                vol = self._calculate_band_limited_volatility(valid_returns, self.low_freq_cutoff)
                band_volatility[i] = vol
        
        return pd.Series(band_volatility, index=data.index)
    
    def _calculate_band_limited_volatility(self, returns: np.ndarray, cutoff: float) -> float:
        """Calculate band-limited volatility using FFT."""
        try:
            # Calculate FFT
            fft = np.fft.fft(returns)
            freqs = np.fft.fftfreq(len(returns))
            
            # Separate low and high frequency components
            low_freq_mask = np.abs(freqs) <= cutoff
            high_freq_mask = np.abs(freqs) > cutoff
            
            # Calculate power in each band
            low_freq_power = np.sum(np.abs(fft[low_freq_mask]) ** 2)
            high_freq_power = np.sum(np.abs(fft[high_freq_mask]) ** 2)
            
            # Return ratio of low to high frequency power
            if high_freq_power > 0:
                return low_freq_power / high_freq_power
            else:
                return low_freq_power
        except:
            return 0.0

class CycleLengthGenerator(VectorizedFeatureGenerator):
    """Generator for cycle length estimates using Lomb-Scargle periodogram."""
    
    def __init__(self, window: int = 20, min_period: int = 2, max_period: int = 10):
        config = FeatureConfig(
            name=f"cycle_length_{window}_{min_period}_{max_period}",
            category=FeatureCategory.SPECTRAL_WAVELET,
            description=f"Cycle length estimates over {window} periods (period range {min_period}-{max_period})",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'min_period': min_period, 'max_period': max_period},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.min_period = min_period
        self.max_period = max_period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        close = data['close'].values
        if len(close) < self.window:
            return pd.Series(np.full(len(close), np.nan), index=data.index)
        
        # Calculate cycle length
        cycle_length = np.full(len(close), np.nan)
        for i in range(self.window - 1, len(close)):
            window_prices = close[i - self.window + 1:i + 1]
            length = self._calculate_cycle_length(window_prices, self.min_period, self.max_period)
            cycle_length[i] = length
        
        return pd.Series(cycle_length, index=data.index)
    
    def _calculate_cycle_length(self, prices: np.ndarray, min_period: int, max_period: int) -> float:
        """Calculate dominant cycle length using simplified periodogram."""
        try:
            # Detrend the data
            x = np.arange(len(prices))
            trend = np.polyfit(x, prices, 1)
            detrended = prices - np.polyval(trend, x)
            
            # Calculate periodogram
            best_period = min_period
            best_power = 0
            
            for period in range(min_period, min(max_period + 1, len(prices) // 2)):
                # Calculate power for this period
                power = self._calculate_period_power(detrended, period)
                
                if power > best_power:
                    best_power = power
                    best_period = period
            
            return float(best_period)
        except:
            return float(min_period)
    
    def _calculate_period_power(self, data: np.ndarray, period: int) -> float:
        """Calculate power for a specific period."""
        try:
            n = len(data)
            if period >= n:
                return 0.0
            
            # Calculate autocorrelation at this lag
            autocorr = np.corrcoef(data[:-period], data[period:])[0, 1]
            
            # Return squared correlation as power
            return autocorr ** 2 if not np.isnan(autocorr) else 0.0
        except:
            return 0.0

class FractalDimensionGenerator(VectorizedFeatureGenerator):
    """Generator for fractal dimension using Katz method."""
    
    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name=f"fractal_dimension_{window}",
            category=FeatureCategory.SPECTRAL_WAVELET,
            description=f"Fractal dimension using Katz method over {window} periods",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        close = data['close'].values
        if len(close) < self.window:
            return pd.Series(np.full(len(close), np.nan), index=data.index)
        
        # Calculate fractal dimension
        fractal_dim = np.full(len(close), np.nan)
        for i in range(self.window - 1, len(close)):
            window_prices = close[i - self.window + 1:i + 1]
            dim = self._calculate_fractal_dimension(window_prices)
            fractal_dim[i] = dim
        
        return pd.Series(fractal_dim, index=data.index)
    
    def _calculate_fractal_dimension(self, prices: np.ndarray) -> float:
        """Calculate fractal dimension using Katz method."""
        try:
            n = len(prices)
            if n < 3:
                return 1.0
            
            # Calculate distances between consecutive points
            distances = np.sqrt(1 + np.diff(prices) ** 2)
            
            # Total length of the curve
            L = np.sum(distances)
            
            # Maximum distance from first point
            d = np.max(np.abs(prices - prices[0]))
            
            if d == 0 or L == 0:
                return 1.0
            
            # Katz fractal dimension
            fractal_dim = np.log(n - 1) / (np.log(L / d) + np.log(n - 1))
            
            return np.clip(fractal_dim, 1.0, 2.0)
        except:
            return 1.0

class DFASlopesGenerator(VectorizedFeatureGenerator):
    """Generator for DFA (Detrended Fluctuation Analysis) slopes."""
    
    def __init__(self, window: int = 20, min_box_size: int = 4, max_box_size: int = 10):
        config = FeatureConfig(
            name=f"dfa_slopes_{window}_{min_box_size}_{max_box_size}",
            category=FeatureCategory.SPECTRAL_WAVELET,
            description=f"DFA slopes over {window} periods (box size {min_box_size}-{max_box_size})",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'min_box_size': min_box_size, 'max_box_size': max_box_size},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.min_box_size = min_box_size
        self.max_box_size = max_box_size
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        close = data['close'].values
        if len(close) < self.window + 1:
            return pd.Series(np.full(len(close), np.nan), index=data.index)
        
        # Calculate returns
        returns = np.diff(close) / close[:-1]
        returns = np.concatenate([[np.nan], returns])
        
        # Calculate DFA slopes
        dfa_slopes = np.full(len(close), np.nan)
        for i in range(self.window, len(close)):
            window_returns = returns[i - self.window + 1:i + 1]
            valid_returns = window_returns[np.isfinite(window_returns)]
            
            if len(valid_returns) > self.max_box_size:
                slope = self._calculate_dfa_slope(valid_returns, self.min_box_size, self.max_box_size)
                dfa_slopes[i] = slope
        
        return pd.Series(dfa_slopes, index=data.index)
    
    def _calculate_dfa_slope(self, returns: np.ndarray, min_box_size: int, max_box_size: int) -> float:
        """Calculate DFA slope."""
        try:
            n = len(returns)
            if n < max_box_size:
                return 0.5
            
            # Calculate cumulative sum (profile)
            profile = np.cumsum(returns - np.mean(returns))
            
            # Calculate fluctuation for different box sizes
            box_sizes = np.arange(min_box_size, min(max_box_size + 1, n // 2))
            fluctuations = []
            
            for box_size in box_sizes:
                # Divide profile into boxes
                n_boxes = n // box_size
                if n_boxes < 2:
                    continue
                
                # Calculate fluctuation for this box size
                fluctuation = 0
                for i in range(n_boxes):
                    start = i * box_size
                    end = start + box_size
                    box_data = profile[start:end]
                    
                    # Detrend the box
                    x = np.arange(len(box_data))
                    trend = np.polyfit(x, box_data, 1)
                    detrended = box_data - np.polyval(trend, x)
                    
                    # Add to fluctuation
                    fluctuation += np.mean(detrended ** 2)
                
                fluctuations.append(fluctuation / n_boxes)
            
            if len(fluctuations) < 2:
                return 0.5
            
            # Calculate slope using linear regression
            log_box_sizes = np.log(box_sizes[:len(fluctuations)])
            log_fluctuations = np.log(fluctuations)
            
            # Linear regression
            slope = np.polyfit(log_box_sizes, log_fluctuations, 1)[0]
            
            return slope
        except:
            return 0.5

def create_default_spectral_wavelet_generators() -> List[FeatureGenerator]:
    """Create default spectral/wavelet feature generators."""
    generators = []
    
    # Wavelet energy generators
    for window in [20, 50]:
        for levels in [3, 5]:
            generators.append(WaveletEnergyGenerator(window, levels))
    
    # Band-limited volatility generators
    for window in [20]:
        for cutoff in [0.1, 0.2]:
            generators.append(BandLimitedVolatilityGenerator(window, cutoff))
    
    # Cycle length generators
    for window in [20, 50]:
        generators.append(CycleLengthGenerator(window, 2, 10))
    
    # Fractal dimension generators
    for window in [20, 50]:
        generators.append(FractalDimensionGenerator(window))
    
    # DFA slopes generators
    for window in [20, 50]:
        generators.append(DFASlopesGenerator(window, 4, 10))
    
    return generators

# Export all generators
__all__ = [
    'WaveletEnergyGenerator',
    'BandLimitedVolatilityGenerator',
    'CycleLengthGenerator',
    'FractalDimensionGenerator',
    'DFASlopesGenerator',
    'create_default_spectral_wavelet_generators'
]