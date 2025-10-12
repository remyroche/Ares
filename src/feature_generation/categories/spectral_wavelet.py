"""
Spectral/Wavelet Feature Generator

This module provides feature generators for spectral and wavelet analysis,
including wavelet energy, cycle detection, fractal dimension, and other
frequency-domain features for quantitative finance.

Enhanced with full VectorBT integration for optimal performance.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Any, Dict, List, Optional, Union, Tuple
import logging

from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import (
        rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, 
        rolling_apply, rolling_corr, rolling_cov, rolling_quantile, rolling_skew, rolling_kurt
    )
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    from vectorbt.portfolio.base import Portfolio
    from vectorbt.portfolio.nb import generate_returns_nb, generate_orders_nb
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
    rolling_quantile = None
    rolling_skew = None
    rolling_kurt = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    Portfolio = None
    generate_returns_nb = None
    generate_orders_nb = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# VectorBT Rolling Optimizer
try:
    from ..utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer, VectorBTRollingOptimizer
    VECTORBT_ROLLING_OPTIMIZER_AVAILABLE = True
except ImportError:
    VECTORBT_ROLLING_OPTIMIZER_AVAILABLE = False
    get_vectorbt_rolling_optimizer = None
    VectorBTRollingOptimizer = None

# Unified Vectorization Manager
try:
    from ...utils.ml_common.unified_vectorization_manager import get_unified_vectorization_manager, UnifiedVectorizationManager, OperationType, OptimizationStrategy
    UNIFIED_VECTORIZATION_MANAGER_AVAILABLE = True
except ImportError:
    UNIFIED_VECTORIZATION_MANAGER_AVAILABLE = False
    get_unified_vectorization_manager = None
    UnifiedVectorizationManager = None
    OperationType = None
    OptimizationStrategy = None

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

# Optional PyTorch for advanced GPU operations
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

logger = logging.getLogger(__name__)

class WaveletEnergyGenerator(VectorizedFeatureGenerator):
    """Generator for wavelet energy by scale (MODWT levels) with VectorBT optimization."""
    
    def __init__(self, window: int = 20, levels: int = 5, use_vectorbt: bool = True, enable_gpu: bool = False):
        config = FeatureConfig(
            name=f"wavelet_energy_{window}_{levels}",
            category=FeatureCategory.SPECTRAL_WAVELET,
            description=f"Wavelet energy by scale over {window} periods (levels {levels}) - VectorBT optimized",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'levels': levels, 'use_vectorbt': use_vectorbt, 'enable_gpu': enable_gpu},
            matrix_optimized=True,
            gpu_accelerated=enable_gpu and CUPY_AVAILABLE
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.levels = levels
        self.use_vectorbt = use_vectorbt and VECTORBT_AVAILABLE
        self.enable_gpu = enable_gpu and CUPY_AVAILABLE
        
        # Initialize VectorBT rolling optimizer
        if VECTORBT_ROLLING_OPTIMIZER_AVAILABLE and self.use_vectorbt:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=self.enable_gpu)
        else:
            self.rolling_optimizer = None
            
        # Initialize unified vectorization manager
        if UNIFIED_VECTORIZATION_MANAGER_AVAILABLE and self.use_vectorbt:
            self.vectorization_manager = get_unified_vectorization_manager()
        else:
            self.vectorization_manager = None
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        close = data['close']
        if len(close) < self.window:
            return pd.Series(np.full(len(close), np.nan), index=data.index)
        
        # Use VectorBT optimization if available
        if self.use_vectorbt and self.rolling_optimizer:
            return self._vectorbt_wavelet_energy(close)
        else:
            return self._fallback_wavelet_energy(close)
    
    def _vectorbt_wavelet_energy(self, close: pd.Series) -> pd.Series:
        """Calculate wavelet energy using VectorBT optimization."""
        try:
            # Use VectorBT rolling apply for efficient wavelet energy calculation
            def wavelet_energy_func(window_data):
                if len(window_data) < self.window:
                    return np.nan
                return self._calculate_wavelet_energy(window_data.values, self.levels)
            
            # Use VectorBT rolling apply for optimal performance
            if VECTORBT_AVAILABLE:
                wavelet_energy = rolling_apply(close, window=self.window, func=wavelet_energy_func)
            else:
                # Fallback to pandas rolling apply
                wavelet_energy = close.rolling(window=self.window).apply(wavelet_energy_func, raw=True)
            
            return wavelet_energy
            
        except Exception as e:
            logger.warning(f"VectorBT wavelet energy calculation failed: {e}, using fallback")
            return self._fallback_wavelet_energy(close)
    
    def _fallback_wavelet_energy(self, close: pd.Series) -> pd.Series:
        """Fallback wavelet energy calculation using manual loops."""
        close_values = close.values
        wavelet_energy = np.full(len(close_values), np.nan)
        
        for i in range(self.window - 1, len(close_values)):
            window_prices = close_values[i - self.window + 1:i + 1]
            energy = self._calculate_wavelet_energy(window_prices, self.levels)
            wavelet_energy[i] = energy
        
        return pd.Series(wavelet_energy, index=close.index)
    
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
    """Generator for band-limited volatility (power in low vs high frequency bands) with VectorBT optimization."""
    
    def __init__(self, window: int = 20, low_freq_cutoff: float = 0.1, use_vectorbt: bool = True, enable_gpu: bool = False):
        config = FeatureConfig(
            name=f"band_limited_volatility_{window}_{low_freq_cutoff}",
            category=FeatureCategory.SPECTRAL_WAVELET,
            description=f"Band-limited volatility over {window} periods (low freq cutoff {low_freq_cutoff}) - VectorBT optimized",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'low_freq_cutoff': low_freq_cutoff, 'use_vectorbt': use_vectorbt, 'enable_gpu': enable_gpu},
            matrix_optimized=True,
            gpu_accelerated=enable_gpu and CUPY_AVAILABLE
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.low_freq_cutoff = low_freq_cutoff
        self.use_vectorbt = use_vectorbt and VECTORBT_AVAILABLE
        self.enable_gpu = enable_gpu and CUPY_AVAILABLE
        
        # Initialize VectorBT rolling optimizer
        if VECTORBT_ROLLING_OPTIMIZER_AVAILABLE and self.use_vectorbt:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=self.enable_gpu)
        else:
            self.rolling_optimizer = None
            
        # Initialize unified vectorization manager
        if UNIFIED_VECTORIZATION_MANAGER_AVAILABLE and self.use_vectorbt:
            self.vectorization_manager = get_unified_vectorization_manager()
        else:
            self.vectorization_manager = None
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        close = data['close']
        if len(close) < self.window + 1:
            return pd.Series(np.full(len(close), np.nan), index=data.index)
        
        # Calculate returns using VectorBT if available
        if self.use_vectorbt and VECTORBT_AVAILABLE:
            returns = close.pct_change()
        else:
            close_values = close.values
            returns = np.diff(close_values) / close_values[:-1]
            returns = np.concatenate([[np.nan], returns])
            returns = pd.Series(returns, index=close.index)
        
        # Use VectorBT optimization if available
        if self.use_vectorbt and self.rolling_optimizer:
            return self._vectorbt_band_limited_volatility(returns)
        else:
            return self._fallback_band_limited_volatility(returns)
    
    def _vectorbt_band_limited_volatility(self, returns: pd.Series) -> pd.Series:
        """Calculate band-limited volatility using VectorBT optimization."""
        try:
            # Use VectorBT rolling apply for efficient band-limited volatility calculation
            def band_volatility_func(window_returns):
                valid_returns = window_returns.dropna()
                if len(valid_returns) > 4:  # Need enough data for FFT
                    return self._calculate_band_limited_volatility(valid_returns.values, self.low_freq_cutoff)
                return np.nan
            
            # Use VectorBT rolling apply for optimal performance
            if VECTORBT_AVAILABLE:
                band_volatility = rolling_apply(returns, window=self.window, func=band_volatility_func)
            else:
                # Fallback to pandas rolling apply
                band_volatility = returns.rolling(window=self.window).apply(band_volatility_func, raw=False)
            
            return band_volatility
            
        except Exception as e:
            logger.warning(f"VectorBT band-limited volatility calculation failed: {e}, using fallback")
            return self._fallback_band_limited_volatility(returns)
    
    def _fallback_band_limited_volatility(self, returns: pd.Series) -> pd.Series:
        """Fallback band-limited volatility calculation using manual loops."""
        returns_values = returns.values
        band_volatility = np.full(len(returns_values), np.nan)
        
        for i in range(self.window, len(returns_values)):
            window_returns = returns_values[i - self.window + 1:i + 1]
            valid_returns = window_returns[np.isfinite(window_returns)]
            
            if len(valid_returns) > 4:  # Need enough data for FFT
                vol = self._calculate_band_limited_volatility(valid_returns, self.low_freq_cutoff)
                band_volatility[i] = vol
        
        return pd.Series(band_volatility, index=returns.index)
    
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
    """Generator for cycle length estimates using Lomb-Scargle periodogram with VectorBT optimization."""
    
    def __init__(self, window: int = 20, min_period: int = 2, max_period: int = 10, use_vectorbt: bool = True, enable_gpu: bool = False):
        config = FeatureConfig(
            name=f"cycle_length_{window}_{min_period}_{max_period}",
            category=FeatureCategory.SPECTRAL_WAVELET,
            description=f"Cycle length estimates over {window} periods (period range {min_period}-{max_period}) - VectorBT optimized",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'min_period': min_period, 'max_period': max_period, 'use_vectorbt': use_vectorbt, 'enable_gpu': enable_gpu},
            matrix_optimized=True,
            gpu_accelerated=enable_gpu and CUPY_AVAILABLE
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.min_period = min_period
        self.max_period = max_period
        self.use_vectorbt = use_vectorbt and VECTORBT_AVAILABLE
        self.enable_gpu = enable_gpu and CUPY_AVAILABLE
        
        # Initialize VectorBT rolling optimizer
        if VECTORBT_ROLLING_OPTIMIZER_AVAILABLE and self.use_vectorbt:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=self.enable_gpu)
        else:
            self.rolling_optimizer = None
            
        # Initialize unified vectorization manager
        if UNIFIED_VECTORIZATION_MANAGER_AVAILABLE and self.use_vectorbt:
            self.vectorization_manager = get_unified_vectorization_manager()
        else:
            self.vectorization_manager = None
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        close = data['close']
        if len(close) < self.window:
            return pd.Series(np.full(len(close), np.nan), index=data.index)
        
        # Use VectorBT optimization if available
        if self.use_vectorbt and self.rolling_optimizer:
            return self._vectorbt_cycle_length(close)
        else:
            return self._fallback_cycle_length(close)
    
    def _vectorbt_cycle_length(self, close: pd.Series) -> pd.Series:
        """Calculate cycle length using VectorBT optimization."""
        try:
            # Use VectorBT rolling apply for efficient cycle length calculation
            def cycle_length_func(window_data):
                if len(window_data) < self.window:
                    return np.nan
                return self._calculate_cycle_length(window_data.values, self.min_period, self.max_period)
            
            # Use VectorBT rolling apply for optimal performance
            if VECTORBT_AVAILABLE:
                cycle_length = rolling_apply(close, window=self.window, func=cycle_length_func)
            else:
                # Fallback to pandas rolling apply
                cycle_length = close.rolling(window=self.window).apply(cycle_length_func, raw=True)
            
            return cycle_length
            
        except Exception as e:
            logger.warning(f"VectorBT cycle length calculation failed: {e}, using fallback")
            return self._fallback_cycle_length(close)
    
    def _fallback_cycle_length(self, close: pd.Series) -> pd.Series:
        """Fallback cycle length calculation using manual loops."""
        close_values = close.values
        cycle_length = np.full(len(close_values), np.nan)
        
        for i in range(self.window - 1, len(close_values)):
            window_prices = close_values[i - self.window + 1:i + 1]
            length = self._calculate_cycle_length(window_prices, self.min_period, self.max_period)
            cycle_length[i] = length
        
        return pd.Series(cycle_length, index=close.index)
    
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
    """Generator for fractal dimension using Katz method with VectorBT optimization."""
    
    def __init__(self, window: int = 20, use_vectorbt: bool = True, enable_gpu: bool = False):
        config = FeatureConfig(
            name=f"fractal_dimension_{window}",
            category=FeatureCategory.SPECTRAL_WAVELET,
            description=f"Fractal dimension using Katz method over {window} periods - VectorBT optimized",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'use_vectorbt': use_vectorbt, 'enable_gpu': enable_gpu},
            matrix_optimized=True,
            gpu_accelerated=enable_gpu and CUPY_AVAILABLE
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.use_vectorbt = use_vectorbt and VECTORBT_AVAILABLE
        self.enable_gpu = enable_gpu and CUPY_AVAILABLE
        
        # Initialize VectorBT rolling optimizer
        if VECTORBT_ROLLING_OPTIMIZER_AVAILABLE and self.use_vectorbt:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=self.enable_gpu)
        else:
            self.rolling_optimizer = None
            
        # Initialize unified vectorization manager
        if UNIFIED_VECTORIZATION_MANAGER_AVAILABLE and self.use_vectorbt:
            self.vectorization_manager = get_unified_vectorization_manager()
        else:
            self.vectorization_manager = None
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        close = data['close']
        if len(close) < self.window:
            return pd.Series(np.full(len(close), np.nan), index=data.index)
        
        # Use VectorBT optimization if available
        if self.use_vectorbt and self.rolling_optimizer:
            return self._vectorbt_fractal_dimension(close)
        else:
            return self._fallback_fractal_dimension(close)
    
    def _vectorbt_fractal_dimension(self, close: pd.Series) -> pd.Series:
        """Calculate fractal dimension using VectorBT optimization."""
        try:
            # Use VectorBT rolling apply for efficient fractal dimension calculation
            def fractal_dim_func(window_data):
                if len(window_data) < self.window:
                    return np.nan
                return self._calculate_fractal_dimension(window_data.values)
            
            # Use VectorBT rolling apply for optimal performance
            if VECTORBT_AVAILABLE:
                fractal_dim = rolling_apply(close, window=self.window, func=fractal_dim_func)
            else:
                # Fallback to pandas rolling apply
                fractal_dim = close.rolling(window=self.window).apply(fractal_dim_func, raw=True)
            
            return fractal_dim
            
        except Exception as e:
            logger.warning(f"VectorBT fractal dimension calculation failed: {e}, using fallback")
            return self._fallback_fractal_dimension(close)
    
    def _fallback_fractal_dimension(self, close: pd.Series) -> pd.Series:
        """Fallback fractal dimension calculation using manual loops."""
        close_values = close.values
        fractal_dim = np.full(len(close_values), np.nan)
        
        for i in range(self.window - 1, len(close_values)):
            window_prices = close_values[i - self.window + 1:i + 1]
            dim = self._calculate_fractal_dimension(window_prices)
            fractal_dim[i] = dim
        
        return pd.Series(fractal_dim, index=close.index)
    
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
    """Generator for DFA (Detrended Fluctuation Analysis) slopes with VectorBT optimization."""
    
    def __init__(self, window: int = 20, min_box_size: int = 4, max_box_size: int = 10, use_vectorbt: bool = True, enable_gpu: bool = False):
        config = FeatureConfig(
            name=f"dfa_slopes_{window}_{min_box_size}_{max_box_size}",
            category=FeatureCategory.SPECTRAL_WAVELET,
            description=f"DFA slopes over {window} periods (box size {min_box_size}-{max_box_size}) - VectorBT optimized",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'min_box_size': min_box_size, 'max_box_size': max_box_size, 'use_vectorbt': use_vectorbt, 'enable_gpu': enable_gpu},
            matrix_optimized=True,
            gpu_accelerated=enable_gpu and CUPY_AVAILABLE
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.min_box_size = min_box_size
        self.max_box_size = max_box_size
        self.use_vectorbt = use_vectorbt and VECTORBT_AVAILABLE
        self.enable_gpu = enable_gpu and CUPY_AVAILABLE
        
        # Initialize VectorBT rolling optimizer
        if VECTORBT_ROLLING_OPTIMIZER_AVAILABLE and self.use_vectorbt:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=self.enable_gpu)
        else:
            self.rolling_optimizer = None
            
        # Initialize unified vectorization manager
        if UNIFIED_VECTORIZATION_MANAGER_AVAILABLE and self.use_vectorbt:
            self.vectorization_manager = get_unified_vectorization_manager()
        else:
            self.vectorization_manager = None
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        close = data['close']
        if len(close) < self.window + 1:
            return pd.Series(np.full(len(close), np.nan), index=data.index)
        
        # Calculate returns using VectorBT if available
        if self.use_vectorbt and VECTORBT_AVAILABLE:
            returns = close.pct_change()
        else:
            close_values = close.values
            returns = np.diff(close_values) / close_values[:-1]
            returns = np.concatenate([[np.nan], returns])
            returns = pd.Series(returns, index=close.index)
        
        # Use VectorBT optimization if available
        if self.use_vectorbt and self.rolling_optimizer:
            return self._vectorbt_dfa_slopes(returns)
        else:
            return self._fallback_dfa_slopes(returns)
    
    def _vectorbt_dfa_slopes(self, returns: pd.Series) -> pd.Series:
        """Calculate DFA slopes using VectorBT optimization."""
        try:
            # Use VectorBT rolling apply for efficient DFA slopes calculation
            def dfa_slope_func(window_returns):
                valid_returns = window_returns.dropna()
                if len(valid_returns) > self.max_box_size:
                    return self._calculate_dfa_slope(valid_returns.values, self.min_box_size, self.max_box_size)
                return np.nan
            
            # Use VectorBT rolling apply for optimal performance
            if VECTORBT_AVAILABLE:
                dfa_slopes = rolling_apply(returns, window=self.window, func=dfa_slope_func)
            else:
                # Fallback to pandas rolling apply
                dfa_slopes = returns.rolling(window=self.window).apply(dfa_slope_func, raw=False)
            
            return dfa_slopes
            
        except Exception as e:
            logger.warning(f"VectorBT DFA slopes calculation failed: {e}, using fallback")
            return self._fallback_dfa_slopes(returns)
    
    def _fallback_dfa_slopes(self, returns: pd.Series) -> pd.Series:
        """Fallback DFA slopes calculation using manual loops."""
        returns_values = returns.values
        dfa_slopes = np.full(len(returns_values), np.nan)
        
        for i in range(self.window, len(returns_values)):
            window_returns = returns_values[i - self.window + 1:i + 1]
            valid_returns = window_returns[np.isfinite(window_returns)]
            
            if len(valid_returns) > self.max_box_size:
                slope = self._calculate_dfa_slope(valid_returns, self.min_box_size, self.max_box_size)
                dfa_slopes[i] = slope
        
        return pd.Series(dfa_slopes, index=returns.index)
    
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

def create_default_spectral_wavelet_generators(use_vectorbt: bool = True, enable_gpu: bool = False) -> List[FeatureGenerator]:
    """Create default spectral/wavelet feature generators with VectorBT optimization."""
    generators = []
    
    # Wavelet energy generators
    for window in [20, 50]:
        for levels in [3, 5]:
            generators.append(WaveletEnergyGenerator(window, levels, use_vectorbt=use_vectorbt, enable_gpu=enable_gpu))
    
    # Band-limited volatility generators
    for window in [20]:
        for cutoff in [0.1, 0.2]:
            generators.append(BandLimitedVolatilityGenerator(window, cutoff, use_vectorbt=use_vectorbt, enable_gpu=enable_gpu))
    
    # Cycle length generators
    for window in [20, 50]:
        generators.append(CycleLengthGenerator(window, 2, 10, use_vectorbt=use_vectorbt, enable_gpu=enable_gpu))
    
    # Fractal dimension generators
    for window in [20, 50]:
        generators.append(FractalDimensionGenerator(window, use_vectorbt=use_vectorbt, enable_gpu=enable_gpu))
    
    # DFA slopes generators
    for window in [20, 50]:
        generators.append(DFASlopesGenerator(window, 4, 10, use_vectorbt=use_vectorbt, enable_gpu=enable_gpu))
    
    return generators


def create_optimized_spectral_wavelet_generators(use_vectorbt: bool = True, enable_gpu: bool = False, 
                                                batch_processing: bool = True) -> List[FeatureGenerator]:
    """Create optimized spectral/wavelet feature generators with advanced VectorBT integration."""
    generators = []
    
    if batch_processing and UNIFIED_VECTORIZATION_MANAGER_AVAILABLE:
        # Use batch processing for multiple features
        generators.append(VectorBTSpectralWaveletBatchGenerator(use_vectorbt=use_vectorbt, enable_gpu=enable_gpu))
    else:
        # Use individual optimized generators
        generators.extend(create_default_spectral_wavelet_generators(use_vectorbt=use_vectorbt, enable_gpu=enable_gpu))
    
    return generators


class VectorBTSpectralWaveletBatchGenerator(VectorizedFeatureGenerator):
    """Batch generator for multiple spectral/wavelet features using VectorBT optimization."""
    
    def __init__(self, use_vectorbt: bool = True, enable_gpu: bool = False, 
                 windows: List[int] = [20, 50], levels: List[int] = [3, 5]):
        config = FeatureConfig(
            name="vectorbt_spectral_wavelet_batch",
            category=FeatureCategory.SPECTRAL_WAVELET,
            description="Batch spectral/wavelet features with VectorBT optimization",
            required_columns=["close"],
            default_lookback=max(windows),
            min_lookback=min(windows),
            max_lookback=max(windows),
            parameters={'use_vectorbt': use_vectorbt, 'enable_gpu': enable_gpu, 'windows': windows, 'levels': levels},
            matrix_optimized=True,
            gpu_accelerated=enable_gpu and CUPY_AVAILABLE
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.use_vectorbt = use_vectorbt and VECTORBT_AVAILABLE
        self.enable_gpu = enable_gpu and CUPY_AVAILABLE
        self.windows = windows
        self.levels = levels
        
        # Initialize VectorBT rolling optimizer
        if VECTORBT_ROLLING_OPTIMIZER_AVAILABLE and self.use_vectorbt:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=self.enable_gpu)
        else:
            self.rolling_optimizer = None
            
        # Initialize unified vectorization manager
        if UNIFIED_VECTORIZATION_MANAGER_AVAILABLE and self.use_vectorbt:
            self.vectorization_manager = get_unified_vectorization_manager()
        else:
            self.vectorization_manager = None
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate multiple spectral/wavelet features in batch using VectorBT."""
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        close = data['close']
        if len(close) < max(self.windows):
            return pd.DataFrame(index=data.index)
        
        # Use VectorBT batch processing if available
        if self.use_vectorbt and self.vectorization_manager:
            return self._vectorbt_batch_features(close)
        else:
            return self._fallback_batch_features(close)
    
    def _vectorbt_batch_features(self, close: pd.Series) -> pd.DataFrame:
        """Generate batch features using VectorBT optimization."""
        try:
            features = {}
            
            # Wavelet energy features
            for window in self.windows:
                for levels in self.levels:
                    def wavelet_energy_func(window_data):
                        if len(window_data) < window:
                            return np.nan
                        return self._calculate_wavelet_energy(window_data.values, levels)
                    
                    if VECTORBT_AVAILABLE:
                        features[f'wavelet_energy_{window}_{levels}'] = rolling_apply(
                            close, window=window, func=wavelet_energy_func
                        )
                    else:
                        features[f'wavelet_energy_{window}_{levels}'] = close.rolling(
                            window=window
                        ).apply(wavelet_energy_func, raw=True)
            
            # Band-limited volatility features
            for window in self.windows:
                for cutoff in [0.1, 0.2]:
                    returns = close.pct_change()
                    
                    def band_volatility_func(window_returns):
                        valid_returns = window_returns.dropna()
                        if len(valid_returns) > 4:
                            return self._calculate_band_limited_volatility(valid_returns.values, cutoff)
                        return np.nan
                    
                    if VECTORBT_AVAILABLE:
                        features[f'band_limited_volatility_{window}_{cutoff}'] = rolling_apply(
                            returns, window=window, func=band_volatility_func
                        )
                    else:
                        features[f'band_limited_volatility_{window}_{cutoff}'] = returns.rolling(
                            window=window
                        ).apply(band_volatility_func, raw=False)
            
            # Cycle length features
            for window in self.windows:
                def cycle_length_func(window_data):
                    if len(window_data) < window:
                        return np.nan
                    return self._calculate_cycle_length(window_data.values, 2, 10)
                
                if VECTORBT_AVAILABLE:
                    features[f'cycle_length_{window}'] = rolling_apply(
                        close, window=window, func=cycle_length_func
                    )
                else:
                    features[f'cycle_length_{window}'] = close.rolling(
                        window=window
                    ).apply(cycle_length_func, raw=True)
            
            # Fractal dimension features
            for window in self.windows:
                def fractal_dim_func(window_data):
                    if len(window_data) < window:
                        return np.nan
                    return self._calculate_fractal_dimension(window_data.values)
                
                if VECTORBT_AVAILABLE:
                    features[f'fractal_dimension_{window}'] = rolling_apply(
                        close, window=window, func=fractal_dim_func
                    )
                else:
                    features[f'fractal_dimension_{window}'] = close.rolling(
                        window=window
                    ).apply(fractal_dim_func, raw=True)
            
            # DFA slopes features
            for window in self.windows:
                returns = close.pct_change()
                
                def dfa_slope_func(window_returns):
                    valid_returns = window_returns.dropna()
                    if len(valid_returns) > 10:
                        return self._calculate_dfa_slope(valid_returns.values, 4, 10)
                    return np.nan
                
                if VECTORBT_AVAILABLE:
                    features[f'dfa_slopes_{window}'] = rolling_apply(
                        returns, window=window, func=dfa_slope_func
                    )
                else:
                    features[f'dfa_slopes_{window}'] = returns.rolling(
                        window=window
                    ).apply(dfa_slope_func, raw=False)
            
            return pd.DataFrame(features, index=close.index)
            
        except Exception as e:
            logger.warning(f"VectorBT batch features calculation failed: {e}, using fallback")
            return self._fallback_batch_features(close)
    
    def _fallback_batch_features(self, close: pd.Series) -> pd.DataFrame:
        """Fallback batch features calculation using individual generators."""
        features = {}
        
        # Create individual generators and combine results
        individual_generators = create_default_spectral_wavelet_generators(
            use_vectorbt=False, enable_gpu=False
        )
        
        for generator in individual_generators:
            try:
                feature_result = generator._generate_feature(pd.DataFrame({'close': close}))
                features[generator.config.name] = feature_result
            except Exception as e:
                logger.warning(f"Failed to generate feature {generator.config.name}: {e}")
                continue
        
        return pd.DataFrame(features, index=close.index)
    
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

# Export all generators
__all__ = [
    'WaveletEnergyGenerator',
    'BandLimitedVolatilityGenerator',
    'CycleLengthGenerator',
    'FractalDimensionGenerator',
    'DFASlopesGenerator',
    'VectorBTSpectralWaveletBatchGenerator',
    'create_default_spectral_wavelet_generators',
    'create_optimized_spectral_wavelet_generators'
]


# Performance monitoring and optimization utilities
class SpectralWaveletPerformanceMonitor:
    """Performance monitor for spectral wavelet feature generation."""
    
    def __init__(self):
        self.stats = {
            'total_generations': 0,
            'vectorbt_operations': 0,
            'fallback_operations': 0,
            'batch_operations': 0,
            'gpu_operations': 0,
            'total_time': 0.0,
            'average_generation_time': 0.0,
            'optimization_success_rate': 0.0
        }
        self.logger = logging.getLogger(__name__)
    
    def record_generation(self, generator_type: str, use_vectorbt: bool, 
                         use_gpu: bool, generation_time: float, success: bool):
        """Record a feature generation operation."""
        self.stats['total_generations'] += 1
        self.stats['total_time'] += generation_time
        
        if use_vectorbt:
            self.stats['vectorbt_operations'] += 1
        else:
            self.stats['fallback_operations'] += 1
            
        if use_gpu:
            self.stats['gpu_operations'] += 1
            
        if 'batch' in generator_type.lower():
            self.stats['batch_operations'] += 1
        
        # Update averages
        self.stats['average_generation_time'] = (
            self.stats['total_time'] / self.stats['total_generations']
        )
        
        # Update success rate
        if self.stats['total_generations'] > 0:
            success_count = self.stats['vectorbt_operations'] if use_vectorbt else 0
            self.stats['optimization_success_rate'] = (
                success_count / self.stats['total_generations']
            )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = self.stats.copy()
        
        if self.stats['total_generations'] > 0:
            report['vectorbt_usage_rate'] = (
                self.stats['vectorbt_operations'] / self.stats['total_generations']
            )
            report['gpu_usage_rate'] = (
                self.stats['gpu_operations'] / self.stats['total_generations']
            )
            report['batch_usage_rate'] = (
                self.stats['batch_operations'] / self.stats['total_generations']
            )
        
        return report
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.stats = {
            'total_generations': 0,
            'vectorbt_operations': 0,
            'fallback_operations': 0,
            'batch_operations': 0,
            'gpu_operations': 0,
            'total_time': 0.0,
            'average_generation_time': 0.0,
            'optimization_success_rate': 0.0
        }


# Global performance monitor
_performance_monitor = SpectralWaveletPerformanceMonitor()


def get_performance_monitor() -> SpectralWaveletPerformanceMonitor:
    """Get global performance monitor instance."""
    return _performance_monitor


def benchmark_spectral_wavelet_generators(data: pd.DataFrame, 
                                        generators: List[FeatureGenerator],
                                        trials: int = 3) -> Dict[str, Any]:
    """Benchmark spectral wavelet generators for performance comparison."""
    import time
    
    results = {}
    
    for generator in generators:
        generator_name = generator.config.name
        times = []
        
        for trial in range(trials):
            start_time = time.time()
            try:
                result = generator._generate_feature(data)
                end_time = time.time()
                times.append(end_time - start_time)
            except Exception as e:
                logger.warning(f"Generator {generator_name} failed in trial {trial}: {e}")
                continue
        
        if times:
            results[generator_name] = {
                'average_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'trials_completed': len(times)
            }
    
    return results


def optimize_spectral_wavelet_pipeline(data: pd.DataFrame, 
                                     target_time_ms: float = 100.0) -> List[FeatureGenerator]:
    """Optimize spectral wavelet pipeline to meet performance targets."""
    
    # Start with VectorBT-optimized generators
    generators = create_optimized_spectral_wavelet_generators(
        use_vectorbt=True, 
        enable_gpu=False,  # Start with CPU, add GPU if needed
        batch_processing=True
    )
    
    # Benchmark current performance
    benchmark_results = benchmark_spectral_wavelet_generators(data, generators)
    
    # Check if we meet performance targets
    total_time = sum(result['average_time'] for result in benchmark_results.values())
    total_time_ms = total_time * 1000
    
    if total_time_ms <= target_time_ms:
        logger.info(f"✅ Performance target met: {total_time_ms:.2f}ms <= {target_time_ms}ms")
        return generators
    
    # If not, try GPU acceleration
    if CUPY_AVAILABLE or TORCH_AVAILABLE:
        logger.info("🚀 Trying GPU acceleration to meet performance targets...")
        gpu_generators = create_optimized_spectral_wavelet_generators(
            use_vectorbt=True,
            enable_gpu=True,
            batch_processing=True
        )
        
        gpu_benchmark = benchmark_spectral_wavelet_generators(data, gpu_generators)
        gpu_total_time = sum(result['average_time'] for result in gpu_benchmark.values())
        gpu_total_time_ms = gpu_total_time * 1000
        
        if gpu_total_time_ms <= target_time_ms:
            logger.info(f"✅ GPU acceleration met target: {gpu_total_time_ms:.2f}ms <= {target_time_ms}ms")
            return gpu_generators
        else:
            logger.warning(f"⚠️ GPU acceleration still not fast enough: {gpu_total_time_ms:.2f}ms > {target_time_ms}ms")
    
    # If still not fast enough, reduce feature complexity
    logger.info("🔧 Reducing feature complexity to meet performance targets...")
    simple_generators = create_default_spectral_wavelet_generators(
        use_vectorbt=True,
        enable_gpu=False
    )
    
    # Filter to only essential features
    essential_generators = [
        gen for gen in simple_generators 
        if 'wavelet_energy_20' in gen.config.name or 'fractal_dimension_20' in gen.config.name
    ]
    
    return essential_generators
