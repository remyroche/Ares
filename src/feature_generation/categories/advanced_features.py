"""
Consolidated Advanced Features Module

This module provides comprehensive advanced feature generators for quantitative finance,
combining statistical, volume, and spectral/wavelet analysis features.

Key Features:
- Advanced statistical indicators (Hurst exponent, jump indicators, CVaR, drawdown measures)
- Advanced volume features (OBV, AD, MFI, VWAP, volume profile analysis)
- Spectral/wavelet analysis (wavelet energy, cycle detection, fractal dimension)
- Full VectorBT integration for optimal performance
- GPU acceleration support
"""

import numpy as np
import pandas as pd
import warnings
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

# Optimization utilities
try:
    from ..utils.vectorization_optimizer import get_vectorization_optimizer
    from ..utils.optimized_feature_pipeline import get_optimized_feature_pipeline
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

# VectorBT Rolling Optimizer
try:
    from ..utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer, VectorBTRollingOptimizer
    VECTORBT_ROLLING_OPTIMIZER_AVAILABLE = True
except ImportError:
    VECTORBT_ROLLING_OPTIMIZER_AVAILABLE = False
    VectorBTRollingOptimizer = None

# Unified Vectorization Manager
try:
    from ...utils.ml_common.unified_vectorization_manager import get_unified_vectorization_manager, UnifiedVectorizationManager, OperationType
    UNIFIED_VECTORIZATION_AVAILABLE = True
except ImportError:
    UNIFIED_VECTORIZATION_AVAILABLE = False
    UnifiedVectorizationManager = None
    OperationType = None

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import (
        rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, 
        rolling_sum, rolling_apply, rolling_corr, rolling_cov,
        rolling_skew, rolling_kurt, rolling_quantile
    )
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    from vectorbt.indicators import OBV, AD, MFI, ADOSC, AROONOSC
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
    rolling_skew = None
    rolling_kurt = None
    rolling_quantile = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    OBV = None
    AD = None
    MFI = None
    ADOSC = None
    AROONOSC = None
    Portfolio = None
    generate_returns_nb = None
    generate_orders_nb = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

# Import tprint for consistent logging
try:
    from src.utils.tprint import tprint
except ImportError:
    def tprint(*args, **kwargs):
        print(*args, **kwargs)

# Import scipy for advanced statistical functions
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
    warnings.warn("SciPy not available. Some advanced features may not work properly")

# Import sklearn for machine learning features
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    StandardScaler = None
    KMeans = None
    PCA = None
    warnings.warn("Scikit-learn not available. Some ML features may not work properly")

# Import PyWavelets for wavelet analysis
try:
    import pywt
    PYWAVELETS_AVAILABLE = True
except ImportError:
    PYWAVELETS_AVAILABLE = False
    pywt = None
    warnings.warn("PyWavelets not available. Wavelet features will be disabled")


# ============================================================================
# ADVANCED STATISTICAL FEATURES
# ============================================================================

class HurstExponentGenerator(VectorizedFeatureGenerator):
    """
    Generator for Hurst exponent features using VectorBT optimization.
    
    The Hurst exponent is a statistical measure used to analyze long-range dependence
    in time series data. It helps identify whether a time series is trending, mean-reverting,
    or random walk.
    
    Hurst Exponent Interpretation:
    - H > 0.5: Persistent/trending behavior (long memory)
    - H = 0.5: Random walk (no memory)
    - H < 0.5: Mean-reverting behavior (anti-persistent)
    
    Parameters:
    - window: Lookback window for calculation (default: 50)
    - min_periods: Minimum periods required for valid calculation (default: 20)
    
    Returns:
    - pd.Series: Hurst exponent values (0.0 to 1.0)
    
    Example:
        >>> generator = HurstExponentGenerator(window=30)
        >>> hurst_values = generator._generate_feature(data)
        >>> print(f"Average Hurst exponent: {hurst_values.mean():.3f}")
    """
    
    def __init__(self, window: int = 50, min_periods: int = 20):
        config = FeatureConfig(
            name="hurst_exponent",
            category=FeatureCategory.STATISTICAL,
            description="Hurst exponent for long-range dependence analysis",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=min_periods,
            max_lookback=200,
            parameters={"window": window, "min_periods": min_periods}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.min_periods = min_periods
        
        # Initialize VectorBT optimizer
        self.vectorbt_optimizer = None
        if VECTORBT_ROLLING_OPTIMIZER_AVAILABLE:
            try:
                self.vectorbt_optimizer = get_vectorbt_rolling_optimizer()
            except Exception as e:
                tprint(f"⚠️ VectorBT optimizer initialization failed: {e}")
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Hurst exponent feature."""
        close = data['close']
        
        if VECTORBT_AVAILABLE and self.vectorbt_optimizer:
            try:
                # Use VectorBT rolling apply for Hurst exponent calculation
                hurst_series = self.vectorbt_optimizer.rolling_apply(
                    close, 
                    self._calculate_hurst_exponent, 
                    window=self.window,
                    min_periods=self.min_periods
                )
                return hurst_series
            except Exception as e:
                warnings.warn(f"VectorBT Hurst exponent calculation failed: {e}, using fallback")
                return self._calculate_hurst_fallback(close, self.window, self.min_periods, data.index)
        else:
            return self._calculate_hurst_fallback(close, self.window, self.min_periods, data.index)
    
    def _calculate_hurst_exponent(self, segment: np.ndarray) -> float:
        """Calculate Hurst exponent for a segment."""
        return self._hurst_exponent(segment)
    
    def _calculate_hurst_fallback(self, close: pd.Series, window: int, min_periods: int, index: pd.Index) -> pd.Series:
        """Fallback Hurst exponent calculation using pandas rolling."""
        hurst_values = []
        for i in range(len(close)):
            if i < min_periods - 1:
                hurst_values.append(np.nan)
            else:
                segment = close.iloc[i-window+1:i+1].values
                hurst = self._hurst_exponent(segment)
                hurst_values.append(hurst)
        
        return pd.Series(hurst_values, index=index)
    
    def _hurst_exponent(self, data: np.ndarray) -> float:
        """Calculate Hurst exponent using R/S analysis."""
        try:
            if len(data) < 10:
                return 0.5
            
            # Remove NaN values
            data = data[~np.isnan(data)]
            if len(data) < 10:
                return 0.5
            
            # Calculate returns
            returns = np.diff(np.log(data))
            if len(returns) < 5:
                return 0.5
            
            # R/S analysis
            n = len(returns)
            mean_return = np.mean(returns)
            deviations = returns - mean_return
            cumulative_deviations = np.cumsum(deviations)
            
            # Calculate range
            R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
            
            # Calculate standard deviation
            S = np.std(returns)
            
            if S == 0 or R == 0:
                return 0.5
            
            # R/S ratio
            rs_ratio = R / S
            
            # Hurst exponent
            hurst = np.log(rs_ratio) / np.log(n)
            
            # Clamp to reasonable range
            return max(0.0, min(1.0, hurst))
            
        except Exception:
            return 0.5


class JumpIndicatorsGenerator(VectorizedFeatureGenerator):
    """Generator for jump detection indicators using VectorBT optimization."""
    
    def __init__(self, window: int = 20, threshold: float = 3.0):
        config = FeatureConfig(
            name="jump_indicators",
            category=FeatureCategory.STATISTICAL,
            description="Jump detection indicators for volatility analysis",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=10,
            max_lookback=100,
            parameters={"window": window, "threshold": threshold}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.threshold = threshold
        
        # Initialize VectorBT optimizer
        self.vectorbt_optimizer = None
        if VECTORBT_ROLLING_OPTIMIZER_AVAILABLE:
            try:
                self.vectorbt_optimizer = get_vectorbt_rolling_optimizer()
            except Exception as e:
                tprint(f"⚠️ VectorBT optimizer initialization failed: {e}")
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate jump indicators feature."""
        close = data['close']
        
        if VECTORBT_AVAILABLE and self.vectorbt_optimizer:
            try:
                # Use VectorBT rolling apply for jump detection
                jump_series = self.vectorbt_optimizer.rolling_apply(
                    close, 
                    self._detect_jumps, 
                    window=self.window
                )
                return jump_series
            except Exception as e:
                warnings.warn(f"VectorBT jump detection failed: {e}, using fallback")
                return self._detect_jumps_fallback(close, self.window, self.threshold, data.index)
        else:
            return self._detect_jumps_fallback(close, self.window, self.threshold, data.index)
    
    def _detect_jumps(self, segment: np.ndarray) -> float:
        """Detect jumps in a segment."""
        return self._jump_indicator(segment, self.threshold)
    
    def _detect_jumps_fallback(self, close: pd.Series, window: int, threshold: float, index: pd.Index) -> pd.Series:
        """Fallback jump detection using pandas rolling."""
        jump_values = []
        for i in range(len(close)):
            if i < window - 1:
                jump_values.append(np.nan)
            else:
                segment = close.iloc[i-window+1:i+1].values
                jump_indicator = self._jump_indicator(segment, threshold)
                jump_values.append(jump_indicator)
        
        return pd.Series(jump_values, index=index)
    
    def _jump_indicator(self, data: np.ndarray, threshold: float) -> float:
        """Calculate jump indicator using Bipower Variation."""
        try:
            if len(data) < 5:
                return 0.0
            
            # Calculate returns
            returns = np.diff(np.log(data))
            if len(returns) < 3:
                return 0.0
            
            # Bipower variation
            abs_returns = np.abs(returns)
            bipower_variation = np.mean(abs_returns[:-1] * abs_returns[1:])
            
            # Realized variance
            realized_variance = np.mean(returns ** 2)
            
            # Jump test statistic
            if bipower_variation == 0:
                return 0.0
            
            jump_stat = (realized_variance - bipower_variation) / bipower_variation
            
            # Binary jump indicator
            return 1.0 if jump_stat > threshold else 0.0
            
        except Exception:
            return 0.0


class CVaRGenerator(VectorizedFeatureGenerator):
    """Generator for Conditional Value at Risk (CVaR) features."""
    
    def __init__(self, window: int = 20, confidence_level: float = 0.05):
        config = FeatureConfig(
            name="cvar",
            category=FeatureCategory.STATISTICAL,
            description="Conditional Value at Risk (CVaR) for tail risk analysis",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=10,
            max_lookback=100,
            parameters={"window": window, "confidence_level": confidence_level}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.confidence_level = confidence_level
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate CVaR feature."""
        close = data['close']
        returns = close.pct_change().dropna()
        
        if VECTORBT_AVAILABLE:
            try:
                # Use VectorBT rolling quantile for VaR calculation
                var_series = rolling_quantile(
                    returns, 
                    window=self.window, 
                    q=self.confidence_level
                )
                
                # Calculate CVaR as mean of returns below VaR
                cvar_series = returns.rolling(window=self.window).apply(
                    lambda x: self._calculate_cvar(x, self.confidence_level),
                    raw=False
                )
                
                return cvar_series
            except Exception as e:
                warnings.warn(f"VectorBT CVaR calculation failed: {e}, using fallback")
                return self._calculate_cvar_fallback(returns, self.window, self.confidence_level, data.index)
        else:
            return self._calculate_cvar_fallback(returns, self.window, self.confidence_level, data.index)
    
    def _calculate_cvar(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate CVaR for a series of returns."""
        try:
            if len(returns) < 5:
                return np.nan
            
            # Calculate VaR
            var = np.percentile(returns, confidence_level * 100)
            
            # Calculate CVaR as mean of returns below VaR
            tail_returns = returns[returns <= var]
            if len(tail_returns) == 0:
                return var
            
            return np.mean(tail_returns)
            
        except Exception:
            return np.nan
    
    def _calculate_cvar_fallback(self, returns: pd.Series, window: int, confidence_level: float, index: pd.Index) -> pd.Series:
        """Fallback CVaR calculation using pandas rolling."""
        cvar_values = []
        for i in range(len(returns)):
            if i < window - 1:
                cvar_values.append(np.nan)
            else:
                segment = returns.iloc[i-window+1:i+1]
                cvar = self._calculate_cvar(segment, confidence_level)
                cvar_values.append(cvar)
        
        return pd.Series(cvar_values, index=index)


class MaxDrawdownGenerator(VectorizedFeatureGenerator):
    """Generator for maximum drawdown features."""
    
    def __init__(self, window: int = 50):
        config = FeatureConfig(
            name="max_drawdown",
            category=FeatureCategory.STATISTICAL,
            description="Maximum drawdown for risk analysis",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=20,
            max_lookback=200,
            parameters={"window": window}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate maximum drawdown feature."""
        close = data['close']
        
        if VECTORBT_AVAILABLE:
            try:
                # Use VectorBT rolling apply for drawdown calculation
                drawdown_series = close.rolling(window=self.window).apply(
                    lambda x: self._calculate_max_drawdown(x),
                    raw=False
                )
                return drawdown_series
            except Exception as e:
                warnings.warn(f"VectorBT drawdown calculation failed: {e}, using fallback")
                return self._calculate_drawdown_fallback(close, self.window, data.index)
        else:
            return self._calculate_drawdown_fallback(close, self.window, data.index)
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown for a price series."""
        try:
            if len(prices) < 2:
                return 0.0
            
            # Calculate running maximum
            running_max = prices.expanding().max()
            
            # Calculate drawdown
            drawdown = (prices - running_max) / running_max
            
            # Return maximum drawdown (most negative value)
            return drawdown.min()
            
        except Exception:
            return 0.0
    
    def _calculate_drawdown_fallback(self, close: pd.Series, window: int, index: pd.Index) -> pd.Series:
        """Fallback drawdown calculation using pandas rolling."""
        drawdown_values = []
        for i in range(len(close)):
            if i < window - 1:
                drawdown_values.append(np.nan)
            else:
                segment = close.iloc[i-window+1:i+1]
                max_dd = self._calculate_max_drawdown(segment)
                drawdown_values.append(max_dd)
        
        return pd.Series(drawdown_values, index=index)


class RollingSkewnessKurtosisGenerator(VectorizedFeatureGenerator):
    """Generator for rolling skewness and kurtosis features."""
    
    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name="rolling_skewness_kurtosis",
            category=FeatureCategory.STATISTICAL,
            description="Rolling skewness and kurtosis for distribution analysis",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=10,
            max_lookback=100,
            parameters={"window": window}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate rolling skewness and kurtosis feature."""
        close = data['close']
        returns = close.pct_change().dropna()
        
        if VECTORBT_AVAILABLE:
            try:
                # Use VectorBT rolling skew and kurt
                skew_series = rolling_skew(returns, window=self.window)
                kurt_series = rolling_kurt(returns, window=self.window)
                
                # Combine skewness and kurtosis
                combined = (skew_series + kurt_series) / 2
                return combined
            except Exception as e:
                warnings.warn(f"VectorBT skewness/kurtosis calculation failed: {e}, using fallback")
                return self._calculate_skew_kurt_fallback(returns, self.window, data.index)
        else:
            return self._calculate_skew_kurt_fallback(returns, self.window, data.index)
    
    def _calculate_skew_kurt_fallback(self, returns: pd.Series, window: int, index: pd.Index) -> pd.Series:
        """Fallback skewness/kurtosis calculation using pandas rolling."""
        skew_kurt_values = []
        for i in range(len(returns)):
            if i < window - 1:
                skew_kurt_values.append(np.nan)
            else:
                segment = returns.iloc[i-window+1:i+1]
                if len(segment) >= 3:
                    skew_val = segment.skew()
                    kurt_val = segment.kurtosis()
                    combined = (skew_val + kurt_val) / 2
                else:
                    combined = np.nan
                skew_kurt_values.append(combined)
        
        return pd.Series(skew_kurt_values, index=index)


class TrendPersistenceGenerator(VectorizedFeatureGenerator):
    """Generator for trend persistence features."""
    
    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name="trend_persistence",
            category=FeatureCategory.STATISTICAL,
            description="Trend persistence analysis using autocorrelation",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=10,
            max_lookback=100,
            parameters={"window": window}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate trend persistence feature."""
        close = data['close']
        returns = close.pct_change().dropna()
        
        if VECTORBT_AVAILABLE:
            try:
                # Use VectorBT rolling correlation for autocorrelation
                autocorr_series = returns.rolling(window=self.window).apply(
                    lambda x: x.autocorr(lag=1) if len(x) > 1 else 0,
                    raw=False
                )
                return autocorr_series
            except Exception as e:
                warnings.warn(f"VectorBT trend persistence calculation failed: {e}, using fallback")
                return self._calculate_trend_persistence_fallback(returns, self.window, data.index)
        else:
            return self._calculate_trend_persistence_fallback(returns, self.window, data.index)
    
    def _calculate_trend_persistence_fallback(self, returns: pd.Series, window: int, index: pd.Index) -> pd.Series:
        """Fallback trend persistence calculation using pandas rolling."""
        persistence_values = []
        for i in range(len(returns)):
            if i < window - 1:
                persistence_values.append(np.nan)
            else:
                segment = returns.iloc[i-window+1:i+1]
                if len(segment) > 1:
                    autocorr = segment.autocorr(lag=1)
                    persistence_values.append(autocorr if not np.isnan(autocorr) else 0)
                else:
                    persistence_values.append(0)
        
        return pd.Series(persistence_values, index=index)


# ============================================================================
# ADVANCED VOLUME FEATURES
# ============================================================================

@dataclass
class VolumeConfig:
    """Configuration for advanced volume features."""
    enable_obv: bool = True
    enable_ad: bool = True
    enable_mfi: bool = True
    enable_vwap: bool = True
    enable_volume_profile: bool = True
    obv_window: int = 20
    ad_window: int = 20
    mfi_window: int = 14
    vwap_window: int = 20
    volume_profile_bins: int = 10


class AdvancedVolumeFeatures(VectorizedFeatureGenerator):
    """Advanced volume features with VectorBT optimization."""
    
    def __init__(self, config: Optional[VolumeConfig] = None):
        if config is None:
            config = VolumeConfig()
        
        self.volume_config = config
        
        feature_config = FeatureConfig(
            name="advanced_volume_features",
            category=FeatureCategory.VOLUME,
            description="Advanced volume features with VectorBT optimization",
            required_columns=["close", "volume"],
            optional_columns=["high", "low", "open"],
            default_lookback=20,
            min_lookback=10,
            max_lookback=100,
            parameters={}
        )
        
        super().__init__(feature_config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        # Initialize VectorBT optimizer
        self.vectorbt_optimizer = None
        if VECTORBT_ROLLING_OPTIMIZER_AVAILABLE:
            try:
                self.vectorbt_optimizer = get_vectorbt_rolling_optimizer()
            except Exception as e:
                tprint(f"⚠️ VectorBT optimizer initialization failed: {e}")
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate advanced volume features."""
        features = self.generate_features(data, **kwargs)
        
        # Return the first feature as representative
        if features:
            first_feature_name = list(features.keys())[0]
            return pd.Series(features[first_feature_name], index=data.index[:len(features[first_feature_name])])
        else:
            return pd.Series(np.zeros(len(data)), index=data.index)
    
    def generate_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """Generate all advanced volume features."""
        features = {}
        
        try:
            # On-Balance Volume (OBV)
            if self.volume_config.enable_obv and VECTORBT_AVAILABLE and OBV is not None:
                try:
                    obv = OBV.run(data['close'], data['volume'])
                    features['obv'] = obv.values
                    features['obv_sma'] = rolling_mean(obv, window=self.volume_config.obv_window).values
                except Exception as e:
                    tprint(f"⚠️ OBV calculation failed: {e}")
            
            # Accumulation/Distribution Line (AD)
            if self.volume_config.enable_ad and VECTORBT_AVAILABLE and AD is not None:
                try:
                    ad = AD.run(data['high'], data['low'], data['close'], data['volume'])
                    features['ad'] = ad.values
                    features['ad_sma'] = rolling_mean(ad, window=self.volume_config.ad_window).values
                except Exception as e:
                    tprint(f"⚠️ AD calculation failed: {e}")
            
            # Money Flow Index (MFI)
            if self.volume_config.enable_mfi and VECTORBT_AVAILABLE and MFI is not None:
                try:
                    mfi = MFI.run(data['high'], data['low'], data['close'], data['volume'], window=self.volume_config.mfi_window)
                    features['mfi'] = mfi.values
                except Exception as e:
                    tprint(f"⚠️ MFI calculation failed: {e}")
            
            # Volume-Weighted Average Price (VWAP)
            if self.volume_config.enable_vwap and VECTORBT_AVAILABLE:
                try:
                    typical_price = (data['high'] + data['low'] + data['close']) / 3
                    vwap = (typical_price * data['volume']).rolling(window=self.volume_config.vwap_window).sum() / data['volume'].rolling(window=self.volume_config.vwap_window).sum()
                    features['vwap'] = vwap.values
                    features['vwap_ratio'] = (data['close'] / vwap).values
                except Exception as e:
                    tprint(f"⚠️ VWAP calculation failed: {e}")
            
            # Volume Rate of Change
            try:
                volume_roc = data['volume'].pct_change(periods=5)
                features['volume_roc'] = volume_roc.values
            except Exception as e:
                tprint(f"⚠️ Volume ROC calculation failed: {e}")
            
            # Volume Profile Analysis
            if self.volume_config.enable_volume_profile:
                try:
                    volume_profile = self._calculate_volume_profile(data)
                    features.update(volume_profile)
                except Exception as e:
                    tprint(f"⚠️ Volume profile calculation failed: {e}")
            
        except Exception as e:
            tprint(f"⚠️ Advanced volume features generation failed: {e}")
        
        return features
    
    def _calculate_volume_profile(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate volume profile features."""
        features = {}
        
        try:
            if 'high' not in data.columns or 'low' not in data.columns:
                return features
            
            # Price range
            price_range = data['high'] - data['low']
            price_center = (data['high'] + data['low']) / 2
            
            # Volume-weighted price
            volume_weighted_price = (data['close'] * data['volume']).rolling(window=20).sum() / data['volume'].rolling(window=20).sum()
            
            # Volume profile features
            features['volume_profile_center'] = price_center.values
            features['volume_profile_range'] = price_range.values
            features['volume_profile_vwp'] = volume_weighted_price.values
            
            # Volume clustering
            volume_ma = data['volume'].rolling(window=20).mean()
            volume_clustering = (data['volume'] / volume_ma).values
            features['volume_clustering'] = volume_clustering
            
        except Exception as e:
            tprint(f"⚠️ Volume profile calculation failed: {e}")
        
        return features


# ============================================================================
# SPECTRAL/WAVELET FEATURES
# ============================================================================

class WaveletEnergyGenerator(VectorizedFeatureGenerator):
    """Generator for wavelet energy features."""
    
    def __init__(self, window: int = 64, wavelet: str = 'db4'):
        config = FeatureConfig(
            name="wavelet_energy",
            category=FeatureCategory.SPECTRAL,
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
    """Generator for band-limited volatility features."""
    
    def __init__(self, window: int = 32, low_freq: float = 0.1, high_freq: float = 0.5):
        config = FeatureConfig(
            name="band_limited_volatility",
            category=FeatureCategory.SPECTRAL,
            description="Band-limited volatility using spectral analysis",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=16,
            max_lookback=128,
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
    """Generator for cycle length detection features."""
    
    def __init__(self, window: int = 64, min_cycle: int = 4, max_cycle: int = 32):
        config = FeatureConfig(
            name="cycle_length",
            category=FeatureCategory.SPECTRAL,
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
    """Generator for fractal dimension features."""
    
    def __init__(self, window: int = 32):
        config = FeatureConfig(
            name="fractal_dimension",
            category=FeatureCategory.SPECTRAL,
            description="Fractal dimension analysis for complexity measurement",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=16,
            max_lookback=128,
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
    """Generator for Detrended Fluctuation Analysis (DFA) slopes."""
    
    def __init__(self, window: int = 64, min_scale: int = 4, max_scale: int = 32):
        config = FeatureConfig(
            name="dfa_slopes",
            category=FeatureCategory.SPECTRAL,
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
            if len(returns) < 16:
                return 0.5
            
            # Remove NaN values
            returns = returns.dropna()
            if len(returns) < 16:
                return 0.5
            
            # Calculate cumulative sum
            y = np.cumsum(returns.values)
            
            # Define scales
            scales = np.logspace(np.log10(self.min_scale), np.log10(min(self.max_scale, len(y)//4)), 10).astype(int)
            scales = np.unique(scales)
            
            if len(scales) < 2:
                return 0.5
            
            # Calculate fluctuation function
            fluctuations = []
            
            for scale in scales:
                if scale >= len(y):
                    continue
                
                # Divide into segments
                n_segments = len(y) // scale
                if n_segments < 2:
                    continue
                
                segment_fluctuations = []
                
                for i in range(n_segments):
                    start_idx = i * scale
                    end_idx = start_idx + scale
                    segment = y[start_idx:end_idx]
                    
                    # Detrend (linear fit)
                    x = np.arange(len(segment))
                    coeffs = np.polyfit(x, segment, 1)
                    trend = np.polyval(coeffs, x)
                    detrended = segment - trend
                    
                    # Calculate fluctuation
                    fluctuation = np.sqrt(np.mean(detrended ** 2))
                    segment_fluctuations.append(fluctuation)
                
                if segment_fluctuations:
                    fluctuations.append(np.mean(segment_fluctuations))
            
            if len(fluctuations) < 2:
                return 0.5
            
            # Calculate slope
            log_scales = np.log(scales[:len(fluctuations)])
            log_fluctuations = np.log(fluctuations)
            
            slope, _ = np.polyfit(log_scales, log_fluctuations, 1)
            
            return slope
            
        except Exception:
            return 0.5
    
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
    """Batch generator for spectral and wavelet features using VectorBT."""
    
    def __init__(self, window: int = 64):
        config = FeatureConfig(
            name="vectorbt_spectral_wavelet_batch",
            category=FeatureCategory.SPECTRAL,
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
        self.vectorbt_optimizer = None
        if VECTORBT_ROLLING_OPTIMIZER_AVAILABLE:
            try:
                self.vectorbt_optimizer = get_vectorbt_rolling_optimizer()
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
# FACTORY FUNCTIONS
# ============================================================================

def create_advanced_feature_generators() -> List[FeatureGenerator]:
    """Create all advanced feature generators."""
    generators = []
    
    # Advanced statistical generators
    generators.append(HurstExponentGenerator())
    generators.append(JumpIndicatorsGenerator())
    generators.append(CVaRGenerator())
    generators.append(MaxDrawdownGenerator())
    generators.append(RollingSkewnessKurtosisGenerator())
    generators.append(TrendPersistenceGenerator())
    
    # Advanced volume generators
    generators.append(AdvancedVolumeFeatures())
    
    # Spectral/wavelet generators
    generators.append(WaveletEnergyGenerator())
    generators.append(BandLimitedVolatilityGenerator())
    generators.append(CycleLengthGenerator())
    generators.append(FractalDimensionGenerator())
    generators.append(DFASlopesGenerator())
    generators.append(VectorBTSpectralWaveletBatchGenerator())
    
    return generators


def create_advanced_statistical_generators() -> List[FeatureGenerator]:
    """Create advanced statistical feature generators."""
    generators = []
    
    for window in [20, 30, 50]:
        generators.append(HurstExponentGenerator(window))
        generators.append(JumpIndicatorsGenerator(window))
        generators.append(CVaRGenerator(window))
        generators.append(MaxDrawdownGenerator(window))
        generators.append(RollingSkewnessKurtosisGenerator(window))
        generators.append(TrendPersistenceGenerator(window))
    
    return generators


def create_advanced_volume_generators() -> List[FeatureGenerator]:
    """Create advanced volume feature generators."""
    generators = []
    
    # Different volume configurations
    configs = [
        VolumeConfig(enable_obv=True, enable_ad=True, enable_mfi=True, enable_vwap=True),
        VolumeConfig(enable_obv=True, enable_ad=False, enable_mfi=True, enable_vwap=True),
        VolumeConfig(enable_obv=False, enable_ad=True, enable_mfi=True, enable_vwap=True),
    ]
    
    for config in configs:
        generators.append(AdvancedVolumeFeatures(config))
    
    return generators


def create_spectral_wavelet_generators() -> List[FeatureGenerator]:
    """Create spectral and wavelet feature generators."""
    generators = []
    
    for window in [32, 64, 128]:
        generators.append(WaveletEnergyGenerator(window))
        generators.append(BandLimitedVolatilityGenerator(window))
        generators.append(CycleLengthGenerator(window))
        generators.append(FractalDimensionGenerator(window))
        generators.append(DFASlopesGenerator(window))
        generators.append(VectorBTSpectralWaveletBatchGenerator(window))
    
    return generators


def process_advanced_features_batch(data: pd.DataFrame, 
                                  generators: Optional[List[FeatureGenerator]] = None,
                                  use_vectorbt: bool = True,
                                  **kwargs) -> pd.DataFrame:
    """
    Process advanced features in batch using VectorBT optimizations.
    
    Args:
        data: Input OHLCV data
        generators: List of feature generators (uses default if None)
        use_vectorbt: Whether to use VectorBT batch processing
        **kwargs: Additional parameters
        
    Returns:
        DataFrame with generated advanced features
    """
    if generators is None:
        generators = create_advanced_feature_generators()
    
    if use_vectorbt and OPTIMIZATION_AVAILABLE:
        try:
            # Use unified optimization system for batch processing
            from ..utils.unified_optimization_system import get_unified_optimization_system
            unified_optimizer = get_unified_optimization_system()
            
            # Process features in batch
            result = unified_optimizer.process_features_batch(data, generators, **kwargs)
            return result
            
        except Exception as e:
            warnings.warn(f"VectorBT batch processing failed: {e}, using sequential processing")
            return _process_advanced_features_sequential(data, generators, **kwargs)
    else:
        return _process_advanced_features_sequential(data, generators, **kwargs)


def _process_advanced_features_sequential(data: pd.DataFrame, 
                                        generators: List[FeatureGenerator],
                                        **kwargs) -> pd.DataFrame:
    """Process advanced features sequentially (fallback)."""
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
    # Advanced Statistical Features
    'HurstExponentGenerator',
    'JumpIndicatorsGenerator',
    'CVaRGenerator',
    'MaxDrawdownGenerator',
    'RollingSkewnessKurtosisGenerator',
    'TrendPersistenceGenerator',
    
    # Advanced Volume Features
    'VolumeConfig',
    'AdvancedVolumeFeatures',
    
    # Spectral/Wavelet Features
    'WaveletEnergyGenerator',
    'BandLimitedVolatilityGenerator',
    'CycleLengthGenerator',
    'FractalDimensionGenerator',
    'DFASlopesGenerator',
    'VectorBTSpectralWaveletBatchGenerator',
    
    # Factory Functions
    'create_advanced_feature_generators',
    'create_advanced_statistical_generators',
    'create_advanced_volume_generators',
    'create_spectral_wavelet_generators',
    'process_advanced_features_batch'
]