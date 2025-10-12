"""
Regime Volatility Feature Generator

This module provides feature generators specifically designed for regime classification
in 15-minute timeframes with 5-30 minute trade durations. Focuses on volatility
regime characteristics rather than short-term trading signals.

Key Features:
- Volatility clustering and persistence
- Volatility regime transitions
- Volatility-of-volatility measures
- Regime stability indicators
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
from scipy import stats
from scipy.signal import find_peaks

from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory
# Optimization utilities
try:
    from ..utils.vectorization_optimizer import get_vectorization_optimizer
    from ..utils.optimized_feature_pipeline import get_optimized_feature_pipeline
    from ..utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer, VectorBTRollingOptimizer
    from ..utils.unified_optimization_system import get_unified_optimization_system, UnifiedOptimizationSystem
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
from ..base_calculations import (
    BaseCalculator,
    BaseCalculationType,
    BaseCalculationConfig,
    create_base_calculator
)

# Import tprint for consistent logging
from src.utils.tprint import tprint

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
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
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

class RegimeVolatilityFeatureGenerator(VectorizedFeatureGenerator):
    """Feature generator for volatility regime features optimized for 15m timeframe."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        # Initialize VectorBT optimizers
        self.vectorbt_optimizer = None
        self.unified_optimizer = None
        if OPTIMIZATION_AVAILABLE:
            try:
                self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(
                    enable_gpu=getattr(config, 'gpu_accelerated', False),
                    enable_parallel=True
                )
                self.unified_optimizer = get_unified_optimization_system()
                tprint("✅ VectorBT optimizers initialized for RegimeVolatilityFeatureGenerator")
            except Exception as e:
                tprint(f"⚠️ VectorBT optimizer initialization failed: {e}")
    
    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="regime_volatility_features",
            category=FeatureCategory.VOLATILITY,
            description="Volatility regime features for 15m timeframe regime classification",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=20,  # 5 hours in 15m periods
            min_lookback=4,       # 1 hour minimum
            max_lookback=80,      # 20 hours maximum
            parameters={
                "regime_windows": [12, 30, 80],  # 3h, 7.5h, 20h in 15m periods (original min, middle, new max)
                "persistence_windows": [8, 20, 64],  # 2h, 5h, 16h (original min, middle, new max)
                "vol_of_vol_windows": [16, 40, 128],  # 4h, 10h, 32h (original min, middle, new max)
                "transition_windows": [4, 12, 32]  # 1h, 3h, 8h (original min, middle, new max)
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate a single volatility regime feature as required by the base class."""
        try:
            # Generate all volatility features
            features_dict = self.generate_features(data, **kwargs)
            
            # Combine all features into a single series (use first feature as representative)
            if features_dict:
                first_feature_name = list(features_dict.keys())[0]
                return pd.Series(features_dict[first_feature_name], index=data.index[:len(features_dict[first_feature_name])])
            else:
                # Return a simple volatility feature if no features generated
                returns = self._get_returns(data)
                if returns is not None and len(returns) > 0:
                    vol_feature = np.abs(returns)  # Simple volatility proxy
                    return pd.Series(vol_feature, index=data.index[1:len(vol_feature)+1])
                else:
                    return pd.Series(np.zeros(len(data)), index=data.index)
                
        except Exception as e:
            tprint(f"_generate_feature: Volatility feature generation failed: {e}")
            return pd.Series(np.zeros(len(data)), index=data.index)

    def generate_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """Generate volatility regime features."""
        features = {}
        
        # Get base calculations
        returns = self._get_returns(data)
        if returns is None:
            return features
        
        # 1. Volatility Regime Persistence
        features.update(self._generate_volatility_persistence_features(returns, data))
        
        # 2. Volatility Clustering Features
        features.update(self._generate_volatility_clustering_features(returns, data))
        
        # 3. Volatility-of-Volatility Features
        features.update(self._generate_vol_of_vol_features(returns, data))
        
        # 4. Volatility Regime Transitions
        features.update(self._generate_volatility_transition_features(returns, data))
        
        # 5. Volatility Regime Stability
        features.update(self._generate_volatility_stability_features(returns, data))
        
        return features
    
    def _get_returns(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Calculate log returns for volatility analysis."""
        if 'close' not in data.columns:
            return None
        
        close_prices = data['close'].values
        if len(close_prices) < 2:
            return None
        
        # Use log returns for better volatility regime analysis
        returns = np.diff(np.log(close_prices))
        return returns
    
    def _generate_volatility_persistence_features(self, returns: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate volatility persistence features for regime detection."""
        features = {}
        windows = self.config.parameters["regime_windows"]
        
        for window in windows:
            if len(returns) < window:
                continue
            
            # Rolling volatility
            vol = self._rolling_volatility(returns, window)
            
            # Volatility persistence (autocorrelation of volatility)
            vol_persistence = self._calculate_volatility_persistence(vol, window // 4)
            
            # Volatility regime strength
            vol_regime_strength = self._calculate_volatility_regime_strength(vol, window)
            
            # Pad to match data length
            vol_padded = np.full(len(data), np.nan)
            vol_persistence_padded = np.full(len(data), np.nan)
            vol_strength_padded = np.full(len(data), np.nan)

            # Account for returns being 1 element shorter than data
            # vol is a rolling result of returns, vol[0] aligns with data[window]
            # vol has length len(returns) - window + 1
            vol_padded[window:window+len(vol)] = vol
            vol_persistence_padded[window:window+len(vol_persistence)] = vol_persistence
            vol_strength_padded[window:window+len(vol_regime_strength)] = vol_regime_strength
            
            features[f'vol_persistence_{window}'] = vol_persistence_padded
            features[f'vol_regime_strength_{window}'] = vol_strength_padded
        
        return features
    
    def _generate_volatility_clustering_features(self, returns: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate volatility clustering features."""
        features = {}
        windows = self.config.parameters["regime_windows"]
        
        for window in windows:
            if len(returns) < window:
                continue
            
            # GARCH-like volatility clustering
            vol_clustering = self._calculate_volatility_clustering(returns, window)
            
            # Volatility regime consistency
            vol_consistency = self._calculate_volatility_consistency(returns, window)
            
            # Pad to match data length
            clustering_padded = np.full(len(data), np.nan)
            consistency_padded = np.full(len(data), np.nan)

            # Account for returns being 1 element shorter than data
            # returns[i] aligns with data[i+1], so feature[window:] aligns with data[window+1:]
            clustering_padded[window+1:] = vol_clustering[window:]
            consistency_padded[window+1:] = vol_consistency[window:]
            
            features[f'vol_clustering_{window}'] = clustering_padded
            features[f'vol_consistency_{window}'] = consistency_padded
        
        return features
    
    def _generate_vol_of_vol_features(self, returns: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate volatility-of-volatility features."""
        features = {}
        windows = self.config.parameters["vol_of_vol_windows"]
        
        for window in windows:
            if len(returns) < window * 2:
                continue
            
            # Calculate volatility of volatility
            vol_of_vol = self._calculate_volatility_of_volatility(returns, window)
            
            # Volatility regime uncertainty
            vol_uncertainty = self._calculate_volatility_uncertainty(returns, window)
            
            # Pad to match data length
            vol_of_vol_padded = np.full(len(data), np.nan)
            uncertainty_padded = np.full(len(data), np.nan)

            # Account for returns being 1 element shorter than data
            # returns[i] aligns with data[i+1], so feature[window:] aligns with data[window+1:]
            vol_of_vol_padded[window*2+1:] = vol_of_vol[window*2:]
            uncertainty_padded[window+1:] = vol_uncertainty[window:]
            
            features[f'vol_of_vol_{window}'] = vol_of_vol_padded
            features[f'vol_uncertainty_{window}'] = uncertainty_padded
        
        return features
    
    def _generate_volatility_transition_features(self, returns: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate volatility regime transition features."""
        features = {}
        windows = self.config.parameters["transition_windows"]
        
        for window in windows:
            if len(returns) < window * 2:
                continue
            
            # Volatility regime change detection
            vol_change = self._detect_volatility_regime_changes(returns, window)
            
            # Transition probability
            transition_prob = self._calculate_volatility_transition_probability(returns, window)
            
            # Pad to match data length
            change_padded = np.full(len(data), np.nan)
            prob_padded = np.full(len(data), np.nan)

            # Account for returns being 1 element shorter than data
            # returns[i] aligns with data[i+1], so feature[window*2:] aligns with data[window*2+1:]
            change_padded[window*2+1:] = vol_change[window*2:]
            prob_padded[window*2+1:] = transition_prob[window*2:]
            
            features[f'vol_regime_change_{window}'] = change_padded
            features[f'vol_transition_prob_{window}'] = prob_padded
        
        return features
    
    def _generate_volatility_stability_features(self, returns: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate volatility regime stability features."""
        features = {}
        windows = self.config.parameters["persistence_windows"]
        
        for window in windows:
            if len(returns) < window:
                continue
            
            # Volatility regime stability
            vol_stability = self._calculate_volatility_stability(returns, window)
            
            # Regime persistence score
            persistence_score = self._calculate_regime_persistence_score(returns, window)
            
            # Pad to match data length
            stability_padded = np.full(len(data), np.nan)
            persistence_padded = np.full(len(data), np.nan)

            # Account for returns being 1 element shorter than data
            # returns[i] aligns with data[i+1], so vol_stability[window:] aligns with data[window+1:]
            stability_padded[window+1:] = vol_stability[window:]
            persistence_padded[window+1:] = persistence_score[window:]
            
            features[f'vol_stability_{window}'] = stability_padded
            features[f'regime_persistence_{window}'] = persistence_padded
        
        return features
    
    def _rolling_volatility(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling volatility - VECTORIZED."""
        if len(returns) < window:
            return np.array([])
        
        # Vectorized approach using pandas rolling
        returns_series = pd.Series(returns)
        vol = self._vectorbt_rolling_operation(returns_series, "std", window).dropna().values
        
        return vol
    
    def _calculate_volatility_persistence(self, vol: np.ndarray, lag: int) -> np.ndarray:
        """Calculate volatility persistence using autocorrelation - OPTIMIZED VECTORIZED."""
        if len(vol) < lag + 1:
            return np.zeros(len(vol))
        
        # OPTIMIZED: Use vectorized autocorrelation calculation
        vol_series = pd.Series(vol)
        
        # Calculate volatility changes for autocorrelation
        vol_changes = vol_series.diff()
        
        # Vectorized persistence using rolling autocorrelation
        persistence = vol_changes.rolling(window=lag+1).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, 
            raw=False
        ).fillna(0).values
        
        return persistence
    
    def _calculate_volatility_regime_strength(self, vol: np.ndarray, window: int) -> np.ndarray:
        """Calculate volatility regime strength - VECTORIZED."""
        if len(vol) < window:
            return np.zeros(len(vol))
        
        # Vectorized regime strength calculation
        vol_series = pd.Series(vol)
        rolling_std = self._vectorbt_rolling_operation(vol_series, "std", window)
        rolling_mean = self._vectorbt_rolling_operation(vol_series, "mean", window)
        
        # Regime strength based on consistency of volatility level
        vol_consistency = 1.0 - (rolling_std / (rolling_mean + 1e-8))
        strength = vol_consistency.clip(0, 1)
        
        return strength.fillna(0).values
    
    def _calculate_volatility_clustering(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate GARCH-like volatility clustering - OPTIMIZED VECTORIZED."""
        if len(returns) < window:
            return np.zeros(len(returns))
        
        # OPTIMIZED: Use vectorized squared returns autocorrelation
        returns_series = pd.Series(returns)
        squared_returns = returns_series ** 2
        
        # Vectorized clustering using rolling autocorrelation
        clustering = squared_returns.rolling(window=window).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, 
            raw=False
        ).fillna(0).values
        
        return clustering
    
    def _calculate_volatility_consistency(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate volatility regime consistency - VECTORIZED."""
        if len(returns) < window:
            return np.zeros(len(returns))
        
        # Vectorized consistency calculation using rolling volatility
        returns_series = pd.Series(returns)
        vol_window_size = max(1, window // 4)
        
        # Calculate rolling volatility
        rolling_vol = self._vectorbt_rolling_operation(returns_series, "std", vol_window_size)
        
        # Calculate consistency using rolling coefficient of variation
        vol_rolling_std = self._vectorbt_rolling_operation(rolling_vol, "std", window)
        vol_rolling_mean = self._vectorbt_rolling_operation(rolling_vol, "mean", window)
        
        cv = vol_rolling_std / (vol_rolling_mean + 1e-8)
        consistency = (1 - cv).clip(0, 1)
        
        return consistency.fillna(0).values
    
    def _calculate_volatility_of_volatility(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate volatility of volatility - VECTORIZED."""
        if len(returns) < window * 2:
            return np.zeros(len(returns))
        
        # Vectorized volatility of volatility calculation
        returns_series = pd.Series(returns)
        
        # Calculate rolling volatility for both windows
        vol1 = self._vectorbt_rolling_operation(returns_series, "std", window).shift(window)
        vol2 = self._vectorbt_rolling_operation(returns_series, "std", window)
        
        # Volatility of volatility
        vol_of_vol = ((vol2 - vol1).abs() / (vol1 + 1e-8)).fillna(0)
        
        return vol_of_vol.values
    
    def _calculate_volatility_uncertainty(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate volatility regime uncertainty - VECTORIZED."""
        if len(returns) < window:
            return np.zeros(len(returns))
        
        # Vectorized uncertainty calculation
        returns_series = pd.Series(returns)
        vol_window_size = max(1, window // 4)
        
        # Calculate rolling volatility
        rolling_vol = self._vectorbt_rolling_operation(returns_series, "std", vol_window_size)
        
        # Calculate uncertainty using rolling coefficient of variation
        vol_rolling_std = self._vectorbt_rolling_operation(rolling_vol, "std", window)
        vol_rolling_mean = self._vectorbt_rolling_operation(rolling_vol, "mean", window)
        
        vol_vol = vol_rolling_std / (vol_rolling_mean + 1e-8)
        uncertainty = vol_vol.clip(0, 1)
        
        return uncertainty.fillna(0).values
    
    def _detect_volatility_regime_changes(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Detect volatility regime changes - VECTORIZED."""
        if len(returns) < window * 2:
            return np.zeros(len(returns))
        
        # Vectorized regime change detection
        returns_series = pd.Series(returns)
        
        # Calculate rolling volatility for both windows
        vol1 = self._vectorbt_rolling_operation(returns_series, "std", window).shift(window)
        vol2 = self._vectorbt_rolling_operation(returns_series, "std", window)
        
        # Significant change threshold (50% change)
        change_ratio = ((vol2 - vol1).abs() / (vol1 + 1e-8)).fillna(0)
        changes = (change_ratio > 0.5).astype(int)
        
        return changes.values
    
    def _calculate_volatility_transition_probability(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate volatility regime transition probability - OPTIMIZED VECTORIZED."""
        if len(returns) < window * 2:
            return np.zeros(len(returns))
        
        # OPTIMIZED: Use vectorized transition probability calculation
        returns_series = pd.Series(returns)
        vol_window_size = max(1, window // 2)
        
        # Calculate rolling volatility
        rolling_vol = self._vectorbt_rolling_operation(returns_series, "std", vol_window_size)
        
        # Vectorized transition probability using volatility changes
        vol_changes = rolling_vol.diff()
        vol_mean = self._vectorbt_rolling_operation(rolling_vol, "mean", window)
        
        # Transition probability based on volatility change rate
        transition_prob = vol_changes.abs() / (vol_mean + 1e-8)
        transition_prob = transition_prob.clip(0, 1)
        
        return transition_prob.fillna(0).values
    
    def _calculate_trend_probability(self, vol_window: pd.Series) -> float:
        """Calculate trend probability for a volatility window."""
        if len(vol_window) < 2:
            return 0.0
        
        # Probability based on volatility trend
        x = np.arange(len(vol_window))
        trend = np.polyfit(x, vol_window, 1)[0]
        mean_vol = vol_window.mean()
        return min(1, max(0, abs(trend) / (mean_vol + 1e-8)))
    
    def _calculate_volatility_stability(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate volatility regime stability - VECTORIZED."""
        if len(returns) < window:
            return np.zeros(len(returns))
        
        # Vectorized stability calculation
        returns_series = pd.Series(returns)
        vol_window_size = max(1, window // 4)
        
        # Calculate rolling volatility
        rolling_vol = self._vectorbt_rolling_operation(returns_series, "std", vol_window_size)
        
        # Calculate stability using rolling coefficient of variation
        vol_rolling_std = self._vectorbt_rolling_operation(rolling_vol, "std", window)
        vol_rolling_mean = self._vectorbt_rolling_operation(rolling_vol, "mean", window)
        
        vol_vol = vol_rolling_std / (vol_rolling_mean + 1e-8)
        stability = (1 - vol_vol).clip(0, 1)
        
        return stability.fillna(0).values
    
    def _calculate_regime_persistence_score(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate regime persistence score - OPTIMIZED VECTORIZED."""
        if len(returns) < window:
            return np.zeros(len(returns))
        
        # OPTIMIZED: Use vectorized persistence calculation
        returns_series = pd.Series(returns)
        vol_window_size = max(1, window // 4)
        
        # Calculate rolling volatility
        rolling_vol = self._vectorbt_rolling_operation(returns_series, "std", vol_window_size)
        
        # Vectorized persistence using rolling autocorrelation
        persistence = rolling_vol.rolling(window=window).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else 0,
            raw=False
        ).fillna(0)
        
        return persistence.values
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if self.vectorbt_optimizer:
            try:
                if operation == 'mean':
                    return self.vectorbt_optimizer.rolling_mean(data, window, **kwargs)
                elif operation == 'std':
                    return self.vectorbt_optimizer.rolling_std(data, window, **kwargs)
                elif operation == 'var':
                    return self.vectorbt_optimizer.rolling_var(data, window, **kwargs)
                elif operation == 'min':
                    return self.vectorbt_optimizer.rolling_min(data, window, **kwargs)
                elif operation == 'max':
                    return self.vectorbt_optimizer.rolling_max(data, window, **kwargs)
                elif operation == 'sum':
                    return self.vectorbt_optimizer.rolling_sum(data, window, **kwargs)
                else:
                    raise ValueError(f"Unsupported operation: {operation}")
            except Exception as e:
                tprint(f"VectorBT operation failed: {e}, using pandas fallback")
                return self._pandas_rolling_operation(data, operation, window, **kwargs)
        else:
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
