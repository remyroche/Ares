"""
import warnings
Utility functions for regime discovery feature engineering.

This module contains vectorized calculations and helper functions
for regime discovery features to reduce complexity in the main class.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

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

except ImportError:
    
    cp = None

class RegimeFeatureUtils:
    """Utility class for regime feature calculations."""
    
    @staticmethod
    def calculate_regime_change_probability_vectorized(series: pd.Series, window: int = 10) -> np.ndarray:
        """Calculate regime change probability using vectorized operations."""
        rolling_mean = series.rolling(window).mean()
        rolling_std = series.rolling(window).std()
        z_scores = (series - rolling_mean) / rolling_std
        regime_change_prob = np.abs(z_scores).rolling(window).mean()
        return regime_change_prob.fillna(0).values
    
    @staticmethod
    def calculate_regime_persistence_vectorized(series: pd.Series, min_duration: int = 5) -> np.ndarray:
        """Calculate regime persistence using vectorized operations."""
        rolling_mean = series.rolling(min_duration).mean()
        regime_changes = (np.abs(series - rolling_mean) > rolling_mean.rolling(min_duration).std()).astype(int)
        
        persistence = np.zeros(len(series))
        current_persistence = 0
        
        for i in range(len(series)):
            if regime_changes.iloc[i] == 0:
                current_persistence += 1
            else:
                current_persistence = 0
            persistence[i] = current_persistence
        
        return persistence
    
    @staticmethod
    def calculate_regime_transition_timing_vectorized(df: pd.DataFrame, volatility_20: pd.Series, 
                                                   volume_mean_20: pd.Series, momentum_10: pd.Series) -> np.ndarray:
        """Calculate regime transition timing using vectorized operations."""
        vol_norm = (volatility_20 - volatility_20.rolling(50).mean()) / volatility_20.rolling(50).std()
        vol_vol_norm = (volume_mean_20 - volume_mean_20.rolling(50).mean()) / volume_mean_20.rolling(50).std()
        mom_norm = (momentum_10 - momentum_10.rolling(50).mean()) / momentum_10.rolling(50).std()
        
        transition_timing = (vol_norm + vol_vol_norm + mom_norm) / 3
        return transition_timing.fillna(0).values
    
    @staticmethod
    def calculate_regime_stability_vectorized(series: pd.Series, window: int = 20) -> np.ndarray:
        """Calculate regime stability using vectorized operations."""
        rolling_mean = series.rolling(window).mean()
        rolling_std = series.rolling(window).std()
        cv = rolling_std / rolling_mean
        stability = 1 / (1 + cv)
        return stability.fillna(0).values
    
    @staticmethod
    def classify_regime_vectorized(series: pd.Series, low_quantile: float = 0.33, 
                                 high_quantile: float = 0.67) -> np.ndarray:
        """Classify series into regimes using quantiles."""
        low_threshold = series.rolling(100).quantile(low_quantile)
        high_threshold = series.rolling(100).quantile(high_quantile)
        
        regime = np.ones(len(series))  # Low regime
        regime[series > high_threshold] = 3  # High regime
        regime[(series > low_threshold) & (series <= high_threshold)] = 2  # Medium regime
        
        return regime
    
    @staticmethod
    def calculate_order_flow_imbalance(df: pd.DataFrame) -> np.ndarray:
        """Calculate order flow imbalance proxy."""
        hl_range = df['high'] - df['low']
        close_position = (df['close'] - df['low']) / (hl_range + 1e-8)
        return (close_position - 0.5) * 2
    
    @staticmethod
    def calculate_volume_profile_features(df: pd.DataFrame, window: int = 20) -> Dict[str, np.ndarray]:
        """Calculate volume profile features."""
        vwap = (df['high'] + df['low'] + df['close']) / 3
        price_range = df['high'] - df['low']
        price_position = (vwap - df['low']) / (price_range + 1e-8)
        
        volume_weighted = price_position * df['volume']
        return {
            'volume_profile_skew': volume_weighted.rolling(window).skew().fillna(0).values,
            'volume_profile_kurtosis': volume_weighted.rolling(window).kurt().fillna(0).values
        }
    
    @staticmethod
    def calculate_price_impact_features(price_changes: pd.Series, volume_changes: pd.Series, 
                                      window: int = 20) -> Dict[str, np.ndarray]:
        """Calculate price impact features."""
        price_impact = price_changes / (volume_changes + 1e-8)
        return {
            'price_impact_ratio': price_impact.fillna(0).values,
            'price_impact_volatility': price_impact.rolling(window).std().fillna(0).values
        }
    
    @staticmethod
    def calculate_liquidity_features(df: pd.DataFrame, window: int = 50) -> Dict[str, np.ndarray]:
        """Calculate liquidity-related features."""
        hl_range = df['high'] - df['low']
        spread_proxy = hl_range / df['close']
        market_depth_proxy = df['volume'] / (hl_range + 1e-8)
        
        spread_norm = (spread_proxy - spread_proxy.rolling(window).mean()) / spread_proxy.rolling(window).std()
        depth_norm = (market_depth_proxy - market_depth_proxy.rolling(window).mean()) / market_depth_proxy.rolling(window).std()
        
        return {
            'spread_proxy': spread_proxy.fillna(0).values,
            'market_depth_proxy': market_depth_proxy.fillna(0).values,
            'liquidity_regime_indicator': (depth_norm - spread_norm).fillna(0).values
        }
    
    @staticmethod
    def calculate_temporal_features(df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate temporal features."""
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            return {}
        
        hour = df.index.hour
        day_of_week = df.index.dayofweek
        
        return {
            'hour_of_day': hour,
            'day_of_week': day_of_week,
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'day_sin': np.sin(2 * np.pi * day_of_week / 7),
            'day_cos': np.cos(2 * np.pi * day_of_week / 7),
            'session_regime_indicator': np.where(hour < 8, 1, np.where(hour < 16, 2, 3))
        }
    
    @staticmethod
    def calculate_volatility_features(price_changes: pd.Series) -> Dict[str, np.ndarray]:
        """Calculate volatility-related features."""
        volatility_5 = price_changes.rolling(5).std()
        volatility_10 = price_changes.rolling(10).std()
        volatility_20 = price_changes.rolling(20).std()
        
        vol_of_vol_20 = volatility_20.rolling(20).std()
        vol_of_vol_50 = volatility_20.rolling(50).std()
        
        volatility_regime = RegimeFeatureUtils.classify_regime_vectorized(volatility_20)
        
        return {
            'volatility_5': volatility_5.fillna(0).values,
            'volatility_10': volatility_10.fillna(0).values,
            'volatility_20': volatility_20.fillna(0).values,
            'vol_of_vol_20': vol_of_vol_20.fillna(0).values,
            'vol_of_vol_50': vol_of_vol_50.fillna(0).values,
            'volatility_regime': volatility_regime,
            'volatility_clustering': volatility_20.rolling(50).apply(
                lambda x: x.autocorr(lag = 1) if len(x) > 1 else 0
            ).fillna(0).values,
            'volatility_persistence': volatility_20.rolling(50).apply(
                lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 1 else 0
            ).fillna(0).values
        }
    
    @staticmethod
    def calculate_volume_features(df: pd.DataFrame, volume_changes: pd.Series, 
                                momentum_5: pd.Series) -> Dict[str, np.ndarray]:
        """Calculate volume-related features."""
        volume_regime = RegimeFeatureUtils.classify_regime_vectorized(df['volume'])
        
        volume_momentum_interaction = volume_changes * momentum_5
        volume_price_divergence = (momentum_5 * volume_changes < 0).astype(int)
        
        rolling_mean = df['volume'].rolling(20).mean()
        rolling_std = df['volume'].rolling(20).std()
        volume_spike_indicator = (df['volume'] > rolling_mean + 2 * rolling_std).astype(int)
        
        return {
            'volume_regime': volume_regime,
            'volume_momentum_interaction': volume_momentum_interaction.fillna(0).values,
            'volume_price_divergence': volume_price_divergence.fillna(0).values,
            'volume_spike_indicator': volume_spike_indicator.fillna(0).values
        }
    
    @staticmethod
    def calculate_price_action_features(df: pd.DataFrame, momentum_10: pd.Series, 
                                      volatility_20: pd.Series) -> Dict[str, np.ndarray]:
        """Calculate price action features."""
        # Normalize indicators
        mom_norm = (momentum_10 - momentum_10.rolling(50).mean()) / momentum_10.rolling(50).std()
        vol_norm = (volatility_20 - volatility_20.rolling(50).mean()) / volatility_20.rolling(50).std()
        
        range_size = (df['high'] - df['low']) / df['close']
        range_norm = (range_size - range_size.rolling(50).mean()) / range_size.rolling(50).std()
        
        # Price action regime classification
        price_action_regime = np.ones(len(df))  # Trending
        price_action_regime[(np.abs(mom_norm) < 0.5) & (vol_norm < 0.5)] = 2  # Consolidation
        price_action_regime[(vol_norm > 1) | (range_norm > 1)] = 3  # High volatility
        
        # Support/resistance proximity
        high_20 = df['high'].rolling(20).max()
        low_20 = df['low'].rolling(20).min()
        resistance_proximity = (high_20 - df['close']) / df['close']
        support_proximity = (df['close'] - low_20) / df['close']
        sr_proximity = 1 / (1 + np.minimum(resistance_proximity, support_proximity))
        
        # Momentum regime
        momentum_regime = np.ones(len(df))  # Bullish
        momentum_regime[momentum_10 < -0.01] = 3  # Bearish
        momentum_regime[(momentum_10 >= -0.01) & (momentum_10 <= 0.01)] = 2  # Neutral
        
        return {
            'price_action_regime': price_action_regime,
            'sr_proximity': sr_proximity.fillna(0).values,
            'momentum_regime': momentum_regime
        }
    
    @staticmethod
    def calculate_regime_strength_features(volatility_20: pd.Series, volume_mean_20: pd.Series, 
                                         momentum_10: pd.Series) -> Dict[str, np.ndarray]:
        """Calculate regime strength features."""
        vol_of_vol = volatility_20.rolling(20).std()
        regime_strength_volatility = 1 / (1 + vol_of_vol)
        
        volume_consistency = 1 / (1 + volume_mean_20.rolling(20).std() / volume_mean_20.rolling(20).mean())
        regime_strength_volume = volume_consistency
        
        momentum_consistency = 1 / (1 + momentum_10.rolling(20).std())
        regime_strength_momentum = momentum_consistency
        
        confidence = (regime_strength_volatility + regime_strength_volume + regime_strength_momentum) / 3
        
        return {
            'regime_strength_volatility': regime_strength_volatility.fillna(0).values,
            'regime_strength_volume': regime_strength_volume.fillna(0).values,
            'regime_strength_momentum': regime_strength_momentum.fillna(0).values,
            'regime_confidence_score': confidence.fillna(0).values
        }
    
    @staticmethod
    def calculate_regime_change_warning_features(volatility_20: pd.Series, volume_mean_20: pd.Series, 
                                               momentum_10: pd.Series, regime_confidence_score: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate regime change warning features."""
        vol_change_prob = RegimeFeatureUtils.calculate_regime_change_probability_vectorized(volatility_20, 10)
        vol_change_prob_vol = RegimeFeatureUtils.calculate_regime_change_probability_vectorized(volume_mean_20, 10)
        mom_change_prob = RegimeFeatureUtils.calculate_regime_change_probability_vectorized(momentum_10, 10)
        
        early_warning = (vol_change_prob + vol_change_prob_vol + mom_change_prob) / 3
        
        regime_strength = pd.Series(regime_confidence_score)
        weakening = np.diff(regime_strength) < 0
        regime_weakening_indicator = np.concatenate([[0], weakening.astype(int)])
        
        readiness = early_warning * (1 + regime_weakening_indicator)
        
        return {
            'regime_change_early_warning': early_warning,
            'regime_weakening_indicator': regime_weakening_indicator,
            'regime_transition_readiness': readiness
        }

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
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
    
    def _vectorbt_apply_operation(self, data: pd.Series, func, 
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)
        
        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
