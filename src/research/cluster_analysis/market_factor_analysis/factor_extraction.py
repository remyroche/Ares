"""
Advanced Feature Engineering for Production HMM Pipeline

This module implements multi-horizon, leakage-safe feature engineering
specifically designed to enhance advanced Markov models (MSM + HSMM)
within a production walk-forward validation framework.

Key Features:
1. Multi-scale feature generation (5, 20, 60 bar horizons)
2. Structural break detection features (for MSM)
3. Duration persistence features (for HSMM)
4. Leakage-safe rolling statistics
5. Theme-based feature organization
6. Production-ready feature filtering
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
from dataclasses import dataclass
from enum import Enum
import logging

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from scipy import stats
from scipy.signal import find_peaks
import talib

from src.utils.logger import system_logger

try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False
    warnings.warn("ruptures not available - some structural break features limited")


class FeatureTheme(Enum):
    """Feature themes for organized feature engineering."""
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    FLOW = "flow"
    MICROSTRUCTURE = "microstructure"
    STRUCTURAL_BREAKS = "structural_breaks"  # NEW: For MSM
    DURATION_PERSISTENCE = "duration_persistence"  # NEW: For HSMM
    REGIME_TRANSITIONS = "regime_transitions"  # NEW: For both


@dataclass
class AdvancedFeatureConfig:
    """Configuration for advanced feature engineering."""
    # Multi-horizon settings
    horizons: List[int] = None  # [5, 20, 60]
    enable_daily_horizon: bool = False  # 126 bars for daily-ish
    
    # Theme enablement
    enable_traditional_themes: bool = True
    enable_structural_break_features: bool = True
    enable_duration_features: bool = True
    enable_regime_transition_features: bool = True
    
    # Rolling statistics settings
    rolling_window: int = 500  # For z-score normalization
    min_periods: int = 50
    clip_outliers: bool = True
    outlier_threshold: float = 10.0  # Standard deviations
    
    # Feature filtering
    variance_threshold: float = 1e-6
    correlation_threshold: float = 0.90
    enable_pca_compression: bool = True
    pca_variance_threshold: float = 0.85
    max_pca_components: int = 2
    
    # Advanced model specific settings
    break_detection_window: int = 100
    persistence_memory: int = 20
    transition_sensitivity: float = 0.8
    
    def __post_init__(self):
        if self.horizons is None:
            self.horizons = [5, 20, 60]
        if self.enable_daily_horizon:
            self.horizons.append(126)


class LeakageSafeRollingStats:
    """Leakage-safe rolling statistics calculator."""
    
    def __init__(self, window: int = 500, min_periods: int = 50):
        self.window = window
        self.min_periods = min_periods
        self.logger = system_logger.getChild('LeakageSafeRolling')
    
    def rolling_zscore(self, series: pd.Series, current_idx: Optional[int] = None) -> pd.Series:
        """
        Calculate rolling z-score ensuring no future data leakage.
        
        Args:
            series: Input time series
            current_idx: Current time index (for production use)
            
        Returns:
            Rolling z-score series
        """
        if current_idx is not None:
            # Production mode: only use data up to current_idx
            available_data = series.iloc[:current_idx + 1]
        else:
            # Backtest mode: use expanding window
            available_data = series
        
        # Calculate rolling mean and std with no lookahead
        rolling_mean = available_data.rolling(
            window=self.window, 
            min_periods=self.min_periods
        ).mean()
        
        rolling_std = available_data.rolling(
            window=self.window, 
            min_periods=self.min_periods
        ).std()
        
        # Calculate z-score
        zscore = (available_data - rolling_mean) / (rolling_std + 1e-8)
        
        return zscore
    
    def rolling_quantile(self, series: pd.Series, quantile: float, 
                        current_idx: Optional[int] = None) -> pd.Series:
        """Calculate rolling quantile with no lookahead."""
        if current_idx is not None:
            available_data = series.iloc[:current_idx + 1]
        else:
            available_data = series
        
        return available_data.rolling(
            window=self.window,
            min_periods=self.min_periods
        ).quantile(quantile)
    
    def rolling_correlation(self, series1: pd.Series, series2: pd.Series,
                           current_idx: Optional[int] = None) -> pd.Series:
        """Calculate rolling correlation with no lookahead."""
        if current_idx is not None:
            data1 = series1.iloc[:current_idx + 1]
            data2 = series2.iloc[:current_idx + 1]
        else:
            data1, data2 = series1, series2
        
        return data1.rolling(
            window=self.window,
            min_periods=self.min_periods
        ).corr(data2)


class AdvancedMarkovFeatureEngine:
    """
    Advanced feature engineering engine for production HMM pipeline.
    
    Generates multi-horizon features with specific enhancements for
    Markov-Switching Models and Hidden Semi-Markov Models.
    """
    
    def __init__(self, config: Optional[AdvancedFeatureConfig] = None):
        self.config = config or AdvancedFeatureConfig()
        self.logger = system_logger.getChild('AdvancedMarkovFeatureEngine')
        
        # Initialize components
        self.rolling_stats = LeakageSafeRollingStats(
            window=self.config.rolling_window,
            min_periods=self.config.min_periods
        )
        
        # Feature storage
        self.feature_metadata = {}
        self.scalers = {}
        self.pca_models = {}
        
    def generate_features(self, data: pd.DataFrame, 
                         current_idx: Optional[int] = None,
                         theme_filter: Optional[List[FeatureTheme]] = None) -> pd.DataFrame:
        """
        Generate comprehensive multi-horizon features.
        
        Args:
            data: OHLCV market data
            current_idx: Current time index (for production)
            theme_filter: Optional theme filter
            
        Returns:
            Feature matrix with all generated features
        """
        self.logger.info(f"🔧 Generating advanced features for {len(data)} observations")
        
        features = pd.DataFrame(index=data.index)
        
        # Determine which themes to generate
        enabled_themes = self._get_enabled_themes(theme_filter)
        
        # Generate features by theme
        for theme in enabled_themes:
            self.logger.debug(f"Generating {theme.value} features")
            
            if theme == FeatureTheme.TREND:
                theme_features = self._generate_trend_features(data, current_idx)
            elif theme == FeatureTheme.MOMENTUM:
                theme_features = self._generate_momentum_features(data, current_idx)
            elif theme == FeatureTheme.VOLATILITY:
                theme_features = self._generate_volatility_features(data, current_idx)
            elif theme == FeatureTheme.FLOW:
                theme_features = self._generate_flow_features(data, current_idx)
            elif theme == FeatureTheme.MICROSTRUCTURE:
                theme_features = self._generate_microstructure_features(data, current_idx)
            elif theme == FeatureTheme.STRUCTURAL_BREAKS:
                theme_features = self._generate_structural_break_features(data, current_idx)
            elif theme == FeatureTheme.DURATION_PERSISTENCE:
                theme_features = self._generate_duration_persistence_features(data, current_idx)
            elif theme == FeatureTheme.REGIME_TRANSITIONS:
                theme_features = self._generate_regime_transition_features(data, current_idx)
            else:
                continue
            
            # Add theme features to main feature set
            for col_name, col_data in theme_features.items():
                features[f"{theme.value}_{col_name}"] = col_data
        
        # Apply feature normalization
        features = self._normalize_features(features, current_idx)
        
        # Store metadata
        self.feature_metadata = {
            'n_features': len(features.columns),
            'themes_generated': [theme.value for theme in enabled_themes],
            'horizons_used': self.config.horizons,
            'generation_timestamp': pd.Timestamp.now().isoformat()
        }
        
        self.logger.info(f"✅ Generated {len(features.columns)} features across {len(enabled_themes)} themes")
        return features
    
    def _get_enabled_themes(self, theme_filter: Optional[List[FeatureTheme]]) -> List[FeatureTheme]:
        """Get list of enabled themes based on configuration."""
        if theme_filter is not None:
            return theme_filter
        
        enabled_themes = []
        
        if self.config.enable_traditional_themes:
            enabled_themes.extend([
                FeatureTheme.TREND,
                FeatureTheme.MOMENTUM,
                FeatureTheme.VOLATILITY,
                FeatureTheme.FLOW,
                FeatureTheme.MICROSTRUCTURE
            ])
        
        if self.config.enable_structural_break_features:
            enabled_themes.append(FeatureTheme.STRUCTURAL_BREAKS)
        
        if self.config.enable_duration_features:
            enabled_themes.append(FeatureTheme.DURATION_PERSISTENCE)
        
        if self.config.enable_regime_transition_features:
            enabled_themes.append(FeatureTheme.REGIME_TRANSITIONS)
        
        return enabled_themes
    
    def _generate_trend_features(self, data: pd.DataFrame, current_idx: Optional[int]) -> Dict[str, pd.Series]:
        """Generate trend/structure theme features."""
        features = {}
        
        for horizon in self.config.horizons:
            # Rolling slope and R²
            slope, r_squared = self._calculate_rolling_regression(data['close'], horizon, current_idx)
            features[f'slope_{horizon}'] = slope
            features[f'r_squared_{horizon}'] = r_squared
            
            # ADX (Average Directional Index)
            if len(data) > horizon:
                try:
                    adx = talib.ADX(data['high'].values, data['low'].values, data['close'].values, timeperiod=horizon)
                    features[f'adx_{horizon}'] = pd.Series(adx, index=data.index)
                except:
                    features[f'adx_{horizon}'] = pd.Series(0.0, index=data.index)
            
            # Hurst exponent (simplified)
            features[f'hurst_{horizon}'] = self._calculate_hurst_exponent(data['close'], horizon, current_idx)
            
            # Rolling skewness and kurtosis of returns
            returns = data['close'].pct_change()
            features[f'return_skew_{horizon}'] = returns.rolling(horizon, min_periods=horizon//2).skew()
            features[f'return_kurt_{horizon}'] = returns.rolling(horizon, min_periods=horizon//2).kurtosis()
            
            # Autocorrelation of absolute returns
            abs_returns = returns.abs()
            features[f'abs_return_autocorr_{horizon}'] = self._rolling_autocorr(abs_returns, 1, horizon, current_idx)
        
        return features
    
    def _generate_momentum_features(self, data: pd.DataFrame, current_idx: Optional[int]) -> Dict[str, pd.Series]:
        """Generate momentum theme features."""
        features = {}
        
        for horizon in self.config.horizons:
            # Basic momentum
            features[f'momentum_{horizon}'] = data['close'].pct_change(horizon)
            
            # RSI
            try:
                rsi = talib.RSI(data['close'].values, timeperiod=horizon)
                features[f'rsi_{horizon}'] = pd.Series(rsi, index=data.index)
            except:
                features[f'rsi_{horizon}'] = pd.Series(50.0, index=data.index)
            
            # MACD-like momentum
            if horizon >= 12:
                fast_ma = data['close'].rolling(max(1, horizon//3)).mean()
                slow_ma = data['close'].rolling(horizon).mean()
                features[f'macd_{horizon}'] = (fast_ma - slow_ma) / slow_ma
            
            # Price position in range
            rolling_max = data['high'].rolling(horizon, min_periods=horizon//2).max()
            rolling_min = data['low'].rolling(horizon, min_periods=horizon//2).min()
            features[f'price_position_{horizon}'] = (
                (data['close'] - rolling_min) / (rolling_max - rolling_min + 1e-8)
            )
        
        return features
    
    def _generate_volatility_features(self, data: pd.DataFrame, current_idx: Optional[int]) -> Dict[str, pd.Series]:
        """Generate volatility theme features."""
        features = {}
        
        returns = data['close'].pct_change()
        
        for horizon in self.config.horizons:
            # Realized volatility
            features[f'realized_vol_{horizon}'] = returns.rolling(horizon, min_periods=horizon//2).std()
            
            # Garman-Klass volatility (if OHLC available)
            if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                gk_vol = self._calculate_garman_klass_volatility(data, horizon)
                features[f'gk_vol_{horizon}'] = gk_vol
            
            # Volatility clustering
            vol = features[f'realized_vol_{horizon}']
            vol_ma = vol.rolling(horizon, min_periods=horizon//2).mean()
            features[f'vol_clustering_{horizon}'] = vol / (vol_ma + 1e-8)
            
            # Volatility skewness
            features[f'vol_skew_{horizon}'] = vol.rolling(horizon, min_periods=horizon//2).skew()
            
            # GARCH-like volatility persistence
            features[f'vol_persistence_{horizon}'] = self._rolling_autocorr(vol, 1, horizon//2, current_idx)
        
        return features
    
    def _generate_flow_features(self, data: pd.DataFrame, current_idx: Optional[int]) -> Dict[str, pd.Series]:
        """Generate liquidity/flow theme features."""
        features = {}
        
        for horizon in self.config.horizons:
            # Relative volume
            if 'volume' in data.columns:
                vol_ma = data['volume'].rolling(horizon, min_periods=horizon//2).mean()
                features[f'rel_volume_{horizon}'] = data['volume'] / (vol_ma + 1e-8)
                
                # Volume momentum
                features[f'volume_momentum_{horizon}'] = data['volume'].pct_change(horizon)
                
                # Volume-price correlation
                features[f'vol_price_corr_{horizon}'] = self.rolling_stats.rolling_correlation(
                    data['volume'], data['close'], current_idx
                )
            else:
                # Placeholder if volume not available
                features[f'rel_volume_{horizon}'] = pd.Series(1.0, index=data.index)
                features[f'volume_momentum_{horizon}'] = pd.Series(0.0, index=data.index)
                features[f'vol_price_corr_{horizon}'] = pd.Series(0.0, index=data.index)
            
            # VWAP deviation (simplified)
            if 'volume' in data.columns:
                vwap = (data['close'] * data['volume']).rolling(horizon).sum() / data['volume'].rolling(horizon).sum()
                features[f'vwap_dev_{horizon}'] = (data['close'] - vwap) / (vwap + 1e-8)
            else:
                features[f'vwap_dev_{horizon}'] = pd.Series(0.0, index=data.index)
        
        return features
    
    def _generate_microstructure_features(self, data: pd.DataFrame, current_idx: Optional[int]) -> Dict[str, pd.Series]:
        """Generate microstructure theme features."""
        features = {}
        
        for horizon in self.config.horizons:
            # High-Low spread proxy
            if all(col in data.columns for col in ['high', 'low']):
                spread_proxy = (data['high'] - data['low']) / data['close']
                features[f'spread_proxy_{horizon}'] = spread_proxy.rolling(horizon, min_periods=horizon//2).mean()
            else:
                features[f'spread_proxy_{horizon}'] = pd.Series(0.01, index=data.index)
            
            # Intrabar volatility
            if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                intrabar_range = (data['high'] - data['low']) / (data['open'] + 1e-8)
                features[f'intrabar_vol_{horizon}'] = intrabar_range.rolling(horizon, min_periods=horizon//2).mean()
                
                # Open-Close gap
                features[f'gap_{horizon}'] = (data['open'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8)
            else:
                features[f'intrabar_vol_{horizon}'] = pd.Series(0.02, index=data.index)
                features[f'gap_{horizon}'] = pd.Series(0.0, index=data.index)
        
        return features
    
    def _generate_structural_break_features(self, data: pd.DataFrame, current_idx: Optional[int]) -> Dict[str, pd.Series]:
        """Generate structural break detection features (for MSM)."""
        features = {}
        
        returns = data['close'].pct_change()
        
        for horizon in self.config.horizons:
            # Variance ratio test statistic
            features[f'variance_ratio_{horizon}'] = self._variance_ratio_statistic(returns, horizon, current_idx)
            
            # Rolling correlation stability
            features[f'corr_stability_{horizon}'] = self._correlation_stability(data, horizon, current_idx)
            
            # Parameter drift detection
            features[f'param_drift_{horizon}'] = self._parameter_drift_indicator(returns, horizon, current_idx)
            
            # Regime probability entropy proxy
            features[f'regime_entropy_{horizon}'] = self._regime_entropy_proxy(data, horizon, current_idx)
            
            # Structural change indicator
            features[f'structural_change_{horizon}'] = self._structural_change_indicator(returns, horizon, current_idx)
        
        return features
    
    def _generate_duration_persistence_features(self, data: pd.DataFrame, current_idx: Optional[int]) -> Dict[str, pd.Series]:
        """Generate duration persistence features (for HSMM)."""
        features = {}
        
        returns = data['close'].pct_change()
        
        for horizon in self.config.horizons:
            # Regime proxy autocorrelation
            vol_regime_proxy = returns.rolling(horizon//4).std()
            features[f'regime_autocorr_{horizon}'] = self._rolling_autocorr(
                vol_regime_proxy, 1, horizon, current_idx
            )
            
            # Volatility clustering intensity
            vol = returns.rolling(horizon//4, min_periods=2).std()
            vol_ma = vol.rolling(horizon, min_periods=horizon//2).mean()
            clustering_intensity = vol / (vol_ma + 1e-8)
            features[f'vol_clustering_intensity_{horizon}'] = clustering_intensity.rolling(horizon//2).std()
            
            # Trend persistence strength
            momentum = data['close'].pct_change(horizon//4)
            features[f'trend_persistence_{horizon}'] = self._rolling_autocorr(
                momentum, 1, horizon, current_idx
            )
            
            # Mean reversion speed
            features[f'mean_reversion_speed_{horizon}'] = self._mean_reversion_speed(returns, horizon, current_idx)
            
            # State duration proxy
            features[f'state_duration_proxy_{horizon}'] = self._state_duration_proxy(returns, horizon, current_idx)
        
        return features
    
    def _generate_regime_transition_features(self, data: pd.DataFrame, current_idx: Optional[int]) -> Dict[str, pd.Series]:
        """Generate regime transition features (for both MSM and HSMM)."""
        features = {}
        
        returns = data['close'].pct_change()
        
        for horizon in self.config.horizons:
            # Transition volatility
            features[f'transition_vol_{horizon}'] = self._transition_volatility_indicator(returns, horizon, current_idx)
            
            # Regime switching probability
            features[f'regime_switch_prob_{horizon}'] = self._regime_switching_probability(data, horizon, current_idx)
            
            # Transition timing indicator
            features[f'transition_timing_{horizon}'] = self._transition_timing_indicator(returns, horizon, current_idx)
            
            # Cross-asset correlation breakdown (if multiple assets available)
            features[f'correlation_breakdown_{horizon}'] = self._correlation_breakdown_indicator(data, horizon, current_idx)
        
        return features
    
    def _calculate_rolling_regression(self, series: pd.Series, window: int, current_idx: Optional[int]) -> Tuple[pd.Series, pd.Series]:
        """Calculate rolling linear regression slope and R²."""
        slopes = []
        r_squareds = []
        
        for i in range(len(series)):
            if current_idx is not None and i > current_idx:
                break
                
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            
            if end_idx - start_idx < window // 2:
                slopes.append(np.nan)
                r_squareds.append(np.nan)
                continue
            
            y = series.iloc[start_idx:end_idx].values
            x = np.arange(len(y))
            
            if len(y) > 1:
                slope, intercept, r_value, _, _ = stats.linregress(x, y)
                slopes.append(slope)
                r_squareds.append(r_value ** 2)
            else:
                slopes.append(np.nan)
                r_squareds.append(np.nan)
        
        slope_series = pd.Series(slopes, index=series.index[:len(slopes)])
        r_squared_series = pd.Series(r_squareds, index=series.index[:len(r_squareds)])
        
        return slope_series, r_squared_series
    
    def _calculate_hurst_exponent(self, series: pd.Series, window: int, current_idx: Optional[int]) -> pd.Series:
        """Calculate rolling Hurst exponent (simplified)."""
        hurst_values = []
        
        for i in range(len(series)):
            if current_idx is not None and i > current_idx:
                break
                
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            
            if end_idx - start_idx < window // 2:
                hurst_values.append(0.5)  # Default to random walk
                continue
            
            data_window = series.iloc[start_idx:end_idx].values
            
            try:
                # Simplified Hurst calculation using R/S statistic
                hurst = self._rs_hurst(data_window)
                hurst_values.append(hurst)
            except:
                hurst_values.append(0.5)
        
        return pd.Series(hurst_values, index=series.index[:len(hurst_values)])
    
    def _rs_hurst(self, data: np.ndarray) -> float:
        """Calculate Hurst exponent using R/S statistic."""
        n = len(data)
        if n < 10:
            return 0.5
        
        # Calculate mean-adjusted cumulative deviations
        mean_data = np.mean(data)
        deviations = np.cumsum(data - mean_data)
        
        # Calculate range and standard deviation
        R = np.max(deviations) - np.min(deviations)
        S = np.std(data)
        
        if S == 0:
            return 0.5
        
        # R/S statistic
        rs = R / S
        
        # Hurst exponent approximation
        hurst = np.log(rs) / np.log(n)
        
        # Clip to reasonable range
        return np.clip(hurst, 0.0, 1.0)
    
    def _rolling_autocorr(self, series: pd.Series, lag: int, window: int, current_idx: Optional[int]) -> pd.Series:
        """Calculate rolling autocorrelation."""
        autocorrs = []
        
        for i in range(len(series)):
            if current_idx is not None and i > current_idx:
                break
                
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            
            if end_idx - start_idx < window // 2:
                autocorrs.append(0.0)
                continue
            
            data_window = series.iloc[start_idx:end_idx].dropna()
            
            if len(data_window) > lag:
                try:
                    autocorr = data_window.autocorr(lag=lag)
                    autocorrs.append(autocorr if not np.isnan(autocorr) else 0.0)
                except:
                    autocorrs.append(0.0)
            else:
                autocorrs.append(0.0)
        
        return pd.Series(autocorrs, index=series.index[:len(autocorrs)])
    
    def _calculate_garman_klass_volatility(self, data: pd.DataFrame, window: int) -> pd.Series:
        """Calculate Garman-Klass volatility estimator."""
        log_hl = np.log(data['high'] / data['low'])
        log_co = np.log(data['close'] / data['open'])
        
        gk = 0.5 * log_hl**2 - (2*np.log(2) - 1) * log_co**2
        
        return gk.rolling(window, min_periods=window//2).mean().apply(np.sqrt)
    
    def _variance_ratio_statistic(self, returns: pd.Series, window: int, current_idx: Optional[int]) -> pd.Series:
        """Calculate variance ratio test statistic for structural breaks."""
        vr_stats = []
        
        for i in range(len(returns)):
            if current_idx is not None and i > current_idx:
                break
                
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            
            if end_idx - start_idx < window // 2:
                vr_stats.append(1.0)
                continue
            
            data_window = returns.iloc[start_idx:end_idx].dropna()
            
            if len(data_window) < 10:
                vr_stats.append(1.0)
                continue
            
            # Split window in half
            mid_point = len(data_window) // 2
            first_half = data_window.iloc[:mid_point]
            second_half = data_window.iloc[mid_point:]
            
            # Calculate variance ratio
            var1 = first_half.var()
            var2 = second_half.var()
            
            if var1 > 0 and var2 > 0:
                vr_stat = max(var1, var2) / min(var1, var2)
            else:
                vr_stat = 1.0
            
            vr_stats.append(vr_stat)
        
        return pd.Series(vr_stats, index=returns.index[:len(vr_stats)])
    
    def _correlation_stability(self, data: pd.DataFrame, window: int, current_idx: Optional[int]) -> pd.Series:
        """Calculate correlation stability indicator."""
        if 'volume' not in data.columns:
            return pd.Series(1.0, index=data.index)
        
        stabilities = []
        returns = data['close'].pct_change()
        
        for i in range(len(data)):
            if current_idx is not None and i > current_idx:
                break
                
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            
            if end_idx - start_idx < window // 2:
                stabilities.append(1.0)
                continue
            
            # Calculate correlation in first and second half of window
            data_window = data.iloc[start_idx:end_idx]
            returns_window = returns.iloc[start_idx:end_idx]
            volume_window = data_window['volume']
            
            mid_point = len(data_window) // 2
            
            try:
                corr1 = returns_window.iloc[:mid_point].corr(volume_window.iloc[:mid_point])
                corr2 = returns_window.iloc[mid_point:].corr(volume_window.iloc[mid_point:])
                
                if not (np.isnan(corr1) or np.isnan(corr2)):
                    stability = 1.0 - abs(corr1 - corr2)
                else:
                    stability = 1.0
            except:
                stability = 1.0
            
            stabilities.append(stability)
        
        return pd.Series(stabilities, index=data.index[:len(stabilities)])
    
    def _parameter_drift_indicator(self, returns: pd.Series, window: int, current_idx: Optional[int]) -> pd.Series:
        """Calculate parameter drift indicator."""
        drift_indicators = []
        
        for i in range(len(returns)):
            if current_idx is not None and i > current_idx:
                break
                
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            
            if end_idx - start_idx < window // 2:
                drift_indicators.append(0.0)
                continue
            
            data_window = returns.iloc[start_idx:end_idx].dropna()
            
            if len(data_window) < 10:
                drift_indicators.append(0.0)
                continue
            
            # Compare mean and variance in first vs second half
            mid_point = len(data_window) // 2
            first_half = data_window.iloc[:mid_point]
            second_half = data_window.iloc[mid_point:]
            
            mean_drift = abs(first_half.mean() - second_half.mean())
            var_drift = abs(first_half.var() - second_half.var())
            
            # Normalize by pooled standard deviation
            pooled_std = np.sqrt((first_half.var() + second_half.var()) / 2)
            
            if pooled_std > 0:
                drift_indicator = (mean_drift + var_drift) / pooled_std
            else:
                drift_indicator = 0.0
            
            drift_indicators.append(drift_indicator)
        
        return pd.Series(drift_indicators, index=returns.index[:len(drift_indicators)])
    
    def _regime_entropy_proxy(self, data: pd.DataFrame, window: int, current_idx: Optional[int]) -> pd.Series:
        """Calculate regime entropy proxy."""
        entropies = []
        returns = data['close'].pct_change()
        
        for i in range(len(data)):
            if current_idx is not None and i > current_idx:
                break
                
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            
            if end_idx - start_idx < window // 2:
                entropies.append(0.5)
                continue
            
            returns_window = returns.iloc[start_idx:end_idx].dropna()
            
            if len(returns_window) < 5:
                entropies.append(0.5)
                continue
            
            # Create simple regime proxy based on volatility quantiles
            vol = returns_window.rolling(5, min_periods=2).std()
            vol_quantiles = vol.quantile([0.33, 0.67])
            
            # Assign regime labels
            regime_labels = np.zeros(len(vol))
            regime_labels[vol <= vol_quantiles.iloc[0]] = 0  # Low vol
            regime_labels[(vol > vol_quantiles.iloc[0]) & (vol <= vol_quantiles.iloc[1])] = 1  # Med vol
            regime_labels[vol > vol_quantiles.iloc[1]] = 2  # High vol
            
            # Calculate entropy
            unique, counts = np.unique(regime_labels, return_counts=True)
            probabilities = counts / len(regime_labels)
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
            
            # Normalize entropy (max entropy for 3 states is log(3))
            normalized_entropy = entropy / np.log(3)
            entropies.append(normalized_entropy)
        
        return pd.Series(entropies, index=data.index[:len(entropies)])
    
    def _structural_change_indicator(self, returns: pd.Series, window: int, current_idx: Optional[int]) -> pd.Series:
        """Calculate structural change indicator using CUSUM-like statistic."""
        change_indicators = []
        
        for i in range(len(returns)):
            if current_idx is not None and i > current_idx:
                break
                
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            
            if end_idx - start_idx < window // 2:
                change_indicators.append(0.0)
                continue
            
            data_window = returns.iloc[start_idx:end_idx].dropna()
            
            if len(data_window) < 10:
                change_indicators.append(0.0)
                continue
            
            # CUSUM-like statistic
            mean_return = data_window.mean()
            cumsum = np.cumsum(data_window - mean_return)
            
            # Normalize by standard deviation
            std_return = data_window.std()
            if std_return > 0:
                normalized_cumsum = cumsum / std_return
                change_indicator = np.max(np.abs(normalized_cumsum))
            else:
                change_indicator = 0.0
            
            change_indicators.append(change_indicator)
        
        return pd.Series(change_indicators, index=returns.index[:len(change_indicators)])
    
    def _mean_reversion_speed(self, returns: pd.Series, window: int, current_idx: Optional[int]) -> pd.Series:
        """Calculate mean reversion speed indicator."""
        reversion_speeds = []
        
        for i in range(len(returns)):
            if current_idx is not None and i > current_idx:
                break
                
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            
            if end_idx - start_idx < window // 2:
                reversion_speeds.append(0.0)
                continue
            
            data_window = returns.iloc[start_idx:end_idx].dropna()
            
            if len(data_window) < 3:
                reversion_speeds.append(0.0)
                continue
            
            # Simple AR(1) coefficient as proxy for mean reversion
            y = data_window.iloc[1:].values
            x = data_window.iloc[:-1].values
            
            if len(x) > 0 and np.std(x) > 0:
                correlation = np.corrcoef(x, y)[0, 1]
                # Mean reversion speed is negative of correlation
                reversion_speed = max(0.0, -correlation)
            else:
                reversion_speed = 0.0
            
            reversion_speeds.append(reversion_speed)
        
        return pd.Series(reversion_speeds, index=returns.index[:len(reversion_speeds)])
    
    def _state_duration_proxy(self, returns: pd.Series, window: int, current_idx: Optional[int]) -> pd.Series:
        """Calculate state duration proxy."""
        duration_proxies = []
        
        for i in range(len(returns)):
            if current_idx is not None and i > current_idx:
                break
                
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            
            if end_idx - start_idx < window // 2:
                duration_proxies.append(1.0)
                continue
            
            returns_window = returns.iloc[start_idx:end_idx].dropna()
            
            if len(returns_window) < 5:
                duration_proxies.append(1.0)
                continue
            
            # Count runs of similar volatility
            vol = returns_window.rolling(3, min_periods=2).std()
            vol_median = vol.median()
            
            # Create binary series (high/low volatility)
            high_vol = (vol > vol_median).astype(int)
            
            # Count run lengths
            runs = []
            current_run = 1
            
            for j in range(1, len(high_vol)):
                if high_vol.iloc[j] == high_vol.iloc[j-1]:
                    current_run += 1
                else:
                    runs.append(current_run)
                    current_run = 1
            runs.append(current_run)
            
            # Average run length as duration proxy
            avg_duration = np.mean(runs) if runs else 1.0
            duration_proxies.append(avg_duration)
        
        return pd.Series(duration_proxies, index=returns.index[:len(duration_proxies)])
    
    def _transition_volatility_indicator(self, returns: pd.Series, window: int, current_idx: Optional[int]) -> pd.Series:
        """Calculate transition volatility indicator."""
        transition_vols = []
        
        for i in range(len(returns)):
            if current_idx is not None and i > current_idx:
                break
                
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            
            if end_idx - start_idx < window // 2:
                transition_vols.append(0.0)
                continue
            
            returns_window = returns.iloc[start_idx:end_idx].dropna()
            
            if len(returns_window) < 10:
                transition_vols.append(0.0)
                continue
            
            # Detect potential transition points using rolling volatility changes
            vol = returns_window.rolling(5, min_periods=2).std()
            vol_changes = vol.diff().abs()
            
            # High volatility changes indicate potential transitions
            transition_vol = vol_changes.mean()
            transition_vols.append(transition_vol)
        
        return pd.Series(transition_vols, index=returns.index[:len(transition_vols)])
    
    def _regime_switching_probability(self, data: pd.DataFrame, window: int, current_idx: Optional[int]) -> pd.Series:
        """Calculate regime switching probability indicator."""
        switch_probs = []
        returns = data['close'].pct_change()
        
        for i in range(len(data)):
            if current_idx is not None and i > current_idx:
                break
                
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            
            if end_idx - start_idx < window // 2:
                switch_probs.append(0.1)
                continue
            
            returns_window = returns.iloc[start_idx:end_idx].dropna()
            
            if len(returns_window) < 10:
                switch_probs.append(0.1)
                continue
            
            # Simple regime switching probability based on volatility clustering
            vol = returns_window.rolling(5, min_periods=2).std()
            vol_changes = vol.diff().abs()
            
            # Probability of switching based on volatility change distribution
            vol_change_threshold = vol_changes.quantile(0.8)
            recent_vol_change = vol_changes.iloc[-1] if len(vol_changes) > 0 else 0
            
            # Probability increases with recent volatility changes
            switch_prob = min(0.5, recent_vol_change / (vol_change_threshold + 1e-8) * 0.2)
            switch_probs.append(switch_prob)
        
        return pd.Series(switch_probs, index=data.index[:len(switch_probs)])
    
    def _transition_timing_indicator(self, returns: pd.Series, window: int, current_idx: Optional[int]) -> pd.Series:
        """Calculate transition timing indicator."""
        timing_indicators = []
        
        for i in range(len(returns)):
            if current_idx is not None and i > current_idx:
                break
                
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            
            if end_idx - start_idx < window // 2:
                timing_indicators.append(0.0)
                continue
            
            returns_window = returns.iloc[start_idx:end_idx].dropna()
            
            if len(returns_window) < 10:
                timing_indicators.append(0.0)
                continue
            
            # Time since last significant volatility change
            vol = returns_window.rolling(5, min_periods=2).std()
            vol_changes = vol.diff().abs()
            vol_threshold = vol_changes.quantile(0.9)
            
            # Find last significant change
            significant_changes = vol_changes > vol_threshold
            if significant_changes.any():
                last_change_idx = significant_changes[::-1].idxmax()
                time_since_change = len(returns_window) - (returns_window.index.get_loc(last_change_idx) + 1)
                
                # Normalize by window size
                timing_indicator = time_since_change / len(returns_window)
            else:
                timing_indicator = 1.0  # Long time since change
            
            timing_indicators.append(timing_indicator)
        
        return pd.Series(timing_indicators, index=returns.index[:len(timing_indicators)])
    
    def _correlation_breakdown_indicator(self, data: pd.DataFrame, window: int, current_idx: Optional[int]) -> pd.Series:
        """Calculate correlation breakdown indicator."""
        # Placeholder - would need multiple assets for real implementation
        return pd.Series(0.0, index=data.index)
    
    def _normalize_features(self, features: pd.DataFrame, current_idx: Optional[int]) -> pd.DataFrame:
        """Apply feature normalization."""
        normalized_features = features.copy()
        
        for col in features.columns:
            # Apply rolling z-score normalization
            normalized_features[col] = self.rolling_stats.rolling_zscore(
                features[col], current_idx
            )
            
            # Clip outliers if enabled
            if self.config.clip_outliers:
                normalized_features[col] = normalized_features[col].clip(
                    -self.config.outlier_threshold,
                    self.config.outlier_threshold
                )
        
        return normalized_features.fillna(0.0)
    
    def apply_feature_filtering(self, features: pd.DataFrame, 
                              target: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply production-ready feature filtering.
        
        Args:
            features: Input feature matrix
            target: Optional target for supervised filtering
            
        Returns:
            Filtered features and filtering metadata
        """
        self.logger.info(f"🔧 Applying feature filtering to {len(features.columns)} features")
        
        filtering_metadata = {
            'original_features': len(features.columns),
            'filtering_steps': []
        }
        
        filtered_features = features.copy()
        
        # Step 1: Remove near-zero variance features
        if self.config.variance_threshold > 0:
            variance_selector = VarianceThreshold(threshold=self.config.variance_threshold)
            selected_features = variance_selector.fit_transform(filtered_features.fillna(0))
            
            # Get selected feature names
            selected_mask = variance_selector.get_support()
            selected_columns = filtered_features.columns[selected_mask]
            
            filtered_features = pd.DataFrame(
                selected_features, 
                index=filtered_features.index,
                columns=selected_columns
            )
            
            filtering_metadata['filtering_steps'].append({
                'step': 'variance_threshold',
                'features_removed': len(features.columns) - len(selected_columns),
                'features_remaining': len(selected_columns)
            })
        
        # Step 2: Correlation-based filtering
        if self.config.correlation_threshold < 1.0:
            filtered_features = self._correlation_filtering(filtered_features, filtering_metadata)
        
        # Step 3: PCA compression per theme (if enabled)
        if self.config.enable_pca_compression:
            filtered_features = self._apply_pca_compression(filtered_features, filtering_metadata)
        
        filtering_metadata['final_features'] = len(filtered_features.columns)
        filtering_metadata['reduction_ratio'] = (
            len(features.columns) - len(filtered_features.columns)
        ) / len(features.columns)
        
        self.logger.info(f"✅ Feature filtering completed: {len(features.columns)} → {len(filtered_features.columns)} features")
        
        return filtered_features, filtering_metadata
    
    def _correlation_filtering(self, features: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """Apply correlation-based feature filtering."""
        # Calculate correlation matrix
        corr_matrix = features.corr().abs()
        
        # Find highly correlated pairs
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop
        to_drop = [
            column for column in upper_triangle.columns 
            if any(upper_triangle[column] > self.config.correlation_threshold)
        ]
        
        # Drop highly correlated features
        filtered_features = features.drop(columns=to_drop)
        
        metadata['filtering_steps'].append({
            'step': 'correlation_filtering',
            'threshold': self.config.correlation_threshold,
            'features_removed': len(to_drop),
            'features_remaining': len(filtered_features.columns)
        })
        
        return filtered_features
    
    def _apply_pca_compression(self, features: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """Apply PCA compression per theme."""
        compressed_features = pd.DataFrame(index=features.index)
        
        # Group features by theme
        theme_groups = {}
        for col in features.columns:
            theme = col.split('_')[0] if '_' in col else 'other'
            if theme not in theme_groups:
                theme_groups[theme] = []
            theme_groups[theme].append(col)
        
        pca_metadata = {}
        
        for theme, theme_features in theme_groups.items():
            if len(theme_features) <= self.config.max_pca_components:
                # Don't compress if fewer features than max components
                for feature in theme_features:
                    compressed_features[feature] = features[feature]
                continue
            
            # Apply PCA to theme features
            theme_data = features[theme_features].fillna(0)
            
            # Determine number of components
            pca = PCA()
            pca.fit(theme_data)
            
            # Find components explaining desired variance
            cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
            n_components = min(
                self.config.max_pca_components,
                np.argmax(cumsum_variance >= self.config.pca_variance_threshold) + 1
            )
            n_components = max(1, n_components)  # At least 1 component
            
            # Apply PCA with selected components
            pca_final = PCA(n_components=n_components)
            pca_features = pca_final.fit_transform(theme_data)
            
            # Add PCA features
            for i in range(n_components):
                compressed_features[f"{theme}_pca_{i}"] = pca_features[:, i]
            
            # Store PCA model for production
            self.pca_models[theme] = pca_final
            
            pca_metadata[theme] = {
                'original_features': len(theme_features),
                'pca_components': n_components,
                'explained_variance': float(cumsum_variance[n_components - 1])
            }
        
        metadata['filtering_steps'].append({
            'step': 'pca_compression',
            'theme_compressions': pca_metadata,
            'total_features_before': len(features.columns),
            'total_features_after': len(compressed_features.columns)
        })
        
        return compressed_features
    
    def get_production_artifacts(self) -> Dict[str, Any]:
        """Get production artifacts for deployment."""
        return {
            'config': self.config.__dict__,
            'feature_metadata': self.feature_metadata,
            'rolling_stats_config': {
                'window': self.rolling_stats.window,
                'min_periods': self.rolling_stats.min_periods
            },
            'scalers': self.scalers,
            'pca_models': {
                theme: {
                    'n_components': model.n_components_,
                    'explained_variance_ratio': model.explained_variance_ratio_.tolist(),
                    'components': model.components_.tolist(),
                    'mean': model.mean_.tolist()
                }
                for theme, model in self.pca_models.items()
            },
            'version': '1.0.0',
            'generation_timestamp': pd.Timestamp.now().isoformat()
        }


# Example usage and testing
if __name__ == "__main__":
    # Generate synthetic market data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='H')
    n_obs = len(dates)
    
    # Create realistic market data with regime structure
    prices = np.zeros(n_obs)
    prices[0] = 100.0
    
    for i in range(1, n_obs):
        # Regime-switching volatility
        if i < n_obs // 3:
            vol = 0.015  # Low vol regime
        elif i < 2 * n_obs // 3:
            vol = 0.035  # High vol regime
        else:
            vol = 0.020  # Medium vol regime
        
        ret = np.random.normal(0, vol)
        prices[i] = prices[i-1] * (1 + ret)
    
    # Create OHLCV data
    test_data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, n_obs)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.003, n_obs))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.003, n_obs))),
        'close': prices,
        'volume': np.random.lognormal(15, 0.3, n_obs)
    }, index=dates)
    
    print("🧪 Testing Advanced Feature Engineering")
    print(f"📊 Test data: {len(test_data)} observations")
    
    # Initialize feature engine
    config = AdvancedFeatureConfig(
        horizons=[5, 20, 60],
        enable_structural_break_features=True,
        enable_duration_features=True,
        enable_regime_transition_features=True
    )
    
    feature_engine = AdvancedMarkovFeatureEngine(config)
    
    # Generate features
    features = feature_engine.generate_features(test_data)
    
    print(f"\n✅ Generated {len(features.columns)} features")
    print(f"📈 Feature themes: {feature_engine.feature_metadata['themes_generated']}")
    
    # Apply filtering
    filtered_features, filtering_metadata = feature_engine.apply_feature_filtering(features)
    
    print(f"\n🔧 Feature filtering: {len(features.columns)} → {len(filtered_features.columns)}")
    print(f"📉 Reduction ratio: {filtering_metadata['reduction_ratio']:.2%}")
    
    # Show sample features by theme
    print(f"\n📊 Sample features by theme:")
    for theme in ['trend', 'structural_breaks', 'duration_persistence']:
        theme_cols = [col for col in filtered_features.columns if col.startswith(theme)]
        if theme_cols:
            print(f"  {theme}: {len(theme_cols)} features")
            print(f"    Examples: {theme_cols[:3]}")
    
    print(f"\n🎯 Advanced feature engineering completed!")
    print(f"   Ready for integration with advanced Markov models")