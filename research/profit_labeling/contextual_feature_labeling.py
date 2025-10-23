"""
Contextual Feature Engineering for Multi-Horizon Profit Labeling

This module provides advanced feature engineering that incorporates rich market
context into labeling decisions. It goes beyond simple OHLCV data to include
technical indicators, market microstructure, sentiment, and regime features.

Key Feature Categories:
1. Technical Indicators (RSI, MACD, Bollinger Bands, etc.)
2. Market Microstructure (Bid-Ask Spread, Order Flow, Volume Profile)
3. Volatility Features (GARCH, Realized Vol, Vol Surface)
4. Market Regime Features (Trend, Mean-Reversion, Volatility Regime)
5. Temporal Features (Time of Day, Day of Week, Calendar Effects)
6. Cross-Asset Features (Correlations, Relative Strength)
7. Sentiment Proxy Features (VIX-like, Put-Call Ratios)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
from datetime import datetime, timedelta
import warnings

# Technical analysis imports
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

# Statistical imports
from scipy import stats
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

from src.utils.logger import get_logger
from src.training.steps.pre_training.multi_horizon_profit_labeler import MultiHorizonConfig


class FeatureCategory(Enum):
    """Enumeration of feature categories."""
    TECHNICAL_INDICATORS = "technical_indicators"
    MARKET_MICROSTRUCTURE = "market_microstructure"
    VOLATILITY_FEATURES = "volatility_features"
    REGIME_FEATURES = "regime_features"
    TEMPORAL_FEATURES = "temporal_features"
    CROSS_ASSET_FEATURES = "cross_asset_features"
    SENTIMENT_FEATURES = "sentiment_features"
    PRICE_ACTION_FEATURES = "price_action_features"
    VOLUME_FEATURES = "volume_features"


class FeatureEngineeringMethod(Enum):
    """Enumeration of feature engineering methods."""
    TRADITIONAL_TA = "traditional_ta"
    STATISTICAL_FEATURES = "statistical_features"
    MACHINE_LEARNING_FEATURES = "machine_learning_features"
    REGIME_AWARE_FEATURES = "regime_aware_features"
    ADAPTIVE_FEATURES = "adaptive_features"
    ALL_METHODS = "all_methods"


@dataclass
class ContextualFeatureConfig:
    """Configuration for contextual feature engineering."""
    # Feature categories to include
    enabled_categories: List[FeatureCategory] = field(default_factory=lambda: [
        FeatureCategory.TECHNICAL_INDICATORS,
        FeatureCategory.VOLATILITY_FEATURES,
        FeatureCategory.REGIME_FEATURES,
        FeatureCategory.PRICE_ACTION_FEATURES,
        FeatureCategory.VOLUME_FEATURES
    ])
    
    # Feature engineering method
    engineering_method: FeatureEngineeringMethod = FeatureEngineeringMethod.ALL_METHODS
    
    # Technical indicator parameters
    rsi_periods: List[int] = field(default_factory=lambda: [14, 21, 30])
    ma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100])
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Volatility parameters
    volatility_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    garch_lags: int = 5
    realized_vol_window: int = 20
    
    # Regime detection parameters
    trend_window: int = 50
    regime_lookback: int = 100
    volatility_regime_quantiles: List[float] = field(default_factory=lambda: [0.25, 0.75])
    
    # Temporal feature parameters
    include_time_features: bool = True
    include_calendar_effects: bool = True
    
    # Feature selection parameters
    max_features: int = 100
    feature_selection_method: str = "mutual_info"  # "f_regression", "mutual_info", "pca"
    feature_selection_k: int = 50
    
    # Scaling parameters
    scale_features: bool = True
    scaler_type: str = "standard"  # "standard", "minmax", "robust"
    
    # Lag features
    include_lag_features: bool = True
    lag_periods: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10])
    
    # Cross-validation for feature stability
    validate_feature_stability: bool = True
    stability_window: int = 100
    min_feature_correlation: float = 0.7


@dataclass
class FeatureEngineeringResult:
    """Result container for feature engineering."""
    features_df: pd.DataFrame
    feature_names: List[str]
    feature_importance: Dict[str, float]
    feature_categories: Dict[str, FeatureCategory]
    scaling_info: Dict[str, Any]
    selection_info: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


class ContextualFeatureEngineer:
    """
    Advanced feature engineering for contextual profit labeling.
    
    This class creates rich feature sets that capture market context,
    technical patterns, and regime information for better labeling decisions.
    """
    
    def __init__(self, config: Optional[ContextualFeatureConfig] = None):
        """Initialize contextual feature engineer."""
        self.config = config or ContextualFeatureConfig()
        self.logger = get_logger('ContextualFeatureEngineer')
        
        # Feature engineering state
        self.feature_scalers: Dict[str, Any] = {}
        self.feature_selectors: Dict[str, Any] = {}
        self.feature_names: List[str] = []
        self.feature_categories: Dict[str, FeatureCategory] = {}
        
        # Cache for expensive calculations
        self.feature_cache: Dict[str, pd.DataFrame] = {}
        
        self.logger.info('🔧 Contextual Feature Engineer initialized')
        self.logger.info(f'   → Enabled categories: {[c.value for c in self.config.enabled_categories]}')
        self.logger.info(f'   → Engineering method: {self.config.engineering_method.value}')
    
    def engineer_features(self, market_data: pd.DataFrame) -> FeatureEngineeringResult:
        """
        Engineer comprehensive feature set from market data.
        
        Args:
            market_data: OHLCV market data
            
        Returns:
            FeatureEngineeringResult with engineered features
        """
        self.logger.info('🏗️ Engineering contextual features')
        
        if len(market_data) < 100:
            self.logger.warning('⚠️ Insufficient data for feature engineering')
            return self._create_empty_result(market_data)
        
        # Initialize feature DataFrame
        features_df = pd.DataFrame(index=market_data.index)
        
        # Engineer features by category
        for category in self.config.enabled_categories:
            try:
                self.logger.info(f'   → Engineering {category.value} features')
                
                if category == FeatureCategory.TECHNICAL_INDICATORS:
                    category_features = self._engineer_technical_indicators(market_data)
                elif category == FeatureCategory.VOLATILITY_FEATURES:
                    category_features = self._engineer_volatility_features(market_data)
                elif category == FeatureCategory.REGIME_FEATURES:
                    category_features = self._engineer_regime_features(market_data)
                elif category == FeatureCategory.PRICE_ACTION_FEATURES:
                    category_features = self._engineer_price_action_features(market_data)
                elif category == FeatureCategory.VOLUME_FEATURES:
                    category_features = self._engineer_volume_features(market_data)
                elif category == FeatureCategory.TEMPORAL_FEATURES:
                    category_features = self._engineer_temporal_features(market_data)
                elif category == FeatureCategory.MARKET_MICROSTRUCTURE:
                    category_features = self._engineer_microstructure_features(market_data)
                elif category == FeatureCategory.SENTIMENT_FEATURES:
                    category_features = self._engineer_sentiment_features(market_data)
                else:
                    continue
                
                # Add category features to main DataFrame
                if not category_features.empty:
                    features_df = pd.concat([features_df, category_features], axis=1)
                    
                    # Track feature categories
                    for col in category_features.columns:
                        self.feature_categories[col] = category
                        
            except Exception as e:
                self.logger.error(f'Failed to engineer {category.value} features: {e}')
        
        # Add lag features if enabled
        if self.config.include_lag_features:
            features_df = self._add_lag_features(features_df)
        
        # Handle missing values
        features_df = self._handle_missing_values(features_df)
        
        # Feature selection
        if len(features_df.columns) > self.config.max_features:
            features_df = self._select_features(features_df, market_data)
        
        # Feature scaling
        if self.config.scale_features:
            features_df = self._scale_features(features_df)
        
        # Calculate feature importance
        feature_importance = self._calculate_feature_importance(features_df, market_data)
        
        # Store feature names
        self.feature_names = list(features_df.columns)
        
        result = FeatureEngineeringResult(
            features_df=features_df,
            feature_names=self.feature_names,
            feature_importance=feature_importance,
            feature_categories=self.feature_categories.copy(),
            scaling_info=self._get_scaling_info(),
            selection_info=self._get_selection_info(),
            metadata={
                'n_features': len(features_df.columns),
                'n_samples': len(features_df),
                'categories_used': [c.value for c in self.config.enabled_categories],
                'engineering_method': self.config.engineering_method.value
            }
        )
        
        self.logger.info(f'✅ Feature engineering completed: {len(features_df.columns)} features')
        return result
    
    def _engineer_technical_indicators(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Engineer technical indicator features."""
        features = pd.DataFrame(index=market_data.index)
        
        if 'close' not in market_data.columns:
            return features
        
        close = market_data['close']
        high = market_data.get('high', close)
        low = market_data.get('low', close)
        volume = market_data.get('volume', pd.Series(1, index=market_data.index))
        
        # Moving averages
        for period in self.config.ma_periods:
            if len(close) > period:
                ma = close.rolling(period).mean()
                features[f'ma_{period}'] = ma
                features[f'ma_ratio_{period}'] = close / ma
                features[f'ma_distance_{period}'] = (close - ma) / ma
        
        # RSI
        if TALIB_AVAILABLE:
            for period in self.config.rsi_periods:
                if len(close) > period * 2:
                    rsi = talib.RSI(close.values, timeperiod=period)
                    features[f'rsi_{period}'] = pd.Series(rsi, index=close.index)
        else:
            # Simple RSI calculation
            for period in self.config.rsi_periods:
                if len(close) > period * 2:
                    delta = close.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    features[f'rsi_{period}'] = rsi
        
        # Bollinger Bands
        if len(close) > self.config.bollinger_period:
            bb_ma = close.rolling(self.config.bollinger_period).mean()
            bb_std = close.rolling(self.config.bollinger_period).std()
            bb_upper = bb_ma + (bb_std * self.config.bollinger_std)
            bb_lower = bb_ma - (bb_std * self.config.bollinger_std)
            
            features['bb_upper'] = bb_upper
            features['bb_lower'] = bb_lower
            features['bb_middle'] = bb_ma
            features['bb_width'] = (bb_upper - bb_lower) / bb_ma
            features['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)
        
        # MACD
        if TALIB_AVAILABLE and len(close) > self.config.macd_slow * 2:
            macd, macd_signal, macd_hist = talib.MACD(
                close.values,
                fastperiod=self.config.macd_fast,
                slowperiod=self.config.macd_slow,
                signalperiod=self.config.macd_signal
            )
            features['macd'] = pd.Series(macd, index=close.index)
            features['macd_signal'] = pd.Series(macd_signal, index=close.index)
            features['macd_histogram'] = pd.Series(macd_hist, index=close.index)
        
        # Stochastic Oscillator
        if len(close) > 14 and 'high' in market_data.columns and 'low' in market_data.columns:
            lowest_low = low.rolling(14).min()
            highest_high = high.rolling(14).max()
            k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
            features['stoch_k'] = k_percent
            features['stoch_d'] = k_percent.rolling(3).mean()
        
        # Williams %R
        if len(close) > 14 and 'high' in market_data.columns and 'low' in market_data.columns:
            highest_high = high.rolling(14).max()
            lowest_low = low.rolling(14).min()
            williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
            features['williams_r'] = williams_r
        
        # Average True Range (ATR)
        if all(col in market_data.columns for col in ['high', 'low', 'close']):
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            features['atr'] = true_range.rolling(14).mean()
            features['atr_ratio'] = features['atr'] / close
        
        return features
    
    def _engineer_volatility_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Engineer volatility-based features."""
        features = pd.DataFrame(index=market_data.index)
        
        if 'close' not in market_data.columns:
            return features
        
        close = market_data['close']
        returns = close.pct_change()
        
        # Rolling volatility for different windows
        for window in self.config.volatility_windows:
            if len(returns) > window:
                vol = returns.rolling(window).std()
                features[f'volatility_{window}'] = vol
                features[f'volatility_rank_{window}'] = vol.rolling(100).rank(pct=True)
        
        # Realized volatility (if high-frequency data)
        if len(returns) > self.config.realized_vol_window:
            realized_vol = np.sqrt(252) * returns.rolling(self.config.realized_vol_window).std()
            features['realized_volatility'] = realized_vol
        
        # GARCH-like volatility
        if len(returns) > 50:
            # Simple EWMA volatility (GARCH approximation)
            ewma_vol = returns.ewm(span=20).std()
            features['ewma_volatility'] = ewma_vol
            
            # Volatility of volatility
            vol_20 = returns.rolling(20).std()
            features['vol_of_vol'] = vol_20.rolling(20).std()
        
        # Parkinson volatility (if OHLC available)
        if all(col in market_data.columns for col in ['high', 'low']):
            high = market_data['high']
            low = market_data['low']
            
            # Parkinson estimator
            parkinson_vol = np.sqrt(np.log(high / low) ** 2 / (4 * np.log(2)))
            features['parkinson_volatility'] = parkinson_vol.rolling(20).mean()
        
        # Garman-Klass volatility (if OHLCV available)
        if all(col in market_data.columns for col in ['open', 'high', 'low', 'close']):
            open_price = market_data['open']
            high = market_data['high']
            low = market_data['low']
            
            # Garman-Klass estimator
            gk_vol = np.log(high / low) * np.log(high / close) + np.log(low / close) * np.log(low / open_price)
            features['garman_klass_volatility'] = np.sqrt(gk_vol).rolling(20).mean()
        
        # Volatility regime features
        if len(returns) > 100:
            vol_20 = returns.rolling(20).std()
            vol_quantiles = vol_20.rolling(100).quantile([0.25, 0.5, 0.75])
            
            features['vol_regime_low'] = (vol_20 < vol_quantiles[0.25]).astype(int)
            features['vol_regime_high'] = (vol_20 > vol_quantiles[0.75]).astype(int)
            features['vol_percentile'] = vol_20.rolling(100).rank(pct=True)
        
        return features
    
    def _engineer_regime_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Engineer market regime features."""
        features = pd.DataFrame(index=market_data.index)
        
        if 'close' not in market_data.columns:
            return features
        
        close = market_data['close']
        returns = close.pct_change()
        
        # Trend regime features
        if len(close) > self.config.trend_window:
            # Linear trend
            for window in [20, 50, 100]:
                if len(close) > window:
                    trend_values = []
                    for i in range(window, len(close)):
                        y = close.iloc[i-window:i].values
                        x = np.arange(len(y))
                        if len(y) > 1:
                            slope, _, r_value, _, _ = stats.linregress(x, y)
                            trend_values.append(slope / close.iloc[i])  # Normalized slope
                        else:
                            trend_values.append(0)
                    
                    trend_series = pd.Series([0] * window + trend_values, index=close.index)
                    features[f'trend_slope_{window}'] = trend_series
                    features[f'trend_strength_{window}'] = abs(trend_series)
        
        # Mean reversion features
        if len(close) > 50:
            for window in [20, 50]:
                if len(close) > window:
                    ma = close.rolling(window).mean()
                    std = close.rolling(window).std()
                    z_score = (close - ma) / std
                    
                    features[f'mean_reversion_{window}'] = z_score
                    features[f'mean_reversion_extreme_{window}'] = (abs(z_score) > 2).astype(int)
        
        # Momentum regime
        if len(close) > 20:
            momentum_5 = close / close.shift(5) - 1
            momentum_20 = close / close.shift(20) - 1
            
            features['momentum_5'] = momentum_5
            features['momentum_20'] = momentum_20
            features['momentum_acceleration'] = momentum_5 - momentum_20
        
        # Volatility clustering (ARCH effects)
        if len(returns) > 50:
            returns_squared = returns ** 2
            arch_test = []
            
            for i in range(20, len(returns_squared)):
                recent_vol = returns_squared.iloc[i-20:i]
                if len(recent_vol) > 5:
                    # Simple ARCH test: correlation of squared returns with lagged squared returns
                    lag_1 = recent_vol.shift(1).dropna()
                    current = recent_vol[1:]
                    
                    if len(lag_1) > 5 and len(current) > 5:
                        corr = np.corrcoef(current, lag_1)[0, 1]
                        arch_test.append(corr if not np.isnan(corr) else 0)
                    else:
                        arch_test.append(0)
                else:
                    arch_test.append(0)
            
            arch_series = pd.Series([0] * 20 + arch_test, index=returns.index)
            features['volatility_clustering'] = arch_series
        
        # Market stress indicators
        if len(returns) > 50:
            # Tail risk measure
            rolling_var_95 = returns.rolling(50).quantile(0.05)  # 5% VaR
            features['tail_risk'] = abs(rolling_var_95)
            
            # Skewness and kurtosis
            features['skewness'] = returns.rolling(50).skew()
            features['kurtosis'] = returns.rolling(50).kurt()
        
        return features
    
    def _engineer_price_action_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Engineer price action features."""
        features = pd.DataFrame(index=market_data.index)
        
        if 'close' not in market_data.columns:
            return features
        
        close = market_data['close']
        returns = close.pct_change()
        
        # Price change features
        features['price_change'] = returns
        features['abs_price_change'] = abs(returns)
        features['price_change_squared'] = returns ** 2
        
        # Price momentum features
        for lag in [1, 2, 3, 5, 10]:
            if len(close) > lag:
                features[f'return_{lag}d'] = close.pct_change(lag)
                features[f'momentum_{lag}d'] = close / close.shift(lag) - 1
        
        # High-Low range features (if available)
        if all(col in market_data.columns for col in ['high', 'low']):
            high = market_data['high']
            low = market_data['low']
            
            features['hl_range'] = (high - low) / close
            features['price_position'] = (close - low) / (high - low)
            
            # Gap features (if open available)
            if 'open' in market_data.columns:
                open_price = market_data['open']
                features['gap'] = (open_price - close.shift(1)) / close.shift(1)
                features['gap_filled'] = ((high >= close.shift(1)) & (low <= close.shift(1))).astype(int)
        
        # Support and resistance levels
        if len(close) > 50:
            # Simple support/resistance using rolling min/max
            for window in [20, 50]:
                if len(close) > window:
                    resistance = close.rolling(window).max()
                    support = close.rolling(window).min()
                    
                    features[f'resistance_distance_{window}'] = (resistance - close) / close
                    features[f'support_distance_{window}'] = (close - support) / close
                    features[f'sr_position_{window}'] = (close - support) / (resistance - support)
        
        # Fractal patterns (simplified)
        if len(close) > 10:
            # Local maxima and minima
            highs = close.rolling(5, center=True).max() == close
            lows = close.rolling(5, center=True).min() == close
            
            features['local_high'] = highs.astype(int)
            features['local_low'] = lows.astype(int)
        
        # Price velocity and acceleration
        if len(returns) > 5:
            velocity = returns.rolling(3).mean()  # 3-period average return
            acceleration = velocity.diff()  # Change in velocity
            
            features['price_velocity'] = velocity
            features['price_acceleration'] = acceleration
        
        return features
    
    def _engineer_volume_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Engineer volume-based features."""
        features = pd.DataFrame(index=market_data.index)
        
        if 'volume' not in market_data.columns:
            return features
        
        volume = market_data['volume']
        close = market_data.get('close', pd.Series(1, index=market_data.index))
        
        # Volume change features
        features['volume_change'] = volume.pct_change()
        features['volume_ratio'] = volume / volume.rolling(20).mean()
        
        # Volume moving averages
        for period in [5, 10, 20, 50]:
            if len(volume) > period:
                vol_ma = volume.rolling(period).mean()
                features[f'volume_ma_{period}'] = vol_ma
                features[f'volume_vs_ma_{period}'] = volume / vol_ma
        
        # On-Balance Volume (OBV)
        if 'close' in market_data.columns:
            returns = close.pct_change()
            obv_direction = np.sign(returns)
            obv = (volume * obv_direction).cumsum()
            features['obv'] = obv
            features['obv_ma'] = obv.rolling(20).mean()
        
        # Volume-Price Trend (VPT)
        if 'close' in market_data.columns:
            returns = close.pct_change()
            vpt = (volume * returns).cumsum()
            features['vpt'] = vpt
        
        # Money Flow Index (MFI) approximation
        if all(col in market_data.columns for col in ['high', 'low', 'close']):
            high = market_data['high']
            low = market_data['low']
            
            typical_price = (high + low + close) / 3
            money_flow = typical_price * volume
            
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
            
            mfi = 100 - (100 / (1 + positive_flow / negative_flow))
            features['mfi'] = mfi
        
        # Volume volatility
        features['volume_volatility'] = volume.rolling(20).std() / volume.rolling(20).mean()
        
        # Volume spikes
        vol_mean = volume.rolling(20).mean()
        vol_std = volume.rolling(20).std()
        features['volume_spike'] = ((volume - vol_mean) / vol_std > 2).astype(int)
        
        return features
    
    def _engineer_temporal_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Engineer temporal features."""
        features = pd.DataFrame(index=market_data.index)
        
        if not self.config.include_time_features:
            return features
        
        # Extract datetime components
        dt_index = pd.to_datetime(market_data.index)
        
        # Time of day features
        features['hour'] = dt_index.hour
        features['minute'] = dt_index.minute
        features['hour_sin'] = np.sin(2 * np.pi * dt_index.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * dt_index.hour / 24)
        
        # Day of week features
        features['day_of_week'] = dt_index.dayofweek
        features['is_weekend'] = (dt_index.dayofweek >= 5).astype(int)
        
        # Market session features (assuming UTC times)
        # Asian session: 00:00-08:00 UTC
        # European session: 08:00-16:00 UTC  
        # US session: 16:00-00:00 UTC
        features['asian_session'] = ((dt_index.hour >= 0) & (dt_index.hour < 8)).astype(int)
        features['european_session'] = ((dt_index.hour >= 8) & (dt_index.hour < 16)).astype(int)
        features['us_session'] = ((dt_index.hour >= 16) | (dt_index.hour < 0)).astype(int)
        
        # Calendar effects (if enabled)
        if self.config.include_calendar_effects:
            features['day_of_month'] = dt_index.day
            features['month'] = dt_index.month
            features['quarter'] = dt_index.quarter
            features['is_month_end'] = (dt_index.day >= 28).astype(int)
            features['is_quarter_end'] = ((dt_index.month % 3 == 0) & (dt_index.day >= 28)).astype(int)
        
        return features
    
    def _engineer_microstructure_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Engineer market microstructure features (simplified)."""
        features = pd.DataFrame(index=market_data.index)
        
        # Note: Real microstructure features would require tick data
        # These are approximations using OHLCV data
        
        if all(col in market_data.columns for col in ['open', 'high', 'low', 'close']):
            open_price = market_data['open']
            high = market_data['high']
            low = market_data['low']
            close = market_data['close']
            
            # Bid-ask spread proxy (high-low range)
            spread_proxy = (high - low) / close
            features['spread_proxy'] = spread_proxy
            
            # Price impact proxy
            price_move = abs(close - open_price) / open_price
            features['price_impact_proxy'] = price_move
            
            # Market efficiency proxy (price discovery)
            efficiency_proxy = abs(close - open_price) / (high - low)
            features['efficiency_proxy'] = efficiency_proxy.replace([np.inf, -np.inf], np.nan)
        
        return features
    
    def _engineer_sentiment_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Engineer sentiment proxy features."""
        features = pd.DataFrame(index=market_data.index)
        
        # Note: Real sentiment features would require external data
        # These are market-based sentiment proxies
        
        if 'close' in market_data.columns:
            close = market_data['close']
            returns = close.pct_change()
            
            # Fear/Greed proxy based on volatility and returns
            if len(returns) > 20:
                vol_20 = returns.rolling(20).std()
                ret_20 = returns.rolling(20).mean()
                
                # High volatility + negative returns = fear
                # Low volatility + positive returns = greed
                fear_proxy = vol_20 * (-ret_20)  # Higher when vol high and returns negative
                features['fear_proxy'] = fear_proxy
                
                # Momentum sentiment
                momentum_sentiment = ret_20 / vol_20  # Risk-adjusted momentum
                features['momentum_sentiment'] = momentum_sentiment
        
        return features
    
    def _add_lag_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged versions of features."""
        if not self.config.include_lag_features:
            return features_df
        
        lagged_features = features_df.copy()
        
        # Select subset of features for lagging (to avoid explosion)
        important_features = [col for col in features_df.columns 
                            if any(keyword in col.lower() for keyword in 
                                   ['rsi', 'ma_ratio', 'volatility', 'momentum', 'trend'])][:20]
        
        for lag in self.config.lag_periods:
            for feature in important_features:
                if feature in features_df.columns:
                    lagged_features[f'{feature}_lag_{lag}'] = features_df[feature].shift(lag)
        
        return lagged_features
    
    def _handle_missing_values(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features."""
        # Forward fill first, then backward fill, then fill with 0
        features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Replace infinite values
        features_df = features_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        return features_df
    
    def _select_features(self, features_df: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
        """Select most important features."""
        if len(features_df.columns) <= self.config.feature_selection_k:
            return features_df
        
        # Prepare target variable (future returns)
        if 'close' in market_data.columns:
            target = market_data['close'].pct_change().shift(-1).fillna(0)
        else:
            # If no target available, return original features
            return features_df
        
        # Align features and target
        common_idx = features_df.index.intersection(target.index)
        X = features_df.loc[common_idx].fillna(0)
        y = target.loc[common_idx]
        
        if len(X) < 50:  # Need sufficient data for feature selection
            return features_df
        
        try:
            if self.config.feature_selection_method == "mutual_info":
                selector = SelectKBest(score_func=mutual_info_regression, k=self.config.feature_selection_k)
            elif self.config.feature_selection_method == "f_regression":
                selector = SelectKBest(score_func=f_regression, k=self.config.feature_selection_k)
            elif self.config.feature_selection_method == "pca":
                # Use PCA for dimensionality reduction
                selector = PCA(n_components=self.config.feature_selection_k)
                
                X_selected = selector.fit_transform(X)
                selected_features_df = pd.DataFrame(
                    X_selected, 
                    index=X.index,
                    columns=[f'pca_{i}' for i in range(X_selected.shape[1])]
                )
                self.feature_selectors['pca'] = selector
                return selected_features_df
            else:
                return features_df
            
            # Fit selector and transform features
            X_selected = selector.fit_transform(X, y)
            selected_feature_names = [features_df.columns[i] for i in selector.get_support(indices=True)]
            
            selected_features_df = pd.DataFrame(
                X_selected,
                index=X.index,
                columns=selected_feature_names
            )
            
            # Store selector for future use
            self.feature_selectors['selector'] = selector
            
            return selected_features_df
            
        except Exception as e:
            self.logger.warning(f'Feature selection failed: {e}')
            return features_df
    
    def _scale_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Scale features."""
        if not self.config.scale_features:
            return features_df
        
        try:
            if self.config.scaler_type == "standard":
                scaler = StandardScaler()
            elif self.config.scaler_type == "minmax":
                scaler = MinMaxScaler()
            else:
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
            
            # Fit scaler and transform features
            features_scaled = scaler.fit_transform(features_df.fillna(0))
            
            scaled_features_df = pd.DataFrame(
                features_scaled,
                index=features_df.index,
                columns=features_df.columns
            )
            
            # Store scaler for future use
            self.feature_scalers['scaler'] = scaler
            
            return scaled_features_df
            
        except Exception as e:
            self.logger.warning(f'Feature scaling failed: {e}')
            return features_df
    
    def _calculate_feature_importance(self, 
                                    features_df: pd.DataFrame, 
                                    market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate feature importance scores."""
        importance = {}
        
        if 'close' not in market_data.columns or len(features_df) < 50:
            return importance
        
        try:
            # Prepare target variable
            target = market_data['close'].pct_change().shift(-1).fillna(0)
            
            # Align features and target
            common_idx = features_df.index.intersection(target.index)
            X = features_df.loc[common_idx].fillna(0)
            y = target.loc[common_idx]
            
            if len(X) < 20:
                return importance
            
            # Calculate mutual information scores
            mi_scores = mutual_info_regression(X, y, random_state=42)
            
            # Normalize scores
            if np.sum(mi_scores) > 0:
                mi_scores = mi_scores / np.sum(mi_scores)
            
            # Create importance dictionary
            for i, feature_name in enumerate(X.columns):
                importance[feature_name] = float(mi_scores[i])
                
        except Exception as e:
            self.logger.warning(f'Feature importance calculation failed: {e}')
        
        return importance
    
    def _get_scaling_info(self) -> Dict[str, Any]:
        """Get scaling information."""
        return {
            'scaler_type': self.config.scaler_type,
            'scaler_fitted': 'scaler' in self.feature_scalers,
            'scale_features': self.config.scale_features
        }
    
    def _get_selection_info(self) -> Dict[str, Any]:
        """Get feature selection information."""
        return {
            'selection_method': self.config.feature_selection_method,
            'max_features': self.config.max_features,
            'selection_k': self.config.feature_selection_k,
            'selector_fitted': 'selector' in self.feature_selectors or 'pca' in self.feature_selectors
        }
    
    def _create_empty_result(self, market_data: pd.DataFrame) -> FeatureEngineeringResult:
        """Create empty feature engineering result."""
        empty_features = pd.DataFrame(index=market_data.index)
        
        return FeatureEngineeringResult(
            features_df=empty_features,
            feature_names=[],
            feature_importance={},
            feature_categories={},
            scaling_info=self._get_scaling_info(),
            selection_info=self._get_selection_info(),
            metadata={'error': 'insufficient_data', 'n_features': 0}
        )
    
    def apply_labeling_adjustments(self,
                                 labels: pd.DataFrame,
                                 features: pd.DataFrame,
                                 config: MultiHorizonConfig) -> pd.DataFrame:
        """Apply feature-based adjustments to labeling."""
        adjusted_labels = labels.copy()
        
        if features.empty:
            return adjusted_labels
        
        # Get common index
        common_idx = labels.index.intersection(features.index)
        if len(common_idx) < 10:
            return adjusted_labels
        
        # Feature-based adjustments
        try:
            # Volatility regime adjustments
            if 'volatility_regime_high' in features.columns:
                high_vol_mask = features.loc[common_idx, 'volatility_regime_high'] == 1
                
                # Increase targets in high volatility periods
                for col in adjusted_labels.columns:
                    if col.endswith('_prob'):
                        adjusted_labels.loc[common_idx[high_vol_mask], col] *= 1.2
            
            # Trend regime adjustments
            trend_cols = [col for col in features.columns if 'trend_strength' in col]
            if trend_cols:
                trend_strength = features.loc[common_idx, trend_cols[0]]
                strong_trend_mask = trend_strength > trend_strength.quantile(0.8)
                
                # Adjust probabilities based on trend strength
                for col in adjusted_labels.columns:
                    if col.endswith('_prob'):
                        trend_multiplier = 1 + (trend_strength * 0.5)  # Up to 50% increase
                        adjusted_labels.loc[common_idx, col] *= trend_multiplier.loc[common_idx]
            
            # Momentum adjustments
            momentum_cols = [col for col in features.columns if 'momentum_' in col]
            if momentum_cols:
                momentum = features.loc[common_idx, momentum_cols[0]]
                
                # Adjust based on momentum direction and strength
                for col in adjusted_labels.columns:
                    if col.endswith('_prob'):
                        momentum_multiplier = 1 + (abs(momentum) * 0.3)  # Up to 30% adjustment
                        adjusted_labels.loc[common_idx, col] *= momentum_multiplier.loc[common_idx]
            
            # Clip probabilities to valid range
            prob_columns = [col for col in adjusted_labels.columns if col.endswith('_prob')]
            for col in prob_columns:
                adjusted_labels[col] = adjusted_labels[col].clip(0.0, 1.0)
                
        except Exception as e:
            self.logger.warning(f'Feature-based label adjustment failed: {e}')
        
        return adjusted_labels


# Convenience functions
def engineer_contextual_features(market_data: pd.DataFrame,
                                config: Optional[ContextualFeatureConfig] = None) -> FeatureEngineeringResult:
    """Convenience function for contextual feature engineering."""
    engineer = ContextualFeatureEngineer(config)
    return engineer.engineer_features(market_data)


def create_feature_enhanced_labels(market_data: pd.DataFrame,
                                 labels: pd.DataFrame,
                                 labeling_config: MultiHorizonConfig,
                                 feature_config: Optional[ContextualFeatureConfig] = None) -> pd.DataFrame:
    """Create feature-enhanced labels."""
    engineer = ContextualFeatureEngineer(feature_config)
    
    # Engineer features
    feature_result = engineer.engineer_features(market_data)
    
    # Apply feature-based adjustments
    enhanced_labels = engineer.apply_labeling_adjustments(
        labels, feature_result.features_df, labeling_config
    )
    
    return enhanced_labels