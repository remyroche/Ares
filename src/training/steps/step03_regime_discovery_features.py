#!/usr/bin/env python3
"""Enhanced Regime Discovery Feature Engineering for Step 3.

This module creates regime-aware features specifically designed for regime discovery,
focusing on features that help distinguish between different market regimes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

class RegimeDiscoveryFeatureEngineer:
    """Enhanced feature engineering specifically for regime discovery."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.feature_cache = {}
        
    def create_regime_discovery_features(self, df: pd.DataFrame, existing_regimes: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Create comprehensive regime discovery features.
        
        Args:
            df: Market data with OHLCV columns
            existing_regimes: Optional existing regime labels for iterative improvement
            
        Returns:
            DataFrame with regime discovery features
        """
        features = pd.DataFrame(index=df.index)
        
        # 1. Regime Transition Prediction Features
        features = pd.concat([features, self._create_regime_transition_features(df)], axis=1)
        
        # 2. Market Microstructure Features
        features = pd.concat([features, self._create_microstructure_features(df)], axis=1)
        
        # 3. Temporal Regime Features
        features = pd.concat([features, self._create_temporal_regime_features(df)], axis=1)
        
        # 4. Volatility Regime Features
        features = pd.concat([features, self._create_volatility_regime_features(df)], axis=1)
        
        # 5. Volume Regime Features
        features = pd.concat([features, self._create_volume_regime_features(df)], axis=1)
        
        # 6. Price Action Regime Features
        features = pd.concat([features, self._create_price_action_regime_features(df)], axis=1)
        
        # 7. Cross-Asset Regime Features (if available)
        features = pd.concat([features, self._create_cross_asset_regime_features(df)], axis=1)
        
        # 8. Regime Persistence Features
        if existing_regimes is not None:
            features = pd.concat([features, self._create_regime_persistence_features(df, existing_regimes)], axis=1)
        
        # 9. Regime Strength Features
        features = pd.concat([features, self._create_regime_strength_features(df)], axis=1)
        
        # 10. Regime Change Early Warning Features
        features = pd.concat([features, self._create_regime_change_warning_features(df)], axis=1)
        
        return features.fillna(0)
    
    def _create_regime_transition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features that predict regime transitions."""
        features = pd.DataFrame(index=df.index)
        
        # 1. Regime Change Probability Indicators
        features['regime_change_prob_volatility'] = self._calculate_regime_change_probability(
            df['close'].pct_change().rolling(20).std(), window=10
        )
        
        features['regime_change_prob_volume'] = self._calculate_regime_change_probability(
            df['volume'].rolling(20).mean(), window=10
        )
        
        features['regime_change_prob_momentum'] = self._calculate_regime_change_probability(
            df['close'].pct_change(10), window=10
        )
        
        # 2. Regime Persistence Indicators
        features['regime_persistence_volatility'] = self._calculate_regime_persistence(
            df['close'].pct_change().rolling(20).std(), min_duration=5
        )
        
        features['regime_persistence_volume'] = self._calculate_regime_persistence(
            df['volume'].rolling(20).mean(), min_duration=5
        )
        
        # 3. Regime Transition Timing Features
        features['regime_transition_timing'] = self._calculate_regime_transition_timing(df)
        
        # 4. Regime Stability Indicators
        features['regime_stability_volatility'] = self._calculate_regime_stability(
            df['close'].pct_change().rolling(20).std(), window=20
        )
        
        features['regime_stability_volume'] = self._calculate_regime_stability(
            df['volume'].rolling(20).mean(), window=20
        )
        
        return features
    
    def _create_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market microstructure features for regime detection."""
        features = pd.DataFrame(index=df.index)
        
        # 1. Order Flow Imbalance Proxies
        features['order_flow_imbalance'] = self._calculate_order_flow_imbalance(df)
        
        # 2. Volume Profile Analysis
        features['volume_profile_skew'] = self._calculate_volume_profile_skew(df)
        features['volume_profile_kurtosis'] = self._calculate_volume_profile_kurtosis(df)
        
        # 3. Price Impact Features
        features['price_impact_ratio'] = self._calculate_price_impact_ratio(df)
        features['price_impact_volatility'] = self._calculate_price_impact_volatility(df)
        
        # 4. Bid-Ask Spread Proxies
        features['spread_proxy'] = self._calculate_spread_proxy(df)
        
        # 5. Market Depth Proxies
        features['market_depth_proxy'] = self._calculate_market_depth_proxy(df)
        
        # 6. Liquidity Regime Indicators
        features['liquidity_regime_indicator'] = self._calculate_liquidity_regime_indicator(df)
        
        return features
    
    def _create_temporal_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal regime features."""
        features = pd.DataFrame(index=df.index)
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)
        
        # 1. Time-of-Day Regime Patterns
        features['hour_of_day'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek
        features['day_of_month'] = df.index.day
        features['month_of_year'] = df.index.month
        
        # 2. Cyclical Regime Features
        features['hour_sin'] = np.sin(2 * np.pi * features['hour_of_day'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour_of_day'] / 24)
        features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        
        # 3. Regime Duration Forecasting Features
        features['regime_duration_forecast'] = self._forecast_regime_duration(df)
        
        # 4. Time-based Volatility Patterns
        features['time_based_volatility'] = self._calculate_time_based_volatility(df)
        
        # 5. Session-based Regime Features
        features['session_regime_indicator'] = self._calculate_session_regime_indicator(df)
        
        return features
    
    def _create_volatility_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volatility regime features."""
        features = pd.DataFrame(index=df.index)
        
        # 1. Multi-timeframe Volatility
        for window in [5, 10, 20, 50]:
            features[f'volatility_{window}'] = df['close'].pct_change().rolling(window).std()
            features[f'volatility_{window}_normalized'] = features[f'volatility_{window}'] / df['close']
        
        # 2. Volatility of Volatility
        features['vol_of_vol_20'] = features['volatility_20'].rolling(20).std()
        features['vol_of_vol_50'] = features['volatility_50'].rolling(50).std()
        
        # 3. Volatility Regime Classification
        features['volatility_regime'] = self._classify_volatility_regime(features['volatility_20'])
        
        # 4. Volatility Clustering Features
        features['volatility_clustering'] = self._calculate_volatility_clustering(df['close'])
        
        # 5. Volatility Persistence
        features['volatility_persistence'] = self._calculate_volatility_persistence(features['volatility_20'])
        
        # 6. Volatility Regime Transitions
        features['volatility_regime_transition'] = self._detect_volatility_regime_transitions(features['volatility_20'])
        
        return features
    
    def _create_volume_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volume regime features."""
        features = pd.DataFrame(index=df.index)
        
        # 1. Volume Regime Classification
        features['volume_regime'] = self._classify_volume_regime(df['volume'])
        
        # 2. Volume-Momentum Interaction
        features['volume_momentum_interaction'] = (
            df['volume'].pct_change(5) * df['close'].pct_change(5)
        )
        
        # 3. Volume-Price Divergence
        features['volume_price_divergence'] = self._calculate_volume_price_divergence(df)
        
        # 4. Volume Regime Persistence
        features['volume_regime_persistence'] = self._calculate_volume_regime_persistence(df['volume'])
        
        # 5. Volume Spike Detection
        features['volume_spike_indicator'] = self._detect_volume_spikes(df['volume'])
        
        # 6. Volume Trend Changes
        features['volume_trend_change'] = self._detect_volume_trend_changes(df['volume'])
        
        return features
    
    def _create_price_action_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price action regime features."""
        features = pd.DataFrame(index=df.index)
        
        # 1. Price Action Regime Classification
        features['price_action_regime'] = self._classify_price_action_regime(df)
        
        # 2. Support/Resistance Proximity
        features['sr_proximity'] = self._calculate_sr_proximity(df)
        
        # 3. Price Momentum Regime
        features['momentum_regime'] = self._classify_momentum_regime(df['close'])
        
        # 4. Trend Strength Indicators
        features['trend_strength'] = self._calculate_trend_strength(df)
        
        # 5. Price Action Persistence
        features['price_action_persistence'] = self._calculate_price_action_persistence(df)
        
        return features
    
    def _create_cross_asset_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cross-asset regime features (if multiple assets available)."""
        features = pd.DataFrame(index=df.index)
        
        # This would be implemented if we have multiple asset data
        # For now, create placeholder features based on single asset
        
        # 1. Cross-timeframe correlation
        features['cross_timeframe_correlation'] = self._calculate_cross_timeframe_correlation(df)
        
        # 2. Regime correlation features
        features['regime_correlation_proxy'] = self._calculate_regime_correlation_proxy(df)
        
        return features
    
    def _create_regime_persistence_features(self, df: pd.DataFrame, existing_regimes: np.ndarray) -> pd.DataFrame:
        """Create features based on existing regime persistence."""
        features = pd.DataFrame(index=df.index)
        
        # 1. Regime Duration
        features['regime_duration'] = self._calculate_regime_duration(existing_regimes)
        
        # 2. Regime Stability Score
        features['regime_stability_score'] = self._calculate_regime_stability_score(existing_regimes)
        
        # 3. Regime Transition Probability
        features['regime_transition_probability'] = self._calculate_regime_transition_probability(existing_regimes)
        
        # 4. Regime Persistence Forecast
        features['regime_persistence_forecast'] = self._forecast_regime_persistence(existing_regimes)
        
        return features
    
    def _create_regime_strength_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create regime strength features."""
        features = pd.DataFrame(index=df.index)
        
        # 1. Regime Strength Indicators
        features['regime_strength_volatility'] = self._calculate_regime_strength_volatility(df)
        features['regime_strength_volume'] = self._calculate_regime_strength_volume(df)
        features['regime_strength_momentum'] = self._calculate_regime_strength_momentum(df)
        
        # 2. Regime Confidence Score
        features['regime_confidence_score'] = self._calculate_regime_confidence_score(df)
        
        # 3. Regime Coherence Score
        features['regime_coherence_score'] = self._calculate_regime_coherence_score(df)
        
        return features
    
    def _create_regime_change_warning_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create early warning features for regime changes."""
        features = pd.DataFrame(index=df.index)
        
        # 1. Regime Change Early Warning
        features['regime_change_early_warning'] = self._calculate_regime_change_early_warning(df)
        
        # 2. Regime Weakening Indicators
        features['regime_weakening_indicator'] = self._calculate_regime_weakening_indicator(df)
        
        # 3. Regime Transition Readiness
        features['regime_transition_readiness'] = self._calculate_regime_transition_readiness(df)
        
        return features
    
    # Helper methods for feature calculations
    
    def _calculate_regime_change_probability(self, series: pd.Series, window: int = 10) -> pd.Series:
        """Calculate probability of regime change based on series characteristics."""
        # Use rolling statistics to detect regime changes
        rolling_mean = series.rolling(window).mean()
        rolling_std = series.rolling(window).std()
        
        # Z-score based regime change probability
        z_scores = (series - rolling_mean) / rolling_std
        regime_change_prob = np.abs(z_scores).rolling(window).mean()
        
        return regime_change_prob.fillna(0)
    
    def _calculate_regime_persistence(self, series: pd.Series, min_duration: int = 5) -> pd.Series:
        """Calculate how long current regime has persisted."""
        # Detect regime changes using rolling statistics
        rolling_mean = series.rolling(min_duration).mean()
        regime_changes = (np.abs(series - rolling_mean) > rolling_mean.rolling(min_duration).std()).astype(int)
        
        # Calculate persistence
        persistence = pd.Series(index=series.index, dtype=float)
        current_persistence = 0
        
        for i in range(len(series)):
            if regime_changes.iloc[i] == 0:
                current_persistence += 1
            else:
                current_persistence = 0
            persistence.iloc[i] = current_persistence
        
        return persistence
    
    def _calculate_regime_transition_timing(self, df: pd.DataFrame) -> pd.Series:
        """Calculate optimal timing for regime transitions."""
        # Combine multiple signals for transition timing
        volatility_signal = df['close'].pct_change().rolling(20).std()
        volume_signal = df['volume'].rolling(20).mean()
        momentum_signal = df['close'].pct_change(10)
        
        # Normalize signals
        vol_norm = (volatility_signal - volatility_signal.rolling(50).mean()) / volatility_signal.rolling(50).std()
        vol_vol_norm = (volume_signal - volume_signal.rolling(50).mean()) / volume_signal.rolling(50).std()
        mom_norm = (momentum_signal - momentum_signal.rolling(50).mean()) / momentum_signal.rolling(50).std()
        
        # Combine signals
        transition_timing = (vol_norm + vol_vol_norm + mom_norm) / 3
        
        return transition_timing.fillna(0)
    
    def _calculate_regime_stability(self, series: pd.Series, window: int = 20) -> pd.Series:
        """Calculate regime stability based on series consistency."""
        # Calculate rolling coefficient of variation
        rolling_mean = series.rolling(window).mean()
        rolling_std = series.rolling(window).std()
        cv = rolling_std / rolling_mean
        
        # Stability is inverse of coefficient of variation
        stability = 1 / (1 + cv)
        
        return stability.fillna(0)
    
    def _calculate_order_flow_imbalance(self, df: pd.DataFrame) -> pd.Series:
        """Calculate order flow imbalance proxy."""
        # Use high-low range and close position as proxy for order flow
        hl_range = df['high'] - df['low']
        close_position = (df['close'] - df['low']) / (hl_range + 1e-8)
        
        # Order flow imbalance based on close position
        order_flow_imbalance = (close_position - 0.5) * 2  # Scale to [-1, 1]
        
        return order_flow_imbalance.fillna(0)
    
    def _calculate_volume_profile_skew(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volume profile skewness."""
        # Use volume-weighted price position as proxy
        vwap = (df['high'] + df['low'] + df['close']) / 3
        price_range = df['high'] - df['low']
        price_position = (vwap - df['low']) / (price_range + 1e-8)
        
        # Calculate skewness of price position weighted by volume
        volume_skew = (price_position * df['volume']).rolling(20).skew()
        
        return volume_skew.fillna(0)
    
    def _calculate_volume_profile_kurtosis(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volume profile kurtosis."""
        vwap = (df['high'] + df['low'] + df['close']) / 3
        price_range = df['high'] - df['low']
        price_position = (vwap - df['low']) / (price_range + 1e-8)
        
        # Calculate kurtosis of price position weighted by volume
        volume_kurtosis = (price_position * df['volume']).rolling(20).kurt()
        
        return volume_kurtosis.fillna(0)
    
    def _calculate_price_impact_ratio(self, df: pd.DataFrame) -> pd.Series:
        """Calculate price impact ratio."""
        # Price impact = price change per unit volume
        price_change = df['close'].pct_change()
        volume_change = df['volume'].pct_change()
        
        # Price impact ratio
        price_impact = price_change / (volume_change + 1e-8)
        
        return price_impact.fillna(0)
    
    def _calculate_price_impact_volatility(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volatility of price impact."""
        price_impact = self._calculate_price_impact_ratio(df)
        price_impact_vol = price_impact.rolling(20).std()
        
        return price_impact_vol.fillna(0)
    
    def _calculate_spread_proxy(self, df: pd.DataFrame) -> pd.Series:
        """Calculate bid-ask spread proxy."""
        # Use high-low range as proxy for spread
        spread_proxy = (df['high'] - df['low']) / df['close']
        
        return spread_proxy.fillna(0)
    
    def _calculate_market_depth_proxy(self, df: pd.DataFrame) -> pd.Series:
        """Calculate market depth proxy."""
        # Use volume relative to price range as depth proxy
        price_range = df['high'] - df['low']
        market_depth = df['volume'] / (price_range + 1e-8)
        
        return market_depth.fillna(0)
    
    def _calculate_liquidity_regime_indicator(self, df: pd.DataFrame) -> pd.Series:
        """Calculate liquidity regime indicator."""
        # Combine spread and depth proxies
        spread_proxy = self._calculate_spread_proxy(df)
        depth_proxy = self._calculate_market_depth_proxy(df)
        
        # Normalize and combine
        spread_norm = (spread_proxy - spread_proxy.rolling(50).mean()) / spread_proxy.rolling(50).std()
        depth_norm = (depth_proxy - depth_proxy.rolling(50).mean()) / depth_proxy.rolling(50).std()
        
        # Liquidity regime (higher is more liquid)
        liquidity_regime = depth_norm - spread_norm
        
        return liquidity_regime.fillna(0)
    
    def _forecast_regime_duration(self, df: pd.DataFrame) -> pd.Series:
        """Forecast regime duration based on current market conditions."""
        # Use volatility and volume patterns to forecast duration
        volatility = df['close'].pct_change().rolling(20).std()
        volume = df['volume'].rolling(20).mean()
        
        # Higher volatility and volume = shorter regime duration
        vol_factor = 1 / (1 + volatility / volatility.rolling(50).mean())
        vol_factor = 1 / (1 + volume / volume.rolling(50).mean())
        
        duration_forecast = vol_factor * vol_factor * 20  # Base duration of 20 periods
        
        return duration_forecast.fillna(20)
    
    def _calculate_time_based_volatility(self, df: pd.DataFrame) -> pd.Series:
        """Calculate time-based volatility patterns."""
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            return pd.Series(0, index=df.index)
        
        # Calculate volatility by hour of day
        hour_volatility = df['close'].pct_change().groupby(df.index.hour).std()
        
        # Map to current data
        time_based_vol = df.index.hour.map(hour_volatility)
        
        return time_based_vol.fillna(0)
    
    def _calculate_session_regime_indicator(self, df: pd.DataFrame) -> pd.Series:
        """Calculate session-based regime indicator."""
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            return pd.Series(0, index=df.index)
        
        # Define trading sessions (simplified)
        hour = df.index.hour
        session = pd.Series(0, index=df.index)
        
        # Asian session (0-8), European session (8-16), US session (16-24)
        session[hour < 8] = 1  # Asian
        session[(hour >= 8) & (hour < 16)] = 2  # European
        session[hour >= 16] = 3  # US
        
        return session
    
    def _classify_volatility_regime(self, volatility: pd.Series) -> pd.Series:
        """Classify volatility regime."""
        # Use quantiles to classify volatility regimes
        low_threshold = volatility.rolling(100).quantile(0.33)
        high_threshold = volatility.rolling(100).quantile(0.67)
        
        regime = pd.Series(1, index=volatility.index)  # Low volatility
        regime[volatility > high_threshold] = 3  # High volatility
        regime[(volatility > low_threshold) & (volatility <= high_threshold)] = 2  # Medium volatility
        
        return regime.fillna(1)
    
    def _calculate_volatility_clustering(self, prices: pd.Series) -> pd.Series:
        """Calculate volatility clustering indicator."""
        returns = prices.pct_change()
        volatility = returns.rolling(20).std()
        
        # Volatility clustering = autocorrelation of volatility
        clustering = volatility.rolling(50).apply(lambda x: x.autocorr(lag=1))
        
        return clustering.fillna(0)
    
    def _calculate_volatility_persistence(self, volatility: pd.Series) -> pd.Series:
        """Calculate volatility persistence."""
        # Use AR(1) coefficient as persistence measure
        persistence = volatility.rolling(50).apply(
            lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 1 else 0
        )
        
        return persistence.fillna(0)
    
    def _detect_volatility_regime_transitions(self, volatility: pd.Series) -> pd.Series:
        """Detect volatility regime transitions."""
        # Use change point detection
        rolling_mean = volatility.rolling(20).mean()
        rolling_std = volatility.rolling(20).std()
        
        # Detect significant changes
        z_scores = (volatility - rolling_mean) / rolling_std
        transitions = (np.abs(z_scores) > 2).astype(int)
        
        return transitions.fillna(0)
    
    def _classify_volume_regime(self, volume: pd.Series) -> pd.Series:
        """Classify volume regime."""
        # Use quantiles to classify volume regimes
        low_threshold = volume.rolling(100).quantile(0.33)
        high_threshold = volume.rolling(100).quantile(0.67)
        
        regime = pd.Series(1, index=volume.index)  # Low volume
        regime[volume > high_threshold] = 3  # High volume
        regime[(volume > low_threshold) & (volume <= high_threshold)] = 2  # Medium volume
        
        return regime.fillna(1)
    
    def _calculate_volume_price_divergence(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volume-price divergence."""
        price_momentum = df['close'].pct_change(5)
        volume_momentum = df['volume'].pct_change(5)
        
        # Divergence = opposite signs of momentum
        divergence = (price_momentum * volume_momentum < 0).astype(int)
        
        return divergence.fillna(0)
    
    def _calculate_volume_regime_persistence(self, volume: pd.Series) -> pd.Series:
        """Calculate volume regime persistence."""
        volume_regime = self._classify_volume_regime(volume)
        return self._calculate_regime_persistence(volume_regime)
    
    def _detect_volume_spikes(self, volume: pd.Series) -> pd.Series:
        """Detect volume spikes."""
        rolling_mean = volume.rolling(20).mean()
        rolling_std = volume.rolling(20).std()
        
        # Volume spike = volume > mean + 2*std
        spikes = (volume > rolling_mean + 2 * rolling_std).astype(int)
        
        return spikes.fillna(0)
    
    def _detect_volume_trend_changes(self, volume: pd.Series) -> pd.Series:
        """Detect volume trend changes."""
        volume_trend = volume.rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        trend_changes = (volume_trend.diff() * volume_trend.shift(1) < 0).astype(int)
        
        return trend_changes.fillna(0)
    
    def _classify_price_action_regime(self, df: pd.DataFrame) -> pd.Series:
        """Classify price action regime."""
        # Combine multiple price action indicators
        momentum = df['close'].pct_change(10)
        volatility = df['close'].pct_change().rolling(20).std()
        range_size = (df['high'] - df['low']) / df['close']
        
        # Normalize indicators
        mom_norm = (momentum - momentum.rolling(50).mean()) / momentum.rolling(50).std()
        vol_norm = (volatility - volatility.rolling(50).mean()) / volatility.rolling(50).std()
        range_norm = (range_size - range_size.rolling(50).mean()) / range_size.rolling(50).std()
        
        # Classify regimes based on combinations
        regime = pd.Series(1, index=df.index)  # Trending
        regime[(mom_norm.abs() < 0.5) & (vol_norm < 0.5)] = 2  # Consolidation
        regime[(vol_norm > 1) | (range_norm > 1)] = 3  # High volatility
        
        return regime.fillna(1)
    
    def _calculate_sr_proximity(self, df: pd.DataFrame) -> pd.Series:
        """Calculate support/resistance proximity."""
        # Use rolling highs and lows as S/R levels
        rolling_high = df['high'].rolling(20).max()
        rolling_low = df['low'].rolling(20).min()
        
        # Calculate proximity to S/R levels
        resistance_proximity = (rolling_high - df['close']) / df['close']
        support_proximity = (df['close'] - rolling_low) / df['close']
        
        # Combined proximity (closer to either S or R = higher value)
        sr_proximity = 1 / (1 + np.minimum(resistance_proximity, support_proximity))
        
        return sr_proximity.fillna(0)
    
    def _classify_momentum_regime(self, prices: pd.Series) -> pd.Series:
        """Classify momentum regime."""
        momentum = prices.pct_change(10)
        
        # Classify momentum regimes
        regime = pd.Series(1, index=prices.index)  # Bullish
        regime[momentum < -0.01] = 3  # Bearish
        regime[(momentum >= -0.01) & (momentum <= 0.01)] = 2  # Neutral
        
        return regime.fillna(2)
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate trend strength."""
        # Use ADX-like calculation
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional movement
        dm_plus = high - high.shift(1)
        dm_minus = low.shift(1) - low
        dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
        dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
        
        # Smoothed values
        tr_smooth = tr.rolling(14).mean()
        dm_plus_smooth = dm_plus.rolling(14).mean()
        dm_minus_smooth = dm_minus.rolling(14).mean()
        
        # Directional indicators
        di_plus = 100 * (dm_plus_smooth / tr_smooth)
        di_minus = 100 * (dm_minus_smooth / tr_smooth)
        
        # ADX (trend strength)
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(14).mean()
        
        return adx.fillna(0)
    
    def _calculate_price_action_persistence(self, df: pd.DataFrame) -> pd.Series:
        """Calculate price action persistence."""
        price_action_regime = self._classify_price_action_regime(df)
        return self._calculate_regime_persistence(price_action_regime)
    
    def _calculate_cross_timeframe_correlation(self, df: pd.DataFrame) -> pd.Series:
        """Calculate cross-timeframe correlation."""
        # Use different lookback periods as proxy for different timeframes
        short_term = df['close'].pct_change(5)
        medium_term = df['close'].pct_change(20)
        long_term = df['close'].pct_change(50)
        
        # Calculate rolling correlation
        correlation = short_term.rolling(50).corr(medium_term)
        
        return correlation.fillna(0)
    
    def _calculate_regime_correlation_proxy(self, df: pd.DataFrame) -> pd.Series:
        """Calculate regime correlation proxy."""
        # Use volatility and volume correlation as proxy
        volatility = df['close'].pct_change().rolling(20).std()
        volume = df['volume'].rolling(20).mean()
        
        correlation = volatility.rolling(50).corr(volume)
        
        return correlation.fillna(0)
    
    def _calculate_regime_duration(self, regimes: np.ndarray) -> pd.Series:
        """Calculate regime duration."""
        duration = pd.Series(index=range(len(regimes)), dtype=float)
        current_duration = 0
        current_regime = regimes[0]
        
        for i in range(len(regimes)):
            if regimes[i] == current_regime:
                current_duration += 1
            else:
                current_duration = 1
                current_regime = regimes[i]
            duration.iloc[i] = current_duration
        
        return duration
    
    def _calculate_regime_stability_score(self, regimes: np.ndarray) -> pd.Series:
        """Calculate regime stability score."""
        # Stability = inverse of regime switching frequency
        regime_changes = np.diff(regimes) != 0
        stability = pd.Series(index=range(len(regimes)), dtype=float)
        
        for i in range(len(regimes)):
            # Look back 20 periods for stability calculation
            start_idx = max(0, i - 19)
            recent_changes = np.sum(regime_changes[start_idx:i])
            stability.iloc[i] = 1 / (1 + recent_changes)
        
        return stability
    
    def _calculate_regime_transition_probability(self, regimes: np.ndarray) -> pd.Series:
        """Calculate regime transition probability."""
        # Calculate transition matrix
        unique_regimes = np.unique(regimes)
        n_regimes = len(unique_regimes)
        transition_matrix = np.zeros((n_regimes, n_regimes))
        
        for i in range(len(regimes) - 1):
            current_regime = regimes[i]
            next_regime = regimes[i + 1]
            current_idx = np.where(unique_regimes == current_regime)[0][0]
            next_idx = np.where(unique_regimes == next_regime)[0][0]
            transition_matrix[current_idx, next_idx] += 1
        
        # Normalize to probabilities
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums > 0)
        
        # Calculate transition probabilities for current regime
        transition_probs = pd.Series(index=range(len(regimes)), dtype=float)
        
        for i in range(len(regimes)):
            current_regime = regimes[i]
            current_idx = np.where(unique_regimes == current_regime)[0][0]
            
            # Probability of transitioning to any other regime
            other_probs = transition_matrix[current_idx, :]
            other_probs[current_idx] = 0  # Exclude staying in same regime
            transition_probs.iloc[i] = np.sum(other_probs)
        
        return transition_probs
    
    def _forecast_regime_persistence(self, regimes: np.ndarray) -> pd.Series:
        """Forecast regime persistence."""
        # Use historical regime durations to forecast
        regime_durations = self._calculate_regime_duration(regimes)
        
        # Forecast based on current duration and historical patterns
        forecast = pd.Series(index=range(len(regimes)), dtype=float)
        
        for i in range(len(regimes)):
            current_duration = regime_durations.iloc[i]
            
            # Look at historical durations for this regime
            current_regime = regimes[i]
            historical_durations = regime_durations[regimes == current_regime]
            
            if len(historical_durations) > 1:
                # Forecast based on historical mean and current duration
                historical_mean = historical_durations.mean()
                forecast.iloc[i] = max(1, historical_mean - current_duration)
            else:
                forecast.iloc[i] = 10  # Default forecast
        
        return forecast
    
    def _calculate_regime_strength_volatility(self, df: pd.DataFrame) -> pd.Series:
        """Calculate regime strength based on volatility."""
        volatility = df['close'].pct_change().rolling(20).std()
        volatility_trend = volatility.rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        
        # Strength = consistency of volatility (inverse of volatility of volatility)
        vol_of_vol = volatility.rolling(20).std()
        strength = 1 / (1 + vol_of_vol)
        
        return strength.fillna(0)
    
    def _calculate_regime_strength_volume(self, df: pd.DataFrame) -> pd.Series:
        """Calculate regime strength based on volume."""
        volume = df['volume'].rolling(20).mean()
        volume_consistency = 1 / (1 + volume.rolling(20).std() / volume.rolling(20).mean())
        
        return volume_consistency.fillna(0)
    
    def _calculate_regime_strength_momentum(self, df: pd.DataFrame) -> pd.Series:
        """Calculate regime strength based on momentum."""
        momentum = df['close'].pct_change(10)
        momentum_consistency = 1 / (1 + momentum.rolling(20).std())
        
        return momentum_consistency.fillna(0)
    
    def _calculate_regime_confidence_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate regime confidence score."""
        # Combine multiple strength indicators
        vol_strength = self._calculate_regime_strength_volatility(df)
        vol_strength = self._calculate_regime_strength_volume(df)
        mom_strength = self._calculate_regime_strength_momentum(df)
        
        # Average confidence score
        confidence = (vol_strength + vol_strength + mom_strength) / 3
        
        return confidence.fillna(0)
    
    def _calculate_regime_coherence_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate regime coherence score."""
        # Coherence = how well different indicators agree on regime
        volatility_regime = self._classify_volatility_regime(df['close'].pct_change().rolling(20).std())
        volume_regime = self._classify_volume_regime(df['volume'])
        momentum_regime = self._classify_momentum_regime(df['close'])
        
        # Calculate agreement between regimes
        agreement = pd.Series(index=df.index, dtype=float)
        for i in range(len(df)):
            regimes = [volatility_regime.iloc[i], volume_regime.iloc[i], momentum_regime.iloc[i]]
            # Coherence = 1 - (variance of regime classifications / max possible variance)
            regime_variance = np.var(regimes)
            max_variance = 2  # For 3 regimes with values 1, 2, 3
            coherence = 1 - (regime_variance / max_variance)
            agreement.iloc[i] = coherence
        
        return agreement.fillna(0)
    
    def _calculate_regime_change_early_warning(self, df: pd.DataFrame) -> pd.Series:
        """Calculate early warning for regime changes."""
        # Combine multiple early warning signals
        vol_change_prob = self._calculate_regime_change_probability(
            df['close'].pct_change().rolling(20).std(), window=10
        )
        vol_change_prob = self._calculate_regime_change_probability(
            df['volume'].rolling(20).mean(), window=10
        )
        mom_change_prob = self._calculate_regime_change_probability(
            df['close'].pct_change(10), window=10
        )
        
        # Combine signals
        early_warning = (vol_change_prob + vol_change_prob + mom_change_prob) / 3
        
        return early_warning.fillna(0)
    
    def _calculate_regime_weakening_indicator(self, df: pd.DataFrame) -> pd.Series:
        """Calculate regime weakening indicator."""
        # Regime weakening = decreasing regime strength
        regime_strength = self._calculate_regime_confidence_score(df)
        weakening = regime_strength.diff() < 0
        
        return weakening.astype(int).fillna(0)
    
    def _calculate_regime_transition_readiness(self, df: pd.DataFrame) -> pd.Series:
        """Calculate regime transition readiness."""
        # Readiness = combination of early warning and weakening
        early_warning = self._calculate_regime_change_early_warning(df)
        weakening = self._calculate_regime_weakening_indicator(df)
        
        readiness = early_warning * (1 + weakening)
        
        return readiness.fillna(0)


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    
    # Generate sample OHLCV data
    prices = 100 + np.cumsum(np.random.randn(1000) * 0.01)
    volumes = np.random.lognormal(10, 1, 1000)
    
    df = pd.DataFrame({
        'open': prices + np.random.randn(1000) * 0.001,
        'high': prices + np.abs(np.random.randn(1000)) * 0.002,
        'low': prices - np.abs(np.random.randn(1000)) * 0.002,
        'close': prices,
        'volume': volumes
    }, index=dates)
    
    # Initialize feature engineer
    engineer = RegimeDiscoveryFeatureEngineer()
    
    # Create regime discovery features
    features = engineer.create_regime_discovery_features(df)
    
    print(f"Created {len(features.columns)} regime discovery features")
    print(f"Feature shape: {features.shape}")
    print(f"Feature columns: {list(features.columns)}")
    
    # Display feature statistics
    print("\nFeature Statistics:")
    print(features.describe())