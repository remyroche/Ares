import pandas as pd
import numpy as np
import logging
import typing
import warnings

'Enhanced Regime Discovery Feature Engineering for Step 3.\n\nThis module creates regime-aware features specifically designed for regime discovery,\nfocusing on features that help distinguish between different market regimes.\n'
warnings.filterwarnings('ignore')

class RegimeDiscoveryFeatureEngineer:
    """Enhanced feature engineering specifically for regime discovery."""

    def __init__(self, config: Dict[str, Any]=None) -> None:
        self.config = config or {}
        self.feature_cache = {}

    def create_regime_discovery_features(self, df: pd.DataFrame, existing_regimes: Optional[np.ndarray]=None) -> pd.DataFrame:
        """
        Create comprehensive regime discovery features with optimized computation.
        
        Args:
            df: Market data with OHLCV columns
            existing_regimes: Optional existing regime labels for iterative improvement
            
        Returns:
            DataFrame with regime discovery features
        """
        self._precompute_common_features(df)
        feature_dict = {}
        feature_dict.update(self._create_regime_transition_features_vectorized(df))
        feature_dict.update(self._create_microstructure_features_vectorized(df))
        feature_dict.update(self._create_temporal_regime_features_vectorized(df))
        feature_dict.update(self._create_volatility_regime_features_vectorized(df))
        feature_dict.update(self._create_volume_regime_features_vectorized(df))
        feature_dict.update(self._create_price_action_regime_features_vectorized(df))
        if existing_regimes is not None:
            feature_dict.update(self._create_regime_persistence_features_vectorized(df, existing_regimes))
        feature_dict.update(self._create_regime_strength_features_vectorized(df))
        feature_dict.update(self._create_regime_change_warning_features_vectorized(df))
        features = pd.DataFrame(feature_dict, index=df.index)
        return features.fillna(0)

    def _precompute_common_features(self, df: pd.DataFrame) -> None:
        """Pre-compute common features for efficiency."""
        self.price_changes = df['close'].pct_change()
        self.volume_changes = df['volume'].pct_change()
        self.volatility_5 = self.price_changes.rolling(5).std()
        self.volatility_10 = self.price_changes.rolling(10).std()
        self.volatility_20 = self.price_changes.rolling(20).std()
        self.volume_mean_5 = df['volume'].rolling(5).mean()
        self.volume_mean_10 = df['volume'].rolling(10).mean()
        self.volume_mean_20 = df['volume'].rolling(20).mean()
        self.high_20 = df['high'].rolling(20).max()
        self.low_20 = df['low'].rolling(20).min()
        self.price_position = (df['close'] - self.low_20) / (self.high_20 - self.low_20)
        self.momentum_5 = df['close'].pct_change(5)
        self.momentum_10 = df['close'].pct_change(10)
        self.momentum_20 = df['close'].pct_change(20)

    def _create_regime_transition_features_vectorized(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Create regime transition features using vectorized operations."""
        features = {}
        features['regime_change_prob_volatility'] = self._calculate_regime_change_probability_vectorized(self.volatility_20, window=10)
        features['regime_change_prob_volume'] = self._calculate_regime_change_probability_vectorized(self.volume_mean_20, window=10)
        features['regime_change_prob_momentum'] = self._calculate_regime_change_probability_vectorized(self.momentum_10, window=10)
        features['regime_persistence_volatility'] = self._calculate_regime_persistence_vectorized(self.volatility_20, min_duration=5)
        features['regime_persistence_volume'] = self._calculate_regime_persistence_vectorized(self.volume_mean_20, min_duration=5)
        features['regime_transition_timing'] = self._calculate_regime_transition_timing_vectorized(df)
        features['regime_stability_volatility'] = self._calculate_regime_stability_vectorized(self.volatility_20, window=20)
        features['regime_stability_volume'] = self._calculate_regime_stability_vectorized(self.volume_mean_20, window=20)
        return features

    def _create_microstructure_features_vectorized(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Create microstructure features using vectorized operations."""
        features = {}
        hl_range = df['high'] - df['low']
        close_position = (df['close'] - df['low']) / (hl_range + 1e-08)
        features['order_flow_imbalance'] = (close_position - 0.5) * 2
        vwap = (df['high'] + df['low'] + df['close']) / 3
        price_range = df['high'] - df['low']
        price_position = (vwap - df['low']) / (price_range + 1e-08)
        features['volume_profile_skew'] = (price_position * df['volume']).rolling(20).skew()
        features['volume_profile_kurtosis'] = (price_position * df['volume']).rolling(20).kurt()
        price_impact = self.price_changes / (self.volume_changes + 1e-08)
        features['price_impact_ratio'] = price_impact
        features['price_impact_volatility'] = price_impact.rolling(20).std()
        features['spread_proxy'] = hl_range / df['close']
        features['market_depth_proxy'] = df['volume'] / (hl_range + 1e-08)
        spread_norm = (features['spread_proxy'] - features['spread_proxy'].rolling(50).mean()) / features['spread_proxy'].rolling(50).std()
        depth_norm = (features['market_depth_proxy'] - features['market_depth_proxy'].rolling(50).mean()) / features['market_depth_proxy'].rolling(50).std()
        features['liquidity_regime_indicator'] = depth_norm - spread_norm
        return features

    def _create_temporal_regime_features_vectorized(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Create temporal regime features using vectorized operations."""
        features = {}
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            return features
        hour = df.index.hour
        day_of_week = df.index.dayofweek
        features['hour_of_day'] = hour
        features['day_of_week'] = day_of_week
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        features['day_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        features['day_cos'] = np.cos(2 * np.pi * day_of_week / 7)
        vol_factor = 1 / (1 + self.volatility_20 / self.volatility_20.rolling(50, min_periods=1).mean())
        volume_factor = 1 / (1 + self.volume_mean_20 / self.volume_mean_20.rolling(50, min_periods=1).mean())
        features['regime_duration_forecast'] = vol_factor * volume_factor * 20
        session = np.zeros(len(df))
        session[hour < 8] = 1
        session[(hour >= 8) & (hour < 16)] = 2
        session[hour >= 16] = 3
        features['session_regime_indicator'] = session
        return features

    def _create_volatility_regime_features_vectorized(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Create volatility regime features using vectorized operations."""
        features = {}
        features['volatility_5'] = self.volatility_5
        features['volatility_10'] = self.volatility_10
        features['volatility_20'] = self.volatility_20
        features['vol_of_vol_20'] = self.volatility_20.rolling(20).std()
        features['vol_of_vol_50'] = self.volatility_20.rolling(50).std()
        low_threshold = self.volatility_20.rolling(100).quantile(0.33)
        high_threshold = self.volatility_20.rolling(100).quantile(0.67)
        vol_regime = np.ones(len(df))
        vol_regime[self.volatility_20 > high_threshold] = 3
        vol_regime[(self.volatility_20 > low_threshold) & (self.volatility_20 <= high_threshold)] = 2
        features['volatility_regime'] = vol_regime
        features['volatility_clustering'] = self.volatility_20.rolling(50).apply(lambda x: x.autocorr(lag=1) if len(x) > 1 else 0)
        features['volatility_persistence'] = self.volatility_20.rolling(50).apply(lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 1 else 0)
        return features

    def _create_volume_regime_features_vectorized(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Create volume regime features using vectorized operations."""
        features = {}
        low_threshold = df['volume'].rolling(100).quantile(0.33)
        high_threshold = df['volume'].rolling(100).quantile(0.67)
        vol_regime = np.ones(len(df))
        vol_regime[df['volume'] > high_threshold] = 3
        vol_regime[(df['volume'] > low_threshold) & (df['volume'] <= high_threshold)] = 2
        features['volume_regime'] = vol_regime
        features['volume_momentum_interaction'] = self.volume_changes * self.momentum_5
        price_momentum = self.momentum_5
        volume_momentum = self.volume_changes
        features['volume_price_divergence'] = (price_momentum * volume_momentum < 0).astype(int)
        rolling_mean = df['volume'].rolling(20).mean()
        rolling_std = df['volume'].rolling(20).std()
        features['volume_spike_indicator'] = (df['volume'] > rolling_mean + 2 * rolling_std).astype(int)
        return features

    def _create_price_action_regime_features_vectorized(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Create price action regime features using vectorized operations."""
        features = {}
        mom_norm = (self.momentum_10 - self.momentum_10.rolling(50).mean()) / self.momentum_10.rolling(50).std()
        vol_norm = (self.volatility_20 - self.volatility_20.rolling(50).mean()) / self.volatility_20.rolling(50).std()
        range_size = (df['high'] - df['low']) / df['close']
        range_norm = (range_size - range_size.rolling(50).mean()) / range_size.rolling(50).std()
        price_action_regime = np.ones(len(df))
        price_action_regime[(np.abs(mom_norm) < 0.5) & (vol_norm < 0.5)] = 2
        price_action_regime[(vol_norm > 1) | (range_norm > 1)] = 3
        features['price_action_regime'] = price_action_regime
        resistance_proximity = (self.high_20 - df['close']) / df['close']
        support_proximity = (df['close'] - self.low_20) / df['close']
        features['sr_proximity'] = 1 / (1 + np.minimum(resistance_proximity, support_proximity))
        momentum_regime = np.ones(len(df))
        momentum_regime[self.momentum_10 < -0.01] = 3
        momentum_regime[(self.momentum_10 >= -0.01) & (self.momentum_10 <= 0.01)] = 2
        features['momentum_regime'] = momentum_regime
        return features

    def _create_regime_persistence_features_vectorized(self, df: pd.DataFrame, existing_regimes: np.ndarray) -> Dict[str, np.ndarray]:
        """Create regime persistence features using vectorized operations."""
        features = {}
        regime_changes = np.diff(existing_regimes) != 0
        regime_duration = np.zeros(len(existing_regimes))
        current_duration = 0
        current_regime = existing_regimes[0]
        for i in range(len(existing_regimes)):
            if existing_regimes[i] == current_regime:
                current_duration += 1
            else:
                current_duration = 1
                current_regime = existing_regimes[i]
            regime_duration[i] = current_duration
        features['regime_duration'] = regime_duration
        stability = np.zeros(len(existing_regimes))
        for i in range(len(existing_regimes)):
            start_idx = max(0, i - 19)
            recent_changes = np.sum(regime_changes[start_idx:i])
            stability[i] = 1 / (1 + recent_changes)
        features['regime_stability_score'] = stability
        return features

    def _create_regime_strength_features_vectorized(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Create regime strength features using vectorized operations."""
        features = {}
        volatility_trend = self.volatility_20.rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        vol_of_vol = self.volatility_20.rolling(20).std()
        features['regime_strength_volatility'] = 1 / (1 + vol_of_vol)
        volume_consistency = 1 / (1 + df['volume'].rolling(20).std() / df['volume'].rolling(20).mean())
        features['regime_strength_volume'] = volume_consistency
        momentum_consistency = 1 / (1 + self.momentum_10.rolling(20).std())
        features['regime_strength_momentum'] = momentum_consistency
        confidence = (features['regime_strength_volatility'] + features['regime_strength_volume'] + features['regime_strength_momentum']) / 3
        features['regime_confidence_score'] = confidence
        return features

    def _create_regime_change_warning_features_vectorized(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Create regime change warning features using vectorized operations."""
        features = {}
        vol_change_prob_volatility = self._calculate_regime_change_probability_vectorized(self.volatility_20, window=10)
        vol_change_prob_volume = self._calculate_regime_change_probability_vectorized(self.volume_mean_20, window=10)
        mom_change_prob = self._calculate_regime_change_probability_vectorized(self.momentum_10, window=10)
        early_warning = (vol_change_prob_volatility + vol_change_prob_volume + mom_change_prob) / 3
        features['regime_change_early_warning'] = early_warning
        regime_strength = features.get('regime_confidence_score', np.zeros(len(df)))
        weakening = np.diff(regime_strength) < 0
        features['regime_weakening_indicator'] = np.concatenate([[0], weakening.astype(int)])
        readiness = early_warning * (1 + features['regime_weakening_indicator'])
        features['regime_transition_readiness'] = readiness
        return features

    def _calculate_regime_change_probability_vectorized(self, series: pd.Series, window: int=10) -> np.ndarray:
        """Calculate regime change probability using vectorized operations."""
        rolling_mean = series.rolling(window).mean()
        rolling_std = series.rolling(window).std()
        z_scores = (series - rolling_mean) / rolling_std
        regime_change_prob = np.abs(z_scores).rolling(window).mean()
        return regime_change_prob.fillna(0).values

    def _calculate_regime_persistence_vectorized(self, series: pd.Series, min_duration: int=5) -> np.ndarray:
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

    def _calculate_regime_transition_timing_vectorized(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate regime transition timing using vectorized operations."""
        vol_norm = (self.volatility_20 - self.volatility_20.rolling(50).mean()) / self.volatility_20.rolling(50).std()
        vol_vol_norm = (self.volume_mean_20 - self.volume_mean_20.rolling(50).mean()) / self.volume_mean_20.rolling(50).std()
        mom_norm = (self.momentum_10 - self.momentum_10.rolling(50).mean()) / self.momentum_10.rolling(50).std()
        transition_timing = (vol_norm + vol_vol_norm + mom_norm) / 3
        return transition_timing.fillna(0).values

    def _calculate_regime_stability_vectorized(self, series: pd.Series, window: int=20) -> np.ndarray:
        """Calculate regime stability using vectorized operations."""
        rolling_mean = series.rolling(window).mean()
        rolling_std = series.rolling(window).std()
        cv = rolling_std / rolling_mean
        stability = 1 / (1 + cv)
        return stability.fillna(0).values

    def _create_regime_transition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features that predict regime transitions."""
        features = pd.DataFrame(index=df.index)
        features['regime_change_prob_volatility'] = self._calculate_regime_change_probability(df['close'].pct_change().rolling(20).std(), window=10)
        features['regime_change_prob_volume'] = self._calculate_regime_change_probability(df['volume'].rolling(20).mean(), window=10)
        features['regime_change_prob_momentum'] = self._calculate_regime_change_probability(df['close'].pct_change(10), window=10)
        features['regime_persistence_volatility'] = self._calculate_regime_persistence(df['close'].pct_change().rolling(20).std(), min_duration=5)
        features['regime_persistence_volume'] = self._calculate_regime_persistence(df['volume'].rolling(20).mean(), min_duration=5)
        features['regime_transition_timing'] = self._calculate_regime_transition_timing(df)
        features['regime_stability_volatility'] = self._calculate_regime_stability(df['close'].pct_change().rolling(20).std(), window=20)
        features['regime_stability_volume'] = self._calculate_regime_stability(df['volume'].rolling(20).mean(), window=20)
        return features

    def _create_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market microstructure features for regime detection."""
        features = pd.DataFrame(index=df.index)
        features['order_flow_imbalance'] = self._calculate_order_flow_imbalance(df)
        features['volume_profile_skew'] = self._calculate_volume_profile_skew(df)
        features['volume_profile_kurtosis'] = self._calculate_volume_profile_kurtosis(df)
        features['price_impact_ratio'] = self._calculate_price_impact_ratio(df)
        features['price_impact_volatility'] = self._calculate_price_impact_volatility(df)
        features['spread_proxy'] = self._calculate_spread_proxy(df)
        features['market_depth_proxy'] = self._calculate_market_depth_proxy(df)
        features['liquidity_regime_indicator'] = self._calculate_liquidity_regime_indicator(df)
        return features

    def _create_temporal_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal regime features."""
        features = pd.DataFrame(index=df.index)
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)
        features['hour_of_day'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek
        features['day_of_month'] = df.index.day
        features['month_of_year'] = df.index.month
        features['hour_sin'] = np.sin(2 * np.pi * features['hour_of_day'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour_of_day'] / 24)
        features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        features['regime_duration_forecast'] = self._forecast_regime_duration(df)
        features['time_based_volatility'] = self._calculate_time_based_volatility(df)
        features['session_regime_indicator'] = self._calculate_session_regime_indicator(df)
        return features

    def _create_volatility_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volatility regime features."""
        features = pd.DataFrame(index=df.index)
        for window in [5, 10, 20, 50]:
            features[f'volatility_{window}'] = df['close'].pct_change().rolling(window).std()
            features[f'volatility_{window}_normalized'] = features[f'volatility_{window}'] / df['close']
        features['vol_of_vol_20'] = features['volatility_20'].rolling(20).std()
        features['vol_of_vol_50'] = features['volatility_50'].rolling(50).std()
        features['volatility_regime'] = self._classify_volatility_regime(features['volatility_20'])
        features['volatility_clustering'] = self._calculate_volatility_clustering(df['close'])
        features['volatility_persistence'] = self._calculate_volatility_persistence(features['volatility_20'])
        features['volatility_regime_transition'] = self._detect_volatility_regime_transitions(features['volatility_20'])
        return features

    def _create_volume_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volume regime features."""
        features = pd.DataFrame(index=df.index)
        features['volume_regime'] = self._classify_volume_regime(df['volume'])
        features['volume_momentum_interaction'] = df['volume'].pct_change(5) * df['close'].pct_change(5)
        features['volume_price_divergence'] = self._calculate_volume_price_divergence(df)
        features['volume_regime_persistence'] = self._calculate_volume_regime_persistence(df['volume'])
        features['volume_spike_indicator'] = self._detect_volume_spikes(df['volume'])
        features['volume_trend_change'] = self._detect_volume_trend_changes(df['volume'])
        return features

    def _create_price_action_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price action regime features."""
        features = pd.DataFrame(index=df.index)
        features['price_action_regime'] = self._classify_price_action_regime(df)
        features['sr_proximity'] = self._calculate_sr_proximity(df)
        features['momentum_regime'] = self._classify_momentum_regime(df['close'])
        features['trend_strength'] = self._calculate_trend_strength(df)
        features['price_action_persistence'] = self._calculate_price_action_persistence(df)
        return features

    def _create_cross_asset_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cross-asset regime features (if multiple assets available)."""
        features = pd.DataFrame(index=df.index)
        features['cross_timeframe_correlation'] = self._calculate_cross_timeframe_correlation(df)
        features['regime_correlation_proxy'] = self._calculate_regime_correlation_proxy(df)
        return features

    def _create_regime_persistence_features(self, df: pd.DataFrame, existing_regimes: np.ndarray) -> pd.DataFrame:
        """Create features based on existing regime persistence."""
        features = pd.DataFrame(index=df.index)
        features['regime_duration'] = self._calculate_regime_duration(existing_regimes)
        features['regime_stability_score'] = self._calculate_regime_stability_score(existing_regimes)
        features['regime_transition_probability'] = self._calculate_regime_transition_probability(existing_regimes)
        features['regime_persistence_forecast'] = self._forecast_regime_persistence(existing_regimes)
        return features

    def _create_regime_strength_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create regime strength features."""
        features = pd.DataFrame(index=df.index)
        features['regime_strength_volatility'] = self._calculate_regime_strength_volatility(df)
        features['regime_strength_volume'] = self._calculate_regime_strength_volume(df)
        features['regime_strength_momentum'] = self._calculate_regime_strength_momentum(df)
        features['regime_confidence_score'] = self._calculate_regime_confidence_score(df)
        features['regime_coherence_score'] = self._calculate_regime_coherence_score(df)
        return features

    def _create_regime_change_warning_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create early warning features for regime changes."""
        features = pd.DataFrame(index=df.index)
        features['regime_change_early_warning'] = self._calculate_regime_change_early_warning(df)
        features['regime_weakening_indicator'] = self._calculate_regime_weakening_indicator(df)
        features['regime_transition_readiness'] = self._calculate_regime_transition_readiness(df)
        return features

    def _calculate_regime_change_probability(self, series: pd.Series, window: int=10) -> pd.Series:
        """Calculate probability of regime change based on series characteristics."""
        rolling_mean = series.rolling(window).mean()
        rolling_std = series.rolling(window).std()
        z_scores = (series - rolling_mean) / rolling_std
        regime_change_prob = np.abs(z_scores).rolling(window).mean()
        return regime_change_prob.fillna(0)

    def _calculate_regime_persistence(self, series: pd.Series, min_duration: int=5) -> pd.Series:
        """Calculate how long current regime has persisted."""
        rolling_mean = series.rolling(min_duration).mean()
        regime_changes = (np.abs(series - rolling_mean) > rolling_mean.rolling(min_duration).std()).astype(int)
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
        volatility_signal = df['close'].pct_change().rolling(20).std()
        volume_signal = df['volume'].rolling(20).mean()
        momentum_signal = df['close'].pct_change(10)
        vol_norm = (volatility_signal - volatility_signal.rolling(50).mean()) / volatility_signal.rolling(50).std()
        vol_vol_norm = (volume_signal - volume_signal.rolling(50).mean()) / volume_signal.rolling(50).std()
        mom_norm = (momentum_signal - momentum_signal.rolling(50).mean()) / momentum_signal.rolling(50).std()
        transition_timing = (vol_norm + vol_vol_norm + mom_norm) / 3
        return transition_timing.fillna(0)

    def _calculate_regime_stability(self, series: pd.Series, window: int=20) -> pd.Series:
        """Calculate regime stability based on series consistency."""
        rolling_mean = series.rolling(window).mean()
        rolling_std = series.rolling(window).std()
        cv = rolling_std / rolling_mean
        stability = 1 / (1 + cv)
        return stability.fillna(0)

    def _calculate_order_flow_imbalance(self, df: pd.DataFrame) -> pd.Series:
        """Calculate order flow imbalance proxy."""
        hl_range = df['high'] - df['low']
        close_position = (df['close'] - df['low']) / (hl_range + 1e-08)
        order_flow_imbalance = (close_position - 0.5) * 2
        return order_flow_imbalance.fillna(0)

    def _calculate_volume_profile_skew(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volume profile skewness."""
        vwap = (df['high'] + df['low'] + df['close']) / 3
        price_range = df['high'] - df['low']
        price_position = (vwap - df['low']) / (price_range + 1e-08)
        volume_skew = (price_position * df['volume']).rolling(20).skew()
        return volume_skew.fillna(0)

    def _calculate_volume_profile_kurtosis(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volume profile kurtosis."""
        vwap = (df['high'] + df['low'] + df['close']) / 3
        price_range = df['high'] - df['low']
        price_position = (vwap - df['low']) / (price_range + 1e-08)
        volume_kurtosis = (price_position * df['volume']).rolling(20).kurt()
        return volume_kurtosis.fillna(0)

    def _calculate_price_impact_ratio(self, df: pd.DataFrame) -> pd.Series:
        """Calculate price impact ratio."""
        price_change = df['close'].pct_change()
        volume_change = df['volume'].pct_change()
        price_impact = price_change / (volume_change + 1e-08)
        return price_impact.fillna(0)

    def _calculate_price_impact_volatility(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volatility of price impact."""
        price_impact = self._calculate_price_impact_ratio(df)
        price_impact_vol = price_impact.rolling(20).std()
        return price_impact_vol.fillna(0)

    def _calculate_spread_proxy(self, df: pd.DataFrame) -> pd.Series:
        """Calculate bid-ask spread proxy."""
        spread_proxy = (df['high'] - df['low']) / df['close']
        return spread_proxy.fillna(0)

    def _calculate_market_depth_proxy(self, df: pd.DataFrame) -> pd.Series:
        """Calculate market depth proxy."""
        price_range = df['high'] - df['low']
        market_depth = df['volume'] / (price_range + 1e-08)
        return market_depth.fillna(0)

    def _calculate_liquidity_regime_indicator(self, df: pd.DataFrame) -> pd.Series:
        """Calculate liquidity regime indicator."""
        spread_proxy = self._calculate_spread_proxy(df)
        depth_proxy = self._calculate_market_depth_proxy(df)
        spread_norm = (spread_proxy - spread_proxy.rolling(50).mean()) / spread_proxy.rolling(50).std()
        depth_norm = (depth_proxy - depth_proxy.rolling(50).mean()) / depth_proxy.rolling(50).std()
        liquidity_regime = depth_norm - spread_norm
        return liquidity_regime.fillna(0)

    def _forecast_regime_duration(self, df: pd.DataFrame) -> pd.Series:
        """Forecast regime duration based on current market conditions."""
        volatility = df['close'].pct_change().rolling(20).std()
        volume = df['volume'].rolling(20).mean()
        vol_factor = 1 / (1 + volatility / volatility.rolling(50, min_periods=1).mean())
        volume_factor = 1 / (1 + volume / volume.rolling(50, min_periods=1).mean())
        duration_forecast = vol_factor * volume_factor * 20
        return duration_forecast.fillna(20)

    def _calculate_time_based_volatility(self, df: pd.DataFrame) -> pd.Series:
        """Calculate time-based volatility patterns."""
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            return pd.Series(0, index=df.index)
        hour_volatility = df['close'].pct_change().groupby(df.index.hour).std()
        time_based_vol = df.index.hour.map(hour_volatility)
        return time_based_vol.fillna(0)

    def _calculate_session_regime_indicator(self, df: pd.DataFrame) -> pd.Series:
        """Calculate session-based regime indicator."""
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            return pd.Series(0, index=df.index)
        hour = df.index.hour
        session = pd.Series(0, index=df.index)
        session[hour < 8] = 1
        session[(hour >= 8) & (hour < 16)] = 2
        session[hour >= 16] = 3
        return session

    def _classify_volatility_regime(self, volatility: pd.Series) -> pd.Series:
        """Classify volatility regime."""
        low_threshold = volatility.rolling(100).quantile(0.33)
        high_threshold = volatility.rolling(100).quantile(0.67)
        regime = pd.Series(1, index=volatility.index)
        regime[volatility > high_threshold] = 3
        regime[(volatility > low_threshold) & (volatility <= high_threshold)] = 2
        return regime.fillna(1)

    def _calculate_volatility_clustering(self, prices: pd.Series) -> pd.Series:
        """Calculate volatility clustering indicator."""
        returns = prices.pct_change()
        volatility = returns.rolling(20).std()
        clustering = volatility.rolling(50).apply(lambda x: x.autocorr(lag=1) if len(x) > 1 else 0)
        return clustering.fillna(0)

    def _calculate_volatility_persistence(self, volatility: pd.Series) -> pd.Series:
        """Calculate volatility persistence."""
        persistence = volatility.rolling(50).apply(lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 1 else 0)
        return persistence.fillna(0)

    def _detect_volatility_regime_transitions(self, volatility: pd.Series) -> pd.Series:
        """Detect volatility regime transitions."""
        rolling_mean = volatility.rolling(20).mean()
        rolling_std = volatility.rolling(20).std()
        z_scores = (volatility - rolling_mean) / rolling_std
        transitions = (np.abs(z_scores) > 2).astype(int)
        return transitions.fillna(0)

    def _classify_volume_regime(self, volume: pd.Series) -> pd.Series:
        """Classify volume regime."""
        low_threshold = volume.rolling(100).quantile(0.33)
        high_threshold = volume.rolling(100).quantile(0.67)
        regime = pd.Series(1, index=volume.index)
        regime[volume > high_threshold] = 3
        regime[(volume > low_threshold) & (volume <= high_threshold)] = 2
        return regime.fillna(1)

    def _calculate_volume_price_divergence(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volume-price divergence."""
        price_momentum = df['close'].pct_change(5)
        volume_momentum = df['volume'].pct_change(5)
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
        spikes = (volume > rolling_mean + 2 * rolling_std).astype(int)
        return spikes.fillna(0)

    def _detect_volume_trend_changes(self, volume: pd.Series) -> pd.Series:
        """Detect volume trend changes."""
        volume_trend = volume.rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        trend_changes = (volume_trend.diff() * volume_trend.shift(1) < 0).astype(int)
        return trend_changes.fillna(0)

    def _classify_price_action_regime(self, df: pd.DataFrame) -> pd.Series:
        """Classify price action regime."""
        momentum = df['close'].pct_change(10)
        volatility = df['close'].pct_change().rolling(20).std()
        range_size = (df['high'] - df['low']) / df['close']
        mom_norm = (momentum - momentum.rolling(50).mean()) / momentum.rolling(50).std()
        vol_norm = (volatility - volatility.rolling(50).mean()) / volatility.rolling(50).std()
        range_norm = (range_size - range_size.rolling(50).mean()) / range_size.rolling(50).std()
        regime = pd.Series(1, index=df.index)
        regime[(mom_norm.abs() < 0.5) & (vol_norm < 0.5)] = 2
        regime[(vol_norm > 1) | (range_norm > 1)] = 3
        return regime.fillna(1)

    def _calculate_sr_proximity(self, df: pd.DataFrame) -> pd.Series:
        """Calculate support/resistance proximity."""
        rolling_high = df['high'].rolling(20).max()
        rolling_low = df['low'].rolling(20).min()
        resistance_proximity = (rolling_high - df['close']) / df['close']
        support_proximity = (df['close'] - rolling_low) / df['close']
        sr_proximity = 1 / (1 + np.minimum(resistance_proximity, support_proximity))
        return sr_proximity.fillna(0)

    def _classify_momentum_regime(self, prices: pd.Series) -> pd.Series:
        """Classify momentum regime."""
        momentum = prices.pct_change(10)
        regime = pd.Series(1, index=prices.index)
        regime[momentum < -0.01] = 3
        regime[(momentum >= -0.01) & (momentum <= 0.01)] = 2
        return regime.fillna(2)

    def _calculate_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate trend strength."""
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        dm_plus = high - high.shift(1)
        dm_minus = low.shift(1) - low
        dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
        dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
        tr_smooth = tr.rolling(14).mean()
        dm_plus_smooth = dm_plus.rolling(14).mean()
        dm_minus_smooth = dm_minus.rolling(14).mean()
        di_plus = 100 * (dm_plus_smooth / tr_smooth)
        di_minus = 100 * (dm_minus_smooth / tr_smooth)
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(14).mean()
        return adx.fillna(0)

    def _calculate_price_action_persistence(self, df: pd.DataFrame) -> pd.Series:
        """Calculate price action persistence."""
        price_action_regime = self._classify_price_action_regime(df)
        return self._calculate_regime_persistence(price_action_regime)

    def _calculate_cross_timeframe_correlation(self, df: pd.DataFrame) -> pd.Series:
        """Calculate cross-timeframe correlation."""
        short_term = df['close'].pct_change(5)
        medium_term = df['close'].pct_change(20)
        long_term = df['close'].pct_change(50)
        correlation = short_term.rolling(50).corr(medium_term)
        return correlation.fillna(0)

    def _calculate_regime_correlation_proxy(self, df: pd.DataFrame) -> pd.Series:
        """Calculate regime correlation proxy."""
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
        regime_changes = np.diff(regimes) != 0
        stability = pd.Series(index=range(len(regimes)), dtype=float)
        for i in range(len(regimes)):
            start_idx = max(0, i - 19)
            recent_changes = np.sum(regime_changes[start_idx:i])
            stability.iloc[i] = 1 / (1 + recent_changes)
        return stability

    def _calculate_regime_transition_probability(self, regimes: np.ndarray) -> pd.Series:
        """Calculate regime transition probability."""
        unique_regimes = np.unique(regimes)
        n_regimes = len(unique_regimes)
        transition_matrix = np.zeros((n_regimes, n_regimes))
        for i in range(len(regimes) - 1):
            current_regime = regimes[i]
            next_regime = regimes[i + 1]
            current_idx = np.where(unique_regimes == current_regime)[0][0]
            next_idx = np.where(unique_regimes == next_regime)[0][0]
            transition_matrix[current_idx, next_idx] += 1
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums > 0)
        transition_probs = pd.Series(index=range(len(regimes)), dtype=float)
        for i in range(len(regimes)):
            current_regime = regimes[i]
            current_idx = np.where(unique_regimes == current_regime)[0][0]
            other_probs = transition_matrix[current_idx, :]
            other_probs[current_idx] = 0
            transition_probs.iloc[i] = np.sum(other_probs)
        return transition_probs

    def _forecast_regime_persistence(self, regimes: np.ndarray) -> pd.Series:
        """Forecast regime persistence."""
        regime_durations = self._calculate_regime_duration(regimes)
        forecast = pd.Series(index=range(len(regimes)), dtype=float)
        for i in range(len(regimes)):
            current_duration = regime_durations.iloc[i]
            current_regime = regimes[i]
            historical_durations = regime_durations[regimes == current_regime]
            if len(historical_durations) > 1:
                historical_mean = historical_durations.mean()
                forecast.iloc[i] = max(1, historical_mean - current_duration)
            else:
                forecast.iloc[i] = 10
        return forecast

    def _calculate_regime_strength_volatility(self, df: pd.DataFrame) -> pd.Series:
        """Calculate regime strength based on volatility."""
        volatility = df['close'].pct_change().rolling(20).std()
        volatility_trend = volatility.rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
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
        vol_strength = self._calculate_regime_strength_volatility(df)
        volume_strength = self._calculate_regime_strength_volume(df)
        mom_strength = self._calculate_regime_strength_momentum(df)
        confidence = (vol_strength + volume_strength + mom_strength) / 3
        return confidence.fillna(0)

    def _calculate_regime_coherence_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate regime coherence score."""
        volatility_regime = self._classify_volatility_regime(df['close'].pct_change().rolling(20).std())
        volume_regime = self._classify_volume_regime(df['volume'])
        momentum_regime = self._classify_momentum_regime(df['close'])
        agreement = pd.Series(index=df.index, dtype=float)
        for i in range(len(df)):
            regimes = [volatility_regime.iloc[i], volume_regime.iloc[i], momentum_regime.iloc[i]]
            regime_variance = np.var(regimes)
            max_variance = 2
            coherence = 1 - regime_variance / max_variance
            agreement.iloc[i] = coherence
        return agreement.fillna(0)

    def _calculate_regime_change_early_warning(self, df: pd.DataFrame) -> pd.Series:
        """Calculate early warning for regime changes."""
        vol_change_prob_volatility = self._calculate_regime_change_probability(df['close'].pct_change().rolling(20).std(), window=10)
        vol_change_prob_volume = self._calculate_regime_change_probability(df['volume'].rolling(20).mean(), window=10)
        mom_change_prob = self._calculate_regime_change_probability(df['close'].pct_change(10), window=10)
        early_warning = (vol_change_prob_volatility + vol_change_prob_volume + mom_change_prob) / 3
        return early_warning.fillna(0)

    def _calculate_regime_weakening_indicator(self, df: pd.DataFrame) -> pd.Series:
        """Calculate regime weakening indicator."""
        regime_strength = self._calculate_regime_confidence_score(df)
        weakening = regime_strength.diff() < 0
        return weakening.astype(int).fillna(0)

    def _calculate_regime_transition_readiness(self, df: pd.DataFrame) -> pd.Series:
        """Calculate regime transition readiness."""
        early_warning = self._calculate_regime_change_early_warning(df)
        weakening = self._calculate_regime_weakening_indicator(df)
        readiness = early_warning * (1 + weakening)
        return readiness.fillna(0)
if __name__ == '__main__':
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    prices = 100 + np.cumsum(np.random.randn(1000) * 0.01)
    volumes = np.random.lognormal(10, 1, 1000)
    df = pd.DataFrame({'open': prices + np.random.randn(1000) * 0.001, 'high': prices + np.abs(np.random.randn(1000)) * 0.002, 'low': prices - np.abs(np.random.randn(1000)) * 0.002, 'close': prices, 'volume': volumes}, index=dates)
    engineer = RegimeDiscoveryFeatureEngineer()
    features = engineer.create_regime_discovery_features(df)
    print(f'Created {len(features.columns)} regime discovery features')
    print(f'Feature shape: {features.shape}')
    print(f'Feature columns: {list(features.columns)}')
    print('\nFeature Statistics:')
    print(features.describe())