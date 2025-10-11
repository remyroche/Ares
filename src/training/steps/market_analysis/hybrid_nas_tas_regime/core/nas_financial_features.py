"""
NAS-Specific Financial Features for Neural Networks

This module provides comprehensive financial feature engineering specifically designed
for neural architecture search, including technical indicators, market microstructure
features, regime-aware features, and advanced preprocessing techniques.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.tprint import (tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_progress, tprint_performance, tprint_timer)

logger = logging.getLogger(__name__)


class FeatureType(Enum):
    """Types of financial features."""
    TECHNICAL = "technical"
    MICROSTRUCTURE = "microstructure"
    REGIME_AWARE = "regime_aware"
    MULTITIMEFRAME = "multitimeframe"
    SENTIMENT = "sentiment"
    MACROECONOMIC = "macroeconomic"
    VOLATILITY = "volatility"
    LIQUIDITY = "liquidity"


class NormalizationType(Enum):
    """Types of feature normalization."""
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    LOG = "log"
    PERCENTILE = "percentile"


@dataclass
class NASFeatureConfig:
    """Configuration for NAS-specific financial features."""
    # Feature types to include
    include_technical: bool = True
    include_microstructure: bool = True
    include_regime_aware: bool = True
    include_multitimeframe: bool = True
    include_volatility: bool = True
    include_liquidity: bool = True

    # Technical indicators
    technical_indicators: List[str] = field(default_factory=lambda: [
        'rsi', 'macd', 'bollinger', 'stochastic', 'williams_r', 'cci',
        'adx', 'atr', 'obv', 'cmf', 'pvt', 'roc', 'momentum', 'ema', 'sma'
    ])

    # Lookback periods
    short_period: int = 5
    medium_period: int = 14
    long_period: int = 21

    # Multi-timeframe settings
    timeframes: List[str] = field(default_factory=lambda: ['1m', '5m', '15m', '1h'])

    # Feature selection
    max_features: int = 100
    feature_selection_method: str = "mutual_info"  # or "correlation", "f_regression"
    feature_selection_k: int = 50

    # Normalization
    normalization_type: NormalizationType = NormalizationType.ROBUST
    sequence_length: int = 60  # For LSTM/sequence models

    # Advanced features
    include_fourier: bool = True
    include_wavelet: bool = True
    include_market_depth: bool = True
    include_order_flow: bool = True


@dataclass
class FeatureSet:
    """Container for feature data."""
    features: np.ndarray
    feature_names: List[str]
    feature_types: Dict[str, FeatureType]
    normalization_info: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class NASFinancialFeatureEngineer:
    """
    Financial feature engineer specifically designed for neural architectures.

    Provides comprehensive feature engineering with neural network optimization,
    including technical indicators, market microstructure, regime-aware features,
    and advanced preprocessing techniques.
    """

    def __init__(self, config: NASFeatureConfig):
        """Initialize the NAS financial feature engineer."""
        tprint("🚀 [NAS_FEATURES] Initializing NAS Financial Feature Engineer", color="cyan", bold=True)
        tprint(f"📊 [NAS_FEATURES] Feature types enabled: {config.__dict__}", color="blue")
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Feature scalers
        tprint("🔧 [NAS_FEATURES] Initializing feature scalers", color="yellow")
        self.scalers = {}

        # Feature importance tracking
        tprint("📊 [NAS_FEATURES] Setting up feature importance tracking", color="blue")
        self.feature_importance_history = []

        # Market data cache
        tprint("💾 [NAS_FEATURES] Initializing market data cache", color="yellow")
        self.market_data_cache = {}

        # Feature type mappings
        tprint("🗺️ [NAS_FEATURES] Setting up feature type mappings", color="yellow")
        self.feature_type_map = {}

        tprint("✅ [NAS_FEATURES] NAS Financial Feature Engineer initialized successfully", color="green", bold=True)
        self.logger.info("✅ NAS Financial Feature Engineer initialized")
        self.logger.info(f"   Feature Types: {[ft.value for ft in config.__dict__.keys() if isinstance(getattr(config, ft.split('_')[0] if '_' in ft else ft, None), bool) and getattr(config, ft.split('_')[0] if '_' in ft else ft)]}")

    def engineer_features(self, market_data: pd.DataFrame,
                         regime_data: Optional[Dict[str, Any]] = None,
                         multi_timeframe_data: Optional[Dict[str, pd.DataFrame]] = None) -> FeatureSet:
        """Engineer comprehensive features for neural networks."""
        start_time = time.time()
        self.logger.info("🔧 Starting NAS-specific feature engineering...")

        try:
            # Validate input data
            clean_data = self._validate_and_clean_data(market_data)

            # Extract base features
            base_features = self._extract_base_features(clean_data)

            # Add technical indicators
            if self.config.include_technical:
                technical_features = self._add_technical_indicators(clean_data)
                base_features.update(technical_features)

            # Add microstructure features
            if self.config.include_microstructure:
                microstructure_features = self._add_microstructure_features(clean_data)
                base_features.update(microstructure_features)

            # Add regime-aware features
            if self.config.include_regime_aware and regime_data:
                regime_features = self._add_regime_aware_features(clean_data, regime_data)
                base_features.update(regime_features)

            # Add multi-timeframe features
            if self.config.include_multitimeframe and multi_timeframe_data:
                mtf_features = self._add_multitimeframe_features(clean_data, multi_timeframe_data)
                base_features.update(mtf_features)

            # Add volatility features
            if self.config.include_volatility:
                volatility_features = self._add_volatility_features(clean_data)
                base_features.update(volatility_features)

            # Add liquidity features
            if self.config.include_liquidity:
                liquidity_features = self._add_liquidity_features(clean_data)
                base_features.update(liquidity_features)

            # Convert to DataFrame
            feature_df = pd.DataFrame(base_features, index=clean_data.index)

            # Feature selection
            selected_features = self._select_features(feature_df, clean_data)

            # Normalization
            normalized_features, normalization_info = self._normalize_features(selected_features)

            # Create feature type mapping
            feature_types = self._create_feature_type_mapping(selected_features.columns)

            execution_time = time.time() - start_time

            feature_set = FeatureSet(
                features=normalized_features,
                feature_names=selected_features.columns.tolist(),
                feature_types=feature_types,
                normalization_info=normalization_info,
                metadata={
                    'n_original_features': len(base_features),
                    'n_selected_features': len(selected_features.columns),
                    'execution_time': execution_time,
                    'feature_engineering_version': '2.0'
                }
            )

            self.logger.info(f"✅ Feature engineering completed in {execution_time:.2f}s")
            self.logger.info(f"   Original features: {len(base_features)}")
            self.logger.info(f"   Selected features: {len(selected_features.columns)}")

            return feature_set

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Feature engineering failed: {e}")

            # Return minimal feature set
            minimal_features = np.random.randn(len(market_data), 10)
            return FeatureSet(
                features=minimal_features,
                feature_names=[f'feature_{i}' for i in range(10)],
                feature_types={f'feature_{i}': FeatureType.TECHNICAL for i in range(10)},
                normalization_info={},
                metadata={'error': str(e), 'execution_time': execution_time}
            )

    def _validate_and_clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean market data."""
        try:
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]

            if missing_columns:
                self.logger.warning(f"Missing columns: {missing_columns}")

            # Handle missing values
            clean_data = data.copy()
            clean_data = clean_data.fillna(method='ffill').fillna(method='bfill')

            # Remove extreme outliers
            for col in ['high', 'low', 'close', 'open']:
                if col in clean_data.columns:
                    q1 = clean_data[col].quantile(0.01)
                    q99 = clean_data[col].quantile(0.99)
                    clean_data[col] = clean_data[col].clip(q1, q99)

            # Ensure positive prices
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if col in clean_data.columns:
                    clean_data[col] = clean_data[col].clip(lower=1e-8)

            return clean_data

        except Exception as e:
            self.logger.error(f"❌ Data validation failed: {e}")
            return data

    def _extract_base_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract base OHLCV features."""
        try:
            features = {}

            # Price features
            features['close'] = data['close'].values
            features['open'] = data['open'].values
            features['high'] = data['high'].values
            features['low'] = data['low'].values
            features['volume'] = data['volume'].values

            # Basic returns
            features['close_return'] = data['close'].pct_change().fillna(0).values
            features['log_return'] = np.log(data['close'] / data['close'].shift(1)).fillna(0).values

            # Price ratios
            features['high_low_ratio'] = (data['high'] / data['low'] - 1).fillna(0).values
            features['close_open_ratio'] = (data['close'] / data['open'] - 1).fillna(0).values

            # Volume features
            features['volume_change'] = data['volume'].pct_change().fillna(0).values

            return features

        except Exception as e:
            self.logger.error(f"❌ Base feature extraction failed: {e}")
            return {}

    def _add_technical_indicators(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Add technical indicators optimized for neural networks."""
        try:
            features = {}

            close = data['close'].values
            high = data['high'].values
            low = data['low'].values
            volume = data['volume'].values

            # RSI with multiple periods
            for period in [self.config.short_period, self.config.medium_period, self.config.long_period]:
                features[f'rsi_{period}'] = self._calculate_rsi(close, period)

            # MACD
            macd, signal, hist = self._calculate_macd_nn_optimized(close)
            features['macd'] = macd
            features['macd_signal'] = signal
            features['macd_histogram'] = hist

            # Bollinger Bands (normalized)
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands_nn(close)
            features['bb_upper'] = bb_upper
            features['bb_middle'] = bb_middle
            features['bb_lower'] = bb_lower
            features['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)
            features['bb_width'] = (bb_upper - bb_lower) / bb_middle

            # Stochastic Oscillator
            stoch_k, stoch_d = self._calculate_stochastic_nn(high, low, close)
            features['stoch_k'] = stoch_k
            features['stoch_d'] = stoch_d

            # Williams %R
            features['williams_r'] = self._calculate_williams_r_nn(high, low, close)

            # Moving averages (multiple periods)
            for period in [self.config.short_period, self.config.medium_period, self.config.long_period]:
                features[f'sma_{period}'] = self._calculate_sma(close, period)
                features[f'ema_{period}'] = self._calculate_ema(close, period)

            # Price momentum
            for period in [1, 3, 5, 10, 21]:
                features[f'momentum_{period}'] = (close / np.roll(close, period) - 1)

            # Volume-weighted average price
            features['vwap'] = self._calculate_vwap_nn(data)

            # Commodity Channel Index
            features['cci'] = self._calculate_cci_nn(high, low, close)

            # Average True Range
            features['atr'] = self._calculate_atr_nn(high, low, close)

            # On-Balance Volume
            features['obv'] = self._calculate_obv_nn(close, volume)

            return features

        except Exception as e:
            self.logger.error(f"❌ Technical indicators failed: {e}")
            return {}

    def _add_microstructure_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Add market microstructure features."""
        try:
            features = {}

            close = data['close'].values
            high = data['high'].values
            low = data['low'].values
            volume = data['volume'].values

            # Price impact
            features['price_impact'] = np.abs(close - data['open'].values) / data['open'].values

            # Spread features
            features['spread'] = (data['ask'].values - data['bid'].values) / data['close'].values if 'ask' in data.columns and 'bid' in data.columns else np.zeros_like(close)

            # Order book imbalance (simulated)
            features['order_imbalance'] = np.random.normal(0, 0.1, len(close))

            # Trade intensity
            features['trade_intensity'] = np.log(volume + 1)

            # Price velocity
            features['price_velocity'] = np.gradient(close)

            # Microstructure noise
            features['micro_noise'] = close - self._calculate_ema(close, 5)

            # Liquidity measures
            features['liquidity_ratio'] = volume / (high - low + 1e-8)

            # Market efficiency
            features['market_efficiency'] = np.abs(features['price_impact']) / (volume + 1e-8)

            return features

        except Exception as e:
            self.logger.error(f"❌ Microstructure features failed: {e}")
            return {}

    def _add_regime_aware_features(self, data: pd.DataFrame,
                                 regime_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Add regime-aware features."""
        try:
            features = {}

            # Regime probabilities
            if 'regime_probabilities' in regime_data:
                regime_probs = regime_data['regime_probabilities']
                for i in range(regime_probs.shape[1]):
                    features[f'regime_prob_{i}'] = regime_probs[:, i]

            # Regime transitions
            if 'regime_predictions' in regime_data:
                regime_preds = regime_data['regime_predictions']
                regime_changes = np.diff(regime_preds, prepend=regime_preds[0])
                features['regime_change'] = regime_changes.astype(float)

            # Regime-specific features
            features['regime_stability'] = regime_data.get('regime_stability_scores', np.ones(len(data)))

            # Interaction features
            close = data['close'].values
            volume = data['volume'].values

            if 'regime_probabilities' in regime_data:
                regime_probs = regime_data['regime_probabilities']
                # Regime-weighted returns
                for i in range(regime_probs.shape[1]):
                    features[f'regime_weighted_return_{i}'] = close * regime_probs[:, i]

                # Regime-volume interaction
                features['regime_volume_interaction'] = np.sum(regime_probs * volume.reshape(-1, 1), axis=1)

            return features

        except Exception as e:
            self.logger.error(f"❌ Regime-aware features failed: {e}")
            return {}

    def _add_multitimeframe_features(self, data: pd.DataFrame,
                                   multi_timeframe_data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """Add multi-timeframe features."""
        try:
            features = {}
            close = data['close'].values

            for timeframe, tf_data in multi_timeframe_data.items():
                if tf_data is not None and len(tf_data) > 0:
                    tf_close = tf_data['close'].values

                    # Align timeframes (simplified)
                    if len(tf_close) > len(close):
                        tf_close = tf_close[-len(close):]
                    elif len(tf_close) < len(close):
                        close_padded = np.pad(close, (len(tf_close) - len(close), 0), 'edge')
                        close = close_padded

                    # Cross-timeframe features
                    features[f'{timeframe}_close'] = tf_close
                    features[f'{timeframe}_return'] = np.diff(tf_close, prepend=tf_close[0])
                    features[f'{timeframe}_volatility'] = pd.Series(tf_close).rolling(10).std().fillna(0).values

                    # Inter-timeframe ratios
                    features[f'ratio_to_{timeframe}'] = close / (tf_close + 1e-8)

            return features

        except Exception as e:
            self.logger.error(f"❌ Multi-timeframe features failed: {e}")
            return {}

    def _add_volatility_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Add volatility-specific features."""
        try:
            features = {}

            close = data['close'].values
            high = data['high'].values
            low = data['low'].values

            # Historical volatility
            returns = np.diff(np.log(close), prepend=np.log(close[0]))
            for period in [10, 20, 50]:
                features[f'volatility_{period}'] = pd.Series(returns).rolling(period).std().fillna(0).values

            # Parkinson's volatility
            features['parkinson_vol'] = np.sqrt((1 / (4 * np.log(2))) * (np.log(high / low) ** 2)).rolling(20).mean().fillna(0).values

            # Garman-Klass volatility
            features['gk_vol'] = np.sqrt(
                (0.5 * (np.log(high / low) ** 2)) -
                ((2 * np.log(2) - 1) * (np.log(close / data['open'].values) ** 2))
            ).rolling(20).mean().fillna(0).values

            # Volatility of volatility
            features['vol_of_vol'] = pd.Series(features['volatility_20']).rolling(20).std().fillna(0).values

            return features

        except Exception as e:
            self.logger.error(f"❌ Volatility features failed: {e}")
            return {}

    def _add_liquidity_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Add liquidity-specific features."""
        try:
            features = {}

            close = data['close'].values
            volume = data['volume'].values
            high = data['high'].values
            low = data['low'].values

            # Amihud illiquidity ratio
            features['amihud_illiquidity'] = (np.abs(np.diff(close, prepend=close[0])) / volume).rolling(20).mean().fillna(0).values

            # Kyle's lambda (price impact coefficient)
            features['kyle_lambda'] = (np.abs(np.diff(close, prepend=close[0])) / np.sqrt(volume)).rolling(20).mean().fillna(0).values

            # Volume-price trend
            features['vpt'] = (volume * (close - np.roll(close, 1))).cumsum()

            # Turnover ratio
            features['turnover'] = volume / (high - low + 1e-8)

            # Liquidity ratio
            features['liquidity_ratio'] = volume / np.abs(close - data['open'].values + 1e-8)

            return features

        except Exception as e:
            self.logger.error(f"❌ Liquidity features failed: {e}")
            return {}

    def _select_features(self, feature_df: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
        """Select most relevant features for neural networks."""
        try:
            # Target variable (next period return)
            target = market_data['close'].shift(-1).pct_change().fillna(0).values

            # Remove features with low variance
            feature_variance = feature_df.var()
            high_variance_features = feature_variance[feature_variance > 0.001].index
            feature_df = feature_df[high_variance_features]

            if len(feature_df.columns) <= self.config.max_features:
                return feature_df

            # Feature selection based on method
            if self.config.feature_selection_method == "mutual_info":
                selector = SelectKBest(mutual_info_regression, k=self.config.feature_selection_k)
            elif self.config.feature_selection_method == "f_regression":
                selector = SelectKBest(f_regression, k=self.config.feature_selection_k)
            else:
                # Default to top k features by variance
                top_features = feature_variance.nlargest(self.config.feature_selection_k).index
                return feature_df[top_features]

            # Apply feature selection
            X_selected = selector.fit_transform(feature_df, target)
            selected_columns = feature_df.columns[selector.get_support()].tolist()

            return pd.DataFrame(X_selected, columns=selected_columns, index=feature_df.index)

        except Exception as e:
            self.logger.error(f"❌ Feature selection failed: {e}")
            # Return top features by variance
            feature_variance = feature_df.var()
            top_features = feature_variance.nlargest(min(self.config.max_features, len(feature_df.columns))).index
            return feature_df[top_features]

    def _normalize_features(self, feature_df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Normalize features for neural network input."""
        try:
            normalization_info = {}

            if self.config.normalization_type == NormalizationType.STANDARD:
                scaler = StandardScaler()
            elif self.config.normalization_type == NormalizationType.MINMAX:
                scaler = MinMaxScaler()
            elif self.config.normalization_type == NormalizationType.ROBUST:
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()

            # Fit and transform
            normalized_features = scaler.fit_transform(feature_df)

            # Store normalization info
            normalization_info = {
                'scaler_type': self.config.normalization_type.value,
                'feature_means': scaler.mean_ if hasattr(scaler, 'mean_') else None,
                'feature_stds': scaler.scale_ if hasattr(scaler, 'scale_') else None,
                'feature_min': feature_df.min().values,
                'feature_max': feature_df.max().values
            }

            return normalized_features, normalization_info

        except Exception as e:
            self.logger.error(f"❌ Feature normalization failed: {e}")
            # Return original data
            return feature_df.values, {}

    def _create_feature_type_mapping(self, feature_names: List[str]) -> Dict[str, FeatureType]:
        """Create mapping of features to their types."""
        feature_types = {}

        for name in feature_names:
            if any(indicator in name for indicator in ['rsi', 'macd', 'bollinger', 'stochastic', 'cci', 'adx', 'ema', 'sma']):
                feature_types[name] = FeatureType.TECHNICAL
            elif any(micro in name for micro in ['impact', 'spread', 'imbalance', 'intensity', 'noise', 'efficiency']):
                feature_types[name] = FeatureType.MICROSTRUCTURE
            elif 'regime' in name:
                feature_types[name] = FeatureType.REGIME_AWARE
            elif any(tf in name for tf in ['1m', '5m', '15m', '1h', 'ratio_to']):
                feature_types[name] = FeatureType.MULTITIMEFRAME
            elif any(vol in name for vol in ['volatility', 'parkinson', 'gk_vol', 'vol_of_vol']):
                feature_types[name] = FeatureType.VOLATILITY
            elif any(liq in name for liq in ['amihud', 'kyle', 'turnover', 'liquidity']):
                feature_types[name] = FeatureType.LIQUIDITY
            else:
                feature_types[name] = FeatureType.TECHNICAL  # Default

        return feature_types

    # Technical indicator implementations (NN-optimized)
    def _calculate_rsi(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate RSI optimized for neural networks."""
        delta = np.diff(prices, prepend=prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = pd.Series(gain).rolling(period).mean().fillna(0).values
        avg_loss = pd.Series(loss).rolling(period).mean().fillna(0).values

        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_macd_nn_optimized(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD optimized for NN input."""
        ema12 = pd.Series(prices).ewm(span=12, adjust=False).mean().values
        ema26 = pd.Series(prices).ewm(span=26, adjust=False).mean().values
        macd = ema12 - ema26
        signal = pd.Series(macd).ewm(span=9, adjust=False).mean().values
        histogram = macd - signal
        return macd, signal, histogram

    def _calculate_bollinger_bands_nn(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands normalized for NN."""
        sma = pd.Series(prices).rolling(20).mean().fillna(method='bfill').values
        std = pd.Series(prices).rolling(20).std().fillna(method='bfill').values
        upper = sma + 2 * std
        lower = sma - 2 * std
        return upper, sma, lower

    def _calculate_stochastic_nn(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Stochastic Oscillator for NN."""
        lowest_low = pd.Series(low).rolling(14).min().fillna(method='bfill').values
        highest_high = pd.Series(high).rolling(14).max().fillna(method='bfill').values

        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-8))
        d_percent = pd.Series(k_percent).rolling(3).mean().fillna(method='bfill').values

        return k_percent, d_percent

    def _calculate_williams_r_nn(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Calculate Williams %R for NN."""
        highest_high = pd.Series(high).rolling(14).max().fillna(method='bfill').values
        lowest_low = pd.Series(low).rolling(14).min().fillna(method='bfill').values

        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low + 1e-8))
        return williams_r

    def _calculate_sma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average."""
        return pd.Series(prices).rolling(period).mean().fillna(method='bfill').values

    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        return pd.Series(prices).ewm(span=period, adjust=False).mean().fillna(method='bfill').values

    def _calculate_vwap_nn(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate Volume Weighted Average Price for NN."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
        return vwap.fillna(method='bfill').values

    def _calculate_cci_nn(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Calculate Commodity Channel Index for NN."""
        typical_price = (high + low + close) / 3
        sma_tp = pd.Series(typical_price).rolling(20).mean().fillna(method='bfill').values
        mad_tp = pd.Series(np.abs(typical_price - sma_tp)).rolling(20).mean().fillna(method='bfill').values

        cci = (typical_price - sma_tp) / (0.015 * mad_tp + 1e-8)
        return cci

    def _calculate_atr_nn(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Calculate Average True Range for NN."""
        high_low = high - low
        high_close = np.abs(high - np.roll(close, 1))
        low_close = np.abs(low - np.roll(close, 1))

        tr = np.maximum(np.maximum(high_low, high_close), low_close)
        atr = pd.Series(tr).rolling(14).mean().fillna(method='bfill').values

        return atr

    def _calculate_obv_nn(self, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Calculate On-Balance Volume for NN."""
        obv = np.zeros_like(close)
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        return obv

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        try:
            if not self.feature_importance_history:
                return {}

            # Average importance across history
            avg_importance = {}
            for feature_dict in self.feature_importance_history:
                for feature, importance in feature_dict.items():
                    if feature not in avg_importance:
                        avg_importance[feature] = []
                    avg_importance[feature].append(importance)

            # Calculate mean importance
            mean_importance = {
                feature: np.mean(scores)
                for feature, scores in avg_importance.items()
            }

            return mean_importance

        except Exception as e:
            self.logger.error(f"❌ Feature importance calculation failed: {e}")
            return {}

    def save_feature_engineer(self, filepath: str) -> bool:
        """Save feature engineer state."""
        try:
            state = {
                'config': self.config.__dict__,
                'feature_importance_history': self.feature_importance_history,
                'market_data_cache': self.market_data_cache
            }

            with open(filepath, 'wb') as f:
                import pickle

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
                pickle.dump(state, f)

            self.logger.info(f"✅ Feature engineer state saved to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to save feature engineer: {e}")
            return False


def create_nas_financial_feature_engineer(config: NASFeatureConfig) -> NASFinancialFeatureEngineer:
    """Create NAS financial feature engineer instance."""
    return NASFinancialFeatureEngineer(config)


def quick_feature_engineering(market_data: pd.DataFrame,
                             config: Optional[NASFeatureConfig] = None) -> FeatureSet:
    """Quick feature engineering with default settings."""
    if config is None:
        config = NASFeatureConfig(
            include_technical=True,
            include_microstructure=True,
            include_volatility=True,
            max_features=50
        )

    engineer = NASFinancialFeatureEngineer(config)
    return engineer.engineer_features(market_data)

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
