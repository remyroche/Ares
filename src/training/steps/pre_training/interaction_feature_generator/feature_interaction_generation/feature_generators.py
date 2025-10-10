"""
Feature Generators for Interactive Feature Generation

This module implements actual feature generation logic using vectorized operations
and optimized algorithms for creating meaningful features from market data.

Key Features:
- Vectorized operations using numpy
- Technical indicators generation
- Rolling statistics computation
- Interaction features creation
- Cross-timeframe features
- Memory-efficient implementations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from scipy import stats
from scipy.signal import find_peaks
import warnings

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance
)

logger = logging.getLogger(__name__)


@dataclass
class FeatureGenerationConfig:
    """Configuration for feature generation."""
    # Technical indicators
    enable_technical_indicators: bool = True
    enable_rolling_stats: bool = True
    enable_interaction_features: bool = True
    enable_cross_timeframe: bool = True
    
    # Rolling windows
    rolling_windows: List[int] = None
    
    # Technical indicator parameters
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    
    # Interaction features
    max_interactions: int = 50
    interaction_types: List[str] = None
    
    # Cross-timeframe
    cross_timeframe_periods: List[int] = None
    
    def __post_init__(self):
        if self.rolling_windows is None:
            self.rolling_windows = [5, 10, 20, 50, 100]
        if self.interaction_types is None:
            self.interaction_types = ['ratio', 'product', 'difference', 'sum']
        if self.cross_timeframe_periods is None:
            self.cross_timeframe_periods = [5, 15, 30, 60]


class TechnicalIndicatorsGenerator:
    """Generate technical indicators using vectorized operations."""
    
    def __init__(self, config: FeatureGenerationConfig):
        self.config = config
    
    def generate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate all technical indicators."""
        features = {}
        
        if 'close' in data.columns:
            close = data['close'].values
            features.update(self._generate_price_indicators(close))
        
        if 'volume' in data.columns:
            volume = data['volume'].values
            features.update(self._generate_volume_indicators(volume))
        
        if 'high' in data.columns and 'low' in data.columns and 'close' in data.columns:
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            features.update(self._generate_ohlc_indicators(high, low, close))
        
        return pd.DataFrame(features, index=data.index)
    
    def _generate_price_indicators(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate price-based technical indicators."""
        features = {}
        
        # RSI
        if len(close) > self.config.rsi_period:
            features['rsi'] = self._calculate_rsi(close, self.config.rsi_period)
        
        # MACD
        if len(close) > self.config.macd_slow:
            macd_line, macd_signal, macd_hist = self._calculate_macd(
                close, self.config.macd_fast, self.config.macd_slow, self.config.macd_signal
            )
            features['macd'] = macd_line
            features['macd_signal'] = macd_signal
            features['macd_histogram'] = macd_hist
        
        # Bollinger Bands
        if len(close) > self.config.bollinger_period:
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(
                close, self.config.bollinger_period, self.config.bollinger_std
            )
            features['bb_upper'] = bb_upper
            features['bb_middle'] = bb_middle
            features['bb_lower'] = bb_lower
            features['bb_width'] = (bb_upper - bb_lower) / bb_middle
            features['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)
        
        return features
    
    def _generate_volume_indicators(self, volume: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate volume-based indicators."""
        features = {}
        
        # Volume moving averages
        for window in [5, 10, 20]:
            if len(volume) > window:
                features[f'volume_ma_{window}'] = self._rolling_mean(volume, window)
                features[f'volume_ratio_{window}'] = volume / (features[f'volume_ma_{window}'] + 1e-8)
        
        # Volume rate of change
        for period in [1, 5, 10]:
            if len(volume) > period:
                features[f'volume_roc_{period}'] = self._rate_of_change(volume, period)
        
        return features
    
    def _generate_ohlc_indicators(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate OHLC-based indicators."""
        features = {}
        
        # Price ranges
        features['hl_range'] = high - low
        features['hc_range'] = np.abs(high - close)
        features['lc_range'] = np.abs(low - close)
        
        # True Range
        if len(close) > 1:
            prev_close = np.roll(close, 1)
            prev_close[0] = close[0]
            tr1 = high - low
            tr2 = np.abs(high - prev_close)
            tr3 = np.abs(low - prev_close)
            features['true_range'] = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Average True Range
        if len(close) > 14:
            features['atr_14'] = self._rolling_mean(features['true_range'], 14)
        
        return features
    
    def _calculate_rsi(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate RSI using vectorized operations."""
        if len(prices) < period + 1:
            return np.full(len(prices), np.nan)
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = self._rolling_mean(gains, period)
        avg_losses = self._rolling_mean(losses, period)
        
        rs = avg_gains / (avg_losses + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        return np.concatenate([[np.nan], rsi])
    
    def _calculate_macd(self, prices: np.ndarray, fast: int, slow: int, signal: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD using vectorized operations."""
        if len(prices) < slow:
            return np.full(len(prices), np.nan), np.full(len(prices), np.nan), np.full(len(prices), np.nan)
        
        ema_fast = self._exponential_moving_average(prices, fast)
        ema_slow = self._exponential_moving_average(prices, slow)
        
        macd_line = ema_fast - ema_slow
        macd_signal = self._exponential_moving_average(macd_line, signal)
        macd_histogram = macd_line - macd_signal
        
        return macd_line, macd_signal, macd_histogram
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int, std_mult: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands using vectorized operations."""
        if len(prices) < period:
            return np.full(len(prices), np.nan), np.full(len(prices), np.nan), np.full(len(prices), np.nan)
        
        sma = self._rolling_mean(prices, period)
        std = self._rolling_std(prices, period)
        
        upper = sma + (std * std_mult)
        lower = sma - (std * std_mult)
        
        return upper, sma, lower
    
    def _rolling_mean(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling mean using vectorized operations."""
        if len(data) < window:
            return np.full(len(data), np.nan)
        
        result = np.full(len(data), np.nan)
        for i in range(window - 1, len(data)):
            result[i] = np.mean(data[i - window + 1:i + 1])
        
        return result
    
    def _rolling_std(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling standard deviation using vectorized operations."""
        if len(data) < window:
            return np.full(len(data), np.nan)
        
        result = np.full(len(data), np.nan)
        for i in range(window - 1, len(data)):
            result[i] = np.std(data[i - window + 1:i + 1])
        
        return result
    
    def _exponential_moving_average(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate exponential moving average."""
        if len(data) < period:
            return np.full(len(data), np.nan)
        
        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        
        return ema
    
    def _rate_of_change(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate rate of change."""
        if len(data) < period + 1:
            return np.full(len(data), np.nan)
        
        result = np.full(len(data), np.nan)
        for i in range(period, len(data)):
            result[i] = (data[i] - data[i - period]) / (data[i - period] + 1e-8)
        
        return result


class RollingStatsGenerator:
    """Generate rolling statistics using vectorized operations."""
    
    def __init__(self, config: FeatureGenerationConfig):
        self.config = config
    
    def generate_rolling_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate rolling statistical features."""
        features = {}
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in ['target', 'timestamp']:
                continue
                
            col_data = data[col].values
            
            for window in self.config.rolling_windows:
                if len(col_data) > window:
                    # Rolling statistics
                    features[f'{col}_mean_{window}'] = self._rolling_mean(col_data, window)
                    features[f'{col}_std_{window}'] = self._rolling_std(col_data, window)
                    features[f'{col}_min_{window}'] = self._rolling_min(col_data, window)
                    features[f'{col}_max_{window}'] = self._rolling_max(col_data, window)
                    features[f'{col}_median_{window}'] = self._rolling_median(col_data, window)
                    features[f'{col}_skew_{window}'] = self._rolling_skew(col_data, window)
                    features[f'{col}_kurt_{window}'] = self._rolling_kurtosis(col_data, window)
                    
                    # Rolling ratios
                    features[f'{col}_current_vs_mean_{window}'] = col_data / (features[f'{col}_mean_{window}'] + 1e-8)
                    features[f'{col}_current_vs_max_{window}'] = col_data / (features[f'{col}_max_{window}'] + 1e-8)
                    features[f'{col}_current_vs_min_{window}'] = col_data / (features[f'{col}_min_{window}'] + 1e-8)
        
        return pd.DataFrame(features, index=data.index)
    
    def _rolling_mean(self, data: np.ndarray, window: int) -> np.ndarray:
        """Vectorized rolling mean calculation."""
        if len(data) < window:
            return np.full(len(data), np.nan)
        
        # Use convolution for efficiency
        kernel = np.ones(window) / window
        result = np.convolve(data, kernel, mode='valid')
        return np.concatenate([np.full(window - 1, np.nan), result])
    
    def _rolling_std(self, data: np.ndarray, window: int) -> np.ndarray:
        """Vectorized rolling standard deviation calculation."""
        if len(data) < window:
            return np.full(len(data), np.nan)
        
        result = np.full(len(data), np.nan)
        for i in range(window - 1, len(data)):
            result[i] = np.std(data[i - window + 1:i + 1])
        
        return result
    
    def _rolling_min(self, data: np.ndarray, window: int) -> np.ndarray:
        """Vectorized rolling minimum calculation."""
        if len(data) < window:
            return np.full(len(data), np.nan)
        
        result = np.full(len(data), np.nan)
        for i in range(window - 1, len(data)):
            result[i] = np.min(data[i - window + 1:i + 1])
        
        return result
    
    def _rolling_max(self, data: np.ndarray, window: int) -> np.ndarray:
        """Vectorized rolling maximum calculation."""
        if len(data) < window:
            return np.full(len(data), np.nan)
        
        result = np.full(len(data), np.nan)
        for i in range(window - 1, len(data)):
            result[i] = np.max(data[i - window + 1:i + 1])
        
        return result
    
    def _rolling_median(self, data: np.ndarray, window: int) -> np.ndarray:
        """Vectorized rolling median calculation."""
        if len(data) < window:
            return np.full(len(data), np.nan)
        
        result = np.full(len(data), np.nan)
        for i in range(window - 1, len(data)):
            result[i] = np.median(data[i - window + 1:i + 1])
        
        return result
    
    def _rolling_skew(self, data: np.ndarray, window: int) -> np.ndarray:
        """Vectorized rolling skewness calculation."""
        if len(data) < window:
            return np.full(len(data), np.nan)
        
        result = np.full(len(data), np.nan)
        for i in range(window - 1, len(data)):
            result[i] = stats.skew(data[i - window + 1:i + 1])
        
        return result
    
    def _rolling_kurtosis(self, data: np.ndarray, window: int) -> np.ndarray:
        """Vectorized rolling kurtosis calculation."""
        if len(data) < window:
            return np.full(len(data), np.nan)
        
        result = np.full(len(data), np.nan)
        for i in range(window - 1, len(data)):
            result[i] = stats.kurtosis(data[i - window + 1:i + 1])
        
        return result


class InteractionFeaturesGenerator:
    """Generate interaction features using vectorized operations."""
    
    def __init__(self, config: FeatureGenerationConfig):
        self.config = config
    
    def generate_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate interaction features between pairs of variables."""
        features = {}
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['target', 'timestamp']]
        
        if len(numeric_cols) < 2:
            return pd.DataFrame(index=data.index)
        
        # Generate interactions for top pairs to avoid explosion
        pairs = self._select_top_pairs(data[numeric_cols], min(self.config.max_interactions, len(numeric_cols) * 2))
        
        for col1, col2 in pairs:
            data1 = data[col1].values
            data2 = data[col2].values
            
            # Avoid division by zero
            data2_safe = data2 + 1e-8
            
            for interaction_type in self.config.interaction_types:
                if interaction_type == 'ratio':
                    features[f'{col1}_div_{col2}'] = data1 / data2_safe
                elif interaction_type == 'product':
                    features[f'{col1}_mul_{col2}'] = data1 * data2
                elif interaction_type == 'difference':
                    features[f'{col1}_sub_{col2}'] = data1 - data2
                elif interaction_type == 'sum':
                    features[f'{col1}_add_{col2}'] = data1 + data2
        
        return pd.DataFrame(features, index=data.index)
    
    def _select_top_pairs(self, data: pd.DataFrame, max_pairs: int) -> List[Tuple[str, str]]:
        """Select top pairs based on correlation for interaction generation."""
        if len(data.columns) < 2:
            return []
        
        # Calculate correlation matrix
        corr_matrix = data.corr().abs()
        
        # Get upper triangle pairs
        pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr = corr_matrix.iloc[i, j]
                if not np.isnan(corr):
                    pairs.append((col1, col2, corr))
        
        # Sort by correlation and take top pairs
        pairs.sort(key=lambda x: x[2], reverse=True)
        return [(pair[0], pair[1]) for pair in pairs[:max_pairs]]


class CrossTimeframeGenerator:
    """Generate cross-timeframe features."""
    
    def __init__(self, config: FeatureGenerationConfig):
        self.config = config
    
    def generate_cross_timeframe_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate cross-timeframe features."""
        features = {}
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['target', 'timestamp']]
        
        for col in numeric_cols:
            col_data = data[col].values
            
            for period in self.config.cross_timeframe_periods:
                if len(col_data) > period:
                    # Cross-timeframe aggregations
                    features[f'ctf_{period}_{col}_mean'] = self._rolling_mean(col_data, period)
                    features[f'ctf_{period}_{col}_std'] = self._rolling_std(col_data, period)
                    features[f'ctf_{period}_{col}_max'] = self._rolling_max(col_data, period)
                    features[f'ctf_{period}_{col}_min'] = self._rolling_min(col_data, period)
                    
                    # Cross-timeframe ratios
                    features[f'ctf_{period}_{col}_current_vs_mean'] = col_data / (features[f'ctf_{period}_{col}_mean'] + 1e-8)
                    features[f'ctf_{period}_{col}_current_vs_max'] = col_data / (features[f'ctf_{period}_{col}_max'] + 1e-8)
                    features[f'ctf_{period}_{col}_current_vs_min'] = col_data / (features[f'ctf_{period}_{col}_min'] + 1e-8)
        
        return pd.DataFrame(features, index=data.index)
    
    def _rolling_mean(self, data: np.ndarray, window: int) -> np.ndarray:
        """Vectorized rolling mean calculation."""
        if len(data) < window:
            return np.full(len(data), np.nan)
        
        kernel = np.ones(window) / window
        result = np.convolve(data, kernel, mode='valid')
        return np.concatenate([np.full(window - 1, np.nan), result])
    
    def _rolling_std(self, data: np.ndarray, window: int) -> np.ndarray:
        """Vectorized rolling standard deviation calculation."""
        if len(data) < window:
            return np.full(len(data), np.nan)
        
        result = np.full(len(data), np.nan)
        for i in range(window - 1, len(data)):
            result[i] = np.std(data[i - window + 1:i + 1])
        
        return result
    
    def _rolling_max(self, data: np.ndarray, window: int) -> np.ndarray:
        """Vectorized rolling maximum calculation."""
        if len(data) < window:
            return np.full(len(data), np.nan)
        
        result = np.full(len(data), np.nan)
        for i in range(window - 1, len(data)):
            result[i] = np.max(data[i - window + 1:i + 1])
        
        return result
    
    def _rolling_min(self, data: np.ndarray, window: int) -> np.ndarray:
        """Vectorized rolling minimum calculation."""
        if len(data) < window:
            return np.full(len(data), np.nan)
        
        result = np.full(len(data), np.nan)
        for i in range(window - 1, len(data)):
            result[i] = np.min(data[i - window + 1:i + 1])
        
        return result


class FeatureGenerator:
    """Main feature generator that coordinates all feature generation."""
    
    def __init__(self, config: Optional[FeatureGenerationConfig] = None):
        self.config = config or FeatureGenerationConfig()
        
        # Initialize generators
        self.technical_generator = TechnicalIndicatorsGenerator(self.config)
        self.rolling_generator = RollingStatsGenerator(self.config)
        self.interaction_generator = InteractionFeaturesGenerator(self.config)
        self.cross_timeframe_generator = CrossTimeframeGenerator(self.config)
        
        tprint_success("🚀 Feature generator initialized")
        tprint_info(f"📊 Rolling windows: {self.config.rolling_windows}")
        tprint_info(f"📊 Max interactions: {self.config.max_interactions}")
        tprint_info(f"📊 Cross-timeframe periods: {self.config.cross_timeframe_periods}")
    
    def generate_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate all types of features."""
        tprint_info("🔧 Starting comprehensive feature generation...")
        
        all_features = []
        
        # Technical indicators
        if self.config.enable_technical_indicators:
            tprint_debug("📈 Generating technical indicators...")
            tech_features = self.technical_generator.generate_all_indicators(data)
            if not tech_features.empty:
                all_features.append(tech_features)
                tprint_info(f"✅ Generated {len(tech_features.columns)} technical indicators")
        
        # Rolling statistics
        if self.config.enable_rolling_stats:
            tprint_debug("📊 Generating rolling statistics...")
            rolling_features = self.rolling_generator.generate_rolling_features(data)
            if not rolling_features.empty:
                all_features.append(rolling_features)
                tprint_info(f"✅ Generated {len(rolling_features.columns)} rolling features")
        
        # Interaction features
        if self.config.enable_interaction_features:
            tprint_debug("🔗 Generating interaction features...")
            interaction_features = self.interaction_generator.generate_interaction_features(data)
            if not interaction_features.empty:
                all_features.append(interaction_features)
                tprint_info(f"✅ Generated {len(interaction_features.columns)} interaction features")
        
        # Cross-timeframe features
        if self.config.enable_cross_timeframe:
            tprint_debug("⏰ Generating cross-timeframe features...")
            ctf_features = self.cross_timeframe_generator.generate_cross_timeframe_features(data)
            if not ctf_features.empty:
                all_features.append(ctf_features)
                tprint_info(f"✅ Generated {len(ctf_features.columns)} cross-timeframe features")
        
        # Combine all features
        if all_features:
            combined_features = pd.concat(all_features, axis=1)
            # Remove any columns that are all NaN
            combined_features = combined_features.dropna(axis=1, how='all')
            tprint_success(f"✅ Generated {len(combined_features.columns)} total features")
            return combined_features
        else:
            tprint_warning("⚠️ No features generated")
            return pd.DataFrame(index=data.index)
    
    def generate_base_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate only base features (technical indicators + rolling stats)."""
        tprint_info("🔧 Generating base features...")
        
        all_features = []
        
        # Technical indicators
        if self.config.enable_technical_indicators:
            tech_features = self.technical_generator.generate_all_indicators(data)
            if not tech_features.empty:
                all_features.append(tech_features)
        
        # Rolling statistics
        if self.config.enable_rolling_stats:
            rolling_features = self.rolling_generator.generate_rolling_features(data)
            if not rolling_features.empty:
                all_features.append(rolling_features)
        
        if all_features:
            combined_features = pd.concat(all_features, axis=1)
            combined_features = combined_features.dropna(axis=1, how='all')
            tprint_success(f"✅ Generated {len(combined_features.columns)} base features")
            return combined_features
        else:
            return pd.DataFrame(index=data.index)
    
    def generate_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate only interaction features."""
        tprint_info("🔧 Generating interaction features...")
        
        features = self.interaction_generator.generate_interaction_features(data)
        if not features.empty:
            features = features.dropna(axis=1, how='all')
            tprint_success(f"✅ Generated {len(features.columns)} interaction features")
        
        return features
    
    def generate_cross_timeframe_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate only cross-timeframe features."""
        tprint_info("🔧 Generating cross-timeframe features...")
        
        features = self.cross_timeframe_generator.generate_cross_timeframe_features(data)
        if not features.empty:
            features = features.dropna(axis=1, how='all')
            tprint_success(f"✅ Generated {len(features.columns)} cross-timeframe features")
        
        return features