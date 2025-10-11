"""
Feature Generation Utilities

This module provides improved feature generation logic with better validation,
error handling, and meaningful feature creation.

Key Features:
- Meaningful feature generation with proper validation
- Technical indicators with robust calculations
- Rolling statistics with edge case handling
- Interaction features with validation
- Cross-timeframe features with proper alignment
- Memory-efficient implementations
- Enhanced validation using generalized validation utilities
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

# Import enhanced validation utilities
from ...utils.validation_utils import (

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
    PreTrainingValidator, ValidationConfig, ValidationContext,
    validate_feature_generation_inputs, ValidationResult
)

# Note: VectorBT optimizations are available in vectorbt_optimized_features.py
# but are not integrated into the core feature generation logic to maintain
# separation of concerns and avoid modifying existing feature generation.

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
    
    # Validation
    min_valid_ratio: float = 0.8  # Minimum ratio of valid values
    max_constant_ratio: float = 0.1  # Maximum ratio of constant features
    
    def __post_init__(self):
        if self.rolling_windows is None:
            self.rolling_windows = [5, 10, 20, 50, 100]
        if self.interaction_types is None:
            self.interaction_types = ['ratio', 'product', 'difference', 'sum']
        if self.cross_timeframe_periods is None:
            self.cross_timeframe_periods = [5, 15, 30, 60]


class FeatureValidator:
    """Validates generated features for quality and usefulness using enhanced validation utilities."""
    
    def __init__(self, config: FeatureGenerationConfig):
        self.config = config
        # Initialize the enhanced validator
        self.validator = PreTrainingValidator(
            ValidationConfig(
                context=ValidationContext.FEATURE_GENERATION,
                enable_logging=True
            )
        )
    
    def validate_features(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate features for quality and usefulness using enhanced validation utilities.
        
        Args:
            features: DataFrame containing features to validate
            
        Returns:
            Dictionary with validation results
        """
        if features.empty:
            return {
                'passed': False,
                'quality_score': 0.0,
                'issues': ['No features provided'],
                'recommendations': ['Generate more features']
            }
        
        # Use enhanced validation
        validation_result = self.validator.validate_features(features, features.columns.tolist())
        
        # Convert ValidationResult to the expected format
        issues = []
        recommendations = []
        
        if not validation_result.is_valid:
            issues.append(validation_result.error_message)
            if validation_result.warnings:
                issues.extend(validation_result.warnings)
        
        # Calculate quality score based on validation result
        quality_score = 1.0 if validation_result.is_valid else 0.0
        if validation_result.warnings:
            quality_score -= len(validation_result.warnings) * 0.1
        
        return {
            'passed': validation_result.is_valid,
            'quality_score': max(0.0, quality_score),
            'issues': issues,
            'recommendations': recommendations,
            'validation_details': validation_result.details
        }
    
    
    
    
    


class ImprovedFeatureGenerator:
    """
    Improved feature generator with better validation and meaningful features.
    
    This class generates meaningful features from market data with proper
    validation and error handling.
    """
    
    def __init__(self, config: FeatureGenerationConfig):
        self.config = config
        self.validator = FeatureValidator(config)
        
        tprint_debug("🔧 ImprovedFeatureGenerator initialized")
    
    def generate_meaningful_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate meaningful features with proper validation.
        
        Args:
            data: Input market data
            
        Returns:
            DataFrame with generated features
            
        Raises:
            ValueError: If input data is invalid or empty
            RuntimeError: If feature generation fails
        """
        tprint_info("🏗️ Generating meaningful features...")
        
        # Fast-fail: Validate input data
        if data.empty:
            raise ValueError("Input data is empty - cannot generate features")
        
        if not self._validate_input_data(data):
            raise ValueError("Invalid input data - missing required columns or insufficient data")
        
        features = {}
        
        # Generate technical indicators
        if self.config.enable_technical_indicators:
            tprint_debug("📊 Generating technical indicators...")
            tech_features = self._generate_technical_indicators(data)
            if not tech_features:
                raise RuntimeError("Failed to generate technical indicators")
            features.update(tech_features)
            tprint_info(f"✅ Generated {len(tech_features)} technical indicators")
        
        # Generate rolling statistics
        if self.config.enable_rolling_stats:
            tprint_debug("📈 Generating rolling statistics...")
            rolling_features = self._generate_rolling_statistics(data)
            if not rolling_features:
                raise RuntimeError("Failed to generate rolling statistics")
            features.update(rolling_features)
            tprint_info(f"✅ Generated {len(rolling_features)} rolling statistics")
        
        # Fast-fail: Must have generated features
        if not features:
            raise RuntimeError("No features generated - check configuration and input data")
        
        # Create DataFrame
        features_df = pd.DataFrame(features, index=data.index)
        
        # Validate generated features
        validation_result = self.validator.validate_features(features_df)
        
        if not validation_result['passed']:
            raise RuntimeError(f"Feature validation failed: {validation_result['issues']}")
        
        tprint_success(f"✅ Generated {len(features_df.columns)} validated features")
        tprint_info(f"📊 Quality score: {validation_result['quality_score']:.3f}")
        
        return features_df
    
    def generate_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate interaction features with validation.
        
        Args:
            data: Input data for interaction generation
            
        Returns:
            DataFrame with interaction features
            
        Raises:
            ValueError: If input data is invalid
            RuntimeError: If interaction generation fails
        """
        tprint_info("🔗 Generating interaction features...")
        
        # Fast-fail: Validate input data
        if data.empty:
            raise ValueError("Input data is empty - cannot generate interactions")
        
        # Get numeric columns for interactions
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            raise ValueError(f"Not enough numeric columns for interactions: {len(numeric_cols)} < 2")
        
        interaction_features = {}
        
        # OPTIMIZATION: Generate interactions more efficiently using vectorized operations
        interaction_count = 0
        max_interactions = min(self.config.max_interactions, len(numeric_cols) * (len(numeric_cols) - 1) // 2)
        
        for i, col1 in enumerate(numeric_cols):
            if interaction_count >= max_interactions:
                break
                
            for col2 in numeric_cols[i+1:]:
                if interaction_count >= max_interactions:
                    break
                
                # Generate different types of interactions
                for interaction_type in self.config.interaction_types:
                    if interaction_count >= max_interactions:
                        break
                    
                    feature_name = f"{col1}_{interaction_type}_{col2}"
                    
                    try:
                        if interaction_type == 'ratio':
                            interaction_features[feature_name] = self._safe_divide(
                                data[col1], data[col2]
                            )
                        elif interaction_type == 'product':
                            interaction_features[feature_name] = data[col1] * data[col2]
                        elif interaction_type == 'difference':
                            interaction_features[feature_name] = data[col1] - data[col2]
                        elif interaction_type == 'sum':
                            interaction_features[feature_name] = data[col1] + data[col2]
                        
                        interaction_count += 1
                        
                    except Exception as e:
                        tprint_debug(f"⚠️ Failed to generate {feature_name}: {e}")
                        continue
        
        # Fast-fail: Must have generated interactions
        if not interaction_features:
            raise RuntimeError("No interaction features generated - check configuration and input data")
        
        # Create DataFrame
        interaction_df = pd.DataFrame(interaction_features, index=data.index)
        
        # Validate interactions
        validation_result = self.validator.validate_features(interaction_df)
        
        if not validation_result['passed']:
            raise RuntimeError(f"Interaction validation failed: {validation_result['issues']}")
        
        tprint_success(f"✅ Generated {len(interaction_df.columns)} validated interactions")
        
        return interaction_df
    
    def generate_cross_timeframe_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate cross-timeframe features with proper alignment.
        
        Args:
            data: Input data for cross-timeframe generation
            
        Returns:
            DataFrame with cross-timeframe features
            
        Raises:
            ValueError: If input data is invalid
            RuntimeError: If cross-timeframe generation fails
        """
        tprint_info("⏰ Generating cross-timeframe features...")
        
        # Fast-fail: Validate input data
        if data.empty:
            raise ValueError("Input data is empty - cannot generate cross-timeframe features")
        
        # Get numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            raise ValueError("No numeric columns found for cross-timeframe generation")
        
        cross_tf_features = {}
        
        # Generate cross-timeframe features
        for period in self.config.cross_timeframe_periods:
            for col in numeric_cols:
                # Generate different aggregations
                cross_tf_features[f'ctf_{period}m_{col}_mean'] = data[col].rolling(period).mean()
                cross_tf_features[f'ctf_{period}m_{col}_std'] = data[col].rolling(period).std()
                cross_tf_features[f'ctf_{period}m_{col}_max'] = data[col].rolling(period).max()
                cross_tf_features[f'ctf_{period}m_{col}_min'] = data[col].rolling(period).min()
                cross_tf_features[f'ctf_{period}m_{col}_median'] = data[col].rolling(period).median()
        
        # Fast-fail: Must have generated features
        if not cross_tf_features:
            raise RuntimeError("No cross-timeframe features generated - check configuration and input data")
        
        # Create DataFrame
        cross_tf_df = pd.DataFrame(cross_tf_features, index=data.index)
        
        # Validate cross-timeframe features
        validation_result = self.validator.validate_features(cross_tf_df)
        
        if not validation_result['passed']:
            raise RuntimeError(f"Cross-timeframe validation failed: {validation_result['issues']}")
        
        tprint_success(f"✅ Generated {len(cross_tf_df.columns)} validated cross-timeframe features")
        
        return cross_tf_df
    
    def _validate_input_data(self, data: pd.DataFrame) -> bool:
        """Validate input data for feature generation using enhanced validation utilities."""
        # Use the enhanced validation utilities
        validation_result = validate_feature_generation_inputs(
            data,
            feature_columns=[],  # No specific features to validate yet
            required_columns=['open', 'high', 'low', 'close', 'volume']
        )
        
        if not validation_result.is_valid:
            tprint_error(f"❌ Input data validation failed: {validation_result.error_message}")
            return False
        
        # Additional checks for feature generation
        if len(data) < 10:
            tprint_warning("⚠️ Insufficient data for feature generation")
            return False
        
        tprint_success("✅ Input data validation passed")
        return True
    
    def _generate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate technical indicators with robust calculations."""
        features = {}
        
        try:
            if 'close' in data.columns:
                close = data['close'].values
                features.update(self._generate_price_indicators(close))
            
            if 'volume' in data.columns:
                volume = data['volume'].values
                features.update(self._generate_volume_indicators(volume))
            
            if all(col in data.columns for col in ['high', 'low', 'close']):
                high = data['high'].values
                low = data['low'].values
                close = data['close'].values
                features.update(self._generate_ohlc_indicators(high, low, close))
                
        except Exception as e:
            tprint_error(f"❌ Technical indicator generation failed: {e}")
        
        return features
    
    def _generate_price_indicators(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate price-based technical indicators."""
        features = {}
        
        # Fast-fail: Validate input data
        if len(close) == 0:
            raise RuntimeError("CRITICAL: Cannot generate price indicators on empty data")
        
        if np.isnan(close).all():
            raise RuntimeError("CRITICAL: Cannot generate price indicators on all-NaN data")
        
        try:
            # RSI
            if len(close) > self.config.rsi_period:
                rsi = self._calculate_rsi(close, self.config.rsi_period)
                if not np.isnan(rsi).all():
                    features['rsi'] = rsi
            
            # MACD
            if len(close) > max(self.config.macd_fast, self.config.macd_slow):
                macd_line, signal_line, histogram = self._calculate_macd(
                    close, self.config.macd_fast, self.config.macd_slow, self.config.macd_signal
                )
                if not np.isnan(macd_line).all():
                    features['macd'] = macd_line
                    features['macd_signal'] = signal_line
                    features['macd_histogram'] = histogram
            
            # Bollinger Bands
            if len(close) > self.config.bollinger_period:
                bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(
                    close, self.config.bollinger_period, self.config.bollinger_std
                )
                if not np.isnan(bb_upper).all():
                    features['bb_upper'] = bb_upper
                    features['bb_middle'] = bb_middle
                    features['bb_lower'] = bb_lower
                    features['bb_width'] = (bb_upper - bb_lower) / (bb_middle + 1e-8)
                    features['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-8)
            
            # Price momentum - always generate these
            price_change = np.concatenate([[0], np.diff(close)])
            price_change_pct = np.concatenate([[0], np.diff(close) / (close[:-1] + 1e-8)])
            features['price_change'] = price_change
            features['price_change_pct'] = price_change_pct
            
        except Exception as e:
            raise RuntimeError(f"CRITICAL: Price indicator calculation failed: {e}")
        
        # Fast-fail: Must have generated features
        if not features:
            raise RuntimeError("CRITICAL: No price indicators generated - check data and configuration")
        
        return features
    
    def _generate_volume_indicators(self, volume: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate volume-based technical indicators."""
        features = {}
        
        # Fast-fail: Validate input data
        if len(volume) == 0:
            raise RuntimeError("CRITICAL: Cannot generate volume indicators on empty data")
        
        if np.isnan(volume).all():
            raise RuntimeError("CRITICAL: Cannot generate volume indicators on all-NaN data")
        
        try:
            # Volume momentum - always generate these
            volume_change = np.concatenate([[0], np.diff(volume)])
            volume_change_pct = np.concatenate([[0], np.diff(volume) / (volume[:-1] + 1e-8)])
            features['volume_change'] = volume_change
            features['volume_change_pct'] = volume_change_pct
            
            # Volume moving averages
            for window in [5, 10, 20]:
                if len(volume) > window:
                    volume_ma = self._rolling_mean(volume, window)
                    if not np.isnan(volume_ma).all():
                        features[f'volume_ma_{window}'] = volume_ma
                        features[f'volume_ratio_{window}'] = volume / (volume_ma + 1e-8)
            
        except Exception as e:
            raise RuntimeError(f"CRITICAL: Volume indicator calculation failed: {e}")
        
        # Fast-fail: Must have generated features
        if not features:
            raise RuntimeError("CRITICAL: No volume indicators generated - check data and configuration")
        
        return features
    
    def _generate_ohlc_indicators(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate OHLC-based technical indicators."""
        features = {}
        
        # Fast-fail: Validate input data
        if len(high) == 0 or len(low) == 0 or len(close) == 0:
            raise RuntimeError("CRITICAL: Cannot generate OHLC indicators on empty data")
        
        if len(high) != len(low) or len(high) != len(close):
            raise RuntimeError("CRITICAL: OHLC arrays must have the same length")
        
        if np.isnan(high).all() or np.isnan(low).all() or np.isnan(close).all():
            raise RuntimeError("CRITICAL: Cannot generate OHLC indicators on all-NaN data")
        
        try:
            # True Range
            tr = self._calculate_true_range(high, low, close)
            if not np.isnan(tr).all():
                features['true_range'] = tr
                
                # Average True Range
                for window in [14, 21]:
                    if len(tr) > window:
                        atr = self._rolling_mean(tr, window)
                        if not np.isnan(atr).all():
                            features[f'atr_{window}'] = atr
            
            # Price position within range - always generate these
            price_position = (close - low) / (high - low + 1e-8)
            range_size = high - low
            range_size_pct = (high - low) / (close + 1e-8)
            
            features['price_position'] = price_position
            features['range_size'] = range_size
            features['range_size_pct'] = range_size_pct
            
        except Exception as e:
            raise RuntimeError(f"CRITICAL: OHLC indicator calculation failed: {e}")
        
        # Fast-fail: Must have generated features
        if not features:
            raise RuntimeError("CRITICAL: No OHLC indicators generated - check data and configuration")
        
        return features
    
    def _generate_rolling_statistics(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate rolling statistics for numeric columns using vectorized operations."""
        features = {}
        
        # Fast-fail: Validate input data
        if data.empty:
            raise RuntimeError("CRITICAL: Cannot generate rolling statistics on empty data")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise RuntimeError("CRITICAL: No numeric columns found for rolling statistics")
        
        try:
            # OPTIMIZATION: Use vectorized operations for better performance
            for col in numeric_cols:
                series = data[col]
                
                # Fast-fail: Check for sufficient data
                if len(series) < 5:
                    tprint_warning(f"⚠️ Skipping {col}: insufficient data ({len(series)} < 5)")
                    continue
                
                for window in self.config.rolling_windows:
                    if len(series) > window:
                        # Use pandas rolling operations for better performance
                        rolling = series.rolling(window=window, min_periods=1)
                        
                        # Basic statistics (vectorized)
                        features[f'{col}_ma_{window}'] = rolling.mean().values
                        features[f'{col}_std_{window}'] = rolling.std().values
                        features[f'{col}_min_{window}'] = rolling.min().values
                        features[f'{col}_max_{window}'] = rolling.max().values
                        
                        # Advanced statistics (only if window is large enough)
                        if window >= 10:
                            features[f'{col}_skew_{window}'] = rolling.skew().values
                            features[f'{col}_kurt_{window}'] = rolling.kurt().values
                        
                        # Percentiles (only if window is large enough)
                        if window >= 20:
                            features[f'{col}_p25_{window}'] = rolling.quantile(0.25).values
                            features[f'{col}_p75_{window}'] = rolling.quantile(0.75).values
                        
        except Exception as e:
            raise RuntimeError(f"CRITICAL: Rolling statistics calculation failed: {e}")
        
        # Fast-fail: Must have generated features
        if not features:
            raise RuntimeError("CRITICAL: No rolling statistics generated - check data and configuration")
        
        return features
    
    # Helper methods for calculations
    def _calculate_rsi(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate RSI with proper handling of edge cases."""
        if len(prices) < period + 1:
            return np.full(len(prices), np.nan)
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = self._rolling_mean(gains, period)
        avg_losses = self._rolling_mean(losses, period)
        
        rs = avg_gains / (avg_losses + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        return np.concatenate([[np.nan] * period, rsi])
    
    def _calculate_macd(self, prices: np.ndarray, fast: int, slow: int, signal: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD with proper handling of edge cases."""
        if len(prices) < slow:
            return np.full(len(prices), np.nan), np.full(len(prices), np.nan), np.full(len(prices), np.nan)
        
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self._calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int, std_mult: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands with proper handling of edge cases."""
        if len(prices) < period:
            return np.full(len(prices), np.nan), np.full(len(prices), np.nan), np.full(len(prices), np.nan)
        
        middle = self._rolling_mean(prices, period)
        std = self._rolling_std(prices, period)
        
        upper = middle + (std * std_mult)
        lower = middle - (std * std_mult)
        
        return upper, middle, lower
    
    def _calculate_true_range(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Calculate True Range with proper handling of edge cases."""
        if len(high) < 2:
            return np.full(len(high), np.nan)
        
        prev_close = np.concatenate([[close[0]], close[:-1]])
        
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = np.nan  # First value is NaN
        
        return tr
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return np.full(len(prices), np.nan)
        
        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def _rolling_mean(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling mean with proper handling of edge cases."""
        if len(data) < window:
            return np.full(len(data), np.nan)
        
        # Use pandas rolling for efficiency and correct length
        series = pd.Series(data)
        rolling = series.rolling(window=window, min_periods=1)
        return rolling.mean().values
    
    def _rolling_std(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling standard deviation."""
        if len(data) < window:
            return np.full(len(data), np.nan)
        
        # Use pandas rolling for efficiency and correct length
        series = pd.Series(data)
        rolling = series.rolling(window=window, min_periods=1)
        return rolling.std().values
    
    def _rolling_min(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling minimum."""
        if len(data) < window:
            return np.full(len(data), np.nan)
        
        # Use pandas rolling for efficiency and correct length
        series = pd.Series(data)
        rolling = series.rolling(window=window, min_periods=1)
        return rolling.min().values
    
    def _rolling_max(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling maximum."""
        if len(data) < window:
            return np.full(len(data), np.nan)
        
        # Use pandas rolling for efficiency and correct length
        series = pd.Series(data)
        rolling = series.rolling(window=window, min_periods=1)
        return rolling.max().values
    
    def _rolling_skew(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling skewness."""
        if len(data) < window:
            return np.full(len(data), np.nan)
        
        result = np.full(len(data), np.nan)
        for i in range(window - 1, len(data)):
            result[i] = stats.skew(data[i - window + 1:i + 1])
        
        return result
    
    def _rolling_kurtosis(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling kurtosis."""
        if len(data) < window:
            return np.full(len(data), np.nan)
        
        result = np.full(len(data), np.nan)
        for i in range(window - 1, len(data)):
            result[i] = stats.kurtosis(data[i - window + 1:i + 1])
        
        return result
    
    def _rolling_percentile(self, data: np.ndarray, window: int, percentile: float) -> np.ndarray:
        """Calculate rolling percentile."""
        if len(data) < window:
            return np.full(len(data), np.nan)
        
        result = np.full(len(data), np.nan)
        for i in range(window - 1, len(data)):
            result[i] = np.percentile(data[i - window + 1:i + 1], percentile)
        
        return result
    
    def _safe_divide(self, numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        """Safely divide arrays, handling division by zero."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
            result = np.where(np.isfinite(result), result, 0)
        return result


# Convenience function
def create_improved_feature_generator(config: Optional[FeatureGenerationConfig] = None) -> ImprovedFeatureGenerator:
    """Create an improved feature generator with the given configuration."""
    if config is None:
        config = FeatureGenerationConfig()
    
    return ImprovedFeatureGenerator(config)


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'open': np.random.randn(n_samples).cumsum() + 100,
        'high': np.random.randn(n_samples).cumsum() + 102,
        'low': np.random.randn(n_samples).cumsum() + 98,
        'close': np.random.randn(n_samples).cumsum() + 100,
        'volume': np.random.lognormal(10, 1, n_samples),
    })
    
    # Create feature generator
    config = FeatureGenerationConfig(
        enable_technical_indicators=True,
        enable_rolling_stats=True,
        enable_interaction_features=True,
        enable_cross_timeframe=True
    )
    
    generator = create_improved_feature_generator(config)
    
    # Generate features
    features = generator.generate_meaningful_features(data)
    print(f"Generated {len(features.columns)} features")
    print(f"Features shape: {features.shape}")
    
    # Generate interactions
    interactions = generator.generate_interaction_features(data)
    print(f"Generated {len(interactions.columns)} interactions")
    
    # Generate cross-timeframe features
    cross_tf = generator.generate_cross_timeframe_features(data)
    print(f"Generated {len(cross_tf.columns)} cross-timeframe features")

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
