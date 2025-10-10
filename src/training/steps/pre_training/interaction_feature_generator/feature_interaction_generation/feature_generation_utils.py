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
    """Validates generated features for quality and usefulness."""
    
    def __init__(self, config: FeatureGenerationConfig):
        self.config = config
    
    def validate_features(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate features for quality and usefulness.
        
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
        
        issues = []
        recommendations = []
        
        # Check for finite values
        finite_ratio = self._check_finite_values(features)
        if finite_ratio < self.config.min_valid_ratio:
            issues.append(f"Low finite value ratio: {finite_ratio:.2%}")
            recommendations.append("Check for NaN or infinite values in input data")
        
        # Check for constant features
        constant_ratio = self._check_constant_features(features)
        if constant_ratio > self.config.max_constant_ratio:
            issues.append(f"High constant feature ratio: {constant_ratio:.2%}")
            recommendations.append("Remove or transform constant features")
        
        # Check for correlation with target (if available)
        correlation_score = self._check_correlation(features)
        
        # Check for variance
        variance_score = self._check_variance(features)
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(
            finite_ratio, constant_ratio, correlation_score, variance_score
        )
        
        return {
            'passed': quality_score > 0.7,
            'quality_score': quality_score,
            'finite_ratio': finite_ratio,
            'constant_ratio': constant_ratio,
            'correlation_score': correlation_score,
            'variance_score': variance_score,
            'issues': issues,
            'recommendations': recommendations
        }
    
    def _check_finite_values(self, features: pd.DataFrame) -> float:
        """Check ratio of finite values."""
        numeric_features = features.select_dtypes(include=[np.number])
        if numeric_features.empty:
            return 0.0
        
        finite_ratios = numeric_features.apply(
            lambda x: np.isfinite(x).sum() / len(x)
        )
        return finite_ratios.mean()
    
    def _check_constant_features(self, features: pd.DataFrame) -> float:
        """Check ratio of constant features."""
        numeric_features = features.select_dtypes(include=[np.number])
        if numeric_features.empty:
            return 1.0
        
        constant_features = (numeric_features.nunique() <= 1).sum()
        return constant_features / len(numeric_features.columns)
    
    def _check_correlation(self, features: pd.DataFrame) -> float:
        """Check correlation between features (avoid multicollinearity)."""
        numeric_features = features.select_dtypes(include=[np.number])
        if len(numeric_features.columns) < 2:
            return 1.0
        
        # Calculate correlation matrix
        corr_matrix = numeric_features.corr().abs()
        
        # Remove diagonal and get upper triangle
        upper_triangle = corr_matrix.where(
            np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        )
        
        # Check for high correlations
        high_corr_ratio = (upper_triangle > 0.95).sum().sum() / upper_triangle.count().sum()
        
        # Return score (lower high correlation ratio is better)
        return max(0.0, 1.0 - high_corr_ratio)
    
    def _check_variance(self, features: pd.DataFrame) -> float:
        """Check variance of features."""
        numeric_features = features.select_dtypes(include=[np.number])
        if numeric_features.empty:
            return 0.0
        
        # Calculate coefficient of variation (std/mean)
        cv_scores = []
        for col in numeric_features.columns:
            series = numeric_features[col].dropna()
            if len(series) > 0 and series.mean() != 0:
                cv = series.std() / abs(series.mean())
                cv_scores.append(cv)
        
        if not cv_scores:
            return 0.0
        
        # Higher coefficient of variation is generally better
        avg_cv = np.mean(cv_scores)
        return min(1.0, avg_cv / 0.1)  # Normalize to 0-1 range
    
    def _calculate_quality_score(
        self, 
        finite_ratio: float, 
        constant_ratio: float, 
        correlation_score: float, 
        variance_score: float
    ) -> float:
        """Calculate overall quality score."""
        # Weighted combination of different quality metrics
        weights = {
            'finite_ratio': 0.3,
            'constant_ratio': 0.2,
            'correlation_score': 0.25,
            'variance_score': 0.25
        }
        
        # Convert constant_ratio to a score (lower is better)
        constant_score = 1.0 - constant_ratio
        
        quality_score = (
            weights['finite_ratio'] * finite_ratio +
            weights['constant_ratio'] * constant_score +
            weights['correlation_score'] * correlation_score +
            weights['variance_score'] * variance_score
        )
        
        return quality_score


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
        """
        tprint_info("🏗️ Generating meaningful features...")
        
        if data.empty:
            tprint_warning("⚠️ Empty input data provided")
            return pd.DataFrame()
        
        # Validate input data
        if not self._validate_input_data(data):
            tprint_error("❌ Invalid input data")
            return pd.DataFrame()
        
        features = {}
        
        try:
            # Generate technical indicators
            if self.config.enable_technical_indicators:
                tprint_debug("📊 Generating technical indicators...")
                tech_features = self._generate_technical_indicators(data)
                features.update(tech_features)
                tprint_info(f"✅ Generated {len(tech_features)} technical indicators")
            
            # Generate rolling statistics
            if self.config.enable_rolling_stats:
                tprint_debug("📈 Generating rolling statistics...")
                rolling_features = self._generate_rolling_statistics(data)
                features.update(rolling_features)
                tprint_info(f"✅ Generated {len(rolling_features)} rolling statistics")
            
            # Create DataFrame
            if features:
                features_df = pd.DataFrame(features, index=data.index)
                
                # Validate generated features
                validation_result = self.validator.validate_features(features_df)
                
                if validation_result['passed']:
                    tprint_success(f"✅ Generated {len(features_df.columns)} validated features")
                    tprint_info(f"📊 Quality score: {validation_result['quality_score']:.3f}")
                else:
                    tprint_warning(f"⚠️ Feature validation issues: {validation_result['issues']}")
                    tprint_info(f"📊 Quality score: {validation_result['quality_score']:.3f}")
                
                return features_df
            else:
                tprint_warning("⚠️ No features generated")
                return pd.DataFrame()
                
        except Exception as e:
            tprint_error(f"❌ Feature generation failed: {e}")
            return pd.DataFrame()
    
    def generate_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate interaction features with validation."""
        tprint_info("🔗 Generating interaction features...")
        
        if data.empty:
            return pd.DataFrame()
        
        # Get numeric columns for interactions
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            tprint_warning("⚠️ Not enough numeric columns for interactions")
            return pd.DataFrame()
        
        interaction_features = {}
        
        try:
            # Generate interactions based on configuration
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    if len(interaction_features) >= self.config.max_interactions:
                        break
                    
                    # Generate different types of interactions
                    for interaction_type in self.config.interaction_types:
                        if len(interaction_features) >= self.config.max_interactions:
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
                        except Exception as e:
                            tprint_debug(f"⚠️ Failed to generate {feature_name}: {e}")
                            continue
                
                if len(interaction_features) >= self.config.max_interactions:
                    break
            
            # Create DataFrame
            if interaction_features:
                interaction_df = pd.DataFrame(interaction_features, index=data.index)
                
                # Validate interactions
                validation_result = self.validator.validate_features(interaction_df)
                
                if validation_result['passed']:
                    tprint_success(f"✅ Generated {len(interaction_df.columns)} validated interactions")
                else:
                    tprint_warning(f"⚠️ Interaction validation issues: {validation_result['issues']}")
                
                return interaction_df
            else:
                tprint_warning("⚠️ No interaction features generated")
                return pd.DataFrame()
                
        except Exception as e:
            tprint_error(f"❌ Interaction generation failed: {e}")
            return pd.DataFrame()
    
    def generate_cross_timeframe_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate cross-timeframe features with proper alignment."""
        tprint_info("⏰ Generating cross-timeframe features...")
        
        if data.empty:
            return pd.DataFrame()
        
        cross_tf_features = {}
        
        try:
            # Get numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            for period in self.config.cross_timeframe_periods:
                for col in numeric_cols:
                    # Generate different aggregations
                    cross_tf_features[f'ctf_{period}m_{col}_mean'] = data[col].rolling(period).mean()
                    cross_tf_features[f'ctf_{period}m_{col}_std'] = data[col].rolling(period).std()
                    cross_tf_features[f'ctf_{period}m_{col}_max'] = data[col].rolling(period).max()
                    cross_tf_features[f'ctf_{period}m_{col}_min'] = data[col].rolling(period).min()
                    cross_tf_features[f'ctf_{period}m_{col}_median'] = data[col].rolling(period).median()
            
            # Create DataFrame
            if cross_tf_features:
                cross_tf_df = pd.DataFrame(cross_tf_features, index=data.index)
                
                # Validate cross-timeframe features
                validation_result = self.validator.validate_features(cross_tf_df)
                
                if validation_result['passed']:
                    tprint_success(f"✅ Generated {len(cross_tf_df.columns)} validated cross-timeframe features")
                else:
                    tprint_warning(f"⚠️ Cross-timeframe validation issues: {validation_result['issues']}")
                
                return cross_tf_df
            else:
                tprint_warning("⚠️ No cross-timeframe features generated")
                return pd.DataFrame()
                
        except Exception as e:
            tprint_error(f"❌ Cross-timeframe generation failed: {e}")
            return pd.DataFrame()
    
    def _validate_input_data(self, data: pd.DataFrame) -> bool:
        """Validate input data for feature generation."""
        if data.empty:
            return False
        
        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = set(required_cols) - set(data.columns)
        
        if missing_cols:
            tprint_warning(f"⚠️ Missing required columns: {missing_cols}")
            return False
        
        # Check for sufficient data
        if len(data) < 10:
            tprint_warning("⚠️ Insufficient data for feature generation")
            return False
        
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
        
        try:
            # RSI
            if len(close) > self.config.rsi_period:
                rsi = self._calculate_rsi(close, self.config.rsi_period)
                features['rsi'] = rsi
            
            # MACD
            if len(close) > max(self.config.macd_fast, self.config.macd_slow):
                macd_line, signal_line, histogram = self._calculate_macd(
                    close, self.config.macd_fast, self.config.macd_slow, self.config.macd_signal
                )
                features['macd'] = macd_line
                features['macd_signal'] = signal_line
                features['macd_histogram'] = histogram
            
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
            
            # Price momentum
            features['price_change'] = np.concatenate([[0], np.diff(close)])
            features['price_change_pct'] = np.concatenate([[0], np.diff(close) / close[:-1]])
            
        except Exception as e:
            tprint_debug(f"⚠️ Price indicator calculation failed: {e}")
        
        return features
    
    def _generate_volume_indicators(self, volume: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate volume-based technical indicators."""
        features = {}
        
        try:
            # Volume momentum
            features['volume_change'] = np.concatenate([[0], np.diff(volume)])
            features['volume_change_pct'] = np.concatenate([[0], np.diff(volume) / (volume[:-1] + 1e-8)])
            
            # Volume moving averages
            for window in [5, 10, 20]:
                if len(volume) > window:
                    features[f'volume_ma_{window}'] = self._rolling_mean(volume, window)
                    features[f'volume_ratio_{window}'] = volume / (features[f'volume_ma_{window}'] + 1e-8)
            
        except Exception as e:
            tprint_debug(f"⚠️ Volume indicator calculation failed: {e}")
        
        return features
    
    def _generate_ohlc_indicators(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate OHLC-based technical indicators."""
        features = {}
        
        try:
            # True Range
            tr = self._calculate_true_range(high, low, close)
            features['true_range'] = tr
            
            # Average True Range
            for window in [14, 21]:
                if len(tr) > window:
                    features[f'atr_{window}'] = self._rolling_mean(tr, window)
            
            # Price position within range
            features['price_position'] = (close - low) / (high - low + 1e-8)
            
            # Range characteristics
            features['range_size'] = high - low
            features['range_size_pct'] = (high - low) / (close + 1e-8)
            
        except Exception as e:
            tprint_debug(f"⚠️ OHLC indicator calculation failed: {e}")
        
        return features
    
    def _generate_rolling_statistics(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate rolling statistics for numeric columns."""
        features = {}
        
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                series = data[col].values
                
                for window in self.config.rolling_windows:
                    if len(series) > window:
                        # Basic statistics
                        features[f'{col}_ma_{window}'] = self._rolling_mean(series, window)
                        features[f'{col}_std_{window}'] = self._rolling_std(series, window)
                        features[f'{col}_min_{window}'] = self._rolling_min(series, window)
                        features[f'{col}_max_{window}'] = self._rolling_max(series, window)
                        
                        # Advanced statistics
                        features[f'{col}_skew_{window}'] = self._rolling_skew(series, window)
                        features[f'{col}_kurt_{window}'] = self._rolling_kurtosis(series, window)
                        
                        # Percentiles
                        features[f'{col}_p25_{window}'] = self._rolling_percentile(series, window, 25)
                        features[f'{col}_p75_{window}'] = self._rolling_percentile(series, window, 75)
                        
        except Exception as e:
            tprint_debug(f"⚠️ Rolling statistics calculation failed: {e}")
        
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
        
        result = np.full(len(data), np.nan)
        for i in range(window - 1, len(data)):
            result[i] = np.mean(data[i - window + 1:i + 1])
        
        return result
    
    def _rolling_std(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling standard deviation."""
        if len(data) < window:
            return np.full(len(data), np.nan)
        
        result = np.full(len(data), np.nan)
        for i in range(window - 1, len(data)):
            result[i] = np.std(data[i - window + 1:i + 1])
        
        return result
    
    def _rolling_min(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling minimum."""
        if len(data) < window:
            return np.full(len(data), np.nan)
        
        result = np.full(len(data), np.nan)
        for i in range(window - 1, len(data)):
            result[i] = np.min(data[i - window + 1:i + 1])
        
        return result
    
    def _rolling_max(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling maximum."""
        if len(data) < window:
            return np.full(len(data), np.nan)
        
        result = np.full(len(data), np.nan)
        for i in range(window - 1, len(data)):
            result[i] = np.max(data[i - window + 1:i + 1])
        
        return result
    
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