"""
import warnings
ATR Volatility Ratio Feature Engineering

This module implements the ATR Volatility Ratio feature for normalizing volatility
and identifying appropriate trading conditions in 15-minute timeframe data.

Formula: r_t = ATR_short / ATR_long
Short-term (1 hour) vs long-term (5 hours) ATR comparison
Skip when r_t > 1.5-2.0 (too jumpy) - no "too quiet" filter
"""

import numpy as np
import pandas as pd
import warnings
import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

# Import logger
from src.utils.logger import system_logger

# Set up logger for this module
logger = system_logger.getChild('ATRVolatilityRatio')

# Import existing utilities
from src.utils.tprint import tprint_info, tprint_warning, tprint_error
from src.utils.common_operations import safe_divide, safe_mean, safe_std
from src.utils.matrix_operations import vectorized_rolling_features

# Import framework components
from src.feature_generation.core.feature_generator import FeatureGenerator, FeatureCategory, FeatureConfig, FeatureResult, VectorizedFeatureGenerator

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

    cp = None

@dataclass
class ATRVolatilityRatioConfig:
    """Configuration for ATR Volatility Ratio feature."""

    # ATR calculation settings
    short_window: int = 4   # Short-term ATR window (1 hour)
    long_window: int = 20   # Long-term ATR window (5 hours)
    min_periods: int = 1    # Minimum periods for rolling calculation

    # Thresholds for interpretation
    high_ratio_threshold: float = 1.5  # Too jumpy - skip signals
    # Removed low_ratio_threshold - no "too quiet" filter

    # Output settings
    include_atr_short: bool = True  # Include short-term ATR
    include_atr_long: bool = True  # Include long-term ATR
    include_atr_ratio: bool = True  # Include ATR ratio
    include_atr_grade: bool = True  # Include normalized grade (0.0-1.0)
    include_atr_class: bool = True  # Include ATR classification

class ATRVolatilityRatioFeature:
    """
    ATR Volatility Ratio Feature Engineering

    Compares short-term vs long-term Average True Range to identify appropriate
    volatility conditions for trading. Higher ratios indicate more volatile conditions.
    """

    def __init__(self, config: Optional[ATRVolatilityRatioConfig] = None):
        """Initialize ATR Volatility Ratio feature."""
        self.config = config or ATRVolatilityRatioConfig()
        tprint_info("📊 ATR Volatility Ratio feature initialized")
        tprint_info(f"   → Short window: {self.config.short_window} bars")
        tprint_info(f"   → Long window: {self.config.long_window} bars")
        tprint_info(f"   → High ratio threshold: {self.config.high_ratio_threshold}")

    def calculate_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate ATR Volatility Ratio features.

        Args:
            data: OHLCV data with columns ['open', 'high', 'low', 'close', 'volume']

        Returns:
            Dictionary of feature Series
        """
        tprint_info("📊 Calculating ATR Volatility Ratio features")

        # Validate input data
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        features = {}

        try:
            # Calculate True Range
            tr1 = data['high'] - data['low']
            tr2 = np.abs(data['high'] - data['close'].shift(1))
            tr3 = np.abs(data['low'] - data['close'].shift(1))
            true_range = np.maximum(tr1, np.maximum(tr2, tr3))

            # Calculate short-term ATR
            if self.config.include_atr_short:
                true_range_series = pd.Series(true_range, index=data.index)
                atr_short = true_range_series.rolling(
                    window=self.config.short_window,
                    min_periods=1
                ).mean()
                features['atr_short'] = atr_short
                tprint_info(f"   → Short-term ATR: mean={atr_short.mean():.3f}, std={atr_short.std():.3f}")

            # Calculate long-term ATR
            if self.config.include_atr_long:
                true_range_series = pd.Series(true_range, index=data.index)
                atr_long = true_range_series.rolling(
                    window=self.config.long_window,
                    min_periods=1
                ).mean()
                features['atr_long'] = atr_long
                tprint_info(f"   → Long-term ATR: mean={atr_long.mean():.3f}, std={atr_long.std():.3f}")

            # Calculate ATR ratio
            if self.config.include_atr_ratio:
                atr_ratio = atr_short / atr_long
                atr_ratio = atr_ratio.fillna(1.0)  # Fill NaN values with 1.0
                atr_ratio = atr_ratio.replace([np.inf, -np.inf], 1.0)  # Replace infinite values
                features['atr_ratio'] = atr_ratio
                tprint_info(f"   → ATR ratio: mean={atr_ratio.mean():.3f}, std={atr_ratio.std():.3f}")

            # Calculate ATR grade (0.0-1.0)
            if self.config.include_atr_grade:
                # Grade decreases as ratio approaches the threshold (too jumpy)
                # No penalty for low ratios (no "too quiet" filter)
                atr_grade = np.clip(1.0 - (atr_ratio / self.config.high_ratio_threshold), 0.0, 1.0)
                features['atr_grade'] = atr_grade
                tprint_info(f"   → ATR grade: mean={atr_grade.mean():.3f}, std={atr_grade.std():.3f}")

            # Calculate ATR classification
            if self.config.include_atr_class and self.config.include_atr_ratio:
                atr_class = pd.Series('moderate', index=data.index)
                atr_class[atr_ratio > self.config.high_ratio_threshold] = 'too_jumpy'
                # No "too_quiet" classification - removed as per requirements
                features['atr_class'] = atr_class

                # Count classifications
                class_counts = atr_class.value_counts()
                tprint_info(f"   → ATR classification: {dict(class_counts)}")

            tprint_info("✅ ATR Volatility Ratio features calculated successfully")
            return features

        except Exception as e:
            tprint_error(f"❌ Error calculating ATR Volatility Ratio features: {e}")
            raise

    def get_feature_names(self) -> list:
        """Get list of feature names this class produces."""
        features = []
        if self.config.include_atr_short:
            features.append('atr_short')
        if self.config.include_atr_long:
            features.append('atr_long')
        if self.config.include_atr_ratio:
            features.append('atr_ratio')
        if self.config.include_atr_grade:
            features.append('atr_grade')
        if self.config.include_atr_class:
            features.append('atr_class')
        return features

    def get_feature_info(self) -> Dict[str, Dict[str, any]]:
        """Get detailed information about the features."""
        return {
            'atr_short': {
                'description': f'Short-term Average True Range over {self.config.short_window} bars',
                'range': '[0, inf)',
                'interpretation': 'Recent volatility measure'
            },
            'atr_long': {
                'description': f'Long-term Average True Range over {self.config.long_window} bars',
                'range': '[0, inf)',
                'interpretation': 'Baseline volatility measure'
            },
            'atr_ratio': {
                'description': 'Ratio of short-term to long-term ATR',
                'range': '[0, inf)',
                'interpretation': 'Higher values indicate increased volatility'
            },
            'atr_grade': {
                'description': 'Normalized ATR grade (0.0-1.0)',
                'range': '[0, 1]',
                'interpretation': '1.0 = moderate volatility, 0.0 = too jumpy'
            },
            'atr_class': {
                'description': 'ATR classification (moderate/too_jumpy)',
                'values': ['moderate', 'too_jumpy'],
                'interpretation': 'Categorical classification based on thresholds'
            }
        }

class ATRVolatilityRatioGenerator(VectorizedFeatureGenerator):
    """
    Framework-compatible ATR Volatility Ratio feature generator.

    Implements the FeatureGenerator interface for integration with the feature bank
    and period lookback optimization system.
    """

    def __init__(self, lookback: int = 4, **kwargs):
        """
        Initialize the ATR Volatility Ratio feature generator.

        Args:
            lookback: Number of periods for short-term ATR calculation
            **kwargs: Additional configuration parameters
        """
        config = FeatureConfig(
            name="atr_volatility_ratio",
            category=FeatureCategory.VOLATILITY,
            description="ATR volatility ratio for adaptive volatility filtering",
            required_columns=['open', 'high', 'low', 'close'],
            optional_columns=['volume'],
            default_lookback=lookback,
            min_lookback=1,
            max_lookback=50,
            parameters={
                'short_window': lookback,
                'long_window': kwargs.get('long_window', 20),
                'high_ratio_threshold': kwargs.get('high_ratio_threshold', 1.5),
                'include_atr_short': kwargs.get('include_atr_short', True),
                'include_atr_long': kwargs.get('include_atr_long', True),
                'include_atr_ratio': kwargs.get('include_atr_ratio', True),
                'include_atr_grade': kwargs.get('include_atr_grade', True),
                'include_atr_class': kwargs.get('include_atr_class', True)
            },
            matrix_optimized=True,
            gpu_accelerated=False,
            enable_feature_selection=True
        )

        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

        # Initialize the feature engine
        feature_config = ATRVolatilityRatioConfig(
            short_window=lookback,
            long_window=kwargs.get('long_window', 20),
            high_ratio_threshold=kwargs.get('high_ratio_threshold', 1.5),
            include_atr_short=kwargs.get('include_atr_short', True),
            include_atr_long=kwargs.get('include_atr_long', True),
            include_atr_ratio=kwargs.get('include_atr_ratio', True),
            include_atr_grade=kwargs.get('include_atr_grade', True),
            include_atr_class=kwargs.get('include_atr_class', True)
        )
        self.feature_engine = ATRVolatilityRatioFeature(feature_config)

    def generate(self, data: pd.DataFrame, lookback: Optional[int] = None) -> FeatureResult:
        """
        Generate ATR Volatility Ratio features.

        Args:
            data: OHLCV data with required columns
            lookback: Override default lookback period

        Returns:
            FeatureResult with generated features
        """
        start_time = time.time()

        try:
            # Use provided lookback or default
            effective_lookback = lookback or self.config.default_lookback

            # Update feature engine configuration if lookback changed
            if effective_lookback != self.config.default_lookback:
                self.feature_engine.config.short_window = effective_lookback

            # Generate features
            features = self.feature_engine.calculate_features(data)

            # Select the primary feature (ATR ratio)
            if 'atr_ratio' in features:
                primary_feature = features['atr_ratio']
            elif 'atr_short' in features:
                primary_feature = features['atr_short']
            else:
                raise ValueError("No primary ATR feature generated")

            computation_time = time.time() - start_time

            return FeatureResult(
                name=self.config.name,
                data=primary_feature,
                config=self.config,
                computation_time=computation_time,
                success=True,
                metadata={
                    'lookback_used': effective_lookback,
                    'all_features': list(features.keys()),
                    'feature_stats': {
                        'mean': float(primary_feature.mean()),
                        'std': float(primary_feature.std()),
                        'min': float(primary_feature.min()),
                        'max': float(primary_feature.max())
                    }
                }
            )

        except Exception as e:
            computation_time = time.time() - start_time
            return FeatureResult(
                name=self.config.name,
                data=pd.Series(dtype=float),
                config=self.config,
                computation_time=computation_time,
                success=False,
                error_message=str(e)
            )

    def get_all_features(self, data: pd.DataFrame, lookback: Optional[int] = None) -> Dict[str, pd.Series]:
        """
        Generate all ATR Volatility Ratio features.

        Args:
            data: OHLCV data with required columns
            lookback: Override default lookback period

        Returns:
            Dictionary of all generated features
        """
        # Use provided lookback or default
        effective_lookback = lookback or self.config.default_lookback

        # Update feature engine configuration if lookback changed
        if effective_lookback != self.config.default_lookback:
            self.feature_engine.config.short_window = effective_lookback

        # Generate all features
        return self.feature_engine.calculate_features(data)

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

# Convenience function for external usage
def calculate_atr_volatility_ratio_features(
    data: pd.DataFrame,
    config: Optional[ATRVolatilityRatioConfig] = None
) -> Dict[str, pd.Series]:
    """
    Calculate ATR Volatility Ratio features.

    Args:
        data: OHLCV data with columns ['open', 'high', 'low', 'close', 'volume']
        config: Optional configuration

    Returns:
        Dictionary of feature Series
    """
    feature_engine = ATRVolatilityRatioFeature(config)
    return feature_engine.calculate_features(data)
