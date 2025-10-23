"""
Bar Efficiency Ratio Feature Engineering

This module implements the Bar Efficiency Ratio feature for measuring directional price action
vs. choppy conditions in 15-minute timeframe data.

Formula: efficiency_t = |close_t - open_t| / (high_t - low_t)
Rolling mean over 2-4 bars (30-60 minutes)
High efficiency (>0.6) = directional, Low efficiency (<0.3) = choppy
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass
import time

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
class BarEfficiencyConfig:
    """Configuration for Bar Efficiency Ratio feature."""

    # Feature settings
    window: int = 3  # Rolling window for efficiency (2-4 bars = 30-60 minutes)
    min_periods: int = 1  # Minimum periods for rolling calculation

    # Thresholds for interpretation
    high_efficiency_threshold: float = 0.6  # High efficiency = directional
    low_efficiency_threshold: float = 0.3   # Low efficiency = choppy

    # Output settings
    include_raw_efficiency: bool = True  # Include raw efficiency values
    include_rolling_efficiency: bool = True  # Include rolling mean efficiency
    include_efficiency_grade: bool = True  # Include normalized grade (0.0-1.0)

class BarEfficiencyRatioFeature:
    """
    Bar Efficiency Ratio Feature Engineering

    Measures the directional movement within a bar relative to its total range.
    Higher efficiency indicates more directional price action, lower efficiency indicates choppy conditions.
    """

    def __init__(self, config: Optional[BarEfficiencyConfig] = None):
        """Initialize Bar Efficiency Ratio feature."""
        self.config = config or BarEfficiencyConfig()
        tprint_info("📊 Bar Efficiency Ratio feature initialized")
        tprint_info(f"   → Window: {self.config.window} bars")
        tprint_info(f"   → High efficiency threshold: {self.config.high_efficiency_threshold}")
        tprint_info(f"   → Low efficiency threshold: {self.config.low_efficiency_threshold}")

    def calculate_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate Bar Efficiency Ratio features.

        Args:
            data: OHLCV data with columns ['open', 'high', 'low', 'close', 'volume']

        Returns:
            Dictionary of feature Series
        """
        tprint_info("📊 Calculating Bar Efficiency Ratio features")

        # Validate input data
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        features = {}

        try:
            # Calculate raw efficiency ratio
            price_range = data['high'] - data['low']
            price_range = price_range.replace(0, np.nan)  # Avoid division by zero

            raw_efficiency = np.abs(data['close'] - data['open']) / price_range
            raw_efficiency = raw_efficiency.fillna(0)  # Set to 0 for zero-range bars
            raw_efficiency = raw_efficiency.replace([np.inf, -np.inf], 0)  # Replace infinite values

            if self.config.include_raw_efficiency:
                features['bar_efficiency_raw'] = raw_efficiency
                tprint_info(f"   → Raw efficiency: mean={raw_efficiency.mean():.3f}, std={raw_efficiency.std():.3f}")

            # Calculate rolling mean efficiency
            if self.config.include_rolling_efficiency:
                rolling_efficiency = raw_efficiency.rolling(
                    window=self.config.window,
                    min_periods=1
                ).mean()
                features['bar_efficiency_rolling'] = rolling_efficiency
                tprint_info(f"   → Rolling efficiency: mean={rolling_efficiency.mean():.3f}, std={rolling_efficiency.std():.3f}")

            # Calculate efficiency grade (0.0-1.0)
            if self.config.include_efficiency_grade:
                # Normalize efficiency to 0-1 range, with 0.6+ efficiency = 1.0 grade
                efficiency_grade = np.clip(raw_efficiency / self.config.high_efficiency_threshold, 0.0, 1.0)
                features['bar_efficiency_grade'] = efficiency_grade
                tprint_info(f"   → Efficiency grade: mean={efficiency_grade.mean():.3f}, std={efficiency_grade.std():.3f}")

            # Calculate efficiency classification
            if self.config.include_rolling_efficiency:
                efficiency_class = pd.Series('choppy', index=data.index)
                efficiency_class[rolling_efficiency >= self.config.high_efficiency_threshold] = 'directional'
                efficiency_class[rolling_efficiency < self.config.low_efficiency_threshold] = 'choppy'
                features['bar_efficiency_class'] = efficiency_class

                # Count classifications
                class_counts = efficiency_class.value_counts()
                tprint_info(f"   → Efficiency classification: {dict(class_counts)}")

            tprint_info("✅ Bar Efficiency Ratio features calculated successfully")
            return features

        except Exception as e:
            tprint_error(f"❌ Error calculating Bar Efficiency Ratio features: {e}")
            raise

    def get_feature_names(self) -> list:
        """Get list of feature names this class produces."""
        features = []
        if self.config.include_raw_efficiency:
            features.append('bar_efficiency_raw')
        if self.config.include_rolling_efficiency:
            features.append('bar_efficiency_rolling')
            features.append('bar_efficiency_class')
        if self.config.include_efficiency_grade:
            features.append('bar_efficiency_grade')
        return features

    def get_feature_info(self) -> Dict[str, Dict[str, any]]:
        """Get detailed information about the features."""
        return {
            'bar_efficiency_raw': {
                'description': 'Raw bar efficiency ratio (|close-open| / (high-low))',
                'range': '[0, 1]',
                'interpretation': 'Higher values indicate more directional price action'
            },
            'bar_efficiency_rolling': {
                'description': f'Rolling mean efficiency over {self.config.window} bars',
                'range': '[0, 1]',
                'interpretation': 'Smoothed efficiency for trend analysis'
            },
            'bar_efficiency_grade': {
                'description': 'Normalized efficiency grade (0.0-1.0)',
                'range': '[0, 1]',
                'interpretation': '1.0 = high efficiency, 0.0 = low efficiency'
            },
            'bar_efficiency_class': {
                'description': 'Efficiency classification (directional/choppy)',
                'values': ['directional', 'choppy'],
                'interpretation': 'Categorical classification based on thresholds'
            }
        }

class BarEfficiencyRatioGenerator(VectorizedFeatureGenerator):
    """
    Framework-compatible Bar Efficiency Ratio feature generator.

    Implements the FeatureGenerator interface for integration with the feature bank
    and period lookback optimization system.
    """

    def __init__(self, lookback: int = 3, **kwargs):
        """
        Initialize the Bar Efficiency Ratio feature generator.

        Args:
            lookback: Number of periods for rolling calculation
            **kwargs: Additional configuration parameters
        """
        config = FeatureConfig(
            name="bar_efficiency_ratio",
            category=FeatureCategory.PRICE_ACTION,
            description="Bar efficiency ratio measuring directional price action vs choppy conditions",
            required_columns=['open', 'high', 'low', 'close'],
            optional_columns=['volume'],
            default_lookback=lookback,
            min_lookback=1,
            max_lookback=20,
            parameters={
                'window': lookback,
                'high_efficiency_threshold': kwargs.get('high_efficiency_threshold', 0.6),
                'low_efficiency_threshold': kwargs.get('low_efficiency_threshold', 0.3),
                'include_raw_efficiency': kwargs.get('include_raw_efficiency', True),
                'include_rolling_efficiency': kwargs.get('include_rolling_efficiency', True),
                'include_efficiency_grade': kwargs.get('include_efficiency_grade', True)
            },
            matrix_optimized=True,
            gpu_accelerated=False,
            enable_feature_selection=True
        )

        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

        # Initialize the feature engine
        feature_config = BarEfficiencyConfig(
            window=lookback,
            high_efficiency_threshold=kwargs.get('high_efficiency_threshold', 0.6),
            low_efficiency_threshold=kwargs.get('low_efficiency_threshold', 0.3),
            include_raw_efficiency=kwargs.get('include_raw_efficiency', True),
            include_rolling_efficiency=kwargs.get('include_rolling_efficiency', True),
            include_efficiency_grade=kwargs.get('include_efficiency_grade', True)
        )
        self.feature_engine = BarEfficiencyRatioFeature(feature_config)

    def generate(self, data: pd.DataFrame, lookback: Optional[int] = None) -> FeatureResult:
        """
        Generate Bar Efficiency Ratio features.

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
                self.feature_engine.config.window = effective_lookback

            # Generate features
            features = self.feature_engine.calculate_features(data)

            # Select the primary feature (rolling efficiency)
            if 'bar_efficiency_rolling' in features:
                primary_feature = features['bar_efficiency_rolling']
            elif 'bar_efficiency_raw' in features:
                primary_feature = features['bar_efficiency_raw']
            else:
                raise ValueError("No primary efficiency feature generated")

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
        Generate all Bar Efficiency Ratio features.

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
            self.feature_engine.config.window = effective_lookback

        # Generate all features
        return self.feature_engine.calculate_features(data)

# Convenience function for external usage
def calculate_bar_efficiency_features(
    data: pd.DataFrame,
    config: Optional[BarEfficiencyConfig] = None
) -> Dict[str, pd.Series]:
    """
    Calculate Bar Efficiency Ratio features.

    Args:
        data: OHLCV data with columns ['open', 'high', 'low', 'close', 'volume']
        config: Optional configuration

    Returns:
        Dictionary of feature Series
    """
    feature_engine = BarEfficiencyRatioFeature(config)
    return feature_engine.calculate_features(data)

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
